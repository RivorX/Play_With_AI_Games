"""IL auto-tuning for batch size and learning rate."""

import gc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from .loss import CombinedLoss


AUTO_TUNE_ALGO_VERSION = 5


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fmt_gib(num_bytes):
    return f"{(float(num_bytes) / (1024.0 ** 3)):.2f} GiB"


def build_model_hash(config):
    """Build model hash used to cache auto-tuned IL hyperparameters."""
    model_cfg = dict(config.get("model", {}) or {})

    # Optional manual override from config.
    manual_hash = str(model_cfg.get("hash", "")).strip()
    if manual_hash:
        return manual_hash

    # Exclude runtime-only fields from hash.
    model_cfg.pop("print_summary", None)
    payload = json.dumps(model_cfg, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_dedicated_vram_bytes(device):
    """Return dedicated device VRAM in bytes (CUDA total memory, no shared system RAM)."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(device)
    return int(getattr(props, "total_memory", 0) or 0)


def get_cuda_vram_snapshot(device):
    """
    Return current CUDA VRAM snapshot (free, total) in bytes.

    Uses CUDA runtime mem_get_info when available; falls back to allocator stats.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0

    total_bytes = get_dedicated_vram_bytes(device)
    free_bytes = 0
    try:
        free_raw, total_raw = torch.cuda.mem_get_info(device)
        free_bytes = int(free_raw)
        total_from_api = int(total_raw)
        if total_from_api > 0:
            total_bytes = total_from_api
    except Exception:
        try:
            reserved = int(torch.cuda.memory_reserved(device))
        except Exception:
            reserved = 0
        free_bytes = max(0, int(total_bytes) - reserved)

    return int(free_bytes), int(total_bytes)


def _build_device_key(config, device):
    props = torch.cuda.get_device_properties(device)
    use_amp = bool(config.get("hardware", {}).get("use_amp", True))
    use_bfloat16 = bool(config.get("hardware", {}).get("use_bfloat16", False))
    return (
        f"{props.name}|cc{props.major}.{props.minor}|vram:{int(props.total_memory)}|"
        f"amp:{int(use_amp)}|bf16:{int(use_bfloat16)}"
    )


def _resolve_cache_path(base_dir, config):
    paths_cfg = config.get("paths", {}) or {}
    auto_cfg = config.get("imitation_learning", {}).get("auto_tune", {}) or {}
    rel_path = auto_cfg.get("cache_path")
    if rel_path is None or str(rel_path).strip() == "":
        rel_path = paths_cfg.get("il_auto_tune_cache", "logs/il_auto_tune_cache.yaml")
    rel_path = str(rel_path)
    cache_path = Path(rel_path)
    if cache_path.is_absolute():
        return cache_path
    return Path(base_dir) / cache_path


def _load_cache(cache_path):
    default = {"version": 1, "entries": {}}
    if not cache_path.exists():
        return default

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return default

    if not isinstance(data, dict):
        return default
    data.setdefault("version", 1)
    entries = data.get("entries")
    if not isinstance(entries, dict):
        data["entries"] = {}
    return data


def _save_cache(cache_path, data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def _is_oom_error(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _compute_vram_budget(total_vram_bytes, free_vram_bytes, config):
    il_cfg = config.get("imitation_learning", {}) or {}
    auto_cfg = il_cfg.get("auto_tune", {}) or {}

    free_util = _safe_float(auto_cfg.get("free_vram_utilization"), 0.92)
    free_util = max(0.20, min(0.99, free_util))

    safety_margin_gib = _safe_float(auto_cfg.get("safety_margin_gib"), 1.00)
    safety_margin_gib = max(0.0, min(8.0, safety_margin_gib))
    safety_margin_bytes = int(safety_margin_gib * (1024.0 ** 3))

    free_limit = int(float(max(0, free_vram_bytes)) * free_util)
    budget_bytes = free_limit

    budget_bytes = max(1, budget_bytes - safety_margin_bytes)

    min_budget_gib = _safe_float(auto_cfg.get("min_budget_gib"), 0.50)
    min_budget_gib = max(0.10, min(8.0, min_budget_gib))
    min_budget_bytes = int(min_budget_gib * (1024.0 ** 3))
    budget_bytes = max(budget_bytes, min_budget_bytes)

    return {
        "budget_bytes": int(budget_bytes),
        "free_limit_bytes": int(free_limit),
        "free_utilization": float(free_util),
        "safety_margin_bytes": int(safety_margin_bytes),
        "min_budget_bytes": int(min_budget_bytes),
        "budget_mode": "free_dedicated_vram",
    }


def _clear_probe_state(model, device):
    for param in model.parameters():
        param.grad = None
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_adamw_optimizer(model, lr, weight_decay, fused_preferred):
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=fused_preferred,
        )
    except TypeError:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    except RuntimeError:
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=False,
        )


def _build_synthetic_batch(config, device, batch_size, input_planes, use_mtl, generator=None):
    boards = torch.randn(batch_size, input_planes, 8, 8, device=device, generator=generator)
    boards = boards.to(memory_format=torch.channels_last)

    move_space = int(config.get("imitation_learning", {}).get("probe_action_size", 4272))
    max_moves = int(config.get("data", {}).get("max_moves_per_game", 200))
    min_total_moves = int(
        config.get("imitation_learning", {}).get("value_move_weight_min_total_moves", 40)
    )
    min_total_moves = max(1, min_total_moves)
    max_moves = max(min_total_moves, max_moves)

    moves = torch.randint(
        0,
        max(2, move_space),
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    outcomes = torch.randint(
        -1,
        2,
        (batch_size, 1),
        device=device,
        dtype=torch.int32,
        generator=generator,
    ).float()
    move_indices = torch.randint(
        0,
        max_moves + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    total_moves = torch.randint(
        min_total_moves,
        max_moves + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )

    targets = {
        "moves": moves,
        "values": outcomes,
        "move_indices": move_indices,
        "total_moves": total_moves,
    }

    if use_mtl:
        win_targets = torch.randint(
            0,
            2,
            (batch_size, 1),
            device=device,
            generator=generator,
        ).float()
        material_targets = (
            torch.rand(batch_size, 1, device=device, generator=generator) * 2.0 - 1.0
        )
        check_targets = torch.randint(
            0,
            2,
            (batch_size, 1),
            device=device,
            generator=generator,
        ).float()
        targets.update(
            {
                "win": win_targets,
                "material": material_targets,
                "check": check_targets,
            }
        )

    return boards, targets


def _probe_peak_bytes(model, config, device, batch_size, use_amp, use_bfloat16):
    """
    Run one realistic synthetic train step and return peak VRAM stats.

    Includes:
    - model forward/backward
    - real IL CombinedLoss computation
    - one AdamW step (allocates optimizer states)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    use_mtl = bool(config.get("model", {}).get("use_multitask_learning", False))
    fused_adamw = bool(device.type == "cuda" and torch.cuda.is_available())
    criterion = CombinedLoss(config)
    probe_optimizer = None

    boards = None
    policy_pred = None
    value_pred = None
    moves = None
    outcomes = None
    move_indices = None
    total_moves = None
    win_pred = material_pred = check_pred = None
    win_targets = material_targets = check_targets = None
    predictions = None
    targets = None
    loss = None
    was_training = model.training

    try:
        model.train()
        _clear_probe_state(model, device)
        probe_optimizer = _build_adamw_optimizer(
            model=model,
            lr=0.0,
            weight_decay=0.0,
            fused_preferred=fused_adamw,
        )
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        input_planes = int(getattr(model, "input_planes", 16))
        boards, synthetic_targets = _build_synthetic_batch(
            config=config,
            device=device,
            batch_size=batch_size,
            input_planes=input_planes,
            use_mtl=use_mtl,
        )
        moves = synthetic_targets["moves"]
        outcomes = synthetic_targets["values"]
        move_indices = synthetic_targets["move_indices"]
        total_moves = synthetic_targets["total_moves"]

        with torch.enable_grad():
            probe_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if use_mtl:
                    policy_pred, value_pred, win_pred, material_pred, check_pred = model(
                        boards, return_aux=True
                    )
                else:
                    policy_pred, value_pred = model(boards, return_aux=False)

                predictions = {
                    "policy": policy_pred,
                    "value": value_pred,
                }
                targets = {
                    "moves": moves,
                    "values": outcomes,
                    "move_indices": move_indices,
                    "total_moves": total_moves,
                }

                if use_mtl:
                    predictions.update(
                        {
                            "win": win_pred,
                            "material": material_pred,
                            "check": check_pred,
                        }
                    )
                    targets.update(
                        {
                            "win": synthetic_targets["win"],
                            "material": synthetic_targets["material"],
                            "check": synthetic_targets["check"],
                        }
                    )

                loss, _ = criterion(predictions, targets)
            loss.backward()
            probe_optimizer.step()

        torch.cuda.synchronize(device)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        return {
            "ok": True,
            "oom": False,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "error": None,
        }
    except RuntimeError as exc:
        if _is_oom_error(exc):
            return {
                "ok": False,
                "oom": True,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "error": str(exc),
            }
        raise
    finally:
        del (
            boards,
            policy_pred,
            value_pred,
            moves,
            outcomes,
            move_indices,
            total_moves,
            win_pred,
            material_pred,
            check_pred,
            win_targets,
            material_targets,
            check_targets,
            predictions,
            targets,
            loss,
            criterion,
            probe_optimizer,
        )
        _clear_probe_state(model, device)
        if not was_training:
            model.eval()


def _round_batch(batch_size, round_to):
    if round_to <= 1:
        return max(1, int(batch_size))
    return max(1, (int(batch_size) // int(round_to)) * int(round_to))


def _find_max_batch_size(model, config, device, budget_info):
    il_cfg = config.get("imitation_learning", {}) or {}
    auto_cfg = il_cfg.get("auto_tune", {}) or {}

    use_amp = bool(config.get("hardware", {}).get("use_amp", True))
    use_bfloat16 = bool(config.get("hardware", {}).get("use_bfloat16", False))

    configured_batch = max(1, _safe_int(il_cfg.get("batch_size"), 1024))
    min_batch = max(1, _safe_int(auto_cfg.get("min_batch_size"), 64))
    max_batch_default = max(32768, configured_batch * 4)
    max_batch = max(min_batch, _safe_int(auto_cfg.get("max_batch_size"), max_batch_default))
    batch_round_to = max(1, _safe_int(auto_cfg.get("batch_round_to"), 64))

    budget_bytes = int(budget_info.get("budget_bytes") or 0)
    probe_peak_metric = str(auto_cfg.get("probe_peak_metric", "reserved")).strip().lower()
    if probe_peak_metric not in {"allocated", "reserved"}:
        probe_peak_metric = "reserved"
    probe_to_train_multiplier = _safe_float(auto_cfg.get("probe_to_train_multiplier"), 1.08)
    probe_to_train_multiplier = max(1.0, min(2.0, probe_to_train_multiplier))

    probe_fit_cache = {}
    probe_effective_peak_cache = {}

    def fits(batch_size):
        batch_size = max(1, int(batch_size))
        cached = probe_fit_cache.get(batch_size)
        if cached is not None:
            return cached

        probe = _probe_peak_bytes(
            model=model,
            config=config,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
            use_bfloat16=use_bfloat16,
        )
        if not probe["ok"]:
            probe_fit_cache[batch_size] = False
            probe_effective_peak_cache[batch_size] = None
            return False

        if probe_peak_metric == "allocated":
            base_peak = int(probe["peak_allocated_bytes"] or 0)
        else:
            base_peak = int(probe["peak_reserved_bytes"] or 0)

        effective_peak = int(float(base_peak) * probe_to_train_multiplier)
        fits_budget = effective_peak <= budget_bytes

        probe_fit_cache[batch_size] = fits_budget
        probe_effective_peak_cache[batch_size] = effective_peak
        return fits_budget

    if not fits(1):
        return 1, {
            "budget_bytes": budget_bytes,
            "probe_count": len(probe_fit_cache),
            "estimated_peak_bytes": None,
            "free_utilization": budget_info.get("free_utilization"),
            "probe_peak_metric": probe_peak_metric,
            "probe_to_train_multiplier": probe_to_train_multiplier,
            "min_batch": min_batch,
            "max_batch": max_batch,
            "batch_round_to": batch_round_to,
            "raw_best_batch": 1,
            "free_limit_bytes": budget_info.get("free_limit_bytes"),
            "safety_margin_bytes": budget_info.get("safety_margin_bytes"),
            "budget_mode": budget_info.get("budget_mode"),
        }

    low = 1
    high = max_batch

    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid):
            low = mid
        else:
            high = mid - 1

    raw_best_batch = low

    rounded_batch = _round_batch(raw_best_batch, batch_round_to)
    if raw_best_batch >= min_batch:
        rounded_min = _round_batch(min_batch, batch_round_to)
        if rounded_min > raw_best_batch:
            rounded_min = raw_best_batch
        rounded_batch = max(rounded_batch, rounded_min)
    rounded_batch = max(1, min(rounded_batch, raw_best_batch))

    if not fits(rounded_batch):
        rounded_batch = raw_best_batch

    final_peak = probe_effective_peak_cache.get(rounded_batch)
    if final_peak is None:
        fits(rounded_batch)
        final_peak = probe_effective_peak_cache.get(rounded_batch)

    return int(rounded_batch), {
        "budget_bytes": budget_bytes,
        "probe_count": len(probe_fit_cache),
        "estimated_peak_bytes": final_peak,
        "free_utilization": budget_info.get("free_utilization"),
        "probe_peak_metric": probe_peak_metric,
        "probe_to_train_multiplier": probe_to_train_multiplier,
        "min_batch": min_batch,
        "max_batch": max_batch,
        "batch_round_to": batch_round_to,
        "raw_best_batch": raw_best_batch,
        "free_limit_bytes": budget_info.get("free_limit_bytes"),
        "safety_margin_bytes": budget_info.get("safety_margin_bytes"),
        "budget_mode": budget_info.get("budget_mode"),
    }


def _clamp_learning_rate(value, auto_cfg):
    lr_min = _safe_float(auto_cfg.get("lr_min"), 5e-5)
    lr_max = _safe_float(auto_cfg.get("lr_max"), 2e-3)
    if lr_min > lr_max:
        lr_min, lr_max = lr_max, lr_min
    return max(lr_min, min(lr_max, float(value))), float(lr_min), float(lr_max)


def _count_trainable_params(model):
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _build_lr_test_candidates(config):
    auto_cfg = config.get("imitation_learning", {}).get("auto_tune", {}) or {}

    manual = auto_cfg.get("lr_test_candidates")
    if isinstance(manual, (list, tuple)):
        values = []
        for raw in manual:
            value = _safe_float(raw, default=-1.0)
            if value > 0:
                values.append(float(value))
        if values:
            return sorted(set(values))

    lr_min = _safe_float(auto_cfg.get("lr_test_min"), 8e-5)
    lr_max = _safe_float(auto_cfg.get("lr_test_max"), 1.2e-3)
    if lr_min <= 0:
        lr_min = 8e-5
    if lr_max <= 0:
        lr_max = 1.2e-3
    if lr_min > lr_max:
        lr_min, lr_max = lr_max, lr_min

    points = _safe_int(auto_cfg.get("lr_test_points"), 6)
    points = max(3, min(12, points))

    if abs(lr_max - lr_min) < 1e-12:
        return [float(lr_min)]

    values = torch.logspace(
        math.log10(lr_min),
        math.log10(lr_max),
        steps=points,
        dtype=torch.float64,
    ).tolist()
    return [float(v) for v in values]


def _snapshot_model_state_cpu(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def _select_learning_rate(model, config, configured_lr, tuned_batch, device, use_amp, use_bfloat16):
    """Find LR by short range test (multiple short train steps on synthetic batches)."""
    auto_cfg = config.get("imitation_learning", {}).get("auto_tune", {}) or {}
    lr_candidates = _build_lr_test_candidates(config)
    test_steps = _safe_int(auto_cfg.get("lr_test_steps"), 2)
    test_steps = max(1, min(5, test_steps))
    test_seed = _safe_int(auto_cfg.get("lr_test_seed"), 12345)

    il_cfg = config.get("imitation_learning", {}) or {}
    weight_decay = _safe_float(il_cfg.get("weight_decay"), 0.0)
    fused_adamw = bool(device.type == "cuda" and torch.cuda.is_available())
    use_mtl = bool(config.get("model", {}).get("use_multitask_learning", False))
    input_planes = int(getattr(model, "input_planes", 16))
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    baseline_state = _snapshot_model_state_cpu(model)
    criterion = CombinedLoss(config)
    was_training = model.training

    best_entry = None
    lr_test_results = []

    try:
        model.train()
        for lr in lr_candidates:
            optimizer = None
            stable = True
            first_loss = None
            last_loss = None
            steps_done = 0
            mean_loss = 0.0
            predictions = None
            targets = None
            synthetic_targets = None
            loss = None

            try:
                model.load_state_dict(baseline_state, strict=True)
                _clear_probe_state(model, device)
                optimizer = _build_adamw_optimizer(
                    model=model,
                    lr=float(lr),
                    weight_decay=weight_decay,
                    fused_preferred=fused_adamw,
                )
                if device.type == "cuda":
                    device_idx = device.index
                    if device_idx is None:
                        device_idx = torch.cuda.current_device()
                    generator = torch.Generator(device=f"cuda:{device_idx}")
                else:
                    generator = torch.Generator()
                generator.manual_seed(int(test_seed))

                for step_idx in range(test_steps):
                    boards, synthetic_targets = _build_synthetic_batch(
                        config=config,
                        device=device,
                        batch_size=tuned_batch,
                        input_planes=input_planes,
                        use_mtl=use_mtl,
                        generator=generator,
                    )

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if use_mtl:
                            policy_pred, value_pred, win_pred, material_pred, check_pred = model(
                                boards, return_aux=True
                            )
                        else:
                            policy_pred, value_pred = model(boards, return_aux=False)

                        predictions = {
                            "policy": policy_pred,
                            "value": value_pred,
                        }
                        targets = {
                            "moves": synthetic_targets["moves"],
                            "values": synthetic_targets["values"],
                            "move_indices": synthetic_targets["move_indices"],
                            "total_moves": synthetic_targets["total_moves"],
                        }
                        if use_mtl:
                            predictions.update(
                                {
                                    "win": win_pred,
                                    "material": material_pred,
                                    "check": check_pred,
                                }
                            )
                            targets.update(
                                {
                                    "win": synthetic_targets["win"],
                                    "material": synthetic_targets["material"],
                                    "check": synthetic_targets["check"],
                                }
                            )

                        loss, _ = criterion(predictions, targets)

                    loss_value = float(loss.detach().item())
                    if not math.isfinite(loss_value):
                        stable = False
                        break

                    if first_loss is None:
                        first_loss = loss_value
                    last_loss = loss_value
                    mean_loss += loss_value

                    loss.backward()
                    optimizer.step()
                    steps_done += 1

                    if first_loss is not None and loss_value > (first_loss * 1.8):
                        stable = False
                        break
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    stable = False
                else:
                    raise
            finally:
                del optimizer, predictions, targets, synthetic_targets, loss
                _clear_probe_state(model, device)

            if steps_done > 0 and first_loss is not None and last_loss is not None:
                mean_loss /= float(steps_done)
                improvement = (first_loss - last_loss) / max(abs(first_loss), 1e-6)
            else:
                mean_loss = float("inf")
                improvement = float("-inf")
                stable = False

            score = float(improvement)
            if not stable:
                score -= 1.0

            entry = {
                "lr": float(lr),
                "stable": bool(stable),
                "steps_done": int(steps_done),
                "first_loss": None if first_loss is None else float(first_loss),
                "last_loss": None if last_loss is None else float(last_loss),
                "mean_loss": None if not math.isfinite(mean_loss) else float(mean_loss),
                "improvement": float(improvement),
                "score": float(score),
            }
            lr_test_results.append(entry)

            if stable and (best_entry is None or entry["score"] > best_entry["score"]):
                best_entry = entry
    finally:
        model.load_state_dict(baseline_state, strict=True)
        _clear_probe_state(model, device)
        del baseline_state, criterion
        if not was_training:
            model.eval()

    auto_cfg = config.get("imitation_learning", {}).get("auto_tune", {}) or {}
    if best_entry is not None:
        selected_lr_raw = float(best_entry["lr"])
    else:
        selected_lr_raw = float(configured_lr)
    selected_lr, lr_min, lr_max = _clamp_learning_rate(selected_lr_raw, auto_cfg)

    lr_factor = float(selected_lr / max(1e-12, float(configured_lr)))
    lr_meta = {
        "method": "lr_range_test",
        "selected_lr_raw": float(selected_lr_raw),
        "lr_min": float(lr_min),
        "lr_max": float(lr_max),
        "test_steps": int(test_steps),
        "test_seed": int(test_seed),
        "candidates": [float(v) for v in lr_candidates],
        "results": lr_test_results,
        "best_entry": best_entry,
        "model_params_millions": float(_count_trainable_params(model) / 1e6),
    }
    return float(selected_lr), lr_factor, "lr_range_test", lr_meta


def resolve_il_hyperparameters(config, model, device, base_dir, mode="config"):
    """
    Resolve IL hyperparameters from config or from auto-tune cache/probe.

    Returns dict with selected values and source metadata.
    """
    il_cfg = config.setdefault("imitation_learning", {})
    configured_batch = max(1, _safe_int(il_cfg.get("batch_size"), 1024))
    configured_lr = _safe_float(il_cfg.get("learning_rate"), 5e-4)
    if configured_lr <= 0:
        configured_lr = 5e-4

    selected_mode = str(mode or "config").strip().lower()
    if selected_mode not in {"auto", "config"}:
        selected_mode = "config"

    model_hash = build_model_hash(config)

    result = {
        "algo_version": AUTO_TUNE_ALGO_VERSION,
        "mode": selected_mode,
        "source": "config",
        "batch_size": configured_batch,
        "learning_rate": configured_lr,
        "model_hash": model_hash,
        "cache_path": str(_resolve_cache_path(base_dir, config)),
        "cache_hit": False,
        "dedicated_vram_bytes": 0,
        "budget_bytes": None,
        "estimated_peak_bytes": None,
        "probe_count": 0,
        "lr_scale_factor": 1.0,
        "lr_scale_mode": None,
    }

    if selected_mode != "auto":
        il_cfg["batch_size"] = configured_batch
        il_cfg["learning_rate"] = configured_lr
        print(
            "IL hyperparameters: using config values "
            f"(batch_size={configured_batch}, learning_rate={configured_lr:.6g})."
        )
        return result

    if device.type != "cuda" or not torch.cuda.is_available():
        il_cfg["batch_size"] = configured_batch
        il_cfg["learning_rate"] = configured_lr
        result["mode"] = "config"
        result["source"] = "config (auto unavailable: non-CUDA)"
        print("IL auto-tune unavailable on non-CUDA device; using config values.")
        return result

    free_vram_bytes, dedicated_vram_bytes = get_cuda_vram_snapshot(device)
    result["dedicated_vram_bytes"] = dedicated_vram_bytes
    result["free_vram_bytes"] = free_vram_bytes
    if dedicated_vram_bytes <= 0:
        il_cfg["batch_size"] = configured_batch
        il_cfg["learning_rate"] = configured_lr
        result["mode"] = "config"
        result["source"] = "config (auto unavailable: VRAM not detected)"
        print("IL auto-tune could not detect dedicated VRAM; using config values.")
        return result

    budget_info = _compute_vram_budget(
        total_vram_bytes=dedicated_vram_bytes,
        free_vram_bytes=free_vram_bytes,
        config=config,
    )
    current_budget_bytes = int(budget_info.get("budget_bytes", 0) or 0)
    result["budget_bytes"] = current_budget_bytes

    cache_path = _resolve_cache_path(base_dir, config)
    cache = _load_cache(cache_path)
    entries = cache.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        cache["entries"] = entries

    device_key = _build_device_key(config, device)
    hash_entries = entries.get(model_hash)
    if isinstance(hash_entries, dict):
        cached_entry = hash_entries.get(device_key)
        if isinstance(cached_entry, dict):
            cached_batch = _safe_int(cached_entry.get("batch_size"), 0)
            cached_lr = _safe_float(cached_entry.get("learning_rate"), 0.0)
            cached_peak = _safe_int(cached_entry.get("estimated_peak_bytes"), 0)
            cached_algo = _safe_int(cached_entry.get("algo_version"), 0)
            cache_usable = True
            if cached_algo != AUTO_TUNE_ALGO_VERSION:
                cache_usable = False
                print(
                    "IL auto-tune: cached params skipped "
                    f"(algo_version={cached_algo}, expected={AUTO_TUNE_ALGO_VERSION})."
                )
            if cached_peak > 0 and current_budget_bytes > 0 and cached_peak > current_budget_bytes:
                cache_usable = False
                print(
                    "IL auto-tune: cached params skipped "
                    f"(cached_peak={_fmt_gib(cached_peak)} > current_budget={_fmt_gib(current_budget_bytes)})."
                )

            if cache_usable and cached_batch > 0 and cached_lr > 0:
                cached_peak = _safe_int(cached_entry.get("estimated_peak_bytes"), 0) or None
                cached_probe_count = _safe_int(cached_entry.get("probe_count"), 0)
                il_cfg["batch_size"] = int(cached_batch)
                il_cfg["learning_rate"] = float(cached_lr)
                result.update(
                    {
                        "source": "auto-cache",
                        "batch_size": int(cached_batch),
                        "learning_rate": float(cached_lr),
                        "cache_hit": True,
                        "budget_bytes": current_budget_bytes or None,
                        "estimated_peak_bytes": cached_peak,
                        "probe_count": cached_probe_count,
                        "lr_scale_factor": _safe_float(
                            cached_entry.get("lr_factor", cached_entry.get("lr_scale_factor")),
                            1.0,
                        ),
                        "lr_scale_mode": cached_entry.get("lr_mode", cached_entry.get("lr_scale_mode")),
                    }
                )
                print(
                    "IL auto-tune: loaded cached values "
                    f"(hash={model_hash[:12]}, batch_size={int(cached_batch)}, lr={float(cached_lr):.6g})."
                )
                return result

    print(f"  Dedicated VRAM total: {_fmt_gib(dedicated_vram_bytes)}")
    print(f"  Dedicated VRAM free:  {_fmt_gib(free_vram_bytes)}")
    print(
        f"  Budget: {_fmt_gib(current_budget_bytes)} "
        f"(free_limit={_fmt_gib(budget_info['free_limit_bytes'])}, "
        f"safety_margin={_fmt_gib(budget_info['safety_margin_bytes'])})"
    )
    print("IL auto-tune: estimating batch size from dedicated GPU VRAM...")
    tuned_batch, tune_info = _find_max_batch_size(
        model=model,
        config=config,
        device=device,
        budget_info=budget_info,
    )

    tuned_lr, lr_factor, lr_mode, lr_meta = _select_learning_rate(
        model=model,
        config=config,
        configured_lr=configured_lr,
        tuned_batch=tuned_batch,
        device=device,
        use_amp=bool(config.get("hardware", {}).get("use_amp", True)),
        use_bfloat16=bool(config.get("hardware", {}).get("use_bfloat16", False)),
    )

    il_cfg["batch_size"] = int(tuned_batch)
    il_cfg["learning_rate"] = float(tuned_lr)

    estimated_peak = tune_info.get("estimated_peak_bytes")
    budget_bytes = tune_info.get("budget_bytes")

    print(
        "IL auto-tune: selected "
        f"batch_size={int(tuned_batch)}, learning_rate={float(tuned_lr):.6g} "
        f"({lr_mode}, factor x{lr_factor:.3f})."
    )
    if budget_bytes:
        print(f"  Target VRAM budget: {_fmt_gib(budget_bytes)}")
    if estimated_peak:
        print(f"  Estimated training peak: {_fmt_gib(estimated_peak)}")
    best_entry = lr_meta.get("best_entry") if isinstance(lr_meta, dict) else None
    if best_entry:
        print(
            "  LR test: "
            f"candidates={len(lr_meta.get('candidates', []))}, "
            f"best_improvement={best_entry.get('improvement', 0.0):+.3f}"
        )

    if not isinstance(hash_entries, dict):
        hash_entries = {}
    hash_entries[device_key] = {
        "batch_size": int(tuned_batch),
        "learning_rate": float(tuned_lr),
        "model_hash": model_hash,
        "algo_version": AUTO_TUNE_ALGO_VERSION,
        "device_key": device_key,
        "device_name": torch.cuda.get_device_name(device),
        "dedicated_vram_bytes": int(dedicated_vram_bytes),
        "free_vram_bytes_at_tune": int(free_vram_bytes),
        "budget_bytes": int(budget_bytes) if budget_bytes is not None else None,
        "estimated_peak_bytes": int(estimated_peak) if estimated_peak is not None else None,
        "probe_count": int(tune_info.get("probe_count", 0)),
        "free_vram_utilization": float(tune_info.get("free_utilization", 0.0)),
        "budget_mode": tune_info.get("budget_mode", "free_dedicated_vram"),
        "free_limit_bytes": int(tune_info.get("free_limit_bytes", 0) or 0),
        "safety_margin_bytes": int(tune_info.get("safety_margin_bytes", 0) or 0),
        "probe_peak_metric": tune_info.get("probe_peak_metric"),
        "probe_to_train_multiplier": float(tune_info.get("probe_to_train_multiplier", 1.0)),
        "lr_mode": lr_mode,
        "lr_factor": float(lr_factor),
        "lr_scale_mode": lr_mode,
        "lr_scale_factor": float(lr_factor),
        "lr_meta": lr_meta,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    entries[model_hash] = hash_entries
    cache["entries"] = entries

    try:
        _save_cache(cache_path, cache)
        print(f"IL auto-tune cache saved: {cache_path}")
    except Exception as exc:
        print(f"Warning: failed to save IL auto-tune cache ({exc})")

    result.update(
        {
            "source": "auto-new",
            "batch_size": int(tuned_batch),
            "learning_rate": float(tuned_lr),
            "cache_hit": False,
            "budget_bytes": budget_bytes,
            "estimated_peak_bytes": estimated_peak,
            "probe_count": int(tune_info.get("probe_count", 0)),
            "lr_scale_factor": float(lr_factor),
            "lr_scale_mode": lr_mode,
            "lr_meta": lr_meta,
            "cache_path": str(cache_path),
        }
    )
    return result
