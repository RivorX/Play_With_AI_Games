"""Shared auto-tuning helpers for IL and RL hyperparameters."""

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


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _fmt_gib(num_bytes):
    return f"{(float(num_bytes) / (1024.0 ** 3)):.2f} GiB"


def build_model_hash(config, include_version=False):
    """Build model hash used to cache auto-tuned IL hyperparameters."""
    model_cfg = dict(config.get("model", {}) or {})

    # Optional manual override from config.
    manual_hash = str(model_cfg.get("hash", "")).strip()
    if manual_hash:
        return manual_hash

    # Exclude runtime-only fields from hash.
    model_cfg.pop("print_summary", None)
    if not include_version:
        # Keep cache stable across cosmetic version label changes.
        model_cfg.pop("version", None)
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


def _normalize_runtime_profile(runtime_profile):
    text = str(runtime_profile or "eager").strip().lower()
    if text in {"compiled", "compile", "torch.compile", "torch_compile"}:
        return "compiled"
    return "eager"


def _get_auto_tune_cfg(config):
    """Return shared auto-tune config (top-level) with legacy IL fallback."""
    shared_cfg = config.get("auto_tune", {}) or {}
    legacy_cfg = config.get("imitation_learning", {}).get("auto_tune", {}) or {}

    if not isinstance(shared_cfg, dict):
        shared_cfg = {}
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}

    merged = dict(legacy_cfg)
    merged.update(shared_cfg)
    return merged


def _auto_tune_enabled_for(config, target):
    auto_cfg = _get_auto_tune_cfg(config)
    target_key = "use_for_rl" if str(target).strip().lower() == "rl" else "use_for_il"
    return _safe_bool(auto_cfg.get(target_key), True)


def _resolve_cache_path(base_dir, config):
    paths_cfg = config.get("paths", {}) or {}
    auto_cfg = _get_auto_tune_cfg(config)
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


def _ensure_cache_entries(cache):
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        entries = {}
        cache["entries"] = entries
    return entries


def _get_cached_entry_for_device(entries, model_hash, legacy_model_hash, device_key, context_label):
    hash_entries = entries.get(model_hash)
    hash_key_used = model_hash
    if not isinstance(hash_entries, dict) and legacy_model_hash:
        legacy_entries = entries.get(legacy_model_hash)
        if isinstance(legacy_entries, dict):
            hash_entries = legacy_entries
            hash_key_used = legacy_model_hash
            print(
                f"{context_label}: found legacy cache key "
                "(included model.version); migrating to version-agnostic hash."
            )

    cached_entry = None
    if isinstance(hash_entries, dict):
        maybe_entry = hash_entries.get(device_key)
        if isinstance(maybe_entry, dict):
            cached_entry = maybe_entry
    return cached_entry, hash_entries, hash_key_used


def _read_cache_profile(entry, runtime_profile, fallback_to_eager=True):
    if not isinstance(entry, dict):
        return None, None

    requested = _normalize_runtime_profile(runtime_profile)

    def _as_profile_payload(raw, fallback_algo):
        if not isinstance(raw, dict):
            return None
        batch_size = _safe_int(raw.get("batch_size"), 0)
        learning_rate = _safe_float(raw.get("learning_rate"), 0.0)
        if batch_size <= 0 or learning_rate <= 0:
            return None

        payload = {
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "estimated_peak_bytes": _safe_int(raw.get("estimated_peak_bytes"), 0) or None,
            "probe_count": _safe_int(raw.get("probe_count"), 0),
            "lr_factor": _safe_float(raw.get("lr_factor", raw.get("lr_scale_factor")), 1.0),
            "lr_mode": raw.get("lr_mode", raw.get("lr_scale_mode")),
            "algo_version": _safe_int(raw.get("algo_version"), fallback_algo),
        }
        return payload

    fallback_algo = _safe_int(entry.get("algo_version"), 0)
    profiles = entry.get("profiles")
    if isinstance(profiles, dict):
        requested_profile = _as_profile_payload(profiles.get(requested), fallback_algo)
        if requested_profile is not None:
            return requested_profile, requested
        if fallback_to_eager and requested != "eager":
            eager_profile = _as_profile_payload(profiles.get("eager"), fallback_algo)
            if eager_profile is not None:
                return eager_profile, "eager"

    def _flat_profile(profile_key):
        suffix = f"_{profile_key}"
        batch_value = entry.get(f"batch_size{suffix}")
        lr_value = entry.get(f"learning_rate{suffix}")
        if batch_value is None and lr_value is None:
            return None
        raw = {
            "batch_size": batch_value,
            "learning_rate": lr_value,
            "estimated_peak_bytes": entry.get(f"estimated_peak_bytes{suffix}"),
            "probe_count": entry.get(f"probe_count{suffix}"),
            "lr_factor": entry.get(f"lr_factor{suffix}", entry.get(f"lr_scale_factor{suffix}")),
            "lr_mode": entry.get(f"lr_mode{suffix}", entry.get(f"lr_scale_mode{suffix}")),
            "algo_version": entry.get(f"algo_version{suffix}", fallback_algo),
        }
        return _as_profile_payload(raw, fallback_algo)

    requested_flat = _flat_profile(requested)
    if requested_flat is not None:
        return requested_flat, requested

    eager_flat = _flat_profile("eager")
    if fallback_to_eager and requested != "eager" and eager_flat is not None:
        return eager_flat, "eager"

    root_profile = _as_profile_payload(entry, fallback_algo)
    if root_profile is not None and (requested == "eager" or fallback_to_eager):
        return root_profile, "eager"

    return None, None


def _cached_profile_is_usable(cached_profile, context_label, current_budget_bytes=None):
    if not isinstance(cached_profile, dict):
        return False

    cached_batch = _safe_int(cached_profile.get("batch_size"), 0)
    cached_lr = _safe_float(cached_profile.get("learning_rate"), 0.0)
    if cached_batch <= 0 or cached_lr <= 0:
        return False

    cached_algo = _safe_int(cached_profile.get("algo_version"), 0)
    if cached_algo != AUTO_TUNE_ALGO_VERSION:
        print(
            f"{context_label}: cached params skipped "
            f"(algo_version={cached_algo}, expected={AUTO_TUNE_ALGO_VERSION})."
        )
        return False

    cached_peak = _safe_int(cached_profile.get("estimated_peak_bytes"), 0)
    if (
        current_budget_bytes is not None
        and current_budget_bytes > 0
        and cached_peak > 0
        and cached_peak > current_budget_bytes
    ):
        print(
            f"{context_label}: cached params skipped "
            f"(cached_peak={_fmt_gib(cached_peak)} > current_budget={_fmt_gib(current_budget_bytes)})."
        )
        return False

    return True


def _write_cache_profile(entry, runtime_profile, profile_payload):
    if not isinstance(entry, dict):
        entry = {}
    if not isinstance(profile_payload, dict):
        return entry

    profile_key = _normalize_runtime_profile(runtime_profile)
    profiles = entry.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[profile_key] = dict(profile_payload)
    entry["profiles"] = profiles

    suffix = f"_{profile_key}"
    entry[f"batch_size{suffix}"] = int(profile_payload.get("batch_size", 0) or 0)
    entry[f"learning_rate{suffix}"] = float(profile_payload.get("learning_rate", 0.0) or 0.0)
    peak_bytes = profile_payload.get("estimated_peak_bytes")
    entry[f"estimated_peak_bytes{suffix}"] = (
        None if peak_bytes is None else int(peak_bytes)
    )
    entry[f"probe_count{suffix}"] = int(profile_payload.get("probe_count", 0) or 0)
    entry[f"lr_mode{suffix}"] = profile_payload.get("lr_mode")
    entry[f"lr_factor{suffix}"] = float(profile_payload.get("lr_factor", 1.0) or 1.0)
    entry[f"algo_version{suffix}"] = int(
        profile_payload.get("algo_version", AUTO_TUNE_ALGO_VERSION)
    )

    if profile_key == "eager":
        entry["batch_size"] = int(profile_payload.get("batch_size", 0) or 0)
        entry["learning_rate"] = float(profile_payload.get("learning_rate", 0.0) or 0.0)
        entry["estimated_peak_bytes"] = (
            None if peak_bytes is None else int(peak_bytes)
        )
        entry["probe_count"] = int(profile_payload.get("probe_count", 0) or 0)
        entry["lr_mode"] = profile_payload.get("lr_mode")
        entry["lr_factor"] = float(profile_payload.get("lr_factor", 1.0) or 1.0)
        entry["lr_scale_mode"] = profile_payload.get("lr_mode")
        entry["lr_scale_factor"] = float(profile_payload.get("lr_factor", 1.0) or 1.0)
        entry["algo_version"] = int(profile_payload.get("algo_version", AUTO_TUNE_ALGO_VERSION))

    entry["last_runtime_profile"] = profile_key
    return entry


def _is_oom_error(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _compute_vram_budget(total_vram_bytes, free_vram_bytes, config):
    auto_cfg = _get_auto_tune_cfg(config)

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
    auto_cfg = _get_auto_tune_cfg(config)

    use_amp = bool(config.get("hardware", {}).get("use_amp", True))
    use_bfloat16 = bool(config.get("hardware", {}).get("use_bfloat16", False))

    configured_batch = max(1, _safe_int(il_cfg.get("batch_size"), 1024))
    min_batch = max(1, _safe_int(auto_cfg.get("min_batch_size"), 64))
    max_batch_default = max(32768, configured_batch * 4)
    max_batch = max(min_batch, _safe_int(auto_cfg.get("max_batch_size"), max_batch_default))
    batch_round_to = max(1, _safe_int(auto_cfg.get("batch_round_to"), 64))
    search_step = max(1, int(batch_round_to))
    if search_step > 1:
        max_batch = max(search_step, _round_batch(max_batch, search_step))

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
            "batch_search_mode": "fast_units",
            "free_limit_bytes": budget_info.get("free_limit_bytes"),
            "safety_margin_bytes": budget_info.get("safety_margin_bytes"),
            "budget_mode": budget_info.get("budget_mode"),
        }

    raw_best_batch = 1

    # Fast search: probe rounded units (batch_round_to) + warm start near configured batch.
    # This cuts probe count versus full integer binary search while keeping robust fit checks.
    if search_step > 1 and not fits(search_step):
        # If even first rounded step does not fit, binary-search the tiny range [1, step-1].
        low = 1
        high = search_step - 1
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        raw_best_batch = int(low)
    else:
        max_units = max(1, int(max_batch // search_step))
        low_units = 1
        high_units = max_units

        warm_units = max(1, min(max_units, int(configured_batch // search_step)))
        if warm_units > 1:
            warm_batch = int(warm_units * search_step)
            if fits(warm_batch):
                low_units = warm_units

                # Find upper region quickly (galloping) starting from configured batch.
                probe_units = min(max_units, max(warm_units + 1, warm_units * 2))
                if probe_units > low_units:
                    if fits(int(probe_units * search_step)):
                        low_units = probe_units
                        while low_units < max_units:
                            next_units = min(max_units, low_units * 2)
                            if next_units <= low_units:
                                break
                            if fits(int(next_units * search_step)):
                                low_units = next_units
                            else:
                                high_units = max(low_units, next_units - 1)
                                break
                    else:
                        high_units = max(low_units, probe_units - 1)
            else:
                high_units = max(1, min(high_units, warm_units - 1))

        if low_units < high_units:
            while low_units < high_units:
                mid_units = (low_units + high_units + 1) // 2
                mid_batch = int(mid_units * search_step)
                if fits(mid_batch):
                    low_units = mid_units
                else:
                    high_units = mid_units - 1

        raw_best_batch = int(low_units * search_step)

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
        "batch_search_mode": "fast_units",
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
    auto_cfg = _get_auto_tune_cfg(config)

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
    auto_cfg = _get_auto_tune_cfg(config)
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

    auto_cfg = _get_auto_tune_cfg(config)
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


def _compile_model_for_auto_tune(model, config, device):
    """
    Try compiling model for probe-only runtime profiling.

    Returns:
    - compiled model wrapper on success, backend label, None
    - None, None, error_message on failure
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None, "CUDA not available"

    use_amp = bool(config.get("hardware", {}).get("use_amp", True))
    use_bfloat16 = bool(config.get("hardware", {}).get("use_bfloat16", False))
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    amp_enabled = bool(use_amp)
    input_planes = int(getattr(model, "input_planes", 16))
    dummy = torch.zeros(
        1,
        input_planes,
        8,
        8,
        device=device,
        dtype=torch.float32,
    ).to(memory_format=torch.channels_last)
    backends = (("default", True), ("cudagraphs", False))
    last_error = None

    for backend_or_mode, is_mode in backends:
        compiled = None
        try:
            if is_mode:
                try:
                    compiled = torch.compile(model, mode=backend_or_mode, dynamic=True)
                except TypeError:
                    compiled = torch.compile(model, mode=backend_or_mode)
            else:
                try:
                    compiled = torch.compile(model, backend=backend_or_mode, dynamic=True)
                except TypeError:
                    compiled = torch.compile(model, backend=backend_or_mode)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    compiled(dummy)
            return compiled, backend_or_mode, None
        except Exception as exc:
            last_error = exc
            compiled = None

    if last_error is None:
        return None, None, "unknown compile error"
    return None, None, f"{type(last_error).__name__}: {last_error}"


def resolve_il_hyperparameters(
    config,
    model,
    device,
    base_dir,
    mode="config",
    runtime_profile="eager",
    fixed_learning_rate=None,
    fallback_to_eager_profile=True,
):
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

    requested_profile = _normalize_runtime_profile(runtime_profile)

    model_hash = build_model_hash(config, include_version=False)
    legacy_model_hash = build_model_hash(config, include_version=True)
    if legacy_model_hash == model_hash:
        legacy_model_hash = None

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
        "runtime_profile_requested": requested_profile,
        "runtime_profile_used": requested_profile,
    }

    if selected_mode != "auto":
        il_cfg["batch_size"] = configured_batch
        il_cfg["learning_rate"] = configured_lr
        print(
            "IL hyperparameters: using config values "
            f"(batch_size={configured_batch}, learning_rate={configured_lr:.6g})."
        )
        return result

    if not _auto_tune_enabled_for(config, target="il"):
        il_cfg["batch_size"] = configured_batch
        il_cfg["learning_rate"] = configured_lr
        result["mode"] = "config"
        result["source"] = "config (auto disabled for IL)"
        print("IL auto-tune disabled by config (auto_tune.use_for_il=false); using config values.")
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
    entries = _ensure_cache_entries(cache)

    device_key = _build_device_key(config, device)
    cached_entry, hash_entries, hash_key_used = _get_cached_entry_for_device(
        entries=entries,
        model_hash=model_hash,
        legacy_model_hash=legacy_model_hash,
        device_key=device_key,
        context_label="IL auto-tune",
    )

    cached_profile, cached_profile_key = _read_cache_profile(
        cached_entry,
        runtime_profile=requested_profile,
        fallback_to_eager=fallback_to_eager_profile,
    )
    if _cached_profile_is_usable(
        cached_profile,
        context_label="IL auto-tune",
        current_budget_bytes=current_budget_bytes,
    ):
        cached_batch = _safe_int(cached_profile.get("batch_size"), 0)
        cached_lr = _safe_float(cached_profile.get("learning_rate"), 0.0)
        cached_peak = _safe_int(cached_profile.get("estimated_peak_bytes"), 0) or None
        cached_probe_count = _safe_int(cached_profile.get("probe_count"), 0)
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
                "lr_scale_factor": _safe_float(cached_profile.get("lr_factor"), 1.0),
                "lr_scale_mode": cached_profile.get("lr_mode"),
                "runtime_profile_used": cached_profile_key or requested_profile,
            }
        )
        profile_note = ""
        if cached_profile_key and cached_profile_key != requested_profile:
            profile_note = f", profile={cached_profile_key} (fallback from {requested_profile})"
        else:
            profile_note = f", profile={cached_profile_key or requested_profile}"
        print(
            "IL auto-tune: loaded cached values "
            f"(hash={model_hash[:12]}, batch_size={int(cached_batch)}, "
            f"lr={float(cached_lr):.6g}{profile_note})."
        )
        if hash_key_used != model_hash and isinstance(hash_entries, dict):
            entries[model_hash] = hash_entries
            cache["entries"] = entries
            try:
                _save_cache(cache_path, cache)
            except Exception:
                pass
        return result

    probe_profile = requested_profile
    probe_model = model
    compile_backend = None
    if requested_profile == "compiled":
        probe_model, compile_backend, compile_error = _compile_model_for_auto_tune(model, config, device)
        if probe_model is None:
            if fallback_to_eager_profile:
                probe_profile = "eager"
                probe_model = model
                print(
                    "IL auto-tune: torch.compile probe unavailable for profile=compiled; "
                    f"falling back to eager ({compile_error})."
                )
            else:
                il_cfg["batch_size"] = configured_batch
                il_cfg["learning_rate"] = configured_lr
                result["mode"] = "config"
                result["source"] = "config (auto unavailable: compile probe failed)"
                result["runtime_profile_used"] = "eager"
                result["compile_probe_error"] = compile_error
                print(
                    "IL auto-tune: compile-specific probe failed; "
                    "keeping current hyperparameters."
                )
                return result
        else:
            result["compile_backend"] = compile_backend

    print(f"  Dedicated VRAM total: {_fmt_gib(dedicated_vram_bytes)}")
    print(f"  Dedicated VRAM free:  {_fmt_gib(free_vram_bytes)}")
    print(
        f"  Budget: {_fmt_gib(current_budget_bytes)} "
        f"(free_limit={_fmt_gib(budget_info['free_limit_bytes'])}, "
        f"safety_margin={_fmt_gib(budget_info['safety_margin_bytes'])})"
    )
    print(
        f"IL auto-tune [{probe_profile}]: estimating batch size from dedicated GPU VRAM..."
    )

    baseline_state = _snapshot_model_state_cpu(model)
    tuned_batch = 0
    tune_info = {}
    tuned_lr = configured_lr
    lr_factor = 1.0
    lr_mode = "config"
    lr_meta = None
    try:
        tuned_batch, tune_info = _find_max_batch_size(
            model=probe_model,
            config=config,
            device=device,
            budget_info=budget_info,
        )

        if fixed_learning_rate is not None and _safe_float(fixed_learning_rate, 0.0) > 0:
            auto_cfg = _get_auto_tune_cfg(config)
            tuned_lr_raw = _safe_float(fixed_learning_rate, configured_lr)
            tuned_lr, _, _ = _clamp_learning_rate(tuned_lr_raw, auto_cfg)
            lr_factor = float(tuned_lr / max(1e-12, float(configured_lr)))
            lr_mode = "fixed"
            lr_meta = {
                "method": "fixed",
                "value": float(tuned_lr),
                "requested_value": float(tuned_lr_raw),
            }
        else:
            tuned_lr, lr_factor, lr_mode, lr_meta = _select_learning_rate(
                model=probe_model,
                config=config,
                configured_lr=configured_lr,
                tuned_batch=tuned_batch,
                device=device,
                use_amp=bool(config.get("hardware", {}).get("use_amp", True)),
                use_bfloat16=bool(config.get("hardware", {}).get("use_bfloat16", False)),
            )
    finally:
        model.load_state_dict(baseline_state, strict=True)
        _clear_probe_state(model, device)
        del baseline_state

    il_cfg["batch_size"] = int(tuned_batch)
    il_cfg["learning_rate"] = float(tuned_lr)

    estimated_peak = tune_info.get("estimated_peak_bytes")
    budget_bytes = tune_info.get("budget_bytes")

    print(
        "IL auto-tune: selected "
        f"batch_size={int(tuned_batch)}, learning_rate={float(tuned_lr):.6g} "
        f"({lr_mode}, factor x{lr_factor:.3f}, profile={probe_profile})."
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
    device_entry = hash_entries.get(device_key)
    if not isinstance(device_entry, dict):
        device_entry = {}

    device_entry.update(
        {
            "model_hash": model_hash,
            "device_key": device_key,
            "device_name": torch.cuda.get_device_name(device),
            "dedicated_vram_bytes": int(dedicated_vram_bytes),
            "free_vram_bytes_at_tune": int(free_vram_bytes),
            "budget_bytes": int(budget_bytes) if budget_bytes is not None else None,
            "free_vram_utilization": float(tune_info.get("free_utilization", 0.0)),
            "budget_mode": tune_info.get("budget_mode", "free_dedicated_vram"),
            "free_limit_bytes": int(tune_info.get("free_limit_bytes", 0) or 0),
            "safety_margin_bytes": int(tune_info.get("safety_margin_bytes", 0) or 0),
            "probe_peak_metric": tune_info.get("probe_peak_metric"),
            "probe_to_train_multiplier": float(tune_info.get("probe_to_train_multiplier", 1.0)),
            "algo_version": AUTO_TUNE_ALGO_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    profile_payload = {
        "runtime_profile": probe_profile,
        "batch_size": int(tuned_batch),
        "learning_rate": float(tuned_lr),
        "algo_version": AUTO_TUNE_ALGO_VERSION,
        "estimated_peak_bytes": int(estimated_peak) if estimated_peak is not None else None,
        "probe_count": int(tune_info.get("probe_count", 0)),
        "lr_mode": lr_mode,
        "lr_factor": float(lr_factor),
        "lr_scale_mode": lr_mode,
        "lr_scale_factor": float(lr_factor),
        "lr_meta": lr_meta,
        "budget_bytes": int(budget_bytes) if budget_bytes is not None else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if probe_profile == "compiled" and compile_backend:
        profile_payload["compile_backend"] = compile_backend

    device_entry = _write_cache_profile(
        entry=device_entry,
        runtime_profile=probe_profile,
        profile_payload=profile_payload,
    )
    hash_entries[device_key] = device_entry
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
            "runtime_profile_used": probe_profile,
        }
    )
    if compile_backend:
        result["compile_backend"] = compile_backend
    return result


def resolve_rl_hyperparameters(config, model, device, base_dir, runtime_profile=None):
    """
    Resolve RL hyperparameters from shared auto-tune cache (produced by IL).

    RL does not run probing; it only reuses cached values when available.
    """
    rl_cfg = config.setdefault("reinforcement_learning", {})
    configured_batch = max(1, _safe_int(rl_cfg.get("batch_size"), 1024))
    configured_lr = _safe_float(rl_cfg.get("learning_rate"), 2e-4)
    if configured_lr <= 0:
        configured_lr = 2e-4

    if runtime_profile is None:
        runtime_profile = "eager"
    requested_profile = _normalize_runtime_profile(runtime_profile)

    model_hash = build_model_hash(config, include_version=False)
    legacy_model_hash = build_model_hash(config, include_version=True)
    if legacy_model_hash == model_hash:
        legacy_model_hash = None

    result = {
        "algo_version": AUTO_TUNE_ALGO_VERSION,
        "source": "config",
        "batch_size": configured_batch,
        "learning_rate": configured_lr,
        "model_hash": model_hash,
        "cache_path": str(_resolve_cache_path(base_dir, config)),
        "cache_hit": False,
        "lr_scale_factor": 1.0,
        "lr_scale_mode": None,
        "runtime_profile_requested": requested_profile,
        "runtime_profile_used": requested_profile,
    }

    if not _auto_tune_enabled_for(config, target="rl"):
        rl_cfg["batch_size"] = configured_batch
        rl_cfg["learning_rate"] = configured_lr
        result["source"] = "config (auto disabled for RL)"
        print("RL auto-tune disabled by config (auto_tune.use_for_rl=false); using config values.")
        return result

    if device.type != "cuda" or not torch.cuda.is_available():
        rl_cfg["batch_size"] = configured_batch
        rl_cfg["learning_rate"] = configured_lr
        result["source"] = "config (auto unavailable: non-CUDA)"
        print("RL auto-tune unavailable on non-CUDA device; using config values.")
        return result

    cache_path = _resolve_cache_path(base_dir, config)
    cache = _load_cache(cache_path)
    entries = _ensure_cache_entries(cache)
    device_key = _build_device_key(config, device)
    cached_entry, hash_entries, hash_key_used = _get_cached_entry_for_device(
        entries=entries,
        model_hash=model_hash,
        legacy_model_hash=legacy_model_hash,
        device_key=device_key,
        context_label="RL auto-tune",
    )

    cached_profile, cached_profile_key = _read_cache_profile(
        cached_entry,
        runtime_profile=requested_profile,
        fallback_to_eager=True,
    )
    if _cached_profile_is_usable(cached_profile, context_label="RL auto-tune"):
        cached_batch = _safe_int(cached_profile.get("batch_size"), 0)
        cached_lr = _safe_float(cached_profile.get("learning_rate"), 0.0)
        rl_cfg["batch_size"] = int(cached_batch)
        rl_cfg["learning_rate"] = float(cached_lr)
        result.update(
            {
                "source": "auto-cache",
                "batch_size": int(cached_batch),
                "learning_rate": float(cached_lr),
                "cache_hit": True,
                "lr_scale_factor": _safe_float(
                    cached_profile.get("lr_factor"),
                    1.0,
                ),
                "lr_scale_mode": cached_profile.get("lr_mode"),
                "runtime_profile_used": cached_profile_key or requested_profile,
            }
        )
        profile_note = ""
        if cached_profile_key and cached_profile_key != requested_profile:
            profile_note = f", profile={cached_profile_key} (fallback from {requested_profile})"
        else:
            profile_note = f", profile={cached_profile_key or requested_profile}"
        print(
            "RL auto-tune: loaded cached values "
            f"(hash={model_hash[:12]}, batch_size={int(cached_batch)}, "
            f"lr={float(cached_lr):.6g}{profile_note})."
        )
        if hash_key_used != model_hash and isinstance(hash_entries, dict):
            entries[model_hash] = hash_entries
            cache["entries"] = entries
            try:
                _save_cache(cache_path, cache)
            except Exception:
                pass
        return result

    rl_cfg["batch_size"] = configured_batch
    rl_cfg["learning_rate"] = configured_lr
    result["source"] = "config (auto-cache miss)"
    print("RL auto-tune: cache miss or unusable entry; using config values.")
    return result
