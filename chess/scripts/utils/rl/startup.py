"""RL startup flow (menu, checkpoint catalog, resume/transfer loading)."""

from datetime import datetime
from pathlib import Path
import sys

from src.model import load_checkpoint_file, transfer_matching_weights


def _print_block_title(title):
    line = "=" * 92
    print(f"\n{line}")
    print(title)
    print(line)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_relative_to_base(path, base_dir):
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _top_folder_from_rel(path_rel):
    parts = path_rel.replace("/", "\\").split("\\")
    if len(parts) <= 1:
        return "root"
    return parts[0]


def _folder_rank(folder):
    key = str(folder).strip().lower()
    if key == "root":
        return 0
    if key == "il":
        return 1
    if key == "rl":
        return 2
    return 3


def _sort_checkpoint_catalog(catalog):
    return sorted(
        catalog,
        key=lambda entry: (
            1 if entry.get("error") else 0,
            _folder_rank(entry.get("folder", "")),
            str(entry.get("folder", "")).lower(),
            1 if entry.get("estimated_elo") is None else 0,
            -float(entry.get("estimated_elo") or 0.0),
            -float(entry.get("mtime_ts") or 0.0),
            str(entry.get("model_name", "")).lower(),
        ),
    )


def _normalize_source_state(source_state, target_keys):
    normalized = {}
    for key, tensor in source_state.items():
        if key in target_keys:
            norm_key = key
        elif key.startswith("module.") and key[7:] in target_keys:
            norm_key = key[7:]
        else:
            norm_key = key
        if norm_key not in normalized:
            normalized[norm_key] = tensor
    return normalized


def _compute_compatibility(target_state, source_state):
    target_keys = set(target_state.keys())
    normalized_source = _normalize_source_state(source_state, target_keys)

    total_tensors = len(target_state)
    total_elements = sum(t.numel() for t in target_state.values())

    matched_tensors = 0
    matched_elements = 0
    missing_tensors = 0
    shape_mismatches = 0

    for key, target_tensor in target_state.items():
        source_tensor = normalized_source.get(key)
        if source_tensor is None:
            missing_tensors += 1
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatches += 1
            continue
        matched_tensors += 1
        matched_elements += target_tensor.numel()

    unexpected_tensors = sum(1 for key in normalized_source.keys() if key not in target_keys)
    compatibility_ratio = (matched_elements / total_elements) if total_elements else 0.0
    strict_resume_ok = (
        matched_tensors == total_tensors
        and missing_tensors == 0
        and shape_mismatches == 0
        and unexpected_tensors == 0
    )

    return {
        "matched_tensors": matched_tensors,
        "total_tensors": total_tensors,
        "matched_elements": matched_elements,
        "total_elements": total_elements,
        "missing_tensors": missing_tensors,
        "shape_mismatches": shape_mismatches,
        "unexpected_tensors": unexpected_tensors,
        "compatibility_ratio": compatibility_ratio,
        "strict_resume_ok": strict_resume_ok,
    }


def _collect_rl_checkpoints(models_dir, best_model_rl_path, rl_dir):
    candidates = []
    if best_model_rl_path.exists():
        candidates.append(best_model_rl_path)

    rl_checkpoints = sorted(
        rl_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in rl_checkpoints:
        if checkpoint not in candidates:
            candidates.append(checkpoint)

    all_checkpoints = sorted(
        models_dir.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in all_checkpoints:
        if checkpoint not in candidates:
            candidates.append(checkpoint)

    return candidates


def _choose_start_mode(has_checkpoints):
    if not has_checkpoints:
        _print_block_title("RL Startup")
        print("No checkpoints found. Starting new RL training.")
        return "new"

    _print_block_title("RL Startup")
    print("1) New RL training (init from best IL if available)")
    print("2) Resume full state (model + optimizer/scaler)")
    print("3) Transfer matching weights only")

    default_choice = "2"
    mapping = {
        "1": "new",
        "2": "resume",
        "3": "transfer",
        "new": "new",
        "resume": "resume",
        "transfer": "transfer",
    }

    while True:
        try:
            choice = input(f"Choose [1/2/3] (default {default_choice}): ").strip().lower()
        except EOFError:
            choice = default_choice
        if not choice:
            choice = default_choice

        selected = mapping.get(choice)
        if selected is not None:
            return selected
        print("Invalid choice. Enter 1, 2, 3, or press Enter for default.")


def _build_checkpoint_catalog(candidates, model, device, base_dir):
    target_state = model.state_dict()
    catalog = []

    for checkpoint_path in candidates:
        path_rel = _path_relative_to_base(checkpoint_path, base_dir)
        stat = checkpoint_path.stat()
        entry = {
            "path": checkpoint_path,
            "path_rel": path_rel,
            "folder": _top_folder_from_rel(path_rel),
            "model_name": checkpoint_path.name,
            "size_mb": stat.st_size / (1024 ** 2),
            "mtime_ts": stat.st_mtime,
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            "epoch": None,
            "version": None,
            "top1": None,
            "val_loss": None,
            "policy_loss": None,
            "estimated_elo": None,
            "optimizer_present": False,
            "compatibility_ratio": 0.0,
            "strict_resume_ok": False,
            "error": None,
        }

        try:
            checkpoint = load_checkpoint_file(str(checkpoint_path), device)
        except Exception as exc:
            entry["error"] = f"load failed: {exc}"
            catalog.append(entry)
            continue

        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            entry["error"] = "missing model_state_dict"
            catalog.append(entry)
            continue

        entry["epoch"] = _safe_int(checkpoint.get("epoch"))
        entry["version"] = checkpoint.get("version")
        entry["top1"] = _safe_float(checkpoint.get("val_policy_top1", checkpoint.get("policy_top1_acc")))
        entry["val_loss"] = _safe_float(checkpoint.get("val_loss", checkpoint.get("loss")))
        entry["policy_loss"] = _safe_float(checkpoint.get("val_policy_loss", checkpoint.get("policy_loss")))
        entry["estimated_elo"] = _safe_float(
            checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo"))
        )
        entry["optimizer_present"] = "optimizer_state_dict" in checkpoint

        compat = _compute_compatibility(target_state, model_state)
        entry.update(compat)
        catalog.append(entry)

    return _sort_checkpoint_catalog(catalog)


def _print_checkpoint_catalog(catalog, start_mode):
    if not catalog:
        print("\nNo checkpoints available.")
        return

    mode_label = "Resume" if start_mode == "resume" else "Transfer"
    _print_block_title(f"Available Checkpoints ({mode_label})")
    print(" ID  Folder  Version  Epoch   Top1      ValLoss    PolLoss      Elo   Compat   Strict  Opt   SizeMB  Updated           Checkpoint")
    print("---- ------- -------- ------ -------- ---------- ---------- -------- -------- ------- ---- ------- ----------------- -----------------------------------------")

    last_folder = None
    for idx, entry in enumerate(catalog, start=1):
        folder = entry.get("folder", "root")
        if folder != last_folder:
            print(f"-- {folder} --")
            last_folder = folder

        if entry.get("error"):
            print(
                f"{idx:>3}  {folder:<7} {'n/a':<8}   n/a      n/a        n/a        n/a      n/a      n/a     n/a   "
                f" n/a  {entry['size_mb']:>6.1f}  {entry['modified']:<17} "
                f"{entry['model_name']} [ERROR: {entry['error']}]"
            )
            continue

        epoch_value = entry.get("epoch")
        epoch_str = f"{epoch_value + 1}" if epoch_value is not None else "n/a"
        version_str = str(entry.get("version") or "n/a")[:8]
        top1 = entry.get("top1")
        top1_str = f"{top1 * 100:6.2f}%" if top1 is not None else "  n/a  "
        val_loss = entry.get("val_loss")
        loss_str = f"{val_loss:8.4f}" if val_loss is not None else "   n/a  "
        pol_loss = entry.get("policy_loss")
        pol_loss_str = f"{pol_loss:8.4f}" if pol_loss is not None else "   n/a  "
        estimated_elo = entry.get("estimated_elo")
        elo_str = f"{int(round(float(estimated_elo))):>6}" if estimated_elo is not None else "  n/a "
        compat_str = f"{entry.get('compatibility_ratio', 0.0) * 100:6.2f}%"
        strict_str = "yes" if entry.get("strict_resume_ok", False) else "no"
        optimizer_str = "yes" if entry.get("optimizer_present", False) else "no"
        print(
            f"{idx:>3}  {folder:<7} {version_str:<8} {epoch_str:>6}  {top1_str:>8}  {loss_str:>10}  {pol_loss_str:>10}  {elo_str:>8}  "
            f"{compat_str:>6}   {strict_str:>5}  {optimizer_str:>3}  "
            f"{entry['size_mb']:>6.1f}  {entry['modified']:<17} {entry['model_name']}"
        )


def _choose_checkpoint_path(catalog):
    if not catalog:
        return None

    valid_entries = [entry for entry in catalog if not entry.get("error")]
    if not valid_entries:
        return None

    if not sys.stdin.isatty():
        return valid_entries[0]["path"]

    while True:
        try:
            choice = input(f"Select checkpoint [1-{len(catalog)}] (default 1): ").strip()
        except EOFError:
            choice = "1"

        if not choice:
            choice = "1"

        try:
            idx = int(choice)
            if 1 <= idx <= len(catalog):
                selected = catalog[idx - 1]
                if selected.get("error"):
                    print("Selected checkpoint is invalid. Choose another one.")
                    continue
                return selected["path"]
        except ValueError:
            pass

        print(f"Invalid choice. Enter a number from 1 to {len(catalog)}.")


def _find_checkpoint_entry(catalog, checkpoint_path):
    for entry in catalog:
        if entry.get("path") == checkpoint_path:
            return entry
    return None


def _print_transfer_report(report):
    print("Transfer report:")
    print(f"  matched tensors:  {report['matched_tensors']}/{report['total_tensors']}")
    print(
        f"  matched elements: {report['matched_elements']:,}/{report['total_elements']:,} "
        f"({report['match_ratio']:.1%})"
    )
    print(f"  shape mismatches: {len(report['shape_mismatch'])}")
    print(f"  missing tensors:  {len(report['missing_keys'])}")
    print(f"  unexpected keys:  {len(report['unexpected_keys'])}")


def plan_rl_startup(model, device, models_dir, best_model_rl_path, rl_dir):
    """Interactive startup menu + checkpoint selection for RL."""
    available_checkpoints = _collect_rl_checkpoints(models_dir, best_model_rl_path, rl_dir)
    selected_checkpoint = None
    checkpoint_catalog = []

    start_mode = _choose_start_mode(has_checkpoints=(len(available_checkpoints) > 0))
    if start_mode in {"resume", "transfer"}:
        print("\nScanning checkpoints (metrics + compatibility)...")
        checkpoint_catalog = _build_checkpoint_catalog(available_checkpoints, model, device, models_dir)
        _print_checkpoint_catalog(checkpoint_catalog, start_mode)
        selected_checkpoint = _choose_checkpoint_path(checkpoint_catalog)

    if start_mode in {"resume", "transfer"} and selected_checkpoint is None:
        print("WARNING: No valid checkpoint available. Falling back to new RL training.")
        start_mode = "new"

    selected_entry = _find_checkpoint_entry(checkpoint_catalog, selected_checkpoint)
    selected_checkpoint_label = None
    if selected_checkpoint is not None:
        selected_checkpoint_label = _path_relative_to_base(selected_checkpoint, models_dir)

    return {
        "start_mode": start_mode,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_label": selected_checkpoint_label,
        "selected_entry": selected_entry,
    }


def apply_rl_startup_plan(
    startup_plan,
    model,
    optimizer,
    scaler,
    device,
    default_new_checkpoint=None,
):
    """Apply selected RL startup plan (load resume/transfer/new-init checkpoint)."""
    start_mode = startup_plan.get("start_mode", "new")
    selected_checkpoint = startup_plan.get("selected_checkpoint")
    selected_checkpoint_label = startup_plan.get("selected_checkpoint_label")
    selected_entry = startup_plan.get("selected_entry") or {}

    selected_compatibility_ratio = selected_entry.get("compatibility_ratio")
    transfer_match_ratio = None
    start_iteration = 0
    best_win_rate = 0.0

    if start_mode in {"resume", "transfer"} and selected_checkpoint is None:
        print("WARNING: Startup plan has no checkpoint. Falling back to new RL training.")
        start_mode = "new"

    if start_mode in {"resume", "transfer"} and selected_checkpoint is not None:
        print(f"\nLoading startup checkpoint: {selected_checkpoint_label}")
        checkpoint = load_checkpoint_file(str(selected_checkpoint), device)

        if start_mode == "resume":
            try:
                model_state = checkpoint.get("model_state_dict")
                if not isinstance(model_state, dict):
                    raise KeyError("missing model_state_dict")
                model.load_state_dict(model_state)

                checkpoint_epoch = _safe_int(checkpoint.get("epoch"))
                if checkpoint_epoch is not None:
                    start_iteration = max(0, checkpoint_epoch + 1)

                checkpoint_win_rate = _safe_float(checkpoint.get("win_rate"))
                if checkpoint_win_rate is not None:
                    best_win_rate = checkpoint_win_rate

                if optimizer is not None:
                    optimizer_state = checkpoint.get("optimizer_state_dict")
                    if isinstance(optimizer_state, dict):
                        optimizer.load_state_dict(optimizer_state)
                    else:
                        print("Resume warning: optimizer_state_dict not found; optimizer reset.")

                if scaler is not None and scaler.is_enabled():
                    scaler_state = checkpoint.get("scaler_state_dict")
                    if isinstance(scaler_state, dict):
                        scaler.load_state_dict(scaler_state)

                print(
                    f"Resume loaded: next_iteration={start_iteration + 1}, "
                    f"best_win_rate={best_win_rate:.2%}"
                )
            except Exception as exc:
                print(f"WARNING: Full resume failed ({exc})")
                print("Falling back to transfer mode (matching tensors only).")
                report = transfer_matching_weights(model, checkpoint)
                transfer_match_ratio = report.get("match_ratio")
                _print_transfer_report(report)
                start_mode = "transfer"
                start_iteration = 0
                best_win_rate = 0.0
        else:
            report = transfer_matching_weights(model, checkpoint)
            transfer_match_ratio = report.get("match_ratio")
            _print_transfer_report(report)
            start_iteration = 0
            best_win_rate = 0.0

    if start_mode == "new":
        init_path = Path(default_new_checkpoint) if default_new_checkpoint is not None else None
        if init_path is not None and init_path.exists():
            print(f"\nInitializing RL from: {init_path}")
            checkpoint = load_checkpoint_file(str(init_path), device)
            model_state = checkpoint.get("model_state_dict")
            try:
                if not isinstance(model_state, dict):
                    raise KeyError("missing model_state_dict")
                model.load_state_dict(model_state)
                print("Loaded initialization checkpoint (strict).")
            except Exception as exc:
                print(f"Strict init load failed ({exc}). Trying transfer.")
                report = transfer_matching_weights(model, checkpoint)
                transfer_match_ratio = report.get("match_ratio")
                _print_transfer_report(report)
        else:
            print("No IL init checkpoint found. Starting RL from random weights.")

    return {
        "start_mode": start_mode,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_label": selected_checkpoint_label,
        "start_iteration": start_iteration,
        "best_win_rate": best_win_rate,
        "selected_compatibility_ratio": selected_compatibility_ratio,
        "transfer_match_ratio": transfer_match_ratio,
    }
