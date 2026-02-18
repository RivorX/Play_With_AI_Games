"""RL startup flow (menu, checkpoint catalog, resume/transfer loading)."""

from pathlib import Path
import sys

from src.model import load_checkpoint_file, transfer_matching_weights
from utils.shared.model_catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_elo,
)


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
        entry = load_checkpoint_metadata(checkpoint_path, base_dir)
        entry["estimated_elo"] = entry.get("elo")
        entry["optimizer_present"] = bool(entry.get("optimizer", False))
        entry["compatibility_ratio"] = None
        entry["strict_resume_ok"] = False

        if entry.get("error"):
            catalog.append(entry)
            continue

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

        compat = _compute_compatibility(target_state, model_state)
        entry.update(compat)
        catalog.append(entry)

    return sort_entries_by_folder_and_elo(catalog)


def _print_checkpoint_catalog(catalog, start_mode):
    if not catalog:
        print("\nNo checkpoints available.")
        return

    mode_label = "Resume" if start_mode == "resume" else "Transfer"
    print_model_table(
        catalog,
        title=f"Available Checkpoints ({mode_label})",
        show_folder=True,
        show_version=True,
        show_modified=True,
        show_swa=True,
        show_opt=True,
        show_compat=True,
        show_strict=True,
        group_by_folder=True,
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
