"""IL startup flow (menu, checkpoint catalog, resume/transfer loading)."""

import sys

from src.model import load_checkpoint_file, transfer_matching_weights
from utils.shared.model_catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_elo,
)


def ask_resume_additional_epochs(completed_epochs, default_additional):
    """Ask user how many epochs to add on top of completed resume epochs."""
    default_additional = max(0, int(default_additional))
    if not sys.stdin.isatty():
        return default_additional

    while True:
        try:
            raw = input(
                f"Resume: completed {completed_epochs} epoch(s). "
                f"How many additional epochs to add? (default {default_additional}): "
            ).strip()
        except EOFError:
            raw = ""

        if raw == "":
            return default_additional

        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
        print("Invalid value. Enter an integer >= 0.")


def ask_il_target_positions(default_millions=15):
    """Ask for the IL training-position target in millions, or max for all."""
    default_is_max = isinstance(default_millions, str) and default_millions.strip().lower() in {
        "max",
        "all",
        "wszystkie",
    }
    if default_is_max:
        if not sys.stdin.isatty():
            return "max"
        default_text = "max"
        default_millions = None
    else:
        try:
            default_millions = float(default_millions)
        except (TypeError, ValueError):
            default_millions = 15.0
        default_millions = max(1.0, default_millions)
        default_text = f"{default_millions:g}"

    if not sys.stdin.isatty():
        return int(round(default_millions * 1_000_000))

    _print_block_title("IL Data")
    print("How many million positions should IL use?")
    print("Enter a number, or max for all available positions.")

    while True:
        try:
            raw = input(f"Target positions in millions (default {default_text}): ").strip().lower()
        except EOFError:
            raw = ""

        if not raw:
            raw = default_text
        if raw in {"max", "all", "wszystkie"}:
            return "max"

        raw = raw.replace(",", ".")
        try:
            value = float(raw)
            if value > 0:
                return int(round(value * 1_000_000))
        except ValueError:
            pass
        print("Invalid value. Enter a number like 15, or max.")


def ask_transfer_freeze_epochs(default_epochs=0):
    """Ask how many epochs to freeze all but changed params after transfer."""
    default_epochs = max(0, int(default_epochs))
    if not sys.stdin.isatty():
        return default_epochs

    while True:
        try:
            raw = input(
                "Transfer: freeze all except changed layers for how many epochs? "
                f"(default {default_epochs}, 0 = disabled): "
            ).strip()
        except EOFError:
            raw = ""

        if raw == "":
            return default_epochs

        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            pass
        print("Invalid value. Enter an integer >= 0.")


def ask_transfer_post_unfreeze_lr(default_lr=None):
    """Ask for optional manual LR applied only after full unfreeze in transfer mode."""
    if not sys.stdin.isatty():
        return default_lr

    default_text = "blank"
    if default_lr is not None:
        try:
            default_text = f"{float(default_lr):.6g}"
        except (TypeError, ValueError):
            default_text = "blank"

    while True:
        try:
            raw = input(
                "Transfer: manual learning rate after full unfreeze? "
                f"(default {default_text}, blank = keep current schedule): "
            ).strip()
        except EOFError:
            raw = ""

        if raw == "":
            return default_lr

        lowered = raw.lower()
        if lowered in {"none", "off", "auto", "schedule"}:
            return None

        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Invalid value. Enter a float > 0, or blank to keep current schedule.")


def suggest_transfer_post_unfreeze_lr(base_lr, compatibility_ratio=None):
    """Suggest a safer post-unfreeze LR based on transfer compatibility."""
    try:
        base_lr = float(base_lr)
    except (TypeError, ValueError):
        return None

    if base_lr <= 0:
        return None

    try:
        compat = float(compatibility_ratio) if compatibility_ratio is not None else None
    except (TypeError, ValueError):
        compat = None

    if compat is None:
        factor = 0.5
    elif compat < 0.70:
        factor = 0.35
    elif compat < 0.80:
        factor = 0.5
    elif compat < 0.90:
        factor = 0.7
    else:
        factor = 0.85

    suggested = base_lr * factor
    return float(f"{suggested:.6g}")


def ask_il_hyperparam_source(default_mode="config"):
    """Ask whether to use auto-tuned IL batch/LR or config values."""
    if default_mode not in {"auto", "config"}:
        default_mode = "config"
    if not sys.stdin.isatty():
        return default_mode

    _print_block_title("IL Speed")
    print("1) Auto batch + LR")
    print("2) Config batch + LR")

    default_choice = "1" if default_mode == "auto" else "2"
    mapping = {
        "1": "auto",
        "2": "config",
        "auto": "auto",
        "config": "config",
    }
    while True:
        try:
            choice = input(f"Choose [1/2] (default {default_choice}): ").strip().lower()
        except EOFError:
            choice = default_choice

        if not choice:
            choice = default_choice

        selected = mapping.get(choice)
        if selected is not None:
            return selected

        print("Invalid choice. Enter 1, 2, or press Enter for default.")


def _print_block_title(title):
    line = "=" * 56
    print(f"\n{line}")
    print(title)
    print(line)


def _path_relative_to_base(path, base_dir):
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _collect_il_checkpoints(best_model_path, il_dir):
    candidates = []
    il_checkpoints = sorted(
        il_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    has_version_best = any(checkpoint.name.endswith("_best.pt") for checkpoint in il_checkpoints)
    if best_model_path.exists() and not has_version_best:
        candidates.append(best_model_path)

    for checkpoint in il_checkpoints:
        if checkpoint not in candidates:
            candidates.append(checkpoint)
    return candidates


def ask_il_start_mode(has_checkpoints):
    if not has_checkpoints:
        _print_block_title("IL Startup")
        print("No checkpoints found. Starting new training.")
        return "new"

    _print_block_title("IL Startup")
    print("1) New")
    print("2) Resume")
    print("3) Transfer weights")

    default_choice = "1"

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


def has_il_checkpoints(best_model_path, il_dir):
    return len(_collect_il_checkpoints(best_model_path, il_dir)) > 0


def _normalize_source_state(source_state, target_keys):
    wrapper_prefixes = ("module", "_orig_mod")

    def _normalize_key(key):
        if key in target_keys:
            return key

        parts = key.split(".")
        while len(parts) > 1 and parts[0] in wrapper_prefixes:
            parts = parts[1:]
            candidate = ".".join(parts)
            if candidate in target_keys:
                return candidate

        if key.startswith("module.") and key[7:] in target_keys:
            return key[7:]
        return key

    normalized = {}
    for key, tensor in source_state.items():
        norm_key = _normalize_key(key)
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


def _build_transfer_report_preview(target_state, source_state):
    """Build transfer compatibility report without mutating model weights."""
    target_keys = set(target_state.keys())
    normalized_source = _normalize_source_state(source_state, target_keys)

    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    shape_mismatch = []

    for key, tensor in normalized_source.items():
        if key not in target_state:
            unexpected_keys.append(key)
            continue
        if target_state[key].shape != tensor.shape:
            shape_mismatch.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
            continue
        matched_keys.append(key)

    matched_set = set(matched_keys)
    for key in target_state.keys():
        if key not in matched_set:
            missing_keys.append(key)

    matched_elements = sum(target_state[k].numel() for k in matched_keys)
    total_elements = sum(v.numel() for v in target_state.values())

    return {
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
        "matched_tensors": len(matched_keys),
        "total_tensors": len(target_state),
        "matched_elements": matched_elements,
        "total_elements": total_elements,
        "match_ratio": (matched_elements / total_elements) if total_elements else 0.0,
    }


def _infer_changed_parameter_names(model, transfer_report):
    """Infer parameter tensors that were not transferred (shape mismatch/missing)."""
    changed_keys = set(transfer_report.get("missing_keys", []))
    for entry in transfer_report.get("shape_mismatch", []):
        if entry:
            changed_keys.add(entry[0])

    if not changed_keys:
        return []

    changed_prefixes = set()
    for key in changed_keys:
        if "." in key:
            changed_prefixes.add(key.rsplit(".", 1)[0])
        else:
            changed_prefixes.add(key)

    changed_params = set()
    for param_name, _ in model.named_parameters():
        if param_name in changed_keys:
            changed_params.add(param_name)
            continue
        for prefix in changed_prefixes:
            if param_name.startswith(prefix + "."):
                changed_params.add(param_name)
                break

    return sorted(changed_params)


def plan_il_startup(model, device, base_dir, best_model_path, il_dir, start_mode=None, base_learning_rate=None):
    """Interactive startup menu + checkpoint selection (no state loading yet)."""
    available_checkpoints = _collect_il_checkpoints(best_model_path, il_dir)
    catalog_base_dir = best_model_path.parent
    selected_checkpoint = None
    checkpoint_catalog = []

    has_checkpoints = len(available_checkpoints) > 0
    if start_mode not in {"new", "resume", "transfer"}:
        start_mode = ask_il_start_mode(has_checkpoints=has_checkpoints)
    elif not has_checkpoints and start_mode in {"resume", "transfer"}:
        _print_block_title("IL Startup")
        print("No checkpoints found. Starting new training.")
        start_mode = "new"

    if start_mode in {"resume", "transfer"}:
        print("\nScanning checkpoints (metrics + compatibility)...")
        checkpoint_catalog = _build_checkpoint_catalog(available_checkpoints, model, device, catalog_base_dir)
        _print_checkpoint_catalog(checkpoint_catalog, start_mode)
        selected_checkpoint = _choose_checkpoint_path(checkpoint_catalog)

    if start_mode in {"resume", "transfer"} and selected_checkpoint is None:
        print("WARNING: No valid checkpoint available. Falling back to new training.")
        start_mode = "new"

    selected_entry = _find_checkpoint_entry(checkpoint_catalog, selected_checkpoint)
    selected_checkpoint_label = None
    transfer_freeze_epochs = 0
    transfer_trainable_param_names = []
    transfer_post_unfreeze_lr = None
    transfer_post_unfreeze_lr_suggested = None

    if selected_checkpoint is not None:
        selected_checkpoint_label = _path_relative_to_base(selected_checkpoint, catalog_base_dir)

    # Ask transfer warmup settings immediately after checkpoint selection
    # (before logger init and data loading).
    if start_mode == "transfer" and selected_checkpoint is not None:
        try:
            checkpoint = load_checkpoint_file(str(selected_checkpoint), device)
            model_state = checkpoint.get("model_state_dict")
            if isinstance(model_state, dict):
                target_keys = set(model.state_dict().keys())
                normalized_state = _normalize_source_state(model_state, target_keys)
                preview_report = _build_transfer_report_preview(model.state_dict(), normalized_state)
                transfer_trainable_param_names = _infer_changed_parameter_names(model, preview_report)
                if transfer_trainable_param_names:
                    compatibility_ratio = None
                    if selected_entry is not None:
                        compatibility_ratio = selected_entry.get("compatibility_ratio")
                    transfer_post_unfreeze_lr_suggested = suggest_transfer_post_unfreeze_lr(
                        base_learning_rate,
                        compatibility_ratio=compatibility_ratio,
                    )
                    print(
                        "Transfer warmup candidates "
                        f"(changed params): {len(transfer_trainable_param_names)}"
                    )
                    if transfer_post_unfreeze_lr_suggested is not None:
                        compat_info = "n/a"
                        if compatibility_ratio is not None:
                            compat_info = f"{float(compatibility_ratio) * 100:.2f}%"
                        print(
                            "Transfer LR suggestion after full unfreeze: "
                            f"{transfer_post_unfreeze_lr_suggested:.6g} "
                            f"(compatibility={compat_info}, current_lr={float(base_learning_rate):.6g})"
                        )
                    transfer_freeze_epochs = ask_transfer_freeze_epochs(default_epochs=0)
                    transfer_post_unfreeze_lr = ask_transfer_post_unfreeze_lr(
                        default_lr=transfer_post_unfreeze_lr_suggested
                    )
                else:
                    print("Transfer warmup skipped: no changed trainable params detected.")
            else:
                print("Transfer warmup skipped: checkpoint has no model_state_dict.")
        except Exception as exc:
            print(f"Transfer warmup setup warning: {exc}")

    return {
        "start_mode": start_mode,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_label": selected_checkpoint_label,
        "selected_entry": selected_entry,
        "transfer_freeze_epochs": transfer_freeze_epochs,
        "transfer_trainable_param_names": transfer_trainable_param_names,
        "transfer_post_unfreeze_lr": transfer_post_unfreeze_lr,
        "transfer_post_unfreeze_lr_suggested": transfer_post_unfreeze_lr_suggested,
    }


def apply_il_startup_plan(startup_plan, model, optimizer, scheduler, scaler, device):
    """Apply previously selected startup plan (loads checkpoint states if needed)."""
    start_mode = startup_plan.get("start_mode", "new")
    selected_checkpoint = startup_plan.get("selected_checkpoint", None)
    selected_checkpoint_label = startup_plan.get("selected_checkpoint_label", None)
    selected_entry = startup_plan.get("selected_entry", None)

    if start_mode in {"resume", "transfer"} and selected_checkpoint is None:
        print("WARNING: Startup plan has no checkpoint. Falling back to new training.")
        start_mode = "new"

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    estimated_elo = None
    estimated_elo_epoch = None
    estimated_elo_nn = None
    estimated_elo_mcts = None
    estimated_elo_mcts_simulations = None
    selected_compatibility_ratio = None
    transfer_match_ratio = None
    transfer_freeze_epochs = int(startup_plan.get("transfer_freeze_epochs", 0) or 0)
    transfer_trainable_param_names = list(startup_plan.get("transfer_trainable_param_names") or [])
    transfer_post_unfreeze_lr = _safe_float(startup_plan.get("transfer_post_unfreeze_lr"))

    if start_mode in {"resume", "transfer"}:
        if selected_checkpoint_label is None:
            selected_checkpoint_label = str(selected_checkpoint)
        print(f"\nLoading startup checkpoint: {selected_checkpoint_label}")
        if selected_entry is not None and not selected_entry.get("error"):
            top1 = selected_entry.get("top1")
            val_loss = selected_entry.get("val_loss")
            policy_loss = selected_entry.get("policy_loss")
            selected_elo = selected_entry.get("estimated_elo")
            selected_compatibility_ratio = selected_entry.get("compatibility_ratio")
            top1_info = f"{top1 * 100:.2f}%" if top1 is not None else "n/a"
            loss_info = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            pol_loss_info = f"{policy_loss:.4f}" if policy_loss is not None else "n/a"
            elo_info = (
                f"{int(round(float(selected_elo)))}"
                if selected_elo is not None
                else "n/a"
            )
            compat_info = f"{(selected_compatibility_ratio or 0.0) * 100:.2f}%"
            strict_info = "yes" if selected_entry.get("strict_resume_ok", False) else "no"
            print(
                "  checkpoint stats: "
                f"top1={top1_info}, val_loss={loss_info}, policy_loss={pol_loss_info}, elo={elo_info}, "
                f"compatibility={compat_info}, strict_resume={strict_info}"
            )
        elif selected_entry is not None and selected_entry.get("error"):
            print(f"  checkpoint scan warning: {selected_entry['error']}")

        checkpoint = load_checkpoint_file(str(selected_checkpoint), device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint missing 'model_state_dict': {selected_checkpoint}")
        target_keys = set(model.state_dict().keys())
        normalized_model_state = _normalize_source_state(checkpoint["model_state_dict"], target_keys)

        estimated_elo = _safe_float(
            checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo"))
        )
        estimated_elo_nn = _safe_float(
            checkpoint.get("estimated_elo_nn", checkpoint.get("last_estimated_elo_nn"))
        )
        estimated_elo_mcts = _safe_float(
            checkpoint.get("estimated_elo_mcts", checkpoint.get("last_estimated_elo_mcts"))
        )
        estimated_elo_mcts_simulations = _safe_int(
            checkpoint.get("estimated_elo_mcts_simulations")
        )
        estimated_elo_epoch = _safe_int(checkpoint.get("estimated_elo_epoch"))
        if estimated_elo_epoch is None:
            epoch_from_ckpt = _safe_int(checkpoint.get("epoch"))
            if epoch_from_ckpt is not None:
                estimated_elo_epoch = epoch_from_ckpt + 1
        if estimated_elo is None and selected_entry is not None:
            estimated_elo = _safe_float(selected_entry.get("estimated_elo"))
        if selected_entry is not None:
            if estimated_elo_nn is None:
                estimated_elo_nn = _safe_float(selected_entry.get("elo_nn"))
            if estimated_elo_mcts is None:
                estimated_elo_mcts = _safe_float(selected_entry.get("elo_mcts"))
            if estimated_elo_mcts_simulations is None:
                estimated_elo_mcts_simulations = _safe_int(
                    selected_entry.get("elo_mcts_simulations")
                )

        if start_mode == "resume":
            try:
                model.load_state_dict(normalized_model_state)
                print("Resume: model state loaded")
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                optimizer_state_loaded = False

                if "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        optimizer_state_loaded = True
                        print("Resume: optimizer state loaded")
                    except Exception as exc:
                        print(f"WARNING: Failed to load optimizer state ({exc}). Using fresh optimizer.")
                else:
                    print("WARNING: Resume checkpoint has no optimizer state.")

                if scheduler is not None and "scheduler_state_dict" in checkpoint:
                    if optimizer_state_loaded:
                        try:
                            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                            print("Resume: scheduler state loaded")
                        except Exception as exc:
                            print(f"WARNING: Failed to load scheduler state ({exc}).")
                    else:
                        print("Resume: scheduler state skipped because optimizer state was not loaded.")

                if scaler is not None and "scaler_state_dict" in checkpoint:
                    try:
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])
                        print("Resume: GradScaler state loaded")
                    except Exception as exc:
                        print(f"WARNING: Failed to load GradScaler state ({exc}).")

                best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("loss", float("inf"))))
                patience_counter = int(checkpoint.get("patience_counter", 0))
                if estimated_elo is not None:
                    elo_epoch_info = (
                        f" (epoch {estimated_elo_epoch})"
                        if estimated_elo_epoch is not None
                        else ""
                    )
                    print(f"Resume: last known Elo={int(round(float(estimated_elo)))}{elo_epoch_info}")
                print(f"Resume: continuing from epoch {start_epoch + 1}")
            except RuntimeError as exc:
                print(f"WARNING: Full resume failed ({exc})")
                print("Falling back to transfer mode (matching tensors only).")
                transfer_report = transfer_matching_weights(model, normalized_model_state)
                transfer_match_ratio = transfer_report.get("match_ratio")
                _print_transfer_report(transfer_report)
                transfer_trainable_param_names = _infer_changed_parameter_names(model, transfer_report)
                if transfer_trainable_param_names:
                    print(
                        "Transfer warmup candidates "
                        f"(changed params): {len(transfer_trainable_param_names)}"
                    )
                    transfer_freeze_epochs = ask_transfer_freeze_epochs(default_epochs=0)
                    transfer_post_unfreeze_lr = ask_transfer_post_unfreeze_lr(default_lr=transfer_post_unfreeze_lr)
                else:
                    print("Transfer warmup skipped: no changed trainable params detected.")
                start_mode = "transfer"
                start_epoch = 0
                best_val_loss = float("inf")
                patience_counter = 0
        else:
            transfer_report = transfer_matching_weights(model, normalized_model_state)
            transfer_match_ratio = transfer_report.get("match_ratio")
            _print_transfer_report(transfer_report)
            if not transfer_trainable_param_names:
                transfer_trainable_param_names = _infer_changed_parameter_names(model, transfer_report)
                if not transfer_trainable_param_names:
                    print("Transfer warmup skipped: no changed trainable params detected.")

    return {
        "start_mode": start_mode,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_label": selected_checkpoint_label,
        "start_epoch": start_epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "estimated_elo": estimated_elo,
        "estimated_elo_epoch": estimated_elo_epoch,
        "estimated_elo_nn": estimated_elo_nn,
        "estimated_elo_mcts": estimated_elo_mcts,
        "estimated_elo_mcts_simulations": estimated_elo_mcts_simulations,
        "selected_compatibility_ratio": selected_compatibility_ratio,
        "transfer_match_ratio": transfer_match_ratio,
        "transfer_freeze_epochs": transfer_freeze_epochs,
        "transfer_trainable_param_names": transfer_trainable_param_names,
        "transfer_post_unfreeze_lr": transfer_post_unfreeze_lr,
    }


def resolve_il_startup(model, optimizer, scheduler, scaler, device, base_dir, best_model_path, il_dir):
    """Compatibility wrapper: plan + apply startup in one call."""
    plan = plan_il_startup(
        model,
        device,
        base_dir,
        best_model_path,
        il_dir,
        base_learning_rate=None,
    )
    return apply_il_startup_plan(plan, model, optimizer, scheduler, scaler, device)
