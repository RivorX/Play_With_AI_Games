"""RL startup flow (menu, checkpoint catalog, resume/transfer loading)."""

from pathlib import Path
import sys

from src.model import (
    load_checkpoint_file,
    normalize_state_dict_keys,
    transfer_matching_weights,
)
from src.models.catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_elo,
)


def _print_block_title(title):
    line = "=" * 76
    print(f"\n{line}")
    print(f" {title}")
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


def select_best_il_checkpoint(best_checkpoint, swa_checkpoint=None):
    """Choose SWA only when it beats canonical best on the same IL monitor.

    ``best_model_il.pt`` is selected during IL by the configured validation
    monitor. An SWA file is optional and must carry a directly comparable
    monitor result; its mere existence is not evidence that it is stronger.
    """
    best_path = Path(best_checkpoint)
    swa_path = Path(swa_checkpoint) if swa_checkpoint is not None else (
        best_path.parent / "best_model_il_swa.pt"
    )
    candidates = []
    for kind, path in (("best", best_path), ("swa", swa_path)):
        if not path.exists():
            continue
        metadata = load_checkpoint_metadata(path)
        if metadata.get("error"):
            continue
        candidates.append((kind, path, metadata))

    if not candidates:
        return best_path, "no readable IL checkpoint metadata; using configured best path"

    canonical = next((item for item in candidates if item[0] == "best"), None)
    swa = next((item for item in candidates if item[0] == "swa"), None)
    if canonical is None:
        return swa[1], "canonical best missing; using available SWA checkpoint"
    if swa is None:
        return canonical[1], "SWA checkpoint unavailable; using canonical IL best"

    best_meta = canonical[2]
    swa_meta = swa[2]
    best_monitor = str(best_meta.get("early_stop_monitor") or "").strip()
    swa_monitor = str(swa_meta.get("early_stop_monitor") or "").strip()
    best_loss = _safe_float(best_meta.get("early_stop_monitor_loss"))
    swa_loss = _safe_float(swa_meta.get("early_stop_monitor_loss"))
    if (
        best_loss is not None
        and swa_loss is not None
        and best_monitor
        and best_monitor == swa_monitor
    ):
        if swa_loss < best_loss:
            return swa[1], (
                f"SWA wins IL monitor: {swa_loss:.5f} < {best_loss:.5f}"
            )
        return canonical[1], (
            f"canonical best wins IL monitor: {best_loss:.5f} <= {swa_loss:.5f}"
        )

    return canonical[1], (
        "SWA has no directly comparable IL monitor; using canonical best_model_il.pt"
    )


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


def _load_with_allowed_missing_prefixes(model, checkpoint, allowed_missing_prefixes):
    """Strictly load every shared tensor while retaining new optional heads."""
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise KeyError("missing model_state_dict")
    target_state = model.state_dict()
    normalized = normalize_state_dict_keys(
        model_state,
        target_keys=set(target_state),
    )
    unexpected = [key for key in normalized if key not in target_state]
    shape_mismatch = [
        key
        for key, tensor in normalized.items()
        if key in target_state and tuple(tensor.shape) != tuple(target_state[key].shape)
    ]
    missing = [key for key in target_state if key not in normalized]
    disallowed_missing = [
        key
        for key in missing
        if not any(str(key).startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or shape_mismatch or disallowed_missing:
        raise RuntimeError(
            "checkpoint is not a compatible optional-head upgrade: "
            f"missing={disallowed_missing}, shape_mismatch={shape_mismatch}, "
            f"unexpected={unexpected}"
        )
    model.load_state_dict(normalized, strict=False)
    return missing


def _collect_rl_checkpoints(models_dir, best_model_rl_path, rl_dir):
    """Return only checkpoints that belong to the RL checkpoint namespace.

    IL checkpoints remain valid for NEW/SELECT initialization, but they cannot
    restore an RL optimizer/replay runtime and therefore must not be offered by
    the RESUME picker.
    """
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

    return candidates


def _collect_transfer_checkpoints(models_dir, best_model_rl_path, rl_dir):
    """Return every model checkpoint valid for NEW/SELECT or weight transfer."""
    candidates = _collect_rl_checkpoints(models_dir, best_model_rl_path, rl_dir)
    all_checkpoints = sorted(
        Path(models_dir).rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in all_checkpoints:
        if checkpoint not in candidates:
            candidates.append(checkpoint)
    return candidates


def _suggest_start_mode_default(best_model_rl_path, rl_dir):
    """Prefer a fresh RL run when the latest RL checkpoint is clearly underperforming."""
    rl_candidates = []
    if best_model_rl_path.exists():
        rl_candidates.append(best_model_rl_path)

    for checkpoint in sorted(
        rl_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if checkpoint not in rl_candidates:
            rl_candidates.append(checkpoint)

    for checkpoint_path in rl_candidates:
        entry = load_checkpoint_metadata(checkpoint_path)
        if entry.get("error"):
            continue

        score_rate = _safe_float(entry.get("score_rate"))
        true_win_rate = _safe_float(entry.get("eval_true_win_rate"))
        optimizer_present = bool(entry.get("optimizer", False))

        if not optimizer_present:
            return "1", f"defaulting to new: {checkpoint_path.name} has no optimizer state for a true resume"
        if score_rate is not None and score_rate < 0.50:
            return "1", f"defaulting to new: latest RL score_rate is only {score_rate:.2%}"
        if true_win_rate is not None and true_win_rate <= 0.0:
            return "1", f"defaulting to new: latest RL true win rate is {true_win_rate:.2%}"
        return "2", f"defaulting to resume: latest RL checkpoint looks healthy enough ({checkpoint_path.name})"

    return "1", "defaulting to new: no healthy RL resume candidate found"


def _choose_start_mode(has_checkpoints, default_choice="2", default_hint=None):
    if not has_checkpoints:
        _print_block_title("RL Startup")
        print("No checkpoints found. Starting new RL training.")
        return "new"

    _print_block_title("RL STARTUP | choose how to initialize the learner")
    print(" [1] NEW       fresh RL run; initialize from the best IL checkpoint")
    print(" [2] RESUME    continue model, optimizer, scaler and iteration counter")
    print("               replay/champion buffers rebuild from newly generated games")
    print(" [3] TRANSFER  copy compatible weights; reset optimizer and schedule")

    if default_hint:
        print(f"\n Recommended: [{default_choice}]  {default_hint}")

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
            choice = input(f"\n Start mode [1/2/3, Enter={default_choice}]: ").strip().lower()
        except EOFError:
            choice = default_choice
        if not choice:
            choice = default_choice

        selected = mapping.get(choice)
        if selected is not None:
            return selected
        print("Invalid choice. Enter 1, 2, 3, or press Enter for default.")


def _choose_new_init_mode(has_default_init, has_checkpoints):
    if not has_checkpoints:
        return "default" if has_default_init else "scratch"

    print("\n Initialization source")
    print(" [1] DEFAULT   best IL checkpoint selected by the project")
    if not has_default_init:
        print("               no default checkpoint found; falls back to scratch")
    print(" [2] SELECT    choose another checkpoint manually")
    print(" [3] SCRATCH   random model weights")

    default_choice = "1" if has_default_init else "3"
    mapping = {
        "1": "default",
        "2": "select",
        "3": "scratch",
        "default": "default",
        "select": "select",
        "scratch": "scratch",
    }
    while True:
        try:
            choice = input(f" Init source [1/2/3, Enter={default_choice}]: ").strip().lower()
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


def plan_rl_startup(
    model,
    device,
    models_dir,
    best_model_rl_path,
    rl_dir,
    default_new_checkpoint=None,
    initial_plan=None,
    prefer_fresh_il=False,
):
    """Choose the startup mode first, then finalize it once the model exists."""
    initial_plan = dict(initial_plan or {})
    available_checkpoints = initial_plan.get("available_checkpoints")
    if available_checkpoints is None:
        available_checkpoints = _collect_rl_checkpoints(models_dir, best_model_rl_path, rl_dir)
    selected_checkpoint = None
    checkpoint_catalog = []
    start_mode = initial_plan.get("start_mode")
    new_init_mode = initial_plan.get("new_init_mode", "default")

    if start_mode is None:
        if prefer_fresh_il:
            default_choice = "1"
            default_hint = "RL49 quality experiment: NEW from canonical IL with empty replay"
        else:
            default_choice, default_hint = _suggest_start_mode_default(best_model_rl_path, rl_dir)
        start_mode = _choose_start_mode(
            has_checkpoints=(len(available_checkpoints) > 0),
            default_choice=default_choice,
            default_hint=default_hint,
        )
        if start_mode == "new":
            available_checkpoints = _collect_transfer_checkpoints(
                models_dir,
                best_model_rl_path,
                rl_dir,
            )
            has_default_init = bool(default_new_checkpoint is not None and Path(default_new_checkpoint).exists())
            new_init_mode = _choose_new_init_mode(
                has_default_init=has_default_init,
                has_checkpoints=(len(available_checkpoints) > 0),
            )
        elif start_mode == "transfer":
            available_checkpoints = _collect_transfer_checkpoints(
                models_dir,
                best_model_rl_path,
                rl_dir,
            )

    if model is None:
        return {
            "start_mode": start_mode,
            "new_init_mode": new_init_mode,
            "available_checkpoints": available_checkpoints,
        }

    if start_mode == "new":
        if new_init_mode == "select":
            print("\nScanning checkpoints (metrics + compatibility)...")
            checkpoint_catalog = _build_checkpoint_catalog(available_checkpoints, model, device, models_dir)
            _print_checkpoint_catalog(checkpoint_catalog, "transfer")
            selected_checkpoint = _choose_checkpoint_path(checkpoint_catalog)
            if selected_checkpoint is None:
                print("WARNING: No valid init checkpoint selected. Falling back to default init.")
                new_init_mode = "default"
    elif start_mode in {"resume", "transfer"}:
        print("\nScanning checkpoints (metrics + compatibility)...")
        checkpoint_catalog = _build_checkpoint_catalog(available_checkpoints, model, device, models_dir)
        if start_mode == "resume":
            checkpoint_catalog = [
                entry
                for entry in checkpoint_catalog
                if bool(entry.get("optimizer"))
                and bool(entry.get("strict_resume_ok"))
            ]
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
        "new_init_mode": new_init_mode,
    }


def apply_rl_startup_plan(
    startup_plan,
    model,
    optimizer,
    scaler,
    device,
    default_new_checkpoint=None,
    require_compatible_new_init=False,
):
    """Apply selected RL startup plan (load resume/transfer/new-init checkpoint)."""
    start_mode = startup_plan.get("start_mode", "new")
    selected_checkpoint = startup_plan.get("selected_checkpoint")
    selected_checkpoint_label = startup_plan.get("selected_checkpoint_label")
    selected_entry = startup_plan.get("selected_entry") or {}
    new_init_mode = startup_plan.get("new_init_mode", "default")

    selected_compatibility_ratio = selected_entry.get("compatibility_ratio")
    transfer_match_ratio = None
    compatible_init_upgrade = False
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
                model_state = normalize_state_dict_keys(
                    model_state,
                    target_keys=set(model.state_dict()),
                )
                try:
                    model.load_state_dict(model_state)
                except RuntimeError:
                    report = transfer_matching_weights(model, checkpoint)
                    allowed_missing = (
                        'search_q_fc.',
                    )
                    if report.get('shape_mismatch') or any(
                        not str(key).startswith(allowed_missing)
                        for key in report.get('missing_keys', [])
                    ):
                        raise
                    print(
                        "Resume upgrade: initialized the optional search-Q head."
                    )

                checkpoint_epoch = _safe_int(checkpoint.get("epoch"))
                if checkpoint_epoch is not None:
                    start_iteration = max(0, checkpoint_epoch + 1)

                checkpoint_score_rate = _safe_float(checkpoint.get("score_rate", checkpoint.get("win_rate")))
                if checkpoint_score_rate is not None:
                    best_win_rate = checkpoint_score_rate

                if optimizer is not None:
                    optimizer_state = checkpoint.get("optimizer_state_dict")
                    if isinstance(optimizer_state, dict):
                        try:
                            optimizer.load_state_dict(optimizer_state)
                        except ValueError as exc:
                            print(
                                "Resume warning: optimizer_state_dict is incompatible with the "
                                f"current RL optimizer groups; optimizer reset. ({exc})"
                            )
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
        init_path = None
        if new_init_mode == "select" and selected_checkpoint is not None:
            init_path = Path(selected_checkpoint)
        elif new_init_mode == "default" and default_new_checkpoint is not None:
            init_path = Path(default_new_checkpoint)
        if init_path is not None and init_path.exists():
            print(f"\nInitializing RL from: {init_path}")
            checkpoint = load_checkpoint_file(str(init_path), device)
            model_state = checkpoint.get("model_state_dict")
            try:
                if not isinstance(model_state, dict):
                    raise KeyError("missing model_state_dict")
                model_state = normalize_state_dict_keys(
                    model_state,
                    target_keys=set(model.state_dict()),
                )
                model.load_state_dict(model_state)
                print("Loaded initialization checkpoint (strict).")
            except Exception as exc:
                try:
                    missing = _load_with_allowed_missing_prefixes(
                        model,
                        checkpoint,
                        ("search_q_fc.",),
                    )
                except Exception as upgrade_exc:
                    if require_compatible_new_init:
                        raise RuntimeError(
                            "Controlled IL initialization failed. The checkpoint may "
                            "omit only the optional search_q_fc head."
                        ) from upgrade_exc
                    print(f"Strict init load failed ({exc}). Trying transfer.")
                    report = transfer_matching_weights(model, checkpoint)
                    transfer_match_ratio = report.get("match_ratio")
                    _print_transfer_report(report)
                else:
                    compatible_init_upgrade = True
                    print(
                        "Loaded initialization checkpoint with compatible optional-head "
                        f"upgrade: initialized {', '.join(missing)} from the model seed."
                    )
        else:
            if require_compatible_new_init:
                raise FileNotFoundError(
                    f"Controlled IL initialization checkpoint is missing: {init_path}"
                )
            print("No IL init checkpoint found. Starting RL from random weights.")

    return {
        "start_mode": start_mode,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_label": selected_checkpoint_label,
        "start_iteration": start_iteration,
        "best_win_rate": best_win_rate,
        "selected_compatibility_ratio": selected_compatibility_ratio,
        "transfer_match_ratio": transfer_match_ratio,
        "compatible_init_upgrade": compatible_init_upgrade,
        "new_init_mode": new_init_mode,
    }
