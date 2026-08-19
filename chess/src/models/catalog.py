"""Unified model catalog formatting for all scripts.

Provides consistent table formatting for model checkpoints across:
- list_models.py
- eval_elo.py  
- train_il.py startup menu
- train_rl.py startup menu
- play.py model selection
"""

from datetime import datetime
from pathlib import Path
import re
import torch


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_elo_mode(checkpoint):
    settings = checkpoint.get("estimated_elo_settings")
    if isinstance(settings, dict):
        return "mcts" if bool(settings.get("use_mcts", False)) else "nn"
    return None


def _checkpoint_display_elo(entry):
    by_sims = entry.get("elo_mcts_by_simulations") or {}
    if by_sims:
        best = next(iter(sorted(by_sims.items(), key=lambda kv: int(kv[0]), reverse=True)), None)
        if best is not None:
            return (best[1] or {}).get("elo")
    for key in ("elo_mcts", "elo_nn", "elo"):
        value = entry.get(key)
        if value is not None:
            return value
    return None


def _normalize_mcts_elo_by_sims(raw_value):
    if not isinstance(raw_value, dict):
        return {}
    normalized = {}
    for key, item in raw_value.items():
        sims = _safe_int(key)
        if sims is None and isinstance(item, dict):
            sims = _safe_int(item.get("simulations"))
        if sims is None:
            continue
        if isinstance(item, dict):
            elo = _safe_float(item.get("elo", item.get("estimated_elo")))
            timestamp = item.get("timestamp")
        else:
            elo = _safe_float(item)
            timestamp = None
        if elo is None:
            continue
        normalized_entry = {
            "elo": float(elo),
            "simulations": int(sims),
            "timestamp": timestamp,
        }
        if isinstance(item, dict):
            if item.get("source") is not None:
                normalized_entry["source"] = str(item.get("source"))
            if isinstance(item.get("settings"), dict):
                normalized_entry["settings"] = dict(item["settings"])
            standard_error = _safe_float(item.get("se"))
            if standard_error is not None:
                normalized_entry["se"] = float(standard_error)
            ci95 = item.get("ci95")
            if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
                ci_low = _safe_float(ci95[0])
                ci_high = _safe_float(ci95[1])
                if ci_low is not None and ci_high is not None:
                    normalized_entry["ci95"] = [float(ci_low), float(ci_high)]
        normalized[int(sims)] = normalized_entry
    return normalized


def _format_mcts_elo_summary(entry, max_items=3):
    by_sims = entry.get("elo_mcts_by_simulations") or {}
    if by_sims:
        items = sorted(by_sims.items(), key=lambda kv: int(kv[0]), reverse=True)
        parts = [
            f"{int(round(float(info.get('elo'))))} | {int(sims)}"
            for sims, info in items[:max_items]
            if info.get("elo") is not None
        ]
        if len(items) > max_items:
            parts.append("...")
        if parts:
            return ",".join(parts)
    elo_mcts = entry.get("elo_mcts")
    if elo_mcts is None:
        return "n/a"
    sims = entry.get("elo_mcts_simulations")
    if sims is not None:
        return f"{int(round(float(elo_mcts)))} | {int(sims)}"
    return f"{int(round(float(elo_mcts)))}"


def _format_elo_number(value):
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _best_mcts_elo_with_sims(entry):
    by_sims = entry.get("elo_mcts_by_simulations") or {}
    if by_sims:
        for sim_count, info in sorted(by_sims.items(), key=lambda item: int(item[0]), reverse=True):
            if isinstance(info, dict) and info.get("elo") is not None:
                return info.get("elo"), int(sim_count)

    elo_mcts = entry.get("elo_mcts")
    sims = entry.get("elo_mcts_simulations")
    if elo_mcts is not None:
        return elo_mcts, sims
    return None, None


def _set_legacy_nn_elo(checkpoint, elo_value, *, std_error=None, ci_low=None, ci_high=None, epoch=None, source=None, timestamp=None, settings=None):
    """Keep legacy estimated_elo as raw NN only; MCTS lives in estimated_elo_mcts."""
    checkpoint["estimated_elo"] = float(elo_value)
    checkpoint["last_estimated_elo"] = float(elo_value)
    if epoch is not None:
        checkpoint["estimated_elo_epoch"] = int(epoch)
    if std_error is not None:
        checkpoint["estimated_elo_se"] = float(std_error)
    else:
        checkpoint.pop("estimated_elo_se", None)
    if ci_low is not None:
        checkpoint["estimated_elo_ci95_low"] = float(ci_low)
    else:
        checkpoint.pop("estimated_elo_ci95_low", None)
    if ci_high is not None:
        checkpoint["estimated_elo_ci95_high"] = float(ci_high)
    else:
        checkpoint.pop("estimated_elo_ci95_high", None)
    if source is not None:
        checkpoint["estimated_elo_source"] = str(source)
    if timestamp is not None:
        checkpoint["estimated_elo_timestamp"] = timestamp
    if settings is not None:
        checkpoint["estimated_elo_settings"] = settings


def _sync_legacy_elo_from_nn(checkpoint):
    nn_elo = _safe_float(checkpoint.get("estimated_elo_nn", checkpoint.get("last_estimated_elo_nn")))
    if nn_elo is None:
        for key in (
            "estimated_elo", "last_estimated_elo", "estimated_elo_se",
            "estimated_elo_ci95_low", "estimated_elo_ci95_high",
            "estimated_elo_epoch", "estimated_elo_source",
            "estimated_elo_timestamp", "estimated_elo_settings",
        ):
            checkpoint.pop(key, None)
        return

    _set_legacy_nn_elo(
        checkpoint,
        nn_elo,
        std_error=_safe_float(checkpoint.get("estimated_elo_nn_se")),
        ci_low=_safe_float(checkpoint.get("estimated_elo_nn_ci95_low")),
        ci_high=_safe_float(checkpoint.get("estimated_elo_nn_ci95_high")),
        epoch=_safe_int(checkpoint.get("estimated_elo_nn_epoch")),
        source=checkpoint.get("estimated_elo_nn_source"),
        timestamp=checkpoint.get("estimated_elo_nn_timestamp"),
        settings=checkpoint.get("estimated_elo_nn_settings"),
    )


def format_elo_summary(entry, *, legacy_label=True):
    """Format checkpoint Elo as separate NN and MCTS values when available."""
    if not entry or entry.get("error"):
        return "n/a"

    parts = []
    nn_text = _format_elo_number(entry.get("elo_nn"))
    if nn_text is not None:
        parts.append(f"NN {nn_text}")

    mcts_elo, sims = _best_mcts_elo_with_sims(entry)
    mcts_text = _format_elo_number(mcts_elo)
    if mcts_text is not None:
        if sims is not None:
            parts.append(f"MCTS {mcts_text} | {int(sims)}")
        else:
            parts.append(f"MCTS {mcts_text}")

    if parts:
        return " | ".join(parts)

    legacy_text = _format_elo_number(entry.get("elo"))
    if legacy_text is not None:
        return f"Elo {legacy_text}" if legacy_label else legacy_text
    return "n/a"


def format_elo_stat_parts(entry):
    """Return compact (label, value) pairs for UI stat cards."""
    if not entry or entry.get("error"):
        return [("Elo", "n/a")]

    parts = []
    nn_text = _format_elo_number(entry.get("elo_nn"))
    if nn_text is not None:
        parts.append(("NN", nn_text))

    mcts_elo, sims = _best_mcts_elo_with_sims(entry)
    mcts_text = _format_elo_number(mcts_elo)
    if mcts_text is not None:
        value = f"{mcts_text} | {int(sims)}" if sims is not None else mcts_text
        parts.append(("MCTS", value))

    if parts:
        return parts

    legacy_text = _format_elo_number(entry.get("elo"))
    return [("Elo", legacy_text or "n/a")]


def persist_checkpoint_elo_metadata(
    checkpoint_path,
    estimated_elo,
    *,
    levels,
    games_per_level,
    use_mcts,
    simulations,
    sf_time,
    source="eval_elo_manual",
    elo_result=None,
):
    """Persist Elo metadata into a checkpoint, split by raw NN vs MCTS."""
    checkpoint_path = Path(checkpoint_path)
    elo_value = _safe_float(estimated_elo)
    if elo_value is None or not checkpoint_path.exists():
        return False, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"Could not open checkpoint for Elo persist: {exc}"

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return False, "Checkpoint has no model_state_dict"

    mode = "mcts" if bool(use_mcts) else "nn"
    mode_prefix = "estimated_elo_mcts" if mode == "mcts" else "estimated_elo_nn"
    # A checkpoint represents one unchanged network. Keep one current NN Elo
    # and one current MCTS Elo per simulation budget, never estimator history.
    checkpoint.pop("elo_evaluation_history", None)
    for key in list(checkpoint):
        if str(key).startswith("training_estimated_elo_"):
            checkpoint.pop(key, None)
    now = datetime.now().isoformat(timespec="seconds")
    settings = {
        "levels": [int(x) for x in levels],
        "games_per_level": int(games_per_level),
        "use_mcts": bool(use_mcts),
        "simulations": int(simulations) if bool(use_mcts) else 0,
        "stockfish_time_limit": float(sf_time),
    }
    if isinstance(elo_result, dict):
        settings["adaptive"] = bool(elo_result.get("adaptive", False))
        actual_games = elo_result.get("actual_games_per_level")
        if isinstance(actual_games, dict):
            settings["actual_games_per_level"] = {
                int(k): int(v) for k, v in actual_games.items()
            }
        if elo_result.get("elo_std_error") is not None:
            settings["elo_std_error"] = float(elo_result.get("elo_std_error"))
        if elo_result.get("elo_ci95") is not None:
            settings["elo_ci95"] = list(elo_result.get("elo_ci95") or [])
        for key in (
            "rating_bracketed", "rating_bracket", "rating_range_status",
            "rating_censored", "elo_lower_bound", "elo_upper_bound",
            "rating_levels", "rating_games", "probe_only_games",
        ):
            if elo_result.get(key) is not None:
                settings[key] = elo_result.get(key)
    elo_std_error = _safe_float(settings.get("elo_std_error"))
    elo_ci95 = settings.get("elo_ci95")
    ci_low = _safe_float(elo_ci95[0]) if isinstance(elo_ci95, (list, tuple)) and len(elo_ci95) == 2 else None
    ci_high = _safe_float(elo_ci95[1]) if isinstance(elo_ci95, (list, tuple)) and len(elo_ci95) == 2 else None

    checkpoint[mode_prefix] = float(elo_value)
    checkpoint[f"last_{mode_prefix}"] = float(elo_value)
    checkpoint[f"{mode_prefix}_timestamp"] = now
    checkpoint[f"{mode_prefix}_settings"] = settings
    checkpoint[f"{mode_prefix}_source"] = str(source)
    if elo_std_error is not None:
        checkpoint[f"{mode_prefix}_se"] = float(elo_std_error)
    if ci_low is not None:
        checkpoint[f"{mode_prefix}_ci95_low"] = float(ci_low)
    if ci_high is not None:
        checkpoint[f"{mode_prefix}_ci95_high"] = float(ci_high)
    if mode == "mcts":
        checkpoint["estimated_elo_mcts_simulations"] = int(simulations)
        by_sims = _normalize_mcts_elo_by_sims(checkpoint.get("estimated_elo_mcts_by_simulations"))
        sim_entry = {
            "elo": float(elo_value),
            "simulations": int(simulations),
            "timestamp": now,
            "source": str(source),
            "settings": settings,
        }
        if elo_std_error is not None:
            sim_entry["se"] = float(elo_std_error)
        if ci_low is not None and ci_high is not None:
            sim_entry["ci95"] = [float(ci_low), float(ci_high)]
        by_sims[int(simulations)] = sim_entry
        checkpoint["estimated_elo_mcts_by_simulations"] = {
            str(int(k)): v for k, v in sorted(by_sims.items(), key=lambda kv: int(kv[0]))
        }

    epoch_raw = checkpoint.get("epoch")
    epoch_idx = _safe_int(epoch_raw)
    if epoch_idx is not None:
        checkpoint[f"{mode_prefix}_epoch"] = epoch_idx + 1
    mode_epoch = _safe_int(checkpoint.get(f"{mode_prefix}_epoch"))

    if mode == "nn":
        _set_legacy_nn_elo(
            checkpoint,
            elo_value,
            std_error=elo_std_error,
            ci_low=ci_low,
            ci_high=ci_high,
            epoch=mode_epoch,
            source=source,
            timestamp=now,
            settings=settings,
        )
    else:
        _sync_legacy_elo_from_nn(checkpoint)

    try:
        torch.save(checkpoint, checkpoint_path)
    except Exception as exc:
        return False, f"Could not save Elo to checkpoint: {exc}"

    return True, None


def load_checkpoint_metadata(checkpoint_path, base_dir=None):
    """Load metadata from a checkpoint file.
    
    Args:
        checkpoint_path: Path to .pt file
        base_dir: Base directory for relative paths (optional)
    
    Returns:
        dict with metadata: epoch, top1, val_loss, policy_loss, elo, swa, optimizer, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    
    if base_dir is not None:
        base_dir = Path(base_dir)
        try:
            path_rel = str(checkpoint_path.resolve().relative_to(base_dir.resolve()))
        except Exception:
            path_rel = str(checkpoint_path)
    else:
        path_rel = checkpoint_path.name
    
    # Determine folder (root, IL, RL)
    parts = Path(path_rel).parts
    if len(parts) <= 1:
        folder = "root"
    else:
        folder = str(parts[0])
    
    entry = {
        "path": checkpoint_path,
        "path_rel": path_rel,
        "model_name": checkpoint_path.name,
        "folder": folder,
        "size_mb": 0.0,
        "modified": "",
        "mtime_ts": 0.0,
        "epoch": None,
        "top1": None,
        "val_loss": None,
        "policy_loss": None,
        "value_loss": None,
        "early_stop_monitor": None,
        "early_stop_monitor_loss": None,
        "elo": None,
        "elo_nn": None,
        "elo_nn_se": None,
        "elo_nn_ci95": None,
        "elo_nn_settings": None,
        "elo_nn_source": None,
        "elo_mcts": None,
        "elo_mcts_simulations": None,
        "score_rate": None,
        "eval_true_win_rate": None,
        "swa": False,
        "optimizer": False,
        "version": None,
        "architecture_id": None,
        "input_encoder_id": None,
        "policy_codec_id": None,
        "startup_mode": None,
        "error": None,
    }
    
    if not checkpoint_path.exists():
        entry["error"] = "file not found"
        return entry
    
    stat = checkpoint_path.stat()
    entry["size_mb"] = stat.st_size / (1024 ** 2)
    entry["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    entry["mtime_ts"] = stat.st_mtime

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        entry["error"] = f"load failed: {exc}"
        return entry

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        entry["error"] = "invalid checkpoint format"
        return entry

    entry["epoch"] = _safe_int(checkpoint.get("epoch"))
    entry["top1"] = _safe_float(checkpoint.get("val_policy_top1", checkpoint.get("policy_top1_acc")))
    entry["val_loss"] = _safe_float(checkpoint.get("val_loss", checkpoint.get("loss")))
    entry["policy_loss"] = _safe_float(checkpoint.get("val_policy_loss", checkpoint.get("policy_loss")))
    entry["value_loss"] = _safe_float(checkpoint.get("val_value_loss", checkpoint.get("value_loss")))
    entry["early_stop_monitor"] = checkpoint.get("early_stop_monitor")
    entry["early_stop_monitor_loss"] = _safe_float(checkpoint.get("early_stop_monitor_loss"))
    legacy_elo = _safe_float(checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo")))
    entry["elo_nn"] = _safe_float(
        checkpoint.get("estimated_elo_nn", checkpoint.get("last_estimated_elo_nn"))
    )
    entry["elo_nn_se"] = _safe_float(checkpoint.get("estimated_elo_nn_se"))
    nn_ci_low = _safe_float(checkpoint.get("estimated_elo_nn_ci95_low"))
    nn_ci_high = _safe_float(checkpoint.get("estimated_elo_nn_ci95_high"))
    if nn_ci_low is not None and nn_ci_high is not None:
        entry["elo_nn_ci95"] = [float(nn_ci_low), float(nn_ci_high)]
    nn_settings = checkpoint.get("estimated_elo_nn_settings")
    entry["elo_nn_settings"] = dict(nn_settings) if isinstance(nn_settings, dict) else None
    entry["elo_nn_source"] = checkpoint.get("estimated_elo_nn_source")
    entry["elo_mcts"] = _safe_float(
        checkpoint.get("estimated_elo_mcts", checkpoint.get("last_estimated_elo_mcts"))
    )
    entry["elo_mcts_simulations"] = _safe_int(checkpoint.get("estimated_elo_mcts_simulations"))
    entry["elo_mcts_by_simulations"] = _normalize_mcts_elo_by_sims(
        checkpoint.get("estimated_elo_mcts_by_simulations")
    )
    if legacy_elo is not None:
        mode = _checkpoint_elo_mode(checkpoint)
        if mode == "mcts" and entry["elo_mcts"] is None:
            entry["elo_mcts"] = legacy_elo
            if entry["elo_mcts_simulations"] is None:
                settings = checkpoint.get("estimated_elo_settings")
                if isinstance(settings, dict):
                    entry["elo_mcts_simulations"] = _safe_int(settings.get("simulations"))
        elif mode == "nn" and entry["elo_nn"] is None:
            entry["elo_nn"] = legacy_elo
        elif mode is None and entry["elo_nn"] is None and entry["elo_mcts"] is None:
            entry["elo_nn"] = legacy_elo
    if entry["elo_mcts"] is not None and entry["elo_mcts_simulations"] is not None:
        entry["elo_mcts_by_simulations"].setdefault(
            int(entry["elo_mcts_simulations"]),
            {"elo": float(entry["elo_mcts"]), "timestamp": checkpoint.get("estimated_elo_mcts_timestamp")},
        )
    entry["elo"] = legacy_elo if legacy_elo is not None else _checkpoint_display_elo(entry)
    entry["score_rate"] = _safe_float(checkpoint.get("score_rate", checkpoint.get("win_rate")))
    entry["eval_true_win_rate"] = _safe_float(
        checkpoint.get("eval_true_win_rate", checkpoint.get("true_win_rate"))
    )
    entry["swa"] = bool(checkpoint.get("swa_enabled", False) or "swa" in checkpoint_path.name.lower())
    entry["optimizer"] = "optimizer_state_dict" in checkpoint
    model_spec = checkpoint.get("model_spec")
    if isinstance(model_spec, dict):
        entry["architecture_id"] = model_spec.get("architecture_id")
        input_spec = model_spec.get("input_spec") or {}
        policy_spec = model_spec.get("policy_spec") or {}
        entry["input_encoder_id"] = input_spec.get("encoder_id")
        entry["policy_codec_id"] = policy_spec.get("codec_id")
        model_kwargs = model_spec.get("model_kwargs") or {}
    else:
        model_kwargs = checkpoint.get("model_architecture") or {}
        entry["architecture_id"] = model_kwargs.get(
            "architecture_id", "se_cnn_v9"
        )
    entry["version"] = checkpoint.get("version", model_kwargs.get("version"))
    entry["startup_mode"] = checkpoint.get("startup_mode")
    
    return entry


def format_model_table_row(
    idx,
    entry,
    show_folder=False,
    show_version=False,
    show_modified=False,
    show_swa=False,
    show_opt=False,
    show_compat=False,
    show_strict=False,
):
    """Format a single model entry as a table row.
    
    Args:
        idx: Row number (1-indexed)
        entry: Model metadata dict from load_checkpoint_metadata
        show_folder: Include folder column (root/IL/RL)
        show_version: Include version column
        show_modified: Include modified timestamp
        show_swa: Include SWA column
        show_opt: Include optimizer column
        show_compat: Include compatibility ratio column
        show_strict: Include strict-resume compatibility flag
    
    Returns:
        str: Formatted table row
    """
    if entry.get("error"):
        error_msg = entry["error"]
        if show_folder:
            return f"{idx:>3}  {entry.get('folder', 'n/a'):<7} ERROR: {error_msg} - {entry['model_name']}"
        else:
            return f"{idx:>3}  ERROR: {error_msg} - {entry['model_name']}"
    
    # Epoch
    epoch = entry.get("epoch")
    epoch_str = f"{epoch + 1}" if epoch is not None else "n/a"
    
    # Version
    version_str = str(entry.get("version") or "n/a")[:16]
    
    # Top1
    top1 = entry.get("top1")
    top1_str = f"{top1 * 100:6.2f}%" if top1 is not None else "  n/a  "
    
    # Val Loss
    val_loss = entry.get("val_loss")
    loss_str = f"{val_loss:8.4f}" if val_loss is not None else "   n/a  "
    
    # Policy Loss
    policy_loss = entry.get("policy_loss")
    pol_loss_str = f"{policy_loss:8.4f}" if policy_loss is not None else "   n/a  "
    
    # Elo
    elo_nn = entry.get("elo_nn")
    elo_nn_str = f"{int(round(float(elo_nn))):>6}" if elo_nn is not None else "  n/a "
    elo_mcts_str = _format_mcts_elo_summary(entry)[:18]
    
    # SWA
    swa_str = "yes" if entry.get("swa") else "no"
    
    # Optimizer
    opt_str = "yes" if entry.get("optimizer", entry.get("optimizer_present")) else "no"

    # Compatibility / strict resume
    compat_ratio = _safe_float(entry.get("compatibility_ratio"))
    compat_str = f"{compat_ratio * 100:6.2f}%" if compat_ratio is not None else "  n/a "
    strict_str = "yes" if bool(entry.get("strict_resume_ok", False)) else "no"
    
    # Size
    size_str = f"{entry['size_mb']:>6.1f}"
    
    # Modified
    modified_str = entry.get("modified", "")
    
    # Build row
    parts = [f"{idx:>3}"]
    
    if show_folder:
        parts.append(f"{entry.get('folder', 'n/a'):<7}")
    
    if show_version:
        parts.append(f"{version_str:<16}")
    
    parts.extend([
        f"{epoch_str:>6}",
        f"{top1_str:>8}",
        f"{loss_str:>10}",
        f"{pol_loss_str:>10}",
        f"{elo_nn_str:>6}",
        f"{elo_mcts_str:>18}",
    ])
    
    if show_swa:
        parts.append(f"{swa_str:>3}")
    
    if show_opt:
        parts.append(f"{opt_str:>3}")

    if show_compat:
        parts.append(f"{compat_str:>7}")

    if show_strict:
        parts.append(f"{strict_str:>6}")
    
    parts.append(f"{size_str}")
    
    if show_modified:
        parts.append(f"{modified_str:<17}")
    
    parts.append(entry["model_name"])
    
    return "  ".join(parts)


def format_model_table_header(
    show_folder=False,
    show_version=False,
    show_modified=False,
    show_swa=False,
    show_opt=False,
    show_compat=False,
    show_strict=False,
):
    """Format table header.
    
    Args:
        show_folder: Include folder column
        show_version: Include version column  
        show_modified: Include modified timestamp
        show_swa: Include SWA column
        show_opt: Include optimizer column
        show_compat: Include compatibility ratio column
        show_strict: Include strict-resume compatibility flag
    
    Returns:
        tuple: (header_line, separator_line)
    """
    parts = [" ID"]
    sep_parts = ["----"]
    
    if show_folder:
        parts.append(" Folder")
        sep_parts.append("-------")
    
    if show_version:
        parts.append(" Version         ")
        sep_parts.append("----------------")
    
    parts.extend([" Epoch", "  Top1   ", " ValLoss  ", " PolLoss  ", " EloNN ", "  EloMCTS | Sims  "])
    sep_parts.extend(["------", "--------", "----------", "----------", "--------", "------------------"])
    
    if show_swa:
        parts.append("SWA")
        sep_parts.append("----")
    
    if show_opt:
        parts.append("Opt")
        sep_parts.append("----")

    if show_compat:
        parts.append(" Compat ")
        sep_parts.append("--------")

    if show_strict:
        parts.append("Strict")
        sep_parts.append("------")
    
    parts.append(" SizeMB")
    sep_parts.append("-------")
    
    if show_modified:
        parts.append(" Updated         ")
        sep_parts.append("-----------------")
    
    parts.append(" Checkpoint")
    sep_parts.append("-" * 30)
    
    header = "  ".join(parts)
    separator = "  ".join(sep_parts)
    
    return header, separator


def print_model_table(
    entries,
    title="Model Checkpoints",
    show_folder=False,
    show_version=False,
    show_modified=False,
    show_swa=False,
    show_opt=False,
    show_compat=False,
    show_strict=False,
    group_by_folder=False,
):
    """Print formatted model table.
    
    Args:
        entries: List of model metadata dicts
        title: Table title
        show_folder: Include folder column
        show_version: Include version column
        show_modified: Include modified timestamp
        show_swa: Include SWA column
        show_opt: Include optimizer column
        show_compat: Include compatibility ratio column
        show_strict: Include strict-resume compatibility flag
        group_by_folder: Add separator lines between folders
    """
    # Calculate separator width
    header, sep = format_model_table_header(
        show_folder,
        show_version,
        show_modified,
        show_swa,
        show_opt,
        show_compat,
        show_strict,
    )
    width = len(header)
    
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(header)
    print(sep)
    
    last_folder = None
    for idx, entry in enumerate(entries, start=1):
        folder = entry.get("folder", "root")
        
        if group_by_folder and folder != last_folder:
            if last_folder is not None and show_folder:
                print()
            if show_folder:
                print(f"-- {folder} --")
            last_folder = folder
        
        row = format_model_table_row(
            idx,
            entry,
            show_folder,
            show_version,
            show_modified,
            show_swa,
            show_opt,
            show_compat,
            show_strict,
        )
        print(row)
    
    # Summary
    total = len(entries)
    valid = sum(1 for e in entries if not e.get("error"))
    with_elo = sum(
        1
        for e in entries
        if _checkpoint_display_elo(e) is not None and not e.get("error")
    )
    with_opt = sum(
        1
        for e in entries
        if e.get("optimizer", e.get("optimizer_present")) and not e.get("error")
    )
    with_swa = sum(1 for e in entries if e.get("swa") and not e.get("error"))
    
    print("-" * width)
    print(
        f"Total: {total} | Valid: {valid} | With Elo: {with_elo} | "
        f"With optimizer: {with_opt} | SWA-tagged: {with_swa}"
    )
    print("=" * width)


def sort_entries_by_folder_and_elo(entries):
    """Sort entries by folder (root/IL/RL) then by Elo (descending).
    
    Args:
        entries: List of model metadata dicts
    
    Returns:
        Sorted list
    """
    def folder_rank(folder):
        key = str(folder).strip().lower()
        if key == "root":
            return 0
        if key == "il":
            return 1
        if key == "rl":
            return 2
        return 3
    
    return sorted(
        entries,
        key=lambda e: (
            1 if e.get("error") else 0,
            folder_rank(e.get("folder", "")),
            str(e.get("folder", "")).lower(),
            1 if _checkpoint_display_elo(e) is None else 0,
            -float(_checkpoint_display_elo(e) or 0.0),
            -float(e.get("mtime_ts") or 0.0),
            str(e.get("path_rel", "")).lower(),
        ),
    )


def _version_sort_key(version):
    """Return a stable natural-sort key for model version labels."""
    text = str(version or "").strip().lower()
    if not text:
        return ((), "")

    if text.startswith("v"):
        text = text[1:]

    parts = re.findall(r"\d+|[a-zA-Z]+", text)
    numbers = []
    suffix_parts = []
    for part in parts:
        if part.isdigit() and not suffix_parts:
            numbers.append(int(part))
        else:
            suffix_parts.append(part)

    return (tuple(numbers), ".".join(suffix_parts))


def sort_entries_by_folder_and_version(entries):
    """Sort entries by folder, then semantic-ish version descending."""
    def folder_rank(folder):
        key = str(folder).strip().lower()
        if key == "root":
            return 0
        if key == "il":
            return 1
        if key == "rl":
            return 2
        return 3

    def neg_version_tuple(version_tuple):
        padded = list(version_tuple[:6])
        padded.extend([0] * (6 - len(padded)))
        return tuple(-int(part) for part in padded)

    def checkpoint_kind_rank(entry):
        name = str(entry.get("model_name", "")).lower()
        if "best" in name and "swa" not in name:
            return 0
        if "latest" in name and "swa" not in name:
            return 1
        if "swa" in name:
            return 2
        return 3

    def suffix_rank(version):
        suffix = _version_sort_key(version)[1]
        return 0 if suffix else 1

    return sorted(
        entries,
        key=lambda e: (
            1 if e.get("error") else 0,
            folder_rank(e.get("folder", "")),
            str(e.get("folder", "")).lower(),
            neg_version_tuple(_version_sort_key(e.get("version"))[0]),
            suffix_rank(e.get("version")),
            str(_version_sort_key(e.get("version"))[1]),
            checkpoint_kind_rank(e),
            -(int(e.get("epoch")) if e.get("epoch") is not None else -1),
            1 if _checkpoint_display_elo(e) is None else 0,
            -float(_checkpoint_display_elo(e) or 0.0),
            -float(e.get("mtime_ts") or 0.0),
            str(e.get("path_rel", "")).lower(),
        ),
    )


def format_compact_model_info(entry):
    """Format compact one-line model info (for GUI/logs).
    
    Args:
        entry: Model metadata dict
    
    Returns:
        str: Compact formatted string
    """
    epoch = entry.get("epoch")
    epoch_str = f"ep{epoch + 1}" if epoch is not None else "ep?"
    
    top1 = entry.get("top1")
    top1_str = f"{top1 * 100:.1f}%" if top1 is not None else "n/a"
    
    elo_nn = entry.get("elo_nn")
    elo_nn_str = f"{int(round(float(elo_nn)))}" if elo_nn is not None else "n/a"
    elo_mcts_str = _format_mcts_elo_summary(entry, max_items=2)
    
    swa_tag = " [SWA]" if entry.get("swa") else ""
    
    return f"{entry['model_name']} ({epoch_str}, Top1={top1_str}, NN={elo_nn_str}, MCTS={elo_mcts_str}){swa_tag}"
