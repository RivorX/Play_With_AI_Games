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
        "elo": None,
        "swa": False,
        "optimizer": False,
        "version": None,
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
    entry["elo"] = _safe_float(checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo")))
    entry["swa"] = bool(checkpoint.get("swa_enabled", False) or "swa" in checkpoint_path.name.lower())
    entry["optimizer"] = "optimizer_state_dict" in checkpoint
    entry["version"] = checkpoint.get("version")
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
    version_str = str(entry.get("version") or "n/a")[:8]
    
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
    elo = entry.get("elo", entry.get("estimated_elo"))
    elo_str = f"{int(round(float(elo))):>6}" if elo is not None else "  n/a "
    
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
        parts.append(f"{version_str:<8}")
    
    parts.extend([
        f"{epoch_str:>6}",
        f"{top1_str:>8}",
        f"{loss_str:>10}",
        f"{pol_loss_str:>10}",
        f"{elo_str:>6}",
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
        parts.append(" Version")
        sep_parts.append("--------")
    
    parts.extend([" Epoch", "  Top1   ", " ValLoss  ", " PolLoss  ", "   Elo  "])
    sep_parts.extend(["------", "--------", "----------", "----------", "--------"])
    
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
        if e.get("elo", e.get("estimated_elo")) is not None and not e.get("error")
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
            1 if e.get("elo", e.get("estimated_elo")) is None else 0,
            -float(e.get("elo", e.get("estimated_elo")) or 0.0),
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
    
    elo = entry.get("elo")
    elo_str = f"{int(round(float(elo)))}" if elo is not None else "n/a"
    
    swa_tag = " [SWA]" if entry.get("swa") else ""
    
    return f"{entry['model_name']} ({epoch_str}, Top1={top1_str}, Elo={elo_str}){swa_tag}"
