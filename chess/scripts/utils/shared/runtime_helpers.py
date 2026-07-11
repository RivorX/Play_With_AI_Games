"""Shared runtime helpers for training scripts."""

import csv
import io
from contextlib import redirect_stdout
from pathlib import Path


def sanitize_filename_tag(raw, default="tag"):
    """
    Sanitize a short label for safe filename usage.

    Preserves dots, underscores, and hyphens (e.g., v5.1, v4_3, v5-beta).
    """
    text = str(raw).strip()
    if not text:
        text = default
    sanitized = "".join(ch if (ch.isalnum() or ch in {".", "_", "-"}) else "_" for ch in text)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized or default


def build_model_file_tag(config):
    """
    Build sanitized model file tag from config.
    
    Extracts version from config and sanitizes it for use in filenames.
    """
    model_cfg = config.get('model', {})
    raw = model_cfg.get('version') or "model"
    return sanitize_filename_tag(raw, default="model")


def build_rl_experiment_name(config):
    """Build RL log experiment name with model version, RL version, and MCTS sims."""
    model_version = sanitize_filename_tag(
        config.get('model', {}).get('version') or "model",
        default="model",
    )
    rl_version = sanitize_filename_tag(
        config.get('reinforcement_learning', {}).get('version') or "rl",
        default="rl",
    )
    mcts_simulations = int(config.get('reinforcement_learning', {}).get('mcts_simulations', 0))
    return f"rl_training_{model_version}_{rl_version}_mcts_{mcts_simulations}"


def build_model_architecture_metadata(config):
    """
    Extract model architecture metadata from config.
    
    Returns dict with all relevant model hyperparameters for checkpoint saving.
    """
    model_cfg = config.get('model', {})
    keys = [
        'version',
        'filters',
        'num_residual_blocks',
        'dropout',
        'history_positions',
        'use_se_blocks',
        'use_se_bottleneck',
        'se_reduction',
        'drop_path_rate',
        'use_coord_conv',
        'use_layer_scale',
        'layer_scale_init',
        'policy_head_channels',
        'policy_head_conv_filters',
        'policy_head_conv_groups',
        'policy_head_global_dim',
        'policy_head_hidden_dim',
        'value_head_filters',
        'value_hidden_dim',
        'moves_left_hidden_dim',
    ]
    metadata = {}
    for key in keys:
        if key in model_cfg:
            metadata[key] = model_cfg[key]
    return metadata


def cleanup_interrupted_log_csv(csv_path, plot_path, mode_label):
    """Delete interrupted run CSV when it has no data rows or when no plot exists yet."""
    if csv_path is None:
        return

    csv_file = Path(csv_path)
    plot_file = Path(plot_path) if plot_path is not None else None
    if not csv_file.exists():
        return

    # Remove header-only / effectively empty CSV files.
    has_data_rows = False
    try:
        with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header_seen = False
            for row in reader:
                if not row or not any(str(cell).strip() for cell in row):
                    continue
                if str(row[0]).strip() == "# config_json":
                    continue
                if not header_seen:
                    header_seen = True
                    continue
                has_data_rows = True
                break
    except Exception:
        # If CSV cannot be parsed, keep previous behavior below.
        has_data_rows = True

    if not has_data_rows:
        try:
            csv_file.unlink()
            print(f"Removed interrupted {mode_label} log CSV (empty/header-only): {csv_file.name}")
        except Exception as exc:
            print(f"Warning: failed to remove interrupted {mode_label} log CSV ({exc})")
        return

    if plot_file is not None and plot_file.exists():
        return

    try:
        csv_file.unlink()
        print(f"Removed interrupted {mode_label} log CSV (no plot yet): {csv_file.name}")
    except Exception as exc:
        print(f"Warning: failed to remove interrupted {mode_label} log CSV ({exc})")


def run_with_optional_stdout_suppression(enabled, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and optionally silence stdout noise."""
    if enabled:
        return fn(*args, **kwargs)
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)
