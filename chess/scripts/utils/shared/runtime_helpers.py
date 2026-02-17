"""Shared runtime helpers for training scripts."""

import io
from contextlib import redirect_stdout
from pathlib import Path


def build_model_file_tag(config):
    """
    Build sanitized model file tag from config.
    
    Extracts version from config and sanitizes it for use in filenames.
    Preserves dots, underscores, and hyphens (e.g., v5.1, v4_3, v5-beta).
    """
    model_cfg = config.get('model', {})
    raw = model_cfg.get('version') or "model"
    text = str(raw).strip()
    if not text:
        text = "model"
    # Keep dots in version tags (e.g. v5.1), normalize the rest.
    sanitized = "".join(ch if (ch.isalnum() or ch in {".", "_", "-"}) else "_" for ch in text)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    sanitized = sanitized.strip("_")
    return sanitized or "model"


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
        'use_se2d_blocks',
        'drop_path_rate',
        'use_coord_conv',
        'use_layer_scale',
        'layer_scale_init',
        'use_multitask_learning',
        'policy_head_conv_filters',
        'policy_head_conv_groups',
        'policy_head_global_dim',
        'policy_head_hidden_dim',
        'value_head_filters',
        'value_hidden_dim',
    ]
    metadata = {}
    for key in keys:
        if key in model_cfg:
            metadata[key] = model_cfg[key]
    return metadata


def cleanup_interrupted_log_csv(csv_path, plot_path, mode_label):
    """Delete run CSV on interrupt if plot PNG was not generated yet."""
    if csv_path is None:
        return

    csv_file = Path(csv_path)
    plot_file = Path(plot_path) if plot_path is not None else None
    if not csv_file.exists():
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

