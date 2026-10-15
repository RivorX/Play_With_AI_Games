"""
Imitation Learning Training Script - v4.5
🆕 v4.5: CRITICAL FIXES - Per-Game Split + Promotions + Better Sampling
🆕 v4.5: CHESS METADATA - Castling, En Passant, Halfmove, Fullmove (16 planes)
🆕 v4.3: WDL VALUE HEAD + TEMPORAL DISCOUNTING
🆕 v4.2: POV + Dynamic Sliding Window
- 🎯 POV: All boards from current player's perspective (flip for black)
- 🔄 Sliding Window: Dynamic history assembly at load time using mmap
- 🎮 GameID tracking: Efficient history reconstruction across positions
- 📊 Chess Metadata: 4 extra planes (castling, en passant, halfmove, fullmove)
- ⚡ WDL: Win/Draw/Loss classification (stronger signal than MSE)
- ⚡ TEMPORAL DISCOUNTING: DISABLED (WDL-only mode, clean ±1.0 targets)
- 🔒 Per-Game Split: Train/Val separated by games (no history leakage)
- 🎲 Per-Game Stride Offset: Per-game offset for unbiased sampling
- 👑 Promotions: Promotion-aware action space (see ACTION_SIZE)
"""

import torch
import torch.optim as optim
import torch._dynamo
import sys
import time
import math
import shutil
import threading
import logging
import csv
import json
import os
from pathlib import Path
import numpy as np
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import (
    create_model,
    load_model,
    save_checkpoint,
)
from src.models.data import create_dataloaders, process_pgn_files
from src.models.data.se_cnn_v9.helpers import ACTION_SIZE, get_position_size
from src.config import default_config_path, load_project_config

# Import training modules
from src.common.logger import TrainingLogger
from src.common.torch_cache import configure_torch_compile_cache
from src.training.il.trainer import train_epoch_il, evaluate_il
from src.training.il.startup import (
    plan_il_startup,
    apply_il_startup_plan,
    ask_resume_additional_epochs,
    ask_il_target_positions,
    ask_il_hyperparam_source,
    ask_il_start_mode,
    has_il_checkpoints,
)
from src.training.il.auto_tune import resolve_il_hyperparameters
from src.training.il.elo_async import ILEloCoordinator
from src.training.il.checkpointing import (
    build_runtime_state,
    finalize_swa_model,
)
from src.common.runtime import (
    build_model_file_tag,
    build_model_architecture_metadata,
    cleanup_interrupted_log_csv,
)
from src.evaluation.elo_runner import (
    apply_checkpoint_elo_seed,
    build_final_mcts_elo_configs,
    build_il_final_elo_configs,
    build_il_periodic_elo_config,
    persist_estimated_elo,
    run_elo_check,
)
from src.models.view import (
    print_active_model_summary,
    print_status_table,
    print_multi_column_table,
)


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None


def _is_fatal_cuda_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "cuda error: unknown error" in text
        or "cuda error: device-side assert" in text
        or "cuda error: an illegal memory access" in text
        or "cuda error: unspecified launch failure" in text
        or "cudnn_status_not_initialized" in text
    )


def _pct(part, total):
    total = float(total or 0.0)
    if total <= 0.0:
        return 0.0
    return 100.0 * float(part or 0.0) / total


def _format_duration(seconds):
    seconds = float(seconds or 0.0)
    if seconds >= 60.0:
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - minutes * 60:04.1f}s"
    return f"{seconds:.2f}s"


def _target_positions_default_millions(data_cfg, default=15):
    raw = (data_cfg or {}).get('target_positions', int(default * 1_000_000))
    if isinstance(raw, str) and raw.strip().lower() in {"max", "all"}:
        return "max"
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, int(round(value / 1_000_000)))


def _format_target_positions(value):
    if isinstance(value, str) and value.strip().lower() in {"max", "all"}:
        return "max"
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "max"
    if value <= 0:
        return "max"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    return f"{value / 1_000_000:.2f}M"


def _format_plot_million_positions(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "n/a"
    if value <= 0:
        return "0M"
    millions = value / 1_000_000.0
    if millions >= 10.0:
        return f"{millions:.0f}M"
    if millions >= 1.0:
        return f"{millions:.1f}M"
    return f"{value / 1000.0:.0f}k"


def _format_il_data_volume_plan(config, target_positions):
    data_cfg = config.get('data', {}) or {}
    split = _safe_float(data_cfg.get('train_split', 0.85), default=0.85)
    split = min(0.999, max(0.001, split))
    selection_cfg = data_cfg.get('target_selection', {}) or {}
    try:
        pool_multiplier = max(1.0, float(selection_cfg.get('train_pool_multiplier', 1.0) or 1.0))
    except (TypeError, ValueError):
        pool_multiplier = 1.0

    if isinstance(target_positions, str) and target_positions.strip().lower() in {'max', 'all', 'wszystkie'}:
        return (
            "IL data plan: per-epoch samples=max; actual train pool will be shown "
            "after cache/dedup selection."
        )

    try:
        target_total = max(1, int(target_positions))
    except (TypeError, ValueError):
        return None

    train_epoch = max(1, int(round(target_total * split)))
    val_count = max(0, target_total - train_epoch)
    planned_train_pool = int(round(train_epoch * pool_multiplier))
    planned_total_pool = planned_train_pool + val_count
    return (
        "IL data plan: "
        f"per epoch={_format_plot_million_positions(target_total)} "
        f"(train={_format_plot_million_positions(train_epoch)}, "
        f"val={_format_plot_million_positions(val_count)}); "
        f"pool up to={_format_plot_million_positions(planned_total_pool)} "
        f"(train x{pool_multiplier:g}, capped by dedup/cache)."
    )


def _checkpoint_resume_tag(checkpoint_path):
    if checkpoint_path is None:
        return None
    try:
        stem = Path(checkpoint_path).stem
    except TypeError:
        return None
    if stem in {"best_model_il", "best_model", "model"}:
        return None
    for suffix in ("_latest_swa", "_best_swa", "_latest", "_best", "_swa"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or None


def _build_il_run_summary(config, *, model_version=None, start_mode=None, source_label=None,
                          pgn_count=None, train_count=None, val_count=None,
                          train_epoch_count=None, total_epoch_count=None,
                          total_pool_count=None, batch_size=None,
                          soft_train=None, soft_val=None):
    data_cfg = config.get('data', {}) or {}
    model_cfg = config.get('model', {}) or {}
    il_cfg = config.get('imitation_learning', {}) or {}
    ppg_cfg = data_cfg.get('positions_per_game', {}) or {}
    dedup_cfg = data_cfg.get('sample_dedup', {}) or {}
    soft_cfg = data_cfg.get('soft_targets', {}) or {}
    sampler_cfg = data_cfg.get('train_sampling', {}) or {}

    def _clean(value):
        if value is None:
            return None
        try:
            if isinstance(value, (np.floating, float)):
                value = float(value)
                return None if value != value else value
            if isinstance(value, (np.integer, int)):
                return int(value)
        except TypeError:
            pass
        return value

    summary = {
        'version': model_version or model_cfg.get('version'),
        'start_mode': start_mode,
        'source': source_label,
        'target_positions': data_cfg.get('target_positions', 'max'),
        'pgn_files': pgn_count,
        'train_positions': train_count,
        'val_positions': val_count,
        'train_epoch_positions': train_epoch_count,
        'total_epoch_positions': total_epoch_count,
        'total_pool_positions': total_pool_count,
        'batch_size': batch_size or il_cfg.get('batch_size'),
        'history_positions': model_cfg.get('history_positions'),
        'positions_per_game': {
            'enabled': bool(ppg_cfg.get('enabled', False)),
            'mode': ppg_cfg.get('mode'),
            'max_total': ppg_cfg.get('max_total'),
            'min_distance': ppg_cfg.get('min_distance'),
            'auto_relax_min_distance': ppg_cfg.get('auto_relax_min_distance'),
        },
        'sample_dedup': {
            'enabled': bool(dedup_cfg.get('enabled', False)),
            'mode': dedup_cfg.get('mode'),
            'max_count': dedup_cfg.get('max_count'),
            'include_history': dedup_cfg.get('include_history'),
        },
        'soft_targets': {
            'enabled': bool(soft_cfg.get('enabled', False)),
            'mode': soft_cfg.get('mode'),
            'include_turn': soft_cfg.get('include_turn'),
            'include_history': soft_cfg.get('include_history'),
            'max_policy_moves': soft_cfg.get('max_policy_moves'),
            'policy_mass_threshold': soft_cfg.get('policy_mass_threshold'),
        },
        'train_sampling': {
            'enabled': bool(sampler_cfg.get('enabled', False)),
            'mode': sampler_cfg.get('mode'),
            'source': sampler_cfg.get('source'),
            'weight_power': sampler_cfg.get('weight_power'),
            'opening_floor': sampler_cfg.get('opening_floor'),
        },
        'soft_stats': {
            'train': soft_train,
            'val': soft_val,
        },
    }
    return {
        key: _clean(value)
        for key, value in summary.items()
        if value is not None
    }


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _safe_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv_metadata_and_epochs(csv_path):
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8', errors='replace') as f:
            rows = list(csv.reader(f))
    except OSError:
        return {}, []
    metadata = {}
    while rows and rows[0] and str(rows[0][0]).strip().startswith('#'):
        row = rows.pop(0)
        if len(row) >= 2:
            try:
                metadata[str(row[0]).strip()] = json.loads(row[1])
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
    if not rows:
        return metadata, []
    header = rows[0]
    if 'epoch' not in header:
        return metadata, []
    epoch_idx = header.index('epoch')
    epochs = []
    for row in rows[1:]:
        if len(row) <= epoch_idx:
            continue
        epoch = _safe_int(row[epoch_idx])
        if epoch is not None:
            epochs.append(epoch)
    return metadata, epochs


def _find_resume_history_csv(logs_dir, model_version, completed_epoch, current_csv_path=None):
    if completed_epoch < 1:
        return None
    csv_dir = Path(logs_dir) / "csv"
    if not csv_dir.exists():
        return None
    current_csv_path = Path(current_csv_path).resolve() if current_csv_path else None
    candidates = []
    prefix = f"il_training_{model_version}_"
    for path in csv_dir.glob(f"{prefix}*.csv"):
        try:
            if current_csv_path and path.resolve() == current_csv_path:
                continue
        except OSError:
            pass
        metadata, epochs = _read_csv_metadata_and_epochs(path)
        if not epochs or max(epochs) < completed_epoch:
            continue
        summary = metadata.get('# run_summary_json') or {}
        summary_version = str(summary.get('version') or '')
        if summary_version and summary_version != str(model_version):
            continue
        exact_epoch = completed_epoch in set(epochs)
        candidates.append((exact_epoch, max(epochs), path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _format_resume_data_line(label, summary):
    if not summary:
        return f"{label}: data n/a"
    soft_stats = summary.get('soft_stats', {}) or {}
    train_soft = soft_stats.get('train') or {}
    avg_occ = train_soft.get('policy_occ_avg')
    target = summary.get('target_positions', 'n/a')
    pgn_files = summary.get('pgn_files', 'n/a')
    train_positions = summary.get('train_positions', 'n/a')
    val_positions = summary.get('val_positions', 'n/a')
    soft_cfg = summary.get('soft_targets', {}) or {}
    soft_mode = soft_cfg.get('mode', 'off' if not soft_cfg.get('enabled') else 'on')
    try:
        avg_occ_text = f"{float(avg_occ):.2f}"
    except (TypeError, ValueError):
        avg_occ_text = "n/a"
    return (
        f"{label}: PGN={pgn_files}, target={_format_target_positions(target)}, "
        f"train/val={train_positions}/{val_positions}, soft={soft_mode}, avg_occ={avg_occ_text}"
    )


def _il_monitor_loss(val_losses, il_cfg):
    """Return checkpoint/early-stop monitor loss and a short label."""
    mode = str(il_cfg.get('early_stop_monitor', 'total') or 'total').strip().lower()
    if mode not in {'value_aware', 'value-aware', 'value'}:
        return _safe_float(val_losses.get('total'), default=float('inf')), "val_loss"

    value_weight = max(0.0, _safe_float(il_cfg.get('early_stop_value_loss_weight', 1.25), default=1.25))
    mlh_weight = max(0.0, _safe_float(il_cfg.get('early_stop_moves_left_loss_weight', 0.03), default=0.03))
    policy = _safe_float(val_losses.get('policy'), default=0.0)
    value = _safe_float(val_losses.get('value'), default=0.0)
    moves_left = _safe_float(val_losses.get('moves_left'), default=0.0)
    monitor = policy + value_weight * value + mlh_weight * moves_left
    label = f"value_aware(policy + {value_weight:g}*value + {mlh_weight:g}*mlh)"
    return monitor, label


def _resolve_lr_plateau_patience(raw_value, max_patience):
    max_patience = max(1, int(max_patience))
    if raw_value is None or str(raw_value).strip().lower() in {"", "auto"}:
        upper = max(1, max_patience - 1)
        return max(1, min(upper, int(math.ceil(max_patience * 0.6))))
    return max(1, int(raw_value))


def _format_il_epoch_profile(
    epoch_num,
    phase_key,
    train_profile,
    train_time,
    eval_time,
    other_time,
    epoch_total_time,
    train_samples,
    val_samples,
):
    epoch_total_time = max(0.0, float(epoch_total_time or 0.0))
    train_time = max(0.0, float(train_time or 0.0))
    eval_time = max(0.0, float(eval_time or 0.0))
    other_time = max(0.0, float(other_time or 0.0))
    train_profile = train_profile or {}
    train_total = max(0.0, float(train_profile.get('total', train_time) or train_time))
    batches = int(train_profile.get('batches', 0) or 0)

    lines = [
        f"[IL PROFILE] Epoch {epoch_num} [{phase_key}]",
        (
            f"  total={_format_duration(epoch_total_time)} | "
            f"train={_format_duration(train_total)} ({_pct(train_total, epoch_total_time):.1f}%) | "
            f"eval={_format_duration(eval_time)} ({_pct(eval_time, epoch_total_time):.1f}%) | "
            f"other={_format_duration(other_time)} ({_pct(other_time, epoch_total_time):.1f}%)"
        ),
    ]

    if train_samples and train_total > 0.0:
        lines.append(f"  train throughput: {train_samples / train_total:,.0f} pos/s")
    if val_samples and eval_time > 0.0:
        lines.append(f"  eval throughput:  {val_samples / eval_time:,.0f} pos/s")

    header = f"  {'stage':<18} {'time':>10} {'epoch%':>8} {'train%':>8} {'ms/batch':>10}"
    lines.append(header)
    lines.append(f"  {'-' * 18} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10}")

    def add_row(label, seconds, train_part=True):
        seconds = max(0.0, float(seconds or 0.0))
        train_pct = f"{_pct(seconds, train_total):7.1f}%" if train_part and train_total > 0.0 else "       -"
        ms_batch = f"{(seconds * 1000.0 / batches):9.1f}" if train_part and batches > 0 else "        -"
        lines.append(
            f"  {label:<18} {_format_duration(seconds):>10} "
            f"{_pct(seconds, epoch_total_time):7.1f}% {train_pct} {ms_batch}"
        )

    if train_profile:
        add_row("data loading", train_profile.get('data', 0.0))
        add_row("forward", train_profile.get('forward', 0.0))
        add_row("backward", train_profile.get('backward', 0.0))
        add_row("optimizer", train_profile.get('optim', 0.0))
        add_row("train metrics", train_profile.get('metrics', 0.0))
    else:
        add_row("train total", train_time)

    add_row("validation", eval_time, train_part=False)
    add_row("other/log/save", other_time, train_part=False)
    return lines


def _emit_il_profile(lines, debug_log_file=None, print_to_console=False):
    if print_to_console:
        for line in lines:
            print(line)
    if debug_log_file is not None:
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")


def _run_il_final_elo(
    model,
    config,
    device,
    elo_config,
    logger,
    epoch_num,
    model_label,
    marker_label,
    interrupted=False,
    checkpoint_path=None,
):
    """Run a blocking final IL Elo check; Ctrl+C cancels only this check."""
    if not isinstance(elo_config, dict):
        return {"skipped": True}

    timing_label = "after Ctrl+C" if interrupted else "at training end"
    print(f"\nFinal IL Elo check {timing_label} ({model_label}). Press Ctrl+C to cancel this check.")
    elo_config = dict(elo_config or {})
    apply_checkpoint_elo_seed(
        elo_config,
        checkpoint_path,
        use_mcts=bool(elo_config.get("use_mcts", False)),
        simulations=int(elo_config.get("mcts_simulations", 0) or 0),
    )
    stop_event = threading.Event()
    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        elo_result = run_elo_check(
            model,
            config,
            device,
            elo_config,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
        print("\nCtrl+C detected. Final IL Elo check cancelled.")
        return {"cancelled": True}
    finally:
        if was_training:
            model.train()

    if elo_result.get("cancelled"):
        print("Final IL Elo check cancelled.")
        return elo_result

    estimated_elo = elo_result.get("estimated_elo")
    if estimated_elo is not None:
        use_mcts = bool(elo_config.get("use_mcts", False))
        simulations = int(elo_config.get("mcts_simulations", 0) or 0)
        mode_key = "mcts" if use_mcts else "nn"
        logger.record_il_mode_elo(
            epoch_num,
            estimated_elo,
            mode=mode_key,
            simulations=simulations,
            label=marker_label,
            update_csv=True,
            std_error=elo_result.get("elo_std_error"),
            ci95=elo_result.get("elo_ci95"),
        )
        mode_label = f"MCTS {simulations} sims" if use_mcts else "raw NN"
        print(f"Final IL Estimated Elo ({model_label}, {mode_label}): {int(round(float(estimated_elo)))}")
        if elo_result.get("local_rating_level") is not None:
            print(
                f"  local calibration: SF {elo_result['local_rating_level']} "
                f"({int(elo_result.get('local_rating_games', 0) or 0)} games nearest 50%)"
            )
        rating_bracket = elo_result.get("rating_bracket")
        if elo_result.get("rating_bracketed") and isinstance(rating_bracket, list):
            print(f"  rating bracket: SF {rating_bracket[0]}-{rating_bracket[1]} (crosses 50% score)")
        elif elo_result.get("rating_censored"):
            bound = elo_result.get("elo_lower_bound", elo_result.get("elo_upper_bound"))
            relation = ">" if elo_result.get("elo_lower_bound") is not None else "<"
            print(f"  warning: rating is range-censored ({relation}{bound}); no 50% Stockfish bracket")
        if elo_result.get("elo_std_error") is not None:
            ci = elo_result.get("elo_ci95")
            ci_str = f", 95% CI {ci[0]}-{ci[1]}" if isinstance(ci, list) and len(ci) == 2 else ""
            ladder = "adaptive" if elo_result.get("adaptive") else "fixed"
            print(f"  uncertainty: ±{elo_result['elo_std_error']} Elo SE{ci_str} ({ladder} ladder)")
        for lvl, res in sorted(elo_result.get("results", {}).items()):
            score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
            games = int(res.get("games", res["wins"] + res["draws"] + res["losses"]) or 0)
            print(f"  vs SF {lvl}: {score_str} (score: {res['score']:.0%}, n={games})")
        print(f"  time {elo_result['total_time']:.1f}s ({elo_result['total_games']} games)")
        if checkpoint_path is not None:
            persist_estimated_elo(
                checkpoint_path,
                estimated_elo,
                levels=list(elo_result.get("levels_requested", []) or []),
                games_per_level=int(elo_config.get("games_per_level", 0) or 0),
                use_mcts=use_mcts,
                simulations=simulations,
                sf_time=float(elo_config.get("stockfish_time_limit", 0.0) or 0.0),
                source="il_final_elo",
                elo_result=elo_result,
                print_prefix="Final IL Elo metadata: ",
                announce_success=False,
            )
    elif elo_result.get("error"):
        print(f"Final IL Elo check failed: {elo_result.get('error')}")
    elif not elo_result.get("skipped"):
        print("Final IL Elo check: inconclusive")

    return elo_result


def _cleanup_cuda_after_eval(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_il_model_for_final_elo(checkpoint_path, config, device, fallback_model=None):
    """Load a checkpoint for final Elo, falling back to the live model if needed."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if fallback_model is not None:
            print(f"Final IL Elo: checkpoint missing ({checkpoint_path.name}); using current in-memory model.")
            return fallback_model, "current model"
        print(f"Final IL Elo skipped: checkpoint missing ({checkpoint_path})")
        return None, None
    try:
        model = load_model(str(checkpoint_path), config, device, strict=True)
        model.eval()
        return model, checkpoint_path.name
    except Exception as exc:
        if fallback_model is not None:
            print(
                f"Final IL Elo: failed to load {checkpoint_path.name} ({exc}); "
                "using current in-memory model."
            )
            return fallback_model, "current model"
        print(f"Final IL Elo skipped: failed to load {checkpoint_path.name} ({exc})")
        return None, None


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # Load config
    config_path = default_config_path()
    
    print(f"Loading config from: {config_path}")
    config = load_project_config()

    model_version = config.get('model', {}).get('version', 'v?.?')
    model_file_tag = build_model_file_tag(config)
    model_architecture = build_model_architecture_metadata(config)
    debug_cfg = config.get('debug', {}) or {}
    il_debug_cfg = debug_cfg.get('il', {}) or {}
    if not isinstance(il_debug_cfg, dict):
        il_debug_cfg = {}
    debug_enabled = bool(debug_cfg.get('enabled', False))
    if not debug_enabled:
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

    # Default to compact model logging in IL unless debug mode is enabled.
    config.setdefault('model', {})
    # Keep the architecture constructor quiet; startup summary uses the shared table.
    config['model']['print_summary'] = False
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if debug_enabled:
            print("✓ TF32 + cuDNN benchmark enabled")
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Check AMP
    use_amp = config['hardware'].get('use_amp', True)
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    
    if use_amp and not torch.cuda.is_available():
        print("⚠️ AMP requested but CUDA not available, disabling AMP")
        use_amp = False
        use_bfloat16 = False
    
    if use_amp:
        amp_dtype = "bfloat16" if use_bfloat16 else "float16"
        if debug_enabled:
            print(f"✓ Mixed Precision Training (AMP) enabled with {amp_dtype}")
        
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            print("⚠️ bfloat16 requested but not supported, falling back to float16")
            use_bfloat16 = False
            config['hardware']['use_bfloat16'] = False
    
    # Create directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    il_dir = base_dir / config['paths']['il_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    il_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = base_dir / config['paths']['best_model_il']
    version_best_model_path = il_dir / f"{model_file_tag}_best.pt"
    latest_checkpoint_path = il_dir / f"{model_file_tag}_latest.pt"
    
    # Get configuration
    history_positions = config['model'].get('history_positions', 0)
    positions_per_game_cfg = config.get('data', {}).get('positions_per_game', {}) or {}
    sample_dedup_cfg = config.get('data', {}).get('sample_dedup', {}) or {}
    soft_targets_cfg = config.get('data', {}).get('soft_targets', {}) or {}
    
    # 🆕 v4.5 FIXED: Calculate expected input planes with chess metadata
    expected_input_planes = 16 * (1 + history_positions)  # 🔧 FIXED: 16 planes (12 pieces + 4 metadata)
    
    print(
        f"IL setup: version={model_version}, blocks={config['model']['num_residual_blocks']}, "
        f"filters={config['model']['filters']}, history={history_positions}, "
        f"input_planes={expected_input_planes}"
    )
    print(
        f"Features: metadata=on, promotions={ACTION_SIZE}, split_by_game=on, "
        f"wdl_value=on"
    )

    # Setup debug logging
    debug_log_file = None
    
    # Create model early so startup menu appears before any data processing.
    print("\nPreparing model for startup menu...")

    model = create_model(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("Model ready")

    hparam_mode = ask_il_hyperparam_source(default_mode="auto")
    selected_start_mode = ask_il_start_mode(
        has_checkpoints=has_il_checkpoints(best_model_path, il_dir)
    )
    target_positions = ask_il_target_positions(
        default_millions=_target_positions_default_millions(config.get('data', {}), default=15)
    )
    config.setdefault('data', {})['target_positions'] = target_positions
    data_volume_plan = _format_il_data_volume_plan(config, target_positions)
    if data_volume_plan:
        print(data_volume_plan)

    startup_plan = plan_il_startup(
        model=model,
        device=device,
        base_dir=base_dir,
        best_model_path=best_model_path,
        il_dir=il_dir,
        start_mode=selected_start_mode,
        base_learning_rate=config['imitation_learning']['learning_rate'],
    )

    if startup_plan.get("start_mode") == "resume":
        resume_tag = _checkpoint_resume_tag(startup_plan.get("selected_checkpoint"))
        if resume_tag and resume_tag != model_file_tag:
            previous_tag = model_file_tag
            model_version = resume_tag
            model_file_tag = resume_tag
            config.setdefault('model', {})['version'] = resume_tag
            if hasattr(model, 'model_version'):
                model.model_version = resume_tag
            model_architecture = build_model_architecture_metadata(config)
            version_best_model_path = il_dir / f"{model_file_tag}_best.pt"
            latest_checkpoint_path = il_dir / f"{model_file_tag}_latest.pt"
            print(
                "Resume naming: using checkpoint tag "
                f"{model_file_tag} for logs/checkpoints (was {previous_tag} from config)."
            )

    # For resume mode, allow extending training by additional epochs.
    il_epochs_cfg = int(config['imitation_learning']['epochs'])
    if startup_plan.get("start_mode") == "resume":
        selected_entry = startup_plan.get("selected_entry") or {}
        checkpoint_epoch = selected_entry.get("epoch")
        if checkpoint_epoch is not None:
            completed_epochs = int(checkpoint_epoch) + 1
            default_additional = max(0, il_epochs_cfg - completed_epochs)
            additional_epochs = ask_resume_additional_epochs(completed_epochs, default_additional)
            config['imitation_learning']['epochs'] = completed_epochs + additional_epochs
            print(
                f"Resume target: completed={completed_epochs}, "
                f"additional={additional_epochs}, total_target={config['imitation_learning']['epochs']}"
            )

    compile_requested = bool(
        config.get("hardware", {}).get("use_compile", False)
        and device.type == "cuda"
        and torch.cuda.is_available()
    )

    target_runtime_profile = "compiled" if (
        str(hparam_mode).strip().lower() == "auto" and compile_requested
    ) else "eager"
    hparam_resolution = resolve_il_hyperparameters(
        config=config,
        model=model,
        device=device,
        base_dir=base_dir,
        mode=hparam_mode,
        runtime_profile=target_runtime_profile,
        fallback_to_eager_profile=True,
    )

    runtime_profile_used = str(hparam_resolution.get("runtime_profile_used") or target_runtime_profile)
    if target_runtime_profile == "compiled" and runtime_profile_used != "compiled":
        print("IL auto-tune: compile probe unavailable, using eager profile instead.")

    model_hash_short = str(hparam_resolution.get("model_hash") or "n/a")[:16]
    hparam_rows = [
        ("Source", hparam_resolution.get("source", "config")),
        ("Runtime profile", hparam_resolution.get("runtime_profile_used", "eager")),
        ("Batch size", config['imitation_learning']['batch_size']),
        ("Learning rate", f"{float(config['imitation_learning']['learning_rate']):.6g}"),
        ("Model hash", model_hash_short),
    ]

    print_status_table(
        "IL Hyperparameters",
        hparam_rows,
    )

    if debug_enabled:
        debug_dir = logs_dir / "debug" / "training_profile"
        debug_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_file = debug_dir / "il_latest_training_profile.txt"
        for stale_profile in debug_dir.glob("il_*training_profile*.txt"):
            if stale_profile != debug_log_file:
                try:
                    stale_profile.unlink()
                except OSError:
                    pass
        profile_debug_enabled = bool(il_debug_cfg.get('profile_training', debug_cfg.get('profile_training', False)))
        profile_mode_label = "every epoch" if profile_debug_enabled else "first epoch of each training phase"
        print(
            f"Debug mode: profile={profile_debug_enabled}, "
            f"gpu_mem_log={il_debug_cfg.get('log_gpu_memory', debug_cfg.get('log_gpu_memory', False))}, "
            f"timing={profile_mode_label}, "
            f"log={debug_log_file}"
        )

        with open(debug_log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"🐛 TRAINING DEBUG LOG - {model_version} Chess Metadata + WDL\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(
                f"Model: {config['model']['filters']} filters, "
                f"{config['model']['num_residual_blocks']} blocks\n"
            )
            f.write(f"Model hash: {hparam_resolution.get('model_hash', 'n/a')}\n")
            f.write(f"Hyperparameter source: {hparam_resolution.get('source', 'config')}\n")
            f.write(
                f"Runtime profile: {hparam_resolution.get('runtime_profile_used', 'eager')} "
                f"(requested={hparam_resolution.get('runtime_profile_requested', 'eager')})\n"
            )
            f.write(f"Batch size: {config['imitation_learning']['batch_size']}\n")
            f.write(f"Learning rate: {config['imitation_learning']['learning_rate']}\n")
            f.write(f"History positions: {history_positions} (dynamic)\n")
            f.write(f"Positions per game: {positions_per_game_cfg}\n")
            f.write(f"Input planes: {expected_input_planes} (16 per position)\n")
            f.write("Chess metadata: Castling, En Passant, Halfmove, Fullmove\n")
            f.write("WDL Value: True (forced)\n")
            f.write("POV enabled: True\n")
            if hparam_resolution.get("mode") == "auto":
                f.write(
                    f"Auto-tune cache: {hparam_resolution.get('cache_path', 'n/a')} "
                    f"(hit={hparam_resolution.get('cache_hit', False)})\n"
                )
                dedicated_vram = hparam_resolution.get("dedicated_vram_bytes")
                if dedicated_vram:
                    f.write(f"Dedicated VRAM: {dedicated_vram}\n")
                budget_bytes = hparam_resolution.get("budget_bytes")
                if budget_bytes:
                    f.write(f"Target VRAM budget: {budget_bytes}\n")
                estimated_peak = hparam_resolution.get("estimated_peak_bytes")
                if estimated_peak:
                    f.write(f"Estimated train peak: {estimated_peak}\n")
            f.write("=" * 70 + "\n\n")

    # Initialize logger after startup menu selection.
    logger = TrainingLogger(
        logs_dir,
        experiment_name=f"IL_{model_version}",
        mode="il",
        verbose=debug_enabled,
    )
    logging_cfg = config.get('logging', {}) or {}
    logger.set_plot_smoothing(
        enabled=logging_cfg.get('il_plot_smoothing_enabled', True),
        alpha=logging_cfg.get('il_plot_smoothing_alpha', 0.35),
        min_points=logging_cfg.get('il_plot_smoothing_min_points', 5),
    )
    global _LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG
    _LAST_RUN_LOG_CSV = logger.csv_path
    _LAST_RUN_LOG_PNG = logger.plot_path
    
    # Per-head learning rates. Value head can use a multiplier to tune calibration
    # separately from the shared trunk/policy path.
    value_head_lr_factor = config['imitation_learning'].get('value_head_lr_factor', 1.0)
    base_lr = config['imitation_learning']['learning_rate']

    def _build_optimizer_for_model(target_model, *, announce_per_layer_lr=False):
        if value_head_lr_factor != 1.0:
            # Separate value head parameters
            value_head_params = []
            other_params = []

            for name, param in target_model.named_parameters():
                if 'value_' in name or 'moves_left_' in name:
                    value_head_params.append(param)
                else:
                    other_params.append(param)

            param_groups = [
                {'params': other_params, 'lr': base_lr, 'lr_multiplier': 1.0},
                {'params': value_head_params, 'lr': base_lr * value_head_lr_factor, 'lr_multiplier': value_head_lr_factor}
            ]

            if announce_per_layer_lr:
                print(
                    f"Per-layer LR: trunk/policy={base_lr:.4f}, "
                    f"value={base_lr * value_head_lr_factor:.4f} ({value_head_lr_factor}x)"
                )
        else:
            param_groups = [{'params': list(target_model.parameters()), 'lr': base_lr, 'lr_multiplier': 1.0}]

        return optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=config['imitation_learning']['weight_decay'],
            fused=True if torch.cuda.is_available() else False
        )

    def _apply_manual_base_lr(target_optimizer, new_base_lr):
        new_base_lr = float(new_base_lr)
        for group in target_optimizer.param_groups:
            multiplier = float(group.get('lr_multiplier', 1.0) or 1.0)
            group['lr'] = new_base_lr * multiplier
            if 'initial_lr' in group:
                group['initial_lr'] = new_base_lr * multiplier

    def _build_scheduler_for_optimizer(target_optimizer):
        return optim.lr_scheduler.LambdaLR(target_optimizer, lr_lambda=_lr_lambda)

    def _build_grad_scaler():
        # 🔧 v4.8: GradScaler only for float16, NOT for bfloat16 (same dynamic range as float32)
        return torch.amp.GradScaler('cuda', enabled=use_amp and not use_bfloat16)

    optimizer = _build_optimizer_for_model(model, announce_per_layer_lr=True)
    
    if debug_enabled and torch.cuda.is_available():
        print("✓ Using fused AdamW optimizer")
    
    # Learning rate scheduler: linear warmup → cosine decay
    total_epochs = config['imitation_learning']['epochs']
    warmup_pct = config['imitation_learning'].get('warmup_pct', 0.05)
    min_lr_ratio = config['imitation_learning'].get('min_lr_ratio', 0.05)
    lr_plateau_scale = 1.0

    warmup_epochs = max(1, int(round(total_epochs * warmup_pct)))

    def _lr_lambda(epoch):
        # Linear warmup: 0 → 1 over warmup_epochs
        if epoch < warmup_epochs:
            return lr_plateau_scale * ((epoch + 1) / warmup_epochs)
        # Cosine decay: 1 → min_lr_ratio
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_plateau_scale * cosine_factor

    scheduler = _build_scheduler_for_optimizer(optimizer)

    if debug_enabled:
        print(
            f"✓ LR schedule: warmup={warmup_epochs} ep → "
            f"cosine decay (min_lr_ratio={min_lr_ratio})"
        )
    
    # AMP Gradient Scaler
    scaler = _build_grad_scaler()

    il_cfg = config.get('imitation_learning', {})
    save_optimizer_state = il_cfg.get('save_optimizer_state', True)

    startup_state = apply_il_startup_plan(
        startup_plan=startup_plan,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    start_mode = startup_state['start_mode']
    selected_checkpoint = startup_state['selected_checkpoint']
    selected_checkpoint_label = startup_state['selected_checkpoint_label']
    start_epoch = startup_state['start_epoch']
    best_val_loss = startup_state['best_val_loss']
    patience_counter = startup_state['patience_counter']
    resumed_estimated_elo = startup_state.get('estimated_elo')
    resumed_estimated_elo_epoch = startup_state.get('estimated_elo_epoch')
    resumed_estimated_elo_nn = startup_state.get('estimated_elo_nn')
    resumed_estimated_elo_nn_se = startup_state.get('estimated_elo_nn_se')
    resumed_estimated_elo_nn_ci95 = startup_state.get('estimated_elo_nn_ci95')
    resumed_estimated_elo_mcts = startup_state.get('estimated_elo_mcts')
    resumed_estimated_elo_mcts_se = startup_state.get('estimated_elo_mcts_se')
    resumed_estimated_elo_mcts_ci95 = startup_state.get('estimated_elo_mcts_ci95')
    resumed_estimated_elo_mcts_simulations = startup_state.get('estimated_elo_mcts_simulations')
    selected_compatibility_ratio = startup_state.get('selected_compatibility_ratio')
    transfer_match_ratio = startup_state.get('transfer_match_ratio')
    transfer_freeze_epochs = int(startup_state.get('transfer_freeze_epochs', 0) or 0)
    transfer_trainable_param_names = startup_state.get('transfer_trainable_param_names') or []
    transfer_post_unfreeze_lr = startup_state.get('transfer_post_unfreeze_lr')
    transfer_post_unfreeze_lr_active = False
    selected_entry = startup_plan.get("selected_entry") or {}

    source_label = "new (scratch)"
    if start_mode in {"resume", "transfer"}:
        source_label = "selected checkpoint"
        if selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name

    print_active_model_summary(
        model,
        config,
        title="Active Model (IL)",
        source_label=source_label,
        startup_mode=start_mode,
        checkpoint_label=selected_checkpoint_label,
        device=device,
        selected_entry=selected_entry,
        include_parameters=debug_enabled,
    )

    resume_history_summary = None
    resume_history_csv = None
    if start_mode == "resume" and start_epoch > 0:
        resume_history_csv = _find_resume_history_csv(
            logs_dir,
            model_version,
            completed_epoch=start_epoch,
            current_csv_path=logger.csv_path,
        )
        if resume_history_csv is not None:
            imported_history = logger.import_il_history_from_csv(
                resume_history_csv,
                completed_epoch=start_epoch,
            )
            if imported_history:
                resume_history_summary = logger.resume_source_summary
                print(
                    f"Resume CSV history: copied {resume_history_csv.name} "
                    f"through epoch {start_epoch}; new rows will append to {logger.csv_path.name}"
                )
            else:
                print(f"Resume CSV history: found {resume_history_csv.name}, but import failed.")
        else:
            print("Resume CSV history: no matching previous CSV found; plot starts from this resume run.")

    if start_mode in {"resume", "transfer"} and int(start_epoch or 0) > 0:
        logger.add_resume_marker(int(start_epoch))

    if resumed_estimated_elo is not None:
        seed_epoch = resumed_estimated_elo_epoch
        if seed_epoch is None:
            seed_epoch = max(1, start_epoch)
        try:
            logger.record_estimated_elo(int(seed_epoch), float(resumed_estimated_elo), update_csv=False)
            print(f"Resume Elo seed: {int(round(float(resumed_estimated_elo)))} (epoch {int(seed_epoch)})")
        except (TypeError, ValueError):
            pass

    if start_mode in {"resume", "transfer"}:
        resume_marker_epoch = max(1, int(start_epoch or 0))
        if resumed_estimated_elo_nn is None and resumed_estimated_elo is not None:
            resumed_estimated_elo_nn = resumed_estimated_elo
        if resumed_estimated_elo_nn is not None:
            logger.record_il_mode_elo(
                resume_marker_epoch,
                resumed_estimated_elo_nn,
                mode="nn",
                label="Resume NN",
                update_csv=True,
                std_error=resumed_estimated_elo_nn_se,
                ci95=resumed_estimated_elo_nn_ci95,
            )
        if resumed_estimated_elo_mcts is not None:
            logger.record_il_mode_elo(
                resume_marker_epoch,
                resumed_estimated_elo_mcts,
                mode="mcts",
                simulations=resumed_estimated_elo_mcts_simulations or 0,
                label="Resume MCTS",
                update_csv=True,
                std_error=resumed_estimated_elo_mcts_se,
                ci95=resumed_estimated_elo_mcts_ci95,
            )

    if start_mode == "new":
        plot_run_context = "startup: new training from scratch"
    elif start_mode == "resume":
        plot_run_context = f"startup: resumed full state, next epoch {start_epoch + 1}"
    else:
        transfer_ratio = transfer_match_ratio
        if transfer_ratio is None:
            transfer_ratio = selected_compatibility_ratio
        if transfer_ratio is None:
            plot_run_context = "startup: transferred matching weights, compatibility n/a"
        else:
            plot_run_context = (
                "startup: transferred matching weights, "
                f"compatibility {transfer_ratio * 100:.2f}%"
            )
        if transfer_freeze_epochs > 0 and transfer_trainable_param_names:
            plot_run_context += (
                f" | freeze changed-only for {transfer_freeze_epochs} ep"
            )
        if transfer_post_unfreeze_lr is not None:
            plot_run_context += f" | post-unfreeze lr={float(transfer_post_unfreeze_lr):.6g}"

    if selected_checkpoint_label and start_mode in {"resume", "transfer"}:
        plot_run_context = f"{plot_run_context} | source: {Path(selected_checkpoint_label).name}"

    # Prepend model architecture info: version | xM params | N blocks Xf
    _total_params = sum(p.numel() for p in model.parameters())
    _blocks  = config['model'].get('num_residual_blocks', '?')
    _filters = config['model'].get('filters', '?')
    _params_str = f"{_total_params / 1e6:.2f}M"
    model_context = f"Model: {model_version} | {_params_str} params | {_blocks} blocks {_filters}f"
    run_context = f"Run: {plot_run_context}"
    plot_run_context = f"{model_context} | {run_context}"
    logger.set_run_context(plot_run_context)
    logger.set_run_context_lines([model_context, run_context])

    if start_mode == "resume" and start_epoch >= config['imitation_learning']['epochs']:
        print(f"ℹ️ Checkpoint already at epoch {start_epoch}, no epochs left to run.")
        return

    # Load data with multi-phase processing
    data_dir = script_dir.parent / config['paths']['data_dir']
    pgn_files = sorted(data_dir.glob('*.pgn'))

    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_dir}")

    logger.set_run_summary_metadata(_build_il_run_summary(
        config,
        model_version=model_version,
        start_mode=start_mode,
        source_label=source_label,
        pgn_count=len(pgn_files),
        batch_size=config['imitation_learning']['batch_size'],
    ))

    print_status_table(
        "Loading Data (IL)",
        [
            ("Data dir", str(data_dir)),
            ("PGN files", f"{len(pgn_files):,}"),
            ("History positions", history_positions),
            ("Positions/game", positions_per_game_cfg.get('max_total', 'all')),
            ("Target samples", _format_target_positions(config.get('data', {}).get('target_positions', 'max'))),
            ("Debug mode", "on" if debug_enabled else "off"),
        ],
    )
    if debug_enabled:
        for pgn in pgn_files:
            print(f"  - {pgn.name}")

    # Convert relative paths in config to absolute paths to keep preprocessing under chess/data.
    config_with_absolute_paths = config.copy()
    config_with_absolute_paths['paths'] = config['paths'].copy()
    config_with_absolute_paths['paths']['data_dir'] = str(data_dir.absolute())

    metadata = process_pgn_files(pgn_files, config_with_absolute_paths)

    # Verify binary format compatibility.
    actual_position_size = metadata.get('position_size')
    expected_position_size = get_position_size()

    if actual_position_size != expected_position_size:
        print(f"\n⚠️ WARNING: Position size mismatch!")
        print(f"  • Expected: {expected_position_size} bytes")
        print(f"  • Actual: {actual_position_size} bytes")
        print(f"  • This may indicate the data was preprocessed with an old format")
        
        if actual_position_size in [48, 60]:
            print(f"  Data was processed with OLD v4.4 format (36B board, no fullmove metadata)")
            print(f"  {model_version} adds fullmove number (38B board)")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        elif actual_position_size in [44, 56]:
            print(f"  ❌ Data was processed with OLD v4.3 format (32B board, no chess metadata)")
            print(f"     🆕 {model_version} uses 38B board with castling, en passant, halfmove, fullmove")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        else:
            print(f"  ❌ Unknown format mismatch - please delete cache and reprocess")
            raise ValueError("Position size mismatch")

    print("Preparing DataLoaders and train sampler...")
    train_loader, val_loader = create_dataloaders(metadata, config)
    train_sampler = getattr(train_loader, "sampler", None)
    if start_mode == "resume" and hasattr(train_sampler, "epoch"):
        try:
            train_sampler.epoch = int(start_epoch)
            print(f"Resume: train sampler epoch set to {int(start_epoch)}")
        except (TypeError, ValueError):
            pass
    print(
        "DataLoaders ready: "
        f"train_pool={len(train_loader.dataset):,}, "
        f"train_epoch={len(train_sampler) if train_sampler is not None else len(train_loader.dataset):,}, "
        f"val={len(val_loader.dataset):,}"
    )

    def _resolve_loader_batch_size(loader, fallback):
        batch_size = getattr(loader, "batch_size", None)
        if batch_size is None:
            batch_sampler = getattr(loader, "batch_sampler", None)
            batch_size = getattr(batch_sampler, "batch_size", None)
        if batch_size is None:
            batch_size = fallback
        try:
            return max(1, int(batch_size))
        except (TypeError, ValueError):
            return max(1, int(fallback))

    trained_batch_size = _resolve_loader_batch_size(
        train_loader,
        config['imitation_learning']['batch_size'],
    )

    train_stats = getattr(train_loader.dataset, 'filter_stats', {}) or {}
    val_stats = getattr(val_loader.dataset, 'filter_stats', {}) or {}

    def _as_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _stat(stats, key, default=0):
        return _as_int(stats.get(key, default), default=default)

    train_final = _stat(train_stats, 'final_count', len(train_loader.dataset))
    val_final = _stat(val_stats, 'final_count', len(val_loader.dataset))
    train_epoch_samples = _stat(
        train_stats,
        'epoch_sample_count',
        len(getattr(train_loader, 'sampler', []) or train_loader.dataset)
        if getattr(train_loader, 'sampler', None) is not None
        else len(train_loader.dataset),
    )

    def _stage_total(stats, stage_name, default=None):
        stages = stats.get('selection_stages', []) if isinstance(stats, dict) else []
        for entry in stages or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get('stage', '')) == str(stage_name):
                return _as_int(entry.get('total', default or 0), default=default or 0)
        return default

    total_final = train_final + val_final
    total_epoch_samples = train_epoch_samples + val_final
    print(
        "IL data actual: "
        f"per epoch={_format_plot_million_positions(total_epoch_samples)} "
        f"(train={_format_plot_million_positions(train_epoch_samples)}, "
        f"val={_format_plot_million_positions(val_final)}); "
        f"pool total={_format_plot_million_positions(total_final)} "
        f"(train_pool={_format_plot_million_positions(train_final)}, "
        f"val={_format_plot_million_positions(val_final)})."
    )
    raw_total = _stage_total(train_stats, 'raw_split', _stat(train_stats, 'binary_positions', total_final))
    ppg_total = _stage_total(train_stats, 'positions_per_game')
    dedup_total = _stage_total(train_stats, 'sample_dedup')
    data_parts = [
        f"PGN={len(pgn_files)}",
        f"raw={_format_plot_million_positions(raw_total)}",
    ]
    if ppg_total is not None:
        data_parts.append(f"ppg={_format_plot_million_positions(ppg_total)}")
    if dedup_total is not None:
        data_parts.append(f"dedup={_format_plot_million_positions(dedup_total)}")
    data_parts.extend([
        f"train_each_epoch={_format_plot_million_positions(train_epoch_samples)}",
        f"epoch_total={_format_plot_million_positions(total_epoch_samples)}",
        f"pool_total={_format_plot_million_positions(total_final)}",
        f"train_pool/val={_format_plot_million_positions(train_final)}/{_format_plot_million_positions(val_final)}",
    ])
    data_context = (
        "Data: " + " | ".join(data_parts)
    )
    logger.set_run_context(f"{plot_run_context} | {data_context}")
    early_context_lines = [model_context, data_context]
    if resume_history_summary:
        early_context_lines.append(_format_resume_data_line("Before resume", resume_history_summary))
    early_context_lines.append(run_context)
    logger.set_run_context_lines(early_context_lines)

    policy_distribution_cache = {}

    def _policy_distribution_summary(policy_values, chunk_size=500_000):
        if policy_values is None:
            return {
                'policy_moves_avg': 0.0,
                'policy_moves_max': 0,
                'policy_soft_rate': 0.0,
                'policy_target_entropy': 0.0,
                'policy_target_effective_moves': 1.0,
                'policy_target_top1_mass': 1.0,
            }
        row_count = int(policy_values.shape[0]) if getattr(policy_values, 'ndim', 0) >= 1 else 0
        if row_count <= 0:
            return {
                'policy_moves_avg': 0.0,
                'policy_moves_max': 0,
                'policy_soft_rate': 0.0,
                'policy_target_entropy': 0.0,
                'policy_target_effective_moves': 1.0,
                'policy_target_top1_mass': 1.0,
            }
        cache_key = (id(policy_values), tuple(policy_values.shape), str(getattr(policy_values, 'dtype', '')))
        if cache_key in policy_distribution_cache:
            return dict(policy_distribution_cache[cache_key])
        nonzero_sum = 0.0
        nonzero_max = 0
        entropy_sum = 0.0
        top1_sum = 0.0
        soft_rows = 0
        for start in range(0, row_count, int(chunk_size)):
            chunk = np.asarray(policy_values[start:start + int(chunk_size)], dtype=np.float32)
            if chunk.size == 0:
                continue
            nonzero = (chunk > 0.0).sum(axis=1)
            nonzero_sum += float(nonzero.sum())
            nonzero_max = max(nonzero_max, int(nonzero.max()) if nonzero.size else 0)
            safe_chunk = np.clip(chunk.astype(np.float64, copy=False), 1e-12, 1.0)
            row_entropy = -np.sum(chunk * np.log(safe_chunk), axis=1)
            entropy_sum += float(row_entropy.sum())
            soft_rows += int((row_entropy > 1e-4).sum())
            top1_sum += float(chunk.max(axis=1).sum())
        entropy_mean = entropy_sum / max(1, row_count)
        summary = {
            'policy_moves_avg': nonzero_sum / max(1, row_count),
            'policy_moves_max': nonzero_max,
            'policy_soft_rate': soft_rows / max(1, row_count),
            'policy_target_entropy': entropy_mean,
            'policy_target_effective_moves': float(np.exp(entropy_mean)),
            'policy_target_top1_mass': top1_sum / max(1, row_count),
        }
        policy_distribution_cache[cache_key] = dict(summary)
        return summary

    def _soft_data_summary(dataset):
        soft = getattr(dataset, 'soft_targets', None)
        if soft is None:
            return None

        def _array(key):
            return np.asarray(soft[key])

        occurrence = _array('occurrence_count')
        sample_weight = _array('sample_weight')
        policy_mass = _array('policy_mass_kept')
        moves_left_log = _array('moves_left_log')
        policy_values = _array('policy_values')
        policy_summary = _policy_distribution_summary(policy_values)
        return {
            'occ_avg': float(occurrence.mean()) if occurrence.size else 0.0,
            'occ_max': float(occurrence.max()) if occurrence.size else 0.0,
            'weight_avg': float(sample_weight.mean()) if sample_weight.size else 0.0,
            'weight_max': float(sample_weight.max()) if sample_weight.size else 0.0,
            'mass_avg': float(policy_mass.mean()) if policy_mass.size else 1.0,
            'mass_min': float(policy_mass.min()) if policy_mass.size else 1.0,
            'below_995': int((policy_mass < 0.995).sum()) if policy_mass.size else 0,
            'moves_avg': float(np.expm1(moves_left_log).mean()) if moves_left_log.size else 0.0,
            'moves_median': float(np.median(np.expm1(moves_left_log))) if moves_left_log.size else 0.0,
            **policy_summary,
        }

    def _soft_basic_summary(dataset):
        soft = getattr(dataset, 'soft_targets', None)
        if soft is None:
            return None

        def _array(key):
            if key not in soft:
                return None
            return np.asarray(soft[key])

        def _mean(arr):
            return float(arr.mean()) if arr is not None and arr.size else 0.0

        def _max(arr):
            return float(arr.max()) if arr is not None and arr.size else 0.0

        occurrence = _array('occurrence_count')
        value_occurrence = _array('value_occurrence_count')
        sample_weight = _array('sample_weight')
        value_sample_weight = _array('value_sample_weight')
        policy_mass = _array('policy_mass_kept')
        policy_values = _array('policy_values')
        policy_summary = _policy_distribution_summary(policy_values)
        return {
            'rows': int(len(dataset)),
            'policy_occ_avg': _mean(occurrence),
            'policy_occ_max': _max(occurrence),
            'value_occ_avg': _mean(value_occurrence),
            'value_occ_max': _max(value_occurrence),
            'policy_weight_avg': _mean(sample_weight),
            'policy_weight_max': _max(sample_weight),
            'value_weight_avg': _mean(value_sample_weight),
            'value_weight_max': _max(value_sample_weight),
            'policy_mass_kept_avg': _mean(policy_mass),
            'policy_mass_kept_min': float(policy_mass.min()) if policy_mass is not None and policy_mass.size else 1.0,
            'policy_soft_rate': policy_summary['policy_soft_rate'],
            'policy_target_entropy': policy_summary['policy_target_entropy'],
            'policy_target_effective_moves': policy_summary['policy_target_effective_moves'],
            'policy_target_top1_mass': policy_summary['policy_target_top1_mass'],
        }

    train_soft_basic = _soft_basic_summary(train_loader.dataset)
    val_soft_basic = _soft_basic_summary(val_loader.dataset)
    if soft_targets_cfg.get('enabled', False):
        def _soft_float(summary, key, default=0.0):
            if not summary:
                return float(default)
            try:
                return float(summary.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        soft_context = (
            f"Soft compression: {soft_targets_cfg.get('source', 'positions_per_game')}/"
            f"{soft_targets_cfg.get('mode', 'fen')}, "
            f"avg_occ T/V={_soft_float(train_soft_basic, 'policy_occ_avg'):.2f}/"
            f"{_soft_float(val_soft_basic, 'policy_occ_avg'):.2f}, "
            f"soft_rows T/V={100.0 * _soft_float(train_soft_basic, 'policy_soft_rate'):.1f}%/"
            f"{100.0 * _soft_float(val_soft_basic, 'policy_soft_rate'):.1f}%, "
            f"target_eff T/V={_soft_float(train_soft_basic, 'policy_target_effective_moves', 1.0):.2f}/"
            f"{_soft_float(val_soft_basic, 'policy_target_effective_moves', 1.0):.2f}, "
            f"top1_mass T/V={_soft_float(train_soft_basic, 'policy_target_top1_mass', 1.0):.3f}/"
            f"{_soft_float(val_soft_basic, 'policy_target_top1_mass', 1.0):.3f}, "
            f"weight T/V={_soft_float(train_soft_basic, 'policy_weight_avg', 1.0):.2f}/"
            f"{_soft_float(val_soft_basic, 'policy_weight_avg', 1.0):.2f}"
        )
    else:
        soft_context = "Soft: off"
    current_run_summary = _build_il_run_summary(
        config,
        model_version=model_version,
        start_mode=start_mode,
        source_label=source_label,
        pgn_count=len(pgn_files),
        train_count=train_final,
        val_count=val_final,
        train_epoch_count=train_epoch_samples,
        total_epoch_count=total_epoch_samples,
        total_pool_count=total_final,
        batch_size=trained_batch_size,
        soft_train=train_soft_basic,
        soft_val=val_soft_basic,
    )
    resume_data_context = None
    if resume_history_summary:
        resume_data_context = (
            "Resume data: "
            f"{_format_resume_data_line('before', resume_history_summary)} -> "
            f"{_format_resume_data_line('after', current_run_summary)}"
        )
    context_lines = [model_context, data_context, soft_context]
    if resume_data_context:
        context_lines.append(resume_data_context)
    context_lines.append(run_context)
    logger.set_run_context(" | ".join(context_lines))
    logger.set_run_context_lines(context_lines)
    logger.set_run_summary_metadata(current_run_summary)

    if debug_enabled:
        train_soft_summary = _soft_data_summary(train_loader.dataset)
        val_soft_summary = _soft_data_summary(val_loader.dataset)
        if train_soft_summary is not None or val_soft_summary is not None:
            def _sf(summary, key, fmt="{:.3f}"):
                if summary is None:
                    return ""
                value = summary.get(key, 0.0)
                return fmt.format(value)

            def _pct(summary, key):
                if summary is None:
                    return ""
                try:
                    return f"{100.0 * float(summary.get(key, 0.0)):.1f}%"
                except (TypeError, ValueError):
                    return ""

            print_multi_column_table(
                "Data Ready (IL) - Soft Targets",
                headers=["Metric", "Train", "Val"],
                rows=[
                    ("occurrence_count avg", _sf(train_soft_summary, 'occ_avg'), _sf(val_soft_summary, 'occ_avg')),
                    ("occurrence_count max", _sf(train_soft_summary, 'occ_max', "{:.0f}"), _sf(val_soft_summary, 'occ_max', "{:.0f}")),
                    ("sample_weight avg", _sf(train_soft_summary, 'weight_avg'), _sf(val_soft_summary, 'weight_avg')),
                    ("sample_weight max", _sf(train_soft_summary, 'weight_max'), _sf(val_soft_summary, 'weight_max')),
                    ("policy moves avg/max", f"{_sf(train_soft_summary, 'policy_moves_avg')} / {_sf(train_soft_summary, 'policy_moves_max', '{:.0f}')}", f"{_sf(val_soft_summary, 'policy_moves_avg')} / {_sf(val_soft_summary, 'policy_moves_max', '{:.0f}')}"),
                    ("policy soft rows", _pct(train_soft_summary, 'policy_soft_rate'), _pct(val_soft_summary, 'policy_soft_rate')),
                    ("target effective moves", _sf(train_soft_summary, 'policy_target_effective_moves'), _sf(val_soft_summary, 'policy_target_effective_moves')),
                    ("target top1 mass", _sf(train_soft_summary, 'policy_target_top1_mass', "{:.4f}"), _sf(val_soft_summary, 'policy_target_top1_mass', "{:.4f}")),
                    ("top-K mass avg", _sf(train_soft_summary, 'mass_avg', "{:.5f}"), _sf(val_soft_summary, 'mass_avg', "{:.5f}")),
                    ("top-K mass min", _sf(train_soft_summary, 'mass_min', "{:.5f}"), _sf(val_soft_summary, 'mass_min', "{:.5f}")),
                    ("rows below 0.995 mass", _sf(train_soft_summary, 'below_995', "{:.0f}"), _sf(val_soft_summary, 'below_995', "{:.0f}")),
                    ("MLH plies avg", _sf(train_soft_summary, 'moves_avg'), _sf(val_soft_summary, 'moves_avg')),
                    ("MLH plies median", _sf(train_soft_summary, 'moves_median'), _sf(val_soft_summary, 'moves_median')),
                ],
            )

    # Stochastic Weight Averaging (SWA) for better generalization with large batches
    use_swa = config['imitation_learning'].get('use_swa', False)
    il_eval_every = max(1, int(config['imitation_learning'].get('eval_every', 1) or 1))
    swa_model = None
    swa_scheduler = None
    swa_auto_start = bool(config['imitation_learning'].get('swa_auto_start', True))
    swa_start = total_epochs + 1 if swa_auto_start else int(config['imitation_learning'].get('swa_start_epoch', total_epochs + 1))
    swa_auto_min_epoch_config = int(config['imitation_learning'].get('swa_auto_min_epoch', swa_start))
    swa_auto_min_epoch_config = max(1, swa_auto_min_epoch_config)
    swa_auto_latest_start_fraction = float(
        config['imitation_learning'].get('swa_auto_latest_start_fraction', 0.75)
    )
    swa_auto_latest_start_fraction = max(0.1, min(1.0, swa_auto_latest_start_fraction))
    swa_auto_plateau_patience = int(config['imitation_learning'].get('swa_auto_plateau_patience', 3))
    swa_auto_plateau_patience = max(1, swa_auto_plateau_patience)
    swa_start_on_early_stop = bool(config['imitation_learning'].get('swa_start_on_early_stop', True))
    swa_auto_min_delta = float(config['imitation_learning'].get('swa_auto_min_delta', min_delta if 'min_delta' in locals() else 0.001))
    swa_auto_min_delta = max(0.0, swa_auto_min_delta)
    swa_transfer_unfreeze_offset = int(
        config['imitation_learning'].get('swa_transfer_unfreeze_offset', 1)
    )
    swa_transfer_unfreeze_offset = max(0, swa_transfer_unfreeze_offset)
    swa_anneal_epochs = max(1, int(config['imitation_learning'].get('swa_anneal_epochs', 10)))
    swa_stop_after_anneal_no_improve = bool(
        config['imitation_learning'].get('swa_stop_after_anneal_no_improve', True)
    )
    swa_stop_after_anneal_grace_evals = max(
        1,
        int(config['imitation_learning'].get('swa_stop_after_anneal_grace_evals', 1)),
    )
    swa_min_updates_before_stop = max(
        1,
        int(
            config['imitation_learning'].get(
                'swa_min_updates_before_stop',
                min(8, swa_anneal_epochs),
            )
        ),
    )
    remaining_epochs_this_run = max(1, int(total_epochs) - int(start_epoch or 0))
    swa_min_updates_this_run = max(1, min(swa_min_updates_before_stop, remaining_epochs_this_run))
    current_run_first_epoch = int(start_epoch or 0) + 1
    current_run_fraction_start_epoch = int(start_epoch or 0) + max(
        1,
        int(math.ceil(remaining_epochs_this_run * swa_auto_latest_start_fraction)),
    )
    swa_auto_latest_start_epoch = current_run_fraction_start_epoch
    swa_auto_update_window_start_epoch = max(
        current_run_first_epoch,
        int(total_epochs) - swa_min_updates_this_run + 1,
    )
    swa_auto_latest_start_epoch = min(
        swa_auto_latest_start_epoch,
        swa_auto_update_window_start_epoch,
    )
    swa_auto_min_epoch = min(
        max(swa_auto_min_epoch_config, current_run_first_epoch),
        swa_auto_latest_start_epoch,
    )
    swa_quality_gate = bool(config['imitation_learning'].get('swa_quality_gate', True))
    swa_update_max_loss_delta = max(
        0.0,
        float(config['imitation_learning'].get('swa_update_max_loss_delta', 0.008)),
    )
    swa_start_reason = "config"
    swa_best_val_loss = None
    swa_plateau_counter = 0
    swa_post_anneal_no_improve_evals = 0
    swa_skipped_updates = 0
    non_blocking_transfers = bool(
        config.get('hardware', {}).get('non_blocking_transfers', True)
        and device.type == 'cuda'
    )

    if use_swa and swa_auto_start and start_mode == "transfer":
        first_full_unfrozen_epoch = start_epoch + transfer_freeze_epochs + 1
        auto_swa_start = max(1, first_full_unfrozen_epoch + swa_transfer_unfreeze_offset)
        if auto_swa_start < swa_start:
            swa_start = auto_swa_start
            swa_start_reason = (
                "auto-transfer "
                f"(first_full_epoch={first_full_unfrozen_epoch}, offset={swa_transfer_unfreeze_offset})"
            )

    def _resolve_swa_lr(start_epoch_for_lr=None):
        il_cfg_local = config['imitation_learning']
        swa_lr_mode = str(il_cfg_local.get('swa_lr_mode', 'auto')).strip().lower()
        manual_swa_lr = float(il_cfg_local.get('swa_lr', 0.0005))
        if swa_lr_mode not in {"auto", "manual"}:
            swa_lr_mode = "auto"

        if swa_lr_mode == "manual":
            return manual_swa_lr, f"manual(config={manual_swa_lr:.6f})"

        effective_base_lr = float(transfer_post_unfreeze_lr or base_lr)
        resolved_start_epoch = int(start_epoch_for_lr or swa_start)
        swa_epoch_idx = max(0, min(total_epochs - 1, resolved_start_epoch - 1))
        lr_factor_at_swa = float(_lr_lambda(swa_epoch_idx))
        effective_lr_at_swa = effective_base_lr * lr_factor_at_swa

        ratio = float(il_cfg_local.get('swa_lr_ratio', 0.12))
        ratio = max(0.01, min(0.50, ratio))

        compat_ratio = transfer_match_ratio
        if compat_ratio is None:
            compat_ratio = selected_compatibility_ratio
        compat_scale = 1.0
        if start_mode == "transfer" and compat_ratio is not None:
            compat_ratio = float(compat_ratio)
            if compat_ratio < 0.70:
                compat_scale = 0.70
            elif compat_ratio < 0.80:
                compat_scale = 0.85
            elif compat_ratio >= 0.90:
                compat_scale = 1.10

        transfer_scale = 0.90 if (start_mode == "transfer" and transfer_freeze_epochs > 0) else 1.0
        target_swa_lr = effective_lr_at_swa * ratio * compat_scale * transfer_scale

        swa_lr_min = float(il_cfg_local.get('swa_lr_min', 3e-5))
        swa_lr_max = float(il_cfg_local.get('swa_lr_max', 2e-4))
        if swa_lr_min > swa_lr_max:
            swa_lr_min, swa_lr_max = swa_lr_max, swa_lr_min
        target_swa_lr = max(swa_lr_min, min(swa_lr_max, target_swa_lr))

        source = (
            f"auto(base={effective_base_lr:.6g}, lr_at_swa={effective_lr_at_swa:.6g}, "
            f"ratio={ratio:.3f}, compat_scale={compat_scale:.2f}, transfer_scale={transfer_scale:.2f})"
        )
        return target_swa_lr, source

    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr, swa_lr_reason = _resolve_swa_lr(swa_start)
        if debug_enabled:
            print(
                f"SWA enabled: planned_start={'auto' if swa_auto_start else swa_start}, auto_start={'on' if swa_auto_start else 'off'}, "
                f"auto_min_epoch={swa_auto_min_epoch}, plateau_patience={swa_auto_plateau_patience}, "
                f"latest_start={swa_auto_latest_start_epoch}, "
                f"min_updates={swa_min_updates_this_run}/{swa_min_updates_before_stop}, "
                f"anneal_epochs={swa_anneal_epochs}, planned_swa_lr={swa_lr:.6f}, "
                f"stop_after_anneal_no_improve={'on' if swa_stop_after_anneal_no_improve else 'off'}, "
                f"source={swa_start_reason}, lr_source={swa_lr_reason}"
            )
    else:
        swa_lr = float(config['imitation_learning'].get('swa_lr', 0.0005))
        swa_lr_reason = "disabled"

    def _activate_swa_if_due(current_epoch_num, reason):
        nonlocal swa_scheduler, swa_lr, swa_lr_reason, swa_start, swa_start_reason
        if not use_swa or swa_model is None or swa_scheduler is not None:
            return False
        if int(current_epoch_num) < int(swa_start):
            return False

        swa_start = int(current_epoch_num)
        swa_start_reason = str(reason)
        swa_lr, swa_lr_reason = _resolve_swa_lr(swa_start)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            swa_lr=swa_lr,
            anneal_epochs=swa_anneal_epochs,
        )
        print(
            f"SWA activated at epoch {swa_start}: "
            f"swa_lr={swa_lr:.6f}, anneal_epochs={swa_anneal_epochs}, "
            f"quality_gate={'on' if swa_quality_gate else 'off'}"
            f"{f'(+{swa_update_max_loss_delta:g})' if swa_quality_gate else ''}, "
            f"source={swa_start_reason}, lr_source={swa_lr_reason}"
        )
        if debug_log_file is not None:
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"SWA activated at epoch {swa_start}: "
                    f"swa_lr={swa_lr:.6f}, anneal_epochs={swa_anneal_epochs}, "
                    f"quality_gate={'on' if swa_quality_gate else 'off'}, "
                    f"source={swa_start_reason}, lr_source={swa_lr_reason}\n"
        )
        return True

    def _swa_updates_collected():
        if swa_model is None:
            return 0
        n_averaged = getattr(swa_model, "n_averaged", None)
        if n_averaged is None:
            return 0
        try:
            if hasattr(n_averaged, "item"):
                return int(n_averaged.item())
            return int(n_averaged)
        except (TypeError, ValueError):
            return 0

    def _maybe_update_swa_after_epoch(current_epoch_num, current_monitor_loss=None, current_monitor_label=None):
        nonlocal swa_skipped_updates
        if not use_swa or swa_model is None or swa_scheduler is None:
            return False

        should_average = True
        skip_reason = None
        monitor_value = None
        if swa_quality_gate:
            try:
                monitor_value = float(current_monitor_loss)
            except (TypeError, ValueError):
                monitor_value = None
            if monitor_value is None or not math.isfinite(monitor_value):
                should_average = False
                skip_reason = "no validation monitor"
            else:
                reference_loss = best_val_loss
                if not math.isfinite(float(reference_loss)):
                    reference_loss = monitor_value
                max_allowed_loss = float(reference_loss) + float(swa_update_max_loss_delta)
                if monitor_value > max_allowed_loss:
                    should_average = False
                    label = current_monitor_label or monitor_label
                    skip_reason = (
                        f"{label}={monitor_value:.4f} > "
                        f"best+gate={max_allowed_loss:.4f}"
                    )

        if should_average:
            swa_model.update_parameters(model)
            print(
                f"       SWA update accepted "
                f"({_swa_updates_collected()} averaged, {swa_skipped_updates} skipped)"
            )
        else:
            swa_skipped_updates += 1
            print(f"       SWA update skipped by quality gate: {skip_reason}")
            if debug_log_file is not None:
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(
                        f"SWA update skipped at epoch {current_epoch_num}: "
                        f"{skip_reason}\n"
                    )

        swa_scheduler.step()
        return should_average

    def _maybe_decay_lr_on_plateau(current_epoch_num, improved):
        nonlocal lr_plateau_scale, lr_plateau_bad_epochs, lr_plateau_cooldown_remaining
        nonlocal patience_counter
        if not lr_plateau_enabled:
            return False
        if use_swa and swa_scheduler is not None:
            return False
        if transfer_post_unfreeze_lr_active:
            return False
        if improved:
            lr_plateau_bad_epochs = 0
            if lr_plateau_cooldown_remaining > 0:
                lr_plateau_cooldown_remaining -= 1
            return False
        if lr_plateau_cooldown_remaining > 0:
            lr_plateau_cooldown_remaining -= 1
            return False

        lr_plateau_bad_epochs += 1
        if lr_plateau_bad_epochs < lr_plateau_patience:
            return False

        old_scale = float(lr_plateau_scale)
        new_scale = max(float(lr_plateau_min_scale), old_scale * float(lr_plateau_factor))
        if new_scale >= old_scale - 1e-12:
            return False

        lr_ratio = new_scale / old_scale
        for group in optimizer.param_groups:
            group['lr'] = float(group.get('lr', 0.0)) * lr_ratio
        lr_plateau_scale = new_scale
        lr_plateau_bad_epochs = 0
        lr_plateau_cooldown_remaining = lr_plateau_cooldown_epochs
        if lr_plateau_reset_patience:
            patience_counter = 0
        print(
            "LR plateau decay: "
            f"scale {old_scale:.3f} -> {new_scale:.3f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"patience={'reset' if lr_plateau_reset_patience else 'kept'}"
        )
        if debug_log_file is not None:
            with open(debug_log_file, 'a', encoding='utf-8') as f:
                f.write(
                    f"LR plateau decay at epoch {current_epoch_num}: "
                    f"scale {old_scale:.6g}->{new_scale:.6g}, "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}\n"
                )
        return True

    max_patience = config['imitation_learning'].get('max_patience', 15)
    min_delta = config['imitation_learning']['min_delta']
    total_epochs = config['imitation_learning']['epochs']
    monitor_label = _il_monitor_loss({'total': 0.0}, config['imitation_learning'])[1]
    lr_plateau_enabled = bool(config['imitation_learning'].get('lr_plateau_decay', True))
    lr_plateau_patience = _resolve_lr_plateau_patience(
        config['imitation_learning'].get('lr_plateau_patience', 'auto'),
        max_patience,
    )
    lr_plateau_factor = float(config['imitation_learning'].get('lr_plateau_factor', 0.7))
    lr_plateau_factor = max(0.05, min(0.95, lr_plateau_factor))
    lr_plateau_min_scale = float(config['imitation_learning'].get('lr_plateau_min_scale', 0.35))
    lr_plateau_min_scale = max(0.01, min(1.0, lr_plateau_min_scale))
    lr_plateau_cooldown_epochs = max(0, int(config['imitation_learning'].get('lr_plateau_cooldown', 1)))
    lr_plateau_reset_patience = bool(config['imitation_learning'].get('lr_plateau_reset_patience', True))
    lr_plateau_bad_epochs = 0
    lr_plateau_cooldown_remaining = 0
    if monitor_label != "val_loss" and start_mode in {"resume", "transfer"}:
        # Older checkpoints stored raw val_loss as best_val_loss. A value-aware
        # monitor is on a different scale, so restart monitor patience safely.
        best_val_loss = float("inf")
        patience_counter = 0
    best_raw_val_loss_for_swa = float("inf")

    # Training loop
    print("\nStarting training...")
    start_msg = f"mode={start_mode}, epochs={total_epochs}, batch={config['imitation_learning']['batch_size']}"
    if selected_checkpoint is not None and start_mode in {"resume", "transfer"}:
        start_msg += f", checkpoint={selected_checkpoint_label}"
    print(start_msg)
    if debug_enabled:
        print(f"Non-blocking transfers: {'enabled' if non_blocking_transfers else 'disabled'}")
        print(
            f"optimizer_state={'on' if save_optimizer_state else 'off'}, "
            f"history={history_positions}, input_planes={expected_input_planes}"
        )
        print("Debug profiling is active")
    profile_enabled = bool(debug_enabled and il_debug_cfg.get('profile_training', debug_cfg.get('profile_training', False)))
    profiled_phase_keys = set()

    def _profile_phase_key(epoch_idx):
        if start_mode == "transfer" and transfer_freeze_epochs > 0:
            return "transfer_frozen" if epoch_idx < transfer_freeze_until_epoch else "transfer_unfrozen"
        return "default"

    if debug_enabled:
        print(f"early_stopping(monitor={monitor_label}, patience={max_patience}, min_delta={min_delta})")
    if start_mode == "transfer" and transfer_post_unfreeze_lr is not None:
        print(f"transfer post-unfreeze lr={float(transfer_post_unfreeze_lr):.6g} (manual override)")
    if debug_enabled:
        print(
            f"paths: best={best_model_path}, version_best={version_best_model_path}, "
            f"latest={latest_checkpoint_path}, checkpoints={il_dir}"
        )

    transfer_freeze_active = False
    transfer_freeze_until_epoch = start_epoch + transfer_freeze_epochs
    if start_mode == "transfer" and transfer_freeze_epochs > 0:
        if transfer_trainable_param_names:
            transfer_trainable_set = set(transfer_trainable_param_names)
            trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
            for param_name, param in model.named_parameters():
                param.requires_grad = param_name in transfer_trainable_set
            trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
            transfer_freeze_active = True
            print(
                "Transfer warmup: frozen trunk active for "
                f"first {transfer_freeze_epochs} epoch(s)."
            )
            print(
                f"  trainable params during warmup: {trainable_after:,}/{trainable_before:,} "
                f"across {len(transfer_trainable_set)} tensors"
            )
        else:
            print(
                "Transfer warmup requested but no changed trainable tensors were detected; "
                "running without freeze."
            )
    elo_config = config.get("elo_estimation", {})
    elo_config_il = build_il_periodic_elo_config(elo_config)
    if elo_config_il.get("use_mcts", False):
        print("Note: IL forces elo_estimation.use_mcts=False for speed.")
    elo_config_il["use_mcts"] = False
    final_elo_config_il, final_elo_config_il_mcts = build_il_final_elo_configs(elo_config)
    final_elo_config_il_mcts_profile = build_final_mcts_elo_configs(elo_config)
    final_il_elo_enabled = True

    elo_coordinator = ILEloCoordinator(
        model=model,
        config=config,
        device=device,
        elo_config_il=elo_config_il,
        logger=logger,
    )
    elo_coordinator.print_startup_summary(verbose=debug_enabled)
    if debug_enabled and final_il_elo_enabled:
        print(
            "Elo final (IL): exact best_model_il Raw NN + MCTS, plus SWA checks, "
            "levels=auto from Stockfish UCI_Elo, "
            f"cap={final_elo_config_il_mcts.get('adaptive_max_total_games')} games, "
            f"time={float(final_elo_config_il_mcts.get('stockfish_time_limit', 0.0)):.2f}s, "
            "mcts_profile="
            + "/".join(
                str(int(cfg.get("mcts_simulations", 0) or 0))
                for cfg in final_elo_config_il_mcts_profile
            )
        )

    # torch.compile: fuses Conv+BN+ReLU kernels → fewer GPU kernel launches.
    # For transfer warmup we compile the frozen model first, then re-compile the
    # fully unfrozen eager model once warmup ends.
    _use_compile = config['hardware'].get('use_compile', False)
    eager_model = model
    _compile_strategy = None

    def _compile_warmup_batch_sizes():
        sizes = []
        for loader in (train_loader, val_loader):
            batch_size = getattr(loader, "batch_size", None)
            try:
                batch_size = int(batch_size)
            except (TypeError, ValueError):
                batch_size = 0
            if batch_size > 0 and batch_size not in sizes:
                sizes.append(batch_size)
        if not sizes:
            sizes.append(1)
        return sizes

    def _maybe_compile_model(current_model, reason_label="startup"):
        if not _use_compile:
            return current_model
        if not torch.cuda.is_available():
            print("⚠️ torch.compile pominięty: CUDA niedostępna")
            return current_model

        configure_torch_compile_cache(torch, device, "il_learner")

        # Important for transfer warmup → unfreeze.
        # Without resetting Dynamo, a new compile call may still reuse guards/
        # cached graphs specialized for the previously frozen parameter set.
        torch._dynamo.reset()

        amp_enabled_for_compile = bool(use_amp and device.type == 'cuda')
        amp_dtype_for_compile = torch.bfloat16 if use_bfloat16 else torch.float16

        # torch.compile jest leniwy — kompilacja i ewentualny błąd dopiero przy pierwszym forward.
        # Robimy próbny forward pass zaraz po compile, żeby złapać błąd TERAZ (nie w środku epoki).
        def _try_compile(backend_or_mode, is_mode=True):
            """Zwraca skompilowany model lub None jeśli się nie uda."""
            try:
                if is_mode:
                    try:
                        _compiled = torch.compile(current_model, mode=backend_or_mode, dynamic=True)
                    except TypeError:
                        _compiled = torch.compile(current_model, mode=backend_or_mode)
                else:
                    try:
                        _compiled = torch.compile(current_model, backend=backend_or_mode, dynamic=True)
                    except TypeError:
                        _compiled = torch.compile(current_model, backend=backend_or_mode)
                # Compile/warm up with the real train/eval batch sizes. This
                # avoids paying the first large-batch graph/autotune cost inside
                # the epoch after compiling only a tiny batch=1 probe.
                with torch.no_grad():
                    with torch.amp.autocast(
                        "cuda",
                        enabled=amp_enabled_for_compile,
                        dtype=amp_dtype_for_compile,
                    ):
                        for warmup_batch_size in _compile_warmup_batch_sizes():
                            _dummy = torch.zeros(
                                warmup_batch_size, expected_input_planes, 8, 8,
                                device=device,
                                dtype=torch.float32,
                            ).to(memory_format=torch.channels_last)
                            _compiled(_dummy, apply_log_softmax=False)
                return _compiled
            except Exception as _e:
                if debug_enabled:
                    print(f"  ✗ {'mode=' + backend_or_mode if is_mode else 'backend=' + backend_or_mode}: {type(_e).__name__}: {_e}")
                return None

        nonlocal _compile_strategy
        preferred_strategy = _compile_strategy
        if preferred_strategy is not None:
            preferred_label = (
                f"mode={preferred_strategy['name']}"
                if preferred_strategy['is_mode']
                else f"backend={preferred_strategy['name']}"
            )
            if debug_enabled:
                print(f"torch.compile: używam zapamiętanej konfiguracji ({preferred_label}, {reason_label})...")
            _compiled_model = _try_compile(
                preferred_strategy['name'],
                is_mode=preferred_strategy['is_mode'],
            )
            if _compiled_model is not None:
                if debug_enabled:
                    if preferred_strategy['is_mode']:
                        print(f"✓ torch.compile enabled (mode={preferred_strategy['name']}, cached choice)")
                    else:
                        print(f"✓ torch.compile enabled (backend={preferred_strategy['name']}, cached choice)")
                return _compiled_model
            if debug_enabled:
                print("  ⚠️ Zapamiętana konfiguracja torch.compile nie powiodła się, fallback do pełnego testu.")

        warmup_sizes = ", ".join(str(size) for size in _compile_warmup_batch_sizes())
        if debug_enabled:
            print(f"torch.compile: testowanie dostępnych backendów ({reason_label}, warmup_batches=[{warmup_sizes}])...")
        _compiled_model = None

        # 1. Inductor default (wymaga Triton — najlepszy wynik)
        _compiled_model = _try_compile('default', is_mode=True)
        if _compiled_model is not None:
            _compile_strategy = {'name': 'default', 'is_mode': True}
            if debug_enabled:
                print("✓ torch.compile enabled (mode=default, inductor+Triton, warmup ~30-60s)")
            return _compiled_model

        # 2. cudagraphs (nie wymaga Triton, ~10-15% gain, stabilny na Windows)
        _compiled_model = _try_compile('cudagraphs', is_mode=False)
        if _compiled_model is not None:
            _compile_strategy = {'name': 'cudagraphs', 'is_mode': False}
            if debug_enabled:
                print("✓ torch.compile enabled (backend=cudagraphs, ~10-15% gain, bez Triton)")
            return _compiled_model

        print("⚠️ torch.compile niedostępny dla bieżącej konfiguracji — trening bez kompilacji")
        return current_model

    model = _maybe_compile_model(model)

    def _get_elo_state_for_checkpoint():
        elo_epoch, elo_value = logger.get_latest_estimated_elo_with_epoch()
        elo_metadata = {}
        if elo_value is not None:
            elo_metadata['estimated_elo'] = float(elo_value)
            elo_metadata['last_estimated_elo'] = float(elo_value)
            elo_metadata['estimated_elo_nn'] = float(elo_value)
            elo_metadata['last_estimated_elo_nn'] = float(elo_value)
        if elo_epoch is not None:
            elo_metadata['estimated_elo_epoch'] = int(elo_epoch)
            elo_metadata['estimated_elo_nn_epoch'] = int(elo_epoch)
        return elo_metadata, elo_epoch, elo_value

    training_interrupted = False
    last_epoch_idx = start_epoch - 1
    completed_epochs_this_run = 0
    last_val_losses = None
    last_val_metrics = None

    try:
        for epoch in range(start_epoch, total_epochs):
            last_epoch_idx = epoch
            val_losses = last_val_losses
            val_metrics = last_val_metrics
            should_stop = False
            if transfer_freeze_active and epoch >= transfer_freeze_until_epoch:
                for param in model.parameters():
                    param.requires_grad = True
                transfer_freeze_active = False
                total_trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(
                    f"Transfer warmup finished at epoch {epoch + 1}. "
                    f"Full model unfrozen ({total_trainable_now:,} trainable params)."
                )
                if _use_compile and torch.cuda.is_available():
                    print(
                        "  torch.compile: świeża instancja pełnego modelu + rekompilacja po odmrożeniu — "
                        "to może potrwać chwilę przed startem epoki."
                    )
                    eager_state = {
                        key: tensor.detach().cpu().clone()
                        for key, tensor in eager_model.state_dict().items()
                    }
                    optimizer_state = optimizer.state_dict()
                    scheduler_state = scheduler.state_dict()
                    scaler_state = scaler.state_dict()
                    swa_model_state = swa_model.state_dict() if swa_model is not None else None
                    swa_scheduler_state = swa_scheduler.state_dict() if swa_scheduler is not None else None

                    new_eager_model = create_model(config).to(device)
                    new_eager_model = new_eager_model.to(memory_format=torch.channels_last)
                    new_eager_model.load_state_dict(eager_state, strict=True)
                    for param in new_eager_model.parameters():
                        param.requires_grad = True

                    eager_model = new_eager_model
                    model = eager_model
                    optimizer = _build_optimizer_for_model(model)
                    optimizer.load_state_dict(optimizer_state)
                    scheduler = _build_scheduler_for_optimizer(optimizer)
                    scheduler.load_state_dict(scheduler_state)
                    scaler = _build_grad_scaler()
                    if scaler_state:
                        scaler.load_state_dict(scaler_state)

                    if use_swa:
                        swa_model = torch.optim.swa_utils.AveragedModel(model)
                        if swa_model_state is not None:
                            swa_model.load_state_dict(swa_model_state)
                        if swa_scheduler_state is not None:
                            swa_scheduler = torch.optim.swa_utils.SWALR(
                                optimizer,
                                swa_lr=swa_lr,
                                anneal_epochs=swa_anneal_epochs,
                            )
                            swa_scheduler.load_state_dict(swa_scheduler_state)
                        else:
                            swa_scheduler = None

                    elo_coordinator.model = model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    model = _maybe_compile_model(model, reason_label=f"epoch {epoch + 1} unfreeze")
                    elo_coordinator.model = model
                if transfer_post_unfreeze_lr is not None:
                    _apply_manual_base_lr(optimizer, transfer_post_unfreeze_lr)
                    transfer_post_unfreeze_lr_active = True
                    print(
                        "  Transfer: applied manual LR after unfreeze: "
                        f"base={float(transfer_post_unfreeze_lr):.6g}"
                    )
            elo_coordinator.poll_results()
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            epoch_lr = optimizer.param_groups[0]['lr']

            phase_key = _profile_phase_key(epoch)
            if profile_enabled:
                profile_this_epoch = True
            else:
                profile_this_epoch = phase_key not in profiled_phase_keys
                if profile_this_epoch:
                    profiled_phase_keys.add(phase_key)
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            epoch_start_time = time.perf_counter()

            # Train epoch
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            train_start_time = time.perf_counter()
            if (
                use_swa
                and swa_auto_start
                and swa_scheduler is None
                and (epoch + 1) >= int(swa_auto_latest_start_epoch)
            ):
                swa_start = epoch + 1
                swa_start_reason = (
                    "latest-start fallback "
                    f"(remaining_run_epochs={remaining_epochs_this_run}, "
                    f"min_updates={swa_min_updates_this_run}/{swa_min_updates_before_stop})"
                )
            if use_swa and swa_scheduler is None and (epoch + 1) >= int(swa_start):
                _activate_swa_if_due(epoch + 1, swa_start_reason)
            use_swa_scheduler_this_epoch = bool(
                use_swa and swa_scheduler is not None
            )
            train_losses, train_metrics, train_profile = train_epoch_il(
                model,
                train_loader,
                optimizer,
                scheduler,
                config,
                device,
                scaler,
                epoch=epoch,
                debug_log_file=debug_log_file,
                profile=profile_this_epoch,
                step_scheduler=(not use_swa_scheduler_this_epoch) and (not transfer_post_unfreeze_lr_active),
                non_blocking_transfer=non_blocking_transfers,
                progress_callback=elo_coordinator.poll_results,
            )
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start_time
            if profile_this_epoch and train_profile is not None:
                train_time = train_profile['total']

            # Print train losses and metrics
            print(f"Train - Loss: {train_losses['total']:.4f}, "
                  f"Policy: {train_losses['policy']:.4f}, "
                  f"Value: {train_losses['value']:.4f}, "
                  f"LR: {epoch_lr:.2e}")

            print(f"       📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                  f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                  f"MAE: {train_metrics['value_mae']:.4f}")


            # Evaluate
            if (epoch + 1) % il_eval_every == 0:
                if profile_this_epoch and device.type == 'cuda':
                    torch.cuda.synchronize()
                eval_start_time = time.perf_counter()
                val_losses, val_metrics = evaluate_il(
                    model,
                    val_loader,
                    config,
                    device,
                    non_blocking_transfer=non_blocking_transfers,
                )
                if profile_this_epoch and device.type == 'cuda':
                    torch.cuda.synchronize()
                eval_time = time.perf_counter() - eval_start_time
                _cleanup_cuda_after_eval(device)

                print(f"Val - Loss: {val_losses['total']:.4f}, "
                      f"Policy: {val_losses['policy']:.4f}, "
                      f"Value: {val_losses['value']:.4f}")

                print(f"     📊 Top-1: {val_metrics['policy_top1_acc']:.2%}, "
                      f"Top-3: {val_metrics['policy_top3_acc']:.2%}, "
                      f"MAE: {val_metrics['value_mae']:.4f}")
                last_val_losses = val_losses
                last_val_metrics = val_metrics


                # Log metrics + periodic Elo estimation.
                estimated_elo = elo_coordinator.evaluate_if_due(epoch + 1)
                logger.log(epoch + 1, train_losses, val_losses, train_metrics, val_metrics, epoch_lr,
                           estimated_elo=estimated_elo)
                completed_epochs_this_run += 1
                logger.plot()
                elo_coordinator.poll_results()

                # Save best model
                current_monitor_loss, current_monitor_label = _il_monitor_loss(
                    val_losses,
                    config['imitation_learning'],
                )
                improvement = best_val_loss - current_monitor_loss
                improved_this_epoch = improvement > min_delta
                if improved_this_epoch:
                    best_val_loss = current_monitor_loss
                    best_raw_val_loss_for_swa = float(val_losses['total'])
                    patience_counter = 0

                    print(
                        f"✓ New best model! {current_monitor_label}: {current_monitor_loss:.4f} "
                        f"(improved by {improvement:.4f}, val_loss={val_losses['total']:.4f})"
                    )
                    print(f"  📊 Val Top-1: {val_metrics['policy_top1_acc']:.2%}")

                    model_to_save = model
                    elo_metadata, elo_epoch_for_state, elo_for_state = _get_elo_state_for_checkpoint()
                    best_metadata = {
                        'val_policy_loss': val_losses['policy'],
                        'val_value_loss': val_losses['value'],
                        'early_stop_monitor': current_monitor_label,
                        'early_stop_monitor_loss': current_monitor_loss,
                        'val_policy_top1': val_metrics['policy_top1_acc'],
                        'val_policy_top3': val_metrics['policy_top3_acc'],
                        'val_value_mae': val_metrics['value_mae'],
                        'use_amp': use_amp,
                        'use_bfloat16': use_bfloat16,
                        'history_positions': history_positions,
                        'input_planes': expected_input_planes,
                        'training_batch_size': trained_batch_size,
                        'positions_per_game': dict(positions_per_game_cfg),
                        'sample_dedup': dict(sample_dedup_cfg),
                        'soft_targets': dict(soft_targets_cfg),
                        'pov_enabled': True,  # 🆕 v4.2
                        'version': model_version,    # 🆕 Track version
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                    }
                    best_metadata.update(elo_metadata)
                    saved_best_path = Path(save_checkpoint(
                        model_to_save,
                        optimizer,
                        epoch,
                        val_losses['total'],
                        str(best_model_path),
                        best_metadata,
                        save_optimizer=False,
                        save_dtype=torch.bfloat16 if use_bfloat16 else None,
                    ))

                    print(f"  💾 Saved to: {saved_best_path}")
                    size_mb = saved_best_path.stat().st_size / (1024**2)
                    print(f"  📦 Model size: {size_mb:.1f} MB (without optimizer)")
                    if version_best_model_path != best_model_path:
                        shutil.copy2(saved_best_path, version_best_model_path)
                        version_best_size_mb = version_best_model_path.stat().st_size / (1024 ** 2)
                        print(
                            f"  💾 Version best updated: {version_best_model_path.name} "
                            f"({version_best_size_mb:.1f} MB)"
                        )
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{max_patience}")

                _maybe_decay_lr_on_plateau(epoch + 1, improved_this_epoch)

                if use_swa_scheduler_this_epoch:
                    _maybe_update_swa_after_epoch(
                        epoch + 1,
                        current_monitor_loss=current_monitor_loss,
                        current_monitor_label=current_monitor_label,
                    )

                if use_swa and swa_stop_after_anneal_no_improve and swa_scheduler is not None:
                    swa_anneal_complete_epoch = int(swa_start) + int(swa_anneal_epochs) - 1
                    if (epoch + 1) >= swa_anneal_complete_epoch:
                        if improvement > min_delta:
                            swa_post_anneal_no_improve_evals = 0
                        else:
                            swa_post_anneal_no_improve_evals += 1
                            print(
                                "SWA anneal complete and validation did not improve: "
                                f"{swa_post_anneal_no_improve_evals}/"
                                f"{swa_stop_after_anneal_grace_evals} post-anneal eval(s)."
                            )
                            if swa_post_anneal_no_improve_evals >= swa_stop_after_anneal_grace_evals:
                                print("\nSWA post-anneal stop triggered.")
                                print(f"Best monitor loss: {best_val_loss:.4f} ({monitor_label})")
                                should_stop = True

                if use_swa and swa_auto_start and swa_scheduler is None:
                    current_val_loss = float(val_losses['total'])
                    if swa_best_val_loss is None or (swa_best_val_loss - current_val_loss) > swa_auto_min_delta:
                        swa_best_val_loss = current_val_loss
                        swa_plateau_counter = 0
                    else:
                        swa_plateau_counter += 1

                    if (epoch + 1) >= swa_auto_min_epoch and swa_plateau_counter >= swa_auto_plateau_patience:
                        next_epoch = min(total_epochs, epoch + 2)
                        if next_epoch < swa_start:
                            swa_start = next_epoch
                            swa_start_reason = (
                                f"auto-plateau(best={swa_best_val_loss:.4f}, "
                                f"patience={swa_plateau_counter}/{swa_auto_plateau_patience}, "
                                f"min_delta={swa_auto_min_delta:g})"
                            )
                            print(f"SWA auto-start scheduled for epoch {swa_start}: {swa_start_reason}")
                            if debug_log_file is not None:
                                with open(debug_log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"SWA auto-start scheduled for epoch {swa_start}: {swa_start_reason}\n")

                # Early stopping
                if patience_counter >= max_patience:
                    if (
                        use_swa
                        and swa_auto_start
                        and swa_start_on_early_stop
                        and swa_scheduler is None
                    ):
                        swa_start = epoch + 1
                        started_swa = _activate_swa_if_due(
                            epoch + 1,
                            (
                                "early-stop fallback "
                                f"(patience={patience_counter}/{max_patience}, "
                                f"configured_min={swa_auto_min_epoch_config}, "
                                f"resolved_min={swa_auto_min_epoch})"
                            ),
                        )
                        if started_swa:
                            _maybe_update_swa_after_epoch(
                                epoch + 1,
                                current_monitor_loss=current_monitor_loss,
                                current_monitor_label=current_monitor_label,
                            )

                    swa_updates = _swa_updates_collected()
                    if (
                        use_swa
                        and swa_scheduler is not None
                        and swa_updates < swa_min_updates_this_run
                    ):
                        print(
                            "\nEarly stopping delayed for SWA: "
                            f"collected {swa_updates}/{swa_min_updates_this_run} "
                            "averaged checkpoints."
                        )
                        print(f"Best monitor loss: {best_val_loss:.4f} ({monitor_label})")
                    else:
                        print(f"\n🛑 Early stopping triggered!")
                        print(f"Best monitor loss: {best_val_loss:.4f} ({monitor_label})")
                        should_stop = True
            else:
                # Log only training metrics
                logger.log(
                    epoch + 1,
                    train_losses,
                    None,
                    train_metrics,
                    None,
                    epoch_lr,
                    estimated_elo=None,
                )
                completed_epochs_this_run += 1
                elo_coordinator.poll_results()
                if use_swa_scheduler_this_epoch:
                    _maybe_update_swa_after_epoch(epoch + 1)
                eval_time = 0.0

            if profile_this_epoch:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                epoch_total_time = time.perf_counter() - epoch_start_time
                other_time = max(0.0, epoch_total_time - train_time - eval_time)
                profile_lines = _format_il_epoch_profile(
                    epoch + 1,
                    phase_key,
                    train_profile,
                    train_time,
                    eval_time,
                    other_time,
                    epoch_total_time,
                    len(train_loader.dataset),
                    len(val_loader.dataset) if eval_time > 0.0 else 0,
                )
                _emit_il_profile(
                    profile_lines,
                    debug_log_file=debug_log_file,
                    print_to_console=bool(il_debug_cfg.get('print_profile_to_console', False)),
                )
                if debug_enabled and train_profile is not None:
                    grad_diag = train_profile.get('grad_diag') or {}
                    if grad_diag:
                        grad_msg = (
                            f"[DEBUG] Epoch {epoch + 1} batch#{grad_diag.get('batch_idx', 0) + 1}: "
                            f"grad_tensors={grad_diag.get('grad_tensors', 0)}/{grad_diag.get('trainable_tensors', 0)}, "
                            f"grad_params={grad_diag.get('grad_params', 0):,}/{grad_diag.get('trainable_params', 0):,}"
                        )
                        missing_grad_names = grad_diag.get('missing_grad_names') or []
                        if missing_grad_names:
                            grad_msg += f", missing_sample={missing_grad_names[:4]}"
                        print(grad_msg)

                    if il_debug_cfg.get('log_gpu_memory', debug_cfg.get('log_gpu_memory', False)):
                        mem_msg = (
                            f"[DEBUG] Epoch {epoch + 1} GPU: "
                            f"peak_alloc={train_profile.get('peak_allocated_mb', 0.0):.1f}MB, "
                            f"peak_reserved={train_profile.get('peak_reserved_mb', 0.0):.1f}MB, "
                            f"curr_alloc={train_profile.get('current_allocated_mb', 0.0):.1f}MB, "
                            f"curr_reserved={train_profile.get('current_reserved_mb', 0.0):.1f}MB"
                        )
                        print(mem_msg)

                    if debug_log_file is not None:
                        with open(debug_log_file, 'a', encoding='utf-8') as f:
                            grad_diag = train_profile.get('grad_diag') or {}
                            if grad_diag:
                                f.write(
                                    f"[DEBUG] Epoch {epoch + 1} batch#{grad_diag.get('batch_idx', 0) + 1}: "
                                    f"grad_tensors={grad_diag.get('grad_tensors', 0)}/{grad_diag.get('trainable_tensors', 0)}, "
                                    f"grad_params={grad_diag.get('grad_params', 0):,}/{grad_diag.get('trainable_params', 0):,}, "
                                    f"missing_sample={grad_diag.get('missing_grad_names', [])[:8]}\n"
                                )
                            if il_debug_cfg.get('log_gpu_memory', debug_cfg.get('log_gpu_memory', False)):
                                f.write(
                                    f"[DEBUG] Epoch {epoch + 1} GPU: "
                                    f"peak_alloc={train_profile.get('peak_allocated_mb', 0.0):.1f}MB, "
                                    f"peak_reserved={train_profile.get('peak_reserved_mb', 0.0):.1f}MB, "
                                    f"curr_alloc={train_profile.get('current_allocated_mb', 0.0):.1f}MB, "
                                    f"curr_reserved={train_profile.get('current_reserved_mb', 0.0):.1f}MB\n"
                                )

            # Save one rolling checkpoint per version for exact resume.
            model_to_save = model
            elo_metadata, elo_epoch_for_state, elo_for_state = _get_elo_state_for_checkpoint()
            latest_metadata = {
                'history_positions': history_positions,
                'input_planes': expected_input_planes,
                'training_batch_size': trained_batch_size,
                'positions_per_game': dict(positions_per_game_cfg),
                'sample_dedup': dict(sample_dedup_cfg),
                'soft_targets': dict(soft_targets_cfg),
                'pov_enabled': True,  # 🆕 v4.2
                'version': model_version,
                'startup_mode': start_mode,
                'model_architecture': model_architecture,
            }
            if val_losses is not None:
                latest_monitor_loss, latest_monitor_label = _il_monitor_loss(
                    val_losses,
                    config['imitation_learning'],
                )
                latest_metadata.update({
                    'val_loss': val_losses['total'],
                    'val_policy_loss': val_losses['policy'],
                    'val_value_loss': val_losses['value'],
                    'early_stop_monitor': latest_monitor_label,
                    'early_stop_monitor_loss': latest_monitor_loss,
                })
            if val_metrics is not None:
                latest_metadata.update({
                    'val_policy_top1': val_metrics['policy_top1_acc'],
                    'val_policy_top3': val_metrics['policy_top3_acc'],
                    'val_value_mae': val_metrics['value_mae'],
                })
            latest_metadata.update(elo_metadata)
            saved_latest_path = Path(save_checkpoint(
                model_to_save,
                optimizer,
                epoch,
                train_losses['total'],
                str(latest_checkpoint_path),
                latest_metadata,
                save_optimizer=save_optimizer_state,
                save_dtype=torch.bfloat16 if use_bfloat16 else None,
                extra_state=build_runtime_state(
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    estimated_elo=elo_for_state,
                    estimated_elo_epoch=elo_epoch_for_state,
                ),
            ))

            latest_size_mb = saved_latest_path.stat().st_size / (1024 ** 2)
            print(f"💾 Latest checkpoint updated: {saved_latest_path.name} ({latest_size_mb:.1f} MB)")

            if should_stop:
                break

            # Cleanup
            if (epoch + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        training_interrupted = True
        print("\nCtrl+C detected. Attempting graceful shutdown...")

    elo_coordinator.shutdown(interrupted=training_interrupted)

    # Final plot
    logger.plot()

    min_interrupt_final_elo_epochs = 10 if start_mode == "new" else 1
    skip_interrupt_final_elo = bool(
        training_interrupted
        and completed_epochs_this_run < min_interrupt_final_elo_epochs
    )
    if skip_interrupt_final_elo:
        if start_mode == "new":
            print(
                "Ctrl+C before 10 completed epochs from scratch; "
                "skipping final IL MCTS Elo checks."
            )
        else:
            print("Ctrl+C happened before any epoch finished; skipping final IL Elo checks.")

    logged_epoch_values = []
    for logged_epoch in getattr(logger, "iterations", []) or []:
        try:
            logged_epoch_values.append(int(logged_epoch))
        except (TypeError, ValueError):
            pass
    final_completed_epoch = max(logged_epoch_values) if logged_epoch_values else max(0, int(start_epoch or 0))
    latest_estimated_epoch, _latest_estimated_elo = logger.get_latest_estimated_elo_with_epoch()
    final_epoch_num = max(1, int(latest_estimated_epoch or final_completed_epoch or 1))
    final_epoch_idx_for_swa = max(0, int(final_completed_epoch or final_epoch_num) - 1)

    # Finalize SWA at training end and also on interrupt (if SWA has updates).
    swa_elo_stop_event = threading.Event()
    try:
        swa_final_result = finalize_swa_model(
            final_epoch_idx=final_epoch_idx_for_swa,
            interrupted=training_interrupted,
            use_swa=use_swa,
            swa_model=swa_model,
            use_amp=use_amp,
            use_bfloat16=use_bfloat16,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            best_model_path=best_model_path,
            swa_start=swa_start,
            history_positions=history_positions,
            expected_input_planes=expected_input_planes,
            model_version=model_version,
            start_mode=start_mode,
            best_val_loss=best_val_loss,
            best_raw_val_loss=best_raw_val_loss_for_swa,
            evaluate_il_fn=evaluate_il,
            elo_config=final_elo_config_il if final_il_elo_enabled and not skip_interrupt_final_elo else None,
            elo_stop_event=swa_elo_stop_event,
            model_architecture=model_architecture,
            training_batch_size=trained_batch_size,
        )
    except KeyboardInterrupt:
        swa_elo_stop_event.set()
        print("\nSecond Ctrl+C detected. Final SWA/Elo step cancelled.")
        swa_final_result = {"finalized": False, "elo_cancelled": True}

    swa_was_active = bool(use_swa and final_epoch_num >= int(swa_start))
    if swa_was_active:
        marker_label = "SWA final" if swa_final_result.get("finalized") else "SWA skipped"
        logger.add_elo_epoch_marker(final_epoch_num, marker_label)

    if swa_final_result.get("finalized"):
        swa_elo = swa_final_result.get("estimated_elo")
        swa_epoch = final_epoch_num

        # Record full SWA metrics for the summary table + gold-star Elo point
        logger.record_swa_metrics(
            val_loss=swa_final_result.get("val_loss"),
            val_policy_loss=swa_final_result.get("val_policy_loss"),
            val_value_loss=swa_final_result.get("val_value_loss"),
            top1=swa_final_result.get("val_top1"),
            top3=swa_final_result.get("val_top3"),
            mae=swa_final_result.get("val_mae"),
            wdl_acc=swa_final_result.get("val_wdl_acc"),
            wdl_ce=swa_final_result.get("val_wdl_ce"),
            elo=swa_elo,
            epoch=swa_epoch,
        )
        if swa_elo is not None:
            logger.record_il_mode_elo(
                swa_epoch,
                swa_elo,
                mode="nn",
                simulations=0,
                label="SWA final NN",
                update_csv=True,
                std_error=swa_final_result.get("estimated_elo_std_error"),
                ci95=swa_final_result.get("estimated_elo_ci95"),
            )

        swa_note = (
            "SWA final: "
            f"val_loss={swa_final_result['val_loss']:.4f}, "
            f"top1={swa_final_result['val_top1']:.2%}, "
            f"mae={swa_final_result['val_mae']:.4f}"
        )
        if swa_elo is not None:
            swa_note += f", elo={int(round(float(swa_elo)))}"
        else:
            swa_note += ", elo=n/a"
        logger.append_final_note(swa_note)
        logger.plot()

        if final_il_elo_enabled and not skip_interrupt_final_elo and not swa_final_result.get("elo_cancelled"):
            swa_model_path = Path(swa_final_result.get("model_path") or (best_model_path.parent / "best_model_il_swa.pt"))
            swa_elo_model, swa_elo_label = _load_il_model_for_final_elo(
                swa_model_path,
                config,
                device,
                fallback_model=None,
            )
            if swa_elo_model is not None:
                swa_mcts_result = _run_il_final_elo(
                    model=swa_elo_model,
                    config=config,
                    device=device,
                    elo_config=final_elo_config_il_mcts,
                    logger=logger,
                    epoch_num=int(swa_epoch),
                    model_label=f"SWA IL: {swa_elo_label} MCTS",
                    marker_label="SWA final MCTS",
                    interrupted=training_interrupted,
                    checkpoint_path=swa_model_path,
                )
                if swa_mcts_result.get("estimated_elo") is not None:
                    logger.append_final_note(
                        f"SWA final MCTS: {int(round(float(swa_mcts_result['estimated_elo'])))}"
                    )
                    logger.plot()
    elif swa_was_active:
        logger.append_final_note(f"SWA final: not saved (epoch {final_epoch_num})")
        logger.plot()

    final_best_elo_result = {}
    if final_il_elo_enabled and not skip_interrupt_final_elo and not swa_final_result.get("elo_cancelled"):
        best_elo_model, best_elo_label = _load_il_model_for_final_elo(
            best_model_path,
            config,
            device,
            fallback_model=model,
        )
        if best_elo_model is not None:
            best_checkpoint_loaded = best_elo_label != "current model"
            final_model_label = f"best IL: {best_elo_label}" if best_checkpoint_loaded else "current model"
            final_marker_label = "Best IL final Elo" if best_checkpoint_loaded else "Current IL final Elo"
            final_note_label = "Best IL final Elo" if best_checkpoint_loaded else "Current IL final Elo"
            print("\n" + "=" * 88)
            print("Final best IL strength check")
            print(f"Checkpoint: {best_elo_label}")
            print(
                "Fresh evaluation: Raw NN, then final-only MCTS profile "
                + "/".join(
                    str(int(cfg.get("mcts_simulations", 0) or 0))
                    for cfg in final_elo_config_il_mcts_profile
                )
            )
            print("=" * 88)
            final_best_elo_results = []
            final_best_mode_plan = [(final_elo_config_il, "Raw NN")]
            final_best_mode_plan.extend(
                (
                    mode_cfg,
                    f"MCTS @{int(mode_cfg.get('mcts_simulations', 0) or 0)}",
                )
                for mode_cfg in final_elo_config_il_mcts_profile
            )
            for mode_cfg, mode_suffix in final_best_mode_plan:
                mode_marker = f"{final_marker_label} {mode_suffix}"
                final_best_elo_result = _run_il_final_elo(
                    model=best_elo_model,
                    config=config,
                    device=device,
                    elo_config=mode_cfg,
                    logger=logger,
                    epoch_num=final_epoch_num,
                    model_label=f"{final_model_label} {mode_suffix}",
                    marker_label=mode_marker,
                    interrupted=training_interrupted,
                    checkpoint_path=best_model_path if best_checkpoint_loaded else None,
                )
                final_best_elo_results.append(final_best_elo_result)
                if final_best_elo_result.get("cancelled"):
                    break
                if final_best_elo_result.get("estimated_elo") is not None:
                    if mode_suffix == "Raw NN":
                        logger.record_best_final_elo(
                            final_epoch_num,
                            final_best_elo_result["estimated_elo"],
                        )
                    logger.append_final_note(
                        f"{final_note_label} {mode_suffix}: "
                        f"{int(round(float(final_best_elo_result['estimated_elo'])))}"
                    )
                    logger.plot()
            successful_final_modes = [
                (mode_suffix, result)
                for (_, mode_suffix), result in zip(final_best_mode_plan, final_best_elo_results)
                if result.get("estimated_elo") is not None
            ]
            if successful_final_modes:
                print("\n" + "-" * 88)
                print("Final best IL results")
                for mode_suffix, result in successful_final_modes:
                    elo_value = int(round(float(result["estimated_elo"])))
                    uncertainty = ""
                    if result.get("elo_std_error") is not None:
                        uncertainty = f" +/-{float(result['elo_std_error']):.1f} SE"
                    ci95 = result.get("elo_ci95")
                    if isinstance(ci95, (list, tuple)) and len(ci95) == 2:
                        uncertainty += f", 95% CI {ci95[0]}-{ci95[1]}"
                    print(f"  {mode_suffix:<8} {elo_value:>5}{uncertainty}")
                nn_result = next(
                    (result for label, result in successful_final_modes if label == "Raw NN"),
                    None,
                )
                if nn_result is not None:
                    nn_elo = float(nn_result["estimated_elo"])
                    for mode_suffix, result in successful_final_modes:
                        if not mode_suffix.startswith("MCTS"):
                            continue
                        mcts_elo = float(result["estimated_elo"])
                        print(
                            f"  {mode_suffix} gain over Raw NN: "
                            f"{mcts_elo - nn_elo:+.0f} Elo"
                        )
                print("-" * 88)
            final_best_elo_result = next(
                (r for r in final_best_elo_results if r.get("estimated_elo") is not None),
                final_best_elo_results[-1] if final_best_elo_results else {},
            )

    if training_interrupted:
        if (
            not skip_interrupt_final_elo
            and not swa_final_result.get("elo_cancelled")
            and swa_final_result.get("estimated_elo") is None
        ):
            final_elo_result = final_best_elo_result or {}
            ran_current_final_elo = False
            if final_elo_result.get("estimated_elo") is None and not final_elo_result.get("cancelled"):
                ran_current_final_elo = True
                current_results = []
                for mode_cfg, mode_suffix in (
                    (final_elo_config_il, "NN"),
                    (final_elo_config_il_mcts, "MCTS"),
                ):
                    final_elo_result = _run_il_final_elo(
                        model=model,
                        config=config,
                        device=device,
                        elo_config=mode_cfg,
                        logger=logger,
                        epoch_num=final_epoch_num,
                        model_label=f"current model {mode_suffix}",
                        marker_label=f"Ctrl+C final Elo {mode_suffix}",
                        interrupted=True,
                    )
                    current_results.append(final_elo_result)
                    if final_elo_result.get("cancelled"):
                        break
                final_elo_result = next(
                    (r for r in current_results if r.get("estimated_elo") is not None),
                    current_results[-1] if current_results else {},
                )
            if ran_current_final_elo and final_elo_result.get("estimated_elo") is not None:
                logger.append_final_note(
                    f"Ctrl+C final Elo: {int(round(float(final_elo_result['estimated_elo'])))}"
                )
                logger.plot()
        cleanup_interrupted_log_csv(_LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG, "IL")
        print("IL training stopped gracefully.")
        return

    print("\nTraining complete.")
    print(f"Best monitor loss: {best_val_loss:.4f} ({monitor_label})")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {il_dir}")
    print(f"Logs: {logs_dir}")
    if debug_log_file:
        print(f"Debug log: {debug_log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup_interrupted_log_csv(_LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG, "IL")
        print("\nCtrl+C detected. IL training stopped gracefully.")
    except Exception as exc:
        if _is_fatal_cuda_error(exc):
            print(
                "\nFatal CUDA error detected during IL training. "
                "This usually means the Windows/NVIDIA driver reset the CUDA context "
                "(for example after display sleep/wake or device change)."
            )
            print(
                "The current Python process cannot safely recover this CUDA context. "
                "Use the latest completed-epoch checkpoint to resume."
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(90)
        raise
