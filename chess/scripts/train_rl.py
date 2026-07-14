"""
Reinforcement Learning Training Script with Comprehensive Metrics

NEW Features:
- Policy Accuracy tracking during training
- Value MAE monitoring
- Enhanced self-play statistics
"""

import os
import sys
import signal
import shutil
import contextlib
import multiprocessing as _stdlib_mp
from collections import defaultdict

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# Windows multiprocessing ("spawn") starts worker processes by re-running this script as __mp_main__.
# Ignore Ctrl+C inside spawned children to avoid noisy KeyboardInterrupt tracebacks during heavy imports.
def _is_spawned_worker_process():
    if __name__ != "__mp_main__":
        return False
    try:
        return _stdlib_mp.parent_process() is not None
    except Exception:
        return True


if _is_spawned_worker_process():
    # Native runtimes (MKL/Fortran/OpenMP) used by numpy/torch may print noisy
    # "forrtl: error (200)" on Ctrl+C in spawned workers. Keep children out of
    # console Ctrl+C handling; parent process performs graceful shutdown.
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "TRUE")
    os.environ.setdefault("KMP_HANDLE_SIGNALS", "0")
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
import time
import pickle
import tempfile
import math
import queue
import threading
from datetime import datetime

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.utils.config import normalize_config
from src.utils.q_delta import (
    Q_DELTA_HIST_BINS as _Q_DELTA_HIST_BINS,
    q_delta_histogram as _q_delta_histogram,
    q_delta_percentile_from_histogram as _q_delta_percentile_from_histogram,
)

from src.model import ChessNet, save_checkpoint, normalize_state_dict_keys, load_checkpoint_file, transfer_matching_weights

# Import MCTS self-play
try:
    from src.batch_selfplay import (
        play_games_mcts_worker,
        _resolve_replay_max_policy_targets,
    )
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("Warning: MCTS self-play not available")

# Import from utils
from utils.shared.logger import TrainingLogger
from utils.shared.elo_runner import (
    build_final_elo_config,
    run_elo_check,
    safe_float as _safe_float,
)
from utils.shared.model_catalog import load_checkpoint_metadata, persist_checkpoint_elo_metadata
from utils.rl.replay import ReplayBuffer
from utils.rl.training_rl import (
    eval_central_inference_runtime,
    evaluate_models,
    evaluate_models_no_mcts,
    train_on_batch_rl,
)
from utils.rl.startup import plan_rl_startup, apply_rl_startup_plan
from utils.rl.console import compact_path, dominant_stage, format_duration, print_panel
from utils.rl.profiler import print_selfplay_profiler
from utils.rl.opponent_scheduler import (
    _safe_score_rate,
    _build_selfplay_opponent_assignments,
)
from utils.rl.persistent_pool import (
    _resolve_central_inference_server_count,
    get_or_create_selfplay_pool,
    _shutdown_selfplay_pool,
    _terminate_workers,
    _is_interrupt_exit_code,
)
from utils.il.auto_tune import resolve_rl_hyperparameters
from utils.shared.metrics import MetricsCalculator
from utils.shared.runtime_helpers import (
    build_rl_experiment_name,
    build_model_file_tag,
    build_model_architecture_metadata,
    cleanup_interrupted_log_csv,
)
from utils.shared.syzygy_manager import ensure_syzygy_tables, describe_syzygy_status


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None
# Keep one replay pass per iteration, but avoid collapsing it into a handful of
# huge-batch updates. Training is a small part of runtime, and ~32 updates give
# the policy head twice as many opportunities to absorb MCTS corrections.
_TARGET_OPTIMIZER_STEPS_PER_REPLAY_PASS = 32
_ANCHOR_CANDIDATE_MAX_GAMES = 200


def _build_eval_config_with_exact_simulations(config, simulations):
    eval_config = dict(config)
    rl_config = dict(config.get('reinforcement_learning', {}) or {})
    rl_config['mcts_simulations'] = max(1, int(simulations))
    rl_config['eval_mcts_simulations_multiplier'] = 1.0
    eval_config['reinforcement_learning'] = rl_config
    return eval_config


def _build_rl_optimizer_param_groups(model, rl_config, base_lr):
    base_lr = float(base_lr)
    backbone_lr_factor = float(rl_config.get('backbone_lr_factor', 0.75))
    policy_lr_factor = float(rl_config.get('policy_head_lr_factor', 1.0))
    value_lr_factor = float(rl_config.get('value_head_lr_factor', 1.5))
    backbone_weight_decay = float(rl_config.get('weight_decay', 0.01))
    head_weight_decay = float(rl_config.get('head_weight_decay', min(backbone_weight_decay, 0.01)))
    no_decay_weight_decay = float(rl_config.get('no_decay_weight_decay', 0.0))

    buckets = {
        'backbone_decay': {'params': [], 'lr_factor': backbone_lr_factor, 'weight_decay': backbone_weight_decay},
        'backbone_no_decay': {'params': [], 'lr_factor': backbone_lr_factor, 'weight_decay': no_decay_weight_decay},
        'policy_decay': {'params': [], 'lr_factor': policy_lr_factor, 'weight_decay': head_weight_decay},
        'policy_no_decay': {'params': [], 'lr_factor': policy_lr_factor, 'weight_decay': no_decay_weight_decay},
        'value_decay': {'params': [], 'lr_factor': value_lr_factor, 'weight_decay': head_weight_decay},
        'value_no_decay': {'params': [], 'lr_factor': value_lr_factor, 'weight_decay': no_decay_weight_decay},
    }

    def _family_for_name(name):
        lowered = name.lower()
        if (
            lowered.startswith('value_')
            or '.value_' in lowered
            or lowered.startswith('moves_left_')
            or '.moves_left_' in lowered
        ):
            return 'value'
        if lowered.startswith('policy_') or '.policy_' in lowered:
            return 'policy'
        return 'backbone'

    def _uses_no_decay(name, param):
        lowered = name.lower()
        return (
            param.ndim < 2
            or lowered.endswith('.bias')
            or lowered.endswith('_bias')
            or 'bn' in lowered
            or 'norm' in lowered
        )

    param_groups = []
    group_summaries = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        family = _family_for_name(name)
        decay_kind = 'no_decay' if _uses_no_decay(name, param) else 'decay'
        bucket_name = f'{family}_{decay_kind}'
        buckets[bucket_name]['params'].append(param)

    for bucket_name, bucket in buckets.items():
        params = bucket['params']
        if not params:
            continue
        group = {
            'params': params,
            'lr': base_lr * float(bucket['lr_factor']),
            'lr_factor': float(bucket['lr_factor']),
            'weight_decay': float(bucket['weight_decay']),
            'name': bucket_name,
        }
        param_groups.append(group)
        group_summaries.append(
            f"{bucket_name}: n={sum(p.numel() for p in params):,}, "
            f"lr_factor={group['lr_factor']:.2f}, wd={group['weight_decay']:.4g}"
        )

    return param_groups, group_summaries


def _merge_q_delta_histogram(target, key, source_hist=None, source_values=None):
    hist = target.setdefault(key, [0] * _Q_DELTA_HIST_BINS)
    if not isinstance(hist, list) or len(hist) != _Q_DELTA_HIST_BINS:
        hist = [0] * _Q_DELTA_HIST_BINS
        target[key] = hist
    if source_hist is None:
        source_hist = _q_delta_histogram(source_values or [])
    source_hist = list(source_hist or [])
    if len(source_hist) != _Q_DELTA_HIST_BINS:
        return
    for idx, count in enumerate(source_hist):
        hist[idx] += int(count or 0)


def _print_rl_startup_plan(
    *,
    config,
    model,
    logger,
    source_label,
    start_mode,
    start_iteration,
    total_iterations,
    worker_plan,
    replay_max_policy_targets,
    funnel_stages,
    warmup_iters,
    min_lr_ratio,
    use_amp,
    use_bfloat16,
):
    """Print the resolved experiment contract once, after all startup decisions."""
    rl_cfg = config.get('reinforcement_learning', {}) or {}
    central_servers = 0
    if worker_plan.get('central_inference'):
        central_servers = _resolve_central_inference_server_count(
            config,
            worker_plan.get('worker_specs', []),
            worker_plan.get('device_type', 'cpu'),
        )
    gpu_name = "CPU"
    if torch.cuda.is_available() and str(config.get('hardware', {}).get('device', '')).startswith('cuda'):
        gpu_name = torch.cuda.get_device_name(0)
    precision = "bf16" if use_bfloat16 else ("AMP fp16" if use_amp else "fp32")
    train_schedule = "fixed"
    if bool(rl_cfg.get('use_lr_schedule', False)):
        train_schedule = f"warmup {warmup_iters} -> cosine x{min_lr_ratio:.2f}"
    current_share = float(rl_cfg.get('self_play_opponent_current_fraction', 0.0) or 0.0)
    best_share = float(rl_cfg.get('self_play_opponent_best_fraction', 0.0) or 0.0)
    promotion_lb = float(rl_cfg.get('promotion_score_lower_bound_min', 0.50) or 0.50)
    anchor_lb = float(rl_cfg.get('promotion_anchor_score_lower_bound_min', 0.47) or 0.47)
    source_text = source_label
    if start_mode == 'resume':
        source_text = f"{source_label}; continue at iteration {start_iteration + 1}"
    if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False)):
        search_budget_text = (
            f"dynamic {int(rl_cfg.get('mcts_dynamic_budget_min', 64))}-"
            f"{int(rl_cfg.get('mcts_dynamic_budget_max', 320))} sims "
            f"(target avg {int(rl_cfg.get('mcts_dynamic_budget_target_avg', 192))})"
        )
    else:
        search_budget_text = f"{int(rl_cfg.get('mcts_simulations', 0))} sims"

    rows = [
        (
            "model",
            f"{config.get('model', {}).get('version', '?')} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params "
            f"| {start_mode} from {source_text}",
        ),
        (
            "runtime",
            f"{gpu_name} | {precision} | {len(worker_plan.get('worker_specs', []))} workers "
            f"| central inference x{central_servers if central_servers else 0}",
        ),
        (
            "self-play",
            f"{int(rl_cfg.get('games_per_iteration', 0))} games/iter | "
            f"current:best={current_share:.0%}:{best_share:.0%} | replay {int(rl_cfg.get('replay_buffer_size', 0)):,} "
            f"| target <= {int(replay_max_policy_targets)} moves",
        ),
        (
            "search",
            f"{search_budget_text} | batch {int(rl_cfg.get('mcts_batch_size', 0))} "
            f"| Q={float(rl_cfg.get('mcts_q_selection_weight', 0.0) or 0.0):.3f} "
            f"| c_puct={float(rl_cfg.get('mcts_c_puct_init', 0.0)):.2f}..{float(rl_cfg.get('mcts_c_puct_max', 0.0)):.2f} "
            f"| temp={float(rl_cfg.get('mcts_temperature', 0.0)):.2f}",
        ),
        (
            "learning",
            f"batch {int(rl_cfg.get('batch_size', 0)):,} | lr {float(rl_cfg.get('learning_rate', 0.0)):.2e} "
            f"| {train_schedule} | AdamW layer-wise",
        ),
        (
            "evaluation",
            f"every {int(rl_cfg.get('eval_every', 1))} iter | funnel {funnel_stages} | "
            f"promote score >= {float(rl_cfg.get('score_rate_threshold', 0.55)):.0%}, LB >= {promotion_lb:.0%} "
            f"| anchor LB >= {anchor_lb:.0%}",
        ),
        (
            "run",
            f"iterations {start_iteration + 1}-{total_iterations} | artifacts {compact_path(logger.csv_path.parent, 3)}/"
            f"{logger.csv_path.stem} (main + quality + performance)",
        ),
    ]
    print_panel("RL RUN PLAN", rows)


def _cuda_memory_stats(device):
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "current_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
        "current_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }


def _print_rl_iteration_profile(
    iteration_num,
    total_iterations,
    stage_times=None,
    total_time_s=None,
    gpu_stats=None,
    include_stage_times=True,
):
    print("Iteration profile:")
    if include_stage_times and stage_times is not None and total_time_s is not None:
        ordered_keys = [
            "setup",
            "selfplay",
            "replay",
            "train",
            "regular_eval",
            "promotion_eval",
            "elo_eval",
            "log",
            "checkpoint",
            "gc",
        ]
        for key in ordered_keys:
            value = float(stage_times.get(key, 0.0) or 0.0)
            if value <= 0.0 and key not in stage_times:
                continue
            pct = (100.0 * value / total_time_s) if total_time_s > 0.0 else 0.0
            print(f"   {key:<12} {value:7.2f}s  ({pct:5.1f}%)")
        print(f"   {'total':<12} {float(total_time_s):7.2f}s")
    if gpu_stats:
        print(
            "   GPU memory   "
            f"alloc={float(gpu_stats.get('current_allocated_mb', 0.0)):.0f} MB, "
            f"reserved={float(gpu_stats.get('current_reserved_mb', 0.0)):.0f} MB, "
            f"peak_alloc={float(gpu_stats.get('peak_allocated_mb', 0.0)):.0f} MB, "
            f"peak_reserved={float(gpu_stats.get('peak_reserved_mb', 0.0)):.0f} MB"
        )
    print(f"   Iteration     {int(iteration_num)}/{int(total_iterations)}")


@contextlib.contextmanager
def _temporary_sigint_cancel_handler(cancel_event, message=None, hard_exit=False, exit_code=130):
    """Temporarily turn Ctrl+C into a direct cancel signal for blocking shutdown work."""
    sigint = getattr(signal, "SIGINT", None)
    state = {"triggered": False}
    if sigint is None or threading.current_thread() is not threading.main_thread():
        yield state
        return

    previous_handler = signal.getsignal(sigint)

    def _handle_sigint(signum, frame):
        first_trigger = not state["triggered"]
        state["triggered"] = True
        if cancel_event is not None:
            with contextlib.suppress(Exception):
                cancel_event.set()
        if first_trigger and message:
            print(message, flush=True)
        if hard_exit:
            with contextlib.suppress(Exception):
                sys.stdout.flush()
            with contextlib.suppress(Exception):
                sys.stderr.flush()
            os._exit(int(exit_code))
        raise KeyboardInterrupt

    signal.signal(sigint, _handle_sigint)
    try:
        yield state
    finally:
        with contextlib.suppress(Exception):
            signal.signal(sigint, previous_handler)


def _snapshot_model_state_cpu(model, share_memory=False):
    normalized_state = normalize_state_dict_keys(model.state_dict())
    snapshot = {}
    for key, tensor in normalized_state.items():
        cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        if share_memory:
            cpu_tensor = cpu_tensor.contiguous()
            cpu_tensor.share_memory_()
        snapshot[key] = cpu_tensor
    return snapshot


def _reset_optimizer_state(optimizer):
    if optimizer is not None:
        optimizer.state.clear()


def _post_promotion_recovery_scale(
    stabilization_remaining,
    stabilization_iterations,
    recovery_remaining,
    recovery_iterations,
    minimum_scale,
):
    minimum_scale = max(0.0, min(1.0, float(minimum_scale)))
    stabilization_remaining = max(0, int(stabilization_remaining or 0))
    recovery_remaining = max(0, int(recovery_remaining or 0))
    recovery_iterations = max(0, int(recovery_iterations or 0))
    if stabilization_remaining > 0:
        return minimum_scale, "stabilize"
    if recovery_remaining <= 0 or recovery_iterations <= 0:
        return 1.0, "steady"
    completed = max(0, recovery_iterations - recovery_remaining + 1)
    progress = max(0.0, min(1.0, float(completed) / float(recovery_iterations)))
    return minimum_scale + (1.0 - minimum_scale) * progress, "recover"


def _round_replay_capacity(value, quantum):
    quantum = max(1, int(quantum))
    value = max(1, int(math.ceil(float(value))))
    return int(math.ceil(value / quantum) * quantum)


def _resolve_train_batch_size(replay_size, rl_cfg):
    configured_batch_size = max(1, int(rl_cfg.get('batch_size', 1024)))
    if not bool(rl_cfg.get('train_dynamic_batch_size_enabled', False)):
        return configured_batch_size

    min_batch_size = max(1, int(rl_cfg.get('train_batch_size_min', configured_batch_size)))
    max_batch_size = max(min_batch_size, int(rl_cfg.get('train_batch_size_max', configured_batch_size)))
    round_to = max(1, int(rl_cfg.get('train_batch_size_round_to', 1)))
    target_steps = _TARGET_OPTIMIZER_STEPS_PER_REPLAY_PASS

    replay_size = max(1, int(replay_size))
    target_batch_size = int(math.ceil(float(replay_size) / float(target_steps)))
    target_batch_size = int(math.ceil(float(target_batch_size) / float(round_to)) * round_to)
    return max(min_batch_size, min(max_batch_size, target_batch_size))


def _build_elo_checkpoint_metadata(estimated_elo, elo_config, source="rl_elo", elo_result=None):
    try:
        elo_value = float(estimated_elo)
    except (TypeError, ValueError):
        return {}

    use_mcts = bool((elo_config or {}).get("use_mcts", False))
    simulations = int((elo_config or {}).get("mcts_simulations", 0) or 0) if use_mcts else 0
    mode_prefix = "estimated_elo_mcts" if use_mcts else "estimated_elo_nn"
    timestamp = datetime.now().isoformat(timespec="seconds")
    settings = {
        "levels": [int(x) for x in list((elo_config or {}).get("levels", []) or [])],
        "games_per_level": int((elo_config or {}).get("games_per_level", 0) or 0),
        "use_mcts": use_mcts,
        "simulations": simulations,
        "stockfish_time_limit": float((elo_config or {}).get("stockfish_time_limit", 0.0) or 0.0),
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
    elo_std_error = _safe_float(settings.get("elo_std_error"))
    elo_ci95 = settings.get("elo_ci95")
    ci_low = _safe_float(elo_ci95[0]) if isinstance(elo_ci95, (list, tuple)) and len(elo_ci95) == 2 else None
    ci_high = _safe_float(elo_ci95[1]) if isinstance(elo_ci95, (list, tuple)) and len(elo_ci95) == 2 else None
    metadata = {
        mode_prefix: elo_value,
        f"last_{mode_prefix}": elo_value,
        f"{mode_prefix}_timestamp": timestamp,
        f"{mode_prefix}_settings": settings,
        f"{mode_prefix}_source": str(source),
        "estimated_elo": elo_value,
        "last_estimated_elo": elo_value,
        "estimated_elo_source": str(source),
        "estimated_elo_timestamp": timestamp,
        "estimated_elo_settings": settings,
    }
    if elo_std_error is not None:
        metadata[f"{mode_prefix}_se"] = float(elo_std_error)
        metadata["estimated_elo_se"] = float(elo_std_error)
    if ci_low is not None:
        metadata[f"{mode_prefix}_ci95_low"] = float(ci_low)
        metadata["estimated_elo_ci95_low"] = float(ci_low)
    if ci_high is not None:
        metadata[f"{mode_prefix}_ci95_high"] = float(ci_high)
        metadata["estimated_elo_ci95_high"] = float(ci_high)
    if use_mcts:
        metadata["estimated_elo_mcts_simulations"] = simulations
        sim_entry = {
            "elo": elo_value,
            "simulations": int(simulations),
            "timestamp": timestamp,
            "source": str(source),
            "settings": settings,
        }
        if elo_std_error is not None:
            sim_entry["se"] = float(elo_std_error)
        if ci_low is not None and ci_high is not None:
            sim_entry["ci95"] = [float(ci_low), float(ci_high)]
        metadata["estimated_elo_mcts_by_simulations"] = {
            str(int(simulations)): {
                **sim_entry,
            }
        }
    return metadata


def _best_mcts_elo_from_metadata(entry):
    """Return the preferred MCTS Elo metadata as (elo, simulations)."""
    if not isinstance(entry, dict):
        return None, None
    by_sims = entry.get("elo_mcts_by_simulations") or {}
    if isinstance(by_sims, dict) and by_sims:
        try:
            sims, info = max(by_sims.items(), key=lambda item: int(item[0]))
            if isinstance(info, dict) and info.get("elo") is not None:
                return float(info.get("elo")), int(sims)
        except Exception:
            pass
    if entry.get("elo_mcts") is not None:
        return entry.get("elo_mcts"), entry.get("elo_mcts_simulations")
    return None, None


def _seed_rl_plot_elo_from_checkpoint(logger, checkpoint_path, base_dir=None):
    """Add iteration-0 Elo to RL plots from the active startup checkpoint metadata."""
    if logger is None or checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return
    entry = load_checkpoint_metadata(checkpoint_path, base_dir=base_dir)
    if entry.get("error"):
        return
    elo_nn = entry.get("elo_nn")
    elo_mcts, mcts_sims = _best_mcts_elo_from_metadata(entry)
    if elo_nn is None and elo_mcts is None:
        return
    logger.record_rl_model_info_elo(
        0,
        elo_nn=elo_nn,
        elo_mcts=elo_mcts,
        mcts_simulations=mcts_sims,
    )
    parts = []
    if elo_nn is not None:
        parts.append(f"raw NN {int(round(float(elo_nn)))}")
    if elo_mcts is not None:
        sims_text = f"@{int(mcts_sims)}" if mcts_sims is not None else ""
        parts.append(f"MCTS {int(round(float(elo_mcts)))}{sims_text}")
    return f"{checkpoint_path.name}: {', '.join(parts)}"


class RLEloCoordinator:
    def __init__(self, model, config, device, logger, checkpoint_targets=None):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.elo_config = dict(config.get("elo_estimation", {}) or {})
        rl_cfg = config.get("reinforcement_learning", {}) or {}
        self.rl_elo_config = dict(rl_cfg.get("stockfish_elo", {}) or {})
        self.checkpoint_targets = [Path(p) for p in list(checkpoint_targets or []) if p is not None]
        self.enabled = bool(self.elo_config.get("enabled", False))
        # RL evaluation is synchronous by design (training is paused while Elo runs).
        # Prefer CUDA for RL Elo/MCTS evaluation when available.
        rl_eval_device_raw = str(self.rl_elo_config.get("device", "cuda")).strip().lower()
        if rl_eval_device_raw in {"same", ""}:
            self.eval_device = device
        elif rl_eval_device_raw == "cpu":
            self.eval_device = torch.device("cpu")
        else:
            self.eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.eval_device.type == "cuda" and not torch.cuda.is_available():
            self.eval_device = torch.device("cpu")
        self.last_elo_iteration = None
        self.last_elo_metadata = {}
        self.interrupted_during_elo = False
        self.il_anchor_surpassed = False
        self.best_il_anchor_score_rate = None
        self.best_il_anchor_true_win_rate = None
        self.best_il_anchor_iteration = None
        self.promoted_evaluations = 0

    def _resolve_eval_mcts_simulations(self):
        if "mcts_eval_simulations" in self.elo_config:
            try:
                return max(1, int(self.elo_config.get("mcts_eval_simulations") or 1))
            except Exception:
                pass
        rl_cfg = self.config.get("reinforcement_learning", {}) or {}
        try:
            base_sims = max(1, int(rl_cfg.get("mcts_simulations", 50) or 50))
        except Exception:
            base_sims = 50
        try:
            multiplier = max(0.01, float(rl_cfg.get("eval_mcts_simulations_multiplier", 1.0) or 1.0))
        except Exception:
            multiplier = 1.0
        return max(1, int(round(float(base_sims) * multiplier)))

    def print_startup_summary(self):
        if not self.enabled:
            return
        nn_cfg = build_final_elo_config(self.elo_config, use_mcts=False)
        final_use_mcts = bool(
            self.rl_elo_config.get(
                "final_use_mcts",
                self.rl_elo_config.get("promoted_use_mcts", False),
            )
        )
        print(
            "Elo eval (RL/IL shared): promoted best vs Stockfish, "
            f"NN cap={int(nn_cfg.get('adaptive_max_total_games', 0) or 0)}, "
            f"target_SE={float(nn_cfg.get('adaptive_target_standard_error', 0.0) or 0.0):.0f}, "
            f"sync_device={self.eval_device.type}, "
            f"promoted_mcts_every={self._promoted_mcts_every()}, "
            f"final_mcts={final_use_mcts}"
        )

    def _promoted_mcts_every(self):
        if not bool(self.rl_elo_config.get("promoted_use_mcts", False)):
            return 0
        return max(1, int(self.rl_elo_config.get("promoted_mcts_every", 3) or 3))

    def observe_il_anchor_eval(self, iteration_num, score_rate=None, true_win_rate=None):
        if score_rate is None:
            return
        try:
            score = float(score_rate)
        except (TypeError, ValueError):
            return
        try:
            true_win = float(true_win_rate) if true_win_rate is not None else 0.0
        except (TypeError, ValueError):
            true_win = 0.0
        if self.best_il_anchor_score_rate is None or score > float(self.best_il_anchor_score_rate):
            self.best_il_anchor_score_rate = score
            self.best_il_anchor_true_win_rate = true_win
            self.best_il_anchor_iteration = int(iteration_num)
        if score > 0.50 and true_win >= 0.0:
            if not self.il_anchor_surpassed:
                print(
                    "RL beat IL anchor "
                    f"at iteration {int(iteration_num)} "
                    f"(score={score:.2%}, true_win={true_win:.2%})."
                )
            self.il_anchor_surpassed = True

    def evaluate_promoted_best(self, iteration_num):
        if not self.enabled:
            return None
        self.promoted_evaluations += 1
        mcts_every = self._promoted_mcts_every()
        run_mcts = mcts_every > 0 and self.promoted_evaluations % mcts_every == 0
        mode_label = "raw NN and scheduled MCTS" if run_mcts else "raw NN"
        print(f"Promoted best model: running shared IL-grade Stockfish Elo for {mode_label}.")
        results = []
        raw_result = self._run_estimate(
            iteration_num,
            reason_label=f"promoted best raw NN {iteration_num}",
            use_mcts=False,
            persist_checkpoints=False,
            source="rl_promoted_elo",
        )
        results.append(raw_result)
        if self.interrupted_during_elo:
            return next((r for r in reversed(results) if r is not None), None)

        if run_mcts:
            mcts_result = self._run_estimate(
                iteration_num,
                reason_label=f"promoted best MCTS {iteration_num}",
                use_mcts=True,
                persist_checkpoints=False,
                source="rl_promoted_elo",
            )
            results.append(mcts_result)
        return next((r for r in reversed(results) if r is not None), None)

    def evaluate_final_best(self, iteration_num, model_override=None):
        if not self.enabled or not bool(self.rl_elo_config.get("final_enabled", True)):
            return None
        final_use_mcts = bool(
            self.rl_elo_config.get(
                "final_use_mcts",
                self.rl_elo_config.get("promoted_use_mcts", False),
            )
        )
        mode_label = "raw NN and MCTS" if final_use_mcts else "raw NN"
        print(f"Final best model: running normal Stockfish Elo for {mode_label}.")
        old_model = self.model
        if model_override is not None:
            self.model = model_override
        try:
            results = []
            raw_result = self._run_estimate(
                iteration_num,
                reason_label=f"final best raw NN {iteration_num}",
                use_mcts=False,
                persist_checkpoints=True,
                source="rl_final_elo",
            )
            results.append(raw_result)
            if self.interrupted_during_elo:
                return next((r for r in reversed(results) if r is not None), None)
            if final_use_mcts:
                mcts_result = self._run_estimate(
                    iteration_num,
                    reason_label=f"final best MCTS {iteration_num}",
                    use_mcts=True,
                    persist_checkpoints=True,
                    source="rl_final_elo",
                )
                results.append(mcts_result)
            return next((r for r in reversed(results) if r is not None), None)
        finally:
            self.model = old_model

    def _run_estimate(
        self,
        iteration_num,
        reason_label,
        *,
        use_mcts,
        persist_checkpoints,
        source,
    ):
        elo_config = build_final_elo_config(self.elo_config, use_mcts=bool(use_mcts))
        if use_mcts:
            elo_config["mcts_simulations"] = self._resolve_eval_mcts_simulations()
        elo_config.setdefault("max_error_logs_per_type", 8)

        print(f"\nEstimating Elo vs Stockfish ({reason_label})...")
        max_attempts = 1
        elo_result = None
        interrupt_message = "\nCtrl+C detected during Stockfish evaluation. Cancelling evaluation..."

        for attempt_idx in range(max_attempts):
            elo_cancel_event = threading.Event()
            sigint_state = {"triggered": False}
            try:
                with _temporary_sigint_cancel_handler(
                    elo_cancel_event,
                    message=interrupt_message,
                ) as sigint_state:
                    elo_result = run_elo_check(
                        self.model,
                        self.config,
                        self.eval_device,
                        elo_config,
                        stop_event=elo_cancel_event,
                    )
            except KeyboardInterrupt:
                with contextlib.suppress(Exception):
                    elo_cancel_event.set()
                self.interrupted_during_elo = True
                if not sigint_state.get("triggered", False):
                    print(interrupt_message)
                return None
            except Exception as exc:
                print(f"Elo estimation failed: {exc}")
                return None

            elo_result = elo_result or {}
            if sigint_state.get("triggered", False) and elo_result.get("cancelled"):
                self.interrupted_during_elo = True
            break

        elo_result = elo_result or {}
        if elo_result.get("cancelled"):
            print("Elo estimation cancelled.")
            return None
        if elo_result.get("error"):
            print(f"Elo estimation failed: {elo_result['error']}")
            return None

        estimated_elo = elo_result.get("estimated_elo")
        if estimated_elo is not None:
            self.last_elo_iteration = int(iteration_num)
            source = str(source)
            try:
                self.last_elo_metadata.update(
                    _build_elo_checkpoint_metadata(
                        estimated_elo,
                        elo_config,
                        source=source,
                        elo_result=elo_result,
                    )
                )
            except Exception as exc:
                print(f"  Elo metadata build failed (result retained): {exc}")
            try:
                self.logger.record_estimated_elo_mode(
                    iteration_num,
                    estimated_elo,
                    mode="mcts" if bool(elo_config.get("use_mcts", False)) else "nn",
                    simulations=int(elo_config.get("mcts_simulations", 0) or 0),
                    update_csv=True,
                    std_error=elo_result.get("elo_std_error"),
                    ci95=elo_result.get("elo_ci95"),
                )
            except Exception as exc:
                print(f"  Elo logger update failed (result retained): {exc}")
            print(f"Estimated Elo: {estimated_elo}")
            if elo_result.get("elo_std_error") is not None:
                ci = elo_result.get("elo_ci95")
                ci_str = f", 95% CI {ci[0]}-{ci[1]}" if isinstance(ci, list) and len(ci) == 2 else ""
                ladder = "adaptive" if elo_result.get("adaptive") else "fixed"
                print(f"  uncertainty: +/-{elo_result['elo_std_error']} Elo SE{ci_str} ({ladder} ladder)")
            for lvl, res in sorted(elo_result.get("results", {}).items()):
                score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                games = int(res.get("games", res["wins"] + res["draws"] + res["losses"]) or 0)
                print(f"  vs SF {lvl}: {score_str} (score: {res['score']:.0%}, n={games})")
            print(f"  time {elo_result.get('total_time', 0.0):.1f}s ({elo_result.get('total_games', 0)} games)")
            if persist_checkpoints:
                for checkpoint_path in self.checkpoint_targets:
                    ok, error = persist_checkpoint_elo_metadata(
                        checkpoint_path,
                        estimated_elo,
                        levels=list(elo_config.get("levels", []) or []),
                        games_per_level=int(elo_config.get("games_per_level", 0) or 0),
                        use_mcts=bool(elo_config.get("use_mcts", False)),
                        simulations=int(elo_config.get("mcts_simulations", 0) or 0),
                        sf_time=float(elo_config.get("stockfish_time_limit", 0.0) or 0.0),
                        source=source,
                        elo_result=elo_result,
                    )
                    if ok:
                        print(f"  Saved Elo metadata into: {checkpoint_path.name}")
                    elif error:
                        print(f"  Elo metadata save failed for {checkpoint_path.name}: {error}")
        elif not elo_result.get("skipped"):
            print("Elo estimation: inconclusive")

        return estimated_elo

    def metadata_for_iteration(self, iteration_num):
        if self.last_elo_iteration == int(iteration_num):
            return dict(self.last_elo_metadata or {})
        return {}


class RLTrainingInterrupted(Exception):
    """Raised when user interrupts RL run (Ctrl+C)."""


def _handle_graceful_interrupt(logger=None, stage=None):
    """Best-effort graceful shutdown on Ctrl+C."""
    stage_suffix = f" during {stage}" if stage else ""
    print(f"\nRL training interrupted by user (Ctrl+C){stage_suffix}.")
    if logger is not None:
        try:
            logger.plot()
        except Exception:
            pass
    cleanup_interrupted_log_csv(_LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG, "RL")
    print("RL training stopped gracefully.")


def _empty_eval_stats():
    return {
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'unresolved': 0,
        'score_rate': 0.0,
        'win_rate': 0.0,
        'draw_rate': 0.0,
        'loss_rate': 0.0,
        'num_games': 0,
        'resolved_games': 0,
    }


def _score_rate_lower_bound(score_rate, num_games, z=1.28):
    """Conservative one-sided normal lower bound for a [0, 1] game score."""
    games = max(1.0, float(num_games or 1))
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    score_se = math.sqrt(score * (1.0 - score) / games)
    return max(0.0, score - z_value * score_se)


def _score_rate_upper_bound(score_rate, num_games, z=1.28):
    """One-sided normal upper bound paired with `_score_rate_lower_bound`."""
    games = max(1.0, float(num_games or 1))
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    score_se = math.sqrt(score * (1.0 - score) / games)
    return min(1.0, score + z_value * score_se)


def _anchor_candidate_needs_more_games(
    score_rate,
    num_games,
    *,
    min_score_rate,
    min_score_lower_bound,
    z,
    max_games,
):
    """Whether an anchor result is still statistically inconclusive."""
    games = max(1, int(num_games or 0))
    if games >= max(1, int(max_games)):
        return False
    lower = _score_rate_lower_bound(score_rate, games, z)
    if float(score_rate or 0.0) >= float(min_score_rate) and lower >= float(min_score_lower_bound):
        return False
    upper = _score_rate_upper_bound(score_rate, games, z)
    return upper >= float(min_score_rate)


def _combine_eval_stats(*stats_items):
    combined = _empty_eval_stats()
    total_games = 0.0
    weighted_unresolved = 0.0
    weighted_wins = 0.0
    weighted_draws = 0.0
    weighted_losses = 0.0
    for item in stats_items:
        if not item:
            continue
        weight = 1.0
        stats = item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            stats, weight = item
        weight = max(0.0, float(weight or 0.0))
        if weight <= 0.0 or not stats:
            continue
        weighted_wins += weight * float(stats.get('wins', 0) or 0)
        weighted_draws += weight * float(stats.get('draws', 0) or 0)
        weighted_losses += weight * float(stats.get('losses', 0) or 0)
        weighted_unresolved += weight * float(stats.get('unresolved', 0) or 0)
        total_games += weight * float(stats.get('num_games', 0) or 0)

    combined['wins'] = int(round(weighted_wins))
    combined['draws'] = int(round(weighted_draws))
    combined['losses'] = int(round(weighted_losses))
    combined['unresolved'] = int(round(weighted_unresolved))
    combined['num_games'] = int(round(total_games))
    combined['resolved_games'] = max(0, combined['num_games'] - combined['unresolved'])
    if total_games > 0:
        combined['score_rate'] = float(
            (weighted_wins + 0.5 * (weighted_draws + weighted_unresolved)) / total_games
        )
        combined['win_rate'] = float(weighted_wins / total_games)
        combined['draw_rate'] = float((weighted_draws + weighted_unresolved) / total_games)
        combined['loss_rate'] = float(weighted_losses / total_games)
    return combined


def _evaluate_anchor_candidate_sequential(
    *,
    model,
    anchor_model,
    eval_config,
    device,
    initial_games,
    max_games,
    min_score_rate,
    min_score_lower_bound,
    stat_gate_z,
    game_index_offset=0,
    use_fixed_openings=True,
    central_runtime=None,
):
    """Evaluate one frozen candidate until the anchor verdict is decisive.

    Batches use disjoint fixed-opening indices. Unlike the old cross-iteration
    streak, every observation therefore belongs to the same model weights.
    """
    batch_games = max(2, int(initial_games or 2))
    if batch_games % 2:
        batch_games += 1
    max_games = max(batch_games, int(max_games or batch_games))
    combined = None
    completed_games = 0
    batch_index = 0

    while completed_games < max_games:
        games_this_batch = min(batch_games, max_games - completed_games)
        stats = evaluate_models(
            model,
            anchor_model,
            eval_config,
            device,
            games_this_batch,
            game_index_offset=game_index_offset + completed_games,
            use_fixed_openings=use_fixed_openings,
            progress_desc=(
                "Eval vs anchor"
                if batch_index == 0
                else f"Anchor tiebreak {completed_games + 1}-{completed_games + games_this_batch}"
            ),
            central_runtime=central_runtime,
        )
        combined = _combine_eval_stats(combined, stats)
        completed_games = int((combined or {}).get('num_games', 0) or 0)
        score_rate = float((combined or {}).get('score_rate', 0.0) or 0.0)
        if not _anchor_candidate_needs_more_games(
            score_rate,
            completed_games,
            min_score_rate=min_score_rate,
            min_score_lower_bound=min_score_lower_bound,
            z=stat_gate_z,
            max_games=max_games,
        ):
            break
        batch_index += 1

    return combined or _empty_eval_stats()


def _evaluate_models_funnel(
    *,
    model,
    best_model,
    config,
    device,
    preliminary_games,
    preliminary_simulations,
    medium_games,
    medium_simulations,
    advanced_games,
    advanced_simulations,
    preliminary_score_rate,
    preliminary_true_win_rate=0.0,
    medium_score_rate=0.53,
    medium_true_win_rate=0.0,
    preliminary_result_weight=0.0,
    game_index_offset=0,
    use_fixed_openings=True,
    central_runtime=None,
):
    preliminary_games = max(1, int(preliminary_games))
    medium_games = max(0, int(medium_games))
    advanced_games = max(0, int(advanced_games))
    preliminary_config = _build_eval_config_with_exact_simulations(config, preliminary_simulations)
    medium_config = _build_eval_config_with_exact_simulations(config, medium_simulations)
    advanced_config = _build_eval_config_with_exact_simulations(config, advanced_simulations)
    preliminary_stats = evaluate_models(
        model,
        best_model,
        preliminary_config,
        device,
        preliminary_games,
        game_index_offset=game_index_offset,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    score_rate = float((preliminary_stats or {}).get('score_rate', 0.0) or 0.0)
    true_win_rate = float((preliminary_stats or {}).get('win_rate', 0.0) or 0.0)
    if (
        score_rate < float(preliminary_score_rate)
        or true_win_rate < float(preliminary_true_win_rate)
    ):
        return preliminary_stats, "funnel:preliminary_reject"
    preliminary_result_weight = max(0.0, float(preliminary_result_weight or 0.0))
    if medium_games <= 0:
        return preliminary_stats, "funnel:preliminary_only"

    medium_stats = evaluate_models(
        model,
        best_model,
        medium_config,
        device,
        medium_games,
        game_index_offset=game_index_offset + preliminary_games,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    medium_score = float((medium_stats or {}).get('score_rate', 0.0) or 0.0)
    medium_true_win = float((medium_stats or {}).get('win_rate', 0.0) or 0.0)
    if (
        medium_score < float(medium_score_rate)
        or medium_true_win < float(medium_true_win_rate)
    ):
        return medium_stats, "funnel:medium_reject"
    if advanced_games <= 0:
        if preliminary_result_weight > 0.0:
            combined = _combine_eval_stats(
                (preliminary_stats, preliminary_result_weight),
                medium_stats,
            )
            return combined, "funnel:medium_weighted"
        return medium_stats, "funnel:medium_only"

    advanced_stats = evaluate_models(
        model,
        best_model,
        advanced_config,
        device,
        advanced_games,
        game_index_offset=game_index_offset + preliminary_games + medium_games,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    combined_items = []
    if preliminary_result_weight > 0.0:
        combined_items.append((preliminary_stats, preliminary_result_weight))
    combined_items.extend([medium_stats, advanced_stats])
    combined = _combine_eval_stats(*combined_items)
    return combined, "funnel:advanced"


def _flatten_dynamic_opponent_payloads(worker_specs, opponent_assignments):
    """Merge worker-local opponent plans into a global sequential plan for dynamic dispatch."""
    global_plan_labels = []
    global_pool_entries = {}
    for rank, _games_for_worker in sorted(worker_specs, key=lambda item: int(item[0])):
        payload = dict(opponent_assignments.get(int(rank), {}) or {})
        global_plan_labels.extend(list(payload.get("plan_labels", []) or []))
        for entry in list(payload.get("pool_entries", []) or []):
            label = str((entry or {}).get("label") or "current")
            state = (entry or {}).get("state")
            if state is None or label == "current" or label in global_pool_entries:
                continue
            global_pool_entries[label] = state
    return {
        "plan_labels": global_plan_labels,
        "pool_entries": [
            {"label": label, "state": state}
            for label, state in global_pool_entries.items()
        ],
    }


def _should_run_anchor_eval(iteration_num, rl_cfg):
    if not bool(rl_cfg.get('anchor_eval_enabled', False)):
        return False
    every = max(1, int(rl_cfg.get('anchor_eval_every', 4)))
    return (iteration_num % every) == 0


def _models_have_identical_state(model_a, model_b):
    if model_a is None or model_b is None:
        return False
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    if state_a.keys() != state_b.keys():
        return False
    for key in state_a.keys():
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            return False
        if not torch.equal(tensor_a, tensor_b):
            return False
    return True


# ==============================================================================
# SELF-PLAY WITH PROPER MCTS
# ==============================================================================

def _resolve_selfplay_worker_plan(config, num_games):
    """Resolve the worker topology once for startup and self-play."""
    rl_cfg = config.get('reinforcement_learning', {})
    cpu_count = max(1, int(mp.cpu_count() or 1))
    raw_workers = rl_cfg.get('self_play_workers', 4)
    auto_workers = (
        isinstance(raw_workers, str)
        and raw_workers.strip().lower() in {'auto', 'automatic'}
    )
    if auto_workers:
        multiplier = max(0.10, float(rl_cfg.get('self_play_worker_auto_multiplier', 1.0)))
        num_workers = max(1, int(math.ceil(cpu_count * multiplier)))
    else:
        try:
            num_workers = int(raw_workers)
        except (TypeError, ValueError):
            num_workers = 4
        if num_workers <= 0:
            auto_workers = True
            multiplier = max(0.10, float(rl_cfg.get('self_play_worker_auto_multiplier', 1.0)))
            num_workers = max(1, int(math.ceil(cpu_count * multiplier)))

    requested_device = str(rl_cfg.get('self_play_device', 'auto') or 'auto').strip().lower()
    if requested_device == 'cpu':
        device_type = 'cpu'
    else:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        self_play_threads = max(1, int(rl_cfg.get('self_play_torch_threads', 1)))
    except (TypeError, ValueError):
        self_play_threads = 1
    worker_cap = max(1, cpu_count // self_play_threads)
    central_requested = bool(
        device_type == 'cuda'
        and rl_cfg.get('self_play_central_inference_enabled', False)
        and (config.get('central_inference', {}) or {}).get('enabled', True)
    )
    if central_requested:
        cap_multiplier = max(1.0, float(rl_cfg.get('self_play_worker_cap_multiplier', 1.0)))
        worker_cap = max(worker_cap, int(math.ceil(cpu_count * cap_multiplier)))
    elif device_type == 'cuda' and auto_workers:
        worker_cap = min(worker_cap, max(1, int(torch.cuda.device_count() or 1)))

    num_workers = max(1, min(int(num_workers), worker_cap, max(1, int(num_games))))
    base_games, remainder = divmod(int(num_games), num_workers)
    games_per_worker = [base_games + (rank < remainder) for rank in range(num_workers)]
    worker_specs = [
        (rank, int(games))
        for rank, games in enumerate(games_per_worker)
        if games > 0
    ]
    return {
        'device_type': device_type,
        'worker_specs': worker_specs,
        'games_per_worker': games_per_worker,
        'base_games': base_games,
        'remainder': remainder,
        'central_inference': central_requested,
    }


def play_games_parallel_mcts(
    model,
    config,
    device,
    num_games,
    replay_buffer=None,
    best_model_state=None,
):
    """
    Parallel self-play using MCTS
    
    This is the PROPER AlphaZero approach:
    - Each worker plays games using MCTS
    - Training targets = MCTS visit distributions
    - High quality training data
    """
    start_time = time.time()
    
    if not MCTS_SELFPLAY_AVAILABLE:
        print("MCTS self-play not available!")
        return [], 0, 0, 0, 0, 0
    
    model_state = model.state_dict()
    
    rl_cfg = config.get('reinforcement_learning', {})
    use_persistent_pool = bool(rl_cfg.get('persistent_self_play_workers', True))
    stream_to_replay = replay_buffer is not None and bool(
        rl_cfg.get('self_play_stream_to_replay', True)
    )
    queue_transport_enabled = bool(rl_cfg.get('self_play_queue_transport', True))
    use_queue_transport = bool(use_persistent_pool and stream_to_replay and queue_transport_enabled)
    dynamic_dispatch_enabled = bool(
        use_persistent_pool
        and use_queue_transport
        and rl_cfg.get('self_play_dynamic_dispatch', True)
    )
    
    worker_plan = _resolve_selfplay_worker_plan(config, num_games)
    device_type = worker_plan['device_type']
    worker_specs = worker_plan['worker_specs']

    use_batch_selfplay = bool(rl_cfg.get('use_batch_selfplay', False))
    max_batch_games_raw = rl_cfg.get('max_batch_games_per_worker', 1)
    try:
        max_batch_games = max(1, int(max_batch_games_raw))
    except Exception:
        max_batch_games = 1
    
    opponent_assignments, _opponent_mix_games, opponent_debug = _build_selfplay_opponent_assignments(
        rl_cfg,
        worker_specs,
        best_model_state=best_model_state,
    )
    dispatch_chunk_raw = rl_cfg.get('self_play_dispatch_chunk_games', max_batch_games)
    try:
        requested_dispatch_chunk_games = max(1, int(dispatch_chunk_raw))
    except Exception:
        requested_dispatch_chunk_games = max_batch_games
    max_initial_fair_chunk_games = max(
        1,
        int(math.ceil(float(num_games) / float(max(1, len(worker_specs))))),
    )
    dynamic_dispatch_chunk_games = min(
        requested_dispatch_chunk_games,
        max_initial_fair_chunk_games,
    )
    global_dynamic_opponent_payload = _flatten_dynamic_opponent_payloads(
        worker_specs,
        opponent_assignments,
    ) if dynamic_dispatch_enabled else {}

    # Shared temp dir used by worker result files.
    temp_dir = Path(tempfile.gettempdir()) / "chess_selfplay_mcts"
    temp_dir.mkdir(exist_ok=True)
    
    processes = []
    result_files = []
    progress_files = []
    model_state_path = None
    queue_total_positions = 0
    queue_game_lengths = []
    queue_total_dropped_positions = 0
    queue_total_truncated_games = 0
    queue_total_claimable_draw_ended_games = 0
    queue_total_adjudicated_games = 0
    queue_total_syzygy_ended_games = 0
    queue_total_syzygy_probe_positions = 0
    queue_total_syzygy_probe_hits = 0
    queue_total_completed_length_sum = 0
    queue_total_truncated_length_sum = 0
    queue_total_completed_white_wins = 0
    queue_total_completed_black_wins = 0
    queue_total_completed_draws = 0
    queue_total_decisive_games = 0
    queue_total_decisive_length_sum = 0
    queue_total_curriculum_dropped_positions = 0
    queue_total_cap_dropped_positions = 0
    queue_total_value_sum = 0.0
    queue_total_value_sq_sum = 0.0
    queue_total_value_count = 0
    queue_total_resigned_games = 0
    queue_search_simulations_used_sum = 0
    queue_search_simulations_budget_sum = 0
    queue_search_samples = 0
    queue_search_simulations_used_samples = []
    queue_search_simulations_budget_samples = []
    queue_wait_total_s = 0.0
    queue_wait_events = 0
    queue_profile_stats = {}
    queue_mcts_quality_stats = defaultdict(float)
    queue_opponent_source_games = defaultdict(int)
    queue_opponent_source_results = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "games": 0})
    selfplay_pool = None
    worker_restart_counts = defaultdict(int)
    try:
        worker_restart_limit = max(0, int(rl_cfg.get('self_play_worker_restart_limit', 2) or 0))
    except Exception:
        worker_restart_limit = 2
    
    interrupted = False
    startup_done_time = None

    def _accumulate_profile_stats(target, source):
        for key, value in dict(source or {}).items():
            if isinstance(value, (int, np.integer)):
                target[str(key)] = int(target.get(str(key), 0)) + int(value)
            else:
                target[str(key)] = float(target.get(str(key), 0.0)) + float(value)

    def _accumulate_mcts_quality_stats(target, source):
        source = dict(source or {})
        if source.get('mcts_q_delta_values') is not None:
            values = target.setdefault('mcts_q_delta_values', [])
            values.extend(list(source.get('mcts_q_delta_values') or []))
        if source.get('mcts_changed_q_delta_values') is not None:
            values = target.setdefault('mcts_changed_q_delta_values', [])
            values.extend(list(source.get('mcts_changed_q_delta_values') or []))
        _merge_q_delta_histogram(
            target,
            'mcts_q_delta_hist',
            source.get('mcts_q_delta_hist'),
            source.get('mcts_q_delta_values'),
        )
        _merge_q_delta_histogram(
            target,
            'mcts_changed_q_delta_hist',
            source.get('mcts_changed_q_delta_hist'),
            source.get('mcts_changed_q_delta_values'),
        )
        for key in [
            'mcts_prior_agreement_samples',
            'mcts_prior_agreement_sum',
            'mcts_prior_changed_count',
            'mcts_prior_top_visit_prob_sum',
            'mcts_top_prior_prob_sum',
            'mcts_policy_kl_sum',
            'mcts_top_visit_prob_sum',
            'mcts_visit_gap_sum',
            'mcts_visit_entropy_sum',
            'mcts_good_target_count',
            'mcts_explored_prior_mass_sum',
            'mcts_visited_move_count_sum',
            'mcts_legal_move_count_sum',
            'mcts_visit_coverage_ratio_sum',
            'mcts_q_delta_samples',
            'mcts_q_delta_sum',
            'mcts_changed_to_lower_q_count',
            'mcts_changed_to_higher_q_count',
            'mcts_changed_q_delta_samples',
            'mcts_changed_q_delta_sum',
            'mcts_policy_uptake_samples',
            'mcts_policy_uptake_weight_sum',
            'mcts_policy_uptake_low_count',
        ]:
            target[key] = float(target.get(key, 0.0)) + float(source.get(key, 0.0) or 0.0)
        for phase in ('opening', 'middlegame', 'endgame'):
            for suffix in (
                'samples',
                'changed_count',
            ):
                key = f'mcts_phase_{phase}_{suffix}'
                target[key] = float(target.get(key, 0.0)) + float(source.get(key, 0.0) or 0.0)

    try:
        if use_persistent_pool:
            selfplay_pool = get_or_create_selfplay_pool(config, worker_specs, device_type, temp_dir)

            task_id = f"{os.getpid()}_{time.time_ns()}"
            model_state_path = temp_dir / f"selfplay_model_{task_id}.pt"
            model_state_cpu = _snapshot_model_state_cpu(model, share_memory=True)

            if not use_queue_transport:
                torch.save(model_state_cpu, model_state_path)

            if dynamic_dispatch_enabled:
                result_files = []
                progress_files = []
                idle_ranks = []
                active_workers = {}
                completed_games_by_rank = {}
                current_progress_files = {}
                next_game_offset = 0

                for rank, _games_for_worker in worker_specs:
                    idle_ranks.append(int(rank))
                    completed_games_by_rank[int(rank)] = 0

                def _payload_for_plan_labels(plan_labels, pool_entries_source=None):
                    plan_labels = list(plan_labels or [])
                    needed_pool_labels = {
                        str(label)
                        for label in plan_labels
                        if str(label) != "current"
                    }
                    source_entries = (
                        list(pool_entries_source or [])
                        if pool_entries_source is not None
                        else list(global_dynamic_opponent_payload.get("pool_entries", []) or [])
                    )
                    return {
                        "plan_labels": plan_labels,
                        "pool_entries": [
                            entry
                            for entry in source_entries
                            if str((entry or {}).get("label") or "current") in needed_pool_labels
                        ],
                    }

                def _selfplay_runtime_overrides():
                    return {
                        'mcts_temperature': rl_cfg.get('mcts_temperature'),
                        'mcts_temperature_threshold': rl_cfg.get('mcts_temperature_threshold'),
                        'mcts_dirichlet_weight': rl_cfg.get('mcts_dirichlet_weight'),
                        'mcts_q_selection_weight': rl_cfg.get('mcts_q_selection_weight'),
                    }

                def _dispatch_chunk(rank, chunk_games, chunk_plan_labels, *, advance_offset):
                    nonlocal next_game_offset
                    rank = int(rank)
                    chunk_games = int(chunk_games)
                    payload = _payload_for_plan_labels(chunk_plan_labels)
                    result_file, progress_file = selfplay_pool.dispatch_task(
                        rank=rank,
                        task_id=task_id,
                        model_state_path=model_state_path,
                        model_state=model_state_cpu,
                        temperature=rl_cfg.get('mcts_temperature'),
                        q_selection_weight=rl_cfg.get('mcts_q_selection_weight'),
                        runtime_overrides=_selfplay_runtime_overrides(),
                        num_games=chunk_games,
                        opponent_payload=payload,
                        stream_results_to_queue=use_queue_transport,
                    )
                    active_workers[rank] = {
                        "chunk_games": chunk_games,
                        "progress_file": progress_file,
                        "plan_labels": list(chunk_plan_labels or []),
                        "payload": payload,
                    }
                    current_progress_files[rank] = progress_file
                    result_files.append(result_file)
                    progress_files.append(progress_file)
                    if advance_offset:
                        next_game_offset += chunk_games
                    return True

                def _dispatch_next_chunk(rank):
                    if next_game_offset >= num_games:
                        return False
                    chunk_games = min(dynamic_dispatch_chunk_games, num_games - next_game_offset)
                    chunk_plan_labels = list(
                        global_dynamic_opponent_payload.get("plan_labels", [])[next_game_offset: next_game_offset + chunk_games]
                    )
                    return _dispatch_chunk(
                        rank,
                        chunk_games,
                        chunk_plan_labels,
                        advance_offset=True,
                    )

                for rank in list(idle_ranks):
                    if not _dispatch_next_chunk(rank):
                        break
                idle_ranks = [rank for rank in idle_ranks if rank not in active_workers]
            else:
                result_files, progress_files = selfplay_pool.submit(
                    task_id=task_id,
                    model_state_path=model_state_path,
                    model_state=model_state_cpu,
                    temperature=rl_cfg.get('mcts_temperature'),
                    q_selection_weight=rl_cfg.get('mcts_q_selection_weight'),
                    runtime_overrides=_selfplay_runtime_overrides(),
                    worker_model_state_paths={},
                    worker_opponent_payloads=opponent_assignments,
                    stream_results_to_queue=use_queue_transport,
                )
                active_workers = {int(rank): {} for rank, _ in worker_specs}
                completed_games_by_rank = {int(rank): 0 for rank, _ in worker_specs}
                current_progress_files = {
                    int(rank): pf
                    for (rank, _), pf in zip(worker_specs, progress_files)
                }
                for (rank, games_for_worker), progress_file in zip(worker_specs, progress_files):
                    payload = opponent_assignments.get(rank) or {}
                    active_workers[int(rank)] = {
                        "chunk_games": int(games_for_worker),
                        "progress_file": progress_file,
                        "plan_labels": list(payload.get("plan_labels", []) or []),
                        "payload": payload,
                    }
            processes = [selfplay_pool.processes[rank] for rank, _ in worker_specs]
        else:
            mp_ctx = mp.get_context('spawn')
            for rank, games_for_worker in worker_specs:
                if device_type == 'cuda':
                    device_id = rank % torch.cuda.device_count()
                else:
                    device_id = 'cpu'

                result_file = temp_dir / f"worker_{rank}_mcts_results.pkl"
                result_files.append(result_file)

                p = mp_ctx.Process(
                    target=play_games_mcts_worker,
                    args=(
                        rank,
                        model_state,
                        config,
                        device_id,
                        games_for_worker,
                        str(result_file),
                        opponent_assignments.get(rank),
                    )
                )
                p.daemon = True
                p.start()
                processes.append(p)

            progress_files = [
                temp_dir / f"worker_{rank}_mcts_results.progress"
                for rank, _ in worker_specs
            ]
            completed_games_by_rank = {int(rank): 0 for rank, _ in worker_specs}
            current_progress_files = {
                int(rank): pf
                for (rank, _), pf in zip(worker_specs, progress_files)
            }

        # Wait for workers with a live games-completed progress bar.
        startup_done_time = time.time()
        games_bar = tqdm(
            total=num_games,
            desc="Self-play gry",
            unit="gra",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            pending = (
                set(active_workers.keys())
                if use_persistent_pool and dynamic_dispatch_enabled
                else {rank for rank, _ in worker_specs}
            )
            rank_to_index = {rank: idx for idx, (rank, _) in enumerate(worker_specs)}
            games_reported = {rank: 0 for rank, _ in worker_specs}
            games_bar_last = 0

            def _read_worker_progress(rank):
                progress_file = current_progress_files.get(int(rank))
                try:
                    raw = progress_file.read_text().strip() if progress_file is not None else ""
                    return max(0, int(raw) if raw else 0)
                except Exception:
                    return 0

            def _redispatch_after_worker_crash(rank, exitcode):
                rank = int(rank)
                if not use_queue_transport:
                    raise RuntimeError(
                        f"Persistent self-play worker {rank} exited unexpectedly with code {exitcode}"
                    )
                worker_restart_counts[rank] += 1
                if worker_restart_counts[rank] > worker_restart_limit:
                    raise RuntimeError(
                        f"Persistent self-play worker {rank} exited unexpectedly with code {exitcode} "
                        f"after {worker_restart_limit} restart attempts"
                    )
                active_info = dict(active_workers.get(rank) or {})
                chunk_games = max(0, int(active_info.get("chunk_games", 0) or 0))
                progress_done = min(chunk_games, _read_worker_progress(rank))
                remaining_games = max(0, chunk_games - progress_done)
                completed_games_by_rank[rank] = int(
                    completed_games_by_rank.get(rank, 0)
                ) + int(progress_done)
                try:
                    progress_file = current_progress_files.get(rank)
                    if progress_file is not None:
                        progress_file.unlink(missing_ok=True)
                except Exception:
                    pass
                active_workers.pop(rank, None)
                current_progress_files.pop(rank, None)
                games_bar.write(
                    "Warning: "
                    f"persistent self-play worker {rank} exited with code {exitcode}; "
                    f"completed={progress_done}/{chunk_games}, "
                    f"remaining={remaining_games}, "
                    f"restart={worker_restart_counts[rank]}/{worker_restart_limit}."
                )
                selfplay_pool.restart_worker(rank)
                if remaining_games <= 0:
                    if dynamic_dispatch_enabled and _dispatch_next_chunk(rank):
                        pending.add(rank)
                    else:
                        pending.discard(rank)
                    return
                if dynamic_dispatch_enabled:
                    labels = list(active_info.get("plan_labels", []) or [])
                    remaining_labels = labels[progress_done:progress_done + remaining_games]
                    _dispatch_chunk(
                        rank,
                        remaining_games,
                        remaining_labels,
                        advance_offset=False,
                    )
                else:
                    payload = dict(active_info.get("payload", {}) or {})
                    labels = list(payload.get("plan_labels", []) or [])
                    if labels:
                        payload["plan_labels"] = labels[progress_done:progress_done + remaining_games]
                    result_file, progress_file = selfplay_pool.dispatch_task(
                        rank=rank,
                        task_id=task_id,
                        model_state_path=model_state_path,
                        model_state=model_state_cpu,
                        temperature=rl_cfg.get('mcts_temperature'),
                        q_selection_weight=rl_cfg.get('mcts_q_selection_weight'),
                        runtime_overrides=_selfplay_runtime_overrides(),
                        num_games=remaining_games,
                        opponent_payload=payload,
                        stream_results_to_queue=use_queue_transport,
                    )
                    active_workers[rank] = {
                        "chunk_games": int(remaining_games),
                        "progress_file": progress_file,
                        "plan_labels": list(payload.get("plan_labels", []) or []),
                        "payload": payload,
                    }
                    current_progress_files[rank] = progress_file
                    result_files.append(result_file)
                    progress_files.append(progress_file)
                pending.add(rank)

            while pending:
                # Update games progress from .progress files
                total_done = 0
                for rank, _ in worker_specs:
                    pf = current_progress_files.get(rank)
                    try:
                        raw = pf.read_text().strip() if pf is not None else ""
                        active_progress = int(raw) if raw else 0
                    except Exception:
                        active_progress = 0
                    games_reported[rank] = int(completed_games_by_rank.get(rank, 0)) + active_progress
                    total_done += games_reported[rank]
                inc = total_done - games_bar_last
                if inc > 0:
                    games_bar.update(inc)
                    games_bar_last = total_done

                if use_persistent_pool:
                    queue_wait_t0 = time.perf_counter()
                    try:
                        message = selfplay_pool.result_queue.get(timeout=0.2)
                        queue_wait_total_s += float(time.perf_counter() - queue_wait_t0)
                        queue_wait_events += 1
                        if message.get('task_id') == task_id:
                            if message.get('type') == 'payload':
                                packed = message.get('positions')
                                chunk_lengths = list(message.get('game_lengths', []) or [])
                                chunk_stats = message.get('stats', {}) or {}
                                if packed is not None:
                                    boards = packed.get('boards')
                                    policy_indices = packed.get('policy_indices')
                                    policy_values = packed.get('policy_values')
                                    policy_lengths = packed.get('policy_lengths')
                                    importance_scores = packed.get('importance_scores')
                                    policy_weights = packed.get('policy_weights')
                                    value_weights = packed.get('value_weights')
                                    moves_left = packed.get('moves_left')
                                    legal_indices = packed.get('legal_indices')
                                    legal_lengths = packed.get('legal_lengths')
                                    source_codes = packed.get('source_codes')
                                    values = packed.get('values')
                                    chunk_positions = int(packed.get('num_positions', 0) or 0)
                                    if (
                                        replay_buffer is not None
                                        and boards is not None
                                        and policy_indices is not None
                                        and policy_values is not None
                                        and policy_lengths is not None
                                        and values is not None
                                    ):
                                        replay_buffer.add_packed_batch(
                                            boards,
                                            policy_indices,
                                            policy_values,
                                            policy_lengths,
                                            values,
                                            importance_scores=importance_scores,
                                            policy_weights=policy_weights,
                                            value_weights=value_weights,
                                            moves_left=moves_left,
                                            legal_indices=legal_indices,
                                            legal_lengths=legal_lengths,
                                            source_codes=source_codes,
                                        )
                                    queue_total_positions += chunk_positions
                                    if values is not None:
                                        flat_values = values.reshape(-1).float()
                                        queue_total_value_sum += float(flat_values.sum().item())
                                        queue_total_value_sq_sum += float((flat_values * flat_values).sum().item())
                                        queue_total_value_count += int(flat_values.numel())
                                queue_game_lengths.extend(chunk_lengths)
                                queue_total_dropped_positions += int(chunk_stats.get('dropped_positions', 0))
                                queue_total_truncated_games += int(chunk_stats.get('truncated_games', 0))
                                queue_total_claimable_draw_ended_games += int(chunk_stats.get('claimable_draw_ended_games', 0))
                                queue_total_adjudicated_games += int(chunk_stats.get('adjudicated_games', 0))
                                queue_total_syzygy_ended_games += int(chunk_stats.get('syzygy_ended_games', 0))
                                queue_total_syzygy_probe_positions += int(chunk_stats.get('syzygy_probe_positions', 0))
                                queue_total_syzygy_probe_hits += int(chunk_stats.get('syzygy_probe_hits', 0))
                                queue_total_completed_length_sum += int(chunk_stats.get('completed_length_sum', 0))
                                queue_total_truncated_length_sum += int(chunk_stats.get('truncated_length_sum', 0))
                                queue_total_completed_white_wins += int(chunk_stats.get('completed_white_wins', 0))
                                queue_total_completed_black_wins += int(chunk_stats.get('completed_black_wins', 0))
                                queue_total_completed_draws += int(chunk_stats.get('completed_draws', 0))
                                queue_total_decisive_games += int(chunk_stats.get('decisive_games', 0))
                                queue_total_decisive_length_sum += int(chunk_stats.get('decisive_length_sum', 0))
                                queue_total_curriculum_dropped_positions += int(chunk_stats.get('curriculum_dropped_positions', 0))
                                queue_total_cap_dropped_positions += int(chunk_stats.get('cap_dropped_positions', 0))
                                queue_total_resigned_games += int(chunk_stats.get('resigned_games', 0))
                                queue_search_simulations_used_sum += int(chunk_stats.get('search_simulations_used_sum', 0))
                                queue_search_simulations_budget_sum += int(chunk_stats.get('search_simulations_budget_sum', 0))
                                queue_search_samples += int(chunk_stats.get('search_samples', 0))
                                queue_search_simulations_used_samples.extend(list(chunk_stats.get('search_simulations_used_samples', []) or []))
                                queue_search_simulations_budget_samples.extend(list(chunk_stats.get('search_simulations_budget_samples', []) or []))
                                _accumulate_mcts_quality_stats(queue_mcts_quality_stats, chunk_stats)
                                _accumulate_profile_stats(queue_profile_stats, chunk_stats.get('profile', {}) or {})
                                source_counts = dict(chunk_stats.get('opponent_source_counts', {}) or {})
                                if source_counts:
                                    for label, count in source_counts.items():
                                        queue_opponent_source_games[str(label)] += int(count)
                                else:
                                    opponent_source = str(chunk_stats.get('opponent_source', 'current'))
                                    chunk_total_games = int(chunk_stats.get('total_games', len(chunk_lengths) or 0))
                                    if chunk_total_games > 0:
                                        queue_opponent_source_games[opponent_source] += chunk_total_games
                                source_results = dict(chunk_stats.get('opponent_source_results', {}) or {})
                                if source_results:
                                    for label, stats in source_results.items():
                                        result_stats = queue_opponent_source_results[str(label)]
                                        result_stats["wins"] += int((stats or {}).get("wins", 0))
                                        result_stats["draws"] += int((stats or {}).get("draws", 0))
                                        result_stats["losses"] += int((stats or {}).get("losses", 0))
                                        result_stats["games"] += int((stats or {}).get("games", 0))
                                else:
                                    learner_wins = int(chunk_stats.get('learner_wins', 0))
                                    learner_draws = int(chunk_stats.get('learner_draws', 0))
                                    learner_losses = int(chunk_stats.get('learner_losses', 0))
                                    learner_total = learner_wins + learner_draws + learner_losses
                                    if learner_total > 0:
                                        opponent_source = str(chunk_stats.get('opponent_source', 'current'))
                                        result_stats = queue_opponent_source_results[opponent_source]
                                        result_stats["wins"] += learner_wins
                                        result_stats["draws"] += learner_draws
                                        result_stats["losses"] += learner_losses
                                        result_stats["games"] += learner_total
                                continue
                            rank = message.get('rank')
                            completed_games_by_rank[int(rank)] = int(
                                completed_games_by_rank.get(int(rank), 0)
                            ) + int(message.get('games', 0) or 0)
                            try:
                                progress_file = current_progress_files.get(int(rank))
                                if progress_file is not None:
                                    progress_file.unlink(missing_ok=True)
                            except Exception:
                                pass
                            active_workers.pop(int(rank), None)
                            if not message.get('ok', False):
                                interrupted = bool(message.get('interrupt', False))
                                if interrupted:
                                    print("\nCtrl+C detected in self-play worker. Stopping workers...")
                                    raise RLTrainingInterrupted("self-play")
                                raise RuntimeError(
                                    f"Persistent self-play worker {rank} failed: {message.get('error', 'unknown error')}"
                                )
                            if dynamic_dispatch_enabled and _dispatch_next_chunk(int(rank)):
                                pending.add(int(rank))
                                continue
                            pending.discard(rank)
                    except queue.Empty:
                        queue_wait_total_s += float(time.perf_counter() - queue_wait_t0)
                        queue_wait_events += 1
                        pass

                    for rank in list(pending):
                        proc = selfplay_pool.processes.get(rank)
                        if proc is None or proc.is_alive():
                            continue
                        interrupted = _is_interrupt_exit_code(proc.exitcode)
                        if interrupted:
                            print("\nCtrl+C detected in self-play worker. Stopping workers...")
                            raise RLTrainingInterrupted("self-play")
                        _redispatch_after_worker_crash(rank, proc.exitcode)
                else:
                    for rank in list(pending):
                        idx = rank_to_index[rank]
                        p = processes[idx]
                        p.join(timeout=0.2)
                        if p.is_alive():
                            continue
                        pending.remove(rank)
                        if _is_interrupt_exit_code(p.exitcode):
                            interrupted = True
                            print("\nCtrl+C detected in self-play worker. Stopping workers...")
                            raise RLTrainingInterrupted("self-play")
                if pending:
                    time.sleep(0.05)

            games_bar.n = num_games
            games_bar.refresh()
        finally:
            games_bar.close()
            for pf in progress_files:
                try:
                    pf.unlink(missing_ok=True)
                except Exception:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C detected during self-play. Stopping workers...")
        if use_persistent_pool:
            _shutdown_selfplay_pool()
        raise RLTrainingInterrupted("self-play")
    except Exception:
        interrupted = True
        if use_persistent_pool:
            _shutdown_selfplay_pool()
        raise
    finally:
        if interrupted and not use_persistent_pool:
            _terminate_workers(processes)
        if model_state_path is not None and not use_queue_transport:
            try:
                model_state_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    selfplay_time = time.time() - start_time
    selfplay_startup_time = max(0.0, float((startup_done_time or start_time) - start_time))
    
    # Collect results
    collection_start = time.time()
    
    all_positions = [] if not stream_to_replay else None
    total_positions = queue_total_positions if use_queue_transport else 0
    game_lengths = list(queue_game_lengths) if use_queue_transport else []
    total_dropped_positions = queue_total_dropped_positions if use_queue_transport else 0
    total_truncated_games = queue_total_truncated_games if use_queue_transport else 0
    total_claimable_draw_ended_games = queue_total_claimable_draw_ended_games if use_queue_transport else 0
    total_adjudicated_games = queue_total_adjudicated_games if use_queue_transport else 0
    total_syzygy_ended_games = queue_total_syzygy_ended_games if use_queue_transport else 0
    total_syzygy_probe_positions = queue_total_syzygy_probe_positions if use_queue_transport else 0
    total_syzygy_probe_hits = queue_total_syzygy_probe_hits if use_queue_transport else 0
    total_completed_length_sum = queue_total_completed_length_sum if use_queue_transport else 0
    total_truncated_length_sum = queue_total_truncated_length_sum if use_queue_transport else 0
    total_completed_white_wins = queue_total_completed_white_wins if use_queue_transport else 0
    total_completed_black_wins = queue_total_completed_black_wins if use_queue_transport else 0
    total_completed_draws = queue_total_completed_draws if use_queue_transport else 0
    total_decisive_games = queue_total_decisive_games if use_queue_transport else 0
    total_decisive_length_sum = queue_total_decisive_length_sum if use_queue_transport else 0
    total_curriculum_dropped_positions = queue_total_curriculum_dropped_positions if use_queue_transport else 0
    total_cap_dropped_positions = queue_total_cap_dropped_positions if use_queue_transport else 0
    total_value_sum = queue_total_value_sum if use_queue_transport else 0.0
    total_value_sq_sum = queue_total_value_sq_sum if use_queue_transport else 0.0
    total_value_count = queue_total_value_count if use_queue_transport else 0
    total_resigned_games = queue_total_resigned_games if use_queue_transport else 0
    total_search_simulations_used_sum = queue_search_simulations_used_sum if use_queue_transport else 0
    total_search_simulations_budget_sum = queue_search_simulations_budget_sum if use_queue_transport else 0
    total_search_samples = queue_search_samples if use_queue_transport else 0
    total_search_simulations_used_samples = list(queue_search_simulations_used_samples) if use_queue_transport else []
    total_search_simulations_budget_samples = list(queue_search_simulations_budget_samples) if use_queue_transport else []
    total_profile_stats = dict(queue_profile_stats) if use_queue_transport else {}
    total_mcts_quality_stats = dict(queue_mcts_quality_stats) if use_queue_transport else defaultdict(float)
    if use_queue_transport:
        avg_queue_wait_ms = 1000.0 * float(queue_wait_total_s) / float(max(1, queue_wait_events))
        total_profile_stats['queue_wait_time_ms'] = float(avg_queue_wait_ms)
        total_profile_stats['queue_wait_total_s'] = float(queue_wait_total_s)
        total_profile_stats['queue_wait_events'] = int(queue_wait_events)
    else:
        total_profile_stats['queue_wait_time_ms'] = 0.0
        total_profile_stats['queue_wait_total_s'] = 0.0
        total_profile_stats['queue_wait_events'] = 0
    opponent_source_games = dict(queue_opponent_source_games) if use_queue_transport else {}
    opponent_source_results = (
        {label: dict(stats) for label, stats in queue_opponent_source_results.items()}
        if use_queue_transport
        else {}
    )
    
    if not use_queue_transport:
        for idx, result_file in enumerate(result_files):
            if result_file.exists():
                try:
                    with open(result_file, 'rb') as f:
                        while True:
                            try:
                                payload = pickle.load(f)
                            except EOFError:
                                break
                            if isinstance(payload, tuple) and len(payload) == 3:
                                positions, lengths, stats = payload
                            else:
                                raise ValueError(f"Unexpected self-play result payload format: {type(payload).__name__}")

                            total_positions += len(positions)
                            for pos in positions:
                                try:
                                    v = float(pos[3].reshape(-1)[0].item())
                                except Exception:
                                    continue
                                total_value_sum += v
                                total_value_sq_sum += v * v
                                total_value_count += 1
                            if stream_to_replay:
                                for position in positions:
                                    replay_buffer.add(position)
                            else:
                                all_positions.extend(positions)
                            game_lengths.extend(lengths)
                            total_dropped_positions += int((stats or {}).get('dropped_positions', 0))
                            total_truncated_games += int((stats or {}).get('truncated_games', 0))
                            total_claimable_draw_ended_games += int((stats or {}).get('claimable_draw_ended_games', 0))
                            total_adjudicated_games += int((stats or {}).get('adjudicated_games', 0))
                            total_syzygy_ended_games += int((stats or {}).get('syzygy_ended_games', 0))
                            total_syzygy_probe_positions += int((stats or {}).get('syzygy_probe_positions', 0))
                            total_syzygy_probe_hits += int((stats or {}).get('syzygy_probe_hits', 0))
                            total_completed_length_sum += int((stats or {}).get('completed_length_sum', 0))
                            total_truncated_length_sum += int((stats or {}).get('truncated_length_sum', 0))
                            total_completed_white_wins += int((stats or {}).get('completed_white_wins', 0))
                            total_completed_black_wins += int((stats or {}).get('completed_black_wins', 0))
                            total_completed_draws += int((stats or {}).get('completed_draws', 0))
                            total_decisive_games += int((stats or {}).get('decisive_games', 0))
                            total_decisive_length_sum += int((stats or {}).get('decisive_length_sum', 0))
                            total_curriculum_dropped_positions += int((stats or {}).get('curriculum_dropped_positions', 0))
                            total_cap_dropped_positions += int((stats or {}).get('cap_dropped_positions', 0))
                            total_resigned_games += int((stats or {}).get('resigned_games', 0))
                            total_search_simulations_used_sum += int((stats or {}).get('search_simulations_used_sum', 0))
                            total_search_simulations_budget_sum += int((stats or {}).get('search_simulations_budget_sum', 0))
                            total_search_samples += int((stats or {}).get('search_samples', 0))
                            total_search_simulations_used_samples.extend(list((stats or {}).get('search_simulations_used_samples', []) or []))
                            total_search_simulations_budget_samples.extend(list((stats or {}).get('search_simulations_budget_samples', []) or []))
                            _accumulate_mcts_quality_stats(total_mcts_quality_stats, stats or {})
                            _accumulate_profile_stats(total_profile_stats, (stats or {}).get('profile', {}) or {})
                            source_counts = dict((stats or {}).get('opponent_source_counts', {}) or {})
                            if source_counts:
                                for label, count in source_counts.items():
                                    label = str(label)
                                    opponent_source_games[label] = int(opponent_source_games.get(label, 0)) + int(count)
                            else:
                                opponent_source = str((stats or {}).get('opponent_source', 'current'))
                                chunk_total_games = int((stats or {}).get('total_games', len(lengths) or 0))
                                if chunk_total_games > 0:
                                    opponent_source_games[opponent_source] = int(opponent_source_games.get(opponent_source, 0)) + chunk_total_games
                            source_results = dict((stats or {}).get('opponent_source_results', {}) or {})
                            if source_results:
                                for label, source_stat in source_results.items():
                                    result_stats = opponent_source_results.setdefault(
                                        str(label),
                                        {"wins": 0, "draws": 0, "losses": 0, "games": 0},
                                    )
                                    result_stats["wins"] += int((source_stat or {}).get('wins', 0))
                                    result_stats["draws"] += int((source_stat or {}).get('draws', 0))
                                    result_stats["losses"] += int((source_stat or {}).get('losses', 0))
                                    result_stats["games"] += int((source_stat or {}).get('games', 0))
                            else:
                                learner_wins = int((stats or {}).get('learner_wins', 0))
                                learner_draws = int((stats or {}).get('learner_draws', 0))
                                learner_losses = int((stats or {}).get('learner_losses', 0))
                                learner_total = learner_wins + learner_draws + learner_losses
                                if learner_total > 0:
                                    opponent_source = str((stats or {}).get('opponent_source', 'current'))
                                    result_stats = opponent_source_results.setdefault(
                                        opponent_source,
                                        {"wins": 0, "draws": 0, "losses": 0, "games": 0},
                                    )
                                    result_stats["wins"] += learner_wins
                                    result_stats["draws"] += learner_draws
                                    result_stats["losses"] += learner_losses
                                    result_stats["games"] += learner_total
                    result_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to load results from worker {idx}: {e}")
            else:
                print(f"Warning: Worker {idx} result file not found")
    
    collection_time = time.time() - collection_start
    total_time = time.time() - start_time
    
    avg_length = np.mean(game_lengths) if game_lengths else 0
    filtered_positions = int(total_curriculum_dropped_positions + total_cap_dropped_positions)
    total_generated_positions = total_positions + total_dropped_positions + filtered_positions
    replay_positions_per_sec = total_positions / total_time if total_time > 0 else 0.0
    played_positions_per_sec = total_generated_positions / total_time if total_time > 0 else 0.0
    # Backward-compatible return value: callers historically called this
    # positions_per_sec, but it has always meant positions retained for replay.
    positions_per_sec = replay_positions_per_sec
    kept_ratio = (
        100.0 * total_positions / total_generated_positions
        if total_generated_positions > 0
        else 0.0
    )
    completed_games = max(0, len(game_lengths) - total_truncated_games)
    decisive_games = int(total_decisive_games)
    completed_draw_rate = (
        float(total_completed_draws) / float(completed_games)
        if completed_games > 0
        else 0.0
    )
    decisive_rate = (
        float(decisive_games) / float(completed_games)
        if completed_games > 0
        else 0.0
    )
    total_auto_draw_ended_games = int(total_claimable_draw_ended_games)
    auto_draw_rate = (
        float(total_auto_draw_ended_games) / float(max(1, len(game_lengths)))
        if game_lengths
        else 0.0
    )
    truncated_rate = (
        float(total_truncated_games) / float(max(1, len(game_lengths)))
        if game_lengths
        else 0.0
    )
    decisive_avg_length = (
        float(total_decisive_length_sum) / float(decisive_games)
        if decisive_games > 0
        else 0.0
    )
    avg_game_value = (total_value_sum / total_value_count) if total_value_count > 0 else 0.0
    value_var = (total_value_sq_sum / total_value_count) - (avg_game_value ** 2) if total_value_count > 0 else 0.0
    value_std = math.sqrt(max(0.0, value_var))
    mcts_quality_samples = int(float(total_mcts_quality_stats.get('mcts_prior_agreement_samples', 0.0) or 0.0))
    mcts_q_delta_samples = int(float(total_mcts_quality_stats.get('mcts_q_delta_samples', 0.0) or 0.0))
    mcts_q_delta_values = list(total_mcts_quality_stats.get('mcts_q_delta_values', []) or [])
    mcts_q_delta_hist = list(total_mcts_quality_stats.get('mcts_q_delta_hist', []) or [])
    if sum(int(v or 0) for v in mcts_q_delta_hist) <= 0 and mcts_q_delta_values:
        mcts_q_delta_hist = _q_delta_histogram(mcts_q_delta_values)
    mcts_changed_q_delta_samples = int(float(total_mcts_quality_stats.get('mcts_changed_q_delta_samples', 0.0) or 0.0))
    mcts_changed_q_delta_values = list(total_mcts_quality_stats.get('mcts_changed_q_delta_values', []) or [])
    mcts_changed_q_delta_hist = list(total_mcts_quality_stats.get('mcts_changed_q_delta_hist', []) or [])
    if sum(int(v or 0) for v in mcts_changed_q_delta_hist) <= 0 and mcts_changed_q_delta_values:
        mcts_changed_q_delta_hist = _q_delta_histogram(mcts_changed_q_delta_values)
    mcts_phase_stats = {}
    for phase in ('opening', 'middlegame', 'endgame'):
        samples = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_samples', 0.0) or 0.0)
        changed = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_changed_count', 0.0) or 0.0)
        mcts_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
        mcts_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
        mcts_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
    selfplay_stats = {
        'startup_time': float(selfplay_startup_time),
        'completed_games': int(completed_games),
        'completed_white_wins': int(total_completed_white_wins),
        'completed_black_wins': int(total_completed_black_wins),
        'completed_draws': int(total_completed_draws),
        'completed_draw_rate': float(completed_draw_rate),
        'avg_game_value': float(avg_game_value),
        'value_std': float(value_std),
        'truncated_games': int(total_truncated_games),
        'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
        'adjudicated_games': int(total_adjudicated_games),
        'syzygy_ended_games': int(total_syzygy_ended_games),
        'syzygy_probe_positions': int(total_syzygy_probe_positions),
        'syzygy_probe_hits': int(total_syzygy_probe_hits),
        'decisive_games': int(decisive_games),
        'decisive_rate': float(decisive_rate),
        'auto_draw_rate': float(auto_draw_rate),
        'truncated_rate': float(truncated_rate),
        'decisive_avg_length': float(decisive_avg_length),
        'curriculum_dropped_positions': int(total_curriculum_dropped_positions),
        'cap_dropped_positions': int(total_cap_dropped_positions),
        'resigned_games': int(total_resigned_games),
        'replay_positions_per_sec': float(replay_positions_per_sec),
        'played_positions_per_sec': float(played_positions_per_sec),
        # One completed simulation/visit is one root->leaf selection followed
        # by terminal/NN evaluation and backpropagation. Do not multiply it by
        # path length: that separate diagnostic is a traversal counter.
        'mcts_simulations_per_sec': (
            float(total_search_simulations_used_sum) / float(total_time)
            if total_time > 0.0
            else 0.0
        ),
        'mcts_nn_evaluations_per_sec': (
            float(total_profile_stats.get('mcts_nn_inference_batch_items', 0) or 0) / float(total_time)
            if total_time > 0.0
            else 0.0
        ),
        'mcts_selection_node_traversals_per_sec': (
            float(total_profile_stats.get('mcts_selection_node_traversals', 0) or 0) / float(total_time)
            if total_time > 0.0
            else 0.0
        ),
        'search_simulations_used_avg': float(total_search_simulations_used_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_simulations_budget_avg': float(total_search_simulations_budget_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_simulations_budget_p10': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 10)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_p50': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 50)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_p90': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 90)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_min': float(np.min(np.asarray(total_search_simulations_budget_samples, dtype=np.float32))) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_max': float(np.max(np.asarray(total_search_simulations_budget_samples, dtype=np.float32))) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_target': float(
            rl_cfg.get('mcts_dynamic_budget_target_avg', rl_cfg.get('mcts_simulations', 0))
            if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False))
            else rl_cfg.get('mcts_simulations', 0)
        ),
        'search_simulations_used_p10': float(np.percentile(np.asarray(total_search_simulations_used_samples, dtype=np.float32), 10)) if total_search_simulations_used_samples else 0.0,
        'search_samples': int(total_search_samples),
        'mcts_prior_agreement_samples': int(mcts_quality_samples),
        'mcts_prior_agreement_rate': (
            float(total_mcts_quality_stats.get('mcts_prior_agreement_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_prior_changed_rate': (
            float(total_mcts_quality_stats.get('mcts_prior_changed_count', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        **mcts_phase_stats,
        'mcts_policy_uptake_weight_mean': (
            float(total_mcts_quality_stats.get('mcts_policy_uptake_weight_sum', 0.0) or 0.0)
            / float(total_mcts_quality_stats.get('mcts_policy_uptake_samples', 0.0) or 0.0)
            if float(total_mcts_quality_stats.get('mcts_policy_uptake_samples', 0.0) or 0.0) > 0.0
            else 1.0
        ),
        'mcts_policy_uptake_low_rate': (
            float(total_mcts_quality_stats.get('mcts_policy_uptake_low_count', 0.0) or 0.0)
            / float(total_mcts_quality_stats.get('mcts_policy_uptake_samples', 0.0) or 0.0)
            if float(total_mcts_quality_stats.get('mcts_policy_uptake_samples', 0.0) or 0.0) > 0.0
            else 0.0
        ),
        'mcts_prior_top_visit_prob_mean': (
            float(total_mcts_quality_stats.get('mcts_prior_top_visit_prob_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_top_prior_prob_mean': (
            float(total_mcts_quality_stats.get('mcts_top_prior_prob_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_policy_kl_mean': (
            float(total_mcts_quality_stats.get('mcts_policy_kl_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_top_visit_prob_mean': (
            float(total_mcts_quality_stats.get('mcts_top_visit_prob_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_visit_gap_mean': (
            float(total_mcts_quality_stats.get('mcts_visit_gap_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_visit_entropy_mean': (
            float(total_mcts_quality_stats.get('mcts_visit_entropy_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_good_target_rate': (
            float(total_mcts_quality_stats.get('mcts_good_target_count', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_explored_prior_mass_mean': (
            float(total_mcts_quality_stats.get('mcts_explored_prior_mass_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_visited_move_count_mean': (
            float(total_mcts_quality_stats.get('mcts_visited_move_count_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_legal_move_count_mean': (
            float(total_mcts_quality_stats.get('mcts_legal_move_count_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_visit_coverage_ratio_mean': (
            float(total_mcts_quality_stats.get('mcts_visit_coverage_ratio_sum', 0.0) or 0.0) / float(mcts_quality_samples)
            if mcts_quality_samples > 0
            else 0.0
        ),
        'mcts_q_delta_samples': int(mcts_q_delta_samples),
        'mcts_q_delta_mean': (
            float(total_mcts_quality_stats.get('mcts_q_delta_sum', 0.0) or 0.0) / float(mcts_q_delta_samples)
            if mcts_q_delta_samples > 0
            else 0.0
        ),
        'mcts_q_delta_p10': (
            _q_delta_percentile_from_histogram(mcts_q_delta_hist, 10)
            if sum(int(v or 0) for v in mcts_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_q_delta_p50': (
            _q_delta_percentile_from_histogram(mcts_q_delta_hist, 50)
            if sum(int(v or 0) for v in mcts_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_q_delta_p90': (
            _q_delta_percentile_from_histogram(mcts_q_delta_hist, 90)
            if sum(int(v or 0) for v in mcts_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_changed_to_lower_q_rate': (
            float(total_mcts_quality_stats.get('mcts_changed_to_lower_q_count', 0.0) or 0.0) / float(mcts_q_delta_samples)
            if mcts_q_delta_samples > 0
            else 0.0
        ),
        'mcts_changed_to_higher_q_count': int(total_mcts_quality_stats.get('mcts_changed_to_higher_q_count', 0.0) or 0.0),
        'mcts_changed_q_delta_samples': int(mcts_changed_q_delta_samples),
        'mcts_changed_q_delta_mean': (
            float(total_mcts_quality_stats.get('mcts_changed_q_delta_sum', 0.0) or 0.0) / float(mcts_changed_q_delta_samples)
            if mcts_changed_q_delta_samples > 0
            else 0.0
        ),
        'mcts_changed_q_delta_p10': (
            _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 10)
            if sum(int(v or 0) for v in mcts_changed_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_changed_q_delta_p50': (
            _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 50)
            if sum(int(v or 0) for v in mcts_changed_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_changed_q_delta_p90': (
            _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 90)
            if sum(int(v or 0) for v in mcts_changed_q_delta_hist) > 0
            else 0.0
        ),
        'mcts_changed_to_higher_q_rate': (
            float(total_mcts_quality_stats.get('mcts_changed_to_higher_q_count', 0.0) or 0.0) / float(mcts_changed_q_delta_samples)
            if mcts_changed_q_delta_samples > 0
            else 0.0
        ),
        'mcts_changed_to_lower_q_when_changed_rate': (
            float(total_mcts_quality_stats.get('mcts_changed_to_lower_q_count', 0.0) or 0.0) / float(mcts_changed_q_delta_samples)
            if mcts_changed_q_delta_samples > 0
            else 0.0
        ),
        'opponent_source_counts': {
            str(label): int(count)
            for label, count in sorted(opponent_source_games.items(), key=lambda item: item[0])
        },
        'opponent_results': {
            str(label): {
                'wins': int((stats or {}).get('wins', 0)),
                'draws': int((stats or {}).get('draws', 0)),
                'losses': int((stats or {}).get('losses', 0)),
                'games': int((stats or {}).get('games', 0)),
                'score_rate': float(_safe_score_rate(
                    (stats or {}).get('wins', 0),
                    (stats or {}).get('draws', 0),
                    (stats or {}).get('losses', 0),
                ) or 0.0),
            }
            for label, stats in sorted(opponent_source_results.items(), key=lambda item: item[0])
        },
        'opponent_debug': dict(opponent_debug or {}),
        'profile': dict(total_profile_stats),
    }
    
    return (
        all_positions or [],
        total_positions,
        avg_length,
        positions_per_sec,
        selfplay_time,
        collection_time,
        selfplay_stats,
    )


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Chess RL trainer | config: {compact_path(config_path, 3)}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = normalize_config(yaml.safe_load(f))

    debug_cfg = config.get('debug', {}) or {}
    rl_debug_cfg = debug_cfg.get('rl', {}) or {}
    if not isinstance(rl_debug_cfg, dict):
        rl_debug_cfg = {}
    debug_enabled = bool(debug_cfg.get('enabled', False))
    profile_training_enabled = bool(
        debug_enabled and rl_debug_cfg.get('profile_training', debug_cfg.get('profile_training', False))
    )
    log_gpu_memory_enabled = bool(
        debug_enabled and rl_debug_cfg.get('log_gpu_memory', debug_cfg.get('log_gpu_memory', False))
    )

    try:
        syzygy_bootstrap = ensure_syzygy_tables(config, chess_dir=script_dir.parent, logger=print)
        syzygy_status = describe_syzygy_status(config, chess_dir=script_dir.parent)
        syzygy_paths = syzygy_status.get('paths', [])
        syzygy_file_count = int(syzygy_status.get('wdl_files', 0))
        if syzygy_bootstrap.get('enabled', False):
            if int(syzygy_bootstrap.get('downloaded_files', 0)) > 0:
                size_mb = float(syzygy_bootstrap.get('downloaded_bytes', 0)) / 1024 / 1024
                print(
                    f"Syzygy auto-download complete: {int(syzygy_bootstrap.get('downloaded_files', 0))} files, "
                    f"{size_mb:.1f} MB -> {syzygy_bootstrap.get('destination')}"
                )
            elif syzygy_file_count > 0:
                print(f"Syzygy ready: {syzygy_file_count} WDL file(s) in {', '.join(str(p) for p in syzygy_paths)}")
            else:
                print("Syzygy enabled, but no local WDL files found yet.")
    except Exception as exc:
        print(f"Syzygy auto-download skipped: {exc}")

    if not MCTS_SELFPLAY_AVAILABLE:
        raise RuntimeError(
            "RL requires MCTS self-play backend, but it is not available "
            "(failed to import src.batch_selfplay.play_games_mcts_worker)."
        )

    rl_cfg = config.get('reinforcement_learning', {})
    replay_multiplier = rl_cfg.get('replay_buffer_multiplier', 0)
    try:
        replay_multiplier = float(replay_multiplier)
    except Exception:
        replay_multiplier = 0
    if replay_multiplier and replay_multiplier > 0:
        games_per_iter = rl_cfg.get('games_per_iteration', 0)
        try:
            games_per_iter = int(games_per_iter)
        except Exception:
            games_per_iter = 0
        batch_size = rl_cfg.get('batch_size', 1024)
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = 1024
        replay_capacity_round_to = max(1, int(rl_cfg.get('replay_buffer_capacity_round_to', 256)))
        replay_cap_min_positions = max(1, int(rl_cfg.get('replay_cap_min_positions', 16)))
        replay_buffer_min_size = int(
            rl_cfg.get(
                'replay_buffer_min_size',
                max(batch_size * 4, games_per_iter * replay_cap_min_positions),
            )
        )
        replay_buffer_min_size = max(1, replay_buffer_min_size)

        bootstrap_positions = rl_cfg.get('replay_buffer_bootstrap_positions_per_iteration', 'auto')
        if isinstance(bootstrap_positions, str) and bootstrap_positions.strip().lower() == 'auto':
            bootstrap_positions = max(batch_size, games_per_iter * replay_cap_min_positions)
        else:
            try:
                bootstrap_positions = int(bootstrap_positions)
            except Exception:
                bootstrap_positions = max(batch_size, games_per_iter * replay_cap_min_positions)
        bootstrap_positions = max(1, int(bootstrap_positions))
        computed_size = _round_replay_capacity(
            max(replay_buffer_min_size, bootstrap_positions * replay_multiplier),
            replay_capacity_round_to,
        )
        rl_cfg['replay_buffer_bootstrap_positions_per_iteration_resolved'] = int(bootstrap_positions)
        rl_cfg['replay_buffer_size'] = computed_size
    else:
        raise ValueError("replay_buffer_multiplier must be > 0 (replay_buffer_size is no longer used).")

    # Self-play save frequency (multiplier of games_per_iteration)
    save_mult = rl_cfg.get('self_play_save_every_games', 0)
    try:
        save_mult = float(save_mult)
    except Exception:
        save_mult = 0
    if save_mult and save_mult > 0:
        games_per_iter = rl_cfg.get('games_per_iteration', 0)
        try:
            games_per_iter = int(games_per_iter)
        except Exception:
            games_per_iter = 0
        save_every = max(1, int(round(games_per_iter * save_mult)))
        rl_cfg['self_play_save_every_games_resolved'] = save_every
    else:
        rl_cfg['self_play_save_every_games_resolved'] = 0

    model_version = config.get('model', {}).get('version', 'v?.?')
    model_file_tag = build_model_file_tag(config)
    model_architecture = build_model_architecture_metadata(config)
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = torch.device(config['hardware']['device'])
    if profile_training_enabled or log_gpu_memory_enabled:
        print(
            "RL debug timing: "
            f"profile_training={profile_training_enabled}, "
            f"log_gpu_memory={log_gpu_memory_enabled}"
        )
    
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    rl_dir = base_dir / config['paths']['rl_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    use_amp = config['hardware'].get('use_amp', True)
    
    if use_bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported")
            use_bfloat16 = False
    
    best_model_il_path = base_dir / config['paths']['best_model_il']
    best_model_il_swa_path = best_model_il_path.parent / "best_model_il_swa.pt"
    rl_init_checkpoint_path = (
        best_model_il_swa_path if best_model_il_swa_path.exists() else best_model_il_path
    )
    best_model_rl_path = base_dir / config['paths']['best_model_rl']
    version_best_model_path = rl_dir / f"{model_file_tag}_best.pt"
    candidate_checkpoint_path = rl_dir / f"{model_file_tag}_candidate.pt"
    latest_checkpoint_path = rl_dir / f"{model_file_tag}_latest.pt"
    total_iterations = int(config['reinforcement_learning']['iterations'])

    startup_choice = plan_rl_startup(
        model=None,
        device=device,
        models_dir=models_dir,
        best_model_rl_path=best_model_rl_path,
        rl_dir=rl_dir,
        default_new_checkpoint=rl_init_checkpoint_path,
    )

    # Windows spawn imports Python and torch in every worker. Launch the stable
    # persistent topology after the quick startup choice so process imports can
    # overlap model construction and checkpoint preparation.
    rl_cfg = config['reinforcement_learning']
    early_plan = _resolve_selfplay_worker_plan(config, int(rl_cfg['games_per_iteration']))
    if bool(rl_cfg.get('persistent_self_play_workers', True)):
        if early_plan['central_inference']:
            early_temp_dir = Path(tempfile.gettempdir()) / "chess_selfplay_mcts"
            early_temp_dir.mkdir(exist_ok=True)
            get_or_create_selfplay_pool(
                config,
                early_plan['worker_specs'],
                early_plan['device_type'],
                early_temp_dir,
            )
            print(f"Initializing model and {len(early_plan['worker_specs'])} self-play workers...")

    config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)

    resolve_rl_hyperparameters(
        config=config,
        model=model,
        device=device,
        base_dir=base_dir,
    )

    startup_plan = plan_rl_startup(
        model=model,
        device=device,
        models_dir=models_dir,
        best_model_rl_path=best_model_rl_path,
        rl_dir=rl_dir,
        default_new_checkpoint=rl_init_checkpoint_path,
        initial_plan=startup_choice,
    )

    # Initialize logger after startup selection.
    logger = TrainingLogger(
        logs_dir,
        experiment_name=build_rl_experiment_name(config),
        mode="rl",
        config_snapshot={
            "model": config.get("model", {}),
            "reinforcement_learning": config.get("reinforcement_learning", {}),
        },
        verbose=False,
    )
    global _LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG
    _LAST_RUN_LOG_CSV = logger.csv_path
    _LAST_RUN_LOG_PNG = logger.plot_path

    elo_coordinator = RLEloCoordinator(
        model=model,
        config=config,
        device=device,
        logger=logger,
        checkpoint_targets=[best_model_rl_path, version_best_model_path],
    )
    rl_optimizer_groups, _rl_optimizer_summary = _build_rl_optimizer_param_groups(
        model,
        config['reinforcement_learning'],
        config['reinforcement_learning']['learning_rate'],
    )
    optimizer = optim.AdamW(
        rl_optimizer_groups,
        lr=config['reinforcement_learning']['learning_rate'],
        fused=True if torch.cuda.is_available() else False
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    startup_state = apply_rl_startup_plan(
        startup_plan=startup_plan,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        default_new_checkpoint=rl_init_checkpoint_path,
    )
    start_mode = startup_state.get("start_mode", "new")
    selected_checkpoint_label = startup_state.get("selected_checkpoint_label")
    start_iteration = int(startup_state.get("start_iteration", 0) or 0)
    resumed_best_win_rate = float(startup_state.get("best_win_rate", 0.0) or 0.0)
    selected_compatibility_ratio = startup_state.get("selected_compatibility_ratio")
    transfer_match_ratio = startup_state.get("transfer_match_ratio")
    new_init_mode = startup_state.get("new_init_mode", startup_plan.get("new_init_mode", "default"))
    selected_checkpoint_path = startup_state.get("selected_checkpoint")
    best_candidate_score_lower_bound = -1.0
    if start_mode == "new":
        with contextlib.suppress(OSError):
            candidate_checkpoint_path.unlink(missing_ok=True)
    elif candidate_checkpoint_path.exists():
        with contextlib.suppress(Exception):
            candidate_payload = torch.load(
                candidate_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            best_candidate_score_lower_bound = float(
                (candidate_payload or {}).get('eval_score_lower_bound', -1.0) or -1.0
            )

    if start_mode == "new":
        if new_init_mode == "scratch":
            source_label = "random weights"
        elif new_init_mode == "select" and selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name
        elif rl_init_checkpoint_path.exists():
            source_label = f"{rl_init_checkpoint_path.name} (init)"
        else:
            source_label = "new (scratch)"
    else:
        source_label = "selected checkpoint"
        if selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name

    elo_seed_checkpoint_path = None
    if start_mode == "new":
        if new_init_mode == "select" and selected_checkpoint_path is not None:
            elo_seed_checkpoint_path = selected_checkpoint_path
        elif new_init_mode == "default":
            elo_seed_checkpoint_path = rl_init_checkpoint_path
    elif start_mode in {"resume", "transfer"} and selected_checkpoint_path is not None:
        elo_seed_checkpoint_path = selected_checkpoint_path
    if elo_seed_checkpoint_path is not None:
        _seed_rl_plot_elo_from_checkpoint(
            logger,
            elo_seed_checkpoint_path,
            base_dir=models_dir,
        )

    if start_mode == "new":
        run_context = "startup: new RL training"
        if new_init_mode == "select" and selected_checkpoint_label:
            ratio = transfer_match_ratio
            if ratio is None:
                ratio = selected_compatibility_ratio
            if ratio is None:
                run_context = f"{run_context} | init={selected_checkpoint_label}"
            else:
                run_context = (
                    f"{run_context} | init={selected_checkpoint_label} | "
                    f"compatibility {ratio * 100:.2f}%"
                )
    elif start_mode == "resume":
        run_context = (
            f"startup: resumed full state, next iteration {start_iteration + 1}, "
            f"best score_rate={resumed_best_win_rate:.2%}"
        )
    else:
        ratio = transfer_match_ratio
        if ratio is None:
            ratio = selected_compatibility_ratio
        if ratio is None:
            run_context = "startup: transferred matching weights, compatibility n/a"
        else:
            run_context = f"startup: transferred matching weights, compatibility {ratio * 100:.2f}%"
    if selected_checkpoint_label and start_mode in {"resume", "transfer"}:
        run_context = f"{run_context} | source={selected_checkpoint_label}"
    logger.set_run_context(run_context)
    best_win_rate_so_far = resumed_best_win_rate

    # Disable repeated model summary prints.
    config['model'] = {**config.get('model', {}), 'print_summary': False}
    best_model = ChessNet(config).to(device)
    best_model = best_model.to(memory_format=torch.channels_last)
    best_model.load_state_dict(model.state_dict())
    anchor_model = None
    anchor_model_available = False
    if bool(config['reinforcement_learning'].get('anchor_eval_enabled', False)) and rl_init_checkpoint_path.exists():
        try:
            anchor_model = ChessNet(config).to(device)
            anchor_model = anchor_model.to(memory_format=torch.channels_last)
            checkpoint = load_checkpoint_file(str(rl_init_checkpoint_path), device)
            model_state = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None
            if not isinstance(model_state, dict):
                raise KeyError("missing model_state_dict")
            normalized_state = normalize_state_dict_keys(model_state, target_keys=set(anchor_model.state_dict().keys()))
            try:
                anchor_model.load_state_dict(normalized_state)
            except Exception:
                transfer_matching_weights(anchor_model, normalized_state)
            anchor_model.eval()
            anchor_model.requires_grad_(False)
            anchor_model_available = True
        except Exception as exc:
            anchor_model = None
            print(f"Anchor eval disabled: failed to load IL best ({exc})")
    anchor_eval_unlocked = bool(
        anchor_model_available
        and anchor_model is not None
        and not _models_have_identical_state(best_model, anchor_model)
    )
    rl_cfg = config['reinforcement_learning']
    base_mcts_q_selection_weight = max(
        0.0,
        float(rl_cfg.get('mcts_q_selection_weight', 0.0) or 0.0),
    )
    
    # Best files are updated only when evaluation confirms model improvement.
    replay_fp16 = config['reinforcement_learning'].get('replay_fp16', False)
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
    replay_buffer = ReplayBuffer(
        config['reinforcement_learning']['replay_buffer_size'],
        max_policy_targets=replay_max_policy_targets,
        use_fp16=replay_fp16,
    )
    replay_capacity_round_to = max(1, int(rl_cfg.get('replay_buffer_capacity_round_to', 256)))
    replay_buffer_min_size = max(1, int(rl_cfg.get('replay_buffer_min_size', replay_buffer.max_size)))
    replay_buffer_ema_alpha = max(
        0.0,
        min(1.0, float(rl_cfg.get('replay_buffer_ema_alpha', 0.25))),
    )
    replay_positions_ema = None
    
    # LR schedule (same shape as IL, but stepped per RL iteration)
    use_lr_schedule = config['reinforcement_learning'].get('use_lr_schedule', False)
    lr_base = float(config['reinforcement_learning']['learning_rate'])
    warmup_pct = float(config['reinforcement_learning'].get('warmup_pct', 0.05))
    min_lr_ratio = float(config['reinforcement_learning'].get('min_lr_ratio', 0.25))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_iters = int(math.ceil(total_iterations * warmup_pct)) if warmup_pct > 0.0 else 0
    if use_lr_schedule and warmup_pct > 0.0 and total_iterations > 1:
        # Need at least two scheduled points so one logged iteration is visibly below base LR.
        warmup_iters = max(2, warmup_iters)
    warmup_iters = min(max(0, warmup_iters), max(1, total_iterations))

    def _compute_lr(iter_idx):
        if not use_lr_schedule:
            return lr_base
        # Linear warmup: 0 -> base_lr over warmup_iters.
        if iter_idx < warmup_iters:
            return lr_base * ((iter_idx + 1) / warmup_iters)

        # Cosine decay: base_lr -> base_lr * min_lr_ratio.
        progress = (iter_idx - warmup_iters) / max(1, total_iterations - warmup_iters)
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_base * multiplier

    def _compute_value_loss_weight(iter_idx):
        rl_cfg_local = config.get('reinforcement_learning', {})
        default_weight = float(rl_cfg_local.get('value_loss_weight', 1.0))
        schedule = rl_cfg_local.get('value_loss_weight_schedule', None)
        if not isinstance(schedule, dict):
            return default_weight

        try:
            start_w = float(schedule.get('start', default_weight))
            end_w = float(schedule.get('end', default_weight))
            power = max(0.1, float(schedule.get('power', 1.0)))
        except Exception:
            return default_weight

        if total_iterations <= 1:
            return end_w

        progress = max(0.0, min(1.0, iter_idx / max(1, total_iterations - 1)))
        shaped_progress = progress ** power
        return start_w + (end_w - start_w) * shaped_progress

    if start_iteration >= total_iterations:
        print(
            f"Resume start iteration ({start_iteration + 1}) exceeds configured total "
            f"({total_iterations}). Nothing to train."
        )
        logger.plot()
        logger.plot_rl_performance()
        logger.plot_rl_data_quality()
        return

    fixed_mcts_temperature_threshold = int(rl_cfg.get('mcts_temperature_threshold', 16))
    fixed_mcts_dirichlet_weight = float(rl_cfg.get('mcts_dirichlet_weight', 0.25))
    score_rate_threshold = float(rl_cfg.get('score_rate_threshold', 0.55))
    true_win_rate_threshold = float(rl_cfg.get('true_win_rate_threshold', 0.0))
    promotion_candidate_streak = 0
    base_mcts_simulations = max(1, int(rl_cfg.get('mcts_simulations', 160) or 160))

    def _resolve_eval_stage_simulations(exact_key, multiplier_key, default_multiplier):
        exact_value = rl_cfg.get(exact_key, None)
        if exact_value not in (None, '', 'auto'):
            return max(1, int(exact_value))
        multiplier = max(0.05, float(rl_cfg.get(multiplier_key, default_multiplier) or default_multiplier))
        return max(1, int(round(base_mcts_simulations * multiplier)))

    funnel_preliminary_games = max(1, int(rl_cfg.get('eval_funnel_preliminary_games', 50)))
    funnel_preliminary_simulations = _resolve_eval_stage_simulations(
        'eval_funnel_preliminary_simulations',
        'eval_funnel_preliminary_simulations_multiplier',
        1.0,
    )
    funnel_medium_games = max(0, int(rl_cfg.get('eval_funnel_medium_games', 80)))
    funnel_medium_simulations = _resolve_eval_stage_simulations(
        'eval_funnel_medium_simulations',
        'eval_funnel_later_simulations_multiplier',
        1.5,
    )
    funnel_advanced_games = max(0, int(rl_cfg.get('eval_funnel_advanced_games', 120)))
    funnel_advanced_simulations = _resolve_eval_stage_simulations(
        'eval_funnel_advanced_simulations',
        'eval_funnel_later_simulations_multiplier',
        1.5,
    )
    funnel_preliminary_score_rate = float(rl_cfg.get('eval_funnel_preliminary_score_rate', 0.45))
    funnel_preliminary_true_win_rate = float(rl_cfg.get('eval_funnel_preliminary_true_win_rate', 0.0))
    funnel_preliminary_result_weight = max(0.0, float(rl_cfg.get('eval_funnel_preliminary_result_weight', 0.0) or 0.0))
    funnel_medium_score_rate = float(rl_cfg.get('eval_funnel_medium_score_rate', 0.53))
    funnel_medium_true_win_rate = float(rl_cfg.get('eval_funnel_medium_true_win_rate', 0.0))
    anchor_eval_simulations = _resolve_eval_stage_simulations(
        'anchor_eval_mcts_simulations',
        'anchor_eval_mcts_simulations_multiplier',
        1.5,
    )
    eval_every = max(1, int(rl_cfg.get('eval_every', 1) or 1))
    no_mcts_eval_enabled = bool(rl_cfg.get('eval_no_mcts_enabled', True))
    no_mcts_eval_every = max(1, int(rl_cfg.get('eval_no_mcts_every', 1)))
    no_mcts_eval_games = max(1, int(rl_cfg.get('eval_no_mcts_games', 30)))
    early_stop_enabled = bool(rl_cfg.get('early_stop_enabled', True))
    early_stop_patience = max(1, int(rl_cfg.get('early_stop_patience', 5)))
    early_stop_patience_increment_on_promotion = max(
        0,
        int(rl_cfg.get('early_stop_patience_increment_on_promotion', 1) or 0),
    )
    current_early_stop_patience = early_stop_patience
    early_stop_min_score_improvement = float(rl_cfg.get('early_stop_min_score_improvement', 0.01))
    early_stop_min_true_win_improvement = float(rl_cfg.get('early_stop_min_true_win_improvement', 0.005))
    post_promotion_stabilization_iters = max(
        0,
        int(rl_cfg.get('post_promotion_stabilization_iterations', 0) or 0),
    )
    post_promotion_lr_scale = max(
        0.05,
        min(1.0, float(rl_cfg.get('post_promotion_lr_scale', 1.0) or 1.0)),
    )
    post_promotion_lr_recovery_iters = max(
        0,
        int(rl_cfg.get('post_promotion_lr_recovery_iterations', 0) or 0),
    )
    diagnostic_ema_alpha = max(
        0.0,
        min(1.0, float(rl_cfg.get('rl_diagnostic_ema_alpha', 0.35) or 0.35)),
    )
    post_promotion_stabilization_remaining = 0
    post_promotion_lr_recovery_remaining = 0
    last_promotion_iteration = None
    mcts_no_mcts_gap_ema = None
    eval_score_rate_ema = None
    eval_true_win_rate_ema = None
    best_eval_score_seen = None
    best_eval_true_win_seen = None
    no_improvement_eval_streak = 0
    training_interrupted = False
    interrupted_stage = None
    last_logged_iteration = start_iteration if start_iteration > 0 else None

    funnel_stages = (
        f"{funnel_preliminary_games}g@{funnel_preliminary_simulations}"
        f" -> {funnel_medium_games}g@{funnel_medium_simulations}"
        f" -> {funnel_advanced_games}g@{funnel_advanced_simulations}"
    )
    _print_rl_startup_plan(
        config=config,
        model=model,
        logger=logger,
        source_label=source_label,
        start_mode=start_mode,
        start_iteration=start_iteration,
        total_iterations=total_iterations,
        worker_plan=early_plan,
        replay_max_policy_targets=replay_max_policy_targets,
        funnel_stages=funnel_stages,
        warmup_iters=warmup_iters,
        min_lr_ratio=min_lr_ratio,
        use_amp=use_amp,
        use_bfloat16=use_bfloat16,
    )

    try:
        for iteration in range(start_iteration, total_iterations):
            print(f"\n[Iteration {iteration + 1}/{total_iterations}] self-play -> replay -> train")
            replay_buffer.set_current_iteration(iteration + 1)
            iteration_profile_printed = False
            iteration_stage_times = {}
            iteration_start_time = time.perf_counter()
            iteration_stage_start = iteration_start_time
            iteration_eval_substage_times = {}
            iteration_notes = []
            promotion_status = "not evaluated"

            if device.type == 'cuda' and torch.cuda.is_available() and (profile_training_enabled or log_gpu_memory_enabled):
                torch.cuda.reset_peak_memory_stats(device)

            def _profile_sync():
                if profile_training_enabled and device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            def _finish_stage(stage_name):
                nonlocal iteration_stage_start
                _profile_sync()
                now = time.perf_counter()
                stage_key = str(stage_name)
                iteration_stage_times[stage_key] = float(iteration_stage_times.get(stage_key, 0.0)) + (now - iteration_stage_start)
                iteration_stage_start = now

            def _start_eval_substage():
                _profile_sync()
                return time.perf_counter()

            def _finish_eval_substage(stage_name, started_at):
                _profile_sync()
                elapsed = max(0.0, time.perf_counter() - float(started_at))
                stage_key = str(stage_name)
                iteration_eval_substage_times[stage_key] = float(
                    iteration_eval_substage_times.get(stage_key, 0.0)
                ) + elapsed

            def _emit_iteration_profile():
                nonlocal iteration_profile_printed
                if iteration_profile_printed or not (profile_training_enabled or log_gpu_memory_enabled):
                    return
                _profile_sync()
                total_time_s = time.perf_counter() - iteration_start_time if profile_training_enabled else None
                gpu_stats = _cuda_memory_stats(device) if log_gpu_memory_enabled else None
                _print_rl_iteration_profile(
                    iteration + 1,
                    total_iterations,
                    iteration_stage_times if profile_training_enabled else None,
                    total_time_s,
                    gpu_stats=gpu_stats,
                    include_stage_times=profile_training_enabled,
                )
                iteration_profile_printed = True
            
            # Update learning rate (cosine decay + warmup)
            current_lr = _compute_lr(iteration)
            post_promotion_stabilizing = post_promotion_stabilization_remaining > 0
            post_promotion_lr_factor, post_promotion_lr_phase = _post_promotion_recovery_scale(
                post_promotion_stabilization_remaining,
                post_promotion_stabilization_iters,
                post_promotion_lr_recovery_remaining,
                post_promotion_lr_recovery_iters,
                post_promotion_lr_scale,
            )
            current_lr *= post_promotion_lr_factor
            for group in optimizer.param_groups:
                group['lr'] = current_lr * float(group.get('lr_factor', 1.0))
            if post_promotion_stabilizing:
                iteration_notes.append(
                    f"post-promotion stabilization: {post_promotion_stabilization_remaining} iter left, "
                    f"LR x{post_promotion_lr_factor:.2f}"
                )
                post_promotion_stabilization_remaining -= 1
            elif post_promotion_lr_phase == "recover":
                iteration_notes.append(
                    f"post-promotion LR recovery: {post_promotion_lr_recovery_remaining} iter left, "
                    f"LR x{post_promotion_lr_factor:.2f}"
                )
                post_promotion_lr_recovery_remaining -= 1
            
            # Fixed search parameters. Evaluation is telemetry, never a control signal.
            current_temp = float(config['reinforcement_learning']['mcts_temperature'])
            current_temp_threshold = fixed_mcts_temperature_threshold
            current_dirichlet_weight = fixed_mcts_dirichlet_weight
            current_mcts_q_selection_weight = float(base_mcts_q_selection_weight)
            current_mcts_effective_q_weight = current_mcts_q_selection_weight
            config['reinforcement_learning']['mcts_temperature'] = current_temp
            config['reinforcement_learning']['mcts_dirichlet_weight'] = current_dirichlet_weight
            config['reinforcement_learning']['mcts_temperature_threshold'] = current_temp_threshold
            config['reinforcement_learning']['mcts_q_selection_weight'] = current_mcts_q_selection_weight
            
            rl_cfg['current_iteration'] = int(iteration + 1)

            current_value_loss_weight = _compute_value_loss_weight(iteration)
            current_policy_loss_weight = float(rl_cfg.get('policy_loss_weight', 1.0))

            _finish_stage('setup')

            # Self-play with MCTS
            model.eval()
            best_model_state_for_selfplay = None
            if bool(rl_cfg.get('self_play_opponent_pool_enabled', False)):
                best_model_state_for_selfplay = _snapshot_model_state_cpu(best_model, share_memory=True)
            performance_profile = {}
            positions, positions_added, avg_game_length, positions_per_sec, selfplay_time, collection_time, selfplay_stats = \
                play_games_parallel_mcts(
                    model,
                    config,
                    device,
                    config['reinforcement_learning']['games_per_iteration'],
                    replay_buffer=replay_buffer,
                    best_model_state=best_model_state_for_selfplay,
                )
            selfplay_stats = dict(selfplay_stats or {})
            selfplay_stats['mcts_dirichlet_weight'] = float(current_dirichlet_weight)
            _finish_stage('selfplay')
            startup_time_in_selfplay = max(
                0.0,
                float((selfplay_stats or {}).get('startup_time', 0.0) or 0.0),
            )
            if startup_time_in_selfplay > 0.0:
                current_selfplay_stage = max(0.0, float(iteration_stage_times.get('selfplay', 0.0) or 0.0))
                transfer_time = min(current_selfplay_stage, startup_time_in_selfplay)
                if transfer_time > 0.0:
                    iteration_stage_times['selfplay'] = float(current_selfplay_stage - transfer_time)
                    iteration_stage_times['setup'] = float(iteration_stage_times.get('setup', 0.0) or 0.0) + float(transfer_time)
             
            # Add to replay buffer
            for position in positions:
                replay_buffer.add(position)

            if positions_added > 0:
                observed_positions = float(positions_added)
                if replay_positions_ema is None:
                    replay_positions_ema = observed_positions
                else:
                    replay_positions_ema = (
                        (1.0 - replay_buffer_ema_alpha) * float(replay_positions_ema)
                        + replay_buffer_ema_alpha * observed_positions
                    )
                target_capacity = _round_replay_capacity(
                    max(
                        replay_buffer_min_size,
                        float(replay_positions_ema) * float(replay_multiplier),
                    ),
                    replay_capacity_round_to,
                )
                if replay_buffer.resize(target_capacity):
                    rl_cfg['replay_buffer_size'] = int(replay_buffer.max_size)
                    iteration_notes.append(
                        f"replay resized to {replay_buffer.max_size:,} "
                        f"(EMA {float(replay_positions_ema):.0f} positions/iter)"
                    )
             
            replay_quality_stats = replay_buffer.quality_stats(
                recent_window_fraction=config['reinforcement_learning'].get('replay_recent_window_fraction', None)
            )
            selfplay_profile = dict((selfplay_stats or {}).get('profile', {}) or {})
            if profile_training_enabled and selfplay_profile:
                print_selfplay_profiler(selfplay_profile, selfplay_time)
            performance_profile = dict(selfplay_profile)
            for metric_name in (
                'replay_positions_per_sec',
                'played_positions_per_sec',
                'mcts_simulations_per_sec',
                'mcts_nn_evaluations_per_sec',
                'mcts_selection_node_traversals_per_sec',
            ):
                performance_profile[metric_name] = float(
                    (selfplay_stats or {}).get(metric_name, 0.0) or 0.0
                )
            _finish_stage('replay')

            avg_policy_entropy = 0.0
            avg_value_pred_std = 0.0
            avg_target_value_std = 0.0
            sample_age_avg = None
            sample_age_p10 = None
            sample_age_p50 = None
            sample_age_p90 = None
            sample_age_new_fraction = None
            sample_age_le1_fraction = None
            
            # Training with metrics
            replay_size = len(replay_buffer)
            base_batch_size = _resolve_train_batch_size(
                replay_size,
                config['reinforcement_learning'],
            )
            total_train_steps = 0
            selected_samples = 0
            if replay_size > 0 and base_batch_size > 0:
                model.train()
                total_loss = 0
                total_policy = 0
                total_value = 0
                total_policy_entropy = 0
                total_value_pred_std = 0
                total_target_value_std = 0
                sample_age_batches = []
                
                # Initialize metrics calculator
                metrics_calc = MetricsCalculator()
                
                train_epochs = max(
                    1,
                    int(config['reinforcement_learning']['train_epochs_per_iteration']),
                )
                replay_steps_per_epoch = max(1, math.ceil(replay_size / base_batch_size))
                uncapped_train_steps = train_epochs * replay_steps_per_epoch
                total_train_steps = uncapped_train_steps
                max_train_steps = max(
                    0,
                    int(config['reinforcement_learning'].get('train_max_steps_per_iteration', 0) or 0),
                )
                if max_train_steps > 0 and total_train_steps > max_train_steps:
                    iteration_notes.append(f"training capped: {total_train_steps} -> {max_train_steps} steps")
                    total_train_steps = max_train_steps

                # Select once per bounded replay pass. Sampling each optimizer batch
                # independently caused silent repeats and omissions inside an epoch.
                training_index_batches = []
                remaining_train_steps = total_train_steps
                for _ in range(train_epochs):
                    if remaining_train_steps <= 0:
                        break
                    epoch_steps = min(replay_steps_per_epoch, remaining_train_steps)
                    epoch_sample_count = min(replay_size, epoch_steps * base_batch_size)
                    epoch_indices = replay_buffer.select_indices(epoch_sample_count)
                    training_index_batches.extend(
                        epoch_indices[start:start + base_batch_size]
                        for start in range(0, int(epoch_indices.size), base_batch_size)
                    )
                    remaining_train_steps -= epoch_steps
                total_train_steps = len(training_index_batches)
                selected_samples = sum(int(indices.size) for indices in training_index_batches)
                replay_quality_stats['train_batch_size'] = int(base_batch_size)
                replay_quality_stats['train_steps'] = int(total_train_steps)
                replay_quality_stats['train_selected_samples'] = int(selected_samples)
                replay_quality_stats['train_replay_coverage'] = (
                    float(selected_samples) / float(replay_size)
                    if replay_size > 0
                    else 0.0
                )
                for batch_indices in tqdm(
                    training_index_batches,
                    desc="Training",
                ):
                    batch = replay_buffer.sample_from_indices(batch_indices)
                    last_sample_ages = getattr(replay_buffer, 'last_sample_ages', None)
                    if last_sample_ages is not None and getattr(last_sample_ages, "size", 0) > 0:
                        sample_age_batches.append(last_sample_ages.copy())
                    loss, policy_loss, value_loss, policy_entropy, value_pred_std, target_value_std = train_on_batch_rl(
                        model,
                        optimizer,
                        batch,
                        config,
                        device,
                        scaler,
                        metrics_calc,
                        value_weight_override=current_value_loss_weight,
                        policy_weight_override=current_policy_loss_weight,
                        anchor_model=anchor_model if anchor_model_available else None,
                    )
                    
                    total_loss += loss
                    total_policy += policy_loss
                    total_value += value_loss
                    total_policy_entropy += policy_entropy
                    total_value_pred_std += value_pred_std
                    total_target_value_std += target_value_std
                avg_policy = total_policy / total_train_steps
                avg_value = total_value / total_train_steps
                avg_loss = total_loss / total_train_steps
                avg_policy_entropy = total_policy_entropy / total_train_steps
                avg_value_pred_std = total_value_pred_std / total_train_steps
                avg_target_value_std = total_target_value_std / total_train_steps
                if sample_age_batches:
                    sample_ages = np.concatenate(sample_age_batches).astype(np.float32, copy=False)
                    sample_age_avg = float(np.mean(sample_ages))
                    sample_age_p10 = float(np.percentile(sample_ages, 10))
                    sample_age_p50 = float(np.percentile(sample_ages, 50))
                    sample_age_p90 = float(np.percentile(sample_ages, 90))
                    sample_age_new_fraction = float(np.mean(sample_ages <= 0.0))
                    sample_age_le1_fraction = float(np.mean(sample_ages <= 1.0))
                
                # Compute metrics
                train_metrics = metrics_calc.compute()
                
            else:
                avg_loss = avg_policy = avg_value = 0
                train_metrics = {}
            if sample_age_avg is not None:
                replay_quality_stats['sample_age_avg'] = sample_age_avg
                replay_quality_stats['sample_age_p10'] = sample_age_p10
                replay_quality_stats['sample_age_p50'] = sample_age_p50
                replay_quality_stats['sample_age_p90'] = sample_age_p90
                replay_quality_stats['sample_age_new_fraction'] = sample_age_new_fraction
                replay_quality_stats['sample_age_le1_fraction'] = sample_age_le1_fraction
            _finish_stage('train')
             
            # Evaluation
            score_rate = None
            true_win_rate = None
            eval_stage = None
            eval_games_total = None
            eval_wins = eval_draws = eval_losses = eval_unresolved = None
            score_lower_bound = None
            anchor_score_rate = None
            anchor_score_lower_bound = None
            anchor_eval_games_total = None
            anchor_true_win_rate = None
            anchor_wins = anchor_draws = anchor_losses = None
            anchor_no_mcts_games_total = None
            anchor_no_mcts_score_rate = None
            anchor_mcts_no_mcts_gap = None
            estimated_elo = None
            no_mcts_score_rate = None
            no_mcts_true_win_rate = None
            no_mcts_draw_rate = None
            no_mcts_loss_rate = None
            no_mcts_wins = no_mcts_draws = no_mcts_losses = no_mcts_unresolved = None
            promoted_best_this_iter = False
            early_stop_reset_reason = None
            early_stop_should_stop = False
            should_run_no_mcts_eval = no_mcts_eval_enabled and ((iteration + 1) % no_mcts_eval_every == 0)
            if should_run_no_mcts_eval:
                regular_eval_t0 = _start_eval_substage()
                model.eval()
                if device.type == 'cuda' and torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                no_mcts_stats = evaluate_models_no_mcts(
                    model,
                    best_model,
                    config,
                    device,
                    no_mcts_eval_games,
                    use_fixed_openings=bool(rl_cfg.get('eval_no_mcts_use_fixed_openings', True)),
                )
                no_mcts_score_rate = float((no_mcts_stats or {}).get('score_rate', 0.0))
                no_mcts_true_win_rate = float((no_mcts_stats or {}).get('win_rate', 0.0))
                no_mcts_draw_rate = float((no_mcts_stats or {}).get('draw_rate', 0.0))
                no_mcts_loss_rate = float((no_mcts_stats or {}).get('loss_rate', 0.0))
                no_mcts_wins = int((no_mcts_stats or {}).get('wins', 0))
                no_mcts_draws = int((no_mcts_stats or {}).get('draws', 0))
                no_mcts_losses = int((no_mcts_stats or {}).get('losses', 0))
                no_mcts_unresolved = int((no_mcts_stats or {}).get('unresolved', 0))
                _finish_eval_substage('regular_eval', regular_eval_t0)
            iteration_number = iteration + 1
            should_run_mcts_eval = (iteration_number % eval_every) == 0
            if should_run_mcts_eval:
                promotion_eval_t0 = _start_eval_substage()
                eval_subject_model = model
                eval_game_index_offset = 0
                eval_subject_model.eval()
                if device.type == 'cuda' and torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                funnel_runtime_games = min(
                    games
                    for games in (
                        funnel_preliminary_games,
                        funnel_medium_games,
                        funnel_advanced_games,
                    )
                    if games > 0
                )
                with eval_central_inference_runtime(
                    eval_subject_model,
                    best_model,
                    config,
                    device,
                    funnel_runtime_games,
                ) as eval_runtime:
                    eval_stats, eval_stage = _evaluate_models_funnel(
                        model=eval_subject_model,
                        best_model=best_model,
                        config=config,
                        device=device,
                        preliminary_games=funnel_preliminary_games,
                        preliminary_simulations=funnel_preliminary_simulations,
                        medium_games=funnel_medium_games,
                        medium_simulations=funnel_medium_simulations,
                        advanced_games=funnel_advanced_games,
                        advanced_simulations=funnel_advanced_simulations,
                        preliminary_score_rate=funnel_preliminary_score_rate,
                        preliminary_true_win_rate=funnel_preliminary_true_win_rate,
                        medium_score_rate=funnel_medium_score_rate,
                        medium_true_win_rate=funnel_medium_true_win_rate,
                        preliminary_result_weight=funnel_preliminary_result_weight,
                        game_index_offset=eval_game_index_offset,
                        use_fixed_openings=bool(rl_cfg.get('eval_fixed_openings_enabled', True)),
                        central_runtime=eval_runtime,
                    )
                _finish_eval_substage('promotion_eval', promotion_eval_t0)
                score_rate = float((eval_stats or {}).get('score_rate', 0.0))
                true_win_rate = float((eval_stats or {}).get('win_rate', 0.0))
                eval_games_total = int((eval_stats or {}).get('num_games', 0) or 0)
                eval_wins = int((eval_stats or {}).get('wins', 0))
                eval_draws = int((eval_stats or {}).get('draws', 0))
                eval_losses = int((eval_stats or {}).get('losses', 0))
                eval_unresolved = int((eval_stats or {}).get('unresolved', 0))
                current_mcts_no_mcts_gap = (
                    float(score_rate) - float(no_mcts_score_rate)
                    if no_mcts_score_rate is not None
                    else None
                )
                if current_mcts_no_mcts_gap is not None:
                    mcts_no_mcts_gap_ema = (
                        current_mcts_no_mcts_gap
                        if mcts_no_mcts_gap_ema is None
                        else (
                            (1.0 - diagnostic_ema_alpha) * float(mcts_no_mcts_gap_ema)
                            + diagnostic_ema_alpha * current_mcts_no_mcts_gap
                        )
                    )
                eval_score_rate_ema = (
                    float(score_rate)
                    if eval_score_rate_ema is None
                    else (
                        (1.0 - diagnostic_ema_alpha) * float(eval_score_rate_ema)
                        + diagnostic_ema_alpha * float(score_rate)
                    )
                )
                eval_true_win_rate_ema = (
                    float(true_win_rate)
                    if eval_true_win_rate_ema is None
                    else (
                        (1.0 - diagnostic_ema_alpha) * float(eval_true_win_rate_ema)
                        + diagnostic_ema_alpha * float(true_win_rate)
                    )
                )
                improved_score = (
                    best_eval_score_seen is None
                    or float(score_rate) >= float(best_eval_score_seen) + early_stop_min_score_improvement
                )
                improved_true_win = (
                    best_eval_true_win_seen is None
                    or float(true_win_rate) >= float(best_eval_true_win_seen) + early_stop_min_true_win_improvement
                )
                is_new_best_candidate = score_rate >= score_rate_threshold and true_win_rate >= true_win_rate_threshold
                promotion_status = (
                    "candidate passed score/win gate"
                    if is_new_best_candidate
                    else (
                        f"hold: score/win gate (score {score_rate:.1%}/{score_rate_threshold:.1%}, "
                        f"true win {true_win_rate:.1%}/{true_win_rate_threshold:.1%})"
                    )
                )
                promotion_stat_gate_enabled = bool(rl_cfg.get('promotion_stat_gate_enabled', True))
                stat_gate_z = max(0.0, float(rl_cfg.get('promotion_stat_gate_z', 1.28) or 1.28))
                stat_gate_games = max(1.0, float(eval_games_total or 1))
                score_rate_clamped = max(0.0, min(1.0, float(score_rate)))
                score_lower_bound = _score_rate_lower_bound(
                    score_rate_clamped,
                    stat_gate_games,
                    stat_gate_z,
                )
                if is_new_best_candidate and promotion_stat_gate_enabled:
                    stat_gate_min_score_lb = float(
                        rl_cfg.get('promotion_score_lower_bound_min', 0.50) or 0.50
                    )
                    if score_lower_bound < stat_gate_min_score_lb:
                        is_new_best_candidate = False
                        promotion_status = (
                            f"blocked: confidence LB {score_lower_bound:.1%} < {stat_gate_min_score_lb:.1%}"
                        )
                anchor_gate_enabled = bool(rl_cfg.get('promotion_require_anchor_non_regression', True))
                min_anchor_score_for_progress = float(rl_cfg.get('promotion_anchor_min_score_rate', 0.50))
                min_anchor_true_win_for_progress = float(
                    rl_cfg.get('promotion_anchor_min_true_win_rate', 0.0)
                )
                min_anchor_score_lower_bound = float(
                    rl_cfg.get('promotion_anchor_score_lower_bound_min', 0.47)
                )
                candidate_passed_best_gate = bool(is_new_best_candidate)
                anchor_diagnostic_due = _should_run_anchor_eval(iteration + 1, rl_cfg)
                anchor_candidate_due = bool(candidate_passed_best_gate and anchor_gate_enabled)
                if (
                    anchor_model_available
                    and anchor_model is not None
                    and (anchor_diagnostic_due or anchor_candidate_due)
                ):
                    if _models_have_identical_state(best_model, anchor_model):
                        anchor_score_rate = score_rate
                        anchor_true_win_rate = true_win_rate
                        anchor_wins = eval_wins
                        anchor_draws = eval_draws
                        anchor_losses = eval_losses
                        anchor_eval_games_total = eval_games_total
                        if no_mcts_score_rate is not None:
                            anchor_no_mcts_games_total = sum(int(value or 0) for value in (
                                no_mcts_wins, no_mcts_draws, no_mcts_losses, no_mcts_unresolved
                            ))
                            anchor_no_mcts_score_rate = no_mcts_score_rate
                        iteration_notes.append("anchor result reused: current best is the IL anchor")
                    else:
                        anchor_eval_stage = 'promotion_eval' if anchor_candidate_due else 'regular_eval'
                        anchor_eval_t0 = _start_eval_substage()
                        anchor_eval_config = _build_eval_config_with_exact_simulations(
                            config,
                            anchor_eval_simulations,
                        )
                        anchor_games = max(2, int(rl_cfg.get('anchor_eval_games', 40)))
                        anchor_use_fixed_openings = bool(
                            rl_cfg.get('anchor_eval_use_fixed_openings', True)
                        )
                        if anchor_candidate_due:
                            max_anchor_games = max(anchor_games, _ANCHOR_CANDIDATE_MAX_GAMES)
                            with eval_central_inference_runtime(
                                eval_subject_model,
                                anchor_model,
                                anchor_eval_config,
                                device,
                                max_anchor_games,
                            ) as anchor_runtime:
                                anchor_stats = _evaluate_anchor_candidate_sequential(
                                    model=eval_subject_model,
                                    anchor_model=anchor_model,
                                    eval_config=anchor_eval_config,
                                    device=device,
                                    initial_games=anchor_games,
                                    max_games=max_anchor_games,
                                    min_score_rate=min_anchor_score_for_progress,
                                    min_score_lower_bound=min_anchor_score_lower_bound,
                                    stat_gate_z=stat_gate_z,
                                    game_index_offset=eval_game_index_offset,
                                    use_fixed_openings=anchor_use_fixed_openings,
                                    central_runtime=anchor_runtime,
                                )
                        else:
                            anchor_stats = evaluate_models(
                                eval_subject_model,
                                anchor_model,
                                anchor_eval_config,
                                device,
                                anchor_games,
                                game_index_offset=eval_game_index_offset,
                                use_fixed_openings=anchor_use_fixed_openings,
                                progress_desc="Eval vs anchor",
                            )
                        anchor_score_rate = float((anchor_stats or {}).get('score_rate', 0.0))
                        anchor_true_win_rate = float((anchor_stats or {}).get('win_rate', 0.0))
                        anchor_wins = int((anchor_stats or {}).get('wins', 0))
                        anchor_draws = int((anchor_stats or {}).get('draws', 0))
                        anchor_losses = int((anchor_stats or {}).get('losses', 0))
                        anchor_eval_games_total = int((anchor_stats or {}).get('num_games', 0) or 0)
                        _finish_eval_substage(anchor_eval_stage, anchor_eval_t0)
                        if should_run_no_mcts_eval:
                            anchor_regular_eval_t0 = _start_eval_substage()
                            anchor_no_mcts_stats = evaluate_models_no_mcts(
                                eval_subject_model,
                                anchor_model,
                                config,
                                device,
                                no_mcts_eval_games,
                                game_index_offset=no_mcts_eval_games,
                                use_fixed_openings=bool(
                                    rl_cfg.get('eval_no_mcts_use_fixed_openings', True)
                                ),
                            )
                            anchor_no_mcts_games_total = int(
                                (anchor_no_mcts_stats or {}).get('num_games', 0) or 0
                            )
                            anchor_no_mcts_score_rate = float(
                                (anchor_no_mcts_stats or {}).get('score_rate', 0.0) or 0.0
                            )
                            _finish_eval_substage('regular_eval', anchor_regular_eval_t0)
                    if anchor_score_rate is not None and anchor_no_mcts_score_rate is not None:
                        anchor_mcts_no_mcts_gap = (
                            float(anchor_score_rate) - float(anchor_no_mcts_score_rate)
                        )
                    anchor_score_lower_bound = _score_rate_lower_bound(
                        anchor_score_rate,
                        anchor_eval_games_total,
                        stat_gate_z,
                    )
                    configured_anchor_games = max(2, int(rl_cfg.get('anchor_eval_games', 40)))
                    if anchor_candidate_due and int(anchor_eval_games_total or 0) > configured_anchor_games:
                        iteration_notes.append(
                            f"anchor tiebreak: {anchor_eval_games_total} games -> "
                            f"{anchor_score_rate:.1%} (LB {anchor_score_lower_bound:.1%})"
                        )
                if anchor_score_rate is not None:
                    elo_coordinator.observe_il_anchor_eval(
                        iteration + 1,
                        score_rate=anchor_score_rate,
                        true_win_rate=anchor_true_win_rate,
                    )
                anchor_regression_blocks_progress = bool(
                    anchor_gate_enabled
                    and anchor_model_available
                    and anchor_model is not None
                    and anchor_score_rate is not None
                    and (
                        float(anchor_score_rate) < min_anchor_score_for_progress
                        or float(anchor_true_win_rate or 0.0) < min_anchor_true_win_for_progress
                    )
                )
                reset_reasons = []
                if anchor_regression_blocks_progress:
                    reset_reasons.append('anchor_regression')
                else:
                    if improved_score:
                        best_eval_score_seen = float(score_rate)
                        reset_reasons.append('score_record')
                    elif best_eval_score_seen is None:
                        best_eval_score_seen = float(score_rate)
                    if improved_true_win:
                        best_eval_true_win_seen = float(true_win_rate)
                        reset_reasons.append('true_win_record')
                    elif best_eval_true_win_seen is None:
                        best_eval_true_win_seen = float(true_win_rate)
                if reset_reasons and not anchor_regression_blocks_progress:
                    no_improvement_eval_streak = 0
                    early_stop_reset_reason = '+'.join(reset_reasons)
                else:
                    no_improvement_eval_streak += 1
                    early_stop_reset_reason = '+'.join(reset_reasons) if reset_reasons else 'none'
                early_stop_should_stop = bool(
                    early_stop_enabled
                    and no_improvement_eval_streak >= current_early_stop_patience
                )
                if (
                    is_new_best_candidate
                    and anchor_gate_enabled
                    and anchor_model_available
                    and anchor_model is not None
                ):
                    if anchor_score_rate is None:
                        is_new_best_candidate = False
                        promotion_status = "blocked: anchor evaluation unavailable"
                    else:
                        anchor_gate_failed = bool(
                            float(anchor_score_rate) < min_anchor_score_for_progress
                            or float(anchor_true_win_rate or 0.0) < min_anchor_true_win_for_progress
                            or anchor_score_lower_bound is None
                            or float(anchor_score_lower_bound) < min_anchor_score_lower_bound
                        )
                        if anchor_gate_failed:
                            is_new_best_candidate = False
                            promotion_status = (
                                f"blocked: anchor {float(anchor_score_rate):.1%}, "
                                f"LB {float(anchor_score_lower_bound or 0.0):.1%}"
                            )
                if (
                    candidate_passed_best_gate
                    and not is_new_best_candidate
                    and float(score_lower_bound or 0.0) > float(best_candidate_score_lower_bound)
                ):
                    candidate_metadata = {
                        'candidate_status': promotion_status,
                        'score_rate': score_rate,
                        'win_rate': true_win_rate,
                        'eval_score_rate': score_rate,
                        'eval_true_win_rate': true_win_rate,
                        'eval_score_lower_bound': score_lower_bound,
                        'anchor_score_rate': anchor_score_rate,
                        'anchor_true_win_rate': anchor_true_win_rate,
                        'anchor_score_lower_bound': anchor_score_lower_bound,
                        'mcts_q_selection_weight': current_mcts_q_selection_weight,
                        'mcts_q_effective_weight': current_mcts_effective_q_weight,
                        'policy_loss': avg_policy,
                        'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                        'value_mae': train_metrics.get('value_mae', 0),
                        'version': model_version,
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                    }
                    save_checkpoint(
                        model,
                        None,
                        iteration,
                        avg_loss,
                        str(candidate_checkpoint_path),
                        candidate_metadata,
                        save_optimizer=False,
                        save_dtype=torch.bfloat16 if use_bfloat16 else None,
                    )
                    best_candidate_score_lower_bound = float(score_lower_bound or 0.0)
                    iteration_notes.append(
                        f"preserved candidate (best LB {best_candidate_score_lower_bound:.1%})"
                    )
                if is_new_best_candidate:
                    no_improvement_eval_streak = 0
                    early_stop_reset_reason = (
                        f"{early_stop_reset_reason}+promotion"
                        if early_stop_reset_reason and early_stop_reset_reason != 'none'
                        else 'promotion'
                    )
                    early_stop_should_stop = False
                elif early_stop_reset_reason == 'anchor_regression':
                    iteration_notes.append(
                        f"anchor regression; early-stop {no_improvement_eval_streak}/{current_early_stop_patience}"
                    )
                elif early_stop_reset_reason == 'none':
                    iteration_notes.append(
                        f"no eval improvement; early-stop {no_improvement_eval_streak}/{current_early_stop_patience}"
                    )
                promotion_candidate_streak = 1 if is_new_best_candidate else 0
                estimated_elo = None
                if is_new_best_candidate:
                    elo_eval_t0 = _start_eval_substage()
                    estimated_elo = elo_coordinator.evaluate_promoted_best(iteration + 1)
                    _finish_eval_substage('elo_eval', elo_eval_t0)
                
                # Log with all metrics
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    learning_rate=current_lr,
                    value_loss_weight=current_value_loss_weight,
                    mcts_q_selection_weight=current_mcts_q_selection_weight,
                    mcts_q_effective_weight=current_mcts_effective_q_weight,
                    early_stop_streak=no_improvement_eval_streak,
                    promotion_candidate_streak=promotion_candidate_streak,
                    early_stop_reset_reason=early_stop_reset_reason,
                    eval_score_rate_ema=eval_score_rate_ema,
                    eval_true_win_rate_ema=eval_true_win_rate_ema,
                    mcts_no_mcts_gap=current_mcts_no_mcts_gap,
                    score_rate=score_rate,
                    true_win_rate=true_win_rate,
                    eval_stage=eval_stage,
                    eval_games=eval_games_total,
                    eval_wins=eval_wins,
                    eval_draws=eval_draws,
                    eval_losses=eval_losses,
                    eval_unresolved=eval_unresolved,
                    eval_score_lower_bound=score_lower_bound,
                    eval_stat_gate_z=stat_gate_z,
                    no_mcts_score_rate=no_mcts_score_rate,
                    no_mcts_win_rate=no_mcts_true_win_rate,
                    no_mcts_games=sum(int(value or 0) for value in (
                        no_mcts_wins, no_mcts_draws, no_mcts_losses, no_mcts_unresolved
                    )),
                    anchor_score_rate=anchor_score_rate,
                    anchor_true_win_rate=anchor_true_win_rate,
                    anchor_games=anchor_eval_games_total,
                    anchor_score_lower_bound=anchor_score_lower_bound,
                    anchor_no_mcts_games=anchor_no_mcts_games_total,
                    anchor_no_mcts_score_rate=anchor_no_mcts_score_rate,
                    anchor_mcts_no_mcts_gap=anchor_mcts_no_mcts_gap,
                    temperature=current_temp,
                    rl_best_model=is_new_best_candidate,
                )
                last_logged_iteration = iteration + 1
                if is_new_best_candidate:
                    promoted_best_this_iter = True
                    promotion_status = "PROMOTED: new best model"
                    if early_stop_patience_increment_on_promotion > 0:
                        current_early_stop_patience += early_stop_patience_increment_on_promotion
                        iteration_notes.append(
                            f"early-stop patience increased to {current_early_stop_patience} eval cycles"
                        )
                    post_promotion_stabilization_remaining = max(
                        post_promotion_stabilization_remaining,
                        post_promotion_stabilization_iters,
                    )
                    post_promotion_lr_recovery_remaining = max(
                        post_promotion_lr_recovery_remaining,
                        post_promotion_lr_recovery_iters,
                    )
                    promotion_candidate_streak = 0
                    last_promotion_iteration = int(iteration + 1)
                    best_model.load_state_dict(model.state_dict())
                    best_win_rate_so_far = max(best_win_rate_so_far, float(score_rate))
                    if int(eval_wins or 0) > 0 and anchor_model_available:
                        anchor_eval_unlocked = True
                    
                    model_to_save = model
                    best_metadata = {
                        'win_rate': true_win_rate,
                        'score_rate': score_rate,
                        'eval_true_win_rate': true_win_rate,
                        'anchor_score_rate': anchor_score_rate,
                        'anchor_true_win_rate': anchor_true_win_rate,
                        'mcts_q_selection_weight': current_mcts_q_selection_weight,
                        'mcts_q_effective_weight': current_mcts_effective_q_weight,
                        'policy_loss': avg_policy,
                        'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                        'value_mae': train_metrics.get('value_mae', 0),
                        'version': model_version,
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                    }
                    best_metadata.update(elo_coordinator.metadata_for_iteration(iteration + 1))
                    save_checkpoint(
                        model_to_save, None, iteration, avg_loss,
                        str(best_model_rl_path),
                        best_metadata,
                        save_optimizer=False,
                        save_dtype=torch.bfloat16 if use_bfloat16 else None
                    )
                    
                    size_mb = best_model_rl_path.stat().st_size / (1024**2)
                    iteration_notes.append(f"saved promoted best ({size_mb:.1f} MB)")
                    if version_best_model_path != best_model_rl_path:
                        shutil.copy2(best_model_rl_path, version_best_model_path)
                    with contextlib.suppress(OSError):
                        candidate_checkpoint_path.unlink(missing_ok=True)
                    best_candidate_score_lower_bound = -1.0
                if early_stop_should_stop:
                    print(
                        "Early stopping: no meaningful improvement in eval "
                        f"for {no_improvement_eval_streak} evaluation cycle(s). "
                        "Finishing final logs before stopping."
                    )
            else:
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    estimated_elo=None,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    learning_rate=current_lr,
                    value_loss_weight=current_value_loss_weight,
                    mcts_q_selection_weight=current_mcts_q_selection_weight,
                    mcts_q_effective_weight=current_mcts_effective_q_weight,
                    early_stop_streak=no_improvement_eval_streak,
                    promotion_candidate_streak=promotion_candidate_streak,
                    early_stop_reset_reason=early_stop_reset_reason,
                    eval_score_rate_ema=eval_score_rate_ema,
                    eval_true_win_rate_ema=eval_true_win_rate_ema,
                    temperature=current_temp,
                )
                last_logged_iteration = iteration + 1
            _finish_stage('eval_log')
            eval_region_total = float(iteration_stage_times.pop('eval_log', 0.0) or 0.0)
            measured_eval_total = 0.0
            for stage_name in ('regular_eval', 'promotion_eval', 'elo_eval'):
                stage_seconds = float(iteration_eval_substage_times.get(stage_name, 0.0) or 0.0)
                iteration_stage_times[stage_name] = stage_seconds
                measured_eval_total += stage_seconds
            iteration_stage_times['log'] = max(0.0, eval_region_total - measured_eval_total)

            latest_metadata = {
                'win_rate': true_win_rate,
                'score_rate': score_rate,
                'eval_true_win_rate': true_win_rate,
                'no_mcts_score_rate': no_mcts_score_rate,
                'no_mcts_win_rate': no_mcts_true_win_rate,
                'no_mcts_true_win_rate': no_mcts_true_win_rate,
                'no_mcts_draw_rate': no_mcts_draw_rate,
                'no_mcts_loss_rate': no_mcts_loss_rate,
                'mcts_q_selection_weight': current_mcts_q_selection_weight,
                'mcts_q_effective_weight': current_mcts_effective_q_weight,
                'policy_loss': avg_policy,
                'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                'value_mae': train_metrics.get('value_mae', 0),
                'version': model_version,
                'startup_mode': start_mode,
                'model_architecture': model_architecture,
                'rl_surpassed_il_anchor': bool(elo_coordinator.il_anchor_surpassed),
                'rl_best_il_anchor_score_rate': elo_coordinator.best_il_anchor_score_rate,
                'rl_best_il_anchor_true_win_rate': elo_coordinator.best_il_anchor_true_win_rate,
                'rl_best_il_anchor_iteration': elo_coordinator.best_il_anchor_iteration,
            }
            latest_metadata.update(elo_coordinator.metadata_for_iteration(iteration + 1))
            save_checkpoint(
                model,
                optimizer,
                iteration,
                avg_loss,
                str(latest_checkpoint_path),
                latest_metadata,
                save_optimizer=True,
                save_dtype=torch.bfloat16 if use_bfloat16 else None,
                extra_state={
                    'scaler_state_dict': scaler.state_dict(),
                    'best_win_rate': best_win_rate_so_far,
                },
            )
            
            _finish_stage('checkpoint')
            gc.collect()
            _finish_stage('gc')
            iteration_total_time_s = time.perf_counter() - iteration_start_time
            logger.log_rl_performance(
                iteration + 1,
                replay_positions_per_sec=positions_per_sec,
                iteration_total_time=iteration_total_time_s,
                avg_game_length=avg_game_length,
                profile=performance_profile,
                stage_times=iteration_stage_times,
            )
            replay_quality_stats['iterations_since_promotion'] = (
                int(iteration + 1) - int(last_promotion_iteration)
                if last_promotion_iteration is not None
                else None
            )
            logger.log_rl_data_quality(
                iteration + 1,
                positions_added=positions_added,
                replay_stats=replay_quality_stats,
                selfplay_stats=selfplay_stats,
                train_policy_entropy=avg_policy_entropy,
            )
            logger.plot()
            logger.plot_rl_performance()
            logger.plot_rl_data_quality()

            completed_games = int((selfplay_stats or {}).get('completed_games', 0) or 0)
            completed_draw_rate = float((selfplay_stats or {}).get('completed_draw_rate', 0.0) or 0.0)
            decisive_rate = float((selfplay_stats or {}).get('decisive_rate', 0.0) or 0.0)
            storage_drops = (
                int((selfplay_stats or {}).get('curriculum_dropped_positions', 0) or 0)
                + int((selfplay_stats or {}).get('cap_dropped_positions', 0) or 0)
            )
            storage_keep_rate = float(positions_added) / max(1.0, float(positions_added + storage_drops))

            sims_used = float((selfplay_stats or {}).get('search_simulations_used_avg', 0.0) or 0.0)
            sims_budget = float((selfplay_stats or {}).get('search_simulations_budget_avg', 0.0) or 0.0)
            completed_visits_per_sec = float(
                (selfplay_stats or {}).get('mcts_simulations_per_sec', 0.0) or 0.0
            )
            nn_evaluations_per_sec = float(
                (selfplay_stats or {}).get('mcts_nn_evaluations_per_sec', 0.0) or 0.0
            )
            budget_utilization = sims_used / sims_budget if sims_budget > 0.0 else 0.0
            budget_p10 = float((selfplay_stats or {}).get('search_simulations_budget_p10', 0.0) or 0.0)
            budget_p50 = float((selfplay_stats or {}).get('search_simulations_budget_p50', 0.0) or 0.0)
            budget_p90 = float((selfplay_stats or {}).get('search_simulations_budget_p90', 0.0) or 0.0)
            budget_spread = f"p10/50/90 {budget_p10:.0f}/{budget_p50:.0f}/{budget_p90:.0f}"
            changed_rate = float((selfplay_stats or {}).get('mcts_prior_changed_rate', 0.0) or 0.0)
            useful_rate = changed_rate * float(
                (selfplay_stats or {}).get('mcts_changed_to_higher_q_rate', 0.0) or 0.0
            )
            harmful_rate = changed_rate * float(
                (selfplay_stats or {}).get('mcts_changed_to_lower_q_when_changed_rate', 0.0) or 0.0
            )
            mcts_samples = int((selfplay_stats or {}).get('mcts_prior_agreement_samples', 0) or 0)
            if mcts_samples > 0:
                search_summary = (
                    f"{completed_visits_per_sec:,.0f} completed visits/s, "
                    f"{nn_evaluations_per_sec:,.0f} NN evals/s | "
                    f"{sims_used:.0f}/{sims_budget:.0f} avg sims ({budget_utilization:.0%}), {budget_spread} | "
                    f"top changed {changed_rate:.1%}: useful {useful_rate:.1%}, harmful {harmful_rate:.1%} | "
                    f"KL {float((selfplay_stats or {}).get('mcts_policy_kl_mean', 0.0) or 0.0):.3f} | "
                    f"visited {float((selfplay_stats or {}).get('mcts_visited_move_count_mean', 0.0) or 0.0):.1f}/"
                    f"{float((selfplay_stats or {}).get('mcts_legal_move_count_mean', 0.0) or 0.0):.1f}"
                )
            else:
                search_summary = (
                    f"{completed_visits_per_sec:,.0f} completed visits/s, "
                    f"{nn_evaluations_per_sec:,.0f} NN evals/s | "
                    f"{sims_used:.0f}/{sims_budget:.0f} avg sims ({budget_utilization:.0%}), "
                    f"{budget_spread}; no quality sample"
                )

            replay_coverage = float(replay_quality_stats.get('train_replay_coverage', 0.0) or 0.0)
            train_summary = (
                f"{total_train_steps} x {base_batch_size:,} | replay coverage {replay_coverage:.0%} | "
                f"loss {avg_loss:.3f} (P {avg_policy:.3f}, V {avg_value:.3f}) | "
                f"policy top1/top3 {float(train_metrics.get('policy_top1_acc', 0.0) or 0.0):.1%}/"
                f"{float(train_metrics.get('policy_top3_acc', 0.0) or 0.0):.1%} | LR {current_lr:.2e}"
            )
            value_std_ratio = (
                avg_value_pred_std / avg_target_value_std
                if avg_target_value_std > 1e-8
                else 0.0
            )
            age_summary = "-"
            if sample_age_p50 is not None and sample_age_p90 is not None:
                age_summary = f"{sample_age_p50:.1f}/{sample_age_p90:.1f} iter"
            value_summary = (
                f"MAE {float(train_metrics.get('value_mae', 0.0) or 0.0):.3f} | "
                f"WDL acc {float(train_metrics.get('value_wdl_acc', 0.0) or 0.0):.1%} | "
                f"pred/target std {value_std_ratio:.2f} | replay age p50/p90 {age_summary}"
            )

            if score_rate is not None:
                eval_parts = [
                    f"MCTS {score_rate:.1%} (LB {float(score_lower_bound or 0.0):.1%}, "
                    f"W/D/L {eval_wins}/{eval_draws}/{eval_losses}, {eval_stage})"
                ]
                if no_mcts_score_rate is not None:
                    eval_parts.append(
                        f"NN {no_mcts_score_rate:.1%}; MCTS lift {float(score_rate - no_mcts_score_rate):+.1%}"
                    )
                if anchor_score_rate is not None:
                    eval_parts.append(
                        f"anchor {anchor_score_rate:.1%} (LB {float(anchor_score_lower_bound or 0.0):.1%})"
                    )
                eval_summary = " | ".join(eval_parts)
            elif no_mcts_score_rate is not None:
                until_mcts_eval = eval_every - (iteration_number % eval_every)
                eval_summary = (
                    f"NN-only {no_mcts_score_rate:.1%} (W/D/L {no_mcts_wins}/{no_mcts_draws}/{no_mcts_losses}) "
                    f"| MCTS eval in {until_mcts_eval} iter"
                )
            else:
                until_mcts_eval = eval_every - (iteration_number % eval_every)
                eval_summary = f"not scheduled; MCTS eval in {until_mcts_eval} iter"

            bottleneck, bottleneck_share = dominant_stage(iteration_stage_times)
            stage_summary = " | ".join(
                f"{name} {format_duration(seconds)}"
                for name, seconds in iteration_stage_times.items()
                if float(seconds or 0.0) >= 0.5
            )
            print_panel(
                f"ITERATION {iteration + 1}/{total_iterations} | {format_duration(iteration_total_time_s)}",
                [
                    (
                        "self-play",
                        f"{completed_games} games | +{positions_added:,} positions ({storage_keep_rate:.0%} kept) | "
                        f"replay {positions_per_sec:.1f} pos/s | "
                        f"played {float((selfplay_stats or {}).get('played_positions_per_sec', 0.0) or 0.0):.1f} pos/s | "
                        f"buffer {len(replay_buffer):,}/{replay_buffer.max_size:,} | "
                        f"decisive/draw {decisive_rate:.1%}/{completed_draw_rate:.1%}",
                    ),
                    ("search", search_summary),
                    ("learning", train_summary),
                    ("value", value_summary),
                    ("evaluation", eval_summary),
                    ("decision", promotion_status),
                    (
                        "runtime",
                        f"bottleneck {bottleneck} ({bottleneck_share:.0%}) | {stage_summary}",
                    ),
                ],
                notes=iteration_notes[:4],
            )
            _emit_iteration_profile()
            if early_stop_should_stop:
                break
    except RLTrainingInterrupted as exc:
        training_interrupted = True
        interrupted_stage = str(exc) or "self-play"
    except KeyboardInterrupt:
        training_interrupted = True
        interrupted_stage = "runtime"
    finally:
        _shutdown_selfplay_pool()

    logger.plot()
    logger.plot_rl_performance()
    logger.plot_rl_data_quality()
    if training_interrupted:
        _handle_graceful_interrupt(logger=logger, stage=interrupted_stage)
        return

    final_elo_iteration = int(last_logged_iteration or config['reinforcement_learning'].get('iterations', 0) or 0)
    if final_elo_iteration > 0:
        elo_coordinator.evaluate_final_best(final_elo_iteration, model_override=best_model)
        logger.plot()
        logger.plot_rl_performance()
        logger.plot_rl_data_quality()

    print("\n=== Training complete ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _handle_graceful_interrupt(logger=None, stage="runtime")
    finally:
        _shutdown_selfplay_pool()


