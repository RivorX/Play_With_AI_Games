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
import csv
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
    # Every Windows ``spawn`` child imports NumPy/PyTorch before its worker
    # entry point can call ``torch.set_num_threads(1)``.  Without setting the
    # native runtime limits here, each CPU-only MCTS/eval process creates its
    # own OpenMP/MKL/BLAS pools and reserves hundreds of MiB of avoidable
    # private address space.  The actual parallelism lives across processes
    # and games, so one native thread per child preserves search throughput.
    for _thread_env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_thread_env_name] = "1"
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

from src.config import default_config_path, load_project_config
from src.mcts.q_delta import (
    Q_DELTA_HIST_BINS as _Q_DELTA_HIST_BINS,
    USEFUL_SEARCH_Q_DELTA_MIN,
    q_delta_histogram as _q_delta_histogram,
    q_delta_percentile_from_histogram as _q_delta_percentile_from_histogram,
)

from src.model import (
    create_model,
    load_checkpoint_file,
    normalize_state_dict_keys,
    save_checkpoint,
    transfer_matching_weights,
)
from src.game import backend as chess
from src.mcts.native import get_native_mcts

# Import MCTS self-play
try:
    from src.mcts.search import (
        MultiGameBatchMCTS,
        _build_sparse_policy_target_from_visits,
        _policy_uptake_weight,
        _resolve_dynamic_simulation_budget,
        _resolve_replay_max_policy_targets,
        _search_correction_metadata,
    )
    from src.selfplay.workers import run_selfplay_worker
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("Warning: MCTS self-play not available")

# Import training modules
from src.common.logger import TrainingLogger
from src.common.torch_cache import configure_torch_compile_cache
from src.evaluation.elo_runner import (
    build_final_elo_config,
    build_final_mcts_elo_configs,
    checkpoint_elo_seed,
    elo_protocol_matches_settings,
    run_elo_check,
    safe_float as _safe_float,
)
from src.models.catalog import load_checkpoint_metadata, persist_checkpoint_elo_metadata
from src.training.rl.replay import (
    REPLAY_SOURCE_CHAMPION,
    REPLAY_SOURCE_LEARNER,
    ReplayBuffer,
)
from src.training.rl.correction_audit import (
    CORRECTION_AUDIT_MAX_ROWS,
    correction_audit_metrics,
    evaluate_correction_cohort,
    freeze_replay_batch,
)
from src.training.rl.trainer import (
    close_eval_central_runtime,
    evaluate_models,
    evaluate_models_no_mcts,
    is_transient_cuda_error,
    prepare_eval_central_runtime,
    resolve_policy_correction_rank_loss_weight,
    train_on_batch_rl,
)
from src.training.rl.startup import (
    apply_rl_startup_plan,
    plan_rl_startup,
    select_best_il_checkpoint,
)
from src.training.rl.console import compact_path, dominant_stage, format_duration, print_panel
from src.training.rl.profiler import print_selfplay_profiler
from src.training.rl.persistent_pool import (
    _resolve_central_inference_server_count,
    borrow_selfplay_central_runtime,
    get_or_create_selfplay_pool,
    _shutdown_selfplay_pool,
    _terminate_workers,
    _is_interrupt_exit_code,
)
from src.training.il.auto_tune import resolve_rl_hyperparameters
from src.common.metrics import MetricsCalculator
from src.common.runtime import (
    build_rl_experiment_name,
    build_model_file_tag,
    build_model_architecture_metadata,
    cleanup_interrupted_log_csv,
)
from src.common.syzygy import ensure_syzygy_tables, describe_syzygy_status


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None
# Keep one replay pass per iteration, but avoid collapsing it into a handful of
# huge-batch updates.  The rl34 replay pass took only ~38 s inside an ~8.5 min
# iteration, while 34 giant-batch steps needed roughly twenty iterations to
# turn strong targets into a promotable model.  Forty-eight updates preserve
# sample coverage and add little compute, but make each iteration a materially
# larger optimization step.
_TARGET_OPTIMIZER_STEPS_PER_REPLAY_PASS = 48
# Anchor decides whether a promoted checkpoint may replace a long-lived safety
# baseline.  A close result is therefore allowed to continue to 256 games; the
# sequential gate still stops obvious wins/losses after the initial batch.
_ANCHOR_CANDIDATE_MAX_GAMES = 256
_CHAMPION_REPLAY_MIN_FRACTION = 0.10
_CHAMPION_REPLAY_DECAY_ITERATIONS = 8
_RAW_CANDIDATE_RETEST_GAMES = 192
_RAW_CANDIDATE_MAX_GAMES = 256
_PROMOTION_CONFIRMATION_GAMES = 128
_PROMOTION_CONFIRMATION_LB_FLOOR = 0.49
_ANCHOR_NO_MCTS_RETEST_GAMES = 256
_ANCHOR_NO_MCTS_MAX_GAMES = 512


def _csv_last_iteration(path):
    """Return the latest numeric iteration without loading a whole log frame."""
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as handle:
            rows = csv.reader(handle)
            header = None
            iteration_index = None
            latest = None
            for raw_row in rows:
                row = [str(cell).replace("\x00", "") for cell in raw_row]
                if not row or str(row[0]).strip().startswith("#"):
                    continue
                if header is None:
                    header = row
                    if "iteration" not in header:
                        return None
                    iteration_index = header.index("iteration")
                    continue
                if iteration_index >= len(row):
                    continue
                try:
                    iteration = int(float(row[iteration_index]))
                except (TypeError, ValueError):
                    continue
                latest = iteration if latest is None else max(latest, iteration)
            return latest
    except (OSError, csv.Error):
        return None


def _resolve_resume_history_csv(
    checkpoint_payload,
    *,
    checkpoint_path,
    base_dir,
    logs_dir,
    experiment_name,
    completed_iteration,
    exclude_path=None,
):
    """Resolve the canonical main CSV belonging to a resumed RL checkpoint."""
    payload = checkpoint_payload if isinstance(checkpoint_payload, dict) else {}
    artifacts = payload.get("rl_run_artifacts")
    if not isinstance(artifacts, dict):
        artifacts = payload.get("run_artifacts")
    artifact_candidates = []
    if isinstance(artifacts, dict):
        artifact_candidates.extend(
            artifacts.get(key)
            for key in ("main_csv", "csv", "log_csv")
        )
    artifact_candidates.extend(
        payload.get(key)
        for key in ("rl_run_log_csv", "run_log_csv", "source_log_csv")
    )
    excluded = Path(exclude_path).resolve() if exclude_path is not None else None
    for raw_path in artifact_candidates:
        if not raw_path:
            continue
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = Path(base_dir) / candidate
        if candidate.exists() and (excluded is None or candidate.resolve() != excluded):
            return candidate

    csv_dir = Path(logs_dir) / "csv"
    if not csv_dir.exists():
        return None
    candidates = []
    checkpoint_mtime = None
    try:
        checkpoint_mtime = Path(checkpoint_path).stat().st_mtime
    except OSError:
        pass
    model_prefix = "_".join(str(experiment_name).split("_")[:2])
    patterns = [f"{experiment_name}_*.csv"]
    if model_prefix and model_prefix != experiment_name:
        patterns.append(f"{model_prefix}_*.csv")
    discovered = {}
    for pattern in patterns:
        for candidate in csv_dir.glob(pattern):
            discovered[candidate.resolve()] = candidate
    for candidate in discovered.values():
        if candidate.name.endswith(("_performance.csv", "_data_quality.csv")):
            continue
        if excluded is not None and candidate.resolve() == excluded:
            continue
        last_iteration = _csv_last_iteration(candidate)
        if last_iteration is None or last_iteration < int(completed_iteration):
            continue
        exact = int(last_iteration) == int(completed_iteration)
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            mtime = 0.0
        not_newer_than_checkpoint = (
            checkpoint_mtime is None or mtime <= checkpoint_mtime + 5.0
        )
        candidates.append((exact, not_newer_than_checkpoint, mtime, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def _rl_run_artifact_metadata(logger, base_dir):
    def _portable(path):
        path = Path(path)
        try:
            return str(path.resolve().relative_to(Path(base_dir).resolve()))
        except (OSError, ValueError):
            return str(path)

    return {
        "main_csv": _portable(logger.csv_path),
        "performance_csv": _portable(logger.performance_log_path),
        "data_quality_csv": _portable(logger.data_quality_log_path),
    }


def _build_eval_config_with_exact_simulations(config, simulations):
    eval_config = dict(config)
    rl_config = dict(config.get('reinforcement_learning', {}) or {})
    rl_config['mcts_simulations'] = max(1, int(simulations))
    rl_config['eval_mcts_simulations_multiplier'] = 1.0
    rl_config['mcts_dynamic_budget_enabled'] = False
    rl_config['eval_mcts_dynamic_budget_enabled'] = False
    eval_config['reinforcement_learning'] = rl_config
    return eval_config


def _use_best_for_selfplay_iteration(
    iteration_number,
    *,
    learner_only_bootstrap_iterations,
    champion_refresh_due,
    learner_selfplay_safe,
):
    if int(iteration_number) <= max(0, int(learner_only_bootstrap_iterations)):
        return False
    return bool(champion_refresh_due or not learner_selfplay_safe)


def _resolve_champion_replay_sample_fraction(base_fraction, iterations_since_promotion):
    """Decay pinned rehearsal without letting the champion signal disappear."""
    base = max(0.0, min(0.40, float(base_fraction or 0.0)))
    if base <= 0.0:
        return 0.0
    floor = min(base, _CHAMPION_REPLAY_MIN_FRACTION)
    age = max(0, int(iterations_since_promotion or 0))
    progress = min(1.0, float(age) / float(_CHAMPION_REPLAY_DECAY_ITERATIONS))
    return float(base + (floor - base) * progress)


def _next_raw_candidate_retest_target(
    *, nn_safe, games_played, score_upper_bound, score_floor
):
    """Spend extra cheap raw games only while a passed MCTS candidate is undecided."""
    if bool(nn_safe):
        return None
    games_played = max(0, int(games_played or 0))
    if games_played < _RAW_CANDIDATE_RETEST_GAMES:
        return _RAW_CANDIDATE_RETEST_GAMES
    if (
        games_played < _RAW_CANDIDATE_MAX_GAMES
        and float(score_upper_bound or 0.0) >= float(score_floor or 0.0)
    ):
        return _RAW_CANDIDATE_MAX_GAMES
    return None


def _build_rl_optimizer_param_groups(model, rl_config, base_lr):
    """Build resume-compatible AdamW groups with one global learning rate.

    Families remain separate so old optimizer checkpoints can be loaded and
    gradient/update telemetry stays readable.  They no longer have independent
    LR multipliers: task balance belongs in the explicit loss contract, not in
    hidden optimizer geometry.
    """
    base_lr = float(base_lr)
    backbone_weight_decay = float(rl_config.get('weight_decay', 0.01))
    no_decay_weight_decay = float(rl_config.get('no_decay_weight_decay', 0.0))

    buckets = {
        'backbone_decay': {'params': [], 'weight_decay': backbone_weight_decay},
        'backbone_no_decay': {'params': [], 'weight_decay': no_decay_weight_decay},
        'policy_decay': {'params': [], 'weight_decay': backbone_weight_decay},
        'policy_no_decay': {'params': [], 'weight_decay': no_decay_weight_decay},
        'value_decay': {'params': [], 'weight_decay': backbone_weight_decay},
        'value_no_decay': {'params': [], 'weight_decay': no_decay_weight_decay},
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
            'lr': base_lr,
            'lr_factor': 1.0,
            'weight_decay': float(bucket['weight_decay']),
            'name': bucket_name,
        }
        param_groups.append(group)
        group_summaries.append(
            f"{bucket_name}: n={sum(p.numel() for p in params):,}, "
            f"lr_factor={group['lr_factor']:.2f}, wd={group['weight_decay']:.4g}"
        )

    return param_groups, group_summaries


def _enforce_uniform_rl_optimizer_contract(optimizer, rl_config, current_lr=None):
    """Migrate loaded optimizers from legacy family-specific LR/decay values."""
    base_lr = float(
        rl_config.get('learning_rate', 0.0)
        if current_lr is None
        else current_lr
    )
    decay = float(rl_config.get('weight_decay', 0.0))
    no_decay = float(rl_config.get('no_decay_weight_decay', 0.0))
    for group in optimizer.param_groups:
        group['lr_factor'] = 1.0
        group['lr'] = base_lr
        group_name = str(group.get('name', ''))
        group['weight_decay'] = no_decay if group_name.endswith('no_decay') else decay


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
    champion_replay_share = float(rl_cfg.get('replay_champion_fraction', 0.15) or 0.0)
    learner_only_bootstrap = max(
        0,
        int(rl_cfg.get('self_play_learner_only_bootstrap_iterations', 0) or 0),
    )
    correction_rank_target = max(
        0.0,
        float(rl_cfg.get('policy_correction_rank_weight', 0.0) or 0.0),
    )
    correction_rank_warmup = max(
        0,
        int(rl_cfg.get('policy_correction_rank_warmup_iterations', 0) or 0),
    )
    promotion_lb = float(rl_cfg.get('promotion_score_lower_bound_min', 0.50) or 0.50)
    anchor_lb = float(rl_cfg.get('promotion_anchor_score_lower_bound_min', 0.47) or 0.47)
    source_text = source_label
    if start_mode == 'resume':
        source_text = f"{source_label}; continue at iteration {start_iteration + 1}"
    if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False)):
        dynamic_min, dynamic_target, dynamic_max, _dynamic_chunk = (
            _resolve_dynamic_simulation_budget(
                rl_cfg.get('mcts_simulations', 192),
                minimum=rl_cfg.get('mcts_dynamic_budget_min', 64),
                maximum_multiplier=rl_cfg.get(
                    'mcts_dynamic_budget_max_multiplier',
                    5.0 / 3.0,
                ),
            )
        )
        search_budget_text = (
            f"dynamic {dynamic_min}-{dynamic_max} sims (avg {dynamic_target})"
        )
    else:
        search_budget_text = f"{int(rl_cfg.get('mcts_simulations', 0))} sims"
    search_detail = (
        f"Gumbel · top {int(rl_cfg.get('mcts_gumbel_max_considered_actions', 16))} "
        f"· c=({float(rl_cfg.get('mcts_gumbel_c_visit', 100.0)):.0f},"
        f"{float(rl_cfg.get('mcts_gumbel_c_scale', 0.10)):.2f}) "
        f"· Q-floor={float(rl_cfg.get('mcts_gumbel_q_range_floor', 0.25)):.2f} "
        f"· target-T={float(rl_cfg.get('mcts_gumbel_target_temperature', 1.0)):.2f}"
    )

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
            f"learner-only first {learner_only_bootstrap} iter, then guarded | "
            f"champion replay {champion_replay_share:.0%} -> "
            f"{min(champion_replay_share, _CHAMPION_REPLAY_MIN_FRACTION):.0%} | "
            f"replay {int(rl_cfg.get('replay_buffer_size', 0)):,} "
            f"| target <= {int(replay_max_policy_targets)} moves",
        ),
        (
            "search",
            f"{search_budget_text} | batch {int(rl_cfg.get('mcts_batch_size', 0))} "
            f"| {search_detail}",
        ),
        (
            "learning",
            f"batch {int(rl_cfg.get('batch_size', 0)):,} | lr {float(rl_cfg.get('learning_rate', 0.0)):.2e} "
            f"| {train_schedule} | correction-rank 0->{correction_rank_target:.2f} "
            f"in {correction_rank_warmup} iter | AdamW uniform LR",
        ),
        (
            "evaluation",
            f"every {int(rl_cfg.get('eval_every', 1))} iter | exact "
            f"{int(rl_cfg.get('mcts_simulations', 192))} sims/side | funnel {funnel_stages} | "
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


def _snapshot_model_state_cpu(model, share_memory=False, dtype=None):
    normalized_state = normalize_state_dict_keys(model.state_dict())
    snapshot = {}
    for key, tensor in normalized_state.items():
        cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        if dtype is not None and torch.is_floating_point(cpu_tensor):
            cpu_tensor = cpu_tensor.to(dtype=dtype)
        if share_memory:
            cpu_tensor = cpu_tensor.contiguous()
            cpu_tensor.share_memory_()
        snapshot[key] = cpu_tensor
    return snapshot


def _load_model_snapshot(model, state):
    if not isinstance(state, dict):
        return False
    normalized = normalize_state_dict_keys(state, target_keys=set(model.state_dict().keys()))
    model.load_state_dict(normalized, strict=True)
    return True


def _pad_replay_matrix(tensor, width, fill_value):
    if int(tensor.shape[1]) >= int(width):
        return tensor
    padded = torch.full(
        (int(tensor.shape[0]), int(width)),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :int(tensor.shape[1])].copy_(tensor)
    return padded


def _merge_replay_batches(recent_batch, champion_batch):
    """Merge regular and pinned-champion samples with sparse-target padding."""
    if champion_batch is None:
        return recent_batch
    recent = list(recent_batch)
    champion = list(champion_batch)
    for index, fill_value in ((1, -1), (2, 0.0), (3, False)):
        width = max(int(recent[index].shape[1]), int(champion[index].shape[1]))
        recent[index] = _pad_replay_matrix(recent[index], width, fill_value)
        champion[index] = _pad_replay_matrix(champion[index], width, fill_value)
    for index, fill_value in ((8, -1), (9, False)):
        width = max(int(recent[index].shape[1]), int(champion[index].shape[1]))
        recent[index] = _pad_replay_matrix(recent[index], width, fill_value)
        champion[index] = _pad_replay_matrix(champion[index], width, fill_value)
    return tuple(torch.cat((left, right), dim=0) for left, right in zip(recent, champion))


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
        discovered_levels = elo_result.get("levels_requested")
        if isinstance(discovered_levels, (list, tuple)):
            settings["levels"] = [int(level) for level in discovered_levels]
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
    }
    if not use_mcts:
        metadata.update({
            "estimated_elo": elo_value,
            "last_estimated_elo": elo_value,
            "estimated_elo_source": str(source),
            "estimated_elo_timestamp": timestamp,
            "estimated_elo_settings": settings,
        })
    if elo_std_error is not None:
        metadata[f"{mode_prefix}_se"] = float(elo_std_error)
        if not use_mcts:
            metadata["estimated_elo_se"] = float(elo_std_error)
    if ci_low is not None:
        metadata[f"{mode_prefix}_ci95_low"] = float(ci_low)
        if not use_mcts:
            metadata["estimated_elo_ci95_low"] = float(ci_low)
    if ci_high is not None:
        metadata[f"{mode_prefix}_ci95_high"] = float(ci_high)
        if not use_mcts:
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


def _merge_elo_checkpoint_metadata(target, incoming):
    """Merge mode metadata while retaining every distinct MCTS budget."""
    incoming = dict(incoming or {})
    incoming_by_sims = dict(incoming.pop("estimated_elo_mcts_by_simulations", {}) or {})
    target.update(incoming)
    if incoming_by_sims:
        merged = dict(target.get("estimated_elo_mcts_by_simulations", {}) or {})
        merged.update(incoming_by_sims)
        target["estimated_elo_mcts_by_simulations"] = {
            str(int(key)): value
            for key, value in sorted(merged.items(), key=lambda item: int(item[0]))
        }


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
        mcts_by_simulations=entry.get("elo_mcts_by_simulations"),
    )
    parts = []
    if elo_nn is not None:
        parts.append(f"raw NN {int(round(float(elo_nn)))}")
    mcts_profile = entry.get("elo_mcts_by_simulations") or {}
    if mcts_profile:
        parts.extend(
            f"MCTS {int(round(float(info['elo'])))}@{int(simulations)}"
            for simulations, info in sorted(mcts_profile.items(), key=lambda item: int(item[0]))
            if isinstance(info, dict) and info.get("elo") is not None
        )
    elif elo_mcts is not None:
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
        self._elo_priors = {}
        self._elo_prior_settings = {}
        self._load_checkpoint_elo_priors()
        self.enabled = True
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
        self.last_promoted_iteration = None
        self.last_promoted_raw_elo = None
        self.last_promoted_mcts_elo = None
        self.last_promoted_mcts_simulations = None
        self.last_mcts_eval_iteration = None

    @staticmethod
    def _elo_prior_key(use_mcts, simulations=0):
        return ("mcts", max(0, int(simulations or 0))) if use_mcts else ("nn", 0)

    def _load_checkpoint_elo_priors(self):
        modes = [(False, 0)]
        modes.extend(
            (True, int(cfg.get("mcts_simulations", 0) or 0))
            for cfg in build_final_mcts_elo_configs(self.elo_config)
        )
        for use_mcts, simulations in modes:
            key = self._elo_prior_key(use_mcts, simulations)
            for checkpoint_path in self.checkpoint_targets:
                elo, source = checkpoint_elo_seed(
                    checkpoint_path,
                    use_mcts=bool(use_mcts),
                    simulations=int(simulations),
                )
                if elo is not None:
                    self._elo_priors[key] = (float(elo), str(source or "previous checkpoint Elo"))
                    entry = load_checkpoint_metadata(checkpoint_path)
                    settings = entry.get("elo_nn_settings") if not use_mcts else None
                    if isinstance(settings, dict):
                        self._elo_prior_settings[key] = dict(settings)
                    break

    def _startup_measurement_matches_protocol(self, *, use_mcts, simulations, elo_config):
        """Reuse unchanged weights only when their benchmark protocol matches."""
        key = self._elo_prior_key(use_mcts, simulations)
        return elo_protocol_matches_settings(
            self._elo_prior_settings.get(key),
            elo_config,
            use_mcts=bool(use_mcts),
            simulations=int(simulations or 0),
        )

    def _resolve_elo_prior(self, use_mcts, simulations=0):
        key = self._elo_prior_key(use_mcts, simulations)
        exact = self._elo_priors.get(key)
        if exact is not None or not use_mcts:
            return exact
        target = max(0, int(simulations or 0))
        candidates = [
            (candidate_key[1], value)
            for candidate_key, value in self._elo_priors.items()
            if candidate_key[0] == "mcts"
        ]
        if not candidates:
            return None
        source_sims, (elo, source) = min(
            candidates,
            key=lambda item: abs(math.log2((item[0] + 1.0) / (target + 1.0))),
        )
        return float(elo), f"{source}; nearest available MCTS budget @{source_sims}"

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
        promoted_mcts_gap = self._promoted_mcts_min_iteration_gap()
        promoted_mcts_text = "off"
        if promoted_mcts_gap > 0:
            promoted_mcts_cfg = self._apply_promoted_elo_budget(
                build_final_elo_config(self.elo_config, use_mcts=True),
                use_mcts=True,
            )
            promoted_mcts_text = (
                f"after promotion, min_gap={promoted_mcts_gap} iteration(s), "
                f"cap={int(promoted_mcts_cfg.get('adaptive_max_total_games', 0) or 0)}"
            )
        final_use_mcts = bool(
            self.rl_elo_config.get(
                "final_use_mcts",
                self.rl_elo_config.get("promoted_use_mcts", False),
            )
        )
        final_mcts_text = "off"
        if final_use_mcts:
            final_mcts_text = "/".join(
                str(int(cfg.get("mcts_simulations", 0) or 0))
                for cfg in build_final_mcts_elo_configs(self.elo_config)
            )
        print(
            "Elo eval (RL/IL shared): promoted best vs Stockfish, "
            f"NN cap={int(nn_cfg.get('adaptive_max_total_games', 0) or 0)}, "
            f"target_SE={float(nn_cfg.get('adaptive_target_standard_error', 0.0) or 0.0):.0f}, "
            f"sync_device={self.eval_device.type}, "
            f"promoted_MCTS={promoted_mcts_text}, "
            f"final_mcts_profile={final_mcts_text}"
        )

    def _promoted_mcts_min_iteration_gap(self):
        if not bool(self.rl_elo_config.get("promoted_use_mcts", False)):
            return 0
        return max(
            1,
            int(self.rl_elo_config.get("promoted_mcts_min_iteration_gap", 10) or 10),
        )

    def _should_run_promoted_mcts(self, iteration_num):
        min_gap = self._promoted_mcts_min_iteration_gap()
        if min_gap <= 0:
            return False
        if self.last_mcts_eval_iteration is None:
            return True
        return int(iteration_num) - int(self.last_mcts_eval_iteration) >= min_gap

    def _apply_promoted_elo_budget(self, elo_config, *, use_mcts):
        """Apply tracking-only Elo limits without weakening final Elo."""
        prefix = "promoted_mcts" if use_mcts else "promoted_nn"
        key_map = {
            "probe_games_per_level": "adaptive_probe_games_per_level",
            "focus_games_per_level": "adaptive_focus_games_per_level",
            "extra_games_per_level": "adaptive_extra_games_per_level",
            "target_focus_levels": "adaptive_target_focus_levels",
            "max_total_games": "adaptive_max_total_games",
            "target_standard_error": "adaptive_target_standard_error",
            "min_games_for_se_stop": "adaptive_min_games_for_se_stop",
        }
        for suffix, runtime_key in key_map.items():
            configured_key = f"{prefix}_{suffix}"
            if configured_key in self.rl_elo_config:
                elo_config[runtime_key] = self.rl_elo_config[configured_key]
        return elo_config

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
        self.last_promoted_iteration = int(iteration_num)
        # Metadata is checkpoint-specific. Do not carry an older best's MCTS
        # rating into a newer promotion whose rate limit skipped that mode.
        self.last_elo_metadata = {}
        self.last_promoted_raw_elo = None
        self.last_promoted_mcts_elo = None
        self.last_promoted_mcts_simulations = None
        run_mcts = self._should_run_promoted_mcts(iteration_num)
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
        if raw_result is not None:
            self.last_promoted_raw_elo = float(raw_result)
        if self.interrupted_during_elo:
            return next((r for r in reversed(results) if r is not None), None)

        if run_mcts:
            mcts_simulations = self._resolve_eval_mcts_simulations()
            mcts_result = self._run_estimate(
                iteration_num,
                reason_label=f"promoted best MCTS {iteration_num}",
                use_mcts=True,
                persist_checkpoints=False,
                source="rl_promoted_elo",
            )
            results.append(mcts_result)
            if mcts_result is not None:
                self.last_promoted_mcts_elo = float(mcts_result)
                self.last_promoted_mcts_simulations = int(mcts_simulations)
                self.last_mcts_eval_iteration = int(iteration_num)
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
        # Do not benchmark an unchanged IL/startup best merely because RL
        # reached shutdown. MCTS Elo belongs only to a model promoted by this
        # run; its final profile may then extend the promotion-time measurement.
        final_use_mcts = bool(final_use_mcts and self.promoted_evaluations > 0)
        final_mcts_configs = (
            build_final_mcts_elo_configs(self.elo_config)
            if final_use_mcts
            else []
        )
        profile_text = "/".join(
            str(int(cfg.get("mcts_simulations", 0) or 0))
            for cfg in final_mcts_configs
        )
        mode_label = f"raw NN and MCTS profile {profile_text}" if final_use_mcts else "raw NN"
        print(f"Final best model: running normal Stockfish Elo for {mode_label}.")
        old_model = self.model
        if model_override is not None:
            self.model = model_override
        try:
            results = []
            raw_elo_config = build_final_elo_config(self.elo_config, use_mcts=False)
            same_promoted_weights = bool(
                self.last_promoted_iteration is not None
                and int(self.last_promoted_iteration) == int(iteration_num)
            )
            raw_result = self.last_promoted_raw_elo if same_promoted_weights else None
            unchanged_startup_best = bool(self.promoted_evaluations == 0)
            startup_prior = self._resolve_elo_prior(False, 0) if unchanged_startup_best else None
            can_reuse_startup = bool(
                startup_prior is not None
                and self._startup_measurement_matches_protocol(
                    use_mcts=False,
                    simulations=0,
                    elo_config=raw_elo_config,
                )
            )
            if raw_result is None and can_reuse_startup:
                raw_result = float(startup_prior[0])
                print(
                    "Final raw NN Elo reused from startup best "
                    "(weights and Stockfish protocol unchanged)."
                )
            if raw_result is None:
                if unchanged_startup_best and startup_prior is not None:
                    print(
                        "Final raw NN Elo is being measured again because the "
                        "Stockfish protocol changed since the stored result."
                    )
                raw_result = self._run_estimate(
                    iteration_num,
                    reason_label=f"final best raw NN {iteration_num}",
                    use_mcts=False,
                    persist_checkpoints=True,
                    source="rl_final_elo",
                )
            elif same_promoted_weights:
                print(
                    "Final raw NN Elo reused from promotion "
                    f"{int(self.last_promoted_iteration)} (same best weights)."
                )
            results.append(raw_result)
            if self.interrupted_during_elo:
                return next((r for r in reversed(results) if r is not None), None)
            if final_use_mcts:
                for final_mcts_config in final_mcts_configs:
                    final_mcts_simulations = int(
                        final_mcts_config.get("mcts_simulations", 0) or 0
                    )
                    if final_mcts_simulations <= 0:
                        continue
                    can_reuse_mcts = bool(
                        same_promoted_weights
                        and self.last_promoted_mcts_elo is not None
                        and int(self.last_promoted_mcts_simulations or 0)
                        == int(final_mcts_simulations)
                    )
                    if can_reuse_mcts:
                        mcts_result = self.last_promoted_mcts_elo
                        print(
                            "Final MCTS Elo reused from promotion "
                            f"{int(self.last_promoted_iteration)} "
                            f"(same weights, {final_mcts_simulations} simulations)."
                        )
                    else:
                        mcts_result = self._run_estimate(
                            iteration_num,
                            reason_label=(
                                f"final best MCTS {iteration_num} "
                                f"@{final_mcts_simulations}"
                            ),
                            use_mcts=True,
                            persist_checkpoints=True,
                            source="rl_final_elo",
                            mcts_simulations=final_mcts_simulations,
                        )
                    results.append(mcts_result)
                    if self.interrupted_during_elo:
                        break
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
        mcts_simulations=None,
    ):
        elo_config = build_final_elo_config(self.elo_config, use_mcts=bool(use_mcts))
        if use_mcts:
            elo_config["mcts_simulations"] = int(
                mcts_simulations
                if mcts_simulations is not None
                else self._resolve_eval_mcts_simulations()
            )
        if str(source) == "rl_promoted_elo":
            self._apply_promoted_elo_budget(elo_config, use_mcts=bool(use_mcts))
        active_simulations = int(elo_config.get("mcts_simulations", 0) or 0) if use_mcts else 0
        previous = self._resolve_elo_prior(bool(use_mcts), active_simulations)
        if previous is not None:
            previous_elo, previous_source = previous
            elo_config["adaptive_initial_elo"] = float(previous_elo)
            elo_config["adaptive_initial_elo_source"] = str(previous_source)
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
            prior_key = self._elo_prior_key(bool(use_mcts), active_simulations)
            self._elo_priors[prior_key] = (
                float(estimated_elo),
                f"latest measured {'MCTS @' + str(active_simulations) if use_mcts else 'raw NN'} Elo",
            )
            self.last_elo_iteration = int(iteration_num)
            source = str(source)
            try:
                _merge_elo_checkpoint_metadata(
                    self.last_elo_metadata,
                    _build_elo_checkpoint_metadata(
                        estimated_elo,
                        elo_config,
                        source=source,
                        elo_result=elo_result,
                    ),
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


def _score_rate_standard_error(score_rate, num_games, standard_error=None):
    games = max(1.0, float(num_games or 1))
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    return (
        math.sqrt(score * (1.0 - score) / games)
        if standard_error is None
        else max(0.0, float(standard_error))
    )


def _score_rate_lower_bound(score_rate, num_games, z=1.28, standard_error=None):
    """Conservative one-sided normal lower bound for a [0, 1] game score."""
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    score_se = _score_rate_standard_error(score, num_games, standard_error)
    return max(0.0, score - z_value * score_se)


def _score_rate_upper_bound(score_rate, num_games, z=1.28, standard_error=None):
    """One-sided normal upper bound paired with `_score_rate_lower_bound`."""
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    score_se = _score_rate_standard_error(score, num_games, standard_error)
    return min(1.0, score + z_value * score_se)


def _paired_score_standard_error(game_scores):
    """Estimate match uncertainty from complete opening/color pairs.

    Games ``2k`` and ``2k+1`` use the same fixed opening with swapped colours.
    Treating their mean as one observation removes a large part of opening and
    colour variance while retaining correlation inside the pair.
    """
    pairs = defaultdict(list)
    for game_idx, score in dict(game_scores or {}).items():
        pairs[int(game_idx) // 2].append(float(score))
    pair_scores = [sum(values) / 2.0 for values in pairs.values() if len(values) == 2]
    pair_count = len(pair_scores)
    if pair_count < 2:
        return None, pair_count
    mean = sum(pair_scores) / pair_count
    sample_variance = sum((value - mean) ** 2 for value in pair_scores) / (pair_count - 1)
    return math.sqrt(sample_variance / pair_count), pair_count


def _attach_paired_score_stats(stats, game_scores):
    stats = dict(stats or {})
    scores = {int(key): float(value) for key, value in dict(game_scores or {}).items()}
    score_se, pair_count = _paired_score_standard_error(scores)
    stats['game_scores'] = scores
    stats['paired_score_pairs'] = int(pair_count)
    stats['paired_score_se'] = score_se
    return stats


def _evaluate_models_with_paired_scores(*args, **kwargs):
    """Run eval while retaining per-opening-pair scores for confidence gates."""
    game_scores = {}
    external_callback = kwargs.pop('progress_callback', None)

    def _on_progress(
        completed,
        game_idx=None,
        wins=0,
        draws=0,
        losses=0,
        unresolved=0,
        plies=0,
        result='*',
        model1_as_white=True,
    ):
        if game_idx is not None:
            game_scores[int(game_idx)] = float(wins) + 0.5 * float(draws + unresolved)
        if external_callback is not None:
            external_callback(
                completed,
                game_idx,
                wins,
                draws,
                losses,
                unresolved,
                plies,
                result,
                model1_as_white,
            )

    kwargs['progress_callback'] = _on_progress
    return _attach_paired_score_stats(evaluate_models(*args, **kwargs), game_scores)


def _promotion_anchor_safety_decision(
    anchor_score_rate,
    anchor_score_lower_bound,
    anchor_no_mcts_score_rate,
    anchor_no_mcts_score_lower_bound,
    rl_cfg,
):
    """Keep every promoted best above the IL floor with and without search."""
    if not bool(rl_cfg.get('promotion_require_anchor_non_regression', True)):
        return True, "anchor guard disabled"
    if anchor_score_rate is None or anchor_score_lower_bound is None:
        return False, "anchor evidence unavailable"
    score_floor = float(rl_cfg.get('promotion_anchor_min_score_rate', 0.50))
    lower_bound_floor = float(
        rl_cfg.get('promotion_anchor_score_lower_bound_min', 0.47)
    )
    score = float(anchor_score_rate)
    lower_bound = float(anchor_score_lower_bound)
    mcts_safe = bool(score >= score_floor and lower_bound >= lower_bound_floor)
    raw_enabled = bool(rl_cfg.get('promotion_anchor_no_mcts_gate_enabled', True))
    raw_score_floor = float(
        rl_cfg.get('promotion_anchor_no_mcts_score_rate_min', 0.50) or 0.50
    )
    raw_lower_bound_floor = float(
        rl_cfg.get('promotion_anchor_no_mcts_score_lower_bound_min', 0.47) or 0.47
    )
    if raw_enabled and (
        anchor_no_mcts_score_rate is None
        or anchor_no_mcts_score_lower_bound is None
    ):
        return False, "raw anchor evidence unavailable"
    raw_score = (
        None if anchor_no_mcts_score_rate is None else float(anchor_no_mcts_score_rate)
    )
    raw_lower = (
        None
        if anchor_no_mcts_score_lower_bound is None
        else float(anchor_no_mcts_score_lower_bound)
    )
    raw_safe = bool(
        not raw_enabled
        or (raw_score >= raw_score_floor and raw_lower >= raw_lower_bound_floor)
    )
    raw_text = "off" if not raw_enabled else (
        f"{raw_score:.1%}, LB {raw_lower:.1%}; "
        f"required {raw_score_floor:.1%}/{raw_lower_bound_floor:.1%}"
    )
    return bool(mcts_safe and raw_safe), (
        f"anchor MCTS {score:.1%}, LB {lower_bound:.1%}; "
        f"required {score_floor:.1%}/{lower_bound_floor:.1%}; raw {raw_text}"
    )


def _promotion_eval_has_full_evidence(eval_stage):
    """Only the complete paired funnel may promote a new best checkpoint."""
    return str(eval_stage or '') in {'funnel:advanced', 'funnel:confirmation'}


def _safe_score_rate(wins, draws, losses):
    total = int(wins or 0) + int(draws or 0) + int(losses or 0)
    return (
        (float(wins or 0) + 0.5 * float(draws or 0)) / float(total)
        if total > 0 else 0.0
    )


def _promotion_nn_safety_decision(no_mcts_score_rate, num_games, z, rl_cfg):
    """Require raw-NN non-inferiority before promoting a search winner."""
    enabled = bool(rl_cfg.get('promotion_no_mcts_gate_enabled', True))
    score_floor = float(
        rl_cfg.get(
            'promotion_no_mcts_score_rate_min',
            0.40,
        ) or 0.40
    )
    upper_floor = float(rl_cfg.get('promotion_no_mcts_upper_bound_min', 0.50) or 0.50)
    if not enabled:
        return True, "", score_floor, None
    if no_mcts_score_rate is None:
        return False, "blocked: NN-only safety evaluation unavailable", score_floor, None
    score = float(no_mcts_score_rate)
    upper = _score_rate_upper_bound(score, num_games, z)
    if score < score_floor:
        return (
            False,
            f"blocked: NN-only {score:.1%} < severe floor {score_floor:.1%}",
            score_floor,
            upper,
        )
    if upper < upper_floor:
        return (
            False,
            f"blocked: NN-only UCB {upper:.1%} < break-even {upper_floor:.1%}",
            score_floor,
            upper,
        )
    return True, "", score_floor, upper


def _anchor_candidate_needs_more_games(
    score_rate,
    num_games,
    *,
    min_score_rate,
    min_score_lower_bound,
    z,
    max_games,
    score_standard_error=None,
):
    """Whether an anchor result is still statistically inconclusive."""
    games = max(1, int(num_games or 0))
    if games >= max(1, int(max_games)):
        return False
    lower = _score_rate_lower_bound(
        score_rate,
        games,
        z,
        standard_error=score_standard_error,
    )
    if float(score_rate or 0.0) >= float(min_score_rate) and lower >= float(min_score_lower_bound):
        return False
    upper = _score_rate_upper_bound(
        score_rate,
        games,
        z,
        standard_error=score_standard_error,
    )
    return upper >= float(min_score_rate)


def _combine_eval_stats(*stats_items):
    combined = _empty_eval_stats()
    total_games = 0.0
    weighted_unresolved = 0.0
    weighted_wins = 0.0
    weighted_draws = 0.0
    weighted_losses = 0.0
    game_scores = {}
    combined_profile = {}
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
        for key, value in dict(stats.get('profile', {}) or {}).items():
            if isinstance(value, (int, float)):
                combined_profile[key] = combined_profile.get(key, 0) + weight * value
        if weight == 1.0:
            game_scores.update({
                int(key): float(value)
                for key, value in dict(stats.get('game_scores', {}) or {}).items()
            })

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
    combined['profile'] = combined_profile
    return _attach_paired_score_stats(combined, game_scores)


def _eval_mcts_move_metrics(eval_stats, prefix):
    """Derive actual Gumbel move-change rates from additive eval counters."""
    profile = dict((eval_stats or {}).get('profile', {}) or {})
    base = f'eval_mcts_{prefix}'
    samples = int(profile.get(f'{base}_move_samples', 0) or 0)
    changed = int(profile.get(f'{base}_changed_count', 0) or 0)
    q_samples = int(profile.get(f'{base}_changed_q_samples', 0) or 0)
    higher = int(profile.get(f'{base}_higher_q_count', 0) or 0)
    lower = int(profile.get(f'{base}_lower_q_count', 0) or 0)
    q_sum = float(profile.get(f'{base}_changed_q_delta_sum', 0.0) or 0.0)
    budget_samples = int(profile.get(f'{base}_budget_samples', 0) or 0)
    budget_sum = float(profile.get(f'{base}_budget_sum', 0.0) or 0.0)
    reduced_budgets = int(profile.get(f'{base}_reduced_budget_count', 0) or 0)
    return {
        'samples': samples,
        'avg_simulations': (
            budget_sum / float(budget_samples)
        ) if budget_samples > 0 else None,
        'reduced_budget_rate': (
            float(reduced_budgets) / float(budget_samples)
        ) if budget_samples > 0 else None,
        'changed_rate': (float(changed) / float(samples)) if samples > 0 else None,
        'higher_q_when_changed_rate': (
            float(higher) / float(q_samples)
        ) if q_samples > 0 else None,
        'lower_q_when_changed_rate': (
            float(lower) / float(q_samples)
        ) if q_samples > 0 else None,
        'changed_q_delta_mean': (q_sum / float(q_samples)) if q_samples > 0 else None,
    }


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
        stats = _evaluate_models_with_paired_scores(
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
        if (stats or {}).get('infrastructure_error'):
            return stats
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
            score_standard_error=(combined or {}).get('paired_score_se'),
        ):
            break
        batch_index += 1

    return combined or _empty_eval_stats()


def _evaluate_anchor_no_mcts_sequential(
    *,
    model,
    anchor_model,
    config,
    device,
    initial_games,
    max_games,
    min_score_rate,
    min_score_lower_bound,
    stat_gate_z,
    game_index_offset=0,
    use_fixed_openings=True,
    initial_stats=None,
):
    """Extend a raw-NN anchor match only while non-inferiority is undecided."""
    initial_games = max(2, int(initial_games or 2))
    if initial_games % 2:
        initial_games += 1
    max_games = max(initial_games, int(max_games or initial_games))
    targets = sorted({
        initial_games,
        min(max_games, max(initial_games, _ANCHOR_NO_MCTS_RETEST_GAMES)),
        max_games,
    })
    combined = initial_stats
    completed_games = int((combined or {}).get('num_games', 0) or 0)

    if completed_games > 0:
        initial_score = float((combined or {}).get('score_rate', 0.0) or 0.0)
        if not _anchor_candidate_needs_more_games(
            initial_score,
            completed_games,
            min_score_rate=min_score_rate,
            min_score_lower_bound=min_score_lower_bound,
            z=stat_gate_z,
            max_games=max_games,
            score_standard_error=(combined or {}).get('paired_score_se'),
        ):
            return combined

    for target_games in targets:
        if completed_games >= target_games:
            continue
        games_this_batch = int(target_games - completed_games)
        stats = evaluate_models_no_mcts(
            model,
            anchor_model,
            config,
            device,
            games_this_batch,
            game_index_offset=game_index_offset + completed_games,
            use_fixed_openings=use_fixed_openings,
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
            score_standard_error=(combined or {}).get('paired_score_se'),
        ):
            break

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
    preliminary_stats = _evaluate_models_with_paired_scores(
        model,
        best_model,
        preliminary_config,
        device,
        preliminary_games,
        game_index_offset=game_index_offset,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    if (preliminary_stats or {}).get('infrastructure_error'):
        return preliminary_stats, "infrastructure_error"
    score_rate = float((preliminary_stats or {}).get('score_rate', 0.0) or 0.0)
    true_win_rate = float((preliminary_stats or {}).get('win_rate', 0.0) or 0.0)
    if (
        score_rate < float(preliminary_score_rate)
        or true_win_rate < float(preliminary_true_win_rate)
    ):
        return preliminary_stats, "funnel:preliminary_reject"
    if medium_games <= 0:
        return preliminary_stats, "funnel:preliminary_only"

    medium_stats = _evaluate_models_with_paired_scores(
        model,
        best_model,
        medium_config,
        device,
        medium_games,
        game_index_offset=game_index_offset + preliminary_games,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    if (medium_stats or {}).get('infrastructure_error'):
        return medium_stats, "infrastructure_error"
    # Every stage now uses the same search budget and disjoint paired openings,
    # so all completed games are commensurate.  Keep the preliminary evidence
    # instead of discarding it and making the reported score jump between small
    # opening subsets.
    through_medium = _combine_eval_stats(preliminary_stats, medium_stats)
    medium_score = float((through_medium or {}).get('score_rate', 0.0) or 0.0)
    medium_true_win = float((through_medium or {}).get('win_rate', 0.0) or 0.0)
    if (
        medium_score < float(medium_score_rate)
        or medium_true_win < float(medium_true_win_rate)
    ):
        return through_medium, "funnel:medium_reject"
    if advanced_games <= 0:
        return through_medium, "funnel:medium_only"

    advanced_stats = _evaluate_models_with_paired_scores(
        model,
        best_model,
        advanced_config,
        device,
        advanced_games,
        game_index_offset=game_index_offset + preliminary_games + medium_games,
        use_fixed_openings=use_fixed_openings,
        central_runtime=central_runtime,
    )
    if (advanced_stats or {}).get('infrastructure_error'):
        return advanced_stats, "infrastructure_error"
    combined = _combine_eval_stats(through_medium, advanced_stats)
    return combined, "funnel:advanced"


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


def _run_selective_reanalyse(replay_buffer, model, config, device):
    """Refresh the hardest reconstructable replay rows with the active search."""
    rl_cfg = config.get('reinforcement_learning', {})
    if not bool(rl_cfg.get('replay_reanalyse_enabled', True)):
        return {'selected': 0, 'updated': 0, 'reason': 'disabled'}
    if replay_buffer is None or len(replay_buffer) <= 0:
        return {'selected': 0, 'updated': 0, 'reason': 'empty'}

    max_positions = max(1, int(rl_cfg.get('replay_reanalyse_max_positions', 128)))
    fraction = max(0.0, min(1.0, float(rl_cfg.get('replay_reanalyse_fraction', 0.01))))
    count = min(max_positions, max(1, int(round(len(replay_buffer) * fraction))))
    indices = replay_buffer.select_hard_position_indices(
        count,
        min_age=int(rl_cfg.get('replay_reanalyse_min_age', 1)),
    )
    if indices.size <= 0:
        return {'selected': 0, 'updated': 0, 'reason': 'no_fen'}

    simulations = max(1, int(rl_cfg.get('replay_reanalyse_simulations', 256)))
    batch_size = max(1, int(rl_cfg.get('replay_reanalyse_batch_positions', 16)))
    policy_mix = max(0.0, min(1.0, float(rl_cfg.get('replay_reanalyse_policy_mix', 0.65))))
    mcts_rl_cfg = dict(rl_cfg)
    mcts_rl_cfg.update({
        'mcts_simulations': simulations,
        'mcts_dynamic_budget_enabled': False,
        'mcts_gumbel_scale': 0.0,
        'mcts_gumbel_eval_scale': 0.0,
    })
    mcts_config = dict(config)
    mcts_config['reinforcement_learning'] = mcts_rl_cfg
    mcts = MultiGameBatchMCTS(model, mcts_config, device)
    hard_positions = replay_buffer.hard_positions_for_indices(indices)
    was_training = bool(model.training)
    model.eval()
    updated = 0
    skipped = 0
    correction_count = 0
    started = time.perf_counter()
    try:
        for start in range(0, len(hard_positions), batch_size):
            chunk_positions = hard_positions[start:start + batch_size]
            chunk_indices = indices[start:start + batch_size]
            states = []
            boards = []
            valid_indices = []
            for replay_idx, payload in zip(chunk_indices.tolist(), chunk_positions):
                try:
                    board = chess.new_board(payload['fen'])
                    encoded_history = [
                        mcts._encode_history_entry(chess.new_board(history_fen))
                        for history_fen in list(payload.get('history_fens', []) or [])
                    ]
                except Exception:
                    skipped += 1
                    continue
                boards.append(board)
                valid_indices.append(int(replay_idx))
                states.append([board, None, False, encoded_history, 0, None, simulations])
            if not states:
                continue
            visits_list, metadata_list = mcts.search_many(
                states,
                num_simulations=simulations,
                add_root_noise=False,
                return_search_metadata=True,
            )
            policy_targets = []
            root_q_values = []
            best_q_values = []
            orig_q_values = []
            policy_kld_values = []
            search_visit_values = []
            policy_weight_values = []
            search_changed_top_values = []
            search_q_delta_values = []
            for board, visits, metadata in zip(boards, visits_list, metadata_list):
                metadata = metadata or {}
                def _metadata_float(key, default=float('nan')):
                    try:
                        value = metadata.get(key, default)
                        return float(default if value is None else value)
                    except (TypeError, ValueError):
                        return float(default)
                probability_target = (metadata or {}).get('policy_target_probs_override')
                target = probability_target if isinstance(probability_target, dict) and probability_target else visits
                policy_targets.append(_build_sparse_policy_target_from_visits(target, board))
                root_q_values.append(_metadata_float('root_value', 0.0))
                best_q_values.append(_metadata_float('best_q'))
                orig_q_values.append(_metadata_float('orig_q'))
                policy_kld_values.append(_metadata_float('policy_kld'))
                search_visit_values.append(int(metadata.get('search_visits', 0) or 0))
                changed_top, q_delta, correction_multiplier = _search_correction_metadata(metadata)
                policy_weight_values.append(
                    _policy_uptake_weight(metadata) * correction_multiplier
                )
                search_changed_top_values.append(changed_top)
                search_q_delta_values.append(q_delta)
                correction_count += int(
                    changed_top
                    and math.isfinite(q_delta)
                    and q_delta > USEFUL_SEARCH_Q_DELTA_MIN
                )
            updated += replay_buffer.update_reanalyzed_targets(
                valid_indices,
                policy_targets,
                root_q_values,
                policy_mix=policy_mix,
                policy_weight_values=policy_weight_values,
                search_changed_top_values=search_changed_top_values,
                search_q_delta_values=search_q_delta_values,
                best_q_values=best_q_values,
                orig_q_values=orig_q_values,
                policy_kld_values=policy_kld_values,
                search_visit_values=search_visit_values,
            )
    finally:
        if was_training:
            model.train()
    return {
        'selected': int(indices.size),
        'updated': int(updated),
        'skipped': int(skipped),
        'corrections': int(correction_count),
        'simulations': simulations,
        'seconds': float(time.perf_counter() - started),
        'reason': 'ok',
    }


def play_games_parallel_mcts(
    model,
    config,
    device,
    num_games,
    replay_buffer=None,
    hard_start_positions=None,
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
    hard_start_positions = list(hard_start_positions or [None] * int(num_games))
    if len(hard_start_positions) < int(num_games):
        hard_start_positions.extend([None] * (int(num_games) - len(hard_start_positions)))
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
    # Queue transport has one canonical scheduler: workers refill completed
    # game slots from a shared FIFO. The old fixed-size outer chunk scheduler
    # created repeated drain waves and is intentionally gone.
    global_game_stream_enabled = bool(dynamic_dispatch_enabled)
    
    worker_plan = _resolve_selfplay_worker_plan(config, num_games)
    device_type = worker_plan['device_type']
    worker_specs = worker_plan['worker_specs']

    max_batch_games_raw = rl_cfg.get('max_batch_games_per_worker', 1)
    try:
        max_batch_games = max(1, int(max_batch_games_raw))
    except Exception:
        max_batch_games = 1
    
    opponent_assignments = {int(rank): {} for rank, _games in worker_specs}
    opponent_debug = {"mode": "same-model"}

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
    queue_search_fresh_simulations_used_sum = 0
    queue_search_inherited_visit_credit_sum = 0
    queue_search_simulations_budget_sum = 0
    queue_search_samples = 0
    queue_search_difficulty_samples = 0
    queue_search_difficulty_sum = 0.0
    queue_search_difficulty_sq_sum = 0.0
    queue_search_difficulty_budget_cross_sum = 0.0
    queue_search_budget_sq_sum = 0.0
    queue_tree_reuse_attempts = 0
    queue_tree_reuse_hits = 0
    queue_tree_inherited_visits_sum = 0
    queue_tree_reuse_credit_samples = 0
    queue_tree_reuse_quality_sum = 0.0
    queue_tree_reuse_candidate_coverage_sum = 0.0
    queue_tree_reuse_visited_prior_mass_sum = 0.0
    queue_tree_reuse_fresh_floor_sum = 0
    queue_tree_reuse_scout_stability_sum = 0.0
    queue_tree_reuse_scout_extra_credit_sum = 0
    queue_tree_reuse_scout_reduced_count = 0
    queue_shared_tree_searches = 0
    queue_hard_start_games = 0
    queue_search_simulations_used_samples = []
    queue_search_simulations_budget_samples = []
    queue_wait_total_s = 0.0
    queue_wait_events = 0
    queue_profile_stats = {}
    queue_mcts_quality_stats = defaultdict(float)
    queue_opponent_source_games = defaultdict(int)
    queue_opponent_source_results = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "games": 0})
    selfplay_pool = None
    interrupted = False
    startup_done_time = None

    def _accumulate_profile_stats(target, source):
        for key, value in dict(source or {}).items():
            if isinstance(value, (int, np.integer)):
                target[str(key)] = int(target.get(str(key), 0)) + int(value)
            else:
                target[str(key)] = float(target.get(str(key), 0.0)) + float(value)

    def _recompute_profile_ratios(profile):
        """Rebuild non-additive worker metrics after cross-process summation."""
        nn_calls = int(profile.get('mcts_nn_inference_calls', 0) or 0)
        nn_items = int(profile.get('mcts_nn_inference_batch_items', 0) or 0)
        legal_items = int(profile.get('mcts_nn_legal_move_items', 0) or 0)
        nn_time = float(profile.get('mcts_nn_inference_time', 0.0) or 0.0)
        search_time = float(profile.get('mcts_search_many_time', 0.0) or 0.0)
        central_requests = int(profile.get('mcts_central_inference_requests', 0) or 0)
        central_items = int(profile.get('mcts_central_inference_server_batch_items', 0) or 0)
        profile['average_batch_size'] = float(nn_items / nn_calls) if nn_calls > 0 else 0.0
        profile['average_legal_moves_per_position'] = (
            float(legal_items / nn_items) if nn_items > 0 else 0.0
        )
        profile['average_legal_moves_per_batch'] = (
            float(legal_items / nn_calls) if nn_calls > 0 else 0.0
        )
        profile['inference_time_per_batch_ms'] = (
            1000.0 * nn_time / float(nn_calls) if nn_calls > 0 else 0.0
        )
        profile['inference_time_per_position_ms'] = (
            1000.0 * nn_time / float(nn_items) if nn_items > 0 else 0.0
        )
        profile['worker_nn_wait_share_pct'] = (
            max(0.0, min(100.0, 100.0 * nn_time / max(1e-8, search_time)))
        )
        profile['central_average_batch_size'] = (
            float(central_items / central_requests) if central_requests > 0 else 0.0
        )
        central_time_keys = {
            'central_remote_wait_ms_per_request': 'mcts_central_inference_remote_wait_time',
            'central_request_submit_ms_per_request': 'mcts_central_inference_request_put_time',
            'central_server_queue_wait_ms_per_request': 'mcts_central_inference_server_queue_wait_time',
            'central_descriptor_queue_wait_ms_per_request': 'mcts_central_inference_server_descriptor_queue_wait_time',
            'central_batch_coalesce_wait_ms_per_request': 'mcts_central_inference_server_batch_coalesce_wait_time',
            'central_server_pipeline_wait_ms_per_request': 'mcts_central_inference_server_pipeline_wait_time',
            'central_server_output_finalize_wait_ms_per_request': 'mcts_central_inference_server_output_finalize_wait_time',
            'central_server_forward_ms_per_request': 'mcts_central_inference_server_forward_time',
            'central_server_h2d_ms_per_request': 'mcts_central_inference_server_h2d_time',
            'central_server_d2h_ms_per_request': 'mcts_central_inference_server_d2h_time',
            'central_server_concat_ms_per_request': 'mcts_central_inference_server_concat_time',
            'central_server_total_ms_per_request': 'mcts_central_inference_server_total_time',
        }
        for ratio_key, total_key in central_time_keys.items():
            total = float(profile.get(total_key, 0.0) or 0.0)
            profile[ratio_key] = (
                1000.0 * total / float(central_requests) if central_requests > 0 else 0.0
            )
        shared_requests = int(profile.get('mcts_central_inference_shared_requests', 0) or 0)
        cache_queries = int(profile.get('mcts_central_inference_cache_queries', 0) or 0)
        cache_bypassed = int(
            profile.get('mcts_central_inference_cache_bypassed_positions', 0) or 0
        )
        cache_hits = int(profile.get('mcts_central_inference_cache_hits', 0) or 0)
        dedup_hits = int(profile.get('mcts_central_inference_dedup_hits', 0) or 0)
        cache_scope = cache_queries + cache_bypassed
        profile['central_shared_memory_request_fraction'] = (
            float(shared_requests) / float(central_requests) if central_requests > 0 else 0.0
        )
        profile['central_shared_memory_mib_avoided'] = (
            float(profile.get('mcts_central_inference_shared_bytes_avoided', 0) or 0)
            / float(1024 ** 2)
        )
        profile['central_cache_hit_rate'] = (
            float(cache_hits) / float(cache_queries) if cache_queries > 0 else 0.0
        )
        profile['central_dedup_hit_rate'] = (
            float(dedup_hits) / float(cache_queries) if cache_queries > 0 else 0.0
        )
        profile['central_nn_saved_rate'] = (
            float(cache_hits + dedup_hits) / float(cache_queries)
            if cache_queries > 0 else 0.0
        )
        profile['central_cache_active_fraction'] = (
            float(cache_queries) / float(cache_scope) if cache_scope > 0 else 0.0
        )
        profile['central_nn_saved_overall_rate'] = (
            float(cache_hits + dedup_hits) / float(cache_scope)
            if cache_scope > 0 else 0.0
        )
        profile['central_shared_slot_wait_ms_per_request'] = (
            1000.0 * float(profile.get('mcts_central_inference_shared_slot_wait_time', 0.0) or 0.0)
            / float(max(1, shared_requests))
        )
        profile['central_server_cache_lookup_ms_per_request'] = (
            1000.0 * float(profile.get('mcts_central_inference_server_cache_lookup_time', 0.0) or 0.0)
            / float(max(1, central_requests))
        )
        profile['central_server_staging_copy_ms_per_request'] = (
            1000.0 * float(profile.get('mcts_central_inference_server_staging_copy_time', 0.0) or 0.0)
            / float(max(1, central_requests))
        )
        profile['central_gpu_batch_fill'] = (
            float(profile.get('mcts_central_inference_gpu_batch_fill_sum', 0.0) or 0.0)
            / float(max(1, central_requests))
        )
        profile['central_batch_target_avg'] = (
            float(profile.get('mcts_central_inference_server_batch_target_sum', 0.0) or 0.0)
            / float(max(1, central_requests))
        )
        profile['central_output_pipeline_rate'] = (
            float(profile.get('mcts_central_inference_server_output_pipeline_requests', 0) or 0)
            / float(max(1, central_requests))
        )
        profile['central_auto_batch_calibrated_rate'] = (
            float(profile.get('mcts_central_inference_server_auto_calibrated_requests', 0) or 0)
            / float(max(1, central_requests))
        )
        native_root_calls = int(profile.get('mcts_native_root_selection_calls', 0) or 0)
        python_root_calls = int(profile.get('mcts_python_root_selection_calls', 0) or 0)
        profile['mcts_native_root_selection_rate'] = (
            float(native_root_calls) / float(native_root_calls + python_root_calls)
            if native_root_calls + python_root_calls > 0 else 0.0
        )
        native_q_calls = int(profile.get('mcts_native_completed_q_calls', 0) or 0)
        python_q_calls = int(profile.get('mcts_python_completed_q_calls', 0) or 0)
        profile['mcts_native_completed_q_rate'] = (
            float(native_q_calls) / float(native_q_calls + python_q_calls)
            if native_q_calls + python_q_calls > 0 else 0.0
        )
        native_tree_leaves = int(profile.get('mcts_native_tree_selected_leaves', 0) or 0)
        python_tree_leaves = int(profile.get('mcts_python_tree_selected_leaves', 0) or 0)
        profile['mcts_native_tree_selection_rate'] = (
            float(native_tree_leaves) / float(native_tree_leaves + python_tree_leaves)
            if native_tree_leaves + python_tree_leaves > 0 else 0.0
        )

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

            def _selfplay_runtime_overrides():
                return {
                    'mcts_temperature': rl_cfg.get('mcts_temperature'),
                    'mcts_temperature_threshold': rl_cfg.get('mcts_temperature_threshold'),
                }

            if global_game_stream_enabled:
                result_files = []
                progress_files = []
                active_workers = {}
                completed_games_by_rank = {int(rank): 0 for rank, _ in worker_specs}
                current_progress_files = {}
                stream_jobs = [
                    {'hard_start': hard_start_positions[game_idx]}
                    for game_idx in range(int(num_games))
                ]
                worker_ranks = [int(rank) for rank, _ in worker_specs]
                initial_total = min(
                    len(stream_jobs),
                    int(max_batch_games) * max(1, len(worker_ranks)),
                )
                initial_base, initial_extra = divmod(
                    initial_total,
                    max(1, len(worker_ranks)),
                )
                initial_jobs_by_rank = {}
                initial_cursor = 0
                for rank_idx, rank in enumerate(worker_ranks):
                    initial_count = initial_base + int(rank_idx < initial_extra)
                    initial_jobs_by_rank[rank] = [
                        {**job, 'task_id': str(task_id)}
                        for job in stream_jobs[
                            initial_cursor:initial_cursor + initial_count
                        ]
                    ]
                    initial_cursor += initial_count
                selfplay_pool.prepare_game_stream(task_id, stream_jobs[initial_cursor:])
                for rank, _games_for_worker in worker_specs:
                    result_file, progress_file = selfplay_pool.dispatch_task(
                        rank=int(rank),
                        task_id=task_id,
                        model_state_path=model_state_path,
                        model_state=model_state_cpu,
                        temperature=rl_cfg.get('mcts_temperature'),
                        runtime_overrides=_selfplay_runtime_overrides(),
                        num_games=0,
                        opponent_payload={},
                        stream_results_to_queue=True,
                        hard_start_positions=[],
                        stream_game_queue=True,
                        initial_game_jobs=initial_jobs_by_rank.get(int(rank), []),
                    )
                    rank = int(rank)
                    active_workers[rank] = {
                        'chunk_games': 0,
                        'progress_file': progress_file,
                        'streaming': True,
                    }
                    current_progress_files[rank] = progress_file
                    result_files.append(result_file)
                    progress_files.append(progress_file)
            else:
                worker_hard_starts = {}
                hard_start_offset = 0
                for worker_rank, worker_games in worker_specs:
                    worker_hard_starts[int(worker_rank)] = hard_start_positions[
                        hard_start_offset:hard_start_offset + int(worker_games)
                    ]
                    hard_start_offset += int(worker_games)
                result_files, progress_files = selfplay_pool.submit(
                    task_id=task_id,
                    model_state_path=model_state_path,
                    model_state=model_state_cpu,
                    temperature=rl_cfg.get('mcts_temperature'),
                    runtime_overrides=_selfplay_runtime_overrides(),
                    worker_model_state_paths={},
                    worker_opponent_payloads=opponent_assignments,
                    stream_results_to_queue=use_queue_transport,
                    worker_hard_start_positions=worker_hard_starts,
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
                        "hard_start_positions": list(worker_hard_starts.get(int(rank), []) or []),
                    }
            processes = [selfplay_pool.processes[rank] for rank, _ in worker_specs]
        else:
            mp_ctx = mp.get_context('spawn')
            one_shot_hard_start_offset = 0
            for rank, games_for_worker in worker_specs:
                if device_type == 'cuda':
                    device_id = rank % torch.cuda.device_count()
                else:
                    device_id = 'cpu'

                result_file = temp_dir / f"worker_{rank}_mcts_results.pkl"
                result_files.append(result_file)
                worker_config = dict(config)
                worker_rl_cfg = dict(rl_cfg)
                worker_rl_cfg['hard_start_positions'] = hard_start_positions[
                    one_shot_hard_start_offset:one_shot_hard_start_offset + int(games_for_worker)
                ]
                one_shot_hard_start_offset += int(games_for_worker)
                worker_config['reinforcement_learning'] = worker_rl_cfg

                p = mp_ctx.Process(
                    target=run_selfplay_worker,
                    args=(
                        rank,
                        model_state,
                        worker_config,
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
            worker_finished_at = {}
            tail_90_reached_at = None
            pending = (
                set(active_workers.keys())
                if use_persistent_pool and global_game_stream_enabled
                else {rank for rank, _ in worker_specs}
            )
            rank_to_index = {rank: idx for idx, (rank, _) in enumerate(worker_specs)}
            games_reported = {rank: 0 for rank, _ in worker_specs}
            games_bar_last = 0

            def _redispatch_after_worker_crash(rank, exitcode):
                raise RuntimeError(
                    f"Persistent self-play worker {int(rank)} exited unexpectedly with code {exitcode}; "
                    "the iteration is aborted because a claimed game cannot be reconstructed safely."
                )

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
                if tail_90_reached_at is None and total_done >= math.ceil(0.90 * num_games):
                    tail_90_reached_at = time.perf_counter()

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
                                    fens = packed.get('fens')
                                    root_q_targets = packed.get('root_q_targets')
                                    history_fens = packed.get('history_fens')
                                    search_changed_top = packed.get('search_changed_top')
                                    search_q_deltas = packed.get('search_q_deltas')
                                    best_q_targets = packed.get('best_q_targets')
                                    played_q_targets = packed.get('played_q_targets')
                                    orig_q_targets = packed.get('orig_q_targets')
                                    policy_kld_targets = packed.get('policy_kld_targets')
                                    search_visits = packed.get('search_visits')
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
                                            fens=fens,
                                            root_q_targets=root_q_targets,
                                            history_fens=history_fens,
                                            search_changed_top=search_changed_top,
                                            search_q_deltas=search_q_deltas,
                                            best_q_targets=best_q_targets,
                                            played_q_targets=played_q_targets,
                                            orig_q_targets=orig_q_targets,
                                            policy_kld_targets=policy_kld_targets,
                                            search_visits=search_visits,
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
                                queue_search_fresh_simulations_used_sum += int(
                                    chunk_stats.get('search_fresh_simulations_used_sum', 0)
                                )
                                queue_search_inherited_visit_credit_sum += int(
                                    chunk_stats.get('search_inherited_visit_credit_sum', 0)
                                )
                                queue_search_simulations_budget_sum += int(chunk_stats.get('search_simulations_budget_sum', 0))
                                queue_search_samples += int(chunk_stats.get('search_samples', 0))
                                queue_search_difficulty_samples += int(chunk_stats.get('search_difficulty_samples', 0))
                                queue_search_difficulty_sum += float(chunk_stats.get('search_difficulty_sum', 0.0) or 0.0)
                                queue_search_difficulty_sq_sum += float(chunk_stats.get('search_difficulty_sq_sum', 0.0) or 0.0)
                                queue_search_difficulty_budget_cross_sum += float(
                                    chunk_stats.get('search_difficulty_budget_cross_sum', 0.0) or 0.0
                                )
                                queue_search_budget_sq_sum += float(chunk_stats.get('search_budget_sq_sum', 0.0) or 0.0)
                                queue_tree_reuse_attempts += int(chunk_stats.get('tree_reuse_attempts', 0))
                                queue_tree_reuse_hits += int(chunk_stats.get('tree_reuse_hits', 0))
                                queue_tree_inherited_visits_sum += int(
                                    chunk_stats.get('tree_inherited_visits_sum', 0) or 0
                                )
                                queue_tree_reuse_credit_samples += int(
                                    chunk_stats.get('tree_reuse_credit_samples', 0) or 0
                                )
                                queue_tree_reuse_quality_sum += float(
                                    chunk_stats.get('tree_reuse_quality_sum', 0.0) or 0.0
                                )
                                queue_tree_reuse_candidate_coverage_sum += float(
                                    chunk_stats.get('tree_reuse_candidate_coverage_sum', 0.0) or 0.0
                                )
                                queue_tree_reuse_visited_prior_mass_sum += float(
                                    chunk_stats.get('tree_reuse_visited_prior_mass_sum', 0.0) or 0.0
                                )
                                queue_tree_reuse_fresh_floor_sum += int(
                                    chunk_stats.get('tree_reuse_fresh_floor_sum', 0) or 0
                                )
                                queue_tree_reuse_scout_stability_sum += float(
                                    chunk_stats.get('tree_reuse_scout_stability_sum', 0.0) or 0.0
                                )
                                queue_tree_reuse_scout_extra_credit_sum += int(
                                    chunk_stats.get('tree_reuse_scout_extra_credit_sum', 0) or 0
                                )
                                queue_tree_reuse_scout_reduced_count += int(
                                    chunk_stats.get('tree_reuse_scout_reduced_count', 0) or 0
                                )
                                queue_shared_tree_searches += int(chunk_stats.get('shared_tree_searches', 0))
                                queue_hard_start_games += int(chunk_stats.get('hard_start_games', 0))
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
                            worker_finished_at[int(rank)] = time.perf_counter()
                            if not message.get('ok', False):
                                interrupted = bool(message.get('interrupt', False))
                                if interrupted:
                                    print("\nCtrl+C detected in self-play worker. Stopping workers...")
                                    raise RLTrainingInterrupted("self-play")
                                raise RuntimeError(
                                    f"Persistent self-play worker {rank} failed: {message.get('error', 'unknown error')}"
                                )
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
    
    selfplay_finished_at = time.perf_counter()
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
    total_search_fresh_simulations_used_sum = (
        queue_search_fresh_simulations_used_sum if use_queue_transport else 0
    )
    total_search_inherited_visit_credit_sum = (
        queue_search_inherited_visit_credit_sum if use_queue_transport else 0
    )
    total_search_simulations_budget_sum = queue_search_simulations_budget_sum if use_queue_transport else 0
    total_search_samples = queue_search_samples if use_queue_transport else 0
    total_search_difficulty_samples = queue_search_difficulty_samples if use_queue_transport else 0
    total_search_difficulty_sum = queue_search_difficulty_sum if use_queue_transport else 0.0
    total_search_difficulty_sq_sum = queue_search_difficulty_sq_sum if use_queue_transport else 0.0
    total_search_difficulty_budget_cross_sum = queue_search_difficulty_budget_cross_sum if use_queue_transport else 0.0
    total_search_budget_sq_sum = queue_search_budget_sq_sum if use_queue_transport else 0.0
    total_tree_reuse_attempts = queue_tree_reuse_attempts if use_queue_transport else 0
    total_tree_reuse_hits = queue_tree_reuse_hits if use_queue_transport else 0
    total_tree_inherited_visits_sum = queue_tree_inherited_visits_sum if use_queue_transport else 0
    total_tree_reuse_credit_samples = (
        queue_tree_reuse_credit_samples if use_queue_transport else 0
    )
    total_tree_reuse_quality_sum = queue_tree_reuse_quality_sum if use_queue_transport else 0.0
    total_tree_reuse_candidate_coverage_sum = (
        queue_tree_reuse_candidate_coverage_sum if use_queue_transport else 0.0
    )
    total_tree_reuse_visited_prior_mass_sum = (
        queue_tree_reuse_visited_prior_mass_sum if use_queue_transport else 0.0
    )
    total_tree_reuse_fresh_floor_sum = (
        queue_tree_reuse_fresh_floor_sum if use_queue_transport else 0
    )
    total_tree_reuse_scout_stability_sum = (
        queue_tree_reuse_scout_stability_sum if use_queue_transport else 0.0
    )
    total_tree_reuse_scout_extra_credit_sum = (
        queue_tree_reuse_scout_extra_credit_sum if use_queue_transport else 0
    )
    total_tree_reuse_scout_reduced_count = (
        queue_tree_reuse_scout_reduced_count if use_queue_transport else 0
    )
    total_shared_tree_searches = queue_shared_tree_searches if use_queue_transport else 0
    total_hard_start_games = queue_hard_start_games if use_queue_transport else 0
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
                            total_search_fresh_simulations_used_sum += int(
                                (stats or {}).get('search_fresh_simulations_used_sum', 0)
                            )
                            total_search_inherited_visit_credit_sum += int(
                                (stats or {}).get('search_inherited_visit_credit_sum', 0)
                            )
                            total_search_simulations_budget_sum += int((stats or {}).get('search_simulations_budget_sum', 0))
                            total_search_samples += int((stats or {}).get('search_samples', 0))
                            total_search_difficulty_samples += int((stats or {}).get('search_difficulty_samples', 0))
                            total_search_difficulty_sum += float((stats or {}).get('search_difficulty_sum', 0.0) or 0.0)
                            total_search_difficulty_sq_sum += float((stats or {}).get('search_difficulty_sq_sum', 0.0) or 0.0)
                            total_search_difficulty_budget_cross_sum += float(
                                (stats or {}).get('search_difficulty_budget_cross_sum', 0.0) or 0.0
                            )
                            total_search_budget_sq_sum += float((stats or {}).get('search_budget_sq_sum', 0.0) or 0.0)
                            total_tree_reuse_attempts += int((stats or {}).get('tree_reuse_attempts', 0))
                            total_tree_reuse_hits += int((stats or {}).get('tree_reuse_hits', 0))
                            total_tree_inherited_visits_sum += int(
                                (stats or {}).get('tree_inherited_visits_sum', 0) or 0
                            )
                            total_tree_reuse_credit_samples += int(
                                (stats or {}).get('tree_reuse_credit_samples', 0) or 0
                            )
                            total_tree_reuse_quality_sum += float(
                                (stats or {}).get('tree_reuse_quality_sum', 0.0) or 0.0
                            )
                            total_tree_reuse_candidate_coverage_sum += float(
                                (stats or {}).get('tree_reuse_candidate_coverage_sum', 0.0) or 0.0
                            )
                            total_tree_reuse_visited_prior_mass_sum += float(
                                (stats or {}).get('tree_reuse_visited_prior_mass_sum', 0.0) or 0.0
                            )
                            total_tree_reuse_fresh_floor_sum += int(
                                (stats or {}).get('tree_reuse_fresh_floor_sum', 0) or 0
                            )
                            total_tree_reuse_scout_stability_sum += float(
                                (stats or {}).get('tree_reuse_scout_stability_sum', 0.0) or 0.0
                            )
                            total_tree_reuse_scout_extra_credit_sum += int(
                                (stats or {}).get('tree_reuse_scout_extra_credit_sum', 0) or 0
                            )
                            total_tree_reuse_scout_reduced_count += int(
                                (stats or {}).get('tree_reuse_scout_reduced_count', 0) or 0
                            )
                            total_shared_tree_searches += int((stats or {}).get('shared_tree_searches', 0))
                            total_hard_start_games += int((stats or {}).get('hard_start_games', 0))
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
    _recompute_profile_ratios(total_profile_stats)
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
    if total_search_difficulty_samples > 0:
        difficulty_n = float(total_search_difficulty_samples)
        difficulty_mean = total_search_difficulty_sum / difficulty_n
        budget_mean_for_difficulty = total_search_simulations_budget_sum / difficulty_n
        difficulty_var = max(
            0.0,
            total_search_difficulty_sq_sum / difficulty_n - difficulty_mean ** 2,
        )
        budget_var = max(
            0.0,
            total_search_budget_sq_sum / difficulty_n - budget_mean_for_difficulty ** 2,
        )
        difficulty_budget_cov = (
            total_search_difficulty_budget_cross_sum / difficulty_n
            - difficulty_mean * budget_mean_for_difficulty
        )
        difficulty_budget_denom = math.sqrt(difficulty_var * budget_var)
        difficulty_budget_correlation = (
            difficulty_budget_cov / difficulty_budget_denom
            if difficulty_budget_denom > 1e-12 else 0.0
        )
    else:
        difficulty_mean = 0.0
        difficulty_budget_correlation = 0.0
    worker_game_counts = np.asarray(
        [int(completed_games_by_rank.get(int(rank), 0)) for rank, _ in worker_specs],
        dtype=np.float32,
    )
    worker_finish_values = list(worker_finished_at.values()) if 'worker_finished_at' in locals() else []
    selfplay_stats = {
        'startup_time': float(selfplay_startup_time),
        'worker_count': int(len(worker_specs)),
        'games_per_worker_mean': (
            float(num_games) / float(len(worker_specs)) if worker_specs else 0.0
        ),
        'games_per_worker_min': float(worker_game_counts.min()) if worker_game_counts.size else 0.0,
        'games_per_worker_max': float(worker_game_counts.max()) if worker_game_counts.size else 0.0,
        'games_per_worker_std': float(worker_game_counts.std()) if worker_game_counts.size else 0.0,
        'worker_finish_spread_s': (
            float(max(worker_finish_values) - min(worker_finish_values))
            if len(worker_finish_values) > 1 else 0.0
        ),
        'tail_10pct_time_s': (
            max(0.0, float(selfplay_finished_at - tail_90_reached_at))
            if tail_90_reached_at is not None else 0.0
        ),
        'global_game_stream_enabled': int(bool(global_game_stream_enabled)),
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
        'replay_candidate_positions': int(total_positions + filtered_positions),
        'played_positions': int(total_generated_positions),
        'resigned_games': int(total_resigned_games),
        'replay_positions_per_sec': float(replay_positions_per_sec),
        'played_positions_per_sec': float(played_positions_per_sec),
        # One completed simulation/visit is one root->leaf selection followed
        # by terminal/NN evaluation and backpropagation. Do not multiply it by
        # path length: that separate diagnostic is a traversal counter.
        'mcts_simulations_per_sec': (
            float(total_search_fresh_simulations_used_sum) / float(total_time)
            if total_time > 0.0
            else 0.0
        ),
        'mcts_effective_simulations_per_sec': (
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
        'search_fresh_simulations_used_avg': float(total_search_fresh_simulations_used_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_inherited_visit_credit_avg': float(total_search_inherited_visit_credit_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_simulations_budget_avg': float(total_search_simulations_budget_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_simulations_budget_p10': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 10)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_p50': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 50)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_p90': float(np.percentile(np.asarray(total_search_simulations_budget_samples, dtype=np.float32), 90)) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_min': float(np.min(np.asarray(total_search_simulations_budget_samples, dtype=np.float32))) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_max': float(np.max(np.asarray(total_search_simulations_budget_samples, dtype=np.float32))) if total_search_simulations_budget_samples else 0.0,
        'search_simulations_budget_target': float(
            rl_cfg.get('mcts_simulations', 0)
        ),
        'search_simulations_used_p10': float(np.percentile(np.asarray(total_search_simulations_used_samples, dtype=np.float32), 10)) if total_search_simulations_used_samples else 0.0,
        'search_samples': int(total_search_samples),
        'search_difficulty_samples': int(total_search_difficulty_samples),
        'search_difficulty_mean': float(difficulty_mean),
        'search_difficulty_budget_correlation': float(difficulty_budget_correlation),
        'tree_reuse_attempts': int(total_tree_reuse_attempts),
        'tree_reuse_hits': int(total_tree_reuse_hits),
        'tree_reuse_hit_rate': (
            float(total_tree_reuse_hits) / float(total_tree_reuse_attempts)
            if total_tree_reuse_attempts > 0 else 0.0
        ),
        'tree_inherited_visits_avg': (
            float(total_tree_inherited_visits_sum) / float(total_tree_reuse_hits)
            if total_tree_reuse_hits > 0 else 0.0
        ),
        'tree_reuse_credit_samples': int(total_tree_reuse_credit_samples),
        'tree_reuse_quality_avg': (
            float(total_tree_reuse_quality_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_candidate_coverage_avg': (
            float(total_tree_reuse_candidate_coverage_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_visited_prior_mass_avg': (
            float(total_tree_reuse_visited_prior_mass_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_fresh_floor_avg': (
            float(total_tree_reuse_fresh_floor_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_scout_stability_avg': (
            float(total_tree_reuse_scout_stability_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_scout_extra_credit_avg': (
            float(total_tree_reuse_scout_extra_credit_sum) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'tree_reuse_scout_reduction_rate': (
            float(total_tree_reuse_scout_reduced_count) / float(total_tree_reuse_credit_samples)
            if total_tree_reuse_credit_samples > 0 else 0.0
        ),
        'shared_tree_search_fraction': (
            float(total_shared_tree_searches) / float(total_search_samples)
            if total_search_samples > 0 else 0.0
        ),
        'hard_start_games': int(total_hard_start_games),
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
    config_path = default_config_path()
    
    print(f"Chess RL trainer | config: {compact_path(config_path, 3)}")
    config = load_project_config()

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
            "(failed to import src.selfplay.workers.run_selfplay_worker)."
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
    fresh_il_start_recommended = bool(
        config['reinforcement_learning'].get('fresh_il_start', False)
    )
    if fresh_il_start_recommended:
        rl_init_checkpoint_path = best_model_il_path
        rl_init_checkpoint_reason = "RL49 recommended fresh start from canonical best_model_il.pt"
    else:
        best_model_il_swa_path = best_model_il_path.parent / "best_model_il_swa.pt"
        rl_init_checkpoint_path, rl_init_checkpoint_reason = select_best_il_checkpoint(
            best_model_il_path,
            best_model_il_swa_path,
        )
    print(f"RL IL initialization: {rl_init_checkpoint_path.name} ({rl_init_checkpoint_reason}).")
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
        prefer_fresh_il=fresh_il_start_recommended,
    )

    # Build/load once in the parent before Windows spawn starts the persistent
    # workers. Children then only map the source-hashed cached DLL.
    native_mcts = get_native_mcts()
    if native_mcts is None:
        print("Native MCTS: unavailable, using Python/NumPy tree search.")
    else:
        print(
            f"Native MCTS: {native_mcts.backend_name}, "
            "persistent full-tree traversal + backup (cached)."
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
    model = create_model(config).to(device)
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

    controlled_new_init = bool(
        fresh_il_start_recommended
        and startup_plan.get("start_mode") == "new"
        and startup_plan.get("new_init_mode", "default") == "default"
    )
    startup_state = apply_rl_startup_plan(
        startup_plan=startup_plan,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        default_new_checkpoint=rl_init_checkpoint_path,
        require_compatible_new_init=controlled_new_init,
    )
    _enforce_uniform_rl_optimizer_contract(
        optimizer,
        config['reinforcement_learning'],
    )
    start_mode = startup_state.get("start_mode", "new")
    selected_checkpoint_label = startup_state.get("selected_checkpoint_label")
    start_iteration = int(startup_state.get("start_iteration", 0) or 0)
    resumed_best_win_rate = float(startup_state.get("best_win_rate", 0.0) or 0.0)
    selected_compatibility_ratio = startup_state.get("selected_compatibility_ratio")
    transfer_match_ratio = startup_state.get("transfer_match_ratio")
    new_init_mode = startup_state.get("new_init_mode", startup_plan.get("new_init_mode", "default"))
    selected_checkpoint_path = startup_state.get("selected_checkpoint")
    if fresh_il_start_recommended and not controlled_new_init:
        print(
            "RL49 note: the recommended fresh canonical-IL start was explicitly "
            f"overridden by menu choice ({start_mode}/{new_init_mode})."
        )
    resume_runtime_payload = {}
    if start_mode == "resume" and selected_checkpoint_path is not None:
        with contextlib.suppress(Exception):
            resume_runtime_payload = torch.load(
                selected_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            ) or {}
        resume_history_csv = _resolve_resume_history_csv(
            resume_runtime_payload,
            checkpoint_path=selected_checkpoint_path,
            base_dir=base_dir,
            logs_dir=logs_dir,
            experiment_name=build_rl_experiment_name(config),
            completed_iteration=start_iteration,
            exclude_path=logger.csv_path,
        )
        history_imported = bool(
            resume_history_csv is not None
            and logger.import_rl_history_from_csv(
                resume_history_csv,
                start_iteration,
            )
        )
        if history_imported:
            print(
                "Resume plots: copied iterations "
                f"1-{start_iteration} from {resume_history_csv.name}; "
                f"red boundary before iteration {start_iteration + 1}."
            )
        else:
            logger.add_resume_marker(start_iteration)
            print(
                "Resume plots: prior CSV was not found; continuing with a "
                f"boundary before iteration {start_iteration + 1}."
            )
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
    best_model = create_model(config).to(device)
    best_model = best_model.to(memory_format=torch.channels_last)
    best_model.load_state_dict(model.state_dict())
    restored_best = False
    if resume_runtime_payload:
        with contextlib.suppress(Exception):
            restored_best = _load_model_snapshot(
                best_model,
                resume_runtime_payload.get('best_model_state_dict'),
            )
    selected_is_version_latest = bool(
        selected_checkpoint_path is not None
        and Path(selected_checkpoint_path).resolve() == latest_checkpoint_path.resolve()
    )
    if start_mode == 'resume' and selected_is_version_latest and not restored_best:
        with contextlib.suppress(Exception):
            payload = load_checkpoint_file(str(version_best_model_path), device)
            restored_best = _load_model_snapshot(
                best_model,
                payload.get('model_state_dict') if isinstance(payload, dict) else None,
            )
    # The promoted best is an immutable evaluation floor. Self-play normally
    # follows a per-iteration learner snapshot, with the best used only to seed
    # champion replay and to recover from a clearly unsafe learner snapshot.
    # Redundant legacy model snapshots in old runtime checkpoints are ignored.
    best_model.eval()
    best_model.requires_grad_(False)
    best_iteration = int(
        resume_runtime_payload.get(
            'best_iteration',
            start_iteration,
        ) or 0
    )
    anchor_model = None
    anchor_model_available = False
    if bool(config['reinforcement_learning'].get('anchor_eval_enabled', False)) and rl_init_checkpoint_path.exists():
        try:
            anchor_model = create_model(config).to(device)
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
    best_differs_from_anchor = bool(
        anchor_model_available
        and anchor_model is not None
        and not _models_have_identical_state(best_model, anchor_model)
    )
    training_model = model
    training_compile_enabled = bool(
        device.type == 'cuda'
        and config.get('hardware', {}).get('use_compile', False)
        and hasattr(torch, 'compile')
    )
    if training_compile_enabled:
        try:
            configure_torch_compile_cache(torch, device, "rl_learner")
            # Only the learner forward/backward is compiled. Self-play,
            # evaluation and checkpoints keep using the original module, which
            # shares these exact parameters and preserves clean state_dict keys.
            training_model = torch.compile(model, mode='default', dynamic=True)
            print(
                "RL learner: torch.compile enabled "
                "(first non-diagnostic training batch compiles lazily)."
            )
        except Exception as exc:
            training_model = model
            training_compile_enabled = False
            print(f"RL learner: torch.compile unavailable, using eager mode ({exc}).")
    rl_cfg = config['reinforcement_learning']
    # Best files are updated only when evaluation confirms model improvement.
    replay_fp16 = config['reinforcement_learning'].get('replay_fp16', False)
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
    replay_buffer = ReplayBuffer(
        config['reinforcement_learning']['replay_buffer_size'],
        max_policy_targets=replay_max_policy_targets,
        use_fp16=replay_fp16,
    )
    champion_replay_fraction = max(
        0.0,
        min(0.40, float(rl_cfg.get('replay_champion_fraction', 0.15) or 0.0)),
    )
    champion_replay_buffer = ReplayBuffer(
        max(1, int(round(replay_buffer.max_size * champion_replay_fraction))),
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
    score_rate_threshold = float(rl_cfg.get('score_rate_threshold', 0.55))
    true_win_rate_threshold = float(rl_cfg.get('true_win_rate_threshold', 0.0))
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
        1.0,
    )
    funnel_advanced_games = max(0, int(rl_cfg.get('eval_funnel_advanced_games', 120)))
    funnel_advanced_simulations = _resolve_eval_stage_simulations(
        'eval_funnel_advanced_simulations',
        'eval_funnel_later_simulations_multiplier',
        1.0,
    )
    funnel_preliminary_score_rate = float(rl_cfg.get('eval_funnel_preliminary_score_rate', 0.45))
    funnel_preliminary_true_win_rate = float(rl_cfg.get('eval_funnel_preliminary_true_win_rate', 0.0))
    funnel_medium_score_rate = float(rl_cfg.get('eval_funnel_medium_score_rate', 0.53))
    funnel_medium_true_win_rate = float(rl_cfg.get('eval_funnel_medium_true_win_rate', 0.0))
    anchor_eval_simulations = _resolve_eval_stage_simulations(
        'anchor_eval_mcts_simulations',
        'anchor_eval_mcts_simulations_multiplier',
        1.0,
    )
    eval_every = max(1, int(rl_cfg.get('eval_every', 1) or 1))
    no_mcts_eval_enabled = bool(rl_cfg.get('eval_no_mcts_enabled', True))
    no_mcts_eval_every = max(1, int(rl_cfg.get('eval_no_mcts_every', 2)))
    no_mcts_eval_games = max(1, int(rl_cfg.get('eval_no_mcts_games', 30)))
    diagnostic_ema_alpha = max(
        0.0,
        min(1.0, float(rl_cfg.get('rl_diagnostic_ema_alpha', 0.35) or 0.35)),
    )
    last_promotion_iteration = best_iteration if best_iteration > 0 else None
    last_reanalyse_iteration = start_iteration if start_iteration > 0 else None
    eval_score_rate_ema = None
    eval_true_win_rate_ema = None
    learner_selfplay_safe = True
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

    persistent_eval_runtime = None
    previous_correction_audit_batch = None
    previous_correction_audit_after = None
    try:
        for iteration in range(start_iteration, total_iterations):
            print(f"\n[Iteration {iteration + 1}/{total_iterations}] self-play -> replay -> train")
            replay_buffer.set_current_iteration(iteration + 1)
            champion_replay_buffer.set_current_iteration(iteration + 1)
            iteration_profile_printed = False
            iteration_stage_times = {}
            iteration_start_time = time.perf_counter()
            iteration_stage_start = iteration_start_time
            iteration_eval_substage_times = {}
            iteration_eval_setup_time = 0.0
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

            def _evaluate_correction_audit(batch):
                if batch is None:
                    return None
                try:
                    return evaluate_correction_cohort(
                        model,
                        batch,
                        device,
                        use_amp=use_amp,
                        use_bfloat16=use_bfloat16,
                    )
                except Exception as exc:
                    iteration_notes.append(
                        f"correction audit skipped: {type(exc).__name__}: {str(exc)[:120]}"
                    )
                    return None

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

            def _prepare_eval_runtime(model1, model2, eval_config, num_games, eval_started_at):
                """Keep eval compilation out of promotion time and reuse it later."""
                nonlocal persistent_eval_runtime, iteration_eval_setup_time
                setup_t0 = time.perf_counter()
                if persistent_eval_runtime is None:
                    persistent_eval_runtime = borrow_selfplay_central_runtime()
                persistent_eval_runtime = prepare_eval_central_runtime(
                    persistent_eval_runtime,
                    model1,
                    model2,
                    eval_config,
                    device,
                    num_games,
                )
                setup_elapsed = max(0.0, time.perf_counter() - setup_t0)
                iteration_stage_times['setup'] = float(
                    iteration_stage_times.get('setup', 0.0) or 0.0
                ) + setup_elapsed
                iteration_eval_setup_time += setup_elapsed
                # The eval timer started before runtime preparation. Move its
                # origin forward so model snapshot/load/warmup is setup only.
                return persistent_eval_runtime, float(eval_started_at) + setup_elapsed

            def _discard_failed_eval_runtime(error_text):
                """Drop a possibly poisoned CUDA context; next iteration starts fresh."""
                nonlocal persistent_eval_runtime
                failed_runtime = persistent_eval_runtime
                persistent_eval_runtime = None
                with contextlib.suppress(Exception):
                    close_eval_central_runtime(failed_runtime)
                # A borrowed eval runtime is owned by the persistent self-play
                # pool.  Destroy that owner as well, otherwise the next eval or
                # self-play stage would reconnect to the same failed process.
                _shutdown_selfplay_pool(timeout_s=1.0)
                if device.type == 'cuda' and torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                error_text = str(error_text or "central inference failure")
                failure_kind = (
                    "transient CUDA/driver failure"
                    if is_transient_cuda_error(error_text)
                    else "central inference process failure"
                )
                print(
                    f"WARNING: MCTS evaluation skipped after {failure_kind}. "
                    "Training will continue and create a fresh inference server "
                    f"for the next iteration. Detail: {error_text}",
                    flush=True,
                )

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
            for group in optimizer.param_groups:
                group['lr'] = current_lr * float(group.get('lr_factor', 1.0))
            
            # Search stays fixed; evaluation only promotes a new best checkpoint.
            current_temp = float(config['reinforcement_learning']['mcts_temperature'])
            current_temp_threshold = fixed_mcts_temperature_threshold
            config['reinforcement_learning']['mcts_temperature'] = current_temp
            config['reinforcement_learning']['mcts_temperature_threshold'] = current_temp_threshold
            
            rl_cfg['current_iteration'] = int(iteration + 1)

            current_value_loss_weight = _compute_value_loss_weight(iteration)
            current_policy_loss_weight = float(rl_cfg.get('policy_loss_weight', 1.0))
            current_correction_rank_weight = resolve_policy_correction_rank_loss_weight(config)

            _finish_stage('setup')

            # Self-play with MCTS
            best_model.eval()
            replay_rotation_before = replay_buffer.storage_rotation_stats()
            games_this_iteration = int(config['reinforcement_learning']['games_per_iteration'])
            hard_start_plan = [None] * games_this_iteration
            hard_start_fraction = max(
                0.0,
                min(1.0, float(rl_cfg.get('self_play_hard_start_fraction', 0.18))),
            )
            hard_start_count = min(
                games_this_iteration,
                int(round(games_this_iteration * hard_start_fraction)),
            )
            if hard_start_count > 0 and len(replay_buffer) > 0:
                hard_indices = replay_buffer.select_hard_position_indices(
                    hard_start_count,
                    min_age=int(rl_cfg.get('self_play_hard_start_min_age', 1)),
                )
                hard_positions = replay_buffer.hard_positions_for_indices(hard_indices)
                if hard_positions:
                    slots = np.random.choice(
                        games_this_iteration,
                        size=len(hard_positions),
                        replace=False,
                    )
                    for slot, hard_position in zip(slots.tolist(), hard_positions):
                        hard_start_plan[int(slot)] = hard_position
                    iteration_notes.append(
                        f"hard starts: {len(hard_positions)}/{games_this_iteration} games"
                    )
            performance_profile = {}
            learner_only_bootstrap_iterations = max(
                0,
                int(rl_cfg.get('self_play_learner_only_bootstrap_iterations', 0) or 0),
            )
            champion_refresh_due = bool(
                iteration + 1 > learner_only_bootstrap_iterations
                and champion_replay_fraction > 0.0
                and len(champion_replay_buffer) <= 0
            )
            use_best_for_selfplay = _use_best_for_selfplay_iteration(
                iteration + 1,
                learner_only_bootstrap_iterations=learner_only_bootstrap_iterations,
                champion_refresh_due=champion_refresh_due,
                learner_selfplay_safe=learner_selfplay_safe,
            )
            selfplay_model = best_model if use_best_for_selfplay else model
            selfplay_source_code = (
                REPLAY_SOURCE_CHAMPION if use_best_for_selfplay else REPLAY_SOURCE_LEARNER
            )
            if iteration + 1 <= learner_only_bootstrap_iterations:
                selfplay_source_label = "learner snapshot (RL49 bootstrap)"
            elif champion_refresh_due:
                selfplay_source_label = "best bootstrap"
            elif use_best_for_selfplay:
                selfplay_source_label = "best safety fallback"
            else:
                selfplay_source_label = "learner snapshot"
            iteration_notes.append(f"self-play source: {selfplay_source_label}")
            positions, positions_added, avg_game_length, positions_per_sec, selfplay_time, collection_time, selfplay_stats = \
                play_games_parallel_mcts(
                    selfplay_model,
                    config,
                    device,
                    games_this_iteration,
                    replay_buffer=replay_buffer,
                    hard_start_positions=hard_start_plan,
                )
            selfplay_stats = dict(selfplay_stats or {})
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
            replay_buffer.relabel_iteration_source(
                iteration + 1,
                selfplay_source_code,
            )

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

            desired_champion_capacity = max(
                1,
                int(round(replay_buffer.max_size * champion_replay_fraction)),
            )
            champion_replay_buffer.resize(desired_champion_capacity)
            champion_rows_added = 0
            if champion_refresh_due:
                champion_indices = replay_buffer.select_iteration_indices(
                    iteration + 1,
                    max_count=champion_replay_buffer.max_size,
                    prefer_useful_search_corrections=True,
                )
                champion_rows_added = replay_buffer.copy_indices_to(
                    champion_replay_buffer,
                    champion_indices,
                    source_code=REPLAY_SOURCE_CHAMPION,
                )
                if champion_rows_added > 0:
                    iteration_notes.append(
                        f"champion replay: {len(champion_replay_buffer):,}/"
                        f"{champion_replay_buffer.max_size:,} pinned positions"
                    )

            replay_quality_stats = replay_buffer.quality_stats(
                recent_window_fraction=config['reinforcement_learning'].get('replay_recent_window_fraction', None)
            )
            replay_quality_stats['champion_replay_size'] = int(len(champion_replay_buffer))
            replay_quality_stats['champion_replay_capacity'] = int(champion_replay_buffer.max_size)
            replay_quality_stats['champion_replay_added'] = int(champion_rows_added)
            replay_rotation_after = replay_buffer.storage_rotation_stats()
            replay_quality_stats['overwritten_positions_iteration'] = max(
                0,
                int(replay_rotation_after['overwritten_positions'])
                - int(replay_rotation_before['overwritten_positions']),
            )
            replay_quality_stats['resize_dropped_positions_iteration'] = max(
                0,
                int(replay_rotation_after['resize_dropped_positions'])
                - int(replay_rotation_before['resize_dropped_positions']),
            )
            selfplay_profile = dict((selfplay_stats or {}).get('profile', {}) or {})
            if profile_training_enabled and selfplay_profile:
                print_selfplay_profiler(selfplay_profile, selfplay_time)
            performance_profile = dict(selfplay_profile)
            for metric_name in (
                'replay_positions_per_sec',
                'played_positions_per_sec',
                'mcts_simulations_per_sec',
                'mcts_effective_simulations_per_sec',
                'mcts_nn_evaluations_per_sec',
                'mcts_selection_node_traversals_per_sec',
                'worker_count',
                'games_per_worker_mean',
                'games_per_worker_min',
                'games_per_worker_max',
                'games_per_worker_std',
                'worker_finish_spread_s',
                'tail_10pct_time_s',
                'global_game_stream_enabled',
            ):
                performance_profile[metric_name] = float(
                    (selfplay_stats or {}).get(metric_name, 0.0) or 0.0
                )
            _finish_stage('replay')

            avg_policy_entropy = 0.0
            avg_value_pred_std = 0.0
            avg_target_value_std = 0.0
            correction_audit_batch = None
            correction_audit_before = None
            correction_audit_after = None
            correction_audit_previous_now = None
            iteration_correction_audit_metrics = correction_audit_metrics(None, None)
            champion_replay_age = (
                int(iteration + 1) - int(last_promotion_iteration)
                if last_promotion_iteration is not None
                # Until this run proves a replacement, the startup champion is
                # still the only verified policy anchor and must not decay.
                else 0
            )
            current_champion_replay_fraction = _resolve_champion_replay_sample_fraction(
                champion_replay_fraction,
                champion_replay_age,
            )
            replay_quality_stats['champion_replay_target_fraction'] = float(
                current_champion_replay_fraction
            )
            sample_age_avg = None
            sample_age_p10 = None
            sample_age_p50 = None
            sample_age_p90 = None
            sample_age_new_fraction = None
            sample_age_le1_fraction = None
            recent_sample_age_avg = None
            recent_sample_age_p50 = None
            recent_sample_age_p90 = None
            recent_sample_age_new_fraction = None
            recent_sample_age_le1_fraction = None
            champion_sample_age_avg = None
            champion_sample_age_p50 = None
            champion_sample_age_p90 = None
            
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
                training_model.train()
                total_loss = 0
                total_policy = 0
                total_value = 0
                total_policy_entropy = 0
                total_value_pred_std = 0
                total_target_value_std = 0
                policy_correction_rows = 0
                policy_rows = 0
                policy_correction_loss_sum = 0.0
                policy_correction_rank_rows = 0
                policy_correction_rank_loss_sum = 0.0
                policy_correction_top1_correct = 0
                policy_correction_sharpen_rows = 0
                policy_correction_sharpen_top1_before_sum = 0.0
                policy_correction_sharpen_top1_after_sum = 0.0
                policy_effective_weight_sum = 0.0
                policy_correction_effective_weight_sum = 0.0
                value_primary_loss_total = 0.0
                value_scalar_aux_loss_total = 0.0
                moves_left_loss_total = 0.0
                search_q_loss_total = 0.0
                search_q_rows_total = 0
                search_q_abs_error_sum_total = 0.0
                value_rows_total = 0
                value_pred_sum_total = 0.0
                value_target_sum_total = 0.0
                root_q_rows_total = 0
                root_q_pred_sum_total = 0.0
                root_q_target_sum_total = 0.0
                root_q_abs_error_sum_total = 0.0
                root_q_error_sum_total = 0.0
                root_q_pred_sq_sum_total = 0.0
                root_q_target_sq_sum_total = 0.0
                root_q_cross_sum_total = 0.0
                value_error_focus_rows_total = 0
                value_error_focus_weight_sum_total = 0.0
                value_effective_weight_sum_total = 0.0
                value_error_focus_abs_error_sum_total = 0.0
                wdl_pred_sums_total = np.zeros(3, dtype=np.float64)
                wdl_target_sums_total = np.zeros(3, dtype=np.float64)
                wdl_brier_sum_total = 0.0
                wdl_ece_counts_total = np.zeros(10, dtype=np.int64)
                wdl_ece_confidence_sums_total = np.zeros(10, dtype=np.float64)
                wdl_ece_correct_sums_total = np.zeros(10, dtype=np.float64)
                value_phase_wdl_totals = {
                    phase: {'rows': 0, 'draw_pred_sum': 0.0, 'draw_target_sum': 0.0}
                    for phase in ('opening', 'middlegame', 'endgame')
                }
                grad_total_norm_sum = 0.0
                grad_clip_scale_sum = 0.0
                grad_clipped_steps = 0
                gradient_probe_metrics = {}
                sample_age_batches = []
                recent_sample_age_batches = []
                champion_sample_age_batches = []
                
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
                training_batch_specs = []
                selected_recent_samples = 0
                selected_champion_samples = 0
                selected_recent_unique = np.zeros(replay_size, dtype=np.bool_)
                for indices in training_index_batches:
                    batch_count = int(indices.size)
                    champion_count = 0
                    if batch_count > 1 and len(champion_replay_buffer) > 0:
                        champion_count = min(
                            batch_count - 1,
                            int(round(batch_count * current_champion_replay_fraction)),
                            int(len(champion_replay_buffer)),
                        )
                    recent_indices = indices[:batch_count - champion_count]
                    training_batch_specs.append((recent_indices, champion_count))
                    selected_recent_samples += int(recent_indices.size)
                    selected_champion_samples += int(champion_count)
                    selected_recent_unique[recent_indices] = True
                selected_samples = selected_recent_samples + selected_champion_samples
                replay_quality_stats['train_batch_size'] = int(base_batch_size)
                replay_quality_stats['train_steps'] = int(total_train_steps)
                replay_quality_stats['train_selected_samples'] = int(selected_samples)
                replay_quality_stats['train_replay_coverage'] = (
                    float(np.count_nonzero(selected_recent_unique)) / float(replay_size)
                    if replay_size > 0
                    else 0.0
                )
                replay_quality_stats['train_replay_passes'] = (
                    float(selected_recent_samples) / float(replay_size)
                    if replay_size > 0
                    else 0.0
                )
                replay_quality_stats['champion_replay_selected_samples'] = int(selected_champion_samples)
                replay_quality_stats['champion_replay_selected_fraction'] = (
                    float(selected_champion_samples) / float(max(1, selected_samples))
                )
                planned_recent_parts = [
                    indices for indices, _ in training_batch_specs if int(indices.size) > 0
                ]
                planned_recent_indices = (
                    np.concatenate(planned_recent_parts)
                    if planned_recent_parts
                    else np.empty(0, dtype=np.int64)
                )
                audit_indices = replay_buffer.select_policy_correction_audit_indices(
                    planned_recent_indices,
                    CORRECTION_AUDIT_MAX_ROWS,
                    seed=0xC0DEC0DE + int(iteration),
                )
                if int(audit_indices.size) > 0:
                    correction_audit_batch = freeze_replay_batch(
                        replay_buffer.sample_from_indices(audit_indices)
                    )
                    correction_audit_before = _evaluate_correction_audit(correction_audit_batch)
                for batch_index, (recent_indices, champion_count) in enumerate(
                    tqdm(training_batch_specs, desc="Training")
                ):
                    recent_batch = replay_buffer.sample_from_indices(recent_indices)
                    age_parts = []
                    recent_ages = getattr(replay_buffer, 'last_sample_ages', None)
                    if recent_ages is not None and getattr(recent_ages, "size", 0) > 0:
                        age_parts.append(recent_ages.copy())
                        recent_sample_age_batches.append(recent_ages.copy())
                    champion_batch = None
                    if champion_count > 0:
                        champion_indices = champion_replay_buffer.select_indices(champion_count)
                        champion_batch = champion_replay_buffer.sample_from_indices(champion_indices)
                        champion_ages = getattr(champion_replay_buffer, 'last_sample_ages', None)
                        if champion_ages is not None and getattr(champion_ages, "size", 0) > 0:
                            age_parts.append(champion_ages.copy())
                            champion_sample_age_batches.append(champion_ages.copy())
                    batch = _merge_replay_batches(recent_batch, champion_batch)
                    if age_parts:
                        sample_age_batches.append(np.concatenate(age_parts))
                    # The first batch retains the graph for task-gradient
                    # diagnostics, which is incompatible with AOTAutograd buffer
                    # donation. Keep that one probe eager; compile all ordinary
                    # optimizer steps that dominate iteration runtime.
                    optimizer_step_model = (
                        model
                        if batch_index == 0
                        else training_model
                    )
                    (
                        loss,
                        policy_loss,
                        value_loss,
                        policy_entropy,
                        value_pred_std,
                        target_value_std,
                        policy_diagnostics,
                    ) = train_on_batch_rl(
                        optimizer_step_model,
                        optimizer,
                        batch,
                        config,
                        device,
                        scaler,
                        metrics_calc,
                        value_weight_override=current_value_loss_weight,
                        policy_weight_override=current_policy_loss_weight,
                        collect_gradient_diagnostics=(batch_index == 0),
                    )
                    
                    total_loss += loss
                    total_policy += policy_loss
                    total_value += value_loss
                    total_policy_entropy += policy_entropy
                    total_value_pred_std += value_pred_std
                    total_target_value_std += target_value_std
                    policy_correction_rows += int(policy_diagnostics.get('correction_rows', 0) or 0)
                    policy_rows += int(policy_diagnostics.get('policy_rows', 0) or 0)
                    policy_correction_loss_sum += float(
                        policy_diagnostics.get('correction_loss_sum', 0.0) or 0.0
                    )
                    policy_correction_rank_rows += int(
                        policy_diagnostics.get('correction_rank_rows', 0) or 0
                    )
                    policy_correction_rank_loss_sum += float(
                        policy_diagnostics.get('correction_rank_loss_sum', 0.0) or 0.0
                    )
                    policy_correction_top1_correct += int(
                        policy_diagnostics.get('correction_top1_correct', 0) or 0
                    )
                    policy_correction_sharpen_rows += int(
                        policy_diagnostics.get('correction_sharpen_rows', 0) or 0
                    )
                    policy_correction_sharpen_top1_before_sum += float(
                        policy_diagnostics.get(
                            'correction_sharpen_top1_before_sum', 0.0
                        ) or 0.0
                    )
                    policy_correction_sharpen_top1_after_sum += float(
                        policy_diagnostics.get(
                            'correction_sharpen_top1_after_sum', 0.0
                        ) or 0.0
                    )
                    policy_effective_weight_sum += float(
                        policy_diagnostics.get('effective_weight_sum', 0.0) or 0.0
                    )
                    policy_correction_effective_weight_sum += float(
                        policy_diagnostics.get('correction_effective_weight_sum', 0.0) or 0.0
                    )
                    value_primary_loss_total += float(
                        policy_diagnostics.get('value_primary_loss', 0.0) or 0.0
                    )
                    value_scalar_aux_loss_total += float(
                        policy_diagnostics.get('value_scalar_aux_loss', 0.0) or 0.0
                    )
                    moves_left_loss_total += float(
                        policy_diagnostics.get('moves_left_loss', 0.0) or 0.0
                    )
                    search_q_loss_total += float(
                        policy_diagnostics.get('search_q_loss', 0.0) or 0.0
                    )
                    search_q_rows_total += int(
                        policy_diagnostics.get('search_q_rows', 0) or 0
                    )
                    search_q_abs_error_sum_total += float(
                        policy_diagnostics.get('search_q_abs_error_sum', 0.0) or 0.0
                    )
                    value_rows_total += int(policy_diagnostics.get('value_rows', 0) or 0)
                    value_pred_sum_total += float(
                        policy_diagnostics.get('value_pred_sum', 0.0) or 0.0
                    )
                    value_target_sum_total += float(
                        policy_diagnostics.get('value_target_sum', 0.0) or 0.0
                    )
                    root_q_rows_total += int(policy_diagnostics.get('root_q_rows', 0) or 0)
                    root_q_pred_sum_total += float(
                        policy_diagnostics.get('root_q_pred_sum', 0.0) or 0.0
                    )
                    root_q_target_sum_total += float(
                        policy_diagnostics.get('root_q_target_sum', 0.0) or 0.0
                    )
                    root_q_abs_error_sum_total += float(
                        policy_diagnostics.get('root_q_abs_error_sum', 0.0) or 0.0
                    )
                    root_q_error_sum_total += float(
                        policy_diagnostics.get('root_q_error_sum', 0.0) or 0.0
                    )
                    root_q_pred_sq_sum_total += float(
                        policy_diagnostics.get('root_q_pred_sq_sum', 0.0) or 0.0
                    )
                    root_q_target_sq_sum_total += float(
                        policy_diagnostics.get('root_q_target_sq_sum', 0.0) or 0.0
                    )
                    root_q_cross_sum_total += float(
                        policy_diagnostics.get('root_q_cross_sum', 0.0) or 0.0
                    )
                    value_error_focus_rows_total += int(
                        policy_diagnostics.get('value_error_focus_rows', 0) or 0
                    )
                    value_error_focus_weight_sum_total += float(
                        policy_diagnostics.get('value_error_focus_weight_sum', 0.0) or 0.0
                    )
                    value_effective_weight_sum_total += float(
                        policy_diagnostics.get('value_effective_weight_sum', 0.0) or 0.0
                    )
                    value_error_focus_abs_error_sum_total += float(
                        policy_diagnostics.get('value_error_focus_abs_error_sum', 0.0) or 0.0
                    )
                    wdl_pred_sums_total += np.asarray(
                        policy_diagnostics.get('wdl_pred_sums', [0.0, 0.0, 0.0]),
                        dtype=np.float64,
                    )
                    wdl_target_sums_total += np.asarray(
                        policy_diagnostics.get('wdl_target_sums', [0.0, 0.0, 0.0]),
                        dtype=np.float64,
                    )
                    wdl_brier_sum_total += float(
                        policy_diagnostics.get('wdl_brier_sum', 0.0) or 0.0
                    )
                    wdl_ece_counts_total += np.asarray(
                        policy_diagnostics.get('wdl_ece_counts', [0] * 10),
                        dtype=np.int64,
                    )
                    wdl_ece_confidence_sums_total += np.asarray(
                        policy_diagnostics.get('wdl_ece_confidence_sums', [0.0] * 10),
                        dtype=np.float64,
                    )
                    wdl_ece_correct_sums_total += np.asarray(
                        policy_diagnostics.get('wdl_ece_correct_sums', [0.0] * 10),
                        dtype=np.float64,
                    )
                    for phase_name, phase_stats in dict(
                        policy_diagnostics.get('value_phase_wdl', {}) or {}
                    ).items():
                        if phase_name not in value_phase_wdl_totals:
                            continue
                        value_phase_wdl_totals[phase_name]['rows'] += int(
                            phase_stats.get('rows', 0) or 0
                        )
                        value_phase_wdl_totals[phase_name]['draw_pred_sum'] += float(
                            phase_stats.get('draw_pred_sum', 0.0) or 0.0
                        )
                        value_phase_wdl_totals[phase_name]['draw_target_sum'] += float(
                            phase_stats.get('draw_target_sum', 0.0) or 0.0
                        )
                    grad_total_norm_sum += float(
                        policy_diagnostics.get('total_grad_norm', 0.0) or 0.0
                    )
                    grad_clip_scale_sum += float(
                        policy_diagnostics.get('grad_clip_scale', 1.0) or 0.0
                    )
                    grad_clipped_steps += int(
                        policy_diagnostics.get('grad_was_clipped', 0) or 0
                    )
                    if batch_index == 0:
                        gradient_probe_metrics = dict(policy_diagnostics)
                if correction_audit_batch is not None:
                    correction_audit_after = _evaluate_correction_audit(correction_audit_batch)
                if previous_correction_audit_batch is not None:
                    correction_audit_previous_now = _evaluate_correction_audit(
                        previous_correction_audit_batch
                    )
                iteration_correction_audit_metrics = correction_audit_metrics(
                    correction_audit_before,
                    correction_audit_after,
                    previous_after=previous_correction_audit_after,
                    previous_now=correction_audit_previous_now,
                )
                previous_correction_audit_batch = correction_audit_batch
                previous_correction_audit_after = correction_audit_after
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
                if recent_sample_age_batches:
                    recent_sample_ages = np.concatenate(recent_sample_age_batches).astype(
                        np.float32, copy=False
                    )
                    recent_sample_age_avg = float(np.mean(recent_sample_ages))
                    recent_sample_age_p50 = float(np.percentile(recent_sample_ages, 50))
                    recent_sample_age_p90 = float(np.percentile(recent_sample_ages, 90))
                    recent_sample_age_new_fraction = float(np.mean(recent_sample_ages <= 0.0))
                    recent_sample_age_le1_fraction = float(np.mean(recent_sample_ages <= 1.0))
                if champion_sample_age_batches:
                    champion_sample_ages = np.concatenate(champion_sample_age_batches).astype(
                        np.float32, copy=False
                    )
                    champion_sample_age_avg = float(np.mean(champion_sample_ages))
                    champion_sample_age_p50 = float(np.percentile(champion_sample_ages, 50))
                    champion_sample_age_p90 = float(np.percentile(champion_sample_ages, 90))
                
                # Compute metrics
                train_metrics = metrics_calc.compute()
                value_prediction_mean = (
                    value_pred_sum_total / float(value_rows_total)
                    if value_rows_total > 0 else 0.0
                )
                value_target_mean = (
                    value_target_sum_total / float(value_rows_total)
                    if value_rows_total > 0 else 0.0
                )
                if root_q_rows_total > 0:
                    root_q_n = float(root_q_rows_total)
                    root_q_pred_mean = root_q_pred_sum_total / root_q_n
                    root_q_target_mean = root_q_target_sum_total / root_q_n
                    root_q_cov = (
                        root_q_cross_sum_total / root_q_n
                        - root_q_pred_mean * root_q_target_mean
                    )
                    root_q_pred_var = max(
                        0.0,
                        root_q_pred_sq_sum_total / root_q_n - root_q_pred_mean ** 2,
                    )
                    root_q_target_var = max(
                        0.0,
                        root_q_target_sq_sum_total / root_q_n - root_q_target_mean ** 2,
                    )
                    root_q_corr_denom = math.sqrt(root_q_pred_var * root_q_target_var)
                    value_root_q_correlation = (
                        root_q_cov / root_q_corr_denom
                        if root_q_corr_denom > 1e-12 else 0.0
                    )
                else:
                    root_q_pred_mean = 0.0
                    root_q_target_mean = 0.0
                    value_root_q_correlation = 0.0
                wdl_rows_total = int(round(float(wdl_target_sums_total.sum())))
                wdl_ece = 0.0
                if wdl_rows_total > 0:
                    for bin_idx, bin_count in enumerate(wdl_ece_counts_total):
                        if int(bin_count) <= 0:
                            continue
                        bin_confidence = (
                            wdl_ece_confidence_sums_total[bin_idx] / float(bin_count)
                        )
                        bin_accuracy = (
                            wdl_ece_correct_sums_total[bin_idx] / float(bin_count)
                        )
                        wdl_ece += (
                            float(bin_count) / float(wdl_rows_total)
                        ) * abs(bin_accuracy - bin_confidence)
                phase_wdl_metrics = {}
                for phase_name, phase_stats in value_phase_wdl_totals.items():
                    phase_rows = int(phase_stats['rows'])
                    phase_wdl_metrics[f'value_draw_probability_{phase_name}'] = (
                        phase_stats['draw_pred_sum'] / float(phase_rows)
                        if phase_rows > 0 else 0.0
                    )
                    phase_wdl_metrics[f'value_draw_target_fraction_{phase_name}'] = (
                        phase_stats['draw_target_sum'] / float(phase_rows)
                        if phase_rows > 0 else 0.0
                    )
                train_metrics.update({
                    'policy_correction_rank_weight': float(current_correction_rank_weight),
                    'policy_correction_loss': (
                        policy_correction_loss_sum / float(policy_correction_rows)
                        if policy_correction_rows > 0
                        else 0.0
                    ),
                    'policy_correction_rank_loss': (
                        policy_correction_rank_loss_sum
                        / float(policy_correction_rank_rows)
                        if policy_correction_rank_rows > 0
                        else 0.0
                    ),
                    'policy_correction_top1_acc': (
                        float(policy_correction_top1_correct) / float(policy_correction_rows)
                        if policy_correction_rows > 0
                        else 0.0
                    ),
                    'policy_correction_fraction': (
                        float(policy_correction_rows) / float(policy_rows)
                        if policy_rows > 0
                        else 0.0
                    ),
                    'policy_correction_weight_share': (
                        policy_correction_effective_weight_sum / policy_effective_weight_sum
                        if policy_effective_weight_sum > 0.0
                        else 0.0
                    ),
                    'policy_correction_sharpened_fraction': (
                        float(policy_correction_sharpen_rows) / float(policy_correction_rows)
                        if policy_correction_rows > 0
                        else 0.0
                    ),
                    'policy_correction_target_top1_before': (
                        policy_correction_sharpen_top1_before_sum
                        / float(policy_correction_sharpen_rows)
                        if policy_correction_sharpen_rows > 0
                        else 0.0
                    ),
                    'policy_correction_target_top1_after': (
                        policy_correction_sharpen_top1_after_sum
                        / float(policy_correction_sharpen_rows)
                        if policy_correction_sharpen_rows > 0
                        else 0.0
                    ),
                    'value_primary_loss': value_primary_loss_total / float(total_train_steps),
                    'value_scalar_aux_loss': value_scalar_aux_loss_total / float(total_train_steps),
                    'moves_left_loss': moves_left_loss_total / float(total_train_steps),
                    'search_q_loss': search_q_loss_total / float(total_train_steps),
                    'search_q_coverage': (
                        float(search_q_rows_total) / float(max(1, value_rows_total))
                    ),
                    'search_q_mae': (
                        search_q_abs_error_sum_total / float(search_q_rows_total)
                        if search_q_rows_total > 0 else 0.0
                    ),
                    'value_prediction_mean': value_prediction_mean,
                    'value_target_mean': value_target_mean,
                    'value_mean_bias': value_prediction_mean - value_target_mean,
                    'value_root_q_pred_mean': root_q_pred_mean,
                    'value_root_q_target_mean': root_q_target_mean,
                    'value_root_q_bias': (
                        root_q_error_sum_total / float(root_q_rows_total)
                        if root_q_rows_total > 0 else 0.0
                    ),
                    'value_root_q_mae': (
                        root_q_abs_error_sum_total / float(root_q_rows_total)
                        if root_q_rows_total > 0 else 0.0
                    ),
                    'value_root_q_correlation': value_root_q_correlation,
                    'value_error_priority_fraction': (
                        float(value_error_focus_rows_total) / float(value_rows_total)
                        if value_rows_total > 0 else 0.0
                    ),
                    'value_error_priority_weight_share': (
                        value_error_focus_weight_sum_total / value_effective_weight_sum_total
                        if value_effective_weight_sum_total > 0.0 else 0.0
                    ),
                    'value_error_priority_mae': (
                        value_error_focus_abs_error_sum_total / float(value_error_focus_rows_total)
                        if value_error_focus_rows_total > 0 else 0.0
                    ),
                    'value_wdl_brier': (
                        wdl_brier_sum_total / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_wdl_ece': wdl_ece,
                    'value_pred_win_probability': (
                        wdl_pred_sums_total[0] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_pred_draw_probability': (
                        wdl_pred_sums_total[1] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_pred_loss_probability': (
                        wdl_pred_sums_total[2] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_target_win_fraction': (
                        wdl_target_sums_total[0] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_target_draw_fraction': (
                        wdl_target_sums_total[1] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'value_target_loss_fraction': (
                        wdl_target_sums_total[2] / float(wdl_rows_total)
                        if wdl_rows_total > 0 else 0.0
                    ),
                    'grad_total_norm': grad_total_norm_sum / float(total_train_steps),
                    'grad_clip_fraction': float(grad_clipped_steps) / float(total_train_steps),
                    'grad_clip_scale_mean': grad_clip_scale_sum / float(total_train_steps),
                    'grad_backbone_norm': float(gradient_probe_metrics.get('grad_backbone_norm', 0.0) or 0.0),
                    'grad_policy_head_norm': float(gradient_probe_metrics.get('grad_policy_head_norm', 0.0) or 0.0),
                    'grad_value_head_norm': float(gradient_probe_metrics.get('grad_value_head_norm', 0.0) or 0.0),
                    'grad_policy_probe_norm': float(gradient_probe_metrics.get('policy_probe_norm', 0.0) or 0.0),
                    'grad_value_probe_norm': float(gradient_probe_metrics.get('value_probe_norm', 0.0) or 0.0),
                    'grad_policy_value_cosine': float(gradient_probe_metrics.get('policy_value_cosine', 0.0) or 0.0),
                    **iteration_correction_audit_metrics,
                    **phase_wdl_metrics,
                })
                
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
            if recent_sample_age_avg is not None:
                replay_quality_stats['recent_sample_age_avg'] = recent_sample_age_avg
                replay_quality_stats['recent_sample_age_p50'] = recent_sample_age_p50
                replay_quality_stats['recent_sample_age_p90'] = recent_sample_age_p90
                replay_quality_stats['recent_sample_age_new_fraction'] = recent_sample_age_new_fraction
                replay_quality_stats['recent_sample_age_le1_fraction'] = recent_sample_age_le1_fraction
            if champion_sample_age_avg is not None:
                replay_quality_stats['champion_sample_age_avg'] = champion_sample_age_avg
                replay_quality_stats['champion_sample_age_p50'] = champion_sample_age_p50
                replay_quality_stats['champion_sample_age_p90'] = champion_sample_age_p90
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
            anchor_no_mcts_score_lower_bound = None
            anchor_no_mcts_stats = None
            anchor_mcts_no_mcts_gap = None
            estimated_elo = None
            no_mcts_score_rate = None
            no_mcts_stats = None
            no_mcts_games_total = None
            no_mcts_true_win_rate = None
            no_mcts_draw_rate = None
            no_mcts_loss_rate = None
            no_mcts_wins = no_mcts_draws = no_mcts_losses = no_mcts_unresolved = None
            eval_mcts_candidate_metrics = _eval_mcts_move_metrics(None, 'model1')
            eval_mcts_reference_metrics = _eval_mcts_move_metrics(None, 'model2')
            eval_infrastructure_failed = False
            eval_infrastructure_error = None
            promoted_best_this_iter = False
            reanalyse_stats = {
                'selected': 0,
                'updated': 0,
                'corrections': 0,
                'seconds': 0.0,
            }
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
                no_mcts_games_total = int((no_mcts_stats or {}).get('num_games', 0) or 0)
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
                try:
                    eval_runtime, promotion_eval_t0 = _prepare_eval_runtime(
                        eval_subject_model,
                        best_model,
                        config,
                        funnel_runtime_games,
                        promotion_eval_t0,
                    )
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
                        game_index_offset=eval_game_index_offset,
                        use_fixed_openings=bool(rl_cfg.get('eval_fixed_openings_enabled', True)),
                        central_runtime=eval_runtime,
                    )
                except Exception as exc:
                    error_text = str(exc)
                    if not (
                        is_transient_cuda_error(error_text)
                        or "central inference" in error_text.lower()
                    ):
                        raise
                    eval_stats = _empty_eval_stats()
                    eval_stats.update({
                        'infrastructure_error': error_text,
                        'transient_cuda_error': is_transient_cuda_error(error_text),
                        'completed_before_failure': 0,
                    })
                    eval_stage = "infrastructure_error"
                if (
                    eval_stage == "funnel:advanced"
                    and not (eval_stats or {}).get('infrastructure_error')
                ):
                    provisional_score = float((eval_stats or {}).get('score_rate', 0.0) or 0.0)
                    provisional_games = int((eval_stats or {}).get('num_games', 0) or 0)
                    provisional_z = max(
                        0.0,
                        float(rl_cfg.get('promotion_stat_gate_z', 1.28) or 1.28),
                    )
                    provisional_lb = _score_rate_lower_bound(
                        provisional_score,
                        provisional_games,
                        provisional_z,
                        standard_error=(eval_stats or {}).get('paired_score_se'),
                    )
                    required_lb = float(
                        rl_cfg.get('promotion_score_lower_bound_min', 0.50) or 0.50
                    )
                    # Only a point-score winner that narrowly misses confidence
                    # pays for another 128 paired games. Obvious losers retain
                    # the existing cheap funnel behavior.
                    if (
                        provisional_score >= score_rate_threshold
                        and _PROMOTION_CONFIRMATION_LB_FLOOR <= provisional_lb < required_lb
                    ):
                        confirmation_config = _build_eval_config_with_exact_simulations(
                            config,
                            funnel_advanced_simulations,
                        )
                        confirmation_stats = _evaluate_models_with_paired_scores(
                            eval_subject_model,
                            best_model,
                            confirmation_config,
                            device,
                            _PROMOTION_CONFIRMATION_GAMES,
                            game_index_offset=eval_game_index_offset + provisional_games,
                            use_fixed_openings=bool(
                                rl_cfg.get('eval_fixed_openings_enabled', True)
                            ),
                            central_runtime=eval_runtime,
                        )
                        if (confirmation_stats or {}).get('infrastructure_error'):
                            eval_stats = confirmation_stats
                            eval_stage = "infrastructure_error"
                        else:
                            eval_stats = _combine_eval_stats(eval_stats, confirmation_stats)
                            eval_stage = "funnel:confirmation"
                            iteration_notes.append(
                                "promotion confidence extension: "
                                f"{int((eval_stats or {}).get('num_games', 0) or 0)} games"
                            )
                _finish_eval_substage('promotion_eval', promotion_eval_t0)
                eval_infrastructure_error = (eval_stats or {}).get('infrastructure_error')
                if eval_infrastructure_error:
                    eval_infrastructure_failed = True
                    _discard_failed_eval_runtime(eval_infrastructure_error)
                    completed_before_failure = int(
                        (eval_stats or {}).get('completed_before_failure', 0) or 0
                    )
                    iteration_notes.append(
                        "MCTS eval skipped after inference-server failure"
                        + (
                            f" ({completed_before_failure} games completed)"
                            if completed_before_failure > 0
                            else ""
                        )
                    )
                    promotion_status = "not evaluated: transient inference failure"
                score_rate = float((eval_stats or {}).get('score_rate', 0.0))
                true_win_rate = float((eval_stats or {}).get('win_rate', 0.0))
                eval_games_total = int((eval_stats or {}).get('num_games', 0) or 0)
                eval_wins = int((eval_stats or {}).get('wins', 0))
                eval_draws = int((eval_stats or {}).get('draws', 0))
                eval_losses = int((eval_stats or {}).get('losses', 0))
                eval_unresolved = int((eval_stats or {}).get('unresolved', 0))
                eval_mcts_candidate_metrics = _eval_mcts_move_metrics(eval_stats, 'model1')
                eval_mcts_reference_metrics = _eval_mcts_move_metrics(eval_stats, 'model2')
                if eval_mcts_candidate_metrics['changed_rate'] is not None:
                    changed_text = f"{eval_mcts_candidate_metrics['changed_rate']:.1%}"
                    higher_rate = eval_mcts_candidate_metrics['higher_q_when_changed_rate']
                    higher_text = "n/a" if higher_rate is None else f"{higher_rate:.1%}"
                    iteration_notes.append(
                        f"eval MCTS changed candidate move {changed_text}; "
                        f"higher-Q among measured changes {higher_text}"
                    )
                if eval_mcts_candidate_metrics['avg_simulations'] is not None:
                    reduced_rate = eval_mcts_candidate_metrics['reduced_budget_rate']
                    reduced_text = "n/a" if reduced_rate is None else f"{reduced_rate:.0%}"
                    iteration_notes.append(
                        f"eval MCTS avg {eval_mcts_candidate_metrics['avg_simulations']:.1f}/"
                        f"{funnel_advanced_simulations} sims; reduced {reduced_text} of roots"
                    )
                current_mcts_no_mcts_gap = (
                    float(score_rate) - float(no_mcts_score_rate)
                    if no_mcts_score_rate is not None
                    else None
                )
                full_eval_evidence = _promotion_eval_has_full_evidence(eval_stage)
                # Short funnel stages are rejection tests, not equally precise
                # observations. Letting preliminary/intermediate scores update
                # the promotion EMA made a later full decision depend on noise.
                if full_eval_evidence:
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
                is_new_best_candidate = bool(
                    full_eval_evidence
                    and score_rate >= score_rate_threshold
                    and true_win_rate >= true_win_rate_threshold
                )
                promotion_status = (
                    "candidate passed score/win gate"
                    if is_new_best_candidate
                    else (
                        f"hold: incomplete eval ({eval_stage}, {eval_games_total} games)"
                        if not full_eval_evidence
                        else (
                            f"hold: score/win gate (score {score_rate:.1%}/{score_rate_threshold:.1%}, "
                            f"true win {true_win_rate:.1%}/{true_win_rate_threshold:.1%})"
                        )
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
                    standard_error=(eval_stats or {}).get('paired_score_se'),
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
                if is_new_best_candidate:
                    nn_safe, nn_block_reason, nn_score_floor, nn_score_upper = (
                        _promotion_nn_safety_decision(
                            no_mcts_score_rate,
                            no_mcts_games_total,
                            stat_gate_z,
                            rl_cfg,
                        )
                    )
                    # Raw policy matches are discontinuous: one changed move can
                    # flip an entire deterministic game.  Spend more cheap raw
                    # games only after the expensive MCTS gate has passed and
                    # the initial raw result would otherwise reject a candidate.
                    raw_target_games = (
                        _next_raw_candidate_retest_target(
                            nn_safe=nn_safe,
                            games_played=no_mcts_games_total,
                            score_upper_bound=nn_score_upper,
                            score_floor=nn_score_floor,
                        )
                        if should_run_no_mcts_eval
                        else None
                    )
                    while raw_target_games is not None:
                        raw_target_games = int(raw_target_games)
                        raw_extra_games = max(0, raw_target_games - int(no_mcts_games_total or 0))
                        if raw_extra_games <= 0:
                            continue
                        regular_eval_t0 = _start_eval_substage()
                        raw_extra_stats = evaluate_models_no_mcts(
                            model,
                            best_model,
                            config,
                            device,
                            raw_extra_games,
                            game_index_offset=int(no_mcts_games_total or 0),
                            use_fixed_openings=bool(
                                rl_cfg.get('eval_no_mcts_use_fixed_openings', True)
                            ),
                        )
                        no_mcts_stats = _combine_eval_stats(no_mcts_stats, raw_extra_stats)
                        no_mcts_score_rate = float((no_mcts_stats or {}).get('score_rate', 0.0))
                        no_mcts_true_win_rate = float((no_mcts_stats or {}).get('win_rate', 0.0))
                        no_mcts_draw_rate = float((no_mcts_stats or {}).get('draw_rate', 0.0))
                        no_mcts_loss_rate = float((no_mcts_stats or {}).get('loss_rate', 0.0))
                        no_mcts_wins = int((no_mcts_stats or {}).get('wins', 0))
                        no_mcts_draws = int((no_mcts_stats or {}).get('draws', 0))
                        no_mcts_losses = int((no_mcts_stats or {}).get('losses', 0))
                        no_mcts_unresolved = int((no_mcts_stats or {}).get('unresolved', 0))
                        no_mcts_games_total = int((no_mcts_stats or {}).get('num_games', 0) or 0)
                        _finish_eval_substage('regular_eval', regular_eval_t0)
                        nn_safe, nn_block_reason, nn_score_floor, nn_score_upper = (
                            _promotion_nn_safety_decision(
                                no_mcts_score_rate,
                                no_mcts_games_total,
                                stat_gate_z,
                                rl_cfg,
                            )
                        )
                        iteration_notes.append(
                            f"raw candidate retest: {no_mcts_games_total} games -> "
                            f"{no_mcts_score_rate:.1%}"
                        )
                        raw_target_games = _next_raw_candidate_retest_target(
                            nn_safe=nn_safe,
                            games_played=no_mcts_games_total,
                            score_upper_bound=nn_score_upper,
                            score_floor=nn_score_floor,
                        )
                    current_mcts_no_mcts_gap = (
                        float(score_rate) - float(no_mcts_score_rate)
                        if no_mcts_score_rate is not None else None
                    )
                    if not nn_safe:
                        is_new_best_candidate = False
                        promotion_status = nn_block_reason
                anchor_gate_enabled = bool(rl_cfg.get('promotion_require_anchor_non_regression', True))
                min_anchor_score_for_progress = float(rl_cfg.get('promotion_anchor_min_score_rate', 0.50))
                min_anchor_true_win_for_progress = float(
                    rl_cfg.get('promotion_anchor_min_true_win_rate', 0.0)
                )
                min_anchor_score_lower_bound = float(
                    rl_cfg.get('promotion_anchor_score_lower_bound_min', 0.47)
                )
                min_anchor_no_mcts_score = float(
                    rl_cfg.get('promotion_anchor_no_mcts_score_rate_min', 0.50) or 0.50
                )
                min_anchor_no_mcts_lower_bound = float(
                    rl_cfg.get(
                        'promotion_anchor_no_mcts_score_lower_bound_min', 0.47
                    ) or 0.47
                )
                candidate_passed_best_gate = bool(is_new_best_candidate)
                anchor_candidate_due = bool(candidate_passed_best_gate and anchor_gate_enabled)
                anchor_decision_due = bool(anchor_candidate_due)
                reuse_best_result_for_anchor = bool(
                    not eval_infrastructure_failed
                    and
                    anchor_model_available
                    and anchor_model is not None
                    and not best_differs_from_anchor
                )
                if (
                    anchor_model_available
                    and anchor_model is not None
                    and (
                        reuse_best_result_for_anchor
                        or anchor_candidate_due
                    )
                ):
                    if reuse_best_result_for_anchor:
                        anchor_score_rate = score_rate
                        anchor_true_win_rate = true_win_rate
                        anchor_wins = eval_wins
                        anchor_draws = eval_draws
                        anchor_losses = eval_losses
                        anchor_eval_games_total = eval_games_total
                        if no_mcts_score_rate is not None:
                            anchor_no_mcts_stats = no_mcts_stats
                            anchor_no_mcts_games_total = sum(int(value or 0) for value in (
                                no_mcts_wins, no_mcts_draws, no_mcts_losses, no_mcts_unresolved
                            ))
                            anchor_no_mcts_score_rate = no_mcts_score_rate
                        iteration_notes.append("anchor result reused: current best is the IL anchor")
                    else:
                        anchor_eval_t0 = _start_eval_substage()
                        anchor_eval_config = _build_eval_config_with_exact_simulations(
                            config,
                            anchor_eval_simulations,
                        )
                        anchor_games = max(2, int(rl_cfg.get('anchor_eval_games', 40)))
                        anchor_use_fixed_openings = bool(
                            rl_cfg.get('anchor_eval_use_fixed_openings', True)
                        )
                        max_anchor_games = max(anchor_games, _ANCHOR_CANDIDATE_MAX_GAMES)
                        try:
                            anchor_runtime, anchor_eval_t0 = _prepare_eval_runtime(
                                eval_subject_model,
                                anchor_model,
                                anchor_eval_config,
                                max_anchor_games,
                                anchor_eval_t0,
                            )
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
                        except Exception as exc:
                            error_text = str(exc)
                            if not (
                                is_transient_cuda_error(error_text)
                                or "central inference" in error_text.lower()
                            ):
                                raise
                            anchor_stats = _empty_eval_stats()
                            anchor_stats.update({
                                'infrastructure_error': error_text,
                                'transient_cuda_error': is_transient_cuda_error(error_text),
                                'completed_before_failure': 0,
                            })
                        anchor_infrastructure_error = (anchor_stats or {}).get(
                            'infrastructure_error'
                        )
                        if anchor_infrastructure_error:
                            eval_infrastructure_failed = True
                            eval_infrastructure_error = anchor_infrastructure_error
                            _discard_failed_eval_runtime(anchor_infrastructure_error)
                            iteration_notes.append(
                                "anchor MCTS eval skipped after inference-server failure"
                            )
                            promotion_status = "not evaluated: transient inference failure"
                        anchor_score_rate = float((anchor_stats or {}).get('score_rate', 0.0))
                        anchor_true_win_rate = float((anchor_stats or {}).get('win_rate', 0.0))
                        anchor_wins = int((anchor_stats or {}).get('wins', 0))
                        anchor_draws = int((anchor_stats or {}).get('draws', 0))
                        anchor_losses = int((anchor_stats or {}).get('losses', 0))
                        anchor_eval_games_total = int((anchor_stats or {}).get('num_games', 0) or 0)
                        _finish_eval_substage('promotion_eval', anchor_eval_t0)
                    raw_anchor_gate_enabled = bool(
                        rl_cfg.get('promotion_anchor_no_mcts_gate_enabled', True)
                    )
                    raw_anchor_required = bool(anchor_decision_due and raw_anchor_gate_enabled)
                    if raw_anchor_required or should_run_no_mcts_eval:
                        anchor_regular_eval_t0 = _start_eval_substage()
                        anchor_no_mcts_max_games = (
                            max(
                                no_mcts_eval_games,
                                int(
                                    rl_cfg.get(
                                        'anchor_no_mcts_max_games',
                                        _ANCHOR_NO_MCTS_MAX_GAMES,
                                    ) or _ANCHOR_NO_MCTS_MAX_GAMES
                                ),
                            )
                            if raw_anchor_required
                            else no_mcts_eval_games
                        )
                        anchor_no_mcts_stats = _evaluate_anchor_no_mcts_sequential(
                            model=eval_subject_model,
                            anchor_model=anchor_model,
                            config=config,
                            device=device,
                            initial_games=no_mcts_eval_games,
                            max_games=anchor_no_mcts_max_games,
                            min_score_rate=min_anchor_no_mcts_score,
                            min_score_lower_bound=min_anchor_no_mcts_lower_bound,
                            stat_gate_z=stat_gate_z,
                            game_index_offset=0,
                            use_fixed_openings=bool(
                                rl_cfg.get('eval_no_mcts_use_fixed_openings', True)
                            ),
                            initial_stats=anchor_no_mcts_stats,
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
                        standard_error=(
                            (eval_stats or {}).get('paired_score_se')
                            if reuse_best_result_for_anchor
                            else (anchor_stats or {}).get('paired_score_se')
                        ),
                    )
                    if anchor_no_mcts_score_rate is not None:
                        anchor_no_mcts_score_lower_bound = _score_rate_lower_bound(
                            anchor_no_mcts_score_rate,
                            anchor_no_mcts_games_total,
                            stat_gate_z,
                            standard_error=(anchor_no_mcts_stats or {}).get('paired_score_se'),
                        )
                    configured_anchor_games = max(2, int(rl_cfg.get('anchor_eval_games', 40)))
                    if anchor_decision_due and int(anchor_eval_games_total or 0) > configured_anchor_games:
                        iteration_notes.append(
                            f"anchor tiebreak: {anchor_eval_games_total} games -> "
                            f"{anchor_score_rate:.1%} (LB {anchor_score_lower_bound:.1%})"
                        )
                    if (
                        raw_anchor_required
                        and int(anchor_no_mcts_games_total or 0) > no_mcts_eval_games
                    ):
                        iteration_notes.append(
                            f"raw anchor tiebreak: {anchor_no_mcts_games_total} games -> "
                            f"{anchor_no_mcts_score_rate:.1%} "
                            f"(LB {anchor_no_mcts_score_lower_bound:.1%})"
                        )
                if anchor_score_rate is not None:
                    elo_coordinator.observe_il_anchor_eval(
                        iteration + 1,
                        score_rate=anchor_score_rate,
                        true_win_rate=anchor_true_win_rate,
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
                        anchor_safe, anchor_reason = _promotion_anchor_safety_decision(
                            anchor_score_rate,
                            anchor_score_lower_bound,
                            anchor_no_mcts_score_rate,
                            anchor_no_mcts_score_lower_bound,
                            rl_cfg,
                        )
                        anchor_gate_failed = bool(
                            not anchor_safe
                            or float(anchor_true_win_rate or 0.0)
                            < min_anchor_true_win_for_progress
                        )
                        if anchor_gate_failed:
                            is_new_best_candidate = False
                            promotion_status = f"blocked: {anchor_reason}"
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
                        'anchor_no_mcts_score_rate': anchor_no_mcts_score_rate,
                        'anchor_no_mcts_score_lower_bound': anchor_no_mcts_score_lower_bound,
                        'policy_loss': avg_policy,
                        'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                        'value_mae': train_metrics.get('value_mae', 0),
                        'version': model_version,
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                        'rl_run_artifacts': _rl_run_artifact_metadata(logger, base_dir),
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
                estimated_elo = None
                if is_new_best_candidate:
                    elo_eval_t0 = _start_eval_substage()
                    estimated_elo = elo_coordinator.evaluate_promoted_best(iteration + 1)
                    _finish_eval_substage('elo_eval', elo_eval_t0)

                if eval_infrastructure_failed:
                    # An incomplete infrastructure-failed match is not evidence
                    # for model quality. Keep it out of promotion state, EMAs and
                    # plots instead of recording a misleading 0% result.
                    is_new_best_candidate = False
                    score_rate = None
                    true_win_rate = None
                    eval_games_total = None
                    eval_wins = eval_draws = eval_losses = eval_unresolved = None
                    score_lower_bound = None
                    anchor_score_rate = None
                    anchor_score_lower_bound = None
                    anchor_eval_games_total = None
                    anchor_true_win_rate = None
                    anchor_wins = anchor_draws = anchor_losses = None
                    current_mcts_no_mcts_gap = None
                    eval_mcts_candidate_metrics = _eval_mcts_move_metrics(None, 'model1')
                    eval_mcts_reference_metrics = _eval_mcts_move_metrics(None, 'model2')
                    estimated_elo = None
                    promotion_status = "not evaluated: transient inference failure"

                # Log with all metrics
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    learning_rate=current_lr,
                    value_loss_weight=current_value_loss_weight,
                    eval_score_rate_ema=eval_score_rate_ema,
                    eval_true_win_rate_ema=eval_true_win_rate_ema,
                    mcts_no_mcts_gap=current_mcts_no_mcts_gap,
                    eval_mcts_move_samples=eval_mcts_candidate_metrics['samples'],
                    eval_mcts_avg_simulations=eval_mcts_candidate_metrics['avg_simulations'],
                    eval_mcts_reduced_budget_rate=(
                        eval_mcts_candidate_metrics['reduced_budget_rate']
                    ),
                    eval_mcts_changed_rate=eval_mcts_candidate_metrics['changed_rate'],
                    eval_mcts_higher_q_when_changed_rate=(
                        eval_mcts_candidate_metrics['higher_q_when_changed_rate']
                    ),
                    eval_mcts_lower_q_when_changed_rate=(
                        eval_mcts_candidate_metrics['lower_q_when_changed_rate']
                    ),
                    eval_mcts_changed_q_delta_mean=(
                        eval_mcts_candidate_metrics['changed_q_delta_mean']
                    ),
                    eval_reference_mcts_changed_rate=(
                        eval_mcts_reference_metrics['changed_rate']
                    ),
                    eval_reference_mcts_avg_simulations=(
                        eval_mcts_reference_metrics['avg_simulations']
                    ),
                    eval_reference_mcts_reduced_budget_rate=(
                        eval_mcts_reference_metrics['reduced_budget_rate']
                    ),
                    eval_reference_mcts_higher_q_when_changed_rate=(
                        eval_mcts_reference_metrics['higher_q_when_changed_rate']
                    ),
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
                    no_mcts_games=no_mcts_games_total,
                    anchor_score_rate=anchor_score_rate,
                    anchor_true_win_rate=anchor_true_win_rate,
                    anchor_games=anchor_eval_games_total,
                    anchor_score_lower_bound=anchor_score_lower_bound,
                    anchor_no_mcts_games=anchor_no_mcts_games_total,
                    anchor_no_mcts_score_rate=anchor_no_mcts_score_rate,
                    anchor_no_mcts_score_lower_bound=anchor_no_mcts_score_lower_bound,
                    anchor_mcts_no_mcts_gap=anchor_mcts_no_mcts_gap,
                    temperature=current_temp,
                    rl_best_model=is_new_best_candidate,
                    best_iteration=(
                        int(iteration + 1)
                        if is_new_best_candidate else int(best_iteration)
                    ),
                )
                last_logged_iteration = iteration + 1
                if is_new_best_candidate:
                    promoted_best_this_iter = True
                    promotion_status = "PROMOTED: new best model"
                    last_promotion_iteration = int(iteration + 1)
                    best_iteration = int(iteration + 1)
                    best_model.load_state_dict(model.state_dict())
                    best_model.eval()
                    champion_replay_buffer.clear()
                    # The comparison baseline changed; old best-relative EMA is
                    # not commensurate with the new champion.
                    eval_score_rate_ema = 0.50
                    eval_true_win_rate_ema = 0.0
                    best_win_rate_so_far = max(best_win_rate_so_far, float(score_rate))
                    if anchor_model_available:
                        best_differs_from_anchor = True
                    
                    model_to_save = model
                    best_metadata = {
                        'win_rate': true_win_rate,
                        'score_rate': score_rate,
                        'eval_true_win_rate': true_win_rate,
                        'anchor_score_rate': anchor_score_rate,
                        'anchor_true_win_rate': anchor_true_win_rate,
                        'anchor_no_mcts_score_rate': anchor_no_mcts_score_rate,
                        'anchor_no_mcts_score_lower_bound': anchor_no_mcts_score_lower_bound,
                        'policy_loss': avg_policy,
                        'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                        'value_mae': train_metrics.get('value_mae', 0),
                        'version': model_version,
                        'startup_mode': start_mode,
                        'model_architecture': model_architecture,
                        'rl_run_artifacts': _rl_run_artifact_metadata(logger, base_dir),
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
                    best_iteration=int(best_iteration),
                    eval_score_rate_ema=eval_score_rate_ema,
                    eval_true_win_rate_ema=eval_true_win_rate_ema,
                    temperature=current_temp,
                )
                last_logged_iteration = iteration + 1
            reanalyse_interval = max(
                1,
                int(rl_cfg.get('replay_reanalyse_stall_interval', 10) or 10),
            )
            stalled_reanalyse_due = bool(
                should_run_mcts_eval
                and not promoted_best_this_iter
                and int(iteration + 1) >= reanalyse_interval
                and (
                    last_reanalyse_iteration is None
                    or int(iteration + 1) - int(last_reanalyse_iteration) >= reanalyse_interval
                )
            )
            if promoted_best_this_iter or stalled_reanalyse_due:
                reanalyse_t0 = _start_eval_substage()
                reanalyse_stats = _run_selective_reanalyse(
                    replay_buffer,
                    model,
                    config,
                    device,
                )
                _finish_eval_substage('reanalyse', reanalyse_t0)
                last_reanalyse_iteration = int(iteration + 1)
                if int(reanalyse_stats.get('updated', 0)) > 0:
                    reason = "promotion" if promoted_best_this_iter else "stalled learner"
                    iteration_notes.append(
                        f"reanalyse ({reason}): "
                        f"{int(reanalyse_stats['updated'])}/{int(reanalyse_stats['selected'])} hard positions "
                        f"({int(reanalyse_stats.get('corrections', 0))} corrections) "
                        f"@{int(reanalyse_stats['simulations'])} sims in "
                        f"{float(reanalyse_stats.get('seconds', 0.0)):.1f}s"
                    )
            if promoted_best_this_iter:
                learner_selfplay_safe = True
            elif (
                not eval_infrastructure_failed
                and (no_mcts_score_rate is not None or score_rate is not None)
            ):
                selfplay_guard_z = max(
                    0.0,
                    float(rl_cfg.get('promotion_stat_gate_z', 1.28) or 1.28),
                )
                guard_failures = []
                raw_selfplay_safe = True
                if no_mcts_score_rate is not None:
                    raw_selfplay_floor = float(
                        rl_cfg.get('promotion_no_mcts_score_rate_min', 0.40) or 0.40
                    )
                    raw_selfplay_upper = _score_rate_upper_bound(
                        no_mcts_score_rate,
                        no_mcts_games_total,
                        selfplay_guard_z,
                        standard_error=(no_mcts_stats or {}).get('paired_score_se'),
                    )
                    raw_selfplay_safe = bool(raw_selfplay_upper >= raw_selfplay_floor)
                    if not raw_selfplay_safe:
                        guard_failures.append(
                            f"raw UCB {raw_selfplay_upper:.1%} < {raw_selfplay_floor:.1%}"
                        )

                # Raw policy can look strong while its value-guided search is
                # clearly worse than the accepted best (rl42 iteration 2:
                # 60.9% raw, 46.4% MCTS). Such a learner must not generate the
                # complete next replay iteration. Use an optimistic bound so a
                # noisy near-50% result still keeps learner exploration alive.
                mcts_selfplay_safe = True
                if score_rate is not None:
                    eval_score_se = _score_rate_standard_error(
                        score_rate,
                        eval_games_total,
                        (eval_stats or {}).get('paired_score_se'),
                    )
                    mcts_selfplay_upper = _score_rate_upper_bound(
                        score_rate,
                        eval_games_total,
                        selfplay_guard_z,
                        standard_error=eval_score_se,
                    )
                    mcts_selfplay_safe = bool(mcts_selfplay_upper >= 0.50)
                    if not mcts_selfplay_safe:
                        guard_failures.append(
                            f"MCTS UCB {mcts_selfplay_upper:.1%} < 50.0%"
                        )
                    if no_mcts_score_rate is not None:
                        raw_score_se = _score_rate_standard_error(
                            no_mcts_score_rate,
                            no_mcts_games_total,
                            (no_mcts_stats or {}).get('paired_score_se'),
                        )
                        relative_search_gap = (
                            float(score_rate) - float(no_mcts_score_rate)
                        )
                        relative_search_gap_upper = (
                            relative_search_gap
                            + selfplay_guard_z
                            * math.sqrt(eval_score_se ** 2 + raw_score_se ** 2)
                        )
                        relative_search_safe = bool(relative_search_gap_upper >= 0.0)
                        mcts_selfplay_safe = bool(
                            mcts_selfplay_safe and relative_search_safe
                        )
                        if not relative_search_safe:
                            guard_failures.append(
                                "relative MCTS-NN UCB "
                                f"{relative_search_gap_upper:+.1%} < 0.0%"
                            )

                learner_selfplay_safe = bool(raw_selfplay_safe and mcts_selfplay_safe)
                if not learner_selfplay_safe:
                    iteration_notes.append(
                        "learner self-play guard: "
                        f"{', '.join(guard_failures)}; "
                        "next iteration uses best"
                    )
            _finish_stage('eval_log')
            eval_region_total = float(iteration_stage_times.pop('eval_log', 0.0) or 0.0)
            measured_eval_total = 0.0
            for stage_name in ('regular_eval', 'promotion_eval', 'reanalyse', 'elo_eval'):
                stage_seconds = float(iteration_eval_substage_times.get(stage_name, 0.0) or 0.0)
                iteration_stage_times[stage_name] = stage_seconds
                measured_eval_total += stage_seconds
            iteration_stage_times['log'] = max(
                0.0,
                eval_region_total - measured_eval_total - iteration_eval_setup_time,
            )

            latest_metadata = {
                'win_rate': true_win_rate,
                'score_rate': score_rate,
                'eval_true_win_rate': true_win_rate,
                'no_mcts_score_rate': no_mcts_score_rate,
                'no_mcts_win_rate': no_mcts_true_win_rate,
                'no_mcts_true_win_rate': no_mcts_true_win_rate,
                'no_mcts_draw_rate': no_mcts_draw_rate,
                'no_mcts_loss_rate': no_mcts_loss_rate,
                'policy_loss': avg_policy,
                'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                'value_mae': train_metrics.get('value_mae', 0),
                'version': model_version,
                'startup_mode': start_mode,
                'model_architecture': model_architecture,
                'rl_run_artifacts': _rl_run_artifact_metadata(logger, base_dir),
                'rl_surpassed_il_anchor': bool(elo_coordinator.il_anchor_surpassed),
                'rl_best_il_anchor_score_rate': elo_coordinator.best_il_anchor_score_rate,
                'rl_best_il_anchor_true_win_rate': elo_coordinator.best_il_anchor_true_win_rate,
                'rl_best_il_anchor_iteration': elo_coordinator.best_il_anchor_iteration,
            }
            latest_metadata.update(elo_coordinator.metadata_for_iteration(iteration + 1))
            runtime_snapshot_dtype = (
                torch.bfloat16 if use_bfloat16 else (torch.float16 if use_amp else None)
            )
            best_runtime_state = _snapshot_model_state_cpu(
                best_model,
                dtype=runtime_snapshot_dtype,
            )
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
                    'best_model_state_dict': best_runtime_state,
                    'best_iteration': best_iteration,
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
            replay_quality_stats['train_policy_correction_sharpened_fraction'] = (
                train_metrics.get('policy_correction_sharpened_fraction', 0.0)
            )
            replay_quality_stats['train_policy_correction_target_top1_before'] = (
                train_metrics.get('policy_correction_target_top1_before', 0.0)
            )
            replay_quality_stats['train_policy_correction_target_top1_after'] = (
                train_metrics.get('policy_correction_target_top1_after', 0.0)
            )
            replay_quality_stats.update({
                key: train_metrics.get(key, value)
                for key, value in iteration_correction_audit_metrics.items()
            })
            replay_quality_stats['reanalyse_selected'] = int(
                reanalyse_stats.get('selected', 0) or 0
            )
            replay_quality_stats['reanalyse_updated'] = int(
                reanalyse_stats.get('updated', 0) or 0
            )
            replay_quality_stats['reanalyse_correction_fraction'] = (
                float(reanalyse_stats.get('corrections', 0) or 0)
                / float(max(1, int(reanalyse_stats.get('updated', 0) or 0)))
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
            budget_max = float((selfplay_stats or {}).get('search_simulations_budget_max', 0.0) or 0.0)
            budget_spread = (
                f"p10/50/90/max {budget_p10:.0f}/{budget_p50:.0f}/{budget_p90:.0f}/{budget_max:.0f}"
            )
            difficulty_budget_corr = float(
                (selfplay_stats or {}).get('search_difficulty_budget_correlation', 0.0) or 0.0
            )
            tree_reuse_hit_rate = float(
                (selfplay_stats or {}).get('tree_reuse_hit_rate', 0.0) or 0.0
            )
            tree_inherited_visits = float(
                (selfplay_stats or {}).get('tree_inherited_visits_avg', 0.0) or 0.0
            )
            tree_reuse_quality = float(
                (selfplay_stats or {}).get('tree_reuse_quality_avg', 0.0) or 0.0
            )
            tree_candidate_coverage = float(
                (selfplay_stats or {}).get('tree_reuse_candidate_coverage_avg', 0.0) or 0.0
            )
            tree_fresh_floor = float(
                (selfplay_stats or {}).get('tree_reuse_fresh_floor_avg', 0.0) or 0.0
            )
            tree_scout_stability = float(
                (selfplay_stats or {}).get('tree_reuse_scout_stability_avg', 0.0) or 0.0
            )
            tree_scout_reduction = float(
                (selfplay_stats or {}).get('tree_reuse_scout_reduction_rate', 0.0) or 0.0
            )
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
                    f"difficulty corr {difficulty_budget_corr:.2f} | "
                    f"tree reuse {tree_reuse_hit_rate:.0%}, inherited {tree_inherited_visits:.1f}, "
                    f"quality/coverage {tree_reuse_quality:.0%}/{tree_candidate_coverage:.0%}, "
                    f"scout stability/cut {tree_scout_stability:.0%}/{tree_scout_reduction:.0%}, "
                    f"fresh floor {tree_fresh_floor:.0f} | "
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
            replay_passes = float(replay_quality_stats.get('train_replay_passes', 0.0) or 0.0)
            champion_selected_fraction = float(
                replay_quality_stats.get('champion_replay_selected_fraction', 0.0) or 0.0
            )
            def _audit_metric(name, default=0.0):
                try:
                    value = float(train_metrics.get(name, default))
                except (TypeError, ValueError):
                    return float(default)
                return value if math.isfinite(value) else float(default)

            train_summary = (
                f"{total_train_steps} x {base_batch_size:,} | replay unique {replay_coverage:.0%}, "
                f"recent {replay_passes:.2f}x, "
                f"champion {champion_selected_fraction:.0%} | "
                f"loss {avg_loss:.3f} (P {avg_policy:.3f}, V {avg_value:.3f}) | "
                f"policy top1/top3 {float(train_metrics.get('policy_top1_acc', 0.0) or 0.0):.1%}/"
                f"{float(train_metrics.get('policy_top3_acc', 0.0) or 0.0):.1%} | "
                f"batch hard-top1 {float(train_metrics.get('policy_correction_top1_acc', 0.0) or 0.0):.1%} "
                f"({float(train_metrics.get('policy_correction_fraction', 0.0) or 0.0):.0%} rows), "
                f"rank debt {float(train_metrics.get('policy_correction_rank_loss', 0.0) or 0.0):.3f} "
                f"x{float(train_metrics.get('policy_correction_rank_weight', 0.0) or 0.0):.3f}, "
                f"sharpen {float(train_metrics.get('policy_correction_sharpened_fraction', 0.0) or 0.0):.0%} "
                f"({float(train_metrics.get('policy_correction_target_top1_before', 0.0) or 0.0):.0%}->"
                f"{float(train_metrics.get('policy_correction_target_top1_after', 0.0) or 0.0):.0%}) | "
                f"audit {int(train_metrics.get('correction_audit_rows', 0) or 0):,}: "
                f"{_audit_metric('correction_audit_top1_before'):.1%}->"
                f"{_audit_metric('correction_audit_top1_after'):.1%}, "
                f"rank +{_audit_metric('correction_audit_rank_gain'):.2f}, "
                f"retain {_audit_metric('correction_audit_retention'):.1%} | "
                f"LR {current_lr:.2e}"
            )
            value_std_ratio = (
                avg_value_pred_std / avg_target_value_std
                if avg_target_value_std > 1e-8
                else 0.0
            )
            age_summary = "-"
            if recent_sample_age_p50 is not None and recent_sample_age_p90 is not None:
                age_summary = f"recent {recent_sample_age_p50:.1f}/{recent_sample_age_p90:.1f}"
                if champion_sample_age_p90 is not None:
                    age_summary += f", champion p90 {champion_sample_age_p90:.1f} iter"
            value_summary = (
                f"MAE {float(train_metrics.get('value_mae', 0.0) or 0.0):.3f} | "
                f"WDL acc {float(train_metrics.get('value_wdl_acc', 0.0) or 0.0):.1%} | "
                f"draw P/target {float(train_metrics.get('value_pred_draw_probability', 0.0) or 0.0):.1%}/"
                f"{float(train_metrics.get('value_target_draw_fraction', 0.0) or 0.0):.1%} | "
                f"Brier/ECE {float(train_metrics.get('value_wdl_brier', 0.0) or 0.0):.3f}/"
                f"{float(train_metrics.get('value_wdl_ece', 0.0) or 0.0):.3f} | "
                f"error focus {float(train_metrics.get('value_error_priority_fraction', 0.0) or 0.0):.0%} rows/"
                f"{float(train_metrics.get('value_error_priority_weight_share', 0.0) or 0.0):.0%} weight | "
                f"pred/target std {value_std_ratio:.2f} | replay age {age_summary}"
            )
            if eval_infrastructure_failed:
                eval_summary = (
                    "MCTS skipped after transient inference failure; "
                    "training continued with a fresh server scheduled next iteration"
                )
            elif score_rate is not None:
                eval_parts = [
                    f"MCTS {score_rate:.1%} (LB {float(score_lower_bound or 0.0):.1%}, "
                    f"W/D/L {eval_wins}/{eval_draws}/{eval_losses}, {eval_stage})"
                ]
                if no_mcts_score_rate is not None:
                    eval_parts.append(
                        f"raw match {no_mcts_score_rate:.1%}; relative search shift "
                        f"{float(score_rate - no_mcts_score_rate):+.1%}"
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
    except RLTrainingInterrupted as exc:
        training_interrupted = True
        interrupted_stage = str(exc) or "self-play"
    except KeyboardInterrupt:
        training_interrupted = True
        interrupted_stage = "runtime"
    finally:
        close_eval_central_runtime(persistent_eval_runtime)
        _shutdown_selfplay_pool()

    logger.plot()
    logger.plot_rl_performance()
    logger.plot_rl_data_quality()
    if training_interrupted:
        _handle_graceful_interrupt(logger=logger, stage=interrupted_stage)
        return

    # `0` is a real best iteration: it means the immutable IL initialization
    # was never replaced. Never relabel that checkpoint as the final loop index.
    final_best_iteration = int(best_iteration)
    if final_best_iteration >= 0 and last_logged_iteration is not None:
        final_elo_enabled = bool(
            elo_coordinator.enabled
            and elo_coordinator.rl_elo_config.get("final_enabled", True)
        )
        final_elo_t0 = time.perf_counter() if final_elo_enabled else None
        elo_coordinator.evaluate_final_best(final_best_iteration, model_override=best_model)
        if final_elo_t0 is not None:
            logger.add_final_elo_runtime(
                int(last_logged_iteration or final_best_iteration),
                time.perf_counter() - final_elo_t0,
            )
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


