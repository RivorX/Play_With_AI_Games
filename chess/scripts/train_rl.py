"""
Reinforcement Learning Training Script with Comprehensive Metrics

NEW Features:
- 📊 Policy Accuracy tracking during training
- 📊 Value MAE monitoring
- 📊 Enhanced self-play statistics
"""

import os
import sys
import signal
import copy
import shutil
import contextlib
import multiprocessing as _stdlib_mp

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

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint, normalize_state_dict_keys

# Import MCTS self-play
try:
    from src.batch_selfplay import play_games_mcts_worker, persistent_selfplay_worker
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("⚠️ MCTS self-play not available")

# Import from utils
from utils.shared.logger import TrainingLogger
from utils.shared.elo_estimator import estimate_model_elo
from utils.rl.replay import ReplayBuffer, PrioritizedReplayBuffer
from utils.rl.temperature import TemperatureSchedule
from utils.rl.training_rl import train_on_batch_rl, evaluate_models
from utils.rl.startup import plan_rl_startup, apply_rl_startup_plan
from utils.il.auto_tune import resolve_rl_hyperparameters
from utils.shared.metrics import MetricsCalculator
from utils.shared.runtime_helpers import (
    build_model_file_tag,
    build_model_architecture_metadata,
    cleanup_interrupted_log_csv,
)
from utils.shared.model_view import print_active_model_summary


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None
_WORKER_INTERRUPT_EXIT_CODE = 130
_SELFPLAY_CONFIG_PRINTED = False
_SELFPLAY_POOL = None


def _print_console_block(unicode_lines, ascii_lines=None):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        for line in unicode_lines:
            line.encode(encoding)
        lines = unicode_lines
    except Exception:
        lines = ascii_lines if ascii_lines is not None else unicode_lines

    for line in lines:
        print(line)


def _resolve_elo_worker_device(main_device, configured_value):
    value = str(configured_value).strip().lower()
    if value in {"same", ""}:
        resolved = main_device
    elif value == "cuda":
        resolved = torch.device("cuda")
    else:
        resolved = torch.device("cpu")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _snapshot_model_state_cpu(model):
    normalized_state = normalize_state_dict_keys(model.state_dict())
    return {
        key: tensor.detach().to(device="cpu", copy=True)
        for key, tensor in normalized_state.items()
    }


def _rl_elo_worker(
    iteration_num,
    model_state_cpu,
    config_snapshot,
    elo_config_snapshot,
    worker_device_str,
    result_queue,
    cancel_event,
):
    try:
        if cancel_event is not None and cancel_event.is_set():
            result_queue.put({"iteration": int(iteration_num), "cancelled": True})
            return

        worker_config = copy.deepcopy(config_snapshot)
        worker_config.setdefault("model", {})
        worker_config["model"]["print_summary"] = False

        worker_device = torch.device(worker_device_str)
        if worker_device.type == "cuda" and not torch.cuda.is_available():
            worker_device = torch.device("cpu")

        worker_model = ChessNet(worker_config).to(worker_device)
        worker_model = worker_model.to(memory_format=torch.channels_last)
        worker_model.load_state_dict(model_state_cpu)
        worker_model.eval()

        elo_result = estimate_model_elo(
            worker_model,
            worker_config,
            worker_device,
            elo_config_snapshot,
            stop_event=cancel_event,
        )
        result_queue.put(
            {
                "iteration": int(iteration_num),
                "result": elo_result,
                "device": worker_device.type,
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"iteration": int(iteration_num), "cancelled": True})
    except Exception as exc:
        result_queue.put({"iteration": int(iteration_num), "error": str(exc)})


class RLEloCoordinator:
    def __init__(self, model, config, device, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.elo_config = dict(config.get("elo_estimation", {}) or {})
        self.enabled = bool(self.elo_config.get("enabled", False))
        self.every_n_evals = max(1, int(self.elo_config.get("rl_every_n_evals", 3)))
        self.final_on_shutdown = bool(self.elo_config.get("final_on_rl_shutdown", True))
        self.min_win_rate_for_stockfish = float(self.elo_config.get("rl_min_win_rate_for_stockfish", 0.50))
        self.last_eval_win_rate = None
        # RL evaluation is synchronous by design (training is paused while Elo runs).
        # Prefer CUDA for RL Elo/MCTS evaluation when available.
        rl_eval_device_raw = str(self.elo_config.get("rl_device", "cuda")).strip().lower()
        if rl_eval_device_raw in {"same", ""}:
            self.eval_device = device
        elif rl_eval_device_raw == "cpu":
            self.eval_device = torch.device("cpu")
        else:
            self.eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.eval_device.type == "cuda" and not torch.cuda.is_available():
            self.eval_device = torch.device("cpu")
        self.eval_counter = 0
        self.last_elo_iteration = None
        self.interrupted_during_elo = False

    def print_startup_summary(self):
        if not self.enabled:
            return
        final_use_mcts = self.elo_config.get("final_rl_use_mcts", self.elo_config.get("use_mcts", False))
        final_sims = self.elo_config.get("final_rl_mcts_simulations", self.elo_config.get("mcts_simulations", 100))
        print(
            f"Elo eval (RL): every {self.every_n_evals} eval(s) vs Stockfish, "
            f"sync_device={self.eval_device.type}, final_on_shutdown={self.final_on_shutdown}, "
            f"mcts={self.elo_config.get('use_mcts', False)}, "
            f"min_win_rate={self.min_win_rate_for_stockfish:.0%}"
        )
        if self.final_on_shutdown:
            print(
                f"Elo final (RL): use_mcts={bool(final_use_mcts)}, "
                f"mcts_simulations={int(final_sims)}"
            )

    def maybe_evaluate(self, iteration_num, win_rate=None):
        if not self.enabled:
            return None
        self.last_eval_win_rate = win_rate
        self.eval_counter += 1
        if (self.eval_counter % self.every_n_evals) != 0:
            return None
        if win_rate is None:
            print("Skipping Stockfish Elo: missing RL win rate vs best model.")
            return None
        if float(win_rate) < self.min_win_rate_for_stockfish:
            print(
                "Skipping Stockfish Elo: "
                f"win rate {float(win_rate):.2%} < required {self.min_win_rate_for_stockfish:.0%}."
            )
            return None
        return self._run_estimate(iteration_num, reason_label=f"iteration {iteration_num}")

    def final_evaluate(self, iteration_num, interrupted=False):
        if not self.enabled or not self.final_on_shutdown:
            return None
        if interrupted and self.interrupted_during_elo:
            print("Skipping final Elo: shutdown was triggered during a Stockfish evaluation.")
            return None
        if self.last_elo_iteration == int(iteration_num):
            final_use_mcts = bool(self.elo_config.get("final_rl_use_mcts", self.elo_config.get("use_mcts", False)))
            regular_use_mcts = bool(self.elo_config.get("use_mcts", False))
            final_sims = int(self.elo_config.get("final_rl_mcts_simulations", self.elo_config.get("mcts_simulations", 100)))
            regular_sims = int(self.elo_config.get("mcts_simulations", 100))
            if final_use_mcts == regular_use_mcts and final_sims == regular_sims:
                print(f"Final Elo skipped: iteration {iteration_num} was already evaluated vs Stockfish.")
                return self.logger.get_latest_estimated_elo()
        return self._run_estimate(
            iteration_num,
            reason_label="shutdown" if interrupted else "final",
            final_override=True,
        )

    def _run_estimate(self, iteration_num, reason_label, final_override=False):
        elo_config = dict(self.elo_config)
        elo_config.setdefault("stockfish_priority", "below_normal")
        elo_config.setdefault("stockfish_hide_window", True)
        elo_config.setdefault("max_error_logs_per_type", 8)
        if final_override:
            # Final/shutdown Elo runs when training is paused/stopped;
            # allow full CPU budget for faster estimation.
            elo_config["prioritize_training"] = False
            elo_config["reserve_dataloader_workers"] = False
            elo_config["stockfish_priority"] = "normal"
            if "final_rl_use_mcts" in self.elo_config:
                elo_config["use_mcts"] = bool(self.elo_config.get("final_rl_use_mcts"))
            if "final_rl_mcts_simulations" in self.elo_config:
                elo_config["mcts_simulations"] = int(self.elo_config.get("final_rl_mcts_simulations"))

        print(f"\nEstimating Elo vs Stockfish ({reason_label})...")
        allow_shutdown_retry = bool(final_override and reason_label == "shutdown")
        max_attempts = 2 if allow_shutdown_retry else 1
        elo_result = None

        for attempt_idx in range(max_attempts):
            elo_cancel_event = threading.Event()
            try:
                elo_result = estimate_model_elo(
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
                print("Ctrl+C detected during Stockfish evaluation. Cancelling evaluation...")
                return None
            except Exception as exc:
                print(f"Elo estimation failed: {exc}")
                return None

            elo_result = elo_result or {}
            if (
                elo_result.get("cancelled")
                and allow_shutdown_retry
                and attempt_idx == 0
                and int(elo_result.get("total_games", 0) or 0) == 0
            ):
                # The first Ctrl+C was used to stop RL; require another Ctrl+C
                # after shutdown Elo starts to cancel this estimation.
                print("First Ctrl+C was consumed by RL shutdown. Press Ctrl+C again to cancel Stockfish Elo.")
                continue
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
            self.logger.record_estimated_elo(iteration_num, estimated_elo, update_csv=True)
            print(f"Estimated Elo: {estimated_elo}")
            for lvl, res in sorted(elo_result.get("results", {}).items()):
                score_str = f"W{res['wins']}/D{res['draws']}/L{res['losses']}"
                print(f"  vs SF {lvl}: {score_str} (score: {res['score']:.0%})")
            print(f"  time {elo_result.get('total_time', 0.0):.1f}s ({elo_result.get('total_games', 0)} games)")
        elif not elo_result.get("skipped"):
            print("Elo estimation: inconclusive")

        return estimated_elo


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


# ==============================================================================
# SELF-PLAY WITH PROPER MCTS
# ==============================================================================

class _PersistentSelfPlayPool:
    def __init__(self, config, worker_specs, device_type, temp_dir):
        self.config = config
        self.worker_specs = list(worker_specs)
        self.device_type = device_type
        self.temp_dir = Path(temp_dir)
        self.mp_ctx = mp.get_context('spawn')
        self.result_queue = self.mp_ctx.Queue()
        self.task_queues = {}
        self.processes = {}
        self.started = False

    def matches(self, worker_specs, device_type, temp_dir):
        return (
            self.worker_specs == list(worker_specs)
            and self.device_type == device_type
            and self.temp_dir == Path(temp_dir)
        )

    def start(self):
        if self.started:
            return

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for rank, _ in self.worker_specs:
            device_id = rank % gpu_count if self.device_type == 'cuda' and gpu_count > 0 else 'cpu'
            task_queue = self.mp_ctx.Queue()
            process = self.mp_ctx.Process(
                target=persistent_selfplay_worker,
                args=(rank, self.config, device_id, task_queue, self.result_queue),
            )
            process.daemon = True
            process.start()
            self.task_queues[rank] = task_queue
            self.processes[rank] = process

        self.started = True

    def submit(
        self,
        task_id,
        model_state_path,
        temperature,
        worker_model_state_paths=None,
        worker_opponent_state_paths=None,
    ):
        result_files = []
        progress_files = []
        worker_model_state_paths = worker_model_state_paths or {}
        worker_opponent_state_paths = worker_opponent_state_paths or {}

        for rank, games_for_worker in self.worker_specs:
            result_file = self.temp_dir / f"worker_{rank}_{task_id}.pkl"
            progress_file = self.temp_dir / f"worker_{rank}_{task_id}.progress"
            try:
                result_file.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                progress_file.unlink(missing_ok=True)
            except Exception:
                pass

            self.task_queues[rank].put({
                'cmd': 'play',
                'task_id': task_id,
                'model_state_path': str(worker_model_state_paths.get(rank, model_state_path)),
                'opponent_model_state_path': (
                    str(worker_opponent_state_paths[rank])
                    if rank in worker_opponent_state_paths
                    else None
                ),
                'num_games': int(games_for_worker),
                'result_file_path': str(result_file),
                'mcts_temperature': temperature,
            })
            result_files.append(result_file)
            progress_files.append(progress_file)

        return result_files, progress_files

    def shutdown(self, timeout_s=5):
        if not self.started:
            return

        for task_queue in self.task_queues.values():
            try:
                task_queue.put({'cmd': 'stop'})
            except Exception:
                pass

        _terminate_workers(list(self.processes.values()), timeout_s=timeout_s)

        for task_queue in self.task_queues.values():
            try:
                task_queue.close()
            except Exception:
                pass
        try:
            self.result_queue.close()
        except Exception:
            pass

        self.task_queues.clear()
        self.processes.clear()
        self.started = False


def _shutdown_selfplay_pool(timeout_s=5):
    global _SELFPLAY_POOL
    if _SELFPLAY_POOL is None:
        return
    try:
        _SELFPLAY_POOL.shutdown(timeout_s=timeout_s)
    finally:
        _SELFPLAY_POOL = None

def _terminate_workers(processes, timeout_s=5):
    """Terminate spawned self-play workers cleanly."""
    all_processes = []
    seen = set()
    for proc in list(processes) + list(mp.active_children()):
        if proc is None:
            continue
        key = proc.pid if proc.pid is not None else id(proc)
        if key in seen:
            continue
        seen.add(key)
        all_processes.append(proc)

    for proc in all_processes:
        if proc.is_alive():
            proc.terminate()

    deadline = time.time() + timeout_s
    for proc in all_processes:
        remaining = max(0.0, deadline - time.time())
        proc.join(timeout=remaining)

    for proc in all_processes:
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=1.0)


def _is_interrupt_exit_code(exit_code):
    if exit_code is None:
        return False
    if exit_code == _WORKER_INTERRUPT_EXIT_CODE:
        return True
    sigint = getattr(signal, "SIGINT", None)
    return sigint is not None and exit_code == -int(sigint)


def play_games_parallel_mcts(model, config, device, num_games, replay_buffer=None):
    """
    Parallel self-play using MCTS
    
    This is the PROPER AlphaZero approach:
    - Each worker plays games using MCTS
    - Training targets = MCTS visit distributions
    - High quality training data
    """
    start_time = time.time()
    
    if not MCTS_SELFPLAY_AVAILABLE:
        print("❌ MCTS self-play not available!")
        return [], 0, 0, 0, 0, 0
    
    model_state = model.state_dict()
    
    rl_cfg = config.get('reinforcement_learning', {})
    use_persistent_pool = bool(rl_cfg.get('persistent_self_play_workers', True))
    stream_to_replay = replay_buffer is not None and bool(
        rl_cfg.get('self_play_stream_to_replay', True)
    )
    
    # Parallel configuration
    num_workers = rl_cfg.get('self_play_workers', 4)
    if isinstance(num_workers, str):
        if num_workers.strip().lower() == 'auto':
            num_workers = max(1, mp.cpu_count() - 1)
        else:
            try:
                num_workers = int(num_workers)
            except ValueError:
                num_workers = 4
    if isinstance(num_workers, (int, float)) and num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)

    self_play_threads_raw = rl_cfg.get('self_play_torch_threads', 1)
    try:
        self_play_threads = max(1, int(self_play_threads_raw))
    except Exception:
        self_play_threads = 1
    max_cpu_sane_workers = max(1, mp.cpu_count() // self_play_threads)
    
    # Self-play device selection
    self_play_device = rl_cfg.get('self_play_device', 'auto')
    if isinstance(self_play_device, str):
        self_play_device = self_play_device.strip().lower()
    else:
        self_play_device = 'auto'
    
    if self_play_device == 'cuda':
        if torch.cuda.is_available():
            device_type = 'cuda'
        else:
            print("⚠️ self_play_device=cuda but no GPU available. Falling back to CPU.")
            device_type = 'cpu'
    elif self_play_device == 'cpu':
        device_type = 'cpu'
    else:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device_type == 'cpu':
        num_workers = min(int(num_workers), max_cpu_sane_workers)
    else:
        num_workers = min(int(num_workers), max_cpu_sane_workers)
        # On GPU, cap workers by available GPUs and configurable workers-per-GPU.
        # MCTS has heavy CPU-side tree logic, so >1 worker per GPU can improve
        # utilization by overlapping tree expansion with batched inference.
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 0:
            workers_per_gpu_raw = rl_cfg.get('self_play_workers_per_gpu', 1)
            try:
                workers_per_gpu = max(1, int(workers_per_gpu_raw))
            except Exception:
                workers_per_gpu = 1
            num_workers = min(num_workers, num_gpus * workers_per_gpu)
    
    num_workers = max(1, num_workers)

    # Cap workers to number of games (no point spawning more workers than games)
    num_workers = min(num_workers, num_games)

    use_batch_selfplay = bool(rl_cfg.get('use_batch_selfplay', False))
    max_batch_games_raw = rl_cfg.get('max_batch_games_per_worker', 1)
    try:
        max_batch_games = max(1, int(max_batch_games_raw))
    except Exception:
        max_batch_games = 1
    
    # Distribute games across workers (handle remainder)
    base_games = num_games // num_workers
    remainder = num_games % num_workers
    games_per_worker = [base_games + (1 if i < remainder else 0) for i in range(num_workers)]
    worker_specs = [(rank, games_per_worker[rank]) for rank in range(num_workers) if games_per_worker[rank] > 0]

    # Shared temp dirs used by worker result files and league runtime snapshots.
    temp_dir = Path(tempfile.gettempdir()) / "chess_selfplay_mcts"
    temp_dir.mkdir(exist_ok=True)
    league_runtime_dir = temp_dir / "league_pool_runtime"
    league_runtime_dir.mkdir(exist_ok=True)

    # League-style snapshot mixing: a fraction of workers uses older checkpoints
    # to generate data against weaker/stale policies (breaks draw-heavy local optima).
    league_cfg = rl_cfg
    league_enabled = bool(league_cfg.get('league_selfplay_enabled', False))
    league_worker_fraction = float(league_cfg.get('league_worker_fraction', 0.30))
    league_worker_fraction = max(0.0, min(1.0, league_worker_fraction))
    league_max_checkpoints = max(1, int(league_cfg.get('league_max_checkpoints', 8)))
    league_min_pool_for_full_fraction = max(
        2,
        int(league_cfg.get('league_min_pool_for_full_fraction', 10)),
    )
    league_runtime_snapshots_enabled = bool(
        league_cfg.get('league_runtime_snapshots_enabled', True)
    )
    league_runtime_snapshot_keep = max(
        2,
        int(league_cfg.get('league_runtime_snapshot_keep', 12)),
    )
    league_opponent_age_bias = float(league_cfg.get('league_opponent_age_bias', 1.25))
    league_recent_opponent_prob = float(league_cfg.get('league_recent_opponent_prob', 0.35))
    league_recent_opponent_prob = max(0.0, min(1.0, league_recent_opponent_prob))

    def _build_time_diverse_pool(candidates, keep_count):
        if not candidates:
            return []
        keep_count = max(1, int(keep_count))
        if len(candidates) <= keep_count:
            return list(candidates)
        idxs = sorted({
            int(round(v))
            for v in np.linspace(0, len(candidates) - 1, num=keep_count)
        })
        if len(idxs) < keep_count:
            for idx in range(len(candidates)):
                if idx in idxs:
                    continue
                idxs.append(idx)
                if len(idxs) >= keep_count:
                    break
            idxs = sorted(idxs)
        return [candidates[idx] for idx in idxs[:keep_count]]

    def _sample_league_opponent(candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        recent_bucket = max(1, len(candidates) // 3)
        if np.random.random() < league_recent_opponent_prob:
            recent_idx = int(np.random.randint(0, recent_bucket))
            return candidates[recent_idx]

        ranks = np.arange(len(candidates), dtype=np.float64)
        age_bias = max(0.0, float(league_opponent_age_bias))
        weights = (ranks + 1.0) ** age_bias
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0:
            return candidates[int(np.random.randint(0, len(candidates)))]
        probs = weights / weights_sum
        sampled_idx = int(np.random.choice(np.arange(len(candidates)), p=probs))
        return candidates[sampled_idx]

    worker_opponent_state_paths = {}
    league_candidates = []
    if league_enabled:
        try:
            # Runtime snapshot pool captured from this RL line (init + iter snapshots).
            if league_runtime_snapshots_enabled and league_runtime_dir.exists():
                runtime_snapshots = sorted(
                    league_runtime_dir.glob('*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                league_candidates.extend(runtime_snapshots)
        except Exception:
            league_candidates = []

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for p in league_candidates:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
        league_candidates = _build_time_diverse_pool(deduped, league_max_checkpoints)

        # Keep league pressure modest when the snapshot pool is still tiny.
        if league_candidates:
            pool_scale = min(1.0, float(len(league_candidates)) / float(league_min_pool_for_full_fraction))
            league_worker_fraction = max(0.0, min(1.0, league_worker_fraction * pool_scale))

        if league_candidates and worker_specs and league_worker_fraction > 0.0:
            league_worker_count = int(round(len(worker_specs) * league_worker_fraction))
            league_worker_count = max(1, min(len(worker_specs), league_worker_count))
            rank_choices = np.random.choice(
                [rank for rank, _ in worker_specs],
                size=league_worker_count,
                replace=False,
            )
            for rank in rank_choices:
                sampled = _sample_league_opponent(league_candidates)
                if sampled is not None:
                    worker_opponent_state_paths[int(rank)] = sampled
    
    global _SELFPLAY_CONFIG_PRINTED
    if not _SELFPLAY_CONFIG_PRINTED:
        _SELFPLAY_CONFIG_PRINTED = True
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        wpg = rl_cfg.get('self_play_workers_per_gpu', 1) if device_type == 'cuda' else '-'
        threads = rl_cfg.get('self_play_torch_threads', 1)
        batch_info = f"on  (max {max_batch_games} gier/worker)" if use_batch_selfplay else "off"
        max_parallel_games = sum(
            min(max_batch_games if use_batch_selfplay else 1, games)
            for _, games in worker_specs
        )
        w_games = f"{base_games}" + (f"  (+1 dla {remainder} workerów)" if remainder else "")
        league_status = (
            f"on ({len(worker_opponent_state_paths)}/{len(worker_specs)} workerów, pool={len(league_candidates)})"
            if league_enabled and league_candidates
            else ("on (brak checkpointów w puli)" if league_enabled else "off")
        )
        print()
        unicode_lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║           KONFIGURACJA SELF-PLAY (stała)             ║",
            "╠══════════════════════════╦═══════════════════════════╣",
            f"║  Urządzenie              ║  {device_type:<25} ║",
            f"║  GPU dostępne            ║  {gpus:<25} ║",
            f"║  Workerów per GPU        ║  {str(wpg):<25} ║",
            f"║  Workerów łącznie        ║  {len(worker_specs):<25} ║",
            f"║  Wątków per worker       ║  {str(threads):<25} ║",
            f"║  Gier per iterację       ║  {num_games:<25} ║",
            f"║  Gier per worker         ║  {w_games:<25} ║",
            f"║  Batch self-play         ║  {batch_info:<25} ║",
            f"║  Gier równolegle max     ║  {max_parallel_games:<25} ║",
            f"║  Symulacje MCTS          ║  {rl_cfg['mcts_simulations']:<25} ║",
            f"║  Rozmiar batcha MCTS     ║  {rl_cfg.get('mcts_batch_size', 32):<25} ║",
            f"║  Reuse drzewa            ║  {str(rl_cfg.get('mcts_reuse_tree', True)):<25} ║",
            f"║  League opponent pool    ║  {league_status:<25} ║",
            "╚══════════════════════════╩═══════════════════════════╝",
        ]
        ascii_lines = [
            "+----------------------------------------------------+",
            "|           KONFIGURACJA SELF-PLAY (stala)          |",
            "+--------------------------+-------------------------+",
            f"|  Urzadzenie              |  {device_type:<25} |",
            f"|  GPU dostepne            |  {gpus:<25} |",
            f"|  Workerow per GPU        |  {str(wpg):<25} |",
            f"|  Workerow lacznie        |  {len(worker_specs):<25} |",
            f"|  Watkow per worker       |  {str(threads):<25} |",
            f"|  Gier per iteracje       |  {num_games:<25} |",
            f"|  Gier per worker         |  {w_games:<25} |",
            f"|  Batch self-play         |  {batch_info:<25} |",
            f"|  Gier rownolegle max     |  {max_parallel_games:<25} |",
            f"|  Symulacje MCTS          |  {rl_cfg['mcts_simulations']:<25} |",
            f"|  Rozmiar batcha MCTS     |  {rl_cfg.get('mcts_batch_size', 32):<25} |",
            f"|  Reuse drzewa            |  {str(rl_cfg.get('mcts_reuse_tree', True)):<25} |",
            f"|  League opponent pool    |  {league_status:<25} |",
            "+--------------------------+-------------------------+",
        ]
        _print_console_block(unicode_lines, ascii_lines=ascii_lines)
        print()
    
    processes = []
    result_files = []
    progress_files = []
    model_state_path = None
    
    interrupted = False
    try:
        if use_persistent_pool:
            global _SELFPLAY_POOL
            if _SELFPLAY_POOL is None or not _SELFPLAY_POOL.matches(worker_specs, device_type, temp_dir):
                _shutdown_selfplay_pool()
                _SELFPLAY_POOL = _PersistentSelfPlayPool(config, worker_specs, device_type, temp_dir)
                _SELFPLAY_POOL.start()

            task_id = f"{os.getpid()}_{time.time_ns()}"
            model_state_path = temp_dir / f"selfplay_model_{task_id}.pt"
            model_state_cpu = {
                key: tensor.detach().cpu()
                for key, tensor in model_state.items()
            }

            if league_enabled and league_runtime_snapshots_enabled:
                runtime_snapshot_path = league_runtime_dir / f"league_runtime_{task_id}.pt"
                torch.save(model_state_cpu, runtime_snapshot_path)
                runtime_snapshots = sorted(
                    league_runtime_dir.glob('*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for stale_path in runtime_snapshots[league_runtime_snapshot_keep:]:
                    try:
                        stale_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            torch.save(model_state_cpu, model_state_path)
            del model_state_cpu

            result_files, progress_files = _SELFPLAY_POOL.submit(
                task_id=task_id,
                model_state_path=model_state_path,
                temperature=rl_cfg.get('mcts_temperature'),
                worker_model_state_paths={},
                worker_opponent_state_paths=worker_opponent_state_paths,
            )
            processes = [_SELFPLAY_POOL.processes[rank] for rank, _ in worker_specs]
        else:
            if league_enabled and league_runtime_snapshots_enabled:
                runtime_snapshot_path = league_runtime_dir / f"league_runtime_{os.getpid()}_{time.time_ns()}.pt"
                model_state_cpu = {
                    key: tensor.detach().cpu()
                    for key, tensor in model_state.items()
                }
                torch.save(model_state_cpu, runtime_snapshot_path)
                del model_state_cpu
                runtime_snapshots = sorted(
                    league_runtime_dir.glob('*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for stale_path in runtime_snapshots[league_runtime_snapshot_keep:]:
                    try:
                        stale_path.unlink(missing_ok=True)
                    except Exception:
                        pass

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
                        str(worker_opponent_state_paths[rank]) if rank in worker_opponent_state_paths else None,
                    )
                )
                p.daemon = True
                p.start()
                processes.append(p)

            progress_files = [
                temp_dir / f"worker_{rank}_mcts_results.progress"
                for rank, _ in worker_specs
            ]

        # Wait for workers with a live games-completed progress bar.
        games_bar = tqdm(
            total=num_games,
            desc="🎮 Self-play gry",
            unit="gra",
            dynamic_ncols=True,
            leave=True,
        )
        try:
            pending = {rank for rank, _ in worker_specs}
            rank_to_index = {rank: idx for idx, (rank, _) in enumerate(worker_specs)}
            games_reported = {rank: 0 for rank, _ in worker_specs}
            games_bar_last = 0
            while pending:
                # Update games progress from .progress files
                total_done = 0
                for rank, pf in zip((rank for rank, _ in worker_specs), progress_files):
                    try:
                        raw = pf.read_text().strip()
                        games_reported[rank] = int(raw) if raw else 0
                    except Exception:
                        pass
                    total_done += games_reported[rank]
                inc = total_done - games_bar_last
                if inc > 0:
                    games_bar.update(inc)
                    games_bar_last = total_done

                if use_persistent_pool:
                    try:
                        message = _SELFPLAY_POOL.result_queue.get(timeout=0.2)
                        if message.get('task_id') == task_id:
                            rank = message.get('rank')
                            pending.discard(rank)
                            if not message.get('ok', False):
                                interrupted = bool(message.get('interrupt', False))
                                if interrupted:
                                    print("\nCtrl+C detected in self-play worker. Stopping workers...")
                                    raise RLTrainingInterrupted("self-play")
                                raise RuntimeError(
                                    f"Persistent self-play worker {rank} failed: {message.get('error', 'unknown error')}"
                                )
                    except queue.Empty:
                        pass

                    for rank in list(pending):
                        proc = _SELFPLAY_POOL.processes.get(rank)
                        if proc is None or proc.is_alive():
                            continue
                        interrupted = _is_interrupt_exit_code(proc.exitcode)
                        if interrupted:
                            print("\nCtrl+C detected in self-play worker. Stopping workers...")
                            raise RLTrainingInterrupted("self-play")
                        raise RuntimeError(
                            f"Persistent self-play worker {rank} exited unexpectedly with code {proc.exitcode}"
                        )
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
        if model_state_path is not None:
            try:
                model_state_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    selfplay_time = time.time() - start_time
    
    # Collect results
    collection_start = time.time()
    
    all_positions = [] if not stream_to_replay else None
    total_positions = 0
    game_lengths = []
    total_dropped_positions = 0
    total_truncated_games = 0
    total_claimable_draw_ended_games = 0
    total_completed_length_sum = 0
    total_truncated_length_sum = 0
    total_completed_white_wins = 0
    total_completed_black_wins = 0
    total_completed_draws = 0
    total_primary_model_wins = 0
    total_opponent_model_wins = 0
    total_model_match_draws = 0
    total_primary_white_games = 0
    total_primary_white_wins = 0
    total_primary_black_games = 0
    total_primary_black_wins = 0
    total_value_sum = 0.0
    total_value_sq_sum = 0.0
    total_value_count = 0
    
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
                        elif isinstance(payload, tuple) and len(payload) == 2:
                            positions, lengths = payload
                            stats = {}
                        else:
                            # Unexpected format: preserve backward compatibility best-effort.
                            positions, lengths, stats = [], [], {}

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
                        total_completed_length_sum += int((stats or {}).get('completed_length_sum', 0))
                        total_truncated_length_sum += int((stats or {}).get('truncated_length_sum', 0))
                        total_completed_white_wins += int((stats or {}).get('completed_white_wins', 0))
                        total_completed_black_wins += int((stats or {}).get('completed_black_wins', 0))
                        total_completed_draws += int((stats or {}).get('completed_draws', 0))
                        total_primary_model_wins += int((stats or {}).get('primary_model_wins', 0))
                        total_opponent_model_wins += int((stats or {}).get('opponent_model_wins', 0))
                        total_model_match_draws += int((stats or {}).get('model_match_draws', 0))
                        total_primary_white_games += int((stats or {}).get('primary_white_games', 0))
                        total_primary_white_wins += int((stats or {}).get('primary_white_wins', 0))
                        total_primary_black_games += int((stats or {}).get('primary_black_games', 0))
                        total_primary_black_wins += int((stats or {}).get('primary_black_wins', 0))
                
                result_file.unlink()
            except Exception as e:
                print(f"⚠️ Warning: Failed to load results from worker {idx}: {e}")
        else:
            print(f"⚠️ Warning: Worker {idx} result file not found")
    
    collection_time = time.time() - collection_start
    total_time = time.time() - start_time
    
    positions_per_sec = total_positions / total_time if total_time > 0 else 0
    avg_length = np.mean(game_lengths) if game_lengths else 0
    
    print(f"✅ MCTS Self-play completed:")
    print(f"   Positions: {total_positions}")
    print(f"   Games: {len(game_lengths)}")
    total_generated_positions = total_positions + total_dropped_positions
    if total_generated_positions > 0:
        kept_ratio = 100.0 * total_positions / total_generated_positions
        print(f"   Kept positions ratio: {kept_ratio:.1f}% ({total_positions}/{total_generated_positions})")
    if game_lengths:
        print(
            "   Truncated games: "
            f"{total_truncated_games}/{len(game_lengths)} "
            f"({(100.0 * total_truncated_games / max(1, len(game_lengths))):.1f}%)"
        )
        print(f"   Claimable-draw ended games: {total_claimable_draw_ended_games}")
        completed_games = max(0, len(game_lengths) - total_truncated_games)
        if completed_games > 0:
            print(
                "   Avg completed game length: "
                f"{(total_completed_length_sum / completed_games):.1f} moves"
            )
        if total_truncated_games > 0:
            print(
                "   Avg truncated game length: "
                f"{(total_truncated_length_sum / total_truncated_games):.1f} moves"
            )
    print(f"   Dropped positions (truncated): {total_dropped_positions}")
    print(f"   Self-play time: {selfplay_time:.1f}s")
    print(f"   Data collection: {collection_time:.3f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Speed: {positions_per_sec:.1f} positions/s")
    print(f"   Avg game length: {avg_length:.1f} moves")
    if total_primary_model_wins + total_opponent_model_wins + total_model_match_draws > 0:
        print(
            "   Model matchup (primary/opponent/draw): "
            f"{total_primary_model_wins}/{total_opponent_model_wins}/{total_model_match_draws}"
        )
    if total_primary_white_games + total_primary_black_games > 0:
        white_wr = (
            float(total_primary_white_wins) / float(total_primary_white_games)
            if total_primary_white_games > 0
            else 0.0
        )
        black_wr = (
            float(total_primary_black_wins) / float(total_primary_black_games)
            if total_primary_black_games > 0
            else 0.0
        )
        print(
            "   Primary winrate by color (W/B): "
            f"{white_wr:.2%}/{black_wr:.2%} "
            f"(games: {total_primary_white_games}/{total_primary_black_games})"
        )

    completed_games = max(0, len(game_lengths) - total_truncated_games)
    completed_draw_rate = (
        float(total_completed_draws) / float(completed_games)
        if completed_games > 0
        else 0.0
    )
    avg_game_value = (total_value_sum / total_value_count) if total_value_count > 0 else 0.0
    value_var = (total_value_sq_sum / total_value_count) - (avg_game_value ** 2) if total_value_count > 0 else 0.0
    value_std = math.sqrt(max(0.0, value_var))
    selfplay_stats = {
        'completed_games': int(completed_games),
        'completed_white_wins': int(total_completed_white_wins),
        'completed_black_wins': int(total_completed_black_wins),
        'completed_draws': int(total_completed_draws),
        'primary_model_wins': int(total_primary_model_wins),
        'opponent_model_wins': int(total_opponent_model_wins),
        'model_match_draws': int(total_model_match_draws),
        'primary_white_games': int(total_primary_white_games),
        'primary_white_wins': int(total_primary_white_wins),
        'primary_black_games': int(total_primary_black_games),
        'primary_black_wins': int(total_primary_black_wins),
        'completed_draw_rate': float(completed_draw_rate),
        'avg_game_value': float(avg_game_value),
        'value_std': float(value_std),
        'truncated_games': int(total_truncated_games),
        'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
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
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

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
        avg_moves = rl_cfg.get('avg_moves_per_game', 0)
        try:
            avg_moves = float(avg_moves)
        except Exception:
            avg_moves = 0
        if avg_moves <= 0:
            raise ValueError("avg_moves_per_game must be > 0 when using replay_buffer_multiplier.")
        computed_size = int(games_per_iter * avg_moves * replay_multiplier)
        rl_cfg['replay_buffer_size'] = computed_size
        print(
            "Replay buffer size: "
            f"{computed_size} (games_per_iteration {games_per_iter} x "
            f"avg_moves_per_game {avg_moves} x multiplier {replay_multiplier})"
        )
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
        print(f"Self-play save every: {save_every} games (games_per_iteration {games_per_iter} x multiplier {save_mult})")
    else:
        rl_cfg['self_play_save_every_games_resolved'] = 0
        print("Self-play save every: disabled")

    worker_log_mode = "verbose" if bool(rl_cfg.get('self_play_worker_verbose', False)) else "minimal"
    wait_bar_mode = "on" if bool(rl_cfg.get('self_play_wait_progress', False)) else "off"
    print(f"Self-play worker logs: {worker_log_mode} (wait-progress: {wait_bar_mode})")

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
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
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
            print("⚠️ bfloat16 not supported")
            use_bfloat16 = False
    
    best_model_il_path = base_dir / config['paths']['best_model_il']
    best_model_rl_path = base_dir / config['paths']['best_model_rl']
    version_best_model_path = rl_dir / f"{model_file_tag}_best.pt"
    latest_checkpoint_path = rl_dir / f"{model_file_tag}_latest.pt"
    total_iterations = int(config['reinforcement_learning']['iterations'])

    print("\nPreparing model for startup menu...")
    config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("Model ready")

    rl_hparam_resolution = resolve_rl_hyperparameters(
        config=config,
        model=model,
        device=device,
        base_dir=base_dir,
    )
    print(
        "RL hyperparameters source: "
        f"{rl_hparam_resolution.get('source', 'config')} | "
        f"batch_size={config['reinforcement_learning']['batch_size']} | "
        f"lr={float(config['reinforcement_learning']['learning_rate']):.6g}"
    )

    startup_plan = plan_rl_startup(
        model=model,
        device=device,
        models_dir=models_dir,
        best_model_rl_path=best_model_rl_path,
        rl_dir=rl_dir,
    )

    # Initialize logger after startup selection.
    logger = TrainingLogger(
        logs_dir,
        experiment_name=f"rl_training_{model_version}_mcts",
        mode="rl"
    )
    global _LAST_RUN_LOG_CSV, _LAST_RUN_LOG_PNG
    _LAST_RUN_LOG_CSV = logger.csv_path
    _LAST_RUN_LOG_PNG = logger.plot_path

    elo_coordinator = RLEloCoordinator(
        model=model,
        config=config,
        device=device,
        logger=logger,
    )
    elo_coordinator.print_startup_summary()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['reinforcement_learning']['learning_rate'],
        weight_decay=config['reinforcement_learning'].get('weight_decay', 0.01),
        fused=True if torch.cuda.is_available() else False
    )

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    startup_state = apply_rl_startup_plan(
        startup_plan=startup_plan,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        default_new_checkpoint=best_model_il_path,
    )
    start_mode = startup_state.get("start_mode", "new")
    selected_checkpoint_label = startup_state.get("selected_checkpoint_label")
    start_iteration = int(startup_state.get("start_iteration", 0) or 0)
    resumed_best_win_rate = float(startup_state.get("best_win_rate", 0.0) or 0.0)
    selected_compatibility_ratio = startup_state.get("selected_compatibility_ratio")
    transfer_match_ratio = startup_state.get("transfer_match_ratio")
    selected_entry = startup_plan.get("selected_entry") or {}

    if start_mode == "new":
        source_label = (
            f"{best_model_il_path.name} (init)"
            if best_model_il_path.exists()
            else "new (scratch)"
        )
    else:
        source_label = "selected checkpoint"
        if selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name

    print_active_model_summary(
        model,
        config,
        title="Active Model (RL)",
        source_label=source_label,
        startup_mode=start_mode,
        checkpoint_label=selected_checkpoint_label,
        device=device,
        selected_entry=selected_entry,
    )

    if start_mode == "new":
        run_context = "startup: new RL training"
    elif start_mode == "resume":
        run_context = (
            f"startup: resumed full state, next iteration {start_iteration + 1}, "
            f"best win_rate={resumed_best_win_rate:.2%}"
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

    
    # Best files are updated only when evaluation confirms model improvement.
    # Initialize replay buffer (prioritized or standard)
    use_prioritized = config['reinforcement_learning'].get('use_prioritized_replay', False)
    
    replay_fp16 = config['reinforcement_learning'].get('replay_fp16', False)

    if use_prioritized:
        replay_buffer = PrioritizedReplayBuffer(
            max_size=config['reinforcement_learning']['replay_buffer_size'],
            alpha=config['reinforcement_learning'].get('priority_alpha', 0.6),
            beta_start=config['reinforcement_learning'].get('priority_beta_start', 0.4),
            beta_end=config['reinforcement_learning'].get('priority_beta_end', 1.0),
            epsilon=config['reinforcement_learning'].get('priority_epsilon', 0.01),
            use_fp16=replay_fp16,
        )
    else:
        replay_buffer = ReplayBuffer(
            config['reinforcement_learning']['replay_buffer_size'],
            use_fp16=replay_fp16,
        )
    
    # Initialize temperature schedule
    use_temp_schedule = config['reinforcement_learning'].get('use_temperature_schedule', False)
    
    if use_temp_schedule:
        raw_decay_iterations = config['reinforcement_learning'].get('temperature_decay_iterations', 'auto')
        if isinstance(raw_decay_iterations, str) and raw_decay_iterations.strip().lower() == 'auto':
            decay_iterations = max(1, int(total_iterations))
        else:
            try:
                decay_iterations = int(raw_decay_iterations)
            except Exception:
                decay_iterations = max(1, int(total_iterations))
            decay_iterations = max(1, min(decay_iterations, int(total_iterations)))

        temp_schedule = TemperatureSchedule(
            start_temp=config['reinforcement_learning'].get('temperature_start', 1.5),
            end_temp=config['reinforcement_learning'].get('temperature_end', 0.5),
            decay_iterations=decay_iterations
        )
    
    # LR schedule (same shape as IL, but stepped per RL iteration)
    use_lr_schedule = config['reinforcement_learning'].get('use_lr_schedule', False)
    lr_base = float(config['reinforcement_learning']['learning_rate'])
    warmup_pct = float(config['reinforcement_learning'].get('warmup_pct', 0.05))
    min_lr_ratio = float(config['reinforcement_learning'].get('min_lr_ratio', 0.25))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_iters = max(1, int(round(total_iterations * warmup_pct)))

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
        except Exception:
            return default_weight

        if total_iterations <= 1:
            return end_w

        progress = max(0.0, min(1.0, iter_idx / max(1, total_iterations - 1)))
        return start_w + (end_w - start_w) * progress

    def _compute_league_worker_fraction(iter_idx):
        rl_cfg_local = config.get('reinforcement_learning', {})
        base_fraction = float(rl_cfg_local.get('league_worker_fraction', 0.35))
        base_fraction = max(0.0, min(1.0, base_fraction))
        if not bool(rl_cfg_local.get('league_worker_fraction_schedule_enabled', True)):
            return base_fraction

        try:
            start_frac = float(rl_cfg_local.get('league_worker_fraction_start', base_fraction))
        except Exception:
            start_frac = base_fraction
        try:
            end_frac = float(rl_cfg_local.get('league_worker_fraction_end', 1.0))
        except Exception:
            end_frac = 1.0

        start_frac = max(0.0, min(1.0, start_frac))
        end_frac = max(0.0, min(1.0, end_frac))

        raw_ramp_iters = rl_cfg_local.get('league_worker_fraction_ramp_iterations', 'auto')
        if isinstance(raw_ramp_iters, str) and raw_ramp_iters.strip().lower() == 'auto':
            ramp_iters = max(1, int(total_iterations))
        else:
            try:
                ramp_iters = max(1, int(raw_ramp_iters))
            except Exception:
                ramp_iters = max(1, int(total_iterations))

        progress = max(0.0, min(1.0, float(iter_idx) / float(max(1, ramp_iters - 1))))
        return start_frac + (end_frac - start_frac) * progress

    def _resolve_dynamic_uniform_fraction(rl_cfg_local, draw_rate):
        base = float(rl_cfg_local.get('priority_uniform_fraction', 0.25))
        if not bool(rl_cfg_local.get('replay_dynamic_uniform_enabled', True)):
            return max(0.0, min(1.0, base))

        target = float(rl_cfg_local.get('replay_dynamic_uniform_draw_target', 0.55))
        gain = float(rl_cfg_local.get('replay_dynamic_uniform_draw_gain', 0.50))
        min_u = float(rl_cfg_local.get('replay_dynamic_uniform_min', 0.20))
        max_u = float(rl_cfg_local.get('replay_dynamic_uniform_max', 0.50))
        dynamic = base + gain * (draw_rate - target)
        return max(min_u, min(max_u, dynamic))

    def _resolve_priority_age_decay_lambda(rl_cfg_local, draw_rate):
        base = float(rl_cfg_local.get('replay_priority_age_decay_lambda', 0.0))
        if base <= 0:
            return 0.0
        if not bool(rl_cfg_local.get('replay_priority_age_decay_dynamic', True)):
            return max(0.0, base)

        target = float(rl_cfg_local.get('replay_dynamic_uniform_draw_target', 0.55))
        draw_gain = float(rl_cfg_local.get('replay_priority_age_decay_draw_gain', 2.0))
        max_lambda = float(rl_cfg_local.get('replay_priority_age_decay_lambda_max', 0.01))
        factor = 1.0 + draw_gain * max(0.0, draw_rate - target)
        return max(0.0, min(max_lambda, base * factor))

    def _resolve_anti_draw_overrides(
        rl_cfg_local,
        base_temp,
        prev_draw_rate,
        base_dirichlet_weight,
        base_temp_threshold,
    ):
        """Increase exploration pressure when previous iteration draw rate is too high."""
        try:
            temp = float(base_temp)
        except Exception:
            temp = float(rl_cfg_local.get('mcts_temperature', 1.0))

        dirichlet_weight = float(base_dirichlet_weight)
        temp_threshold = int(base_temp_threshold)

        if prev_draw_rate is None or not bool(rl_cfg_local.get('anti_draw_adaptive_exploration', True)):
            return temp, dirichlet_weight, temp_threshold, 0.0

        high_draw_threshold = float(rl_cfg_local.get('anti_draw_high_draw_threshold', 0.95))
        if prev_draw_rate <= high_draw_threshold:
            return temp, dirichlet_weight, temp_threshold, 0.0

        severity = max(0.0, min(1.0, (prev_draw_rate - high_draw_threshold) / max(1e-6, 1.0 - high_draw_threshold)))
        max_temp_boost = float(rl_cfg_local.get('anti_draw_max_temp_boost', 0.25))
        max_dirichlet_boost = float(rl_cfg_local.get('anti_draw_max_dirichlet_weight_boost', 0.20))
        threshold_boost_moves = int(rl_cfg_local.get('anti_draw_temp_threshold_boost_moves', 20))
        threshold_cap = int(rl_cfg_local.get('anti_draw_temp_threshold_cap', max(1, temp_threshold + threshold_boost_moves)))

        boosted_temp = min(2.0, temp + max_temp_boost * severity)
        boosted_dirichlet_weight = min(0.50, dirichlet_weight + max_dirichlet_boost * severity)
        boosted_temp_threshold = temp_threshold + int(round(threshold_boost_moves * severity))
        boosted_temp_threshold = min(threshold_cap, boosted_temp_threshold)
        return boosted_temp, boosted_dirichlet_weight, boosted_temp_threshold, severity

    if use_lr_schedule:
        print(
            f"✅ LR schedule: warmup={warmup_iters} iters → "
            f"cosine decay (base_lr={lr_base:.6g}, min_lr_ratio={min_lr_ratio})"
        )

    print("\n=== Starting RL training with PROPER MCTS ===")
    print("🎯 OPTIMIZATIONS:")
    print(f"   • MCTS self-play (AlphaZero approach)")
    print(f"   • MCTS simulations: {config['reinforcement_learning']['mcts_simulations']}")
    print(f"   • Tree reuse: {config['reinforcement_learning'].get('mcts_reuse_tree', True)}")
    print(f"   • Batch MCTS: {config['reinforcement_learning'].get('mcts_batch_size', 32)}")
    print(
        "   • Dynamic c_puct: "
        f"{config['reinforcement_learning'].get('mcts_dynamic_c_puct', True)} "
        f"(init={float(config['reinforcement_learning'].get('mcts_c_puct_init', config['reinforcement_learning'].get('mcts_c_puct', 1.5))):.2f}, "
        f"base={int(config['reinforcement_learning'].get('mcts_c_puct_base', 19652))}, "
        f"max={config['reinforcement_learning'].get('mcts_c_puct_max', None)})"
    )
    print(
        "   • FPU: "
        f"{config['reinforcement_learning'].get('mcts_use_fpu', True)} "
        f"(reduction={float(config['reinforcement_learning'].get('mcts_fpu_reduction', 0.30)):.2f}, "
        f"absolute={config['reinforcement_learning'].get('mcts_fpu_absolute', None)})"
    )
    print(f"   • Persistent self-play workers: {config['reinforcement_learning'].get('persistent_self_play_workers', True)}")
    print(f"   • Stream self-play to replay: {config['reinforcement_learning'].get('self_play_stream_to_replay', True)}")
    print(f"   • 🆕 Prioritized Replay: {use_prioritized}")
    print(f"   • 🆕 Temperature Schedule: {use_temp_schedule}")
    print(f"   • 📊 Policy Accuracy & Value MAE tracking")
    print(f"   • Replay buffer capacity: {config['reinforcement_learning']['replay_buffer_size']:,} positions")
    print(
        f"   • Estimated positions per iteration: "
        f"{int(config['reinforcement_learning']['games_per_iteration'] * config['reinforcement_learning']['avg_moves_per_game']):,}"
    )
    if use_prioritized:
        print(
            f"   • Prioritized replay uniform mix: "
            f"{float(config['reinforcement_learning'].get('priority_uniform_fraction', 0.25)):.0%} uniform"
        )
    
    if start_iteration >= total_iterations:
        print(
            f"Resume start iteration ({start_iteration + 1}) exceeds configured total "
            f"({total_iterations}). Nothing to train."
        )
        logger.plot()
        return

    rl_cfg = config['reinforcement_learning']
    base_mcts_temperature_threshold = int(rl_cfg.get('mcts_temperature_threshold', 16))
    base_mcts_dirichlet_weight = float(rl_cfg.get('mcts_dirichlet_weight', 0.25))
    prev_completed_draw_rate = None

    training_interrupted = False
    interrupted_stage = None
    last_logged_iteration = start_iteration if start_iteration > 0 else None

    try:
        for iteration in range(start_iteration, total_iterations):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{total_iterations}")
            print('='*70)
            
            # Update learning rate (cosine decay + warmup)
            current_lr = _compute_lr(iteration)
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            
            # Get current temperature
            if use_temp_schedule:
                base_current_temp = temp_schedule.get_temperature(iteration)
            else:
                base_current_temp = config['reinforcement_learning']['mcts_temperature']

            current_temp, current_dirichlet_weight, current_temp_threshold, anti_draw_severity = _resolve_anti_draw_overrides(
                rl_cfg,
                base_current_temp,
                prev_completed_draw_rate,
                base_mcts_dirichlet_weight,
                base_mcts_temperature_threshold,
            )
            print(f"🌡️ Temperature: {current_temp:.2f}")
            config['reinforcement_learning']['mcts_temperature'] = current_temp
            config['reinforcement_learning']['mcts_dirichlet_weight'] = current_dirichlet_weight
            config['reinforcement_learning']['mcts_temperature_threshold'] = current_temp_threshold
            if anti_draw_severity > 0.0:
                print(
                    f"🚨 Anti-draw boost active (severity={anti_draw_severity:.2f}): "
                    f"dirichlet_weight={current_dirichlet_weight:.3f}, "
                    f"temp_threshold={current_temp_threshold}"
                )
            
            if use_lr_schedule:
                print(f"📉 LR: {current_lr:.2e}")

            current_league_fraction = _compute_league_worker_fraction(iteration)
            rl_cfg['league_worker_fraction'] = float(current_league_fraction)
            print(f"🥊 League worker fraction: {current_league_fraction:.2f}")

            current_value_loss_weight = _compute_value_loss_weight(iteration)
            print(f"⚖️ Value loss weight: {current_value_loss_weight:.3f}")
            
            # Update beta for importance sampling
            if use_prioritized:
                progress = iteration / total_iterations
                replay_buffer.update_beta(progress)
                print(f"🎯 Beta (IS): {replay_buffer.beta:.3f}")
            
            # Self-play with MCTS
            model.eval()
            positions, positions_added, avg_game_length, positions_per_sec, selfplay_time, collection_time, selfplay_stats = \
                play_games_parallel_mcts(
                    model,
                    config,
                    device,
                    config['reinforcement_learning']['games_per_iteration'],
                    replay_buffer=replay_buffer,
                )
            
            # Add to replay buffer
            for position in positions:
                replay_buffer.add(position)
            
            print(f"Replay buffer: {len(replay_buffer)} positions (+{positions_added})")
            print(
                f"📊 Self-play stats: draw_rate={float((selfplay_stats or {}).get('completed_draw_rate', 0.0)):.2%}, "
                f"avg_value={float((selfplay_stats or {}).get('avg_game_value', 0.0)):.3f}, "
                f"value_std={float((selfplay_stats or {}).get('value_std', 0.0)):.3f}"
            )
            pw_games = int((selfplay_stats or {}).get('primary_white_games', 0))
            pb_games = int((selfplay_stats or {}).get('primary_black_games', 0))
            if pw_games + pb_games > 0:
                pw_wr = (
                    float((selfplay_stats or {}).get('primary_white_wins', 0)) / float(pw_games)
                    if pw_games > 0
                    else 0.0
                )
                pb_wr = (
                    float((selfplay_stats or {}).get('primary_black_wins', 0)) / float(pb_games)
                    if pb_games > 0
                    else 0.0
                )
                print(
                    f"📊 Primary WR by color: white={pw_wr:.2%} ({pw_games} gier), "
                    f"black={pb_wr:.2%} ({pb_games} gier)"
                )
            prev_completed_draw_rate = float((selfplay_stats or {}).get('completed_draw_rate', 0.0))

            dynamic_uniform_fraction = float(config['reinforcement_learning'].get('priority_uniform_fraction', 0.25))
            dynamic_age_decay_lambda = 0.0
            avg_sample_age = 0.0
            avg_policy_entropy = 0.0
            avg_value_pred_std = 0.0
            
            # Training with metrics
            if len(replay_buffer) >= config['reinforcement_learning']['batch_size']:
                print("Training...")
                model.train()
                total_loss = 0
                total_policy = 0
                total_value = 0
                total_policy_entropy = 0
                total_value_pred_std = 0
                total_sample_age = 0
                sample_age_steps = 0
                
                # 📊 Initialize metrics calculator
                metrics_calc = MetricsCalculator()
                
                num_batches = len(replay_buffer) // config['reinforcement_learning']['batch_size']
                
                for batch_idx in tqdm(
                    range(config['reinforcement_learning']['train_epochs_per_iteration'] * num_batches),
                    desc="Training",
                ):
                    # Prioritized or standard sampling
                    if use_prioritized:
                        draw_rate = float((selfplay_stats or {}).get('completed_draw_rate', 0.0))
                        uniform_fraction = _resolve_dynamic_uniform_fraction(
                            config['reinforcement_learning'],
                            draw_rate,
                        )
                        age_decay_lambda = _resolve_priority_age_decay_lambda(
                            config['reinforcement_learning'],
                            draw_rate,
                        )
                        dynamic_uniform_fraction = float(uniform_fraction)
                        dynamic_age_decay_lambda = float(age_decay_lambda)
                        if batch_idx == 0:
                            print(
                                f"🎛️ Replay mix: uniform={uniform_fraction:.3f}, "
                                f"age_decay={age_decay_lambda:.6f}, draw_rate={draw_rate:.2%}"
                            )
                        if uniform_fraction > 0:
                            batch, indices, weights = replay_buffer.sample_mixed(
                                config['reinforcement_learning']['batch_size'],
                                uniform_fraction=uniform_fraction,
                                age_decay_lambda=age_decay_lambda,
                            )
                        else:
                            batch, indices, weights = replay_buffer.sample(
                                config['reinforcement_learning']['batch_size'],
                                age_decay_lambda=age_decay_lambda,
                            )

                        loss, policy_loss, value_loss, policy_entropy, value_pred_std = train_on_batch_rl(
                            model,
                            optimizer,
                            batch,
                            indices,
                            weights,
                            config,
                            device,
                            scaler,
                            replay_buffer,
                            metrics_calc,
                            value_weight_override=current_value_loss_weight,
                        )
                        total_sample_age += float(getattr(replay_buffer, 'last_sample_age_mean', 0.0))
                        sample_age_steps += 1
                    else:
                        batch = replay_buffer.sample(config['reinforcement_learning']['batch_size'])
                        loss, policy_loss, value_loss, policy_entropy, value_pred_std = train_on_batch_rl(
                            model,
                            optimizer,
                            batch,
                            None,
                            None,
                            config,
                            device,
                            scaler,
                            replay_buffer,
                            metrics_calc,
                            value_weight_override=current_value_loss_weight,
                        )
                    
                    total_loss += loss
                    total_policy += policy_loss
                    total_value += value_loss
                    total_policy_entropy += policy_entropy
                    total_value_pred_std += value_pred_std
                avg_loss = total_loss / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_policy = total_policy / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_value = total_value / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_policy_entropy = total_policy_entropy / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_value_pred_std = total_value_pred_std / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_sample_age = (total_sample_age / sample_age_steps) if sample_age_steps > 0 else 0.0
                
                # 📊 Compute metrics
                train_metrics = metrics_calc.compute()
                
                print(f"Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
                print(f"📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                      f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                      f"MAE: {train_metrics['value_mae']:.4f}")
                print(
                    f"📈 Entropy: {avg_policy_entropy:.4f}, "
                    f"Pred value std: {avg_value_pred_std:.4f}, "
                    f"Avg sample age: {avg_sample_age:.1f}"
                )
            else:
                avg_loss = avg_policy = avg_value = 0
                train_metrics = {}
            
            # Evaluation
            win_rate = None
            estimated_elo = None
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
                print("Evaluating vs best...")
                model.eval()
                win_rate = evaluate_models(
                    model, best_model, config, device,
                    config['reinforcement_learning']['eval_games']
                )
                print(f"Win rate: {win_rate:.2%}")
                estimated_elo = elo_coordinator.maybe_evaluate(iteration + 1, win_rate=win_rate)
                
                # Log with all metrics
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    estimated_elo=estimated_elo,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    win_rate=win_rate,
                    buffer_size=len(replay_buffer),
                    avg_game_length=avg_game_length,
                    positions_per_sec=positions_per_sec,
                    selfplay_time=selfplay_time,
                    data_collection_time=collection_time,
                    temperature=current_temp,
                    beta=replay_buffer.beta if use_prioritized else None,
                    completed_draw_rate=(selfplay_stats or {}).get('completed_draw_rate', None),
                    dynamic_uniform_fraction=dynamic_uniform_fraction if use_prioritized else None,
                    priority_age_decay_lambda=dynamic_age_decay_lambda if use_prioritized else None,
                    avg_sample_age=avg_sample_age if use_prioritized else None,
                    avg_game_value=(selfplay_stats or {}).get('avg_game_value', None),
                    value_std=(selfplay_stats or {}).get('value_std', None),
                    policy_entropy=avg_policy_entropy,
                    value_pred_std=avg_value_pred_std,
                )
                last_logged_iteration = iteration + 1
                logger.plot()
                
                if win_rate >= config['reinforcement_learning']['win_rate_threshold']:
                    print("✅ New best model!")
                    best_model.load_state_dict(model.state_dict())
                    best_win_rate_so_far = max(best_win_rate_so_far, float(win_rate))
                    
                    model_to_save = model
                    save_checkpoint(
                        model_to_save, None, iteration, avg_loss,
                        str(best_model_rl_path),
                        {
                            'win_rate': win_rate,
                            'policy_loss': avg_policy,
                            'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                            'value_mae': train_metrics.get('value_mae', 0),
                            'version': model_version,
                            'startup_mode': start_mode,
                            'model_architecture': model_architecture,
                        },
                        save_optimizer=False,
                        save_dtype=torch.bfloat16 if use_bfloat16 else None
                    )
                    
                    size_mb = best_model_rl_path.stat().st_size / (1024**2)
                    print(f"💾 Saved: {best_model_rl_path} ({size_mb:.1f} MB)")
                    if version_best_model_path != best_model_rl_path:
                        shutil.copy2(best_model_rl_path, version_best_model_path)
                        version_best_size_mb = version_best_model_path.stat().st_size / (1024 ** 2)
                        print(
                            f"💾 Version best updated: {version_best_model_path.name} "
                            f"({version_best_size_mb:.1f} MB)"
                        )
            else:
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    estimated_elo=None,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    buffer_size=len(replay_buffer),
                    avg_game_length=avg_game_length,
                    positions_per_sec=positions_per_sec,
                    selfplay_time=selfplay_time,
                    data_collection_time=collection_time,
                    temperature=current_temp,
                    beta=replay_buffer.beta if use_prioritized else None,
                    completed_draw_rate=(selfplay_stats or {}).get('completed_draw_rate', None),
                    dynamic_uniform_fraction=dynamic_uniform_fraction if use_prioritized else None,
                    priority_age_decay_lambda=dynamic_age_decay_lambda if use_prioritized else None,
                    avg_sample_age=avg_sample_age if use_prioritized else None,
                    avg_game_value=(selfplay_stats or {}).get('avg_game_value', None),
                    value_std=(selfplay_stats or {}).get('value_std', None),
                    policy_entropy=avg_policy_entropy,
                    value_pred_std=avg_value_pred_std,
                )
                last_logged_iteration = iteration + 1
            
            latest_metadata = {
                'win_rate': win_rate,
                'policy_loss': avg_policy,
                'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                'value_mae': train_metrics.get('value_mae', 0),
                'version': model_version,
                'startup_mode': start_mode,
                'model_architecture': model_architecture,
            }
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
            latest_size_mb = latest_checkpoint_path.stat().st_size / (1024 ** 2)
            print(f"💾 Latest checkpoint updated: {latest_checkpoint_path.name} ({latest_size_mb:.1f} MB)")
            
            gc.collect()
    except RLTrainingInterrupted as exc:
        training_interrupted = True
        interrupted_stage = str(exc) or "self-play"
    except KeyboardInterrupt:
        training_interrupted = True
        interrupted_stage = "runtime"
    finally:
        _shutdown_selfplay_pool()

    final_elo_iteration = last_logged_iteration
    if final_elo_iteration is not None:
        try:
            elo_coordinator.final_evaluate(final_elo_iteration, interrupted=training_interrupted)
        except KeyboardInterrupt:
            training_interrupted = True
            interrupted_stage = "elo shutdown"
            print("Final Elo interrupted by user. Shutting down...")
    elif training_interrupted:
        print("Final Elo skipped: no completed iteration available to evaluate.")

    logger.plot()
    if training_interrupted:
        _handle_graceful_interrupt(logger=logger, stage=interrupted_stage)
        return

    print("\n=== Training complete ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _handle_graceful_interrupt(logger=None, stage="runtime")
