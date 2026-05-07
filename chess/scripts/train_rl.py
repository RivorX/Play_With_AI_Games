"""
Reinforcement Learning Training Script with Comprehensive Metrics

NEW Features:
- đź“Š Policy Accuracy tracking during training
- đź“Š Value MAE monitoring
- đź“Š Enhanced self-play statistics
"""

import os
import sys
import signal
import copy
import shutil
import contextlib
import subprocess
import multiprocessing as _stdlib_mp
from collections import defaultdict, deque

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
from utils.shared.elo_estimator import estimate_model_elo
from utils.rl.replay import ReplayBuffer
from utils.rl.temperature import TemperatureSchedule, AdaptiveTemperatureController
from utils.rl.training_rl import train_on_batch_rl, evaluate_models, evaluate_models_no_mcts
from utils.rl.startup import plan_rl_startup, apply_rl_startup_plan
from utils.rl.profiler import print_selfplay_profiler
from utils.rl.opponent_scheduler import (
    _safe_score_rate,
    _build_selfplay_opponent_assignments,
    _update_adaptive_opponent_scheduler,
)
from utils.rl.persistent_pool import (
    get_or_create_selfplay_pool,
    _shutdown_selfplay_pool,
    _terminate_process_tree,
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
from utils.shared.model_view import print_active_model_summary
from utils.shared.syzygy_manager import ensure_syzygy_tables, describe_syzygy_status


_LAST_RUN_LOG_CSV = None
_LAST_RUN_LOG_PNG = None
_SELFPLAY_CONFIG_PRINTED = False


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
            "eval_log",
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


def _round_replay_capacity(value, quantum):
    quantum = max(1, int(quantum))
    value = max(1, int(math.ceil(float(value))))
    return int(math.ceil(value / quantum) * quantum)


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
        self.min_score_rate_for_stockfish = float(
            self.elo_config.get("rl_min_score_rate_for_stockfish", 0.50)
        )
        self.min_true_win_rate_for_stockfish = float(
            self.elo_config.get("rl_min_true_win_rate_for_stockfish", 0.0)
        )
        self.last_eval_score_rate = None
        self.last_eval_true_win_rate = None
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
            f"min_score_rate={self.min_score_rate_for_stockfish:.0%}, "
            f"min_true_win_rate={self.min_true_win_rate_for_stockfish:.0%}"
        )
        if self.final_on_shutdown:
            print(
                f"Elo final (RL): use_mcts={bool(final_use_mcts)}, "
                f"mcts_simulations={int(final_sims)}"
            )
            print("Elo final (RL): after Ctrl+C, one shutdown Elo runs; press Ctrl+C again to cancel it.")

    def maybe_evaluate(self, iteration_num, score_rate=None, true_win_rate=None):
        if not self.enabled:
            return None
        self.last_eval_score_rate = score_rate
        self.last_eval_true_win_rate = true_win_rate
        self.eval_counter += 1
        if (self.eval_counter % self.every_n_evals) != 0:
            return None
        if score_rate is None:
            print("Skipping Stockfish Elo: missing RL score rate vs best model.")
            return None
        if float(score_rate) < self.min_score_rate_for_stockfish:
            print(
                "Skipping Stockfish Elo: "
                f"score rate {float(score_rate):.2%} < required {self.min_score_rate_for_stockfish:.0%}."
            )
            return None
        if true_win_rate is not None and float(true_win_rate) < self.min_true_win_rate_for_stockfish:
            print(
                "Skipping Stockfish Elo: "
                f"true win rate {float(true_win_rate):.2%} < required {self.min_true_win_rate_for_stockfish:.0%}."
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

    def _run_estimate_subprocess(self, iteration_num, elo_config, interrupt_message):
        mp_ctx = mp.get_context('spawn')
        result_queue = mp_ctx.Queue()
        cancel_event = mp_ctx.Event()
        worker_device = str(self.eval_device.type)
        model_state_cpu = _snapshot_model_state_cpu(self.model, share_memory=True)
        process = mp_ctx.Process(
            target=_rl_elo_worker,
            args=(
                int(iteration_num),
                model_state_cpu,
                self.config,
                elo_config,
                worker_device,
                result_queue,
                cancel_event,
            ),
        )
        process.daemon = True
        process.start()

        try:
            while True:
                try:
                    message = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if not process.is_alive():
                        break
                    continue
                if message is None:
                    continue
                if int(message.get("iteration", iteration_num)) != int(iteration_num):
                    continue
                if message.get("cancelled"):
                    self.interrupted_during_elo = True
                    return None
                if message.get("error"):
                    print(f"Elo estimation failed: {message['error']}")
                    return None
                return (message.get("result") or {})
        except KeyboardInterrupt:
            with contextlib.suppress(Exception):
                cancel_event.set()
            self.interrupted_during_elo = True
            print(interrupt_message)
            # Force-stop worker process tree immediately on user interrupt.
            _terminate_process_tree(process, timeout_s=0.0)
            return None
        finally:
            _terminate_process_tree(process, timeout_s=0.0)
            with contextlib.suppress(Exception):
                result_queue.close()

    def _run_estimate(self, iteration_num, reason_label, final_override=False):
        elo_config = dict(self.elo_config)
        # RL Elo is synchronous: training is paused while Stockfish runs, so
        # do not keep "training-friendly" CPU reservations that were intended
        # for async IL/background evaluation.
        elo_config.setdefault("stockfish_priority", "normal")
        elo_config.setdefault("stockfish_hide_window", True)
        elo_config.setdefault("max_error_logs_per_type", 8)
        elo_config["prioritize_training"] = False
        elo_config["reserve_dataloader_workers"] = False
        elo_config["free_threads_utilization"] = 1.0
        # RL Elo should not use the generic auto "cpu_total - 2" worker reserve.
        elo_config["auto_worker_reserve_cpus"] = 0
        if not final_override:
            if "rl_games_per_level" in self.elo_config:
                elo_config["games_per_level"] = int(self.elo_config.get("rl_games_per_level"))
            if "rl_levels" in self.elo_config:
                elo_config["levels"] = list(self.elo_config.get("rl_levels") or [])
            if "rl_stockfish_time_limit" in self.elo_config:
                elo_config["stockfish_time_limit"] = float(self.elo_config.get("rl_stockfish_time_limit"))
            if "rl_max_moves" in self.elo_config:
                elo_config["max_moves"] = int(self.elo_config.get("rl_max_moves"))
            if "rl_workers" in self.elo_config:
                elo_config["workers"] = int(self.elo_config.get("rl_workers"))
            if "rl_stockfish_threads" in self.elo_config:
                elo_config["stockfish_threads"] = int(self.elo_config.get("rl_stockfish_threads"))
            if "rl_stockfish_hash_mb" in self.elo_config:
                elo_config["stockfish_hash_mb"] = int(self.elo_config.get("rl_stockfish_hash_mb"))
        if final_override:
            # Final/shutdown Elo runs when training is paused/stopped;
            # allow full CPU budget for faster estimation.
            elo_config["prioritize_training"] = False
            elo_config["reserve_dataloader_workers"] = False
            elo_config["free_threads_utilization"] = 1.0
            elo_config["stockfish_priority"] = "normal"
            # Final estimate is blocking/user-visible, so always show progress.
            elo_config["progress_bar"] = "always"
            if "final_rl_use_mcts" in self.elo_config:
                elo_config["use_mcts"] = bool(self.elo_config.get("final_rl_use_mcts"))
            if "final_rl_mcts_simulations" in self.elo_config:
                elo_config["mcts_simulations"] = int(self.elo_config.get("final_rl_mcts_simulations"))
            if "final_rl_games_per_level" in self.elo_config:
                elo_config["games_per_level"] = int(self.elo_config.get("final_rl_games_per_level"))
            if "final_rl_levels" in self.elo_config:
                elo_config["levels"] = list(self.elo_config.get("final_rl_levels") or [])
            if "final_rl_stockfish_time_limit" in self.elo_config:
                elo_config["stockfish_time_limit"] = float(self.elo_config.get("final_rl_stockfish_time_limit"))
            if "final_rl_max_moves" in self.elo_config:
                elo_config["max_moves"] = int(self.elo_config.get("final_rl_max_moves"))
            if "final_rl_workers" in self.elo_config:
                elo_config["workers"] = int(self.elo_config.get("final_rl_workers"))
            if "final_rl_stockfish_threads" in self.elo_config:
                elo_config["stockfish_threads"] = int(self.elo_config.get("final_rl_stockfish_threads"))
            if "final_rl_stockfish_hash_mb" in self.elo_config:
                elo_config["stockfish_hash_mb"] = int(self.elo_config.get("final_rl_stockfish_hash_mb"))

        print(f"\nEstimating Elo vs Stockfish ({reason_label})...")
        if interrupted := bool(final_override and reason_label == "shutdown"):
            print("Training stop requested by user. Running one final Elo estimate; press Ctrl+C again to cancel.")
        max_attempts = 1
        elo_result = None
        if interrupted:
            interrupt_message = "\nSecond Ctrl+C detected. Final Elo estimation aborted immediately."
        else:
            interrupt_message = "\nCtrl+C detected during Stockfish evaluation. Cancelling evaluation..."

        for attempt_idx in range(max_attempts):
            if interrupted:
                elo_result = self._run_estimate_subprocess(iteration_num, elo_config, interrupt_message)
                break
            elo_cancel_event = threading.Event()
            sigint_state = {"triggered": False}
            try:
                with _temporary_sigint_cancel_handler(
                    elo_cancel_event,
                    message=interrupt_message,
                ) as sigint_state:
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
            logger.plot_rl_performance()
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
    }


def _combine_eval_stats(*stats_items):
    combined = _empty_eval_stats()
    total_games = 0
    for item in stats_items:
        if not item:
            continue
        combined['wins'] += int(item.get('wins', 0) or 0)
        combined['draws'] += int(item.get('draws', 0) or 0)
        combined['losses'] += int(item.get('losses', 0) or 0)
        combined['unresolved'] += int(item.get('unresolved', 0) or 0)
        total_games += int(item.get('num_games', 0) or 0)

    combined['num_games'] = int(total_games)
    if total_games > 0:
        combined['score_rate'] = float((combined['wins'] + 0.5 * combined['draws']) / total_games)
        combined['win_rate'] = float(combined['wins'] / total_games)
        combined['draw_rate'] = float(combined['draws'] / total_games)
        combined['loss_rate'] = float(combined['losses'] / total_games)
    return combined


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
    every = max(1, int(rl_cfg.get('anchor_eval_every', 1)))
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

def play_games_parallel_mcts(
    model,
    config,
    device,
    num_games,
    replay_buffer=None,
    best_model_state=None,
    recent_snapshot_pool=None,
    adaptive_scheduler_state=None,
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
            print("Warning: self_play_device=cuda but no GPU available. Falling back to CPU.")
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

    opponent_assignments, _opponent_mix_games, opponent_debug = _build_selfplay_opponent_assignments(
        rl_cfg,
        worker_specs,
        best_model_state=best_model_state,
        recent_snapshot_pool=recent_snapshot_pool,
        adaptive_scheduler_state=adaptive_scheduler_state,
    )
    if _opponent_mix_games:
        mix_parts = [
            f"{label}={count}"
            for label, count in sorted(_opponent_mix_games.items(), key=lambda item: str(item[0]))
        ]
        weight_parts = [
            f"{label}={float(weight):.1%}"
            for label, weight in sorted(
                dict((opponent_debug or {}).get("source_weights", {}) or {}).items(),
                key=lambda item: str(item[0]),
            )
        ]
        suffix = f" ({', '.join(weight_parts)})" if weight_parts else ""
        print(f"Planned opponent mix: {', '.join(mix_parts)}{suffix}")
    model_fallbacks = dict((opponent_debug or {}).get("model_fallbacks", {}) or {})
    if model_fallbacks:
        fallback_parts = [
            f"{label}->{count}"
            for label, count in sorted(model_fallbacks.items(), key=lambda item: str(item[0]))
        ]
        print(
            "Warning: opponent scheduler used current-model fallback for missing opponent models: "
            + ", ".join(fallback_parts)
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
    
    global _SELFPLAY_CONFIG_PRINTED
    if not _SELFPLAY_CONFIG_PRINTED:
        _SELFPLAY_CONFIG_PRINTED = True
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        wpg = rl_cfg.get('self_play_workers_per_gpu', 1) if device_type == 'cuda' else '-'
        threads = rl_cfg.get('self_play_torch_threads', 1)
        batch_info = f"on  (max {max_batch_games} gier/worker)" if use_batch_selfplay else "off"
        dispatch_chunk_info = (
            str(dynamic_dispatch_chunk_games)
            if dynamic_dispatch_enabled
            else "-"
        )
        max_parallel_games = sum(
            min(max_batch_games if use_batch_selfplay else 1, games)
            for _, games in worker_specs
        )
        w_games = f"{base_games}" + (f"  (+1 dla {remainder} workerĂłw)" if remainder else "")
        print()
        if opponent_debug.get("selected_recent_pool"):
            selected_recent_parts = []
            for entry in opponent_debug.get("selected_recent_pool", []):
                selected_recent_parts.append(
                    f"{entry.get('label')}("
                    f"score={float(entry.get('avg_score') or 0.0):.1%}, "
                    f"draw={float(entry.get('avg_draw_rate') or 0.0):.1%}, "
                    f"sel={float(entry.get('selection_score') or 0.0):.2f})"
                )
            print("Selected recent pool: " + ", ".join(selected_recent_parts))

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
            f"|  Chunk dispatch          |  {dispatch_chunk_info:<25} |",
            f"|  Gier rownolegle max     |  {max_parallel_games:<25} |",
            f"|  Symulacje MCTS          |  {rl_cfg['mcts_simulations']:<25} |",
            f"|  Rozmiar batcha MCTS     |  {rl_cfg.get('mcts_batch_size', 32):<25} |",
            f"|  Reuse drzewa            |  {str(rl_cfg.get('mcts_reuse_tree', True)):<25} |",
            "+--------------------------+-------------------------+",
        ]
        _print_console_block(ascii_lines)
        print()
    
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
    queue_search_samples = 0
    queue_search_simulations_used_samples = []
    queue_adaptive_stopped_early = 0
    queue_adaptive_stop_reasons = {}
    queue_wait_total_s = 0.0
    queue_wait_events = 0
    queue_profile_stats = {}
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

                def _dispatch_next_chunk(rank):
                    nonlocal next_game_offset
                    if next_game_offset >= num_games:
                        return False
                    chunk_games = min(dynamic_dispatch_chunk_games, num_games - next_game_offset)
                    chunk_plan_labels = list(
                        global_dynamic_opponent_payload.get("plan_labels", [])[next_game_offset: next_game_offset + chunk_games]
                    )
                    needed_pool_labels = {
                        str(label)
                        for label in chunk_plan_labels
                        if str(label) != "current"
                    }
                    payload = {
                        "plan_labels": chunk_plan_labels,
                        "pool_entries": [
                            entry
                            for entry in list(global_dynamic_opponent_payload.get("pool_entries", []) or [])
                            if str((entry or {}).get("label") or "current") in needed_pool_labels
                        ],
                    }
                    result_file, progress_file = selfplay_pool.dispatch_task(
                        rank=int(rank),
                        task_id=task_id,
                        model_state_path=model_state_path,
                        model_state=model_state_cpu,
                        temperature=rl_cfg.get('mcts_temperature'),
                        num_games=chunk_games,
                        opponent_payload=payload,
                        stream_results_to_queue=use_queue_transport,
                    )
                    active_workers[int(rank)] = {
                        "chunk_games": int(chunk_games),
                        "progress_file": progress_file,
                    }
                    current_progress_files[int(rank)] = progress_file
                    result_files.append(result_file)
                    progress_files.append(progress_file)
                    next_game_offset += chunk_games
                    return True

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
                                queue_search_samples += int(chunk_stats.get('search_samples', 0))
                                queue_search_simulations_used_samples.extend(list(chunk_stats.get('search_simulations_used_samples', []) or []))
                                queue_adaptive_stopped_early += int(chunk_stats.get('adaptive_stopped_early', 0))
                                for reason, count in dict(chunk_stats.get('adaptive_stop_reasons', {}) or {}).items():
                                    reason = str(reason)
                                    queue_adaptive_stop_reasons[reason] = (
                                        int(queue_adaptive_stop_reasons.get(reason, 0)) + int(count)
                                    )
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
    total_search_samples = queue_search_samples if use_queue_transport else 0
    total_search_simulations_used_samples = list(queue_search_simulations_used_samples) if use_queue_transport else []
    total_adaptive_stopped_early = queue_adaptive_stopped_early if use_queue_transport else 0
    total_adaptive_stop_reasons = dict(queue_adaptive_stop_reasons) if use_queue_transport else {}
    total_profile_stats = dict(queue_profile_stats) if use_queue_transport else {}
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
                            total_search_samples += int((stats or {}).get('search_samples', 0))
                            total_search_simulations_used_samples.extend(list((stats or {}).get('search_simulations_used_samples', []) or []))
                            total_adaptive_stopped_early += int((stats or {}).get('adaptive_stopped_early', 0))
                            for reason, count in dict((stats or {}).get('adaptive_stop_reasons', {}) or {}).items():
                                reason = str(reason)
                                total_adaptive_stop_reasons[reason] = (
                                    int(total_adaptive_stop_reasons.get(reason, 0)) + int(count)
                                )
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
    
    positions_per_sec = total_positions / total_time if total_time > 0 else 0
    avg_length = np.mean(game_lengths) if game_lengths else 0
    filtered_positions = int(total_curriculum_dropped_positions + total_cap_dropped_positions)
    def _format_plies(value):
        return f"{value:.1f} plies (~{value / 2.0:.1f} full moves)"
    
    print("MCTS Self-play completed:")
    print(f"   Positions: {total_positions}")
    print(f"   Games: {len(game_lengths)}")
    if opponent_source_games:
        mix_parts = [
            f"{label}={count}"
            for label, count in sorted(opponent_source_games.items(), key=lambda item: item[0])
        ]
        print(f"   Opponent mix: {', '.join(mix_parts)}")
    if opponent_source_results:
        score_parts = []
        for label, stats in sorted(opponent_source_results.items(), key=lambda item: item[0]):
            score_rate = _safe_score_rate(
                (stats or {}).get('wins', 0),
                (stats or {}).get('draws', 0),
                (stats or {}).get('losses', 0),
            )
            if score_rate is None:
                continue
            score_parts.append(f"{label}={score_rate:.1%}")
        if score_parts:
            print(f"   Opponent score: {', '.join(score_parts)}")
    total_generated_positions = total_positions + total_dropped_positions + filtered_positions
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
                f"{_format_plies(total_completed_length_sum / completed_games)}"
            )
        if total_truncated_games > 0:
            print(
                "   Avg truncated game length: "
                f"{_format_plies(total_truncated_length_sum / total_truncated_games)}"
            )
        if total_decisive_games > 0:
            print(
                "   Avg decisive game length: "
                f"{_format_plies(total_decisive_length_sum / total_decisive_games)}"
            )
    if filtered_positions > 0:
        print(
            "   Replay filtering drops: "
            f"curriculum={int(total_curriculum_dropped_positions)}, "
            f"cap={int(total_cap_dropped_positions)}"
        )
    if total_syzygy_probe_positions > 0:
        syzygy_hit_rate = 100.0 * float(total_syzygy_probe_hits) / float(max(1, total_syzygy_probe_positions))
        print(
            "   Syzygy probes: "
            f"{int(total_syzygy_probe_hits)}/{int(total_syzygy_probe_positions)} "
            f"({syzygy_hit_rate:.1f}% hits)"
        )
    if total_syzygy_ended_games > 0:
        print(f"   Syzygy-ended games: {int(total_syzygy_ended_games)}")
    if total_adjudicated_games > 0:
        print(f"   Adjudicated decisive games: {int(total_adjudicated_games)}")
    if total_resigned_games > 0:
        print(f"   Resigned games: {int(total_resigned_games)}")
    print(f"   Dropped positions (truncated): {total_dropped_positions}")
    print(f"   Self-play time: {selfplay_time:.1f}s")
    print(f"   Data collection: {collection_time:.3f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Speed: {positions_per_sec:.1f} positions/s")
    print(f"   Avg game length: {_format_plies(avg_length)}")

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
        'search_simulations_used_avg': float(total_search_simulations_used_sum) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'search_simulations_used_p10': float(np.percentile(np.asarray(total_search_simulations_used_samples, dtype=np.float32), 10)) if total_search_simulations_used_samples else 0.0,
        'search_samples': int(total_search_samples),
        'adaptive_stopped_early': int(total_adaptive_stopped_early),
        'adaptive_stop_rate': float(total_adaptive_stopped_early) / float(total_search_samples) if total_search_samples > 0 else 0.0,
        'adaptive_stop_reasons': dict(total_adaptive_stop_reasons),
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
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    debug_cfg = config.get('debug', {}) or {}
    profile_training_enabled = bool(debug_cfg.get('profile_training', False))
    log_gpu_memory_enabled = bool(debug_cfg.get('log_gpu_memory', False))

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
        print(
            "Replay buffer capacity: "
            f"{computed_size} (dynamic bootstrap: {bootstrap_positions} positions/iter x multiplier {replay_multiplier}, "
            f"min_size={replay_buffer_min_size}, round_to={replay_capacity_round_to})"
        )
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
    if profile_training_enabled or log_gpu_memory_enabled:
        print(
            "RL debug timing: "
            f"profile_training={profile_training_enabled}, "
            f"log_gpu_memory={log_gpu_memory_enabled}"
        )
    
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
            print("Warning: bfloat16 not supported")
            use_bfloat16 = False
    
    best_model_il_path = base_dir / config['paths']['best_model_il']
    best_model_il_swa_path = best_model_il_path.parent / "best_model_il_swa.pt"
    rl_init_checkpoint_path = (
        best_model_il_swa_path if best_model_il_swa_path.exists() else best_model_il_path
    )
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
        default_new_checkpoint=rl_init_checkpoint_path,
    )

    # Initialize logger after startup selection.
    logger = TrainingLogger(
        logs_dir,
        experiment_name=build_rl_experiment_name(config),
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
        default_new_checkpoint=rl_init_checkpoint_path,
    )
    start_mode = startup_state.get("start_mode", "new")
    selected_checkpoint_label = startup_state.get("selected_checkpoint_label")
    start_iteration = int(startup_state.get("start_iteration", 0) or 0)
    resumed_best_win_rate = float(startup_state.get("best_win_rate", 0.0) or 0.0)
    selected_compatibility_ratio = startup_state.get("selected_compatibility_ratio")
    transfer_match_ratio = startup_state.get("transfer_match_ratio")
    new_init_mode = startup_state.get("new_init_mode", startup_plan.get("new_init_mode", "default"))
    selected_entry = startup_plan.get("selected_entry") or {}

    if start_mode == "new":
        if new_init_mode == "select" and selected_checkpoint_label:
            source_label = Path(selected_checkpoint_label).name
        elif rl_init_checkpoint_path.exists():
            source_label = f"{rl_init_checkpoint_path.name} (init)"
        else:
            source_label = "new (scratch)"
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
            anchor_model_available = True
            print(f"Anchor eval enabled vs IL best: {rl_init_checkpoint_path.name}")
        except Exception as exc:
            anchor_model = None
            print(f"Anchor eval disabled: failed to load IL best ({exc})")
    anchor_eval_unlocked = bool(
        anchor_model_available
        and anchor_model is not None
        and not _models_have_identical_state(best_model, anchor_model)
    )
    if anchor_model_available and not anchor_eval_unlocked:
        print("Anchor eval locked until the first promoted best has at least one true win vs current best.")

    rl_cfg = config['reinforcement_learning']
    recent_snapshot_keep = max(0, int(rl_cfg.get('self_play_recent_snapshots_to_keep', 4)))
    recent_selfplay_snapshots = deque(maxlen=recent_snapshot_keep) if recent_snapshot_keep > 0 else deque(maxlen=0)
    adaptive_opponent_scheduler_state = {
        'score_history': {},
        'bucket_score_history': {},
        'score_history_exact': {},
        'draw_history_exact': {},
        'games_history_exact': {},
        'bucket_draw_history': {},
    }
    if recent_snapshot_keep > 0:
        recent_selfplay_snapshots.append({
            'label': 'recent_init',
            'iteration': 0,
            'state': _snapshot_model_state_cpu(model, share_memory=True),
        })

    
    # Best files are updated only when evaluation confirms model improvement.
    replay_fp16 = config['reinforcement_learning'].get('replay_fp16', False)
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
    replay_buffer = ReplayBuffer(
        config['reinforcement_learning']['replay_buffer_size'],
        max_policy_targets=replay_max_policy_targets,
        use_fp16=replay_fp16,
        decisive_sampling_fraction=float(
            config['reinforcement_learning'].get('replay_decisive_sampling_fraction', 0.0)
        ),
        decisive_value_epsilon=float(
            config['reinforcement_learning'].get('replay_decisive_value_epsilon', 0.05)
        ),
        hard_negative_sampling_fraction=float(
            config['reinforcement_learning'].get('replay_hard_negative_sampling_fraction', 0.0)
        ),
        hard_negative_min_importance=float(
            config['reinforcement_learning'].get('replay_hard_negative_min_importance', 0.0)
        ),
        recent_sampling_fraction=float(
            config['reinforcement_learning'].get('replay_recent_sampling_fraction', 0.0)
        ),
        recent_window_fraction=float(
            config['reinforcement_learning'].get('replay_recent_window_fraction', 0.25)
        ),
        quality_sampling_fraction=float(
            config['reinforcement_learning'].get('replay_quality_sampling_fraction', 0.0)
        ),
        quality_min_importance=float(
            config['reinforcement_learning'].get('replay_quality_min_importance', 0.0)
        ),
        quality_value_bonus=float(
            config['reinforcement_learning'].get('replay_quality_value_bonus', 0.0)
        ),
        resize_preserve_decisive_fraction=float(
            config['reinforcement_learning'].get('replay_resize_preserve_decisive_fraction', 0.0)
        ),
        resize_preserve_decisive_min_count=int(
            config['reinforcement_learning'].get('replay_resize_preserve_decisive_min_count', 0)
        ),
    )
    replay_capacity_round_to = max(1, int(rl_cfg.get('replay_buffer_capacity_round_to', 256)))
    replay_buffer_min_size = max(1, int(rl_cfg.get('replay_buffer_min_size', replay_buffer.max_size)))
    replay_buffer_ema_alpha = max(
        0.0,
        min(1.0, float(rl_cfg.get('replay_buffer_ema_alpha', 0.25))),
    )
    replay_positions_ema = None
    
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
    adaptive_temp_controller = AdaptiveTemperatureController(config['reinforcement_learning'])
    
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

    if use_lr_schedule:
        print(
            f"LR schedule: warmup={warmup_iters} iters -> "
            f"cosine decay (base_lr={lr_base:.6g}, min_lr_ratio={min_lr_ratio})"
        )

    print("\n=== Starting RL training with PROPER MCTS ===")
    print("OPTIMIZATIONS:")
    print(f"   - MCTS self-play (AlphaZero approach)")
    print(f"   - MCTS simulations: {config['reinforcement_learning']['mcts_simulations']}")
    print(f"   - Tree reuse: {config['reinforcement_learning'].get('mcts_reuse_tree', True)}")
    print(f"   - Batch MCTS: {config['reinforcement_learning'].get('mcts_batch_size', 32)}")
    print(
        "   - Dynamic c_puct: "
        f"{config['reinforcement_learning'].get('mcts_dynamic_c_puct', True)} "
        f"(init={float(config['reinforcement_learning'].get('mcts_c_puct_init', config['reinforcement_learning'].get('mcts_c_puct', 1.5))):.2f}, "
        f"base={int(config['reinforcement_learning'].get('mcts_c_puct_base', 19652))}, "
        f"max={config['reinforcement_learning'].get('mcts_c_puct_max', None)})"
    )
    print(
        "   - FPU: "
        f"{config['reinforcement_learning'].get('mcts_use_fpu', True)} "
        f"(reduction={float(config['reinforcement_learning'].get('mcts_fpu_reduction', 0.30)):.2f}, "
        f"absolute={config['reinforcement_learning'].get('mcts_fpu_absolute', None)})"
    )
    print(f"   - Persistent self-play workers: {config['reinforcement_learning'].get('persistent_self_play_workers', True)}")
    print(f"   - Stream self-play to replay: {config['reinforcement_learning'].get('self_play_stream_to_replay', True)}")
    if bool(config['reinforcement_learning'].get('self_play_opponent_pool_enabled', False)):
        print(
            "   - Opponent pool: "
            f"current={float(config['reinforcement_learning'].get('self_play_opponent_current_fraction', 0.4)):.0%}, "
            f"best={float(config['reinforcement_learning'].get('self_play_opponent_best_fraction', 0.3)):.0%}, "
            f"recent={float(config['reinforcement_learning'].get('self_play_opponent_recent_fraction', 0.3)):.0%}"
        )
        print(
            "   - Recent frozen snapshots: "
            f"{int(config['reinforcement_learning'].get('self_play_recent_snapshots_to_keep', 4))}"
        )
    if bool(config['reinforcement_learning'].get('self_play_opponent_pool_enabled', False)) and bool(
        config['reinforcement_learning'].get('self_play_opponent_adaptive_enabled', False)
    ):
        print(
            "   - Adaptive opponent scheduler: "
            f"on (target={float(config['reinforcement_learning'].get('self_play_opponent_adaptive_target_score', 0.50)):.0%}, "
            f"band=+/-{float(config['reinforcement_learning'].get('self_play_opponent_adaptive_band', 0.15)):.0%}, "
            f"current_floor={float(config['reinforcement_learning'].get('self_play_opponent_current_min_fraction', 0.50)):.0%}, "
            f"best_floor={float(config['reinforcement_learning'].get('self_play_opponent_best_min_fraction', 0.0)):.0%}, "
            f"recent_cap={float(config['reinforcement_learning'].get('self_play_opponent_recent_max_fraction', 1.0)):.0%})"
        )
    if bool(config['reinforcement_learning'].get('self_play_resignation_enabled', False)):
        print(
            "   - Resignation: "
            f"enabled (threshold={float(config['reinforcement_learning'].get('self_play_resignation_threshold', 0.92)):.2f}, "
            f"patience={int(config['reinforcement_learning'].get('self_play_resignation_patience', 3))}, "
            f"disable_fraction={float(config['reinforcement_learning'].get('self_play_resignation_disable_fraction', 0.10)):.0%})"
        )
    print(f"   - Temperature Schedule: {use_temp_schedule}")
    print(f"   - Policy Accuracy & Value MAE tracking")
    print(f"   - Replay buffer capacity: {config['reinforcement_learning']['replay_buffer_size']:,} positions")
    print(f"   - Replay policy target cap: {replay_max_policy_targets} moves/position")
    print(
        f"   - Replay buffer dynamic sizing: "
        f"bootstrap={int(config['reinforcement_learning'].get('replay_buffer_bootstrap_positions_per_iteration_resolved', 0)):,}, "
        f"multiplier={replay_multiplier:.2f}, "
        f"ema_alpha={replay_buffer_ema_alpha:.2f}, "
        f"min_size={replay_buffer_min_size:,}"
    )
    if start_iteration >= total_iterations:
        print(
            f"Resume start iteration ({start_iteration + 1}) exceeds configured total "
            f"({total_iterations}). Nothing to train."
        )
        logger.plot()
        logger.plot_rl_performance()
        return

    base_mcts_temperature_threshold = int(rl_cfg.get('mcts_temperature_threshold', 16))
    base_mcts_dirichlet_weight = float(rl_cfg.get('mcts_dirichlet_weight', 0.25))
    adaptive_dirichlet_enabled = bool(rl_cfg.get('adaptive_dirichlet_enabled', True))
    adaptive_dirichlet_min_weight = float(
        rl_cfg.get('adaptive_dirichlet_min_weight', max(0.0, base_mcts_dirichlet_weight * 0.30))
    )
    adaptive_dirichlet_draw_scale = float(
        rl_cfg.get('adaptive_dirichlet_draw_scale', 0.75)
    )
    adaptive_dirichlet_value_guard_scale = float(
        rl_cfg.get('adaptive_dirichlet_value_guard_scale', 0.85)
    )
    adaptive_dirichlet_eval_scale = float(
        rl_cfg.get('adaptive_dirichlet_eval_scale', 0.85)
    )
    adaptive_draw_target = float(rl_cfg.get('adaptive_temperature_draw_target', 0.30))
    adaptive_draw_band = max(0.01, float(rl_cfg.get('adaptive_temperature_draw_band', 0.05)))
    score_rate_threshold = float(rl_cfg.get('score_rate_threshold', 0.55))
    true_win_rate_threshold = float(rl_cfg.get('true_win_rate_threshold', 0.0))
    staged_eval_enabled = bool(rl_cfg.get('eval_staged_enabled', False))
    eval_stage1_games = max(1, int(rl_cfg.get('eval_stage1_games', rl_cfg.get('eval_games', 50))))
    eval_stage2_games = max(0, int(rl_cfg.get('eval_stage2_games', 0)))
    eval_stage2_gate_score_rate = float(rl_cfg.get('eval_stage2_gate_score_rate', score_rate_threshold))
    eval_stage2_gate_true_win_rate = float(rl_cfg.get('eval_stage2_gate_true_win_rate', true_win_rate_threshold))
    no_mcts_eval_enabled = bool(rl_cfg.get('eval_no_mcts_enabled', True))
    no_mcts_eval_every = max(1, int(rl_cfg.get('eval_no_mcts_every', 1)))
    no_mcts_eval_games = max(1, int(rl_cfg.get('eval_no_mcts_games', 30)))
    early_stop_enabled = bool(rl_cfg.get('early_stop_enabled', True))
    early_stop_patience = max(1, int(rl_cfg.get('early_stop_patience', 5)))
    early_stop_min_score_improvement = float(rl_cfg.get('early_stop_min_score_improvement', 0.01))
    early_stop_min_true_win_improvement = float(rl_cfg.get('early_stop_min_true_win_improvement', 0.005))
    prev_completed_draw_rate = None
    prev_decisive_rate = None
    prev_eval_score_rate_for_temp = None
    best_eval_score_seen = None
    best_eval_true_win_seen = None
    no_improvement_eval_streak = 0
    value_guard_recent_mae = deque(
        maxlen=max(2, int(rl_cfg.get('value_guard_mae_window', 3)))
    )
    value_guard_poor_eval_streak = 0
    last_eval_score_rate_for_guard = None

    training_interrupted = False
    interrupted_stage = None
    last_logged_iteration = start_iteration if start_iteration > 0 else None

    try:
        for iteration in range(start_iteration, total_iterations):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{total_iterations}")
            print('='*70)
            iteration_profile_printed = False
            iteration_stage_times = {}
            iteration_start_time = time.perf_counter()
            iteration_stage_start = iteration_start_time

            if device.type == 'cuda' and torch.cuda.is_available() and (profile_training_enabled or log_gpu_memory_enabled):
                torch.cuda.reset_peak_memory_stats(device)

            def _profile_sync():
                if profile_training_enabled and device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            def _finish_stage(stage_name):
                nonlocal iteration_stage_start
                if not profile_training_enabled:
                    return
                _profile_sync()
                now = time.perf_counter()
                stage_key = str(stage_name)
                iteration_stage_times[stage_key] = float(iteration_stage_times.get(stage_key, 0.0)) + (now - iteration_stage_start)
                iteration_stage_start = now

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
                group['lr'] = current_lr
            
            # Get current temperature
            if use_temp_schedule:
                base_current_temp = temp_schedule.get_temperature(iteration)
            else:
                base_current_temp = config['reinforcement_learning']['mcts_temperature']

            current_temp, current_temp_threshold, temp_debug = adaptive_temp_controller.compute(
                base_temp=base_current_temp,
                base_threshold=base_mcts_temperature_threshold,
                prev_draw_rate=prev_completed_draw_rate,
                prev_decisive_rate=prev_decisive_rate,
                last_eval_score_rate=last_eval_score_rate_for_guard,
                previous_eval_score_rate=prev_eval_score_rate_for_temp,
                value_guard_streak=value_guard_poor_eval_streak,
            )
            current_dirichlet_weight = base_mcts_dirichlet_weight
            if adaptive_dirichlet_enabled:
                if prev_completed_draw_rate is not None:
                    draw_excess = max(0.0, float(prev_completed_draw_rate) - adaptive_draw_target)
                    if draw_excess > 0.0:
                        excess_ratio = min(1.0, draw_excess / adaptive_draw_band)
                        target_scale = 1.0 - (1.0 - adaptive_dirichlet_draw_scale) * excess_ratio
                        current_dirichlet_weight *= max(0.0, target_scale)
                if value_guard_poor_eval_streak > 0:
                    current_dirichlet_weight *= adaptive_dirichlet_value_guard_scale ** int(value_guard_poor_eval_streak)
                if (
                    last_eval_score_rate_for_guard is not None
                    and prev_eval_score_rate_for_temp is not None
                    and float(last_eval_score_rate_for_guard) <= float(prev_eval_score_rate_for_temp) + 0.005
                ):
                    current_dirichlet_weight *= adaptive_dirichlet_eval_scale
                current_dirichlet_weight = max(
                    adaptive_dirichlet_min_weight,
                    min(base_mcts_dirichlet_weight, float(current_dirichlet_weight)),
                )
            print(f"Temperature: {current_temp:.2f}")
            if temp_debug.get("enabled", False):
                print(
                    "   Adaptive temp: "
                    f"base={base_current_temp:.2f}, "
                    f"adjust={float(temp_debug.get('adjustment', 0.0)):+.3f}, "
                    f"threshold={int(current_temp_threshold)}, "
                    f"reason={temp_debug.get('reason', 'stable')}"
                )
            if adaptive_dirichlet_enabled and base_mcts_dirichlet_weight > 0.0:
                print(
                    "   Adaptive dirichlet: "
                    f"base={base_mcts_dirichlet_weight:.3f}, current={current_dirichlet_weight:.3f}"
                )
            config['reinforcement_learning']['mcts_temperature'] = current_temp
            config['reinforcement_learning']['mcts_dirichlet_weight'] = current_dirichlet_weight
            config['reinforcement_learning']['mcts_temperature_threshold'] = current_temp_threshold
            
            if use_lr_schedule:
                print(f"LR: {current_lr:.2e}")

            rl_cfg['current_iteration'] = int(iteration + 1)

            current_value_loss_weight = _compute_value_loss_weight(iteration)
            if bool(rl_cfg.get('value_guard_enabled', True)):
                guard_patience = max(1, int(rl_cfg.get('value_guard_patience', 2)))
                guard_scale = max(0.1, min(1.0, float(rl_cfg.get('value_guard_scale', 0.85))))
                guard_min_weight = max(0.05, float(rl_cfg.get('value_guard_min_weight', 0.40)))
                if len(value_guard_recent_mae) >= value_guard_recent_mae.maxlen:
                    if value_guard_poor_eval_streak >= guard_patience:
                        current_value_loss_weight = max(
                            guard_min_weight,
                            current_value_loss_weight * (guard_scale ** value_guard_poor_eval_streak),
                        )
            print(f"Value loss weight: {current_value_loss_weight:.3f}")

            _finish_stage('setup')

            # Self-play with MCTS
            model.eval()
            best_model_state_for_selfplay = None
            if bool(rl_cfg.get('self_play_opponent_pool_enabled', False)):
                best_model_state_for_selfplay = _snapshot_model_state_cpu(best_model, share_memory=True)
            positions, positions_added, avg_game_length, positions_per_sec, selfplay_time, collection_time, selfplay_stats = \
                play_games_parallel_mcts(
                    model,
                    config,
                    device,
                    config['reinforcement_learning']['games_per_iteration'],
                    replay_buffer=replay_buffer,
                    best_model_state=best_model_state_for_selfplay,
                    recent_snapshot_pool=list(recent_selfplay_snapshots),
                    adaptive_scheduler_state=adaptive_opponent_scheduler_state,
                )
            _finish_stage('selfplay')
            if profile_training_enabled:
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
                    print(
                        "Replay buffer resized: "
                        f"{replay_buffer.max_size:,} "
                        f"(ema_positions_per_iter={float(replay_positions_ema):.1f}, multiplier={replay_multiplier:.2f})"
                    )
             
            print(f"Replay buffer: {len(replay_buffer)}/{replay_buffer.max_size} positions (+{positions_added})")
            print(
                f"Self-play stats: draw_rate={float((selfplay_stats or {}).get('completed_draw_rate', 0.0)):.2%}, "
                f"avg_value={float((selfplay_stats or {}).get('avg_game_value', 0.0)):.3f}, "
                f"value_std={float((selfplay_stats or {}).get('value_std', 0.0)):.3f}"
            )
            print(
                f"MCTS sims: avg={float((selfplay_stats or {}).get('search_simulations_used_avg', 0.0)):.1f}, "
                f"p10={float((selfplay_stats or {}).get('search_simulations_used_p10', 0.0)):.1f}, "
                f"early={100.0 * float((selfplay_stats or {}).get('adaptive_stop_rate', 0.0)):.1f}%, "
                f"samples={int((selfplay_stats or {}).get('search_samples', 0))}"
            )
            adaptive_stop_reasons = dict((selfplay_stats or {}).get('adaptive_stop_reasons', {}) or {})
            if adaptive_stop_reasons:
                reason_text = ", ".join(
                    f"{reason}={count}"
                    for reason, count in sorted(adaptive_stop_reasons.items())
                )
                print(f"MCTS adaptive stops: {reason_text}")
            print(
                f"Replay shaping: curriculum_drop={int((selfplay_stats or {}).get('curriculum_dropped_positions', 0))}, "
                f"cap_drop={int((selfplay_stats or {}).get('cap_dropped_positions', 0))}"
            )
            print(
                f"Endings: decisive_rate={float((selfplay_stats or {}).get('decisive_rate', 0.0)):.2%}, "
                f"auto_draw_rate={float((selfplay_stats or {}).get('auto_draw_rate', 0.0)):.2%}, "
                f"truncated_rate={float((selfplay_stats or {}).get('truncated_rate', 0.0)):.2%}"
            )
            selfplay_profile = dict((selfplay_stats or {}).get('profile', {}) or {})
            if profile_training_enabled and selfplay_profile:
                print_selfplay_profiler(selfplay_profile, selfplay_time)
            logger.log_rl_performance(
                iteration + 1,
                positions_per_sec=positions_per_sec,
                selfplay_time=selfplay_time,
                data_collection_time=collection_time,
                avg_game_length=avg_game_length,
                profile=selfplay_profile,
            )
            adaptive_opponent_scheduler_state, observed_scores = _update_adaptive_opponent_scheduler(
                rl_cfg,
                adaptive_opponent_scheduler_state,
                (selfplay_stats or {}).get('opponent_results', {}),
                iteration + 1,
            )
            if observed_scores:
                observed_parts = [
                    f"{label}={score:.1%}"
                    for label, score in sorted(observed_scores.items(), key=lambda item: item[0])
                ]
                print(f"Adaptive opponent scores: {', '.join(observed_parts)}")
            opponent_debug = dict((selfplay_stats or {}).get('opponent_debug', {}) or {})
            selected_recent_pool = list(opponent_debug.get('selected_recent_pool', []) or [])
            if selected_recent_pool:
                selected_parts = [
                    (
                        f"{str(entry.get('label', 'recent'))}:"
                        f"score={float(entry.get('avg_score') or 0.0):.1%},"
                        f"draw={float(entry.get('avg_draw_rate') or 0.0):.1%},"
                        f"sel={float(entry.get('selection_score') or 0.0):.2f}"
                    )
                    for entry in selected_recent_pool
                ]
                print(f"Recent pool selected: {', '.join(selected_parts)}")
            prev_completed_draw_rate = float((selfplay_stats or {}).get('completed_draw_rate', 0.0))
            prev_decisive_rate = float((selfplay_stats or {}).get('decisive_rate', 0.0))
            _finish_stage('replay')

            avg_policy_entropy = 0.0
            avg_value_pred_std = 0.0
            avg_target_value_std = 0.0
            
            # Training with metrics
            replay_size = len(replay_buffer)
            base_batch_size = int(config['reinforcement_learning']['batch_size'])
            if replay_size > 0 and base_batch_size > 0:
                print("Training...")
                model.train()
                total_loss = 0
                total_policy = 0
                total_value = 0
                total_policy_entropy = 0
                total_value_pred_std = 0
                total_target_value_std = 0
                
                # đź“Š Initialize metrics calculator
                metrics_calc = MetricsCalculator()
                
                full_batches, remainder_batch = divmod(replay_size, base_batch_size)
                batch_sizes = [base_batch_size] * full_batches
                if remainder_batch > 0:
                    batch_sizes.append(remainder_batch)
                if not batch_sizes:
                    batch_sizes = [min(base_batch_size, replay_size)]
                total_train_steps = config['reinforcement_learning']['train_epochs_per_iteration'] * len(batch_sizes)
                
                for batch_idx in tqdm(
                    range(total_train_steps),
                    desc="Training",
                ):
                    current_batch_size = batch_sizes[batch_idx % len(batch_sizes)]
                    batch = replay_buffer.sample(current_batch_size)
                    loss, policy_loss, value_loss, policy_entropy, value_pred_std, target_value_std = train_on_batch_rl(
                        model,
                        optimizer,
                        batch,
                        config,
                        device,
                        scaler,
                        metrics_calc,
                        value_weight_override=current_value_loss_weight,
                    )
                    
                    total_loss += loss
                    total_policy += policy_loss
                    total_value += value_loss
                    total_policy_entropy += policy_entropy
                    total_value_pred_std += value_pred_std
                    total_target_value_std += target_value_std
                avg_loss = total_loss / total_train_steps
                avg_policy = total_policy / total_train_steps
                avg_value = total_value / total_train_steps
                avg_policy_entropy = total_policy_entropy / total_train_steps
                avg_value_pred_std = total_value_pred_std / total_train_steps
                avg_target_value_std = total_target_value_std / total_train_steps
                
                # đź“Š Compute metrics
                train_metrics = metrics_calc.compute()
                
                print(f"Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
                print(f"Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                      f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                      f"MAE: {train_metrics['value_mae']:.4f}")
                print(
                    f"Entropy: {avg_policy_entropy:.4f}, "
                    f"Pred value std: {avg_value_pred_std:.4f}, "
                    f"Target value std: {avg_target_value_std:.4f}"
                )
                current_train_mae = float(train_metrics.get('value_mae', 0.0) or 0.0)
                if current_train_mae > 0.0:
                    value_guard_recent_mae.append(current_train_mae)
            else:
                avg_loss = avg_policy = avg_value = 0
                train_metrics = {}
            _finish_stage('train')
             
            # Evaluation
            score_rate = None
            true_win_rate = None
            eval_wins = eval_draws = eval_losses = eval_unresolved = None
            anchor_score_rate = None
            anchor_true_win_rate = None
            anchor_wins = anchor_draws = anchor_losses = None
            estimated_elo = None
            no_mcts_score_rate = None
            no_mcts_true_win_rate = None
            no_mcts_draw_rate = None
            no_mcts_loss_rate = None
            no_mcts_wins = no_mcts_draws = no_mcts_losses = no_mcts_unresolved = None
            if no_mcts_eval_enabled and ((iteration + 1) % no_mcts_eval_every == 0):
                print(f"Evaluating vs best without MCTS ({no_mcts_eval_games} games)...")
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
                print(
                    f"No-MCTS eval W/D/L: {no_mcts_wins}/{no_mcts_draws}/{no_mcts_losses} "
                    f"(score: {no_mcts_score_rate:.2%}, true win rate: {no_mcts_true_win_rate:.2%}, "
                    f"unresolved: {no_mcts_unresolved})"
                )
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
                print("Evaluating vs best...")
                model.eval()
                if device.type == 'cuda' and torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                if staged_eval_enabled:
                    print(f"Stage 1 eval ({eval_stage1_games} games)...")
                    eval_stage1_stats = evaluate_models(
                        model,
                        best_model,
                        config,
                        device,
                        eval_stage1_games,
                        game_index_offset=0,
                    )
                    stage1_score_rate = float((eval_stage1_stats or {}).get('score_rate', 0.0))
                    stage1_true_win_rate = float((eval_stage1_stats or {}).get('win_rate', 0.0))
                    print(
                        f"Stage 1: score={stage1_score_rate:.2%}, "
                        f"true_win_rate={stage1_true_win_rate:.2%}"
                    )
                    if (
                        eval_stage2_games > 0
                        and stage1_score_rate >= eval_stage2_gate_score_rate
                        and stage1_true_win_rate >= eval_stage2_gate_true_win_rate
                    ):
                        print(f"Stage 2 eval (+{eval_stage2_games} games)...")
                        eval_stage2_stats = evaluate_models(
                            model,
                            best_model,
                            config,
                            device,
                            eval_stage2_games,
                            game_index_offset=eval_stage1_games,
                        )
                        eval_stats = _combine_eval_stats(eval_stage1_stats, eval_stage2_stats)
                    else:
                        print("Stage 2 skipped: stage-1 edge not strong enough.")
                        eval_stats = eval_stage1_stats
                else:
                    eval_stats = evaluate_models(
                        model, best_model, config, device,
                        config['reinforcement_learning']['eval_games']
                    )
                score_rate = float((eval_stats or {}).get('score_rate', 0.0))
                true_win_rate = float((eval_stats or {}).get('win_rate', 0.0))
                eval_wins = int((eval_stats or {}).get('wins', 0))
                eval_draws = int((eval_stats or {}).get('draws', 0))
                eval_losses = int((eval_stats or {}).get('losses', 0))
                eval_unresolved = int((eval_stats or {}).get('unresolved', 0))
                print(f"Score rate vs best: {score_rate:.2%}")
                print(
                    f"Eval vs best W/D/L: {eval_wins}/{eval_draws}/{eval_losses} "
                    f"(true win rate: {true_win_rate:.2%}, unresolved draws at cap: {eval_unresolved})"
                )
                if bool(rl_cfg.get('value_guard_enabled', True)):
                    mae_tol = max(0.0, float(rl_cfg.get('value_guard_mae_increase_tolerance', 0.01)))
                    min_score_gain = float(rl_cfg.get('value_guard_min_score_gain', 0.005))
                    mae_worsening = False
                    if len(value_guard_recent_mae) >= value_guard_recent_mae.maxlen:
                        first_mae = float(value_guard_recent_mae[0])
                        last_mae = float(value_guard_recent_mae[-1])
                        mae_worsening = (last_mae - first_mae) >= mae_tol
                    poor_eval = False
                    if last_eval_score_rate_for_guard is not None:
                        poor_eval = float(score_rate) <= float(last_eval_score_rate_for_guard) + min_score_gain
                    if mae_worsening and poor_eval:
                        value_guard_poor_eval_streak += 1
                        print(
                            f"Value guard: streak={value_guard_poor_eval_streak} "
                            f"(MAE worsened, eval gain <= {min_score_gain:.3f})"
                        )
                    else:
                        value_guard_poor_eval_streak = 0
                    prev_eval_score_rate_for_temp = last_eval_score_rate_for_guard
                    last_eval_score_rate_for_guard = float(score_rate)
                if anchor_model_available and anchor_model is not None and _should_run_anchor_eval(iteration + 1, rl_cfg):
                    if not anchor_eval_unlocked:
                        print("Anchor eval skipped: locked until first true-win promotion vs best.")
                    elif _models_have_identical_state(best_model, anchor_model):
                        print("Anchor eval skipped: current best still matches IL-best.")
                    else:
                        print(f"Anchor eval vs IL-best ({int(rl_cfg.get('anchor_eval_games', 40))} games)...")
                        anchor_stats = evaluate_models(
                            model,
                            anchor_model,
                            config,
                            device,
                            int(rl_cfg.get('anchor_eval_games', 40)),
                            use_fixed_openings=bool(rl_cfg.get('anchor_eval_use_fixed_openings', True)),
                        )
                        anchor_score_rate = float((anchor_stats or {}).get('score_rate', 0.0))
                        anchor_true_win_rate = float((anchor_stats or {}).get('win_rate', 0.0))
                        anchor_wins = int((anchor_stats or {}).get('wins', 0))
                        anchor_draws = int((anchor_stats or {}).get('draws', 0))
                        anchor_losses = int((anchor_stats or {}).get('losses', 0))
                        print(
                            f"Anchor W/D/L: {anchor_wins}/{anchor_draws}/{anchor_losses} "
                            f"(score: {anchor_score_rate:.2%}, true win rate: {anchor_true_win_rate:.2%})"
                        )
                estimated_elo = elo_coordinator.maybe_evaluate(
                    iteration + 1,
                    score_rate=score_rate,
                    true_win_rate=true_win_rate,
                )
                
                # Log with all metrics
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    estimated_elo=estimated_elo,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    score_rate=score_rate,
                    win_rate=true_win_rate,
                    true_win_rate=true_win_rate,
                    eval_wins=eval_wins,
                    eval_draws=eval_draws,
                    eval_losses=eval_losses,
                    eval_unresolved=eval_unresolved,
                    no_mcts_score_rate=no_mcts_score_rate,
                    no_mcts_win_rate=no_mcts_true_win_rate,
                    no_mcts_true_win_rate=no_mcts_true_win_rate,
                    no_mcts_draw_rate=no_mcts_draw_rate,
                    no_mcts_loss_rate=no_mcts_loss_rate,
                    no_mcts_wins=no_mcts_wins,
                    no_mcts_draws=no_mcts_draws,
                    no_mcts_losses=no_mcts_losses,
                    no_mcts_unresolved=no_mcts_unresolved,
                    anchor_score_rate=anchor_score_rate,
                    anchor_true_win_rate=anchor_true_win_rate,
                    anchor_wins=anchor_wins,
                    anchor_draws=anchor_draws,
                    anchor_losses=anchor_losses,
                    buffer_size=len(replay_buffer),
                    avg_game_length=avg_game_length,
                    temperature=current_temp,
                    beta=None,
                    completed_draw_rate=(selfplay_stats or {}).get('completed_draw_rate', None),
                    dynamic_uniform_fraction=None,
                    priority_age_decay_lambda=None,
                    avg_sample_age=None,
                    avg_game_value=(selfplay_stats or {}).get('avg_game_value', None),
                    value_std=(selfplay_stats or {}).get('value_std', None),
                    policy_entropy=avg_policy_entropy,
                    value_pred_std=avg_value_pred_std,
                    selfplay_decisive_games=(selfplay_stats or {}).get('decisive_games', None),
                    selfplay_decisive_rate=(selfplay_stats or {}).get('decisive_rate', None),
                    selfplay_auto_draw_rate=(selfplay_stats or {}).get('auto_draw_rate', None),
                    selfplay_truncated_rate=(selfplay_stats or {}).get('truncated_rate', None),
                    selfplay_decisive_avg_length=(selfplay_stats or {}).get('decisive_avg_length', None),
                    selfplay_curriculum_dropped_positions=(selfplay_stats or {}).get('curriculum_dropped_positions', None),
                    selfplay_cap_dropped_positions=(selfplay_stats or {}).get('cap_dropped_positions', None),
                    adaptive_temp_adjustment=temp_debug.get('adjustment', None),
                    adaptive_temp_threshold=current_temp_threshold,
                )
                last_logged_iteration = iteration + 1
                if score_rate >= score_rate_threshold and true_win_rate >= true_win_rate_threshold:
                    print("New best model!")
                    logger.add_rl_best_model_marker(
                        iteration + 1,
                        f"RL best {iteration + 1}",
                    )
                    best_model.load_state_dict(model.state_dict())
                    best_win_rate_so_far = max(best_win_rate_so_far, float(score_rate))
                    if int(eval_wins or 0) > 0 and anchor_model_available:
                        anchor_eval_unlocked = True
                    
                    model_to_save = model
                    save_checkpoint(
                        model_to_save, None, iteration, avg_loss,
                        str(best_model_rl_path),
                        {
                            'win_rate': true_win_rate,
                            'score_rate': score_rate,
                            'eval_true_win_rate': true_win_rate,
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
                    print(f"Saved: {best_model_rl_path} ({size_mb:.1f} MB)")
                    if version_best_model_path != best_model_rl_path:
                        shutil.copy2(best_model_rl_path, version_best_model_path)
                        version_best_size_mb = version_best_model_path.stat().st_size / (1024 ** 2)
                        print(
                            f"Version best updated: {version_best_model_path.name} "
                            f"({version_best_size_mb:.1f} MB)"
                        )
                improved_score = (
                    best_eval_score_seen is None
                    or float(score_rate) >= float(best_eval_score_seen) + early_stop_min_score_improvement
                )
                improved_true_win = (
                    best_eval_true_win_seen is None
                    or float(true_win_rate) >= float(best_eval_true_win_seen) + early_stop_min_true_win_improvement
                )
                if improved_score:
                    best_eval_score_seen = float(score_rate)
                elif best_eval_score_seen is None:
                    best_eval_score_seen = float(score_rate)
                if improved_true_win:
                    best_eval_true_win_seen = float(true_win_rate)
                elif best_eval_true_win_seen is None:
                    best_eval_true_win_seen = float(true_win_rate)

                if improved_score or improved_true_win:
                    no_improvement_eval_streak = 0
                else:
                    no_improvement_eval_streak += 1
                    print(
                        "Early-stop monitor: "
                        f"no-improvement eval streak {no_improvement_eval_streak}/{early_stop_patience}"
                    )
                if early_stop_enabled and no_improvement_eval_streak >= early_stop_patience:
                    print(
                        "Early stopping: no meaningful improvement in eval "
                        f"for {no_improvement_eval_streak} evaluation cycle(s)."
                    )
                    logger.plot()
                    logger.plot_rl_performance()
                    _finish_stage('eval_log')
                    _emit_iteration_profile()
                    break
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
                    no_mcts_score_rate=no_mcts_score_rate,
                    no_mcts_win_rate=no_mcts_true_win_rate,
                    no_mcts_true_win_rate=no_mcts_true_win_rate,
                    no_mcts_draw_rate=no_mcts_draw_rate,
                    no_mcts_loss_rate=no_mcts_loss_rate,
                    no_mcts_wins=no_mcts_wins,
                    no_mcts_draws=no_mcts_draws,
                    no_mcts_losses=no_mcts_losses,
                    no_mcts_unresolved=no_mcts_unresolved,
                    temperature=current_temp,
                    beta=None,
                    completed_draw_rate=(selfplay_stats or {}).get('completed_draw_rate', None),
                    dynamic_uniform_fraction=None,
                    priority_age_decay_lambda=None,
                    avg_sample_age=None,
                    avg_game_value=(selfplay_stats or {}).get('avg_game_value', None),
                    value_std=(selfplay_stats or {}).get('value_std', None),
                    policy_entropy=avg_policy_entropy,
                    value_pred_std=avg_value_pred_std,
                    selfplay_decisive_games=(selfplay_stats or {}).get('decisive_games', None),
                    selfplay_decisive_rate=(selfplay_stats or {}).get('decisive_rate', None),
                    selfplay_auto_draw_rate=(selfplay_stats or {}).get('auto_draw_rate', None),
                    selfplay_truncated_rate=(selfplay_stats or {}).get('truncated_rate', None),
                    selfplay_decisive_avg_length=(selfplay_stats or {}).get('decisive_avg_length', None),
                    selfplay_curriculum_dropped_positions=(selfplay_stats or {}).get('curriculum_dropped_positions', None),
                    selfplay_cap_dropped_positions=(selfplay_stats or {}).get('cap_dropped_positions', None),
                    adaptive_temp_adjustment=temp_debug.get('adjustment', None),
                    adaptive_temp_threshold=current_temp_threshold,
                )
                last_logged_iteration = iteration + 1
            _finish_stage('eval_log')

            if recent_snapshot_keep > 0:
                recent_selfplay_snapshots.append({
                    'label': f"recent_iter_{iteration + 1}",
                    'iteration': int(iteration + 1),
                    'state': _snapshot_model_state_cpu(model, share_memory=True),
                })

            logger.plot()
            logger.plot_rl_performance()

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
            print(f"Latest checkpoint updated: {latest_checkpoint_path.name} ({latest_size_mb:.1f} MB)")
            
            _finish_stage('checkpoint')
            gc.collect()
            _finish_stage('gc')
            _emit_iteration_profile()
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
    logger.plot_rl_performance()
    if training_interrupted:
        _handle_graceful_interrupt(logger=logger, stage=interrupted_stage)
        return

    print("\n=== Training complete ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _handle_graceful_interrupt(logger=None, stage="runtime")


