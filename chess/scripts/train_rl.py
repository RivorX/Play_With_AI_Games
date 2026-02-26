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
import multiprocessing as _stdlib_mp

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

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint

# Import MCTS self-play
try:
    from src.batch_selfplay import play_games_mcts_worker
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("⚠️ MCTS self-play not available")

# Import from utils
from utils.shared.logger import TrainingLogger
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


def play_games_parallel_mcts(model, config, device, num_games):
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
        return [], 0, 0, 0, 0
    
    model_state = model.state_dict()
    
    rl_cfg = config.get('reinforcement_learning', {})
    show_worker_wait_bar = bool(rl_cfg.get('self_play_wait_progress', False))
    
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
        num_workers = min(int(num_workers), mp.cpu_count())
    else:
        num_workers = int(num_workers)
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
    
    global _SELFPLAY_CONFIG_PRINTED
    if not _SELFPLAY_CONFIG_PRINTED:
        _SELFPLAY_CONFIG_PRINTED = True
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        wpg = rl_cfg.get('self_play_workers_per_gpu', 1) if device_type == 'cuda' else '-'
        batch_info = f"on  (max {max_batch_games} gier/worker)" if use_batch_selfplay else "off"
        w_games = f"{base_games}" + (f"  (+1 dla {remainder} workerów)" if remainder else "")
        print()
        print("╔══════════════════════════════════════════════════════╗")
        print("║           KONFIGURACJA SELF-PLAY (stała)             ║")
        print("╠══════════════════════════╦═══════════════════════════╣")
        print(f"║  Urządzenie              ║  {device_type:<25} ║")
        print(f"║  GPU dostępne            ║  {gpus:<25} ║")
        print(f"║  Workerów per GPU        ║  {str(wpg):<25} ║")
        print(f"║  Workerów łącznie        ║  {len(worker_specs):<25} ║")
        print(f"║  Gier per iterację       ║  {num_games:<25} ║")
        print(f"║  Gier per worker         ║  {w_games:<25} ║")
        print(f"║  Batch self-play         ║  {batch_info:<25} ║")
        print(f"║  Symulacje MCTS          ║  {rl_cfg['mcts_simulations']:<25} ║")
        print(f"║  Rozmiar batcha MCTS     ║  {rl_cfg.get('mcts_batch_size', 32):<25} ║")
        print(f"║  Reuse drzewa            ║  {str(rl_cfg.get('mcts_reuse_tree', True)):<25} ║")
        print("╚══════════════════════════╩═══════════════════════════╝")
        print()
    
    # Create temp directory for results
    temp_dir = Path(tempfile.gettempdir()) / "chess_selfplay_mcts"
    temp_dir.mkdir(exist_ok=True)
    
    # Prepare worker tasks
    mp_ctx = mp.get_context('spawn')
    processes = []
    result_files = []
    
    interrupted = False
    try:
        for rank, games_for_worker in worker_specs:
            if device_type == 'cuda':
                device_id = rank % torch.cuda.device_count()
            else:
                device_id = 'cpu'

            result_file = temp_dir / f"worker_{rank}_mcts_results.pkl"
            result_files.append(result_file)

            p = mp_ctx.Process(
                target=play_games_mcts_worker,
                args=(rank, model_state, config, device_id, games_for_worker, str(result_file))
            )
            # If parent exits unexpectedly, daemonic workers are cleaned up automatically.
            p.daemon = True
            p.start()
            processes.append(p)

        # Build progress file paths (workers write completed-game counts here)
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
            pending = set(range(len(processes)))
            games_reported = [0] * len(processes)
            games_bar_last = 0
            while pending:
                # Update games progress from .progress files
                total_done = 0
                for idx in range(len(processes)):
                    pf = progress_files[idx]
                    try:
                        raw = pf.read_text().strip()
                        games_reported[idx] = int(raw) if raw else 0
                    except Exception:
                        pass
                    total_done += games_reported[idx]
                # Add games from already-finished workers (they may have removed progress files)
                inc = total_done - games_bar_last
                if inc > 0:
                    games_bar.update(inc)
                    games_bar_last = total_done

                for idx in list(pending):
                    p = processes[idx]
                    p.join(timeout=0.2)
                    if p.is_alive():
                        continue
                    pending.remove(idx)
                    if _is_interrupt_exit_code(p.exitcode):
                        interrupted = True
                        print("\nCtrl+C detected in self-play worker. Stopping workers...")
                        raise RLTrainingInterrupted("self-play")
                if pending:
                    time.sleep(0.05)

            # Final sync: make sure bar reaches 100 %
            games_bar.n = num_games
            games_bar.refresh()
        finally:
            games_bar.close()
            # Clean up progress files
            for pf in progress_files:
                try:
                    pf.unlink(missing_ok=True)
                except Exception:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C detected during self-play. Stopping workers...")
        raise RLTrainingInterrupted("self-play")
    except Exception:
        interrupted = True
        raise
    finally:
        if interrupted:
            _terminate_workers(processes)
    
    selfplay_time = time.time() - start_time
    
    # Collect results
    collection_start = time.time()
    
    all_positions = []
    game_lengths = []
    
    for rank, result_file in enumerate(result_files):
        if result_file.exists():
            try:
                with open(result_file, 'rb') as f:
                    while True:
                        try:
                            positions, lengths = pickle.load(f)
                        except EOFError:
                            break
                        all_positions.extend(positions)
                        game_lengths.extend(lengths)
                
                # Clean up
                result_file.unlink()
            except Exception as e:
                print(f"⚠️ Warning: Failed to load results from worker {rank}: {e}")
        else:
            print(f"⚠️ Warning: Worker {rank} result file not found")
    
    collection_time = time.time() - collection_start
    total_time = time.time() - start_time
    
    positions_per_sec = len(all_positions) / total_time if total_time > 0 else 0
    avg_length = np.mean(game_lengths) if game_lengths else 0
    
    print(f"✅ MCTS Self-play completed:")
    print(f"   Positions: {len(all_positions)}")
    print(f"   Games: {len(game_lengths)}")
    print(f"   Self-play time: {selfplay_time:.1f}s")
    print(f"   Data collection: {collection_time:.3f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Speed: {positions_per_sec:.1f} positions/s")
    print(f"   Avg game length: {avg_length:.1f} moves")
    
    return all_positions, avg_length, positions_per_sec, selfplay_time, collection_time


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
    checkpoint_every = config['reinforcement_learning'].get('checkpoint_every', 10)
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

    # Disable repeated model summary prints.
    config['model'] = {**config.get('model', {}), 'print_summary': False}
    best_model = ChessNet(config).to(device)
    best_model = best_model.to(memory_format=torch.channels_last)
    best_model.load_state_dict(model.state_dict())
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
        temp_schedule = TemperatureSchedule(
            start_temp=config['reinforcement_learning'].get('temperature_start', 1.5),
            end_temp=config['reinforcement_learning'].get('temperature_end', 0.5),
            decay_iterations=config['reinforcement_learning'].get('temperature_decay_iterations', 500)
        )
    
    # LR schedule (cosine decay + warmup, RL)
    use_lr_schedule = config['reinforcement_learning'].get('use_lr_schedule', False)
    lr_base = config['reinforcement_learning']['learning_rate']
    lr_warmup_iters = int(config['reinforcement_learning'].get('lr_warmup_iterations', 0))
    lr_max = config['reinforcement_learning'].get('lr_max', lr_base)
    lr_min = config['reinforcement_learning'].get('lr_min', lr_base * 0.1)

    def _compute_lr(iter_idx):
        if not use_lr_schedule:
            return lr_base
        if lr_warmup_iters > 0 and iter_idx < lr_warmup_iters:
            t = (iter_idx + 1) / lr_warmup_iters
            return lr_min + (lr_max - lr_min) * t
        # Cosine decay from lr_max to lr_min
        total_decay = max(1, total_iterations - lr_warmup_iters)
        t = (iter_idx - lr_warmup_iters) / total_decay
        t = max(0.0, min(1.0, t))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return lr_min + (lr_max - lr_min) * cosine

    if use_lr_schedule:
        print(f"✅ LR schedule: warmup={lr_warmup_iters} iters, "
              f"lr_max={lr_max:.6f}, lr_min={lr_min:.6f}")

    print("\n=== Starting RL training with PROPER MCTS ===")
    print("🎯 OPTIMIZATIONS:")
    print(f"   • MCTS self-play (AlphaZero approach)")
    print(f"   • MCTS simulations: {config['reinforcement_learning']['mcts_simulations']}")
    print(f"   • Tree reuse: {config['reinforcement_learning'].get('mcts_reuse_tree', True)}")
    print(f"   • Batch MCTS: {config['reinforcement_learning'].get('mcts_batch_size', 32)}")
    print(f"   • 🆕 Prioritized Replay: {use_prioritized}")
    print(f"   • 🆕 Temperature Schedule: {use_temp_schedule}")
    print(f"   • 📊 Policy Accuracy & Value MAE tracking")
    
    if start_iteration >= total_iterations:
        print(
            f"Resume start iteration ({start_iteration + 1}) exceeds configured total "
            f"({total_iterations}). Nothing to train."
        )
        logger.plot()
        return

    training_interrupted = False
    interrupted_stage = None

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
                current_temp = temp_schedule.get_temperature(iteration)
                print(f"🌡️ Temperature: {current_temp:.2f}")
                config['reinforcement_learning']['mcts_temperature'] = current_temp
            else:
                current_temp = config['reinforcement_learning']['mcts_temperature']
            
            if use_lr_schedule:
                print(f"📉 LR: {current_lr:.2e}")
            
            # Update beta for importance sampling
            if use_prioritized:
                progress = iteration / total_iterations
                replay_buffer.update_beta(progress)
                print(f"🎯 Beta (IS): {replay_buffer.beta:.3f}")
            
            # Self-play with MCTS
            model.eval()
            positions, avg_game_length, positions_per_sec, selfplay_time, collection_time = \
                play_games_parallel_mcts(
                    model,
                    config,
                    device,
                    config['reinforcement_learning']['games_per_iteration']
                )
            
            # Add to replay buffer
            for position in positions:
                replay_buffer.add(position)
            
            print(f"Replay buffer: {len(replay_buffer)} positions")
            
            # Training with metrics
            if len(replay_buffer) >= config['reinforcement_learning']['batch_size']:
                print("Training...")
                model.train()
                total_loss = 0
                total_policy = 0
                total_value = 0
                
                # 📊 Initialize metrics calculator
                metrics_calc = MetricsCalculator()
                
                num_batches = len(replay_buffer) // config['reinforcement_learning']['batch_size']
                
                for batch_idx in tqdm(
                    range(config['reinforcement_learning']['train_epochs_per_iteration'] * num_batches),
                    desc="Training",
                ):
                    # Prioritized or standard sampling
                    if use_prioritized:
                        batch_data, indices, weights = replay_buffer.sample(
                            config['reinforcement_learning']['batch_size']
                        )
                        
                        # Convert to tensors
                        boards = torch.stack([b for b, _, _ in batch_data])
                        policies = torch.stack([p for _, p, _ in batch_data])
                        values = torch.stack([v for _, _, v in batch_data])
                        batch = (boards, policies, values)
                        
                        loss, policy_loss, value_loss = train_on_batch_rl(
                            model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer, metrics_calc
                        )
                    else:
                        batch = replay_buffer.sample(config['reinforcement_learning']['batch_size'])
                        loss, policy_loss, value_loss = train_on_batch_rl(
                            model, optimizer, batch, None, None, config, device, scaler, replay_buffer, metrics_calc
                        )
                    
                    total_loss += loss
                    total_policy += policy_loss
                    total_value += value_loss
                    
                    if batch_idx % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                avg_loss = total_loss / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_policy = total_policy / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                avg_value = total_value / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
                
                # 📊 Compute metrics
                train_metrics = metrics_calc.compute()
                
                print(f"Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
                print(f"📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                      f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                      f"MAE: {train_metrics['value_mae']:.4f}")
            else:
                avg_loss = avg_policy = avg_value = 0
                train_metrics = {}
            
            # Evaluation
            win_rate = None
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
                print("Evaluating vs best...")
                model.eval()
                win_rate = evaluate_models(
                    model, best_model, config, device,
                    config['reinforcement_learning']['eval_games']
                )
                print(f"Win rate: {win_rate:.2%}")
                
                # Log with all metrics
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
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
                    beta=replay_buffer.beta if use_prioritized else None
                )
                logger.plot()
                
                if win_rate >= config['reinforcement_learning']['win_rate_threshold']:
                    print("✅ New best model!")
                    best_model.load_state_dict(model.state_dict())
                    
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
            else:
                logger.log(
                    iteration + 1,
                    train_metrics=train_metrics,
                    avg_loss=avg_loss,
                    policy_loss=avg_policy,
                    value_loss=avg_value,
                    buffer_size=len(replay_buffer),
                    avg_game_length=avg_game_length,
                    positions_per_sec=positions_per_sec,
                    selfplay_time=selfplay_time,
                    data_collection_time=collection_time,
                    temperature=current_temp,
                    beta=replay_buffer.beta if use_prioritized else None
                )
            
            # Checkpoints
            if (iteration + 1) % checkpoint_every == 0:
                if (iteration + 1) % config['reinforcement_learning']['eval_every'] != 0:
                    win_rate = evaluate_models(model, best_model, config, device,
                                            config['reinforcement_learning']['eval_games'])
                
                checkpoint_name = f"rl_iter_{iteration + 1:02d}_{model_file_tag}.pt"
                checkpoint_path = rl_dir / checkpoint_name
                
                model_to_save = model
                save_checkpoint(
                    model_to_save, None, iteration, avg_loss,
                    str(checkpoint_path),
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
                
                size_mb = checkpoint_path.stat().st_size / (1024**2)
                print(f"💾 Checkpoint: {checkpoint_path.name} ({size_mb:.1f} MB)")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except RLTrainingInterrupted as exc:
        training_interrupted = True
        interrupted_stage = str(exc) or "self-play"
    except KeyboardInterrupt:
        training_interrupted = True
        interrupted_stage = "runtime"

    logger.plot()
    if training_interrupted:
        _handle_graceful_interrupt(logger=None, stage=interrupted_stage)
        return

    print("\n=== Training complete ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _handle_graceful_interrupt(logger=None, stage="runtime")
