"""Persistent self-play worker pool utilities."""

import contextlib
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

from src.batch_selfplay import persistent_selfplay_worker

WORKER_INTERRUPT_EXIT_CODE = 130
_SELFPLAY_POOL = None

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

    def _prepare_task_files(self, rank, task_id):
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
        return result_file, progress_file

    def dispatch_task(
        self,
        rank,
        task_id,
        model_state_path,
        temperature,
        num_games,
        opponent_payload=None,
        model_state=None,
        stream_results_to_queue=False,
    ):
        result_file, progress_file = self._prepare_task_files(rank, task_id)
        self.task_queues[rank].put({
            'cmd': 'play',
            'task_id': task_id,
            'model_state': model_state,
            'model_state_path': str(model_state_path),
            'opponent_payload': opponent_payload or {},
            'num_games': int(num_games),
            'result_file_path': str(result_file),
            'mcts_temperature': temperature,
            'stream_results_to_queue': bool(stream_results_to_queue),
        })
        return result_file, progress_file

    def submit(
        self,
        task_id,
        model_state_path,
        temperature,
        worker_model_state_paths=None,
        worker_opponent_payloads=None,
        model_state=None,
        stream_results_to_queue=False,
    ):
        result_files = []
        progress_files = []
        worker_model_state_paths = worker_model_state_paths or {}
        worker_opponent_payloads = worker_opponent_payloads or {}

        for rank, games_for_worker in self.worker_specs:
            opponent_payload = worker_opponent_payloads.get(rank) or {}
            result_file, progress_file = self.dispatch_task(
                rank=rank,
                task_id=task_id,
                model_state_path=worker_model_state_paths.get(rank, model_state_path),
                temperature=temperature,
                num_games=int(games_for_worker),
                opponent_payload=opponent_payload,
                model_state=model_state,
                stream_results_to_queue=stream_results_to_queue,
            )
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


def get_or_create_selfplay_pool(config, worker_specs, device_type, temp_dir):
    global _SELFPLAY_POOL
    if _SELFPLAY_POOL is None or not _SELFPLAY_POOL.matches(worker_specs, device_type, temp_dir):
        _shutdown_selfplay_pool()
        _SELFPLAY_POOL = _PersistentSelfPlayPool(config, worker_specs, device_type, temp_dir)
        _SELFPLAY_POOL.start()
    return _SELFPLAY_POOL


def _terminate_process_tree(proc, timeout_s=0.5):
    if proc is None:
        return
    with contextlib.suppress(Exception):
        if not proc.is_alive():
            proc.join(timeout=0.0)
            return
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        proc.join(timeout=max(0.0, float(timeout_s)))
    with contextlib.suppress(Exception):
        if proc.is_alive():
            proc.kill()
    with contextlib.suppress(Exception):
        proc.join(timeout=0.2)

    if os.name == "nt":
        pid = getattr(proc, "pid", None)
        if pid:
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )


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
        _terminate_process_tree(proc, timeout_s=timeout_s)


def _is_interrupt_exit_code(exit_code):
    if exit_code is None:
        return False
    if exit_code == WORKER_INTERRUPT_EXIT_CODE:
        return True
    sigint = getattr(signal, "SIGINT", None)
    return sigint is not None and exit_code == -int(sigint)



