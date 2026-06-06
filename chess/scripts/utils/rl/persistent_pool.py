"""Persistent self-play worker pool utilities."""

import contextlib
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

from src.batch_selfplay import persistent_selfplay_worker, central_inference_server

WORKER_INTERRUPT_EXIT_CODE = 130
_SELFPLAY_POOL = None


def _resolve_central_inference_server_count(config, worker_specs, device_type):
    rl_cfg = config.get('reinforcement_learning', {})
    central_enabled = bool(
        rl_cfg.get('self_play_central_inference_enabled', False)
        and device_type == 'cuda'
        and torch.cuda.is_available()
    )
    if not central_enabled:
        return 0

    raw_value = rl_cfg.get('self_play_central_inference_servers', 'auto')
    if str(raw_value).strip().lower() not in {'auto', 'automatic'}:
        return max(1, int(raw_value or 1))

    worker_count = max(1, len(list(worker_specs)))
    target_workers_per_server = max(
        4,
        int(rl_cfg.get('self_play_central_inference_auto_workers_per_server', 10) or 10),
    )
    by_workers = max(1, (worker_count + target_workers_per_server - 1) // target_workers_per_server)

    min_auto = max(1, int(rl_cfg.get('self_play_central_inference_auto_min_servers', 1) or 1))
    max_auto = max(1, int(rl_cfg.get('self_play_central_inference_auto_max_servers', 4) or 4))
    max_auto = max(min_auto, max_auto)
    by_vram = max_auto
    try:
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / float(1024 ** 3)
        if total_gib < 10.0:
            by_vram = 1
        elif total_gib < 14.0:
            by_vram = min(by_vram, 2)
        elif total_gib < 24.0:
            by_vram = min(by_vram, 3)
    except Exception:
        by_vram = min(by_vram, 2)

    desired = max(by_workers, min_auto)
    return max(1, min(desired, by_vram, max_auto))


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
        rl_cfg = config.get('reinforcement_learning', {})
        self.central_inference_enabled = bool(
            rl_cfg.get('self_play_central_inference_enabled', False)
            and device_type == 'cuda'
            and torch.cuda.is_available()
        )
        self.central_inference_server_count = _resolve_central_inference_server_count(
            config,
            self.worker_specs,
            device_type,
        )
        self.inference_request_queues = [
            self.mp_ctx.Queue() for _ in range(self.central_inference_server_count)
        ]
        self.inference_control_queues = [
            self.mp_ctx.Queue() for _ in range(self.central_inference_server_count)
        ]
        self.inference_request_queue = self.inference_request_queues[0] if self.inference_request_queues else None
        self.inference_control_queue = self.inference_control_queues[0] if self.inference_control_queues else None
        self.inference_response_receivers = {}
        self.inference_response_senders = {}
        self.inference_processes = []
        self.inference_process = None
        self._central_task_id = None
        self._central_loaded_labels = set()
        self.started = False

    def _worker_runtime_args(self, rank):
        rank = int(rank)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        inference_server_idx = (
            rank % self.central_inference_server_count
            if self.central_inference_enabled and self.central_inference_server_count > 0
            else 0
        )
        device_id = (
            'cpu'
            if self.central_inference_enabled
            else (rank % gpu_count if self.device_type == 'cuda' and gpu_count > 0 else 'cpu')
        )
        inference_request_queue = (
            self.inference_request_queues[inference_server_idx]
            if self.central_inference_enabled
            else None
        )
        inference_response_queue = self.inference_response_receivers.get(rank)
        return device_id, inference_request_queue, inference_response_queue

    def _spawn_worker(self, rank, task_queue=None):
        rank = int(rank)
        if task_queue is None:
            task_queue = self.mp_ctx.Queue()
        device_id, inference_request_queue, inference_response_queue = self._worker_runtime_args(rank)
        process = self.mp_ctx.Process(
            target=persistent_selfplay_worker,
            args=(
                rank,
                self.config,
                device_id,
                task_queue,
                self.result_queue,
                inference_request_queue,
                inference_response_queue,
            ),
        )
        process.daemon = True
        process.start()
        self.task_queues[rank] = task_queue
        self.processes[rank] = process
        return process

    def _drain_worker_responses(self, rank):
        response_queue = self.inference_response_receivers.get(int(rank))
        if response_queue is None:
            return 0
        drained = 0
        while True:
            try:
                if hasattr(response_queue, "poll") and hasattr(response_queue, "recv"):
                    if not response_queue.poll(0.0):
                        break
                    response_queue.recv()
                else:
                    response_queue.get_nowait()
                drained += 1
            except Exception:
                break
        return drained

    def restart_worker(self, rank):
        rank = int(rank)
        old_process = self.processes.get(rank)
        if old_process is not None:
            _terminate_process_tree(old_process, timeout_s=0.5)
        old_queue = self.task_queues.get(rank)
        if old_queue is not None:
            try:
                old_queue.close()
            except Exception:
                pass
        self._drain_worker_responses(rank)
        return self._spawn_worker(rank)

    def matches(self, worker_specs, device_type, temp_dir):
        rl_cfg = self.config.get('reinforcement_learning', {})
        wanted_central = bool(
            rl_cfg.get('self_play_central_inference_enabled', False)
            and device_type == 'cuda'
            and torch.cuda.is_available()
        )
        wanted_servers = _resolve_central_inference_server_count(self.config, worker_specs, device_type)
        return (
            self.worker_specs == list(worker_specs)
            and self.device_type == device_type
            and self.temp_dir == Path(temp_dir)
            and self.central_inference_enabled == wanted_central
            and self.central_inference_server_count == wanted_servers
        )

    def start(self):
        if self.started:
            return

        if self.central_inference_enabled:
            for rank, _ in self.worker_specs:
                recv_conn, send_conn = self.mp_ctx.Pipe(duplex=False)
                self.inference_response_receivers[int(rank)] = recv_conn
                self.inference_response_senders[int(rank)] = send_conn
            for server_idx in range(self.central_inference_server_count):
                inference_process = self.mp_ctx.Process(
                    target=central_inference_server,
                    args=(
                        self.config,
                        0,
                        self.inference_request_queues[server_idx],
                        self.inference_response_senders,
                        self.inference_control_queues[server_idx],
                        9000 + (int(os.getpid()) % 100000) * 10 + int(server_idx),
                    ),
                )
                inference_process.daemon = True
                inference_process.start()
                self.inference_processes.append(inference_process)
            self.inference_process = self.inference_processes[0] if self.inference_processes else None

        for rank, _ in self.worker_specs:
            self._spawn_worker(rank)

        self.started = True

    def _prepare_central_inference_models(self, task_id, model_state, model_state_path, opponent_payload):
        if not self.central_inference_enabled or not self.inference_request_queues:
            return
        task_id = str(task_id)
        clear = task_id != self._central_task_id
        if clear:
            self._central_task_id = task_id
            self._central_loaded_labels = set()

        models_to_load = []
        if "learner" not in self._central_loaded_labels:
            models_to_load.append({
                "label": "learner",
                "state": model_state,
                "state_path": str(model_state_path) if model_state is None and model_state_path is not None else None,
            })
            self._central_loaded_labels.add("learner")

        for entry in list((opponent_payload or {}).get("pool_entries", []) or []):
            label = str((entry or {}).get("label") or "current")
            if label == "current" or label in self._central_loaded_labels:
                continue
            state = (entry or {}).get("state")
            if state is None:
                continue
            models_to_load.append({"label": label, "state": state, "state_path": None})
            self._central_loaded_labels.add(label)

        if models_to_load or clear:
            for request_queue in self.inference_request_queues:
                request_queue.put({
                    "cmd": "load_models",
                    "task_id": task_id,
                    "clear": bool(clear),
                    "models": models_to_load,
                })
            timeout_s = float(self.config.get('reinforcement_learning', {}).get('self_play_central_inference_load_timeout_s', 300.0))
            import time
            end_time = time.time() + max(1.0, timeout_s)
            pending_servers = set(range(len(self.inference_control_queues)))
            load_messages = []
            while pending_servers:
                if time.time() >= end_time:
                    raise TimeoutError(
                        f"Central inference did not acknowledge model load for task {task_id} "
                        f"from servers {sorted(pending_servers)}."
                    )
                for server_idx in list(pending_servers):
                    remaining = max(0.01, end_time - time.time())
                    try:
                        message = self.inference_control_queues[server_idx].get(timeout=min(remaining, 0.25))
                    except Exception:
                        continue
                    if message.get("type") == "models_loaded" and str(message.get("task_id")) == task_id:
                        load_messages.append(dict(message))
                        pending_servers.discard(server_idx)
            if load_messages:
                load_times = [
                    float(message.get("load_s", 0.0) or 0.0)
                    for message in load_messages
                    if message.get("load_s") is not None
                ]
                model_summary = next(
                    (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
                    "models ready",
                )
                pid_summary = ",".join(
                    str(int(message.get("pid")))
                    for message in load_messages
                    if message.get("pid") is not None
                )
                print(
                    "Central inference: "
                    f"servers={len(load_messages)}, {model_summary}"
                    f"{f', load={max(load_times):.2f}s' if load_times else ''}"
                    f"{f', pids={pid_summary}' if pid_summary else ''}.",
                    flush=True,
                )

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

    def _worker_opponent_payload(self, opponent_payload):
        if not self.central_inference_enabled:
            return opponent_payload or {}
        payload = opponent_payload or {}
        return {
            "label": payload.get("label", "current"),
            "plan_labels": list(payload.get("plan_labels", []) or []),
            "pool_entries": [
                {"label": str((entry or {}).get("label") or "current")}
                for entry in list(payload.get("pool_entries", []) or [])
                if str((entry or {}).get("label") or "current") != "current"
            ],
        }

    def dispatch_task(
        self,
        rank,
        task_id,
        model_state_path,
        temperature,
        num_games,
        q_selection_weight=None,
        runtime_overrides=None,
        opponent_payload=None,
        model_state=None,
        stream_results_to_queue=False,
    ):
        result_file, progress_file = self._prepare_task_files(rank, task_id)
        self._prepare_central_inference_models(task_id, model_state, model_state_path, opponent_payload)
        worker_model_state = None if self.central_inference_enabled else model_state
        worker_opponent_payload = self._worker_opponent_payload(opponent_payload)
        self.task_queues[rank].put({
            'cmd': 'play',
            'task_id': task_id,
            'model_state': worker_model_state,
            'model_state_path': str(model_state_path),
            'opponent_payload': worker_opponent_payload,
            'num_games': int(num_games),
            'result_file_path': str(result_file),
            'mcts_temperature': temperature,
            'mcts_q_selection_weight': q_selection_weight,
            'rl_runtime_overrides': dict(runtime_overrides or {}),
            'stream_results_to_queue': bool(stream_results_to_queue),
        })
        return result_file, progress_file

    def submit(
        self,
        task_id,
        model_state_path,
        temperature,
        q_selection_weight=None,
        runtime_overrides=None,
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
                q_selection_weight=q_selection_weight,
                runtime_overrides=runtime_overrides,
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
        for request_queue in self.inference_request_queues:
            try:
                request_queue.put({'cmd': 'stop'})
            except Exception:
                pass

        all_processes = list(self.processes.values())
        all_processes.extend(self.inference_processes)
        _terminate_workers(all_processes, timeout_s=timeout_s)

        for task_queue in self.task_queues.values():
            try:
                task_queue.close()
            except Exception:
                pass
        try:
            self.result_queue.close()
        except Exception:
            pass
        try:
            for request_queue in self.inference_request_queues:
                request_queue.close()
        except Exception:
            pass
        try:
            for control_queue in self.inference_control_queues:
                control_queue.close()
        except Exception:
            pass
        for response_conn in list(self.inference_response_receivers.values()) + list(self.inference_response_senders.values()):
            try:
                response_conn.close()
            except Exception:
                pass

        self.task_queues.clear()
        self.processes.clear()
        self.inference_response_receivers.clear()
        self.inference_response_senders.clear()
        self.inference_request_queues = []
        self.inference_control_queues = []
        self.inference_request_queue = None
        self.inference_control_queue = None
        self.inference_processes = []
        self.inference_process = None
        self._central_task_id = None
        self._central_loaded_labels = set()
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



