"""Shared central inference session wrapper.

This module wraps the RL central inference server/proxy so tools outside
self-play can reuse the same GPU batching path without copying orchestration.
"""

from __future__ import annotations

import contextlib
import copy
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from src.batch_selfplay import _RemoteInferenceModel, central_inference_server


def snapshot_model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


class CentralInferenceSession:
    """Manage one or more RL-style central inference server processes."""

    def __init__(
        self,
        *,
        config: dict,
        device: torch.device,
        workers: int,
        model_state: dict[str, Any] | None = None,
        model_states: dict[str, dict[str, Any]] | None = None,
        options: dict | None = None,
        option_prefix: str = "eval_elo",
        model_label: str = "learner",
        rank_base: int = 730000,
    ):
        self.config = config
        self.device = device
        self.workers = max(1, int(workers))
        self.options = dict(options or {})
        self.option_prefix = str(option_prefix).rstrip("_")
        self.model_label = str(model_label or "learner")
        if model_states is not None:
            self.model_states = {
                str(label): state
                for label, state in dict(model_states).items()
                if state is not None
            }
        elif model_state is not None:
            self.model_states = {self.model_label: model_state}
        else:
            self.model_states = {}
        if not self.model_states:
            raise ValueError("CentralInferenceSession requires model_state or model_states.")
        self.rank_base = int(rank_base)

        self._session = None
        self._rank_lock = threading.Lock()
        self._thread_ranks: dict[int, int] = {}
        self._next_rank = 0

    def _key(self, suffix: str) -> str:
        return f"{self.option_prefix}_{suffix}" if self.option_prefix else suffix

    @staticmethod
    def _shared_key(suffix: str) -> str:
        suffix = str(suffix)
        prefix = "central_inference_"
        return suffix[len(prefix):] if suffix.startswith(prefix) else suffix

    def _option(self, suffix: str, default=None):
        prefixed_key = self._key(suffix)
        if prefixed_key in self.options:
            return self.options[prefixed_key]
        if suffix in self.options:
            return self.options[suffix]
        shared_key = self._shared_key(suffix)
        if shared_key in self.options:
            return self.options[shared_key]
        shared_cfg = (self.config or {}).get("central_inference", {}) or {}
        if shared_key in shared_cfg:
            return shared_cfg[shared_key]
        if suffix in shared_cfg:
            return shared_cfg[suffix]
        return default

    def _float_option(self, suffix: str, default: float) -> float:
        raw_value = self._option(suffix, default)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    def resolve_server_count(self) -> int:
        # A session is bound to one concrete CUDA device. Never allow an
        # explicit override to create competing model-server processes there.
        if self.device.type == "cuda":
            return 1

        raw_value = self._option("central_inference_servers", "auto")
        if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
            target = max(1, int(self._option("central_inference_auto_workers_per_server", 10) or 10))
            min_servers = max(1, int(self._option("central_inference_auto_min_servers", 1) or 1))
            server_count = int(np.ceil(self.workers / float(target)))
            max_default = max(server_count, min_servers)
            max_servers = max(1, int(self._option("central_inference_auto_max_servers", max_default) or max_default))
            return max(1, min(max_servers, max(min_servers, server_count)))
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    def _server_config(self) -> dict:
        server_config = copy.deepcopy(self.config)
        central_cfg = server_config.setdefault("central_inference", {})
        central_cfg["max_batch_size"] = int(
            self._option(
                "central_inference_max_batch_size",
                central_cfg.get("max_batch_size", 256),
            )
            or 256
        )
        central_cfg["flush_ms"] = float(
            self._option(
                "central_inference_flush_ms",
                central_cfg.get("flush_ms", 2.0),
            )
            or 2.0
        )
        central_cfg["cache_enabled"] = bool(
            self._option("central_inference_cache_enabled", False)
        )
        central_cfg["use_compile"] = bool(
            self._option("central_inference_use_compile", False)
        )
        warmup_batches = self._option("central_inference_compile_warmup_batches", None)
        if warmup_batches is not None:
            central_cfg["compile_warmup_batches"] = warmup_batches
        central_cfg["transport_dtype"] = str(
            self._option("central_inference_transport_dtype", "float16") or "float16"
        )
        central_cfg["cudnn_benchmark"] = bool(
            self._option("central_inference_cudnn_benchmark", False)
        )
        central_cfg["sync_timing"] = bool(
            self._option("central_inference_sync_timing", False)
        )
        return server_config

    def start(self):
        if self._session is not None:
            return self

        server_count = self.resolve_server_count()
        mp_ctx = mp.get_context("spawn")
        device_id = 0
        if self.device.type == "cuda":
            try:
                device_id = int(self.device.index if self.device.index is not None else torch.cuda.current_device())
            except Exception:
                device_id = 0

        server_config = self._server_config()
        request_queues = []
        control_queues = []
        response_queues_by_rank = {}
        rank_to_server = {}
        processes = []

        for server_idx in range(server_count):
            request_queue = mp_ctx.Queue()
            control_queue = mp_ctx.Queue()
            response_queues = {}
            for rank in range(self.workers):
                if rank % server_count != server_idx:
                    continue
                response_queue = mp_ctx.Queue()
                response_queues[rank] = response_queue
                response_queues_by_rank[rank] = response_queue
                rank_to_server[rank] = server_idx

            process = mp_ctx.Process(
                target=central_inference_server,
                args=(server_config, device_id, request_queue, response_queues, control_queue, self.rank_base + server_idx),
            )
            process.daemon = True
            process.start()
            request_queues.append(request_queue)
            control_queues.append(control_queue)
            processes.append(process)

        task_id = f"central_load_{os.getpid()}_{int(time.time() * 1000)}"
        model_payload = [
            {"label": label, "state": state}
            for label, state in self.model_states.items()
        ]
        for request_queue in request_queues:
            request_queue.put(
                {
                    "cmd": "load_models",
                    "task_id": task_id,
                    "clear": True,
                    "models": model_payload,
                }
            )

        timeout_s = float(self._option("central_inference_load_timeout_s", 180.0) or 180.0)
        deadline = time.time() + max(5.0, timeout_s)
        load_messages = []
        for control_queue in control_queues:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Central inference model load timed out.")
                try:
                    message = control_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    continue
                if message and message.get("type") == "models_loaded" and message.get("task_id") == task_id:
                    load_messages.append(dict(message))
                    break

        self._thread_ranks = {}
        self._next_rank = 0
        self._session = {
            "server_count": server_count,
            "request_queues": request_queues,
            "control_queues": control_queues,
            "response_queues_by_rank": response_queues_by_rank,
            "rank_to_server": rank_to_server,
            "processes": processes,
            "server_config": server_config,
            "load_messages": load_messages,
        }
        return self

    def close(self):
        session = self._session
        self._session = None
        self._thread_ranks = {}
        self._next_rank = 0
        if not session:
            return

        for request_queue in session.get("request_queues", []) or []:
            with contextlib.suppress(Exception):
                request_queue.put({"cmd": "stop"})
            with contextlib.suppress(Exception):
                request_queue.cancel_join_thread()
        for process in session.get("processes", []) or []:
            with contextlib.suppress(Exception):
                process.join(timeout=3.0)
            if getattr(process, "is_alive", lambda: False)():
                with contextlib.suppress(Exception):
                    process.terminate()
                with contextlib.suppress(Exception):
                    process.join(timeout=1.0)
                if getattr(process, "is_alive", lambda: False)():
                    with contextlib.suppress(Exception):
                        process.kill()
                    with contextlib.suppress(Exception):
                        process.join(timeout=0.5)
            with contextlib.suppress(Exception):
                process.close()

        for queue_obj in list(session.get("request_queues", []) or []) + list(session.get("control_queues", []) or []):
            with contextlib.suppress(Exception):
                queue_obj.cancel_join_thread()
            with contextlib.suppress(Exception):
                queue_obj.close()
        for queue_obj in (session.get("response_queues_by_rank", {}) or {}).values():
            with contextlib.suppress(Exception):
                queue_obj.cancel_join_thread()
            with contextlib.suppress(Exception):
                queue_obj.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _rank_for_current_thread(self) -> int:
        if self._session is None:
            return -1
        thread_id = threading.get_ident()
        with self._rank_lock:
            rank = self._thread_ranks.get(thread_id)
            if rank is not None:
                return rank
            rank = int(self._next_rank % self.workers)
            self._next_rank += 1
            self._thread_ranks[thread_id] = rank
            return rank

    def remote_model_for_current_thread(self, model_label: str | None = None):
        if self._session is None:
            return None
        rank = self._rank_for_current_thread()
        if rank < 0:
            return None
        label = str(model_label or self.model_label)
        if label not in self.model_states:
            raise KeyError(f"Central inference model label is not loaded: {label}")
        server_idx = int((self._session.get("rank_to_server", {}) or {}).get(rank, 0))
        return _RemoteInferenceModel(
            label,
            self._session["request_queues"][server_idx],
            self._session["response_queues_by_rank"][rank],
            rank,
            timeout_s=self._float_option("central_inference_timeout_s", 0.0),
            stall_warning_s=self._float_option("central_inference_stall_warning_s", 60.0),
            debug_enabled=bool(self._option("central_inference_debug", False)),
            transport_dtype=str(self._option("central_inference_transport_dtype", "float16") or "float16"),
        )

    def describe(self) -> str:
        session = self._session
        if not session:
            return "not started"
        server_config = session.get("server_config", {}) or {}
        central_cfg = server_config.get("central_inference", {}) or {}
        load_messages = list(session.get("load_messages", []) or [])
        load_times = [
            float(message.get("load_s", 0.0) or 0.0)
            for message in load_messages
            if message.get("load_s") is not None
        ]
        model_summary = next(
            (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
            "",
        )
        pid_summary = ",".join(
            str(int(message.get("pid")))
            for message in load_messages
            if message.get("pid") is not None
        )
        load_summary = ""
        if model_summary:
            load_summary += f", {model_summary}"
        if load_times:
            load_summary += f", load={max(load_times):.2f}s"
        if pid_summary:
            load_summary += f", pids={pid_summary}"
        return (
            f"servers={int(session.get('server_count', 0) or 0)}, "
            f"workers={self.workers}, "
            f"target_workers/server={self._option('central_inference_auto_workers_per_server', 10)}, "
            f"max_batch={central_cfg.get('max_batch_size')}, "
            f"flush={central_cfg.get('flush_ms')}ms"
            f"{load_summary}"
        )
