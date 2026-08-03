"""Central batched NN inference shared by self-play and evaluation."""

import hashlib
import os
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.inference.shared_memory import shared_inference_array
from src.mcts.native import get_native_mcts
from src.mcts.search import (
    _configure_inductor_for_selfplay,
    _debug_nested,
    _maybe_compile_selfplay_model,
)


def _configure_selfplay_worker_runtime(config, device_id):
    rl_cfg = config.get('reinforcement_learning', {})
    self_play_threads = rl_cfg.get('self_play_torch_threads', None)

    _configure_inductor_for_selfplay(config)

    if self_play_threads is not None:
        try:
            self_play_threads = int(self_play_threads)
            if self_play_threads > 0:
                torch.set_num_threads(self_play_threads)
                try:
                    torch.set_num_interop_threads(max(1, min(4, self_play_threads)))
                except Exception:
                    pass
        except Exception:
            pass

    if device_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    return device


def _build_selfplay_worker_model(config, device):
    from src.model import create_model

    worker_config = dict(config)
    worker_config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = create_model(worker_config).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def _load_worker_model_state(model, model_state, rank):
    from src.model import normalize_state_dict_keys

    # Accept both raw state_dict and full training checkpoints.
    if isinstance(model_state, dict):
        if 'model_state_dict' in model_state and isinstance(model_state.get('model_state_dict'), dict):
            model_state = model_state['model_state_dict']
        elif 'state_dict' in model_state and isinstance(model_state.get('state_dict'), dict):
            model_state = model_state['state_dict']

    if model_state:
        model_state = {
            k: v for k, v in model_state.items()
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        }
        model_state = normalize_state_dict_keys(
            model_state,
            target_keys=set(model.state_dict().keys()),
        )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing = [
        key for key in missing
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    unexpected = [
        key for key in unexpected
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Worker {rank}: incompatible state_dict. "
            f"Missing={missing}, Unexpected={unexpected}"
        )


class RemoteInferenceClient:
    """Small model-like proxy used by MCTS workers with central GPU inference."""

    def __init__(
        self,
        model_label,
        request_queue,
        response_queue,
        worker_rank,
        timeout_s=120.0,
        stall_warning_s=60.0,
        debug_enabled=False,
        transport_dtype="float16",
        shared_buffer=None,
        shared_call_lock=None,
    ):
        self.model_label = str(model_label or "learner")
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_rank = int(worker_rank)
        timeout_s = float(timeout_s)
        self.timeout_s = None if timeout_s <= 0.0 else max(1.0, timeout_s)
        self.stall_warning_s = max(0.0, float(stall_warning_s))
        self.debug_enabled = bool(debug_enabled)
        self._request_counter = 0
        self._training = False
        self.last_server_batch_size = 0
        self.last_server_batch_target = 0
        self.last_server_batch_auto = False
        self.last_server_batch_auto_calibrated = False
        self._printed_first_request = False
        self._printed_first_response = False
        self.supports_remote_legal_gather = True
        self.last_response_compact_policy = False
        self.last_request_put_s = 0.0
        self.last_remote_wait_s = 0.0
        self.last_server_queue_wait_s = 0.0
        self.last_server_descriptor_queue_wait_s = 0.0
        self.last_server_batch_coalesce_wait_s = 0.0
        self.last_server_pipeline_wait_s = 0.0
        self.last_server_output_finalize_wait_s = 0.0
        self.last_server_output_pipeline_used = False
        self.last_server_total_s = 0.0
        self.last_server_concat_s = 0.0
        self.last_server_h2d_s = 0.0
        self.last_server_forward_s = 0.0
        self.last_server_d2h_s = 0.0
        self.last_server_send_s = 0.0
        self.last_shared_slot_wait_s = 0.0
        self.last_shared_bytes_avoided = 0
        self.last_cache_queries = 0
        self.last_cache_bypassed_positions = 0
        self.last_cache_hits = 0
        self.last_dedup_hits = 0
        self.last_cache_suspensions = 0
        self.last_cache_reactivations = 0
        self.last_nn_evaluated_positions = 0
        self.last_server_cache_lookup_s = 0.0
        self.last_server_staging_copy_s = 0.0
        self.last_gpu_batch_fill = 0.0
        self.last_transport_shared = False
        transport_dtype = str(transport_dtype or "float16").lower()
        self.transport_dtype = "float32" if transport_dtype in {"float32", "fp32"} else "float16"
        self._transport_torch_dtype = torch.float32 if self.transport_dtype == "float32" else torch.float16
        self._transport_np_dtype = np.float32 if self.transport_dtype == "float32" else np.float16
        self.shared_buffer = shared_buffer
        self._shared_arrays = None
        self._shared_call_lock = shared_call_lock or threading.Lock()
        if shared_buffer is not None:
            self._shared_arrays = {
                name: shared_inference_array(shared_buffer, name)
                for name in (
                    "boards", "legal_indices", "legal_counts",
                    "policy_logits", "value_logits",
                    "request_tokens", "response_tokens",
                )
            }

    @property
    def training(self):
        return self._training

    def eval(self):
        self._training = False
        return self

    def train(self, mode=True):
        self._training = bool(mode)
        return self

    def to(self, *args, **kwargs):
        return self

    def _receive_response(self, wait_s):
        if hasattr(self.response_queue, "poll") and hasattr(self.response_queue, "recv"):
            if not self.response_queue.poll(wait_s):
                raise queue.Empty()
            return self.response_queue.recv()
        return self.response_queue.get(timeout=wait_s)

    def _ts(self):
        return time.strftime("%H:%M:%S")

    def __call__(self, board_tensors, apply_log_softmax=False, legal_index_matrix=None, **kwargs):
        if self._shared_arrays is None:
            return self._call_impl(
                board_tensors,
                apply_log_softmax=apply_log_softmax,
                legal_index_matrix=legal_index_matrix,
                **kwargs,
            )
        wait_t0 = time.perf_counter()
        with self._shared_call_lock:
            self.last_shared_slot_wait_s = time.perf_counter() - wait_t0
            return self._call_impl(
                board_tensors,
                apply_log_softmax=apply_log_softmax,
                legal_index_matrix=legal_index_matrix,
                **kwargs,
            )

    def _call_impl(self, board_tensors, apply_log_softmax=False, legal_index_matrix=None, **kwargs):
        if apply_log_softmax:
            raise ValueError("Remote inference proxy only supports raw policy logits.")
        self._request_counter += 1
        request_id = f"{os.getpid()}_{id(self)}_{self._request_counter}"
        if isinstance(board_tensors, torch.Tensor):
            boards_source = board_tensors.detach().to('cpu').contiguous().numpy()
        else:
            boards_source = np.asarray(board_tensors)
        batch_size = int(boards_source.shape[0])
        legal_source = None
        legal_cols = 0
        if legal_index_matrix is not None:
            legal_source = np.asarray(legal_index_matrix, dtype=np.int16)
            legal_cols = int(legal_source.shape[1]) if legal_source.ndim == 2 else 0
        legal_counts = kwargs.get("legal_counts")
        if legal_counts is None:
            legal_counts_source = np.full(batch_size, legal_cols, dtype=np.int16)
        else:
            legal_counts_source = np.asarray(legal_counts, dtype=np.int16).reshape(-1)

        shared_eligible = bool(
            self._shared_arrays is not None
            and self.transport_dtype == "float16"
            and legal_source is not None
            and batch_size <= int(self.shared_buffer.get("capacity", 0))
            and legal_cols <= int(self.shared_buffer.get("max_legal_moves", 0))
            and int(boards_source.shape[1]) == int(self.shared_buffer.get("input_planes", -1))
        )
        request = {
            "cmd": "infer",
            "rank": self.worker_rank,
            "request_id": request_id,
            "model_label": self.model_label,
            # perf_counter is monotonic and system-wide on supported Python
            # platforms, so it is safe for latency measurements across the
            # worker/server process boundary.
        }
        shared_slot = None
        if shared_eligible:
            shared_slot = int((self._request_counter - 1) % int(self.shared_buffer.get("slots", 1)))
            np.copyto(
                self._shared_arrays["boards"][shared_slot, :batch_size],
                boards_source,
                casting="unsafe",
            )
            self._shared_arrays["legal_indices"][shared_slot, :batch_size, :legal_cols] = legal_source
            self._shared_arrays["legal_counts"][shared_slot, :batch_size] = legal_counts_source
            shared_token = (
                ((int(os.getpid()) & 0x7FFFFFFF) << 32)
                | (int(self._request_counter) & 0xFFFFFFFF)
            )
            self._shared_arrays["request_tokens"][shared_slot] = shared_token
            request.update({
                "shared_memory": True,
                "shared_slot": shared_slot,
                "batch_size": batch_size,
                "legal_cols": legal_cols,
                "input_bytes": int(batch_size * np.prod(boards_source.shape[1:]) * np.dtype(np.float16).itemsize),
                "shared_token": int(shared_token),
            })
            self.last_transport_shared = True
        else:
            boards_np = np.ascontiguousarray(boards_source, dtype=self._transport_np_dtype)
            request["boards"] = boards_np
            if legal_source is not None:
                request["legal_index_matrix"] = np.ascontiguousarray(legal_source, dtype=np.int16)
                request["legal_counts"] = np.ascontiguousarray(legal_counts_source, dtype=np.int16)
            self.last_transport_shared = False
        # Timestamp after all local packing/copies.  The server-side delta now
        # measures descriptor queueing/batching only, not worker preparation.
        request_started_perf = time.perf_counter()
        request["queued_at_perf"] = request_started_perf
        put_t0 = request_started_perf
        self.request_queue.put(request)
        self.last_request_put_s = time.perf_counter() - put_t0
        if self.debug_enabled and not self._printed_first_request:
            self._printed_first_request = True
            print(
                f"[{self._ts()}] Central inference: worker {self.worker_rank} sent first "
                f"{self.model_label} request ({batch_size} positions, "
                f"transport={'shared' if shared_eligible else 'queue'}).",
                flush=True,
            )
        started_at = time.time()
        deadline = None if self.timeout_s is None else started_at + self.timeout_s
        next_warning_at = (
            started_at + self.stall_warning_s
            if self.stall_warning_s > 0.0
            else None
        )
        while True:
            now = time.time()
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(f"Central inference timed out for model '{self.model_label}'.")
                wait_s = min(max(remaining, 0.01), 1.0)
            else:
                wait_s = 1.0
            if self.debug_enabled and next_warning_at is not None and now >= next_warning_at:
                waited_s = now - started_at
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} still waiting "
                    f"{waited_s:.0f}s for model '{self.model_label}'.",
                    flush=True,
                )
                next_warning_at = now + self.stall_warning_s
            try:
                response = self._receive_response(wait_s)
            except queue.Empty:
                continue
            if response.get("request_id") != request_id:
                response_request_id = str(response.get("request_id") or "")
                if not response_request_id.startswith(f"{os.getpid()}_"):
                    continue
                raise RuntimeError(
                    "Central inference response routing failed: "
                    f"worker {self.worker_rank} received response for another request."
                )
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "central inference failed"))
            self.last_remote_wait_s = time.perf_counter() - request_started_perf
            self.last_server_batch_size = int(response.get("server_batch_size", 0) or 0)
            self.last_server_batch_target = int(response.get("server_batch_target", 0) or 0)
            self.last_server_batch_auto = bool(response.get("server_batch_auto", False))
            self.last_server_batch_auto_calibrated = bool(
                response.get("server_batch_auto_calibrated", False)
            )
            self.last_response_compact_policy = bool(response.get("compact_policy", False))
            self.last_server_queue_wait_s = float(response.get("server_queue_wait_s", 0.0) or 0.0)
            self.last_server_descriptor_queue_wait_s = float(
                response.get("server_descriptor_queue_wait_s", 0.0) or 0.0
            )
            self.last_server_batch_coalesce_wait_s = float(
                response.get("server_batch_coalesce_wait_s", 0.0) or 0.0
            )
            self.last_server_pipeline_wait_s = float(
                response.get("server_pipeline_wait_s", 0.0) or 0.0
            )
            self.last_server_output_finalize_wait_s = float(
                response.get("server_output_finalize_wait_s", 0.0) or 0.0
            )
            self.last_server_output_pipeline_used = bool(
                response.get("server_output_pipeline_used", False)
            )
            self.last_server_total_s = float(response.get("server_total_time_s", 0.0) or 0.0)
            self.last_server_concat_s = float(response.get("server_concat_time_s", 0.0) or 0.0)
            self.last_server_h2d_s = float(response.get("server_h2d_time_s", 0.0) or 0.0)
            self.last_server_forward_s = float(response.get("server_forward_time_s", 0.0) or 0.0)
            self.last_server_d2h_s = float(response.get("server_d2h_time_s", 0.0) or 0.0)
            self.last_server_send_s = float(response.get("server_send_time_s", 0.0) or 0.0)
            self.last_shared_bytes_avoided = int(response.get("shared_bytes_avoided", 0) or 0)
            self.last_cache_queries = int(response.get("cache_queries", 0) or 0)
            self.last_cache_bypassed_positions = int(
                response.get("cache_bypassed_positions", 0) or 0
            )
            self.last_cache_hits = int(response.get("cache_hits", 0) or 0)
            self.last_dedup_hits = int(response.get("dedup_hits", 0) or 0)
            self.last_cache_suspensions = int(response.get("cache_suspensions", 0) or 0)
            self.last_cache_reactivations = int(response.get("cache_reactivations", 0) or 0)
            self.last_nn_evaluated_positions = int(response.get("nn_evaluated_positions", 0) or 0)
            self.last_server_cache_lookup_s = float(response.get("server_cache_lookup_s", 0.0) or 0.0)
            self.last_server_staging_copy_s = float(response.get("server_staging_copy_s", 0.0) or 0.0)
            self.last_gpu_batch_fill = float(response.get("gpu_batch_fill", 0.0) or 0.0)
            if self.debug_enabled and not self._printed_first_response:
                self._printed_first_response = True
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} received first "
                    f"{self.model_label} response (server_batch={self.last_server_batch_size}).",
                    flush=True,
                )
            if bool(response.get("shared_memory", False)):
                response_slot = int(response.get("shared_slot", shared_slot if shared_slot is not None else 0))
                expected_token = int(response.get("shared_token", 0) or 0)
                actual_token = int(self._shared_arrays["response_tokens"][response_slot])
                if expected_token <= 0 or actual_token != expected_token:
                    raise RuntimeError(
                        "Central inference shared-memory response token mismatch."
                    )
                response_batch = int(response.get("batch_size", batch_size))
                policy_cols = int(response.get("policy_cols", legal_cols))
                policy_np = self._shared_arrays[
                    "policy_logits"
                ][response_slot, :response_batch, :policy_cols].copy()
                value_np = self._shared_arrays[
                    "value_logits"
                ][response_slot, :response_batch, :3].copy()
                policy = torch.from_numpy(policy_np)
                value = torch.from_numpy(value_np)
            else:
                policy_dtype = np.float16 if self.last_response_compact_policy else np.float32
                policy = torch.from_numpy(np.asarray(response["policy_logits"], dtype=policy_dtype))
                value = torch.from_numpy(np.asarray(response["value_logits"], dtype=np.float32))
            return policy, value


def _central_inference_option(config, key, default=None):
    shared_cfg = (config or {}).get('central_inference', {}) or {}
    return shared_cfg.get(key, default)


class _AdaptiveInferenceCacheGate:
    """Limit exact-cache CPU cost without disabling it for the whole run.

    A weak active window suspends hashing temporarily.  Positions processed
    while suspended form a cheap cooldown; after it expires the cache gets a
    fresh probe window and can recover in a later game phase.
    """

    def __init__(
        self,
        enabled,
        probe_positions=8192,
        min_saved_rate=0.10,
        cooldown_positions=65536,
    ):
        self.enabled = bool(enabled)
        self.probe_positions = max(1, int(probe_positions))
        self.min_saved_rate = max(0.0, float(min_saved_rate))
        self.cooldown_positions = max(1, int(cooldown_positions))
        self.reset()

    def reset(self):
        self.active = bool(self.enabled)
        self.window_queries = 0
        self.window_saved = 0
        self.cooldown_progress = 0

    def record_active(self, queries, saved):
        """Record an active-cache group; return True when it is suspended."""
        if not self.active:
            return False
        self.window_queries += max(0, int(queries))
        self.window_saved += max(0, int(saved))
        if self.window_queries < self.probe_positions:
            return False
        saved_rate = float(self.window_saved) / float(max(1, self.window_queries))
        self.window_queries = 0
        self.window_saved = 0
        if saved_rate >= self.min_saved_rate:
            return False
        self.active = False
        self.cooldown_progress = 0
        return True

    def record_bypass(self, positions):
        """Record an unhashed group; return True when a fresh probe is armed."""
        if self.active or not self.enabled:
            return False
        self.cooldown_progress += max(0, int(positions))
        if self.cooldown_progress < self.cooldown_positions:
            return False
        self.active = True
        self.cooldown_progress = 0
        self.window_queries = 0
        self.window_saved = 0
        return True


def _prioritize_central_inference_process():
    """Keep the GPU feeder responsive when self-play saturates every CPU.

    Workers are CPU-heavy and normally occupy all logical processors. On
    Windows, ABOVE_NORMAL prevents the single GPU-owner process from waiting
    behind CPU-saturating tree-search processes without changing affinity or
    stealing a dedicated core while the server is idle.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process_handle = kernel32.GetCurrentProcess()
        above_normal_priority_class = 0x00008000
        return bool(kernel32.SetPriorityClass(process_handle, above_normal_priority_class))
    except Exception:
        return False


def _central_compile_batch_buckets(max_batch):
    """Return a small fixed shape set for compiled central inference.

    CUDAGraph records one graph per observed batch shape. Keeping at most eight
    shapes avoids graph proliferation while retaining dense buckets around the
    measured self-play operating range (roughly 250-320 positions).
    """
    max_batch = max(1, int(max_batch))
    if max_batch <= 64:
        # Elo/eval servers run much smaller groups; retaining 1/8/16 avoids
        # turning an eight-position request into a padded batch of 32.
        candidates = (1, 8, 16, 32, 48, max_batch)
    elif max_batch <= 128:
        candidates = (1, 8, 16, 32, 64, 96, max_batch)
    else:
        candidates = (
            32,
            64,
            128,
            192,
            256,
            max_batch - 96,
            max_batch - 64,
            max_batch,
        )
    return tuple(sorted({
        max(1, min(max_batch, int(batch_size)))
        for batch_size in candidates
    }))


def _central_compile_graph_batch_size(actual_batch_size, buckets):
    """Map a real batch to the smallest compiled shape that can contain it."""
    actual_batch_size = max(1, int(actual_batch_size))
    for bucket in buckets:
        if bucket >= actual_batch_size:
            return int(bucket)
    return actual_batch_size


class _AdaptiveBatchController:
    """Calibrate a stable server batch target for the current worker load.

    Candidates are measured in sustained windows using completed positions per
    wall second, including batching/queue effects.  The controller deliberately locks after one pass:
    worker count is fixed for a server lifetime, while continuous exploration
    would add latency noise to every RL iteration.
    """

    _WINDOW_BATCHES = 48

    def __init__(self, max_batch, compile_buckets, worker_count, enabled=True, clock=None):
        self.max_batch = max(1, int(max_batch))
        self.enabled = bool(enabled)
        workers = max(1, int(worker_count or 1))
        # On the production RTX 5060 Ti path, controlled end-to-end A/B showed
        # that 18 workers need the full 384 target: allowing the kernel-only
        # tuner to fall to 256 reduced played positions/s by about 8%. Scale
        # that saturation floor with worker count, while retaining a 50% floor
        # for low-parallelism runs.
        load_floor = int(np.ceil(self.max_batch * min(1.0, workers / 18.0)))
        minimum = min(
            self.max_batch,
            max(64, int(0.5 * self.max_batch), load_floor),
        )
        candidates = {
            int(batch)
            for batch in compile_buckets
            if minimum <= int(batch) <= self.max_batch
        }
        candidates.add(self.max_batch)
        candidates = sorted(candidates)
        if not self.enabled or len(candidates) <= 1:
            self._order = [self.max_batch]
        else:
            if workers <= 8:
                preferred = minimum
            elif workers <= 12:
                preferred = int(0.70 * self.max_batch)
            elif workers <= 16:
                preferred = int(0.85 * self.max_batch)
            else:
                preferred = self.max_batch
            # Start from the likely-good production point, then perform the
            # complete calibration pass without privileging its measurement.
            self._order = sorted(
                candidates,
                key=lambda batch: (abs(batch - preferred), -batch),
            )
        # Each candidate is measured once in each direction. This cancels most
        # of the opening/endgame phase bias that made a single sequential pass
        # consistently prefer whichever batch happened to run later.
        self._schedule = (
            self._order
            if len(self._order) == 1
            else self._order + list(reversed(self._order))
        )
        self._probe_index = 0
        self._window_batches = 0
        self._window_positions = 0
        self._window_started_at = None
        self._clock = clock or time.perf_counter
        self._scores = {}
        self._selected = int(self._schedule[0])
        self._calibrated = not self.enabled or len(self._order) <= 1

    @property
    def target(self):
        return int(self._selected if self._calibrated else self._schedule[self._probe_index])

    @property
    def calibrated(self):
        return bool(self._calibrated)

    def record(self, target, positions, service_s=0.0):
        if self._calibrated or int(target) != self.target:
            return
        positions = max(0, int(positions))
        if positions <= 0:
            return
        completed_at = float(self._clock())
        if self._window_started_at is None:
            # The controller is created before model warm-up. Start timing at
            # the first completed batch so compilation/load time cannot bias
            # the first candidate.
            self._window_started_at = completed_at
            return
        self._window_batches += 1
        self._window_positions += positions
        if self._window_batches < self._WINDOW_BATCHES:
            return
        measured_target = self.target
        elapsed_s = max(1e-9, completed_at - self._window_started_at)
        self._scores.setdefault(measured_target, []).append(
            float(self._window_positions) / elapsed_s
        )
        self._window_batches = 0
        self._window_positions = 0
        self._window_started_at = None
        self._probe_index += 1
        if self._probe_index >= len(self._schedule):
            mean_scores = {
                batch: float(sum(scores)) / float(len(scores))
                for batch, scores in self._scores.items()
            }
            best_score = max(mean_scores.values())
            # Prefer the smaller batch when results are effectively tied. It
            # lowers response latency without sacrificing measured throughput.
            competitive = [
                batch for batch, score in mean_scores.items()
                if score >= best_score * 0.99
            ]
            self._selected = int(min(competitive))
            self._calibrated = True


def central_inference_server(
    config,
    device_id,
    request_queue,
    response_queues,
    control_queue=None,
    server_rank=None,
    shared_buffers=None,
):
    """Own GPU inference and batch requests coming from self-play workers."""
    rl_cfg = config.get('reinforcement_learning', {})
    _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
    max_batch = max(1, int(_central_inference_option(
        config,
        'max_batch_size',
        rl_cfg.get('mcts_batch_size', 256),
    )))
    flush_ms = max(0.0, float(_central_inference_option(config, 'flush_ms', 5.0)))
    debug_enabled = bool(
        debug_root_enabled and central_debug_cfg.get(
            'verbose',
            rl_cfg.get('self_play_central_inference_debug', False),
        )
    )
    quiet_startup = True
    central_use_compile = bool(_central_inference_option(config, 'use_compile', False))
    compile_batch_buckets = (
        _central_compile_batch_buckets(max_batch)
        if central_use_compile
        else ()
    )
    auto_batch_enabled = bool(
        _central_inference_option(config, 'auto_batch_size', True)
    )
    output_pipeline_enabled = bool(
        _central_inference_option(config, 'output_pipeline_enabled', True)
    )
    sync_timing = bool(_central_inference_option(config, 'sync_timing', debug_enabled))
    transport_dtype = str(_central_inference_option(config, 'transport_dtype', 'float16') or 'float16').lower()
    transport_np_dtype = np.float32 if transport_dtype in {'float32', 'fp32'} else np.float16
    shared_buffers = {
        int(rank): spec
        for rank, spec in dict(shared_buffers or {}).items()
        if spec is not None
    }
    shared_arrays_by_rank = {
        int(rank): {
            name: shared_inference_array(spec, name)
            for name in (
                'boards', 'legal_indices', 'legal_counts',
                'policy_logits', 'value_logits',
                'request_tokens', 'response_tokens',
            )
        }
        for rank, spec in shared_buffers.items()
    }
    batch_controller = _AdaptiveBatchController(
        max_batch=max_batch,
        compile_buckets=_central_compile_batch_buckets(max_batch),
        worker_count=(
            len(shared_arrays_by_rank)
            or (len(response_queues) if response_queues is not None else 1)
        ),
        enabled=auto_batch_enabled,
    )
    cache_enabled = bool(_central_inference_option(config, 'cache_enabled', True))
    cache_max_entries = max(
        0,
        int(_central_inference_option(config, 'cache_max_entries', 50000) or 0),
    )
    cache = OrderedDict()
    native_server_kernels = get_native_mcts()
    native_hash_scratch = np.empty((max_batch, 2), dtype=np.uint64)
    # Native bulk hashing made exact-cache probing cheap enough to give the
    # initially cold cache a real warm-up window.  A short 8k probe used to
    # suspend and clear it before repeated middlegames/endgames could arrive.
    # The model-load path below is the only place that must invalidate entries.
    cache_probe_positions = 32768
    cache_min_saved_rate = 0.10
    # With a persistently cold cache this samples about 1/9 of positions.  At
    # the measured 3-7% all-miss hashing penalty that caps average overhead
    # below roughly 1%, while still revisiting later middlegame/endgame phases.
    cache_reprobe_cooldown_positions = 65536
    cache_gate = _AdaptiveInferenceCacheGate(
        cache_enabled and cache_max_entries > 0,
        probe_positions=cache_probe_positions,
        min_saved_rate=cache_min_saved_rate,
        cooldown_positions=cache_reprobe_cooldown_positions,
    )
    pinned_staging_enabled = bool(
        _central_inference_option(config, 'pinned_staging_enabled', False)
    )
    pinned_staging = {}
    gpu_input_buffers = {}
    prepare_executor = None
    central_use_bfloat16 = bool(_central_inference_option(
        config,
        'use_bfloat16',
        config.get('hardware', {}).get('use_bfloat16', False),
    ))
    central_amp_dtype = torch.bfloat16 if central_use_bfloat16 else torch.float16

    try:
        _prioritize_central_inference_process()
        device = _configure_selfplay_worker_runtime(config, device_id)
        output_copy_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        async_output_buffers = [None, None]

        def _async_output_buffer(slot, rows, policy_width):
            slot = int(slot) & 1
            current = async_output_buffers[slot]
            needs_new = (
                current is None
                or int(current['rows']) < int(rows)
                or int(current['policy_width']) < int(policy_width)
            )
            if needs_new:
                current = {
                    'rows': int(rows),
                    'policy_width': int(policy_width),
                    'gpu_policy': torch.empty(
                        (rows, policy_width), device=device, dtype=torch.float16
                    ),
                    'gpu_value': torch.empty((rows, 3), device=device, dtype=torch.float32),
                    'cpu_policy': torch.empty(
                        (rows, policy_width), dtype=torch.float16, pin_memory=True
                    ),
                    'cpu_value': torch.empty(
                        (rows, 3), dtype=torch.float32, pin_memory=True
                    ),
                    'copy_done': torch.cuda.Event(),
                    'pending': False,
                }
                async_output_buffers[slot] = current
            if current.get('pending'):
                current['copy_done'].synchronize()
                current['pending'] = False
            return current
        try:
            compile_rank = int(server_rank)
        except Exception:
            compile_rank = 900000 + int(device_id) * 100
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = bool(
                _central_inference_option(config, 'cudnn_benchmark', False)
            )
        models = {}
        compiled_base_models = {}
        compiled_wrappers = {}

        def _ts():
            return time.strftime("%H:%M:%S")

        if not quiet_startup:
            print(
                f"[{_ts()}] Central inference: server ready on {device} "
                f"(pid={os.getpid()}, compile_rank={compile_rank}, max_batch={max_batch}, flush={flush_ms:.1f}ms, "
                f"transport={transport_np_dtype.__name__}, "
                f"compile={'on' if central_use_compile else 'off'}, "
                f"cudnn.benchmark={torch.backends.cudnn.benchmark if device.type == 'cuda' else 'n/a'}).",
                flush=True,
            )

        def _warmup_central_model(model, label):
            if not central_use_compile or device.type != 'cuda':
                return None
            warmup_batches = list(compile_batch_buckets)
            if not warmup_batches:
                return None
            history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
            input_planes = 16 * (1 + history_positions)
            use_amp = bool(config.get('hardware', {}).get('use_amp', False))
            dummy_dtype = torch.float16 if use_amp and transport_np_dtype == np.float16 else torch.float32
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warming compiled model '{label}' "
                    f"for batches={warmup_batches}...",
                    flush=True,
                )
            warmup_t0 = time.perf_counter()
            with torch.inference_mode():
                for batch_size in warmup_batches:
                    dummy_input = torch.zeros(
                        batch_size,
                        input_planes,
                        8,
                        8,
                        device=device,
                        dtype=dummy_dtype,
                    ).to(memory_format=torch.channels_last)
                    with torch.autocast(device_type='cuda', enabled=use_amp, dtype=central_amp_dtype):
                        model(dummy_input, apply_log_softmax=False)
                    torch.cuda.synchronize(device)
            warmup_s = time.perf_counter() - warmup_t0
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warmup for '{label}' done "
                    f"in {warmup_s:.2f}s.",
                    flush=True,
                )
            return warmup_s

        def _load_model(label, state):
            nonlocal central_use_compile
            label = str(label or "learner")
            load_info = {
                "label": label,
                "compiled": bool(central_use_compile),
                "reused": False,
                "warmup_s": None,
            }
            if debug_enabled:
                print(f"[{_ts()}] Central inference: loading model '{label}' on {device}...", flush=True)
            if central_use_compile:
                if label in compiled_wrappers and label in compiled_base_models:
                    base_model = compiled_base_models[label]
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = compiled_wrappers[label]
                    load_info["reused"] = True
                    if debug_enabled:
                        print(f"[{_ts()}] Central inference: reused compiled model '{label}'.", flush=True)
                else:
                    base_model = _build_selfplay_worker_model(config, device)
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = _maybe_compile_selfplay_model(
                        base_model,
                        config,
                        device,
                        rank=compile_rank,
                        model_label=f"central:{label}",
                        warmup_batch_size=(
                            compile_batch_buckets[0]
                            if compile_batch_buckets
                            else 1
                        ),
                    )
                    compiled_base_models[label] = base_model
                    if model is not base_model:
                        try:
                            load_info["warmup_s"] = _warmup_central_model(model, label)
                            compiled_wrappers[label] = model
                        except Exception as exc:
                            # Never rebuild Inductor/Triton caches inside a live
                            # inference server.  A display/driver reset can make
                            # the current CUDA context temporarily unhealthy;
                            # compiling again in that context only adds memory
                            # pressure and delays recovery.  The eager module is
                            # already built, loaded and resident on the target
                            # device, so fail over to that exact model.
                            print(
                                f"WARNING: Central inference: compile warm-up failed for '{label}' "
                                f"({type(exc).__name__}: {exc}). "
                                "Using the prepared eager model; compile will be retried only "
                                "after the server process is restarted.",
                                flush=True,
                            )
                            compiled_wrappers.pop(label, None)
                            model = base_model
                            load_info["compiled"] = False
                            load_info["warmup_s"] = None
                            central_use_compile = False
                    else:
                        load_info["compiled"] = False
            else:
                model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(model, state, rank=-1)
                model.eval()
            models[label] = model
            if debug_enabled:
                print(f"[{_ts()}] Central inference: model '{label}' ready.", flush=True)
            return load_info

        def _response_queue_for(req):
            if isinstance(response_queues, dict):
                rank = int(req.get("rank", -1))
                response_queue = response_queues.get(rank)
                if response_queue is None:
                    raise RuntimeError(f"Central inference has no response channel for worker {rank}.")
                return response_queue
            return response_queues

        def _shared_request_arrays(req):
            if not bool(req.get('shared_memory', False)):
                return None
            return shared_arrays_by_rank.get(int(req.get('rank', -1)))

        def _request_boards(req):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                return np.asarray(req['boards'], dtype=transport_np_dtype)
            slot = int(req.get('shared_slot', 0))
            batch_size = int(req.get('batch_size', 0))
            return arrays['boards'][slot, :batch_size]

        def _request_legal_indices(req):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                value = req.get('legal_index_matrix')
                return None if value is None else np.asarray(value, dtype=np.int16)
            slot = int(req.get('shared_slot', 0))
            batch_size = int(req.get('batch_size', 0))
            legal_cols = int(req.get('legal_cols', 0))
            return arrays['legal_indices'][slot, :batch_size, :legal_cols]

        def _request_legal_counts(req, batch_size, legal_cols):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                value = req.get('legal_counts')
                if value is None:
                    return np.full(batch_size, legal_cols, dtype=np.int16)
                return np.asarray(value, dtype=np.int16).reshape(-1)
            slot = int(req.get('shared_slot', 0))
            return arrays['legal_counts'][slot, :batch_size]

        def _send_response(response_channel, response):
            if hasattr(response_channel, "send"):
                response_channel.send(response)
            else:
                response_channel.put(response)

        def _put_response(req, payload):
            response = {
                "request_id": req.get("request_id"),
                "rank": int(req.get("rank", -1)),
            }
            response.update(payload)
            _send_response(_response_queue_for(req), response)

        def _get_pinned(name, dtype, rows, trailing_shape):
            if not (pinned_staging_enabled and device.type == 'cuda'):
                return None
            np_dtype = np.dtype(dtype)
            torch_dtype = torch.float16 if np_dtype == np.float16 else torch.float32
            key = (str(name), np_dtype.str, tuple(trailing_shape))
            tensor = pinned_staging.get(key)
            if tensor is None or int(tensor.shape[0]) < int(rows):
                capacity = max(max_batch, int(rows))
                try:
                    tensor = torch.empty(
                        (capacity, *tuple(trailing_shape)),
                        dtype=torch_dtype,
                        pin_memory=True,
                    )
                except Exception:
                    return None
                pinned_staging[key] = tensor
            return tensor

        def _prepare_group_inputs(requests, compute_hashes=False):
            """Assemble one server batch without touching the model or cache."""
            prepare_t0 = time.perf_counter()
            staging_copy_time = 0.0
            use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
            input_np_dtype = transport_np_dtype if use_amp else np.float32
            board_batches = [np.asarray(_request_boards(req)) for req in requests]
            batch_sizes = [int(batch.shape[0]) for batch in board_batches]
            total_n = int(sum(batch_sizes))
            input_planes = int(board_batches[0].shape[1])

            board_staging = _get_pinned(
                'boards', input_np_dtype, total_n, (input_planes, 8, 8)
            )
            if board_staging is not None:
                staging_t0 = time.perf_counter()
                boards = board_staging[:total_n].numpy()
                cursor = 0
                for board_batch, batch_size in zip(board_batches, batch_sizes):
                    np.copyto(
                        boards[cursor:cursor + batch_size],
                        board_batch,
                        casting='unsafe',
                    )
                    cursor += batch_size
                staging_copy_time += time.perf_counter() - staging_t0
            else:
                converted_batches = [
                    np.asarray(batch, dtype=input_np_dtype)
                    for batch in board_batches
                ]
                boards = (
                    converted_batches[0]
                    if len(converted_batches) == 1
                    else np.concatenate(converted_batches, axis=0)
                )

            legal_index_batches = []
            legal_count_batches = []
            compact_policy = True
            max_legal_cols = 0
            for req, batch_size in zip(requests, batch_sizes):
                legal_idx = _request_legal_indices(req)
                if legal_idx is None or legal_idx.ndim != 2 or int(legal_idx.shape[0]) != batch_size:
                    compact_policy = False
                    legal_index_batches = []
                    legal_count_batches = []
                    break
                legal_cols = int(legal_idx.shape[1])
                legal_index_batches.append(legal_idx)
                legal_count_batches.append(
                    _request_legal_counts(req, batch_size, legal_cols)
                )
                max_legal_cols = max(max_legal_cols, legal_cols)

            if compact_policy and max_legal_cols > 0:
                if len(legal_index_batches) == 1:
                    # Shared-memory requests already own a contiguous legal
                    # matrix only when the request uses the full arena stride.
                    # Narrow column slices retain the arena row stride and need
                    # one compact copy for torch/native hashing.
                    legal_indices = np.ascontiguousarray(
                        legal_index_batches[0],
                        dtype=np.int16,
                    )
                    legal_counts = legal_count_batches[0][:total_n]
                elif all(
                    int(legal_idx.shape[1]) == max_legal_cols
                    for legal_idx in legal_index_batches
                ):
                    # np.concatenate performs the common equal-width case in C;
                    # the previous row-group assignment loop added Python and
                    # zero-fill overhead before every forward.
                    legal_indices = np.concatenate(legal_index_batches, axis=0)
                    legal_counts = np.concatenate(
                        [
                            counts[:batch_size]
                            for counts, batch_size in zip(
                                legal_count_batches, batch_sizes
                            )
                        ],
                        axis=0,
                    )
                else:
                    legal_indices = np.zeros((total_n, max_legal_cols), dtype=np.int16)
                    legal_counts = np.empty(total_n, dtype=np.int16)
                    cursor = 0
                    for legal_idx, counts, batch_size in zip(
                        legal_index_batches, legal_count_batches, batch_sizes
                    ):
                        cols = int(legal_idx.shape[1])
                        legal_indices[cursor:cursor + batch_size, :cols] = legal_idx
                        legal_counts[cursor:cursor + batch_size] = counts[:batch_size]
                        cursor += batch_size
            else:
                compact_policy = False
                legal_indices = None
                legal_counts = None

            native_hashes = None
            if (
                compute_hashes
                and compact_policy
                and native_server_kernels is not None
            ):
                native_hashes = np.empty((total_n, 2), dtype=np.uint64)
                native_server_kernels.hash_positions(
                    boards,
                    legal_indices,
                    legal_counts,
                    native_hashes,
                )

            return {
                'started_at': prepare_t0,
                'boards': boards,
                'board_staging': board_staging,
                'batch_sizes': batch_sizes,
                'total_n': total_n,
                'input_planes': input_planes,
                'legal_index_batches': legal_index_batches,
                'legal_indices': legal_indices,
                'legal_counts': legal_counts,
                'native_hashes': native_hashes,
                'compact_policy': compact_policy,
                'max_legal_cols': max_legal_cols,
                'concat_time': time.perf_counter() - prepare_t0,
                'staging_copy_time': staging_copy_time,
            }

        def _finalize_infer_group(ctx):
            finalize_t0 = time.perf_counter()
            async_output = ctx.get('async_output')
            if async_output is not None:
                async_output['copy_done'].synchronize()
                async_output['pending'] = False
                rows = int(ctx['eval_count'])
                width = int(ctx['max_legal_cols'])
                evaluated_policy = async_output['cpu_policy'][:rows, :width].numpy()
                evaluated_values = async_output['cpu_value'][:rows, :3].numpy()
                finalize_wait = time.perf_counter() - finalize_t0
                ctx['d2h_time'] += finalize_wait
            else:
                policy_chunks = ctx['policy_np_chunks']
                value_chunks = ctx['value_np_chunks']
                evaluated_policy = (
                    policy_chunks[0] if len(policy_chunks) == 1
                    else np.concatenate(policy_chunks, axis=0)
                ) if ctx['eval_count'] > 0 else None
                evaluated_values = (
                    value_chunks[0] if len(value_chunks) == 1
                    else np.concatenate(value_chunks, axis=0)
                ) if ctx['eval_count'] > 0 else None
                finalize_wait = 0.0

            policy_np_batch = ctx['policy_np_batch']
            value_np_batch = ctx['value_np_batch']
            if ctx['eval_count'] > 0:
                if ctx['compact_policy'] and not ctx['group_cache_active']:
                    policy_np_batch = evaluated_policy
                    value_np_batch = evaluated_values
                elif ctx['compact_policy']:
                    for eval_idx, original_idx in enumerate(ctx['miss_indices']):
                        legal_count = int(ctx['legal_counts'][original_idx])
                        policy_np_batch[original_idx, :ctx['max_legal_cols']] = evaluated_policy[eval_idx]
                        value_np_batch[original_idx] = evaluated_values[eval_idx]
                        key = ctx['cache_keys'][original_idx]
                        if key is not None:
                            cache[key] = (
                                evaluated_policy[eval_idx, :legal_count].copy(),
                                evaluated_values[eval_idx].copy(),
                            )
                            cache.move_to_end(key)
                            while len(cache) > cache_max_entries:
                                cache.popitem(last=False)
                    for idx in np.flatnonzero(ctx['dedup_hit_flags']):
                        representative = int(ctx['representative_for'][idx])
                        policy_np_batch[idx] = policy_np_batch[representative]
                        value_np_batch[idx] = value_np_batch[representative]
                else:
                    policy_np_batch = evaluated_policy
                    value_np_batch = evaluated_values

            total_n = int(ctx['total_n'])
            group_cache_active = bool(ctx['group_cache_active'])
            group_cache_queries = total_n if group_cache_active else 0
            group_cache_bypassed = total_n if ctx['compact_policy'] and not group_cache_active else 0
            group_cache_saved = (
                int(np.count_nonzero(ctx['cache_hit_flags']))
                + int(np.count_nonzero(ctx['dedup_hit_flags']))
                if group_cache_active else 0
            )
            cache_suspended = False
            cache_reactivated = False
            if group_cache_queries > 0:
                cache_suspended = cache_gate.record_active(group_cache_queries, group_cache_saved)
            elif group_cache_bypassed > 0:
                cache_reactivated = cache_gate.record_bypass(group_cache_bypassed)

            effective_forward_batch = (
                float(ctx['eval_count']) / float(ctx['forward_chunk_count'])
                if ctx['forward_chunk_count'] > 0 else 0.0
            )
            server_total_before_publish = (
                float(ctx['launch_total_time']) + finalize_wait
                if async_output is not None
                else time.perf_counter() - ctx['group_t0']
            )
            publish_send_time = 0.0
            cursor = 0
            for req_idx, (req, batch_size) in enumerate(zip(ctx['requests'], ctx['batch_sizes'])):
                req_end = cursor + batch_size
                req_legal_cols = (
                    int(ctx['legal_index_batches'][req_idx].shape[1])
                    if ctx['compact_policy'] else int(policy_np_batch.shape[1])
                )
                policy_slice = policy_np_batch[cursor:req_end, :req_legal_cols]
                value_slice = value_np_batch[cursor:req_end]
                cache_queries = batch_size if group_cache_queries > 0 else 0
                cache_bypassed_positions = batch_size if group_cache_bypassed > 0 else 0
                if group_cache_active:
                    cache_hits = int(np.count_nonzero(ctx['cache_hit_flags'][cursor:req_end]))
                    dedup_hits = int(np.count_nonzero(ctx['dedup_hit_flags'][cursor:req_end]))
                else:
                    cache_hits = dedup_hits = 0
                nn_evaluated = (
                    max(0, cache_queries - cache_hits - dedup_hits)
                    if cache_queries else batch_size
                )
                payload = {
                    'ok': True,
                    'server_batch_size': int(round(effective_forward_batch)),
                    'server_batch_target': int(ctx['batch_target']),
                    'server_batch_auto': bool(auto_batch_enabled),
                    'server_batch_auto_calibrated': bool(batch_controller.calibrated),
                    'server_request_group_positions': total_n,
                    'compact_policy': bool(ctx['compact_policy']),
                    'server_queue_wait_s': float(ctx['total_queue_waits'][req_idx]),
                    'server_descriptor_queue_wait_s': float(ctx['descriptor_queue_waits'][req_idx]),
                    'server_batch_coalesce_wait_s': float(ctx['batch_coalesce_waits'][req_idx]),
                    'server_pipeline_wait_s': float(ctx['pipeline_wait_time']),
                    'server_output_finalize_wait_s': float(finalize_wait),
                    'server_output_pipeline_used': bool(async_output is not None),
                    'server_total_time_s': float(server_total_before_publish),
                    'server_concat_time_s': float(ctx['concat_time']),
                    'server_h2d_time_s': float(ctx['h2d_time']),
                    'server_forward_time_s': float(ctx['forward_time']),
                    'server_d2h_time_s': float(ctx['d2h_time']),
                    'server_send_time_s': float(publish_send_time),
                    'server_cache_lookup_s': float(ctx['cache_lookup_time'] * batch_size / max(1, total_n)),
                    'server_staging_copy_s': float(ctx['staging_copy_time'] * batch_size / max(1, total_n)),
                    'cache_queries': int(cache_queries),
                    'cache_bypassed_positions': int(cache_bypassed_positions),
                    'cache_hits': int(cache_hits),
                    'dedup_hits': int(dedup_hits),
                    'nn_evaluated_positions': int(nn_evaluated),
                    'cache_entries': int(len(cache)),
                    'cache_active': bool(cache_gate.active),
                    'cache_suspensions': int(cache_suspended and req_idx == 0),
                    'cache_reactivations': int(cache_reactivated and req_idx == 0),
                    'gpu_batch_fill': float(effective_forward_batch / max(1, ctx['batch_target'])),
                }
                arrays = _shared_request_arrays(req)
                if arrays is not None and ctx['compact_policy']:
                    slot = int(req.get('shared_slot', 0))
                    shared_token = int(req.get('shared_token', 0) or 0)
                    arrays['policy_logits'][slot, :batch_size, :req_legal_cols] = policy_slice
                    arrays['value_logits'][slot, :batch_size, :3] = value_slice
                    arrays['response_tokens'][slot] = shared_token
                    payload.update({
                        'shared_memory': True,
                        'shared_slot': slot,
                        'shared_token': shared_token,
                        'batch_size': batch_size,
                        'policy_cols': req_legal_cols,
                        'shared_bytes_avoided': int(
                            int(req.get('input_bytes', 0) or 0)
                            + batch_size * req_legal_cols * np.dtype(np.int16).itemsize
                            + policy_slice.nbytes + value_slice.nbytes
                        ),
                    })
                else:
                    payload['policy_logits'] = policy_slice
                    payload['value_logits'] = value_slice
                    payload['shared_memory'] = False
                    payload['shared_bytes_avoided'] = 0
                cursor = req_end
                send_t0 = time.perf_counter()
                _put_response(req, payload)
                publish_send_time += time.perf_counter() - send_t0
            return {
                'total_time': float(server_total_before_publish + publish_send_time),
                'concat_time': ctx['concat_time'],
                'h2d_time': ctx['h2d_time'],
                'forward_time': ctx['forward_time'],
                'd2h_time': ctx['d2h_time'],
                'send_time': publish_send_time,
                'cache_lookup_time': ctx['cache_lookup_time'],
                'staging_copy_time': ctx['staging_copy_time'],
                'output_finalize_wait_time': finalize_wait,
                'output_pipeline_used': bool(async_output is not None),
                'batch_target': int(ctx['batch_target']),
                'compact_policy': bool(ctx['compact_policy']),
                'positions': total_n,
                'nn_evaluated_positions': int(ctx['eval_count']),
                'cache_hits': int(np.count_nonzero(ctx['cache_hit_flags'])) if group_cache_active else 0,
                'dedup_hits': int(np.count_nonzero(ctx['dedup_hit_flags'])) if group_cache_active else 0,
                'requests': int(len(ctx['requests'])),
            }

        def _infer_group(model_label, requests, prepared=None, *, defer_output=False, output_slot=0, batch_target=None):
            nonlocal central_use_compile
            prepared = prepared if prepared is not None else _prepare_group_inputs(
                requests,
                compute_hashes=bool(cache_gate.active),
            )
            group_t0 = float(prepared.get('started_at', time.perf_counter()))
            concat_time = 0.0
            staging_copy_time = 0.0
            cache_lookup_time = 0.0
            h2d_time = 0.0
            forward_time = 0.0
            d2h_time = 0.0
            model_label = str(model_label)
            model = models.get(model_label)
            if model is None:
                raise RuntimeError(f"Central inference has no model loaded for label '{model_label}'.")

            # Split pre-service latency at the server dequeue/preparation
            # boundary. descriptor_queue_wait is real IPC/server backlog;
            # batch_coalesce_wait is deliberate waiting before this batch starts
            # being assembled. With pipelined preparation, any wait after the
            # assembly began remains inside server_total_time instead.
            # Their sum is the legacy
            # server_queue_wait value kept for backwards-compatible dashboards.
            descriptor_queue_waits = []
            batch_coalesce_waits = []
            total_queue_waits = []
            for req in requests:
                queued_at_perf = req.get("queued_at_perf")
                dequeued_at_perf = float(
                    req.get("server_dequeued_at_perf", group_t0) or group_t0
                )
                if queued_at_perf is None:
                    descriptor_wait = 0.0
                else:
                    descriptor_wait = max(
                        0.0,
                        dequeued_at_perf - float(queued_at_perf),
                    )
                coalesce_wait = max(0.0, group_t0 - dequeued_at_perf)
                descriptor_queue_waits.append(descriptor_wait)
                batch_coalesce_waits.append(coalesce_wait)
                total_queue_waits.append(descriptor_wait + coalesce_wait)

            use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
            input_np_dtype = transport_np_dtype if use_amp else np.float32
            boards = prepared['boards']
            board_staging = prepared['board_staging']
            batch_sizes = prepared['batch_sizes']
            total_n = int(prepared['total_n'])
            input_planes = int(prepared['input_planes'])
            legal_index_batches = prepared['legal_index_batches']
            legal_indices = prepared['legal_indices']
            legal_counts = prepared['legal_counts']
            compact_policy = bool(prepared['compact_policy'])
            max_legal_cols = int(prepared['max_legal_cols'])
            concat_time += float(prepared['concat_time'])
            staging_copy_time += float(prepared['staging_copy_time'])
            prepare_ready_at = group_t0 + float(prepared['concat_time'])
            pipeline_wait_time = max(0.0, time.perf_counter() - prepare_ready_at)

            cache_t0 = time.perf_counter()
            group_cache_active = bool(cache_gate.active and compact_policy)
            # The adaptive cache is intentionally bypassed for most groups when
            # its measured save rate is low.  Do not allocate a second complete
            # result batch in that hot path: evaluated_policy/evaluated_values
            # can be published directly after D2H.
            policy_np_batch = (
                np.zeros((total_n, max_legal_cols), dtype=np.float16)
                if compact_policy and group_cache_active else None
            )
            value_np_batch = (
                np.zeros((total_n, 3), dtype=np.float32)
                if group_cache_active else None
            )
            # Keep a complete, cache-independent context contract.  The hot
            # path normally bypasses the exact cache, but deferred output
            # finalization must not depend on variables created only by the
            # cache branch.
            cache_hit_flags = np.zeros(total_n, dtype=np.bool_)
            dedup_hit_flags = np.zeros(total_n, dtype=np.bool_)
            cache_keys = [None] * total_n
            representative_for = np.arange(total_n, dtype=np.int32)
            miss_indices = []
            if group_cache_active:
                pending_representatives = {}
                native_hashes = prepared.get('native_hashes')
                if native_hashes is None and native_server_kernels is not None:
                    native_hashes = native_hash_scratch[:total_n]
                    native_server_kernels.hash_positions(
                        boards,
                        legal_indices,
                        legal_counts,
                        native_hashes,
                    )
                for idx in range(total_n):
                    legal_count = int(legal_counts[idx])
                    if native_hashes is not None:
                        key = (
                            model_label,
                            int(native_hashes[idx, 0]),
                            int(native_hashes[idx, 1]),
                        )
                    else:
                        digest = hashlib.blake2b(digest_size=16, person=b'chess-nn-cache')
                        digest.update(memoryview(boards[idx]).cast('B'))
                        digest.update(legal_count.to_bytes(2, 'little', signed=False))
                        if legal_count > 0:
                            digest.update(memoryview(legal_indices[idx, :legal_count]).cast('B'))
                        key = (model_label, digest.digest())
                    cache_keys[idx] = key
                    cached = cache.get(key)
                    if cached is not None:
                        cached_policy, cached_value = cached
                        policy_np_batch[idx, :len(cached_policy)] = cached_policy
                        value_np_batch[idx] = cached_value
                        cache.move_to_end(key)
                        cache_hit_flags[idx] = True
                        continue
                    representative = pending_representatives.get(key)
                    if representative is not None:
                        representative_for[idx] = int(representative)
                        dedup_hit_flags[idx] = True
                        continue
                    pending_representatives[key] = idx
                    miss_indices.append(idx)
            cache_lookup_time += time.perf_counter() - cache_t0

            eval_count = len(miss_indices) if group_cache_active else total_n
            eval_boards = None
            eval_board_tensor = None
            eval_legal_indices = None
            if eval_count > 0:
                sequential_misses = (
                    not group_cache_active
                    or (
                        eval_count == total_n
                        and all(
                            idx == position
                            for position, idx in enumerate(miss_indices)
                        )
                    )
                )
                if sequential_misses:
                    eval_boards = boards
                    eval_board_tensor = board_staging[:total_n] if board_staging is not None else None
                    eval_legal_indices = legal_indices
                else:
                    miss_staging = _get_pinned(
                        'cache_misses', input_np_dtype, eval_count, (input_planes, 8, 8)
                    )
                    staging_t0 = time.perf_counter()
                    if miss_staging is not None:
                        eval_board_tensor = miss_staging[:eval_count]
                        eval_boards = eval_board_tensor.numpy()
                        np.copyto(eval_boards, boards[miss_indices], casting='unsafe')
                    else:
                        eval_boards = np.ascontiguousarray(boards[miss_indices])
                    eval_legal_indices = (
                        np.ascontiguousarray(legal_indices[miss_indices])
                        if compact_policy else None
                    )
                    staging_copy_time += time.perf_counter() - staging_t0

            policy_np_chunks = []
            value_np_chunks = []
            forward_chunk_count = 0
            async_output = None
            for batch_start in range(0, eval_count, max_batch):
                batch_end = min(eval_count, batch_start + max_batch)
                actual_batch_size = int(batch_end - batch_start)
                h2d_t0 = time.perf_counter()
                source_tensor = (
                    eval_board_tensor[batch_start:batch_end]
                    if eval_board_tensor is not None
                    else torch.from_numpy(eval_boards[batch_start:batch_end])
                )
                if compile_batch_buckets and device.type == 'cuda':
                    graph_batch_size = _central_compile_graph_batch_size(
                        actual_batch_size,
                        compile_batch_buckets,
                    )
                    buffer_key = (
                        graph_batch_size,
                        int(source_tensor.shape[1]),
                        source_tensor.dtype,
                    )
                    tensor = gpu_input_buffers.get(buffer_key)
                    if tensor is None:
                        tensor = torch.empty(
                            (
                                graph_batch_size,
                                int(source_tensor.shape[1]),
                                8,
                                8,
                            ),
                            device=device,
                            dtype=source_tensor.dtype,
                            memory_format=torch.channels_last,
                        )
                        gpu_input_buffers[buffer_key] = tensor
                    tensor[:actual_batch_size].copy_(
                        source_tensor,
                        non_blocking=True,
                    )
                else:
                    tensor = source_tensor.to(
                        device,
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                legal_index_tensor = None
                if compact_policy:
                    legal_index_tensor = torch.from_numpy(
                        eval_legal_indices[batch_start:batch_end]
                    ).to(device, non_blocking=True).long()
                if sync_timing and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                h2d_time += time.perf_counter() - h2d_t0
                with torch.inference_mode():
                    forward_t0 = time.perf_counter()
                    try:
                        if use_amp and device.type == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=central_amp_dtype):
                                policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                        else:
                            policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    except Exception as exc:
                        eager_model = compiled_base_models.get(model_label)
                        if eager_model is None or eager_model is model:
                            raise
                        print(
                            f"WARNING: Central inference: compiled forward failed for '{model_label}' "
                            f"({type(exc).__name__}: {exc}). "
                            "Switching immediately to the prepared eager model.",
                            flush=True,
                        )
                        # Disable every compiled wrapper in this server. They
                        # share one CUDA context, so retrying another label after
                        # a driver fault is both expensive and unreliable.
                        for prepared_label, prepared_model in compiled_base_models.items():
                            prepared_model.eval()
                            models[prepared_label] = prepared_model
                        compiled_wrappers.clear()
                        central_use_compile = False
                        eager_model.eval()
                        model = eager_model
                        if use_amp and device.type == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=central_amp_dtype):
                                policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                        else:
                            policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    if int(policy_logits.shape[0]) != actual_batch_size:
                        policy_logits = policy_logits[:actual_batch_size]
                        value_logits = value_logits[:actual_batch_size]
                    if sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_time += time.perf_counter() - forward_t0
                    forward_chunk_count += 1
                    d2h_t0 = time.perf_counter()
                    if compact_policy and legal_index_tensor is not None:
                        policy_logits = torch.gather(policy_logits, 1, legal_index_tensor)
                    policy_fp16 = policy_logits.to(dtype=torch.float16)
                    use_async_output = bool(
                        defer_output
                        and output_copy_stream is not None
                        and compact_policy
                        and not group_cache_active
                        and eval_count <= max_batch
                        and batch_start == 0
                    )
                    if use_async_output:
                        policy_width = int(policy_fp16.shape[1])
                        async_output = _async_output_buffer(
                            output_slot,
                            actual_batch_size,
                            policy_width,
                        )
                        async_output['gpu_policy'][:actual_batch_size, :policy_width].copy_(
                            policy_fp16
                        )
                        async_output['gpu_value'][:actual_batch_size, :3].copy_(
                            value_logits
                        )
                        compute_done = torch.cuda.Event()
                        compute_done.record()
                        with torch.cuda.stream(output_copy_stream):
                            output_copy_stream.wait_event(compute_done)
                            async_output['cpu_policy'][:actual_batch_size, :policy_width].copy_(
                                async_output['gpu_policy'][:actual_batch_size, :policy_width],
                                non_blocking=True,
                            )
                            async_output['cpu_value'][:actual_batch_size, :3].copy_(
                                async_output['gpu_value'][:actual_batch_size, :3],
                                non_blocking=True,
                            )
                            async_output['copy_done'].record(output_copy_stream)
                        async_output['pending'] = True
                    elif (
                        device.type == 'cuda'
                        and policy_fp16.dtype == torch.float16
                        and value_logits.dtype == torch.float16
                    ):
                        policy_width = int(policy_fp16.shape[1])
                        packed_output = torch.cat((policy_fp16, value_logits), dim=1)
                        packed_np = packed_output.cpu().numpy()
                        policy_np_chunks.append(packed_np[:, :policy_width])
                        value_np_chunks.append(
                            packed_np[:, policy_width:].astype(np.float32)
                        )
                    else:
                        policy_np_chunks.append(policy_fp16.cpu().numpy())
                        value_np_chunks.append(value_logits.float().cpu().numpy())
                    if sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    d2h_time += time.perf_counter() - d2h_t0

            ctx = {
                'async_output': async_output,
                'eval_count': eval_count,
                'max_legal_cols': max_legal_cols,
                'policy_np_chunks': policy_np_chunks,
                'value_np_chunks': value_np_chunks,
                'policy_np_batch': policy_np_batch,
                'value_np_batch': value_np_batch,
                'compact_policy': compact_policy,
                'group_cache_active': group_cache_active,
                'miss_indices': miss_indices,
                'legal_counts': legal_counts,
                'cache_keys': cache_keys,
                'dedup_hit_flags': dedup_hit_flags,
                'representative_for': representative_for,
                'cache_hit_flags': cache_hit_flags,
                'total_n': total_n,
                'forward_chunk_count': forward_chunk_count,
                'requests': requests,
                'batch_sizes': batch_sizes,
                'legal_index_batches': legal_index_batches,
                'total_queue_waits': total_queue_waits,
                'descriptor_queue_waits': descriptor_queue_waits,
                'batch_coalesce_waits': batch_coalesce_waits,
                'pipeline_wait_time': pipeline_wait_time,
                'group_t0': group_t0,
                'concat_time': concat_time,
                'h2d_time': h2d_time,
                'forward_time': forward_time,
                'd2h_time': d2h_time,
                'cache_lookup_time': cache_lookup_time,
                'staging_copy_time': staging_copy_time,
                'batch_target': int(batch_target or max_batch),
                'launch_total_time': time.perf_counter() - group_t0,
            }
            if async_output is not None:
                return {'deferred_finalize': lambda ctx=ctx: _finalize_infer_group(ctx)}
            return _finalize_infer_group(ctx)

        pending = []
        pending_positions = 0
        pending_positions_by_model = {}
        pending_started_at = None
        first_infer_seen = False
        processed_requests = 0
        processed_positions = 0
        last_debug_print = time.perf_counter()
        last_batch_positions = 0
        last_batch_time_s = 0.0
        last_stage_stats = {}
        interval_infer_time = 0.0
        interval_idle_wait_time = 0.0
        interval_batch_wait_time = 0.0
        interval_batches = 0
        # NumPy concatenation/legal packing for batch N+1 can run while CUDA is
        # evaluating batch N.  A single preparation thread is intentional: it
        # preserves request order and avoids turning memory copies into a new CPU
        # contention point. The current pinned buffer is single-buffered, so do
        # not pipeline that optional mode until it has independent staging slots.
        if not pinned_staging_enabled:
            prepare_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='central-inference-prep',
            )
        pipeline_depth = 2 if prepare_executor is not None else 1

        def _request_positions(req):
            if bool(req.get("shared_memory", False)):
                return int(req.get("batch_size", 0) or 0)
            boards = req.get("boards")
            shape = getattr(boards, "shape", None)
            if shape:
                return int(shape[0])
            return int(np.asarray(boards).shape[0])

        def _handle_request_item(item):
            nonlocal first_infer_seen, pending_started_at, pending_positions
            cmd = item.get("cmd")
            if cmd == "stop":
                return "stop"
            if cmd == "load_models":
                task_id = item.get("task_id")
                if bool(item.get("clear", False)):
                    models.clear()
                    cache.clear()
                    cache_gate.reset()
                loaded_models = []
                load_t0 = time.perf_counter()
                for entry in list(item.get("models", []) or []):
                    state = entry.get("state")
                    state_path = entry.get("state_path")
                    if state is None and state_path:
                        state = torch.load(state_path, map_location='cpu')
                    if state is not None:
                        loaded_models.append(_load_model(entry.get("label", "learner"), state))
                if loaded_models:
                    # A label may have received new weights. Never reuse outputs
                    # computed by a previous model generation.
                    cache.clear()
                    cache_gate.reset()
                if loaded_models:
                    def _format_loaded_model(info):
                        label = str(info.get("label", "model"))
                        if info.get("reused"):
                            return f"{label}(reused)"
                        warmup_s = info.get("warmup_s")
                        if warmup_s is not None:
                            return f"{label}(warmup {float(warmup_s):.2f}s)"
                        if info.get("compiled"):
                            return f"{label}(compiled)"
                        return label

                    model_summary = ", ".join(_format_loaded_model(info) for info in loaded_models)
                    if not quiet_startup:
                        print(
                            f"[{_ts()}] Central inference: models ready on {device}: "
                            f"{model_summary} ({time.perf_counter() - load_t0:.2f}s).",
                            flush=True,
                        )
                if control_queue is not None:
                    control_queue.put({
                        "type": "models_loaded",
                        "task_id": task_id,
                        "labels": sorted(models.keys()),
                        "pid": int(os.getpid()),
                        "server_rank": int(compile_rank),
                        "device": str(device),
                         "max_batch": int(max_batch),
                         "auto_batch": bool(auto_batch_enabled),
                         "batch_target": int(batch_controller.target),
                        "flush_ms": float(flush_ms),
                        "transport": str(transport_np_dtype.__name__),
                        "shared_memory": bool(shared_arrays_by_rank),
                        "shared_workers": int(len(shared_arrays_by_rank)),
                        "cache_enabled": bool(cache_enabled and cache_max_entries > 0),
                        "cache_max_entries": int(cache_max_entries),
                        "pinned_staging": bool(pinned_staging_enabled and device.type == 'cuda'),
                        "compile": bool(central_use_compile),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device.type == 'cuda' else None,
                        "model_summary": model_summary if loaded_models else "",
                        "load_s": float(time.perf_counter() - load_t0),
                    })
                return "control"
            if cmd == "infer":
                item["server_dequeued_at_perf"] = time.perf_counter()
                arrays = _shared_request_arrays(item)
                if arrays is not None:
                    slot = int(item.get('shared_slot', 0))
                    expected_token = int(item.get('shared_token', 0) or 0)
                    actual_token = int(arrays['request_tokens'][slot])
                    if expected_token <= 0 or actual_token != expected_token:
                        _put_response(item, {
                            "ok": False,
                            "error": (
                                "Central inference rejected a stale shared-memory "
                                "request descriptor."
                            ),
                        })
                        return "ignored"
                if pending_started_at is None:
                    pending_started_at = time.perf_counter()
                item_positions = _request_positions(item)
                if debug_enabled and not first_infer_seen:
                    first_infer_seen = True
                    print(
                        f"[{_ts()}] Central inference: received first inference request "
                        f"from worker {int(item.get('rank', -1))} "
                        f"({item_positions} positions).",
                        flush=True,
                    )
                pending.append(item)
                pending_positions += item_positions
                model_label = str(item.get("model_label", "learner"))
                pending_positions_by_model[model_label] = (
                    int(pending_positions_by_model.get(model_label, 0)) + item_positions
                )
                return "infer"
            return "ignored"

        def _record_stage_stats(stage_stats, elapsed_s):
            nonlocal last_batch_time_s, last_batch_positions, last_stage_stats
            nonlocal interval_infer_time, interval_batches
            nonlocal processed_requests, processed_positions
            last_batch_time_s = max(0.0, float(elapsed_s))
            last_batch_positions = int(stage_stats.get('positions', 0) or 0)
            last_stage_stats = dict(stage_stats)
            interval_infer_time += last_batch_time_s
            interval_batches += 1
            processed_requests += int(stage_stats.get('requests', 0) or 0)
            processed_positions += last_batch_positions
            batch_controller.record(
                int(stage_stats.get('batch_target', batch_controller.target) or batch_controller.target),
                last_batch_positions,
            )

        while True:
            get_t0 = time.perf_counter()
            if pending_started_at is None:
                item = request_queue.get()
            else:
                elapsed_s = time.perf_counter() - pending_started_at
                remaining_s = (flush_ms / 1000.0) - elapsed_s
                if remaining_s <= 0.0:
                    item = None
                else:
                    try:
                        item = request_queue.get(timeout=remaining_s)
                    except queue.Empty:
                        item = None
            get_wait_s = time.perf_counter() - get_t0
            if pending_started_at is None:
                interval_idle_wait_time += get_wait_s
            else:
                interval_batch_wait_time += get_wait_s

            if item is not None:
                item_status = _handle_request_item(item)
                if item_status == "stop":
                    break
                if item_status == "control":
                    continue

            batch_target = int(batch_controller.target)
            # batch_target is the useful forward target for each loaded model,
            # not for the mixed learner/opponent queue as a whole.  Coalesce
            # already-available work for every label before serial GPU calls.
            drain_target_positions = batch_target * max(1, len(models)) * pipeline_depth
            while (
                pending
                and pending_positions < drain_target_positions
                and not any(
                    int(model_positions) >= batch_target * pipeline_depth
                    for model_positions in pending_positions_by_model.values()
                )
            ):
                try:
                    drained_item = request_queue.get_nowait()
                except queue.Empty:
                    break
                item_status = _handle_request_item(drained_item)
                if item_status == "stop":
                    return
                if item_status == "control":
                    continue

            should_flush = (
                pending
                and (
                    any(
                        int(model_positions) >= batch_target
                        for model_positions in pending_positions_by_model.values()
                    )
                    or (
                        pending_started_at is not None
                        and (time.perf_counter() - pending_started_at) >= (flush_ms / 1000.0)
                    )
                    or item is None
                )
            )
            if not should_flush:
                continue

            grouped = {}
            for req in pending:
                grouped.setdefault(str(req.get("model_label", "learner")), []).append(req)
            pending = []
            pending_positions = 0
            pending_positions_by_model.clear()
            pending_started_at = None
            grouped_items = sorted(
                grouped.items(),
                key=lambda item_pair: sum(_request_positions(req) for req in item_pair[1]),
                reverse=True,
            )
            inference_tasks = []
            for model_label, requests in grouped_items:
                request_batches = []
                current_batch = []
                current_positions = 0
                for req in requests:
                    request_positions = _request_positions(req)
                    if current_batch and current_positions + request_positions > batch_target:
                        request_batches.append(current_batch)
                        current_batch = []
                        current_positions = 0
                    current_batch.append(req)
                    current_positions += request_positions
                if current_batch:
                    request_batches.append(current_batch)

                inference_tasks.extend(
                    (model_label, request_batch)
                    for request_batch in request_batches
                )

            prepared_future = None
            if prepare_executor is not None and len(inference_tasks) > 1:
                prepared_future = prepare_executor.submit(
                    _prepare_group_inputs,
                    inference_tasks[0][1],
                    bool(cache_gate.active),
                )

            deferred_stage = None
            pipeline_outputs = bool(
                output_pipeline_enabled and output_copy_stream is not None
                and len(inference_tasks) > 1
                and not cache_gate.active
            )
            for task_idx, (model_label, request_batch) in enumerate(inference_tasks):
                next_future = None
                prepared_inputs = None
                try:
                    if prepared_future is not None:
                        prepared_inputs = prepared_future.result()
                    else:
                        prepared_inputs = _prepare_group_inputs(
                            request_batch,
                            compute_hashes=bool(cache_gate.active),
                        )
                    if prepare_executor is not None and task_idx + 1 < len(inference_tasks):
                        next_future = prepare_executor.submit(
                            _prepare_group_inputs,
                            inference_tasks[task_idx + 1][1],
                            bool(cache_gate.active),
                        )
                    try:
                        infer_t0 = time.perf_counter()
                        launched = _infer_group(
                            model_label,
                            request_batch,
                            prepared=prepared_inputs,
                            defer_output=pipeline_outputs,
                            output_slot=task_idx & 1,
                            batch_target=batch_target,
                        )
                        # Launch batch N first, then wait for and publish N-1.
                        # Its D2H/output copy therefore overlaps this forward.
                        if deferred_stage is not None:
                            previous_stats = deferred_stage['finalize']()
                            _record_stage_stats(
                                previous_stats,
                                time.perf_counter() - deferred_stage['started_at'],
                            )
                            deferred_stage = None
                        finalize = launched.get('deferred_finalize')
                        if finalize is not None:
                            deferred_stage = {
                                'finalize': finalize,
                                'started_at': infer_t0,
                            }
                        else:
                            _record_stage_stats(
                                launched,
                                time.perf_counter() - infer_t0,
                            )
                    finally:
                        prepared_future = next_future
                except Exception as exc:
                    prepared_future = next_future
                    if deferred_stage is not None:
                        try:
                            previous_stats = deferred_stage['finalize']()
                            _record_stage_stats(
                                previous_stats,
                                time.perf_counter() - deferred_stage['started_at'],
                            )
                        finally:
                            deferred_stage = None
                    for req in request_batch:
                        _put_response(req, {
                            "ok": False,
                            "error": str(exc),
                        })
            if deferred_stage is not None:
                final_stats = deferred_stage['finalize']()
                _record_stage_stats(
                    final_stats,
                    time.perf_counter() - deferred_stage['started_at'],
                )
            if debug_enabled and (time.perf_counter() - last_debug_print) >= 5.0:
                interval_s = max(1e-9, time.perf_counter() - last_debug_print)
                last_debug_print = time.perf_counter()
                pos_per_s = (
                    float(last_batch_positions) / max(1e-9, float(last_batch_time_s))
                    if last_batch_positions > 0
                    else 0.0
                )
                active_pct = 100.0 * float(interval_infer_time) / interval_s
                idle_pct = 100.0 * float(interval_idle_wait_time) / interval_s
                batch_wait_pct = 100.0 * float(interval_batch_wait_time) / interval_s
                print(
                    f"[{_ts()}] Central inference: heartbeat "
                    f"requests={processed_requests}, positions={processed_positions}, "
                    f"last_batch={last_batch_positions} pos/{last_batch_time_s:.3f}s "
                    f"({pos_per_s:.1f} pos/s), "
                    f"server_active={active_pct:.1f}%, idle_no_requests={idle_pct:.1f}%, "
                    f"batch_wait={batch_wait_pct:.1f}%, batches={interval_batches}, "
                    f"compact={bool(last_stage_stats.get('compact_policy', False))}, "
                    f"h2d={float(last_stage_stats.get('h2d_time', 0.0)):.3f}s, "
                    f"fwd={float(last_stage_stats.get('forward_time', 0.0)):.3f}s, "
                    f"d2h={float(last_stage_stats.get('d2h_time', 0.0)):.3f}s, "
                    f"send={float(last_stage_stats.get('send_time', 0.0)):.3f}s, "
                    f"last_flush_models={len(grouped)}, pending={len(pending)}.",
                    flush=True,
                )
                interval_infer_time = 0.0
                interval_idle_wait_time = 0.0
                interval_batch_wait_time = 0.0
                interval_batches = 0
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Central inference server failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if prepare_executor is not None:
            prepare_executor.shutdown(wait=True, cancel_futures=True)
