"""Bounded CUDA-graph wrapper for neural-network forward inference.

The graph covers only the model forward. Callers remain responsible for input
transfer, legal-move gathering and output transfer. CUDA graph outputs are
cloned before they leave this module, so a later replay cannot overwrite a
result that the caller still owns.

This runner is intended for one immutable, evaluation-mode model per instance.
It captures only exact input batch shapes; all other inputs use the eager path.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from src.inference.prediction import predict_raw


_DEVICE_SETUP_LOCKS_GUARD = threading.Lock()
_DEVICE_SETUP_LOCKS: dict[tuple[str, int | None], threading.RLock] = {}


def cuda_model_setup_lock(device: torch.device | str):
    """Return a process-local lock for CUDA model snapshot/capture setup.

    CUDA graph capture and copying live model weights to CPU both enqueue work
    against the same model/device. Serializing those setup operations prevents
    an otherwise unrelated state-dict snapshot from invalidating a concurrent
    graph capture. Ordinary graph replays do not take this lock.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return nullcontext()
    index = device.index
    if index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    key = (device.type, int(index))
    with _DEVICE_SETUP_LOCKS_GUARD:
        return _DEVICE_SETUP_LOCKS.setdefault(key, threading.RLock())


@dataclass
class _GraphEntry:
    key: tuple[Any, ...]
    batch_size: int
    static_input: torch.Tensor
    graph: torch.cuda.CUDAGraph
    replay_stream: torch.cuda.Stream
    static_policy: torch.Tensor
    static_value: torch.Tensor
    input_ready: torch.cuda.Event
    output_ready: torch.cuda.Event
    last_use: torch.cuda.Event | None = None
    estimated_bytes: int = 0


class CUDAGraphForwardRunner:
    """Replay a model forward for a bounded set of exact batch shapes.

    Args:
        model: A stable, evaluation-mode PyTorch model. Its weights must not
            change during this runner's lifetime; call :meth:`clear` before
            using changed weights.
        device: CUDA device holding the model.
        use_amp: Whether the existing inference AMP mode should be preserved.
        amp_dtype: Existing AMP dtype (for example ``torch.bfloat16``).
        exact_batch_sizes: Exact permitted graph batch sizes. Each shape is
            captured at most once; all other sizes use the eager path. Keeping
            this list small bounds private graph-pool use.
        min_capture_observations: Number of observations of the same input
            shape, dtype and device required before its first graph capture.
        max_entries: Hard per-model graph-count limit.
        max_graph_memory_fraction: Approximate per-model graph-pool budget as a
            fraction of total device memory. Separate model runners should use
            small fractions whose sum fits the application's memory budget.
        min_free_memory_bytes: Preserve at least this amount of device memory
            before starting a new capture.
        warmup_iterations: Eager warmups required before graph capture.

    The object serializes host-side access to its static buffers. CUDA events
    protect them between caller streams while leaving forward execution
    asynchronous.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str,
        *,
        use_amp: bool = False,
        amp_dtype: torch.dtype | None = None,
        exact_batch_sizes: Iterable[int] = (),
        min_capture_observations: int = 1,
        max_entries: int = 8,
        max_graph_memory_fraction: float = 0.04,
        min_free_memory_bytes: int = 512 * 1024 * 1024,
        warmup_iterations: int = 3,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.exact_batch_sizes = tuple(
            sorted({int(size) for size in exact_batch_sizes if int(size) > 0})
        )
        self.min_capture_observations = max(1, int(min_capture_observations))
        self.max_entries = max(0, int(max_entries))
        self.max_graph_memory_fraction = float(max_graph_memory_fraction)
        self.min_free_memory_bytes = max(0, int(min_free_memory_bytes))
        self.warmup_iterations = max(1, int(warmup_iterations))

        if not math.isfinite(self.max_graph_memory_fraction) or not (
            0.0 <= self.max_graph_memory_fraction <= 0.5
        ):
            raise ValueError("max_graph_memory_fraction must be finite and between 0 and 0.5")
        if (
            self.device.type == "cuda"
            and self.device.index is None
            and torch.cuda.is_available()
        ):
            self.device = torch.device("cuda", torch.cuda.current_device())

        self._lock = threading.RLock()
        self._entries: dict[tuple[Any, ...], _GraphEntry] = {}
        self._estimated_graph_bytes = 0
        self._stats: dict[str, Any] = {
            "captures": 0,
            "replays": 0,
            "eager_fallbacks": 0,
            "capture_failures": 0,
            "replay_failures": 0,
            "entries": 0,
            "estimated_graph_bytes": 0,
            "last_fallback_reason": None,
            "last_capture_seconds": 0.0,
        }
        self._shape_observations: dict[tuple[Any, ...], int] = {}

    @property
    def stats(self) -> dict[str, Any]:
        """Return a snapshot of counters suitable for profiling logs."""
        with self._lock:
            result = dict(self._stats)
            result["entries"] = len(self._entries)
            result["estimated_graph_bytes"] = self._estimated_graph_bytes
            result["exact_batch_sizes"] = self.exact_batch_sizes
            result["min_capture_observations"] = self.min_capture_observations
            result["shape_observations"] = {
                f"{key[0]}|{key[1]}|{key[2]}|N={key[3]}": int(count)
                for key, count in self._shape_observations.items()
            }
            return result

    def clear(self) -> None:
        """Synchronize outstanding graph work and release all captured entries.

        Call this before changing model weights or when the model is unloaded.
        """
        with self._lock:
            for entry in self._entries.values():
                # Static buffers and private graph pools must outlive queued
                # replay/clone work. This is a lifecycle operation, not a hot
                # path, so waiting here is intentional.
                entry.replay_stream.synchronize()
                if entry.last_use is not None:
                    entry.last_use.synchronize()
            self._entries.clear()
            self._estimated_graph_bytes = 0
            self._shape_observations.clear()
            self._stats["entries"] = 0
            self._stats["estimated_graph_bytes"] = 0

    def run_forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``model(inputs, apply_log_softmax=False)`` and return owned outputs."""
        with self._lock:
            actual_batch = int(inputs.shape[0]) if inputs.ndim else 0
            fallback_reason = self._fallback_reason(inputs)
            if fallback_reason is None and actual_batch not in self.exact_batch_sizes:
                fallback_reason = "unconfigured_exact_batch_size"

            if fallback_reason is not None:
                return self._fallback(inputs, fallback_reason)

            key = (str(self.device), tuple(inputs.shape[1:]), inputs.dtype, actual_batch)
            observations = self._shape_observations.get(key, 0) + 1
            self._shape_observations[key] = observations
            entry = self._entries.get(key)
            if entry is None:
                if observations < self.min_capture_observations:
                    return self._fallback(inputs, "awaiting_hot_shape_observations")
                if len(self._entries) >= self.max_entries:
                    return self._fallback(inputs, "entry_limit")
                if not self._memory_guard(actual_batch, int(inputs.shape[1]), inputs.element_size()):
                    return self._fallback(
                        inputs,
                        self._stats["last_fallback_reason"] or "memory_guard",
                    )
                with cuda_model_setup_lock(self.device):
                    entry = self._capture_entry(key, actual_batch, inputs)
                if entry is None:
                    return self._fallback(inputs, self._stats["last_fallback_reason"] or "capture_failed")
                self._entries[key] = entry
                self._estimated_graph_bytes += entry.estimated_bytes
                self._stats["captures"] += 1
                self._stats["entries"] = len(self._entries)
                self._stats["estimated_graph_bytes"] = self._estimated_graph_bytes

            try:
                return self._replay(entry, inputs)
            except Exception:
                # A launch failure can indicate an unsupported graph path.
                # Remove it, then verify the ordinary eager forward; a real
                # model error is deliberately allowed to propagate there.
                self._stats["replay_failures"] += 1
                self._drop_entry(key)
                return self._fallback(inputs, "replay_failed")

    def _fallback_reason(self, inputs: torch.Tensor) -> str | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "cuda_unavailable"
        if inputs.device.type != "cuda" or inputs.device != self.device:
            return "input_not_on_runner_device"
        if not inputs.is_floating_point() or inputs.ndim != 4 or tuple(inputs.shape[-2:]) != (8, 8):
            return "unsupported_input_shape_or_dtype"
        if inputs.shape[0] <= 0:
            return "empty_batch"
        if bool(getattr(self.model, "training", False)):
            return "model_in_training_mode"
        if not callable(self.model):
            return "model_not_callable"
        try:
            model_device = next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return "model_has_no_parameters"
        if model_device != self.device:
            return "model_not_on_runner_device"
        return None

    def _forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return predict_raw(
            self.model,
            inputs,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            amp_cache_enabled=False,
        )

    def _fallback(self, inputs: torch.Tensor, reason: str) -> tuple[torch.Tensor, torch.Tensor]:
        self._stats["eager_fallbacks"] += 1
        self._stats["last_fallback_reason"] = str(reason)
        # Eager outputs are newly allocated model results and do not alias any
        # graph-owned static storage.
        return self._forward(inputs)

    def _memory_guard(self, batch_size: int, planes: int, item_size: int) -> bool:
        if self.max_entries == 0 or self.max_graph_memory_fraction <= 0:
            return False
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            self._stats["last_fallback_reason"] = "memory_info_unavailable"
            return False
        budget_bytes = int(total_bytes * self.max_graph_memory_fraction)
        # Reserve a conservative activation estimate before capture. The real
        # allocator delta is measured afterward and replaces this estimate.
        estimated_bytes = max(
            64 * 1024 * 1024,
            int(batch_size * planes * 8 * 8 * item_size * 16),
        )
        if self._estimated_graph_bytes + estimated_bytes > budget_bytes:
            self._stats["last_fallback_reason"] = "graph_memory_budget"
            return False
        if free_bytes - estimated_bytes < self.min_free_memory_bytes:
            self._stats["last_fallback_reason"] = "minimum_free_memory"
            return False
        return True

    def _capture_entry(
        self,
        key: tuple[Any, ...],
        batch_size: int,
        sample_inputs: torch.Tensor,
    ) -> _GraphEntry | None:
        started = time.perf_counter()
        channels = int(sample_inputs.shape[1])
        static_input = torch.empty(
            (batch_size, channels, 8, 8),
            device=self.device,
            dtype=sample_inputs.dtype,
            memory_format=torch.channels_last,
        )
        replay_stream = torch.cuda.Stream(device=self.device)
        input_ready = torch.cuda.Event(blocking=False)
        output_ready = torch.cuda.Event(blocking=False)
        graph = torch.cuda.CUDAGraph()
        current_stream = torch.cuda.current_stream(self.device)

        try:
            # The capture stream owns all captured work. Warmups establish
            # cuDNN/BLAS choices and allocator state outside capture.
            replay_stream.wait_stream(current_stream)
            with torch.cuda.stream(replay_stream):
                static_input.zero_()
                self._forward(static_input)
                for _ in range(self.warmup_iterations - 1):
                    self._forward(static_input)
            current_stream.wait_stream(replay_stream)

            # Confirm the eager path works before classifying a capture-only
            # exception as a compatibility fallback.
            with torch.cuda.stream(replay_stream):
                self._forward(static_input)
            current_stream.wait_stream(replay_stream)
            replay_stream.synchronize()
            free_before, _ = torch.cuda.mem_get_info(self.device)

            with torch.cuda.graph(graph, stream=replay_stream):
                static_policy, static_value = predict_raw(
                    self.model,
                    static_input,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    amp_cache_enabled=False,
                )
                if not isinstance(static_policy, torch.Tensor) or not isinstance(static_value, torch.Tensor):
                    raise TypeError("model graph forward must return policy and value tensors")
                if static_policy.shape[0] != batch_size or static_value.shape[0] != batch_size:
                    raise ValueError("model graph output batch does not match captured shape")

            # Graph memory pools are private to each captured graph. Record the
            # actual device-memory delta for diagnostics and the bounded cache.
            free_after, total_after = torch.cuda.mem_get_info(self.device)
            used_bytes = max(0, int(free_before) - int(free_after))
            estimated_bytes = max(
                64 * 1024 * 1024,
                used_bytes,
                int(batch_size * channels * 8 * 8 * sample_inputs.element_size() * 16),
            )
            budget_bytes = int(total_after * self.max_graph_memory_fraction)
            if self._estimated_graph_bytes + estimated_bytes > budget_bytes:
                del graph, static_policy, static_value, static_input
                replay_stream.synchronize()
                self._stats["capture_failures"] += 1
                self._stats["last_fallback_reason"] = "captured_graph_exceeds_memory_budget"
                return None

            self._stats["last_capture_seconds"] = time.perf_counter() - started
            self._stats["last_fallback_reason"] = None
            return _GraphEntry(
                key=key,
                batch_size=batch_size,
                static_input=static_input,
                graph=graph,
                replay_stream=replay_stream,
                static_policy=static_policy,
                static_value=static_value,
                input_ready=input_ready,
                output_ready=output_ready,
                estimated_bytes=estimated_bytes,
            )
        except Exception as exc:
            # Capture incompatibility is permitted to fall back, but forward
            # correctness is checked by _fallback() before a result is returned.
            try:
                replay_stream.synchronize()
            except Exception:
                pass
            self._stats["capture_failures"] += 1
            self._stats["last_fallback_reason"] = f"capture_error:{type(exc).__name__}"
            return None

    def _replay(
        self,
        entry: _GraphEntry,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(inputs.shape[0]) != entry.batch_size:
            raise ValueError("CUDA graph input batch does not match its captured shape")
        current_stream = torch.cuda.current_stream(self.device)
        if entry.last_use is not None:
            current_stream.wait_event(entry.last_use)

        entry.static_input.copy_(inputs, non_blocking=True)
        entry.input_ready.record(current_stream)

        entry.replay_stream.wait_event(entry.input_ready)
        with torch.cuda.stream(entry.replay_stream):
            entry.graph.replay()
            entry.output_ready.record(entry.replay_stream)

        current_stream.wait_event(entry.output_ready)
        # These clones are essential: callers may collect several chunk
        # results and only gather legal moves after run_forward returns.
        policy = entry.static_policy.clone()
        value = entry.static_value.clone()
        if policy.shape[0] != entry.batch_size or value.shape[0] != entry.batch_size:
            raise RuntimeError("CUDA graph returned an unexpected output batch")
        if entry.last_use is None:
            entry.last_use = torch.cuda.Event(blocking=False)
        entry.last_use.record(current_stream)
        self._stats["replays"] += 1
        return policy, value

    def _drop_entry(self, key: tuple[Any, ...]) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        try:
            entry.replay_stream.synchronize()
            if entry.last_use is not None:
                entry.last_use.synchronize()
        finally:
            self._estimated_graph_bytes = max(
                0, self._estimated_graph_bytes - int(entry.estimated_bytes)
            )
            self._stats["entries"] = len(self._entries)
            self._stats["estimated_graph_bytes"] = self._estimated_graph_bytes

