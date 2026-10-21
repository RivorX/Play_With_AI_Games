"""Fixed-size shared-memory slots for central neural-network inference.

Large NumPy payloads must not travel through ``multiprocessing.Queue`` on the
hot self-play path.  The parent creates one buffer per worker before spawning
processes; workers and the GPU server then exchange only small slot
descriptors.  ``RawArray`` owns the cross-platform shared allocation and avoids
the manual unlink/resource-tracker lifecycle of named shared-memory segments.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np


_ALIGNMENT = 64
_SHARED_INFERENCE_SCHEMA_VERSION = 2
_UINT64_MAX = (1 << 64) - 1


def _align(value: int, alignment: int = _ALIGNMENT) -> int:
    value = int(value)
    alignment = max(1, int(alignment))
    return ((value + alignment - 1) // alignment) * alignment


def _array_layout(offset: int, shape, dtype):
    dtype = np.dtype(dtype)
    offset = _align(offset, max(_ALIGNMENT, dtype.itemsize))
    shape = tuple(int(dim) for dim in shape)
    size = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    return {
        "offset": offset,
        "shape": shape,
        "dtype": dtype.str,
        "nbytes": size,
    }, offset + size


def create_shared_inference_buffer(
    mp_context,
    *,
    slots: int,
    capacity: int,
    input_planes: int,
    max_legal_moves: int,
) -> dict[str, Any]:
    """Allocate one worker's input/output slot arena."""
    slots = max(1, int(slots))
    capacity = max(1, int(capacity))
    input_planes = max(1, int(input_planes))
    max_legal_moves = max(1, int(max_legal_moves))

    offset = 0
    arrays = {}
    arrays["boards"], offset = _array_layout(
        offset,
        (slots, capacity, input_planes, 8, 8),
        np.float16,
    )
    arrays["legal_indices"], offset = _array_layout(
        offset,
        (slots, capacity, max_legal_moves),
        np.int16,
    )
    arrays["legal_counts"], offset = _array_layout(
        offset,
        (slots, capacity),
        np.int16,
    )
    arrays["policy_logits"], offset = _array_layout(
        offset,
        (slots, capacity, max_legal_moves),
        np.float16,
    )
    arrays["value_logits"], offset = _array_layout(
        offset,
        (slots, capacity, 3),
        np.float32,
    )
    arrays["request_tokens"], offset = _array_layout(
        offset,
        (slots,),
        np.int64,
    )
    arrays["response_tokens"], offset = _array_layout(
        offset,
        (slots,),
        np.int64,
    )
    # The server publishes an odd value while model weights/cache state are
    # changing and an even value only after the complete model set is ready.
    # Keep this naturally aligned for atomic uint64 loads/stores on Windows
    # x64; clients use a seqlock read (epoch, incarnation, epoch).
    arrays["evaluation_generation"], offset = _array_layout(
        offset,
        (1,),
        np.uint64,
    )
    arrays["server_incarnation"], offset = _array_layout(
        offset,
        (2,),
        np.uint64,
    )
    total_bytes = _align(offset)
    raw_buffer = mp_context.RawArray(ctypes.c_ubyte, total_bytes)
    # Generation 1 means invalid until the inference server completes its
    # first successful model-load transaction.
    generation_spec = {
        "buffer": raw_buffer,
        "arrays": arrays,
        "schema_version": _SHARED_INFERENCE_SCHEMA_VERSION,
    }
    shared_inference_array(generation_spec, "evaluation_generation")[0] = np.uint64(1)
    shared_inference_array(generation_spec, "server_incarnation").fill(0)
    return {
        "buffer": raw_buffer,
        "arrays": arrays,
        "schema_version": _SHARED_INFERENCE_SCHEMA_VERSION,
        # A lock gives Python-level cross-process ordering on every supported
        # platform; the underlying uint64 is also naturally aligned for the
        # Windows x64 hardware atomicity guarantee.
        "evaluation_generation_lock": mp_context.Lock(),
        "slots": slots,
        "capacity": capacity,
        "input_planes": input_planes,
        "max_legal_moves": max_legal_moves,
        "total_bytes": total_bytes,
    }


def shared_inference_array(spec: dict[str, Any], name: str) -> np.ndarray:
    """Return a zero-copy NumPy view for one named arena array."""
    layout = spec["arrays"][str(name)]
    return np.ndarray(
        tuple(layout["shape"]),
        dtype=np.dtype(layout["dtype"]),
        buffer=spec["buffer"],
        offset=int(layout["offset"]),
    )


def shared_inference_generation_arrays(spec: dict[str, Any] | None):
    """Return the optional shared generation views for schema-v2 buffers.

    Older descriptors remain usable for ordinary inference, but they cannot
    prove which live model generation produced an evaluation. Callers must
    treat the returned ``None`` as unknown and disable/fail closed any
    persistent NN-result reuse.
    """
    if not isinstance(spec, dict):
        return None
    schema_version = int(spec.get("schema_version", 1) or 1)
    if schema_version < _SHARED_INFERENCE_SCHEMA_VERSION:
        return None
    if schema_version > _SHARED_INFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported shared inference schema version {schema_version}."
        )
    arrays = spec.get("arrays")
    if not isinstance(arrays, dict) or not {
        "evaluation_generation", "server_incarnation"
    }.issubset(arrays):
        raise ValueError(
            "Shared inference schema v2 requires evaluation-generation arrays."
        )
    epoch = shared_inference_array(spec, "evaluation_generation")
    incarnation = shared_inference_array(spec, "server_incarnation")
    if epoch.dtype != np.dtype(np.uint64) or epoch.shape != (1,):
        raise ValueError("Invalid shared inference evaluation-generation layout.")
    if incarnation.dtype != np.dtype(np.uint64) or incarnation.shape != (2,):
        raise ValueError("Invalid shared inference server-incarnation layout.")
    if not epoch.flags.aligned or not incarnation.flags.aligned:
        raise ValueError("Shared inference generation fields must be naturally aligned.")
    lock = spec.get("evaluation_generation_lock")
    if lock is None:
        raise ValueError(
            "Shared inference schema v2 requires an evaluation-generation lock."
        )
    return epoch, incarnation, lock


def read_shared_evaluation_generation(
    spec: dict[str, Any] | None,
    model_label: str,
    *,
    views=None,
):
    """Read a stable model-generation token, or return ``None`` if unknown.

    The aligned uint64 epoch is odd during a model/cache transition and even
    only after commit. The shared lock makes the epoch/incarnation snapshot
    coherent across processes; this read is local and occurs at search
    boundaries, never for each inference leaf.
    """
    views = views if views is not None else shared_inference_generation_arrays(spec)
    if views is None:
        return None
    epoch, incarnation, lock = views
    with lock:
        before = int(epoch[0])
        if before == 0 or (before & 1):
            return None
        first, second = int(incarnation[0]), int(incarnation[1])
        after = int(epoch[0])
        if before != after or after == 0 or (after & 1) or (first == 0 and second == 0):
            return None
    return (f"{first:016x}{second:016x}", str(model_label or "learner"), after)


class SharedInferenceGenerationPublisher:
    """Server-side odd/even epoch publisher for shared inference buffers."""

    def __init__(self, specs):
        self._views = []
        for spec in specs or ():
            views = shared_inference_generation_arrays(spec)
            if views is not None:
                self._views.append(views)
        self.epoch = 1
        self.incarnation = None

    @property
    def stable(self) -> bool:
        return self.epoch > 0 and not (self.epoch & 1)

    def initialize(self, incarnation: str) -> None:
        incarnation = str(incarnation).lower()
        if len(incarnation) != 32:
            raise ValueError("Server incarnation must be a 128-bit hexadecimal UUID.")
        try:
            high = int(incarnation[:16], 16)
            low = int(incarnation[16:], 16)
        except ValueError as exc:
            raise ValueError("Server incarnation must be a 128-bit hexadecimal UUID.") from exc
        self.epoch = 1
        self.incarnation = incarnation
        for epoch, shared_incarnation, lock in self._views:
            with lock:
                epoch[0] = np.uint64(1)
                shared_incarnation[0] = np.uint64(high)
                shared_incarnation[1] = np.uint64(low)

    def begin_update(self) -> None:
        # Leave a failed/incomplete transaction odd. Repeated updates while
        # already invalid retain that odd epoch and the next commit advances
        # to a new even generation.
        if self.epoch & 1:
            return
        if self.epoch >= _UINT64_MAX - 1:
            # Use the final odd value as invalid and refuse to wrap: a stale
            # generation must never become valid through an ABA rollover.
            self.epoch = _UINT64_MAX
        else:
            self.epoch += 1
        for epoch, _, lock in self._views:
            with lock:
                epoch[0] = np.uint64(self.epoch)

    def commit(self) -> int:
        if not (self.epoch & 1):
            raise RuntimeError("Cannot commit a shared model generation without an update.")
        if self.epoch >= _UINT64_MAX - 1:
            # Wrapping could make a stale token appear current. Fail closed.
            raise OverflowError("Shared inference generation epoch exhausted.")
        self.epoch += 1
        for epoch, _, lock in self._views:
            with lock:
                epoch[0] = np.uint64(self.epoch)
        return self.epoch


def shared_inference_bytes(spec: dict[str, Any] | None) -> int:
    return int((spec or {}).get("total_bytes", 0) or 0)
