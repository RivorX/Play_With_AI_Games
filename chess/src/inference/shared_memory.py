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
    total_bytes = _align(offset)
    raw_buffer = mp_context.RawArray(ctypes.c_ubyte, total_bytes)
    return {
        "buffer": raw_buffer,
        "arrays": arrays,
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


def shared_inference_bytes(spec: dict[str, Any] | None) -> int:
    return int((spec or {}).get("total_bytes", 0) or 0)
