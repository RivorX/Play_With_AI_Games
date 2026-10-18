"""SE-CNN v9 16-plane-per-position board encoder."""

from __future__ import annotations

import numpy as np


ENCODER_ID = "planes_history_v1"
PLANES_PER_POSITION = 16


def input_planes(history_positions: int) -> int:
    return PLANES_PER_POSITION * (1 + int(history_positions))


def encode(board, flip_perspective=None, dtype=None):
    from src.models.data.se_cnn_v9.helpers import board_to_tensor

    kwargs = {"flip_perspective": flip_perspective}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return board_to_tensor(board, **kwargs)


def encode_pair(board, dtype=None):
    from src.models.data.se_cnn_v9.helpers import board_to_tensor_pair

    return board_to_tensor_pair(board) if dtype is None else board_to_tensor_pair(
        board, dtype=dtype
    )


def stack_history(current_tensor, history_tensors, history_positions, dtype=None):
    """Stack v9's 16-plane history frames and current position."""
    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")
    current = np.asarray(current_tensor)
    if current.shape != (16, 8, 8):
        raise ValueError(f"current_tensor must have shape (16, 8, 8), got {current.shape}")
    history = list(history_tensors or ())[-history_positions:] if history_positions else []
    output_dtype = np.dtype(dtype if dtype is not None else current.dtype)
    frames = [
        np.zeros((16, 8, 8), dtype=output_dtype)
        for _ in range(history_positions - len(history))
    ]
    for index, tensor in enumerate(history):
        frame = np.asarray(tensor)
        if frame.shape != (16, 8, 8):
            raise ValueError(
                f"history tensor {index} must have shape (16, 8, 8), got {frame.shape}"
            )
        frames.append(np.asarray(frame, dtype=output_dtype))
    frames.append(np.asarray(current, dtype=output_dtype))
    return np.concatenate(frames, axis=0)
