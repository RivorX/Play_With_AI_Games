"""SE-CNN v9 16-plane-per-position board encoder."""

from __future__ import annotations


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
