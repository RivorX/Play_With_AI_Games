"""SE-CNN v9 compact 1858-action policy codec."""

from __future__ import annotations


CODEC_ID = "lc0_1858_v1"
ACTION_SIZE = 1858
POLICY_PLANES = 73


def move_to_index(move, board):
    from src.models.data.se_cnn_v9.helpers import move_to_index as encode_move

    return encode_move(move, board)


def index_to_move(index, is_black_turn=False, board=None):
    from src.models.data.se_cnn_v9.helpers import index_to_move as decode_move

    return decode_move(index, is_black_turn=is_black_turn, board=board)


def policy_index_maps():
    from src.models.data.se_cnn_v9.helpers import get_policy_index_maps

    return get_policy_index_maps()
