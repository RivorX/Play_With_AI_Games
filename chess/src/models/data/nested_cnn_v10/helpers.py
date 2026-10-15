"""CNN v10 board tensor construction.

Channel layout for a single board (20 x 8 x 8):
  0:6   side-to-move pieces, P/N/B/R/Q/K
  6:12  opponent pieces, P/N/B/R/Q/K
  12    previous occurrences of this position (0, 0.5, 1); -1 if unknown
  13:17 current side castling rights, K/Q then opponent K/Q (constant planes)
  17    en-passant target square
  18    halfmove clock clipped at 150 and normalized by 150
  19    constant all-one board extent plane

For a history stack, each prior position contributes only planes 0:13;
the current position contributes planes 0:20. Prior frames are chronological.
Missing padded history has empty piece planes and repetition=-1.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from src.game import backend as chess


_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def _opposite_color(color):
    return chess.BLACK if color == chess.WHITE else chess.WHITE


def _normalize_repetition_count(repetition_count):
    if repetition_count is None:
        return -1.0
    if isinstance(repetition_count, bool) or not isinstance(repetition_count, Integral):
        raise TypeError("repetition_count must be an integer or None")
    value = int(repetition_count)
    if value < 0:
        raise ValueError("repetition_count must be nonnegative or None")
    return min(value, 2) / 2.0


def _piece_planes(board, pov_color):
    """Return side-to-move-first piece planes in the requested POV."""
    masks = np.asarray(
        [
            int(board[color, piece_type])
            for color in (pov_color, _opposite_color(pov_color))
            for piece_type in _PIECE_TYPES
        ],
        dtype="<u8",
    )
    planes = np.unpackbits(masks.view(np.uint8), bitorder="little").reshape(12, 8, 8)
    if pov_color == chess.BLACK:
        planes = planes[:, ::-1, ::-1]
    return planes


def _set_constant_plane(tensor, index, value):
    tensor[index].fill(value)


def _write_frame(board, tensor, pov_color, repetition_count):
    if tensor.shape != (13, 8, 8):
        raise ValueError(f"frame destination must have shape (13, 8, 8), got {tensor.shape}")
    tensor[:12] = _piece_planes(board, pov_color)
    _set_constant_plane(tensor, 12, _normalize_repetition_count(repetition_count))


def _write_metadata(board, tensor, pov_color):
    if tensor.shape != (7, 8, 8):
        raise ValueError(f"metadata destination must have shape (7, 8, 8), got {tensor.shape}")

    opponent = _opposite_color(pov_color)
    rights = (
        chess.has_kingside_castling_rights(board, pov_color),
        chess.has_queenside_castling_rights(board, pov_color),
        chess.has_kingside_castling_rights(board, opponent),
        chess.has_queenside_castling_rights(board, opponent),
    )
    for index, has_right in enumerate(rights):
        _set_constant_plane(tensor, index, float(bool(has_right)))

    tensor[4].fill(0.0)
    if board.en_passant_square is not None:
        square = chess.square_index(board.en_passant_square)
        row, col = divmod(square, 8)
        if pov_color == chess.BLACK:
            row, col = 7 - row, 7 - col
        tensor[4, row, col] = 1.0

    halfmove_clock = int(board.halfmove_clock)
    if halfmove_clock < 0:
        raise ValueError(f"halfmove_clock must be nonnegative, got {halfmove_clock}")
    _set_constant_plane(tensor, 5, min(halfmove_clock, 150) / 150.0)
    _set_constant_plane(tensor, 6, 1.0)


def board_to_tensor(
    board,
    flip_perspective=None,
    dtype=np.float32,
    out=None,
    *,
    repetition_count=None,
):
    """Encode a board as ``(20, 8, 8)`` for one explicit or default POV.

    The default POV is the side to move. Repetition provenance is never guessed
    from the halfmove clock or a FEN: pass a known previous-occurrence count, or
    leave it ``None`` to use the documented -1 unknown sentinel.
    """
    if flip_perspective is None:
        pov_color = board.turn
    elif isinstance(flip_perspective, (bool, np.bool_)):
        pov_color = chess.BLACK if bool(flip_perspective) else chess.WHITE
    else:
        raise TypeError("flip_perspective must be bool or None")
    if out is None:
        tensor = np.empty((20, 8, 8), dtype=dtype)
    else:
        tensor = np.asarray(out)
        if tensor.shape != (20, 8, 8):
            raise ValueError(f"board_to_tensor out must have shape (20, 8, 8), got {tensor.shape}")
    _write_frame(board, tensor[:13], pov_color, repetition_count)
    _write_metadata(board, tensor[13:20], pov_color)
    return tensor


def board_to_tensor_pair(board, dtype=np.float32, *, repetition_count=None):
    """Encode both color POVs while extracting each piece mask once."""
    white = np.empty((20, 8, 8), dtype=dtype)
    black = np.empty((20, 8, 8), dtype=dtype)
    canonical = _piece_planes(board, chess.WHITE)
    white[:12] = canonical
    black[:6] = canonical[6:12, ::-1, ::-1]
    black[6:12] = canonical[:6, ::-1, ::-1]
    repetition_plane = _normalize_repetition_count(repetition_count)
    white[12].fill(repetition_plane)
    black[12].fill(repetition_plane)
    _write_metadata(board, white[13:20], chess.WHITE)
    _write_metadata(board, black[13:20], chess.BLACK)
    return white, black


def empty_history_frame(dtype=np.float32):
    """Return an explicit missing-history frame: no pieces, unknown repetition."""
    frame = np.zeros((13, 8, 8), dtype=dtype)
    frame[12].fill(-1.0)
    return frame


def stack_encoded_history(current_tensor, history_tensors, history_positions, dtype=None):
    """Compose already POV-aligned 20-plane tensors into the v10 stack layout."""
    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")
    current = np.asarray(current_tensor)
    if current.shape != (20, 8, 8):
        raise ValueError(f"current_tensor must have shape (20, 8, 8), got {current.shape}")
    history = list(history_tensors or ())[-history_positions:] if history_positions else []
    for index, tensor in enumerate(history):
        history[index] = np.asarray(tensor)
        if history[index].shape != (20, 8, 8):
            raise ValueError(
                f"history tensor {index} must have shape (20, 8, 8), got {history[index].shape}"
            )

    output_dtype = np.dtype(dtype if dtype is not None else current.dtype)
    frames = [empty_history_frame(output_dtype) for _ in range(history_positions - len(history))]
    frames.extend(np.asarray(item[:13], dtype=output_dtype) for item in history)
    frames.append(np.asarray(current[:13], dtype=output_dtype))
    frames.append(np.asarray(current[13:20], dtype=output_dtype))
    return np.concatenate(frames, axis=0)


def encode_history(
    current_board,
    history_boards=(),
    history_positions=0,
    *,
    repetition_counts=None,
    flip_perspective=None,
    dtype=np.float32,
):
    """Encode chronological prior boards plus the current board.

    ``repetition_counts`` (if supplied) describes the untruncated sequence
    ``history_boards`` followed by ``current_board``. If omitted, repetition is
    unknown for every encoded frame. History metadata other than piece state and
    repetition is intentionally omitted; only current metadata follows the stack.
    """
    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")
    all_boards = list(history_boards or ()) + [current_board]
    if repetition_counts is None:
        counts = [None] * len(all_boards)
    else:
        counts = list(repetition_counts)
        if len(counts) != len(all_boards):
            raise ValueError(
                "repetition_counts must match history_boards followed by current_board "
                f"({len(all_boards)} entries), got {len(counts)}"
            )

    selected_start = max(0, len(all_boards) - (history_positions + 1))
    selected_boards = all_boards[selected_start:]
    selected_counts = counts[selected_start:]
    if flip_perspective is None:
        pov_color = current_board.turn
    elif isinstance(flip_perspective, (bool, np.bool_)):
        pov_color = chess.BLACK if bool(flip_perspective) else chess.WHITE
    else:
        raise TypeError("flip_perspective must be bool or None")

    frames = [empty_history_frame(dtype) for _ in range(history_positions - (len(selected_boards) - 1))]
    for board, repetition_count in zip(selected_boards[:-1], selected_counts[:-1]):
        frame = np.empty((13, 8, 8), dtype=dtype)
        _write_frame(board, frame, pov_color, repetition_count)
        frames.append(frame)

    current = np.empty((20, 8, 8), dtype=dtype)
    _write_frame(current_board, current[:13], pov_color, selected_counts[-1])
    _write_metadata(current_board, current[13:20], pov_color)
    frames.append(current[:13])
    frames.append(current[13:20])
    return np.concatenate(frames, axis=0)


def compact_to_tensor(compact_board, turn=chess.WHITE, flip_perspective=False, dtype=np.float32):
    """Encode the existing 38-byte compact board record.

    The record contains exact castling rights, en-passant square and halfmove
    clock, but no game-history provenance. Its repetition plane therefore stays
    unknown (-1), even when a clock value might tempt an inference.
    """
    if len(compact_board) != 38:
        raise ValueError(f"Invalid compact board size: {len(compact_board)} (expected 38)")
    if turn not in (chess.WHITE, chess.BLACK):
        raise ValueError(f"turn must be chess.WHITE or chess.BLACK, got {turn!r}")
    code_to_piece = {
        1: (chess.WHITE, chess.PAWN),
        2: (chess.WHITE, chess.KNIGHT),
        3: (chess.WHITE, chess.BISHOP),
        4: (chess.WHITE, chess.ROOK),
        5: (chess.WHITE, chess.QUEEN),
        6: (chess.WHITE, chess.KING),
        7: (chess.BLACK, chess.PAWN),
        8: (chess.BLACK, chess.KNIGHT),
        9: (chess.BLACK, chess.BISHOP),
        10: (chess.BLACK, chess.ROOK),
        11: (chess.BLACK, chess.QUEEN),
        12: (chess.BLACK, chess.KING),
    }
    board = chess.empty_board()
    for pair_idx in range(32):
        byte = compact_board[pair_idx]
        for offset, code in enumerate(((byte >> 4) & 0x0F, byte & 0x0F)):
            piece_spec = code_to_piece.get(code)
            if piece_spec is not None:
                board[chess.square_from_index(pair_idx * 2 + offset)] = chess.Piece(*piece_spec)
            elif code != 0:
                raise ValueError(f"Invalid piece code {code} in compact board")
    castling_byte = compact_board[32]
    if castling_byte & 0xF0:
        raise ValueError(f"Invalid castling flags 0x{castling_byte:02x}")
    board.castling_rights = chess.castling_rights_from_flags(castling_byte)
    board.turn = turn
    ep_square = compact_board[33]
    if ep_square != 255 and ep_square > 63:
        raise ValueError(f"Invalid en-passant square index {ep_square}")
    board.en_passant_square = None if ep_square == 255 else chess.square_from_index(ep_square)
    board.halfmove_clock = int.from_bytes(compact_board[34:36], "big")
    board.fullmove_number = int.from_bytes(compact_board[36:38], "big")
    return board_to_tensor(board, flip_perspective=flip_perspective, dtype=dtype)

