"""
Data processing helper functions for chess AI
v4.3: POV (Point of View) + Dynamic Sliding Window
- 🎯 POV: All boards from perspective of current player
- 🔄 Sliding Window: Dynamic history assembly using mmap
- 🎮 GameID tracking: Track games for history reconstruction
- Value targets use final W/D/L outcomes from side-to-move POV.
"""

from src.game import backend as chess
import numpy as np
import struct
from functools import lru_cache


# ==============================================================================
# 🆕 TEMPORAL VALUE DISCOUNTING - FIX FOR MAE = 0.8
# ==============================================================================

def compute_discounted_outcome(result, current_turn):
    """
    Return final outcome from the side-to-move POV.

    Args:
        result: Game result ('1-0', '0-1', '1/2-1/2').
        current_turn: Side to move (chess.WHITE or chess.BLACK).

    Returns:
        float: Outcome in {-1.0, 0.0, 1.0}.
    """
    if result == '1-0':
        return 1.0 if current_turn == chess.WHITE else -1.0
    if result == '0-1':
        return -1.0 if current_turn == chess.WHITE else 1.0
    return 0.0


# ==============================================================================
# POV (POINT OF VIEW) BOARD REPRESENTATION
# ==============================================================================

_PIECE_TYPES_WITH_INDEX = (
    (chess.PAWN, 0),
    (chess.KNIGHT, 1),
    (chess.BISHOP, 2),
    (chess.ROOK, 3),
    (chess.QUEEN, 4),
    (chess.KING, 5),
)


def _canonical_piece_planes(board):
    """Return white pieces followed by black pieces in board orientation."""
    masks = np.asarray(
        [
            chess.piece_mask(board, piece_type, color)
            for color in (chess.WHITE, chess.BLACK)
            for piece_type, _piece_idx in _PIECE_TYPES_WITH_INDEX
        ],
        dtype='<u8',
    )
    return np.unpackbits(
        masks.view(np.uint8),
        bitorder='little',
    ).reshape(12, 8, 8)


def _fill_metadata_planes(tensor, board, pov_color, should_flip):
    tensor[12:14].fill(0.0)

    kingside = chess.has_kingside_castling_rights(board, pov_color)
    queenside = chess.has_queenside_castling_rights(board, pov_color)
    if kingside or queenside:
        king_sq = chess.king_square(board, pov_color)
        if king_sq is not None:
            king_sq = chess.square_index(king_sq)
            king_row, king_col = divmod(king_sq, 8)
            if should_flip:
                king_row, king_col = 7 - king_row, 7 - king_col
            tensor[12, king_row, king_col] = 1.0
            if kingside:
                tensor[12, king_row, 0 if should_flip else 7] = 1.0
            if queenside:
                tensor[12, king_row, 7 if should_flip else 0] = 1.0

    if board.en_passant_square is not None:
        ep_square = chess.square_index(board.en_passant_square)
        ep_row, ep_col = divmod(ep_square, 8)
        if should_flip:
            ep_row, ep_col = 7 - ep_row, 7 - ep_col
        tensor[13, ep_row, ep_col] = 1.0

    tensor[14].fill(min(board.halfmove_clock / 50.0, 1.0))
    tensor[15].fill(min(board.fullmove_number / 100.0, 1.0))


def board_to_tensor(board, flip_perspective=None, dtype=np.float32, out=None):
    """
    Convert chess.Board to tensor representation with POV (Point of View)
    
    🆕 v4.5: EXTENDED WITH CHESS METADATA (16 planes total)
    
    POV System:
    - Channels 0-5: Current player's pieces (White if white to move, Black if black to move)
    - Channels 6-11: Opponent's pieces
    - Channels 12-15: Chess metadata (NEW!)
        - Channel 12: Castling rights (1.0 where king/rook can castle)
        - Channel 13: En passant square (1.0 at target square)
        - Channel 14: Halfmove clock (normalized 0.0-1.0, scaled by 50)
        - Channel 15: Fullmove number (normalized 0.0-1.0, scaled by 100)
    - Board orientation: Always from current player's perspective
    
    Args:
        board: Current chess.Board
        flip_perspective: Override automatic flip (for history boards)
                         If None, auto-detect from board.turn
                         If True, flip (for black's perspective)
                         If False, don't flip (for white's perspective)
        dtype: Output dtype. Self-play history uses float16 to avoid creating
               a temporary float32 tensor only to cast it immediately.
        out: Optional preallocated ``(16, 8, 8)`` NumPy destination. When
             supplied, its dtype is preserved and no per-position output
             allocation is performed.
    
    Returns: 
        (16, 8, 8) tensor from current player's perspective (was 12, now 16)
    """
    # Determine if we need to flip
    if flip_perspective is None:
        should_flip = (board.turn == chess.BLACK)
    else:
        should_flip = flip_perspective
    canonical = _canonical_piece_planes(board)
    if out is None:
        tensor = np.empty((16, 8, 8), dtype=dtype)
    else:
        tensor = np.asarray(out)
        if tensor.shape != (16, 8, 8):
            raise ValueError(
                f"board_to_tensor out must have shape (16, 8, 8), got {tensor.shape}"
            )
    if should_flip:
        tensor[:6] = canonical[6:12, ::-1, ::-1]
        tensor[6:12] = canonical[:6, ::-1, ::-1]
        pov_color = chess.BLACK
    else:
        tensor[:12] = canonical
        pov_color = chess.WHITE
    _fill_metadata_planes(tensor, board, pov_color, should_flip)
    return tensor


def board_to_tensor_pair(board, dtype=np.float32):
    """Encode both POVs while reading and unpacking native bitboards once."""
    canonical = _canonical_piece_planes(board)
    white = np.empty((16, 8, 8), dtype=dtype)
    black = np.empty((16, 8, 8), dtype=dtype)
    white[:12] = canonical
    black[:6] = canonical[6:12, ::-1, ::-1]
    black[6:12] = canonical[:6, ::-1, ::-1]
    _fill_metadata_planes(white, board, chess.WHITE, False)
    _fill_metadata_planes(black, board, chess.BLACK, True)
    return white, black


_COMPACT_PIECE_SPECS = (
    (chess.PAWN, chess.WHITE, 1),
    (chess.KNIGHT, chess.WHITE, 2),
    (chess.BISHOP, chess.WHITE, 3),
    (chess.ROOK, chess.WHITE, 4),
    (chess.QUEEN, chess.WHITE, 5),
    (chess.KING, chess.WHITE, 6),
    (chess.PAWN, chess.BLACK, 7),
    (chess.KNIGHT, chess.BLACK, 8),
    (chess.BISHOP, chess.BLACK, 9),
    (chess.ROOK, chess.BLACK, 10),
    (chess.QUEEN, chess.BLACK, 11),
    (chess.KING, chess.BLACK, 12),
)


def board_to_compact(board):
    """
    Convert board to ultra-compact binary representation
    
    🆕 v4.5: EXTENDED FORMAT (38 bytes total)
    - 32 bytes: pieces (64 squares × 4 bits)
    - 1 byte: castling rights (4 bits: K, Q, k, q)
    - 1 byte: en passant square (0-63, 255=none)
    - 2 bytes: halfmove clock (uint16)
    - 2 bytes: fullmove number (uint16)
    
    NOTE: Stores board in ORIGINAL orientation (not POV)
    POV conversion happens at tensor conversion time
    """
    # Fill nibbles directly from native bitboards.  This avoids building
    # a 64-entry piece map and then looping over all 64 squares for every ply.
    packed = bytearray(38)
    for piece_type, color, code in _COMPACT_PIECE_SPECS:
        bitboard = chess.piece_mask(board, piece_type, color)
        while bitboard:
            lowest_bit = bitboard & -bitboard
            square = lowest_bit.bit_length() - 1
            byte_index = square >> 1
            if square & 1:
                packed[byte_index] |= code
            else:
                packed[byte_index] |= code << 4
            bitboard ^= lowest_bit

    # Byte 32: castling rights (K, Q, k, q).
    castling_byte = 0
    if chess.has_kingside_castling_rights(board, chess.WHITE):
        castling_byte |= 0b1000  # K
    if chess.has_queenside_castling_rights(board, chess.WHITE):
        castling_byte |= 0b0100  # Q
    if chess.has_kingside_castling_rights(board, chess.BLACK):
        castling_byte |= 0b0010  # k
    if chess.has_queenside_castling_rights(board, chess.BLACK):
        castling_byte |= 0b0001  # q
    packed[32] = castling_byte
    packed[33] = (
        chess.square_index(board.en_passant_square)
        if board.en_passant_square is not None
        else 255
    )
    struct.pack_into('>HH', packed, 34, board.halfmove_clock, board.fullmove_number)
    
    return bytes(packed)


def compact_to_board(compact_board, turn=chess.WHITE):
    """Reconstruct a native bulletchess board from the compact encoding."""
    if len(compact_board) != 38:
        raise ValueError(f"Invalid compact board size: {len(compact_board)} (expected 38)")

    code_to_piece = {
        1: chess.Piece(chess.WHITE, chess.PAWN),
        2: chess.Piece(chess.WHITE, chess.KNIGHT),
        3: chess.Piece(chess.WHITE, chess.BISHOP),
        4: chess.Piece(chess.WHITE, chess.ROOK),
        5: chess.Piece(chess.WHITE, chess.QUEEN),
        6: chess.Piece(chess.WHITE, chess.KING),
        7: chess.Piece(chess.BLACK, chess.PAWN),
        8: chess.Piece(chess.BLACK, chess.KNIGHT),
        9: chess.Piece(chess.BLACK, chess.BISHOP),
        10: chess.Piece(chess.BLACK, chess.ROOK),
        11: chess.Piece(chess.BLACK, chess.QUEEN),
        12: chess.Piece(chess.BLACK, chess.KING),
    }

    board = chess.empty_board()
    for i in range(32):
        byte = compact_board[i]
        codes = ((byte >> 4) & 0x0F, byte & 0x0F)
        for offset, code in enumerate(codes):
            piece = code_to_piece.get(code)
            if piece is not None:
                board[chess.square_from_index(i * 2 + offset)] = piece

    castling_byte = compact_board[32]
    board.castling_rights = chess.castling_rights_from_flags(castling_byte)

    # bulletchess validates an en-passant square against the side to move, so
    # turn must be restored before assigning that square.
    board.turn = turn
    ep_square = compact_board[33]
    board.en_passant_square = None if ep_square == 255 else chess.square_from_index(ep_square)
    board.halfmove_clock = int(struct.unpack('>H', compact_board[34:36])[0])
    board.fullmove_number = int(struct.unpack('>H', compact_board[36:38])[0])
    return board


def compact_to_tensor(compact_board, flip_perspective=False):
    """
    Convert compact representation back to tensor with POV support
    
    🆕 v4.5: EXTENDED FORMAT (38 bytes → 16 planes)
    
    Args:
        compact_board: 38-byte compact representation (32B pieces + 6B metadata)
        flip_perspective: If True, flip board for black's perspective
    
    Returns:
        (16, 8, 8) tensor with metadata planes
    """
    tensor = np.zeros((16, 8, 8), dtype=np.float32)
    
    # Decode piece codes
    code_to_piece = {
        1: (chess.PAWN, chess.WHITE),
        2: (chess.KNIGHT, chess.WHITE),
        3: (chess.BISHOP, chess.WHITE),
        4: (chess.ROOK, chess.WHITE),
        5: (chess.QUEEN, chess.WHITE),
        6: (chess.KING, chess.WHITE),
        7: (chess.PAWN, chess.BLACK),
        8: (chess.KNIGHT, chess.BLACK),
        9: (chess.BISHOP, chess.BLACK),
        10: (chess.ROOK, chess.BLACK),
        11: (chess.QUEEN, chess.BLACK),
        12: (chess.KING, chess.BLACK),
    }
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # === PIECE PLANES (0-11) ===
    for i in range(32):
        byte = compact_board[i]
        code1 = (byte >> 4) & 0x0F
        code2 = byte & 0x0F
        
        square1 = i * 2
        square2 = i * 2 + 1
        
        for square, code in [(square1, code1), (square2, code2)]:
            if code == 0:
                continue
            
            piece_type, piece_color = code_to_piece[code]
            
            # Get original coordinates
            row = square // 8
            col = square % 8
            
            # Flip if needed
            if flip_perspective:
                row = 7 - row
                col = 7 - col
            
            # Get piece index
            piece_idx = piece_to_idx[piece_type]
            
            # Determine channel based on POV
            if flip_perspective:
                # Black's perspective
                if piece_color == chess.BLACK:
                    channel = piece_idx  # Current player
                else:
                    channel = piece_idx + 6  # Opponent
            else:
                # White's perspective
                if piece_color == chess.WHITE:
                    channel = piece_idx  # Current player
                else:
                    channel = piece_idx + 6  # Opponent
            
            tensor[channel, row, col] = 1.0
    
    # === METADATA PLANES (12-15) ===
    
    # Byte 32: Castling rights
    castling_byte = compact_board[32]
    has_K = bool(castling_byte & 0b1000)
    has_Q = bool(castling_byte & 0b0100)
    has_k = bool(castling_byte & 0b0010)
    has_q = bool(castling_byte & 0b0001)
    
    # Channel 12: Castling rights - precise squares (matching board_to_tensor)
    # 🔧 v4.4 FIX: Use exact king/rook squares, not entire back rank!
    # Standard chess: King on e-file (col 4), Rooks on a/h-files (col 0/7)
    # In POV, always row 0 (current player's back rank)
    
    if flip_perspective:
        # Black's perspective
        # After flip: e8 (4,7) -> (3,0), a8 (0,7) -> (7,0), h8 (7,7) -> (0,0)
        # Kingside (black)
        if has_k:
            tensor[12, 0, 3] = 1.0  # King (e8 flipped)
            tensor[12, 0, 0] = 1.0  # Rook (h8 flipped)
        # Queenside (black)
        if has_q:
            tensor[12, 0, 3] = 1.0  # King (e8 flipped)
            tensor[12, 0, 7] = 1.0  # Rook (a8 flipped)
    else:
        # White's perspective
        # Kingside (white)
        if has_K:
            tensor[12, 0, 4] = 1.0  # King (e1)
            tensor[12, 0, 7] = 1.0  # Rook (h1)
        # Queenside (white)
        if has_Q:
            tensor[12, 0, 4] = 1.0  # King (e1)
            tensor[12, 0, 0] = 1.0  # Rook (a1)
    
    # Byte 33: En passant square
    ep_square = compact_board[33]
    if ep_square != 255:
        ep_row, ep_col = ep_square // 8, ep_square % 8
        if flip_perspective:
            ep_row, ep_col = 7 - ep_row, 7 - ep_col
        tensor[13, ep_row, ep_col] = 1.0
    
    # Bytes 34-35: Halfmove clock
    halfmove_clock = struct.unpack('>H', compact_board[34:36])[0]
    halfmove_normalized = min(halfmove_clock / 50.0, 1.0)
    tensor[14, :, :] = halfmove_normalized

    # Bytes 36-37: Fullmove number
    fullmove_number = struct.unpack('>H', compact_board[36:38])[0]
    fullmove_normalized = min(fullmove_number / 100.0, 1.0)
    tensor[15, :, :] = fullmove_normalized
    
    return tensor


# ==============================================================================
# POV-AWARE MOVE ENCODING
# ==============================================================================

ACTION_PLANES = 73
AZ_ACTION_SIZE = 64 * ACTION_PLANES  # 4672 intermediate 8x8x73 plane space
ACTION_SIZE = 1858                   # LC0-style compact policy space
MAX_LEGAL_MOVES = 256                # Safe fixed padding for legal-only policy loss

# Queen-like directions: N, NE, E, SE, S, SW, W, NW
_QUEENLIKE_DIRECTIONS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
_QUEENLIKE_DIRECTION_TO_INDEX = {
    direction: idx for idx, direction in enumerate(_QUEENLIKE_DIRECTIONS)
}

# Knight offsets in POV coordinates
_KNIGHT_DELTAS = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)
_KNIGHT_DELTA_TO_INDEX = {delta: idx for idx, delta in enumerate(_KNIGHT_DELTAS)}

# Underpromotions: piece-major order, each with [capture-left, forward, capture-right]
_UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMOTION_PIECE_TO_INDEX = {
    piece: idx for idx, piece in enumerate(_UNDERPROMOTION_PIECES)
}
_UNDERPROMOTION_DELTAS = (
    (1, -1),
    (1, 0),
    (1, 1),
)
_UNDERPROMOTION_DELTA_TO_INDEX = {
    delta: idx for idx, delta in enumerate(_UNDERPROMOTION_DELTAS)
}

_HFLIP_INV_INDEX_MAP = None
_POLICY_INDEX_TO_AZ_INDEX = None
_AZ_INDEX_TO_POLICY_INDEX = None


def _to_pov_square(square, is_black_turn):
    return square ^ 63 if is_black_turn else square


def _from_pov_square(square, is_black_turn):
    return square ^ 63 if is_black_turn else square


def _square_to_coords(square):
    return square // 8, square % 8


def _coords_to_square(row, col):
    if row < 0 or row > 7 or col < 0 or col > 7:
        return None
    return row * 8 + col


def _encode_queenlike_plane(dr, dc):
    if dr == 0 and dc == 0:
        return None

    abs_dr = abs(dr)
    abs_dc = abs(dc)
    if not (dr == 0 or dc == 0 or abs_dr == abs_dc):
        return None

    distance = max(abs_dr, abs_dc)
    if distance < 1 or distance > 7:
        return None

    step = (
        0 if dr == 0 else (1 if dr > 0 else -1),
        0 if dc == 0 else (1 if dc > 0 else -1),
    )
    direction_idx = _QUEENLIKE_DIRECTION_TO_INDEX.get(step)
    if direction_idx is None:
        return None
    return direction_idx * 7 + (distance - 1)


def _decode_queenlike_plane(plane):
    direction_idx = plane // 7
    distance = (plane % 7) + 1
    dr, dc = _QUEENLIKE_DIRECTIONS[direction_idx]
    return dr * distance, dc * distance


def _is_valid_az_policy_index(index):
    """Return True for AZ plane entries that correspond to a possible chess move."""
    if index < 0 or index >= AZ_ACTION_SIZE:
        return False

    from_square = index // ACTION_PLANES
    plane = index % ACTION_PLANES
    from_row, from_col = _square_to_coords(from_square)

    if plane < 56:
        dr, dc = _decode_queenlike_plane(plane)
        return _coords_to_square(from_row + dr, from_col + dc) is not None

    if plane < 64:
        dr, dc = _KNIGHT_DELTAS[plane - 56]
        return _coords_to_square(from_row + dr, from_col + dc) is not None

    if plane < ACTION_PLANES:
        dir_idx = (plane - 64) % 3
        dr, dc = _UNDERPROMOTION_DELTAS[dir_idx]
        # In POV coordinates all promotions move from row 6 to row 7.
        return from_row == 6 and _coords_to_square(from_row + dr, from_col + dc) is not None

    return False


def _build_policy_maps():
    policy_to_az = [idx for idx in range(AZ_ACTION_SIZE) if _is_valid_az_policy_index(idx)]
    if len(policy_to_az) != ACTION_SIZE:
        raise RuntimeError(f"Expected {ACTION_SIZE} compact policy moves, got {len(policy_to_az)}")
    az_to_policy = np.full(AZ_ACTION_SIZE, -1, dtype=np.int32)
    for policy_idx, az_idx in enumerate(policy_to_az):
        az_to_policy[int(az_idx)] = int(policy_idx)
    return np.asarray(policy_to_az, dtype=np.int64), az_to_policy


def get_policy_index_maps():
    global _POLICY_INDEX_TO_AZ_INDEX, _AZ_INDEX_TO_POLICY_INDEX
    if _POLICY_INDEX_TO_AZ_INDEX is None or _AZ_INDEX_TO_POLICY_INDEX is None:
        _POLICY_INDEX_TO_AZ_INDEX, _AZ_INDEX_TO_POLICY_INDEX = _build_policy_maps()
    return _POLICY_INDEX_TO_AZ_INDEX, _AZ_INDEX_TO_POLICY_INDEX


def policy_index_to_az_index(index):
    policy_to_az, _ = get_policy_index_maps()
    index = int(index)
    if index < 0 or index >= ACTION_SIZE:
        raise ValueError(f"Policy index out of range: {index} (expected 0-{ACTION_SIZE - 1})")
    return int(policy_to_az[index])


def az_index_to_policy_index(index):
    _, az_to_policy = get_policy_index_maps()
    index = int(index)
    if index < 0 or index >= AZ_ACTION_SIZE:
        raise ValueError(f"AZ index out of range: {index} (expected 0-{AZ_ACTION_SIZE - 1})")
    policy_index = int(az_to_policy[index])
    if policy_index < 0:
        raise ValueError(f"AZ index {index} does not correspond to a compact policy move")
    return policy_index


def move_to_index(move, board):
    """
    Convert chess.Move to compact LC0-style policy index with POV rotation.
    """
    is_black_turn = (board.turn == chess.BLACK)
    az_index = _move_to_az_index_cached(
        chess.move_origin_index(move),
        chess.move_destination_index(move),
        move.promotion or 0,
        is_black_turn,
    )
    return az_index_to_policy_index(az_index)


@lru_cache(maxsize=65536)
def _move_to_index_cached(from_square_raw, to_square_raw, promotion, is_black_turn):
    az_index = _move_to_az_index_cached(from_square_raw, to_square_raw, promotion, is_black_turn)
    return az_index_to_policy_index(az_index)


@lru_cache(maxsize=65536)
def _move_to_az_index_cached(from_square_raw, to_square_raw, promotion, is_black_turn):
    from_square = _to_pov_square(from_square_raw, is_black_turn)
    to_square = _to_pov_square(to_square_raw, is_black_turn)
    promotion = promotion or None

    from_row, from_col = _square_to_coords(from_square)
    to_row, to_col = _square_to_coords(to_square)
    dr = to_row - from_row
    dc = to_col - from_col

    # Underpromotions use dedicated planes.
    if promotion in _UNDERPROMOTION_PIECE_TO_INDEX:
        piece_idx = _UNDERPROMOTION_PIECE_TO_INDEX[promotion]
        dir_idx = _UNDERPROMOTION_DELTA_TO_INDEX.get((dr, dc))
        if dir_idx is None:
            raise ValueError(
                f"Unsupported underpromotion delta: {(dr, dc)} for move "
                f"({from_square_raw}->{to_square_raw}, promotion={promotion})"
            )
        plane = 64 + piece_idx * 3 + dir_idx
        return from_square * ACTION_PLANES + plane

    # Queen-like moves (includes queen promotions by design).
    plane = _encode_queenlike_plane(dr, dc)
    if plane is not None:
        return from_square * ACTION_PLANES + plane

    # Knight moves.
    knight_idx = _KNIGHT_DELTA_TO_INDEX.get((dr, dc))
    if knight_idx is not None:
        return from_square * ACTION_PLANES + (56 + knight_idx)

    raise ValueError(
        f"Unsupported move for AZ action encoding: "
        f"({from_square_raw}->{to_square_raw}, promotion={promotion}) (delta={(dr, dc)})"
    )


def index_to_move(index, is_black_turn=False, board=None):
    """
    Convert compact LC0-style policy index back to chess.Move.

    Args:
        index: Move index in [0, ACTION_SIZE).
        is_black_turn: Whether current player to move is black.
        board: Optional board; if provided, queen promotions are reconstructed when applicable.
    """
    if index < 0 or index >= ACTION_SIZE:
        return None
    index = policy_index_to_az_index(index)

    from_square_pov = index // ACTION_PLANES
    plane = index % ACTION_PLANES
    from_row, from_col = _square_to_coords(from_square_pov)

    if plane < 56:
        dr, dc = _decode_queenlike_plane(plane)
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = None
    elif plane < 64:
        dr, dc = _KNIGHT_DELTAS[plane - 56]
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = None
    else:
        promo_plane = plane - 64
        piece_idx = promo_plane // 3
        dir_idx = promo_plane % 3
        if piece_idx < 0 or piece_idx >= len(_UNDERPROMOTION_PIECES):
            return None
        dr, dc = _UNDERPROMOTION_DELTAS[dir_idx]
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = _UNDERPROMOTION_PIECES[piece_idx]

    to_square_pov = _coords_to_square(to_row, to_col)
    if to_square_pov is None:
        return None

    from_square = _from_pov_square(from_square_pov, is_black_turn)
    to_square = _from_pov_square(to_square_pov, is_black_turn)

    # Queen promotions are encoded in queen-like planes; reconstruct when board is available.
    if promotion is None and board is not None:
        piece = chess.piece_at(board, from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if to_rank == 0 or to_rank == 7:
                promotion = chess.QUEEN

    return chess.new_move(from_square, to_square, promotion=promotion)


def _mirror_file_pov_square(square):
    rank, file = _square_to_coords(square)
    return rank * 8 + (7 - file)


def build_hflip_inverse_index_map():
    """
    Build inverse action index map for horizontal file flip (a<->h) in POV space.

    Returns:
        np.ndarray[int64]: inverse map with shape (ACTION_SIZE,).
    """
    global _HFLIP_INV_INDEX_MAP
    if _HFLIP_INV_INDEX_MAP is not None:
        return _HFLIP_INV_INDEX_MAP

    policy_to_az, az_to_policy = get_policy_index_maps()
    forward_map = np.empty(ACTION_SIZE, dtype=np.int64)

    for idx, az_idx in enumerate(policy_to_az):
        from_square = int(az_idx) // ACTION_PLANES
        plane = int(az_idx) % ACTION_PLANES
        mirrored_from = _mirror_file_pov_square(from_square)

        if plane < 56:
            direction_idx = plane // 7
            distance = (plane % 7) + 1
            dr, dc = _QUEENLIKE_DIRECTIONS[direction_idx]
            mirrored_direction_idx = _QUEENLIKE_DIRECTION_TO_INDEX[(dr, -dc)]
            mirrored_plane = mirrored_direction_idx * 7 + (distance - 1)
        elif plane < 64:
            knight_idx = plane - 56
            dr, dc = _KNIGHT_DELTAS[knight_idx]
            mirrored_knight_idx = _KNIGHT_DELTA_TO_INDEX[(dr, -dc)]
            mirrored_plane = 56 + mirrored_knight_idx
        else:
            promo_plane = plane - 64
            piece_idx = promo_plane // 3
            dir_idx = promo_plane % 3
            dr, dc = _UNDERPROMOTION_DELTAS[dir_idx]
            mirrored_dir_idx = _UNDERPROMOTION_DELTA_TO_INDEX[(dr, -dc)]
            mirrored_plane = 64 + piece_idx * 3 + mirrored_dir_idx

        mirrored_az = mirrored_from * ACTION_PLANES + mirrored_plane
        mirrored_policy = int(az_to_policy[mirrored_az])
        if mirrored_policy < 0:
            raise RuntimeError(f"Mirrored AZ index {mirrored_az} is not in compact policy map")
        forward_map[idx] = mirrored_policy

    inverse_map = np.empty_like(forward_map)
    inverse_map[forward_map] = np.arange(ACTION_SIZE, dtype=np.int64)
    _HFLIP_INV_INDEX_MAP = inverse_map
    return _HFLIP_INV_INDEX_MAP

# ==============================================================================
# AUXILIARY TASK HELPERS
# ==============================================================================

def compute_material_balance(board):
    """
    Compute material balance from current player's perspective
    
    Returns:
        float: Material balance normalized to [-1, 1]
               Positive = current player ahead, Negative = behind
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = sum(
        value * len(board[chess.WHITE, piece_type])
        for piece_type, value in piece_values.items()
    )
    black_material = sum(
        value * len(board[chess.BLACK, piece_type])
        for piece_type, value in piece_values.items()
    )
    
    # From current player's perspective
    if board.turn == chess.WHITE:
        balance = white_material - black_material
    else:
        balance = black_material - white_material
    
    # Normalize to [-1, 1] (divide by max possible material ~40)
    return np.tanh(balance / 10.0)


def is_in_check(board):
    """
    Check if current player's king is in check
    
    Returns:
        float: 1.0 if in check, 0.0 otherwise
    """
    return 1.0 if chess.is_check(board) else 0.0


def will_win(board, game_result):
    """
    Determine if current player will win based on game result
    
    Args:
        board: chess.Board at current position
        game_result: Game result string ('1-0', '0-1', '1/2-1/2')
    
    Returns:
        float: 1.0 if current player wins, 0.0 otherwise
    """
    if game_result == '1/2-1/2':
        return 0.0
    
    if board.turn == chess.WHITE:
        return 1.0 if game_result == '1-0' else 0.0
    else:
        return 1.0 if game_result == '0-1' else 0.0


def get_turn_from_move_idx(move_idx):
    """
    Determine whose turn it is from move index
    
    Args:
        move_idx: 0-based move index
    
    Returns:
        chess.WHITE or chess.BLACK
    """
    return chess.WHITE if move_idx % 2 == 0 else chess.BLACK


# ==============================================================================
# BINARY FORMAT HELPERS - FIXED TO INCLUDE move_target
# ==============================================================================

def get_position_size(history_positions=0):
    """
    Calculate size of binary position record
    
    🆕 v4.4 FORMAT (Extended with metadata):
    [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [ActorElo (2B)]
    
    Board format changed: 32B pieces + 6B metadata (castling, en passant, halfmove, fullmove)
    
    Args:
        history_positions: Number of history positions (NOT used in new format)
    
    Returns:
        int: Size in bytes
    """
    base_size = 38  # Board (compact) - 🆕 NOW 38 bytes instead of 32!
    base_size += 4  # GameID (uint32) — supports up to ~4 billion unique games
    base_size += 2  # MoveIdx (uint16)
    base_size += 2  # MoveTarget (uint16) - the move label (0-4671)
    base_size += 4  # Outcome (float32)
    base_size += 2  # ActorElo (uint16) - player who selected MoveTarget
    
    return base_size


def pack_position_data(board, game_id, move_idx, move_target, outcome, actor_elo=0):
    """
    Pack position data into binary format
    
    v4.4 FORMAT (Extended with metadata):
    [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [ActorElo (2B)]
    
    Args:
        board: chess.Board
        game_id: Unique game identifier (0 – 4294967295, uint32)
        move_idx: Move index in game (0-based)
        move_target: Target move index (0-4671) - THIS IS THE LABEL
        outcome: Game outcome value
    
    Returns:
        bytes: Packed binary data
    """
    data = bytearray()
    
    # Pack board (38 bytes - INCLUDES METADATA!)
    data.extend(board_to_compact(board))
    
    # Pack metadata
    data.extend(struct.pack('I', game_id))          # GameID (4 bytes, uint32)
    data.extend(struct.pack('H', move_idx))         # MoveIdx (2 bytes)
    data.extend(struct.pack('H', move_target))      # MoveTarget (2 bytes)
    data.extend(struct.pack('f', outcome))          # Outcome (4 bytes)
    data.extend(struct.pack('H', max(0, min(65535, int(actor_elo or 0)))))
    
    return bytes(data)


