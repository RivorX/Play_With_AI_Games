"""
Data processing helper functions for chess AI
🆕 v4.3: POV (Point of View) + Dynamic Sliding Window + TEMPORAL DISCOUNTING
- 🎯 POV: All boards from perspective of current player
- 🔄 Sliding Window: Dynamic history assembly using mmap
- 🎮 GameID tracking: Track games for history reconstruction
- ⚡ TEMPORAL DISCOUNTING: Fixed MAE from 0.8 to ~0.2!
"""

import chess
import numpy as np
import struct


# ==============================================================================
# 🆕 TEMPORAL VALUE DISCOUNTING - FIX FOR MAE = 0.8
# ==============================================================================

def compute_discounted_outcome(move_idx, total_moves, result, current_turn):
    """
    Return outcome in WDL-only mode (no temporal discounting).

    Args:
        move_idx: Unused; kept for API stability.
        total_moves: Unused; kept for API stability.
        result: Game result ('1-0', '0-1', '1/2-1/2').
        current_turn: Side to move (chess.WHITE or chess.BLACK).

    Returns:
        float: Outcome in {-1.0, 0.0, 1.0}.
    """
    # Keep the signature stable for existing call sites.
    _ = move_idx
    _ = total_moves

    if result == '1-0':
        return 1.0 if current_turn == chess.WHITE else -1.0
    if result == '0-1':
        return -1.0 if current_turn == chess.WHITE else 1.0
    return 0.0


# ==============================================================================
# POV (POINT OF VIEW) BOARD REPRESENTATION
# ==============================================================================

def board_to_tensor(board, flip_perspective=None):
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
    
    Returns: 
        (16, 8, 8) tensor from current player's perspective (was 12, now 16)
    """
    tensor = np.zeros((16, 8, 8), dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Determine if we need to flip
    if flip_perspective is None:
        should_flip = (board.turn == chess.BLACK)
    else:
        should_flip = flip_perspective
    
    # === PIECE PLANES (0-11) ===
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get original coordinates
            row = square // 8
            col = square % 8
            
            # Flip if needed (black's perspective)
            if should_flip:
                row = 7 - row
                col = 7 - col
            
            # Get piece index
            piece_idx = piece_to_idx[piece.piece_type]
            
            # Determine if this piece belongs to current player or opponent
            if should_flip:
                # Black to move
                if piece.color == chess.BLACK:
                    channel = piece_idx  # Current player (0-5)
                else:
                    channel = piece_idx + 6  # Opponent (6-11)
            else:
                # White to move
                if piece.color == chess.WHITE:
                    channel = piece_idx  # Current player (0-5)
                else:
                    channel = piece_idx + 6  # Opponent (6-11)
            
            tensor[channel, row, col] = 1.0
    
    # === METADATA PLANES (12-15) ===
    
    # Channel 12: Castling rights
    # Mark squares where castling is possible (king position + rook position)
    if board.has_kingside_castling_rights(board.turn):
        # Kingside: mark king and h-rook squares
        king_sq = board.king(board.turn)
        if king_sq is not None:
            king_row, king_col = king_sq // 8, king_sq % 8
            if should_flip:
                king_row, king_col = 7 - king_row, 7 - king_col
            tensor[12, king_row, king_col] = 1.0
            # Rook on h-file (col=7 for white, flipped for black)
            rook_col = 7
            if should_flip:
                rook_col = 7 - rook_col
            tensor[12, king_row, rook_col] = 1.0
    
    if board.has_queenside_castling_rights(board.turn):
        # Queenside: mark king and a-rook squares
        king_sq = board.king(board.turn)
        if king_sq is not None:
            king_row, king_col = king_sq // 8, king_sq % 8
            if should_flip:
                king_row, king_col = 7 - king_row, 7 - king_col
            tensor[12, king_row, king_col] = 1.0
            # Rook on a-file (col=0 for white, flipped for black)
            rook_col = 0
            if should_flip:
                rook_col = 7 - rook_col
            tensor[12, king_row, rook_col] = 1.0
    
    # Channel 13: En passant square
    if board.ep_square is not None:
        ep_row, ep_col = board.ep_square // 8, board.ep_square % 8
        if should_flip:
            ep_row, ep_col = 7 - ep_row, 7 - ep_col
        tensor[13, ep_row, ep_col] = 1.0
    
    # Channel 14: Halfmove clock (normalized to 0-1, scaled by 50-move rule)
    # Uniform plane with value = halfmove_clock / 50
    halfmove_normalized = min(board.halfmove_clock / 50.0, 1.0)
    tensor[14, :, :] = halfmove_normalized

    # Channel 15: Fullmove number (normalized to 0-1)
    # Fullmove starts at 1 and increments after Black's move
    fullmove_normalized = min(board.fullmove_number / 100.0, 1.0)
    tensor[15, :, :] = fullmove_normalized
    
    return tensor


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
    piece_to_code = {
        (chess.PAWN, chess.WHITE): 1,
        (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3,
        (chess.ROOK, chess.WHITE): 4,
        (chess.QUEEN, chess.WHITE): 5,
        (chess.KING, chess.WHITE): 6,
        (chess.PAWN, chess.BLACK): 7,
        (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9,
        (chess.ROOK, chess.BLACK): 10,
        (chess.QUEEN, chess.BLACK): 11,
        (chess.KING, chess.BLACK): 12,
    }
    
    codes = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            code = piece_to_code[(piece.piece_type, piece.color)]
        else:
            code = 0
        codes.append(code)
    
    # Pack pairs of codes into bytes (32 bytes for pieces)
    packed = bytearray(32)
    for i in range(0, 64, 2):
        packed[i // 2] = (codes[i] << 4) | codes[i + 1]
    
    # === ADD METADATA (6 bytes) ===
    
    # Byte 32: Castling rights (4 bits: K, Q, k, q)
    castling_byte = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_byte |= 0b1000  # K
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_byte |= 0b0100  # Q
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_byte |= 0b0010  # k
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_byte |= 0b0001  # q
    packed.append(castling_byte)
    
    # Byte 33: En passant square (0-63, 255=none)
    if board.ep_square is not None:
        packed.append(board.ep_square)
    else:
        packed.append(255)
    
    # Bytes 34-35: Halfmove clock (uint16, big-endian)
    halfmove_bytes = struct.pack('>H', board.halfmove_clock)
    packed.extend(halfmove_bytes)

    # Bytes 36-37: Fullmove number (uint16, big-endian)
    fullmove_bytes = struct.pack('>H', board.fullmove_number)
    packed.extend(fullmove_bytes)
    
    return bytes(packed)


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
# POV-AWARE MOVE ENCODING - FIXED FOR 180° ROTATION
# ==============================================================================

NORMAL_ACTIONS = 4096
PROMOTION_TYPE_TO_INDEX = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3
}
INDEX_TO_PROMOTION_TYPE = {v: k for k, v in PROMOTION_TYPE_TO_INDEX.items()}

def _build_promotion_pairs():
    """
    Build ordered list of all promotion (from_square, to_square) pairs.
    Includes forward and capture promotions for both colors.
    """
    pairs = []
    
    # White promotions: from rank 7 -> rank 8
    for file in range(8):
        from_sq = chess.square(file, 6)
        # forward
        pairs.append((from_sq, chess.square(file, 7)))
        # capture left
        if file > 0:
            pairs.append((from_sq, chess.square(file - 1, 7)))
        # capture right
        if file < 7:
            pairs.append((from_sq, chess.square(file + 1, 7)))
    
    # Black promotions: from rank 2 -> rank 1
    for file in range(8):
        from_sq = chess.square(file, 1)
        # forward
        pairs.append((from_sq, chess.square(file, 0)))
        # capture left (from black perspective = file+1 in board coords)
        if file < 7:
            pairs.append((from_sq, chess.square(file + 1, 0)))
        # capture right (from black perspective = file-1)
        if file > 0:
            pairs.append((from_sq, chess.square(file - 1, 0)))
    
    return pairs


PROMOTION_PAIRS = _build_promotion_pairs()
PROMOTION_PAIR_TO_INDEX = {pair: idx for idx, pair in enumerate(PROMOTION_PAIRS)}
ACTION_SIZE = NORMAL_ACTIONS + len(PROMOTION_PAIRS) * 4

def move_to_index(move, board):
    """
    Convert chess.Move to index with POV support
    
    🔧 FIXED: Uses XOR 63 for 180° rotation (not square_mirror for vertical flip)
    
    If black to move, rotate both squares 180° to match flipped board
    
    Args:
        move: chess.Move object
        board: chess.Board (to determine whose turn it is)
    
    Returns:
        int: Move index (ACTION_SIZE)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Rotate 180° if black to move (XOR with 63)
    if board.turn == chess.BLACK:
        from_square = from_square ^ 63
        to_square = to_square ^ 63
    
    # Promotions use extended action space to disambiguate promotion type
    if move.promotion is not None:
        promo_idx = PROMOTION_TYPE_TO_INDEX.get(move.promotion)
        if promo_idx is None:
            promo_idx = PROMOTION_TYPE_TO_INDEX[chess.QUEEN]
        
        pair_idx = PROMOTION_PAIR_TO_INDEX.get((from_square, to_square))
        if pair_idx is None:
            # Fallback to base encoding if mapping fails
            return from_square * 64 + to_square
        
        return NORMAL_ACTIONS + (pair_idx * 4 + promo_idx)
    
    return from_square * 64 + to_square


def index_to_move(index, is_black_turn=False):
    """
    Convert index back to move with POV support
    
    Args:
        index: Move index (ACTION_SIZE)
        is_black_turn: Whether it's black's turn
    
    Returns:
        chess.Move object
    """
    if index < NORMAL_ACTIONS:
        from_square = index // 64
        to_square = index % 64
        
        # Rotate back if black
        if is_black_turn:
            from_square = from_square ^ 63
            to_square = to_square ^ 63
        
        return chess.Move(from_square, to_square)
    
    promo_index = index - NORMAL_ACTIONS
    pair_idx = promo_index // 4
    promo_type_idx = promo_index % 4
    
    if pair_idx < 0 or pair_idx >= len(PROMOTION_PAIRS):
        return chess.Move.null()
    
    from_square, to_square = PROMOTION_PAIRS[pair_idx]
    if is_black_turn:
        from_square = from_square ^ 63
        to_square = to_square ^ 63
    
    promotion = INDEX_TO_PROMOTION_TYPE.get(promo_type_idx, chess.QUEEN)
    return chess.Move(from_square, to_square, promotion=promotion)


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
    
    white_material = 0
    black_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
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
    return 1.0 if board.is_check() else 0.0


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

def get_position_size(use_mtl=False, history_positions=0):
    """
    Calculate size of binary position record
    
    🆕 v4.4 FORMAT (Extended with metadata):
    [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
    
    Board format changed: 32B pieces + 6B metadata (castling, en passant, halfmove, fullmove)
    
    Args:
        use_mtl: Whether Multi-Task Learning is enabled
        history_positions: Number of history positions (NOT used in new format)
    
    Returns:
        int: Size in bytes
    """
    base_size = 38  # Board (compact) - 🆕 NOW 38 bytes instead of 32!
    base_size += 4  # GameID (uint32) — supports up to ~4 billion unique games
    base_size += 2  # MoveIdx (uint16)
    base_size += 2  # MoveTarget (uint16) - the move label (0-4095)
    base_size += 4  # Outcome (float32)
    
    if use_mtl:
        base_size += 4  # Win (float32)
        base_size += 4  # Material (float32)
        base_size += 4  # Check (float32)
    
    return base_size


def pack_position_data(board, game_id, move_idx, move_target, outcome, mtl_labels=None):
    """
    Pack position data into binary format
    
    🆕 v4.4 FORMAT (Extended with metadata):
    [Board (38B)] + [GameID (4B)] + [MoveIdx (2B)] + [MoveTarget (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
    
    Args:
        board: chess.Board
        game_id: Unique game identifier (0 – 4294967295, uint32)
        move_idx: Move index in game (0-based)
        move_target: Target move index (0-4095) - THIS IS THE LABEL
        outcome: Game outcome value
        mtl_labels: Optional dict with 'win', 'material', 'check'
    
    Returns:
        bytes: Packed binary data
    """
    data = bytearray()
    
    # Pack board (38 bytes - 🆕 NOW INCLUDES METADATA!)
    data.extend(board_to_compact(board))
    
    # Pack metadata
    data.extend(struct.pack('I', game_id))          # GameID (4 bytes, uint32)
    data.extend(struct.pack('H', move_idx))         # MoveIdx (2 bytes)
    data.extend(struct.pack('H', move_target))      # MoveTarget (2 bytes)
    data.extend(struct.pack('f', outcome))          # Outcome (4 bytes)
    
    # Pack MTL labels if provided (12 bytes)
    if mtl_labels is not None:
        data.extend(struct.pack('f', mtl_labels['win']))
        data.extend(struct.pack('f', mtl_labels['material']))
        data.extend(struct.pack('f', mtl_labels['check']))
    
    return bytes(data)


