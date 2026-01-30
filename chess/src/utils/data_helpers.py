"""
Data processing helper functions for chess AI
🆕 v4.2: POV (Point of View) + Dynamic Sliding Window
- 🎯 POV: All boards from perspective of current player
- 🔄 Sliding Window: Dynamic history assembly using mmap
- 🎮 GameID tracking: Track games for history reconstruction
"""

import chess
import numpy as np
import struct


# ==============================================================================
# POV (POINT OF VIEW) BOARD REPRESENTATION
# ==============================================================================

def board_to_tensor(board, flip_perspective=None):
    """
    Convert chess.Board to tensor representation with POV (Point of View)
    
    POV System:
    - Channels 0-5: Current player's pieces (White if white to move, Black if black to move)
    - Channels 6-11: Opponent's pieces
    - Board orientation: Always from current player's perspective
    
    Args:
        board: Current chess.Board
        flip_perspective: Override automatic flip (for history boards)
                         If None, auto-detect from board.turn
                         If True, flip (for black's perspective)
                         If False, don't flip (for white's perspective)
    
    Returns: 
        (12, 8, 8) tensor from current player's perspective
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Determine if we need to flip
    if flip_perspective is None:
        should_flip = (board.turn == chess.BLACK)
    else:
        should_flip = flip_perspective
    
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
    
    return tensor


def board_to_compact(board):
    """
    Convert board to ultra-compact binary representation
    Each position: 64 squares × 4 bits = 32 bytes
    
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
    
    # Pack pairs of codes into bytes
    packed = bytearray(32)
    for i in range(0, 64, 2):
        packed[i // 2] = (codes[i] << 4) | codes[i + 1]
    
    return bytes(packed)


def compact_to_tensor(compact_board, flip_perspective=False):
    """
    Convert compact representation back to tensor with POV support
    
    Args:
        compact_board: 32-byte compact representation
        flip_perspective: If True, flip board for black's perspective
    
    Returns:
        (12, 8, 8) tensor
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
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
    
    return tensor


# ==============================================================================
# POV-AWARE MOVE ENCODING
# ==============================================================================

def move_to_index(move, board):
    """
    Convert chess.Move to index (0-4095) with POV support
    
    If black to move, flip the move coordinates to match flipped board
    
    Args:
        move: chess.Move object
        board: chess.Board (to determine whose turn it is)
    
    Returns:
        int: Move index (0-4095)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Flip move if black to move
    if board.turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    
    return from_square * 64 + to_square


def index_to_move(index, is_black_turn=False):
    """
    Convert index back to move with POV support
    
    Args:
        index: Move index (0-4095)
        is_black_turn: If True, unflip the move
    
    Returns:
        chess.Move
    """
    from_square = index // 64
    to_square = index % 64
    
    # Unflip if black's turn
    if is_black_turn:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    
    return chess.Move(from_square, to_square)


# ==============================================================================
# DYNAMIC SLIDING WINDOW HELPERS
# ==============================================================================

def should_include_position(move_idx, stride):
    """
    Determine if position should be included based on sliding window stride
    
    Args:
        move_idx: Move index in game (0-based)
        stride: Sliding window stride (1 = all positions, 2 = every other, etc.)
    
    Returns:
        bool: True if position should be included
    """
    return (move_idx % stride) == 0


def extract_game_id(data_bytes, position_size):
    """
    Extract game_id from binary position data
    
    Binary format:
    [Board (32B)] + [GameID (2B, 'H')] + [MoveIdx (2B, 'H')] + [Outcome (4B, 'f')] + [MTL...]
    
    Args:
        data_bytes: Raw bytes from binary file
        position_size: Total size of position record
    
    Returns:
        tuple: (game_id, move_idx)
    """
    game_id = struct.unpack('H', data_bytes[32:34])[0]
    move_idx = struct.unpack('H', data_bytes[34:36])[0]
    return game_id, move_idx


def get_turn_from_move_idx(move_idx):
    """
    Determine whose turn it is based on move index
    
    Args:
        move_idx: Move index in game (0-based)
    
    Returns:
        chess.Color: WHITE if even move, BLACK if odd move
    """
    return chess.WHITE if (move_idx % 2) == 0 else chess.BLACK


# ==============================================================================
# MULTI-TASK LEARNING HELPER FUNCTIONS
# ==============================================================================

def compute_material_balance(board):
    """
    Compute material balance for current player
    
    Returns:
        float: Material advantage in pawns (-1.0 to +1.0)
               Positive = current player ahead
               Negative = current player behind
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King doesn't count for material
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
    
    # Return from perspective of current player
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


# ==============================================================================
# BINARY FORMAT HELPERS
# ==============================================================================

def get_position_size(use_mtl=False, history_positions=0):
    """
    Calculate size of binary position record
    
    NEW FORMAT (without embedded history):
    [Board (32B)] + [GameID (2B)] + [MoveIdx (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
    
    Args:
        use_mtl: Whether Multi-Task Learning is enabled
        history_positions: Number of history positions (NOT used in new format)
    
    Returns:
        int: Size in bytes
    """
    base_size = 32  # Board (compact)
    base_size += 2  # GameID (uint16)
    base_size += 2  # MoveIdx (uint16)
    base_size += 4  # Outcome (float32)
    
    if use_mtl:
        base_size += 4  # Win (float32)
        base_size += 4  # Material (float32)
        base_size += 4  # Check (float32)
    
    return base_size


def pack_position_data(board, game_id, move_idx, outcome, mtl_labels=None):
    """
    Pack position data into binary format
    
    NEW FORMAT:
    [Board (32B)] + [GameID (2B)] + [MoveIdx (2B)] + [Outcome (4B)] + [MTL (12B if enabled)]
    
    Args:
        board: chess.Board
        game_id: Unique game identifier (0-65535)
        move_idx: Move index in game (0-based)
        outcome: Game outcome value
        mtl_labels: Optional dict with 'win', 'material', 'check'
    
    Returns:
        bytes: Packed binary data
    """
    data = bytearray()
    
    # Pack board (32 bytes)
    data.extend(board_to_compact(board))
    
    # Pack metadata (8 bytes)
    data.extend(struct.pack('H', game_id))      # GameID (2 bytes)
    data.extend(struct.pack('H', move_idx))     # MoveIdx (2 bytes)
    data.extend(struct.pack('f', outcome))      # Outcome (4 bytes)
    
    # Pack MTL labels if provided (12 bytes)
    if mtl_labels is not None:
        data.extend(struct.pack('f', mtl_labels['win']))
        data.extend(struct.pack('f', mtl_labels['material']))
        data.extend(struct.pack('f', mtl_labels['check']))
    
    return bytes(data)


def unpack_position_data(data_bytes, use_mtl=False):
    """
    Unpack position data from binary format
    
    Args:
        data_bytes: Raw bytes from file
        use_mtl: Whether MTL labels are included
    
    Returns:
        dict: {
            'board_compact': bytes (32),
            'game_id': int,
            'move_idx': int,
            'outcome': float,
            'win': float (if MTL),
            'material': float (if MTL),
            'check': float (if MTL)
        }
    """
    result = {}
    
    # Unpack board (32 bytes)
    result['board_compact'] = data_bytes[:32]
    
    # Unpack metadata
    result['game_id'] = struct.unpack('H', data_bytes[32:34])[0]
    result['move_idx'] = struct.unpack('H', data_bytes[34:36])[0]
    result['outcome'] = struct.unpack('f', data_bytes[36:40])[0]
    
    # Unpack MTL labels if present
    if use_mtl:
        result['win'] = struct.unpack('f', data_bytes[40:44])[0]
        result['material'] = struct.unpack('f', data_bytes[44:48])[0]
        result['check'] = struct.unpack('f', data_bytes[48:52])[0]
    
    return result


# ==============================================================================
# DATASET STATISTICS HELPERS
# ==============================================================================

def analyze_dataset_distribution(binary_file, position_size, sample_size=10000):
    """
    Analyze dataset distribution (game lengths, move counts, etc.)
    
    Args:
        binary_file: Path to binary dataset
        position_size: Size of each position record
        sample_size: Number of positions to sample
    
    Returns:
        dict: Statistics about dataset
    """
    import mmap
    import random
    
    stats = {
        'total_positions': 0,
        'unique_games': set(),
        'max_move_idx': 0,
        'move_distribution': {},
        'outcome_distribution': {'win': 0, 'loss': 0, 'draw': 0}
    }
    
    with open(binary_file, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        total_positions = len(mm) // position_size
        stats['total_positions'] = total_positions
        
        # Sample random positions
        sample_indices = random.sample(range(total_positions), min(sample_size, total_positions))
        
        for idx in sample_indices:
            offset = idx * position_size
            data = mm[offset:offset + position_size]
            
            game_id = struct.unpack('H', data[32:34])[0]
            move_idx = struct.unpack('H', data[34:36])[0]
            outcome = struct.unpack('f', data[36:40])[0]
            
            stats['unique_games'].add(game_id)
            stats['max_move_idx'] = max(stats['max_move_idx'], move_idx)
            
            # Track move distribution
            stats['move_distribution'][move_idx] = stats['move_distribution'].get(move_idx, 0) + 1
            
            # Track outcome distribution
            if outcome > 0.5:
                stats['outcome_distribution']['win'] += 1
            elif outcome < -0.5:
                stats['outcome_distribution']['loss'] += 1
            else:
                stats['outcome_distribution']['draw'] += 1
        
        mm.close()
    
    stats['unique_games'] = len(stats['unique_games'])
    
    return stats


def print_dataset_stats(stats):
    """Pretty print dataset statistics"""
    print(f"\n{'='*70}")
    print("📊 Dataset Statistics")
    print(f"{'='*70}")
    print(f"  Total positions: {stats['total_positions']:,}")
    print(f"  Unique games: {stats['unique_games']:,}")
    print(f"  Max move index: {stats['max_move_idx']}")
    print(f"\n  Outcome distribution:")
    print(f"    Wins:  {stats['outcome_distribution']['win']:,}")
    print(f"    Draws: {stats['outcome_distribution']['draw']:,}")
    print(f"    Losses: {stats['outcome_distribution']['loss']:,}")
    print(f"\n  Top 10 most common move indices:")
    sorted_moves = sorted(stats['move_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for move_idx, count in sorted_moves:
        print(f"    Move {move_idx}: {count:,} positions")
    print(f"{'='*70}\n")