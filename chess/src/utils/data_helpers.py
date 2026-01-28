"""
Data processing helper functions for chess AI
Extracted from data.py for better code organization
"""

import chess
import numpy as np
import struct


def board_to_tensor(board, history_boards=None, history_positions=0):
    """
    Convert chess.Board to tensor representation with optional history
    
    Args:
        board: Current chess.Board
        history_boards: List of previous chess.Board objects (oldest to newest)
        history_positions: Number of history positions to include
    
    Returns: 
        - Without history: (12, 8, 8) tensor
        - With history: (12 * (1 + history_positions), 8, 8) tensor
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            piece_idx = piece_to_idx[piece.piece_type]
            if piece.color == chess.WHITE:
                channel = piece_idx
            else:
                channel = piece_idx + 6
            tensor[channel, row, col] = 1.0
    
    # If no history needed, return current board only
    if history_positions == 0 or history_boards is None:
        return tensor
    
    # Build history tensors
    history_tensors = []
    
    # Get last N boards from history
    if history_boards:
        history_list = history_boards[-history_positions:]
    else:
        history_list = []
    
    # Pad with empty boards if not enough history
    while len(history_list) < history_positions:
        empty_board = np.zeros((12, 8, 8), dtype=np.float32)
        history_tensors.append(empty_board)
    
    # Add actual history boards (convert each to tensor)
    for hist_board in history_list:
        hist_tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = hist_board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_idx = piece_to_idx[piece.piece_type]
                if piece.color == chess.WHITE:
                    channel = piece_idx
                else:
                    channel = piece_idx + 6
                hist_tensor[channel, row, col] = 1.0
        
        history_tensors.append(hist_tensor)
    
    # Stack: [oldest_history, ..., newest_history, current]
    all_tensors = history_tensors + [tensor]
    stacked = np.concatenate(all_tensors, axis=0)
    
    return stacked


def board_to_compact(board):
    """
    Convert board to ultra-compact binary representation
    Each position: 64 squares × 4 bits = 32 bytes
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


def compact_to_tensor(compact_board):
    """Convert compact representation back to tensor"""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    for i in range(32):
        byte = compact_board[i]
        code1 = (byte >> 4) & 0x0F
        code2 = byte & 0x0F
        
        square1 = i * 2
        square2 = i * 2 + 1
        
        for square, code in [(square1, code1), (square2, code2)]:
            if code == 0:
                continue
            
            row = square // 8
            col = square % 8
            
            if 1 <= code <= 6:
                channel = code - 1
            else:
                channel = code - 1
            
            tensor[channel, row, col] = 1.0
    
    return tensor


def move_to_index(move):
    """Convert chess.Move to index (0-4095)"""
    return move.from_square * 64 + move.to_square


def index_to_move(index):
    """Convert index back to move"""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)


# ==============================================================================
# MULTI-TASK LEARNING HELPER FUNCTIONS
# ==============================================================================

def compute_material_balance(board):
    """
    Compute material balance for current player
    
    Returns:
        float: Material advantage in pawns (-10 to +10)
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