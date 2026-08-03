"""Native bulletchess runtime contract used by search and self-play.

This module intentionally has no python-chess fallback.  Runtime boards and
moves stay as native bulletchess objects for their whole lifetime.  Tooling
which still needs python-chess (UCI/SVG/Syzygy) must convert explicitly at its
outer boundary, never inside an MCTS simulation.
"""

from __future__ import annotations

import bulletchess as _bc


BACKEND_NAME = "bulletchess"

Board = _bc.Board
Move = _bc.Move
Piece = _bc.Piece
Square = _bc.Square
Color = _bc.Color
PieceType = _bc.PieceType

WHITE = _bc.WHITE
BLACK = _bc.BLACK
PAWN = _bc.PAWN
KNIGHT = _bc.KNIGHT
BISHOP = _bc.BISHOP
ROOK = _bc.ROOK
QUEEN = _bc.QUEEN
KING = _bc.KING
PIECE_TYPES = _bc.PIECE_TYPES
SQUARES = _bc.SQUARES

CHECK = _bc.CHECK
CHECKMATE = _bc.CHECKMATE
STALEMATE = _bc.STALEMATE
INSUFFICIENT_MATERIAL = _bc.INSUFFICIENT_MATERIAL
FIFTY_MOVE_TIMEOUT = _bc.FIFTY_MOVE_TIMEOUT
THREEFOLD_REPETITION = _bc.THREEFOLD_REPETITION
DRAW = _bc.DRAW
FORCED_DRAW = _bc.FORCED_DRAW

WHITE_KINGSIDE = _bc.WHITE_KINGSIDE
WHITE_QUEENSIDE = _bc.WHITE_QUEENSIDE
BLACK_KINGSIDE = _bc.BLACK_KINGSIDE
BLACK_QUEENSIDE = _bc.BLACK_QUEENSIDE

_CASTLING_BY_COLOR = {
    WHITE: (WHITE_KINGSIDE, WHITE_QUEENSIDE),
    BLACK: (BLACK_KINGSIDE, BLACK_QUEENSIDE),
}


def new_board(fen: str | None = None) -> Board:
    return Board() if fen is None else Board.from_fen(str(fen))


def empty_board() -> Board:
    return Board.empty()


def board_from_fen(fen: str) -> Board:
    return Board.from_fen(str(fen))


def board_fen(board: Board) -> str:
    return board.fen()


def copy_board(board: Board) -> Board:
    return board.copy()


def root_board(board: Board) -> Board:
    root = board.copy()
    for _ in range(len(root.history)):
        root.undo()
    return root


def legal_moves(board: Board) -> list[Move]:
    return board.legal_moves()


def apply_move(board: Board, move: Move | None) -> None:
    board.apply(move)


def undo_move(board: Board) -> Move | None:
    return board.undo()


def move_history(board: Board) -> list[Move]:
    return board.history


def last_move(board: Board) -> Move | None:
    history = board.history
    return history[-1] if history else None


def move_from_uci(uci: str) -> Move | None:
    return Move.from_uci(str(uci))


def move_uci(move: Move | None) -> str:
    return "0000" if move is None else move.uci()


def square_index(square: Square | int) -> int:
    return int(square) if isinstance(square, int) else int(square.index())


def square_from_index(index: int) -> Square:
    return SQUARES[int(index)]


def square_rank(square: Square | int) -> int:
    return square_index(square) >> 3


def square_file(square: Square | int) -> int:
    return square_index(square) & 7


def move_origin_index(move: Move) -> int:
    return int(move.origin.index())


def move_destination_index(move: Move) -> int:
    return int(move.destination.index())


def new_move(from_square: int, to_square: int, promotion: PieceType | None = None) -> Move:
    return Move(
        square_from_index(from_square),
        square_from_index(to_square),
        promote_to=promotion,
    )


def piece_at(board: Board, square: Square | int):
    return board[square_from_index(square) if isinstance(square, int) else square]


def piece_mask(board: Board, piece_type: PieceType, color: Color) -> int:
    return int(board[color, piece_type])


def occupied_mask(board: Board) -> int:
    return int(board[WHITE]) | int(board[BLACK])


def piece_count(board: Board) -> int:
    return len(board[WHITE]) + len(board[BLACK])


def king_square(board: Board, color: Color) -> Square | None:
    kings = board[color, KING]
    return next(iter(kings), None)


def has_kingside_castling_rights(board: Board, color: Color) -> bool:
    return _CASTLING_BY_COLOR[color][0] in board.castling_rights


def has_queenside_castling_rights(board: Board, color: Color) -> bool:
    return _CASTLING_BY_COLOR[color][1] in board.castling_rights


def is_capture(board: Board, move: Move) -> bool:
    return bool(move.is_capture(board))


def is_en_passant(board: Board, move: Move) -> bool:
    ep_square = board.en_passant_square
    if ep_square is None or move.destination != ep_square:
        return False
    piece = board[move.origin]
    return piece is not None and piece.piece_type == PAWN and board[move.destination] is None


def is_castling(board: Board, move: Move) -> bool:
    piece = board[move.origin]
    return bool(
        piece is not None
        and piece.piece_type == KING
        and abs(move_destination_index(move) - move_origin_index(move)) == 2
    )


def is_check(board: Board) -> bool:
    return board in CHECK


def gives_check(board: Board, move: Move) -> bool:
    """Check a legal move without allocating/copying another board."""
    board.apply(move)
    try:
        return board in CHECK
    finally:
        board.undo()


def is_insufficient_material(board: Board) -> bool:
    return board in INSUFFICIENT_MATERIAL


def is_game_over(board: Board, *, claim_draw: bool = False) -> bool:
    return board in CHECKMATE or board in (DRAW if claim_draw else FORCED_DRAW)


def can_claim_draw(board: Board) -> bool:
    return board in DRAW


def can_claim_threefold_repetition(board: Board) -> bool:
    return board in THREEFOLD_REPETITION


def result(board: Board, *, claim_draw: bool = False) -> str:
    if board in CHECKMATE:
        return "0-1" if board.turn == WHITE else "1-0"
    if board in (DRAW if claim_draw else FORCED_DRAW):
        return "1/2-1/2"
    return "*"


def position_key(board: Board) -> int:
    return hash(board)


def castling_rights_from_flags(flags: int):
    rights = []
    if flags & 0b1000:
        rights.append(WHITE_KINGSIDE)
    if flags & 0b0100:
        rights.append(WHITE_QUEENSIDE)
    if flags & 0b0010:
        rights.append(BLACK_KINGSIDE)
    if flags & 0b0001:
        rights.append(BLACK_QUEENSIDE)
    return _bc.CastlingRights(rights)
