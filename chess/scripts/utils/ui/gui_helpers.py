"""UI and PGN helper functions for local chess play."""

from datetime import datetime
import io
from pathlib import Path

import chess
import chess.pgn
import chess.svg
import pygame


def create_piece_surfaces(square_size=80):
    """Create piece sprites using python-chess SVG rendering.

    Falls back to text-based symbols if SVG loading fails on the host.
    """
    pieces = {}
    for symbol in _iter_piece_symbols():
        piece = chess.Piece.from_symbol(symbol)
        try:
            svg_markup = chess.svg.piece(piece, size=square_size)
            svg_bytes = io.BytesIO(svg_markup.encode("utf-8"))
            surface = pygame.image.load(svg_bytes).convert_alpha()
            if surface.get_size() != (square_size, square_size):
                surface = pygame.transform.smoothscale(surface, (square_size, square_size))
            pieces[symbol] = _fit_piece_surface(surface, square_size, fill_ratio=0.60)
        except pygame.error:
            fallback = _create_fallback_piece_surface(symbol, square_size)
            pieces[symbol] = _fit_piece_surface(fallback, square_size, fill_ratio=0.60)

    return pieces


def _iter_piece_symbols():
    """Yield all piece symbols without hardcoded symbol arrays."""
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            yield chess.Piece(piece_type, color).symbol()


def _fit_piece_surface(surface, square_size, fill_ratio=0.60):
    """Crop transparent margins and scale piece to better fill a square."""
    bbox = surface.get_bounding_rect(min_alpha=1)
    if bbox.width <= 0 or bbox.height <= 0:
        return surface

    cropped = surface.subsurface(bbox).copy()
    target = max(1, int(square_size * fill_ratio))
    scale = min(target / cropped.get_width(), target / cropped.get_height())
    new_w = max(1, int(cropped.get_width() * scale))
    new_h = max(1, int(cropped.get_height() * scale))
    scaled = pygame.transform.smoothscale(cropped, (new_w, new_h))

    canvas = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
    x = (square_size - new_w) // 2
    y = (square_size - new_h) // 2
    canvas.blit(scaled, (x, y))
    return canvas


def _create_fallback_piece_surface(symbol, square_size):
    """Fallback piece style if SVG loading is unavailable."""
    surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
    unicode_piece = chess.UNICODE_PIECE_SYMBOLS.get(symbol, symbol)
    font = pygame.font.SysFont("Segoe UI Symbol", int(square_size * 0.72), bold=False)
    text_color = (248, 248, 248) if symbol.isupper() else (28, 28, 28)
    outline_color = (24, 24, 24) if symbol.isupper() else (235, 235, 235)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            outline = font.render(unicode_piece, True, outline_color)
            outline_rect = outline.get_rect(
                center=(square_size // 2 + dx, square_size // 2 + dy)
            )
            surface.blit(outline, outline_rect)

    text = font.render(unicode_piece, True, text_color)
    text_rect = text.get_rect(center=(square_size // 2, square_size // 2))
    surface.blit(text, text_rect)
    return surface


def resolve_games_dir(config, base_dir):
    """Resolve and create the directory used for PGN autosaves."""
    logs_dir = base_dir / config.get("paths", {}).get("logs_dir", "logs")
    games_dir_cfg = config.get("paths", {}).get("games_dir")
    if games_dir_cfg:
        games_dir = Path(games_dir_cfg)
        if not games_dir.is_absolute():
            games_dir = base_dir / games_dir
    else:
        games_dir = logs_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    return games_dir


def get_game_mode_labels(game_mode, human_color):
    """Get sidebar labels for current game mode."""
    if game_mode == "human_vs_ai":
        mode_text = "Human vs AI"
        you_text = f"You: {'White' if human_color == chess.WHITE else 'Black'}"
    elif game_mode == "ai_vs_ai":
        mode_text = "AI vs AI"
        you_text = "Spectator Mode"
    else:
        mode_text = "Human vs Human"
        you_text = "2 Player Mode"
    return mode_text, you_text


def get_turn_color(game_mode, board_turn, human_color):
    """Get color used for current turn text in sidebar."""
    if game_mode == "human_vs_ai":
        if board_turn == human_color:
            return (255, 255, 200)
        return (200, 200, 255)
    if board_turn == chess.WHITE:
        return (255, 255, 200)
    return (200, 200, 255)


def format_recent_moves(move_san_history, limit=12):
    """Format recent SAN moves for sidebar display."""
    recent = move_san_history[-limit:]
    start_idx = max(0, len(move_san_history) - limit)
    return [f"{start_idx + i + 1}. {san}" for i, san in enumerate(recent)]


def get_result_message(result):
    """Map PGN result code to UI text and color."""
    if result == "1-0":
        return "White Wins!", (100, 255, 100)
    if result == "0-1":
        return "Black Wins!", (100, 255, 100)
    return "Draw!", (200, 200, 200)


def _resolve_player_names(game_mode, human_color, model1_name, model2_name):
    if game_mode == "human_vs_ai":
        if human_color == chess.WHITE:
            return "Human", model1_name
        return model1_name, "Human"
    if game_mode == "ai_vs_ai":
        return model1_name, model2_name
    return "Human", "Human"


def _resolve_termination(board, termination):
    if termination:
        return termination
    if not board.is_game_over():
        return "unterminated"
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.can_claim_threefold_repetition():
        return "threefold repetition"
    if board.can_claim_fifty_moves():
        return "fifty-move rule"
    return "game over"


def build_pgn_game(
    board,
    move_history,
    initial_fen,
    version,
    game_mode,
    mcts_enabled,
    game_index,
    human_color,
    model1_name,
    model2_name,
    result_override=None,
    termination=None,
):
    """Build PGN object for current game state."""
    game = chess.pgn.Game()
    now = datetime.now()

    white_name, black_name = _resolve_player_names(
        game_mode=game_mode,
        human_color=human_color,
        model1_name=model1_name,
        model2_name=model2_name,
    )

    if result_override is not None:
        result = result_override
    elif board.is_game_over():
        result = board.result()
    else:
        result = "*"

    game.headers["Event"] = "Chess AI GUI"
    game.headers["Site"] = "Local"
    game.headers["Date"] = now.strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_index)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = _resolve_termination(board, termination)
    game.headers["ModelVersion"] = version
    game.headers["GameMode"] = game_mode
    game.headers["MCTS"] = "enabled" if mcts_enabled else "disabled"

    if initial_fen != chess.STARTING_FEN:
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = initial_fen

    node = game
    replay_board = chess.Board(initial_fen)
    for move in move_history:
        if move not in replay_board.legal_moves:
            break
        node = node.add_variation(move)
        replay_board.push(move)

    return game


def save_game_to_pgn(games_dir, version, game_index, game):
    """Persist a PGN game object and return output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{version}_{timestamp}_{game_index:03d}.pgn"
    output_path = games_dir / filename
    with open(output_path, "w", encoding="utf-8", newline="\n") as file_obj:
        print(game, file=file_obj, end="\n\n")
    return output_path
