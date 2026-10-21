"""Pygame chess interface for human play and accelerated model matches."""

import torch
import chess
import sys
import io
import math
import ctypes
import multiprocessing
import os
import re
import time
import wave
import threading
import numpy as np
from collections import OrderedDict
from array import array
from pathlib import Path

# Windows ``spawn`` imports this module once in every match worker. Hide the
# interactive greeting before any worker imports Pygame.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_IS_MULTIPROCESSING_CHILD = (
    __name__ == "__mp_main__"
    or multiprocessing.parent_process() is not None
)
if not _IS_MULTIPROCESSING_CHILD:
    import pygame
else:
    pygame = None
import argparse

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.mcts.single_game import SingleGameMCTS
from src.game import backend as native_chess
from src.config import default_config_path, load_project_config
from src.models.registry import MODEL_REGISTRY
from src.models.data.registry import get_input_encoder
from src.mcts.search import _board_position_key
from src.ui.play_match_session import PlayMatchSession
from src.ui.play_ai_turn import PlayAITurn
from src.ui.play_visuals import centered_rect, fit_middle_text, fit_text as fit_ui_text
if not _IS_MULTIPROCESSING_CHILD:
    from src.ui.play_workspace_layout import (
        compute_workspace_layout,
        draw_workspace,
        workspace_scale,
    )

#  v4.2: Import board_to_tensor from data_helpers
#  v4.4: Added move_to_index for POV-aware move encoding
from src.models.data.se_cnn_v9.helpers import board_to_tensor, move_to_index


def _native_board(board):
    """One explicit UI -> search conversion; never used inside MCTS."""
    return native_chess.board_from_fen(board.fen())


def _position_counts_for_history(history_boards, current_board):
    """Return exact occurrence counts for this GUI game's known line."""
    counts = {}
    for board in history_boards:
        key = _board_position_key(_native_board(board))
        counts[key] = int(counts.get(key, 0)) + 1
    key = _board_position_key(_native_board(current_board))
    counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _python_move(move):
    """Convert a native search result back to the python-chess UI board."""
    return None if move is None else chess.Move.from_uci(native_chess.move_uci(move))


def _finite_json_number(value):
    """Return a JSON-safe finite number; search can use +/-inf as sentinels."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value):
    """Convert small model/config metadata structures to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_json_number(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return str(value)


def _match_wdl_summary(snapshot):
    """Derive model-identity W/D/L rates from resolved games only."""
    def count(key):
        try:
            return max(0, int(snapshot.get(key, 0) or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    completed = count("completed")
    model1_wins = count("model1_wins")
    model2_wins = count("model2_wins")
    draws = count("draws")
    # A resolved game always has exactly one W/D/L result. Trust the result
    # counters for the bar shares, and completed for displayed rates; if a
    # malformed/partial snapshot disagrees, the UI still cannot divide by 0.
    outcome_count = model1_wins + draws + model2_wins
    resolved = completed
    score1_x2 = count("model1_score_x2")
    score2_x2 = count("model2_score_x2")
    if resolved and score1_x2 + score2_x2 == 0:
        score1_x2 = 2 * model1_wins + draws
        score2_x2 = 2 * model2_wins + draws
    return {
        "resolved": resolved,
        "model1_wins": model1_wins,
        "draws": draws,
        "model2_wins": model2_wins,
        "model1_win_pct": 100.0 * model1_wins / resolved if resolved else None,
        "draw_pct": 100.0 * draws / resolved if resolved else None,
        "model2_win_pct": 100.0 * model2_wins / resolved if resolved else None,
        "model1_points_pct": 100.0 * score1_x2 / (2 * resolved) if resolved else None,
        "model2_points_pct": 100.0 * score2_x2 / (2 * resolved) if resolved else None,
        "outcome_count": outcome_count,
    }


def _match_wdl_segment_widths(summary, width):
    """Allocate exact adjacent pixel widths for model1/draw/model2 results."""
    width = max(0, int(width))
    total = int(summary.get("outcome_count", 0) or 0)
    if width == 0 or total <= 0:
        return (0, 0, 0)
    model1_wins = int(summary["model1_wins"])
    draws = int(summary["draws"])
    model1_width = (width * model1_wins + total // 2) // total
    through_draws = (width * (model1_wins + draws) + total // 2) // total
    draw_width = through_draws - model1_width
    model2_width = width - through_draws
    return (model1_width, draw_width, model2_width)


def _install_shared_model_forward_lock(model):
    """Serialize only model forward calls shared by visible and batch games."""
    if model is None:
        return None
    existing_lock = getattr(model, "_play_forward_lock", None)
    if existing_lock is not None:
        return existing_lock
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return None

    lock = threading.RLock()

    def _locked_forward(*args, **kwargs):
        with lock:
            return forward(*args, **kwargs)

    try:
        model.forward = _locked_forward
        model._play_forward_lock = lock
    except (AttributeError, TypeError):
        return None
    return lock

# Import application modules
from src.evaluation.model_match import (
    prepare_models_for_play_search,
    run_model_match,
)
from src.models.catalog import format_elo_summary, load_checkpoint_metadata
if not _IS_MULTIPROCESSING_CHILD:
    from src.ui.game_setup import (
        ModelCompatibilityError,
        load_model_from_checkpoint,
        select_models,
        show_model_load_error,
        write_setup_log,
    )
    from src.ui.gui_helpers import (
        build_pgn_game,
        create_piece_surfaces,
        get_game_mode_labels,
        get_result_message,
        get_turn_color,
        resolve_games_dir,
        save_game_to_pgn,
        start_piece_asset_prefetch,
    )
    from src.ui.theme import (
        ACCENT as UI_ACCENT,
        BORDER as UI_BORDER,
        BORDER_STRONG as UI_BORDER_STRONG,
        MUTED as UI_MUTED,
        SURFACE as UI_SURFACE,
        SURFACE_RAISED as UI_SURFACE_RAISED,
        TEXT as UI_TEXT,
        build_background as build_ui_background,
        draw_card as draw_ui_card,
    )

def _enable_windows_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# Constants
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
BOARD_MIN_SIZE = 416
LEFT_PANEL_WIDTH = 320
RIGHT_PANEL_WIDTH = 360
PANEL_GAP = 22
OUTER_MARGIN_X = 24
OUTER_MARGIN_Y = 24
WINDOW_WIDTH = (
    OUTER_MARGIN_X * 2
    + LEFT_PANEL_WIDTH
    + PANEL_GAP
    + BOARD_SIZE
    + PANEL_GAP
    + RIGHT_PANEL_WIDTH
)
WINDOW_HEIGHT = OUTER_MARGIN_Y * 2 + BOARD_SIZE
FPS = 60
WINDOW_MIN_WIDTH = 720
WINDOW_MIN_HEIGHT = 540
CANVAS_MIN_WIDTH = WINDOW_MIN_WIDTH
CANVAS_MIN_HEIGHT = WINDOW_MIN_HEIGHT
PANEL_MIN_WIDTH = 120
DEFAULT_WINDOW_SCREEN_WIDTH_RATIO = 0.92
DEFAULT_WINDOW_SCREEN_HEIGHT_RATIO = 0.92
ANALYSIS_DISPLAY_ROWS = 10
MCTS_SEARCH_SNAPSHOT_LIMIT = 128

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
BOARD_LIGHT = (226, 222, 201)
BOARD_DARK = (104, 133, 115)
HIGHLIGHT = (186, 202, 68, 150)
SELECTED = (246, 246, 105, 150)
LEGAL_MOVE = (100, 100, 100, 120)
CAPTURE_MOVE = (200, 50, 50, 120)
TEXT_COLOR = (240, 245, 252)
APP_BG = (7, 10, 16)


def _get_default_window_size():
    """Resolve a safe default window size from the current display."""
    fallback_width = max(CANVAS_MIN_WIDTH, WINDOW_WIDTH)
    fallback_height = max(CANVAS_MIN_HEIGHT, WINDOW_HEIGHT)
    try:
        info = pygame.display.Info()
        screen_width = int(getattr(info, "current_w", 0) or 0)
        screen_height = int(getattr(info, "current_h", 0) or 0)
    except Exception:
        screen_width = 0
        screen_height = 0

    if screen_width <= 0 or screen_height <= 0:
        return fallback_width, fallback_height

    default_width = int(screen_width * DEFAULT_WINDOW_SCREEN_WIDTH_RATIO)
    default_height = int(screen_height * DEFAULT_WINDOW_SCREEN_HEIGHT_RATIO)
    default_width = max(WINDOW_MIN_WIDTH, min(screen_width, default_width))
    default_height = max(WINDOW_MIN_HEIGHT, min(screen_height, default_height))
    return default_width, default_height


def _normalize_window_size(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (
            max(WINDOW_MIN_WIDTH, int(value[0])),
            max(WINDOW_MIN_HEIGHT, int(value[1])),
        )
    except (TypeError, ValueError):
        return None


def _apply_setup_play_config(config, setup):
    """Carry GUI setup choices into the config used to create the play screen."""
    try:
        visible_move_seconds = float(
            setup.get(
                "min_visible_move_seconds",
                config.get("play", {}).get("min_visible_move_seconds", 1.0),
            )
        )
        if not math.isfinite(visible_move_seconds) or visible_move_seconds < 0:
            raise ValueError("min_visible_move_seconds must be finite and non-negative")
    except (AttributeError, TypeError, ValueError):
        visible_move_seconds = 1.0
    config.setdefault("play", {})["min_visible_move_seconds"] = visible_move_seconds
    return visible_move_seconds


def _position_native_window(width, height):
    """Move the current window to the top-left corner with an exact size."""
    window_handle = _get_native_window_handle()
    if not window_handle:
        return False

    try:
        user32 = ctypes.windll.user32
        swp_framechanged = 0x0020
        swp_showwindow = 0x0040
        user32.SetWindowPos(
            int(window_handle),
            0,
            0,
            0,
            int(width),
            int(height),
            swp_framechanged | swp_showwindow,
        )
        return True
    except Exception:
        return False


def _get_native_window_handle():
    try:
        wm_info = pygame.display.get_wm_info()
    except Exception:
        return None
    return wm_info.get("window")


def _maximize_native_window():
    """Maximize the current window using the native window handle."""
    window_handle = _get_native_window_handle()
    if not window_handle:
        return False

    try:
        ctypes.windll.user32.ShowWindow(int(window_handle), 3)
        return True
    except Exception:
        return False
class ChessGUI:
    """Chess game GUI with support for multiple game modes and MCTS toggle - v4.2"""

    def __init__(
        self,
        model1,
        model2,
        config,
        device,
        game_mode="human_vs_ai",
        enable_mcts=True,
        model1_name=None,
        model2_name=None,
        console_verbose=False,
        initial_window_size=None,
        initial_window_maximized=False,
        model1_meta=None,
        model2_meta=None,
        model1_checkpoint_path=None,
        model2_checkpoint_path=None,
        use_mcts_white=None,
        use_mcts_black=None,
        mcts_simulations_white=None,
        mcts_simulations_black=None,
        match_games=1,
        human_color=chess.WHITE,
    ):
        self.model1 = model1  # White AI or main AI
        self.model2 = model2  # Black AI (for AI vs AI mode)
        self.config = config
        self.version = config.get('model', {}).get('version', 'v?.?')
        self.device = device
        self.game_mode = game_mode  # "human_vs_ai", "ai_vs_ai", "human_vs_human"
        self.inspector_tab = "moves"
        self.analysis_display_rows = ANALYSIS_DISPLAY_ROWS
        self.model1_name = model1_name or f"ChessAI-{self.version}-A"
        self.model2_name = model2_name or f"ChessAI-{self.version}-B"
        self.model1_meta = dict(model1_meta or {})
        self.model2_meta = dict(model2_meta or {})
        self.model1_checkpoint_path = model1_checkpoint_path
        self.model2_checkpoint_path = model2_checkpoint_path
        self.console_verbose = bool(console_verbose)
        default_mcts_sims = int(config.get("reinforcement_learning", {}).get("mcts_simulations", 100))
        try:
            self.mcts_simulations_white = max(1, int(mcts_simulations_white if mcts_simulations_white is not None else default_mcts_sims))
        except (TypeError, ValueError):
            self.mcts_simulations_white = default_mcts_sims
        try:
            self.mcts_simulations_black = max(1, int(mcts_simulations_black if mcts_simulations_black is not None else default_mcts_sims))
        except (TypeError, ValueError):
            self.mcts_simulations_black = default_mcts_sims
        try:
            self.match_total_games = max(1, min(500, int(match_games)))
        except (TypeError, ValueError):
            self.match_total_games = 1
        
        #  MCTS toggle
        self.mcts_enabled = enable_mcts
        self.use_mcts_white = bool(enable_mcts if use_mcts_white is None else use_mcts_white)
        self.use_mcts_black = bool(enable_mcts if use_mcts_black is None else use_mcts_black)

        self.match_lock = threading.Lock()
        self.match_generation = 0
        self.match_session = None
        self.last_match_session = None
        self.draining_match_session = None
        self.match_detached = False
        self.manual_analysis_mode = False
        self.match_start_pending = False
        self.visible_match_start_pending = False
        self.visible_match_finished = False
        self.visible_match_result = None
        self.visible_match_winner = None
        self.visible_match_error = None
        self.visible_mcts_searches = []
        self.match_started_at = time.perf_counter()
        self.match_started_at_unix = time.time()
        self.match_result_dir = script_dir.parent / "logs" / "analysis" / "play_matches"
        self.inference_lock = threading.RLock()
        self.model_forward_locks = {
            id(model): _install_shared_model_forward_lock(model)
            for model in (model1, model2)
            if model is not None
        }
        self.ai_generation = 0
        self.ai_turn = None
        self.draining_ai_turn = None
        self.ai_turn_history_key = None
        self._pending_ai_compute_duration = None
        self.pending_mcts_reset = False
        self.ai_status_error = None
        self.preview_analysis_generation = 0
        self.preview_analysis_turn = None
        self.draining_preview_analysis_turn = None
        self.preview_analysis_pending = None
        self.preview_analysis_pending_key = None
        self.preview_analysis_cache = {}
        self.selected_analysis_candidate = None
        self.pending_manual_move = None
        self.match_stats = self._new_match_stats()
        
        # History depth can differ per loaded checkpoint architecture.
        default_history = int(config['model'].get('history_positions', 0))
        def _model_history_positions(model, fallback):
            model_spec = getattr(model, 'model_spec', {}) or {}
            input_spec = model_spec.get('input_spec', {}) or {}
            return int(
                input_spec.get(
                    'history_positions',
                    getattr(model, 'history_positions', fallback),
                )
                or 0
            )

        self.model1_history_positions = (
            _model_history_positions(model1, default_history)
            if model1
            else default_history
        )
        self.model2_history_positions = (
            _model_history_positions(model2, self.model1_history_positions)
            if model2
            else self.model1_history_positions
        )
        
        # Only create MCTS if enabled
        if model1 and self.mcts_enabled:
            self.mcts1 = SingleGameMCTS(
                model1,
                config,
                device,
                single_root_batch_cap=(config.get("play", {}) or {}).get(
                    "single_game_mcts_batch_cap", 0
                ),
            )
        else:
            self.mcts1 = None
            
        if model2 and self.mcts_enabled:
            self.mcts2 = SingleGameMCTS(
                model2,
                config,
                device,
                single_root_batch_cap=(config.get("play", {}) or {}).get(
                    "single_game_mcts_batch_cap", 0
                ),
            )
        else:
            self.mcts2 = None

        default_window_width, default_window_height = _get_default_window_size()
        self.base_width = max(WINDOW_MIN_WIDTH, default_window_width)
        self.base_height = max(WINDOW_MIN_HEIGHT, default_window_height)
        self.display_flags = pygame.RESIZABLE
        self.maximized = False
        self.ui_scale = 1.0
        start_width = default_window_width
        start_height = default_window_height
        if isinstance(initial_window_size, (list, tuple)) and len(initial_window_size) == 2:
            try:
                start_width = max(WINDOW_MIN_WIDTH, int(initial_window_size[0]))
                start_height = max(WINDOW_MIN_HEIGHT, int(initial_window_size[1]))
            except (TypeError, ValueError):
                start_width = default_window_width
                start_height = default_window_height
        self.restore_window_size = (start_width, start_height)
        existing_surface = pygame.display.get_surface()
        if existing_surface is not None and existing_surface.get_size() == self.restore_window_size:
            self.screen = existing_surface
        else:
            self.screen = pygame.display.set_mode(self.restore_window_size, self.display_flags)
        self.viewport_rect = pygame.Rect(0, 0, self.base_width, self.base_height)
        self.viewport_scale = 1.0
        self.canvas = pygame.Surface((self.base_width, self.base_height))
        self.background = pygame.Surface((self.base_width, self.base_height))
        self.left_panel_rect = pygame.Rect(0, 0, 0, 0)
        self.board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
        self.board_frame_rect = self.board_rect.copy()
        self.right_panel_rect = pygame.Rect(0, 0, 0, 0)
        self.header_rect = pygame.Rect(0, 0, 0, 0)
        self.top_player_rect = pygame.Rect(0, 0, 0, 0)
        self.bottom_player_rect = pygame.Rect(0, 0, 0, 0)
        self.inspector_tab_rects = {}
        self.match_summary_rect = None
        self.inspector_body_rect = pygame.Rect(0, 0, 0, 0)
        self.square_size = SQUARE_SIZE
        self.piece_square_size = None
        self.pieces = {}
        self.action_buttons = []
        self.modal_buttons = []
        self.button_hover_state = {}
        self.hover_any_button = False
        self.mouse_canvas_pos = (-9999, -9999)
        self.current_cursor_kind = None
        self.history_rect = pygame.Rect(0, 0, 0, 0)
        self.analysis_rect = pygame.Rect(0, 0, 0, 0)
        self.match_panel_rect = None
        self.history_entry_buttons = []
        self.analysis_entry_buttons = []
        self.history_scroll_rows = 0
        self.analysis_scroll_rows = 0
        self.selected_history_ply = None
        self.history_preview_board = None
        self.history_preview_move = None
        self.selected_analysis_move = None
        self.preview_move = None
        self.preview_move_started_at = 0
        self.last_analysis_click = {"key": None, "time_ms": 0}
        self.ai_paused = False
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self.side_time_stats = {
            chess.WHITE: {"total": 0.0, "moves": 0, "last": 0.0},
            chess.BLACK: {"total": 0.0, "moves": 0, "last": 0.0},
        }
        self.analysis_cache = self._empty_analysis_cache()
        self.analysis_cache_by_color = self._empty_analysis_cache_by_color()
        self.mcts_analysis_cache_by_color = self._empty_analysis_cache_by_color()
        self.completed_mcts_snapshots = OrderedDict()
        self.pending_ai_choice = None
        self.last_ai_choice = None
        self._resize_canvas(*self.restore_window_size)
        self._update_viewport()
        try:
            pygame.display.set_window_minimum_size((WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT))
        except Exception:
            pass

        pygame.display.set_caption(f"Chess AI {self.version}")
        self.clock = pygame.time.Clock()
        if bool(initial_window_maximized) or initial_window_size is None:
            self._set_window_maximized(True)
        
        # Load piece images
        self._ensure_piece_surfaces()
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.human_color = human_color  # Only used in human_vs_ai mode
        self.use_mcts = self.mcts_enabled  # Can be toggled during game
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        self.move_san_history = []
        play_cfg = config.get("play", {}) if isinstance(config, dict) else {}
        self.sound_enabled = bool(play_cfg.get("enable_sounds", True))
        try:
            self.sound_volume = float(play_cfg.get("sound_volume", 0.12))
        except (TypeError, ValueError):
            self.sound_volume = 0.12
        self.sound_volume = max(0.0, min(1.0, self.sound_volume))
        self.sound_effects = {}
        self.sfx_channel = None
        self._last_move_sfx_ms = -100000
        self._init_audio()
        
        #  v4.2: Board history for neural network input
        # Store chess.Board objects (not tensors) for history
        self.board_history = []
        self.position_counts = _position_counts_for_history([], self.board)
        self.position_history_complete = True
        
        # Flip board for black
        self.flipped = False

        # PGN autosave state
        self.save_games_pgn = bool(config.get("play", {}).get("save_games_pgn", False))
        if self.save_games_pgn:
            base_dir = script_dir.parent
            self.games_dir = resolve_games_dir(config, base_dir)
        else:
            self.games_dir = None
        self.initial_fen = self.board.fen()
        self.current_game_saved = False
        self.game_index = 1
        if self._match_enabled():
            self._prime_visible_match_opening()
        self._refresh_analysis_cache()

    def _resize_canvas(self, width, height):
        self.base_width = max(WINDOW_MIN_WIDTH, int(width))
        self.base_height = max(WINDOW_MIN_HEIGHT, int(height))
        self.canvas = pygame.Surface((self.base_width, self.base_height))
        self.background = build_ui_background(self.base_width, self.base_height)
        self._update_fonts()
        self._refresh_layout()
        self._ensure_piece_surfaces()

    def _update_fonts(self):
        self.ui_scale = workspace_scale(self.base_width, self.base_height)
        density_scale = max(0.82, min(2.2, 1.0 + 0.6 * (self.ui_scale - 1.0)))
        self.text_font = pygame.font.SysFont(
            "Segoe UI", max(18, int(round(28 * self.ui_scale))), bold=True
        )
        self.medium_font = pygame.font.SysFont(
            "Segoe UI", max(15, int(round(20 * self.ui_scale))), bold=True
        )
        self.small_font = pygame.font.SysFont(
            "Segoe UI", max(12, int(round(17 * self.ui_scale)))
        )
        self.tiny_font = pygame.font.SysFont(
            "Segoe UI", max(11, int(round(15 * self.ui_scale)))
        )
        self.inspector_small_font = pygame.font.SysFont(
            "Segoe UI", max(11, int(round(14 * density_scale)))
        )
        self.inspector_tiny_font = pygame.font.SysFont(
            "Segoe UI", max(10, int(round(12 * density_scale)))
        )

    def _ensure_piece_surfaces(self):
        if int(self.square_size) <= 0:
            return
        if self.piece_square_size == int(self.square_size):
            return
        self.pieces = create_piece_surfaces(int(self.square_size))
        self.piece_square_size = int(self.square_size)

    def _init_audio(self):
        if not self.sound_enabled:
            return

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            mixer_cfg = pygame.mixer.get_init()
            sample_rate = int(mixer_cfg[0]) if mixer_cfg else 44100
            current_channels = max(1, pygame.mixer.get_num_channels())
            if current_channels < 8:
                pygame.mixer.set_num_channels(8)
            self.sfx_channel = pygame.mixer.Channel(1)
            self.sfx_channel.set_volume(1.0)

            self.sound_effects = {
                "move": self._build_sound_from_sequence(
                    [(560.0, 36, 0.32), (0.0, 8, 0.0), (720.0, 24, 0.26)],
                    sample_rate=sample_rate,
                ),
                "check": self._build_sound_from_sequence(
                    [(760.0, 44, 0.38), (980.0, 72, 0.42)],
                    sample_rate=sample_rate,
                ),
                "mate": self._build_sound_from_sequence(
                    [(980.0, 54, 0.44), (780.0, 60, 0.40), (620.0, 100, 0.36)],
                    sample_rate=sample_rate,
                ),
            }
        except Exception:
            self.sound_enabled = False
            self.sound_effects = {}
            self.sfx_channel = None

    def _build_sound_from_sequence(self, segments, sample_rate=None):
        if sample_rate is None:
            mixer_cfg = pygame.mixer.get_init()
            sample_rate = int(mixer_cfg[0]) if mixer_cfg else 44100
        waveform = array("h")
        for frequency_hz, duration_ms, amplitude in segments:
            waveform.extend(
                self._build_tone_samples(
                    frequency_hz=frequency_hz,
                    duration_ms=duration_ms,
                    amplitude=amplitude,
                    sample_rate=sample_rate,
                )
            )

        if not waveform:
            waveform.extend([0, 0, 0, 0])

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(waveform.tobytes())
        wav_buffer.seek(0)
        return pygame.mixer.Sound(file=wav_buffer)

    def _build_tone_samples(self, frequency_hz, duration_ms, amplitude, sample_rate):
        sample_count = max(1, int(float(duration_ms) * float(sample_rate) / 1000.0))
        if frequency_hz <= 0:
            return array("h", [0] * sample_count)

        clamped_amp = max(0.0, min(1.0, float(amplitude)))
        gain = 32767.0 * self.sound_volume * clamped_amp
        attack_samples = max(2, int(sample_count * 0.12))
        release_samples = max(2, int(sample_count * 0.20))
        release_start = max(attack_samples, sample_count - release_samples)
        phase_step = 2.0 * math.pi * float(frequency_hz) / float(sample_rate)
        phase = 0.0

        samples = array("h")
        for idx in range(sample_count):
            if idx < attack_samples:
                envelope = idx / float(max(1, attack_samples - 1))
            elif idx >= release_start:
                tail_idx = idx - release_start
                envelope = 1.0 - (tail_idx / float(max(1, release_samples - 1)))
                envelope = max(0.0, envelope)
            else:
                envelope = 1.0

            value = int(gain * envelope * math.sin(phase))
            value = max(-32767, min(32767, value))
            samples.append(value)
            phase += phase_step

        if samples:
            samples[-1] = 0
        return samples

    def _play_sfx(self, key):
        if not self.sound_enabled:
            return
        sound = self.sound_effects.get(key)
        if sound is None:
            return
        now_ms = pygame.time.get_ticks()
        if key == "move":
            if now_ms - self._last_move_sfx_ms < 45:
                return
            self._last_move_sfx_ms = now_ms

        try:
            if self.sfx_channel is not None:
                if self.sfx_channel.get_busy() and key in ("check", "mate"):
                    self.sfx_channel.fadeout(16)
                self.sfx_channel.play(sound, fade_ms=4)
            else:
                sound.play(fade_ms=4)
        except Exception:
            pass

    def _refresh_layout(self):
        layout = compute_workspace_layout(self.base_width, self.base_height)
        self.workspace_layout = layout
        self.header_rect = layout["header"]
        self.left_panel_rect = layout["board_area"]
        self.board_rect = layout["board"]
        self.board_frame_rect = layout["board_frame"]
        self.right_panel_rect = layout["inspector"]
        self.top_player_rect = layout["top_player"]
        self.bottom_player_rect = layout["bottom_player"]
        self.inspector_tab_rects = layout["tabs"]
        show_match_summary = self._match_enabled() and self.inspector_tab != "match"
        self.match_summary_rect = layout["match_summary"] if show_match_summary else None
        body_key = "inspector_body_with_summary" if show_match_summary else "inspector_body_without_summary"
        self.inspector_body_rect = layout[body_key]
        self.square_size = max(1, self.board_rect.width // 8)

    def _update_viewport(self):
        win_w, win_h = self.screen.get_size()
        self.viewport_scale = 1.0
        self.viewport_rect = pygame.Rect(0, 0, win_w, win_h)

    def _window_to_canvas(self, pos):
        if pos is None:
            return None
        if not self.viewport_rect.collidepoint(pos):
            return None
        x = int(pos[0] - self.viewport_rect.left)
        y = int(pos[1] - self.viewport_rect.top)
        x = max(0, min(self.base_width - 1, x))
        y = max(0, min(self.base_height - 1, y))
        return x, y

    def _set_window_size(self, width, height):
        width = max(WINDOW_MIN_WIDTH, int(width))
        height = max(WINDOW_MIN_HEIGHT, int(height))
        self.maximized = False
        self.restore_window_size = (width, height)
        self.screen = pygame.display.set_mode((width, height), self.display_flags)
        pygame.event.pump()
        _position_native_window(width, height)
        self._resize_canvas(width, height)
        self._update_viewport()

    def _sync_window_surface(self):
        current_surface = pygame.display.get_surface()
        if current_surface is not None:
            self.screen = current_surface
        self._resize_canvas(*self.screen.get_size())
        self._update_viewport()

    def _set_window_maximized(self, enabled=True):
        if enabled:
            if not self.maximized:
                self.restore_window_size = self.screen.get_size()
            self.maximized = bool(_maximize_native_window())
            pygame.event.pump()
            self._sync_window_surface()
        else:
            self.screen = pygame.display.set_mode(self.restore_window_size, self.display_flags)
            pygame.event.pump()
            _position_native_window(*self.restore_window_size)
            self.maximized = False
        self._resize_canvas(*self.screen.get_size())
        self._update_viewport()

    def get_window_state(self):
        return {
            "window_size": [int(self.screen.get_width()), int(self.screen.get_height())],
            "window_maximized": bool(self.maximized),
            "window_fullscreen": False,
        }
    
    def square_to_coords(self, square):
        """Convert chess square to screen coordinates"""
        file = square % 8
        rank = square // 8
        
        if self.flipped:
            x = self.board_rect.left + (7 - file) * self.square_size
            y = self.board_rect.top + rank * self.square_size
        else:
            x = self.board_rect.left + file * self.square_size
            y = self.board_rect.top + (7 - rank) * self.square_size
        
        return x, y
    
    def coords_to_square(self, x, y):
        """Convert screen coordinates to chess square"""
        if not self.board_rect.collidepoint(x, y):
            return None

        local_x = x - self.board_rect.left
        local_y = y - self.board_rect.top
        file = local_x // self.square_size
        rank = 7 - (local_y // self.square_size)
        
        if self.flipped:
            file = 7 - file
            rank = 7 - rank
        
        return rank * 8 + file
    
    def draw_board(self):
        """Draw the chessboard"""
        scale = self.ui_scale
        board_outer = self.board_frame_rect.copy()
        radius = max(8, round(12 * scale))
        pygame.draw.rect(self.canvas, (22, 30, 32), board_outer, border_radius=radius)
        pygame.draw.rect(
            self.canvas,
            UI_BORDER_STRONG,
            board_outer,
            width=max(1, round(scale)),
            border_radius=radius,
        )

        for rank in range(8):
            for file in range(8):
                x = self.board_rect.left + file * self.square_size
                y = self.board_rect.top + rank * self.square_size
                
                color = BOARD_LIGHT if (rank + file) % 2 == 0 else BOARD_DARK
                pygame.draw.rect(self.canvas, color, (x, y, self.square_size, self.square_size))
        
        # Draw file/rank labels
        label_font = pygame.font.SysFont("Segoe UI", max(11, int(self.square_size * 0.18)))
        files = "abcdefgh"
        ranks = "87654321"
        
        for i in range(8):
            # File labels (bottom)
            file_label_color = (55, 73, 58) if (7 + i) % 2 == 0 else (235, 237, 225)
            file_label = label_font.render(
                files[i] if not self.flipped else files[7-i],
                True, file_label_color
            )
            self.canvas.blit(
                file_label,
                (
                    self.board_rect.left + i * self.square_size + self.square_size - max(8, int(self.square_size * 0.18)),
                    self.board_rect.bottom - max(16, int(self.square_size * 0.22)),
                ),
            )
            
            # Rank labels (left)
            left_file = 7 if self.flipped else 0
            rank_label_color = (55, 73, 58) if (i + left_file) % 2 == 0 else (235, 237, 225)
            rank_label = label_font.render(
                ranks[i] if not self.flipped else ranks[7-i],
                True, rank_label_color
            )
            self.canvas.blit(
                rank_label,
                (
                    self.board_rect.left + max(3, int(self.square_size * 0.06)),
                    self.board_rect.top + i * self.square_size + max(2, int(self.square_size * 0.05)),
                ),
            )
        
        display_board = self._display_board()
        # A history preview highlights exactly the move selected in the list;
        # live play continues against ``self.board`` without being rewound.
        last_move = self.history_preview_move if self._history_preview_active() else (
            self.move_history[-1] if self.move_history else None
        )
        if last_move is not None:
            for square, color in (
                (last_move.from_square, SELECTED),
                (last_move.to_square, HIGHLIGHT),
            ):
                x, y = self.square_to_coords(square)
                s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                s.fill(color)
                self.canvas.blit(s, (x, y))
        
        # Highlight selected square
        if not self._history_preview_active() and self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(SELECTED)
            self.canvas.blit(s, (x, y))
        
        # Highlight legal moves
        for move in self.legal_moves if not self._history_preview_active() else ():
            x, y = self.square_to_coords(move.to_square)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            
            if self.board.piece_at(move.to_square):
                # Capture - draw semi-transparent red overlay
                s.fill(CAPTURE_MOVE)
            else:
                # Normal move - draw circle
                radius = max(6, int(self.square_size * 0.16))
                pygame.draw.circle(
                    s,
                    LEGAL_MOVE,
                    (self.square_size // 2, self.square_size // 2),
                    radius,
                )
            
            self.canvas.blit(s, (x, y))

        self._draw_preview_move()
    
    def draw_pieces(self):
        """Draw chess pieces"""
        display_board = self._display_board()
        for square in chess.SQUARES:
            piece = display_board.piece_at(square)
            if piece:
                x, y = self.square_to_coords(square)
                piece_surface = self.pieces[piece.symbol()]
                self.canvas.blit(piece_surface, (x, y))

    def _ai_thinking_icon(self):
        """Animated text icon for AI thinking state."""
        frames = ["[   ]", "[=  ]", "[== ]", "[===]"]
        idx = (pygame.time.get_ticks() // 180) % len(frames)
        return frames[idx]

    @staticmethod
    def _fit_text(font, text, max_width):
        return fit_ui_text(font, text, max_width)

    @staticmethod
    def _format_version(metadata):
        if not metadata:
            return "n/a"
        version = metadata.get("version")
        if not version:
            return "n/a"
        version_text = str(version)
        return version_text if version_text.startswith("v") else f"v{version_text}"

    @staticmethod
    def _format_duration(seconds):
        try:
            total_seconds = max(0.0, float(seconds))
        except (TypeError, ValueError):
            total_seconds = 0.0
        if total_seconds >= 60.0:
            minutes = int(total_seconds // 60)
            secs = total_seconds - minutes * 60
            return f"{minutes}m {secs:04.1f}s"
        return f"{total_seconds:.1f}s"

    def _current_thinking_elapsed(self):
        if not self.ai_thinking or self.ai_thinking_started_at is None:
            return 0.0
        end_time = self.ai_pause_started_at if self.ai_paused and self.ai_pause_started_at is not None else time.perf_counter()
        return max(0.0, end_time - self.ai_thinking_started_at)

    def _thinking_dots(self):
        frames = [".", "..", "..."]
        idx = (pygame.time.get_ticks() // 350) % len(frames)
        return frames[idx]

    @staticmethod
    def _empty_analysis_cache():
        return {"fen": None, "rows": [], "side": None, "label": ""}

    @staticmethod
    def _empty_analysis_cache_by_color():
        return {
            chess.WHITE: {"rows": [], "label": "White", "fen": None},
            chess.BLACK: {"rows": [], "label": "Black", "fen": None},
        }

    def _new_match_stats(self):
        match_enabled = self._match_enabled()
        return {
            "completed": 0,
            "attempted": 0,
            "unresolved": 0,
            "invalid_results": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "model1_wins": 0,
            "model2_wins": 0,
            "model1_score_x2": 0,
            "model2_score_x2": 0,
            "model1_as_white": 0,
            "model1_as_black": 0,
            "visible_counted": False,
            "background_done": not match_enabled,
            "background_starting": bool(match_enabled and self.match_total_games > 1),
            "background_attempted": 0,
            "background_completed": 0,
            "background_unresolved": 0,
            "background_invalid_results": 0,
            "background_total": max(0, self.match_total_games - 1) if match_enabled else 0,
            "background_remaining": max(0, self.match_total_games - 1) if match_enabled else 0,
            "background_workers": 0,
            "active_games_per_worker": 0,
            "central_inference": False,
            "match_engine": "shared_eval",
            "runtime_load_s": 0.0,
            "runtime_status": "idle",
            "status": "starting" if match_enabled and self.match_total_games > 1 else "visible_running" if match_enabled else "idle",
            "error": None,
            "plies_total": 0,
            "started_at": self.match_started_at,
            "finished_at": None,
        }

    def _match_enabled(self):
        return (
            self.game_mode == "ai_vs_ai"
            and self.model1 is not None
            and self.model2 is not None
            and self.match_total_games >= 1
            and not self.match_detached
        )

    def _history_path_key(self, ply):
        ply = max(0, min(int(ply), len(self.move_history)))
        return (str(self.initial_fen), tuple(move.uci() for move in self.move_history[:ply]))

    def _analysis_model_for_color(self, color):
        if self.game_mode == "ai_vs_ai":
            return self.model1 if color == chess.WHITE else self.model2
        if self.game_mode == "human_vs_human":
            return None
        return self.model1

    def _analysis_position_key(self, board, ply, color, model=None):
        if model is None:
            model = self._analysis_model_for_color(color)
        if model is None:
            return None
        return (board.fen(), bool(color), id(model), self._history_path_key(ply))

    @staticmethod
    def _freeze_analysis_rows(rows):
        return tuple(
            tuple((str(key), value) for key, value in dict(row).items())
            for row in list(rows or [])[:ANALYSIS_DISPLAY_ROWS]
        )

    @staticmethod
    def _thaw_analysis_rows(rows):
        return [dict(row) for row in rows or ()]

    @staticmethod
    def _build_search_policy_rows(search_result, root_board):
        if search_result is None:
            return []
        policy = getattr(search_result, "improved_policy", None) or getattr(
            search_result, "training_policy", None
        )
        if not policy:
            return []
        scored = []
        for native_move, probability in policy.items():
            move = _python_move(native_move)
            if move is None or move not in root_board.legal_moves:
                continue
            try:
                probability = max(0.0, float(probability))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(probability):
                continue
            scored.append((move, probability))
        total = sum(probability for _, probability in scored)
        if total <= 0.0:
            return []
        selected = _python_move(search_result.selected_move)
        # This is a policy target, so keep its actual probability order.
        # The selected MCTS action is marked in-place when rows are rendered.
        scored.sort(key=lambda item: item[1], reverse=True)
        rows = []
        for move, probability in scored[:ANALYSIS_DISPLAY_ROWS]:
            try:
                san = root_board.san(move)
            except (ValueError, AssertionError):
                san = move.uci()
            rows.append(
                {
                    "move": move.uci(),
                    "san": san,
                    "probability": probability / total,
                    "selected": move == selected,
                }
            )
        return rows

    def _snapshot_cache_value(self, snapshot, *, context):
        """Return detached display data; stored snapshots are never mutated."""
        return {
            "fen": snapshot["root_fen"],
            "history_key": snapshot["root_history_key"],
            "root_fen": snapshot["root_fen"],
            "root_ply": snapshot["root_ply"],
            "rows": self._thaw_analysis_rows(snapshot["policy_rows"]),
            "policy_source": snapshot["policy_source"],
            "policy_root_fen": snapshot["root_fen"],
            "policy_history_key": snapshot["root_history_key"],
            "policy_root_ply": snapshot["root_ply"],
            "mcts_rows": self._thaw_analysis_rows(snapshot["mcts_rows"]),
            "source": snapshot["policy_source"],
            "mcts_source": "completed-search",
            "mcts_context": context,
            "mcts_root_fen": snapshot["root_fen"],
            "mcts_history_key": snapshot["root_history_key"],
            "mcts_root_ply": snapshot["root_ply"],
            "mcts_color": snapshot["color"],
            "policy_color": snapshot["color"],
            "selected_move": snapshot["selected_move"],
            "selected_san": snapshot["selected_san"],
            "model_label": snapshot["model_label"],
            "status": "ready",
        }

    def _record_completed_mcts_snapshot(
        self, *, color, model, root_board, root_ply, history_key,
        search_result, policy_rows=None, policy_source="nn-policy",
    ):
        """Keep only compact, immutable display data for a completed root."""
        if model is None or search_result is None or search_result.selected_move is None:
            return None
        key = self._analysis_position_key(
            root_board, root_ply, color, model=model
        )
        if key is None:
            return None
        selected_move = _python_move(search_result.selected_move)
        if selected_move is None or selected_move not in root_board.legal_moves:
            return None
        try:
            selected_san = root_board.san(selected_move)
        except (ValueError, AssertionError):
            selected_san = selected_move.uci()
        if policy_rows is None:
            policy_rows = ()
        snapshot = {
            "root_fen": root_board.fen(),
            "root_ply": int(root_ply),
            "root_history_key": (str(history_key[0]), tuple(history_key[1])),
            "root_moves_uci": tuple(history_key[1]),
            "model_id": id(model),
            "model_label": self._side_info(color).get("name", "AI"),
            "color": bool(color),
            "selected_move": selected_move.uci(),
            "selected_san": selected_san,
            "policy_source": str(policy_source),
            "policy_rows": self._freeze_analysis_rows(policy_rows),
            "mcts_rows": self._freeze_analysis_rows(
                self._build_mcts_analysis_rows(search_result)
            ),
        }
        self.completed_mcts_snapshots[key] = snapshot
        self.completed_mcts_snapshots.move_to_end(key)
        while len(self.completed_mcts_snapshots) > MCTS_SEARCH_SNAPSHOT_LIMIT:
            self.completed_mcts_snapshots.popitem(last=False)
        return snapshot

    def _latest_mcts_snapshot_for_live_line(self, current_ply):
        if not self.completed_mcts_snapshots:
            return None
        current_path = self._history_path_key(current_ply)
        current_moves = current_path[1]
        for snapshot in reversed(self.completed_mcts_snapshots.values()):
            root_path = snapshot["root_history_key"]
            root_moves = snapshot["root_moves_uci"]
            if (
                root_path[0] == current_path[0]
                and len(root_moves) <= len(current_moves)
                and current_moves[:len(root_moves)] == root_moves
            ):
                model = self._analysis_model_for_color(snapshot["color"])
                if model is not None and id(model) == snapshot["model_id"]:
                    return snapshot
        return None

    def _prior_mcts_snapshot_for_history(self, current_ply):
        """Use only the immediately preceding decision that made this ply."""
        current_path = self._history_path_key(current_ply)
        current_moves = current_path[1]
        if current_ply <= 0:
            return None
        for snapshot in reversed(self.completed_mcts_snapshots.values()):
            root_ply = snapshot["root_ply"]
            if (
                root_ply + 1 != current_ply
                or snapshot["root_history_key"][0] != current_path[0]
                or snapshot["root_moves_uci"] != current_moves[:root_ply]
                or snapshot["selected_move"] != current_moves[root_ply]
            ):
                continue
            model = self._analysis_model_for_color(snapshot["color"])
            if model is not None and id(model) == snapshot["model_id"]:
                return snapshot
        return None

    def _prune_mcts_snapshots_for_line(self, retained_moves):
        retained_uci = tuple(move.uci() for move in retained_moves)
        initial_fen = str(self.initial_fen)
        root_fens = [chess.Board(self.initial_fen).fen()]
        root_board = chess.Board(self.initial_fen)
        for move in retained_moves:
            root_board.push(move)
            root_fens.append(root_board.fen())
        for key, snapshot in list(self.completed_mcts_snapshots.items()):
            root_moves = snapshot["root_moves_uci"]
            root_ply = snapshot["root_ply"]
            if (
                snapshot["root_history_key"][0] != initial_fen
                or root_ply > len(retained_uci)
                or root_moves != retained_uci[:root_ply]
                or snapshot["root_fen"] != root_fens[root_ply]
            ):
                self.completed_mcts_snapshots.pop(key, None)

    def _history_preview_active(self):
        return self.history_preview_board is not None and self.selected_history_ply is not None

    def _display_board(self):
        return self.history_preview_board if self._history_preview_active() else self.board

    def _history_preview_status(self):
        if not self._history_preview_active():
            return None
        ply = int(self.selected_history_ply)
        san = self.move_san_history[ply - 1] if 0 < ply <= len(self.move_san_history) else "start"
        if san == "start":
            return "Preview · starting position"
        move_number = (ply + 1) // 2
        mover_suffix = "." if ply % 2 else "..."
        return f"Preview · after {move_number}{mover_suffix}{san}"

    def _set_history_preview(self, ply):
        try:
            ply = int(ply)
        except (TypeError, ValueError):
            return False
        if not 0 <= ply <= len(self.move_history):
            return False
        preview = chess.Board(self.initial_fen)
        for move in self.move_history[:ply]:
            if move not in preview.legal_moves:
                return False
            preview.push(move)
        self.history_preview_board = preview
        self.history_preview_move = self.move_history[ply - 1] if ply else None
        self.selected_history_ply = ply
        self.selected_square = None
        self.legal_moves = []
        self.selected_analysis_move = None
        self.selected_analysis_candidate = None
        self.preview_move = None
        self._queue_position_analysis(preview, ply)
        return True

    def _return_to_live_board(self):
        self.history_preview_board = None
        self.history_preview_move = None
        self.selected_square = None
        self.legal_moves = []
        self.selected_analysis_move = None
        self.selected_analysis_candidate = None
        self.preview_move = None
        self.preview_analysis_pending_key = None
        self._cancel_preview_analysis()

    def _history_prev(self):
        current_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        return self._set_history_preview(max(0, int(current_ply or 0) - 1))

    def _history_next(self):
        if not self._history_preview_active():
            return False
        current_ply = int(self.selected_history_ply or 0)
        if current_ply >= len(self.move_history):
            self._return_to_live_board()
            return True
        return self._set_history_preview(current_ply + 1)

    def _cancel_preview_analysis(self):
        self.preview_analysis_generation += 1
        turn = self.preview_analysis_turn
        if turn is not None:
            turn.invalidate()
            if not turn.is_done:
                self.draining_preview_analysis_turn = turn
            self.preview_analysis_turn = None
        self.preview_analysis_pending = None

    def _queue_position_analysis(self, board, ply):
        """Analyze an inspected position without touching a live MCTS tree."""
        model = self._analysis_model_for_color(board.turn)
        if model is None:
            return False
        try:
            ply = int(ply)
        except (TypeError, ValueError):
            return False
        ply = max(0, min(ply, len(self.move_history)))
        position = board.copy(stack=True)
        history = [item.copy(stack=True) for item in self.board_history[:ply]]
        key = self._analysis_position_key(position, ply, position.turn, model=model)
        if key is None:
            return False
        snapshot = self.completed_mcts_snapshots.get(key)
        if snapshot is not None:
            self.preview_analysis_cache[key] = self._snapshot_cache_value(
                snapshot, context="position"
            )
            return True
        cached = self.preview_analysis_cache.get(key)
        if cached is not None and cached.get("status") == "ready":
            return True
        # Exact live search output (including MCTS visits/Q) is preferable to
        # another forward pass when the whole path, model, side and FEN match.
        for cache in (
            self.analysis_cache_by_color.get(position.turn, {}),
            self.mcts_analysis_cache_by_color.get(position.turn, {}),
        ):
            if (
                cache.get("fen") == position.fen()
                and cache.get("history_key") == self._history_path_key(ply)
                and cache.get("rows")
            ):
                self.preview_analysis_cache[key] = {
                    "fen": position.fen(),
                    "color": position.turn,
                    "history_key": self._history_path_key(ply),
                    "rows": list(self.analysis_cache_by_color.get(position.turn, {}).get("rows", [])),
                    "mcts_rows": list(self.mcts_analysis_cache_by_color.get(position.turn, {}).get("rows", [])),
                    "status": "ready",
                    "source": "cached-search",
                }
                return True

        if self.preview_analysis_pending_key == key:
            active = self.preview_analysis_turn
            if active is not None and getattr(active, "preview_request", {}).get("key") == key:
                return True
            if self.preview_analysis_pending is not None and self.preview_analysis_pending.get("key") == key:
                return True

        active = self.preview_analysis_turn
        if active is not None and getattr(active, "preview_request", {}).get("key") != key:
            active.invalidate()
            if not active.is_done:
                self.draining_preview_analysis_turn = active
            self.preview_analysis_turn = None

        if not self._native_board_has_legal_moves(position):
            self.preview_analysis_cache[key] = {
                "fen": position.fen(), "color": position.turn,
                "history_key": self._history_path_key(ply), "rows": [],
                "mcts_rows": [], "status": "terminal", "source": "nn-policy",
            }
            return True
        request = {
            "key": key,
            "fen": position.fen(),
            "color": position.turn,
            "ply": ply,
            "board": position,
            "history": history,
            "model": model,
            "history_positions": self._history_positions_for_model(model),
        }
        self.preview_analysis_cache[key] = {
            "fen": position.fen(),
            "color": position.turn,
            "history_key": self._history_path_key(ply),
            "rows": [],
            "mcts_rows": [],
            "status": "pending",
            "source": "nn-policy",
        }
        if len(self.preview_analysis_cache) > 64:
            oldest_key = next(iter(self.preview_analysis_cache))
            if oldest_key != key:
                self.preview_analysis_cache.pop(oldest_key, None)
        self.preview_analysis_pending_key = key
        self.preview_analysis_pending = request
        self._tick_preview_analysis()
        return True

    @staticmethod
    def _native_board_has_legal_moves(board):
        return bool(list(board.legal_moves))

    def _tick_preview_analysis(self):
        draining = self.draining_preview_analysis_turn
        if draining is not None and draining.is_done:
            draining.wait(timeout=0)
            self.draining_preview_analysis_turn = None

        turn = self.preview_analysis_turn
        if turn is not None and turn.is_done:
            turn.wait(timeout=0)
            self.preview_analysis_turn = None
            request = getattr(turn, "preview_request", None)
            result = turn.result or {}
            if (
                request is not None
                and not turn.cancelled.is_set()
                and turn.error is None
                and request["key"] == self.preview_analysis_pending_key
            ):
                cache = self.preview_analysis_cache.setdefault(request["key"], {})
                cache.update(
                    fen=request["fen"],
                    color=request["color"],
                    history_key=request["key"][3],
                    rows=list(result.get("analysis_rows") or []),
                    mcts_rows=[],
                    status="ready",
                    source="nn-policy",
                )
                if self.preview_analysis_pending is not None and self.preview_analysis_pending.get("key") == request["key"]:
                    self.preview_analysis_pending = None
            elif request is not None and turn.error is not None and not turn.cancelled.is_set():
                cache = self.preview_analysis_cache.setdefault(request["key"], {})
                cache.update(
                    fen=request["fen"], color=request["color"],
                    history_key=request["key"][3], rows=[], mcts_rows=[],
                    status="error", error=f"{type(turn.error).__name__}: {turn.error}",
                    source="nn-policy",
                )

        request = self.preview_analysis_pending
        if (
            request is None
            or self.preview_analysis_turn is not None
            or self.draining_preview_analysis_turn is not None
        ):
            return
        self.preview_analysis_pending = None
        self.preview_analysis_pending_key = request["key"]
        self.preview_analysis_generation += 1
        generation = self.preview_analysis_generation

        def compute():
            return self._get_network_move(
                request["model"],
                board_snapshot=request["board"],
                history_snapshot=request["history"],
                history_positions=request["history_positions"],
            )

        turn = PlayAITurn(
            generation=generation,
            board_fen=request["fen"],
            compute=compute,
            min_visible_move_seconds=0.0,
        )
        turn.preview_request = request
        self.preview_analysis_turn = turn.start()

    def _display_analysis_cache(self, color):
        display_board = self._display_board()
        display_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        key = self._analysis_position_key(display_board, display_ply, color)
        if key is None:
            return {"rows": [], "mcts_rows": [], "fen": display_board.fen(), "status": "unavailable"}
        display_fen = display_board.fen()
        history_key = self._history_path_key(display_ply)
        policy = self.analysis_cache_by_color.get(color, {})
        mcts = self.mcts_analysis_cache_by_color.get(color, {})
        policy_matches = policy.get("fen") == display_fen and policy.get("history_key") == history_key
        mcts_matches = mcts.get("fen") == display_fen and mcts.get("history_key") == history_key

        # A completed result is authoritative for its exact root and survives
        # later per-color cache refreshes and unrelated NN forwards.
        exact_snapshot = self.completed_mcts_snapshots.get(key)
        if exact_snapshot is not None:
            result = self._snapshot_cache_value(exact_snapshot, context="position")
            if policy_matches and policy.get("rows"):
                result["rows"] = list(policy["rows"])
                result["policy_color"] = color
                result["policy_source"] = "nn-policy"
            else:
                exact_preview = self.preview_analysis_cache.get(key, {})
                if exact_preview.get("source") == "nn-policy" and exact_preview.get("rows"):
                    result["rows"] = list(exact_preview["rows"])
                    result["policy_color"] = color
                    result["policy_source"] = "nn-policy"
            result["fen"] = display_fen
            result["history_key"] = history_key
            return result

        cached = self.preview_analysis_cache.get(key)
        if cached is not None:
            result = dict(cached)
        else:
            result = {"rows": [], "mcts_rows": [], "status": "empty"}

        # NN policy belongs to the displayed position; it may arrive before
        # or after the MCTS task, but must never replace a completed search.
        if policy_matches and policy.get("rows"):
            result["rows"] = list(policy["rows"])
            result["policy_color"] = color
            result["policy_source"] = "nn-policy"
        if mcts_matches and mcts.get("rows") and not self._history_preview_active():
            result["mcts_rows"] = list(mcts["rows"])
            result["mcts_root_fen"] = display_fen
            result["mcts_history_key"] = history_key
            result["mcts_root_ply"] = display_ply
            result["mcts_color"] = color
            result["mcts_context"] = "position"

        result["fen"] = display_fen
        result["history_key"] = history_key
        if self._history_preview_active():
            # Historical positions may show only their own saved search or
            # their own NN fallback. The sole permitted mismatch is the
            # exact previous decision that selected this displayed move.
            if not result.get("mcts_rows"):
                prior = self._prior_mcts_snapshot_for_history(display_ply)
                if prior is not None:
                    result["mcts_rows"] = self._thaw_analysis_rows(prior["mcts_rows"])
                    result["rows"] = self._thaw_analysis_rows(prior["policy_rows"])
                    result["mcts_source"] = "completed-search"
                    result["mcts_context"] = "prior-move"
                    result["policy_source"] = prior["policy_source"]
                    result["policy_color"] = prior["color"]
                    result["policy_root_fen"] = prior["root_fen"]
                    result["policy_history_key"] = prior["root_history_key"]
                    result["policy_root_ply"] = prior["root_ply"]
                    result["mcts_root_fen"] = prior["root_fen"]
                    result["mcts_history_key"] = prior["root_history_key"]
                    result["mcts_root_ply"] = prior["root_ply"]
                    result["mcts_color"] = prior["color"]
                    result["selected_move"] = prior["selected_move"]
                    result["selected_san"] = prior["selected_san"]
                    result["model_label"] = prior["model_label"]
            if result.get("rows") or result.get("mcts_rows"):
                result["status"] = "ready"
            return result

        # Live play keeps the last completed result visible while the new
        # root is waiting/running. Its distinct root metadata drives labels
        # and candidate clicks so those moves cannot be mistaken for current.
        latest = self._latest_mcts_snapshot_for_live_line(display_ply)
        if latest is not None:
            # The visible policy and tree rows must describe the same saved
            # search root. Current-position NN rows may belong to the other
            # side after the last move and must not be shown beside stale MCTS.
            result["rows"] = self._thaw_analysis_rows(latest["policy_rows"])
            result["mcts_rows"] = self._thaw_analysis_rows(latest["mcts_rows"])
            result["mcts_source"] = "completed-search"
            result["mcts_context"] = "last-search"
            result["policy_source"] = latest["policy_source"]
            result["policy_color"] = latest["color"]
            result["policy_root_fen"] = latest["root_fen"]
            result["policy_history_key"] = latest["root_history_key"]
            result["policy_root_ply"] = latest["root_ply"]
            result["mcts_root_fen"] = latest["root_fen"]
            result["mcts_history_key"] = latest["root_history_key"]
            result["mcts_root_ply"] = latest["root_ply"]
            result["mcts_color"] = latest["color"]
            result["selected_move"] = latest["selected_move"]
            result["selected_san"] = latest["selected_san"]
            result["model_label"] = latest["model_label"]
        if result.get("rows") or result.get("mcts_rows"):
            result["status"] = "ready"
        return result

    def _visible_match_active(self):
        return (
            self._match_enabled()
            and not self.visible_match_finished
            and not self.visible_match_start_pending
        )

    def _board_result_for_match(self):
        if not self._match_enabled():
            return self.board.result() if self.board.is_game_over() else "*"
        if not self.board.is_game_over(claim_draw=True):
            return "*"
        return self.board.result(claim_draw=True)

    def _prime_visible_match_opening(self):
        """Start visible game zero at the standard position with no prefix."""
        if not self._match_enabled():
            return
        self.board = chess.Board()
        self.initial_fen = self.board.fen()
        self.board_history = []
        self.position_counts = _position_counts_for_history([], self.board)
        self.position_history_complete = True
        self.move_history = []
        self.move_san_history = []
        self.current_game_saved = False
        self.pending_ai_choice = None
        self.last_ai_choice = None
        self.game_over = bool(self._board_result_for_match() != "*")
        self._reset_selection_state()
        self._reset_ai_state(paused=False)
        if self.draining_ai_turn is None and self.ai_turn is None:
            self._sync_mcts_histories()
            self.pending_mcts_reset = False
        else:
            self.pending_mcts_reset = True
        self._reset_analysis_storage()
        self._refresh_analysis_cache()
        if not self.match_start_pending:
            self._ensure_match_session_started()
        if self.game_over:
            self._record_visible_match_result()

    def _ensure_match_session_started(self):
        """Start games 1..N-1 as soon as the first visible position is ready."""
        if not self._match_enabled() or self.match_start_pending:
            return None
        if self.match_session is None:
            self.match_session = self._new_match_session()
        if self.match_total_games > 1 and self.match_session.thread is None:
            self._start_remaining_match_games(self.match_session)
        return self.match_session

    def _record_visible_match_result(self):
        if not self._visible_match_active() or not self.game_over:
            return
        result = self._board_result_for_match()
        if result == "*":
            self._record_visible_match_error(
                "Visible game reached a terminal state without a chess result."
            )
            return
        winner_model = None
        if result == "1-0":
            winner_model = self.model1_name
        elif result == "0-1":
            winner_model = self.model2_name

        self.visible_match_finished = True
        self.visible_match_result = result
        self.visible_match_winner = winner_model
        self.game_over = True
        # _apply_move saves a completed PGN before reaching this method. Keep
        # the save idempotent for terminal openings as well, without writing
        # the same visible game twice.
        self._save_current_game()
        session = self._ensure_match_session_started()
        if session is None:
            return
        session.record_visible_game(
            result=result,
            plies=len(self.move_history),
            model1_as_white=True,
            winner_model=winner_model,
            final_fen=self.board.fen(),
            moves_uci=[move.uci() for move in self.move_history],
            moves_san=self.move_san_history,
            mcts_searches=self.visible_mcts_searches,
        )
        with self.match_lock:
            self.match_stats["visible_counted"] = True
            self.match_stats["status"] = "waiting_background" if self.match_total_games > 1 else "completed"
            self.match_stats["background_done"] = self.match_total_games <= 1
            self.match_stats["finished_at"] = time.perf_counter() if self.match_total_games <= 1 else None
        if self.match_total_games <= 1:
            session.finish_without_runner()

    def _record_visible_match_error(self, error):
        if not self._visible_match_active():
            return
        self.visible_match_finished = True
        self.visible_match_error = str(error)
        self.game_over = True
        session = self._ensure_match_session_started()
        if session is None:
            return
        session.record_visible_error(
            self.visible_match_error,
            final_fen=self.board.fen(),
            moves_uci=[move.uci() for move in self.move_history],
            moves_san=self.move_san_history,
            mcts_searches=self.visible_mcts_searches,
        )
        with self.match_lock:
            self.match_stats["visible_counted"] = True
            self.match_stats["status"] = "error"
            self.match_stats["error"] = self.visible_match_error
            self.match_stats["background_done"] = True
            self.match_stats["finished_at"] = time.perf_counter()
        if self.match_total_games <= 1:
            session.finish_without_runner()
        else:
            session.request_stop()

    def _new_match_session(self):
        raw_max_moves = self.config.get("play", {}).get("match_max_moves")
        try:
            max_moves = None if raw_max_moves is None else max(1, int(raw_max_moves))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "play.match_max_moves must be null or a positive ply count"
            ) from exc
        def model_manifest(model, path, meta):
            model_spec = dict(getattr(model, "model_spec", {}) or {}) if model is not None else {}
            checkpoint_path = path or meta.get("path")
            try:
                checkpoint_path = str(Path(checkpoint_path).resolve()) if checkpoint_path else None
            except (OSError, TypeError, ValueError):
                checkpoint_path = str(checkpoint_path) if checkpoint_path else None
            source = {}
            for key in ("model_name", "architecture_id", "mtime_ts", "size_mb"):
                if meta.get(key) is not None:
                    source[key] = _json_safe(meta[key])
            return {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": "unavailable_not_computed",
                # Persist the loaded checkpoint's complete lightweight contract
                # (encoder/history, policy codec, value head, model kwargs).
                # model_spec contains architecture metadata, not tensor weights.
                "model_spec": _json_safe(model_spec or {
                    "architecture_id": meta.get("architecture_id")
                }),
                "checkpoint_metadata": source,
            }

        def mcts_manifest(mcts):
            runtime_config = getattr(mcts, "config", {}) if mcts is not None else {}
            rl = dict((runtime_config or {}).get("reinforcement_learning", {}) or {})
            engine = getattr(mcts, "_multi", None) if mcts is not None else None
            resolved_settings = {
                key: value for key, value in rl.items()
                if str(key).startswith(("mcts_", "eval_mcts_"))
            }
            # Normalization stores the effective values flat under
            # reinforcement_learning; several defaults are applied inside the
            # engine, so capture those resolved attributes as well.
            engine_fields = {
                "mcts_gumbel_c_visit": "gumbel_c_visit",
                "mcts_gumbel_c_scale": "gumbel_c_scale",
                "mcts_gumbel_max_considered_actions": "gumbel_max_considered_actions",
                "mcts_gumbel_scale": "gumbel_scale",
                "mcts_gumbel_eval_scale": "gumbel_eval_scale",
                "mcts_gumbel_q_range_floor": "gumbel_q_range_floor",
                "mcts_gumbel_target_temperature": "gumbel_target_temperature",
                "mcts_gumbel_use_mixed_value": "gumbel_use_mixed_value",
                "mcts_dynamic_budget_enabled": "dynamic_budget_enabled",
                "mcts_dynamic_budget_min": "dynamic_budget_minimum",
                "mcts_dynamic_budget_max_multiplier": "dynamic_budget_maximum_multiplier",
                "eval_mcts_dynamic_budget_enabled": "eval_easy_cut_enabled",
                "eval_mcts_dynamic_budget_min_fraction": "eval_easy_cut_minimum_fraction",
                "eval_mcts_dynamic_budget_difficulty_threshold": "eval_easy_cut_difficulty_threshold",
                "eval_mcts_dynamic_budget_chunk": "eval_easy_cut_chunk",
                "mcts_reuse_tree": "reuse_tree",
                "mcts_tree_reuse_visit_credit_enabled": "tree_reuse_visit_credit_enabled",
                "mcts_cache_history_tensors": "cache_history_tensors",
                "mcts_single_root_batch_cap": "single_root_batch_cap",
                "mcts_batch_size": "eval_batch_size",
                "mcts_tactical_priors_enabled": "tactical_priors_enabled",
            }
            for config_key, attr in engine_fields.items():
                if engine is not None and hasattr(engine, attr):
                    resolved_settings[config_key] = getattr(engine, attr)
            return {
                "enabled": mcts is not None,
                "resolved_settings": _json_safe(resolved_settings),
                "input_contract": _json_safe({
                    key: getattr(engine, attr)
                    for key, attr in (
                        ("encoder_id", "input_encoder_id"),
                        ("history_positions", "history_positions"),
                        ("repetition_plane_mode", "repetition_plane_mode"),
                    )
                    if engine is not None and hasattr(engine, attr)
                }),
            }

        return PlayMatchSession(
            total_games=self.match_total_games,
            model1_name=self.model1_name,
            model2_name=self.model2_name,
            output_dir=self.match_result_dir,
            match_settings={
                "use_mcts_model1": bool(self.use_mcts_white),
                "use_mcts_model2": bool(self.use_mcts_black),
                "simulations_model1": int(self.mcts_simulations_white),
                "simulations_model2": int(self.mcts_simulations_black),
                "max_moves": max_moves,
                "first_game_visible": True,
                "first_game_index": 0,
                "visible_game_opening": "standard_startpos_no_prefix",
                "visible_game_initial_fen": chess.Board().fen(),
                "background_game_opening_policy": "canonical_paired_openings",
                "background_game_index_offset": 1,
                "models": {
                    "model1": model_manifest(self.model1, self.model1_checkpoint_path, self.model1_meta),
                    "model2": model_manifest(self.model2, self.model2_checkpoint_path, self.model2_meta),
                },
                "mcts_runtime": {
                    "model1": mcts_manifest(self.mcts1),
                    "model2": mcts_manifest(self.mcts2),
                },
            },
        )

    def _start_remaining_match_games(self, session):
        remaining_games = max(0, self.match_total_games - 1)
        if remaining_games <= 0:
            session.finish_without_runner()
            return

        raw_max_moves = session.match_settings.get("max_moves")

        def run_match(**callbacks):
            return run_model_match(
                self.model1,
                self.model2,
                self.config,
                self.device,
                num_games=remaining_games,
                game_index_offset=1,
                use_mcts_model1=self.use_mcts_white,
                use_mcts_model2=self.use_mcts_black,
                simulations_model1=self.mcts_simulations_white,
                simulations_model2=self.mcts_simulations_black,
                max_moves=raw_max_moves,
                stop_event=callbacks["stop_event"],
                progress_callback=callbacks["progress_callback"],
                runtime_callback=callbacks["runtime_callback"],
                show_progress=self.console_verbose,
                verbose=self.console_verbose,
            )

        session.start(run_match)

    def _match_snapshot(self):
        if self.match_session is not None:
            snapshot = self.match_session.snapshot()
        elif self.last_match_session is not None:
            snapshot = self.last_match_session.snapshot()
        elif self.draining_match_session is not None:
            snapshot = self.draining_match_session.snapshot()
        else:
            with self.match_lock:
                snapshot = dict(self.match_stats)
            snapshot["total"] = int(self.match_total_games)
        snapshot["detached_for_analysis"] = bool(self.match_detached)
        snapshot["manual_analysis_mode"] = bool(self.manual_analysis_mode)
        return snapshot

    def _tick_match_lifecycle(self):
        previous = self.draining_match_session
        if previous is not None:
            if previous.is_done:
                previous.wait(timeout=0)
                self.draining_match_session = None
            elif not self.match_start_pending:
                return
        if not self.match_start_pending:
            return
        previous = self.draining_match_session
        if previous is not None and not previous.is_done:
            return
        if previous is not None:
            previous.wait(timeout=0)
        self.draining_match_session = None
        self.match_start_pending = False
        self.visible_match_start_pending = False
        if self._match_enabled() and not self.visible_match_finished:
            with self.match_lock:
                self.match_stats["status"] = "visible_running"
                self.match_stats["background_done"] = False
                self.match_stats["background_starting"] = False
            if self.pending_mcts_reset:
                self._sync_mcts_histories()
                self.pending_mcts_reset = False
            self._ensure_match_session_started()
            self._refresh_analysis_cache()
            if self.game_over:
                self._record_visible_match_result()

    def _stop_background_match(self, reason="play screen exited"):
        session = self.match_session
        if session is None:
            return
        self.last_match_session = session
        if self._match_enabled() and not self.visible_match_finished:
            self.visible_match_finished = True
            session.record_visible_cancel(
                reason,
                final_fen=self.board.fen(),
                moves_uci=[move.uci() for move in self.move_history],
                moves_san=self.move_san_history,
                mcts_searches=self.visible_mcts_searches,
            )
            if self.match_total_games <= 1:
                session.finish_without_runner()
        if not session.is_done:
            session.request_stop()
            self.draining_match_session = session
        self.match_session = None

    def _detach_match_for_analysis(self, reason="match detached for manual analysis"):
        if self._match_enabled():
            self._stop_background_match(reason=reason)
            self.match_detached = True
            self.match_start_pending = False
            self.visible_match_start_pending = False
        self.manual_analysis_mode = True

    def _can_resume_ai(self):
        if not self.manual_analysis_mode or not self.ai_paused or self.game_over:
            return False
        if self.pending_manual_move is not None:
            return False
        if self._history_preview_active() or self.ai_turn is not None:
            return False
        if self.draining_ai_turn is not None and not self.draining_ai_turn.is_done:
            return False
        if self.draining_match_session is not None and not self.draining_match_session.is_done:
            return False
        if self.draining_preview_analysis_turn is not None and not self.draining_preview_analysis_turn.is_done:
            return False
        return True

    def _resume_ai(self):
        if not self._can_resume_ai():
            return False
        if self.pending_mcts_reset:
            self._sync_mcts_histories()
            self.pending_mcts_reset = False
        self.manual_analysis_mode = False
        self.ai_status_error = None
        self._reset_ai_state(paused=False)
        return True

    def _reset_analysis_storage(self):
        self.analysis_cache = self._empty_analysis_cache()
        self.analysis_cache_by_color = self._empty_analysis_cache_by_color()
        self.mcts_analysis_cache_by_color = self._empty_analysis_cache_by_color()
        self.completed_mcts_snapshots = OrderedDict()
        self.preview_analysis_cache = {}
        self.preview_analysis_pending_key = None
        self.preview_analysis_pending = None
        self.pending_ai_choice = None
        self.last_ai_choice = None

    def _reset_selection_state(self):
        self._cancel_preview_analysis()
        self.selected_square = None
        self.legal_moves = []
        self.selected_history_ply = None
        self.history_preview_board = None
        self.history_preview_move = None
        self.selected_analysis_move = None
        self.selected_analysis_candidate = None
        self.preview_move = None
        self.history_scroll_rows = 0
        self.analysis_scroll_rows = 0

    def _reset_ai_state(self, paused=False):
        self.ai_thinking = False
        self.ai_paused = bool(paused)
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self._pending_ai_compute_duration = None

    def _set_preview_move(self, color, move_uci, fen=None):
        if not move_uci:
            self.preview_move = None
            self.preview_move_started_at = 0
            return
        self.preview_move = {"color": color, "move": move_uci, "fen": fen}
        self.preview_move_started_at = pygame.time.get_ticks()

    @staticmethod
    def _board_from_fen(fen):
        if not fen:
            return None
        try:
            return chess.Board(str(fen))
        except Exception:
            return None

    def _find_analysis_rewind_ply(self, fen):
        if not fen:
            return None
        if self.board.fen() == fen:
            return len(self.move_history)
        initial = chess.Board(self.initial_fen)
        if initial.fen() == fen:
            return 0
        for idx in range(min(len(self.board_history), len(self.move_history)) - 1, -1, -1):
            hist_board = self.board_history[idx]
            if not hist_board:
                continue
            after_move = hist_board.copy(stack=True)
            try:
                after_move.push(self.move_history[idx])
            except (IndexError, ValueError):
                continue
            if after_move.fen() == fen:
                return idx + 1
        return None

    def _select_analysis_entry(self, entry):
        if not entry:
            return False
        display_board = self._display_board()
        display_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        entry_fen = entry.get("fen")
        entry_history_key = entry.get("history_key")
        if (
            entry_fen != display_board.fen()
            or entry_history_key != self._history_path_key(display_ply)
        ):
            # A live last-search row may refer to the immediately previous
            # root. Select it by opening that exact retained history position;
            # never apply its move to the current board by accident.
            try:
                root_ply = int(entry.get("root_ply"))
                root_initial_fen, root_moves = entry_history_key
            except (TypeError, ValueError):
                return False
            live_path = self._history_path_key(len(self.move_history))
            if (
                root_initial_fen != live_path[0]
                or root_ply != len(root_moves)
                or root_ply > len(live_path[1])
                or live_path[1][:root_ply] != tuple(root_moves)
                or not self._set_history_preview(root_ply)
            ):
                return False
            display_board = self._display_board()
            display_ply = self.selected_history_ply
            if (
                display_board.fen() != entry_fen
                or self._history_path_key(display_ply) != entry_history_key
            ):
                return False
        try:
            move = chess.Move.from_uci(str(entry.get("move")))
        except ValueError:
            return False
        color = entry.get("color")
        if color != display_board.turn or move not in display_board.legal_moves:
            return False
        self.selected_analysis_candidate = dict(entry)
        self.selected_analysis_move = (color, move.uci())
        self._set_preview_move(color, move.uci(), display_board.fen())
        return True

    def _apply_selected_analysis_choice(self):
        if self.pending_manual_move is not None:
            return False
        candidate = self.selected_analysis_candidate
        if not candidate:
            return False
        display_board = self._display_board()
        display_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        if (
            candidate.get("fen") != display_board.fen()
            or candidate.get("history_key") != self._history_path_key(display_ply)
        ):
            return False
        try:
            move = chess.Move.from_uci(str(candidate.get("move")))
        except ValueError:
            return False
        if candidate.get("color") != display_board.turn or move not in display_board.legal_moves:
            return False
        if not self._branch_to_ply(display_ply, reason="candidate played from inspected position"):
            return False
        if self.board.fen() != candidate.get("fen") or move not in self.board.legal_moves:
            return False
        if self.draining_ai_turn is not None and not self.draining_ai_turn.is_done:
            # The invalidated search still owns the visible MCTS object. Defer
            # tree mutation and move application until its worker drains.
            self.pending_manual_move = {
                "fen": candidate["fen"],
                "move": move.uci(),
                "candidate": dict(candidate),
            }
            self.selected_analysis_candidate = dict(candidate)
            self.selected_analysis_move = (candidate["color"], move.uci())
            return True
        color = self.board.turn
        self._apply_move(move)
        self.selected_analysis_move = (color, move.uci())
        self.selected_analysis_candidate = dict(candidate)
        self._set_preview_move(color, move.uci(), candidate.get("fen"))
        return True

    def _apply_analysis_choice(self, color, move_uci, fen):
        """Compatibility helper for callers that explicitly choose a move."""
        board = self._display_board()
        ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        if fen != board.fen():
            target_ply = self._find_analysis_rewind_ply(fen)
            if target_ply is None or not self._set_history_preview(target_ply):
                return False
            board = self._display_board()
            ply = target_ply
        entry = {
            "color": color,
            "move": move_uci,
            "fen": fen,
            "history_key": self._history_path_key(ply),
            "source": "explicit",
        }
        return self._select_analysis_entry(entry) and self._apply_selected_analysis_choice()

    def _draw_preview_move(self):
        if not self.preview_move:
            return
        move_uci = self.preview_move.get("move")
        if not move_uci:
            return
        preview_board = self._display_board()
        if self.preview_move.get("fen") != preview_board.fen():
            return
        try:
            move = chess.Move.from_uci(str(move_uci))
        except ValueError:
            return

        pulse = 0.55 + 0.45 * math.sin((pygame.time.get_ticks() - self.preview_move_started_at) / 180.0)
        alpha = int(100 + 80 * max(0.0, min(1.0, pulse)))
        from_x, from_y = self.square_to_coords(move.from_square)
        to_x, to_y = self.square_to_coords(move.to_square)
        moving_piece = preview_board.piece_at(move.from_square)
        is_capture = move in preview_board.legal_moves and preview_board.is_capture(move)
        overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        overlay.fill((98, 171, 255, alpha))
        self.canvas.blit(overlay, (from_x, from_y))

        overlay_to = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        if is_capture:
            overlay_to.fill((255, 126, 94, alpha))
        else:
            overlay_to.fill((255, 209, 94, alpha))
        self.canvas.blit(overlay_to, (to_x, to_y))

        start = (from_x + self.square_size // 2, from_y + self.square_size // 2)
        end = (to_x + self.square_size // 2, to_y + self.square_size // 2)
        line_color = (255, 166, 77) if is_capture else (244, 205, 74)
        line_width = max(3, self.square_size // 14)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 0:
            ux = dx / length
            uy = dy / length
            arrow_size = max(12, self.square_size // 5)
            shaft_end = (
                end[0] - ux * arrow_size * 0.8,
                end[1] - uy * arrow_size * 0.8,
            )
            pygame.draw.line(self.canvas, line_color, start, shaft_end, line_width)
            perp_x = -uy
            perp_y = ux
            arrow_half = max(6, arrow_size * 0.38)
            arrow_points = [
                end,
                (
                    shaft_end[0] + perp_x * arrow_half,
                    shaft_end[1] + perp_y * arrow_half,
                ),
                (
                    shaft_end[0] - perp_x * arrow_half,
                    shaft_end[1] - perp_y * arrow_half,
                ),
            ]
            pygame.draw.polygon(self.canvas, line_color, arrow_points)
        else:
            pygame.draw.circle(self.canvas, line_color, end, max(8, self.square_size // 9))

        if is_capture:
            cross_pad = max(10, self.square_size // 5)
            cross_left = to_x + cross_pad
            cross_top = to_y + cross_pad
            cross_right = to_x + self.square_size - cross_pad
            cross_bottom = to_y + self.square_size - cross_pad
            cross_width = max(3, self.square_size // 15)
            pygame.draw.line(self.canvas, (255, 230, 230), (cross_left, cross_top), (cross_right, cross_bottom), cross_width)
            pygame.draw.line(self.canvas, (255, 230, 230), (cross_right, cross_top), (cross_left, cross_bottom), cross_width)

        if moving_piece is not None:
            piece_surface = self.pieces.get(moving_piece.symbol())
            if piece_surface is not None:
                ghost_piece = piece_surface.copy()
                ghost_piece.set_alpha(128)
                self.canvas.blit(ghost_piece, (to_x, to_y))

    def _toggle_pause(self):
        if self.manual_analysis_mode:
            self._resume_ai()
            return
        if self.game_mode != "ai_vs_ai":
            return
        if not self.ai_paused:
            self.ai_paused = True
            if self.ai_thinking and self.ai_pause_started_at is None:
                self.ai_pause_started_at = time.perf_counter()
            if self.ai_turn is None and not self.game_over:
                # A paused game may have no active search yet. Keep NN policy
                # candidates available without claiming that MCTS selected a
                # move it has not searched.
                self._queue_position_analysis(self.board, len(self.move_history))
        else:
            if self.ai_thinking and self.ai_pause_started_at is not None and self.ai_thinking_started_at is not None:
                paused_for = max(0.0, time.perf_counter() - self.ai_pause_started_at)
                self.ai_thinking_started_at += paused_for
            self.ai_pause_started_at = None
            self.ai_paused = False

    def _current_analysis_model(self):
        if self.game_mode == "human_vs_human":
            return None
        if self.game_mode == "ai_vs_ai":
            return self.model1 if self.board.turn == chess.WHITE else self.model2
        return self.model1

    def _analysis_target_color(self):
        if self.game_mode == "human_vs_human":
            return None
        if self.game_mode == "human_vs_ai":
            if self.board.turn == self.human_color:
                return None
            return self.board.turn
        return self.board.turn

    def _side_has_analysis(self, color):
        if self.game_mode == "human_vs_human":
            return False
        if self.manual_analysis_mode:
            return self._analysis_model_for_color(color) is not None
        if self.game_mode == "human_vs_ai":
            # Keep the model's last completed search visible on the human's
            # turn as well as during the engine's next root search.
            return self.model1 is not None
        if self.game_mode == "ai_vs_ai":
            return bool((color == chess.WHITE and self.model1 is not None) or (color == chess.BLACK and self.model2 is not None))
        return False

    def _side_has_visible_analysis(self, color):
        if not self._side_has_analysis(color):
            return False
        cache = self._display_analysis_cache(color)
        policy_rows = list(cache.get("rows", []))
        mcts_rows = list(cache.get("mcts_rows", []))
        return bool(
            policy_rows
            or mcts_rows
            or cache.get("status") == "pending"
            or self._pending_ai_choice_matches_display(color)
        )

    def _side_uses_mcts(self, color):
        if self.game_mode == "ai_vs_ai":
            return bool(self.use_mcts_white if color == chess.WHITE else self.use_mcts_black)
        return bool(self.use_mcts)

    def _build_mcts_analysis_rows(self, search_result):
        """Format the exact search that selected the played move."""
        if not search_result or not search_result.visits:
            return []

        visit_counts = search_result.visits
        total_visits = float(sum(max(0.0, float(v)) for v in visit_counts.values()))
        if total_visits <= 0.0:
            return []

        selected_move = search_result.selected_move
        selection_scores = search_result.selection_score

        def _row_order(item):
            native_move, visits = item
            score = float(selection_scores.get(native_move, -float("inf")))
            return (
                float(visits),
                score,
            )

        rows = []
        for native_move, visits in sorted(visit_counts.items(), key=_row_order, reverse=True):
            move = _python_move(native_move)
            try:
                san = self.board.san(move)
            except Exception:
                san = move.uci()
            visit_value = max(0.0, float(visits))
            rows.append(
                {
                    "move": move.uci(),
                    "san": san,
                    "probability": visit_value / total_visits,
                    "visits": int(round(visit_value)),
                    "completed_q": search_result.completed_q.get(native_move),
                    "selection_score": selection_scores.get(native_move),
                    "selected": native_move == selected_move,
                }
            )
            if len(rows) >= ANALYSIS_DISPLAY_ROWS:
                break
        return rows

    def _pending_ai_choice_matches_display(self, color):
        choice = self.pending_ai_choice
        if not choice or self._history_preview_active():
            return False
        return (
            choice.get("color") == color
            and choice.get("fen") == self.board.fen()
            and choice.get("history_key") == self._history_path_key(len(self.move_history))
        )

    def _last_ai_choice_matches_display(self, color):
        choice = self.last_ai_choice
        if not choice or self._history_preview_active():
            return False
        return (
            choice.get("resulting_fen") == self.board.fen()
            and choice.get("resulting_history_key") == self._history_path_key(len(self.move_history))
        )

    def _visible_mcts_record(self, *, color, turn, result, played_move, history_key):
        """Build one bounded, post-search record from the completed visible root."""
        if not self._visible_match_active() or played_move is None:
            return None
        search_result = result.get("search_result")
        mcts = self.mcts1 if color == chess.WHITE else self.mcts2
        root = getattr(mcts, "root", None)
        edges = getattr(root, "edges", None)
        if search_result is None or root is None or edges is None:
            return None

        root_board = self.board.copy(stack=True)
        expected_fen = str(getattr(turn, "board_fen", "") or "")
        # Tree reuse may promote a child. Only serialize it when it is exactly
        # the root position searched for this turn.
        try:
            if (
                root_board.fen() != expected_fen
                or native_chess.board_fen(root.board) != expected_fen
                or tuple(move.uci() for move in history_key[1])
                != tuple(move.uci() for move in self.move_history)
            ):
                return None
        except (AttributeError, TypeError, ValueError):
            return None

        metadata = dict(getattr(search_result, "metadata", {}) or {})

        def keyed_values(values):
            output = {}
            for move, value in dict(values or {}).items():
                try:
                    output[native_chess.move_uci(move)] = value
                except (AttributeError, TypeError, ValueError):
                    continue
            return output

        priors = {}
        cumulative = {}
        try:
            for move, prior, visits in zip(
                edges.moves, edges.base_priors, edges.visit_counts
            ):
                uci = native_chess.move_uci(move)
                priors[uci] = _finite_json_number(prior)
                cumulative[uci] = max(0, int(visits))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

        fresh = keyed_values(getattr(search_result, "visits", {}))
        completed_q = keyed_values(getattr(search_result, "completed_q", {}))
        improved = keyed_values(getattr(search_result, "improved_policy", {}))
        training = keyed_values(getattr(search_result, "training_policy", {}))
        scores = keyed_values(getattr(search_result, "selection_score", {}))
        # One row per legal root action (at most 218 in chess); missing values
        # remain explicit nulls/zeros instead of being silently omitted.
        actions = []
        for uci, prior in priors.items():
            score = _finite_json_number(scores.get(uci))
            actions.append({
                "uci": uci,
                "prior": prior,
                "fresh_visits": max(0, int(fresh.get(uci, 0) or 0)),
                "cumulative_visits": cumulative.get(uci, 0),
                "completed_q_root_player": _finite_json_number(completed_q.get(uci)),
                "improved_policy": _finite_json_number(improved.get(uci)),
                "training_policy": _finite_json_number(training.get(uci)),
                "selection_score": score,
                "selection_eligible": score is not None,
            })

        selected_uci = native_chess.move_uci(search_result.selected_move)
        override = metadata.get("selected_move_override")
        try:
            override_uci = native_chess.move_uci(override) if override is not None else None
        except (AttributeError, TypeError, ValueError):
            override_uci = str(override) if override is not None else None
        selected_row = next((row for row in actions if row["uci"] == selected_uci), {})
        config = getattr(mcts, "config", {}) or {}
        rl_config = dict(config.get("reinforcement_learning", {}) or {})
        runtime_keys = (
            "simulation_budget", "simulations_used", "fresh_simulations_used",
            "inherited_visit_credit", "dynamic_budget_enabled",
            "tree_reuse_attempted", "tree_reused", "tree_inherited_visits",
            "tree_reuse_quality", "eval_easy_cut_enabled", "eval_easy_cut_reduced",
            "gumbel_considered_actions", "gumbel_action_budget",
            "gumbel_initial_round_visits", "gumbel_width_adapted", "gumbel_scale",
            "gumbel_q_range_floor", "gumbel_target_temperature",
        )
        runtime = {key: _json_safe(metadata.get(key)) for key in runtime_keys if key in metadata}
        return {
            "ply": len(self.move_history),
            "root_fen": expected_fen,
            "root_history_uci": list(history_key[1]),
            "root_color": "white" if color == chess.WHITE else "black",
            "q_perspective": "root_player",
            "played_uci": played_move.uci(),
            "played_san": root_board.san(played_move),
            "selected_uci": selected_uci,
            "selected_move_override_uci": override_uci,
            "selected_score": selected_row.get("selection_score"),
            "root_noise": False,
            "requested_simulations": max(1, int(result.get("requested_simulations", 1) or 1)),
            "single_root_batch_cap": _json_safe(rl_config.get("mcts_single_root_batch_cap")),
            "runtime": runtime,
            "actions": actions,
        }

    def _publish_ai_turn_analysis(self, turn, result, history_key):
        """Publish completed search alternatives before pacing applies the move."""
        if getattr(turn, "play_analysis_published", False):
            move_uci = getattr(turn, "play_selected_move_uci", None)
            try:
                return chess.Move.from_uci(move_uci) if move_uci else None
            except ValueError:
                return None
        color = self.board.turn
        label = "White" if color == chess.WHITE else "Black"
        move = None
        kind = str(result.get("kind") or "policy")
        if kind == "mcts":
            search_result = result.get("search_result")
            rows = self._build_mcts_analysis_rows(search_result) if search_result is not None else []
            self.mcts_analysis_cache_by_color[color] = {
                "rows": rows,
                "label": label,
                "fen": turn.board_fen,
                "history_key": history_key,
            }
            # The result's selected_move is the engine decision. It is not
            # necessarily the most-visited edge (for example under Gumbel).
            native_move = search_result.selected_move if search_result is not None else None
            move = _python_move(native_move)
            root_board = self.board.copy(stack=True)
            record = self._visible_mcts_record(
                color=color, turn=turn, result=result, played_move=move,
                history_key=history_key,
            )
            if record is not None:
                self.visible_mcts_searches.append(record)
            policy_cache = self.analysis_cache_by_color.get(color, {})
            if (
                policy_cache.get("fen") == turn.board_fen
                and policy_cache.get("history_key") == history_key
                and policy_cache.get("rows")
            ):
                policy_rows = list(policy_cache["rows"])
                policy_source = "nn-policy"
            else:
                policy_rows = self._build_search_policy_rows(search_result, root_board)
                policy_source = (
                    "mcts-improved-policy"
                    if getattr(search_result, "improved_policy", None)
                    else "mcts-policy-target"
                )
            self._record_completed_mcts_snapshot(
                color=color,
                model=self._analysis_model_for_color(color),
                root_board=root_board,
                root_ply=len(self.move_history),
                history_key=history_key,
                search_result=search_result,
                policy_rows=policy_rows,
                policy_source=policy_source,
            )
        else:
            move = result.get("move")
            rows = list(result.get("analysis_rows") or [])
            self.analysis_cache_by_color[color] = {
                "rows": rows,
                "label": label,
                "fen": turn.board_fen,
                "history_key": history_key,
            }

        self.analysis_cache = {
            "fen": turn.board_fen,
            "history_key": history_key,
            "rows": list(self.analysis_cache_by_color.get(color, {}).get("rows", [])),
            "side": color,
            "label": label,
        }
        if move is not None and move in self.board.legal_moves:
            self.pending_ai_choice = {
                "color": color,
                "move": move.uci(),
                "san": self.board.san(move),
                "fen": turn.board_fen,
                "history_key": history_key,
                "kind": kind,
            }
            self._set_preview_move(color, move.uci(), turn.board_fen)
        else:
            self.pending_ai_choice = None
        turn.play_selected_move_uci = move.uci() if move is not None else None
        turn.play_analysis_published = True
        return move

    def _refresh_analysis_cache(self):
        current_fen = self.board.fen()
        current_history_key = self._history_path_key(len(self.move_history))
        if (
            self.analysis_cache.get("fen") == current_fen
            and self.analysis_cache.get("history_key") == current_history_key
        ):
            return

        if self.manual_analysis_mode:
            analysis_color = self.board.turn
            side_label = "White" if analysis_color == chess.WHITE else "Black"
            self.analysis_cache = {
                "fen": current_fen,
                "history_key": current_history_key,
                "rows": [],
                "side": analysis_color,
                "label": side_label,
            }
            self.analysis_cache_by_color[analysis_color] = {
                "rows": [], "label": side_label, "fen": current_fen,
                "history_key": current_history_key,
            }
            self.mcts_analysis_cache_by_color[analysis_color] = {
                "rows": [], "label": side_label, "fen": current_fen,
                "history_key": current_history_key,
            }
            self._queue_position_analysis(self.board, len(self.move_history))
            return

        # In human-vs-AI, after AI makes a move it's the human turn.
        # Keep the last AI analysis visible instead of clearing the panel.
        if self.game_mode == "human_vs_ai" and self.board.turn == self.human_color:
            ai_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
            side_label = "White" if ai_color == chess.WHITE else "Black"
            ai_rows = list(self.analysis_cache_by_color.get(ai_color, {}).get("rows", []))
            self.analysis_cache = {
                "fen": current_fen,
                "history_key": current_history_key,
                "rows": ai_rows,
                "side": ai_color,
                "label": side_label,
            }
            return

        model = self._current_analysis_model()
        analysis_color = self._analysis_target_color()
        side_label = "White" if analysis_color == chess.WHITE else "Black"
        if model is None or analysis_color is None:
            self.analysis_cache = {"fen": current_fen, "history_key": current_history_key, "rows": [], "side": analysis_color, "label": side_label if analysis_color is not None else ""}
            return

        # Policy analysis for the side to move is produced by its actual AI
        # search below. Running another forward pass on the Pygame thread here
        # would freeze input immediately after a human move or restart.
        if self._ai_should_move() or self._match_enabled():
            self.analysis_cache = {"fen": current_fen, "history_key": current_history_key, "rows": [], "side": analysis_color, "label": side_label}
            self.analysis_cache_by_color[analysis_color] = {
                "rows": [],
                "label": side_label,
                "fen": current_fen,
                "history_key": current_history_key,
            }
            self.mcts_analysis_cache_by_color[analysis_color] = {
                "rows": [],
                "label": side_label,
                "fen": current_fen,
                "history_key": current_history_key,
            }
            if not (self._match_enabled() and self.game_over) and (
                self.ai_paused or self._side_uses_mcts(analysis_color)
            ):
                self._queue_position_analysis(self.board, len(self.move_history))
            return

        try:
            history_positions = self._history_positions_for_model(model)
            board_tensor = torch.FloatTensor(
                self._build_history_tensor(self.board, history_positions=history_positions, model=model)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with self.inference_lock:
                    policy_logits, _ = model(
                        board_tensor,
                        apply_log_softmax=False,
                    )
                logits = policy_logits.float().cpu().numpy()[0]

            native_board = _native_board(self.board)
            legal_moves = native_chess.legal_moves(native_board)
            if not legal_moves:
                self.analysis_cache = {"fen": current_fen, "history_key": current_history_key, "rows": [], "side": analysis_color, "label": side_label}
                self.analysis_cache_by_color[analysis_color] = {"rows": [], "label": side_label, "fen": current_fen, "history_key": current_history_key}
                return

            scored_moves = []
            for move in legal_moves:
                idx = move_to_index(move, native_board)
                scored_moves.append((move, float(logits[idx])))

            max_logit = max(score for _, score in scored_moves)
            exp_scores = []
            score_sum = 0.0
            for move, score in scored_moves:
                exp_score = math.exp(score - max_logit)
                exp_scores.append((move, exp_score))
                score_sum += exp_score

            rows = []
            sorted_exp_scores = sorted(exp_scores, key=lambda item: item[1], reverse=True)
            for native_move, exp_score in sorted_exp_scores:
                move = _python_move(native_move)
                probability = (exp_score / score_sum) if score_sum > 0 else 0.0
                rows.append(
                    {
                        "move": move.uci(),
                        "san": self.board.san(move),
                        "probability": probability,
                    }
                )
                if len(rows) >= ANALYSIS_DISPLAY_ROWS:
                    break
            self.analysis_cache = {"fen": current_fen, "history_key": current_history_key, "rows": rows, "side": analysis_color, "label": side_label}
            self.analysis_cache_by_color[analysis_color] = {"rows": rows, "label": side_label, "fen": current_fen, "history_key": current_history_key}
            # MCTS rows are filled by ai_move from the exact search used to
            # choose the move. A second display-only tree used to reset every
            # position and could disagree with the played MCTS.
            self.mcts_analysis_cache_by_color[analysis_color] = {
                "rows": [],
                "label": side_label,
                "fen": current_fen,
                "history_key": current_history_key,
            }
        except Exception:
            self.analysis_cache = {"fen": current_fen, "history_key": current_history_key, "rows": [], "side": analysis_color, "label": side_label}
            self.analysis_cache_by_color[analysis_color] = {"rows": [], "label": side_label, "fen": current_fen, "history_key": current_history_key}
            self.mcts_analysis_cache_by_color[analysis_color] = {"rows": [], "label": side_label, "fen": current_fen, "history_key": current_history_key}

    def _side_info(self, color):
        side_name = "White" if color == chess.WHITE else "Black"
        if self.game_mode == "human_vs_human":
            return {
                "side": side_name,
                "is_human": True,
                "name": "Human",
                "version": "n/a",
                "elo": "n/a",
            }

        if self.game_mode == "human_vs_ai" and color == self.human_color:
            return {
                "side": side_name,
                "is_human": True,
                "name": "You",
                "version": "n/a",
                "elo": "n/a",
            }

        if self.game_mode == "ai_vs_ai":
            is_white = color == chess.WHITE
            model_name = self.model1_name if is_white else self.model2_name
            model_meta = self.model1_meta if is_white else self.model2_meta
        else:
            model_name = self.model1_name
            model_meta = self.model1_meta

        uses_mcts = self._side_uses_mcts(color)
        simulations = self.mcts_simulations_white if color == chess.WHITE else self.mcts_simulations_black

        return {
            "side": side_name,
            "is_human": False,
            "name": Path(str(model_name or "model.pt")).stem,
            "version": self._format_version(model_meta),
            "elo": format_elo_summary(model_meta),
            "search_mode": f"MCTS · {simulations} sims" if uses_mcts else "NN only",
        }

    def _draw_card(self, rect, fill=(40, 49, 63), border=(88, 105, 132)):
        draw_ui_card(
            self.canvas,
            rect,
            fill=fill,
            border=border,
            radius=max(9, round(12 * self.ui_scale)),
        )

    def _player_card_lines(self, side_info, compact_level=0):
        compact_level = max(0, min(2, int(compact_level)))
        name_font = self.medium_font if compact_level <= 0 else self.small_font
        lines = [(self.tiny_font, "title"), (name_font, side_info["name"])]
        if side_info["is_human"]:
            lines.append((self.tiny_font, "Local player"))
        else:
            version = side_info["version"]
            identity = side_info["elo"] if version == "n/a" else f"{version}  |  {side_info['elo']}"
            lines.append((self.tiny_font, identity))
            lines.append((self.tiny_font, side_info["search_mode"]))
        return lines

    def _draw_player_card(self, rect, title, side_info, active=False, compact_level=0):
        accent = UI_ACCENT if active else UI_BORDER
        draw_ui_card(
            self.canvas,
            rect,
            fill=UI_SURFACE_RAISED if active else UI_SURFACE,
            border=accent,
            radius=12,
            accent=UI_ACCENT if active else None,
        )

        inner_left = rect.left + 16
        inner_width = rect.width - 32
        y = rect.top + 10
        side_color = UI_TEXT
        subtitle_color = UI_ACCENT if active else UI_MUTED
        meta_color = UI_MUTED
        line_gap = max(5, self.tiny_font.get_height() // 3)

        row_specs = self._player_card_lines(side_info, compact_level=compact_level)
        for idx, (font, text) in enumerate(row_specs):
            if idx == 0:
                rendered = font.render(title.upper(), True, subtitle_color)
            elif idx == 1:
                identity = text if side_info["is_human"] else self._compact_model_label(text)
                rendered = font.render(fit_middle_text(font, identity, inner_width), True, side_color)
            else:
                color = UI_ACCENT if not side_info["is_human"] and text == side_info.get("search_mode") else meta_color
                rendered = font.render(self._fit_text(font, text, inner_width), True, color)
            self.canvas.blit(rendered, (inner_left, y))
            y += rendered.get_height() + line_gap

        dot_color = UI_ACCENT if active else UI_BORDER_STRONG
        pygame.draw.circle(self.canvas, dot_color, (rect.right - 16, rect.top + 16), 4)

    def _player_card_height(self, side_info, compact_level=0):
        top_pad = 10
        bottom_pad = 12
        line_gap = max(5, self.tiny_font.get_height() // 3)
        lines = self._player_card_lines(side_info, compact_level=compact_level)
        total_height = top_pad + bottom_pad
        for idx, (font, _) in enumerate(lines):
            total_height += font.get_height()
            if idx < len(lines) - 1:
                total_height += line_gap
        return total_height

    def _analysis_row_height(self, show_details=False):
        font_height = self.inspector_tiny_font.get_height()
        return max(19, font_height + (5 if show_details else 4))

    def _analysis_card_height(self, color, row_limit=None):
        if not self._side_has_analysis(color):
            return 0
        cache = self._display_analysis_cache(color)
        policy_rows = list(cache.get("rows", []))
        mcts_rows = list(cache.get("mcts_rows", []))
        if row_limit is None:
            row_limit = ANALYSIS_DISPLAY_ROWS
        row_limit = max(1, int(row_limit))
        row_count = max(1, min(row_limit, max(len(policy_rows), len(mcts_rows))))
        policy_section_h = self.tiny_font.get_height() + 8 + row_count * (self._analysis_row_height() + 2)
        mcts_section_h = self.tiny_font.get_height() + 8 + row_count * (self._analysis_row_height(show_details=True) + 2)
        sections_h = policy_section_h
        if self._side_uses_mcts(color):
            sections_h += 8 + mcts_section_h
        return 48 + sections_h + 8

    def _draw_analysis_column(
        self, rect, rows, cache_fen, color, label, row_limit, fill_color,
        show_visits=False, source="policy", history_key=None, root_ply=None,
    ):
        scale = self.workspace_layout["density"]
        font = self.inspector_tiny_font
        clip = self.canvas.get_clip()
        self.canvas.set_clip(rect.clip(self.canvas.get_rect()))
        label_surface = font.render(
            self._fit_text(font, label, rect.width), True, UI_MUTED
        )
        self.canvas.blit(label_surface, (rect.left, rect.top))
        row_h = self._analysis_row_height(show_details=show_visits)
        top = rect.top + font.get_height() + max(2, round(3 * scale))
        if not rows:
            waiting = font.render("—", True, UI_MUTED)
            self.canvas.blit(waiting, (rect.left, top))
            self.canvas.set_clip(clip)
            return

        row_limit = max(1, int(row_limit))
        max_scroll = max(0, len(rows) - row_limit)
        self.analysis_scroll_rows = max(0, min(int(self.analysis_scroll_rows), max_scroll))
        offset = self.analysis_scroll_rows
        visible_rows = rows[offset:offset + row_limit]
        for local_idx, row in enumerate(visible_rows):
            idx = offset + local_idx + 1
            row_rect = pygame.Rect(rect.left, top + local_idx * (row_h + 1), rect.width, row_h)
            probability = max(0.0, min(1.0, float(row.get("probability", 0.0))))
            fill_width = max(8, int((row_rect.width - 2) * probability))
            row_fill = (25, 35, 38) if idx % 2 else (21, 30, 34)
            pygame.draw.rect(self.canvas, row_fill, row_rect, border_radius=max(2, round(3 * scale)))
            pygame.draw.rect(
                self.canvas,
                fill_color,
                pygame.Rect(row_rect.left + 1, row_rect.top + 1, fill_width, row_rect.height - 2),
                border_radius=max(2, round(3 * scale)),
            )
            is_selected = bool(row.get("selected"))
            is_clicked = self.selected_analysis_move == (color, row.get("move"))
            if is_selected or is_clicked:
                pygame.draw.rect(self.canvas, UI_ACCENT, row_rect, width=1, border_radius=max(2, round(3 * scale)))

            rank_x = row_rect.left + max(5, round(7 * scale))
            rank_text = f"*{idx}." if is_selected else f"{idx}."
            rank_surface = font.render(rank_text, True, (225, 178, 104) if is_selected else UI_MUTED)
            top_y = row_rect.centery - rank_surface.get_height() // 2
            self.canvas.blit(rank_surface, (rank_x, top_y))
            metric = None
            metric_color = UI_ACCENT
            if show_visits:
                completed_q = row.get("completed_q")
                try:
                    completed_q = float(completed_q) if completed_q is not None else None
                except (TypeError, ValueError):
                    completed_q = None
                metric = f"N {int(row.get('visits', 0))}"
                if completed_q is not None and math.isfinite(completed_q):
                    metric += f" · Q {completed_q:+.2f}"
                metric_color = (225, 178, 104)
            else:
                metric = f"{probability * 100:.0f}%"
            metric_surface = font.render(metric, True, metric_color)
            right_x = row_rect.right - metric_surface.get_width() - max(4, round(6 * scale))
            rank_right = rank_x + rank_surface.get_width() + max(4, round(5 * scale))
            san_surface = font.render(
                self._fit_text(font, row.get("san", row.get("move", "?")), max(0, right_x - rank_right - 4)),
                True,
                UI_TEXT,
            )
            self.canvas.blit(san_surface, (rank_right, top_y))
            self.canvas.blit(metric_surface, (right_x, top_y))
            self.analysis_entry_buttons.append(
                {
                    "rect": row_rect.copy(),
                    "color": color,
                    "move": row.get("move"),
                    "san": row.get("san"),
                    "fen": cache_fen,
                    "history_key": history_key,
                    "root_ply": root_ply,
                    "source": source,
                    "probability": probability,
                    "visits": row.get("visits"),
                    "completed_q": row.get("completed_q"),
                    "selected": is_selected,
                }
            )
        self.canvas.set_clip(clip)

    def _draw_analysis_card(self, rect, color, active=False, row_limit=None):
        if rect.height <= 0 or not self._side_has_analysis(color):
            return
        display_board = self._display_board()
        display_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        cache = self._display_analysis_cache(color)
        display_fen = display_board.fen()
        if cache.get("fen") != display_fen:
            cache = {"fen": display_fen, "rows": [], "mcts_rows": [], "status": "empty"}
        policy_rows = list(cache.get("rows", []))
        mcts_rows = list(cache.get("mcts_rows", []))
        policy_source = cache.get("policy_source", cache.get("source", "nn-policy"))
        label = "White" if color == chess.WHITE else "Black"
        policy_color = cache.get("policy_color", color)
        policy_root_fen = cache.get("policy_root_fen", display_fen)
        policy_history_key = cache.get("policy_history_key", cache.get("history_key"))
        policy_root_ply = cache.get("policy_root_ply", display_ply)
        mcts_color = cache.get("mcts_color", color)
        border = UI_ACCENT if active else UI_BORDER
        self._draw_card(rect, fill=UI_SURFACE_RAISED, border=border)
        self.analysis_entry_buttons = [entry for entry in self.analysis_entry_buttons if entry.get("color") != color]
        pending_choice = (
            self.pending_ai_choice
            if self._pending_ai_choice_matches_display(color)
            else None
        )
        last_choice = (
            self.last_ai_choice
            if self._last_ai_choice_matches_display(color)
            else None
        )
        if pending_choice:
            title_text = f"Model move · {pending_choice['san']}"
            title_color = UI_ACCENT
        elif cache.get("mcts_context") == "last-search":
            search_side_label = "White" if mcts_color == chess.WHITE else "Black"
            title_text = f"Last {search_side_label} search · played {cache.get('selected_san', '?')}"
            title_color = UI_ACCENT
        elif cache.get("mcts_context") == "prior-move":
            title_text = f"Search before {cache.get('selected_san', '?')}"
            title_color = UI_ACCENT
        elif last_choice:
            title_text = f"Last model move · {last_choice['san']}"
            title_color = UI_ACCENT
        elif mcts_rows and cache.get("selected_san"):
            title_text = f"MCTS chose · {cache['selected_san']}"
            title_color = UI_ACCENT
        else:
            title_text = f"{label} candidates"
            title_color = UI_TEXT
        title_surface = self.inspector_small_font.render(
            self._fit_text(self.inspector_small_font, title_text, rect.width - 16),
            True,
            title_color,
        )
        self.canvas.blit(title_surface, (rect.left + 8, rect.top + 5))

        preview_only = self._history_preview_active() and not mcts_rows
        if preview_only:
            if policy_source == "mcts-improved-policy":
                subtitle = "Saved MCTS policy · no separate NN rows"
            else:
                subtitle = "NN policy · * top move · MCTS unavailable"
        elif cache.get("mcts_context") == "last-search":
            root_ply = int(cache.get("mcts_root_ply", 0) or 0)
            model_label = self._compact_model_label(cache.get("model_label", label))
            subtitle = f"{model_label} · root after {root_ply} plies · select to inspect"
        elif cache.get("mcts_context") == "prior-move":
            root_ply = int(cache.get("mcts_root_ply", 0) or 0)
            model_label = self._compact_model_label(cache.get("model_label", label))
            if policy_source == "mcts-improved-policy":
                subtitle = f"{model_label} · sequential-halving choice · root ply {root_ply}"
            elif policy_source == "mcts-policy-target":
                subtitle = f"{model_label} · MCTS choice · root ply {root_ply}"
            else:
                subtitle = f"{model_label} · root before this ply ({root_ply}) · select to inspect"
        elif mcts_rows:
            subtitle = "MCTS · * marks selected move"
        elif cache.get("source") == "nn-policy":
            if self._side_uses_mcts(color):
                if self.ai_paused and self.ai_turn is None:
                    subtitle = "Policy · MCTS waits"
                elif self.ai_turn is not None and not self.ai_turn.is_done:
                    subtitle = "Policy · MCTS searching"
                else:
                    subtitle = "Policy · no MCTS choice"
            else:
                subtitle = "Policy · * top move"
        else:
            subtitle = "Policy · probability"
        if pending_choice:
            remaining = 0.0
            if self.ai_turn is not None:
                remaining = max(0.0, float(self.ai_turn.presentation_remaining_seconds))
            choice_source = "MCTS selected" if pending_choice.get("kind") == "mcts" else "Policy selected"
            if self.ai_paused:
                subtitle = f"{choice_source} · paused"
            elif remaining > 0.0:
                subtitle = f"{choice_source} · {remaining:.1f}s"
            else:
                subtitle = f"{choice_source} · applying"
        elif last_choice and cache.get("mcts_context") != "last-search":
            subtitle = "Current position candidates"
        title_bottom = rect.top + 5 + self.inspector_small_font.get_height()
        subtitle_surface = self.inspector_tiny_font.render(
            self._fit_text(self.inspector_tiny_font, subtitle, rect.width - 16),
            True,
            UI_ACCENT if pending_choice else UI_MUTED,
        )
        self.canvas.blit(subtitle_surface, (rect.left + 8, title_bottom + 1))

        if row_limit is None:
            row_limit = ANALYSIS_DISPLAY_ROWS
        row_limit = max(1, int(row_limit))
        inner_left = rect.left + 6
        inner_top = title_bottom + subtitle_surface.get_height() + 4
        inner_width = rect.width - 12
        bottom_pad = max(3, round(4 * self.workspace_layout["density"]))
        available_h = max(1, rect.bottom - bottom_pad - inner_top)
        has_policy = bool(policy_rows)
        has_mcts = bool(mcts_rows)
        if not has_policy and not has_mcts:
            status = cache.get("status")
            message = {
                "pending": "Analyzing this exact position…",
                "error": "NN analysis failed",
                "terminal": "No legal moves in this position",
            }.get(status, "Candidates appear after the engine searches this position.")
            message_surface = self.inspector_tiny_font.render(self._fit_text(self.inspector_tiny_font, message, inner_width), True, UI_MUTED)
            self.canvas.blit(message_surface, (inner_left, inner_top + max(0, (available_h - message_surface.get_height()) // 2)))
            return

        policy_h = self._analysis_row_height()
        mcts_h = self._analysis_row_height(show_details=True)
        section_label_h = self.inspector_tiny_font.get_height() + 4
        n_rows = min(row_limit, max(len(policy_rows), len(mcts_rows)))
        stacked_need = section_label_h * (int(has_policy) + int(has_mcts))
        if has_policy:
            stacked_need += n_rows * (policy_h + 2)
        if has_mcts:
            stacked_need += n_rows * (mcts_h + 2)
        side_by_side = has_policy and has_mcts and available_h < stacked_need + 6
        cache_history_key = cache.get("history_key")
        mcts_root_fen = cache.get("mcts_root_fen", display_fen)
        mcts_history_key = cache.get("mcts_history_key", cache_history_key)
        mcts_root_ply = cache.get("mcts_root_ply", display_ply)
        policy_column_label = (
            "Gumbel target · %"
            if policy_source == "mcts-improved-policy"
            else "MCTS target · %"
            if policy_source == "mcts-policy-target"
            else "Policy target · %"
        )
        policy_fill = (48, 91, 79) if color == chess.WHITE else (56, 88, 80)
        mcts_fill = (120, 105, 68) if color == chess.WHITE else (137, 112, 68)

        if side_by_side:
            gap = max(5, round(7 * self.workspace_layout["density"]))
            column_w = max(1, (inner_width - gap) // 2)
            left_rect = pygame.Rect(inner_left, inner_top, column_w, available_h)
            right_rect = pygame.Rect(inner_left + column_w + gap, inner_top, column_w, available_h)
            policy_limit = max(1, min(row_limit, (available_h - section_label_h) // (policy_h + 2)))
            mcts_limit = max(1, min(row_limit, (available_h - section_label_h) // (mcts_h + 2)))
            self._draw_analysis_column(
                left_rect, policy_rows, policy_root_fen, policy_color, policy_column_label, policy_limit,
                policy_fill, source="policy", history_key=policy_history_key,
                root_ply=policy_root_ply,
            )
            self._draw_analysis_column(
                right_rect, mcts_rows, mcts_root_fen, mcts_color,
                f"N / Q ({'White' if mcts_color == chess.WHITE else 'Black'})", mcts_limit,
                mcts_fill, show_visits=True, source="mcts", history_key=mcts_history_key,
                root_ply=mcts_root_ply,
            )
            return

        current_y = inner_top
        if has_policy:
            visible = min(row_limit, max(1, (available_h - section_label_h) // (policy_h + 2)))
            section_h = min(available_h, section_label_h + visible * (policy_h + 2))
            self._draw_analysis_column(
                pygame.Rect(inner_left, current_y, inner_width, section_h),
                policy_rows, policy_root_fen, policy_color,
                (
                    "Gumbel policy target"
                    if policy_source == "mcts-improved-policy"
                    else "MCTS policy target"
                    if policy_source == "mcts-policy-target"
                    else "Policy target"
                ),
                visible, policy_fill, source="policy", history_key=policy_history_key,
                root_ply=policy_root_ply,
            )
            current_y += section_h + 3
            available_h -= section_h + 3
        if has_mcts and available_h > section_label_h:
            visible = min(row_limit, max(1, (available_h - section_label_h) // (mcts_h + 2)))
            section_h = min(available_h, section_label_h + visible * (mcts_h + 2))
            self._draw_analysis_column(
                pygame.Rect(inner_left, current_y, inner_width, section_h),
                mcts_rows, mcts_root_fen, mcts_color,
                f"Visits / Q ({'White' if mcts_color == chess.WHITE else 'Black'})",
                visible, mcts_fill, show_visits=True, source="mcts", history_key=mcts_history_key,
                root_ply=mcts_root_ply,
            )

    def _resolve_left_panel_layout(self, panel, status_rect, top_info, bottom_info, top_display_color, bottom_display_color):
        panel_inner_left = panel.left + 14
        panel_inner_width = panel.width - 28
        start_y = status_rect.bottom + 14
        end_y = panel.bottom - 14
        available_height = max(0, end_y - start_y)
        top_has_analysis = self._side_has_visible_analysis(top_display_color)
        bottom_has_analysis = self._side_has_visible_analysis(bottom_display_color)

        candidates = []
        for compact_level in (0, 1, 2):
            top_card_h = self._player_card_height(top_info, compact_level=compact_level)
            bottom_card_h = self._player_card_height(bottom_info, compact_level=compact_level)
            for row_limit in range(ANALYSIS_DISPLAY_ROWS, 0, -1):
                inner_gap = max(4, self.tiny_font.get_height() // 3 + 1 - compact_level)
                top_group_h = top_card_h + ((inner_gap + self._analysis_card_height(top_display_color, row_limit=row_limit)) if top_has_analysis else 0)
                bottom_group_h = bottom_card_h + ((inner_gap + self._analysis_card_height(bottom_display_color, row_limit=row_limit)) if bottom_has_analysis else 0)
                total_height = top_group_h + bottom_group_h + inner_gap
                candidates.append(
                    {
                        "compact_level": compact_level,
                        "row_limit": row_limit,
                        "inner_gap": inner_gap,
                        "top_card_h": top_card_h,
                        "bottom_card_h": bottom_card_h,
                        "top_group_h": top_group_h,
                        "bottom_group_h": bottom_group_h,
                        "fits": total_height <= available_height,
                        "overflow": total_height - available_height,
                    }
                )

        layout = next((item for item in candidates if item["fits"]), None)
        if layout is None and candidates:
            layout = min(candidates, key=lambda item: item["overflow"])
        return {
            "panel_inner_left": panel_inner_left,
            "panel_inner_width": panel_inner_width,
            "start_y": start_y,
            "end_y": end_y,
            "available_height": available_height,
            "top_has_analysis": top_has_analysis,
            "bottom_has_analysis": bottom_has_analysis,
            **(layout or {
                "compact_level": 2,
                "row_limit": 1,
                "inner_gap": 4,
                "top_card_h": self._player_card_height(top_info, compact_level=2),
                "bottom_card_h": self._player_card_height(bottom_info, compact_level=2),
                "top_group_h": self._player_card_height(top_info, compact_level=2),
                "bottom_group_h": self._player_card_height(bottom_info, compact_level=2),
            }),
        }

    @staticmethod
    def _lerp_color(color_a, color_b, t):
        t = max(0.0, min(1.0, float(t)))
        return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))

    def _update_button_hover(self, action_key, hovered):
        current = float(self.button_hover_state.get(action_key, 0.0))
        step = 0.24
        if hovered:
            current = min(1.0, current + step)
        else:
            current = max(0.0, current - step)
        self.button_hover_state[action_key] = current
        return current

    def _update_mouse_cursor(self, use_hand):
        target = pygame.SYSTEM_CURSOR_HAND if use_hand else pygame.SYSTEM_CURSOR_ARROW
        if target == self.current_cursor_kind:
            return
        try:
            pygame.mouse.set_cursor(pygame.cursors.Cursor(target))
            self.current_cursor_kind = target
        except Exception:
            pass

    def _draw_action_button(self, rect, label, enabled=True, active=False, danger=False, hover_t=0.0):
        if not enabled:
            fill = (35, 45, 48)
            border = UI_BORDER
            text_color = UI_MUTED
        elif danger:
            fill = (93, 51, 51)
            border = (166, 103, 95)
            text_color = (246, 231, 222)
        elif active:
            fill = (47, 93, 83)
            border = UI_ACCENT
            text_color = UI_TEXT
        else:
            fill = (37, 57, 61)
            border = UI_BORDER
            text_color = UI_TEXT

        if enabled:
            hover_fill = tuple(min(255, c + 26) for c in fill)
            hover_border = tuple(min(255, c + 34) for c in border)
            fill = self._lerp_color(fill, hover_fill, hover_t)
            border = self._lerp_color(border, hover_border, hover_t)

        animated_rect = rect.move(0, -int(round(2 * max(0.0, hover_t))))
        shadow_rect = animated_rect.move(0, 2)
        pygame.draw.rect(self.canvas, (9, 14, 17), shadow_rect, border_radius=9)
        pygame.draw.rect(self.canvas, fill, animated_rect, border_radius=8)
        pygame.draw.rect(self.canvas, border, animated_rect, width=1, border_radius=8)
        label_text = self._fit_text(self.small_font, label, animated_rect.width - 14)
        text = self.small_font.render(label_text, True, text_color)
        self.canvas.blit(text, text.get_rect(center=animated_rect.center))

    def _scaled_piece_icon(self, symbol, target_size, fill_ratio=0.84):
        surface = self.pieces.get(symbol)
        if surface is None:
            return None
        bbox = surface.get_bounding_rect(min_alpha=1)
        if bbox.width <= 0 or bbox.height <= 0:
            return pygame.transform.smoothscale(surface, (target_size, target_size))
        cropped = surface.subsurface(bbox).copy()
        target_inner = max(1, int(target_size * fill_ratio))
        scale = min(target_inner / cropped.get_width(), target_inner / cropped.get_height())
        new_w = max(1, int(cropped.get_width() * scale))
        new_h = max(1, int(cropped.get_height() * scale))
        scaled = pygame.transform.smoothscale(cropped, (new_w, new_h))
        canvas = pygame.Surface((target_size, target_size), pygame.SRCALPHA)
        canvas.blit(scaled, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        return canvas

    def _draw_left_panel(self):
        panel = self.left_panel_rect
        self.analysis_entry_buttons = []

        title = self.small_font.render("PLAY / ARENA", True, UI_TEXT)
        self.canvas.blit(title, (panel.left + 16, panel.top + 14))

        mode_text, _ = get_game_mode_labels(self.game_mode, self.human_color)
        top_display_color = chess.WHITE if self.flipped else chess.BLACK
        bottom_display_color = chess.BLACK if top_display_color == chess.WHITE else chess.WHITE
        top_info = self._side_info(top_display_color)
        bottom_info = self._side_info(bottom_display_color)

        status_height = 18 + self.small_font.get_height() + 8 + self.tiny_font.get_height() + 16
        status_rect = pygame.Rect(panel.left + 14, panel.top + 48, panel.width - 28, status_height)
        self._draw_card(status_rect, fill=UI_SURFACE, border=UI_BORDER)
        if self.ai_status_error:
            status_line = "AI search failed; restart to retry"
            status_color = (255, 150, 132)
        elif self.ai_paused:
            status_line = "Paused"
            status_color = (255, 214, 120)
        elif self.ai_thinking:
            thinking_side = "White" if self.ai_thinking_color == chess.WHITE else "Black"
            thinking_text = f"{thinking_side} thinking{self._thinking_dots()}"
            elapsed = self._format_duration(self._current_thinking_elapsed())
            status_line = f"{thinking_text}  {elapsed}"
            status_color = (255, 214, 120)
        else:
            status_line = mode_text
            status_color = (168, 183, 206)
        self.canvas.blit(
            self.small_font.render(self._fit_text(self.small_font, status_line, status_rect.width - 24), True, status_color),
            (status_rect.left + 12, status_rect.top + 10),
        )
        turn_side = "White" if self.board.turn == chess.WHITE else "Black"
        turn_color = get_turn_color(self.game_mode, self.board.turn, self.human_color)
        self.canvas.blit(
            self.tiny_font.render(
                self._fit_text(
                    self.tiny_font,
                    self.ai_status_error or f"{turn_side} to move",
                    status_rect.width - 24,
                ),
                True,
                (255, 150, 132) if self.ai_status_error else turn_color,
            ),
            (status_rect.left + 12, status_rect.top + 40),
        )

        layout = self._resolve_left_panel_layout(
            panel,
            status_rect,
            top_info,
            bottom_info,
            top_display_color,
            bottom_display_color,
        )
        panel_inner_left = layout["panel_inner_left"]
        panel_inner_width = layout["panel_inner_width"]
        compact_level = layout["compact_level"]
        row_limit = layout["row_limit"]
        gap = layout["inner_gap"]

        top_card = pygame.Rect(
            panel_inner_left,
            layout["start_y"],
            panel_inner_width,
            layout["top_card_h"],
        )
        current_y = top_card.bottom
        top_analysis_rect = None
        if layout["top_has_analysis"]:
            current_y += gap
            top_analysis_rect = pygame.Rect(
                panel_inner_left,
                current_y,
                panel_inner_width,
                self._analysis_card_height(top_display_color, row_limit=row_limit),
            )
            current_y = top_analysis_rect.bottom

        bottom_card = pygame.Rect(
            panel_inner_left,
            layout["end_y"] - layout["bottom_card_h"],
            panel_inner_width,
            layout["bottom_card_h"],
        )
        bottom_analysis_rect = None
        if layout["bottom_has_analysis"]:
            bottom_analysis_rect = pygame.Rect(
                panel_inner_left,
                bottom_card.top - gap - self._analysis_card_height(bottom_display_color, row_limit=row_limit),
                panel_inner_width,
                self._analysis_card_height(bottom_display_color, row_limit=row_limit),
            )

        self._draw_player_card(
            top_card,
            str(top_info.get("side", "Top")),
            top_info,
            active=self.board.turn == top_display_color,
            compact_level=compact_level,
        )
        if top_analysis_rect is not None:
            self._draw_analysis_card(
                top_analysis_rect,
                top_display_color,
                active=self.board.turn == top_display_color,
                row_limit=row_limit,
            )
        if bottom_analysis_rect is not None:
            self._draw_analysis_card(
                bottom_analysis_rect,
                bottom_display_color,
                active=self.board.turn == bottom_display_color,
                row_limit=row_limit,
            )
        self._draw_player_card(
            bottom_card,
            str(bottom_info.get("side", "Bottom")),
            bottom_info,
            active=self.board.turn == bottom_display_color,
            compact_level=compact_level,
        )

    def _draw_right_panel(self):
        panel = self.right_panel_rect
        padding = 12
        title_height = max(34, self.text_font.get_height() + 10)
        action_rows = 4 if self.game_mode == "ai_vs_ai" or (self.game_mode == "human_vs_ai" and self.mcts_enabled) else 3
        compact = panel.height < 760
        btn_h = 34 if compact else 38
        btn_gap = 7 if compact else 9
        action_header_height = max(28, self.tiny_font.get_height() + 15)
        button_area_height = action_header_height + action_rows * btn_h + (action_rows - 1) * btn_gap + 14
        match_panel_h = self._match_panel_height() if self._match_enabled() else 0
        section_gap = 10
        content_top = panel.top + padding + title_height + 7
        content_bottom = panel.bottom - padding
        available_height = max(1, content_bottom - content_top)
        fixed_height = button_area_height + (match_panel_h + section_gap if match_panel_h else 0)
        history_height = max(108, available_height - fixed_height - section_gap)
        if history_height + fixed_height + section_gap > available_height:
            overflow = history_height + fixed_height + section_gap - available_height
            button_area_height = max(150, button_area_height - overflow)
            history_height = max(
                90,
                available_height - button_area_height - (match_panel_h + section_gap if match_panel_h else 0) - section_gap,
            )

        self._draw_card(panel, fill=UI_SURFACE, border=UI_BORDER)
        title = self.small_font.render("Move history", True, UI_TEXT)
        self.canvas.blit(title, (panel.left + padding + 2, panel.top + padding + 1))
        badge_text = "MODEL MATCH" if self.game_mode == "ai_vs_ai" else "YOUR GAME"
        badge = self.tiny_font.render(
            fit_ui_text(self.tiny_font, badge_text, panel.width - title.get_width() - 3 * padding),
            True,
            UI_MUTED,
        )
        self.canvas.blit(badge, badge.get_rect(right=panel.right - padding, centery=panel.top + padding + 1 + title.get_height() // 2))

        history_rect = pygame.Rect(panel.left + padding, content_top, panel.width - 2 * padding, history_height)
        self.history_rect = history_rect
        button_area = pygame.Rect(
            panel.left + padding,
            history_rect.bottom + section_gap,
            panel.width - 2 * padding,
            button_area_height,
        )
        match_area = (
            pygame.Rect(panel.left + padding, button_area.bottom + section_gap, panel.width - 2 * padding, match_panel_h)
            if match_panel_h
            else None
        )
        self.match_panel_rect = match_area

        self._draw_card(history_rect, fill=(20, 28, 33), border=UI_BORDER)
        history_header_h = max(24, self.tiny_font.get_height() + 10)
        row_h = max(21, min(26, self.tiny_font.get_height() + 7))
        max_rows = max(1, min(14, (history_rect.height - history_header_h - 8) // row_h))
        rows = []
        for i in range(0, len(self.move_san_history), 2):
            rows.append(
                (
                    (i // 2) + 1,
                    self.move_san_history[i],
                    self.move_san_history[i + 1] if i + 1 < len(self.move_san_history) else "",
                )
            )

        num_col_w = 30
        col_gap = 5
        row_left = history_rect.left + 10
        move_col_w = max(28, (history_rect.width - 20 - num_col_w - (col_gap * 2)) // 2)
        white_x = row_left + num_col_w + col_gap
        black_x = white_x + move_col_w + col_gap
        for label, x, width in (("#", row_left, num_col_w), ("WHITE", white_x, move_col_w), ("BLACK", black_x, move_col_w)):
            header = self.tiny_font.render(self._fit_text(self.tiny_font, label, width), True, UI_MUTED)
            self.canvas.blit(header, (x, history_rect.top + 7))

        max_scroll = max(0, len(rows) - max_rows)
        self.history_scroll_rows = max(0, min(self.history_scroll_rows, max_scroll))
        start_idx = max(0, len(rows) - max_rows - self.history_scroll_rows)
        end_idx = max(0, len(rows) - self.history_scroll_rows)
        visible_rows = rows[start_idx:end_idx]

        if not rows:
            empty = self.small_font.render("No moves yet", True, UI_MUTED)
            empty_y = history_rect.top + history_header_h + (history_rect.height - history_header_h) // 2
            self.canvas.blit(empty, empty.get_rect(center=(history_rect.centerx, empty_y)))

        y = history_rect.top + history_header_h
        self.history_entry_buttons = []
        for move_no, white_move, black_move in visible_rows:
            row_rect = pygame.Rect(history_rect.left + 6, y - 2, history_rect.width - 12, row_h)
            white_ply = (move_no - 1) * 2 + 1
            black_ply = min(len(self.move_history), white_ply + 1)
            if self.selected_history_ply in (white_ply, black_ply):
                pygame.draw.rect(self.canvas, (46, 80, 72), row_rect, border_radius=6)
            elif move_no % 2 == 1:
                pygame.draw.rect(self.canvas, (25, 35, 38), row_rect, border_radius=6)
            self.canvas.blit(
                self.tiny_font.render(f"{move_no:>2}", True, UI_MUTED),
                (row_left, y),
            )
            white_rect = pygame.Rect(white_x - 2, y - 2, move_col_w + 4, row_h)
            black_rect = pygame.Rect(black_x - 2, y - 2, move_col_w + 4, row_h)
            self.canvas.blit(
                self.tiny_font.render(self._fit_text(self.tiny_font, white_move, move_col_w), True, UI_TEXT),
                (white_x, y),
            )
            self.canvas.blit(
                self.tiny_font.render(self._fit_text(self.tiny_font, black_move, move_col_w), True, UI_TEXT),
                (black_x, y),
            )
            self.history_entry_buttons.append({"rect": white_rect, "ply": white_ply})
            if black_move:
                self.history_entry_buttons.append({"rect": black_rect, "ply": black_ply})
            y += row_h

        if max_scroll > 0:
            track = pygame.Rect(history_rect.right - 8, history_rect.top + history_header_h, 4, max(1, history_rect.height - history_header_h - 8))
            pygame.draw.rect(self.canvas, (42, 58, 60), track, border_radius=2)
            thumb_h = max(18, int(track.height * (max_rows / max(1, len(rows)))))
            travel = max(0, track.height - thumb_h)
            scroll_ratio = 0.0 if max_scroll <= 0 else (1.0 - (self.history_scroll_rows / max_scroll))
            thumb_y = track.top + int(scroll_ratio * travel)
            pygame.draw.rect(self.canvas, (105, 143, 135), pygame.Rect(track.left, thumb_y, track.width, thumb_h), border_radius=2)

        self._draw_card(button_area, fill=(20, 28, 33), border=UI_BORDER)
        action_title = self.tiny_font.render("ACTIONS", True, UI_MUTED)
        self.canvas.blit(action_title, (button_area.left + 10, button_area.top + 7))

        btn_gap = 7 if compact else 9
        btn_w = (button_area.width - 3 * btn_gap) // 2
        btn_h = min(
            34 if compact else 38,
            max(28, (button_area.height - action_header_height - 14 - (action_rows - 1) * btn_gap) // action_rows),
        )
        x1 = button_area.left + btn_gap
        x2 = x1 + btn_w + btn_gap
        y0 = max(button_area.top + 25, button_area.top + 7 + action_title.get_height() + 8)

        buttons = [
            ("menu", "Menu", True, False, True, pygame.Rect(x1, y0, btn_w, btn_h)),
            ("flip", "Flip board", True, self.flipped, False, pygame.Rect(x2, y0, btn_w, btn_h)),
            (
                "undo",
                "Undo",
                bool(self.move_history) and not self.ai_thinking and not self._match_enabled(),
                False,
                False,
                pygame.Rect(x1, y0 + btn_h + btn_gap, btn_w, btn_h),
            ),
            (
                "restart",
                "Restart",
                True,
                False,
                False,
                pygame.Rect(x2, y0 + btn_h + btn_gap, btn_w, btn_h),
            ),
            (
                "rewind",
                "Rewind to selected",
                self.selected_history_ply is not None and not self.ai_thinking and not self._match_enabled(),
                False,
                False,
                pygame.Rect(x1, y0 + (btn_h + btn_gap) * 2, btn_w * 2 + btn_gap, btn_h),
            ),
        ]
        next_row_y = y0 + (btn_h + btn_gap) * 3
        if self.game_mode == "ai_vs_ai":
            match_status = str(self._match_snapshot().get("status", "starting"))
            match_action_label = {
                "starting": "Starting match",
                "visible_running": "Visible game running",
                "stopping_previous": "Stopping previous match",
                "loading_models": "Starting match",
                "starting_workers": "Starting match",
                "running": "Match running",
                "waiting_remaining": "Waiting for games",
                "cancelling": "Stopping match",
                "cancelled": "Match cancelled",
                "completed": "Match complete",
                "error": "Match error",
                "incomplete": "Match ended",
            }.get(match_status, "Match running")
            buttons.append(
                (
                    "pause",
                    match_action_label if self._match_enabled() else ("Resume game" if self.ai_paused else "Pause game"),
                    not self._match_enabled(),
                    self.ai_paused,
                    False,
                    pygame.Rect(x1, next_row_y, btn_w * 2 + btn_gap, btn_h),
                )
            )
            next_row_y += btn_h + btn_gap
        if self.game_mode == "human_vs_ai" and self.mcts_enabled:
            buttons.append(
                (
                    "toggle_mcts",
                    f"MCTS: {'ON' if self.use_mcts else 'OFF'}",
                    not self.ai_thinking,
                    self.use_mcts,
                    False,
                    pygame.Rect(x1, next_row_y, btn_w * 2 + btn_gap, btn_h),
                )
            )

        self.hover_any_button = False
        self.action_buttons = []
        for action, label, enabled, active, danger, rect in buttons:
            hovered = bool(enabled and rect.collidepoint(self.mouse_canvas_pos))
            hover_t = self._update_button_hover(action, hovered)
            if hovered:
                self.hover_any_button = True
            self._draw_action_button(
                rect,
                label,
                enabled=enabled,
                active=active,
                danger=danger,
                hover_t=hover_t,
            )
            self.action_buttons.append({"action": action, "rect": rect, "enabled": enabled})

        if match_area is not None:
            self._draw_match_panel(match_area)

    def _match_model_label(self, model_name):
        label = Path(str(model_name or "model")).stem
        return label or "model"

    def _draw_wdl_segment_bar(self, rect, summary):
        """Draw a width-exact Model 1 / draw / Model 2 result bar."""
        if rect.width <= 0 or rect.height <= 0:
            return
        radius = max(3, rect.height // 2)
        pygame.draw.rect(self.canvas, (39, 54, 57), rect, border_radius=radius)
        widths = _match_wdl_segment_widths(summary, rect.width)
        colors = ((232, 229, 211), (225, 178, 104), (43, 51, 55))
        x = rect.left
        for segment_width, color in zip(widths, colors):
            if segment_width > 0:
                pygame.draw.rect(
                    self.canvas,
                    color,
                    pygame.Rect(x, rect.top, segment_width, rect.height),
                )
            x += segment_width
        pygame.draw.rect(self.canvas, UI_BORDER_STRONG, rect, width=1, border_radius=radius)

    def _draw_match_wdl_summary(self, rect, snapshot, heading="MODEL RESULTS"):
        """Draw live model-identity W/D/L and keep win% distinct from points."""
        self._draw_card(rect, fill=UI_SURFACE, border=UI_BORDER)
        previous_clip = self.canvas.get_clip()
        self.canvas.set_clip(rect.clip(self.canvas.get_rect()))
        summary = _match_wdl_summary(snapshot)
        scale = self.ui_scale
        pad = max(8, round(11 * scale))
        inner_left = rect.left + pad
        inner_width = max(1, rect.width - 2 * pad)
        title_text = (
            f"{heading} · {summary['resolved']} resolved"
            if summary["resolved"] > 0
            else f"{heading} · no resolved games"
        )
        title = self.tiny_font.render(
            fit_ui_text(self.tiny_font, title_text, inner_width), True, UI_MUTED
        )
        self.canvas.blit(title, (inner_left, rect.top + pad))

        col_gap = max(3, round(5 * scale))
        col_width = max(1, (inner_width - 2 * col_gap) // 3)
        col_xs = (
            inner_left,
            inner_left + col_width + col_gap,
            inner_left + 2 * (col_width + col_gap),
        )
        labels = (
            f"MODEL 1 · {self._compact_model_label(self.model1_name)}",
            "DRAWS",
            f"MODEL 2 · {self._compact_model_label(self.model2_name)}",
        )
        label_y = rect.top + pad + title.get_height() + max(3, round(4 * scale))
        for index, (x, label) in enumerate(zip(col_xs, labels)):
            color = UI_TEXT if index == 0 else (225, 178, 104) if index == 1 else (181, 193, 191)
            surface = self.tiny_font.render(
                fit_middle_text(self.tiny_font, label, col_width), True, color
            )
            self.canvas.blit(surface, surface.get_rect(center=(x + col_width // 2, label_y + surface.get_height() // 2)))

        bar_height = max(10, round(13 * scale))
        bar_y = label_y + self.tiny_font.get_height() + max(4, round(6 * scale))
        bar_rect = pygame.Rect(inner_left, bar_y, inner_width, bar_height)
        self._draw_wdl_segment_bar(bar_rect, summary)

        resolved = summary["resolved"]
        model1_win = "—" if resolved <= 0 else f"{summary['model1_win_pct']:.0f}%"
        model2_win = "—" if resolved <= 0 else f"{summary['model2_win_pct']:.0f}%"
        draw_rate = "—" if resolved <= 0 else f"{summary['draw_pct']:.0f}%"
        model1_points = "—" if resolved <= 0 else f"{summary['model1_points_pct']:.1f}%"
        model2_points = "—" if resolved <= 0 else f"{summary['model2_points_pct']:.1f}%"
        metrics = (
            (f"{summary['model1_wins']}W · {model1_win} win", f"PTS {model1_points}"),
            (f"{summary['draws']}D · {draw_rate}", "½ point each" if resolved else "No results yet"),
            (f"{summary['model2_wins']}W · {model2_win} win", f"PTS {model2_points}"),
        )
        line_gap = max(2, round(3 * scale))
        line1_y = bar_rect.bottom + max(4, round(5 * scale))
        for x, (primary, secondary) in zip(col_xs, metrics):
            primary_surface = self.tiny_font.render(
                fit_ui_text(self.tiny_font, primary, col_width), True, UI_TEXT
            )
            secondary_surface = self.tiny_font.render(
                fit_ui_text(self.tiny_font, secondary, col_width), True, UI_MUTED
            )
            self.canvas.blit(primary_surface, primary_surface.get_rect(center=(x + col_width // 2, line1_y + primary_surface.get_height() // 2)))
            line2_y = line1_y + primary_surface.get_height() + line_gap
            self.canvas.blit(secondary_surface, secondary_surface.get_rect(center=(x + col_width // 2, line2_y + secondary_surface.get_height() // 2)))
        self.canvas.set_clip(previous_clip)

    def _compact_model_label(self, model_name):
        """Keep model, architecture, and epoch identity visible in narrow cards."""
        name = self._match_model_label(model_name)
        labels = []
        version = re.search(r"v\d+(?:\.\d+)?", name, re.IGNORECASE)
        if version:
            labels.append(version.group(0).lower())
        rl_run = re.search(r"rl[-_ ]?\d+", name, re.IGNORECASE)
        if rl_run:
            labels.append(rl_run.group(0).replace("_", "").replace(" ", "").upper())
        elif re.search(r"(?:^|[_\-.])il(?:$|[_\-.])", name, re.IGNORECASE):
            labels.append("IL")
        epoch = re.search(r"(?:epoch|iter|ep)[-_ ]?(\d+)", name, re.IGNORECASE)
        if epoch:
            labels.append(f"E{epoch.group(1)}")
        return " ".join(labels) if labels else name

    def _match_panel_height(self):
        tiny_height = self.tiny_font.get_height()
        row_height = max(36, self.small_font.get_height() + tiny_height + 12)
        row_top_offset = 37 + 3 * tiny_height
        required = row_top_offset + 2 * row_height + 10
        return max(190, required, 88 + tiny_height * 7)

    def _draw_match_panel(self, rect):
        snapshot = self._match_snapshot()
        completed = int(snapshot.get("completed", 0) or 0)
        unresolved = int(snapshot.get("unresolved", 0) or 0)
        invalid = int(snapshot.get("invalid_results", 0) or 0)
        attempted = max(
            completed + unresolved + invalid,
            int(snapshot.get("attempted", 0) or 0),
        )
        total = max(1, int(snapshot.get("total", self.match_total_games) or self.match_total_games))
        model1_wins = int(snapshot.get("model1_wins", 0) or 0)
        model2_wins = int(snapshot.get("model2_wins", 0) or 0)
        draws = int(snapshot.get("draws", 0) or 0)
        model1_score_x2 = int(snapshot.get("model1_score_x2", 0) or 0)
        model2_score_x2 = int(snapshot.get("model2_score_x2", 0) or 0)
        status = str(snapshot.get("status", "starting"))
        runtime_status = str(snapshot.get("runtime_status", status))
        stage_elapsed = float(snapshot.get("runtime_stage_elapsed_s", 0.0) or 0.0)
        active_games = int(snapshot.get("active_games", 0) or 0)
        worker_count = int(snapshot.get("background_workers", 0) or 0)
        central_inference = bool(snapshot.get("central_inference", False))
        visible_game = dict(snapshot.get("visible_game") or {})
        visible_finished = bool(snapshot.get("visible_game_finished", self.visible_match_finished))
        background_total = max(0, int(snapshot.get("background_total", total - 1) or 0))
        background_attempted = max(0, int(snapshot.get("background_attempted", 0) or 0))
        background_completed = max(0, int(snapshot.get("background_completed", 0) or 0))
        background_done = bool(snapshot.get("background_done", total <= 1))
        background_starting = bool(snapshot.get("background_starting", False))
        error = snapshot.get("error")
        scale = self.ui_scale
        padding = max(10, round(12 * scale))

        previous_clip = self.canvas.get_clip()
        self.canvas.set_clip(rect.clip(self.canvas.get_rect()))
        self._draw_card(rect, fill=UI_SURFACE_RAISED, border=UI_BORDER)
        title_y = rect.top + padding
        title = self.tiny_font.render("AI MATCH", True, UI_MUTED)
        self.canvas.blit(title, (rect.left + padding, title_y))
        game_count = self.tiny_font.render(
            fit_ui_text(self.tiny_font, "1 GAME" if total == 1 else f"{total} GAMES", rect.width - title.get_width() - 3 * padding),
            True,
            UI_ACCENT,
        )
        self.canvas.blit(game_count, game_count.get_rect(right=rect.right - padding, centery=title_y + title.get_height() // 2))

        if self._visible_match_active():
            side = "White" if self.board.turn == chess.WHITE else "Black"
            line1 = f"Game 1 · {side} to move"
            if total == 1:
                line2 = "Waiting for first result"
            elif background_done:
                line2 = "Background batch complete · game 1 still live"
            elif background_attempted or active_games:
                line2 = f"Background {background_completed}/{background_total} resolved · {active_games} active"
            elif background_starting:
                line2 = "Remaining games starting in background"
            else:
                line2 = "Remaining games running in background"
        elif visible_finished and status not in {"completed", "cancelled", "error", "incomplete"}:
            game_result = str(visible_game.get("result", self.visible_match_result or "*"))
            winner = visible_game.get("winner_model") or self.visible_match_winner
            first_result = "First game · draw" if game_result == "1/2-1/2" else (
                f"First game · {self._match_model_label(winner)} won" if winner else "First game finished"
            )
            line1 = first_result
            if background_done:
                line2 = f"Background finished · {completed}/{total} resolved"
            elif background_attempted or active_games:
                line2 = f"Background {background_completed}/{background_total} resolved · {active_games} active"
            else:
                line2 = "Background batch is finishing startup"
        elif error or status == "error":
            line1 = "Match stopped with an error"
            line2 = str(error or "See match details")
        elif status == "completed":
            line1 = f"Match complete · {completed}/{total} games"
            line2 = f"{draws} draws · results saved"
        elif status == "cancelled":
            line1 = f"Match cancelled · {completed}/{total} resolved"
            line2 = "Unfinished games are excluded"
        elif status == "incomplete":
            line1 = f"Match ended early · {completed}/{total} resolved"
            line2 = "Unfinished games are excluded"
        elif runtime_status in {"loading_models", "snapshotting_models", "starting_server", "models_loaded"} or status in {"starting", "loading_models", "starting_workers", "stopping_previous"}:
            line1 = "Preparing models and starting the match"
            line2 = f"{stage_elapsed:.1f}s · {'GPU evaluator' if central_inference else 'local evaluator'}"
        elif status == "cancelling":
            line1 = f"Stopping match · {completed}/{total} resolved"
            line2 = "Finishing active game workers"
        else:
            line1 = f"Playing · {completed}/{total} games resolved"
            line2 = f"{active_games} games active · {worker_count} workers · {stage_elapsed:.1f}s"
        if status == "running" and completed == 0 and stage_elapsed > 0 and not visible_finished:
            line2 = f"{active_games} games active · {stage_elapsed:.1f}s"

        text_width = rect.width - 2 * padding
        line1_surf = self.tiny_font.render(fit_ui_text(self.tiny_font, line1, text_width), True, UI_TEXT)
        line2_surf = self.tiny_font.render(fit_ui_text(self.tiny_font, line2, text_width), True, UI_MUTED)
        line_height = max(line1_surf.get_height(), line2_surf.get_height())
        line_gap = max(4, round(7 * scale))
        line1_y = title_y + title.get_height() + line_gap
        line2_y = line1_y + line_height + max(2, round(4 * scale))
        self.canvas.blit(line1_surf, (rect.left + padding, line1_y))
        self.canvas.blit(line2_surf, (rect.left + padding, line2_y))

        bar_height = max(4, round(6 * scale))
        bar_rect = pygame.Rect(rect.left + padding, line2_y + line_height + line_gap, rect.width - 2 * padding, bar_height)
        pygame.draw.rect(self.canvas, (39, 54, 57), bar_rect, border_radius=bar_height // 2)
        fill_w = int(round(bar_rect.width * min(1.0, attempted / float(total))))
        if fill_w > 0:
            pygame.draw.rect(self.canvas, UI_ACCENT, pygame.Rect(bar_rect.left, bar_rect.top, fill_w, bar_rect.height), border_radius=bar_height // 2)

        denominator = max(1, completed)
        model1_score = model1_score_x2 / (2.0 * denominator) * 100.0
        model2_score = model2_score_x2 / (2.0 * denominator) * 100.0
        row_gap = max(6, round(8 * scale))
        row_top = bar_rect.bottom + row_gap
        row_height = max(
            round(48 * scale),
            self.small_font.get_height() + self.tiny_font.get_height() + round(20 * scale),
        )
        for index, (name, wins, losses, points, accent) in enumerate((
            (self._compact_model_label(self.model1_name), model1_wins, model2_wins, model1_score, UI_ACCENT),
            (self._compact_model_label(self.model2_name), model2_wins, model1_wins, model2_score, (225, 178, 104)),
        )):
            top = row_top + index * row_height
            row_rect = pygame.Rect(rect.left + padding, top, rect.width - 2 * padding, row_height - max(2, round(3 * scale)))
            radius = max(6, round(8 * scale))
            pygame.draw.rect(self.canvas, (23, 33, 37), row_rect, border_radius=radius)
            pygame.draw.rect(self.canvas, (44, 62, 65), row_rect, width=max(1, round(scale)), border_radius=radius)
            win_rate = wins / denominator * 100.0 if completed else None
            win_label = f"{win_rate:.0f}% wins" if win_rate is not None else "—"
            win_surf = self.small_font.render(
                fit_ui_text(self.small_font, win_label, row_rect.width - 2 * padding),
                True,
                accent,
            )
            name_x = row_rect.left + padding
            name_right = row_rect.right - padding - win_surf.get_width() - max(8, round(12 * scale))
            name_width = max(16, name_right - name_x)
            name_surf = self.tiny_font.render(
                fit_middle_text(self.tiny_font, name, name_width),
                True,
                UI_TEXT,
            )
            top_y = row_rect.top + max(2, round(4 * scale))
            self.canvas.blit(name_surf, (name_x, top_y))
            self.canvas.blit(win_surf, win_surf.get_rect(right=row_rect.right - padding, top=top_y))
            points = points if completed else None
            detail_text = (
                f"W {wins} · D {draws} · L {losses}  |  {points:.1f}% pts"
                if points is not None
                else "Waiting for first result"
            )
            details = self.tiny_font.render(
                fit_ui_text(self.tiny_font, detail_text, row_rect.width - 2 * padding),
                True,
                UI_MUTED,
            )
            self.canvas.blit(
                details,
                (name_x, row_rect.bottom - details.get_height() - max(2, round(4 * scale))),
            )

        self.canvas.set_clip(previous_clip)
    def draw_hud(self):
        draw_workspace(self)

    def _show_match_result_modal(self):
        if not self._match_enabled():
            return False
        if self.match_session is not None:
            return self.match_session.is_done
        return False

    def _draw_match_result_modal(self):
        snapshot = self._match_snapshot()
        completed = int(snapshot.get("completed", 0) or 0)
        total = int(snapshot.get("total", self.match_total_games) or self.match_total_games)
        unresolved = int(snapshot.get("unresolved", 0) or 0)
        invalid = int(snapshot.get("invalid_results", 0) or 0)
        score1_x2 = int(snapshot.get("model1_score_x2", 0) or 0)
        score2_x2 = int(snapshot.get("model2_score_x2", 0) or 0)
        score1 = 50.0 if completed <= 0 else 100.0 * score1_x2 / (2.0 * completed)
        score2 = 50.0 if completed <= 0 else 100.0 * score2_x2 / (2.0 * completed)
        status = str(snapshot.get("status", "incomplete"))
        partial = unresolved > 0 or invalid > 0
        model1_name = self._match_model_label(self.model1_name)
        model2_name = self._match_model_label(self.model2_name)

        if status == "completed":
            title_text = "MATCH RESULT · PARTIAL" if partial else "MATCH COMPLETE"
            if completed <= 0:
                hero_label, hero_name = "NO RESULT", "No resolved games"
            elif abs(score1 - score2) < 1e-9:
                hero_label, hero_name = ("NO POINTS LEADER", "") if partial else ("MATCH TIED", "")
            else:
                leader = model1_name if score1 > score2 else model2_name
                hero_label = "POINTS LEADER" if partial else "MATCH WINNER"
                hero_name = leader
        elif status == "cancelled":
            title_text, hero_label, hero_name = "MATCH CANCELLED", "MATCH STOPPED", ""
        elif status == "error":
            title_text, hero_label, hero_name = "MATCH ERROR", "MATCH COULD NOT FINISH", ""
        else:
            title_text, hero_label, hero_name = "MATCH ENDED EARLY", "PARTIAL RESULT", ""

        overlay = pygame.Surface((self.base_width, self.base_height), pygame.SRCALPHA)
        overlay.fill((7, 12, 15, 178))
        self.canvas.blit(overlay, (0, 0))
        scale = self.ui_scale
        pad = max(16, round(24 * scale))
        modal = centered_rect(
            pygame,
            self.canvas.get_rect(),
            round(680 * scale),
            round(480 * scale),
            inset=round(18 * scale),
        )
        shadow_rect = modal.inflate(round(14 * scale), round(14 * scale)).clip(self.canvas.get_rect())
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 80), shadow_surface.get_rect(), border_radius=round(22 * scale))
        self.canvas.blit(shadow_surface, shadow_rect.topleft)
        self._draw_card(modal, fill=UI_SURFACE_RAISED, border=UI_ACCENT)
        pygame.draw.rect(
            self.canvas,
            UI_ACCENT,
            pygame.Rect(modal.left + pad, modal.top, modal.width - 2 * pad, max(2, round(3 * scale))),
            border_radius=max(1, round(2 * scale)),
        )

        title = self.small_font.render(fit_ui_text(self.small_font, title_text, modal.width - 2 * pad), True, UI_MUTED)
        self.canvas.blit(title, title.get_rect(center=(modal.centerx, modal.top + round(28 * scale))))
        hero = self.text_font.render(fit_ui_text(self.text_font, hero_label, modal.width - 2 * pad), True, UI_TEXT)
        self.canvas.blit(hero, hero.get_rect(center=(modal.centerx, modal.top + round(66 * scale))))
        if hero_name:
            winner_name = self.small_font.render(fit_middle_text(self.small_font, hero_name, modal.width - 2 * pad), True, UI_ACCENT)
            self.canvas.blit(winner_name, winner_name.get_rect(center=(modal.centerx, modal.top + round(104 * scale))))

        stats_rect = pygame.Rect(
            modal.left + pad,
            modal.top + round(132 * scale),
            modal.width - 2 * pad,
            round(166 * scale),
        )
        self._draw_match_wdl_summary(stats_rect, snapshot, heading="W/D/L · WIN RATE · POINTS")

        attempted = int(snapshot.get("attempted", completed + unresolved + invalid) or 0)
        detail_y = stats_rect.bottom + max(6, round(11 * scale))
        detail_text = f"Processed {attempted}/{total}  ·  resolved {completed}  ·  unresolved {unresolved}  ·  invalid {invalid}"
        detail = self.tiny_font.render(fit_ui_text(self.tiny_font, detail_text, modal.width - 2 * pad), True, UI_MUTED)
        self.canvas.blit(detail, detail.get_rect(center=(modal.centerx, detail_y + detail.get_height() // 2)))

        extra = str(snapshot.get("error") or snapshot.get("cancel_reason") or "")
        if not extra and snapshot.get("persistence_error"):
            extra = "Could not save the match results"
        if not extra and snapshot.get("result_path"):
            try:
                extra = f"Saved · {Path(snapshot['result_path']).relative_to(script_dir.parent)}"
            except ValueError:
                extra = f"Saved · {snapshot['result_path']}"
        if extra:
            extra_surf = self.tiny_font.render(fit_ui_text(self.tiny_font, extra, modal.width - 2 * pad), True, UI_MUTED)
            self.canvas.blit(
                extra_surf,
                extra_surf.get_rect(
                    center=(modal.centerx, detail_y + self.tiny_font.get_height() + max(2, round(4 * scale)))
                ),
            )

        btn_gap = max(6, round(12 * scale))
        btn_h = max(30, round(40 * scale))
        btn_w = (modal.width - 2 * pad - btn_gap) // 2
        btn_y = modal.bottom - btn_h - max(10, round(16 * scale))
        restart_rect = pygame.Rect(modal.left + pad, btn_y, btn_w, btn_h)
        menu_rect = pygame.Rect(restart_rect.right + btn_gap, btn_y, btn_w, btn_h)
        for action, label, danger, button_rect in (
            ("modal_restart", "Run again", False, restart_rect),
            ("modal_menu", "Menu", True, menu_rect),
        ):
            hovered = button_rect.collidepoint(self.mouse_canvas_pos)
            hover_t = self._update_button_hover(action, hovered)
            if hovered:
                self.hover_any_button = True
            self._draw_action_button(button_rect, label, enabled=True, danger=danger, hover_t=hover_t)
            self.modal_buttons.append({"action": action, "rect": button_rect, "enabled": True})
    def _draw_match_preview_status(self):
        if not self._match_enabled() or self._show_match_result_modal():
            return
        # The board is the actual first game until it ends. While the remaining
        # evaluator runs, the waiting result is shown by the endgame modal.
        return

    def _draw_visible_match_waiting_modal(self):
        snapshot = self._match_snapshot()
        visible_game = dict(snapshot.get("visible_game") or {})
        total = max(1, int(snapshot.get("total", self.match_total_games) or 1))
        attempted = max(0, int(snapshot.get("attempted", 1) or 0))
        background_total = max(0, int(snapshot.get("background_total", total - 1) or 0))
        background_completed = max(0, int(snapshot.get("background_completed", 0) or 0))
        background_done = bool(snapshot.get("background_done", total <= 1))
        active_games = max(0, int(snapshot.get("active_games", 0) or 0))
        result = str(visible_game.get("result", self.visible_match_result or "*"))
        winner_model = visible_game.get("winner_model") or self.visible_match_winner
        if result == "1/2-1/2":
            hero_text, winner_name = "FIRST GAME · DRAW", ""
        elif winner_model:
            hero_text, winner_name = "FIRST GAME WINNER", self._match_model_label(winner_model)
        else:
            hero_text, winner_name = "FIRST GAME RESULT", ""

        self.hover_any_button = False
        overlay = pygame.Surface((self.base_width, self.base_height), pygame.SRCALPHA)
        overlay.fill((7, 12, 15, 178))
        self.canvas.blit(overlay, (0, 0))
        scale = self.ui_scale
        pad = max(16, round(24 * scale))
        modal = centered_rect(
            pygame,
            self.canvas.get_rect(),
            round(700 * scale),
            round(490 * scale),
            inset=round(18 * scale),
        )
        self._draw_card(modal, fill=UI_SURFACE_RAISED, border=UI_ACCENT)
        pygame.draw.rect(
            self.canvas,
            UI_ACCENT,
            pygame.Rect(modal.left + pad, modal.top, modal.width - 2 * pad, max(2, round(3 * scale))),
            border_radius=max(1, round(2 * scale)),
        )

        title = self.small_font.render("FIRST GAME RESULT", True, UI_MUTED)
        self.canvas.blit(title, title.get_rect(center=(modal.centerx, modal.top + round(28 * scale))))
        hero = self.text_font.render(fit_ui_text(self.text_font, hero_text, modal.width - 2 * pad), True, UI_TEXT)
        self.canvas.blit(hero, hero.get_rect(center=(modal.centerx, modal.top + round(70 * scale))))
        if winner_name:
            winner = self.small_font.render(fit_middle_text(self.small_font, winner_name, modal.width - 2 * pad), True, UI_ACCENT)
            self.canvas.blit(winner, winner.get_rect(center=(modal.centerx, modal.top + round(112 * scale))))

        score_rect = pygame.Rect(
            modal.left + pad,
            modal.top + round(145 * scale),
            modal.width - 2 * pad,
            round(150 * scale),
        )
        self._draw_match_wdl_summary(score_rect, snapshot, heading="LIVE MODEL SCORE")

        waiting_box = pygame.Rect(
            modal.left + pad,
            score_rect.bottom + max(8, round(10 * scale)),
            modal.width - 2 * pad,
            round(92 * scale),
        )
        waiting_radius = max(8, round(13 * scale))
        pygame.draw.rect(self.canvas, UI_SURFACE, waiting_box, border_radius=waiting_radius)
        pygame.draw.rect(self.canvas, UI_BORDER, waiting_box, width=max(1, round(scale)), border_radius=waiting_radius)
        if background_total <= 0:
            status_line = "The visible game is the whole match"
            background_line = "No background games"
        else:
            status_line = "Background batch complete" if background_done else "Remaining games are running in background"
            background_line = f"{background_completed}/{background_total} background resolved · {active_games} active"
        overall_line = f"Overall {min(total, max(0, attempted))}/{total} games attempted"
        line_y = waiting_box.top + max(6, round(8 * scale))
        for index, line in enumerate((status_line, background_line, overall_line)):
            rendered = self.tiny_font.render(
                fit_ui_text(self.tiny_font, line, waiting_box.width - 2 * pad),
                True,
                UI_TEXT if index == 0 else UI_MUTED,
            )
            self.canvas.blit(rendered, rendered.get_rect(center=(waiting_box.centerx, line_y + rendered.get_height() // 2)))
            line_y += rendered.get_height() + max(1, round(2 * scale))
        progress_height = max(4, round(6 * scale))
        progress_rect = pygame.Rect(waiting_box.left + pad, waiting_box.bottom - max(13, round(16 * scale)), waiting_box.width - 2 * pad, progress_height)
        pygame.draw.rect(self.canvas, (39, 54, 57), progress_rect, border_radius=progress_height // 2)
        fill_ratio = min(1.0, max(0.0, attempted / float(total)))
        fill_w = int(round(progress_rect.width * fill_ratio))
        if fill_w:
            pygame.draw.rect(self.canvas, UI_ACCENT, pygame.Rect(progress_rect.left, progress_rect.top, fill_w, progress_rect.height), border_radius=progress_height // 2)

        btn_gap = max(6, round(12 * scale))
        btn_h = max(30, round(40 * scale))
        btn_w = (modal.width - 2 * pad - btn_gap) // 2
        btn_y = modal.bottom - btn_h - max(10, round(16 * scale))
        restart_rect = pygame.Rect(modal.left + pad, btn_y, btn_w, btn_h)
        menu_rect = pygame.Rect(restart_rect.right + btn_gap, btn_y, btn_w, btn_h)
        for action, label, danger, button_rect in (
            ("modal_restart", "Run again", False, restart_rect),
            ("modal_menu", "Menu", True, menu_rect),
        ):
            hovered = button_rect.collidepoint(self.mouse_canvas_pos)
            hover_t = self._update_button_hover(action, hovered)
            if hovered:
                self.hover_any_button = True
            self._draw_action_button(button_rect, label, enabled=True, danger=danger, hover_t=hover_t)
            self.modal_buttons.append({"action": action, "rect": button_rect, "enabled": True})
    def _draw_endgame_modal(self):
        self.modal_buttons = []
        if self._history_preview_active():
            return
        if self._match_enabled() and not self._show_match_result_modal():
            if self.game_over and self.visible_match_finished:
                self._draw_visible_match_waiting_modal()
            return
        if self._show_match_result_modal():
            self._draw_match_result_modal()
            return
        if not self.game_over:
            return

        self.hover_any_button = False
        overlay = pygame.Surface((self.base_width, self.base_height), pygame.SRCALPHA)
        overlay.fill((7, 12, 15, 178))
        self.canvas.blit(overlay, (0, 0))
        scale = self.ui_scale
        pad = max(16, round(24 * scale))
        modal = centered_rect(
            pygame,
            self.canvas.get_rect(),
            round(620 * scale),
            round(460 * scale),
            inset=round(18 * scale),
        )
        shadow_rect = modal.inflate(round(14 * scale), round(14 * scale)).clip(self.canvas.get_rect())
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 82), shadow_surface.get_rect(), border_radius=round(22 * scale))
        self.canvas.blit(shadow_surface, shadow_rect.topleft)
        self._draw_card(modal, fill=UI_SURFACE_RAISED, border=UI_BORDER_STRONG)
        pygame.draw.rect(
            self.canvas,
            UI_ACCENT,
            pygame.Rect(modal.left + pad, modal.top, modal.width - 2 * pad, max(2, round(3 * scale))),
            border_radius=max(1, round(2 * scale)),
        )

        result = self._board_result_for_match()
        winner_color = chess.WHITE if result == "1-0" else chess.BLACK if result == "0-1" else None
        if winner_color == chess.WHITE:
            hero_text, hero_color, accent_color, piece_symbol = "White wins", UI_TEXT, UI_ACCENT, "K"
        elif winner_color == chess.BLACK:
            hero_text, hero_color, accent_color, piece_symbol = "Black wins", UI_TEXT, (225, 178, 104), "k"
        else:
            hero_text, hero_color, accent_color, piece_symbol = "Draw", UI_TEXT, UI_ACCENT, None

        title = self.small_font.render("GAME OVER", True, UI_MUTED)
        self.canvas.blit(title, title.get_rect(center=(modal.centerx, modal.top + round(30 * scale))))
        icon_center_y = modal.top + round(110 * scale)
        if piece_symbol is not None:
            icon_size = min(round(100 * scale), max(round(64 * scale), modal.width // 7))
            icon_rect = pygame.Rect(0, 0, icon_size + round(24 * scale), icon_size + round(24 * scale))
            icon_rect.center = (modal.centerx, icon_center_y)
            icon_radius = max(8, round(20 * scale))
            pygame.draw.rect(self.canvas, UI_SURFACE, icon_rect, border_radius=icon_radius)
            pygame.draw.rect(self.canvas, accent_color, icon_rect, width=max(1, round(2 * scale)), border_radius=icon_radius)
            piece_surface = self._scaled_piece_icon(piece_symbol, icon_size)
            if piece_surface is not None:
                self.canvas.blit(piece_surface, piece_surface.get_rect(center=icon_rect.center))
            hero_y = icon_rect.bottom + round(16 * scale)
        else:
            icon_size = min(round(82 * scale), max(round(56 * scale), modal.width // 8))
            icon_rect = pygame.Rect(0, 0, icon_size * 2 + round(36 * scale), icon_size + round(24 * scale))
            icon_rect.center = (modal.centerx, icon_center_y)
            icon_radius = max(8, round(20 * scale))
            pygame.draw.rect(self.canvas, UI_SURFACE, icon_rect, border_radius=icon_radius)
            pygame.draw.rect(self.canvas, accent_color, icon_rect, width=max(1, round(2 * scale)), border_radius=icon_radius)
            for symbol, center_x in (("K", icon_rect.centerx - icon_size // 2), ("k", icon_rect.centerx + icon_size // 2)):
                piece_surface = self._scaled_piece_icon(symbol, icon_size, fill_ratio=0.82)
                if piece_surface is not None:
                    self.canvas.blit(piece_surface, piece_surface.get_rect(center=(center_x, icon_rect.centery)))
            hero_y = icon_rect.bottom + round(16 * scale)

        hero = self.text_font.render(fit_ui_text(self.text_font, hero_text, modal.width - 2 * pad), True, hero_color)
        self.canvas.blit(hero, hero.get_rect(center=(modal.centerx, hero_y + hero.get_height() // 2)))
        subtitle_text = "Checkmate" if self.board.is_checkmate() else get_result_message(result)[0].rstrip("!")
        subtitle = self.small_font.render(fit_ui_text(self.small_font, subtitle_text, modal.width - 2 * pad), True, accent_color)
        subtitle_y = hero_y + hero.get_height() + max(2, round(4 * scale))
        self.canvas.blit(subtitle, subtitle.get_rect(center=(modal.centerx, subtitle_y + subtitle.get_height() // 2)))

        stats_rect = pygame.Rect(modal.left + pad, modal.top + round(290 * scale), modal.width - 2 * pad, round(66 * scale))
        stats_radius = max(8, round(12 * scale))
        pygame.draw.rect(self.canvas, UI_SURFACE, stats_rect, border_radius=stats_radius)
        pygame.draw.rect(self.canvas, UI_BORDER, stats_rect, width=max(1, round(scale)), border_radius=stats_radius)
        if self._match_enabled():
            match = self._match_snapshot()
            completed = int(match.get("completed", 0) or 0)
            total = int(match.get("total", self.match_total_games) or self.match_total_games)
            draws = int(match.get("draws", 0) or 0)
            summary_lines = (
                f"Match game 1 of {total}  ·  {completed}/{total} resolved",
                f"Draws {draws}  ·  {self._match_model_label(self.model1_name)} vs {self._match_model_label(self.model2_name)}",
            )
        else:
            white_name = self._side_info(chess.WHITE)["name"]
            black_name = self._side_info(chess.BLACK)["name"]
            summary_lines = (
                f"{len(self.move_history)} half-moves",
                f"White {white_name}  ·  Black {black_name}",
            )
        for index, line in enumerate(summary_lines):
            rendered = self.tiny_font.render(fit_ui_text(self.tiny_font, line, stats_rect.width - 2 * pad), True, UI_TEXT if index == 0 else UI_MUTED)
            center_y = stats_rect.top + (stats_rect.height * (0.31 if index == 0 else 0.72))
            self.canvas.blit(rendered, rendered.get_rect(center=(stats_rect.centerx, round(center_y))))

        btn_gap = max(6, round(12 * scale))
        btn_h = max(30, round(40 * scale))
        btn_w = (modal.width - 2 * pad - btn_gap) // 2
        btn_y = modal.bottom - btn_h - max(10, round(16 * scale))
        restart_rect = pygame.Rect(modal.left + pad, btn_y, btn_w, btn_h)
        menu_rect = pygame.Rect(restart_rect.right + btn_gap, btn_y, btn_w, btn_h)
        for action, label, danger, button_rect in (
            ("modal_restart", "Play again", False, restart_rect),
            ("modal_menu", "Menu", True, menu_rect),
        ):
            hovered = button_rect.collidepoint(self.mouse_canvas_pos)
            hover_t = self._update_button_hover(action, hovered)
            if hovered:
                self.hover_any_button = True
            self._draw_action_button(button_rect, label, enabled=True, danger=danger, hover_t=hover_t)
            self.modal_buttons.append({"action": action, "rect": button_rect, "enabled": True})
    def _handle_ui_click(self, pos):
        # History remains inspectable behind an end-of-game/result overlay.
        # Selecting a ply hides that modal through _draw_endgame_modal().
        for entry in self.history_entry_buttons:
            if entry["rect"].collidepoint(pos):
                self._set_history_preview(int(entry["ply"]))
                return "handled"

        if not self._history_preview_active() and (self.game_over or self._show_match_result_modal()) and self.modal_buttons:
            for button in self.modal_buttons:
                if not button["rect"].collidepoint(pos):
                    continue
                if not button["enabled"]:
                    return "handled"
                if button["action"] == "modal_restart":
                    self.restart_game()
                    return "handled"
                if button["action"] == "modal_menu":
                    if not self._show_match_result_modal():
                        self._save_current_game()
                    return "menu"
            return "handled"

        for entry in self.analysis_entry_buttons:
            if not entry["rect"].collidepoint(pos):
                continue
            self._select_analysis_entry(entry)
            return "handled"

        for button in self.action_buttons:
            if not button["rect"].collidepoint(pos):
                continue
            if not button["enabled"]:
                return "handled"

            action = button["action"]
            if action in {"history_live", "live_board"}:
                self._return_to_live_board()
                return "handled"
            if action == "history_prev":
                self._history_prev()
                return "handled"
            if action == "history_next":
                self._history_next()
                return "handled"
            if action.startswith("tab_"):
                self.inspector_tab = action.removeprefix("tab_")
                self.history_entry_buttons = []
                self.analysis_entry_buttons = []
                return "handled"
            if action == "menu":
                if not self._match_enabled():
                    if self.board.is_game_over():
                        self._save_current_game()
                    else:
                        self._save_current_game(
                            force_incomplete=True,
                            termination="abandoned by return-to-menu",
                        )
                return "menu"
            if action == "flip":
                self.flipped = not self.flipped
                return "handled"
            if action == "undo":
                self.undo_moves()
                return "handled"
            if action == "rewind" and self.selected_history_ply is not None:
                self._rewind_to_ply(self.selected_history_ply)
                return "handled"
            if action == "restart":
                self.restart_game()
                return "handled"
            if action == "pause":
                self._toggle_pause()
                return "handled"
            if action == "resume_ai":
                self._resume_ai()
                return "handled"
            if action == "play_candidate":
                self._apply_selected_analysis_choice()
                return "handled"
            if action == "toggle_mcts" and self.game_mode == "human_vs_ai" and self.mcts_enabled:
                self.use_mcts = not self.use_mcts
                return "handled"

        return None

    def _record_pre_move_state(self):
        """Store pre-move state for history-aware inference."""
        self.board_history.append(self.board.copy())
        current_native = _native_board(self.board)
        current_key = _board_position_key(current_native)
        repetition_count = max(
            0,
            int(self.position_counts.get(current_key, 1)) - 1,
        )
        for mcts in (self.mcts1, self.mcts2):
            if mcts:
                mcts.update_history(
                    current_native,
                    repetition_count=(
                        repetition_count
                        if mcts._multi._path_dependent_repetition
                        else None
                    ),
                )

    def _rebuild_position_counts(self):
        self.position_counts = _position_counts_for_history(
            self.board_history,
            self.board,
        )

    def _sync_mcts_histories(self):
        """Rebuild MCTS histories after undoing moves."""
        self._rebuild_position_counts()
        for mcts in (self.mcts1, self.mcts2):
            if mcts:
                mcts.reset_tree()
                previous_counts = {}
                for hist_board in self.board_history:
                    native_board = _native_board(hist_board)
                    key = _board_position_key(native_board)
                    repetition_count = max(0, int(previous_counts.get(key, 0)))
                    mcts.update_history(
                        native_board,
                        repetition_count=(
                            repetition_count
                            if mcts._multi._path_dependent_repetition
                            else None
                        ),
                    )
                    previous_counts[key] = repetition_count + 1

    def _branch_to_ply(self, target_ply, reason="continued from inspected position"):
        try:
            target_ply = int(target_ply)
        except (TypeError, ValueError):
            return False
        target_ply = max(0, min(target_ply, len(self.move_history)))

        original_moves = list(self.move_history)
        self.pending_manual_move = None
        if self._match_enabled():
            self._detach_match_for_analysis(reason=reason)
        else:
            self.manual_analysis_mode = True
        self._cancel_preview_analysis()
        self._invalidate_ai_turn()
        self.ai_paused = True

        rebuilt_board = chess.Board(self.initial_fen)
        rebuilt_history = []
        rebuilt_moves = []
        rebuilt_san = []
        for move in original_moves[:target_ply]:
            rebuilt_history.append(rebuilt_board.copy())
            rebuilt_san.append(rebuilt_board.san(move))
            rebuilt_board.push(move)
            rebuilt_moves.append(move)

        self.board = rebuilt_board
        self.board_history = rebuilt_history
        self._rebuild_position_counts()
        self.move_history = rebuilt_moves
        self.move_san_history = rebuilt_san
        self._prune_mcts_snapshots_for_line(rebuilt_moves)
        self.game_over = self.board.is_game_over()
        self.current_game_saved = False
        self._reset_selection_state()
        self._reset_ai_state(paused=True)
        if self.draining_ai_turn is not None and self.draining_ai_turn.is_done:
            self.draining_ai_turn.wait(timeout=0)
            self.draining_ai_turn = None
        self.pending_mcts_reset = self.draining_ai_turn is not None
        if not self.pending_mcts_reset:
            self._sync_mcts_histories()
        self._refresh_analysis_cache()
        return True

    def _rewind_to_ply(self, target_ply):
        """Branch the playable line at the selected position and keep AI paused."""
        try:
            target_ply = int(target_ply)
        except (TypeError, ValueError):
            return 0
        target_ply = max(0, min(target_ply, len(self.move_history)))
        removed = len(self.move_history) - target_ply
        if not self._branch_to_ply(target_ply, reason="rewound from inspected position"):
            return 0
        return removed

    def undo_moves(self, plies=None):
        """Branch and undo from the inspected ply (two half-moves vs one AI)."""
        if plies is None:
            plies = 2 if self.game_mode == "human_vs_ai" else 1
        try:
            plies = max(0, int(plies))
        except (TypeError, ValueError):
            return 0
        if plies <= 0 or not self.move_history:
            return 0
        start_ply = self.selected_history_ply if self._history_preview_active() else len(self.move_history)
        start_ply = max(0, min(int(start_ply or 0), len(self.move_history)))
        target_ply = max(0, start_ply - plies)
        removed = len(self.move_history) - target_ply
        if not self._branch_to_ply(target_ply, reason="undo from inspected position"):
            return 0
        return removed

    def _save_current_game(self, force_incomplete=False, termination=None):
        """Save current game to PGN once."""
        if self.current_game_saved:
            return None
        if not self.move_history:
            return None
        match_result = self._board_result_for_match()
        is_complete = match_result != "*"
        if not is_complete and not force_incomplete:
            return None
        if not self.save_games_pgn:
            self.current_game_saved = True
            self.game_index += 1
            return None

        result_override = None if self.board.is_game_over() else (match_result if is_complete else "*")
        game = build_pgn_game(
            board=self.board,
            move_history=self.move_history,
            initial_fen=self.initial_fen,
            version=self.version,
            game_mode=self.game_mode,
            mcts_enabled=self.mcts_enabled,
            game_index=self.game_index,
            human_color=self.human_color,
            model1_name=self.model1_name,
            model2_name=self.model2_name,
            result_override=result_override,
            termination=termination,
        )
        output_path = save_game_to_pgn(
            games_dir=self.games_dir,
            version=self.version,
            game_index=self.game_index,
            game=game,
        )

        self.current_game_saved = True
        self.game_index += 1
        if self.console_verbose:
            print(f"Saved game PGN: {output_path}")
        return output_path

    def _apply_move(self, move):
        """Apply move, update histories, and persist finished game."""
        if (
            self.pending_mcts_reset
            and self.draining_ai_turn is not None
            and not self.draining_ai_turn.is_done
        ):
            return False
        moving_color = self.board.turn
        if self.ai_thinking and self.ai_thinking_color == moving_color and self.ai_thinking_started_at is not None:
            elapsed = self._pending_ai_compute_duration
            if elapsed is None:
                elapsed = self._current_thinking_elapsed()
            elapsed = max(0.0, float(elapsed))
            self.side_time_stats[moving_color]["total"] += elapsed
            self.side_time_stats[moving_color]["moves"] += 1
            self.side_time_stats[moving_color]["last"] = elapsed
        self._pending_ai_compute_duration = None
        old_history_rows = (len(self.move_san_history) + 1) // 2
        keep_history_preview = self._history_preview_active()
        san_move = self.board.san(move)
        self._record_pre_move_state()
        self.board.push(move)
        post_native_board = _native_board(self.board)
        post_key = _board_position_key(post_native_board)
        self.position_counts[post_key] = int(self.position_counts.get(post_key, 0)) + 1
        native_move = native_chess.move_from_uci(move.uci())
        # Reconcile each model's own tree after every real ply. The board
        # history was appended before pushing the move, while the supplied
        # board/counts describe the exact resulting position.
        for mcts in (self.mcts1, self.mcts2):
            if mcts:
                mcts.advance_root(
                    native_move,
                    board=post_native_board,
                    position_counts=self.position_counts,
                    repetition_known=self.position_history_complete,
                )
        self.move_history.append(move)
        self.move_san_history.append(san_move)
        new_history_rows = (len(self.move_san_history) + 1) // 2
        if keep_history_preview:
            self.history_scroll_rows += max(0, new_history_rows - old_history_rows)
        else:
            self.history_scroll_rows = 0
            self.selected_history_ply = None
        self.selected_analysis_move = None
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self.preview_move = None

        if self.board.is_checkmate():
            self._play_sfx("mate")
        elif self.board.is_check():
            self._play_sfx("check")
        else:
            self._play_sfx("move")

        if self._board_result_for_match() != "*":
            self.game_over = True
            self._save_current_game()
            self._record_visible_match_result()
        # Once the visible match game finishes, the background evaluator takes
        # ownership of both loaded models. Avoid a redundant synchronous NN
        # analysis forward on the UI thread during that run.
        if not (self._match_enabled() and self.game_over):
            self._refresh_analysis_cache()
        return True
    
    def handle_click(self, pos):
        """Handle mouse click"""
        if self._history_preview_active() or self.game_over or self.ai_thinking:
            return

        if self.draining_ai_turn is not None and not self.draining_ai_turn.is_done:
            return
        if self.pending_manual_move is not None:
            return
        if self.manual_analysis_mode and not self.ai_paused:
            return
        
        # In AI vs AI mode, no human interaction
        if self.game_mode == "ai_vs_ai" and not self.manual_analysis_mode:
            return
        
        # In human vs AI mode, check if it's human's turn
        if (
            self.game_mode == "human_vs_ai"
            and not self.manual_analysis_mode
            and self.board.turn != self.human_color
        ):
            return
        
        square = self.coords_to_square(pos[0], pos[1])
        if square is None:
            return
        
        piece = self.board.piece_at(square)
        
        # If square is selected and clicking on legal move
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            if move in self.board.legal_moves:
                piece_moving = self.board.piece_at(self.selected_square)
                if piece_moving and piece_moving.piece_type == chess.PAWN:
                    if (piece_moving.color == chess.WHITE and square >= 56) or \
                       (piece_moving.color == chess.BLACK and square <= 7):
                        move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self._apply_move(move)
                    self.selected_square = None
                    self.legal_moves = []
                    return
        
        # Select piece
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
        else:
            self.selected_square = None
            self.legal_moves = []
    
    def _history_positions_for_model(self, model):
        if model is self.model2:
            return self.model2_history_positions
        return self.model1_history_positions

    def _build_history_tensor(
        self,
        current_board,
        history_positions,
        model=None,
        history_boards=None,
    ):
        """
         v4.2: Build tensor with history using POV-aware board_to_tensor
        
        The board_to_tensor function from data_helpers automatically handles:
        - POV (perspective from current player)
        - Flipping for black to move
        
        Args:
            current_board: chess.Board for current position
        
        Returns:
            numpy array: (input_planes, 8, 8) tensor
        """
        history_positions = max(0, int(history_positions))
        if history_boards is None:
            history_boards = self.board_history

        encoder_id = getattr(model, "input_encoder_id", None)
        if encoder_id is None and model is not None:
            model_spec = getattr(model, "model_spec", {}) or {}
            encoder_id = (model_spec.get("input_spec") or {}).get("encoder_id")
        encoder_id = str(
            encoder_id
            or self.config.get("model", {}).get("input_encoder_id", "planes_history_v1")
        )
        if encoder_id == "planes_history_v2":
            encoder = get_input_encoder(encoder_id)
            native_current = _native_board(current_board)
            native_all_history = [_native_board(board) for board in history_boards]
            all_boards = native_all_history + [native_current]
            seen = {}
            all_repetition_counts = []
            for history_board in all_boards:
                key = _board_position_key(history_board)
                all_repetition_counts.append(int(seen.get(key, 0)))
                seen[key] = int(seen.get(key, 0)) + 1
            native_history = (
                native_all_history[-history_positions:] if history_positions else []
            )
            repetition_counts = (
                all_repetition_counts[-(len(native_history) + 1):-1]
                + [all_repetition_counts[-1]]
                if native_history
                else [all_repetition_counts[-1]]
            )
            return encoder.encode_history(
                native_current,
                history_boards=native_history,
                history_positions=history_positions,
                repetition_counts=repetition_counts,
                dtype=np.float32,
            )

        if history_positions == 0:
            # No history - just current board
            # board_to_tensor automatically handles POV
            return board_to_tensor(_native_board(current_board))
        
        # Build history tensors
        tensors = []
        
        # Get last N boards from history
        if history_boards:
            history_boards = history_boards[-history_positions:]
            # Convert history boards to tensors with POV
            for hist_board in history_boards:
                hist_tensor = board_to_tensor(
                    _native_board(hist_board),
                    flip_perspective=(current_board.turn == chess.BLACK),
                )
                tensors.append(hist_tensor)
        
        # Pad with ZEROS if not enough history (matching training data!)
        while len(tensors) < history_positions:
            #  v4.4 FIX: Use zeros, not chess.Board() - matches BinaryChessDataset padding
            empty_tensor = np.zeros((16, 8, 8), dtype=np.float32)
            tensors.insert(0, empty_tensor)
        
        # Add current board
        current_tensor = board_to_tensor(_native_board(current_board))
        tensors.append(current_tensor)
        
        # Stack: [oldest_history, ..., newest_history, current]
        # Shape: (16 * (history_positions + 1), 8, 8)
        return np.concatenate(tensors, axis=0)
    
    def _get_network_move(
        self,
        model,
        board_snapshot=None,
        history_snapshot=None,
        history_positions=None,
    ):
        """
         v4.2: Get move directly from network (no MCTS)
        
        Uses POV-aware board_to_tensor for correct history handling
        """
        board_snapshot = board_snapshot or self.board.copy(stack=True)
        if history_snapshot is None:
            history_snapshot = [board.copy(stack=True) for board in self.board_history]
        if history_positions is None:
            history_positions = self._history_positions_for_model(model)
        board_tensor = torch.FloatTensor(
            self._build_history_tensor(
                board_snapshot,
                history_positions=history_positions,
                model=model,
                history_boards=history_snapshot,
            )
        ).unsqueeze(0).to(self.device)
        
        # Get policy from model
        with torch.no_grad():
            with self.inference_lock:
                policy_logits, _ = model(
                    board_tensor,
                    apply_log_softmax=False,
                )
            policy = policy_logits.float().cpu().numpy()[0]
        
        # Find best legal move
        best_score = -float('inf')
        best_move = None
        native_board = _native_board(board_snapshot)
        scored_moves = []
        for native_move in native_chess.legal_moves(native_board):
            #  v4.4 FIX: Use POV-aware move_to_index (handles black's perspective)
            idx = move_to_index(native_move, native_board)
            score = float(policy[idx])
            scored_moves.append((native_move, score))
            if score > best_score:
                best_score = policy[idx]
                best_move = _python_move(native_move)

        rows = []
        if scored_moves:
            max_logit = max(score for _, score in scored_moves)
            exp_scores = [
                (native_move, math.exp(score - max_logit))
                for native_move, score in scored_moves
            ]
            score_sum = sum(score for _, score in exp_scores)
            for native_move, score in sorted(exp_scores, key=lambda item: item[1], reverse=True):
                move = _python_move(native_move)
                rows.append(
                    {
                        "move": move.uci(),
                        "san": board_snapshot.san(move),
                        "probability": score / score_sum if score_sum > 0 else 0.0,
                        "selected": move == best_move,
                    }
                )
                if len(rows) >= ANALYSIS_DISPLAY_ROWS:
                    break

        return {"move": best_move, "analysis_rows": rows}

    def _ai_should_move(self):
        if self.game_over:
            return False
        if self.game_mode == "ai_vs_ai":
            return not self._match_enabled() or self._visible_match_active()
        if self.game_mode == "human_vs_ai":
            return self.board.turn != self.human_color
        return False

    def _invalidate_ai_turn(self):
        """Invalidate a pending result without blocking the Pygame thread."""
        self.ai_generation += 1
        turn = self.ai_turn
        if turn is not None:
            turn.invalidate()
            if not turn.is_done:
                self.draining_ai_turn = turn
            self.ai_turn = None
        self.ai_turn_history_key = None
        self.pending_ai_choice = None
        self._reset_ai_state(paused=self.ai_paused)

    def _tick_ai_turn(self):
        """Collect completed work and apply it only to the captured position."""
        draining = self.draining_ai_turn
        if draining is not None and draining.is_done:
            self.draining_ai_turn = None
            if self.pending_mcts_reset:
                self._sync_mcts_histories()
                self.pending_mcts_reset = False
            pending_move = self.pending_manual_move
            if pending_move is not None:
                self.pending_manual_move = None
                try:
                    move = chess.Move.from_uci(pending_move["move"])
                except (KeyError, ValueError):
                    move = None
                if (
                    pending_move.get("fen") == self.board.fen()
                    and move is not None
                    and move in self.board.legal_moves
                ):
                    color = self.board.turn
                    if self._apply_move(move):
                        self.selected_analysis_move = (color, move.uci())
                        self.selected_analysis_candidate = dict(pending_move["candidate"])
                        self._set_preview_move(color, move.uci(), pending_move["fen"])
                else:
                    self.selected_analysis_candidate = None
                    self.selected_analysis_move = None
            else:
                self._refresh_analysis_cache()

        turn = self.ai_turn
        if turn is None or not turn.is_done:
            return
        if (
            turn.cancelled.is_set()
            or turn.generation != self.ai_generation
            or turn.board_fen != self.board.fen()
            or not self._ai_should_move()
        ):
            self.ai_turn = None
            self.ai_turn_history_key = None
            self._reset_ai_state(paused=self.ai_paused)
            return

        if turn.error is not None:
            self.ai_turn = None
            self.ai_turn_history_key = None
            self.pending_ai_choice = None
            self.ai_status_error = f"{type(turn.error).__name__}: {turn.error}"
            if self.game_mode == "ai_vs_ai":
                self.ai_paused = True
            if self._visible_match_active():
                self._record_visible_match_error(self.ai_status_error)
            self._reset_ai_state(paused=self.ai_paused)
            return
        result = turn.result or {}
        result_history_key = self.ai_turn_history_key
        move = self._publish_ai_turn_analysis(turn, result, result_history_key)

        if move is None or move not in self.board.legal_moves:
            self.ai_turn = None
            self.ai_turn_history_key = None
            self.pending_ai_choice = None
            self.ai_status_error = "AI search returned no legal move; restart the game to retry."
            if self.game_mode == "ai_vs_ai":
                self.ai_paused = True
            if self._visible_match_active():
                self._record_visible_match_error(self.ai_status_error)
            self._reset_ai_state(paused=self.ai_paused)
            return
        # A completed search is inspectable immediately, even while the user
        # has paused or the visible-move deadline is still being held.
        if self.ai_paused or not turn.is_ready_to_apply:
            return

        self.ai_turn = None
        self.ai_turn_history_key = None
        self.ai_status_error = None
        self._pending_ai_compute_duration = turn.compute_duration_seconds
        choice = dict(self.pending_ai_choice or {})
        if self._apply_move(move) and choice:
            choice["resulting_fen"] = self.board.fen()
            choice["resulting_history_key"] = self._history_path_key(len(self.move_history))
            self.last_ai_choice = choice
        self.pending_ai_choice = None
        self._reset_ai_state(paused=self.ai_paused)

    def _start_ai_turn(self):
        if self.ai_turn is not None or self.draining_ai_turn is not None:
            return
        if self.pending_mcts_reset:
            self._sync_mcts_histories()
            self.pending_mcts_reset = False
        if not self._ai_should_move() or self.ai_paused or self.ai_status_error:
            return

        color = self.board.turn
        if self.game_mode == "ai_vs_ai":
            current_model = self.model1 if color == chess.WHITE else self.model2
            current_mcts = self.mcts1 if color == chess.WHITE else self.mcts2
        else:
            current_model = self.model1
            current_mcts = self.mcts1
        if current_model is None:
            return

        board_snapshot = self.board.copy(stack=True)
        history_snapshot = [board.copy(stack=True) for board in self.board_history]
        position_counts_snapshot = dict(self.position_counts)
        current_fen = board_snapshot.fen()
        self.ai_turn_history_key = self._history_path_key(len(self.move_history))
        history_positions = self._history_positions_for_model(current_model)
        use_mcts = self._side_uses_mcts(color) and current_mcts is not None
        if use_mcts:
            simulations = self.mcts_simulations_white if color == chess.WHITE else self.mcts_simulations_black

            def compute():
                search_result = current_mcts.search(
                    _native_board(board_snapshot),
                    simulations,
                    add_root_noise=False,
                    position_counts=position_counts_snapshot,
                    repetition_known=self.position_history_complete,
                )
                return {
                    "kind": "mcts",
                    "search_result": search_result,
                    "requested_simulations": simulations,
                    "root_noise": False,
                }
        else:
            def compute():
                return self._get_network_move(
                    current_model,
                    board_snapshot=board_snapshot,
                    history_snapshot=history_snapshot,
                    history_positions=history_positions,
                )

        self.ai_generation += 1
        play_cfg = self.config.get("play", {}) if isinstance(self.config, dict) else {}
        try:
            min_visible_seconds = max(
                0.0,
                float(play_cfg.get("min_visible_move_seconds", 1.0)),
            )
        except (TypeError, ValueError):
            min_visible_seconds = 1.0
        self.ai_turn = PlayAITurn(
            generation=self.ai_generation,
            board_fen=current_fen,
            compute=compute,
            min_visible_move_seconds=min_visible_seconds,
        ).start()
        self.ai_thinking = True
        self.ai_thinking_color = color
        self.ai_thinking_started_at = time.perf_counter()
        self.ai_pause_started_at = None

    def ai_move(self):
        """Start or poll one asynchronous AI turn; never search on this thread."""
        if not self.ai_thinking:
            if not self._ai_should_move() or self.ai_paused or self.ai_status_error:
                return
            self.ai_thinking = True
            self.ai_thinking_color = self.board.turn
            self.ai_thinking_started_at = time.perf_counter()
            self.ai_pause_started_at = None
            return
        if self.ai_turn is None and self.draining_ai_turn is None:
            self._start_ai_turn()
    
    def restart_game(self):
        """Restart the game"""
        if self.move_history and not self.current_game_saved:
            if self.board.is_game_over():
                self._save_current_game()
            else:
                self._save_current_game(
                    force_incomplete=True,
                    termination="abandoned by restart",
                )

        self._stop_background_match(reason="visible game abandoned by restart")
        self.match_detached = False
        self.inspector_tab = "moves"
        self.manual_analysis_mode = False
        self.last_match_session = None
        self.ai_status_error = None
        self.pending_manual_move = None
        active_ai_turn = self.ai_turn is not None and not self.ai_turn.is_done
        self._invalidate_ai_turn()
        self.pending_mcts_reset = self.pending_mcts_reset or active_ai_turn or self.draining_ai_turn is not None
        self.visible_match_finished = False
        self.visible_match_result = None
        self.visible_match_winner = None
        self.visible_match_error = None
        self.visible_mcts_searches = []
        self.visible_match_start_pending = self.draining_match_session is not None
        self.match_start_pending = self.visible_match_start_pending
        self.match_started_at = time.perf_counter()
        self.match_started_at_unix = time.time()
        self.board = chess.Board()
        self.initial_fen = self.board.fen()
        self.game_over = False
        self.move_history = []
        self.move_san_history = []
        self.board_history = []  #  Clear board history
        self.position_counts = _position_counts_for_history([], self.board)
        self.position_history_complete = True
        self.current_game_saved = False
        self._reset_selection_state()
        self._reset_ai_state(paused=False)
        
        #  v4.2: Reset MCTS trees and histories
        if not self.pending_mcts_reset:
            if self.mcts1:
                self.mcts1.reset_tree()
            if self.mcts2:
                self.mcts2.reset_tree()
        with self.match_lock:
            self.match_generation += 1
            self.match_stats = self._new_match_stats()
            if self.visible_match_start_pending:
                self.match_stats["status"] = "stopping_previous"
                self.match_stats["background_starting"] = False
        self._reset_analysis_storage()
        self._refresh_analysis_cache()
        if self._match_enabled():
            self._prime_visible_match_opening()
    
    def run(self):
        """Main game loop.

        Returns:
            str: 'menu' to return to setup, 'quit' to exit application.
        """
        running = True
        exit_action = "quit"
        
        while running:
            self.clock.tick(FPS)
            self._tick_match_lifecycle()
            self._tick_ai_turn()
            self._tick_preview_analysis()
            mapped_mouse = self._window_to_canvas(pygame.mouse.get_pos())
            if mapped_mouse is None:
                self.mouse_canvas_pos = (-9999, -9999)
            else:
                self.mouse_canvas_pos = mapped_mouse
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if not self._match_enabled():
                        if self.board.is_game_over():
                            self._save_current_game()
                        else:
                            self._save_current_game(
                                force_incomplete=True,
                                termination="abandoned by quit",
                            )
                    exit_action = "quit"
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    if self.maximized:
                        self._sync_window_surface()
                    else:
                        self._set_window_size(event.w, event.h)

                elif event.type == pygame.MOUSEWHEEL:
                    mapped_pos = self._window_to_canvas(pygame.mouse.get_pos())
                    if mapped_pos is not None and self.analysis_rect.collidepoint(mapped_pos):
                        self.analysis_scroll_rows = max(0, self.analysis_scroll_rows + int(event.y))
                    elif mapped_pos is not None and self.history_rect.collidepoint(mapped_pos):
                        self.history_scroll_rows = max(0, self.history_scroll_rows + int(event.y))
                 
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mapped_pos = self._window_to_canvas(event.pos)
                        if mapped_pos is not None:
                            ui_action = self._handle_ui_click(mapped_pos)
                            if ui_action == "menu":
                                exit_action = "menu"
                                running = False
                                continue
                            if ui_action == "handled":
                                continue
                            self.handle_click(mapped_pos)
                 
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.restart_game()
                    elif event.key == pygame.K_u:
                        self.undo_moves()
                    elif event.key == pygame.K_ESCAPE:
                        if not self._match_enabled():
                            if self.board.is_game_over():
                                self._save_current_game()
                            else:
                                self._save_current_game(
                                    force_incomplete=True,
                                    termination="abandoned by return-to-menu",
                                )
                        exit_action = "menu"
                        running = False
                    elif event.key == pygame.K_f:
                        self.flipped = not self.flipped
                    elif event.key == pygame.K_SPACE and self.game_mode == "ai_vs_ai":
                        self._toggle_pause()
                    elif event.key == pygame.K_m and self.game_mode == "human_vs_ai" and self.mcts_enabled:
                        self.use_mcts = not self.use_mcts
                        if self.console_verbose:
                            print(f" Toggled AI mode: {'MCTS' if self.use_mcts else 'Network-only'}")
            
            # AI move
            if not self.game_over and self._ai_should_move():
                self.ai_move()
            
            # Draw everything
            self.canvas.blit(self.background, (0, 0))
            self.draw_board()
            self.draw_pieces()
            self._draw_match_preview_status()
            self.draw_hud()
            self._draw_endgame_modal()
            self._update_mouse_cursor(self.hover_any_button)

            self.screen.fill(APP_BG)
            self.screen.blit(self.canvas, (0, 0))

            pygame.display.flip()
        self._update_mouse_cursor(False)
        self._stop_background_match(reason=f"play screen exited: {exit_action}")
        self._invalidate_ai_turn()
        # A cancelled search owns its SingleGame session until its worker
        # returns. Close immediately when already drained; main() closes the
        # remaining session after waiting for any outstanding worker.
        if self.draining_ai_turn is None or self.draining_ai_turn.is_done:
            for mcts in (self.mcts1, self.mcts2):
                if mcts is not None:
                    mcts.close()
        return exit_action


def main():
    _enable_windows_dpi_awareness()
    pygame.init()
    start_piece_asset_prefetch()
    parser = argparse.ArgumentParser(description="Chess AI Game")
    parser.add_argument(
        "--no-mcts",
        action="store_true",
        help="Disable MCTS (use network-only mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logs (model details and setup summary).",
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(sorted(MODEL_REGISTRY)),
        default=None,
        help="Select the model profile used when no checkpoint supplies its own model_spec.",
    )
    args = parser.parse_args()

    config_path = default_config_path()
    config = load_project_config(architecture_id=args.architecture)
    if args.architecture and config.get("model", {}).get("architecture_id") != args.architecture:
        raise ValueError(
            f"Active config model profile conflicts with --architecture {args.architecture!r}."
        )

    verbose_console = bool(args.verbose)
    config.setdefault("model", {})
    config["model"]["print_summary"] = verbose_console

    model_version = config.get("model", {}).get("version", "v?.?")

    device = torch.device(config["hardware"]["device"])
    if device.type == "cuda" and torch.cuda.is_available():
        device_label = torch.cuda.get_device_name(device.index or 0)
    else:
        device_label = str(device).upper()
    print(f"\nChess AI Play  ·  {model_version}  ·  {device_label}")
    print("Choose a mode and models in the setup window.")
    if verbose_console:
        print(f"Config: {config_path}")

    default_use_mcts = not args.no_mcts

    if verbose_console:
        history_positions = config["model"].get("history_positions", 0)
        print(f"History positions: {history_positions}")
        input_encoder = get_input_encoder(
            config["model"].get("input_encoder_id", "planes_history_v1")
        )
        print(f"Input planes: {input_encoder.input_planes(history_positions)}")

    base_dir = script_dir.parent
    models_root = (base_dir / config.get("paths", {}).get("models_dir", "models")).resolve()
    last_window_size = None
    last_window_maximized = True
    pending_match_session = None
    pending_ai_turn = None
    previous_gui = None
    gui = None
    model1 = None
    model2 = None

    def drain_previous_match():
        nonlocal pending_match_session, pending_ai_turn, previous_gui, gui, model1, model2
        session = pending_match_session
        if session is not None and not session.is_done:
            print("Stopping previous model match before loading another checkpoint...")
            session.request_stop()
            session.wait()
        pending_match_session = None
        ai_turn = pending_ai_turn
        if ai_turn is not None and not ai_turn.is_done:
            print("Waiting for the previous AI search to finish before loading another checkpoint...")
            ai_turn.wait()
        pending_ai_turn = None
        if previous_gui is not None:
            # Drop the old GUI's search trees and model references only after
            # its evaluator has drained, before the next checkpoint is loaded.
            for mcts in (previous_gui.mcts1, previous_gui.mcts2):
                if mcts is not None:
                    mcts.close()
            previous_gui.mcts1 = None
            previous_gui.mcts2 = None
            previous_gui.model1 = None
            previous_gui.model2 = None
            previous_gui = None
        gui = None
        model1 = None
        model2 = None

    while True:
        setup = select_models(
            base_dir,
            config,
            default_use_mcts=default_use_mcts,
            initial_window_size=last_window_size,
            initial_window_maximized=last_window_maximized,
        )
        if setup is None:
            drain_previous_match()
            print("Setup cancelled. Exiting.")
            break

        setup_window_size = _normalize_window_size(setup.get("window_size"))
        if setup_window_size is not None:
            last_window_size = setup_window_size
        last_window_maximized = bool(setup.get("window_maximized", False))
        _apply_setup_play_config(config, setup)

        model1_path = setup["model1_path"]
        model2_path = setup["model2_path"]
        game_mode = setup["game_mode"]
        human_color = setup["human_color"]
        enable_mcts = bool(setup["use_mcts"])
        setup_use_mcts_white = bool(setup.get("use_mcts_white", enable_mcts))
        setup_use_mcts_black = bool(setup.get("use_mcts_black", enable_mcts))
        setup_mcts_simulations = setup.get("mcts_simulations")
        setup_mcts_simulations_white = setup.get("mcts_simulations_white")
        setup_mcts_simulations_black = setup.get("mcts_simulations_black")
        setup_match_games = setup.get("match_games", 1)
        # The in-game Menu remains immediate. Once the user chooses another
        # setup, drain the cancelled evaluator before loading models or taking
        # GPU time for a second match.
        drain_previous_match()
        if setup_mcts_simulations is not None:
            try:
                sims_value = max(1, int(setup_mcts_simulations))
                config.setdefault("reinforcement_learning", {})
                config["reinforcement_learning"]["mcts_simulations"] = sims_value
            except (TypeError, ValueError):
                pass
        if not enable_mcts and verbose_console:
            print("MCTS disabled from setup window")

        setup_log_path = write_setup_log(base_dir, config, setup)
        if verbose_console:
            print(f"Setup log saved: {setup_log_path}")

        if game_mode in ["human_vs_ai", "ai_vs_ai"]:
            if model1_path is None:
                print("No model selected. Exiting.")
                break

            if verbose_console:
                print()
                print("Loading models...")
            model1 = None
            model2 = None
            loading_model_path = model1_path
            try:
                model1 = load_model_from_checkpoint(
                    model1_path, config, device, verbose=verbose_console
                )
                if game_mode == "ai_vs_ai":
                    if model2_path is None:
                        print("No second model selected. Exiting.")
                        break
                    loading_model_path = model2_path
                    model2 = load_model_from_checkpoint(
                        model2_path, config, device, verbose=verbose_console
                    )
            except Exception as exc:
                if isinstance(exc, ModelCompatibilityError):
                    print(f"Model checkpoint is incompatible: {loading_model_path.name}")
                else:
                    print(f"Could not load model {loading_model_path.name}: {exc}")
                if verbose_console:
                    print(f"  {exc}")
                model1 = None
                model2 = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not show_model_load_error(loading_model_path, exc):
                    break
                continue
        else:
            model1 = None
            model2 = None

        if model1 is not None:
            try:
                prepared_count = prepare_models_for_play_search(model1, model2, device)
            except Exception as exc:
                # Layout preparation is an optional CUDA optimization; keep
                # the playable NCHW models if the conversion is unavailable.
                prepared_count = 0
                if verbose_console:
                    print(f"Could not prepare channels-last search models: {exc}")
            if prepared_count and verbose_console:
                print(f"Prepared {prepared_count} model(s) for channels-last MCTS search")

        model1_meta = {}
        model2_meta = {}
        if model1_path is not None:
            try:
                model1_meta = load_checkpoint_metadata(model1_path, base_dir=models_root)
            except Exception:
                model1_meta = {}
        if model2_path is not None:
            try:
                model2_meta = load_checkpoint_metadata(model2_path, base_dir=models_root)
            except Exception:
                model2_meta = {}

        if verbose_console:
            print()
            print("=" * 50)
            print(f"Chess AI {model_version} - Pygame GUI")
            print("=" * 50)
            print()
            print(f"Setup log: {setup_log_path}")
            print()

        gui = ChessGUI(
            model1,
            model2,
            config,
            device,
            game_mode,
            enable_mcts=enable_mcts,
            model1_name=model1_path.name if model1_path else None,
            model2_name=model2_path.name if model2_path else None,
            console_verbose=verbose_console,
            initial_window_size=last_window_size,
            initial_window_maximized=last_window_maximized,
            model1_meta=model1_meta,
            model2_meta=model2_meta,
            model1_checkpoint_path=str(model1_path.resolve()) if model1_path else None,
            model2_checkpoint_path=str(model2_path.resolve()) if model2_path else None,
            use_mcts_white=setup_use_mcts_white,
            use_mcts_black=setup_use_mcts_black,
            mcts_simulations_white=setup_mcts_simulations_white,
            mcts_simulations_black=setup_mcts_simulations_black,
            match_games=setup_match_games,
            human_color=human_color,
        )

        if game_mode == "human_vs_ai":
            gui.flipped = human_color == chess.BLACK
            if verbose_console:
                if human_color == chess.BLACK:
                    print()
                    print("You are playing as Black!")
                else:
                    print()
                    print("You are playing as White!")
        elif game_mode == "ai_vs_ai":
            if verbose_console:
                print()
                print("Watching AI vs AI match...")
                print(f"White: {model1_path.name}")
                print(f"Black: {model2_path.name}")
        else:
            if verbose_console:
                print()
                print("2-Player mode activated!")

        if verbose_console:
            print()
            if enable_mcts:
                print("AI Mode: MCTS")
            else:
                print("AI Mode: Network-only")
            print()
            print("Starting game...")
            print()

        exit_action = gui.run()
        pending_match_session = gui.draining_match_session
        pending_ai_turn = gui.draining_ai_turn
        state = gui.get_window_state()
        gui_window_size = _normalize_window_size(state.get("window_size"))
        if gui_window_size is not None:
            last_window_size = gui_window_size
        last_window_maximized = bool(state.get("window_maximized", last_window_maximized))
        if exit_action == "menu":
            previous_gui = gui
            if verbose_console:
                print("Returned to setup menu.")
            continue
        previous_gui = gui
        drain_previous_match()
        break

    pygame.quit()


if __name__ == "__main__":
    main()

