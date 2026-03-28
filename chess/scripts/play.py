"""
Chess GUI Game Interface - v4.5
 v4.5: CRITICAL FIXES - Promotions support
 v4.4: Compatible with POV + Dynamic Sliding Window
-  POV: Automatic perspective handling
-  Sliding Window: Correct history assembly
-  MCTS toggle: --no-mcts flag for network-only mode
-  Promotions: Promotion-aware action space (see ACTION_SIZE)
-  Fixed imports for v4.5
"""

import torch
import chess
import yaml
import sys
import io
import math
import ctypes
import time
import wave
from array import array
from pathlib import Path
import pygame
import argparse

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet
from src.batch_selfplay import MCTS, select_move_by_visits

#  v4.2: Import board_to_tensor from data_helpers
#  v4.4: Added move_to_index for POV-aware move encoding
from src.utils.data_helpers import board_to_tensor, move_to_index

# Import from utils
from utils.ui.game_setup import load_model_from_checkpoint, select_models, write_setup_log
from utils.shared.model_catalog import load_checkpoint_metadata
from utils.ui.gui_helpers import (
    build_pgn_game,
    create_piece_surfaces,
    get_game_mode_labels,
    get_result_message,
    get_turn_color,
    resolve_games_dir,
    save_game_to_pgn,
    start_piece_asset_prefetch,
)

# Initialize Pygame
pygame.init()

# Constants
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
BOARD_MIN_SIZE = 416
BOARD_MAX_SIZE = 960
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
WINDOW_MIN_WIDTH = 980
WINDOW_MIN_HEIGHT = 700
MAX_GAME_UI_SCALE = 1.35
PANEL_MIN_WIDTH = 120
DEFAULT_WINDOW_SCREEN_WIDTH_RATIO = 0.92
DEFAULT_WINDOW_SCREEN_HEIGHT_RATIO = 0.92

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68, 150)
SELECTED = (246, 246, 105, 150)
LEGAL_MOVE = (100, 100, 100, 120)
CAPTURE_MOVE = (200, 50, 50, 120)
SIDEBAR_BG = (34, 40, 52)
TEXT_COLOR = (245, 248, 252)
APP_BG = (12, 16, 22)


def _get_default_window_size():
    """Resolve a safe default window size from the current display."""
    fallback_width = max(WINDOW_MIN_WIDTH, WINDOW_WIDTH)
    fallback_height = max(WINDOW_MIN_HEIGHT, WINDOW_HEIGHT)
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


def _get_maximized_window_size():
    """Resolve the largest safe client size for the current display."""
    default_width, default_height = _get_default_window_size()
    try:
        info = pygame.display.Info()
        screen_width = int(getattr(info, "current_w", 0) or 0)
        screen_height = int(getattr(info, "current_h", 0) or 0)
    except Exception:
        screen_width = 0
        screen_height = 0

    if screen_width <= 0 or screen_height <= 0:
        return default_width, default_height

    return (
        max(WINDOW_MIN_WIDTH, screen_width),
        max(WINDOW_MIN_HEIGHT, screen_height),
    )


def _maximize_native_window():
    """Ask the OS to maximize the current window when available."""
    try:
        wm_info = pygame.display.get_wm_info()
    except Exception:
        return False

    window_handle = wm_info.get("window")
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
        use_mcts_white=None,
        use_mcts_black=None,
        mcts_simulations_white=None,
        mcts_simulations_black=None,
    ):
        self.model1 = model1  # White AI or main AI
        self.model2 = model2  # Black AI (for AI vs AI mode)
        self.config = config
        self.version = config.get('model', {}).get('version', 'v?.?')
        self.device = device
        self.game_mode = game_mode  # "human_vs_ai", "ai_vs_ai", "human_vs_human"
        self.model1_name = model1_name or f"ChessAI-{self.version}-A"
        self.model2_name = model2_name or f"ChessAI-{self.version}-B"
        self.model1_meta = dict(model1_meta or {})
        self.model2_meta = dict(model2_meta or {})
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
        
        #  MCTS toggle
        self.mcts_enabled = enable_mcts
        self.use_mcts_white = bool(enable_mcts if use_mcts_white is None else use_mcts_white)
        self.use_mcts_black = bool(enable_mcts if use_mcts_black is None else use_mcts_black)
        
        # History depth can differ per loaded checkpoint architecture.
        default_history = int(config['model'].get('history_positions', 0))
        self.model1_history_positions = int(getattr(model1, 'history_positions', default_history)) if model1 else default_history
        self.model2_history_positions = int(getattr(model2, 'history_positions', self.model1_history_positions)) if model2 else self.model1_history_positions
        self.history_positions = self.model1_history_positions
        
        # Only create MCTS if enabled
        if model1 and self.mcts_enabled:
            self.mcts1 = MCTS(model1, config, device)
            self.mcts1.history_positions = self.model1_history_positions
        else:
            self.mcts1 = None
            
        if model2 and self.mcts_enabled:
            self.mcts2 = MCTS(model2, config, device)
            self.mcts2.history_positions = self.model2_history_positions
        else:
            self.mcts2 = None

        default_window_width, default_window_height = _get_default_window_size()
        self.base_width = max(WINDOW_MIN_WIDTH, default_window_width)
        self.base_height = max(WINDOW_MIN_HEIGHT, default_window_height)
        self.display_flags = pygame.RESIZABLE
        self.maximized = False
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
        self.screen = pygame.display.set_mode(self.restore_window_size, self.display_flags)
        self.viewport_rect = pygame.Rect(0, 0, self.base_width, self.base_height)
        self.viewport_scale = 1.0
        self.canvas = pygame.Surface((self.base_width, self.base_height))
        self.background = pygame.Surface((self.base_width, self.base_height))
        self.left_panel_rect = pygame.Rect(0, 0, 0, 0)
        self.board_rect = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
        self.right_panel_rect = pygame.Rect(0, 0, 0, 0)
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
        self.history_entry_buttons = []
        self.analysis_entry_buttons = []
        self.history_scroll_rows = 0
        self.selected_history_ply = None
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
        self.analysis_cache = {"fen": None, "rows": [], "side": None, "label": ""}
        self.analysis_cache_by_color = {
            chess.WHITE: {"rows": [], "label": "White", "fen": None},
            chess.BLACK: {"rows": [], "label": "Black", "fen": None},
        }
        self._resize_canvas(*self.restore_window_size)
        self._update_viewport()
        try:
            pygame.display.set_window_minimum_size((WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT))
        except Exception:
            pass

        pygame.display.set_caption(f"Chess AI {self.version}")
        self.clock = pygame.time.Clock()
        if bool(initial_window_maximized) or initial_window_size is None:
            self._toggle_maximized()
        
        # Load piece images
        self._ensure_piece_surfaces()
        
        # Fonts
        self.text_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.medium_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 17)
        self.tiny_font = pygame.font.SysFont("Segoe UI", 15)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.human_color = chess.WHITE  # Only used in human_vs_ai mode
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
        self._refresh_analysis_cache()

    @staticmethod
    def _build_vertical_gradient(width, height, top_color, bottom_color):
        surface = pygame.Surface((width, height))
        denom = max(1, height - 1)
        for y in range(height):
            t = y / denom
            color = (
                int(top_color[0] + (bottom_color[0] - top_color[0]) * t),
                int(top_color[1] + (bottom_color[1] - top_color[1]) * t),
                int(top_color[2] + (bottom_color[2] - top_color[2]) * t),
            )
            pygame.draw.line(surface, color, (0, y), (width, y))
        return surface

    def _resize_canvas(self, width, height):
        self.base_width = max(WINDOW_MIN_WIDTH, int(width))
        self.base_height = max(WINDOW_MIN_HEIGHT, int(height))
        self.canvas = pygame.Surface((self.base_width, self.base_height))
        self.background = self._build_vertical_gradient(
            self.base_width, self.base_height, (27, 32, 40), (15, 19, 25)
        )
        self._refresh_layout()
        self._ensure_piece_surfaces()

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
        content_left = OUTER_MARGIN_X
        content_top = OUTER_MARGIN_Y
        content_width = max(1, self.base_width - 2 * OUTER_MARGIN_X)
        content_height = max(1, self.base_height - 2 * OUTER_MARGIN_Y)

        left_min = max(PANEL_MIN_WIDTH, 210)
        right_min = max(PANEL_MIN_WIDTH, 230)
        left_pref = max(left_min, min(300, int(content_width * 0.20)))
        right_pref = max(right_min, min(340, int(content_width * 0.22)))

        board_limit_by_h = min(BOARD_MAX_SIZE, content_height)
        board_w_with_pref = content_width - left_pref - right_pref - (2 * PANEL_GAP)
        if board_w_with_pref >= BOARD_MIN_SIZE:
            left_width = left_pref
            right_width = right_pref
            board_size = min(board_limit_by_h, board_w_with_pref)
        else:
            left_width = left_min
            right_width = right_min
            board_size = min(board_limit_by_h, content_width - left_width - right_width - (2 * PANEL_GAP))

        board_size = max(8 * 24, int(board_size))
        board_size = max(8, (board_size // 8) * 8)
        board_size = min(board_size, board_limit_by_h)
        self.square_size = max(1, board_size // 8)
        board_size = self.square_size * 8

        layout_width = left_width + PANEL_GAP + board_size + PANEL_GAP + right_width
        start_x = content_left + max(0, (content_width - layout_width) // 2)
        left_x = start_x
        board_x = left_x + left_width + PANEL_GAP
        right_x = board_x + board_size + PANEL_GAP
        board_y = content_top + max(0, (content_height - board_size) // 2)

        self.left_panel_rect = pygame.Rect(left_x, content_top, left_width, content_height)
        self.board_rect = pygame.Rect(board_x, board_y, board_size, board_size)
        self.right_panel_rect = pygame.Rect(right_x, content_top, right_width, content_height)

    def _update_viewport(self):
        win_w, win_h = self.screen.get_size()
        self.viewport_rect = pygame.Rect(0, 0, win_w, win_h)
        self.viewport_scale = 1.0

    def _window_to_canvas(self, pos):
        if pos is None:
            return None
        x = int(pos[0])
        y = int(pos[1])
        x = max(0, min(self.base_width - 1, x))
        y = max(0, min(self.base_height - 1, y))
        return x, y

    def _set_window_size(self, width, height):
        width = max(WINDOW_MIN_WIDTH, int(width))
        height = max(WINDOW_MIN_HEIGHT, int(height))
        self.maximized = False
        self.restore_window_size = (width, height)
        self.screen = pygame.display.set_mode((width, height), self.display_flags)
        self._resize_canvas(width, height)
        self._update_viewport()

    def _toggle_maximized(self):
        if not self.maximized:
            self.restore_window_size = self.screen.get_size()
            maximized_size = _get_maximized_window_size()
            self.screen = pygame.display.set_mode(maximized_size, self.display_flags)
            pygame.event.pump()
            _maximize_native_window()
            self.maximized = True
        else:
            self.screen = pygame.display.set_mode(self.restore_window_size, self.display_flags)
            self.maximized = False
        self._resize_canvas(*self.screen.get_size())
        self._update_viewport()

    def get_window_state(self):
        return {
            "window_size": [int(self.screen.get_width()), int(self.screen.get_height())],
            "window_maximized": bool(self.maximized),
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
        # Board frame over gradient background
        board_outer = self.board_rect.copy()
        pygame.draw.rect(self.canvas, (30, 36, 45), board_outer, border_radius=12)
        pygame.draw.rect(self.canvas, (66, 79, 98), board_outer, width=2, border_radius=12)

        for rank in range(8):
            for file in range(8):
                x = self.board_rect.left + file * self.square_size
                y = self.board_rect.top + rank * self.square_size
                
                color = WHITE if (rank + file) % 2 == 0 else BLACK
                pygame.draw.rect(self.canvas, color, (x, y, self.square_size, self.square_size))
        
        # Draw file/rank labels
        label_font = pygame.font.SysFont("Segoe UI", max(11, int(self.square_size * 0.18)))
        files = "abcdefgh"
        ranks = "87654321"
        
        for i in range(8):
            # File labels (bottom)
            file_label = label_font.render(
                files[i] if not self.flipped else files[7-i], 
                True, (100, 100, 100)
            )
            self.canvas.blit(
                file_label,
                (
                    self.board_rect.left + i * self.square_size + self.square_size - max(8, int(self.square_size * 0.18)),
                    self.board_rect.bottom - max(16, int(self.square_size * 0.22)),
                ),
            )
            
            # Rank labels (left)
            rank_label = label_font.render(
                ranks[i] if not self.flipped else ranks[7-i], 
                True, (100, 100, 100)
            )
            self.canvas.blit(
                rank_label,
                (
                    self.board_rect.left + max(3, int(self.square_size * 0.06)),
                    self.board_rect.top + i * self.square_size + max(2, int(self.square_size * 0.05)),
                ),
            )
        
        # Highlight last move
        if self.move_history:
            last_move = self.move_history[-1]
            for square in [last_move.from_square, last_move.to_square]:
                x, y = self.square_to_coords(square)
                s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                s.fill(HIGHLIGHT)
                self.canvas.blit(s, (x, y))
        
        # Highlight selected square
        if self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(SELECTED)
            self.canvas.blit(s, (x, y))
        
        # Highlight legal moves
        for move in self.legal_moves:
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
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
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
        text = str(text)
        if font.size(text)[0] <= max_width:
            return text
        suffix = "..."
        trimmed = text
        while trimmed and font.size(trimmed + suffix)[0] > max_width:
            trimmed = trimmed[:-1]
        return (trimmed + suffix) if trimmed else suffix

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
    def _format_elo(metadata):
        if not metadata:
            return "n/a"
        elo_value = metadata.get("elo")
        if elo_value is None:
            return "n/a"
        try:
            return str(int(round(float(elo_value))))
        except (TypeError, ValueError):
            return "n/a"

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
        for idx in range(len(self.board_history) - 1, -1, -1):
            hist_board = self.board_history[idx]
            if hist_board and hist_board.fen() == fen:
                return idx
        return None

    def _apply_analysis_choice(self, color, move_uci, fen):
        if not move_uci:
            return False
        target_ply = self._find_analysis_rewind_ply(fen)
        if target_ply is None:
            return False
        try:
            move = chess.Move.from_uci(str(move_uci))
        except ValueError:
            return False
        self._rewind_to_ply(target_ply)
        if self.board.turn != color:
            return False
        if move not in self.board.legal_moves:
            return False
        self.ai_thinking = False
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self.ai_paused = False
        self._apply_move(move)
        self.selected_analysis_move = (color, move_uci)
        self._set_preview_move(color, move_uci, fen)
        return True

    def _draw_preview_move(self):
        if not self.preview_move:
            return
        move_uci = self.preview_move.get("move")
        if not move_uci:
            return
        preview_board = self._board_from_fen(self.preview_move.get("fen")) or self.board
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
        if self.game_mode != "ai_vs_ai":
            return
        if not self.ai_paused:
            self.ai_paused = True
            if self.ai_thinking and self.ai_pause_started_at is None:
                self.ai_pause_started_at = time.perf_counter()
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

    def _side_uses_mcts(self, color):
        if self.game_mode == "ai_vs_ai":
            return bool(self.use_mcts_white if color == chess.WHITE else self.use_mcts_black)
        return bool(self.use_mcts)

    def _refresh_analysis_cache(self):
        current_fen = self.board.fen()
        if self.analysis_cache.get("fen") == current_fen:
            return

        model = self._current_analysis_model()
        side_label = "White" if self.board.turn == chess.WHITE else "Black"
        if model is None:
            self.analysis_cache = {"fen": current_fen, "rows": [], "side": self.board.turn, "label": side_label}
            self.analysis_cache_by_color[self.board.turn] = {"rows": [], "label": side_label, "fen": current_fen}
            return

        try:
            history_positions = self._history_positions_for_model(model)
            board_tensor = torch.FloatTensor(
                self._build_history_tensor(self.board, history_positions=history_positions)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, _ = model(
                    board_tensor,
                    apply_log_softmax=False,
                )
                logits = policy_logits.float().cpu().numpy()[0]

            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                self.analysis_cache = {"fen": current_fen, "rows": [], "side": self.board.turn, "label": side_label}
                self.analysis_cache_by_color[self.board.turn] = {"rows": [], "label": side_label, "fen": current_fen}
                return

            scored_moves = []
            for move in legal_moves:
                idx = move_to_index(move, self.board)
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
            for move, exp_score in sorted_exp_scores:
                probability = (exp_score / score_sum) if score_sum > 0 else 0.0
                if probability < 0.10:
                    continue
                rows.append(
                    {
                        "move": move.uci(),
                        "san": self.board.san(move),
                        "probability": probability,
                    }
                )
                if len(rows) >= 5:
                    break
            if not rows and sorted_exp_scores:
                move, exp_score = sorted_exp_scores[0]
                probability = (exp_score / score_sum) if score_sum > 0 else 0.0
                rows.append(
                    {
                        "move": move.uci(),
                        "san": self.board.san(move),
                        "probability": probability,
                    }
                )
            self.analysis_cache = {"fen": current_fen, "rows": rows, "side": self.board.turn, "label": side_label}
            self.analysis_cache_by_color[self.board.turn] = {"rows": rows, "label": side_label, "fen": current_fen}
        except Exception:
            self.analysis_cache = {"fen": current_fen, "rows": [], "side": self.board.turn, "label": side_label}
            self.analysis_cache_by_color[self.board.turn] = {"rows": [], "label": side_label, "fen": current_fen}

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

        return {
            "side": side_name,
            "is_human": False,
            "name": Path(str(model_name or "model.pt")).stem,
            "version": self._format_version(model_meta),
            "elo": self._format_elo(model_meta),
            "avg_time": self.side_time_stats[color]["total"] / max(1, self.side_time_stats[color]["moves"]),
            "total_time": self.side_time_stats[color]["total"],
        }

    def _draw_card(self, rect, fill=(40, 49, 63), border=(88, 105, 132)):
        pygame.draw.rect(self.canvas, fill, rect, border_radius=10)
        pygame.draw.rect(self.canvas, border, rect, width=1, border_radius=10)

    def _player_card_lines(self, side_info):
        lines = [
            (self.tiny_font, "title"),
            (self.small_font, f"{side_info['side']} side"),
            (self.medium_font, side_info["name"]),
        ]
        if side_info["is_human"]:
            lines.append((self.tiny_font, "Type: Human"))
        else:
            lines.extend(
                [
                    (self.tiny_font, f"Version: {side_info['version']}"),
                    (self.tiny_font, f"Elo: {side_info['elo']}"),
                    (self.tiny_font, f"Avg: {self._format_duration(side_info['avg_time'])}"),
                    (self.tiny_font, f"Total: {self._format_duration(side_info['total_time'])}"),
                ]
            )
        return lines

    def _draw_player_card(self, rect, title, side_info, active=False):
        accent = (102, 168, 240) if active else (92, 109, 136)
        self._draw_card(rect, fill=(35, 43, 56), border=accent)

        inner_left = rect.left + 12
        inner_width = rect.width - 24
        y = rect.top + 10
        side_color = (238, 242, 249) if side_info["side"] == "White" else (209, 219, 234)
        subtitle_color = (170, 193, 223)
        meta_color = (172, 185, 205)
        line_gap = max(5, self.tiny_font.get_height() // 3)

        row_specs = self._player_card_lines(side_info)
        for idx, (font, text) in enumerate(row_specs):
            if idx == 0:
                rendered = font.render(title, True, subtitle_color)
            elif idx == 1:
                rendered = font.render(self._fit_text(font, text, inner_width), True, side_color)
            elif idx == 2:
                rendered = font.render(self._fit_text(font, text, inner_width), True, TEXT_COLOR)
            else:
                rendered = font.render(self._fit_text(font, text, inner_width), True, meta_color)
            self.canvas.blit(rendered, (inner_left, y))
            y += rendered.get_height() + line_gap

    def _player_card_height(self, side_info):
        top_pad = 10
        bottom_pad = 12
        line_gap = max(5, self.tiny_font.get_height() // 3)
        lines = self._player_card_lines(side_info)
        total_height = top_pad + bottom_pad
        for idx, (font, _) in enumerate(lines):
            total_height += font.get_height()
            if idx < len(lines) - 1:
                total_height += line_gap
        return total_height

    def _draw_analysis_card(self, rect, color, active=False):
        label = "White" if color == chess.WHITE else "Black"
        cache = self.analysis_cache_by_color.get(color, {"rows": [], "fen": None})
        rows = list(cache.get("rows", []))
        border = (102, 168, 240) if active else (92, 109, 136)
        self._draw_card(rect, fill=(23, 30, 41), border=border)
        self.analysis_entry_buttons = [entry for entry in self.analysis_entry_buttons if entry.get("color") != color]
        self.canvas.blit(
            self.small_font.render(f"Top 5 ruchow ({label})", True, TEXT_COLOR),
            (rect.left + 12, rect.top + 10),
        )
        if not rows:
            self.canvas.blit(
                self.tiny_font.render("Brak danych. Pojawia sie po ruchu tej strony.", True, (172, 185, 205)),
                (rect.left + 12, rect.top + 40),
            )
            return

        y = rect.top + 40
        row_h = max(24, self.tiny_font.get_height() + 10)
        for idx, row in enumerate(rows, start=1):
            row_rect = pygame.Rect(rect.left + 8, y - 2, rect.width - 16, row_h)
            probability = max(0.0, min(1.0, float(row.get("probability", 0.0))))
            if idx % 2 == 1:
                pygame.draw.rect(self.canvas, (29, 37, 49), row_rect, border_radius=6)
            else:
                pygame.draw.rect(self.canvas, (25, 32, 44), row_rect, border_radius=6)
            fill_width = max(10, int((row_rect.width - 2) * probability))
            fill_rect = pygame.Rect(row_rect.left + 1, row_rect.top + 1, fill_width, row_rect.height - 2)
            fill_color = (54, 95, 152) if color == chess.WHITE else (76, 109, 160)
            pygame.draw.rect(self.canvas, fill_color, fill_rect, border_radius=6)
            if self.selected_analysis_move == (color, row.get("move")):
                pygame.draw.rect(self.canvas, (160, 205, 255), row_rect, width=2, border_radius=6)
            self.canvas.blit(
                self.tiny_font.render(f"{idx}.", True, (140, 154, 179)),
                (row_rect.left + 8, y),
            )
            pct_surface = self.tiny_font.render(f"{probability * 100:4.1f}%", True, (164, 201, 238))
            pct_x = row_rect.right - pct_surface.get_width() - 8
            san_width = max(40, pct_x - (row_rect.left + 28) - 8)
            self.canvas.blit(
                self.tiny_font.render(self._fit_text(self.tiny_font, row["san"], san_width), True, (216, 226, 239)),
                (row_rect.left + 28, y),
            )
            self.canvas.blit(pct_surface, (pct_x, y))
            self.analysis_entry_buttons.append(
                {
                    "rect": row_rect.copy(),
                    "color": color,
                    "move": row.get("move"),
                    "fen": cache.get("fen"),
                }
            )
            y += row_h + 2

    def _analysis_card_height(self, color):
        rows = list(self.analysis_cache_by_color.get(color, {}).get("rows", []))
        row_count = max(1, min(5, len(rows)))
        row_h = max(24, self.tiny_font.get_height() + 10)
        return 18 + self.small_font.get_height() + 12 + row_count * (row_h + 2) + 10

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
            fill = (54, 60, 74)
            border = (83, 92, 111)
            text_color = (128, 137, 154)
        elif danger:
            fill = (138, 70, 74)
            border = (200, 118, 120)
            text_color = (245, 235, 236)
        elif active:
            fill = (66, 104, 162)
            border = (125, 177, 246)
            text_color = (240, 246, 252)
        else:
            fill = (59, 77, 106)
            border = (105, 145, 206)
            text_color = (230, 239, 250)

        if enabled:
            hover_fill = tuple(min(255, c + 26) for c in fill)
            hover_border = tuple(min(255, c + 34) for c in border)
            fill = self._lerp_color(fill, hover_fill, hover_t)
            border = self._lerp_color(border, hover_border, hover_t)

        animated_rect = rect.move(0, -int(round(2 * max(0.0, hover_t))))
        shadow_rect = animated_rect.move(0, 2)
        pygame.draw.rect(self.canvas, (12, 16, 24), shadow_rect, border_radius=8)
        pygame.draw.rect(self.canvas, fill, animated_rect, border_radius=8)
        pygame.draw.rect(self.canvas, border, animated_rect, width=1, border_radius=8)
        text = self.small_font.render(label, True, text_color)
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

        self.canvas.blit(
            self.text_font.render("Match View", True, TEXT_COLOR),
            (panel.left + 16, panel.top + 14),
        )

        mode_text, _ = get_game_mode_labels(self.game_mode, self.human_color)
        top_display_color = chess.WHITE if self.flipped else chess.BLACK
        bottom_display_color = chess.BLACK if top_display_color == chess.WHITE else chess.WHITE
        top_info = self._side_info(top_display_color)
        bottom_info = self._side_info(bottom_display_color)

        status_height = 18 + self.small_font.get_height() + 8 + self.tiny_font.get_height() + 16
        status_rect = pygame.Rect(panel.left + 14, panel.top + 52, panel.width - 28, status_height)
        self._draw_card(status_rect, fill=(28, 35, 47), border=(77, 95, 124))
        if self.ai_paused:
            status_line = "Gra wstrzymana"
            status_color = (255, 214, 120)
        elif self.ai_thinking:
            thinking_side = "White" if self.ai_thinking_color == chess.WHITE else "Black"
            thinking_text = f"{thinking_side} mysli{self._thinking_dots()}"
            elapsed = self._format_duration(self._current_thinking_elapsed())
            status_line = f"{thinking_text}  {elapsed}"
            status_color = (255, 214, 120)
        else:
            status_line = mode_text
            status_color = (168, 183, 206)
        self.canvas.blit(
            self.small_font.render(status_line, True, status_color),
            (status_rect.left + 12, status_rect.top + 10),
        )
        turn_side = "White" if self.board.turn == chess.WHITE else "Black"
        turn_color = get_turn_color(self.game_mode, self.board.turn, self.human_color)
        self.canvas.blit(
            self.tiny_font.render(f"Ruch: {turn_side}", True, turn_color),
            (status_rect.left + 12, status_rect.top + 40),
        )

        panel_inner_left = panel.left + 14
        panel_inner_width = panel.width - 28
        top_card_h = self._player_card_height(top_info)
        bottom_card_h = self._player_card_height(bottom_info)
        top_analysis_h = self._analysis_card_height(top_display_color)
        bottom_analysis_h = self._analysis_card_height(bottom_display_color)
        gap = max(10, self.tiny_font.get_height() // 2 + 4)
        top_card = pygame.Rect(panel_inner_left, status_rect.bottom + 14, panel_inner_width, top_card_h)
        top_analysis_rect = pygame.Rect(panel_inner_left, top_card.bottom + gap, panel_inner_width, top_analysis_h)
        bottom_card = pygame.Rect(panel_inner_left, panel.bottom - 14 - bottom_card_h, panel_inner_width, bottom_card_h)
        bottom_analysis_rect = pygame.Rect(
            panel_inner_left,
            bottom_card.top - gap - bottom_analysis_h,
            panel_inner_width,
            bottom_analysis_h,
        )
        self._draw_player_card(top_card, "Top board side", top_info, active=self.board.turn == top_display_color)
        self._draw_player_card(bottom_card, "Bottom board side", bottom_info, active=self.board.turn == bottom_display_color)
        available_gap = bottom_analysis_rect.top - top_analysis_rect.bottom
        if available_gap < 8:
            compressed_gap = max(6, gap - max(0, (8 - available_gap) // 2))
            top_analysis_rect.top = top_card.bottom + compressed_gap
            top_analysis_rect.height = top_analysis_h
            bottom_analysis_rect.top = bottom_card.top - compressed_gap - bottom_analysis_h
        if top_analysis_rect.bottom <= bottom_analysis_rect.top - 4:
            self._draw_analysis_card(top_analysis_rect, top_display_color, active=self.board.turn == top_display_color)
            self._draw_analysis_card(bottom_analysis_rect, bottom_display_color, active=self.board.turn == bottom_display_color)

    def _draw_right_panel(self):
        panel = self.right_panel_rect
        row_h = max(24, self.tiny_font.get_height() + 8)
        max_rows = 10
        history_header_h = 16 + self.tiny_font.get_height() + 10
        history_height = history_header_h + (max_rows * row_h) + 12
        action_rows = 4 if self.game_mode in ("ai_vs_ai", "human_vs_ai") else 3
        button_area_height = 34 + action_rows * 38 + (action_rows - 1) * 10 + 14
        content_height = 56 + history_height + 14 + button_area_height + 14
        content_rect = pygame.Rect(panel.left, panel.top, panel.width, content_height)
        self._draw_card(content_rect, fill=(28, 35, 47), border=(77, 95, 124))

        self.canvas.blit(
            self.text_font.render("Moves", True, TEXT_COLOR),
            (content_rect.left + 16, content_rect.top + 14),
        )

        history_rect = pygame.Rect(content_rect.left + 14, content_rect.top + 56, content_rect.width - 28, history_height)
        self.history_rect = history_rect
        button_area = pygame.Rect(content_rect.left + 14, history_rect.bottom + 14, content_rect.width - 28, button_area_height)

        self._draw_card(history_rect, fill=(23, 30, 41), border=(60, 76, 102))
        self.canvas.blit(
            self.tiny_font.render("No   White                         Black", True, (172, 191, 220)),
            (history_rect.left + 10, history_rect.top + 10),
        )

        rows = []
        for i in range(0, len(self.move_san_history), 2):
            rows.append(
                (
                    (i // 2) + 1,
                    self.move_san_history[i],
                    self.move_san_history[i + 1] if i + 1 < len(self.move_san_history) else "",
                )
            )

        num_col_w = 24
        col_gap = 8
        row_left = history_rect.left + 10
        move_col_w = max(56, (history_rect.width - 20 - num_col_w - (col_gap * 2)) // 2)
        white_x = row_left + num_col_w + col_gap
        black_x = white_x + move_col_w + col_gap

        max_scroll = max(0, len(rows) - max_rows)
        self.history_scroll_rows = max(0, min(self.history_scroll_rows, max_scroll))
        start_idx = max(0, len(rows) - max_rows - self.history_scroll_rows)
        end_idx = max(0, len(rows) - self.history_scroll_rows)
        visible_rows = rows[start_idx:end_idx]

        y = history_rect.top + history_header_h
        self.history_entry_buttons = []
        for move_no, white_move, black_move in visible_rows:
            row_rect = pygame.Rect(history_rect.left + 6, y - 2, history_rect.width - 12, row_h)
            white_ply = (move_no - 1) * 2 + 1
            black_ply = min(len(self.move_history), white_ply + 1)
            if self.selected_history_ply in (white_ply, black_ply):
                pygame.draw.rect(self.canvas, (52, 78, 115), row_rect, border_radius=6)
            elif move_no % 2 == 1:
                pygame.draw.rect(self.canvas, (27, 35, 47), row_rect, border_radius=6)
            self.canvas.blit(
                self.tiny_font.render(f"{move_no:>2}", True, (140, 154, 179)),
                (row_left, y),
            )
            white_rect = pygame.Rect(white_x - 2, y - 2, move_col_w + 4, row_h)
            black_rect = pygame.Rect(black_x - 2, y - 2, move_col_w + 4, row_h)
            self.canvas.blit(
                self.tiny_font.render(self._fit_text(self.tiny_font, white_move, move_col_w), True, (216, 226, 239)),
                (white_x, y),
            )
            self.canvas.blit(
                self.tiny_font.render(self._fit_text(self.tiny_font, black_move, move_col_w), True, (216, 226, 239)),
                (black_x, y),
            )
            self.history_entry_buttons.append({"rect": white_rect, "ply": white_ply})
            if black_move:
                self.history_entry_buttons.append({"rect": black_rect, "ply": black_ply})
            y += row_h

        if max_scroll > 0:
            track = pygame.Rect(history_rect.right - 8, history_rect.top + history_header_h, 4, history_rect.height - history_header_h - 8)
            pygame.draw.rect(self.canvas, (47, 58, 75), track, border_radius=2)
            thumb_h = max(24, int(track.height * (max_rows / max(1, len(rows)))))
            travel = max(0, track.height - thumb_h)
            scroll_ratio = 0.0 if max_scroll <= 0 else (1.0 - (self.history_scroll_rows / max_scroll))
            thumb_y = track.top + int(scroll_ratio * travel)
            pygame.draw.rect(self.canvas, (128, 147, 176), pygame.Rect(track.left, thumb_y, track.width, thumb_h), border_radius=2)

        self._draw_card(button_area, fill=(23, 30, 41), border=(60, 76, 102))
        self.canvas.blit(
            self.tiny_font.render("Actions", True, (172, 191, 220)),
            (button_area.left + 10, button_area.top + 10),
        )

        btn_gap = 10
        btn_w = (button_area.width - 3 * btn_gap) // 2
        btn_h = 38
        x1 = button_area.left + btn_gap
        x2 = x1 + btn_w + btn_gap
        y0 = button_area.top + 34

        buttons = [
            ("menu", "Back to menu", True, False, True, pygame.Rect(x1, y0, btn_w, btn_h)),
            ("flip", "Flip board", True, self.flipped, False, pygame.Rect(x2, y0, btn_w, btn_h)),
            (
                "undo",
                "Undo",
                bool(self.move_history) and not self.ai_thinking,
                False,
                False,
                pygame.Rect(x1, y0 + btn_h + btn_gap, btn_w, btn_h),
            ),
            (
                "restart",
                "Restart",
                not self.ai_thinking,
                False,
                False,
                pygame.Rect(x2, y0 + btn_h + btn_gap, btn_w, btn_h),
            ),
            (
                "rewind",
                "Rewind to selected",
                self.selected_history_ply is not None and not self.ai_thinking,
                False,
                False,
                pygame.Rect(x1, y0 + (btn_h + btn_gap) * 2, btn_w * 2 + btn_gap, btn_h),
            ),
        ]
        next_row_y = y0 + (btn_h + btn_gap) * 3
        if self.game_mode == "ai_vs_ai":
            buttons.append(
                (
                    "pause",
                    "Resume game" if self.ai_paused else "Pause game",
                    True,
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

    def draw_hud(self):
        self._draw_left_panel()
        self._draw_right_panel()

    def _draw_endgame_modal(self):
        self.modal_buttons = []
        if not self.game_over:
            return

        # Modal captures interaction; ignore hover state from underlying UI.
        self.hover_any_button = False
        overlay = pygame.Surface((self.base_width, self.base_height), pygame.SRCALPHA)
        overlay.fill((8, 12, 20, 152))
        self.canvas.blit(overlay, (0, 0))

        modal_w = min(max(500, int(self.base_width * 0.38)), 660)
        modal_h = min(max(340, int(self.base_height * 0.42)), 420)
        modal = pygame.Rect(
            (self.base_width - modal_w) // 2,
            (self.base_height - modal_h) // 2,
            modal_w,
            modal_h,
        )
        shadow = modal.inflate(18, 18)
        shadow_surface = pygame.Surface((shadow.width, shadow.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (5, 8, 14, 90), shadow_surface.get_rect(), border_radius=28)
        self.canvas.blit(shadow_surface, shadow.topleft)
        self._draw_card(modal, fill=(28, 36, 49), border=(118, 149, 196))

        result = self.board.result()
        is_draw = result == "1/2-1/2"
        winner_color = None
        if result == "1-0":
            winner_color = chess.WHITE
        elif result == "0-1":
            winner_color = chess.BLACK

        if winner_color == chess.WHITE:
            hero_text = "Zwycieza Bialy"
            hero_color = (242, 246, 252)
            accent_color = (151, 191, 245)
            piece_symbol = "K"
        elif winner_color == chess.BLACK:
            hero_text = "Zwycieza Czarny"
            hero_color = (236, 214, 166)
            accent_color = (255, 184, 102)
            piece_symbol = "k"
        else:
            hero_text = "Remis"
            hero_color = (232, 236, 243)
            accent_color = (156, 176, 205)
            piece_symbol = None

        title_y = modal.top + 24
        title = self.small_font.render("Game Finished", True, (162, 179, 203))
        self.canvas.blit(title, title.get_rect(center=(modal.centerx, title_y + title.get_height() // 2)))

        if piece_symbol is not None:
            hero_size = max(104, min(148, int(modal.width * 0.20)))
            icon_bg = pygame.Rect(0, 0, hero_size + 42, hero_size + 42)
            icon_bg.center = (modal.centerx, modal.top + 116)
            pygame.draw.rect(self.canvas, (34, 43, 58), icon_bg, border_radius=24)
            pygame.draw.rect(self.canvas, accent_color, icon_bg, width=2, border_radius=24)
            piece_surface = self._scaled_piece_icon(piece_symbol, hero_size)
            if piece_surface is not None:
                self.canvas.blit(piece_surface, piece_surface.get_rect(center=icon_bg.center))
            message_y = icon_bg.bottom + 18
        elif is_draw:
            hero_size = max(78, min(108, int(modal.width * 0.14)))
            duo_bg = pygame.Rect(0, 0, hero_size * 2 + 56, hero_size + 34)
            duo_bg.center = (modal.centerx, modal.top + 112)
            pygame.draw.rect(self.canvas, (34, 43, 58), duo_bg, border_radius=24)
            pygame.draw.rect(self.canvas, accent_color, duo_bg, width=2, border_radius=24)
            white_icon = self._scaled_piece_icon("K", hero_size, fill_ratio=0.82)
            black_icon = self._scaled_piece_icon("k", hero_size, fill_ratio=0.82)
            if white_icon is not None:
                self.canvas.blit(white_icon, white_icon.get_rect(center=(duo_bg.centerx - hero_size // 2, duo_bg.centery)))
            if black_icon is not None:
                self.canvas.blit(black_icon, black_icon.get_rect(center=(duo_bg.centerx + hero_size // 2, duo_bg.centery)))
            message_y = duo_bg.bottom + 18
        else:
            message_y = modal.top + 92

        hero_surface = self.text_font.render(hero_text, True, hero_color)
        self.canvas.blit(hero_surface, hero_surface.get_rect(center=(modal.centerx, message_y + hero_surface.get_height() // 2)))

        subtitle_text = "Partia zakonczona mata" if self.board.is_checkmate() else get_result_message(result)[0]
        subtitle_surface = self.small_font.render(subtitle_text, True, accent_color)
        self.canvas.blit(subtitle_surface, subtitle_surface.get_rect(center=(modal.centerx, message_y + 42)))

        stats_rect = pygame.Rect(modal.left + 28, modal.bottom - 130, modal.width - 56, 54)
        pygame.draw.rect(self.canvas, (22, 29, 40), stats_rect, border_radius=14)
        pygame.draw.rect(self.canvas, (76, 96, 126), stats_rect, width=1, border_radius=14)

        white_name = self._side_info(chess.WHITE)["name"]
        black_name = self._side_info(chess.BLACK)["name"]
        summary_text = f"Ruchy: {len(self.move_history)}   |   White: {white_name}   |   Black: {black_name}"
        summary_surface = self.tiny_font.render(self._fit_text(self.tiny_font, summary_text, stats_rect.width - 24), True, (196, 208, 225))
        self.canvas.blit(summary_surface, summary_surface.get_rect(center=stats_rect.center))

        btn_w = max(170, (modal.width - 28 * 2 - 14) // 2)
        btn_h = 42
        btn_y = modal.bottom - btn_h - 26
        restart_rect = pygame.Rect(modal.left + 28, btn_y, btn_w, btn_h)
        menu_rect = pygame.Rect(restart_rect.right + 14, btn_y, btn_w, btn_h)

        buttons = [
            ("modal_restart", "Play again", True, False, False, restart_rect),
            ("modal_menu", "Back to menu", True, False, True, menu_rect),
        ]
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
            self.modal_buttons.append({"action": action, "rect": rect, "enabled": enabled})

    def _handle_ui_click(self, pos):
        if self.game_over and self.modal_buttons:
            for button in self.modal_buttons:
                if not button["rect"].collidepoint(pos):
                    continue
                if not button["enabled"]:
                    return "handled"
                if button["action"] == "modal_restart":
                    self.restart_game()
                    return "handled"
                if button["action"] == "modal_menu":
                    self._save_current_game()
                    return "menu"
            return "handled"

        for entry in self.history_entry_buttons:
            if entry["rect"].collidepoint(pos):
                self.selected_history_ply = int(entry["ply"])
                return "handled"

        now_ms = pygame.time.get_ticks()
        for entry in self.analysis_entry_buttons:
            if not entry["rect"].collidepoint(pos):
                continue
            entry_key = (entry.get("color"), entry.get("move"), entry.get("fen"))
            is_double_click = (
                self.last_analysis_click.get("key") == entry_key
                and (now_ms - int(self.last_analysis_click.get("time_ms", 0))) <= 420
            )
            self.last_analysis_click = {"key": entry_key, "time_ms": now_ms}
            self.selected_analysis_move = (entry.get("color"), entry.get("move"))
            self._set_preview_move(entry.get("color"), entry.get("move"), entry.get("fen"))
            if is_double_click and self._apply_analysis_choice(entry.get("color"), entry.get("move"), entry.get("fen")):
                return "handled"
            return "handled"

        for button in self.action_buttons:
            if not button["rect"].collidepoint(pos):
                continue
            if not button["enabled"]:
                return "handled"

            action = button["action"]
            if action == "menu":
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
            if action == "toggle_mcts" and self.game_mode == "human_vs_ai" and self.mcts_enabled:
                self.use_mcts = not self.use_mcts
                return "handled"

        return None

    def _record_pre_move_state(self):
        """Store pre-move state for history-aware inference."""
        self.board_history.append(self.board.copy())
        if self.mcts1:
            self.mcts1.update_history(self.board)
        if self.mcts2:
            self.mcts2.update_history(self.board)

    def _sync_mcts_histories(self):
        """Rebuild MCTS histories after undoing moves."""
        for mcts in (self.mcts1, self.mcts2):
            if mcts:
                mcts.reset_tree()
                for hist_board in self.board_history:
                    mcts.update_history(hist_board)

    def _rewind_to_ply(self, target_ply):
        try:
            target_ply = int(target_ply)
        except (TypeError, ValueError):
            return 0
        target_ply = max(0, min(target_ply, len(self.move_history)))

        original_moves = list(self.move_history)
        if target_ply == len(original_moves):
            return 0

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
        self.move_history = rebuilt_moves
        self.move_san_history = rebuilt_san
        self.selected_square = None
        self.legal_moves = []
        self.ai_thinking = False
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self.game_over = self.board.is_game_over()
        self.ai_paused = self.game_mode == "ai_vs_ai" and not self.game_over
        self.current_game_saved = False
        self.selected_history_ply = None
        self.selected_analysis_move = None
        self.preview_move = None
        self.history_scroll_rows = 0
        self._sync_mcts_histories()
        self._refresh_analysis_cache()
        return len(original_moves) - target_ply

    def undo_moves(self, plies=None):
        """Undo last moves. In human-vs-AI defaults to 2 plies."""
        if plies is None:
            plies = 2 if self.game_mode == "human_vs_ai" else 1

        undone = 0
        while undone < plies and self.move_history and self.board.move_stack:
            self.board.pop()
            self.move_history.pop()
            if self.move_san_history:
                self.move_san_history.pop()
            if self.board_history:
                self.board_history.pop()
            undone += 1

        if undone > 0:
            self.selected_square = None
            self.legal_moves = []
            self.ai_thinking = False
            self.game_over = False
            self.current_game_saved = False
            self.selected_history_ply = None
            self.selected_analysis_move = None
            self.preview_move = None
            self.history_scroll_rows = 0
            self._sync_mcts_histories()
            self._refresh_analysis_cache()

        return undone

    def _save_current_game(self, force_incomplete=False, termination=None):
        """Save current game to PGN once."""
        if self.current_game_saved:
            return None
        if not self.move_history:
            return None
        if not self.board.is_game_over() and not force_incomplete:
            return None
        if not self.save_games_pgn:
            self.current_game_saved = True
            self.game_index += 1
            return None

        result_override = None if self.board.is_game_over() else "*"
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
        moving_color = self.board.turn
        if self.ai_thinking and self.ai_thinking_color == moving_color and self.ai_thinking_started_at is not None:
            elapsed = self._current_thinking_elapsed()
            self.side_time_stats[moving_color]["total"] += elapsed
            self.side_time_stats[moving_color]["moves"] += 1
            self.side_time_stats[moving_color]["last"] = elapsed
        san_move = self.board.san(move)
        self._record_pre_move_state()
        # Move is known here, so we can advance cached MCTS roots directly.
        for mcts in (self.mcts1, self.mcts2):
            if mcts:
                mcts.advance_root(move)
        self.board.push(move)
        self.move_history.append(move)
        self.move_san_history.append(san_move)
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

        if self.board.is_game_over():
            self.game_over = True
            self._save_current_game()
        self._refresh_analysis_cache()
    
    def handle_click(self, pos):
        """Handle mouse click"""
        if self.game_over or self.ai_thinking:
            return
        
        # In AI vs AI mode, no human interaction
        if self.game_mode == "ai_vs_ai":
            return
        
        # In human vs AI mode, check if it's human's turn
        if self.game_mode == "human_vs_ai" and self.board.turn != self.human_color:
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

    def _build_history_tensor(self, current_board, history_positions):
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

        if history_positions == 0:
            # No history - just current board
            # board_to_tensor automatically handles POV
            return board_to_tensor(current_board)
        
        # Build history tensors
        tensors = []
        
        # Get last N boards from history
        if self.board_history:
            history_boards = self.board_history[-history_positions:]
            # Convert history boards to tensors with POV
            for hist_board in history_boards:
                hist_tensor = board_to_tensor(hist_board, flip_perspective=(current_board.turn == chess.BLACK))
                tensors.append(hist_tensor)
        
        # Pad with ZEROS if not enough history (matching training data!)
        while len(tensors) < history_positions:
            #  v4.4 FIX: Use zeros, not chess.Board() - matches BinaryChessDataset padding
            import numpy as np
            empty_tensor = np.zeros((16, 8, 8), dtype=np.float32)
            tensors.insert(0, empty_tensor)
        
        # Add current board
        current_tensor = board_to_tensor(current_board)
        tensors.append(current_tensor)
        
        # Stack: [oldest_history, ..., newest_history, current]
        # Shape: (16 * (history_positions + 1), 8, 8)
        import numpy as np
        return np.concatenate(tensors, axis=0)
    
    def _get_network_move(self, model):
        """
         v4.2: Get move directly from network (no MCTS)
        
        Uses POV-aware board_to_tensor for correct history handling
        """
        # Build tensor with history and POV
        history_positions = self._history_positions_for_model(model)
        board_tensor = torch.FloatTensor(
            self._build_history_tensor(self.board, history_positions=history_positions)
        ).unsqueeze(0).to(self.device)
        
        # Get policy from model
        with torch.no_grad():
            policy_logits, _ = model(
                board_tensor,
                apply_log_softmax=False,
            )
            policy = policy_logits.float().cpu().numpy()[0]
        
        # Find best legal move
        best_score = -1
        best_move = None
        for move in self.board.legal_moves:
            #  v4.4 FIX: Use POV-aware move_to_index (handles black's perspective)
            idx = move_to_index(move, self.board)
            if policy[idx] > best_score:
                best_score = policy[idx]
                best_move = move
        
        return best_move
    
    def ai_move(self):
        """Make AI move"""
        # In human vs human mode, no AI moves
        if self.game_mode == "human_vs_human":
            return
        
        # In human vs AI mode, check if it's AI's turn
        if self.game_mode == "human_vs_ai" and self.board.turn == self.human_color:
            return
        
        if self.game_over:
            return

        if self.ai_paused:
            return
        
        if not self.ai_thinking:
            self.ai_thinking = True
            self.ai_thinking_color = self.board.turn
            self.ai_thinking_started_at = time.perf_counter()
            self.ai_pause_started_at = None
            return
        
        # Select the appropriate model
        if self.game_mode == "ai_vs_ai":
            current_model = self.model1 if self.board.turn == chess.WHITE else self.model2
            current_mcts = self.mcts1 if self.board.turn == chess.WHITE else self.mcts2
        else:  # human_vs_ai
            current_model = self.model1
            current_mcts = self.mcts1
        
        #  Get AI move - with MCTS or network-only
        if self._side_uses_mcts(self.board.turn) and current_mcts is not None:
            # MCTS mode (if available)
            current_mcts_sims = self.mcts_simulations_white if self.board.turn == chess.WHITE else self.mcts_simulations_black
            visit_counts = current_mcts.search(
                self.board, 
                current_mcts_sims
            )
            move, _ = select_move_by_visits(visit_counts, temperature=0)
        else:
            #  v4.2: Network-only mode with POV support
            move = self._get_network_move(current_model)
        
        if move:
            self._apply_move(move)
        
        self.ai_thinking = False
    
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

        self.board = chess.Board()
        self.initial_fen = self.board.fen()
        self.selected_square = None
        self.legal_moves = []
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        self.move_san_history = []
        self.board_history = []  #  Clear board history
        self.current_game_saved = False
        self.history_scroll_rows = 0
        self.selected_history_ply = None
        self.selected_analysis_move = None
        self.ai_paused = False
        self.ai_thinking_color = None
        self.ai_thinking_started_at = None
        self.ai_pause_started_at = None
        self.preview_move = None
        
        #  v4.2: Reset MCTS trees and histories
        if self.mcts1:
            self.mcts1.reset_tree()
        if self.mcts2:
            self.mcts2.reset_tree()
        self.analysis_cache = {"fen": None, "rows": [], "side": None, "label": ""}
        self.analysis_cache_by_color = {
            chess.WHITE: {"rows": [], "label": "White", "fen": None},
            chess.BLACK: {"rows": [], "label": "Black", "fen": None},
        }
        self._refresh_analysis_cache()
    
    def run(self):
        """Main game loop.

        Returns:
            str: 'menu' to return to setup, 'quit' to exit application.
        """
        running = True
        exit_action = "quit"
        
        while running:
            self.clock.tick(FPS)
            mapped_mouse = self._window_to_canvas(pygame.mouse.get_pos())
            if mapped_mouse is None:
                self.mouse_canvas_pos = (-9999, -9999)
            else:
                self.mouse_canvas_pos = mapped_mouse
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
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
                    self._set_window_size(event.w, event.h)

                elif event.type == pygame.MOUSEWHEEL:
                    mapped_pos = self._window_to_canvas(pygame.mouse.get_pos())
                    if mapped_pos is not None and self.history_rect.collidepoint(mapped_pos):
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
                    elif event.key == pygame.K_F11:
                        self._toggle_maximized()
                    elif event.key == pygame.K_ESCAPE:
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
            if not self.game_over and (self.game_mode == "ai_vs_ai" or 
                                        (self.game_mode == "human_vs_ai" and self.board.turn != self.human_color)):
                self.ai_move()
            
            # Draw everything
            self.canvas.blit(self.background, (0, 0))
            self.draw_board()
            self.draw_pieces()
            self.draw_hud()
            self._draw_endgame_modal()
            self._update_mouse_cursor(self.hover_any_button)

            self.screen.fill(APP_BG)
            self.screen.blit(self.canvas, (0, 0))

            pygame.display.flip()
        self._update_mouse_cursor(False)
        return exit_action


def main():
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
    args = parser.parse_args()

    config_path = script_dir.parent / "config" / "config.yaml"
    print(f"Loading config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)

    verbose_console = bool(args.verbose)
    config.setdefault("model", {})
    config["model"]["print_summary"] = verbose_console

    model_version = config.get("model", {}).get("version", "v?.?")

    device = torch.device(config["hardware"]["device"])
    print(f"Using device: {device}")

    default_use_mcts = not args.no_mcts

    if verbose_console:
        history_positions = config["model"].get("history_positions", 0)
        print(f"History positions: {history_positions}")
        print(f"Input planes: {16 * (1 + history_positions)}")

    base_dir = script_dir.parent
    models_root = (base_dir / config.get("paths", {}).get("models_dir", "models")).resolve()
    last_window_size = None
    last_window_maximized = True

    while True:
        setup = select_models(
            base_dir,
            config,
            default_use_mcts=default_use_mcts,
            initial_window_size=last_window_size,
            initial_window_maximized=True,
        )
        if setup is None:
            print("Setup cancelled. Exiting.")
            break

        setup_window_size = setup.get("window_size")
        if isinstance(setup_window_size, (list, tuple)) and len(setup_window_size) == 2:
            try:
                last_window_size = [
                    max(WINDOW_MIN_WIDTH, int(setup_window_size[0])),
                    max(WINDOW_MIN_HEIGHT, int(setup_window_size[1])),
                ]
            except (TypeError, ValueError):
                pass
        last_window_maximized = bool(setup.get("window_maximized", False))

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
            model1 = load_model_from_checkpoint(
                model1_path, config, device, ChessNet, verbose=verbose_console
            )

            model2 = None
            if game_mode == "ai_vs_ai":
                if model2_path is None:
                    print("No second model selected. Exiting.")
                    break
                model2 = load_model_from_checkpoint(
                    model2_path, config, device, ChessNet, verbose=verbose_console
                )
        else:
            model1 = None
            model2 = None

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
            initial_window_maximized=True,
            model1_meta=model1_meta,
            model2_meta=model2_meta,
            use_mcts_white=setup_use_mcts_white,
            use_mcts_black=setup_use_mcts_black,
            mcts_simulations_white=setup_mcts_simulations_white,
            mcts_simulations_black=setup_mcts_simulations_black,
        )

        if game_mode == "human_vs_ai":
            gui.human_color = human_color
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
        state = gui.get_window_state()
        gui_window_size = state.get("window_size")
        if isinstance(gui_window_size, (list, tuple)) and len(gui_window_size) == 2:
            try:
                last_window_size = [
                    max(WINDOW_MIN_WIDTH, int(gui_window_size[0])),
                    max(WINDOW_MIN_HEIGHT, int(gui_window_size[1])),
                ]
            except (TypeError, ValueError):
                pass
        last_window_maximized = bool(state.get("window_maximized", last_window_maximized))
        if exit_action == "menu":
            if verbose_console:
                print("Returned to setup menu.")
            continue
        break

    pygame.quit()


if __name__ == "__main__":
    main()

