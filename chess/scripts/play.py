"""
Chess GUI Game Interface - v4.2
🆕 UPDATED: Compatible with POV + Dynamic Sliding Window
- 🎯 POV: Automatic perspective handling
- 🔄 Sliding Window: Correct history assembly
- 🎮 MCTS toggle: --no-mcts flag for network-only mode
- ✅ Fixed imports for v4.2
"""

import torch
import chess
import yaml
import sys
from pathlib import Path
import pygame
import argparse

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet
from src.mcts import MCTS, select_move_by_visits

# 🆕 v4.2: Import board_to_tensor from data_helpers
from src.utils.data_helpers import board_to_tensor

# Import from utils
from utils.gui_helpers import create_piece_surfaces, load_model_from_checkpoint, select_models

# Initialize Pygame
pygame.init()

# Constants
SQUARE_SIZE = 80
BOARD_SIZE = SQUARE_SIZE * 8
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
FPS = 60

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68, 150)
SELECTED = (246, 246, 105, 150)
LEGAL_MOVE = (100, 100, 100, 120)
CAPTURE_MOVE = (200, 50, 50, 120)
SIDEBAR_BG = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)


class ChessGUI:
    """Chess game GUI with support for multiple game modes and MCTS toggle - v4.2"""
    
    def __init__(self, model1, model2, config, device, game_mode="human_vs_ai", enable_mcts=True):
        self.model1 = model1  # White AI or main AI
        self.model2 = model2  # Black AI (for AI vs AI mode)
        self.config = config
        self.device = device
        self.game_mode = game_mode  # "human_vs_ai", "ai_vs_ai", "human_vs_human"
        
        # 🆕 MCTS toggle
        self.mcts_enabled = enable_mcts
        
        # Get history_positions from config
        self.history_positions = config['model'].get('history_positions', 0)
        
        # Only create MCTS if enabled
        if model1 and self.mcts_enabled:
            self.mcts1 = MCTS(model1, config, device)
        else:
            self.mcts1 = None
            
        if model2 and self.mcts_enabled:
            self.mcts2 = MCTS(model2, config, device)
        else:
            self.mcts2 = None
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI v4.2")
        self.clock = pygame.time.Clock()
        
        # Load piece images
        self.pieces = create_piece_surfaces(SQUARE_SIZE)
        
        # Fonts
        self.text_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.human_color = chess.WHITE  # Only used in human_vs_ai mode
        self.use_mcts = self.mcts_enabled  # Can be toggled during game
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        
        # 🆕 v4.2: Board history for neural network input
        # Store chess.Board objects (not tensors) for history
        self.board_history = []
        
        # Flip board for black
        self.flipped = False
    
    def square_to_coords(self, square):
        """Convert chess square to screen coordinates"""
        file = square % 8
        rank = square // 8
        
        if self.flipped:
            x = (7 - file) * SQUARE_SIZE
            y = rank * SQUARE_SIZE
        else:
            x = file * SQUARE_SIZE
            y = (7 - rank) * SQUARE_SIZE
        
        return x, y
    
    def coords_to_square(self, x, y):
        """Convert screen coordinates to chess square"""
        if x >= BOARD_SIZE:
            return None
        
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        
        if self.flipped:
            file = 7 - file
            rank = 7 - rank
        
        return rank * 8 + file
    
    def draw_board(self):
        """Draw the chessboard"""
        for rank in range(8):
            for file in range(8):
                x = file * SQUARE_SIZE
                y = rank * SQUARE_SIZE
                
                color = WHITE if (rank + file) % 2 == 0 else BLACK
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        
        # Draw file/rank labels
        label_font = pygame.font.SysFont('Arial', 16)
        files = 'abcdefgh'
        ranks = '87654321'
        
        for i in range(8):
            # File labels (bottom)
            file_label = label_font.render(
                files[i] if not self.flipped else files[7-i], 
                True, (100, 100, 100)
            )
            self.screen.blit(file_label, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 18))
            
            # Rank labels (left)
            rank_label = label_font.render(
                ranks[i] if not self.flipped else ranks[7-i], 
                True, (100, 100, 100)
            )
            self.screen.blit(rank_label, (5, i * SQUARE_SIZE + 5))
        
        # Highlight last move
        if self.move_history:
            last_move = self.move_history[-1]
            for square in [last_move.from_square, last_move.to_square]:
                x, y = self.square_to_coords(square)
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                s.fill(HIGHLIGHT)
                self.screen.blit(s, (x, y))
        
        # Highlight selected square
        if self.selected_square is not None:
            x, y = self.square_to_coords(self.selected_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(SELECTED)
            self.screen.blit(s, (x, y))
        
        # Highlight legal moves
        for move in self.legal_moves:
            x, y = self.square_to_coords(move.to_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            
            if self.board.piece_at(move.to_square):
                # Capture - draw semi-transparent red overlay
                s.fill(CAPTURE_MOVE)
            else:
                # Normal move - draw circle
                pygame.draw.circle(s, LEGAL_MOVE, (SQUARE_SIZE//2, SQUARE_SIZE//2), 12)
            
            self.screen.blit(s, (x, y))
    
    def draw_pieces(self):
        """Draw chess pieces"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self.square_to_coords(square)
                piece_surface = self.pieces[piece.symbol()]
                self.screen.blit(piece_surface, (x, y))
    
    def draw_sidebar(self):
        """Draw sidebar with game info"""
        x_start = BOARD_SIZE
        
        # Background
        pygame.draw.rect(self.screen, SIDEBAR_BG, (x_start, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        
        y = 20
        
        # Title
        title = self.text_font.render("Chess AI v4.2", True, TEXT_COLOR)
        self.screen.blit(title, (x_start + 20, y))
        y += 50
        
        # Separator line
        pygame.draw.line(self.screen, (80, 80, 80), 
                        (x_start + 20, y), (x_start + SIDEBAR_WIDTH - 20, y), 2)
        y += 20
        
        # Game mode info
        if self.game_mode == "human_vs_ai":
            mode_text = "Human vs AI"
            you_text = f"You: {'White' if self.human_color == chess.WHITE else 'Black'}"
        elif self.game_mode == "ai_vs_ai":
            mode_text = "AI vs AI"
            you_text = "Spectator Mode"
        else:  # human_vs_human
            mode_text = "Human vs Human"
            you_text = "2 Player Mode"
        
        mode = self.small_font.render(mode_text, True, (150, 200, 255))
        self.screen.blit(mode, (x_start + 20, y))
        y += 25
        
        you = self.small_font.render(you_text, True, TEXT_COLOR)
        self.screen.blit(you, (x_start + 20, y))
        y += 35
        
        # Game info
        turn_text = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        
        if self.game_mode == "human_vs_ai":
            turn_color = (255, 255, 200) if self.board.turn == self.human_color else (200, 200, 255)
        else:
            turn_color = (255, 255, 200) if self.board.turn == chess.WHITE else (200, 200, 255)
        
        turn = self.text_font.render(turn_text, True, turn_color)
        self.screen.blit(turn, (x_start + 20, y))
        y += 40
        
        # 🆕 AI Mode indicator
        if self.game_mode == "human_vs_ai" and self.model1:
            if self.mcts_enabled:
                ai_mode_text = f"AI Mode: {'MCTS' if self.use_mcts else 'Network'}"
                ai_mode_color = (100, 255, 100) if self.use_mcts else (255, 200, 100)
            else:
                ai_mode_text = "AI Mode: Network Only"
                ai_mode_color = (255, 200, 100)
            
            ai_mode = self.small_font.render(ai_mode_text, True, ai_mode_color)
            self.screen.blit(ai_mode, (x_start + 20, y))
            y += 30
        
        # AI status
        if self.ai_thinking:
            status = self.text_font.render("🤔 AI thinking...", True, (255, 200, 0))
            self.screen.blit(status, (x_start + 20, y))
            
            # Animated dots
            dots = "." * (pygame.time.get_ticks() // 500 % 4)
            dots_text = self.text_font.render(dots, True, (255, 200, 0))
            self.screen.blit(dots_text, (x_start + 200, y))
        
        y += 50
        
        # Separator line
        pygame.draw.line(self.screen, (80, 80, 80), 
                        (x_start + 20, y), (x_start + SIDEBAR_WIDTH - 20, y), 2)
        y += 20
        
        # Move history
        history_title = self.text_font.render("Move History", True, TEXT_COLOR)
        self.screen.blit(history_title, (x_start + 20, y))
        y += 30
        
        for i, move in enumerate(self.move_history[-12:]):
            move_num = len(self.move_history) - 12 + i + 1
            if move_num > 0:
                move_text = f"{move_num}. {move.uci()}"
                color = (200, 200, 200) if i == len(self.move_history[-12:]) - 1 else (150, 150, 150)
                text = self.small_font.render(move_text, True, color)
                self.screen.blit(text, (x_start + 30, y))
                y += 22
        
        # Game over message
        if self.game_over:
            y = WINDOW_HEIGHT - 180
            
            # Box background
            box_rect = pygame.Rect(x_start + 20, y, SIDEBAR_WIDTH - 40, 120)
            pygame.draw.rect(self.screen, (60, 60, 60), box_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), box_rect, 2)
            
            y += 15
            result = self.board.result()
            
            if result == '1-0':
                msg = "White Wins!"
                color = (100, 255, 100)
            elif result == '0-1':
                msg = "Black Wins!"
                color = (100, 255, 100)
            else:
                msg = "Draw!"
                color = (200, 200, 200)
            
            game_over_text = self.text_font.render(msg, True, color)
            rect = game_over_text.get_rect(center=(x_start + SIDEBAR_WIDTH//2, y + 30))
            self.screen.blit(game_over_text, rect)
            
            restart_text = self.small_font.render("Press R to restart", True, (200, 200, 200))
            rect = restart_text.get_rect(center=(x_start + SIDEBAR_WIDTH//2, y + 70))
            self.screen.blit(restart_text, rect)
        
        # Controls (bottom)
        y = WINDOW_HEIGHT - 110
        controls_title = self.small_font.render("Controls:", True, (150, 150, 150))
        self.screen.blit(controls_title, (x_start + 20, y))
        y += 25
        
        controls = ["R - Restart game", "F - Flip board"]
        
        # 🆕 Show MCTS toggle only if enabled
        if self.game_mode == "human_vs_ai" and self.mcts_enabled:
            controls.append("M - Toggle MCTS")
        
        for control in controls:
            text = self.small_font.render(control, True, (120, 120, 120))
            self.screen.blit(text, (x_start + 20, y))
            y += 20
    
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
                    # 🆕 Save current board to history before making move
                    self.board_history.append(self.board.copy())
                    
                    # 🆕 v4.2: Update MCTS history too
                    if self.mcts1:
                        self.mcts1.update_history(self.board)
                    if self.mcts2:
                        self.mcts2.update_history(self.board)
                    
                    self.board.push(move)
                    self.move_history.append(move)
                    self.selected_square = None
                    self.legal_moves = []
                    
                    if self.board.is_game_over():
                        self.game_over = True
                    return
        
        # Select piece
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
        else:
            self.selected_square = None
            self.legal_moves = []
    
    def _build_history_tensor(self, current_board):
        """
        🆕 v4.2: Build tensor with history using POV-aware board_to_tensor
        
        The board_to_tensor function from data_helpers automatically handles:
        - POV (perspective from current player)
        - Flipping for black to move
        
        Args:
            current_board: chess.Board for current position
        
        Returns:
            numpy array: (input_planes, 8, 8) tensor
        """
        if self.history_positions == 0:
            # No history - just current board
            # board_to_tensor automatically handles POV
            return board_to_tensor(current_board)
        
        # Build history list
        history_boards = []
        
        # Get last N boards from history
        if self.board_history:
            history_boards = self.board_history[-self.history_positions:]
        
        # Pad with empty boards if not enough history
        while len(history_boards) < self.history_positions:
            history_boards.insert(0, chess.Board())  # Empty board at start
        
        # Convert all boards to tensors with POV
        # IMPORTANT: All boards should be from CURRENT player's perspective
        tensors = []
        
        # Add history boards (oldest to newest)
        for hist_board in history_boards:
            # board_to_tensor handles POV automatically based on current_board.turn
            hist_tensor = board_to_tensor(hist_board, flip_perspective=(current_board.turn == chess.BLACK))
            tensors.append(hist_tensor)
        
        # Add current board
        current_tensor = board_to_tensor(current_board)
        tensors.append(current_tensor)
        
        # Stack: [oldest_history, ..., newest_history, current]
        # Shape: (12 * (history_positions + 1), 8, 8)
        import numpy as np
        return np.concatenate(tensors, axis=0)
    
    def _get_network_move(self, model):
        """
        🆕 v4.2: Get move directly from network (no MCTS)
        
        Uses POV-aware board_to_tensor for correct history handling
        """
        # Build tensor with history and POV
        board_tensor = torch.FloatTensor(
            self._build_history_tensor(self.board)
        ).unsqueeze(0).to(self.device)
        
        # Get policy from model
        with torch.no_grad():
            policy_log_probs, _ = model(board_tensor, return_aux=False)
            policy = torch.exp(policy_log_probs).cpu().numpy()[0]
        
        # Find best legal move
        best_score = -1
        best_move = None
        for move in self.board.legal_moves:
            idx = move.from_square * 64 + move.to_square
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
        
        if not self.ai_thinking:
            self.ai_thinking = True
            return
        
        # Select the appropriate model
        if self.game_mode == "ai_vs_ai":
            current_model = self.model1 if self.board.turn == chess.WHITE else self.model2
            current_mcts = self.mcts1 if self.board.turn == chess.WHITE else self.mcts2
        else:  # human_vs_ai
            current_model = self.model1
            current_mcts = self.mcts1
        
        # 🆕 Get AI move - with MCTS or network-only
        if self.use_mcts and current_mcts is not None:
            # MCTS mode (if available)
            visit_counts = current_mcts.search(
                self.board, 
                self.config['reinforcement_learning']['mcts_simulations']
            )
            move, _ = select_move_by_visits(visit_counts, temperature=0)
        else:
            # 🆕 v4.2: Network-only mode with POV support
            move = self._get_network_move(current_model)
        
        if move:
            # 🆕 Save current board to history before making move
            self.board_history.append(self.board.copy())
            
            # 🆕 v4.2: Update MCTS history too
            if self.mcts1:
                self.mcts1.update_history(self.board)
            if self.mcts2:
                self.mcts2.update_history(self.board)
            
            self.board.push(move)
            self.move_history.append(move)
        
        self.ai_thinking = False
        
        if self.board.is_game_over():
            self.game_over = True
    
    def restart_game(self):
        """Restart the game"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        self.board_history = []  # 🆕 Clear board history
        
        # 🆕 v4.2: Reset MCTS trees and histories
        if self.mcts1:
            self.mcts1.reset_tree()
        if self.mcts2:
            self.mcts2.reset_tree()
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            self.clock.tick(FPS)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.restart_game()
                    elif event.key == pygame.K_f:
                        self.flipped = not self.flipped
                    elif event.key == pygame.K_m and self.game_mode == "human_vs_ai" and self.mcts_enabled:
                        self.use_mcts = not self.use_mcts
                        print(f"🔄 Toggled AI mode: {'MCTS' if self.use_mcts else 'Network-only'}")
            
            # AI move
            if not self.game_over and (self.game_mode == "ai_vs_ai" or 
                                        (self.game_mode == "human_vs_ai" and self.board.turn != self.human_color)):
                self.ai_move()
            
            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.draw_sidebar()
            
            pygame.display.flip()
        
        pygame.quit()


def main():
    # 🆕 Parse command-line arguments
    parser = argparse.ArgumentParser(description='Chess AI Game v4.2')
    parser.add_argument('--no-mcts', action='store_true', 
                       help='Disable MCTS (use network-only mode)')
    args = parser.parse_args()
    
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # 🆕 MCTS toggle
    enable_mcts = not args.no_mcts
    if not enable_mcts:
        print("⚠️  MCTS DISABLED - Using network-only mode")
    
    # 🆕 Show config info
    history_positions = config['model'].get('history_positions', 0)
    print(f"📜 History positions: {history_positions}")
    print(f"🔢 Input planes: {12 * (1 + history_positions)}")
    
    base_dir = script_dir.parent
    
    # Select models and game mode
    model1_path, model2_path, game_mode = select_models(base_dir, config)
    
    if game_mode in ["human_vs_ai", "ai_vs_ai"]:
        if model1_path is None:
            print("⚠️  No model selected. Exiting.")
            return
        
        # Load models
        print("\nLoading models...")
        model1 = load_model_from_checkpoint(model1_path, config, device, ChessNet)
        
        model2 = None
        if game_mode == "ai_vs_ai":
            if model2_path is None:
                print("⚠️  No second model selected. Exiting.")
                return
            model2 = load_model_from_checkpoint(model2_path, config, device, ChessNet)
    else:  # human_vs_human
        model1 = None
        model2 = None
    
    # Start GUI
    print("\n" + "="*50)
    print("Chess AI v4.2 - Pygame GUI")
    print("="*50)
    print("\n🆕 v4.2 Features:")
    print("  • POV (Point of View) - perspective handling")
    print("  • Dynamic Sliding Window - history support")
    print("  • MCTS toggle - network-only mode available")
    print("\nControls:")
    print("  • Click to select and move pieces")
    print("  • R - Restart game")
    print("  • F - Flip board")
    if game_mode == "human_vs_ai" and enable_mcts:
        print("  • M - Toggle MCTS on/off")
    print()
    
    # 🆕 Pass enable_mcts to GUI
    gui = ChessGUI(model1, model2, config, device, game_mode, enable_mcts=enable_mcts)
    
    # Choose color (only for human vs AI)
    if game_mode == "human_vs_ai":
        print("Choose your color:")
        print("  1. White (you start)")
        print("  2. Black (AI starts)")
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '2':
            gui.human_color = chess.BLACK
            gui.flipped = True
            print("\nYou are playing as Black!")
        else:
            print("\nYou are playing as White!")
    elif game_mode == "ai_vs_ai":
        print("\nWatching AI vs AI match...")
        print(f"White: {model1_path.name}")
        print(f"Black: {model2_path.name}")
    else:
        print("\n2-Player mode activated!")
    
    # 🆕 Display mode
    if enable_mcts:
        print(f"\n🎮 AI Mode: MCTS (press M to toggle)")
    else:
        print(f"\n🎮 AI Mode: Network-only (faster)")
    
    print("\nStarting game... Good luck! 🎮\n")
    
    gui.run()


if __name__ == "__main__":
    main()