import torch
import chess
import yaml
import sys
from pathlib import Path
import pygame
import os

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet
from src.mcts import MCTS, select_move_by_visits
from src.data import board_to_tensor

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


def create_piece_surfaces():
    """Create colored surfaces with piece letters"""
    pieces = {}
    font = pygame.font.SysFont('Arial', 56, bold=True)
    
    piece_chars = {
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'P', 'n': 'N', 'b': 'B', 'r': 'R', 'q': 'Q', 'k': 'K'
    }
    
    for piece_symbol, char in piece_chars.items():
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        
        # Determine color
        if piece_symbol.isupper():
            # White pieces - white with black outline
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)
        else:
            # Black pieces - black with white outline
            text_color = (0, 0, 0)
            outline_color = (255, 255, 255)
        
        # Draw outline
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 or dy != 0:
                    outline_text = font.render(char, True, outline_color)
                    outline_rect = outline_text.get_rect(center=(SQUARE_SIZE//2 + dx, SQUARE_SIZE//2 + dy))
                    surface.blit(outline_text, outline_rect)
        
        # Draw main text
        text = font.render(char, True, text_color)
        text_rect = text.get_rect(center=(SQUARE_SIZE//2, SQUARE_SIZE//2))
        surface.blit(text, text_rect)
        
        pieces[piece_symbol] = surface
    
    return pieces


class ChessGUI:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.mcts = MCTS(model, config, device)
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.clock = pygame.time.Clock()
        
        # Load piece images
        self.pieces = create_piece_surfaces()
        
        # Fonts
        self.text_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.human_color = chess.WHITE
        self.use_mcts = True
        self.ai_thinking = False
        self.game_over = False
        self.move_history = []
        
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
            file_label = label_font.render(files[i] if not self.flipped else files[7-i], True, (100, 100, 100))
            self.screen.blit(file_label, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 18))
            
            # Rank labels (left)
            rank_label = label_font.render(ranks[i] if not self.flipped else ranks[7-i], True, (100, 100, 100))
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
        title = self.text_font.render("Chess AI", True, TEXT_COLOR)
        self.screen.blit(title, (x_start + 20, y))
        y += 50
        
        # Separator line
        pygame.draw.line(self.screen, (80, 80, 80), (x_start + 20, y), (x_start + SIDEBAR_WIDTH - 20, y), 2)
        y += 20
        
        # Game info
        turn_text = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        turn_color = (255, 255, 200) if self.board.turn == self.human_color else (200, 200, 255)
        turn = self.text_font.render(turn_text, True, turn_color)
        self.screen.blit(turn, (x_start + 20, y))
        y += 40
        
        you_text = f"You: {'White' if self.human_color == chess.WHITE else 'Black'}"
        you = self.small_font.render(you_text, True, TEXT_COLOR)
        self.screen.blit(you, (x_start + 20, y))
        y += 30
        
        mode = "MCTS" if self.use_mcts else "Network"
        ai_mode = self.small_font.render(f"AI Mode: {mode}", True, TEXT_COLOR)
        self.screen.blit(ai_mode, (x_start + 20, y))
        y += 40
        
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
        pygame.draw.line(self.screen, (80, 80, 80), (x_start + 20, y), (x_start + SIDEBAR_WIDTH - 20, y), 2)
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
        y = WINDOW_HEIGHT - 90
        controls_title = self.small_font.render("Controls:", True, (150, 150, 150))
        self.screen.blit(controls_title, (x_start + 20, y))
        y += 25
        
        controls = [
            "R - Restart game",
            "F - Flip board",
            "M - Toggle MCTS"
        ]
        for control in controls:
            text = self.small_font.render(control, True, (120, 120, 120))
            self.screen.blit(text, (x_start + 20, y))
            y += 20
    
    def handle_click(self, pos):
        """Handle mouse click"""
        if self.game_over or self.ai_thinking:
            return
        
        if self.board.turn != self.human_color:
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
    
    def ai_move(self):
        """Make AI move"""
        if self.board.turn == self.human_color or self.game_over:
            return
        
        if not self.ai_thinking:
            self.ai_thinking = True
            return
        
        # Get AI move
        if self.use_mcts:
            visit_counts = self.mcts.search(
                self.board, 
                self.config['reinforcement_learning']['mcts_simulations']
            )
            move, _ = select_move_by_visits(visit_counts, temperature=0)
        else:
            board_tensor = torch.FloatTensor(board_to_tensor(self.board)).unsqueeze(0).to(self.device)
            policy, _ = self.model.predict(board_tensor)
            
            best_score = -1
            best_move = None
            for move in self.board.legal_moves:
                idx = move.from_square * 64 + move.to_square
                if policy[idx] > best_score:
                    best_score = policy[idx]
                    best_move = move
            move = best_move
        
        if move:
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
                    elif event.key == pygame.K_m:
                        self.use_mcts = not self.use_mcts
            
            # AI move
            if not self.game_over and self.board.turn != self.human_color:
                self.ai_move()
            
            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.draw_sidebar()
            
            pygame.display.flip()
        
        pygame.quit()


def main():
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = ChessNet(config).to(device)
    
    base_dir = script_dir.parent
    model_path = base_dir / config['paths']['best_model']
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")
        print("Using untrained model!")
    
    model.eval()
    
    # Start GUI
    print("\n" + "="*50)
    print("Chess AI - Pygame GUI")
    print("="*50)
    print("\nControls:")
    print("  • Click to select and move pieces")
    print("  • R - Restart game")
    print("  • F - Flip board")
    print("  • M - Toggle MCTS on/off")
    print()
    
    gui = ChessGUI(model, config, device)
    
    # Choose color
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
    
    print("\nStarting game... Good luck! 🎮\n")
    
    gui.run()


if __name__ == "__main__":
    main()