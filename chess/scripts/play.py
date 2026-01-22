import torch
import chess
import chess.svg
import yaml
import os
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet
from src.mcts import MCTS, select_move_by_visits
from src.data import board_to_tensor


def display_board(board):
    """Display board in terminal"""
    print("\n" + str(board) + "\n")


def get_human_move(board):
    """Get move from human player"""
    while True:
        try:
            move_str = input("Your move (e.g., e2e4): ").strip()
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except:
            print("Invalid format! Use format like 'e2e4'")


def get_ai_move(board, model, mcts, device, use_mcts=True, simulations=100):
    """Get move from AI"""
    if use_mcts:
        print(f"AI thinking (MCTS with {simulations} simulations)...")
        visit_counts = mcts.search(board, simulations)
        move, _ = select_move_by_visits(visit_counts, temperature=0)
    else:
        print("AI thinking (network only)...")
        board_tensor = torch.FloatTensor(board_to_tensor(board)).unsqueeze(0).to(device)
        policy, _ = model.predict(board_tensor)
        
        # Get best legal move
        best_score = -1
        best_move = None
        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            if policy[idx] > best_score:
                best_score = policy[idx]
                best_move = move
        move = best_move
    
    return move


def play_game(model, config, device, human_color=chess.WHITE, use_mcts=True):
    """Play a game against the AI"""
    board = chess.Board()
    mcts = MCTS(model, config, device)
    
    print("\n" + "="*50)
    print("Chess Game Started!")
    print(f"You are playing as: {'White' if human_color == chess.WHITE else 'Black'}")
    print(f"AI uses: {'MCTS' if use_mcts else 'Network only'}")
    print("="*50)
    
    display_board(board)
    
    while not board.is_game_over():
        if board.turn == human_color:
            # Human's turn
            print(f"Your turn ({'White' if human_color == chess.WHITE else 'Black'})")
            move = get_human_move(board)
        else:
            # AI's turn
            move = get_ai_move(board, model, mcts, device, use_mcts, 
                             simulations=config['reinforcement_learning']['mcts_simulations'])
            print(f"AI plays: {move}")
        
        board.push(move)
        display_board(board)
    
    # Game over
    print("\n" + "="*50)
    print("Game Over!")
    result = board.result()
    print(f"Result: {result}")
    
    if result == '1-0':
        winner = "White wins!"
    elif result == '0-1':
        winner = "Black wins!"
    else:
        winner = "Draw!"
    
    print(winner)
    print("="*50)


def main():
    # Load config (relative to script location)
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
    
    # Game settings
    print("\n" + "="*50)
    print("Game Settings")
    print("="*50)
    
    color_input = input("Play as White or Black? (w/b): ").strip().lower()
    human_color = chess.WHITE if color_input == 'w' else chess.BLACK
    
    mcts_input = input("Use MCTS? (y/n): ").strip().lower()
    use_mcts = mcts_input == 'y'
    
    # Play game
    play_game(model, config, device, human_color, use_mcts)
    
    # Play again?
    while True:
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again == 'y':
            play_game(model, config, device, human_color, use_mcts)
        else:
            break
    
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()