import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import chess
from tqdm import tqdm
import numpy as np
from collections import deque
from pathlib import Path
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.mcts import MCTS, select_move_by_visits
from src.data import board_to_tensor, move_to_index


class ReplayBuffer:
    """Replay buffer for storing self-play games"""
    
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, position):
        """Add (board_tensor, policy_target, value_target)"""
        self.buffer.append(position)
    
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        boards = torch.stack([b for b, _, _ in batch])
        policies = torch.stack([p for _, p, _ in batch])
        values = torch.stack([v for _, _, v in batch])
        
        return boards, policies, values
    
    def __len__(self):
        return len(self.buffer)


def play_game(model, mcts, config, device):
    """
    Play one self-play game with MCTS
    Returns: list of (board_tensor, policy_target, outcome)
    """
    board = chess.Board()
    game_history = []
    
    move_count = 0
    temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
    
    while not board.is_game_over() and move_count < 200:
        # Run MCTS
        visit_counts = mcts.search(
            board, 
            config['reinforcement_learning']['mcts_simulations']
        )
        
        # Create policy target (normalized visit counts)
        policy_target = torch.zeros(4096)
        total_visits = sum(visit_counts.values())
        for move, visits in visit_counts.items():
            idx = move_to_index(move)
            policy_target[idx] = visits / total_visits
        
        # Select move with temperature
        temperature = config['reinforcement_learning']['mcts_temperature'] if move_count < temp_threshold else 0.1
        move, _ = select_move_by_visits(visit_counts, temperature)
        
        # Store position
        board_tensor = torch.FloatTensor(board_to_tensor(board))
        game_history.append((board_tensor, policy_target, board.turn))
        
        # Make move
        board.push(move)
        move_count += 1
    
    # Get game outcome
    result = board.result()
    if result == '1-0':
        outcome = 1.0
    elif result == '0-1':
        outcome = -1.0
    else:
        outcome = 0.0
    
    # Add outcomes to history
    training_data = []
    for board_tensor, policy_target, turn in game_history:
        # Value from perspective of player to move
        value = outcome if turn == chess.WHITE else -outcome
        training_data.append((board_tensor, policy_target, torch.FloatTensor([value])))
    
    return training_data


def train_on_batch(model, optimizer, batch, config, device):
    """Train model on one batch"""
    boards, policy_targets, value_targets = batch
    boards = boards.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    policy_pred, value_pred = model(boards)
    
    # Compute losses
    # Policy loss: cross-entropy with MCTS-improved policy
    policy_loss = -(policy_targets * policy_pred).sum(dim=1).mean()
    
    # Value loss: MSE with game outcome
    value_loss = nn.MSELoss()(value_pred, value_targets)
    
    # Combined loss
    policy_weight = config['reinforcement_learning']['policy_loss_weight']
    value_weight = config['reinforcement_learning']['value_loss_weight']
    loss = policy_weight * policy_loss + value_weight * value_loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_models(model1, model2, config, device, num_games=20):
    """
    Play games between two models
    Returns: win rate of model1
    """
    mcts1 = MCTS(model1, config, device)
    mcts2 = MCTS(model2, config, device)
    
    wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        board = chess.Board()
        
        # Alternate who plays white
        if game_idx % 2 == 0:
            current_mcts = mcts1
            other_mcts = mcts2
        else:
            current_mcts = mcts2
            other_mcts = mcts1
        
        move_count = 0
        while not board.is_game_over() and move_count < 200:
            # Select MCTS based on turn
            mcts = current_mcts if board.turn == chess.WHITE else other_mcts
            
            # Run MCTS with fewer simulations for speed
            visit_counts = mcts.search(board, num_simulations=50)
            move, _ = select_move_by_visits(visit_counts, temperature=0)
            
            board.push(move)
            move_count += 1
        
        # Count result
        result = board.result()
        if game_idx % 2 == 0:
            # model1 is white
            if result == '1-0':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
        else:
            # model1 is black
            if result == '0-1':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
    
    return (wins + draws) / num_games


def main():
    # Load config (relative to script location)
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Get base directory
    base_dir = script_dir.parent
    
    # Load IL-trained model
    print("\n=== Loading IL model ===")
    model = ChessNet(config).to(device)
    
    best_model_path = base_dir / config['paths']['best_model']
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {best_model_path}")
    else:
        print("Warning: No IL model found. Starting from scratch.")
    
    # Best model for evaluation
    best_model = ChessNet(config).to(device)
    best_model.load_state_dict(model.state_dict())
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['reinforcement_learning']['learning_rate']
    )
    
    # MCTS
    mcts = MCTS(model, config, device)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(config['reinforcement_learning']['replay_buffer_size'])
    
    # Training loop
    print("\n=== Starting RL training ===")
    
    for iteration in range(config['reinforcement_learning']['iterations']):
        print(f"\n=== Iteration {iteration + 1}/{config['reinforcement_learning']['iterations']} ===")
        
        # Self-play
        print("Generating self-play games...")
        for game_idx in tqdm(range(config['reinforcement_learning']['games_per_iteration'])):
            game_data = play_game(model, mcts, config, device)
            for position in game_data:
                replay_buffer.add(position)
            
            # Periodic cleanup
            if game_idx % 10 == 0:
                gc.collect()
        
        print(f"Replay buffer size: {len(replay_buffer)}")
        
        # Training
        if len(replay_buffer) >= config['reinforcement_learning']['batch_size']:
            print("Training on self-play data...")
            
            total_loss = 0
            total_policy = 0
            total_value = 0
            
            num_batches = len(replay_buffer) // config['reinforcement_learning']['batch_size']
            
            for batch_idx in tqdm(range(config['reinforcement_learning']['train_epochs_per_iteration'] * num_batches)):
                batch = replay_buffer.sample(config['reinforcement_learning']['batch_size'])
                loss, policy_loss, value_loss = train_on_batch(model, optimizer, batch, config, device)
                
                total_loss += loss
                total_policy += policy_loss
                total_value += value_loss
                
                # Periodic cleanup
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            avg_loss = total_loss / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            avg_policy = total_policy / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            avg_value = total_value / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            
            print(f"Training - Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
        
        # Evaluation
        if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
            print("Evaluating against best model...")
            win_rate = evaluate_models(
                model, best_model, config, device,
                config['reinforcement_learning']['eval_games']
            )
            print(f"Win rate vs best: {win_rate:.2%}")
            
            # Update best model if new model is better
            if win_rate >= config['reinforcement_learning']['win_rate_threshold']:
                print("✓ New best model!")
                best_model.load_state_dict(model.state_dict())
                
                best_model_path = base_dir / config['paths']['best_model']
                save_checkpoint(
                    model, optimizer, iteration, avg_loss,
                    str(best_model_path),
                    {'win_rate': win_rate}
                )
        
        # Save checkpoint
        checkpoint_path = base_dir / config['paths']['rl_checkpoint']
        save_checkpoint(
            model, optimizer, iteration, avg_loss if 'avg_loss' in locals() else 0,
            str(checkpoint_path)
        )
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n=== RL training complete ===")


if __name__ == "__main__":
    main()