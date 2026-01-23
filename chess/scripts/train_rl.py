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
import csv
import matplotlib.pyplot as plt
from datetime import datetime

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


class TrainingLogger:
    """Logger for RL training metrics with CSV and plotting"""
    
    def __init__(self, log_dir, experiment_name="rl_training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_{timestamp}.png"
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'avg_loss', 'policy_loss', 'value_loss',
                'win_rate', 'buffer_size'
            ])
        
        # Store metrics for plotting
        self.iterations = []
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.win_rates = []
        
        print(f"📊 Logging to: {self.csv_path}")
    
    def log(self, iteration, avg_loss, policy_loss, value_loss, 
            win_rate=None, buffer_size=None):
        """Log metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, avg_loss, policy_loss, value_loss,
                win_rate if win_rate is not None else '',
                buffer_size if buffer_size is not None else ''
            ])
        
        # Store for plotting
        self.iterations.append(iteration)
        self.losses.append(avg_loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        
        if win_rate is not None:
            self.win_rates.append((iteration, win_rate))
    
    def plot(self):
        """Generate training plots"""
        if len(self.iterations) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        ax.plot(self.iterations, self.losses, 'b-', label='Total Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Policy Loss
        ax = axes[0, 1]
        ax.plot(self.iterations, self.policy_losses, 'g-', label='Policy Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Value Loss
        ax = axes[1, 0]
        ax.plot(self.iterations, self.value_losses, 'r-', label='Value Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Win Rate
        ax = axes[1, 1]
        if self.win_rates:
            win_iters, win_vals = zip(*self.win_rates)
            ax.plot(win_iters, win_vals, 'mo-', label='Win Rate vs Best', linewidth=2, markersize=8)
            ax.axhline(y=0.5, color='gray', linestyle='--', label='50% baseline', alpha=0.5)
            ax.axhline(y=0.55, color='green', linestyle='--', label='55% threshold', alpha=0.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate vs Best Model')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")


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
        
        # Create policy target
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
    policy_loss = -(policy_targets * policy_pred).sum(dim=1).mean()
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
            mcts = current_mcts if board.turn == chess.WHITE else other_mcts
            visit_counts = mcts.search(board, num_simulations=50)
            move, _ = select_move_by_visits(visit_counts, temperature=0)
            board.push(move)
            move_count += 1
        
        # Count result
        result = board.result()
        if game_idx % 2 == 0:
            if result == '1-0':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
        else:
            if result == '0-1':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
    
    return (wins + draws) / num_games


def main():
    # Load config
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
    
    # Get directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    rl_dir = base_dir / config['paths']['rl_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    
    # Check bfloat16 support
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    if use_bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("⚠️  bfloat16 not supported, using float32")
            use_bfloat16 = False
        else:
            print("✓ bfloat16 enabled for model storage")
    
    # Initialize logger
    logger = TrainingLogger(logs_dir, experiment_name="rl_training")
    
    # Load IL-trained model
    print("\n=== Loading IL model ===")
    model = ChessNet(config).to(device)
    
    # Load best IL model
    best_model_il_path = base_dir / config['paths']['best_model_il']
    if best_model_il_path.exists():
        checkpoint = torch.load(best_model_il_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded IL model from {best_model_il_path}")
    else:
        print("⚠️  No IL model found. Starting from scratch.")
    
    # Best model for evaluation
    best_model = ChessNet(config).to(device)
    best_model.load_state_dict(model.state_dict())
    
    # Optimizer - AdamW with weight decay (NOWY optimizer dla RL!)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['reinforcement_learning']['learning_rate'],
        weight_decay=config['reinforcement_learning'].get('weight_decay', 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # MCTS
    mcts = MCTS(model, config, device)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(config['reinforcement_learning']['replay_buffer_size'])
    
    # Best model path
    best_model_rl_path = base_dir / config['paths']['best_model_rl']
    checkpoint_every = config['reinforcement_learning'].get('checkpoint_every', 5)
    
    # Training loop
    print("\n=== Starting RL training ===")
    print("💾 Saving strategy: NO optimizer state (minimal file size)")
    print(f"Checkpoints saved every {checkpoint_every} iterations to: {rl_dir}")
    print(f"Best model saved to: {best_model_rl_path}")
    
    for iteration in range(config['reinforcement_learning']['iterations']):
        print(f"\n=== Iteration {iteration + 1}/{config['reinforcement_learning']['iterations']} ===")
        
        # Self-play
        print("Generating self-play games...")
        for game_idx in tqdm(range(config['reinforcement_learning']['games_per_iteration'])):
            game_data = play_game(model, mcts, config, device)
            for position in game_data:
                replay_buffer.add(position)
            
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
                
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            avg_loss = total_loss / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            avg_policy = total_policy / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            avg_value = total_value / (num_batches * config['reinforcement_learning']['train_epochs_per_iteration'])
            
            print(f"Training - Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
        else:
            avg_loss = 0
            avg_policy = 0
            avg_value = 0
        
        # Evaluation
        win_rate = None
        if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
            print("Evaluating against best model...")
            win_rate = evaluate_models(
                model, best_model, config, device,
                config['reinforcement_learning']['eval_games']
            )
            print(f"Win rate vs best: {win_rate:.2%}")
            
            # Log with win rate
            logger.log(iteration + 1, avg_loss, avg_policy, avg_value, 
                      win_rate, len(replay_buffer))
            logger.plot()
            
            # Update best model if better (overwrite previous)
            if win_rate >= config['reinforcement_learning']['win_rate_threshold']:
                print("✓ New best model!")
                best_model.load_state_dict(model.state_dict())
                
                model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
                save_checkpoint(
                    model_to_save, 
                    None,  # ← BEZ optimizer
                    iteration, 
                    avg_loss,
                    str(best_model_rl_path),
                    {'win_rate': win_rate},
                    save_optimizer=False
                )
                
                # Restore dtype
                if use_bfloat16:
                    model = model.to(torch.float32)
                
                print(f"  💾 Saved to: {best_model_rl_path}")
                size_mb = best_model_rl_path.stat().st_size / (1024**2)
                print(f"  📦 Model size: {size_mb:.1f} MB (no optimizer)")
        else:
            # Log without win rate
            logger.log(iteration + 1, avg_loss, avg_policy, avg_value, 
                      buffer_size=len(replay_buffer))
        
        # Save checkpoint every N iterations (ALSO without optimizer)
        if (iteration + 1) % checkpoint_every == 0:
            # Evaluate for checkpoint if not done this iteration
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] != 0:
                win_rate = evaluate_models(
                    model, best_model, config, device,
                    config['reinforcement_learning']['eval_games']
                )
                print(f"Checkpoint eval - Win rate vs best: {win_rate:.2%}")
            
            checkpoint_name = f"rl_iteration_{iteration+1}_winrate_{win_rate:.3f}.pt"
            checkpoint_path = rl_dir / checkpoint_name
            
            model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
            save_checkpoint(
                model_to_save, 
                None,  # ← BEZ optimizer (również w checkpointach!)
                iteration, 
                avg_loss if 'avg_loss' in locals() else 0,
                str(checkpoint_path),
                {'win_rate': win_rate},
                save_optimizer=False  # ← BEZ optimizer
            )
            
            # Restore dtype
            if use_bfloat16:
                model = model.to(torch.float32)
            
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"💾 Checkpoint saved: {checkpoint_path} ({size_mb:.1f} MB)")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final plot
    logger.plot()
    
    print("\n=== RL training complete ===")
    print(f"Best model: {best_model_rl_path}")
    print(f"Checkpoints: {rl_dir}")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()