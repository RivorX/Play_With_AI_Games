import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
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
import time
import pickle
import tempfile

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.mcts import BatchMCTS, select_move_by_visits
from src.data import board_to_tensor, move_to_index

# Import MCTS self-play
try:
    from src.batch_selfplay import play_games_mcts_worker
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("⚠️ MCTS self-play not available")


# ==============================================================================
# 🆕 PRIORITIZED REPLAY BUFFER
# ==============================================================================

class PrioritizedReplayBuffer:
    """
    🆕 Prioritized Experience Replay
    
    Samples positions based on TD error priority:
    - High error positions → sampled more often
    - Low error positions → sampled less often
    
    Benefits:
    - Faster learning from "difficult" positions
    - Better sample efficiency
    - Improved convergence
    """
    
    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_end=1.0, epsilon=0.01):
        """
        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta_start: Initial importance sampling correction
            beta_end: Final beta value
            epsilon: Small constant to avoid zero priority
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        print(f"🎯 Prioritized Replay Buffer:")
        print(f"   Alpha (priority): {alpha}")
        print(f"   Beta (IS correction): {beta_start} → {beta_end}")
        print(f"   Epsilon: {epsilon}")
    
    def add(self, position, priority=None):
        """Add position with optional initial priority"""
        if priority is None:
            # New positions get max priority (will be sampled quickly)
            priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(position)
        else:
            self.buffer[self.position] = position
        
        self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, beta=None):
        """
        Sample batch with prioritized sampling
        
        Returns:
            batch: List of positions
            indices: Indices of sampled positions
            weights: Importance sampling weights
        """
        if beta is None:
            beta = self.beta
        
        # Get priorities for all positions
        priorities = self.priorities[:self.size]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Get batch
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled positions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def update_beta(self, progress):
        """Update beta (importance sampling correction) based on training progress"""
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def __len__(self):
        return self.size


# ==============================================================================
# 🆕 TEMPERATURE SCHEDULE (Curriculum Learning)
# ==============================================================================

class TemperatureSchedule:
    """
    🆕 Adaptive temperature schedule for exploration
    
    Strategy:
    - Start with HIGH temperature → more exploration
    - Gradually DECREASE → more exploitation
    - Helps model learn diverse strategies early, then refine
    
    Benefits:
    - Better exploration-exploitation balance
    - Faster convergence
    - More robust policies
    """
    
    def __init__(self, start_temp, end_temp, decay_iterations):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.decay_iterations = decay_iterations
        
        print(f"🌡️ Temperature Schedule:")
        print(f"   Start: {start_temp} (high exploration)")
        print(f"   End: {end_temp} (low exploration)")
        print(f"   Decay over: {decay_iterations} iterations")
    
    def get_temperature(self, iteration):
        """Get temperature for current iteration"""
        if iteration >= self.decay_iterations:
            return self.end_temp
        
        # Linear decay
        progress = iteration / self.decay_iterations
        temp = self.start_temp + (self.end_temp - self.start_temp) * progress
        
        return temp


# ==============================================================================
# STANDARD REPLAY BUFFER (Fallback)
# ==============================================================================

class ReplayBuffer:
    """Standard replay buffer (uniform sampling)"""
    
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, position):
        self.buffer.append(position)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        boards = torch.stack([b for b, _, _ in batch])
        policies = torch.stack([p for _, p, _ in batch])
        values = torch.stack([v for _, _, v in batch])
        
        return boards, policies, values
    
    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# TRAINING LOGGER
# ==============================================================================

class TrainingLogger:
    """Logger for RL training"""
    
    def __init__(self, log_dir, experiment_name="rl_training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_{timestamp}.png"
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'avg_loss', 'policy_loss', 'value_loss',
                'win_rate', 'buffer_size', 'avg_game_length', 'positions_per_sec',
                'selfplay_time', 'data_collection_time', 'temperature', 'beta'
            ])
        
        self.iterations = []
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.win_rates = []
        self.temperatures = []
        
        print(f"📊 Logging to: {self.csv_path}")
    
    def log(self, iteration, avg_loss, policy_loss, value_loss, 
            win_rate=None, buffer_size=None, avg_game_length=None, 
            positions_per_sec=None, selfplay_time=None, data_collection_time=None,
            temperature=None, beta=None):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, avg_loss, policy_loss, value_loss,
                win_rate if win_rate is not None else '',
                buffer_size if buffer_size is not None else '',
                avg_game_length if avg_game_length is not None else '',
                positions_per_sec if positions_per_sec is not None else '',
                selfplay_time if selfplay_time is not None else '',
                data_collection_time if data_collection_time is not None else '',
                temperature if temperature is not None else '',
                beta if beta is not None else ''
            ])
        
        self.iterations.append(iteration)
        self.losses.append(avg_loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        
        if win_rate is not None:
            self.win_rates.append((iteration, win_rate))
        
        if temperature is not None:
            self.temperatures.append((iteration, temperature))
    
    def plot(self):
        if len(self.iterations) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')
        
        # Total Loss
        ax = axes[0, 0]
        ax.plot(self.iterations, self.losses, 'b-', label='Total Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Policy Loss
        ax = axes[0, 1]
        ax.plot(self.iterations, self.policy_losses, 'g-', label='Policy Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[0, 2]
        ax.plot(self.iterations, self.value_losses, 'r-', label='Value Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Win Rate
        ax = axes[1, 0]
        if self.win_rates:
            win_iters, win_vals = zip(*self.win_rates)
            ax.plot(win_iters, win_vals, 'mo-', label='Win Rate', linewidth=2, markersize=8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate vs Best')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 🆕 Temperature Schedule
        ax = axes[1, 1]
        if self.temperatures:
            temp_iters, temp_vals = zip(*self.temperatures)
            ax.plot(temp_iters, temp_vals, 'orange', linewidth=2, label='Temperature')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Temperature')
            ax.set_title('🌡️ Temperature Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Loss Comparison
        ax = axes[1, 2]
        ax.plot(self.iterations, self.losses, 'b-', label='Total', linewidth=2, alpha=0.7)
        ax.plot(self.iterations, self.policy_losses, 'g--', label='Policy', linewidth=1.5, alpha=0.7)
        ax.plot(self.iterations, self.value_losses, 'r--', label='Value', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('All Losses Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()


# ==============================================================================
# 🎯 SELF-PLAY WITH PROPER MCTS
# ==============================================================================

def play_games_parallel_mcts(model, config, device, num_games):
    """
    ✅ CORRECT: Parallel self-play using MCTS
    
    This is the PROPER AlphaZero approach:
    - Each worker plays games using MCTS
    - Training targets = MCTS visit distributions
    - High quality training data
    """
    start_time = time.time()
    
    if not MCTS_SELFPLAY_AVAILABLE:
        print("❌ MCTS self-play not available!")
        return [], 0, 0, 0, 0
    
    model_state = model.state_dict()
    
    # Parallel configuration
    num_workers = config['reinforcement_learning'].get('self_play_workers', 4)
    
    if not torch.cuda.is_available():
        num_workers = min(num_workers, mp.cpu_count())
        device_type = 'cpu'
    else:
        num_gpus = torch.cuda.device_count()
        device_type = 'cuda'
    
    games_per_worker = num_games // num_workers
    
    print(f"🎯 Parallel MCTS Self-Play:")
    if torch.cuda.is_available():
        print(f"   GPUs available: {torch.cuda.device_count()}")
    print(f"   Workers: {num_workers}")
    print(f"   Games per worker: {games_per_worker}")
    print(f"   Total games: {num_games}")
    print(f"   MCTS simulations: {config['reinforcement_learning']['mcts_simulations']}")
    
    # Create temp directory for results
    temp_dir = Path(tempfile.gettempdir()) / "chess_selfplay_mcts"
    temp_dir.mkdir(exist_ok=True)
    
    # Prepare worker tasks
    mp_ctx = mp.get_context('spawn')
    processes = []
    result_files = []
    
    for rank in range(num_workers):
        if device_type == 'cuda':
            device_id = rank % torch.cuda.device_count()
        else:
            device_id = 'cpu'
        
        result_file = temp_dir / f"worker_{rank}_mcts_results.pkl"
        result_files.append(result_file)
        
        p = mp_ctx.Process(
            target=play_games_mcts_worker,
            args=(rank, model_state, config, device_id, games_per_worker, str(result_file))
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers
    for p in tqdm(processes, desc="MCTS Self-play workers"):
        p.join()
    
    selfplay_time = time.time() - start_time
    
    # Collect results
    collection_start = time.time()
    
    all_positions = []
    game_lengths = []
    
    for rank, result_file in enumerate(result_files):
        if result_file.exists():
            try:
                with open(result_file, 'rb') as f:
                    positions, lengths = pickle.load(f)
                    all_positions.extend(positions)
                    game_lengths.extend(lengths)
                
                # Clean up
                result_file.unlink()
            except Exception as e:
                print(f"⚠️ Warning: Failed to load results from worker {rank}: {e}")
        else:
            print(f"⚠️ Warning: Worker {rank} result file not found")
    
    collection_time = time.time() - collection_start
    total_time = time.time() - start_time
    
    positions_per_sec = len(all_positions) / total_time if total_time > 0 else 0
    avg_length = np.mean(game_lengths) if game_lengths else 0
    
    print(f"✅ MCTS Self-play completed:")
    print(f"   Positions: {len(all_positions)}")
    print(f"   Games: {len(game_lengths)}")
    print(f"   Self-play time: {selfplay_time:.1f}s")
    print(f"   Data collection: {collection_time:.3f}s")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Speed: {positions_per_sec:.1f} positions/s")
    print(f"   Avg game length: {avg_length:.1f} moves")
    
    return all_positions, avg_length, positions_per_sec, selfplay_time, collection_time


# ==============================================================================
# 🆕 TRAINING FUNCTION WITH PRIORITIZED REPLAY
# ==============================================================================

def train_on_batch_prioritized(model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer):
    """
    🆕 Train on batch with prioritized replay
    
    Returns TD errors for priority update
    """
    boards, policy_targets, value_targets = batch
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_targets = policy_targets.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    weights = torch.FloatTensor(weights).to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards)
        
        # Policy loss
        policy_loss = -(policy_targets * policy_pred).sum(dim=1)
        
        # Value loss (TD error for prioritization)
        value_loss = (value_pred.squeeze() - value_targets.squeeze()) ** 2
        
        # 🆕 Apply importance sampling weights
        policy_loss = (policy_loss * weights).mean()
        value_loss = (value_loss * weights).mean()
        
        policy_weight = config['reinforcement_learning']['policy_loss_weight']
        value_weight = config['reinforcement_learning']['value_loss_weight']
        loss = policy_weight * policy_loss + value_weight * value_loss
    
    scaler.scale(loss).backward()
    
    grad_clip = config['reinforcement_learning'].get('grad_clip', 1.0)
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    # 🆕 Compute TD errors for priority update (detached, on CPU)
    with torch.no_grad():
        td_errors = torch.abs(value_pred.squeeze() - value_targets.squeeze()).cpu().numpy()
    
    # 🆕 Update priorities
    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        replay_buffer.update_priorities(indices, td_errors)
    
    return loss.item(), policy_loss.item(), value_loss.item()


def train_on_batch(model, optimizer, batch, config, device, scaler):
    """Standard training (for regular replay buffer)"""
    boards, policy_targets, value_targets = batch
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_targets = policy_targets.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards)
        
        policy_loss = -(policy_targets * policy_pred).sum(dim=1).mean()
        value_loss = nn.MSELoss()(value_pred, value_targets)
        
        policy_weight = config['reinforcement_learning']['policy_loss_weight']
        value_weight = config['reinforcement_learning']['value_loss_weight']
        loss = policy_weight * policy_loss + value_weight * value_loss
    
    scaler.scale(loss).backward()
    
    grad_clip = config['reinforcement_learning'].get('grad_clip', 1.0)
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), policy_loss.item(), value_loss.item()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_models(model1, model2, config, device, num_games=20):
    """Evaluate model1 vs model2"""
    mcts1 = BatchMCTS(model1, config, device)
    mcts2 = BatchMCTS(model2, config, device)
    
    wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        board = chess.Board()
        
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
        
        mcts1.reset_tree()
        mcts2.reset_tree()
        
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


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    rl_dir = base_dir / config['paths']['rl_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    use_amp = config['hardware'].get('use_amp', True)
    
    if use_bfloat16 and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("⚠️ bfloat16 not supported")
            use_bfloat16 = False
    
    logger = TrainingLogger(logs_dir, experiment_name="rl_training_mcts")
    
    print("\n=== Loading IL model ===")
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    best_model_il_path = base_dir / config['paths']['best_model_il']
    if best_model_il_path.exists():
        checkpoint = torch.load(best_model_il_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded IL model from {best_model_il_path}")
    else:
        print("⚠️ No IL model found")
    
    best_model = ChessNet(config).to(device)
    best_model = best_model.to(memory_format=torch.channels_last)
    best_model.load_state_dict(model.state_dict())
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['reinforcement_learning']['learning_rate'],
        weight_decay=config['reinforcement_learning'].get('weight_decay', 0.01),
        fused=True if torch.cuda.is_available() else False
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 🆕 Initialize replay buffer (prioritized or standard)
    use_prioritized = config['reinforcement_learning'].get('use_prioritized_replay', False)
    
    if use_prioritized:
        replay_buffer = PrioritizedReplayBuffer(
            max_size=config['reinforcement_learning']['replay_buffer_size'],
            alpha=config['reinforcement_learning'].get('priority_alpha', 0.6),
            beta_start=config['reinforcement_learning'].get('priority_beta_start', 0.4),
            beta_end=config['reinforcement_learning'].get('priority_beta_end', 1.0),
            epsilon=config['reinforcement_learning'].get('priority_epsilon', 0.01)
        )
    else:
        replay_buffer = ReplayBuffer(config['reinforcement_learning']['replay_buffer_size'])
    
    # 🆕 Initialize temperature schedule
    use_temp_schedule = config['reinforcement_learning'].get('use_temperature_schedule', False)
    
    if use_temp_schedule:
        temp_schedule = TemperatureSchedule(
            start_temp=config['reinforcement_learning'].get('temperature_start', 1.5),
            end_temp=config['reinforcement_learning'].get('temperature_end', 0.5),
            decay_iterations=config['reinforcement_learning'].get('temperature_decay_iterations', 500)
        )
    
    best_model_rl_path = base_dir / config['paths']['best_model_rl']
    checkpoint_every = config['reinforcement_learning'].get('checkpoint_every', 10)
    
    print("\n=== Starting RL training with PROPER MCTS ===")
    print("🎯 OPTIMIZATIONS:")
    print(f"   • MCTS self-play (AlphaZero approach)")
    print(f"   • MCTS simulations: {config['reinforcement_learning']['mcts_simulations']}")
    print(f"   • Tree reuse: {config['reinforcement_learning'].get('mcts_reuse_tree', True)}")
    print(f"   • Batch MCTS: {config['reinforcement_learning'].get('mcts_batch_size', 32)}")
    print(f"   • 🆕 Prioritized Replay: {use_prioritized}")
    print(f"   • 🆕 Temperature Schedule: {use_temp_schedule}")
    
    total_iterations = config['reinforcement_learning']['iterations']
    
    for iteration in range(total_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{total_iterations}")
        print('='*70)
        
        # 🆕 Get current temperature
        if use_temp_schedule:
            current_temp = temp_schedule.get_temperature(iteration)
            print(f"🌡️ Temperature: {current_temp:.2f}")
            # Update config for self-play
            config['reinforcement_learning']['mcts_temperature'] = current_temp
        else:
            current_temp = config['reinforcement_learning']['mcts_temperature']
        
        # 🆕 Update beta for importance sampling
        if use_prioritized:
            progress = iteration / total_iterations
            replay_buffer.update_beta(progress)
            print(f"🎯 Beta (IS): {replay_buffer.beta:.3f}")
        
        # 🎯 Self-play with MCTS
        model.eval()
        
        positions, avg_game_length, positions_per_sec, selfplay_time, collection_time = \
            play_games_parallel_mcts(
                model,
                config,
                device,
                config['reinforcement_learning']['games_per_iteration']
            )
        
        # Add to replay buffer
        for position in positions:
            replay_buffer.add(position)
        
        print(f"Replay buffer: {len(replay_buffer)} positions")
        
        # Training
        if len(replay_buffer) >= config['reinforcement_learning']['batch_size']:
            print("Training...")
            model.train()
            total_loss = 0
            total_policy = 0
            total_value = 0
            
            num_batches = len(replay_buffer) // config['reinforcement_learning']['batch_size']
            
            for batch_idx in tqdm(range(config['reinforcement_learning']['train_epochs_per_iteration'] * num_batches), desc="Training"):
                # 🆕 Prioritized sampling
                if use_prioritized:
                    batch_data, indices, weights = replay_buffer.sample(
                        config['reinforcement_learning']['batch_size']
                    )
                    
                    # Convert to tensors
                    boards = torch.stack([b for b, _, _ in batch_data])
                    policies = torch.stack([p for _, p, _ in batch_data])
                    values = torch.stack([v for _, _, v in batch_data])
                    batch = (boards, policies, values)
                    
                    loss, policy_loss, value_loss = train_on_batch_prioritized(
                        model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer
                    )
                else:
                    # Standard sampling
                    batch = replay_buffer.sample(config['reinforcement_learning']['batch_size'])
                    loss, policy_loss, value_loss = train_on_batch(
                        model, optimizer, batch, config, device, scaler
                    )
                
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
            
            print(f"Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
        else:
            avg_loss = avg_policy = avg_value = 0
        
        # Evaluation
        win_rate = None
        if (iteration + 1) % config['reinforcement_learning']['eval_every'] == 0:
            print("Evaluating vs best...")
            model.eval()
            win_rate = evaluate_models(
                model, best_model, config, device,
                config['reinforcement_learning']['eval_games']
            )
            print(f"Win rate: {win_rate:.2%}")
            
            # Log with temperature and beta
            logger.log(
                iteration + 1, avg_loss, avg_policy, avg_value,
                win_rate, len(replay_buffer), avg_game_length, positions_per_sec,
                selfplay_time, collection_time,
                temperature=current_temp,
                beta=replay_buffer.beta if use_prioritized else None
            )
            logger.plot()
            
            if win_rate >= config['reinforcement_learning']['win_rate_threshold']:
                print("✅ New best model!")
                best_model.load_state_dict(model.state_dict())
                
                model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
                save_checkpoint(
                    model_to_save, None, iteration, avg_loss,
                    str(best_model_rl_path),
                    {'win_rate': win_rate},
                    save_optimizer=False
                )
                
                if use_bfloat16:
                    model = model.to(torch.float32)
                
                size_mb = best_model_rl_path.stat().st_size / (1024**2)
                print(f"💾 Saved: {best_model_rl_path} ({size_mb:.1f} MB)")
        else:
            logger.log(
                iteration + 1, avg_loss, avg_policy, avg_value,
                buffer_size=len(replay_buffer), avg_game_length=avg_game_length,
                positions_per_sec=positions_per_sec, selfplay_time=selfplay_time,
                data_collection_time=collection_time,
                temperature=current_temp,
                beta=replay_buffer.beta if use_prioritized else None
            )
        
        # Checkpoints
        if (iteration + 1) % checkpoint_every == 0:
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] != 0:
                win_rate = evaluate_models(model, best_model, config, device,
                                        config['reinforcement_learning']['eval_games'])
            
            checkpoint_name = f"rl_iter_{iteration+1}_wr_{win_rate:.3f}.pt"
            checkpoint_path = rl_dir / checkpoint_name
            
            model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
            save_checkpoint(
                model_to_save, None, iteration, avg_loss,
                str(checkpoint_path),
                {'win_rate': win_rate},
                save_optimizer=False
            )
            
            if use_bfloat16:
                model = model.to(torch.float32)
            
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"💾 Checkpoint: {checkpoint_path} ({size_mb:.1f} MB)")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.plot()
    print("\n=== Training complete ===")