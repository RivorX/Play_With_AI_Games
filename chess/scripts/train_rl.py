"""
Reinforcement Learning Training Script with Comprehensive Metrics

NEW Features:
- 📊 Policy Accuracy tracking during training
- 📊 Value MAE monitoring
- 📊 Enhanced self-play statistics
"""

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
import time
import pickle
import tempfile

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint

# Import MCTS self-play
try:
    from src.batch_selfplay import play_games_mcts_worker
    MCTS_SELFPLAY_AVAILABLE = True
except ImportError:
    MCTS_SELFPLAY_AVAILABLE = False
    print("⚠️ MCTS self-play not available")

# Import from utils
from utils import (
    TrainingLogger,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    TemperatureSchedule,
    train_on_batch_rl,
    evaluate_models
)
from utils.metrics import MetricsCalculator


# ==============================================================================
# SELF-PLAY WITH PROPER MCTS
# ==============================================================================

def play_games_parallel_mcts(model, config, device, num_games):
    """
    Parallel self-play using MCTS
    
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
    
    # Initialize logger
    logger = TrainingLogger(
        logs_dir, 
        experiment_name="rl_training_mcts",
        mode="rl"
    )
    
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
    
    # Initialize replay buffer (prioritized or standard)
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
    
    # Initialize temperature schedule
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
    print(f"   • 📊 Policy Accuracy & Value MAE tracking")
    
    total_iterations = config['reinforcement_learning']['iterations']
    
    for iteration in range(total_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{total_iterations}")
        print('='*70)
        
        # Get current temperature
        if use_temp_schedule:
            current_temp = temp_schedule.get_temperature(iteration)
            print(f"🌡️ Temperature: {current_temp:.2f}")
            config['reinforcement_learning']['mcts_temperature'] = current_temp
        else:
            current_temp = config['reinforcement_learning']['mcts_temperature']
        
        # Update beta for importance sampling
        if use_prioritized:
            progress = iteration / total_iterations
            replay_buffer.update_beta(progress)
            print(f"🎯 Beta (IS): {replay_buffer.beta:.3f}")
        
        # Self-play with MCTS
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
        
        # Training with metrics
        if len(replay_buffer) >= config['reinforcement_learning']['batch_size']:
            print("Training...")
            model.train()
            total_loss = 0
            total_policy = 0
            total_value = 0
            
            # 📊 Initialize metrics calculator
            metrics_calc = MetricsCalculator()
            
            num_batches = len(replay_buffer) // config['reinforcement_learning']['batch_size']
            
            for batch_idx in tqdm(range(config['reinforcement_learning']['train_epochs_per_iteration'] * num_batches), desc="Training"):
                # Prioritized or standard sampling
                if use_prioritized:
                    batch_data, indices, weights = replay_buffer.sample(
                        config['reinforcement_learning']['batch_size']
                    )
                    
                    # Convert to tensors
                    boards = torch.stack([b for b, _, _ in batch_data])
                    policies = torch.stack([p for _, p, _ in batch_data])
                    values = torch.stack([v for _, _, v in batch_data])
                    batch = (boards, policies, values)
                    
                    loss, policy_loss, value_loss = train_on_batch_rl(
                        model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer, metrics_calc
                    )
                else:
                    batch = replay_buffer.sample(config['reinforcement_learning']['batch_size'])
                    loss, policy_loss, value_loss = train_on_batch_rl(
                        model, optimizer, batch, None, None, config, device, scaler, replay_buffer, metrics_calc
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
            
            # 📊 Compute metrics
            train_metrics = metrics_calc.compute()
            
            print(f"Loss: {avg_loss:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}")
            print(f"📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
                  f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
                  f"MAE: {train_metrics['value_mae']:.4f}")
        else:
            avg_loss = avg_policy = avg_value = 0
            train_metrics = {}
        
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
            
            # Log with all metrics
            logger.log(
                iteration + 1,
                train_metrics=train_metrics,
                avg_loss=avg_loss,
                policy_loss=avg_policy,
                value_loss=avg_value,
                win_rate=win_rate,
                buffer_size=len(replay_buffer),
                avg_game_length=avg_game_length,
                positions_per_sec=positions_per_sec,
                selfplay_time=selfplay_time,
                data_collection_time=collection_time,
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
                    {
                        'win_rate': win_rate,
                        'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                        'value_mae': train_metrics.get('value_mae', 0)
                    },
                    save_optimizer=False
                )
                
                if use_bfloat16:
                    model = model.to(torch.float32)
                
                size_mb = best_model_rl_path.stat().st_size / (1024**2)
                print(f"💾 Saved: {best_model_rl_path} ({size_mb:.1f} MB)")
        else:
            logger.log(
                iteration + 1,
                train_metrics=train_metrics,
                avg_loss=avg_loss,
                policy_loss=avg_policy,
                value_loss=avg_value,
                buffer_size=len(replay_buffer),
                avg_game_length=avg_game_length,
                positions_per_sec=positions_per_sec,
                selfplay_time=selfplay_time,
                data_collection_time=collection_time,
                temperature=current_temp,
                beta=replay_buffer.beta if use_prioritized else None
            )
        
        # Checkpoints
        if (iteration + 1) % checkpoint_every == 0:
            if (iteration + 1) % config['reinforcement_learning']['eval_every'] != 0:
                win_rate = evaluate_models(model, best_model, config, device,
                                        config['reinforcement_learning']['eval_games'])
            
            checkpoint_name = f"rl_iter_{iteration+1}_wr_{win_rate:.3f}_top1_{train_metrics.get('policy_top1_acc', 0):.3f}.pt"
            checkpoint_path = rl_dir / checkpoint_name
            
            model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
            save_checkpoint(
                model_to_save, None, iteration, avg_loss,
                str(checkpoint_path),
                {
                    'win_rate': win_rate,
                    'policy_top1_acc': train_metrics.get('policy_top1_acc', 0),
                    'value_mae': train_metrics.get('value_mae', 0)
                },
                save_optimizer=False
            )
            
            if use_bfloat16:
                model = model.to(torch.float32)
            
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"💾 Checkpoint: {checkpoint_path.name} ({size_mb:.1f} MB)")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.plot()
    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()