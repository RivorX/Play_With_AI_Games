"""
Training and evaluation functions with comprehensive metrics and profiling
🆕 UPDATED: Added epoch profiling and GPU memory tracking
"""

import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import gc
import time

from .loss import LabelSmoothingNLLLoss
from .replay import PrioritizedReplayBuffer
from .metrics import MetricsCalculator, compute_batch_metrics

import sys
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.mcts import BatchMCTS, select_move_by_visits


# ==============================================================================
# 🆕 PROFILING UTILITIES
# ==============================================================================

class EpochProfiler:
    """Profile time breakdown per epoch and save to file"""
    
    def __init__(self, enabled=True, log_file=None):
        self.enabled = enabled
        self.log_file = log_file
        self.reset()
    
    def reset(self):
        self.times = {
            'data_loading': 0.0,
            'data_transfer': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optimizer': 0.0,
            'metrics': 0.0,
        }
        self.start_time = None
    
    def start(self):
        if self.enabled:
            self.start_time = time.perf_counter()
    
    def record(self, key):
        if self.enabled and self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.times[key] += elapsed
            self.start_time = None
    
    def print_and_save_summary(self, epoch):
        """Print summary and save to file"""
        if not self.enabled:
            return
        
        total = sum(self.times.values())
        if total == 0:
            return
        
        labels = {
            'data_loading': '📊 Data Loading',
            'data_transfer': '🔄 Data Transfer',
            'forward': '➡️  Forward Pass',
            'backward': '⬅️  Backward Pass',
            'optimizer': '🔧 Optimizer Step',
            'metrics': '📈 Metrics',
        }
        
        # Build output
        lines = []
        lines.append("=" * 70)
        lines.append(f"⏱️  EPOCH {epoch} TIME BREAKDOWN")
        lines.append("=" * 70)
        
        for key, label in labels.items():
            t = self.times[key]
            pct = 100 * t / total if total > 0 else 0
            lines.append(f"{label:20s} {t:7.2f}s ({pct:5.1f}%)")
        
        lines.append("=" * 70)
        lines.append(f"TOTAL:              {total:7.2f}s")
        lines.append("")
        
        output = "\n".join(lines)
        
        # Print to console
        print("\n" + output)
        
        # Save to file
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(output + "\n")
            except Exception as e:
                print(f"⚠️ Failed to write profiling log: {e}")


def log_gpu_memory(log_file=None):
    """Log current GPU memory usage to console and file"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        msg = (f"🎮 GPU Memory: {allocated:.2f}GB allocated, "
               f"{reserved:.2f}GB reserved, "
               f"{max_allocated:.2f}GB peak\n")
        
        print(msg, end='')
        
        # Save to file
        if log_file:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg)
            except Exception as e:
                print(f"⚠️ Failed to write memory log: {e}")


# ==============================================================================
# IMITATION LEARNING FUNCTIONS
# ==============================================================================

def train_epoch_il(model, train_loader, optimizer, scheduler, config, device, scaler, epoch=0, debug_log_file=None):
    """
    🆕 UPDATED: Train for one epoch with profiling support
    
    Args:
        debug_log_file: Path to debug log file (optional)
    
    Returns:
        Tuple of (losses_dict, metrics_dict)
    """
    model.train()
    
    # Debug config
    debug_enabled = config.get('debug', {}).get('enabled', False)
    profile_training = config.get('debug', {}).get('profile_training', False) and debug_enabled
    log_memory = config.get('debug', {}).get('log_gpu_memory', False) and debug_enabled
    profile_every = config.get('debug', {}).get('profile_every_n_epochs', 1)
    
    # Only profile if it's the right epoch
    should_profile = profile_training and (epoch % profile_every == 0)
    profiler = EpochProfiler(enabled=should_profile, log_file=debug_log_file)
    
    # Loss weights
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    # MTL weights
    use_mtl = config['model'].get('use_multitask_learning', False)
    win_weight = config['model'].get('win_prediction_weight', 0.3) if use_mtl else 0
    material_weight = config['model'].get('material_prediction_weight', 0.2) if use_mtl else 0
    check_weight = config['model'].get('check_prediction_weight', 0.15) if use_mtl else 0
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    criterion_value = nn.MSELoss()
    
    if use_mtl:
        criterion_win = nn.BCEWithLogitsLoss()
        criterion_material = nn.MSELoss()
        criterion_check = nn.BCEWithLogitsLoss()
    
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch_data in enumerate(pbar):
        # ⏱️ Profile: Data loading
        profiler.start()
        
        # Unpack batch
        if isinstance(batch_data, dict):
            boards = batch_data['board']
            moves = batch_data['move']
            outcomes = batch_data['value']
            
            if use_mtl:
                win_targets = batch_data['win']
                material_targets = batch_data['material']
                check_targets = batch_data['check']
        else:
            boards, moves, outcomes = batch_data
            use_mtl = False
        
        profiler.record('data_loading')
        
        # ⏱️ Profile: Data transfer
        profiler.start()
        
        # Move to device
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        outcomes = outcomes.to(device, non_blocking=True)
        
        if use_mtl:
            win_targets = win_targets.to(device, non_blocking=True)
            material_targets = material_targets.to(device, non_blocking=True)
            check_targets = check_targets.to(device, non_blocking=True)
        
        profiler.record('data_transfer')
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Training
        use_amp = config['hardware'].get('use_amp', True)
        amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
        
        # ⏱️ Profile: Forward pass
        profiler.start()
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            if use_mtl:
                policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
            else:
                policy_pred, value_pred = model(boards, return_aux=False)
            
            # Main losses
            policy_loss = criterion_policy(policy_pred, moves)
            value_loss = criterion_value(value_pred, outcomes)
            
            # Total loss
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Add auxiliary losses
            if use_mtl:
                win_loss = criterion_win(win_pred.squeeze(), win_targets.squeeze())
                material_loss = criterion_material(material_pred, material_targets)
                check_loss = criterion_check(check_pred.squeeze(), check_targets.squeeze())
                
                loss = loss + win_weight * win_loss + material_weight * material_loss + check_weight * check_loss
                
                total_win_loss += win_loss.item()
                total_material_loss += material_loss.item()
                total_check_loss += check_loss.item()
        
        profiler.record('forward')
        
        # ⏱️ Profile: Backward pass
        profiler.start()
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['imitation_learning']['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['imitation_learning']['grad_clip']
            )
        
        profiler.record('backward')
        
        # ⏱️ Profile: Optimizer step
        profiler.start()
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        profiler.record('optimizer')
        
        # Track losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        # ⏱️ Profile: Metrics
        profiler.start()
        
        # Update metrics
        with torch.no_grad():
            metrics_calc.update(policy_pred, value_pred, moves, outcomes)
        
        profiler.record('metrics')
        
        # Progress bar
        if batch_idx % config['logging']['print_every'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute current batch metrics for display
            batch_metrics = compute_batch_metrics(policy_pred.detach(), value_pred.detach(), 
                                                 moves, outcomes)
            
            postfix = {
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}",
                'top1': f"{batch_metrics['policy_top1_acc']:.2%}",
                'lr': f"{current_lr:.2e}"
            }
            
            if use_mtl:
                postfix['win'] = f"{win_loss.item():.3f}"
                postfix['mat'] = f"{material_loss.item():.3f}"
                postfix['chk'] = f"{check_loss.item():.3f}"
            
            pbar.set_postfix(postfix)
        
        # Periodic garbage collection
        if batch_idx % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print profiling summary
    if should_profile:
        profiler.print_and_save_summary(epoch + 1)
    
    # Log GPU memory
    if log_memory:
        log_gpu_memory(log_file=debug_log_file)
    
    n = len(train_loader)
    
    # Return losses
    losses = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n
    }
    
    if use_mtl:
        losses.update({
            'win': total_win_loss / n,
            'material': total_material_loss / n,
            'check': total_check_loss / n
        })
    
    # Compute final metrics
    metrics = metrics_calc.compute()
    
    return losses, metrics


def evaluate_il(model, val_loader, config, device):
    """Evaluate model (unchanged)"""
    model.eval()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    # MTL weights
    use_mtl = config['model'].get('use_multitask_learning', False)
    win_weight = config['model'].get('win_prediction_weight', 0.3) if use_mtl else 0
    material_weight = config['model'].get('material_prediction_weight', 0.2) if use_mtl else 0
    check_weight = config['model'].get('check_prediction_weight', 0.15) if use_mtl else 0
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    criterion_value = nn.MSELoss()
    
    if use_mtl:
        criterion_win = nn.BCEWithLogitsLoss()
        criterion_material = nn.MSELoss()
        criterion_check = nn.BCEWithLogitsLoss()
    
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            if isinstance(batch_data, dict):
                boards = batch_data['board']
                moves = batch_data['move']
                outcomes = batch_data['value']
                
                if use_mtl:
                    win_targets = batch_data['win']
                    material_targets = batch_data['material']
                    check_targets = batch_data['check']
            else:
                boards, moves, outcomes = batch_data
                use_mtl = False
            
            boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)
            
            if use_mtl:
                win_targets = win_targets.to(device, non_blocking=True)
                material_targets = material_targets.to(device, non_blocking=True)
                check_targets = check_targets.to(device, non_blocking=True)
            
            use_amp = config['hardware'].get('use_amp', True)
            amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if use_mtl:
                    policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
                else:
                    policy_pred, value_pred = model(boards, return_aux=False)
                
                policy_loss = criterion_policy(policy_pred, moves)
                value_loss = criterion_value(value_pred, outcomes)
                
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                if use_mtl:
                    win_loss = criterion_win(win_pred.squeeze(), win_targets.squeeze())
                    material_loss = criterion_material(material_pred, material_targets)
                    check_loss = criterion_check(check_pred.squeeze(), check_targets.squeeze())
                    
                    loss = loss + win_weight * win_loss + material_weight * material_loss + check_weight * check_loss
                    
                    total_win_loss += win_loss.item()
                    total_material_loss += material_loss.item()
                    total_check_loss += check_loss.item()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            metrics_calc.update(policy_pred, value_pred, moves, outcomes)
    
    n = len(val_loader)
    
    losses = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n
    }
    
    if use_mtl:
        losses.update({
            'win': total_win_loss / n,
            'material': total_material_loss / n,
            'check': total_check_loss / n
        })
    
    metrics = metrics_calc.compute()
    
    return losses, metrics


# ==============================================================================
# REINFORCEMENT LEARNING FUNCTIONS (unchanged)
# ==============================================================================

def train_on_batch_rl(model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer, metrics_calc=None):
    """Train on batch with optional prioritized replay and metrics"""
    boards, policy_targets, value_targets = batch
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_targets = policy_targets.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards, return_aux=False)
        
        policy_loss = -(policy_targets * policy_pred).sum(dim=1)
        value_loss = (value_pred.squeeze() - value_targets.squeeze()) ** 2
        
        if weights is not None:
            policy_loss = (policy_loss * weights).mean()
            value_loss = (value_loss * weights).mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss = value_loss.mean()
        
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
    
    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        with torch.no_grad():
            td_errors = torch.abs(value_pred.squeeze() - value_targets.squeeze()).cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)
    
    if metrics_calc is not None:
        with torch.no_grad():
            target_moves = policy_targets.argmax(dim=1)
            metrics_calc.update(policy_pred, value_pred, target_moves, value_targets.unsqueeze(1))
    
    return loss.item(), policy_loss.item(), value_loss.item()


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