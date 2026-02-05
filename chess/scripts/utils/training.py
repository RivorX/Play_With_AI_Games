"""
Training and evaluation functions with WDL and move-weighted losses
🆕 v4.3 UPDATED: Integrated WDL classification and move-weighted training

CRITICAL CHANGES:
1. Value head now outputs (B, 3) WDL logits instead of (B, 1) scalar
2. Win prediction uses move-weighted BCE loss
3. CombinedLoss class simplifies loss computation
"""

import torch
import torch.nn as nn
import chess
from tqdm import tqdm
import gc
import time

from .loss import LabelSmoothingNLLLoss, CombinedLoss, WDLLoss, MoveWeightedBCELoss
from .replay import PrioritizedReplayBuffer
from .metrics import MetricsCalculator, compute_batch_metrics

import sys
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.mcts import BatchMCTS, select_move_by_visits


# ==============================================================================
# 🆕 v4.3 TRAINING WITH WDL + MOVE WEIGHTING
# ==============================================================================

def train_epoch_il(model, train_loader, optimizer, scheduler, config, device, scaler, epoch=0, debug_log_file=None):
    """
    🆕 v4.3: Train one epoch with WDL value head and move-weighted losses
    
    Key Changes:
    - Value predictions are now (B, 3) WDL logits
    - Win predictions use move-weighted BCE
    - Simplified loss computation via CombinedLoss
    
    Args:
        model: ChessNet model
        train_loader: DataLoader with training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dict
        device: torch.device
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        debug_log_file: Path to debug log (optional)
    
    Returns:
        Tuple of (losses_dict, metrics_dict)
    """
    model.train()
    
    # Check if using WDL value head
    use_wdl = config['model'].get('use_wdl_value', True)
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # 🆕 v4.3: Use CombinedLoss for all loss computation
    if use_wdl:
        criterion = CombinedLoss(config)
    else:
        # Legacy mode: separate losses
        policy_weight = config['imitation_learning']['policy_loss_weight']
        value_weight = config['imitation_learning']['value_loss_weight']
        label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
        
        criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
        criterion_value = nn.MSELoss()
        
        if use_mtl:
            win_weight = config['model'].get('win_prediction_weight', 0.3)
            material_weight = config['model'].get('material_prediction_weight', 0.2)
            check_weight = config['model'].get('check_prediction_weight', 0.15)
            criterion_win = nn.BCEWithLogitsLoss()
            criterion_material = nn.MSELoss()
            criterion_check = nn.BCEWithLogitsLoss()
    
    # Accumulators
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    # 🔍 DIAGNOSTIC: Track target distributions (gated by config)
    first_batch_targets = True
    first_batch_predictions = True
    show_batch0_diagnostics = (
        config.get('debug', {}).get('enabled', False) and
        config.get('debug', {}).get('print_batch0_diagnostics', False)
    )
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch
        if isinstance(batch_data, dict):
            boards = batch_data['board']
            moves = batch_data['move']
            outcomes = batch_data['value']
            
            # 🔧 v4.4 FIX: Get move_idx (not move_indices) from dict
            move_indices = batch_data.get('move_idx', None)
            total_moves = batch_data.get('total_moves', None)
            
            if use_mtl:
                win_targets = batch_data['win']
                material_targets = batch_data['material']
                check_targets = batch_data['check']
        else:
            # 🔧 v4.4 FIX: Unpack move_indices from tuple
            if len(batch_data) == 5:
                boards, moves, outcomes, move_indices, total_moves = batch_data
            elif len(batch_data) == 4:
                boards, moves, outcomes, move_indices = batch_data
                total_moves = None
            else:
                boards, moves, outcomes = batch_data
                move_indices = None
                total_moves = None
            use_mtl = False
        
        # Move to device
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        outcomes = outcomes.to(device, non_blocking=True)
        
        if move_indices is not None:
            move_indices = move_indices.to(device, non_blocking=True)
        if total_moves is not None:
            total_moves = total_moves.to(device, non_blocking=True)
        
        if use_mtl:
            win_targets = win_targets.to(device, non_blocking=True)
            material_targets = material_targets.to(device, non_blocking=True)
            check_targets = check_targets.to(device, non_blocking=True)
        
        # 🔍 DIAGNOSTIC: Print distributions for first batch
        if show_batch0_diagnostics and first_batch_targets:
            print(f"\n🔍 DIAGNOSTIC - Batch 0:")
            print(f"  Outcome targets (Value):")
            print(f"    Min: {outcomes.min().item():.3f}, Max: {outcomes.max().item():.3f}, Mean: {outcomes.mean().item():.3f}")
            print(f"    Unique values: {torch.unique(outcomes).cpu().numpy()[:10]}")  # First 10 unique
            print(f"    Distribution: +1: {(outcomes > 0.9).sum().item()}, 0: {(outcomes.abs() < 0.1).sum().item()}, -1: {(outcomes < -0.9).sum().item()}")
            
            if use_mtl:
                print(f"  Win targets (MTL):")
                print(f"    Min: {win_targets.min().item():.3f}, Max: {win_targets.max().item():.3f}, Mean: {win_targets.mean().item():.3f}")
                print(f"    Distribution: 1.0: {(win_targets > 0.9).sum().item()}, 0.0: {(win_targets < 0.1).sum().item()}")
            
            first_batch_targets = False
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision
        use_amp = config['hardware'].get('use_amp', True)
        amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            if use_mtl:
                policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
            else:
                policy_pred, value_pred = model(boards, return_aux=False)
            
            # 🆕 v4.3: Compute loss using CombinedLoss
            if use_wdl:
                # Pack predictions and targets for CombinedLoss
                predictions = {
                    'policy': policy_pred,
                    'value': value_pred,  # 🆕 Now (B, 3) WDL logits!
                }
                
                targets = {
                    'moves': moves,
                    'values': outcomes,  # Still scalar {-1, 0, +1}
                }
                if move_indices is not None:
                    targets['move_indices'] = move_indices
                if total_moves is not None:
                    targets['total_moves'] = total_moves
                                # 🔍 DIAGNOSTIC: Print WDL predictions for first batch
                if show_batch0_diagnostics and first_batch_predictions and batch_idx == 0:
                    wdl_probs = torch.softmax(value_pred[:10], dim=1)
                    value_scalars = (wdl_probs[:, 0] * 1.0 + 
                                    wdl_probs[:, 1] * 0.0 + 
                                    wdl_probs[:, 2] * (-1.0))
                    print(f"\n  🔍 WDL Predictions (first 10 samples):")
                    print(f"    Value pred shape: {value_pred.shape}")
                    print(f"    Sample logits: {value_pred[:3].float().cpu().detach().numpy()}")
                    print(f"    WDL Probs (Win/Draw/Loss) -> Scalar vs Target:")
                    for i in range(10):
                        print(f"      [{i}] W:{wdl_probs[i,0]:.3f} D:{wdl_probs[i,1]:.3f} L:{wdl_probs[i,2]:.3f} "
                              f"-> Scalar:{value_scalars[i]:.3f} | Target: {outcomes[i].item():+.3f}")
                    print(f"    Mean predicted scalar: {value_scalars.mean().item():.3f}")
                    print(f"    Mean target: {outcomes[:10].mean().item():.3f}\n")
                    first_batch_predictions = False
                
                if use_mtl:
                    predictions.update({
                        'win': win_pred,
                        'material': material_pred,
                        'check': check_pred,
                    })
                    
                    targets.update({
                        'win': win_targets,
                        'material': material_targets,
                        'check': check_targets,
                    })
                    if move_indices is not None:
                        targets['move_indices'] = move_indices  # 🆕 For move weighting
                    if total_moves is not None:
                        targets['total_moves'] = total_moves
                
                # Single loss computation
                loss, loss_dict = criterion(predictions, targets)
                
                # Extract individual losses for logging
                policy_loss = loss_dict['policy']
                value_loss = loss_dict['value']
                
                if use_mtl:
                    win_loss = loss_dict['win']
                    material_loss = loss_dict['material']
                    check_loss = loss_dict['check']
                
            else:
                # Legacy mode
                policy_loss = criterion_policy(policy_pred, moves)
                value_loss = criterion_value(value_pred, outcomes)
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                if use_mtl:
                    win_loss = criterion_win(win_pred.squeeze(), win_targets.squeeze())
                    material_loss = criterion_material(material_pred, material_targets)
                    check_loss = criterion_check(check_pred.squeeze(), check_targets.squeeze())
                    
                    loss = loss + (win_weight * win_loss + 
                                  material_weight * material_loss + 
                                  check_weight * check_loss)
                
                # Convert to items for logging
                policy_loss = policy_loss.item()
                value_loss = value_loss.item()
                
                if use_mtl:
                    win_loss = win_loss.item()
                    material_loss = material_loss.item()
                    check_loss = check_loss.item()
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        grad_clip = config['imitation_learning'].get('grad_clip', 1.0)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        # Note: metrics_calc.update() auto-handles WDL predictions
        metrics_calc.update(
            policy_pred,
            value_pred,
            moves,
            outcomes,
            move_indices=move_indices,
            total_moves=total_moves,
            value_weight_min=config['imitation_learning'].get('value_move_weight_min', 0.1),
            value_weight_min_total_moves=config['imitation_learning'].get('value_move_weight_min_total_moves', 40),
            value_max_moves=config['data'].get('max_moves_per_game', 200),
            value_use_game_length=config['imitation_learning'].get('value_move_weight_use_game_length', False),
        )
        
        # Accumulate losses
        if use_wdl:
            total_loss += loss_dict['total']
            total_policy_loss += loss_dict['policy']
            total_value_loss += loss_dict['value']
            
            if use_mtl:
                total_win_loss += loss_dict['win']
                total_material_loss += loss_dict['material']
                total_check_loss += loss_dict['check']
        else:
            total_loss += loss.item()
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            
            if use_mtl:
                total_win_loss += win_loss
                total_material_loss += material_loss
                total_check_loss += check_loss
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'policy': f'{total_policy_loss / (batch_idx + 1):.4f}',
            'value': f'{total_value_loss / (batch_idx + 1):.4f}',
        })
    
    # Step scheduler (per-batch for OneCycleLR/CosineAnnealing, NOT for ReduceLROnPlateau)
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    # Compute final metrics
    n = len(train_loader)
    
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


def evaluate_il(model, val_loader, config, device):
    """
    🆕 v4.3: Evaluate model with WDL value head
    
    Identical to train_epoch_il but without gradient updates
    """
    model.eval()
    
    use_wdl = config['model'].get('use_wdl_value', True)
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # Setup criterion
    if use_wdl:
        criterion = CombinedLoss(config)
    else:
        policy_weight = config['imitation_learning']['policy_loss_weight']
        value_weight = config['imitation_learning']['value_loss_weight']
        label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
        
        criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
        criterion_value = nn.MSELoss()
        
        if use_mtl:
            win_weight = config['model'].get('win_prediction_weight', 0.3)
            material_weight = config['model'].get('material_prediction_weight', 0.2)
            check_weight = config['model'].get('check_prediction_weight', 0.15)
            criterion_win = nn.BCEWithLogitsLoss()
            criterion_material = nn.MSELoss()
            criterion_check = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            if isinstance(batch_data, dict):
                boards = batch_data['board']
                moves = batch_data['move']
                outcomes = batch_data['value']
                move_indices = batch_data.get('move_idx', None)  # 🔧 v4.4 FIX: move_idx not move_indices
                total_moves = batch_data.get('total_moves', None)
                
                if use_mtl:
                    win_targets = batch_data['win']
                    material_targets = batch_data['material']
                    check_targets = batch_data['check']
            else:
                # 🔧 v4.4 FIX: Unpack move_indices from tuple
                if len(batch_data) == 5:
                    boards, moves, outcomes, move_indices, total_moves = batch_data
                elif len(batch_data) == 4:
                    boards, moves, outcomes, move_indices = batch_data
                    total_moves = None
                else:
                    boards, moves, outcomes = batch_data
                    move_indices = None
                    total_moves = None
                use_mtl = False
            
            boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)
            
            if move_indices is not None:
                move_indices = move_indices.to(device, non_blocking=True)
            if total_moves is not None:
                total_moves = total_moves.to(device, non_blocking=True)
            
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
                
                if use_wdl:
                    predictions = {
                        'policy': policy_pred,
                        'value': value_pred,
                    }
                    
                    targets = {
                        'moves': moves,
                        'values': outcomes,
                    }
                    if move_indices is not None:
                        targets['move_indices'] = move_indices
                    if total_moves is not None:
                        targets['total_moves'] = total_moves
                    
                    if use_mtl:
                        predictions.update({
                            'win': win_pred,
                            'material': material_pred,
                            'check': check_pred,
                        })
                        
                        targets.update({
                            'win': win_targets,
                            'material': material_targets,
                            'check': check_targets,
                        })
                        if move_indices is not None:
                            targets['move_indices'] = move_indices
                        if total_moves is not None:
                            targets['total_moves'] = total_moves
                    
                    loss, loss_dict = criterion(predictions, targets)
                    
                    policy_loss = loss_dict['policy']
                    value_loss = loss_dict['value']
                    
                    if use_mtl:
                        win_loss = loss_dict['win']
                        material_loss = loss_dict['material']
                        check_loss = loss_dict['check']
                    
                else:
                    policy_loss = criterion_policy(policy_pred, moves)
                    value_loss = criterion_value(value_pred, outcomes)
                    loss = policy_weight * policy_loss + value_weight * value_loss
                    
                    if use_mtl:
                        win_loss = criterion_win(win_pred.squeeze(), win_targets.squeeze())
                        material_loss = criterion_material(material_pred, material_targets)
                        check_loss = criterion_check(check_pred.squeeze(), check_targets.squeeze())
                        
                        loss = loss + (win_weight * win_loss + 
                                      material_weight * material_loss + 
                                      check_weight * check_loss)
                    
                    policy_loss = policy_loss.item()
                    value_loss = value_loss.item()
                    
                    if use_mtl:
                        win_loss = win_loss.item()
                        material_loss = material_loss.item()
                        check_loss = check_loss.item()
            
            if use_wdl:
                total_loss += loss_dict['total']
                total_policy_loss += loss_dict['policy']
                total_value_loss += loss_dict['value']
                
                if use_mtl:
                    total_win_loss += loss_dict['win']
                    total_material_loss += loss_dict['material']
                    total_check_loss += loss_dict['check']
            else:
                total_loss += loss.item()
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                
                if use_mtl:
                    total_win_loss += win_loss
                    total_material_loss += material_loss
                    total_check_loss += check_loss
            
            metrics_calc.update(
                policy_pred,
                value_pred,
                moves,
                outcomes,
                move_indices=move_indices,
                total_moves=total_moves,
                value_weight_min=config['imitation_learning'].get('value_move_weight_min', 0.1),
                value_weight_min_total_moves=config['imitation_learning'].get('value_move_weight_min_total_moves', 40),
                value_max_moves=config['data'].get('max_moves_per_game', 200),
                value_use_game_length=config['imitation_learning'].get('value_move_weight_use_game_length', False),
            )
    
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


def evaluate_models(model1, model2, config, device, num_games=100):
    """
    Evaluate model1 vs model2
    
    🔧 v4.4: Increased from 20→100 games for statistical significance
    """
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
