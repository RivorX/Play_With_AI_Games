"""
Training and evaluation functions for IL with WDL and move-weighted losses
"""

import time
import torch
from tqdm import tqdm

from .loss import CombinedLoss
from ..shared.metrics import MetricsCalculator

def train_epoch_il(model, train_loader, optimizer, scheduler, config, device, scaler,
                   epoch=0, debug_log_file=None, profile=False):
    """
    đź†• v4.3: Train one epoch with WDL value head and move-weighted losses
    
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
        Tuple of (losses_dict, metrics_dict, profile_stats or None)
    """
    model.train()
    
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    # Accumulators
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    profile_stats = None
    profile_enabled = bool(profile)
    if profile_enabled:
        timers = {
            'data': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optim': 0.0,
            'metrics': 0.0,
        }
        batch_count = 0

        def _sync():
            if device.type == 'cuda':
                torch.cuda.synchronize()

        data_timer_start = time.perf_counter()
    
    # đź”Ť DIAGNOSTIC: Track target distributions (gated by config)
    first_batch_targets = True
    first_batch_predictions = True
    show_batch0_diagnostics = (
        config.get('debug', {}).get('enabled', False) and
        config.get('debug', {}).get('print_batch0_diagnostics', False)
    )
    
    # âšˇ Pre-read AMP config outside loop (avoid dict lookups per batch)
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    for batch_idx, batch_data in enumerate(pbar):
        if profile_enabled:
            _sync()
            timers['data'] += time.perf_counter() - data_timer_start
        # Unpack batch
        if isinstance(batch_data, dict):
            boards = batch_data['board']
            moves = batch_data['move']
            outcomes = batch_data['value']
            
            # đź”§ v4.4 FIX: Get move_idx (not move_indices) from dict
            move_indices = batch_data.get('move_idx', None)
            total_moves = batch_data.get('total_moves', None)
            
            if use_mtl:
                win_targets = batch_data['win']
                material_targets = batch_data['material']
                check_targets = batch_data['check']
        else:
            # đź”§ v4.4 FIX: Unpack move_indices from tuple
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
        
        # đź”Ť DIAGNOSTIC: Print distributions for first batch
        if show_batch0_diagnostics and first_batch_targets:
            print(f"\nđź”Ť DIAGNOSTIC - Batch 0:")
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

        if profile_enabled:
            _sync()
            t0 = time.perf_counter()
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            if use_mtl:
                policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
            else:
                policy_pred, value_pred = model(boards, return_aux=False)
            
            # đź†• v4.3: Compute loss using CombinedLoss
            # Pack predictions and targets for CombinedLoss
            predictions = {
                'policy': policy_pred,
                'value': value_pred,  # đź†• Now (B, 3) WDL logits!
            }
                
            targets = {
                'moves': moves,
                'values': outcomes,  # Still scalar {-1, 0, +1}
            }
            if move_indices is not None:
                targets['move_indices'] = move_indices
            if total_moves is not None:
                targets['total_moves'] = total_moves
            # đź”Ť DIAGNOSTIC: Print WDL predictions for first batch
            if show_batch0_diagnostics and first_batch_predictions and batch_idx == 0:
                wdl_probs = torch.softmax(value_pred[:10], dim=1)
                value_scalars = (wdl_probs[:, 0] * 1.0 + 
                                wdl_probs[:, 1] * 0.0 + 
                                wdl_probs[:, 2] * (-1.0))
                print(f"\n  đź”Ť WDL Predictions (first 10 samples):")
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
                    targets['move_indices'] = move_indices  # đź†• For move weighting
                if total_moves is not None:
                    targets['total_moves'] = total_moves
                
            # Single loss computation
            loss, loss_dict = criterion(predictions, targets)
                
        
        if profile_enabled:
            _sync()
            timers['forward'] += time.perf_counter() - t0
            t0 = time.perf_counter()

        # Backward pass
        scaler.scale(loss).backward()

        if profile_enabled:
            _sync()
            timers['backward'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Gradient clipping
        grad_clip = config['imitation_learning'].get('grad_clip', 1.0)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        if profile_enabled:
            _sync()
            timers['optim'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        
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
        total_loss += loss_dict['total']
        total_policy_loss += loss_dict['policy']
        total_value_loss += loss_dict['value']
            
        if use_mtl:
            total_win_loss += loss_dict['win']
            total_material_loss += loss_dict['material']
            total_check_loss += loss_dict['check']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'policy': f'{total_policy_loss / (batch_idx + 1):.4f}',
            'value': f'{total_value_loss / (batch_idx + 1):.4f}',
        })

        if profile_enabled:
            _sync()
            timers['metrics'] += time.perf_counter() - t0
            batch_count += 1
            data_timer_start = time.perf_counter()
    
    # Step scheduler once per epoch.
    if scheduler is not None:
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

    if profile_enabled:
        total_profile_time = sum(timers.values())
        profile_stats = {
            'data': timers['data'],
            'forward': timers['forward'],
            'backward': timers['backward'],
            'optim': timers['optim'],
            'metrics': timers['metrics'],
            'total': total_profile_time,
            'batches': max(1, batch_count),
        }
    
    return losses, metrics, profile_stats


def evaluate_il(model, val_loader, config, device):
    """
    đź†• v4.3: Evaluate model with WDL value head
    
    Identical to train_epoch_il but without gradient updates
    """
    model.eval()
    
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    # âšˇ Pre-read AMP config outside loop
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.inference_mode():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            if isinstance(batch_data, dict):
                boards = batch_data['board']
                moves = batch_data['move']
                outcomes = batch_data['value']
                move_indices = batch_data.get('move_idx', None)  # đź”§ v4.4 FIX: move_idx not move_indices
                total_moves = batch_data.get('total_moves', None)
                
                if use_mtl:
                    win_targets = batch_data['win']
                    material_targets = batch_data['material']
                    check_targets = batch_data['check']
            else:
                # đź”§ v4.4 FIX: Unpack move_indices from tuple
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
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if use_mtl:
                    policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
                else:
                    policy_pred, value_pred = model(boards, return_aux=False)
                
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
                    
            
            total_loss += loss_dict['total']
            total_policy_loss += loss_dict['policy']
            total_value_loss += loss_dict['value']
                
            if use_mtl:
                total_win_loss += loss_dict['win']
                total_material_loss += loss_dict['material']
                total_check_loss += loss_dict['check']
            
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
