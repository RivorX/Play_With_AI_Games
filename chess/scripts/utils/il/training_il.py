"""
Training and evaluation functions for IL with WDL and move-weighted losses
"""

import time
import torch
from tqdm import tqdm

from .loss import CombinedLoss
from ..shared.metrics import MetricsCalculator

def train_epoch_il(
    model,
    train_loader,
    optimizer,
    scheduler,
    config,
    device,
    scaler,
    epoch=0,
    debug_log_file=None,
    profile=False,
    step_scheduler=True,
    non_blocking_transfer=True,
):
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
        step_scheduler: Whether to step the provided scheduler at epoch end
        non_blocking_transfer: Use async host->device copies when possible
    
    Returns:
        Tuple of (losses_dict, metrics_dict, profile_stats or None)
    """
    model.train()

    debug_cfg = config.get('debug', {})
    debug_enabled = bool(debug_cfg.get('enabled', False))
    log_gpu_memory = bool(debug_enabled and debug_cfg.get('log_gpu_memory', False))
    log_grad_diagnostics = bool(debug_enabled)
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    # Accumulators
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    profile_stats = None
    profile_enabled = bool(profile)
    grad_diag_logged = False
    grad_diag = None
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

    if device.type == 'cuda' and torch.cuda.is_available() and (profile_enabled or log_gpu_memory):
        torch.cuda.reset_peak_memory_stats(device)
    
    # đź”Ť DIAGNOSTIC: Track target distributions (gated by config)
    first_batch_targets = True
    first_batch_predictions = True
    show_batch0_diagnostics = (
        debug_enabled and
        debug_cfg.get('print_batch0_diagnostics', False)
    )
    
    # âšˇ Pre-read AMP config outside loop (avoid dict lookups per batch)
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    non_blocking = bool(non_blocking_transfer and device.type == 'cuda')
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

        
        # Move to device
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
        moves = moves.to(device, non_blocking=non_blocking)
        outcomes = outcomes.to(device, non_blocking=non_blocking)
        
        if move_indices is not None:
            move_indices = move_indices.to(device, non_blocking=non_blocking)
        if total_moves is not None:
            total_moves = total_moves.to(device, non_blocking=non_blocking)
        

        # đź”Ť DIAGNOSTIC: Print distributions for first batch
        if show_batch0_diagnostics and first_batch_targets:
            print(f"\nđź”Ť DIAGNOSTIC - Batch 0:")
            print(f"  Outcome targets (Value):")
            print(f"    Min: {outcomes.min().item():.3f}, Max: {outcomes.max().item():.3f}, Mean: {outcomes.mean().item():.3f}")
            print(f"    Unique values: {torch.unique(outcomes).cpu().numpy()[:10]}")  # First 10 unique
            print(f"    Distribution: +1: {(outcomes > 0.9).sum().item()}, 0: {(outcomes.abs() < 0.1).sum().item()}, -1: {(outcomes < -0.9).sum().item()}")
            
            first_batch_targets = False
        
        optimizer.zero_grad(set_to_none=True)

        if profile_enabled:
            _sync()
            t0 = time.perf_counter()
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            policy_pred, value_pred = model(boards)
            
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

        if log_grad_diagnostics and not grad_diag_logged:
            trainable_tensors = 0
            grad_tensors = 0
            trainable_params = 0
            grad_params = 0
            missing_grad_names = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                trainable_tensors += 1
                trainable_params += int(param.numel())
                if param.grad is not None:
                    grad_tensors += 1
                    grad_params += int(param.numel())
                elif len(missing_grad_names) < 8:
                    missing_grad_names.append(name)

            grad_diag = {
                'batch_idx': int(batch_idx),
                'trainable_tensors': int(trainable_tensors),
                'grad_tensors': int(grad_tensors),
                'trainable_params': int(trainable_params),
                'grad_params': int(grad_params),
                'missing_grad_names': missing_grad_names,
            }

            if device.type == 'cuda' and torch.cuda.is_available() and log_gpu_memory:
                grad_diag.update({
                    'peak_allocated_mb': float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                    'peak_reserved_mb': float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
                    'current_allocated_mb': float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
                    'current_reserved_mb': float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
                })
            grad_diag_logged = True
        
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
    
    # Step scheduler once per epoch (if enabled by caller).
    if scheduler is not None and step_scheduler:
        scheduler.step()
    
    # Compute final metrics
    n = len(train_loader)
    
    losses = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n
    }
    
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

    if grad_diag is not None:
        if profile_stats is None:
            profile_stats = {}
        profile_stats['grad_diag'] = grad_diag

    if device.type == 'cuda' and torch.cuda.is_available() and (profile_enabled or log_gpu_memory):
        if profile_stats is None:
            profile_stats = {}
        profile_stats.update({
            'peak_allocated_mb': float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            'peak_reserved_mb': float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
            'current_allocated_mb': float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
            'current_reserved_mb': float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        })
    
    return losses, metrics, profile_stats


def evaluate_il(model, val_loader, config, device, non_blocking_transfer=True):
    """
    Evaluate model with WDL value head
    
    Identical to train_epoch_il but without gradient updates
    """
    model.eval()
    
    # WDL-only path
    criterion = CombinedLoss(config)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    metrics_calc = MetricsCalculator()
    
    # Pre-read AMP config outside loop
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    non_blocking = bool(non_blocking_transfer and device.type == 'cuda')
    
    with torch.inference_mode():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
            if isinstance(batch_data, dict):
                boards = batch_data['board']
                moves = batch_data['move']
                outcomes = batch_data['value']
                move_indices = batch_data.get('move_idx', None)
                total_moves = batch_data.get('total_moves', None)
            else:
                if len(batch_data) == 5:
                    boards, moves, outcomes, move_indices, total_moves = batch_data
                elif len(batch_data) == 4:
                    boards, moves, outcomes, move_indices = batch_data
                    total_moves = None
                else:
                    boards, moves, outcomes = batch_data
                    move_indices = None
                    total_moves = None
            
            boards = boards.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
            moves = moves.to(device, non_blocking=non_blocking)
            outcomes = outcomes.to(device, non_blocking=non_blocking)
            
            if move_indices is not None:
                move_indices = move_indices.to(device, non_blocking=non_blocking)
            if total_moves is not None:
                total_moves = total_moves.to(device, non_blocking=non_blocking)
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                policy_pred, value_pred = model(boards)
                
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
                    
                loss, loss_dict = criterion(predictions, targets)
            
            total_loss += loss_dict['total']
            total_policy_loss += loss_dict['policy']
            total_value_loss += loss_dict['value']
            
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
    
    metrics = metrics_calc.compute()
    
    return losses, metrics
