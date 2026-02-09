"""
Imitation Learning Training Script - v4.5
🆕 v4.5: CRITICAL FIXES - Per-Game Split + Promotions + Better Sampling
🆕 v4.5: CHESS METADATA - Castling, En Passant, Halfmove, Fullmove (16 planes)
🆕 v4.3: WDL VALUE HEAD + TEMPORAL DISCOUNTING (conditional on use_wdl)
🆕 v4.2: POV + Dynamic Sliding Window
- 🎯 POV: All boards from current player's perspective (flip for black)
- 🔄 Sliding Window: Dynamic history assembly at load time using mmap
- 🎮 GameID tracking: Efficient history reconstruction across positions
- 📊 Chess Metadata: 4 extra planes (castling, en passant, halfmove, fullmove)
- ⚡ WDL: Win/Draw/Loss classification (stronger signal than MSE)
- ⚡ TEMPORAL DISCOUNTING: DISABLED for WDL (clean ±1.0), ENABLED for MSE
- 🔒 Per-Game Split: Train/Val separated by games (no history leakage)
- 🎲 Per-Game Stride Offset: Per-game offset for unbiased sampling
- 👑 Promotions: Promotion-aware action space (see ACTION_SIZE)
"""

import torch
import torch.optim as optim
import yaml
import sys
import time
from pathlib import Path
import numpy as np
import math
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.data import process_pgn_files, create_dataloaders
from src.utils.data_helpers import ACTION_SIZE

# Import from utils
from utils.shared.logger import TrainingLogger
from utils.il.training_il import train_epoch_il, evaluate_il


def main():
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_version = config.get('model', {}).get('version', 'v?.?')
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("✓ TF32 + cuDNN benchmark enabled")
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Check AMP
    use_amp = config['hardware'].get('use_amp', True)
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    
    if use_amp and not torch.cuda.is_available():
        print("⚠️ AMP requested but CUDA not available, disabling AMP")
        use_amp = False
        use_bfloat16 = False
    
    if use_amp:
        amp_dtype = "bfloat16" if use_bfloat16 else "float16"
        print(f"✓ Mixed Precision Training (AMP) enabled with {amp_dtype}")
        
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            print("⚠️ bfloat16 requested but not supported, falling back to float16")
            use_bfloat16 = False
            config['hardware']['use_bfloat16'] = False
    
    # Create directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    il_dir = base_dir / config['paths']['il_checkpoints_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    il_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    use_mtl = config['model'].get('use_multitask_learning', False)
    history_positions = config['model'].get('history_positions', 0)
    stride = config['data'].get('sliding_window_stride', 1)
    
    # 🆕 v4.5 FIXED: Calculate expected input planes with chess metadata
    expected_input_planes = 16 * (1 + history_positions)  # 🔧 FIXED: 16 planes (12 pieces + 4 metadata)
    
    print("\n" + "="*70)
    print(f"🆕 {model_version} CRITICAL FIXES + {model_version} METADATA + v4.3 WDL + v4.2 POV")
    print("="*70)
    print(f"  • POV: Boards from current player's perspective")
    print(f"  • History positions: {history_positions} (assembled dynamically)")
    print(f"  • Sliding window stride: {stride}x (per-game offset)")
    print(f"  • Expected input planes: {expected_input_planes} (16 × {1 + history_positions})")
    print(f"  • 🆕 Chess metadata: Castling, En Passant, Halfmove, Fullmove")
    print(f"  • 🆕 Promotions: {ACTION_SIZE} actions ({ACTION_SIZE - 4096} promotion actions)")
    print(f"  • 🆕 Per-game split: No validation leakage")
    print(f"  • MTL enabled: {use_mtl}")
    print(f"  • 🆕 WDL Value: {config['model'].get('use_wdl_value', True)}")
    print("="*70 + "\n")
    
    # Setup debug logging
    debug_enabled = config.get('debug', {}).get('enabled', False)
    debug_log_file = None
    
    if debug_enabled:
        debug_dir = logs_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_file = debug_dir / f"training_profile_{timestamp}.txt"
        
        print("\n" + "="*70)
        print("🐛 DEBUG MODE ENABLED")
        print("="*70)
        print(f"  • Profile training:  {config['debug'].get('profile_training', False)}")
        print(f"  • Log GPU memory:    {config['debug'].get('log_gpu_memory', False)}")
        print(f"  • Profile every:     {config['debug'].get('profile_every_n_epochs', 1)} epochs")
        print(f"  • Log file:          {debug_log_file}")
        print("="*70 + "\n")
        
        with open(debug_log_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"🐛 TRAINING DEBUG LOG - {model_version} Chess Metadata + WDL\n")
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {config['model']['filters']} filters, {config['model']['num_residual_blocks']} blocks\n")
            f.write(f"Batch size: {config['imitation_learning']['batch_size']}\n")
            f.write(f"Learning rate: {config['imitation_learning']['learning_rate']}\n")
            f.write(f"History positions: {history_positions} (dynamic)\n")
            f.write(f"Sliding window stride: {stride}x\n")
            f.write(f"Input planes: {expected_input_planes} (16 per position)\n")
            f.write(f"Chess metadata: Castling, En Passant, Halfmove, Fullmove\n")
            f.write(f"MTL enabled: {use_mtl}\n")
            f.write(f"WDL Value: {config['model'].get('use_wdl_value', True)}\n")
            f.write(f"POV enabled: True\n")
            f.write("="*70 + "\n\n")
    
    # Initialize logger
    logger = TrainingLogger(
        logs_dir, 
        experiment_name=f"il_training_{model_version}", 
        mode="il",
        use_mtl=use_mtl
    )
    
    # Load data with multi-phase processing
    print(f"\n=== Loading data ({model_version} Chess Metadata + POV + Sliding Window) ===")
    print(f"Mode: {'SEQUENTIAL' if config['data']['files_at_once'] == 1 else 'BATCH'}")
    print(f"Files at once: {config['data']['files_at_once']}")
    print(f"Phase 1 workers: {config['data'].get('phase1_threads', 1)}")
    print(f"Phase 2 workers: {config['data'].get('phase2_threads', 1)}")
    
    # Find PGN files
    data_dir = script_dir.parent / config['paths']['data_dir']
    pgn_files = sorted(data_dir.glob('*.pgn'))
    
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_dir}")
    
    print(f"Found {len(pgn_files)} PGN file(s):")
    for pgn in pgn_files:
        print(f"  • {pgn.name}")
    
    # 🔧 FIX: Convert relative paths in config to absolute paths
    # This ensures data/preprocessing is created in chess/data/ regardless of where script is called from
    config_with_absolute_paths = config.copy()
    config_with_absolute_paths['paths'] = config['paths'].copy()
    config_with_absolute_paths['paths']['data_dir'] = str(data_dir.absolute())
    
    # Process with smart tracking
    metadata = process_pgn_files(pgn_files, config_with_absolute_paths)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"Total positions (before stride): {metadata['total_positions']:,}")
    
    # 🔧 v4.5: Check if WDL is enabled (affects temporal discounting)
    use_wdl = config['model'].get('use_wdl_value', True)
    if use_wdl:
        print(f"  • WDL enabled: Temporal discounting DISABLED (clean ±1.0 targets) ✓")
    else:
        print(f"  • MSE regression: Temporal discounting ENABLED (sqrt scaling) ⚠️")
    
    # 🔧 v4.5: Verify binary format compatibility
    # Layout: [Board 38B] + [GameID 4B] + [MoveIdx 2B] + [MoveTarget 2B] + [Outcome 4B] + [MTL 12B]
    actual_position_size = metadata.get('position_size')
    expected_position_size = 50 if not use_mtl else 62  # 🆕 v4.5: Board is now 38B (was 36B)
    
    if actual_position_size != expected_position_size:
        print(f"\n⚠️ WARNING: Position size mismatch!")
        print(f"  • Expected: {expected_position_size} bytes")
        print(f"  • Actual: {actual_position_size} bytes")
        print(f"  • This may indicate the data was preprocessed with an old format or different MTL setting")
        
        # Try to determine if it's just a MTL mismatch
        if actual_position_size == 50 and use_mtl:
            print(f"  ❌ Data was processed WITHOUT MTL, but config has use_multitask_learning=True")
            print(f"     Please either:")
            print(f"     1. Set use_multitask_learning=False in config.yaml, OR")
            print(f"     2. Delete cache and reprocess data with MTL enabled")
            raise ValueError("MTL mismatch between data and config")
        elif actual_position_size == 62 and not use_mtl:
            print(f"  ❌ Data was processed WITH MTL, but config has use_multitask_learning=False")
            print(f"     Please either:")
            print(f"     1. Set use_multitask_learning=True in config.yaml, OR")
            print(f"     2. Delete cache and reprocess data without MTL")
            raise ValueError("MTL mismatch between data and config")
        elif actual_position_size in [48, 60]:  # Old v4.4 format (36B board)
            print(f"  âťŚ Data was processed with OLD v4.4 format (36B board, no fullmove metadata)")
            print(f"     đź†• {model_version} adds fullmove number (38B board)")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        elif actual_position_size in [44, 56]:  # Old v4.3 format (32B board)
            print(f"  ❌ Data was processed with OLD v4.3 format (32B board, no chess metadata)")
            print(f"     🆕 {model_version} uses 38B board with castling, en passant, halfmove, fullmove")
            print(f"     Please delete cache (data/preprocessing/) and reprocess with {model_version}")
            raise ValueError("Old data format - please reprocess")
        else:
            print(f"  ❌ Unknown format mismatch - please delete cache and reprocess")
            raise ValueError("Position size mismatch")
    
    print(f"\n✅ Binary Format Validation:")
    print(f"  • Position size: {actual_position_size} bytes ✓")
    print(f"  • Board size: 38 bytes (32B pieces + 6B metadata) ✓")
    print(f"  • Chess metadata: Castling, En Passant, Halfmove, Fullmove ✓")
    print(f"  • MTL: {'ENABLED' if use_mtl else 'DISABLED'} ✓")
    print(f"  • POV: Boards from current player's perspective ✓")
    print(f"  • Sliding Window: Dynamic history assembly ✓")
    
    # Create dataloaders (they will apply sliding window and build history dynamically)
    train_loader, val_loader = create_dataloaders(metadata, config)
    
    # Create model with correct input planes
    print("\n=== Creating model ===")
    print(f"Input planes: {expected_input_planes} (16 × {1 + history_positions})")
    print(f"  • Per position: 12 pieces + 4 metadata (castling, en passant, halfmove, fullmove)")
    print(f"Architecture: {config['model']['num_residual_blocks']} blocks, {config['model']['filters']} filters")
    
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("✓ Model converted to channels_last memory format")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 🆕 Per-layer learning rates - value head with lower LR to prevent overfitting
    value_head_lr_factor = config['imitation_learning'].get('value_head_lr_factor', 1.0)
    base_lr = config['imitation_learning']['learning_rate']
    
    if value_head_lr_factor != 1.0:
        # Separate value head parameters
        value_head_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'value_' in name:  # value_conv1, value_conv2, value_bn, value_fc1, value_fc2
                value_head_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': base_lr},
            {'params': value_head_params, 'lr': base_lr * value_head_lr_factor}
        ]
        
        print(f"\n🎯 Per-layer Learning Rates:")
        print(f"  • Trunk + Policy head: {base_lr:.4f}")
        print(f"  • Value head: {base_lr * value_head_lr_factor:.4f} ({value_head_lr_factor}x)")
    else:
        param_groups = model.parameters()
    
    # Optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=config['imitation_learning']['weight_decay'],
        fused=True if torch.cuda.is_available() else False
    )
    
    if torch.cuda.is_available():
        print("✓ Using fused AdamW optimizer")
    
    # Learning rate scheduler
    scheduler_type = config['imitation_learning'].get('scheduler_type', 'cosine_warm_restarts')
    scheduler = None
    
    if scheduler_type == 'onecycle':
        total_steps = len(train_loader) * config['imitation_learning']['epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['imitation_learning']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=10000.0
        )
        print("✓ OneCycleLR scheduler enabled")
    
    elif scheduler_type == 'cosine_decay':
        total_epochs = config['imitation_learning']['epochs']
        warmup_pct = config['imitation_learning'].get('warmup_pct', 0.05)
        min_lr_ratio = config['imitation_learning'].get('min_lr_ratio', 0.05)
        
        warmup_pct = max(0.0, min(1.0, float(warmup_pct)))
        min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
        
        warmup_epochs = int(round(total_epochs * warmup_pct))
        warmup_epochs = max(0, min(warmup_epochs, total_epochs))
        
        if warmup_epochs > 0:
            start_factor = 1.0 / warmup_epochs if warmup_epochs > 1 else 1.0
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs
            )
            
            if warmup_epochs >= total_epochs:
                scheduler = warmup_scheduler
            else:
                cosine_epochs = total_epochs - warmup_epochs
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, cosine_epochs),
                    eta_min=base_lr * min_lr_ratio
                )
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
        else:
            start_factor = 1.0
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_epochs),
                eta_min=base_lr * min_lr_ratio
            )
        
        print(f"✓ Cosine decay + warmup: warmup={warmup_epochs} ep "
              f"({warmup_pct:.0%}), start_factor={start_factor:.3f}, min_lr_ratio={min_lr_ratio}")
    
    elif scheduler_type == 'cosine_warm_restarts':
        t0 = config['imitation_learning'].get('cosine_t0', 10)
        t_mult = config['imitation_learning'].get('cosine_t_mult', 2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=t_mult,
            eta_min=config['imitation_learning']['learning_rate'] / 100
        )
        print(f"✓ CosineAnnealingWarmRestarts: T0={t0}, T_mult={t_mult}")
    
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print("✓ ReduceLROnPlateau scheduler enabled")
    
    elif scheduler_type == 'none':
        print("✓ No LR scheduler (constant learning rate)")
    
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    # AMP Gradient Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # 🆕 Stochastic Weight Averaging (SWA) for better generalization with large batches
    use_swa = config['imitation_learning'].get('use_swa', False)
    swa_model = None
    swa_scheduler = None
    swa_start = config['imitation_learning'].get('swa_start_epoch', 15)
    
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr = config['imitation_learning'].get('swa_lr', 0.0005)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
        print(f"\n🎯 Stochastic Weight Averaging (SWA):")
        print(f"  • Start epoch: {swa_start}")
        print(f"  • SWA LR: {swa_lr:.6f}")
        print(f"  • Benefits: Better generalization, flatter minima")
    
    # Best model path
    best_model_path = base_dir / config['paths']['best_model_il']
    
    # Training loop
    print("\n=== Starting training ===")
    print("💾 Saving strategy: NO optimizer state (minimal file size)")
    print("📊 Tracking: Policy Accuracy (Top-1, Top-3) and Value MAE")
    
    if history_positions > 0:
        print(f"📜 History: {history_positions} positions (assembled dynamically with sliding window)")
        print(f"🔢 Input planes: {expected_input_planes}")
    
    if stride > 1:
        print(f"⚡ Sliding window stride: {stride}x (sampling every {stride} positions)")
    
    if debug_enabled:
        print("🐛 Debug mode active - profiling enabled")
    profile_enabled = debug_enabled and config.get('debug', {}).get('profile_training', False)
    profile_every = config.get('debug', {}).get('profile_every_n_epochs', 1)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = config['imitation_learning'].get('max_patience', 15)
    min_delta = config['imitation_learning']['min_delta']
    checkpoint_every = config['imitation_learning'].get('checkpoint_every', 5)
    
    print(f"Early stopping: patience={max_patience}, min_delta={min_delta}")
    print(f"Checkpoints saved every {checkpoint_every} epochs to: {il_dir}")
    print(f"Best model saved to: {best_model_path}")
    
    for epoch in range(config['imitation_learning']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['imitation_learning']['epochs']}")
        epoch_lr = optimizer.param_groups[0]['lr']
        
        profile_this_epoch = profile_enabled and ((epoch + 1) % profile_every == 0)
        if profile_this_epoch and device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
        
        # Train epoch
        if profile_this_epoch and device.type == 'cuda':
            torch.cuda.synchronize()
        train_start_time = time.perf_counter()
        train_losses, train_metrics, train_profile = train_epoch_il(
            model, train_loader, optimizer, scheduler, config, device, scaler,
            epoch=epoch, debug_log_file=debug_log_file, profile=profile_this_epoch
        )
        if profile_this_epoch and device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.perf_counter() - train_start_time
        if profile_this_epoch and train_profile is not None:
            train_time = train_profile['total']
        
        # 🆕 SWA: Update averaged model after swa_start epoch
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler is not None:
                swa_scheduler.step()
            print(f"       🎯 SWA: Updated averaged weights (epoch {epoch + 1 - swa_start}/{config['imitation_learning']['epochs'] - swa_start})")
        
        # Print train losses and metrics
        print(f"Train - Loss: {train_losses['total']:.4f}, "
              f"Policy: {train_losses['policy']:.4f}, "
              f"Value: {train_losses['value']:.4f}, "
              f"LR: {epoch_lr:.2e}")
        
        print(f"       📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
              f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
              f"MAE: {train_metrics['value_mae']:.4f}")
        
        if use_mtl:
            print(f"       🆕 MTL - Win: {train_losses['win']:.4f}, "
                  f"Material: {train_losses['material']:.4f}, "
                  f"Check: {train_losses['check']:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            eval_start_time = time.perf_counter()
            val_losses, val_metrics = evaluate_il(model, val_loader, config, device)
            if profile_this_epoch and device.type == 'cuda':
                torch.cuda.synchronize()
            eval_time = time.perf_counter() - eval_start_time
            
            # Step ReduceLROnPlateau scheduler (needs val_loss)
            if scheduler_type == 'reduce_on_plateau' and scheduler is not None:
                scheduler.step(val_losses['total'])
            
            print(f"Val - Loss: {val_losses['total']:.4f}, "
                  f"Policy: {val_losses['policy']:.4f}, "
                  f"Value: {val_losses['value']:.4f}")
            
            print(f"     📊 Top-1: {val_metrics['policy_top1_acc']:.2%}, "
                  f"Top-3: {val_metrics['policy_top3_acc']:.2%}, "
                  f"MAE: {val_metrics['value_mae']:.4f}")
            
            if use_mtl:
                print(f"     🆕 MTL - Win: {val_losses['win']:.4f}, "
                      f"Material: {val_losses['material']:.4f}, "
                      f"Check: {val_losses['check']:.4f}")
            
            # Log metrics
            logger.log(epoch + 1, train_losses, val_losses, train_metrics, val_metrics, epoch_lr)
            logger.plot()
            
            # Save best model
            improvement = best_val_loss - val_losses['total']
            if improvement > min_delta:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                print(f"✓ New best model! Val loss: {val_losses['total']:.4f} "
                      f"(improved by {improvement:.4f})")
                print(f"  📊 Val Top-1: {val_metrics['policy_top1_acc']:.2%}")
                
                model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
                save_checkpoint(
                    model_to_save, 
                    None,
                    epoch, 
                    val_losses['total'],
                    str(best_model_path),
                    {
                        'val_policy_loss': val_losses['policy'], 
                        'val_value_loss': val_losses['value'],
                        'val_policy_top1': val_metrics['policy_top1_acc'],
                        'val_policy_top3': val_metrics['policy_top3_acc'],
                        'val_value_mae': val_metrics['value_mae'],
                        'use_amp': use_amp,
                        'use_bfloat16': use_bfloat16,
                        'use_mtl': use_mtl,
                        'history_positions': history_positions,
                        'input_planes': expected_input_planes,
                        'sliding_window_stride': stride,
                        'pov_enabled': True,  # 🆕 v4.2
                        'version': model_version     # 🆕 Track version
                    },
                    save_optimizer=False
                )
                
                if use_bfloat16:
                    model = model.to(torch.float32)
                
                print(f"  💾 Saved to: {best_model_path}")
                size_mb = best_model_path.stat().st_size / (1024**2)
                print(f"  📦 Model size: {size_mb:.1f} MB (no optimizer)")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{max_patience}")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        else:
            # Log only training metrics
            logger.log(epoch + 1, train_losses, None, train_metrics, None, epoch_lr)
            eval_time = 0.0
        
        if profile_this_epoch:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            epoch_total_time = time.perf_counter() - epoch_start_time
            other_time = max(0.0, epoch_total_time - train_time - eval_time)
            if train_profile is not None:
                batches = max(1, train_profile['batches'])
                profile_msg = (
                    f"[PROFILE] Epoch {epoch + 1}: "
                    f"data={train_profile['data']:.2f}s ({train_profile['data']*1000/batches:.1f}ms/b), "
                    f"fwd={train_profile['forward']:.2f}s ({train_profile['forward']*1000/batches:.1f}ms/b), "
                    f"bwd={train_profile['backward']:.2f}s ({train_profile['backward']*1000/batches:.1f}ms/b), "
                    f"optim={train_profile['optim']:.2f}s ({train_profile['optim']*1000/batches:.1f}ms/b), "
                    f"metrics={train_profile['metrics']:.2f}s ({train_profile['metrics']*1000/batches:.1f}ms/b), "
                    f"train_total={train_profile['total']:.2f}s, "
                    f"eval={eval_time:.2f}s, "
                    f"other={other_time:.2f}s, "
                    f"epoch_total={epoch_total_time:.2f}s"
                )
            else:
                profile_msg = (
                    f"[PROFILE] Epoch {epoch + 1}: "
                    f"train={train_time:.2f}s, "
                    f"eval={eval_time:.2f}s, "
                    f"other={other_time:.2f}s, "
                    f"total={epoch_total_time:.2f}s"
                )
            print(profile_msg)
            if debug_log_file is not None:
                with open(debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(profile_msg + "\n")

        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0:
            # Get current val loss if not already computed
            if (epoch + 1) % config['imitation_learning']['eval_every'] != 0:
                val_losses, val_metrics = evaluate_il(model, val_loader, config, device)
            
            checkpoint_name = f"il_epoch_{epoch+1}_valloss_{val_losses['total']:.4f}_top1_{val_metrics['policy_top1_acc']:.3f}.pt"
            checkpoint_path = il_dir / checkpoint_name
            
            model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
            save_checkpoint(
                model_to_save, 
                None,
                epoch, 
                train_losses['total'],
                str(checkpoint_path),
                {
                    'val_loss': val_losses['total'],
                    'val_policy_loss': val_losses['policy'],
                    'val_value_loss': val_losses['value'],
                    'val_policy_top1': val_metrics['policy_top1_acc'],
                    'val_policy_top3': val_metrics['policy_top3_acc'],
                    'val_value_mae': val_metrics['value_mae'],
                    'use_mtl': use_mtl,
                    'history_positions': history_positions,
                    'input_planes': expected_input_planes,
                    'sliding_window_stride': stride,
                    'pov_enabled': True,  # 🆕 v4.2
                    'version': model_version     # 🆕 Track version
                },
                save_optimizer=False
            )
            
            if use_bfloat16:
                model = model.to(torch.float32)
            
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"💾 Checkpoint saved: {checkpoint_path.name} ({size_mb:.1f} MB)")
        
        # Cleanup
        if (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final plot
    logger.plot()
    
    # 🆕 SWA: Finalize and save averaged model
    if use_swa and swa_model is not None:
        print("\n" + "="*70)
        print("🎯 Finalizing SWA (Stochastic Weight Averaging)")
        print("="*70)
        
        # Update BatchNorm statistics for SWA model
        print("Updating BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # Evaluate SWA model
        print("Evaluating SWA model...")
        swa_val_losses, swa_val_metrics = evaluate_il(swa_model.module, val_loader, config, device)
        
        print(f"\nSWA Model Performance:")
        print(f"  Val Loss: {swa_val_losses['total']:.4f}")
        print(f"  Val MAE: {swa_val_metrics['value_mae']:.4f}")
        print(f"  Val Top-1: {swa_val_metrics['policy_top1_acc']:.2%}")
        
        # Save SWA model
        swa_model_path = best_model_path.parent / "best_model_il_swa.pt"
        swa_model_to_save = swa_model.module.to(torch.bfloat16) if use_bfloat16 else swa_model.module
        
        save_checkpoint(
            swa_model_to_save,
            None,
            epoch,
            swa_val_losses['total'],
            str(swa_model_path),
            {
                'val_loss': swa_val_losses['total'],
                'val_policy_loss': swa_val_losses['policy'],
                'val_value_loss': swa_val_losses['value'],
                'val_policy_top1': swa_val_metrics['policy_top1_acc'],
                'val_policy_top3': swa_val_metrics['policy_top3_acc'],
                'val_value_mae': swa_val_metrics['value_mae'],
                'swa_enabled': True,
                'swa_start_epoch': swa_start,
                'use_mtl': use_mtl,
                'history_positions': history_positions,
                'input_planes': expected_input_planes,
                'sliding_window_stride': stride,
                'pov_enabled': True,
                'version': model_version
            },
            save_optimizer=False
        )
        
        size_mb = swa_model_path.stat().st_size / (1024**2)
        print(f"\n✅ SWA model saved: {swa_model_path.name} ({size_mb:.1f} MB)")
        print(f"  Val loss improvement: {best_val_loss - swa_val_losses['total']:.4f}")
        print("="*70)
    
    print("\n=== Training complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {il_dir}")
    print(f"Logs: {logs_dir}")
    if debug_log_file:
        print(f"Debug log: {debug_log_file}")
    
    print("\n" + "="*70)
    print(f"🆕 {model_version} Features Used:")
    print("="*70)
    print(f"  ✓ Chess Metadata - castling, en passant, halfmove, fullmove")
    print(f"  ✓ WDL Value Head - Win/Draw/Loss classification")
    print(f"  ✓ POV (Point of View) - boards from current player's perspective")
    print(f"  ✓ Dynamic Sliding Window - history assembled at load time")
    print(f"  ✓ GameID tracking - efficient history reconstruction")
    print(f"  ✓ Stride {stride}x - sampled every {stride} positions")
    if use_wdl:
        print(f"  ✓ Temporal Discounting - DISABLED for WDL (clean targets)")
    else:
        print(f"  ✓ Temporal Discounting - ENABLED for MSE (sqrt scaling)")
    if use_mtl:
        print(f"  ✓ Multi-Task Learning - win, material, check predictions")
    if use_swa:
        print(f"  ✓ Stochastic Weight Averaging - better generalization with large batches")
    if value_head_lr_factor != 1.0:
        print(f"  ✓ Per-layer LR - value head at {value_head_lr_factor}x to prevent overfitting")
    print("="*70)


if __name__ == "__main__":
    main()

