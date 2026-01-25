"""
Imitation Learning Training Script with Comprehensive Metrics
"""

import torch
import torch.optim as optim
import yaml
import sys
from pathlib import Path
import numpy as np
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.data import parse_pgn_files, create_dataloaders

# Import from utils
from utils import TrainingLogger, train_epoch_il, evaluate_il


def main():
    # Load config
    config_path = script_dir.parent / 'config' / 'config.yaml'
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
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
    
    # Check if MTL is enabled
    use_mtl = config['model'].get('use_multitask_learning', False)
    
    # Initialize logger
    logger = TrainingLogger(
        logs_dir, 
        experiment_name="il_training", 
        mode="il",
        use_mtl=use_mtl
    )
    
    # Load data
    print("\n=== Loading data ===")
    metadata = parse_pgn_files(
        config['paths']['data_dir'], 
        config,
        num_workers=config['hardware']['num_workers']
    )
    
    print(f"Total positions: {metadata['total_positions']}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(metadata, config)
    
    # Create model
    print("\n=== Creating model ===")
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    print("✓ Model converted to channels_last memory format")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['imitation_learning']['learning_rate'],
        weight_decay=config['imitation_learning']['weight_decay'],
        fused=True if torch.cuda.is_available() else False
    )
    
    if torch.cuda.is_available():
        print("✓ Using fused AdamW optimizer")
    
    # Learning rate scheduler
    use_onecycle = config['imitation_learning'].get('use_onecycle_lr', True)
    scheduler = None
    
    if use_onecycle:
        total_steps = len(train_loader) * config['imitation_learning']['epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['imitation_learning']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        print("✓ OneCycleLR scheduler enabled")
    
    # AMP Gradient Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Best model path
    best_model_path = base_dir / config['paths']['best_model_il']
    
    # Training loop
    print("\n=== Starting training ===")
    print("💾 Saving strategy: NO optimizer state (minimal file size)")
    print("📊 NEW: Tracking Policy Accuracy (Top-1, Top-3) and Value MAE")
    
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
        
        # Train
        train_losses, train_metrics = train_epoch_il(
            model, train_loader, optimizer, scheduler, config, device, scaler
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print train losses and metrics
        print(f"Train - Loss: {train_losses['total']:.4f}, "
              f"Policy: {train_losses['policy']:.4f}, "
              f"Value: {train_losses['value']:.4f}, "
              f"LR: {current_lr:.2e}")
        
        print(f"       📊 Top-1: {train_metrics['policy_top1_acc']:.2%}, "
              f"Top-3: {train_metrics['policy_top3_acc']:.2%}, "
              f"MAE: {train_metrics['value_mae']:.4f}")
        
        if use_mtl:
            print(f"       🆕 MTL - Win: {train_losses['win']:.4f}, "
                  f"Material: {train_losses['material']:.4f}, "
                  f"Check: {train_losses['check']:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            val_losses, val_metrics = evaluate_il(model, val_loader, config, device)
            
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
            logger.log(epoch + 1, train_losses, val_losses, train_metrics, val_metrics, current_lr)
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
                        'use_mtl': use_mtl
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
            logger.log(epoch + 1, train_losses, None, train_metrics, None, current_lr)
        
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
                    'use_mtl': use_mtl
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
    
    print("\n=== Training complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {il_dir}")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()