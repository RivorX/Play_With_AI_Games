import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gc

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, save_checkpoint
from src.data import parse_pgn_files, create_dataloaders


class LabelSmoothingNLLLoss(nn.Module):
    """NLL Loss with label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, log_probs, targets):
        # log_probs: (batch, num_classes) in log space
        # targets: (batch,) class indices
        
        num_classes = log_probs.size(-1)
        
        # One-hot encode targets
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


def train_epoch(model, train_loader, optimizer, scheduler, config, device, scaler):
    """Train for one epoch with AMP"""
    model.train()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    criterion_value = nn.MSELoss()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (boards, moves, outcomes) in enumerate(pbar):
        boards = boards.to(device)
        moves = moves.to(device)
        outcomes = outcomes.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training (updated API)
        use_amp = config['hardware'].get('use_amp', True)
        amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            policy_pred, value_pred = model(boards)
            
            # Compute losses
            policy_loss = criterion_policy(policy_pred, moves)
            value_loss = criterion_value(value_pred, outcomes)
            
            loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['imitation_learning']['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['imitation_learning']['grad_clip']
            )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()
        
        # Track losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        if batch_idx % config['logging']['print_every'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Periodic garbage collection
        if batch_idx % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    n = len(train_loader)
    return total_loss / n, total_policy_loss / n, total_value_loss / n


def evaluate(model, val_loader, config, device):
    """Evaluate model on validation set"""
    model.eval()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    criterion_value = nn.MSELoss()
    
    with torch.no_grad():
        for boards, moves, outcomes in tqdm(val_loader, desc="Evaluating"):
            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)
            
            # Use AMP for inference too (updated API)
            use_amp = config['hardware'].get('use_amp', True)
            amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
            
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                policy_pred, value_pred = model(boards)
                
                policy_loss = criterion_policy(policy_pred, moves)
                value_loss = criterion_value(value_pred, outcomes)
                
                loss = policy_weight * policy_loss + value_weight * value_loss
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    n = len(val_loader)
    return total_loss / n, total_policy_loss / n, total_value_loss / n


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
    
    # Check AMP availability
    use_amp = config['hardware'].get('use_amp', True)
    use_bfloat16 = config['hardware'].get('use_bfloat16', False)
    
    if use_amp and not torch.cuda.is_available():
        print("⚠️  AMP requested but CUDA not available, disabling AMP")
        use_amp = False
        use_bfloat16 = False
    
    if use_amp:
        amp_dtype = "bfloat16" if use_bfloat16 else "float16"
        print(f"✓ Mixed Precision Training (AMP) enabled with {amp_dtype}")
        
        # Check bfloat16 support
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            print("⚠️  bfloat16 requested but not supported on this GPU, falling back to float16")
            use_bfloat16 = False
            config['hardware']['use_bfloat16'] = False
    
    # Create directories
    base_dir = script_dir.parent
    models_dir = base_dir / config['paths']['models_dir']
    logs_dir = base_dir / config['paths']['logs_dir']
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("\n=== Loading data ===")
    
    preprocessing_dir = base_dir / "data" / "preprocessing"
    min_elo = config['data']['min_elo']
    max_games = config['data']['max_games']
    max_moves = config['data']['max_moves_per_game']
    
    binary_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}.bin"
    metadata_file = preprocessing_dir / f"positions_elo{min_elo}_games{max_games}_moves{max_moves}_meta.pkl"
    
    use_cache = True
    if binary_file.exists() and metadata_file.exists():
        file_size_mb = binary_file.stat().st_size / (1024**2)
        response = input(f"\nFound cached binary data:\n{binary_file}\nSize: {file_size_mb:.2f} MB\n\nUse cached data? (y/n): ").strip().lower()
        use_cache = response == 'y'
        
        if not use_cache:
            print("Will reprocess PGN files...")
            binary_file.unlink()
            metadata_file.unlink()
            print(f"Deleted cache files")
    
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    if config['model'].get('use_se_blocks', True):
        print("✓ SE-Blocks enabled for better feature learning")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['imitation_learning']['learning_rate'],
        weight_decay=config['imitation_learning']['weight_decay']
    )
    
    # Learning rate scheduler - OneCycleLR for better convergence
    use_onecycle = config['imitation_learning'].get('use_onecycle_lr', True)
    scheduler = None
    
    if use_onecycle:
        total_steps = len(train_loader) * config['imitation_learning']['epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['imitation_learning']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / 25
            final_div_factor=10000.0  # min_lr = max_lr / 10000
        )
        print("✓ OneCycleLR scheduler enabled")
    
    # AMP Gradient Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Label smoothing
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    if label_smoothing > 0:
        print(f"✓ Label smoothing enabled (α={label_smoothing})")
    
    # Training loop
    print("\n=== Starting training ===")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = config['imitation_learning'].get('max_patience', 10)
    min_delta = config['imitation_learning']['min_delta']
    
    print(f"Early stopping: patience={max_patience}, min_delta={min_delta}")
    
    for epoch in range(config['imitation_learning']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['imitation_learning']['epochs']}")
        
        # Train
        train_loss, train_policy, train_value = train_epoch(
            model, train_loader, optimizer, scheduler, config, device, scaler
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Policy: {train_policy:.4f}, Value: {train_value:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            val_loss, val_policy, val_value = evaluate(model, val_loader, config, device)
            print(f"Val - Loss: {val_loss:.4f}, Policy: {val_policy:.4f}, Value: {val_value:.4f}")
            
            # Save best model
            improvement = best_val_loss - val_loss
            if improvement > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_path = base_dir / config['paths']['best_model']
                
                # Convert model to bfloat16 if enabled for storage efficiency
                if use_bfloat16:
                    model_to_save = model.to(torch.bfloat16)
                    save_checkpoint(
                        model_to_save, optimizer, epoch, val_loss,
                        str(best_model_path),
                        {
                            'val_policy_loss': val_policy, 
                            'val_value_loss': val_value,
                            'use_amp': use_amp,
                            'use_bfloat16': use_bfloat16,
                            'label_smoothing': label_smoothing
                        }
                    )
                    # Move model back to original dtype
                    model = model.to(device).to(torch.float32)
                else:
                    save_checkpoint(
                        model, optimizer, epoch, val_loss,
                        str(best_model_path),
                        {
                            'val_policy_loss': val_policy, 
                            'val_value_loss': val_value,
                            'use_amp': use_amp,
                            'use_bfloat16': use_bfloat16,
                            'label_smoothing': label_smoothing
                        }
                    )
                
                print(f"✓ New best model saved! Val loss: {val_loss:.4f} (improved by {improvement:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{max_patience}")
            
            # Early stopping with extended patience
            if patience_counter >= max_patience:
                print(f"\n🛑 Early stopping triggered after {patience_counter} epochs without improvement!")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        # Save checkpoint
        if (epoch + 1) % config['imitation_learning']['save_every'] == 0:
            checkpoint_path = base_dir / config['paths']['il_checkpoint']
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                str(checkpoint_path)
            )
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n=== Training complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {base_dir / config['paths']['best_model']}")
    
    # Print training summary
    print("\n" + "="*60)
    print("Training Configuration Summary:")
    print(f"  • Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    if use_amp:
        amp_dtype = "bfloat16" if use_bfloat16 else "float16"
        print(f"  • AMP Dtype: {amp_dtype}")
    print(f"  • Label Smoothing: {label_smoothing}")
    print(f"  • OneCycleLR: {'Enabled' if use_onecycle else 'Disabled'}")
    print(f"  • SE-Blocks: {'Enabled' if config['model'].get('use_se_blocks', True) else 'Disabled'}")
    print(f"  • Early Stopping Patience: {max_patience}")
    print("="*60)


if __name__ == "__main__":
    main()