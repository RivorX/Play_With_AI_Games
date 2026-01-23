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
import csv
import matplotlib.pyplot as plt
from datetime import datetime

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
        num_classes = log_probs.size(-1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


class TrainingLogger:
    """Logger for training metrics with CSV and plotting"""
    
    def __init__(self, log_dir, experiment_name="il_training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_{timestamp}.png"
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_policy_loss', 'train_value_loss',
                'val_loss', 'val_policy_loss', 'val_value_loss', 'learning_rate'
            ])
        
        # Store metrics for plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_policy_losses = []
        self.val_policy_losses = []
        self.train_value_losses = []
        self.val_value_losses = []
        
        print(f"📊 Logging to: {self.csv_path}")
    
    def log(self, epoch, train_loss, train_policy, train_value, 
            val_loss=None, val_policy=None, val_value=None, lr=None):
        """Log metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_policy, train_value,
                val_loss if val_loss is not None else '',
                val_policy if val_policy is not None else '',
                val_value if val_value is not None else '',
                lr if lr is not None else ''
            ])
        
        # Store for plotting
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_policy_losses.append(train_policy)
        self.train_value_losses.append(train_value)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.val_policy_losses.append(val_policy)
            self.val_value_losses.append(val_value)
    
    def plot(self):
        """Generate training plots"""
        if len(self.epochs) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        ax.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        if self.val_losses:
            val_epochs = [e for e in self.epochs if e <= len(self.val_losses)]
            ax.plot(val_epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Policy Loss
        ax = axes[0, 1]
        ax.plot(self.epochs, self.train_policy_losses, 'b-', label='Train Policy', linewidth=2)
        if self.val_policy_losses:
            val_epochs = [e for e in self.epochs if e <= len(self.val_policy_losses)]
            ax.plot(val_epochs, self.val_policy_losses, 'r-', label='Val Policy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Value Loss
        ax = axes[1, 0]
        ax.plot(self.epochs, self.train_value_losses, 'b-', label='Train Value', linewidth=2)
        if self.val_value_losses:
            val_epochs = [e for e in self.epochs if e <= len(self.val_value_losses)]
            ax.plot(val_epochs, self.val_value_losses, 'r-', label='Val Value', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss Comparison
        ax = axes[1, 1]
        if self.val_losses:
            ax.plot(self.epochs, self.train_losses, 'b-', label='Train Total', linewidth=2, alpha=0.7)
            val_epochs = [e for e in self.epochs if e <= len(self.val_losses)]
            ax.plot(val_epochs, self.val_losses, 'r-', label='Val Total', linewidth=2, alpha=0.7)
            ax.plot(self.epochs, self.train_policy_losses, 'b--', label='Train Policy', linewidth=1.5, alpha=0.5)
            ax.plot(val_epochs, self.val_policy_losses, 'r--', label='Val Policy', linewidth=1.5, alpha=0.5)
            ax.plot(self.epochs, self.train_value_losses, 'b:', label='Train Value', linewidth=1.5, alpha=0.5)
            ax.plot(val_epochs, self.val_value_losses, 'r:', label='Val Value', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('All Losses Comparison')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")


def train_epoch(model, train_loader, optimizer, scheduler, config, device, scaler):
    """Train for one epoch with AMP and optimized data loading"""
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
        # Convert to channels_last for better GPU performance
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        outcomes = outcomes.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Training
        use_amp = config['hardware'].get('use_amp', True)
        amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            policy_pred, value_pred = model(boards)
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
        if batch_idx % 200 == 0:
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
            boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            outcomes = outcomes.to(device, non_blocking=True)
            
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
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    print("✓ cuDNN benchmark mode enabled")
    
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
        
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            print("⚠️  bfloat16 requested but not supported, falling back to float16")
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
    
    # Initialize logger
    logger = TrainingLogger(logs_dir, experiment_name="il_training")
    
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
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = config['imitation_learning'].get('max_patience', 10)
    min_delta = config['imitation_learning']['min_delta']
    checkpoint_every = config['imitation_learning'].get('checkpoint_every', 5)
    
    print(f"Early stopping: patience={max_patience}, min_delta={min_delta}")
    print(f"Checkpoints saved every {checkpoint_every} epochs to: {il_dir}")
    print(f"Best model saved to: {best_model_path}")
    
    for epoch in range(config['imitation_learning']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['imitation_learning']['epochs']}")
        
        # Train
        train_loss, train_policy, train_value = train_epoch(
            model, train_loader, optimizer, scheduler, config, device, scaler
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train - Loss: {train_loss:.4f}, Policy: {train_policy:.4f}, Value: {train_value:.4f}, LR: {current_lr:.2e}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            val_loss, val_policy, val_value = evaluate(model, val_loader, config, device)
            print(f"Val - Loss: {val_loss:.4f}, Policy: {val_policy:.4f}, Value: {val_value:.4f}")
            
            # Log metrics
            logger.log(epoch + 1, train_loss, train_policy, train_value, 
                      val_loss, val_policy, val_value, current_lr)
            logger.plot()
            
            # Save best model (overwrite previous)
            improvement = best_val_loss - val_loss
            if improvement > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                
                print(f"✓ New best model! Val loss: {val_loss:.4f} (improved by {improvement:.4f})")
                
                model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
                save_checkpoint(
                    model_to_save, 
                    None,  # ← BEZ optimizer
                    epoch, 
                    val_loss,
                    str(best_model_path),
                    {
                        'val_policy_loss': val_policy, 
                        'val_value_loss': val_value,
                        'use_amp': use_amp,
                        'use_bfloat16': use_bfloat16,
                    },
                    save_optimizer=False
                )
                
                # Restore original dtype
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
            logger.log(epoch + 1, train_loss, train_policy, train_value, lr=current_lr)
        
        # Save checkpoint every N epochs (ALSO without optimizer)
        if (epoch + 1) % checkpoint_every == 0:
            # Get current val loss
            if (epoch + 1) % config['imitation_learning']['eval_every'] != 0:
                val_loss, val_policy, val_value = evaluate(model, val_loader, config, device)
            
            checkpoint_name = f"il_epoch_{epoch+1}_valloss_{val_loss:.4f}.pt"
            checkpoint_path = il_dir / checkpoint_name
            
            model_to_save = model.to(torch.bfloat16) if use_bfloat16 else model
            save_checkpoint(
                model_to_save, 
                None,  # ← BEZ optimizer (również w checkpointach!)
                epoch, 
                train_loss,
                str(checkpoint_path),
                {
                    'val_loss': val_loss,
                    'val_policy_loss': val_policy,
                    'val_value_loss': val_value
                },
                save_optimizer=False  # ← BEZ optimizer
            )
            
            # Restore dtype
            if use_bfloat16:
                model = model.to(torch.float32)
            
            size_mb = checkpoint_path.stat().st_size / (1024**2)
            print(f"💾 Checkpoint saved: {checkpoint_path} ({size_mb:.1f} MB)")
        
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