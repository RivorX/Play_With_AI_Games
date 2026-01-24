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
    """
    🆕 Logger for training metrics with MTL support
    """
    
    def __init__(self, log_dir, experiment_name="il_training", use_mtl=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_mtl = use_mtl
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_{timestamp}.png"
        
        # Initialize CSV with MTL columns
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'epoch', 'train_loss', 'train_policy_loss', 'train_value_loss',
                'val_loss', 'val_policy_loss', 'val_value_loss', 'learning_rate'
            ]
            
            if use_mtl:
                header.extend([
                    'train_win_loss', 'train_material_loss', 'train_check_loss',
                    'val_win_loss', 'val_material_loss', 'val_check_loss'
                ])
            
            writer.writerow(header)
        
        # Store metrics for plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_policy_losses = []
        self.val_policy_losses = []
        self.train_value_losses = []
        self.val_value_losses = []
        
        if use_mtl:
            self.train_win_losses = []
            self.val_win_losses = []
            self.train_material_losses = []
            self.val_material_losses = []
            self.train_check_losses = []
            self.val_check_losses = []
        
        print(f"📊 Logging to: {self.csv_path}")
    
    def log(self, epoch, train_losses, val_losses=None, lr=None):
        """
        Log metrics to CSV
        
        Args:
            epoch: Current epoch
            train_losses: Dict with train losses
            val_losses: Dict with validation losses (optional)
            lr: Learning rate (optional)
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            row = [
                epoch,
                train_losses['total'],
                train_losses['policy'],
                train_losses['value'],
                val_losses['total'] if val_losses else '',
                val_losses['policy'] if val_losses else '',
                val_losses['value'] if val_losses else '',
                lr if lr is not None else ''
            ]
            
            if self.use_mtl:
                row.extend([
                    train_losses.get('win', ''),
                    train_losses.get('material', ''),
                    train_losses.get('check', ''),
                    val_losses.get('win', '') if val_losses else '',
                    val_losses.get('material', '') if val_losses else '',
                    val_losses.get('check', '') if val_losses else ''
                ])
            
            writer.writerow(row)
        
        # Store for plotting
        self.epochs.append(epoch)
        self.train_losses.append(train_losses['total'])
        self.train_policy_losses.append(train_losses['policy'])
        self.train_value_losses.append(train_losses['value'])
        
        if val_losses is not None:
            self.val_losses.append(val_losses['total'])
            self.val_policy_losses.append(val_losses['policy'])
            self.val_value_losses.append(val_losses['value'])
        
        if self.use_mtl:
            self.train_win_losses.append(train_losses.get('win', 0))
            self.train_material_losses.append(train_losses.get('material', 0))
            self.train_check_losses.append(train_losses.get('check', 0))
            
            if val_losses is not None:
                self.val_win_losses.append(val_losses.get('win', 0))
                self.val_material_losses.append(val_losses.get('material', 0))
                self.val_check_losses.append(val_losses.get('check', 0))
    
    def plot(self):
        """Generate training plots"""
        if len(self.epochs) < 2:
            return
        
        # Determine plot layout
        if self.use_mtl:
            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        if not self.use_mtl:
            # Standard plots (2x2)
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
        
        else:
            # 🆕 MTL plots (3x3)
            val_epochs = [e for e in self.epochs if e <= len(self.val_losses)] if self.val_losses else []
            
            # Row 1: Main tasks
            # Total Loss
            ax = axes[0, 0]
            ax.plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
            if self.val_losses:
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Policy Loss
            ax = axes[0, 1]
            ax.plot(self.epochs, self.train_policy_losses, 'b-', label='Train', linewidth=2)
            if self.val_policy_losses:
                ax.plot(val_epochs, self.val_policy_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Policy Loss')
            ax.set_title('Policy Loss (Main)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Value Loss
            ax = axes[0, 2]
            ax.plot(self.epochs, self.train_value_losses, 'b-', label='Train', linewidth=2)
            if self.val_value_losses:
                ax.plot(val_epochs, self.val_value_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value Loss')
            ax.set_title('Value Loss (Main)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Row 2: Auxiliary tasks
            # Win prediction
            ax = axes[1, 0]
            ax.plot(self.epochs, self.train_win_losses, 'b-', label='Train', linewidth=2)
            if self.val_win_losses:
                ax.plot(val_epochs, self.val_win_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Win Loss')
            ax.set_title('🎯 Win Prediction (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Material prediction
            ax = axes[1, 1]
            ax.plot(self.epochs, self.train_material_losses, 'b-', label='Train', linewidth=2)
            if self.val_material_losses:
                ax.plot(val_epochs, self.val_material_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Material Loss')
            ax.set_title('⚖️ Material Count (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Check detection
            ax = axes[1, 2]
            ax.plot(self.epochs, self.train_check_losses, 'b-', label='Train', linewidth=2)
            if self.val_check_losses:
                ax.plot(val_epochs, self.val_check_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Check Loss')
            ax.set_title('👑 Check Detection (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Row 3: Comparisons
            # All main losses
            ax = axes[2, 0]
            ax.plot(self.epochs, self.train_losses, 'b-', label='Total', linewidth=2, alpha=0.7)
            ax.plot(self.epochs, self.train_policy_losses, 'g--', label='Policy', linewidth=1.5, alpha=0.7)
            ax.plot(self.epochs, self.train_value_losses, 'r--', label='Value', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Main Tasks (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # All auxiliary losses
            ax = axes[2, 1]
            ax.plot(self.epochs, self.train_win_losses, 'b-', label='Win', linewidth=2, alpha=0.7)
            ax.plot(self.epochs, self.train_material_losses, 'g-', label='Material', linewidth=2, alpha=0.7)
            ax.plot(self.epochs, self.train_check_losses, 'r-', label='Check', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Auxiliary Tasks (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Train vs Val (main)
            ax = axes[2, 2]
            if self.val_losses:
                ax.plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Total Loss')
                ax.set_title('Train vs Validation')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")


def train_epoch(model, train_loader, optimizer, scheduler, config, device, scaler):
    """
    🆕 Train for one epoch with Multi-Task Learning support
    """
    model.train()
    
    # Loss weights
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    # 🆕 MTL weights
    use_mtl = config['model'].get('use_multitask_learning', False)
    win_weight = config['model'].get('win_prediction_weight', 0.3) if use_mtl else 0
    material_weight = config['model'].get('material_prediction_weight', 0.2) if use_mtl else 0
    check_weight = config['model'].get('check_prediction_weight', 0.15) if use_mtl else 0
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    
    # 🆕 MTL losses
    total_win_loss = 0
    total_material_loss = 0
    total_check_loss = 0
    
    criterion_policy = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    criterion_value = nn.MSELoss()
    
    # 🆕 MTL criteria (BCEWithLogitsLoss is AMP-safe!)
    if use_mtl:
        criterion_win = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE
        criterion_material = nn.MSELoss()
        criterion_check = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch (handle both dict and tuple format)
        if isinstance(batch_data, dict):
            boards = batch_data['board']
            moves = batch_data['move']
            outcomes = batch_data['value']
            
            # 🆕 MTL targets
            if use_mtl:
                win_targets = batch_data['win']
                material_targets = batch_data['material']
                check_targets = batch_data['check']
        else:
            # Old tuple format (backwards compatibility)
            boards, moves, outcomes = batch_data
            use_mtl = False
        
        # Move to device
        boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        outcomes = outcomes.to(device, non_blocking=True)
        
        if use_mtl:
            win_targets = win_targets.to(device, non_blocking=True)
            material_targets = material_targets.to(device, non_blocking=True)
            check_targets = check_targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Training
        use_amp = config['hardware'].get('use_amp', True)
        amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            # 🆕 Forward pass with MTL
            if use_mtl:
                policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
            else:
                policy_pred, value_pred = model(boards, return_aux=False)
            
            # Main losses
            policy_loss = criterion_policy(policy_pred, moves)
            value_loss = criterion_value(value_pred, outcomes)
            
            # Total loss starts with main tasks
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            # 🆕 Add auxiliary losses
            if use_mtl:
                win_loss = criterion_win(win_pred, win_targets)
                material_loss = criterion_material(material_pred, material_targets)
                check_loss = criterion_check(check_pred, check_targets)
                
                loss = loss + win_weight * win_loss + material_weight * material_loss + check_weight * check_loss
                
                total_win_loss += win_loss.item()
                total_material_loss += material_loss.item()
                total_check_loss += check_loss.item()
        
        # Backward pass
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
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Track losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        # Progress bar
        if batch_idx % config['logging']['print_every'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            postfix = {
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            }
            
            # 🆕 Add MTL losses to progress bar
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
    
    n = len(train_loader)
    
    # 🆕 Return all losses as dict
    result = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n
    }
    
    if use_mtl:
        result.update({
            'win': total_win_loss / n,
            'material': total_material_loss / n,
            'check': total_check_loss / n
        })
    
    return result


def evaluate(model, val_loader, config, device):
    """
    🆕 Evaluate model with MTL support
    """
    model.eval()
    
    policy_weight = config['imitation_learning']['policy_loss_weight']
    value_weight = config['imitation_learning']['value_loss_weight']
    label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
    
    # 🆕 MTL weights
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
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating"):
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
            
            # Move to device
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
                # Forward pass
                if use_mtl:
                    policy_pred, value_pred, win_pred, material_pred, check_pred = model(boards, return_aux=True)
                else:
                    policy_pred, value_pred = model(boards, return_aux=False)
                
                # Main losses
                policy_loss = criterion_policy(policy_pred, moves)
                value_loss = criterion_value(value_pred, outcomes)
                
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                # MTL losses
                if use_mtl:
                    win_loss = criterion_win(win_pred, win_targets)
                    material_loss = criterion_material(material_pred, material_targets)
                    check_loss = criterion_check(check_pred, check_targets)
                    
                    loss = loss + win_weight * win_loss + material_weight * material_loss + check_weight * check_loss
                    
                    total_win_loss += win_loss.item()
                    total_material_loss += material_loss.item()
                    total_check_loss += check_loss.item()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    n = len(val_loader)
    
    result = {
        'total': total_loss / n,
        'policy': total_policy_loss / n,
        'value': total_value_loss / n
    }
    
    if use_mtl:
        result.update({
            'win': total_win_loss / n,
            'material': total_material_loss / n,
            'check': total_check_loss / n
        })
    
    return result


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
    logger = TrainingLogger(logs_dir, experiment_name="il_training", use_mtl=use_mtl)
    
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
    max_patience = config['imitation_learning'].get('max_patience', 15)
    min_delta = config['imitation_learning']['min_delta']
    checkpoint_every = config['imitation_learning'].get('checkpoint_every', 5)
    
    print(f"Early stopping: patience={max_patience}, min_delta={min_delta}")
    print(f"Checkpoints saved every {checkpoint_every} epochs to: {il_dir}")
    print(f"Best model saved to: {best_model_path}")
    
    for epoch in range(config['imitation_learning']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['imitation_learning']['epochs']}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, config, device, scaler
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print train losses
        print(f"Train - Loss: {train_losses['total']:.4f}, "
              f"Policy: {train_losses['policy']:.4f}, "
              f"Value: {train_losses['value']:.4f}, "
              f"LR: {current_lr:.2e}")
        
        if use_mtl:
            print(f"       🆕 MTL - Win: {train_losses['win']:.4f}, "
                  f"Material: {train_losses['material']:.4f}, "
                  f"Check: {train_losses['check']:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['imitation_learning']['eval_every'] == 0:
            val_losses = evaluate(model, val_loader, config, device)
            
            print(f"Val - Loss: {val_losses['total']:.4f}, "
                  f"Policy: {val_losses['policy']:.4f}, "
                  f"Value: {val_losses['value']:.4f}")
            
            if use_mtl:
                print(f"     🆕 MTL - Win: {val_losses['win']:.4f}, "
                      f"Material: {val_losses['material']:.4f}, "
                      f"Check: {val_losses['check']:.4f}")
            
            # Log metrics
            logger.log(epoch + 1, train_losses, val_losses, current_lr)
            logger.plot()
            
            # Save best model
            improvement = best_val_loss - val_losses['total']
            if improvement > min_delta:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                print(f"✓ New best model! Val loss: {val_losses['total']:.4f} "
                      f"(improved by {improvement:.4f})")
                
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
            logger.log(epoch + 1, train_losses, lr=current_lr)
        
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0:
            # Get current val loss if not already computed
            if (epoch + 1) % config['imitation_learning']['eval_every'] != 0:
                val_losses = evaluate(model, val_loader, config, device)
            
            checkpoint_name = f"il_epoch_{epoch+1}_valloss_{val_losses['total']:.4f}.pt"
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
                    'use_mtl': use_mtl
                },
                save_optimizer=False
            )
            
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