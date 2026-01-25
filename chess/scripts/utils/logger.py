"""
Unified training logger for both IL and RL training
"""

import csv
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Universal logger for training metrics
    Supports both IL and RL modes with optional MTL and detailed metrics
    """
    
    def __init__(self, log_dir, experiment_name="training", mode="il", use_mtl=False):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            mode: "il" or "rl"
            use_mtl: Whether using multi-task learning
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.use_mtl = use_mtl
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.plot_path = self.log_dir / f"{experiment_name}_{timestamp}.png"
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if mode == "il":
                header = [
                    'epoch', 'train_loss', 'train_policy_loss', 'train_value_loss',
                    'val_loss', 'val_policy_loss', 'val_value_loss', 'learning_rate',
                    # 📊 NEW: Metrics
                    'train_policy_top1', 'train_policy_top3', 'train_value_mae',
                    'val_policy_top1', 'val_policy_top3', 'val_value_mae'
                ]
                
                if use_mtl:
                    header.extend([
                        'train_win_loss', 'train_material_loss', 'train_check_loss',
                        'val_win_loss', 'val_material_loss', 'val_check_loss'
                    ])
            
            else:  # RL mode
                header = [
                    'iteration', 'avg_loss', 'policy_loss', 'value_loss',
                    'win_rate', 'buffer_size', 'avg_game_length', 'positions_per_sec',
                    'selfplay_time', 'data_collection_time', 'temperature', 'beta',
                    # 📊 NEW: Metrics
                    'policy_top1_acc', 'policy_top3_acc', 'value_mae'
                ]
            
            writer.writerow(header)
        
        # Storage for plotting
        self.iterations = []
        self.train_losses = []
        self.val_losses = []
        self.train_policy_losses = []
        self.val_policy_losses = []
        self.train_value_losses = []
        self.val_value_losses = []
        
        # 📊 NEW: Metrics storage
        self.train_policy_top1 = []
        self.train_policy_top3 = []
        self.train_value_mae = []
        self.val_policy_top1 = []
        self.val_policy_top3 = []
        self.val_value_mae = []
        
        if use_mtl:
            self.train_win_losses = []
            self.val_win_losses = []
            self.train_material_losses = []
            self.val_material_losses = []
            self.train_check_losses = []
            self.val_check_losses = []
        
        if mode == "rl":
            self.win_rates = []
            self.temperatures = []
        
        print(f"📊 Logging to: {self.csv_path}")
    
    def log(self, iteration, train_losses=None, val_losses=None, 
            train_metrics=None, val_metrics=None, lr=None, **kwargs):
        """
        Log metrics to CSV
        
        Args:
            iteration: Current epoch/iteration
            train_losses: Dict with train losses (IL mode)
            val_losses: Dict with validation losses (IL mode, optional)
            train_metrics: Dict with train metrics (NEW)
            val_metrics: Dict with validation metrics (NEW)
            lr: Learning rate (IL mode, optional)
            **kwargs: Additional metrics (RL mode)
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if self.mode == "il":
                row = [
                    iteration,
                    train_losses['total'],
                    train_losses['policy'],
                    train_losses['value'],
                    val_losses['total'] if val_losses else '',
                    val_losses['policy'] if val_losses else '',
                    val_losses['value'] if val_losses else '',
                    lr if lr is not None else '',
                    # 📊 NEW: Metrics
                    train_metrics.get('policy_top1_acc', '') if train_metrics else '',
                    train_metrics.get('policy_top3_acc', '') if train_metrics else '',
                    train_metrics.get('value_mae', '') if train_metrics else '',
                    val_metrics.get('policy_top1_acc', '') if val_metrics else '',
                    val_metrics.get('policy_top3_acc', '') if val_metrics else '',
                    val_metrics.get('value_mae', '') if val_metrics else ''
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
                
                # Store for plotting
                self.iterations.append(iteration)
                self.train_losses.append(train_losses['total'])
                self.train_policy_losses.append(train_losses['policy'])
                self.train_value_losses.append(train_losses['value'])
                
                if train_metrics:
                    self.train_policy_top1.append(train_metrics.get('policy_top1_acc', 0))
                    self.train_policy_top3.append(train_metrics.get('policy_top3_acc', 0))
                    self.train_value_mae.append(train_metrics.get('value_mae', 0))
                
                if val_losses is not None:
                    self.val_losses.append(val_losses['total'])
                    self.val_policy_losses.append(val_losses['policy'])
                    self.val_value_losses.append(val_losses['value'])
                
                if val_metrics:
                    self.val_policy_top1.append(val_metrics.get('policy_top1_acc', 0))
                    self.val_policy_top3.append(val_metrics.get('policy_top3_acc', 0))
                    self.val_value_mae.append(val_metrics.get('value_mae', 0))
                
                if self.use_mtl:
                    self.train_win_losses.append(train_losses.get('win', 0))
                    self.train_material_losses.append(train_losses.get('material', 0))
                    self.train_check_losses.append(train_losses.get('check', 0))
                    
                    if val_losses is not None:
                        self.val_win_losses.append(val_losses.get('win', 0))
                        self.val_material_losses.append(val_losses.get('material', 0))
                        self.val_check_losses.append(val_losses.get('check', 0))
            
            else:  # RL mode
                row = [
                    iteration,
                    kwargs.get('avg_loss', ''),
                    kwargs.get('policy_loss', ''),
                    kwargs.get('value_loss', ''),
                    kwargs.get('win_rate', ''),
                    kwargs.get('buffer_size', ''),
                    kwargs.get('avg_game_length', ''),
                    kwargs.get('positions_per_sec', ''),
                    kwargs.get('selfplay_time', ''),
                    kwargs.get('data_collection_time', ''),
                    kwargs.get('temperature', ''),
                    kwargs.get('beta', ''),
                    # 📊 NEW: Metrics
                    train_metrics.get('policy_top1_acc', '') if train_metrics else '',
                    train_metrics.get('policy_top3_acc', '') if train_metrics else '',
                    train_metrics.get('value_mae', '') if train_metrics else ''
                ]
                
                # Store for plotting
                self.iterations.append(iteration)
                if 'avg_loss' in kwargs:
                    self.train_losses.append(kwargs['avg_loss'])
                    self.train_policy_losses.append(kwargs.get('policy_loss', 0))
                    self.train_value_losses.append(kwargs.get('value_loss', 0))
                
                if train_metrics:
                    self.train_policy_top1.append(train_metrics.get('policy_top1_acc', 0))
                    self.train_policy_top3.append(train_metrics.get('policy_top3_acc', 0))
                    self.train_value_mae.append(train_metrics.get('value_mae', 0))
                
                if 'win_rate' in kwargs and kwargs['win_rate'] is not None:
                    self.win_rates.append((iteration, kwargs['win_rate']))
                
                if 'temperature' in kwargs and kwargs['temperature'] is not None:
                    self.temperatures.append((iteration, kwargs['temperature']))
            
            writer.writerow(row)
    
    def plot(self):
        """Generate training plots"""
        if len(self.iterations) < 2:
            return
        
        if self.mode == "il":
            self._plot_il()
        else:
            self._plot_rl()
    
    def _plot_il(self):
        """Plot IL training progress"""
        if self.use_mtl:
            fig, axes = plt.subplots(4, 3, figsize=(18, 18))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(15, 14))
        
        fig.suptitle('IL Training Progress', fontsize=16, fontweight='bold')
        
        val_epochs = [e for e in self.iterations if e <= len(self.val_losses)] if self.val_losses else []
        
        if not self.use_mtl:
            # ============================================================
            # ROW 1: LOSSES
            # ============================================================
            
            # Total Loss
            ax = axes[0, 0]
            ax.plot(self.iterations, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            if self.val_losses:
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Policy Loss
            ax = axes[0, 1]
            ax.plot(self.iterations, self.train_policy_losses, 'b-', label='Train Policy', linewidth=2)
            if self.val_policy_losses:
                ax.plot(val_epochs, self.val_policy_losses, 'r-', label='Val Policy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Policy Loss')
            ax.set_title('Policy Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ============================================================
            # ROW 2: METRICS
            # ============================================================
            
            # Policy Accuracy
            ax = axes[1, 0]
            if self.train_policy_top1:
                ax.plot(self.iterations, self.train_policy_top1, 'b-', label='Train Top-1', linewidth=2)
                ax.plot(self.iterations, self.train_policy_top3, 'b--', label='Train Top-3', linewidth=2, alpha=0.7)
            if self.val_policy_top1:
                ax.plot(val_epochs, self.val_policy_top1, 'r-', label='Val Top-1', linewidth=2)
                ax.plot(val_epochs, self.val_policy_top3, 'r--', label='Val Top-3', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('📊 Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Value MAE
            ax = axes[1, 1]
            if self.train_value_mae:
                ax.plot(self.iterations, self.train_value_mae, 'b-', label='Train MAE', linewidth=2)
            if self.val_value_mae:
                ax.plot(val_epochs, self.val_value_mae, 'r-', label='Val MAE', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE')
            ax.set_title('📊 Value MAE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ============================================================
            # ROW 3: COMPARISON
            # ============================================================
            
            # Value Loss
            ax = axes[2, 0]
            ax.plot(self.iterations, self.train_value_losses, 'b-', label='Train Value', linewidth=2)
            if self.val_value_losses:
                ax.plot(val_epochs, self.val_value_losses, 'r-', label='Val Value', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value Loss')
            ax.set_title('Value Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Loss Comparison
            ax = axes[2, 1]
            if self.val_losses:
                ax.plot(self.iterations, self.train_losses, 'b-', label='Train Total', linewidth=2, alpha=0.7)
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val Total', linewidth=2, alpha=0.7)
                ax.plot(self.iterations, self.train_policy_losses, 'b--', label='Train Policy', linewidth=1.5, alpha=0.5)
                ax.plot(val_epochs, self.val_policy_losses, 'r--', label='Val Policy', linewidth=1.5, alpha=0.5)
                ax.plot(self.iterations, self.train_value_losses, 'b:', label='Train Value', linewidth=1.5, alpha=0.5)
                ax.plot(val_epochs, self.val_value_losses, 'r:', label='Val Value', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('All Losses Comparison')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        else:
            # MTL plots (4x3)
            
            # Row 1: Main tasks
            ax = axes[0, 0]
            ax.plot(self.iterations, self.train_losses, 'b-', label='Train', linewidth=2)
            if self.val_losses:
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[0, 1]
            ax.plot(self.iterations, self.train_policy_losses, 'b-', label='Train', linewidth=2)
            if self.val_policy_losses:
                ax.plot(val_epochs, self.val_policy_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Policy Loss')
            ax.set_title('Policy Loss (Main)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[0, 2]
            ax.plot(self.iterations, self.train_value_losses, 'b-', label='Train', linewidth=2)
            if self.val_value_losses:
                ax.plot(val_epochs, self.val_value_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value Loss')
            ax.set_title('Value Loss (Main)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Row 2: Auxiliary tasks
            ax = axes[1, 0]
            ax.plot(self.iterations, self.train_win_losses, 'b-', label='Train', linewidth=2)
            if self.val_win_losses:
                ax.plot(val_epochs, self.val_win_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Win Loss')
            ax.set_title('🎯 Win Prediction (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            ax.plot(self.iterations, self.train_material_losses, 'b-', label='Train', linewidth=2)
            if self.val_material_losses:
                ax.plot(val_epochs, self.val_material_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Material Loss')
            ax.set_title('⚖️ Material Count (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 2]
            ax.plot(self.iterations, self.train_check_losses, 'b-', label='Train', linewidth=2)
            if self.val_check_losses:
                ax.plot(val_epochs, self.val_check_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Check Loss')
            ax.set_title('👑 Check Detection (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Row 3: Metrics
            ax = axes[2, 0]
            if self.train_policy_top1:
                ax.plot(self.iterations, self.train_policy_top1, 'b-', label='Train Top-1', linewidth=2)
                ax.plot(self.iterations, self.train_policy_top3, 'b--', label='Train Top-3', linewidth=2, alpha=0.7)
            if self.val_policy_top1:
                ax.plot(val_epochs, self.val_policy_top1, 'r-', label='Val Top-1', linewidth=2)
                ax.plot(val_epochs, self.val_policy_top3, 'r--', label='Val Top-3', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('📊 Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[2, 1]
            if self.train_value_mae:
                ax.plot(self.iterations, self.train_value_mae, 'b-', label='Train', linewidth=2)
            if self.val_value_mae:
                ax.plot(val_epochs, self.val_value_mae, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE')
            ax.set_title('📊 Value MAE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[2, 2]
            ax.plot(self.iterations, self.train_losses, 'b-', label='Total', linewidth=2, alpha=0.7)
            ax.plot(self.iterations, self.train_policy_losses, 'g--', label='Policy', linewidth=1.5, alpha=0.7)
            ax.plot(self.iterations, self.train_value_losses, 'r--', label='Value', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Main Tasks (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Row 4: Comparisons
            ax = axes[3, 0]
            ax.plot(self.iterations, self.train_win_losses, 'b-', label='Win', linewidth=2, alpha=0.7)
            ax.plot(self.iterations, self.train_material_losses, 'g-', label='Material', linewidth=2, alpha=0.7)
            ax.plot(self.iterations, self.train_check_losses, 'r-', label='Check', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Auxiliary Tasks (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[3, 1]
            if self.val_losses:
                ax.plot(self.iterations, self.train_losses, 'b-', label='Train', linewidth=2)
                ax.plot(val_epochs, self.val_losses, 'r-', label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Total Loss')
                ax.set_title('Train vs Validation')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Summary metrics
            ax = axes[3, 2]
            ax.axis('off')
            if self.val_policy_top1 and len(self.val_policy_top1) > 0:
                summary_text = (
                    f"Latest Validation Metrics:\n\n"
                    f"Policy Top-1: {self.val_policy_top1[-1]:.2%}\n"
                    f"Policy Top-3: {self.val_policy_top3[-1]:.2%}\n"
                    f"Value MAE: {self.val_value_mae[-1]:.4f}\n"
                    f"Total Loss: {self.val_losses[-1]:.4f}\n"
                )
                ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")
    
    def _plot_rl(self):
        """Plot RL training progress"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')
        
        # Row 1: Losses
        ax = axes[0, 0]
        ax.plot(self.iterations, self.train_losses, 'b-', label='Total Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(self.iterations, self.train_policy_losses, 'g-', label='Policy Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        ax.plot(self.iterations, self.train_value_losses, 'r-', label='Value Loss', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 2: Metrics
        ax = axes[1, 0]
        if self.train_policy_top1:
            ax.plot(self.iterations, self.train_policy_top1, 'b-', label='Top-1', linewidth=2)
            ax.plot(self.iterations, self.train_policy_top3, 'b--', label='Top-3', linewidth=2, alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Accuracy')
            ax.set_title('📊 Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        if self.train_value_mae:
            ax.plot(self.iterations, self.train_value_mae, 'r-', label='Value MAE', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('MAE')
            ax.set_title('📊 Value MAE')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        if self.win_rates:
            win_iters, win_vals = zip(*self.win_rates)
            ax.plot(win_iters, win_vals, 'mo-', label='Win Rate', linewidth=2, markersize=8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate vs Best')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 3: Additional
        ax = axes[2, 0]
        if self.temperatures:
            temp_iters, temp_vals = zip(*self.temperatures)
            ax.plot(temp_iters, temp_vals, 'orange', linewidth=2, label='Temperature')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Temperature')
            ax.set_title('🌡️ Temperature Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[2, 1]
        ax.plot(self.iterations, self.train_losses, 'b-', label='Total', linewidth=2, alpha=0.7)
        ax.plot(self.iterations, self.train_policy_losses, 'g--', label='Policy', linewidth=1.5, alpha=0.7)
        ax.plot(self.iterations, self.train_value_losses, 'r--', label='Value', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('All Losses Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary
        ax = axes[2, 2]
        ax.axis('off')
        if len(self.iterations) > 0:
            summary_text = (
                f"Latest Metrics:\n\n"
                f"Policy Top-1: {self.train_policy_top1[-1]:.2%}\n" if self.train_policy_top1 else "" +
                f"Policy Top-3: {self.train_policy_top3[-1]:.2%}\n" if self.train_policy_top3 else "" +
                f"Value MAE: {self.train_value_mae[-1]:.4f}\n" if self.train_value_mae else "" +
                f"Total Loss: {self.train_losses[-1]:.4f}\n" if self.train_losses else ""
            )
            ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                   verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")