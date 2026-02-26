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
                    'train_policy_top1', 'train_policy_top3',
                    'train_value_mae', 'train_value_mae_weighted',
                    'train_value_wdl_acc', 'train_value_wdl_ce',
                    'val_policy_top1', 'val_policy_top3',
                    'val_value_mae', 'val_value_mae_weighted',
                    'val_value_wdl_acc', 'val_value_wdl_ce',
                    # 🆕 Elo estimation
                    'estimated_elo'
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
                    'policy_top1_acc', 'policy_top3_acc',
                    'value_mae', 'value_mae_weighted',
                    'value_wdl_acc', 'value_wdl_ce'
                ]
            
            writer.writerow(header)
        
        # Storage for plotting
        self.iterations = []
        self.train_losses = []
        self.val_losses = []
        self.val_iterations = []
        self.train_policy_losses = []
        self.val_policy_losses = []
        self.train_value_losses = []
        self.val_value_losses = []
        
        # 📊 NEW: Metrics storage
        self.train_policy_top1 = []
        self.train_policy_top3 = []
        self.train_value_mae = []
        self.train_value_mae_weighted = []
        self.train_value_wdl_acc = []
        self.train_value_wdl_ce = []
        self.val_policy_top1 = []
        self.val_policy_top3 = []
        self.val_value_mae = []
        self.val_value_mae_weighted = []
        self.val_value_wdl_acc = []
        self.val_value_wdl_ce = []
        
        # 🆕 Elo estimation storage
        self.estimated_elos = []  # (epoch, elo) tuples
        
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

        # Optional run context shown in plot header (e.g. startup mode/resume/transfer info).
        self.run_context_text = None
        # Optional notes shown in summary panel (e.g. final SWA metrics).
        self.final_notes = []
        # Optional epoch markers drawn on IL Elo chart (e.g. SWA final epoch).
        self.elo_epoch_markers = []  # (epoch, label)
        
        print(f"📊 Logging to: {self.csv_path}")

    def set_run_context(self, text):
        """Set optional short context displayed on generated PNG plots."""
        if text is None:
            self.run_context_text = None
            return
        text = str(text).strip()
        self.run_context_text = text if text else None

    def append_final_note(self, text):
        """Append short note shown in IL summary panel."""
        if text is None:
            return
        text = str(text).strip()
        if not text:
            return
        self.final_notes.append(text)

    def add_elo_epoch_marker(self, iteration, label):
        """Add or update an IL Elo-chart marker at a specific epoch."""
        if self.mode != "il":
            return
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            return

        text = str(label).strip() if label is not None else ""
        if not text:
            text = f"Epoch {iteration}"

        replaced = False
        for idx, (it, _) in enumerate(self.elo_epoch_markers):
            if int(it) == iteration:
                self.elo_epoch_markers[idx] = (iteration, text)
                replaced = True
                break
        if not replaced:
            self.elo_epoch_markers.append((iteration, text))
            self.elo_epoch_markers.sort(key=lambda x: x[0])

    def record_estimated_elo(self, iteration, estimated_elo, update_csv=True):
        """Record estimated Elo for a specific epoch/iteration (supports async updates)."""
        if self.mode != "il" or estimated_elo is None:
            return

        try:
            iteration = int(iteration)
            estimated_elo = float(estimated_elo)
        except (TypeError, ValueError):
            return

        # Upsert in-memory storage.
        replaced = False
        for idx, (it, _) in enumerate(self.estimated_elos):
            if int(it) == iteration:
                self.estimated_elos[idx] = (iteration, estimated_elo)
                replaced = True
                break
        if not replaced:
            self.estimated_elos.append((iteration, estimated_elo))
            self.estimated_elos.sort(key=lambda x: x[0])

        if not update_csv:
            return

        # Backfill CSV row for this epoch if it already exists.
        try:
            with open(self.csv_path, 'r', newline='') as f:
                rows = list(csv.reader(f))
            if not rows:
                return

            header = rows[0]
            if 'estimated_elo' not in header:
                return
            elo_col = header.index('estimated_elo')
            target_epoch = str(iteration)

            updated = False
            for row in rows[1:]:
                if not row:
                    continue
                if row[0] == target_epoch:
                    while len(row) <= elo_col:
                        row.append('')
                    row[elo_col] = str(int(round(estimated_elo)))
                    updated = True
                    break

            if updated:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
        except Exception:
            # CSV backfill is best-effort only.
            pass

    def get_latest_estimated_elo(self):
        """Return latest known Elo value or None."""
        if self.mode != "il" or not self.estimated_elos:
            return None
        return float(self.estimated_elos[-1][1])

    def get_latest_estimated_elo_with_epoch(self):
        """Return (epoch, elo) for latest known Elo, or (None, None)."""
        if self.mode != "il" or not self.estimated_elos:
            return None, None
        epoch, elo = self.estimated_elos[-1]
        try:
            return int(epoch), float(elo)
        except (TypeError, ValueError):
            return None, None

    def _plot_il_elo_panel(self, ax):
        """Render IL Elo panel, including optional epoch markers (e.g. SWA final)."""
        has_elos = bool(self.estimated_elos)
        has_markers = bool(self.elo_epoch_markers)
        if not has_elos and not has_markers:
            ax.axis('off')
            return

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Elo')
        ax.set_title('Estimated Elo (vs Stockfish)')
        ax.grid(True, alpha=0.3)

        if has_elos:
            elo_epochs, elo_vals = zip(*self.estimated_elos)
            ax.plot(elo_epochs, elo_vals, 'go-', label='Estimated Elo', linewidth=2, markersize=8)
            for ref_elo, ref_label in [(1200, 'Beginner'), (1500, 'Club'), (1800, 'Expert'), (2000, 'Candidate Master')]:
                if min(elo_vals) - 200 <= ref_elo <= max(elo_vals) + 200:
                    ax.axhline(y=ref_elo, color='gray', linestyle=':', alpha=0.4)
                    ax.text(elo_epochs[0], ref_elo + 15, ref_label, fontsize=8, color='gray', alpha=0.6)

        if has_markers:
            for marker_epoch, marker_label in self.elo_epoch_markers:
                ax.axvline(
                    x=marker_epoch,
                    color='black',
                    linestyle='--',
                    alpha=0.55,
                    linewidth=1.4,
                    label=marker_label,
                )
                if has_elos:
                    _, nearest_elo = min(
                        self.estimated_elos,
                        key=lambda pair: abs(int(pair[0]) - int(marker_epoch)),
                    )
                    ax.annotate(
                        marker_label,
                        xy=(marker_epoch, nearest_elo),
                        xytext=(4, 8),
                        textcoords='offset points',
                        fontsize=8,
                        color='black',
                    )
                else:
                    ax.text(
                        marker_epoch,
                        0.95,
                        marker_label,
                        transform=ax.get_xaxis_transform(),
                        rotation=90,
                        va='top',
                        ha='left',
                        fontsize=8,
                        color='black',
                    )

        if self.iterations:
            x_min = min(self.iterations)
            x_max = max(self.iterations)
            if x_min == x_max:
                x_min -= 1
                x_max += 1
            ax.set_xlim(x_min, x_max)

        ax.legend(fontsize=8)
    
    def log(self, iteration, train_losses=None, val_losses=None, 
            train_metrics=None, val_metrics=None, lr=None, estimated_elo=None, **kwargs):
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
                    train_metrics.get('value_mae_weighted', '') if train_metrics else '',
                    train_metrics.get('value_wdl_acc', '') if train_metrics else '',
                    train_metrics.get('value_wdl_ce', '') if train_metrics else '',
                    val_metrics.get('policy_top1_acc', '') if val_metrics else '',
                    val_metrics.get('policy_top3_acc', '') if val_metrics else '',
                    val_metrics.get('value_mae', '') if val_metrics else '',
                    val_metrics.get('value_mae_weighted', '') if val_metrics else '',
                    val_metrics.get('value_wdl_acc', '') if val_metrics else '',
                    val_metrics.get('value_wdl_ce', '') if val_metrics else ''
                ]
                
                # 🆕 Elo estimation
                if estimated_elo is not None:
                    try:
                        row.append(int(round(float(estimated_elo))))
                    except (TypeError, ValueError):
                        row.append('')
                        estimated_elo = None
                else:
                    row.append('')
                
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
                    self.train_value_mae_weighted.append(train_metrics.get('value_mae_weighted', 0))
                    self.train_value_wdl_acc.append(train_metrics.get('value_wdl_acc', 0))
                    self.train_value_wdl_ce.append(train_metrics.get('value_wdl_ce', 0))
                
                if val_losses is not None:
                    self.val_iterations.append(iteration)
                    self.val_losses.append(val_losses['total'])
                    self.val_policy_losses.append(val_losses['policy'])
                    self.val_value_losses.append(val_losses['value'])
                
                if val_metrics:
                    self.val_policy_top1.append(val_metrics.get('policy_top1_acc', 0))
                    self.val_policy_top3.append(val_metrics.get('policy_top3_acc', 0))
                    self.val_value_mae.append(val_metrics.get('value_mae', 0))
                    self.val_value_mae_weighted.append(val_metrics.get('value_mae_weighted', 0))
                    self.val_value_wdl_acc.append(val_metrics.get('value_wdl_acc', 0))
                    self.val_value_wdl_ce.append(val_metrics.get('value_wdl_ce', 0))
                
                # 🆕 Elo estimation storage
                if estimated_elo is not None:
                    # CSV already contains this value in the current row.
                    self.record_estimated_elo(iteration, estimated_elo, update_csv=False)
                
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
                    train_metrics.get('value_mae', '') if train_metrics else '',
                    train_metrics.get('value_mae_weighted', '') if train_metrics else '',
                    train_metrics.get('value_wdl_acc', '') if train_metrics else '',
                    train_metrics.get('value_wdl_ce', '') if train_metrics else ''
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
                    self.train_value_mae_weighted.append(train_metrics.get('value_mae_weighted', 0))
                    self.train_value_wdl_acc.append(train_metrics.get('value_wdl_acc', 0))
                    self.train_value_wdl_ce.append(train_metrics.get('value_wdl_ce', 0))
                
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
            fig, axes = plt.subplots(5, 3, figsize=(18, 22))
        else:
            fig, axes = plt.subplots(5, 2, figsize=(15, 22))
        
        if self.run_context_text:
            fig.suptitle(
                f"IL Training Progress\n{self.run_context_text}",
                fontsize=14,
                fontweight='bold',
            )
        else:
            fig.suptitle('IL Training Progress', fontsize=16, fontweight='bold')
        
        val_epochs = self.val_iterations if self.val_iterations else []
        
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
            ax.set_title('Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Value MAE
            ax = axes[1, 1]
            if self.train_value_mae:
                ax.plot(self.iterations, self.train_value_mae, 'b-', label='Train MAE', linewidth=2)
            if self.train_value_mae_weighted:
                ax.plot(self.iterations, self.train_value_mae_weighted, 'b--', label='Train MAE (weighted)', linewidth=2, alpha=0.8)
            if self.val_value_mae:
                ax.plot(val_epochs, self.val_value_mae, 'r-', label='Val MAE', linewidth=2)
            if self.val_value_mae_weighted:
                ax.plot(val_epochs, self.val_value_mae_weighted, 'r--', label='Val MAE (weighted)', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE')
            ax.set_title('Value MAE')
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
            
            # ============================================================
            # ROW 4: WDL METRICS
            # ============================================================
            
            # WDL Accuracy
            ax = axes[3, 0]
            if self.train_value_wdl_acc:
                ax.plot(self.iterations, self.train_value_wdl_acc, 'b-', label='Train WDL Acc', linewidth=2)
            if self.val_value_wdl_acc:
                ax.plot(val_epochs, self.val_value_wdl_acc, 'r-', label='Val WDL Acc', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Value WDL Accuracy')
            ax.set_ylim([0, 1])
            if self.train_value_wdl_acc or self.val_value_wdl_acc:
                ax.legend()
            ax.grid(True, alpha=0.3)
            
            # WDL Cross-Entropy
            ax = axes[3, 1]
            if self.train_value_wdl_ce:
                ax.plot(self.iterations, self.train_value_wdl_ce, 'b-', label='Train WDL CE', linewidth=2)
            if self.val_value_wdl_ce:
                ax.plot(val_epochs, self.val_value_wdl_ce, 'r-', label='Val WDL CE', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('CE')
            ax.set_title('Value WDL Cross-Entropy')
            if self.train_value_wdl_ce or self.val_value_wdl_ce:
                ax.legend()
            ax.grid(True, alpha=0.3)

            # ============================================================
            # ROW 5: ELO ESTIMATION + SUMMARY
            # ============================================================
            ax = axes[4, 0]
            self._plot_il_elo_panel(ax)
            
            ax = axes[4, 1]
            ax.axis('off')
            if self.val_policy_top1 and self.val_losses:
                summary_lines = [
                    "Latest Validation Metrics:",
                    "",
                    f"Policy Top-1: {self.val_policy_top1[-1]:.2%}",
                    f"Policy Top-3: {self.val_policy_top3[-1]:.2%}",
                    f"Value MAE: {self.val_value_mae[-1]:.4f}",
                ]
                if self.val_value_mae_weighted:
                    summary_lines.append(f"Value MAE (w): {self.val_value_mae_weighted[-1]:.4f}")
                if self.val_value_wdl_acc:
                    summary_lines.append(f"WDL Acc: {self.val_value_wdl_acc[-1]:.2%}")
                if self.val_value_wdl_ce:
                    summary_lines.append(f"WDL CE: {self.val_value_wdl_ce[-1]:.4f}")
                summary_lines.append(f"Total Loss: {self.val_losses[-1]:.4f}")
                if self.estimated_elos:
                    summary_lines.append(f"Est. Elo: {self.estimated_elos[-1][1]}")
                if self.final_notes:
                    summary_lines.append("")
                    summary_lines.append("Notes:")
                    summary_lines.extend(self.final_notes[-3:])
                summary_text = "\n".join(summary_lines)
                ax.text(
                    0.1,
                    0.5,
                    summary_text,
                    fontsize=12,
                    family='monospace',
                    verticalalignment='center',
                )
        
        else:
            # MTL plots (5x3)
            
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
            ax.set_title('Win Prediction (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            ax.plot(self.iterations, self.train_material_losses, 'b-', label='Train', linewidth=2)
            if self.val_material_losses:
                ax.plot(val_epochs, self.val_material_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Material Loss')
            ax.set_title('Material Count (Aux)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 2]
            ax.plot(self.iterations, self.train_check_losses, 'b-', label='Train', linewidth=2)
            if self.val_check_losses:
                ax.plot(val_epochs, self.val_check_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Check Loss')
            ax.set_title('Check Detection (Aux)')
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
            ax.set_title('Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[2, 1]
            if self.train_value_mae:
                ax.plot(self.iterations, self.train_value_mae, 'b-', label='Train', linewidth=2)
            if self.train_value_mae_weighted:
                ax.plot(self.iterations, self.train_value_mae_weighted, 'b--', label='Train (weighted)', linewidth=2, alpha=0.8)
            if self.val_value_mae:
                ax.plot(val_epochs, self.val_value_mae, 'r-', label='Val', linewidth=2)
            if self.val_value_mae_weighted:
                ax.plot(val_epochs, self.val_value_mae_weighted, 'r--', label='Val (weighted)', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE')
            ax.set_title('Value MAE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Value WDL metrics
            ax = axes[2, 2]
            if self.train_value_wdl_acc:
                ax.plot(self.iterations, self.train_value_wdl_acc, 'b-', label='Train WDL Acc', linewidth=2)
            if self.val_value_wdl_acc:
                ax.plot(val_epochs, self.val_value_wdl_acc, 'r-', label='Val WDL Acc', linewidth=2)
            if self.train_value_wdl_ce:
                ax.plot(self.iterations, self.train_value_wdl_ce, 'b--', label='Train WDL CE', linewidth=2, alpha=0.8)
            if self.val_value_wdl_ce:
                ax.plot(val_epochs, self.val_value_wdl_ce, 'r--', label='Val WDL CE', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric')
            ax.set_title('Value WDL Metrics')
            if self.train_value_wdl_acc or self.val_value_wdl_acc or self.train_value_wdl_ce or self.val_value_wdl_ce:
                ax.legend(fontsize=8)
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
            
            # Main tasks (Train)
            ax = axes[3, 2]
            ax.plot(self.iterations, self.train_losses, 'b-', label='Total', linewidth=2, alpha=0.7)
            ax.plot(self.iterations, self.train_policy_losses, 'g--', label='Policy', linewidth=1.5, alpha=0.7)
            ax.plot(self.iterations, self.train_value_losses, 'r--', label='Value', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Main Tasks (Train)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Summary metrics
            # Elo plot (MTL mode)
            ax = axes[4, 0]
            self._plot_il_elo_panel(ax)
            axes[4, 1].axis('off')
            ax = axes[4, 2]
            ax.axis('off')
            if self.val_policy_top1 and len(self.val_policy_top1) > 0:
                summary_lines = [
                    "Latest Validation Metrics:",
                    "",
                    f"Policy Top-1: {self.val_policy_top1[-1]:.2%}",
                    f"Policy Top-3: {self.val_policy_top3[-1]:.2%}",
                    f"Value MAE: {self.val_value_mae[-1]:.4f}",
                ]
                if self.val_value_mae_weighted:
                    summary_lines.append(f"Value MAE (w): {self.val_value_mae_weighted[-1]:.4f}")
                if self.val_value_wdl_acc:
                    summary_lines.append(f"WDL Acc: {self.val_value_wdl_acc[-1]:.2%}")
                if self.val_value_wdl_ce:
                    summary_lines.append(f"WDL CE: {self.val_value_wdl_ce[-1]:.4f}")
                summary_lines.append(f"Total Loss: {self.val_losses[-1]:.4f}")
                if self.estimated_elos:
                    summary_lines.append(f"Est. Elo: {self.estimated_elos[-1][1]}")
                if self.final_notes:
                    summary_lines.append("")
                    summary_lines.append("Notes:")
                    summary_lines.extend(self.final_notes[-3:])
                summary_text = "\n".join(summary_lines)
                ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")
    
    def _plot_rl(self):
        """Plot RL training progress"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        if self.run_context_text:
            fig.suptitle(
                f"RL Training Progress\n{self.run_context_text}",
                fontsize=14,
                fontweight='bold',
            )
        else:
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
            ax.set_title('Policy Accuracy')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        if self.train_value_mae:
            ax.plot(self.iterations, self.train_value_mae, 'r-', label='Value MAE', linewidth=2)
        if self.train_value_mae_weighted:
            ax.plot(self.iterations, self.train_value_mae_weighted, 'r--', label='Value MAE (weighted)', linewidth=2, alpha=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MAE')
        ax.set_title('Value MAE')
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
            ax.set_title('Temperature Schedule')
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
            summary_lines = ["Latest Metrics:", ""]
            if self.train_policy_top1:
                summary_lines.append(f"Policy Top-1: {self.train_policy_top1[-1]:.2%}")
            if self.train_policy_top3:
                summary_lines.append(f"Policy Top-3: {self.train_policy_top3[-1]:.2%}")
            if self.train_value_mae:
                summary_lines.append(f"Value MAE: {self.train_value_mae[-1]:.4f}")
            if self.train_value_mae_weighted:
                summary_lines.append(f"Value MAE (w): {self.train_value_mae_weighted[-1]:.4f}")
            if self.train_value_wdl_acc:
                summary_lines.append(f"WDL Acc: {self.train_value_wdl_acc[-1]:.2%}")
            if self.train_value_wdl_ce:
                summary_lines.append(f"WDL CE: {self.train_value_wdl_ce[-1]:.4f}")
            if self.train_losses:
                summary_lines.append(f"Total Loss: {self.train_losses[-1]:.4f}")
            summary_text = "\n".join(summary_lines)
            ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                   verticalalignment='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")
