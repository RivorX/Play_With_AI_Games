"""
Unified training logger for both IL and RL training
"""

import csv
import textwrap
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

class TrainingLogger:
    """
    Universal logger for training metrics
    Supports both IL and RL modes with detailed metrics
    """
    
    def __init__(self, log_dir, experiment_name="training", mode="il"):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            mode: "il" or "rl"
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
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
            
            else:  # RL mode
                header = [
                    'iteration', 'avg_loss', 'policy_loss', 'value_loss',
                    'score_rate', 'buffer_size', 'avg_game_length', 'positions_per_sec',
                    'selfplay_time', 'data_collection_time', 'temperature', 'beta',
                    'true_win_rate', 'eval_wins', 'eval_draws', 'eval_losses', 'eval_unresolved',
                    'anchor_score_rate', 'anchor_true_win_rate', 'anchor_wins', 'anchor_draws', 'anchor_losses',
                    # 📊 NEW: Metrics
                    'policy_top1_acc', 'policy_top3_acc',
                    'value_mae', 'value_mae_weighted',
                    'value_wdl_acc', 'value_wdl_ce',
                    # 🧠 RL stability telemetry
                    'completed_draw_rate', 'dynamic_uniform_fraction',
                    'priority_age_decay_lambda', 'avg_sample_age',
                    'avg_game_value', 'value_std',
                    'policy_entropy', 'value_pred_std',
                    'selfplay_decisive_games', 'selfplay_decisive_rate',
                    'selfplay_auto_draw_rate',
                    'selfplay_truncated_rate', 'selfplay_decisive_avg_length',
                    'selfplay_curriculum_dropped_positions', 'selfplay_cap_dropped_positions',
                    'adaptive_temp_adjustment', 'adaptive_temp_threshold',
                    'estimated_elo',
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
        
        if mode == "rl":
            self.win_rates = []
            self.true_win_rates = []
            self.anchor_score_rates = []
            self.anchor_true_win_rates = []
            self.temperatures = []
            self.adaptive_temp_adjustments = []
            self.adaptive_temp_thresholds = []

        # Optional run context shown in plot header (e.g. startup mode/resume/transfer info).
        self.run_context_text = None
        # Optional notes shown in summary panel (e.g. final SWA metrics).
        self.final_notes = []
        # Optional epoch markers drawn on IL Elo chart (e.g. SWA final epoch).
        self.elo_epoch_markers = []  # (epoch, label)
        # 🆕 SWA elo stored separately for distinct visual treatment in plots.
        self.swa_elo_info = None  # (epoch, elo) or None
        # 🆕 Full SWA metrics for summary panel.
        self.swa_metrics = None  # dict: {epoch, val_loss, top1, top3, mae, wdl_acc, wdl_ce, elo}
        
        print(f"📊 Logging to: {self.csv_path}")

    def set_run_context(self, text):
        """Set optional short context displayed on generated PNG plots."""
        if text is None:
            self.run_context_text = None
            return
        text = str(text).strip()
        self.run_context_text = text if text else None

    def _build_plot_suptitle(self, base_title, wrap_width=88):
        """Build a wrapped suptitle to avoid huge plot bounding boxes."""
        if not self.run_context_text:
            return base_title

        wrapped_context = textwrap.fill(
            self.run_context_text,
            width=max(40, int(wrap_width)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        return f"{base_title}\n{wrapped_context}"

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
        if estimated_elo is None:
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
        if not self.estimated_elos:
            return None
        return float(self.estimated_elos[-1][1])

    def get_latest_estimated_elo_with_epoch(self):
        """Return (epoch, elo) for latest known Elo, or (None, None)."""
        if not self.estimated_elos:
            return None, None
        epoch, elo = self.estimated_elos[-1]
        try:
            return int(epoch), float(elo)
        except (TypeError, ValueError):
            return None, None

    def record_swa_elo(self, epoch, elo):
        """Store SWA model Elo for distinct visual treatment (gold star) in plots.

        Also backfills the CSV via record_estimated_elo so the value persists.
        Must be called AFTER the regular training elos have been recorded so the
        SWA point is visually distinguishable from the training-time estimates.
        """
        if self.mode != "il":
            return
        try:
            epoch = int(epoch)
            elo = float(elo)
        except (TypeError, ValueError):
            return
        self.swa_elo_info = (epoch, elo)
        # Mirror elo into swa_metrics if already initialised.
        if self.swa_metrics is not None:
            self.swa_metrics['elo'] = elo
            self.swa_metrics['epoch'] = epoch
        else:
            self.swa_metrics = {'epoch': epoch, 'elo': elo}
        # Also persist to CSV and in-memory estimated_elos list.
        self.record_estimated_elo(epoch, elo, update_csv=True)

    def record_swa_metrics(self, val_loss=None, top1=None, top3=None,
                            mae=None, mae_weighted=None,
                            val_policy_loss=None, val_value_loss=None,
                            wdl_acc=None, wdl_ce=None,
                            elo=None, epoch=None):
        """Store full SWA evaluation metrics for the summary table and CSV.

        Writes a dedicated row with epoch='SWA' to the CSV file so metrics
        survive script restarts and can be auto-loaded by regen_plot.py.
        """
        if self.mode != "il":
            return

        def _f(v):
            try:
                f = float(v)
                return None if (f != f) else f  # NaN guard
            except (TypeError, ValueError):
                return None

        if self.swa_metrics is None:
            self.swa_metrics = {}
        m = self.swa_metrics
        if epoch is not None:
            try:
                m['epoch'] = int(epoch)
            except (TypeError, ValueError):
                pass
        for k, v in [('val_loss', val_loss), ('top1', top1), ('top3', top3),
                     ('mae', mae), ('mae_weighted', mae_weighted),
                     ('val_policy_loss', val_policy_loss),
                     ('val_value_loss', val_value_loss),
                     ('wdl_acc', wdl_acc), ('wdl_ce', wdl_ce), ('elo', elo)]:
            fv = _f(v)
            if fv is not None:
                m[k] = fv
        # Keep swa_elo_info in sync.
        if 'elo' in m and 'epoch' in m:
            self.swa_elo_info = (m['epoch'], m['elo'])
            # Backfill elo into regular estimated_elos for gold-star plot point
            self.record_estimated_elo(m['epoch'], m['elo'], update_csv=True)

        # Write / overwrite the dedicated SWA row in the CSV
        self._write_swa_csv_row()

    def _write_swa_csv_row(self):
        """Upsert an 'epoch=SWA' row at the end of the CSV with current swa_metrics."""
        if not self.swa_metrics:
            return
        m = self.swa_metrics

        def _s(v, fmt=None):
            if v is None:
                return ''
            try:
                fv = float(v)
                if fv != fv:  # NaN
                    return ''
                return str(round(fv, 6)) if fmt is None else fmt.format(fv)
            except (TypeError, ValueError):
                return ''

        try:
            with open(self.csv_path, 'r', newline='') as f:
                rows = list(csv.reader(f))
        except Exception:
            return

        if not rows:
            return
        header = rows[0]

        # Build a full-width row (all train cols empty, val cols from swa_metrics)
        col_map = {name: i for i, name in enumerate(header)}
        row = [''] * len(header)
        row[0] = 'SWA'  # epoch marker

        def _set(col_name, value):
            if col_name in col_map and value not in ('', None):
                row[col_map[col_name]] = value

        _set('val_loss',          _s(m.get('val_loss')))
        _set('val_policy_loss',   _s(m.get('val_policy_loss')))
        _set('val_value_loss',    _s(m.get('val_value_loss')))
        _set('val_policy_top1',   _s(m.get('top1')))
        _set('val_policy_top3',   _s(m.get('top3')))
        _set('val_value_mae',     _s(m.get('mae')))
        _set('val_value_mae_weighted', _s(m.get('mae_weighted')))
        _set('val_value_wdl_acc', _s(m.get('wdl_acc')))
        _set('val_value_wdl_ce',  _s(m.get('wdl_ce')))
        _set('estimated_elo',     _s(m.get('elo'), '{:.0f}'))

        # Remove any existing SWA row, then append new one
        data_rows = [r for r in rows[1:] if not r or r[0] != 'SWA']
        data_rows.append(row)

        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_rows)
        except Exception:
            pass

    @staticmethod
    def load_swa_row_from_csv(csv_path):
        """Read the 'epoch=SWA' row from a CSV and return a metrics dict (or None).

        Keys match record_swa_metrics kwargs:
            epoch, val_loss, top1, top3, mae, mae_weighted,
            val_policy_loss, val_value_loss, wdl_acc, wdl_ce, elo
        """
        try:
            with open(csv_path, 'r', newline='') as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return None

        for row in rows:
            if row.get('epoch', '').strip().upper() == 'SWA':
                def _sf(k):
                    v = row.get(k, '')
                    try:
                        f = float(v)
                        return None if (f != f) else f
                    except (TypeError, ValueError):
                        return None
                return {
                    'val_loss':        _sf('val_loss'),
                    'val_policy_loss': _sf('val_policy_loss'),
                    'val_value_loss':  _sf('val_value_loss'),
                    'top1':            _sf('val_policy_top1'),
                    'top3':            _sf('val_policy_top3'),
                    'mae':             _sf('val_value_mae'),
                    'mae_weighted':    _sf('val_value_mae_weighted'),
                    'wdl_acc':         _sf('val_value_wdl_acc'),
                    'wdl_ce':          _sf('val_value_wdl_ce'),
                    'elo':             _sf('estimated_elo'),
                }
        return None

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

        # Visible training window for the current plot.  This avoids drawing
        # transfer/resume seed Elo points that belong to epochs far outside the
        # currently plotted run, which otherwise creates a misleading clipped
        # horizontal line to the right edge of the panel.
        if self.iterations:
            x_min = min(self.iterations)
            x_max = max(self.iterations)
            if x_min == x_max:
                x_min -= 1
                x_max += 1
        else:
            x_min = None
            x_max = None

        # Determine SWA epoch to exclude from the regular line
        swa_epoch = int(self.swa_elo_info[0]) if self.swa_elo_info else None

        if has_elos:
            visible_elos = self.estimated_elos
            if x_min is not None and x_max is not None:
                visible_elos = [
                    (ep, val) for ep, val in self.estimated_elos
                    if x_min <= int(ep) <= x_max
                ]
            if not visible_elos:
                visible_elos = self.estimated_elos[-1:]

            # Plot regular elo line (exclude SWA point so it gets its own marker)
            regular_elos = [
                (ep, val) for ep, val in visible_elos
                if swa_epoch is None or int(ep) != swa_epoch
            ]
            if regular_elos:
                elo_epochs, elo_vals = zip(*regular_elos)
                if len(regular_elos) == 1:
                    ax.plot(
                        elo_epochs,
                        elo_vals,
                        color='green',
                        marker='o',
                        linestyle='None',
                        label='Estimated Elo',
                        markersize=8,
                    )
                else:
                    ax.plot(elo_epochs, elo_vals, 'go-', label='Estimated Elo', linewidth=2, markersize=8)
                all_elo_vals = elo_vals
            else:
                all_elo_vals = [v for _, v in visible_elos]

            for ref_elo, ref_label in [(1200, 'Beginner'), (1500, 'Club'), (1800, 'Expert'), (2000, 'Candidate Master')]:
                all_vals = [v for _, v in visible_elos]
                if min(all_vals) - 200 <= ref_elo <= max(all_vals) + 200:
                    x0 = visible_elos[0][0]
                    ax.axhline(y=ref_elo, color='gray', linestyle=':', alpha=0.4)
                    ax.text(x0, ref_elo + 15, ref_label, fontsize=8, color='gray', alpha=0.6)

        # 🆕 Plot SWA elo as a distinct gold star with annotation
        if self.swa_elo_info:
            sw_ep, sw_elo = self.swa_elo_info
            ax.plot(
                sw_ep, sw_elo,
                marker='*', markersize=18,
                color='gold', markeredgecolor='darkorange', markeredgewidth=1.5,
                linestyle='None',
                label=f'SWA final ({int(round(sw_elo))})',
                zorder=5,
            )
            ax.annotate(
                f'SWA\n{int(round(sw_elo))}',
                xy=(sw_ep, sw_elo),
                xytext=(-38, 6),
                textcoords='offset points',
                fontsize=8,
                color='darkorange',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2,
                                shrinkB=12),
            )

        if has_markers:
            for marker_epoch, marker_label in self.elo_epoch_markers:
                # Skip SWA marker vertical line — the gold star already marks it
                if self.swa_elo_info and int(marker_epoch) == swa_epoch:
                    continue
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

        # Compute x-axis range, padding right side to show SWA star fully
        if x_min is not None and x_max is not None:
            if self.swa_elo_info:
                sw_ep = int(self.swa_elo_info[0])
                if x_min <= sw_ep and sw_ep >= x_max - 1:
                    x_max = sw_ep + max(2, int((x_max - x_min) * 0.08) + 1)
            ax.set_xlim(x_min, x_max)

        ax.legend(fontsize=8, loc='lower right')
    
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
                else:
                    self.train_policy_top1.append(0.0)
                    self.train_policy_top3.append(0.0)
                    self.train_value_mae.append(0.0)
                    self.train_value_mae_weighted.append(0.0)
                    self.train_value_wdl_acc.append(0.0)
                    self.train_value_wdl_ce.append(0.0)
                
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
            
            else:  # RL mode
                row = [
                    iteration,
                    kwargs.get('avg_loss', ''),
                    kwargs.get('policy_loss', ''),
                    kwargs.get('value_loss', ''),
                    kwargs.get('score_rate', kwargs.get('win_rate', '')),
                    kwargs.get('buffer_size', ''),
                    kwargs.get('avg_game_length', ''),
                    kwargs.get('positions_per_sec', ''),
                    kwargs.get('selfplay_time', ''),
                    kwargs.get('data_collection_time', ''),
                    kwargs.get('temperature', ''),
                    kwargs.get('beta', ''),
                    kwargs.get('true_win_rate', ''),
                    kwargs.get('eval_wins', ''),
                    kwargs.get('eval_draws', ''),
                    kwargs.get('eval_losses', ''),
                    kwargs.get('eval_unresolved', ''),
                    kwargs.get('anchor_score_rate', ''),
                    kwargs.get('anchor_true_win_rate', ''),
                    kwargs.get('anchor_wins', ''),
                    kwargs.get('anchor_draws', ''),
                    kwargs.get('anchor_losses', ''),
                    # 📊 NEW: Metrics
                    train_metrics.get('policy_top1_acc', '') if train_metrics else '',
                    train_metrics.get('policy_top3_acc', '') if train_metrics else '',
                    train_metrics.get('value_mae', '') if train_metrics else '',
                    train_metrics.get('value_mae_weighted', '') if train_metrics else '',
                    train_metrics.get('value_wdl_acc', '') if train_metrics else '',
                    train_metrics.get('value_wdl_ce', '') if train_metrics else '',
                    kwargs.get('completed_draw_rate', ''),
                    kwargs.get('dynamic_uniform_fraction', ''),
                    kwargs.get('priority_age_decay_lambda', ''),
                    kwargs.get('avg_sample_age', ''),
                    kwargs.get('avg_game_value', ''),
                    kwargs.get('value_std', ''),
                    kwargs.get('policy_entropy', ''),
                    kwargs.get('value_pred_std', ''),
                    kwargs.get('selfplay_decisive_games', ''),
                    kwargs.get('selfplay_decisive_rate', ''),
                    kwargs.get('selfplay_auto_draw_rate', ''),
                    kwargs.get('selfplay_truncated_rate', ''),
                    kwargs.get('selfplay_decisive_avg_length', ''),
                    kwargs.get('selfplay_curriculum_dropped_positions', ''),
                    kwargs.get('selfplay_cap_dropped_positions', ''),
                    kwargs.get('adaptive_temp_adjustment', ''),
                    kwargs.get('adaptive_temp_threshold', ''),
                ]

                if estimated_elo is not None:
                    try:
                        row.append(int(round(float(estimated_elo))))
                    except (TypeError, ValueError):
                        row.append('')
                        estimated_elo = None
                else:
                    row.append('')
                
                # Store for plotting
                self.iterations.append(iteration)
                if 'avg_loss' in kwargs:
                    self.train_losses.append(kwargs['avg_loss'])
                    self.train_policy_losses.append(kwargs.get('policy_loss', 0))
                    self.train_value_losses.append(kwargs.get('value_loss', 0))
                elif 'train_losses' in kwargs:
                    self.train_losses.append(kwargs['train_losses'].get('total', 0))
                    self.train_policy_losses.append(kwargs['train_losses'].get('policy', 0))
                    self.train_value_losses.append(kwargs['train_losses'].get('value', 0))
                
                if train_metrics:
                    self.train_policy_top1.append(train_metrics.get('policy_top1_acc', 0))
                    self.train_policy_top3.append(train_metrics.get('policy_top3_acc', 0))
                    self.train_value_mae.append(train_metrics.get('value_mae', 0))
                    self.train_value_mae_weighted.append(train_metrics.get('value_mae_weighted', 0))
                    self.train_value_wdl_acc.append(train_metrics.get('value_wdl_acc', 0))
                    self.train_value_wdl_ce.append(train_metrics.get('value_wdl_ce', 0))
                else:
                    self.train_policy_top1.append(0.0)
                    self.train_policy_top3.append(0.0)
                    self.train_value_mae.append(0.0)
                    self.train_value_mae_weighted.append(0.0)
                    self.train_value_wdl_acc.append(0.0)
                    self.train_value_wdl_ce.append(0.0)
                
                score_rate = kwargs.get('score_rate', kwargs.get('win_rate'))
                if score_rate is not None:
                    self.win_rates.append((iteration, score_rate))
                if 'true_win_rate' in kwargs and kwargs['true_win_rate'] is not None:
                    self.true_win_rates.append((iteration, kwargs['true_win_rate']))
                if 'anchor_score_rate' in kwargs and kwargs['anchor_score_rate'] is not None:
                    self.anchor_score_rates.append((iteration, kwargs['anchor_score_rate']))
                if 'anchor_true_win_rate' in kwargs and kwargs['anchor_true_win_rate'] is not None:
                    self.anchor_true_win_rates.append((iteration, kwargs['anchor_true_win_rate']))
                 
                if 'temperature' in kwargs and kwargs['temperature'] is not None:
                    self.temperatures.append((iteration, kwargs['temperature']))
                if 'adaptive_temp_adjustment' in kwargs and kwargs['adaptive_temp_adjustment'] is not None:
                    self.adaptive_temp_adjustments.append((iteration, kwargs['adaptive_temp_adjustment']))
                if 'adaptive_temp_threshold' in kwargs and kwargs['adaptive_temp_threshold'] is not None:
                    self.adaptive_temp_thresholds.append((iteration, kwargs['adaptive_temp_threshold']))

                if estimated_elo is not None:
                    self.record_estimated_elo(iteration, estimated_elo, update_csv=False)
            
            writer.writerow(row)
    
    def plot(self):
        """Generate training plots"""
        if len(self.iterations) < 2:
            return
        
        if self.mode == "il":
            self._plot_il()
        else:
            self._plot_rl()

    # ------------------------------------------------------------------
    # Summary panel
    # ------------------------------------------------------------------
    def _plot_il_summary_panel(self, ax):
        """Render styled two-column summary table: Best Model vs SWA Model."""
        ax.axis('off')
        ax.set_title('Training Summary', fontsize=14, fontweight='bold', pad=8)

        if not self.val_losses or not self.iterations:
            return

        # ── Find best model (min val_loss) ───────────────────────────────
        best_idx = min(range(len(self.val_losses)), key=lambda i: self.val_losses[i])
        best_epoch = self.iterations[best_idx]

        def _at(lst, idx):
            return lst[idx] if lst and idx < len(lst) else None

        best = {
            'epoch':    best_epoch,
            'val_loss': self.val_losses[best_idx],
            'top1':     _at(self.val_policy_top1, best_idx),
            'top3':     _at(self.val_policy_top3, best_idx),
            'mae':      _at(self.val_value_mae, best_idx),
            'wdl_acc':  _at(self.val_value_wdl_acc, best_idx),
            'wdl_ce':   _at(self.val_value_wdl_ce, best_idx),
        }
        # Elo closest to best_epoch (excluding SWA entry)
        swa_ep = int(self.swa_elo_info[0]) if self.swa_elo_info else None
        regular_elos = [(ep, v) for ep, v in self.estimated_elos
                        if swa_ep is None or int(ep) != swa_ep]
        if regular_elos:
            _, best['elo'] = min(regular_elos, key=lambda p: abs(int(p[0]) - best_epoch))
        else:
            best['elo'] = None

        swa = self.swa_metrics  # dict or None

        # ── Formatters ───────────────────────────────────────────────────
        def _fl(v):  return f"{v:.4f}" if v is not None else "—"
        def _fp(v):  return f"{v:.2%}"  if v is not None else "—"
        def _fe(v):  return f"{int(round(v))}" if v is not None else "—"
        def _sv(d, k): return d.get(k) if d else None

        def _delta(bv, sv, mode='less', pct=False):
            if bv is None or sv is None:
                return ""
            d = sv - bv
            s = f"{d:+.2%}" if pct else f"{d:+.4f}"
            good = (d < -1e-6) if mode == 'less' else (d > 1e-6)
            bad  = (d >  1e-6) if mode == 'less' else (d < -1e-6)
            arrow = "▲" if good else ("▼" if bad else "")
            return f"{arrow} {s}".strip() if arrow else s

        # Elo delta is special (integer, 'more' is better)
        def _delta_elo(bv, sv):
            if bv is None or sv is None or not sv or not bv:
                return ""
            d = int(round(sv - bv))
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "")
            return f"{arrow} {d:+d}".strip() if arrow else f"{d:+d}"

        swa_ep_label = swa.get('epoch', '?') if swa else "—"
        col_labels = [
            "Metric",
            f"Best (ep {best_epoch})",
            f"SWA (ep {swa_ep_label})",
            "Δ  SWA – Best",
        ]

        rows_raw = [
            ("Val Loss",
             _fl(best['val_loss']),
             _fl(_sv(swa,'val_loss')),
             _delta(best['val_loss'], _sv(swa,'val_loss'), 'less')),
            ("Policy Top-1",
             _fp(best['top1']),
             _fp(_sv(swa,'top1')),
             _delta(best['top1'], _sv(swa,'top1'), 'more', pct=True)),
            ("Policy Top-3",
             _fp(best['top3']),
             _fp(_sv(swa,'top3')),
             _delta(best['top3'], _sv(swa,'top3'), 'more', pct=True)),
            ("Value MAE",
             _fl(best['mae']),
             _fl(_sv(swa,'mae')),
             _delta(best['mae'], _sv(swa,'mae'), 'less')),
            ("WDL Acc",
             _fp(best['wdl_acc']),
             _fp(_sv(swa,'wdl_acc')),
             _delta(best['wdl_acc'], _sv(swa,'wdl_acc'), 'more', pct=True)),
            ("WDL CE",
             _fl(best['wdl_ce']),
             _fl(_sv(swa,'wdl_ce')),
             _delta(best['wdl_ce'], _sv(swa,'wdl_ce'), 'less')),
            ("Est. Elo",
             _fe(best['elo']),
             _fe(_sv(swa,'elo')),
             _delta_elo(best['elo'], _sv(swa,'elo'))),
        ]

        cell_text = [list(r) for r in rows_raw]

        # ── Colour palette ───────────────────────────────────────────────
        C_BEST   = '#2E7D32'   # dark green header
        C_SWA    = '#E65100'   # deep orange header
        C_DELTA  = '#1565C0'   # dark blue header
        C_METRIC = '#424242'   # dark grey header
        BG_EVEN  = '#F5F5F5'
        BG_ODD   = '#FFFFFF'
        BG_SWA   = '#FFF8E1'   # warm yellow tint for SWA values
        BG_UP    = '#C8E6C9'   # light green — improvement
        BG_DOWN  = '#FFCDD2'   # light red   — regression
        BG_NEUT  = '#E3F2FD'   # light blue  — neutral delta

        def _delta_bg(d_str):
            if '▲' in d_str: return BG_UP
            if '▼' in d_str: return BG_DOWN
            if d_str and d_str != "—": return BG_NEUT
            return BG_ODD

        col_colors = [C_METRIC, C_BEST, C_SWA, C_DELTA]
        row_colors = []
        for i, row in enumerate(rows_raw):
            bg = BG_EVEN if i % 2 == 0 else BG_ODD
            row_colors.append([bg, bg, BG_SWA, _delta_bg(row[3])])

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=row_colors,
            colColours=col_colors,
            cellLoc='center',
            loc='center',
            bbox=[0.0, 0.05, 1.0, 0.95],
            colWidths=[0.28, 0.22, 0.22, 0.28],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)

        # Header row styling
        for col_i, hdr_c in enumerate([C_METRIC, C_BEST, C_SWA, C_DELTA]):
            cell = tbl[0, col_i]
            cell.set_facecolor(hdr_c)
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_height(cell.get_height() * 1.6)

        # Metric column: bold
        for row_i in range(1, len(rows_raw) + 1):
            tbl[row_i, 0].set_text_props(fontweight='bold', ha='left')
            tbl[row_i, 0].PAD = 0.05
            for col_i in range(4):
                tbl[row_i, col_i].set_height(tbl[row_i, col_i].get_height() * 1.3)


    def _plot_il(self):
        """Plot IL training progress"""
        fig, axes = plt.subplots(5, 2, figsize=(15, 22))
        
        if self.run_context_text:
            fig.suptitle(
                self._build_plot_suptitle("IL Training Progress"),
                fontsize=14,
                fontweight='bold',
                y=0.985,
            )
        else:
            fig.suptitle('IL Training Progress', fontsize=16, fontweight='bold', y=0.985)
        
        val_epochs = self.val_iterations if self.val_iterations else []
        
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
            ax.plot(self.iterations, self.train_value_mae, 'b-', label='Train MAE', linewidth=2, alpha=0.75, zorder=2)
        if self.train_value_mae_weighted:
            ax.plot(self.iterations, self.train_value_mae_weighted, 'b--', label='Train MAE (weighted)', linewidth=2, alpha=0.6, zorder=2)
        if self.val_value_mae:
            ax.plot(val_epochs, self.val_value_mae, 'r-', label='Val MAE', linewidth=2.5, zorder=3)
        if self.val_value_mae_weighted:
            ax.plot(val_epochs, self.val_value_mae_weighted, 'r--', label='Val MAE (weighted)', linewidth=2, alpha=0.85, zorder=3)
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
        self._plot_il_summary_panel(ax)
    

        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.04, top=0.94, hspace=0.42, wspace=0.28)
        fig.savefig(self.plot_path, dpi=150)
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")
    
    def _plot_rl(self):
        """Plot RL training progress"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        if self.run_context_text:
            fig.suptitle(
                self._build_plot_suptitle("RL Training Progress"),
                fontsize=14,
                fontweight='bold',
                y=0.98,
            )
        else:
            fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold', y=0.98)
        
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
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MAE')
        ax.set_title('Value MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        if self.win_rates or self.anchor_score_rates:
            if self.win_rates:
                win_iters, win_vals = zip(*self.win_rates)
                ax.plot(win_iters, win_vals, 'mo-', label='Score Rate', linewidth=2, markersize=8)
            if self.true_win_rates:
                true_iters, true_vals = zip(*self.true_win_rates)
                ax.plot(true_iters, true_vals, 'co-', label='True Win Rate', linewidth=1.5, markersize=6)
            if self.anchor_score_rates:
                anchor_iters, anchor_vals = zip(*self.anchor_score_rates)
                ax.plot(anchor_iters, anchor_vals, 'yo-', label='Anchor Score Rate', linewidth=2, markersize=7)
            if self.anchor_true_win_rates:
                anchor_true_iters, anchor_true_vals = zip(*self.anchor_true_win_rates)
                ax.plot(anchor_true_iters, anchor_true_vals, 'ko--', label='Anchor True Win Rate', linewidth=1.5, markersize=5, alpha=0.85)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Rate')
            ax.set_title('Score/Win Rate vs Best + Anchor')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 3: Additional
        ax = axes[2, 0]
        if self.temperatures:
            temp_iters, temp_vals = zip(*self.temperatures)
            ax.plot(temp_iters, temp_vals, 'orange', linewidth=2, label='Applied Temperature')
            if self.adaptive_temp_adjustments:
                adj_iters, adj_vals = zip(*self.adaptive_temp_adjustments)
                ax.plot(adj_iters, adj_vals, color='tab:red', linestyle='--', linewidth=1.5, label='Adaptive Adjustment')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Temperature')
            ax.set_title('Applied Temperature Schedule')
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
            if self.train_value_wdl_acc:
                summary_lines.append(f"WDL Acc: {self.train_value_wdl_acc[-1]:.2%}")
            if self.train_value_wdl_ce:
                summary_lines.append(f"WDL CE: {self.train_value_wdl_ce[-1]:.4f}")
            if self.train_losses:
                summary_lines.append(f"Total Loss: {self.train_losses[-1]:.4f}")
            summary_text = "\n".join(summary_lines)
            ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                   verticalalignment='center')
        
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.92, hspace=0.38, wspace=0.28)
        fig.savefig(self.plot_path, dpi=150)
        plt.close()
        
        print(f"📈 Plot saved to: {self.plot_path}")
