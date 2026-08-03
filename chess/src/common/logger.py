"""Shared training logger coordinating domain-specific IL and RL loggers."""

import csv
import textwrap
from datetime import datetime
from pathlib import Path

from src.training.il.logger import IL_GRADIENT_COLUMNS, ILLoggerMixin
from src.training.rl.log_schema import (
    RL_DATA_QUALITY_COLUMNS,
    RL_MAIN_COLUMNS,
    RL_PERFORMANCE_COLUMNS,
)
from src.training.rl.logger import RLLoggerMixin
from src.common.logger_helpers import (
    _CSV_CONFIG_METADATA_KEY,
    _CSV_RUN_SUMMARY_METADATA_KEY,
    _CSV_RESUME_METADATA_KEY,
    _clean_optional_float,
    _clean_elo_ci95,
    _upsert_mcts_elo_by_simulations,
    _read_csv_rows_preserving_metadata,
    _upsert_metadata_row,
)


class TrainingLogger(RLLoggerMixin, ILLoggerMixin):
    """
    Universal logger for training metrics
    Supports both IL and RL modes with detailed metrics
    """
    
    def __init__(self, log_dir, experiment_name="training", mode="il", config_snapshot=None, verbose=True):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            mode: "il" or "rl"
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir = self.log_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.verbose = bool(verbose)
        self.config_snapshot = dict(config_snapshot or {})
        
        # Windows forbids ':' in file names, so the requested minute-precision
        # timestamp uses HH-MM. Add a numeric suffix only for two runs started
        # within the same minute; seconds stay out of all user-facing names.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base_stem = f"{experiment_name}_{timestamp}"
        stem = base_stem
        collision = 2
        while (self.csv_dir / f"{stem}.csv").exists() or (self.log_dir / f"{stem}.png").exists():
            stem = f"{base_stem}_{collision}"
            collision += 1
        self.csv_path = self.csv_dir / f"{stem}.csv"
        self.plot_path = self.log_dir / f"{stem}.png"
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if mode == "il":
                header = [
                    'epoch', 'train_loss', 'train_policy_loss', 'train_value_loss',
                    'train_moves_left_loss',
                    'val_loss', 'val_policy_loss', 'val_value_loss', 'val_moves_left_loss', 'learning_rate',
                    # NEW: Metrics
                    'train_policy_top1', 'train_policy_top3',
                    'train_policy_target_mass_top1', 'train_policy_target_mass_top3', 'train_policy_target_mass_top5',
                    'train_policy_entropy', 'train_policy_effective_moves', 'train_policy_top1_prob',
                    'train_policy_legal_entropy', 'train_policy_legal_effective_moves',
                    'train_policy_legal_top1_prob', 'train_policy_legal_top1_margin',
                    'train_policy_target_entropy', 'train_policy_target_effective_moves', 'train_policy_target_top1_mass',
                    'train_policy_target_support_top1_prob', 'train_policy_target_support_top1_margin',
                    'train_value_mae', 'train_value_mae_opening', 'train_value_mae_middlegame', 'train_value_mae_endgame',
                    'train_value_wdl_acc', 'train_value_wdl_ce',
                    'train_value_wdl_ce_opening', 'train_value_wdl_ce_middlegame', 'train_value_wdl_ce_endgame',
                    'train_value_std_ratio_opening', 'train_value_std_ratio_middlegame', 'train_value_std_ratio_endgame',
                    'train_moves_left_loss_opening', 'train_moves_left_loss_middlegame', 'train_moves_left_loss_endgame',
                    'train_moves_left_mae', 'train_moves_left_mae_opening',
                    'train_moves_left_mae_middlegame', 'train_moves_left_mae_endgame',
                    'val_policy_top1', 'val_policy_top3',
                    'val_policy_target_mass_top1', 'val_policy_target_mass_top3', 'val_policy_target_mass_top5',
                    'val_policy_entropy', 'val_policy_effective_moves', 'val_policy_top1_prob',
                    'val_policy_legal_entropy', 'val_policy_legal_effective_moves',
                    'val_policy_legal_top1_prob', 'val_policy_legal_top1_margin',
                    'val_policy_target_entropy', 'val_policy_target_effective_moves', 'val_policy_target_top1_mass',
                    'val_policy_target_support_top1_prob', 'val_policy_target_support_top1_margin',
                    'val_value_mae', 'val_value_mae_opening', 'val_value_mae_middlegame', 'val_value_mae_endgame',
                    'val_value_wdl_acc', 'val_value_wdl_ce',
                    'val_value_wdl_ce_opening', 'val_value_wdl_ce_middlegame', 'val_value_wdl_ce_endgame',
                    'val_value_std_ratio_opening', 'val_value_std_ratio_middlegame', 'val_value_std_ratio_endgame',
                    'val_moves_left_loss_opening', 'val_moves_left_loss_middlegame', 'val_moves_left_loss_endgame',
                    'val_moves_left_mae', 'val_moves_left_mae_opening',
                    'val_moves_left_mae_middlegame', 'val_moves_left_mae_endgame',
                    'train_soft_occurrence_avg', 'train_soft_occurrence_max', 'train_soft_sample_weight_avg',
                    'train_soft_policy_mass_kept_avg', 'train_soft_policy_mass_kept_min',
                    'val_soft_occurrence_avg', 'val_soft_occurrence_max', 'val_soft_sample_weight_avg',
                    'val_soft_policy_mass_kept_avg', 'val_soft_policy_mass_kept_min',
                    # Elo estimation
                    'estimated_elo',
                    'estimated_elo_se',
                    'estimated_elo_ci95_low',
                    'estimated_elo_ci95_high',
                    'estimated_elo_nn',
                    'estimated_elo_nn_se',
                    'estimated_elo_nn_ci95_low',
                    'estimated_elo_nn_ci95_high',
                    'estimated_elo_mcts',
                    'estimated_elo_mcts_se',
                    'estimated_elo_mcts_ci95_low',
                    'estimated_elo_mcts_ci95_high',
                    'estimated_elo_mcts_simulations',
                    'estimated_elo_mcts_by_simulations',
                    'estimated_elo_nn_label',
                    'estimated_elo_mcts_label',
                    'train_val_loss_gap',
                    'policy_top1_gap',
                    'policy_top3_gap',
                    'value_mae_gap',
                    'value_wdl_acc_gap',
                    'best_val_loss_so_far',
                    'best_val_policy_top1_so_far',
                    'best_val_value_mae_so_far',
                    *IL_GRADIENT_COLUMNS,
                ]
            
            else:  # RL mode
                header = list(RL_MAIN_COLUMNS)
            if config_snapshot is not None:
                for metadata_row in _upsert_metadata_row([], _CSV_CONFIG_METADATA_KEY, config_snapshot):
                    writer.writerow(metadata_row)
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
        self.train_moves_left_losses = []
        self.val_moves_left_losses = []
        
        # NEW: Metrics storage
        self.train_policy_top1 = []
        self.train_policy_top3 = []
        self.train_policy_target_mass_top1 = []
        self.train_policy_target_mass_top3 = []
        self.train_policy_target_mass_top5 = []
        self.train_policy_entropy = []
        self.train_policy_effective_moves = []
        self.train_policy_top1_prob = []
        self.train_policy_legal_entropy = []
        self.train_policy_legal_effective_moves = []
        self.train_policy_legal_top1_prob = []
        self.train_policy_legal_top1_margin = []
        self.train_policy_target_entropy = []
        self.train_policy_target_effective_moves = []
        self.train_policy_target_top1_mass = []
        self.train_policy_target_support_top1_prob = []
        self.train_policy_target_support_top1_margin = []
        self.train_value_mae = []
        self.train_value_mae_opening = []
        self.train_value_mae_middlegame = []
        self.train_value_mae_endgame = []
        self.train_value_wdl_acc = []
        self.train_value_wdl_ce = []
        self.train_value_wdl_ce_opening = []
        self.train_value_wdl_ce_middlegame = []
        self.train_value_wdl_ce_endgame = []
        self.train_value_std_ratio_opening = []
        self.train_value_std_ratio_middlegame = []
        self.train_value_std_ratio_endgame = []
        self.train_moves_left_loss_opening = []
        self.train_moves_left_loss_middlegame = []
        self.train_moves_left_loss_endgame = []
        self.train_moves_left_mae = []
        self.train_moves_left_mae_opening = []
        self.train_moves_left_mae_middlegame = []
        self.train_moves_left_mae_endgame = []
        self.val_policy_top1 = []
        self.val_policy_top3 = []
        self.val_policy_target_mass_top1 = []
        self.val_policy_target_mass_top3 = []
        self.val_policy_target_mass_top5 = []
        self.val_policy_entropy = []
        self.val_policy_effective_moves = []
        self.val_policy_top1_prob = []
        self.val_policy_legal_entropy = []
        self.val_policy_legal_effective_moves = []
        self.val_policy_legal_top1_prob = []
        self.val_policy_legal_top1_margin = []
        self.val_policy_target_entropy = []
        self.val_policy_target_effective_moves = []
        self.val_policy_target_top1_mass = []
        self.val_policy_target_support_top1_prob = []
        self.val_policy_target_support_top1_margin = []
        self.val_value_mae = []
        self.val_value_mae_opening = []
        self.val_value_mae_middlegame = []
        self.val_value_mae_endgame = []
        self.val_value_wdl_acc = []
        self.val_value_wdl_ce = []
        self.val_value_wdl_ce_opening = []
        self.val_value_wdl_ce_middlegame = []
        self.val_value_wdl_ce_endgame = []
        self.val_value_std_ratio_opening = []
        self.val_value_std_ratio_middlegame = []
        self.val_value_std_ratio_endgame = []
        self.val_moves_left_loss_opening = []
        self.val_moves_left_loss_middlegame = []
        self.val_moves_left_loss_endgame = []
        self.val_moves_left_mae = []
        self.val_moves_left_mae_opening = []
        self.val_moves_left_mae_middlegame = []
        self.val_moves_left_mae_endgame = []
        self.train_soft_occurrence_avg = []
        self.train_soft_occurrence_max = []
        self.train_soft_sample_weight_avg = []
        self.train_soft_policy_mass_kept_avg = []
        self.train_soft_policy_mass_kept_min = []
        self.val_soft_occurrence_avg = []
        self.val_soft_occurrence_max = []
        self.val_soft_sample_weight_avg = []
        self.val_soft_policy_mass_kept_avg = []
        self.val_soft_policy_mass_kept_min = []
        self._ensure_il_gradient_storage()
        
        # Elo estimation storage
        self.estimated_elos = []  # (epoch, elo) tuples
        self.estimated_elo_errors = {}  # epoch -> {"se": float, "ci95": (low, high)}
        self._pending_rl_elo_by_iteration = {}
        self.best_final_elo_info = None  # (epoch, elo) for exact final best-model Elo
        self.il_mode_elo_markers = []  # dicts: epoch, elo, mode, simulations, label
        
        if mode == "rl":
            self.details_dir = self.log_dir / "details"
            self.details_dir.mkdir(parents=True, exist_ok=True)
            self.details_prefix = self.csv_path.stem
            self.performance_dir = self.details_dir
            self.performance_log_path = self.csv_dir / f"{self.details_prefix}_performance.csv"
            self.performance_plot_path = self.details_dir / f"{self.details_prefix}_performance.png"
            self.data_quality_log_path = self.csv_dir / f"{self.details_prefix}_data_quality.csv"
            self.data_quality_plot_path = self.details_dir / f"{self.details_prefix}_data_quality.png"
            self.debug_training_profile_dir = self.log_dir / "debug" / "training_profile"
            self.debug_training_profile_dir.mkdir(parents=True, exist_ok=True)
            self.latest_training_profile_path = self.csv_dir / "rl_latest_training_profile.csv"
            for stale_profile in list(self.debug_training_profile_dir.glob("rl_*training_profile*.csv")) + list(self.csv_dir.glob("rl_*training_profile*.csv")):
                if stale_profile != self.latest_training_profile_path:
                    try:
                        stale_profile.unlink()
                    except OSError:
                        pass
            performance_header = list(RL_PERFORMANCE_COLUMNS)
            with open(self.performance_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(performance_header)
            with open(self.latest_training_profile_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(performance_header)
            with open(self.data_quality_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(RL_DATA_QUALITY_COLUMNS)
        else:
            self.details_dir = None
            self.details_prefix = None
            self.performance_dir = None
            self.performance_log_path = None
            self.performance_plot_path = None
            self.data_quality_log_path = None
            self.data_quality_plot_path = None

        # Optional run context shown in plot header (e.g. startup mode/resume/transfer info).
        self.run_context_text = None
        self.run_context_lines = []
        self.run_summary_metadata = None
        self.plot_smoothing_enabled = False
        self.plot_smoothing_alpha = 0.35
        self.plot_smoothing_min_points = 5
        # Optional notes shown in summary panel (e.g. final SWA metrics).
        self.final_notes = []
        # Optional epoch markers drawn on IL Elo chart (e.g. SWA final epoch).
        self.elo_epoch_markers = []  # (epoch, label)
        # SWA elo stored separately for distinct visual treatment in plots.
        self.swa_elo_info = None  # (epoch, elo) or None
        # Full SWA metrics for summary panel.
        self.swa_metrics = None  # dict: {epoch, val_loss, top1, top3, mae, wdl_acc, wdl_ce, elo}
        # Resume metadata and plot markers.
        self.resume_markers = []  # dicts: x, next_epoch, label
        self.resume_source_csv_path = None
        self.resume_source_summary = None
        self.resume_current_summary = None
        self.resume_history_epoch = None
        
        if self.verbose:
            print(f"Logging to: {self.csv_path}")
        if self.mode == "rl" and self.verbose:
            print(f"RL performance log: {self.performance_log_path}")
            print(f"RL data-quality log: {self.data_quality_log_path}")

    def set_run_context(self, text):
        """Set optional short context displayed on generated PNG plots."""
        if text is None:
            self.run_context_text = None
            self.run_context_lines = []
            return
        text = str(text).strip()
        self.run_context_text = text if text else None
        self.run_context_lines = []

    def set_run_context_lines(self, lines):
        """Set explicit context lines displayed under the plot title."""
        if not lines:
            self.run_context_lines = []
            return
        cleaned = []
        for line in lines:
            text = str(line).strip() if line is not None else ""
            if text:
                cleaned.append(text)
        self.run_context_lines = cleaned
        self.run_context_text = " | ".join(cleaned) if cleaned else self.run_context_text

    def set_run_summary_metadata(self, summary):
        """Write/update a compact run summary as leading CSV metadata."""
        if summary is None:
            return
        summary = dict(summary)
        self.run_summary_metadata = summary
        if self.resume_source_csv_path is not None:
            self.resume_current_summary = dict(summary)
            self._write_resume_metadata()
        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
            metadata_rows = _upsert_metadata_row(
                metadata_rows,
                _CSV_RUN_SUMMARY_METADATA_KEY,
                summary,
            )
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerows(rows)
        except Exception:
            pass
        if self.resume_source_csv_path is not None or self.resume_markers:
            self._write_resume_metadata()

    def add_resume_marker(self, completed_epoch, label=None):
        """Draw a vertical boundary before the first continued epoch/iteration."""
        if self.mode not in {"il", "rl"}:
            return
        try:
            completed_epoch = int(completed_epoch)
        except (TypeError, ValueError):
            return
        if completed_epoch < 1:
            return
        next_epoch = completed_epoch + 1
        unit = "ep" if self.mode == "il" else "iter"
        marker = {
            'x': float(completed_epoch) + 0.5,
            'completed_epoch': completed_epoch,
            'next_epoch': next_epoch,
            'completed_iteration': completed_epoch if self.mode == "rl" else None,
            'next_iteration': next_epoch if self.mode == "rl" else None,
            'label': str(label or f"resume -> {unit} {next_epoch}"),
        }
        self.resume_markers = [
            existing for existing in self.resume_markers
            if int(existing.get('completed_epoch', -1)) != completed_epoch
        ]
        self.resume_markers.append(marker)
        self.resume_markers.sort(key=lambda item: float(item.get('x', 0.0)))
        self._write_resume_metadata()

    def _write_resume_metadata(self):
        if self.resume_source_csv_path is None and not self.resume_markers:
            return
        payload = {
            'source_csv': str(self.resume_source_csv_path) if self.resume_source_csv_path else None,
            'history_epoch': self.resume_history_epoch,
            'markers': self.resume_markers,
            'final_elo_markers': self.il_mode_elo_markers,
            'previous_run': self.resume_source_summary,
            'current_run': self.resume_current_summary,
        }
        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
            metadata_rows = _upsert_metadata_row(
                metadata_rows,
                _CSV_RESUME_METADATA_KEY,
                payload,
            )
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerows(rows)
        except Exception:
            pass





    def set_plot_smoothing(self, enabled=True, alpha=0.35, min_points=5):
        """Configure light EMA smoothing for plot lines."""
        self.plot_smoothing_enabled = bool(enabled)
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.35
        self.plot_smoothing_alpha = min(1.0, max(0.05, alpha))
        try:
            min_points = int(min_points)
        except (TypeError, ValueError):
            min_points = 5
        self.plot_smoothing_min_points = max(3, min_points)

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

    def _build_plot_context_lines(self, wrap_width=140):
        if self.run_context_lines:
            source_lines = self.run_context_lines
        elif self.run_context_text:
            source_lines = [self.run_context_text]
        else:
            return []
        wrapped_lines = []
        for line in source_lines:
            wrapped = textwrap.fill(
                str(line),
                width=max(50, int(wrap_width)),
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(wrapped.splitlines() or [""])
        return wrapped_lines

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





    def record_estimated_elo(self, iteration, estimated_elo, update_csv=True, std_error=None, ci95=None):
        """Record estimated Elo for a specific epoch/iteration (supports async updates)."""
        if estimated_elo is None:
            return

        try:
            iteration = int(iteration)
            estimated_elo = float(estimated_elo)
        except (TypeError, ValueError):
            return
        std_error = _clean_optional_float(std_error)
        ci_low, ci_high = _clean_elo_ci95(ci95)

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
        if std_error is not None or ci_low is not None or ci_high is not None:
            self.estimated_elo_errors[int(iteration)] = {
                'se': std_error,
                'ci95': (ci_low, ci_high),
            }

        if not update_csv:
            return

        # Backfill CSV row for this epoch if it already exists.
        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
            if not rows:
                return

            header = list(rows[0])
            for col in ('estimated_elo', 'estimated_elo_se', 'estimated_elo_ci95_low', 'estimated_elo_ci95_high'):
                if col not in header:
                    header.append(col)
                    for row in rows[1:]:
                        row.append('')
            elo_col = header.index('estimated_elo')
            se_col = header.index('estimated_elo_se')
            ci_low_col = header.index('estimated_elo_ci95_low')
            ci_high_col = header.index('estimated_elo_ci95_high')
            target_epoch = str(iteration)

            updated = False
            for row in rows[1:]:
                if not row:
                    continue
                if row[0] == target_epoch:
                    while len(row) <= elo_col:
                        row.append('')
                    while len(row) < len(header):
                        row.append('')
                    row[elo_col] = str(int(round(estimated_elo)))
                    if std_error is not None:
                        row[se_col] = f"{std_error:.1f}"
                    if ci_low is not None:
                        row[ci_low_col] = str(int(round(ci_low)))
                    if ci_high is not None:
                        row[ci_high_col] = str(int(round(ci_high)))
                    updated = True
                    break

            if updated:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(metadata_rows)
                    rows[0] = header
                    writer.writerows(rows)
        except Exception:
            # CSV backfill is best-effort only.
            pass

    def record_estimated_elo_mode(
        self,
        iteration,
        estimated_elo,
        *,
        mode="nn",
        simulations=0,
        update_csv=True,
        std_error=None,
        ci95=None,
    ):
        """Backfill mode-specific Elo columns in RL CSV."""
        if self.mode != "rl" or estimated_elo is None:
            return
        try:
            iteration = int(iteration)
            elo_value = int(round(float(estimated_elo)))
        except (TypeError, ValueError):
                return
        mode = "mcts" if str(mode).lower() == "mcts" else "nn"
        std_error = _clean_optional_float(std_error)
        ci_low, ci_high = _clean_elo_ci95(ci95)
        prefix = 'estimated_elo_mcts' if mode == "mcts" else 'estimated_elo_nn'
        pending = self._pending_rl_elo_by_iteration.setdefault(int(iteration), {})
        pending[prefix] = elo_value
        if std_error is not None:
            pending[f'{prefix}_se'] = f"{std_error:.1f}"
        if ci_low is not None:
            pending[f'{prefix}_ci95_low'] = str(int(round(ci_low)))
        if ci_high is not None:
            pending[f'{prefix}_ci95_high'] = str(int(round(ci_high)))
        if mode == "mcts":
            try:
                pending['estimated_elo_mcts_simulations'] = int(simulations or 0)
            except (TypeError, ValueError):
                pending['estimated_elo_mcts_simulations'] = ''
            pending['estimated_elo_mcts_by_simulations'] = _upsert_mcts_elo_by_simulations(
                pending.get('estimated_elo_mcts_by_simulations'),
                simulations,
                elo_value,
                std_error=std_error,
                ci95=(ci_low, ci_high),
            )
        if update_csv and self.csv_path.exists():
            try:
                metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
                if not rows:
                    return
                header = list(rows[0])
                for col in (
                    'estimated_elo_nn', 'estimated_elo_nn_se',
                    'estimated_elo_nn_ci95_low', 'estimated_elo_nn_ci95_high',
                    'estimated_elo_mcts', 'estimated_elo_mcts_se',
                    'estimated_elo_mcts_ci95_low', 'estimated_elo_mcts_ci95_high',
                    'estimated_elo_mcts_simulations',
                    'estimated_elo_mcts_by_simulations',
                ):
                    if col not in header:
                        header.append(col)
                        for row in rows[1:]:
                            row.append('')
                target_col = prefix
                target_idx = header.index(target_col)
                se_idx = header.index(f'{prefix}_se')
                ci_low_idx = header.index(f'{prefix}_ci95_low')
                ci_high_idx = header.index(f'{prefix}_ci95_high')
                sims_idx = header.index('estimated_elo_mcts_simulations')
                by_sims_idx = header.index('estimated_elo_mcts_by_simulations')
                iter_col = 'iteration' if 'iteration' in header else 'epoch'
                iter_idx = header.index(iter_col)
                for row in rows[1:]:
                    while len(row) < len(header):
                        row.append('')
                    try:
                        row_it = int(float(row[iter_idx]))
                    except (TypeError, ValueError):
                        continue
                    if row_it == iteration:
                        row[target_idx] = str(elo_value)
                        if std_error is not None:
                            row[se_idx] = f"{std_error:.1f}"
                        if ci_low is not None:
                            row[ci_low_idx] = str(int(round(ci_low)))
                        if ci_high is not None:
                            row[ci_high_idx] = str(int(round(ci_high)))
                        if mode == "mcts":
                            try:
                                row[sims_idx] = str(int(simulations or 0))
                            except (TypeError, ValueError):
                                row[sims_idx] = ''
                            row[by_sims_idx] = _upsert_mcts_elo_by_simulations(
                                row[by_sims_idx],
                                simulations,
                                elo_value,
                                std_error=std_error,
                                ci95=(ci_low, ci_high),
                            )
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(metadata_rows)
                    writer.writerow(header)
                    writer.writerows(rows[1:])
            except Exception:
                pass


    def get_latest_estimated_elo_with_epoch(self):
        """Return (epoch, elo) for latest known Elo, or (None, None)."""
        if not self.estimated_elos:
            return None, None
        epoch, elo = self.estimated_elos[-1]
        try:
            return int(epoch), float(elo)
        except (TypeError, ValueError):
            return None, None

    def _plot_elo_errorbars(self, ax, xs, ys, yerrs, color, *, label=None, x_offset=0.0, alpha=0.38):
        if label and "se" in str(label).lower():
            label = "_nolegend_"
        clean_xs, clean_ys, clean_errs = [], [], []
        for x, y, err in zip(xs or [], ys or [], yerrs or []):
            err_value = _clean_optional_float(err)
            if err_value is None or err_value <= 0.0:
                continue
            try:
                clean_xs.append(float(x) + float(x_offset))
                clean_ys.append(float(y))
                clean_errs.append(float(err_value))
            except (TypeError, ValueError):
                continue
        if not clean_xs:
            return
        ax.errorbar(
            clean_xs,
            clean_ys,
            yerr=clean_errs,
            fmt='none',
            ecolor=color,
            elinewidth=1.35,
            capsize=3.0,
            capthick=1.0,
            alpha=alpha,
            label=label,
            zorder=2,
        )





    
    def log(self, iteration, train_losses=None, val_losses=None,
            train_metrics=None, val_metrics=None, lr=None, estimated_elo=None, **kwargs):
        """Dispatch one canonical row to the domain logger."""
        method = self._log_il if self.mode == 'il' else self._log_rl
        return method(
            iteration, train_losses=train_losses, val_losses=val_losses,
            train_metrics=train_metrics, val_metrics=val_metrics, lr=lr,
            estimated_elo=estimated_elo, **kwargs,
        )
    
    def plot(self):
        """Generate training plots"""
        if self.mode == "il":
            if (
                len(self.iterations) < 1
                and not self.estimated_elos
                and not self.elo_epoch_markers
                and not self.il_mode_elo_markers
            ):
                return
            self._plot_il()
        else:
            if len(self.iterations) < 2:
                return
            self._plot_rl()
