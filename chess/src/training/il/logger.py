"""IL-specific CSV logging, history import and plots."""

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.common.logger_helpers import (
    _CSV_RUN_SUMMARY_METADATA_KEY,
    _CSV_RESUME_METADATA_KEY,
    _clean_optional_float,
    _clean_elo_ci95,
    _format_million_count,
    _read_csv_rows_preserving_metadata,
    _read_csv_dict_rows,
    _metadata_json_payload,
    _parse_mcts_elo_by_simulations,
    _csv_float,
    _csv_int,
    _upsert_mcts_elo_by_simulations,
    _apply_sorted_legend,
)
from src.common.dashboard_layout import create_dashboard, finish_dashboard


IL_GRADIENT_METRIC_KEYS = (
    'grad_total_norm',
    'grad_clip_fraction',
    'grad_clip_scale_mean',
    'grad_backbone_norm',
    'grad_policy_head_norm',
    'grad_value_head_norm',
    'grad_policy_probe_norm',
    'grad_value_probe_norm',
    'grad_policy_value_cosine',
)
IL_GRADIENT_COLUMNS = tuple(f'train_{key}' for key in IL_GRADIENT_METRIC_KEYS)


class ILLoggerMixin:
    def _ensure_il_gradient_storage(self):
        for key in IL_GRADIENT_METRIC_KEYS:
            attr = f'train_{key}'
            if not hasattr(self, attr):
                setattr(self, attr, [])

    def import_il_history_from_csv(self, source_csv_path, completed_epoch):
        """Seed this IL logger from a previous CSV and keep rows up to completed_epoch."""
        if self.mode != "il" or source_csv_path is None:
            return False
        source_csv_path = Path(source_csv_path)
        if not source_csv_path.exists() or source_csv_path.resolve() == self.csv_path.resolve():
            return False
        try:
            completed_epoch = int(completed_epoch)
        except (TypeError, ValueError):
            return False
        if completed_epoch < 1:
            return False

        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(source_csv_path)
        except Exception:
            return False
        if not rows:
            return False

        header = list(rows[0])
        epoch_idx = header.index('epoch') if 'epoch' in header else None
        if epoch_idx is None:
            return False

        kept_rows = []
        for raw_row in rows[1:]:
            row = list(raw_row)
            if len(row) <= epoch_idx:
                continue
            try:
                epoch_value = int(float(row[epoch_idx]))
            except (TypeError, ValueError):
                continue
            if epoch_value <= completed_epoch:
                kept_rows.append(row)

        if not kept_rows:
            return False

        # Older runs predate optimizer telemetry.  Extend both the copied
        # header and historical rows so resumed epochs remain column-aligned.
        missing_gradient_columns = [
            column for column in IL_GRADIENT_COLUMNS if column not in header
        ]
        if missing_gradient_columns:
            old_width = len(header)
            header.extend(missing_gradient_columns)
            for row in kept_rows:
                if len(row) < old_width:
                    row.extend([''] * (old_width - len(row)))
                row.extend([''] * len(missing_gradient_columns))

        try:
            shutil.copy2(source_csv_path, self.csv_path.with_suffix(self.csv_path.suffix + ".pre_resume_copy"))
        except Exception:
            pass

        self.resume_source_csv_path = str(source_csv_path)
        self.resume_source_summary = _metadata_json_payload(metadata_rows, _CSV_RUN_SUMMARY_METADATA_KEY)
        resume_payload = _metadata_json_payload(metadata_rows, _CSV_RESUME_METADATA_KEY) or {}
        self.resume_history_epoch = completed_epoch
        self._reset_il_plot_storage()

        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerow(header)
                writer.writerows(kept_rows)
        except Exception:
            return False

        dict_rows = []
        for raw_row in kept_rows:
            row = list(raw_row)
            if len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            dict_rows.append(dict(zip(header, row[:len(header)])))
        self._load_il_plot_history(dict_rows)
        for marker in resume_payload.get('final_elo_markers') or []:
            if not isinstance(marker, dict):
                continue
            self.record_il_mode_elo(
                marker.get('epoch'),
                marker.get('elo'),
                mode=marker.get('mode', 'nn'),
                simulations=marker.get('simulations', 0),
                label=marker.get('label'),
                update_csv=False,
                std_error=marker.get('std_error'),
                ci95=marker.get('ci95'),
            )
        existing_resume_markers = resume_payload.get('markers') or []
        if isinstance(existing_resume_markers, list):
            for marker in existing_resume_markers:
                if not isinstance(marker, dict):
                    continue
                try:
                    completed = int(marker.get('completed_epoch'))
                except (TypeError, ValueError):
                    continue
                self.add_resume_marker(completed, marker.get('label'))
        self.add_resume_marker(completed_epoch)
        return True

    def _reset_il_plot_storage(self):
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
        for name in [
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
        ]:
            setattr(self, name, [])
        self.estimated_elos = []
        self.estimated_elo_errors = {}
        self.il_mode_elo_markers = []
        for key in IL_GRADIENT_METRIC_KEYS:
            setattr(self, f'train_{key}', [])

    def _load_il_plot_history(self, rows):
        for row in rows:
            epoch = _csv_int(row, 'epoch')
            if epoch is None:
                epoch_text = str(row.get('epoch', '')).strip().upper()
                if epoch_text == 'SWA':
                    swa_epoch = self.estimated_elos[-1][0] if self.estimated_elos else (self.iterations[-1] if self.iterations else 0)
                    swa_nn_elo = _csv_float(row, 'estimated_elo_nn')
                    swa_mcts_elo = _csv_float(row, 'estimated_elo_mcts')
                    legacy_swa_elo = _csv_float(row, 'estimated_elo')
                    self.swa_metrics = {
                        'epoch': swa_epoch,
                        'val_loss': _csv_float(row, 'val_loss'),
                        'val_policy_loss': _csv_float(row, 'val_policy_loss'),
                        'val_value_loss': _csv_float(row, 'val_value_loss'),
                        'top1': _csv_float(row, 'val_policy_top1'),
                        'top3': _csv_float(row, 'val_policy_top3'),
                        'mae': _csv_float(row, 'val_value_mae'),
                        'wdl_acc': _csv_float(row, 'val_value_wdl_acc'),
                        'wdl_ce': _csv_float(row, 'val_value_wdl_ce'),
                    }
                    if swa_nn_elo is not None:
                        self.swa_metrics['elo'] = swa_nn_elo
                        self.swa_elo_info = (swa_epoch, swa_nn_elo)
                        self.record_il_mode_elo(
                            swa_epoch,
                            swa_nn_elo,
                            mode='nn',
                            label=str(row.get('estimated_elo_nn_label') or 'SWA final NN'),
                            update_csv=False,
                        )
                    elif legacy_swa_elo is not None:
                        self.swa_metrics['elo_mcts'] = legacy_swa_elo
                        self.swa_elo_info = (swa_epoch, legacy_swa_elo)
                    if swa_mcts_elo is not None:
                        self.swa_metrics['elo_mcts'] = swa_mcts_elo
                        self.record_il_mode_elo(
                            swa_epoch,
                            swa_mcts_elo,
                            mode='mcts',
                            simulations=_csv_int(row, 'estimated_elo_mcts_simulations', 0),
                            label=str(row.get('estimated_elo_mcts_label') or 'SWA final MCTS'),
                            update_csv=False,
                        )
                continue
            self.iterations.append(epoch)
            self.train_losses.append(_csv_float(row, 'train_loss', 0.0))
            self.train_policy_losses.append(_csv_float(row, 'train_policy_loss', 0.0))
            self.train_value_losses.append(_csv_float(row, 'train_value_loss', 0.0))
            self.train_moves_left_losses.append(_csv_float(row, 'train_moves_left_loss', 0.0))
            if _csv_float(row, 'val_loss') is not None:
                self.val_iterations.append(epoch)
                self.val_losses.append(_csv_float(row, 'val_loss', 0.0))
                self.val_policy_losses.append(_csv_float(row, 'val_policy_loss', 0.0))
                self.val_value_losses.append(_csv_float(row, 'val_value_loss', 0.0))
                self.val_moves_left_losses.append(_csv_float(row, 'val_moves_left_loss', 0.0))
            for attr, col in [
                ('train_policy_top1', 'train_policy_top1'),
                ('train_policy_top3', 'train_policy_top3'),
                ('train_policy_target_mass_top1', 'train_policy_target_mass_top1'),
                ('train_policy_target_mass_top3', 'train_policy_target_mass_top3'),
                ('train_policy_target_mass_top5', 'train_policy_target_mass_top5'),
                ('train_policy_entropy', 'train_policy_entropy'),
                ('train_policy_effective_moves', 'train_policy_effective_moves'),
                ('train_policy_top1_prob', 'train_policy_top1_prob'),
                ('train_policy_legal_entropy', 'train_policy_legal_entropy'),
                ('train_policy_legal_effective_moves', 'train_policy_legal_effective_moves'),
                ('train_policy_legal_top1_prob', 'train_policy_legal_top1_prob'),
                ('train_policy_legal_top1_margin', 'train_policy_legal_top1_margin'),
                ('train_policy_target_entropy', 'train_policy_target_entropy'),
                ('train_policy_target_effective_moves', 'train_policy_target_effective_moves'),
                ('train_policy_target_top1_mass', 'train_policy_target_top1_mass'),
                ('train_policy_target_support_top1_prob', 'train_policy_target_support_top1_prob'),
                ('train_policy_target_support_top1_margin', 'train_policy_target_support_top1_margin'),
                ('train_value_mae', 'train_value_mae'),
                ('train_value_mae_opening', 'train_value_mae_opening'),
                ('train_value_mae_middlegame', 'train_value_mae_middlegame'),
                ('train_value_mae_endgame', 'train_value_mae_endgame'),
                ('train_value_wdl_acc', 'train_value_wdl_acc'),
                ('train_value_wdl_ce', 'train_value_wdl_ce'),
                ('train_value_wdl_ce_opening', 'train_value_wdl_ce_opening'),
                ('train_value_wdl_ce_middlegame', 'train_value_wdl_ce_middlegame'),
                ('train_value_wdl_ce_endgame', 'train_value_wdl_ce_endgame'),
                ('train_value_std_ratio_opening', 'train_value_std_ratio_opening'),
                ('train_value_std_ratio_middlegame', 'train_value_std_ratio_middlegame'),
                ('train_value_std_ratio_endgame', 'train_value_std_ratio_endgame'),
                ('train_moves_left_loss_opening', 'train_moves_left_loss_opening'),
                ('train_moves_left_loss_middlegame', 'train_moves_left_loss_middlegame'),
                ('train_moves_left_loss_endgame', 'train_moves_left_loss_endgame'),
                ('train_moves_left_mae', 'train_moves_left_mae'),
                ('train_moves_left_mae_opening', 'train_moves_left_mae_opening'),
                ('train_moves_left_mae_middlegame', 'train_moves_left_mae_middlegame'),
                ('train_moves_left_mae_endgame', 'train_moves_left_mae_endgame'),
                ('train_soft_occurrence_avg', 'train_soft_occurrence_avg'),
                ('train_soft_occurrence_max', 'train_soft_occurrence_max'),
                ('train_soft_sample_weight_avg', 'train_soft_sample_weight_avg'),
                ('train_soft_policy_mass_kept_avg', 'train_soft_policy_mass_kept_avg'),
                ('train_soft_policy_mass_kept_min', 'train_soft_policy_mass_kept_min'),
            ]:
                getattr(self, attr).append(_csv_float(row, col, 0.0))
            self._ensure_il_gradient_storage()
            for key, column in zip(IL_GRADIENT_METRIC_KEYS, IL_GRADIENT_COLUMNS):
                getattr(self, f'train_{key}').append(_csv_float(row, column, None))
            # Older CSVs lack a validity marker, so all-zero probe triplets
            # cannot be distinguished from unsupported probes. Treat them as
            # missing instead of presenting them as evidence of neutral gradients.
            policy_probe = _csv_float(row, 'train_grad_policy_probe_norm')
            value_probe = _csv_float(row, 'train_grad_value_probe_norm')
            if not (policy_probe is not None and policy_probe > 0.0) and not (
                value_probe is not None and value_probe > 0.0
            ):
                policy_probe = None
                value_probe = None
            cosine = _csv_float(row, 'train_grad_policy_value_cosine')
            if (
                policy_probe is None
                or value_probe is None
                or policy_probe <= 0.0
                or value_probe <= 0.0
            ):
                cosine = None
            self.train_grad_policy_probe_norm[-1] = policy_probe
            self.train_grad_value_probe_norm[-1] = value_probe
            self.train_grad_policy_value_cosine[-1] = cosine
            if _csv_float(row, 'val_loss') is not None:
                for attr, col in [
                    ('val_policy_top1', 'val_policy_top1'),
                    ('val_policy_top3', 'val_policy_top3'),
                    ('val_policy_target_mass_top1', 'val_policy_target_mass_top1'),
                    ('val_policy_target_mass_top3', 'val_policy_target_mass_top3'),
                    ('val_policy_target_mass_top5', 'val_policy_target_mass_top5'),
                    ('val_policy_entropy', 'val_policy_entropy'),
                    ('val_policy_effective_moves', 'val_policy_effective_moves'),
                    ('val_policy_top1_prob', 'val_policy_top1_prob'),
                    ('val_policy_legal_entropy', 'val_policy_legal_entropy'),
                    ('val_policy_legal_effective_moves', 'val_policy_legal_effective_moves'),
                    ('val_policy_legal_top1_prob', 'val_policy_legal_top1_prob'),
                    ('val_policy_legal_top1_margin', 'val_policy_legal_top1_margin'),
                    ('val_policy_target_entropy', 'val_policy_target_entropy'),
                    ('val_policy_target_effective_moves', 'val_policy_target_effective_moves'),
                    ('val_policy_target_top1_mass', 'val_policy_target_top1_mass'),
                    ('val_policy_target_support_top1_prob', 'val_policy_target_support_top1_prob'),
                    ('val_policy_target_support_top1_margin', 'val_policy_target_support_top1_margin'),
                    ('val_value_mae', 'val_value_mae'),
                    ('val_value_mae_opening', 'val_value_mae_opening'),
                    ('val_value_mae_middlegame', 'val_value_mae_middlegame'),
                    ('val_value_mae_endgame', 'val_value_mae_endgame'),
                    ('val_value_wdl_acc', 'val_value_wdl_acc'),
                    ('val_value_wdl_ce', 'val_value_wdl_ce'),
                    ('val_value_wdl_ce_opening', 'val_value_wdl_ce_opening'),
                    ('val_value_wdl_ce_middlegame', 'val_value_wdl_ce_middlegame'),
                    ('val_value_wdl_ce_endgame', 'val_value_wdl_ce_endgame'),
                    ('val_value_std_ratio_opening', 'val_value_std_ratio_opening'),
                    ('val_value_std_ratio_middlegame', 'val_value_std_ratio_middlegame'),
                    ('val_value_std_ratio_endgame', 'val_value_std_ratio_endgame'),
                    ('val_moves_left_loss_opening', 'val_moves_left_loss_opening'),
                    ('val_moves_left_loss_middlegame', 'val_moves_left_loss_middlegame'),
                    ('val_moves_left_loss_endgame', 'val_moves_left_loss_endgame'),
                    ('val_moves_left_mae', 'val_moves_left_mae'),
                    ('val_moves_left_mae_opening', 'val_moves_left_mae_opening'),
                    ('val_moves_left_mae_middlegame', 'val_moves_left_mae_middlegame'),
                    ('val_moves_left_mae_endgame', 'val_moves_left_mae_endgame'),
                    ('val_soft_occurrence_avg', 'val_soft_occurrence_avg'),
                    ('val_soft_occurrence_max', 'val_soft_occurrence_max'),
                    ('val_soft_sample_weight_avg', 'val_soft_sample_weight_avg'),
                    ('val_soft_policy_mass_kept_avg', 'val_soft_policy_mass_kept_avg'),
                    ('val_soft_policy_mass_kept_min', 'val_soft_policy_mass_kept_min'),
                ]:
                    getattr(self, attr).append(_csv_float(row, col, 0.0))

            nn_elo = _csv_float(row, 'estimated_elo_nn')
            mcts_elo = _csv_float(row, 'estimated_elo_mcts')
            elo = _csv_float(row, 'estimated_elo')
            elo_is_mode_duplicate = (
                elo is not None
                and (
                    (nn_elo is not None and abs(float(elo) - float(nn_elo)) <= 0.5)
                    or (mcts_elo is not None and abs(float(elo) - float(mcts_elo)) <= 0.5)
                )
            )
            if elo is not None and not elo_is_mode_duplicate:
                self.record_estimated_elo(
                    epoch,
                    elo,
                    update_csv=False,
                    std_error=_csv_float(row, 'estimated_elo_se'),
                    ci95=(
                        _csv_float(row, 'estimated_elo_ci95_low'),
                        _csv_float(row, 'estimated_elo_ci95_high'),
                    ),
                )
            if nn_elo is not None:
                nn_label = str(row.get('estimated_elo_nn_label') or '').strip()
                if not nn_label:
                    if elo is not None and abs(float(elo) - float(nn_elo)) <= 0.5:
                        nn_label = 'Best NN'
                    else:
                        nn_label = 'NN'
                nn_marker_epoch = epoch
                if self.estimated_elos and any(token in nn_label.lower() for token in ('best', 'final', 'swa')):
                    latest_regular_epoch = int(self.estimated_elos[-1][0])
                    if int(epoch) > latest_regular_epoch:
                        nn_marker_epoch = latest_regular_epoch
                self.record_il_mode_elo(
                    nn_marker_epoch,
                    nn_elo,
                    mode='nn',
                    label=nn_label,
                    update_csv=False,
                    std_error=_csv_float(row, 'estimated_elo_nn_se'),
                    ci95=(
                        _csv_float(row, 'estimated_elo_nn_ci95_low'),
                        _csv_float(row, 'estimated_elo_nn_ci95_high'),
                    ),
                )
            if mcts_elo is not None:
                mcts_label = str(row.get('estimated_elo_mcts_label') or '').strip()
                if not mcts_label:
                    if elo is not None and abs(float(elo) - float(mcts_elo)) <= 0.5:
                        mcts_label = 'Best MCTS'
                    else:
                        mcts_label = 'MCTS'
                mcts_marker_epoch = epoch
                if self.estimated_elos and any(token in mcts_label.lower() for token in ('best', 'final', 'swa')):
                    latest_regular_epoch = int(self.estimated_elos[-1][0])
                    if int(epoch) > latest_regular_epoch:
                        mcts_marker_epoch = latest_regular_epoch
                self.record_il_mode_elo(
                    mcts_marker_epoch,
                    mcts_elo,
                    mode='mcts',
                    simulations=_csv_int(row, 'estimated_elo_mcts_simulations', 0),
                    label=mcts_label,
                    update_csv=False,
                    std_error=_csv_float(row, 'estimated_elo_mcts_se'),
                    ci95=(
                        _csv_float(row, 'estimated_elo_mcts_ci95_low'),
                        _csv_float(row, 'estimated_elo_mcts_ci95_high'),
                    ),
                )
            for simulations, sim_entry in sorted(
                _parse_mcts_elo_by_simulations(
                    row.get("estimated_elo_mcts_by_simulations")
                ).items()
            ):
                self.record_il_mode_elo(
                    epoch,
                    sim_entry["elo"],
                    mode="mcts",
                    simulations=simulations,
                    label=f"Final MCTS @{simulations}",
                    update_csv=False,
                    std_error=sim_entry.get("se"),
                    ci95=sim_entry.get("ci95"),
                )
        self._ensure_best_nn_marker_from_regular_elo()

    def _ensure_best_nn_marker_from_regular_elo(self):
        if self.mode != "il" or not self.estimated_elos or not self.val_losses:
            return
        for marker in self.il_mode_elo_markers:
            if marker.get('mode') == 'nn' and 'best' in str(marker.get('label', '')).lower():
                return
        try:
            best_idx = min(range(len(self.val_losses)), key=lambda i: self.val_losses[i])
            best_epoch = int(self.val_iterations[best_idx]) if best_idx < len(self.val_iterations) else int(self.iterations[best_idx])
        except (TypeError, ValueError, IndexError):
            return
        try:
            source_epoch, elo = min(self.estimated_elos, key=lambda pair: abs(int(pair[0]) - best_epoch))
            marker_epoch = int(self.estimated_elos[-1][0])
        except (TypeError, ValueError):
            return
        error_info = (self.estimated_elo_errors.get(int(source_epoch), {}) or {})
        self.record_il_mode_elo(
            marker_epoch,
            elo,
            mode='nn',
            label='Best NN',
            update_csv=False,
            std_error=error_info.get('se'),
            ci95=error_info.get('ci95'),
        )

    def record_best_final_elo(self, epoch, elo):
        """Store exact final best-model Elo for the IL summary and Elo panel."""
        if self.mode != "il":
            return
        try:
            epoch = int(epoch)
            elo = float(elo)
        except (TypeError, ValueError):
            return
        self.best_final_elo_info = (epoch, elo)

    def record_il_mode_elo(
        self,
        epoch,
        elo,
        *,
        mode="nn",
        simulations=0,
        label=None,
        update_csv=True,
        std_error=None,
        ci95=None,
    ):
        """Store an IL raw-NN or MCTS Elo marker for the Elo panel."""
        if self.mode != "il" or elo is None:
            return
        try:
            epoch = int(epoch)
            elo_value = float(elo)
        except (TypeError, ValueError):
            return

        mode = "mcts" if str(mode).lower() == "mcts" else "nn"
        std_error = _clean_optional_float(std_error)
        ci_low, ci_high = _clean_elo_ci95(ci95)
        try:
            simulations = int(simulations or 0)
        except (TypeError, ValueError):
            simulations = 0
        label_text = str(label).strip() if label is not None else ""
        if not label_text:
            label_text = "MCTS" if mode == "mcts" else "NN"

        marker = {
            "epoch": epoch,
            "elo": elo_value,
            "mode": mode,
            "simulations": simulations,
            "label": label_text,
            "std_error": std_error,
            "ci95": (ci_low, ci_high),
        }

        replaced = False
        for idx, existing in enumerate(self.il_mode_elo_markers):
            if (
                int(existing.get("epoch", -1)) == epoch
                and existing.get("mode") == mode
                and (
                    mode == "nn"
                    or int(existing.get("simulations", 0) or 0) == simulations
                )
            ):
                self.il_mode_elo_markers[idx] = marker
                replaced = True
                break
        if not replaced:
            self.il_mode_elo_markers.append(marker)
            self.il_mode_elo_markers.sort(key=lambda item: (int(item["epoch"]), item["mode"], item["label"]))

        if not update_csv or not self.csv_path.exists():
            return

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
                'estimated_elo_nn_label', 'estimated_elo_mcts_label',
            ):
                if col not in header:
                    header.append(col)
                    for row in rows[1:]:
                        row.append('')
            iter_col = 'epoch' if 'epoch' in header else 'iteration'
            iter_idx = header.index(iter_col)
            target_col = 'estimated_elo_mcts' if mode == "mcts" else 'estimated_elo_nn'
            target_idx = header.index(target_col)
            se_idx = header.index(f'{target_col}_se')
            ci_low_idx = header.index(f'{target_col}_ci95_low')
            ci_high_idx = header.index(f'{target_col}_ci95_high')
            sims_idx = header.index('estimated_elo_mcts_simulations')
            by_sims_idx = header.index('estimated_elo_mcts_by_simulations')
            label_col = 'estimated_elo_mcts_label' if mode == "mcts" else 'estimated_elo_nn_label'
            label_idx = header.index(label_col)

            target_row = None
            for row in rows[1:]:
                while len(row) < len(header):
                    row.append('')
                try:
                    row_epoch = int(float(row[iter_idx]))
                except (TypeError, ValueError):
                    continue
                if row_epoch == epoch:
                    target_row = row
                    break
            if target_row is None:
                target_row = [''] * len(header)
                target_row[iter_idx] = str(epoch)
                rows.append(target_row)

            target_row[target_idx] = str(int(round(elo_value)))
            if std_error is not None:
                target_row[se_idx] = f"{std_error:.1f}"
            if ci_low is not None:
                target_row[ci_low_idx] = str(int(round(ci_low)))
            if ci_high is not None:
                target_row[ci_high_idx] = str(int(round(ci_high)))
            if mode == "mcts":
                target_row[sims_idx] = str(simulations) if simulations > 0 else ''
                target_row[by_sims_idx] = _upsert_mcts_elo_by_simulations(
                    target_row[by_sims_idx],
                    simulations,
                    elo_value,
                    std_error=std_error,
                    ci95=(ci_low, ci_high),
                )
            target_row[label_idx] = label_text
            rows[0] = header
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerows(rows)
        except Exception:
            pass
        if self.resume_source_csv_path is not None or self.resume_markers:
            self._write_resume_metadata()

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
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
        except Exception:
            return

        if not rows:
            return
        header = rows[0]
        for col in (
            'estimated_elo_nn', 'estimated_elo_nn_se',
            'estimated_elo_nn_ci95_low', 'estimated_elo_nn_ci95_high',
            'estimated_elo_mcts', 'estimated_elo_mcts_se',
            'estimated_elo_mcts_ci95_low', 'estimated_elo_mcts_ci95_high',
            'estimated_elo_mcts_simulations',
            'estimated_elo_mcts_by_simulations',
            'estimated_elo_nn_label', 'estimated_elo_mcts_label',
        ):
            if col not in header:
                header.append(col)
                for existing_row in rows[1:]:
                    existing_row.append('')

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
        _set('val_value_wdl_acc', _s(m.get('wdl_acc')))
        _set('val_value_wdl_ce',  _s(m.get('wdl_ce')))
        _set('estimated_elo_nn',  _s(m.get('elo'), '{:.0f}'))
        if m.get('elo') is not None:
            _set('estimated_elo_nn_label', 'SWA final NN')

        # Remove any existing SWA row, then append new one
        data_rows = [r for r in rows[1:] if not r or r[0] != 'SWA']
        data_rows.append(row)

        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerow(header)
                writer.writerows(data_rows)
        except Exception:
            pass

    def _il_elo_visual_data(self):
        """Return Elo data with SWA-only records removed from the dashboard."""
        swa_epoch = None
        swa_elo = None
        if self.swa_elo_info:
            try:
                swa_epoch, swa_elo = int(self.swa_elo_info[0]), float(self.swa_elo_info[1])
            except (TypeError, ValueError):
                swa_epoch = swa_elo = None

        def is_swa_point(epoch, elo):
            if swa_epoch is None or swa_elo is None:
                return False
            try:
                return int(epoch) == swa_epoch and abs(float(elo) - swa_elo) <= 0.5
            except (TypeError, ValueError):
                return False

        elo_points = [
            (epoch, value) for epoch, value in self.estimated_elos
            if not is_swa_point(epoch, value)
        ]
        epoch_markers = []
        for epoch, label in self.elo_epoch_markers:
            label_text = str(label or "")
            try:
                marker_epoch = int(epoch)
            except (TypeError, ValueError):
                continue
            if "swa" in label_text.lower() or (swa_epoch is not None and marker_epoch == swa_epoch):
                continue
            epoch_markers.append((marker_epoch, label_text))

        mode_markers = []
        for marker in self.il_mode_elo_markers:
            if "swa" in str(marker.get("label", "")).lower():
                continue
            if is_swa_point(marker.get("epoch"), marker.get("elo")):
                continue
            mode_markers.append(marker)
        return elo_points, epoch_markers, mode_markers

    def _plot_il_summary_panel(self, ax):
        """Compare the best-loss and latest validation checkpoints."""
        ax.axis('off')
        valid_indices = range(min(len(self.val_losses), len(self.val_iterations)))
        if not valid_indices:
            ax.set_title('Best vs Latest Validation', fontsize=14, fontweight='bold', pad=8)
            return

        best_idx = min(valid_indices, key=lambda index: self.val_losses[index])
        best_epoch = self.val_iterations[best_idx]
        latest_idx = max(
            valid_indices,
            key=lambda index: (int(self.val_iterations[index]), index),
        )
        latest_epoch = self.val_iterations[latest_idx]

        def at(series, index):
            return series[index] if len(series) > index else None

        def number(value, digits=4):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "—"
            return f"{value:.{digits}f}" if value == value else "—"

        def percent(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "—"
            return f"{value:.2%}" if value == value else "—"

        elo_points, _, mode_markers = self._il_elo_visual_data()
        def elo_at(epoch, mode):
            if mode == 'nn' and self.best_final_elo_info:
                try:
                    final_epoch, final_elo = self.best_final_elo_info
                    if int(final_epoch) == int(epoch):
                        return final_elo
                except (TypeError, ValueError):
                    pass
            candidates = []
            for marker in mode_markers:
                if marker.get('mode') != mode:
                    continue
                try:
                    if int(marker.get('epoch')) != int(epoch):
                        continue
                    candidates.append((int(marker.get('simulations', 0) or 0), marker.get('elo')))
                except (TypeError, ValueError):
                    continue
            if candidates:
                return max(candidates, key=lambda candidate: candidate[0])[1]
            if mode == 'nn':
                for point_epoch, value in reversed(elo_points):
                    try:
                        if int(point_epoch) == int(epoch):
                            return value
                    except (TypeError, ValueError):
                        continue
            return None

        best_nn_elo = elo_at(best_epoch, 'nn')
        latest_nn_elo = elo_at(latest_epoch, 'nn')
        best_mcts_elo = elo_at(best_epoch, 'mcts')
        latest_mcts_elo = elo_at(latest_epoch, 'mcts')
        elo_text = lambda value: number(value, digits=0) if value is not None else ''

        rows = [
            ('Validation loss', number(at(self.val_losses, best_idx)), number(at(self.val_losses, latest_idx))),
            ('Policy top-1', percent(at(self.val_policy_top1, best_idx)), percent(at(self.val_policy_top1, latest_idx))),
            ('Policy top-3', percent(at(self.val_policy_top3, best_idx)), percent(at(self.val_policy_top3, latest_idx))),
            ('Value MAE', number(at(self.val_value_mae, best_idx)), number(at(self.val_value_mae, latest_idx))),
            ('WDL accuracy', percent(at(self.val_value_wdl_acc, best_idx)), percent(at(self.val_value_wdl_acc, latest_idx))),
            ('WDL cross-entropy', number(at(self.val_value_wdl_ce, best_idx)), number(at(self.val_value_wdl_ce, latest_idx))),
            ('Estimated Elo NN', elo_text(best_nn_elo), elo_text(latest_nn_elo)),
            ('Estimated Elo MCTS', elo_text(best_mcts_elo), elo_text(latest_mcts_elo)),
        ]

        ax.set_title(
            f'Validation · Best ep {best_epoch} vs Latest ep {latest_epoch}',
            fontsize=12,
            fontweight='bold',
            pad=8,
        )
        table = ax.table(
            cellText=rows,
            colLabels=['Metric', f'Best · ep {best_epoch}', f'Latest · ep {latest_epoch}'],
            cellLoc='center',
            loc='center',
            bbox=[0.02, 0.04, 0.96, 0.92],
            colWidths=[0.46, 0.27, 0.27],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for column, color in enumerate(('#424242', '#2E7D32', '#1565C0')):
            cell = table[0, column]
            cell.set_facecolor(color)
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_height(cell.get_height() * 1.4)
        for row_index, (label, _, _) in enumerate(rows, start=1):
            background = '#F5F5F5' if row_index % 2 else '#FFFFFF'
            for column in range(3):
                table[row_index, column].set_facecolor(background)
            table[row_index, 0].set_text_props(fontweight='bold', ha='left')

    def _plot_il_elo_panel(self, ax):
        """Render non-SWA Elo estimates and checkpoints."""
        elo_points, epoch_markers, mode_markers = self._il_elo_visual_data()
        has_best_final = bool(self.best_final_elo_info)
        if not elo_points and not epoch_markers and not mode_markers and not has_best_final:
            ax.axis('off')
            return

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Elo')
        ax.set_title('Estimated Elo (vs Stockfish)')
        ax.grid(True, alpha=0.3)

        if self.iterations:
            x_min, x_max = min(self.iterations), max(self.iterations)
            if x_min == x_max:
                x_min -= 1
                x_max += 1
        else:
            x_min = x_max = None

        visible_elos = elo_points
        if x_min is not None and x_max is not None:
            visible_elos = [(epoch, value) for epoch, value in elo_points if x_min <= int(epoch) <= x_max]
            if not visible_elos and elo_points:
                visible_elos = elo_points[-1:]
        if visible_elos:
            epochs, values = zip(*visible_elos)
            ax.plot(epochs, values, 'go-', label='Estimated Elo', linewidth=2, markersize=7)
            errors = [(self.estimated_elo_errors.get(int(epoch), {}) or {}).get('se') for epoch, _ in visible_elos]
            self._plot_elo_errorbars(ax, epochs, values, errors, 'green', label='Estimated Elo ±SE')
            for reference, label in [(1200, 'Beginner'), (1500, 'Club'), (1800, 'Expert'), (2000, 'Candidate Master')]:
                if min(values) - 200 <= reference <= max(values) + 200:
                    ax.axhline(reference, color='gray', linestyle=':', alpha=0.4)
                    ax.text(epochs[0], reference + 15, label, fontsize=8, color='gray', alpha=0.6)

        for marker in mode_markers:
            try:
                epoch = int(marker.get('epoch'))
                elo = float(marker.get('elo'))
            except (TypeError, ValueError):
                continue
            mode = marker.get('mode', 'nn')
            label_text = str(marker.get('label', '')).strip()
            marker_kind = next((kind for kind in ('Best', 'Current', 'Resume', 'Final') if kind.lower() in label_text.lower()), 'Checkpoint')
            sims = int(marker.get('simulations', 0) or 0)
            suffix = f' @{sims}' if mode == 'mcts' and sims > 0 else ''
            display = f'{marker_kind} {mode.upper()}{suffix} ({int(round(elo))})'
            color = '#2563EB' if mode == 'nn' else '#7C3AED'
            ax.plot(epoch, elo, marker='D' if mode == 'nn' else 'P', markersize=9,
                    color=color, linestyle='None', label=display, zorder=6)
            self._plot_elo_errorbars(
                ax, [epoch], [elo], [marker.get('std_error')], color,
                label=f'{marker_kind} {mode.upper()}{suffix} ±SE', x_offset=0.0, alpha=0.45,
            )

        if self.best_final_elo_info:
            best_epoch, best_elo = self.best_final_elo_info
            ax.plot(best_epoch, best_elo, marker='D', markersize=9, color='#2563EB',
                    markeredgecolor='#1E3A8A', markeredgewidth=1.2, linestyle='None',
                    label=f'Best final ({int(round(best_elo))})', zorder=7)

        for marker_epoch, marker_label in epoch_markers:
            ax.axvline(marker_epoch, color='black', linestyle='--', alpha=0.55,
                       linewidth=1.4, label=marker_label)
            if visible_elos:
                _, nearest_elo = min(visible_elos, key=lambda pair: abs(int(pair[0]) - marker_epoch))
                ax.annotate(marker_label, xy=(marker_epoch, nearest_elo), xytext=(4, 8),
                            textcoords='offset points', fontsize=8, color='black')

        if x_min is not None and x_max is not None:
            for marker in mode_markers:
                try:
                    marker_epoch = int(marker.get('epoch'))
                except (TypeError, ValueError):
                    continue
                if x_min <= marker_epoch and marker_epoch >= x_max - 1:
                    x_max = marker_epoch + max(2, int((x_max - x_min) * 0.08) + 1)
            ax.set_xlim(x_min, x_max)
        _apply_sorted_legend(ax, fontsize=8, loc='lower right')

    def _plot_il(self):
        """Plot IL training progress"""
        context_lines = self._build_plot_context_lines(wrap_width=150)
        elo_points, elo_epoch_markers, elo_mode_markers = self._il_elo_visual_data()
        has_elo_data = bool(
            elo_points or elo_epoch_markers or elo_mode_markers or self.best_final_elo_info
        )
        outcome_panels = [(5, 0)]
        if has_elo_data:
            outcome_panels.append((5, 1))
        outcome_panels.append((5, 2))
        sections = [
            ('Learning and Losses', [(0, 0), (0, 1), (0, 2), (0, 3)], 2),
            ('Policy Quality', [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)], 3),
            ('Value and WDL', [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)], 3),
            ('Moves-Left Auxiliary Task', [(3, 2)], 1),
            ('Optimizer Diagnostics', [(4, 0), (4, 1), (4, 2)], 3),
            ('Data Diagnostics and Outcomes', outcome_panels, 3 if has_elo_data else 2),
        ]
        fig, axes = create_dashboard(sections, width=19.0, panel_height=3.5)

        val_epochs = self.val_iterations if self.val_iterations else []

        colors = {
            'train': '#2563EB',
            'val': '#DC2626',
            'policy': '#0F766E',
            'value': '#7C3AED',
            'gap': '#EA580C',
            'lr': '#475569',
            'elo': '#16A34A',
            'muted': '#64748B',
        }

        def _style_axis(ax, title, ylabel=None, percent=False):
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
            ax.set_xlabel('Epoch')
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_alpha(0.18)
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        def _set_percent_ylim(ax, values, *, min_pad=0.02, include=None):
            plot_values = [float(v) for v in values if v is not None]
            if include:
                plot_values.extend(float(v) for v in include if v is not None)
            if not plot_values:
                return
            y_min = min(plot_values)
            y_max = max(plot_values)
            pad = max(float(min_pad), (y_max - y_min) * 0.18)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

        def _smooth_values(ys):
            if not self.plot_smoothing_enabled or len(ys) < self.plot_smoothing_min_points:
                return ys
            smoothed = []
            prev = None
            alpha = float(self.plot_smoothing_alpha)
            for value in ys:
                try:
                    current = float(value)
                except (TypeError, ValueError):
                    smoothed.append(value)
                    continue
                if prev is None:
                    prev = current
                else:
                    prev = alpha * current + (1.0 - alpha) * prev
                smoothed.append(prev)
            return smoothed

        def _plot_line(ax, xs, ys, label, color, style='-', marker=None, linewidth=2.0, alpha=0.95, smooth=None):
            if not xs or not ys:
                return
            use_smooth = self.plot_smoothing_enabled if smooth is None else bool(smooth)
            ys_to_plot = _smooth_values(list(ys)) if use_smooth else list(ys)
            marker_to_plot = None if use_smooth and self.plot_smoothing_enabled else marker
            ax.plot(
                xs,
                ys_to_plot,
                linestyle=style,
                marker=marker_to_plot,
                color=color,
                label=label,
                linewidth=linewidth,
                markersize=4 if marker_to_plot else 0,
                alpha=alpha,
            )

        def _epoch_map(series):
            return {int(ep): value for ep, value in zip(self.iterations, series)}

        def _val_gap(train_series, val_series):
            train_by_epoch = _epoch_map(train_series)
            xs, ys = [], []
            for ep, val in zip(val_epochs, val_series):
                try:
                    ep_i = int(ep)
                    train_val = train_by_epoch[ep_i]
                    xs.append(ep)
                    ys.append(float(val) - float(train_val))
                except (KeyError, TypeError, ValueError):
                    continue
            return xs, ys

        def _relative_val_gap(train_series, val_series):
            train_by_epoch = _epoch_map(train_series)
            xs, ys = [], []
            for ep, val in zip(val_epochs, val_series):
                try:
                    train_val = float(train_by_epoch[int(ep)])
                    val_value = float(val)
                    if abs(train_val) <= 1e-12:
                        continue
                    xs.append(ep)
                    ys.append((val_value - train_val) / abs(train_val))
                except (KeyError, TypeError, ValueError):
                    continue
            return xs, ys

        def _best_epoch_and_value(xs, ys, mode='min'):
            if not xs or not ys:
                return None, None
            pairs = list(zip(xs, ys))
            if mode == 'max':
                return max(pairs, key=lambda item: item[1])
            return min(pairs, key=lambda item: item[1])

        def _mark_best(ax, xs, ys, mode='min', label='best'):
            ep, val = _best_epoch_and_value(xs, ys, mode=mode)
            if ep is None:
                return
            ax.scatter([ep], [val], s=42, color='#111827', zorder=5)
            try:
                place_left = float(ep) >= max(float(x) for x in xs) - 0.1
            except (TypeError, ValueError):
                place_left = False
            ax.annotate(
                f"{label}: {val:.4f}" if abs(float(val)) < 10 else f"{label}: {val:.0f}",
                xy=(ep, val),
                xytext=(-8, 7) if place_left else (7, 7),
                textcoords='offset points',
                fontsize=8,
                color='#111827',
                ha='right' if place_left else 'left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CBD5E1', alpha=0.9),
            )

        def _show_no_data(ax, message='Available in next run'):
            ax.text(
                0.5,
                0.5,
                message,
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=10,
                color=colors['muted'],
                bbox=dict(boxstyle='round,pad=0.35', fc='#F8FAFC', ec='#CBD5E1', alpha=0.95),
            )

        def _draw_resume_markers(ax, annotate=False):
            if not self.resume_markers:
                return
            for marker in self.resume_markers:
                try:
                    x_value = float(marker.get('x'))
                except (TypeError, ValueError):
                    continue
                ax.axvline(
                    x_value,
                    color='#111827',
                    linestyle='--',
                    linewidth=1.1,
                    alpha=0.42,
                    zorder=1,
                )
                if annotate:
                    ax.text(
                        x_value,
                        0.98,
                        str(marker.get('label') or 'resume'),
                        transform=ax.get_xaxis_transform(),
                        ha='left',
                        va='top',
                        rotation=90,
                        fontsize=7.5,
                        color='#111827',
                        bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#CBD5E1', alpha=0.85),
                    )

        def _latest_finite(series):
            for value in reversed(list(series or [])):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value == value:
                    return value
            return None

        def _legend_below(ax, *extra_axes, ncol=3):
            handles, labels = [], []
            for legend_ax in (ax,) + extra_axes:
                current_handles, current_labels = legend_ax.get_legend_handles_labels()
                handles.extend(current_handles)
                labels.extend(current_labels)
                existing = legend_ax.get_legend()
                if existing is not None:
                    existing.remove()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=max(1, min(int(ncol), len(labels))),
                    fontsize=7.5,
                    frameon=False,
                )

        # The dashboard helper groups these panels by their learning topic;
        # the training summary stays last as the closing read.
        # Row 1: losses
        ax = axes[0, 0]
        _plot_line(ax, self.iterations, self.train_losses, 'Train', colors['train'])
        _plot_line(ax, val_epochs, self.val_losses, 'Val', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_losses, mode='min', label='best val')
        _style_axis(ax, 'Total Loss + Learning Rate', 'Loss')
        if self.iterations:
            ax_lr = ax.twinx()
            lr_values = []
            try:
                for row in _read_csv_dict_rows(self.csv_path):
                    try:
                        if str(row.get('epoch', '')).strip().upper() == 'SWA':
                            continue
                        lr_values.append(float(row.get('learning_rate', '')))
                    except (TypeError, ValueError):
                        lr_values.append(None)
            except Exception:
                lr_values = []
            if lr_values and len(lr_values) == len(self.iterations):
                xs = [x for x, lr in zip(self.iterations, lr_values) if lr is not None]
                ys = [lr for lr in lr_values if lr is not None]
                if xs:
                    ax_lr.plot(xs, ys, color=colors['lr'], linestyle=':', linewidth=1.8, label='LR')
                    ax_lr.set_ylabel('LR')
                    ax_lr.tick_params(axis='y', labelcolor=colors['lr'])
                    ax_lr.spines['right'].set_alpha(0.18)
        _apply_sorted_legend(ax, fontsize=8, loc='best')

        ax = axes[0, 1]
        _plot_line(ax, self.iterations, self.train_policy_losses, 'Train Policy', colors['train'])
        _plot_line(ax, val_epochs, self.val_policy_losses, 'Val Policy', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_policy_losses, mode='min', label='best')
        _style_axis(ax, 'Policy Loss', 'Loss')
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[0, 2]
        _plot_line(ax, self.iterations, self.train_value_losses, 'Train Value', colors['train'])
        _plot_line(ax, val_epochs, self.val_value_losses, 'Val Value', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_value_losses, mode='min', label='best')
        _style_axis(ax, 'Value Loss', 'Loss')
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[0, 3]
        _style_axis(ax, 'Relative Loss Generalization Gap', 'Val − train / |train|', percent=True)
        loss_gap_series = [
            ('Total', self.train_losses, self.val_losses, '#111827'),
            ('Policy', self.train_policy_losses, self.val_policy_losses, colors['policy']),
            ('Value', self.train_value_losses, self.val_value_losses, colors['value']),
            ('Moves-left', self.train_moves_left_losses, self.val_moves_left_losses, colors['gap']),
        ]
        has_loss_gap = False
        for label, train_series, val_series, color in loss_gap_series:
            gap_x, gap_y = _relative_val_gap(train_series, val_series)
            if gap_x:
                has_loss_gap = True
                _plot_line(ax, gap_x, gap_y, label, color, marker='o', smooth=False)
        if has_loss_gap:
            ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Requires matching train and validation epochs')

        # Row 2: policy quality
        ax = axes[1, 0]
        _plot_line(ax, self.iterations, self.train_policy_top1, 'Train Top-1', colors['train'])
        _plot_line(ax, self.iterations, self.train_policy_top3, 'Train Top-3', colors['train'], style='--', alpha=0.65)
        _plot_line(ax, val_epochs, self.val_policy_top1, 'Val Top-1', colors['val'], marker='o')
        _plot_line(ax, val_epochs, self.val_policy_top3, 'Val Top-3', colors['val'], style='--', marker='o', alpha=0.75)
        _mark_best(ax, val_epochs, self.val_policy_top1, mode='max', label='best top1')
        _style_axis(ax, 'Policy Top-k Accuracy', 'Accuracy', percent=True)
        _set_percent_ylim(
            ax,
            list(self.train_policy_top1)
            + list(self.train_policy_top3)
            + list(self.val_policy_top1)
            + list(self.val_policy_top3),
            min_pad=0.03,
        )
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[1, 1]
        _style_axis(ax, 'Policy Sharpness vs Target', 'Effective legal moves')
        policy_sharpness_values = (
            list(self.train_policy_effective_moves)
            + list(self.val_policy_effective_moves)
            + list(self.train_policy_legal_effective_moves)
            + list(self.val_policy_legal_effective_moves)
            + list(self.train_policy_target_effective_moves)
            + list(self.val_policy_target_effective_moves)
        )
        policy_sharpness_values = [float(v) for v in policy_sharpness_values if v is not None and float(v or 0.0) > 0.0]
        if policy_sharpness_values:
            _plot_line(ax, self.iterations, self.train_policy_effective_moves, 'Train model', colors['train'])
            _plot_line(ax, val_epochs, self.val_policy_effective_moves, 'Val model', colors['val'], marker='o')
            if any(float(v or 0.0) > 0.0 for v in self.train_policy_legal_effective_moves):
                _plot_line(ax, self.iterations, self.train_policy_legal_effective_moves, 'Train legal', '#0F766E', style=':', alpha=0.8)
            if any(float(v or 0.0) > 0.0 for v in self.val_policy_legal_effective_moves):
                _plot_line(ax, val_epochs, self.val_policy_legal_effective_moves, 'Val legal', '#B45309', style=':', marker='o', alpha=0.8)
            _plot_line(ax, self.iterations, self.train_policy_target_effective_moves, 'Train target', '#0891B2', style='--', alpha=0.8)
            _plot_line(ax, val_epochs, self.val_policy_target_effective_moves, 'Val target', '#F59E0B', style='--', marker='o', alpha=0.8)
            y_min = min(policy_sharpness_values)
            y_max = max(policy_sharpness_values)
            pad = max(0.5, (y_max - y_min) * 0.18)
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Available in next run')

        ax = axes[1, 2]
        _plot_line(ax, self.iterations, self.train_policy_target_mass_top1, 'Train mass@1', colors['train'], alpha=0.85)
        _plot_line(ax, self.iterations, self.train_policy_target_mass_top3, 'Train mass@3', colors['train'], style='--', alpha=0.72)
        _plot_line(ax, self.iterations, self.train_policy_target_mass_top5, 'Train mass@5', colors['train'], style=':', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_policy_target_mass_top1, 'Val mass@1', colors['val'], marker='o', alpha=0.85)
        _plot_line(ax, val_epochs, self.val_policy_target_mass_top3, 'Val mass@3', colors['val'], style='--', marker='o', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_policy_target_mass_top5, 'Val mass@5', colors['val'], style=':', marker='o', alpha=0.72)
        if any(float(v or 0.0) > 0.0 for v in self.train_policy_target_support_top1_prob):
            _plot_line(ax, self.iterations, self.train_policy_target_support_top1_prob, 'Train support top1', '#0F766E', style='-.', alpha=0.82)
        if any(float(v or 0.0) > 0.0 for v in self.val_policy_target_support_top1_prob):
            _plot_line(ax, val_epochs, self.val_policy_target_support_top1_prob, 'Val support top1', '#B45309', style='-.', marker='o', alpha=0.82)
        _style_axis(ax, 'Soft Target Coverage + Support Confidence', 'Probability / mass', percent=True)
        _set_percent_ylim(
            ax,
            list(self.train_policy_target_mass_top1)
            + list(self.train_policy_target_mass_top3)
            + list(self.train_policy_target_mass_top5)
            + list(self.val_policy_target_mass_top1)
            + list(self.val_policy_target_mass_top3)
            + list(self.val_policy_target_mass_top5)
            + list(self.train_policy_target_support_top1_prob)
            + list(self.val_policy_target_support_top1_prob),
            min_pad=0.03,
        )
        policy_mass_values = (
            list(self.train_policy_target_mass_top1)
            + list(self.train_policy_target_mass_top3)
            + list(self.train_policy_target_mass_top5)
            + list(self.val_policy_target_mass_top1)
            + list(self.val_policy_target_mass_top3)
            + list(self.val_policy_target_mass_top5)
            + list(self.train_policy_target_support_top1_prob)
            + list(self.val_policy_target_support_top1_prob)
        )
        if any(float(v or 0.0) > 0.0 for v in policy_mass_values):
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Needs new CSV columns')

        ax = axes[1, 3]
        _style_axis(ax, 'Policy Top-k Generalization Gap', 'Val − train (percentage points)')
        top1_gap_x, top1_gap = _val_gap(self.train_policy_top1, self.val_policy_top1)
        top3_gap_x, top3_gap = _val_gap(self.train_policy_top3, self.val_policy_top3)
        if top1_gap or top3_gap:
            _plot_line(ax, top1_gap_x, top1_gap, 'Top-1', colors['policy'], marker='o', smooth=False)
            _plot_line(ax, top3_gap_x, top3_gap, 'Top-3', colors['elo'], marker='o', smooth=False)
            ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Requires matching train and validation epochs')

        ax = axes[1, 4]
        _style_axis(ax, 'Top-3 Accuracy Headroom', 'Top-3 − top-1 (percentage points)', percent=True)
        train_headroom = [
            float(top3) - float(top1)
            for top1, top3 in zip(self.train_policy_top1, self.train_policy_top3)
        ]
        val_headroom = [
            float(top3) - float(top1)
            for top1, top3 in zip(self.val_policy_top1, self.val_policy_top3)
        ]
        if train_headroom or val_headroom:
            _plot_line(ax, self.iterations[:len(train_headroom)], train_headroom, 'Train', colors['train'], smooth=False)
            _plot_line(ax, val_epochs[:len(val_headroom)], val_headroom, 'Val', colors['val'], marker='o', smooth=False)
            ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Requires top-1 and top-3 policy accuracy')

        ax = axes[1, 5]
        _style_axis(ax, 'Legal Policy Confidence and Margin', 'Probability gap', percent=True)
        confidence_values = (
            list(self.train_policy_legal_top1_prob)
            + list(self.val_policy_legal_top1_prob)
            + list(self.train_policy_legal_top1_margin)
            + list(self.val_policy_legal_top1_margin)
        )
        if any(value is not None and float(value or 0.0) > 0.0 for value in confidence_values):
            _plot_line(ax, self.iterations, self.train_policy_legal_top1_prob, 'Train top-1 prob', colors['train'])
            _plot_line(ax, val_epochs, self.val_policy_legal_top1_prob, 'Val top-1 prob', colors['val'], marker='o')
            _plot_line(ax, self.iterations, self.train_policy_legal_top1_margin, 'Train top-1 margin', colors['policy'], style='--')
            _plot_line(ax, val_epochs, self.val_policy_legal_top1_margin, 'Val top-1 margin', colors['elo'], style='--', marker='o')
            ax.set_ylim(bottom=0.0, top=1.0)
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'Legal confidence telemetry unavailable')

        # Row 3: value
        ax = axes[2, 0]
        _plot_line(ax, self.iterations, self.train_value_mae, 'Train MAE', colors['train'])
        _plot_line(ax, val_epochs, self.val_value_mae, 'Val MAE', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_value_mae, mode='min', label='best mae')
        _style_axis(ax, 'Value Scalar MAE', 'MAE')
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[2, 1]
        _style_axis(ax, 'Value MAE by Phase', 'MAE')
        value_phase_mae_values = (
            list(self.train_value_mae_opening) + list(self.train_value_mae_middlegame) + list(self.train_value_mae_endgame)
            + list(self.val_value_mae_opening) + list(self.val_value_mae_middlegame) + list(self.val_value_mae_endgame)
        )
        if any(float(v or 0.0) > 0.0 for v in value_phase_mae_values):
            _plot_line(ax, self.iterations, self.train_value_mae_opening, 'Train Opening', colors['train'], style='--', alpha=0.72)
            _plot_line(ax, self.iterations, self.train_value_mae_middlegame, 'Train Middlegame', colors['train'], alpha=0.95)
            _plot_line(ax, self.iterations, self.train_value_mae_endgame, 'Train Endgame', colors['train'], style=':', alpha=0.72)
            _plot_line(ax, val_epochs, self.val_value_mae_opening, 'Val Opening', colors['val'], style='--', marker='o', alpha=0.72)
            _plot_line(ax, val_epochs, self.val_value_mae_middlegame, 'Val Middlegame', colors['val'], marker='o', alpha=0.95)
            _plot_line(ax, val_epochs, self.val_value_mae_endgame, 'Val Endgame', colors['val'], style=':', marker='o', alpha=0.72)
            _apply_sorted_legend(ax, fontsize=7, loc='best')
        else:
            _show_no_data(ax, 'Available in next run')

        ax = axes[2, 2]
        _plot_line(ax, self.iterations, self.train_value_wdl_ce, 'Train WDL CE', colors['train'], smooth=False)
        _plot_line(ax, val_epochs, self.val_value_wdl_ce, 'Val WDL CE', colors['val'], marker='o', smooth=False)
        _mark_best(ax, val_epochs, self.val_value_wdl_ce, mode='min', label='best')
        _style_axis(ax, 'Value WDL Cross-Entropy', 'CE')
        if self.train_value_wdl_ce or self.val_value_wdl_ce:
            _apply_sorted_legend(ax, fontsize=8)

        ax = axes[2, 3]
        _plot_line(ax, self.iterations, self.train_value_wdl_acc, 'Train WDL accuracy', colors['train'])
        _plot_line(ax, val_epochs, self.val_value_wdl_acc, 'Val WDL accuracy', colors['val'], marker='o')
        _style_axis(ax, 'Value WDL Accuracy', 'Accuracy', percent=True)
        if any(float(value or 0.0) > 0.0 for value in self.train_value_wdl_acc + self.val_value_wdl_acc):
            ax.set_ylim(0.0, 1.0)
            _mark_best(ax, val_epochs, self.val_value_wdl_acc, mode='max', label='best')
            _apply_sorted_legend(ax, fontsize=8)
        else:
            _show_no_data(ax, 'WDL accuracy telemetry unavailable')

        # Row 4: value diagnostics and moves-left
        ax = axes[3, 0]
        _plot_line(ax, self.iterations, self.train_value_std_ratio_opening, 'Train Opening', colors['train'], style='--', alpha=0.72)
        _plot_line(ax, self.iterations, self.train_value_std_ratio_middlegame, 'Train Middlegame', colors['train'], alpha=0.95)
        _plot_line(ax, self.iterations, self.train_value_std_ratio_endgame, 'Train Endgame', colors['train'], style=':', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_value_std_ratio_opening, 'Val Opening', colors['val'], style='--', marker='o', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_value_std_ratio_middlegame, 'Val Middlegame', colors['val'], marker='o', alpha=0.95)
        _plot_line(ax, val_epochs, self.val_value_std_ratio_endgame, 'Val Endgame', colors['val'], style=':', marker='o', alpha=0.72)
        ax.axhline(1.0, color=colors['muted'], linestyle=':', linewidth=1.2)
        _style_axis(ax, 'Value Std Ratio by Phase', 'Pred std / target std')
        if (
            self.train_value_std_ratio_opening or self.train_value_std_ratio_middlegame
            or self.train_value_std_ratio_endgame or self.val_value_std_ratio_opening
            or self.val_value_std_ratio_middlegame or self.val_value_std_ratio_endgame
        ):
            _apply_sorted_legend(ax, fontsize=7, loc='best')
        else:
            _show_no_data(ax, 'Needs new CSV columns')

        ax = axes[3, 1]
        _plot_line(ax, self.iterations, self.train_value_wdl_ce_opening, 'Train Opening', colors['train'], style='--', alpha=0.75)
        _plot_line(ax, self.iterations, self.train_value_wdl_ce_middlegame, 'Train Middlegame', colors['train'], alpha=0.95)
        _plot_line(ax, self.iterations, self.train_value_wdl_ce_endgame, 'Train Endgame', colors['train'], style=':', alpha=0.75)
        _plot_line(ax, val_epochs, self.val_value_wdl_ce_opening, 'Val Opening', colors['val'], style='--', marker='o', alpha=0.75)
        _plot_line(ax, val_epochs, self.val_value_wdl_ce_middlegame, 'Val Middlegame', colors['val'], marker='o', alpha=0.95)
        _plot_line(ax, val_epochs, self.val_value_wdl_ce_endgame, 'Val Endgame', colors['val'], style=':', marker='o', alpha=0.75)
        _style_axis(ax, 'Value WDL CE by Phase', 'CE')
        if (
            self.train_value_wdl_ce_opening or self.train_value_wdl_ce_middlegame
            or self.train_value_wdl_ce_endgame or self.val_value_wdl_ce_opening
            or self.val_value_wdl_ce_middlegame or self.val_value_wdl_ce_endgame
        ):
            _apply_sorted_legend(ax, fontsize=7, loc='best')

        ax = axes[3, 2]
        _plot_line(ax, self.iterations, self.train_moves_left_losses, 'Train Global', colors['train'], linewidth=2.2)
        _plot_line(ax, self.iterations, self.train_moves_left_loss_opening, 'Train Opening', colors['train'], style='--', alpha=0.72)
        _plot_line(ax, self.iterations, self.train_moves_left_loss_middlegame, 'Train Middlegame', colors['train'], style='-.', alpha=0.72)
        _plot_line(ax, self.iterations, self.train_moves_left_loss_endgame, 'Train Endgame', colors['train'], style=':', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_moves_left_losses, 'Val Global', colors['val'], marker='o', linewidth=2.2)
        _plot_line(ax, val_epochs, self.val_moves_left_loss_opening, 'Val Opening', colors['val'], style='--', marker='o', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_moves_left_loss_middlegame, 'Val Middlegame', colors['val'], style='-.', marker='o', alpha=0.72)
        _plot_line(ax, val_epochs, self.val_moves_left_loss_endgame, 'Val Endgame', colors['val'], style=':', marker='o', alpha=0.72)
        _style_axis(ax, 'Moves-Left SmoothL1 Loss by Phase', 'Loss')
        if self.train_moves_left_losses or self.val_moves_left_losses:
            _apply_sorted_legend(ax, fontsize=7, loc='best')
        else:
            _show_no_data(ax, 'Needs new CSV columns')

        # Row 5: optimizer diagnostics.  The shared-tower probe is sampled once
        # per epoch; clipping statistics cover every optimizer batch.
        self._ensure_il_gradient_storage()

        ax = axes[4, 0]
        _style_axis(ax, 'Policy vs Value + Moves-Left Gradient Probe', 'Probe grad norm (log)')
        policy_probe = self.train_grad_policy_probe_norm
        value_probe = self.train_grad_value_probe_norm
        cosine_values = self.train_grad_policy_value_cosine
        has_task_gradient = any(float(v or 0.0) > 0.0 for v in policy_probe + value_probe)
        if has_task_gradient:
            policy_plot = [
                float(v) if v is not None and float(v) > 0.0 else float('nan')
                for v in policy_probe
            ]
            value_plot = [
                float(v) if v is not None and float(v) > 0.0 else float('nan')
                for v in value_probe
            ]
            cosine_plot = [
                float(v) if v is not None else float('nan')
                for v in cosine_values
            ]
            _plot_line(ax, self.iterations, policy_plot, 'Policy objective', colors['train'], marker='o', smooth=False)
            _plot_line(ax, self.iterations, value_plot, 'Value + MLH objective', colors['value'], marker='o', smooth=False)
            ax.set_yscale('log')
            cosine_ax = ax.twinx()
            _plot_line(cosine_ax, self.iterations, cosine_plot, 'Policy/value cosine', colors['gap'], style='--', marker='o', smooth=False)
            cosine_ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.0)
            cosine_ax.set_ylim(-1.05, 1.05)
            cosine_ax.tick_params(axis='y', labelcolor=colors['gap'])
            cosine_ax.spines['right'].set_alpha(0.18)
            ax.text(
                0.02, 0.96, 'cosine: + aligned, - conflicting; value objective includes moves-left loss',
                transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color='#475569',
            )
            _legend_below(ax, cosine_ax, ncol=3)
        else:
            _show_no_data(ax, 'No valid shared-norm probe samples recorded')

        ax = axes[4, 1]
        _style_axis(ax, 'Gradient Norms by Parameter Family', 'Gradient norm (log)')
        family_values = (
            self.train_grad_backbone_norm
            + self.train_grad_policy_head_norm
            + self.train_grad_value_head_norm
        )
        if any(float(v or 0.0) > 0.0 for v in family_values):
            _plot_line(ax, self.iterations, self.train_grad_backbone_norm, 'Shared tower', colors['train'], smooth=False)
            _plot_line(ax, self.iterations, self.train_grad_policy_head_norm, 'Policy head', colors['policy'], smooth=False)
            _plot_line(ax, self.iterations, self.train_grad_value_head_norm, 'Value heads', colors['value'], smooth=False)
            ax.set_yscale('log')
            _legend_below(ax, ncol=3)
        else:
            _show_no_data(ax, 'Gradient telemetry is available in the next epoch')

        ax = axes[4, 2]
        _style_axis(ax, 'Optimizer Step and Clipping', 'Total gradient norm (log)')
        if any(float(v or 0.0) > 0.0 for v in self.train_grad_total_norm):
            _plot_line(ax, self.iterations, self.train_grad_total_norm, 'Total before clip', '#111827', smooth=False)
            ax.set_yscale('log')
            clip_ax = ax.twinx()
            _plot_line(clip_ax, self.iterations, self.train_grad_clip_fraction, 'Batches clipped', colors['val'], style='--', smooth=False)
            _plot_line(clip_ax, self.iterations, self.train_grad_clip_scale_mean, 'Mean applied scale', colors['elo'], style=':', smooth=False)
            clip_ax.set_ylim(0.0, 1.05)
            clip_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            clip_ax.set_ylabel('Fraction / scale')
            clip_ax.spines['right'].set_alpha(0.18)
            _legend_below(ax, clip_ax, ncol=3)
        else:
            _show_no_data(ax, 'Gradient telemetry is available in the next epoch')

        # Row 6: data diagnostics, Elo, and final summary
        ax = axes[5, 0]
        _plot_line(ax, self.iterations, self.train_soft_occurrence_avg, 'Train occurrence', colors['train'])
        _plot_line(ax, val_epochs, self.val_soft_occurrence_avg, 'Val occurrence', colors['val'], marker='o')
        _plot_line(ax, self.iterations, self.train_soft_occurrence_max, 'Train max occurrence', colors['train'], style=':', alpha=0.7)
        _plot_line(ax, val_epochs, self.val_soft_occurrence_max, 'Val max occurrence', colors['val'], style=':', marker='o', alpha=0.7)
        if any(float(v or 0.0) > 100.0 for v in list(self.train_soft_occurrence_max) + list(self.val_soft_occurrence_max)):
            ax.set_yscale('log')
        _style_axis(ax, 'Soft Target Compression', 'Occurrence count')
        ax2 = ax.twinx()
        _plot_line(ax2, self.iterations, self.train_soft_sample_weight_avg, 'Train weight', '#0891B2', style='--', alpha=0.75)
        _plot_line(ax2, val_epochs, self.val_soft_sample_weight_avg, 'Val weight', '#F59E0B', style='--', marker='o', alpha=0.75)
        ax2.set_ylabel('Avg sample weight')
        ax2.tick_params(axis='y', labelcolor='#0891B2')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')
        train_occ_latest = _latest_finite(self.train_soft_occurrence_avg)
        val_occ_latest = _latest_finite(self.val_soft_occurrence_avg)
        train_weight_latest = _latest_finite(self.train_soft_sample_weight_avg)
        val_weight_latest = _latest_finite(self.val_soft_sample_weight_avg)
        train_occ_max_latest = _latest_finite(self.train_soft_occurrence_max)
        val_occ_max_latest = _latest_finite(self.val_soft_occurrence_max)
        train_target_eff_latest = _latest_finite(self.train_policy_target_effective_moves)
        val_target_eff_latest = _latest_finite(self.val_policy_target_effective_moves)
        train_target_top1_latest = _latest_finite(self.train_policy_target_mass_top1)
        val_target_top1_latest = _latest_finite(self.val_policy_target_mass_top1)
        pool_train_occ = None
        pool_val_occ = None
        pool_train_target_eff = None
        pool_val_target_eff = None
        pool_train_soft_rate = None
        pool_val_soft_rate = None
        soft_stats = (self.run_summary_metadata or {}).get('soft_stats', {}) or {}
        train_soft_stats = soft_stats.get('train') or {}
        val_soft_stats = soft_stats.get('val') or {}
        train_epoch_positions = (self.run_summary_metadata or {}).get('train_epoch_positions')
        total_epoch_positions = (self.run_summary_metadata or {}).get('total_epoch_positions')
        total_pool_positions = (self.run_summary_metadata or {}).get('total_pool_positions')
        if train_soft_stats.get('policy_occ_avg') is not None:
            pool_train_occ = float(train_soft_stats.get('policy_occ_avg') or 0.0)
        if val_soft_stats.get('policy_occ_avg') is not None:
            pool_val_occ = float(val_soft_stats.get('policy_occ_avg') or 0.0)
        if train_soft_stats.get('policy_target_effective_moves') is not None:
            pool_train_target_eff = float(train_soft_stats.get('policy_target_effective_moves') or 1.0)
        if val_soft_stats.get('policy_target_effective_moves') is not None:
            pool_val_target_eff = float(val_soft_stats.get('policy_target_effective_moves') or 1.0)
        if train_soft_stats.get('policy_soft_rate') is not None:
            pool_train_soft_rate = float(train_soft_stats.get('policy_soft_rate') or 0.0)
        if val_soft_stats.get('policy_soft_rate') is not None:
            pool_val_soft_rate = float(val_soft_stats.get('policy_soft_rate') or 0.0)
        if train_occ_latest is not None or val_occ_latest is not None:
            info_lines = [
                "sampled avg_occ: "
                f"T={train_occ_latest:.2f}" if train_occ_latest is not None else "sampled avg_occ: T=-",
                f"V={val_occ_latest:.2f}" if val_occ_latest is not None else "V=-",
            ]
            weight_line = (
                "sampled weight: "
                f"T={train_weight_latest:.2f}" if train_weight_latest is not None else "sampled weight: T=-"
            )
            weight_line += f", V={val_weight_latest:.2f}" if val_weight_latest is not None else ", V=-"
            max_line = (
                "sampled max_occ: "
                f"T={train_occ_max_latest:.0f}" if train_occ_max_latest is not None else "sampled max_occ: T=-"
            )
            max_line += f", V={val_occ_max_latest:.0f}" if val_occ_max_latest is not None else ", V=-"
            eff_line = (
                "target_eff_moves: "
                f"T={train_target_eff_latest:.2f}" if train_target_eff_latest is not None else "target_eff_moves: T=-"
            )
            eff_line += f", V={val_target_eff_latest:.2f}" if val_target_eff_latest is not None else ", V=-"
            top1_line = (
                "target_top1_mass: "
                f"T={train_target_top1_latest:.3f}" if train_target_top1_latest is not None else "target_top1_mass: T=-"
            )
            top1_line += f", V={val_target_top1_latest:.3f}" if val_target_top1_latest is not None else ", V=-"
            pool_lines = []
            if pool_train_occ is not None or pool_val_occ is not None:
                pool_line = (
                    "pool avg_occ: "
                    f"T={pool_train_occ:.2f}" if pool_train_occ is not None else "pool avg_occ: T=-"
                )
                pool_line += f", V={pool_val_occ:.2f}" if pool_val_occ is not None else ", V=-"
                pool_lines.append(pool_line)
            if pool_train_soft_rate is not None or pool_val_soft_rate is not None:
                pool_line = (
                    "pool soft_rows: "
                    f"T={100.0 * pool_train_soft_rate:.1f}%" if pool_train_soft_rate is not None else "pool soft_rows: T=-"
                )
                pool_line += (
                    f", V={100.0 * pool_val_soft_rate:.1f}%" if pool_val_soft_rate is not None else ", V=-"
                )
                pool_lines.append(pool_line)
            if pool_train_target_eff is not None or pool_val_target_eff is not None:
                pool_line = (
                    "pool eff_moves: "
                    f"T={pool_train_target_eff:.2f}" if pool_train_target_eff is not None else "pool eff_moves: T=-"
                )
                pool_line += f", V={pool_val_target_eff:.2f}" if pool_val_target_eff is not None else ", V=-"
                pool_lines.append(pool_line)
            detail_lines = [
                (
                    "data: "
                    f"train/epoch={_format_million_count(train_epoch_positions)}, "
                    f"epoch_total={_format_million_count(total_epoch_positions)}, "
                    f"pool_total={_format_million_count(total_pool_positions)}"
                ),
                f"{info_lines[0]}, {info_lines[1]}",
                eff_line,
                top1_line,
                weight_line,
            ]
            detail_lines[1:1] = pool_lines[:2]
            ax.text(
                0.02,
                0.04,
                "\n".join(detail_lines),
                transform=ax.transAxes,
                ha='left',
                va='bottom',
                fontsize=8,
                color='#334155',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#CBD5E1', alpha=0.92),
            )

        if has_elo_data:
            ax = axes[5, 1]
            self._plot_il_elo_panel(ax)
            ax.set_facecolor('#FFFFFF')

        ax = axes[5, 2]
        self._plot_il_summary_panel(ax)

        for panel_key, marker_ax in axes.items():
            if panel_key == (5, 2):
                continue
            _draw_resume_markers(marker_ax, annotate=(panel_key == (0, 0)))

        finish_dashboard(
            fig,
            self.plot_path,
            'IL Training Progress',
            context_lines[:4],
        )

        print(f"Plot saved to: {self.plot_path}")

    def _log_il(self, iteration, train_losses=None, val_losses=None,
                train_metrics=None, val_metrics=None, lr=None, estimated_elo=None, **kwargs):
        row = [
            iteration,
            train_losses['total'],
            train_losses['policy'],
            train_losses['value'],
            train_losses.get('moves_left', ''),
            val_losses['total'] if val_losses else '',
            val_losses['policy'] if val_losses else '',
            val_losses['value'] if val_losses else '',
            val_losses.get('moves_left', '') if val_losses else '',
            lr if lr is not None else '',
            # NEW: Metrics
            train_metrics.get('policy_top1_acc', '') if train_metrics else '',
            train_metrics.get('policy_top3_acc', '') if train_metrics else '',
            train_metrics.get('policy_target_mass_top1', '') if train_metrics else '',
            train_metrics.get('policy_target_mass_top3', '') if train_metrics else '',
            train_metrics.get('policy_target_mass_top5', '') if train_metrics else '',
            train_metrics.get('policy_entropy', '') if train_metrics else '',
            train_metrics.get('policy_effective_moves', '') if train_metrics else '',
            train_metrics.get('policy_top1_prob', '') if train_metrics else '',
            train_metrics.get('policy_legal_entropy', '') if train_metrics else '',
            train_metrics.get('policy_legal_effective_moves', '') if train_metrics else '',
            train_metrics.get('policy_legal_top1_prob', '') if train_metrics else '',
            train_metrics.get('policy_legal_top1_margin', '') if train_metrics else '',
            train_metrics.get('policy_target_entropy', '') if train_metrics else '',
            train_metrics.get('policy_target_effective_moves', '') if train_metrics else '',
            train_metrics.get('policy_target_top1_mass', '') if train_metrics else '',
            train_metrics.get('policy_target_support_top1_prob', '') if train_metrics else '',
            train_metrics.get('policy_target_support_top1_margin', '') if train_metrics else '',
            train_metrics.get('value_mae', '') if train_metrics else '',
            train_metrics.get('value_mae_opening', '') if train_metrics else '',
            train_metrics.get('value_mae_middlegame', '') if train_metrics else '',
            train_metrics.get('value_mae_endgame', '') if train_metrics else '',
            train_metrics.get('value_wdl_acc', '') if train_metrics else '',
            train_metrics.get('value_wdl_ce', '') if train_metrics else '',
            train_metrics.get('value_wdl_ce_opening', '') if train_metrics else '',
            train_metrics.get('value_wdl_ce_middlegame', '') if train_metrics else '',
            train_metrics.get('value_wdl_ce_endgame', '') if train_metrics else '',
            train_metrics.get('value_std_ratio_opening', '') if train_metrics else '',
            train_metrics.get('value_std_ratio_middlegame', '') if train_metrics else '',
            train_metrics.get('value_std_ratio_endgame', '') if train_metrics else '',
            train_metrics.get('moves_left_loss_opening', '') if train_metrics else '',
            train_metrics.get('moves_left_loss_middlegame', '') if train_metrics else '',
            train_metrics.get('moves_left_loss_endgame', '') if train_metrics else '',
            train_metrics.get('moves_left_mae', '') if train_metrics else '',
            train_metrics.get('moves_left_mae_opening', '') if train_metrics else '',
            train_metrics.get('moves_left_mae_middlegame', '') if train_metrics else '',
            train_metrics.get('moves_left_mae_endgame', '') if train_metrics else '',
            val_metrics.get('policy_top1_acc', '') if val_metrics else '',
            val_metrics.get('policy_top3_acc', '') if val_metrics else '',
            val_metrics.get('policy_target_mass_top1', '') if val_metrics else '',
            val_metrics.get('policy_target_mass_top3', '') if val_metrics else '',
            val_metrics.get('policy_target_mass_top5', '') if val_metrics else '',
            val_metrics.get('policy_entropy', '') if val_metrics else '',
            val_metrics.get('policy_effective_moves', '') if val_metrics else '',
            val_metrics.get('policy_top1_prob', '') if val_metrics else '',
            val_metrics.get('policy_legal_entropy', '') if val_metrics else '',
            val_metrics.get('policy_legal_effective_moves', '') if val_metrics else '',
            val_metrics.get('policy_legal_top1_prob', '') if val_metrics else '',
            val_metrics.get('policy_legal_top1_margin', '') if val_metrics else '',
            val_metrics.get('policy_target_entropy', '') if val_metrics else '',
            val_metrics.get('policy_target_effective_moves', '') if val_metrics else '',
            val_metrics.get('policy_target_top1_mass', '') if val_metrics else '',
            val_metrics.get('policy_target_support_top1_prob', '') if val_metrics else '',
            val_metrics.get('policy_target_support_top1_margin', '') if val_metrics else '',
            val_metrics.get('value_mae', '') if val_metrics else '',
            val_metrics.get('value_mae_opening', '') if val_metrics else '',
            val_metrics.get('value_mae_middlegame', '') if val_metrics else '',
            val_metrics.get('value_mae_endgame', '') if val_metrics else '',
            val_metrics.get('value_wdl_acc', '') if val_metrics else '',
            val_metrics.get('value_wdl_ce', '') if val_metrics else '',
            val_metrics.get('value_wdl_ce_opening', '') if val_metrics else '',
            val_metrics.get('value_wdl_ce_middlegame', '') if val_metrics else '',
            val_metrics.get('value_wdl_ce_endgame', '') if val_metrics else '',
            val_metrics.get('value_std_ratio_opening', '') if val_metrics else '',
            val_metrics.get('value_std_ratio_middlegame', '') if val_metrics else '',
            val_metrics.get('value_std_ratio_endgame', '') if val_metrics else '',
            val_metrics.get('moves_left_loss_opening', '') if val_metrics else '',
            val_metrics.get('moves_left_loss_middlegame', '') if val_metrics else '',
            val_metrics.get('moves_left_loss_endgame', '') if val_metrics else '',
            val_metrics.get('moves_left_mae', '') if val_metrics else '',
            val_metrics.get('moves_left_mae_opening', '') if val_metrics else '',
            val_metrics.get('moves_left_mae_middlegame', '') if val_metrics else '',
            val_metrics.get('moves_left_mae_endgame', '') if val_metrics else '',
            train_metrics.get('soft_occurrence_avg', '') if train_metrics else '',
            train_metrics.get('soft_occurrence_max', '') if train_metrics else '',
            train_metrics.get('soft_sample_weight_avg', '') if train_metrics else '',
            train_metrics.get('soft_policy_mass_kept_avg', '') if train_metrics else '',
            train_metrics.get('soft_policy_mass_kept_min', '') if train_metrics else '',
            val_metrics.get('soft_occurrence_avg', '') if val_metrics else '',
            val_metrics.get('soft_occurrence_max', '') if val_metrics else '',
            val_metrics.get('soft_sample_weight_avg', '') if val_metrics else '',
            val_metrics.get('soft_policy_mass_kept_avg', '') if val_metrics else '',
            val_metrics.get('soft_policy_mass_kept_min', '') if val_metrics else '',
        ]

        # Elo estimation
        if estimated_elo is not None:
            try:
                row.append(int(round(float(estimated_elo))))
            except (TypeError, ValueError):
                row.append('')
                estimated_elo = None
        else:
            row.append('')
        row.extend([
            kwargs.get('estimated_elo_se', ''),
            kwargs.get('estimated_elo_ci95_low', ''),
            kwargs.get('estimated_elo_ci95_high', ''),
        ])

        row.extend([
            kwargs.get('estimated_elo_nn', ''),
            kwargs.get('estimated_elo_nn_se', ''),
            kwargs.get('estimated_elo_nn_ci95_low', ''),
            kwargs.get('estimated_elo_nn_ci95_high', ''),
            kwargs.get('estimated_elo_mcts', ''),
            kwargs.get('estimated_elo_mcts_se', ''),
            kwargs.get('estimated_elo_mcts_ci95_low', ''),
            kwargs.get('estimated_elo_mcts_ci95_high', ''),
            kwargs.get('estimated_elo_mcts_simulations', ''),
            kwargs.get('estimated_elo_mcts_by_simulations', ''),
            kwargs.get('estimated_elo_nn_label', ''),
            kwargs.get('estimated_elo_mcts_label', ''),
        ])

        def _metric_value(metrics, key):
            if not metrics:
                return None
            value = metrics.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return None if value != value else value

        def _loss_value(losses, key):
            if not losses:
                return None
            value = losses.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return None if value != value else value

        def _gap(val_value, train_value):
            if val_value is None or train_value is None:
                return ''
            return val_value - train_value

        current_val_loss = _loss_value(val_losses, 'total')
        current_val_top1 = _metric_value(val_metrics, 'policy_top1_acc')
        current_val_mae = _metric_value(val_metrics, 'value_mae')
        if current_val_loss is not None:
            best_val_loss_so_far = min(self.val_losses + [current_val_loss]) if self.val_losses else current_val_loss
        else:
            best_val_loss_so_far = min(self.val_losses) if self.val_losses else ''
        if current_val_top1 is not None:
            best_val_top1_so_far = max(self.val_policy_top1 + [current_val_top1]) if self.val_policy_top1 else current_val_top1
        else:
            best_val_top1_so_far = max(self.val_policy_top1) if self.val_policy_top1 else ''
        if current_val_mae is not None:
            best_val_mae_so_far = min(self.val_value_mae + [current_val_mae]) if self.val_value_mae else current_val_mae
        else:
            best_val_mae_so_far = min(self.val_value_mae) if self.val_value_mae else ''

        row.extend([
            _gap(_loss_value(val_losses, 'total'), _loss_value(train_losses, 'total')),
            _gap(_metric_value(val_metrics, 'policy_top1_acc'), _metric_value(train_metrics, 'policy_top1_acc')),
            _gap(_metric_value(val_metrics, 'policy_top3_acc'), _metric_value(train_metrics, 'policy_top3_acc')),
            _gap(_metric_value(val_metrics, 'value_mae'), _metric_value(train_metrics, 'value_mae')),
            _gap(_metric_value(val_metrics, 'value_wdl_acc'), _metric_value(train_metrics, 'value_wdl_acc')),
            best_val_loss_so_far,
            best_val_top1_so_far,
            best_val_mae_so_far,
        ])
        row.extend([
            train_metrics.get(key, '') if train_metrics else ''
            for key in IL_GRADIENT_METRIC_KEYS
        ])

        # Store for plotting
        self.iterations.append(iteration)
        self.train_losses.append(train_losses['total'])
        self.train_policy_losses.append(train_losses['policy'])
        self.train_value_losses.append(train_losses['value'])
        self.train_moves_left_losses.append(train_losses.get('moves_left', 0.0))
        self._ensure_il_gradient_storage()
        for key in IL_GRADIENT_METRIC_KEYS:
            getattr(self, f'train_{key}').append(
                train_metrics.get(key) if train_metrics else None
            )

        if train_metrics:
            self.train_policy_top1.append(train_metrics.get('policy_top1_acc', 0))
            self.train_policy_top3.append(train_metrics.get('policy_top3_acc', 0))
            self.train_policy_target_mass_top1.append(train_metrics.get('policy_target_mass_top1', 0))
            self.train_policy_target_mass_top3.append(train_metrics.get('policy_target_mass_top3', 0))
            self.train_policy_target_mass_top5.append(train_metrics.get('policy_target_mass_top5', 0))
            self.train_policy_entropy.append(train_metrics.get('policy_entropy', 0))
            self.train_policy_effective_moves.append(train_metrics.get('policy_effective_moves', 0))
            self.train_policy_top1_prob.append(train_metrics.get('policy_top1_prob', 0))
            self.train_policy_legal_entropy.append(train_metrics.get('policy_legal_entropy', 0))
            self.train_policy_legal_effective_moves.append(train_metrics.get('policy_legal_effective_moves', 0))
            self.train_policy_legal_top1_prob.append(train_metrics.get('policy_legal_top1_prob', 0))
            self.train_policy_legal_top1_margin.append(train_metrics.get('policy_legal_top1_margin', 0))
            self.train_policy_target_entropy.append(train_metrics.get('policy_target_entropy', 0))
            self.train_policy_target_effective_moves.append(train_metrics.get('policy_target_effective_moves', 0))
            self.train_policy_target_top1_mass.append(train_metrics.get('policy_target_top1_mass', 0))
            self.train_policy_target_support_top1_prob.append(train_metrics.get('policy_target_support_top1_prob', 0))
            self.train_policy_target_support_top1_margin.append(train_metrics.get('policy_target_support_top1_margin', 0))
            self.train_value_mae.append(train_metrics.get('value_mae', 0))
            self.train_value_mae_opening.append(train_metrics.get('value_mae_opening', 0))
            self.train_value_mae_middlegame.append(train_metrics.get('value_mae_middlegame', 0))
            self.train_value_mae_endgame.append(train_metrics.get('value_mae_endgame', 0))
            self.train_value_wdl_acc.append(train_metrics.get('value_wdl_acc', 0))
            self.train_value_wdl_ce.append(train_metrics.get('value_wdl_ce', 0))
            self.train_value_wdl_ce_opening.append(train_metrics.get('value_wdl_ce_opening', 0))
            self.train_value_wdl_ce_middlegame.append(train_metrics.get('value_wdl_ce_middlegame', 0))
            self.train_value_wdl_ce_endgame.append(train_metrics.get('value_wdl_ce_endgame', 0))
            self.train_value_std_ratio_opening.append(train_metrics.get('value_std_ratio_opening', 0))
            self.train_value_std_ratio_middlegame.append(train_metrics.get('value_std_ratio_middlegame', 0))
            self.train_value_std_ratio_endgame.append(train_metrics.get('value_std_ratio_endgame', 0))
            self.train_moves_left_loss_opening.append(train_metrics.get('moves_left_loss_opening', 0))
            self.train_moves_left_loss_middlegame.append(train_metrics.get('moves_left_loss_middlegame', 0))
            self.train_moves_left_loss_endgame.append(train_metrics.get('moves_left_loss_endgame', 0))
            self.train_moves_left_mae.append(train_metrics.get('moves_left_mae', 0))
            self.train_moves_left_mae_opening.append(train_metrics.get('moves_left_mae_opening', 0))
            self.train_moves_left_mae_middlegame.append(train_metrics.get('moves_left_mae_middlegame', 0))
            self.train_moves_left_mae_endgame.append(train_metrics.get('moves_left_mae_endgame', 0))
            self.train_soft_occurrence_avg.append(train_metrics.get('soft_occurrence_avg', 0))
            self.train_soft_occurrence_max.append(train_metrics.get('soft_occurrence_max', 0))
            self.train_soft_sample_weight_avg.append(train_metrics.get('soft_sample_weight_avg', 0))
            self.train_soft_policy_mass_kept_avg.append(train_metrics.get('soft_policy_mass_kept_avg', 0))
            self.train_soft_policy_mass_kept_min.append(train_metrics.get('soft_policy_mass_kept_min', 0))
        else:
            self.train_policy_top1.append(0.0)
            self.train_policy_top3.append(0.0)
            self.train_policy_target_mass_top1.append(0.0)
            self.train_policy_target_mass_top3.append(0.0)
            self.train_policy_target_mass_top5.append(0.0)
            self.train_policy_entropy.append(0.0)
            self.train_policy_effective_moves.append(0.0)
            self.train_policy_top1_prob.append(0.0)
            self.train_policy_legal_entropy.append(0.0)
            self.train_policy_legal_effective_moves.append(0.0)
            self.train_policy_legal_top1_prob.append(0.0)
            self.train_policy_legal_top1_margin.append(0.0)
            self.train_policy_target_entropy.append(0.0)
            self.train_policy_target_effective_moves.append(0.0)
            self.train_policy_target_top1_mass.append(0.0)
            self.train_policy_target_support_top1_prob.append(0.0)
            self.train_policy_target_support_top1_margin.append(0.0)
            self.train_value_mae.append(0.0)
            self.train_value_mae_opening.append(0.0)
            self.train_value_mae_middlegame.append(0.0)
            self.train_value_mae_endgame.append(0.0)
            self.train_value_wdl_acc.append(0.0)
            self.train_value_wdl_ce.append(0.0)
            self.train_value_wdl_ce_opening.append(0.0)
            self.train_value_wdl_ce_middlegame.append(0.0)
            self.train_value_wdl_ce_endgame.append(0.0)
            self.train_value_std_ratio_opening.append(0.0)
            self.train_value_std_ratio_middlegame.append(0.0)
            self.train_value_std_ratio_endgame.append(0.0)
            self.train_moves_left_loss_opening.append(0.0)
            self.train_moves_left_loss_middlegame.append(0.0)
            self.train_moves_left_loss_endgame.append(0.0)
            self.train_moves_left_mae.append(0.0)
            self.train_moves_left_mae_opening.append(0.0)
            self.train_moves_left_mae_middlegame.append(0.0)
            self.train_moves_left_mae_endgame.append(0.0)
            self.train_soft_occurrence_avg.append(0.0)
            self.train_soft_occurrence_max.append(0.0)
            self.train_soft_sample_weight_avg.append(0.0)
            self.train_soft_policy_mass_kept_avg.append(0.0)
            self.train_soft_policy_mass_kept_min.append(0.0)

        if val_losses is not None:
            self.val_iterations.append(iteration)
            self.val_losses.append(val_losses['total'])
            self.val_policy_losses.append(val_losses['policy'])
            self.val_value_losses.append(val_losses['value'])
            self.val_moves_left_losses.append(val_losses.get('moves_left', 0.0))

        if val_metrics:
            self.val_policy_top1.append(val_metrics.get('policy_top1_acc', 0))
            self.val_policy_top3.append(val_metrics.get('policy_top3_acc', 0))
            self.val_policy_target_mass_top1.append(val_metrics.get('policy_target_mass_top1', 0))
            self.val_policy_target_mass_top3.append(val_metrics.get('policy_target_mass_top3', 0))
            self.val_policy_target_mass_top5.append(val_metrics.get('policy_target_mass_top5', 0))
            self.val_policy_entropy.append(val_metrics.get('policy_entropy', 0))
            self.val_policy_effective_moves.append(val_metrics.get('policy_effective_moves', 0))
            self.val_policy_top1_prob.append(val_metrics.get('policy_top1_prob', 0))
            self.val_policy_legal_entropy.append(val_metrics.get('policy_legal_entropy', 0))
            self.val_policy_legal_effective_moves.append(val_metrics.get('policy_legal_effective_moves', 0))
            self.val_policy_legal_top1_prob.append(val_metrics.get('policy_legal_top1_prob', 0))
            self.val_policy_legal_top1_margin.append(val_metrics.get('policy_legal_top1_margin', 0))
            self.val_policy_target_entropy.append(val_metrics.get('policy_target_entropy', 0))
            self.val_policy_target_effective_moves.append(val_metrics.get('policy_target_effective_moves', 0))
            self.val_policy_target_top1_mass.append(val_metrics.get('policy_target_top1_mass', 0))
            self.val_policy_target_support_top1_prob.append(val_metrics.get('policy_target_support_top1_prob', 0))
            self.val_policy_target_support_top1_margin.append(val_metrics.get('policy_target_support_top1_margin', 0))
            self.val_value_mae.append(val_metrics.get('value_mae', 0))
            self.val_value_mae_opening.append(val_metrics.get('value_mae_opening', 0))
            self.val_value_mae_middlegame.append(val_metrics.get('value_mae_middlegame', 0))
            self.val_value_mae_endgame.append(val_metrics.get('value_mae_endgame', 0))
            self.val_value_wdl_acc.append(val_metrics.get('value_wdl_acc', 0))
            self.val_value_wdl_ce.append(val_metrics.get('value_wdl_ce', 0))
            self.val_value_wdl_ce_opening.append(val_metrics.get('value_wdl_ce_opening', 0))
            self.val_value_wdl_ce_middlegame.append(val_metrics.get('value_wdl_ce_middlegame', 0))
            self.val_value_wdl_ce_endgame.append(val_metrics.get('value_wdl_ce_endgame', 0))
            self.val_value_std_ratio_opening.append(val_metrics.get('value_std_ratio_opening', 0))
            self.val_value_std_ratio_middlegame.append(val_metrics.get('value_std_ratio_middlegame', 0))
            self.val_value_std_ratio_endgame.append(val_metrics.get('value_std_ratio_endgame', 0))
            self.val_moves_left_loss_opening.append(val_metrics.get('moves_left_loss_opening', 0))
            self.val_moves_left_loss_middlegame.append(val_metrics.get('moves_left_loss_middlegame', 0))
            self.val_moves_left_loss_endgame.append(val_metrics.get('moves_left_loss_endgame', 0))
            self.val_moves_left_mae.append(val_metrics.get('moves_left_mae', 0))
            self.val_moves_left_mae_opening.append(val_metrics.get('moves_left_mae_opening', 0))
            self.val_moves_left_mae_middlegame.append(val_metrics.get('moves_left_mae_middlegame', 0))
            self.val_moves_left_mae_endgame.append(val_metrics.get('moves_left_mae_endgame', 0))
            self.val_soft_occurrence_avg.append(val_metrics.get('soft_occurrence_avg', 0))
            self.val_soft_occurrence_max.append(val_metrics.get('soft_occurrence_max', 0))
            self.val_soft_sample_weight_avg.append(val_metrics.get('soft_sample_weight_avg', 0))
            self.val_soft_policy_mass_kept_avg.append(val_metrics.get('soft_policy_mass_kept_avg', 0))
            self.val_soft_policy_mass_kept_min.append(val_metrics.get('soft_policy_mass_kept_min', 0))

        # Elo estimation storage
        if estimated_elo is not None:
            # CSV already contains this value in the current row.
            self.record_estimated_elo(iteration, estimated_elo, update_csv=False)
        with open(self.csv_path, 'a', newline='') as handle:
            csv.writer(handle).writerow(row)
