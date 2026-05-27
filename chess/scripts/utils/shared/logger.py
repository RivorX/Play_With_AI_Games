"""
Unified training logger for both IL and RL training
"""

import csv
import json
import textwrap
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from datetime import datetime
from pathlib import Path


_CSV_CONFIG_METADATA_KEY = "# config_json"


def _json_safe_config(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_config_metadata_row(row):
    return bool(row) and str(row[0]).strip() == _CSV_CONFIG_METADATA_KEY


def _read_csv_rows_preserving_metadata(csv_path):
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.reader(f))
    metadata_rows = []
    if rows and _is_config_metadata_row(rows[0]):
        metadata_rows = [rows[0]]
        rows = rows[1:]
    return metadata_rows, rows


def _read_csv_dict_rows(csv_path):
    metadata_rows, rows = _read_csv_rows_preserving_metadata(csv_path)
    if not rows:
        return []
    header = list(rows[0])
    result = []
    for raw_row in rows[1:]:
        row = list(raw_row)
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        result.append(dict(zip(header, row[:len(header)])))
    return result


def _legend_display_y(handle):
    """Return the display-space y of a legend handle's latest finite point."""
    ax = getattr(handle, "axes", None)
    if ax is None:
        return None
    try:
        xdata = list(handle.get_xdata(orig=False))
        ydata = list(handle.get_ydata(orig=False))
    except Exception:
        return None
    if not xdata or not ydata:
        return None
    for x_value, y_value in reversed(list(zip(xdata, ydata))):
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if y_float != y_float or x_float != x_float:
            continue
        try:
            return float(ax.transData.transform((x_float, y_float))[1])
        except Exception:
            return None
    return None


def _sorted_legend_items(handles, labels):
    items = [
        (idx, handle, label, _legend_display_y(handle))
        for idx, (handle, label) in enumerate(zip(handles or [], labels or []))
        if label and not str(label).startswith("_")
    ]
    items.sort(key=lambda item: (item[3] is not None, item[3] if item[3] is not None else -item[0]), reverse=True)
    return [item[1] for item in items], [item[2] for item in items]


def _apply_sorted_legend(ax, handles=None, labels=None, **kwargs):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    handles, labels = _sorted_legend_items(handles, labels)
    if handles:
        return ax.legend(handles, labels, **kwargs)
    return None


class TrainingLogger:
    """
    Universal logger for training metrics
    Supports both IL and RL modes with detailed metrics
    """
    
    def __init__(self, log_dir, experiment_name="training", mode="il", config_snapshot=None):
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
                    'estimated_elo',
                    'train_val_loss_gap',
                    'policy_top1_gap',
                    'policy_top3_gap',
                    'value_mae_gap',
                    'value_wdl_acc_gap',
                    'best_val_loss_so_far',
                    'best_val_policy_top1_so_far',
                    'best_val_value_mae_so_far',
                ]
            
            else:  # RL mode
                header = [
                    'iteration', 'avg_loss', 'policy_loss', 'value_loss', 'learning_rate',
                    'value_loss_weight', 'mcts_q_value_scale', 'mcts_q_selection_weight',
                    'mcts_q_effective_weight', 'mcts_q_value_trust',
                    'value_guard_streak', 'mcts_no_mcts_gap',
                    'mcts_q_ablation_gap', 'mcts_qoff_score_rate', 'mcts_qoff_win_rate',
                    'mcts_qoff_draw_rate', 'mcts_qoff_loss_rate', 'mcts_qoff_games',
                    'score_rate', 'buffer_size', 'avg_game_length', 'temperature', 'beta',
                    'true_win_rate', 'eval_stage', 'eval_games',
                    'eval_wins', 'eval_draws', 'eval_losses', 'eval_unresolved',
                    'no_mcts_score_rate', 'no_mcts_win_rate',
                    'no_mcts_draw_rate', 'no_mcts_loss_rate',
                    'no_mcts_wins', 'no_mcts_draws', 'no_mcts_losses', 'no_mcts_unresolved',
                    'anchor_score_rate', 'anchor_true_win_rate', 'anchor_wins', 'anchor_draws', 'anchor_losses',
                    # 📊 NEW: Metrics
                    'policy_top1_acc', 'policy_top3_acc',
                    'value_mae', 'value_mae_weighted',
                    'value_wdl_acc', 'value_wdl_ce',
                    'value_mae_opening', 'value_mae_middlegame', 'value_mae_endgame',
                    'value_samples_opening', 'value_samples_middlegame', 'value_samples_endgame',
                    # 🧠 RL headline telemetry
                    'completed_draw_rate',
                    'avg_game_value', 'value_std',
                    'policy_entropy', 'value_pred_std',
                    'selfplay_decisive_rate', 'selfplay_auto_draw_rate',
                    'selfplay_truncated_rate',
                    'adaptive_temp_adjustment', 'adaptive_temp_threshold',
                    'rl_best_model',
                    'estimated_elo_nn', 'estimated_elo_mcts', 'estimated_elo_mcts_simulations',
                ]
            if config_snapshot is not None:
                writer.writerow([
                    _CSV_CONFIG_METADATA_KEY,
                    json.dumps(
                        _json_safe_config(config_snapshot),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(',', ':'),
                    ),
                ])
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
        self._pending_rl_elo_by_iteration = {}
        self.best_final_elo_info = None  # (epoch, elo) for exact final best-model Elo
        
        if mode == "rl":
            self.details_dir = self.log_dir / "details"
            self.details_dir.mkdir(parents=True, exist_ok=True)
            self.details_prefix = self.csv_path.stem
            self.performance_dir = self.details_dir
            self.performance_log_path = self.details_dir / f"{self.details_prefix}_performance.csv"
            self.performance_plot_path = self.details_dir / f"{self.details_prefix}_performance.png"
            self.data_quality_log_path = self.details_dir / f"{self.details_prefix}_data_quality.csv"
            self.data_quality_plot_path = self.details_dir / f"{self.details_prefix}_data_quality.png"
            self.debug_training_profile_dir = self.log_dir / "debug" / "training_profile"
            self.debug_training_profile_dir.mkdir(parents=True, exist_ok=True)
            self.latest_training_profile_path = self.debug_training_profile_dir / "rl_latest_training_profile.csv"
            for stale_profile in self.debug_training_profile_dir.glob("rl_*training_profile*.csv"):
                if stale_profile != self.latest_training_profile_path:
                    try:
                        stale_profile.unlink()
                    except OSError:
                        pass
            performance_header = [
                'iteration',
                'timestamp',
                'positions_per_sec',
                'iteration_total_time_s',
                'stage_setup_pct',
                'stage_selfplay_pct',
                'stage_replay_pct',
                'stage_train_pct',
                'stage_eval_log_pct',
                'stage_checkpoint_pct',
                'stage_gc_pct',
                'stage_setup_time_s',
                'stage_selfplay_time_s',
                'stage_replay_time_s',
                'stage_train_time_s',
                'stage_eval_log_time_s',
                'stage_checkpoint_time_s',
                'stage_gc_time_s',
                'selfplay_time_s',
                'data_collection_time_s',
                'avg_game_length',
                'mcts_avg_batch_size',
                'mcts_central_avg_batch_size',
                'central_remote_wait_ms_per_request',
                'central_request_put_ms_per_request',
                'central_server_queue_wait_ms_per_request',
                'central_server_concat_ms_per_request',
                'central_server_h2d_ms_per_request',
                'central_server_forward_ms_per_request',
                'central_server_d2h_ms_per_request',
                'central_server_total_ms_per_request',
                'central_remote_wait_time_s',
                'central_server_queue_wait_time_s',
                'central_server_forward_time_s',
                'central_server_total_time_s',
                'mcts_gpu_utilization_pct',
                'mcts_inference_ms_per_position',
                'mcts_inference_ms_per_batch',
                'mcts_worker_nn_wait_ms_per_position',
                'mcts_worker_nn_wait_ms_per_batch',
                'central_server_queue_wait_ms_per_position',
                'central_server_forward_ms_per_position',
                'central_server_total_ms_per_position',
                'mcts_search_many_time_s',
                'mcts_nn_inference_time_s',
                'mcts_nn_inference_calls',
                'mcts_nn_inference_batch_items',
                'mcts_avg_legal_moves_per_position',
                'queue_wait_time_ms',
                'mcts_search_selection_time_s',
                'mcts_search_backprop_time_s',
                'mcts_search_adaptive_stop_time_s',
                'mcts_search_metadata_time_s',
                'mcts_batch_expand_eval_time_s',
                'mcts_board_to_tensor_time_s',
                'mcts_batch_expand_dedup_terminal_time_s',
                'mcts_batch_expand_legal_moves_time_s',
                'mcts_batch_expand_history_time_s',
                'mcts_batch_expand_input_pack_time_s',
                'mcts_batch_expand_legal_index_pack_time_s',
                'mcts_batch_expand_cpu_policy_time_s',
                'mcts_batch_expand_value_fanout_time_s',
                'mcts_policy_target_build_time_s',
                'mcts_policy_target_postgame_time_s',
                'mcts_move_selection_time_s',
                'mcts_adjudication_time_s',
                'mcts_syzygy_time_s',
            ]
            with open(self.performance_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(performance_header)
            with open(self.latest_training_profile_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(performance_header)
            with open(self.data_quality_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration',
                    'timestamp',
                    'positions_added',
                    'replay_size',
                    'replay_capacity',
                    'replay_fill_rate',
                    'replay_decisive_fraction',
                    'replay_draw_fraction',
                    'replay_value_mean',
                    'replay_value_std',
                    'replay_value_positive_fraction',
                    'replay_value_neutral_fraction',
                    'replay_value_negative_fraction',
                    'replay_recent_decisive_fraction',
                    'replay_recent_draw_fraction',
                    'replay_recent_value_mean',
                    'replay_recent_value_std',
                    'replay_recent_value_positive_fraction',
                    'replay_recent_value_neutral_fraction',
                    'replay_recent_value_negative_fraction',
                    'policy_weight_mean',
                    'policy_weight_p10',
                    'policy_weight_low_fraction',
                    'value_weight_mean',
                    'value_weight_p10',
                    'value_weight_low_fraction',
                    'policy_target_len_mean',
                    'policy_target_len_p90',
                    'policy_target_entropy_mean',
                    'policy_target_top1_prob_mean',
                    'policy_target_top3_prob_mean',
                    'policy_target_effective_moves',
                    'importance_mean',
                    'importance_p90',
                    'sample_age_avg',
                    'sample_age_p10',
                    'sample_age_p50',
                    'sample_age_p90',
                    'sample_age_new_fraction',
                    'sample_age_le1_fraction',
                    'selfplay_completed_games',
                    'selfplay_draw_rate',
                    'replay_vs_selfplay_draw_gap',
                    'selfplay_decisive_rate',
                    'selfplay_truncated_rate',
                    'selfplay_auto_draw_rate',
                    'selfplay_avg_game_value',
                    'selfplay_value_std',
                    'mcts_avg_sims',
                    'mcts_avg_budget',
                    'mcts_budget_p10',
                    'mcts_budget_p90',
                    'mcts_budget_min',
                    'mcts_budget_max',
                    'mcts_p10_sims',
                    'mcts_extra_budget_rate',
                    'mcts_adaptive_stop_rate',
                    'mcts_prior_agreement_samples',
                    'mcts_prior_agreement_rate',
                    'mcts_prior_changed_rate',
                    'mcts_search_discovery_rate',
                    'mcts_search_discovery_weight_mean',
                    'mcts_q_delta_samples',
                    'mcts_changed_to_lower_q_rate',
                    'mcts_q_delta_mean',
                    'mcts_q_delta_p10',
                    'mcts_q_delta_p50',
                    'mcts_q_delta_p90',
                    'mcts_changed_q_delta_samples',
                    'mcts_changed_to_higher_q_rate',
                    'mcts_changed_to_lower_q_when_changed_rate',
                    'mcts_changed_q_delta_mean',
                    'mcts_changed_q_delta_p10',
                    'mcts_changed_q_delta_p50',
                    'mcts_changed_q_delta_p90',
                    'mcts_policy_kl_mean',
                    'mcts_prior_top_visit_prob_mean',
                    'mcts_top_prior_prob_mean',
                    'mcts_explored_prior_mass_mean',
                    'mcts_visited_move_count_mean',
                    'mcts_legal_move_count_mean',
                    'train_policy_entropy',
                    'train_target_value_std',
                    'train_value_mae',
                    'train_value_mae_opening',
                    'train_value_mae_middlegame',
                    'train_value_mae_endgame',
                    'train_value_samples_opening',
                    'train_value_samples_middlegame',
                    'train_value_samples_endgame',
                    'eval_mcts_score_rate',
                    'eval_no_mcts_score_rate',
                    'eval_mcts_qoff_score_rate',
                    'eval_mcts_q_ablation_gap',
                    'eval_mcts_games',
                    'eval_mcts_qoff_games',
                ])
            self.win_rates = []
            self.true_win_rates = []
            self.no_mcts_score_rates = []
            self.no_mcts_true_win_rates = []
            self.no_mcts_draw_rates = []
            self.no_mcts_loss_rates = []
            self.anchor_score_rates = []
            self.anchor_true_win_rates = []
            self.rl_best_model_markers = []
            self.temperatures = []
            self.adaptive_temp_adjustments = []
            self.adaptive_temp_thresholds = []
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
        self.plot_smoothing_enabled = False
        self.plot_smoothing_alpha = 0.35
        self.plot_smoothing_min_points = 5
        # Optional notes shown in summary panel (e.g. final SWA metrics).
        self.final_notes = []
        # Optional epoch markers drawn on IL Elo chart (e.g. SWA final epoch).
        self.elo_epoch_markers = []  # (epoch, label)
        # 🆕 SWA elo stored separately for distinct visual treatment in plots.
        self.swa_elo_info = None  # (epoch, elo) or None
        # 🆕 Full SWA metrics for summary panel.
        self.swa_metrics = None  # dict: {epoch, val_loss, top1, top3, mae, wdl_acc, wdl_ce, elo}
        
        print(f"Logging to: {self.csv_path}")
        if self.mode == "rl":
            print(f"RL performance log: {self.performance_log_path}")
            print(f"RL data-quality log: {self.data_quality_log_path}")

    def set_run_context(self, text):
        """Set optional short context displayed on generated PNG plots."""
        if text is None:
            self.run_context_text = None
            return
        text = str(text).strip()
        self.run_context_text = text if text else None

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

    def add_rl_best_model_marker(self, iteration, label=None):
        """Mark an RL iteration where the best model was replaced."""
        if self.mode != "rl":
            return
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            return

        text = str(label).strip() if label is not None else ""
        if not text:
            text = "New RL best"

        replaced = False
        for idx, (it, _) in enumerate(self.rl_best_model_markers):
            if int(it) == iteration:
                self.rl_best_model_markers[idx] = (iteration, text)
                replaced = True
                break
        if not replaced:
            self.rl_best_model_markers.append((iteration, text))
            self.rl_best_model_markers.sort(key=lambda x: x[0])

    def log_rl_performance(
        self,
        iteration,
        positions_per_sec=None,
        iteration_total_time=None,
        selfplay_time=None,
        data_collection_time=None,
        avg_game_length=None,
        profile=None,
        stage_times=None,
    ):
        """Append RL speed/profile metrics to the single latest performance log."""
        if self.mode != "rl" or self.performance_log_path is None:
            return
        profile = dict(profile or {})
        stage_times = dict(stage_times or {})

        def _float_or_none(value):
            try:
                if value is None or value == '':
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _int_or_none(value):
            try:
                if value is None or value == '':
                    return None
                return int(float(value))
            except (TypeError, ValueError):
                return None

        nn_calls = _int_or_none(profile.get('mcts_nn_inference_calls'))
        nn_items = _int_or_none(profile.get('mcts_nn_inference_batch_items'))
        if nn_calls and nn_calls > 0 and nn_items is not None:
            profile['average_batch_size'] = float(nn_items) / float(nn_calls)
            nn_time = _float_or_none(profile.get('mcts_nn_inference_time'))
            if nn_time is not None:
                profile['inference_time_per_batch_ms'] = 1000.0 * float(nn_time) / float(nn_calls)
        if nn_items and nn_items > 0:
            nn_time = _float_or_none(profile.get('mcts_nn_inference_time'))
            if nn_time is not None:
                profile['inference_time_per_position_ms'] = 1000.0 * float(nn_time) / float(nn_items)
            legal_items = _int_or_none(profile.get('mcts_nn_legal_move_items'))
            if legal_items is not None:
                profile['average_legal_moves_per_position'] = float(legal_items) / float(nn_items)

        central_requests = _int_or_none(profile.get('mcts_central_inference_requests'))
        central_items = _int_or_none(profile.get('mcts_central_inference_server_batch_items'))
        if central_requests and central_requests > 0 and central_items is not None:
            profile['central_average_batch_size'] = float(central_items) / float(central_requests)
            for raw_key, avg_key in [
                ('mcts_central_inference_remote_wait_time', 'central_remote_wait_ms_per_request'),
                ('mcts_central_inference_request_put_time', 'central_request_put_ms_per_request'),
                ('mcts_central_inference_server_queue_wait_time', 'central_server_queue_wait_ms_per_request'),
                ('mcts_central_inference_server_concat_time', 'central_server_concat_ms_per_request'),
                ('mcts_central_inference_server_h2d_time', 'central_server_h2d_ms_per_request'),
                ('mcts_central_inference_server_forward_time', 'central_server_forward_ms_per_request'),
                ('mcts_central_inference_server_d2h_time', 'central_server_d2h_ms_per_request'),
                ('mcts_central_inference_server_total_time', 'central_server_total_ms_per_request'),
            ]:
                raw_value = _float_or_none(profile.get(raw_key))
                if raw_value is not None:
                    profile[avg_key] = 1000.0 * float(raw_value) / float(central_requests)

        search_time = _float_or_none(profile.get('mcts_search_many_time'))
        nn_time = _float_or_none(profile.get('mcts_nn_inference_time'))
        if search_time and search_time > 0.0 and nn_time is not None:
            profile['gpu_utilization_pct'] = max(0.0, min(100.0, 100.0 * float(nn_time) / float(search_time)))

        worker_wait_batch_ms = _float_or_none(profile.get('inference_time_per_batch_ms'))
        worker_wait_pos_ms = _float_or_none(profile.get('inference_time_per_position_ms'))
        if worker_wait_batch_ms is not None:
            profile.setdefault('worker_nn_wait_ms_per_batch', worker_wait_batch_ms)
        if worker_wait_pos_ms is not None:
            profile.setdefault('worker_nn_wait_ms_per_position', worker_wait_pos_ms)

        central_avg_batch = _float_or_none(profile.get('central_average_batch_size'))
        for src_key, dst_key in [
            ('central_server_queue_wait_ms_per_request', 'central_server_queue_wait_ms_per_position'),
            ('central_server_forward_ms_per_request', 'central_server_forward_ms_per_position'),
            ('central_server_total_ms_per_request', 'central_server_total_ms_per_position'),
        ]:
            value = _float_or_none(profile.get(src_key))
            if value is not None and central_avg_batch and central_avg_batch > 0.0:
                profile.setdefault(dst_key, float(value) / float(central_avg_batch))

        def _value(key, default=''):
            value = profile.get(key, default)
            return default if value is None else value

        iteration_total_float = _float_or_none(iteration_total_time)

        def _stage_pct(key):
            stage_value = _float_or_none(stage_times.get(key))
            if stage_value is None or iteration_total_float is None or iteration_total_float <= 0.0:
                return ''
            return float(stage_value) / float(iteration_total_float)

        row = [
            int(iteration),
            datetime.now().isoformat(timespec='seconds'),
            '' if positions_per_sec is None else positions_per_sec,
            '' if iteration_total_time is None else iteration_total_time,
            _stage_pct('setup'),
            _stage_pct('selfplay'),
            _stage_pct('replay'),
            _stage_pct('train'),
            _stage_pct('eval_log'),
            _stage_pct('checkpoint'),
            _stage_pct('gc'),
            stage_times.get('setup', ''),
            stage_times.get('selfplay', ''),
            stage_times.get('replay', ''),
            stage_times.get('train', ''),
            stage_times.get('eval_log', ''),
            stage_times.get('checkpoint', ''),
            stage_times.get('gc', ''),
            '' if selfplay_time is None else selfplay_time,
            '' if data_collection_time is None else data_collection_time,
            '' if avg_game_length is None else avg_game_length,
            _value('average_batch_size'),
            _value('central_average_batch_size'),
            _value('central_remote_wait_ms_per_request'),
            _value('central_request_put_ms_per_request'),
            _value('central_server_queue_wait_ms_per_request'),
            _value('central_server_concat_ms_per_request'),
            _value('central_server_h2d_ms_per_request'),
            _value('central_server_forward_ms_per_request'),
            _value('central_server_d2h_ms_per_request'),
            _value('central_server_total_ms_per_request'),
            _value('mcts_central_inference_remote_wait_time'),
            _value('mcts_central_inference_server_queue_wait_time'),
            _value('mcts_central_inference_server_forward_time'),
            _value('mcts_central_inference_server_total_time'),
            _value('gpu_utilization_pct'),
            _value('inference_time_per_position_ms'),
            _value('inference_time_per_batch_ms'),
            _value('worker_nn_wait_ms_per_position'),
            _value('worker_nn_wait_ms_per_batch'),
            _value('central_server_queue_wait_ms_per_position'),
            _value('central_server_forward_ms_per_position'),
            _value('central_server_total_ms_per_position'),
            _value('mcts_search_many_time'),
            _value('mcts_nn_inference_time'),
            _value('mcts_nn_inference_calls'),
            _value('mcts_nn_inference_batch_items'),
            _value('average_legal_moves_per_position'),
            _value('queue_wait_time_ms'),
            _value('mcts_search_selection_time'),
            _value('mcts_search_backprop_time'),
            _value('mcts_search_adaptive_stop_time'),
            _value('mcts_search_metadata_time'),
            _value('mcts_batch_expand_eval_time'),
            _value('mcts_board_to_tensor_time'),
            _value('mcts_batch_expand_dedup_terminal_time'),
            _value('mcts_batch_expand_legal_moves_time'),
            _value('mcts_batch_expand_history_time'),
            _value('mcts_batch_expand_input_pack_time'),
            _value('mcts_batch_expand_legal_index_pack_time'),
            _value('mcts_batch_expand_cpu_policy_time'),
            _value('mcts_batch_expand_value_fanout_time'),
            _value('mcts_policy_target_build_time'),
            _value('mcts_policy_target_postgame_time'),
            _value('mcts_move_selection_time'),
            _value('mcts_adjudication_time'),
            _value('mcts_syzygy_time'),
        ]
        with open(self.performance_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        latest_path = getattr(self, 'latest_training_profile_path', None)
        if latest_path is not None:
            with open(latest_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

    def plot_rl_performance(self):
        """Generate a performance plot from this run's RL details CSV."""
        if self.mode != "rl" or self.performance_log_path is None:
            return
        if not self.performance_log_path.exists():
            return

        try:
            with open(self.performance_log_path, 'r', newline='') as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return
        if not rows:
            return

        avg_rows = rows[-min(5, len(rows)):]

        def _series(column):
            xs = []
            ys = []
            for row in rows:
                try:
                    x = int(float(row.get('iteration', '')))
                    raw = row.get(column, '')
                    if raw is None or raw == '':
                        continue
                    y = float(raw)
                except (TypeError, ValueError):
                    continue
                xs.append(x)
                ys.append(y)
            return xs, ys

        def _series_any(*columns):
            for column in columns:
                xs, ys = _series(column)
                if xs:
                    return xs, ys
            return [], []

        def _values(column, subset=None):
            values = []
            for row in (subset or rows):
                raw = row.get(column, '')
                if raw is None or raw == '':
                    continue
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
            return values

        def _avg(column, subset=None):
            values = _values(column, subset=subset)
            if not values:
                return None
            return sum(values) / len(values)

        def _avg_any(columns, subset=None):
            for column in columns:
                value = _avg(column, subset=subset)
                if value is not None:
                    return value
            return None

        def _sum(columns, subset=None):
            total = 0.0
            found = False
            for column in columns:
                value = _avg(column, subset=subset)
                if value is not None:
                    total += float(value)
                    found = True
            return total if found else None

        def _avg_total_seconds(total_column, per_request_column=None, subset=None):
            value = _avg(total_column, subset=subset)
            if value is not None:
                return value
            return None

        def _latest(column):
            for row in reversed(rows):
                raw = row.get(column, '')
                if raw not in ('', None):
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return raw
            return None

        fig, axes = plt.subplots(4, 2, figsize=(18, 15))
        fig.patch.set_facecolor('#F7F8FA')
        fig.suptitle('RL Performance Details', fontsize=17, fontweight='bold', y=0.985)

        colors = {
            'green': '#16A34A',
            'blue': '#2563EB',
            'purple': '#7C3AED',
            'red': '#DC2626',
            'orange': '#EA580C',
            'cyan': '#0891B2',
            'slate': '#475569',
            'black': '#111827',
        }

        def _style_axis(ax, title, ylabel=None):
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
            ax.set_xlabel('Iteration')
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_alpha(0.18)

        def _plot(ax, column, label, color, style='-', marker='o', linewidth=2.0, alpha=0.95):
            xs, ys = _series(column)
            if xs:
                ax.plot(xs, ys, linestyle=style, marker=marker, linewidth=linewidth, markersize=4, color=color, alpha=alpha, label=label)
            return xs, ys

        def _plot_any(ax, columns, label, color, style='-', marker='o', linewidth=2.0, alpha=0.95):
            xs, ys = _series_any(*columns)
            if xs:
                ax.plot(xs, ys, linestyle=style, marker=marker, linewidth=linewidth, markersize=4, color=color, alpha=alpha, label=label)
            return xs, ys

        ax = axes[0, 0]
        _plot(ax, 'positions_per_sec', 'Positions/s', colors['green'])
        _style_axis(ax, 'Throughput', 'positions / second')
        ax2 = ax.twinx()
        xs2, ys2 = _series('mcts_central_avg_batch_size')
        if xs2:
            ax2.plot(xs2, ys2, color=colors['purple'], linestyle='--', marker='s', linewidth=1.9, markersize=4, label='Central avg batch')
        ax2.set_ylabel('central batch')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[0, 1]
        for column, label, color, style in [
            ('iteration_total_time_s', 'Iteration total', colors['black'], '-'),
            ('stage_selfplay_time_s', 'Self-play stage', colors['red'], '--'),
            ('stage_train_time_s', 'Train stage', colors['blue'], ':'),
            ('stage_eval_log_time_s', 'Eval/log stage', colors['orange'], '-.'),
        ]:
            _plot(ax, column, label, color, style=style, marker=None, linewidth=1.9, alpha=0.88)
        _style_axis(ax, 'Runtime Breakdown', 'seconds')
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, fontsize=8, loc='best')

        ax = axes[1, 0]
        for column, label, color, style in [
            ('central_remote_wait_ms_per_request', 'worker wait', colors['red'], '-'),
            ('central_server_queue_wait_ms_per_request', 'server queue', colors['cyan'], '--'),
            ('central_server_forward_ms_per_request', 'server forward', colors['green'], '-'),
            ('central_server_total_ms_per_request', 'server total', colors['purple'], '--'),
        ]:
            _plot(ax, column, label, color, style=style, marker=None, linewidth=1.8, alpha=0.9)
        _style_axis(ax, 'Central Inference Latency', 'ms / request')
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, fontsize=8, loc='best')

        ax = axes[1, 1]
        _plot(ax, 'mcts_avg_batch_size', 'Worker avg batch', colors['cyan'])
        _plot(ax, 'mcts_central_avg_batch_size', 'Central avg batch', colors['green'], marker='^')
        _style_axis(ax, 'Batching And Occupancy', 'batch items')
        ax2 = ax.twinx()
        xs, ys = _series('mcts_gpu_utilization_pct')
        if ys and any(float(y) > 100.0 for y in ys):
            xs = []
            ys = []
            for row in rows:
                try:
                    iteration = int(float(row.get('iteration', '')))
                    nn_time = float(row.get('mcts_nn_inference_time_s', '') or 0.0)
                    search_time = float(row.get('mcts_search_many_time_s', '') or 0.0)
                except (TypeError, ValueError):
                    continue
                if search_time <= 0.0:
                    continue
                xs.append(iteration)
                ys.append(max(0.0, min(100.0, 100.0 * nn_time / search_time)))
        if xs:
            ax2.plot(xs, ys, color=colors['orange'], marker='o', linestyle='--', linewidth=1.8, markersize=4, label='GPU proxy')
        ax2.set_ylabel('GPU proxy')
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(PercentFormatter(100.0))
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[2, 0]
        _plot_any(ax, ['mcts_worker_nn_wait_ms_per_batch', 'mcts_inference_ms_per_batch'], 'worker NN wait/batch', colors['red'])
        _plot(ax, 'central_server_forward_ms_per_request', 'server forward/batch', colors['green'], style='--', marker='s')
        _plot(ax, 'central_server_queue_wait_ms_per_request', 'server queue/batch', colors['purple'], style='--', marker='^')
        _plot(ax, 'queue_wait_time_ms', 'result queue ms', colors['cyan'], style=':', marker='^')
        _style_axis(ax, 'Inference And IPC Cost', 'ms')
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, fontsize=8, loc='best')

        ax = axes[2, 1]
        expand_breakdown = [
            ('legal moves', _avg('mcts_batch_expand_legal_moves_time_s', avg_rows), colors['blue']),
            ('history', _avg('mcts_batch_expand_history_time_s', avg_rows), colors['purple']),
            ('input pack', _avg('mcts_batch_expand_input_pack_time_s', avg_rows), colors['green']),
            ('legal index', _avg('mcts_batch_expand_legal_index_pack_time_s', avg_rows), colors['cyan']),
            ('board tensor', _avg('mcts_board_to_tensor_time_s', avg_rows), colors['orange']),
            ('cpu policy', _avg('mcts_batch_expand_cpu_policy_time_s', avg_rows), colors['red']),
            ('value fanout', _avg('mcts_batch_expand_value_fanout_time_s', avg_rows), colors['slate']),
            ('dedup/terminal', _avg('mcts_batch_expand_dedup_terminal_time_s', avg_rows), colors['black']),
        ]
        expand_breakdown = [(label, value, color) for label, value, color in expand_breakdown if value is not None and value > 0.0]
        expand_breakdown.sort(key=lambda item: item[1])
        if expand_breakdown:
            labels = [item[0] for item in expand_breakdown]
            values = [item[1] for item in expand_breakdown]
            bar_colors = [item[2] for item in expand_breakdown]
            ax.barh(labels, values, color=bar_colors, alpha=0.86)
        else:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                "Detailed expand/tensor timing\navailable from next logged iteration",
                ha='center',
                va='center',
                fontsize=11,
                color=colors['slate'],
            )
        _style_axis(ax, f'Expand / Tensor Cost AVG (last {len(avg_rows)})', 'worker-summed seconds')
        ax.set_xlabel('seconds')

        ax = axes[3, 0]
        timing_breakdown = [
            ('worker NN wait', _avg_total_seconds('mcts_nn_inference_time_s', subset=avg_rows), colors['red']),
            ('server queue', _avg_total_seconds('central_server_queue_wait_time_s', 'central_server_queue_wait_ms_per_request', avg_rows), colors['purple']),
            ('server forward', _avg_total_seconds('central_server_forward_time_s', 'central_server_forward_ms_per_request', avg_rows), colors['green']),
            ('server total', _avg_total_seconds('central_server_total_time_s', 'central_server_total_ms_per_request', avg_rows), colors['cyan']),
            ('selection', _avg('mcts_search_selection_time_s', avg_rows), colors['blue']),
            ('expand/eval', _avg('mcts_batch_expand_eval_time_s', avg_rows), colors['orange']),
            ('backprop', _avg('mcts_search_backprop_time_s', avg_rows), colors['slate']),
        ]
        timing_breakdown = [(label, value, color) for label, value, color in timing_breakdown if value is not None and value > 0.0]
        timing_breakdown.sort(key=lambda item: item[1])
        if timing_breakdown:
            labels = [item[0] for item in timing_breakdown]
            values = [item[1] for item in timing_breakdown]
            bar_colors = [item[2] for item in timing_breakdown]
            ax.barh(labels, values, color=bar_colors, alpha=0.86)
        else:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                "Timing breakdown\navailable from next logged iteration",
                ha='center',
                va='center',
                fontsize=11,
                color=colors['slate'],
            )
        _style_axis(ax, f'MCTS Timing AVG (last {len(avg_rows)})', 'worker/server summed seconds')
        ax.set_xlabel('seconds')

        ax = axes[3, 1]
        ax.axis('off')
        left_lines = ["Loop", ""]
        for column, label, fmt in [
            ('positions_per_sec', 'Positions/s', '{:.1f}'),
            ('iteration_total_time_s', 'Iter total', '{:.1f}s'),
            ('stage_selfplay_time_s', 'Self-play', '{:.1f}s'),
            ('stage_train_time_s', 'Train', '{:.1f}s'),
            ('stage_eval_log_time_s', 'Eval/log', '{:.1f}s'),
            ('stage_selfplay_pct', 'Self-play share', '{:.1%}'),
        ]:
            value = _avg(column, avg_rows)
            if isinstance(value, (int, float)):
                left_lines.append(f"{label}: {fmt.format(value)}")
        latest_positions = _latest('positions_per_sec')
        avg_positions = _avg('positions_per_sec', rows)
        if isinstance(latest_positions, (int, float)) and isinstance(avg_positions, (int, float)):
            left_lines.extend(["", f"Latest pos/s: {latest_positions:.1f}", f"Run avg pos/s: {avg_positions:.1f}"])

        right_lines = ["Central/MCTS", ""]
        for column, label, fmt in [
            ('mcts_central_avg_batch_size', 'Central batch', '{:.1f}'),
            ('mcts_avg_batch_size', 'Worker batch', '{:.1f}'),
            ('mcts_gpu_utilization_pct', 'GPU proxy', '{:.1f}%'),
            ('central_remote_wait_ms_per_request', 'Worker wait', '{:.0f}ms'),
            ('central_server_queue_wait_ms_per_request', 'Server queue', '{:.0f}ms'),
            ('central_server_forward_ms_per_request', 'Forward', '{:.0f}ms'),
            ('mcts_worker_nn_wait_ms_per_position', 'Wait ms/pos', '{:.1f}ms'),
            ('central_server_forward_ms_per_position', 'Forward ms/pos', '{:.1f}ms'),
            ('queue_wait_time_ms', 'Result queue', '{:.0f}ms'),
        ]:
            if column == 'mcts_worker_nn_wait_ms_per_position':
                value = _avg_any(['mcts_worker_nn_wait_ms_per_position', 'mcts_inference_ms_per_position'], avg_rows)
            else:
                value = _avg(column, avg_rows)
            if isinstance(value, (int, float)):
                right_lines.append(f"{label}: {fmt.format(value)}")

        bottleneck_hint = None
        avg_queue = _avg('central_server_queue_wait_ms_per_request', avg_rows)
        avg_forward = _avg('central_server_forward_ms_per_request', avg_rows)
        avg_batch = _avg('mcts_central_avg_batch_size', avg_rows)
        if avg_queue is not None and avg_forward is not None:
            if avg_queue > avg_forward * 1.5:
                bottleneck_hint = "Likely bottleneck: batching/queue wait"
            elif avg_forward > avg_queue * 1.2:
                bottleneck_hint = "Likely bottleneck: GPU forward"
            else:
                bottleneck_hint = "Likely bottleneck: mixed queue+forward"
        if bottleneck_hint:
            right_lines.extend(["", bottleneck_hint])
        if avg_batch is not None and avg_batch < 64:
            right_lines.append("Batch target: increase central batch")

        ax.set_title('AVG Summary', fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.text(
            0.04,
            0.82,
            f"AVG Summary (last {len(avg_rows)} iters)",
            fontsize=10.0,
            family='monospace',
            fontweight='bold',
            verticalalignment='top',
            transform=ax.transAxes,
        )
        ax.text(
            0.04,
            0.74,
            "\n".join(left_lines),
            fontsize=8.7,
            family='monospace',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#CBD5E1', alpha=0.95),
        )
        ax.text(
            0.52,
            0.74,
            "\n".join(right_lines),
            fontsize=8.7,
            family='monospace',
            verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#CBD5E1', alpha=0.95),
        )

        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.06, top=0.94, hspace=0.46, wspace=0.30)
        fig.savefig(self.performance_plot_path, dpi=150)
        plt.close(fig)

    def log_rl_data_quality(
        self,
        iteration,
        *,
        positions_added=None,
        replay_stats=None,
        selfplay_stats=None,
        train_metrics=None,
        eval_stats=None,
        train_policy_entropy=None,
        train_target_value_std=None,
    ):
        """Append detailed RL data-quality metrics for this run."""
        if self.mode != "rl" or self.data_quality_log_path is None:
            return
        replay_stats = dict(replay_stats or {})
        selfplay_stats = dict(selfplay_stats or {})
        train_metrics = dict(train_metrics or {})
        eval_stats = dict(eval_stats or {})

        def _value(mapping, key, default=''):
            value = mapping.get(key, default)
            return default if value is None else value

        replay_draw = _value(replay_stats, 'draw_fraction', None)
        selfplay_draw = _value(selfplay_stats, 'completed_draw_rate', None)
        try:
            replay_vs_selfplay_draw_gap = float(replay_draw) - float(selfplay_draw)
        except (TypeError, ValueError):
            replay_vs_selfplay_draw_gap = ''

        row = [
            int(iteration),
            datetime.now().isoformat(timespec='seconds'),
            '' if positions_added is None else positions_added,
            _value(replay_stats, 'size'),
            _value(replay_stats, 'capacity'),
            _value(replay_stats, 'fill_rate'),
            _value(replay_stats, 'decisive_fraction'),
            _value(replay_stats, 'draw_fraction'),
            _value(replay_stats, 'value_mean'),
            _value(replay_stats, 'value_std'),
            _value(replay_stats, 'value_positive_fraction'),
            _value(replay_stats, 'value_neutral_fraction'),
            _value(replay_stats, 'value_negative_fraction'),
            _value(replay_stats, 'recent_decisive_fraction'),
            _value(replay_stats, 'recent_draw_fraction'),
            _value(replay_stats, 'recent_value_mean'),
            _value(replay_stats, 'recent_value_std'),
            _value(replay_stats, 'recent_value_positive_fraction'),
            _value(replay_stats, 'recent_value_neutral_fraction'),
            _value(replay_stats, 'recent_value_negative_fraction'),
            _value(replay_stats, 'policy_weight_mean'),
            _value(replay_stats, 'policy_weight_p10'),
            _value(replay_stats, 'policy_weight_low_fraction'),
            _value(replay_stats, 'value_weight_mean'),
            _value(replay_stats, 'value_weight_p10'),
            _value(replay_stats, 'value_weight_low_fraction'),
            _value(replay_stats, 'policy_target_len_mean'),
            _value(replay_stats, 'policy_target_len_p90'),
            _value(replay_stats, 'policy_target_entropy_mean'),
            _value(replay_stats, 'policy_target_top1_prob_mean'),
            _value(replay_stats, 'policy_target_top3_prob_mean'),
            _value(replay_stats, 'policy_target_effective_moves'),
            _value(replay_stats, 'importance_mean'),
            _value(replay_stats, 'importance_p90'),
            _value(replay_stats, 'sample_age_avg'),
            _value(replay_stats, 'sample_age_p10'),
            _value(replay_stats, 'sample_age_p50'),
            _value(replay_stats, 'sample_age_p90'),
            _value(replay_stats, 'sample_age_new_fraction'),
            _value(replay_stats, 'sample_age_le1_fraction'),
            _value(selfplay_stats, 'completed_games'),
            _value(selfplay_stats, 'completed_draw_rate'),
            replay_vs_selfplay_draw_gap,
            _value(selfplay_stats, 'decisive_rate'),
            _value(selfplay_stats, 'truncated_rate'),
            _value(selfplay_stats, 'auto_draw_rate'),
            _value(selfplay_stats, 'avg_game_value'),
            _value(selfplay_stats, 'value_std'),
            _value(selfplay_stats, 'search_simulations_used_avg'),
            _value(selfplay_stats, 'search_simulations_budget_avg'),
            _value(selfplay_stats, 'search_simulations_budget_p10'),
            _value(selfplay_stats, 'search_simulations_budget_p90'),
            _value(selfplay_stats, 'search_simulations_budget_min'),
            _value(selfplay_stats, 'search_simulations_budget_max'),
            _value(selfplay_stats, 'search_simulations_used_p10'),
            _value(selfplay_stats, 'search_extra_budget_rate'),
            _value(selfplay_stats, 'adaptive_stop_rate'),
            _value(selfplay_stats, 'mcts_prior_agreement_samples'),
            _value(selfplay_stats, 'mcts_prior_agreement_rate'),
            _value(selfplay_stats, 'mcts_prior_changed_rate'),
            _value(selfplay_stats, 'mcts_search_discovery_rate'),
            _value(selfplay_stats, 'mcts_search_discovery_weight_mean'),
            _value(selfplay_stats, 'mcts_q_delta_samples'),
            _value(selfplay_stats, 'mcts_changed_to_lower_q_rate'),
            _value(selfplay_stats, 'mcts_q_delta_mean'),
            _value(selfplay_stats, 'mcts_q_delta_p10'),
            _value(selfplay_stats, 'mcts_q_delta_p50'),
            _value(selfplay_stats, 'mcts_q_delta_p90'),
            _value(selfplay_stats, 'mcts_changed_q_delta_samples'),
            _value(selfplay_stats, 'mcts_changed_to_higher_q_rate'),
            _value(selfplay_stats, 'mcts_changed_to_lower_q_when_changed_rate'),
            _value(selfplay_stats, 'mcts_changed_q_delta_mean'),
            _value(selfplay_stats, 'mcts_changed_q_delta_p10'),
            _value(selfplay_stats, 'mcts_changed_q_delta_p50'),
            _value(selfplay_stats, 'mcts_changed_q_delta_p90'),
            _value(selfplay_stats, 'mcts_policy_kl_mean'),
            _value(selfplay_stats, 'mcts_prior_top_visit_prob_mean'),
            _value(selfplay_stats, 'mcts_top_prior_prob_mean'),
            _value(selfplay_stats, 'mcts_explored_prior_mass_mean'),
            _value(selfplay_stats, 'mcts_visited_move_count_mean'),
            _value(selfplay_stats, 'mcts_legal_move_count_mean'),
            '' if train_policy_entropy is None else train_policy_entropy,
            '' if train_target_value_std is None else train_target_value_std,
            _value(train_metrics, 'value_mae'),
            _value(train_metrics, 'value_mae_opening'),
            _value(train_metrics, 'value_mae_middlegame'),
            _value(train_metrics, 'value_mae_endgame'),
            _value(train_metrics, 'value_samples_opening'),
            _value(train_metrics, 'value_samples_middlegame'),
            _value(train_metrics, 'value_samples_endgame'),
            _value(eval_stats, 'mcts_score_rate'),
            _value(eval_stats, 'no_mcts_score_rate'),
            _value(eval_stats, 'mcts_qoff_score_rate'),
            _value(eval_stats, 'mcts_q_ablation_gap'),
            _value(eval_stats, 'mcts_games'),
            _value(eval_stats, 'mcts_qoff_games'),
        ]
        with open(self.data_quality_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def plot_rl_data_quality(self):
        """Generate a data-quality plot from this run's RL details CSV."""
        if self.mode != "rl" or self.data_quality_log_path is None:
            return
        if not self.data_quality_log_path.exists():
            return

        try:
            with open(self.data_quality_log_path, 'r', newline='') as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return
        if not rows:
            return

        def _series(column):
            xs = []
            ys = []
            for row in rows:
                try:
                    x = int(float(row.get('iteration', '')))
                    raw = row.get(column, '')
                    if raw is None or raw == '':
                        continue
                    y = float(raw)
                except (TypeError, ValueError):
                    continue
                xs.append(x)
                ys.append(y)
            return xs, ys

        fig, axes = plt.subplots(5, 3, figsize=(20, 21.2))
        fig.patch.set_facecolor('#F7F8FA')
        fig.suptitle('RL Data Quality Details', fontsize=17, fontweight='bold', y=0.982)

        colors = {
            'green': '#16A34A',
            'blue': '#2563EB',
            'purple': '#7C3AED',
            'red': '#DC2626',
            'orange': '#EA580C',
            'cyan': '#0891B2',
            'slate': '#475569',
            'black': '#111827',
            'pink': '#DB2777',
        }

        def _style_axis(ax, title, ylabel=None, percent=False):
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
            ax.set_xlabel('Iteration')
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_alpha(0.18)
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        def _plot(ax, column, label, color, style='-', marker=None, linewidth=2.0, alpha=0.95):
            xs, ys = _series(column)
            if xs:
                ax.plot(xs, ys, linestyle=style, marker=marker, linewidth=linewidth, markersize=4 if marker else 0, color=color, alpha=alpha, label=label)
            return xs, ys

        def _set_tight_ylim(ax, values, *, min_pad=0.01, center_zero=False):
            values = [float(v) for v in values if v is not None]
            if not values:
                return
            y_min = min(values)
            y_max = max(values)
            if center_zero:
                span = max(abs(y_min), abs(y_max), min_pad)
                ax.set_ylim(-span * 1.18, span * 1.18)
                return
            pad = max(min_pad, (y_max - y_min) * 0.18)
            ax.set_ylim(y_min - pad, y_max + pad)

        def _latest(column):
            for row in reversed(rows):
                raw = row.get(column, '')
                if raw not in ('', None):
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        return raw
            return None

        ax = axes[0, 0]
        for column, label, color, style in [
            ('selfplay_decisive_rate', 'Self-play decisive', colors['green'], '-'),
            ('selfplay_draw_rate', 'Self-play draw', colors['cyan'], '-'),
            ('selfplay_auto_draw_rate', 'Auto-draw claim', colors['orange'], '-.'),
            ('replay_decisive_fraction', 'Replay decisive', colors['slate'], '--'),
            ('replay_draw_fraction', 'Replay draw', colors['blue'], '--'),
            ('selfplay_truncated_rate', 'Truncated', colors['red'], ':'),
        ]:
            _plot(ax, column, label, color, style=style)
        _style_axis(ax, 'Outcome Mix', 'rate', percent=True)
        outcome_values = []
        for column in [
            'selfplay_decisive_rate',
            'selfplay_draw_rate',
            'selfplay_auto_draw_rate',
            'replay_decisive_fraction',
            'replay_draw_fraction',
            'selfplay_truncated_rate',
        ]:
            _, values = _series(column)
            outcome_values.extend(values)
        if outcome_values:
            ax.set_ylim(0.0, min(1.0, max(0.35, max(outcome_values) * 1.18)))
        else:
            ax.set_ylim([0, 1])
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, loc='best', fontsize=8)

        ax = axes[0, 1]
        for column, label, color, style in [
            ('sample_age_avg', 'Avg age', colors['purple'], '-'),
            ('sample_age_p50', 'p50', '#A855F7', '--'),
            ('sample_age_p90', 'p90', '#6D28D9', ':'),
        ]:
            _plot(ax, column, label, color, style=style)
        _style_axis(ax, 'Replay Sample Age', 'iterations old')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = [], []
        frac_new_xs, _ = _series('sample_age_new_fraction')
        frac_le1_xs, _ = _series('sample_age_le1_fraction')
        if frac_new_xs or frac_le1_xs:
            ax2 = ax.twinx()
            _plot(ax2, 'sample_age_new_fraction', 'age=0', colors['orange'], style='--', linewidth=1.7)
            _plot(ax2, 'sample_age_le1_fraction', 'age<=1', colors['green'], style=':', linewidth=1.8)
            ax2.set_ylabel('batch fraction')
            ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax2.set_ylim([0, 1])
            ax2.spines['right'].set_alpha(0.18)
            lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[1, 0]
        for column, label, color, style in [
            ('replay_value_positive_fraction', 'Replay win-ish', colors['green'], '-'),
            ('replay_value_neutral_fraction', 'Replay neutral', colors['cyan'], '-'),
            ('replay_value_negative_fraction', 'Replay loss-ish', colors['red'], '-'),
            ('recent_value_neutral_fraction', 'Recent neutral', colors['slate'], '--'),
        ]:
            _plot(ax, column, label, color, style=style)
        _style_axis(ax, 'Replay Value Balance', 'fraction', percent=True)
        ax.set_ylim([0, 1])
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, loc='best', fontsize=8)

        ax = axes[1, 1]
        _plot(ax, 'policy_target_entropy_mean', 'Target entropy', colors['purple'], style='-')
        _plot(ax, 'train_policy_entropy', 'Model entropy', colors['black'], style=':')
        _plot(ax, 'policy_target_effective_moves', 'Effective moves', colors['orange'], style='-.')
        _style_axis(ax, 'Policy Distribution Shape')
        ax2 = ax.twinx()
        _plot(ax2, 'policy_target_top1_prob_mean', 'Target top-1', colors['green'], style='--')
        _plot(ax2, 'policy_target_top3_prob_mean', 'Target top-3', colors['cyan'], style='-.')
        ax2.set_ylabel('probability')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[2, 0]
        _, target_len_mean = _plot(ax, 'policy_target_len_mean', 'Target moves mean', colors['blue'], style='-')
        _, target_len_p90 = _plot(ax, 'policy_target_len_p90', 'Target moves p90', colors['cyan'], style='--')
        _style_axis(ax, 'Policy Target Width / Weight', 'target moves')
        _set_tight_ylim(ax, list(target_len_mean) + list(target_len_p90), min_pad=0.15)
        ax2 = ax.twinx()
        _, weight_mean = _plot(ax2, 'policy_weight_mean', 'Policy weight mean', colors['slate'], style=':', linewidth=2.0)
        _, weight_p10 = _plot(ax2, 'policy_weight_p10', 'Policy weight p10', colors['red'], style='-.', linewidth=2.0)
        _set_tight_ylim(ax2, list(weight_mean) + list(weight_p10), min_pad=0.03)
        ax2.set_ylabel('policy weight')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[2, 1]
        _, agreement = _plot(ax, 'mcts_prior_agreement_rate', 'MCTS kept network top', colors['purple'], style='-', linewidth=2.2)
        _, prior_visit = _plot(ax, 'mcts_prior_top_visit_prob_mean', 'Network top visit share', colors['blue'], style='--', linewidth=2.0)
        _, mcts_top_prior = _plot(ax, 'mcts_top_prior_prob_mean', 'MCTS top network prior', colors['slate'], style=':', linewidth=2.0)
        _style_axis(ax, 'Network vs MCTS Choice', 'agreement / probability', percent=True)
        _set_tight_ylim(ax, list(agreement) + list(prior_visit) + list(mcts_top_prior), min_pad=0.01)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(max(0.0, y0), min(1.0, y1))
        ax2 = ax.twinx()
        _, changed_rate = _plot(ax2, 'mcts_prior_changed_rate', 'Changed top move', colors['orange'], style='-', linewidth=2.4)
        _, discovery_rate = _plot(ax2, 'mcts_search_discovery_rate', 'Discovery boosted', colors['green'], style='--', linewidth=2.0)
        choice_rates = list(changed_rate) + list(discovery_rate)
        if choice_rates:
            ax2.set_ylim(0.0, min(1.0, max(0.02, max(choice_rates) * 1.3)))
        else:
            ax2.set_ylim([0, 0.05])
        ax2.set_ylabel('changed rate')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[2, 2]
        rate_values = []
        for column, label, color, style in [
            ('mcts_changed_to_higher_q_rate', 'Changed to higher Q', colors['green'], '--'),
            ('mcts_changed_to_lower_q_when_changed_rate', 'Changed to lower Q', colors['red'], ':'),
        ]:
            _, values = _plot(ax, column, label, color, style=style, linewidth=2.2)
            rate_values.extend(values)
        _style_axis(ax, 'Changed Move Q Quality', 'rate among changed moves', percent=True)
        if rate_values:
            ax.set_ylim(0.0, min(1.0, max(0.02, max(rate_values) * 1.25)))
        else:
            ax.set_ylim([0, 0.05])
        q_lines = []
        for _, values in [
            _series('mcts_changed_q_delta_p10'),
            _series('mcts_changed_q_delta_p50'),
            _series('mcts_changed_q_delta_p90'),
        ]:
            q_lines.extend(values)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = [], []
        if q_lines:
            ax2 = ax.twinx()
            _plot(ax2, 'mcts_changed_q_delta_p10', 'changed Qd p10', colors['slate'], style=':', linewidth=1.5)
            _plot(ax2, 'mcts_changed_q_delta_p50', 'changed Qd p50', colors['black'], style='--', linewidth=1.6)
            _plot(ax2, 'mcts_changed_q_delta_p90', 'changed Qd p90', colors['purple'], style='-.', linewidth=1.6)
            _set_tight_ylim(ax2, q_lines, min_pad=0.005, center_zero=True)
            ax2.set_ylabel('Q delta on changed top')
            ax2.spines['right'].set_alpha(0.18)
            lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[0, 2]
        _, value_mean = _plot(ax, 'selfplay_avg_game_value', 'Self-play value mean', colors['blue'], style='-')
        ax.axhline(0.0, color=colors['slate'], linestyle=':', linewidth=1.0, alpha=0.65)
        _style_axis(ax, 'Value Signal', 'mean value')
        _set_tight_ylim(ax, value_mean, min_pad=0.01, center_zero=True)
        ax2 = ax.twinx()
        _, selfplay_std = _plot(ax2, 'selfplay_value_std', 'Self-play value std', colors['cyan'], style='--')
        _, replay_std = _plot(ax2, 'replay_value_std', 'Replay value std', colors['green'], style='-')
        _, train_std = _plot(ax2, 'train_target_value_std', 'Train target std', colors['black'], style=':')
        _set_tight_ylim(ax2, list(selfplay_std) + list(replay_std) + list(train_std), min_pad=0.01)
        ax2.set_ylabel('value std')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[1, 2]
        ax2 = ax.twinx()
        _plot(ax, 'replay_size', 'Replay size', colors['black'])
        ax2_xs, ax2_ys = _series('replay_capacity')
        if ax2_xs:
            ax2.plot(ax2_xs, ax2_ys, color=colors['purple'], linestyle=':', linewidth=1.8, label='Capacity')
        added_xs, added_ys = _series('positions_added')
        if added_xs:
            ax2.plot(added_xs, added_ys, color=colors['blue'], linestyle='--', linewidth=1.8, label='Positions added')
        _style_axis(ax, 'Replay Size / Capacity', 'stored positions')
        ax2.set_ylabel('capacity / added')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[3, 0]
        _, avg_sims = _plot(ax, 'mcts_avg_sims', 'Avg sims used', colors['blue'], style='-', linewidth=2.1)
        _, avg_budget = _plot(ax, 'mcts_avg_budget', 'Avg budget', colors['slate'], style='--', linewidth=1.9)
        _, budget_p10 = _plot(ax, 'mcts_budget_p10', 'Budget bottom 10%', colors['cyan'], style=':', linewidth=1.8)
        _, budget_p90 = _plot(ax, 'mcts_budget_p90', 'Budget top 10%', colors['purple'], style='-.', linewidth=1.8)
        _, budget_min = _plot(ax, 'mcts_budget_min', 'Budget min', colors['red'], style=':', linewidth=1.35, alpha=0.72)
        _, budget_max = _plot(ax, 'mcts_budget_max', 'Budget max', colors['green'], style=':', linewidth=1.35, alpha=0.72)
        _, p10_sims = _plot(ax, 'mcts_p10_sims', 'p10 sims used', colors['cyan'], style=':', linewidth=2.0)
        _style_axis(ax, 'MCTS Search Budget', 'simulations')
        _set_tight_ylim(
            ax,
            (
                list(avg_sims)
                + list(avg_budget)
                + list(budget_p10)
                + list(budget_p90)
                + list(budget_min)
                + list(budget_max)
                + list(p10_sims)
            ),
            min_pad=4.0,
        )
        ax2 = ax.twinx()
        _, stop_rate = _plot(ax2, 'mcts_adaptive_stop_rate', 'Adaptive stop', colors['orange'], style='-.', linewidth=2.0)
        _, extra_rate = _plot(ax2, 'mcts_extra_budget_rate', 'Extra budget', colors['green'], style='--', linewidth=1.8)
        ax2.set_ylabel('rate')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        budget_rates = list(stop_rate) + list(extra_rate)
        if budget_rates:
            ax2.set_ylim(0.0, min(1.0, max(0.08, max(budget_rates) * 1.25)))
        else:
            ax2.set_ylim([0, 0.1])
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[3, 1]
        _, kl_values = _plot(ax, 'mcts_policy_kl_mean', 'Policy KL', colors['purple'], style='-', linewidth=2.0)
        _style_axis(ax, 'MCTS Policy Shift', 'KL')
        _set_tight_ylim(ax, kl_values, min_pad=0.005)
        ax2 = ax.twinx()
        _, explored_mass = _plot(ax2, 'mcts_explored_prior_mass_mean', 'Explored prior mass', colors['green'], style='--', linewidth=2.0)
        _set_tight_ylim(ax2, explored_mass, min_pad=0.02)
        y0, y1 = ax2.get_ylim()
        ax2.set_ylim(max(0.0, y0), min(1.0, y1))
        ax2.set_ylabel('explored prior mass')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[3, 2]
        sample_values = []
        for column, label, color, style in [
            ('train_value_samples_opening', 'Opening samples', colors['blue'], '--'),
            ('train_value_samples_middlegame', 'Middlegame samples', colors['orange'], '-.'),
            ('train_value_samples_endgame', 'Endgame samples', colors['green'], ':'),
        ]:
            _, values = _plot(ax, column, label, color, style=style, linewidth=2.0)
            sample_values.extend(values)
        _style_axis(ax, 'Value Phase Samples', 'samples')
        _set_tight_ylim(ax, sample_values, min_pad=8.0)
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, loc='best', fontsize=8)

        ax = axes[4, 0]
        mae_values = []
        for column, label, color, style in [
            ('train_value_mae', 'All positions', colors['black'], '-'),
            ('train_value_mae_opening', 'Opening', colors['blue'], '--'),
            ('train_value_mae_middlegame', 'Middlegame', colors['orange'], '-.'),
            ('train_value_mae_endgame', 'Endgame', colors['green'], ':'),
        ]:
            _, values = _plot(ax, column, label, color, style=style, linewidth=2.1)
            mae_values.extend(values)
        _style_axis(ax, 'Value MAE by Game Phase', 'MAE')
        _set_tight_ylim(ax, mae_values, min_pad=0.015)
        if ax.get_legend_handles_labels()[0]:
            _apply_sorted_legend(ax, loc='best', fontsize=8)

        ax = axes[4, 1]
        eval_values = []
        for column, label, color, style in [
            ('eval_mcts_score_rate', 'MCTS Q-on', colors['purple'], '-'),
            ('eval_mcts_qoff_score_rate', 'MCTS Q-off', colors['green'], '--'),
            ('eval_no_mcts_score_rate', 'No-MCTS', colors['slate'], ':'),
        ]:
            _, values = _plot(ax, column, label, color, style=style, marker='o', linewidth=2.2)
            eval_values.extend(values)
        _style_axis(ax, 'Eval Search Comparison', 'score rate', percent=True)
        if eval_values:
            ax.set_ylim(0.0, min(1.0, max(0.55, max(eval_values) * 1.18)))
        else:
            ax.set_ylim([0, 1])
        ax.axhline(0.5, color=colors['black'], linestyle=':', linewidth=1.0, alpha=0.5)
        ax2 = ax.twinx()
        _, gap_values = _plot(ax2, 'eval_mcts_q_ablation_gap', 'Q-on minus Q-off', colors['red'], style='-.', marker='s', linewidth=1.8)
        if gap_values:
            _set_tight_ylim(ax2, gap_values, min_pad=0.02, center_zero=True)
        ax2.set_ylabel('Q gap')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, loc='best', fontsize=8)

        ax = axes[4, 2]
        ax.axis('off')
        latest_lines = ["Latest Data Quality", ""]
        for column, label, fmt in [
            ('selfplay_draw_rate', 'Self-play draw', '{:.2%}'),
            ('replay_draw_fraction', 'Replay draw', '{:.2%}'),
            ('replay_vs_selfplay_draw_gap', 'Replay draw gap', '{:+.2%}'),
            ('sample_age_avg', 'Avg sample age', '{:.2f}'),
            ('sample_age_le1_fraction', 'Age <= 1', '{:.2%}'),
            ('sample_age_p90', 'p90 sample age', '{:.1f}'),
            ('policy_target_top1_prob_mean', 'Target top-1', '{:.2%}'),
            ('policy_target_effective_moves', 'Effective moves', '{:.2f}'),
            ('mcts_prior_changed_rate', 'MCTS changed', '{:.2%}'),
            ('mcts_search_discovery_rate', 'Discovery boost', '{:.2%}'),
            ('mcts_changed_q_delta_p50', 'Changed Qd p50', '{:+.4f}'),
            ('mcts_changed_to_higher_q_rate', 'Higher-Q changed', '{:.2%}'),
            ('mcts_changed_to_lower_q_when_changed_rate', 'Lower-Q changed', '{:.2%}'),
            ('train_value_mae_opening', 'MAE opening', '{:.4f}'),
            ('train_value_mae_middlegame', 'MAE middlegame', '{:.4f}'),
            ('train_value_mae_endgame', 'MAE endgame', '{:.4f}'),
            ('eval_mcts_score_rate', 'MCTS Q-on', '{:.2%}'),
            ('eval_mcts_qoff_score_rate', 'MCTS Q-off', '{:.2%}'),
            ('eval_mcts_q_ablation_gap', 'Q-on minus Q-off', '{:+.2%}'),
            ('selfplay_auto_draw_rate', 'Auto-draw claim', '{:.2%}'),
            ('mcts_adaptive_stop_rate', 'Adaptive stop', '{:.2%}'),
            ('mcts_budget_min', 'MCTS budget min', '{:.0f}'),
            ('mcts_budget_p10', 'MCTS budget b10', '{:.0f}'),
            ('mcts_avg_budget', 'MCTS budget avg', '{:.1f}'),
            ('mcts_budget_p90', 'MCTS budget t10', '{:.0f}'),
            ('mcts_budget_max', 'MCTS budget max', '{:.0f}'),
        ]:
            value = _latest(column)
            if isinstance(value, (int, float)):
                latest_lines.append(f"{label}: {fmt.format(value)}")
        ax.set_title('Summary', fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.text(
            0.04,
            0.52,
            "\n".join(latest_lines),
            fontsize=8.6,
            family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.55', fc='white', ec='#CBD5E1', alpha=0.95),
        )

        fig.subplots_adjust(left=0.055, right=0.945, bottom=0.045, top=0.945, hspace=0.55, wspace=0.42)
        fig.savefig(self.data_quality_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.18)
        plt.close(fig)

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
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
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
                    writer.writerows(metadata_rows)
                    writer.writerows(rows)
        except Exception:
            # CSV backfill is best-effort only.
            pass

    def record_estimated_elo_mode(self, iteration, estimated_elo, *, mode="nn", simulations=0, update_csv=True):
        """Backfill mode-specific Elo columns in RL CSV."""
        if self.mode != "rl" or estimated_elo is None:
            return
        try:
            iteration = int(iteration)
            elo_value = int(round(float(estimated_elo)))
        except (TypeError, ValueError):
            return
        mode = "mcts" if str(mode).lower() == "mcts" else "nn"
        pending = self._pending_rl_elo_by_iteration.setdefault(int(iteration), {})
        pending['estimated_elo_mcts' if mode == "mcts" else 'estimated_elo_nn'] = elo_value
        if mode == "mcts":
            try:
                pending['estimated_elo_mcts_simulations'] = int(simulations or 0)
            except (TypeError, ValueError):
                pending['estimated_elo_mcts_simulations'] = ''
        if update_csv and self.csv_path.exists():
            try:
                metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
                if not rows:
                    return
                header = list(rows[0])
                for col in ('estimated_elo_nn', 'estimated_elo_mcts', 'estimated_elo_mcts_simulations'):
                    if col not in header:
                        header.append(col)
                        for row in rows[1:]:
                            row.append('')
                target_col = 'estimated_elo_mcts' if mode == "mcts" else 'estimated_elo_nn'
                target_idx = header.index(target_col)
                sims_idx = header.index('estimated_elo_mcts_simulations')
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
                        if mode == "mcts":
                            try:
                                row[sims_idx] = str(int(simulations or 0))
                            except (TypeError, ValueError):
                                row[sims_idx] = ''
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(metadata_rows)
                    writer.writerow(header)
                    writer.writerows(rows[1:])
            except Exception:
                pass

    def record_rl_model_info_elo(
        self,
        iteration=0,
        *,
        elo_nn=None,
        elo_mcts=None,
        mcts_simulations=None,
    ):
        """Seed RL Elo plot from checkpoint metadata, usually the IL init model."""
        if self.mode != "rl" or self.csv_path is None:
            return

        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            iteration = 0

        def _clean_elo(value):
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return None

        elo_nn = _clean_elo(elo_nn)
        elo_mcts = _clean_elo(elo_mcts)
        try:
            mcts_simulations = int(mcts_simulations) if mcts_simulations not in (None, '') else None
        except (TypeError, ValueError):
            mcts_simulations = None
        if elo_nn is None and elo_mcts is None:
            return

        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
            if not rows:
                return
            header = list(rows[0])
            for col in ('estimated_elo_nn', 'estimated_elo_mcts', 'estimated_elo_mcts_simulations'):
                if col not in header:
                    header.append(col)
                    for row in rows[1:]:
                        row.append('')

            iter_col = 'iteration' if 'iteration' in header else 'epoch'
            iter_idx = header.index(iter_col)
            target_row = None
            for row in rows[1:]:
                while len(row) < len(header):
                    row.append('')
                try:
                    if int(float(row[iter_idx])) == iteration:
                        target_row = row
                        break
                except (TypeError, ValueError):
                    continue
            if target_row is None:
                target_row = [''] * len(header)
                target_row[iter_idx] = str(iteration)
                rows.insert(1, target_row)

            if elo_nn is not None:
                target_row[header.index('estimated_elo_nn')] = str(elo_nn)
            if elo_mcts is not None:
                target_row[header.index('estimated_elo_mcts')] = str(elo_mcts)
            if mcts_simulations is not None:
                target_row[header.index('estimated_elo_mcts_simulations')] = str(mcts_simulations)

            rows[0] = header
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerows(rows)
        except Exception:
            return

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
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
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
                writer.writerows(metadata_rows)
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
            rows = _read_csv_dict_rows(csv_path)
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
        swa_elo_value = float(self.swa_elo_info[1]) if self.swa_elo_info else None

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
                if (
                    swa_epoch is None
                    or int(ep) != swa_epoch
                    or swa_elo_value is None
                    or abs(float(val) - swa_elo_value) > 0.5
                )
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

        if self.best_final_elo_info:
            best_ep, best_elo = self.best_final_elo_info
            ax.plot(
                best_ep, best_elo,
                marker='D', markersize=9,
                color='#2563EB', markeredgecolor='#1E3A8A', markeredgewidth=1.2,
                linestyle='None',
                label=f'Best final ({int(round(best_elo))})',
                zorder=6,
            )
            ax.annotate(
                f'Best\n{int(round(best_elo))}',
                xy=(best_ep, best_elo),
                xytext=(8, -22),
                textcoords='offset points',
                fontsize=8,
                color='#1E3A8A',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#1E3A8A', lw=1.1,
                                shrinkB=6),
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
            if self.best_final_elo_info:
                best_ep = int(self.best_final_elo_info[0])
                if x_min <= best_ep and best_ep >= x_max - 1:
                    x_max = best_ep + max(2, int((x_max - x_min) * 0.08) + 1)
            ax.set_xlim(x_min, x_max)

        _apply_sorted_legend(ax, fontsize=8, loc='lower right')
    
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
                pending_elo = dict(self._pending_rl_elo_by_iteration.pop(int(iteration), {}) or {})
                estimated_elo_nn = kwargs.get('estimated_elo_nn', pending_elo.get('estimated_elo_nn', ''))
                estimated_elo_mcts = kwargs.get('estimated_elo_mcts', pending_elo.get('estimated_elo_mcts', ''))
                estimated_elo_mcts_simulations = kwargs.get(
                    'estimated_elo_mcts_simulations',
                    pending_elo.get('estimated_elo_mcts_simulations', ''),
                )
                mcts_no_mcts_gap = kwargs.get('mcts_no_mcts_gap', '')
                if mcts_no_mcts_gap == '':
                    try:
                        score = kwargs.get('score_rate', kwargs.get('win_rate', ''))
                        no_mcts_score = kwargs.get('no_mcts_score_rate', '')
                        if score not in ('', None) and no_mcts_score not in ('', None):
                            mcts_no_mcts_gap = float(score) - float(no_mcts_score)
                    except (TypeError, ValueError):
                        mcts_no_mcts_gap = ''
                row = [
                    iteration,
                    kwargs.get('avg_loss', ''),
                    kwargs.get('policy_loss', ''),
                    kwargs.get('value_loss', ''),
                    kwargs.get('learning_rate', kwargs.get('lr', '')),
                    kwargs.get('value_loss_weight', ''),
                    kwargs.get('mcts_q_value_scale', ''),
                    kwargs.get('mcts_q_selection_weight', ''),
                    kwargs.get('mcts_q_effective_weight', ''),
                    kwargs.get('mcts_q_value_trust', ''),
                    kwargs.get('value_guard_streak', ''),
                    mcts_no_mcts_gap,
                    kwargs.get('mcts_q_ablation_gap', ''),
                    kwargs.get('mcts_qoff_score_rate', ''),
                    kwargs.get('mcts_qoff_win_rate', ''),
                    kwargs.get('mcts_qoff_draw_rate', ''),
                    kwargs.get('mcts_qoff_loss_rate', ''),
                    kwargs.get('mcts_qoff_games', ''),
                    kwargs.get('score_rate', kwargs.get('win_rate', '')),
                    kwargs.get('buffer_size', ''),
                    kwargs.get('avg_game_length', ''),
                    kwargs.get('temperature', ''),
                    kwargs.get('beta', ''),
                    kwargs.get('true_win_rate', ''),
                    kwargs.get('eval_stage', ''),
                    kwargs.get('eval_games', ''),
                    kwargs.get('eval_wins', ''),
                    kwargs.get('eval_draws', ''),
                    kwargs.get('eval_losses', ''),
                    kwargs.get('eval_unresolved', ''),
                    kwargs.get('no_mcts_score_rate', ''),
                    kwargs.get('no_mcts_win_rate', kwargs.get('no_mcts_true_win_rate', '')),
                    kwargs.get('no_mcts_draw_rate', ''),
                    kwargs.get('no_mcts_loss_rate', ''),
                    kwargs.get('no_mcts_wins', ''),
                    kwargs.get('no_mcts_draws', ''),
                    kwargs.get('no_mcts_losses', ''),
                    kwargs.get('no_mcts_unresolved', ''),
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
                    train_metrics.get('value_mae_opening', '') if train_metrics else '',
                    train_metrics.get('value_mae_middlegame', '') if train_metrics else '',
                    train_metrics.get('value_mae_endgame', '') if train_metrics else '',
                    train_metrics.get('value_samples_opening', '') if train_metrics else '',
                    train_metrics.get('value_samples_middlegame', '') if train_metrics else '',
                    train_metrics.get('value_samples_endgame', '') if train_metrics else '',
                    kwargs.get('completed_draw_rate', ''),
                    kwargs.get('avg_game_value', ''),
                    kwargs.get('value_std', ''),
                    kwargs.get('policy_entropy', ''),
                    kwargs.get('value_pred_std', ''),
                    kwargs.get('selfplay_decisive_rate', ''),
                    kwargs.get('selfplay_auto_draw_rate', ''),
                    kwargs.get('selfplay_truncated_rate', ''),
                    kwargs.get('adaptive_temp_adjustment', ''),
                    kwargs.get('adaptive_temp_threshold', ''),
                    1 if bool(kwargs.get('rl_best_model', False)) else '',
                ]

                row.extend([
                    estimated_elo_nn,
                    estimated_elo_mcts,
                    estimated_elo_mcts_simulations,
                ])
                 
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
                if 'no_mcts_score_rate' in kwargs and kwargs['no_mcts_score_rate'] is not None:
                    self.no_mcts_score_rates.append((iteration, kwargs['no_mcts_score_rate']))
                no_mcts_win_rate = kwargs.get('no_mcts_win_rate', kwargs.get('no_mcts_true_win_rate'))
                if no_mcts_win_rate is not None:
                    self.no_mcts_true_win_rates.append((iteration, no_mcts_win_rate))
                if 'no_mcts_draw_rate' in kwargs and kwargs['no_mcts_draw_rate'] is not None:
                    self.no_mcts_draw_rates.append((iteration, kwargs['no_mcts_draw_rate']))
                if 'no_mcts_loss_rate' in kwargs and kwargs['no_mcts_loss_rate'] is not None:
                    self.no_mcts_loss_rates.append((iteration, kwargs['no_mcts_loss_rate']))
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

                # RL stores Elo only in mode-specific columns:
                # estimated_elo_nn, estimated_elo_mcts and estimated_elo_mcts_simulations.
            
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
        best_epoch = self.val_iterations[best_idx] if best_idx < len(self.val_iterations) else self.iterations[best_idx]

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
        swa_elo_value = float(self.swa_elo_info[1]) if self.swa_elo_info else None
        regular_elos = [(ep, v) for ep, v in self.estimated_elos
                        if (
                            swa_ep is None
                            or int(ep) != swa_ep
                            or swa_elo_value is None
                            or abs(float(v) - swa_elo_value) > 0.5
                        )]
        if self.best_final_elo_info:
            best['elo'] = float(self.best_final_elo_info[1])
        elif regular_elos:
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
        fig, axes = plt.subplots(4, 3, figsize=(19, 16))
        fig.patch.set_facecolor('#F7F8FA')

        if self.run_context_text:
            fig.suptitle(
                self._build_plot_suptitle("IL Training Progress"),
                fontsize=15,
                fontweight='bold',
                y=0.985,
            )
        else:
            fig.suptitle('IL Training Progress', fontsize=18, fontweight='bold', y=0.985)

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

        def _plot_line(ax, xs, ys, label, color, style='-', marker=None, linewidth=2.0, alpha=0.95):
            if not xs or not ys:
                return
            ys_to_plot = _smooth_values(list(ys))
            marker_to_plot = None if self.plot_smoothing_enabled else marker
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

        # Row 1: loss and LR
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

        # Row 2: policy and value quality
        ax = axes[1, 0]
        _plot_line(ax, self.iterations, self.train_policy_top1, 'Train Top-1', colors['train'])
        _plot_line(ax, self.iterations, self.train_policy_top3, 'Train Top-3', colors['train'], style='--', alpha=0.65)
        _plot_line(ax, val_epochs, self.val_policy_top1, 'Val Top-1', colors['val'], marker='o')
        _plot_line(ax, val_epochs, self.val_policy_top3, 'Val Top-3', colors['val'], style='--', marker='o', alpha=0.75)
        _mark_best(ax, val_epochs, self.val_policy_top1, mode='max', label='best top1')
        _style_axis(ax, 'Policy Accuracy', 'Accuracy', percent=True)
        ax.set_ylim([0, 1])
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[1, 1]
        _plot_line(ax, self.iterations, self.train_value_mae, 'Train MAE', colors['train'])
        _plot_line(ax, self.iterations, self.train_value_mae_weighted, 'Train Weighted', colors['train'], style='--', alpha=0.6)
        _plot_line(ax, val_epochs, self.val_value_mae, 'Val MAE', colors['val'], marker='o')
        _plot_line(ax, val_epochs, self.val_value_mae_weighted, 'Val Weighted', colors['val'], style='--', marker='o', alpha=0.75)
        _mark_best(ax, val_epochs, self.val_value_mae, mode='min', label='best mae')
        _style_axis(ax, 'Value Scalar MAE', 'MAE')
        _apply_sorted_legend(ax, fontsize=8)

        ax = axes[1, 2]
        _plot_line(ax, self.iterations, self.train_value_wdl_acc, 'Train WDL Acc', colors['train'])
        _plot_line(ax, val_epochs, self.val_value_wdl_acc, 'Val WDL Acc', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_value_wdl_acc, mode='max', label='best')
        _style_axis(ax, 'WDL Accuracy', 'Accuracy', percent=True)
        ax.set_ylim([0, 1])
        if self.train_value_wdl_acc or self.val_value_wdl_acc:
            _apply_sorted_legend(ax, fontsize=8)

        # Row 3: diagnostics and gaps
        ax = axes[2, 0]
        gap_xs, gap_ys = _val_gap(self.train_losses, self.val_losses)
        _plot_line(ax, gap_xs, gap_ys, 'Val - Train Loss', colors['gap'], marker='o')
        ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
        _style_axis(ax, 'Generalization Gap: Loss', 'Gap')
        if gap_xs:
            _apply_sorted_legend(ax, fontsize=8)

        ax = axes[2, 1]
        gap_xs, top1_gap = _val_gap(self.train_policy_top1, self.val_policy_top1)
        _, top3_gap = _val_gap(self.train_policy_top3, self.val_policy_top3)
        _plot_line(ax, gap_xs, top1_gap, 'Top-1 gap', colors['gap'], marker='o')
        _plot_line(ax, gap_xs, top3_gap, 'Top-3 gap', '#F59E0B', style='--', marker='o', alpha=0.85)
        ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
        _style_axis(ax, 'Generalization Gap: Policy', 'Val - Train', percent=True)
        if gap_xs:
            _apply_sorted_legend(ax, fontsize=8)

        ax = axes[2, 2]
        gap_xs, mae_gap = _val_gap(self.train_value_mae, self.val_value_mae)
        _, wdl_gap = _val_gap(self.train_value_wdl_acc, self.val_value_wdl_acc)
        _plot_line(ax, gap_xs, mae_gap, 'MAE gap', colors['gap'], marker='o')
        ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
        _style_axis(ax, 'Generalization Gap: Value', 'Val - Train MAE')
        ax2 = ax.twinx()
        _plot_line(ax2, gap_xs, wdl_gap, 'WDL acc gap', '#0891B2', style='--', marker='o', alpha=0.75)
        ax2.set_ylabel('Val - Train WDL Acc')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.tick_params(axis='y', labelcolor='#0891B2')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2:
            _apply_sorted_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        # Row 4: WDL CE, Elo, summary
        ax = axes[3, 0]
        _plot_line(ax, self.iterations, self.train_value_wdl_ce, 'Train WDL CE', colors['train'])
        _plot_line(ax, val_epochs, self.val_value_wdl_ce, 'Val WDL CE', colors['val'], marker='o')
        _mark_best(ax, val_epochs, self.val_value_wdl_ce, mode='min', label='best')
        _style_axis(ax, 'Value WDL Cross-Entropy', 'CE')
        if self.train_value_wdl_ce or self.val_value_wdl_ce:
            _apply_sorted_legend(ax, fontsize=8)

        ax = axes[3, 1]
        self._plot_il_elo_panel(ax)
        ax.set_facecolor('#FFFFFF')

        ax = axes[3, 2]
        self._plot_il_summary_panel(ax)

        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.045, top=0.925, hspace=0.45, wspace=0.28)
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        
        print(f"Plot saved to: {self.plot_path}")
    
    def _plot_rl(self):
        """Plot RL training progress"""
        rows = []
        try:
            rows = _read_csv_dict_rows(self.csv_path)
        except OSError:
            rows = []

        def _series(column):
            xs, ys = [], []
            for row in rows:
                try:
                    iteration = int(float(row.get('iteration', '')))
                    raw = row.get(column, '')
                    if raw is None or raw == '':
                        continue
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                xs.append(iteration)
                ys.append(value)
            return xs, ys

        def _latest(column):
            for row in reversed(rows):
                raw = row.get(column, '')
                if raw is None or raw == '':
                    continue
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return raw
            return None

        fig, axes = plt.subplots(4, 3, figsize=(19, 16))
        fig.patch.set_facecolor('#F7F8FA')
        if self.run_context_text:
            fig.suptitle(
                self._build_plot_suptitle("RL Training Progress"),
                fontsize=15,
                fontweight='bold',
                y=0.985,
            )
        else:
            fig.suptitle('RL Training Progress', fontsize=18, fontweight='bold', y=0.985)

        colors = {
            'total': '#2563EB',
            'policy': '#0F766E',
            'value': '#7C3AED',
            'eval': '#DC2626',
            'nomcts': '#EA580C',
            'draw': '#0891B2',
            'replay': '#475569',
            'mcts': '#9333EA',
            'muted': '#64748B',
            'good': '#16A34A',
            'lr': '#475569',
        }

        def _style_axis(ax, title, ylabel=None, percent=False):
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
            ax.set_xlabel('Iteration')
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.8)
            for spine in ax.spines.values():
                spine.set_alpha(0.18)
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

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
                prev = current if prev is None else alpha * current + (1.0 - alpha) * prev
                smoothed.append(prev)
            return smoothed

        def _rl_best_markers():
            markers = list(self.rl_best_model_markers or [])
            if markers:
                return markers
            for row in rows:
                try:
                    iteration = int(float(row.get('iteration', '')))
                except (TypeError, ValueError):
                    continue
                raw_marker = str(row.get('rl_best_model', '') or '').strip().lower()
                if raw_marker in {'1', 'true', 'yes', 'y'}:
                    markers.append((iteration, f"RL best {iteration}"))
            if markers:
                return markers
            for row in rows:
                try:
                    iteration = int(float(row.get('iteration', '')))
                    score = float(row.get('score_rate', ''))
                    true_win = float(row.get('true_win_rate', '0') or 0.0)
                except (TypeError, ValueError):
                    continue
                if score >= 0.55 and true_win >= 0.0:
                    markers.append((iteration, f"RL best {iteration}"))
            return markers

        def _split_on_best_boundaries(xs, ys):
            pairs = list(zip(list(xs), list(ys)))
            if len(pairs) <= 1:
                return [pairs] if pairs else []
            boundaries = []
            for marker_iter, _ in _rl_best_markers():
                try:
                    boundaries.append(float(marker_iter) + 0.5)
                except (TypeError, ValueError):
                    continue
            if not boundaries:
                return [pairs]
            boundaries.sort()
            segments = []
            current = [pairs[0]]
            for prev_pair, pair in zip(pairs, pairs[1:]):
                try:
                    prev_x = float(prev_pair[0])
                    x = float(pair[0])
                except (TypeError, ValueError):
                    current.append(pair)
                    continue
                if any(min(prev_x, x) < boundary < max(prev_x, x) for boundary in boundaries):
                    segments.append(current)
                    current = [pair]
                else:
                    current.append(pair)
            if current:
                segments.append(current)
            return segments

        def _plot_line(
            ax,
            xs,
            ys,
            label,
            color,
            style='-',
            marker=None,
            linewidth=2.0,
            alpha=0.95,
            smooth=None,
            split_on_best=False,
        ):
            if not xs or not ys:
                return
            use_smooth = self.plot_smoothing_enabled if smooth is None else bool(smooth)
            label_pending = True
            segments = _split_on_best_boundaries(xs, ys) if split_on_best else [list(zip(list(xs), list(ys)))]
            for segment in segments:
                if not segment:
                    continue
                seg_xs, seg_ys = zip(*segment)
                marker_to_plot = marker
                if use_smooth and len(seg_xs) >= self.plot_smoothing_min_points:
                    marker_to_plot = None
                ax.plot(
                    list(seg_xs),
                    _smooth_values(list(seg_ys)) if use_smooth else list(seg_ys),
                    linestyle=style,
                    marker=marker_to_plot,
                    color=color,
                    label=label if label_pending else None,
                    linewidth=linewidth,
                    markersize=4 if marker_to_plot else 0,
                    alpha=alpha,
                )
                label_pending = False

        def _draw_rl_best_boundaries(ax):
            markers = _rl_best_markers()
            if not markers:
                return
            y_min, y_max = ax.get_ylim()
            label_added = False
            for marker_iter, marker_text in markers:
                try:
                    x = float(marker_iter) + 0.5
                except (TypeError, ValueError):
                    continue
                ax.axvline(
                    x=x,
                    color='#6B7280',
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.70,
                    label='New best model' if not label_added else None,
                    zorder=1,
                )
                ax.text(
                    x,
                    y_max - (y_max - y_min) * 0.04,
                    str(marker_text or 'New best'),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=7,
                    color='#4B5563',
                    alpha=0.85,
                )
                label_added = True

        def _plot_column(
            ax,
            column,
            label,
            color,
            style='-',
            marker=None,
            linewidth=2.0,
            alpha=0.95,
            smooth=None,
            split_on_best=False,
        ):
            xs, ys = _series(column)
            _plot_line(
                ax,
                xs,
                ys,
                label,
                color,
                style=style,
                marker=marker,
                linewidth=linewidth,
                alpha=alpha,
                smooth=smooth,
                split_on_best=split_on_best,
            )
            return xs, ys

        def _safe_legend(ax, handles=None, labels=None, **kwargs):
            if handles is None or labels is None:
                handles, labels = ax.get_legend_handles_labels()
            if handles:
                _apply_sorted_legend(ax, handles, labels, **kwargs)

        def _mark_best(ax, xs, ys, mode='max', label='best'):
            if not xs or not ys:
                return
            x, y = (max if mode == 'max' else min)(zip(xs, ys), key=lambda item: item[1])
            ax.scatter([x], [y], s=42, color='#111827', zorder=5)
            ax.annotate(
                f"{label}: {y:.2%}" if abs(float(y)) <= 1.5 else f"{label}: {y:.4f}",
                xy=(x, y),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=8,
                color='#111827',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CBD5E1', alpha=0.9),
            )

        ax = axes[0, 0]
        _plot_line(ax, self.iterations, self.train_policy_losses, 'Policy', colors['policy'], linewidth=2.1)
        _plot_line(ax, self.iterations, self.train_value_losses, 'Value', colors['value'], style='--', linewidth=2.1, alpha=0.88)
        _style_axis(ax, 'Policy / Value Losses', 'Policy / Value loss')
        ax_total = ax.twinx()
        _plot_line(ax_total, self.iterations, self.train_losses, 'Total (weighted)', colors['total'], style=':', linewidth=2.2, alpha=0.9)
        ax_total.set_ylabel('Weighted total loss')
        ax_total.tick_params(axis='y', labelcolor=colors['total'])
        ax_total.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_total.get_legend_handles_labels()
        _safe_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[0, 1]
        _plot_line(ax, self.iterations, self.train_policy_top1, 'Top-1', colors['policy'])
        _plot_line(ax, self.iterations, self.train_policy_top3, 'Top-3', colors['policy'], style='--', alpha=0.65)
        _style_axis(ax, 'Policy Accuracy', 'Accuracy', percent=True)
        policy_values = [
            float(v)
            for v in list(self.train_policy_top1) + list(self.train_policy_top3)
            if v is not None
        ]
        if policy_values:
            y_min = max(0.0, min(policy_values))
            y_max = min(1.0, max(policy_values))
            pad = max(0.01, (y_max - y_min) * 0.18)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
        else:
            ax.set_ylim([0, 1])
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[0, 2]
        _plot_line(ax, self.iterations, self.train_value_mae, 'Value MAE', colors['value'])
        _plot_line(ax, self.iterations, self.train_value_wdl_acc, 'WDL Acc', colors['good'], style='--', alpha=0.82)
        _style_axis(ax, 'Value Quality', 'MAE / Accuracy')
        ax2 = ax.twinx()
        _plot_column(ax2, 'value_pred_std', 'Pred std', '#DB2777', style=':', linewidth=1.8)
        _plot_column(ax2, 'value_std', 'Target/game std', '#475569', style='-.', linewidth=1.6, alpha=0.8)
        ax2.set_ylabel('Std')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        _safe_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[1, 0]
        mcts_xs, mcts_ys = _plot_column(ax, 'score_rate', 'MCTS score', colors['eval'], marker='o', smooth=True, split_on_best=True)
        _plot_column(ax, 'true_win_rate', 'MCTS win', colors['eval'], style=':', marker='o', alpha=0.75, smooth=True, split_on_best=True)
        no_xs, no_ys = _plot_column(ax, 'no_mcts_score_rate', 'No-MCTS score', colors['nomcts'], style='--', marker='s', smooth=True, split_on_best=True)
        _plot_column(ax, 'no_mcts_win_rate', 'No-MCTS win', '#F59E0B', style=':', marker='s', alpha=0.75, smooth=True, split_on_best=True)
        ax.axhline(0.5, color=colors['muted'], linestyle=':', linewidth=1.2)
        _mark_best(ax, mcts_xs, mcts_ys, mode='max', label='best MCTS')
        _mark_best(ax, no_xs, no_ys, mode='max', label='best no-MCTS')
        _style_axis(ax, 'Current vs Best: MCTS / No-MCTS', 'Score rate', percent=True)
        ax.set_ylim([0, 1])
        _draw_rl_best_boundaries(ax)
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[1, 1]
        gap_xs, gap_ys = _series('mcts_no_mcts_gap')
        if not gap_xs:
            gap_xs, gap_ys = [], []
            for row in rows:
                try:
                    iteration = int(float(row.get('iteration', '')))
                    mcts = row.get('score_rate', '')
                    no_mcts = row.get('no_mcts_score_rate', '')
                    if mcts in ('', None) or no_mcts in ('', None):
                        continue
                    gap_xs.append(iteration)
                    gap_ys.append(float(mcts) - float(no_mcts))
                except (TypeError, ValueError):
                    continue
        _plot_line(ax, gap_xs, gap_ys, 'MCTS - No-MCTS', colors['eval'], marker='o')
        ax.axhline(0.0, color=colors['muted'], linestyle=':', linewidth=1.2)
        ax.axhline(0.05, color=colors['good'], linestyle=':', linewidth=1.0, alpha=0.55)
        _style_axis(ax, 'MCTS Gap vs No-MCTS', 'Score delta')
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[1, 2]
        anchor_xs, anchor_ys = _plot_column(ax, 'anchor_score_rate', 'vs Anchor/IL score', colors['mcts'], style='--', marker='s', alpha=0.92, smooth=True, split_on_best=True)
        anchor_win_xs, _ = _plot_column(ax, 'anchor_true_win_rate', 'vs Anchor true win', '#6D28D9', style=':', marker='s', alpha=0.75, smooth=True, split_on_best=True)
        _style_axis(ax, 'Current vs IL Anchor', 'Score rate', percent=True)
        ax.set_ylim([0, 1])
        if anchor_xs or anchor_win_xs:
            ax.axhline(0.5, color=colors['muted'], linestyle=':', linewidth=1.2)
            ax.axhline(0.55, color=colors['good'], linestyle=':', linewidth=1.0, alpha=0.55)
            _mark_best(ax, anchor_xs, anchor_ys, mode='max', label='best')
            _draw_rl_best_boundaries(ax)
            _safe_legend(ax, fontsize=8, loc='best')
        else:
            ax.text(
                0.5,
                0.5,
                'No IL-anchor eval yet',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=10,
                color=colors['muted'],
            )

        ax = axes[2, 0]
        nn_elo_xs, nn_elo_ys = _series('estimated_elo_nn')
        mcts_elo_xs, mcts_elo_ys = _series('estimated_elo_mcts')
        if not nn_elo_xs and not mcts_elo_xs:
            elo_xs, elo_ys = [], []
            _plot_line(ax, elo_xs, elo_ys, 'Estimated Elo', '#059669', marker='o', linewidth=2.2)
            plotted_elo_points = list(zip(elo_xs, elo_ys))
        else:
            _plot_line(ax, nn_elo_xs, nn_elo_ys, 'Raw NN Elo', '#2563EB', marker='o', linewidth=2.2)
            sims_xs, sims_ys = _series('estimated_elo_mcts_simulations')
            sims_by_iter = {int(x): int(round(float(y))) for x, y in zip(sims_xs, sims_ys)}
            mcts_label = 'MCTS Elo'
            if sims_by_iter:
                unique_sims = sorted(set(sims_by_iter.values()))
                if len(unique_sims) == 1:
                    mcts_label = f"MCTS Elo ({unique_sims[0]} sims)"
                else:
                    mcts_label = "MCTS Elo (sims marked)"
            _plot_line(ax, mcts_elo_xs, mcts_elo_ys, mcts_label, '#059669', marker='s', linewidth=2.2)
            for x, y in zip(mcts_elo_xs, mcts_elo_ys):
                sims = sims_by_iter.get(int(x))
                if sims:
                    ax.annotate(
                        f"{sims}",
                        xy=(x, y),
                        xytext=(5, -10),
                        textcoords='offset points',
                        fontsize=7,
                        color='#047857',
                    )
            plotted_elo_points = list(zip(nn_elo_xs, nn_elo_ys)) + list(zip(mcts_elo_xs, mcts_elo_ys))
        if plotted_elo_points:
            x, y = max(plotted_elo_points, key=lambda item: item[1])
            ax.scatter([x], [y], s=42, color='#111827', zorder=5)
            ax.annotate(
                f"best Elo: {int(round(float(y)))}",
                xy=(x, y),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=8,
                color='#111827',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#CBD5E1', alpha=0.9),
            )
        _style_axis(ax, 'Estimated Elo', 'Elo')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if not plotted_elo_points:
            ax.text(
                0.5,
                0.5,
                'No Elo estimates yet',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=10,
                color=colors['muted'],
            )
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[2, 1]
        _plot_column(ax, 'completed_draw_rate', 'Self-play draw', colors['draw'])
        _plot_column(ax, 'selfplay_decisive_rate', 'Self-play decisive', colors['good'], style='--')
        _plot_column(ax, 'selfplay_auto_draw_rate', 'Auto draw', '#0284C7', style=':', alpha=0.82)
        _style_axis(ax, 'Self-Play Outcomes', 'Rate', percent=True)
        ax.set_ylim([0, 1])
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[2, 2]
        _plot_column(ax, 'policy_entropy', 'Policy entropy', colors['policy'])
        _style_axis(ax, 'Training Signal Stability', 'Policy entropy')
        ax2 = ax.twinx()
        _, pred_std_values = _plot_column(ax2, 'value_pred_std', 'Value pred std', '#DB2777', style='--', alpha=0.95, linewidth=2.0)
        _, target_std_values = _plot_column(ax2, 'value_std', 'Target/game std', '#64748B', style=':', alpha=0.95, linewidth=2.1)
        std_values = list(pred_std_values) + list(target_std_values)
        if std_values:
            std_min = min(std_values)
            std_max = max(std_values)
            pad = max(0.01, (std_max - std_min) * 0.18)
            ax2.set_ylim(std_min - pad, std_max + pad)
        ax2.set_ylabel('Value std')
        ax2.tick_params(axis='y', labelcolor='#64748B')
        ax2.spines['right'].set_alpha(0.18)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        _safe_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[3, 0]
        if self.temperatures:
            temp_iters, temp_vals = zip(*self.temperatures)
            _plot_line(ax, temp_iters, temp_vals, 'Temperature', '#F97316')
        if self.adaptive_temp_adjustments:
            adj_iters, adj_vals = zip(*self.adaptive_temp_adjustments)
            _plot_line(ax, adj_iters, adj_vals, 'Adaptive adjustment', colors['eval'], style='--', alpha=0.82)
        _style_axis(ax, 'Temperature / Exploration', 'Value')
        lr_xs, lr_ys = _series('learning_rate')
        lines2, labels2 = [], []
        if lr_xs:
            ax_lr = ax.twinx()
            ax_lr.plot(
                lr_xs,
                lr_ys,
                color=colors['lr'],
                linestyle=':',
                marker='.',
                linewidth=1.9,
                markersize=4,
                label='LR',
                zorder=4,
            )
            ax_lr.set_ylabel('LR')
            lr_top = max(float(y) for y in lr_ys if y is not None)
            ax_lr.set_ylim(0.0, lr_top * 1.05 if lr_top > 0.0 else 1.0)
            ax_lr.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax_lr.tick_params(axis='y', labelcolor=colors['lr'])
            ax_lr.spines['right'].set_alpha(0.18)
            lines2, labels2 = ax_lr.get_legend_handles_labels()
        lines, labels = ax.get_legend_handles_labels()
        if lines or lines2:
            _safe_legend(ax, lines + lines2, labels + labels2, fontsize=8, loc='best')

        ax = axes[3, 1]
        _plot_column(ax, 'mcts_q_effective_weight', 'Effective Q', colors['mcts'], marker='o')
        _plot_column(ax, 'mcts_q_selection_weight', 'Q selection', '#0F766E', style='--', marker='s', alpha=0.86)
        _plot_column(ax, 'mcts_q_value_scale', 'Q trust add', '#64748B', style='-.', marker='.', alpha=0.72)
        _plot_column(ax, 'mcts_q_value_trust', 'Value trust', colors['policy'], style=':', marker='^', alpha=0.84)
        _plot_column(ax, 'value_loss_weight', 'Value loss weight', colors['value'], style=':', marker='s', alpha=0.84)
        _style_axis(ax, 'Search / Value Control', 'Weight')
        _safe_legend(ax, fontsize=8, loc='best')

        ax = axes[3, 2]
        ax.axis('off')
        if len(self.iterations) > 0:
            latest_mcts = _latest('score_rate')
            latest_no_mcts = _latest('no_mcts_score_rate')
            gap = (
                float(latest_mcts) - float(latest_no_mcts)
                if isinstance(latest_mcts, (int, float)) and isinstance(latest_no_mcts, (int, float))
                else None
            )
            ax.set_title('Summary', fontsize=11, fontweight='bold', loc='left', pad=8)

            def _fmt(value, mode='float'):
                if value is None:
                    return '-'
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return str(value)
                if mode == 'pct':
                    return f"{value:.2%}"
                if mode == 'signed_pct':
                    return f"{value:+.2%}"
                if mode == 'lr':
                    return f"{value:.2e}"
                if mode == 'temp':
                    return f"{value:.3f}"
                if mode == 'int':
                    return f"{int(round(value))}"
                return f"{value:.4f}"

            latest_loss = self.train_losses[-1] if self.train_losses else None
            latest_value_loss = self.train_value_losses[-1] if self.train_value_losses else None
            latest_value_mae = self.train_value_mae[-1] if self.train_value_mae else None
            latest_policy_top1 = self.train_policy_top1[-1] if self.train_policy_top1 else None
            latest_anchor = _latest('anchor_score_rate')
            latest_elo_nn = _latest('estimated_elo_nn')
            latest_elo_mcts = _latest('estimated_elo_mcts')
            latest_mcts_sims = _latest('estimated_elo_mcts_simulations')
            mcts_elo_label = 'MCTS Elo'
            if latest_mcts_sims is not None:
                mcts_elo_label = f"MCTS Elo ({int(round(float(latest_mcts_sims)))} sims)"
            summary_rows = [
                ['Training', 'Loss', _fmt(latest_loss)],
                ['', 'Value loss', _fmt(latest_value_loss)],
                ['', 'Value MAE', _fmt(latest_value_mae)],
                ['', 'Policy top-1', _fmt(latest_policy_top1, 'pct')],
                ['Eval', 'MCTS score', _fmt(latest_mcts, 'pct')],
                ['', 'No-MCTS score', _fmt(latest_no_mcts, 'pct')],
                ['', 'MCTS gap', _fmt(gap, 'signed_pct')],
                ['', 'Anchor score', _fmt(latest_anchor, 'pct')],
                ['', 'Raw NN Elo', _fmt(latest_elo_nn, 'int')],
                ['', mcts_elo_label, _fmt(latest_elo_mcts, 'int')],
                ['Control', 'Self-play draw', _fmt(_latest('completed_draw_rate'), 'pct')],
                ['', 'Temperature', _fmt(_latest('temperature'), 'temp')],
                ['', 'MCTS Q scale', _fmt(_latest('mcts_q_value_scale'), 'temp')],
                ['', 'Value trust', _fmt(_latest('mcts_q_value_trust'), 'temp')],
                ['', 'Value weight', _fmt(_latest('value_loss_weight'), 'temp')],
            ]
            table = ax.table(
                cellText=summary_rows,
                colLabels=['Group', 'Metric', 'Value'],
                colWidths=[0.25, 0.43, 0.32],
                bbox=[0.02, 0.04, 0.96, 0.86],
                cellLoc='left',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#CBD5E1')
                cell.set_linewidth(0.65)
                if row == 0:
                    cell.set_facecolor('#E2E8F0')
                    cell.set_text_props(weight='bold', color='#111827')
                    continue
                cell.set_facecolor('#FFFFFF' if row % 2 else '#F8FAFC')
                if col == 0 and summary_rows[row - 1][0]:
                    cell.set_text_props(weight='bold', color='#111827')
                elif col == 2:
                    cell.set_text_props(color='#111827')

        fig.subplots_adjust(left=0.055, right=0.955, bottom=0.045, top=0.935, hspace=0.48, wspace=0.42)
        fig.savefig(self.plot_path, dpi=150, bbox_inches='tight', pad_inches=0.18)
        plt.close(fig)
        
        print(f"Plot saved to: {self.plot_path}")
