"""RL-specific canonical CSV logging and plots."""

import csv
from datetime import datetime

from .rl_log_schema import (
    RL_DATA_QUALITY_COLUMNS, RL_LOG_SCHEMA_VERSION, RL_MAIN_COLUMNS,
    RL_PERFORMANCE_COLUMNS, append_csv_record, csv_row,
)
from .rl_plotting import render_rl_data_quality, render_rl_main, render_rl_performance
from ..shared.logger_common import (
    _read_csv_rows_preserving_metadata,
)


class RLLoggerMixin:
    def log_rl_performance(
        self,
        iteration,
        positions_per_sec=None,
        iteration_total_time=None,
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
                ('mcts_central_inference_server_queue_wait_time', 'central_server_queue_wait_ms_per_request'),
                ('mcts_central_inference_server_forward_time', 'central_server_forward_ms_per_request'),
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

        stage_values = {
            key: _float_or_none(stage_times.get(key))
            for key in ('setup', 'selfplay', 'replay', 'train', 'eval_log', 'checkpoint', 'gc')
        }
        measured_stages = {key: value for key, value in stage_values.items() if value is not None}
        bottleneck_stage = max(measured_stages, key=measured_stages.get) if measured_stages else ''
        record = {
            'iteration': int(iteration),
            'schema_version': RL_LOG_SCHEMA_VERSION,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'positions_per_sec': positions_per_sec,
            'iteration_total_time_s': iteration_total_time,
            'bottleneck_stage': bottleneck_stage,
            'avg_game_length': avg_game_length,
            'mcts_avg_batch_size': _value('average_batch_size'),
            'mcts_central_avg_batch_size': _value('central_average_batch_size'),
            'mcts_gpu_busy_proxy_pct': _value('gpu_utilization_pct'),
            'central_remote_wait_ms_per_request': _value('central_remote_wait_ms_per_request'),
            'central_server_queue_wait_ms_per_request': _value('central_server_queue_wait_ms_per_request'),
            'central_server_forward_ms_per_request': _value('central_server_forward_ms_per_request'),
            'central_server_total_ms_per_request': _value('central_server_total_ms_per_request'),
            'mcts_worker_nn_wait_ms_per_position': _value('worker_nn_wait_ms_per_position'),
            'mcts_worker_nn_wait_ms_per_batch': _value('worker_nn_wait_ms_per_batch'),
            'central_server_queue_wait_ms_per_position': _value('central_server_queue_wait_ms_per_position'),
            'central_server_forward_ms_per_position': _value('central_server_forward_ms_per_position'),
            'central_server_total_ms_per_position': _value('central_server_total_ms_per_position'),
            'mcts_search_many_time_s': _value('mcts_search_many_time'),
            'mcts_nn_inference_time_s': _value('mcts_nn_inference_time'),
            'mcts_nn_inference_calls': _value('mcts_nn_inference_calls'),
            'mcts_nn_inference_batch_items': _value('mcts_nn_inference_batch_items'),
            'mcts_avg_legal_moves_per_position': _value('average_legal_moves_per_position'),
            'result_queue_wait_ms': _value('queue_wait_time_ms'),
            'mcts_search_selection_time_s': _value('mcts_search_selection_time'),
            'mcts_search_backprop_time_s': _value('mcts_search_backprop_time'),
            'mcts_search_metadata_time_s': _value('mcts_search_metadata_time'),
            'mcts_batch_expand_eval_time_s': _value('mcts_batch_expand_eval_time'),
            'mcts_batch_expand_eval_calls': _value('mcts_batch_expand_eval_calls'),
            'mcts_board_to_tensor_time_s': _value('mcts_board_to_tensor_time'),
            'mcts_batch_expand_legal_moves_time_s': _value('mcts_batch_expand_legal_moves_time'),
            'mcts_batch_expand_tensor_pack_time_s': _value('mcts_batch_expand_tensor_pack_time'),
            'mcts_batch_expand_history_time_s': _value('mcts_batch_expand_history_time'),
            'mcts_batch_expand_input_pack_time_s': _value('mcts_batch_expand_input_pack_time'),
            'mcts_batch_expand_legal_index_pack_time_s': _value('mcts_batch_expand_legal_index_pack_time'),
            'mcts_batch_expand_cpu_policy_time_s': _value('mcts_batch_expand_cpu_policy_time'),
            'mcts_batch_expand_value_fanout_time_s': _value('mcts_batch_expand_value_fanout_time'),
            'mcts_policy_target_build_time_s': _value('mcts_policy_target_build_time'),
            'mcts_move_selection_time_s': _value('mcts_move_selection_time'),
            'mcts_adjudication_time_s': _value('mcts_adjudication_time'),
            'mcts_syzygy_time_s': _value('mcts_syzygy_time'),
        }
        record.update({f'stage_{key}_time_s': value for key, value in stage_values.items()})
        append_csv_record(self.performance_log_path, RL_PERFORMANCE_COLUMNS, record)
        latest_path = getattr(self, 'latest_training_profile_path', None)
        if latest_path is not None:
            append_csv_record(latest_path, RL_PERFORMANCE_COLUMNS, record)

    def plot_rl_performance(self):
        """Generate a performance plot from this run's RL details CSV."""
        if self.mode != "rl" or self.performance_log_path is None:
            return
        if not self.performance_log_path.exists():
            return

        render_rl_performance(self.performance_log_path, self.performance_plot_path)
        return

    def log_rl_data_quality(
        self,
        iteration,
        *,
        positions_added=None,
        replay_stats=None,
        selfplay_stats=None,
        train_policy_entropy=None,
    ):
        """Append detailed RL data-quality metrics for this run."""
        if self.mode != "rl" or self.data_quality_log_path is None:
            return
        replay_stats = dict(replay_stats or {})
        selfplay_stats = dict(selfplay_stats or {})

        def _value(mapping, key, default=''):
            value = mapping.get(key, default)
            return default if value is None else value

        def _opponent_bucket(label):
            label = str(label or 'current').strip().lower()
            return 'best' if label == 'best' else 'current'

        opponent_buckets = {
            label: {
                'planned_share': 0.0,
                'actual_share': 0.0,
                'games': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'score_rate': '',
            }
            for label in ('current', 'best')
        }
        opponent_debug = dict(_value(selfplay_stats, 'opponent_debug', {}) or {})
        opponent_source_counts = dict(_value(selfplay_stats, 'opponent_source_counts', {}) or {})
        opponent_results = dict(_value(selfplay_stats, 'opponent_results', {}) or {})
        for label, weight in dict(opponent_debug.get('source_weights', {}) or {}).items():
            bucket = opponent_buckets[_opponent_bucket(label)]
            bucket['planned_share'] += float(weight or 0.0)
        for label, count in opponent_source_counts.items():
            bucket = opponent_buckets[_opponent_bucket(label)]
            bucket['games'] += int(count or 0)
        for label, stats in opponent_results.items():
            stats = dict(stats or {})
            bucket = opponent_buckets[_opponent_bucket(label)]
            if label not in opponent_source_counts:
                bucket['games'] += int(stats.get('games', 0) or 0)
            bucket['wins'] += int(stats.get('wins', 0) or 0)
            bucket['draws'] += int(stats.get('draws', 0) or 0)
            bucket['losses'] += int(stats.get('losses', 0) or 0)
        opponent_games_total = sum(int(bucket['games']) for bucket in opponent_buckets.values())
        for bucket in opponent_buckets.values():
            games = int(bucket['games'])
            if opponent_games_total > 0:
                bucket['actual_share'] = float(games) / float(opponent_games_total)
            scored_games = int(bucket['wins']) + int(bucket['draws']) + int(bucket['losses'])
            if scored_games > 0:
                bucket['score_rate'] = (
                    float(bucket['wins']) + 0.5 * float(bucket['draws'])
                ) / float(scored_games)

        def _float(mapping, key):
            try:
                value = mapping.get(key)
                return None if value in (None, '') else float(value)
            except (TypeError, ValueError):
                return None

        def _ratio(numerator, denominator):
            try:
                denominator = float(denominator)
                return float(numerator) / denominator if denominator > 0.0 else ''
            except (TypeError, ValueError):
                return ''

        def _product(left, right):
            try:
                return float(left) * float(right)
            except (TypeError, ValueError):
                return ''

        replay_positive = _float(replay_stats, 'value_positive_fraction')
        replay_negative = _float(replay_stats, 'value_negative_fraction')
        replay_value_skew = (
            replay_positive - replay_negative
            if replay_positive is not None and replay_negative is not None
            else ''
        )
        target_entropy = _float(replay_stats, 'policy_target_entropy_mean')
        try:
            policy_entropy_ratio = float(train_policy_entropy) / max(float(target_entropy), 1e-8)
        except (TypeError, ValueError):
            policy_entropy_ratio = ''

        curriculum_drops = _float(selfplay_stats, 'curriculum_dropped_positions') or 0.0
        cap_drops = _float(selfplay_stats, 'cap_dropped_positions') or 0.0
        try:
            storage_candidates = float(positions_added) + curriculum_drops + cap_drops
            replay_storage_keep_rate = (
                float(positions_added) / storage_candidates if storage_candidates > 0.0 else ''
            )
        except (TypeError, ValueError):
            replay_storage_keep_rate = ''

        planned_share_total = sum(float(bucket['planned_share']) for bucket in opponent_buckets.values())
        opponent_mix_error = (
            0.5 * sum(
                abs(float(bucket['actual_share']) - float(bucket['planned_share']))
                for bucket in opponent_buckets.values()
            )
            if planned_share_total > 0.0 and opponent_games_total > 0
            else ''
        )
        mcts_avg_sims = _float(selfplay_stats, 'search_simulations_used_avg')
        mcts_avg_budget = _float(selfplay_stats, 'search_simulations_budget_avg')
        changed_rate = _float(selfplay_stats, 'mcts_prior_changed_rate')
        higher_q_rate = _float(selfplay_stats, 'mcts_changed_to_higher_q_rate')
        lower_q_rate = _float(selfplay_stats, 'mcts_changed_to_lower_q_when_changed_rate')
        opponent_adaptive_factors = dict(opponent_debug.get('adaptive_factors', {}) or {})

        record = {
            'iteration': int(iteration),
            'schema_version': RL_LOG_SCHEMA_VERSION,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'positions_added': positions_added,
            'replay_size': _value(replay_stats, 'size'),
            'replay_capacity': _value(replay_stats, 'capacity'),
            'replay_fill_rate': _value(replay_stats, 'fill_rate'),
            'train_batch_size': _value(replay_stats, 'train_batch_size'),
            'train_steps': _value(replay_stats, 'train_steps'),
            'train_selected_samples': _value(replay_stats, 'train_selected_samples'),
            'train_replay_coverage': _value(replay_stats, 'train_replay_coverage'),
            'replay_decisive_fraction': _value(replay_stats, 'decisive_fraction'),
            'replay_draw_fraction': _value(replay_stats, 'draw_fraction'),
            'replay_value_mean': _value(replay_stats, 'value_mean'),
            'replay_value_std': _value(replay_stats, 'value_std'),
            'replay_value_positive_fraction': _value(replay_stats, 'value_positive_fraction'),
            'replay_value_neutral_fraction': _value(replay_stats, 'value_neutral_fraction'),
            'replay_value_negative_fraction': _value(replay_stats, 'value_negative_fraction'),
            'replay_value_skew': replay_value_skew,
            'policy_weight_mean': _value(replay_stats, 'policy_weight_mean'),
            'policy_weight_p10': _value(replay_stats, 'policy_weight_p10'),
            'policy_weight_low_fraction': _value(replay_stats, 'policy_weight_low_fraction'),
            'value_weight_mean': _value(replay_stats, 'value_weight_mean'),
            'value_weight_p10': _value(replay_stats, 'value_weight_p10'),
            'value_weight_low_fraction': _value(replay_stats, 'value_weight_low_fraction'),
            'replay_source_learner_fraction': _value(replay_stats, 'source_learner_fraction'),
            'replay_source_frozen_best_fraction': _value(replay_stats, 'source_frozen_best_fraction'),
            'replay_policy_weight_learner_share': _value(replay_stats, 'source_learner_policy_weight_share'),
            'replay_policy_weight_frozen_best_share': _value(replay_stats, 'source_frozen_best_policy_weight_share'),
            'policy_target_len_mean': _value(replay_stats, 'policy_target_len_mean'),
            'policy_target_len_p90': _value(replay_stats, 'policy_target_len_p90'),
            'policy_target_entropy_mean': _value(replay_stats, 'policy_target_entropy_mean'),
            'policy_target_top1_prob_mean': _value(replay_stats, 'policy_target_top1_prob_mean'),
            'policy_target_top3_prob_mean': _value(replay_stats, 'policy_target_top3_prob_mean'),
            'policy_target_effective_moves': _value(replay_stats, 'policy_target_effective_moves'),
            'policy_entropy_ratio': policy_entropy_ratio,
            'sample_age_avg': _value(replay_stats, 'sample_age_avg'),
            'sample_age_p50': _value(replay_stats, 'sample_age_p50'),
            'sample_age_p90': _value(replay_stats, 'sample_age_p90'),
            'sample_age_new_fraction': _value(replay_stats, 'sample_age_new_fraction'),
            'sample_age_le1_fraction': _value(replay_stats, 'sample_age_le1_fraction'),
            'iterations_since_promotion': _value(replay_stats, 'iterations_since_promotion'),
            'selfplay_completed_games': _value(selfplay_stats, 'completed_games'),
            'selfplay_draw_rate': _value(selfplay_stats, 'completed_draw_rate'),
            'selfplay_decisive_rate': _value(selfplay_stats, 'decisive_rate'),
            'selfplay_auto_draw_rate': _value(selfplay_stats, 'auto_draw_rate'),
            'selfplay_truncated_rate': _value(selfplay_stats, 'truncated_rate'),
            'selfplay_avg_game_value': _value(selfplay_stats, 'avg_game_value'),
            'selfplay_value_std': _value(selfplay_stats, 'value_std'),
            'selfplay_replay_storage_keep_rate': replay_storage_keep_rate,
            'opponent_mix_error': opponent_mix_error,
            'opponent_promotion_transition_progress': opponent_adaptive_factors.get(
                '_promotion_transition_progress', ''
            ),
            'mcts_dirichlet_weight': _value(selfplay_stats, 'mcts_dirichlet_weight'),
            'mcts_avg_sims': mcts_avg_sims,
            'mcts_avg_budget': mcts_avg_budget,
            'mcts_budget_utilization': _ratio(mcts_avg_sims, mcts_avg_budget),
            'mcts_budget_p10': _value(selfplay_stats, 'search_simulations_budget_p10'),
            'mcts_budget_p90': _value(selfplay_stats, 'search_simulations_budget_p90'),
            'mcts_prior_agreement_rate': _value(selfplay_stats, 'mcts_prior_agreement_rate'),
            'mcts_prior_changed_rate': changed_rate,
            'mcts_changed_opening_rate': _value(selfplay_stats, 'mcts_changed_opening_rate'),
            'mcts_changed_middlegame_rate': _value(selfplay_stats, 'mcts_changed_middlegame_rate'),
            'mcts_changed_endgame_rate': _value(selfplay_stats, 'mcts_changed_endgame_rate'),
            'mcts_useful_change_rate': _product(changed_rate, higher_q_rate),
            'mcts_harmful_change_rate': _product(changed_rate, lower_q_rate),
            'mcts_changed_to_higher_q_rate': higher_q_rate,
            'mcts_changed_to_lower_q_when_changed_rate': lower_q_rate,
            'mcts_changed_q_delta_mean': _value(selfplay_stats, 'mcts_changed_q_delta_mean'),
            'mcts_changed_q_delta_p10': _value(selfplay_stats, 'mcts_changed_q_delta_p10'),
            'mcts_changed_q_delta_p90': _value(selfplay_stats, 'mcts_changed_q_delta_p90'),
            'mcts_policy_kl_mean': _value(selfplay_stats, 'mcts_policy_kl_mean'),
            'mcts_top_visit_prob_mean': _value(selfplay_stats, 'mcts_top_visit_prob_mean'),
            'mcts_visit_gap_mean': _value(selfplay_stats, 'mcts_visit_gap_mean'),
            'mcts_visit_entropy_mean': _value(selfplay_stats, 'mcts_visit_entropy_mean'),
            'mcts_good_target_rate': _value(selfplay_stats, 'mcts_good_target_rate'),
            'mcts_explored_prior_mass_mean': _value(selfplay_stats, 'mcts_explored_prior_mass_mean'),
            'mcts_visited_move_count_mean': _value(selfplay_stats, 'mcts_visited_move_count_mean'),
            'mcts_legal_move_count_mean': _value(selfplay_stats, 'mcts_legal_move_count_mean'),
            'mcts_visit_coverage_ratio_mean': _value(selfplay_stats, 'mcts_visit_coverage_ratio_mean'),
        }
        for label in ('current', 'best'):
            bucket = opponent_buckets[label]
            record.update({
                f'opponent_{label}_planned_share': bucket['planned_share'],
                f'opponent_{label}_actual_share': bucket['actual_share'],
                f'opponent_{label}_games': bucket['games'],
                f'opponent_{label}_score_rate': bucket['score_rate'],
            })
        append_csv_record(self.data_quality_log_path, RL_DATA_QUALITY_COLUMNS, record)

    def plot_rl_data_quality(self):
        """Generate a data-quality plot from this run's RL details CSV."""
        if self.mode != "rl" or self.data_quality_log_path is None:
            return
        if not self.data_quality_log_path.exists():
            return

        render_rl_data_quality(self.csv_path, self.data_quality_log_path, self.data_quality_plot_path)
        return

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

    def _plot_rl(self):
        """Plot RL training progress"""
        rendered = render_rl_main(
            self.csv_path,
            self.data_quality_log_path,
            self.performance_log_path,
            self.plot_path,
            self.run_context_lines,
        )
        return rendered

    def _log_rl(self, iteration, train_losses=None, val_losses=None,
                train_metrics=None, val_metrics=None, lr=None, estimated_elo=None, **kwargs):
        pending_elo = dict(self._pending_rl_elo_by_iteration.pop(int(iteration), {}) or {})
        estimated_elo_nn = kwargs.get('estimated_elo_nn', pending_elo.get('estimated_elo_nn', ''))
        estimated_elo_nn_se = kwargs.get('estimated_elo_nn_se', pending_elo.get('estimated_elo_nn_se', ''))
        estimated_elo_nn_ci95_low = kwargs.get(
            'estimated_elo_nn_ci95_low',
            pending_elo.get('estimated_elo_nn_ci95_low', ''),
        )
        estimated_elo_nn_ci95_high = kwargs.get(
            'estimated_elo_nn_ci95_high',
            pending_elo.get('estimated_elo_nn_ci95_high', ''),
        )
        estimated_elo_mcts = kwargs.get('estimated_elo_mcts', pending_elo.get('estimated_elo_mcts', ''))
        estimated_elo_mcts_se = kwargs.get('estimated_elo_mcts_se', pending_elo.get('estimated_elo_mcts_se', ''))
        estimated_elo_mcts_ci95_low = kwargs.get(
            'estimated_elo_mcts_ci95_low',
            pending_elo.get('estimated_elo_mcts_ci95_low', ''),
        )
        estimated_elo_mcts_ci95_high = kwargs.get(
            'estimated_elo_mcts_ci95_high',
            pending_elo.get('estimated_elo_mcts_ci95_high', ''),
        )
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
        def _count_games(*keys):
            values = [kwargs.get(key) for key in keys]
            if not any(value not in (None, '') for value in values):
                return ''
            try:
                return sum(int(value or 0) for value in values)
            except (TypeError, ValueError):
                return ''

        def _score_lower_bound(score, games):
            explicit = None
            try:
                if score in (None, '') or games in (None, '') or float(games) <= 0.0:
                    return ''
                score = max(0.0, min(1.0, float(score)))
                z_value = max(0.0, float(kwargs.get('eval_stat_gate_z', 1.28) or 1.28))
                standard_error = (score * (1.0 - score) / float(games)) ** 0.5
                explicit = max(0.0, score - z_value * standard_error)
            except (TypeError, ValueError):
                pass
            return '' if explicit is None else explicit

        score_rate = kwargs.get('score_rate', kwargs.get('win_rate', ''))
        eval_games = kwargs.get('eval_games', '')
        no_mcts_games = kwargs.get('no_mcts_games', _count_games(
            'no_mcts_wins', 'no_mcts_draws', 'no_mcts_losses', 'no_mcts_unresolved'
        ))
        anchor_games = kwargs.get('anchor_games', _count_games(
            'anchor_wins', 'anchor_draws', 'anchor_losses'
        ))
        anchor_no_mcts_games = kwargs.get('anchor_no_mcts_games', '')
        no_mcts_score = kwargs.get('no_mcts_score_rate', '')
        anchor_score = kwargs.get('anchor_score_rate', '')
        anchor_no_mcts_score = kwargs.get('anchor_no_mcts_score_rate', '')
        record = {
            'iteration': iteration,
            'schema_version': RL_LOG_SCHEMA_VERSION,
            'avg_loss': kwargs.get('avg_loss', ''),
            'policy_loss': kwargs.get('policy_loss', ''),
            'value_loss': kwargs.get('value_loss', ''),
            'learning_rate': kwargs.get('learning_rate', kwargs.get('lr', '')),
            'value_loss_weight': kwargs.get('value_loss_weight', ''),
            'mcts_q_selection_weight': kwargs.get('mcts_q_selection_weight', ''),
            'mcts_q_effective_weight': kwargs.get('mcts_q_effective_weight', ''),
            'temperature': kwargs.get('temperature', ''),
            'policy_top1_acc': train_metrics.get('policy_top1_acc', '') if train_metrics else '',
            'policy_top3_acc': train_metrics.get('policy_top3_acc', '') if train_metrics else '',
            'value_mae': train_metrics.get('value_mae', '') if train_metrics else '',
            'value_wdl_acc': train_metrics.get('value_wdl_acc', '') if train_metrics else '',
            'value_mae_opening': train_metrics.get('value_mae_opening', '') if train_metrics else '',
            'value_mae_middlegame': train_metrics.get('value_mae_middlegame', '') if train_metrics else '',
            'value_mae_endgame': train_metrics.get('value_mae_endgame', '') if train_metrics else '',
            'value_std_ratio_opening': train_metrics.get('value_std_ratio_opening', '') if train_metrics else '',
            'value_std_ratio_middlegame': train_metrics.get('value_std_ratio_middlegame', '') if train_metrics else '',
            'value_std_ratio_endgame': train_metrics.get('value_std_ratio_endgame', '') if train_metrics else '',
            'eval_stage': kwargs.get('eval_stage', ''),
            'eval_games': eval_games,
            'eval_wins': kwargs.get('eval_wins', ''),
            'eval_draws': kwargs.get('eval_draws', ''),
            'eval_losses': kwargs.get('eval_losses', ''),
            'eval_unresolved': kwargs.get('eval_unresolved', ''),
            'score_rate': score_rate,
            'true_win_rate': kwargs.get('true_win_rate', ''),
            'eval_score_lower_bound': kwargs.get(
                'eval_score_lower_bound', _score_lower_bound(score_rate, eval_games)
            ),
            'eval_score_rate_ema': kwargs.get('eval_score_rate_ema', ''),
            'eval_true_win_rate_ema': kwargs.get('eval_true_win_rate_ema', ''),
            'no_mcts_games': no_mcts_games,
            'no_mcts_score_rate': no_mcts_score,
            'no_mcts_win_rate': kwargs.get('no_mcts_win_rate', kwargs.get('no_mcts_true_win_rate', '')),
            'no_mcts_score_lower_bound': _score_lower_bound(no_mcts_score, no_mcts_games),
            'mcts_no_mcts_gap': mcts_no_mcts_gap,
            'anchor_games': anchor_games,
            'anchor_score_rate': anchor_score,
            'anchor_true_win_rate': kwargs.get('anchor_true_win_rate', ''),
            'anchor_score_lower_bound': kwargs.get(
                'anchor_score_lower_bound', _score_lower_bound(anchor_score, anchor_games)
            ),
            'anchor_no_mcts_games': anchor_no_mcts_games,
            'anchor_no_mcts_score_rate': anchor_no_mcts_score,
            'anchor_no_mcts_score_lower_bound': _score_lower_bound(
                anchor_no_mcts_score, anchor_no_mcts_games
            ),
            'anchor_mcts_no_mcts_gap': kwargs.get('anchor_mcts_no_mcts_gap', ''),
            'early_stop_streak': kwargs.get('early_stop_streak', ''),
            'promotion_candidate_streak': kwargs.get('promotion_candidate_streak', ''),
            'early_stop_reset_reason': kwargs.get('early_stop_reset_reason', ''),
            'rl_best_model': 1 if bool(kwargs.get('rl_best_model', False)) else '',
            'estimated_elo_nn': estimated_elo_nn,
            'estimated_elo_nn_se': estimated_elo_nn_se,
            'estimated_elo_nn_ci95_low': estimated_elo_nn_ci95_low,
            'estimated_elo_nn_ci95_high': estimated_elo_nn_ci95_high,
            'estimated_elo_mcts': estimated_elo_mcts,
            'estimated_elo_mcts_se': estimated_elo_mcts_se,
            'estimated_elo_mcts_ci95_low': estimated_elo_mcts_ci95_low,
            'estimated_elo_mcts_ci95_high': estimated_elo_mcts_ci95_high,
            'estimated_elo_mcts_simulations': estimated_elo_mcts_simulations,
        }
        row = csv_row(RL_MAIN_COLUMNS, record)

        # Kept only as a cheap guard for plot(); RL plots read canonical CSVs.
        self.iterations.append(iteration)
        with open(self.csv_path, 'a', newline='') as handle:
            csv.writer(handle).writerow(row)
