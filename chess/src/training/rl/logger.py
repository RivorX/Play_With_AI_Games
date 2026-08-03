"""RL-specific canonical CSV logging and plots."""

import csv
import re
from datetime import datetime
from pathlib import Path

from .log_schema import (
    RL_DATA_QUALITY_COLUMNS, RL_LOG_SCHEMA_VERSION, RL_MAIN_COLUMNS,
    RL_PERFORMANCE_COLUMNS, append_csv_record, csv_row,
)
from .plotting import render_rl_data_quality, render_rl_main, render_rl_performance
from src.common.logger_helpers import (
    _CSV_CONFIG_METADATA_KEY,
    _CSV_RESUME_METADATA_KEY,
    _CSV_RUN_SUMMARY_METADATA_KEY,
    _metadata_json_payload,
    _parse_mcts_elo_by_simulations,
    _read_csv_rows_preserving_metadata,
    _upsert_metadata_row,
    _upsert_mcts_elo_by_simulations,
)


class RLLoggerMixin:
    @staticmethod
    def _rl_history_rows(source_path, columns, completed_iteration):
        """Read and schema-align historical RL rows up to a resume boundary."""
        source_path = Path(source_path)
        metadata_rows, raw_rows = _read_csv_rows_preserving_metadata(source_path)
        if not raw_rows:
            return metadata_rows, []
        header = list(raw_rows[0])
        if "iteration" not in header:
            return metadata_rows, []
        history = []
        for values in raw_rows[1:]:
            values = list(values)
            values.extend([""] * max(0, len(header) - len(values)))
            row = dict(zip(header, values[:len(header)]))
            try:
                iteration = int(float(row.get("iteration", "")))
            except (TypeError, ValueError):
                continue
            if iteration <= completed_iteration:
                history.append(csv_row(columns, row))
        return metadata_rows, history

    @classmethod
    def _rl_history_source_chain(cls, source_csv_path, completed_iteration):
        """Recover contiguous pre-metadata resume segments from older RL logs."""
        selected = Path(source_csv_path)
        match = re.match(
            r"^(?P<prefix>.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(?:_\d+)?$",
            selected.stem,
        )
        if match is None:
            return [selected]
        prefix = match.group("prefix")
        chain = [selected]

        while True:
            _, current_rows = cls._rl_history_rows(
                chain[0],
                RL_MAIN_COLUMNS,
                completed_iteration,
            )
            current_iterations = [
                int(float(row[0]))
                for row in current_rows
                if row and str(row[0]).strip()
            ]
            if not current_iterations or min(current_iterations) <= 1:
                break
            required_previous = min(current_iterations) - 1
            candidates = []
            for candidate in selected.parent.glob(f"{prefix}_*.csv"):
                if candidate in chain or candidate.name.endswith(
                    ("_performance.csv", "_data_quality.csv")
                ):
                    continue
                _, candidate_rows = cls._rl_history_rows(
                    candidate,
                    RL_MAIN_COLUMNS,
                    required_previous,
                )
                candidate_iterations = [
                    int(float(row[0]))
                    for row in candidate_rows
                    if row and str(row[0]).strip()
                ]
                if not candidate_iterations or max(candidate_iterations) != required_previous:
                    continue
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    mtime = 0.0
                candidates.append((mtime, candidate))
            if not candidates:
                break
            candidates.sort(reverse=True)
            chain.insert(0, candidates[0][1])
        return chain

    def import_rl_history_from_csv(self, source_csv_path, completed_iteration):
        """Seed all three RL logs from the run selected for a true resume.

        Historical rows are mapped by column name, so a continuation remains
        readable after a telemetry schema upgrade. The new files keep the
        current config metadata and record every resume boundary.
        """
        if self.mode != "rl" or source_csv_path is None:
            return False
        source_csv_path = Path(source_csv_path)
        if not source_csv_path.exists() or source_csv_path.resolve() == self.csv_path.resolve():
            return False
        try:
            completed_iteration = int(completed_iteration)
        except (TypeError, ValueError):
            return False
        if completed_iteration < 1:
            return False

        try:
            source_chain = self._rl_history_source_chain(
                source_csv_path,
                completed_iteration,
            )
            source_metadata, _ = self._rl_history_rows(
                source_csv_path,
                RL_MAIN_COLUMNS,
                completed_iteration,
            )
            main_by_iteration = {}
            inferred_resume_boundaries = []
            for source_index, history_path in enumerate(source_chain):
                _, history_rows = self._rl_history_rows(
                    history_path,
                    RL_MAIN_COLUMNS,
                    completed_iteration,
                )
                segment_iterations = []
                for row in history_rows:
                    iteration = int(float(row[0]))
                    main_by_iteration[iteration] = row
                    segment_iterations.append(iteration)
                if source_index > 0 and segment_iterations:
                    first_iteration = min(segment_iterations)
                    if first_iteration > 1:
                        inferred_resume_boundaries.append(first_iteration - 1)
            main_history = [
                main_by_iteration[iteration]
                for iteration in sorted(main_by_iteration)
            ]
        except (OSError, csv.Error):
            return False
        if not main_history:
            return False

        try:
            current_metadata, _ = _read_csv_rows_preserving_metadata(self.csv_path)
        except (OSError, csv.Error):
            current_metadata = []
        current_config = _metadata_json_payload(current_metadata, _CSV_CONFIG_METADATA_KEY)
        if current_config is not None:
            current_metadata = _upsert_metadata_row(
                current_metadata,
                _CSV_CONFIG_METADATA_KEY,
                current_config,
            )

        sidecars = (
            ("performance", self.performance_log_path, RL_PERFORMANCE_COLUMNS),
            ("data_quality", self.data_quality_log_path, RL_DATA_QUALITY_COLUMNS),
        )

        try:
            with open(self.csv_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(current_metadata)
                writer.writerow(RL_MAIN_COLUMNS)
                writer.writerows(main_history)
            for suffix, destination_path, columns in sidecars:
                history_by_iteration = {}
                for history_main_path in source_chain:
                    history_source = history_main_path.with_name(
                        f"{history_main_path.stem}_{suffix}.csv"
                    )
                    if not history_source.exists():
                        continue
                    _, segment_history = self._rl_history_rows(
                        history_source,
                        columns,
                        completed_iteration,
                    )
                    for row in segment_history:
                        history_by_iteration[int(float(row[0]))] = row
                history = [
                    history_by_iteration[iteration]
                    for iteration in sorted(history_by_iteration)
                ]
                with open(destination_path, "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    writer.writerows(history)
        except (OSError, csv.Error):
            return False

        self.resume_source_csv_path = str(source_csv_path)
        self.resume_source_summary = _metadata_json_payload(
            source_metadata,
            _CSV_RUN_SUMMARY_METADATA_KEY,
        )
        self.resume_history_epoch = completed_iteration
        self.iterations = sorted(main_by_iteration)
        resume_payload = _metadata_json_payload(
            source_metadata,
            _CSV_RESUME_METADATA_KEY,
        ) or {}
        self.resume_markers = []
        for marker in resume_payload.get("markers") or []:
            if not isinstance(marker, dict):
                continue
            raw_completed = marker.get(
                "completed_iteration",
                marker.get("completed_epoch"),
            )
            try:
                completed = int(raw_completed)
            except (TypeError, ValueError):
                continue
            self.add_resume_marker(completed, marker.get("label"))
        for completed in inferred_resume_boundaries:
            self.add_resume_marker(completed)
        self.add_resume_marker(completed_iteration)
        return True

    def log_rl_performance(
        self,
        iteration,
        replay_positions_per_sec=None,
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
                ('mcts_central_inference_request_put_time', 'central_request_submit_ms_per_request'),
                ('mcts_central_inference_server_queue_wait_time', 'central_server_queue_wait_ms_per_request'),
                (
                    'mcts_central_inference_server_descriptor_queue_wait_time',
                    'central_descriptor_queue_wait_ms_per_request',
                ),
                (
                    'mcts_central_inference_server_batch_coalesce_wait_time',
                    'central_batch_coalesce_wait_ms_per_request',
                ),
                (
                    'mcts_central_inference_server_pipeline_wait_time',
                    'central_server_pipeline_wait_ms_per_request',
                ),
                (
                    'mcts_central_inference_server_output_finalize_wait_time',
                    'central_server_output_finalize_wait_ms_per_request',
                ),
                ('mcts_central_inference_server_concat_time', 'central_server_concat_ms_per_request'),
                ('mcts_central_inference_server_h2d_time', 'central_server_h2d_ms_per_request'),
                ('mcts_central_inference_server_forward_time', 'central_server_forward_ms_per_request'),
                ('mcts_central_inference_server_d2h_time', 'central_server_d2h_ms_per_request'),
                ('mcts_central_inference_server_total_time', 'central_server_total_ms_per_request'),
            ]:
                raw_value = _float_or_none(profile.get(raw_key))
                if raw_value is not None:
                    profile[avg_key] = 1000.0 * float(raw_value) / float(central_requests)
        if central_requests and central_requests > 0:
            shared_requests = _int_or_none(
                profile.get('mcts_central_inference_shared_requests')
            ) or 0
            cache_queries = _int_or_none(
                profile.get('mcts_central_inference_cache_queries')
            ) or 0
            cache_bypassed = _int_or_none(
                profile.get('mcts_central_inference_cache_bypassed_positions')
            ) or 0
            cache_hits = _int_or_none(
                profile.get('mcts_central_inference_cache_hits')
            ) or 0
            dedup_hits = _int_or_none(
                profile.get('mcts_central_inference_dedup_hits')
            ) or 0
            cache_scope = cache_queries + cache_bypassed
            profile.setdefault(
                'central_shared_memory_request_fraction',
                float(shared_requests) / float(central_requests),
            )
            profile.setdefault(
                'central_shared_memory_mib_avoided',
                float(profile.get('mcts_central_inference_shared_bytes_avoided', 0) or 0)
                / float(1024 ** 2),
            )
            profile.setdefault(
                'central_shared_slot_wait_ms_per_request',
                1000.0 * float(profile.get('mcts_central_inference_shared_slot_wait_time', 0.0) or 0.0)
                / float(max(1, shared_requests)),
            )
            profile.setdefault(
                'central_cache_hit_rate',
                float(cache_hits) / float(cache_queries) if cache_queries > 0 else 0.0,
            )
            profile.setdefault(
                'central_dedup_hit_rate',
                float(dedup_hits) / float(cache_queries) if cache_queries > 0 else 0.0,
            )
            profile.setdefault(
                'central_nn_saved_rate',
                float(cache_hits + dedup_hits) / float(cache_queries)
                if cache_queries > 0 else 0.0,
            )
            profile.setdefault(
                'central_cache_active_fraction',
                float(cache_queries) / float(cache_scope) if cache_scope > 0 else 0.0,
            )
            profile.setdefault(
                'central_nn_saved_overall_rate',
                float(cache_hits + dedup_hits) / float(cache_scope)
                if cache_scope > 0 else 0.0,
            )
            profile.setdefault(
                'central_server_cache_lookup_ms_per_request',
                1000.0 * float(profile.get('mcts_central_inference_server_cache_lookup_time', 0.0) or 0.0)
                / float(central_requests),
            )
            profile.setdefault(
                'central_server_staging_copy_ms_per_request',
                1000.0 * float(profile.get('mcts_central_inference_server_staging_copy_time', 0.0) or 0.0)
                / float(central_requests),
            )
            profile.setdefault(
                'central_gpu_batch_fill',
                float(profile.get('mcts_central_inference_gpu_batch_fill_sum', 0.0) or 0.0)
                / float(central_requests),
            )
            profile.setdefault(
                'central_batch_target_avg',
                float(profile.get('mcts_central_inference_server_batch_target_sum', 0) or 0)
                / float(central_requests),
            )
            profile.setdefault(
                'central_output_pipeline_rate',
                float(profile.get('mcts_central_inference_server_output_pipeline_requests', 0) or 0)
                / float(central_requests),
            )
            profile.setdefault(
                'central_auto_batch_calibrated_rate',
                float(profile.get('mcts_central_inference_server_auto_calibrated_requests', 0) or 0)
                / float(central_requests),
            )

        search_time = _float_or_none(profile.get('mcts_search_many_time'))
        nn_time = _float_or_none(profile.get('mcts_nn_inference_time'))
        if search_time and search_time > 0.0 and nn_time is not None:
            profile['worker_nn_wait_share_pct'] = max(
                0.0,
                min(100.0, 100.0 * float(nn_time) / float(search_time)),
            )

        worker_wait_batch_ms = _float_or_none(profile.get('inference_time_per_batch_ms'))
        worker_wait_pos_ms = _float_or_none(profile.get('inference_time_per_position_ms'))
        if worker_wait_batch_ms is not None:
            profile.setdefault('worker_nn_wait_ms_per_batch', worker_wait_batch_ms)
        if worker_wait_pos_ms is not None:
            profile.setdefault('worker_nn_wait_ms_per_position', worker_wait_pos_ms)

        worker_avg_batch = _float_or_none(profile.get('average_batch_size'))
        for src_key, dst_key in [
            ('central_server_queue_wait_ms_per_request', 'central_server_queue_wait_ms_per_position'),
            ('central_server_forward_ms_per_request', 'central_server_forward_ms_per_position'),
            ('central_server_total_ms_per_request', 'central_server_total_ms_per_position'),
        ]:
            value = _float_or_none(profile.get(src_key))
            if value is not None and worker_avg_batch and worker_avg_batch > 0.0:
                profile.setdefault(dst_key, float(value) / float(worker_avg_batch))

        remote_ms = _float_or_none(profile.get('central_remote_wait_ms_per_request')) or 0.0
        queue_ms = _float_or_none(profile.get('central_server_queue_wait_ms_per_request')) or 0.0
        server_total_ms = _float_or_none(profile.get('central_server_total_ms_per_request')) or 0.0
        pipeline_wait_ms = _float_or_none(
            profile.get('central_server_pipeline_wait_ms_per_request')
        ) or 0.0
        server_stage_ms = {
            key: _float_or_none(profile.get(key)) or 0.0
            for key in (
                'central_server_concat_ms_per_request',
                'central_server_h2d_ms_per_request',
                'central_server_forward_ms_per_request',
                'central_server_d2h_ms_per_request',
                'central_server_cache_lookup_ms_per_request',
                'central_server_staging_copy_ms_per_request',
            )
        }
        server_other_ms = max(
            0.0,
            server_total_ms - pipeline_wait_ms - sum(server_stage_ms.values()),
        )
        worker_ipc_ms = max(0.0, remote_ms - queue_ms - server_total_ms)

        def _value(key, default=''):
            value = profile.get(key, default)
            return default if value is None else value

        stage_values = {
            key: _float_or_none(stage_times.get(key))
            for key in (
                'setup', 'selfplay', 'replay', 'train',
                'regular_eval', 'promotion_eval', 'reanalyse', 'elo_eval', 'log',
                'checkpoint', 'gc',
            )
        }
        measured_stages = {key: value for key, value in stage_values.items() if value is not None}
        bottleneck_stage = max(measured_stages, key=measured_stages.get) if measured_stages else ''
        record = {
            'iteration': int(iteration),
            'schema_version': RL_LOG_SCHEMA_VERSION,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'selfplay_worker_count': _value('worker_count'),
            'selfplay_games_per_worker_mean': _value('games_per_worker_mean'),
            'selfplay_games_per_worker_min': _value('games_per_worker_min'),
            'selfplay_games_per_worker_max': _value('games_per_worker_max'),
            'selfplay_games_per_worker_std': _value('games_per_worker_std'),
            'selfplay_worker_finish_spread_s': _value('worker_finish_spread_s'),
            'selfplay_tail_10pct_time_s': _value('tail_10pct_time_s'),
            'selfplay_global_game_stream_enabled': _value('global_game_stream_enabled'),
            'replay_positions_per_sec': replay_positions_per_sec,
            'played_positions_per_sec': _value('played_positions_per_sec'),
            'mcts_simulations_per_sec': _value('mcts_simulations_per_sec'),
            'mcts_effective_simulations_per_sec': _value(
                'mcts_effective_simulations_per_sec'
            ),
            'mcts_nn_evaluations_per_sec': _value('mcts_nn_evaluations_per_sec'),
            'mcts_selection_node_traversals_per_sec': _value(
                'mcts_selection_node_traversals_per_sec'
            ),
            'iteration_total_time_s': iteration_total_time,
            'bottleneck_stage': bottleneck_stage,
            'avg_game_length': avg_game_length,
            'mcts_avg_batch_size': _value('average_batch_size'),
            'mcts_central_avg_batch_size': _value('central_average_batch_size'),
            # This is worker wall-time spent waiting for NN replies, not NVML GPU use.
            'mcts_worker_nn_wait_share_pct': _value(
                'worker_nn_wait_share_pct',
                _value('gpu_utilization_pct'),
            ),
            'central_remote_wait_ms_per_request': _value('central_remote_wait_ms_per_request'),
            'central_request_submit_ms_per_request': _value(
                'central_request_submit_ms_per_request'
            ),
            'central_server_queue_wait_ms_per_request': _value('central_server_queue_wait_ms_per_request'),
            'central_descriptor_queue_wait_ms_per_request': _value(
                'central_descriptor_queue_wait_ms_per_request'
            ),
            'central_batch_coalesce_wait_ms_per_request': _value(
                'central_batch_coalesce_wait_ms_per_request'
            ),
            'central_server_pipeline_wait_ms_per_request': _value(
                'central_server_pipeline_wait_ms_per_request'
            ),
            'central_server_output_finalize_wait_ms_per_request': _value(
                'central_server_output_finalize_wait_ms_per_request'
            ),
            'central_server_concat_ms_per_request': _value('central_server_concat_ms_per_request'),
            'central_server_h2d_ms_per_request': _value('central_server_h2d_ms_per_request'),
            'central_server_forward_ms_per_request': _value('central_server_forward_ms_per_request'),
            'central_server_d2h_ms_per_request': _value('central_server_d2h_ms_per_request'),
            'central_server_other_ms_per_request': server_other_ms,
            'central_worker_ipc_ms_per_request': worker_ipc_ms,
            'central_server_total_ms_per_request': _value('central_server_total_ms_per_request'),
            'central_shared_memory_request_fraction': _value('central_shared_memory_request_fraction'),
            'central_shared_memory_mib_avoided': _value('central_shared_memory_mib_avoided'),
            'central_shared_slot_wait_ms_per_request': _value('central_shared_slot_wait_ms_per_request'),
            'central_cache_hit_rate': _value('central_cache_hit_rate'),
            'central_dedup_hit_rate': _value('central_dedup_hit_rate'),
            'central_nn_saved_rate': _value('central_nn_saved_rate'),
            'central_cache_active_fraction': _value('central_cache_active_fraction'),
            'central_nn_saved_overall_rate': _value('central_nn_saved_overall_rate'),
            'central_cache_suspensions': _value(
                'mcts_central_inference_cache_suspensions'
            ),
            'central_cache_reactivations': _value(
                'mcts_central_inference_cache_reactivations'
            ),
            'central_nn_evaluated_positions': _value(
                'mcts_central_inference_nn_evaluated_positions'
            ),
            'central_server_cache_lookup_ms_per_request': _value(
                'central_server_cache_lookup_ms_per_request'
            ),
            'central_server_staging_copy_ms_per_request': _value(
                'central_server_staging_copy_ms_per_request'
            ),
            'central_gpu_batch_fill': _value('central_gpu_batch_fill'),
            'central_batch_target_avg': _value('central_batch_target_avg'),
            'central_output_pipeline_rate': _value('central_output_pipeline_rate'),
            'central_auto_batch_calibrated_rate': _value(
                'central_auto_batch_calibrated_rate'
            ),
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
            'mcts_search_root_setup_time_s': _value('mcts_search_root_setup_time'),
            'mcts_search_selection_time_s': _value('mcts_search_selection_time'),
            'mcts_search_backprop_time_s': _value('mcts_search_backprop_time'),
            'mcts_native_tree_expand_sync_time_s': _value(
                'mcts_native_tree_expand_sync_time'
            ),
            'mcts_native_tree_backup_time_s': _value('mcts_native_tree_backup_time'),
            'mcts_native_tree_import_time_s': _value('mcts_native_tree_import_time'),
            'mcts_native_tree_sync_time_s': _value('mcts_native_tree_sync_time'),
            'mcts_native_tree_selection_batches': _value(
                'mcts_native_tree_selection_batches'
            ),
            'mcts_native_tree_backup_batches': _value(
                'mcts_native_tree_backup_batches'
            ),
            'mcts_native_tree_selected_leaves': _value(
                'mcts_native_tree_selected_leaves'
            ),
            'mcts_python_tree_selected_leaves': _value(
                'mcts_python_tree_selected_leaves'
            ),
            'mcts_native_tree_selection_rate': _value(
                'mcts_native_tree_selection_rate'
            ),
            'mcts_native_tree_import_nodes': _value('mcts_native_tree_import_nodes'),
            'mcts_native_tree_import_edges': _value('mcts_native_tree_import_edges'),
            'mcts_native_tree_fallbacks': _value('mcts_native_tree_fallbacks'),
            'mcts_search_metadata_time_s': _value('mcts_search_metadata_time'),
            'mcts_native_root_selection_calls': _value('mcts_native_root_selection_calls'),
            'mcts_python_root_selection_calls': _value('mcts_python_root_selection_calls'),
            'mcts_native_root_selection_rate': _value('mcts_native_root_selection_rate'),
            'mcts_native_completed_q_calls': _value('mcts_native_completed_q_calls'),
            'mcts_python_completed_q_calls': _value('mcts_python_completed_q_calls'),
            'mcts_native_completed_q_rate': _value('mcts_native_completed_q_rate'),
            'mcts_batch_expand_dedup_time_s': _value('mcts_batch_expand_dedup_terminal_time'),
            'mcts_batch_expand_eval_time_s': _value('mcts_batch_expand_eval_time'),
            'mcts_batch_expand_eval_calls': _value('mcts_batch_expand_eval_calls'),
            'mcts_board_materialize_time_s': _value('mcts_board_materialize_time'),
            'mcts_terminal_checks_time_s': _value('mcts_terminal_checks_time'),
            'mcts_board_to_tensor_time_s': _value('mcts_board_to_tensor_time'),
            'mcts_batch_expand_legal_moves_time_s': _value('mcts_batch_expand_legal_moves_time'),
            'mcts_batch_expand_move_index_time_s': _value('mcts_batch_expand_move_index_time'),
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

        render_rl_performance(
            self.performance_log_path,
            self.performance_plot_path,
            self.data_quality_log_path,
            self.csv_path,
        )
        return

    def add_final_elo_runtime(self, iteration, elapsed_seconds):
        """Charge the post-loop final Elo evaluation to the final iteration.

        The final Stockfish check runs after the normal iteration row has
        already been written. Patch that row instead of appending a duplicate
        iteration, and keep the aggregate iteration time/bottleneck coherent.
        """
        if self.mode != "rl":
            return
        try:
            iteration = int(iteration)
            elapsed_seconds = max(0.0, float(elapsed_seconds))
        except (TypeError, ValueError):
            return
        if iteration <= 0 or elapsed_seconds <= 0.0:
            return

        paths = [self.performance_log_path, getattr(self, 'latest_training_profile_path', None)]
        seen_paths = set()
        for path in paths:
            if path is None or path in seen_paths or not path.exists():
                continue
            seen_paths.add(path)
            try:
                metadata_rows, rows = _read_csv_rows_preserving_metadata(path)
                if not rows:
                    continue
                header = list(rows[0])
                required = (
                    'stage_elo_eval_time_s',
                    'iteration_total_time_s',
                    'bottleneck_stage',
                )
                for column in required:
                    if column not in header:
                        header.append(column)
                        for row in rows[1:]:
                            row.append('')

                iter_column = 'iteration' if 'iteration' in header else 'epoch'
                iter_idx = header.index(iter_column)
                target_row = None
                for row in reversed(rows[1:]):
                    while len(row) < len(header):
                        row.append('')
                    try:
                        if int(float(row[iter_idx])) == iteration:
                            target_row = row
                            break
                    except (TypeError, ValueError):
                        continue
                if target_row is None:
                    continue

                def _number_at(column):
                    try:
                        return float(target_row[header.index(column)] or 0.0)
                    except (TypeError, ValueError, IndexError):
                        return 0.0

                elo_idx = header.index('stage_elo_eval_time_s')
                total_idx = header.index('iteration_total_time_s')
                target_row[elo_idx] = str(_number_at('stage_elo_eval_time_s') + elapsed_seconds)
                target_row[total_idx] = str(_number_at('iteration_total_time_s') + elapsed_seconds)

                stage_columns = [
                    column
                    for column in header
                    if column.startswith('stage_') and column.endswith('_time_s')
                ]
                if stage_columns:
                    target_row[header.index('bottleneck_stage')] = max(
                        stage_columns,
                        key=_number_at,
                    )[len('stage_'):-len('_time_s')]

                rows[0] = header
                with open(path, 'w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerows(metadata_rows)
                    writer.writerows(rows)
            except Exception:
                # Runtime backfill is diagnostic only; never fail a completed run.
                continue

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
        mcts_search_samples = _float(selfplay_stats, 'search_samples')
        hard_start_games = _float(selfplay_stats, 'hard_start_games')
        changed_rate = _float(selfplay_stats, 'mcts_prior_changed_rate')
        higher_q_rate = _float(selfplay_stats, 'mcts_changed_to_higher_q_rate')
        lower_q_rate = _float(selfplay_stats, 'mcts_changed_to_lower_q_when_changed_rate')
        opponent_adaptive_factors = dict(opponent_debug.get('adaptive_factors', {}) or {})

        record = {
            'iteration': int(iteration),
            'schema_version': RL_LOG_SCHEMA_VERSION,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'positions_added': positions_added,
            'replay_candidate_positions': _value(selfplay_stats, 'replay_candidate_positions'),
            'played_positions': _value(selfplay_stats, 'played_positions'),
            'replay_cap_dropped_positions': _value(selfplay_stats, 'cap_dropped_positions'),
            'replay_overwritten_positions': _value(
                replay_stats,
                'overwritten_positions_iteration',
            ),
            'replay_resize_dropped_positions': _value(
                replay_stats,
                'resize_dropped_positions_iteration',
            ),
            'replay_evicted_positions_total': _value(replay_stats, 'evicted_positions'),
            'replay_size': _value(replay_stats, 'size'),
            'replay_capacity': _value(replay_stats, 'capacity'),
            'replay_fill_rate': _value(replay_stats, 'fill_rate'),
            'train_batch_size': _value(replay_stats, 'train_batch_size'),
            'train_steps': _value(replay_stats, 'train_steps'),
            'train_selected_samples': _value(replay_stats, 'train_selected_samples'),
            'train_replay_coverage': _value(replay_stats, 'train_replay_coverage'),
            'train_replay_passes': _value(replay_stats, 'train_replay_passes'),
            'champion_replay_size': _value(replay_stats, 'champion_replay_size'),
            'champion_replay_capacity': _value(replay_stats, 'champion_replay_capacity'),
            'champion_replay_added': _value(replay_stats, 'champion_replay_added'),
            'champion_replay_selected_samples': _value(
                replay_stats, 'champion_replay_selected_samples'
            ),
            'champion_replay_selected_fraction': _value(
                replay_stats, 'champion_replay_selected_fraction'
            ),
            'champion_replay_target_fraction': _value(
                replay_stats, 'champion_replay_target_fraction'
            ),
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
            'root_q_coverage': _value(replay_stats, 'root_q_coverage'),
            'root_q_mean': _value(replay_stats, 'root_q_mean'),
            'root_q_std': _value(replay_stats, 'root_q_std'),
            'best_q_coverage': _value(replay_stats, 'best_q_coverage'),
            'best_q_mean': _value(replay_stats, 'best_q_mean'),
            'played_q_coverage': _value(replay_stats, 'played_q_coverage'),
            'orig_q_coverage': _value(replay_stats, 'orig_q_coverage'),
            'policy_kld_coverage': _value(replay_stats, 'policy_kld_coverage'),
            'policy_kld_mean': _value(replay_stats, 'policy_kld_mean'),
            'deblunder_value_fraction': _value(replay_stats, 'deblunder_value_fraction'),
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
            'replay_policy_correction_fraction': _value(
                replay_stats, 'policy_correction_fraction'
            ),
            'replay_policy_correction_q_delta_mean': _value(
                replay_stats, 'policy_correction_q_delta_mean'
            ),
            'replay_policy_correction_weight_share': _value(
                replay_stats, 'policy_correction_weight_share'
            ),
            'train_policy_correction_sharpened_fraction': _value(
                replay_stats, 'train_policy_correction_sharpened_fraction'
            ),
            'train_policy_correction_target_top1_before': _value(
                replay_stats, 'train_policy_correction_target_top1_before'
            ),
            'train_policy_correction_target_top1_after': _value(
                replay_stats, 'train_policy_correction_target_top1_after'
            ),
            'correction_audit_rows': _value(replay_stats, 'correction_audit_rows'),
            'correction_audit_top1_before': _value(replay_stats, 'correction_audit_top1_before'),
            'correction_audit_top1_after': _value(replay_stats, 'correction_audit_top1_after'),
            'correction_audit_top1_gain': _value(replay_stats, 'correction_audit_top1_gain'),
            'correction_audit_rank_before': _value(replay_stats, 'correction_audit_rank_before'),
            'correction_audit_rank_after': _value(replay_stats, 'correction_audit_rank_after'),
            'correction_audit_rank_gain': _value(replay_stats, 'correction_audit_rank_gain'),
            'correction_audit_target_probability_before': _value(
                replay_stats, 'correction_audit_target_probability_before'
            ),
            'correction_audit_target_probability_after': _value(
                replay_stats, 'correction_audit_target_probability_after'
            ),
            'correction_audit_target_probability_gain': _value(
                replay_stats, 'correction_audit_target_probability_gain'
            ),
            'correction_audit_logit_margin_before': _value(
                replay_stats, 'correction_audit_logit_margin_before'
            ),
            'correction_audit_logit_margin_after': _value(
                replay_stats, 'correction_audit_logit_margin_after'
            ),
            'correction_audit_logit_margin_gain': _value(
                replay_stats, 'correction_audit_logit_margin_gain'
            ),
            'correction_audit_retention': _value(replay_stats, 'correction_audit_retention'),
            'correction_audit_retention_rows': _value(
                replay_stats, 'correction_audit_retention_rows'
            ),
            'reanalyse_selected': _value(replay_stats, 'reanalyse_selected'),
            'reanalyse_updated': _value(replay_stats, 'reanalyse_updated'),
            'reanalyse_correction_fraction': _value(
                replay_stats, 'reanalyse_correction_fraction'
            ),
            'sample_age_avg': _value(replay_stats, 'sample_age_avg'),
            'sample_age_p50': _value(replay_stats, 'sample_age_p50'),
            'sample_age_p90': _value(replay_stats, 'sample_age_p90'),
            'sample_age_new_fraction': _value(replay_stats, 'sample_age_new_fraction'),
            'sample_age_le1_fraction': _value(replay_stats, 'sample_age_le1_fraction'),
            'iterations_since_promotion': _value(replay_stats, 'iterations_since_promotion'),
            'recent_sample_age_avg': _value(replay_stats, 'recent_sample_age_avg'),
            'recent_sample_age_p50': _value(replay_stats, 'recent_sample_age_p50'),
            'recent_sample_age_p90': _value(replay_stats, 'recent_sample_age_p90'),
            'recent_sample_age_new_fraction': _value(
                replay_stats, 'recent_sample_age_new_fraction'
            ),
            'recent_sample_age_le1_fraction': _value(
                replay_stats, 'recent_sample_age_le1_fraction'
            ),
            'champion_sample_age_avg': _value(replay_stats, 'champion_sample_age_avg'),
            'champion_sample_age_p50': _value(replay_stats, 'champion_sample_age_p50'),
            'champion_sample_age_p90': _value(replay_stats, 'champion_sample_age_p90'),
            'selfplay_completed_games': _value(selfplay_stats, 'completed_games'),
            'selfplay_draw_rate': _value(selfplay_stats, 'completed_draw_rate'),
            'selfplay_decisive_rate': _value(selfplay_stats, 'decisive_rate'),
            'selfplay_auto_draw_rate': _value(selfplay_stats, 'auto_draw_rate'),
            'selfplay_truncated_rate': _value(selfplay_stats, 'truncated_rate'),
            'selfplay_avg_game_value': _value(selfplay_stats, 'avg_game_value'),
            'selfplay_value_std': _value(selfplay_stats, 'value_std'),
            'selfplay_replay_storage_keep_rate': replay_storage_keep_rate,
            'selfplay_hard_start_games': hard_start_games if hard_start_games is not None else '',
            'selfplay_hard_start_fraction': _ratio(
                hard_start_games,
                (_float(selfplay_stats, 'completed_games') or 0.0)
                + (_float(selfplay_stats, 'truncated_games') or 0.0),
            ),
            'opponent_mix_error': opponent_mix_error,
            'opponent_promotion_transition_progress': opponent_adaptive_factors.get(
                '_promotion_transition_progress', ''
            ),
            'mcts_avg_sims': mcts_avg_sims,
            'mcts_fresh_sims': _value(
                selfplay_stats, 'search_fresh_simulations_used_avg'
            ),
            'mcts_inherited_visit_credit': _value(
                selfplay_stats, 'search_inherited_visit_credit_avg'
            ),
            'mcts_avg_budget': mcts_avg_budget,
            'mcts_budget_utilization': _ratio(mcts_avg_sims, mcts_avg_budget),
            'mcts_budget_target': _value(selfplay_stats, 'search_simulations_budget_target'),
            'mcts_budget_min': _value(selfplay_stats, 'search_simulations_budget_min'),
            'mcts_budget_p10': _value(selfplay_stats, 'search_simulations_budget_p10'),
            'mcts_budget_p50': _value(selfplay_stats, 'search_simulations_budget_p50'),
            'mcts_budget_p90': _value(selfplay_stats, 'search_simulations_budget_p90'),
            'mcts_budget_max': _value(selfplay_stats, 'search_simulations_budget_max'),
            'mcts_difficulty_mean': _value(selfplay_stats, 'search_difficulty_mean'),
            'mcts_difficulty_budget_correlation': _value(
                selfplay_stats, 'search_difficulty_budget_correlation'
            ),
            'mcts_tree_reuse_hit_rate': _value(selfplay_stats, 'tree_reuse_hit_rate'),
            'mcts_tree_inherited_visits_avg': _value(
                selfplay_stats, 'tree_inherited_visits_avg'
            ),
            'mcts_tree_reuse_credit_samples': _value(
                selfplay_stats, 'tree_reuse_credit_samples'
            ),
            'mcts_tree_reuse_credit_fraction': _ratio(
                _float(selfplay_stats, 'tree_reuse_credit_samples'),
                mcts_search_samples,
            ),
            'mcts_tree_reuse_quality_avg': _value(
                selfplay_stats, 'tree_reuse_quality_avg'
            ),
            'mcts_tree_reuse_candidate_coverage_avg': _value(
                selfplay_stats, 'tree_reuse_candidate_coverage_avg'
            ),
            'mcts_tree_reuse_visited_prior_mass_avg': _value(
                selfplay_stats, 'tree_reuse_visited_prior_mass_avg'
            ),
            'mcts_tree_reuse_fresh_floor_avg': _value(
                selfplay_stats, 'tree_reuse_fresh_floor_avg'
            ),
            'mcts_tree_reuse_scout_stability_avg': _value(
                selfplay_stats, 'tree_reuse_scout_stability_avg'
            ),
            'mcts_tree_reuse_scout_extra_credit_avg': _value(
                selfplay_stats, 'tree_reuse_scout_extra_credit_avg'
            ),
            'mcts_tree_reuse_scout_reduction_rate': _value(
                selfplay_stats, 'tree_reuse_scout_reduction_rate'
            ),
            'mcts_shared_tree_search_fraction': _value(
                selfplay_stats, 'shared_tree_search_fraction'
            ),
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
        mcts_by_simulations=None,
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
            if not _parse_mcts_elo_by_simulations(mcts_by_simulations):
                return

        serialized_by_simulations = ""
        for simulations, entry in sorted(
            _parse_mcts_elo_by_simulations(mcts_by_simulations).items()
        ):
            serialized_by_simulations = _upsert_mcts_elo_by_simulations(
                serialized_by_simulations,
                simulations,
                entry["elo"],
                std_error=entry.get("se"),
                ci95=entry.get("ci95"),
            )
        if elo_mcts is not None and mcts_simulations is not None:
            serialized_by_simulations = _upsert_mcts_elo_by_simulations(
                serialized_by_simulations,
                mcts_simulations,
                elo_mcts,
            )

        try:
            metadata_rows, rows = _read_csv_rows_preserving_metadata(self.csv_path)
            if not rows:
                return
            header = list(rows[0])
            for col in (
                'estimated_elo_nn',
                'estimated_elo_mcts',
                'estimated_elo_mcts_simulations',
                'estimated_elo_mcts_by_simulations',
            ):
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
            if serialized_by_simulations:
                target_row[header.index('estimated_elo_mcts_by_simulations')] = serialized_by_simulations

            rows[0] = header
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(metadata_rows)
                writer.writerows(rows)
        except Exception:
            return

    def _plot_rl(self):
        """Plot RL training progress"""
        context_lines = list(self.run_context_lines or [])
        if not context_lines and self.run_context_text:
            context_lines.append(self.run_context_text)
        rl_cfg = dict((self.config_snapshot or {}).get('reinforcement_learning', {}) or {})
        target_sims = int(rl_cfg.get('mcts_simulations', 0) or 0)
        if target_sims > 0:
            if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False)):
                min_sims = int(rl_cfg.get('mcts_dynamic_budget_min', target_sims) or target_sims)
                max_sims = int(round(
                    target_sims * float(rl_cfg.get('mcts_dynamic_budget_max_multiplier', 1.0) or 1.0)
                ))
                context_lines.append(
                    f"MCTS simulations: target {target_sims} per move, dynamic {min_sims}-{max_sims}"
                )
            else:
                context_lines.append(f"MCTS simulations: {target_sims} per move")
        rendered = render_rl_main(
            self.csv_path,
            self.data_quality_log_path,
            self.performance_log_path,
            self.plot_path,
            context_lines,
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
        estimated_elo_mcts_by_simulations = kwargs.get(
            'estimated_elo_mcts_by_simulations',
            pending_elo.get('estimated_elo_mcts_by_simulations', ''),
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
            'value_primary_loss': train_metrics.get('value_primary_loss', '') if train_metrics else '',
            'value_scalar_aux_loss': train_metrics.get('value_scalar_aux_loss', '') if train_metrics else '',
            'moves_left_loss': train_metrics.get('moves_left_loss', '') if train_metrics else '',
            'search_q_loss': train_metrics.get('search_q_loss', '') if train_metrics else '',
            'search_q_coverage': train_metrics.get('search_q_coverage', '') if train_metrics else '',
            'search_q_mae': train_metrics.get('search_q_mae', '') if train_metrics else '',
            'learning_rate': kwargs.get('learning_rate', kwargs.get('lr', '')),
            'value_loss_weight': kwargs.get('value_loss_weight', ''),
            'temperature': kwargs.get('temperature', ''),
            'policy_top1_acc': train_metrics.get('policy_top1_acc', '') if train_metrics else '',
            'policy_top3_acc': train_metrics.get('policy_top3_acc', '') if train_metrics else '',
            'policy_correction_loss': train_metrics.get('policy_correction_loss', '') if train_metrics else '',
            'policy_correction_rank_loss': train_metrics.get('policy_correction_rank_loss', '') if train_metrics else '',
            'policy_correction_rank_weight': train_metrics.get('policy_correction_rank_weight', '') if train_metrics else '',
            'policy_correction_top1_acc': train_metrics.get('policy_correction_top1_acc', '') if train_metrics else '',
            'policy_correction_fraction': train_metrics.get('policy_correction_fraction', '') if train_metrics else '',
            'policy_correction_weight_share': train_metrics.get('policy_correction_weight_share', '') if train_metrics else '',
            'correction_audit_rows': train_metrics.get('correction_audit_rows', '') if train_metrics else '',
            'correction_audit_top1_before': train_metrics.get('correction_audit_top1_before', '') if train_metrics else '',
            'correction_audit_top1_after': train_metrics.get('correction_audit_top1_after', '') if train_metrics else '',
            'correction_audit_top1_gain': train_metrics.get('correction_audit_top1_gain', '') if train_metrics else '',
            'correction_audit_rank_before': train_metrics.get('correction_audit_rank_before', '') if train_metrics else '',
            'correction_audit_rank_after': train_metrics.get('correction_audit_rank_after', '') if train_metrics else '',
            'correction_audit_rank_gain': train_metrics.get('correction_audit_rank_gain', '') if train_metrics else '',
            'correction_audit_target_probability_before': train_metrics.get('correction_audit_target_probability_before', '') if train_metrics else '',
            'correction_audit_target_probability_after': train_metrics.get('correction_audit_target_probability_after', '') if train_metrics else '',
            'correction_audit_target_probability_gain': train_metrics.get('correction_audit_target_probability_gain', '') if train_metrics else '',
            'correction_audit_logit_margin_before': train_metrics.get('correction_audit_logit_margin_before', '') if train_metrics else '',
            'correction_audit_logit_margin_after': train_metrics.get('correction_audit_logit_margin_after', '') if train_metrics else '',
            'correction_audit_logit_margin_gain': train_metrics.get('correction_audit_logit_margin_gain', '') if train_metrics else '',
            'correction_audit_retention': train_metrics.get('correction_audit_retention', '') if train_metrics else '',
            'correction_audit_retention_rows': train_metrics.get('correction_audit_retention_rows', '') if train_metrics else '',
            'grad_total_norm': train_metrics.get('grad_total_norm', '') if train_metrics else '',
            'grad_clip_fraction': train_metrics.get('grad_clip_fraction', '') if train_metrics else '',
            'grad_clip_scale_mean': train_metrics.get('grad_clip_scale_mean', '') if train_metrics else '',
            'grad_backbone_norm': train_metrics.get('grad_backbone_norm', '') if train_metrics else '',
            'grad_policy_head_norm': train_metrics.get('grad_policy_head_norm', '') if train_metrics else '',
            'grad_value_head_norm': train_metrics.get('grad_value_head_norm', '') if train_metrics else '',
            'grad_policy_probe_norm': train_metrics.get('grad_policy_probe_norm', '') if train_metrics else '',
            'grad_value_probe_norm': train_metrics.get('grad_value_probe_norm', '') if train_metrics else '',
            'grad_policy_value_cosine': train_metrics.get('grad_policy_value_cosine', '') if train_metrics else '',
            'value_mae': train_metrics.get('value_mae', '') if train_metrics else '',
            'value_wdl_acc': train_metrics.get('value_wdl_acc', '') if train_metrics else '',
            'value_wdl_brier': train_metrics.get('value_wdl_brier', '') if train_metrics else '',
            'value_wdl_ece': train_metrics.get('value_wdl_ece', '') if train_metrics else '',
            'value_pred_win_probability': train_metrics.get('value_pred_win_probability', '') if train_metrics else '',
            'value_pred_draw_probability': train_metrics.get('value_pred_draw_probability', '') if train_metrics else '',
            'value_pred_loss_probability': train_metrics.get('value_pred_loss_probability', '') if train_metrics else '',
            'value_target_win_fraction': train_metrics.get('value_target_win_fraction', '') if train_metrics else '',
            'value_target_draw_fraction': train_metrics.get('value_target_draw_fraction', '') if train_metrics else '',
            'value_target_loss_fraction': train_metrics.get('value_target_loss_fraction', '') if train_metrics else '',
            'value_draw_probability_opening': train_metrics.get('value_draw_probability_opening', '') if train_metrics else '',
            'value_draw_probability_middlegame': train_metrics.get('value_draw_probability_middlegame', '') if train_metrics else '',
            'value_draw_probability_endgame': train_metrics.get('value_draw_probability_endgame', '') if train_metrics else '',
            'value_draw_target_fraction_opening': train_metrics.get('value_draw_target_fraction_opening', '') if train_metrics else '',
            'value_draw_target_fraction_middlegame': train_metrics.get('value_draw_target_fraction_middlegame', '') if train_metrics else '',
            'value_draw_target_fraction_endgame': train_metrics.get('value_draw_target_fraction_endgame', '') if train_metrics else '',
            'value_error_priority_fraction': train_metrics.get('value_error_priority_fraction', '') if train_metrics else '',
            'value_error_priority_weight_share': train_metrics.get('value_error_priority_weight_share', '') if train_metrics else '',
            'value_error_priority_mae': train_metrics.get('value_error_priority_mae', '') if train_metrics else '',
            'value_prediction_mean': train_metrics.get('value_prediction_mean', '') if train_metrics else '',
            'value_target_mean': train_metrics.get('value_target_mean', '') if train_metrics else '',
            'value_mean_bias': train_metrics.get('value_mean_bias', '') if train_metrics else '',
            'value_root_q_pred_mean': train_metrics.get('value_root_q_pred_mean', '') if train_metrics else '',
            'value_root_q_target_mean': train_metrics.get('value_root_q_target_mean', '') if train_metrics else '',
            'value_root_q_bias': train_metrics.get('value_root_q_bias', '') if train_metrics else '',
            'value_root_q_mae': train_metrics.get('value_root_q_mae', '') if train_metrics else '',
            'value_root_q_correlation': train_metrics.get('value_root_q_correlation', '') if train_metrics else '',
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
            'eval_mcts_move_samples': kwargs.get('eval_mcts_move_samples', ''),
            'eval_mcts_avg_simulations': kwargs.get('eval_mcts_avg_simulations', ''),
            'eval_mcts_reduced_budget_rate': kwargs.get(
                'eval_mcts_reduced_budget_rate', ''
            ),
            'eval_mcts_changed_rate': kwargs.get('eval_mcts_changed_rate', ''),
            'eval_mcts_higher_q_when_changed_rate': kwargs.get(
                'eval_mcts_higher_q_when_changed_rate', ''
            ),
            'eval_mcts_lower_q_when_changed_rate': kwargs.get(
                'eval_mcts_lower_q_when_changed_rate', ''
            ),
            'eval_mcts_changed_q_delta_mean': kwargs.get('eval_mcts_changed_q_delta_mean', ''),
            'eval_reference_mcts_changed_rate': kwargs.get(
                'eval_reference_mcts_changed_rate', ''
            ),
            'eval_reference_mcts_avg_simulations': kwargs.get(
                'eval_reference_mcts_avg_simulations', ''
            ),
            'eval_reference_mcts_reduced_budget_rate': kwargs.get(
                'eval_reference_mcts_reduced_budget_rate', ''
            ),
            'eval_reference_mcts_higher_q_when_changed_rate': kwargs.get(
                'eval_reference_mcts_higher_q_when_changed_rate', ''
            ),
            'anchor_games': anchor_games,
            'anchor_score_rate': anchor_score,
            'anchor_true_win_rate': kwargs.get('anchor_true_win_rate', ''),
            'anchor_score_lower_bound': kwargs.get(
                'anchor_score_lower_bound', _score_lower_bound(anchor_score, anchor_games)
            ),
            'anchor_no_mcts_games': anchor_no_mcts_games,
            'anchor_no_mcts_score_rate': anchor_no_mcts_score,
            'anchor_no_mcts_score_lower_bound': kwargs.get(
                'anchor_no_mcts_score_lower_bound',
                _score_lower_bound(anchor_no_mcts_score, anchor_no_mcts_games),
            ),
            'anchor_mcts_no_mcts_gap': kwargs.get('anchor_mcts_no_mcts_gap', ''),
            'rl_best_model': 1 if bool(kwargs.get('rl_best_model', False)) else '',
            'best_iteration': kwargs.get('best_iteration', ''),
            'estimated_elo_nn': estimated_elo_nn,
            'estimated_elo_nn_se': estimated_elo_nn_se,
            'estimated_elo_nn_ci95_low': estimated_elo_nn_ci95_low,
            'estimated_elo_nn_ci95_high': estimated_elo_nn_ci95_high,
            'estimated_elo_mcts': estimated_elo_mcts,
            'estimated_elo_mcts_se': estimated_elo_mcts_se,
            'estimated_elo_mcts_ci95_low': estimated_elo_mcts_ci95_low,
            'estimated_elo_mcts_ci95_high': estimated_elo_mcts_ci95_high,
            'estimated_elo_mcts_simulations': estimated_elo_mcts_simulations,
            'estimated_elo_mcts_by_simulations': estimated_elo_mcts_by_simulations,
        }
        row = csv_row(RL_MAIN_COLUMNS, record)

        # Kept only as a cheap guard for plot(); RL plots read canonical CSVs.
        self.iterations.append(iteration)
        with open(self.csv_path, 'a', newline='') as handle:
            csv.writer(handle).writerow(row)
