"""Self-play game engine built on the canonical batched MCTS search."""

import math
import time

import numpy as np
import torch

from src.game import backend as chess
from src.mcts.q_delta import (
    Q_DELTA_HIST_BINS as _Q_DELTA_HIST_BINS,
    RELIABLE_POLICY_TARGET_GAP_MIN,
    RELIABLE_POLICY_TARGET_TOP1_MIN,
    policy_target_is_reliable,
    q_delta_histogram as _q_delta_histogram,
    q_delta_percentile_from_histogram as _q_delta_percentile_from_histogram,
)
from src.mcts.search import (
    MultiGameBatchMCTS,
    _EMPTY_HISTORY_TENSOR,
    _REPLAY_SOURCE_UNKNOWN,
    _SELFPLAY_SELFPLAY_OPENING_LINES,
    _board_position_key,
    _build_history_tensor_from_encoded,
    _build_sparse_policy_target_from_visits,
    _get_syzygy_oracle,
    _record_position_count,
    _replay_source_code,
    _resolve_replay_max_policy_targets,
    _resolve_selfplay_max_moves,
    _search_correction_metadata,
    _select_move_from_visits_safe,
)
from src.models.data.se_cnn_v9.helpers import move_to_index


class SelfPlayEngine:
    """
    True batch self-play: multiple games in parallel, shared GPU eval batches.
    """

    def __init__(
        self,
        model,
        config,
        device,
        max_batch_games_per_worker,
        opponent_model=None,
        opponent_source_label="current",
        opponent_models_by_label=None,
        opponent_plan_labels=None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.opponent_model = opponent_model
        self.opponent_source_label = str(opponent_source_label or "current")
        rl_cfg = config.get('reinforcement_learning', {})
        self.mcts_phase_opening_max_fullmove = max(
            1,
            int(rl_cfg.get('value_phase_opening_max_fullmove', 12)),
        )
        self.mcts_phase_endgame_min_fullmove = max(
            self.mcts_phase_opening_max_fullmove + 1,
            int(rl_cfg.get('value_phase_endgame_min_fullmove', 40)),
        )

        self.mcts = MultiGameBatchMCTS(model, config, device)
        self.opponent_models_by_label = {}
        if isinstance(opponent_models_by_label, dict):
            for label, opp_model in opponent_models_by_label.items():
                if opp_model is not None:
                    self.opponent_models_by_label[str(label)] = opp_model
        elif opponent_model is not None:
            self.opponent_models_by_label[self.opponent_source_label] = opponent_model
        self.opponent_mcts_by_label = {
            str(label): MultiGameBatchMCTS(opp_model, config, device)
            for label, opp_model in self.opponent_models_by_label.items()
            if opp_model is not None
        }
        self.opponent_mcts = next(iter(self.opponent_mcts_by_label.values()), None)
        self.opponent_plan_labels = list(opponent_plan_labels or [])
        self._warned_missing_opponent_labels = set()
        self.history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
        self.share_trees = bool(rl_cfg.get('self_play_share_trees', True))

        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.num_simulations = int(config['reinforcement_learning']['mcts_simulations'])
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)
        self.max_moves = _resolve_selfplay_max_moves(config)
        self.auto_claim_draw = bool(rl_cfg.get('self_play_auto_claim_draw', False))
        self.claim_draw_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_draw_after_moves', self.max_moves)),
        )
        self.claim_repetition_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_repetition_after_moves', min(self.claim_draw_after_moves, 80))),
        )
        self.progress_report_interval_games = max(
            1,
            int(rl_cfg.get('self_play_progress_interval_games', 1)),
        )
        self.opening_diversity_enabled = bool(
            rl_cfg.get('self_play_opening_diversity_enabled', False)
        )
        self.opening_diversity_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_opening_diversity_fraction', 0.5))),
        )
        self.opening_diversity_min_plies = max(
            0,
            int(rl_cfg.get('self_play_opening_diversity_min_plies', 2)),
        )
        self.opening_diversity_max_plies = max(
            self.opening_diversity_min_plies,
            int(rl_cfg.get('self_play_opening_diversity_max_plies', 6)),
        )
        self.adjudication_enabled = bool(rl_cfg.get('self_play_adjudication_enabled', False))
        self.adjudication_min_moves = max(
            0,
            int(rl_cfg.get('self_play_adjudication_min_moves', 80)),
        )
        self.adjudication_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_adjudication_threshold', 0.92))),
        )
        self.adjudication_patience = max(
            1,
            int(rl_cfg.get('self_play_adjudication_patience', 6)),
        )
        self.syzygy = _get_syzygy_oracle(config)
        self.resignation_enabled = bool(rl_cfg.get('self_play_resignation_enabled', False))
        self.resignation_min_moves = max(
            0,
            int(rl_cfg.get('self_play_resignation_min_moves', 60)),
        )
        self.resignation_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_resignation_threshold', 0.92))),
        )
        self.resignation_patience = max(
            1,
            int(rl_cfg.get('self_play_resignation_patience', 3)),
        )
        self.resignation_disable_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_resignation_disable_fraction', 0.10))),
        )
        self.randomize_learner_color = bool(rl_cfg.get('self_play_randomize_learner_color', True))
        self.replay_dynamic_cap_enabled = bool(rl_cfg.get('replay_dynamic_cap_enabled', False))
        self.replay_cap_fraction_decisive = float(rl_cfg.get('replay_cap_fraction_decisive', 0.65))
        self.replay_cap_fraction_draw = float(rl_cfg.get('replay_cap_fraction_draw', 0.60))
        self.replay_cap_min_positions = int(rl_cfg.get('replay_cap_min_positions', 16))
        self.replay_cap_max_positions = int(rl_cfg.get('replay_cap_max_positions', 120))
        self.policy_target_max_moves = _resolve_replay_max_policy_targets(config)
        self.mcts_good_target_min_top_visit_prob = RELIABLE_POLICY_TARGET_TOP1_MIN
        self.mcts_good_target_min_visit_gap = RELIABLE_POLICY_TARGET_GAP_MIN
        self.store_frozen_best_positions = bool(
            rl_cfg.get('self_play_store_frozen_best_positions', True)
        )
        self.max_positions_per_game = max(
            0,
            int(rl_cfg.get('replay_max_positions_per_game', 32)),
        )
        self.hard_start_positions = list(rl_cfg.get('hard_start_positions', []) or [])

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0
        self._progress_base = 0
        self._plan_cursor = 0
        # Performance CSV/PNG should not depend on debug.rl.profile_training.
        # That flag controls console/debug verbosity, not collection of cheap counters.
        self.profile_enabled = True
        self._profile_stats = {}
        self.reset_profile_stats()

    def reset_profile_stats(self):
        self._profile_stats = {
            'move_selection_time': 0.0,
            'move_selection_calls': 0,
            'adjudication_time': 0.0,
            'adjudication_calls': 0,
            'syzygy_time': 0.0,
            'syzygy_calls': 0,
            'policy_target_build_time': 0.0,
            'policy_target_build_calls': 0,
            'policy_target_postgame_time': 0.0,
            'policy_target_postgame_calls': 0,
        }

    def _profile_add(self, key, value):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = float(self._profile_stats.get(key, 0.0)) + float(value)

    def _profile_inc(self, key, value=1):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = int(self._profile_stats.get(key, 0)) + int(value)

    def _merge_profile_stats(self, target, source):
        for key, value in dict(source or {}).items():
            if isinstance(value, (int, np.integer)):
                target[key] = int(target.get(key, 0)) + int(value)
            else:
                target[key] = float(target.get(key, 0.0)) + float(value)

    def _aggregate_engine_profile_stats(self):
        aggregated = dict(self._profile_stats)
        self._merge_profile_stats(
            aggregated,
            {f"learner_mcts_{k}": v for k, v in self.mcts.get_profile_stats().items()},
        )
        for label, opponent_mcts in self.opponent_mcts_by_label.items():
            self._merge_profile_stats(
                aggregated,
                {f"opponent_mcts_{label}_{k}": v for k, v in opponent_mcts.get_profile_stats().items()},
            )
        search_many_time = float(aggregated.get('learner_mcts_search_many_time', 0.0))
        batch_expand_time = float(aggregated.get('learner_mcts_batch_expand_eval_time', 0.0))
        batch_expand_calls = int(aggregated.get('learner_mcts_batch_expand_eval_calls', 0) or 0)
        board_tensor_time = float(aggregated.get('learner_mcts_board_to_tensor_time', 0.0))
        board_tensor_calls = int(aggregated.get('learner_mcts_board_to_tensor_calls', 0) or 0)
        nn_time = float(aggregated.get('learner_mcts_nn_inference_time', 0.0))
        nn_calls = int(aggregated.get('learner_mcts_nn_inference_calls', 0) or 0)
        nn_batch_items = int(aggregated.get('learner_mcts_nn_inference_batch_items', 0) or 0)
        selection_node_traversals = int(
            aggregated.get('learner_mcts_selection_node_traversals', 0) or 0
        )
        nn_h2d_time = float(aggregated.get('learner_mcts_nn_h2d_time', 0.0))
        nn_gpu_forward_time = float(aggregated.get('learner_mcts_nn_gpu_forward_time', 0.0))
        nn_gpu_postprocess_time = float(aggregated.get('learner_mcts_nn_gpu_postprocess_time', 0.0))
        nn_d2h_time = float(aggregated.get('learner_mcts_nn_d2h_time', 0.0))
        nn_legal_move_items = int(aggregated.get('learner_mcts_nn_legal_move_items', 0) or 0)
        central_requests = int(aggregated.get('learner_mcts_central_inference_requests', 0) or 0)
        central_batch_items = int(aggregated.get('learner_mcts_central_inference_server_batch_items', 0) or 0)
        central_request_put_time = float(aggregated.get('learner_mcts_central_inference_request_put_time', 0.0) or 0.0)
        central_remote_wait_time = float(aggregated.get('learner_mcts_central_inference_remote_wait_time', 0.0) or 0.0)
        central_server_queue_wait_time = float(aggregated.get('learner_mcts_central_inference_server_queue_wait_time', 0.0) or 0.0)
        central_server_descriptor_queue_wait_time = float(
            aggregated.get(
                'learner_mcts_central_inference_server_descriptor_queue_wait_time',
                0.0,
            ) or 0.0
        )
        central_server_batch_coalesce_wait_time = float(
            aggregated.get(
                'learner_mcts_central_inference_server_batch_coalesce_wait_time',
                0.0,
            ) or 0.0
        )
        central_server_total_time = float(aggregated.get('learner_mcts_central_inference_server_total_time', 0.0) or 0.0)
        central_server_concat_time = float(aggregated.get('learner_mcts_central_inference_server_concat_time', 0.0) or 0.0)
        central_server_h2d_time = float(aggregated.get('learner_mcts_central_inference_server_h2d_time', 0.0) or 0.0)
        central_server_forward_time = float(aggregated.get('learner_mcts_central_inference_server_forward_time', 0.0) or 0.0)
        central_server_d2h_time = float(aggregated.get('learner_mcts_central_inference_server_d2h_time', 0.0) or 0.0)
        for label in self.opponent_mcts_by_label.keys():
            prefix = f"opponent_mcts_{label}_"
            search_many_time += float(aggregated.get(prefix + 'search_many_time', 0.0))
            batch_expand_time += float(aggregated.get(prefix + 'batch_expand_eval_time', 0.0))
            batch_expand_calls += int(aggregated.get(prefix + 'batch_expand_eval_calls', 0) or 0)
            board_tensor_time += float(aggregated.get(prefix + 'board_to_tensor_time', 0.0))
            board_tensor_calls += int(aggregated.get(prefix + 'board_to_tensor_calls', 0) or 0)
            nn_time += float(aggregated.get(prefix + 'nn_inference_time', 0.0))
            nn_calls += int(aggregated.get(prefix + 'nn_inference_calls', 0) or 0)
            nn_batch_items += int(aggregated.get(prefix + 'nn_inference_batch_items', 0) or 0)
            selection_node_traversals += int(
                aggregated.get(prefix + 'selection_node_traversals', 0) or 0
            )
            nn_h2d_time += float(aggregated.get(prefix + 'nn_h2d_time', 0.0))
            nn_gpu_forward_time += float(aggregated.get(prefix + 'nn_gpu_forward_time', 0.0))
            nn_gpu_postprocess_time += float(aggregated.get(prefix + 'nn_gpu_postprocess_time', 0.0))
            nn_d2h_time += float(aggregated.get(prefix + 'nn_d2h_time', 0.0))
            nn_legal_move_items += int(aggregated.get(prefix + 'nn_legal_move_items', 0) or 0)
            central_requests += int(aggregated.get(prefix + 'central_inference_requests', 0) or 0)
            central_batch_items += int(aggregated.get(prefix + 'central_inference_server_batch_items', 0) or 0)
            central_request_put_time += float(aggregated.get(prefix + 'central_inference_request_put_time', 0.0) or 0.0)
            central_remote_wait_time += float(aggregated.get(prefix + 'central_inference_remote_wait_time', 0.0) or 0.0)
            central_server_queue_wait_time += float(aggregated.get(prefix + 'central_inference_server_queue_wait_time', 0.0) or 0.0)
            central_server_descriptor_queue_wait_time += float(
                aggregated.get(
                    prefix + 'central_inference_server_descriptor_queue_wait_time',
                    0.0,
                ) or 0.0
            )
            central_server_batch_coalesce_wait_time += float(
                aggregated.get(
                    prefix + 'central_inference_server_batch_coalesce_wait_time',
                    0.0,
                ) or 0.0
            )
            central_server_total_time += float(aggregated.get(prefix + 'central_inference_server_total_time', 0.0) or 0.0)
            central_server_concat_time += float(aggregated.get(prefix + 'central_inference_server_concat_time', 0.0) or 0.0)
            central_server_h2d_time += float(aggregated.get(prefix + 'central_inference_server_h2d_time', 0.0) or 0.0)
            central_server_forward_time += float(aggregated.get(prefix + 'central_inference_server_forward_time', 0.0) or 0.0)
            central_server_d2h_time += float(aggregated.get(prefix + 'central_inference_server_d2h_time', 0.0) or 0.0)
        aggregated['mcts_search_many_time'] = float(search_many_time)
        aggregated['mcts_batch_expand_eval_time'] = float(batch_expand_time)
        aggregated['mcts_batch_expand_eval_calls'] = int(batch_expand_calls)
        aggregated['mcts_board_to_tensor_time'] = float(board_tensor_time)
        aggregated['mcts_board_to_tensor_calls'] = int(board_tensor_calls)
        aggregated['mcts_nn_inference_time'] = float(nn_time)
        aggregated['mcts_nn_inference_calls'] = int(nn_calls)
        aggregated['mcts_nn_inference_batch_items'] = int(nn_batch_items)
        aggregated['mcts_selection_node_traversals'] = int(selection_node_traversals)
        aggregated['mcts_nn_h2d_time'] = float(nn_h2d_time)
        aggregated['mcts_nn_gpu_forward_time'] = float(nn_gpu_forward_time)
        aggregated['mcts_nn_gpu_postprocess_time'] = float(nn_gpu_postprocess_time)
        aggregated['mcts_nn_d2h_time'] = float(nn_d2h_time)
        aggregated['mcts_nn_legal_move_items'] = int(nn_legal_move_items)
        aggregated['mcts_central_inference_requests'] = int(central_requests)
        aggregated['mcts_central_inference_server_batch_items'] = int(central_batch_items)
        aggregated['mcts_central_inference_request_put_time'] = float(central_request_put_time)
        aggregated['mcts_central_inference_remote_wait_time'] = float(central_remote_wait_time)
        aggregated['mcts_central_inference_server_queue_wait_time'] = float(central_server_queue_wait_time)
        aggregated['mcts_central_inference_server_descriptor_queue_wait_time'] = float(
            central_server_descriptor_queue_wait_time
        )
        aggregated['mcts_central_inference_server_batch_coalesce_wait_time'] = float(
            central_server_batch_coalesce_wait_time
        )
        aggregated['mcts_central_inference_server_total_time'] = float(central_server_total_time)
        aggregated['mcts_central_inference_server_concat_time'] = float(central_server_concat_time)
        aggregated['mcts_central_inference_server_h2d_time'] = float(central_server_h2d_time)
        aggregated['mcts_central_inference_server_forward_time'] = float(central_server_forward_time)
        aggregated['mcts_central_inference_server_d2h_time'] = float(central_server_d2h_time)
        central_extra_metrics = (
            'central_inference_shared_requests',
            'central_inference_shared_bytes_avoided',
            'central_inference_compact_policy_requests',
            'central_inference_compact_output_bytes_avoided',
            'central_inference_shared_slot_wait_time',
            'central_inference_cache_queries',
            'central_inference_cache_bypassed_positions',
            'central_inference_cache_hits',
            'central_inference_dedup_hits',
            'central_inference_cache_suspensions',
            'central_inference_cache_reactivations',
            'central_inference_nn_evaluated_positions',
            'central_inference_server_cache_lookup_time',
            'central_inference_server_staging_copy_time',
            'central_inference_server_pipeline_wait_time',
            'central_inference_gpu_batch_fill_sum',
            'central_inference_server_batch_target_sum',
            'central_inference_server_output_finalize_wait_time',
            'central_inference_server_output_pipeline_requests',
            'central_inference_server_auto_calibrated_requests',
        )
        for metric in central_extra_metrics:
            total = aggregated.get(f'learner_mcts_{metric}', 0) or 0
            for label in self.opponent_mcts_by_label.keys():
                total += aggregated.get(f'opponent_mcts_{label}_{metric}', 0) or 0
            aggregated[f'mcts_{metric}'] = total
        extra_mcts_time_metrics = [
            'search_root_setup_time',
            'search_selection_time',
            'search_backprop_time',
            'native_tree_expand_sync_time',
            'native_tree_backup_time',
            'native_tree_import_time',
            'native_tree_sync_time',
            'search_metadata_time',
            'board_materialize_time',
            'terminal_checks_time',
            'batch_expand_dedup_terminal_time',
            'batch_expand_legal_moves_time',
            'batch_expand_move_index_time',
            'batch_expand_tensor_pack_time',
            'batch_expand_history_time',
            'batch_expand_input_pack_time',
            'batch_expand_legal_index_pack_time',
            'batch_expand_cpu_policy_time',
            'batch_expand_value_fanout_time',
        ]
        for metric in extra_mcts_time_metrics:
            total = float(aggregated.get(f'learner_mcts_{metric}', 0.0) or 0.0)
            for label in self.opponent_mcts_by_label.keys():
                total += float(aggregated.get(f'opponent_mcts_{label}_{metric}', 0.0) or 0.0)
            aggregated[f'mcts_{metric}'] = float(total)
        for metric in (
            'native_root_selection_calls',
            'python_root_selection_calls',
            'native_completed_q_calls',
            'python_completed_q_calls',
            'native_tree_selection_batches',
            'native_tree_backup_batches',
            'native_tree_selected_leaves',
            'python_tree_selected_leaves',
            'native_tree_import_nodes',
            'native_tree_import_edges',
            'native_tree_fallbacks',
        ):
            total = int(aggregated.get(f'learner_mcts_{metric}', 0) or 0)
            for label in self.opponent_mcts_by_label.keys():
                total += int(aggregated.get(f'opponent_mcts_{label}_{metric}', 0) or 0)
            aggregated[f'mcts_{metric}'] = int(total)
        root_selection_calls = int(aggregated.get('mcts_native_root_selection_calls', 0) or 0)
        root_selection_calls += int(aggregated.get('mcts_python_root_selection_calls', 0) or 0)
        aggregated['mcts_native_root_selection_rate'] = (
            float(aggregated.get('mcts_native_root_selection_calls', 0) or 0)
            / float(root_selection_calls)
            if root_selection_calls > 0 else 0.0
        )
        completed_q_calls = int(aggregated.get('mcts_native_completed_q_calls', 0) or 0)
        completed_q_calls += int(aggregated.get('mcts_python_completed_q_calls', 0) or 0)
        aggregated['mcts_native_completed_q_rate'] = (
            float(aggregated.get('mcts_native_completed_q_calls', 0) or 0)
            / float(completed_q_calls)
            if completed_q_calls > 0 else 0.0
        )
        tree_selected_leaves = int(
            aggregated.get('mcts_native_tree_selected_leaves', 0) or 0
        )
        tree_selected_leaves += int(
            aggregated.get('mcts_python_tree_selected_leaves', 0) or 0
        )
        aggregated['mcts_native_tree_selection_rate'] = (
            float(aggregated.get('mcts_native_tree_selected_leaves', 0) or 0)
            / float(tree_selected_leaves)
            if tree_selected_leaves > 0 else 0.0
        )
        aggregated['average_batch_size'] = float(nn_batch_items / nn_calls) if nn_calls > 0 else 0.0
        aggregated['central_average_batch_size'] = (
            float(central_batch_items / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_remote_wait_ms_per_request'] = (
            1000.0 * float(central_remote_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_request_submit_ms_per_request'] = (
            1000.0 * float(central_request_put_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_queue_wait_ms_per_request'] = (
            1000.0 * float(central_server_queue_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_descriptor_queue_wait_ms_per_request'] = (
            1000.0 * float(
                central_server_descriptor_queue_wait_time / central_requests
            ) if central_requests > 0 else 0.0
        )
        aggregated['central_batch_coalesce_wait_ms_per_request'] = (
            1000.0 * float(
                central_server_batch_coalesce_wait_time / central_requests
            ) if central_requests > 0 else 0.0
        )
        aggregated['central_server_forward_ms_per_request'] = (
            1000.0 * float(central_server_forward_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_h2d_ms_per_request'] = (
            1000.0 * float(central_server_h2d_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_d2h_ms_per_request'] = (
            1000.0 * float(central_server_d2h_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_concat_ms_per_request'] = (
            1000.0 * float(central_server_concat_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_total_ms_per_request'] = (
            1000.0 * float(central_server_total_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['average_legal_moves_per_position'] = (
            float(nn_legal_move_items / nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        aggregated['average_legal_moves_per_batch'] = (
            float(nn_legal_move_items / nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_batch_ms'] = (
            1000.0 * float(nn_time) / float(nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_position_ms'] = (
            1000.0 * float(nn_time) / float(nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        worker_nn_wait_share_pct = 100.0 * float(nn_time) / max(1e-8, float(search_many_time))
        aggregated['worker_nn_wait_share_pct'] = float(
            max(0.0, min(100.0, worker_nn_wait_share_pct))
        )
        return aggregated

    def _prune_policy_target_visits(self, visit_counts):
        if not visit_counts:
            return visit_counts

        items = sorted(
            ((move, float(count)) for move, count in visit_counts.items() if float(count) > 0.0),
            key=lambda pair: (-pair[1], pair[0].uci()),
        )
        if not items:
            return visit_counts
        kept = items[:self.policy_target_max_moves]
        return {move: int(max(1.0, round(count))) for move, count in kept}

    def _prune_policy_target_weights(self, policy_weights):
        """Limit a Gumbel improved-policy target without quantizing probabilities."""
        if not policy_weights:
            return policy_weights
        items = sorted(
            ((move, float(weight)) for move, weight in policy_weights.items() if float(weight) > 0.0),
            key=lambda pair: (-pair[1], pair[0].uci()),
        )
        return dict(items[:self.policy_target_max_moves])

    def _mcts_phase_for_board(self, board):
        try:
            fullmove_number = int(getattr(board, 'fullmove_number', 1) or 1)
        except (TypeError, ValueError):
            fullmove_number = 1
        if fullmove_number <= int(self.mcts_phase_opening_max_fullmove):
            return 'opening'
        if fullmove_number >= int(self.mcts_phase_endgame_min_fullmove):
            return 'endgame'
        return 'middlegame'

    def _should_store_policy_position(self, learner_turn, game_opponent_mcts, opponent_label):
        if learner_turn or game_opponent_mcts is None:
            return True
        label = str(opponent_label or "")
        if label == "best":
            return bool(self.store_frozen_best_positions)
        return False

    @staticmethod
    def _select_evenly_spaced_candidates(candidates, effective_cap):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        ordered = sorted(candidates, key=lambda item: int(item['history_idx']))
        positions = np.linspace(0, len(ordered) - 1, num=effective_cap)
        selected_history = []
        seen_history = set()
        for pos in positions:
            idx = int(round(float(pos)))
            idx = max(0, min(idx, len(ordered) - 1))
            history_idx = int(ordered[idx]['history_idx'])
            if history_idx in seen_history:
                continue
            seen_history.add(history_idx)
            selected_history.append(history_idx)
        if len(selected_history) < effective_cap:
            for item in ordered:
                history_idx = int(item['history_idx'])
                if history_idx in seen_history:
                    continue
                seen_history.add(history_idx)
                selected_history.append(history_idx)
                if len(selected_history) >= effective_cap:
                    break
        selected_lookup = set(selected_history[:effective_cap])
        return [item for item in ordered if int(item['history_idx']) in selected_lookup]

    def _select_top_scored_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        del history_len
        ranked = sorted(
            candidates,
            key=lambda item: (-self._candidate_importance(item), int(item['history_idx'])),
        )
        top_quota = int(round(effective_cap * self.replay_importance_top_fraction))
        top_quota = max(1, min(int(effective_cap), top_quota))

        selected = []
        selected_history = set()
        for item in ranked:
            history_idx = int(item['history_idx'])
            if history_idx in selected_history:
                continue
            selected.append(item)
            selected_history.add(history_idx)
            if len(selected) >= top_quota:
                break

        remaining = int(effective_cap - len(selected))
        if remaining > 0:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_history
            ]
            for item in self._select_evenly_spaced_candidates(leftovers, remaining):
                history_idx = int(item['history_idx'])
                if history_idx in selected_history:
                    continue
                selected.append(item)
                selected_history.add(history_idx)
                if len(selected) >= effective_cap:
                    break

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected[:effective_cap]

    def _allocate_phase_stratified_caps(self, buckets, effective_cap):
        bucket_count = len(buckets)
        allocations = [0] * bucket_count
        if effective_cap <= 0 or bucket_count <= 0:
            return allocations

        bucket_sizes = [len(bucket) for bucket in buckets]
        non_empty = [idx for idx, size in enumerate(bucket_sizes) if size > 0]
        if not non_empty:
            return allocations

        if effective_cap < len(non_empty):
            desired = []
            for idx in range(bucket_count):
                if bucket_sizes[idx] <= 0:
                    desired.append(0.0)
                elif idx == bucket_count - 1:
                    desired.append(1.0)
                elif idx == bucket_count - 2:
                    desired.append(0.75)
                else:
                    desired.append(0.5)
        else:
            # Preserve calm opening and middlegame positions instead of letting
            # tactical/endgame importance dominate a heavily capped game.
            desired = [0.30, 0.45, 0.25]

        remaining = int(effective_cap)
        for idx in non_empty:
            allocations[idx] = 1
            remaining -= 1

        if remaining <= 0:
            return allocations

        desired_counts = [float(effective_cap) * desired[idx] for idx in range(bucket_count)]
        remainders = []
        for idx in range(bucket_count):
            if bucket_sizes[idx] <= allocations[idx]:
                continue
            extra = int(math.floor(max(0.0, desired_counts[idx] - allocations[idx])))
            if extra > 0:
                grant = min(extra, bucket_sizes[idx] - allocations[idx], remaining)
                allocations[idx] += grant
                remaining -= grant
            remainder = max(0.0, desired_counts[idx] - allocations[idx])
            remainders.append((remainder, idx))

        while remaining > 0:
            progressed = False
            remainders.sort(key=lambda pair: (-pair[0], pair[1]))
            for _, idx in remainders:
                if remaining <= 0:
                    break
                if allocations[idx] >= bucket_sizes[idx]:
                    continue
                allocations[idx] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        if remaining > 0:
            fill_order = sorted(
                non_empty,
                key=lambda idx: (
                    -bucket_sizes[idx],
                    -desired_counts[idx],
                    -idx,
                ),
            )
            while remaining > 0:
                progressed = False
                for idx in fill_order:
                    if remaining <= 0:
                        break
                    if allocations[idx] >= bucket_sizes[idx]:
                        continue
                    allocations[idx] += 1
                    remaining -= 1
                    progressed = True
                if not progressed:
                    break

        return allocations

    def _select_phase_stratified_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)

        if effective_cap < 3:
            return self._select_top_scored_candidates(candidates, effective_cap, history_len)

        denom = float(max(1, history_len - 1))
        buckets = [[], [], []]
        for item in candidates:
            progress = float(item['history_idx']) / denom
            if progress < (1.0 / 3.0):
                buckets[0].append(item)
            elif progress < (2.0 / 3.0):
                buckets[1].append(item)
            else:
                buckets[2].append(item)

        allocations = self._allocate_phase_stratified_caps(buckets, effective_cap)
        selected = []
        selected_ids = set()
        for bucket, bucket_cap in zip(buckets, allocations):
            for item in self._select_top_scored_candidates(bucket, bucket_cap, history_len):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        if len(selected) < effective_cap:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_ids
            ]
            for item in self._select_top_scored_candidates(
                leftovers,
                effective_cap - len(selected),
                history_len,
            ):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        return selected

    def _select_candidates_with_source_balance(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)

        buckets = {}
        for item in candidates:
            source_code = int(item.get('source_code', _REPLAY_SOURCE_UNKNOWN))
            buckets.setdefault(source_code, []).append(item)
        non_empty_sources = [
            source_code for source_code, bucket in buckets.items()
            if len(bucket) > 0
        ]
        if len(non_empty_sources) <= 1:
            return self._select_phase_stratified_candidates(candidates, effective_cap, history_len)

        total_candidates = max(1, len(candidates))
        allocations = {}
        remaining = int(effective_cap)
        if effective_cap >= len(non_empty_sources):
            for source_code in non_empty_sources:
                allocations[source_code] = 1
                remaining -= 1
        else:
            ranked_sources = sorted(
                non_empty_sources,
                key=lambda source_code: (
                    -max(self._candidate_importance(item) for item in buckets[source_code]),
                    source_code,
                ),
            )
            for source_code in ranked_sources[:effective_cap]:
                allocations[source_code] = 1
            remaining = 0

        desired = {
            source_code: float(effective_cap) * len(buckets[source_code]) / float(total_candidates)
            for source_code in non_empty_sources
        }
        while remaining > 0:
            progressed = False
            ranked_sources = sorted(
                non_empty_sources,
                key=lambda source_code: (
                    -(desired[source_code] - allocations.get(source_code, 0)),
                    -len(buckets[source_code]),
                    source_code,
                ),
            )
            for source_code in ranked_sources:
                if remaining <= 0:
                    break
                current = int(allocations.get(source_code, 0))
                if current >= len(buckets[source_code]):
                    continue
                allocations[source_code] = current + 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        selected = []
        selected_ids = set()
        for source_code in sorted(non_empty_sources):
            bucket_cap = int(allocations.get(source_code, 0))
            if bucket_cap <= 0:
                continue
            bucket = buckets[source_code]
            bucket_selected = self._select_phase_stratified_candidates(
                bucket,
                bucket_cap,
                history_len,
            )
            for item in bucket_selected:
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        if len(selected) < effective_cap:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_ids
            ]
            remaining_cap = int(effective_cap - len(selected))
            refill = self._select_phase_stratified_candidates(
                leftovers,
                remaining_cap,
                history_len,
            )
            for item in refill:
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)
                if len(selected) >= effective_cap:
                    break

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected[:effective_cap]

    def _select_history_indices_to_keep(self, candidates, history_len, is_decisive=False):
        if not candidates:
            return [], 0, 0

        filtered = list(candidates)
        curriculum_dropped = 0

        # Determine effective cap limits
        if self.replay_dynamic_cap_enabled:
            fraction = self.replay_cap_fraction_decisive if is_decisive else self.replay_cap_fraction_draw
            effective_cap = int(math.ceil(len(filtered) * fraction))
            effective_cap = max(self.replay_cap_min_positions, effective_cap)
            effective_cap = min(self.replay_cap_max_positions, effective_cap)
        else:
            effective_cap = self.max_positions_per_game

        if effective_cap <= 0 or len(filtered) <= effective_cap:
            selected = filtered
            cap_dropped = 0
        else:
            # Keep the stored trajectory representative. Search confidence and
            # tactical salience belong in the target distribution, not in a
            # second hidden sampling policy layered on top of it.
            selected = self._select_evenly_spaced_candidates(filtered, effective_cap)
            cap_dropped = len(filtered) - len(selected)

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected, int(curriculum_dropped), int(cap_dropped)

    def _build_history_selection_candidates(self, gs):
        candidates = []
        for history_idx, history_entry in enumerate(gs['game_history']):
            importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
            candidates.append({
                'history_idx': history_idx,
                'importance_score': importance_score,
                'source_code': int(history_entry[6]) if len(history_entry) > 6 else _REPLAY_SOURCE_UNKNOWN,
            })
        return candidates

    def _build_training_position_from_history_entry(
        self,
        gs,
        history_idx,
        history_len,
        outcome,
        draw_value_target=0.0,
    ):
        history_entry = gs['game_history'][int(history_idx)]
        history_count, policy_indices, policy_values, turn = history_entry[:4]
        importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
        policy_weight = float(history_entry[5]) if len(history_entry) > 5 else 1.0
        source_code = int(history_entry[6]) if len(history_entry) > 6 else _REPLAY_SOURCE_UNKNOWN
        legal_indices = history_entry[7] if len(history_entry) > 7 and history_entry[7] is not None else policy_indices
        fen = history_entry[8] if len(history_entry) > 8 else None
        root_q = float(history_entry[9]) if len(history_entry) > 9 else float('nan')
        history_fens = tuple(history_entry[10] or ()) if len(history_entry) > 10 else ()
        search_changed_top = bool(history_entry[11]) if len(history_entry) > 11 else False
        search_q_delta = float(history_entry[12]) if len(history_entry) > 12 else float('nan')
        best_q = float(history_entry[13]) if len(history_entry) > 13 else float('nan')
        played_q = float(history_entry[14]) if len(history_entry) > 14 else float('nan')
        orig_q = float(history_entry[15]) if len(history_entry) > 15 else float('nan')
        policy_kld = float(history_entry[16]) if len(history_entry) > 16 else float('nan')
        search_visits = int(history_entry[17]) if len(history_entry) > 17 else 0

        if outcome == 0.0:
            value = draw_value_target
        else:
            signed_outcome = outcome if turn == chess.WHITE else -outcome
            value = float(signed_outcome)
        value_weight = 1.0
        # The live board-history cache is intentionally bounded.  The replay
        # target still knows the exact distance to the end from its stable game
        # index, so it must not depend on the cache length.
        moves_left = max(0, int(history_len) - int(history_idx))

        return {
            'history_idx': int(history_idx),
            'history_count': history_count,
            'policy_indices': policy_indices,
            'policy_values': policy_values,
            'turn': turn,
            'value': value,
            'importance_score': importance_score,
            'policy_weight': policy_weight,
            'value_weight': value_weight,
            'moves_left': moves_left,
            'legal_indices': legal_indices,
            'source_code': source_code,
            'fen': fen,
            'root_q': root_q,
            'history_fens': history_fens,
            'search_changed_top': search_changed_top,
            'search_q_delta': search_q_delta,
            'best_q': best_q,
            'played_q': played_q,
            'orig_q': orig_q,
            'policy_kld': policy_kld,
            'search_visits': search_visits,
        }

    def _build_replay_history_tensor(self, item, history_positions, encoded_fen_cache):
        """Rebuild one selected input from compact FEN history after the game."""
        current_fen = item.get('fen')
        if not current_fen:
            raise RuntimeError("self-play replay row is missing its current FEN")
        history_fens = list(item.get('history_fens', ()) or [])
        if history_positions > 0:
            history_fens = history_fens[-int(history_positions):]
        else:
            history_fens = []

        encoded_history = []
        for fen in history_fens + [current_fen]:
            encoded = encoded_fen_cache.get(fen)
            if encoded is None:
                encoded = self.mcts._encode_history_entry(fen)
                encoded_fen_cache[fen] = encoded
            encoded_history.append(encoded)

        return _build_history_tensor_from_encoded(
            turn=item['turn'],
            encoded_history=encoded_history,
            history_count=len(history_fens),
            history_positions=history_positions,
            empty_history_tensor=_EMPTY_HISTORY_TENSOR,
        )

    def _compute_position_importance(self, board, move, visit_counts, root, search_metadata=None):
        importance = 1.0

        if visit_counts:
            visits = np.asarray(list(visit_counts.values()), dtype=np.float32)
            total_visits = float(visits.sum())
            if total_visits > 0.0:
                probs = visits / total_visits
                top_prob = float(probs.max())
                entropy = 0.0
                if probs.size > 1:
                    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
                    entropy /= float(np.log(probs.size))
                importance += 0.35 * (1.0 - top_prob)
                importance += 0.30 * entropy

        if root is not None:
            root_visits = int(getattr(root, 'visit_count', 0) or 0)
            if root_visits > 0:
                root_value = float(root.value_sum / max(1, root_visits))
                importance += 0.30 * abs(root_value)

        # MCTS disagreements are precisely the positions that can teach the raw
        # policy something new. Keep them ahead of merely tactical/evenly-spaced
        # positions when a game's replay cap is tight.
        if isinstance(search_metadata, dict):
            reliable_target = policy_target_is_reliable(
                search_metadata.get('top_visit_prob'),
                search_metadata.get('visit_gap'),
            )
            if (
                reliable_target
                and float(search_metadata.get('prior_mcts_agree', 1.0) or 0.0) < 0.5
            ):
                importance += 0.45
                try:
                    q_delta = float(search_metadata.get('mcts_q_delta', 0.0) or 0.0)
                except (TypeError, ValueError):
                    q_delta = 0.0
                importance += 0.20 * min(1.0, max(0.0, q_delta) / 0.30)
            try:
                policy_kl = float(search_metadata.get('mcts_policy_kl', 0.0) or 0.0)
            except (TypeError, ValueError):
                policy_kl = 0.0
            importance += 0.10 * min(1.0, max(0.0, policy_kl) / 0.20)

        if chess.is_capture(board, move):
            importance += 0.30
            gain = self.mcts._captured_piece_value(board, move) - self.mcts._moving_piece_value(board, move)
            if gain > 0.0:
                importance += 0.20 * min(1.0, gain / 4.0)
        if move.promotion is not None:
            importance += 0.30
        try:
            if chess.gives_check(board, move):
                importance += 0.15
        except Exception:
            pass
        last_move = chess.last_move(board)
        if last_move is not None and move.destination == last_move.destination:
            importance += 0.10

        return float(importance)

    def _append_selected_positions_from_game(
        self,
        positions,
        gs,
        outcome,
        *,
        history_positions,
    ):
        postgame_t0 = time.perf_counter() if self.profile_enabled else None
        history_len = len(gs['game_history'])
        selection_candidates = self._build_history_selection_candidates(gs)
        selected_candidates, curriculum_dropped, cap_dropped = self._select_history_indices_to_keep(
            selection_candidates,
            history_len,
            is_decisive=(outcome != 0.0),
        )
        encoded_fen_cache = {}

        for candidate in selected_candidates:
            item = self._build_training_position_from_history_entry(
                gs,
                candidate['history_idx'],
                history_len,
                outcome,
                draw_value_target=0.0,
            )
            board_tensor_np = self._build_replay_history_tensor(
                item,
                history_positions,
                encoded_fen_cache,
            )
            if board_tensor_np.dtype != np.float32:
                board_tensor_np = board_tensor_np.astype(np.float32, copy=False)
            board_tensor = torch.from_numpy(board_tensor_np)
            positions.append((
                board_tensor,
                item['policy_indices'],
                item['policy_values'],
                torch.tensor([item['value']], dtype=torch.float32),
                float(item.get('importance_score', 0.0)),
                float(item.get('policy_weight', 1.0)),
                float(item.get('value_weight', 1.0)),
                int(item.get('source_code', _REPLAY_SOURCE_UNKNOWN)),
                float(item.get('moves_left', 0.0)),
                item.get('legal_indices', item['policy_indices']),
                item.get('fen'),
                item.get('root_q', float('nan')),
                item.get('history_fens', ()),
                bool(item.get('search_changed_top', False)),
                float(item.get('search_q_delta', float('nan'))),
                float(item.get('best_q', float('nan'))),
                float(item.get('played_q', float('nan'))),
                float(item.get('orig_q', float('nan'))),
                float(item.get('policy_kld', float('nan'))),
                int(item.get('search_visits', 0)),
                int(gs.get('_replay_game_id', -1)),
                int(item.get('history_idx', -1)),
            ))
        if self.profile_enabled:
            self._profile_add('policy_target_postgame_time', time.perf_counter() - postgame_t0)
            self._profile_inc('policy_target_postgame_calls', 1)
        return history_len, int(curriculum_dropped), int(cap_dropped)

    def _should_auto_claim_draw(self, board, move_count):
        if not self.auto_claim_draw:
            return False
        if (
            move_count >= self.claim_repetition_after_moves
            and chess.can_claim_threefold_repetition(board)
        ):
            return True
        return bool(
            move_count >= self.claim_draw_after_moves
            and chess.can_claim_draw(board)
        )

    def _sample_opening_prefix(self):
        if not self.opening_diversity_enabled:
            return ()
        if self.opening_diversity_fraction <= 0.0:
            return ()
        if np.random.random() > self.opening_diversity_fraction:
            return ()

        line = _SELFPLAY_SELFPLAY_OPENING_LINES[
            int(np.random.randint(0, len(_SELFPLAY_SELFPLAY_OPENING_LINES)))
        ]
        max_plies = min(len(line), self.opening_diversity_max_plies)
        min_plies = min(max_plies, self.opening_diversity_min_plies)
        if max_plies <= 0:
            return ()
        if min_plies >= max_plies:
            plies = max_plies
        else:
            plies = int(np.random.randint(min_plies, max_plies + 1))
        return line[:plies]

    def _maybe_adjudicate_game(self, gs, root, board):
        if not self.adjudication_enabled or root is None:
            return None
        if gs['move_count'] < self.adjudication_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        white_value = root_value if board.turn == chess.WHITE else -root_value
        threshold = self.adjudication_threshold

        if white_value >= threshold:
            gs['white_advantage_streak'] = int(gs.get('white_advantage_streak', 0)) + 1
            gs['black_advantage_streak'] = 0
            if gs['white_advantage_streak'] >= self.adjudication_patience:
                return '1-0'
        elif white_value <= -threshold:
            gs['black_advantage_streak'] = int(gs.get('black_advantage_streak', 0)) + 1
            gs['white_advantage_streak'] = 0
            if gs['black_advantage_streak'] >= self.adjudication_patience:
                return '0-1'
        else:
            gs['white_advantage_streak'] = 0
            gs['black_advantage_streak'] = 0

        return None

    def _clear_game_search_state(self, gs):
        tree_key = gs.get('_native_tree_key', id(gs.get('board')))
        self.mcts.release_native_tree(tree_key)
        opponent_mcts = gs.get('opponent_mcts')
        if opponent_mcts is not None:
            opponent_mcts.release_native_tree(tree_key)
        gs['root'] = None
        gs['opponent_root'] = None
        gs['_root_synced'] = False
        gs['_opponent_root_synced'] = False

    @staticmethod
    def _advance_search_root(root, move):
        if root is None:
            return None, False
        edge_idx = root.edges._get_move_index(move) if root.edges is not None else None
        child = (
            root.edges.nodes[int(edge_idx)]
            if edge_idx is not None and int(edge_idx) < len(root.edges.nodes)
            else None
        )
        if child is None:
            return None, False
        _ = child.board
        return child.detach_as_root(), True

    def _maybe_finish_with_syzygy(self, gs, board):
        if self.syzygy is None or chess.is_game_over(board, claim_draw=False):
            return None
        if not self.syzygy.can_probe(board):
            return None
        gs['syzygy_probe_positions'] = int(gs.get('syzygy_probe_positions', 0)) + 1
        value = self.syzygy.probe_value(board)
        if value is None:
            return None
        gs['syzygy_probe_hits'] = int(gs.get('syzygy_probe_hits', 0)) + 1

        if value > 0:
            result = '1-0' if board.turn == chess.WHITE else '0-1'
        elif value < 0:
            result = '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            result = '1/2-1/2'

        gs['done'] = True
        gs['syzygy_result'] = result
        self._clear_game_search_state(gs)
        self._mark_game_completed(gs)
        return result

    def _maybe_resign_game(self, gs, root, board):
        if not self.resignation_enabled or gs.get('resignation_disabled', False):
            return None
        if root is None or gs['move_count'] < self.resignation_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        if root_value <= -self.resignation_threshold:
            gs['resign_streak'] = int(gs.get('resign_streak', 0)) + 1
            if gs['resign_streak'] >= self.resignation_patience:
                return '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            gs['resign_streak'] = 0
        return None

    def _apply_opening_prefix(self, gs):
        prefix = self._sample_opening_prefix()
        if not prefix:
            return

        board = gs['board']
        for uci in prefix:
            try:
                move = chess.move_from_uci(uci)
            except Exception:
                break
            if move is None or move not in chess.legal_moves(board):
                break

            gs['board_history'].append(self.mcts._encode_history_entry(board))
            gs.setdefault('fen_history', []).append(chess.board_fen(board))
            self._trim_history_cache(gs['board_history'])
            self._trim_history_cache(gs['fen_history'])
            chess.apply_move(board, move)
            gs['move_count'] += 1
            _record_position_count(gs['position_counts'], board)

            if chess.is_game_over(board, claim_draw=False) or gs['move_count'] >= self.max_moves:
                gs['done'] = True
                break

    def _trim_history_cache(self, history):
        """Keep only entries that can still affect the model input or replay row."""
        limit = max(0, int(self.history_positions))
        if limit == 0:
            history.clear()
        elif len(history) > limit:
            del history[:-limit]

    def play_games(
        self,
        num_games,
        game_job_queue=None,
        stream_task_id=None,
        initial_game_jobs=None,
        completed_chunk_callback=None,
        completed_chunk_games=8,
    ):
        all_positions = []
        game_lengths = []
        self._games_completed = 0
        self._plan_cursor = 0
        self.reset_profile_stats()
        self.mcts.reset_profile_stats()
        for opponent_mcts in self.opponent_mcts_by_label.values():
            opponent_mcts.reset_profile_stats()
        total_dropped_positions = 0
        total_truncated_games = 0
        total_claimable_draw_ended_games = 0
        total_completed_length_sum = 0
        total_truncated_length_sum = 0
        total_completed_white_wins = 0
        total_completed_black_wins = 0
        total_completed_draws = 0
        total_decisive_games = 0
        total_decisive_length_sum = 0
        total_curriculum_dropped_positions = 0
        total_cap_dropped_positions = 0
        total_resigned_games = 0
        total_syzygy_ended_games = 0
        total_syzygy_probe_positions = 0
        total_syzygy_probe_hits = 0
        total_search_simulations_used = 0
        total_search_fresh_simulations_used = 0
        total_search_inherited_visit_credit = 0
        total_search_simulations_budget = 0
        total_search_samples = 0
        total_search_difficulty_samples = 0
        total_search_difficulty_sum = 0.0
        total_search_difficulty_sq_sum = 0.0
        total_search_difficulty_budget_cross_sum = 0.0
        total_search_budget_sq_sum = 0.0
        total_tree_reuse_attempts = 0
        total_tree_reuse_hits = 0
        total_tree_inherited_visits = 0
        total_tree_reuse_credit_samples = 0
        total_tree_reuse_quality_sum = 0.0
        total_tree_reuse_candidate_coverage_sum = 0.0
        total_tree_reuse_visited_prior_mass_sum = 0.0
        total_tree_reuse_fresh_floor_sum = 0
        total_tree_reuse_scout_stability_sum = 0.0
        total_tree_reuse_scout_extra_credit_sum = 0
        total_tree_reuse_scout_reduced_count = 0
        total_shared_tree_searches = 0
        total_hard_start_games = 0
        search_simulations_used_samples = []
        search_simulations_budget_samples = []
        total_mcts_quality_stats = {
            'mcts_prior_agreement_samples': 0.0,
            'mcts_prior_agreement_sum': 0.0,
            'mcts_prior_changed_count': 0.0,
            'mcts_prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'mcts_policy_kl_sum': 0.0,
            'mcts_top_visit_prob_sum': 0.0,
            'mcts_visit_gap_sum': 0.0,
            'mcts_visit_entropy_sum': 0.0,
            'mcts_good_target_count': 0.0,
            'mcts_explored_prior_mass_sum': 0.0,
            'mcts_visited_move_count_sum': 0.0,
            'mcts_legal_move_count_sum': 0.0,
            'mcts_visit_coverage_ratio_sum': 0.0,
            'mcts_q_delta_samples': 0.0,
            'mcts_q_delta_sum': 0.0,
            'mcts_q_delta_values': [],
            'mcts_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_changed_to_lower_q_count': 0.0,
            'mcts_changed_to_higher_q_count': 0.0,
            'mcts_changed_q_delta_samples': 0.0,
            'mcts_changed_q_delta_sum': 0.0,
            'mcts_changed_q_delta_values': [],
            'mcts_changed_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_policy_uptake_samples': 0.0,
            'mcts_policy_uptake_weight_sum': 0.0,
            'mcts_policy_uptake_low_count': 0.0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            total_mcts_quality_stats[f'mcts_phase_{phase}_samples'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_changed_count'] = 0.0
        opponent_source_counts = {}
        opponent_source_results = {}

        batch_plan_labels = list(self.opponent_plan_labels[self._plan_cursor:self._plan_cursor + num_games])
        self._plan_cursor += num_games
        positions, lengths, batch_stats = self._play_batch(
            num_games,
            batch_plan_labels=batch_plan_labels,
            game_job_queue=game_job_queue,
            stream_task_id=stream_task_id,
            initial_game_jobs=initial_game_jobs,
            completed_chunk_callback=completed_chunk_callback,
            completed_chunk_games=completed_chunk_games,
        )
        all_positions.extend(positions)
        game_lengths.extend(lengths)
        total_dropped_positions += int(batch_stats.get('dropped_positions', 0))
        total_truncated_games += int(batch_stats.get('truncated_games', 0))
        total_claimable_draw_ended_games += int(batch_stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum += int(batch_stats.get('completed_length_sum', 0))
        total_truncated_length_sum += int(batch_stats.get('truncated_length_sum', 0))
        total_completed_white_wins += int(batch_stats.get('completed_white_wins', 0))
        total_completed_black_wins += int(batch_stats.get('completed_black_wins', 0))
        total_completed_draws += int(batch_stats.get('completed_draws', 0))
        total_decisive_games += int(batch_stats.get('decisive_games', 0))
        total_decisive_length_sum += int(batch_stats.get('decisive_length_sum', 0))
        total_curriculum_dropped_positions += int(batch_stats.get('curriculum_dropped_positions', 0))
        total_cap_dropped_positions += int(batch_stats.get('cap_dropped_positions', 0))
        total_resigned_games += int(batch_stats.get('resigned_games', 0))
        total_syzygy_ended_games += int(batch_stats.get('syzygy_ended_games', 0))
        total_syzygy_probe_positions += int(batch_stats.get('syzygy_probe_positions', 0))
        total_syzygy_probe_hits += int(batch_stats.get('syzygy_probe_hits', 0))
        total_search_simulations_used += int(batch_stats.get('search_simulations_used_sum', 0))
        total_search_fresh_simulations_used += int(
            batch_stats.get('search_fresh_simulations_used_sum', 0)
        )
        total_search_inherited_visit_credit += int(
            batch_stats.get('search_inherited_visit_credit_sum', 0)
        )
        total_search_simulations_budget += int(batch_stats.get('search_simulations_budget_sum', 0))
        total_search_samples += int(batch_stats.get('search_samples', 0))
        total_search_difficulty_samples += int(batch_stats.get('search_difficulty_samples', 0))
        total_search_difficulty_sum += float(batch_stats.get('search_difficulty_sum', 0.0) or 0.0)
        total_search_difficulty_sq_sum += float(batch_stats.get('search_difficulty_sq_sum', 0.0) or 0.0)
        total_search_difficulty_budget_cross_sum += float(
            batch_stats.get('search_difficulty_budget_cross_sum', 0.0) or 0.0
        )
        total_search_budget_sq_sum += float(batch_stats.get('search_budget_sq_sum', 0.0) or 0.0)
        total_tree_reuse_attempts += int(batch_stats.get('tree_reuse_attempts', 0))
        total_tree_reuse_hits += int(batch_stats.get('tree_reuse_hits', 0))
        total_tree_inherited_visits += int(batch_stats.get('tree_inherited_visits_sum', 0))
        total_tree_reuse_credit_samples += int(
            batch_stats.get('tree_reuse_credit_samples', 0) or 0
        )
        total_tree_reuse_quality_sum += float(batch_stats.get('tree_reuse_quality_sum', 0.0) or 0.0)
        total_tree_reuse_candidate_coverage_sum += float(
            batch_stats.get('tree_reuse_candidate_coverage_sum', 0.0) or 0.0
        )
        total_tree_reuse_visited_prior_mass_sum += float(
            batch_stats.get('tree_reuse_visited_prior_mass_sum', 0.0) or 0.0
        )
        total_tree_reuse_fresh_floor_sum += int(
            batch_stats.get('tree_reuse_fresh_floor_sum', 0) or 0
        )
        total_tree_reuse_scout_stability_sum += float(
            batch_stats.get('tree_reuse_scout_stability_sum', 0.0) or 0.0
        )
        total_tree_reuse_scout_extra_credit_sum += int(
            batch_stats.get('tree_reuse_scout_extra_credit_sum', 0) or 0
        )
        total_tree_reuse_scout_reduced_count += int(
            batch_stats.get('tree_reuse_scout_reduced_count', 0) or 0
        )
        total_shared_tree_searches += int(batch_stats.get('shared_tree_searches', 0))
        total_hard_start_games += int(batch_stats.get('hard_start_games', 0))
        search_simulations_used_samples.extend(list(batch_stats.get('search_simulations_used_samples', []) or []))
        search_simulations_budget_samples.extend(list(batch_stats.get('search_simulations_budget_samples', []) or []))
        for key in total_mcts_quality_stats:
            if key in {'mcts_q_delta_values', 'mcts_changed_q_delta_values'}:
                total_mcts_quality_stats[key].extend(list(batch_stats.get(key, []) or []))
            elif key in {'mcts_q_delta_hist', 'mcts_changed_q_delta_hist'}:
                source_hist = list(batch_stats.get(key, []) or [])
                if len(source_hist) == _Q_DELTA_HIST_BINS:
                    for idx, count in enumerate(source_hist):
                        total_mcts_quality_stats[key][idx] += int(count or 0)
            else:
                total_mcts_quality_stats[key] += float(batch_stats.get(key, 0.0) or 0.0)
        for label, count in dict(batch_stats.get('opponent_source_counts', {}) or {}).items():
            opponent_source_counts[str(label)] = int(opponent_source_counts.get(str(label), 0)) + int(count)
        for label, stats in dict(batch_stats.get('opponent_source_results', {}) or {}).items():
            result_stats = opponent_source_results.setdefault(
                str(label),
                {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
            )
            result_stats['wins'] += int((stats or {}).get('wins', 0))
            result_stats['draws'] += int((stats or {}).get('draws', 0))
            result_stats['losses'] += int((stats or {}).get('losses', 0))
            result_stats['games'] += int((stats or {}).get('games', 0))

        total_games = int(batch_stats.get('total_games', len(game_lengths)) or 0)
        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        avg_search_simulations_used = (
            float(total_search_simulations_used) / float(total_search_samples)
            if total_search_samples > 0
            else 0.0
        )
        p10_search_simulations_used = (
            float(np.percentile(np.asarray(search_simulations_used_samples, dtype=np.float32), 10))
            if search_simulations_used_samples
            else 0.0
        )
        budget_arr = (
            np.asarray(search_simulations_budget_samples, dtype=np.float32)
            if search_simulations_budget_samples
            else None
        )
        p10_search_simulations_budget = float(np.percentile(budget_arr, 10)) if budget_arr is not None else 0.0
        p50_search_simulations_budget = float(np.percentile(budget_arr, 50)) if budget_arr is not None else 0.0
        p90_search_simulations_budget = float(np.percentile(budget_arr, 90)) if budget_arr is not None else 0.0
        min_search_simulations_budget = float(np.min(budget_arr)) if budget_arr is not None else 0.0
        max_search_simulations_budget = float(np.max(budget_arr)) if budget_arr is not None else 0.0
        mcts_quality_samples = int(total_mcts_quality_stats['mcts_prior_agreement_samples'])
        mcts_q_delta_samples = int(total_mcts_quality_stats['mcts_q_delta_samples'])
        mcts_q_delta_values = list(total_mcts_quality_stats.get('mcts_q_delta_values', []) or [])
        mcts_q_delta_hist = list(total_mcts_quality_stats.get('mcts_q_delta_hist', []) or [])
        mcts_changed_q_delta_samples = int(total_mcts_quality_stats['mcts_changed_q_delta_samples'])
        mcts_changed_q_delta_values = list(total_mcts_quality_stats.get('mcts_changed_q_delta_values', []) or [])
        mcts_changed_q_delta_hist = list(total_mcts_quality_stats.get('mcts_changed_q_delta_hist', []) or [])
        mcts_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_samples', 0.0) or 0.0)
            changed = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_changed_count', 0.0) or 0.0)
            mcts_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            mcts_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            mcts_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
        self.last_selfplay_stats = {
            'truncated_games': total_truncated_games,
            'completed_games': max(0, total_games - total_truncated_games),
            'dropped_positions': int(total_dropped_positions),
            'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
            'completed_length_sum': int(total_completed_length_sum),
            'truncated_length_sum': int(total_truncated_length_sum),
            'completed_white_wins': int(total_completed_white_wins),
            'completed_black_wins': int(total_completed_black_wins),
            'completed_draws': int(total_completed_draws),
            'decisive_games': int(total_decisive_games),
            'decisive_length_sum': int(total_decisive_length_sum),
            'curriculum_dropped_positions': int(total_curriculum_dropped_positions),
            'cap_dropped_positions': int(total_cap_dropped_positions),
            'resigned_games': int(total_resigned_games),
            'syzygy_ended_games': int(total_syzygy_ended_games),
            'syzygy_probe_positions': int(total_syzygy_probe_positions),
            'syzygy_probe_hits': int(total_syzygy_probe_hits),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
            'search_simulations_used_avg': float(avg_search_simulations_used),
            'search_simulations_budget_avg': (
                float(total_search_simulations_budget) / float(total_search_samples)
                if total_search_samples > 0
                else 0.0
            ),
            'search_simulations_budget_p10': float(p10_search_simulations_budget),
            'search_simulations_budget_p50': float(p50_search_simulations_budget),
            'search_simulations_budget_p90': float(p90_search_simulations_budget),
            'search_simulations_budget_min': float(min_search_simulations_budget),
            'search_simulations_budget_max': float(max_search_simulations_budget),
            'search_simulations_budget_target': float(
                self.num_simulations
            ),
            'search_simulations_used_p10': float(p10_search_simulations_used),
            'search_simulations_used_sum': int(total_search_simulations_used),
            'search_fresh_simulations_used_sum': int(total_search_fresh_simulations_used),
            'search_inherited_visit_credit_sum': int(total_search_inherited_visit_credit),
            'search_simulations_budget_sum': int(total_search_simulations_budget),
            'search_simulations_used_samples': list(search_simulations_used_samples),
            'search_simulations_budget_samples': list(search_simulations_budget_samples),
            'search_samples': int(total_search_samples),
            'search_difficulty_samples': int(total_search_difficulty_samples),
            'search_difficulty_sum': float(total_search_difficulty_sum),
            'search_difficulty_sq_sum': float(total_search_difficulty_sq_sum),
            'search_difficulty_budget_cross_sum': float(total_search_difficulty_budget_cross_sum),
            'search_budget_sq_sum': float(total_search_budget_sq_sum),
            'tree_reuse_attempts': int(total_tree_reuse_attempts),
            'tree_reuse_hits': int(total_tree_reuse_hits),
            'tree_inherited_visits_sum': int(total_tree_inherited_visits),
            'tree_reuse_credit_samples': int(total_tree_reuse_credit_samples),
            'tree_reuse_quality_sum': float(total_tree_reuse_quality_sum),
            'tree_reuse_candidate_coverage_sum': float(total_tree_reuse_candidate_coverage_sum),
            'tree_reuse_visited_prior_mass_sum': float(total_tree_reuse_visited_prior_mass_sum),
            'tree_reuse_fresh_floor_sum': int(total_tree_reuse_fresh_floor_sum),
            'tree_reuse_scout_stability_sum': float(total_tree_reuse_scout_stability_sum),
            'tree_reuse_scout_extra_credit_sum': int(total_tree_reuse_scout_extra_credit_sum),
            'tree_reuse_scout_reduced_count': int(total_tree_reuse_scout_reduced_count),
            'shared_tree_searches': int(total_shared_tree_searches),
            'hard_start_games': int(total_hard_start_games),
            'mcts_prior_agreement_samples': int(mcts_quality_samples),
            'mcts_prior_agreement_sum': float(total_mcts_quality_stats['mcts_prior_agreement_sum']),
            'mcts_prior_agreement_rate': (
                float(total_mcts_quality_stats['mcts_prior_agreement_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_count': int(total_mcts_quality_stats['mcts_prior_changed_count']),
            'mcts_prior_changed_rate': (
                float(total_mcts_quality_stats['mcts_prior_changed_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            **mcts_phase_stats,
            'mcts_policy_uptake_samples': int(total_mcts_quality_stats['mcts_policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(total_mcts_quality_stats['mcts_policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(total_mcts_quality_stats['mcts_policy_uptake_low_count'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(total_mcts_quality_stats['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_prior_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(total_mcts_quality_stats['mcts_policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(total_mcts_quality_stats['mcts_policy_kl_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(total_mcts_quality_stats['mcts_visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(total_mcts_quality_stats['mcts_visit_gap_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(total_mcts_quality_stats['mcts_visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(total_mcts_quality_stats['mcts_visit_entropy_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(total_mcts_quality_stats['mcts_good_target_count']),
            'mcts_good_target_rate': (
                float(total_mcts_quality_stats['mcts_good_target_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(total_mcts_quality_stats['mcts_visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(total_mcts_quality_stats['mcts_visited_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(total_mcts_quality_stats['mcts_legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(total_mcts_quality_stats['mcts_legal_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': int(mcts_q_delta_samples),
            'mcts_q_delta_sum': float(total_mcts_quality_stats['mcts_q_delta_sum']),
            'mcts_q_delta_hist': mcts_q_delta_hist,
            'mcts_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_q_delta_sum']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 10)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 10))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 50)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 50))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 90)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 90))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(total_mcts_quality_stats['mcts_changed_to_lower_q_count']),
            'mcts_changed_to_lower_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(total_mcts_quality_stats['mcts_changed_to_higher_q_count']),
            'mcts_changed_q_delta_samples': int(mcts_changed_q_delta_samples),
            'mcts_changed_q_delta_sum': float(total_mcts_quality_stats['mcts_changed_q_delta_sum']),
            'mcts_changed_q_delta_hist': mcts_changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_changed_q_delta_sum']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 10)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 10))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 50)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 50))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 90)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 90))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_higher_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'total_games': int(total_games),
            'profile': self._aggregate_engine_profile_stats(),
        }
        return all_positions, game_lengths

    def _report_progress(self):
        progress = self._progress_base + self._games_completed
        if self._progress_file is None:
            return
        try:
            with open(self._progress_file, 'w') as _pf:
                _pf.write(str(progress))
        except Exception:
            pass

    def _mark_game_completed(self, gs):
        if gs.get('_completion_reported', False):
            return
        gs['_completion_reported'] = True
        self._games_completed += 1
        if (
            self._games_completed == 1
            or (self._games_completed % self.progress_report_interval_games) == 0
        ):
            self._report_progress()

    def _finalize_completed_game_states(self, completed_game_states):
        """Convert finished games to replay rows without retaining them for the batch."""
        positions = []
        game_lengths = []
        stats = {
            'total_games': 0,
            'dropped_positions': 0,
            'truncated_games': 0,
            'claimable_draw_ended_games': 0,
            'adjudicated_games': 0,
            'syzygy_ended_games': 0,
            'syzygy_probe_positions': 0,
            'syzygy_probe_hits': 0,
            'resigned_games': 0,
            'completed_length_sum': 0,
            'truncated_length_sum': 0,
            'completed_white_wins': 0,
            'completed_black_wins': 0,
            'completed_draws': 0,
            'learner_wins': 0,
            'learner_draws': 0,
            'learner_losses': 0,
            'decisive_games': 0,
            'decisive_length_sum': 0,
            'curriculum_dropped_positions': 0,
            'cap_dropped_positions': 0,
            'hard_start_games': 0,
            'opponent_source_counts': {},
            'opponent_source_results': {},
        }
        history_positions = int(self.config.get('model', {}).get('history_positions', 0))

        for gs in completed_game_states:
            stats['total_games'] += 1
            stats['hard_start_games'] += int(bool(gs.get('_hard_start', False)))
            board = gs['board']
            opponent_label = str(
                gs.get('opponent_source_label', self.opponent_source_label) or "current"
            )
            source_counts = stats['opponent_source_counts']
            source_counts[opponent_label] = int(source_counts.get(opponent_label, 0)) + 1
            ended_by_claimable_draw = bool(gs.get('ended_by_auto_claim_draw', False))
            stats['claimable_draw_ended_games'] += int(ended_by_claimable_draw)
            adjudicated_result = gs.get('adjudicated_result')
            stats['adjudicated_games'] += int(adjudicated_result is not None)
            syzygy_result = gs.get('syzygy_result')
            stats['syzygy_ended_games'] += int(syzygy_result is not None)
            stats['syzygy_probe_positions'] += int(gs.get('syzygy_probe_positions', 0) or 0)
            stats['syzygy_probe_hits'] += int(gs.get('syzygy_probe_hits', 0) or 0)
            resigned_result = gs.get('resigned_result')
            stats['resigned_games'] += int(resigned_result is not None)
            result = (
                resigned_result
                if resigned_result is not None
                else adjudicated_result
                if adjudicated_result is not None
                else syzygy_result
                if syzygy_result is not None
                else ('1/2-1/2' if ended_by_claimable_draw else chess.result(board, claim_draw=False))
            )
            game_ply_len = int(gs.get('move_count', len(gs['game_history'])))
            game_lengths.append(game_ply_len)
            if result == '*':
                stats['truncated_games'] += 1
                stats['dropped_positions'] += len(gs['game_history'])
                stats['truncated_length_sum'] += game_ply_len
                continue

            if result == '1-0':
                outcome = 1.0
                stats['completed_white_wins'] += 1
                stats['decisive_games'] += 1
            elif result == '0-1':
                outcome = -1.0
                stats['completed_black_wins'] += 1
                stats['decisive_games'] += 1
            else:
                outcome = 0.0
                stats['completed_draws'] += 1

            if opponent_label in self.opponent_mcts_by_label:
                learner_color = gs.get('learner_color', chess.WHITE)
                learner_outcome = outcome if learner_color == chess.WHITE else -outcome
                result_stats = stats['opponent_source_results'].setdefault(
                    opponent_label,
                    {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
                )
                if learner_outcome > 0.0:
                    stats['learner_wins'] += 1
                    result_stats['wins'] += 1
                elif learner_outcome < 0.0:
                    stats['learner_losses'] += 1
                    result_stats['losses'] += 1
                else:
                    stats['learner_draws'] += 1
                    result_stats['draws'] += 1
                result_stats['games'] += 1

            stats['completed_length_sum'] += game_ply_len
            if outcome != 0.0:
                stats['decisive_length_sum'] += game_ply_len
            _, curriculum_dropped, cap_dropped = self._append_selected_positions_from_game(
                positions,
                gs,
                outcome,
                history_positions=history_positions,
            )
            stats['curriculum_dropped_positions'] += curriculum_dropped
            stats['cap_dropped_positions'] += cap_dropped

        return positions, game_lengths, stats

    def _play_batch(
        self,
        batch_size,
        batch_plan_labels=None,
        game_job_queue=None,
        stream_task_id=None,
        initial_game_jobs=None,
        completed_chunk_callback=None,
        completed_chunk_games=8,
    ):
        max_moves = self.max_moves

        streaming_games = game_job_queue is not None
        total_games_to_play = max(0, int(batch_size))
        active_limit = (
            self.max_batch_games_per_worker
            if streaming_games
            else min(self.max_batch_games_per_worker, total_games_to_play)
        )
        games_started = 0
        game_source_exhausted = False
        initial_jobs = list(initial_game_jobs or [])
        initial_job_cursor = 0
        game_states = []
        completed_game_states = []
        output_positions = []
        output_game_lengths = []
        finalized_game_stats = None

        def _merge_finalized_stats(target, source):
            if target is None:
                target = {
                    key: ({} if isinstance(value, dict) else 0)
                    for key, value in source.items()
                }
            for key, value in source.items():
                if key == 'opponent_source_counts':
                    for label, count in value.items():
                        target[key][label] = int(target[key].get(label, 0)) + int(count)
                elif key == 'opponent_source_results':
                    for label, row in value.items():
                        dst = target[key].setdefault(
                            label, {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0}
                        )
                        for field in ('wins', 'draws', 'losses', 'games'):
                            dst[field] += int(row.get(field, 0) or 0)
                else:
                    target[key] += int(value or 0)
            return target

        def _flush_completed_games(force=False):
            nonlocal finalized_game_stats
            limit = max(1, int(completed_chunk_games or 1))
            if not completed_game_states or (not force and len(completed_game_states) < limit):
                return
            states = list(completed_game_states)
            completed_game_states.clear()
            chunk_positions, chunk_lengths, chunk_stats = (
                self._finalize_completed_game_states(states)
            )
            chunk_job_ids = [
                int(gs['_stream_job_id'])
                for gs in states
                if gs.get('_stream_job_id') is not None
            ]
            finalized_game_stats = _merge_finalized_stats(finalized_game_stats, chunk_stats)
            if completed_chunk_callback is None:
                output_positions.extend(chunk_positions)
                output_game_lengths.extend(chunk_lengths)
            else:
                completed_chunk_callback(chunk_positions, chunk_lengths, chunk_job_ids)

        def _new_game_state(game_idx, game_spec=None):
            game_spec = dict(game_spec or {})
            game_opponent_label = (
                str(game_spec.get('opponent_label'))
                if game_spec.get('opponent_label') is not None
                else str(batch_plan_labels[game_idx])
                if batch_plan_labels is not None and game_idx < len(batch_plan_labels)
                else self.opponent_source_label
            )
            opponent_mcts = self.opponent_mcts_by_label.get(game_opponent_label)
            if game_opponent_label != "current" and opponent_mcts is None:
                if game_opponent_label not in self._warned_missing_opponent_labels:
                    print(
                        "Warning: opponent plan references "
                        f"'{game_opponent_label}', but this worker has no model for it. "
                        "Falling back to current."
                    )
                    self._warned_missing_opponent_labels.add(game_opponent_label)
                game_opponent_label = "current"
                opponent_mcts = None
            has_frozen_opponent = opponent_mcts is not None
            hard_start = game_spec.get('hard_start')
            if hard_start is None and game_idx < len(self.hard_start_positions):
                candidate = self.hard_start_positions[game_idx]
                if isinstance(candidate, dict) and candidate.get('fen'):
                    hard_start = candidate
            initial_board = chess.new_board(hard_start.get('fen')) if hard_start else chess.new_board()
            hard_history_fens = list((hard_start or {}).get('history_fens', []) or [])[-self.history_positions:]
            encoded_hard_history = []
            for history_fen in hard_history_fens:
                try:
                    encoded_hard_history.append(self.mcts._encode_history_entry(chess.new_board(history_fen)))
                except Exception:
                    continue
            fen_parts = chess.board_fen(initial_board).split()
            try:
                fullmove = max(1, int(fen_parts[5]))
            except (IndexError, TypeError, ValueError):
                fullmove = 1
            initial_ply = (fullmove - 1) * 2 + (0 if initial_board.turn == chess.WHITE else 1)
            gs = {
                'board': initial_board,
                'board_history': encoded_hard_history,
                'fen_history': hard_history_fens,
                'position_counts': {_board_position_key(initial_board): 1},
                'root': None,
                '_root_synced': False,
                'opponent_root': None,
                '_opponent_root_synced': False,
                'game_history': [],
                'move_count': initial_ply,
                'done': False,
                'white_advantage_streak': 0,
                'black_advantage_streak': 0,
                'adjudicated_result': None,
                'syzygy_result': None,
                'syzygy_probe_positions': 0,
                'syzygy_probe_hits': 0,
                'resigned_result': None,
                'resign_streak': 0,
                'resignation_disabled': bool(np.random.random() < self.resignation_disable_fraction),
                'opponent_source_label': game_opponent_label,
                'opponent_mcts': opponent_mcts,
                'has_frozen_opponent': has_frozen_opponent,
                'learner_color': (
                    chess.WHITE
                    if not has_frozen_opponent or not self.randomize_learner_color
                    else (chess.WHITE if np.random.random() < 0.5 else chess.BLACK)
                ),
                '_completion_reported': False,
                '_finalized_for_batch': False,
                '_hard_start': bool(hard_start),
                '_native_tree_key': id(initial_board),
                '_stream_job_id': game_spec.get('job_id'),
                # The global dispatcher id is stable across workers for this
                # iteration. Replay combines it with insertion iteration, so
                # ids may safely restart from zero in the next generation.
                '_replay_game_id': (
                    int(game_spec['job_id'])
                    if game_spec.get('job_id') is not None
                    else -1
                ),
            }
            if not hard_start:
                self._apply_opening_prefix(gs)
            return gs

        def _fill_active_slots():
            nonlocal games_started, game_source_exhausted, total_games_to_play, initial_job_cursor
            while len(game_states) < active_limit and not game_source_exhausted:
                game_spec = None
                if streaming_games:
                    if initial_job_cursor < len(initial_jobs):
                        game_spec = initial_jobs[initial_job_cursor]
                        initial_job_cursor += 1
                    else:
                        game_spec = game_job_queue.get()
                    if str(game_spec.get('task_id', '')) != str(stream_task_id or ''):
                        raise RuntimeError('Self-play game stream task-id mismatch.')
                    if game_spec.get('cmd') == 'end_game_stream':
                        game_source_exhausted = True
                        break
                    total_games_to_play += 1
                elif games_started >= total_games_to_play:
                    game_source_exhausted = True
                    break
                game_states.append(_new_game_state(games_started, game_spec=game_spec))
                games_started += 1

        def _retire_completed_games():
            if not game_states:
                return
            active = []
            for gs in game_states:
                if gs.get('done', False):
                    if not gs.get('_finalized_for_batch', False):
                        gs['_finalized_for_batch'] = True
                        completed_game_states.append(gs)
                        _flush_completed_games()
                else:
                    active.append(gs)
            game_states[:] = active
            _flush_completed_games()

        _fill_active_slots()

        # Initialize search statistics used across this batch
        search_stats = {
            'sim_used': 0,
            'fresh_sim_used': 0,
            'inherited_visit_credit': 0,
            'sim_budget': 0,
            'samples': 0,
            'samples_list': [],
            'budget_samples_list': [],
            'difficulty_samples': 0,
            'difficulty_sum': 0.0,
            'difficulty_sq_sum': 0.0,
            'difficulty_budget_cross_sum': 0.0,
            'budget_sq_sum': 0.0,
            'tree_reuse_attempts': 0,
            'tree_reuse_hits': 0,
            'tree_inherited_visits_sum': 0,
            'tree_reuse_credit_samples': 0,
            'tree_reuse_quality_sum': 0.0,
            'tree_reuse_candidate_coverage_sum': 0.0,
            'tree_reuse_visited_prior_mass_sum': 0.0,
            'tree_reuse_fresh_floor_sum': 0,
            'tree_reuse_scout_stability_sum': 0.0,
            'tree_reuse_scout_extra_credit_sum': 0,
            'tree_reuse_scout_reduced_count': 0,
            'shared_tree_searches': 0,
        }
        target_quality = {
            'samples': 0,
            'changed': 0,
            'agreement_sum': 0.0,
            'prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'policy_kl_sum': 0.0,
            'top_visit_prob_sum': 0.0,
            'visit_gap_sum': 0.0,
            'visit_entropy_sum': 0.0,
            'good_target_count': 0,
            'explored_prior_mass_sum': 0.0,
            'visited_move_count_sum': 0.0,
            'legal_move_count_sum': 0.0,
            'visit_coverage_ratio_sum': 0.0,
            'q_comparable': 0,
            'q_delta_sum': 0.0,
            'q_delta_values': [],
            'changed_to_lower_q': 0,
            'changed_to_higher_q': 0,
            'changed_q_comparable': 0,
            'changed_q_delta_sum': 0.0,
            'changed_q_delta_values': [],
            'policy_uptake_samples': 0,
            'policy_uptake_weight_sum': 0.0,
            'policy_uptake_low_count': 0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            target_quality[f'{phase}_samples'] = 0
            target_quality[f'{phase}_changed_count'] = 0

        def _accumulate_target_quality(search_metadata, board):
            if not isinstance(search_metadata, dict):
                return
            agree = search_metadata.get('prior_mcts_agree', None)
            if agree is None:
                return
            target_quality['samples'] += 1
            agree_value = float(agree)
            changed_top = agree_value < 0.5
            phase = self._mcts_phase_for_board(board)
            target_quality[f'{phase}_samples'] += 1
            if changed_top:
                target_quality[f'{phase}_changed_count'] += 1
            target_quality['agreement_sum'] += agree_value
            if changed_top:
                target_quality['changed'] += 1
            for meta_key, sum_key in [
                ('prior_top_visit_prob', 'prior_top_visit_prob_sum'),
                ('mcts_top_prior_prob', 'mcts_top_prior_prob_sum'),
                ('mcts_policy_kl', 'policy_kl_sum'),
                ('top_visit_prob', 'top_visit_prob_sum'),
                ('visit_gap', 'visit_gap_sum'),
                ('visit_entropy', 'visit_entropy_sum'),
                ('explored_prior_mass', 'explored_prior_mass_sum'),
            ]:
                value = search_metadata.get(meta_key, None)
                if value is not None:
                    target_quality[sum_key] += float(value)
            try:
                top_visit_prob = float(search_metadata.get('top_visit_prob', 0.0) or 0.0)
                visit_gap = float(search_metadata.get('visit_gap', 0.0) or 0.0)
            except (TypeError, ValueError):
                top_visit_prob = 0.0
                visit_gap = 0.0
            if (
                top_visit_prob >= float(self.mcts_good_target_min_top_visit_prob)
                and visit_gap >= float(self.mcts_good_target_min_visit_gap)
            ):
                target_quality['good_target_count'] += 1
            visited_count = search_metadata.get('visited_move_count', None)
            legal_count = search_metadata.get('legal_move_count', None)
            if visited_count is not None and legal_count is not None:
                visited_count = float(visited_count)
                legal_count = float(legal_count)
                target_quality['visited_move_count_sum'] += float(visited_count)
                target_quality['legal_move_count_sum'] += float(legal_count)
                if legal_count > 0:
                    target_quality['visit_coverage_ratio_sum'] += float(visited_count) / float(legal_count)
            q_delta = search_metadata.get('mcts_q_delta', None)
            if q_delta is not None:
                target_quality['q_comparable'] += 1
                q_delta_value = float(q_delta)
                target_quality['q_delta_sum'] += q_delta_value
                target_quality['q_delta_values'].append(q_delta_value)
                if changed_top:
                    target_quality['changed_q_comparable'] += 1
                    target_quality['changed_q_delta_sum'] += q_delta_value
                    target_quality['changed_q_delta_values'].append(q_delta_value)
                    if q_delta_value < -0.02:
                        target_quality['changed_to_lower_q'] += 1
                    elif q_delta_value > 0.02:
                        target_quality['changed_to_higher_q'] += 1

        while game_states:
            active_indices = []
            learner_indices = []
            grouped_opponent_indices = {}
            for i, gs in enumerate(game_states):
                if gs['done']:
                    continue
                board = gs['board']
                syzygy_t0 = time.perf_counter() if self.profile_enabled else None
                self._maybe_finish_with_syzygy(gs, board)
                if self.profile_enabled:
                    self._profile_add('syzygy_time', time.perf_counter() - syzygy_t0)
                    self._profile_inc('syzygy_calls', 1)
                if gs['done']:
                    continue

                active_indices.append(i)
                if not gs.get('has_frozen_opponent', False):
                    gs['_learner_turn_cache'] = True
                    learner_indices.append(i)
                    continue

                learner_turn = bool(board.turn == gs.get('learner_color', chess.WHITE))
                gs['_learner_turn_cache'] = learner_turn
                if learner_turn:
                    learner_indices.append(i)
                else:
                    label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
                    grouped_opponent_indices.setdefault(label, []).append(i)

            if not active_indices:
                _retire_completed_games()
                _fill_active_slots()
                if not game_states:
                    break
                continue

            visit_counts_by_index = {}

            def _run_search_for_indices(indices, mcts_ref, root_key, synced_key):
                if not indices:
                    return
                group_states = []
                for i in indices:
                    gs = game_states[i]
                    group_states.append([
                        gs['board'],
                        gs.get(root_key),
                        bool(gs.get(synced_key, False)),
                        gs['board_history'],
                        int(gs.get('move_count', 0) or 0),
                        gs.get('position_counts'),
                        None,
                        gs['_native_tree_key'],
                    ])
                visit_counts_list_group, search_metadata_group = mcts_ref.search_many(
                    group_states,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                    return_search_metadata=True,
                )
                for gs_idx, local_state, visit_counts, search_metadata in zip(
                    indices,
                    group_states,
                    visit_counts_list_group,
                    search_metadata_group,
                ):
                    gs = game_states[gs_idx]
                    gs[root_key] = local_state[1]
                    gs[synced_key] = bool(local_state[2])
                    visit_counts_by_index[gs_idx] = (visit_counts, search_metadata)
                    used = int(search_metadata.get('simulations_used', 0)) if isinstance(search_metadata, dict) else 0
                    search_stats['sim_used'] += used
                    if isinstance(search_metadata, dict):
                        search_stats['fresh_sim_used'] += int(
                            search_metadata.get('fresh_simulations_used', used) or 0
                        )
                        search_stats['inherited_visit_credit'] += int(
                            search_metadata.get('inherited_visit_credit', 0) or 0
                        )
                        budget = int(search_metadata.get('simulation_budget', self.num_simulations) or self.num_simulations)
                        search_stats['sim_budget'] += budget
                        search_stats['budget_samples_list'].append(budget)
                    else:
                        search_stats['fresh_sim_used'] += used
                        search_stats['sim_budget'] += int(self.num_simulations)
                        search_stats['budget_samples_list'].append(int(self.num_simulations))
                    search_stats['samples'] += 1
                    search_stats['samples_list'].append(used)
                    if isinstance(search_metadata, dict):
                        reuse_attempted = bool(search_metadata.get('tree_reuse_attempted', False))
                        tree_reused = bool(search_metadata.get('tree_reused', False))
                        search_stats['tree_reuse_attempts'] += int(reuse_attempted)
                        search_stats['tree_reuse_hits'] += int(tree_reused)
                        search_stats['tree_inherited_visits_sum'] += int(
                            search_metadata.get('tree_inherited_visits', 0) or 0
                        )
                        if tree_reused:
                            search_stats['tree_reuse_credit_samples'] += 1
                            search_stats['tree_reuse_quality_sum'] += float(
                                search_metadata.get('tree_reuse_quality', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_candidate_coverage_sum'] += float(
                                search_metadata.get('tree_reuse_candidate_coverage', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_visited_prior_mass_sum'] += float(
                                search_metadata.get('tree_reuse_visited_prior_mass', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_fresh_floor_sum'] += int(
                                search_metadata.get('tree_reuse_fresh_floor', budget) or budget
                            )
                            search_stats['tree_reuse_scout_stability_sum'] += float(
                                search_metadata.get('tree_reuse_scout_stability', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_scout_extra_credit_sum'] += int(
                                search_metadata.get('tree_reuse_scout_extra_credit', 0) or 0
                            )
                            search_stats['tree_reuse_scout_reduced_count'] += int(bool(
                                search_metadata.get('tree_reuse_scout_search_reduced', False)
                            ))
                        search_stats['shared_tree_searches'] += int(
                            self.share_trees and not bool(gs.get('has_frozen_opponent', False))
                        )
                        difficulty = float(search_metadata.get('dynamic_budget_difficulty', 0.0) or 0.0)
                        if math.isfinite(difficulty):
                            search_stats['difficulty_samples'] += 1
                            search_stats['difficulty_sum'] += difficulty
                            search_stats['difficulty_sq_sum'] += difficulty * difficulty
                            search_stats['difficulty_budget_cross_sum'] += difficulty * float(budget)
                            search_stats['budget_sq_sum'] += float(budget) * float(budget)
            if not self.opponent_mcts_by_label:
                _run_search_for_indices(active_indices, self.mcts, 'root', '_root_synced')
            else:
                _run_search_for_indices(learner_indices, self.mcts, 'root', '_root_synced')
                for label, indices in grouped_opponent_indices.items():
                    opponent_mcts = self.opponent_mcts_by_label.get(label)
                    if opponent_mcts is None:
                        continue
                    _run_search_for_indices(indices, opponent_mcts, 'opponent_root', '_opponent_root_synced')
            for idx in active_indices:
                gs = game_states[idx]
                board = gs['board']
                visit_payload = visit_counts_by_index.get(idx, None)
                if visit_payload is None:
                    continue
                visit_counts, search_metadata = visit_payload

                if self.temp_threshold > 0 and gs['move_count'] < self.temp_threshold:
                    temperature = self.temperature
                else:
                    temperature = 0.0
                learner_turn = bool(gs.get('_learner_turn_cache', True))
                game_opponent_mcts = gs.get('opponent_mcts')
                root_key = 'root' if learner_turn or game_opponent_mcts is None else 'opponent_root'
                synced_key = '_root_synced' if learner_turn or game_opponent_mcts is None else '_opponent_root_synced'
                root = gs.get(root_key)
                adjudication_t0 = time.perf_counter() if self.profile_enabled else None
                adjudicated_result = self._maybe_adjudicate_game(gs, root, board)
                if self.profile_enabled:
                    self._profile_add('adjudication_time', time.perf_counter() - adjudication_t0)
                    self._profile_inc('adjudication_calls', 1)
                if adjudicated_result is not None:
                    gs['adjudicated_result'] = adjudicated_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                resigned_result = self._maybe_resign_game(gs, root, board)
                if resigned_result is not None:
                    gs['resigned_result'] = resigned_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                move_t0 = time.perf_counter() if self.profile_enabled else None
                selected_move_override = (
                    search_metadata.get('selected_move_override')
                    if isinstance(search_metadata, dict)
                    else None
                )
                if selected_move_override in visit_counts:
                    # Gumbel noise already supplies self-play exploration; the
                    # Sequential-Halving winner is the action prescribed by the
                    # algorithm, independent of the legacy visit temperature.
                    move = selected_move_override
                else:
                    move = self._select_move_from_visits(visit_counts, temperature)
                if (
                    isinstance(search_metadata, dict)
                    and root is not None
                    and root.expanded
                    and root.edges is not None
                ):
                    played_edge_idx = root.edges._get_move_index(move)
                    if played_edge_idx is not None:
                        played_edge_visits = int(root.edges.visit_counts[int(played_edge_idx)])
                        if played_edge_visits > 0:
                            search_metadata['played_q'] = float(max(
                                -1.0,
                                min(
                                    1.0,
                                    -float(root.edges.value_sums[int(played_edge_idx)])
                                    / float(played_edge_visits),
                                ),
                            ))
                if self.profile_enabled:
                    self._profile_add('move_selection_time', time.perf_counter() - move_t0)
                    self._profile_inc('move_selection_calls', 1)
                store_policy_position = self._should_store_policy_position(
                    learner_turn,
                    game_opponent_mcts,
                    gs.get('opponent_source_label', self.opponent_source_label),
                )
                if store_policy_position:
                    _accumulate_target_quality(search_metadata, board)
                    policy_t0 = time.perf_counter() if self.profile_enabled else None
                    target_visit_counts = visit_counts
                    policy_is_probability_target = False
                    if isinstance(search_metadata, dict):
                        probability_target = search_metadata.get('policy_target_probs_override')
                        if isinstance(probability_target, dict) and probability_target:
                            target_visit_counts = probability_target
                            policy_is_probability_target = True
                        else:
                            override_visit_counts = search_metadata.get('policy_visit_counts_override')
                            if isinstance(override_visit_counts, dict) and override_visit_counts:
                                target_visit_counts = override_visit_counts
                    policy_visit_counts = (
                        self._prune_policy_target_weights(target_visit_counts)
                        if policy_is_probability_target
                        else self._prune_policy_target_visits(target_visit_counts)
                    )
                    policy_indices, policy_values = _build_sparse_policy_target_from_visits(policy_visit_counts, board)
                    if root is not None:
                        # The root was expanded by this search, so its cached
                        # legal indices are exactly the current board's legal
                        # mask.  Reusing them avoids regenerating legal moves
                        # and remapping every move for replay storage.
                        _, cached_legal_indices = root.get_legal_moves_and_indices()
                        legal_indices_full = torch.from_numpy(
                            cached_legal_indices.astype(np.int16, copy=False)
                        )
                    else:
                        legal_indices_full = torch.tensor(
                            [move_to_index(legal_move, board) for legal_move in chess.legal_moves(board)],
                            dtype=torch.int16,
                        )
                    history_count = len(gs['board_history'])
                    importance_score = self._compute_position_importance(
                        board,
                        move,
                        policy_visit_counts,
                        root,
                        search_metadata=search_metadata,
                    )
                    # Every completed Gumbel root is one policy example. Its
                    # uncertainty is already encoded by the soft improved-policy
                    # target, so confidence must not scale CE a second time.
                    policy_weight = 1.0
                    search_changed_top, search_q_delta = (
                        _search_correction_metadata(search_metadata)
                    )
                    policy_uptake_weight = policy_weight
                    target_quality['policy_uptake_samples'] += 1
                    target_quality['policy_uptake_weight_sum'] += policy_uptake_weight
                    replay_source_code = _replay_source_code(
                        learner_turn,
                        game_opponent_mcts,
                        gs.get('opponent_source_label', self.opponent_source_label),
                    )
                    # PCR no longer splits roots into full/fast searches. Every
                    # completed root now owns valid search metadata; preserve it
                    # directly in replay instead of gating it on the removed
                    # ``full_search`` flag.
                    replay_search_metadata = (
                        search_metadata if isinstance(search_metadata, dict) else {}
                    )
                    gs['game_history'].append((
                        history_count,
                        policy_indices,
                        policy_values,
                        board.turn,
                        importance_score,
                        policy_weight,
                        replay_source_code,
                        legal_indices_full,
                        chess.board_fen(board),
                        (
                            float(replay_search_metadata.get('root_value'))
                            if replay_search_metadata.get('root_value') is not None
                            else float('nan')
                        ),
                        tuple(gs.get('fen_history', [])[-self.history_positions:]),
                        search_changed_top,
                        search_q_delta,
                        float(replay_search_metadata.get('best_q'))
                        if replay_search_metadata.get('best_q') is not None else float('nan'),
                        float(replay_search_metadata.get('played_q'))
                        if replay_search_metadata.get('played_q') is not None else float('nan'),
                        float(replay_search_metadata.get('orig_q'))
                        if replay_search_metadata.get('orig_q') is not None else float('nan'),
                        float(replay_search_metadata.get('policy_kld'))
                        if replay_search_metadata.get('policy_kld') is not None else float('nan'),
                        int(replay_search_metadata.get('search_visits', 0) or 0),
                    ))
                    if self.profile_enabled:
                        self._profile_add('policy_target_build_time', time.perf_counter() - policy_t0)
                        self._profile_inc('policy_target_build_calls', 1)

                # Update history BEFORE making the move
                # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
                gs['board_history'].append(self.mcts._encode_history_entry(board))
                gs.setdefault('fen_history', []).append(chess.board_fen(board))
                self._trim_history_cache(gs['board_history'])
                self._trim_history_cache(gs['fen_history'])

                # Keep both side-specific MCTS trees synchronized with the
                # actual game line. Mixed-opponent self-play otherwise starts
                # every move from a fresh root and produces noisier targets.
                if self.share_trees or gs['has_frozen_opponent']:
                    root_keys = [('root', '_root_synced')]
                    if gs['has_frozen_opponent']:
                        # Different networks must retain independent trees: their
                        # priors and Q values are not interchangeable.
                        root_keys.append(('opponent_root', '_opponent_root_synced'))
                    for candidate_root_key, candidate_synced_key in root_keys:
                        next_root, next_synced = self._advance_search_root(
                            gs.get(candidate_root_key),
                            move,
                        )
                        gs[candidate_root_key] = next_root
                        gs[candidate_synced_key] = next_synced
                else:
                    self._clear_game_search_state(gs)

                chess.apply_move(board, move)
                gs['move_count'] += 1
                _record_position_count(gs['position_counts'], board)

                forced_game_over = chess.is_game_over(board, claim_draw=False)
                auto_claim_draw = self._should_auto_claim_draw(board, gs['move_count'])
                if forced_game_over or auto_claim_draw or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    gs['ended_by_auto_claim_draw'] = auto_claim_draw
                    # Free up memory immediately
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
            _retire_completed_games()
            _fill_active_slots()

        _flush_completed_games(force=True)
        positions = output_positions
        game_lengths = output_game_lengths
        finalized_game_stats = finalized_game_stats or {}
        dropped_positions = int(finalized_game_stats.get('dropped_positions', 0))
        truncated_games = int(finalized_game_stats.get('truncated_games', 0))
        claimable_draw_ended_games = int(finalized_game_stats.get('claimable_draw_ended_games', 0))
        adjudicated_games = int(finalized_game_stats.get('adjudicated_games', 0))
        syzygy_ended_games = int(finalized_game_stats.get('syzygy_ended_games', 0))
        syzygy_probe_positions = int(finalized_game_stats.get('syzygy_probe_positions', 0))
        syzygy_probe_hits = int(finalized_game_stats.get('syzygy_probe_hits', 0))
        resigned_games = int(finalized_game_stats.get('resigned_games', 0))
        completed_length_sum = int(finalized_game_stats.get('completed_length_sum', 0))
        truncated_length_sum = int(finalized_game_stats.get('truncated_length_sum', 0))
        completed_white_wins = int(finalized_game_stats.get('completed_white_wins', 0))
        completed_black_wins = int(finalized_game_stats.get('completed_black_wins', 0))
        completed_draws = int(finalized_game_stats.get('completed_draws', 0))
        learner_wins = int(finalized_game_stats.get('learner_wins', 0))
        learner_draws = int(finalized_game_stats.get('learner_draws', 0))
        learner_losses = int(finalized_game_stats.get('learner_losses', 0))
        opponent_source_counts = dict(finalized_game_stats.get('opponent_source_counts', {}) or {})
        opponent_source_results = dict(finalized_game_stats.get('opponent_source_results', {}) or {})
        decisive_games = int(finalized_game_stats.get('decisive_games', 0))
        decisive_length_sum = int(finalized_game_stats.get('decisive_length_sum', 0))
        curriculum_dropped_positions = int(finalized_game_stats.get('curriculum_dropped_positions', 0))
        cap_dropped_positions = int(finalized_game_stats.get('cap_dropped_positions', 0))
        finalized_game_count = int(finalized_game_stats.get('total_games', 0))
        hard_start_games = int(finalized_game_stats.get('hard_start_games', 0))

        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        target_quality_samples = int(target_quality['samples'])
        target_quality_q_samples = int(target_quality['q_comparable'])
        target_quality_changed_q_samples = int(target_quality['changed_q_comparable'])
        q_delta_values = list(target_quality.get('q_delta_values', []) or [])
        q_delta_hist = _q_delta_histogram(q_delta_values)
        changed_q_delta_values = list(target_quality.get('changed_q_delta_values', []) or [])
        changed_q_delta_hist = _q_delta_histogram(changed_q_delta_values)
        target_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(target_quality.get(f'{phase}_samples', 0) or 0)
            changed = float(target_quality.get(f'{phase}_changed_count', 0) or 0)
            target_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            target_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            target_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
        return positions, game_lengths, {
            'total_games': finalized_game_count,
            'truncated_games': int(truncated_games),
            'dropped_positions': int(dropped_positions),
            'claimable_draw_ended_games': int(claimable_draw_ended_games),
            'adjudicated_games': int(adjudicated_games),
            'syzygy_ended_games': int(syzygy_ended_games),
            'syzygy_probe_positions': int(syzygy_probe_positions),
            'syzygy_probe_hits': int(syzygy_probe_hits),
            'search_simulations_used_sum': int(search_stats['sim_used']),
            'search_fresh_simulations_used_sum': int(search_stats['fresh_sim_used']),
            'search_inherited_visit_credit_sum': int(search_stats['inherited_visit_credit']),
            'search_simulations_budget_sum': int(search_stats['sim_budget']),
            'search_samples': int(search_stats['samples']),
            'search_simulations_used_samples': list(search_stats['samples_list']),
            'search_simulations_budget_samples': list(search_stats['budget_samples_list']),
            'search_difficulty_samples': int(search_stats['difficulty_samples']),
            'search_difficulty_sum': float(search_stats['difficulty_sum']),
            'search_difficulty_sq_sum': float(search_stats['difficulty_sq_sum']),
            'search_difficulty_budget_cross_sum': float(search_stats['difficulty_budget_cross_sum']),
            'search_budget_sq_sum': float(search_stats['budget_sq_sum']),
            'tree_reuse_attempts': int(search_stats['tree_reuse_attempts']),
            'tree_reuse_hits': int(search_stats['tree_reuse_hits']),
            'tree_inherited_visits_sum': int(search_stats['tree_inherited_visits_sum']),
            'tree_reuse_credit_samples': int(search_stats['tree_reuse_credit_samples']),
            'tree_reuse_quality_sum': float(search_stats['tree_reuse_quality_sum']),
            'tree_reuse_candidate_coverage_sum': float(
                search_stats['tree_reuse_candidate_coverage_sum']
            ),
            'tree_reuse_visited_prior_mass_sum': float(
                search_stats['tree_reuse_visited_prior_mass_sum']
            ),
            'tree_reuse_fresh_floor_sum': int(search_stats['tree_reuse_fresh_floor_sum']),
            'tree_reuse_scout_stability_sum': float(
                search_stats['tree_reuse_scout_stability_sum']
            ),
            'tree_reuse_scout_extra_credit_sum': int(
                search_stats['tree_reuse_scout_extra_credit_sum']
            ),
            'tree_reuse_scout_reduced_count': int(
                search_stats['tree_reuse_scout_reduced_count']
            ),
            'shared_tree_searches': int(search_stats['shared_tree_searches']),
            'hard_start_games': hard_start_games,
            'mcts_prior_agreement_samples': target_quality_samples,
            'mcts_prior_agreement_sum': float(target_quality['agreement_sum']),
            'mcts_prior_changed_count': int(target_quality['changed']),
            'mcts_prior_agreement_rate': (
                float(target_quality['agreement_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_rate': (
                float(target_quality['changed']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            **target_phase_stats,
            'mcts_policy_uptake_samples': int(target_quality['policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(target_quality['policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(target_quality['policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(target_quality['policy_uptake_weight_sum']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(target_quality['policy_uptake_low_count']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(target_quality['prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(target_quality['prior_top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(target_quality['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(target_quality['mcts_top_prior_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(target_quality['policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(target_quality['policy_kl_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(target_quality['top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(target_quality['top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(target_quality['visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(target_quality['visit_gap_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(target_quality['visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(target_quality['visit_entropy_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(target_quality['good_target_count']),
            'mcts_good_target_rate': (
                float(target_quality['good_target_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(target_quality['explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(target_quality['explored_prior_mass_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(target_quality['visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(target_quality['visited_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(target_quality['legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(target_quality['legal_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(target_quality['visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(target_quality['visit_coverage_ratio_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': target_quality_q_samples,
            'mcts_q_delta_sum': float(target_quality['q_delta_sum']),
            'mcts_q_delta_values': q_delta_values,
            'mcts_q_delta_hist': q_delta_hist,
            'mcts_q_delta_mean': (
                float(target_quality['q_delta_sum']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(q_delta_hist, 10)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(q_delta_hist, 50)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(q_delta_hist, 90)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(target_quality['changed_to_lower_q']),
            'mcts_changed_to_lower_q_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(target_quality['changed_to_higher_q']),
            'mcts_changed_q_delta_samples': int(target_quality_changed_q_samples),
            'mcts_changed_q_delta_sum': float(target_quality['changed_q_delta_sum']),
            'mcts_changed_q_delta_values': changed_q_delta_values,
            'mcts_changed_q_delta_hist': changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(target_quality['changed_q_delta_sum']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 10)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 50)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 90)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(target_quality['changed_to_higher_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'resigned_games': int(resigned_games),
            'completed_length_sum': int(completed_length_sum),
            'truncated_length_sum': int(truncated_length_sum),
            'completed_white_wins': int(completed_white_wins),
            'completed_black_wins': int(completed_black_wins),
            'completed_draws': int(completed_draws),
            'learner_wins': int(learner_wins),
            'learner_draws': int(learner_draws),
            'learner_losses': int(learner_losses),
            'decisive_games': int(decisive_games),
            'decisive_length_sum': int(decisive_length_sum),
            'curriculum_dropped_positions': int(curriculum_dropped_positions),
            'cap_dropped_positions': int(cap_dropped_positions),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
        }

    def _select_move_from_visits(self, visit_counts, temperature):
        return _select_move_from_visits_safe(visit_counts, temperature)
