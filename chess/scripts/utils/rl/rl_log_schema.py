"""Versioned, purpose-specific CSV contracts for RL training logs."""

from __future__ import annotations

import csv


RL_LOG_SCHEMA_VERSION = 3


# One row per training iteration. This is the promotion/strength dashboard source,
# not a copy of replay and profiler telemetry.
RL_MAIN_COLUMNS = (
    "iteration", "schema_version",
    "avg_loss", "policy_loss", "value_loss", "learning_rate", "value_loss_weight",
    "mcts_q_selection_weight", "mcts_q_effective_weight",
    "temperature", "policy_top1_acc", "policy_top3_acc",
    "value_mae", "value_wdl_acc",
    "value_mae_opening", "value_mae_middlegame", "value_mae_endgame",
    "value_std_ratio_opening", "value_std_ratio_middlegame", "value_std_ratio_endgame",
    "eval_stage", "eval_games", "eval_wins", "eval_draws", "eval_losses", "eval_unresolved",
    "score_rate", "true_win_rate", "eval_score_lower_bound",
    "eval_score_rate_ema", "eval_true_win_rate_ema",
    "no_mcts_games", "no_mcts_score_rate", "no_mcts_win_rate",
    "no_mcts_score_lower_bound", "mcts_no_mcts_gap",
    "anchor_games", "anchor_score_rate", "anchor_true_win_rate", "anchor_score_lower_bound",
    "anchor_no_mcts_games", "anchor_no_mcts_score_rate", "anchor_no_mcts_score_lower_bound",
    "anchor_mcts_no_mcts_gap",
    "early_stop_streak", "promotion_candidate_streak", "early_stop_reset_reason",
    "rl_best_model",
    "estimated_elo_nn", "estimated_elo_nn_se", "estimated_elo_nn_ci95_low",
    "estimated_elo_nn_ci95_high", "estimated_elo_mcts", "estimated_elo_mcts_se",
    "estimated_elo_mcts_ci95_low", "estimated_elo_mcts_ci95_high",
    "estimated_elo_mcts_simulations",
)


# Replay, target and search behaviour. Evaluation and profiler fields intentionally
# do not live here; their canonical homes are the main and performance CSVs.
RL_DATA_QUALITY_COLUMNS = (
    "iteration", "schema_version", "timestamp",
    "positions_added", "replay_size", "replay_capacity", "replay_fill_rate",
    "train_batch_size", "train_steps", "train_selected_samples", "train_replay_coverage",
    "replay_decisive_fraction", "replay_draw_fraction", "replay_value_mean",
    "replay_value_std", "replay_value_positive_fraction", "replay_value_neutral_fraction",
    "replay_value_negative_fraction", "replay_value_skew",
    "policy_weight_mean", "policy_weight_p10", "policy_weight_low_fraction",
    "value_weight_mean", "value_weight_p10", "value_weight_low_fraction",
    "replay_source_learner_fraction", "replay_source_frozen_best_fraction",
    "replay_policy_weight_learner_share", "replay_policy_weight_frozen_best_share",
    "policy_target_len_mean", "policy_target_len_p90", "policy_target_entropy_mean",
    "policy_target_top1_prob_mean", "policy_target_top3_prob_mean",
    "policy_target_effective_moves", "policy_entropy_ratio",
    "sample_age_avg", "sample_age_p50", "sample_age_p90", "sample_age_new_fraction",
    "sample_age_le1_fraction", "iterations_since_promotion",
    "selfplay_completed_games", "selfplay_draw_rate", "selfplay_decisive_rate",
    "selfplay_auto_draw_rate", "selfplay_truncated_rate", "selfplay_avg_game_value",
    "selfplay_value_std", "selfplay_replay_storage_keep_rate",
    "opponent_current_planned_share", "opponent_current_actual_share",
    "opponent_current_games", "opponent_current_score_rate",
    "opponent_best_planned_share", "opponent_best_actual_share",
    "opponent_best_games", "opponent_best_score_rate", "opponent_mix_error",
    "opponent_promotion_transition_progress",
    "mcts_dirichlet_weight", "mcts_avg_sims", "mcts_avg_budget",
    "mcts_budget_utilization", "mcts_budget_p10", "mcts_budget_p90",
    "mcts_prior_agreement_rate", "mcts_prior_changed_rate",
    "mcts_changed_opening_rate", "mcts_changed_middlegame_rate",
    "mcts_changed_endgame_rate", "mcts_useful_change_rate", "mcts_harmful_change_rate",
    "mcts_changed_to_higher_q_rate", "mcts_changed_to_lower_q_when_changed_rate",
    "mcts_changed_q_delta_mean", "mcts_changed_q_delta_p10", "mcts_changed_q_delta_p90",
    "mcts_policy_kl_mean", "mcts_top_visit_prob_mean", "mcts_visit_gap_mean",
    "mcts_visit_entropy_mean", "mcts_good_target_rate", "mcts_explored_prior_mass_mean",
    "mcts_visited_move_count_mean", "mcts_legal_move_count_mean",
    "mcts_visit_coverage_ratio_mean",
)


# Throughput and bottleneck diagnosis. Raw totals are kept for stage attribution;
# latency keeps only request- and position-normalized forms used for decisions.
RL_PERFORMANCE_COLUMNS = (
    "iteration", "schema_version", "timestamp", "positions_per_sec",
    "iteration_total_time_s", "bottleneck_stage",
    "stage_setup_time_s", "stage_selfplay_time_s", "stage_replay_time_s",
    "stage_train_time_s", "stage_eval_log_time_s", "stage_checkpoint_time_s",
    "stage_gc_time_s", "avg_game_length",
    "mcts_avg_batch_size", "mcts_central_avg_batch_size", "mcts_gpu_busy_proxy_pct",
    "central_remote_wait_ms_per_request", "central_server_queue_wait_ms_per_request",
    "central_server_forward_ms_per_request", "central_server_total_ms_per_request",
    "mcts_worker_nn_wait_ms_per_position", "mcts_worker_nn_wait_ms_per_batch",
    "central_server_queue_wait_ms_per_position", "central_server_forward_ms_per_position",
    "central_server_total_ms_per_position",
    "mcts_search_many_time_s", "mcts_nn_inference_time_s", "mcts_nn_inference_calls",
    "mcts_nn_inference_batch_items", "mcts_avg_legal_moves_per_position",
    "result_queue_wait_ms", "mcts_search_selection_time_s", "mcts_search_backprop_time_s",
    "mcts_search_metadata_time_s", "mcts_batch_expand_eval_time_s",
    "mcts_batch_expand_eval_calls", "mcts_board_to_tensor_time_s",
    "mcts_batch_expand_legal_moves_time_s", "mcts_batch_expand_tensor_pack_time_s",
    "mcts_batch_expand_history_time_s", "mcts_batch_expand_input_pack_time_s",
    "mcts_batch_expand_legal_index_pack_time_s", "mcts_batch_expand_cpu_policy_time_s",
    "mcts_batch_expand_value_fanout_time_s", "mcts_policy_target_build_time_s",
    "mcts_move_selection_time_s", "mcts_adjudication_time_s", "mcts_syzygy_time_s",
)


def csv_row(columns, record):
    """Return a schema-ordered row while normalising missing values."""
    return ["" if record.get(column) is None else record.get(column, "") for column in columns]


def append_csv_record(path, columns, record):
    with open(path, "a", newline="") as handle:
        csv.writer(handle).writerow(csv_row(columns, record))
