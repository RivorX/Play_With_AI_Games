"""Configuration normalization for the staged IL data pipeline."""

from __future__ import annotations


# The YAML is grouped by the pipeline stage that owns each knob.  Runtime code
# still receives the compact legacy-shaped mapping, so cache and training code
# do not need to know presentation details.
_DATA_PHASE_KEYS = {
    'phase_1_binary': ('min_elo', 'max_games', 'max_moves_per_game', 'game_filters'),
    'phase_2_split': ('train_split',),
    'phase_3_positions': ('positions_per_game',),
    'phase_4_deduplication': ('sample_dedup',),
    'phase_5_training_pool': ('target_positions', 'target_selection'),
    'phase_6_soft_targets': ('soft_targets',),
    'phase_7_epoch_sampling': ('train_sampling',),
}


_RL_SECTION_PREFIXES = {
    'run': '',
    'self_play': 'self_play_',
    'search': 'mcts_',
    'training': '',
    'replay': 'replay_',
    'evaluation': '',
    'losses': '',
}

_RL_KEY_ALIASES = {
    'self_play.max_batch_games_per_worker': 'max_batch_games_per_worker',
    'self_play.syzygy_paths': 'syzygy_paths',
    'search.playout_cap_enabled': 'mcts_playout_cap_randomization_enabled',
    'losses.policy': 'policy_loss_weight',
    'losses.value': 'value_loss_weight',
    'losses.value_scalar_aux': 'value_aux_scalar_loss_weight',
    'losses.moves_left': 'moves_left_loss_weight',
    'losses.search_q': 'search_q_loss_weight',
    'losses.search_error': 'search_error_loss_weight',
    'losses.value_error_focus_fraction': 'value_error_focus_fraction',
    'losses.value_error_focus_max_multiplier': 'value_error_focus_max_multiplier',
    'losses.wdl_label_smoothing': 'value_wdl_label_smoothing',
}


# Stable implementation choices intentionally hidden from YAML. They keep the
# staged presentation small while preserving the exact current RL behavior.
_RL_DEFAULTS = {
    # Self-play/search runtime.
    'self_play_worker_cap_multiplier': 1.7, 'self_play_torch_threads': 1,
    'self_play_history_fp16': True, 'self_play_central_inference_enabled': True,
    'self_play_share_trees': True,
    'self_play_compile_lock_timeout_s': 900, 'self_play_save_every_games': 1,
    'use_batch_selfplay': True, 'self_play_dispatch_chunk_games': 96,
    'self_play_auto_claim_draw': True, 'self_play_claim_repetition_after_moves': 64,
    'self_play_claim_draw_after_moves': 120, 'syzygy_enabled': True,
    'syzygy_auto_download_enabled': True, 'self_play_progress_interval_games': 5,
    'self_play_opening_diversity_enabled': True,
    'self_play_randomize_learner_color': True, 'self_play_store_frozen_best_positions': False,
    'mcts_gumbel_c_scale': 0.10, 'mcts_gumbel_q_range_floor': 0.25,
    'mcts_gumbel_target_temperature': 1.20,
    'mcts_reuse_tree': True, 'mcts_tree_reuse_visit_credit_enabled': True,
    'mcts_cache_history_tensors': True,
    # Optimizer/replay mechanics.
    'train_dynamic_batch_size_enabled': True, 'train_batch_size_round_to': 256,
    'train_epochs_per_iteration': 1, 'train_max_steps_per_iteration': 0,
    'no_decay_weight_decay': 0.0,
    'rl_diagnostic_ema_alpha': 0.35, 'use_lr_schedule': True,
    'use_augmentation': True,
    'augment_horizontal_flip': True, 'replay_buffer_bootstrap_positions_per_iteration': 'auto',
    'replay_buffer_ema_alpha': 0.15, 'replay_buffer_min_size': 4096,
    'replay_buffer_capacity_round_to': 256, 'replay_fp16': True,
    'replay_dynamic_cap_enabled': True, 'replay_champion_fraction': 0.15,
    'value_error_focus_fraction': 0.25, 'value_error_focus_max_multiplier': 1.50,
    'search_q_loss_weight': 0.20, 'search_error_loss_weight': 0.05,
    'search_value_blend_max': 0.25, 'search_value_uncertainty_temperature': 0.35,
    'deblunder_threshold': 0.15, 'deblunder_width': 0.10,
    'deblunder_value_min_weight': 0.35, 'deblunder_policy_boost_max': 1.35,
    # Evaluation/promotion mechanics.
    'eval_worker_restart_limit': 2, 'eval_cpu_threads_to_reserve': 0, 'eval_torch_threads': 1,
    'eval_central_inference_enabled': True,
    'eval_central_inference_min_games': 2, 'eval_auto_claim_draw': True,
    'eval_claim_repetition_after_moves': 70, 'eval_claim_draw_after_moves': 180,
    # Raw-NN safety is consumed by the actor/promotion gate on MCTS-eval
    # iterations. Running it between those decisions only adds chart density.
    'eval_no_mcts_enabled': True, 'eval_no_mcts_every': 2,
    'eval_no_mcts_use_fixed_openings': True, 'eval_funnel_preliminary_simulations': 'auto',
    'eval_mcts_dynamic_budget_enabled': True,
    'eval_mcts_dynamic_budget_min_fraction': 5.0 / 6.0,
    'eval_mcts_dynamic_budget_difficulty_threshold': 0.75,
    'eval_mcts_dynamic_budget_chunk': 16,
    'eval_funnel_preliminary_simulations_multiplier': 1.0,
    # Funnel stages use one search budget and disjoint paired openings, so every
    # completed game contributes to the cumulative decision.
    'eval_funnel_preliminary_true_win_rate': 0.0,
    'eval_funnel_medium_simulations': 'auto', 'eval_funnel_medium_true_win_rate': 0.02,
    'eval_funnel_advanced_simulations': 'auto', 'eval_fixed_openings_enabled': True,
    'eval_fixed_openings_pair_games': True, 'eval_fixed_openings_max_plies': 6,
    'promotion_stat_gate_enabled': True, 'promotion_stat_gate_z': 1.28,
    'promotion_score_lower_bound_min': 0.50, 'promotion_require_anchor_non_regression': True,
    'promotion_no_mcts_gate_enabled': True, 'promotion_no_mcts_score_rate_min': 0.40,
    'promotion_no_mcts_upper_bound_min': 0.50,
    'promotion_anchor_min_score_rate': 0.50, 'promotion_anchor_min_true_win_rate': 0.0,
    'promotion_anchor_no_mcts_gate_enabled': True,
    'promotion_anchor_no_mcts_score_rate_min': 0.50,
    'promotion_anchor_no_mcts_score_lower_bound_min': 0.47,
    'anchor_no_mcts_max_games': 512,
    # Promotion candidates always force an anchor match. The slower periodic
    # diagnostic is only for drift visibility after best diverges from anchor.
    'anchor_eval_enabled': True, 'anchor_eval_every': 8,
    'anchor_eval_mcts_simulations': 'auto', 'anchor_eval_mcts_simulations_multiplier': 1.0,
    'anchor_eval_use_fixed_openings': True,
    'actor_gate_enabled': True, 'actor_gate_score_rate_min': 0.50,
    'actor_gate_no_mcts_score_rate_min': 0.43,
    'value_metric_weight_min': 0.20,
    'value_metric_max_fullmove': 80, 'value_phase_opening_max_fullmove': 12,
    'value_phase_endgame_min_fullmove': 40, 'value_phase_max_fullmove': 120,
}


def normalize_data_config(config):
    """Flatten staged ``data:`` YAML into the mapping consumed by training code.

    Old flat configs continue to work.  New staged values take precedence when
    both forms are present, which makes an intentional migration unambiguous.
    The input mapping is updated in place and returned for convenient use just
    after YAML loading.
    """
    if not isinstance(config, dict):
        return config
    data = config.get('data')
    if not isinstance(data, dict):
        return config

    has_staged_data = any(name in data for name in _DATA_PHASE_KEYS)
    if not has_staged_data:
        return config

    normalized = dict(data)
    for phase_name, keys in _DATA_PHASE_KEYS.items():
        phase_cfg = data.get(phase_name, {})
        normalized.pop(phase_name, None)
        if not isinstance(phase_cfg, dict):
            continue
        for key in keys:
            if key in phase_cfg:
                normalized[key] = phase_cfg[key]

    runtime_cfg = data.get('runtime', {})
    normalized.pop('runtime', None)
    if isinstance(runtime_cfg, dict):
        normalized.update(runtime_cfg)

    config['data'] = normalized
    return config


def normalize_rl_config(config):
    """Flatten the grouped RL YAML and inject stable runtime defaults."""
    if not isinstance(config, dict):
        return config
    rl_cfg = config.get('reinforcement_learning')
    if not isinstance(rl_cfg, dict):
        return config

    has_grouped_rl = any(section in rl_cfg for section in _RL_SECTION_PREFIXES)
    if not has_grouped_rl:
        return config

    normalized = dict(_RL_DEFAULTS)
    for section, prefix in _RL_SECTION_PREFIXES.items():
        values = rl_cfg.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            runtime_key = _RL_KEY_ALIASES.get(f'{section}.{key}', f'{prefix}{key}')
            normalized[runtime_key] = value

    stockfish_cfg = rl_cfg.get('stockfish_elo')
    if isinstance(stockfish_cfg, dict):
        normalized['stockfish_elo'] = stockfish_cfg

    normalized['mcts_batch_size'] = int(normalized.get('mcts_simulations', 192))
    config['reinforcement_learning'] = normalized
    return config


def normalize_config(config):
    """Normalize every grouped presentation section used by the runtime."""
    return normalize_rl_config(normalize_data_config(config))
