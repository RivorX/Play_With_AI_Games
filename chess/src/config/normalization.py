"""Normalize readable staged YAML into the compact runtime contract."""

from __future__ import annotations

from copy import deepcopy


def _with_defaults(defaults, values):
    """Recursively inject internal defaults while preserving explicit overrides."""
    result = deepcopy(defaults)
    for key, value in (values or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _with_defaults(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


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

_PATH_DEFAULTS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'best_model_il': 'models/best_model_il.pt',
    'best_model_rl': 'models/best_model_rl.pt',
    'il_checkpoints_dir': 'models/IL',
    'rl_checkpoints_dir': 'models/RL',
}

_DATA_DEFAULTS = {
    'game_filters': {
        'enabled': True,
    },
    'positions_per_game': {
        'enabled': True,
        'selection_mode': 'smart',
    },
    'sample_dedup': {
        'enabled': True,
        'mode': 'fen',
        'include_turn': True,
        'include_history': True,
        'max_count': 1,
    },
    'target_selection': {
        'mode': 'count_aware',
        'soft_candidate_selection': {
            'enabled': True,
        },
    },
    'soft_targets': {
        'enabled': True,
        'source': 'raw_split',
        'mode': 'fen_no_counters',
        'include_turn': True,
        'include_history': False,
        'policy_rating_weight': {
            'enabled': True,
            'min_elo': 2300,
            'apply_to_value': True,
        },
    },
    'train_sampling': {
        'enabled': True,
    },
    'preprocess_threads': 'all',
    'soft_target_workers': 'auto',
    'soft_target_chunk_size': 'auto',
}

_AUTO_TUNE_DEFAULTS = {
    'use_for_rl': False,
    'cache_path': 'logs/il_auto_tune_cache.yaml',
    'dedicated_vram_utilization': 0.90,
    'max_batch_size': 24576,
    'batch_round_to': 256,
    'batch_selection': 'throughput',
    'throughput_candidate_count': 7,
    'throughput_tolerance': 0.03,
    'lr_test_min': 0.0002,
    'lr_test_max': 0.005,
    'lr_test_points': 10,
    'lr_test_steps': 20,
    'lr_min': 0.0002,
    'lr_max': 0.005,
}

_IL_DEFAULTS = {
    'lr_plateau_patience': 'auto',
    'lr_plateau_factor': 0.7,
    'lr_plateau_min_scale': 0.35,
    'lr_plateau_cooldown': 1,
    'lr_plateau_reset_patience': True,
    'swa_auto_min_epoch': 36,
    'swa_anneal_epochs': 5,
    'swa_min_updates_before_stop': 8,
    'swa_lr_ratio': 0.08,
    'swa_lr_max': 0.00012,
    'swa_bn_refresh_max_batches': 512,
}

_HARDWARE_DEFAULTS = {
    'device': 'cuda',
    'use_amp': True,
    'use_bfloat16': True,
    'use_compile': True,
    'num_workers': 4,
    'train_prefetch_factor': 1,
    'cuda_train_prefetch_queue_size': 2,
    'cuda_eval_prefetch_queue_size': 1,
    'dataloader_worker_restart_limit': 2,
    'dataloader_worker_failure_tail_tolerance_batches': 2,
    'dataloader_board_dtype': 'float16',
    'dataloader_return_numpy': True,
    'dataloader_block_shuffle': True,
    'dataloader_block_shuffle_size': 65536,
    'dataloader_train_in_order': False,
    'dataloader_val_in_order': False,
}

_CENTRAL_INFERENCE_DEFAULTS = {
    'enabled': True,
    'servers': 'auto',
    'auto_min_servers': 1,
    'auto_workers_per_server': 9,
    'auto_max_servers': 3,
    'max_batch_size': 384,
    'auto_batch_size': True,
    'flush_ms': 0.5,
    'transport_dtype': 'float16',
    'shared_memory_enabled': True,
    'shared_memory_slots_per_worker': 2,
    'shared_memory_slot_batch_size': 192,
    'cache_enabled': False,
    'cache_max_entries': 50000,
    'pinned_staging_enabled': False,
    'use_bfloat16': False,
    'timeout_s': 0,
    'stall_warning_s': 15,
    'load_timeout_s': 300,
    'use_compile': True,
    # Fixed compile buckets make cuDNN autotuning stable.  On the RTX 5060 Ti
    # paired 192-simulation self-play improved played positions/s by ~1.4%
    # and reduced descriptor queue wait by ~9.5%; the one-time warmup is
    # amortized by the persistent inference server.
    'cudnn_benchmark': True,
    'sync_timing': False,
}

_PLAY_DEFAULTS = {
    'match_workers': 'auto',
    'match_batch_games': 'auto',
    'match_max_moves': 220,
}

_LOGGING_DEFAULTS = {
    'log_level': 'INFO',
    'print_every': 10,
    'il_plot_smoothing_enabled': True,
    'il_plot_smoothing_alpha': 0.65,
    'il_plot_smoothing_min_points': 5,
}

_ELO_DEFAULTS = {
    'stockfish_path': 'stockfish',
    'stockfish_time_limit': 0.10,
    'max_moves': 150,
    'stockfish_threads': 1,
    'stockfish_hash_mb': 32,
    'prioritize_training': True,
    'reserve_dataloader_workers': True,
    'stockfish_priority': 'below_normal',
    'paired_openings_enabled': True,
    'nn_eval_workers': 4,
    'nn_eval_free_threads_utilization': 0.35,
    'il_async_process_restart_limit': 2,
    'batch_model_moves': True,
    'batch_raw_model_moves': False,
    'eval_elo_central_inference_enabled': True,
    'eval_elo_central_inference_servers': 1,
    'eval_elo_central_inference_max_batch_size': 64,
    'eval_elo_central_inference_use_compile': True,
    'eval_elo_central_client_groups': 1,
    'eval_elo_central_games_per_worker_chunk': 6,
    'eval_elo_central_target_chunks_per_group': 2,
    'eval_elo_central_max_chunk_games': 36,
    # Adaptive estimator implementation defaults. Stockfish supplies the
    # supported Elo range at runtime; no hand-maintained level list is needed.
    'adaptive_probe_games_per_level': 12,
    'adaptive_min_batch_games': 36,
    'adaptive_focus_games_per_level': 48,
    'adaptive_extra_games_per_level': 12,
    'adaptive_target_focus_levels': 3,
    'adaptive_max_total_games': 288,
    'adaptive_hard_max_total_games': 720,
    'adaptive_budget_extension_games': 72,
    'adaptive_target_standard_error': 35.0,
    'adaptive_min_games_for_se_stop': 144,
    'adaptive_refine_until_target_se': True,
    'paired_openings_max_plies': 6,
    'nn_eval_adaptive_probe_games_per_level': 12,
    'nn_eval_adaptive_min_batch_games': 36,
    'nn_eval_adaptive_focus_games_per_level': 64,
    'nn_eval_adaptive_extra_games_per_level': 16,
    'nn_eval_adaptive_max_total_games': 384,
    'nn_eval_adaptive_target_standard_error': 30.0,
    'nn_eval_adaptive_min_games_for_se_stop': 192,
    'mcts_eval_workers': 0,
    'mcts_eval_simulations': 192,
    'final_mcts_profile_multipliers': [0.5, 1.0, 2.0],
    'il_eval_every': 2,
}

_DEBUG_DEFAULTS = {
    'enabled': False,
    'il': {
        'profile_training': False,
        'print_profile_to_console': False,
        'log_gpu_memory': False,
        'print_batch0_diagnostics': False,
        'print_wdl_loss_diagnostics': False,
    },
    'rl': {
        'profile_training': False,
        'profile_mcts_detail': False,
        'profile_mcts_sample_rate': 64,
        'log_gpu_memory': False,
        'central_inference': {
            'verbose': False,
            'stall_warning_s': 15,
        },
    },
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
    'losses.policy': 'policy_loss_weight',
    'losses.value': 'value_loss_weight',
    'losses.value_scalar_aux': 'value_aux_scalar_loss_weight',
    'losses.moves_left': 'moves_left_loss_weight',
    'losses.search_q': 'search_q_loss_weight',
    'losses.policy_correction_rank_weight': 'policy_correction_rank_weight',
    'losses.policy_correction_rank_warmup_iterations': 'policy_correction_rank_warmup_iterations',
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
    'self_play_auto_claim_draw': True, 'self_play_claim_repetition_after_moves': 64,
    'self_play_claim_draw_after_moves': 120, 'syzygy_enabled': True,
    'syzygy_auto_download_enabled': True, 'self_play_progress_interval_games': 5,
    'self_play_opening_diversity_enabled': True,
    'self_play_device': 'cuda', 'self_play_worker_auto_multiplier': 1.50,
    'max_batch_games_per_worker': 48,
    'self_play_randomize_learner_color': True, 'self_play_store_frozen_best_positions': False,
    'mcts_gumbel_c_scale': 0.10, 'mcts_gumbel_q_range_floor': 0.25,
    'mcts_gumbel_target_temperature': 1.00,
    'mcts_reuse_tree': True, 'mcts_tree_reuse_visit_credit_enabled': True,
    'mcts_cache_history_tensors': True,
    'mcts_temperature': 0.0, 'mcts_temperature_threshold': 0,
    # Optimizer/replay mechanics.
    'train_dynamic_batch_size_enabled': True, 'train_batch_size_round_to': 256,
    'train_batch_size_min': 1024, 'train_batch_size_max': 6144,
    'train_epochs_per_iteration': 1, 'train_max_steps_per_iteration': 0,
    'no_decay_weight_decay': 0.0,
    'rl_diagnostic_ema_alpha': 0.35, 'use_lr_schedule': True,
    'use_augmentation': True,
    'augment_horizontal_flip': True, 'replay_buffer_bootstrap_positions_per_iteration': 'auto',
    'replay_buffer_ema_alpha': 0.15, 'replay_buffer_min_size': 4096,
    'replay_buffer_capacity_round_to': 256, 'replay_fp16': True,
    'replay_max_policy_targets': 218,
    'replay_dynamic_cap_enabled': True, 'replay_champion_fraction': 0.15,
    'replay_cap_fraction_decisive': 0.70, 'replay_cap_fraction_draw': 0.65,
    'replay_cap_min_positions': 24, 'replay_cap_max_positions': 120,
    'value_error_focus_fraction': 0.25, 'value_error_focus_max_multiplier': 1.50,
    'search_q_loss_weight': 0.20,
    'deblunder_threshold': 0.15, 'deblunder_width': 0.10,
    'deblunder_value_min_weight': 0.35, 'deblunder_policy_boost_max': 2.00,
    # Evaluation/promotion mechanics.
    'eval_worker_restart_limit': 2, 'eval_cpu_threads_to_reserve': 0, 'eval_torch_threads': 1,
    'eval_workers': 'auto', 'eval_batch_games': 32,
    'eval_parallel_worker_start': True, 'eval_parallel_worker_start_threads': 4,
    'eval_mcts_simulations_multiplier': 1.0, 'eval_max_moves': 320,
    'eval_central_inference_enabled': True,
    'eval_central_inference_min_games': 2, 'eval_auto_claim_draw': True,
    'eval_claim_repetition_after_moves': 70, 'eval_claim_draw_after_moves': 180,
    # Raw-NN safety is consumed by the promotion gate on MCTS-eval iterations.
    # Running it between those decisions only adds chart density.
    'eval_no_mcts_enabled': True, 'eval_no_mcts_every': 2,
    'eval_no_mcts_use_fixed_openings': True, 'eval_funnel_preliminary_simulations': 'auto',
    'eval_funnel_preliminary_simulations_multiplier': 1.0,
    'eval_funnel_later_simulations_multiplier': 1.0,
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
    # Promotion candidates always force a direct anchor match. There is no
    # periodic anchor diagnostic between promotion attempts.
    'anchor_eval_enabled': True,
    'anchor_eval_mcts_simulations': 'auto', 'anchor_eval_mcts_simulations_multiplier': 1.0,
    'anchor_eval_use_fixed_openings': True,
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
        config['data'] = deepcopy(_DATA_DEFAULTS)
        config['paths'] = _with_defaults(_PATH_DEFAULTS, config.get('paths', {}))
        return config

    has_staged_data = any(name in data for name in _DATA_PHASE_KEYS)
    if not has_staged_data:
        config['data'] = _with_defaults(_DATA_DEFAULTS, data)
        config['paths'] = _with_defaults(_PATH_DEFAULTS, config.get('paths', {}))
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

    config['data'] = _with_defaults(_DATA_DEFAULTS, normalized)
    config['paths'] = _with_defaults(_PATH_DEFAULTS, config.get('paths', {}))
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
    config = normalize_rl_config(normalize_data_config(config))
    config['auto_tune'] = _with_defaults(
        _AUTO_TUNE_DEFAULTS,
        config.get('auto_tune', {}),
    )
    config['imitation_learning'] = _with_defaults(
        _IL_DEFAULTS,
        config.get('imitation_learning', {}),
    )
    config['hardware'] = _with_defaults(
        _HARDWARE_DEFAULTS,
        config.get('hardware', {}),
    )
    config['central_inference'] = _with_defaults(
        _CENTRAL_INFERENCE_DEFAULTS,
        config.get('central_inference', {}),
    )
    config['play'] = _with_defaults(_PLAY_DEFAULTS, config.get('play', {}))
    config['logging'] = _with_defaults(
        _LOGGING_DEFAULTS,
        config.get('logging', {}),
    )
    config['elo_estimation'] = _with_defaults(
        _ELO_DEFAULTS,
        config.get('elo_estimation', {}),
    )
    config['debug'] = _with_defaults(_DEBUG_DEFAULTS, config.get('debug', {}))
    return config
