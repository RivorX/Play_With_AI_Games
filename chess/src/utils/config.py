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
