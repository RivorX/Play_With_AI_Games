"""Fixed opponent assignment for RL self-play.

RL uses a stable current-vs-best curriculum.  Adaptive source reweighting,
recent snapshots, and anchor transitions made the generated data depend on
short, noisy matchup histories without improving the verified baseline.
"""

from collections import Counter

import numpy as np


def _safe_score_rate(wins, draws, losses):
    total = int(wins or 0) + int(draws or 0) + int(losses or 0)
    return (float(wins or 0) + 0.5 * float(draws or 0)) / total if total > 0 else 0.0


def _allocate_counts(total_games, current_fraction, best_fraction, has_best):
    """Return a rounded fixed current/best game allocation."""
    if total_games <= 0:
        return {"current": 0, "best": 0}
    if not has_best or best_fraction <= 0.0:
        return {"current": total_games, "best": 0}

    weight_total = max(1e-12, float(current_fraction) + float(best_fraction))
    best_games = int(round(total_games * float(best_fraction) / weight_total))
    best_games = max(0, min(total_games, best_games))
    return {"current": total_games - best_games, "best": best_games}


def _build_selfplay_opponent_assignments(
    rl_cfg,
    worker_specs,
    best_model_state=None,
):
    """Assign an exact fixed mix of current and frozen-best games to workers."""
    if not bool(rl_cfg.get("self_play_opponent_pool_enabled", False)) or not worker_specs:
        return {}, {}, {}

    current_fraction = max(0.0, float(rl_cfg.get("self_play_opponent_current_fraction", 0.30)))
    best_fraction = max(0.0, float(rl_cfg.get("self_play_opponent_best_fraction", 0.70)))
    total_games = sum(max(0, int(games)) for _, games in worker_specs)
    counts = _allocate_counts(total_games, current_fraction, best_fraction, best_model_state is not None)

    game_plan = ["current"] * counts["current"] + ["best"] * counts["best"]
    np.random.shuffle(game_plan)

    assignments = {}
    offset = 0
    for rank, games in sorted(worker_specs, key=lambda item: int(item[0])):
        game_count = max(0, int(games))
        labels = game_plan[offset:offset + game_count]
        offset += game_count
        assignments[int(rank)] = {
            "plan_labels": labels,
            "pool_entries": ([{"label": "best", "state": best_model_state}] if "best" in labels else []),
        }

    assigned = Counter(game_plan)
    source_weights = {
        "current": counts["current"] / total_games if total_games else 0.0,
        "best": counts["best"] / total_games if total_games else 0.0,
    }
    debug = {
        "source_weights": source_weights,
        "target_games": dict(counts),
        "assigned_games": dict(assigned),
        "assigned_total_games": int(sum(assigned.values())),
        "expected_total_games": int(total_games),
    }
    return assignments, dict(assigned), debug
