"""Homogeneous guarded-actor assignment for RL self-play.

Opponent diversity now lives in replay and evaluation. Mixing model weights
inside live self-play split central-inference batches and still allowed learner
collapse, so training workers always receive one actor for both colours.
"""


def _safe_score_rate(wins, draws, losses):
    total = int(wins or 0) + int(draws or 0) + int(losses or 0)
    return (float(wins or 0) + 0.5 * float(draws or 0)) / total if total > 0 else 0.0


def _build_selfplay_opponent_assignments(
    rl_cfg,
    worker_specs,
    best_model_state=None,
):
    """Assign the same guarded actor to both colours in every game."""
    if not worker_specs:
        return {}, {}, {}

    total_games = sum(max(0, int(games)) for _, games in worker_specs)
    assignments = {}
    for rank, games in sorted(worker_specs, key=lambda item: int(item[0])):
        game_count = max(0, int(games))
        assignments[int(rank)] = {
            "plan_labels": ["current"] * game_count,
            "pool_entries": [],
        }

    counts = {"current": int(total_games), "best": 0}
    source_weights = {"current": 1.0 if total_games else 0.0, "best": 0.0}
    debug = {
        "source_weights": source_weights,
        "target_games": dict(counts),
        "assigned_games": dict(counts),
        "assigned_total_games": int(total_games),
        "expected_total_games": int(total_games),
    }
    return assignments, dict(counts), debug
