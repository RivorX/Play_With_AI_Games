"""Pure Elo rating math shared by every evaluation execution path."""

from __future__ import annotations

import math


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def performance_rating(opponent_elos: list[float], scores: list[float]) -> float | None:
    """Maximum-likelihood performance rating for fractional game scores."""
    if not scores or len(opponent_elos) != len(scores):
        return None
    score_pct = float(sum(scores)) / float(len(scores))
    avg_opp = float(sum(opponent_elos)) / float(len(opponent_elos))
    if score_pct <= 0.0:
        return avg_opp - 800.0
    if score_pct >= 1.0:
        return avg_opp + 800.0
    lo, hi = avg_opp - 1000.0, avg_opp + 1000.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        expected = sum(expected_score(mid, opponent) for opponent in opponent_elos) / len(scores)
        if expected < score_pct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def elo_standard_error(opponent_elos: list[float], estimated_elo: float | None) -> float | None:
    if estimated_elo is None or not opponent_elos:
        return None
    scale = math.log(10.0) / 400.0
    information = 0.0
    for opponent in opponent_elos:
        p = expected_score(float(estimated_elo), float(opponent))
        information += (scale * scale) * max(1e-6, p * (1.0 - p))
    return None if information <= 0.0 else 1.0 / math.sqrt(information)


def local_performance_rating(opponent_elo: float, scores: list[float]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    n = len(scores)
    p = (float(sum(scores)) + 0.5) / float(n + 1)
    p = max(1e-6, min(1.0 - 1e-6, p))
    rating = float(opponent_elo) + 400.0 * math.log10(p / (1.0 - p))
    scale = math.log(10.0) / 400.0
    standard_error = 1.0 / max(1e-12, scale * math.sqrt(float(n) * p * (1.0 - p)))
    return rating, standard_error


def local_performance_rating_clustered(
    opponent_elo: float,
    scores: list[float],
    cluster_ids: list[object] | None = None,
) -> tuple[float | None, float | None]:
    """Return the usual local point estimate with a cluster-robust SE.

    Color-swapped games from one opening are deliberately paired.  Their
    scores are not independent samples, so use their score-sum residuals as
    independent clusters.  A cancelled/failed half of a pair remains a valid
    one-observation cluster instead of being silently discarded.
    """
    rating, iid_standard_error = local_performance_rating(opponent_elo, scores)
    if (
        rating is None
        or not cluster_ids
        or len(cluster_ids) != len(scores)
    ):
        return rating, iid_standard_error

    clusters: dict[object, list[float]] = {}
    for cluster_id, score in zip(cluster_ids, scores):
        clusters.setdefault(cluster_id, []).append(float(score))
    total_games = float(len(scores))
    p = (float(sum(scores)) + 0.5) / float(total_games + 1.0)
    p = max(1e-6, min(1.0 - 1e-6, p))
    cluster_count = len(clusters)
    cluster_size_square_sum = sum(
        float(len(cluster_scores) ** 2)
        for cluster_scores in clusters.values()
    )
    # A zero empirical between-cluster residual is not evidence of zero
    # uncertainty (notably all draws, or every pair scoring 1/2).  Add a weak
    # one-pseudocluster variance guard.  It remains below iid variance once
    # several independent opening clusters exhibit genuine anti-correlation,
    # but gives a single paired opening its full cluster-level uncertainty.
    guard_variance = (
        p * (1.0 - p) * cluster_size_square_sum / (total_games * total_games)
        / float(cluster_count + 1 if cluster_count > 1 else 1)
    )
    residual_sum_squares = sum(
        (sum(cluster_scores) - len(cluster_scores) * p) ** 2
        for cluster_scores in clusters.values()
    )
    if cluster_count >= 2:
        empirical_variance = (
            float(cluster_count) / float(cluster_count - 1)
            * residual_sum_squares
            / (total_games * total_games)
        )
    else:
        empirical_variance = 0.0
    score_variance = max(guard_variance, empirical_variance)
    scale = math.log(10.0) / 400.0
    standard_error = math.sqrt(max(0.0, score_variance)) / max(
        1e-12,
        scale * p * (1.0 - p),
    )
    return rating, standard_error


def nearest_level_rating(
    opponent_elos: list[float],
    scores: list[float],
    cluster_ids: list[object] | None = None,
) -> tuple[float | None, float | None, int | None, int]:
    """Estimate strength from the tested level carrying the most local evidence.

    Stockfish ``UCI_Elo`` levels are not guaranteed to follow the textbook Elo
    slope under a fixed, short time control. The level closest to a 50% score is
    therefore the honest local calibration point; distant levels remain useful
    for bracketing but must not pull the reported rating away from that crossing.
    """
    if not scores or len(opponent_elos) != len(scores):
        return None, None, None, 0
    grouped: dict[int, list[tuple[float, object | None]]] = {}
    valid_clusters = cluster_ids is not None and len(cluster_ids) == len(scores)
    for index, (opponent, score) in enumerate(zip(opponent_elos, scores)):
        cluster_id = cluster_ids[index] if valid_clusters else None
        grouped.setdefault(int(round(float(opponent))), []).append((float(score), cluster_id))
    if not grouped:
        return None, None, None, 0

    def local_key(item):
        level, observations = item
        n = len(observations)
        smoothed_score = (float(sum(score for score, _ in observations)) + 0.5) / float(n + 1)
        return abs(smoothed_score - 0.5), -n, level

    level, observations = min(grouped.items(), key=local_key)
    level_scores = [score for score, _ in observations]
    level_clusters = [cluster_id for _, cluster_id in observations]
    rating, standard_error = local_performance_rating_clustered(
        float(level),
        level_scores,
        level_clusters if valid_clusters else None,
    )
    return rating, standard_error, int(level), int(len(level_scores))


def elo_fit_diagnostics(
    opponent_elos: list[float],
    scores: list[float],
    estimated_elo: float | None,
) -> dict:
    """Diagnose how well a fixed-slope textbook Elo curve fits the ladder."""
    model_se = elo_standard_error(opponent_elos, estimated_elo)
    result = {
        "model_standard_error": model_se,
        "standard_error": model_se,
        "overdispersion": 1.0,
        "pearson_chi2": 0.0,
        "degrees_of_freedom": 0,
        "fit_warning": False,
        "calibration_warning": False,
    }
    if estimated_elo is None or not opponent_elos or len(opponent_elos) != len(scores):
        return result
    grouped: dict[int, list[float]] = {}
    for opponent, score in zip(opponent_elos, scores):
        grouped.setdefault(int(round(float(opponent))), []).append(float(score))
    if len(grouped) < 3:
        return result
    pearson = 0.0
    for level, level_scores in grouped.items():
        n = float(len(level_scores))
        expected = expected_score(float(estimated_elo), float(level))
        variance = max(1e-6, n * expected * (1.0 - expected))
        pearson += ((float(sum(level_scores)) - n * expected) ** 2) / variance
    degrees_of_freedom = max(1, len(grouped) - 1)
    raw_dispersion = pearson / float(degrees_of_freedom)
    overdispersion = max(1.0, raw_dispersion)
    result.update(
        {
            "standard_error": None if model_se is None else model_se * math.sqrt(overdispersion),
            "overdispersion": overdispersion,
            "pearson_chi2": pearson,
            "degrees_of_freedom": degrees_of_freedom,
            "fit_warning": raw_dispersion >= 1.5,
            "calibration_warning": raw_dispersion >= 1.5,
        }
    )
    return result
