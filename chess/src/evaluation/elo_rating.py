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


def nearest_level_rating(
    opponent_elos: list[float],
    scores: list[float],
) -> tuple[float | None, float | None, int | None, int]:
    """Estimate strength from the tested level carrying the most local evidence.

    Stockfish ``UCI_Elo`` levels are not guaranteed to follow the textbook Elo
    slope under a fixed, short time control. The level closest to a 50% score is
    therefore the honest local calibration point; distant levels remain useful
    for bracketing but must not pull the reported rating away from that crossing.
    """
    if not scores or len(opponent_elos) != len(scores):
        return None, None, None, 0
    grouped: dict[int, list[float]] = {}
    for opponent, score in zip(opponent_elos, scores):
        grouped.setdefault(int(round(float(opponent))), []).append(float(score))
    if not grouped:
        return None, None, None, 0

    def local_key(item):
        level, level_scores = item
        n = len(level_scores)
        smoothed_score = (float(sum(level_scores)) + 0.5) / float(n + 1)
        return abs(smoothed_score - 0.5), -n, level

    level, level_scores = min(grouped.items(), key=local_key)
    rating, standard_error = local_performance_rating(float(level), level_scores)
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
