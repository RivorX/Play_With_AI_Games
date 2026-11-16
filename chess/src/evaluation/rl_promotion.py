"""Shared statistical contracts for RL checkpoint selection and promotion.

MCTS selector and independent confirmation decide promotion. Raw-NN results
remain diagnostic and are never pooled into an MCTS promotion decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from statistics import NormalDist


def empty_eval_stats():
    return {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "unresolved": 0,
        "score_rate": 0.0,
        "win_rate": 0.0,
        "draw_rate": 0.0,
        "loss_rate": 0.0,
        "num_games": 0,
        "resolved_games": 0,
    }


def safe_score_rate(wins, draws, losses):
    total = int(wins or 0) + int(draws or 0) + int(losses or 0)
    return (
        (float(wins or 0) + 0.5 * float(draws or 0)) / float(total)
        if total > 0 else 0.0
    )


def score_rate_standard_error(score_rate, num_games, standard_error=None):
    games = max(1.0, float(num_games or 1))
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    if standard_error is None:
        return math.sqrt(score * (1.0 - score) / games)
    error = float(standard_error)
    # Invalid telemetry is missing evidence, never zero uncertainty.
    return max(0.0, error) if math.isfinite(error) else float("inf")


def score_rate_lower_bound(score_rate, num_games, z=1.28, standard_error=None):
    """Conservative one-sided normal lower bound for a [0, 1] game score."""
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    return max(0.0, score - z_value * score_rate_standard_error(
        score, num_games, standard_error
    ))


def score_rate_upper_bound(score_rate, num_games, z=1.28, standard_error=None):
    """One-sided normal upper bound paired with ``score_rate_lower_bound``."""
    score = max(0.0, min(1.0, float(score_rate or 0.0)))
    z_value = max(0.0, float(z or 0.0))
    return min(1.0, score + z_value * score_rate_standard_error(
        score, num_games, standard_error
    ))


def paired_score_standard_error(game_scores, opening_count=None):
    """Estimate score uncertainty from paired colors, clustered by opening."""
    pairs = defaultdict(list)
    for game_idx, score in dict(game_scores or {}).items():
        pairs[int(game_idx) // 2].append(float(score))
    indexed_pair_scores = [
        (pair_idx, sum(values) / 2.0)
        for pair_idx, values in pairs.items()
        if len(values) == 2
    ]
    pair_scores = [score for _, score in indexed_pair_scores]
    pair_count = len(pair_scores)
    try:
        opening_count = int(opening_count) if opening_count is not None else 0
    except (TypeError, ValueError):
        opening_count = 0
    if opening_count > 0:
        clustered = defaultdict(list)
        for pair_idx, score in indexed_pair_scores:
            clustered[int(pair_idx) % opening_count].append(float(score))
        independent_scores = [
            sum(values) / len(values) for values in clustered.values()
        ]
    else:
        independent_scores = pair_scores
    independent_count = len(independent_scores)
    if independent_count < 2:
        return None, pair_count, independent_count
    mean = sum(independent_scores) / independent_count
    sample_variance = sum(
        (value - mean) ** 2 for value in independent_scores
    ) / (independent_count - 1)
    # Identical outcomes cannot imply certainty; the guard decays with n.
    smoothed_mean = (sum(independent_scores) + 0.5) / (independent_count + 1)
    variance_guard = smoothed_mean * (1.0 - smoothed_mean) / (
        independent_count * (independent_count + 1)
    )
    return math.sqrt(max(sample_variance / independent_count, variance_guard)), pair_count, independent_count


def attach_paired_score_stats(stats, game_scores, opening_count=None):
    stats = dict(stats or {})
    scores = {int(key): float(value) for key, value in dict(game_scores or {}).items()}
    score_se, pair_count, independent_count = paired_score_standard_error(
        scores, opening_count=opening_count
    )
    stats["game_scores"] = scores
    stats["paired_score_pairs"] = int(pair_count)
    stats["paired_score_unique_openings"] = int(independent_count)
    stats["paired_opening_count"] = int(opening_count or 0)
    stats["paired_score_se"] = score_se
    return stats


def combine_eval_stats(*stats_items):
    combined = empty_eval_stats()
    total_games = weighted_wins = weighted_draws = 0.0
    weighted_losses = weighted_unresolved = 0.0
    game_scores = {}
    seen_game_indices = set()
    duplicate_game_indices = set()
    inherited_errors = []
    opening_counts = set()
    combined_profile = {}
    for item in stats_items:
        if not item:
            continue
        weight = 1.0
        stats = item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            stats, weight = item
        weight = max(0.0, float(weight or 0.0))
        if weight <= 0.0 or not stats:
            continue
        if stats.get("infrastructure_error"):
            inherited_errors.append(str(stats["infrastructure_error"]))
        weighted_wins += weight * float(stats.get("wins", 0) or 0)
        weighted_draws += weight * float(stats.get("draws", 0) or 0)
        weighted_losses += weight * float(stats.get("losses", 0) or 0)
        weighted_unresolved += weight * float(stats.get("unresolved", 0) or 0)
        total_games += weight * float(stats.get("num_games", 0) or 0)
        for key, value in dict(stats.get("profile", {}) or {}).items():
            if isinstance(value, (int, float)):
                combined_profile[key] = combined_profile.get(key, 0) + weight * value
        if weight == 1.0:
            for key, value in dict(stats.get("game_scores", {}) or {}).items():
                try:
                    index = int(key)
                    score = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if index in seen_game_indices:
                    duplicate_game_indices.add(index)
                seen_game_indices.add(index)
                game_scores[index] = score
            opening_count = int(stats.get("paired_opening_count", 0) or 0)
            if opening_count > 0:
                opening_counts.add(opening_count)

    combined["wins"] = int(round(weighted_wins))
    combined["draws"] = int(round(weighted_draws))
    combined["losses"] = int(round(weighted_losses))
    combined["unresolved"] = int(round(weighted_unresolved))
    combined["num_games"] = int(round(total_games))
    combined["resolved_games"] = max(0, combined["num_games"] - combined["unresolved"])
    if total_games > 0:
        combined["score_rate"] = float(
            (weighted_wins + 0.5 * (weighted_draws + weighted_unresolved)) / total_games
        )
        combined["win_rate"] = float(weighted_wins / total_games)
        combined["draw_rate"] = float((weighted_draws + weighted_unresolved) / total_games)
        combined["loss_rate"] = float(weighted_losses / total_games)
    combined["profile"] = combined_profile
    opening_count = next(iter(opening_counts)) if len(opening_counts) == 1 else None
    combined = attach_paired_score_stats(combined, game_scores, opening_count=opening_count)
    if duplicate_game_indices:
        indices = ", ".join(str(index) for index in sorted(duplicate_game_indices)[:8])
        inherited_errors.append("duplicate game indices while combining evaluation stages: " + indices)
    if len(opening_counts) > 1:
        inherited_errors.append("inconsistent paired opening counts while combining evaluation stages")
    if inherited_errors:
        combined["infrastructure_error"] = "; ".join(inherited_errors)
    return combined


def _paired_evidence_error(stats, expected_games, *, start_index=0, expected_openings=None):
    """Return a fail-closed reason unless all paired-game telemetry agrees."""
    stats = dict(stats or {})
    try:
        expected_games = int(expected_games)
        start_index = int(start_index)
        raw_games = stats["num_games"]
        if isinstance(raw_games, bool) or not isinstance(raw_games, Real):
            return "game count must be a numeric integer"
        games = int(raw_games)
        if float(raw_games) != games:
            return "game count must be a numeric integer"
        counts = [stats.get(key, 0) for key in ("wins", "draws", "losses", "unresolved")]
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in counts):
            return "WDL counts must be numeric integers"
        parsed_counts = [int(value) for value in counts]
        if any(float(value) != parsed for value, parsed in zip(counts, parsed_counts)):
            return "WDL counts must be integers"
        wins, draws, losses, unresolved = parsed_counts
        score_map = dict(stats["game_scores"])
        parsed_scores = {}
        for key, value in score_map.items():
            if isinstance(key, bool):
                return "game score indices must be integers"
            if isinstance(key, Real):
                if not math.isfinite(float(key)) or float(key) != int(key):
                    return "game score indices must be integers"
                index = int(key)
            elif isinstance(key, str):
                # Historical JSON maps use stringified integer keys.
                index = int(key)
            else:
                return "game score indices must be integers"
            if index in parsed_scores:
                return "duplicate game indices in score map"
            if isinstance(value, bool) or not isinstance(value, Real):
                return "game scores must be numeric values"
            parsed_scores[index] = float(value)
        raw_pair_counts = (
            stats["paired_score_pairs"],
            stats["paired_score_unique_openings"],
            stats.get("paired_opening_count", 0) or 0,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            or float(value) != int(value)
            for value in raw_pair_counts
        ):
            return "paired counts must be numeric integers"
        reported_pairs, reported_openings, opening_count = map(int, raw_pair_counts)
        reported_se = float(stats["paired_score_se"])
        score_rate = float(stats["score_rate"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return "required paired evaluation evidence is missing or malformed"
    if expected_games <= 0 or expected_games % 2 or start_index % 2:
        return "requested evaluation must contain complete colour-swapped pairs"
    if games != expected_games:
        return f"reported {games} games, expected {expected_games}"
    expected_indices = set(range(start_index, start_index + expected_games))
    if set(parsed_scores) != expected_indices:
        return "game score indices are not the requested contiguous range"
    scores = list(parsed_scores.values())
    if any(not math.isfinite(value) or value not in (0.0, 0.5, 1.0) for value in scores):
        return "game scores must be finite win/draw/loss values"
    if min(wins, draws, losses, unresolved) < 0:
        return "WDL counts cannot be negative"
    if wins + draws + losses + unresolved != expected_games:
        return "WDL counts do not sum to the requested game count"
    score_wins = sum(value == 1.0 for value in scores)
    score_draws = sum(value == 0.5 for value in scores)
    score_losses = sum(value == 0.0 for value in scores)
    if (wins, draws + unresolved, losses) != (score_wins, score_draws, score_losses):
        return "WDL counts disagree with per-game outcomes"
    expected_score = (wins + 0.5 * (draws + unresolved)) / float(expected_games)
    evidence_score = sum(scores) / float(expected_games)
    if (
        not math.isfinite(score_rate) or not 0.0 <= score_rate <= 1.0
        or abs(score_rate - expected_score) > 1e-9
        or abs(score_rate - evidence_score) > 1e-9
    ):
        return "aggregate score disagrees with WDL counts or per-game scores"
    if expected_openings is None:
        expected_openings = expected_games // 2
    if reported_pairs != expected_games // 2 or reported_openings != int(expected_openings):
        return "paired game or unique-opening counts do not match the requested suite"
    if opening_count < 0 or (opening_count and opening_count < expected_openings):
        return "paired opening count is inconsistent with the requested suite"
    derived_se, pair_count, independent_count = paired_score_standard_error(
        parsed_scores, opening_count=opening_count or None
    )
    if pair_count != expected_games // 2 or independent_count != int(expected_openings):
        return "per-game scores do not form the requested complete opening pairs"
    if (
        derived_se is None or not math.isfinite(reported_se) or reported_se < 0.0
        or abs(reported_se - derived_se) > max(1e-9, 1e-6 * derived_se)
    ):
        return "reported paired standard error disagrees with per-game evidence"
    return None


def promotion_eval_has_full_evidence(eval_stage, stats, expected_games):
    """Only complete paired selector evidence may nominate a checkpoint."""
    expected_games = int(expected_games)
    stats = dict(stats or {})
    try:
        reported_opening_count = int(stats.get("paired_opening_count", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        str(eval_stage or "") == "funnel:advanced"
        and not stats.get("infrastructure_error")
        and reported_opening_count == expected_games // 2
        and _paired_evidence_error(stats, expected_games) is None
    )


def promotion_anchor_safety_decision(
    anchor_score_rate,
    anchor_score_lower_bound,
    anchor_no_mcts_score_rate,
    anchor_no_mcts_score_lower_bound,
    rl_cfg,
):
    """Keep every promoted best above the configured anchor floor."""
    if not bool(rl_cfg.get("promotion_require_anchor_non_regression", True)):
        return True, "anchor guard disabled"
    if anchor_score_rate is None or anchor_score_lower_bound is None:
        return False, "anchor evidence unavailable"
    score_floor = float(rl_cfg.get("promotion_anchor_min_score_rate", 0.50))
    lower_bound_floor = float(rl_cfg.get("promotion_anchor_score_lower_bound_min", 0.47))
    score = float(anchor_score_rate)
    lower_bound = float(anchor_score_lower_bound)
    mcts_safe = score >= score_floor and lower_bound >= lower_bound_floor
    return bool(mcts_safe), (
        f"anchor MCTS {score:.1%}, LB {lower_bound:.1%}; "
        f"required {score_floor:.1%}/{lower_bound_floor:.1%}"
    )


def promotion_confirmation_decision(
    stats,
    *,
    suite_id,
    expected_suite_id,
    expected_games=192,
    expected_openings=96,
    score_min=0.53,
    lower_bound_min=0.50,
    z=1.645,
    expected_suite_ids=None,
):
    """Require complete paired evidence from a named independent suite."""
    stats = dict(stats or {})
    try:
        observed_games = int(stats.get("num_games", 0) or 0)
        observed_openings = int(stats.get("paired_score_unique_openings", 0) or 0)
        observed_pairs = int(stats.get("paired_score_pairs", 0) or 0)
        observed_opening_count = int(stats.get("paired_opening_count", 0) or 0)
        score_map = dict(stats.get("game_scores") or {})
        score_indices = {int(index) for index in score_map}
        score_values = [float(value) for value in score_map.values()]
        outcome_count = sum(
            int(stats.get(key, 0) or 0)
            for key in ("wins", "draws", "losses", "unresolved")
        )
    except (TypeError, ValueError):
        observed_games = observed_openings = observed_pairs = 0
        observed_opening_count = outcome_count = 0
        score_map, score_indices, score_values = {}, set(), []
    result = {
        "passed": False,
        "status": "",
        "score_rate": None,
        "score_lower_bound": None,
        "score_upper_bound": None,
        "score_se": None,
        "games": observed_games,
        "unique_openings": observed_openings,
        "suite_id": str(suite_id or ""),
        "simulations": None,
    }
    if stats.get("infrastructure_error"):
        result["status"] = "blocked: confirmation infrastructure error"
        result["error"] = str(stats.get("infrastructure_error"))
        return result
    if not expected_suite_id or str(suite_id or "") != str(expected_suite_id):
        result["status"] = "blocked: confirmation suite ID mismatch"
        return result
    if expected_suite_ids is not None:
        expected_suite_ids = [str(value) for value in expected_suite_ids]
        observed_suite_ids = [
            str(value) for value in (stats.get("suite_ids") or [])
        ]
        result["suite_ids"] = observed_suite_ids
        if not expected_suite_ids or observed_suite_ids != expected_suite_ids:
            result["status"] = "blocked: confirmation suite IDs mismatch"
            return result
    identity_fields = (
        ("candidate_model_state_sha256_before", "candidate_model_state_sha256_after"),
        ("opponent_model_state_sha256_before", "opponent_model_state_sha256_after"),
    )
    for before_key, after_key in identity_fields:
        before = stats.get(before_key)
        after = stats.get(after_key)
        if before is not None and after is not None and str(before) != str(after):
            result["status"] = "blocked: model state SHA changed during confirmation"
            result["error"] = f"{before_key} does not match {after_key}"
            return result
    expected_games = int(expected_games)
    expected_openings = int(expected_openings)
    if expected_games != 2 * expected_openings:
        result["status"] = "blocked: incomplete confirmation evidence"
        return result
    evidence_error = _paired_evidence_error(
        stats, expected_games, start_index=0, expected_openings=expected_openings
    )
    if evidence_error or observed_opening_count != expected_openings:
        result["status"] = "blocked: incomplete confirmation evidence"
        result["error"] = evidence_error or "paired opening count does not match the confirmation suite"
        return result
    try:
        score = float(stats["score_rate"])
        score_se = float(stats["paired_score_se"])
    except (KeyError, TypeError, ValueError):
        result["status"] = "blocked: confirmation score or paired SE unavailable"
        return result
    if (
        not math.isfinite(score) or not 0.0 <= score <= 1.0
        or not math.isfinite(score_se) or score_se < 0.0
    ):
        result["status"] = "blocked: invalid confirmation score or paired SE"
        return result
    if abs(sum(score_values) / float(expected_games) - score) > 1e-9:
        result["status"] = "blocked: confirmation score disagrees with game evidence"
        return result
    expected_score = (
        float(stats.get("wins", 0) or 0)
        + 0.5 * float(stats.get("draws", 0) or 0)
        + 0.5 * float(stats.get("unresolved", 0) or 0)
    ) / float(expected_games)
    if abs(expected_score - score) > 1e-9:
        result["status"] = "blocked: confirmation score disagrees with WDL counts"
        return result
    lower_bound = score_rate_lower_bound(score, expected_games, z, standard_error=score_se)
    upper_bound = score_rate_upper_bound(score, expected_games, z, standard_error=score_se)
    result.update({
        "score_rate": score,
        "score_se": score_se,
        "score_lower_bound": lower_bound,
        "score_upper_bound": upper_bound,
    })
    if score < float(score_min):
        result["status"] = f"blocked: holdout score {score:.1%} < {float(score_min):.1%}"
    elif lower_bound < float(lower_bound_min):
        result["status"] = f"blocked: holdout LB {lower_bound:.1%} < {float(lower_bound_min):.1%}"
    else:
        result["passed"] = True
        result["status"] = "passed: independent promotion confirmation"
    return result


def promotion_raw_nn_decision(
    score_rate,
    num_games,
    z,
    rl_cfg,
    *,
    enabled_override=None,
    standard_error=None,
):
    """Classify a raw-NN diagnostic without making it a promotion gate.

    ``enabled_override`` is retained for callers loading legacy configs; the
    persisted promotion gate setting is intentionally ignored.
    """
    # Legacy callers may still pass this override to suppress telemetry. The
    # persisted config flag cannot control promotion or diagnostic scheduling.
    enabled = True if enabled_override is None else bool(enabled_override)
    score_floor = float(rl_cfg.get("promotion_no_mcts_score_rate_min", 0.48) or 0.48)
    if not enabled:
        return True, "", score_floor, None, "disabled"
    if (
        score_rate is None or not math.isfinite(float(score_rate))
        or int(num_games or 0) <= 0
    ):
        return False, "diagnostic unavailable", score_floor, None, "unavailable"
    score = float(score_rate)
    upper = score_rate_upper_bound(score, num_games, z, standard_error=standard_error)
    lower = score_rate_lower_bound(score, num_games, z, standard_error=standard_error)
    if lower >= score_floor:
        return True, "", score_floor, upper, "passed"
    if upper < score_floor:
        return False, (
            f"diagnostic below floor: UB {upper:.1%} < {score_floor:.1%}"
        ), score_floor, upper, "regression"
    return False, (
        f"diagnostic inconclusive: LB {lower:.1%} < floor {score_floor:.1%} "
        f"<= UB {upper:.1%}"
    ), score_floor, upper, "inconclusive"


def anchor_candidate_needs_more_games(
    score_rate,
    num_games,
    *,
    min_score_rate,
    min_score_lower_bound,
    z,
    max_games,
    score_standard_error=None,
):
    games = max(1, int(num_games or 0))
    if games >= max(1, int(max_games)):
        return False
    lower = score_rate_lower_bound(score_rate, games, z, standard_error=score_standard_error)
    if float(score_rate or 0.0) >= float(min_score_rate) and lower >= float(min_score_lower_bound):
        return False
    upper = score_rate_upper_bound(score_rate, games, z, standard_error=score_standard_error)
    return upper >= float(min_score_rate)


def eval_mcts_move_metrics(eval_stats, prefix):
    profile = dict((eval_stats or {}).get("profile", {}) or {})
    base = f"eval_mcts_{prefix}"
    samples = int(profile.get(f"{base}_move_samples", 0) or 0)
    changed = int(profile.get(f"{base}_changed_count", 0) or 0)
    q_samples = int(profile.get(f"{base}_changed_q_samples", 0) or 0)
    higher = int(profile.get(f"{base}_higher_q_count", 0) or 0)
    lower = int(profile.get(f"{base}_lower_q_count", 0) or 0)
    q_sum = float(profile.get(f"{base}_changed_q_delta_sum", 0.0) or 0.0)
    budget_samples = int(profile.get(f"{base}_budget_samples", 0) or 0)
    budget_sum = float(profile.get(f"{base}_budget_sum", 0.0) or 0.0)
    reduced_budgets = int(profile.get(f"{base}_reduced_budget_count", 0) or 0)
    return {
        "samples": samples,
        "avg_simulations": budget_sum / float(budget_samples) if budget_samples else None,
        "reduced_budget_rate": float(reduced_budgets) / float(budget_samples) if budget_samples else None,
        "changed_rate": float(changed) / float(samples) if samples else None,
        "higher_q_when_changed_rate": float(higher) / float(q_samples) if q_samples else None,
        "lower_q_when_changed_rate": float(lower) / float(q_samples) if q_samples else None,
        "changed_q_delta_mean": q_sum / float(q_samples) if q_samples else None,
    }


def evaluate_mcts_selector_funnel(
    *,
    model,
    best_model,
    config,
    device,
    preliminary_games,
    preliminary_simulations,
    medium_games,
    medium_simulations,
    advanced_games,
    advanced_simulations,
    preliminary_score_rate,
    preliminary_true_win_rate=0.0,
    medium_score_rate=0.53,
    medium_true_win_rate=0.0,
    force_medium=False,
    game_index_offset=0,
    use_fixed_openings=True,
    central_runtime=None,
    evaluate_stage=None,
    build_config=None,
    opening_count=None,
    combine_stats=None,
    score_upper_bound=None,
    progress_factory=None,
):
    """Run cumulative, paired MCTS selector stages through injected runtime APIs."""
    preliminary_games = max(1, int(preliminary_games))
    medium_games = max(0, int(medium_games))
    advanced_games = max(0, int(advanced_games))
    if any(games % 2 for games in (preliminary_games, medium_games, advanced_games)) or int(game_index_offset) % 2:
        raise ValueError("Promotion stages require complete colour-swapped opening pairs.")
    if len({int(preliminary_simulations), int(medium_simulations), int(advanced_simulations)}) != 1:
        raise ValueError("Cumulative promotion evidence requires the same MCTS budget in every stage.")
    if use_fixed_openings and opening_count is not None and game_index_offset + preliminary_games + medium_games + advanced_games > 2 * int(opening_count):
        raise ValueError("Promotion budget exceeds the unique fixed opening suite.")
    if evaluate_stage is None or build_config is None or combine_stats is None or score_upper_bound is None:
        raise TypeError("Promotion funnel requires evaluation, config, combine, and CI adapters.")
    stage_configs = [
        build_config(config, preliminary_simulations),
        build_config(config, medium_simulations),
        build_config(config, advanced_simulations),
    ]
    total_games = preliminary_games + medium_games + advanced_games
    offsets = [
        int(game_index_offset),
        int(game_index_offset) + preliminary_games,
        int(game_index_offset) + preliminary_games + medium_games,
    ]
    progress = progress_factory(total_games) if progress_factory is not None else None
    rejecting_looks = int(not bool(force_medium) and (medium_games > 0 or advanced_games > 0))
    rejecting_looks += int(advanced_games > 0)
    nominal_alpha = 1.0 - NormalDist().cdf(1.28)
    early_z = (
        NormalDist().inv_cdf(1.0 - nominal_alpha / rejecting_looks)
        if rejecting_looks > 0 else 1.28
    )

    def _stage_failure(stats, expected_games, offset):
        reason = _paired_evidence_error(
            stats, expected_games, start_index=offset,
        )
        if reason:
            failed = dict(stats or {})
            failed["infrastructure_error"] = f"invalid paired evaluation evidence: {reason}"
            return failed
        return None

    def _upper_bound(score, games, standard_error):
        try:
            return score_upper_bound(
                score, games, z=early_z, standard_error=standard_error
            )
        except TypeError:
            # Preserve compatibility with injected legacy adapters. The local
            # implementation retains the corrected bound and supplied SE.
            return score_rate_upper_bound(
                score, games, z=early_z, standard_error=standard_error
            )

    try:
        preliminary_stats = evaluate_stage(
            model, best_model, stage_configs[0], device, preliminary_games,
            offsets[0], use_fixed_openings, central_runtime,
            progress=progress, progress_desc="Eval vs best",
        )
        if (preliminary_stats or {}).get("infrastructure_error"):
            return preliminary_stats, "infrastructure_error"
        failure = _stage_failure(preliminary_stats, preliminary_games, offsets[0])
        if failure:
            return failure, "infrastructure_error"
        score = float((preliminary_stats or {}).get("score_rate", 0.0) or 0.0)
        true_win = float((preliminary_stats or {}).get("win_rate", 0.0) or 0.0)
        if (
            not bool(force_medium)
            and (
                _upper_bound(
                    score, preliminary_stats.get("num_games", preliminary_games),
                    preliminary_stats.get("paired_score_se"),
                ) < float(preliminary_score_rate)
                or true_win < float(preliminary_true_win_rate)
            )
        ):
            return preliminary_stats, "funnel:preliminary_reject"
        if medium_games <= 0:
            return preliminary_stats, "funnel:preliminary_only"
        medium_stats = evaluate_stage(
            model, best_model, stage_configs[1], device, medium_games,
            offsets[1], use_fixed_openings, central_runtime,
            progress=progress, progress_desc="Eval vs best",
        )
        if (medium_stats or {}).get("infrastructure_error"):
            return medium_stats, "infrastructure_error"
        failure = _stage_failure(medium_stats, medium_games, offsets[1])
        if failure:
            return failure, "infrastructure_error"
        through_medium = combine_stats(preliminary_stats, medium_stats)
        failure = _stage_failure(
            through_medium, preliminary_games + medium_games, offsets[0]
        )
        if failure:
            return failure, "infrastructure_error"
        score = float((through_medium or {}).get("score_rate", 0.0) or 0.0)
        true_win = float((through_medium or {}).get("win_rate", 0.0) or 0.0)
        if (
            _upper_bound(
                score, through_medium.get("num_games", preliminary_games + medium_games),
                through_medium.get("paired_score_se"),
            ) < float(medium_score_rate)
            or true_win < float(medium_true_win_rate)
        ):
            return through_medium, "funnel:medium_reject"
        if advanced_games <= 0:
            return through_medium, "funnel:medium_only"
        advanced_stats = evaluate_stage(
            model, best_model, stage_configs[2], device, advanced_games,
            offsets[2], use_fixed_openings, central_runtime,
            progress=progress, progress_desc="Eval vs best",
        )
        if (advanced_stats or {}).get("infrastructure_error"):
            return advanced_stats, "infrastructure_error"
        failure = _stage_failure(advanced_stats, advanced_games, offsets[2])
        if failure:
            return failure, "infrastructure_error"
        combined = combine_stats(through_medium, advanced_stats)
        failure = _stage_failure(combined, total_games, offsets[0])
        if failure:
            return failure, "infrastructure_error"
        combined["promotion_mcts_simulations"] = int(advanced_simulations)
        return combined, "funnel:advanced"
    finally:
        if progress is not None:
            progress.close()
