"""Opponent scheduling helpers for RL self-play."""

from collections import defaultdict, deque

import numpy as np
import torch

def _safe_score_rate(wins, draws, losses):
    total = int(wins) + int(draws) + int(losses)
    if total <= 0:
        return None
    return float((float(wins) + 0.5 * float(draws)) / float(total))


def _state_dicts_identical(state_a, state_b):
    if state_a is None or state_b is None:
        return False
    if state_a.keys() != state_b.keys():
        return False
    for key in state_a.keys():
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            return False
        if not torch.equal(tensor_a, tensor_b):
            return False
    return True


def _adaptive_factor_from_history(history_values, target_score, band, min_factor, max_factor):
    if not history_values:
        return 1.0
    avg_score = float(sum(history_values) / len(history_values))
    distance = min(1.0, abs(avg_score - target_score) / band)
    closeness = max(0.0, 1.0 - distance)
    return float(min_factor + (max_factor - min_factor) * closeness)


def _average_score_from_history(history_values):
    if not history_values:
        return None
    return float(sum(float(value) for value in history_values) / len(history_values))


def _score_closeness(avg_score, target_score, band):
    if avg_score is None:
        return 0.5
    distance = min(1.0, abs(float(avg_score) - float(target_score)) / max(1e-8, float(band)))
    return max(0.0, 1.0 - distance)


def _allocate_counts_from_weights(labels, weights, total_count):
    labels = [str(label) for label in list(labels or [])]
    if not labels or int(total_count) <= 0:
        return {}

    total_count = int(total_count)
    weights_arr = np.asarray(list(weights or []), dtype=np.float64)
    if weights_arr.size != len(labels):
        weights_arr = np.ones((len(labels),), dtype=np.float64)
    weights_arr = np.clip(weights_arr, 0.0, None)

    weight_sum = float(weights_arr.sum())
    if weight_sum <= 0.0:
        weights_arr.fill(1.0 / float(len(labels)))
    else:
        weights_arr /= weight_sum

    raw = weights_arr * float(total_count)
    base = np.floor(raw).astype(np.int64)
    counts = {label: int(base[idx]) for idx, label in enumerate(labels)}
    assigned = int(sum(counts.values()))
    remainder = int(total_count - assigned)
    if remainder > 0:
        order = sorted(
            range(len(labels)),
            key=lambda idx: (-(raw[idx] - float(base[idx])), labels[idx]),
        )
        for idx in order[:remainder]:
            counts[labels[idx]] = int(counts.get(labels[idx], 0)) + 1
    return counts


def _cap_recent_latest_counts(recent_entries, counts, total_games, rl_cfg):
    if not recent_entries or not counts or int(total_games) <= 0:
        return dict(counts)

    latest_count = max(0, int(rl_cfg.get('self_play_recent_latest_cap_count', 2)))
    latest_fraction = max(
        0.0,
        min(1.0, float(rl_cfg.get('self_play_recent_latest_max_fraction', 1.0))),
    )
    if latest_count <= 0 or latest_fraction >= 1.0:
        return dict(counts)

    ordered = sorted(
        list(recent_entries),
        key=lambda entry: (
            -float(entry.get("recency_bias", 0.0)),
            str(entry.get("label", "")),
        ),
    )
    latest_labels = {str(entry.get("label", "recent")) for entry in ordered[:latest_count]}
    older_labels = [
        str(entry.get("label", "recent"))
        for entry in ordered
        if str(entry.get("label", "recent")) not in latest_labels
    ]
    if not latest_labels or not older_labels:
        return dict(counts)

    adjusted = {str(label): int(count) for label, count in dict(counts).items()}
    latest_total = int(sum(adjusted.get(label, 0) for label in latest_labels))
    latest_cap = int(np.floor(float(total_games) * latest_fraction))
    if latest_total <= latest_cap:
        return adjusted

    excess = int(latest_total - latest_cap)
    for label in sorted(latest_labels, key=lambda item: (-adjusted.get(item, 0), item)):
        if excess <= 0:
            break
        take = min(excess, max(0, int(adjusted.get(label, 0))))
        adjusted[label] = int(adjusted.get(label, 0)) - take
        excess -= take

    if excess <= 0:
        older_weights = [
            max(0.0, float(entry.get("weight", 1.0)))
            for entry in ordered
            if str(entry.get("label", "recent")) in older_labels
        ]
        older_extra = _allocate_counts_from_weights(older_labels, older_weights, latest_total - latest_cap)
        for label, extra in older_extra.items():
            adjusted[str(label)] = int(adjusted.get(str(label), 0)) + int(extra)
    return adjusted


def _safe_draw_rate(wins, draws, losses):
    total = int(wins) + int(draws) + int(losses)
    if total <= 0:
        return None
    return float(float(draws) / float(total))


def _draw_heaviness_penalty(draw_rate, target_draw_rate, band, min_factor):
    if draw_rate is None:
        return 1.0
    if draw_rate <= target_draw_rate:
        return 1.0
    distance = min(1.0, (float(draw_rate) - float(target_draw_rate)) / max(1e-8, float(band)))
    return float(1.0 - (1.0 - float(min_factor)) * distance)


def _normalize_weight_map(weight_map):
    normalized = {
        str(label): max(0.0, float(weight))
        for label, weight in dict(weight_map or {}).items()
    }
    total = float(sum(normalized.values()))
    if total <= 0.0:
        return normalized
    return {label: (weight / total) for label, weight in normalized.items()}


def _bounded_normalize_weight_map(weight_map, min_shares=None, max_shares=None):
    """Normalize weights onto a bounded simplex while preserving proportions."""
    labels = [str(label) for label in dict(weight_map or {}).keys()]
    if not labels:
        return {}

    base = {
        label: max(0.0, float(dict(weight_map or {}).get(label, 0.0)))
        for label in labels
    }
    if float(sum(base.values())) <= 0.0:
        equal = 1.0 / float(len(labels))
        base = {label: equal for label in labels}
    else:
        total_base = float(sum(base.values()))
        base = {label: float(value) / total_base for label, value in base.items()}

    min_shares = {str(k): max(0.0, min(1.0, float(v))) for k, v in dict(min_shares or {}).items()}
    max_shares = {str(k): max(0.0, min(1.0, float(v))) for k, v in dict(max_shares or {}).items()}
    lower = {label: float(min_shares.get(label, 0.0)) for label in labels}
    upper = {
        label: max(float(lower[label]), float(max_shares.get(label, 1.0)))
        for label in labels
    }

    lower_total = float(sum(lower.values()))
    if lower_total > 1.0:
        scale = 1.0 / max(1e-8, lower_total)
        lower = {label: float(value) * scale for label, value in lower.items()}
        upper = {label: max(float(lower[label]), float(upper[label])) for label in labels}
    upper_total = float(sum(upper.values()))
    if upper_total < 1.0:
        upper = {label: 1.0 for label in labels}

    assigned = {}
    free = set(labels)
    remaining = 1.0
    eps = 1e-9
    while free:
        free_base_total = float(sum(base[label] for label in free))
        if free_base_total <= eps:
            proposal = {label: remaining / float(len(free)) for label in free}
        else:
            proposal = {
                label: remaining * float(base[label]) / free_base_total
                for label in free
            }

        clamped = []
        for label, value in proposal.items():
            if value < lower[label] - eps:
                clamped.append((label, lower[label]))
            elif value > upper[label] + eps:
                clamped.append((label, upper[label]))

        if not clamped:
            assigned.update(proposal)
            break

        for label, value in clamped:
            if label not in free:
                continue
            assigned[label] = float(value)
            remaining -= float(value)
            free.remove(label)
        remaining = max(0.0, remaining)

    return _normalize_weight_map(assigned)


def _canonicalize_opponent_bucket(label):
    normalized = str(label or "current").strip().lower()
    if normalized.startswith("recent"):
        return "recent"
    if normalized == "best":
        return "best"
    return "current"


def _build_selfplay_opponent_candidates(
    rl_cfg,
    best_model_state=None,
    recent_snapshot_pool=None,
    scheduler_state=None,
):
    current_fraction = max(0.0, float(rl_cfg.get('self_play_opponent_current_fraction', 0.4)))
    best_fraction = max(0.0, float(rl_cfg.get('self_play_opponent_best_fraction', 0.3)))
    recent_fraction = max(0.0, float(rl_cfg.get('self_play_opponent_recent_fraction', 0.3)))
    recent_snapshot_pool = list(recent_snapshot_pool or [])
    scheduler_state = dict(scheduler_state or {})
    adaptive_enabled = bool(rl_cfg.get('self_play_opponent_adaptive_enabled', False))
    exact_score_history = scheduler_state.get("score_history_exact", {}) or {}
    target_score = float(rl_cfg.get('self_play_opponent_adaptive_target_score', 0.50))
    band = max(0.05, float(rl_cfg.get('self_play_opponent_adaptive_band', 0.15)))
    min_factor = max(0.20, float(rl_cfg.get('self_play_opponent_adaptive_min_factor', 0.60)))
    max_factor = max(min_factor, float(rl_cfg.get('self_play_opponent_adaptive_max_factor', 1.40)))
    min_score = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_min_score', 0.0))))
    max_score = max(min_score, min(1.0, float(rl_cfg.get('self_play_opponent_max_score', 1.0))))
    best_min_score = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_best_min_score', min_score))))
    recent_min_score = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_recent_min_score', min_score))))
    recent_max_score = max(recent_min_score, min(1.0, float(rl_cfg.get('self_play_opponent_recent_max_score', max_score))))
    recent_candidate_limit = max(1, int(rl_cfg.get('self_play_recent_candidate_pool_size', 4)))
    recent_recency_bias = max(0.0, float(rl_cfg.get('self_play_recent_recency_bias', 0.35)))
    recent_min_games_for_confidence = max(
        1.0,
        float(rl_cfg.get('self_play_recent_confidence_games', 24)),
    )
    recent_uncertainty_bonus = max(
        0.0,
        min(0.5, float(rl_cfg.get('self_play_recent_uncertainty_bonus', 0.15))),
    )
    recent_in_band_boost = max(
        1.0,
        float(rl_cfg.get('self_play_recent_in_band_boost', 1.10)),
    )
    target_draw_rate = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_target_draw_rate', 0.45))))
    draw_band = max(0.01, float(rl_cfg.get('self_play_opponent_draw_band', 0.20)))
    draw_penalty_min_factor = max(0.20, min(1.0, float(rl_cfg.get('self_play_opponent_draw_penalty_min_factor', 0.70))))
    exact_draw_history = scheduler_state.get("draw_history_exact", {}) or {}
    exact_games_history = scheduler_state.get("games_history_exact", {}) or {}

    candidates = []
    if current_fraction > 0.0:
        candidates.append({
            "label": "current",
            "weight": float(current_fraction),
            "payload": None,
        })
    if best_model_state is not None and best_fraction > 0.0:
        best_factor = 1.0
        if adaptive_enabled:
            best_factor = _adaptive_factor_from_history(
                exact_score_history.get("best", []) or [],
                target_score,
                band,
                min_factor,
                max_factor,
            )
        best_factor *= _draw_heaviness_penalty(
            _average_score_from_history(exact_draw_history.get("best", []) or []),
            target_draw_rate,
            draw_band,
            draw_penalty_min_factor,
        )
        candidates.append({
            "label": "best",
            "weight": float(best_fraction * best_factor),
            "payload": {
                "label": "best",
                "state": best_model_state,
            },
        })
    if recent_snapshot_pool and recent_fraction > 0.0:
        recent_count = max(1, len(recent_snapshot_pool))
        recent_entries_all = []
        dedup_states = []
        if best_model_state is not None:
            dedup_states.append(best_model_state)
        for idx, recent_entry in enumerate(recent_snapshot_pool):
            recent_state = recent_entry.get("state")
            if recent_state is None:
                continue
            if any(_state_dicts_identical(recent_state, existing_state) for existing_state in dedup_states):
                continue
            recency_bias = float(idx + 1) / float(recent_count)
            label = str(recent_entry.get("label", f"recent_{idx}"))
            factor = 1.0
            avg_score = _average_score_from_history(exact_score_history.get(label, []) or [])
            avg_draw_rate = _average_score_from_history(exact_draw_history.get(label, []) or [])
            avg_games = _average_score_from_history(exact_games_history.get(label, []) or [])
            in_band = True
            if avg_score is not None and (avg_score < recent_min_score or avg_score > recent_max_score):
                in_band = False
            if adaptive_enabled:
                factor = _adaptive_factor_from_history(
                    exact_score_history.get(label, []) or [],
                    target_score,
                    band,
                    min_factor,
                    max_factor,
                )
            draw_penalty = _draw_heaviness_penalty(
                avg_draw_rate,
                target_draw_rate,
                draw_band,
                draw_penalty_min_factor,
            )
            factor *= draw_penalty
            confidence = 0.0
            if avg_games is not None:
                confidence = min(1.0, float(avg_games) / recent_min_games_for_confidence)
            uncertainty_bonus = (1.0 - confidence) * recent_uncertainty_bonus
            match_quality = _score_closeness(avg_score, target_score, band)
            if in_band:
                match_quality = min(1.0, match_quality * recent_in_band_boost)
            selection_score = float(
                (
                    (1.0 - recent_recency_bias) * match_quality
                    + recent_recency_bias * recency_bias
                    + uncertainty_bonus
                )
                * draw_penalty
            )
            recent_entries_all.append({
                "label": label,
                "state": recent_state,
                "weight": float((0.75 + 0.25 * recency_bias) * factor),
                "avg_score": avg_score,
                "avg_draw_rate": avg_draw_rate,
                "avg_games": avg_games,
                "in_band": bool(in_band),
                "recency_bias": recency_bias,
                "draw_penalty": float(draw_penalty),
                "selection_score": selection_score,
            })
            dedup_states.append(recent_state)
        if recent_entries_all:
            in_band_entries = [entry for entry in recent_entries_all if bool(entry.get("in_band", False))]
            candidate_entries = list(in_band_entries)
            candidate_entries.sort(
                key=lambda entry: (
                    -float(entry.get("selection_score", 0.0)),
                    -float(entry.get("recency_bias", 0.0)),
                    str(entry.get("label", "")),
                )
            )
            recent_entries = candidate_entries[:recent_candidate_limit]
            if recent_entries:
                candidates.append({
                    "label": "recent",
                    "weight": float(recent_fraction),
                    "payload": {
                        "entries": recent_entries,
                        "all_entries": recent_entries_all,
                    },
                })
    filtered_candidates = []
    for candidate in candidates:
        label = str(candidate.get("label", "current"))
        if label == "best":
            avg_score = _average_score_from_history(exact_score_history.get("best", []) or [])
            if avg_score is not None and avg_score < best_min_score:
                continue
        elif label != "current":
            bucket_history = scheduler_state.get("bucket_score_history", {}) or {}
            avg_score = _average_score_from_history(bucket_history.get(label, []) or [])
            if avg_score is not None and (avg_score < min_score or avg_score > max_score):
                continue
        filtered_candidates.append(candidate)
    if not any(str(candidate.get("label")) == "current" for candidate in filtered_candidates) and current_fraction > 0.0:
        filtered_candidates.append({
            "label": "current",
            "weight": float(max(current_fraction, 1e-6)),
            "payload": None,
        })
    return filtered_candidates


def _compute_adaptive_opponent_weights(rl_cfg, candidates, scheduler_state=None):
    scheduler_state = dict(scheduler_state or {})
    adaptive_enabled = bool(rl_cfg.get('self_play_opponent_adaptive_enabled', False))
    base_weights = {
        str(candidate["label"]): max(0.0, float(candidate.get("weight", 0.0)))
        for candidate in candidates
    }
    if not adaptive_enabled or not candidates:
        return _normalize_weight_map(base_weights), {}

    score_history = (
        scheduler_state.get("bucket_score_history")
        or scheduler_state.get("score_history")
        or {}
    )
    target_score = float(rl_cfg.get('self_play_opponent_adaptive_target_score', 0.50))
    band = max(0.05, float(rl_cfg.get('self_play_opponent_adaptive_band', 0.15)))
    min_factor = max(0.20, float(rl_cfg.get('self_play_opponent_adaptive_min_factor', 0.60)))
    max_factor = max(min_factor, float(rl_cfg.get('self_play_opponent_adaptive_max_factor', 1.40)))
    current_min_fraction = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_current_min_fraction', 0.50))))
    current_max_fraction = max(current_min_fraction, min(1.0, float(rl_cfg.get('self_play_opponent_current_max_fraction', 1.0))))
    best_min_fraction = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_best_min_fraction', 0.0))))
    recent_max_fraction = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_recent_max_fraction', 1.0))))

    adjusted = {}
    debug_factors = {}
    for candidate in candidates:
        label = str(candidate["label"])
        base_weight = base_weights.get(label, 0.0)
        factor = 1.0
        history_values = score_history.get(label, []) or []
        if label != "current" and history_values:
            factor = _adaptive_factor_from_history(
                history_values,
                target_score,
                band,
                min_factor,
                max_factor,
            )
        adjusted[label] = base_weight * factor
        debug_factors[label] = float(factor)

    total_weight = float(sum(adjusted.values()))
    if total_weight <= 0.0:
        return _normalize_weight_map(base_weights), debug_factors
    min_shares = {}
    max_shares = {}
    if "current" in adjusted:
        min_shares["current"] = current_min_fraction
        max_shares["current"] = current_max_fraction
    if "best" in adjusted:
        min_shares["best"] = best_min_fraction
    if "recent" in adjusted:
        max_shares["recent"] = recent_max_fraction
    normalized = _bounded_normalize_weight_map(
        adjusted,
        min_shares=min_shares,
        max_shares=max_shares,
    )
    return normalized, debug_factors


def _update_adaptive_opponent_scheduler(rl_cfg, scheduler_state, opponent_results, iteration_num):
    state = dict(scheduler_state or {})
    if not bool(rl_cfg.get('self_play_opponent_adaptive_enabled', False)):
        return state, {}

    update_every = max(1, int(rl_cfg.get('self_play_opponent_adaptive_update_every', 2)))
    min_games = max(1, int(rl_cfg.get('self_play_opponent_adaptive_min_games', 8)))
    history_size = max(1, int(rl_cfg.get('self_play_opponent_adaptive_history_size', 4)))
    bucket_score_history = state.get("bucket_score_history")
    if not isinstance(bucket_score_history, dict):
        bucket_score_history = state.get("score_history")
    if not isinstance(bucket_score_history, dict):
        bucket_score_history = {}
    exact_score_history = state.get("score_history_exact")
    if not isinstance(exact_score_history, dict):
        exact_score_history = {}
    exact_draw_history = state.get("draw_history_exact")
    if not isinstance(exact_draw_history, dict):
        exact_draw_history = {}
    exact_games_history = state.get("games_history_exact")
    if not isinstance(exact_games_history, dict):
        exact_games_history = {}
    bucket_draw_history = state.get("bucket_draw_history")
    if not isinstance(bucket_draw_history, dict):
        bucket_draw_history = {}

    observed_scores = {}
    exact_observed_scores = {}
    exact_observed_games = {}
    observed_draw_rates = {}
    exact_observed_draw_rates = {}
    bucket_stats = {}
    for label, stats in dict(opponent_results or {}).items():
        games = int((stats or {}).get("games", 0))
        if games >= min_games:
            score_rate = _safe_score_rate(
                (stats or {}).get("wins", 0),
                (stats or {}).get("draws", 0),
                (stats or {}).get("losses", 0),
            )
            if score_rate is not None:
                exact_observed_scores[str(label)] = float(score_rate)
                exact_observed_games[str(label)] = int(games)
            draw_rate = _safe_draw_rate(
                (stats or {}).get("wins", 0),
                (stats or {}).get("draws", 0),
                (stats or {}).get("losses", 0),
            )
            if draw_rate is not None:
                exact_observed_draw_rates[str(label)] = float(draw_rate)
        bucket = _canonicalize_opponent_bucket(label)
        bucket_entry = bucket_stats.setdefault(
            bucket,
            {"wins": 0, "draws": 0, "losses": 0, "games": 0},
        )
        bucket_entry["wins"] += int((stats or {}).get("wins", 0))
        bucket_entry["draws"] += int((stats or {}).get("draws", 0))
        bucket_entry["losses"] += int((stats or {}).get("losses", 0))
        bucket_entry["games"] += int((stats or {}).get("games", 0))

    for label, stats in bucket_stats.items():
        games = int((stats or {}).get("games", 0))
        if games < min_games:
            continue
        score_rate = _safe_score_rate(
            (stats or {}).get("wins", 0),
            (stats or {}).get("draws", 0),
            (stats or {}).get("losses", 0),
        )
        if score_rate is None:
            continue
        observed_scores[str(label)] = float(score_rate)
        draw_rate = _safe_draw_rate(
            (stats or {}).get("wins", 0),
            (stats or {}).get("draws", 0),
            (stats or {}).get("losses", 0),
        )
        if draw_rate is not None:
            observed_draw_rates[str(label)] = float(draw_rate)
    if (int(iteration_num) % update_every) != 0:
        state["bucket_score_history"] = bucket_score_history
        state["score_history_exact"] = exact_score_history
        state["draw_history_exact"] = exact_draw_history
        state["games_history_exact"] = exact_games_history
        state["bucket_draw_history"] = bucket_draw_history
        state["score_history"] = bucket_score_history
        return state, observed_scores

    for label, score_rate in exact_observed_scores.items():
        history = deque(exact_score_history.get(str(label), []), maxlen=history_size)
        history.append(float(score_rate))
        exact_score_history[str(label)] = list(history)
    for label, draw_rate in exact_observed_draw_rates.items():
        history = deque(exact_draw_history.get(str(label), []), maxlen=history_size)
        history.append(float(draw_rate))
        exact_draw_history[str(label)] = list(history)
    for label, games in exact_observed_games.items():
        history = deque(exact_games_history.get(str(label), []), maxlen=history_size)
        history.append(float(games))
        exact_games_history[str(label)] = list(history)

    for label, score_rate in observed_scores.items():
        history = deque(bucket_score_history.get(str(label), []), maxlen=history_size)
        history.append(float(score_rate))
        bucket_score_history[str(label)] = list(history)
    for label, draw_rate in observed_draw_rates.items():
        history = deque(bucket_draw_history.get(str(label), []), maxlen=history_size)
        history.append(float(draw_rate))
        bucket_draw_history[str(label)] = list(history)

    state["bucket_score_history"] = bucket_score_history
    state["score_history_exact"] = exact_score_history
    state["draw_history_exact"] = exact_draw_history
    state["games_history_exact"] = exact_games_history
    state["bucket_draw_history"] = bucket_draw_history
    state["score_history"] = bucket_score_history
    return state, observed_scores


def _observed_result_metrics(opponent_results, label):
    stats = dict((opponent_results or {}).get(str(label), {}) or {})
    games = int(stats.get("games", 0) or 0)
    if games <= 0:
        return None, None, None
    score_rate = _safe_score_rate(
        stats.get("wins", 0),
        stats.get("draws", 0),
        stats.get("losses", 0),
    )
    draw_rate = _safe_draw_rate(
        stats.get("wins", 0),
        stats.get("draws", 0),
        stats.get("losses", 0),
    )
    return score_rate, draw_rate, games


def refresh_recent_pool_debug_after_selfplay(
    rl_cfg,
    selected_recent_pool,
    scheduler_state,
    opponent_results,
):
    """Refresh recent-pool debug rows with the just-finished self-play results."""
    selected_recent_pool = list(selected_recent_pool or [])
    if not selected_recent_pool:
        return []

    scheduler_state = dict(scheduler_state or {})
    exact_score_history = scheduler_state.get("score_history_exact", {}) or {}
    exact_draw_history = scheduler_state.get("draw_history_exact", {}) or {}
    exact_games_history = scheduler_state.get("games_history_exact", {}) or {}

    target_score = float(rl_cfg.get('self_play_opponent_adaptive_target_score', 0.50))
    band = max(0.05, float(rl_cfg.get('self_play_opponent_adaptive_band', 0.15)))
    recent_recency_bias = max(0.0, float(rl_cfg.get('self_play_recent_recency_bias', 0.35)))
    recent_min_games_for_confidence = max(
        1.0,
        float(rl_cfg.get('self_play_recent_confidence_games', 24)),
    )
    recent_uncertainty_bonus = max(
        0.0,
        min(0.5, float(rl_cfg.get('self_play_recent_uncertainty_bonus', 0.15))),
    )
    recent_in_band_boost = max(
        1.0,
        float(rl_cfg.get('self_play_recent_in_band_boost', 1.10)),
    )
    recent_min_score = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_recent_min_score', 0.0))))
    recent_max_score = max(recent_min_score, min(1.0, float(rl_cfg.get('self_play_opponent_recent_max_score', 1.0))))
    target_draw_rate = max(0.0, min(1.0, float(rl_cfg.get('self_play_opponent_target_draw_rate', 0.45))))
    draw_band = max(0.01, float(rl_cfg.get('self_play_opponent_draw_band', 0.20)))
    draw_penalty_min_factor = max(0.20, min(1.0, float(rl_cfg.get('self_play_opponent_draw_penalty_min_factor', 0.70))))

    refreshed = []
    for entry in selected_recent_pool:
        entry = dict(entry or {})
        label = str(entry.get("label", ""))
        observed_score, observed_draw, observed_games = _observed_result_metrics(opponent_results, label)

        avg_score = observed_score
        if avg_score is None:
            avg_score = _average_score_from_history(exact_score_history.get(label, []) or [])
        avg_draw_rate = observed_draw
        if avg_draw_rate is None:
            avg_draw_rate = _average_score_from_history(exact_draw_history.get(label, []) or [])
        avg_games = observed_games
        if avg_games is None:
            avg_games = _average_score_from_history(exact_games_history.get(label, []) or [])

        in_band = True
        if avg_score is not None and (avg_score < recent_min_score or avg_score > recent_max_score):
            in_band = False
        draw_penalty = _draw_heaviness_penalty(
            avg_draw_rate,
            target_draw_rate,
            draw_band,
            draw_penalty_min_factor,
        )
        confidence = 0.0
        if avg_games is not None:
            confidence = min(1.0, float(avg_games) / recent_min_games_for_confidence)
        uncertainty_bonus = (1.0 - confidence) * recent_uncertainty_bonus
        match_quality = _score_closeness(avg_score, target_score, band)
        if in_band:
            match_quality = min(1.0, match_quality * recent_in_band_boost)
        recency_bias = float(entry.get("recency_bias", 0.0) or 0.0)
        selection_score = float(
            (
                (1.0 - recent_recency_bias) * match_quality
                + recent_recency_bias * recency_bias
                + uncertainty_bonus
            )
            * draw_penalty
        )

        entry["avg_score"] = avg_score
        entry["avg_draw_rate"] = avg_draw_rate
        entry["avg_games"] = avg_games
        entry["selection_score"] = selection_score
        entry["draw_penalty"] = float(draw_penalty)
        entry["in_band"] = bool(in_band)
        entry["observed_games"] = int(observed_games or 0)
        refreshed.append(entry)

    return refreshed


def _build_selfplay_opponent_assignments(
    rl_cfg,
    worker_specs,
    best_model_state=None,
    recent_snapshot_pool=None,
    adaptive_scheduler_state=None,
):
    enabled = bool(rl_cfg.get('self_play_opponent_pool_enabled', False))
    if not enabled or not worker_specs:
        return {}, {}, {}

    candidates = _build_selfplay_opponent_candidates(
        rl_cfg,
        best_model_state=best_model_state,
        recent_snapshot_pool=recent_snapshot_pool,
        scheduler_state=adaptive_scheduler_state,
    )
    if not candidates:
        return {}, {}, {}

    source_weights, _adaptive_debug = _compute_adaptive_opponent_weights(
        rl_cfg,
        candidates,
        scheduler_state=adaptive_scheduler_state,
    )

    total_games = int(sum(max(0, int(games)) for _, games in worker_specs))
    if total_games <= 0:
        return {}, {}, {}

    source_labels = [str(label) for label in source_weights.keys()]
    target_games = _allocate_counts_from_weights(
        source_labels,
        [float(source_weights.get(label, 0.0)) for label in source_labels],
        total_games,
    )

    bucket_game_plan = []
    for label, count in target_games.items():
        bucket_game_plan.extend([str(label)] * max(0, int(count)))
    if len(bucket_game_plan) < total_games:
        order = sorted(source_weights.keys(), key=lambda key: (-source_weights[key], key))
        idx = 0
        while len(bucket_game_plan) < total_games and order:
            bucket_game_plan.append(str(order[idx % len(order)]))
            idx += 1
    elif len(bucket_game_plan) > total_games:
        bucket_game_plan = bucket_game_plan[:total_games]
    np.random.shuffle(bucket_game_plan)

    sorted_workers = sorted(worker_specs, key=lambda item: (int(item[0])))
    payloads_by_label = {
        str(candidate["label"]): candidate.get("payload")
        for candidate in candidates
    }
    recent_entries = list((payloads_by_label.get("recent") or {}).get("entries", []) or [])
    recent_weights = [float(entry.get("weight", 1.0)) for entry in recent_entries]
    recent_bucket_total = int(sum(1 for label in bucket_game_plan if str(label) == "recent"))
    recent_quota = _allocate_counts_from_weights(
        [str(entry.get("label", "recent")) for entry in recent_entries],
        recent_weights,
        recent_bucket_total,
    )
    recent_quota = _cap_recent_latest_counts(
        recent_entries,
        recent_quota,
        total_games,
        rl_cfg,
    )
    recent_label_plan = []
    for entry in recent_entries:
        entry_label = str(entry.get("label", "recent"))
        recent_label_plan.extend([entry_label] * max(0, int(recent_quota.get(entry_label, 0))))
    if len(recent_label_plan) < recent_bucket_total and recent_entries:
        top_label = str(recent_entries[0].get("label", "recent"))
        recent_label_plan.extend([top_label] * int(recent_bucket_total - len(recent_label_plan)))
    np.random.shuffle(recent_label_plan)
    recent_label_cursor = 0

    assignments = {}
    assigned_counts = defaultdict(int)
    debug_info = {
        "source_weights": {str(k): float(v) for k, v in source_weights.items()},
        "target_games": {str(k): int(v) for k, v in target_games.items()},
        "selected_recent_pool": [],
        "recent_pool_all": [],
        "model_fallbacks": {},
    }
    offset = 0
    for rank, games_for_worker in sorted_workers:
        worker_plan_buckets = list(bucket_game_plan[offset: offset + int(games_for_worker)])
        offset += int(games_for_worker)
        plan_labels = []
        pool_entries = {}
        for bucket_label in worker_plan_buckets:
            if bucket_label == "current":
                plan_labels.append("current")
                assigned_counts["current"] += 1
                continue
            if bucket_label == "best":
                best_payload = payloads_by_label.get("best") or {}
                best_label = str(best_payload.get("label", "best"))
                best_state = best_payload.get("state")
                if best_state is None:
                    plan_labels.append("current")
                    assigned_counts["current"] += 1
                    debug_info["model_fallbacks"][best_label] = (
                        int(debug_info["model_fallbacks"].get(best_label, 0)) + 1
                    )
                    continue
                plan_labels.append(best_label)
                pool_entries[best_label] = best_state
                assigned_counts[best_label] += 1
                continue
            if bucket_label == "recent" and recent_entries:
                if recent_label_cursor < len(recent_label_plan):
                    chosen_label = str(recent_label_plan[recent_label_cursor])
                    recent_label_cursor += 1
                else:
                    chosen_label = str(recent_entries[0].get("label", "recent"))
                chosen_state = None
                for entry in recent_entries:
                    if str(entry.get("label", "recent")) == chosen_label:
                        chosen_state = entry.get("state")
                        break
                if chosen_state is None:
                    plan_labels.append("current")
                    assigned_counts["current"] += 1
                    debug_info["model_fallbacks"][chosen_label] = (
                        int(debug_info["model_fallbacks"].get(chosen_label, 0)) + 1
                    )
                    continue
                plan_labels.append(chosen_label)
                pool_entries[chosen_label] = chosen_state
                assigned_counts[chosen_label] += 1
                continue
            plan_labels.append("current")
            assigned_counts["current"] += 1
            debug_info["model_fallbacks"][str(bucket_label)] = (
                int(debug_info["model_fallbacks"].get(str(bucket_label), 0)) + 1
            )

        non_current_labels = {
            str(label)
            for label in plan_labels
            if str(label) != "current"
        }
        missing_labels = sorted(label for label in non_current_labels if label not in pool_entries)
        if missing_labels:
            for label in missing_labels:
                debug_info["model_fallbacks"][label] = (
                    int(debug_info["model_fallbacks"].get(label, 0)) + plan_labels.count(label)
                )
            plan_labels = [
                "current" if str(label) in set(missing_labels) else str(label)
                for label in plan_labels
            ]

        assignments[int(rank)] = {
            "plan_labels": plan_labels,
            "pool_entries": [
                {"label": label, "state": state}
                for label, state in pool_entries.items()
            ],
        }

    debug_info["assigned_games"] = {str(k): int(v) for k, v in assigned_counts.items()}
    debug_info["assigned_total_games"] = int(sum(assigned_counts.values()))
    debug_info["expected_total_games"] = int(total_games)

    recent_payload = payloads_by_label.get("recent") or {}
    for entry in list(recent_payload.get("entries", []) or []):
        debug_info["selected_recent_pool"].append({
            "label": str(entry.get("label", "")),
            "avg_score": entry.get("avg_score", None),
            "avg_draw_rate": entry.get("avg_draw_rate", None),
            "avg_games": entry.get("avg_games", None),
            "selection_score": entry.get("selection_score", None),
            "draw_penalty": entry.get("draw_penalty", None),
            "recency_bias": entry.get("recency_bias", None),
        })
    for entry in list(recent_payload.get("all_entries", []) or []):
        debug_info["recent_pool_all"].append({
            "label": str(entry.get("label", "")),
            "avg_score": entry.get("avg_score", None),
            "avg_draw_rate": entry.get("avg_draw_rate", None),
            "avg_games": entry.get("avg_games", None),
            "selection_score": entry.get("selection_score", None),
            "draw_penalty": entry.get("draw_penalty", None),
            "recency_bias": entry.get("recency_bias", None),
            "in_band": bool(entry.get("in_band", False)),
        })

    return assignments, dict(assigned_counts), debug_info



