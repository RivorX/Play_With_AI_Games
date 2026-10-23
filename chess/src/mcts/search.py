"""Batched Gumbel AlphaZero search for RL self-play.

Sequential Halving selects root actions and completed-Q policy improvement is
used below the root and for replay targets.
"""

import errno
import hashlib
import logging
import math
import os
import shutil
import struct
import tempfile
import threading
import time
import weakref
import zlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.common.syzygy import resolve_syzygy_paths, syzygy_piece_counts
from src.inference.prediction import (
    copy_compact_policy_value_to_cpu,
    predict_raw,
    value_logits_to_scalar,
)
from src.game import backend as chess
from src.common.torch_cache import configure_torch_compile_cache, torch_compile_cache_paths
from src.models.data.se_cnn_v9.helpers import (
    ACTION_SIZE,
    MAX_LEGAL_MOVES,
    _move_to_index_cached,
    board_to_tensor,
    board_to_tensor_pair,
    move_to_index,
)
from src.models.data.registry import get_input_encoder
from src.mcts.native import get_native_mcts
from src.mcts.result import SearchResult
from src.mcts.q_delta import USEFUL_SEARCH_Q_DELTA_MIN


_EMPTY_HISTORY_TENSOR = np.zeros((16, 8, 8), dtype=np.float32)
_INPUT_PIECE_SPECS = tuple(
    (color, piece_type)
    for color in (chess.WHITE, chess.BLACK)
    for piece_type in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )
)
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

_SYZYGY_ORACLE_CACHE = {}
_SELFPLAY_COMPILE_LOCKFILE = "selfplay_torch_compile.lock"
# bulletchess Move objects are immutable and hash by move geometry.  Policy
# indices therefore need to be derived from origin/destination/promotion only
# once per colour and worker process, rather than once for every expanded leaf.
_MOVE_POLICY_INDEX_CACHE = ({}, {})
_DEFAULT_REPLAY_MAX_POLICY_TARGETS = 256
_REPLAY_SOURCE_UNKNOWN = 0
_REPLAY_SOURCE_LEARNER = 1
_REPLAY_SOURCE_FROZEN_BEST = 2
_DEFAULT_LOCAL_CUDA_GRAPH_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 8, 16)
_FEEDBACK_MULTIROOT_LOCAL_CUDA_GRAPH_BATCH_SIZES = (1, 2, 4, 6, 8, 12, 14, 16)

# Gumbel search is the policy-improvement operator. Target confidence controls
# the regular policy CE weight, while useful top-move changes are handled by the
# separate bounded rank objective. Do not boost the same correction again here:
# doing so stacks with policy-surprise sampling and lets a minority of rows own
# most of the policy gradient.
_DYNAMIC_BUDGET_MARGINAL_PENALTY = 0.35
_DYNAMIC_BUDGET_TARGET_CHUNKS = 12
_DYNAMIC_BUDGET_MIN_FRESH_LEAVES_PER_FINALIST = 2
_DYNAMIC_BUDGET_EARLY_STOP_Q_GAP = 0.12
_DYNAMIC_BUDGET_REUSE_FRESH_FLOOR_BONUS = 32
_DYNAMIC_BUDGET_CREDIT_KEYS = 4096
_TREE_REUSE_MIN_FRESH_SIMULATIONS = 64
_TREE_REUSE_CONSERVATIVE_FRESH_FRACTION = 0.50
_TREE_REUSE_HIGH_QUALITY_FRESH_FRACTION = 1.0 / 3.0
_TREE_REUSE_MAX_VISIT_CREDIT_DISCOUNT = 0.75
_TREE_REUSE_CANDIDATE_SUPPORT_VISITS = 2.0
_TREE_REUSE_SCOUT_SIMULATIONS = 32
_TREE_REUSE_SCOUT_MIN_TREE_QUALITY = 0.70
_TREE_REUSE_SCOUT_MIN_CANDIDATE_COVERAGE = 0.75
_TREE_REUSE_SCOUT_STABILITY_MIN = 0.70
_TREE_REUSE_SCOUT_POLICY_JS_SCALE = 0.05
_TREE_REUSE_SCOUT_MIN_FRESH_FRACTION = 1.0 / 6.0
# At the low end of the dynamic budget, one noisy leaf evaluation per root
# candidate is too little evidence for the first Sequential-Halving cut. Keep
# the configured action count as a cap and narrow only when the *fresh* action
# budget cannot fund two complete first-round passes.
_GUMBEL_MIN_INITIAL_VISITS_PER_ACTION = 2


_POLICY_TARGET_MODES = frozenset({'baseline', 'challenge_delta'})
_POLICY_TARGET_DEFAULT_PRIOR_FRACTION = 0.5


def _resolve_policy_target_config(rl_cfg):
    """Validate the optional one-axis stored policy-target experiment."""
    rl_cfg = rl_cfg if isinstance(rl_cfg, dict) else {}
    mode = str(rl_cfg.get('mcts_policy_target_mode', 'baseline')).strip().lower()
    if mode not in _POLICY_TARGET_MODES:
        choices = ', '.join(sorted(_POLICY_TARGET_MODES))
        raise ValueError(
            f"reinforcement_learning.search.policy_target_mode must be one of {choices}; got {mode!r}"
        )
    try:
        prior_fraction = float(rl_cfg.get(
            'mcts_challenge_delta_prior_fraction',
            _POLICY_TARGET_DEFAULT_PRIOR_FRACTION,
        ))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'reinforcement_learning.search.challenge_delta_prior_fraction must be a finite number in (0, 1)'
        ) from exc
    if not math.isfinite(prior_fraction) or not 0.0 < prior_fraction < 1.0:
        raise ValueError(
            'reinforcement_learning.search.challenge_delta_prior_fraction must be a finite number in (0, 1)'
        )
    return mode, prior_fraction


def _build_challenge_delta_policy_target(
    policy_target,
    prior_distribution,
    *,
    changed_top,
    q_delta,
    prior_fraction=_POLICY_TARGET_DEFAULT_PRIOR_FRACTION,
):
    """Subtract prior-aligned mass from a useful completed-Q correction.

    For normalized completed-Q target ``pi`` and prior ``p``, the treatment is
    ``q_i = max(pi_i - alpha * p_i, 0) / Z``. Invalid inputs, unchanged top
    moves, or weak Q evidence return the original target unchanged.
    """
    fallback = {'applied': False, 'removed_prior_mass': 0.0}
    if not changed_top:
        return policy_target, fallback
    try:
        delta = float(q_delta)
        alpha = float(prior_fraction)
    except (TypeError, ValueError):
        return policy_target, fallback
    if (
        not math.isfinite(delta)
        or delta <= USEFUL_SEARCH_Q_DELTA_MIN
        or not math.isfinite(alpha)
        or not 0.0 < alpha < 1.0
        or not isinstance(policy_target, dict)
        or not policy_target
        or not isinstance(prior_distribution, dict)
        or not prior_distribution
    ):
        return policy_target, fallback
    try:
        if not set(policy_target).issubset(prior_distribution):
            return policy_target, fallback
        target_values = {move: float(value) for move, value in policy_target.items()}
        prior_values = {move: float(value) for move, value in prior_distribution.items()}
    except (TypeError, ValueError):
        return policy_target, fallback
    if (
        any(not math.isfinite(value) or value < 0.0 for value in target_values.values())
        or any(not math.isfinite(value) or value < 0.0 for value in prior_values.values())
    ):
        return policy_target, fallback
    try:
        target_total = math.fsum(target_values.values())
        prior_total = math.fsum(prior_values.values())
    except OverflowError:
        return policy_target, fallback
    if (
        not math.isfinite(target_total)
        or not math.isfinite(prior_total)
        or target_total <= 1e-12
        or prior_total <= 1e-12
    ):
        return policy_target, fallback

    transformed = {}
    removed_prior_mass = 0.0
    for move, prior_value in prior_values.items():
        prior_probability = prior_value / prior_total
        search_probability = target_values.get(move, 0.0) / target_total
        retained = max(0.0, search_probability - alpha * prior_probability)
        removed_prior_mass += search_probability - retained
        if retained > 0.0:
            transformed[move] = retained
    transformed_total = math.fsum(transformed.values())
    if not transformed or not math.isfinite(transformed_total) or transformed_total <= 1e-12:
        return policy_target, fallback
    normalized = {
        move: probability / transformed_total
        for move, probability in transformed.items()
    }
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        return policy_target, fallback
    return normalized, {
        'applied': True,
        'removed_prior_mass': float(max(0.0, min(1.0, removed_prior_mass))),
    }


def _summarize_tree_reuse_quality(root, gumbel, max_considered_actions):
    """Measure whether inherited work is useful to the *new* root search.

    Raw visit count is insufficient: a large subtree can still contain almost
    no information about the actions selected by the new root's independent
    Gumbel Top-k.  Candidate coverage and support are therefore measured after
    sampling the new Gumbel noise.  Prior mass and effective action count keep
    the score sensitive to broader, reusable search information without making
    a deliberately concentrated, clear position automatically worthless.
    """
    empty = {
        'quality': 0.0,
        'candidate_coverage': 0.0,
        'candidate_support': 0.0,
        'visited_prior_mass': 0.0,
        'effective_action_ratio': 0.0,
        'considered_actions': 0,
    }
    if root is None or not root.expanded or root.edges is None or not root.edges.moves:
        return empty

    visits = np.maximum(
        0.0,
        np.asarray(root.edges.visit_counts, dtype=np.float64),
    )
    priors = np.asarray(root.edges.base_priors, dtype=np.float64)
    if visits.size == 0 or priors.size != visits.size or float(visits.sum()) <= 0.0:
        return empty
    prior_total = float(priors.sum())
    if prior_total <= 0.0 or not np.isfinite(prior_total):
        probs = np.full(visits.size, 1.0 / float(visits.size), dtype=np.float64)
    else:
        probs = np.clip(priors / prior_total, 1e-12, 1.0)
        probs /= float(probs.sum())

    considered = min(
        visits.size,
        max(1, int(max_considered_actions)),
    )
    noise = np.asarray(gumbel, dtype=np.float64)
    if noise.size != visits.size:
        noise = np.zeros(visits.size, dtype=np.float64)
    candidate_scores = np.log(np.maximum(probs, np.finfo(np.float64).tiny)) + noise
    if considered >= visits.size:
        candidate_indices = np.arange(visits.size, dtype=np.int64)
    else:
        candidate_indices = np.argpartition(candidate_scores, -considered)[-considered:]
    candidate_visits = visits[candidate_indices]
    candidate_coverage = float(np.mean(candidate_visits > 0.0))
    candidate_support = float(np.mean(np.minimum(
        1.0,
        candidate_visits / _TREE_REUSE_CANDIDATE_SUPPORT_VISITS,
    )))
    visited_prior_mass = float(probs[visits > 0.0].sum())
    total_visits = float(visits.sum())
    squared_sum = float(np.square(visits).sum())
    effective_actions = (
        (total_visits * total_visits) / squared_sum
        if squared_sum > 0.0 else 0.0
    )
    effective_action_ratio = min(1.0, effective_actions / float(considered))
    quality = (
        0.45 * candidate_coverage
        + 0.30 * candidate_support
        + 0.15 * visited_prior_mass
        + 0.10 * effective_action_ratio
    )
    return {
        'quality': float(max(0.0, min(1.0, quality))),
        'candidate_coverage': float(max(0.0, min(1.0, candidate_coverage))),
        'candidate_support': float(max(0.0, min(1.0, candidate_support))),
        'visited_prior_mass': float(max(0.0, min(1.0, visited_prior_mass))),
        'effective_action_ratio': float(max(0.0, min(1.0, effective_action_ratio))),
        'considered_actions': int(considered),
    }


def _resolve_tree_reuse_fresh_floor(budget, quality):
    budget = max(1, int(budget))
    quality = max(0.0, min(1.0, float(quality)))
    fresh_fraction = (
        _TREE_REUSE_CONSERVATIVE_FRESH_FRACTION
        - quality * (
            _TREE_REUSE_CONSERVATIVE_FRESH_FRACTION
            - _TREE_REUSE_HIGH_QUALITY_FRESH_FRACTION
        )
    )
    return max(
        min(budget, _TREE_REUSE_MIN_FRESH_SIMULATIONS),
        int(math.ceil(budget * fresh_fraction)),
    )


def _resolve_tree_reuse_visit_credit(
    budget,
    inherited_visits,
    *,
    quality,
):
    """Credit only inherited work supported by the new Gumbel candidate set.

    An inherited visit contains a valid evaluation and backup, but it was
    selected while this node was below the previous root.  It therefore lacks
    the new root's Gumbel allocation and is not equivalent to a fresh root
    simulation.  Quality controls both its discount and the fresh-search floor:
    weak/narrow trees trigger more new work, while broad, relevant subtrees can
    safely replace more GPU evaluations.
    """
    budget = max(1, int(budget))
    inherited_visits = max(0, int(inherited_visits))
    if inherited_visits <= 0:
        return 0
    quality = max(0.0, min(1.0, float(quality)))
    fresh_floor = _resolve_tree_reuse_fresh_floor(budget, quality)
    # Reuse quality must gate whether inherited work counts at all.  The old
    # 0.50 base credited half of a subtree even at quality == 0, so narrow trees
    # routinely replaced fresh root search despite missing the new Gumbel
    # candidate set.  Keep the throughput win for broad trees, but make the
    # credit proportional to measured relevance.
    visit_discount = quality * _TREE_REUSE_MAX_VISIT_CREDIT_DISCOUNT
    discounted_visits = int(math.floor(inherited_visits * visit_discount))
    return min(discounted_visits, max(0, budget - fresh_floor))


def _policy_jensen_shannon_divergence(before, after):
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.size == 0 or before.size != after.size:
        return 1.0
    before = np.maximum(before, np.finfo(np.float64).tiny)
    after = np.maximum(after, np.finfo(np.float64).tiny)
    before /= float(before.sum())
    after /= float(after.sum())
    midpoint = 0.5 * (before + after)
    divergence = 0.5 * float(np.sum(before * np.log(before / midpoint)))
    divergence += 0.5 * float(np.sum(after * np.log(after / midpoint)))
    return float(max(0.0, min(math.log(2.0), divergence)))


def _resolve_tree_reuse_post_scout_credit(
    budget,
    inherited_visits,
    initial_credit,
    *,
    tree_quality,
    winner_stable,
    policy_js_divergence,
):
    """Grant extra credit only after a fresh root-level stability audit."""
    budget = max(1, int(budget))
    inherited_visits = max(0, int(inherited_visits))
    initial_credit = max(0, int(initial_credit))
    tree_quality = max(0.0, min(1.0, float(tree_quality)))
    base_floor = _resolve_tree_reuse_fresh_floor(budget, tree_quality)
    result = {
        'extra_credit': 0,
        'total_credit': initial_credit,
        'fresh_floor': base_floor,
        'policy_stability': 0.0,
        'combined_stability': 0.0,
        'early_stop': False,
    }
    if inherited_visits <= 0 or not winner_stable:
        return result

    policy_js_divergence = max(0.0, float(policy_js_divergence))
    policy_stability = math.exp(
        -policy_js_divergence / _TREE_REUSE_SCOUT_POLICY_JS_SCALE
    )
    combined_stability = tree_quality * policy_stability
    result['policy_stability'] = float(max(0.0, min(1.0, policy_stability)))
    result['combined_stability'] = float(max(0.0, min(1.0, combined_stability)))
    if combined_stability < _TREE_REUSE_SCOUT_STABILITY_MIN:
        return result

    confidence = (
        (combined_stability - _TREE_REUSE_SCOUT_STABILITY_MIN)
        / (1.0 - _TREE_REUSE_SCOUT_STABILITY_MIN)
    )
    aggressive_floor = max(
        min(budget, _TREE_REUSE_SCOUT_SIMULATIONS),
        int(math.ceil(budget * _TREE_REUSE_SCOUT_MIN_FRESH_FRACTION)),
    )
    fresh_floor = int(math.ceil(
        base_floor - confidence * max(0, base_floor - aggressive_floor)
    ))
    total_credit = min(
        int(math.floor(inherited_visits * combined_stability)),
        max(0, budget - fresh_floor),
    )
    total_credit = max(initial_credit, total_credit)
    result['extra_credit'] = max(0, total_credit - initial_credit)
    result['total_credit'] = total_credit
    result['fresh_floor'] = fresh_floor
    result['early_stop'] = result['extra_credit'] > 0
    return result


def _resolve_dynamic_simulation_budget(
    simulations,
    *,
    minimum=64,
    maximum_multiplier=5.0 / 3.0,
):
    """Resolve a scalable min/target/max grid from the base simulations.

    ``simulations`` is the desired batch average. The minimum remains an
    absolute quality floor so easy positions can stay cheap when the base
    budget grows. The upper bound and chunk size scale automatically.
    """
    target = max(1, int(simulations))
    minimum = min(target, max(1, int(minimum)))
    maximum_multiplier = max(1.0, float(maximum_multiplier))
    requested_maximum = max(target, int(round(target * maximum_multiplier)))
    span = max(0, requested_maximum - minimum)
    if span <= 0:
        return minimum, target, target, 1
    chunk = max(1, target // _DYNAMIC_BUDGET_TARGET_CHUNKS)
    maximum = minimum + (span // chunk) * chunk
    return minimum, target, maximum, chunk


def _allocate_dynamic_simulation_budgets(
    difficulty_scores,
    *,
    minimum=64,
    target_average=192,
    maximum=320,
    chunk=64,
):
    """Allocate difficulty-monotonic simulation chunks at an exact total.

    Each possible extra chunk receives a positive, diminishing marginal utility
    derived from position difficulty. Globally selecting the best marginal
    chunks is a bounded water-filling allocator: harder roots cannot receive
    less compute than easier roots, but one root also cannot consume the whole
    group. A final partial chunk preserves the requested integer total exactly.
    """
    scores = [float(max(0.0, min(1.0, score))) for score in difficulty_scores]
    if not scores:
        return []
    minimum = max(1, int(minimum))
    maximum = max(minimum, int(maximum))
    chunk = max(1, int(chunk))
    target_average = max(float(minimum), min(float(maximum), float(target_average)))
    max_chunks = max(0, (maximum - minimum) // chunk)
    target_total = int(round(target_average * len(scores)))
    target_extra = max(0, target_total - minimum * len(scores))
    target_chunks = max(0, min(max_chunks * len(scores), target_extra // chunk))

    priorities = [max(0.02, score) ** 2 for score in scores]
    marginal_chunks = []
    for root_idx, score in enumerate(scores):
        for chunk_idx in range(max_chunks):
            marginal_score = priorities[root_idx] / (
                1.0 + _DYNAMIC_BUDGET_MARGINAL_PENALTY * float(chunk_idx)
            )
            marginal_chunks.append((marginal_score, -root_idx, chunk_idx, root_idx))
    marginal_chunks.sort(reverse=True)

    allocated_chunks = [0] * len(scores)
    for _, _, _, root_idx in marginal_chunks[:target_chunks]:
        allocated_chunks[root_idx] += 1
    budgets = [minimum + chunk * count for count in allocated_chunks]

    remaining = max(0, target_extra - sum(budget - minimum for budget in budgets))
    if remaining > 0:
        difficulty_order = sorted(
            range(len(scores)),
            key=lambda idx: (scores[idx], -idx),
            reverse=True,
        )
        for root_idx in difficulty_order:
            capacity = max(0, maximum - budgets[root_idx])
            if capacity <= 0:
                continue
            granted = min(capacity, remaining)
            budgets[root_idx] += granted
            remaining -= granted
            if remaining <= 0:
                break
    # Integer remainders can otherwise create a one-simulation inversion between
    # two nearly equal roots. Sorting the already-computed budget multiset by
    # difficulty preserves bounds and the exact total while making the contract
    # strictly monotonic (ties are stable by original index).
    difficulty_order = sorted(range(len(scores)), key=lambda idx: (scores[idx], idx))
    sorted_budgets = sorted(budgets)
    monotonic_budgets = [minimum] * len(scores)
    for root_idx, budget in zip(difficulty_order, sorted_budgets):
        monotonic_budgets[root_idx] = int(budget)
    return monotonic_budgets


def _allocate_eval_easy_cut_simulation_budgets(
    difficulty_scores,
    *,
    target=192,
    minimum_fraction=5.0 / 6.0,
    difficulty_threshold=0.75,
    chunk=16,
):
    """Cut only clearly easy eval roots; never boost another root in exchange."""
    target = max(1, int(target))
    chunk = max(1, min(target, int(chunk)))
    minimum_fraction = max(0.50, min(1.0, float(minimum_fraction)))
    minimum = max(chunk, int(round(target * minimum_fraction / chunk)) * chunk)
    minimum = min(target, minimum)
    threshold = max(1e-6, min(1.0, float(difficulty_threshold)))
    budgets = []
    for raw_difficulty in difficulty_scores:
        difficulty = max(0.0, min(1.0, float(raw_difficulty)))
        if difficulty >= threshold:
            budgets.append(target)
            continue
        progress = difficulty / threshold
        raw_budget = minimum + (target - minimum) * progress
        quantized = int(round(raw_budget / chunk)) * chunk
        budgets.append(max(minimum, min(target, quantized)))
    return budgets


def _gumbel_policy_target_telemetry(target_probs, prior_probs, visit_counts, value_sums):
    """Describe one stored target; attach measured Q labels only to its top move."""
    target = np.asarray(target_probs, dtype=np.float64)
    prior = np.asarray(prior_probs, dtype=np.float64)
    visits = np.asarray(visit_counts, dtype=np.float64)
    values = np.asarray(value_sums, dtype=np.float64)
    if (
        target.ndim != 1
        or target.size == 0
        or prior.shape != target.shape
        or visits.shape != target.shape
        or values.shape != target.shape
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(prior))
        or not np.all(np.isfinite(visits))
        or not np.all(np.isfinite(values))
        or np.any(target < 0.0)
        or np.any(prior < 0.0)
        or np.any(visits < 0.0)
    ):
        return None
    target_total = float(target.sum())
    prior_total = float(prior.sum())
    if target_total <= 1e-12 or prior_total <= 1e-12:
        return None
    target = target / target_total
    prior = prior / prior_total
    prior_top_idx = int(np.argmax(prior))
    target_top_idx = int(np.argmax(target))
    ordered = np.sort(target)
    top = float(ordered[-1])
    second = float(ordered[-2]) if ordered.size > 1 else 0.0
    entropy = float(-np.sum(target * np.log(np.clip(target, 1e-12, 1.0))))
    result = {
        'prior_mcts_agree': 1.0 if prior_top_idx == target_top_idx else 0.0,
        'prior_top_visit_prob': float(target[prior_top_idx]),
        'mcts_top_prior_prob': float(prior[target_top_idx]),
        'mcts_policy_kl': float(np.sum(
            target * (
                np.log(np.clip(target, 1e-12, 1.0))
                - np.log(np.clip(prior, 1e-12, 1.0))
            )
        )),
        'policy_kld': None,
        'top_visit_prob': top,
        'visit_gap': max(0.0, top - second),
        'visit_entropy': (
            entropy / max(1e-12, math.log(len(target))) if len(target) > 1 else 0.0
        ),
        'best_q': None,
        'mcts_q_delta': None,
        'mcts_changed_to_lower_q': None,
    }
    result['policy_kld'] = result['mcts_policy_kl']
    if visits[target_top_idx] > 0.0:
        target_q = -float(values[target_top_idx]) / float(visits[target_top_idx])
        if math.isfinite(target_q):
            result['best_q'] = float(max(-1.0, min(1.0, target_q)))
            if visits[prior_top_idx] > 0.0:
                prior_q = -float(values[prior_top_idx]) / float(visits[prior_top_idx])
                if math.isfinite(prior_q):
                    q_delta = float(max(-2.0, min(2.0, target_q - prior_q)))
                    result['mcts_q_delta'] = q_delta
                    result['mcts_changed_to_lower_q'] = (
                        1.0 if prior_top_idx != target_top_idx and q_delta < -0.02 else 0.0
                    )
    return result


def _clear_gumbel_policy_target_telemetry(metadata):
    """Mark target-derived measurements unavailable when the target is invalid.

    In particular, an invalid distribution cannot be counted as agreement with
    the prior: self-play aggregation treats ``prior_mcts_agree is None`` as an
    unavailable sample and excludes it from agreement/correction rates.
    """
    for key in (
        'prior_mcts_agree',
        'prior_top_visit_prob',
        'mcts_top_prior_prob',
        'mcts_policy_kl',
        'policy_kld',
        'top_visit_prob',
        'visit_gap',
        'visit_entropy',
        'best_q',
        'mcts_q_delta',
        'mcts_changed_to_lower_q',
    ):
        metadata[key] = None


def _gumbel_prior_difficulty(root):
    """Prior-only uncertainty used to assign a budget before Gumbel search.

    Sequential Halving needs to know its complete budget before it starts, so
    this score assigns compute from the clean NN prior.
    """
    if root is None or not root.expanded or root.edges is None:
        return 1.0
    priors = np.asarray(root.edges.base_priors, dtype=np.float64)
    if priors.size <= 1:
        return 0.0
    total = float(priors.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 1.0
    probs = np.clip(priors / total, 1e-12, 1.0)
    entropy = float(-(probs * np.log(probs)).sum()) / max(1e-12, math.log(probs.size))
    ordered = np.sort(probs)
    top = float(ordered[-1])
    second = float(ordered[-2]) if ordered.size > 1 else 0.0
    relative_gap = max(0.0, min(1.0, (top - second) / max(1e-12, top + second)))
    ambiguity = 1.0 - relative_gap
    branching = min(1.0, math.log1p(float(probs.size - 1)) / math.log(16.0))
    raw_value = getattr(root, 'raw_value', None)
    value_uncertainty = 1.0 - abs(float(raw_value)) if raw_value is not None else 0.5
    value_uncertainty = max(0.0, min(1.0, value_uncertainty))
    difficulty = (
        0.40 * ambiguity
        + 0.25 * entropy
        + 0.15 * (1.0 - top)
        + 0.10 * branching
        + 0.10 * value_uncertainty
    )
    return float(max(0.0, min(1.0, difficulty)))


@lru_cache(maxsize=128)
def _gumbel_considered_visit_sequence(max_num_considered_actions, num_simulations):
    """Sequential-Halving visit schedule used by DeepMind's MCTX.

    A value in the returned sequence is the fresh visit count an action must
    currently have to remain eligible at the root.  This first samples actions
    without replacement and then repeatedly halves the candidate set.
    """
    action_count = max(1, int(max_num_considered_actions))
    simulation_count = max(0, int(num_simulations))
    if simulation_count <= 0:
        return ()
    if action_count <= 1:
        return tuple(range(simulation_count))
    log2_actions = int(math.ceil(math.log2(action_count)))
    sequence = []
    visits = [0] * action_count
    considered = action_count
    while len(sequence) < simulation_count:
        extra_visits = max(1, int(simulation_count / (log2_actions * considered)))
        for _ in range(extra_visits):
            sequence.extend(visits[:considered])
            # Every emitted block represents one complete additional visit to
            # each action that is still considered.  Advancing the threshold
            # only once after ``extra_visits`` blocks repeats a stale threshold
            # and makes root selection fall back to the least-visited action;
            # that accidentally keeps eliminated actions alive instead of
            # performing Sequential Halving.
            for idx in range(considered):
                visits[idx] += 1
        considered = max(2, considered // 2)
    return tuple(sequence[:simulation_count])


@lru_cache(maxsize=256)
def _resolve_gumbel_considered_actions(
    max_num_considered_actions,
    legal_action_count,
    num_simulations,
):
    """Resolve root width from the fresh Sequential-Halving budget.

    MCTX permits a single first-round visit, so this is a chess-specific quality
    guard rather than a correctness fix. The configured value remains the cap.
    When it cannot receive two complete passes, step down to the next lower
    power of two; this avoids unstable 16 -> 15 boundary changes.
    """
    simulation_count = max(0, int(num_simulations))
    limit = min(
        max(1, int(max_num_considered_actions)),
        max(1, int(legal_action_count)),
        max(1, simulation_count),
    )
    if limit <= 1 or simulation_count <= 0:
        return 1

    rounds = int(math.ceil(math.log2(limit)))
    initial_visits = simulation_count // max(1, rounds * limit)
    if initial_visits >= _GUMBEL_MIN_INITIAL_VISITS_PER_ACTION:
        return int(limit)

    candidate = 1 << (int(limit).bit_length() - 1)
    if candidate == limit:
        candidate //= 2
    while candidate > 1:
        rounds = int(math.ceil(math.log2(candidate)))
        initial_visits = simulation_count // max(1, rounds * candidate)
        if initial_visits >= _GUMBEL_MIN_INITIAL_VISITS_PER_ACTION:
            return int(candidate)
        candidate //= 2
    return 1


def _gumbel_initial_round_visits(considered_actions, num_simulations):
    considered = max(1, int(considered_actions))
    simulations = max(0, int(num_simulations))
    if simulations <= 0:
        return 0
    if considered <= 1:
        return simulations
    rounds = int(math.ceil(math.log2(considered)))
    return max(1, simulations // max(1, rounds * considered))


def _gumbel_final_action_eligibility(visit_counts):
    """Return the MCTX final-action mask and its maximum-visit cutoff."""
    visits = np.asarray(visit_counts, dtype=np.float64)
    if visits.size == 0:
        return np.zeros(0, dtype=np.bool_), 0.0
    threshold = float(np.max(visits))
    return visits == threshold, threshold


def _adaptive_challenger_index(scores, winner):
    """Return the best-scoring legal challenger, independent of edge order."""
    values = np.asarray(scores, dtype=np.float64)
    winner = int(winner)
    if values.size < 2 or not 0 <= winner < values.size:
        return None
    challenger_scores = values.copy()
    challenger_scores[winner] = -np.inf
    challenger = int(np.argmax(challenger_scores))
    return challenger if np.isfinite(challenger_scores[challenger]) else None


def _dynamic_budget_uncertainty_score(
    score_gap,
    q_gap,
    winner_switches,
    finalist_support,
    stable_checkpoints,
    tree_reuse_quality,
):
    """Combine current decision margin, evidence, stability and retained quality."""
    def margin_uncertainty(value, scale):
        if value is None:
            return 1.0
        value = float(value)
        if not math.isfinite(value):
            return 1.0
        return 1.0 - min(1.0, max(0.0, value) / scale)

    quality = float(tree_reuse_quality)
    if not math.isfinite(quality):
        quality = 0.0
    quality = min(1.0, max(0.0, quality))
    signals = (
        margin_uncertainty(score_gap, 0.5),
        margin_uncertainty(
            q_gap,
            max(0.36, 3.0 * _DYNAMIC_BUDGET_EARLY_STOP_Q_GAP),
        ),
        min(1.0, max(0, int(winner_switches)) / 2.0),
        1.0 - min(1.0, max(0, int(finalist_support)) / 4.0),
        1.0 - min(1.0, max(0, int(stable_checkpoints) - 1) / 3.0),
        1.0 - quality,
    )
    return float(sum(signals) / len(signals))


def _provenance_count_or_none(stats, key):
    """Return an aggregate only when every inference row was attributed."""
    if stats is None or not bool(stats.get('nn_provenance_complete', False)):
        return None
    return int(stats[key])


def _gumbel_final_visit_counts(node, state):
    """Return fresh counts in the active root schedule's visit scope."""
    edges = node.edges
    initial = np.asarray(state['initial_visits'], dtype=np.float64)
    if state.get('root_schedule') == 'revisiting':
        pass_initial = np.asarray(
            state.get('pass_initial_visits', state['initial_visits']),
            dtype=np.float64,
        )
        return np.maximum(
            0.0,
            edges.visit_counts.astype(np.float64) - pass_initial,
        )
    fresh = edges.fresh_count_scratch
    np.subtract(edges.visit_counts, initial, out=fresh)
    np.maximum(fresh, 0.0, out=fresh)
    return fresh


def _gumbel_revisiting_pass_budgets(action_budget, considered_actions, max_passes=3):
    """Split a Gumbel root budget into bounded, complete re-ranking passes.

    Each pass gets enough selections to complete one visit-count layer for
    every Sequential-Halving stage at the resolved width. The candidate width
    is resolved once from the full action budget; revisits do not silently
    shrink the root shortlist. Any remainder is distributed across the first
    passes, and the returned budgets sum exactly to ``action_budget``.
    """
    budget = max(0, int(action_budget))
    width = max(1, int(considered_actions))
    if budget <= 0:
        return ()
    if width <= 1:
        return (budget,)

    halving_stages = max(1, int(math.ceil(math.log2(width))))
    minimum_pass_budget = max(1, width * halving_stages)
    pass_count = min(
        max(1, int(max_passes)),
        max(1, budget // minimum_pass_budget),
    )
    base, remainder = divmod(budget, pass_count)
    return tuple(
        int(base + (1 if pass_index < remainder else 0))
        for pass_index in range(pass_count)
    )


def _gumbel_revisiting_feedback_boundaries(considered_actions, num_simulations):
    """Return visit offsets where Sequential Halving actually removes actions.

    Batches may cross repeated visit-count layers while the candidate width is
    fixed. They must end at each real width reduction so the next stage sees
    the completed evidence. Once the schedule reaches two actions, it no
    longer halves; the remaining pass can run in ordinary batches.
    """
    action_count = max(1, int(considered_actions))
    simulation_count = max(0, int(num_simulations))
    if action_count <= 2 or simulation_count <= 0:
        return ()

    log2_actions = int(math.ceil(math.log2(action_count)))
    considered = action_count
    scheduled = 0
    boundaries = []
    while scheduled < simulation_count:
        extra_visits = max(
            1,
            int(simulation_count / (log2_actions * considered)),
        )
        stage_end = min(
            simulation_count,
            scheduled + extra_visits * considered,
        )
        next_considered = max(2, considered // 2)
        if next_considered < considered and stage_end < simulation_count:
            boundaries.append(int(stage_end))
        scheduled = stage_end
        considered = next_considered
    return tuple(boundaries)


def _replay_source_code(learner_turn, game_opponent_mcts, opponent_label):
    if learner_turn or game_opponent_mcts is None:
        return _REPLAY_SOURCE_LEARNER
    label = str(opponent_label or "")
    if label == "best":
        return _REPLAY_SOURCE_FROZEN_BEST
    return _REPLAY_SOURCE_UNKNOWN


def _search_correction_metadata(search_metadata):
    """Return the persisted top-change flag and measured Q delta."""
    metadata = search_metadata if isinstance(search_metadata, dict) else {}
    try:
        prior_mcts_agree = float(metadata.get('prior_mcts_agree'))
        changed_top = math.isfinite(prior_mcts_agree) and prior_mcts_agree < 0.5
    except (TypeError, ValueError, OverflowError):
        changed_top = False
    try:
        q_delta = float(metadata.get('mcts_q_delta', float('nan')))
    except (TypeError, ValueError):
        q_delta = float('nan')
    return changed_top, q_delta


def _debug_scope(config, name):
    root = config.get('debug', {}) or {}
    if not isinstance(root, dict):
        return {}, {}, False
    scoped = root.get(name, {}) or {}
    if not isinstance(scoped, dict):
        scoped = {}
    return root, scoped, bool(root.get('enabled', False))


def _debug_bool(config, scope_name, key, default=False):
    root, scoped, enabled = _debug_scope(config, scope_name)
    return bool(enabled and scoped.get(key, root.get(key, default)))


def _debug_nested(config, scope_name, nested_name):
    root, scoped, enabled = _debug_scope(config, scope_name)
    nested = scoped.get(nested_name, {}) or {}
    if not isinstance(nested, dict):
        nested = {}
    return root, nested, enabled


class _InductorSMWarningFilter(logging.Filter):
    """Drop noisy Inductor warning for low-SM GPUs during self-play."""

    _needle = "Not enough SMs to use max_autotune_gemm mode"

    def filter(self, record):
        try:
            return self._needle not in record.getMessage()
        except Exception:
            return True


class _SyzygyOracle:
    def __init__(self, paths, max_pieces=6):
        self.paths = tuple(str(p) for p in paths)
        path_objects = tuple(Path(p) for p in self.paths)
        self.wdl_piece_counts = syzygy_piece_counts(path_objects, extension=".rtbw")
        self.dtz_piece_counts = syzygy_piece_counts(path_objects, extension=".rtbz")
        available_max = max(self.wdl_piece_counts, default=0)
        self.max_pieces = min(max(2, int(max_pieces)), available_max)
        self._tb = None
        self._python_chess = None

        if not self.paths or not self.wdl_piece_counts:
            return

        try:
            import chess as python_chess
            import chess.syzygy as python_syzygy
        except Exception:
            return
        self._python_chess = python_chess

        tablebase = None
        for idx, path in enumerate(self.paths):
            try:
                if idx == 0:
                    tablebase = python_syzygy.open_tablebase(path)
                else:
                    tablebase.add_directory(path)
            except Exception:
                continue
        self._tb = tablebase

    @property
    def enabled(self):
        return self._tb is not None

    def can_probe(self, board):
        # Avoid allocating a piece map on every self-play ply; native color
        # bitboards provide the same count directly.
        piece_count = chess.piece_count(board)
        has_castling_rights = any(
            chess.has_kingside_castling_rights(board, color)
            or chess.has_queenside_castling_rights(board, color)
            for color in (chess.WHITE, chess.BLACK)
        )
        return (
            self.enabled
            and not has_castling_rights
            and piece_count <= self.max_pieces
            and (piece_count <= 2 or piece_count in self.wdl_piece_counts)
        )

    @lru_cache(maxsize=65_536)
    def _probe_wdl_fen(self, fen):
        probe_board = self._python_chess.Board(fen)
        return int(self._tb.probe_wdl(probe_board))

    @lru_cache(maxsize=16_384)
    def _probe_dtz_fen(self, fen):
        probe_board = self._python_chess.Board(fen)
        return int(self._tb.probe_dtz(probe_board))

    def probe_wdl(self, board):
        if not self.can_probe(board):
            return None
        try:
            # Keep the native board throughout search and cross the python-chess
            # boundary only for a tablebase probe. Full FEN preserves the
            # side-to-move and fifty-move clock.
            return self._probe_wdl_fen(chess.board_fen(board))
        except Exception:
            return None

    def probe_dtz(self, board):
        """Return a local DTZ value when both WDL and DTZ tables cover it."""
        if not self.can_probe(board):
            return None
        if chess.piece_count(board) not in self.dtz_piece_counts:
            return None
        try:
            return self._probe_dtz_fen(chess.board_fen(board))
        except Exception:
            return None

    def probe_value(self, board):
        """Return an exact side-to-move value, or None when WDL is insufficient."""
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if abs(int(wdl)) < 2:
            # Draw, cursed win and blessed loss are all draws with the
            # fifty-move rule used by search.
            return 0.0

        halfmove_clock = int(board.halfmove_clock)
        if halfmove_clock <= 0:
            # python-chess documents WDL as exact for positions reached directly
            # after a capture or pawn move.
            return _syzygy_wdl_to_value(wdl)

        piece_count = chess.piece_count(board)
        if piece_count not in self.dtz_piece_counts:
            # WDL alone cannot tell whether the next zeroing move arrives before
            # the remaining fifty-move budget. Falling back to NN/MCTS is safer
            # than turning a theoretical win into a false terminal result.
            return None
        try:
            dtz = self._probe_dtz_fen(chess.board_fen(board))
        except Exception:
            return None
        if halfmove_clock + abs(int(dtz)) > 100:
            return 0.0
        return _syzygy_wdl_to_value(wdl)

    def result_for_board(self, board):
        value = self.probe_value(board)
        if value is None:
            return None
        if value > 0:
            return '1-0' if board.turn == chess.WHITE else '0-1'
        if value < 0:
            return '0-1' if board.turn == chess.WHITE else '1-0'
        return '1/2-1/2'


def _resolve_syzygy_paths(config):
    return tuple(
        str(path)
        for path in resolve_syzygy_paths(config)
        if path.exists() and path.is_dir()
    )


def _get_syzygy_oracle(config):
    rl_cfg = config.get('reinforcement_learning', {})
    if not bool(rl_cfg.get('syzygy_enabled', False)):
        return None
    paths = _resolve_syzygy_paths(config)
    if not paths:
        return None
    max_pieces = int(rl_cfg.get('syzygy_max_pieces', 6) or 6)
    cache_key = (paths, max_pieces)
    oracle = _SYZYGY_ORACLE_CACHE.get(cache_key)
    if oracle is None:
        oracle = _SyzygyOracle(paths, max_pieces=max_pieces)
        _SYZYGY_ORACLE_CACHE[cache_key] = oracle
    return oracle if oracle.enabled else None



def _lock_file_identity(stat_result):
    return (
        getattr(stat_result, "st_dev", None),
        getattr(stat_result, "st_ino", None),
        getattr(stat_result, "st_size", None),
        getattr(stat_result, "st_mtime_ns", None),
        getattr(stat_result, "st_ctime_ns", None),
    )


def _windows_process_creation_generation(process_handle):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_times = kernel32.GetProcessTimes
    get_times.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    get_times.restype = wintypes.BOOL
    created = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not get_times(
        process_handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        return None
    return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)


def _current_process_creation_generation():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        return _windows_process_creation_generation(get_current_process())
    except (AttributeError, OSError):
        return None


def _process_liveness(pid, expected_generation=None):
    """Return True for live, False for verified dead, or None if unknown."""
    if int(pid) <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        wait_for_single = kernel32.WaitForSingleObject
        wait_for_single.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        wait_for_single.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        handle = open_process(0x00101000, False, int(pid))  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # ERROR_INVALID_PARAMETER means the PID does not identify a process.
            # Access denied and all other errors are unknown: fail closed.
            return False if ctypes.get_last_error() == 87 else None
        try:
            wait_result = wait_for_single(handle, 0)
            if wait_result == 0:  # WAIT_OBJECT_0: process exited.
                return False
            if wait_result != 0x102:  # Anything except WAIT_TIMEOUT is unknown.
                return None
            generation = _windows_process_creation_generation(handle)
            if generation is None:
                return None
            if expected_generation is not None:
                return generation == expected_generation
            return True
        finally:
            close_handle(handle)

    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return None
        return None
    return True



def _unlink_lock_if_unchanged(lock_path, expected_identity, expected_payload=None):
    """Remove a lock only when its path still names the inspected file."""
    try:
        before = lock_path.stat()
        if _lock_file_identity(before) != expected_identity:
            return False
        payload = lock_path.read_bytes()
        after = lock_path.stat()
        if _lock_file_identity(after) != expected_identity:
            return False
        if expected_payload is not None and payload != expected_payload:
            return False
        lock_path.unlink()
        return True
    except OSError:
        return False


class _SelfPlayCompileLock:
    def __init__(self, lock_path, timeout_s=900.0, poll_s=0.2, stale_after_s=1800.0):
        self.lock_path = Path(lock_path)
        self.timeout_s = max(1.0, float(timeout_s))
        self.poll_s = max(0.05, float(poll_s))
        self.stale_after_s = max(self.timeout_s, float(stale_after_s))
        self._fd = None
        self._owned_identity = None
        self._owned_payload = None

    def __enter__(self):
        start_t = time.perf_counter()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                lock_created_at = time.time()
                generation = _current_process_creation_generation()
                generation_field = f" {generation}" if generation is not None else ""
                payload = (
                    f"{os.getpid()} {lock_created_at:.6f}{generation_field}\n"
                    .encode("ascii", errors="ignore")
                )
                written = 0
                while written < len(payload):
                    written += os.write(self._fd, payload[written:])
                fd_identity = _lock_file_identity(os.fstat(self._fd))
                self._owned_payload = self.lock_path.read_bytes()
                self._owned_identity = _lock_file_identity(self.lock_path.stat())
                if fd_identity[:2] != self._owned_identity[:2]:
                    os.close(self._fd)
                    self._fd = None
                    self._owned_identity = None
                    self._owned_payload = None
                    continue
                return self
            except FileExistsError:
                try:
                    stat = self.lock_path.stat()
                    identity = _lock_file_identity(stat)
                    age_s = max(0.0, time.time() - float(stat.st_mtime))
                    payload = self.lock_path.read_bytes()
                    if _lock_file_identity(self.lock_path.stat()) != identity:
                        continue
                except OSError:
                    identity = None
                    age_s = 0.0
                    payload = b""

                owner_pid = None
                owner_generation = None
                if identity is not None:
                    try:
                        fields = payload.decode("ascii", errors="strict").strip().split()
                        if len(fields) >= 2:
                            parsed_pid = int(fields[0])
                            float(fields[1])  # Keep malformed records on age fallback.
                            if parsed_pid > 0:
                                owner_pid = parsed_pid
                                if len(fields) >= 3:
                                    try:
                                        owner_generation = int(fields[2])
                                    except ValueError:
                                        owner_generation = None
                    except (UnicodeError, ValueError, IndexError):
                        pass

                reclaim = False
                if owner_pid is not None:
                    reclaim = _process_liveness(owner_pid, owner_generation) is False
                elif identity is not None and age_s > self.stale_after_s:
                    # Only legacy malformed/empty files use age fallback. Never
                    # unlink a fresh file that may be between create and write.
                    reclaim = True
                if reclaim and identity is not None:
                    if _unlink_lock_if_unchanged(self.lock_path, identity, payload):
                        continue
                if (time.perf_counter() - start_t) >= self.timeout_s:
                    raise TimeoutError(f"Timed out waiting for compile lock: {self.lock_path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
        finally:
            self._fd = None
            try:
                if self._owned_identity is not None:
                    _unlink_lock_if_unchanged(
                        self.lock_path,
                        self._owned_identity,
                        self._owned_payload,
                    )
            finally:
                self._owned_identity = None
                self._owned_payload = None

def _resolve_selfplay_compile_enabled(config, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return False
    rl_cfg = config.get('reinforcement_learning', {})
    if 'self_play_use_compile' in rl_cfg:
        return bool(rl_cfg.get('self_play_use_compile', False))
    return bool(config.get('hardware', {}).get('use_compile', False))


def _resolve_selfplay_compile_use_lock(config):
    rl_cfg = config.get('reinforcement_learning', {})
    if 'self_play_compile_use_lock' in rl_cfg:
        return bool(rl_cfg.get('self_play_compile_use_lock', False))
    # A persistent cache is shared by central inference processes across runs.
    # Serialize a cache miss so only one process builds a given artifact.
    return True


def _selfplay_compile_cache_paths(rank, device):
    """Return a persistent, environment-safe Inductor/Triton cache namespace."""
    # Central self-play, eval and Elo use the same graph cache. Inductor's cache
    # key already includes the graph and compile options, so separating these
    # short-lived processes by PID only prevents valid reuse.
    rank_value = int(rank)
    role = "central" if rank_value >= 700000 else f"worker_{rank_value}"
    return torch_compile_cache_paths(torch, device, role)


def _configure_selfplay_compile_cache(rank, device):
    root, worker_dir = _selfplay_compile_cache_paths(rank, device)
    if root is None or worker_dir is None:
        return None
    role = worker_dir.name
    configure_torch_compile_cache(torch, device, role)
    return root


def _reset_selfplay_compile_cache(rank, device):
    """Discard only the selected persistent Inductor/Triton role cache.

    Local workers get separate directories; central self-play/eval processes
    intentionally share one directory so compatible graphs survive restarts.
    """
    if device.type != 'cuda':
        return False
    root, worker_dir = _selfplay_compile_cache_paths(rank, device)
    if root is None or worker_dir is None:
        return False
    try:
        # Keep deletion constrained to one known role below the versioned root.
        if worker_dir.resolve().parent != root.resolve():
            return False
        shutil.rmtree(worker_dir, ignore_errors=True)
        _configure_selfplay_compile_cache(rank, device)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _configure_inductor_for_selfplay(config):
    rl_cfg = config.get('reinforcement_learning', {})

    # Allow explicit control of Inductor compile worker threads.
    compile_threads_raw = rl_cfg.get('self_play_compile_threads', None)
    if compile_threads_raw is not None:
        try:
            compile_threads = int(compile_threads_raw)
            if compile_threads > 0:
                os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
        except Exception:
            pass

    # Silence the low-SM max_autotune warning in self-play logs by default.
    suppress_sm_warning = bool(
        rl_cfg.get('self_play_suppress_inductor_sm_warning', True)
    )
    if not suppress_sm_warning:
        return

    try:
        from torch._inductor import config as inductor_config
        inductor_config.max_autotune_gemm = False
    except Exception:
        pass

    try:
        logger = logging.getLogger("torch._inductor.utils")
        has_filter = any(
            isinstance(existing, _InductorSMWarningFilter)
            for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(_InductorSMWarningFilter())
    except Exception:
        pass


def _maybe_compile_selfplay_model(
    model,
    config,
    device,
    rank,
    model_label="learner",
    warmup_batch_size=1,
):
    if not _resolve_selfplay_compile_enabled(config, device):
        return model
    if not hasattr(torch, "compile"):
        return model

    rl_cfg = config.get('reinforcement_learning', {})
    compile_root = _configure_selfplay_compile_cache(rank, device)
    lock_path = (
        compile_root / _SELFPLAY_COMPILE_LOCKFILE
        if compile_root is not None
        else Path(tempfile.gettempdir()) / _SELFPLAY_COMPILE_LOCKFILE
    )
    timeout_s = float(rl_cfg.get('self_play_compile_lock_timeout_s', 900.0) or 900.0)
    use_compile_lock = _resolve_selfplay_compile_use_lock(config)

    model_cfg = config.get('model', {}) or {}
    input_spec = (
        (getattr(model, 'model_spec', {}) or {}).get('input_spec', {}) or {}
    )
    history_positions = int(
        input_spec.get('history_positions', model_cfg.get('history_positions', 0))
        or 0
    )
    encoder = get_input_encoder(
        str(
            input_spec.get(
                'encoder_id',
                getattr(
                    model,
                    'input_encoder_id',
                    model_cfg.get('input_encoder_id', 'planes_history_v1'),
                ),
            )
        )
    )
    expected_input_planes = int(
        input_spec.get(
            'input_planes',
            encoder.input_planes(history_positions),
        )
    )
    use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
    use_bfloat16 = bool(config.get('hardware', {}).get('use_bfloat16', False))
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    if bool(rl_cfg.get('self_play_compile_reset_dynamo', False)):
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    def _compile_and_warmup(target_model):
        compiled_model = torch.compile(
            target_model,
            mode="reduce-overhead",
            dynamic=True,
        )
        dummy_input = torch.zeros(
            max(1, int(warmup_batch_size)),
            expected_input_planes,
            8,
            8,
            device=device,
            dtype=torch.float32,
        )
        if device.type == 'cuda':
            dummy_input = dummy_input.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            with torch.autocast(
                device_type='cuda',
                enabled=use_amp,
                dtype=amp_dtype,
            ):
                compiled_model(dummy_input, apply_log_softmax=False)
        return compiled_model

    def _compile_once():
        if use_compile_lock:
            with _SelfPlayCompileLock(lock_path, timeout_s=timeout_s):
                return _compile_and_warmup(model)
        return _compile_and_warmup(model)

    try:
        return _compile_once()
    except Exception as first_exc:
        # A stale or partially written Triton artifact is recoverable. Rebuild
        # this process's cache once before abandoning compile for the run.
        if _reset_selfplay_compile_cache(rank, device):
            try:
                return _compile_once()
            except Exception as retry_exc:
                exc = retry_exc
        else:
            exc = first_exc
        print(
            f"WARNING: Self-play worker {rank}: torch.compile cache rebuild failed for {model_label}; "
            f"using eager inference "
            f"({type(exc).__name__}: {exc})"
        )
        return model


def _copy_board_fast(board):
    """Native C-level bulletchess board copy."""
    return chess.copy_board(board)


def _board_position_key(board):
    """
    Repetition identity for FIDE position matching.

    The halfmove and fullmove counters do not define a position for repetition
    purposes. An en-passant target does define one only when the side to move
    has a legal en-passant capture; an uncapturable or pinned capture must not
    distinguish otherwise identical positions.
    """
    piece_masks = tuple(
        int(board[color, piece_type])
        for color, piece_type in _INPUT_PIECE_SPECS
    )
    castling_rights = (
        int(chess.has_kingside_castling_rights(board, chess.WHITE))
        | (int(chess.has_queenside_castling_rights(board, chess.WHITE)) << 1)
        | (int(chess.has_kingside_castling_rights(board, chess.BLACK)) << 2)
        | (int(chess.has_queenside_castling_rights(board, chess.BLACK)) << 3)
    )
    ep_square = board.en_passant_square
    if ep_square is not None and any(
        chess.is_en_passant(board, move) for move in chess.legal_moves(board)
    ):
        ep_square = chess.square_index(ep_square) + 1
    else:
        ep_square = 0
    # Twelve 64-bit piece masks plus side, rights and EP sentinel/index. This
    # exact fixed-width encoding avoids retaining a nested tuple and 12 PyLongs
    # on every node whose repetition key is cached.
    return struct.pack(
        "<12QBBB",
        *piece_masks,
        int(board.turn == chess.BLACK),
        castling_rights,
        ep_square,
    )


def _board_tree_reuse_key(board):
    """Keep root reuse tied to the backend's full board hash, including clocks."""
    return chess.position_key(board)


def _board_tree_reuse_identity(board):
    """Return exact board/rule identity for reuse validation, not a hash alone."""
    return (
        _board_position_key(board),
        int(board.halfmove_clock),
        int(board.fullmove_number),
    )


def _canonical_context_value(value):
    """Freeze plain model/input metadata into deterministic comparable tuples."""
    if value is None or isinstance(value, (str, int, bool, bytes)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        try:
            return tuple(sorted(
                (
                    _canonical_context_value(key),
                    _canonical_context_value(item),
                )
                for key, item in value.items()
            ))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_context_value(item) for item in value)
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _canonical_position_counts(position_counts):
    """Canonicalize exact repetition provenance for a single live root."""
    if not isinstance(position_counts, dict):
        return None
    if len(position_counts) > 0xFFFFFFFF:
        return None
    items = []
    for key, count in position_counts.items():
        # Position keys are fixed-width exact encodings from
        # ``_board_position_key``. Do not call ``bytes(int)`` here: Python
        # interprets an integer as a requested allocation length, and native
        # board hashes may be large enough to raise MemoryError.
        if (
            not isinstance(key, (bytes, bytearray, memoryview))
            or isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
        ):
            return None
        try:
            key_bytes = bytes(key)
            count_value = int(count)
        except (TypeError, ValueError, OverflowError):
            return None
        if not key_bytes or count_value <= 0 or count_value > 0xFFFFFFFF:
            return None
        items.append((key_bytes, count_value))
    items.sort(key=lambda item: item[0])
    return tuple(items)


def _history_tensor_context_digest(tensor):
    """Digest an exact encoded history frame including shape and dtype."""
    array = np.ascontiguousarray(np.asarray(tensor))
    digest = hashlib.blake2b(digest_size=32)
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(struct.pack('<I', array.ndim))
    for dimension in array.shape:
        digest.update(struct.pack('<Q', int(dimension)))
    digest.update(array.tobytes(order='C'))
    return digest.digest()


@dataclass(frozen=True)
class _RootReuseContext:
    """Immutable provenance for one root's NN inputs and chess rule context.

    ``history_frames`` stores both POVs for the exact encoded frames used by
    the root.  It is one bounded root-level tuple, not a per-node history map.
    ``current_frame`` lets a selected child derive its expected next-root
    history before ``detach_as_root`` drops the parent link.  A derived child
    leaves that field unset until its observed input context is validated.
    """

    board_identity: tuple
    history_frames: tuple
    current_frame: tuple | None
    history_positions: int
    position_counts: tuple
    repetition_known: bool
    input_contract: tuple
    evaluation_generation: tuple

    def matches(self, observed):
        if not isinstance(observed, _RootReuseContext):
            return False
        if (
            self.board_identity != observed.board_identity
            or self.history_frames != observed.history_frames
            or self.history_positions != observed.history_positions
            or self.position_counts != observed.position_counts
            or self.repetition_known != observed.repetition_known
            or self.input_contract != observed.input_contract
            or self.evaluation_generation != observed.evaluation_generation
        ):
            return False
        # A just-promoted child has all of its expected input provenance, but
        # its own frame is filled only after the caller-provided context agrees.
        return self.current_frame is None or self.current_frame == observed.current_frame

    def with_observed_frame(self, observed):
        if not self.matches(observed):
            return None
        return observed

    def advance_to_child(self, child_board):
        """Derive the exact expected context after one actual tree edge."""
        if (
            not self.repetition_known
            or not self.position_counts
            or (self.history_positions > 0 and self.current_frame is None)
        ):
            return None

        counts = dict(self.position_counts)
        child_position_key = _board_position_key(child_board)
        counts[child_position_key] = int(counts.get(child_position_key, 0)) + 1
        next_history = self.history_frames
        if self.history_positions:
            next_history = (next_history + (self.current_frame,))[-self.history_positions:]
            padding = self.history_positions - len(next_history)
            if padding:
                # The leading entries are already represented by the exact
                # padding-frame digest stored in the prior root anchor.
                next_history = self.history_frames[:padding] + next_history

        return replace(
            self,
            board_identity=_board_tree_reuse_identity(child_board),
            history_frames=next_history,
            current_frame=None,
            position_counts=tuple(sorted(counts.items(), key=lambda item: item[0])),
        )


def _position_counts_from_board(board):
    """Rebuild repetition counts when a caller supplies a board with a stack."""
    moves = list(chess.move_history(board))
    if not moves:
        return {_board_position_key(board): 1}
    replay = chess.root_board(board)
    counts = {_board_position_key(replay): 1}
    for move in moves:
        chess.apply_move(replay, move)
        key = _board_position_key(replay)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _repetition_count_before_node(node, root_position_counts, provenance_known=True):
    """Return exact prior occurrences for a node, or None when history is partial."""
    if not provenance_known:
        return None
    key = node.get_position_key()
    root = node
    while root.parent is not None:
        root = root.parent
    occurrences = int((root_position_counts or {}).get(key, 0))
    if node is root:
        return max(0, min(2, occurrences - 1))

    # root_position_counts already includes the real-game root. Count only
    # simulated descendants strictly between root and the requested node.
    cursor = node.parent
    while cursor is not None and cursor is not root:
        if cursor.get_position_key() == key:
            occurrences += 1
        cursor = cursor.parent
    return max(0, min(2, occurrences))


def _record_position_count(position_counts, board):
    key = _board_position_key(board)
    position_counts[key] = int(position_counts.get(key, 0)) + 1


def _is_threefold_repetition_on_path(node, root_position_counts):
    """Check whether this simulated descendant is the third path occurrence."""
    if node is None or node.parent is None:
        # Real-game draw claiming owns the root. This helper only detects a
        # repetition created by moves selected inside the current search tree.
        return False
    target_key = node.get_position_key()
    occurrences = int((root_position_counts or {}).get(target_key, 0))
    cursor = node
    while cursor.parent is not None:
        if cursor.get_position_key() == target_key:
            occurrences += 1
            if occurrences >= 3:
                return True
        cursor = cursor.parent
    return False


def _is_search_path_draw_candidate(node, board, root_position_counts):
    """Detect history-aware draws before the relatively costly legal-move scan."""
    if int(board.halfmove_clock) >= 150:
        return True
    if chess.piece_count(board) <= 4 and chess.is_insufficient_material(board):
        return True
    if node.parent is None:
        return False
    if int(board.halfmove_clock) >= 100:
        return True
    # Three occurrences require at least two reversible four-ply cycles since
    # the last pawn move/capture. Most tactical leaves can skip the path walk.
    return bool(
        int(board.halfmove_clock) >= 8
        and _is_threefold_repetition_on_path(node, root_position_counts)
    )


def _softmax_padded_legal_logits(legal_logits_batch, legal_counts):
    """Vectorized legal-only softmax for a padded batch of policy logits."""
    logits = np.asarray(legal_logits_batch, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D legal logits, got shape {logits.shape}.")
    counts = np.asarray(legal_counts, dtype=np.int32)
    if counts.ndim != 1 or int(counts.shape[0]) != int(logits.shape[0]):
        raise ValueError("Legal-count vector does not match the logits batch.")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        return np.zeros_like(logits, dtype=np.float32)

    valid = np.arange(logits.shape[1], dtype=np.int32)[None, :] < counts[:, None]
    masked_logits = np.where(valid, logits, -np.inf)
    row_max = np.max(masked_logits, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    probabilities = np.zeros_like(logits, dtype=np.float32)
    np.exp(masked_logits - row_max, out=probabilities, where=valid)
    normalizers = probabilities.sum(axis=1, keepdims=True)
    np.divide(
        probabilities,
        normalizers + 1.0e-8,
        out=probabilities,
        where=normalizers > 0.0,
    )
    return probabilities


def _syzygy_wdl_to_value(wdl):
    """Map Syzygy WDL to a hard result under the fifty-move rule."""
    wdl = int(wdl)
    if wdl == 2:
        return 1.0
    if wdl == -2:
        return -1.0
    # Cursed wins (+1) and blessed losses (-1) are draws under the rule.
    return 0.0


_SELFPLAY_OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"),
    ("e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5"),
    ("d2d4", "d7d5", "g1f3", "g8f6", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"),
    ("g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6"),
    ("g1f3", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"),
    ("e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"),
    ("d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6"),
    ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"),
)

# Keep eval fixed openings relatively balanced, but let self-play sample from
# a broader, sharper pool so RL escapes quiet draw-heavy shells more often.
_SELFPLAY_SHARP_OPENING_LINES = (
    ("e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5"),
    ("e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "d7d5"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f1c4"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "c2c3"),
    ("e2e4", "e7e5", "d2d4", "e5d4", "c2c3", "d4c3", "f1c4"),
    ("e2e4", "c7c5", "d2d4", "c5d4", "c2c3", "d4c3", "b1c3"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"),
    ("e2e4", "c7c5", "b1c3", "b8c6", "f2f4", "g7g6", "g1f3", "f8g7"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4"),
    ("e2e4", "d7d5", "e4d5", "g8f6", "c2c4", "e7e6", "b1c3", "e6d5"),
    ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6"),
    ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f4"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4"),
    ("d2d4", "f7f5", "e2e4", "f5e4", "b1c3", "g8f6", "c1g5"),
    ("d2d4", "g8f6", "c2c4", "e7e5", "d4e5", "f6g4"),
)
_SELFPLAY_SELFPLAY_OPENING_LINES = _SELFPLAY_OPENING_LINES + _SELFPLAY_SHARP_OPENING_LINES


def _stratify_eval_opening_lines(lines):
    """Interleave opening families and line depths across every suite prefix."""
    groups = {}
    family_counts = {}
    depth_counts = {}
    for line in lines:
        family = tuple(line[:2])
        depth = len(line)
        groups.setdefault((family, depth), []).append(line)
        family_counts[family] = family_counts.get(family, 0) + 1
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    rng = np.random.default_rng(490050)
    for group in groups.values():
        rng.shuffle(group)
    used_families = dict.fromkeys(family_counts, 0)
    used_depths = dict.fromkeys(depth_counts, 0)
    ordered = []
    for index in range(len(lines)):
        candidates = []
        for (family, depth), group in groups.items():
            if not group:
                continue
            family_deficit = (
                (index + 1) * family_counts[family] / len(lines)
                - used_families[family]
            )
            depth_deficit = (
                (index + 1) * depth_counts[depth] / len(lines)
                - used_depths[depth]
            )
            candidates.append((family_deficit + depth_deficit, family, depth))
        best_score = max(candidate[0] for candidate in candidates)
        best = [
            (family, depth)
            for score, family, depth in candidates
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
        ]
        family, depth = best[int(rng.integers(len(best)))]
        ordered.append(groups[(family, depth)].pop())
        used_families[family] += 1
        used_depths[depth] += 1
    return tuple(ordered)


def _build_eval_opening_lines():
    """Deduplicate the original 96 positions, then stratify their stage order."""
    lines = _SELFPLAY_OPENING_LINES + _SELFPLAY_SHARP_OPENING_LINES
    unique = []
    seen_positions = set()
    for depth in range(4, 11):
        for line in lines:
            if len(line) < depth:
                continue
            prefix = tuple(line[:depth])
            board = chess.new_board()
            for uci in prefix:
                chess.apply_move(board, chess.move_from_uci(uci))
            position_key = _board_position_key(board)
            if position_key in seen_positions:
                continue
            seen_positions.add(position_key)
            unique.append(prefix)
            if len(unique) >= 96:
                return _stratify_eval_opening_lines(unique)
    return _stratify_eval_opening_lines(unique)


_EVAL_OPENING_SUITE_ID = "curated-family-depth-stratified-v3"
_EVAL_OPENING_LINES = _build_eval_opening_lines()


class MCTSEdgeStats:
    """Compact array-backed storage for all child-edge statistics of one node."""

    __slots__ = (
        "moves",
        "nodes",
        "base_priors",
        "log_base_priors",
        "visit_counts",
        "total_counts",
        "value_sums",
        "virtual_losses",
        "total_count_sum",
        "stats_version",
        "completed_q_cache_version",
        "completed_q_cache",
        "improved_policy_cache_version",
        "improved_policy_cache",
        "selection_score_scratch",
        "q_score_scratch",
        "fresh_count_scratch",
        "eligibility_scratch",
        "_move_to_child",
    )

    def __init__(self):
        self.moves = ()
        self.nodes = []
        self.base_priors = np.empty(0, dtype=np.float32)
        self.log_base_priors = np.empty(0, dtype=np.float64)
        self.visit_counts = np.empty(0, dtype=np.int32)
        self.total_counts = np.empty(0, dtype=np.float32)
        self.value_sums = np.empty(0, dtype=np.float32)
        self.virtual_losses = np.empty(0, dtype=np.int16)
        self.total_count_sum = 0.0
        self.stats_version = 0
        self.completed_q_cache_version = -1
        self.completed_q_cache = np.empty(0, dtype=np.float64)
        self.improved_policy_cache_version = -1
        self.improved_policy_cache = np.empty(0, dtype=np.float32)
        self.selection_score_scratch = np.empty(0, dtype=np.float64)
        self.q_score_scratch = np.empty(0, dtype=np.float32)
        self.fresh_count_scratch = np.empty(0, dtype=np.float64)
        self.eligibility_scratch = np.empty(0, dtype=np.bool_)
        self._move_to_child = None

    def reset(self, legal_moves, legal_priors):
        legal_moves = tuple(legal_moves)
        legal_priors = np.asarray(legal_priors, dtype=np.float32)
        child_count = len(legal_moves)

        self.moves = legal_moves
        self.nodes = [None] * child_count
        self.base_priors = legal_priors.copy()
        # Persistent C++ traversal only needs the compact edge statistics.
        # Gumbel scratch/cache arrays are used by root bookkeeping and by the
        # Python correctness fallback, so allocating them eagerly for every
        # expanded interior node wastes memory without helping the native path.
        self.log_base_priors = np.empty(0, dtype=np.float64)
        self.visit_counts = np.zeros(child_count, dtype=np.int32)
        self.total_counts = np.zeros(child_count, dtype=np.float32)
        self.value_sums = np.zeros(child_count, dtype=np.float32)
        self.virtual_losses = np.zeros(child_count, dtype=np.int16)
        self.total_count_sum = 0.0
        self.stats_version = 0
        self.completed_q_cache_version = -1
        self.completed_q_cache = np.empty(0, dtype=np.float64)
        self.improved_policy_cache_version = -1
        self.improved_policy_cache = np.empty(0, dtype=np.float32)
        self.selection_score_scratch = np.empty(0, dtype=np.float64)
        self.q_score_scratch = np.empty(0, dtype=np.float32)
        self.fresh_count_scratch = np.empty(0, dtype=np.float64)
        self.eligibility_scratch = np.empty(0, dtype=np.bool_)
        self._move_to_child = None

    def ensure_gumbel_workspace(self):
        """Materialize optional Gumbel arrays only for nodes that use them."""
        child_count = int(self.base_priors.size)
        if self.log_base_priors.size == child_count:
            return
        self.log_base_priors = np.log(
            np.maximum(
                self.base_priors.astype(np.float64),
                np.finfo(np.float64).tiny,
            )
        )
        self.completed_q_cache = np.empty(child_count, dtype=np.float64)
        self.improved_policy_cache = np.empty(child_count, dtype=np.float32)
        self.selection_score_scratch = np.empty(child_count, dtype=np.float64)
        self.q_score_scratch = np.empty(child_count, dtype=np.float32)
        self.fresh_count_scratch = np.empty(child_count, dtype=np.float64)
        self.eligibility_scratch = np.empty(child_count, dtype=np.bool_)
        self.completed_q_cache_version = -1
        self.improved_policy_cache_version = -1

    def mark_search_stats_changed(self):
        """Invalidate completed-Q policy only after real visit/value updates."""
        self.stats_version += 1

    def iter_nodes(self):
        for idx, move in enumerate(self.moves):
            child = self.nodes[idx]
            if child is not None:
                yield move, child

    def _get_move_index(self, move):
        move_to_child = self._move_to_child
        if move_to_child is None:
            move_to_child = {child_move: idx for idx, child_move in enumerate(self.moves)}
            self._move_to_child = move_to_child
        return move_to_child.get(move)

    def get_or_create_child(self, parent, idx):
        idx = int(idx)
        child = self.nodes[idx]
        if child is not None:
            return child

        child = MCTSNode(
            board=None,
            parent=parent,
            move=self.moves[idx],
            copy_board=False,
        )
        child.parent_edge_index = idx
        self.nodes[idx] = child
        return child

    def get_child_for_move(self, parent, move):
        idx = self._get_move_index(move)
        if idx is None:
            return None
        return self.get_or_create_child(parent, idx)

    def visit_dict(self):
        return {
            move: int(visits)
            for move, visits in zip(self.moves, self.visit_counts)
        }

    def __len__(self):
        return len(self.moves)


class MCTSNode:
    """Node in the MCTS tree."""

    __slots__ = (
        "_board",
        "_parent_ref",
        "move",
        "edges",
        "parent_edge_index",
        "_root_visit_count",
        "_root_value_sum",
        "expanded",
        "_root_virtual_loss",
        "_position_key_cache",
        "_tree_reuse_key_cache",
        "_board_tensor",
        "_legal_moves",
        "_legal_indices",
        "terminal_value",
        "raw_value",
        "_reuse_context_anchor",
        "__weakref__",
    )

    def __init__(self, board=None, parent=None, move=None, copy_board=True):
        if board is not None:
            self._board = _copy_board_fast(board) if copy_board else board
        else:
            self._board = None
        self.parent = parent
        self.move = move

        self.edges = MCTSEdgeStats()
        self.parent_edge_index = -1
        self._root_visit_count = 0
        self._root_value_sum = 0.0
        self.expanded = False
        self._root_virtual_loss = 0

        self._position_key_cache = None
        self._tree_reuse_key_cache = None
        self._board_tensor = None
        self._legal_moves = None
        self._legal_indices = None
        self.terminal_value = None
        self.raw_value = None
        self._reuse_context_anchor = None

    @property
    def parent(self):
        """Non-owning parent link; edge storage exclusively owns descendants."""
        return None if self._parent_ref is None else self._parent_ref()

    @parent.setter
    def parent(self, value):
        self._parent_ref = None if value is None else weakref.ref(value)

    @property
    def board(self):
        if self._board is None:
            self._board = _copy_board_fast(self.parent.board)
            chess.apply_move(self._board, self.move)
        return self._board

    def is_leaf(self):
        return not self.expanded

    def _has_parent_edge(self):
        return self.parent is not None and self.parent_edge_index >= 0

    def detach_as_root(self):
        """Promote this child to root while preserving edge-owned search stats."""
        if self._board is None and self.parent is not None:
            _ = self.board
        if self._has_parent_edge():
            parent = self.parent
            parent_context = getattr(parent, "_reuse_context_anchor", None)
            self._reuse_context_anchor = (
                parent_context.advance_to_child(self.board)
                if isinstance(parent_context, _RootReuseContext)
                else None
            )
            idx = int(self.parent_edge_index)
            edges = parent.edges
            self._root_visit_count = int(edges.visit_counts[idx])
            self._root_value_sum = float(edges.value_sums[idx])
            self._root_virtual_loss = 0
            # Drop the old parent -> selected-child link before releasing the
            # parent. This lets CPython reclaim siblings/ancestors immediately
            # while the promoted subtree and all its descendants stay alive.
            edges.nodes[idx] = None
        self.parent = None
        self.parent_edge_index = -1
        return self

    @property
    def visit_count(self):
        if self._has_parent_edge():
            return int(self.parent.edges.visit_counts[int(self.parent_edge_index)])
        return int(self._root_visit_count)

    @visit_count.setter
    def visit_count(self, value):
        value = int(value)
        if self._has_parent_edge():
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
            previous_total = float(edges.total_counts[idx])
            edges.visit_counts[idx] = value
            edges.total_counts[idx] = float(value + int(edges.virtual_losses[idx]))
            edges.total_count_sum += float(edges.total_counts[idx]) - previous_total
            edges.mark_search_stats_changed()
        else:
            self._root_visit_count = value

    @property
    def value_sum(self):
        if self._has_parent_edge():
            return float(self.parent.edges.value_sums[int(self.parent_edge_index)])
        return float(self._root_value_sum)

    @value_sum.setter
    def value_sum(self, value):
        value = float(value)
        if self._has_parent_edge():
            edges = self.parent.edges
            edges.value_sums[int(self.parent_edge_index)] = value
            edges.mark_search_stats_changed()
        else:
            self._root_value_sum = value

    @property
    def virtual_loss(self):
        if self._has_parent_edge():
            return int(self.parent.edges.virtual_losses[int(self.parent_edge_index)])
        return int(self._root_virtual_loss)

    @virtual_loss.setter
    def virtual_loss(self, value):
        value = int(value)
        if self._has_parent_edge():
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
            previous_total = float(edges.total_counts[idx])
            edges.virtual_losses[idx] = value
            edges.total_counts[idx] = float(int(edges.visit_counts[idx]) + value)
            edges.total_count_sum += float(edges.total_counts[idx]) - previous_total
        else:
            self._root_virtual_loss = value

    def add_virtual_loss(self, n=1):
        parent = self.parent
        edge_index = self.parent_edge_index
        if parent is not None and edge_index >= 0:
            edges = parent.edges
            visit_count = int(edges.visit_counts[edge_index])
            virtual_loss = int(edges.virtual_losses[edge_index])
            virtual_loss += int(n)
            edges.virtual_losses[edge_index] = virtual_loss
            edges.total_counts[edge_index] = float(visit_count + virtual_loss)
            edges.total_count_sum += int(n)
            return

        self._root_virtual_loss += int(n)

    def remove_virtual_loss(self, n=1):
        parent = self.parent
        edge_index = self.parent_edge_index
        if parent is not None and edge_index >= 0:
            edges = parent.edges
            visit_count = int(edges.visit_counts[edge_index])
            virtual_loss = int(edges.virtual_losses[edge_index])
            virtual_loss -= int(n)
            edges.virtual_losses[edge_index] = virtual_loss
            edges.total_counts[edge_index] = float(visit_count + virtual_loss)
            edges.total_count_sum -= int(n)
            return

        self._root_virtual_loss -= int(n)

    def expand_children(self, legal_moves, legal_priors):
        self.edges.reset(legal_moves, legal_priors)
        self.expanded = True
        self._board_tensor = None
        self._legal_moves = None
        self._legal_indices = None
        self._position_key_cache = None
        self._tree_reuse_key_cache = None

    def iter_child_nodes(self):
        return self.edges.iter_nodes()

    def get_child_for_move(self, move):
        return self.edges.get_child_for_move(self, move)

    def child_visit_dict(self):
        return self.edges.visit_dict()

    def get_position_key(self):
        if self._position_key_cache is None:
            self._position_key_cache = _board_position_key(self.board)
        return self._position_key_cache

    def get_tree_reuse_key(self):
        if self._tree_reuse_key_cache is None:
            self._tree_reuse_key_cache = _board_tree_reuse_key(self.board)
        return self._tree_reuse_key_cache

    def get_legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = tuple(chess.legal_moves(self.board))
        return self._legal_moves

    def get_legal_indices(self):
        if self._legal_indices is None:
            legal_moves = self.get_legal_moves()
            is_black_turn = self.board.turn == chess.BLACK
            legal_indices = np.empty(len(legal_moves), dtype=np.int32)
            move_index_cache = _MOVE_POLICY_INDEX_CACHE[int(is_black_turn)]
            move_to_idx = _move_to_index_cached
            for idx, move in enumerate(legal_moves):
                policy_index = move_index_cache.get(move)
                if policy_index is None:
                    policy_index = move_to_idx(
                        chess.move_origin_index(move),
                        chess.move_destination_index(move),
                        move.promotion or 0,
                        is_black_turn,
                    )
                    move_index_cache[move] = policy_index
                legal_indices[idx] = policy_index
            self._legal_indices = legal_indices
        return self._legal_indices

    def get_legal_moves_and_indices(self):
        return self.get_legal_moves(), self.get_legal_indices()


def _build_sparse_policy_target_from_visits(visit_counts, board):
    """
    Convert MCTS visit counts into sparse policy targets.

    Returns:
        (indices, probs)
        indices: int16 tensor of action indices
        probs: float32 tensor of normalized visit probabilities
    """
    if not visit_counts:
        return (
            torch.empty(0, dtype=torch.int16),
            torch.empty(0, dtype=torch.float32),
        )

    moves = list(visit_counts.keys())
    indices = np.empty(len(moves), dtype=np.int16)
    visits = np.empty(len(moves), dtype=np.float32)

    for i, move in enumerate(moves):
        indices[i] = move_to_index(move, board)
        visits[i] = float(visit_counts[move])

    total_visits = float(visits.sum())
    if total_visits > 0:
        probs = visits / total_visits
    else:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)

    return (
        torch.from_numpy(indices),
        torch.from_numpy(probs.astype(np.float32, copy=False)),
    )


def _optional_int32_search_count(value):
    if value is None:
        return -1
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return -1
    return count if 0 <= count <= (1 << 31) - 1 else -1


def _pack_positions_for_transfer(positions, max_policy_targets=None):
    """
    Pack self-play positions into batched tensors for queue transport.
    """
    if not positions:
        return None

    batch_size = len(positions)
    boards = torch.stack([pos[0] for pos in positions]).contiguous()
    values = torch.stack([pos[3] for pos in positions]).contiguous()
    max_len = max(int(pos[1].numel()) for pos in positions)
    if max_policy_targets is not None:
        max_len = min(max_len, max(1, int(max_policy_targets)))

    policy_indices = torch.full((batch_size, max_len), -1, dtype=torch.int16)
    policy_values = torch.zeros((batch_size, max_len), dtype=torch.float32)
    policy_lengths = torch.zeros((batch_size,), dtype=torch.int16)
    max_legal_len = min(
        MAX_LEGAL_MOVES,
        max(int((pos[9] if len(pos) > 9 and pos[9] is not None else pos[1]).numel()) for pos in positions),
    )
    legal_indices = torch.full((batch_size, max_legal_len), -1, dtype=torch.int16)
    legal_lengths = torch.zeros((batch_size,), dtype=torch.int16)
    importance_scores = torch.zeros((batch_size,), dtype=torch.float32)
    policy_weights = torch.ones((batch_size,), dtype=torch.float32)
    value_weights = torch.ones((batch_size,), dtype=torch.float32)
    moves_left = torch.zeros((batch_size, 1), dtype=torch.float32)
    source_codes = torch.zeros((batch_size,), dtype=torch.int8)
    root_q_targets = torch.full((batch_size, 1), float('nan'), dtype=torch.float32)
    search_changed_top = torch.zeros((batch_size,), dtype=torch.bool)
    search_q_deltas = torch.full((batch_size,), float('nan'), dtype=torch.float32)
    best_q_targets = torch.full((batch_size, 1), float('nan'), dtype=torch.float32)
    played_q_targets = torch.full((batch_size, 1), float('nan'), dtype=torch.float32)
    orig_q_targets = torch.full((batch_size, 1), float('nan'), dtype=torch.float32)
    policy_kld_targets = torch.full((batch_size,), float('nan'), dtype=torch.float32)
    search_visits = torch.zeros((batch_size,), dtype=torch.int32)
    game_ids = torch.full((batch_size,), -1, dtype=torch.int64)
    game_ply_indices = torch.full((batch_size,), -1, dtype=torch.int16)
    regret_targets = torch.full((batch_size,), float('nan'), dtype=torch.float32)
    archive_ids = torch.full((batch_size,), -1, dtype=torch.int64)
    search_budget_assigned = torch.full((batch_size,), -1, dtype=torch.int32)
    search_simulations_effective = torch.full((batch_size,), -1, dtype=torch.int32)
    search_simulations_fresh = torch.full((batch_size,), -1, dtype=torch.int32)
    search_visits_inherited_credit = torch.full((batch_size,), -1, dtype=torch.int32)
    fens = []
    history_fens = []

    for row_idx, pos in enumerate(positions):
        _, indices, probs, _ = pos[:4]
        count = int(indices.numel())
        if max_policy_targets is not None:
            count = min(count, max(1, int(max_policy_targets)))
        importance_scores[row_idx] = float(pos[4]) if len(pos) > 4 else 0.0
        policy_weights[row_idx] = float(pos[5]) if len(pos) > 5 else 1.0
        value_weights[row_idx] = float(pos[6]) if len(pos) > 6 else 1.0
        source_codes[row_idx] = int(pos[7]) if len(pos) > 7 else _REPLAY_SOURCE_UNKNOWN
        moves_left[row_idx, 0] = float(pos[8]) if len(pos) > 8 else 0.0
        legal_payload = pos[9] if len(pos) > 9 and pos[9] is not None else indices
        legal_count = min(int(legal_payload.numel()), max_legal_len)
        if legal_count > 0:
            legal_indices[row_idx, :legal_count] = legal_payload[:legal_count].to(dtype=torch.int16)
        legal_lengths[row_idx] = legal_count
        fens.append(str(pos[10]) if len(pos) > 10 and pos[10] else None)
        if len(pos) > 11 and pos[11] is not None:
            root_q_targets[row_idx, 0] = float(pos[11])
        history_fens.append(list(pos[12] or ()) if len(pos) > 12 else [])
        search_changed_top[row_idx] = bool(pos[13]) if len(pos) > 13 else False
        if len(pos) > 14 and pos[14] is not None:
            search_q_deltas[row_idx] = float(pos[14])
        if len(pos) > 15 and pos[15] is not None:
            best_q_targets[row_idx, 0] = float(pos[15])
        if len(pos) > 16 and pos[16] is not None:
            played_q_targets[row_idx, 0] = float(pos[16])
        if len(pos) > 17 and pos[17] is not None:
            orig_q_targets[row_idx, 0] = float(pos[17])
        if len(pos) > 18 and pos[18] is not None:
            policy_kld_targets[row_idx] = float(pos[18])
        if len(pos) > 19 and pos[19] is not None:
            search_visits[row_idx] = int(pos[19])
        if len(pos) > 20 and pos[20] is not None:
            game_ids[row_idx] = int(pos[20])
        if len(pos) > 21 and pos[21] is not None:
            game_ply_indices[row_idx] = int(pos[21])
        if len(pos) > 22 and pos[22] is not None:
            regret_targets[row_idx] = float(pos[22])
        if len(pos) > 23 and pos[23] is not None:
            archive_ids[row_idx] = int(pos[23])
        if len(pos) > 24:
            search_budget_assigned[row_idx] = _optional_int32_search_count(pos[24])
        if len(pos) > 25:
            search_simulations_effective[row_idx] = _optional_int32_search_count(pos[25])
        if len(pos) > 26:
            search_simulations_fresh[row_idx] = _optional_int32_search_count(pos[26])
        if len(pos) > 27:
            search_visits_inherited_credit[row_idx] = _optional_int32_search_count(pos[27])
        if count <= 0:
            continue
        policy_indices[row_idx, :count] = indices.to(dtype=torch.int16)
        policy_values[row_idx, :count] = probs.to(dtype=torch.float32)
        policy_lengths[row_idx] = count

    return {
        'boards': boards,
        'policy_indices': policy_indices,
        'policy_values': policy_values,
        'policy_lengths': policy_lengths,
        'legal_indices': legal_indices,
        'legal_lengths': legal_lengths,
        'values': values,
        'importance_scores': importance_scores,
        'policy_weights': policy_weights,
        'value_weights': value_weights,
        'moves_left': moves_left,
        'source_codes': source_codes,
        'fens': fens,
        'root_q_targets': root_q_targets,
        'history_fens': history_fens,
        'search_changed_top': search_changed_top,
        'search_q_deltas': search_q_deltas,
        'best_q_targets': best_q_targets,
        'played_q_targets': played_q_targets,
        'orig_q_targets': orig_q_targets,
        'policy_kld_targets': policy_kld_targets,
        'search_visits': search_visits,
        'game_ids': game_ids,
        'game_ply_indices': game_ply_indices,
        'regret_targets': regret_targets,
        'archive_ids': archive_ids,
        'search_budget_assigned': search_budget_assigned,
        'search_simulations_effective': search_simulations_effective,
        'search_simulations_fresh': search_simulations_fresh,
        'search_visits_inherited_credit': search_visits_inherited_credit,
        'num_positions': batch_size,
    }


def _resolve_replay_max_policy_targets(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('replay_max_policy_targets', _DEFAULT_REPLAY_MAX_POLICY_TARGETS)
    try:
        return max(1, int(raw_value))
    except Exception:
        return _DEFAULT_REPLAY_MAX_POLICY_TARGETS


def _select_move_from_visits_safe(visit_counts, temperature):
    """
    Select move from visit counts with numeric safeguards.
    """
    if temperature == 0 or len(visit_counts) == 1:
        # Preserve insertion-order tie breaking without allocating a key list
        # and a NumPy array for the common deterministic-selection path.
        return max(visit_counts, key=visit_counts.get)

    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        return moves[np.random.randint(len(moves))]

    probs = visits_temp / total
    probs = np.clip(probs, 0.0, 1.0)
    probs_sum = float(probs.sum())
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return moves[np.random.randint(len(moves))]

    probs /= probs_sum
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


def _resolve_selfplay_max_moves(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('self_play_max_moves', 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _build_history_tensor_from_encoded(
    turn,
    encoded_history,
    history_count,
    history_positions,
    empty_history_tensor,
):
    """Build model input from cached history tensors without FEN parsing."""
    use_black_pov = (turn == chess.BLACK)
    available = len(encoded_history)
    if available <= 0:
        return empty_history_tensor

    current_idx = max(0, min(int(history_count), available - 1))
    current_encoded = encoded_history[current_idx]
    current_tensor = current_encoded[1] if use_black_pov else current_encoded[0]

    if history_positions <= 0:
        return current_tensor

    end = max(0, min(int(history_count), available))
    start = max(0, end - int(history_positions))
    history_slice = encoded_history[start:end]

    history_tensors = []
    for encoded in history_slice:
        history_tensors.append(encoded[1] if use_black_pov else encoded[0])

    pad_count = max(0, int(history_positions) - len(history_tensors))
    if pad_count:
        history_tensors = [empty_history_tensor] * pad_count + history_tensors

    history_tensors.append(current_tensor)
    return np.concatenate(history_tensors, axis=0)


# =============================================================================
# TRUE BATCH MCTS (MULTI-GAME) FOR BETTER GPU UTILIZATION
# =============================================================================

class MultiGameBatchMCTS:
    """
    Batch MCTS that searches multiple games in parallel and batches
    leaf evaluations across all games for higher GPU utilization.
    """

    def __init__(
        self,
        model,
        config,
        device,
        *,
        local_cuda_graph_forward=None,
        local_cuda_graph_exact_batch_sizes=None,
    ):
        self.model = model
        self._search_context_invalidation_epoch = 0
        model_spec = getattr(model, 'model_spec', None)
        if isinstance(model_spec, dict):
            # Search inputs are part of the checkpoint contract. In matches
            # between architectures, do not inherit the other player's
            # encoder/history settings from the process-wide config.
            from src.models.registry import config_for_model_spec

            config = config_for_model_spec(config, model_spec)
        self.config = config
        self.device = device
        if local_cuda_graph_forward is None:
            local_cuda_graph_forward = (config.get('reinforcement_learning', {}) or {}).get(
                'mcts_local_cuda_graph_forward'
            )
        if local_cuda_graph_forward is not None and type(local_cuda_graph_forward) is not bool:
            raise ValueError('local_cuda_graph_forward must be null, false, or true')
        # Auto changes only forward execution on benchmarked local paths.
        # False is a shared config opt-out; True keeps explicit opt-in
        # available for other local models and search modes.
        self._local_cuda_graph_forward_configured = local_cuda_graph_forward
        self._local_cuda_graph_forward_enabled = local_cuda_graph_forward is not False
        self._local_cuda_graph_exact_batch_sizes_custom = (
            local_cuda_graph_exact_batch_sizes is not None
        )
        if local_cuda_graph_exact_batch_sizes is None:
            local_cuda_graph_exact_batch_sizes = _DEFAULT_LOCAL_CUDA_GRAPH_BATCH_SIZES
        self._local_cuda_graph_exact_batch_sizes = tuple(sorted({
            int(size)
            for size in local_cuda_graph_exact_batch_sizes
            if int(size) > 0
        }))
        self._local_cuda_graph_forward_runner = None
        self._local_cuda_graph_forward_generation = None
        self._local_cuda_graph_forward_runner_key = None
        self._local_cuda_graph_forward_lock = threading.Lock()
        self._local_cuda_graph_forward_context = threading.local()
        model_cfg = config.get('model', {}) or {}
        input_spec = (
            (model_spec or {}).get('input_spec', {})
            if isinstance(model_spec, dict)
            else {}
        ) or {}
        self.input_encoder_id = str(
            input_spec.get('encoder_id')
            or getattr(model, 'input_encoder_id', None)
            or model_cfg.get('input_encoder_id', 'planes_history_v1')
        )
        policy_spec = (
            (model_spec or {}).get('policy_spec', {})
            if isinstance(model_spec, dict)
            else {}
        ) or {}
        self.policy_codec_id = str(
            policy_spec.get('codec_id')
            or getattr(model, 'policy_codec_id', None)
            or model_cfg.get('policy_codec_id', 'lc0_1858_v1')
        )
        self.input_encoder = get_input_encoder(self.input_encoder_id)
        # V10 always has a path-sensitive 13/20-plane layout. Whether the
        # repetition plane itself was supervised is a separate checkpoint
        # contract: legacy V10 checkpoints were trained with it unknown (-1),
        # even though MCTS still knows exact repetition counts for draw rules.
        self._path_dependent_repetition = self.input_encoder_id == 'planes_history_v2'
        self.repetition_plane_mode = 'unknown'
        if self._path_dependent_repetition:
            self.repetition_plane_mode = self.input_encoder.resolve_repetition_plane_mode(
                model_spec=model_spec if isinstance(model_spec, dict) else None,
                model_config=model_cfg,
            )
        self._encode_exact_repetition = (
            self._path_dependent_repetition
            and self.repetition_plane_mode == 'exact'
        )
        rl_cfg = config['reinforcement_learning']
        algorithm = str(rl_cfg.get('mcts_algorithm', 'gumbel')).strip().lower()
        if algorithm != 'gumbel':
            raise ValueError(
                "reinforcement_learning.search.algorithm must be 'gumbel'"
            )
        self.mcts_algorithm = 'gumbel'
        (
            self.policy_target_mode,
            self.challenge_delta_prior_fraction,
        ) = _resolve_policy_target_config(rl_cfg)

        self.gumbel_max_considered_actions = max(
            1,
            int(rl_cfg.get('mcts_gumbel_max_considered_actions', 16)),
        )
        self.gumbel_root_schedule = str(
            rl_cfg.get('mcts_gumbel_root_schedule', 'standard')
        ).strip().lower()
        if self.gumbel_root_schedule not in {'standard', 'revisiting'}:
            raise ValueError(
                "reinforcement_learning.mcts_gumbel_root_schedule must be "
                "'standard' or 'revisiting'"
            )
        self.gumbel_reuse_q_scale = str(
            rl_cfg.get('mcts_gumbel_reuse_q_scale', 'fresh')
        ).strip().lower()
        if self.gumbel_reuse_q_scale not in {'fresh', 'total'}:
            raise ValueError(
                "reinforcement_learning.search.gumbel_reuse_q_scale must be "
                "'fresh' or 'total'"
            )
        self.gumbel_scale = max(0.0, float(rl_cfg.get('mcts_gumbel_scale', 1.0)))
        self.gumbel_eval_scale = max(0.0, float(rl_cfg.get('mcts_gumbel_eval_scale', 0.0)))
        self.gumbel_c_visit = max(0.0, float(rl_cfg.get('mcts_gumbel_c_visit', 100.0)))
        self.gumbel_c_scale = max(0.0, float(rl_cfg.get('mcts_gumbel_c_scale', 0.10)))
        self.gumbel_q_range_floor = max(
            1e-8,
            float(rl_cfg.get('mcts_gumbel_q_range_floor', 0.25)),
        )
        self.gumbel_target_temperature = max(
            0.05,
            float(rl_cfg.get('mcts_gumbel_target_temperature', 0.95)),
        )
        self.gumbel_use_mixed_value = bool(
            rl_cfg.get('mcts_gumbel_use_mixed_value', True)
        )
        self.dynamic_budget_enabled = bool(
            config['reinforcement_learning'].get('mcts_dynamic_budget_enabled', False)
        )
        self.dynamic_budget_strategy = str(
            rl_cfg.get('mcts_dynamic_budget_strategy', 'feedback')
        ).strip().lower()
        if self.dynamic_budget_strategy not in {'prior', 'feedback'}:
            raise ValueError(
                "reinforcement_learning.search.mcts_dynamic_budget_strategy must be "
                "'prior' or 'feedback'"
            )
        self.dynamic_budget_minimum = max(
            1,
            int(config['reinforcement_learning'].get('mcts_dynamic_budget_min', 64)),
        )
        self.dynamic_budget_maximum_multiplier = max(
            1.0,
            float(
                config['reinforcement_learning'].get(
                    'mcts_dynamic_budget_max_multiplier',
                    5.0 / 3.0,
                )
            ),
        )
        # Single-game sessions may carry only earned, bounded budget savings
        # between successive positions. Explicit tree keys are required so
        # temporary board identities and unrelated games never share credit.
        self._dynamic_budget_credits = OrderedDict()
        self._dynamic_budget_credit_generation = None
        self.eval_easy_cut_enabled = bool(
            rl_cfg.get('eval_mcts_dynamic_budget_enabled', False)
        )
        self.eval_easy_cut_minimum_fraction = max(
            0.50,
            min(
                1.0,
                float(rl_cfg.get('eval_mcts_dynamic_budget_min_fraction', 5.0 / 6.0)),
            ),
        )
        self.eval_easy_cut_difficulty_threshold = max(
            0.0,
            min(
                1.0,
                float(rl_cfg.get('eval_mcts_dynamic_budget_difficulty_threshold', 0.75)),
            ),
        )
        self.eval_easy_cut_chunk = max(
            1,
            int(rl_cfg.get('eval_mcts_dynamic_budget_chunk', 16)),
        )
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)
        # SingleGameMCTS is the interactive path and has no other roots with
        # which to fill an inference batch. A positive cap batches more of its
        # exact per-phase simulation budget together; 0 keeps the legacy
        # quarter-of-remaining schedule. Multi-root RL/eval scheduling is
        # deliberately unchanged.
        self.single_root_batch_cap = max(
            0,
            int(config['reinforcement_learning'].get('mcts_single_root_batch_cap', 0) or 0),
        )
        # History configuration (POV)
        self.history_positions = int(
            input_spec.get(
                'history_positions',
                model_cfg.get('history_positions', 0),
            )
            or 0
        )
        self.history_storage_dtype = (
            np.float16
            if bool(config['reinforcement_learning'].get('self_play_history_fp16', True))
            else np.float32
        )
        remote_fp16_input = bool(
            getattr(self.model, 'supports_remote_legal_gather', False)
            and str(getattr(self.model, 'transport_dtype', '')).lower() == 'float16'
        )
        # Central inference transports CUDA self-play inputs as FP16 already.
        # Building the complete input batch in that dtype avoids an otherwise
        # redundant FP32 scratch write and FP32->FP16 copy in every request.
        self.input_storage_dtype = (
            np.float16
            if self.device.type == 'cuda' or remote_fp16_input
            else np.float32
        )

        # Tree reuse
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)
        self.syzygy = _get_syzygy_oracle(config)
        self.cache_node_tensors = False
        self.cache_history_tensors = bool(
            config['reinforcement_learning'].get('mcts_cache_history_tensors', True)
        )
        # Exact v10 repetition planes depend on the complete root path, so a
        # node-identity cache could leak one path's count into another. Legacy
        # v10 checkpoints use an unknown (-1) repetition plane; their complete
        # per-node encoding is path-independent and remains cacheable.
        if self._encode_exact_repetition:
            self.cache_history_tensors = False
        # History pairs are 4 KiB each in FP16. Keeping one on every expanded
        # node made tree RAM grow with all visits, although only recent
        # ancestors are revisited while assembling leaf inputs. A bounded LRU
        # preserves the useful hot set without pinning the full forest.
        self._history_tensor_cache_capacity = max(
            256,
            int(self.eval_batch_size) * max(1, int(self.history_positions)) * 2,
        )
        self._history_tensor_cache = OrderedDict()
        self.tree_reuse_visit_credit_enabled = bool(
            config['reinforcement_learning'].get(
                'mcts_tree_reuse_visit_credit_enabled',
                True,
            )
        )

        self.tactical_priors_enabled = bool(
            config['reinforcement_learning'].get('mcts_tactical_priors_enabled', False)
        )
        self.tactical_capture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_capture_bonus', 0.0)),
        )
        self.tactical_winning_capture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_winning_capture_bonus', 0.0)),
        )
        self.tactical_check_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_check_bonus', 0.0)),
        )
        self.tactical_promotion_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_promotion_bonus', 0.0)),
        )
        self.tactical_recapture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_recapture_bonus', 0.0)),
        )

        # Inference optimization (AMP on GPU)
        self.use_amp = config.get('hardware', {}).get('use_amp', False) and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
        if self._path_dependent_repetition:
            self._history_frame_planes = int(self.input_encoder.PLANES_PER_POSITION)
            self._current_planes = int(self.input_encoder.input_planes(0))
            self._empty_history_tensor = np.zeros(
                (self._history_frame_planes, 8, 8),
                dtype=self.history_storage_dtype,
            )
            self._empty_history_tensor[12].fill(-1.0)
        else:
            self._history_frame_planes = int(_EMPTY_HISTORY_TENSOR.shape[0])
            self._current_planes = self._history_frame_planes
            self._empty_history_tensor = _EMPTY_HISTORY_TENSOR.astype(
                self.history_storage_dtype, copy=False
            )
        # Keep the legacy attribute as the temporal frame width; v10 has 13
        # planes per historical frame and a separate 20-plane current state.
        self._board_planes = self._history_frame_planes
        self._history_planes = int(self.history_positions) * self._history_frame_planes
        self._input_planes = int(self.input_encoder.input_planes(self.history_positions))
        self._legal_index_scratch = {}
        self._legal_index_tensor_scratch = {}
        self._board_input_scratch = {}
        self._board_input_tensor_scratch = {}
        self._native_board_plane_scratch = {}
        self._use_pinned_staging = bool(
            config.get('hardware', {}).get('pin_memory', True)
            and self.device.type == 'cuda'
            and torch.cuda.is_available()
        )
        # Keep lightweight performance counters always on so details/performance CSV/PNG
        # stays useful even when verbose RL debug profiling is disabled. Tight-loop leaf
        # operations are sampled; verbose CUDA/sub-stage timers remain opt-in.
        self.profile_enabled = True
        self.profile_detail_enabled = _debug_bool(
            config,
            'rl',
            'profile_mcts_detail',
            _debug_bool(config, 'rl', 'profile_training', False),
        )
        debug_root = config.get('debug', {}) or {}
        debug_rl = debug_root.get('rl', {}) or {}
        try:
            self.profile_sample_rate = int(debug_rl.get('profile_mcts_sample_rate', debug_root.get('profile_mcts_sample_rate', 64)))
        except (TypeError, ValueError):
            self.profile_sample_rate = 64
        self.profile_sample_rate = max(0, int(self.profile_sample_rate))
        self._profile_sample_counts = {}
        self.cuda_stage_profile_enabled = bool(
            self.profile_detail_enabled
            and _debug_bool(config, 'rl', 'profile_training', False)
        )
        self._profile_stats = {}
        self._native_mcts = get_native_mcts()
        self._native_input_planes_enabled = (
            self._native_mcts is not None
            and not self._path_dependent_repetition
        )
        self._native_move_indices_enabled = self._native_mcts is not None
        # The native forest persists across plies. Python only materializes
        # evaluated leaves and root telemetry; traversal, virtual visits,
        # backup, rerooting and periodic dead-branch compaction stay in C++.
        self._native_tree_enabled = self._native_mcts is not None
        self._native_forest = None

        self._tactical_capture_enabled = (
            self.tactical_capture_bonus > 0.0
            or self.tactical_winning_capture_bonus > 0.0
        )
        self._tactical_promotion_enabled = self.tactical_promotion_bonus > 0.0
        self._tactical_check_enabled = self.tactical_check_bonus > 0.0
        self._tactical_recapture_enabled = self.tactical_recapture_bonus > 0.0
        self.reset_profile_stats()

    def reset_profile_stats(self):
        self._profile_sample_counts = {}
        self._native_root_selection_calls = 0
        self._python_root_selection_calls = 0
        self._native_completed_q_calls = 0
        self._python_completed_q_calls = 0
        self._profile_stats = {
            'search_many_time': 0.0,
            'search_many_calls': 0,
            # Sum of root->leaf path lengths. This is a selection traversal
            # counter, not the number of completed MCTS simulations/visits.
            'selection_node_traversals': 0,
            'search_root_setup_time': 0.0,
            'search_selection_time': 0.0,
            'search_backprop_time': 0.0,
            'native_tree_expand_sync_time': 0.0,
            'native_tree_backup_time': 0.0,
            'native_tree_import_time': 0.0,
            'native_tree_sync_time': 0.0,
            'native_tree_selection_batches': 0,
            'native_tree_backup_batches': 0,
            'native_tree_selected_leaves': 0,
            'python_tree_selected_leaves': 0,
            'native_tree_import_nodes': 0,
            'native_tree_import_edges': 0,
            'native_tree_fallbacks': 0,
            'search_metadata_time': 0.0,
            'board_materialize_time': 0.0,
            'board_materialize_calls': 0,
            'terminal_checks_time': 0.0,
            'terminal_checks_calls': 0,
            'batch_expand_eval_time': 0.0,
            'batch_expand_eval_calls': 0,
            'batch_expand_dedup_terminal_time': 0.0,
            'batch_expand_legal_moves_time': 0.0,
            'batch_expand_move_index_time': 0.0,
            'batch_expand_tensor_pack_time': 0.0,
            'batch_expand_history_time': 0.0,
            'batch_expand_input_pack_time': 0.0,
            'batch_expand_legal_index_pack_time': 0.0,
            'batch_expand_cpu_policy_time': 0.0,
            'batch_expand_value_fanout_time': 0.0,
            'board_to_tensor_time': 0.0,
            'board_to_tensor_calls': 0,
            'nn_inference_time': 0.0,
            'nn_inference_calls': 0,
            'nn_inference_batch_items': 0,
            'nn_h2d_time': 0.0,
            'nn_gpu_forward_time': 0.0,
            'nn_gpu_postprocess_time': 0.0,
            'nn_d2h_time': 0.0,
            'nn_legal_move_items': 0,
            'central_inference_shared_requests': 0,
            'central_inference_shared_bytes_avoided': 0,
            'central_inference_compact_policy_requests': 0,
            'central_inference_compact_output_bytes_avoided': 0,
            'central_inference_shared_slot_wait_time': 0.0,
            'central_inference_cache_queries': 0,
            'central_inference_cache_bypassed_positions': 0,
            'central_inference_cache_hits': 0,
            'central_inference_dedup_hits': 0,
            'central_inference_cache_suspensions': 0,
            'central_inference_cache_reactivations': 0,
            'central_inference_nn_evaluated_positions': 0,
            'central_inference_server_cache_lookup_time': 0.0,
            'central_inference_server_staging_copy_time': 0.0,
            'central_inference_gpu_batch_fill_sum': 0.0,
            'central_inference_server_batch_target_sum': 0,
            'central_inference_server_output_finalize_wait_time': 0.0,
            'central_inference_server_output_pipeline_requests': 0,
            'central_inference_server_auto_calibrated_requests': 0,
        }

    def _evaluation_generation_token(self):
        """Return a stable evaluator generation token for this search call.

        Remote inference exposes a server epoch so a long-lived proxy notices
        model reloads without a per-leaf RPC. Local eager modules use tensor
        identity/version counters. Unknown local callables can still search,
        but cannot safely reuse a root.
        """
        model = self.model
        invalidation_epoch = int(
            getattr(self, '_search_context_invalidation_epoch', 0)
        )
        getter = getattr(model, 'get_evaluation_generation', None)
        if callable(getter):
            try:
                generation = getter()
            except Exception:
                return None
            frozen = _canonical_context_value(generation)
            if generation is None or frozen is None:
                return None
            return ('remote', invalidation_epoch, frozen)

        if bool(getattr(model, 'supports_remote_legal_gather', False)):
            return None
        if not isinstance(model, torch.nn.Module):
            return None
        try:
            # Training mode disables tree reuse below, but remains observable
            # so a local search can reject a train/eval transition that occurs
            # while its batched evaluations are in flight.
            training_modes = tuple(
                (str(name), bool(module.training))
                for name, module in model.named_modules()
            )
            tensors = []
            for kind, named_tensors in (
                ('p', model.named_parameters()),
                ('b', model.named_buffers()),
            ):
                for name, tensor in named_tensors:
                    version = getattr(tensor, '_version', None)
                    if version is None:
                        return None
                    tensors.append((
                        kind,
                        str(name),
                        id(tensor),
                        int(tensor.data_ptr()),
                        int(version),
                        tuple(int(dim) for dim in tensor.shape),
                        str(tensor.dtype),
                        str(tensor.device),
                    ))
            if not tensors:
                return None
            return (
                'local',
                invalidation_epoch,
                id(model),
                type(model).__module__,
                type(model).__qualname__,
                _canonical_context_value(getattr(model, 'model_spec', None)),
                training_modes,
                tuple(tensors),
            )
        except Exception:
            return None

    def _local_cuda_graph_auto_eligible(
        self,
        original_game_count,
        native_forest,
        *,
        feedback_active=False,
        all_roots_adaptive=False,
    ):
        """Limit automatic graphs to measured native revisiting contracts."""
        if (
            original_game_count < 1
            or self.gumbel_root_schedule != 'revisiting'
            or native_forest is None
            or not native_forest.active
        ):
            return False
        from src.models.architecture.se_cnn_v9 import SECNNV9
        from src.models.architecture.nested_cnn_v10 import NestedCNNV10

        # Unknown wrappers/subclasses may change forward semantics. They can
        # still opt in explicitly, but never enter automatic graph capture.
        if original_game_count == 1:
            return type(self.model) in (SECNNV9, NestedCNNV10)
        return bool(
            original_game_count > 1
            and feedback_active
            and all_roots_adaptive
            and type(self.model) is SECNNV9
        )

    def _local_cuda_graph_batch_sizes_for_call(self, *, auto_eligible, game_count):
        if (
            auto_eligible
            and game_count > 1
            and self._local_cuda_graph_forward_configured is None
            and not self._local_cuda_graph_exact_batch_sizes_custom
        ):
            return _FEEDBACK_MULTIROOT_LOCAL_CUDA_GRAPH_BATCH_SIZES
        return self._local_cuda_graph_exact_batch_sizes

    def _prepare_local_cuda_graph_forward(
        self,
        evaluation_generation,
        *,
        auto_eligible=False,
        exact_batch_sizes=None,
    ):
        """Select a stable local forward runner once per search, never per leaf.

        Graphs are only considered for an eval-mode local ``nn.Module`` with
        a stable generation token. Remote, training, and unknown evaluators
        stay on their existing eager/transport path.
        """
        context = self._local_cuda_graph_forward_context
        context.runner = None
        if exact_batch_sizes is None:
            exact_batch_sizes = self._local_cuda_graph_exact_batch_sizes
        else:
            exact_batch_sizes = tuple(
                sorted({int(size) for size in exact_batch_sizes if int(size) > 0})
            )
        runner_key = (evaluation_generation, exact_batch_sizes)
        eligible = not (
            not self._local_cuda_graph_forward_enabled
            or (self._local_cuda_graph_forward_configured is None and not auto_eligible)
            or self.device.type != 'cuda'
            or not torch.cuda.is_available()
            or evaluation_generation is None
            or not isinstance(self.model, torch.nn.Module)
            or bool(getattr(self.model, 'supports_remote_legal_gather', False))
            or any(bool(module.training) for module in self.model.modules())
        )
        if not eligible:
            # A prior stable eval generation must not retain graph pools after
            # the model becomes mutable, remote, or otherwise ineligible.
            with self._local_cuda_graph_forward_lock:
                old_runner = self._local_cuda_graph_forward_runner
                if old_runner is not None:
                    old_runner.clear()
                self._local_cuda_graph_forward_runner = None
                self._local_cuda_graph_forward_generation = None
                self._local_cuda_graph_forward_runner_key = None
            return

        with self._local_cuda_graph_forward_lock:
            if self._local_cuda_graph_forward_runner_key != runner_key:
                old_runner = self._local_cuda_graph_forward_runner
                if old_runner is not None:
                    old_runner.clear()
                self._local_cuda_graph_forward_runner = None
                self._local_cuda_graph_forward_generation = evaluation_generation
                self._local_cuda_graph_forward_runner_key = runner_key
                try:
                    from src.inference.cuda_graph_forward import CUDAGraphForwardRunner

                    self._local_cuda_graph_forward_runner = CUDAGraphForwardRunner(
                        self.model,
                        self.device,
                        use_amp=self.use_amp,
                        amp_dtype=self.amp_dtype,
                        exact_batch_sizes=exact_batch_sizes,
                        min_capture_observations=2,
                        max_entries=len(exact_batch_sizes),
                        max_graph_memory_fraction=0.04,
                        min_free_memory_bytes=512 * 1024 * 1024,
                        warmup_iterations=3,
                    )
                except Exception:
                    # Optional acceleration must never prevent ordinary search.
                    self._local_cuda_graph_forward_runner = None
            context.runner = self._local_cuda_graph_forward_runner

    def _local_model_forward(self, inputs, model_kwargs, provenance=None):
        """Run the unchanged raw model contract, optionally through local graphs.

        Returns ``(policy, value, used_runner)``. The third value lets callers
        avoid reading a stale compact-policy flag after a graph replay.
        """
        context = getattr(self, '_local_cuda_graph_forward_context', None)
        runner = getattr(context, 'runner', None) if context is not None else None
        raw_graph_call = model_kwargs == {'apply_log_softmax': False}
        policy, value = predict_raw(
            self.model,
            inputs,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            graph_runner=runner if raw_graph_call else None,
            model_kwargs=model_kwargs,
            provenance=provenance,
        )
        return policy, value, bool(runner is not None and raw_graph_call)

    def _local_cuda_readback_buffers(self, batch_size, legal_width, device):
        """Return thread-local pinned outputs and one reusable transfer event."""
        device = torch.device(device)
        device_key = str(device)
        batch_size = int(batch_size)
        legal_width = int(legal_width)
        policy_elements = batch_size * legal_width
        context = self._local_cuda_graph_forward_context
        scratch = getattr(context, 'mcts_readback_scratch', None)
        if (
            scratch is not None
            and scratch['device'] == device_key
            and scratch['policy_capacity_elements'] >= policy_elements
            and scratch['value_capacity'] >= batch_size
        ):
            return (
                scratch['policy_storage'][:policy_elements].view(
                    batch_size,
                    legal_width,
                ),
                scratch['values_storage'][:batch_size],
                scratch['event'],
            )

        previous = scratch if scratch is not None and scratch['device'] == device_key else None
        graph_batch_capacity = max(
            self._local_cuda_graph_exact_batch_sizes,
            default=1,
        )
        policy_capacity_elements = max(
            policy_elements,
            graph_batch_capacity * int(MAX_LEGAL_MOVES),
            int(previous['policy_capacity_elements']) if previous is not None else 0,
        )
        value_capacity = max(
            batch_size,
            graph_batch_capacity,
            int(previous['value_capacity']) if previous is not None else 0,
        )
        with torch.cuda.device(device):
            if (
                previous is not None
                and previous['policy_capacity_elements'] >= policy_capacity_elements
            ):
                policy_storage = previous['policy_storage']
            else:
                policy_storage = torch.empty(
                    (policy_capacity_elements,),
                    dtype=torch.float16,
                    pin_memory=True,
                )
            if previous is not None and previous['value_capacity'] >= value_capacity:
                values_storage = previous['values_storage']
            else:
                values_storage = torch.empty(
                    (value_capacity,),
                    dtype=torch.float32,
                    pin_memory=True,
                )
            transfer_event = (
                previous['event']
                if previous is not None
                else torch.cuda.Event(blocking=False)
            )
        context.mcts_readback_scratch = {
            'device': device_key,
            'policy_capacity_elements': policy_capacity_elements,
            'value_capacity': value_capacity,
            'policy_storage': policy_storage,
            'values_storage': values_storage,
            'event': transfer_event,
        }
        return (
            policy_storage[:policy_elements].view(batch_size, legal_width),
            values_storage[:batch_size],
            transfer_event,
        )

    @property
    def local_cuda_graph_forward_stats(self):
        """Expose bounded runner counters for opt-in profiling harnesses."""
        runner = self._local_cuda_graph_forward_runner
        if runner is None:
            return {
                'enabled': self._local_cuda_graph_forward_enabled,
                'configured': self._local_cuda_graph_forward_configured,
                'active': False,
                'entries': 0,
                'captures': 0,
                'replays': 0,
                'eager_fallbacks': 0,
                'last_fallback_reason': 'not_initialized',
            }
        return {
            'enabled': True,
            'configured': self._local_cuda_graph_forward_configured,
            'active': getattr(self._local_cuda_graph_forward_context, 'runner', None) is runner,
            **runner.stats,
        }

    def clear_local_cuda_graph_forward(self):
        """Release captured graphs before unloading or replacing a local model."""
        with self._local_cuda_graph_forward_lock:
            runner = self._local_cuda_graph_forward_runner
            if runner is not None:
                runner.clear()
            self._local_cuda_graph_forward_runner = None
            self._local_cuda_graph_forward_generation = None
            self._local_cuda_graph_forward_runner_key = None
            self._local_cuda_graph_forward_context.runner = None
            self._local_cuda_graph_forward_context.mcts_readback_scratch = None

    def invalidate_search_context(self):
        """Invalidate root reuse after custom or opaque weight mutation.

        Standard local ``load_state_dict``/``copy_`` and optimizer updates are
        detected by tensor versions; call this after writes such as
        ``parameter.data.copy_(...)`` that bypass those counters.
        """
        self._search_context_invalidation_epoch = int(
            getattr(self, '_search_context_invalidation_epoch', 0)
        ) + 1
        self._dynamic_budget_credits.clear()
        self._dynamic_budget_credit_generation = None
        if hasattr(self, '_local_cuda_graph_forward_lock'):
            self.clear_local_cuda_graph_forward()

    def _root_reuse_input_contract(self):
        return (
            self.input_encoder_id,
            self.policy_codec_id,
            int(self.history_positions),
            int(self._input_planes),
            int(self._current_planes),
            int(self._history_frame_planes),
            str(np.dtype(self.input_storage_dtype)),
            str(np.dtype(self.history_storage_dtype)),
            bool(self._path_dependent_repetition),
            bool(self._encode_exact_repetition),
            str(self.repetition_plane_mode),
            bool(self.use_amp),
            str(self.amp_dtype),
            str(self.device),
            _canonical_context_value(getattr(self.model, 'model_spec', None)),
            bool(self.tactical_priors_enabled),
            float(self.tactical_capture_bonus),
            float(self.tactical_winning_capture_bonus),
            float(self.tactical_check_bonus),
            float(self.tactical_promotion_bonus),
            float(self.tactical_recapture_bonus),
        )

    def _history_pair_context_digest(self, entry, *, repetition_count=None):
        encoded = self._encode_history_entry(
            entry,
            repetition_count=repetition_count,
        )
        return (
            self._history_context_frame_digest(encoded[0]),
            self._history_context_frame_digest(encoded[1]),
        )

    def _history_context_frame_digest(self, tensor):
        """Hash a history frame after the same cast used for model input."""
        # Provenance must describe the values that actually reach the model,
        # not the caller's temporary storage dtype. Elo may retain FP32
        # history pairs while GPU/central search transports them as FP16; the
        # batch builder casts both representations to input_storage_dtype.
        # Hashing before that cast falsely rejects otherwise identical roots.
        frame = self._history_frame_for_model(tensor)
        frame = np.asarray(frame, dtype=np.dtype(self.input_storage_dtype))
        return _history_tensor_context_digest(frame)

    def _model_is_training_for_reuse(self):
        """Return whether evaluator training mode makes root reuse unsafe.

        Local ``nn.Module`` trees need every submodule checked because a child
        can remain in training mode even when its parent flag is false. Remote
        inference proxies are intentionally not modules; their explicit
        ``training`` property is the mode contract. Unknown evaluators fail
        closed for reuse without affecting whether they can run a search.
        """
        model = self.model
        if isinstance(model, torch.nn.Module):
            return any(module.training for module in model.modules())
        training = getattr(model, 'training', None)
        return True if training is None else bool(training)

    def _build_root_reuse_context(
        self,
        board,
        board_history,
        position_counts,
        repetition_known,
        evaluation_generation,
        *,
        evaluation_generation_is_canonical=False,
    ):
        """Build exact bounded NN-history and rule provenance for a root."""
        if (
            evaluation_generation is None
            or not bool(repetition_known)
            or self._model_is_training_for_reuse()
        ):
            return None
        canonical_counts = _canonical_position_counts(position_counts)
        if canonical_counts is None:
            return None
        counts_map = dict(canonical_counts)
        position_key = _board_position_key(board)
        if int(counts_map.get(position_key, 0)) <= 0:
            return None

        history_positions = max(0, int(self.history_positions))
        entries = (
            list(board_history[-history_positions:])
            if history_positions and board_history else []
        )
        history_frames = [
            self._history_pair_context_digest(entry)
            for entry in entries
        ]
        if history_positions:
            empty_frame = self._history_frame_for_model(self._empty_history_tensor)
            empty_digest = self._history_context_frame_digest(empty_frame)
            padding = max(0, history_positions - len(history_frames))
            history_frames = [(empty_digest, empty_digest)] * padding + history_frames

        current_repetition_count = (
            max(0, int(counts_map.get(position_key, 0)) - 1)
            if self._encode_exact_repetition else None
        )
        current_frame = (
            self._history_pair_context_digest(
                board,
                repetition_count=current_repetition_count,
            )
            if history_positions else None
        )
        return _RootReuseContext(
            board_identity=_board_tree_reuse_identity(board),
            history_frames=tuple(history_frames),
            current_frame=current_frame,
            history_positions=history_positions,
            position_counts=canonical_counts,
            repetition_known=True,
            input_contract=self._root_reuse_input_contract(),
            evaluation_generation=(
                evaluation_generation
                if evaluation_generation_is_canonical
                else _canonical_context_value(evaluation_generation)
            ),
        )

    def _verify_search_generation(self, start_generation, roots, game_states, mapping_states):
        """Reject mixed-generation results and invalidate their reusable roots."""
        remote = bool(getattr(self.model, 'supports_remote_legal_gather', False))
        if start_generation is None and not remote:
            return
        end_generation = self._evaluation_generation_token()
        if end_generation is not None and end_generation == start_generation:
            return

        for idx, root in enumerate(roots):
            if root is not None:
                root._reuse_context_anchor = None
            state = game_states[idx]
            if mapping_states[idx]:
                state['root'] = None
                state['_root_synced'] = False
            else:
                if len(state) > 1:
                    state[1] = None
                if len(state) > 2:
                    state[2] = False
        raise RuntimeError(
            'MCTS evaluator generation changed or became unavailable during search; '
            'discarded the search result and invalidated its reusable roots.'
        )

    def _profile_add(self, key, value):
        self._profile_stats[key] = float(self._profile_stats.get(key, 0.0)) + float(value)

    def _profile_inc(self, key, value=1):
        self._profile_stats[key] = int(self._profile_stats.get(key, 0)) + int(value)

    def _profile_sample_begin(self, key):
        if self.profile_detail_enabled:
            return time.perf_counter(), 1
        if not self.profile_enabled or self.profile_sample_rate <= 0:
            return None, 0
        count = int(self._profile_sample_counts.get(key, 0)) + 1
        self._profile_sample_counts[key] = count
        if count % self.profile_sample_rate != 0:
            return None, 0
        return time.perf_counter(), self.profile_sample_rate

    def _profile_sample_finish(self, key, start_time, scale):
        if start_time is None or scale <= 0:
            return
        self._profile_add(f'{key}_time', (time.perf_counter() - start_time) * float(scale))
        self._profile_inc(f'{key}_calls', int(scale))

    def get_profile_stats(self):
        profile = dict(self._profile_stats)
        profile['native_root_selection_calls'] = int(self._native_root_selection_calls)
        profile['python_root_selection_calls'] = int(self._python_root_selection_calls)
        profile['native_completed_q_calls'] = int(self._native_completed_q_calls)
        profile['python_completed_q_calls'] = int(self._python_completed_q_calls)
        return profile

    def _board_to_tensor_profiled(
        self,
        board,
        flip_perspective=None,
        storage_dtype=None,
        out=None,
        repetition_count=None,
    ):
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor')
        dtype = np.float32 if storage_dtype is None else storage_dtype
        if self._path_dependent_repetition:
            repetition_count = self.input_encoder.repetition_count_for_mode(
                self.repetition_plane_mode,
                repetition_count,
            )
            direct_out = (
                isinstance(out, np.ndarray)
                and out.dtype == np.dtype(dtype)
                and out.flags.writeable
                and (out.flags.c_contiguous or out.flags.f_contiguous)
            )
            tensor = self.input_encoder.encode(
                board,
                flip_perspective=flip_perspective,
                dtype=dtype,
                out=out if direct_out else None,
                repetition_count=repetition_count,
            )
            if out is not None and not direct_out:
                # Match the historical allocation-then-cast behavior for
                # mismatched dtypes and layouts that the encoder cannot safely
                # fill as a direct contiguous destination.
                np.copyto(out, tensor)
                tensor = out
        else:
            # Preserve the established v9 native/helper path exactly.
            tensor = board_to_tensor(
                board,
                flip_perspective=flip_perspective,
                dtype=dtype,
                out=out,
            )
        self._profile_sample_finish('board_to_tensor', sample_t0, sample_scale)
        return tensor

    def _history_tensor_pair_for_board(self, board, repetition_count=None):
        storage_dtype = self.history_storage_dtype
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor_pair')
        if self._path_dependent_repetition:
            repetition_count = self.input_encoder.repetition_count_for_mode(
                self.repetition_plane_mode,
                repetition_count,
            )
            white_tensor, black_tensor = self.input_encoder.encode_pair(
                board,
                dtype=storage_dtype,
                repetition_count=repetition_count,
            )
        else:
            white_tensor, black_tensor = board_to_tensor_pair(
                board, dtype=storage_dtype
            )
        if sample_t0 is not None and sample_scale > 0:
            self._profile_add('board_to_tensor_time', (time.perf_counter() - sample_t0) * float(sample_scale))
            self._profile_inc('board_to_tensor_calls', int(sample_scale) * 2)
        return (white_tensor, black_tensor)

    def _history_tensor_pair_for_node(self, node, repetition_count=None):
        if self._encode_exact_repetition:
            # Exact repetition planes are path dependent. Do not reuse a pair
            # cached only by node/transposition identity.
            return self._history_tensor_pair_for_board(
                node.board,
                repetition_count=repetition_count,
            )
        if not self.cache_history_tensors:
            return self._history_tensor_pair_for_board(
                node.board,
                repetition_count=repetition_count,
            )

        cache_key = id(node)
        cached_entry = self._history_tensor_cache.get(cache_key)
        if cached_entry is not None:
            node_ref, cached = cached_entry
            if node_ref() is node:
                self._history_tensor_cache.move_to_end(cache_key)
                return cached
            del self._history_tensor_cache[cache_key]

        cached = self._history_tensor_pair_for_board(
            node.board,
            repetition_count=repetition_count,
        )
        self._history_tensor_cache[cache_key] = (weakref.ref(node), cached)
        self._history_tensor_cache.move_to_end(cache_key)
        while len(self._history_tensor_cache) > self._history_tensor_cache_capacity:
            self._history_tensor_cache.popitem(last=False)
        return cached

    def _encode_native_history_pairs(self, pending):
        """Encode the cache misses from one leaf batch in a single native call."""
        if not pending:
            return {}

        profile_t0 = time.perf_counter() if self.profile_enabled else None
        pending_items = list(pending.items())
        pair_count = len(pending_items)
        row_count = pair_count * 2
        (
            piece_masks,
            black_to_move,
            castling_masks,
            en_passant_masks,
            halfmove_values,
            fullmove_values,
        ) = self._get_native_board_plane_scratch(row_count)
        encoded_pairs = np.empty(
            (row_count, 16, 8, 8),
            dtype=self.history_storage_dtype,
        )
        for pair_idx, (_cache_key, node) in enumerate(pending_items):
            board = node.board
            white_row = pair_idx * 2
            black_row = white_row + 1
            for plane_idx, (color, piece_type) in enumerate(_INPUT_PIECE_SPECS):
                mask = int(board[color, piece_type])
                piece_masks[white_row, plane_idx] = mask
                piece_masks[black_row, plane_idx] = mask

            castling_rights = board.castling_rights
            ep_square = board.en_passant_square
            ep_mask = 0 if ep_square is None else 1 << chess.square_index(ep_square)
            halfmove_value = min(float(board.halfmove_clock) / 50.0, 1.0)
            fullmove_value = min(float(board.fullmove_number) / 100.0, 1.0)
            for row_idx, is_black in ((white_row, False), (black_row, True)):
                black_to_move[row_idx] = int(is_black)
                pov_color = chess.BLACK if is_black else chess.WHITE
                kingside_flag = (
                    chess.BLACK_KINGSIDE if is_black else chess.WHITE_KINGSIDE
                )
                queenside_flag = (
                    chess.BLACK_QUEENSIDE if is_black else chess.WHITE_QUEENSIDE
                )
                kingside = kingside_flag in castling_rights
                queenside = queenside_flag in castling_rights
                castling_mask = 0
                if kingside or queenside:
                    king_square = chess.king_square(board, pov_color)
                    if king_square is not None:
                        king_index = chess.square_index(king_square)
                        castling_mask |= 1 << king_index
                        back_rank_base = (king_index >> 3) << 3
                        if kingside:
                            castling_mask |= 1 << (back_rank_base + 7)
                        if queenside:
                            castling_mask |= 1 << back_rank_base
                castling_masks[row_idx] = castling_mask
                en_passant_masks[row_idx] = ep_mask
                halfmove_values[row_idx] = halfmove_value
                fullmove_values[row_idx] = fullmove_value

        self._native_mcts.encode_board_planes(
            piece_masks,
            black_to_move,
            castling_masks,
            en_passant_masks,
            halfmove_values,
            fullmove_values,
            encoded_pairs,
        )
        resolved = {}
        cache = self._history_tensor_cache
        for pair_idx, (cache_key, node) in enumerate(pending_items):
            pair = encoded_pairs[pair_idx * 2:pair_idx * 2 + 2].copy()
            cached = (pair[0], pair[1])
            cache[cache_key] = (weakref.ref(node), cached)
            cache.move_to_end(cache_key)
            resolved[cache_key] = (node, cached)
            while len(cache) > self._history_tensor_cache_capacity:
                cache.popitem(last=False)

        if profile_t0 is not None:
            self._profile_add('board_to_tensor_time', time.perf_counter() - profile_t0)
            self._profile_inc('board_to_tensor_calls', pair_count * 2)
        return resolved

    def _fill_history_prefixes_for_nodes(
        self,
        nodes,
        game_indices,
        board_histories,
        root_position_counts,
        root_repetition_known,
        destinations,
    ):
        """Fill a leaf batch's histories while batching newly used MCTS ancestors."""
        if self.history_positions <= 0:
            return

        profile_t0 = time.perf_counter() if self.profile_detail_enabled else None
        native_batch = (
            self._native_input_planes_enabled
            and self.cache_history_tensors
        )
        cache = self._history_tensor_cache
        pending = {}
        resolved = {}
        rows = []
        history_positions = int(self.history_positions)
        for row_idx, node in enumerate(nodes):
            entries = self._collect_node_history_entries(
                node,
                board_histories[game_indices[row_idx]],
            )
            use_black_pov = node.board.turn == chess.BLACK
            first_slot = max(0, history_positions - len(entries))
            if native_batch:
                for entry in entries:
                    if not isinstance(entry, MCTSNode):
                        continue
                    cache_key = id(entry)
                    already_resolved = resolved.get(cache_key)
                    if already_resolved is not None and already_resolved[0] is entry:
                        continue
                    cached_entry = cache.get(cache_key)
                    if cached_entry is not None:
                        node_ref, cached = cached_entry
                        if node_ref() is entry:
                            cache.move_to_end(cache_key)
                            resolved[cache_key] = (entry, cached)
                            continue
                        del cache[cache_key]
                    pending.setdefault(cache_key, entry)
            gi = int(game_indices[row_idx])
            rows.append((
                node,
                entries,
                first_slot,
                use_black_pov,
                gi,
                destinations[row_idx],
            ))

        if pending:
            resolved.update(self._encode_native_history_pairs(pending))

        for _node, entries, first_slot, use_black_pov, gi, destination in rows:
            destination.fill(0.0)
            if self._path_dependent_repetition:
                destination.reshape(
                    history_positions,
                    self._history_frame_planes,
                    8,
                    8,
                )[:, 12].fill(-1.0)
            for slot, entry in enumerate(entries, start=first_slot):
                begin = slot * self._board_planes
                end = begin + self._board_planes
                repetition_count = (
                    _repetition_count_before_node(
                        entry,
                        root_position_counts[gi],
                        root_repetition_known[gi],
                    )
                    if self._path_dependent_repetition
                    and isinstance(entry, MCTSNode)
                    else None
                )
                if native_batch and isinstance(entry, MCTSNode):
                    cached = resolved.get(id(entry))
                    if cached is not None and cached[0] is entry:
                        pair = cached[1]
                        tensor = pair[1] if use_black_pov else pair[0]
                    else:
                        tensor = self._history_tensor_for_entry(entry, use_black_pov)
                else:
                    tensor = self._history_tensor_for_entry(
                        entry,
                        use_black_pov,
                        repetition_count=repetition_count,
                    )
                destination[begin:end] = tensor

        if profile_t0 is not None:
            self._profile_add('batch_expand_history_time', time.perf_counter() - profile_t0)

    @staticmethod
    def _is_encoded_history_entry(entry):
        return (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], np.ndarray)
            and isinstance(entry[1], np.ndarray)
        )

    def _encode_history_entry(self, board_or_fen, repetition_count=None):
        """
        Convert history item to cached tensors for both POVs:
        (white_pov_tensor, black_pov_tensor).
        """
        if self._is_encoded_history_entry(board_or_fen):
            return board_or_fen
        if isinstance(board_or_fen, str):
            board_obj = chess.board_from_fen(board_or_fen)
        else:
            board_obj = board_or_fen
        if not isinstance(board_obj, chess.Board):
            raise TypeError(f"Unsupported history entry type: {type(board_or_fen)}")

        return self._history_tensor_pair_for_board(
            board_obj,
            repetition_count=repetition_count,
        )

    def _build_history_tensor(
        self,
        current_board,
        board_history,
        current_tensor=None,
        repetition_count=None,
    ):
        """
        Build tensor with history using POV-aware board_to_tensor.

        Args:
            current_board: chess.Board for current position
            board_history: list of previous boards (from real game history)
        """
        if self.history_positions == 0:
            if current_tensor is not None:
                return current_tensor
            return self._board_to_tensor_profiled(
                current_board,
                repetition_count=repetition_count,
            )

        history_tensors = []
        use_black_pov = current_board.turn == chess.BLACK
        if board_history:
            history_boards = board_history[-self.history_positions:]
            for hist_entry in history_boards:
                encoded = self._encode_history_entry(hist_entry)
                hist_tensor = encoded[1] if use_black_pov else encoded[0]
                history_tensors.append(self._history_frame_for_model(hist_tensor))

        pad_count = max(0, self.history_positions - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        if current_tensor is None:
            current_tensor = self._board_to_tensor_profiled(
                current_board,
                repetition_count=repetition_count,
            )
        elif self._path_dependent_repetition and not self._encode_exact_repetition:
            current_tensor = np.array(current_tensor, copy=True)
            current_tensor[12].fill(-1.0)
        history_tensors.append(current_tensor)

        if self._path_dependent_repetition:
            frames = [np.asarray(item) for item in history_tensors[:-1]]
            frames = [self._empty_history_tensor] * (
                self.history_positions - len(frames)
            ) + frames
            return np.concatenate(
                frames + [current_tensor[:13], current_tensor[13:20]],
                axis=0,
            )

        return np.concatenate(history_tensors, axis=0)

    def _history_tensor_for_entry(
        self,
        entry,
        use_black_pov,
        repetition_count=None,
    ):
        if self._is_encoded_history_entry(entry):
            if any(
                tensor.shape != (self._current_planes, 8, 8)
                for tensor in entry
            ):
                raise ValueError(
                    "Cached self-play history tensor does not match the active "
                    f"encoder {self.input_encoder_id!r} ({self._current_planes} planes)"
                )
            tensor = entry[1] if use_black_pov else entry[0]
            return self._history_frame_for_model(tensor)

        if isinstance(entry, MCTSNode):
            encoded = self._history_tensor_pair_for_node(
                entry,
                repetition_count=repetition_count,
            )
            tensor = encoded[1] if use_black_pov else encoded[0]
            return self._history_frame_for_model(tensor)

        if isinstance(entry, chess.Board):
            tensor = self._board_to_tensor_profiled(
                entry,
                flip_perspective=use_black_pov,
                storage_dtype=self.history_storage_dtype,
                repetition_count=repetition_count,
            )
            return self._history_frame_for_model(tensor)

        encoded = self._encode_history_entry(
            entry,
            repetition_count=repetition_count,
        )
        tensor = encoded[1] if use_black_pov else encoded[0]
        return self._history_frame_for_model(tensor)

    def _history_frame_for_model(self, tensor):
        """Return one history frame under the checkpoint's repetition contract."""
        if not self._path_dependent_repetition:
            return tensor
        frame = tensor[:self._history_frame_planes]
        if not self._encode_exact_repetition and not np.all(frame[12] == -1.0):
            # Encoded history can arrive from callers that know repetition
            # counts. Do not leak that feature into checkpoints trained with
            # unknown repetition planes, and do not mutate the caller's cache.
            frame = np.array(frame, copy=True)
            frame[12].fill(-1.0)
        return frame

    def _collect_node_history_entries(self, node, board_history):
        """
        Return the last history positions for a search leaf.

        board_history contains real-game positions before the current root.
        For deeper MCTS leaves, parent nodes represent the simulated positions
        immediately before the leaf and must be part of the NN history planes.
        """
        if self.history_positions <= 0:
            return []

        simulated_history = []
        ancestor = node.parent
        while ancestor is not None and len(simulated_history) < self.history_positions:
            simulated_history.append(ancestor)
            ancestor = ancestor.parent
        if simulated_history:
            simulated_history.reverse()

        keep_from_game = max(0, int(self.history_positions) - len(simulated_history))
        real_history = list(board_history[-keep_from_game:]) if board_history and keep_from_game > 0 else []
        return real_history + simulated_history

    def _fill_history_prefix_for_node(self, node, board_history, destination):
        """Write history directly into the inference scratch without a temporary."""
        if self.history_positions <= 0:
            return
        destination.fill(0.0)
        if self._path_dependent_repetition:
            destination.reshape(
                int(self.history_positions),
                self._history_frame_planes,
                8,
                8,
            )[:, 12].fill(-1.0)
        use_black_pov = (node.board.turn == chess.BLACK)
        history_entries = self._collect_node_history_entries(node, board_history)
        first_slot = max(0, int(self.history_positions) - len(history_entries))
        for slot, entry in enumerate(history_entries, start=first_slot):
            begin = slot * self._board_planes
            end = begin + self._board_planes
            destination[begin:end] = self._history_tensor_for_entry(
                entry,
                use_black_pov,
            )

    def _current_tensor_for_node(self, node, out=None, repetition_count=None):
        if not self.cache_node_tensors:
            return self._board_to_tensor_profiled(
                node.board,
                storage_dtype=self.input_storage_dtype,
                out=out,
                repetition_count=repetition_count,
            )

        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = self._board_to_tensor_profiled(
                node.board,
                storage_dtype=self.input_storage_dtype,
                repetition_count=repetition_count,
            )
            node._board_tensor = cached
        if out is not None:
            np.copyto(out, cached)
            return out
        return cached

    def _get_native_board_plane_scratch(self, batch_size):
        batch_size = int(batch_size)
        scratch = self._native_board_plane_scratch.get(batch_size)
        if scratch is None:
            scratch = (
                np.empty((batch_size, 12), dtype=np.uint64),
                np.empty(batch_size, dtype=np.uint8),
                np.empty(batch_size, dtype=np.uint64),
                np.empty(batch_size, dtype=np.uint64),
                np.empty(batch_size, dtype=np.float32),
                np.empty(batch_size, dtype=np.float32),
            )
            self._native_board_plane_scratch[batch_size] = scratch
        return scratch

    def _pack_current_tensors_native(self, nodes, destination):
        (
            piece_masks,
            black_to_move,
            castling_masks,
            en_passant_masks,
            halfmove_values,
            fullmove_values,
        ) = self._get_native_board_plane_scratch(len(nodes))
        for row_idx, node in enumerate(nodes):
            board = node.board
            for plane_idx, (color, piece_type) in enumerate(_INPUT_PIECE_SPECS):
                piece_masks[row_idx, plane_idx] = int(board[color, piece_type])
            is_black = board.turn == chess.BLACK
            black_to_move[row_idx] = int(is_black)
            pov_color = chess.BLACK if is_black else chess.WHITE
            castling_mask = 0
            kingside = chess.has_kingside_castling_rights(board, pov_color)
            queenside = chess.has_queenside_castling_rights(board, pov_color)
            if kingside or queenside:
                king_square = chess.king_square(board, pov_color)
                if king_square is not None:
                    king_index = chess.square_index(king_square)
                    castling_mask |= 1 << king_index
                    back_rank_base = (king_index >> 3) << 3
                    if kingside:
                        castling_mask |= 1 << (back_rank_base + 7)
                    if queenside:
                        castling_mask |= 1 << back_rank_base
            castling_masks[row_idx] = castling_mask
            ep_square = board.en_passant_square
            en_passant_masks[row_idx] = (
                0 if ep_square is None else 1 << chess.square_index(ep_square)
            )
            halfmove_values[row_idx] = min(float(board.halfmove_clock) / 50.0, 1.0)
            fullmove_values[row_idx] = min(float(board.fullmove_number) / 100.0, 1.0)
        self._native_mcts.encode_board_planes(
            piece_masks,
            black_to_move,
            castling_masks,
            en_passant_masks,
            halfmove_values,
            fullmove_values,
            destination,
        )

    def _encode_legal_indices_native(self, nodes, legal_moves_per_node):
        result = [None] * len(nodes)
        uncached_rows = [
            row_idx
            for row_idx, node in enumerate(nodes)
            if node._legal_indices is None
        ]
        for row_idx, node in enumerate(nodes):
            if node._legal_indices is not None:
                result[row_idx] = node._legal_indices
        if not uncached_rows:
            return result

        move_counts = [
            len(legal_moves_per_node[row_idx])
            for row_idx in uncached_rows
        ]
        node_offsets = np.empty(len(uncached_rows) + 1, dtype=np.int32)
        node_offsets[0] = 0
        np.cumsum(
            np.asarray(move_counts, dtype=np.int32),
            dtype=np.int32,
            out=node_offsets[1:],
        )
        total_moves = int(node_offsets[-1])
        move_hashes = np.fromiter(
            (
                hash(move)
                for row_idx in uncached_rows
                for move in legal_moves_per_node[row_idx]
            ),
            dtype=np.int64,
            count=total_moves,
        )
        black_to_move = np.fromiter(
            (
                int(nodes[row_idx].board.turn == chess.BLACK)
                for row_idx in uncached_rows
            ),
            dtype=np.uint8,
            count=len(uncached_rows),
        )
        encoded = np.empty(total_moves, dtype=np.int32)
        self._native_mcts.encode_move_hashes(
            move_hashes,
            node_offsets,
            black_to_move,
            encoded,
        )
        for local_idx, row_idx in enumerate(uncached_rows):
            begin = int(node_offsets[local_idx])
            end = int(node_offsets[local_idx + 1])
            node = nodes[row_idx]
            node._legal_indices = encoded[begin:end]
            result[row_idx] = node._legal_indices
        return result

    def _get_legal_index_scratch(self, batch_size, max_legal_count, dtype=np.int64):
        dtype = np.dtype(dtype)
        key = (int(batch_size), int(max_legal_count), dtype.str)
        scratch = self._legal_index_scratch.get(key)
        if scratch is None:
            scratch = np.empty((batch_size, max_legal_count), dtype=dtype)
            self._legal_index_scratch[key] = scratch
        return scratch

    def _get_legal_index_source_tensor(self, legal_index_matrix):
        if not self._use_pinned_staging:
            return torch.from_numpy(legal_index_matrix)

        key = tuple(int(dim) for dim in legal_index_matrix.shape)
        scratch = self._legal_index_tensor_scratch.get(key)
        if scratch is None:
            scratch = torch.empty(key, dtype=torch.long, pin_memory=True)
            self._legal_index_tensor_scratch[key] = scratch
        scratch.copy_(torch.from_numpy(legal_index_matrix))
        return scratch

    def _get_board_input_scratch(self, batch_size):
        key = int(batch_size)
        scratch = self._board_input_scratch.get(key)
        if scratch is None:
            if self._use_pinned_staging:
                torch_dtype = (
                    torch.float16
                    if self.input_storage_dtype == np.float16
                    else torch.float32
                )
                tensor_scratch = torch.empty(
                    (batch_size, self._input_planes, 8, 8),
                    dtype=torch_dtype,
                    pin_memory=True,
                )
                self._board_input_tensor_scratch[key] = tensor_scratch
                scratch = tensor_scratch.numpy()
            else:
                scratch = np.empty(
                    (batch_size, self._input_planes, 8, 8),
                    dtype=self.input_storage_dtype,
                )
            self._board_input_scratch[key] = scratch
        return scratch

    def _get_board_input_source_tensor(self, boards_np):
        if not self._use_pinned_staging:
            return torch.from_numpy(boards_np)

        key = int(boards_np.shape[0])
        tensor_scratch = self._board_input_tensor_scratch.get(key)
        if tensor_scratch is None:
            raise RuntimeError(f"Missing pinned board input scratch for batch size {key}")
        return tensor_scratch

    @staticmethod
    def _captured_piece_value(board, move):
        if chess.is_en_passant(board, move):
            return _PIECE_VALUES[chess.PAWN]
        captured_piece = chess.piece_at(board, move.destination)
        if captured_piece is None:
            return 0.0
        return float(_PIECE_VALUES.get(captured_piece.piece_type, 0.0))

    @staticmethod
    def _moving_piece_value(board, move):
        moving_piece = chess.piece_at(board, move.origin)
        if moving_piece is None:
            return 0.0
        return float(_PIECE_VALUES.get(moving_piece.piece_type, 0.0))

    @staticmethod
    def _move_may_give_check_fast(board, move):
        """Cheap geometric prefilter before applying a move to test check."""
        moved_piece = chess.piece_at(board, move.origin)
        if moved_piece is None:
            return False

        king_square = chess.king_square(board, board.turn.opposite)
        if king_square is None:
            return False

        target_square = move.destination
        piece_type = move.promotion or moved_piece.piece_type

        from_row = chess.square_rank(target_square)
        from_col = chess.square_file(target_square)
        king_row = chess.square_rank(king_square)
        king_col = chess.square_file(king_square)
        dr = king_row - from_row
        dc = king_col - from_col
        abs_dr = abs(dr)
        abs_dc = abs(dc)

        if piece_type == chess.KNIGHT:
            return (abs_dr, abs_dc) in {(1, 2), (2, 1)}
        if piece_type == chess.BISHOP:
            return abs_dr == abs_dc
        if piece_type == chess.ROOK:
            return dr == 0 or dc == 0
        if piece_type == chess.QUEEN:
            return dr == 0 or dc == 0 or abs_dr == abs_dc
        if piece_type == chess.KING:
            return max(abs_dr, abs_dc) == 1
        if piece_type == chess.PAWN:
            if moved_piece.color == chess.WHITE:
                return dr == 1 and abs_dc == 1
            return dr == -1 and abs_dc == 1
        return False

    def _tactical_prior_multipliers(self, board, legal_moves):
        if not self.tactical_priors_enabled or not legal_moves:
            return np.ones((len(legal_moves),), dtype=np.float32)

        if not (
            self._tactical_capture_enabled
            or self._tactical_promotion_enabled
            or self._tactical_check_enabled
            or self._tactical_recapture_enabled
        ):
            return np.ones((len(legal_moves),), dtype=np.float32)

        multipliers = np.ones((len(legal_moves),), dtype=np.float32)
        last_move = chess.last_move(board) if self._tactical_recapture_enabled else None

        for idx, move in enumerate(legal_moves):
            bonus = 0.0
            if self._tactical_capture_enabled and chess.is_capture(board, move):
                bonus += self.tactical_capture_bonus
                gain = self._captured_piece_value(board, move) - self._moving_piece_value(board, move)
                if gain > 0.0:
                    bonus += self.tactical_winning_capture_bonus * min(1.0, gain / 4.0)
            if self._tactical_promotion_enabled and move.promotion is not None:
                bonus += self.tactical_promotion_bonus

            should_eval_check = (
                self._tactical_check_enabled
                and (
                    bonus > 0.0
                    or (
                        not self._tactical_capture_enabled
                        and not self._tactical_promotion_enabled
                        and not self._tactical_recapture_enabled
                    )
                )
            )
            if should_eval_check:
                try:
                    if self._move_may_give_check_fast(board, move) and chess.gives_check(board, move):
                        bonus += self.tactical_check_bonus
                except Exception:
                    pass
            if (
                self._tactical_recapture_enabled
                and last_move is not None
                and move.destination == last_move.destination
            ):
                bonus += self.tactical_recapture_bonus
            multipliers[idx] = 1.0 + float(bonus)

        return multipliers

    def _gumbel_raw_completed_qvalues(self, node):
        """Return completed Q in the root player's value scale."""
        edges = node.edges
        child_count = int(edges.visit_counts.size)
        if child_count <= 0:
            return np.empty(0, dtype=np.float64)

        raw_value = node.raw_value
        if raw_value is None:
            node_visits = int(getattr(node, 'visit_count', 0) or 0)
            raw_value = float(node.value_sum / node_visits) if node_visits > 0 else 0.0
        raw_value = float(max(-1.0, min(1.0, raw_value)))

        visits = edges.visit_counts.astype(np.float64, copy=False)
        priors = edges.base_priors.astype(np.float64, copy=False)
        qvalues = np.zeros(child_count, dtype=np.float64)
        visited = visits > 0.0
        if visited.any():
            # Child values use the child side-to-move perspective.
            qvalues[visited] = -edges.value_sums[visited].astype(np.float64) / visits[visited]

        completion_value = raw_value
        sum_visits = float(visits.sum())
        if self.gumbel_use_mixed_value and visited.any() and sum_visits > 0.0:
            safe_priors = np.maximum(priors, np.finfo(np.float64).tiny)
            visited_prior_sum = float(safe_priors[visited].sum())
            if visited_prior_sum > 0.0:
                weighted_q = float(
                    np.sum(safe_priors[visited] * qvalues[visited]) / visited_prior_sum
                )
                completion_value = (raw_value + sum_visits * weighted_q) / (sum_visits + 1.0)

        return np.where(visited, qvalues, completion_value)

    def _gumbel_normalized_completed_qvalues(self, node):
        """Return the cached, normalized completed-Q vector for one node.

        Real visits/value sums only change after an inference batch returns.
        Virtual selections inside that batch affect the scalar Gumbel scale,
        but not this comparatively expensive completion and normalization.
        """
        edges = node.edges
        if (
            edges.completed_q_cache_version == edges.stats_version
            and edges.completed_q_cache.size == edges.base_priors.size
        ):
            return edges.completed_q_cache

        child_count = int(edges.visit_counts.size)
        if child_count <= 0:
            return np.empty(0, dtype=np.float32)

        raw_value = node.raw_value
        if raw_value is None:
            node_visits = int(getattr(node, 'visit_count', 0) or 0)
            raw_value = float(node.value_sum / node_visits) if node_visits > 0 else 0.0
        raw_value = float(max(-1.0, min(1.0, raw_value)))

        native = self._native_mcts
        if native is not None:
            completed = edges.completed_q_cache
            if completed.size != child_count:
                completed = np.empty(child_count, dtype=np.float64)
                edges.completed_q_cache = completed
            native.completed_q(
                edges.visit_counts,
                edges.value_sums,
                edges.base_priors,
                raw_value,
                self.gumbel_use_mixed_value,
                self.gumbel_q_range_floor,
                completed,
            )
            edges.completed_q_cache_version = edges.stats_version
            self._native_completed_q_calls += 1
            return completed

        self._python_completed_q_calls += 1
        completed = self._gumbel_raw_completed_qvalues(node)
        q_min = float(np.min(completed))
        q_max = float(np.max(completed))
        q_range = q_max - q_min
        if q_range > 1e-8:
            # Full min-max scaling turns an arbitrarily small sibling-Q spread
            # into a maximum-strength preference. That is especially harmful
            # while the scalar value head is under-dispersed. Preserve the
            # ordering, but keep the magnitude meaningful.
            completed = (completed - q_min) / max(q_range, self.gumbel_q_range_floor)
        else:
            completed.fill(0.0)
        edges.completed_q_cache = completed
        edges.completed_q_cache_version = edges.stats_version
        return edges.completed_q_cache

    def _gumbel_completed_qvalues(self, node, scale_visit_counts=None):
        """Return MCTX-style completed Q scaled for the current visit phase."""
        edges = node.edges
        edges.ensure_gumbel_workspace()
        completed = self._gumbel_normalized_completed_qvalues(node)
        scale_visits = (
            edges.visit_counts
            if scale_visit_counts is None
            else np.asarray(scale_visit_counts, dtype=np.float64)
        )
        max_visit = float(np.max(scale_visits)) if scale_visits.size else 0.0
        scale = (self.gumbel_c_visit + max_visit) * self.gumbel_c_scale
        scaled = edges.q_score_scratch
        np.multiply(completed, scale, out=scaled)
        return scaled

    def _gumbel_improved_policy(self, node, scale_visit_counts=None):
        """Policy target softmax(log prior + completed Q), without root Gumbel."""
        edges = node.edges
        edges.ensure_gumbel_workspace()
        use_cache = scale_visit_counts is None
        if (
            use_cache
            and edges.improved_policy_cache_version == edges.stats_version
            and edges.improved_policy_cache.size == edges.base_priors.size
        ):
            return edges.improved_policy_cache

        native = self._native_mcts
        if native is not None:
            completed = self._gumbel_normalized_completed_qvalues(node)
            if scale_visit_counts is None:
                max_visit = float(np.max(edges.visit_counts)) if edges.visit_counts.size else 0.0
            else:
                scale_visits = np.asarray(scale_visit_counts, dtype=np.float64)
                max_visit = float(np.max(scale_visits)) if scale_visits.size else 0.0
            if use_cache:
                probs = edges.improved_policy_cache
                if probs.size != edges.base_priors.size:
                    probs = np.empty(edges.base_priors.size, dtype=np.float32)
                    edges.improved_policy_cache = probs
            else:
                probs = np.empty(edges.base_priors.size, dtype=np.float32)
            native.improved_policy(
                edges.log_base_priors,
                edges.base_priors,
                completed,
                max_visit,
                self.gumbel_c_visit,
                self.gumbel_c_scale,
                probs,
            )
            if use_cache:
                edges.improved_policy_cache_version = edges.stats_version
            return probs

        priors = edges.base_priors.astype(np.float64, copy=False)
        logits = edges.log_base_priors.copy()
        logits += self._gumbel_completed_qvalues(
            node,
            scale_visit_counts=scale_visit_counts,
        ).astype(np.float64, copy=False)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            prior_total = float(priors.sum())
            if prior_total > 0.0:
                probs = priors / prior_total
            else:
                probs = np.full(priors.size, 1.0 / max(1, priors.size), dtype=np.float64)
        else:
            probs /= total
        probs = probs.astype(np.float32, copy=False)
        if use_cache:
            edges.improved_policy_cache = probs
            edges.improved_policy_cache_version = edges.stats_version
            return edges.improved_policy_cache
        return probs

    def _gumbel_root_q_scale_visit_counts(self, node, state, *, include_virtual=False):
        """Return root Q-scale counts, leaving fresh search accounting intact.

        ``fresh`` preserves the existing root behavior. ``total`` lets valid
        inherited edge evidence set Q confidence while Sequential Halving and
        visit credit continue to use only fresh visits.
        """
        edges = node.edges
        edges.ensure_gumbel_workspace()
        if self.gumbel_reuse_q_scale == 'total':
            return edges.total_counts if include_virtual else edges.visit_counts

        source = edges.total_counts if include_virtual else edges.visit_counts
        initial = state.get('_initial_visits_f64')
        if initial is None:
            initial = np.asarray(state['initial_visits'], dtype=np.float64)
            state['_initial_visits_f64'] = initial
        scratch = edges.fresh_count_scratch
        np.subtract(source, initial, out=scratch)
        np.maximum(scratch, 0.0, out=scratch)
        return scratch

    def _gumbel_policy_target(self, improved_policy):
        """Temperature-scale only the training target, not action selection."""
        probs = np.asarray(improved_policy)
        if probs.size <= 1 or abs(self.gumbel_target_temperature - 1.0) <= 1e-8:
            return probs.astype(np.float32, copy=False)
        native = self._native_mcts
        if native is not None:
            native_probs = np.asarray(probs, dtype=np.float32)
            softened = np.empty(probs.size, dtype=np.float32)
            native.soften_policy(native_probs, self.gumbel_target_temperature, softened)
            return softened
        probs = probs.astype(np.float64, copy=False)
        logits = np.log(np.maximum(probs, np.finfo(np.float64).tiny))
        logits /= self.gumbel_target_temperature
        logits -= float(np.max(logits))
        softened = np.exp(logits)
        total = float(softened.sum())
        if total <= 0.0 or not np.isfinite(total):
            return probs.astype(np.float32, copy=False)
        softened /= total
        return softened.astype(np.float32, copy=False)

    @staticmethod
    def _apply_edge_virtual_loss(edges, edge_index):
        """Apply one virtual visit without a child-property round trip."""
        virtual_loss = int(edges.virtual_losses[edge_index]) + 1
        edges.virtual_losses[edge_index] = virtual_loss
        edges.total_counts[edge_index] = float(
            int(edges.visit_counts[edge_index]) + virtual_loss
        )
        edges.total_count_sum += 1.0

    def _select_child_gumbel(self, node, apply_virtual_loss=False):
        """Full-Gumbel deterministic interior action selection."""
        edges = node.edges
        if not edges.moves:
            return None
        if len(edges.moves) == 1:
            selected_idx = 0
        else:
            probs = self._gumbel_improved_policy(node)
            counts = edges.total_counts
            scores = edges.selection_score_scratch
            np.divide(counts, 1.0 + float(counts.sum()), out=scores)
            np.subtract(probs, scores, out=scores)
            selected_idx = int(np.argmax(scores))
        if apply_virtual_loss:
            self._apply_edge_virtual_loss(edges, selected_idx)
        child = edges.nodes[selected_idx]
        if child is None:
            child = MCTSNode(
                board=None,
                parent=node,
                move=edges.moves[selected_idx],
                copy_board=False,
            )
            child.parent_edge_index = selected_idx
            edges.nodes[selected_idx] = child
        return child

    def _select_root_gumbel(self, node, state, apply_virtual_loss=False):
        """Select a root child using Gumbel Top-k and Sequential Halving."""
        edges = node.edges
        if not edges.moves:
            return None
        if len(edges.moves) == 1:
            selected_idx = 0
            if apply_virtual_loss:
                self._apply_edge_virtual_loss(edges, selected_idx)
            child = edges.nodes[selected_idx]
            if child is None:
                child = MCTSNode(
                    board=None,
                    parent=node,
                    move=edges.moves[selected_idx],
                    copy_board=False,
                )
                child.parent_edge_index = selected_idx
                edges.nodes[selected_idx] = child
            return child

        edges.ensure_gumbel_workspace()

        initial = state.get('_initial_visits_f64')
        if initial is None:
            initial = np.asarray(state['initial_visits'], dtype=np.float64)
            state['_initial_visits_f64'] = initial
        pass_initial = state.get('_pass_initial_visits_f64')
        if pass_initial is None:
            pass_initial = np.asarray(
                state.get('pass_initial_visits', state['initial_visits']),
                dtype=np.float64,
            )
            state['_pass_initial_visits_f64'] = pass_initial
            state['_pass_initial_total'] = float(pass_initial.sum())
        sequence = state['sequence']
        simulation_index = int(
            round(edges.total_count_sum - state['_pass_initial_total'])
        )
        considered_visit = sequence[min(simulation_index, len(sequence) - 1)] if sequence else 0

        native = self._native_mcts
        if native is not None:
            gumbel = state.get('_gumbel_f64')
            if gumbel is None:
                gumbel = np.asarray(state['gumbel'], dtype=np.float64)
                state['_gumbel_f64'] = gumbel
            selected_idx = native.select_root(
                edges.log_base_priors,
                gumbel,
                self._gumbel_normalized_completed_qvalues(node),
                initial,
                pass_initial,
                edges.visit_counts,
                edges.total_counts,
                edges.virtual_losses,
                considered_visit,
                self.gumbel_c_visit,
                self.gumbel_c_scale,
                apply_virtual_loss,
                self.gumbel_reuse_q_scale == 'total',
            )
            if apply_virtual_loss:
                edges.total_count_sum += 1.0
            self._native_root_selection_calls += 1
            child = edges.nodes[selected_idx]
            if child is None:
                child = MCTSNode(
                    board=None,
                    parent=node,
                    move=edges.moves[selected_idx],
                    copy_board=False,
                )
                child.parent_edge_index = selected_idx
                edges.nodes[selected_idx] = child
            return child

        cumulative_fresh = edges.fresh_count_scratch
        np.subtract(edges.total_counts, initial, out=cumulative_fresh)
        np.maximum(cumulative_fresh, 0.0, out=cumulative_fresh)
        pass_fresh = state.get('_pass_fresh_scratch')
        if pass_fresh is None or pass_fresh.size != len(edges.moves):
            pass_fresh = np.empty(len(edges.moves), dtype=np.float64)
            state['_pass_fresh_scratch'] = pass_fresh
        np.subtract(edges.total_counts, pass_initial, out=pass_fresh)
        np.maximum(pass_fresh, 0.0, out=pass_fresh)
        scores = edges.selection_score_scratch
        gumbel = state.get('_gumbel_f64')
        if gumbel is None:
            gumbel = np.asarray(state['gumbel'], dtype=np.float64)
            state['_gumbel_f64'] = gumbel
        base_scores = state.get('_base_scores')
        if base_scores is None:
            base_scores = edges.log_base_priors.copy()
            base_scores -= float(np.max(base_scores))
            base_scores += gumbel
            state['_base_scores'] = base_scores
        np.copyto(scores, base_scores)
        q_scale_counts = (
            edges.total_counts
            if self.gumbel_reuse_q_scale == 'total'
            else cumulative_fresh
        )
        max_q_scale_visits = (
            float(np.max(q_scale_counts)) if q_scale_counts.size else 0.0
        )
        if (
            state.get('_q_score_version') != edges.stats_version
            or state.get('_q_score_max_visit') != max_q_scale_visits
        ):
            scale = (self.gumbel_c_visit + max_q_scale_visits) * self.gumbel_c_scale
            np.multiply(
                self._gumbel_normalized_completed_qvalues(node),
                scale,
                out=edges.q_score_scratch,
            )
            state['_q_score_version'] = edges.stats_version
            state['_q_score_max_visit'] = max_q_scale_visits
        np.add(scores, edges.q_score_scratch, out=scores)
        eligible = edges.eligibility_scratch
        np.equal(pass_fresh, float(considered_visit), out=eligible)
        if not eligible.any():
            # Defensive fallback for externally supplied/reused roots with
            # inconsistent counters. Prefer the least freshly visited actions.
            np.equal(pass_fresh, float(np.min(pass_fresh)), out=eligible)
        scores[~eligible] = -np.inf
        selected_idx = int(np.argmax(scores))
        self._python_root_selection_calls += 1
        if apply_virtual_loss:
            self._apply_edge_virtual_loss(edges, selected_idx)
        child = edges.nodes[selected_idx]
        if child is None:
            child = MCTSNode(
                board=None,
                parent=node,
                move=edges.moves[selected_idx],
                copy_board=False,
            )
            child.parent_edge_index = selected_idx
            edges.nodes[selected_idx] = child
        return child

    def _gumbel_final_action_index(self, node, state):
        edges = node.edges
        if not edges.moves:
            return None
        edges.ensure_gumbel_workspace()
        initial = np.asarray(state['initial_visits'], dtype=np.float64)
        pass_initial = np.asarray(
            state.get('pass_initial_visits', state['initial_visits']),
            dtype=np.float64,
        )
        revisiting = state.get('root_schedule') == 'revisiting'
        native = self._native_mcts
        if native is not None:
            gumbel = state.get('_gumbel_f64')
            if gumbel is None:
                gumbel = np.asarray(state['gumbel'], dtype=np.float64)
                state['_gumbel_f64'] = gumbel
            completed_q = self._gumbel_normalized_completed_qvalues(node)
            if revisiting:
                final_visit_counts = _gumbel_final_visit_counts(node, state)
                _, final_visit_threshold = _gumbel_final_action_eligibility(
                    final_visit_counts
                )
                return native.final_action_eligible(
                    edges.log_base_priors,
                    gumbel,
                    completed_q,
                    initial,
                    pass_initial,
                    edges.visit_counts,
                    final_visit_threshold,
                    self.gumbel_c_visit,
                    self.gumbel_c_scale,
                    self.gumbel_reuse_q_scale == 'total',
                )
            return native.final_action(
                edges.log_base_priors,
                gumbel,
                completed_q,
                initial,
                edges.visit_counts,
                self.gumbel_c_visit,
                self.gumbel_c_scale,
                self.gumbel_reuse_q_scale == 'total',
            )

        final_visit_counts = _gumbel_final_visit_counts(node, state)
        eligible, _ = _gumbel_final_action_eligibility(final_visit_counts)
        scores = edges.selection_score_scratch
        np.copyto(scores, edges.log_base_priors)
        scores -= float(np.max(scores))
        np.add(scores, np.asarray(state['gumbel'], dtype=np.float64), out=scores)
        np.add(
            scores,
            self._gumbel_completed_qvalues(
                node,
                scale_visit_counts=self._gumbel_root_q_scale_visit_counts(
                    node,
                    state,
                ),
            ),
            out=scores,
        )
        scores[~eligible] = -np.inf
        return int(np.argmax(scores))

    def _backpropagate_and_remove_virtual_loss(self, search_path, value):
        """Commit visits and release virtual loss in one reverse traversal."""
        for node in reversed(search_path):
            parent = node.parent
            edge_index = node.parent_edge_index
            if parent is not None and edge_index >= 0:
                edges = parent.edges
                # Keep the old property-path rounding/order exactly: value_sum
                # was read as a Python float before assignment to float32.
                edges.value_sums[edge_index] = float(edges.value_sums[edge_index]) + value
                visit_count = int(edges.visit_counts[edge_index]) + 1
                edges.visit_counts[edge_index] = visit_count
                virtual_loss = int(edges.virtual_losses[edge_index]) - 1
                edges.virtual_losses[edge_index] = virtual_loss
                edges.total_counts[edge_index] = float(visit_count + virtual_loss)
                # Replacing one virtual visit with one real visit leaves the
                # cached total unchanged.
                edges.mark_search_stats_changed()
            else:
                node._root_value_sum = float(node._root_value_sum) + value
                node._root_visit_count = int(node._root_visit_count) + 1
                node._root_virtual_loss -= 1
            value = -value

    @staticmethod
    def _fresh_child_visit_dict(root, initial_child_visits):
        if root is None or not root.expanded or root.edges is None:
            return {}
        after = root.edges.visit_counts.astype(np.int64, copy=False)
        if initial_child_visits is None or len(initial_child_visits) != len(after):
            before = np.zeros_like(after)
        else:
            before = np.asarray(initial_child_visits, dtype=np.int64)
        fresh = np.maximum(0, after - before)
        return {
            move: int(count)
            for move, count in zip(root.edges.moves, fresh)
            if int(count) > 0
        }

    def _summarize_root_search(
        self,
        root,
        simulation_budget,
        initial_root_visits=0,
        initial_child_visits=None,
        inherited_visit_credit=0,
    ):
        budget = max(1, int(simulation_budget))
        total_root_visits = 0 if root is None else int(getattr(root, 'visit_count', 0) or 0)
        fresh_used = max(0, total_root_visits - max(0, int(initial_root_visits)))
        inherited_visit_credit = max(0, int(inherited_visit_credit))
        effective_used = fresh_used + inherited_visit_credit
        summary = {
            'simulations_used': effective_used,
            'fresh_simulations_used': fresh_used,
            'inherited_visit_credit': inherited_visit_credit,
            'simulation_budget': budget,
            'top_visit_prob': 0.0,
            'visit_gap': 0.0,
            'visit_entropy': 1.0,
            'prior_mcts_agree': None,
            'prior_top_visit_prob': None,
            'prior_top_visit_rank': None,
            'mcts_top_prior_prob': None,
            'mcts_q_delta': None,
            'mcts_changed_to_lower_q': None,
            'mcts_policy_kl': None,
            'explored_prior_mass': 0.0,
            'unexplored_prior_mass': 1.0,
            'visited_move_count': 0,
            'legal_move_count': 0,
            'root_value': 0.0,
            'best_q': None,
            'played_q': None,
            'orig_q': None,
            'policy_kld': None,
            'policy_target_intervention_applied': False,
            'policy_target_removed_prior_mass': 0.0,
            'search_visits': int(fresh_used),
            'stopped_early': effective_used < budget,
            'policy_weight': float(max(0.0, min(1.0, effective_used / float(budget)))),
        }
        if root is None or not root.expanded or root.edges is None:
            return summary

        cumulative_visits = root.edges.visit_counts.astype(np.float32, copy=False)
        if initial_child_visits is not None and len(initial_child_visits) == len(cumulative_visits):
            all_visits = np.maximum(
                0.0,
                cumulative_visits - np.asarray(initial_child_visits, dtype=np.float32),
            )
        else:
            all_visits = cumulative_visits
        priors = root.edges.base_priors.astype(np.float32, copy=False)
        legal_count = int(all_visits.size)
        visited_mask = all_visits > 0.0
        visited_count = int(visited_mask.sum())
        prior_total = float(priors.sum()) if priors.size == all_visits.size else 0.0
        explored_prior_mass = (
            float(priors[visited_mask].sum()) / prior_total
            if prior_total > 0.0 and priors.size == all_visits.size
            else 0.0
        )
        explored_prior_mass = float(max(0.0, min(1.0, explored_prior_mass)))
        summary['legal_move_count'] = legal_count
        summary['visited_move_count'] = visited_count
        summary['explored_prior_mass'] = explored_prior_mass
        summary['unexplored_prior_mass'] = float(max(0.0, 1.0 - explored_prior_mass))

        visits = all_visits[visited_mask]
        total = float(visits.sum())
        if total <= 0.0:
            return summary

        root_value = 0.0
        root_visits = int(getattr(root, 'visit_count', 0) or 0)
        if root_visits > 0:
            root_value = float(root.value_sum / max(1, root_visits))
            root_value = float(max(-1.0, min(1.0, root_value)))

        summary['root_value'] = root_value
        if root.raw_value is not None:
            summary['orig_q'] = float(max(-1.0, min(1.0, root.raw_value)))

        visits.sort()
        top = float(visits[-1])
        second = float(visits[-2]) if visits.size > 1 else 0.0
        probs = visits / total
        entropy = 0.0
        if probs.size > 1:
            entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
            entropy /= float(np.log(probs.size))

        summary['top_visit_prob'] = top / total
        summary['visit_gap'] = max(0.0, (top - second) / total)
        summary['visit_entropy'] = float(max(0.0, min(1.0, entropy)))

        if all_visits.size > 0 and priors.size == all_visits.size and prior_total > 0.0:
            prior_probs = priors / prior_total
            visit_probs = all_visits / max(1e-8, float(all_visits.sum()))
            prior_top_idx = int(np.argmax(prior_probs))
            mcts_top_idx = int(np.argmax(all_visits))
            summary['prior_mcts_agree'] = 1.0 if prior_top_idx == mcts_top_idx else 0.0
            summary['prior_top_visit_prob'] = float(visit_probs[prior_top_idx])
            summary['mcts_top_prior_prob'] = float(prior_probs[mcts_top_idx])
            rank_order = np.argsort(-all_visits)
            rank_matches = np.where(rank_order == prior_top_idx)[0]
            if rank_matches.size > 0:
                summary['prior_top_visit_rank'] = int(rank_matches[0]) + 1
            active = all_visits > 0.0
            if active.any():
                kl_terms = visit_probs[active] * (
                    np.log(np.clip(visit_probs[active], 1e-12, 1.0))
                    - np.log(np.clip(prior_probs[active], 1e-12, 1.0))
                )
                summary['mcts_policy_kl'] = float(max(0.0, float(kl_terms.sum())))
                summary['policy_kld'] = summary['mcts_policy_kl']
            if all_visits[prior_top_idx] > 0.0 and all_visits[mcts_top_idx] > 0.0:
                prior_q = -float(root.edges.value_sums[prior_top_idx]) / max(
                    1.0, float(cumulative_visits[prior_top_idx])
                )
                mcts_q = -float(root.edges.value_sums[mcts_top_idx]) / max(
                    1.0, float(cumulative_visits[mcts_top_idx])
                )
                q_delta = float(max(-2.0, min(2.0, mcts_q - prior_q)))
                summary['mcts_q_delta'] = q_delta
                summary['mcts_changed_to_lower_q'] = (
                    1.0
                    if prior_top_idx != mcts_top_idx and q_delta < -0.02
                    else 0.0
                )
            if cumulative_visits[mcts_top_idx] > 0.0:
                summary['best_q'] = float(max(
                    -1.0,
                    min(
                        1.0,
                        -float(root.edges.value_sums[mcts_top_idx])
                        / float(cumulative_visits[mcts_top_idx]),
                    ),
                ))
        return summary

    def _syzygy_root_result(self, root, board, position_counts, repetition_known):
        """Return a safe root move when forced or locally provable by Syzygy.

        Tablebase values are side-to-move values. Root history matters because
        the tablebase does not know whether a child move itself claims a
        threefold draw, so WDL shortcuts require complete history and no
        already-repeated root. Uncovered choices fall back to ordinary MCTS,
        except that one proven winning child is sufficient to prove a win.
        """
        legal_moves = (
            tuple(root.edges.moves)
            if root.expanded and root.edges is not None
            else root.get_legal_moves()
        )
        if not legal_moves:
            return None
        root_key = _board_position_key(board)
        root_occurrences = int((position_counts or {}).get(root_key, 0))
        # The caller owns claimable/forced root draws. Do not disguise one as
        # a forced move or tablebase result in this lightweight path.
        if (
            int(board.halfmove_clock) >= 100
            or chess.is_insufficient_material(board)
            or (repetition_known and root_occurrences >= 3)
        ):
            return None
        if len(legal_moves) == 1:
            return {
                'selected_move': legal_moves[0],
                'legal_moves': legal_moves,
                'root_values': None,
                'best_value': None,
                'stop_reason': 'forced_move',
            }
        if self.syzygy is None or not self.syzygy.enabled or not repetition_known:
            return None
        if any(int(count) > 1 for count in (position_counts or {}).values()):
            return None

        values = []
        proven_zeroing_wins = []
        immediate_mates = []
        uncovered_child = False
        for move in legal_moves:
            child_board = _copy_board_fast(board)
            chess.apply_move(child_board, move)
            child_key = _board_position_key(child_board)
            child_clock = int(child_board.halfmove_clock)
            child_repetitions = int((position_counts or {}).get(child_key, 0))

            # Checkmate takes precedence over both the 50-move claim and a
            # third-occurrence draw (the normal leaf evaluator does the same).
            if chess.is_check(child_board) and not chess.legal_moves(child_board):
                immediate_mates.append(move)
                values.append(1.0)
                continue

            if (
                child_clock >= 100
                or child_repetitions >= 2
                or chess.is_insufficient_material(child_board)
            ):
                # Search treats a claimable fifty-move/threefold descendant as
                # a draw before consulting Syzygy (unless it is checkmate).
                child_value = 0.0
            else:
                child_value = self.syzygy.probe_value(child_board)
            if child_value is None:
                uncovered_child = True
                values.append(None)
                continue
            if child_value == -1.0 and child_clock == 0 and child_repetitions < 2:
                # WDL is exact immediately after a zeroing move.  A single
                # such losing child proves this root has a winning action even
                # when quiet alternatives fall outside the local TB coverage.
                proven_zeroing_wins.append((move, self.syzygy.probe_dtz(child_board)))
            values.append(-float(child_value))

        if immediate_mates:
            return {
                'selected_move': immediate_mates[0],
                'legal_moves': legal_moves,
                'root_values': None,
                'best_value': 1.0,
                'stop_reason': 'checkmate_in_one',
            }
        if proven_zeroing_wins:
            if len(proven_zeroing_wins) == 1:
                selected_move = proven_zeroing_wins[0][0]
            elif all(dtz is not None and dtz < 0 for _, dtz in proven_zeroing_wins):
                # A losing child has negative DTZ. The shortest safe zeroing
                # line is the closest-to-zero losing DTZ among winning moves.
                selected_move = min(
                    proven_zeroing_wins,
                    key=lambda item: abs(int(item[1])),
                )[0]
            else:
                # Several exact winning moves but incomplete DTZ coverage: let
                # normal Gumbel search compare them instead of cycling through
                # an arbitrary legal-order tie.
                return None
            return {
                'selected_move': selected_move,
                'legal_moves': legal_moves,
                'root_values': None,
                'best_value': 1.0,
                'stop_reason': 'syzygy_proven_root_win',
            }
        if uncovered_child:
            return None

        best_value = max(values)
        best_indices = [
            index for index, value in enumerate(values) if value == best_value
        ]
        if best_value > 0.0 and len(best_indices) > 1:
            # Root-winning actions lead to losing (negative-DTZ) child
            # positions. A safe DTZ ordering avoids repeatedly choosing an
            # arbitrary WDL-equivalent line.
            # Re-probe only tied winners. Their WDL probes were exact above;
            # use each child's signed DTZ (rather than the root-side sign).
            tied_dtz = []
            for index in best_indices:
                child_board = _copy_board_fast(board)
                chess.apply_move(child_board, legal_moves[index])
                tied_dtz.append(self.syzygy.probe_dtz(child_board))
            if not all(value is not None and value < 0 for value in tied_dtz):
                return None
            selected_offset = min(
                range(len(best_indices)),
                key=lambda offset: abs(int(tied_dtz[offset])),
            )
            best_index = best_indices[selected_offset]
        elif best_value < 0.0 and len(best_indices) > 1:
            # In a forced loss, prefer delaying the opponent's winning
            # zeroing progress; child DTZ is positive because the opponent is
            # the winning side to move.
            tied_dtz = []
            for index in best_indices:
                child_board = _copy_board_fast(board)
                chess.apply_move(child_board, legal_moves[index])
                tied_dtz.append(self.syzygy.probe_dtz(child_board))
            if not all(value is not None and value > 0 for value in tied_dtz):
                return None
            selected_offset = max(
                range(len(best_indices)),
                key=lambda offset: int(tied_dtz[offset]),
            )
            best_index = best_indices[selected_offset]
        else:
            best_index = best_indices[0]
        return {
            'selected_move': legal_moves[best_index],
            'legal_moves': legal_moves,
            'root_values': tuple(values),
            'best_value': float(best_value),
            'stop_reason': 'syzygy_exact_root',
        }

    def search_many(
        self,
        game_states,
        num_simulations,
        add_root_noise=False,
        return_search_metadata=False,
        *,
        time_limit_seconds=None,
        budget_mode=None,
        include_diagnostics=True,
    ):
        """
        Run MCTS for multiple games and batch leaf evaluations across games.

        Args:
            game_states: list of dicts with keys: board, root, board_history
            num_simulations: simulations per game
        Returns:
            List[Dict[chess.Move, float]] search weights for each game in order
        """
        inference_cfg = self.config.get('inference', {}) or {}
        if budget_mode is None:
            if time_limit_seconds is not None:
                effective_budget_mode = 'time'
            else:
                # The shared core also serves self-play, reanalysis, and Elo.
                # None of those paths may inherit an interactive wall-clock
                # setting from global config; only an explicit API request
                # opts into timed search.
                effective_budget_mode = 'simulations'
        else:
            effective_budget_mode = str(budget_mode).strip().lower()
        if effective_budget_mode not in {'simulations', 'time'}:
            raise ValueError("budget_mode must be 'simulations' or 'time'")
        requested_time_limit = None
        if effective_budget_mode == 'time':
            raw_time_limit = (
                time_limit_seconds
                if time_limit_seconds is not None
                else inference_cfg.get('move_time_seconds', 0.2)
            )
            if isinstance(raw_time_limit, (bool, np.bool_)):
                raise ValueError('time_limit_seconds must be a finite positive number')
            try:
                requested_time_limit = float(raw_time_limit)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    'time_limit_seconds must be a finite positive number'
                ) from exc
            if not math.isfinite(requested_time_limit) or requested_time_limit <= 0.0:
                raise ValueError('time_limit_seconds must be a finite positive number')
        if not game_states:
            if return_search_metadata:
                return [], []
            return []
        perf_counter = time.perf_counter
        profile_detail = self.profile_detail_enabled
        profile_timing = self.profile_enabled
        search_t0 = perf_counter()
        time_deadline = (
            search_t0 + requested_time_limit
            if requested_time_limit is not None
            else None
        )
        budget_stop_reason = 'simulations_complete'
        evaluation_generation = self._evaluation_generation_token()
        if evaluation_generation != self._dynamic_budget_credit_generation:
            self._dynamic_budget_credits.clear()
            self._dynamic_budget_credit_generation = evaluation_generation
        if bool(getattr(self.model, 'supports_remote_legal_gather', False)):
            if not callable(getattr(self.model, 'get_evaluation_generation', None)):
                raise RuntimeError(
                    'Remote MCTS inference requires get_evaluation_generation() '
                    'to guard tree reuse and prevent mixed-model search results.'
                )
            if evaluation_generation is None:
                raise RuntimeError(
                    'Remote MCTS inference generation is unavailable or changing; '
                    'wait for model loading to finish before searching.'
                )
        game_count = len(game_states)
        boards = [None] * game_count
        roots = [None] * game_count
        initial_root_visits = [0] * game_count
        initial_child_visits = [None] * game_count
        initial_child_value_sums = [None] * game_count
        incoming_root_present = [False] * game_count
        root_reused_flags = [False] * game_count
        root_synced_flags = [False] * game_count
        board_histories = [None] * game_count
        root_position_counts = [None] * game_count
        root_repetition_known = [False] * game_count
        budget_overrides = [None] * game_count
        native_tree_keys = [None] * game_count
        explicit_native_tree_keys = [False] * game_count
        is_mapping_state = [False] * game_count

        # Initialize / reuse roots per game.
        # Supports dict states and the self-play hot-path packed state whose
        # optional stable native-tree key is at index 7. Reject longer packed
        # states: silently accepting appended legacy fields previously caused
        # every eval game to bind its native tree under the boolean ``False``.
        root_setup_t0 = perf_counter() if profile_timing else None
        for idx, gs in enumerate(game_states):
            if isinstance(gs, dict):
                is_mapping_state[idx] = True
                board = gs['board']
                root = gs.get('root', None)
                root_synced = bool(gs.get('_root_synced', False))
                board_history = gs.get('board_history', [])
                position_counts = gs.get('position_counts')
                position_history_complete = gs.get('position_history_complete')
                budget_override = gs.get('simulation_budget_override')
                native_tree_key = gs.get('_native_tree_key', id(board))
                explicit_native_tree_keys[idx] = '_native_tree_key' in gs
            else:
                if len(gs) > 9:
                    raise ValueError(
                        "packed MCTS state supports at most 9 fields; "
                        "use a mapping for named/extended state"
                    )
                board = gs[0]
                root = gs[1] if len(gs) > 1 else None
                root_synced = bool(gs[2]) if len(gs) > 2 else False
                board_history = gs[3] if len(gs) > 3 else []
                position_counts = gs[5] if len(gs) > 5 else None
                budget_override = gs[6] if len(gs) > 6 else None
                native_tree_key = gs[7] if len(gs) > 7 else id(board)
                explicit_native_tree_keys[idx] = len(gs) > 7
                position_history_complete = gs[8] if len(gs) > 8 else None

            boards[idx] = board
            incoming_root_present[idx] = root is not None
            board_histories[idx] = board_history
            root_position_counts[idx] = (
                position_counts
                if isinstance(position_counts, dict)
                else _position_counts_from_board(board)
            )
            if position_history_complete is not None:
                root_repetition_known[idx] = bool(position_history_complete)
            elif isinstance(position_counts, dict):
                # Existing callers that provide explicit occurrence counts are
                # treated as supplying complete provenance unless they mark it
                # partial (legacy FEN-only hard starts do so explicitly).
                root_repetition_known[idx] = True
            else:
                # A zero halfmove clock follows an irreversible move, so no
                # earlier occurrence of the current position can exist. A
                # nonzero-clock FEN without a retained move stack is unknown.
                root_repetition_known[idx] = bool(
                    int(board.halfmove_clock) == 0
                    or bool(chess.move_history(board))
                )
            budget_overrides[idx] = (
                max(1, int(budget_override)) if budget_override is not None else None
            )
            native_tree_keys[idx] = native_tree_key

            observed_context = None
            observed_context = self._build_root_reuse_context(
                board,
                board_history,
                root_position_counts[idx],
                root_repetition_known[idx],
                evaluation_generation,
            )
            if self.reuse_tree and root is not None:
                anchor = getattr(root, '_reuse_context_anchor', None)
                if (
                    isinstance(anchor, _RootReuseContext)
                    and observed_context is not None
                    and anchor.matches(observed_context)
                ):
                    root._reuse_context_anchor = anchor.with_observed_frame(
                        observed_context
                    )
                    root_reused_flags[idx] = True
                    root_synced = False
                elif not root_synced:
                    target_identity = _board_tree_reuse_identity(board)
                    target_key = _board_tree_reuse_key(board)
                    matched_child = None
                    if (
                        isinstance(anchor, _RootReuseContext)
                        and observed_context is not None
                    ):
                        for move in root.edges.moves:
                            child = root.get_child_for_move(move)
                            if child is None:
                                continue
                            if child.get_tree_reuse_key() != target_key:
                                continue
                            _ = child.board  # Ensure board is instantiated.
                            if _board_tree_reuse_identity(child.board) != target_identity:
                                continue
                            expected_child = anchor.advance_to_child(child.board)
                            if (
                                expected_child is not None
                                and expected_child.matches(observed_context)
                            ):
                                matched_child = child
                                break
                    if matched_child is not None:
                        root = matched_child.detach_as_root()
                        root._reuse_context_anchor = observed_context
                        root_reused_flags[idx] = True
                    else:
                        root = MCTSNode(board)
                        root._reuse_context_anchor = observed_context
                else:
                    root = MCTSNode(board)
                    root._reuse_context_anchor = observed_context
                    root_synced = False
            else:
                root = MCTSNode(board)
                root._reuse_context_anchor = observed_context
                root_synced = False

            roots[idx] = root
            root_synced_flags[idx] = bool(root_synced)
        if profile_timing:
            self._profile_add('search_root_setup_time', perf_counter() - root_setup_t0)

        # Exact tablebase ranking may replace the final move in deterministic
        # play. Lightweight callers can skip the search entirely; diagnostic
        # callers still run it and retain their policy/value targets. Self-play
        # noise always follows the ordinary Gumbel path.
        root_shortcuts = [None] * game_count
        root_proofs = [None] * game_count
        if not add_root_noise and return_search_metadata:
            for idx, board in enumerate(boards):
                root_result = self._syzygy_root_result(
                    roots[idx],
                    board,
                    root_position_counts[idx],
                    root_repetition_known[idx],
                )
                if root_result is None:
                    continue
                if root_result['stop_reason'] == 'forced_move':
                    if not include_diagnostics:
                        root_shortcuts[idx] = root_result
                else:
                    root_proofs[idx] = root_result
                    if not include_diagnostics:
                        root_shortcuts[idx] = root_result

        time_budget_active = effective_budget_mode == 'time'
        dynamic_budget_requested = bool(
            not time_budget_active
            and self.dynamic_budget_enabled
            and any(value is None for value in budget_overrides)
        )
        dynamic_budget_feedback_active = bool(
            dynamic_budget_requested
            and self.dynamic_budget_strategy == 'feedback'
        )
        # Determine the effective controller before graph selection so the
        # multi-root auto route cannot accidentally include fixed-budget or
        # time-mode batches. Explicit graph=True remains an override.
        dynamic_budget_active = bool(
            dynamic_budget_feedback_active
            or (
                dynamic_budget_requested
                and self.dynamic_budget_strategy == 'prior'
                and game_count > 1
            )
        )
        effective_root_schedule = (
            'revisiting'
            if time_budget_active or dynamic_budget_feedback_active
            else self.gumbel_root_schedule
        )

        native_forest = self._native_forest
        if self._native_tree_enabled:
            native_import_t0 = perf_counter() if profile_timing else None
            try:
                if native_forest is None or not native_forest.active:
                    native_forest = self._native_mcts.create_forest(
                        [],
                        c_visit=self.gumbel_c_visit,
                        c_scale=self.gumbel_c_scale,
                        q_range_floor=self.gumbel_q_range_floor,
                        use_mixed_value=self.gumbel_use_mixed_value,
                    )
                    self._native_forest = native_forest
                nodes_before = native_forest.node_count
                edges_before = native_forest.edge_count
                native_forest.prepare_roots(roots, native_tree_keys)
                self._profile_inc(
                    'native_tree_import_nodes',
                    max(0, native_forest.node_count - nodes_before),
                )
                self._profile_inc(
                    'native_tree_import_edges',
                    max(0, native_forest.edge_count - edges_before),
                )
            except Exception:
                self._profile_inc('native_tree_fallbacks')
                # A persistent forest can already contain authoritative child
                # statistics, so silently switching to stale Python traversal
                # would be incorrect. Fail loudly and preserve diagnosability.
                raise
            finally:
                if profile_timing:
                    self._profile_add(
                        'native_tree_import_time',
                        perf_counter() - native_import_t0,
                    )
        if self._local_cuda_graph_forward_enabled or self._local_cuda_graph_forward_runner is not None:
            graph_auto_eligible = self._local_cuda_graph_auto_eligible(
                game_count,
                native_forest,
                feedback_active=dynamic_budget_feedback_active,
                all_roots_adaptive=all(value is None for value in budget_overrides),
            )
            self._prepare_local_cuda_graph_forward(
                evaluation_generation,
                auto_eligible=graph_auto_eligible,
                exact_batch_sizes=self._local_cuda_graph_batch_sizes_for_call(
                    auto_eligible=graph_auto_eligible,
                    game_count=game_count,
                ),
            )
        feedback_eval_stats = (
            [
                {
                    'nn_fresh_evaluations': 0,
                    'nn_cache_hits': 0,
                    'nn_deduplicated_evaluations': 0,
                    'nn_unknown_rows': 0,
                    'nn_provenance_complete': True,
                    'mcts_duplicate_leaf_selections': 0,
                    'inherited_leaf_evaluations': 0,
                    'exact_leaf_evaluations': 0,
                }
                if budget_overrides[idx] is None else None
                for idx in range(game_count)
            ]
            if dynamic_budget_feedback_active
            else None
        )

        for idx, root in enumerate(roots):
            initial_root_visits[idx] = (
                0 if root is None else int(getattr(root, 'visit_count', 0) or 0)
            )
            if root is not None and root.expanded and root.edges is not None:
                initial_child_visits[idx] = root.edges.visit_counts.copy()
                if dynamic_budget_feedback_active:
                    initial_child_value_sums[idx] = root.edges.value_sums.copy()

        fixed_simulation_budget = max(1, int(num_simulations))
        native_dirty_root_indices = set()

        def _sync_native_roots(root_indices=None):
            if native_forest is None:
                return
            sync_t0 = perf_counter() if profile_timing else None
            if root_indices is None:
                native_forest.sync_roots()
                native_dirty_root_indices.clear()
            else:
                # A completed feedback checkpoint reads only these roots. The
                # other roots retain their last exported snapshot and need not
                # cross the native/Python boundary again yet.
                for root_idx in root_indices:
                    root_idx = int(root_idx)
                    if root_idx < 0 or root_idx >= game_count:
                        continue
                    root = roots[root_idx]
                    if root is not None and root.edges is not None:
                        native_forest.sync_node(root)
                        native_dirty_root_indices.discard(root_idx)
            if profile_timing:
                self._profile_add(
                    'native_tree_sync_time',
                    perf_counter() - sync_t0,
                )

        eval_easy_cut_active = bool(
            not time_budget_active
            and not dynamic_budget_feedback_active
            and self.eval_easy_cut_enabled
            and not add_root_noise
            and game_count > 1
            and all(value is None for value in budget_overrides)
        )
        simulation_budgets = [fixed_simulation_budget] * game_count
        gumbel_root_states = [None] * game_count
        game_ptr = 0
        dynamic_minimum, dynamic_target, dynamic_maximum, dynamic_chunk = (
            _resolve_dynamic_simulation_budget(
                fixed_simulation_budget,
                minimum=self.dynamic_budget_minimum,
                maximum_multiplier=self.dynamic_budget_maximum_multiplier,
            )
        )

        def _run_simulation_phase(
            remaining,
            root_selection_states=None,
            post_batch_callback=None,
            post_feedback_callback=None,
            native_root_pass_update=None,
            deadline=None,
        ):
            nonlocal game_ptr
            if native_forest is not None and root_selection_states is not None:
                if native_root_pass_update is None:
                    native_forest.configure_root_states(root_selection_states)
                else:
                    native_forest.configure_root_pass(native_root_pass_update)
            total_remaining = int(sum(remaining))
            while total_remaining > 0:
                # A selected batch is always fully evaluated and backed up.
                # The soft deadline is observed before starting the next batch,
                # so neither native nor Python virtual losses can be stranded.
                if deadline is not None and perf_counter() >= deadline:
                    break
                # Keep decisions in coarse batches. This preserves central GPU
                # batching even though roots receive different total budgets.
                active_count = sum(1 for value in remaining if value > 0)
                max_remaining_any_game = max(remaining) if remaining else 1
                if (
                    native_forest is not None
                    and game_count == 1
                    and self.single_root_batch_cap > 0
                ):
                    slots_per_game = min(
                        max(1, int(self.eval_batch_size)),
                        self.single_root_batch_cap,
                        max(1, max_remaining_any_game),
                    )
                else:
                    slots_per_game = max(1, min(
                        self.eval_batch_size // max(1, active_count),
                        max(1, max_remaining_any_game // 4),
                    ))
                batch_size = min(
                    self.eval_batch_size,
                    total_remaining,
                    slots_per_game * max(1, active_count),
                )
                leaf_nodes = []
                search_paths = []
                selection_ids = None
                leaf_game_indices = []
                scheduled_game_indices = []
                selected_this_batch = [0] * game_count
                feedback_windows = [0] * game_count
                if root_selection_states is not None:
                    for state_idx, root_state in enumerate(root_selection_states):
                        if not root_state:
                            continue
                        sequence = root_state.get('sequence', ())
                        progress = max(0, int(root_state.get('_pass_scheduled', 0)))
                        if not sequence or progress >= len(sequence):
                            continue
                        boundaries = root_state.get('_feedback_boundaries', ())
                        boundary_index = max(
                            0,
                            int(root_state.get('_feedback_boundary_index', 0)),
                        )
                        while (
                            boundary_index < len(boundaries)
                            and int(boundaries[boundary_index]) <= progress
                        ):
                            boundary_index += 1
                        root_state['_feedback_boundary_index'] = boundary_index
                        if boundary_index < len(boundaries):
                            feedback_windows[state_idx] = max(
                                1,
                                int(boundaries[boundary_index]) - progress,
                            )
                selection_node_traversals_this_batch = 0

                selection_t0 = perf_counter() if profile_timing else None
                for _ in range(batch_size):
                    found = False
                    for _ in range(game_count):
                        if (
                            feedback_windows[game_ptr] > 0
                            and selected_this_batch[game_ptr]
                            >= feedback_windows[game_ptr]
                        ):
                            game_ptr = (game_ptr + 1) % game_count
                            continue
                        current_root = roots[game_ptr]
                        if (
                            remaining[game_ptr] > 0
                            and not (
                                selected_this_batch[game_ptr] > 0
                                and current_root is not None
                                and not current_root.expanded
                            )
                        ):
                            found = True
                            break
                        game_ptr = (game_ptr + 1) % game_count

                    if not found:
                        break

                    gs_idx = game_ptr
                    scheduled_game_indices.append(gs_idx)
                    selected_this_batch[gs_idx] += 1

                    remaining[gs_idx] -= 1
                    total_remaining -= 1
                    game_ptr = (game_ptr + 1) % game_count

                if native_forest is not None and scheduled_game_indices:
                    leaf_nodes, selection_ids, depths = native_forest.select_batch(
                        scheduled_game_indices
                    )
                    native_dirty_root_indices.update(scheduled_game_indices)
                    leaf_game_indices.extend(scheduled_game_indices)
                    selection_node_traversals_this_batch = int(
                        depths.sum(dtype=np.int64)
                    )
                    self._profile_inc('native_tree_selection_batches')
                    self._profile_inc(
                        'native_tree_selected_leaves',
                        len(scheduled_game_indices),
                    )
                else:
                    self._profile_inc(
                        'python_tree_selected_leaves',
                        len(scheduled_game_indices),
                    )
                    for gs_idx in scheduled_game_indices:
                        node = roots[gs_idx]
                        search_path = [node]
                        node._root_virtual_loss += 1

                        while node.expanded:
                            root_state = (
                                root_selection_states[gs_idx]
                                if root_selection_states is not None
                                else None
                            )
                            if node is roots[gs_idx] and root_state is not None:
                                node = self._select_root_gumbel(
                                    node,
                                    root_state,
                                    apply_virtual_loss=True,
                                )
                            else:
                                node = self._select_child_gumbel(
                                    node,
                                    apply_virtual_loss=True,
                                )
                            if node is None:
                                break
                            search_path.append(node)

                        search_paths.append(search_path)
                        selection_node_traversals_this_batch += len(search_path)
                        leaf_nodes.append(node)
                        leaf_game_indices.append(gs_idx)
                if profile_timing:
                    self._profile_add('search_selection_time', perf_counter() - selection_t0)
                self._profile_inc(
                    'selection_node_traversals',
                    selection_node_traversals_this_batch,
                )

                if not leaf_nodes:
                    break

                adaptive_leaf_candidates = []
                if dynamic_budget_feedback_active and root_selection_states is not None:
                    seen_adaptive_candidates = set()
                    for node, gs_idx in zip(leaf_nodes, leaf_game_indices):
                        node_id = id(node)
                        if (
                            node_id not in seen_adaptive_candidates
                            and node is not roots[gs_idx]
                            and node.raw_value is None
                            and not node.expanded
                        ):
                            seen_adaptive_candidates.add(node_id)
                            adaptive_leaf_candidates.append((node, gs_idx))
                evaluation_trace = {} if dynamic_budget_feedback_active else None
                values = self._batch_expand_and_evaluate(
                    leaf_nodes,
                    leaf_game_indices,
                    board_histories,
                    root_position_counts,
                    root_repetition_known,
                    adaptive_stats=feedback_eval_stats,
                    evaluation_trace=evaluation_trace,
                )
                if adaptive_leaf_candidates:
                    # Count one new-leaf identity per attributed NN/cache row;
                    # MCTS duplicates and inference-level dedup rows add none.
                    sources_by_node = (
                        evaluation_trace.get('row_sources_by_node', {})
                        if isinstance(evaluation_trace, dict)
                        else {}
                    )
                    for node, gs_idx in adaptive_leaf_candidates:
                        source = sources_by_node.get(id(node))
                        if source not in {'fresh_nn', 'cache_hit', 'exact_terminal'}:
                            continue
                        root = roots[gs_idx]
                        if root is None:
                            continue
                        state = root_selection_states[gs_idx]
                        edge_cache = state.setdefault(
                            '_adaptive_root_edge_index_cache', {}
                        ) if state is not None else None
                        edge_index = None
                        if edge_cache is not None:
                            path_ids = []
                            current = node
                            while current is not None and current is not root:
                                cached_edge = edge_cache.get(id(current))
                                if cached_edge is not None:
                                    edge_index = int(cached_edge)
                                    break
                                path_ids.append(id(current))
                                parent = current.parent
                                if parent is root:
                                    edge_index = int(current.parent_edge_index)
                                    break
                                current = parent
                            if edge_index is None:
                                continue
                            for path_id in path_ids:
                                edge_cache[path_id] = edge_index
                        else:
                            root_edge = node
                            while root_edge.parent is not None and root_edge.parent is not root:
                                root_edge = root_edge.parent
                            if root_edge.parent is not root:
                                continue
                            edge_index = int(root_edge.parent_edge_index)
                        if state is not None and 0 <= edge_index < len(root.edges.moves):
                            if source in {'fresh_nn', 'cache_hit'}:
                                state.setdefault(
                                    '_adaptive_fresh_leaf_ids_by_edge',
                                    [set() for _ in root.edges.moves],
                                )[edge_index].add(id(node))
                            else:
                                state.setdefault(
                                    '_adaptive_exact_leaf_ids_by_edge',
                                    [set() for _ in root.edges.moves],
                                )[edge_index].add(id(node))

                backprop_t0 = perf_counter() if profile_timing else None
                if native_forest is not None:
                    expand_sync_t0 = perf_counter() if profile_timing else None
                    native_forest.sync_expansions(leaf_nodes)
                    if profile_timing:
                        self._profile_add(
                            'native_tree_expand_sync_time',
                            perf_counter() - expand_sync_t0,
                        )
                    backup_t0 = perf_counter() if profile_timing else None
                    native_forest.backup_batch(
                        selection_ids,
                        values,
                    )
                    if profile_timing:
                        self._profile_add(
                            'native_tree_backup_time',
                            perf_counter() - backup_t0,
                        )
                    self._profile_inc('native_tree_backup_batches')
                else:
                    backprop_t0 = perf_counter() if profile_timing else None
                    for search_path, value in zip(search_paths, values):
                        self._backpropagate_and_remove_virtual_loss(search_path, value)
                if profile_timing:
                    self._profile_add(
                        'search_backprop_time',
                        perf_counter() - backprop_t0,
                    )
                if root_selection_states is not None:
                    for state_idx, count in enumerate(selected_this_batch):
                        root_state = root_selection_states[state_idx]
                        if root_state:
                            root_state['_pass_scheduled'] = int(
                                root_state.get('_pass_scheduled', 0)
                            ) + int(count)
                feedback_checkpoint_indices = []
                if post_feedback_callback is not None and root_selection_states is not None:
                    for state_idx, count in enumerate(selected_this_batch):
                        root_state = root_selection_states[state_idx]
                        if count <= 0 or not root_state:
                            continue
                        progress = int(root_state.get('_pass_scheduled', 0))
                        if progress in root_state.get('_feedback_boundaries', ()):
                            feedback_checkpoint_indices.append(state_idx)
                if post_batch_callback is not None:
                    _sync_native_roots()
                elif feedback_checkpoint_indices:
                    _sync_native_roots(feedback_checkpoint_indices)
                if post_batch_callback is not None:
                    removed = max(
                        0,
                        int(post_batch_callback(remaining, selected_this_batch) or 0),
                    )
                    total_remaining = max(0, total_remaining - removed)
                if feedback_checkpoint_indices:
                    removed = max(
                        0,
                        int(post_feedback_callback(feedback_checkpoint_indices, remaining) or 0),
                    )
                    total_remaining = max(0, total_remaining - removed)
            if native_dirty_root_indices:
                _sync_native_roots(tuple(native_dirty_root_indices))

        scout_difficulties = [0.0] * game_count
        # Evaluate every unexpanded root exactly once so priors/raw values exist
        # before assigning the complete Sequential-Halving budget.
        root_setup = [
            1
            if (
                root is not None
                and not root.expanded
                and root_shortcuts[idx] is None
            )
            else 0
            for idx, root in enumerate(roots)
        ]
        if any(root_setup):
            _run_simulation_phase(root_setup)
        for idx, root in enumerate(roots):
            if (
                root is not None
                and root.expanded
                and root.edges is not None
                and initial_child_visits[idx] is None
            ):
                # Root expansion is setup, not root-action evidence. Snapshot
                # zero edge statistics now so both adaptive Q and its evidence
                # floor count only simulations spent choosing legal moves.
                initial_child_visits[idx] = root.edges.visit_counts.copy()
                if dynamic_budget_feedback_active:
                    initial_child_value_sums[idx] = root.edges.value_sums.copy()

        if dynamic_budget_active or eval_easy_cut_active:
            scout_difficulties = [_gumbel_prior_difficulty(root) for root in roots]
        adaptive_indices = [
            idx for idx, value in enumerate(budget_overrides) if value is None
        ]
        if adaptive_indices:
            if dynamic_budget_feedback_active:
                # Feedback begins with a shared scout pass; the immutable
                # target-sized pool is released only after that evidence exists.
                adaptive_total = fixed_simulation_budget * len(adaptive_indices)
            else:
                # Preserve the HEAD prior allocator's mixed-override total.
                overridden_total = sum(
                    int(value) for value in budget_overrides if value is not None
                )
                adaptive_total = max(
                    len(adaptive_indices),
                    fixed_simulation_budget * game_count - overridden_total,
                )
            adaptive_target = float(adaptive_total) / float(len(adaptive_indices))
            if eval_easy_cut_active:
                adaptive_budgets = _allocate_eval_easy_cut_simulation_budgets(
                    [scout_difficulties[idx] for idx in adaptive_indices],
                    target=fixed_simulation_budget,
                    minimum_fraction=self.eval_easy_cut_minimum_fraction,
                    difficulty_threshold=self.eval_easy_cut_difficulty_threshold,
                    chunk=self.eval_easy_cut_chunk,
                )
            elif dynamic_budget_feedback_active:
                scout_budget = min(
                    int(dynamic_maximum),
                    max(
                        int(dynamic_minimum),
                        (int(fixed_simulation_budget) + 1) // 2,
                    ),
                )
                adaptive_budgets = [scout_budget] * len(adaptive_indices)
            elif dynamic_budget_active:
                adaptive_budgets = _allocate_dynamic_simulation_budgets(
                    [scout_difficulties[idx] for idx in adaptive_indices],
                    minimum=dynamic_minimum,
                    target_average=adaptive_target,
                    maximum=dynamic_maximum,
                    chunk=dynamic_chunk,
                )
            else:
                base_budget, remainder = divmod(adaptive_total, len(adaptive_indices))
                adaptive_budgets = [
                    int(base_budget + (1 if idx < remainder else 0))
                    for idx in range(len(adaptive_indices))
                ]
            for idx, budget in zip(adaptive_indices, adaptive_budgets):
                simulation_budgets[idx] = int(budget)
            dynamic_target = (
                float(sum(adaptive_budgets)) / float(len(adaptive_budgets))
                if eval_easy_cut_active and adaptive_budgets
                else adaptive_target
            )
        for idx, budget_override in enumerate(budget_overrides):
            if budget_override is not None:
                simulation_budgets[idx] = int(budget_override)
        prior_dynamic_credit = 0
        if (
            dynamic_budget_feedback_active
            and game_count == 1
            and explicit_native_tree_keys[0]
        ):
            tree_key = native_tree_keys[0]
            try:
                prior_dynamic_credit = int(self._dynamic_budget_credits.get(tree_key, 0))
            except (TypeError, ValueError):
                prior_dynamic_credit = 0
            prior_dynamic_credit = max(
                0,
                min(dynamic_maximum - dynamic_target, prior_dynamic_credit),
            )
            simulation_budgets[0] = min(
                dynamic_maximum,
                int(simulation_budgets[0]) + prior_dynamic_credit,
            )
        dynamic_budget_initial_budgets = list(simulation_budgets)
        dynamic_budget_pool_total = (
            int(fixed_simulation_budget) * len(adaptive_indices)
            + int(prior_dynamic_credit)
            if dynamic_budget_feedback_active else 0
        )

        setup_visits = [
            max(0, int(getattr(root, 'visit_count', 0) or 0) - initial_root_visits[idx])
            if root is not None else 0
            for idx, root in enumerate(roots)
        ]
        root_gumbel_scale = self.gumbel_scale if add_root_noise else self.gumbel_eval_scale
        root_gumbel_noises = [None] * game_count
        tree_reuse_quality = [
            {
                'quality': 0.0,
                'candidate_coverage': 0.0,
                'candidate_support': 0.0,
                'visited_prior_mass': 0.0,
                'effective_action_ratio': 0.0,
                'considered_actions': 0,
            }
            for _ in range(game_count)
        ]
        # Sample the new root's noise before assigning reuse credit.  This makes
        # the saved work conditional on the candidates the upcoming Sequential
        # Halving search will actually compare, instead of trusting raw visits
        # accumulated under the previous root.
        for idx, root in enumerate(roots):
            if root is None or not root.expanded or not root.edges.moves:
                continue
            noise = (
                np.random.gumbel(size=len(root.edges.moves)).astype(np.float32)
                * float(root_gumbel_scale)
            )
            root_gumbel_noises[idx] = noise
            if root_reused_flags[idx]:
                pre_credit_budget = max(
                    0,
                    int(simulation_budgets[idx]) - int(setup_visits[idx]),
                )
                tree_reuse_quality[idx] = _summarize_tree_reuse_quality(
                    root,
                    noise,
                    _resolve_gumbel_considered_actions(
                        self.gumbel_max_considered_actions,
                        len(root.edges.moves),
                        pre_credit_budget,
                    ),
                )
        inherited_visit_credits = [0] * game_count
        if (
            add_root_noise
            and self.reuse_tree
            and self.tree_reuse_visit_credit_enabled
            and not dynamic_budget_feedback_active
        ):
            for idx, budget in enumerate(simulation_budgets):
                if not root_reused_flags[idx]:
                    continue
                inherited_visit_credits[idx] = _resolve_tree_reuse_visit_credit(
                    budget,
                    initial_root_visits[idx],
                    quality=tree_reuse_quality[idx]['quality'],
                )
        action_budgets = [
            (
                0
                if root_shortcuts[idx] is not None
                else max(1, int(simulation_budgets[idx]))
            )
            if time_budget_active
            else max(
                0,
                int(simulation_budgets[idx])
                - int(setup_visits[idx])
                - int(inherited_visit_credits[idx]),
            )
            for idx in range(game_count)
        ]
        # ``_run_simulation_phase`` consumes its mutable remaining-budget list
        # in place. Preserve the assigned budget for post-search telemetry and
        # Sequential Halving diagnostics before that mutation starts.
        gumbel_action_budgets = list(action_budgets)

        pre_scout_winners = [None] * game_count
        pre_scout_policies = [None] * game_count
        tree_reuse_scout_results = [
            {
                'scout_simulations': 0,
                'winner_stable': False,
                'policy_js_divergence': 0.0,
                'policy_stability': 0.0,
                'combined_stability': 0.0,
                'extra_credit': 0,
                'fresh_floor': (
                    _resolve_tree_reuse_fresh_floor(
                        simulation_budgets[idx],
                        tree_reuse_quality[idx]['quality'],
                    )
                    if root_reused_flags[idx]
                    else int(simulation_budgets[idx])
                ),
                'search_reduced': False,
            }
            for idx in range(game_count)
        ]
        for idx, root in enumerate(roots):
            if root is None or not root.expanded or not root.edges.moves:
                continue
            initial = initial_child_visits[idx]
            if initial is None or len(initial) != len(root.edges.moves):
                initial = np.zeros(len(root.edges.moves), dtype=np.int32)
            # Feedback starts with a reduced scout quota, but its later
            # transfers can fund the ordinary target-sized search. Resolve
            # shortlist width from that nominal target so the scout does not
            # permanently halve the candidate set before extra work arrives.
            width_budget = (
                int(fixed_simulation_budget)
                if dynamic_budget_feedback_active and budget_overrides[idx] is None
                else action_budgets[idx]
            )
            considered = _resolve_gumbel_considered_actions(
                self.gumbel_max_considered_actions,
                len(root.edges.moves),
                width_budget,
            )
            pass_budgets = (
                (
                    (int(action_budgets[idx]),)
                    if time_budget_active and action_budgets[idx] > 0
                    else _gumbel_revisiting_pass_budgets(
                        action_budgets[idx],
                        considered,
                    )
                )
                if effective_root_schedule == 'revisiting'
                else ((int(action_budgets[idx]),) if action_budgets[idx] > 0 else ())
            )
            root_sequence = (
                _gumbel_considered_visit_sequence(considered, pass_budgets[0])
                if effective_root_schedule == 'revisiting' and pass_budgets
                else _gumbel_considered_visit_sequence(
                    considered,
                    action_budgets[idx],
                )
            )
            initial_copy = np.asarray(initial, dtype=np.int32).copy()
            feedback_boundaries = (
                _gumbel_revisiting_feedback_boundaries(
                    considered,
                    action_budgets[idx],
                )
                if effective_root_schedule == 'standard'
                else ()
            )
            if dynamic_budget_feedback_active and action_budgets[idx] > 0:
                # A completed pass is also a valid stability checkpoint once
                # the halving schedule has reached its final two candidates.
                if not feedback_boundaries or feedback_boundaries[-1] != int(action_budgets[idx]):
                    feedback_boundaries = tuple(feedback_boundaries) + (int(action_budgets[idx]),)
            gumbel_root_states[idx] = {
                'initial_visits': initial_copy,
                'pass_initial_visits': initial_copy.copy(),
                'gumbel': root_gumbel_noises[idx],
                'considered_actions': int(considered),
                'sequence': root_sequence,
                'root_schedule': effective_root_schedule,
                'q_scale_total_visits': self.gumbel_reuse_q_scale == 'total',
                'pass_budgets': [int(value) for value in pass_budgets],
                'pass_considered_actions': [int(considered)] * len(pass_budgets),
                'pass_index': 0,
                '_pass_scheduled': 0,
                '_feedback_boundaries': feedback_boundaries,
                '_feedback_boundary_index': 0,
                '_initial_visits_f64': np.asarray(initial_copy, dtype=np.float64),
                '_pass_initial_visits_f64': np.asarray(initial_copy, dtype=np.float64),
                '_pass_initial_total': float(initial_copy.sum()),
                '_adaptive_fresh_leaf_ids_by_edge': (
                    [set() for _ in root.edges.moves]
                    if dynamic_budget_feedback_active
                    and root is not None and root.edges is not None
                    else []
                ),
                '_adaptive_exact_leaf_ids_by_edge': (
                    [set() for _ in root.edges.moves]
                    if dynamic_budget_feedback_active
                    and root is not None and root.edges is not None
                    else []
                ),
                '_adaptive_last_winner': None,
                '_adaptive_stable_checkpoints': 0,
                '_adaptive_q_gap': 0.0,
                '_adaptive_score_gap': None,
                '_adaptive_challenger': None,
                '_adaptive_uncertainty_score': 0.0,
                '_adaptive_uncertainty_rank': 0.0,
                '_adaptive_transferred_in': 0,
                '_adaptive_winner_switches': 0,
                '_adaptive_fresh_floor': int(dynamic_minimum),
                '_adaptive_early_stopped': False,
                '_adaptive_stop_reason': 'budget_exhausted',
            }
            if dynamic_budget_feedback_active:
                state = gumbel_root_states[idx]
                state['initial_value_sums'] = np.asarray(
                    initial_child_value_sums[idx]
                    if initial_child_value_sums[idx] is not None
                    else np.zeros_like(initial_copy, dtype=np.float32),
                    dtype=np.float32,
                ).copy()
            if time_budget_active:
                state = gumbel_root_states[idx]
                state['_stable_selected_idx'] = self._gumbel_final_action_index(
                    root,
                    state,
                )
                state['_stable_checkpoint_pass'] = -1
                state['_stable_checkpoint_visit'] = 0
            if (
                effective_root_schedule == 'standard'
                and add_root_noise
                and root_reused_flags[idx]
                and action_budgets[idx] > _TREE_REUSE_SCOUT_SIMULATIONS
                and tree_reuse_quality[idx]['quality'] >= _TREE_REUSE_SCOUT_MIN_TREE_QUALITY
                and tree_reuse_quality[idx]['candidate_coverage']
                >= _TREE_REUSE_SCOUT_MIN_CANDIDATE_COVERAGE
            ):
                pre_scout_winners[idx] = self._gumbel_final_action_index(
                    root,
                    gumbel_root_states[idx],
                )
                pre_scout_policies[idx] = self._gumbel_improved_policy(
                    root,
                    scale_visit_counts=self._gumbel_root_q_scale_visit_counts(
                        root,
                        gumbel_root_states[idx],
                    ),
                ).copy()

        def _apply_ready_tree_reuse_scouts(remaining, selected_this_batch):
            removed_total = 0
            for idx, selected_count in enumerate(selected_this_batch):
                pre_policy = pre_scout_policies[idx]
                state = gumbel_root_states[idx]
                root = roots[idx]
                if (
                    selected_count <= 0
                    or pre_policy is None
                    or state is None
                    or root is None
                ):
                    continue
                fresh_counts = np.maximum(
                    0.0,
                    root.edges.visit_counts.astype(np.float64)
                    - np.asarray(state['initial_visits'], dtype=np.float64),
                )
                scout_simulations = int(round(float(fresh_counts.sum())))
                if scout_simulations < _TREE_REUSE_SCOUT_SIMULATIONS:
                    continue

                # Mark the root before doing the audit so it is evaluated only
                # once. The enclosing simulation loop keeps every other root in
                # the same continuous batch stream; no phase boundary or flush.
                pre_scout_policies[idx] = None
                post_winner = self._gumbel_final_action_index(root, state)
                post_policy = self._gumbel_improved_policy(
                    root,
                    scale_visit_counts=self._gumbel_root_q_scale_visit_counts(
                        root,
                        state,
                    ),
                ).copy()
                policy_js = _policy_jensen_shannon_divergence(pre_policy, post_policy)
                winner_stable = pre_scout_winners[idx] == post_winner
                scout_credit = _resolve_tree_reuse_post_scout_credit(
                    simulation_budgets[idx],
                    initial_root_visits[idx],
                    inherited_visit_credits[idx],
                    tree_quality=tree_reuse_quality[idx]['quality'],
                    winner_stable=winner_stable,
                    policy_js_divergence=policy_js,
                )
                extra_credit = min(
                    int(scout_credit['extra_credit']),
                    max(0, int(remaining[idx])),
                )
                if extra_credit > 0:
                    inherited_visit_credits[idx] += extra_credit
                    remaining[idx] -= extra_credit
                    removed_total += extra_credit
                tree_reuse_scout_results[idx] = {
                    'scout_simulations': int(scout_simulations),
                    'winner_stable': bool(winner_stable),
                    'policy_js_divergence': float(policy_js),
                    'policy_stability': float(scout_credit['policy_stability']),
                    'combined_stability': float(scout_credit['combined_stability']),
                    'extra_credit': int(extra_credit),
                    'fresh_floor': int(scout_credit['fresh_floor']),
                    'search_reduced': bool(extra_credit > 0),
                }
            return removed_total

        reuse_scout_callback = (
            _apply_ready_tree_reuse_scouts
            if (
                not dynamic_budget_feedback_active
                and
                effective_root_schedule == 'standard'
                and any(policy is not None for policy in pre_scout_policies)
            )
            else None
        )
        if time_budget_active:
            def _record_stable_gumbel_choices(root_indices, _remaining=None):
                for idx in root_indices:
                    root = roots[idx]
                    state = gumbel_root_states[idx]
                    if root is None or state is None or not root.edges.moves:
                        continue
                    selected_idx = self._gumbel_final_action_index(root, state)
                    if selected_idx is not None:
                        state['_stable_selected_idx'] = int(selected_idx)
                        state['_stable_checkpoint_pass'] = int(
                            state.get('pass_index', 0)
                        )
                        state['_stable_checkpoint_visit'] = int(
                            state.get('_pass_scheduled', 0)
                        )

        def _adaptive_uncertainty_score(root_idx):
            """One bounded urgency score shared by funding and metadata."""
            state = gumbel_root_states[root_idx]
            if state is None:
                return 0.0
            leaf_counts = state.get('_adaptive_fresh_leaf_ids_by_edge', ())
            support = min(
                (
                    len(leaf_counts[int(edge)])
                    for edge in (
                        state.get('_adaptive_last_winner'),
                        state.get('_adaptive_challenger'),
                    )
                    if edge is not None and int(edge) < len(leaf_counts)
                ),
                default=0,
            )
            return _dynamic_budget_uncertainty_score(
                state.get('_adaptive_score_gap'),
                state.get('_adaptive_q_gap'),
                state.get('_adaptive_winner_switches', 0),
                support,
                state.get('_adaptive_stable_checkpoints', 0),
                tree_reuse_quality[root_idx]['quality'],
            )

        def _adaptive_feedback_callback(root_indices, remaining):
            """Stop only after completed SH feedback with fresh evidence."""
            removed_total = 0
            for idx in root_indices:
                if (
                    not dynamic_budget_feedback_active
                    or budget_overrides[idx] is not None
                    or idx >= len(remaining)
                ):
                    continue
                root = roots[idx]
                state = gumbel_root_states[idx]
                if (
                    root is None
                    or state is None
                    or not root.edges.moves
                    or state.get('_adaptive_early_stopped', False)
                ):
                    continue

                final_counts = _gumbel_final_visit_counts(root, state)
                eligible, _ = _gumbel_final_action_eligibility(final_counts)
                if int(eligible.sum()) < 2:
                    winner = self._gumbel_final_action_index(root, state)
                    if winner is None:
                        continue
                    state['_adaptive_last_winner'] = int(winner)
                    state['_adaptive_stable_checkpoints'] = 1
                    continue

                # Final action follows canonical max-visit selection. The
                # challenger guard inspects every legal edge because the
                # Gumbel shortlist is rank-based, not a prefix of edge order.
                # Reuse this completed-Q score vector for both decisions rather
                # than asking the native final-action kernel to build it first.
                scale_counts = self._gumbel_root_q_scale_visit_counts(root, state)
                scores = root.edges.log_base_priors.astype(np.float64, copy=True)
                scores -= float(np.max(scores))
                scores += np.asarray(state['gumbel'], dtype=np.float64)
                scores += self._gumbel_completed_qvalues(
                    root,
                    scale_visit_counts=scale_counts,
                )
                winner = int(np.argmax(np.where(eligible, scores, -np.inf)))
                challenger = _adaptive_challenger_index(scores, winner)
                if challenger is None:
                    continue
                score_gap = float(scores[int(winner)] - scores[int(challenger)])
                state['_adaptive_score_gap'] = score_gap
                state['_adaptive_challenger'] = int(challenger)

                previous_winner = state.get('_adaptive_last_winner')
                if previous_winner is not None and previous_winner != int(winner):
                    state['_adaptive_winner_switches'] = int(
                        state.get('_adaptive_winner_switches', 0)
                    ) + 1
                stable_checkpoints = (
                    int(state.get('_adaptive_stable_checkpoints', 0)) + 1
                    if previous_winner == int(winner)
                    else 1
                )
                state['_adaptive_last_winner'] = int(winner)
                state['_adaptive_stable_checkpoints'] = stable_checkpoints

                initial_visits = np.asarray(state['initial_visits'], dtype=np.float64)
                fresh_visits = np.maximum(
                    0.0,
                    root.edges.visit_counts.astype(np.float64) - initial_visits,
                )
                initial_sums = np.asarray(
                    state.get('initial_value_sums', np.zeros_like(fresh_visits)),
                    dtype=np.float64,
                )
                fresh_sums = (
                    root.edges.value_sums.astype(np.float64) - initial_sums
                )
                fresh_q = np.zeros(fresh_visits.size, dtype=np.float64)
                visited = fresh_visits > 0.0
                fresh_q[visited] = -fresh_sums[visited] / fresh_visits[visited]
                q_gap = float(fresh_q[int(winner)] - fresh_q[challenger])
                state['_adaptive_q_gap'] = q_gap
                leaf_ids = state.get('_adaptive_fresh_leaf_ids_by_edge', ())
                winner_leaf_count = (
                    len(leaf_ids[int(winner)]) if int(winner) < len(leaf_ids) else 0
                )
                challenger_leaf_count = (
                    len(leaf_ids[challenger]) if challenger < len(leaf_ids) else 0
                )
                exact_leaf_ids = state.get('_adaptive_exact_leaf_ids_by_edge', ())
                winner_exact_count = (
                    len(exact_leaf_ids[int(winner)])
                    if int(winner) < len(exact_leaf_ids) else 0
                )
                challenger_exact_count = (
                    len(exact_leaf_ids[challenger])
                    if challenger < len(exact_leaf_ids) else 0
                )
                state['_adaptive_winner_fresh_leaf_count'] = int(winner_leaf_count)
                state['_adaptive_challenger_fresh_leaf_count'] = int(challenger_leaf_count)
                state['_adaptive_winner_exact_leaf_count'] = int(winner_exact_count)
                state['_adaptive_challenger_exact_leaf_count'] = int(challenger_exact_count)
                state['_adaptive_uncertainty_score'] = _adaptive_uncertainty_score(idx)
                quality = float(tree_reuse_quality[idx]['quality'])
                base_floor = min(
                    dynamic_target,
                    dynamic_minimum
                    + int(round(
                        (1.0 - quality)
                        * min(
                            _DYNAMIC_BUDGET_REUSE_FRESH_FLOOR_BONUS,
                            max(0, dynamic_target - dynamic_minimum),
                        )
                    )) if root_reused_flags[idx] else dynamic_minimum,
                )
                state['_adaptive_fresh_floor'] = int(base_floor)
                action_visits = int(round(float(fresh_visits.sum())))
                if (
                    action_visits >= int(base_floor)
                    and stable_checkpoints >= 2
                    and score_gap >= 0.0
                    and q_gap >= _DYNAMIC_BUDGET_EARLY_STOP_Q_GAP
                    and (
                        winner_leaf_count >= _DYNAMIC_BUDGET_MIN_FRESH_LEAVES_PER_FINALIST
                        or winner_exact_count > 0
                    )
                    and (
                        challenger_leaf_count >= _DYNAMIC_BUDGET_MIN_FRESH_LEAVES_PER_FINALIST
                        or challenger_exact_count > 0
                    )
                ):
                    state['_adaptive_early_stopped'] = True
                    state['_adaptive_stop_reason'] = 'stable_gumbel_winner_q_gap'
                    removed = max(0, int(remaining[idx]))
                    remaining[idx] = 0
                    removed_total += removed
            return removed_total

        def _run_adaptive_extension_passes():
            """Spend the fixed feedback pool in uncertainty-ranked SH passes."""
            if not dynamic_budget_feedback_active or time_budget_active:
                return

            def _actual_fresh(root_idx):
                root = roots[root_idx]
                if root is None:
                    return 0
                return max(
                    0,
                    int(getattr(root, 'visit_count', 0) or 0)
                    - int(initial_root_visits[root_idx]),
                )

            # This is a hard ledger: root setup and all later passes consume
            # the same per-call target pool. A donor's unused allocation can
            # therefore be transferred only once.
            original_pool = int(dynamic_budget_pool_total)
            minimum_pass = min(
                (
                    max(
                        1,
                        int(state.get('considered_actions', 1))
                        * int(math.ceil(math.log2(max(2, int(state.get('considered_actions', 1)))))),
                    )
                    for idx in adaptive_indices
                    if (state := gumbel_root_states[idx]) is not None
                ),
                default=1,
            )
            max_extension_rounds = max(1, original_pool // minimum_pass + 1)
            for _ in range(max_extension_rounds):
                pool = max(
                    0,
                    int(original_pool)
                    - sum(
                        _actual_fresh(idx)
                        for idx in adaptive_indices
                    ),
                )
                if pool <= 0:
                    break
                candidates = [
                    idx for idx in adaptive_indices
                    if (
                        gumbel_root_states[idx] is not None
                        and root_shortcuts[idx] is None
                        and roots[idx] is not None
                        and roots[idx].edges is not None
                        and roots[idx].edges.moves
                        and len(roots[idx].edges.moves) > 1
                        and not gumbel_root_states[idx].get('_adaptive_early_stopped', False)
                        and simulation_budgets[idx] < dynamic_maximum
                    )
                ]
                if not candidates:
                    break

                for idx in candidates:
                    gumbel_root_states[idx]['_adaptive_uncertainty_score'] = (
                        _adaptive_uncertainty_score(idx)
                    )
                # Every observed signal contributes to the same bounded
                # priority. Prior difficulty only breaks exact score ties.
                candidates.sort(
                    key=lambda idx: (
                        -float(gumbel_root_states[idx]['_adaptive_uncertainty_score']),
                        -float(scout_difficulties[idx]),
                        idx,
                    )
                )
                for rank, idx in enumerate(candidates):
                    gumbel_root_states[idx]['_adaptive_uncertainty_rank'] = (
                        float(len(candidates) - rank) / float(len(candidates))
                    )
                grants = {}
                available = int(pool)
                for idx in candidates:
                    state = gumbel_root_states[idx]
                    considered = max(1, int(state.get('considered_actions', 1)))
                    pass_cost = max(
                        1,
                        considered * int(math.ceil(math.log2(considered)))
                        if considered > 1 else 1,
                    )
                    capacity = max(
                        0,
                        int(dynamic_maximum) - int(simulation_budgets[idx]),
                    )
                    if available < pass_cost or capacity < pass_cost:
                        continue
                    grants[idx] = pass_cost
                    available -= pass_cost
                if not grants:
                    break

                pass_remaining = [0] * game_count
                for idx, grant in grants.items():
                    root = roots[idx]
                    state = gumbel_root_states[idx]
                    # Preserve the original considered shortlist across passes;
                    # each grant is a complete balanced SH pass for this width.
                    considered = max(1, int(state['considered_actions']))
                    pass_initial = np.asarray(
                        root.edges.visit_counts,
                        dtype=np.int32,
                    ).copy()
                    state['pass_initial_visits'] = pass_initial
                    state['_pass_initial_visits_f64'] = np.asarray(
                        pass_initial,
                        dtype=np.float64,
                    )
                    state['_pass_initial_total'] = float(pass_initial.sum())
                    state['considered_actions'] = int(considered)
                    state.setdefault('pass_considered_actions', []).append(int(considered))
                    state['pass_budgets'].append(int(grant))
                    state['pass_index'] = len(state['pass_budgets']) - 1
                    state['_pass_scheduled'] = 0
                    state['sequence'] = _gumbel_considered_visit_sequence(
                        considered,
                        grant,
                    )
                    boundaries = _gumbel_revisiting_feedback_boundaries(
                        considered,
                        grant,
                    )
                    if not boundaries or boundaries[-1] != grant:
                        boundaries = tuple(boundaries) + (grant,)
                    state['_feedback_boundaries'] = boundaries
                    state['_feedback_boundary_index'] = 0
                    simulation_budgets[idx] += int(grant)
                    gumbel_action_budgets[idx] += int(grant)
                    state['_adaptive_transferred_in'] = int(
                        state.get('_adaptive_transferred_in', 0)
                    ) + int(grant)
                    pass_remaining[idx] = int(grant)

                _run_simulation_phase(
                    pass_remaining,
                    gumbel_root_states,
                    post_feedback_callback=_adaptive_feedback_callback,
                    native_root_pass_update=[
                        gumbel_root_states[idx]
                        if idx in grants else None
                        for idx in range(game_count)
                    ],
                )
                progressed = 0
                for idx, grant in grants.items():
                    state = gumbel_root_states[idx]
                    actual = max(0, int(state.get('_pass_scheduled', 0)))
                    state.setdefault('pass_actual_visits', []).append(actual)
                    if actual < len(state['sequence']):
                        state['sequence'] = state['sequence'][:actual]
                    progressed += actual
                if progressed <= 0:
                    break

        if time_budget_active:
            time_pass_index = 0
            while True:
                if perf_counter() >= time_deadline:
                    budget_stop_reason = (
                        'deadline_after_root_setup'
                        if time_pass_index == 0
                        else 'deadline'
                    )
                    break

                pass_remaining = [0] * game_count
                pass_budgets = [0] * game_count
                for idx, (root, state) in enumerate(zip(roots, gumbel_root_states)):
                    if (
                        state is None
                        or root is None
                        or root_shortcuts[idx] is not None
                        or not root.edges.moves
                    ):
                        continue
                    pass_budget = max(1, int(gumbel_action_budgets[idx]))
                    pass_budgets[idx] = pass_budget
                    pass_remaining[idx] = pass_budget
                    if time_pass_index > 0:
                        pass_initial = np.asarray(
                            root.edges.visit_counts,
                            dtype=np.int32,
                        ).copy()
                        state['pass_initial_visits'] = pass_initial
                        state['_pass_initial_visits_f64'] = np.asarray(
                            pass_initial,
                            dtype=np.float64,
                        )
                        state['_pass_initial_total'] = float(pass_initial.sum())
                        state['pass_budgets'].append(pass_budget)
                        state.setdefault('pass_considered_actions', []).append(
                            int(state['considered_actions'])
                        )
                    state['pass_index'] = int(time_pass_index)
                    state['_pass_scheduled'] = 0
                    state['sequence'] = _gumbel_considered_visit_sequence(
                        state['considered_actions'],
                        pass_budget,
                    )
                    state['_feedback_boundaries'] = (
                        _gumbel_revisiting_feedback_boundaries(
                            state['considered_actions'],
                            pass_budget,
                        )
                    )
                    state['_feedback_boundary_index'] = 0

                if not any(pass_remaining):
                    budget_stop_reason = 'no_searchable_root'
                    break
                _run_simulation_phase(
                    pass_remaining,
                    gumbel_root_states,
                    post_feedback_callback=_record_stable_gumbel_choices,
                    native_root_pass_update=(
                        [
                            state if (
                                state is not None
                                and int(state.get('pass_index', 0)) == time_pass_index
                                and time_pass_index > 0
                            ) else None
                            for state in gumbel_root_states
                        ]
                        if time_pass_index > 0 else None
                    ),
                    deadline=time_deadline,
                )
                completed_any = False
                incomplete = False
                for idx, state in enumerate(gumbel_root_states):
                    if state is None or pass_budgets[idx] <= 0:
                        continue
                    actual = max(0, int(state.get('_pass_scheduled', 0)))
                    state.setdefault('pass_actual_visits', []).append(actual)
                    if actual >= pass_budgets[idx]:
                        completed_any = True
                        _record_stable_gumbel_choices((idx,))
                    else:
                        incomplete = True
                        state['sequence'] = state['sequence'][:actual]
                if incomplete:
                    budget_stop_reason = (
                        'deadline' if perf_counter() >= time_deadline else 'no_search_progress'
                    )
                    break
                if not completed_any:
                    budget_stop_reason = 'no_search_progress'
                    break
                time_pass_index += 1
        elif effective_root_schedule == 'revisiting':
            max_pass_count = max(
                (len(state['pass_budgets']) for state in gumbel_root_states if state),
                default=0,
            )
            for pass_index in range(max_pass_count):
                pass_remaining = [0] * game_count
                for idx, (root, state) in enumerate(zip(roots, gumbel_root_states)):
                    if (
                        state is None
                        or state.get('_adaptive_early_stopped', False)
                        or pass_index >= len(state['pass_budgets'])
                    ):
                        continue
                    pass_budget = int(state['pass_budgets'][pass_index])
                    pass_remaining[idx] = pass_budget
                    if pass_index > 0:
                        pass_initial = np.asarray(
                            root.edges.visit_counts,
                            dtype=np.int32,
                        ).copy()
                        state['pass_initial_visits'] = pass_initial
                        state['_pass_initial_visits_f64'] = np.asarray(
                            pass_initial,
                            dtype=np.float64,
                        )
                        state['_pass_initial_total'] = float(pass_initial.sum())
                    state['pass_index'] = int(pass_index)
                    state['_pass_scheduled'] = 0
                    state['sequence'] = _gumbel_considered_visit_sequence(
                        state['considered_actions'],
                        pass_budget,
                    )
                    state['_feedback_boundaries'] = (
                        _gumbel_revisiting_feedback_boundaries(
                            state['considered_actions'],
                            pass_budget,
                        )
                    )
                    if dynamic_budget_feedback_active and (
                        not state['_feedback_boundaries']
                        or state['_feedback_boundaries'][-1] != pass_budget
                    ):
                        state['_feedback_boundaries'] = tuple(
                            state['_feedback_boundaries']
                        ) + (pass_budget,)
                    state['_feedback_boundary_index'] = 0

                _run_simulation_phase(
                    pass_remaining,
                    gumbel_root_states,
                    post_batch_callback=reuse_scout_callback,
                    post_feedback_callback=(
                        _adaptive_feedback_callback
                        if dynamic_budget_feedback_active else None
                    ),
                    native_root_pass_update=(
                        [
                            state if (
                                state is not None
                                and int(state.get('pass_index', 0)) == pass_index
                                and pass_index > 0
                            ) else None
                            for state in gumbel_root_states
                        ]
                        if pass_index > 0 else None
                    ),
                )
                for state in gumbel_root_states:
                    if (
                        state is not None
                        and state.get('root_schedule') == 'revisiting'
                        and state.get('pass_index') == pass_index
                    ):
                        actual = max(0, int(state.get('_pass_scheduled', 0)))
                        state.setdefault('pass_actual_visits', []).append(actual)
                        if actual < len(state['sequence']):
                            state['sequence'] = state['sequence'][:actual]
            _run_adaptive_extension_passes()
        else:
            _run_simulation_phase(
                action_budgets,
                gumbel_root_states,
                post_batch_callback=reuse_scout_callback,
                post_feedback_callback=(
                    _adaptive_feedback_callback
                    if dynamic_budget_feedback_active else None
                ),
            )
            for idx, state in enumerate(gumbel_root_states):
                if state is None:
                    continue
                root = roots[idx]
                if root is None:
                    actual = 0
                else:
                    fresh = np.maximum(
                        0.0,
                        root.edges.visit_counts.astype(np.float64)
                        - np.asarray(state['initial_visits'], dtype=np.float64),
                    )
                    actual = int(round(float(fresh.sum())))
                state['pass_actual_visits'] = [actual]
            _run_adaptive_extension_passes()

        if dynamic_budget_feedback_active:
            for state in gumbel_root_states:
                if state is None:
                    continue
                completed_passes = len(state.get('pass_actual_visits', ()))
                state['pass_budgets'] = state.get('pass_budgets', [])[:completed_passes]
                state['pass_considered_actions'] = state.get(
                    'pass_considered_actions', []
                )[:completed_passes]

        if native_forest is not None:
            native_sync_t0 = perf_counter() if profile_timing else None
            native_forest.sync_roots()
            if profile_timing:
                self._profile_add(
                    'native_tree_sync_time',
                    perf_counter() - native_sync_t0,
                )

        if (
            dynamic_budget_feedback_active
            and game_count == 1
            and explicit_native_tree_keys[0]
            and evaluation_generation is not None
            and gumbel_root_states[0] is not None
        ):
            tree_key = native_tree_keys[0]
            root = roots[0]
            fresh_used = max(
                0,
                int(getattr(root, 'visit_count', 0) or 0)
                - int(initial_root_visits[0]),
            ) if root is not None else 0
            earned = 0
            if gumbel_root_states[0].get('_adaptive_early_stopped', False):
                earned = min(
                    max(0, dynamic_maximum - dynamic_target),
                    max(
                        0,
                        int(dynamic_target)
                        + int(prior_dynamic_credit)
                        - int(fresh_used),
                    ),
                )
            if earned > 0:
                self._dynamic_budget_credits[tree_key] = int(earned)
                self._dynamic_budget_credits.move_to_end(tree_key)
                while len(self._dynamic_budget_credits) > _DYNAMIC_BUDGET_CREDIT_KEYS:
                    self._dynamic_budget_credits.popitem(last=False)
            else:
                self._dynamic_budget_credits.pop(tree_key, None)

        # Sync roots back to caller-provided state containers.
        for idx, gs in enumerate(game_states):
            if is_mapping_state[idx]:
                gs['root'] = roots[idx]
                gs['_root_synced'] = bool(root_synced_flags[idx])
            else:
                if len(gs) > 1:
                    gs[1] = roots[idx]
                if len(gs) > 2:
                    gs[2] = bool(root_synced_flags[idx])

        dynamic_budget_pool_used = (
            sum(
                max(
                    0,
                    int(getattr(roots[idx], 'visit_count', 0) or 0)
                    - int(initial_root_visits[idx]),
                )
                for idx in adaptive_indices
                if roots[idx] is not None
            )
            if dynamic_budget_feedback_active else 0
        )
        dynamic_budget_pool_unused = max(
            0,
            int(dynamic_budget_pool_total) - int(dynamic_budget_pool_used),
        )

        result = []
        search_metadata = []
        metadata_t0 = perf_counter() if profile_timing else None
        for idx, root in enumerate(roots):
            fresh_used = max(
                0,
                int(getattr(root, 'visit_count', 0) or 0)
                - int(initial_root_visits[idx]),
            ) if root is not None else 0
            feedback_stats = (
                feedback_eval_stats[idx]
                if feedback_eval_stats is not None
                and idx < len(feedback_eval_stats)
                else None
            )
            feedback_state = gumbel_root_states[idx]
            basic_metadata = {
                'mcts_algorithm': self.mcts_algorithm,
                'simulations_used': int(fresh_used + inherited_visit_credits[idx]),
                'fresh_simulations_used': int(fresh_used),
                'actual_fresh_simulations': int(fresh_used),
                'simulation_budget': int(simulation_budgets[idx]),
                'simulation_planning_quantum': (
                    int(fixed_simulation_budget) if time_budget_active else None
                ),
                'budget_mode': effective_budget_mode,
                'requested_time_limit_seconds': requested_time_limit,
                'elapsed_time_seconds': float(perf_counter() - search_t0),
                'budget_stop_reason': budget_stop_reason,
                'search_diagnostics_included': bool(include_diagnostics),
                'gumbel_configured_root_schedule': self.gumbel_root_schedule,
                'gumbel_effective_root_schedule': effective_root_schedule,
                'dynamic_budget_enabled': bool(
                    dynamic_budget_active and budget_overrides[idx] is None
                ),
                'dynamic_budget_strategy': self.dynamic_budget_strategy,
                'dynamic_budget_initial_budget': (
                    int(dynamic_budget_initial_budgets[idx])
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else None
                ),
                'dynamic_budget_transferred_in': (
                    int(feedback_state.get('_adaptive_transferred_in', 0))
                    if dynamic_budget_feedback_active and feedback_state is not None
                    else 0
                ),
                'dynamic_budget_extra_simulations_used': (
                    max(
                        0,
                        int(fresh_used)
                        - int(dynamic_budget_initial_budgets[idx]),
                    )
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_saved_simulations': (
                    max(0, int(simulation_budgets[idx]) - int(fresh_used))
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_pool_total': (
                    int(dynamic_budget_pool_total)
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_pool_used': (
                    int(dynamic_budget_pool_used)
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_pool_unused': (
                    int(dynamic_budget_pool_unused)
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_pool_overrun': (
                    max(
                        0,
                        int(dynamic_budget_pool_used)
                        - int(dynamic_budget_pool_total),
                    )
                    if dynamic_budget_feedback_active
                    and budget_overrides[idx] is None else 0
                ),
                'dynamic_budget_spent_credit': (
                    min(
                        int(prior_dynamic_credit),
                        max(0, int(fresh_used) - int(fixed_simulation_budget)),
                    )
                    if dynamic_budget_feedback_active
                    and game_count == 1
                    and explicit_native_tree_keys[idx] else 0
                ),
                'nn_fresh_evaluations': (
                    _provenance_count_or_none(
                        feedback_stats,
                        'nn_fresh_evaluations',
                    )
                ),
                'nn_cache_hits': (
                    _provenance_count_or_none(feedback_stats, 'nn_cache_hits')
                ),
                'nn_deduplicated_evaluations': (
                    _provenance_count_or_none(
                        feedback_stats,
                        'nn_deduplicated_evaluations',
                    )
                ),
                'nn_unknown_rows': (
                    int(feedback_stats['nn_unknown_rows'])
                    if feedback_stats is not None else None
                ),
                'nn_provenance_complete': (
                    bool(feedback_stats['nn_provenance_complete'])
                    if feedback_stats is not None else False
                ),
                'mcts_duplicate_leaf_selections': (
                    int(feedback_stats['mcts_duplicate_leaf_selections'])
                    if feedback_stats is not None else None
                ),
                'inherited_leaf_evaluations': (
                    int(feedback_stats['inherited_leaf_evaluations'])
                    if feedback_stats is not None else None
                ),
                'exact_leaf_evaluations': (
                    int(feedback_stats['exact_leaf_evaluations'])
                    if feedback_stats is not None else None
                ),
                'dynamic_budget_uncertainty_score': (
                    float(feedback_state.get('_adaptive_uncertainty_score', 0.0))
                    if dynamic_budget_feedback_active and feedback_state is not None
                    else None
                ),
                'dynamic_budget_uncertainty_rank': (
                    float(feedback_state.get('_adaptive_uncertainty_rank', 0.0))
                    if dynamic_budget_feedback_active and feedback_state is not None
                    else None
                ),
                'dynamic_budget_minimum': (
                    int(dynamic_minimum) if dynamic_budget_active else None
                ),
                'dynamic_budget_target': (
                    int(dynamic_target) if dynamic_budget_active else None
                ),
                'dynamic_budget_maximum': (
                    int(dynamic_maximum) if dynamic_budget_active else None
                ),
                'dynamic_budget_early_stopped': bool(
                    gumbel_root_states[idx].get('_adaptive_early_stopped', False)
                    if gumbel_root_states[idx] is not None else False
                ),
                'dynamic_budget_stop_reason': (
                    gumbel_root_states[idx].get('_adaptive_stop_reason', 'budget_exhausted')
                    if dynamic_budget_feedback_active and gumbel_root_states[idx] is not None
                    else 'fixed_override'
                    if budget_overrides[idx] is not None
                    else 'prior_allocation_complete'
                    if dynamic_budget_active
                    else 'disabled'
                ),
                'dynamic_budget_q_gap': (
                    float(gumbel_root_states[idx].get('_adaptive_q_gap', 0.0))
                    if dynamic_budget_feedback_active
                    and gumbel_root_states[idx] is not None else None
                ),
                'dynamic_budget_score_gap': (
                    gumbel_root_states[idx].get('_adaptive_score_gap')
                    if dynamic_budget_feedback_active
                    and gumbel_root_states[idx] is not None else None
                ),
                'dynamic_budget_stable_checkpoints': (
                    int(gumbel_root_states[idx].get('_adaptive_stable_checkpoints', 0))
                    if dynamic_budget_feedback_active
                    and gumbel_root_states[idx] is not None else None
                ),
                'dynamic_budget_fresh_floor': (
                    int(gumbel_root_states[idx].get('_adaptive_fresh_floor', dynamic_minimum))
                    if dynamic_budget_feedback_active and gumbel_root_states[idx] is not None
                    else None
                ),
                'dynamic_budget_distinct_new_leaf_evaluations': (
                    sum(
                        len(edge_leaf_ids)
                        for edge_leaf_ids in gumbel_root_states[idx].get(
                            '_adaptive_fresh_leaf_ids_by_edge', ()
                        )
                    )
                    if dynamic_budget_feedback_active and gumbel_root_states[idx] is not None
                    else None
                ),
                'dynamic_budget_actual_fresh_simulations': (
                    int(fresh_used) if dynamic_budget_active else None
                ),
                'dynamic_budget_earned_credit': (
                    int(self._dynamic_budget_credits.get(native_tree_keys[idx], 0))
                    if (
                        dynamic_budget_active
                        and game_count == 1
                        and explicit_native_tree_keys[idx]
                    )
                    else 0
                ),
            }
            shortcut = root_shortcuts[idx]
            if shortcut is not None:
                selected_move = shortcut['selected_move']
                if root is not None and root.expanded and root.edges is not None:
                    root_result = self._fresh_child_visit_dict(
                        root,
                        initial_child_visits[idx],
                    )
                else:
                    root_result = {move: 0.0 for move in shortcut['legal_moves']}
                if not root_result:
                    root_result = {move: 0.0 for move in shortcut['legal_moves']}
                basic_metadata['selected_move_override'] = selected_move
                basic_metadata['budget_stop_reason'] = shortcut['stop_reason']
                basic_metadata['root_shortcut'] = shortcut['stop_reason']
                basic_metadata['selected_move_decision_source'] = (
                    'syzygy_root' if shortcut['best_value'] is not None else 'forced_move'
                )
                if shortcut['best_value'] is not None:
                    basic_metadata['syzygy_root_value'] = shortcut['best_value']
                result.append(root_result)
                search_metadata.append(basic_metadata)
                continue

            if not include_diagnostics:
                root_result = self._fresh_child_visit_dict(
                    root,
                    initial_child_visits[idx],
                )
                if not root_result and root is not None and root.expanded:
                    root_result = {move: 0.0 for move in root.edges.moves}
                selected_move = None
                state = gumbel_root_states[idx]
                if root is not None and state is not None and root.edges.moves:
                    stable_idx = state.get('_stable_selected_idx')
                    use_stable = bool(
                        time_budget_active
                        and budget_stop_reason.startswith('deadline')
                        and stable_idx is not None
                    )
                    selected_idx = (
                        int(stable_idx)
                        if use_stable
                        else self._gumbel_final_action_index(root, state)
                    )
                    if selected_idx is not None:
                        selected_move = root.edges.moves[selected_idx]
                        basic_metadata['selected_move_override'] = selected_move
                    if use_stable:
                        basic_metadata['budget_decision_source'] = (
                            'last_completed_feedback_checkpoint'
                        )
                        basic_metadata['budget_decision_checkpoint_pass'] = int(
                            state.get('_stable_checkpoint_pass', -1)
                        )
                        basic_metadata['budget_decision_checkpoint_visit'] = int(
                            state.get('_stable_checkpoint_visit', 0)
                        )
                if state is not None:
                    basic_metadata['gumbel_root_schedule'] = str(
                        state.get('root_schedule', effective_root_schedule)
                    )
                    basic_metadata['gumbel_action_budget'] = int(
                        gumbel_action_budgets[idx]
                    )
                    basic_metadata['gumbel_pass_count'] = len(
                        state.get('pass_actual_visits', ())
                    )
                    basic_metadata['gumbel_pass_budgets'] = [
                        int(value) for value in state.get('pass_budgets', ())
                    ]
                    basic_metadata['gumbel_pass_actual_visits'] = [
                        int(value) for value in state.get('pass_actual_visits', ())
                    ]
                result.append(root_result)
                search_metadata.append(basic_metadata)
                continue

            metadata = self._summarize_root_search(
                root,
                max(int(simulation_budgets[idx]), int(fresh_used)),
                initial_root_visits=initial_root_visits[idx],
                initial_child_visits=initial_child_visits[idx],
                inherited_visit_credit=inherited_visit_credits[idx],
            )
            metadata.update(basic_metadata)
            metadata['mcts_algorithm'] = self.mcts_algorithm
            metadata['dynamic_budget_enabled'] = bool(
                dynamic_budget_active and budget_overrides[idx] is None
            )
            metadata['eval_easy_cut_enabled'] = bool(eval_easy_cut_active)
            metadata['eval_easy_cut_reduced'] = bool(
                eval_easy_cut_active
                and int(simulation_budgets[idx]) < int(fixed_simulation_budget)
            )
            metadata['eval_easy_cut_requested_budget'] = int(fixed_simulation_budget)
            metadata['simulation_budget_overridden'] = budget_overrides[idx] is not None
            metadata['tree_reuse_attempted'] = bool(
                self.reuse_tree and incoming_root_present[idx]
            )
            metadata['tree_reused'] = bool(root_reused_flags[idx])
            metadata['tree_inherited_visits'] = (
                int(initial_root_visits[idx]) if root_reused_flags[idx] else 0
            )
            metadata['root_outgoing_inherited_visits'] = (
                int(np.asarray(initial_child_visits[idx], dtype=np.int64).sum())
                if root_reused_flags[idx] and initial_child_visits[idx] is not None
                else 0
            )
            metadata['mcts_gumbel_reuse_q_scale'] = self.gumbel_reuse_q_scale
            reuse_quality = tree_reuse_quality[idx]
            metadata['tree_reuse_quality'] = float(reuse_quality['quality'])
            metadata['tree_reuse_candidate_coverage'] = float(
                reuse_quality['candidate_coverage']
            )
            metadata['tree_reuse_candidate_support'] = float(
                reuse_quality['candidate_support']
            )
            metadata['tree_reuse_visited_prior_mass'] = float(
                reuse_quality['visited_prior_mass']
            )
            metadata['tree_reuse_effective_action_ratio'] = float(
                reuse_quality['effective_action_ratio']
            )
            scout_result = tree_reuse_scout_results[idx]
            metadata['tree_reuse_fresh_floor'] = int(scout_result['fresh_floor'])
            metadata['tree_reuse_scout_simulations'] = int(
                scout_result['scout_simulations']
            )
            metadata['tree_reuse_scout_winner_stable'] = bool(
                scout_result['winner_stable']
            )
            metadata['tree_reuse_scout_policy_js'] = float(
                scout_result['policy_js_divergence']
            )
            metadata['tree_reuse_scout_stability'] = float(
                scout_result['combined_stability']
            )
            metadata['tree_reuse_scout_extra_credit'] = int(
                scout_result['extra_credit']
            )
            metadata['tree_reuse_scout_search_reduced'] = bool(
                scout_result['search_reduced']
            )
            metadata['dynamic_budget_difficulty'] = float(scout_difficulties[idx])
            metadata['dynamic_budget_target_average'] = (
                float(dynamic_target)
                if dynamic_budget_active or eval_easy_cut_active
                else int(fixed_simulation_budget)
            )
            fresh_visits = self._fresh_child_visit_dict(root, initial_child_visits[idx])
            root_result = root.child_visit_dict() if root is not None else {}
            gumbel_state = gumbel_root_states[idx]
            if root is not None and gumbel_state is not None and root.edges.moves:
                fresh_count_vector = np.maximum(
                    0.0,
                    root.edges.visit_counts.astype(np.float64)
                    - np.asarray(gumbel_state['initial_visits'], dtype=np.float64),
                )
                improved = self._gumbel_improved_policy(
                    root,
                    scale_visit_counts=self._gumbel_root_q_scale_visit_counts(
                        root,
                        gumbel_state,
                    ),
                )
                policy_target_probs = self._gumbel_policy_target(improved)
                stable_idx = gumbel_state.get('_stable_selected_idx')
                use_stable = bool(
                    time_budget_active
                    and budget_stop_reason.startswith('deadline')
                    and stable_idx is not None
                )
                selected_idx = (
                    int(stable_idx)
                    if use_stable
                    else self._gumbel_final_action_index(root, gumbel_state)
                )
                selected_move = root.edges.moves[selected_idx] if selected_idx is not None else None
                if use_stable:
                    metadata['budget_decision_source'] = (
                        'last_completed_feedback_checkpoint'
                    )
                    metadata['budget_decision_checkpoint_pass'] = int(
                        gumbel_state.get('_stable_checkpoint_pass', -1)
                    )
                    metadata['budget_decision_checkpoint_visit'] = int(
                        gumbel_state.get('_stable_checkpoint_visit', 0)
                    )
                    metadata['budget_decision_score_source'] = (
                        'checkpoint_statistics; per_move_scores_are_final_statistics'
                    )
                completed_q = self._gumbel_raw_completed_qvalues(root)
                normalized_completed_q = self._gumbel_normalized_completed_qvalues(root)
                selection_scores = root.edges.log_base_priors.astype(np.float64, copy=True)
                selection_scores -= float(np.max(selection_scores))
                selection_scores += np.asarray(gumbel_state['gumbel'], dtype=np.float64)
                selection_scores += normalized_completed_q * (
                    (
                        self.gumbel_c_visit
                        + float(np.max(self._gumbel_root_q_scale_visit_counts(root, gumbel_state)))
                    )
                    * self.gumbel_c_scale
                )
                final_visit_counts = _gumbel_final_visit_counts(root, gumbel_state)
                eligible, _ = _gumbel_final_action_eligibility(final_visit_counts)
                selection_scores[~eligible] = -np.inf
                policy_target = {
                    move: float(probability)
                    for move, probability in zip(root.edges.moves, policy_target_probs)
                    if float(probability) > 0.0
                }
                if policy_target:
                    metadata['policy_target_probs_override'] = policy_target
                    metadata['policy_target_by_move'] = policy_target
                metadata['gumbel_considered_actions'] = int(
                    gumbel_state.get('considered_actions', 0) or 0
                )
                metadata['gumbel_action_budget'] = int(gumbel_action_budgets[idx])
                metadata['gumbel_root_schedule'] = str(
                    gumbel_state.get('root_schedule', 'standard')
                )
                if gumbel_state.get('root_schedule') == 'revisiting':
                    metadata['tree_reuse_post_scout_credit_disabled'] = True
                    metadata['tree_reuse_post_scout_credit_disabled_reason'] = (
                        'fixed_revisiting_pass_budget'
                    )
                metadata['gumbel_pass_count'] = int(
                    len(gumbel_state.get('pass_budgets', ()))
                )
                metadata['gumbel_pass_budgets'] = [
                    int(value)
                    for value in gumbel_state.get('pass_budgets', ())
                ]
                metadata['gumbel_pass_considered_actions'] = [
                    int(value)
                    for value in gumbel_state.get('pass_considered_actions', ())
                ]
                metadata['gumbel_pass_actual_visits'] = [
                    int(value)
                    for value in gumbel_state.get('pass_actual_visits', ())
                ]
                metadata['gumbel_pass_initial_round_visits'] = [
                    int(
                        _gumbel_initial_round_visits(
                            gumbel_state.get('considered_actions', 0),
                            pass_budget,
                        )
                    )
                    for pass_budget in gumbel_state.get('pass_budgets', ())
                ]
                metadata['gumbel_final_pass_index'] = int(
                    gumbel_state.get('pass_index', 0)
                )
                metadata['gumbel_initial_round_visits'] = int(
                    (
                        metadata['gumbel_pass_initial_round_visits'][0]
                        if (
                            gumbel_state.get('root_schedule') == 'revisiting'
                            and metadata['gumbel_pass_initial_round_visits']
                        )
                        else _gumbel_initial_round_visits(
                            gumbel_state.get('considered_actions', 0),
                            gumbel_action_budgets[idx],
                        )
                    )
                )
                metadata['gumbel_width_adapted'] = bool(
                    int(gumbel_state.get('considered_actions', 0) or 0)
                    < min(len(root.edges.moves), self.gumbel_max_considered_actions)
                )
                metadata['gumbel_scale'] = float(
                    self.gumbel_scale if add_root_noise else self.gumbel_eval_scale
                )
                metadata['gumbel_q_range_floor'] = float(self.gumbel_q_range_floor)
                metadata['gumbel_target_temperature'] = float(self.gumbel_target_temperature)
                if selected_move is not None:
                    metadata['selected_move_override'] = selected_move
                metadata['completed_q_by_move'] = {
                    move: float(value)
                    for move, value in zip(root.edges.moves, completed_q)
                }
                metadata['improved_policy_by_move'] = {
                    move: float(value)
                    for move, value in zip(root.edges.moves, improved)
                }
                metadata['selection_score_by_move'] = {
                    move: float(value)
                    for move, value in zip(root.edges.moves, selection_scores)
                }
                metadata['selection_score_by_move_source'] = 'final_search_statistics'

                # Full Gumbel trains against the temperature-scaled completed-Q
                # target; target telemetry and Q labels must use that exact map.
                priors = root.edges.base_priors.astype(np.float64, copy=False)
                prior_total = float(priors.sum())
                prior_probs = (
                    priors / prior_total
                    if prior_total > 0.0
                    else np.full_like(priors, 1.0 / len(priors))
                )
                prior_top_idx = int(np.argmax(prior_probs))
                target_probs = policy_target_probs.astype(np.float64, copy=False)
                original_changed_top = prior_top_idx != int(np.argmax(target_probs))
                metadata['selected_prior_agree'] = (
                    None
                    if selected_idx is None
                    else (1.0 if prior_top_idx == selected_idx else 0.0)
                )
                metadata['selected_q_delta'] = None
                cumulative = root.edges.visit_counts.astype(np.float64, copy=False)
                target_telemetry = _gumbel_policy_target_telemetry(
                    target_probs,
                    prior_probs,
                    cumulative,
                    root.edges.value_sums,
                )
                if target_telemetry is not None:
                    metadata.update(target_telemetry)
                else:
                    _clear_gumbel_policy_target_telemetry(metadata)
                if (
                    selected_idx is not None
                    and cumulative[prior_top_idx] > 0.0
                    and cumulative[selected_idx] > 0.0
                ):
                    prior_q = -float(root.edges.value_sums[prior_top_idx]) / cumulative[prior_top_idx]
                    selected_q = -float(root.edges.value_sums[selected_idx]) / cumulative[selected_idx]
                    metadata['selected_q_delta'] = float(
                        max(-2.0, min(2.0, selected_q - prior_q))
                    )

                # Transform eligibility uses the original improved target and its
                # measured Q evidence. Only after an applied transformation do we
                # recalculate target telemetry for the distribution stored in replay.
                if self.policy_target_mode == 'challenge_delta' and policy_target:
                    prior_by_move = {
                        move: float(probability)
                        for move, probability in zip(root.edges.moves, prior_probs)
                    }
                    transformed_target, intervention = _build_challenge_delta_policy_target(
                        policy_target,
                        prior_by_move,
                        changed_top=original_changed_top,
                        q_delta=metadata.get('mcts_q_delta'),
                        prior_fraction=self.challenge_delta_prior_fraction,
                    )
                    if intervention['applied']:
                        metadata['policy_target_probs_override'] = transformed_target
                        metadata['policy_target_by_move'] = transformed_target
                        metadata['policy_target_intervention_applied'] = True
                        metadata['policy_target_removed_prior_mass'] = float(
                            intervention['removed_prior_mass']
                        )
                        transformed_probs = np.asarray([
                            float(transformed_target.get(move, 0.0))
                            for move in root.edges.moves
                        ], dtype=np.float64)
                        transformed_telemetry = _gumbel_policy_target_telemetry(
                            transformed_probs,
                            prior_probs,
                            cumulative,
                            root.edges.value_sums,
                        )
                        if transformed_telemetry is not None:
                            metadata.update(transformed_telemetry)
                        else:
                            _clear_gumbel_policy_target_telemetry(metadata)

                root_result = {
                    move: float(count)
                    for move, count in fresh_visits.items()
                    if float(count) > 0.0
                }
            elif fresh_visits:
                metadata['policy_visit_counts_override'] = fresh_visits
            proof = root_proofs[idx]
            if proof is not None:
                metadata['mcts_selected_move'] = metadata.get(
                    'selected_move_override'
                )
                metadata['selected_move_override'] = proof['selected_move']
                metadata['selected_move_decision_source'] = 'syzygy_root'
                metadata['selection_score_applies_to_selected_move'] = False
                metadata['syzygy_root_value'] = proof['best_value']
                metadata['syzygy_root_decision_reason'] = proof['stop_reason']
            result.append(root_result)
            search_metadata.append(metadata)
        if profile_timing:
            self._profile_add('search_metadata_time', perf_counter() - metadata_t0)
        self._verify_search_generation(
            evaluation_generation,
            roots,
            game_states,
            is_mapping_state,
        )
        elapsed_time_seconds = float(perf_counter() - search_t0)
        for metadata in search_metadata:
            metadata['elapsed_time_seconds'] = elapsed_time_seconds
        self._profile_add('search_many_time', perf_counter() - search_t0)
        self._profile_inc('search_many_calls', 1)
        if return_search_metadata:
            return result, search_metadata
        return result

    def search_results_many(
        self,
        game_states,
        num_simulations,
        add_root_noise=False,
        *,
        time_limit_seconds=None,
        budget_mode=None,
        include_diagnostics=True,
    ):
        """Return complete root-search decisions for public MCTS consumers."""
        visits, metadata = self.search_many(
            game_states,
            num_simulations=num_simulations,
            add_root_noise=add_root_noise,
            return_search_metadata=True,
            time_limit_seconds=time_limit_seconds,
            budget_mode=budget_mode,
            include_diagnostics=include_diagnostics,
        )
        return [
            SearchResult.from_legacy_result(root_visits, root_metadata)
            for root_visits, root_metadata in zip(visits, metadata)
        ]

    def release_native_tree(self, tree_key) -> None:
        self._release_native_tree(tree_key, preserve_budget_credit=False)

    def release_native_tree_preserving_budget_credit(self, tree_key) -> None:
        """Drop a tree after an unsearched move without losing game credit."""
        self._release_native_tree(tree_key, preserve_budget_credit=True)

    def _release_native_tree(self, tree_key, *, preserve_budget_credit):
        if not preserve_budget_credit:
            self._dynamic_budget_credits.pop(tree_key, None)
        forest = self._native_forest
        if forest is not None and forest.active:
            forest.release_tree(tree_key)

    def _sync_native_node_for_promotion(self, node, tree_key) -> None:
        """Refresh one retained descendant before it is promoted without search.

        Native backups update every traversed edge, while Python edge arrays are
        eagerly synchronized only for active roots. A game can make multiple
        actual moves between searches, so the parent of a later promoted child
        may still be a stale Python descendant. Read that node's edge arrays
        from the retained forest before detach copies the incoming edge into
        the new root. Do not replace the node's own root totals: those are
        distinct from its parent-edge statistics.
        """
        forest = self._native_forest
        if forest is None or not forest.active:
            return
        node_id = forest._node_ids.get(id(node))
        active_root = forest._active_roots.get(tree_key)
        if node_id is None or active_root is None:
            return
        if int(active_root[1]) == int(node_id):
            # The just-searched active root was already exported at search end.
            return
        forest.sync_node(node, sync_root_stats=False)

    def _batch_expand_and_evaluate(
        self,
        nodes,
        game_indices,
        board_histories,
        root_position_counts,
        root_repetition_known,
        adaptive_stats=None,
        evaluation_trace=None,
    ):
        """
        Batch expansion + evaluation for leaf nodes across games.
        """
        perf_counter = time.perf_counter
        profile_detail = self.profile_detail_enabled
        profile_timing = self.profile_enabled
        eval_t0 = perf_counter() if profile_timing else None
        # The same leaf can appear multiple times in one batch.
        # Evaluate each unique node once and fan-out value to duplicates.
        dedup_t0 = perf_counter() if profile_timing else None
        node_occurrences = {}
        unique_entries = []
        for idx, node in enumerate(nodes):
            node_id = id(node)
            if node_id not in node_occurrences:
                node_occurrences[node_id] = [idx]
                unique_entries.append((node, game_indices[idx]))
            else:
                node_occurrences[node_id].append(idx)
        if adaptive_stats is not None:
            seen_root_nodes = set()
            for node, gi in zip(nodes, game_indices):
                key = (int(gi), id(node))
                if key in seen_root_nodes:
                    stats = adaptive_stats[int(gi)]
                    if stats is not None:
                        stats['mcts_duplicate_leaf_selections'] += 1
                else:
                    seen_root_nodes.add(key)
        node_sources = {} if evaluation_trace is not None else None
        inherited_node_ids = set()
        if profile_timing:
            self._profile_add('batch_expand_dedup_terminal_time', perf_counter() - dedup_t0)

        terminal_values = {}
        non_terminal_nodes = []
        non_terminal_game_indices = []
        legal_moves_per_node = []
        legal_indices_per_node = []
        legal_counts = []
        max_legal_count = 0
        for node, gi in unique_entries:
            if node.raw_value is not None or node.expanded:
                inherited_node_ids.add(id(node))
            # A promoted root still needs children/visits so callers can choose
            # a move. Reuse exact terminal values only while the node is a
            # simulated descendant.
            if node.parent is not None and node.terminal_value is not None:
                terminal_values[id(node)] = float(node.terminal_value)
                continue
            board_t0, board_scale = self._profile_sample_begin('board_materialize')
            board = node.board
            self._profile_sample_finish('board_materialize', board_t0, board_scale)
            terminal_t0, terminal_scale = self._profile_sample_begin('terminal_checks')
            is_check = chess.is_check(board)
            is_path_draw = _is_search_path_draw_candidate(
                node,
                board,
                root_position_counts[gi],
            )
            self._profile_sample_finish('terminal_checks', terminal_t0, terminal_scale)
            # A non-check draw cannot be checkmate, so avoid generating and
            # encoding legal moves for a leaf whose value is already known.
            if is_path_draw and not is_check:
                terminal_values[id(node)] = 0.0
                continue

            # Generating legal moves is the dominant CPU operation here. Do it
            # once, after every safe pre-check, and reuse the encoded indices.
            legal_t0, legal_scale = self._profile_sample_begin('batch_expand_legal_moves')
            legal_moves = node.get_legal_moves()
            self._profile_sample_finish('batch_expand_legal_moves', legal_t0, legal_scale)
            legal_count = len(legal_moves)
            if legal_count <= 0:
                node.terminal_value = -1.0 if is_check else 0.0
                terminal_values[id(node)] = node.terminal_value
                continue
            if is_path_draw:
                terminal_values[id(node)] = 0.0
                continue

            # Probe simulated leaves only. A terminal root would have no child
            # visit distribution for UI/eval callers to play from; the
            # self-play game loop handles real-root tablebase results.
            syzygy_value = (
                self.syzygy.probe_value(board)
                if node.parent is not None and self.syzygy is not None
                else None
            )
            if syzygy_value is not None:
                node.terminal_value = float(syzygy_value)
                terminal_values[id(node)] = node.terminal_value
                continue
            non_terminal_nodes.append(node)
            non_terminal_game_indices.append(gi)
            legal_moves_per_node.append(legal_moves)
            if not self._native_move_indices_enabled:
                index_t0, index_scale = self._profile_sample_begin('batch_expand_move_index')
                legal_indices = node.get_legal_indices()
                self._profile_sample_finish('batch_expand_move_index', index_t0, index_scale)
                legal_indices_per_node.append(legal_indices)
            legal_counts.append(legal_count)
            if legal_count > max_legal_count:
                max_legal_count = legal_count
        if (
            getattr(self, '_native_move_indices_enabled', False)
            and non_terminal_nodes
        ):
            index_t0 = perf_counter() if profile_timing else None
            legal_indices_per_node = self._encode_legal_indices_native(
                non_terminal_nodes,
                legal_moves_per_node,
            )
            if profile_timing:
                self._profile_add(
                    'batch_expand_move_index_time',
                    perf_counter() - index_t0,
                )
        values_by_node_id = {}

        if adaptive_stats is not None:
            for node, gi in unique_entries:
                stats = adaptive_stats[int(gi)]
                if stats is not None and id(node) in inherited_node_ids:
                    stats['inherited_leaf_evaluations'] += 1
                    if node_sources is not None:
                        node_sources[id(node)] = 'inherited'
            for node, gi in unique_entries:
                node_id = id(node)
                if node_id in terminal_values:
                    stats = adaptive_stats[int(gi)]
                    if stats is not None:
                        stats['exact_leaf_evaluations'] += 1
                    if node_sources is not None:
                        node_sources[node_id] = 'exact_terminal'

        if non_terminal_nodes:
            batch_n = len(non_terminal_nodes)
            boards_np = self._get_board_input_scratch(batch_n)
            tensor_pack_t0 = perf_counter() if profile_timing else None
            if self._native_input_planes_enabled:
                self._pack_current_tensors_native(
                    non_terminal_nodes,
                    boards_np[:, self._history_planes:, :, :],
                )
            if self.history_positions > 0:
                self._fill_history_prefixes_for_nodes(
                    non_terminal_nodes,
                    non_terminal_game_indices,
                    board_histories,
                    root_position_counts,
                    root_repetition_known,
                    boards_np[:, :self._history_planes, :, :],
                )
            for row_idx, (node, gi) in enumerate(zip(non_terminal_nodes, non_terminal_game_indices)):
                if not self._native_input_planes_enabled:
                    repetition_count = (
                        _repetition_count_before_node(
                            node,
                            root_position_counts[gi],
                            root_repetition_known[gi],
                        )
                        if self._path_dependent_repetition
                        else None
                    )
                    self._current_tensor_for_node(
                        node,
                        out=boards_np[row_idx, self._history_planes:, :, :],
                        repetition_count=repetition_count,
                    )
            if profile_timing:
                self._profile_add('batch_expand_tensor_pack_time', perf_counter() - tensor_pack_t0)

            remote_legal_gather = bool(getattr(self.model, 'supports_remote_legal_gather', False))
            use_cuda_stage_timing = (
                self.cuda_stage_profile_enabled
                and self.device.type == 'cuda'
                and torch.cuda.is_available()
            )

            def _cuda_event():
                if not use_cuda_stage_timing:
                    return None
                return torch.cuda.Event(enable_timing=True)

            h2d_start = _cuda_event()
            h2d_end = _cuda_event()
            if remote_legal_gather:
                board_tensors = boards_np
            else:
                if h2d_start is not None:
                    h2d_start.record()
                board_tensors = self._get_board_input_source_tensor(boards_np).to(
                    self.device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
                if h2d_end is not None:
                    h2d_end.record()

            legal_index_matrix = None
            if max_legal_count > 0:
                legal_pack_t0 = perf_counter() if profile_timing else None
                legal_index_matrix = self._get_legal_index_scratch(
                    len(non_terminal_nodes),
                    max_legal_count,
                    dtype=np.int16 if remote_legal_gather else np.int64,
                )
                legal_index_matrix.fill(0)
                for row_idx, legal_indices in enumerate(legal_indices_per_node):
                    legal_count = legal_counts[row_idx]
                    if legal_count:
                        legal_index_matrix[row_idx, :legal_count] = legal_indices
                if profile_timing:
                    self._profile_add('batch_expand_legal_index_pack_time', perf_counter() - legal_pack_t0)

            inference_t0 = perf_counter()
            h2d_legal_start = None
            h2d_legal_end = None
            wdl_start = None
            wdl_end = None
            gather_start = None
            gather_end = None
            forward_start = _cuda_event()
            forward_end = _cuda_event()
            if forward_start is not None:
                forward_start.record()
            model_kwargs = {'apply_log_softmax': False}
            prediction_provenance = {} if adaptive_stats is not None else None
            if (
                legal_index_matrix is not None
                and remote_legal_gather
            ):
                model_kwargs['legal_index_matrix'] = legal_index_matrix
                model_kwargs['legal_counts'] = np.asarray(legal_counts, dtype=np.int16)
            with torch.inference_mode():
                policy_logits_batch, values_batch, used_local_forward_runner = (
                    self._local_model_forward(
                        board_tensors,
                        model_kwargs,
                        provenance=prediction_provenance,
                    )
                )
                if forward_end is not None:
                    forward_end.record()
                compact_policy_response = (
                    False
                    if used_local_forward_runner
                    else bool(getattr(self.model, 'last_response_compact_policy', False))
                )
                
                if values_batch.ndim == 2 and values_batch.shape[1] == 3:
                    wdl_start = _cuda_event()
                    wdl_end = _cuda_event()
                    if wdl_start is not None:
                        wdl_start.record()
                    values_batch = value_logits_to_scalar(values_batch)
                    if wdl_end is not None:
                        wdl_end.record()
                else:
                    values_batch = value_logits_to_scalar(values_batch)

                # Gather only legal move logits on GPU before transferring to CPU.
                if max_legal_count > 0:
                    if compact_policy_response:
                        legal_logits_batch = policy_logits_batch.to(dtype=torch.float16).cpu().numpy()
                    else:
                        h2d_legal_start = _cuda_event()
                        h2d_legal_end = _cuda_event()
                        if h2d_legal_start is not None:
                            h2d_legal_start.record()
                        legal_index_tensor = self._get_legal_index_source_tensor(legal_index_matrix).to(
                            self.device,
                            non_blocking=True,
                        )
                        if h2d_legal_end is not None:
                            h2d_legal_end.record()
                        gather_start = _cuda_event()
                        gather_end = _cuda_event()
                        if gather_start is not None:
                            gather_start.record()
                        legal_logits_batch = torch.gather(policy_logits_batch, 1, legal_index_tensor)
                        if gather_end is not None:
                            gather_end.record()
                else:
                    legal_logits_batch = None

                d2h_legal_start = None
                d2h_legal_end = None
                d2h_values_start = None
                d2h_values_end = None
                if (
                    isinstance(legal_logits_batch, torch.Tensor)
                    and isinstance(values_batch, torch.Tensor)
                    and legal_logits_batch.device.type == 'cuda'
                    and values_batch.device == legal_logits_batch.device
                    and not remote_legal_gather
                    and not compact_policy_response
                ):
                    d2h_legal_start = _cuda_event()
                    d2h_legal_end = _cuda_event()
                    if d2h_legal_start is not None:
                        d2h_legal_start.record()
                    pinned_policy, pinned_values, transfer_event = (
                        self._local_cuda_readback_buffers(
                            int(legal_logits_batch.shape[0]),
                            int(legal_logits_batch.shape[1]),
                            legal_logits_batch.device,
                        )
                    )
                    legal_logits_batch, values_batch = copy_compact_policy_value_to_cpu(
                        legal_logits_batch,
                        values_batch,
                        pinned_policy=pinned_policy,
                        pinned_values=pinned_values,
                        transfer_event=transfer_event,
                    )
                    if d2h_legal_end is not None:
                        d2h_legal_end.record()
                else:
                    if isinstance(legal_logits_batch, torch.Tensor):
                        d2h_legal_start = _cuda_event()
                        d2h_legal_end = _cuda_event()
                        if d2h_legal_start is not None:
                            d2h_legal_start.record()
                        legal_logits_batch = legal_logits_batch.to(
                            dtype=torch.float16
                        ).cpu().numpy()
                        if d2h_legal_end is not None:
                            d2h_legal_end.record()
                    if isinstance(values_batch, torch.Tensor):
                        d2h_values_start = _cuda_event()
                        d2h_values_end = _cuda_event()
                        if d2h_values_start is not None:
                            d2h_values_start.record()
                        values_batch = values_batch.float().cpu().numpy()
                        if d2h_values_end is not None:
                            d2h_values_end.record()
            self._profile_inc('nn_inference_calls', 1)
            self._profile_inc('nn_inference_batch_items', batch_n)
            self._profile_inc('nn_legal_move_items', sum(legal_counts))
            self._profile_add('nn_inference_time', perf_counter() - inference_t0)
            if adaptive_stats is not None:
                row_sources = (
                    prediction_provenance.get('row_sources', ())
                    if isinstance(prediction_provenance, dict)
                    else ()
                )
                for row_idx, (node, gi) in enumerate(
                    zip(non_terminal_nodes, non_terminal_game_indices)
                ):
                    stats = adaptive_stats[int(gi)]
                    source = (
                        str(row_sources[row_idx]).strip().lower()
                        if row_idx < len(row_sources)
                        else 'unknown'
                    )
                    if source not in {'fresh_nn', 'cache_hit', 'deduplicated'}:
                        source = 'unknown'
                    if stats is not None:
                        if source == 'fresh_nn':
                            stats['nn_fresh_evaluations'] += 1
                        elif source == 'cache_hit':
                            stats['nn_cache_hits'] += 1
                        elif source == 'deduplicated':
                            stats['nn_deduplicated_evaluations'] += 1
                        else:
                            stats['nn_unknown_rows'] += 1
                            stats['nn_provenance_complete'] = False
                    if node_sources is not None:
                        node_sources[id(node)] = source
            server_batch_size = int(getattr(self.model, 'last_server_batch_size', 0) or 0)
            remote_request_seen = bool(
                server_batch_size > 0
                or int(getattr(self.model, 'last_cache_queries', 0) or 0) > 0
                or bool(getattr(self.model, 'last_transport_shared', False))
            )
            if remote_request_seen:
                self._profile_inc('central_inference_requests', 1)
                if compact_policy_response:
                    self._profile_inc('central_inference_compact_policy_requests', 1)
                    self._profile_inc(
                        'central_inference_compact_output_bytes_avoided',
                        max(0, batch_n * (ACTION_SIZE - max_legal_count) * 2),
                    )
                self._profile_inc('central_inference_server_batch_items', server_batch_size)
                self._profile_add(
                    'central_inference_remote_wait_time',
                    float(getattr(self.model, 'last_remote_wait_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_request_put_time',
                    float(getattr(self.model, 'last_request_put_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_queue_wait_time',
                    float(getattr(self.model, 'last_server_queue_wait_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_descriptor_queue_wait_time',
                    float(
                        getattr(
                            self.model,
                            'last_server_descriptor_queue_wait_s',
                            0.0,
                        ) or 0.0
                    ),
                )
                self._profile_add(
                    'central_inference_server_batch_coalesce_wait_time',
                    float(
                        getattr(
                            self.model,
                            'last_server_batch_coalesce_wait_s',
                            0.0,
                        ) or 0.0
                    ),
                )
                self._profile_add(
                    'central_inference_server_pipeline_wait_time',
                    float(
                        getattr(
                            self.model,
                            'last_server_pipeline_wait_s',
                            0.0,
                        ) or 0.0
                    ),
                )
                self._profile_inc(
                    'central_inference_server_batch_target_sum',
                    int(getattr(self.model, 'last_server_batch_target', 0) or 0),
                )
                self._profile_add(
                    'central_inference_server_output_finalize_wait_time',
                    float(
                        getattr(
                            self.model,
                            'last_server_output_finalize_wait_s',
                            0.0,
                        ) or 0.0
                    ),
                )
                self._profile_inc(
                    'central_inference_server_output_pipeline_requests',
                    int(bool(getattr(self.model, 'last_server_output_pipeline_used', False))),
                )
                self._profile_inc(
                    'central_inference_server_auto_calibrated_requests',
                    int(bool(getattr(self.model, 'last_server_batch_auto_calibrated', False))),
                )
                self._profile_add(
                    'central_inference_server_total_time',
                    float(getattr(self.model, 'last_server_total_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_concat_time',
                    float(getattr(self.model, 'last_server_concat_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_h2d_time',
                    float(getattr(self.model, 'last_server_h2d_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_forward_time',
                    float(getattr(self.model, 'last_server_forward_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_d2h_time',
                    float(getattr(self.model, 'last_server_d2h_s', 0.0) or 0.0),
                )
                self._profile_inc(
                    'central_inference_shared_requests',
                    int(bool(getattr(self.model, 'last_transport_shared', False))),
                )
                self._profile_inc(
                    'central_inference_shared_bytes_avoided',
                    int(getattr(self.model, 'last_shared_bytes_avoided', 0) or 0),
                )
                self._profile_add(
                    'central_inference_shared_slot_wait_time',
                    float(getattr(self.model, 'last_shared_slot_wait_s', 0.0) or 0.0),
                )
                self._profile_inc(
                    'central_inference_cache_queries',
                    int(getattr(self.model, 'last_cache_queries', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_cache_bypassed_positions',
                    int(getattr(self.model, 'last_cache_bypassed_positions', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_cache_hits',
                    int(getattr(self.model, 'last_cache_hits', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_dedup_hits',
                    int(getattr(self.model, 'last_dedup_hits', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_cache_suspensions',
                    int(getattr(self.model, 'last_cache_suspensions', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_cache_reactivations',
                    int(getattr(self.model, 'last_cache_reactivations', 0) or 0),
                )
                self._profile_inc(
                    'central_inference_nn_evaluated_positions',
                    int(getattr(self.model, 'last_nn_evaluated_positions', 0) or 0),
                )
                self._profile_add(
                    'central_inference_server_cache_lookup_time',
                    float(getattr(self.model, 'last_server_cache_lookup_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_staging_copy_time',
                    float(getattr(self.model, 'last_server_staging_copy_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_gpu_batch_fill_sum',
                    float(getattr(self.model, 'last_gpu_batch_fill', 0.0) or 0.0),
                )
            if profile_detail:
                if use_cuda_stage_timing:
                    torch.cuda.synchronize(self.device)

                    def _elapsed_s(start_event, end_event):
                        if start_event is None or end_event is None:
                            return 0.0
                        return max(0.0, float(start_event.elapsed_time(end_event)) / 1000.0)

                    h2d_time = _elapsed_s(h2d_start, h2d_end)
                    h2d_time += _elapsed_s(h2d_legal_start, h2d_legal_end)
                    gpu_postprocess_time = (
                        _elapsed_s(wdl_start, wdl_end)
                        + _elapsed_s(gather_start, gather_end)
                    )
                    d2h_time = _elapsed_s(d2h_legal_start, d2h_legal_end) + _elapsed_s(d2h_values_start, d2h_values_end)
                    self._profile_add('nn_h2d_time', h2d_time)
                    self._profile_add('nn_gpu_forward_time', _elapsed_s(forward_start, forward_end))
                    self._profile_add('nn_gpu_postprocess_time', gpu_postprocess_time)
                    self._profile_add('nn_d2h_time', d2h_time)

            cpu_policy_t0 = perf_counter() if profile_timing else None
            legal_probabilities_batch = _softmax_padded_legal_logits(
                legal_logits_batch,
                legal_counts,
            )
            for idx, node in enumerate(non_terminal_nodes):
                value = float(values_batch[idx])

                legal_moves = legal_moves_per_node[idx]
                legal_count = legal_counts[idx]

                if legal_moves:
                    legal_probs = legal_probabilities_batch[idx, :legal_count]
                    if self.tactical_priors_enabled:
                        tactical_multipliers = self._tactical_prior_multipliers(node.board, legal_moves)
                        if tactical_multipliers.size == legal_probs.size:
                            legal_probs = legal_probs * tactical_multipliers
                            legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                else:
                    legal_probs = np.array([])

                # Expand only if not already expanded (avoid overwriting priors in same batch).
                if not node.expanded:
                    node.raw_value = value
                    node.expand_children(legal_moves, legal_probs)

                values_by_node_id[id(node)] = value
            if profile_timing:
                self._profile_add('batch_expand_cpu_policy_time', perf_counter() - cpu_policy_t0)

        fanout_t0 = perf_counter() if profile_timing else None
        all_values = [0.0] * len(nodes)
        for node_id, indices in node_occurrences.items():
            value = values_by_node_id.get(node_id, terminal_values.get(node_id, 0.0))
            for idx in indices:
                all_values[idx] = value
        if profile_timing:
            self._profile_add('batch_expand_value_fanout_time', perf_counter() - fanout_t0)

        if profile_timing:
            self._profile_add('batch_expand_eval_time', perf_counter() - eval_t0)
            self._profile_inc('batch_expand_eval_calls', 1)
        if node_sources is not None:
            evaluation_trace['row_sources_by_node'] = node_sources
        return all_values
