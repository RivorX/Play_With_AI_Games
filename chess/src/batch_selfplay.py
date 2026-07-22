"""Batched Gumbel AlphaZero search for RL self-play.

Sequential Halving selects root actions and completed-Q policy improvement is
used below the root and for replay targets.
"""

import os
import hashlib
import tempfile
import torch
from src import chess_backend as chess
import numpy as np
import math
import pickle
import shutil
import time
import logging
import queue
import threading
import weakref
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from src.data import board_to_tensor, move_to_index
from src.utils.data_helpers import MAX_LEGAL_MOVES, _move_to_index_cached, board_to_tensor_pair
from src.utils.shared_inference import shared_inference_array
from src.utils.q_delta import (
    Q_DELTA_HIST_BINS as _Q_DELTA_HIST_BINS,
    USEFUL_SEARCH_Q_DELTA_MIN as _POLICY_CORRECTION_Q_DELTA_MIN,
    q_delta_histogram as _q_delta_histogram,
    q_delta_percentile_from_histogram as _q_delta_percentile_from_histogram,
)


_EMPTY_HISTORY_TENSOR = np.zeros((16, 8, 8), dtype=np.float32)
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
_DEFAULT_REPLAY_MAX_POLICY_TARGETS = 256
_REPLAY_SOURCE_UNKNOWN = 0
_REPLAY_SOURCE_LEARNER = 1
_REPLAY_SOURCE_FROZEN_BEST = 2

# Full Gumbel searches are the policy-improvement operator.  PCR remains the
# hard quality gate (fast searches never train policy).  Inside the complete
# targets, give a modest extra weight only when search changes the actor's top
# move and both visited moves show a positive Q delta.  This focuses limited
# policy capacity on actual improvements without discarding agreement rows or
# letting a noisy Q estimate dominate the objective.
_POLICY_UPTAKE_LOW_THRESHOLD = 0.75
_POLICY_CORRECTION_WEIGHT_MULTIPLIER = 1.35
_DYNAMIC_BUDGET_MARGINAL_PENALTY = 0.35
_DYNAMIC_BUDGET_TARGET_CHUNKS = 12
_TREE_REUSE_MIN_FRESH_SIMULATIONS = 64
_TREE_REUSE_CONSERVATIVE_FRESH_FRACTION = 0.50
_TREE_REUSE_HIGH_QUALITY_FRESH_FRACTION = 1.0 / 3.0
_TREE_REUSE_BASE_VISIT_CREDIT_DISCOUNT = 0.50
_TREE_REUSE_MAX_VISIT_CREDIT_DISCOUNT = 0.75
_TREE_REUSE_CANDIDATE_SUPPORT_VISITS = 2.0
_TREE_REUSE_SCOUT_SIMULATIONS = 32
_TREE_REUSE_SCOUT_MIN_TREE_QUALITY = 0.70
_TREE_REUSE_SCOUT_MIN_CANDIDATE_COVERAGE = 0.75
_TREE_REUSE_SCOUT_STABILITY_MIN = 0.70
_TREE_REUSE_SCOUT_POLICY_JS_SCALE = 0.05
_TREE_REUSE_SCOUT_MIN_FRESH_FRACTION = 1.0 / 6.0


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
    full_search,
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
    if not full_search or inherited_visits <= 0:
        return 0
    quality = max(0.0, min(1.0, float(quality)))
    fresh_floor = _resolve_tree_reuse_fresh_floor(budget, quality)
    visit_discount = (
        _TREE_REUSE_BASE_VISIT_CREDIT_DISCOUNT
        + quality * (
            _TREE_REUSE_MAX_VISIT_CREDIT_DISCOUNT
            - _TREE_REUSE_BASE_VISIT_CREDIT_DISCOUNT
        )
    )
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
    full_search,
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
    if not full_search or inherited_visits <= 0 or not winner_stable:
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


def _select_playout_cap_full_indices(
    indices,
    full_search_fraction,
    *,
    forced_full_indices=(),
    difficulty_scores=None,
    rng=None,
):
    """Select a stable full-search quota, preferring difficult positions."""
    indices = [int(idx) for idx in list(indices or [])]
    if not indices:
        return set()
    forced = {int(idx) for idx in forced_full_indices if int(idx) in indices}
    group_size = len(indices)
    if group_size < 4:
        desired_full = group_size
    else:
        fraction = max(0.0, min(1.0, float(full_search_fraction or 0.0)))
        desired_full = max(1, min(group_size, int(round(group_size * fraction))))
    desired_full = max(desired_full, len(forced))
    available = [idx for idx in indices if idx not in forced]
    extra_needed = max(0, desired_full - len(forced))
    selected = set(forced)
    if extra_needed > 0:
        if difficulty_scores is not None:
            score_by_index = {
                int(idx): float(difficulty_scores[pos])
                for pos, idx in enumerate(indices)
            }
            available.sort(key=lambda idx: (score_by_index.get(idx, 0.0), -idx), reverse=True)
            selected.update(available[:extra_needed])
        else:
            rng = np.random if rng is None else rng
            selected.update(int(idx) for idx in rng.permutation(available)[:extra_needed])
    return selected


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


def _replay_source_code(learner_turn, game_opponent_mcts, opponent_label):
    if learner_turn or game_opponent_mcts is None:
        return _REPLAY_SOURCE_LEARNER
    label = str(opponent_label or "")
    if label == "best":
        return _REPLAY_SOURCE_FROZEN_BEST
    return _REPLAY_SOURCE_UNKNOWN


def _policy_uptake_weight(
    search_metadata,
    *,
    good_target_min_top_visit_prob=0.55,
    good_target_min_visit_gap=0.12,
):
    """Return how much of the requested search budget produced this target.

    Every complete Gumbel target has equal policy weight.  PCR-fast positions
    are explicitly zeroed by the caller and still train final-outcome value.
    """
    metadata = search_metadata if isinstance(search_metadata, dict) else {}
    try:
        completeness = float(metadata.get('policy_weight', 1.0) or 0.0)
    except (TypeError, ValueError):
        completeness = 1.0
    completeness = max(0.0, min(1.0, completeness))

    return completeness


def _search_correction_metadata(search_metadata, *, full_search):
    """Return persisted correction flag, Q delta and bounded policy weight."""
    metadata = search_metadata if isinstance(search_metadata, dict) else {}
    try:
        changed_top = float(metadata.get('prior_mcts_agree', 1.0) or 0.0) < 0.5
    except (TypeError, ValueError):
        changed_top = False
    try:
        q_delta = float(metadata.get('mcts_q_delta', float('nan')))
    except (TypeError, ValueError):
        q_delta = float('nan')
    useful_correction = bool(
        full_search
        and changed_top
        and math.isfinite(q_delta)
        and q_delta > _POLICY_CORRECTION_Q_DELTA_MIN
    )
    multiplier = _POLICY_CORRECTION_WEIGHT_MULTIPLIER if useful_correction else 1.0
    return changed_top, q_delta, multiplier


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
        self.max_pieces = max(2, int(max_pieces))
        self.paths = tuple(str(p) for p in paths)
        self._tb = None
        self._python_chess = None

        if not self.paths:
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
        return self.enabled and chess.piece_count(board) <= self.max_pieces

    def probe_wdl(self, board):
        if not self.can_probe(board):
            return None
        try:
            # Explicit cold-path boundary: this runs only on real game roots
            # with <= max_pieces, never inside an MCTS simulation.
            probe_board = self._python_chess.Board(chess.board_fen(board))
            return int(self._tb.probe_wdl(probe_board))
        except Exception:
            return None

    def result_for_board(self, board):
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        value = _syzygy_wdl_to_value(wdl)
        if value > 0:
            return '1-0' if board.turn == chess.WHITE else '0-1'
        if value < 0:
            return '0-1' if board.turn == chess.WHITE else '1-0'
        return '1/2-1/2'


def _resolve_syzygy_paths(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_paths = rl_cfg.get('syzygy_paths', []) or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]

    chess_root = Path(__file__).resolve().parents[1]
    resolved = []
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        path_str = str(raw_path).strip()
        if not path_str:
            continue
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = chess_root / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            resolved.append(str(candidate))
    return tuple(resolved)


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


class _SelfPlayCompileLock:
    def __init__(self, lock_path, timeout_s=900.0, poll_s=0.2, stale_after_s=1800.0):
        self.lock_path = Path(lock_path)
        self.timeout_s = max(1.0, float(timeout_s))
        self.poll_s = max(0.05, float(poll_s))
        self.stale_after_s = max(self.timeout_s, float(stale_after_s))
        self._fd = None

    def __enter__(self):
        start_t = time.perf_counter()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = f"{os.getpid()} {time.time():.6f}\n".encode("ascii", errors="ignore")
                os.write(self._fd, payload)
                return self
            except FileExistsError:
                try:
                    stat = self.lock_path.stat()
                    age_s = max(0.0, time.time() - float(stat.st_mtime))
                    if age_s > self.stale_after_s:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
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
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass


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
    # Default to parallel compile across workers.
    return False


def _configure_selfplay_compile_cache(rank, device):
    if device.type != 'cuda':
        return None
    root = Path(tempfile.gettempdir()) / "play_with_ai_games" / "torch_compile_cache"
    worker_dir = root / f"cuda_worker_{int(rank)}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(worker_dir / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(worker_dir / "triton")
    return root


def _reset_selfplay_compile_cache(rank, device):
    """Discard only this worker's temporary Inductor/Triton cache.

    A rank gets its own directory, so a failed compiled graph can be rebuilt
    without touching project files or cache entries owned by another worker.
    """
    if device.type != 'cuda':
        return False
    root = Path(tempfile.gettempdir()) / "play_with_ai_games" / "torch_compile_cache"
    worker_dir = root / f"cuda_worker_{int(rank)}"
    try:
        # Keep the destructive operation constrained to the known temporary
        # cache layout even if this helper is called with an unexpected rank.
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


def _maybe_compile_selfplay_model(model, config, device, rank, model_label="learner"):
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

    history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
    expected_input_planes = 16 * (1 + history_positions)
    use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
    use_bfloat16 = bool(config.get('hardware', {}).get('use_bfloat16', False))
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    if bool(rl_cfg.get('self_play_compile_reset_dynamo', False)):
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    def _compile_and_warmup(target_model):
        compiled_model = torch.compile(target_model, mode='default', dynamic=True)
        dummy_input = torch.zeros(
            1,
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
    Fast board identity for tree reuse.
    """
    return chess.position_key(board)


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
        self.log_base_priors = np.log(
            np.maximum(self.base_priors.astype(np.float64), np.finfo(np.float64).tiny)
        )
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
        self.selection_score_scratch = np.empty(child_count, dtype=np.float64)
        self.q_score_scratch = np.empty(child_count, dtype=np.float32)
        self.fresh_count_scratch = np.empty(child_count, dtype=np.float64)
        self.eligibility_scratch = np.empty(child_count, dtype=np.bool_)
        self._move_to_child = None

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
        "_board_tensor",
        "_history_tensor_pair",
        "_legal_moves",
        "_legal_indices",
        "raw_value",
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
        self._board_tensor = None
        self._history_tensor_pair = None
        self._legal_moves = None
        self._legal_indices = None
        self.raw_value = None

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
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
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

    def get_legal_moves(self):
        if self._legal_moves is None:
            self._legal_moves = tuple(chess.legal_moves(self.board))
        return self._legal_moves

    def get_legal_indices(self):
        if self._legal_indices is None:
            legal_moves = self.get_legal_moves()
            is_black_turn = self.board.turn == chess.BLACK
            legal_indices = np.empty(len(legal_moves), dtype=np.int32)
            move_to_idx = _move_to_index_cached
            for idx, move in enumerate(legal_moves):
                legal_indices[idx] = move_to_idx(
                    chess.move_origin_index(move),
                    chess.move_destination_index(move),
                    move.promotion or 0,
                    is_black_turn,
                )
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
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        return moves[int(np.argmax(visits))]

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

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        rl_cfg = config['reinforcement_learning']

        self.gumbel_max_considered_actions = max(
            1,
            int(rl_cfg.get('mcts_gumbel_max_considered_actions', 16)),
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
            1.0,
            float(rl_cfg.get('mcts_gumbel_target_temperature', 1.20)),
        )
        self.gumbel_use_mixed_value = bool(
            rl_cfg.get('mcts_gumbel_use_mixed_value', True)
        )

        self.dynamic_budget_enabled = bool(
            config['reinforcement_learning'].get('mcts_dynamic_budget_enabled', False)
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
        self.playout_cap_randomization_enabled = bool(
            rl_cfg.get('mcts_playout_cap_randomization_enabled', False)
        )
        self.playout_cap_full_search_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('mcts_playout_cap_full_search_fraction', 0.70))),
        )
        self.playout_cap_fast_simulations = max(
            1,
            int(rl_cfg.get('mcts_playout_cap_fast_simulations', 16)),
        )
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
        # History configuration (POV)
        self.history_positions = config['model'].get('history_positions', 0)
        self.history_storage_dtype = (
            np.float16
            if bool(config['reinforcement_learning'].get('self_play_history_fp16', True))
            else np.float32
        )

        # Tree reuse
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)
        self.syzygy = _get_syzygy_oracle(config)
        self.cache_node_tensors = False
        self.cache_history_tensors = bool(
            config['reinforcement_learning'].get('mcts_cache_history_tensors', True)
        )
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
        self._empty_history_tensor = _EMPTY_HISTORY_TENSOR.astype(self.history_storage_dtype, copy=False)
        self._board_planes = int(self._empty_history_tensor.shape[0])
        self._history_planes = int(self.history_positions) * self._board_planes
        self._input_planes = self._board_planes + self._history_planes
        self._legal_index_scratch = {}
        self._legal_index_tensor_scratch = {}
        self._board_input_scratch = {}
        self._board_input_tensor_scratch = {}
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
        self._profile_stats = {
            'search_many_time': 0.0,
            'search_many_calls': 0,
            # Sum of root->leaf path lengths. This is a selection traversal
            # counter, not the number of completed MCTS simulations/visits.
            'selection_node_traversals': 0,
            'search_root_setup_time': 0.0,
            'search_selection_time': 0.0,
            'search_backprop_time': 0.0,
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
        }

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
        return dict(self._profile_stats)

    def _board_to_tensor_profiled(self, board, flip_perspective=None, storage_dtype=None):
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor')
        tensor = board_to_tensor(
            board,
            flip_perspective=flip_perspective,
            dtype=np.float32 if storage_dtype is None else storage_dtype,
        )
        self._profile_sample_finish('board_to_tensor', sample_t0, sample_scale)
        return tensor

    def _history_tensor_pair_for_board(self, board):
        storage_dtype = self.history_storage_dtype
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor_pair')
        white_tensor, black_tensor = board_to_tensor_pair(board, dtype=storage_dtype)
        if sample_t0 is not None and sample_scale > 0:
            self._profile_add('board_to_tensor_time', (time.perf_counter() - sample_t0) * float(sample_scale))
            self._profile_inc('board_to_tensor_calls', int(sample_scale) * 2)
        return (white_tensor, black_tensor)

    def _history_tensor_pair_for_node(self, node):
        cached = getattr(node, "_history_tensor_pair", None)
        if cached is None or not self.cache_history_tensors:
            cached = self._history_tensor_pair_for_board(node.board)
            if self.cache_history_tensors:
                node._history_tensor_pair = cached
        return cached

    @staticmethod
    def _is_encoded_history_entry(entry):
        return (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], np.ndarray)
            and isinstance(entry[1], np.ndarray)
        )

    def _encode_history_entry(self, board_or_fen):
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

        return self._history_tensor_pair_for_board(board_obj)

    def _build_history_tensor(self, current_board, board_history, current_tensor=None):
        """
        Build tensor with history using POV-aware board_to_tensor.

        Args:
            current_board: chess.Board for current position
            board_history: list of previous boards (from real game history)
        """
        if self.history_positions == 0:
            if current_tensor is not None:
                return current_tensor
            return self._board_to_tensor_profiled(current_board)

        history_tensors = []
        use_black_pov = current_board.turn == chess.BLACK
        if board_history:
            history_boards = board_history[-self.history_positions:]
            for hist_entry in history_boards:
                encoded = self._encode_history_entry(hist_entry)
                hist_tensor = encoded[1] if use_black_pov else encoded[0]
                history_tensors.append(hist_tensor)

        pad_count = max(0, self.history_positions - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        if current_tensor is None:
            current_tensor = self._board_to_tensor_profiled(current_board)
        history_tensors.append(current_tensor)

        return np.concatenate(history_tensors, axis=0)

    def _history_tensor_for_entry(self, entry, use_black_pov):
        if self._is_encoded_history_entry(entry):
            return entry[1] if use_black_pov else entry[0]

        if isinstance(entry, MCTSNode):
            encoded = self._history_tensor_pair_for_node(entry)
            return encoded[1] if use_black_pov else encoded[0]

        if isinstance(entry, chess.Board):
            return self._board_to_tensor_profiled(
                entry,
                flip_perspective=use_black_pov,
                storage_dtype=self.history_storage_dtype,
            )

        encoded = self._encode_history_entry(entry)
        return encoded[1] if use_black_pov else encoded[0]

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

    def _build_history_prefix_for_node(self, node, board_history):
        if self.history_positions <= 0:
            return None

        use_black_pov = (node.board.turn == chess.BLACK)
        history_entries = self._collect_node_history_entries(node, board_history)
        history_tensors = [
            self._history_tensor_for_entry(entry, use_black_pov)
            for entry in history_entries
        ]

        pad_count = max(0, int(self.history_positions) - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        return np.concatenate(history_tensors, axis=0) if history_tensors else None

    def _current_tensor_for_node(self, node):
        if not self.cache_node_tensors:
            return self._board_to_tensor_profiled(node.board)

        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = self._board_to_tensor_profiled(node.board)
            node._board_tensor = cached
        return cached

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
                tensor_scratch = torch.empty(
                    (batch_size, self._input_planes, 8, 8),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self._board_input_tensor_scratch[key] = tensor_scratch
                scratch = tensor_scratch.numpy()
            else:
                scratch = np.empty((batch_size, self._input_planes, 8, 8), dtype=np.float32)
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

        visits = edges.visit_counts.astype(np.float64, copy=False)
        priors = edges.base_priors.astype(np.float64, copy=False)
        child_count = int(visits.size)
        if child_count <= 0:
            return np.empty(0, dtype=np.float32)

        qvalues = np.zeros(child_count, dtype=np.float64)
        visited = visits > 0.0
        if visited.any():
            # Child values use the child side-to-move perspective.
            qvalues[visited] = -edges.value_sums[visited].astype(np.float64) / visits[visited]

        raw_value = node.raw_value
        if raw_value is None:
            node_visits = int(getattr(node, 'visit_count', 0) or 0)
            raw_value = float(node.value_sum / node_visits) if node_visits > 0 else 0.0
        raw_value = float(max(-1.0, min(1.0, raw_value)))

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

        completed = np.where(visited, qvalues, completion_value)
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
        use_cache = scale_visit_counts is None
        if (
            use_cache
            and edges.improved_policy_cache_version == edges.stats_version
            and edges.improved_policy_cache.size == edges.base_priors.size
        ):
            return edges.improved_policy_cache

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

    def _gumbel_policy_target(self, improved_policy):
        """Soften only the stored training target, not search action selection."""
        probs = np.asarray(improved_policy, dtype=np.float64)
        if probs.size <= 1 or self.gumbel_target_temperature <= 1.0 + 1e-8:
            return probs.astype(np.float32, copy=False)
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

        initial = state.get('_initial_visits_f64')
        if initial is None:
            initial = np.asarray(state['initial_visits'], dtype=np.float64)
            state['_initial_visits_f64'] = initial
            state['_initial_total'] = float(initial.sum())
        fresh_counts = edges.fresh_count_scratch
        np.subtract(edges.total_counts, initial, out=fresh_counts)
        np.maximum(fresh_counts, 0.0, out=fresh_counts)
        sequence = state['sequence']
        simulation_index = int(round(edges.total_count_sum - state['_initial_total']))
        considered_visit = sequence[min(simulation_index, len(sequence) - 1)] if sequence else 0

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
        max_fresh = float(np.max(fresh_counts)) if fresh_counts.size else 0.0
        if (
            state.get('_q_score_version') != edges.stats_version
            or state.get('_q_score_max_visit') != max_fresh
        ):
            scale = (self.gumbel_c_visit + max_fresh) * self.gumbel_c_scale
            np.multiply(
                self._gumbel_normalized_completed_qvalues(node),
                scale,
                out=edges.q_score_scratch,
            )
            state['_q_score_version'] = edges.stats_version
            state['_q_score_max_visit'] = max_fresh
        np.add(scores, edges.q_score_scratch, out=scores)
        eligible = edges.eligibility_scratch
        np.equal(fresh_counts, float(considered_visit), out=eligible)
        if not eligible.any():
            # Defensive fallback for externally supplied/reused roots with
            # inconsistent counters. Prefer the least freshly visited actions.
            np.equal(fresh_counts, float(np.min(fresh_counts)), out=eligible)
        scores[~eligible] = -np.inf
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

    def _gumbel_final_action_index(self, node, state):
        edges = node.edges
        if not edges.moves:
            return None
        initial = np.asarray(state['initial_visits'], dtype=np.float64)
        fresh = edges.fresh_count_scratch
        np.subtract(edges.visit_counts, initial, out=fresh)
        np.maximum(fresh, 0.0, out=fresh)
        most_visited = float(np.max(fresh)) if fresh.size else 0.0
        scores = edges.selection_score_scratch
        np.copyto(scores, edges.log_base_priors)
        scores -= float(np.max(scores))
        np.add(scores, np.asarray(state['gumbel'], dtype=np.float64), out=scores)
        np.add(
            scores,
            self._gumbel_completed_qvalues(node, scale_visit_counts=fresh),
            out=scores,
        )
        eligible = edges.eligibility_scratch
        np.equal(fresh, most_visited, out=eligible)
        scores[~eligible] = -np.inf
        return int(np.argmax(scores))

    def _select_child(self, node):
        """Select an interior child using completed-Q policy improvement."""
        return self._select_child_gumbel(node)

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

    def search_many(self, game_states, num_simulations, add_root_noise=False, return_search_metadata=False):
        """
        Run MCTS for multiple games and batch leaf evaluations across games.

        Args:
            game_states: list of dicts with keys: board, root, board_history
            num_simulations: simulations per game
        Returns:
            List[Dict[chess.Move, float]] search weights for each game in order
        """
        if not game_states:
            if return_search_metadata:
                return [], []
            return []
        perf_counter = time.perf_counter
        profile_detail = self.profile_detail_enabled
        profile_timing = self.profile_enabled
        search_t0 = perf_counter()

        game_count = len(game_states)
        boards = [None] * game_count
        roots = [None] * game_count
        initial_root_visits = [0] * game_count
        initial_child_visits = [None] * game_count
        incoming_root_present = [False] * game_count
        root_reused_flags = [False] * game_count
        root_synced_flags = [False] * game_count
        board_histories = [None] * game_count
        root_position_counts = [None] * game_count
        budget_overrides = [None] * game_count
        playout_cap_eligible = [False] * game_count
        playout_cap_forced_full = [False] * game_count
        is_mapping_state = [False] * game_count

        # Initialize / reuse roots per game.
        # Supports both dict states and packed list states:
        # [board, root, root_synced, board_history, move_count, position_counts].
        root_setup_t0 = perf_counter() if profile_timing else None
        for idx, gs in enumerate(game_states):
            if isinstance(gs, dict):
                is_mapping_state[idx] = True
                board = gs['board']
                root = gs.get('root', None)
                root_synced = bool(gs.get('_root_synced', False))
                board_history = gs.get('board_history', [])
                position_counts = gs.get('position_counts')
                budget_override = gs.get('simulation_budget_override')
                playout_cap_eligible[idx] = bool(gs.get('playout_cap_eligible', False))
                playout_cap_forced_full[idx] = bool(gs.get('playout_cap_forced_full', False))
            else:
                board = gs[0]
                root = gs[1] if len(gs) > 1 else None
                root_synced = bool(gs[2]) if len(gs) > 2 else False
                board_history = gs[3] if len(gs) > 3 else []
                position_counts = gs[5] if len(gs) > 5 else None
                budget_override = gs[6] if len(gs) > 6 else None
                playout_cap_eligible[idx] = bool(gs[7]) if len(gs) > 7 else False
                playout_cap_forced_full[idx] = bool(gs[8]) if len(gs) > 8 else False

            boards[idx] = board
            incoming_root_present[idx] = root is not None
            board_histories[idx] = board_history
            root_position_counts[idx] = (
                position_counts
                if isinstance(position_counts, dict)
                else _position_counts_from_board(board)
            )
            budget_overrides[idx] = (
                max(1, int(budget_override)) if budget_override is not None else None
            )

            if self.reuse_tree and root is not None:
                if root_synced:
                    root_synced = False
                    root_reused_flags[idx] = True
                else:
                    target_key = _board_position_key(board)
                    if root.get_position_key() == target_key:
                        root_reused_flags[idx] = True
                    else:
                        for move in root.edges.moves:
                            child = root.get_child_for_move(move)
                            if child is not None and child.get_position_key() == target_key:
                                _ = child.board  # Ensure board is instantiated
                                root = child.detach_as_root()
                                root_reused_flags[idx] = True
                                break
                        else:
                            root = MCTSNode(board)
            else:
                root = MCTSNode(board)
                root_synced = False

            roots[idx] = root
            initial_root_visits[idx] = 0 if root is None else int(getattr(root, 'visit_count', 0) or 0)
            if root is not None and root.expanded and root.edges is not None:
                initial_child_visits[idx] = root.edges.visit_counts.copy()
            root_synced_flags[idx] = bool(root_synced)
        if profile_timing:
            self._profile_add('search_root_setup_time', perf_counter() - root_setup_t0)

        fixed_simulation_budget = max(1, int(num_simulations))
        dynamic_budget_active = bool(
            self.dynamic_budget_enabled
            and game_count > 1
            and any(value is None for value in budget_overrides)
        )
        eval_easy_cut_active = bool(
            self.eval_easy_cut_enabled
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
        ):
            nonlocal game_ptr
            total_remaining = int(sum(remaining))
            while total_remaining > 0:
                # Keep decisions in coarse batches. This preserves central GPU
                # batching even though roots receive different total budgets.
                active_count = sum(1 for value in remaining if value > 0)
                max_remaining_any_game = max(remaining) if remaining else 1
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
                leaf_game_indices = []
                selected_this_batch = [0] * game_count
                selection_node_traversals_this_batch = 0

                selection_t0 = perf_counter() if profile_timing else None
                for _ in range(batch_size):
                    found = False
                    for _ in range(game_count):
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
                    selected_this_batch[gs_idx] += 1

                    remaining[gs_idx] -= 1
                    total_remaining -= 1
                    game_ptr = (game_ptr + 1) % game_count
                if profile_timing:
                    self._profile_add('search_selection_time', perf_counter() - selection_t0)
                self._profile_inc(
                    'selection_node_traversals',
                    selection_node_traversals_this_batch,
                )

                if not leaf_nodes:
                    break

                values = self._batch_expand_and_evaluate(
                    leaf_nodes,
                    leaf_game_indices,
                    board_histories,
                    root_position_counts,
                )

                backprop_t0 = perf_counter() if profile_timing else None
                for search_path, value in zip(search_paths, values):
                    self._backpropagate_and_remove_virtual_loss(search_path, value)
                if profile_timing:
                    self._profile_add('search_backprop_time', perf_counter() - backprop_t0)
                if post_batch_callback is not None:
                    removed = max(
                        0,
                        int(post_batch_callback(remaining, selected_this_batch) or 0),
                    )
                    total_remaining = max(0, total_remaining - removed)

        scout_difficulties = [0.0] * game_count
        # Evaluate every unexpanded root exactly once so priors/raw values exist
        # before assigning the complete Sequential-Halving budget.
        root_setup = [
            1 if root is not None and not root.expanded else 0
            for root in roots
        ]
        if any(root_setup):
            _run_simulation_phase(root_setup)

        if dynamic_budget_active or eval_easy_cut_active or any(playout_cap_eligible):
            scout_difficulties = [_gumbel_prior_difficulty(root) for root in roots]
        playout_cap_full_indices = {
            idx for idx in range(game_count) if not playout_cap_eligible[idx]
        }
        eligible_playout_cap_indices = [
            idx for idx, eligible in enumerate(playout_cap_eligible) if eligible
        ]
        if eligible_playout_cap_indices:
            forced_full = [
                idx for idx in eligible_playout_cap_indices if playout_cap_forced_full[idx]
            ]
            selected_full = _select_playout_cap_full_indices(
                eligible_playout_cap_indices,
                self.playout_cap_full_search_fraction,
                forced_full_indices=forced_full,
                difficulty_scores=[scout_difficulties[idx] for idx in eligible_playout_cap_indices],
            )
            playout_cap_full_indices.update(selected_full)
            for idx in eligible_playout_cap_indices:
                if idx not in selected_full:
                    budget_overrides[idx] = int(self.playout_cap_fast_simulations)
        overridden_total = sum(
            int(value) for value in budget_overrides if value is not None
        )
        adaptive_indices = [
            idx for idx, value in enumerate(budget_overrides) if value is None
        ]
        if adaptive_indices:
            requested_total = fixed_simulation_budget * game_count
            adaptive_total = max(
                len(adaptive_indices),
                requested_total - overridden_total,
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
                tree_reuse_quality[idx] = _summarize_tree_reuse_quality(
                    root,
                    noise,
                    min(
                        self.gumbel_max_considered_actions,
                        max(1, int(simulation_budgets[idx])),
                    ),
                )
        inherited_visit_credits = [0] * game_count
        if add_root_noise and self.reuse_tree and self.tree_reuse_visit_credit_enabled:
            for idx, budget in enumerate(simulation_budgets):
                if not root_reused_flags[idx]:
                    continue
                inherited_visit_credits[idx] = _resolve_tree_reuse_visit_credit(
                    budget,
                    initial_root_visits[idx],
                    full_search=idx in playout_cap_full_indices,
                    quality=tree_reuse_quality[idx]['quality'],
                )
        action_budgets = [
            max(
                0,
                int(simulation_budgets[idx])
                - int(setup_visits[idx])
                - int(inherited_visit_credits[idx]),
            )
            for idx in range(game_count)
        ]
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
                    if root_reused_flags[idx] and idx in playout_cap_full_indices
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
            considered = min(
                len(root.edges.moves),
                self.gumbel_max_considered_actions,
                max(1, action_budgets[idx]),
            )
            gumbel_root_states[idx] = {
                'initial_visits': np.asarray(initial, dtype=np.int32),
                'gumbel': root_gumbel_noises[idx],
                'considered_actions': int(considered),
                'sequence': _gumbel_considered_visit_sequence(
                    considered,
                    action_budgets[idx],
                ),
            }
            if (
                add_root_noise
                and root_reused_flags[idx]
                and idx in playout_cap_full_indices
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
                    scale_visit_counts=np.zeros(len(root.edges.moves), dtype=np.float64),
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
                    scale_visit_counts=fresh_counts,
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
                    full_search=idx in playout_cap_full_indices,
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

        _run_simulation_phase(
            action_budgets,
            gumbel_root_states,
            post_batch_callback=_apply_ready_tree_reuse_scouts,
        )

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

        result = []
        search_metadata = []
        metadata_t0 = perf_counter() if profile_timing else None
        for idx, root in enumerate(roots):
            metadata = self._summarize_root_search(
                root,
                simulation_budgets[idx],
                initial_root_visits=initial_root_visits[idx],
                initial_child_visits=initial_child_visits[idx],
                inherited_visit_credit=inherited_visit_credits[idx],
            )
            metadata['dynamic_budget_enabled'] = bool(dynamic_budget_active)
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
            metadata['playout_cap_eligible'] = bool(playout_cap_eligible[idx])
            metadata['playout_cap_full_search'] = bool(idx in playout_cap_full_indices)
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
                    scale_visit_counts=fresh_count_vector,
                )
                policy_target_probs = self._gumbel_policy_target(improved)
                selected_idx = self._gumbel_final_action_index(root, gumbel_state)
                selected_move = root.edges.moves[selected_idx] if selected_idx is not None else None
                policy_target = {
                    move: float(probability)
                    for move, probability in zip(root.edges.moves, policy_target_probs)
                    if float(probability) > 0.0
                }
                if policy_target:
                    metadata['policy_target_probs_override'] = policy_target
                metadata['gumbel_considered_actions'] = int(
                    gumbel_state.get('considered_actions', 0) or 0
                )
                metadata['gumbel_scale'] = float(
                    self.gumbel_scale if add_root_noise else self.gumbel_eval_scale
                )
                metadata['gumbel_q_range_floor'] = float(self.gumbel_q_range_floor)
                metadata['gumbel_target_temperature'] = float(self.gumbel_target_temperature)
                if selected_move is not None:
                    metadata['selected_move_override'] = selected_move

                # Full Gumbel trains against the improved completed-Q policy,
                # so quality telemetry must compare the prior to that policy
                # rather than to the incidental Sequential-Halving visit shape.
                priors = root.edges.base_priors.astype(np.float64, copy=False)
                prior_total = float(priors.sum())
                prior_probs = priors / prior_total if prior_total > 0.0 else np.full_like(priors, 1.0 / len(priors))
                prior_top_idx = int(np.argmax(prior_probs))
                improved_top_idx = int(np.argmax(improved))
                selected_prior_agree = (
                    None if selected_idx is None else (1.0 if prior_top_idx == selected_idx else 0.0)
                )
                metadata['selected_prior_agree'] = selected_prior_agree
                metadata['selected_q_delta'] = None
                metadata['prior_mcts_agree'] = 1.0 if prior_top_idx == improved_top_idx else 0.0
                metadata['prior_top_visit_prob'] = float(improved[prior_top_idx])
                metadata['mcts_top_prior_prob'] = float(prior_probs[improved_top_idx])
                metadata['mcts_policy_kl'] = float(np.sum(
                    improved.astype(np.float64) * (
                        np.log(np.clip(improved.astype(np.float64), 1e-12, 1.0))
                        - np.log(np.clip(prior_probs, 1e-12, 1.0))
                    )
                ))
                metadata['policy_kld'] = metadata['mcts_policy_kl']
                ordered = np.sort(improved.astype(np.float64))
                top = float(ordered[-1])
                second = float(ordered[-2]) if ordered.size > 1 else 0.0
                metadata['top_visit_prob'] = top
                metadata['visit_gap'] = max(0.0, top - second)
                target_entropy = float(-np.sum(
                    improved.astype(np.float64) * np.log(np.clip(improved.astype(np.float64), 1e-12, 1.0))
                ))
                metadata['visit_entropy'] = (
                    target_entropy / max(1e-12, math.log(len(improved)))
                    if len(improved) > 1 else 0.0
                )

                cumulative = root.edges.visit_counts.astype(np.float64, copy=False)
                if cumulative[improved_top_idx] > 0.0:
                    metadata['best_q'] = float(max(
                        -1.0,
                        min(
                            1.0,
                            -float(root.edges.value_sums[improved_top_idx])
                            / float(cumulative[improved_top_idx]),
                        ),
                    ))
                if cumulative[prior_top_idx] > 0.0 and cumulative[improved_top_idx] > 0.0:
                    prior_q = -float(root.edges.value_sums[prior_top_idx]) / cumulative[prior_top_idx]
                    improved_q = -float(root.edges.value_sums[improved_top_idx]) / cumulative[improved_top_idx]
                    q_delta = float(max(-2.0, min(2.0, improved_q - prior_q)))
                    metadata['mcts_q_delta'] = q_delta
                    metadata['mcts_changed_to_lower_q'] = (
                        1.0 if prior_top_idx != improved_top_idx and q_delta < -0.02 else 0.0
                    )
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

                root_result = {
                    move: float(count)
                    for move, count in fresh_visits.items()
                    if float(count) > 0.0
                }
                # Consumers without metadata (eval/play compatibility paths)
                # still choose the exact Sequential-Halving winner on ties.
                if selected_move is not None and selected_move in root_result:
                    root_result[selected_move] = float(root_result[selected_move]) + 0.25
            elif fresh_visits:
                metadata['policy_visit_counts_override'] = fresh_visits
            result.append(root_result)
            search_metadata.append(metadata)
        if profile_timing:
            self._profile_add('search_metadata_time', perf_counter() - metadata_t0)
        self._profile_add('search_many_time', perf_counter() - search_t0)
        self._profile_inc('search_many_calls', 1)
        if return_search_metadata:
            return result, search_metadata
        return result

    def _batch_expand_and_evaluate(
        self,
        nodes,
        game_indices,
        board_histories,
        root_position_counts,
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
            index_t0, index_scale = self._profile_sample_begin('batch_expand_move_index')
            legal_indices = node.get_legal_indices()
            self._profile_sample_finish('batch_expand_move_index', index_t0, index_scale)
            legal_count = len(legal_indices)
            if legal_count <= 0:
                terminal_values[id(node)] = -1.0 if is_check else 0.0
                continue
            if is_path_draw:
                terminal_values[id(node)] = 0.0
                continue

            # Probe simulated leaves only. A terminal root would have no child
            # visit distribution for UI/eval callers to play from; the
            # self-play game loop handles real-root tablebase results.
            syzygy_wdl = (
                self.syzygy.probe_wdl(board)
                if node.parent is not None and self.syzygy is not None
                else None
            )
            if syzygy_wdl is not None:
                terminal_values[id(node)] = _syzygy_wdl_to_value(syzygy_wdl)
                continue
            non_terminal_nodes.append(node)
            non_terminal_game_indices.append(gi)
            legal_moves_per_node.append(legal_moves)
            legal_indices_per_node.append(legal_indices)
            legal_counts.append(legal_count)
            if legal_count > max_legal_count:
                max_legal_count = legal_count
        values_by_node_id = {}

        if non_terminal_nodes:
            history_prefix_cache = {}

            def _get_history_prefix_for_node(game_idx, node):
                if self.history_positions <= 0:
                    return None

                cache_key = (int(game_idx), id(node))
                if cache_key in history_prefix_cache:
                    return history_prefix_cache[cache_key]

                cached = self._build_history_prefix_for_node(node, board_histories[game_idx])
                history_prefix_cache[cache_key] = cached
                return cached

            batch_n = len(non_terminal_nodes)
            boards_np = self._get_board_input_scratch(batch_n)
            tensor_pack_t0 = perf_counter() if profile_timing else None
            for row_idx, (node, gi) in enumerate(zip(non_terminal_nodes, non_terminal_game_indices)):
                current_tensor = self._current_tensor_for_node(node)
                history_t0 = perf_counter() if profile_detail else None
                history_prefix = _get_history_prefix_for_node(gi, node)
                if profile_detail:
                    self._profile_add('batch_expand_history_time', perf_counter() - history_t0)
                input_pack_t0 = perf_counter() if profile_detail else None
                if history_prefix is None:
                    boards_np[row_idx, :, :, :] = current_tensor
                else:
                    boards_np[row_idx, :self._history_planes, :, :] = history_prefix
                    boards_np[row_idx, self._history_planes:, :, :] = current_tensor
                if profile_detail:
                    self._profile_add('batch_expand_input_pack_time', perf_counter() - input_pack_t0)
            if profile_timing:
                self._profile_add('batch_expand_tensor_pack_time', perf_counter() - tensor_pack_t0)

            use_cuda_stage_timing = (
                self.cuda_stage_profile_enabled
                and self.device.type == 'cuda'
                and torch.cuda.is_available()
            )

            def _cuda_event():
                if not use_cuda_stage_timing:
                    return None
                return torch.cuda.Event(enable_timing=True)

            remote_legal_gather = bool(getattr(self.model, 'supports_remote_legal_gather', False))
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
            if (
                legal_index_matrix is not None
                and remote_legal_gather
            ):
                model_kwargs['legal_index_matrix'] = legal_index_matrix
                model_kwargs['legal_counts'] = np.asarray(legal_counts, dtype=np.int16)
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        policy_logits_batch, values_batch = self.model(
                            board_tensors,
                            **model_kwargs,
                        )
                else:
                    policy_logits_batch, values_batch = self.model(
                        board_tensors,
                        **model_kwargs,
                    )
                if forward_end is not None:
                    forward_end.record()
                compact_policy_response = bool(getattr(self.model, 'last_response_compact_policy', False))
                
                if values_batch.dim() == 2 and values_batch.shape[1] == 3:
                    wdl_start = _cuda_event()
                    wdl_end = _cuda_event()
                    if wdl_start is not None:
                        wdl_start.record()
                    wdl_probs = torch.softmax(values_batch, dim=1)
                    values_batch = (wdl_probs[:, 0] - wdl_probs[:, 2])
                    if wdl_end is not None:
                        wdl_end.record()

                # Gather only legal move logits on GPU before transferring to CPU.
                if max_legal_count > 0:
                    if compact_policy_response:
                        legal_logits_batch = policy_logits_batch.to(dtype=torch.float16).cpu().numpy()
                        d2h_legal_start = None
                        d2h_legal_end = None
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
                        d2h_legal_start = _cuda_event()
                        d2h_legal_end = _cuda_event()
                        if d2h_legal_start is not None:
                            d2h_legal_start.record()
                        legal_logits_batch = legal_logits_batch.to(dtype=torch.float16).cpu().numpy()
                        if d2h_legal_end is not None:
                            d2h_legal_end.record()
                else:
                    legal_logits_batch = None
                    d2h_legal_start = None
                    d2h_legal_end = None
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
            server_batch_size = int(getattr(self.model, 'last_server_batch_size', 0) or 0)
            remote_request_seen = bool(
                server_batch_size > 0
                or int(getattr(self.model, 'last_cache_queries', 0) or 0) > 0
                or bool(getattr(self.model, 'last_transport_shared', False))
            )
            if remote_request_seen:
                self._profile_inc('central_inference_requests', 1)
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
        return all_values


class BatchSelfPlayMCTSBatch:
    """
    True batch self-play: multiple games in parallel, shared GPU eval batches.
    """

    def __init__(
        self,
        model,
        config,
        device,
        max_batch_games_per_worker,
        opponent_model=None,
        opponent_source_label="current",
        opponent_models_by_label=None,
        opponent_plan_labels=None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.opponent_model = opponent_model
        self.opponent_source_label = str(opponent_source_label or "current")
        rl_cfg = config.get('reinforcement_learning', {})
        self.mcts_phase_opening_max_fullmove = max(
            1,
            int(rl_cfg.get('value_phase_opening_max_fullmove', 12)),
        )
        self.mcts_phase_endgame_min_fullmove = max(
            self.mcts_phase_opening_max_fullmove + 1,
            int(rl_cfg.get('value_phase_endgame_min_fullmove', 40)),
        )

        self.mcts = MultiGameBatchMCTS(model, config, device)
        self.opponent_models_by_label = {}
        if isinstance(opponent_models_by_label, dict):
            for label, opp_model in opponent_models_by_label.items():
                if opp_model is not None:
                    self.opponent_models_by_label[str(label)] = opp_model
        elif opponent_model is not None:
            self.opponent_models_by_label[self.opponent_source_label] = opponent_model
        self.opponent_mcts_by_label = {
            str(label): MultiGameBatchMCTS(opp_model, config, device)
            for label, opp_model in self.opponent_models_by_label.items()
            if opp_model is not None
        }
        self.opponent_mcts = next(iter(self.opponent_mcts_by_label.values()), None)
        self.opponent_plan_labels = list(opponent_plan_labels or [])
        self._warned_missing_opponent_labels = set()
        self.history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
        self.share_trees = bool(rl_cfg.get('self_play_share_trees', True))

        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.num_simulations = int(config['reinforcement_learning']['mcts_simulations'])
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)
        self.max_moves = _resolve_selfplay_max_moves(config)
        self.auto_claim_draw = bool(rl_cfg.get('self_play_auto_claim_draw', False))
        self.claim_draw_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_draw_after_moves', self.max_moves)),
        )
        self.claim_repetition_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_repetition_after_moves', min(self.claim_draw_after_moves, 80))),
        )
        self.progress_report_interval_games = max(
            1,
            int(rl_cfg.get('self_play_progress_interval_games', 1)),
        )
        self.opening_diversity_enabled = bool(
            rl_cfg.get('self_play_opening_diversity_enabled', False)
        )
        self.opening_diversity_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_opening_diversity_fraction', 0.5))),
        )
        self.opening_diversity_min_plies = max(
            0,
            int(rl_cfg.get('self_play_opening_diversity_min_plies', 2)),
        )
        self.opening_diversity_max_plies = max(
            self.opening_diversity_min_plies,
            int(rl_cfg.get('self_play_opening_diversity_max_plies', 6)),
        )
        self.adjudication_enabled = bool(rl_cfg.get('self_play_adjudication_enabled', False))
        self.adjudication_min_moves = max(
            0,
            int(rl_cfg.get('self_play_adjudication_min_moves', 80)),
        )
        self.adjudication_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_adjudication_threshold', 0.92))),
        )
        self.adjudication_patience = max(
            1,
            int(rl_cfg.get('self_play_adjudication_patience', 6)),
        )
        self.syzygy = _get_syzygy_oracle(config)
        self.resignation_enabled = bool(rl_cfg.get('self_play_resignation_enabled', False))
        self.resignation_min_moves = max(
            0,
            int(rl_cfg.get('self_play_resignation_min_moves', 60)),
        )
        self.resignation_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_resignation_threshold', 0.92))),
        )
        self.resignation_patience = max(
            1,
            int(rl_cfg.get('self_play_resignation_patience', 3)),
        )
        self.resignation_disable_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_resignation_disable_fraction', 0.10))),
        )
        self.randomize_learner_color = bool(rl_cfg.get('self_play_randomize_learner_color', True))
        self.replay_dynamic_cap_enabled = bool(rl_cfg.get('replay_dynamic_cap_enabled', False))
        self.replay_cap_fraction_decisive = float(rl_cfg.get('replay_cap_fraction_decisive', 0.65))
        self.replay_cap_fraction_draw = float(rl_cfg.get('replay_cap_fraction_draw', 0.60))
        self.replay_cap_min_positions = int(rl_cfg.get('replay_cap_min_positions', 16))
        self.replay_cap_max_positions = int(rl_cfg.get('replay_cap_max_positions', 120))
        self.policy_target_max_moves = _resolve_replay_max_policy_targets(config)
        self.deblunder_threshold = max(
            0.0, float(rl_cfg.get('deblunder_threshold', 0.15))
        )
        self.deblunder_width = max(
            1e-6, float(rl_cfg.get('deblunder_width', 0.10))
        )
        self.deblunder_value_min_weight = max(
            0.0, min(1.0, float(rl_cfg.get('deblunder_value_min_weight', 0.35)))
        )
        self.deblunder_policy_boost_max = max(
            1.0, float(rl_cfg.get('deblunder_policy_boost_max', 1.35))
        )
        self.mcts_good_target_min_top_visit_prob = 0.55
        self.mcts_good_target_min_visit_gap = 0.12
        self.store_frozen_best_positions = bool(
            rl_cfg.get('self_play_store_frozen_best_positions', True)
        )
        self.replay_importance_top_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('replay_importance_top_fraction', 0.70))),
        )
        self.max_positions_per_game = max(
            0,
            int(rl_cfg.get('replay_max_positions_per_game', 32)),
        )
        self.playout_cap_randomization_enabled = bool(
            rl_cfg.get('mcts_playout_cap_randomization_enabled', False)
        )
        self.playout_cap_full_search_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('mcts_playout_cap_full_search_fraction', 0.40))),
        )
        self.playout_cap_fast_simulations = max(
            1,
            int(rl_cfg.get('mcts_playout_cap_fast_simulations', 48)),
        )
        self.hard_start_positions = list(rl_cfg.get('hard_start_positions', []) or [])

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0
        self._progress_base = 0
        self._plan_cursor = 0
        # Performance CSV/PNG should not depend on debug.rl.profile_training.
        # That flag controls console/debug verbosity, not collection of cheap counters.
        self.profile_enabled = True
        self._profile_stats = {}
        self.reset_profile_stats()

    def reset_profile_stats(self):
        self._profile_stats = {
            'move_selection_time': 0.0,
            'move_selection_calls': 0,
            'adjudication_time': 0.0,
            'adjudication_calls': 0,
            'syzygy_time': 0.0,
            'syzygy_calls': 0,
            'policy_target_build_time': 0.0,
            'policy_target_build_calls': 0,
            'policy_target_postgame_time': 0.0,
            'policy_target_postgame_calls': 0,
        }

    def _profile_add(self, key, value):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = float(self._profile_stats.get(key, 0.0)) + float(value)

    def _profile_inc(self, key, value=1):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = int(self._profile_stats.get(key, 0)) + int(value)

    def _merge_profile_stats(self, target, source):
        for key, value in dict(source or {}).items():
            if isinstance(value, (int, np.integer)):
                target[key] = int(target.get(key, 0)) + int(value)
            else:
                target[key] = float(target.get(key, 0.0)) + float(value)

    def _aggregate_engine_profile_stats(self):
        aggregated = dict(self._profile_stats)
        self._merge_profile_stats(
            aggregated,
            {f"learner_mcts_{k}": v for k, v in self.mcts.get_profile_stats().items()},
        )
        for label, opponent_mcts in self.opponent_mcts_by_label.items():
            self._merge_profile_stats(
                aggregated,
                {f"opponent_mcts_{label}_{k}": v for k, v in opponent_mcts.get_profile_stats().items()},
            )
        search_many_time = float(aggregated.get('learner_mcts_search_many_time', 0.0))
        batch_expand_time = float(aggregated.get('learner_mcts_batch_expand_eval_time', 0.0))
        batch_expand_calls = int(aggregated.get('learner_mcts_batch_expand_eval_calls', 0) or 0)
        board_tensor_time = float(aggregated.get('learner_mcts_board_to_tensor_time', 0.0))
        board_tensor_calls = int(aggregated.get('learner_mcts_board_to_tensor_calls', 0) or 0)
        nn_time = float(aggregated.get('learner_mcts_nn_inference_time', 0.0))
        nn_calls = int(aggregated.get('learner_mcts_nn_inference_calls', 0) or 0)
        nn_batch_items = int(aggregated.get('learner_mcts_nn_inference_batch_items', 0) or 0)
        selection_node_traversals = int(
            aggregated.get('learner_mcts_selection_node_traversals', 0) or 0
        )
        nn_h2d_time = float(aggregated.get('learner_mcts_nn_h2d_time', 0.0))
        nn_gpu_forward_time = float(aggregated.get('learner_mcts_nn_gpu_forward_time', 0.0))
        nn_gpu_postprocess_time = float(aggregated.get('learner_mcts_nn_gpu_postprocess_time', 0.0))
        nn_d2h_time = float(aggregated.get('learner_mcts_nn_d2h_time', 0.0))
        nn_legal_move_items = int(aggregated.get('learner_mcts_nn_legal_move_items', 0) or 0)
        central_requests = int(aggregated.get('learner_mcts_central_inference_requests', 0) or 0)
        central_batch_items = int(aggregated.get('learner_mcts_central_inference_server_batch_items', 0) or 0)
        central_request_put_time = float(aggregated.get('learner_mcts_central_inference_request_put_time', 0.0) or 0.0)
        central_remote_wait_time = float(aggregated.get('learner_mcts_central_inference_remote_wait_time', 0.0) or 0.0)
        central_server_queue_wait_time = float(aggregated.get('learner_mcts_central_inference_server_queue_wait_time', 0.0) or 0.0)
        central_server_descriptor_queue_wait_time = float(
            aggregated.get(
                'learner_mcts_central_inference_server_descriptor_queue_wait_time',
                0.0,
            ) or 0.0
        )
        central_server_batch_coalesce_wait_time = float(
            aggregated.get(
                'learner_mcts_central_inference_server_batch_coalesce_wait_time',
                0.0,
            ) or 0.0
        )
        central_server_total_time = float(aggregated.get('learner_mcts_central_inference_server_total_time', 0.0) or 0.0)
        central_server_concat_time = float(aggregated.get('learner_mcts_central_inference_server_concat_time', 0.0) or 0.0)
        central_server_h2d_time = float(aggregated.get('learner_mcts_central_inference_server_h2d_time', 0.0) or 0.0)
        central_server_forward_time = float(aggregated.get('learner_mcts_central_inference_server_forward_time', 0.0) or 0.0)
        central_server_d2h_time = float(aggregated.get('learner_mcts_central_inference_server_d2h_time', 0.0) or 0.0)
        for label in self.opponent_mcts_by_label.keys():
            prefix = f"opponent_mcts_{label}_"
            search_many_time += float(aggregated.get(prefix + 'search_many_time', 0.0))
            batch_expand_time += float(aggregated.get(prefix + 'batch_expand_eval_time', 0.0))
            batch_expand_calls += int(aggregated.get(prefix + 'batch_expand_eval_calls', 0) or 0)
            board_tensor_time += float(aggregated.get(prefix + 'board_to_tensor_time', 0.0))
            board_tensor_calls += int(aggregated.get(prefix + 'board_to_tensor_calls', 0) or 0)
            nn_time += float(aggregated.get(prefix + 'nn_inference_time', 0.0))
            nn_calls += int(aggregated.get(prefix + 'nn_inference_calls', 0) or 0)
            nn_batch_items += int(aggregated.get(prefix + 'nn_inference_batch_items', 0) or 0)
            selection_node_traversals += int(
                aggregated.get(prefix + 'selection_node_traversals', 0) or 0
            )
            nn_h2d_time += float(aggregated.get(prefix + 'nn_h2d_time', 0.0))
            nn_gpu_forward_time += float(aggregated.get(prefix + 'nn_gpu_forward_time', 0.0))
            nn_gpu_postprocess_time += float(aggregated.get(prefix + 'nn_gpu_postprocess_time', 0.0))
            nn_d2h_time += float(aggregated.get(prefix + 'nn_d2h_time', 0.0))
            nn_legal_move_items += int(aggregated.get(prefix + 'nn_legal_move_items', 0) or 0)
            central_requests += int(aggregated.get(prefix + 'central_inference_requests', 0) or 0)
            central_batch_items += int(aggregated.get(prefix + 'central_inference_server_batch_items', 0) or 0)
            central_request_put_time += float(aggregated.get(prefix + 'central_inference_request_put_time', 0.0) or 0.0)
            central_remote_wait_time += float(aggregated.get(prefix + 'central_inference_remote_wait_time', 0.0) or 0.0)
            central_server_queue_wait_time += float(aggregated.get(prefix + 'central_inference_server_queue_wait_time', 0.0) or 0.0)
            central_server_descriptor_queue_wait_time += float(
                aggregated.get(
                    prefix + 'central_inference_server_descriptor_queue_wait_time',
                    0.0,
                ) or 0.0
            )
            central_server_batch_coalesce_wait_time += float(
                aggregated.get(
                    prefix + 'central_inference_server_batch_coalesce_wait_time',
                    0.0,
                ) or 0.0
            )
            central_server_total_time += float(aggregated.get(prefix + 'central_inference_server_total_time', 0.0) or 0.0)
            central_server_concat_time += float(aggregated.get(prefix + 'central_inference_server_concat_time', 0.0) or 0.0)
            central_server_h2d_time += float(aggregated.get(prefix + 'central_inference_server_h2d_time', 0.0) or 0.0)
            central_server_forward_time += float(aggregated.get(prefix + 'central_inference_server_forward_time', 0.0) or 0.0)
            central_server_d2h_time += float(aggregated.get(prefix + 'central_inference_server_d2h_time', 0.0) or 0.0)
        aggregated['mcts_search_many_time'] = float(search_many_time)
        aggregated['mcts_batch_expand_eval_time'] = float(batch_expand_time)
        aggregated['mcts_batch_expand_eval_calls'] = int(batch_expand_calls)
        aggregated['mcts_board_to_tensor_time'] = float(board_tensor_time)
        aggregated['mcts_board_to_tensor_calls'] = int(board_tensor_calls)
        aggregated['mcts_nn_inference_time'] = float(nn_time)
        aggregated['mcts_nn_inference_calls'] = int(nn_calls)
        aggregated['mcts_nn_inference_batch_items'] = int(nn_batch_items)
        aggregated['mcts_selection_node_traversals'] = int(selection_node_traversals)
        aggregated['mcts_nn_h2d_time'] = float(nn_h2d_time)
        aggregated['mcts_nn_gpu_forward_time'] = float(nn_gpu_forward_time)
        aggregated['mcts_nn_gpu_postprocess_time'] = float(nn_gpu_postprocess_time)
        aggregated['mcts_nn_d2h_time'] = float(nn_d2h_time)
        aggregated['mcts_nn_legal_move_items'] = int(nn_legal_move_items)
        aggregated['mcts_central_inference_requests'] = int(central_requests)
        aggregated['mcts_central_inference_server_batch_items'] = int(central_batch_items)
        aggregated['mcts_central_inference_request_put_time'] = float(central_request_put_time)
        aggregated['mcts_central_inference_remote_wait_time'] = float(central_remote_wait_time)
        aggregated['mcts_central_inference_server_queue_wait_time'] = float(central_server_queue_wait_time)
        aggregated['mcts_central_inference_server_descriptor_queue_wait_time'] = float(
            central_server_descriptor_queue_wait_time
        )
        aggregated['mcts_central_inference_server_batch_coalesce_wait_time'] = float(
            central_server_batch_coalesce_wait_time
        )
        aggregated['mcts_central_inference_server_total_time'] = float(central_server_total_time)
        aggregated['mcts_central_inference_server_concat_time'] = float(central_server_concat_time)
        aggregated['mcts_central_inference_server_h2d_time'] = float(central_server_h2d_time)
        aggregated['mcts_central_inference_server_forward_time'] = float(central_server_forward_time)
        aggregated['mcts_central_inference_server_d2h_time'] = float(central_server_d2h_time)
        central_extra_metrics = (
            'central_inference_shared_requests',
            'central_inference_shared_bytes_avoided',
            'central_inference_shared_slot_wait_time',
            'central_inference_cache_queries',
            'central_inference_cache_bypassed_positions',
            'central_inference_cache_hits',
            'central_inference_dedup_hits',
            'central_inference_cache_suspensions',
            'central_inference_cache_reactivations',
            'central_inference_nn_evaluated_positions',
            'central_inference_server_cache_lookup_time',
            'central_inference_server_staging_copy_time',
            'central_inference_gpu_batch_fill_sum',
        )
        for metric in central_extra_metrics:
            total = aggregated.get(f'learner_mcts_{metric}', 0) or 0
            for label in self.opponent_mcts_by_label.keys():
                total += aggregated.get(f'opponent_mcts_{label}_{metric}', 0) or 0
            aggregated[f'mcts_{metric}'] = total
        extra_mcts_time_metrics = [
            'search_root_setup_time',
            'search_selection_time',
            'search_backprop_time',
            'search_metadata_time',
            'board_materialize_time',
            'terminal_checks_time',
            'batch_expand_dedup_terminal_time',
            'batch_expand_legal_moves_time',
            'batch_expand_move_index_time',
            'batch_expand_tensor_pack_time',
            'batch_expand_history_time',
            'batch_expand_input_pack_time',
            'batch_expand_legal_index_pack_time',
            'batch_expand_cpu_policy_time',
            'batch_expand_value_fanout_time',
        ]
        for metric in extra_mcts_time_metrics:
            total = float(aggregated.get(f'learner_mcts_{metric}', 0.0) or 0.0)
            for label in self.opponent_mcts_by_label.keys():
                total += float(aggregated.get(f'opponent_mcts_{label}_{metric}', 0.0) or 0.0)
            aggregated[f'mcts_{metric}'] = float(total)
        aggregated['average_batch_size'] = float(nn_batch_items / nn_calls) if nn_calls > 0 else 0.0
        aggregated['central_average_batch_size'] = (
            float(central_batch_items / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_remote_wait_ms_per_request'] = (
            1000.0 * float(central_remote_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_request_put_ms_per_request'] = (
            1000.0 * float(central_request_put_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_queue_wait_ms_per_request'] = (
            1000.0 * float(central_server_queue_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_descriptor_queue_wait_ms_per_request'] = (
            1000.0 * float(
                central_server_descriptor_queue_wait_time / central_requests
            ) if central_requests > 0 else 0.0
        )
        aggregated['central_batch_coalesce_wait_ms_per_request'] = (
            1000.0 * float(
                central_server_batch_coalesce_wait_time / central_requests
            ) if central_requests > 0 else 0.0
        )
        aggregated['central_server_forward_ms_per_request'] = (
            1000.0 * float(central_server_forward_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_h2d_ms_per_request'] = (
            1000.0 * float(central_server_h2d_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_d2h_ms_per_request'] = (
            1000.0 * float(central_server_d2h_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_concat_ms_per_request'] = (
            1000.0 * float(central_server_concat_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_total_ms_per_request'] = (
            1000.0 * float(central_server_total_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['average_legal_moves_per_position'] = (
            float(nn_legal_move_items / nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        aggregated['average_legal_moves_per_batch'] = (
            float(nn_legal_move_items / nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_batch_ms'] = (
            1000.0 * float(nn_time) / float(nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_position_ms'] = (
            1000.0 * float(nn_time) / float(nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        worker_nn_wait_share_pct = 100.0 * float(nn_time) / max(1e-8, float(search_many_time))
        aggregated['worker_nn_wait_share_pct'] = float(
            max(0.0, min(100.0, worker_nn_wait_share_pct))
        )
        return aggregated

    def _prune_policy_target_visits(self, visit_counts):
        if not visit_counts:
            return visit_counts

        items = sorted(
            ((move, float(count)) for move, count in visit_counts.items() if float(count) > 0.0),
            key=lambda pair: (-pair[1], pair[0].uci()),
        )
        if not items:
            return visit_counts
        kept = items[:self.policy_target_max_moves]
        return {move: int(max(1.0, round(count))) for move, count in kept}

    def _prune_policy_target_weights(self, policy_weights):
        """Limit a Gumbel improved-policy target without quantizing probabilities."""
        if not policy_weights:
            return policy_weights
        items = sorted(
            ((move, float(weight)) for move, weight in policy_weights.items() if float(weight) > 0.0),
            key=lambda pair: (-pair[1], pair[0].uci()),
        )
        return dict(items[:self.policy_target_max_moves])

    def _mcts_phase_for_board(self, board):
        try:
            fullmove_number = int(getattr(board, 'fullmove_number', 1) or 1)
        except (TypeError, ValueError):
            fullmove_number = 1
        if fullmove_number <= int(self.mcts_phase_opening_max_fullmove):
            return 'opening'
        if fullmove_number >= int(self.mcts_phase_endgame_min_fullmove):
            return 'endgame'
        return 'middlegame'

    def _should_store_policy_position(self, learner_turn, game_opponent_mcts, opponent_label):
        if learner_turn or game_opponent_mcts is None:
            return True
        label = str(opponent_label or "")
        if label == "best":
            return bool(self.store_frozen_best_positions)
        return False

    @staticmethod
    def _candidate_importance(item):
        try:
            return float(item.get('importance_score', 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _select_evenly_spaced_candidates(candidates, effective_cap):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        ordered = sorted(candidates, key=lambda item: int(item['history_idx']))
        positions = np.linspace(0, len(ordered) - 1, num=effective_cap)
        selected_history = []
        seen_history = set()
        for pos in positions:
            idx = int(round(float(pos)))
            idx = max(0, min(idx, len(ordered) - 1))
            history_idx = int(ordered[idx]['history_idx'])
            if history_idx in seen_history:
                continue
            seen_history.add(history_idx)
            selected_history.append(history_idx)
        if len(selected_history) < effective_cap:
            for item in ordered:
                history_idx = int(item['history_idx'])
                if history_idx in seen_history:
                    continue
                seen_history.add(history_idx)
                selected_history.append(history_idx)
                if len(selected_history) >= effective_cap:
                    break
        selected_lookup = set(selected_history[:effective_cap])
        return [item for item in ordered if int(item['history_idx']) in selected_lookup]

    def _select_top_scored_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        del history_len
        ranked = sorted(
            candidates,
            key=lambda item: (-self._candidate_importance(item), int(item['history_idx'])),
        )
        top_quota = int(round(effective_cap * self.replay_importance_top_fraction))
        top_quota = max(1, min(int(effective_cap), top_quota))

        selected = []
        selected_history = set()
        for item in ranked:
            history_idx = int(item['history_idx'])
            if history_idx in selected_history:
                continue
            selected.append(item)
            selected_history.add(history_idx)
            if len(selected) >= top_quota:
                break

        remaining = int(effective_cap - len(selected))
        if remaining > 0:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_history
            ]
            for item in self._select_evenly_spaced_candidates(leftovers, remaining):
                history_idx = int(item['history_idx'])
                if history_idx in selected_history:
                    continue
                selected.append(item)
                selected_history.add(history_idx)
                if len(selected) >= effective_cap:
                    break

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected[:effective_cap]

    def _allocate_phase_stratified_caps(self, buckets, effective_cap):
        bucket_count = len(buckets)
        allocations = [0] * bucket_count
        if effective_cap <= 0 or bucket_count <= 0:
            return allocations

        bucket_sizes = [len(bucket) for bucket in buckets]
        non_empty = [idx for idx, size in enumerate(bucket_sizes) if size > 0]
        if not non_empty:
            return allocations

        if effective_cap < len(non_empty):
            desired = []
            for idx in range(bucket_count):
                if bucket_sizes[idx] <= 0:
                    desired.append(0.0)
                elif idx == bucket_count - 1:
                    desired.append(1.0)
                elif idx == bucket_count - 2:
                    desired.append(0.75)
                else:
                    desired.append(0.5)
        else:
            # Preserve calm opening and middlegame positions instead of letting
            # tactical/endgame importance dominate a heavily capped game.
            desired = [0.30, 0.45, 0.25]

        remaining = int(effective_cap)
        for idx in non_empty:
            allocations[idx] = 1
            remaining -= 1

        if remaining <= 0:
            return allocations

        desired_counts = [float(effective_cap) * desired[idx] for idx in range(bucket_count)]
        remainders = []
        for idx in range(bucket_count):
            if bucket_sizes[idx] <= allocations[idx]:
                continue
            extra = int(math.floor(max(0.0, desired_counts[idx] - allocations[idx])))
            if extra > 0:
                grant = min(extra, bucket_sizes[idx] - allocations[idx], remaining)
                allocations[idx] += grant
                remaining -= grant
            remainder = max(0.0, desired_counts[idx] - allocations[idx])
            remainders.append((remainder, idx))

        while remaining > 0:
            progressed = False
            remainders.sort(key=lambda pair: (-pair[0], pair[1]))
            for _, idx in remainders:
                if remaining <= 0:
                    break
                if allocations[idx] >= bucket_sizes[idx]:
                    continue
                allocations[idx] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        if remaining > 0:
            fill_order = sorted(
                non_empty,
                key=lambda idx: (
                    -bucket_sizes[idx],
                    -desired_counts[idx],
                    -idx,
                ),
            )
            while remaining > 0:
                progressed = False
                for idx in fill_order:
                    if remaining <= 0:
                        break
                    if allocations[idx] >= bucket_sizes[idx]:
                        continue
                    allocations[idx] += 1
                    remaining -= 1
                    progressed = True
                if not progressed:
                    break

        return allocations

    def _select_phase_stratified_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)

        if effective_cap < 3:
            return self._select_top_scored_candidates(candidates, effective_cap, history_len)

        denom = float(max(1, history_len - 1))
        buckets = [[], [], []]
        for item in candidates:
            progress = float(item['history_idx']) / denom
            if progress < (1.0 / 3.0):
                buckets[0].append(item)
            elif progress < (2.0 / 3.0):
                buckets[1].append(item)
            else:
                buckets[2].append(item)

        allocations = self._allocate_phase_stratified_caps(buckets, effective_cap)
        selected = []
        selected_ids = set()
        for bucket, bucket_cap in zip(buckets, allocations):
            for item in self._select_top_scored_candidates(bucket, bucket_cap, history_len):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        if len(selected) < effective_cap:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_ids
            ]
            for item in self._select_top_scored_candidates(
                leftovers,
                effective_cap - len(selected),
                history_len,
            ):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        return selected

    def _select_candidates_with_source_balance(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)

        buckets = {}
        for item in candidates:
            source_code = int(item.get('source_code', _REPLAY_SOURCE_UNKNOWN))
            buckets.setdefault(source_code, []).append(item)
        non_empty_sources = [
            source_code for source_code, bucket in buckets.items()
            if len(bucket) > 0
        ]
        if len(non_empty_sources) <= 1:
            return self._select_phase_stratified_candidates(candidates, effective_cap, history_len)

        total_candidates = max(1, len(candidates))
        allocations = {}
        remaining = int(effective_cap)
        if effective_cap >= len(non_empty_sources):
            for source_code in non_empty_sources:
                allocations[source_code] = 1
                remaining -= 1
        else:
            ranked_sources = sorted(
                non_empty_sources,
                key=lambda source_code: (
                    -max(self._candidate_importance(item) for item in buckets[source_code]),
                    source_code,
                ),
            )
            for source_code in ranked_sources[:effective_cap]:
                allocations[source_code] = 1
            remaining = 0

        desired = {
            source_code: float(effective_cap) * len(buckets[source_code]) / float(total_candidates)
            for source_code in non_empty_sources
        }
        while remaining > 0:
            progressed = False
            ranked_sources = sorted(
                non_empty_sources,
                key=lambda source_code: (
                    -(desired[source_code] - allocations.get(source_code, 0)),
                    -len(buckets[source_code]),
                    source_code,
                ),
            )
            for source_code in ranked_sources:
                if remaining <= 0:
                    break
                current = int(allocations.get(source_code, 0))
                if current >= len(buckets[source_code]):
                    continue
                allocations[source_code] = current + 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        selected = []
        selected_ids = set()
        for source_code in sorted(non_empty_sources):
            bucket_cap = int(allocations.get(source_code, 0))
            if bucket_cap <= 0:
                continue
            bucket = buckets[source_code]
            bucket_selected = self._select_phase_stratified_candidates(
                bucket,
                bucket_cap,
                history_len,
            )
            for item in bucket_selected:
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        if len(selected) < effective_cap:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_ids
            ]
            remaining_cap = int(effective_cap - len(selected))
            refill = self._select_phase_stratified_candidates(
                leftovers,
                remaining_cap,
                history_len,
            )
            for item in refill:
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)
                if len(selected) >= effective_cap:
                    break

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected[:effective_cap]

    def _select_history_indices_to_keep(self, candidates, history_len, is_decisive=False):
        if not candidates:
            return [], 0, 0

        filtered = list(candidates)
        curriculum_dropped = 0

        # Determine effective cap limits
        if self.replay_dynamic_cap_enabled:
            fraction = self.replay_cap_fraction_decisive if is_decisive else self.replay_cap_fraction_draw
            effective_cap = int(math.ceil(len(filtered) * fraction))
            effective_cap = max(self.replay_cap_min_positions, effective_cap)
            effective_cap = min(self.replay_cap_max_positions, effective_cap)
        else:
            effective_cap = self.max_positions_per_game

        if effective_cap <= 0 or len(filtered) <= effective_cap:
            selected = filtered
            cap_dropped = 0
        else:
            selected = self._select_candidates_with_source_balance(
                filtered,
                effective_cap,
                history_len,
            )
            cap_dropped = len(filtered) - len(selected)

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected, int(curriculum_dropped), int(cap_dropped)

    def _build_history_selection_candidates(self, gs):
        candidates = []
        for history_idx, history_entry in enumerate(gs['game_history']):
            importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
            candidates.append({
                'history_idx': history_idx,
                'importance_score': importance_score,
                'source_code': int(history_entry[6]) if len(history_entry) > 6 else _REPLAY_SOURCE_UNKNOWN,
            })
        return candidates

    def _build_training_position_from_history_entry(
        self,
        gs,
        history_idx,
        history_len,
        outcome,
        draw_value_target=0.0,
    ):
        history_entry = gs['game_history'][int(history_idx)]
        history_count, policy_indices, policy_values, turn = history_entry[:4]
        importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
        policy_weight = float(history_entry[5]) if len(history_entry) > 5 else 1.0
        source_code = int(history_entry[6]) if len(history_entry) > 6 else _REPLAY_SOURCE_UNKNOWN
        legal_indices = history_entry[7] if len(history_entry) > 7 and history_entry[7] is not None else policy_indices
        fen = history_entry[8] if len(history_entry) > 8 else None
        root_q = float(history_entry[9]) if len(history_entry) > 9 else float('nan')
        history_fens = tuple(history_entry[10] or ()) if len(history_entry) > 10 else ()
        search_changed_top = bool(history_entry[11]) if len(history_entry) > 11 else False
        search_q_delta = float(history_entry[12]) if len(history_entry) > 12 else float('nan')
        best_q = float(history_entry[13]) if len(history_entry) > 13 else float('nan')
        played_q = float(history_entry[14]) if len(history_entry) > 14 else float('nan')
        orig_q = float(history_entry[15]) if len(history_entry) > 15 else float('nan')
        policy_kld = float(history_entry[16]) if len(history_entry) > 16 else float('nan')
        search_visits = int(history_entry[17]) if len(history_entry) > 17 else 0

        if outcome == 0.0:
            value = draw_value_target
        else:
            signed_outcome = outcome if turn == chess.WHITE else -outcome
            value = float(signed_outcome)
        value_weight = 1.0
        moves_left = max(0, len(gs.get('board_history', [])) - int(history_count))

        return {
            'history_idx': int(history_idx),
            'history_count': history_count,
            'policy_indices': policy_indices,
            'policy_values': policy_values,
            'turn': turn,
            'value': value,
            'importance_score': importance_score,
            'policy_weight': policy_weight,
            'value_weight': value_weight,
            'moves_left': moves_left,
            'legal_indices': legal_indices,
            'source_code': source_code,
            'fen': fen,
            'root_q': root_q,
            'history_fens': history_fens,
            'search_changed_top': search_changed_top,
            'search_q_delta': search_q_delta,
            'best_q': best_q,
            'played_q': played_q,
            'orig_q': orig_q,
            'policy_kld': policy_kld,
            'search_visits': search_visits,
        }

    def _deblunder_weights_for_history(self, gs, outcome):
        """Return per-ply policy/value weights without rewriting final results.

        A later exploratory blunder makes the final result a noisy estimate of
        earlier positions' best-play value.  Those winner-WDL rows are reduced,
        while a reliable MCTS correction gets a bounded policy emphasis.
        """
        history = list(gs.get('game_history', []) or [])
        count = len(history)
        if count <= 0:
            return [], []
        regrets = np.zeros(count, dtype=np.float32)
        correction_strengths = np.zeros(count, dtype=np.float32)
        target_errors = np.full(count, 2.0, dtype=np.float32)
        base_policy_weights = np.ones(count, dtype=np.float32)
        for idx, entry in enumerate(history):
            base_policy_weight = float(entry[5]) if len(entry) > 5 else 1.0
            base_policy_weights[idx] = base_policy_weight
            best_q = float(entry[13]) if len(entry) > 13 else float('nan')
            played_q = float(entry[14]) if len(entry) > 14 else float('nan')
            q_delta = float(entry[12]) if len(entry) > 12 else float('nan')
            turn = entry[3]
            signed_outcome = float(outcome if turn == chess.WHITE else -outcome)
            target_error = (
                abs(best_q - signed_outcome)
                if math.isfinite(best_q)
                else 2.0
            )
            target_errors[idx] = target_error
            regret = (
                max(0.0, best_q - played_q)
                if math.isfinite(best_q) and math.isfinite(played_q)
                else 0.0
            )
            regrets[idx] = regret
            correction = max(
                regret,
                max(0.0, q_delta) if math.isfinite(q_delta) else 0.0,
            )
            progress = max(
                0.0,
                min(1.0, (correction - self.deblunder_threshold) / self.deblunder_width),
            )
            smooth = progress * progress * (3.0 - 2.0 * progress)
            correction_strengths[idx] = float(smooth)

        future_regret = np.maximum.accumulate(regrets[::-1])[::-1]
        value_weights = np.ones(count, dtype=np.float32)
        for idx, regret in enumerate(future_regret):
            progress = max(
                0.0,
                min(1.0, (float(regret) - self.deblunder_threshold) / self.deblunder_width),
            )
            smooth = progress * progress * (3.0 - 2.0 * progress)
            value_weights[idx] = float(
                1.0 - (1.0 - self.deblunder_value_min_weight) * smooth
            )
        policy_weights = np.ones(count, dtype=np.float32)
        for idx, base_policy_weight in enumerate(base_policy_weights):
            if base_policy_weight <= 0.0:
                # PCR-fast searches intentionally never supervise policy.
                policy_weights[idx] = 0.0
                continue
            outcome_reliability = math.exp(-float(target_errors[idx]) / 0.50)
            # If a future blunder already made the game outcome unreliable for
            # this row, do not use that same contaminated result to reject an
            # otherwise well-searched policy correction.
            value_reliability = float(value_weights[idx])
            reliability = (
                (1.0 - value_reliability)
                + value_reliability * outcome_reliability
            )
            inherited_excess = max(0.0, float(base_policy_weight) - 1.0) * reliability
            policy_weights[idx] = min(
                self.deblunder_policy_boost_max,
                max(0.0, min(1.0, float(base_policy_weight)))
                + inherited_excess
                + (self.deblunder_policy_boost_max - 1.0)
                * float(correction_strengths[idx])
                * reliability,
            )
        return policy_weights.tolist(), value_weights.tolist()

    def _compute_position_importance(self, board, move, visit_counts, root, search_metadata=None):
        importance = 1.0

        if visit_counts:
            visits = np.asarray(list(visit_counts.values()), dtype=np.float32)
            total_visits = float(visits.sum())
            if total_visits > 0.0:
                probs = visits / total_visits
                top_prob = float(probs.max())
                entropy = 0.0
                if probs.size > 1:
                    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
                    entropy /= float(np.log(probs.size))
                importance += 0.35 * (1.0 - top_prob)
                importance += 0.30 * entropy

        if root is not None:
            root_visits = int(getattr(root, 'visit_count', 0) or 0)
            if root_visits > 0:
                root_value = float(root.value_sum / max(1, root_visits))
                importance += 0.30 * abs(root_value)

        # MCTS disagreements are precisely the positions that can teach the raw
        # policy something new. Keep them ahead of merely tactical/evenly-spaced
        # positions when a game's replay cap is tight.
        if isinstance(search_metadata, dict):
            if float(search_metadata.get('prior_mcts_agree', 1.0) or 0.0) < 0.5:
                importance += 0.45
                try:
                    q_delta = float(search_metadata.get('mcts_q_delta', 0.0) or 0.0)
                except (TypeError, ValueError):
                    q_delta = 0.0
                importance += 0.20 * min(1.0, max(0.0, q_delta) / 0.30)
            try:
                policy_kl = float(search_metadata.get('mcts_policy_kl', 0.0) or 0.0)
            except (TypeError, ValueError):
                policy_kl = 0.0
            importance += 0.10 * min(1.0, max(0.0, policy_kl) / 0.20)

        if chess.is_capture(board, move):
            importance += 0.30
            gain = self.mcts._captured_piece_value(board, move) - self.mcts._moving_piece_value(board, move)
            if gain > 0.0:
                importance += 0.20 * min(1.0, gain / 4.0)
        if move.promotion is not None:
            importance += 0.30
        try:
            if chess.gives_check(board, move):
                importance += 0.15
        except Exception:
            pass
        last_move = chess.last_move(board)
        if last_move is not None and move.destination == last_move.destination:
            importance += 0.10

        return float(importance)

    def _append_selected_positions_from_game(
        self,
        positions,
        gs,
        outcome,
        *,
        history_positions,
    ):
        postgame_t0 = time.perf_counter() if self.profile_enabled else None
        history_len = len(gs['game_history'])
        selection_candidates = self._build_history_selection_candidates(gs)
        selected_candidates, curriculum_dropped, cap_dropped = self._select_history_indices_to_keep(
            selection_candidates,
            history_len,
            is_decisive=(outcome != 0.0),
        )
        deblunder_policy_weights, deblunder_value_weights = (
            self._deblunder_weights_for_history(gs, outcome)
        )

        for candidate in selected_candidates:
            item = self._build_training_position_from_history_entry(
                gs,
                candidate['history_idx'],
                history_len,
                outcome,
                draw_value_target=0.0,
            )
            history_idx = int(candidate['history_idx'])
            if history_idx < len(deblunder_policy_weights):
                item['policy_weight'] = float(deblunder_policy_weights[history_idx])
            if history_idx < len(deblunder_value_weights):
                item['value_weight'] = float(deblunder_value_weights[history_idx])
            board_tensor_np = _build_history_tensor_from_encoded(
                turn=item['turn'],
                encoded_history=gs['board_history'],
                history_count=item['history_count'],
                history_positions=history_positions,
                empty_history_tensor=_EMPTY_HISTORY_TENSOR,
            )
            if board_tensor_np.dtype != np.float32:
                board_tensor_np = board_tensor_np.astype(np.float32, copy=False)
            board_tensor = torch.from_numpy(board_tensor_np)
            positions.append((
                board_tensor,
                item['policy_indices'],
                item['policy_values'],
                torch.tensor([item['value']], dtype=torch.float32),
                float(item.get('importance_score', 0.0)),
                float(item.get('policy_weight', 1.0)),
                float(item.get('value_weight', 1.0)),
                int(item.get('source_code', _REPLAY_SOURCE_UNKNOWN)),
                float(item.get('moves_left', 0.0)),
                item.get('legal_indices', item['policy_indices']),
                item.get('fen'),
                item.get('root_q', float('nan')),
                item.get('history_fens', ()),
                bool(item.get('search_changed_top', False)),
                float(item.get('search_q_delta', float('nan'))),
                float(item.get('best_q', float('nan'))),
                float(item.get('played_q', float('nan'))),
                float(item.get('orig_q', float('nan'))),
                float(item.get('policy_kld', float('nan'))),
                int(item.get('search_visits', 0)),
            ))
        if self.profile_enabled:
            self._profile_add('policy_target_postgame_time', time.perf_counter() - postgame_t0)
            self._profile_inc('policy_target_postgame_calls', 1)
        return history_len, int(curriculum_dropped), int(cap_dropped)

    def _should_auto_claim_draw(self, board, move_count):
        if not self.auto_claim_draw:
            return False
        if (
            move_count >= self.claim_repetition_after_moves
            and chess.can_claim_threefold_repetition(board)
        ):
            return True
        return bool(
            move_count >= self.claim_draw_after_moves
            and chess.can_claim_draw(board)
        )

    def _sample_opening_prefix(self):
        if not self.opening_diversity_enabled:
            return ()
        if self.opening_diversity_fraction <= 0.0:
            return ()
        if np.random.random() > self.opening_diversity_fraction:
            return ()

        line = _SELFPLAY_SELFPLAY_OPENING_LINES[
            int(np.random.randint(0, len(_SELFPLAY_SELFPLAY_OPENING_LINES)))
        ]
        max_plies = min(len(line), self.opening_diversity_max_plies)
        min_plies = min(max_plies, self.opening_diversity_min_plies)
        if max_plies <= 0:
            return ()
        if min_plies >= max_plies:
            plies = max_plies
        else:
            plies = int(np.random.randint(min_plies, max_plies + 1))
        return line[:plies]

    def _maybe_adjudicate_game(self, gs, root, board):
        if not self.adjudication_enabled or root is None:
            return None
        if gs['move_count'] < self.adjudication_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        white_value = root_value if board.turn == chess.WHITE else -root_value
        threshold = self.adjudication_threshold

        if white_value >= threshold:
            gs['white_advantage_streak'] = int(gs.get('white_advantage_streak', 0)) + 1
            gs['black_advantage_streak'] = 0
            if gs['white_advantage_streak'] >= self.adjudication_patience:
                return '1-0'
        elif white_value <= -threshold:
            gs['black_advantage_streak'] = int(gs.get('black_advantage_streak', 0)) + 1
            gs['white_advantage_streak'] = 0
            if gs['black_advantage_streak'] >= self.adjudication_patience:
                return '0-1'
        else:
            gs['white_advantage_streak'] = 0
            gs['black_advantage_streak'] = 0

        return None

    @staticmethod
    def _clear_game_search_state(gs):
        gs['root'] = None
        gs['opponent_root'] = None
        gs['_root_synced'] = False
        gs['_opponent_root_synced'] = False

    @staticmethod
    def _advance_search_root(root, move):
        if root is None:
            return None, False
        edge_idx = root.edges._get_move_index(move) if root.edges is not None else None
        child = (
            root.edges.nodes[int(edge_idx)]
            if edge_idx is not None and int(edge_idx) < len(root.edges.nodes)
            else None
        )
        if child is None:
            return None, False
        _ = child.board
        return child.detach_as_root(), True

    def _maybe_finish_with_syzygy(self, gs, board):
        if self.syzygy is None or chess.is_game_over(board, claim_draw=False):
            return None
        if not self.syzygy.can_probe(board):
            return None
        gs['syzygy_probe_positions'] = int(gs.get('syzygy_probe_positions', 0)) + 1
        wdl = self.syzygy.probe_wdl(board)
        if wdl is None:
            return None
        gs['syzygy_probe_hits'] = int(gs.get('syzygy_probe_hits', 0)) + 1

        value = _syzygy_wdl_to_value(wdl)
        if value > 0:
            result = '1-0' if board.turn == chess.WHITE else '0-1'
        elif value < 0:
            result = '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            result = '1/2-1/2'

        gs['done'] = True
        gs['syzygy_result'] = result
        self._clear_game_search_state(gs)
        self._mark_game_completed(gs)
        return result

    def _maybe_resign_game(self, gs, root, board):
        if not self.resignation_enabled or gs.get('resignation_disabled', False):
            return None
        if root is None or gs['move_count'] < self.resignation_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        if root_value <= -self.resignation_threshold:
            gs['resign_streak'] = int(gs.get('resign_streak', 0)) + 1
            if gs['resign_streak'] >= self.resignation_patience:
                return '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            gs['resign_streak'] = 0
        return None

    def _apply_opening_prefix(self, gs):
        prefix = self._sample_opening_prefix()
        if not prefix:
            return

        board = gs['board']
        for uci in prefix:
            try:
                move = chess.move_from_uci(uci)
            except Exception:
                break
            if move is None or move not in chess.legal_moves(board):
                break

            gs['board_history'].append(self.mcts._encode_history_entry(board))
            chess.apply_move(board, move)
            gs['move_count'] += 1
            _record_position_count(gs['position_counts'], board)

            if chess.is_game_over(board, claim_draw=False) or gs['move_count'] >= self.max_moves:
                gs['done'] = True
                break

    def play_games(self, num_games):
        all_positions = []
        game_lengths = []
        self._games_completed = 0
        self._plan_cursor = 0
        self.reset_profile_stats()
        self.mcts.reset_profile_stats()
        for opponent_mcts in self.opponent_mcts_by_label.values():
            opponent_mcts.reset_profile_stats()
        total_dropped_positions = 0
        total_truncated_games = 0
        total_claimable_draw_ended_games = 0
        total_completed_length_sum = 0
        total_truncated_length_sum = 0
        total_completed_white_wins = 0
        total_completed_black_wins = 0
        total_completed_draws = 0
        total_decisive_games = 0
        total_decisive_length_sum = 0
        total_curriculum_dropped_positions = 0
        total_cap_dropped_positions = 0
        total_resigned_games = 0
        total_syzygy_ended_games = 0
        total_syzygy_probe_positions = 0
        total_syzygy_probe_hits = 0
        total_search_simulations_used = 0
        total_search_fresh_simulations_used = 0
        total_search_inherited_visit_credit = 0
        total_search_simulations_budget = 0
        total_search_samples = 0
        total_playout_cap_full_search_samples = 0
        total_playout_cap_fast_search_samples = 0
        total_search_difficulty_samples = 0
        total_search_difficulty_sum = 0.0
        total_search_difficulty_sq_sum = 0.0
        total_search_difficulty_budget_cross_sum = 0.0
        total_search_budget_sq_sum = 0.0
        total_full_search_difficulty_sum = 0.0
        total_fast_search_difficulty_sum = 0.0
        total_tree_reuse_attempts = 0
        total_tree_reuse_hits = 0
        total_tree_inherited_visits = 0
        total_tree_reuse_credit_samples = 0
        total_tree_reuse_quality_sum = 0.0
        total_tree_reuse_candidate_coverage_sum = 0.0
        total_tree_reuse_visited_prior_mass_sum = 0.0
        total_tree_reuse_fresh_floor_sum = 0
        total_tree_reuse_scout_stability_sum = 0.0
        total_tree_reuse_scout_extra_credit_sum = 0
        total_tree_reuse_scout_reduced_count = 0
        total_shared_tree_searches = 0
        total_hard_start_games = 0
        search_simulations_used_samples = []
        search_simulations_budget_samples = []
        total_mcts_quality_stats = {
            'mcts_prior_agreement_samples': 0.0,
            'mcts_prior_agreement_sum': 0.0,
            'mcts_prior_changed_count': 0.0,
            'mcts_prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'mcts_policy_kl_sum': 0.0,
            'mcts_top_visit_prob_sum': 0.0,
            'mcts_visit_gap_sum': 0.0,
            'mcts_visit_entropy_sum': 0.0,
            'mcts_good_target_count': 0.0,
            'mcts_explored_prior_mass_sum': 0.0,
            'mcts_visited_move_count_sum': 0.0,
            'mcts_legal_move_count_sum': 0.0,
            'mcts_visit_coverage_ratio_sum': 0.0,
            'mcts_q_delta_samples': 0.0,
            'mcts_q_delta_sum': 0.0,
            'mcts_q_delta_values': [],
            'mcts_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_changed_to_lower_q_count': 0.0,
            'mcts_changed_to_higher_q_count': 0.0,
            'mcts_changed_q_delta_samples': 0.0,
            'mcts_changed_q_delta_sum': 0.0,
            'mcts_changed_q_delta_values': [],
            'mcts_changed_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_policy_uptake_samples': 0.0,
            'mcts_policy_uptake_weight_sum': 0.0,
            'mcts_policy_uptake_low_count': 0.0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            total_mcts_quality_stats[f'mcts_phase_{phase}_samples'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_changed_count'] = 0.0
        opponent_source_counts = {}
        opponent_source_results = {}

        batch_plan_labels = list(self.opponent_plan_labels[self._plan_cursor:self._plan_cursor + num_games])
        self._plan_cursor += num_games
        positions, lengths, batch_stats = self._play_batch(num_games, batch_plan_labels=batch_plan_labels)
        all_positions.extend(positions)
        game_lengths.extend(lengths)
        total_dropped_positions += int(batch_stats.get('dropped_positions', 0))
        total_truncated_games += int(batch_stats.get('truncated_games', 0))
        total_claimable_draw_ended_games += int(batch_stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum += int(batch_stats.get('completed_length_sum', 0))
        total_truncated_length_sum += int(batch_stats.get('truncated_length_sum', 0))
        total_completed_white_wins += int(batch_stats.get('completed_white_wins', 0))
        total_completed_black_wins += int(batch_stats.get('completed_black_wins', 0))
        total_completed_draws += int(batch_stats.get('completed_draws', 0))
        total_decisive_games += int(batch_stats.get('decisive_games', 0))
        total_decisive_length_sum += int(batch_stats.get('decisive_length_sum', 0))
        total_curriculum_dropped_positions += int(batch_stats.get('curriculum_dropped_positions', 0))
        total_cap_dropped_positions += int(batch_stats.get('cap_dropped_positions', 0))
        total_resigned_games += int(batch_stats.get('resigned_games', 0))
        total_syzygy_ended_games += int(batch_stats.get('syzygy_ended_games', 0))
        total_syzygy_probe_positions += int(batch_stats.get('syzygy_probe_positions', 0))
        total_syzygy_probe_hits += int(batch_stats.get('syzygy_probe_hits', 0))
        total_search_simulations_used += int(batch_stats.get('search_simulations_used_sum', 0))
        total_search_fresh_simulations_used += int(
            batch_stats.get('search_fresh_simulations_used_sum', 0)
        )
        total_search_inherited_visit_credit += int(
            batch_stats.get('search_inherited_visit_credit_sum', 0)
        )
        total_search_simulations_budget += int(batch_stats.get('search_simulations_budget_sum', 0))
        total_search_samples += int(batch_stats.get('search_samples', 0))
        total_playout_cap_full_search_samples += int(batch_stats.get('playout_cap_full_search_samples', 0))
        total_playout_cap_fast_search_samples += int(batch_stats.get('playout_cap_fast_search_samples', 0))
        total_search_difficulty_samples += int(batch_stats.get('search_difficulty_samples', 0))
        total_search_difficulty_sum += float(batch_stats.get('search_difficulty_sum', 0.0) or 0.0)
        total_search_difficulty_sq_sum += float(batch_stats.get('search_difficulty_sq_sum', 0.0) or 0.0)
        total_search_difficulty_budget_cross_sum += float(
            batch_stats.get('search_difficulty_budget_cross_sum', 0.0) or 0.0
        )
        total_search_budget_sq_sum += float(batch_stats.get('search_budget_sq_sum', 0.0) or 0.0)
        total_full_search_difficulty_sum += float(
            batch_stats.get('full_search_difficulty_sum', 0.0) or 0.0
        )
        total_fast_search_difficulty_sum += float(
            batch_stats.get('fast_search_difficulty_sum', 0.0) or 0.0
        )
        total_tree_reuse_attempts += int(batch_stats.get('tree_reuse_attempts', 0))
        total_tree_reuse_hits += int(batch_stats.get('tree_reuse_hits', 0))
        total_tree_inherited_visits += int(batch_stats.get('tree_inherited_visits_sum', 0))
        total_tree_reuse_credit_samples += int(
            batch_stats.get('tree_reuse_credit_samples', 0) or 0
        )
        total_tree_reuse_quality_sum += float(batch_stats.get('tree_reuse_quality_sum', 0.0) or 0.0)
        total_tree_reuse_candidate_coverage_sum += float(
            batch_stats.get('tree_reuse_candidate_coverage_sum', 0.0) or 0.0
        )
        total_tree_reuse_visited_prior_mass_sum += float(
            batch_stats.get('tree_reuse_visited_prior_mass_sum', 0.0) or 0.0
        )
        total_tree_reuse_fresh_floor_sum += int(
            batch_stats.get('tree_reuse_fresh_floor_sum', 0) or 0
        )
        total_tree_reuse_scout_stability_sum += float(
            batch_stats.get('tree_reuse_scout_stability_sum', 0.0) or 0.0
        )
        total_tree_reuse_scout_extra_credit_sum += int(
            batch_stats.get('tree_reuse_scout_extra_credit_sum', 0) or 0
        )
        total_tree_reuse_scout_reduced_count += int(
            batch_stats.get('tree_reuse_scout_reduced_count', 0) or 0
        )
        total_shared_tree_searches += int(batch_stats.get('shared_tree_searches', 0))
        total_hard_start_games += int(batch_stats.get('hard_start_games', 0))
        search_simulations_used_samples.extend(list(batch_stats.get('search_simulations_used_samples', []) or []))
        search_simulations_budget_samples.extend(list(batch_stats.get('search_simulations_budget_samples', []) or []))
        for key in total_mcts_quality_stats:
            if key in {'mcts_q_delta_values', 'mcts_changed_q_delta_values'}:
                total_mcts_quality_stats[key].extend(list(batch_stats.get(key, []) or []))
            elif key in {'mcts_q_delta_hist', 'mcts_changed_q_delta_hist'}:
                source_hist = list(batch_stats.get(key, []) or [])
                if len(source_hist) == _Q_DELTA_HIST_BINS:
                    for idx, count in enumerate(source_hist):
                        total_mcts_quality_stats[key][idx] += int(count or 0)
            else:
                total_mcts_quality_stats[key] += float(batch_stats.get(key, 0.0) or 0.0)
        for label, count in dict(batch_stats.get('opponent_source_counts', {}) or {}).items():
            opponent_source_counts[str(label)] = int(opponent_source_counts.get(str(label), 0)) + int(count)
        for label, stats in dict(batch_stats.get('opponent_source_results', {}) or {}).items():
            result_stats = opponent_source_results.setdefault(
                str(label),
                {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
            )
            result_stats['wins'] += int((stats or {}).get('wins', 0))
            result_stats['draws'] += int((stats or {}).get('draws', 0))
            result_stats['losses'] += int((stats or {}).get('losses', 0))
            result_stats['games'] += int((stats or {}).get('games', 0))

        total_games = len(game_lengths)
        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        avg_search_simulations_used = (
            float(total_search_simulations_used) / float(total_search_samples)
            if total_search_samples > 0
            else 0.0
        )
        p10_search_simulations_used = (
            float(np.percentile(np.asarray(search_simulations_used_samples, dtype=np.float32), 10))
            if search_simulations_used_samples
            else 0.0
        )
        budget_arr = (
            np.asarray(search_simulations_budget_samples, dtype=np.float32)
            if search_simulations_budget_samples
            else None
        )
        p10_search_simulations_budget = float(np.percentile(budget_arr, 10)) if budget_arr is not None else 0.0
        p50_search_simulations_budget = float(np.percentile(budget_arr, 50)) if budget_arr is not None else 0.0
        p90_search_simulations_budget = float(np.percentile(budget_arr, 90)) if budget_arr is not None else 0.0
        min_search_simulations_budget = float(np.min(budget_arr)) if budget_arr is not None else 0.0
        max_search_simulations_budget = float(np.max(budget_arr)) if budget_arr is not None else 0.0
        mcts_quality_samples = int(total_mcts_quality_stats['mcts_prior_agreement_samples'])
        mcts_q_delta_samples = int(total_mcts_quality_stats['mcts_q_delta_samples'])
        mcts_q_delta_values = list(total_mcts_quality_stats.get('mcts_q_delta_values', []) or [])
        mcts_q_delta_hist = list(total_mcts_quality_stats.get('mcts_q_delta_hist', []) or [])
        mcts_changed_q_delta_samples = int(total_mcts_quality_stats['mcts_changed_q_delta_samples'])
        mcts_changed_q_delta_values = list(total_mcts_quality_stats.get('mcts_changed_q_delta_values', []) or [])
        mcts_changed_q_delta_hist = list(total_mcts_quality_stats.get('mcts_changed_q_delta_hist', []) or [])
        mcts_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_samples', 0.0) or 0.0)
            changed = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_changed_count', 0.0) or 0.0)
            mcts_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            mcts_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            mcts_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
        self.last_selfplay_stats = {
            'truncated_games': total_truncated_games,
            'completed_games': max(0, total_games - total_truncated_games),
            'dropped_positions': int(total_dropped_positions),
            'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
            'completed_length_sum': int(total_completed_length_sum),
            'truncated_length_sum': int(total_truncated_length_sum),
            'completed_white_wins': int(total_completed_white_wins),
            'completed_black_wins': int(total_completed_black_wins),
            'completed_draws': int(total_completed_draws),
            'decisive_games': int(total_decisive_games),
            'decisive_length_sum': int(total_decisive_length_sum),
            'curriculum_dropped_positions': int(total_curriculum_dropped_positions),
            'cap_dropped_positions': int(total_cap_dropped_positions),
            'resigned_games': int(total_resigned_games),
            'syzygy_ended_games': int(total_syzygy_ended_games),
            'syzygy_probe_positions': int(total_syzygy_probe_positions),
            'syzygy_probe_hits': int(total_syzygy_probe_hits),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
            'search_simulations_used_avg': float(avg_search_simulations_used),
            'search_simulations_budget_avg': (
                float(total_search_simulations_budget) / float(total_search_samples)
                if total_search_samples > 0
                else 0.0
            ),
            'search_simulations_budget_p10': float(p10_search_simulations_budget),
            'search_simulations_budget_p50': float(p50_search_simulations_budget),
            'search_simulations_budget_p90': float(p90_search_simulations_budget),
            'search_simulations_budget_min': float(min_search_simulations_budget),
            'search_simulations_budget_max': float(max_search_simulations_budget),
            'search_simulations_budget_target': float(
                self.num_simulations
            ),
            'search_simulations_used_p10': float(p10_search_simulations_used),
            'search_simulations_used_sum': int(total_search_simulations_used),
            'search_fresh_simulations_used_sum': int(total_search_fresh_simulations_used),
            'search_inherited_visit_credit_sum': int(total_search_inherited_visit_credit),
            'search_simulations_budget_sum': int(total_search_simulations_budget),
            'search_simulations_used_samples': list(search_simulations_used_samples),
            'search_simulations_budget_samples': list(search_simulations_budget_samples),
            'search_samples': int(total_search_samples),
            'playout_cap_full_search_samples': int(total_playout_cap_full_search_samples),
            'playout_cap_fast_search_samples': int(total_playout_cap_fast_search_samples),
            'search_difficulty_samples': int(total_search_difficulty_samples),
            'search_difficulty_sum': float(total_search_difficulty_sum),
            'search_difficulty_sq_sum': float(total_search_difficulty_sq_sum),
            'search_difficulty_budget_cross_sum': float(total_search_difficulty_budget_cross_sum),
            'search_budget_sq_sum': float(total_search_budget_sq_sum),
            'full_search_difficulty_sum': float(total_full_search_difficulty_sum),
            'fast_search_difficulty_sum': float(total_fast_search_difficulty_sum),
            'tree_reuse_attempts': int(total_tree_reuse_attempts),
            'tree_reuse_hits': int(total_tree_reuse_hits),
            'tree_inherited_visits_sum': int(total_tree_inherited_visits),
            'tree_reuse_credit_samples': int(total_tree_reuse_credit_samples),
            'tree_reuse_quality_sum': float(total_tree_reuse_quality_sum),
            'tree_reuse_candidate_coverage_sum': float(total_tree_reuse_candidate_coverage_sum),
            'tree_reuse_visited_prior_mass_sum': float(total_tree_reuse_visited_prior_mass_sum),
            'tree_reuse_fresh_floor_sum': int(total_tree_reuse_fresh_floor_sum),
            'tree_reuse_scout_stability_sum': float(total_tree_reuse_scout_stability_sum),
            'tree_reuse_scout_extra_credit_sum': int(total_tree_reuse_scout_extra_credit_sum),
            'tree_reuse_scout_reduced_count': int(total_tree_reuse_scout_reduced_count),
            'shared_tree_searches': int(total_shared_tree_searches),
            'hard_start_games': int(total_hard_start_games),
            'mcts_prior_agreement_samples': int(mcts_quality_samples),
            'mcts_prior_agreement_sum': float(total_mcts_quality_stats['mcts_prior_agreement_sum']),
            'mcts_prior_agreement_rate': (
                float(total_mcts_quality_stats['mcts_prior_agreement_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_count': int(total_mcts_quality_stats['mcts_prior_changed_count']),
            'mcts_prior_changed_rate': (
                float(total_mcts_quality_stats['mcts_prior_changed_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            **mcts_phase_stats,
            'mcts_policy_uptake_samples': int(total_mcts_quality_stats['mcts_policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(total_mcts_quality_stats['mcts_policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(total_mcts_quality_stats['mcts_policy_uptake_low_count'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(total_mcts_quality_stats['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_prior_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(total_mcts_quality_stats['mcts_policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(total_mcts_quality_stats['mcts_policy_kl_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(total_mcts_quality_stats['mcts_visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(total_mcts_quality_stats['mcts_visit_gap_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(total_mcts_quality_stats['mcts_visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(total_mcts_quality_stats['mcts_visit_entropy_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(total_mcts_quality_stats['mcts_good_target_count']),
            'mcts_good_target_rate': (
                float(total_mcts_quality_stats['mcts_good_target_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(total_mcts_quality_stats['mcts_visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(total_mcts_quality_stats['mcts_visited_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(total_mcts_quality_stats['mcts_legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(total_mcts_quality_stats['mcts_legal_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': int(mcts_q_delta_samples),
            'mcts_q_delta_sum': float(total_mcts_quality_stats['mcts_q_delta_sum']),
            'mcts_q_delta_hist': mcts_q_delta_hist,
            'mcts_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_q_delta_sum']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 10)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 10))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 50)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 50))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 90)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 90))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(total_mcts_quality_stats['mcts_changed_to_lower_q_count']),
            'mcts_changed_to_lower_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(total_mcts_quality_stats['mcts_changed_to_higher_q_count']),
            'mcts_changed_q_delta_samples': int(mcts_changed_q_delta_samples),
            'mcts_changed_q_delta_sum': float(total_mcts_quality_stats['mcts_changed_q_delta_sum']),
            'mcts_changed_q_delta_hist': mcts_changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_changed_q_delta_sum']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 10)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 10))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 50)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 50))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 90)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 90))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_higher_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'total_games': int(total_games),
            'profile': self._aggregate_engine_profile_stats(),
        }
        return all_positions, game_lengths

    def _report_progress(self):
        progress = self._progress_base + self._games_completed
        if self._progress_file is None:
            return
        try:
            with open(self._progress_file, 'w') as _pf:
                _pf.write(str(progress))
        except Exception:
            pass

    def _mark_game_completed(self, gs):
        if gs.get('_completion_reported', False):
            return
        gs['_completion_reported'] = True
        self._games_completed += 1
        if (
            self._games_completed == 1
            or (self._games_completed % self.progress_report_interval_games) == 0
        ):
            self._report_progress()

    def _play_batch(self, batch_size, batch_plan_labels=None):
        max_moves = self.max_moves

        total_games_to_play = max(0, int(batch_size))
        active_limit = min(self.max_batch_games_per_worker, total_games_to_play)
        games_started = 0
        game_states = []
        completed_game_states = []

        def _new_game_state(game_idx):
            game_opponent_label = (
                str(batch_plan_labels[game_idx])
                if batch_plan_labels is not None and game_idx < len(batch_plan_labels)
                else self.opponent_source_label
            )
            opponent_mcts = self.opponent_mcts_by_label.get(game_opponent_label)
            if game_opponent_label != "current" and opponent_mcts is None:
                if game_opponent_label not in self._warned_missing_opponent_labels:
                    print(
                        "Warning: opponent plan references "
                        f"'{game_opponent_label}', but this worker has no model for it. "
                        "Falling back to current."
                    )
                    self._warned_missing_opponent_labels.add(game_opponent_label)
                game_opponent_label = "current"
                opponent_mcts = None
            has_frozen_opponent = opponent_mcts is not None
            hard_start = None
            if game_idx < len(self.hard_start_positions):
                candidate = self.hard_start_positions[game_idx]
                if isinstance(candidate, dict) and candidate.get('fen'):
                    hard_start = candidate
            initial_board = chess.new_board(hard_start.get('fen')) if hard_start else chess.new_board()
            hard_history_fens = list((hard_start or {}).get('history_fens', []) or [])[-self.history_positions:]
            encoded_hard_history = []
            for history_fen in hard_history_fens:
                try:
                    encoded_hard_history.append(self.mcts._encode_history_entry(chess.new_board(history_fen)))
                except Exception:
                    continue
            fen_parts = chess.board_fen(initial_board).split()
            try:
                fullmove = max(1, int(fen_parts[5]))
            except (IndexError, TypeError, ValueError):
                fullmove = 1
            initial_ply = (fullmove - 1) * 2 + (0 if initial_board.turn == chess.WHITE else 1)
            gs = {
                'board': initial_board,
                'board_history': encoded_hard_history,
                'fen_history': hard_history_fens,
                'position_counts': {_board_position_key(initial_board): 1},
                'root': None,
                '_root_synced': False,
                'opponent_root': None,
                '_opponent_root_synced': False,
                'game_history': [],
                'move_count': initial_ply,
                'done': False,
                'white_advantage_streak': 0,
                'black_advantage_streak': 0,
                'adjudicated_result': None,
                'syzygy_result': None,
                'syzygy_probe_positions': 0,
                'syzygy_probe_hits': 0,
                'resigned_result': None,
                'resign_streak': 0,
                'resignation_disabled': bool(np.random.random() < self.resignation_disable_fraction),
                'opponent_source_label': game_opponent_label,
                'opponent_mcts': opponent_mcts,
                'has_frozen_opponent': has_frozen_opponent,
                'learner_color': (
                    chess.WHITE
                    if not has_frozen_opponent or not self.randomize_learner_color
                    else (chess.WHITE if np.random.random() < 0.5 else chess.BLACK)
                ),
                '_completion_reported': False,
                '_finalized_for_batch': False,
                '_hard_start': bool(hard_start),
            }
            if not hard_start:
                self._apply_opening_prefix(gs)
            return gs

        def _fill_active_slots():
            nonlocal games_started
            while games_started < total_games_to_play and len(game_states) < active_limit:
                game_states.append(_new_game_state(games_started))
                games_started += 1

        def _retire_completed_games():
            if not game_states:
                return
            active = []
            for gs in game_states:
                if gs.get('done', False):
                    if not gs.get('_finalized_for_batch', False):
                        gs['_finalized_for_batch'] = True
                        completed_game_states.append(gs)
                else:
                    active.append(gs)
            game_states[:] = active

        _fill_active_slots()

        # Initialize search statistics used across this batch
        search_stats = {
            'sim_used': 0,
            'fresh_sim_used': 0,
            'inherited_visit_credit': 0,
            'sim_budget': 0,
            'samples': 0,
            'samples_list': [],
            'budget_samples_list': [],
            'full_search_samples': 0,
            'fast_search_samples': 0,
            'difficulty_samples': 0,
            'difficulty_sum': 0.0,
            'difficulty_sq_sum': 0.0,
            'difficulty_budget_cross_sum': 0.0,
            'budget_sq_sum': 0.0,
            'full_search_difficulty_sum': 0.0,
            'fast_search_difficulty_sum': 0.0,
            'tree_reuse_attempts': 0,
            'tree_reuse_hits': 0,
            'tree_inherited_visits_sum': 0,
            'tree_reuse_credit_samples': 0,
            'tree_reuse_quality_sum': 0.0,
            'tree_reuse_candidate_coverage_sum': 0.0,
            'tree_reuse_visited_prior_mass_sum': 0.0,
            'tree_reuse_fresh_floor_sum': 0,
            'tree_reuse_scout_stability_sum': 0.0,
            'tree_reuse_scout_extra_credit_sum': 0,
            'tree_reuse_scout_reduced_count': 0,
            'shared_tree_searches': 0,
        }
        target_quality = {
            'samples': 0,
            'changed': 0,
            'agreement_sum': 0.0,
            'prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'policy_kl_sum': 0.0,
            'top_visit_prob_sum': 0.0,
            'visit_gap_sum': 0.0,
            'visit_entropy_sum': 0.0,
            'good_target_count': 0,
            'explored_prior_mass_sum': 0.0,
            'visited_move_count_sum': 0.0,
            'legal_move_count_sum': 0.0,
            'visit_coverage_ratio_sum': 0.0,
            'q_comparable': 0,
            'q_delta_sum': 0.0,
            'q_delta_values': [],
            'changed_to_lower_q': 0,
            'changed_to_higher_q': 0,
            'changed_q_comparable': 0,
            'changed_q_delta_sum': 0.0,
            'changed_q_delta_values': [],
            'policy_uptake_samples': 0,
            'policy_uptake_weight_sum': 0.0,
            'policy_uptake_low_count': 0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            target_quality[f'{phase}_samples'] = 0
            target_quality[f'{phase}_changed_count'] = 0

        def _accumulate_target_quality(search_metadata, board):
            if not isinstance(search_metadata, dict):
                return
            agree = search_metadata.get('prior_mcts_agree', None)
            if agree is None:
                return
            target_quality['samples'] += 1
            agree_value = float(agree)
            changed_top = agree_value < 0.5
            phase = self._mcts_phase_for_board(board)
            target_quality[f'{phase}_samples'] += 1
            if changed_top:
                target_quality[f'{phase}_changed_count'] += 1
            target_quality['agreement_sum'] += agree_value
            if changed_top:
                target_quality['changed'] += 1
            for meta_key, sum_key in [
                ('prior_top_visit_prob', 'prior_top_visit_prob_sum'),
                ('mcts_top_prior_prob', 'mcts_top_prior_prob_sum'),
                ('mcts_policy_kl', 'policy_kl_sum'),
                ('top_visit_prob', 'top_visit_prob_sum'),
                ('visit_gap', 'visit_gap_sum'),
                ('visit_entropy', 'visit_entropy_sum'),
                ('explored_prior_mass', 'explored_prior_mass_sum'),
            ]:
                value = search_metadata.get(meta_key, None)
                if value is not None:
                    target_quality[sum_key] += float(value)
            try:
                top_visit_prob = float(search_metadata.get('top_visit_prob', 0.0) or 0.0)
                visit_gap = float(search_metadata.get('visit_gap', 0.0) or 0.0)
            except (TypeError, ValueError):
                top_visit_prob = 0.0
                visit_gap = 0.0
            if (
                top_visit_prob >= float(self.mcts_good_target_min_top_visit_prob)
                and visit_gap >= float(self.mcts_good_target_min_visit_gap)
            ):
                target_quality['good_target_count'] += 1
            visited_count = search_metadata.get('visited_move_count', None)
            legal_count = search_metadata.get('legal_move_count', None)
            if visited_count is not None and legal_count is not None:
                visited_count = float(visited_count)
                legal_count = float(legal_count)
                target_quality['visited_move_count_sum'] += float(visited_count)
                target_quality['legal_move_count_sum'] += float(legal_count)
                if legal_count > 0:
                    target_quality['visit_coverage_ratio_sum'] += float(visited_count) / float(legal_count)
            q_delta = search_metadata.get('mcts_q_delta', None)
            if q_delta is not None:
                target_quality['q_comparable'] += 1
                q_delta_value = float(q_delta)
                target_quality['q_delta_sum'] += q_delta_value
                target_quality['q_delta_values'].append(q_delta_value)
                if changed_top:
                    target_quality['changed_q_comparable'] += 1
                    target_quality['changed_q_delta_sum'] += q_delta_value
                    target_quality['changed_q_delta_values'].append(q_delta_value)
                    if q_delta_value < -0.02:
                        target_quality['changed_to_lower_q'] += 1
                    elif q_delta_value > 0.02:
                        target_quality['changed_to_higher_q'] += 1

        while len(completed_game_states) < total_games_to_play:
            active_indices = []
            learner_indices = []
            grouped_opponent_indices = {}
            for i, gs in enumerate(game_states):
                if gs['done']:
                    continue
                board = gs['board']
                syzygy_t0 = time.perf_counter() if self.profile_enabled else None
                self._maybe_finish_with_syzygy(gs, board)
                if self.profile_enabled:
                    self._profile_add('syzygy_time', time.perf_counter() - syzygy_t0)
                    self._profile_inc('syzygy_calls', 1)
                if gs['done']:
                    continue

                active_indices.append(i)
                if not gs.get('has_frozen_opponent', False):
                    gs['_learner_turn_cache'] = True
                    learner_indices.append(i)
                    continue

                learner_turn = bool(board.turn == gs.get('learner_color', chess.WHITE))
                gs['_learner_turn_cache'] = learner_turn
                if learner_turn:
                    learner_indices.append(i)
                else:
                    label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
                    grouped_opponent_indices.setdefault(label, []).append(i)

            if not active_indices:
                _retire_completed_games()
                _fill_active_slots()
                if not game_states:
                    break
                continue

            visit_counts_by_index = {}

            def _run_search_for_indices(indices, mcts_ref, root_key, synced_key):
                if not indices:
                    return
                group_states = []
                for i in indices:
                    gs = game_states[i]
                    playout_cap_eligible = bool(self.playout_cap_randomization_enabled)
                    force_full_search = bool(gs.get(
                        '_hard_start_first_search',
                        bool(gs.get('_hard_start', False)),
                    ))
                    gs['_playout_cap_full_search'] = True
                    group_states.append([
                        gs['board'],
                        gs.get(root_key),
                        bool(gs.get(synced_key, False)),
                        gs['board_history'],
                        int(gs.get('move_count', 0) or 0),
                        gs.get('position_counts'),
                        None,
                        playout_cap_eligible,
                        force_full_search,
                    ])
                visit_counts_list_group, search_metadata_group = mcts_ref.search_many(
                    group_states,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                    return_search_metadata=True,
                )
                for gs_idx, local_state, visit_counts, search_metadata in zip(
                    indices,
                    group_states,
                    visit_counts_list_group,
                    search_metadata_group,
                ):
                    gs = game_states[gs_idx]
                    gs[root_key] = local_state[1]
                    gs[synced_key] = bool(local_state[2])
                    if isinstance(search_metadata, dict):
                        gs['_playout_cap_full_search'] = bool(
                            search_metadata.get('playout_cap_full_search', True)
                        )
                    visit_counts_by_index[gs_idx] = (visit_counts, search_metadata)
                    used = int(search_metadata.get('simulations_used', 0)) if isinstance(search_metadata, dict) else 0
                    search_stats['sim_used'] += used
                    if isinstance(search_metadata, dict):
                        search_stats['fresh_sim_used'] += int(
                            search_metadata.get('fresh_simulations_used', used) or 0
                        )
                        search_stats['inherited_visit_credit'] += int(
                            search_metadata.get('inherited_visit_credit', 0) or 0
                        )
                        budget = int(search_metadata.get('simulation_budget', self.num_simulations) or self.num_simulations)
                        search_stats['sim_budget'] += budget
                        search_stats['budget_samples_list'].append(budget)
                    else:
                        search_stats['fresh_sim_used'] += used
                        search_stats['sim_budget'] += int(self.num_simulations)
                        search_stats['budget_samples_list'].append(int(self.num_simulations))
                    search_stats['samples'] += 1
                    search_stats['samples_list'].append(used)
                    if bool(gs.get('_playout_cap_full_search', True)):
                        search_stats['full_search_samples'] += 1
                    else:
                        search_stats['fast_search_samples'] += 1
                    if isinstance(search_metadata, dict):
                        reuse_attempted = bool(search_metadata.get('tree_reuse_attempted', False))
                        tree_reused = bool(search_metadata.get('tree_reused', False))
                        search_stats['tree_reuse_attempts'] += int(reuse_attempted)
                        search_stats['tree_reuse_hits'] += int(tree_reused)
                        search_stats['tree_inherited_visits_sum'] += int(
                            search_metadata.get('tree_inherited_visits', 0) or 0
                        )
                        if tree_reused and bool(gs.get('_playout_cap_full_search', True)):
                            search_stats['tree_reuse_credit_samples'] += 1
                            search_stats['tree_reuse_quality_sum'] += float(
                                search_metadata.get('tree_reuse_quality', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_candidate_coverage_sum'] += float(
                                search_metadata.get('tree_reuse_candidate_coverage', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_visited_prior_mass_sum'] += float(
                                search_metadata.get('tree_reuse_visited_prior_mass', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_fresh_floor_sum'] += int(
                                search_metadata.get('tree_reuse_fresh_floor', budget) or budget
                            )
                            search_stats['tree_reuse_scout_stability_sum'] += float(
                                search_metadata.get('tree_reuse_scout_stability', 0.0) or 0.0
                            )
                            search_stats['tree_reuse_scout_extra_credit_sum'] += int(
                                search_metadata.get('tree_reuse_scout_extra_credit', 0) or 0
                            )
                            search_stats['tree_reuse_scout_reduced_count'] += int(bool(
                                search_metadata.get('tree_reuse_scout_search_reduced', False)
                            ))
                        search_stats['shared_tree_searches'] += int(
                            self.share_trees and not bool(gs.get('has_frozen_opponent', False))
                        )
                        difficulty = float(search_metadata.get('dynamic_budget_difficulty', 0.0) or 0.0)
                        if math.isfinite(difficulty):
                            search_stats['difficulty_samples'] += 1
                            search_stats['difficulty_sum'] += difficulty
                            search_stats['difficulty_sq_sum'] += difficulty * difficulty
                            search_stats['difficulty_budget_cross_sum'] += difficulty * float(budget)
                            search_stats['budget_sq_sum'] += float(budget) * float(budget)
                            if bool(gs.get('_playout_cap_full_search', True)):
                                search_stats['full_search_difficulty_sum'] += difficulty
                            else:
                                search_stats['fast_search_difficulty_sum'] += difficulty

            if not self.opponent_mcts_by_label:
                _run_search_for_indices(active_indices, self.mcts, 'root', '_root_synced')
            else:
                _run_search_for_indices(learner_indices, self.mcts, 'root', '_root_synced')
                for label, indices in grouped_opponent_indices.items():
                    opponent_mcts = self.opponent_mcts_by_label.get(label)
                    if opponent_mcts is None:
                        continue
                    _run_search_for_indices(indices, opponent_mcts, 'opponent_root', '_opponent_root_synced')
            for idx in active_indices:
                game_states[idx]['_hard_start_first_search'] = False

            for idx in active_indices:
                gs = game_states[idx]
                board = gs['board']
                visit_payload = visit_counts_by_index.get(idx, None)
                if visit_payload is None:
                    continue
                visit_counts, search_metadata = visit_payload

                if self.temp_threshold > 0 and gs['move_count'] < self.temp_threshold:
                    temperature = self.temperature
                else:
                    temperature = 0.0
                learner_turn = bool(gs.get('_learner_turn_cache', True))
                game_opponent_mcts = gs.get('opponent_mcts')
                root_key = 'root' if learner_turn or game_opponent_mcts is None else 'opponent_root'
                synced_key = '_root_synced' if learner_turn or game_opponent_mcts is None else '_opponent_root_synced'
                root = gs.get(root_key)
                adjudication_t0 = time.perf_counter() if self.profile_enabled else None
                adjudicated_result = self._maybe_adjudicate_game(gs, root, board)
                if self.profile_enabled:
                    self._profile_add('adjudication_time', time.perf_counter() - adjudication_t0)
                    self._profile_inc('adjudication_calls', 1)
                if adjudicated_result is not None:
                    gs['adjudicated_result'] = adjudicated_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                resigned_result = self._maybe_resign_game(gs, root, board)
                if resigned_result is not None:
                    gs['resigned_result'] = resigned_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                move_t0 = time.perf_counter() if self.profile_enabled else None
                selected_move_override = (
                    search_metadata.get('selected_move_override')
                    if isinstance(search_metadata, dict)
                    else None
                )
                if selected_move_override in visit_counts:
                    # Gumbel noise already supplies self-play exploration; the
                    # Sequential-Halving winner is the action prescribed by the
                    # algorithm, independent of the legacy visit temperature.
                    move = selected_move_override
                else:
                    move = self._select_move_from_visits(visit_counts, temperature)
                if (
                    isinstance(search_metadata, dict)
                    and root is not None
                    and root.expanded
                    and root.edges is not None
                ):
                    played_edge_idx = root.edges._get_move_index(move)
                    if played_edge_idx is not None:
                        played_edge_visits = int(root.edges.visit_counts[int(played_edge_idx)])
                        if played_edge_visits > 0:
                            search_metadata['played_q'] = float(max(
                                -1.0,
                                min(
                                    1.0,
                                    -float(root.edges.value_sums[int(played_edge_idx)])
                                    / float(played_edge_visits),
                                ),
                            ))
                if self.profile_enabled:
                    self._profile_add('move_selection_time', time.perf_counter() - move_t0)
                    self._profile_inc('move_selection_calls', 1)
                store_policy_position = self._should_store_policy_position(
                    learner_turn,
                    game_opponent_mcts,
                    gs.get('opponent_source_label', self.opponent_source_label),
                )
                if store_policy_position:
                    _accumulate_target_quality(search_metadata, board)
                    policy_t0 = time.perf_counter() if self.profile_enabled else None
                    target_visit_counts = visit_counts
                    policy_is_probability_target = False
                    if isinstance(search_metadata, dict):
                        probability_target = search_metadata.get('policy_target_probs_override')
                        if isinstance(probability_target, dict) and probability_target:
                            target_visit_counts = probability_target
                            policy_is_probability_target = True
                        else:
                            override_visit_counts = search_metadata.get('policy_visit_counts_override')
                            if isinstance(override_visit_counts, dict) and override_visit_counts:
                                target_visit_counts = override_visit_counts
                    policy_visit_counts = (
                        self._prune_policy_target_weights(target_visit_counts)
                        if policy_is_probability_target
                        else self._prune_policy_target_visits(target_visit_counts)
                    )
                    policy_indices, policy_values = _build_sparse_policy_target_from_visits(policy_visit_counts, board)
                    if root is not None:
                        # The root was expanded by this search, so its cached
                        # legal indices are exactly the current board's legal
                        # mask.  Reusing them avoids regenerating legal moves
                        # and remapping every move for replay storage.
                        _, cached_legal_indices = root.get_legal_moves_and_indices()
                        legal_indices_full = torch.from_numpy(
                            cached_legal_indices.astype(np.int16, copy=False)
                        )
                    else:
                        legal_indices_full = torch.tensor(
                            [move_to_index(legal_move, board) for legal_move in chess.legal_moves(board)],
                            dtype=torch.int16,
                        )
                    history_count = len(gs['board_history'])
                    importance_score = self._compute_position_importance(
                        board,
                        move,
                        policy_visit_counts,
                        root,
                        search_metadata=search_metadata,
                    )
                    policy_weight = _policy_uptake_weight(
                        search_metadata,
                        good_target_min_top_visit_prob=self.mcts_good_target_min_top_visit_prob,
                        good_target_min_visit_gap=self.mcts_good_target_min_visit_gap,
                    )
                    full_search = bool(gs.get('_playout_cap_full_search', True))
                    search_changed_top, search_q_delta, correction_multiplier = (
                        _search_correction_metadata(
                            search_metadata,
                            full_search=full_search,
                        )
                    )
                    if not full_search:
                        # True PCR: cheap searches contribute game outcomes but
                        # never teach the policy from low-visit targets.
                        policy_weight = 0.0
                    else:
                        policy_weight *= correction_multiplier
                    policy_uptake_weight = policy_weight
                    target_quality['policy_uptake_samples'] += 1
                    target_quality['policy_uptake_weight_sum'] += policy_uptake_weight
                    if policy_uptake_weight < _POLICY_UPTAKE_LOW_THRESHOLD:
                        target_quality['policy_uptake_low_count'] += 1
                    replay_source_code = _replay_source_code(
                        learner_turn,
                        game_opponent_mcts,
                        gs.get('opponent_source_label', self.opponent_source_label),
                    )
                    gs['game_history'].append((
                        history_count,
                        policy_indices,
                        policy_values,
                        board.turn,
                        importance_score,
                        policy_weight,
                        replay_source_code,
                        legal_indices_full,
                        chess.board_fen(board),
                        (
                            float(search_metadata.get('root_value'))
                            if full_search and isinstance(search_metadata, dict)
                            else float('nan')
                        ),
                        tuple(gs.get('fen_history', [])[-self.history_positions:]),
                        search_changed_top,
                        search_q_delta,
                        float(search_metadata.get('best_q'))
                        if full_search and search_metadata.get('best_q') is not None else float('nan'),
                        float(search_metadata.get('played_q'))
                        if full_search and search_metadata.get('played_q') is not None else float('nan'),
                        float(search_metadata.get('orig_q'))
                        if search_metadata.get('orig_q') is not None else float('nan'),
                        float(search_metadata.get('policy_kld'))
                        if full_search and search_metadata.get('policy_kld') is not None else float('nan'),
                        int(search_metadata.get('search_visits', 0) or 0) if full_search else 0,
                    ))
                    if self.profile_enabled:
                        self._profile_add('policy_target_build_time', time.perf_counter() - policy_t0)
                        self._profile_inc('policy_target_build_calls', 1)

                # Update history BEFORE making the move
                # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
                gs['board_history'].append(self.mcts._encode_history_entry(board))
                gs.setdefault('fen_history', []).append(chess.board_fen(board))

                # Keep both side-specific MCTS trees synchronized with the
                # actual game line. Mixed-opponent self-play otherwise starts
                # every move from a fresh root and produces noisier targets.
                if self.share_trees or gs['has_frozen_opponent']:
                    root_keys = [('root', '_root_synced')]
                    if gs['has_frozen_opponent']:
                        # Different networks must retain independent trees: their
                        # priors and Q values are not interchangeable.
                        root_keys.append(('opponent_root', '_opponent_root_synced'))
                    for candidate_root_key, candidate_synced_key in root_keys:
                        next_root, next_synced = self._advance_search_root(
                            gs.get(candidate_root_key),
                            move,
                        )
                        gs[candidate_root_key] = next_root
                        gs[candidate_synced_key] = next_synced
                else:
                    self._clear_game_search_state(gs)

                chess.apply_move(board, move)
                gs['move_count'] += 1
                _record_position_count(gs['position_counts'], board)

                forced_game_over = chess.is_game_over(board, claim_draw=False)
                auto_claim_draw = self._should_auto_claim_draw(board, gs['move_count'])
                if forced_game_over or auto_claim_draw or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    gs['ended_by_auto_claim_draw'] = auto_claim_draw
                    # Free up memory immediately
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
            _retire_completed_games()
            _fill_active_slots()

        positions = []
        game_lengths = []
        dropped_positions = 0
        truncated_games = 0
        claimable_draw_ended_games = 0
        adjudicated_games = 0
        syzygy_ended_games = 0
        syzygy_probe_positions = 0
        syzygy_probe_hits = 0
        resigned_games = 0
        completed_length_sum = 0
        truncated_length_sum = 0
        completed_white_wins = 0
        completed_black_wins = 0
        completed_draws = 0
        learner_wins = 0
        learner_draws = 0
        learner_losses = 0
        opponent_source_counts = {}
        opponent_source_results = {}
        decisive_games = 0
        decisive_length_sum = 0
        curriculum_dropped_positions = 0
        cap_dropped_positions = 0
        history_positions = int(self.config.get('model', {}).get('history_positions', 0))

        for gs in completed_game_states:
            board = gs['board']
            opponent_label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
            opponent_source_counts[opponent_label] = int(opponent_source_counts.get(opponent_label, 0)) + 1
            ended_by_claimable_draw = (
                bool(gs.get('ended_by_auto_claim_draw', False))
            )
            if ended_by_claimable_draw:
                claimable_draw_ended_games += 1
            adjudicated_result = gs.get('adjudicated_result', None)
            if adjudicated_result is not None:
                adjudicated_games += 1
            syzygy_result = gs.get('syzygy_result', None)
            if syzygy_result is not None:
                syzygy_ended_games += 1
            syzygy_probe_positions += int(gs.get('syzygy_probe_positions', 0) or 0)
            syzygy_probe_hits += int(gs.get('syzygy_probe_hits', 0) or 0)
            resigned_result = gs.get('resigned_result', None)
            if resigned_result is not None:
                resigned_games += 1
            result = (
                resigned_result
                if resigned_result is not None
                else (
                    adjudicated_result
                    if adjudicated_result is not None
                    else (
                        syzygy_result
                        if syzygy_result is not None
                        else ('1/2-1/2' if ended_by_claimable_draw else chess.result(board, claim_draw=False))
                    )
                )
            )
            game_ply_len = int(gs.get('move_count', len(gs['game_history'])))
            if result == '*':
                truncated_games += 1
                stored_history_len = len(gs['game_history'])
                dropped_positions += stored_history_len
                truncated_length_sum += game_ply_len
                game_lengths.append(game_ply_len)
                continue
            if result == '1-0':
                outcome = 1.0
                completed_white_wins += 1
                decisive_games += 1
            elif result == '0-1':
                outcome = -1.0
                completed_black_wins += 1
                decisive_games += 1
            else:
                outcome = 0.0
                completed_draws += 1

            if opponent_label in self.opponent_mcts_by_label:
                learner_color = gs.get('learner_color', chess.WHITE)
                learner_outcome = outcome if learner_color == chess.WHITE else -outcome
                result_stats = opponent_source_results.setdefault(
                    opponent_label,
                    {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
                )
                if learner_outcome > 0.0:
                    learner_wins += 1
                    result_stats['wins'] += 1
                elif learner_outcome < 0.0:
                    learner_losses += 1
                    result_stats['losses'] += 1
                else:
                    learner_draws += 1
                    result_stats['draws'] += 1
                result_stats['games'] += 1
            completed_length_sum += game_ply_len
            if outcome != 0.0:
                decisive_length_sum += game_ply_len
            _, game_curriculum_dropped, game_cap_dropped = self._append_selected_positions_from_game(
                positions,
                gs,
                outcome,
                history_positions=history_positions,
            )
            curriculum_dropped_positions += game_curriculum_dropped
            cap_dropped_positions += game_cap_dropped

            game_lengths.append(game_ply_len)

        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        target_quality_samples = int(target_quality['samples'])
        target_quality_q_samples = int(target_quality['q_comparable'])
        target_quality_changed_q_samples = int(target_quality['changed_q_comparable'])
        q_delta_values = list(target_quality.get('q_delta_values', []) or [])
        q_delta_hist = _q_delta_histogram(q_delta_values)
        changed_q_delta_values = list(target_quality.get('changed_q_delta_values', []) or [])
        changed_q_delta_hist = _q_delta_histogram(changed_q_delta_values)
        target_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(target_quality.get(f'{phase}_samples', 0) or 0)
            changed = float(target_quality.get(f'{phase}_changed_count', 0) or 0)
            target_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            target_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            target_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
        return positions, game_lengths, {
            'total_games': int(len(completed_game_states)),
            'truncated_games': int(truncated_games),
            'dropped_positions': int(dropped_positions),
            'claimable_draw_ended_games': int(claimable_draw_ended_games),
            'adjudicated_games': int(adjudicated_games),
            'syzygy_ended_games': int(syzygy_ended_games),
            'syzygy_probe_positions': int(syzygy_probe_positions),
            'syzygy_probe_hits': int(syzygy_probe_hits),
            'search_simulations_used_sum': int(search_stats['sim_used']),
            'search_fresh_simulations_used_sum': int(search_stats['fresh_sim_used']),
            'search_inherited_visit_credit_sum': int(search_stats['inherited_visit_credit']),
            'search_simulations_budget_sum': int(search_stats['sim_budget']),
            'search_samples': int(search_stats['samples']),
            'search_simulations_used_samples': list(search_stats['samples_list']),
            'search_simulations_budget_samples': list(search_stats['budget_samples_list']),
            'playout_cap_full_search_samples': int(search_stats['full_search_samples']),
            'playout_cap_fast_search_samples': int(search_stats['fast_search_samples']),
            'search_difficulty_samples': int(search_stats['difficulty_samples']),
            'search_difficulty_sum': float(search_stats['difficulty_sum']),
            'search_difficulty_sq_sum': float(search_stats['difficulty_sq_sum']),
            'search_difficulty_budget_cross_sum': float(search_stats['difficulty_budget_cross_sum']),
            'search_budget_sq_sum': float(search_stats['budget_sq_sum']),
            'full_search_difficulty_sum': float(search_stats['full_search_difficulty_sum']),
            'fast_search_difficulty_sum': float(search_stats['fast_search_difficulty_sum']),
            'tree_reuse_attempts': int(search_stats['tree_reuse_attempts']),
            'tree_reuse_hits': int(search_stats['tree_reuse_hits']),
            'tree_inherited_visits_sum': int(search_stats['tree_inherited_visits_sum']),
            'tree_reuse_credit_samples': int(search_stats['tree_reuse_credit_samples']),
            'tree_reuse_quality_sum': float(search_stats['tree_reuse_quality_sum']),
            'tree_reuse_candidate_coverage_sum': float(
                search_stats['tree_reuse_candidate_coverage_sum']
            ),
            'tree_reuse_visited_prior_mass_sum': float(
                search_stats['tree_reuse_visited_prior_mass_sum']
            ),
            'tree_reuse_fresh_floor_sum': int(search_stats['tree_reuse_fresh_floor_sum']),
            'tree_reuse_scout_stability_sum': float(
                search_stats['tree_reuse_scout_stability_sum']
            ),
            'tree_reuse_scout_extra_credit_sum': int(
                search_stats['tree_reuse_scout_extra_credit_sum']
            ),
            'tree_reuse_scout_reduced_count': int(
                search_stats['tree_reuse_scout_reduced_count']
            ),
            'shared_tree_searches': int(search_stats['shared_tree_searches']),
            'hard_start_games': int(sum(bool(gs.get('_hard_start', False)) for gs in completed_game_states)),
            'mcts_prior_agreement_samples': target_quality_samples,
            'mcts_prior_agreement_sum': float(target_quality['agreement_sum']),
            'mcts_prior_changed_count': int(target_quality['changed']),
            'mcts_prior_agreement_rate': (
                float(target_quality['agreement_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_rate': (
                float(target_quality['changed']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            **target_phase_stats,
            'mcts_policy_uptake_samples': int(target_quality['policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(target_quality['policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(target_quality['policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(target_quality['policy_uptake_weight_sum']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(target_quality['policy_uptake_low_count']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(target_quality['prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(target_quality['prior_top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(target_quality['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(target_quality['mcts_top_prior_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(target_quality['policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(target_quality['policy_kl_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(target_quality['top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(target_quality['top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(target_quality['visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(target_quality['visit_gap_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(target_quality['visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(target_quality['visit_entropy_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(target_quality['good_target_count']),
            'mcts_good_target_rate': (
                float(target_quality['good_target_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(target_quality['explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(target_quality['explored_prior_mass_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(target_quality['visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(target_quality['visited_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(target_quality['legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(target_quality['legal_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(target_quality['visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(target_quality['visit_coverage_ratio_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': target_quality_q_samples,
            'mcts_q_delta_sum': float(target_quality['q_delta_sum']),
            'mcts_q_delta_values': q_delta_values,
            'mcts_q_delta_hist': q_delta_hist,
            'mcts_q_delta_mean': (
                float(target_quality['q_delta_sum']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(q_delta_hist, 10)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(q_delta_hist, 50)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(q_delta_hist, 90)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(target_quality['changed_to_lower_q']),
            'mcts_changed_to_lower_q_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(target_quality['changed_to_higher_q']),
            'mcts_changed_q_delta_samples': int(target_quality_changed_q_samples),
            'mcts_changed_q_delta_sum': float(target_quality['changed_q_delta_sum']),
            'mcts_changed_q_delta_values': changed_q_delta_values,
            'mcts_changed_q_delta_hist': changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(target_quality['changed_q_delta_sum']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 10)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 50)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 90)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(target_quality['changed_to_higher_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'resigned_games': int(resigned_games),
            'completed_length_sum': int(completed_length_sum),
            'truncated_length_sum': int(truncated_length_sum),
            'completed_white_wins': int(completed_white_wins),
            'completed_black_wins': int(completed_black_wins),
            'completed_draws': int(completed_draws),
            'learner_wins': int(learner_wins),
            'learner_draws': int(learner_draws),
            'learner_losses': int(learner_losses),
            'decisive_games': int(decisive_games),
            'decisive_length_sum': int(decisive_length_sum),
            'curriculum_dropped_positions': int(curriculum_dropped_positions),
            'cap_dropped_positions': int(cap_dropped_positions),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
        }

    def _select_move_from_visits(self, visit_counts, temperature):
        return _select_move_from_visits_safe(visit_counts, temperature)


# ============================================================================
# PARALLEL WORKER FOR MULTIPROCESSING
# ============================================================================

def _build_worker_logger(rank, worker_verbose):
    def _wlog(message):
        if worker_verbose:
            print(f"Worker {rank}: {message}")

    return _wlog


def _configure_selfplay_worker_runtime(config, device_id):
    rl_cfg = config.get('reinforcement_learning', {})
    self_play_threads = rl_cfg.get('self_play_torch_threads', None)

    _configure_inductor_for_selfplay(config)

    if self_play_threads is not None:
        try:
            self_play_threads = int(self_play_threads)
            if self_play_threads > 0:
                torch.set_num_threads(self_play_threads)
                try:
                    torch.set_num_interop_threads(max(1, min(4, self_play_threads)))
                except Exception:
                    pass
        except Exception:
            pass

    if device_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    return device


def _build_selfplay_worker_model(config, device):
    from src.model import ChessNet

    worker_config = dict(config)
    worker_config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = ChessNet(worker_config).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def _load_worker_model_state(model, model_state, rank):
    from src.model import normalize_state_dict_keys

    # Accept both raw state_dict and full training checkpoints.
    if isinstance(model_state, dict):
        if 'model_state_dict' in model_state and isinstance(model_state.get('model_state_dict'), dict):
            model_state = model_state['model_state_dict']
        elif 'state_dict' in model_state and isinstance(model_state.get('state_dict'), dict):
            model_state = model_state['state_dict']

    if model_state:
        model_state = {
            k: v for k, v in model_state.items()
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        }
        model_state = normalize_state_dict_keys(
            model_state,
            target_keys=set(model.state_dict().keys()),
        )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing = [
        key for key in missing
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    unexpected = [
        key for key in unexpected
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Worker {rank}: incompatible state_dict. "
            f"Missing={missing}, Unexpected={unexpected}"
        )


class _RemoteInferenceModel:
    """Small model-like proxy used by MCTS workers with central GPU inference."""

    def __init__(
        self,
        model_label,
        request_queue,
        response_queue,
        worker_rank,
        timeout_s=120.0,
        stall_warning_s=60.0,
        debug_enabled=False,
        transport_dtype="float16",
        shared_buffer=None,
        shared_call_lock=None,
    ):
        self.model_label = str(model_label or "learner")
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_rank = int(worker_rank)
        timeout_s = float(timeout_s)
        self.timeout_s = None if timeout_s <= 0.0 else max(1.0, timeout_s)
        self.stall_warning_s = max(0.0, float(stall_warning_s))
        self.debug_enabled = bool(debug_enabled)
        self._request_counter = 0
        self._training = False
        self.last_server_batch_size = 0
        self._printed_first_request = False
        self._printed_first_response = False
        self.supports_remote_legal_gather = True
        self.last_response_compact_policy = False
        self.last_request_put_s = 0.0
        self.last_remote_wait_s = 0.0
        self.last_server_queue_wait_s = 0.0
        self.last_server_descriptor_queue_wait_s = 0.0
        self.last_server_batch_coalesce_wait_s = 0.0
        self.last_server_total_s = 0.0
        self.last_server_concat_s = 0.0
        self.last_server_h2d_s = 0.0
        self.last_server_forward_s = 0.0
        self.last_server_d2h_s = 0.0
        self.last_server_send_s = 0.0
        self.last_shared_slot_wait_s = 0.0
        self.last_shared_bytes_avoided = 0
        self.last_cache_queries = 0
        self.last_cache_bypassed_positions = 0
        self.last_cache_hits = 0
        self.last_dedup_hits = 0
        self.last_cache_suspensions = 0
        self.last_cache_reactivations = 0
        self.last_nn_evaluated_positions = 0
        self.last_server_cache_lookup_s = 0.0
        self.last_server_staging_copy_s = 0.0
        self.last_gpu_batch_fill = 0.0
        self.last_transport_shared = False
        transport_dtype = str(transport_dtype or "float16").lower()
        self.transport_dtype = "float32" if transport_dtype in {"float32", "fp32"} else "float16"
        self._transport_torch_dtype = torch.float32 if self.transport_dtype == "float32" else torch.float16
        self._transport_np_dtype = np.float32 if self.transport_dtype == "float32" else np.float16
        self.shared_buffer = shared_buffer
        self._shared_arrays = None
        self._shared_call_lock = shared_call_lock or threading.Lock()
        if shared_buffer is not None:
            self._shared_arrays = {
                name: shared_inference_array(shared_buffer, name)
                for name in (
                    "boards", "legal_indices", "legal_counts",
                    "policy_logits", "value_logits",
                    "request_tokens", "response_tokens",
                )
            }

    @property
    def training(self):
        return self._training

    def eval(self):
        self._training = False
        return self

    def train(self, mode=True):
        self._training = bool(mode)
        return self

    def to(self, *args, **kwargs):
        return self

    def _receive_response(self, wait_s):
        if hasattr(self.response_queue, "poll") and hasattr(self.response_queue, "recv"):
            if not self.response_queue.poll(wait_s):
                raise queue.Empty()
            return self.response_queue.recv()
        return self.response_queue.get(timeout=wait_s)

    def _ts(self):
        return time.strftime("%H:%M:%S")

    def __call__(self, board_tensors, apply_log_softmax=False, legal_index_matrix=None, **kwargs):
        if self._shared_arrays is None:
            return self._call_impl(
                board_tensors,
                apply_log_softmax=apply_log_softmax,
                legal_index_matrix=legal_index_matrix,
                **kwargs,
            )
        wait_t0 = time.perf_counter()
        with self._shared_call_lock:
            self.last_shared_slot_wait_s = time.perf_counter() - wait_t0
            return self._call_impl(
                board_tensors,
                apply_log_softmax=apply_log_softmax,
                legal_index_matrix=legal_index_matrix,
                **kwargs,
            )

    def _call_impl(self, board_tensors, apply_log_softmax=False, legal_index_matrix=None, **kwargs):
        if apply_log_softmax:
            raise ValueError("Remote inference proxy only supports raw policy logits.")
        self._request_counter += 1
        request_id = f"{os.getpid()}_{id(self)}_{self._request_counter}"
        if isinstance(board_tensors, torch.Tensor):
            boards_source = board_tensors.detach().to('cpu').contiguous().numpy()
        else:
            boards_source = np.asarray(board_tensors)
        batch_size = int(boards_source.shape[0])
        legal_source = None
        legal_cols = 0
        if legal_index_matrix is not None:
            legal_source = np.asarray(legal_index_matrix, dtype=np.int16)
            legal_cols = int(legal_source.shape[1]) if legal_source.ndim == 2 else 0
        legal_counts = kwargs.get("legal_counts")
        if legal_counts is None:
            legal_counts_source = np.full(batch_size, legal_cols, dtype=np.int16)
        else:
            legal_counts_source = np.asarray(legal_counts, dtype=np.int16).reshape(-1)

        shared_eligible = bool(
            self._shared_arrays is not None
            and self.transport_dtype == "float16"
            and legal_source is not None
            and batch_size <= int(self.shared_buffer.get("capacity", 0))
            and legal_cols <= int(self.shared_buffer.get("max_legal_moves", 0))
            and int(boards_source.shape[1]) == int(self.shared_buffer.get("input_planes", -1))
        )
        request = {
            "cmd": "infer",
            "rank": self.worker_rank,
            "request_id": request_id,
            "model_label": self.model_label,
            # perf_counter is monotonic and system-wide on supported Python
            # platforms, so it is safe for latency measurements across the
            # worker/server process boundary.
        }
        shared_slot = None
        if shared_eligible:
            shared_slot = int((self._request_counter - 1) % int(self.shared_buffer.get("slots", 1)))
            np.copyto(
                self._shared_arrays["boards"][shared_slot, :batch_size],
                boards_source,
                casting="unsafe",
            )
            self._shared_arrays["legal_indices"][shared_slot, :batch_size, :legal_cols] = legal_source
            self._shared_arrays["legal_counts"][shared_slot, :batch_size] = legal_counts_source
            shared_token = (
                ((int(os.getpid()) & 0x7FFFFFFF) << 32)
                | (int(self._request_counter) & 0xFFFFFFFF)
            )
            self._shared_arrays["request_tokens"][shared_slot] = shared_token
            request.update({
                "shared_memory": True,
                "shared_slot": shared_slot,
                "batch_size": batch_size,
                "legal_cols": legal_cols,
                "input_bytes": int(batch_size * np.prod(boards_source.shape[1:]) * np.dtype(np.float16).itemsize),
                "shared_token": int(shared_token),
            })
            self.last_transport_shared = True
        else:
            boards_np = np.ascontiguousarray(boards_source, dtype=self._transport_np_dtype)
            request["boards"] = boards_np
            if legal_source is not None:
                request["legal_index_matrix"] = np.ascontiguousarray(legal_source, dtype=np.int16)
                request["legal_counts"] = np.ascontiguousarray(legal_counts_source, dtype=np.int16)
            self.last_transport_shared = False
        # Timestamp after all local packing/copies.  The server-side delta now
        # measures descriptor queueing/batching only, not worker preparation.
        request["queued_at_perf"] = time.perf_counter()
        put_t0 = time.perf_counter()
        self.request_queue.put(request)
        self.last_request_put_s = time.perf_counter() - put_t0
        if self.debug_enabled and not self._printed_first_request:
            self._printed_first_request = True
            print(
                f"[{self._ts()}] Central inference: worker {self.worker_rank} sent first "
                f"{self.model_label} request ({batch_size} positions, "
                f"transport={'shared' if shared_eligible else 'queue'}).",
                flush=True,
            )
        started_at = time.time()
        deadline = None if self.timeout_s is None else started_at + self.timeout_s
        next_warning_at = (
            started_at + self.stall_warning_s
            if self.stall_warning_s > 0.0
            else None
        )
        while True:
            now = time.time()
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(f"Central inference timed out for model '{self.model_label}'.")
                wait_s = min(max(remaining, 0.01), 1.0)
            else:
                wait_s = 1.0
            if self.debug_enabled and next_warning_at is not None and now >= next_warning_at:
                waited_s = now - started_at
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} still waiting "
                    f"{waited_s:.0f}s for model '{self.model_label}'.",
                    flush=True,
                )
                next_warning_at = now + self.stall_warning_s
            try:
                response = self._receive_response(wait_s)
            except queue.Empty:
                continue
            if response.get("request_id") != request_id:
                response_request_id = str(response.get("request_id") or "")
                if not response_request_id.startswith(f"{os.getpid()}_"):
                    continue
                raise RuntimeError(
                    "Central inference response routing failed: "
                    f"worker {self.worker_rank} received response for another request."
                )
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "central inference failed"))
            self.last_remote_wait_s = time.time() - started_at
            self.last_server_batch_size = int(response.get("server_batch_size", 0) or 0)
            self.last_response_compact_policy = bool(response.get("compact_policy", False))
            self.last_server_queue_wait_s = float(response.get("server_queue_wait_s", 0.0) or 0.0)
            self.last_server_descriptor_queue_wait_s = float(
                response.get("server_descriptor_queue_wait_s", 0.0) or 0.0
            )
            self.last_server_batch_coalesce_wait_s = float(
                response.get("server_batch_coalesce_wait_s", 0.0) or 0.0
            )
            self.last_server_total_s = float(response.get("server_total_time_s", 0.0) or 0.0)
            self.last_server_concat_s = float(response.get("server_concat_time_s", 0.0) or 0.0)
            self.last_server_h2d_s = float(response.get("server_h2d_time_s", 0.0) or 0.0)
            self.last_server_forward_s = float(response.get("server_forward_time_s", 0.0) or 0.0)
            self.last_server_d2h_s = float(response.get("server_d2h_time_s", 0.0) or 0.0)
            self.last_server_send_s = float(response.get("server_send_time_s", 0.0) or 0.0)
            self.last_shared_bytes_avoided = int(response.get("shared_bytes_avoided", 0) or 0)
            self.last_cache_queries = int(response.get("cache_queries", 0) or 0)
            self.last_cache_bypassed_positions = int(
                response.get("cache_bypassed_positions", 0) or 0
            )
            self.last_cache_hits = int(response.get("cache_hits", 0) or 0)
            self.last_dedup_hits = int(response.get("dedup_hits", 0) or 0)
            self.last_cache_suspensions = int(response.get("cache_suspensions", 0) or 0)
            self.last_cache_reactivations = int(response.get("cache_reactivations", 0) or 0)
            self.last_nn_evaluated_positions = int(response.get("nn_evaluated_positions", 0) or 0)
            self.last_server_cache_lookup_s = float(response.get("server_cache_lookup_s", 0.0) or 0.0)
            self.last_server_staging_copy_s = float(response.get("server_staging_copy_s", 0.0) or 0.0)
            self.last_gpu_batch_fill = float(response.get("gpu_batch_fill", 0.0) or 0.0)
            if self.debug_enabled and not self._printed_first_response:
                self._printed_first_response = True
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} received first "
                    f"{self.model_label} response (server_batch={self.last_server_batch_size}).",
                    flush=True,
                )
            if bool(response.get("shared_memory", False)):
                response_slot = int(response.get("shared_slot", shared_slot if shared_slot is not None else 0))
                expected_token = int(response.get("shared_token", 0) or 0)
                actual_token = int(self._shared_arrays["response_tokens"][response_slot])
                if expected_token <= 0 or actual_token != expected_token:
                    raise RuntimeError(
                        "Central inference shared-memory response token mismatch."
                    )
                response_batch = int(response.get("batch_size", batch_size))
                policy_cols = int(response.get("policy_cols", legal_cols))
                policy_np = self._shared_arrays[
                    "policy_logits"
                ][response_slot, :response_batch, :policy_cols].copy()
                value_np = self._shared_arrays[
                    "value_logits"
                ][response_slot, :response_batch, :3].copy()
                policy = torch.from_numpy(policy_np)
                value = torch.from_numpy(value_np)
            else:
                policy_dtype = np.float16 if self.last_response_compact_policy else np.float32
                policy = torch.from_numpy(np.asarray(response["policy_logits"], dtype=policy_dtype))
                value = torch.from_numpy(np.asarray(response["value_logits"], dtype=np.float32))
            return policy, value


def _central_inference_option(config, key, default=None):
    shared_cfg = (config or {}).get('central_inference', {}) or {}
    return shared_cfg.get(key, default)


class _AdaptiveInferenceCacheGate:
    """Limit exact-cache CPU cost without disabling it for the whole run.

    A weak active window suspends hashing temporarily.  Positions processed
    while suspended form a cheap cooldown; after it expires the cache gets a
    fresh probe window and can recover in a later game phase.
    """

    def __init__(
        self,
        enabled,
        probe_positions=8192,
        min_saved_rate=0.10,
        cooldown_positions=65536,
    ):
        self.enabled = bool(enabled)
        self.probe_positions = max(1, int(probe_positions))
        self.min_saved_rate = max(0.0, float(min_saved_rate))
        self.cooldown_positions = max(1, int(cooldown_positions))
        self.reset()

    def reset(self):
        self.active = bool(self.enabled)
        self.window_queries = 0
        self.window_saved = 0
        self.cooldown_progress = 0

    def record_active(self, queries, saved):
        """Record an active-cache group; return True when it is suspended."""
        if not self.active:
            return False
        self.window_queries += max(0, int(queries))
        self.window_saved += max(0, int(saved))
        if self.window_queries < self.probe_positions:
            return False
        saved_rate = float(self.window_saved) / float(max(1, self.window_queries))
        self.window_queries = 0
        self.window_saved = 0
        if saved_rate >= self.min_saved_rate:
            return False
        self.active = False
        self.cooldown_progress = 0
        return True

    def record_bypass(self, positions):
        """Record an unhashed group; return True when a fresh probe is armed."""
        if self.active or not self.enabled:
            return False
        self.cooldown_progress += max(0, int(positions))
        if self.cooldown_progress < self.cooldown_positions:
            return False
        self.active = True
        self.cooldown_progress = 0
        self.window_queries = 0
        self.window_saved = 0
        return True


def _prioritize_central_inference_process():
    """Keep the GPU feeder responsive when self-play saturates every CPU.

    Workers are CPU-heavy and normally occupy all logical processors. On
    Windows, ABOVE_NORMAL prevents the single GPU-owner process from waiting
    behind CPU-saturating tree-search processes without changing affinity or
    stealing a dedicated core while the server is idle.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        process_handle = kernel32.GetCurrentProcess()
        above_normal_priority_class = 0x00008000
        return bool(kernel32.SetPriorityClass(process_handle, above_normal_priority_class))
    except Exception:
        return False


def central_inference_server(
    config,
    device_id,
    request_queue,
    response_queues,
    control_queue=None,
    server_rank=None,
    shared_buffers=None,
):
    """Own GPU inference and batch requests coming from self-play workers."""
    rl_cfg = config.get('reinforcement_learning', {})
    _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
    max_batch = max(1, int(_central_inference_option(
        config,
        'max_batch_size',
        rl_cfg.get('mcts_batch_size', 256),
    )))
    flush_ms = max(0.0, float(_central_inference_option(config, 'flush_ms', 5.0)))
    debug_enabled = bool(
        debug_root_enabled and central_debug_cfg.get(
            'verbose',
            rl_cfg.get('self_play_central_inference_debug', False),
        )
    )
    quiet_startup = True
    central_use_compile = bool(_central_inference_option(config, 'use_compile', False))
    sync_timing = bool(_central_inference_option(config, 'sync_timing', debug_enabled))
    transport_dtype = str(_central_inference_option(config, 'transport_dtype', 'float16') or 'float16').lower()
    transport_np_dtype = np.float32 if transport_dtype in {'float32', 'fp32'} else np.float16
    shared_buffers = {
        int(rank): spec
        for rank, spec in dict(shared_buffers or {}).items()
        if spec is not None
    }
    shared_arrays_by_rank = {
        int(rank): {
            name: shared_inference_array(spec, name)
            for name in (
                'boards', 'legal_indices', 'legal_counts',
                'policy_logits', 'value_logits',
                'request_tokens', 'response_tokens',
            )
        }
        for rank, spec in shared_buffers.items()
    }
    cache_enabled = bool(_central_inference_option(config, 'cache_enabled', True))
    cache_max_entries = max(
        0,
        int(_central_inference_option(config, 'cache_max_entries', 50000) or 0),
    )
    cache = OrderedDict()
    # Hashing exact 80-plane histories costs roughly 3-7% on an all-miss A/B.
    # Keep the optimization only while it saves enough GPU evaluations to pay
    # for itself; reset the probe whenever model weights change.
    cache_probe_positions = 8192
    cache_min_saved_rate = 0.10
    # With a persistently cold cache this samples about 1/9 of positions.  At
    # the measured 3-7% all-miss hashing penalty that caps average overhead
    # below roughly 1%, while still revisiting later middlegame/endgame phases.
    cache_reprobe_cooldown_positions = 65536
    cache_gate = _AdaptiveInferenceCacheGate(
        cache_enabled and cache_max_entries > 0,
        probe_positions=cache_probe_positions,
        min_saved_rate=cache_min_saved_rate,
        cooldown_positions=cache_reprobe_cooldown_positions,
    )
    pinned_staging_enabled = bool(
        _central_inference_option(config, 'pinned_staging_enabled', False)
    )
    pinned_staging = {}
    central_use_bfloat16 = bool(_central_inference_option(
        config,
        'use_bfloat16',
        config.get('hardware', {}).get('use_bfloat16', False),
    ))
    central_amp_dtype = torch.bfloat16 if central_use_bfloat16 else torch.float16

    try:
        _prioritize_central_inference_process()
        device = _configure_selfplay_worker_runtime(config, device_id)
        try:
            compile_rank = int(server_rank)
        except Exception:
            compile_rank = 900000 + int(os.getpid())
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = bool(
                _central_inference_option(config, 'cudnn_benchmark', False)
            )
        models = {}
        compiled_base_models = {}
        compiled_wrappers = {}

        def _ts():
            return time.strftime("%H:%M:%S")

        if not quiet_startup:
            print(
                f"[{_ts()}] Central inference: server ready on {device} "
                f"(pid={os.getpid()}, compile_rank={compile_rank}, max_batch={max_batch}, flush={flush_ms:.1f}ms, "
                f"transport={transport_np_dtype.__name__}, "
                f"compile={'on' if central_use_compile else 'off'}, "
                f"cudnn.benchmark={torch.backends.cudnn.benchmark if device.type == 'cuda' else 'n/a'}).",
                flush=True,
            )

        def _warmup_central_model(model, label):
            if not central_use_compile or device.type != 'cuda':
                return None
            raw_batches = _central_inference_option(
                config,
                'compile_warmup_batches',
                [1, 32, 128, max_batch],
            )
            try:
                warmup_batches = [
                    max(1, int(batch_size))
                    for batch_size in list(raw_batches or [])
                ]
            except Exception:
                warmup_batches = [1, 32, 128, max_batch]
            warmup_batches = sorted({batch_size for batch_size in warmup_batches if batch_size > 0})
            if not warmup_batches:
                return None
            history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
            input_planes = 16 * (1 + history_positions)
            use_amp = bool(config.get('hardware', {}).get('use_amp', False))
            dummy_dtype = torch.float16 if use_amp and transport_np_dtype == np.float16 else torch.float32
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warming compiled model '{label}' "
                    f"for batches={warmup_batches}...",
                    flush=True,
                )
            warmup_t0 = time.perf_counter()
            with torch.inference_mode():
                for batch_size in warmup_batches:
                    dummy_input = torch.zeros(
                        batch_size,
                        input_planes,
                        8,
                        8,
                        device=device,
                        dtype=dummy_dtype,
                    ).to(memory_format=torch.channels_last)
                    with torch.autocast(device_type='cuda', enabled=use_amp, dtype=central_amp_dtype):
                        model(dummy_input, apply_log_softmax=False)
                    torch.cuda.synchronize(device)
            warmup_s = time.perf_counter() - warmup_t0
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warmup for '{label}' done "
                    f"in {warmup_s:.2f}s.",
                    flush=True,
                )
            return warmup_s

        def _load_model(label, state):
            nonlocal central_use_compile
            label = str(label or "learner")
            load_info = {
                "label": label,
                "compiled": bool(central_use_compile),
                "reused": False,
                "warmup_s": None,
            }
            if debug_enabled:
                print(f"[{_ts()}] Central inference: loading model '{label}' on {device}...", flush=True)
            if central_use_compile:
                if label in compiled_wrappers and label in compiled_base_models:
                    base_model = compiled_base_models[label]
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = compiled_wrappers[label]
                    load_info["reused"] = True
                    if debug_enabled:
                        print(f"[{_ts()}] Central inference: reused compiled model '{label}'.", flush=True)
                else:
                    base_model = _build_selfplay_worker_model(config, device)
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = _maybe_compile_selfplay_model(
                        base_model,
                        config,
                        device,
                        rank=compile_rank,
                        model_label=f"central:{label}",
                    )
                    compiled_base_models[label] = base_model
                    if model is not base_model:
                        try:
                            load_info["warmup_s"] = _warmup_central_model(model, label)
                            compiled_wrappers[label] = model
                        except Exception as exc:
                            # torch.compile is lazy: a cached Triton/Inductor
                            # artifact can fail only for a later warm-up shape.
                            # First rebuild only this server's disposable cache
                            # and retry. A benchmark must still be able to use
                            # the exact eager model if that retry also fails.
                            recovered = False
                            if _reset_selfplay_compile_cache(compile_rank, device):
                                retry_model = _maybe_compile_selfplay_model(
                                    base_model,
                                    config,
                                    device,
                                    rank=compile_rank,
                                    model_label=f"central:{label}:cache-retry",
                                )
                                if retry_model is not base_model:
                                    try:
                                        load_info["warmup_s"] = _warmup_central_model(retry_model, label)
                                        compiled_wrappers[label] = retry_model
                                        model = retry_model
                                        recovered = True
                                        print(
                                            f"Central inference: rebuilt torch.compile cache for '{label}'.",
                                            flush=True,
                                        )
                                    except Exception as retry_exc:
                                        exc = retry_exc
                            if not recovered:
                                print(
                                    f"WARNING: Central inference: torch.compile disabled for '{label}' "
                                    f"after cache rebuild failed ({type(exc).__name__}: {exc}). "
                                    "Continuing with eager inference.",
                                    flush=True,
                                )
                                compiled_wrappers.pop(label, None)
                                model = base_model
                                load_info["compiled"] = False
                                load_info["warmup_s"] = None
                                central_use_compile = False
                                try:
                                    torch._dynamo.reset()
                                except Exception:
                                    pass
                    else:
                        load_info["compiled"] = False
            else:
                model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(model, state, rank=-1)
                model.eval()
            models[label] = model
            if debug_enabled:
                print(f"[{_ts()}] Central inference: model '{label}' ready.", flush=True)
            return load_info

        def _response_queue_for(req):
            if isinstance(response_queues, dict):
                rank = int(req.get("rank", -1))
                response_queue = response_queues.get(rank)
                if response_queue is None:
                    raise RuntimeError(f"Central inference has no response channel for worker {rank}.")
                return response_queue
            return response_queues

        def _shared_request_arrays(req):
            if not bool(req.get('shared_memory', False)):
                return None
            return shared_arrays_by_rank.get(int(req.get('rank', -1)))

        def _request_boards(req):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                return np.asarray(req['boards'], dtype=transport_np_dtype)
            slot = int(req.get('shared_slot', 0))
            batch_size = int(req.get('batch_size', 0))
            return arrays['boards'][slot, :batch_size]

        def _request_legal_indices(req):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                value = req.get('legal_index_matrix')
                return None if value is None else np.asarray(value, dtype=np.int16)
            slot = int(req.get('shared_slot', 0))
            batch_size = int(req.get('batch_size', 0))
            legal_cols = int(req.get('legal_cols', 0))
            return arrays['legal_indices'][slot, :batch_size, :legal_cols]

        def _request_legal_counts(req, batch_size, legal_cols):
            arrays = _shared_request_arrays(req)
            if arrays is None:
                value = req.get('legal_counts')
                if value is None:
                    return np.full(batch_size, legal_cols, dtype=np.int16)
                return np.asarray(value, dtype=np.int16).reshape(-1)
            slot = int(req.get('shared_slot', 0))
            return arrays['legal_counts'][slot, :batch_size]

        def _send_response(response_channel, response):
            if hasattr(response_channel, "send"):
                response_channel.send(response)
            else:
                response_channel.put(response)

        def _put_response(req, payload):
            response = {
                "request_id": req.get("request_id"),
                "rank": int(req.get("rank", -1)),
            }
            response.update(payload)
            _send_response(_response_queue_for(req), response)

        def _infer_group(model_label, requests):
            nonlocal central_use_compile
            group_t0 = time.perf_counter()
            concat_time = 0.0
            staging_copy_time = 0.0
            cache_lookup_time = 0.0
            h2d_time = 0.0
            forward_time = 0.0
            d2h_time = 0.0
            send_time = 0.0
            model_label = str(model_label)
            model = models.get(model_label)
            if model is None:
                raise RuntimeError(f"Central inference has no model loaded for label '{model_label}'.")

            # Split pre-service latency at the exact server dequeue boundary.
            # descriptor_queue_wait is real backlog/IPC delivery time;
            # batch_coalesce_wait is deliberate waiting after receipt so other
            # requests can join the same GPU batch.  Their sum is the legacy
            # server_queue_wait value kept for backwards-compatible dashboards.
            descriptor_queue_waits = []
            batch_coalesce_waits = []
            total_queue_waits = []
            for req in requests:
                queued_at_perf = req.get("queued_at_perf")
                dequeued_at_perf = float(
                    req.get("server_dequeued_at_perf", group_t0) or group_t0
                )
                if queued_at_perf is None:
                    descriptor_wait = 0.0
                else:
                    descriptor_wait = max(
                        0.0,
                        dequeued_at_perf - float(queued_at_perf),
                    )
                coalesce_wait = max(0.0, group_t0 - dequeued_at_perf)
                descriptor_queue_waits.append(descriptor_wait)
                batch_coalesce_waits.append(coalesce_wait)
                total_queue_waits.append(descriptor_wait + coalesce_wait)

            use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
            input_np_dtype = transport_np_dtype if use_amp else np.float32
            concat_t0 = time.perf_counter()
            board_batches = [np.asarray(_request_boards(req)) for req in requests]
            batch_sizes = [int(batch.shape[0]) for batch in board_batches]
            total_n = int(sum(batch_sizes))
            input_planes = int(board_batches[0].shape[1])

            def _get_pinned(name, dtype, rows, trailing_shape):
                if not (pinned_staging_enabled and device.type == 'cuda'):
                    return None
                np_dtype = np.dtype(dtype)
                torch_dtype = torch.float16 if np_dtype == np.float16 else torch.float32
                key = (str(name), np_dtype.str, tuple(trailing_shape))
                tensor = pinned_staging.get(key)
                if tensor is None or int(tensor.shape[0]) < int(rows):
                    capacity = max(max_batch, int(rows))
                    try:
                        tensor = torch.empty(
                            (capacity, *tuple(trailing_shape)),
                            dtype=torch_dtype,
                            pin_memory=True,
                        )
                    except Exception:
                        return None
                    pinned_staging[key] = tensor
                return tensor

            board_staging = _get_pinned(
                'boards', input_np_dtype, total_n, (input_planes, 8, 8)
            )
            if board_staging is not None:
                staging_t0 = time.perf_counter()
                boards = board_staging[:total_n].numpy()
                cursor = 0
                for board_batch, batch_size in zip(board_batches, batch_sizes):
                    np.copyto(
                        boards[cursor:cursor + batch_size],
                        board_batch,
                        casting='unsafe',
                    )
                    cursor += batch_size
                staging_copy_time += time.perf_counter() - staging_t0
            else:
                converted_batches = [
                    np.asarray(batch, dtype=input_np_dtype)
                    for batch in board_batches
                ]
                boards = (
                    converted_batches[0]
                    if len(converted_batches) == 1
                    else np.concatenate(converted_batches, axis=0)
                )

            legal_index_batches = []
            legal_count_batches = []
            compact_policy = True
            max_legal_cols = 0
            for req, batch_size in zip(requests, batch_sizes):
                legal_idx = _request_legal_indices(req)
                if legal_idx is None or legal_idx.ndim != 2 or int(legal_idx.shape[0]) != batch_size:
                    compact_policy = False
                    legal_index_batches = []
                    legal_count_batches = []
                    break
                legal_cols = int(legal_idx.shape[1])
                legal_index_batches.append(legal_idx)
                legal_count_batches.append(
                    _request_legal_counts(req, batch_size, legal_cols)
                )
                max_legal_cols = max(max_legal_cols, legal_cols)

            if compact_policy and max_legal_cols > 0:
                legal_indices = np.zeros((total_n, max_legal_cols), dtype=np.int16)
                legal_counts = np.zeros(total_n, dtype=np.int16)
                cursor = 0
                for legal_idx, counts, batch_size in zip(
                    legal_index_batches, legal_count_batches, batch_sizes
                ):
                    cols = int(legal_idx.shape[1])
                    legal_indices[cursor:cursor + batch_size, :cols] = legal_idx
                    legal_counts[cursor:cursor + batch_size] = np.clip(
                        counts[:batch_size], 0, cols
                    )
                    cursor += batch_size
            else:
                compact_policy = False
                legal_indices = None
                legal_counts = None
            concat_time += time.perf_counter() - concat_t0

            cache_hit_flags = np.zeros(total_n, dtype=np.bool_)
            dedup_hit_flags = np.zeros(total_n, dtype=np.bool_)
            cache_keys = [None] * total_n
            representative_for = np.arange(total_n, dtype=np.int32)
            miss_indices = []
            pending_representatives = {}
            policy_np_batch = (
                np.zeros((total_n, max_legal_cols), dtype=np.float16)
                if compact_policy else None
            )
            value_np_batch = np.zeros((total_n, 3), dtype=np.float32)

            cache_t0 = time.perf_counter()
            group_cache_active = bool(cache_gate.active and compact_policy)
            if group_cache_active:
                for idx in range(total_n):
                    legal_count = int(legal_counts[idx])
                    digest = hashlib.blake2b(digest_size=16, person=b'chess-nn-cache')
                    digest.update(memoryview(boards[idx]).cast('B'))
                    digest.update(legal_count.to_bytes(2, 'little', signed=False))
                    if legal_count > 0:
                        digest.update(memoryview(legal_indices[idx, :legal_count]).cast('B'))
                    key = (model_label, digest.digest())
                    cache_keys[idx] = key
                    cached = cache.get(key)
                    if cached is not None:
                        cached_policy, cached_value = cached
                        policy_np_batch[idx, :len(cached_policy)] = cached_policy
                        value_np_batch[idx] = cached_value
                        cache.move_to_end(key)
                        cache_hit_flags[idx] = True
                        continue
                    representative = pending_representatives.get(key)
                    if representative is not None:
                        representative_for[idx] = int(representative)
                        dedup_hit_flags[idx] = True
                        continue
                    pending_representatives[key] = idx
                    miss_indices.append(idx)
            else:
                miss_indices = list(range(total_n))
            cache_lookup_time += time.perf_counter() - cache_t0

            eval_count = len(miss_indices)
            eval_boards = None
            eval_board_tensor = None
            eval_legal_indices = None
            if eval_count > 0:
                sequential_misses = eval_count == total_n and all(
                    idx == position for position, idx in enumerate(miss_indices)
                )
                if sequential_misses:
                    eval_boards = boards
                    eval_board_tensor = board_staging[:total_n] if board_staging is not None else None
                    eval_legal_indices = legal_indices
                else:
                    miss_staging = _get_pinned(
                        'cache_misses', input_np_dtype, eval_count, (input_planes, 8, 8)
                    )
                    staging_t0 = time.perf_counter()
                    if miss_staging is not None:
                        eval_board_tensor = miss_staging[:eval_count]
                        eval_boards = eval_board_tensor.numpy()
                        np.copyto(eval_boards, boards[miss_indices], casting='unsafe')
                    else:
                        eval_boards = np.ascontiguousarray(boards[miss_indices])
                    eval_legal_indices = (
                        np.ascontiguousarray(legal_indices[miss_indices])
                        if compact_policy else None
                    )
                    staging_copy_time += time.perf_counter() - staging_t0

            policy_np_chunks = []
            value_np_chunks = []
            for batch_start in range(0, eval_count, max_batch):
                batch_end = min(eval_count, batch_start + max_batch)
                h2d_t0 = time.perf_counter()
                source_tensor = (
                    eval_board_tensor[batch_start:batch_end]
                    if eval_board_tensor is not None
                    else torch.from_numpy(eval_boards[batch_start:batch_end])
                )
                tensor = source_tensor.to(
                    device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
                legal_index_tensor = None
                if compact_policy:
                    legal_index_tensor = torch.from_numpy(
                        eval_legal_indices[batch_start:batch_end]
                    ).to(device, non_blocking=True).long()
                if sync_timing and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                h2d_time += time.perf_counter() - h2d_t0
                with torch.inference_mode():
                    forward_t0 = time.perf_counter()
                    try:
                        if use_amp and device.type == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=central_amp_dtype):
                                policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                        else:
                            policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    except Exception as exc:
                        eager_model = compiled_base_models.get(model_label)
                        if eager_model is None or eager_model is model:
                            raise
                        recovered = False
                        if _reset_selfplay_compile_cache(compile_rank, device):
                            retry_model = _maybe_compile_selfplay_model(
                                eager_model,
                                config,
                                device,
                                rank=compile_rank,
                                model_label=f"central:{model_label}:cache-retry",
                            )
                            if retry_model is not eager_model:
                                try:
                                    if use_amp and device.type == 'cuda':
                                        with torch.autocast(device_type='cuda', dtype=central_amp_dtype):
                                            policy_logits, value_logits = retry_model(
                                                tensor, apply_log_softmax=False
                                            )
                                    else:
                                        policy_logits, value_logits = retry_model(
                                            tensor, apply_log_softmax=False
                                        )
                                    model = retry_model
                                    models[model_label] = retry_model
                                    compiled_wrappers[model_label] = retry_model
                                    recovered = True
                                    print(
                                        f"Central inference: rebuilt torch.compile cache for '{model_label}'.",
                                        flush=True,
                                    )
                                except Exception as retry_exc:
                                    exc = retry_exc
                        if not recovered:
                            print(
                                f"WARNING: Central inference: compiled forward failed for '{model_label}' "
                                f"after cache rebuild ({type(exc).__name__}: {exc}). "
                                "Continuing with eager inference.",
                                flush=True,
                            )
                            eager_model.eval()
                            model = eager_model
                            models[model_label] = eager_model
                            compiled_wrappers.pop(model_label, None)
                            central_use_compile = False
                            try:
                                torch._dynamo.reset()
                            except Exception:
                                pass
                            if use_amp and device.type == 'cuda':
                                with torch.autocast(device_type='cuda', dtype=central_amp_dtype):
                                    policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                            else:
                                policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    if sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_time += time.perf_counter() - forward_t0
                    d2h_t0 = time.perf_counter()
                    if compact_policy and legal_index_tensor is not None:
                        policy_logits = torch.gather(policy_logits, 1, legal_index_tensor)
                    policy_np_chunks.append(policy_logits.to(dtype=torch.float16).cpu().numpy())
                    value_np_chunks.append(value_logits.float().cpu().numpy())
                    if sync_timing and device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    d2h_time += time.perf_counter() - d2h_t0

            if eval_count > 0:
                evaluated_policy = (
                    policy_np_chunks[0] if len(policy_np_chunks) == 1
                    else np.concatenate(policy_np_chunks, axis=0)
                )
                evaluated_values = (
                    value_np_chunks[0] if len(value_np_chunks) == 1
                    else np.concatenate(value_np_chunks, axis=0)
                )
                if compact_policy:
                    for eval_idx, original_idx in enumerate(miss_indices):
                        legal_count = int(legal_counts[original_idx])
                        policy_np_batch[original_idx, :max_legal_cols] = evaluated_policy[eval_idx]
                        value_np_batch[original_idx] = evaluated_values[eval_idx]
                        key = cache_keys[original_idx]
                        if key is not None:
                            cache[key] = (
                                evaluated_policy[eval_idx, :legal_count].copy(),
                                evaluated_values[eval_idx].copy(),
                            )
                            cache.move_to_end(key)
                            while len(cache) > cache_max_entries:
                                cache.popitem(last=False)
                    for idx in np.flatnonzero(dedup_hit_flags):
                        representative = int(representative_for[idx])
                        policy_np_batch[idx] = policy_np_batch[representative]
                        value_np_batch[idx] = value_np_batch[representative]
                else:
                    policy_np_batch = evaluated_policy
                    value_np_batch = evaluated_values

            group_cache_queries = total_n if group_cache_active else 0
            group_cache_bypassed = total_n if compact_policy and not group_cache_active else 0
            group_cache_saved = int(np.count_nonzero(cache_hit_flags)) + int(
                np.count_nonzero(dedup_hit_flags)
            )
            cache_suspended = False
            cache_reactivated = False
            if group_cache_queries > 0:
                cache_suspended = cache_gate.record_active(
                    group_cache_queries,
                    group_cache_saved,
                )
                if cache_suspended:
                    cache.clear()
            elif group_cache_bypassed > 0:
                # Reactivation applies to the next group, so every position is
                # counted exactly once as either queried or bypassed.
                cache_reactivated = cache_gate.record_bypass(group_cache_bypassed)

            forward_chunk_count = len(policy_np_chunks)
            effective_forward_batch = (
                float(eval_count) / float(forward_chunk_count)
                if forward_chunk_count > 0 else 0.0
            )
            cursor = 0
            for req_idx, (req, batch_size) in enumerate(zip(requests, batch_sizes)):
                req_end = cursor + batch_size
                req_legal_cols = (
                    int(legal_index_batches[req_idx].shape[1])
                    if compact_policy else int(policy_np_batch.shape[1])
                )
                policy_slice = policy_np_batch[cursor:req_end, :req_legal_cols]
                value_slice = value_np_batch[cursor:req_end]
                cache_queries = batch_size if group_cache_queries > 0 else 0
                cache_bypassed_positions = batch_size if group_cache_bypassed > 0 else 0
                cache_hits = int(np.count_nonzero(cache_hit_flags[cursor:req_end]))
                dedup_hits = int(np.count_nonzero(dedup_hit_flags[cursor:req_end]))
                nn_evaluated = max(0, cache_queries - cache_hits - dedup_hits) if cache_queries else batch_size
                payload = {
                    "ok": True,
                    "server_batch_size": int(round(effective_forward_batch)),
                    "server_request_group_positions": total_n,
                    "compact_policy": bool(compact_policy),
                    "server_queue_wait_s": (
                        float(total_queue_waits[req_idx])
                        if req_idx < len(total_queue_waits) else 0.0
                    ),
                    "server_descriptor_queue_wait_s": (
                        float(descriptor_queue_waits[req_idx])
                        if req_idx < len(descriptor_queue_waits) else 0.0
                    ),
                    "server_batch_coalesce_wait_s": (
                        float(batch_coalesce_waits[req_idx])
                        if req_idx < len(batch_coalesce_waits) else 0.0
                    ),
                    "server_total_time_s": float(time.perf_counter() - group_t0),
                    "server_concat_time_s": float(concat_time),
                    "server_h2d_time_s": float(h2d_time),
                    "server_forward_time_s": float(forward_time),
                    "server_d2h_time_s": float(d2h_time),
                    "server_send_time_s": float(send_time),
                    "server_cache_lookup_s": float(
                        cache_lookup_time * batch_size / max(1, total_n)
                    ),
                    "server_staging_copy_s": float(
                        staging_copy_time * batch_size / max(1, total_n)
                    ),
                    "cache_queries": int(cache_queries),
                    "cache_bypassed_positions": int(cache_bypassed_positions),
                    "cache_hits": int(cache_hits),
                    "dedup_hits": int(dedup_hits),
                    "nn_evaluated_positions": int(nn_evaluated),
                    "cache_entries": int(len(cache)),
                    "cache_active": bool(cache_gate.active),
                    "cache_suspensions": int(cache_suspended and req_idx == 0),
                    "cache_reactivations": int(cache_reactivated and req_idx == 0),
                    "gpu_batch_fill": float(effective_forward_batch / max(1, max_batch)),
                }
                arrays = _shared_request_arrays(req)
                if arrays is not None and compact_policy:
                    slot = int(req.get('shared_slot', 0))
                    shared_token = int(req.get('shared_token', 0) or 0)
                    arrays['policy_logits'][slot, :batch_size, :req_legal_cols] = policy_slice
                    arrays['value_logits'][slot, :batch_size, :3] = value_slice
                    # Publish the token last. The Queue/Pipe response is the
                    # notification that makes the preceding slot writes visible
                    # to the worker; the token prevents a late response from
                    # being mistaken for contents of a reused slot.
                    arrays['response_tokens'][slot] = shared_token
                    payload.update({
                        "shared_memory": True,
                        "shared_slot": slot,
                        "shared_token": shared_token,
                        "batch_size": batch_size,
                        "policy_cols": req_legal_cols,
                        "shared_bytes_avoided": int(
                            int(req.get('input_bytes', 0) or 0)
                            + batch_size * req_legal_cols * np.dtype(np.int16).itemsize
                            + policy_slice.nbytes
                            + value_slice.nbytes
                        ),
                    })
                else:
                    payload["policy_logits"] = policy_slice
                    payload["value_logits"] = value_slice
                    payload["shared_memory"] = False
                    payload["shared_bytes_avoided"] = 0
                cursor = req_end
                send_t0 = time.perf_counter()
                _put_response(req, payload)
                send_time += time.perf_counter() - send_t0
            return {
                "total_time": time.perf_counter() - group_t0,
                "concat_time": concat_time,
                "h2d_time": h2d_time,
                "forward_time": forward_time,
                "d2h_time": d2h_time,
                "send_time": send_time,
                "compact_policy": bool(compact_policy),
                "positions": int(total_n),
                "nn_evaluated_positions": int(eval_count),
                "cache_hits": int(np.count_nonzero(cache_hit_flags)),
                "dedup_hits": int(np.count_nonzero(dedup_hit_flags)),
                "requests": int(len(requests)),
            }

        pending = []
        pending_positions = 0
        pending_positions_by_model = {}
        pending_started_at = None
        first_infer_seen = False
        processed_requests = 0
        processed_positions = 0
        last_debug_print = time.perf_counter()
        last_batch_positions = 0
        last_batch_time_s = 0.0
        last_stage_stats = {}
        interval_infer_time = 0.0
        interval_idle_wait_time = 0.0
        interval_batch_wait_time = 0.0
        interval_batches = 0

        def _request_positions(req):
            if bool(req.get("shared_memory", False)):
                return int(req.get("batch_size", 0) or 0)
            boards = req.get("boards")
            shape = getattr(boards, "shape", None)
            if shape:
                return int(shape[0])
            return int(np.asarray(boards).shape[0])

        def _handle_request_item(item):
            nonlocal first_infer_seen, pending_started_at, pending_positions
            cmd = item.get("cmd")
            if cmd == "stop":
                return "stop"
            if cmd == "load_models":
                task_id = item.get("task_id")
                if bool(item.get("clear", False)):
                    models.clear()
                    cache.clear()
                    cache_gate.reset()
                loaded_models = []
                load_t0 = time.perf_counter()
                for entry in list(item.get("models", []) or []):
                    state = entry.get("state")
                    state_path = entry.get("state_path")
                    if state is None and state_path:
                        state = torch.load(state_path, map_location='cpu')
                    if state is not None:
                        loaded_models.append(_load_model(entry.get("label", "learner"), state))
                if loaded_models:
                    # A label may have received new weights. Never reuse outputs
                    # computed by a previous model generation.
                    cache.clear()
                    cache_gate.reset()
                if loaded_models:
                    def _format_loaded_model(info):
                        label = str(info.get("label", "model"))
                        if info.get("reused"):
                            return f"{label}(reused)"
                        warmup_s = info.get("warmup_s")
                        if warmup_s is not None:
                            return f"{label}(warmup {float(warmup_s):.2f}s)"
                        if info.get("compiled"):
                            return f"{label}(compiled)"
                        return label

                    model_summary = ", ".join(_format_loaded_model(info) for info in loaded_models)
                    if not quiet_startup:
                        print(
                            f"[{_ts()}] Central inference: models ready on {device}: "
                            f"{model_summary} ({time.perf_counter() - load_t0:.2f}s).",
                            flush=True,
                        )
                if control_queue is not None:
                    control_queue.put({
                        "type": "models_loaded",
                        "task_id": task_id,
                        "labels": sorted(models.keys()),
                        "pid": int(os.getpid()),
                        "server_rank": int(compile_rank),
                        "device": str(device),
                        "max_batch": int(max_batch),
                        "flush_ms": float(flush_ms),
                        "transport": str(transport_np_dtype.__name__),
                        "shared_memory": bool(shared_arrays_by_rank),
                        "shared_workers": int(len(shared_arrays_by_rank)),
                        "cache_enabled": bool(cache_enabled and cache_max_entries > 0),
                        "cache_max_entries": int(cache_max_entries),
                        "pinned_staging": bool(pinned_staging_enabled and device.type == 'cuda'),
                        "compile": bool(central_use_compile),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device.type == 'cuda' else None,
                        "model_summary": model_summary if loaded_models else "",
                        "load_s": float(time.perf_counter() - load_t0),
                    })
                return "control"
            if cmd == "infer":
                item["server_dequeued_at_perf"] = time.perf_counter()
                arrays = _shared_request_arrays(item)
                if arrays is not None:
                    slot = int(item.get('shared_slot', 0))
                    expected_token = int(item.get('shared_token', 0) or 0)
                    actual_token = int(arrays['request_tokens'][slot])
                    if expected_token <= 0 or actual_token != expected_token:
                        _put_response(item, {
                            "ok": False,
                            "error": (
                                "Central inference rejected a stale shared-memory "
                                "request descriptor."
                            ),
                        })
                        return "ignored"
                if pending_started_at is None:
                    pending_started_at = time.perf_counter()
                item_positions = _request_positions(item)
                if debug_enabled and not first_infer_seen:
                    first_infer_seen = True
                    print(
                        f"[{_ts()}] Central inference: received first inference request "
                        f"from worker {int(item.get('rank', -1))} "
                        f"({item_positions} positions).",
                        flush=True,
                    )
                pending.append(item)
                pending_positions += item_positions
                model_label = str(item.get("model_label", "learner"))
                pending_positions_by_model[model_label] = (
                    int(pending_positions_by_model.get(model_label, 0)) + item_positions
                )
                return "infer"
            return "ignored"

        while True:
            get_t0 = time.perf_counter()
            if pending_started_at is None:
                item = request_queue.get()
            else:
                elapsed_s = time.perf_counter() - pending_started_at
                remaining_s = (flush_ms / 1000.0) - elapsed_s
                if remaining_s <= 0.0:
                    item = None
                else:
                    try:
                        item = request_queue.get(timeout=remaining_s)
                    except queue.Empty:
                        item = None
            get_wait_s = time.perf_counter() - get_t0
            if pending_started_at is None:
                interval_idle_wait_time += get_wait_s
            else:
                interval_batch_wait_time += get_wait_s

            if item is not None:
                item_status = _handle_request_item(item)
                if item_status == "stop":
                    break
                if item_status == "control":
                    continue

            # max_batch is the useful forward target for each loaded model,
            # not for the mixed learner/opponent queue as a whole.  Coalesce
            # already-available work for every label before serial GPU calls.
            drain_target_positions = max_batch * max(1, len(models))
            while (
                pending
                and pending_positions < drain_target_positions
                and not any(
                    int(model_positions) >= max_batch
                    for model_positions in pending_positions_by_model.values()
                )
            ):
                try:
                    drained_item = request_queue.get_nowait()
                except queue.Empty:
                    break
                item_status = _handle_request_item(drained_item)
                if item_status == "stop":
                    return
                if item_status == "control":
                    continue

            should_flush = (
                pending
                and (
                    any(
                        int(model_positions) >= max_batch
                        for model_positions in pending_positions_by_model.values()
                    )
                    or (
                        pending_started_at is not None
                        and (time.perf_counter() - pending_started_at) >= (flush_ms / 1000.0)
                    )
                    or item is None
                )
            )
            if not should_flush:
                continue

            grouped = {}
            for req in pending:
                grouped.setdefault(str(req.get("model_label", "learner")), []).append(req)
            pending = []
            pending_positions = 0
            pending_positions_by_model.clear()
            pending_started_at = None
            grouped_items = sorted(
                grouped.items(),
                key=lambda item_pair: sum(_request_positions(req) for req in item_pair[1]),
                reverse=True,
            )
            for model_label, requests in grouped_items:
                request_batches = []
                current_batch = []
                current_positions = 0
                for req in requests:
                    request_positions = _request_positions(req)
                    if current_batch and current_positions + request_positions > max_batch:
                        request_batches.append(current_batch)
                        current_batch = []
                        current_positions = 0
                    current_batch.append(req)
                    current_positions += request_positions
                if current_batch:
                    request_batches.append(current_batch)

                for request_batch in request_batches:
                    try:
                        infer_t0 = time.perf_counter()
                        stage_stats = _infer_group(model_label, request_batch)
                        last_batch_time_s = time.perf_counter() - infer_t0
                        last_batch_positions = int(stage_stats.get("positions", 0) or 0)
                        last_stage_stats = dict(stage_stats)
                        interval_infer_time += last_batch_time_s
                        interval_batches += 1
                        processed_requests += len(request_batch)
                        processed_positions += last_batch_positions
                    except Exception as exc:
                        for req in request_batch:
                            _put_response(req, {
                                "ok": False,
                                "error": str(exc),
                            })
            if debug_enabled and (time.perf_counter() - last_debug_print) >= 5.0:
                interval_s = max(1e-9, time.perf_counter() - last_debug_print)
                last_debug_print = time.perf_counter()
                pos_per_s = (
                    float(last_batch_positions) / max(1e-9, float(last_batch_time_s))
                    if last_batch_positions > 0
                    else 0.0
                )
                active_pct = 100.0 * float(interval_infer_time) / interval_s
                idle_pct = 100.0 * float(interval_idle_wait_time) / interval_s
                batch_wait_pct = 100.0 * float(interval_batch_wait_time) / interval_s
                print(
                    f"[{_ts()}] Central inference: heartbeat "
                    f"requests={processed_requests}, positions={processed_positions}, "
                    f"last_batch={last_batch_positions} pos/{last_batch_time_s:.3f}s "
                    f"({pos_per_s:.1f} pos/s), "
                    f"server_active={active_pct:.1f}%, idle_no_requests={idle_pct:.1f}%, "
                    f"batch_wait={batch_wait_pct:.1f}%, batches={interval_batches}, "
                    f"compact={bool(last_stage_stats.get('compact_policy', False))}, "
                    f"h2d={float(last_stage_stats.get('h2d_time', 0.0)):.3f}s, "
                    f"fwd={float(last_stage_stats.get('forward_time', 0.0)):.3f}s, "
                    f"d2h={float(last_stage_stats.get('d2h_time', 0.0)):.3f}s, "
                    f"send={float(last_stage_stats.get('send_time', 0.0)):.3f}s, "
                    f"last_flush_models={len(grouped)}, pending={len(pending)}.",
                    flush=True,
                )
                interval_infer_time = 0.0
                interval_idle_wait_time = 0.0
                interval_batch_wait_time = 0.0
                interval_batches = 0
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Central inference server failed: {exc}")
        import traceback
        traceback.print_exc()


def _create_selfplay_engine(
    model,
    config,
    device,
    num_games,
    wlog,
    opponent_model=None,
    opponent_source_label="current",
    opponent_models_by_label=None,
    opponent_plan_labels=None,
):
    rl_cfg = config.get('reinforcement_learning', {})
    use_batch = rl_cfg.get('use_batch_selfplay', False)
    max_batch_games = rl_cfg.get('max_batch_games_per_worker', 256)

    if use_batch:
        configured_max = max_batch_games
        mode_label = "batch"
    else:
        configured_max = 1
        mode_label = "single-via-batch"

    engine = BatchSelfPlayMCTSBatch(
        model,
        config,
        device,
        configured_max,
        opponent_model=opponent_model,
        opponent_source_label=opponent_source_label,
        opponent_models_by_label=opponent_models_by_label,
        opponent_plan_labels=opponent_plan_labels,
    )
    actual_max = min(configured_max, num_games)
    wlog(
        f"Self-play engine: {mode_label} "
        f"(max {configured_max}, actual {actual_max})"
    )
    return engine


def _play_games_with_engine(
    rank,
    engine,
    config,
    num_games,
    result_file_path,
    wlog,
    result_queue=None,
    task_id=None,
    stream_results_to_queue=False,
):
    rl_cfg = config.get('reinforcement_learning', {})
    progress_file_path = result_file_path.replace('.pkl', '.progress')

    def _write_progress(n):
        try:
            with open(progress_file_path, 'w') as _pf:
                _pf.write(str(n))
        except Exception:
            pass

    if hasattr(engine, '_progress_file'):
        engine._progress_file = progress_file_path

    if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False)):
        dynamic_min, dynamic_target, dynamic_max, _dynamic_chunk = (
            _resolve_dynamic_simulation_budget(
                rl_cfg.get('mcts_simulations', 192),
                minimum=rl_cfg.get('mcts_dynamic_budget_min', 64),
                maximum_multiplier=rl_cfg.get(
                    'mcts_dynamic_budget_max_multiplier',
                    5.0 / 3.0,
                ),
            )
        )
        search_budget_text = (
            f"dynamic {dynamic_min}-{dynamic_max}, avg {dynamic_target} sims/move"
        )
    else:
        search_budget_text = f"{int(rl_cfg.get('mcts_simulations', 0))} sims/move"
    tree_mode = (
        "shared tree + reuse"
        if bool(rl_cfg.get('self_play_share_trees', True))
        else "fresh tree after each move"
    )
    wlog(
        f"Playing {num_games} games with Gumbel AlphaZero "
        f"({search_budget_text}; {tree_mode})"
    )

    save_every = rl_cfg.get('self_play_save_every_games_resolved', None)
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
    if stream_results_to_queue:
        save_every = max(
            1,
            int(
                rl_cfg.get(
                    'self_play_queue_chunk_games',
                    rl_cfg.get('max_batch_games_per_worker', 1),
                )
            ),
        )
    elif save_every is None:
        save_mult = rl_cfg.get('self_play_save_every_games', 0)
        try:
            save_mult = float(save_mult)
        except Exception:
            save_mult = 0
        if save_mult and save_mult > 0:
            games_per_iter = rl_cfg.get('games_per_iteration', 0)
            try:
                games_per_iter = int(games_per_iter)
            except Exception:
                games_per_iter = 0
            save_every = max(1, int(round(games_per_iter * save_mult)))
        else:
            save_every = 0

    total_positions = 0
    total_games = 0
    total_dropped_positions = 0
    total_truncated_games = 0
    total_claimable_draw_ended_games = 0
    total_completed_length_sum = 0
    total_truncated_length_sum = 0

    if save_every and save_every > 0:
        if not stream_results_to_queue:
            with open(result_file_path, 'wb') as f:
                pass
        games_left = num_games
        while games_left > 0:
            chunk_games = min(save_every, games_left)
            if hasattr(engine, '_progress_file'):
                engine._progress_file = progress_file_path
            if hasattr(engine, '_progress_base'):
                engine._progress_base = total_games
            positions, game_lengths = engine.play_games(chunk_games)
            stats = getattr(engine, 'last_selfplay_stats', {}) or {}

            if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
                    'game_lengths': list(game_lengths),
                    'stats': stats,
                })
            else:
                with open(result_file_path, 'ab') as f:
                    pickle.dump((positions, game_lengths, stats), f)

            total_positions += len(positions)
            total_games += len(game_lengths)
            total_dropped_positions += int(stats.get('dropped_positions', 0))
            total_truncated_games += int(stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(stats.get('truncated_length_sum', 0))
            _write_progress(total_games)
            games_left -= chunk_games
    else:
        if hasattr(engine, '_progress_file'):
            engine._progress_file = progress_file_path
        positions, game_lengths = engine.play_games(num_games)
        stats = getattr(engine, 'last_selfplay_stats', {}) or {}
        total_positions = len(positions)
        total_games = len(game_lengths)
        total_dropped_positions = int(stats.get('dropped_positions', 0))
        total_truncated_games = int(stats.get('truncated_games', 0))
        total_claimable_draw_ended_games = int(stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum = int(stats.get('completed_length_sum', 0))
        total_truncated_length_sum = int(stats.get('truncated_length_sum', 0))
        _write_progress(total_games)

        if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
                    'game_lengths': list(game_lengths),
                    'stats': stats,
                })
        else:
            with open(result_file_path, 'wb') as f:
                pickle.dump((positions, game_lengths, stats), f)

    wlog(f"Generated {total_positions} positions from {total_games} games")
    if total_truncated_games > 0 or total_dropped_positions > 0:
        wlog(
            "Dropped due to truncation: "
            f"{total_dropped_positions} positions across {total_truncated_games} games"
        )
    if total_claimable_draw_ended_games > 0:
        wlog(f"Claimable-draw ended games: {total_claimable_draw_ended_games}")
    if total_games > 0:
        def _format_plies(value):
            return f"{value:.1f} plies (~{value / 2.0:.1f} full moves)"
        completed_games = max(0, total_games - total_truncated_games)
        if completed_games > 0:
            wlog(
                f"Avg completed game length: {_format_plies(total_completed_length_sum / completed_games)}"
            )
        if total_truncated_games > 0:
            wlog(
                f"Avg truncated game length: {_format_plies(total_truncated_length_sum / total_truncated_games)}"
            )
    wlog(f"Saved to {result_file_path}")
    return total_positions, total_games


def persistent_selfplay_worker(
    rank,
    config,
    device_id,
    task_queue,
    result_queue,
    inference_request_queue=None,
    inference_response_queue=None,
    inference_shared_buffer=None,
):
    """
    Persistent worker for self-play on Windows.

    The process is spawned once, then receives lightweight play tasks so we avoid
    repeated interpreter startup, imports, model construction and argument pickling.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)
    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Persistent worker started on {device}")

        central_inference_enabled = inference_request_queue is not None and inference_response_queue is not None
        _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
        central_debug_enabled = bool(
            debug_root_enabled and central_debug_cfg.get(
                'verbose',
                rl_cfg.get('self_play_central_inference_debug', False),
            )
        )
        central_stall_warning_s = float(
            central_debug_cfg.get(
                'stall_warning_s',
                _central_inference_option(config, 'stall_warning_s', 60.0),
            )
        )
        central_transport_dtype = str(
            _central_inference_option(config, 'transport_dtype', 'float16') or 'float16'
        )
        shared_inference_call_lock = threading.Lock()
        model = None if central_inference_enabled else _build_selfplay_worker_model(config, device)
        inference_model = (
            _RemoteInferenceModel(
                "learner",
                inference_request_queue,
                inference_response_queue,
                worker_rank=rank,
                timeout_s=float(_central_inference_option(config, 'timeout_s', 0.0)),
                stall_warning_s=central_stall_warning_s,
                debug_enabled=central_debug_enabled,
                transport_dtype=central_transport_dtype,
                shared_buffer=inference_shared_buffer,
                shared_call_lock=shared_inference_call_lock,
            )
            if central_inference_enabled
            else _maybe_compile_selfplay_model(
                model,
                config,
                device,
                rank,
                model_label="learner",
            )
        )
        opponent_model = None
        engine = None
        engine_signature = None

        while True:
            task = task_queue.get()
            if task is None or task.get('cmd') == 'stop':
                wlog("Stopping persistent worker")
                break

            result_file_path = task['result_file_path']
            try:
                runtime_overrides = dict(task.get('rl_runtime_overrides') or {})
                runtime_overrides['hard_start_positions'] = list(
                    task.get('hard_start_positions', []) or []
                )
                if runtime_overrides:
                    config.setdefault('reinforcement_learning', {}).update({
                        key: value
                        for key, value in runtime_overrides.items()
                        if value is not None
                    })
                    rl_cfg = config.get('reinforcement_learning', {})
                if central_inference_enabled and central_debug_enabled:
                    opponent_payload_preview = task.get('opponent_payload') or {}
                    preview_labels = sorted({
                        str((entry or {}).get('label') or 'current')
                        for entry in list(opponent_payload_preview.get('pool_entries', []) or [])
                    })
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Worker {rank}: central task started "
                        f"games={int(task['num_games'])}, opponent_models={preview_labels or ['current']}.",
                        flush=True,
                    )
                if not central_inference_enabled:
                    if task.get('model_state') is not None:
                        model_state = task['model_state']
                    else:
                        model_state = torch.load(task['model_state_path'], map_location='cpu')
                    _load_worker_model_state(model, model_state, rank)
                    model.eval()

                opponent_payload = task.get('opponent_payload') or {}
                opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
                opponent_pool_entries = list(opponent_payload.get('pool_entries', []) or [])
                opponent_models_by_label = {}
                opponent_model = None
                opponent_label = str(opponent_payload.get('label') or 'current')
                for entry in opponent_pool_entries:
                    entry_label = str((entry or {}).get('label') or 'current')
                    entry_state = (entry or {}).get('state')
                    if entry_label == 'current':
                        continue
                    if central_inference_enabled:
                        pooled_model = _RemoteInferenceModel(
                            entry_label,
                            inference_request_queue,
                            inference_response_queue,
                            worker_rank=rank,
                            timeout_s=float(_central_inference_option(config, 'timeout_s', 0.0)),
                            stall_warning_s=central_stall_warning_s,
                            debug_enabled=central_debug_enabled,
                            transport_dtype=central_transport_dtype,
                            shared_buffer=inference_shared_buffer,
                            shared_call_lock=shared_inference_call_lock,
                        )
                    else:
                        if entry_state is None:
                            continue
                        pooled_model = _build_selfplay_worker_model(config, device)
                        _load_worker_model_state(pooled_model, entry_state, rank)
                        pooled_model.eval()
                        pooled_model = _maybe_compile_selfplay_model(
                            pooled_model,
                            config,
                            device,
                            rank,
                            model_label=f"opponent:{entry_label}",
                        )
                    opponent_models_by_label[entry_label] = pooled_model
                    if opponent_model is None:
                        opponent_model = pooled_model
                    if opponent_label == 'current':
                        opponent_label = entry_label

                engine = _create_selfplay_engine(
                    inference_model,
                    config,
                    device,
                    int(task['num_games']),
                    wlog,
                    opponent_model=opponent_model,
                    opponent_source_label=opponent_label,
                    opponent_models_by_label=opponent_models_by_label,
                    opponent_plan_labels=opponent_plan_labels,
                )
                engine_signature = None

                if hasattr(engine, 'temperature') and task.get('mcts_temperature') is not None:
                    engine.temperature = float(task['mcts_temperature'])
                total_positions, total_games = _play_games_with_engine(
                    rank,
                    engine,
                    config,
                    int(task['num_games']),
                    result_file_path,
                    wlog,
                    result_queue=result_queue,
                    task_id=task.get('task_id'),
                    stream_results_to_queue=bool(task.get('stream_results_to_queue', False)),
                )
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task['task_id'],
                    'ok': True,
                    'positions': total_positions,
                    'games': total_games,
                })
            except KeyboardInterrupt:
                try:
                    with open(result_file_path, 'wb') as f:
                        pickle.dump(([], []), f)
                finally:
                    result_queue.put({
                        'type': 'result',
                        'rank': rank,
                        'task_id': task.get('task_id'),
                        'ok': False,
                        'interrupt': True,
                    })
                    raise SystemExit(130)
            except Exception as e:
                print(f"ERROR: Worker {rank} failed: {e}")
                import traceback
                traceback.print_exc()
                with open(result_file_path, 'wb') as f:
                    pickle.dump(([], []), f)
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task.get('task_id'),
                    'ok': False,
                    'error': str(e),
                })
    except KeyboardInterrupt:
        raise SystemExit(130)


def play_games_mcts_worker(
    rank,
    model_state,
    config,
    device_id,
    num_games,
    result_file_path,
    opponent_payload=None,
):
    """
    One-shot worker function for parallel MCTS self-play.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Starting on {device}")

        if isinstance(model_state, (str, bytes)):
            model_state = torch.load(model_state, map_location='cpu')

        model = _build_selfplay_worker_model(config, device)
        _load_worker_model_state(model, model_state, rank)
        model = _maybe_compile_selfplay_model(
            model,
            config,
            device,
            rank,
            model_label="learner",
        )
        opponent_model = None
        opponent_label = 'current'
        opponent_models_by_label = {}
        opponent_plan_labels = []
        if isinstance(opponent_payload, dict):
            opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
            for entry in list(opponent_payload.get('pool_entries', []) or []):
                entry_label = str((entry or {}).get('label') or 'current')
                entry_state = (entry or {}).get('state')
                if entry_state is None or entry_label == 'current':
                    continue
                pooled_model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(pooled_model, entry_state, rank)
                pooled_model.eval()
                pooled_model = _maybe_compile_selfplay_model(
                    pooled_model,
                    config,
                    device,
                    rank,
                    model_label=f"opponent:{entry_label}",
                )
                opponent_models_by_label[entry_label] = pooled_model
                if opponent_model is None:
                    opponent_model = pooled_model
                    opponent_label = entry_label

        engine = _create_selfplay_engine(
            model,
            config,
            device,
            num_games,
            wlog,
            opponent_model=opponent_model,
            opponent_source_label=opponent_label,
            opponent_models_by_label=opponent_models_by_label,
            opponent_plan_labels=opponent_plan_labels,
        )
        _play_games_with_engine(rank, engine, config, num_games, result_file_path, wlog)
    except KeyboardInterrupt:
        try:
            with open(result_file_path, 'wb') as f:
                pickle.dump(([], []), f)
        finally:
            raise SystemExit(130)
    except Exception as e:
        print(f"ERROR: Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        with open(result_file_path, 'wb') as f:
            pickle.dump(([], []), f)


from src.mcts_compat import BatchMCTS, MCTS, select_move_by_visits
