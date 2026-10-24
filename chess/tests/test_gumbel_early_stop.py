import numpy as np
import pytest

from src.mcts.search import (
    MultiGameBatchMCTS,
    _adaptive_gumbel_stop_decision,
    _adaptive_stop_checkpoint_evidence,
    _adaptive_stop_monitor_indices,
)


def evidence(**overrides):
    values = {
        "legacy_ready": True,
        "q_gap": 0.20,
        "target_budget": 192,
        "legacy_fresh_floor": 64,
        "fresh_visits": np.asarray([100, 10, 8, 0, 0], dtype=np.float64),
        "monitor_indices": (0, 1, 2),
        "new_fresh_finalist_evidence": True,
    }
    values.update(overrides)
    return values


def test_current_rule_matches_legacy_gap_boundary_and_nan_behavior():
    state = {}
    assert _adaptive_gumbel_stop_decision(evidence(q_gap=0.12), state, mode="current")
    assert not _adaptive_gumbel_stop_decision(evidence(q_gap=0.119), state, mode="current")
    assert not _adaptive_gumbel_stop_decision(
        evidence(q_gap=float("nan")), state, mode="current"
    )
    assert not _adaptive_gumbel_stop_decision(
        evidence(legacy_ready=False), state, mode="current"
    )


def test_disabled_vetoes_only_the_gumbel_early_stop_decision():
    assert not _adaptive_gumbel_stop_decision(
        evidence(), {"_adaptive_stop_evidence_checkpoints": 10}, mode="disabled"
    )


def test_cautious_rule_requires_budget_fraction_competitor_coverage_and_fresh_checkpoints():
    state = {"_adaptive_stop_evidence_checkpoints": 3}
    assert _adaptive_gumbel_stop_decision(evidence(), state, mode="cautious")
    assert not _adaptive_gumbel_stop_decision(
        evidence(fresh_visits=np.asarray([70, 10, 8, 0, 0])), state, mode="cautious"
    )
    assert not _adaptive_gumbel_stop_decision(
        evidence(fresh_visits=np.asarray([180, 4, 8, 0, 0])), state, mode="cautious"
    )
    assert not _adaptive_gumbel_stop_decision(
        evidence(new_fresh_finalist_evidence=False), state, mode="cautious"
    )
    assert not _adaptive_gumbel_stop_decision(
        evidence(q_gap=0.0), state, mode="cautious"
    )
    assert not _adaptive_gumbel_stop_decision(
        evidence(), {"_adaptive_stop_evidence_checkpoints": 2}, mode="cautious"
    )


def test_engine_default_uses_cautious_rule_and_inherited_visits_cannot_meet_fresh_floor():
    engine = MultiGameBatchMCTS.__new__(MultiGameBatchMCTS)
    low_fresh = evidence(fresh_visits=np.asarray([70, 10, 8, 0, 0]))
    state = {"_adaptive_stop_evidence_checkpoints": 3}
    assert _adaptive_gumbel_stop_decision(low_fresh, state, mode="current")
    assert not engine._adaptive_stop_decision(low_fresh, state)

    inherited_heavy = {
        "_adaptive_stop_evidence_checkpoints": 3,
        "inherited_edge_visits": np.asarray([10_000, 10_000, 10_000]),
    }
    assert not engine._adaptive_stop_decision(low_fresh, inherited_heavy)


def test_monitor_set_always_includes_winner_challenger_and_active_rivals_only():
    scores = np.asarray([1.0, 0.9, 0.8, 0.7, -1.0, -2.0])
    final_counts = np.asarray([20, 18, 16, 14, 1, 0], dtype=np.float64)
    monitors = _adaptive_stop_monitor_indices(
        scores, final_counts, winner=0, challenger=5, limit=4
    )
    assert 0 in monitors
    assert 5 in monitors
    assert set(monitors) == {0, 1, 2, 5}
    assert len(monitors) <= 4


def test_checkpoint_count_needs_new_visits_and_stable_winner_challenger_finalists():
    state = {}
    first = _adaptive_stop_checkpoint_evidence(
        np.asarray([5, 4, 2]), (0, 1), winner=0, challenger=1, state=state
    )
    assert first["has_new_evidence"]
    assert first["stable_checkpoints"] == 1

    state.update({
        "_adaptive_stop_checkpoint_fresh_visits": first["fresh_visits_snapshot"],
        "_adaptive_stop_finalist_signature": first["finalist_signature"],
        "_adaptive_stop_evidence_checkpoints": first["stable_checkpoints"],
    })
    second = _adaptive_stop_checkpoint_evidence(
        np.asarray([6, 5, 2]), (0, 1), winner=0, challenger=1, state=state
    )
    assert second["has_new_evidence"]
    assert second["stable_checkpoints"] == 2

    state.update({
        "_adaptive_stop_checkpoint_fresh_visits": second["fresh_visits_snapshot"],
        "_adaptive_stop_finalist_signature": second["finalist_signature"],
        "_adaptive_stop_evidence_checkpoints": second["stable_checkpoints"],
    })
    stale = _adaptive_stop_checkpoint_evidence(
        np.asarray([6, 5, 2]), (0, 1), winner=0, challenger=1, state=state
    )
    assert not stale["has_new_evidence"]
    assert stale["stable_checkpoints"] == 0

    changed = _adaptive_stop_checkpoint_evidence(
        np.asarray([7, 5, 3]), (0, 2), winner=0, challenger=2, state=state
    )
    assert changed["has_new_evidence"]
    assert changed["finalist_signature"] != second["finalist_signature"]
    assert changed["stable_checkpoints"] == 1


def test_unknown_stop_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown adaptive Gumbel stop mode"):
        _adaptive_gumbel_stop_decision(evidence(), {}, mode="bad")
