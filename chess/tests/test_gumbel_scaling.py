import math

import pytest
import numpy as np

from src.config.normalization import normalize_rl_config
from src.mcts.native import get_native_mcts
from src.mcts.gumbel_scaling import effective_q_scale_visits, gumbel_q_scale
from src.mcts.search import MCTSNode


@pytest.mark.parametrize("visits", [0, 64, 192, 256, 1536])
def test_exponent_one_preserves_legacy_linear_scale(visits):
    assert gumbel_q_scale(100.0, 0.10, visits, 256.0, 1.0) == pytest.approx(
        (100.0 + visits) * 0.10
    )


def test_sublinear_scale_preserves_low_visits_and_softens_high_visits():
    assert effective_q_scale_visits(192, 256, 0.5) == pytest.approx(192)
    assert effective_q_scale_visits(256, 256, 0.5) == pytest.approx(256)
    assert effective_q_scale_visits(1536, 256, 0.5) == pytest.approx(256 * math.sqrt(6))
    assert gumbel_q_scale(100, 0.1, 1536, 256, 0.5) < gumbel_q_scale(
        100, 0.1, 1536, 256, 1.0
    )


def test_zero_exponent_caps_visit_contribution_after_knee():
    assert effective_q_scale_visits(1536, 256, 0.0) == pytest.approx(256)


def test_config_normalization_accepts_soft_curve_and_rejects_superlinear_exponent():
    config = {"reinforcement_learning": {"search": {
        "gumbel_q_scale_visit_knee": 256,
        "gumbel_q_scale_visit_exponent": 0.5,
    }}}
    normalize_rl_config(config)
    rl = config["reinforcement_learning"]
    assert rl["mcts_gumbel_q_scale_visit_knee"] == 256
    assert rl["mcts_gumbel_q_scale_visit_exponent"] == 0.5

    invalid = {"reinforcement_learning": {"search": {
        "gumbel_q_scale_visit_exponent": 1.01,
    }}}
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        normalize_rl_config(invalid)


def test_native_stateless_root_and_final_paths_use_soft_visit_scale():
    native = get_native_mcts()
    if native is None:
        pytest.skip("optional native MCTS compiler/library unavailable")

    log_priors = np.asarray([0.0, -75.0], dtype=np.float64)
    gumbel = np.zeros(2, dtype=np.float64)
    completed_q = np.asarray([0.0, 1.0], dtype=np.float64)
    initial = np.zeros(2, dtype=np.float64)
    pass_initial = np.zeros(2, dtype=np.float64)
    visits = np.full(2, 1536, dtype=np.int32)
    total_counts = visits.astype(np.float32)
    virtual_losses = np.zeros(2, dtype=np.int16)
    params = (100.0, 0.10, 256.0, 0.5)

    selected = native.select_root(
        log_priors, gumbel, completed_q, initial, pass_initial, visits,
        total_counts, virtual_losses, 1536, *params, False, True,
    )
    assert selected == 0
    assert native.final_action(
        log_priors, gumbel, completed_q, initial, visits, *params, True,
    ) == 0
    assert native.final_action_eligible(
        log_priors, gumbel, completed_q, initial, pass_initial, visits,
        1536, *params, True,
    ) == 0

    priors = np.asarray([1.0, math.exp(-75.0)], dtype=np.float32)
    policy = np.empty(2, dtype=np.float32)
    native.improved_policy(
        log_priors, priors, completed_q, 1536, *params, policy,
    )
    scale = gumbel_q_scale(*params[:2], 1536, *params[2:])
    logits = log_priors + completed_q * scale
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    assert np.allclose(policy, expected, atol=1e-6)


def test_native_forest_root_selection_uses_soft_visit_scale():
    native = get_native_mcts()
    if native is None:
        pytest.skip("optional native MCTS compiler/library unavailable")

    root = MCTSNode()
    root.expanded = True
    root.raw_value = 0.0
    root.edges.reset((0, 1), np.asarray([1.0, math.exp(-75.0)], dtype=np.float32))
    root.edges.visit_counts[:] = 1536
    root.edges.value_sums[:] = np.asarray([0.0, -1536.0], dtype=np.float32)
    root.edges.total_counts[:] = 1536
    root.edges.total_count_sum = 3072.0
    forest = native.create_forest(
        [root], c_visit=100.0, c_scale=0.10,
        q_scale_visit_knee=256.0, q_scale_visit_exponent=0.5,
        q_range_floor=0.25, use_mixed_value=True,
    )
    try:
        forest.configure_root_states([{
            "initial_visits": np.zeros(2, dtype=np.int32),
            "pass_initial_visits": np.zeros(2, dtype=np.int32),
            "gumbel": np.zeros(2, dtype=np.float64),
            "sequence": np.asarray([1536], dtype=np.int32),
            "q_scale_total_visits": True,
        }])
        leaves, _, _ = forest.select_batch([0])
        assert leaves[0].move == 0
    finally:
        forest.close()
