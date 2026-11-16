"""Policy target weighting from completed MCTS root-search work."""

import math


def teacher_policy_weight_from_effective_simulations(
    effective_simulations,
    *,
    nominal_simulations,
    min_budget_ratio,
):
    """Map completed effective root visits to a policy weight.

    ``effective_simulations`` must be the completed ``simulations_used`` count
    (fresh visits plus any credited inherited visits), never the assigned
    ``simulation_budget``. ``None`` means the count is unavailable or invalid;
    callers can then preserve an existing replay weight. Below the configured
    threshold, policy learning is masked. Otherwise the uncapped weight is the
    effective count divided by the nominal budget.
    """
    if effective_simulations is None or isinstance(effective_simulations, bool):
        return None
    try:
        actual = float(effective_simulations)
        nominal = int(nominal_simulations)
        threshold = float(min_budget_ratio)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(actual)
        or actual < 0.0
        or not actual.is_integer()
        or nominal <= 0
        or not math.isfinite(threshold)
    ):
        return None
    threshold = min(1.0, max(0.0, threshold))
    actual_count = int(actual)
    if actual_count / float(nominal) < threshold:
        return 0.0
    return actual_count / float(nominal)
