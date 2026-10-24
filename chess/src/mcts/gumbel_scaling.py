"""Shared visit-dependent Q scaling for Gumbel MCTS."""

from __future__ import annotations


def effective_q_scale_visits(max_visits: float, knee: float, exponent: float) -> float:
    """Compress visits above ``knee`` with a sublinear power curve.

    With exponent 1 this is exactly the historical linear visit count. With
    exponent 0 the count plateaus at the knee. Values between them grow
    sublinearly after the knee while leaving lower visit budgets unchanged.
    Config validation is responsible for keeping knee positive and exponent
    in [0, 1].
    """
    visits = max(0.0, float(max_visits))
    knee = float(knee)
    exponent = float(exponent)
    if visits <= knee or exponent >= 1.0:
        return visits
    if exponent <= 0.0:
        return knee
    return knee * (visits / knee) ** exponent


def gumbel_q_scale(
    c_visit: float,
    c_scale: float,
    max_visits: float,
    knee: float,
    exponent: float,
) -> float:
    """Return the Q multiplier, preserving the legacy schedule at exponent 1."""
    return (
        float(c_visit)
        + effective_q_scale_visits(max_visits, knee, exponent)
    ) * float(c_scale)
