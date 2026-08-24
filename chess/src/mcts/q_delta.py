"""Shared histogram contract for compact MCTS Q-delta telemetry."""

import math

import numpy as np


Q_DELTA_HIST_MIN = -2.0
Q_DELTA_HIST_MAX = 2.0
Q_DELTA_HIST_BINS = 200
USEFUL_SEARCH_Q_DELTA_MIN = 0.02
# A correction is safe to force as the policy winner only when the improved
# Gumbel target has both enough mass on its best move and a material gap over
# the runner-up.  Keep this contract shared by self-play, replay audits and the
# learner so telemetry and training never disagree about a "good target".
RELIABLE_POLICY_TARGET_TOP1_MIN = 0.55
RELIABLE_POLICY_TARGET_GAP_MIN = 0.12


def q_delta_histogram(values):
    if not values:
        return [0] * Q_DELTA_HIST_BINS
    array = np.asarray(values, dtype=np.float32)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [0] * Q_DELTA_HIST_BINS
    array = np.clip(array, Q_DELTA_HIST_MIN, Q_DELTA_HIST_MAX)
    histogram, _ = np.histogram(
        array,
        bins=Q_DELTA_HIST_BINS,
        range=(Q_DELTA_HIST_MIN, Q_DELTA_HIST_MAX),
    )
    return [int(value) for value in histogram.tolist()]


def q_delta_percentile_from_histogram(histogram, percentile):
    counts = np.asarray(histogram or [], dtype=np.float64)
    if counts.size == 0 or float(counts.sum()) <= 0.0:
        return 0.0
    threshold = max(1.0, math.ceil((float(percentile) / 100.0) * float(counts.sum())))
    index = int(np.searchsorted(np.cumsum(counts), threshold, side="left"))
    index = max(0, min(index, counts.size - 1))
    width = (Q_DELTA_HIST_MAX - Q_DELTA_HIST_MIN) / float(counts.size)
    bin_low = float(Q_DELTA_HIST_MIN + index * width)
    bin_high = float(bin_low + width)
    if bin_low <= 0.0 <= bin_high:
        return 0.0
    return float(bin_low + 0.5 * width)
