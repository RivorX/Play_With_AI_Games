"""Compact, decision-oriented dashboards for RL training CSV logs.

The three public renderers deliberately split responsibilities:

* main: strength, learning and promotion safety;
* data quality: replay, targets and MCTS behaviour;
* performance: throughput, latency and bottlenecks.

Keeping the plots here prevents the CSV logger from becoming a second plotting
application. The MCTS move-correction rate is the one intentional repeat: it is
both a headline training signal and a detailed search-quality diagnostic.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter

from src.common.dashboard_layout import create_dashboard, finish_dashboard
from src.common.logger_helpers import (
    _CSV_RESUME_METADATA_KEY,
    _metadata_json_payload,
    _parse_mcts_elo_by_simulations,
    _read_csv_rows_preserving_metadata,
)


_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def _rows(path):
    if path is None or not Path(path).exists():
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as handle:
            raw = [[cell.replace("\x00", "") for cell in row] for row in csv.reader(handle)]
    except OSError:
        return []
    while raw and (not raw[0] or str(raw[0][0]).strip().startswith("#")):
        raw.pop(0)
    if not raw:
        return []
    header = raw[0]
    result = []
    for values in raw[1:]:
        values = values + [""] * max(0, len(header) - len(values))
        result.append(dict(zip(header, values[: len(header)])))
    return result


def _rl_config(path):
    """Read the normalized RL config embedded in the CSV metadata row."""
    if path is None or not Path(path).exists():
        return {}
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                if str(row[0]).strip() == "# config_json" and len(row) > 1:
                    payload = json.loads(str(row[1]).replace("\x00", ""))
                    return dict(payload.get("reinforcement_learning", {}) or {})
                if not str(row[0]).strip().startswith("#"):
                    break
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


def _resume_markers(path):
    """Read resume boundaries embedded in the canonical main RL CSV."""
    if path is None or not Path(path).exists():
        return []
    try:
        metadata_rows, _ = _read_csv_rows_preserving_metadata(path)
    except (OSError, csv.Error):
        return []
    payload = _metadata_json_payload(
        metadata_rows,
        _CSV_RESUME_METADATA_KEY,
    ) or {}
    markers = []
    for marker in payload.get("markers") or []:
        if not isinstance(marker, dict):
            continue
        x = _number(marker.get("x"))
        if x is None:
            raw_completed = marker.get(
                "completed_iteration",
                marker.get("completed_epoch"),
            )
            completed = _number(raw_completed)
            if completed is None:
                continue
            x = completed + 0.5
        markers.append(
            {
                "x": float(x),
                "label": str(marker.get("label") or "resume"),
            }
        )
    return sorted(markers, key=lambda item: item["x"])


def _mark_resume_boundaries(axes, markers):
    """Draw the same red continuation boundary on every time-series panel."""
    if not markers:
        return
    flattened = (
        list(axes.values())
        if isinstance(axes, dict)
        else np.asarray(axes, dtype=object).reshape(-1)
    )
    for axis_index, ax in enumerate(flattened):
        if ax is None:
            continue
        for marker in markers:
            ax.axvline(
                marker["x"],
                color="#dc2626",
                linestyle="--",
                linewidth=1.35,
                alpha=0.90,
                zorder=8,
            )
            if axis_index == 0:
                ax.annotate(
                    marker["label"],
                    xy=(marker["x"], 1.0),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(3, -3),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color="#b91c1c",
                    fontsize=7,
                    rotation=90,
                )


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _series(rows, column):
    xs, ys = [], []
    for row in rows:
        x = _number(row.get("iteration"))
        y = _number(row.get(column))
        if x is not None and y is not None:
            xs.append(int(x))
            ys.append(y)
    return xs, ys


def _rl_reanalysis_plot_rows(rows):
    """Add plotted ratios only where their logged denominators are observed."""
    plotted = []
    for source in rows:
        row = dict(source)
        updated = _number(row.get("reanalyse_updated"))
        correction_fraction = _number(row.get("reanalyse_correction_fraction"))
        row["_reanalyse_correction_fraction_observed"] = (
            correction_fraction if updated is not None and updated > 0.0 else None
        )

        selected = _number(row.get("reanalyse_selected"))
        eligible = _number(row.get("reanalyse_eligible_rows"))
        row["_reanalyse_selected_eligible_fraction"] = (
            selected / eligible
            if selected is not None and eligible is not None and eligible > 0.0
            else None
        )
        plotted.append(row)
    return plotted


def _rl_reanalysis_summary(rows):
    """Summarize per-iteration reanalysis records without pooling repeated rows."""
    ordered = sorted(
        (row for row in rows if _number(row.get("iteration")) is not None),
        key=lambda row: _number(row.get("iteration")),
    )
    if not ordered:
        return {}

    def value(row, key):
        return _number(row.get(key))

    active_rows = [
        row for row in ordered
        if (value(row, "reanalyse_selected") or 0.0) > 0.0
        or (value(row, "reanalyse_updated") or 0.0) > 0.0
    ]
    selected_iterations = [
        int(value(row, "iteration"))
        for row in ordered
        if value(row, "reanalyse_selected") is not None
        and value(row, "reanalyse_selected") > 0.0
    ]
    intervals = [right - left for left, right in zip(selected_iterations, selected_iterations[1:])]
    cadence = intervals[0] if intervals and all(step == intervals[0] for step in intervals) else None
    latest = ordered[-1]
    last_active = active_rows[-1] if active_rows else None
    return {
        "latest_iteration": int(value(latest, "iteration")),
        "latest_selected": value(latest, "reanalyse_selected"),
        "latest_updated": value(latest, "reanalyse_updated"),
        "last_active_iteration": int(value(last_active, "iteration")) if last_active else None,
        "last_active_selected": value(last_active, "reanalyse_selected") if last_active else None,
        "last_active_updated": value(last_active, "reanalyse_updated") if last_active else None,
        "selected_cadence": cadence,
        "last_active_skipped": value(last_active, "reanalyse_skipped") if last_active else None,
        "last_active_eligible": value(last_active, "reanalyse_eligible_rows") if last_active else None,
        "last_active_age": value(last_active, "reanalyse_selected_target_age_mean") if last_active else None,
        "last_active_actor_kl": value(last_active, "reanalyse_selected_actor_kl_mean") if last_active else None,
        "last_active_actor_kl_coverage": value(last_active, "reanalyse_selected_actor_kl_coverage") if last_active else None,
        "last_active_sims_mean": value(last_active, "reanalyse_simulations_mean") if last_active else None,
        "last_active_sims_min": value(last_active, "reanalyse_simulations_min") if last_active else None,
        "last_active_sims_max": value(last_active, "reanalyse_simulations_max") if last_active else None,
    }


def _rl_reanalysis_count_text(value):
    return "n/a" if value is None else f"{value:,.0f}"


def _rl_reanalysis_activity_note(summary, historical_note=None):
    if not summary:
        lines = ["No reanalysis iteration records."]
    else:
        latest = (
            f"Latest i{summary['latest_iteration']}: "
            f"{_rl_reanalysis_count_text(summary['latest_selected'])} selected, "
            f"{_rl_reanalysis_count_text(summary['latest_updated'])} updated."
        )
        if summary["last_active_iteration"] is None:
            activity = "No active reanalysis iteration recorded."
        else:
            activity = (
                f"Last active i{summary['last_active_iteration']}: "
                f"{_rl_reanalysis_count_text(summary['last_active_selected'])}/"
                f"{_rl_reanalysis_count_text(summary['last_active_updated'])} selected/updated."
            )
        cadence = (
            f"Selected every {summary['selected_cadence']} iterations (observed)."
            if summary["selected_cadence"] is not None
            else "Selection cadence unavailable or irregular."
        )
        age = (
            "n/a" if summary["last_active_age"] is None
            else f"{summary['last_active_age']:.2f}"
        )
        actor_kl = (
            "n/a" if summary["last_active_actor_kl"] is None
            else f"{summary['last_active_actor_kl']:.4f}"
        )
        actor_kl_coverage = (
            "n/a" if summary["last_active_actor_kl_coverage"] is None
            else f"{summary['last_active_actor_kl_coverage']:.0%}"
        )
        detail = (
            "Last active: skipped "
            f"{_rl_reanalysis_count_text(summary['last_active_skipped'])}, eligible "
            f"{_rl_reanalysis_count_text(summary['last_active_eligible'])}, age {age}, "
            f"actor KL {actor_kl} ({actor_kl_coverage} coverage)."
        )
        budget_values = (
            summary["last_active_sims_mean"],
            summary["last_active_sims_min"],
            summary["last_active_sims_max"],
        )
        if all(value is None for value in budget_values):
            budget = "Effective reanalysis simulations per updated target: not logged."
        else:
            mean = "n/a" if budget_values[0] is None else f"{budget_values[0]:.1f}"
            low = "n/a" if budget_values[1] is None else f"{budget_values[1]:.0f}"
            high = "n/a" if budget_values[2] is None else f"{budget_values[2]:.0f}"
            budget = f"Effective sims per updated target: mean {mean}, range {low}–{high}."
        lines = [latest, f"{activity} {cadence}", detail, budget]
    if historical_note:
        lines.append(str(historical_note))
    return "\n".join(lines)


def _draw_rl_reanalysis_activity(ax, rows, *, historical_note=None):
    """Draw replay target refreshes and denominator-explicit rates."""
    plotted = _rl_reanalysis_plot_rows(rows)
    _style(ax, "Replay reanalysis activity (replay positions; not fresh self-play)")
    shown = False
    count_values = []
    for column, label, color, linestyle in (
        ("reanalyse_updated", "updated replay targets", _COLORS[2], "-"),
        ("reanalyse_selected", "selected replay positions", _COLORS[0], "--"),
        ("reanalyse_skipped", "skipped selected positions", _COLORS[1], ":"),
    ):
        shown |= _line(ax, plotted, column, label, color, ls=linestyle)
        _, values = _series(plotted, column)
        count_values.extend(values)
    ax.set_ylabel("replay positions / iteration", fontsize=7.5)
    ax.set_xlabel("training iteration")
    ax.set_ylim(bottom=0.0)
    if count_values:
        ax.set_ylim(top=max(1.0, max(count_values) * 1.65))

    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    correction = _line(
        ax2,
        plotted,
        "_reanalyse_correction_fraction_observed",
        "search corrections vs current NN / updated",
        _COLORS[4],
        ls="--",
    )
    selected_share = _line(
        ax2,
        plotted,
        "_reanalyse_selected_eligible_fraction",
        "selected / eligible rows",
        _COLORS[5],
        ls=":",
    )
    ax2.set_ylabel("fraction (each line uses its labeled denominator)", fontsize=7.0)
    ax2.set_ylim(0.0, 1.0)
    _legend_below(ax, ax2, ncol=2)

    summary = _rl_reanalysis_summary(plotted)
    note = _rl_reanalysis_activity_note(summary, historical_note)
    ax.text(
        0.015,
        0.985,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color="#4b5563",
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )
    if not (shown or correction or selected_share):
        _no_data(ax, "Replay reanalysis telemetry unavailable")
    return shown or correction or selected_share


def _draw_rl_reanalysis_budget(ax, rows):
    """Draw actual simulations consumed per successfully updated replay target."""
    _style(ax, "Effective simulations per successfully updated target")
    shown = False
    for column, label, color, linestyle in (
        ("reanalyse_simulations_mean", "mean", _COLORS[0], "-"),
        ("reanalyse_simulations_min", "min", _COLORS[2], "--"),
        ("reanalyse_simulations_max", "max", _COLORS[4], "--"),
    ):
        shown |= _line(ax, rows, column, label, color, ls=linestyle)
    ax.set_ylabel("effective simulations / updated position", fontsize=7.5)
    ax.set_xlabel("training iteration")
    ax.set_ylim(bottom=0.0)
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax, "Actual reanalysis simulations per update were not logged")
    return shown


def _count_rate_data(rows, count_column, duration_column):
    """Return measured counts per positive wall-second plus pooled totals."""
    xs, rates = [], []
    total_count = total_duration = 0.0
    for row in rows:
        iteration = _number(row.get("iteration"))
        count = _number(row.get(count_column))
        duration = _number(row.get(duration_column))
        if (
            iteration is None or count is None or count < 0.0
            or duration is None or duration <= 0.0
        ):
            continue
        xs.append(int(iteration))
        rates.append(count / duration)
        total_count += count
        total_duration += duration
    return xs, rates, total_count, total_duration


def _weighted_rate_summary(rows, rate_column, duration_column):
    """Pool an already measured rate using its observed wall-time denominator."""
    weighted_count = total_duration = 0.0
    for row in rows:
        rate = _number(row.get(rate_column))
        duration = _number(row.get(duration_column))
        if rate is None or rate < 0.0 or duration is None or duration <= 0.0:
            continue
        weighted_count += rate * duration
        total_duration += duration
    return weighted_count / total_duration if total_duration > 0.0 else None


def _regime_change_iterations(rows):
    """Return promoted-best regime boundaries."""
    promotions = set()
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        iteration = int(iteration)
        if (_number(row.get("rl_best_model")) or 0.0) > 0.0:
            promotions.add(iteration)
    return sorted(promotions), []


def _selfplay_best_intervals(rows):
    """Return exact iteration ranges generated by the frozen best actor."""
    events = []
    ordered_rows = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        ordered_rows.append((int(iteration), row))
    ordered_rows.sort(key=lambda item: item[0])
    rows_by_iteration = {iteration: row for iteration, row in ordered_rows}

    for iteration, row in ordered_rows:
        actor = str(row.get("selfplay_actor") or "").strip().lower()
        explicit_flag = _number(row.get("selfplay_best_vs_best"))
        has_explicit_source = actor in {"best", "learner"} or explicit_flag is not None
        if has_explicit_source:
            is_best = actor == "best" or bool(explicit_flag is not None and explicit_flag > 0.0)
            if not is_best:
                continue
            replay_rebuild = False
            safety_fallback = False
        else:
            # Schema <= 50 did not persist the chosen actor. Reconstruct only
            # from causal signals: champion rows added in this iteration, and
            # the previous iteration's post-eval guard state (the next
            # iteration's pre-self-play state). Never use rolling source shares.
            replay_rebuild = (_number(row.get("champion_replay_added")) or 0.0) > 0.0
            previous_row = rows_by_iteration.get(iteration - 1)
            previous_safe = (
                _number(previous_row.get("learner_actor_safe"))
                if previous_row is not None
                else None
            )
            safety_fallback = previous_safe is not None and previous_safe <= 0.0
            if not (replay_rebuild or safety_fallback):
                continue

        reason = str(row.get("selfplay_best_reason") or "other").strip().lower()
        if not has_explicit_source:
            legacy_reasons = []
            if safety_fallback:
                legacy_reasons.append("safety_fallback")
            if replay_rebuild:
                legacy_reasons.append("replay_rebuild")
            reason = "+".join(legacy_reasons)
        # Refilling the pinned champion reservoir is routine bookkeeping after
        # a promotion, not a learner safety regime.  Do not turn those one-off
        # iterations into visual regime changes.  Old logs called this
        # ``replay_rebuild``; new logs use the accurate ``champion_refresh``.
        if reason in {"replay_rebuild", "champion_refresh"}:
            continue
        if reason in {
            "safety_fallback+replay_rebuild",
            "safety_fallback+champion_refresh",
        }:
            reason = "safety_fallback"
        if reason != "safety_fallback":
            reason = "other"
        events.append((iteration, reason))

    intervals = []
    for iteration, reason in sorted(set(events)):
        if intervals and intervals[-1][2] == reason and iteration == intervals[-1][1] + 1:
            start, _, interval_reason = intervals[-1]
            intervals[-1] = (start, iteration, interval_reason)
        else:
            intervals.append((iteration, iteration, reason))
    return intervals


def _split_series_at_boundaries(xs, ys, break_after):
    """Split a line after each event so unlike regimes are never connected."""
    boundaries = sorted({int(value) for value in (break_after or ())})
    if not xs or not boundaries:
        return [(list(xs), list(ys))] if xs else []
    segments = []
    segment_x, segment_y = [], []
    for x, y in zip(xs, ys):
        if segment_x and any(segment_x[-1] <= boundary < x for boundary in boundaries):
            segments.append((segment_x, segment_y))
            segment_x, segment_y = [], []
        segment_x.append(x)
        segment_y.append(y)
    if segment_x:
        segments.append((segment_x, segment_y))
    return segments


def _last(rows, column, default=None):
    for row in reversed(rows):
        value = _number(row.get(column))
        if value is not None:
            return value
    return default


def _relative_holdout_gain_pct(row, before_column, reduction_column):
    """Express a held-out error reduction as a percent of its before value."""
    before = _number(row.get(before_column))
    reduction = _number(row.get(reduction_column))
    if before is None or reduction is None or before <= 0.0:
        return None
    return 100.0 * reduction / before


def _ratio_if_positive(row, numerator_column, denominator_column):
    numerator = _number(row.get(numerator_column))
    denominator = _number(row.get(denominator_column))
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _rl_reuse_accounting_summary(rows):
    """Pool reuse and simulation accounting using each metric's real sample count."""
    provenance_rows = 0.0
    effective_sum = 0.0
    fresh_sum = 0.0
    credited_sum = 0.0
    provenance_coverage_available = True
    candidate_rows = 0.0
    provenance_mismatches = 0.0
    provenance_partial = 0.0
    provenance_metrics_seen = False
    provenance_incomplete_rows = 0

    hit_count = 0.0
    attempt_count = 0.0
    unknown_attempt_rows = 0
    hit_metrics_seen = False
    hit_rates = []
    inherited_visits_sum = 0.0
    inherited_visits_hits = 0.0
    hit_metrics_incomplete = False
    raw_visits_incomplete = False

    for row in rows:
        row_count = _number(row.get("replay_search_budget_provenance_rows"))
        row_effective = _number(row.get("replay_search_simulations_effective_sum"))
        row_fresh = _number(row.get("replay_search_simulations_fresh_sum"))
        row_credited = _number(row.get("replay_search_visits_inherited_credit_sum"))
        if "replay_search_budget_provenance_rows" in row:
            provenance_metrics_seen = True
        if row_count is not None and row_count > 0.0:
            provenance_metrics_seen = True
            if all(
                value is not None and value >= 0.0
                for value in (row_effective, row_fresh, row_credited)
            ):
                provenance_rows += row_count
                effective_sum += row_effective
                fresh_sum += row_fresh
                credited_sum += row_credited
                coverage = _number(row.get("replay_search_budget_provenance_coverage"))
                if coverage is not None and 0.0 < coverage <= 1.0:
                    candidate_rows += row_count / coverage
                else:
                    provenance_coverage_available = False
            else:
                provenance_coverage_available = False
                provenance_incomplete_rows += 1
        elif "replay_search_budget_provenance_rows" in row and row_count is None:
            provenance_coverage_available = False
            provenance_incomplete_rows += 1

        partial = _number(row.get("replay_search_budget_provenance_partial_rows"))
        mismatch = _number(row.get("replay_search_budget_accounting_mismatch_rows"))
        if partial is not None:
            provenance_metrics_seen = True
            provenance_partial += max(0.0, partial)
        if mismatch is not None:
            provenance_metrics_seen = True
            provenance_mismatches += max(0.0, mismatch)

        telemetry_complete = _number(row.get("selfplay_telemetry_complete"))
        if telemetry_complete is not None and telemetry_complete < 1.0:
            hit_metrics_incomplete = True
            raw_visits_incomplete = True
            continue
        hit_rate = _number(row.get("mcts_tree_reuse_hit_rate"))
        reused_roots = _number(row.get("mcts_tree_reuse_credit_samples"))
        raw_visits = _number(row.get("mcts_tree_inherited_visits_avg"))
        if (
            "mcts_tree_reuse_hit_rate" in row
            or "mcts_tree_reuse_credit_samples" in row
        ) and (hit_rate is None or reused_roots is None):
            hit_metrics_incomplete = True
        if reused_roots is not None and reused_roots > 0.0 and raw_visits is None:
            raw_visits_incomplete = True
        if hit_rate is not None and reused_roots is not None and 0.0 <= hit_rate <= 1.0 and reused_roots >= 0.0:
            hit_metrics_seen = True
            hit_rates.append(hit_rate)
            hit_count += reused_roots
            if hit_rate > 0.0 and reused_roots > 0.0:
                attempts = reused_roots / hit_rate
                rounded_attempts = round(attempts)
                if math.isclose(attempts, rounded_attempts, rel_tol=1e-7, abs_tol=1e-5):
                    attempt_count += float(rounded_attempts)
                else:
                    unknown_attempt_rows += 1
            elif hit_rate == 0.0 and reused_roots == 0.0:
                # The log stores hits and the ratio, but not attempts. With zero
                # hits the denominator cannot be recovered for a mixed-rate pool.
                unknown_attempt_rows += 1
            else:
                unknown_attempt_rows += 1
            if reused_roots > 0.0 and raw_visits is not None and raw_visits >= 0.0:
                inherited_visits_sum += raw_visits * reused_roots
                inherited_visits_hits += reused_roots

    if provenance_rows > 0.0:
        effective_per_row = effective_sum / provenance_rows
        fresh_per_row = fresh_sum / provenance_rows
        credited_per_row = credited_sum / provenance_rows
        composition_matches = math.isclose(
            effective_sum, fresh_sum + credited_sum, rel_tol=1e-7, abs_tol=1e-5
        ) and provenance_mismatches == 0.0
        extra_pct = (
            100.0 * credited_sum / fresh_sum
            if composition_matches and fresh_sum > 0.0 else None
        )
        coverage = (
            min(1.0, provenance_rows / candidate_rows)
            if provenance_coverage_available and candidate_rows > 0.0 else None
        )
    else:
        effective_per_row = fresh_per_row = credited_per_row = extra_pct = None
        composition_matches = None
        coverage = None

    if hit_metrics_seen and unknown_attempt_rows == 0 and attempt_count > 0.0:
        reuse_hit_rate = hit_count / attempt_count
        reuse_attempts = attempt_count
    elif hit_metrics_seen and hit_rates and all(rate == hit_rates[0] for rate in hit_rates):
        # A constant recorded rate has the same weighted mean for any positive
        # denominator; its count still cannot be reported when hits are zero.
        reuse_hit_rate = hit_rates[0]
        reuse_attempts = attempt_count if attempt_count > 0.0 else None
    else:
        reuse_hit_rate = None
        reuse_attempts = None

    return {
        "provenance_rows": int(provenance_rows),
        "provenance_available": provenance_metrics_seen,
        "provenance_coverage": coverage,
        "provenance_partial_rows": int(provenance_partial),
        "provenance_mismatch_rows": int(provenance_mismatches),
        "provenance_incomplete_rows": provenance_incomplete_rows,
        "effective_per_row": effective_per_row,
        "fresh_per_row": fresh_per_row,
        "credited_per_row": credited_per_row,
        "extra_credit_pct_of_fresh": extra_pct,
        "composition_matches": composition_matches,
        "reuse_hit_rate": reuse_hit_rate if not hit_metrics_incomplete else None,
        "reuse_hits": int(hit_count) if hit_metrics_seen else None,
        "reuse_attempts": int(reuse_attempts) if reuse_attempts is not None else None,
        "reuse_unknown_attempt_rows": unknown_attempt_rows,
        "raw_inherited_visits_per_hit": (
            inherited_visits_sum / inherited_visits_hits
            if inherited_visits_hits > 0.0 and not raw_visits_incomplete else None
        ),
        "raw_inherited_visit_hits": int(inherited_visits_hits),
    }


def _selfplay_search_simulation_parts(row):
    """Return per-root fresh simulations and reuse credit when accounting agrees."""
    telemetry_complete = _number(row.get("selfplay_telemetry_complete"))
    if telemetry_complete is not None and telemetry_complete < 1.0:
        return None
    fresh = _number(row.get("mcts_fresh_sims"))
    credited = _number(row.get("mcts_inherited_visit_credit"))
    if fresh is None or credited is None or fresh < 0.0 or credited < 0.0:
        return None
    effective = _number(row.get("mcts_avg_sims"))
    if effective is not None and not math.isclose(
        effective, fresh + credited, rel_tol=1e-6, abs_tol=1e-4
    ):
        return None
    return fresh, credited


def _selfplay_inherited_visit_parts(row):
    """Return fresh and raw inherited visits per all search roots when denominators are known.

    The raw inherited-visit average is conditional on a reused root. Convert it
    to an all-search-root average with the reuse hit count and search-sample
    count; hit rate is conditional on reuse attempts and is not that fraction.
    """
    telemetry_complete = _number(row.get("selfplay_telemetry_complete"))
    if telemetry_complete is not None and telemetry_complete < 1.0:
        return None

    fresh = _number(row.get("mcts_fresh_sims"))
    search_samples = _number(row.get("mcts_search_samples"))
    if search_samples is None:
        # Older rows did not persist search_samples. Their generated-position
        # count is equivalent only when every search is a current self-play
        # move that produces a history position, with no post-search exits.
        legacy_fields = (
            "played_positions",
            "opponent_current_actual_share",
            "opponent_best_actual_share",
            "selfplay_resigned_games",
            "selfplay_adjudicated_games",
        )
        values = {key: _number(row.get(key)) for key in legacy_fields}
        if (
            telemetry_complete is None
            or telemetry_complete < 1.0
            or any(value is None for value in values.values())
            or not math.isclose(values["opponent_current_actual_share"], 1.0)
            or not math.isclose(values["opponent_best_actual_share"], 0.0)
            or values["selfplay_resigned_games"] != 0.0
            or values["selfplay_adjudicated_games"] != 0.0
        ):
            return None
        search_samples = values["played_positions"]

    hits = _number(row.get("mcts_tree_reuse_credit_samples"))
    raw_per_hit = _number(row.get("mcts_tree_inherited_visits_avg"))
    if (
        fresh is None
        or fresh < 0.0
        or search_samples is None
        or search_samples <= 0.0
        or hits is None
        or hits < 0.0
        or hits > search_samples
    ):
        return None
    if hits == 0.0:
        inherited_per_root = 0.0
    elif raw_per_hit is None or raw_per_hit < 0.0:
        return None
    else:
        inherited_per_root = raw_per_hit * hits / search_samples
    return fresh, inherited_per_root


def _rl_reuse_summary_subtitle(summary, *, reuse_q_scale=None):
    """Format a compact, denominator-explicit reuse summary for the dashboard."""
    provenance_rows = summary["provenance_rows"]
    if provenance_rows > 0 and summary["composition_matches"]:
        coverage = summary["provenance_coverage"]
        coverage_text = (
            f" ({coverage:.0%} coverage)" if coverage is not None else " (coverage unknown)"
        )
        provenance_line = (
            f"Replay provenance weighted over {provenance_rows:,} target rows{coverage_text}: "
            f"{summary['effective_per_row']:.2f} effective sims/root = "
            f"{summary['fresh_per_row']:.2f} fresh + {summary['credited_per_row']:.2f} credited "
            f"({summary['extra_credit_pct_of_fresh']:.2f}% over fresh)."
        )
        if summary["provenance_incomplete_rows"]:
            provenance_line += (
                f" {summary['provenance_incomplete_rows']:,} iteration row(s) had missing provenance."
            )
    elif provenance_rows > 0:
        provenance_line = (
            f"Replay search provenance covers {provenance_rows:,} rows, but effective/fresh/credited "
            "totals do not reconcile; pooled additional-budget percentage is unavailable."
        )
    else:
        provenance_line = (
            "Replay effective/fresh/credited averages are n/a because complete search provenance is unavailable."
        )
    if summary["provenance_partial_rows"] or summary["provenance_mismatch_rows"]:
        provenance_line += (
            f" Logged partial={summary['provenance_partial_rows']:,}, "
            f"accounting mismatch={summary['provenance_mismatch_rows']:,}."
        )

    hit_rate = summary["reuse_hit_rate"]
    raw_visits = summary["raw_inherited_visits_per_hit"]
    if hit_rate is None:
        reuse_line = (
            "Self-play reuse hit-rate average is n/a because the logged hit/attempt counters cannot be pooled completely."
        )
    elif summary["reuse_attempts"] is not None:
        reuse_line = (
            f"Self-play reuse hit rate among attempted roots: {hit_rate:.1%} "
            f"({summary['reuse_hits']:,}/{summary['reuse_attempts']:,}; not all search roots);"
        )
    else:
        reuse_line = (
            f"Self-play reuse hit rate: {hit_rate:.1%} (attempt-weighting unavailable);"
        )
    if raw_visits is None:
        reuse_line += " raw visits per successful reuse: n/a."
    else:
        reuse_line += (
            f" {raw_visits:.2f} raw prior visits per hit, weighted over "
            f"{summary['raw_inherited_visit_hits']:,} hits;"
        )
        if reuse_q_scale == "total":
            reuse_line += " they inform total-visit Q scaling, not fresh compute or credited sims."
        else:
            reuse_line += " they are not fresh compute or credited sims."
    return [provenance_line, reuse_line]


def _mean(rows, column, window=5):
    values = [_number(row.get(column)) for row in rows[-window:]]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _style(ax, title, percent=False):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.tick_params(labelsize=8)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _line(
    ax,
    rows,
    column,
    label,
    color=None,
    *,
    ls="-",
    lw=1.8,
    alpha=1.0,
    break_after=None,
):
    xs, ys = _series(rows, column)
    if not xs:
        return False
    for segment_index, (segment_x, segment_y) in enumerate(
        _split_series_at_boundaries(xs, ys, break_after)
    ):
        ax.plot(
            segment_x,
            segment_y,
            label=label if segment_index == 0 else None,
            color=color,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            marker="o",
            markersize=3.5,
        )
    return True


def _value_search_alignment_panel(ax, rows, *, break_after=None):
    """Show whether the production value scalar agrees with searched root-Q."""
    _style(ax, "NN value vs searched root-Q")
    shown = _line(
        ax, rows, "value_root_q_mae", "root-Q MAE", _COLORS[0],
        lw=2.0, break_after=break_after,
    )
    shown |= _line(
        ax, rows, "value_root_q_bias", "signed bias", _COLORS[3],
        ls="--", lw=1.8, break_after=break_after,
    )
    ax.axhline(0.0, color="#374151", lw=1.0, ls=":", label="zero bias")
    ax.set_ylabel("value error", fontsize=7.5, color=_COLORS[0])
    ax.tick_params(axis="y", labelcolor=_COLORS[0])

    ax2 = ax.twinx()
    ax2.tick_params(axis="y", labelcolor=_COLORS[2], labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    correlation = _line(
        ax2, rows, "value_root_q_correlation", "correlation", _COLORS[2],
        lw=1.8, break_after=break_after,
    )
    ax2.set_ylabel("correlation", fontsize=7.5, color=_COLORS[2])
    latest_mae = _last(rows, "value_root_q_mae")
    latest_bias = _last(rows, "value_root_q_bias")
    latest_correlation = _last(rows, "value_root_q_correlation")
    latest = []
    if latest_mae is not None:
        latest.append(f"MAE {latest_mae:.3f}")
    if latest_bias is not None:
        latest.append(f"bias {latest_bias:+.3f}")
    if latest_correlation is not None:
        latest.append(f"corr {latest_correlation:.1%}")
    if latest:
        ax.text(
            0.02, 0.96, "latest: " + " | ".join(latest),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            color="#4b5563",
        )
    _legend_below(ax, ax2, ncol=2)
    if not (shown or correlation):
        _no_data(ax)


def _promotion_strength_panel(ax, rows, rl_cfg):
    """Show promotion evidence without combining distinct evaluation protocols."""
    _style(ax, "Promotion evidence", percent=True)
    selector_score_min = float(rl_cfg.get("score_rate_threshold", 0.55) or 0.55)
    selector_lb_min = float(
        rl_cfg.get("promotion_score_lower_bound_min", 0.50) or 0.50
    )
    confirmation_score_min = float(
        rl_cfg.get("promotion_confirmation_score_rate_min", 0.53) or 0.53
    )
    raw_score_min = float(
        rl_cfg.get("promotion_no_mcts_score_rate_min", 0.48) or 0.48
    )
    confirmation_lb_min = float(
        rl_cfg.get("promotion_confirmation_score_lower_bound_min", 0.50) or 0.50
    )
    advanced_games = max(
        1, int(_number(rl_cfg.get("eval_funnel_advanced_games")) or 192)
    )

    # Keep thresholds visible as subtle guides; their concise values are
    # summarized below the legend instead of becoming extra legend entries.
    gate_color = "#6b7280"
    for value, linestyle in (
        (selector_score_min, ":"),
        (confirmation_score_min, "--"),
        (raw_score_min, ":"),
    ):
        ax.axhline(value, color=gate_color, lw=0.8, ls=linestyle, alpha=0.48,
                   zorder=0)
    for value in sorted({selector_lb_min, confirmation_lb_min}):
        ax.axhline(value, color="#374151", lw=0.9, ls="-.", alpha=0.62,
                   zorder=0)

    selector, raw_nn, holdout = [], [], []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue

        score = _number(row.get("score_rate"))
        count = int(_number(row.get("eval_games")) or 0)
        stage = str(row.get("eval_stage") or "").strip().lower()
        is_full = stage == "funnel:advanced" and count >= advanced_games
        if score is not None:
            candidate_status = str(
                row.get("mcts_candidate_status") or row.get("promotion_status") or ""
            ).strip().lower()
            selector.append({
                "x": int(iteration), "score": score,
                "lower": _number(row.get("eval_score_lower_bound")),
                "upper": _number(row.get("eval_score_upper_bound")),
                "games": count, "full": is_full, "status": candidate_status,
                "decision_status": str(row.get("promotion_status") or "").strip().lower(),
            })

        # Missing means unmeasured; never coerce raw NN evidence to zero.
        raw_score = _number(row.get("no_mcts_score_rate"))
        if raw_score is not None:
            raw_nn.append({
                "x": int(iteration), "score": raw_score,
                "lower": _number(row.get("no_mcts_score_lower_bound")),
                "upper": _number(row.get("no_mcts_score_upper_bound")),
                "games": int(_number(row.get("no_mcts_games")) or 0),
                "status": str(row.get("promotion_raw_nn_status") or "").strip().lower(),
            })

        holdout_score = _number(row.get("promotion_confirmation_score_rate"))
        if holdout_score is not None:
            holdout.append({
                "x": int(iteration), "score": holdout_score,
                "lower": _number(row.get("promotion_confirmation_score_lower_bound")),
                "upper": _number(row.get("promotion_confirmation_score_upper_bound")),
                "games": int(_number(row.get("promotion_confirmation_games")) or 0),
                "status": str(row.get("promotion_confirmation_status") or "").strip().lower(),
            })

    if not selector and not raw_nn and not holdout:
        _no_data(ax, "Promotion funnel telemetry is unavailable in this CSV")
        return False

    def draw_evidence(items, *, offset, color, label, marker):
        """Draw one protocol, preserving sparse stages and logged intervals."""
        # Do not draw a continuous trend across iterations where that protocol
        # was not run. In particular, holdout evidence is intentionally sparse.
        segment = []
        for item in sorted(items, key=lambda value: value["x"]):
            if segment and item["x"] != segment[-1]["x"] + 1:
                if len(segment) > 1:
                    ax.plot(
                        [point["x"] + offset for point in segment],
                        [point["score"] for point in segment],
                        color=color, lw=1.0, alpha=0.35, zorder=1,
                    )
                segment = []
            segment.append(item)
        if len(segment) > 1:
            ax.plot(
                [point["x"] + offset for point in segment],
                [point["score"] for point in segment],
                color=color, lw=1.0, alpha=0.35, zorder=1,
            )
        for item in items:
            x = item["x"] + offset
            score = item["score"]
            lower, upper = item["lower"], item["upper"]
            if lower is not None or upper is not None:
                lower_error = max(0.0, score - lower) if lower is not None else 0.0
                upper_error = max(0.0, upper - score) if upper is not None else 0.0
                ax.errorbar(
                    x, score, yerr=np.array([[lower_error], [upper_error]]),
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=2.2,
                    alpha=0.78, zorder=3,
                )

            candidate_state = item.get("candidate_status", "")
            # Only the persisted candidate state means promotion. A passing
            # holdout is evidence, not itself a promotion decision.
            promoted = candidate_state == "promoted" or candidate_state.startswith("promoted:")
            archived = candidate_state == "archived"
            point_marker = "*" if promoted else ("h" if archived else item.get("marker", marker))
            face = color if item.get("full", True) or promoted or archived else "white"
            edge = "#15803d" if promoted else ("#a16207" if archived else color)
            ax.scatter(
                [x], [score], marker=point_marker,
                s=68 if promoted else (48 if archived else 33),
                facecolors=face, edgecolors=edge, linewidths=1.0, zorder=6,
                label=label,
            )

    for item in selector:
        item["candidate_status"] = item["status"]
        item["marker"] = "o" if item["full"] else "^"
    draw_evidence(selector, offset=-0.14, color=_COLORS[0],
                  label="selector MCTS", marker="o")
    draw_evidence(raw_nn, offset=0.0, color=_COLORS[3],
                  label="raw NN safety", marker="s")
    draw_evidence(holdout, offset=0.14, color="#c2410c",
                  label="independent confirmation", marker="D")

    ax.axhline(0.5, color="#374151", lw=0.8, ls="--", alpha=0.7)
    # Keep the historical framing when all results fit, but expand it for
    # weak/partial evaluations instead of silently clipping them (for example,
    # a 29% 48-game preliminary result). Include interval endpoints and gates
    # so both evidence and decision thresholds remain visible.
    evidence = [item["score"] for item in selector + raw_nn + holdout]
    evidence.extend(
        endpoint
        for item in selector + raw_nn + holdout
        for endpoint in (item["lower"], item["upper"])
        if endpoint is not None
    )
    evidence.extend([
        0.5, selector_score_min, confirmation_score_min, raw_score_min,
        selector_lb_min, confirmation_lb_min,
    ])
    evidence = [value for value in evidence if value is not None and math.isfinite(value)]
    data_min, data_max = min(evidence), max(evidence)
    span = data_max - data_min
    padding = max(0.025, span * 0.12)
    y_min = max(0.0, min(0.35, data_min - padding))
    y_max = min(1.0, max(0.78, data_max + padding))
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("match points per game", fontsize=8)
    ax.set_xlabel("iteration", fontsize=7)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    marker_handles = []
    if any(item["candidate_status"] == "archived" for item in selector):
        marker_handles.append(("h", "#a16207", "archived"))
    if any(item["candidate_status"] == "promoted" or
           str(item["candidate_status"]).startswith("promoted:") for item in selector):
        marker_handles.append(("*", "#15803d", "promoted"))
    from matplotlib.lines import Line2D
    for symbol, color, label in marker_handles:
        unique[label] = Line2D([], [], linestyle="none", marker=symbol,
                               markerfacecolor=color, markeredgecolor=color,
                               markersize=7)
    ax.legend(
        list(unique.values()), list(unique.keys()), fontsize=6.3, frameon=False,
        ncol=max(1, min(3, len(unique))), loc="upper center",
        bbox_to_anchor=(0.5, -0.22), borderaxespad=0.0,
        columnspacing=0.9, handlelength=1.4,
    )
    ax.text(
        0.5, -0.39,
        f"Gates: selector {selector_score_min:.0%} / LB {selector_lb_min:.0%}"
        f"  ·  raw NN {raw_score_min:.0%}"
        f"  ·  holdout {confirmation_score_min:.0%} / LB {confirmation_lb_min:.0%}",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
        color="#4b5563", clip_on=False,
    )
    return True


def _fill_between_with_breaks(ax, xs, lower, upper, *, break_after=None, **kwargs):
    offset = 0
    for segment_x, segment_lower in _split_series_at_boundaries(xs, lower, break_after):
        stop = offset + len(segment_x)
        ax.fill_between(segment_x, segment_lower, upper[offset:stop], **kwargs)
        offset = stop


def _stacked_iteration_bars(
    ax,
    rows,
    specs,
    value_resolver,
    *,
    title,
    ylabel,
    legend_columns=3,
    seamless=False,
):
    """Draw a compact stacked decomposition with one bar per iteration."""
    iterations = []
    series = {key: [] for key, _, _ in specs}
    totals = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        resolved_values = value_resolver(row)
        if resolved_values is None:
            continue
        resolved = dict(resolved_values)
        values = {
            key: max(0.0, _number(resolved.get(key)) or 0.0)
            for key, _, _ in specs
        }
        iterations.append(int(iteration))
        totals.append(sum(values.values()))
        for key in series:
            series[key].append(values[key])

    _style(ax, title)
    if not iterations or not any(totals):
        _no_data(ax)
        return False

    bottoms = [0.0] * len(iterations)
    for key, label, color in specs:
        values = series[key]
        if not any(values):
            continue
        ax.bar(
            iterations,
            values,
            bottom=bottoms,
            width=1.0 if seamless else 0.76,
            color=color,
            label=label,
            edgecolor="none" if seamless else "white",
            linewidth=0.0 if seamless else 0.3,
            antialiased=not seamless,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    if seamless and len(iterations) > 1:
        boundaries = [
            0.5 * (left + right)
            for left, right in zip(iterations[:-1], iterations[1:])
        ]
        boundary_heights = [
            max(left, right)
            for left, right in zip(totals[:-1], totals[1:])
        ]
        ax.vlines(
            boundaries,
            0.0,
            boundary_heights,
            colors="#111827",
            linewidth=0.22,
            alpha=0.72,
            zorder=2,
        )

    if len(iterations) <= 12:
        padding = max(totals) * 0.015
        for iteration, total in zip(iterations, totals):
            ax.text(iteration, total + padding, f"{total:.1f}", ha="center", va="bottom", fontsize=6.2, color="#374151")
    ax.set_ylim(0.0, max(totals) * 1.12)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=6.2,
        frameon=False,
        ncol=max(1, int(legend_columns)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    return True


def _composition_donut(
    ax,
    averaged,
    *,
    title,
    center_label,
    unit,
    decimals=1,
    min_share=0.005,
    min_value=0.0,
    autopct_min=4.0,
):
    """Draw a filtered decomposition donut with percentage and absolute cost."""
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    raw_total = sum(max(0.0, value) for _, _, value, _ in averaged)
    visible = [
        (key, label, value, color)
        for key, label, value, color in averaged
        if value > 0.0
        and (value >= float(min_value) or (raw_total > 0.0 and value / raw_total >= float(min_share)))
    ]
    visible_total = sum(value for _, _, value, _ in visible)
    if visible_total <= 0.0:
        _no_data(ax)
        return False

    values = [value for _, _, value, _ in visible]
    colors = [color for _, _, _, color in visible]

    def _autopct(percent):
        value = visible_total * percent / 100.0
        whole_percent = 100.0 * value / raw_total
        if whole_percent < float(autopct_min):
            return ""
        return f"{whole_percent:.1f}%\n{value:.{decimals}f}{unit}"

    ax.grid(False)
    wedges, _, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.46, "edgecolor": "white", "linewidth": 1.0},
        autopct=_autopct,
        pctdistance=0.76,
        textprops={"fontsize": 7, "fontweight": "bold", "color": "#111827"},
    )
    ax.text(
        0,
        0,
        f"{center_label}\n{raw_total:.{decimals}f}{unit}",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#374151",
    )
    legend_labels = [
        f"{label} | {100.0 * value / raw_total:.1f}% | {value:.{decimals}f}{unit}"
        for _, label, value, _ in visible
    ]
    ax.legend(
        wedges,
        legend_labels,
        fontsize=6.2,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
    )
    ax.set_aspect("equal")
    return True


_RUNTIME_STAGE_SPECS = (
    ("selfplay", "self-play", "#2563eb"),
    ("replay_setup", "setup/replay", "#0891b2"),
    ("train", "training", "#16a34a"),
    ("evaluation", "evaluation", "#f59e0b"),
    ("elo_eval", "Elo eval", "#ea580c"),
    ("maintenance", "maintenance", "#7c3aed"),
    ("additional", "other phases", "#64748b"),
    ("other", "unaccounted", "#cbd5e1"),
)


def _runtime_composition_data(rows):
    """Group exclusive iteration stages and retain only a measured residual."""
    iterations, totals = [], []
    stage_values = {column: [] for column, _, _ in _RUNTIME_STAGE_SPECS}
    other_values = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        logged_total = _number(row.get("iteration_total_time_s"))
        raw_stages = {
            str(column): _number(value)
            for column, value in row.items()
            if str(column).startswith("stage_") and str(column).endswith("_time_s")
        }
        # Blank stage cells mean unknown, not zero. The residual is available
        # only when the row has a logged total and every stage cell is numeric.
        stage_complete = bool(raw_stages) and all(value is not None for value in raw_stages.values())
        numeric = {key: max(0.0, value) for key, value in raw_stages.items() if value is not None}
        def group(columns):
            present = [column for column in columns if column in raw_stages]
            if not present or any(raw_stages[column] is None for column in present):
                return None
            return sum(numeric.get(column, 0.0) for column in present)

        values = {
            "selfplay": group(("stage_selfplay_time_s",)),
            "replay_setup": group((
                "stage_setup_time_s", "stage_replay_time_s", "stage_reanalyse_time_s",
            )),
            "train": group(("stage_train_time_s",)),
            "evaluation": group((
                "stage_regular_eval_time_s", "stage_promotion_eval_time_s",
                "stage_promotion_confirmation_time_s", "stage_eval_log_time_s",
            )),
            "elo_eval": group(("stage_elo_eval_time_s",)),
            "maintenance": group((
                "stage_log_time_s", "stage_checkpoint_time_s", "stage_gc_time_s",
            )),
        }
        grouped_columns = {
            "stage_selfplay_time_s", "stage_setup_time_s", "stage_replay_time_s",
            "stage_reanalyse_time_s", "stage_train_time_s", "stage_regular_eval_time_s",
            "stage_promotion_eval_time_s", "stage_promotion_confirmation_time_s",
            "stage_elo_eval_time_s", "stage_eval_log_time_s", "stage_log_time_s",
            "stage_checkpoint_time_s", "stage_gc_time_s",
        }
        additional_columns = [
            key for key in raw_stages if key not in grouped_columns
        ]
        values["additional"] = (
            sum(numeric.get(key, 0.0) for key in additional_columns)
            if all(raw_stages[key] is not None for key in additional_columns)
            else None
        )
        measured_total = sum(value for value in values.values() if value is not None)
        residual = None
        if logged_total is not None and stage_complete:
            delta = logged_total - measured_total
            # Stage timers are intended to be exclusive; tiny clock/rounding
            # error is treated as zero. A negative residual is an accounting
            # mismatch, never an invented negative/zero-duration stage.
            residual = max(0.0, delta) if delta >= -0.01 else None
        values["other"] = residual
        iterations.append(int(iteration))
        totals.append(max(0.0, logged_total) if logged_total is not None else None)
        for column in stage_values:
            stage_values[column].append(values.get(column))
        other_values.append(residual)
    return iterations, totals, stage_values, other_values


def _donut_note(ax, message):
    ax.text(
        0.5, -0.20, message,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=6.2, color="#536273", wrap=True,
    )


def _overall_runtime_donut(ax, rows):
    """Show summed, complete iteration wall-time from exclusive stage timers."""
    _, totals, stage_values, other_values = _runtime_composition_data(rows)
    complete_indices = [
        index for index, total in enumerate(totals)
        if total is not None and other_values[index] is not None
    ]
    if not complete_indices:
        ax.set_title("Overall iteration time by phase", fontsize=10, fontweight="bold", loc="left")
        _no_data(ax, "Complete total and stage timers unavailable")
        return False

    averaged = []
    measured_zero = []
    for column, label, color in _RUNTIME_STAGE_SPECS:
        values = [stage_values[column][index] for index in complete_indices]
        if values and all(value is not None for value in values):
            total = sum(values)
            if total > 0.0:
                averaged.append((column, label, total, color))
            else:
                measured_zero.append(label)
    _composition_donut(
        ax,
        averaged,
        title="Overall iteration time by phase",
        center_label="complete logged\niterations",
        unit="s",
        decimals=1,
        min_share=0.005,
        min_value=0.5,
    )
    note = "Stage wall timers are exclusive; unaccounted is only the measured residual."
    if measured_zero:
        note += " Logged at 0s: " + ", ".join(measured_zero) + "."
    _donut_note(ax, note)
    return True


def _selfplay_worker_search_donut(ax, rows):
    """Partition worker search time into batch expansion and the search remainder."""
    search = expansion = 0.0
    samples = 0
    for row in rows:
        total = _number(row.get("mcts_search_many_time_s"))
        expanded = _number(row.get("mcts_batch_expand_eval_time_s"))
        if total is None or expanded is None or total < 0.0 or expanded < 0.0:
            continue
        if expanded > total:
            continue
        search += total
        expansion += min(total, expanded)
        samples += 1
    averaged = []
    if samples and search > 0.0:
        averaged = [
            ("expansion", "batch expand/eval", expansion, "#db2777"),
            ("other_search", "other search", max(0.0, search - expansion), "#2563eb"),
        ]
    _composition_donut(
        ax,
        averaged,
        title="Self-play search breakdown",
        center_label="summed worker\nsearch seconds",
        unit="s",
        decimals=0,
        min_share=0.0,
        min_value=0.0,
    )
    if averaged:
        _donut_note(ax, f"{samples} iterations. Worker-seconds sum across workers and can exceed wall time.")
    return bool(averaged)


def _cpu_worker_search_partition(rows):
    """Return a non-overlapping worker-search partition outside measured NN wait."""
    totals = {"selection": 0.0, "backprop": 0.0, "remainder": 0.0}
    samples = 0
    for row in rows:
        wait = _number(row.get("mcts_nn_inference_time_s"))
        search = _number(row.get("mcts_search_many_time_s"))
        selection = _number(row.get("mcts_search_selection_time_s"))
        backprop = _number(row.get("mcts_search_backprop_time_s"))
        values = (search, wait, selection, backprop)
        if any(value is None or value < 0.0 for value in values):
            continue
        known = wait + selection + backprop
        if known > search:
            continue
        totals["selection"] += selection
        totals["backprop"] += backprop
        totals["remainder"] += max(0.0, search - known)
        samples += 1
    return totals, samples


def _complete_exclusive_stage_rows(rows):
    """Return rows with a complete stage-timer vector and its exact costs.

    Stage counters are exclusive iteration wall timers.  A blank cell is
    unknown, so rows with any missing stage timer are excluded from additive
    stacks and ranked totals rather than silently treating it as zero.
    """
    stage_columns = sorted({
        str(column)
        for row in rows
        for column in row
        if str(column).startswith("stage_") and str(column).endswith("_time_s")
    })
    complete = []
    if not stage_columns:
        return stage_columns, complete
    for row in rows:
        iteration = _number(row.get("iteration"))
        total = _number(row.get("iteration_total_time_s"))
        values = {column: _number(row.get(column)) for column in stage_columns}
        if iteration is None or total is None or any(value is None or value < 0 for value in values.values()):
            continue
        stage_sum = sum(values.values())
        residual = total - stage_sum
        if residual < -0.01:
            continue
        values["unaccounted"] = max(0.0, residual)
        complete.append((int(iteration), total, values))
    return stage_columns, complete


def _exclusive_stage_labels(column):
    return column.removeprefix("stage_").removesuffix("_time_s").replace("_", " ")


_OVERLAPPING_WORKER_TIMERS = (
    ("mcts_search_many_time_s", "search worker", "#2563eb"),
    ("mcts_nn_inference_time_s", "NN wait", "#db2777"),
    ("mcts_search_root_setup_time_s", "root setup", "#0891b2"),
    ("mcts_search_selection_time_s", "tree selection", "#16a34a"),
    ("mcts_search_backprop_time_s", "backpropagation", "#7c3aed"),
    ("mcts_board_to_tensor_time_s", "board encoding", "#ea580c"),
    ("mcts_batch_expand_tensor_pack_time_s", "input tensor packing", "#06b6d4"),
    ("mcts_batch_expand_cpu_policy_time_s", "CPU policy", "#dc2626"),
    ("mcts_batch_expand_move_index_time_s", "move indexing", "#a16207"),
    ("mcts_policy_target_build_time_s", "policy target build", "#64748b"),
)


def _cpu_path_timer_donut(ax, rows):
    """Show exclusive worker search timers without claiming process CPU time."""
    totals, samples = _cpu_worker_search_partition(rows)
    cpu_seconds = sum(totals.values())
    counters = [
        (key, label, value, color)
        for key, label, color, value in (
            ("selection", "tree selection", "#2563eb", totals["selection"]),
            ("backprop", "backpropagation", "#16a34a", totals["backprop"]),
            ("remainder", "other outside NN wait", "#64748b", totals["remainder"]),
        )
        if value > 0.0
    ] if samples and cpu_seconds > 0.0 else []
    _composition_donut(
        ax,
        counters,
        title="CPU-side search worker time",
        center_label="search outside\nNN wait",
        unit="s",
        decimals=0,
        min_share=0.005,
        min_value=0.5,
        autopct_min=0.0,
    )
    if counters:
        _donut_note(
            ax,
        f"{samples} iterations: search − NN wait = selection + backprop + remainder. Worker wall-time, not process CPU time/utilization.",
        )
    return bool(counters)


def _gpu_path_timer_donut(ax, rows):
    """Show latest host-timed central inference phases; no device utilization is inferred."""
    columns = (
        ("central_server_h2d_ms_per_request", "H2D", "#0891b2"),
        ("central_server_forward_ms_per_request", "forward", "#7c3aed"),
        ("central_server_d2h_ms_per_request", "D2H", "#f59e0b"),
    )
    latest = None
    latest_iteration = None
    for row in reversed(rows):
        values = [_number(row.get(column)) for column, _, _ in columns]
        if all(value is not None and value >= 0.0 for value in values):
            latest = values
            latest_iteration = _number(row.get("iteration"))
            break
    averaged = []
    if latest:
        averaged = [
            (column, label, value, color)
            for (column, label, color), value in zip(columns, latest)
        ]
    _composition_donut(
        ax,
        averaged,
        title="GPU inference path",
        center_label="latest\nms / request",
        unit="ms",
        decimals=2,
        min_share=0.0,
        min_value=0.0,
    )
    if averaged:
        iteration_label = "n/a" if latest_iteration is None else f"{latest_iteration:g}"
        _donut_note(ax, f"Iteration {iteration_label}: host-timed phases; no device-utilization counter.")
    return bool(averaged)


def _legend_below(ax, *extra_axes, ncol=2, y=-0.17):
    """Keep legends outside the data rectangle and merge twin-axis entries."""
    handles, labels = ax.get_legend_handles_labels()
    for extra_ax in extra_axes:
        extra_handles, extra_labels = extra_ax.get_legend_handles_labels()
        handles += extra_handles
        labels += extra_labels
    if not handles:
        return
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        list(unique.values()),
        list(unique.keys()),
        fontsize=6.8,
        frameon=False,
        ncol=max(1, min(int(ncol), len(unique))),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        columnspacing=1.1,
        handlelength=2.0,
    )


def _no_data(ax, message="Brak danych w CSV"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="#6b7280")
    ax.set_xticks([])
    ax.set_yticks([])


def _finish(fig, output_path, title, subtitle=None, *, hspace=0.48, wspace=0.30):
    if getattr(fig, "_training_dashboard_layout", False):
        return finish_dashboard(fig, output_path, title, subtitle)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.04, ha="left", y=0.988)
    plot_top = 0.91
    if subtitle:
        # bbox_inches="tight" includes unconstrained fig.text in the output
        # bounds. A long single line therefore widened the entire PNG instead
        # of staying inside the 18-inch dashboard. Wrap explicitly and reserve
        # vertical room for any extra lines.
        subtitle_width = max(72, int(fig.get_figwidth() * 9.0))
        wrapped_subtitle = textwrap.fill(
            str(subtitle),
            width=subtitle_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        subtitle_lines = wrapped_subtitle.count("\n") + 1
        fig.text(
            0.04,
            0.958,
            wrapped_subtitle,
            fontsize=8.5,
            color="#4b5563",
            ha="left",
            va="top",
            linespacing=1.25,
        )
        plot_top -= 0.018 * max(0, subtitle_lines - 1)
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=max(0.84, plot_top),
        hspace=hspace,
        wspace=wspace,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fmt(value, kind="number"):
    if value is None:
        return "n/a"
    if kind == "percent":
        return f"{100.0 * value:.1f}%"
    if kind == "integer":
        return f"{value:,.0f}"
    if kind == "seconds":
        return f"{value:.1f}s"
    return f"{value:.3f}"


def _mark_regime_changes(
    ax,
    promotion_iterations,
    recovery_iterations=(),
    *,
    label=False,
):
    # Matplotlib event artists participate in autoscaling.  A sparse metric
    # (notably anchor/Elo) must keep the range implied by its measurements,
    # even when a later promotion/recovery event exists in another panel.
    xlim = ax.get_xlim()
    for index, promotion_iteration in enumerate(promotion_iterations):
        ax.axvline(
            promotion_iteration,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.42,
            label="promotion" if label and index == 0 else None,
        )
    for index, recovery_iteration in enumerate(recovery_iterations):
        ax.axvline(
            recovery_iteration,
            color="#ea580c",
            linestyle="-.",
            linewidth=1.15,
            alpha=0.58,
            label="recovery" if label and index == 0 else None,
        )
    ax.set_xlim(xlim)


def _mark_selfplay_best_intervals(ax, intervals, *, label=False):
    """Shade iterations whose games were generated best vs the same best."""
    styles = {
        "safety_fallback": ("#f59e0b", "best x best - safety fallback"),
        "other": ("#7c3aed", "best x best"),
    }
    xlim = ax.get_xlim()
    labelled_reasons = set()
    for start, end, reason in intervals:
        color, legend_label = styles.get(reason, styles["other"])
        show_label = bool(label and reason not in labelled_reasons)
        ax.axvspan(
            float(start) - 0.45,
            float(end) + 0.45,
            color=color,
            alpha=0.14,
            linewidth=0.0,
            zorder=0.2,
            label=legend_label if show_label else None,
        )
        if show_label:
            labelled_reasons.add(reason)
    ax.set_xlim(xlim)


def _search_correction_rate_panel(
    ax,
    rows,
    *,
    break_after=(),
    promotion_iterations=(),
    recovery_iterations=(),
):
    """Show how often MCTS overrides the NN top move and produces a useful change."""
    _style(ax, "How often MCTS changes the NN move", percent=True)
    shown = _line(
        ax, rows, "mcts_prior_changed_rate", "changed top", _COLORS[0], lw=2.1,
        break_after=break_after,
    )
    shown |= _line(
        ax, rows, "mcts_useful_change_rate", "useful change", _COLORS[2], lw=1.9,
        break_after=break_after,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=2)
    if not shown:
        _no_data(ax)
    return shown


def _latest_search_decision_donut(ax, rows):
    """Summarize the latest MCTS top-move decision as a true partition."""
    changed = _last(rows, "mcts_prior_changed_rate")
    useful = _last(rows, "mcts_useful_change_rate")
    harmful = _last(rows, "mcts_harmful_change_rate")
    ax.set_title("Latest MCTS move decisions", fontsize=10, fontweight="bold", loc="left")
    if changed is None:
        _no_data(ax)
        return False

    changed = max(0.0, min(1.0, changed))
    useful = max(0.0, min(changed, useful or 0.0))
    harmful = max(0.0, min(changed - useful, harmful or 0.0))
    neutral = max(0.0, changed - useful - harmful)
    slices = [
        ("NN move kept", 1.0 - changed, "#94a3b8"),
        ("useful change", useful, _COLORS[2]),
        ("neutral / unknown change", neutral, _COLORS[0]),
        ("harmful change", harmful, _COLORS[1]),
    ]
    visible = [(label, value, color) for label, value, color in slices if value > 1e-9]
    values = [value for _, value, _ in visible]
    wedges, _, _ = ax.pie(
        values,
        colors=[color for _, _, color in visible],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.46, "edgecolor": "white", "linewidth": 1.0},
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4.0 else "",
        pctdistance=0.76,
        textprops={"fontsize": 7, "fontweight": "bold", "color": "#111827"},
    )
    ax.text(
        0,
        0,
        f"{100.0 * changed:.1f}%\nchanged",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#374151",
    )
    ax.legend(
        wedges,
        [f"{label} | {100.0 * value:.1f}%" for label, value, _ in visible],
        fontsize=6.4,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
    )
    ax.grid(False)
    ax.set_aspect("equal")
    return True


def _objective_composition_panel(ax, rows, rl_cfg):
    """Show the positive, weighted terms that actually form the train objective."""
    policy_weight = float(rl_cfg.get("policy_loss_weight", 1.0) or 0.0)
    default_value_weight = float(rl_cfg.get("value_loss_weight", 1.0) or 0.0)
    scalar_aux_weight = float(
        rl_cfg.get("value_scalar_aux_loss_weight", 0.0) or 0.0
    )
    consistency_weight = float(
        rl_cfg.get("value_search_consistency_loss_weight", 0.0) or 0.0
    )
    moves_weight = float(rl_cfg.get("moves_left_loss_weight", 0.05) or 0.0)
    default_search_q_weight = float(
        rl_cfg.get("search_q_loss_weight", 0.0) or 0.0
    )

    def _values(row):
        value_weight = _number(row.get("value_loss_weight"))
        value_weight = default_value_weight if value_weight is None else value_weight
        search_q_weight = _number(row.get("search_q_weight"))
        search_q_weight = (
            default_search_q_weight if search_q_weight is None else search_q_weight
        )
        scalar_aux_loss = max(
            0.0, _number(row.get("value_scalar_aux_loss")) or 0.0
        )
        value_primary_loss = _number(row.get("value_primary_loss"))
        if value_primary_loss is None:
            combined_value_loss = max(0.0, _number(row.get("value_loss")) or 0.0)
            value_primary_loss = max(
                0.0,
                combined_value_loss - scalar_aux_weight * scalar_aux_loss,
            )
        values = {
            "policy_ce": policy_weight * max(
                0.0, _number(row.get("policy_loss")) or 0.0
            ),
            "policy_rank": 0.0,
            "wdl": value_weight * max(0.0, value_primary_loss),
            "scalar_aux": value_weight * scalar_aux_weight * scalar_aux_loss,
            "root_q_consistency": consistency_weight * max(
                0.0, _number(row.get("value_search_consistency_loss")) or 0.0
            ),
            "search_q": search_q_weight * max(
                0.0, _number(row.get("search_q_loss")) or 0.0
            ),
            "moves_left": moves_weight * max(
                0.0, _number(row.get("moves_left_loss")) or 0.0
            ),
        }
        rank_contribution = _number(row.get("policy_correction_rank_contribution"))
        if rank_contribution is not None:
            values["policy_rank"] = policy_weight * max(0.0, rank_contribution)
        else:
            # Older schemas did not log the rank term separately. The objective
            # residual is exact for those runs because all other positive terms
            # and the final averaged objective were already logged.
            total = _number(row.get("avg_loss"))
            if total is not None:
                values["policy_rank"] = max(0.0, total - sum(values.values()))
        return values

    specs = (
        ("policy_ce", "policy CE", _COLORS[0]),
        ("policy_rank", "policy rank", _COLORS[5]),
        ("wdl", "WDL value", _COLORS[1]),
        ("scalar_aux", "scalar W-L", _COLORS[4]),
        ("root_q_consistency", "WDL-root-Q", _COLORS[3]),
        ("search_q", "search-Q", _COLORS[2]),
        ("moves_left", "moves-left", "#94a3b8"),
    )
    shown = _stacked_iteration_bars(
        ax,
        rows,
        specs,
        _values,
        title="Weighted loss decomposition",
        ylabel="loss contribution",
        legend_columns=4,
        seamless=True,
    )
    if not shown:
        _no_data(ax, "Loss-component telemetry is available from schema 14")
    return shown


def _champion_replay_panel(
    ax,
    rows,
    promotion_iterations,
    recovery_iterations,
    target_fraction,
):
    """Render the stable champion reservoir that replaced live opponent mixing."""
    _style(ax, "Champion replay stability", percent=True)
    shown = _line(
        ax,
        rows,
        "champion_replay_selected_fraction",
        "training share",
        _COLORS[3],
        lw=2.2,
        break_after=tuple(promotion_iterations) + tuple(recovery_iterations),
    )
    shown |= _line(
        ax,
        rows,
        "champion_replay_target_fraction",
        "dynamic target",
        "#7c3aed",
        ls="--",
        lw=1.7,
        break_after=tuple(promotion_iterations) + tuple(recovery_iterations),
    )
    fill_x, fill_y = [], []
    for row in rows:
        iteration = _number(row.get("iteration"))
        size = _number(row.get("champion_replay_size"))
        capacity = _number(row.get("champion_replay_capacity"))
        if iteration is None or size is None or capacity is None or capacity <= 0.0:
            continue
        fill_x.append(int(iteration)); fill_y.append(max(0.0, min(1.0, size / capacity)))
    if fill_x:
        for segment_index, (segment_x, segment_y) in enumerate(
            _split_series_at_boundaries(
                fill_x,
                fill_y,
                tuple(promotion_iterations) + tuple(recovery_iterations),
            )
        ):
            ax.plot(
                segment_x,
                segment_y,
                color=_COLORS[0],
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label="reservoir fill" if segment_index == 0 else None,
            )
        shown = True
    if target_fraction > 0.0 and not _series(rows, "champion_replay_target_fraction")[0]:
        ax.axhline(target_fraction, color="#7c3aed", linestyle=":", linewidth=1.2,
                   label=f"target share {100.0 * target_fraction:.0f}%")

    latest_size = _last(rows, "champion_replay_size")
    latest_capacity = _last(rows, "champion_replay_capacity")
    latest_added = _last(rows, "champion_replay_added")
    if latest_size is not None and latest_capacity is not None:
        text = f"reservoir {latest_size:,.0f}/{latest_capacity:,.0f}"
        if latest_added is not None:
            text += f" | +{latest_added:,.0f} latest"
        ax.text(0.98, 0.04, text, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="#4b5563")
    ax.set_ylim(0.0, 1.05)
    _mark_regime_changes(
        ax,
        promotion_iterations,
        recovery_iterations,
        label=True,
    )
    _legend_below(ax, ncol=2)
    if not shown:
        _no_data(ax, "Champion replay telemetry is available from schema 12")
    return shown


def render_rl_main(main_csv_path, data_quality_csv_path, performance_csv_path, output_path,
                   run_context_lines=None):
    """Render the single screen used to judge whether the run is improving."""
    main = _rows(main_csv_path)
    detail = _rows(data_quality_csv_path)
    performance = _rows(performance_csv_path)
    rl_cfg = _rl_config(main_csv_path)
    if not main:
        return False

    holdout_gain_metrics = (
        ("holdout_policy_kl", "holdout_policy_kl_before", "holdout_policy_kl_reduction"),
        ("holdout_wdl_ce", "holdout_value_wdl_ce_before", "holdout_value_wdl_ce_reduction"),
        ("holdout_brier", "holdout_value_brier_before", "holdout_value_brier_reduction"),
        ("holdout_value_mae", "holdout_value_mae_before", "holdout_value_mae_reduction"),
    )
    for row in main:
        for target, before, reduction in holdout_gain_metrics:
            row[f"_{target}_relative_gain_pct"] = _relative_holdout_gain_pct(
                row, before, reduction
            )

    sections = [
        ("Strength and evaluation", [(0, 0), (0, 1), (0, 2)], 3),
        ("Learning signals", [(1, 2), (2, 0), (2, 1)], 3),
        ("Search and calibration", [(1, 1), (2, 2)], 2),
        ("Measured timing and hardware path", [(3, column) for column in range(4)], 4),
        ("Holdout learning audit", [(4, 0), (4, 1)], 2),
        ("Gradient health", [(5, 0), (5, 1)], 2),
    ]
    fig, axes = create_dashboard(sections)
    promotion_iterations, recovery_iterations = _regime_change_iterations(main)
    selfplay_best_intervals = _selfplay_best_intervals(detail)
    matchup_breaks = tuple(promotion_iterations) + tuple(recovery_iterations)

    ax = axes[0, 0]
    _promotion_strength_panel(ax, main, rl_cfg)

    ax = axes[0, 1]
    _value_search_alignment_panel(ax, main, break_after=recovery_iterations)

    ax = axes[0, 2]
    _style(ax, "Estimated Elo with uncertainty")
    shown = False
    nn_xs, nn_ys, nn_lows, nn_highs = [], [], [], []
    for row in main:
        x = _number(row.get("iteration"))
        y = _number(row.get("estimated_elo_nn"))
        if x is None or y is None:
            continue
        low = _number(row.get("estimated_elo_nn_ci95_low"))
        high = _number(row.get("estimated_elo_nn_ci95_high"))
        nn_xs.append(int(x)); nn_ys.append(y)
        nn_lows.append(max(0.0, y - low) if low is not None else 0.0)
        nn_highs.append(max(0.0, high - y) if high is not None else 0.0)
    if nn_xs:
        ax.errorbar(
            nn_xs,
            nn_ys,
            yerr=np.asarray([nn_lows, nn_highs]),
            color="#0EA5E9",
            label="NN",
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            alpha=0.95,
        )
        shown = True

    mcts_series = {}
    for row in main:
        x = _number(row.get("iteration"))
        if x is None:
            continue
        by_simulations = _parse_mcts_elo_by_simulations(
            row.get("estimated_elo_mcts_by_simulations")
        )
        if not by_simulations:
            simulations = _number(row.get("estimated_elo_mcts_simulations"))
            elo = _number(row.get("estimated_elo_mcts"))
            if simulations is not None and elo is not None:
                low = _number(row.get("estimated_elo_mcts_ci95_low"))
                high = _number(row.get("estimated_elo_mcts_ci95_high"))
                entry = {"elo": elo}
                if low is not None and high is not None:
                    entry["ci95"] = [low, high]
                by_simulations = {int(simulations): entry}
        for simulations, entry in by_simulations.items():
            elo = float(entry["elo"])
            ci95 = entry.get("ci95")
            low = _number(ci95[0]) if isinstance(ci95, (list, tuple)) and len(ci95) == 2 else None
            high = _number(ci95[1]) if isinstance(ci95, (list, tuple)) and len(ci95) == 2 else None
            bucket = mcts_series.setdefault(
                int(simulations),
                {"xs": [], "ys": [], "lows": [], "highs": []},
            )
            bucket["xs"].append(int(x)); bucket["ys"].append(elo)
            bucket["lows"].append(max(0.0, elo - low) if low is not None else 0.0)
            bucket["highs"].append(max(0.0, high - elo) if high is not None else 0.0)

    mcts_palette = ("#F59E0B", "#7C3AED", "#EF4444", "#10B981", "#EC4899")
    for index, (simulations, bucket) in enumerate(sorted(mcts_series.items())):
        ax.errorbar(
            bucket["xs"],
            bucket["ys"],
            yerr=np.asarray([bucket["lows"], bucket["highs"]]),
            color=mcts_palette[index % len(mcts_palette)],
            label=f"MCTS @{simulations}",
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            alpha=0.95,
        )
        shown = True
    _legend_below(ax, ncol=min(4, 1 + len(mcts_series)))
    if not shown:
        _no_data(ax)

    _overall_runtime_donut(axes[3, 0], performance)
    _selfplay_worker_search_donut(axes[3, 1], performance)
    _cpu_path_timer_donut(axes[3, 2], performance)
    _gpu_path_timer_donut(axes[3, 3], performance)

    ax = axes[1, 1]
    _search_correction_rate_panel(
        ax,
        detail,
        break_after=matchup_breaks,
        promotion_iterations=promotion_iterations,
        recovery_iterations=recovery_iterations,
    )

    ax = axes[1, 2]
    _objective_composition_panel(ax, main, rl_cfg)

    ax = axes[2, 0]
    _style(ax, "Policy learning")
    shown = _line(
        ax, main, "policy_loss", "policy loss", _COLORS[0], lw=2.1,
        break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "policy_correction_loss", "useful correction loss",
        _COLORS[5], ls="--", lw=1.7, break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "policy_correction_rank_contribution", "rank contribution",
        _COLORS[3], ls=":", lw=1.7, break_after=recovery_iterations,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    accuracy = _line(
        ax2, main, "policy_top1_acc", "top-1", _COLORS[2], lw=1.8,
        break_after=recovery_iterations,
    )
    ax.set_ylabel("policy loss", fontsize=7.5, color=_COLORS[0])
    ax2.set_ylabel("top-1", fontsize=7.5, color=_COLORS[2])
    ax.tick_params(axis="y", labelcolor=_COLORS[0])
    ax2.tick_params(axis="y", labelcolor=_COLORS[2])
    _legend_below(ax, ax2, ncol=2)
    if not (shown or accuracy): _no_data(ax)

    ax = axes[2, 1]
    _style(ax, "Value learning")
    shown = _line(
        ax, main, "value_primary_loss", "WDL CE", _COLORS[3], lw=2.1,
        break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "value_scalar_aux_loss", "scalar W-L auxiliary",
        _COLORS[0], ls="--", lw=1.5, break_after=recovery_iterations,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    mae = _line(
        ax2, main, "value_mae", "value MAE", "#111827", lw=1.8,
        break_after=recovery_iterations,
    )
    mae |= _line(
        ax2, main, "value_wdl_brier", "WDL Brier", _COLORS[1], ls="--", lw=1.5,
        break_after=recovery_iterations,
    )
    mae |= _line(
        ax2, main, "value_wdl_ece", "WDL ECE", _COLORS[5], ls=":", lw=1.5,
        break_after=recovery_iterations,
    )
    _legend_below(ax, ax2, ncol=4)
    if not (shown or mae): _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "Value spread and draw calibration")
    shown = _line(
        ax, main, "value_std_ratio_opening", "opening", _COLORS[0],
        break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "value_std_ratio_middlegame", "middlegame", _COLORS[3],
        break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "value_std_ratio_endgame", "endgame", _COLORS[1],
        break_after=recovery_iterations,
    )
    ax.axhline(1.0, color="#374151", lw=1, ls="--", label="calibrated")
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    draw_shown = _line(
        ax2, main, "value_pred_draw_probability", "predicted draw P",
        _COLORS[4], ls="--", lw=1.6, break_after=recovery_iterations,
    )
    draw_shown |= _line(
        ax2, main, "value_target_draw_fraction", "target draw fraction",
        "#111827", ls=":", lw=1.6, break_after=recovery_iterations,
    )
    _legend_below(ax, ax2, ncol=3)
    if not (shown or draw_shown):
        _no_data(ax)

    ax = axes[4, 0]
    _style(ax, "Holdout policy and value error reduction")
    shown = False
    for column, label, color in (
        ("_holdout_policy_kl_relative_gain_pct", "policy KL", _COLORS[0]),
        ("_holdout_wdl_ce_relative_gain_pct", "WDL cross-entropy", _COLORS[3]),
        ("_holdout_brier_relative_gain_pct", "WDL Brier", _COLORS[2]),
        ("_holdout_value_mae_relative_gain_pct", "value MAE", _COLORS[4]),
    ):
        shown |= _line(ax, main, column, label, color, break_after=recovery_iterations)
    ax.axhline(0.0, color="#374151", lw=1.0, ls=":", label="no change")
    ax.set_ylabel("reduction vs before (%)", fontsize=7.5)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax, "Held-out before/after policy and value metrics unavailable")
    else:
        holdout_rows = _last(main, "holdout_rows")
        if holdout_rows is not None:
            ax.text(
                0.98, 0.96, f"latest holdout n={holdout_rows:,.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#4b5563",
            )

    ax = axes[4, 1]
    _style(ax, "Holdout root-Q MAE reduction by phase")
    shown = False
    for column, label, color in (
        ("holdout_root_q_mae_reduction", "overall", _COLORS[0]),
        ("holdout_root_q_mae_opening_reduction", "opening", _COLORS[2]),
        ("holdout_root_q_mae_middlegame_reduction", "middlegame", _COLORS[3]),
        ("holdout_root_q_mae_endgame_reduction", "endgame", _COLORS[4]),
    ):
        shown |= _line(ax, main, column, label, color, break_after=recovery_iterations)
    ax.axhline(0.0, color="#374151", lw=1.0, ls=":", label="no change")
    ax.set_ylabel("MAE reduction (value units)", fontsize=7.5)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax, "Held-out root-Q phase errors unavailable")

    ax = axes[5, 0]
    _style(ax, "Gradient norms and clipping")
    shown = False
    for column, label, color in (
        ("grad_total_norm", "total norm", _COLORS[0]),
        ("grad_backbone_norm", "backbone", _COLORS[1]),
        ("grad_policy_head_norm", "policy head", _COLORS[2]),
        ("grad_value_head_norm", "value head", _COLORS[3]),
    ):
        shown |= _line(ax, main, column, label, color, break_after=recovery_iterations)
    ax.set_ylabel("gradient L2 norm", fontsize=7.5)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    clipped = _line(
        ax2, main, "grad_clip_fraction", "steps clipped", "#f59e0b",
        ls="--", break_after=recovery_iterations,
    )
    ax2.set_ylabel("clipped optimizer steps", fontsize=7.2, color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    ax2.set_ylim(0.0, 1.0)
    _legend_below(ax, ax2, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or clipped):
        _no_data(ax, "Gradient norm and clipping telemetry unavailable")

    ax = axes[5, 1]
    _style(ax, "Policy/value gradient alignment")
    shown = _line(
        ax, main, "grad_policy_value_cosine", "policy/value cosine", _COLORS[5],
        lw=2.0, break_after=recovery_iterations,
    )
    ax.axhline(0.0, color="#374151", lw=1.0, ls="--", label="orthogonal")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cosine similarity (−1 to +1)", fontsize=7.5)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax, "Policy/value gradient cosine unavailable")

    for plot_index, plot_ax in enumerate((
        axes[0, 0], axes[0, 1], axes[0, 2],
        axes[1, 2],
        axes[2, 0], axes[2, 1], axes[2, 2],
    )):
        _mark_regime_changes(
            plot_ax,
            promotion_iterations,
            recovery_iterations,
            label=(plot_index == 0),
        )
    # Best-actor intervals describe self-play provenance, not sparse external
    # Elo evaluations. They do belong on value/objective/search timelines
    # because actor changes can affect both targets and optimization difficulty.
    timeline_axes = [
        axes[0, 0], axes[0, 1], axes[1, 1], axes[1, 2],
        axes[2, 0], axes[2, 1], axes[2, 2],
        axes[4, 0], axes[4, 1], axes[5, 0], axes[5, 1],
    ]
    for plot_index, plot_ax in enumerate(timeline_axes):
        _mark_selfplay_best_intervals(
            plot_ax,
            selfplay_best_intervals,
            label=(plot_index == 0),
        )
    _mark_resume_boundaries(
        {key: ax for key, ax in axes.items() if key[0] != 3},
        _resume_markers(main_csv_path),
    )

    context = [str(line) for line in (run_context_lines or []) if line]
    _finish(
        fig,
        output_path,
        "RL training - strength and learning dashboard",
        context or None,
    )
    return True


def render_rl_data_quality(
    main_csv_path,
    data_quality_csv_path,
    output_path,
    *,
    reanalysis_context=None,
):
    """Render the compact dashboard used to diagnose RL data and search quality."""
    main = _rows(main_csv_path)
    rows = _rows(data_quality_csv_path)
    rl_cfg = _rl_config(main_csv_path)
    if not rows:
        return False
    reuse_summary = _rl_reuse_accounting_summary(rows)
    for row in rows:
        for target, numerator, denominator in (
            ("_syzygy_ended_rate", "selfplay_syzygy_ended_games", "selfplay_completed_games"),
            ("_resigned_rate", "selfplay_resigned_games", "selfplay_completed_games"),
            ("_adjudicated_rate", "selfplay_adjudicated_games", "selfplay_completed_games"),
        ):
            row[target] = _ratio_if_positive(row, numerator, denominator)
    sections = [
        ("Replay feed and retention", [(0, 0), (0, 1), (1, 0), (3, 0)], 3),
        ("Targets and correction learning", [(0, 2), (1, 1), (1, 2)], 3),
        ("Search and self-play quality", [(2, 0), (2, 2), (3, 1), (3, 2)], 3),
        ("Replay freshness and training coverage", [(4, 0), (4, 1), (4, 2)], 3),
        ("Adaptive search budget and reuse", [(5, 0), (5, 1), (5, 2)], 3),
        ("Game termination and opening breadth", [(6, 0), (6, 1)], 2),
        ("MCTS reuse accounting", [(7, 0), (7, 1)], 2),
    ]
    fig, axes = create_dashboard(sections)
    promotion_iterations, recovery_iterations = _regime_change_iterations(main)
    selfplay_best_intervals = _selfplay_best_intervals(rows)
    selfplay_breaks = tuple(promotion_iterations) + tuple(recovery_iterations)

    ax = axes[0, 0]
    intake_specs = (
        ("kept", "kept", _COLORS[0]),
        ("cap_rejected", "cap rejected", _COLORS[4]),
        ("overwritten", "FIFO overwritten", _COLORS[1]),
        ("resize_dropped", "resize dropped", "#7c3aed"),
    )

    def _replay_intake_values(row):
        return {
            "kept": _number(row.get("positions_added")) or 0.0,
            "cap_rejected": _number(row.get("replay_cap_dropped_positions")) or 0.0,
            "overwritten": _number(row.get("replay_overwritten_positions")) or 0.0,
            "resize_dropped": _number(row.get("replay_resize_dropped_positions")) or 0.0,
        }

    shown = _stacked_iteration_bars(
        ax,
        rows,
        intake_specs,
        _replay_intake_values,
        title="Replay intake and eviction",
        ylabel="positions",
        legend_columns=2,
        seamless=True,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax)

    ax = axes[0, 1]
    _style(ax, "Replay outcome balance and sampling bias", percent=True)
    shown = _line(
        ax,
        rows,
        "replay_decisive_fraction",
        "replay decisive",
        _COLORS[0],
        lw=1.9,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax,
        rows,
        "replay_draw_fraction",
        "replay draw",
        "#64748b",
        lw=1.9,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax,
        rows,
        "selfplay_decisive_rate",
        "played decisive",
        "#16a34a",
        ls="--",
        lw=1.6,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax,
        rows,
        "selfplay_draw_rate",
        "played-game draw rate",
        "#f59e0b",
        ls="--",
        lw=1.7,
        break_after=selfplay_breaks,
    )
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    skew = _line(
        ax2,
        rows,
        "replay_value_skew",
        "value sign skew",
        "#7c3aed",
        ls=":",
        lw=1.7,
        break_after=selfplay_breaks,
    )
    ax2.axhline(0.0, color="#7c3aed", lw=0.8, ls=":", alpha=0.45)
    ax2.set_ylabel("positive - negative target share", fontsize=7.2, color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")

    latest_decisive = _last(rows, "replay_decisive_fraction")
    latest_draw = _last(rows, "replay_draw_fraction")
    latest_played_draw = _last(rows, "selfplay_draw_rate")
    latest_skew = _last(rows, "replay_value_skew")
    if None not in (latest_decisive, latest_draw):
        played_draw_text = (
            "n/a" if latest_played_draw is None else f"{100.0 * latest_played_draw:.1f}%"
        )
        skew_text = "n/a" if latest_skew is None else f"{latest_skew:+.3f}"
        ax.text(
            0.98,
            0.04,
            (
                f"latest replay D/draw {100.0 * latest_decisive:.1f}% / "
                f"{100.0 * latest_draw:.1f}% | played draw {played_draw_text} | "
                f"skew {skew_text}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#4b5563",
        )
    ax.set_ylim(0.0, 1.05)
    _mark_regime_changes(
        ax, promotion_iterations, recovery_iterations, label=True,
    )
    _legend_below(ax, ax2, ncol=3)
    if not (shown or skew):
        _no_data(ax)

    ax = axes[0, 2]
    _style(ax, "Value learning vs outcome / root-Q")
    shown = _line(
        ax, main, "value_loss", "value loss", _COLORS[3], lw=2.1,
        break_after=recovery_iterations,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    mae = _line(
        ax2, main, "value_mae", "outcome MAE", "#111827", lw=1.8,
        break_after=recovery_iterations,
    )
    root_q_mae = _line(
        ax2,
        main,
        "value_root_q_mae",
        "root-Q MAE",
        _COLORS[1],
        lw=1.8,
        break_after=recovery_iterations,
    )
    latest_root_q_bias = _last(main, "value_root_q_bias")
    latest_root_q_corr = _last(main, "value_root_q_correlation")
    if latest_root_q_bias is not None or latest_root_q_corr is not None:
        bias_text = "n/a" if latest_root_q_bias is None else f"{latest_root_q_bias:+.3f}"
        corr_text = "n/a" if latest_root_q_corr is None else f"{latest_root_q_corr:+.2f}"
        ax.text(
            0.02,
            0.96,
            f"latest root-Q bias {bias_text} | corr {corr_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#4b5563",
        )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ax2, ncol=3)
    if not (shown or mae or root_q_mae):
        _no_data(ax)

    ax = axes[1, 0]
    _champion_replay_panel(
        ax,
        rows,
        promotion_iterations,
        recovery_iterations,
        max(0.0, min(0.40, float(rl_cfg.get("replay_champion_fraction", 0.15) or 0.15))),
    )

    ax = axes[1, 1]
    _style(ax, "Policy target shape")
    shown = _line(
        ax, rows, "policy_target_effective_moves", "effective moves", _COLORS[4], lw=2.0,
        break_after=selfplay_breaks,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    probs = _line(
        ax2, rows, "policy_target_top1_prob_mean", "top-1 mass", _COLORS[0],
        break_after=selfplay_breaks,
    )
    probs |= _line(
        ax2, rows, "policy_target_top3_prob_mean", "top-3 mass", _COLORS[2],
        break_after=selfplay_breaks,
    )
    _legend_below(ax, ax2, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or probs):
        _no_data(ax)

    ax = axes[1, 2]
    _style(ax, "MCTS correction absorption audit", percent=True)
    shown = _line(
        ax, rows, "correction_audit_top1_before", "top-1 before", _COLORS[3],
        ls=":", lw=1.7, break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "correction_audit_top1_after", "top-1 after", "#7c3aed",
        lw=2.1, break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "correction_audit_retention", "next-iteration retention", _COLORS[2],
        ls="--", lw=1.7, break_after=selfplay_breaks,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    kl_shown = _line(
        ax2, rows, "correction_audit_target_kl_before", "target KL before", _COLORS[0],
        ls=":", lw=1.4, break_after=selfplay_breaks,
    )
    kl_shown |= _line(
        ax2, rows, "correction_audit_target_kl_after", "target KL after", _COLORS[4],
        ls="--", lw=1.7, break_after=selfplay_breaks,
    )
    latest_rows = _last(rows, "correction_audit_rows")
    latest_top1_gain = _last(rows, "correction_audit_top1_gain")
    latest_probability_gain = _last(rows, "correction_audit_target_probability_gain")
    latest_kl_closed = _last(rows, "correction_audit_target_kl_closed_fraction")
    if latest_rows is not None:
        ax.text(
            0.02, 0.96,
            f"latest n={int(latest_rows):,} | top-1 "
            f"{(latest_top1_gain or 0.0):+.1%} | target P "
            f"{(latest_probability_gain or 0.0):+.2%} | KL closed "
            f"{(latest_kl_closed or 0.0):+.1%}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color="#4b5563",
        )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ax2, ncol=2)
    if not (shown or kl_shown):
        _no_data(ax)

    ax = axes[2, 0]
    budgets = (64, 128, 192, 256, 320)
    root_counts = [_last(rows, f"mcts_budget_{budget}_root_count") for budget in budgets]
    _style(ax, "MCTS simulation budget by iteration")
    shown = _line(
        ax, rows, "mcts_avg_budget", "average allocated budget", _COLORS[0], lw=2.2,
        break_after=selfplay_breaks,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=2)
    ax.set_ylabel("allocated simulations / root", fontsize=7.5)
    latest_budget_json = _last(rows, "mcts_budget_quality_levels_json")
    all_level_summary = []
    if latest_budget_json:
        try:
            levels = json.loads(latest_budget_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            levels = {}
        if isinstance(levels, dict):
            for level in sorted(
                (key for key in levels if str(key).isdigit()),
                key=lambda key: int(key),
            ):
                item = levels.get(level) or {}
                count = int(item.get("samples") or 0)
                rate = item.get("useful_change_rate")
                shown_rate = "—" if count <= 0 or rate is None else f"{float(rate):.0%}"
                all_level_summary.append(f"{level}:{shown_rate}[{count}]")
    if all_level_summary:
        summary_lines = [
            " · ".join(all_level_summary[start:start + 6])
            for start in range(0, len(all_level_summary), 6)
        ]
        coverage = _last(rows, "mcts_budget_quality_coverage")
        coverage_text = (
            f" | known {float(coverage):.0%}"
            if coverage is not None else ""
        )
        ax.text(
            0.02, 0.96,
            "useful correction by 16-step budget" + coverage_text + "\n"
            + "\n".join(summary_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=5.7,
            color="#4b5563",
        )
    elif any((count or 0.0) > 0.0 for count in root_counts):
        budget_summary = []
        for budget, count in zip(budgets, root_counts):
            if (count or 0.0) <= 0.0:
                continue
            useful = _last(rows, f"mcts_budget_{budget}_useful_change_rate") or 0.0
            budget_summary.append(f"{budget}: {useful:.0%} (n={int(count):,})")
        summary_lines = [
            " · ".join(budget_summary[start:start + 3])
            for start in range(0, len(budget_summary), 3)
        ]
        ax.text(
            0.02, 0.96,
            "latest useful corrections | " + "\n".join(summary_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color="#4b5563",
        )
    if not shown:
        _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "Q gain when search changes move")
    shown = _line(
        ax, rows, "mcts_changed_q_delta_p10", "p10", _COLORS[1], ls="--",
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_changed_q_delta_mean", "mean", _COLORS[0], lw=2.1,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_changed_q_delta_p90", "p90", _COLORS[2], ls="--",
        break_after=selfplay_breaks,
    )
    p10_x, p10_y = _series(rows, "mcts_changed_q_delta_p10")
    p90_x, p90_y = _series(rows, "mcts_changed_q_delta_p90")
    if p10_x and p10_x == p90_x:
        _fill_between_with_breaks(
            ax, p10_x, p10_y, p90_y, break_after=selfplay_breaks,
            color=_COLORS[0], alpha=0.08, linewidth=0,
        )
    ax.axhline(0.0, color="#374151", lw=1, ls=":")
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax)

    ax = axes[3, 0]
    _style(ax, "Independent-game diversity in replay")
    shown = _line(
        ax, rows, "replay_games", "games represented", _COLORS[0], lw=2.0,
        break_after=selfplay_breaks,
    )
    ax.set_ylabel("games", fontsize=8)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    positions = _line(
        ax2, rows, "replay_positions_per_game_mean", "positions/game mean",
        _COLORS[3], lw=1.8, break_after=selfplay_breaks,
    )
    positions |= _line(
        ax2, rows, "replay_positions_per_game_p90", "positions/game p90",
        _COLORS[1], ls="--", lw=1.7, break_after=selfplay_breaks,
    )
    ax2.set_ylabel("positions per game", fontsize=7.5)
    _legend_below(ax, ax2, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or positions):
        _no_data(ax)

    ax = axes[3, 1]
    _style(ax, "Search changes by phase", percent=True)
    shown = _line(
        ax, rows, "mcts_changed_opening_rate", "opening", _COLORS[0],
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_changed_middlegame_rate", "middlegame", _COLORS[3],
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_changed_endgame_rate", "endgame", _COLORS[1],
        break_after=selfplay_breaks,
    )
    opening_unique = _last(rows, "selfplay_opening_unique_8ply")
    opening_effective = _last(rows, "selfplay_opening_effective_8ply")
    opening_top1 = _last(rows, "selfplay_opening_top1_8ply_rate")
    if (
        opening_unique is not None
        or opening_effective is not None
        or opening_top1 is not None
    ):
        unique_text = "n/a" if opening_unique is None else f"{opening_unique:.0f}"
        effective_text = "n/a" if opening_effective is None else f"{opening_effective:.1f}"
        top1_text = "n/a" if opening_top1 is None else f"{opening_top1:.1%}"
        ax.text(
            0.02,
            0.96,
            f"played 8-ply: unique {unique_text} | effective {effective_text} | top-1 {top1_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#4b5563",
        )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax)

    _latest_search_decision_donut(axes[3, 2], rows)

    ax = axes[4, 0]
    _style(ax, "Replay sample age")
    shown = False
    for column, label, color in (
        ("sample_age_avg", "mean", _COLORS[0]),
        ("sample_age_p50", "p50", _COLORS[2]),
        ("sample_age_p90", "p90", _COLORS[3]),
    ):
        shown |= _line(ax, rows, column, label, color, break_after=selfplay_breaks)
    ax.set_ylabel("iterations old at sample time", fontsize=7.5)
    _legend_below(ax, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax, "Replay sample-age telemetry unavailable")

    ax = axes[4, 1]
    _style(ax, "Training sample coverage and replay fill", percent=True)
    shown = False
    for column, label, color in (
        ("train_replay_coverage", "sample coverage", _COLORS[0]),
        ("train_game_coverage", "game coverage", _COLORS[2]),
        ("replay_fill_rate", "buffer fill", _COLORS[3]),
    ):
        shown |= _line(ax, rows, column, label, color, break_after=selfplay_breaks)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    passes = _line(
        ax2, rows, "train_replay_passes", "replay passes", _COLORS[4],
        ls="--", break_after=selfplay_breaks,
    )
    ax2.set_ylabel("train-selected positions / replay size", fontsize=7.2, color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    _legend_below(ax, ax2, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or passes):
        _no_data(ax, "Training coverage and replay-fill telemetry unavailable")

    ax = axes[4, 2]
    _draw_rl_reanalysis_activity(
        ax,
        rows,
        historical_note=reanalysis_context,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)

    ax = axes[5, 0]
    _style(ax, "Adaptive simulation budget spread")
    shown = False
    for column, label, color, style in (
        ("mcts_budget_p10", "p10", _COLORS[0], "--"),
        ("mcts_budget_p50", "p50", _COLORS[2], "-"),
        ("mcts_budget_p90", "p90", _COLORS[3], "--"),
        ("mcts_budget_target", "configured target", "#111827", ":"),
    ):
        shown |= _line(
            ax, rows, column, label, color, ls=style, break_after=selfplay_breaks,
        )
    p10_x, p10_y = _series(rows, "mcts_budget_p10")
    p90_x, p90_y = _series(rows, "mcts_budget_p90")
    if p10_x and p10_x == p90_x:
        _fill_between_with_breaks(
            ax, p10_x, p10_y, p90_y, break_after=selfplay_breaks,
            color=_COLORS[0], alpha=0.08, linewidth=0,
        )
    ax.set_ylabel("simulations per position", fontsize=7.5)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    utilization = _line(
        ax2, rows, "mcts_budget_utilization", "mean budget utilization", _COLORS[4],
        ls="-.", break_after=selfplay_breaks,
    )
    ax2.set_ylabel("budget used / allocated", fontsize=7.2, color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    _legend_below(ax, ax2, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or utilization):
        _no_data(ax, "Adaptive simulation-budget distribution unavailable")

    ax = axes[5, 1]
    _style(ax, "Budget assignment vs search difficulty")
    shown = _line(
        ax, rows, "mcts_difficulty_budget_correlation", "difficulty/budget correlation",
        _COLORS[0], lw=2.0, break_after=selfplay_breaks,
    )
    ax.axhline(0.0, color="#374151", lw=1.0, ls=":", label="no association")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("per-position correlation (−1 to +1)", fontsize=7.2)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    coverage = _line(
        ax2, rows, "mcts_budget_quality_coverage", "budget-quality sample coverage",
        _COLORS[4], ls="--", break_after=selfplay_breaks,
    )
    ax2.set_ylabel("positions with quality labels", fontsize=7.0, color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    ax2.set_ylim(0.0, 1.0)
    _legend_below(ax, ax2, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or coverage):
        _no_data(ax, "Difficulty-to-budget correlation unavailable")
    else:
        ax.text(
            0.02, 0.97, "Positive correlation means harder roots receive more simulations.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#4b5563",
        )

    ax = axes[5, 2]
    _style(ax, "Tree reuse quality and coverage", percent=True)
    shown = False
    for column, label, color in (
        ("mcts_tree_reuse_hit_rate", "reuse hit rate", _COLORS[0]),
        ("mcts_tree_reuse_quality_avg", "reuse quality", _COLORS[2]),
        ("mcts_tree_reuse_candidate_coverage_avg", "candidate coverage", _COLORS[3]),
        ("mcts_tree_reuse_visited_prior_mass_avg", "visited prior mass", _COLORS[4]),
        ("mcts_tree_reuse_visit_credit_share", "inherited visit share", _COLORS[5]),
    ):
        shown |= _line(ax, rows, column, label, color, break_after=selfplay_breaks)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax, "Tree-reuse quality counters unavailable")

    ax = axes[6, 0]
    _style(ax, "Self-play termination rates", percent=True)
    shown = False
    for column, label, color in (
        ("selfplay_auto_draw_rate", "automatic draw", _COLORS[0]),
        ("selfplay_truncated_rate", "move-limit truncated", _COLORS[1]),
        ("_syzygy_ended_rate", "Syzygy-ended / completed", _COLORS[2]),
        ("_resigned_rate", "resigned / completed", _COLORS[3]),
        ("_adjudicated_rate", "adjudicated / completed", _COLORS[4]),
    ):
        shown |= _line(ax, rows, column, label, color, break_after=selfplay_breaks)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if shown:
        ax.text(
            0.02, 0.97, "Each rate uses its logged or completed-game denominator; rates are not stacked.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#4b5563",
        )
    else:
        _no_data(ax, "Self-play termination telemetry unavailable")

    ax = axes[6, 1]
    _style(ax, "Self-play opening breadth")
    shown = _line(
        ax, rows, "selfplay_opening_unique_8ply", "unique 8-ply prefixes", _COLORS[0],
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "selfplay_opening_effective_8ply", "effective prefixes", _COLORS[2],
        break_after=selfplay_breaks,
    )
    ax.set_ylabel("prefix count (unique / effective)", fontsize=7.3)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    concentration = _line(
        ax2, rows, "selfplay_opening_top1_8ply_rate", "most common prefix share",
        _COLORS[4], ls="--", break_after=selfplay_breaks,
    )
    concentration |= _line(
        ax2, rows, "selfplay_opening_prefix_coverage_completed", "prefix coverage",
        _COLORS[5], ls=":", break_after=selfplay_breaks,
    )
    ax2.set_ylabel("share of completed games", fontsize=7.0, color="#b45309")
    ax2.tick_params(axis="y", labelcolor="#b45309")
    _legend_below(ax, ax2, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or concentration):
        _no_data(ax, "Opening-prefix diversity telemetry unavailable")

    ax = axes[7, 0]
    _style(ax, "Fresh + inherited visits per search root")
    search_iterations = []
    fresh_per_root = []
    inherited_per_root = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        parts = _selfplay_inherited_visit_parts(row)
        if iteration is None or parts is None:
            continue
        search_iterations.append(int(iteration))
        fresh_per_root.append(parts[0])
        inherited_per_root.append(parts[1])
    if search_iterations:
        ax.bar(
            search_iterations, fresh_per_root, width=0.68, color=_COLORS[0],
            label="fresh simulations / root", edgecolor="white", linewidth=0.35,
        )
        ax.bar(
            search_iterations, inherited_per_root, bottom=fresh_per_root,
            width=0.68, color="#db2777", label="raw inherited visits / root",
            edgecolor="white", linewidth=0.35,
        )
        for iteration, fresh, inherited in zip(
            search_iterations, fresh_per_root, inherited_per_root
        ):
            ax.annotate(
                f"{fresh + inherited:.0f}",
                (iteration, fresh + inherited),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color="#9d174d",
            )
    effective_shown = _line(
        ax, rows, "mcts_avg_sims", "effective simulations used", "#111827",
        lw=1.8, break_after=selfplay_breaks,
    )
    budget_shown = _line(
        ax, rows, "mcts_avg_budget", "allocated budget", "#64748b",
        ls="--", break_after=selfplay_breaks,
    )
    ax.set_ylabel("visits or simulations / search root", fontsize=7.5)
    _legend_below(ax, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (search_iterations or effective_shown or budget_shown):
        _no_data(ax, "Per-search-root inherited-visit accounting unavailable")

    ax = axes[7, 1]
    _style(ax, "Reuse hit rate and raw inherited visits", percent=True)
    hit_rate_shown = _line(
        ax, rows, "mcts_tree_reuse_hit_rate", "reuse hits / attempts", _COLORS[0],
        break_after=selfplay_breaks,
    )
    ax.set_ylim(0.0, 1.05)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    raw_visits_shown = _line(
        ax2, rows, "mcts_tree_inherited_visits_avg", "raw inherited visits / reused root",
        _COLORS[4], ls="--", break_after=selfplay_breaks,
    )
    q_scale_note = (
        " (Q scale: total)"
        if rl_cfg.get("mcts_gumbel_reuse_q_scale") == "total" else ""
    )
    ax2.set_ylabel(
        f"prior-root visits / reuse{q_scale_note}",
        fontsize=7.0,
        color="#b45309",
    )
    ax2.tick_params(axis="y", labelcolor="#b45309")
    _legend_below(ax, ax2, ncol=2)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (hit_rate_shown or raw_visits_shown):
        _no_data(ax, "Tree reuse hit and inherited-visit telemetry unavailable")

    for plot_index, plot_ax in enumerate(
        ax for key, ax in axes.items() if key != (3, 2)
    ):
        _mark_selfplay_best_intervals(
            plot_ax,
            selfplay_best_intervals,
            label=(plot_index == 0),
        )
    _legend_below(axes[0, 0], ncol=3)

    _mark_resume_boundaries(
        {key: ax for key, ax in axes.items() if key != (3, 2)},
        _resume_markers(main_csv_path),
    )
    _finish(
        fig,
        output_path,
        "RL details - data and search quality",
        _rl_reuse_summary_subtitle(
            reuse_summary,
            reuse_q_scale=rl_cfg.get("mcts_gumbel_reuse_q_scale"),
        ),
        hspace=0.78,
    )
    return True


def render_rl_reanalysis_activity(
    data_quality_csv_path,
    output_path,
    *,
    historical_note=None,
):
    """Render a focused replay-reanalysis view from an existing data-quality CSV."""
    rows = _rows(data_quality_csv_path)
    if not rows:
        return False
    fig, (activity_ax, budget_ax) = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.2),
        sharex=True,
        facecolor="#f3f5f8",
        gridspec_kw={"height_ratios": (1.5, 0.8)},
    )
    activity_ax.set_facecolor("white")
    budget_ax.set_facecolor("white")
    _draw_rl_reanalysis_activity(
        activity_ax,
        rows,
        historical_note=historical_note,
    )
    _draw_rl_reanalysis_budget(budget_ax, rows)
    _finish(
        fig,
        output_path,
        "RL replay reanalysis diagnostics",
        "Position counts are per iteration and refer to replay targets, not fresh self-play positions. "
        "Correction rate is search target correction versus the current NN, not learner uptake.",
        hspace=0.85,
    )
    return True


def render_rl_performance(
    performance_csv_path,
    output_path,
    data_quality_csv_path=None,
    main_csv_path=None,
):
    """Render throughput, latency and CPU/GPU-feed bottlenecks without duplicates."""
    rows = [dict(row) for row in _rows(performance_csv_path)]
    if not rows:
        return False

    quality_by_iteration = {}
    for quality_row in _rows(data_quality_csv_path):
        iteration = _number(quality_row.get("iteration"))
        if iteration is not None:
            quality_by_iteration[int(iteration)] = quality_row
    for row in rows:
        iteration = _number(row.get("iteration"))
        quality = quality_by_iteration.get(int(iteration)) if iteration is not None else None
        # Keep source counters distinct from compatibility estimates below.
        # The dedicated fresh-NPS panel must never relabel effective/fallback
        # simulations as fresh completed search work.
        row["logged_fresh_mcts_nps"] = _number(row.get("mcts_simulations_per_sec"))
        row["logged_effective_mcts_nps"] = _number(
            row.get("mcts_effective_simulations_per_sec")
        )
        replay_rate = _number(row.get("replay_positions_per_sec"))
        if replay_rate is None:
            replay_rate = _number(row.get("positions_per_sec"))  # schema <= 6
        if replay_rate is not None:
            row["replay_positions_per_sec"] = replay_rate

        played_rate = _number(row.get("played_positions_per_sec"))
        keep_rate = (
            _number(quality.get("selfplay_replay_storage_keep_rate"))
            if quality else None
        )
        if played_rate is None and replay_rate is not None and keep_rate and keep_rate > 0.0:
            played_rate = replay_rate / keep_rate
            row["played_positions_per_sec"] = played_rate

        average_sims = _number(quality.get("mcts_avg_sims")) if quality else None
        simulations_rate = _number(row.get("mcts_simulations_per_sec"))
        if simulations_rate is None and played_rate is not None and average_sims and average_sims > 0.0:
            simulations_rate = played_rate * average_sims
            row["mcts_simulations_per_sec"] = simulations_rate

        nn_evaluations_rate = _number(row.get("mcts_nn_evaluations_per_sec"))
        if nn_evaluations_rate is None:
            nn_items = _number(row.get("mcts_nn_inference_batch_items"))
            selfplay_seconds = _number(row.get("stage_selfplay_time_s"))
            if nn_items is not None and selfplay_seconds and selfplay_seconds > 0.0:
                row["mcts_nn_evaluations_per_sec"] = nn_items / selfplay_seconds

        traversal_rate = _number(row.get("mcts_selection_node_traversals_per_sec"))
        if traversal_rate is None:
            traversal_rate = _number(row.get("mcts_nodes_per_sec"))  # schema <= 6
            if traversal_rate is not None:
                row["mcts_selection_node_traversals_per_sec"] = traversal_rate
        if traversal_rate is not None and simulations_rate is not None and simulations_rate > 0.0:
            row["mcts_selection_path_length"] = traversal_rate / simulations_rate

        # This rate is non-additive. Older worker aggregation summed it across
        # chunks, producing values such as 18.0 and a misleading 1800% axis.
        native_tree_leaves = _number(row.get("mcts_native_tree_selected_leaves"))
        python_tree_leaves = _number(row.get("mcts_python_tree_selected_leaves"))
        if native_tree_leaves is not None or python_tree_leaves is not None:
            native_tree_leaves = max(0.0, native_tree_leaves or 0.0)
            python_tree_leaves = max(0.0, python_tree_leaves or 0.0)
            tree_leaves = native_tree_leaves + python_tree_leaves
            row["mcts_native_tree_selection_rate"] = (
                native_tree_leaves / tree_leaves if tree_leaves > 0.0 else 0.0
            )

    sections = [
        ("Time cost summaries", [(6, index) for index in range(4)], 4),
        ("Exclusive iteration wall time", [(7, index) for index in range(4)], 2),
        ("MCTS worker time costs", [(8, index) for index in range(4)], 2),
        ("Search and neural throughput", [(9, index) for index in range(3)], 3),
        ("Self-play throughput and iteration cost", [(0, 0), (0, 1)], 2),
        ("Inference latency and server phases", [(1, 1), (1, 2)], 2),
        ("Batching and cache efficiency", [(2, 2), (3, 0)], 2),
        ("MCTS worker cost and native tree", [(0, 2), (2, 1), (3, 1)], 3),
        ("Self-play worker balance", [(4, 0), (4, 1)], 2),
        ("Per-request latency trends", [(5, 0), (5, 1)], 2),
    ]
    fig, axes = create_dashboard(sections)

    _overall_runtime_donut(axes[6, 0], rows)
    _selfplay_worker_search_donut(axes[6, 1], rows)
    _cpu_path_timer_donut(axes[6, 2], rows)
    _gpu_path_timer_donut(axes[6, 3], rows)

    exact_stage_columns, exact_stage_rows = _complete_exclusive_stage_rows(rows)
    stage_specs = [
        (column, _exclusive_stage_labels(column), _COLORS[index % len(_COLORS)])
        for index, column in enumerate(exact_stage_columns)
    ] + [("unaccounted", "unaccounted", "#cbd5e1")]

    def _exact_stage_values(row):
        iteration = _number(row.get("iteration"))
        total = _number(row.get("iteration_total_time_s"))
        values = {column: _number(row.get(column)) for column in exact_stage_columns}
        if iteration is None or total is None or not exact_stage_columns:
            return None
        if any(value is None or value < 0.0 for value in values.values()):
            return None
        residual = total - sum(values.values())
        if residual < -0.01:
            return None
        values["unaccounted"] = max(0.0, residual)
        return values

    _stacked_iteration_bars(
        axes[7, 0], rows, stage_specs, _exact_stage_values,
        title="Exact stage wall time per iteration",
        ylabel="exclusive seconds / iteration", legend_columns=4,
    )
    axes[7, 0].text(
        0.02, 0.97,
        f"Complete timer rows only ({len(exact_stage_rows)}/{len(rows)}); exclusive stage counters plus measured residual.",
        transform=axes[7, 0].transAxes, ha="left", va="top", fontsize=6.2, color="#6b7280",
    )

    def _stage_share_values(row):
        values = _exact_stage_values(row)
        total = _number(row.get("iteration_total_time_s"))
        if values is None or total is None or total <= 0.0:
            return None
        return {key: 100.0 * value / total for key, value in values.items()}

    _stacked_iteration_bars(
        axes[7, 1], rows, stage_specs, _stage_share_values,
        title="Iteration wall-time composition",
        ylabel="share of logged iteration total", legend_columns=4,
    )
    axes[7, 1].set_ylim(0.0, 100.0)
    axes[7, 1].text(
        0.02, 0.97, "100% bars use the same complete rows as the seconds stack.",
        transform=axes[7, 1].transAxes, ha="left", va="top", fontsize=6.2, color="#6b7280",
    )

    eval_specs = [
        (column, _exclusive_stage_labels(column), _COLORS[index % len(_COLORS)])
        for index, column in enumerate(exact_stage_columns)
        if column in {
            "stage_regular_eval_time_s", "stage_promotion_eval_time_s",
            "stage_promotion_confirmation_time_s", "stage_elo_eval_time_s",
            "stage_eval_log_time_s",
        }
    ]
    def _eval_stage_values(row):
        values = _exact_stage_values(row)
        if values is None or not eval_specs:
            return None
        return values if any(values[column] > 0.0 for column, _, _ in eval_specs) else None
    _stacked_iteration_bars(
        axes[7, 2], rows, eval_specs, _eval_stage_values,
        title="Evaluation time per iteration",
        ylabel="exclusive seconds / iteration", legend_columns=3,
    )
    axes[7, 2].set_xticks([
        int(_number(row.get("iteration")))
        for row in rows
        if _eval_stage_values(row) is not None
    ])
    axes[7, 2].text(
        0.02, 0.97, "Iterations with recorded evaluation time only; complete rows only.",
        transform=axes[7, 2].transAxes, ha="left", va="top", fontsize=6.1, color="#6b7280",
    )

    ax = axes[7, 3]
    _style(ax, "Accumulated exclusive stage cost")
    if exact_stage_rows:
        totals = {column: 0.0 for column in exact_stage_columns}
        totals["unaccounted"] = 0.0
        for _, _, values in exact_stage_rows:
            for column, value in values.items():
                totals[column] += value
        ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        visible = [(key, value) for key, value in ranked if value > 0.0]
        if visible:
            labels = [_exclusive_stage_labels(key) for key, _ in visible]
            costs = [value for _, value in visible]
            ax.barh(labels, costs, color=[_COLORS[i % len(_COLORS)] for i in range(len(labels))])
            ax.invert_yaxis()
            ax.set_xlabel("summed seconds", fontsize=7.5)
            ax.set_xlim(0.0, max(costs) * 1.42)
            ax.tick_params(axis="y", labelsize=6.8)
            for y, (_, value) in enumerate(visible):
                ax.text(value, y, f"  {value:.1f}s  |  {value / len(exact_stage_rows):.1f}s/iter", va="center", fontsize=6.2)
            ax.text(0.02, 0.97, f"{len(exact_stage_rows)} complete iterations; ranked by accumulated wall time.",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#6b7280")
        else:
            _no_data(ax, "No positive stage costs")
    else:
        _no_data(ax, "Complete exclusive stage telemetry unavailable")

    ax = axes[8, 0]
    _style(ax, "Iteration cost by phase over time")
    phase_specs = (
        ("selfplay", "self-play", "#2563eb"),
        ("replay_setup", "setup / replay / reanalysis", "#0891b2"),
        ("train", "training", "#16a34a"),
        ("evaluation", "evaluation", "#f59e0b"),
        ("elo_eval", "Elo evaluation", "#ea580c"),
        ("maintenance", "logging / checkpoint / GC", "#7c3aed"),
    )
    iterations, _, stage_values, _ = _runtime_composition_data(rows)
    shown = False
    for key, label, color in phase_specs:
        ys = stage_values[key]
        if any(y is not None for y in ys):
            ax.plot(iterations, [np.nan if y is None else y for y in ys],
                    marker="o", ms=2.5, lw=1.3, color=color, label=label)
            shown = True
    ax.set_ylabel("seconds / iteration", fontsize=7.5)
    _legend_below(ax, ncol=3)
    if shown:
        ax.text(0.02, 0.97, "Measured exclusive phase wall timers; gaps mean the timer was unavailable.",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#6b7280")
    else:
        _no_data(ax, "Exclusive phase timers unavailable")

    ax = axes[8, 1]
    search_specs = [
        ("nn_wait", "NN inference wait", "#db2777"),
        ("selection", "tree selection", "#2563eb"),
        ("backprop", "backpropagation", "#16a34a"),
        ("remainder", "other search outside NN wait", "#64748b"),
    ]
    def _search_worker_parts(row):
        search = _number(row.get("mcts_search_many_time_s"))
        wait = _number(row.get("mcts_nn_inference_time_s"))
        selection = _number(row.get("mcts_search_selection_time_s"))
        backprop = _number(row.get("mcts_search_backprop_time_s"))
        values = (search, wait, selection, backprop)
        if any(value is None or value < 0.0 for value in values) or wait + selection + backprop > search + 1e-6:
            return None
        return {
            "nn_wait": wait, "selection": selection, "backprop": backprop,
            "remainder": max(0.0, search - wait - selection - backprop),
        }
    _stacked_iteration_bars(
        ax, rows, search_specs, _search_worker_parts,
        title="Self-play search cost per iteration",
        ylabel="summed worker-seconds", legend_columns=2,
    )
    ax.text(0.02, 0.97, "Valid complete search partitions only; worker-seconds sum across workers, not elapsed time.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#6b7280")

    ax = axes[8, 2]
    _style(ax, "Ranked MCTS worker timer cost")
    timer_totals = []
    for column, label, color in _OVERLAPPING_WORKER_TIMERS:
        values = [_number(row.get(column)) for row in rows]
        values = [value for value in values if value is not None and value >= 0.0]
        if values and sum(values) > 0.0:
            timer_totals.append((label, sum(values), len(values), color))
    timer_totals.sort(key=lambda item: item[1], reverse=True)
    if timer_totals:
        labels = [item[0] for item in timer_totals]
        totals = [item[1] for item in timer_totals]
        ax.barh(labels, totals, color=[item[3] for item in timer_totals])
        ax.invert_yaxis()
        ax.set_xlabel("summed worker-seconds", fontsize=7.5)
        ax.set_xlim(0.0, max(totals) * 1.42)
        ax.tick_params(axis="y", labelsize=6.8)
        for y, (_, total, count, _) in enumerate(timer_totals):
            ax.text(total, y, f"  {total:.0f}s | {total / count:.1f}s/observed iter", va="center", fontsize=6.2)
        ax.text(0.02, 0.97, "Overlapping counters—not a partition. Each row is an independently measured timer sum.",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#9a3412")
    else:
        _no_data(ax, "MCTS worker timers unavailable")

    ax = axes[8, 3]
    _style(ax, "Absolute worker hot-path timers")
    shown = False
    for index, (column, label, color) in enumerate(_OVERLAPPING_WORKER_TIMERS[2:]):
        xs, ys = _series(rows, column)
        if xs:
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, label=label, color=color)
            shown = True
    ax.set_ylabel("summed worker-seconds / iteration", fontsize=7.1)
    _legend_below(ax, ncol=2)
    if shown:
        ax.text(0.02, 0.97, "Overlapping counters—not a partition; compare absolute cost across iterations.",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#9a3412")
    else:
        _no_data(ax, "Detailed worker hot-path timers unavailable")

    ax = axes[9, 0]
    _style(ax, "MCTS NPS: fresh and effective search visits")
    fresh_shown = _line(
        ax, rows, "logged_fresh_mcts_nps", "fresh completed simulations/s", "#2563eb"
    )
    effective_shown = _line(
        ax, rows, "logged_effective_mcts_nps", "effective visits/s (incl. inherited credit)", "#9333ea"
    )
    ax.set_ylabel("MCTS simulations / visits per second", fontsize=7.2)
    _legend_below(ax, ncol=2)
    if fresh_shown or effective_shown:
        fresh_mean = _weighted_rate_summary(
            rows, "logged_fresh_mcts_nps", "stage_selfplay_time_s"
        )
        effective_mean = _weighted_rate_summary(
            rows, "logged_effective_mcts_nps", "stage_selfplay_time_s"
        )
        note = "Fresh NPS counts newly completed simulations; effective visits include inherited credit."
        if fresh_mean is not None:
            note += f" Wall-time weighted fresh mean: {fresh_mean:,.0f}/s."
        if effective_mean is not None and effective_shown:
            note += f" Effective mean: {effective_mean:,.0f}/s."
        ax.text(0.02, 0.97, textwrap.fill(note, width=64), transform=ax.transAxes,
                ha="left", va="top", fontsize=6.1, color="#536273")
    else:
        _no_data(ax, "Logged fresh/effective MCTS throughput unavailable")

    ax = axes[9, 1]
    _style(ax, "Neural inference throughput")
    req_x, req_rates, req_count, req_seconds = _count_rate_data(
        rows, "mcts_nn_inference_batch_items", "stage_selfplay_time_s"
    )
    central_x, central_rates, central_count, central_seconds = _count_rate_data(
        rows, "central_nn_evaluated_positions", "stage_selfplay_time_s"
    )
    if req_x:
        ax.plot(req_x, req_rates, marker="o", ms=3, lw=1.5, color="#0891b2",
                label="MCTS NN positions requested/s")
    if central_x:
        ax.plot(central_x, central_rates, marker="o", ms=3, lw=1.5, color="#16a34a",
                label="central fresh NN evaluations/s")
    ax.set_ylabel("positions / self-play wall-second", fontsize=7.2)
    _legend_below(ax, ncol=2)
    if req_x or central_x:
        cfg = _rl_config(main_csv_path)
        configured_central = cfg.get("self_play_central_inference_enabled") is True
        device = str(cfg.get("self_play_device") or "").strip().lower()
        gpu_confirmed = configured_central and device.startswith(("cuda", "gpu"))
        note = "MCTS counts include requested leaf items; central count is fresh server work after cache/dedup."
        if gpu_confirmed:
            note += " Run metadata confirms a CUDA central backend; throughput is not GPU utilization."
        elif central_x:
            note += " Central device is not confirmed as GPU in run metadata."
        if req_seconds > 0.0:
            note += f" Pooled requested rate: {req_count / req_seconds:,.0f}/s."
        if central_seconds > 0.0:
            note += f" Pooled central fresh rate: {central_count / central_seconds:,.0f}/s."
        ax.text(0.02, 0.97, textwrap.fill(note, width=64), transform=ax.transAxes,
                ha="left", va="top", fontsize=6.1, color="#536273")
    else:
        _no_data(ax, "Measured NN position counts and self-play wall time unavailable")

    ax = axes[9, 2]
    _style(ax, "MCTS selection throughput and path length")
    traversals_shown = _line(
        ax, rows, "mcts_selection_node_traversals_per_sec",
        "selection-node traversals/s", "#ea580c"
    )
    path_x, path_y = [], []
    for row in rows:
        iteration = _number(row.get("iteration"))
        traversals = _number(row.get("mcts_selection_node_traversals_per_sec"))
        fresh_rate = _number(row.get("logged_fresh_mcts_nps"))
        if iteration is None or traversals is None or traversals < 0.0 or fresh_rate is None or fresh_rate <= 0.0:
            continue
        path_x.append(int(iteration))
        path_y.append(traversals / fresh_rate)
    if path_x:
        ax2 = ax.twinx()
        ax2.tick_params(labelsize=8)
        ax2.plot(path_x, path_y, marker="s", ms=2.8, lw=1.2, color="#7c3aed",
                 label="traversals per fresh simulation")
        ax2.set_ylabel("traversals / fresh simulation", fontsize=7.0, color="#7c3aed")
        _legend_below(ax, ax2, ncol=2)
    else:
        ax.set_ylabel("selection-node traversals / second", fontsize=7.2)
        _legend_below(ax, ncol=2)
    if traversals_shown or path_x:
        ax.text(0.02, 0.97, textwrap.fill(
                    "Path length uses directly logged fresh MCTS NPS; gaps mean missing telemetry.",
                    width=64,
                ),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#536273")
    else:
        _no_data(ax, "Selection traversal throughput unavailable")

    ax = axes[0, 0]
    _style(ax, "Self-play throughput")
    shown = _line(ax, rows, "replay_positions_per_sec", "replay positions/s", _COLORS[0])
    shown |= _line(ax, rows, "played_positions_per_sec", "played positions/s", _COLORS[2])
    ax.set_ylabel("positions/s", fontsize=8)
    _legend_below(ax, ncol=2)
    if not shown: _no_data(ax)
    latest_path = _last(rows, "mcts_selection_path_length")
    if latest_path is not None:
        ax.text(
            0.98, 0.04, f"latest selection path: {latest_path:.2f} traversals/sim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#4b5563",
        )

    stage_specs = _RUNTIME_STAGE_SPECS
    iterations, totals, stage_values, other_values = _runtime_composition_data(rows)

    ax = axes[0, 1]
    _style(ax, "Runtime composition per iteration")
    has_stage_values = any(
        value is not None
        for values in stage_values.values()
        for value in values
    )
    if iterations and (any(value is not None and value > 0.0 for value in totals) or has_stage_values):
        bottoms = [0.0] * len(iterations)
        runtime_total = sum(value for value in totals if value is not None)
        visible_stage_count = 0
        plotted_stack = [0.0] * len(iterations)
        for column, label, color in stage_specs:
            if column == "other":
                continue
            raw_values = stage_values[column]
            values = [np.nan if value is None else value for value in raw_values]
            component_total = sum(value for value in raw_values if value is not None)
            if component_total < 1.0 or (
                runtime_total > 0.0 and component_total / runtime_total < 0.002
            ):
                continue
            ax.bar(iterations, values, bottom=bottoms, width=0.76, color=color,
                   label=label, edgecolor="white", linewidth=0.35)
            bottoms = [
                bottom + (value if np.isfinite(value) else 0.0)
                for bottom, value in zip(bottoms, values)
            ]
            plotted_stack = [
                current + (value if np.isfinite(value) else 0.0)
                for current, value in zip(plotted_stack, values)
            ]
            visible_stage_count += 1
        known_residuals = [value for value in other_values if value is not None]
        other_total = sum(known_residuals)
        if other_total >= 1.0 and (runtime_total <= 0.0 or other_total / runtime_total >= 0.002):
            residual_plot = [np.nan if value is None else value for value in other_values]
            ax.bar(iterations, residual_plot, bottom=bottoms, width=0.76, color="#cbd5e1",
                   label="unaccounted", edgecolor="white", linewidth=0.35, hatch="//")
            plotted_stack = [
                current + (value if np.isfinite(value) else 0.0)
                for current, value in zip(plotted_stack, residual_plot)
            ]
            visible_stage_count += 1
        # Iteration total is the reference line. It exposes a mismatch when
        # timer rows are incomplete or exclusive stage accounting drifts.
        logged_totals = [
            (iteration, total) for iteration, total in zip(iterations, totals)
            if total is not None
        ]
        if logged_totals:
            ax.plot([item[0] for item in logged_totals], [item[1] for item in logged_totals],
                    color="#111827", lw=1.0, ls="--", marker="o", ms=2.3,
                    label="iteration total", zorder=5)
        if len(logged_totals) <= 12 and logged_totals:
            padding = max(value for _, value in logged_totals) * 0.015
            for iteration, total in logged_totals:
                ax.text(iteration, total + padding, f"{total:.0f}s", ha="center", va="bottom",
                        fontsize=6.5, color="#374151")
        max_logged_total = max((value for value in totals if value is not None), default=0.0)
        ax.set_ylim(0.0, max(max_logged_total, max(plotted_stack)) * 1.10)
        ax.set_ylabel("seconds", fontsize=8)
        ax.set_axisbelow(True)
        if any(value is None for value in other_values):
            ax.text(0.02, 0.97, "unaccounted: unknown where total or a stage timer is missing",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                    color="#6b7280")
        elif other_total < 1.0 or (runtime_total > 0.0 and other_total / runtime_total < 0.002):
            ax.text(0.02, 0.97, "unaccounted ≈ 0: logged stages cover iteration time",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                    color="#6b7280")
        _legend_below(ax, ncol=min(4, max(1, visible_stage_count)))
    else:
        _no_data(ax)

    ax = axes[0, 2]
    _style(ax, "Worker time per 1k NN evaluations")
    active_x, active_y, wait_x, wait_y = [], [], [], []
    inconsistent = 0
    for row in rows:
        iteration = _number(row.get("iteration"))
        positions = _number(row.get("mcts_nn_inference_batch_items"))
        search = _number(row.get("mcts_search_many_time_s"))
        wait = _number(row.get("mcts_nn_inference_time_s"))
        if iteration is None or positions is None or positions <= 0 or search is None or wait is None:
            continue
        if wait < 0.0 or search < 0.0 or wait > search + 0.01:
            inconsistent += 1
            continue
        factor = 1e6 / positions
        wait_x.append(int(iteration)); wait_y.append(wait * factor)
        active_x.append(int(iteration)); active_y.append(max(0.0, search - wait) * factor)
    if wait_x:
        ax.plot(wait_x, wait_y, marker="o", ms=3, lw=1.6, color="#db2777",
                label="waiting for NN")
        ax.plot(active_x, active_y, marker="o", ms=3, lw=1.6, color="#2563eb",
                label="search outside NN wait")
        _legend_below(ax, ncol=2)
        ax.set_ylabel("worker ms / 1k NN evaluations", fontsize=8)
        ax.text(0.02, 0.97, "Both lines partition measured search time; summed across workers.",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color="#6b7280")
        if inconsistent:
            ax.text(0.02, 0.89, f"omitted {inconsistent} rows with inconsistent timers",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                    color="#9a3412")
    else:
        _no_data(ax, "Search and NN wait timers unavailable")

    def _central_latency_parts(row):
        required = (
            "central_remote_wait_ms_per_request",
            "central_request_submit_ms_per_request",
            "central_server_queue_wait_ms_per_request",
            "central_server_pipeline_wait_ms_per_request",
            "central_server_concat_ms_per_request",
            "central_server_cache_lookup_ms_per_request",
            "central_server_staging_copy_ms_per_request",
            "central_server_h2d_ms_per_request",
            "central_server_forward_ms_per_request",
            "central_server_d2h_ms_per_request",
            "central_server_total_ms_per_request",
        )
        if any(_number(row.get(column)) is None for column in required):
            return None
        remote = max(0.0, _number(row.get("central_remote_wait_ms_per_request")))
        queue = max(0.0, _number(row.get("central_server_queue_wait_ms_per_request")))
        server_total = max(0.0, _number(row.get("central_server_total_ms_per_request")))
        descriptor_raw = _number(row.get("central_descriptor_queue_wait_ms_per_request"))
        coalesce_raw = _number(row.get("central_batch_coalesce_wait_ms_per_request"))
        request_submit = max(0.0, _number(row.get("central_request_submit_ms_per_request")))
        if descriptor_raw is None and coalesce_raw is None:
            descriptor_queue = queue
            batch_coalesce = 0.0
        else:
            descriptor_queue = max(
                0.0,
                descriptor_raw
                if descriptor_raw is not None
                else queue - max(0.0, coalesce_raw or 0.0),
            )
            batch_coalesce = max(
                0.0,
                coalesce_raw
                if coalesce_raw is not None
                else queue - descriptor_queue,
            )
        request_submit = min(request_submit, descriptor_queue)
        parts = {
            "request_submit": request_submit,
            "descriptor_queue": max(0.0, descriptor_queue - request_submit),
            "batch_coalesce": batch_coalesce,
            "pipeline_wait": max(
                0.0,
                _number(row.get("central_server_pipeline_wait_ms_per_request")) or 0.0,
            ),
            "concat": max(0.0, _number(row.get("central_server_concat_ms_per_request")) or 0.0),
            "cache": max(0.0, _number(row.get("central_server_cache_lookup_ms_per_request")) or 0.0),
            "staging": max(0.0, _number(row.get("central_server_staging_copy_ms_per_request")) or 0.0),
            "h2d": max(0.0, _number(row.get("central_server_h2d_ms_per_request")) or 0.0),
            "forward": max(0.0, _number(row.get("central_server_forward_ms_per_request")) or 0.0),
            "d2h": max(0.0, _number(row.get("central_server_d2h_ms_per_request")) or 0.0),
        }
        known_server = sum(
            parts[key]
            for key in (
                "pipeline_wait", "concat", "cache", "staging",
                "h2d", "forward", "d2h",
            )
        )
        logged_server_other = _number(row.get("central_server_other_ms_per_request"))
        logged_ipc_other = _number(row.get("central_worker_ipc_ms_per_request"))
        parts["server_other"] = max(0.0, logged_server_other if logged_server_other is not None else server_total - known_server)
        parts["ipc_other"] = max(0.0, logged_ipc_other if logged_ipc_other is not None else remote - queue - server_total)
        return parts

    ax = axes[1, 1]
    position_latency_specs = (
        ("waiting", "queue / batch wait", "#f59e0b"),
        ("transfer", "data movement", "#0891b2"),
        ("forward", "GPU forward", "#16a34a"),
        ("remainder", "IPC / other", "#64748b"),
    )

    def _position_latency_parts(row):
        worker_batch = _number(row.get("mcts_avg_batch_size"))
        worker_wait = _number(row.get("mcts_worker_nn_wait_ms_per_position"))
        if worker_batch is None or worker_batch <= 0.0 or worker_wait is None:
            return None
        central = _central_latency_parts(row)
        if central is None:
            return None
        waiting = sum(central[key] for key in (
            "request_submit", "descriptor_queue", "batch_coalesce", "pipeline_wait",
        ))
        transfer = sum(central[key] for key in (
            "concat", "cache", "staging", "h2d", "d2h",
        ))
        remainder = central["server_other"] + central["ipc_other"]
        return {
            "waiting": waiting / worker_batch,
            "transfer": transfer / worker_batch,
            "forward": central["forward"] / worker_batch,
            "remainder": remainder / worker_batch,
        }

    _stacked_iteration_bars(
        ax,
        rows,
        position_latency_specs,
        _position_latency_parts,
        title="Latency composition per position",
        ylabel="ms / position",
        legend_columns=4,
    )

    def _latest_central_server_donut(ax, source_rows):
        component_columns = (
            ("central_server_pipeline_wait_ms_per_request", "pipeline wait", "#c2410c"),
            ("central_server_concat_ms_per_request", "concat", "#0891b2"),
            ("central_server_cache_lookup_ms_per_request", "cache lookup", "#84cc16"),
            ("central_server_staging_copy_ms_per_request", "staging copy", "#06b6d4"),
            ("central_server_h2d_ms_per_request", "H2D", "#2563eb"),
            ("central_server_forward_ms_per_request", "forward", "#16a34a"),
            ("central_server_d2h_ms_per_request", "D2H", "#7c3aed"),
        )
        required = tuple(column for column, _, _ in component_columns) + (
            "central_server_total_ms_per_request",
        )
        averaged = []
        latest_iteration = None
        for row in reversed(source_rows):
            values = [_number(row.get(column)) for column in required]
            if any(value is None or value < 0.0 for value in values):
                continue
            total = values[-1]
            components = values[:-1]
            residual = total - sum(components)
            if residual < -0.01:
                continue
            averaged = [
                (column, label, value, color)
                for (column, label, color), value in zip(component_columns, components)
            ]
            averaged.append(("other", "other server work", max(0.0, residual), "#64748b"))
            latest_iteration = _number(row.get("iteration"))
            break
        _composition_donut(
            ax,
            averaged,
            title="Central server phases per request",
            center_label=(
                "latest\nms / request"
                if latest_iteration is None
                else f"iteration {latest_iteration:g}\nms / request"
            ),
            unit="ms",
            decimals=2,
            min_share=0.005,
            min_value=0.05,
        )
        if averaged:
            _donut_note(ax, "Host-observed server phases only; excludes client and queue wait.")
        return bool(averaged)

    _latest_central_server_donut(axes[1, 2], rows)

    for row in rows:
        item_count = _number(row.get("mcts_nn_inference_batch_items"))
        if item_count is None or item_count <= 0.0:
            continue
        for source, target in (
            ("mcts_search_selection_time_s", "hot_selection_ms_per_1k"),
            ("mcts_board_to_tensor_time_s", "hot_encoding_ms_per_1k"),
            ("mcts_batch_expand_tensor_pack_time_s", "hot_pack_ms_per_1k"),
            ("mcts_batch_expand_cpu_policy_time_s", "hot_policy_ms_per_1k"),
            ("mcts_batch_expand_move_index_time_s", "hot_move_ms_per_1k"),
            ("mcts_native_tree_import_time_s", "hot_native_import_ms_per_1k"),
            ("mcts_native_tree_sync_time_s", "hot_native_sync_ms_per_1k"),
        ):
            seconds = _number(row.get(source))
            if seconds is not None and seconds >= 0.0:
                row[target] = 1e6 * seconds / item_count
    ax = axes[2, 1]
    _style(ax, "MCTS CPU hot paths and native tree")
    hot = False
    for column, label, color in (
        ("hot_selection_ms_per_1k", "tree selection", _COLORS[0]),
        ("hot_encoding_ms_per_1k", "board encoding", _COLORS[2]),
        ("hot_pack_ms_per_1k", "input packing", "#06b6d4"),
        ("hot_policy_ms_per_1k", "CPU policy", _COLORS[1]),
        ("hot_move_ms_per_1k", "move encoding", _COLORS[4]),
        ("hot_native_import_ms_per_1k", "native import", "#9333ea"),
        ("hot_native_sync_ms_per_1k", "native sync", "#f59e0b"),
    ):
        hot |= _line(ax, rows, column, label, color)
    ax.set_ylabel("worker ms / 1k NN positions", fontsize=8)
    ax.text(
        0.02, 0.97,
        "Timer counters may overlap; compare trends, not totals.",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
        color="#6b7280",
    )
    _legend_below(ax, ncol=3)
    if not hot:
        _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "Batching and worker NN wait")
    shown = _line(ax, rows, "mcts_avg_batch_size", "worker batch", _COLORS[0])
    shown |= _line(
        ax,
        rows,
        "mcts_central_avg_batch_size",
        "shared batch seen/request",
        _COLORS[2],
    )
    shown |= _line(
        ax,
        rows,
        "central_batch_target_avg",
        "auto batch target",
        "#0f766e",
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(100.0))
    wait_share = _line(
        ax2, rows, "mcts_worker_nn_wait_share_pct", "worker NN wait share", _COLORS[3]
    )
    _legend_below(ax, ax2, ncol=4)
    if not (shown or wait_share):
        _no_data(ax)

    ax = axes[3, 0]
    _style(ax, "Inference cache and GPU output efficiency", percent=True)
    efficiency_shown = False
    for column, label, color in (
        ("central_nn_saved_overall_rate", "NN evals saved overall", "#16a34a"),
        ("central_cache_active_fraction", "cache active coverage", "#0891b2"),
        ("central_gpu_batch_fill", "GPU batch fill", "#7c3aed"),
        ("central_compact_policy_rate", "legal-only GPU output", "#2563eb"),
        ("central_output_pipeline_rate", "overlapped GPU output", "#db2777"),
    ):
        efficiency_shown |= _line(ax, rows, column, label, color)
    _legend_below(ax, ncol=3)
    if not efficiency_shown:
        _no_data(ax, "Central inference efficiency telemetry unavailable")

    ax = axes[3, 1]
    _style(ax, "Native tree execution", percent=True)
    native_shown = False
    for column, label, color in (
        ("mcts_native_root_selection_rate", "native root selection", "#ea580c"),
        ("mcts_native_completed_q_rate", "native completed-Q", "#dc2626"),
        ("mcts_native_tree_selection_rate", "native full-tree traversal", "#9333ea"),
    ):
        native_shown |= _line(ax, rows, column, label, color)
    _legend_below(ax, ncol=3)
    if not native_shown:
        _no_data(ax, "Native tree counters unavailable")

    ax = axes[4, 0]
    _style(ax, "Self-play games completed per worker")
    shown = False
    for column, label, color in (
        ("selfplay_games_per_worker_min", "minimum", _COLORS[1]),
        ("selfplay_games_per_worker_mean", "mean", _COLORS[0]),
        ("selfplay_games_per_worker_max", "maximum", _COLORS[2]),
    ):
        shown |= _line(ax, rows, column, label, color)
    ax.set_ylabel("completed games / worker", fontsize=7.5)
    _legend_below(ax, ncol=3)
    if shown:
        latest_workers = _last(rows, "selfplay_worker_count")
        if latest_workers is not None:
            ax.text(
                0.98, 0.96, f"latest workers={latest_workers:.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#4b5563",
            )
    else:
        _no_data(ax, "Per-worker completed-game distribution unavailable")

    ax = axes[4, 1]
    _style(ax, "Worker completion spread and final 10% tail")
    shown = _line(
        ax, rows, "selfplay_worker_finish_spread_s", "worker finish-time spread", _COLORS[0],
    )
    shown |= _line(
        ax, rows, "selfplay_tail_10pct_time_s", "time after 90% games complete", _COLORS[4],
        ls="--",
    )
    ax.set_ylabel("seconds", fontsize=7.5)
    _legend_below(ax, ncol=2)
    if shown:
        ax.text(
            0.02, 0.97,
            "Separate worker scheduling measures; neither is an iteration-time component.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="#4b5563",
        )
    else:
        _no_data(ax, "Worker completion timing unavailable")

    ax = axes[5, 0]
    _style(ax, "Request queue, wait and IPC latency")
    shown = False
    for column, label, color, style in (
        ("central_remote_wait_ms_per_request", "worker remote wait", _COLORS[0], "-"),
        ("central_server_queue_wait_ms_per_request", "server queue wait", _COLORS[1], "-"),
        ("central_descriptor_queue_wait_ms_per_request", "descriptor queue", _COLORS[2], "--"),
        ("central_batch_coalesce_wait_ms_per_request", "batch coalescing", _COLORS[3], "--"),
        ("central_worker_ipc_ms_per_request", "worker IPC", _COLORS[4], ":"),
    ):
        shown |= _line(ax, rows, column, label, color, ls=style)
    ax.set_ylabel("ms / request", fontsize=7.5)
    _legend_below(ax, ncol=2)
    if shown:
        ax.text(
            0.02, 0.97, "Independent path timers; do not add these lines.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#4b5563",
        )
    else:
        _no_data(ax, "Request queue and IPC timing unavailable")

    ax = axes[5, 1]
    _style(ax, "Central inference phase latency over time")
    shown = False
    for column, label, color, style in (
        ("central_server_total_ms_per_request", "server total", "#111827", "-"),
        ("central_server_forward_ms_per_request", "forward", _COLORS[2], "-"),
        ("central_server_h2d_ms_per_request", "H2D", _COLORS[0], "--"),
        ("central_server_d2h_ms_per_request", "D2H", _COLORS[3], "--"),
        ("central_server_cache_lookup_ms_per_request", "cache lookup", _COLORS[4], ":"),
        ("central_server_staging_copy_ms_per_request", "staging copy", _COLORS[5], ":"),
    ):
        shown |= _line(ax, rows, column, label, color, ls=style)
    ax.set_ylabel("host-timed ms / request", fontsize=7.2)
    _legend_below(ax, ncol=3)
    if shown:
        ax.text(
            0.02, 0.97,
            "Trends only, not a stack; GPU forward is host-observed timing.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="#4b5563",
        )
    else:
        _no_data(ax, "Central inference phase timing unavailable")

    _mark_resume_boundaries(
        {
            key: ax for key, ax in axes.items()
            if key not in {
                (1, 2),  # central server donut
                (6, 0), (6, 1), (6, 2), (6, 3),  # timing donuts
                (7, 3), (8, 2),  # ranked cost summaries
            }
        },
        _resume_markers(main_csv_path or performance_csv_path),
    )
    _finish(
        fig,
        output_path,
        "RL details - performance",
        "NN wait is worker wall-time blocked on inference; central backlog is request submission-to-server-dequeue latency.",
        hspace=0.78,
    )
    return True
