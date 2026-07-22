"""Compact, decision-oriented dashboards for RL training CSV logs.

The three public renderers deliberately split responsibilities:

* main: strength, learning and promotion safety;
* data quality: replay, targets and MCTS behaviour;
* performance: throughput, latency and bottlenecks.

Keeping the plots here prevents the CSV logger from becoming a second plotting
application and makes it much harder to accidentally repeat the same metric on
several dashboards.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


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


def _regime_change_iterations(rows):
    """Return baseline promotions and learner/actor recoveries from old or new logs."""
    promotions = set()
    recoveries = set()
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        iteration = int(iteration)
        if (_number(row.get("rl_best_model")) or 0.0) > 0.0:
            promotions.add(iteration)
        actor_status = str(row.get("actor_status", "") or "").strip().lower()
        if (
            (_number(row.get("actor_recovered")) or 0.0) > 0.0
            or "recover" in actor_status
        ):
            recoveries.add(iteration)
    # Promotion already explains the regime change if both flags were emitted
    # for the same iteration; avoid drawing two coincident markers.
    recoveries.difference_update(promotions)
    return sorted(promotions), sorted(recoveries)


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


def _mean(rows, column, window=5):
    values = [_number(row.get(column)) for row in rows[-window:]]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _anchor_rows_with_reused_best(rows, first_promotion_iteration):
    """Backfill the pre-promotion anchor series from the identical best match.

    Before the first promotion, frozen best and the IL anchor are the same model.
    New logs already persist the reused result on every evaluation; this fallback
    also makes older logs render the complete pre-promotion history.
    """
    if first_promotion_iteration is None:
        return list(rows)
    column_map = {
        "anchor_games": "eval_games",
        "anchor_score_rate": "score_rate",
        "anchor_true_win_rate": "true_win_rate",
        "anchor_score_lower_bound": "eval_score_lower_bound",
        "anchor_no_mcts_games": "no_mcts_games",
        "anchor_no_mcts_score_rate": "no_mcts_score_rate",
        "anchor_no_mcts_score_lower_bound": "no_mcts_score_lower_bound",
        "anchor_mcts_no_mcts_gap": "mcts_no_mcts_gap",
    }
    result = []
    for row in rows:
        rendered = dict(row)
        iteration = _number(row.get("iteration"))
        if iteration is not None and iteration <= first_promotion_iteration:
            for anchor_column, best_column in column_map.items():
                if _number(rendered.get(anchor_column)) is None:
                    rendered[anchor_column] = rendered.get(best_column, "")
        result.append(rendered)
    return result


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


def _fill_between_with_breaks(ax, xs, lower, upper, *, break_after=None, **kwargs):
    offset = 0
    for segment_x, segment_lower in _split_series_at_boundaries(xs, lower, break_after):
        stop = offset + len(segment_x)
        ax.fill_between(segment_x, segment_lower, upper[offset:stop], **kwargs)
        offset = stop


def _stacked_iteration_bars(ax, rows, specs, value_resolver, *, title, ylabel, legend_columns=3):
    """Draw a compact stacked decomposition with one bar per iteration."""
    iterations = []
    series = {key: [] for key, _, _ in specs}
    totals = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        resolved = dict(value_resolver(row) or {})
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
            width=0.76,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.3,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

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


def _whole_run_composition(rows, specs, value_resolver, weight_column=None):
    """Aggregate every logged iteration using the same resolver as the bars.

    Runtime/CPU costs are summed. Per-request latency components use request-
    weighted means so a short iteration has no more influence than a long one.
    """
    selected_rows = list(rows)
    if not selected_rows:
        return [], 0
    sums = {key: 0.0 for key, _, _ in specs}
    total_weight = 0.0
    for row in selected_rows:
        weight = 1.0
        if weight_column is not None:
            weight = max(0.0, _number(row.get(weight_column)) or 0.0)
            if weight <= 0.0:
                continue
        resolved = dict(value_resolver(row) or {})
        for key in sums:
            sums[key] += weight * max(0.0, _number(resolved.get(key)) or 0.0)
        total_weight += weight
    divisor = total_weight if weight_column is not None else 1.0
    aggregated = [
        (key, label, sums[key] / max(1e-12, divisor), color)
        for key, label, color in specs
    ]
    return aggregated, len(selected_rows)


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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.04, ha="left", y=0.988)
    if subtitle:
        fig.text(0.04, 0.958, subtitle, fontsize=8.5, color="#4b5563", ha="left", va="top")
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.045,
        top=0.91,
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


def _nested_composition_donut(ax, inner, outer, *, title, inner_name, outer_name):
    """Compare two part-to-whole distributions without duplicating line charts."""
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    inner_values = [max(0.0, float(value or 0.0)) for _, value, _ in inner]
    outer_values = [max(0.0, float(value or 0.0)) for _, value, _ in outer]
    if sum(inner_values) <= 0.0 or sum(outer_values) <= 0.0:
        _no_data(ax)
        return False
    colors = [color for _, _, color in outer]
    outer_wedges, _ = ax.pie(
        outer_values,
        radius=1.0,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.pie(
        inner_values,
        radius=0.68,
        colors=[color for _, _, color in inner],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.text(0, 0.10, outer_name, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(0, -0.08, f"inner: {inner_name}", ha="center", va="center", fontsize=6.8, color="#4b5563")
    labels = [
        f"{label}: {100.0 * value / max(1e-12, sum(outer_values)):.1f}% {outer_name} / "
        f"{100.0 * inner_values[index] / max(1e-12, sum(inner_values)):.1f}% {inner_name}"
        for index, (label, value, _) in enumerate(outer)
    ]
    ax.legend(outer_wedges, labels, fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.02))
    ax.set_aspect("equal")
    return True


def _latest_quality_bars(ax, rows):
    specs = (
        ("replay keep", "selfplay_replay_storage_keep_rate", _COLORS[0]),
        ("champion share", "champion_replay_selected_fraction", "#7c3aed"),
        ("useful correction rows", "replay_policy_correction_fraction", "#db2777"),
        ("hard starts", "selfplay_hard_start_fraction", "#0f766e"),
        ("full search", "mcts_full_search_fraction", "#ca8a04"),
        ("useful corrections", "mcts_useful_change_rate", _COLORS[4]),
        ("search coverage", "mcts_visit_coverage_ratio_mean", _COLORS[2]),
        ("root-Q coverage", "root_q_coverage", _COLORS[5]),
    )
    available = [(label, _last(rows, column), color) for label, column, color in specs]
    available = [(label, value, color) for label, value, color in available if value is not None]
    _style(ax, "Latest safeguards and coverage", percent=True)
    if not available:
        _no_data(ax)
        return False
    labels = [label for label, _, _ in available]
    values = [max(0.0, min(1.0, value)) for _, value, _ in available]
    y = np.arange(len(labels))
    ax.barh(y, values, color=[color for _, _, color in available], alpha=0.88)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.05)
    ax.invert_yaxis()
    for row, value in zip(y, values):
        ax.text(min(1.01, value + 0.02), row, f"{100.0 * value:.1f}%", va="center", fontsize=7)
    return True


def _actor_guard_panel(ax, rows, promotion_iterations, recovery_iterations=()):
    """Show how far the accepted self-play actor trails the live learner."""
    _style(ax, "Actor freshness and guard actions")
    actor_x, actor_y = _series(rows, "actor_iteration")
    if not actor_x:
        _no_data(ax, "Actor telemetry is available from schema 12")
        return False

    ax.plot(
        actor_x,
        actor_x,
        color="#9ca3af",
        linestyle=":",
        linewidth=1.2,
        label="learner generation",
    )
    ax.step(
        actor_x,
        actor_y,
        where="post",
        color=_COLORS[0],
        linewidth=2.2,
        label="accepted actor generation",
    )
    ax.fill_between(
        actor_x,
        actor_y,
        actor_x,
        step="post",
        color=_COLORS[0],
        alpha=0.08,
        label="actor lag",
    )

    updated_x, updated_y = [], []
    for row in rows:
        iteration = _number(row.get("iteration"))
        actor_iteration = _number(row.get("actor_iteration"))
        if iteration is None or actor_iteration is None:
            continue
        if (_number(row.get("actor_updated")) or 0.0) > 0.0:
            updated_x.append(int(iteration)); updated_y.append(actor_iteration)
    if updated_x:
        ax.scatter(updated_x, updated_y, color=_COLORS[2], marker="o", s=42,
                   zorder=4, label="actor updated")

    latest_iteration = int(actor_x[-1])
    latest_actor = int(round(actor_y[-1]))
    latest_status = next(
        (str(row.get("actor_status", "")).strip() for row in reversed(rows)
         if str(row.get("actor_status", "")).strip()),
        "",
    )
    if len(latest_status) > 70:
        latest_status = latest_status[:67].rstrip() + "..."
    status_text = f"actor {latest_actor} | lag {max(0, latest_iteration - latest_actor)} iter"
    latest_reference = _last(rows, "actor_reference_score_rate")
    latest_delta = _last(rows, "actor_score_delta")
    if latest_reference is not None:
        status_text += f" | reference {100.0 * latest_reference:.1f}%"
    if latest_delta is not None:
        status_text += f" | learner delta {100.0 * latest_delta:+.1f}pp"
    if latest_status:
        status_text += f"\n{latest_status}"
    ax.text(0.02, 0.96, status_text, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color="#374151")
    ax.set_ylabel("generation iteration", fontsize=8)
    ax.set_ylim(bottom=min(0.0, min(actor_y) - 0.5))
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
    return True


def _task_gradient_interaction_panel(ax, rows, break_after=()):
    """Show whether policy and value agree on changes to the shared tower."""
    _style(ax, "Policy vs value gradient interaction")
    shown = _line(
        ax, rows, "grad_policy_probe_norm", "policy @ shared tower",
        _COLORS[0], lw=1.9, break_after=break_after,
    )
    shown |= _line(
        ax, rows, "grad_value_probe_norm", "value @ shared tower",
        _COLORS[3], lw=1.9, break_after=break_after,
    )
    if shown:
        ax.set_yscale("log")
        ax.set_ylabel("gradient norm (log)", fontsize=8)
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    cosine = _line(
        ax2, rows, "grad_policy_value_cosine", "policy/value cosine",
        _COLORS[4], ls="--", lw=1.8, break_after=break_after,
    )
    if cosine:
        ax2.set_ylim(-1.05, 1.05)
        ax2.axhline(0.0, color="#9ca3af", lw=1, ls=":")
        ax2.axhspan(-1.0, -0.20, color="#dc2626", alpha=0.035, zorder=0)
        ax2.axhspan(0.20, 1.0, color="#16a34a", alpha=0.035, zorder=0)
        ax2.set_ylabel("task cosine", fontsize=8, color=_COLORS[4])
        cosine_values = [
            value for value in (_number(row.get("grad_policy_value_cosine")) for row in rows)
            if value is not None
        ]
        latest_cosine = cosine_values[-1] if cosine_values else None
        mean_cosine = float(np.mean(cosine_values)) if cosine_values else None
        latest_policy = _last(rows, "grad_policy_probe_norm")
        latest_value = _last(rows, "grad_value_probe_norm")
        if latest_cosine is not None:
            state = (
                "aligned" if latest_cosine > 0.20
                else "conflicting" if latest_cosine < -0.20
                else "mostly independent"
            )
            summary = f"latest {latest_cosine:+.2f} ({state}) | mean {mean_cosine:+.2f}"
            if latest_policy is not None and latest_value is not None and latest_policy > 0.0:
                summary += f" | value/policy norm {latest_value / latest_policy:.1f}x"
            ax.text(
                0.02, 0.96, summary, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.2, color="#374151",
            )
    _legend_below(ax, ax2, ncol=2)
    if not (shown or cosine):
        _no_data(ax, "Gradient telemetry is available from schema 14")
    return bool(shown or cosine)


def _gradient_norms_panel(ax, rows, break_after=()):
    """Separate the update magnitude in the shared tower and both heads."""
    _style(ax, "Gradient norms and clipping")
    shown = _line(
        ax, rows, "grad_total_norm", "total before clip", "#111827", lw=2.0,
        break_after=break_after,
    )
    shown |= _line(
        ax, rows, "grad_backbone_norm", "shared tower", _COLORS[0], lw=1.8,
        break_after=break_after,
    )
    shown |= _line(
        ax, rows, "grad_policy_head_norm", "policy head", _COLORS[2], lw=1.8,
        break_after=break_after,
    )
    shown |= _line(
        ax, rows, "grad_value_head_norm", "value heads", _COLORS[3], lw=1.8,
        break_after=break_after,
    )
    if shown:
        ax.set_yscale("log")
        ax.set_ylabel("gradient norm (log)", fontsize=8)
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    clipped = _line(
        ax2, rows, "grad_clip_fraction", "batches clipped",
        _COLORS[1], ls="--", lw=1.7, break_after=break_after,
    )
    if clipped:
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("clipped batches", fontsize=8, color=_COLORS[1])
    latest_scale = _last(rows, "grad_clip_scale_mean")
    if latest_scale is not None:
        ax.text(
            0.02, 0.96, f"latest mean applied scale {latest_scale:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color="#374151",
        )
    _legend_below(ax, ax2, ncol=3)
    if not (shown or clipped):
        _no_data(ax, "Gradient telemetry is available from schema 14")
    return bool(shown or clipped)


def _objective_composition_panel(ax, rows, rl_cfg):
    """Show the positive, weighted terms that actually form the train objective."""
    policy_weight = float(rl_cfg.get("policy_loss_weight", 1.0) or 0.0)
    value_weight = float(rl_cfg.get("value_loss_weight", 1.0) or 0.0)
    scalar_weight = value_weight * float(rl_cfg.get("value_aux_scalar_loss_weight", 0.25) or 0.0)
    moves_weight = float(rl_cfg.get("moves_left_loss_weight", 0.05) or 0.0)
    search_q_weight = float(rl_cfg.get("search_q_loss_weight", 0.20) or 0.0)
    search_error_weight = float(rl_cfg.get("search_error_loss_weight", 0.05) or 0.0)

    def _values(row):
        def weighted(column, weight):
            return max(0.0, (_number(row.get(column)) or 0.0) * weight)
        return {
            "policy": weighted("policy_loss", policy_weight),
            "wdl": weighted("value_primary_loss", value_weight),
            "scalar": weighted("value_scalar_aux_loss", scalar_weight),
            "search_q": weighted("search_q_loss", search_q_weight),
            "search_error": weighted("search_error_loss", search_error_weight),
            "moves_left": weighted("moves_left_loss", moves_weight),
        }

    specs = (
        ("policy", "policy", _COLORS[0]),
        ("wdl", "WDL value", _COLORS[3]),
        ("scalar", "scalar value", "#c084fc"),
        ("search_q", "search-Q", _COLORS[2]),
        ("search_error", "Q error", "#f97316"),
        ("moves_left", "moves-left", _COLORS[1]),
    )
    shown = _stacked_iteration_bars(
        ax,
        rows,
        specs,
        _values,
        title="Weighted training objective",
        ylabel="positive loss contribution",
        legend_columns=4,
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
    rl_cfg = _rl_config(main_csv_path)
    if not main:
        return False

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.2))
    promotion_iterations, recovery_iterations = _regime_change_iterations(main)
    matchup_breaks = tuple(promotion_iterations) + tuple(recovery_iterations)
    first_promotion_iteration = min(promotion_iterations) if promotion_iterations else None

    ax = axes[0, 0]
    _style(ax, "Learner vs promoted best", percent=True)
    shown = _line(ax, main, "score_rate", "MCTS", _COLORS[0], break_after=matchup_breaks)
    shown |= _line(
        ax, main, "eval_score_lower_bound", "MCTS lower bound", _COLORS[0],
        ls=":", break_after=matchup_breaks,
    )
    shown |= _line(
        ax, main, "eval_score_rate_ema", "MCTS EMA", _COLORS[5],
        ls="--", lw=1.5, break_after=matchup_breaks,
    )
    shown |= _line(
        ax, main, "no_mcts_score_rate", "NN only", _COLORS[1],
        break_after=matchup_breaks,
    )
    shown |= _line(
        ax, main, "actor_reference_score_rate", "accepted actor reference",
        "#9333ea", ls="--", lw=1.6, break_after=matchup_breaks,
    )
    ax.axhline(0.5, color="#374151", lw=1, ls="--", label="break-even")
    promotion_score_floor = float(rl_cfg.get("score_rate_threshold", 0.55) or 0.55)
    actor_score_floor = float(rl_cfg.get("actor_gate_score_rate_min", 0.50) or 0.50)
    ax.axhline(actor_score_floor, color="#9333ea", lw=1, ls=":",
               label=f"actor floor {100.0 * actor_score_floor:.0f}%")
    ax.axhline(promotion_score_floor, color="#16a34a", lw=1, ls=":",
               label=f"promotion {100.0 * promotion_score_floor:.0f}%")
    eval_wins = _last(main, "eval_wins")
    eval_draws = _last(main, "eval_draws")
    eval_losses = _last(main, "eval_losses")
    if eval_wins is not None and eval_draws is not None and eval_losses is not None:
        ax.text(0.98, 0.04, f"latest W/D/L {eval_wins:.0f}/{eval_draws:.0f}/{eval_losses:.0f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#4b5563")
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax)

    ax = axes[0, 1]
    _style(ax, "Learner vs immutable IL anchor", percent=True)
    if first_promotion_iteration is None:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.58,
            "Best is still the IL anchor",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            0.5,
            0.42,
            "Results are reused from Learner vs promoted best\n(no additional games)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#6b7280",
        )
    else:
        anchor_rows = _anchor_rows_with_reused_best(main, first_promotion_iteration)
        shown = _line(
            ax, anchor_rows, "anchor_score_rate", "MCTS score", _COLORS[0],
            break_after=recovery_iterations,
        )
        shown |= _line(
            ax, anchor_rows, "anchor_score_lower_bound", "lower bound", _COLORS[0],
            ls=":", break_after=recovery_iterations,
        )
        shown |= _line(
            ax, anchor_rows, "anchor_no_mcts_score_rate", "NN only", _COLORS[1],
            break_after=recovery_iterations,
        )
        shown |= _line(
            ax, anchor_rows, "anchor_true_win_rate", "win rate", _COLORS[2],
            break_after=recovery_iterations,
        )
        ax.axhline(0.5, color="#374151", lw=1, ls="--")
        ax.text(
            0.98,
            0.04,
            "separate IL-anchor evaluation active",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#6b7280",
        )
        _legend_below(ax, ncol=2)
        if not shown:
            _no_data(ax)

    ax = axes[0, 2]
    _style(ax, "Estimated Elo with uncertainty")
    shown = False
    for column, low_column, high_column, label, color in (
        ("estimated_elo_nn", "estimated_elo_nn_ci95_low", "estimated_elo_nn_ci95_high", "NN", _COLORS[1]),
        ("estimated_elo_mcts", "estimated_elo_mcts_ci95_low", "estimated_elo_mcts_ci95_high", "MCTS", _COLORS[0]),
    ):
        xs, ys, lows, highs = [], [], [], []
        for row in main:
            x = _number(row.get("iteration")); y = _number(row.get(column))
            if x is None or y is None:
                continue
            low = _number(row.get(low_column)); high = _number(row.get(high_column))
            xs.append(int(x)); ys.append(y)
            lows.append(max(0.0, y - low) if low is not None else 0.0)
            highs.append(max(0.0, high - y) if high is not None else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=np.asarray([lows, highs]), color=color, label=label,
                        marker="o", markersize=4, linewidth=1.8, capsize=3, alpha=0.95)
            shown = True
    _legend_below(ax, ncol=2)
    if not shown:
        _no_data(ax)

    ax = axes[1, 0]
    _style(ax, "Relative score shift when both sides use MCTS", percent=True)
    shown = _line(
        ax,
        detail,
        "eval_mcts_no_mcts_gap",
        "MCTS match score - raw match score",
        _COLORS[0],
        break_after=matchup_breaks,
    )
    shown |= _line(
        ax, detail, "eval_mcts_no_mcts_gap_ema", "shift EMA", _COLORS[2],
        break_after=matchup_breaks,
    )
    if not shown:
        shown = _line(
            ax,
            main,
            "mcts_no_mcts_gap",
            "MCTS match score - raw match score",
            _COLORS[0],
            break_after=matchup_breaks,
        )
    ax.axhline(0.0, color="#374151", lw=1, ls="--")
    ax.text(
        0.02,
        0.96,
        "Not absolute MCTS strength: both candidate and reference change mode",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
        color="#4b5563",
    )
    _legend_below(ax, ncol=2)
    if not shown:
        _no_data(ax)

    ax = axes[1, 1]
    _actor_guard_panel(ax, main, promotion_iterations, recovery_iterations)

    ax = axes[1, 2]
    _style(ax, "Latest promotion and safety gates", percent=True)
    latest_nn_score = _last(main, "no_mcts_score_rate")
    latest_nn_games = _last(main, "no_mcts_games")
    stat_z = float(rl_cfg.get("promotion_stat_gate_z", 1.28) or 1.28)
    latest_nn_upper = None
    if latest_nn_score is not None and latest_nn_games is not None and latest_nn_games > 0.0:
        latest_nn_upper = min(
            1.0,
            latest_nn_score
            + stat_z * math.sqrt(
                max(0.0, latest_nn_score * (1.0 - latest_nn_score)) / latest_nn_games
            ),
        )
    readiness = [
        ("MCTS score", _last(main, "score_rate"), promotion_score_floor, _COLORS[0]),
        ("MCTS lower bound", _last(main, "eval_score_lower_bound"),
         float(rl_cfg.get("promotion_score_lower_bound_min", 0.50) or 0.50), _COLORS[5]),
        ("NN severe floor", latest_nn_score,
         float(rl_cfg.get("promotion_no_mcts_score_rate_min", 0.40) or 0.40), _COLORS[3]),
        ("NN non-regression UCB", latest_nn_upper,
         float(rl_cfg.get("promotion_no_mcts_upper_bound_min", 0.50) or 0.50), _COLORS[4]),
        ("Anchor score", _last(main, "anchor_score_rate"),
         float(rl_cfg.get("promotion_anchor_min_score_rate", 0.50) or 0.50), _COLORS[2]),
        ("Anchor lower bound", _last(main, "anchor_score_lower_bound"),
         float(rl_cfg.get("promotion_anchor_score_lower_bound_min", 0.47) or 0.47), _COLORS[4]),
    ]
    readiness = [(label, value, threshold, color) for label, value, threshold, color in readiness
                 if value is not None]
    shown = bool(readiness)
    if shown:
        y = np.arange(len(readiness))
        values = [value for _, value, _, _ in readiness]
        thresholds = [threshold for _, _, threshold, _ in readiness]
        colors = [color for _, _, _, color in readiness]
        left = min(0.25, min(values + thresholds) - 0.03)
        right = max(0.60, max(values + thresholds) + 0.05)
        ax.hlines(y, left, values, color=colors, linewidth=3, alpha=0.35)
        ax.scatter(values, y, color=colors, s=45, zorder=3, label="latest result")
        ax.scatter(thresholds, y, color="#111827", marker="|", s=150, linewidths=1.6,
                   zorder=4, label="required floor")
        ax.set_yticks(y, [label for label, _, _, _ in readiness]); ax.invert_yaxis()
        ax.set_xlim(left, right)
        for row_idx, value, threshold in zip(y, values, thresholds):
            ax.text(value + 0.008, row_idx, f"{100.0 * value:.1f}%", va="center", fontsize=7)
            ax.text(threshold, row_idx + 0.22, f"min {100.0 * threshold:.0f}%",
                    ha="center", va="bottom", fontsize=5.8, color="#4b5563")
        _legend_below(ax, ncol=2)
    else:
        _no_data(ax)

    ax = axes[2, 0]
    _style(ax, "Policy learning, useful MCTS corrections and LR")
    shown = _line(
        ax, main, "policy_loss", "policy loss", _COLORS[0], lw=2.1,
        break_after=recovery_iterations,
    )
    shown |= _line(
        ax, main, "policy_correction_loss", "useful correction loss",
        _COLORS[5], ls="--", lw=1.7, break_after=recovery_iterations,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    accuracy = _line(
        ax2, main, "policy_top1_acc", "top-1", _COLORS[2], lw=1.8,
        break_after=recovery_iterations,
    )
    accuracy |= _line(
        ax2, main, "policy_correction_top1_acc", "useful correction top-1",
        _COLORS[3], ls=":", lw=1.8, break_after=recovery_iterations,
    )
    lr_ax = ax.twinx()
    lr_ax.spines["right"].set_position(("outward", 40))
    lr_ax.patch.set_visible(False)
    lr_ax.tick_params(axis="y", labelcolor=_COLORS[4], labelsize=7)
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lr_ax.yaxis.get_offset_text().set_fontsize(6.5)
    lr = _line(
        lr_ax,
        main,
        "learning_rate",
        "learning rate",
        _COLORS[4],
        ls="--",
        lw=1.8,
        break_after=recovery_iterations,
    )
    ax.text(
        0.02, 0.96,
        "Correction = full search changed top move and improved root Q by > 0.02",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
        color="#4b5563",
    )
    ax.set_ylabel("policy loss", fontsize=7.5, color=_COLORS[0])
    ax2.set_ylabel("top-1", fontsize=7.5, color=_COLORS[2])
    lr_ax.set_ylabel("LR", fontsize=7.5, color=_COLORS[4], labelpad=3)
    ax.tick_params(axis="y", labelcolor=_COLORS[0])
    ax2.tick_params(axis="y", labelcolor=_COLORS[2])
    _legend_below(ax, ax2, lr_ax, ncol=3)
    if not (shown or accuracy or lr): _no_data(ax)

    ax = axes[2, 1]
    _style(ax, "Value learning")
    shown = _line(
        ax, main, "value_loss", "value loss", _COLORS[3], lw=2.1,
        break_after=recovery_iterations,
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

    for plot_index, plot_ax in enumerate((
        axes[0, 0], axes[0, 1], axes[0, 2],
        axes[1, 0], axes[1, 1],
        axes[2, 0], axes[2, 1], axes[2, 2],
    )):
        _mark_regime_changes(
            plot_ax,
            promotion_iterations,
            recovery_iterations,
            label=(plot_index == 0),
        )
    _legend_below(axes[0, 0], ncol=4)

    context = " | ".join(str(line) for line in (run_context_lines or []) if line)
    _finish(
        fig,
        output_path,
        "RL training - learner / actor decision dashboard",
        context or None,
        hspace=0.72,
        wspace=0.38,
    )
    return True


def render_rl_data_quality(main_csv_path, data_quality_csv_path, output_path):
    """Render the compact dashboard used to diagnose RL data and search quality."""
    main = _rows(main_csv_path)
    rows = _rows(data_quality_csv_path)
    rl_cfg = _rl_config(main_csv_path)
    if not rows:
        return False
    fig, axes = plt.subplots(5, 3, figsize=(18, 22))
    promotion_iterations, recovery_iterations = _regime_change_iterations(main)
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
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not shown:
        _no_data(ax)

    ax = axes[0, 1]
    _style(ax, "Replay age by source")
    shown = _line(
        ax, rows, "recent_sample_age_avg", "current mean", _COLORS[0], lw=1.6,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "recent_sample_age_p50", "current p50", _COLORS[2], lw=1.8,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "recent_sample_age_p90", "current p90", _COLORS[1], ls="--",
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "champion_sample_age_p50", "champion p50", "#7c3aed", lw=1.6,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "champion_sample_age_p90", "champion p90", "#7c3aed", ls="--",
        break_after=selfplay_breaks,
    )
    p50_x, p50_y = _series(rows, "recent_sample_age_p50")
    p90_x, p90_y = _series(rows, "recent_sample_age_p90")
    if not shown:
        shown = _line(
            ax, rows, "sample_age_avg", "combined mean (legacy)", _COLORS[0], lw=1.8,
            break_after=selfplay_breaks,
        )
        shown |= _line(
            ax, rows, "sample_age_p50", "combined p50 (legacy)", _COLORS[2], lw=1.8,
            break_after=selfplay_breaks,
        )
        shown |= _line(
            ax, rows, "sample_age_p90", "combined p90 (legacy)", _COLORS[1], ls="--",
            break_after=selfplay_breaks,
        )
        p50_x, p50_y = _series(rows, "sample_age_p50")
        p90_x, p90_y = _series(rows, "sample_age_p90")
    if p50_x and p50_x == p90_x:
        _fill_between_with_breaks(
            ax, p50_x, p50_y, p90_y, break_after=selfplay_breaks,
            color=_COLORS[0], alpha=0.08, linewidth=0,
        )
    ax.set_ylabel("iterations old", fontsize=8)
    _mark_regime_changes(
        ax, promotion_iterations, recovery_iterations, label=True,
    )
    _legend_below(ax, ncol=4)
    if not shown:
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
    _style(ax, "Supervision health", percent=True)
    shown = _line(
        ax, rows, "policy_weight_mean", "policy weight mean", _COLORS[0], lw=1.9,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "policy_weight_p10", "policy weight p10", _COLORS[1], ls="--",
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "policy_weight_low_fraction", "low-weight samples", _COLORS[3],
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "root_q_coverage", "root-Q coverage", _COLORS[2], lw=1.9,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_full_search_fraction", "full-search samples", _COLORS[4],
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax,
        rows,
        "replay_policy_correction_fraction",
        "useful correction rows",
        "#db2777",
        lw=1.8,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax,
        rows,
        "replay_policy_correction_weight_share",
        "correction weight share",
        "#7c3aed",
        ls="--",
        lw=1.6,
        break_after=selfplay_breaks,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax)

    ax = axes[2, 0]
    _style(ax, "MCTS simulation allocation")
    shown = _line(
        ax, rows, "mcts_budget_p10", "p10", _COLORS[1], ls="--", lw=1.4,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_budget_p50", "p50", _COLORS[2], lw=1.5,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_avg_budget", "average", _COLORS[0], lw=2.4,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_fresh_sims", "fresh after reuse", "#0891b2", ls=":", lw=1.8,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_tree_reuse_fresh_floor_avg", "adaptive fresh floor",
        "#f59e0b", ls="-.", lw=1.6, break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_tree_reuse_scout_extra_credit_avg", "post-scout extra credit",
        "#db2777", ls=":", lw=1.6, break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_budget_p90", "p90", _COLORS[3], ls="--", lw=1.4,
        break_after=selfplay_breaks,
    )
    p10_x, p10_y = _series(rows, "mcts_budget_p10")
    p90_x, p90_y = _series(rows, "mcts_budget_p90")
    if p10_x and p10_x == p90_x:
        _fill_between_with_breaks(
            ax, p10_x, p10_y, p90_y, break_after=selfplay_breaks,
            color=_COLORS[0], alpha=0.08, linewidth=0,
        )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    allocation_quality = _line(
        ax2, rows, "mcts_difficulty_budget_correlation", "difficulty-budget corr",
        "#7c3aed", ls=":", lw=1.8, break_after=selfplay_breaks,
    )
    allocation_quality |= _line(
        ax2, rows, "mcts_tree_reuse_hit_rate", "tree reuse hit rate",
        "#059669", ls="--", lw=1.6, break_after=selfplay_breaks,
    )
    allocation_quality |= _line(
        ax2, rows, "mcts_tree_reuse_credit_fraction", "reuse-credit eligible",
        "#db2777", ls="-.", lw=1.5, break_after=selfplay_breaks,
    )
    allocation_quality |= _line(
        ax2, rows, "mcts_tree_reuse_scout_reduction_rate", "search reduced after scout",
        "#0f766e", ls=":", lw=1.6, break_after=selfplay_breaks,
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("difficulty correlation", fontsize=7.5, color="#7c3aed")
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ax2, ncol=3)
    ax.set_ylabel("simulations / position", fontsize=8)
    if not shown:
        _no_data(ax)

    ax = axes[2, 1]
    _style(ax, "Search correction rate", percent=True)
    shown = _line(
        ax, rows, "mcts_prior_changed_rate", "changed top", _COLORS[0], lw=2.0,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_useful_change_rate", "useful", _COLORS[2], lw=1.8,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_harmful_change_rate", "harmful", _COLORS[1], lw=1.8,
        break_after=selfplay_breaks,
    )
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
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
    _style(ax, "Search breadth")
    shown = _line(
        ax, rows, "mcts_visited_move_count_mean", "visited moves", _COLORS[0], lw=2.0,
        break_after=selfplay_breaks,
    )
    shown |= _line(
        ax, rows, "mcts_legal_move_count_mean", "legal moves", _COLORS[1], ls="--",
        break_after=selfplay_breaks,
    )
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    coverage = _line(
        ax2, rows, "mcts_visit_coverage_ratio_mean", "coverage", _COLORS[2], lw=1.8,
        break_after=selfplay_breaks,
    )
    coverage |= _line(
        ax2, rows, "mcts_tree_reuse_quality_avg", "reuse quality", "#7c3aed", lw=1.8,
        break_after=selfplay_breaks,
    )
    coverage |= _line(
        ax2, rows, "mcts_tree_reuse_candidate_coverage_avg", "Gumbel candidate coverage",
        "#f59e0b", ls="--", lw=1.6, break_after=selfplay_breaks,
    )
    coverage |= _line(
        ax2, rows, "mcts_tree_reuse_visited_prior_mass_avg", "reused prior mass",
        "#0891b2", ls=":", lw=1.7, break_after=selfplay_breaks,
    )
    coverage |= _line(
        ax2, rows, "mcts_tree_reuse_scout_stability_avg", "post-scout stability",
        "#db2777", ls="-.", lw=1.6, break_after=selfplay_breaks,
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("coverage / reuse quality", fontsize=7.5)
    _legend_below(ax, ax2, ncol=3)
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    if not (shown or coverage):
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
    _mark_regime_changes(ax, promotion_iterations, recovery_iterations)
    _legend_below(ax, ncol=3)
    if not shown:
        _no_data(ax)

    _latest_quality_bars(axes[3, 2], rows)

    _task_gradient_interaction_panel(axes[4, 0], main, recovery_iterations)
    _gradient_norms_panel(axes[4, 1], main, recovery_iterations)
    _objective_composition_panel(axes[4, 2], main, rl_cfg)
    for gradient_ax in axes[4, :]:
        _mark_regime_changes(gradient_ax, promotion_iterations, recovery_iterations)

    _finish(
        fig,
        output_path,
        "RL details - data, search and optimization quality",
        "Actor replay stability, MCTS behavior, task-gradient interaction and the effective training objective.",
        hspace=0.78,
    )
    return True


def render_rl_performance(performance_csv_path, output_path, data_quality_csv_path=None):
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

    fig = plt.figure(figsize=(18, 13.2))
    grid = fig.add_gridspec(3, 3)
    axes = np.empty((3, 3), dtype=object)
    for row_idx in range(3):
        for column_idx in range(3):
            axes[row_idx, column_idx] = fig.add_subplot(grid[row_idx, column_idx])

    ax = axes[0, 0]
    _style(ax, "Actor self-play throughput")
    shown = _line(ax, rows, "replay_positions_per_sec", "replay positions/s", _COLORS[0])
    shown |= _line(ax, rows, "played_positions_per_sec", "played positions/s", _COLORS[2])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    visits = _line(
        ax2,
        rows,
        "mcts_simulations_per_sec",
        "completed MCTS visits/s",
        _COLORS[3],
    )
    ax.set_ylabel("positions/s", fontsize=8)
    if visits:
        ax2.set_ylabel("completed simulations/s", fontsize=8)
    else:
        ax2.set_visible(False)
    _legend_below(ax, ax2, ncol=3)
    if not (shown or visits): _no_data(ax)
    latest_path = _last(rows, "mcts_selection_path_length")
    if latest_path is not None:
        ax.text(
            0.98, 0.04, f"latest selection path: {latest_path:.2f} traversals/sim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#4b5563",
        )

    stage_specs = (
        ("stage_selfplay_time_s", "actor self-play", "#2563eb"),
        ("stage_replay_time_s", "replay", "#0891b2"),
        ("stage_train_time_s", "train", "#16a34a"),
        ("stage_regular_eval_time_s", "regular eval", "#eab308"),
        ("stage_promotion_eval_time_s", "promotion eval", "#f59e0b"),
        ("stage_elo_eval_time_s", "Elo eval", "#ec4899"),
        ("stage_log_time_s", "decision/log", "#a855f7"),
        # Schemas <= 7 did not split evaluation. Keep the old value visible
        # without pretending that historical time belonged to one new bucket.
        ("stage_eval_log_time_s", "legacy eval/log", "#fbbf24"),
        ("stage_checkpoint_time_s", "checkpoint", "#9333ea"),
        ("stage_setup_time_s", "setup", "#64748b"),
        ("stage_gc_time_s", "gc", "#db2777"),
    )
    iterations, totals = [], []
    stage_values = {column: [] for column, _, _ in stage_specs}
    other_values = []
    for row in rows:
        iteration = _number(row.get("iteration"))
        if iteration is None:
            continue
        values = {
            column: max(0.0, _number(row.get(column)) or 0.0)
            for column, _, _ in stage_specs
        }
        measured_total = sum(values.values())
        logged_total = _number(row.get("iteration_total_time_s"))
        total = max(measured_total, logged_total or 0.0)
        iterations.append(int(iteration))
        totals.append(total)
        for column in stage_values:
            stage_values[column].append(values[column])
        other_values.append(max(0.0, total - measured_total))

    ax = axes[0, 1]
    _style(ax, "Runtime composition per iteration")
    if iterations and any(totals):
        bottoms = [0.0] * len(iterations)
        runtime_total = sum(totals)
        visible_stage_count = 0
        for column, label, color in stage_specs:
            values = stage_values[column]
            component_total = sum(values)
            if component_total < 1.0 or (
                runtime_total > 0.0 and component_total / runtime_total < 0.002
            ):
                continue
            ax.bar(iterations, values, bottom=bottoms, width=0.76, color=color,
                   label=label, edgecolor="white", linewidth=0.35)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
            visible_stage_count += 1
        other_total = sum(other_values)
        if other_total >= 1.0 and (runtime_total <= 0.0 or other_total / runtime_total >= 0.002):
            ax.bar(iterations, other_values, bottom=bottoms, width=0.76, color="#d1d5db",
                   label="other/unaccounted", edgecolor="white", linewidth=0.35)
            visible_stage_count += 1
        if len(iterations) <= 12:
            padding = max(totals) * 0.015
            for iteration, total in zip(iterations, totals):
                ax.text(iteration, total + padding, f"{total:.0f}s", ha="center", va="bottom",
                        fontsize=6.5, color="#374151")
        ax.set_ylim(0.0, max(totals) * 1.10)
        ax.set_ylabel("seconds", fontsize=8)
        ax.set_axisbelow(True)
        _legend_below(ax, ncol=min(4, max(1, visible_stage_count)))
    else:
        _no_data(ax)

    ax = axes[0, 2]
    run_iterations = len(iterations)
    runtime_total = [
        (column, label, sum(stage_values[column]) / max(1, run_iterations), color)
        for column, label, color in stage_specs
    ]
    if run_iterations > 0:
        runtime_total.append((
            "other", "other", sum(other_values) / max(1, run_iterations), "#d1d5db"
        ))
    _composition_donut(
        ax,
        runtime_total,
        title="Average runtime composition",
        center_label="avg / iteration",
        unit="s",
        decimals=0,
        min_share=0.005,
        min_value=1.0,
    )

    ax = axes[1, 0]
    central_specs = (
        ("descriptor_queue", "queue backlog", "#f59e0b"),
        ("batch_coalesce", "batch coalesce", "#f97316"),
        ("concat", "concat", "#0891b2"),
        ("cache", "cache/dedupe", "#84cc16"),
        ("staging", "pinned staging", "#06b6d4"),
        ("h2d", "H2D", "#2563eb"),
        ("forward", "forward", "#16a34a"),
        ("d2h", "D2H", "#7c3aed"),
        ("server_other", "server other", "#64748b"),
        ("ipc_other", "IPC/worker", "#db2777"),
    )

    def _central_latency_parts(row):
        remote = max(0.0, _number(row.get("central_remote_wait_ms_per_request")) or 0.0)
        queue = max(0.0, _number(row.get("central_server_queue_wait_ms_per_request")) or 0.0)
        server_total = max(0.0, _number(row.get("central_server_total_ms_per_request")) or 0.0)
        descriptor_raw = _number(row.get("central_descriptor_queue_wait_ms_per_request"))
        coalesce_raw = _number(row.get("central_batch_coalesce_wait_ms_per_request"))
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
        parts = {
            "descriptor_queue": descriptor_queue,
            "batch_coalesce": batch_coalesce,
            "concat": max(0.0, _number(row.get("central_server_concat_ms_per_request")) or 0.0),
            "cache": max(0.0, _number(row.get("central_server_cache_lookup_ms_per_request")) or 0.0),
            "staging": max(0.0, _number(row.get("central_server_staging_copy_ms_per_request")) or 0.0),
            "h2d": max(0.0, _number(row.get("central_server_h2d_ms_per_request")) or 0.0),
            "forward": max(0.0, _number(row.get("central_server_forward_ms_per_request")) or 0.0),
            "d2h": max(0.0, _number(row.get("central_server_d2h_ms_per_request")) or 0.0),
        }
        known_server = sum(parts[key] for key in ("concat", "cache", "staging", "h2d", "forward", "d2h"))
        logged_server_other = _number(row.get("central_server_other_ms_per_request"))
        logged_ipc_other = _number(row.get("central_worker_ipc_ms_per_request"))
        parts["server_other"] = max(0.0, logged_server_other if logged_server_other is not None else server_total - known_server)
        parts["ipc_other"] = max(0.0, logged_ipc_other if logged_ipc_other is not None else remote - queue - server_total)
        return parts

    _stacked_iteration_bars(
        ax,
        rows,
        central_specs,
        _central_latency_parts,
        title="Central request composition",
        ylabel="ms / request",
        legend_columns=4,
    )

    ax = axes[1, 1]
    position_latency_specs = (*central_specs, ("local_other", "local/other", "#111827"))

    def _position_latency_parts(row):
        worker_batch = max(0.0, _number(row.get("mcts_avg_batch_size")) or 0.0)
        worker_wait = max(0.0, _number(row.get("mcts_worker_nn_wait_ms_per_position")) or 0.0)
        if worker_batch <= 0.0:
            return {"local_other": worker_wait}
        parts = {
            key: value / worker_batch
            for key, value in _central_latency_parts(row).items()
        }
        parts["local_other"] = max(0.0, worker_wait - sum(parts.values()))
        return parts

    _stacked_iteration_bars(
        ax,
        rows,
        position_latency_specs,
        _position_latency_parts,
        title="Latency composition per position",
        ylabel="ms / position",
        legend_columns=4,
    )

    ax = axes[1, 2]
    central_average, _ = _whole_run_composition(
        rows,
        central_specs,
        _central_latency_parts,
        weight_column="mcts_nn_inference_calls",
    )
    _composition_donut(
        ax,
        central_average,
        title="Average central request composition",
        center_label="avg / request",
        unit="ms",
        decimals=1,
        min_share=0.01,
        min_value=0.1,
    )

    ax = axes[2, 0]
    search_specs = (
        ("inference_wait", "waiting for NN inference", "#db2777"),
        ("board_copy", "board materialization/push", "#dc2626"),
        ("terminal", "terminal/draw", "#f59e0b"),
        ("legal_mixed", "legal + move encoding", "#ea580c"),
        ("legal", "legal move generation", "#fb923c"),
        ("move_index", "move encoding", "#facc15"),
        ("legacy_leaf", "legacy leaf prep (mixed)", "#9ca3af"),
        ("selection", "tree selection", "#2563eb"),
        ("encoding", "board encoding", "#0891b2"),
        ("packing", "input packing", "#06b6d4"),
        ("policy", "CPU policy", "#16a34a"),
        ("backprop", "backprop/other", "#7c3aed"),
    )

    def _mcts_search_parts(row):
        schema_version = _number(row.get("schema_version")) or 0.0
        tensor_pack = max(0.0, _number(row.get("mcts_batch_expand_tensor_pack_time_s")) or 0.0)
        legal_index_pack = max(0.0, _number(row.get("mcts_batch_expand_legal_index_pack_time_s")) or 0.0)
        board_encoding = max(0.0, _number(row.get("mcts_board_to_tensor_time_s")) or 0.0)
        encoding_in_pack = min(tensor_pack, board_encoding)
        tree_other = sum(
            max(0.0, _number(row.get(column)) or 0.0)
            for column in (
                "mcts_search_backprop_time_s",
                "mcts_search_root_setup_time_s",
                "mcts_search_metadata_time_s",
                "mcts_batch_expand_dedup_time_s",
                "mcts_batch_expand_value_fanout_time_s",
            )
        )
        old_leaf_prep = max(0.0, _number(row.get("mcts_batch_expand_legal_moves_time_s")) or 0.0)
        return {
            # In central-inference mode this is worker wall-time blocked on the
            # response queue. It is intentionally separate from measured CPU
            # work so a slow search is not misdiagnosed as Python tree cost.
            "inference_wait": _number(row.get("mcts_nn_inference_time_s")) or 0.0,
            "board_copy": _number(row.get("mcts_board_materialize_time_s")) or 0.0,
            "terminal": _number(row.get("mcts_terminal_checks_time_s")) or 0.0,
            # Schema 5 measured legal generation and policy-index encoding together.
            # Schema 6 separates them; older schemas measured mixed leaf preparation.
            "legal_mixed": old_leaf_prep if 5.0 <= schema_version < 6.0 else 0.0,
            "legal": old_leaf_prep if schema_version >= 6.0 else 0.0,
            "move_index": _number(row.get("mcts_batch_expand_move_index_time_s")) or 0.0,
            "legacy_leaf": old_leaf_prep if schema_version < 5.0 else 0.0,
            "selection": _number(row.get("mcts_search_selection_time_s")) or 0.0,
            "encoding": encoding_in_pack,
            "packing": max(0.0, tensor_pack - encoding_in_pack) + legal_index_pack,
            "policy": _number(row.get("mcts_batch_expand_cpu_policy_time_s")) or 0.0,
            "backprop": tree_other,
        }

    _stacked_iteration_bars(
        ax,
        rows,
        search_specs,
        _mcts_search_parts,
        title="Measured MCTS search composition",
        ylabel="worker-summed seconds",
        legend_columns=4,
    )

    search_average, _ = _whole_run_composition(
        rows, search_specs, _mcts_search_parts,
    )
    search_average = [
        (key, label, value / max(1, run_iterations), color)
        for key, label, value, color in search_average
    ]
    _composition_donut(
        axes[2, 1],
        search_average,
        title="Average measured MCTS search composition",
        center_label="avg / iteration",
        unit="s",
        decimals=0,
        min_share=0.01,
        min_value=1.0,
        autopct_min=7.0,
    )

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
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(100.0))
    wait_share = _line(ax2, rows, "mcts_worker_nn_wait_share_pct", "worker NN wait share", _COLORS[3])
    if not wait_share:
        wait_share = _line(ax2, rows, "mcts_gpu_busy_proxy_pct", "worker NN wait share", _COLORS[3])
    if not wait_share:
        wait_share = _line(ax2, rows, "mcts_gpu_utilization_pct", "worker NN wait share", _COLORS[3])
    for column, label, color in (
        ("central_nn_saved_overall_rate", "NN evals saved overall", "#16a34a"),
        ("central_cache_active_fraction", "cache active coverage", "#0891b2"),
        ("central_gpu_batch_fill", "GPU batch fill", "#7c3aed"),
    ):
        xs, ys = _series(rows, column)
        if xs:
            ax2.plot(xs, [100.0 * value for value in ys], marker="o", ms=3, lw=1.5, label=label, color=color)
            wait_share = True
    _legend_below(ax, ax2, ncol=3)
    if not (shown or wait_share): _no_data(ax)

    _finish(
        fig,
        output_path,
        "RL details - performance",
        "Homogeneous actor-vs-actor self-play. Completed MCTS visit = one finished simulation; selection traversals count root-to-leaf path elements. "
        "Waiting for NN inference is worker-summed wall-time blocked on the response; it is not CPU or hardware GPU utilization.",
        hspace=0.78,
    )
    return True
