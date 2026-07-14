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
import math
from pathlib import Path

import matplotlib.pyplot as plt
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


def _style(ax, title, percent=False):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.tick_params(labelsize=8)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _line(ax, rows, column, label, color=None, *, ls="-", lw=1.8, alpha=1.0):
    xs, ys = _series(rows, column)
    if not xs:
        return False
    ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
            marker="o", markersize=3.5)
    return True


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


def _average_composition(rows, specs, value_resolver, window=5):
    """Return last-window component means using the same resolver as stacked bars."""
    selected_rows = list(rows[-max(1, int(window)):])
    if not selected_rows:
        return [], 0
    sums = {key: 0.0 for key, _, _ in specs}
    for row in selected_rows:
        resolved = dict(value_resolver(row) or {})
        for key in sums:
            sums[key] += max(0.0, _number(resolved.get(key)) or 0.0)
    divisor = float(len(selected_rows))
    averaged = [
        (key, label, sums[key] / divisor, color)
        for key, label, color in specs
    ]
    return averaged, len(selected_rows)


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
        if percent < float(autopct_min):
            return ""
        value = visible_total * percent / 100.0
        return f"{percent:.1f}%\n{value:.{decimals}f}{unit}"

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
        f"{center_label}\n{visible_total:.{decimals}f}{unit}",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#374151",
    )
    legend_labels = [
        f"{label} | {100.0 * value / visible_total:.1f}% | {value:.{decimals}f}{unit}"
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


def _legend(ax, *, ncol=2, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, frameon=False, ncol=ncol, loc=loc)


def _no_data(ax, message="Brak danych w CSV"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="#6b7280")
    ax.set_xticks([])
    ax.set_yticks([])


def _finish(fig, output_path, title, subtitle=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.04, ha="left", y=0.988)
    if subtitle:
        fig.text(0.04, 0.958, subtitle, fontsize=8.5, color="#4b5563", ha="left", va="top")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.045, top=0.91, hspace=0.48, wspace=0.30)
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


def render_rl_main(main_csv_path, data_quality_csv_path, performance_csv_path, output_path,
                   run_context_lines=None):
    """Render the single screen used to judge whether the run is improving."""
    main = _rows(main_csv_path)
    detail = _rows(data_quality_csv_path)
    perf = _rows(performance_csv_path)
    if not main:
        return False

    fig, axes = plt.subplots(4, 3, figsize=(18, 15))

    ax = axes[0, 0]
    _style(ax, "Current vs best", percent=True)
    shown = _line(ax, main, "score_rate", "MCTS", _COLORS[0])
    shown |= _line(ax, main, "eval_score_lower_bound", "MCTS lower bound", _COLORS[0], ls=":")
    shown |= _line(ax, main, "no_mcts_score_rate", "NN only", _COLORS[1])
    ax.axhline(0.5, color="#374151", lw=1, ls="--", label="break-even")
    ax.axhline(0.55, color="#16a34a", lw=1, ls=":", label="promotion 55%")
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[0, 1]
    _style(ax, "Current vs IL anchor", percent=True)
    shown = _line(ax, main, "anchor_score_rate", "MCTS score", _COLORS[0])
    shown |= _line(ax, main, "anchor_score_lower_bound", "lower bound", _COLORS[0], ls=":")
    shown |= _line(ax, main, "anchor_no_mcts_score_rate", "NN only", _COLORS[1])
    shown |= _line(ax, main, "anchor_true_win_rate", "win rate", _COLORS[2])
    ax.axhline(0.5, color="#374151", lw=1, ls="--")
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[0, 2]
    _style(ax, "Estimated Elo")
    shown = _line(ax, main, "estimated_elo_nn", "NN", _COLORS[1])
    shown |= _line(ax, main, "estimated_elo_mcts", "MCTS", _COLORS[0])
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[1, 0]
    _style(ax, "MCTS added value", percent=True)
    shown = _line(ax, detail, "eval_mcts_no_mcts_gap", "gap", _COLORS[0])
    shown |= _line(ax, detail, "eval_mcts_no_mcts_gap_ema", "gap EMA", _COLORS[2])
    if not shown:
        shown = _line(ax, main, "mcts_no_mcts_gap", "gap", _COLORS[0])
    ax.axhline(0.0, color="#374151", lw=1, ls="--")
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[1, 1]
    _style(ax, "Search changes move", percent=True)
    shown = _line(ax, detail, "mcts_prior_changed_rate", "changed top", _COLORS[0])
    absolute_quality = _line(ax, detail, "mcts_useful_change_rate", "useful change", _COLORS[2])
    absolute_quality |= _line(ax, detail, "mcts_harmful_change_rate", "harmful change", _COLORS[1])
    if not absolute_quality:  # v1 logs stored only rates conditional on a changed move.
        _line(ax, detail, "mcts_changed_to_higher_q_rate", "higher Q | changed", _COLORS[2])
        _line(ax, detail, "mcts_changed_to_lower_q_when_changed_rate", "lower Q | changed", _COLORS[1])
    _legend(ax, ncol=1)
    if not shown:
        _no_data(ax)

    ax = axes[1, 2]
    _style(ax, "Self-play outcomes", percent=True)
    shown = _line(ax, detail, "selfplay_decisive_rate", "decisive", _COLORS[2])
    shown |= _line(ax, detail, "selfplay_draw_rate", "draw", _COLORS[0])
    shown |= _line(ax, detail, "selfplay_auto_draw_rate", "auto draw", _COLORS[3])
    shown |= _line(ax, detail, "selfplay_truncated_rate", "truncated", _COLORS[1])
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "Training losses")
    shown = _line(ax, main, "avg_loss", "total", "#111827", lw=2.2)
    shown |= _line(ax, main, "policy_loss", "policy", _COLORS[0])
    shown |= _line(ax, main, "value_loss", "value", _COLORS[3])
    _legend(ax, ncol=3)
    if not shown:
        _no_data(ax)

    ax = axes[2, 0]
    _style(ax, "Training accuracy", percent=True)
    shown = _line(ax, main, "policy_top1_acc", "policy top-1", _COLORS[0])
    shown |= _line(ax, main, "policy_top3_acc", "policy top-3", _COLORS[2])
    shown |= _line(ax, main, "value_wdl_acc", "value WDL", _COLORS[3])
    _legend(ax, ncol=2)
    if not shown:
        _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "Value MAE")
    shown = _line(ax, main, "value_mae", "overall", "#111827", lw=2.2)
    shown |= _line(ax, main, "value_mae_opening", "opening", _COLORS[0])
    shown |= _line(ax, main, "value_mae_middlegame", "middlegame", _COLORS[3])
    shown |= _line(ax, main, "value_mae_endgame", "endgame", _COLORS[1])
    _legend(ax, ncol=2)
    if not shown:
        _no_data(ax)

    ax = axes[3, 0]
    _style(ax, "Value spread calibration")
    shown = _line(ax, main, "value_std_ratio_opening", "opening", _COLORS[0])
    shown |= _line(ax, main, "value_std_ratio_middlegame", "middlegame", _COLORS[3])
    shown |= _line(ax, main, "value_std_ratio_endgame", "endgame", _COLORS[1])
    if not shown:  # Backward compatibility with v1 details CSVs.
        shown = _line(ax, detail, "train_value_std_ratio_opening", "opening", _COLORS[0])
        shown |= _line(ax, detail, "train_value_std_ratio_middlegame", "middlegame", _COLORS[3])
        shown |= _line(ax, detail, "train_value_std_ratio_endgame", "endgame", _COLORS[1])
    ax.axhline(1.0, color="#374151", lw=1, ls="--", label="calibrated")
    _legend(ax)
    if not shown:
        _no_data(ax)

    ax = axes[3, 1]
    _style(ax, "Training controls")
    shown = _line(ax, main, "mcts_q_effective_weight", "Q weight", _COLORS[0])
    shown |= _line(ax, main, "value_loss_weight", "value loss weight", _COLORS[3])
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=8)
    lr = _line(ax2, main, "learning_rate", "learning rate", _COLORS[1])
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if handles or h2:
        ax.legend(handles + h2, labels + l2, fontsize=7, frameon=False, ncol=2)
    if not (shown or lr):
        _no_data(ax)

    ax = axes[3, 2]
    ax.axis("off")
    summary = [
        ("MCTS vs best", _fmt(_last(main, "score_rate"), "percent")),
        ("MCTS lower bound", _fmt(_last(main, "eval_score_lower_bound"), "percent")),
        ("NN vs best", _fmt(_last(main, "no_mcts_score_rate"), "percent")),
        ("MCTS - NN", _fmt(_last(detail, "eval_mcts_no_mcts_gap", _last(main, "mcts_no_mcts_gap")), "percent")),
        ("Anchor", _fmt(_last(main, "anchor_score_rate"), "percent")),
        ("Anchor NN", _fmt(_last(main, "anchor_no_mcts_score_rate"), "percent")),
        ("Anchor lower bound", _fmt(_last(main, "anchor_score_lower_bound"), "percent")),
        ("MCTS Elo", _fmt(_last(main, "estimated_elo_mcts"), "integer")),
        ("Policy top-1", _fmt(_last(main, "policy_top1_acc"), "percent")),
        ("Value MAE", _fmt(_last(main, "value_mae"))),
        ("Changed top", _fmt(_last(detail, "mcts_prior_changed_rate"), "percent")),
        ("Good MCTS target", _fmt(_last(detail, "mcts_good_target_rate"), "percent")),
        ("Replay throughput", f"{_fmt(_last(perf, 'replay_positions_per_sec', _last(perf, 'positions_per_sec')), 'integer')} pos/s"),
        ("Played throughput", f"{_fmt(_last(perf, 'played_positions_per_sec'), 'integer')} pos/s"),
    ]
    ax.set_title("Latest checkpoint", fontsize=10, fontweight="bold", loc="left")
    table = ax.table(cellText=summary, colLabels=("Signal", "Latest"), loc="center",
                     cellLoc="left", colWidths=(0.62, 0.30))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if row == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold")

    promotion_iterations, _ = _series(main, "rl_best_model")
    for plot_ax in axes.flat[:-1]:
        for promotion_iteration in promotion_iterations:
            plot_ax.axvline(
                promotion_iteration,
                color="#111827",
                linestyle="--",
                linewidth=1.0,
                alpha=0.45,
            )

    context = " | ".join(str(line) for line in (run_context_lines or []) if line)
    _finish(fig, output_path, "RL training - decision dashboard", context or None)
    return True


def render_rl_data_quality(main_csv_path, data_quality_csv_path, output_path):
    """Render replay, target and search diagnostics without evaluation duplication."""
    main = _rows(main_csv_path)
    rows = _rows(data_quality_csv_path)
    if not rows:
        return False
    fig, axes = plt.subplots(4, 3, figsize=(18, 15))

    ax = axes[0, 0]
    _style(ax, "Replay inventory")
    shown = _line(ax, rows, "replay_size", "positions", _COLORS[0])
    shown |= _line(ax, rows, "replay_capacity", "capacity", _COLORS[1], ls="--")
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    fill = _line(ax2, rows, "replay_fill_rate", "filled", _COLORS[2])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or fill): _no_data(ax)

    ax = axes[0, 1]
    _style(ax, "Replay freshness", percent=True)
    shown = _line(ax, rows, "sample_age_new_fraction", "new", _COLORS[2])
    shown |= _line(ax, rows, "sample_age_le1_fraction", "age <= 1", _COLORS[0])
    promotion_iterations, _ = _series(main, "rl_best_model")
    for index, promotion_iteration in enumerate(promotion_iterations):
        ax.axvline(
            promotion_iteration,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label="promotion" if index == 0 else None,
        )
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[0, 2]
    _style(ax, "Replay vs self-play outcomes", percent=True)
    shown = _line(ax, rows, "replay_decisive_fraction", "replay decisive", _COLORS[2])
    shown |= _line(ax, rows, "selfplay_decisive_rate", "self-play decisive", _COLORS[2], ls="--")
    shown |= _line(ax, rows, "replay_draw_fraction", "replay draw", _COLORS[0])
    shown |= _line(ax, rows, "selfplay_draw_rate", "self-play draw", _COLORS[0], ls="--")
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[1, 0]
    _style(ax, "Replay value balance", percent=True)
    shown = _line(ax, rows, "replay_value_positive_fraction", "positive", _COLORS[2])
    shown |= _line(ax, rows, "replay_value_neutral_fraction", "neutral", _COLORS[0])
    shown |= _line(ax, rows, "replay_value_negative_fraction", "negative", _COLORS[1])
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[1, 1]
    _style(ax, "Replay source composition", percent=True)
    shown = _line(ax, rows, "replay_source_learner_fraction", "learner positions", _COLORS[0])
    shown |= _line(ax, rows, "replay_source_frozen_best_fraction", "best positions", _COLORS[1])
    shown |= _line(ax, rows, "replay_policy_weight_learner_share", "learner weight", _COLORS[0], ls="--")
    shown |= _line(ax, rows, "replay_policy_weight_frozen_best_share", "best weight", _COLORS[1], ls="--")
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[1, 2]
    _style(ax, "Opponent mix and score", percent=True)
    shown = _line(ax, rows, "opponent_current_planned_share", "current planned", _COLORS[0], ls="--")
    shown |= _line(ax, rows, "opponent_current_actual_share", "current actual", _COLORS[0])
    shown |= _line(ax, rows, "opponent_mix_error", "mix error", _COLORS[1])
    shown |= _line(ax, rows, "opponent_current_score_rate", "vs current", _COLORS[2], ls="--")
    shown |= _line(ax, rows, "opponent_best_score_rate", "vs best", _COLORS[3], ls="--")
    ax.axhline(0.5, color="#374151", lw=1, ls=":")
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[2, 0]
    _style(ax, "Policy target shape")
    shown = _line(ax, rows, "policy_target_entropy_mean", "entropy", _COLORS[3])
    shown |= _line(ax, rows, "policy_target_effective_moves", "effective moves", _COLORS[4])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    probs = _line(ax2, rows, "policy_target_top1_prob_mean", "top-1 mass", _COLORS[0])
    probs |= _line(ax2, rows, "policy_target_top3_prob_mean", "top-3 mass", _COLORS[2])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False, ncol=2)
    if not (shown or probs): _no_data(ax)

    ax = axes[2, 1]
    _style(ax, "Policy target width and weight")
    shown = _line(ax, rows, "policy_weight_mean", "weight mean", _COLORS[0])
    shown |= _line(ax, rows, "policy_weight_p10", "weight p10", _COLORS[1])
    shown |= _line(ax, rows, "policy_weight_low_fraction", "low fraction", _COLORS[3])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    width = _line(ax2, rows, "policy_target_len_mean", "moves mean", _COLORS[2])
    width |= _line(ax2, rows, "policy_target_len_p90", "moves p90", _COLORS[4])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False, ncol=2)
    if not (shown or width): _no_data(ax)

    ax = axes[2, 2]
    _style(ax, "MCTS simulation allocation")
    shown = _line(ax, rows, "mcts_budget_min", "min", _COLORS[4], ls=":", lw=1.1, alpha=0.75)
    shown |= _line(ax, rows, "mcts_budget_p10", "p10", _COLORS[1], ls="--", lw=1.4)
    shown |= _line(ax, rows, "mcts_budget_p50", "p50", _COLORS[2], lw=1.5)
    shown |= _line(ax, rows, "mcts_avg_budget", "average", _COLORS[0], lw=2.4)
    shown |= _line(ax, rows, "mcts_budget_p90", "p90", _COLORS[3], ls="--", lw=1.4)
    shown |= _line(ax, rows, "mcts_budget_max", "max", _COLORS[4], ls=":", lw=1.1, alpha=0.75)
    shown |= _line(ax, rows, "mcts_budget_target", "target average", "#222222", ls="--", lw=1.2)
    p10_x, p10_y = _series(rows, "mcts_budget_p10")
    p90_x, p90_y = _series(rows, "mcts_budget_p90")
    if p10_x and p10_x == p90_x:
        ax.fill_between(p10_x, p10_y, p90_y, color=_COLORS[0], alpha=0.08, linewidth=0)
    _legend(ax, ncol=3)
    ax.set_ylabel("simulations / position", fontsize=8)
    if not shown: _no_data(ax)

    ax = axes[3, 0]
    _style(ax, "MCTS visit shape")
    shown = _line(ax, rows, "mcts_top_visit_prob_mean", "top visit", _COLORS[0])
    shown |= _line(ax, rows, "mcts_visit_gap_mean", "visit gap", _COLORS[2])
    shown |= _line(ax, rows, "mcts_visit_entropy_mean", "entropy", _COLORS[3])
    shown |= _line(ax, rows, "mcts_good_target_rate", "good target", _COLORS[4], ls="--")
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[3, 1]
    _style(ax, "MCTS change quality")
    shown = _line(ax, rows, "mcts_prior_changed_rate", "changed", _COLORS[0])
    absolute_quality = _line(ax, rows, "mcts_useful_change_rate", "useful", _COLORS[2])
    absolute_quality |= _line(ax, rows, "mcts_harmful_change_rate", "harmful", _COLORS[1])
    if not absolute_quality:
        shown |= _line(ax, rows, "mcts_changed_to_higher_q_rate", "higher Q | changed", _COLORS[2])
        shown |= _line(ax, rows, "mcts_changed_to_lower_q_when_changed_rate", "lower Q | changed", _COLORS[1])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    delta = _line(ax2, rows, "mcts_changed_q_delta_mean", "raw Q gain", _COLORS[3], ls="--")
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False, ncol=2)
    if not (shown or delta): _no_data(ax)

    ax = axes[3, 2]
    _style(ax, "Search changes by phase", percent=True)
    shown = _line(ax, rows, "mcts_changed_opening_rate", "opening", _COLORS[0])
    shown |= _line(ax, rows, "mcts_changed_middlegame_rate", "middlegame", _COLORS[3])
    shown |= _line(ax, rows, "mcts_changed_endgame_rate", "endgame", _COLORS[1])
    _legend(ax)
    if not shown: _no_data(ax)

    _finish(fig, output_path, "RL details - replay, targets and search",
            "Diagnostics only; evaluation strength and training losses live on the main dashboard.")
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

    fig, axes = plt.subplots(4, 3, figsize=(18, 15.5))

    ax = axes[0, 0]
    _style(ax, "Self-play throughput")
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
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or visits): _no_data(ax)

    stage_specs = (
        ("stage_selfplay_time_s", "self-play", "#2563eb"),
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
        _legend(ax, ncol=min(4, max(1, visible_stage_count)), loc="best")
    else:
        _no_data(ax)

    ax = axes[0, 2]
    window = min(5, len(iterations))
    runtime_average = [
        (column, label, sum(stage_values[column][-window:]) / max(1, window), color)
        for column, label, color in stage_specs
    ]
    if window > 0:
        runtime_average.append(("other", "other", sum(other_values[-window:]) / window, "#d1d5db"))
    _composition_donut(
        ax,
        runtime_average,
        title=f"Average runtime composition (last {window})",
        center_label="avg total",
        unit="s",
        decimals=0,
        min_share=0.005,
        min_value=1.0,
    )

    ax = axes[1, 0]
    central_specs = (
        ("queue", "queue", "#f59e0b"),
        ("concat", "concat", "#0891b2"),
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
        parts = {
            "queue": queue,
            "concat": max(0.0, _number(row.get("central_server_concat_ms_per_request")) or 0.0),
            "h2d": max(0.0, _number(row.get("central_server_h2d_ms_per_request")) or 0.0),
            "forward": max(0.0, _number(row.get("central_server_forward_ms_per_request")) or 0.0),
            "d2h": max(0.0, _number(row.get("central_server_d2h_ms_per_request")) or 0.0),
        }
        known_server = parts["concat"] + parts["h2d"] + parts["forward"] + parts["d2h"]
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
    central_average, _ = _average_composition(
        rows, central_specs, _central_latency_parts, window=window,
    )
    _composition_donut(
        ax,
        central_average,
        title=f"Average central request composition (last {window})",
        center_label="avg request",
        unit="ms",
        decimals=1,
        min_share=0.01,
        min_value=0.1,
    )

    ax = axes[2, 2]
    _style(ax, "Inference volume")
    shown = _line(ax, rows, "mcts_nn_inference_calls", "NN calls", _COLORS[0])
    shown |= _line(ax, rows, "mcts_batch_expand_eval_calls", "expand calls", _COLORS[3])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    volume = _line(ax2, rows, "mcts_nn_inference_batch_items", "batch items", _COLORS[2])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or volume): _no_data(ax)

    ax = axes[2, 0]
    cpu_specs = (
        ("board_copy", "board copy/push", "#dc2626"),
        ("terminal", "terminal/draw", "#f59e0b"),
        ("legal_mixed", "legal + move encoding", "#ea580c"),
        ("legal", "python-chess legal", "#fb923c"),
        ("move_index", "move encoding", "#facc15"),
        ("legacy_leaf", "legacy leaf prep (mixed)", "#9ca3af"),
        ("selection", "tree selection", "#2563eb"),
        ("encoding", "board encoding", "#0891b2"),
        ("packing", "input packing", "#06b6d4"),
        ("policy", "CPU policy", "#16a34a"),
        ("backprop", "backprop/other", "#7c3aed"),
    )

    def _mcts_cpu_parts(row):
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
        cpu_specs,
        _mcts_cpu_parts,
        title="Measured MCTS CPU composition",
        ylabel="worker-summed seconds",
        legend_columns=4,
    )

    cpu_average, _ = _average_composition(
        rows, cpu_specs, _mcts_cpu_parts, window=window,
    )
    _composition_donut(
        axes[2, 1],
        cpu_average,
        title=f"Average measured MCTS CPU (last {window})",
        center_label="avg measured",
        unit="s",
        decimals=0,
        min_share=0.01,
        min_value=1.0,
        autopct_min=7.0,
    )

    ax = axes[3, 0]
    _style(ax, "Search work throughput")
    shown = _line(ax, rows, "replay_positions_per_sec", "replay positions/s", _COLORS[0])
    shown |= _line(ax, rows, "played_positions_per_sec", "played positions/s", _COLORS[2])
    ax.set_ylabel("positions/s", fontsize=8)
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    simulations = _line(ax2, rows, "mcts_simulations_per_sec", "simulations/s", _COLORS[3])
    if simulations:
        ax2.set_ylabel("simulations/s", fontsize=8)
    else:
        ax2.set_visible(False)
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or simulations): _no_data(ax)
    latest_path = _last(rows, "mcts_selection_path_length")
    if latest_path is not None:
        ax.text(
            0.98, 0.04, f"latest selection path: {latest_path:.2f} traversals/sim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#4b5563",
        )

    ax = axes[3, 1]
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
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or wait_share): _no_data(ax)

    ax = axes[3, 2]
    ax.axis("off")
    stage = {
        "self-play": _mean(rows, "stage_selfplay_time_s") or 0.0,
        "train": _mean(rows, "stage_train_time_s") or 0.0,
        "promotion eval": _mean(rows, "stage_promotion_eval_time_s") or 0.0,
        "regular eval": _mean(rows, "stage_regular_eval_time_s") or 0.0,
        "Elo eval": _mean(rows, "stage_elo_eval_time_s") or 0.0,
        "decision/log": _mean(rows, "stage_log_time_s") or 0.0,
        "legacy eval/log": _mean(rows, "stage_eval_log_time_s") or 0.0,
        "replay": _mean(rows, "stage_replay_time_s") or 0.0,
    }
    bottleneck = max(stage, key=stage.get) if any(stage.values()) else "n/a"
    latest_traversals = _last(rows, "mcts_selection_node_traversals_per_sec")
    latest_cpu_parts = _mcts_cpu_parts(rows[-1])
    latest_central_parts = _central_latency_parts(rows[-1])
    cpu_labels = {key: label for key, label, _ in cpu_specs}
    central_labels = {key: label for key, label, _ in central_specs}
    cpu_hotspot = max(latest_cpu_parts, key=latest_cpu_parts.get) if any(latest_cpu_parts.values()) else None
    central_hotspot = max(latest_central_parts, key=latest_central_parts.get) if any(latest_central_parts.values()) else None
    summary = [
        ("Replay positions", f"{_fmt(_last(rows, 'replay_positions_per_sec'))} pos/s"),
        ("Played positions", f"{_fmt(_last(rows, 'played_positions_per_sec'))} pos/s"),
        ("Completed MCTS visits", f"{_fmt(_last(rows, 'mcts_simulations_per_sec'), 'integer')} visits/s"),
        ("NN evaluations", f"{_fmt(_last(rows, 'mcts_nn_evaluations_per_sec'), 'integer')} evals/s"),
        ("Selection traversals", f"{_fmt(latest_traversals, 'integer')} nodes/s"),
        ("Avg selection path", f"{_fmt(_last(rows, 'mcts_selection_path_length'))} nodes/sim"),
        ("Iteration", _fmt(_last(rows, "iteration_total_time_s"), "seconds")),
        ("Shared batch/request", _fmt(_last(rows, "mcts_central_avg_batch_size"))),
        ("Request latency", f"{_fmt(_last(rows, 'central_remote_wait_ms_per_request'))} ms"),
        ("Central bottleneck", (
            f"{central_labels[central_hotspot]} | {_fmt(latest_central_parts[central_hotspot])} ms"
            if central_hotspot else "n/a"
        )),
        ("CPU hot spot", (
            f"{cpu_labels[cpu_hotspot]} | {_fmt(latest_cpu_parts[cpu_hotspot])} s"
            if cpu_hotspot else "n/a"
        )),
        ("Worker NN wait share", _fmt((
            _last(
                rows,
                "mcts_worker_nn_wait_share_pct",
                _last(rows, "mcts_gpu_busy_proxy_pct", _last(rows, "mcts_gpu_utilization_pct", 0.0)),
            ) or 0.0
        ) / 100.0, "percent")),
        ("Main stage", bottleneck),
    ]
    ax.set_title("Performance snapshot", fontsize=10, fontweight="bold", loc="left")
    table = ax.table(cellText=summary, colLabels=("Signal", "Latest"), loc="center",
                     cellLoc="left", colWidths=(0.58, 0.34))
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1.0, 1.32)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if row == 0:
            cell.set_facecolor("#eff6ff"); cell.set_text_props(weight="bold")

    _finish(
        fig,
        output_path,
        "RL details - performance",
        "Completed MCTS visit = one finished simulation; selection traversals count root-to-leaf path elements. "
        "Worker NN wait share is not hardware GPU utilization.",
    )
    return True
