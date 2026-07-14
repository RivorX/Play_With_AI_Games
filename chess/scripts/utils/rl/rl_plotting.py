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

    ax = axes[2, 0]
    _style(ax, "Training losses")
    shown = _line(ax, main, "avg_loss", "total", "#111827", lw=2.2)
    shown |= _line(ax, main, "policy_loss", "policy", _COLORS[0])
    shown |= _line(ax, main, "value_loss", "value", _COLORS[3])
    _legend(ax, ncol=3)
    if not shown:
        _no_data(ax)

    ax = axes[2, 1]
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
        ("Throughput", f"{_fmt(_last(perf, 'positions_per_sec'), 'integer')} pos/s"),
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
    _style(ax, "MCTS budget and coverage")
    shown = _line(ax, rows, "mcts_avg_sims", "simulations", _COLORS[0])
    shown |= _line(ax, rows, "mcts_avg_budget", "budget", _COLORS[1], ls="--")
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    coverage = _line(ax2, rows, "mcts_explored_prior_mass_mean", "prior covered", _COLORS[2])
    coverage |= _line(ax2, rows, "mcts_visit_coverage_ratio_mean", "moves visited", _COLORS[3])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False, ncol=2)
    if not (shown or coverage): _no_data(ax)

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


def render_rl_performance(performance_csv_path, output_path):
    """Render the minimum set needed to locate a throughput bottleneck."""
    rows = _rows(performance_csv_path)
    if not rows:
        return False
    fig, axes = plt.subplots(3, 3, figsize=(18, 11.5))

    ax = axes[0, 0]
    _style(ax, "Throughput and central batch")
    shown = _line(ax, rows, "positions_per_sec", "positions/s", _COLORS[0])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    batch = _line(ax2, rows, "mcts_central_avg_batch_size", "central batch", _COLORS[2])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or batch): _no_data(ax)

    stage_specs = (
        ("stage_selfplay_time_s", "self-play", "#2563eb"),
        ("stage_replay_time_s", "replay", "#0891b2"),
        ("stage_train_time_s", "train", "#16a34a"),
        ("stage_eval_log_time_s", "eval/log", "#f59e0b"),
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
        for column, label, color in stage_specs:
            values = stage_values[column]
            ax.bar(iterations, values, bottom=bottoms, width=0.76, color=color,
                   label=label, edgecolor="white", linewidth=0.35)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        if any(other_values):
            ax.bar(iterations, other_values, bottom=bottoms, width=0.76, color="#d1d5db",
                   label="other/unaccounted", edgecolor="white", linewidth=0.35)
        if len(iterations) <= 12:
            padding = max(totals) * 0.015
            for iteration, total in zip(iterations, totals):
                ax.text(iteration, total + padding, f"{total:.0f}s", ha="center", va="bottom",
                        fontsize=6.5, color="#374151")
        ax.set_ylim(0.0, max(totals) * 1.10)
        ax.set_ylabel("seconds", fontsize=8)
        ax.set_axisbelow(True)
        ax.legend(
            fontsize=6.5,
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
        )
    else:
        _no_data(ax)

    ax = axes[0, 2]
    window = min(5, len(iterations))
    _style(ax, f"Average runtime composition (last {window})")
    if window > 0 and any(totals[-window:]):
        average_specs = []
        for column, label, color in stage_specs:
            values = stage_values[column][-window:]
            average_specs.append((label, sum(values) / window, color))
        if any(other_values[-window:]):
            average_specs.append(("other", sum(other_values[-window:]) / window, "#d1d5db"))
        left = 0.0
        average_total = sum(value for _, value, _ in average_specs)
        for label, value, color in average_specs:
            if value <= 0.0:
                continue
            ax.barh([0], [value], left=left, height=0.52, color=color, label=label,
                    edgecolor="white", linewidth=0.5)
            if average_total > 0.0 and value / average_total >= 0.08:
                ax.text(left + value / 2.0, 0, f"{label}\n{value:.0f}s", ha="center", va="center",
                        fontsize=7, color="white" if color != "#d1d5db" else "#111827",
                        fontweight="bold")
            left += value
        ax.set_yticks([])
        ax.set_xlabel("seconds", fontsize=8)
        ax.set_ylim(-0.65, 0.65)
        _legend(ax, ncol=2, loc="lower center")
    else:
        _no_data(ax)

    ax = axes[1, 0]
    _style(ax, "Central request latency")
    shown = _line(ax, rows, "central_remote_wait_ms_per_request", "remote wait", _COLORS[1])
    shown |= _line(ax, rows, "central_server_queue_wait_ms_per_request", "queue", _COLORS[4])
    shown |= _line(ax, rows, "central_server_forward_ms_per_request", "forward", _COLORS[2])
    shown |= _line(ax, rows, "central_server_total_ms_per_request", "server total", _COLORS[0], lw=2.2)
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[1, 1]
    _style(ax, "Latency per position")
    shown = _line(ax, rows, "mcts_worker_nn_wait_ms_per_position", "worker wait/pos", _COLORS[1])
    shown |= _line(ax, rows, "central_server_queue_wait_ms_per_position", "queue/pos", _COLORS[4])
    shown |= _line(ax, rows, "central_server_forward_ms_per_position", "forward/pos", _COLORS[2])
    shown |= _line(ax, rows, "central_server_total_ms_per_position", "server total/pos", _COLORS[0], lw=2.2)
    _legend(ax)
    if not shown: _no_data(ax)

    ax = axes[1, 2]
    _style(ax, "Batching and occupancy")
    shown = _line(ax, rows, "mcts_avg_batch_size", "worker batch", _COLORS[0])
    shown |= _line(ax, rows, "mcts_central_avg_batch_size", "central batch", _COLORS[2])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8); ax2.yaxis.set_major_formatter(PercentFormatter(100.0))
    gpu = _line(ax2, rows, "mcts_gpu_busy_proxy_pct", "GPU busy proxy", _COLORS[3])
    if not gpu:
        gpu = _line(ax2, rows, "mcts_gpu_utilization_pct", "GPU busy proxy", _COLORS[3])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or gpu): _no_data(ax)

    ax = axes[2, 0]
    _style(ax, "Inference volume")
    shown = _line(ax, rows, "mcts_nn_inference_calls", "NN calls", _COLORS[0])
    shown |= _line(ax, rows, "mcts_batch_expand_eval_calls", "expand calls", _COLORS[3])
    ax2 = ax.twinx(); ax2.tick_params(labelsize=8)
    volume = _line(ax2, rows, "mcts_nn_inference_batch_items", "batch items", _COLORS[2])
    h, l = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    if h or h2: ax.legend(h + h2, l + l2, fontsize=7, frameon=False)
    if not (shown or volume): _no_data(ax)

    ax = axes[2, 1]
    _style(ax, "MCTS CPU cost (last 5)")
    cpu_columns = (
        ("mcts_batch_expand_eval_time_s", "expand/eval"), ("mcts_batch_expand_tensor_pack_time_s", "tensor pack"),
        ("mcts_search_selection_time_s", "selection"), ("mcts_search_backprop_time_s", "backprop"),
        ("mcts_policy_target_build_time_s", "target build"), ("mcts_move_selection_time_s", "move select"),
    )
    pairs = [(label, _mean(rows, column) or 0.0) for column, label in cpu_columns]
    pairs.sort(key=lambda pair: pair[1])
    if any(value for _, value in pairs):
        ax.barh([label for label, _ in pairs], [value for _, value in pairs], color=_COLORS[3], alpha=0.82)
        ax.set_xlabel("worker-summed seconds", fontsize=8)
    else: _no_data(ax)

    ax = axes[2, 2]
    ax.axis("off")
    stage = {
        "self-play": _mean(rows, "stage_selfplay_time_s") or 0.0,
        "train": _mean(rows, "stage_train_time_s") or 0.0,
        "eval/log": _mean(rows, "stage_eval_log_time_s") or 0.0,
        "replay": _mean(rows, "stage_replay_time_s") or 0.0,
    }
    bottleneck = max(stage, key=stage.get) if any(stage.values()) else "n/a"
    summary = [
        ("Latest throughput", f"{_fmt(_last(rows, 'positions_per_sec'), 'integer')} pos/s"),
        ("5-it throughput", f"{_fmt(_mean(rows, 'positions_per_sec'), 'integer')} pos/s"),
        ("Iteration", _fmt(_last(rows, "iteration_total_time_s"), "seconds")),
        ("Central batch", _fmt(_last(rows, "mcts_central_avg_batch_size"))),
        ("Request latency", f"{_fmt(_last(rows, 'central_remote_wait_ms_per_request'))} ms"),
        ("Latency / position", f"{_fmt(_last(rows, 'mcts_worker_nn_wait_ms_per_position'))} ms"),
        ("GPU busy proxy", _fmt((
            _last(rows, "mcts_gpu_busy_proxy_pct", _last(rows, "mcts_gpu_utilization_pct", 0.0)) or 0.0
        ) / 100.0, "percent")),
        ("Main stage", bottleneck),
    ]
    ax.set_title("Performance snapshot", fontsize=10, fontweight="bold", loc="left")
    table = ax.table(cellText=summary, colLabels=("Signal", "Latest"), loc="center",
                     cellLoc="left", colWidths=(0.58, 0.34))
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if row == 0:
            cell.set_facecolor("#eff6ff"); cell.set_text_props(weight="bold")

    _finish(fig, output_path, "RL details - performance",
            "Each panel answers one question: speed, stage cost, latency, batching or CPU overhead.")
    return True
