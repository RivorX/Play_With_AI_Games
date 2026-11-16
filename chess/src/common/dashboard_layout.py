"""Shared category based layout for training dashboards."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


_BACKGROUND = "#f3f5f8"
_CARD = "#ffffff"
_BORDER = "#dce2e9"
_SECTION = "#e9eef4"


def create_dashboard(sections, *, width=19.0, panel_height=3.5):
    """Create a titled, vertically growing dashboard with categorized panels.

    ``sections`` contains ``(title, keys, columns)`` tuples. Keys are returned
    in a dictionary and may be any hashable value. A four column grid is
    supported for compact categories such as donut charts.
    """
    if width <= 0 or panel_height <= 0:
        raise ValueError("width and panel_height must be positive")

    normalized = []
    seen = set()
    for title, keys, columns in sections:
        keys = list(keys)
        columns = int(columns)
        if columns < 1 or columns > 4:
            raise ValueError("section columns must be between 1 and 4")
        if len(set(keys)) != len(keys):
            raise ValueError("panel keys must be unique within a section")
        duplicates = seen.intersection(keys)
        if duplicates:
            raise ValueError(f"panel keys must be unique across sections: {duplicates!r}")
        seen.update(keys)
        if keys:
            normalized.append((str(title), keys, columns))

    header_inches = 1.35
    footer_inches = 0.32
    section_band_inches = 0.34
    section_gap_inches = 0.22
    chart_top_inches = 0.25
    standard_legend_inches = 0.90
    donut_legend_inches = 1.85
    chart_row_gap_inches = 0.20
    row_counts = [math.ceil(len(keys) / columns) for _, keys, columns in normalized]
    legend_reserves = [
        donut_legend_inches if columns == 4 else standard_legend_inches
        for _, _, columns in normalized
    ]
    height = header_inches + footer_inches
    height += sum(
        section_band_inches
        + rows * (chart_top_inches + panel_height + reserve)
        + max(0, rows - 1) * chart_row_gap_inches
        for rows, reserve in zip(row_counts, legend_reserves)
    )
    height += max(0, len(normalized) - 1) * section_gap_inches

    fig = plt.figure(figsize=(width, height), facecolor=_BACKGROUND)
    fig._training_dashboard_layout = True
    fig._training_dashboard_header_inches = header_inches
    fig._training_dashboard_legend_reserves_inches = tuple(legend_reserves)
    axes = {}
    outer_heights = []
    for index, (rows, reserve) in enumerate(zip(row_counts, legend_reserves)):
        body_height = (
            rows * (chart_top_inches + panel_height + reserve)
            + max(0, rows - 1) * chart_row_gap_inches
        )
        outer_heights.extend((section_band_inches, body_height))
        if index < len(row_counts) - 1:
            outer_heights.append(section_gap_inches)

    if outer_heights:
        grid = fig.add_gridspec(
            len(outer_heights), 1,
            height_ratios=outer_heights,
            left=0.045, right=0.97,
            top=1.0 - header_inches / height,
            bottom=footer_inches / height,
            hspace=0,
        )
        grid_row = 0
        for section_index, (title, keys, columns) in enumerate(normalized):
            band = fig.add_subplot(grid[grid_row, 0])
            band.set_facecolor(_SECTION)
            band.set_xticks([])
            band.set_yticks([])
            for spine in band.spines.values():
                spine.set_visible(False)
            band.text(
                0.012, 0.5, title,
                transform=band.transAxes,
                va="center", ha="left",
                color="#263445", fontsize=11, fontweight="bold",
            )

            rows = math.ceil(len(keys) / columns)
            body_rows = []
            axis_row_indices = []
            legend_reserve = legend_reserves[section_index]
            for row in range(rows):
                axis_row_indices.append(len(body_rows) + 1)
                body_rows.extend((chart_top_inches, panel_height, legend_reserve))
                if row < rows - 1:
                    body_rows.append(chart_row_gap_inches)
            panel_grid = GridSpecFromSubplotSpec(
                len(body_rows), columns,
                subplot_spec=grid[grid_row + 1, 0],
                height_ratios=body_rows,
                hspace=0,
                wspace=0.24,
            )
            grid_row += 2
            for index, key in enumerate(keys):
                ax = fig.add_subplot(
                    panel_grid[axis_row_indices[index // columns], index % columns]
                )
                ax.set_facecolor(_CARD)
                for spine in ax.spines.values():
                    spine.set_color(_BORDER)
                    spine.set_linewidth(0.8)
                ax.tick_params(colors="#536273", labelsize=8)
                ax.xaxis.label.set_color("#536273")
                ax.yaxis.label.set_color("#536273")
                ax.title.set_color("#263445")
                axes[key] = ax
            if grid_row < len(outer_heights):
                grid_row += 1

    return fig, axes


def finish_dashboard(fig, output_path, title, subtitle=None):
    """Add report headers, save at 150 DPI, and close the figure."""
    width, height = fig.get_size_inches()
    header_inches = getattr(fig, "_training_dashboard_header_inches", 0.0)
    subtitle_lines = []
    if subtitle is not None:
        raw_lines = subtitle if isinstance(subtitle, (list, tuple)) else [subtitle]
        wrap_width = max(55, int(width * 12))
        for item in raw_lines:
            subtitle_lines.extend(textwrap.wrap(str(item), width=wrap_width) or [""])
    if subtitle_lines:
        subtitle_top_inches = 0.70
        subtitle_line_inches = 0.16
        needed_header = subtitle_top_inches + len(subtitle_lines) * subtitle_line_inches + 0.15
        if needed_header > header_inches:
            extra = needed_header - header_inches
            old_height = height
            height += extra
            fig.set_size_inches(width, height, forward=True)
            for ax in fig.axes:
                position = ax.get_position()
                ax.set_position((
                    position.x0,
                    position.y0 * old_height / height,
                    position.width,
                    position.height * old_height / height,
                ))
            fig._training_dashboard_header_inches = needed_header

    fig.text(
        0.045, 1.0 - 0.43 / height, str(title),
        ha="left", va="center", fontsize=20, fontweight="bold",
        color="#172433",
    )
    if subtitle_lines:
        fig.text(
            0.046, 1.0 - 0.70 / height, "\n".join(subtitle_lines),
            ha="left", va="top", fontsize=9.5, color="#536273",
            linespacing=1.2,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
