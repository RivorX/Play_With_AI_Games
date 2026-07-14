"""Compact, knowledge-dense console rendering for RL training."""

from __future__ import annotations

from collections.abc import Iterable


def format_duration(seconds):
    """Format a duration without wasting console width."""
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "-"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def compact_path(path, keep_parts=3):
    """Keep the useful tail of a path for console summaries."""
    if path is None:
        return "-"
    text = str(path).replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    if len(parts) <= keep_parts:
        return text
    return ".../" + "/".join(parts[-keep_parts:])


def print_panel(title: str, rows: Iterable[tuple[str, object]], *, width=100, notes=None):
    """Print a stable ASCII panel that also works in legacy Windows terminals."""
    width = max(64, int(width))
    label_width = 11
    line = "=" * width
    print(f"\n{line}")
    print(f" {title}")
    print("-" * width)
    for label, value in rows:
        if value in (None, ""):
            continue
        print(f" {str(label).upper():<{label_width}} {value}")
    clean_notes = [str(note).strip() for note in (notes or []) if str(note).strip()]
    if clean_notes:
        print("-" * width)
        for note in clean_notes:
            print(f" NOTE        {note}")
    print(line)


def dominant_stage(stage_times):
    """Return the slowest meaningful stage and its share of measured time."""
    stage_times = dict(stage_times or {})
    measured = {
        str(key): max(0.0, float(value or 0.0))
        for key, value in stage_times.items()
        if str(key) not in {"total", "other"}
    }
    total = sum(measured.values())
    if not measured or total <= 0.0:
        return "-", 0.0
    name, seconds = max(measured.items(), key=lambda item: item[1])
    return name, seconds / total

