"""Small CSV and plotting helpers shared by the IL and RL loggers."""

import csv
import json

_CSV_CONFIG_METADATA_KEY = "# config_json"
_CSV_RUN_SUMMARY_METADATA_KEY = "# run_summary_json"
_CSV_RESUME_METADATA_KEY = "# resume_json"


def _clean_optional_float(value):
    try:
        if value in (None, ''):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_elo_ci95(ci95):
    if not isinstance(ci95, (list, tuple)) or len(ci95) != 2:
        return None, None
    low = _clean_optional_float(ci95[0])
    high = _clean_optional_float(ci95[1])
    return low, high


def _format_million_count(value):
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return "n/a"
    if value <= 0:
        return "0M"
    millions = value / 1_000_000.0
    if millions >= 10.0:
        return f"{millions:.0f}M"
    if millions >= 1.0:
        return f"{millions:.1f}M"
    return f"{value / 1000.0:.0f}k"


def _is_metadata_row(row):
    return bool(row) and str(row[0]).strip().startswith("#")


def _json_safe_config(value):
    if isinstance(value, dict):
        return {str(k): _json_safe_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_config_metadata_row(row):
    return bool(row) and str(row[0]).strip() == _CSV_CONFIG_METADATA_KEY


def _is_run_summary_metadata_row(row):
    return bool(row) and str(row[0]).strip() == _CSV_RUN_SUMMARY_METADATA_KEY


def _read_csv_rows_preserving_metadata(csv_path):
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.reader(f))
    metadata_rows = []
    while rows and _is_metadata_row(rows[0]):
        metadata_rows.append(rows[0])
        rows = rows[1:]
    return metadata_rows, rows


def _upsert_metadata_row(metadata_rows, key, payload):
    payload_json = json.dumps(
        _json_safe_config(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    new_row = [key, payload_json]
    filtered = [row for row in metadata_rows if not (row and str(row[0]).strip() == key)]
    if key == _CSV_RUN_SUMMARY_METADATA_KEY:
        return [new_row] + filtered
    if key == _CSV_CONFIG_METADATA_KEY:
        insert_at = 1 if filtered and _is_run_summary_metadata_row(filtered[0]) else 0
        return filtered[:insert_at] + [new_row] + filtered[insert_at:]
    return filtered + [new_row]


def _read_csv_dict_rows(csv_path):
    metadata_rows, rows = _read_csv_rows_preserving_metadata(csv_path)
    if not rows:
        return []
    header = list(rows[0])
    result = []
    for raw_row in rows[1:]:
        row = list(raw_row)
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        result.append(dict(zip(header, row[:len(header)])))
    return result


def _metadata_json_payload(metadata_rows, key):
    for row in metadata_rows or []:
        if row and str(row[0]).strip() == key and len(row) >= 2:
            try:
                return json.loads(row[1])
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
    return None


def _csv_float(row, key, default=None):
    try:
        value = row.get(key, '')
        if value in (None, ''):
            return default
        value = float(value)
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def _csv_int(row, key, default=None):
    value = _csv_float(row, key, default=None)
    if value is None:
        return default
    try:
        return int(round(value))
    except (TypeError, ValueError):
        return default


def _legend_display_y(handle):
    """Return the display-space y of a legend handle's latest finite point."""
    ax = getattr(handle, "axes", None)
    if ax is None:
        return None
    try:
        xdata = list(handle.get_xdata(orig=False))
        ydata = list(handle.get_ydata(orig=False))
    except Exception:
        return None
    if not xdata or not ydata:
        return None
    for x_value, y_value in reversed(list(zip(xdata, ydata))):
        try:
            x_float = float(x_value)
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if y_float != y_float or x_float != x_float:
            continue
        try:
            return float(ax.transData.transform((x_float, y_float))[1])
        except Exception:
            return None
    return None


def _sorted_legend_items(handles, labels):
    items = [
        (idx, handle, label, _legend_display_y(handle))
        for idx, (handle, label) in enumerate(zip(handles or [], labels or []))
        if label and not str(label).startswith("_")
    ]
    items.sort(key=lambda item: (item[3] is not None, item[3] if item[3] is not None else -item[0]), reverse=True)
    return [item[1] for item in items], [item[2] for item in items]


def _apply_sorted_legend(ax, handles=None, labels=None, **kwargs):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    handles, labels = _sorted_legend_items(handles, labels)
    if handles:
        return ax.legend(handles, labels, **kwargs)
    return None
