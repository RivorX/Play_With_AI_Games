"""Shared model presentation helpers for train/eval scripts."""

from __future__ import annotations

from utils.shared.model_catalog import print_model_table, sort_entries_by_folder_and_elo


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_int(value):
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_percent(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_loss(value):
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _count_params(module, trainable_only=False):
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _print_kv_table(title, rows):
    if not rows:
        return

    key_w = max(len("Field"), max(len(str(k)) for k, _ in rows))
    val_w = max(len("Value"), max(len(str(v)) for _, v in rows))
    width = key_w + val_w + 7

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"| {'Field':<{key_w}} | {'Value':<{val_w}} |")
    print(f"| {'-' * key_w} | {'-' * val_w} |")
    for key, value in rows:
        print(f"| {str(key):<{key_w}} | {str(value):<{val_w}} |")
    print("=" * width)


def _print_multi_column_table(title, headers, rows):
    if not headers:
        return

    header_cells = [str(cell) for cell in headers]
    col_count = len(header_cells)
    normalized_rows = []

    for row in rows:
        cells = [str(cell) for cell in row]
        if len(cells) < col_count:
            cells.extend([""] * (col_count - len(cells)))
        elif len(cells) > col_count:
            cells = cells[:col_count]
        normalized_rows.append(cells)

    col_widths = [len(cell) for cell in header_cells]
    for row in normalized_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    table_lines = [
        "| " + " | ".join(f"{header_cells[idx]:<{col_widths[idx]}}" for idx in range(col_count)) + " |",
        "| " + " | ".join("-" * col_widths[idx] for idx in range(col_count)) + " |",
    ]

    for row in normalized_rows:
        table_lines.append(
            "| " + " | ".join(f"{row[idx]:<{col_widths[idx]}}" for idx in range(col_count)) + " |"
        )

    width = max(len(title), max(len(line) for line in table_lines))
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    for line in table_lines:
        print(line)
    print("=" * width)


def _print_param_table(title, rows):
    if not rows:
        return

    part_w = max(len("Part"), max(len(str(part)) for part, _, _ in rows))
    params_w = max(len("Params"), max(len(str(params)) for _, params, _ in rows))
    share_w = max(len("Share"), max(len(str(share)) for _, _, share in rows))
    width = part_w + params_w + share_w + 10

    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"| {'Part':<{part_w}} | {'Params':>{params_w}} | {'Share':>{share_w}} |")
    print(f"| {'-' * part_w} | {'-' * params_w} | {'-' * share_w} |")
    for part, params, share in rows:
        print(f"| {str(part):<{part_w}} | {str(params):>{params_w}} | {str(share):>{share_w}} |")
    print("=" * width)


def _build_model_rows(
    model,
    config,
    *,
    source_label=None,
    startup_mode=None,
    checkpoint_label=None,
    load_mode=None,
    device=None,
    selected_entry=None,
):
    model_cfg = config.get("model", {})
    history = _safe_int(getattr(model, "history_positions", model_cfg.get("history_positions", 0)))
    input_planes = _safe_int(getattr(model, "input_planes", None))
    if input_planes is None and history is not None:
        input_planes = 16 * (1 + history)

    version = model_cfg.get("version", getattr(model, "model_version", "n/a"))
    filters = _safe_int(model_cfg.get("filters"))
    blocks = _safe_int(model_cfg.get("num_residual_blocks"))
    dropout = _safe_float(model_cfg.get("dropout"))
    drop_path_rate = _safe_float(model_cfg.get("drop_path_rate"))
    policy_groups = _safe_int(model_cfg.get("policy_head_conv_groups"))
    use_coord_conv = bool(model_cfg.get("use_coord_conv", False))
    use_se = bool(model_cfg.get("use_se_blocks", False))
    use_se_bottleneck = bool(model_cfg.get("use_se_bottleneck", True))
    se_reduction = _safe_int(model_cfg.get("se_reduction"))
    use_layer_scale = bool(model_cfg.get("use_layer_scale", False))
    layer_scale_init = model_cfg.get("layer_scale_init")
    use_mtl = bool(model_cfg.get("use_multitask_learning", False))

    rows = [("Version", version)]
    if source_label:
        rows.append(("Source", source_label))
    if startup_mode:
        rows.append(("Startup mode", startup_mode))
    if checkpoint_label:
        rows.append(("Checkpoint", str(checkpoint_label)))
    if load_mode:
        rows.append(("Load mode", load_mode))
    if device is not None:
        rows.append(("Device", str(device)))
    if blocks is not None:
        rows.append(("Residual blocks", blocks))
    if filters is not None:
        rows.append(("Filters", filters))
    if history is not None:
        rows.append(("History positions", history))
    if input_planes is not None and history is not None:
        rows.append(("Input planes", f"{input_planes} (16 x {1 + history})"))
    elif input_planes is not None:
        rows.append(("Input planes", input_planes))
    if dropout is not None:
        rows.append(("Dropout", f"{dropout:.3f}"))

    if use_coord_conv:
        rows.append(("CoordConv", "enabled"))
    if use_se:
        if use_se_bottleneck and se_reduction is not None:
            rows.append(("SE blocks", f"enabled (bottleneck, r={se_reduction})"))
        elif use_se_bottleneck:
            rows.append(("SE blocks", "enabled (bottleneck)"))
        else:
            rows.append(("SE blocks", "enabled (no bottleneck)"))
    if use_layer_scale:
        if layer_scale_init is None:
            rows.append(("LayerScale", "enabled"))
        else:
            rows.append(("LayerScale", f"enabled (init={layer_scale_init})"))
    if drop_path_rate is not None and drop_path_rate > 0:
        rows.append(("Stochastic depth", f"{drop_path_rate:.3f}"))
    if policy_groups is not None and policy_groups > 1:
        rows.append(("Policy conv groups", policy_groups))
    if use_mtl:
        rows.append(("MTL heads", "enabled"))

    entry = selected_entry or {}
    if entry.get("error"):
        rows.append(("Checkpoint scan", f"error: {entry['error']}"))
        return rows

    epoch = _safe_int(entry.get("epoch"))
    top1 = _safe_float(entry.get("top1"))
    val_loss = _safe_float(entry.get("val_loss"))
    policy_loss = _safe_float(entry.get("policy_loss"))
    elo = _safe_float(entry.get("elo", entry.get("estimated_elo")))
    compat = _safe_float(entry.get("compatibility_ratio"))
    strict_ok = entry.get("strict_resume_ok")

    if epoch is not None:
        rows.append(("Selected epoch", epoch + 1))
    if top1 is not None:
        rows.append(("Selected Top1", _fmt_percent(top1)))
    if val_loss is not None:
        rows.append(("Selected ValLoss", _fmt_loss(val_loss)))
    if policy_loss is not None:
        rows.append(("Selected PolLoss", _fmt_loss(policy_loss)))
    if elo is not None:
        rows.append(("Selected Elo", int(round(elo))))
    if compat is not None:
        rows.append(("Compatibility", f"{compat * 100:.2f}%"))
    if strict_ok is not None:
        rows.append(("Strict resume", "yes" if strict_ok else "no"))

    return rows


def _build_parameter_rows(model, num_blocks):
    stem_params = _count_params(getattr(model, "conv_block", None))
    tower_params = _count_params(getattr(model, "residual_tower", None))
    final_bn_params = _count_params(getattr(model, "final_bn", None))

    policy_params = sum(
        _count_params(getattr(model, name, None))
        for name in ("policy_conv", "policy_bn", "policy_global_fc", "policy_fc1", "policy_fc2")
    )
    value_params = sum(
        _count_params(getattr(model, name, None))
        for name in ("value_conv", "value_bn", "value_fc1", "value_fc2")
    )

    mtl_params = 0
    if getattr(model, "use_mtl", False):
        mtl_params = sum(
            _count_params(getattr(model, name, None))
            for name in ("win_fc1", "win_fc2", "material_fc1", "material_fc2", "check_fc")
        )

    total_params = _count_params(model)
    trainable_params = _count_params(model, trainable_only=True)
    frozen_params = max(0, total_params - trainable_params)
    per_block = tower_params // max(1, int(num_blocks or 1))

    core_parts = [
        ("Stem", stem_params),
        (f"Residual tower ({int(num_blocks or 0)} blocks, ~{per_block:,}/block)", tower_params),
        ("Final BN", final_bn_params),
        ("Policy head", policy_params),
        ("Value head", value_params),
    ]
    if mtl_params > 0:
        core_parts.append(("MTL heads", mtl_params))

    rows = []
    for part, params in core_parts:
        if params <= 0:
            continue
        share = (params / total_params * 100.0) if total_params > 0 else 0.0
        rows.append((part, f"{params:,}", f"{share:6.2f}%"))

    rows.append(("Trainable", _fmt_int(trainable_params), "100.00%" if trainable_params == total_params else ""))
    if frozen_params > 0:
        share = (frozen_params / total_params * 100.0) if total_params > 0 else 0.0
        rows.append(("Frozen", _fmt_int(frozen_params), f"{share:6.2f}%"))
    rows.append(("Total", _fmt_int(total_params), "100.00%"))
    return rows


def print_active_model_summary(
    model,
    config,
    *,
    title="Active Model",
    source_label=None,
    startup_mode=None,
    checkpoint_label=None,
    load_mode=None,
    device=None,
    selected_entry=None,
):
    """Print concise model and parameter summaries as tables."""
    model_rows = _build_model_rows(
        model,
        config,
        source_label=source_label,
        startup_mode=startup_mode,
        checkpoint_label=checkpoint_label,
        load_mode=load_mode,
        device=device,
        selected_entry=selected_entry,
    )
    _print_kv_table(title, model_rows)

    num_blocks = _safe_int(config.get("model", {}).get("num_residual_blocks"))
    param_rows = _build_parameter_rows(model, num_blocks)
    _print_param_table(f"{title} - Parameters", param_rows)


def print_selected_models_table(entries, title="Selected Models", keep_input_order=True):
    """Print selected checkpoint entries using the shared model table."""
    if not entries:
        print("\nNo selected models.")
        return

    if keep_input_order:
        ordered = list(entries)
    else:
        ordered = sort_entries_by_folder_and_elo(list(entries))
    show_compat = any(_safe_float(e.get("compatibility_ratio")) is not None for e in ordered)
    show_strict = any("strict_resume_ok" in e for e in ordered)

    print_model_table(
        ordered,
        title=title,
        show_folder=True,
        show_version=True,
        show_modified=True,
        show_swa=True,
        show_opt=True,
        show_compat=show_compat,
        show_strict=show_strict,
        group_by_folder=not keep_input_order,
    )


def print_status_table(title, rows):
    """Print a simple key/value status table."""
    _print_kv_table(title, rows)


def print_multi_column_table(title, headers, rows):
    """Print a simple fixed-width table with any number of columns."""
    _print_multi_column_table(title, headers, rows)
