"""Interactive Elo estimation tool for chess models.

This script intentionally does not accept CLI arguments.
Run it without parameters and use the menu.
"""

import argparse
import contextlib
import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.config import default_config_path, load_project_config
from src.common.runtime import (
    legacy_architecture_scoped_file,
    model_alias_file,
)
from src.model import (
    create_model,
    load_checkpoint_file,
    normalize_state_dict_keys,
    resolve_model_spec,
    transfer_matching_weights,
)
from src.evaluation.elo_estimator import ensure_stockfish
from src.evaluation.elo_runner import (
    apply_checkpoint_elo_seed,
    build_eval_elo_config,
    elo_protocol_settings,
    mode_label,
    persist_estimated_elo,
    resolve_eval_workers,
    run_elo_check,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from src.models.catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_elo,
)
from src.models.view import print_selected_models_table


def _is_tty():
    return sys.stdin.isatty()


def _print_block(title):
    line = "=" * 96
    print(f"\n{line}")
    print(title)
    print(line)


def _prompt_int(prompt, default, min_value=None):
    default = int(default)
    if not _is_tty():
        return default

    while True:
        try:
            raw = input(f"{prompt} (default {default}): ").strip()
        except EOFError:
            raw = ""

        if not raw:
            return default

        try:
            value = int(raw)
        except ValueError:
            print("Invalid integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_float(prompt, default, min_value=None):
    default = float(default)
    if not _is_tty():
        return default

    while True:
        try:
            raw = input(f"{prompt} (default {default}): ").strip()
        except EOFError:
            raw = ""

        if not raw:
            return default

        try:
            value = float(raw)
        except ValueError:
            print("Invalid float.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return value


def _prompt_menu(prompt, options, default_idx=0):
    default_idx = max(0, min(int(default_idx), len(options) - 1))
    if not _is_tty():
        return default_idx

    while True:
        print(prompt)
        for idx, option in enumerate(options, start=1):
            print(f"{idx}) {option}")
        try:
            raw = input(f"Choose [1-{len(options)}] (default {default_idx + 1}): ").strip()
        except EOFError:
            raw = ""

        if not raw:
            return default_idx

        try:
            value = int(raw)
        except ValueError:
            print("Invalid choice.")
            continue

        if 1 <= value <= len(options):
            return value - 1
        print("Choice out of range.")


def _path_rel(path, base_dir):
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except Exception:
        return str(path)


def load_config(config_path):
    config = load_project_config(config_path=config_path)
    config = copy.deepcopy(config)
    config.setdefault("model", {})
    config["model"]["print_summary"] = False
    return config


def _model_arch_keys():
    return [
        "version",
        "filters",
        "num_residual_blocks",
        "dropout",
        "history_positions",
        "use_se_blocks",
        "use_se_bottleneck",
        "se_reduction",
        "drop_path_rate",
        "use_coord_conv",
        "use_layer_scale",
        "layer_scale_init",
        "policy_head_channels",
        "policy_head_conv_filters",
        "policy_head_conv_groups",
        "policy_head_global_dim",
        "policy_head_hidden_dim",
        "value_head_filters",
        "value_hidden_dim",
        "moves_left_hidden_dim",
    ]


def _apply_model_overrides(base_config, overrides):
    merged = copy.deepcopy(base_config)
    merged.setdefault("model", {})
    for key in _model_arch_keys():
        if key in overrides and overrides[key] is not None:
            merged["model"][key] = overrides[key]
    merged["model"]["print_summary"] = False
    return merged


def _infer_model_overrides_from_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return {}

    overrides = {}

    conv_w = state_dict.get("conv_block.0.conv.weight")
    if isinstance(conv_w, torch.Tensor):
        overrides["filters"] = int(conv_w.shape[0])
        input_plus_coords = int(conv_w.shape[1])
        input_planes = input_plus_coords - 2
        if input_planes > 0:
            if input_planes % 16 == 0:
                overrides["history_positions"] = int(input_planes // 16 - 1)
            overrides["use_coord_conv"] = True
    else:
        stem_w = state_dict.get("conv_block.0.weight")
        if isinstance(stem_w, torch.Tensor):
            overrides["filters"] = int(stem_w.shape[0])
            input_planes = int(stem_w.shape[1])
            if input_planes > 0 and input_planes % 16 == 0:
                overrides["history_positions"] = int(input_planes // 16 - 1)
            overrides["use_coord_conv"] = False

    block_ids = set()
    prefix = "residual_tower."
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        part = suffix.split(".", 1)[0]
        if part.isdigit():
            block_ids.add(int(part))
    if block_ids:
        overrides["num_residual_blocks"] = len(block_ids)

    policy_conv_w = state_dict.get("policy_conv.weight")
    if isinstance(policy_conv_w, torch.Tensor):
        policy_out = int(policy_conv_w.shape[0])
        policy_in_per_group = int(policy_conv_w.shape[1])
        overrides["policy_head_channels"] = policy_out
        overrides["policy_head_conv_filters"] = policy_out
        filters = overrides.get("filters")
        if isinstance(filters, int) and policy_in_per_group > 0 and filters % policy_in_per_group == 0:
            overrides["policy_head_conv_groups"] = int(filters // policy_in_per_group)

    policy_global_w = state_dict.get("policy_global_fc.weight")
    if isinstance(policy_global_w, torch.Tensor):
        overrides["policy_head_global_dim"] = int(policy_global_w.shape[0])

    policy_fc1_w = state_dict.get("policy_fc1.weight")
    if isinstance(policy_fc1_w, torch.Tensor):
        overrides["policy_head_hidden_dim"] = int(policy_fc1_w.shape[0])

    value_conv_w = state_dict.get("value_conv.weight")
    if isinstance(value_conv_w, torch.Tensor):
        overrides["value_head_filters"] = int(value_conv_w.shape[0])

    value_fc1_w = state_dict.get("value_fc1.weight")
    if isinstance(value_fc1_w, torch.Tensor):
        overrides["value_hidden_dim"] = int(value_fc1_w.shape[0])
    moves_left_fc1_w = state_dict.get("moves_left_fc1.weight")
    if isinstance(moves_left_fc1_w, torch.Tensor):
        overrides["moves_left_hidden_dim"] = int(moves_left_fc1_w.shape[0])

    se_fc1_keys = [key for key in state_dict.keys() if ".se.fc1.weight" in key]
    se_fc_keys = [key for key in state_dict.keys() if ".se.fc.weight" in key]
    use_se_blocks = bool(se_fc1_keys or se_fc_keys)
    overrides["use_se_blocks"] = use_se_blocks
    if use_se_blocks:
        overrides["use_se_bottleneck"] = bool(se_fc1_keys)
        if se_fc1_keys:
            fc1_weight = state_dict.get(se_fc1_keys[0])
            if isinstance(fc1_weight, torch.Tensor) and fc1_weight.ndim == 4:
                in_ch = int(fc1_weight.shape[1])
                mid = int(fc1_weight.shape[0])
                if mid > 0 and in_ch > 0:
                    overrides["se_reduction"] = max(1, in_ch // mid)
    overrides["use_layer_scale"] = any(".layer_scale.gamma" in key for key in state_dict.keys())

    return overrides


def _extract_arch_overrides(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}

    direct = checkpoint.get("model_architecture")
    if isinstance(direct, dict):
        overrides = {k: direct.get(k) for k in _model_arch_keys() if k in direct}
        if "use_se_blocks" not in overrides and "use_se2d_blocks" in direct:
            overrides["use_se_blocks"] = direct.get("use_se2d_blocks")
        return overrides

    legacy = {}
    for key in _model_arch_keys():
        if key in checkpoint and checkpoint[key] is not None:
            legacy[key] = checkpoint[key]
    if "use_se_blocks" not in legacy and checkpoint.get("use_se2d_blocks") is not None:
        legacy["use_se_blocks"] = checkpoint.get("use_se2d_blocks")
    return legacy


def collect_model_candidates(chess_dir, config, max_candidates=200):
    candidates = []

    best_rel = config.get("paths", {}).get("best_model_il", "models/best_model_il.pt")
    architecture_id = str(config.get("model", {}).get("architecture_id", "se_cnn_v9"))
    configured_best_path = (chess_dir / best_rel).resolve()
    best_path = model_alias_file(configured_best_path, architecture_id, "il")
    best_swa_path = model_alias_file(
        configured_best_path.parent / "best_model_il_swa.pt",
        architecture_id,
        "il_swa",
    ).resolve()
    legacy_best_path = legacy_architecture_scoped_file(
        configured_best_path, architecture_id
    ).resolve()
    legacy_best_swa_path = legacy_architecture_scoped_file(
        configured_best_path.parent / "best_model_il_swa.pt",
        architecture_id,
    ).resolve()
    models_rel = config.get("paths", {}).get("models_dir", "models")
    models_dir = (chess_dir / models_rel).resolve()

    for path in (best_path, best_swa_path, legacy_best_path, legacy_best_swa_path):
        if path.exists() and path not in candidates:
            candidates.append(path)

    all_pts = []
    if models_dir.exists():
        all_pts = sorted(models_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in all_pts[:max_candidates]:
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    return candidates, best_path, models_dir


def build_model_catalog(model_paths, base_dir):
    entries = [load_checkpoint_metadata(path, base_dir) for path in model_paths]
    return sort_entries_by_folder_and_elo(entries)


def print_model_catalog(catalog):
    print_model_table(
        catalog,
        title="Available Checkpoints",
        show_folder=True,
        show_version=True,
        show_modified=True,
        show_swa=True,
        show_opt=True,
        group_by_folder=True,
    )


def _parse_id_selection(raw, max_id):
    selected = set()
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2:
                continue
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 1 <= idx <= max_id:
                    selected.add(idx)
        else:
            try:
                idx = int(token)
            except ValueError:
                continue
            if 1 <= idx <= max_id:
                selected.add(idx)

    return sorted(selected)


def choose_model_entries(catalog, best_path):
    valid = [entry for entry in catalog if not entry.get("error")]
    if not valid:
        return []

    best_valid_idx = 0
    for idx, entry in enumerate(valid):
        if entry["path"].resolve() == best_path.resolve():
            best_valid_idx = idx
            break

    if not _is_tty():
        return [valid[best_valid_idx]]

    _print_block("Model Selection")
    mode = _prompt_menu(
        "Select model scope:",
        [
            "Best model only",
            "Choose model IDs",
            "All listed models",
        ],
        default_idx=0,
    )

    if mode == 0:
        return [valid[best_valid_idx]]

    if mode == 2:
        return valid

    # Choose by IDs from full catalog view
    while True:
        try:
            raw = input(f"Enter IDs (e.g. 1,3,5-7) from 1 to {len(catalog)}: ").strip()
        except EOFError:
            raw = ""

        ids = _parse_id_selection(raw, len(catalog))
        if not ids:
            print("No valid IDs selected.")
            continue

        chosen = []
        for idx in ids:
            entry = catalog[idx - 1]
            if entry.get("error"):
                print(f"Skipping invalid checkpoint ID {idx}: {entry['path_rel']}")
                continue
            chosen.append(entry)

        if chosen:
            return chosen
        print("All selected IDs were invalid. Try again.")


def choose_eval_settings(elo_cfg, inference_cfg=None):
    default_games = int(elo_cfg.get("adaptive_focus_games_per_level", 20))
    default_use_mcts = False
    default_sims = int(elo_cfg.get("mcts_eval_simulations", 100))
    inference_cfg = dict(inference_cfg or {})
    default_budget_mode = str(inference_cfg.get("budget_mode", "simulations")).strip().lower()
    if default_budget_mode not in {"simulations", "time"}:
        default_budget_mode = "simulations"
    default_move_time = float(inference_cfg.get("move_time_seconds", 0.2))
    default_sf_time = float(elo_cfg.get("stockfish_time_limit", 0.05))
    default_max_moves = int(elo_cfg.get("max_moves", 150))
    default_sf_path = str(elo_cfg.get("stockfish_path", "stockfish"))
    default_workers = int(elo_cfg.get("workers", 0))

    if _is_tty():
        _print_block("Evaluation Settings")
        print("Stockfish levels: detected automatically from UCI_Elo")
        print(f"Adaptive focus:  {default_games} initial games/selected level; precision adds more")
    levels = None
    games_per_level = None
    workers = default_workers

    if _is_tty():
        mcts_default_idx = 1 if default_use_mcts else 0
        mcts_mode = _prompt_menu(
            "Inference mode:",
            [
                "Raw NN only",
                "MCTS only (choose budget)",
                "Raw NN + MCTS (choose budget)",
            ],
            default_idx=mcts_default_idx,
        )
        eval_modes = ["nn"] if mcts_mode == 0 else ["mcts"] if mcts_mode == 1 else ["nn", "mcts"]
    else:
        eval_modes = ["mcts"] if default_use_mcts else ["nn"]

    simulations = default_sims
    budget_mode = default_budget_mode
    move_time_seconds = default_move_time
    if "mcts" in eval_modes:
        if _is_tty():
            budget_choice = _prompt_menu(
                "MCTS move budget:",
                ["Fixed simulations", "Wall-clock time"],
                default_idx=1 if default_budget_mode == "time" else 0,
            )
            budget_mode = "time" if budget_choice == 1 else "simulations"
        if budget_mode == "time":
            move_time_seconds = _prompt_float(
                "MCTS time per move (seconds)", default_move_time, min_value=0.001
            )
        else:
            simulations = _prompt_int("MCTS simulations per move", default_sims, min_value=1)

    sf_time = _prompt_float("Stockfish time per move (seconds)", default_sf_time, min_value=0.0)
    max_moves = _prompt_int("Max moves per game", default_max_moves, min_value=1)
    sf_path = default_sf_path

    return {
        "levels": levels,
        "games_per_level": games_per_level,
        "workers": workers,
        "use_mcts": "mcts" in eval_modes and len(eval_modes) == 1,
        "eval_modes": eval_modes,
        "simulations": simulations,
        "budget_mode": budget_mode,
        "move_time_seconds": move_time_seconds,
        "sf_time": sf_time,
        "max_moves": max_moves,
        "sf_path": sf_path,
    }


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        print(f"  x Not found: {checkpoint_path}")
        return None

    try:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
        source_state = checkpoint.get("model_state_dict")
        if not isinstance(source_state, dict):
            print(f"  x Invalid checkpoint (missing model_state_dict): {checkpoint_path.name}")
            return None
    except Exception as exc:
        print(f"  x Failed to load {checkpoint_path.name}: {exc}")
        return None

    load_mode = "strict_current"
    model = None

    # 1) Canonical path: the checkpoint selects its own architecture/data contract.
    try:
        spec = resolve_model_spec(config, checkpoint)
        model = create_model(config, model_spec=spec).to(device)
        model = model.to(memory_format=torch.channels_last)
        normalized = normalize_state_dict_keys(
            source_state, target_keys=set(model.state_dict())
        )
        model.load_state_dict(normalized)
        load_mode = f"strict_{spec['architecture_id']}"
    except Exception:
        model = None

    # 2) Try architecture saved in checkpoint (or inferred from state_dict)
    if model is None:
        arch_overrides = _extract_arch_overrides(checkpoint)
        inferred_overrides = _infer_model_overrides_from_state_dict(source_state)
        combined_overrides = dict(inferred_overrides)
        combined_overrides.update({k: v for k, v in arch_overrides.items() if v is not None})

        if combined_overrides:
            try:
                cfg_arch = _apply_model_overrides(config, combined_overrides)
                model = create_model(cfg_arch).to(device)
                model = model.to(memory_format=torch.channels_last)
                try:
                    model.load_state_dict(source_state)
                    load_mode = "strict_checkpoint_arch"
                except Exception:
                    # Legacy deltas (e.g. old CoordConv bias=True) - load matching tensors.
                    report = transfer_matching_weights(model, checkpoint)
                    load_mode = f"transfer_checkpoint_arch ({report.get('match_ratio', 0.0):.1%})"
            except Exception:
                model = None

    # 3) Last resort: transfer into current architecture
    if model is None:
        try:
            model = create_model(config).to(device)
            model = model.to(memory_format=torch.channels_last)
            report = transfer_matching_weights(model, checkpoint)
            load_mode = f"transfer_current ({report.get('match_ratio', 0.0):.1%})"
        except Exception as exc:
            print(f"  x Failed to load {checkpoint_path.name}: {exc}")
            return None

    info_parts = []
    epoch = _safe_int(checkpoint.get("epoch"))
    if epoch is not None:
        info_parts.append(f"epoch {epoch + 1}")

    val_loss = _safe_float(checkpoint.get("val_loss", checkpoint.get("loss")))
    if val_loss is not None:
        info_parts.append(f"val_loss={val_loss:.4f}")

    policy_loss = _safe_float(checkpoint.get("val_policy_loss", checkpoint.get("policy_loss")))
    if policy_loss is not None:
        info_parts.append(f"policy_loss={policy_loss:.4f}")

    top1 = _safe_float(checkpoint.get("val_policy_top1", checkpoint.get("policy_top1_acc")))
    if top1 is not None:
        info_parts.append(f"top1={top1:.2%}")

    elo_nn = _safe_float(checkpoint.get("estimated_elo_nn", checkpoint.get("last_estimated_elo_nn")))
    elo_mcts = _safe_float(checkpoint.get("estimated_elo_mcts", checkpoint.get("last_estimated_elo_mcts")))
    if elo_nn is not None:
        info_parts.append(f"nn_elo={int(round(elo_nn))}")
    if elo_mcts is not None:
        sims = _safe_int(checkpoint.get("estimated_elo_mcts_simulations"))
        suffix = f"@{sims}" if sims is not None else ""
        info_parts.append(f"mcts_elo={int(round(elo_mcts))}{suffix}")

    version = checkpoint.get("version")
    if version is not None:
        info_parts.append(str(version))

    info_parts.append(f"mode={load_mode}")
    info = ", ".join(info_parts) if info_parts else "no metadata"
    print(f"  + Loaded: {checkpoint_path.name} ({info})")

    model.eval()
    return model


def format_results_table(all_results: list[dict]) -> str:
    """Format results as a comparison table."""
    if not all_results:
        return "No results."

    all_levels = set()
    for result in all_results:
        all_levels.update(result.get("results", {}).keys())
    levels = sorted(all_levels)

    lines = []
    lines.append("")
    lines.append("=" * 96)
    lines.append("ELO ESTIMATION RESULTS")
    lines.append("=" * 96)
    lines.append("")

    for result in all_results:
        name = result.get("model_name", "unknown")
        mode_label = result.get("mode_label", result.get("mode", ""))
        elo = result.get("estimated_elo")
        elo_str = str(elo) if elo is not None else "N/A"
        lines.append(f"- {name} [{mode_label}]")
        lines.append(f"  Estimated Elo: {elo_str}")
        timed_summary = _timed_mcts_work_summary(result)
        if timed_summary:
            lines.append(f"  {timed_summary}")
        rating_bracket = result.get("rating_bracket")
        if result.get("rating_bracketed") and isinstance(rating_bracket, list):
            lines.append(
                f"  Rating bracket: SF {rating_bracket[0]}-{rating_bracket[1]} "
                "(crosses 50% score)"
            )
        elif result.get("rating_censored"):
            bound = result.get("elo_lower_bound", result.get("elo_upper_bound"))
            relation = ">" if result.get("elo_lower_bound") is not None else "<"
            lines.append(f"  Rating is range-censored: {relation}{bound}; no 50% Stockfish bracket.")
        if result.get("elo_std_error") is not None:
            ci = result.get("elo_ci95")
            ci_str = f", 95% CI {ci[0]}-{ci[1]}" if isinstance(ci, list) and len(ci) == 2 else ""
            lines.append(f"  Uncertainty: +/-{result['elo_std_error']} Elo SE{ci_str} (adaptive ladder)")
        local_level = result.get("local_rating_level")
        local_games = int(result.get("local_rating_games", 0) or 0)
        if local_level is not None:
            lines.append(
                f"  Calibration point: SF {local_level}, {local_games} games nearest 50% score."
            )
        rating_games = int(result.get("rating_games", result.get("total_games", 0)) or 0)
        probe_only_games = int(result.get("probe_only_games", 0) or 0)
        rating_levels = list(result.get("rating_levels") or [])
        if rating_levels and probe_only_games > 0:
            lines.append(
                f"  Rating fit: {rating_games} focused games on levels {rating_levels}; "
                f"{probe_only_games} probe games used only for level selection."
            )
        if result.get("calibration_warning"):
            curve_elo = result.get("elo_curve_estimate")
            curve_text = f"; fixed-slope curve would report {curve_elo}" if curve_elo is not None else ""
            lines.append(
                "  Calibration note: Stockfish level scores do not follow the textbook Elo slope "
                f"at this time control (dispersion {float(result.get('elo_overdispersion', 1.0)):.2f}{curve_text})."
            )

        if result.get("results"):
            parts = []
            for lvl in levels:
                if lvl in result["results"]:
                    r = result["results"][lvl]
                    games = int(r.get("games", r.get("wins", 0) + r.get("draws", 0) + r.get("losses", 0)) or 0)
                    local_elo = r.get("local_performance_elo")
                    local_text = f", local~{local_elo}" if local_elo is not None and games >= 8 else ""
                    parts.append(
                        f"vs {lvl}: W{r['wins']}/D{r['draws']}/L{r['losses']} "
                        f"({r['score']:.0%}, n={games}{local_text})"
                    )
            lines.append("  " + " | ".join(parts))

        elapsed = result.get("total_time", 0.0)
        games = result.get("total_games", 0)
        lines.append(f"  Time: {elapsed:.1f}s ({games} games)")
        lines.append("")

    if len(all_results) > 1:
        lines.append("-" * 96)
        lines.append("COMPARISON (sorted by Elo)")
        lines.append("-" * 96)

        sorted_results = sorted(
            all_results,
            key=lambda r: r.get("estimated_elo") or -9999,
            reverse=True,
        )

        lines.append(f"{'#':<4} {'Model':<48} {'Mode':<14} {'Elo':>8} {'AvgScore':>10}")
        for idx, result in enumerate(sorted_results, start=1):
            name = result.get("model_name", "unknown")[:48]
            mode_label = str(result.get("mode_label", result.get("mode", "")))[:14]
            elo = result.get("estimated_elo")
            elo_str = str(elo) if elo is not None else "N/A"
            model_results = result.get("results", {})
            if model_results:
                avg_score = sum(v["score"] for v in model_results.values()) / len(model_results)
            else:
                avg_score = 0.0
            lines.append(f"{idx:<4} {name:<48} {mode_label:<14} {elo_str:>8} {avg_score:>10.0%}")

        lines.append("")

    lines.append("=" * 96)
    return "\n".join(lines)


def _timed_mcts_work_summary(result: dict) -> str | None:
    if result.get("mode") != "mcts" or result.get("mcts_budget_mode") != "time":
        return None
    seconds = _safe_float(result.get("mcts_move_time_seconds"))
    workers = _safe_int(result.get("mcts_time_search_workers"))
    calls = _safe_int(result.get("mcts_search_calls"))
    fresh = _safe_int(result.get("mcts_fresh_simulations"))
    average = _safe_float(result.get("mcts_avg_simulations_per_move"))
    if average is None and calls and fresh is not None:
        average = float(fresh) / max(1, calls)
    budget = f"{seconds:g}s/move" if seconds is not None else "timed/move"
    if workers is None:
        worker_text = "parallelism unavailable"
    else:
        worker_text = f"{workers} game worker" if workers == 1 else f"{workers} game workers"
    if average is None:
        work_text = "actual simulations/move unavailable"
    else:
        detail = f" ({fresh} fresh / {calls} search calls)" if fresh is not None and calls is not None else ""
        work_text = f"{average:.1f} average fresh simulations/move{detail}"
    style = result.get("mcts_time_search_style")
    style_text = f", path={style}" if style else ""
    return f"Timed search: {budget}, {worker_text}{style_text}; {work_text}."


def main():
    parser = argparse.ArgumentParser(description="Estimate NN/MCTS Elo against Stockfish")
    parser.add_argument("--model", action="append", default=[], help="checkpoint path to preselect")
    args = parser.parse_args()

    chess_dir = script_dir.parent
    config_path = default_config_path()
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    cfg_device = str(config.get("hardware", {}).get("device", "cuda")).strip().lower()
    if cfg_device not in {"cpu", "cuda"}:
        cfg_device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg_device == "cuda" and not torch.cuda.is_available():
        print("Config device is CUDA but CUDA is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg_device)
    print(f"Using device from config: {device}")

    elo_cfg = dict(config.get("elo_estimation", {}) or {})
    settings = choose_eval_settings(elo_cfg, config.get("inference", {}))
    levels = settings["levels"]
    games_per_level = settings["games_per_level"]
    workers = settings["workers"]
    eval_modes = list(settings.get("eval_modes") or (["mcts"] if settings.get("use_mcts") else ["nn"]))
    simulations = settings["simulations"]
    budget_mode = settings["budget_mode"]
    move_time_seconds = settings["move_time_seconds"]
    sf_time = settings["sf_time"]
    max_moves = settings["max_moves"]
    sf_path = settings["sf_path"]

    model_paths, best_path, models_dir = collect_model_candidates(chess_dir, config)
    if not model_paths:
        print(f"No model checkpoints found in: {models_dir}")
        sys.exit(1)

    catalog = build_model_catalog(model_paths, models_dir)
    print_model_catalog(catalog)

    if args.model:
        requested = []
        for raw_path in args.model:
            candidate = Path(raw_path).expanduser()
            candidate = (
                (chess_dir / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
            requested.append(candidate)
        catalog_by_path = {Path(entry["path"]).resolve(): entry for entry in catalog}
        missing = [path for path in requested if path not in catalog_by_path]
        if missing:
            print("Unknown checkpoint(s):")
            for path in missing:
                print(f"  {path}")
            sys.exit(2)
        selected_entries = [catalog_by_path[path] for path in requested]
    else:
        selected_entries = choose_model_entries(catalog, best_path)
    if not selected_entries:
        print("No valid models selected.")
        sys.exit(1)
    print_selected_models_table(selected_entries, title="Selected Models For Elo Evaluation")

    selected_paths = [entry["path"] for entry in selected_entries]

    def _resolved_mode_config(mode_name):
        mode_use_mcts = mode_name == "mcts"
        runtime_cfg = build_eval_elo_config(
            elo_cfg,
            levels=levels,
            games_per_level=games_per_level,
            use_mcts=mode_use_mcts,
            simulations=simulations,
            budget_mode=budget_mode,
            time_limit_seconds=move_time_seconds,
            sf_time=sf_time,
            max_moves=max_moves,
            sf_path=sf_path,
            workers=resolve_eval_workers(workers, mode_use_mcts, elo_cfg),
            standalone=True,
        )
        if mode_use_mcts and budget_mode == "time":
            central = bool(
                device.type == "cuda"
                and int(runtime_cfg.get("workers", 1) or 1) > 1
                and runtime_cfg.get("eval_elo_central_inference_enabled", False)
            )
            batched_roots = bool(runtime_cfg.get("batch_model_moves", True))
            runtime_cfg["mcts_time_search_central_inference"] = central
            runtime_cfg["mcts_time_search_batch_model_moves"] = batched_roots
            runtime_cfg["mcts_time_search_style"] = (
                "central_inference" if central else "batched_roots" if batched_roots else "independent_roots"
            )
        return runtime_cfg

    mode_runtime_configs = {
        mode_name: _resolved_mode_config(mode_name)
        for mode_name in eval_modes
    }
    total_games = len(selected_paths) * sum(
        int(
            runtime_cfg.get(
                "adaptive_hard_max_total_games",
                runtime_cfg.get("adaptive_max_total_games", 0),
            )
            or 0
        )
        for runtime_cfg in mode_runtime_configs.values()
    )
    mode_labels = []
    if "nn" in eval_modes:
        mode_labels.append("Raw NN")
    if "mcts" in eval_modes:
        mode_labels.append(
            f"MCTS ({move_time_seconds:g}s/move)"
            if budget_mode == "time"
            else f"MCTS ({simulations} sims)"
        )
    mode_str = " + ".join(mode_labels)
    if workers > 0:
        workers_str = f"{workers} workers" if workers > 1 else "sequential"
    else:
        workers_str = "auto"
    if budget_mode == "time" and "mcts" in eval_modes:
        timed_workers = int(mode_runtime_configs["mcts"].get("workers", 1) or 1)
        timed_worker_text = f"timed MCTS {timed_workers} game workers"
        if "nn" in eval_modes:
            nn_workers = int(mode_runtime_configs["nn"].get("workers", 1) or 1)
            workers_str = f"Raw NN {nn_workers} workers; {timed_worker_text}"
        else:
            workers_str = timed_worker_text

    _print_block("Elo Estimation Plan")
    print(f"Models:         {len(selected_paths)}")
    print("Levels:         auto from Stockfish UCI_Elo")
    print(f"Total games:    <= {total_games} hard safety cap (usually stops earlier)")
    print(f"Mode:           {mode_str}")
    if "mcts" in eval_modes:
        if budget_mode == "time":
            print(f"MCTS time:      {move_time_seconds:g}s/move")
        else:
            print(f"MCTS sims:      {simulations}")
    for mode_name in eval_modes:
        runtime_cfg = mode_runtime_configs[mode_name]
        label = "MCTS adaptive" if mode_name == "mcts" else "NN adaptive"
        print(
            f"{label + ':':<16} probe {int(runtime_cfg.get('adaptive_probe_games_per_level', 0) or 0)}/level"
            f" · focus >= {int(runtime_cfg.get('adaptive_focus_games_per_level', 0) or 0)}"
            f" · target SE {float(runtime_cfg.get('adaptive_target_standard_error', 0.0) or 0.0):.0f}"
            f" · budget {int(runtime_cfg.get('adaptive_max_total_games', 0) or 0)}"
            f"/{int(runtime_cfg.get('adaptive_hard_max_total_games', runtime_cfg.get('adaptive_max_total_games', 0)) or 0)}"
        )
    print(f"Workers:        {workers_str}")
    if workers <= 0 and "nn" in eval_modes:
        raw_workers = resolve_eval_workers(workers, False, elo_cfg)
        if raw_workers > 0:
            print(f"Raw NN workers: {raw_workers}")
    if "mcts" in eval_modes and budget_mode == "time":
        timed_cfg = mode_runtime_configs["mcts"]
        mcts_workers = int(timed_cfg.get("workers", 1) or 1)
        requested_timed_workers = int(
            timed_cfg.get("mcts_timed_parallel_games_requested_workers", mcts_workers) or mcts_workers
        )
        cap = int(elo_cfg.get("mcts_timed_parallel_games", 4) or 4)
        clamp_note = (
            f", requested {requested_timed_workers} capped at {mcts_workers}"
            if requested_timed_workers > mcts_workers
            else ""
        )
        print(f"MCTS workers:   {mcts_workers} (timed cap {cap}{clamp_note})")
    elif workers <= 0 and "mcts" in eval_modes:
        mcts_workers = int(mode_runtime_configs["mcts"].get("workers", 0) or 0)
        if mcts_workers > 0:
            print(f"MCTS workers:   {mcts_workers}")
        else:
            print("MCTS workers:   auto -> full CPU budget")
    if "mcts" in eval_modes:
        central_enabled = bool(
            mode_runtime_configs["mcts"].get(
                "mcts_time_search_central_inference",
                elo_cfg.get("eval_elo_central_inference_enabled", False),
            )
        )
        if central_enabled:
            servers = elo_cfg.get("eval_elo_central_inference_servers", "auto")
            print(f"MCTS central:   enabled (servers={servers})")
        else:
            print("MCTS central:   disabled")
        if budget_mode == "time":
            print(
                "Timed path:     "
                f"{mode_runtime_configs['mcts'].get('mcts_time_search_style', 'unknown')}"
            )
    print(f"SF time/move:   {sf_time}s")
    print(f"Max moves:      {max_moves}")
    if bool(elo_cfg.get("paired_openings_enabled", True)):
        print(
            "Openings:       paired colors · "
            f"{int(elo_cfg.get('paired_openings_max_plies', 8) or 0)} fixed plies"
        )
    else:
        print("Openings:       start position only")

    # Ensure Stockfish is available
    sf_path = ensure_stockfish(sf_path)

    all_results = []
    t0 = time.perf_counter()

    for idx, entry in enumerate(selected_entries, start=1):
        model_path = entry["path"]
        print(f"\n[{idx}/{len(selected_paths)}] {model_path.name}")

        model = load_model(model_path, config, device)
        if model is None:
            continue

        for eval_mode in eval_modes:
            mode_use_mcts = eval_mode == "mcts"
            mode_label_text = mode_label(
                mode_use_mcts,
                simulations,
                budget_mode=budget_mode,
                time_limit_seconds=move_time_seconds,
            )
            print(f"  Mode: {mode_label_text}")

            mode_elo_cfg = dict(mode_runtime_configs[eval_mode])
            initial_elo = apply_checkpoint_elo_seed(
                mode_elo_cfg,
                model_path,
                use_mcts=mode_use_mcts,
                simulations=simulations,
                budget_mode=budget_mode,
                time_limit_seconds=move_time_seconds,
            )
            if initial_elo is not None:
                seed_source = mode_elo_cfg.get(
                    "adaptive_initial_elo_source",
                    "explicit adaptive_initial_elo",
                )
                print(f"  Scheduling seed: {initial_elo:.0f} Elo ({seed_source})")

            result = run_elo_check(model, config, device, mode_elo_cfg)
            if result.get("cancelled"):
                print("  Elo estimation cancelled. Exiting now and releasing worker resources.", flush=True)
                if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()
                with contextlib.suppress(Exception):
                    sys.stdout.flush()
                    sys.stderr.flush()
                os._exit(130)

            result["model_name"] = model_path.name
            result["model_path"] = str(model_path)
            result["mode"] = "mcts" if mode_use_mcts else "nn"
            result["mode_label"] = mode_label_text
            result["mcts_simulations"] = int(simulations) if mode_use_mcts and budget_mode == "simulations" else 0
            result["mcts_budget_mode"] = budget_mode if mode_use_mcts else "raw_nn"
            result["mcts_move_time_seconds"] = move_time_seconds if mode_use_mcts else None
            if mode_use_mcts and budget_mode == "time":
                for result_key, config_key in (
                    ("mcts_time_search_protocol", "mcts_time_search_protocol"),
                    ("mcts_time_search_workers", "workers"),
                    ("mcts_time_search_batch_model_moves", "mcts_time_search_batch_model_moves"),
                    ("mcts_time_search_central_inference", "mcts_time_search_central_inference"),
                    ("mcts_time_search_style", "mcts_time_search_style"),
                    ("mcts_timed_parallel_games", "mcts_timed_parallel_games"),
                ):
                    if result.get(result_key) is None:
                        result[result_key] = mode_elo_cfg.get(config_key)
                protocol_runtime_cfg = dict(mode_elo_cfg)
                for result_key, protocol_key in (
                    ("mcts_time_search_workers", "workers"),
                    ("mcts_time_search_batch_model_moves", "mcts_time_search_batch_model_moves"),
                    ("mcts_time_search_central_inference", "mcts_time_search_central_inference"),
                    ("mcts_time_search_style", "mcts_time_search_style"),
                ):
                    if result.get(result_key) is not None:
                        protocol_runtime_cfg[protocol_key] = result[result_key]
            else:
                protocol_runtime_cfg = mode_elo_cfg
            result["elo_protocol_settings"] = elo_protocol_settings(
                protocol_runtime_cfg,
                use_mcts=mode_use_mcts,
                simulations=simulations,
                budget_mode=budget_mode,
                time_limit_seconds=move_time_seconds,
            )
            all_results.append(result)

            elo = result.get("estimated_elo")
            print(f"  Estimated {mode_label_text} Elo: {elo if elo is not None else 'N/A'}")
            timed_summary = _timed_mcts_work_summary(result)
            if timed_summary:
                print(f"  {timed_summary}")

            persist_estimated_elo(
                model_path,
                elo,
                levels=list(result.get("levels_requested", []) or []),
                games_per_level=int(result.get("games_per_level_requested", 0) or 0),
                use_mcts=mode_use_mcts,
                simulations=simulations,
                budget_mode=budget_mode,
                time_limit_seconds=move_time_seconds,
                sf_time=sf_time,
                source="eval_elo_manual",
                elo_result=result,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t0

    print(format_results_table(all_results))
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
