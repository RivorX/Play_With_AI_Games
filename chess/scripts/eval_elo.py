"""Interactive Elo estimation tool for chess models.

This script intentionally does not accept CLI arguments.
Run it without parameters and use the menu.
"""

import contextlib
import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, transfer_matching_weights
from src.utils.config import normalize_config
from utils.shared.elo_estimator import ensure_stockfish
from utils.shared.elo_runner import (
    build_eval_elo_config,
    mode_label,
    persist_estimated_elo,
    resolve_eval_workers,
    run_elo_check,
    safe_float as _safe_float,
    safe_int as _safe_int,
)
from utils.shared.model_catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_elo,
)
from utils.shared.model_view import print_selected_models_table


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
    with open(config_path, "r", encoding="utf-8") as f:
        config = normalize_config(yaml.safe_load(f))
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
    best_path = (chess_dir / best_rel).resolve()
    best_swa_path = (best_path.parent / "best_model_il_swa.pt").resolve()
    models_rel = config.get("paths", {}).get("models_dir", "models")
    models_dir = (chess_dir / models_rel).resolve()

    for path in (best_path, best_swa_path):
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


def choose_eval_settings(elo_cfg):
    levels = list(elo_cfg.get("levels", [1320, 1500, 1700, 1900, 2200]))
    default_games = int(elo_cfg.get("adaptive_focus_games_per_level", 20))
    default_use_mcts = False
    default_sims = int(elo_cfg.get("mcts_eval_simulations", 100))
    default_sf_time = float(elo_cfg.get("stockfish_time_limit", 0.05))
    default_max_moves = int(elo_cfg.get("max_moves", 150))
    default_sf_path = str(elo_cfg.get("stockfish_path", "stockfish"))
    default_workers = int(elo_cfg.get("workers", 0))

    if _is_tty():
        _print_block("Evaluation Settings")
        print(f"Levels (from config): {levels}")
    games_label = "Focus cap per selected level"
    games_per_level = _prompt_int(games_label, default_games, min_value=1)
    workers = default_workers

    if _is_tty():
        mcts_default_idx = 1 if default_use_mcts else 0
        mcts_mode = _prompt_menu(
            "Inference mode:",
            [
                "Raw NN only",
                "MCTS only (choose simulations)",
                "Raw NN + MCTS (choose simulations)",
            ],
            default_idx=mcts_default_idx,
        )
        eval_modes = ["nn"] if mcts_mode == 0 else ["mcts"] if mcts_mode == 1 else ["nn", "mcts"]
    else:
        eval_modes = ["mcts"] if default_use_mcts else ["nn"]

    simulations = default_sims
    if "mcts" in eval_modes:
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        source_state = checkpoint.get("model_state_dict")
        if not isinstance(source_state, dict):
            print(f"  x Invalid checkpoint (missing model_state_dict): {checkpoint_path.name}")
            return None
    except Exception as exc:
        print(f"  x Failed to load {checkpoint_path.name}: {exc}")
        return None

    load_mode = "strict_current"
    model = None

    # 1) Fast path: strict load with current config
    try:
        model = ChessNet(config).to(device)
        model = model.to(memory_format=torch.channels_last)
        model.load_state_dict(source_state)
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
                model = ChessNet(cfg_arch).to(device)
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
            model = ChessNet(config).to(device)
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
        if result.get("elo_std_error") is not None:
            ci = result.get("elo_ci95")
            ci_str = f", 95% CI {ci[0]}-{ci[1]}" if isinstance(ci, list) and len(ci) == 2 else ""
            lines.append(f"  Uncertainty: +/-{result['elo_std_error']} Elo SE{ci_str} (adaptive ladder)")
        rating_games = int(result.get("rating_games", result.get("total_games", 0)) or 0)
        probe_only_games = int(result.get("probe_only_games", 0) or 0)
        rating_levels = list(result.get("rating_levels") or [])
        if rating_levels and probe_only_games > 0:
            lines.append(
                f"  Rating fit: {rating_games} focused games on levels {rating_levels}; "
                f"{probe_only_games} probe games used only for level selection."
            )
        if result.get("fit_warning"):
            model_se = result.get("elo_model_std_error")
            model_se_text = f"; ideal-curve SE would be {model_se}" if model_se is not None else ""
            lines.append(
                "  Fit warning: non-monotonic level results; uncertainty was inflated "
                f"(dispersion {float(result.get('elo_overdispersion', 1.0)):.2f}{model_se_text})."
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


def main():
    if len(sys.argv) > 1:
        print("CLI arguments are disabled in this script.")
        print("Run without arguments and use the interactive menu:")
        print("  python chess/scripts/eval_elo.py")
        sys.exit(2)

    chess_dir = script_dir.parent
    config_path = chess_dir / "config" / "config.yaml"
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

    elo_cfg = config.get("elo_estimation", {})
    settings = choose_eval_settings(elo_cfg)
    levels = settings["levels"]
    games_per_level = settings["games_per_level"]
    workers = settings["workers"]
    eval_modes = list(settings.get("eval_modes") or (["mcts"] if settings.get("use_mcts") else ["nn"]))
    simulations = settings["simulations"]
    sf_time = settings["sf_time"]
    max_moves = settings["max_moves"]
    sf_path = settings["sf_path"]

    model_paths, best_path, models_dir = collect_model_candidates(chess_dir, config)
    if not model_paths:
        print(f"No model checkpoints found in: {models_dir}")
        sys.exit(1)

    catalog = build_model_catalog(model_paths, models_dir)
    print_model_catalog(catalog)

    selected_entries = choose_model_entries(catalog, best_path)
    if not selected_entries:
        print("No valid models selected.")
        sys.exit(1)
    print_selected_models_table(selected_entries, title="Selected Models For Elo Evaluation")

    selected_paths = [entry["path"] for entry in selected_entries]

    def _resolved_mode_config(mode_name):
        mode_use_mcts = mode_name == "mcts"
        return build_eval_elo_config(
            elo_cfg,
            levels=levels,
            games_per_level=games_per_level,
            use_mcts=mode_use_mcts,
            simulations=simulations,
            sf_time=sf_time,
            max_moves=max_moves,
            sf_path=sf_path,
            workers=resolve_eval_workers(workers, mode_use_mcts, elo_cfg),
            standalone=True,
        )

    mode_runtime_configs = {
        mode_name: _resolved_mode_config(mode_name)
        for mode_name in eval_modes
    }
    total_games = len(selected_paths) * sum(
        int(runtime_cfg.get("adaptive_max_total_games", 0) or 0)
        for runtime_cfg in mode_runtime_configs.values()
    )
    mode_labels = []
    if "nn" in eval_modes:
        mode_labels.append("Raw NN")
    if "mcts" in eval_modes:
        mode_labels.append(f"MCTS ({simulations} sims)")
    mode_str = " + ".join(mode_labels)
    if workers > 0:
        workers_str = f"{workers} workers" if workers > 1 else "sequential"
    else:
        workers_str = "auto"

    _print_block("Elo Estimation Plan")
    print(f"Models:         {len(selected_paths)}")
    print(f"Levels:         {levels}")
    print(f"Total games:    <= {total_games}")
    print(f"Mode:           {mode_str}")
    if "mcts" in eval_modes:
        print(f"MCTS sims:      {simulations}")
    for mode_name in eval_modes:
        runtime_cfg = mode_runtime_configs[mode_name]
        label = "MCTS adaptive" if mode_name == "mcts" else "NN adaptive"
        print(
            f"{label + ':':<16} probe {int(runtime_cfg.get('adaptive_probe_games_per_level', 0) or 0)}/level"
            f" · focus <= {int(runtime_cfg.get('adaptive_focus_games_per_level', 0) or 0)}"
            f" · target SE {float(runtime_cfg.get('adaptive_target_standard_error', 0.0) or 0.0):.0f}"
            f" · cap {int(runtime_cfg.get('adaptive_max_total_games', 0) or 0)}"
        )
    print(f"Workers:        {workers_str}")
    if workers <= 0 and "nn" in eval_modes:
        raw_workers = resolve_eval_workers(workers, False, elo_cfg)
        if raw_workers > 0:
            print(f"Raw NN workers: {raw_workers}")
    if workers <= 0 and "mcts" in eval_modes:
        mcts_workers = resolve_eval_workers(workers, True, elo_cfg)
        if mcts_workers > 0:
            print(f"MCTS workers:   {mcts_workers}")
        else:
            print("MCTS workers:   auto -> full CPU budget")
    if "mcts" in eval_modes:
        central_enabled = bool(elo_cfg.get("eval_elo_central_inference_enabled", False))
        if central_enabled:
            servers = elo_cfg.get("eval_elo_central_inference_servers", "auto")
            print(f"MCTS central:   enabled (servers={servers})")
        else:
            print("MCTS central:   disabled")
    print(f"SF time/move:   {sf_time}s")
    print(f"Max moves:      {max_moves}")
    if bool(elo_cfg.get("paired_openings_enabled", True)):
        print(
            "Openings:       paired colors · "
            f"{int(elo_cfg.get('paired_openings_max_plies', 6) or 0)} fixed plies"
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
            mode_label_text = mode_label(mode_use_mcts, simulations)
            print(f"  Mode: {mode_label_text}")

            mode_workers = resolve_eval_workers(workers, mode_use_mcts, elo_cfg)
            mode_elo_cfg = dict(mode_runtime_configs[eval_mode])
            mode_elo_cfg["workers"] = int(mode_workers)
            initial_elo = entry.get("elo_mcts" if mode_use_mcts else "elo_nn")
            if mode_use_mcts:
                by_sims = dict(entry.get("elo_mcts_by_simulations") or {})
                sims_entry = by_sims.get(int(simulations)) or by_sims.get(str(int(simulations)))
                if isinstance(sims_entry, dict) and _safe_float(sims_entry.get("elo")) is not None:
                    initial_elo = _safe_float(sims_entry.get("elo"))
            if _safe_float(initial_elo) is not None:
                mode_elo_cfg["adaptive_initial_elo"] = float(initial_elo)

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
            result["mcts_simulations"] = int(simulations) if mode_use_mcts else 0
            all_results.append(result)

            elo = result.get("estimated_elo")
            print(f"  Estimated {mode_label_text} Elo: {elo if elo is not None else 'N/A'}")

            persist_estimated_elo(
                model_path,
                elo,
                levels=levels,
                games_per_level=games_per_level,
                use_mcts=mode_use_mcts,
                simulations=simulations,
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
