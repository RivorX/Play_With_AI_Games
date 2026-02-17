"""Interactive Elo estimation tool for chess models.

This script intentionally does not accept CLI arguments.
Run it without parameters and use the menu.
"""

import copy
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet, transfer_matching_weights
from utils.shared.elo_estimator import EloEstimator, ensure_stockfish
from utils.shared.model_catalog import (
    load_checkpoint_metadata,
    print_model_table,
)


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


def _is_tty():
    return sys.stdin.isatty()


def _print_block(title):
    line = "=" * 96
    print(f"\n{line}")
    print(title)
    print(line)


def _prompt_text(prompt, default):
    if not _is_tty():
        return str(default)
    try:
        raw = input(f"{prompt} (default {default}): ").strip()
    except EOFError:
        raw = ""
    return raw or str(default)


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


def _prompt_yes_no(prompt, default):
    default = bool(default)
    default_str = "y" if default else "n"
    if not _is_tty():
        return default

    while True:
        try:
            raw = input(f"{prompt} [y/n] (default {default_str}): ").strip().lower()
        except EOFError:
            raw = ""

        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Invalid choice. Enter y or n.")


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
        config = yaml.safe_load(f)
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
        "use_se2d_blocks",
        "drop_path_rate",
        "use_coord_conv",
        "use_layer_scale",
        "layer_scale_init",
        "use_multitask_learning",
        "policy_head_conv_filters",
        "policy_head_conv_groups",
        "policy_head_global_dim",
        "policy_head_hidden_dim",
        "value_head_filters",
        "value_hidden_dim",
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

    overrides["use_se2d_blocks"] = any(".se.fc1.weight" in key for key in state_dict.keys())
    overrides["use_layer_scale"] = any(".layer_scale.gamma" in key for key in state_dict.keys())
    overrides["use_multitask_learning"] = "win_fc1.weight" in state_dict

    return overrides


def _extract_arch_overrides(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}

    direct = checkpoint.get("model_architecture")
    if isinstance(direct, dict):
        return {k: direct.get(k) for k in _model_arch_keys() if k in direct}

    legacy = {}
    for key in _model_arch_keys():
        if key in checkpoint and checkpoint[key] is not None:
            legacy[key] = checkpoint[key]
    return legacy


def collect_model_candidates(chess_dir, config, max_candidates=200):
    candidates = []

    best_rel = config.get("paths", {}).get("best_model_il", "models/best_model_il.pt")
    best_path = (chess_dir / best_rel).resolve()
    best_swa_path = (best_path.parent / "best_model_il_swa.pt").resolve()

    for path in (best_path, best_swa_path):
        if path.exists() and path not in candidates:
            candidates.append(path)

    models_dir = chess_dir / "models"
    all_pts = []
    if models_dir.exists():
        all_pts = sorted(models_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in all_pts[:max_candidates]:
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    return candidates, best_path


def build_model_catalog(model_paths, base_dir):
    return [load_checkpoint_metadata(path, base_dir) for path in model_paths]


def print_model_catalog(catalog):
    print_model_table(
        catalog,
        title="Available Checkpoints",
        show_folder=False,
        show_version=False,
        show_modified=False,
        show_swa=False,
        show_opt=False,
        group_by_folder=False,
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
    default_games = int(elo_cfg.get("games_per_level", 6))
    default_use_mcts = bool(elo_cfg.get("use_mcts", False))
    default_sims = int(elo_cfg.get("mcts_simulations", 100))
    default_sf_time = float(elo_cfg.get("stockfish_time_limit", 0.05))
    default_max_moves = int(elo_cfg.get("max_moves", 150))
    default_sf_path = str(elo_cfg.get("stockfish_path", "stockfish"))
    default_workers = int(elo_cfg.get("workers", 0))

    if _is_tty():
        _print_block("Evaluation Settings")
        print(f"Levels (from config): {levels}")
    games_per_level = _prompt_int("Games per level", default_games, min_value=1)
    
    workers = _prompt_int("Parallel workers (0=auto)", default_workers, min_value=0)

    if _is_tty():
        mcts_default_idx = 1 if default_use_mcts else 0
        mcts_mode = _prompt_menu(
            "Inference mode:",
            [
                "Raw network (faster)",
                "MCTS (stronger, slower)",
            ],
            default_idx=mcts_default_idx,
        )
        use_mcts = mcts_mode == 1
    else:
        use_mcts = default_use_mcts

    simulations = default_sims
    if use_mcts:
        simulations = _prompt_int("MCTS simulations", default_sims, min_value=1)

    sf_time = _prompt_float("Stockfish time per move (seconds)", default_sf_time, min_value=0.0)
    max_moves = _prompt_int("Max moves per game", default_max_moves, min_value=1)
    sf_path = _prompt_text("Stockfish path", default_sf_path)

    return {
        "levels": levels,
        "games_per_level": games_per_level,
        "workers": workers,
        "use_mcts": use_mcts,
        "simulations": simulations,
        "sf_time": sf_time,
        "max_moves": max_moves,
        "sf_path": sf_path,
    }


def choose_output_path(chess_dir, model_count):
    default_save = model_count > 1
    save_csv = _prompt_yes_no("Save CSV report", default_save)
    if not save_csv:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = chess_dir / "logs" / f"elo_comparison_{timestamp}.csv"
    raw = _prompt_text("Output CSV path", str(default_path))

    output_path = Path(raw)
    if not output_path.is_absolute():
        output_path = (chess_dir / output_path).resolve()
    return output_path


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

    elo = _safe_float(checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo")))
    if elo is not None:
        info_parts.append(f"elo={int(round(elo))}")

    version = checkpoint.get("version")
    if version is not None:
        info_parts.append(str(version))

    info_parts.append(f"mode={load_mode}")
    info = ", ".join(info_parts) if info_parts else "no metadata"
    print(f"  + Loaded: {checkpoint_path.name} ({info})")

    model.eval()
    return model


def persist_estimated_elo(
    checkpoint_path: Path,
    estimated_elo,
    *,
    levels: list[int],
    games_per_level: int,
    use_mcts: bool,
    simulations: int,
    sf_time: float,
):
    """Persist Elo into checkpoint metadata (always overwrites existing value)."""
    elo_value = _safe_float(estimated_elo)
    if elo_value is None:
        return

    if not checkpoint_path.exists():
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  ! Could not open checkpoint for Elo persist: {exc}")
        return

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return

    existing_elo = _safe_float(checkpoint.get("estimated_elo", checkpoint.get("last_estimated_elo")))
    
    checkpoint["estimated_elo"] = float(elo_value)
    checkpoint["last_estimated_elo"] = float(elo_value)

    epoch_raw = checkpoint.get("epoch")
    epoch_idx = _safe_int(epoch_raw)
    if epoch_idx is not None:
        checkpoint["estimated_elo_epoch"] = epoch_idx + 1
    elif checkpoint.get("estimated_elo_epoch") is None:
        checkpoint["estimated_elo_epoch"] = None

    checkpoint["estimated_elo_source"] = "eval_elo_manual"
    checkpoint["estimated_elo_timestamp"] = datetime.now().isoformat(timespec="seconds")
    checkpoint["estimated_elo_settings"] = {
        "levels": [int(x) for x in levels],
        "games_per_level": int(games_per_level),
        "use_mcts": bool(use_mcts),
        "simulations": int(simulations),
        "stockfish_time_limit": float(sf_time),
    }

    try:
        torch.save(checkpoint, checkpoint_path)
        if existing_elo is not None:
            print(f"  ✓ Updated Elo: {int(round(existing_elo))} → {int(round(elo_value))}")
        else:
            print(f"  ✓ Saved Elo into checkpoint metadata: {int(round(elo_value))}")
    except Exception as exc:
        print(f"  ! Could not save Elo to checkpoint: {exc}")


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
        elo = result.get("estimated_elo")
        elo_str = str(elo) if elo is not None else "N/A"
        lines.append(f"- {name}")
        lines.append(f"  Estimated Elo: {elo_str}")

        if result.get("results"):
            parts = []
            for lvl in levels:
                if lvl in result["results"]:
                    r = result["results"][lvl]
                    parts.append(
                        f"vs {lvl}: W{r['wins']}/D{r['draws']}/L{r['losses']} ({r['score']:.0%})"
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

        lines.append(f"{'#':<4} {'Model':<56} {'Elo':>8} {'AvgScore':>10}")
        for idx, result in enumerate(sorted_results, start=1):
            name = result.get("model_name", "unknown")[:56]
            elo = result.get("estimated_elo")
            elo_str = str(elo) if elo is not None else "N/A"
            model_results = result.get("results", {})
            if model_results:
                avg_score = sum(v["score"] for v in model_results.values()) / len(model_results)
            else:
                avg_score = 0.0
            lines.append(f"{idx:<4} {name:<56} {elo_str:>8} {avg_score:>10.0%}")

        lines.append("")

    lines.append("=" * 96)
    return "\n".join(lines)


def save_results_csv(all_results: list[dict], output_path: Path, levels: list[int]):
    """Save results to CSV for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model", "estimated_elo", "total_games", "time_seconds"]
        for lvl in sorted(levels):
            header.extend([f"vs_{lvl}_wins", f"vs_{lvl}_draws", f"vs_{lvl}_losses", f"vs_{lvl}_score"])
        writer.writerow(header)

        for result in all_results:
            row = [
                result.get("model_name", ""),
                result.get("estimated_elo", ""),
                result.get("total_games", 0),
                f"{result.get('total_time', 0):.1f}",
            ]
            for lvl in sorted(levels):
                if lvl in result.get("results", {}):
                    r = result["results"][lvl]
                    row.extend([r["wins"], r["draws"], r["losses"], f"{r['score']:.3f}"])
                else:
                    row.extend(["", "", "", ""])
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


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
    use_mcts = settings["use_mcts"]
    simulations = settings["simulations"]
    sf_time = settings["sf_time"]
    max_moves = settings["max_moves"]
    sf_path = settings["sf_path"]

    model_paths, best_path = collect_model_candidates(chess_dir, config)
    if not model_paths:
        print("No model checkpoints found in chess/models.")
        sys.exit(1)

    catalog = build_model_catalog(model_paths, chess_dir)
    print_model_catalog(catalog)

    selected_entries = choose_model_entries(catalog, best_path)
    if not selected_entries:
        print("No valid models selected.")
        sys.exit(1)

    selected_paths = [entry["path"] for entry in selected_entries]

    output_path = choose_output_path(chess_dir, len(selected_paths))

    total_games = len(selected_paths) * len(levels) * games_per_level
    mode_str = f"MCTS ({simulations} sims)" if use_mcts else "Raw network"
    workers_str = f"{workers} workers" if workers > 1 else "sequential"

    _print_block("Elo Estimation Plan")
    print(f"Models:         {len(selected_paths)}")
    print(f"Levels:         {levels}")
    print(f"Games/level:    {games_per_level}")
    print(f"Total games:    {total_games}")
    print(f"Mode:           {mode_str}")
    print(f"Workers:        {workers_str}")
    print(f"SF time/move:   {sf_time}s")
    print(f"Max moves:      {max_moves}")
    print(f"Stockfish path: {sf_path}")
    if output_path is not None:
        print(f"CSV output:     {output_path}")

    # Ensure Stockfish is available
    sf_path = ensure_stockfish(sf_path)

    all_results = []
    t0 = time.perf_counter()

    for idx, model_path in enumerate(selected_paths, start=1):
        print(f"\n[{idx}/{len(selected_paths)}] {model_path.name}")

        model = load_model(model_path, config, device)
        if model is None:
            continue

        estimator = EloEstimator(model, config, device, sf_path)
        result = estimator.estimate(
            levels=levels,
            games_per_level=games_per_level,
            stockfish_time_limit=sf_time,
            max_moves=max_moves,
            use_mcts=use_mcts,
            simulations=simulations,
            workers=workers,
        )

        result["model_name"] = model_path.name
        result["model_path"] = str(model_path)
        all_results.append(result)

        elo = result.get("estimated_elo")
        print(f"  Estimated Elo: {elo if elo is not None else 'N/A'}")

        persist_estimated_elo(
            model_path,
            elo,
            levels=levels,
            games_per_level=games_per_level,
            use_mcts=use_mcts,
            simulations=simulations,
            sf_time=sf_time,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t0

    print(format_results_table(all_results))
    print(f"Total time: {total_elapsed:.1f}s")

    if output_path is not None and all_results:
        save_results_csv(all_results, output_path, levels)


if __name__ == "__main__":
    main()
