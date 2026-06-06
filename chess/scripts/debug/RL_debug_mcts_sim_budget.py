"""
Compare eval MCTS budgets on the same model pair and the same opening sample.

This is a diagnostic for cases where a learner scores better at a lower eval
simulation count than at a higher one. Each simulation budget plays candidate
vs opponent on identical fixed-opening game indices, with paired colors.
"""

import argparse
import copy
import csv
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.model import load_model
from utils.rl.training_rl import evaluate_models


PRESETS = {
    "quick": {
        "games": 60,
        "repeats": 1,
        "simulations": "160,320",
        "opening_plies": "6",
    },
    "standard": {
        "games": 100,
        "repeats": 2,
        "simulations": "160,320",
        "opening_plies": "6",
    },
    "full": {
        "games": 200,
        "repeats": 2,
        "simulations": "160,320",
        "opening_plies": "6",
    },
}


def _parse_number_list(raw_value, *, cast, minimum, label):
    values = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(max(minimum, cast(part)))
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError(f"Provide at least one {label}.")
    return values


def _parse_simulations(raw_value):
    return _parse_number_list(raw_value, cast=int, minimum=1, label="simulation count")


def _parse_opening_plies(raw_value):
    return _parse_number_list(raw_value, cast=int, minimum=0, label="opening-prefix ply count")


def _score_ci95(wins, draws, losses, unresolved=0):
    games = int(wins) + int(draws) + int(losses) + int(unresolved)
    if games <= 0:
        return 0.0, 0.0
    score = (float(wins) + 0.5 * float(draws)) / float(games)
    second_moment = (float(wins) + 0.25 * float(draws)) / float(games)
    variance = max(0.0, second_moment - score * score)
    margin = 1.96 * math.sqrt(variance / float(games))
    return max(0.0, score - margin), min(1.0, score + margin)


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def _resolve_checkpoint(raw_value, config, *, default_kind):
    paths_cfg = config.get("paths", {})
    aliases = {
        "latest": PROJECT_DIR / "models" / "RL" / "v8_latest.pt",
        "rl_latest": PROJECT_DIR / "models" / "RL" / "v8_latest.pt",
        "best": PROJECT_DIR / paths_cfg.get("best_model_rl", "models/best_model_rl.pt"),
        "rl_best": PROJECT_DIR / paths_cfg.get("best_model_rl", "models/best_model_rl.pt"),
        "il": PROJECT_DIR / paths_cfg.get("best_model_il", "models/best_model_il.pt"),
        "il_best": PROJECT_DIR / paths_cfg.get("best_model_il", "models/best_model_il.pt"),
        "il_swa": PROJECT_DIR / "models" / "best_model_il_swa.pt",
    }
    if raw_value is None:
        raw_value = "latest" if default_kind == "candidate" else "best"
    key = str(raw_value).strip().lower()
    path = aliases.get(key)
    if path is None:
        path = Path(raw_value)
        if not path.is_absolute():
            path = PROJECT_DIR / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def _build_eval_config(config, *, simulations, opening_plies, workers, no_central_inference):
    eval_config = copy.deepcopy(config)
    eval_config.setdefault("model", {})["print_summary"] = False
    rl_cfg = eval_config.setdefault("reinforcement_learning", {})
    rl_cfg["mcts_simulations"] = max(1, int(simulations))
    rl_cfg["eval_mcts_simulations_multiplier"] = 1.0
    rl_cfg["eval_fixed_openings_enabled"] = int(opening_plies) > 0
    rl_cfg["eval_fixed_openings_pair_games"] = True
    rl_cfg["eval_fixed_openings_max_plies"] = max(0, int(opening_plies))
    if workers is not None:
        rl_cfg["eval_workers"] = max(1, int(workers))
    if no_central_inference:
        rl_cfg["eval_central_inference_enabled"] = False
    return eval_config


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _trial_key(row):
    return (
        int(row["simulations"]),
        int(row["opening_plies"]),
        int(row["repeat"]),
        int(row["opening_offset_pairs"]),
    )


def _summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["opening_plies"]), int(row["simulations"]))].append(row)

    summaries = []
    for (opening_plies, simulations), group_rows in grouped.items():
        wins = sum(int(row["wins"]) for row in group_rows)
        draws = sum(int(row["draws"]) for row in group_rows)
        losses = sum(int(row["losses"]) for row in group_rows)
        unresolved = sum(int(row["unresolved"]) for row in group_rows)
        games = wins + draws + losses + unresolved
        score_rate = (wins + 0.5 * draws) / games if games > 0 else 0.0
        ci_low, ci_high = _score_ci95(wins, draws, losses, unresolved)
        repeat_scores = [float(row["score_rate"]) for row in group_rows]
        decisive_games = wins + losses
        summaries.append({
            "opening_plies": opening_plies,
            "simulations": simulations,
            "trials": len(group_rows),
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "unresolved": unresolved,
            "score_rate": score_rate,
            "score_edge_vs_parity": score_rate - 0.5,
            "score_ci95_low": ci_low,
            "score_ci95_high": ci_high,
            "repeat_score_std": statistics.stdev(repeat_scores) if len(repeat_scores) > 1 else 0.0,
            "decisive_games": decisive_games,
            "decisive_win_rate": wins / decisive_games if decisive_games > 0 else 0.0,
            "elapsed_s": sum(float(row["elapsed_s"]) for row in group_rows),
        })

    baseline_by_opening = {}
    for row in summaries:
        opening = int(row["opening_plies"])
        current = baseline_by_opening.get(opening)
        if current is None or int(row["simulations"]) < int(current["simulations"]):
            baseline_by_opening[opening] = row

    for row in summaries:
        baseline = baseline_by_opening[int(row["opening_plies"])]
        row["baseline_simulations"] = int(baseline["simulations"])
        row["score_delta_vs_baseline"] = float(row["score_rate"]) - float(baseline["score_rate"])
        row["decisive_delta_vs_baseline"] = (
            float(row["decisive_win_rate"]) - float(baseline["decisive_win_rate"])
        )

    summaries.sort(key=lambda row: (int(row["opening_plies"]), int(row["simulations"])))
    return summaries


def _print_summary(summary_rows):
    print("\nMCTS budget comparison:")
    for row in summary_rows:
        print(
            f"  opening={int(row['opening_plies']):>2} sims={int(row['simulations']):>4} "
            f"| W/D/L={int(row['wins'])}/{int(row['draws'])}/{int(row['losses'])} "
            f"| score={float(row['score_rate']):.2%} "
            f"| CI95=[{float(row['score_ci95_low']):.2%}, {float(row['score_ci95_high']):.2%}] "
            f"| delta_vs_{int(row['baseline_simulations'])}={float(row['score_delta_vs_baseline']):+.2%}"
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Debug whether a larger eval MCTS budget hurts on the same fixed-opening sample."
    )
    parser.add_argument("--checkpoint", default=None, help="Candidate checkpoint path or alias: latest, best, il, il_swa.")
    parser.add_argument("--opponent", default=None, help="Opponent checkpoint path or alias: best, latest, il, il_swa.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--games", type=int, default=None, help="Override games per simulation budget.")
    parser.add_argument("--repeats", type=int, default=None, help="Repeat the same budget comparison on shifted openings.")
    parser.add_argument("--simulations-list", default=None, help="Comma-separated budgets, e.g. 160,320,480.")
    parser.add_argument("--opening-plies-list", default=None, help="Comma-separated opening-prefix depths, e.g. 0,6.")
    parser.add_argument("--repeat-offset-pairs", type=int, default=50, help="Opening-pair offset between repeats.")
    parser.add_argument("--device", default=None, help="Override config hardware.device.")
    parser.add_argument("--workers", type=int, default=None, help="Override eval worker count.")
    parser.add_argument("--no-central-inference", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Raw trial CSV path.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume an existing raw trial CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print the debug plan without loading models.")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.output is not None and args.resume is not None:
        raise ValueError("Use either --output or --resume, not both.")

    config = _load_config()
    preset = PRESETS[args.preset]
    games = max(2, int(args.games if args.games is not None else preset["games"]))
    if games % 2 != 0:
        games += 1
    repeats = max(1, int(args.repeats if args.repeats is not None else preset["repeats"]))
    simulations = _parse_simulations(args.simulations_list or preset["simulations"])
    opening_plies = _parse_opening_plies(args.opening_plies_list or preset["opening_plies"])
    repeat_offset_pairs = max(0, int(args.repeat_offset_pairs))

    candidate_path = _resolve_checkpoint(args.checkpoint, config, default_kind="candidate")
    opponent_path = _resolve_checkpoint(args.opponent, config, default_kind="opponent")

    output_path = args.resume or args.output
    if output_path is None:
        output_dir = PROJECT_DIR / config.get("paths", {}).get("logs_dir", "logs") / "csv"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"mcts_sim_budget_debug_{timestamp}.csv"
    output_path = output_path.resolve()

    rows = []
    if args.resume is not None:
        if not output_path.exists():
            raise FileNotFoundError(f"Resume CSV does not exist: {output_path}")
        with open(output_path, newline="", encoding="utf-8") as file_obj:
            rows = list(csv.DictReader(file_obj))
    completed_keys = {_trial_key(row) for row in rows}

    trials = []
    for opening_ply_count in opening_plies:
        for repeat in range(repeats):
            opening_offset_pairs = int(repeat * repeat_offset_pairs)
            for simulation_count in simulations:
                trials.append((simulation_count, opening_ply_count, repeat, opening_offset_pairs))
    pending_trials = [trial for trial in trials if trial not in completed_keys]

    print(f"Preset: {args.preset}")
    print(f"Candidate: {candidate_path}")
    print(f"Opponent:  {opponent_path}")
    print(f"Games per budget: {games} paired-color games")
    print(f"Repeats: {repeats} | repeat offset pairs: {repeat_offset_pairs}")
    print(f"Simulations: {simulations}")
    print(f"Opening prefix plies: {opening_plies}")
    print(f"Pending trials: {len(pending_trials)} | pending games: {len(pending_trials) * games}")
    if args.dry_run:
        return

    device_name = args.device or config.get("hardware", {}).get("device", "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    base_config = copy.deepcopy(config)
    base_config.setdefault("model", {})["print_summary"] = False

    print(f"\nLoading candidate: {candidate_path}")
    candidate = load_model(str(candidate_path), base_config, device, strict=True)
    candidate.eval()
    print(f"Loading opponent:  {opponent_path}")
    if candidate_path == opponent_path:
        opponent = candidate
    else:
        opponent = load_model(str(opponent_path), base_config, device, strict=True)
        opponent.eval()
    print(f"Device: {device}")

    raw_fieldnames = [
        "timestamp", "candidate", "opponent", "preset", "simulations",
        "opening_plies", "repeat", "opening_offset_pairs", "game_index_offset",
        "games", "wins", "draws", "losses", "unresolved", "score_rate",
        "score_edge_vs_parity", "score_ci95_low", "score_ci95_high",
        "decisive_games", "decisive_win_rate", "elapsed_s",
    ]

    for trial_idx, (simulation_count, opening_ply_count, repeat, opening_offset_pairs) in enumerate(
        pending_trials,
        start=1,
    ):
        game_index_offset = int(opening_offset_pairs * 2)
        eval_config = _build_eval_config(
            base_config,
            simulations=simulation_count,
            opening_plies=opening_ply_count,
            workers=args.workers,
            no_central_inference=args.no_central_inference,
        )
        use_fixed_openings = int(opening_ply_count) > 0
        print(
            f"\n[{trial_idx}/{len(pending_trials)}] sims={simulation_count}, "
            f"opening={opening_ply_count}, repeat={repeat + 1}/{repeats}, "
            f"offset_pairs={opening_offset_pairs}"
        )
        started = time.perf_counter()
        stats = evaluate_models(
            candidate,
            opponent,
            eval_config,
            device,
            games,
            game_index_offset=game_index_offset,
            use_fixed_openings=use_fixed_openings,
            model1_mcts_config=eval_config,
            model2_mcts_config=eval_config,
        )
        elapsed_s = time.perf_counter() - started
        wins = int(stats.get("wins", 0))
        draws = int(stats.get("draws", 0))
        losses = int(stats.get("losses", 0))
        unresolved = int(stats.get("unresolved", 0))
        score_rate = float(stats.get("score_rate", 0.0))
        decisive_games = wins + losses
        ci_low, ci_high = _score_ci95(wins, draws, losses, unresolved)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidate": str(candidate_path),
            "opponent": str(opponent_path),
            "preset": args.preset,
            "simulations": simulation_count,
            "opening_plies": opening_ply_count,
            "repeat": repeat,
            "opening_offset_pairs": opening_offset_pairs,
            "game_index_offset": game_index_offset,
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "unresolved": unresolved,
            "score_rate": score_rate,
            "score_edge_vs_parity": score_rate - 0.5,
            "score_ci95_low": ci_low,
            "score_ci95_high": ci_high,
            "decisive_games": decisive_games,
            "decisive_win_rate": wins / decisive_games if decisive_games > 0 else 0.0,
            "elapsed_s": elapsed_s,
        }
        rows.append(row)
        _write_csv(output_path, rows, raw_fieldnames)
        summary_rows = _summarize_rows(rows)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
        _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
        print(
            f"W/D/L={wins}/{draws}/{losses} | score={score_rate:.2%} "
            f"| CI95=[{ci_low:.2%}, {ci_high:.2%}] | elapsed={elapsed_s:.1f}s"
        )

    if not rows:
        print("\nNo completed trials to summarize.")
        return

    summary_rows = _summarize_rows(rows)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    _print_summary(summary_rows)
    print(f"\nSaved raw trials: {output_path}")
    print(f"Saved summary:    {summary_path}")


if __name__ == "__main__":
    main()
