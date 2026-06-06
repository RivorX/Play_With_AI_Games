"""
Benchmark MCTS Q steering on one frozen chess model.

Each trial plays Q-on against Q-off using the same network, paired openings,
and alternating colors. Sweeps can vary Q weight, search simulations, opening
prefix depth, and repeat offsets without mixing in RL training drift.
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

scripts_dir = Path(__file__).resolve().parent.parent
project_dir = scripts_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(scripts_dir))

from src.model import load_model
from utils.rl.training_rl import evaluate_models


PRESETS = {
    'quick': {
        'games': 100,
        'repeats': 1,
        'simulations': '160',
        'q_weights': '0,0.02,0.04,0.06,0.08,0.10',
        'opening_plies': '6',
    },
    'selfplay': {
        'games': 200,
        'repeats': 2,
        'simulations': '160',
        'q_weights': '0,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12',
        'opening_plies': '6',
    },
    'full': {
        'games': 200,
        'repeats': 1,
        'simulations': '160,320',
        'q_weights': '0,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12',
        'opening_plies': '6',
    },
}


def _parse_number_list(raw_value, *, cast, minimum, label):
    values = []
    for part in str(raw_value).split(','):
        part = part.strip()
        if not part:
            continue
        values.append(max(minimum, cast(part)))
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError(f"Provide at least one {label}.")
    return values


def _parse_q_weights(raw_value):
    return _parse_number_list(raw_value, cast=float, minimum=0.0, label="Q weight")


def _parse_simulations(raw_value):
    return _parse_number_list(raw_value, cast=int, minimum=1, label="simulation count")


def _parse_opening_plies(raw_value):
    return _parse_number_list(raw_value, cast=int, minimum=0, label="opening-prefix ply count")


def _build_q_config(config, q_weight, simulations, opening_plies):
    q_config = copy.deepcopy(config)
    rl_cfg = q_config.setdefault('reinforcement_learning', {})
    rl_cfg['mcts_simulations'] = max(1, int(simulations))
    rl_cfg['eval_mcts_simulations_multiplier'] = 1.0
    rl_cfg['mcts_q_selection_weight'] = max(0.0, float(q_weight))
    rl_cfg['eval_fixed_openings_enabled'] = int(opening_plies) > 0
    rl_cfg['eval_fixed_openings_pair_games'] = True
    rl_cfg['eval_fixed_openings_max_plies'] = max(0, int(opening_plies))
    return q_config


def _default_checkpoint(project_dir, config):
    best_il = project_dir / config['paths']['best_model_il']
    best_il_swa = best_il.parent / 'best_model_il_swa.pt'
    return best_il_swa if best_il_swa.exists() else best_il


def _score_ci95(wins, draws, losses, unresolved=0):
    games = int(wins) + int(draws) + int(losses) + int(unresolved)
    if games <= 0:
        return 0.0, 0.0
    score = (float(wins) + 0.5 * float(draws)) / float(games)
    second_moment = (float(wins) + 0.25 * float(draws)) / float(games)
    variance = max(0.0, second_moment - score * score)
    margin = 1.96 * math.sqrt(variance / float(games))
    return max(0.0, score - margin), min(1.0, score + margin)


def _trial_key(row):
    return (
        int(row['simulations']),
        int(row['opening_plies']),
        round(float(row['q_weight']), 12),
        int(row['repeat']),
    )


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        wins = sum(int(row['wins']) for row in group_rows)
        draws = sum(int(row['draws']) for row in group_rows)
        losses = sum(int(row['losses']) for row in group_rows)
        unresolved = sum(int(row['unresolved']) for row in group_rows)
        games = wins + draws + losses + unresolved
        score_rate = (wins + 0.5 * draws) / games if games > 0 else 0.0
        decisive_games = wins + losses
        ci_low, ci_high = _score_ci95(wins, draws, losses, unresolved)
        repeat_scores = [float(row['score_rate']) for row in group_rows]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update({
            'trials': len(group_rows),
            'games': games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'unresolved': unresolved,
            'score_rate': score_rate,
            'q_on_edge_vs_parity': score_rate - 0.5,
            'score_ci95_low': ci_low,
            'score_ci95_high': ci_high,
            'repeat_score_std': statistics.stdev(repeat_scores) if len(repeat_scores) > 1 else 0.0,
            'decisive_games': decisive_games,
            'decisive_win_rate': wins / decisive_games if decisive_games > 0 else 0.0,
            'elapsed_s': sum(float(row['elapsed_s']) for row in group_rows),
        })
        summaries.append(summary)
    return summaries


def _attach_observed_baseline_edges(summary_rows):
    baseline_by_profile = {}
    for row in summary_rows:
        if abs(float(row['q_weight'])) <= 1e-12:
            baseline_by_profile[(int(row['simulations']), int(row['opening_plies']))] = float(row['score_rate'])
    for row in summary_rows:
        baseline = baseline_by_profile.get((int(row['simulations']), int(row['opening_plies'])), 0.5)
        row['q_on_edge_vs_observed_qoff'] = float(row['score_rate']) - float(baseline)
    return summary_rows


def _write_summaries(raw_path, rows):
    summary_rows = _attach_observed_baseline_edges(
        _summarize_rows(rows, ('simulations', 'opening_plies', 'q_weight'))
    )
    summary_rows.sort(key=lambda row: (
        int(row['simulations']),
        int(row['opening_plies']),
        float(row['q_weight']),
    ))
    summary_path = raw_path.with_name(f"{raw_path.stem}_summary.csv")
    _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))

    ranking_rows = _summarize_rows(rows, ('q_weight',))
    ranking_rows.sort(key=lambda row: (
        -float(row['score_ci95_low']),
        -float(row['score_rate']),
        float(row['q_weight']),
    ))
    ranking_path = raw_path.with_name(f"{raw_path.stem}_ranking.csv")
    _write_csv(ranking_path, ranking_rows, list(ranking_rows[0].keys()))
    return summary_path, ranking_path, summary_rows, ranking_rows


def _print_plan(*, preset, games, repeats, simulations, q_weights, opening_plies, pending_trials):
    print(f"Preset: {preset}")
    print(f"Games per trial: {games} paired-color games")
    print(f"Repeats per profile: {repeats}")
    print(f"Simulations: {simulations}")
    print(f"Opening prefix plies: {opening_plies}")
    print(f"Q weights: {q_weights}")
    print(f"Pending trials: {pending_trials} | pending games: {pending_trials * games}")


def _print_profile_ranking(summary_rows, *, preferred_simulations):
    candidates = [
        row for row in summary_rows
        if int(row['simulations']) == int(preferred_simulations)
        and int(row['opening_plies']) == 6
        and float(row['q_weight']) > 0.0
    ]
    if not candidates:
        candidates = [row for row in summary_rows if float(row['q_weight']) > 0.0]
    candidates.sort(key=lambda row: (
        -float(row['score_ci95_low']),
        -float(row['score_rate']),
        float(row['q_weight']),
    ))
    if not candidates:
        return
    print("\nTop Q candidates:")
    for row in candidates[:5]:
        print(
            f"  sims={int(row['simulations']):>3} opening={int(row['opening_plies']):>2} "
            f"Q={float(row['q_weight']):.3f} | score={float(row['score_rate']):.2%} "
            f"CI95=[{float(row['score_ci95_low']):.2%}, {float(row['score_ci95_high']):.2%}] "
            f"| decisive={float(row['decisive_win_rate']):.2%}"
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run paired Q-on vs Q-off MCTS benchmarks on one frozen model."
    )
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--preset', choices=sorted(PRESETS), default='selfplay')
    parser.add_argument('--games', type=int, default=None, help="Override games per trial.")
    parser.add_argument('--repeats', type=int, default=None, help="Override repeats per profile.")
    parser.add_argument('--simulations', '--sims', type=int, default=None, help="Test one simulation count.")
    parser.add_argument('--simulations-list', default=None, help="Comma-separated simulation counts.")
    parser.add_argument('--q-weights', default=None, help="Comma-separated Q selection weights.")
    parser.add_argument('--opening-plies-list', default=None, help="Comma-separated opening-prefix depths, e.g. 0,6.")
    parser.add_argument('--repeat-offset-pairs', type=int, default=5, help="Shift paired openings between repeats.")
    parser.add_argument('--device', default=None, help="Override config hardware.device.")
    parser.add_argument('--workers', type=int, default=None, help="Override eval worker count.")
    parser.add_argument('--no-central-inference', action='store_true')
    parser.add_argument('--output', type=Path, default=None, help="Raw trial CSV path.")
    parser.add_argument('--resume', type=Path, default=None, help="Resume an existing raw trial CSV.")
    parser.add_argument('--dry-run', action='store_true', help="Print the benchmark plan without loading a model.")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.output is not None and args.resume is not None:
        raise ValueError("Use either --output or --resume, not both.")
    if args.simulations is not None and args.simulations_list is not None:
        raise ValueError("Use either --simulations or --simulations-list, not both.")

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    config_path = project_dir / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as file_obj:
        config = yaml.safe_load(file_obj) or {}

    preset = PRESETS[args.preset]
    games = max(2, int(args.games if args.games is not None else preset['games']))
    if games % 2 != 0:
        games += 1
    repeats = max(1, int(args.repeats if args.repeats is not None else preset['repeats']))
    simulation_raw = (
        str(args.simulations)
        if args.simulations is not None
        else (args.simulations_list if args.simulations_list is not None else preset['simulations'])
    )
    simulations = _parse_simulations(simulation_raw)
    q_weights = _parse_q_weights(args.q_weights if args.q_weights is not None else preset['q_weights'])
    opening_plies = _parse_opening_plies(
        args.opening_plies_list if args.opening_plies_list is not None else preset['opening_plies']
    )
    repeat_offset_pairs = max(0, int(args.repeat_offset_pairs))

    output_path = args.resume or args.output
    if output_path is None:
        output_dir = project_dir / config['paths']['logs_dir'] / 'csv'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'mcts_q_benchmark_{timestamp}.csv'
    output_path = output_path.resolve()

    rows = []
    if args.resume is not None:
        if not output_path.exists():
            raise FileNotFoundError(f"Resume CSV does not exist: {output_path}")
        with open(output_path, newline='', encoding='utf-8') as file_obj:
            rows = list(csv.DictReader(file_obj))
    completed_keys = {_trial_key(row) for row in rows}
    trials = [
        (simulation_count, opening_ply_count, q_weight, repeat)
        for simulation_count in simulations
        for opening_ply_count in opening_plies
        for q_weight in q_weights
        for repeat in range(repeats)
    ]
    pending_trials = [trial for trial in trials if trial not in completed_keys]
    _print_plan(
        preset=args.preset,
        games=games,
        repeats=repeats,
        simulations=simulations,
        q_weights=q_weights,
        opening_plies=opening_plies,
        pending_trials=len(pending_trials),
    )
    if args.dry_run:
        return

    device_name = args.device or config.get('hardware', {}).get('device', 'cpu')
    if str(device_name).startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device_name = 'cpu'
    device = torch.device(device_name)
    checkpoint_path = args.checkpoint or _default_checkpoint(project_dir, config)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    base_config = copy.deepcopy(config)
    base_config.setdefault('model', {})['print_summary'] = False
    rl_cfg = base_config.setdefault('reinforcement_learning', {})
    if args.workers is not None:
        rl_cfg['eval_workers'] = max(1, int(args.workers))
    if args.no_central_inference:
        rl_cfg['eval_central_inference_enabled'] = False

    print(f"\nLoading frozen model: {checkpoint_path}")
    print(f"Device: {device}")
    model = load_model(str(checkpoint_path), base_config, device, strict=True)
    model.eval()

    raw_fieldnames = [
        'timestamp', 'checkpoint', 'preset', 'simulations', 'opening_plies',
        'q_weight', 'repeat', 'opening_offset_pairs', 'games',
        'wins', 'draws', 'losses', 'unresolved', 'score_rate',
        'q_on_edge_vs_parity', 'score_ci95_low', 'score_ci95_high',
        'decisive_games', 'decisive_win_rate', 'elapsed_s',
    ]
    for trial_idx, (simulation_count, opening_ply_count, q_weight, repeat) in enumerate(pending_trials, start=1):
        opening_offset_pairs = int(repeat * repeat_offset_pairs)
        game_index_offset = int(opening_offset_pairs * 2)
        qon_config = _build_q_config(base_config, q_weight, simulation_count, opening_ply_count)
        qoff_config = _build_q_config(base_config, 0.0, simulation_count, opening_ply_count)
        print(
            f"\n[{trial_idx}/{len(pending_trials)}] sims={simulation_count}, "
            f"opening={opening_ply_count}, Q={q_weight:.4f}, repeat={repeat + 1}/{repeats}, "
            f"offset_pairs={opening_offset_pairs}"
        )
        started = time.perf_counter()
        stats = evaluate_models(
            model,
            model,
            qon_config,
            device,
            games,
            game_index_offset=game_index_offset,
            use_fixed_openings=int(opening_ply_count) > 0,
            model1_mcts_config=qon_config,
            model2_mcts_config=qoff_config,
        )
        elapsed_s = time.perf_counter() - started
        wins = int(stats.get('wins', 0))
        draws = int(stats.get('draws', 0))
        losses = int(stats.get('losses', 0))
        unresolved = int(stats.get('unresolved', 0))
        score_rate = float(stats.get('score_rate', 0.0))
        decisive_games = wins + losses
        ci_low, ci_high = _score_ci95(wins, draws, losses, unresolved)
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'checkpoint': str(checkpoint_path),
            'preset': args.preset,
            'simulations': simulation_count,
            'opening_plies': opening_ply_count,
            'q_weight': q_weight,
            'repeat': repeat,
            'opening_offset_pairs': opening_offset_pairs,
            'games': games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'unresolved': unresolved,
            'score_rate': score_rate,
            'q_on_edge_vs_parity': score_rate - 0.5,
            'score_ci95_low': ci_low,
            'score_ci95_high': ci_high,
            'decisive_games': decisive_games,
            'decisive_win_rate': wins / decisive_games if decisive_games > 0 else 0.0,
            'elapsed_s': elapsed_s,
        }
        rows.append(row)
        _write_csv(output_path, rows, raw_fieldnames)
        summary_path, ranking_path, summary_rows, _ranking_rows = _write_summaries(output_path, rows)
        print(
            f"W/D/L={wins}/{draws}/{losses} | score={score_rate:.2%} "
            f"| CI95=[{ci_low:.2%}, {ci_high:.2%}] | Q-on edge={score_rate - 0.5:+.2%} "
            f"| {elapsed_s:.1f}s"
        )

    if not rows:
        print("\nNo completed trials to summarize.")
        return
    summary_path, ranking_path, summary_rows, _ranking_rows = _write_summaries(output_path, rows)
    preferred_simulations = int(config.get('reinforcement_learning', {}).get('mcts_simulations', 160))
    _print_profile_ranking(summary_rows, preferred_simulations=preferred_simulations)
    print(f"\nSaved raw trials: {output_path}")
    print(f"Saved profile summary: {summary_path}")
    print(f"Saved overall ranking: {ranking_path}")


if __name__ == '__main__':
    main()
