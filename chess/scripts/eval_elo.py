"""
Elo Estimation Tool — Compare trained models against Stockfish

Usage examples:
  # Evaluate best model (default)
  python eval_elo.py

  # Evaluate a specific checkpoint
  python eval_elo.py --model models/IL/il_epoch_10_valloss_3.3240_top1_0.474.pt

  # Compare multiple models
  python eval_elo.py --model models/best_model_il.pt models/IL/il_epoch_5*.pt

  # Use MCTS (slower, but stronger Elo)
  python eval_elo.py --mcts --simulations 200

  # Custom levels and more games
  python eval_elo.py --levels 800 1200 1600 2000 --games 10

  # Quick test (fewer games, fewer levels)
  python eval_elo.py --quick
"""

import argparse
import copy
import glob
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.model import ChessNet
from utils.shared.elo_estimator import EloEstimator, ensure_stockfish


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config.setdefault('model', {})
    config['model']['print_summary'] = False
    return config


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    """Load model from checkpoint."""
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Print checkpoint info
        info_parts = []
        if 'epoch' in checkpoint:
            info_parts.append(f"epoch {checkpoint['epoch'] + 1}")
        if 'loss' in checkpoint:
            info_parts.append(f"val_loss={checkpoint['loss']:.4f}")
        if 'val_policy_top1' in checkpoint:
            info_parts.append(f"top1={checkpoint['val_policy_top1']:.2%}")
        if 'version' in checkpoint:
            info_parts.append(f"{checkpoint['version']}")
        info = ', '.join(info_parts)
        print(f"  ✓ Loaded: {checkpoint_path.name} ({info})")
    else:
        print(f"  ✗ Not found: {checkpoint_path}")
        return None

    model.eval()
    return model


def resolve_model_paths(patterns: list[str], base_dir: Path) -> list[Path]:
    """Resolve model path patterns (supports wildcards)."""
    paths = []
    for pattern in patterns:
        p = Path(pattern)
        if not p.is_absolute():
            p = base_dir / pattern

        # Support glob patterns
        if '*' in str(p) or '?' in str(p):
            matches = sorted(glob.glob(str(p)))
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(p)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def format_results_table(all_results: list[dict]) -> str:
    """Format results as a nice comparison table."""
    if not all_results:
        return "No results."

    # Gather all levels
    all_levels = set()
    for r in all_results:
        all_levels.update(r['results'].keys())
    levels = sorted(all_levels)

    # Header
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  ELO ESTIMATION RESULTS")
    lines.append("=" * 90)
    lines.append("")

    # Per-model details
    for r in all_results:
        name = r['model_name']
        elo = r.get('estimated_elo')
        elo_str = str(elo) if elo is not None else "N/A"
        lines.append(f"  ♚ {name}")
        lines.append(f"    Estimated Elo: {elo_str}")

        if r.get('results'):
            parts = []
            for lvl in levels:
                if lvl in r['results']:
                    res = r['results'][lvl]
                    parts.append(f"vs {lvl}: W{res['wins']}/D{res['draws']}/L{res['losses']} ({res['score']:.0%})")
            lines.append(f"    {' | '.join(parts)}")

        elapsed = r.get('total_time', 0)
        games = r.get('total_games', 0)
        lines.append(f"    ⏱ {elapsed:.1f}s ({games} games)")
        lines.append("")

    # Summary table if multiple models
    if len(all_results) > 1:
        lines.append("-" * 90)
        lines.append("  COMPARISON (sorted by Elo)")
        lines.append("-" * 90)

        # Sort by Elo (None last)
        sorted_results = sorted(
            all_results,
            key=lambda r: r.get('estimated_elo') or -9999,
            reverse=True
        )

        lines.append(f"  {'#':<4} {'Model':<50} {'Elo':>6}  {'Score':>6}")
        lines.append(f"  {'─'*4} {'─'*50} {'─'*6}  {'─'*6}")

        for i, r in enumerate(sorted_results, 1):
            name = r['model_name'][:50]
            elo = r.get('estimated_elo')
            elo_str = str(elo) if elo is not None else "N/A"
            total_score = sum(
                res['score'] for res in r.get('results', {}).values()
            )
            num_levels = max(1, len(r.get('results', {})))
            avg_score = total_score / num_levels if r.get('results') else 0
            lines.append(f"  {i:<4} {name:<50} {elo_str:>6}  {avg_score:>5.0%}")

        lines.append("")

    lines.append("=" * 90)
    return "\n".join(lines)


def save_results_csv(all_results: list[dict], output_path: Path, levels: list[int]):
    """Save results to CSV for further analysis."""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        header = ['model', 'estimated_elo', 'total_games', 'time_seconds']
        for lvl in sorted(levels):
            header.extend([f'vs_{lvl}_wins', f'vs_{lvl}_draws', f'vs_{lvl}_losses', f'vs_{lvl}_score'])
        writer.writerow(header)

        # Data
        for r in all_results:
            row = [
                r['model_name'],
                r.get('estimated_elo', ''),
                r.get('total_games', 0),
                f"{r.get('total_time', 0):.1f}",
            ]
            for lvl in sorted(levels):
                if lvl in r.get('results', {}):
                    res = r['results'][lvl]
                    row.extend([res['wins'], res['draws'], res['losses'], f"{res['score']:.3f}"])
                else:
                    row.extend(['', '', '', ''])
            writer.writerow(row)

    print(f"\n📄 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate Elo of trained chess models by playing against Stockfish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model', nargs='+', default=None,
        help='Model checkpoint path(s). Supports wildcards. Default: best_model_il.pt'
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to config YAML. Default: chess/config/config.yaml'
    )
    parser.add_argument(
        '--levels', nargs='+', type=int, default=None,
        help='Stockfish UCI_Elo levels. Default from config or [1000,1300,1600,1900,2200]'
    )
    parser.add_argument(
        '--games', type=int, default=None,
        help='Games per level (half as white, half as black). Default from config or 6'
    )
    parser.add_argument(
        '--mcts', action='store_true',
        help='Use MCTS for model moves (stronger but slower)'
    )
    parser.add_argument(
        '--simulations', type=int, default=None,
        help='MCTS simulations per move (default: 100)'
    )
    parser.add_argument(
        '--sf-time', type=float, default=None,
        help='Seconds per Stockfish move (default: 0.05)'
    )
    parser.add_argument(
        '--stockfish', default=None,
        help='Path to Stockfish binary'
    )
    parser.add_argument(
        '--device', default='',
        help='Device override (cuda/cpu)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer games (2/level), fewer levels [1200,1600,2000]'
    )
    parser.add_argument(
        '--output', default=None,
        help='Save results to CSV file'
    )
    args = parser.parse_args()

    # Resolve paths
    chess_dir = script_dir.parent
    config_path = Path(args.config) if args.config else chess_dir / 'config' / 'config.yaml'
    config = load_config(config_path)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get('hardware', {}).get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Elo config
    elo_cfg = config.get('elo_estimation', {})

    if args.quick:
        levels = [1320, 1600, 2000]
        games_per_level = 2
    else:
        levels = args.levels or elo_cfg.get('levels', [1320, 1500, 1700, 1900, 2200])
        games_per_level = args.games or elo_cfg.get('games_per_level', 6)

    use_mcts = args.mcts or elo_cfg.get('use_mcts', False)
    simulations = args.simulations or elo_cfg.get('mcts_simulations', 100)
    sf_time = args.sf_time or elo_cfg.get('stockfish_time_limit', 0.05)
    sf_path = args.stockfish or elo_cfg.get('stockfish_path', 'stockfish')

    # Resolve model paths
    if args.model:
        model_paths = resolve_model_paths(args.model, chess_dir)
    else:
        # Default: best model
        best = chess_dir / config.get('paths', {}).get('best_model_il', 'models/best_model_il.pt')
        model_paths = [best]

    if not model_paths:
        print("No model files found!")
        sys.exit(1)

    # Print plan
    total_games = len(model_paths) * len(levels) * games_per_level
    mode_str = f"MCTS ({simulations} sims)" if use_mcts else "Raw network"
    print(f"\n{'='*60}")
    print(f"  Elo Estimation Plan")
    print(f"{'='*60}")
    print(f"  Models:         {len(model_paths)}")
    print(f"  Levels:         {levels}")
    print(f"  Games/level:    {games_per_level}")
    print(f"  Total games:    {total_games}")
    print(f"  Mode:           {mode_str}")
    print(f"  SF time/move:   {sf_time}s")
    print(f"{'='*60}\n")

    # Ensure Stockfish is available
    sf_path = ensure_stockfish(sf_path)
    print()

    # Run evaluations
    all_results = []
    t0 = time.perf_counter()

    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] {model_path.name}")
        model = load_model(model_path, config, device)
        if model is None:
            continue

        estimator = EloEstimator(model, config, device, sf_path)
        result = estimator.estimate(
            levels=levels,
            games_per_level=games_per_level,
            stockfish_time_limit=sf_time,
            max_moves=150,
            use_mcts=use_mcts,
            simulations=simulations,
        )

        result['model_name'] = model_path.name
        result['model_path'] = str(model_path)
        all_results.append(result)

        elo = result.get('estimated_elo')
        print(f"  → Estimated Elo: {elo if elo is not None else 'N/A'}")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t0

    # Print results
    print(format_results_table(all_results))
    print(f"  Total time: {total_elapsed:.1f}s")

    # Save CSV
    if args.output:
        output_path = Path(args.output)
    elif len(all_results) > 1:
        # Auto-save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = chess_dir / 'logs' / f'elo_comparison_{timestamp}.csv'
    else:
        output_path = None

    if output_path and all_results:
        save_results_csv(all_results, output_path, levels)


if __name__ == '__main__':
    main()
