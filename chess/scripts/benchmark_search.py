"""Run paired full-game Gumbel budget comparisons using the Elo runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

CHESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHESS_DIR))

from src.config import load_project_config


DEFAULT_BENCHMARK_CONFIG = CHESS_DIR / "config" / "benchmark.yaml"


def _resolve_path(value: str | Path, *, relative_to: Path = CHESS_DIR) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def load_benchmark_spec(path: str | Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    """Load and validate the user-facing benchmark-only YAML settings."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("Benchmark YAML root must be a mapping")
    benchmark = raw.get("benchmark", raw)
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be a mapping")

    baseline = _positive_int(benchmark.get("baseline_simulations", 192), "baseline_simulations")
    variants_raw = benchmark.get("variants")
    variants = None
    if variants_raw is not None:
        if not isinstance(variants_raw, list) or not variants_raw:
            raise ValueError("variants must be a non-empty list")
        variants = []
        names = set()
        for i, raw_variant in enumerate(variants_raw):
            if not isinstance(raw_variant, dict):
                raise ValueError(f"variants[{i}] must be a mapping")
            name = str(raw_variant.get("name", "")).strip()
            if not name or name in names:
                raise ValueError(f"variants[{i}] needs a unique non-empty name")
            names.add(name)
            overrides = raw_variant.get("config_overrides", {}) or {}
            if not isinstance(overrides, dict):
                raise ValueError(f"variants[{i}].config_overrides must be a mapping")
            variants.append({
                "name": name,
                "simulations": _positive_int(raw_variant.get("simulations"), f"variants[{i}].simulations"),
                "config_overrides": overrides,
            })
        simulations = [variant["simulations"] for variant in variants]
    else:
        simulations_raw = benchmark.get("simulations", [384, 768, 1536])
        if not isinstance(simulations_raw, list) or not simulations_raw:
            raise ValueError("simulations must be a non-empty list of positive integers")
        simulations = [_positive_int(value, "simulations entry") for value in simulations_raw]
    _validate_simulations(baseline, simulations, variants)
    games = _positive_int(benchmark.get("games_per_profile", 20), "games_per_profile")
    workers_raw = benchmark.get("workers")
    workers = None if workers_raw in (None, "auto") else _positive_int(workers_raw, "workers")
    checkpoint = benchmark.get("checkpoint", "models/RL/v9.9b_latest.pt")
    output_dir = benchmark.get("output_dir", "logs/benchmark/gumbel_scaling_20games_20261024")
    runtime_config = raw.get("runtime_config")
    search_overrides = benchmark.get("search_overrides", {}) or {}
    if not isinstance(search_overrides, dict):
        raise ValueError("search_overrides must be a mapping")
    central_overrides = benchmark.get("central_inference_overrides", {}) or {}
    if not isinstance(central_overrides, dict):
        raise ValueError("central_inference_overrides must be a mapping")
    return {
        "baseline_simulations": baseline,
        "simulations": simulations,
        "variants": variants,
        "search_overrides": search_overrides,
        "central_inference_overrides": central_overrides,
        "games_per_profile": games,
        "opening_suite": str(benchmark.get("opening_suite", "elo_first_10")),
        "workers": workers,
        "checkpoint": _resolve_path(checkpoint),
        "output_dir": _resolve_path(output_dir),
        "resume": bool(benchmark.get("resume", False)),
        "runtime_config": _resolve_path(runtime_config) if runtime_config else None,
        "config_file": source,
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or str(value).strip() != str(result):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_simulations(baseline: int, simulations: list[int], variants) -> None:
    if variants is None and baseline in simulations:
        raise ValueError("simulations must not repeat baseline_simulations")
    if variants is None and len(set(simulations)) != len(simulations):
        raise ValueError("simulations entries must be unique")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_BENCHMARK_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", type=int)
    parser.add_argument("--simulations", type=int, nargs="+")
    parser.add_argument("--games-per-profile", type=int)
    parser.add_argument("--workers", help="Positive worker count or 'auto' from the Elo config")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading the model or starting workers")
    return parser


def resolve_options(args: argparse.Namespace) -> dict[str, Any]:
    options = load_benchmark_spec(args.config)
    if args.checkpoint is not None:
        options["checkpoint"] = _resolve_path(args.checkpoint)
    if args.baseline is not None:
        options["baseline_simulations"] = _positive_int(args.baseline, "baseline")
    if args.simulations is not None:
        options["simulations"] = [_positive_int(v, "simulations entry") for v in args.simulations]
        options["variants"] = None
    if args.games_per_profile is not None:
        options["games_per_profile"] = _positive_int(args.games_per_profile, "games-per-profile")
    if args.workers is not None:
        options["workers"] = None if args.workers.strip().lower() == "auto" else _positive_int(args.workers, "workers")
    if args.output is not None:
        options["output_dir"] = _resolve_path(args.output)
    if args.resume is not None:
        options["resume"] = bool(args.resume)
    _validate_simulations(options["baseline_simulations"], options["simulations"], options["variants"])
    return options


def _runtime_config(options: dict[str, Any]) -> dict[str, Any]:
    return load_project_config(config_path=options["runtime_config"])


def dry_run_summary(options: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from src.evaluation.benchmark import resolve_benchmark_runtime_plan

    hardware = config.get("hardware", {}) or {}
    device_type = str(hardware.get("device", "cuda")).strip().lower()
    if device_type not in {"cpu", "cuda"}:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda" and not torch.cuda.is_available():
        device_type = "cpu"
    resolved = resolve_benchmark_runtime_plan(
        config,
        device_type=device_type,
        baseline=options["baseline_simulations"],
        simulations=options["simulations"],
        games_per_profile=options["games_per_profile"],
        workers=options["workers"],
        opening_suite=options["opening_suite"],
        variants=options["variants"],
        search_overrides=options["search_overrides"],
        central_inference_overrides=options["central_inference_overrides"],
    )
    plan = resolved["plan"]
    return {
        "mode": "dry-run",
        "checkpoint": str(options["checkpoint"]),
        "baseline_simulations": options["baseline_simulations"],
        "profiles": [
            {
                "name": profile["name"],
                "simulations": profile["simulations"],
                "search_overrides": profile["search_overrides"],
            }
            for profile in plan["profiles"]
        ],
        "search_overrides": options["search_overrides"],
        "games_per_profile": plan["games_per_profile"],
        "total_games": len(plan["games"]),
        "runtime": {
            "device": device_type,
            "workers": resolved["effective_workers"],
            "worker_resolution": resolved["worker_resolution"],
            "inference_route": resolved["inference_route"],
            "central_cuda_graph": resolved["central_cuda_graph_forward_enabled"],
            "central_compile": resolved["central_use_compile"],
            "central_graph_shapes": resolved["central_cuda_graph_hot_shapes"],
            "local_cuda_graph": resolved["local_cuda_graph_policy"],
        },
        "output_dir": str(options["output_dir"]),
        "resume": options["resume"],
    }


def _progress(event: Any) -> None:
    if not isinstance(event, dict):
        print(f"[benchmark] {event}", flush=True)
        return
    print("[benchmark] " + format_result_summary(event), flush=True)


def format_result_summary(summary: dict[str, Any], output_dir: Path | None = None) -> str:
    lines = [
        f"Benchmark {summary.get('status', 'unknown')}: "
        f"{summary.get('games_completed', 0)}/{summary.get('games_expected', 0)} games completed"
    ]
    for profile in summary.get("profiles", []):
        games = int(profile.get("games_completed", 0) or 0)
        wins = int(profile.get("wins", 0) or 0)
        draws = int(profile.get("draws", 0) or 0)
        losses = int(profile.get("losses", 0) or 0)
        if games:
            percentages = f"win {100.0 * wins / games:.1f}%, score {100.0 * float(profile.get('score_fraction', 0.0) or 0.0):.1f}%"
        else:
            percentages = "win n/a, score n/a"
        fresh_profile = profile.get("mean_fresh_simulations_profile")
        fresh_baseline = profile.get("mean_fresh_simulations_baseline")
        fresh_text = (
            f"{fresh_profile:.0f} vs {fresh_baseline:.0f}"
            if isinstance(fresh_profile, (int, float)) and isinstance(fresh_baseline, (int, float))
            else "n/a"
        )
        lines.append(
            f"{profile.get('name')} ({profile.get('simulations')}): W-D-L {wins}-{draws}-{losses} "
            f"({games}/{profile.get('games_expected', 0)}), {percentages}, "
            f"fresh sims/move {fresh_text}"
        )
    if output_dir is not None:
        lines.append(f"Artifacts: {output_dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        options = resolve_options(args)
        config = _runtime_config(options)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            summary = dry_run_summary(options, config)
        except (ValueError, TypeError) as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if not options["checkpoint"].is_file():
        print(f"Checkpoint not found: {options['checkpoint']}", file=sys.stderr)
        return 2

    try:
        import torch
        from scripts.eval_elo import load_model
        from src.evaluation.benchmark import run_benchmark
    except ImportError as exc:
        print(f"Benchmark runtime is not available yet: {exc}", file=sys.stderr)
        return 2

    try:
        cfg_device = str((config.get("hardware", {}) or {}).get("device", "cuda")).strip().lower()
        if cfg_device not in {"cpu", "cuda"}:
            cfg_device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg_device == "cuda" and not torch.cuda.is_available():
            print("CUDA configured but unavailable; using CPU.", file=sys.stderr)
            cfg_device = "cpu"
        device = torch.device(cfg_device)
        model = load_model(options["checkpoint"], config, device)
        if model is None:
            print(f"Could not load model checkpoint: {options['checkpoint']}", file=sys.stderr)
            return 2
        model.eval()
        result = run_benchmark(
            config,
            model=model,
            device=device,
            checkpoint=options["checkpoint"],
            baseline=options["baseline_simulations"],
            simulations=options["simulations"],
            games_per_profile=options["games_per_profile"],
            workers=options["workers"],
            output_dir=options["output_dir"],
            resume=options["resume"],
            opening_suite=options["opening_suite"],
            variants=options["variants"],
            search_overrides=options["search_overrides"],
            central_inference_overrides=options["central_inference_overrides"],
            progress_callback=_progress,
        )
    except Exception as exc:
        print(f"Benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = result.get("summary", result) if isinstance(result, dict) else {}
    status = str(result.get("status", "unknown")) if isinstance(result, dict) else "unknown"
    print(format_result_summary(summary, options["output_dir"]))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
