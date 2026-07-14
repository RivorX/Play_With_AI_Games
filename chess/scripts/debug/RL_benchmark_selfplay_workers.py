"""Benchmark RL self-play throughput across central-inference worker counts."""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml


scripts_dir = Path(__file__).resolve().parent.parent
project_dir = scripts_dir.parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(scripts_dir))

from src.model import load_model
from src.utils.config import normalize_config
from train_rl import play_games_parallel_mcts
from utils.rl.persistent_pool import _shutdown_selfplay_pool


class _DiscardReplay:
    """Enable production queue transport without retaining benchmark positions."""

    def add_packed_batch(self, *_args, **_kwargs):
        return None

    def add(self, *_args, **_kwargs):
        return None


def _parse_workers(raw_value):
    workers = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if token:
            workers.append(max(1, int(token)))
    workers = list(dict.fromkeys(workers))
    if not workers:
        raise ValueError("Provide at least one worker count.")
    return workers


def _shared_cpu_state(model):
    result = {}
    for key, value in model.state_dict().items():
        tensor = value.detach().to("cpu").contiguous().clone()
        tensor.share_memory_()
        result[key] = tensor
    return result


def _profile_value(profile, key):
    try:
        return float((profile or {}).get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_trial_config(base_config, workers, max_moves, central_max_batch=None, flush_ms=None):
    config = copy.deepcopy(base_config)
    rl_cfg = config["reinforcement_learning"]
    rl_cfg["self_play_workers"] = int(workers)
    rl_cfg["self_play_worker_cap_multiplier"] = max(
        float(rl_cfg.get("self_play_worker_cap_multiplier", 1.0) or 1.0),
        float(workers) / float(max(1, __import__("os").cpu_count() or 1)) + 0.1,
    )
    rl_cfg["self_play_max_moves"] = max(4, int(max_moves))
    rl_cfg["self_play_progress_interval_games"] = max(1, int(workers))
    rl_cfg["persistent_self_play_workers"] = True
    rl_cfg["self_play_stream_to_replay"] = True
    rl_cfg["self_play_queue_transport"] = True
    rl_cfg["self_play_dynamic_dispatch"] = True
    rl_cfg["self_play_central_inference_enabled"] = True
    central_cfg = config.setdefault("central_inference", {})
    if central_max_batch is not None:
        central_cfg["max_batch_size"] = max(1, int(central_max_batch))
    if flush_ms is not None:
        central_cfg["flush_ms"] = max(0.0, float(flush_ms))
    return config


def _run_trial(model, best_state, config, device, workers, games, repeat, phase):
    result = play_games_parallel_mcts(
        model,
        config,
        device,
        int(games),
        replay_buffer=_DiscardReplay(),
        best_model_state=best_state,
    )
    _, kept_positions, avg_length, kept_positions_per_sec, elapsed_s, _, stats = result
    profile = dict((stats or {}).get("profile", {}) or {})
    played_positions = int(round(float(avg_length) * float(games)))
    played_positions_per_sec = float(
        (stats or {}).get("played_positions_per_sec", 0.0) or 0.0
    )
    if played_positions_per_sec <= 0.0 and elapsed_s > 0.0:
        played_positions_per_sec = float(played_positions) / float(elapsed_s)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": str(phase),
        "workers": int(workers),
        "central_max_batch": int(config.get("central_inference", {}).get("max_batch_size", 0) or 0),
        "flush_ms": float(config.get("central_inference", {}).get("flush_ms", 0.0) or 0.0),
        "repeat": int(repeat),
        "games": int(games),
        "positions": int(played_positions),
        "kept_positions": int(kept_positions),
        "avg_game_length": float(avg_length),
        "elapsed_s": float(elapsed_s),
        "played_positions_per_sec": float(played_positions_per_sec),
        "replay_positions_per_sec": float(kept_positions_per_sec),
        "mcts_simulations_per_sec": float(
            (stats or {}).get("mcts_simulations_per_sec", 0.0) or 0.0
        ),
        "mcts_nn_evaluations_per_sec": float(
            (stats or {}).get("mcts_nn_evaluations_per_sec", 0.0) or 0.0
        ),
        "mcts_selection_node_traversals_per_sec": float(
            (stats or {}).get("mcts_selection_node_traversals_per_sec", 0.0) or 0.0
        ),
        "worker_batch": _profile_value(profile, "average_batch_size"),
        "central_batch": _profile_value(profile, "central_average_batch_size"),
        "remote_wait_ms": _profile_value(profile, "central_remote_wait_ms_per_request"),
        "queue_wait_ms": _profile_value(profile, "central_server_queue_wait_ms_per_request"),
        "forward_ms": _profile_value(profile, "central_server_forward_ms_per_request"),
    }


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="6,10,14,18,20")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--warmup-games", type=int, default=0,
                        help="Warm-up games; 0 uses the measured game count for identical pool topology.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-moves", type=int, default=64)
    parser.add_argument("--central-max-batch", type=int, default=None)
    parser.add_argument("--flush-ms", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(project_dir / "config" / "config.yaml", "r", encoding="utf-8") as file_obj:
        base_config = normalize_config(yaml.safe_load(file_obj) or {})
    base_config.setdefault("model", {})["print_summary"] = False
    checkpoint = args.checkpoint or project_dir / "models" / "RL" / "v9.9b_best.pt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device(base_config.get("hardware", {}).get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    worker_counts = _parse_workers(args.workers)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = project_dir / "logs" / "csv" / f"selfplay_worker_benchmark_{stamp}.csv"
    output = output.resolve()

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device} | workers: {worker_counts} | measured games: {args.games}")
    model = load_model(str(checkpoint), base_config, device, strict=True)
    model.eval()
    best_state = _shared_cpu_state(model)
    rows = []
    try:
        for workers in worker_counts:
            config = _build_trial_config(
                base_config,
                workers,
                args.max_moves,
                central_max_batch=args.central_max_batch,
                flush_ms=args.flush_ms,
            )
            warmup_games = max(workers, int(args.warmup_games or args.games))
            print(f"\n[{workers} workers] warm-up: {warmup_games} games")
            warmup = _run_trial(model, best_state, config, device, workers, warmup_games, 0, "warmup")
            print(
                f"  warm-up played/replay {warmup['played_positions_per_sec']:.1f}/"
                f"{warmup['replay_positions_per_sec']:.1f} pos/s | "
                f"batch {warmup['central_batch']:.1f} | wait {warmup['remote_wait_ms']:.2f} ms"
            )
            for repeat in range(1, max(1, int(args.repeats)) + 1):
                print(f"[{workers} workers] measured {repeat}/{max(1, int(args.repeats))}: {args.games} games")
                row = _run_trial(model, best_state, config, device, workers, args.games, repeat, "measured")
                rows.append(row)
                _write_rows(output, rows)
                print(
                    f"  played/replay {row['played_positions_per_sec']:.2f}/"
                    f"{row['replay_positions_per_sec']:.2f} pos/s | "
                    f"{row['mcts_simulations_per_sec']:.0f} completed visits/s | "
                    f"{row['mcts_selection_node_traversals_per_sec']:.0f} selection traversals/s | "
                    f"worker/central batch {row['worker_batch']:.1f}/{row['central_batch']:.1f} | "
                    f"wait queue/fwd {row['remote_wait_ms']:.2f}/"
                    f"{row['queue_wait_ms']:.2f}/{row['forward_ms']:.2f} ms"
                )
    finally:
        _shutdown_selfplay_pool()

    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["workers"]), []).append(float(row["played_positions_per_sec"]))
    ranking = sorted(
        ((statistics.mean(values), workers, values) for workers, values in grouped.items()),
        reverse=True,
    )
    print("\nWorker ranking (played positions/s):")
    for mean_pos_s, workers, values in ranking:
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(f"  {workers:>2} workers: {mean_pos_s:.2f} pos/s (sd {spread:.2f})")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
