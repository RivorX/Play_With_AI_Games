"""
Run a replay ablation with an evolving RL learner as the self-play generator.

The accumulated_replay branch is the feedback driver:

1. generate MCTS games with the current driver model,
2. add the same new positions to every ablation branch,
3. train every branch with its own replay strategy,
4. evaluate all branches against the frozen IL anchor,
5. use the updated accumulated_replay model to generate the next round.

This keeps the normal RL feedback loop while preserving a controlled, shared
dataset for replay comparisons.
"""

import argparse
import copy
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = SCRIPTS_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

PRESETS = {
    "quick": {
        "rounds": 2,
        "games_per_round": 16,
        "eval_games": 40,
    },
    "standard": {
        "rounds": 3,
        "games_per_round": 48,
        "eval_games": 100,
    },
    "full": {
        "rounds": 5,
        "games_per_round": 100,
        "eval_games": 200,
    },
}

VARIANT_FRESH = "fresh_only"
VARIANT_ACCUMULATED = "accumulated_replay"
VARIANT_LEGACY = "legacy_full_pass"
VARIANT_ORDER = [
    VARIANT_FRESH,
    VARIANT_ACCUMULATED,
    VARIANT_LEGACY,
]


def _round_up(value, quantum):
    quantum = max(1, int(quantum))
    return int(math.ceil(float(value) / float(quantum)) * quantum)


def _resolve_target_steps(rl_cfg):
    raw_value = rl_cfg.get("train_target_steps_per_iteration", "auto")
    if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
        return max(1, int(round(float(rl_cfg.get("replay_buffer_multiplier", 1) or 1))))
    return max(1, int(raw_value))


def _resolve_dynamic_batch_size(replay_size, rl_cfg):
    configured_batch = max(1, int(rl_cfg.get("batch_size", 1024)))
    if not bool(rl_cfg.get("train_dynamic_batch_size_enabled", False)):
        return configured_batch
    min_batch = max(1, int(rl_cfg.get("train_batch_size_min", configured_batch)))
    max_batch = max(min_batch, int(rl_cfg.get("train_batch_size_max", configured_batch)))
    round_to = max(1, int(rl_cfg.get("train_batch_size_round_to", 1)))
    target_batch = _round_up(
        math.ceil(float(max(1, replay_size)) / float(_resolve_target_steps(rl_cfg))),
        round_to,
    )
    return max(min_batch, min(max_batch, target_batch))


def _default_checkpoint(config):
    best_il = PROJECT_DIR / config["paths"]["best_model_il"]
    best_il_swa = best_il.parent / "best_model_il_swa.pt"
    return best_il_swa if best_il_swa.exists() else best_il


def _seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_generation_config(config, args):
    generation_config = copy.deepcopy(config)
    rl_cfg = generation_config.setdefault("reinforcement_learning", {})
    rl_cfg["self_play_opponent_pool_enabled"] = False
    rl_cfg["self_play_stream_to_replay"] = False
    if args.workers is not None:
        rl_cfg["self_play_workers"] = max(1, int(args.workers))
    if args.no_central_inference:
        rl_cfg["self_play_central_inference_enabled"] = False
    return generation_config


def _build_eval_config(config, args):
    eval_config = copy.deepcopy(config)
    rl_cfg = eval_config.setdefault("reinforcement_learning", {})
    sims = int(args.eval_sims or rl_cfg.get("mcts_simulations", 160))
    rl_cfg["mcts_simulations"] = max(1, sims)
    rl_cfg["eval_mcts_simulations_multiplier"] = 1.0
    rl_cfg["eval_fixed_openings_enabled"] = True
    rl_cfg["eval_fixed_openings_pair_games"] = True
    if args.eval_workers is not None:
        rl_cfg["eval_workers"] = max(1, int(args.eval_workers))
    if args.no_central_inference:
        rl_cfg["eval_central_inference_enabled"] = False
    return eval_config


def _make_replay_buffer(config, capacity):
    from utils.rl.replay import ReplayBuffer

    rl_cfg = config["reinforcement_learning"]
    return ReplayBuffer(
        max(1, int(capacity)),
        max_policy_targets=int(rl_cfg.get("replay_max_policy_targets", 256)),
        use_fp16=bool(rl_cfg.get("replay_fp16", False)),
        decisive_sampling_fraction=float(rl_cfg.get("replay_decisive_sampling_fraction", 0.0)),
        decisive_value_epsilon=float(rl_cfg.get("replay_decisive_value_epsilon", 0.05)),
        hard_negative_sampling_fraction=float(rl_cfg.get("replay_hard_negative_sampling_fraction", 0.0)),
        hard_negative_min_importance=float(rl_cfg.get("replay_hard_negative_min_importance", 0.0)),
        recent_sampling_fraction=float(rl_cfg.get("replay_recent_sampling_fraction", 0.0)),
        recent_window_fraction=float(rl_cfg.get("replay_recent_window_fraction", 0.25)),
        quality_sampling_fraction=float(rl_cfg.get("replay_quality_sampling_fraction", 0.0)),
        quality_min_importance=float(rl_cfg.get("replay_quality_min_importance", 0.0)),
        quality_value_bonus=float(rl_cfg.get("replay_quality_value_bonus", 0.0)),
        value_balanced_sampling_fraction=float(rl_cfg.get("replay_value_balanced_sampling_fraction", 0.0)),
        value_balance_epsilon=float(rl_cfg.get("replay_value_balance_epsilon", 0.05)),
        weighted_sampling_power=float(rl_cfg.get("replay_weighted_sampling_power", 1.0)),
        resize_preserve_decisive_fraction=float(rl_cfg.get("replay_resize_preserve_decisive_fraction", 0.0)),
        resize_preserve_decisive_min_count=int(rl_cfg.get("replay_resize_preserve_decisive_min_count", 0)),
    )


def _clone_positions(positions):
    cloned = []
    for position in positions:
        items = []
        for item in position:
            items.append(item.clone() if torch.is_tensor(item) else item)
        cloned.append(tuple(items))
    return cloned


def _summarize_positions(positions):
    values = []
    importance = []
    policy_weights = []
    value_weights = []
    policy_support = []
    for position in positions:
        try:
            values.append(float(position[3].reshape(-1)[0].item()))
            policy_support.append(int(position[1].numel()))
            importance.append(float(position[4]) if len(position) > 4 else 0.0)
            policy_weights.append(float(position[5]) if len(position) > 5 else 1.0)
            value_weights.append(float(position[6]) if len(position) > 6 else 1.0)
        except Exception:
            continue
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "positions": len(positions),
        "draw_target_fraction": float(np.mean(np.abs(values_np) <= 0.05)) if values_np.size else 0.0,
        "value_std": float(np.std(values_np)) if values_np.size else 0.0,
        "importance_avg": float(np.mean(importance)) if importance else 0.0,
        "policy_weight_avg": float(np.mean(policy_weights)) if policy_weights else 0.0,
        "value_weight_avg": float(np.mean(value_weights)) if value_weights else 0.0,
        "policy_support_avg": float(np.mean(policy_support)) if policy_support else 0.0,
    }


def _build_optimizer(model, config, learning_rate):
    from train_rl import _build_rl_optimizer_param_groups

    param_groups, _summaries = _build_rl_optimizer_param_groups(
        model,
        config["reinforcement_learning"],
        learning_rate,
    )
    return optim.AdamW(
        param_groups,
        lr=learning_rate,
        fused=True if torch.cuda.is_available() else False,
    )


def _train_variant_round(
    model,
    optimizer,
    scaler,
    replay_buffer,
    config,
    variant,
    legacy_batch_size,
    anchor_model=None,
):
    from utils.rl.training_rl import train_on_batch_rl
    from utils.shared.metrics import MetricsCalculator

    replay_size = len(replay_buffer)
    if replay_size <= 0:
        raise ValueError("Cannot train an ablation variant on an empty replay buffer.")
    if variant == VARIANT_LEGACY:
        batch_size = min(replay_size, max(1, int(legacy_batch_size)))
    else:
        batch_size = min(replay_size, _resolve_dynamic_batch_size(replay_size, config["reinforcement_learning"]))

    selected = replay_buffer.select_indices(replay_size)
    index_batches = [
        selected[start:start + batch_size]
        for start in range(0, int(selected.size), batch_size)
    ]
    metrics_calc = MetricsCalculator()
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "policy_entropy": 0.0,
        "value_pred_std": 0.0,
        "target_value_std": 0.0,
    }
    model.train()
    for indices in index_batches:
        batch = replay_buffer.sample_from_indices(indices)
        values = train_on_batch_rl(
            model,
            optimizer,
            batch,
            config,
            next(model.parameters()).device,
            scaler,
            metrics_calc,
            anchor_model=anchor_model,
        )
        for key, value in zip(totals, values):
            totals[key] += float(value)
    model.eval()
    count = max(1, len(index_batches))
    metrics = metrics_calc.compute()
    return {
        "replay_size": replay_size,
        "train_batch_size": batch_size,
        "train_steps": len(index_batches),
        "selected_samples": int(selected.size),
        **{key: value / count for key, value in totals.items()},
        "value_mae": float(metrics.get("value_mae", 0.0)),
        "policy_top1_acc": float(metrics.get("policy_top1_acc", 0.0)),
    }


def _score_ci95(stats):
    wins = int(stats.get("wins", 0))
    draws = int(stats.get("draws", 0))
    losses = int(stats.get("losses", 0))
    unresolved = int(stats.get("unresolved", 0))
    games = wins + draws + losses + unresolved
    if games <= 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / games
    second_moment = (wins + 0.25 * draws) / games
    variance = max(0.0, second_moment - score * score)
    margin = 1.96 * math.sqrt(variance / games)
    return max(0.0, score - margin), min(1.0, score + margin)


def _evaluate_pair(model1, model2, eval_config, device, games, label, round_idx):
    from utils.rl.training_rl import evaluate_models

    print(f"\nEval round {round_idx}: {label} ({games} paired-color games)")
    started = time.perf_counter()
    stats = evaluate_models(
        model1,
        model2,
        eval_config,
        device,
        num_games=games,
        use_fixed_openings=True,
    )
    elapsed_s = time.perf_counter() - started
    ci_low, ci_high = _score_ci95(stats)
    row = {
        "round": round_idx,
        "matchup": label,
        "games": games,
        "wins": int(stats.get("wins", 0)),
        "draws": int(stats.get("draws", 0)),
        "losses": int(stats.get("losses", 0)),
        "unresolved": int(stats.get("unresolved", 0)),
        "score_rate": float(stats.get("score_rate", 0.0)),
        "score_ci95_low": ci_low,
        "score_ci95_high": ci_high,
        "elapsed_s": elapsed_s,
    }
    print(
        f"  W/D/L={row['wins']}/{row['draws']}/{row['losses']} | "
        f"score={row['score_rate']:.2%} | CI95=[{ci_low:.2%}, {ci_high:.2%}]"
    )
    return row


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

FEEDBACK_PRESETS = {name: dict(values) for name, values in PRESETS.items()}
FEEDBACK_PRESETS["full"]["eval_games"] = 100


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Test replay strategies inside an evolving learner -> self-play -> learner RL loop."
    )
    parser.add_argument("--preset", choices=sorted(FEEDBACK_PRESETS), default="quick")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--games-per-round", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-sims", type=int, default=None)
    parser.add_argument("--legacy-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--no-central-inference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_feedback_report(path, dataset_path, generation_rows, train_rows, eval_rows):
    lines = [
        "RL REPLAY FEEDBACK AUDIT",
        f"Dataset: {dataset_path}",
        "",
        "Method",
        "- accumulated_replay is the feedback driver.",
        "- Round N+1 games are generated by the driver after training round N.",
        "- All branches train on the same newly generated positions.",
        "- The anchor stays frozen and is used as an evaluation baseline plus the configured policy-KL reference.",
        "",
        "Generation by evolving driver",
    ]
    for row in generation_rows:
        candidates = (
            int(row["positions"])
            + int(row["curriculum_dropped_positions"])
            + int(row["cap_dropped_positions"])
        )
        keep_rate = float(row["positions"]) / max(1, candidates)
        lines.append(
            f"- round {int(row['round'])}: positions={int(row['positions'])}, "
            f"keep={keep_rate:.1%}, draw_targets={float(row['draw_target_fraction']):.1%}, "
            f"value_std={float(row['value_std']):.3f}, "
            f"changed_top={float(row['mcts_prior_changed_rate']):.1%}, "
            f"KL={float(row['mcts_policy_kl_mean']):.3f}, "
            f"uptake={float(row['mcts_policy_uptake_weight_mean']):.2f}, "
            f"gated={float(row['mcts_policy_uptake_low_rate']):.1%}, "
            f"drops={int(row['curriculum_dropped_positions'])}+{int(row['cap_dropped_positions'])}"
        )

    lines.extend(["", "Training dose"])
    cumulative_steps = {variant: 0 for variant in VARIANT_ORDER}
    for row in train_rows:
        variant = str(row["variant"])
        cumulative_steps[variant] += int(row["train_steps"])
        lines.append(
            f"- round {int(row['round'])} {variant}: replay={int(row['replay_size'])}, "
            f"batch={int(row['train_batch_size'])}, steps={int(row['train_steps'])}, "
            f"cumulative_steps={cumulative_steps[variant]}, value_mae={float(row['value_mae']):.3f}"
        )

    lines.extend(["", "Evaluation vs frozen anchor"])
    for row in eval_rows:
        if not str(row["matchup"]).endswith("_vs_anchor"):
            continue
        lines.append(
            f"- round {int(row['round'])} {row['matchup']}: "
            f"score={float(row['score_rate']):.2%}, "
            f"CI95=[{float(row['score_ci95_low']):.2%}, {float(row['score_ci95_high']):.2%}], "
            f"W/D/L={int(row['wins'])}/{int(row['draws'])}/{int(row['losses'])}"
        )

    lines.extend([
        "",
        "Direct replay comparisons",
    ])
    for row in eval_rows:
        if str(row["matchup"]).endswith("_vs_anchor"):
            continue
        lines.append(
            f"- round {int(row['round'])} {row['matchup']}: "
            f"score={float(row['score_rate']):.2%}, "
            f"CI95=[{float(row['score_ci95_low']):.2%}, {float(row['score_ci95_high']):.2%}]"
        )
    lines.extend([
        "",
        "Reading guide",
        "- accumulated_replay below fresh_only: retained older positions may be harmful.",
        "- legacy_full_pass below accumulated_replay: growing optimizer-step dose is harmful.",
        "- all branches falling together: inspect generated MCTS targets, storage shaping, LR, or the feedback loop itself.",
        "- accumulated_replay is the production-like driver; its round-by-round anchor score is the main trajectory.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_dataset(path, checkpoint, generation_rows, rounds):
    payload = {
        "format": "rl_replay_feedback_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "feedback_driver": VARIANT_ACCUMULATED,
        "generation_rows": generation_rows,
        "rounds": rounds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _print_plan(args, checkpoint, rounds, games_per_round, eval_games, eval_sims):
    print("RL replay feedback audit")
    print(f"Preset: {args.preset}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Feedback driver: {VARIANT_ACCUMULATED}")
    print(f"Loop: generate -> shared replay update -> train -> eval, repeated {rounds} time(s)")
    print(f"Generate per round: {games_per_round} evolving-driver MCTS games")
    print(f"Variants: {', '.join(VARIANT_ORDER)}")
    print(f"Eval every round: {eval_games} paired-color games, {eval_sims} sims")
    print(f"Legacy fixed batch: {max(1, int(args.legacy_batch_size))}")


def main():
    args = _build_parser().parse_args()
    with open(args.config, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    preset = FEEDBACK_PRESETS[args.preset]
    rounds = max(1, int(args.rounds or preset["rounds"]))
    games_per_round = max(2, int(args.games_per_round or preset["games_per_round"]))
    eval_games = max(2, int(args.eval_games or preset["eval_games"]))
    if eval_games % 2:
        eval_games += 1
    eval_sims = max(1, int(args.eval_sims or config["reinforcement_learning"].get("mcts_simulations", 160)))
    checkpoint = (args.checkpoint or _default_checkpoint(config)).resolve()
    _print_plan(args, checkpoint, rounds, games_per_round, eval_games, eval_sims)
    if args.dry_run:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    device_name = args.device or config.get("hardware", {}).get("device", "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    config = copy.deepcopy(config)
    config.setdefault("model", {})["print_summary"] = False
    config.setdefault("hardware", {})["device"] = str(device)
    learning_rate = float(args.learning_rate or config["reinforcement_learning"]["learning_rate"])
    config["reinforcement_learning"]["learning_rate"] = learning_rate
    generation_config = _build_generation_config(config, args)
    eval_config = _build_eval_config(config, args)

    from src.model import load_model
    from train_rl import play_games_parallel_mcts
    from utils.rl.persistent_pool import _shutdown_selfplay_pool

    print(f"\nLoading frozen anchor: {checkpoint}")
    anchor_model = load_model(str(checkpoint), config, device, strict=True)
    anchor_model.eval()
    anchor_model.requires_grad_(False)
    variants = {}
    for variant in VARIANT_ORDER:
        model = load_model(str(checkpoint), config, device, strict=True)
        model.eval()
        variants[variant] = {
            "model": model,
            "optimizer": _build_optimizer(model, config, learning_rate),
            "scaler": torch.amp.GradScaler("cuda", enabled=bool(config["hardware"].get("use_amp", True))),
            "replay": None,
        }

    output_dir = (args.output_dir or (PROJECT_DIR / config["paths"]["logs_dir"] / "csv")).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = (
        PROJECT_DIR
        / config["paths"]["logs_dir"]
        / "replay_ablation"
        / f"replay_feedback_dataset_{timestamp}.pt"
    ).resolve()
    generated_rounds = []
    generation_rows = []
    train_rows = []
    eval_rows = []
    cumulative_positions = 0

    try:
        for round_idx in range(1, rounds + 1):
            print(f"\n=== Feedback round {round_idx}/{rounds}: generate with updated {VARIANT_ACCUMULATED} ===")
            _seed_everything(int(config.get("seed", 42)) + round_idx)
            generation_config["reinforcement_learning"]["current_iteration"] = round_idx
            result = play_games_parallel_mcts(
                variants[VARIANT_ACCUMULATED]["model"],
                generation_config,
                device,
                games_per_round,
                replay_buffer=None,
            )
            positions, positions_added, avg_length, positions_per_sec, selfplay_time, collection_time, stats = result
            positions = _clone_positions(positions)
            generated_rounds.append(positions)
            cumulative_positions += len(positions)
            summary = _summarize_positions(positions)
            generation_row = {
                "round": round_idx,
                **summary,
                "reported_positions_added": int(positions_added),
                "cumulative_positions": cumulative_positions,
                "avg_game_length": float(avg_length),
                "positions_per_sec": float(positions_per_sec),
                "selfplay_time_s": float(selfplay_time),
                "collection_time_s": float(collection_time),
                "completed_games": int((stats or {}).get("completed_games", 0)),
                "completed_draw_rate": float((stats or {}).get("completed_draw_rate", 0.0)),
                "decisive_rate": float((stats or {}).get("decisive_rate", 0.0)),
                "curriculum_dropped_positions": int((stats or {}).get("curriculum_dropped_positions", 0)),
                "cap_dropped_positions": int((stats or {}).get("cap_dropped_positions", 0)),
                "mcts_prior_changed_rate": float((stats or {}).get("mcts_prior_changed_rate", 0.0)),
                "mcts_policy_kl_mean": float((stats or {}).get("mcts_policy_kl_mean", 0.0)),
                "mcts_policy_uptake_weight_mean": float((stats or {}).get("mcts_policy_uptake_weight_mean", 1.0)),
                "mcts_policy_uptake_low_rate": float((stats or {}).get("mcts_policy_uptake_low_rate", 0.0)),
                "mcts_q_delta_mean": float((stats or {}).get("mcts_q_delta_mean", 0.0)),
            }
            generation_rows.append(generation_row)
            _save_dataset(dataset_path, checkpoint, generation_rows, generated_rounds)

            print(f"\n=== Feedback round {round_idx}/{rounds}: train shared positions ===")
            for variant in VARIANT_ORDER:
                state = variants[variant]
                if variant == VARIANT_FRESH:
                    replay_buffer = _make_replay_buffer(config, len(positions))
                    state["replay"] = replay_buffer
                else:
                    if state["replay"] is None:
                        state["replay"] = _make_replay_buffer(config, cumulative_positions)
                    elif state["replay"].max_size < cumulative_positions:
                        state["replay"].resize(cumulative_positions)
                    replay_buffer = state["replay"]
                replay_buffer.set_current_iteration(round_idx)
                for position in positions:
                    replay_buffer.add(position)
                _seed_everything(int(config.get("seed", 42)) + round_idx * 100)
                started = time.perf_counter()
                train_stats = _train_variant_round(
                    state["model"],
                    state["optimizer"],
                    state["scaler"],
                    replay_buffer,
                    config,
                    variant,
                    args.legacy_batch_size,
                    anchor_model=anchor_model,
                )
                row = {
                    "round": round_idx,
                    "variant": variant,
                    "new_positions": len(positions),
                    **train_stats,
                    "elapsed_s": time.perf_counter() - started,
                }
                train_rows.append(row)
                print(
                    f"  {variant:<20} replay={row['replay_size']:>6}, "
                    f"batch={row['train_batch_size']:>5}, steps={row['train_steps']:>3}, "
                    f"loss={row['loss']:.4f}, value_mae={row['value_mae']:.4f}"
                )

            for variant in VARIANT_ORDER:
                eval_rows.append(_evaluate_pair(
                    variants[variant]["model"],
                    anchor_model,
                    eval_config,
                    device,
                    eval_games,
                    f"{variant}_vs_anchor",
                    round_idx,
                ))
            eval_rows.append(_evaluate_pair(
                variants[VARIANT_ACCUMULATED]["model"],
                variants[VARIANT_FRESH]["model"],
                eval_config,
                device,
                eval_games,
                f"{VARIANT_ACCUMULATED}_vs_{VARIANT_FRESH}",
                round_idx,
            ))
            eval_rows.append(_evaluate_pair(
                variants[VARIANT_LEGACY]["model"],
                variants[VARIANT_ACCUMULATED]["model"],
                eval_config,
                device,
                eval_games,
                f"{VARIANT_LEGACY}_vs_{VARIANT_ACCUMULATED}",
                round_idx,
            ))
    finally:
        _shutdown_selfplay_pool()

    train_csv_path = output_dir / f"rl_replay_feedback_{timestamp}_train.csv"
    eval_csv_path = output_dir / f"rl_replay_feedback_{timestamp}_eval.csv"
    generation_csv_path = output_dir / f"rl_replay_feedback_{timestamp}_generation.csv"
    report_path = output_dir / f"rl_replay_feedback_{timestamp}.txt"
    _write_csv(train_csv_path, train_rows)
    _write_csv(eval_csv_path, eval_rows)
    _write_csv(generation_csv_path, generation_rows)
    _write_feedback_report(report_path, dataset_path, generation_rows, train_rows, eval_rows)
    print(f"\nSaved feedback train detail: {train_csv_path}")
    print(f"Saved feedback eval detail: {eval_csv_path}")
    print(f"Saved feedback generation detail: {generation_csv_path}")
    print(f"Saved feedback report: {report_path}")
    print(f"Saved feedback dataset: {dataset_path}")


if __name__ == "__main__":
    main()
