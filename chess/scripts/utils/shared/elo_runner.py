"""Shared orchestration helpers for Elo evaluation entrypoints."""

from pathlib import Path

import torch

from utils.shared.elo_estimator import estimate_model_elo
from utils.shared.model_catalog import persist_checkpoint_elo_metadata


ADAPTIVE_OVERRIDE_KEYS = (
    "adaptive_probe_games_per_level",
    "adaptive_min_batch_games",
    "adaptive_focus_games_per_level",
    "adaptive_extra_games_per_level",
    "adaptive_target_focus_levels",
    "adaptive_max_total_games",
    "adaptive_target_standard_error",
    "adaptive_min_games_for_se_stop",
)


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mode_key(use_mcts: bool) -> str:
    return "mcts" if use_mcts else "nn"


def mode_label(use_mcts: bool, simulations: int | None = None) -> str:
    if use_mcts:
        sims = safe_int(simulations) or 0
        return f"MCTS ({sims} sims)" if sims > 0 else "MCTS"
    return "Raw NN"


def resolve_eval_workers(requested_workers, use_mcts: bool, elo_cfg: dict) -> int:
    requested = safe_int(requested_workers)
    if requested is None:
        requested = 0
    if requested > 0:
        return requested
    key = "mcts_eval_workers" if use_mcts else "nn_eval_workers"
    mode_workers = safe_int((elo_cfg or {}).get(key))
    if mode_workers is None:
        return 0
    return max(0, int(mode_workers))


def _apply_mode_adaptive_overrides(runtime_cfg: dict, use_mcts: bool):
    prefix = "mcts_eval" if use_mcts else "nn_eval"
    for key in ADAPTIVE_OVERRIDE_KEYS:
        mode_override = f"{prefix}_{key}"
        if mode_override in runtime_cfg:
            runtime_cfg[key] = runtime_cfg[mode_override]


def build_eval_elo_config(
    elo_cfg: dict,
    *,
    levels=None,
    games_per_level=None,
    use_mcts: bool,
    simulations=None,
    sf_time=None,
    max_moves=None,
    sf_path=None,
    workers=None,
    standalone: bool = False,
) -> dict:
    """Build a mode-specific runtime Elo config for NN or MCTS checks."""
    source = dict(elo_cfg or {})
    runtime_cfg = dict(source)
    _apply_mode_adaptive_overrides(runtime_cfg, use_mcts)

    if levels is not None:
        runtime_cfg["levels"] = list(levels)
    if games_per_level is not None:
        runtime_cfg["games_per_level"] = int(games_per_level)
    if sf_time is not None:
        runtime_cfg["stockfish_time_limit"] = float(sf_time)
    if max_moves is not None:
        runtime_cfg["max_moves"] = int(max_moves)
    if sf_path is not None:
        runtime_cfg["stockfish_path"] = str(sf_path)

    if simulations is None and use_mcts:
        simulations = source.get("mcts_eval_simulations", source.get("mcts_simulations", 0))
    elif simulations is None:
        simulations = source.get("mcts_simulations", 0)

    runtime_cfg["enabled"] = True
    runtime_cfg["use_mcts"] = bool(use_mcts)
    runtime_cfg["mcts_simulations"] = int(simulations or 0)
    runtime_cfg["workers"] = int(resolve_eval_workers(workers, use_mcts, source))
    runtime_cfg["batch_model_moves"] = bool(source.get("batch_model_moves", True))
    runtime_cfg["batch_raw_model_moves"] = bool(source.get("batch_raw_model_moves", False))
    runtime_cfg["stockfish_hide_window"] = bool(source.get("stockfish_hide_window", True))

    if standalone:
        runtime_cfg["stockfish_priority"] = str(source.get("eval_elo_stockfish_priority", "normal"))
        runtime_cfg["prioritize_training"] = False
        runtime_cfg["reserve_dataloader_workers"] = False
        runtime_cfg["free_threads_utilization"] = 1.0
        runtime_cfg["auto_worker_reserve_cpus"] = 0
        runtime_cfg["progress_bar"] = "always"

    return runtime_cfg


def build_il_periodic_elo_config(elo_cfg: dict) -> dict:
    """IL epoch checks use Raw NN settings and keep training-friendly CPU limits."""
    resolved = build_eval_elo_config(
        elo_cfg or {},
        use_mcts=False,
        simulations=(elo_cfg or {}).get("mcts_simulations", 0),
        workers=(elo_cfg or {}).get("nn_eval_workers"),
        standalone=False,
    )
    if "il_eval_every" in (elo_cfg or {}):
        resolved["eval_every"] = (elo_cfg or {})["il_eval_every"]
    if "nn_eval_free_threads_utilization" in (elo_cfg or {}):
        resolved["free_threads_utilization"] = (elo_cfg or {})["nn_eval_free_threads_utilization"]
    return resolved


def build_il_final_elo_config(elo_cfg: dict, *, use_mcts: bool = True) -> dict:
    """Final IL checks run after training, so they can use standalone/full CPU settings."""
    source = dict(elo_cfg or {})
    simulations = source.get("mcts_eval_simulations", source.get("mcts_simulations", 0))
    workers = source.get("mcts_eval_workers" if use_mcts else "nn_eval_workers", source.get("workers", 0))
    resolved = build_eval_elo_config(
        source,
        use_mcts=use_mcts,
        simulations=simulations,
        workers=workers,
        standalone=True,
    )
    resolved["stockfish_priority"] = "normal"
    return resolved


def build_il_final_elo_configs(elo_cfg: dict) -> tuple[dict, dict]:
    return (
        build_il_final_elo_config(elo_cfg, use_mcts=False),
        build_il_final_elo_config(elo_cfg, use_mcts=True),
    )


def run_elo_check(model, config: dict, device, elo_config: dict, *, stop_event=None) -> dict:
    return estimate_model_elo(model, config, device, elo_config, stop_event=stop_event)


def persist_estimated_elo(
    checkpoint_path: Path,
    estimated_elo,
    *,
    levels: list[int],
    games_per_level: int,
    use_mcts: bool,
    simulations: int,
    sf_time: float,
    source: str,
    elo_result: dict | None = None,
    print_prefix: str = "  ",
    announce_success: bool = True,
):
    """Persist Elo into checkpoint metadata and print a concise update."""
    elo_value = safe_float(estimated_elo)
    if elo_value is None or checkpoint_path is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"{print_prefix}! Could not open checkpoint for Elo persist: {exc}")
        return
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return

    existing_key = "estimated_elo_mcts" if use_mcts else "estimated_elo_nn"
    existing_elo = safe_float(checkpoint.get(existing_key))
    ok, error = persist_checkpoint_elo_metadata(
        checkpoint_path,
        elo_value,
        levels=levels,
        games_per_level=games_per_level,
        use_mcts=use_mcts,
        simulations=simulations,
        sf_time=sf_time,
        source=source,
        elo_result=elo_result,
    )
    if ok and announce_success:
        label = "MCTS Elo" if use_mcts else "NN Elo"
        if existing_elo is not None:
            print(f"{print_prefix}✓ Updated {label}: {int(round(existing_elo))} -> {int(round(elo_value))}")
        else:
            print(f"{print_prefix}✓ Saved {label} into checkpoint metadata: {int(round(elo_value))}")
    elif error:
        print(f"{print_prefix}! {error}")
