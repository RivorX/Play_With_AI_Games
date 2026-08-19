"""Shared orchestration helpers for Elo evaluation entrypoints."""

import math
from pathlib import Path

import torch

from src.evaluation.elo_estimator import estimate_model_elo
from src.models.catalog import persist_checkpoint_elo_metadata


ADAPTIVE_OVERRIDE_KEYS = (
    "adaptive_probe_games_per_level",
    "adaptive_min_batch_games",
    "adaptive_focus_games_per_level",
    "adaptive_extra_games_per_level",
    "adaptive_target_focus_levels",
    "adaptive_max_total_games",
    "adaptive_target_standard_error",
    "adaptive_min_games_for_se_stop",
    "adaptive_hard_max_total_games",
    "adaptive_budget_extension_games",
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


def elo_protocol_matches_settings(
    settings: dict | None,
    runtime_cfg: dict,
    *,
    use_mcts: bool,
    simulations: int = 0,
) -> bool:
    """Whether a stored rating used the same strength-defining protocol."""
    if not isinstance(settings, dict):
        return False
    try:
        stored_time = float(settings["stockfish_time_limit"])
        active_time = float((runtime_cfg or {}).get("stockfish_time_limit", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isclose(stored_time, active_time, rel_tol=0.0, abs_tol=1e-9):
        return False
    if bool(settings.get("use_mcts", False)) != bool(use_mcts):
        return False
    if use_mcts and safe_int(settings.get("simulations")) != max(0, int(simulations or 0)):
        return False
    return True


def checkpoint_elo_seed(
    checkpoint_path,
    *,
    use_mcts: bool,
    simulations: int = 0,
) -> tuple[float | None, str | None]:
    """Read an unbiased scheduling seed from checkpoint Elo metadata.

    The seed only determines which Stockfish levels are tested first. It is
    never included as a synthetic game or prior in the final Elo fit.
    """
    if checkpoint_path is None:
        return None, None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    if not isinstance(checkpoint, dict):
        return None, None

    if not use_mcts:
        for key in ("estimated_elo_nn", "last_estimated_elo_nn", "estimated_elo", "last_estimated_elo"):
            elo = safe_float(checkpoint.get(key))
            if elo is not None and math.isfinite(elo):
                return float(elo), f"{checkpoint_path.name} previous raw NN Elo"
        return None, None

    requested_sims = max(0, int(simulations or 0))
    candidates: list[tuple[int, float]] = []
    by_simulations = checkpoint.get("estimated_elo_mcts_by_simulations")
    if isinstance(by_simulations, dict):
        for raw_sims, raw_entry in by_simulations.items():
            sims = safe_int(raw_sims)
            if sims is None and isinstance(raw_entry, dict):
                sims = safe_int(raw_entry.get("simulations"))
            elo = safe_float(raw_entry.get("elo")) if isinstance(raw_entry, dict) else safe_float(raw_entry)
            if sims is not None and elo is not None and math.isfinite(elo):
                candidates.append((max(0, int(sims)), float(elo)))

    if candidates:
        exact = next((item for item in candidates if item[0] == requested_sims), None)
        chosen = exact
        if chosen is None:
            chosen = min(
                candidates,
                key=lambda item: abs(math.log2((item[0] + 1.0) / (requested_sims + 1.0))),
            )
        sims, elo = chosen
        relation = "same" if sims == requested_sims else "nearest"
        return elo, f"{checkpoint_path.name} previous MCTS Elo @{sims} ({relation} budget)"

    elo = safe_float(checkpoint.get("estimated_elo_mcts", checkpoint.get("last_estimated_elo_mcts")))
    if elo is not None and math.isfinite(elo):
        stored_sims = safe_int(checkpoint.get("estimated_elo_mcts_simulations"))
        suffix = f" @{stored_sims}" if stored_sims is not None else ""
        return float(elo), f"{checkpoint_path.name} previous MCTS Elo{suffix}"
    return None, None


def apply_checkpoint_elo_seed(
    runtime_cfg: dict,
    checkpoint_path,
    *,
    use_mcts: bool,
    simulations: int = 0,
) -> float | None:
    """Attach checkpoint Elo as an opponent-selection seed when none is set."""
    existing = safe_float((runtime_cfg or {}).get("adaptive_initial_elo"))
    if existing is not None and math.isfinite(existing):
        return float(existing)
    elo, source = checkpoint_elo_seed(
        checkpoint_path,
        use_mcts=use_mcts,
        simulations=simulations,
    )
    if elo is None:
        return None
    runtime_cfg["adaptive_initial_elo"] = float(elo)
    runtime_cfg["adaptive_initial_elo_source"] = str(source or "previous checkpoint Elo")
    return float(elo)


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
        focus_games = max(1, int(games_per_level))
        runtime_cfg["games_per_level"] = focus_games
        runtime_cfg["adaptive_focus_games_per_level"] = max(
            int(runtime_cfg.get("adaptive_probe_games_per_level", 1) or 1),
            focus_games,
        )
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


def build_final_elo_config(elo_cfg: dict, *, use_mcts: bool = True) -> dict:
    """Build the shared full-strength Elo config used by IL and RL."""
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


def build_final_mcts_elo_configs(elo_cfg: dict) -> list[dict]:
    """Build final-only MCTS budgets as multiples of the standard Elo budget."""
    source = dict(elo_cfg or {})
    default_simulations = max(
        1,
        int(source.get("mcts_eval_simulations", source.get("mcts_simulations", 1)) or 1),
    )
    raw_profile = source.get("final_mcts_profile_multipliers")
    if not isinstance(raw_profile, (list, tuple, set)):
        raw_profile = (1.0,)

    simulations = []
    for multiplier in raw_profile:
        try:
            multiplier = float(multiplier)
            if not math.isfinite(multiplier) or multiplier <= 0.0:
                continue
            value = max(1, int(round(default_simulations * multiplier)))
        except (TypeError, ValueError, OverflowError):
            continue
        if value not in simulations:
            simulations.append(value)
    if default_simulations not in simulations:
        simulations.append(default_simulations)
    simulations.sort()

    configs = []
    for simulation_count in simulations:
        resolved = build_final_elo_config(source, use_mcts=True)
        resolved["mcts_simulations"] = int(simulation_count)
        configs.append(resolved)
    return configs


def build_il_final_elo_config(elo_cfg: dict, *, use_mcts: bool = True) -> dict:
    """Backward-compatible IL name for the shared final Elo configuration."""
    return build_final_elo_config(elo_cfg, use_mcts=use_mcts)


def build_il_final_elo_configs(elo_cfg: dict) -> tuple[dict, dict]:
    return (
        build_final_elo_config(elo_cfg, use_mcts=False),
        build_final_elo_config(elo_cfg, use_mcts=True),
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
