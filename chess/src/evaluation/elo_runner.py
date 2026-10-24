"""Shared orchestration helpers for Elo evaluation entrypoints."""

import math
from pathlib import Path

import torch

from src.evaluation.elo_estimator import (
    _EVAL_OPENING_LINES,
    _EVAL_OPENING_SUITE_ID,
    _available_cpu_count,
)
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


def mode_label(
    use_mcts: bool,
    simulations: int | None = None,
    *,
    budget_mode: str = "simulations",
    time_limit_seconds: float | None = None,
    adaptive_budget: bool = False,
    adaptive_strategy: str = "prior",
) -> str:
    if use_mcts:
        if str(budget_mode or "simulations").strip().lower() == "time":
            seconds = safe_float(time_limit_seconds)
            return f"MCTS ({seconds:g}s/move)" if seconds is not None else "MCTS (timed)"
        sims = safe_int(simulations) or 0
        if adaptive_budget and sims > 0:
            strategy = str(adaptive_strategy or "prior").strip().lower()
            return f"MCTS {strategy} (~{sims} avg sims)"
        return f"MCTS ({sims} sims)" if sims > 0 else "MCTS"
    return "Raw NN"


def elo_protocol_settings(
    runtime_cfg: dict,
    *,
    use_mcts: bool,
    simulations: int = 0,
    budget_mode: str = "simulations",
    time_limit_seconds: float | None = None,
) -> dict:
    """Return the strength-defining part of an Elo evaluation protocol."""
    runtime_cfg = dict(runtime_cfg or {})
    paired_openings = bool(runtime_cfg.get("paired_openings_enabled", True))
    budget_mode = str(budget_mode or "simulations").strip().lower()
    timed = bool(use_mcts and budget_mode == "time")
    dynamic_budget = bool(
        use_mcts
        and not timed
        and runtime_cfg.get("mcts_dynamic_budget_enabled", False)
    )
    dynamic_strategy = str(
        runtime_cfg.get("mcts_dynamic_budget_strategy", "feedback") or "feedback"
    ).strip().lower()
    if dynamic_strategy not in {"prior", "feedback"}:
        raise ValueError("mcts_dynamic_budget_strategy must be 'prior' or 'feedback'")
    efficiency_feedback = bool(
        use_mcts
        and dynamic_strategy == "feedback"
        and (
            dynamic_budget
            or (timed and runtime_cfg.get("mcts_dynamic_budget_enabled", False))
        )
        and runtime_cfg.get("search_efficiency_feedback_enabled", False)
    )
    settings = {
        "use_mcts": bool(use_mcts),
        "budget_mode": budget_mode if use_mcts else "raw_nn",
        "simulations": max(0, int(simulations or 0)) if use_mcts and not timed else 0,
        "time_limit_seconds": (
            float(time_limit_seconds) if timed and time_limit_seconds is not None else None
        ),
        "stockfish_time_limit": float(
            runtime_cfg.get("stockfish_time_limit", 0.0) or 0.0
        ),
        "stockfish_threads": max(1, int(runtime_cfg.get("stockfish_threads", 1) or 1)),
        "stockfish_hash_mb": max(1, int(runtime_cfg.get("stockfish_hash_mb", 64) or 64)),
        "max_moves": max(1, int(runtime_cfg.get("max_moves", 150) or 150)),
        "paired_openings_enabled": paired_openings,
        "paired_openings_max_plies": (
            max(0, int(runtime_cfg.get("paired_openings_max_plies", 8) or 0))
            if paired_openings
            else 0
        ),
        "opening_suite_id": _EVAL_OPENING_SUITE_ID if paired_openings else "disabled",
        "opening_suite_positions": len(_EVAL_OPENING_LINES) if paired_openings else 0,
        "claim_draw": True,
        "rating_method": "nearest_50pct_stockfish_level",
        # Strict MCTS Elo requires an exact per-root budget. Adaptive budgets
        # are kept in a separately versioned protocol below.
        "mcts_simulation_budget_strict": bool(use_mcts and not timed and not dynamic_budget),
        **(
            {
                "mcts_root_schedule": "revisiting"
                if timed or dynamic_budget
                else str(runtime_cfg.get("mcts_gumbel_root_schedule", "standard"))
            }
            if use_mcts
            else {}
        ),
        "pair_aware_standard_error": paired_openings,
        **(
            {"time_search_workers": max(1, int(runtime_cfg.get("workers", 1) or 1))}
            if timed
            else {}
        ),
        **(_timed_search_protocol(runtime_cfg) if timed else {}),
    }
    if dynamic_budget:
        settings.update({
            "mcts_dynamic_budget_enabled": True,
            "mcts_dynamic_budget_strategy": dynamic_strategy,
            "mcts_dynamic_budget_protocol": (
                (
                    "search-feedback-e1"
                    if efficiency_feedback
                    else "search-feedback-v2"
                )
                if dynamic_strategy == "feedback" else "prior-v1"
            ),
            "mcts_dynamic_budget_target_simulations": max(0, int(simulations or 0)),
            "mcts_dynamic_budget_min": max(
                1, int(runtime_cfg.get("mcts_dynamic_budget_min", 64))
            ),
            "mcts_dynamic_budget_max_multiplier": float(
                runtime_cfg.get("mcts_dynamic_budget_max_multiplier", 5.0 / 3.0)
            ),
        })
    if efficiency_feedback:
        settings.update({
            "mcts_search_efficiency_feedback_enabled": True,
            "mcts_search_efficiency_feedback_protocol": "weighted-sh-pass-v1",
        })
    if dynamic_budget and dynamic_strategy == "feedback":
        inference_path = str(
            runtime_cfg.get("mcts_dynamic_budget_inference", "") or ""
        ).strip().lower()
        if inference_path not in {"local", "central"}:
            inference_path = (
                "central" if runtime_cfg.get("eval_elo_central_inference_enabled", False)
                and int(runtime_cfg.get("workers", 1) or 1) > 1 else "local"
            )
        settings.update({
            "mcts_dynamic_budget_inference": inference_path,
            "mcts_dynamic_budget_pool_scope": (
                "central_client_group" if inference_path == "central" else "per_search_batch"
            ),
        })
    return settings


def _timed_search_protocol(runtime_cfg: dict) -> dict:
    """Describe the effective path so sequential and parallel Elo stay distinct."""
    central = bool(runtime_cfg.get("mcts_time_search_central_inference", False))
    batched_roots = bool(
        runtime_cfg.get(
            "mcts_time_search_batch_model_moves",
            runtime_cfg.get("batch_model_moves", True),
        )
    )
    return {
        "time_search_protocol": "mcts-deadline-v1",
        "time_search_batch_model_moves": batched_roots,
        "time_search_central_inference": central,
        "time_search_style": str(
            runtime_cfg.get(
                "mcts_time_search_style",
                "central_inference" if central else "batched_roots" if batched_roots else "independent_roots",
            )
        ),
    }


def elo_protocol_matches_settings(
    settings: dict | None,
    runtime_cfg: dict,
    *,
    use_mcts: bool,
    simulations: int = 0,
    budget_mode: str = "simulations",
    time_limit_seconds: float | None = None,
) -> bool:
    """Whether a stored rating used the same strength-defining protocol."""
    if not isinstance(settings, dict):
        return False
    expected = elo_protocol_settings(
        runtime_cfg,
        use_mcts=use_mcts,
        simulations=simulations,
        budget_mode=budget_mode,
        time_limit_seconds=time_limit_seconds,
    )
    if bool(expected.get("mcts_search_efficiency_feedback_enabled", False)) != bool(
        settings.get("mcts_search_efficiency_feedback_enabled", False)
    ):
        return False
    for key, active_value in expected.items():
        if key not in settings:
            return False
        stored_value = settings.get(key)
        if isinstance(active_value, float):
            try:
                if not math.isclose(
                    float(stored_value), active_value, rel_tol=0.0, abs_tol=1e-9
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif stored_value != active_value:
            return False
    return True


def checkpoint_elo_seed(
    checkpoint_path,
    *,
    use_mcts: bool,
    simulations: int = 0,
    budget_mode: str = "simulations",
    time_limit_seconds: float | None = None,
    runtime_cfg: dict | None = None,
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

    budget_mode = str(budget_mode or "simulations").strip().lower()
    if budget_mode == "time":
        requested_seconds = safe_float(time_limit_seconds)
        by_time = checkpoint.get("estimated_elo_mcts_by_time")
        if requested_seconds is None or not math.isfinite(requested_seconds) or requested_seconds <= 0:
            return _checkpoint_simulation_seed(checkpoint, checkpoint_path, simulations, approximate=True)
        time_candidates: list[tuple[float, float, dict | None]] = []
        if isinstance(by_time, dict):
            for raw_seconds, raw_entry in by_time.items():
                entry_seconds = safe_float(raw_seconds)
                if entry_seconds is None and isinstance(raw_entry, dict):
                    entry_seconds = safe_float(raw_entry.get("time_limit_seconds"))
                elo = safe_float(raw_entry.get("elo")) if isinstance(raw_entry, dict) else safe_float(raw_entry)
                if (
                    entry_seconds is not None
                    and math.isfinite(entry_seconds)
                    and entry_seconds > 0
                    and elo is not None
                    and math.isfinite(elo)
                ):
                    settings = raw_entry.get("settings") if isinstance(raw_entry, dict) else None
                    time_candidates.append((float(entry_seconds), float(elo), settings if isinstance(settings, dict) else None))

        same_budget = [
            item for item in time_candidates
            if math.isclose(item[0], requested_seconds, rel_tol=0.0, abs_tol=1e-9)
        ]
        for entry_seconds, elo, settings in same_budget:
            if elo_protocol_matches_settings(
                settings,
                runtime_cfg or {},
                use_mcts=True,
                simulations=simulations,
                budget_mode="time",
                time_limit_seconds=requested_seconds,
            ):
                return elo, (
                    f"{checkpoint_path.name} previous timed MCTS Elo "
                    f"@{_time_budget_key(entry_seconds)}s (matching protocol)"
                )
        if same_budget:
            entry_seconds, elo, _settings = same_budget[-1]
            return elo, (
                f"{checkpoint_path.name} approximate timed MCTS scheduling prior "
                f"@{_time_budget_key(entry_seconds)}s (protocol differs or is legacy)"
            )
        if time_candidates:
            entry_seconds, elo, _settings = min(
                time_candidates,
                key=lambda item: abs(math.log(item[0] / requested_seconds)),
            )
            return elo, (
                f"{checkpoint_path.name} approximate timed MCTS scheduling prior "
                f"@{_time_budget_key(entry_seconds)}s (nearest time budget)"
            )
        return _checkpoint_simulation_seed(checkpoint, checkpoint_path, simulations, approximate=True)

    if bool((runtime_cfg or {}).get("mcts_dynamic_budget_enabled", False)):
        adaptive_entries = _adaptive_simulation_entries(checkpoint)
        requested_sims = max(0, int(simulations or 0))
        same_target = [item for item in adaptive_entries if item[0] == requested_sims]
        for entry_sims, elo, settings in same_target:
            if elo_protocol_matches_settings(
                settings,
                runtime_cfg or {},
                use_mcts=True,
                simulations=entry_sims,
                budget_mode="simulations",
            ):
                return elo, (
                    f"{checkpoint_path.name} previous adaptive MCTS Elo "
                    f"@{entry_sims} target average (matching protocol)"
                )
        compatible = []
        for entry_sims, elo, settings in adaptive_entries:
            candidate_runtime = dict(runtime_cfg or {})
            candidate_runtime["mcts_dynamic_budget_enabled"] = True
            if elo_protocol_matches_settings(
                settings,
                candidate_runtime,
                use_mcts=True,
                simulations=entry_sims,
                budget_mode="simulations",
            ):
                compatible.append((entry_sims, elo))
        if compatible:
            entry_sims, elo = min(
                compatible,
                key=lambda item: abs(math.log2((item[0] + 1.0) / (requested_sims + 1.0))),
            )
            return elo, (
                f"{checkpoint_path.name} approximate adaptive MCTS scheduling prior "
                f"@{entry_sims} target average (same protocol)"
            )
        # Fixed-budget results are useful only to place the initial ladder;
        # adaptive ratings from a different strategy are not interchangeable.
        return _checkpoint_simulation_seed(checkpoint, checkpoint_path, simulations, approximate=True)

    return _checkpoint_simulation_seed(checkpoint, checkpoint_path, simulations)


def _adaptive_simulation_entries(checkpoint):
    raw_entries = checkpoint.get("estimated_elo_mcts_adaptive_by_simulations")
    candidates = []
    if not isinstance(raw_entries, dict):
        return candidates
    for raw_sims, raw_entry in raw_entries.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        sims = safe_int(raw_sims)
        if sims is None:
            sims = safe_int(entry.get("simulations"))
        elo = safe_float(entry.get("elo")) if entry else safe_float(raw_entry)
        if sims is None or elo is None or not math.isfinite(elo):
            continue
        settings = entry.get("settings") if isinstance(entry.get("settings"), dict) else None
        candidates.append((max(0, int(sims)), float(elo), settings))
    return candidates


def _checkpoint_simulation_seed(checkpoint, checkpoint_path, simulations, *, approximate=False):
    """Use the nearest same-checkpoint fixed-simulation MCTS rating as a prior."""
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
        chosen = next((item for item in candidates if item[0] == requested_sims), None)
        if chosen is None:
            chosen = min(
                candidates,
                key=lambda item: abs(math.log2((item[0] + 1.0) / (requested_sims + 1.0))),
            )
        sims, elo = chosen
        relation = "same" if sims == requested_sims else "nearest"
        if approximate:
            return elo, f"{checkpoint_path.name} approximate MCTS scheduling prior @{sims} ({relation} simulation budget)"
        return elo, f"{checkpoint_path.name} previous MCTS Elo @{sims} ({relation} budget)"

    elo = safe_float(checkpoint.get("estimated_elo_mcts", checkpoint.get("last_estimated_elo_mcts")))
    if elo is not None and math.isfinite(elo):
        stored_sims = safe_int(checkpoint.get("estimated_elo_mcts_simulations"))
        suffix = f" @{stored_sims}" if stored_sims is not None else ""
        if approximate:
            return float(elo), f"{checkpoint_path.name} approximate previous MCTS scheduling prior{suffix}"
        return float(elo), f"{checkpoint_path.name} previous MCTS Elo{suffix}"
    return None, None


def apply_checkpoint_elo_seed(
    runtime_cfg: dict,
    checkpoint_path,
    *,
    use_mcts: bool,
    simulations: int = 0,
    budget_mode: str = "simulations",
    time_limit_seconds: float | None = None,
) -> float | None:
    """Attach checkpoint Elo as an opponent-selection seed when none is set."""
    existing = safe_float((runtime_cfg or {}).get("adaptive_initial_elo"))
    if existing is not None and math.isfinite(existing):
        return float(existing)
    elo, source = checkpoint_elo_seed(
        checkpoint_path,
        use_mcts=use_mcts,
        simulations=simulations,
        budget_mode=budget_mode,
        time_limit_seconds=time_limit_seconds,
        runtime_cfg=runtime_cfg,
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
    budget_mode=None,
    time_limit_seconds=None,
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
    if budget_mode is None:
        inference_cfg = source.get("inference", {}) or {}
        budget_mode = (
            inference_cfg.get("budget_mode", "simulations")
            if standalone
            else "simulations"
        )
    budget_mode = str(budget_mode or "simulations").strip().lower()
    if budget_mode not in {"simulations", "time"}:
        raise ValueError("mcts_budget_mode must be 'simulations' or 'time'")
    runtime_cfg["mcts_budget_mode"] = budget_mode if use_mcts else "raw_nn"
    if use_mcts and budget_mode == "time":
        if time_limit_seconds is None:
            inference_cfg = source.get("inference", {}) or {}
            time_limit_seconds = inference_cfg.get("move_time_seconds", 0.2)
        if isinstance(time_limit_seconds, bool):
            raise ValueError("inference.move_time_seconds must be a finite positive number")
        time_limit_seconds = safe_float(time_limit_seconds)
        if time_limit_seconds is None or not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
            raise ValueError("inference.move_time_seconds must be a finite positive number")
        runtime_cfg["mcts_move_time_seconds"] = float(time_limit_seconds)
    else:
        runtime_cfg.pop("mcts_move_time_seconds", None)
    if "mcts_gumbel_root_schedule" in source:
        runtime_cfg["mcts_gumbel_root_schedule"] = str(
            source["mcts_gumbel_root_schedule"]
        )
    else:
        # Let run_elo_check/estimator inherit the configured RL root schedule.
        runtime_cfg.pop("mcts_gumbel_root_schedule", None)
    # Keep a positive compatibility value for callers that still pass a
    # simulation-count positional argument; timed search ignores it.
    runtime_cfg["mcts_simulations"] = max(1, int(simulations or 1))
    runtime_cfg["workers"] = int(resolve_eval_workers(workers, use_mcts, source))
    runtime_cfg["batch_model_moves"] = bool(source.get("batch_model_moves", True))
    runtime_cfg["batch_raw_model_moves"] = bool(source.get("batch_raw_model_moves", False))
    runtime_cfg["stockfish_hide_window"] = bool(source.get("stockfish_hide_window", True))
    if use_mcts and budget_mode == "time":
        cap = source.get("mcts_timed_parallel_games", 4)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("elo_estimation.mcts_timed_parallel_games must be a positive integer")
        requested = int(runtime_cfg["workers"])
        if requested <= 0:
            requested = cap
        runtime_cfg["mcts_timed_parallel_games_requested_workers"] = int(requested)
        runtime_cfg["workers"] = max(1, min(requested, cap, _available_cpu_count()))
        runtime_cfg["mcts_timed_parallel_games"] = int(cap)
        runtime_cfg["mcts_time_search_protocol"] = "mcts-deadline-v1"
        runtime_cfg["batch_model_moves"] = True

    if standalone:
        runtime_cfg["stockfish_priority"] = str(source.get("eval_elo_stockfish_priority", "normal"))
        runtime_cfg["prioritize_training"] = False
        runtime_cfg["reserve_dataloader_workers"] = False
        runtime_cfg["free_threads_utilization"] = 1.0
        runtime_cfg["auto_worker_reserve_cpus"] = 0
        runtime_cfg["progress_bar"] = "always"

    return runtime_cfg


def build_eval_elo_runtime_config(
    config: dict,
    elo_cfg: dict,
    *,
    device,
    use_mcts: bool,
    simulations: int,
    workers=None,
    budget_mode: str = "simulations",
    time_limit_seconds=None,
    levels=None,
    games_per_level=None,
    sf_time=None,
    max_moves=None,
    sf_path=None,
) -> dict:
    """Build the exact mode-specific runtime policy used by ``eval_elo``.

    Keep search schedule, adaptive-budget routing, worker resolution and the
    central-inference decision in one place so reusable evaluation tools can
    share the same contract instead of recreating a subtly different path.
    """
    elo_cfg = dict(elo_cfg or {})
    runtime_cfg = build_eval_elo_config(
        elo_cfg,
        levels=levels,
        games_per_level=games_per_level,
        use_mcts=use_mcts,
        simulations=simulations,
        budget_mode=budget_mode,
        time_limit_seconds=time_limit_seconds,
        sf_time=sf_time,
        max_moves=max_moves,
        sf_path=sf_path,
        workers=resolve_eval_workers(workers, use_mcts, elo_cfg),
        standalone=True,
    )

    rl_config = (config or {}).get("reinforcement_learning", {}) or {}
    search_config = rl_config.get("search", {}) or {}
    runtime_cfg["mcts_gumbel_root_schedule"] = str(
        rl_config.get(
            "mcts_gumbel_root_schedule",
            search_config.get("gumbel_root_schedule", "standard"),
        )
    )
    runtime_cfg["search_efficiency_feedback_enabled"] = bool(
        ((config or {}).get("inference", {}) or {}).get(
            "search_efficiency_feedback_enabled", False
        )
    )

    budget_mode = str(budget_mode or "simulations").strip().lower()
    if use_mcts and budget_mode == "simulations":
        runtime_cfg["mcts_dynamic_budget_enabled"] = bool(
            rl_config.get("mcts_dynamic_budget_enabled", False)
        )
        runtime_cfg["mcts_dynamic_budget_strategy"] = str(
            rl_config.get("mcts_dynamic_budget_strategy", "feedback") or "feedback"
        ).strip().lower()
        runtime_cfg["mcts_dynamic_budget_min"] = int(
            rl_config.get("mcts_dynamic_budget_min", 64)
        )
        runtime_cfg["mcts_dynamic_budget_max_multiplier"] = float(
            rl_config.get("mcts_dynamic_budget_max_multiplier", 5.0 / 3.0)
        )
        runtime_cfg["eval_elo_central_inference_enabled"] = bool(
            rl_config.get("eval_elo_central_inference_enabled", True)
        )
        central_dynamic = bool(
            runtime_cfg["mcts_dynamic_budget_enabled"]
            and runtime_cfg["mcts_dynamic_budget_strategy"] == "feedback"
            and getattr(device, "type", None) == "cuda"
            and int(runtime_cfg.get("workers", 1) or 1) > 1
            and runtime_cfg["eval_elo_central_inference_enabled"]
        )
        runtime_cfg["mcts_dynamic_budget_inference"] = (
            "central" if central_dynamic else "local"
        )
    elif use_mcts and budget_mode == "time" and runtime_cfg[
        "search_efficiency_feedback_enabled"
    ]:
        runtime_cfg["mcts_dynamic_budget_enabled"] = bool(
            rl_config.get("mcts_dynamic_budget_enabled", False)
        )
        runtime_cfg["mcts_dynamic_budget_strategy"] = str(
            rl_config.get("mcts_dynamic_budget_strategy", "feedback") or "feedback"
        ).strip().lower()

    if use_mcts and budget_mode == "time":
        central = bool(
            getattr(device, "type", None) == "cuda"
            and int(runtime_cfg.get("workers", 1) or 1) > 1
            and runtime_cfg.get("eval_elo_central_inference_enabled", False)
        )
        batched_roots = bool(runtime_cfg.get("batch_model_moves", True))
        runtime_cfg["mcts_time_search_central_inference"] = central
        runtime_cfg["mcts_time_search_batch_model_moves"] = batched_roots
        runtime_cfg["mcts_time_search_style"] = (
            "central_inference"
            if central
            else "batched_roots"
            if batched_roots
            else "independent_roots"
        )
    return runtime_cfg


def create_elo_estimator(model, config: dict, device, elo_config: dict, *, stop_event=None):
    """Create the shared Elo runtime used by the CLI and benchmark module."""
    # Import locally to keep the runner's helper imports acyclic.
    from src.evaluation.elo_estimator import EloEstimator

    resolved = dict(elo_config or {})
    return EloEstimator(
        model,
        config,
        device,
        resolved.get("stockfish_path", "stockfish"),
        stop_event=stop_event,
        elo_config=resolved,
    )


def build_il_periodic_elo_config(elo_cfg: dict) -> dict:
    """IL epoch checks use Raw NN settings and keep training-friendly CPU limits."""
    resolved = build_eval_elo_config(
        elo_cfg or {},
        use_mcts=False,
        simulations=(elo_cfg or {}).get("mcts_simulations", 0),
        budget_mode="simulations",
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
        budget_mode="simulations",
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
    resolved = dict(elo_config or {})
    rl_cfg = dict((config or {}).get("reinforcement_learning", {}) or {})
    search_cfg = dict(rl_cfg.get("search", {}) or {})
    configured_schedule = rl_cfg.get(
        "mcts_gumbel_root_schedule",
        search_cfg.get("gumbel_root_schedule", "standard"),
    )
    for key, fallback in (
        ("mcts_budget_mode", "simulations"),
        ("mcts_gumbel_root_schedule", configured_schedule),
        ("mcts_dynamic_budget_strategy", rl_cfg.get(
            "mcts_dynamic_budget_strategy",
            search_cfg.get("dynamic_budget_strategy", "feedback"),
        )),
    ):
        resolved.setdefault(key, fallback)
    if str(resolved.get("mcts_budget_mode", "simulations")).strip().lower() == "time":
        inference_cfg = dict((config or {}).get("inference", {}) or {})
        resolved.setdefault(
            "mcts_move_time_seconds",
            inference_cfg.get("move_time_seconds", 0.2),
        )
    else:
        resolved.pop("mcts_move_time_seconds", None)
    estimator = create_elo_estimator(
        model,
        config,
        device,
        resolved,
        stop_event=stop_event,
    )
    return estimator.estimate(
        levels=None,
        games_per_level=resolved.get("games_per_level"),
        stockfish_time_limit=resolved.get("stockfish_time_limit", 0.05),
        max_moves=resolved.get("max_moves", 150),
        use_mcts=resolved.get("use_mcts", False),
        simulations=resolved.get("mcts_simulations", 100),
        workers=resolved.get("workers", 0),
    )


def persist_estimated_elo(
    checkpoint_path: Path,
    estimated_elo,
    *,
    levels: list[int],
    games_per_level: int,
    use_mcts: bool,
    simulations: int,
    budget_mode: str | None = None,
    time_limit_seconds: float | None = None,
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

    if budget_mode is None:
        budget_mode = (elo_result or {}).get("mcts_budget_mode", "simulations")
    if time_limit_seconds is None:
        time_limit_seconds = (elo_result or {}).get("mcts_move_time_seconds")
    timed_mcts = bool(use_mcts and str(budget_mode).strip().lower() == "time")
    adaptive_mcts = bool(
        use_mcts
        and not timed_mcts
        and isinstance(elo_result, dict)
        and elo_result.get("mcts_dynamic_budget_enabled", False)
    )
    existing_key = (
        "estimated_elo_nn" if not use_mcts else
        "estimated_elo_mcts" if not timed_mcts and not adaptive_mcts else
        None
    )
    existing_elo = safe_float(checkpoint.get(existing_key)) if existing_key else None
    ok, error = persist_checkpoint_elo_metadata(
        checkpoint_path,
        elo_value,
        levels=levels,
        games_per_level=games_per_level,
        use_mcts=use_mcts,
        simulations=simulations,
        budget_mode=budget_mode,
        time_limit_seconds=time_limit_seconds,
        sf_time=sf_time,
        source=source,
        elo_result=elo_result,
    )
    if ok and announce_success:
        label = (
            f"MCTS Elo @{float(time_limit_seconds):g}s/move"
            if timed_mcts and time_limit_seconds is not None
            else f"Adaptive MCTS Elo (~{int(simulations)} target avg sims)"
            if adaptive_mcts
            else "MCTS Elo" if use_mcts else "NN Elo"
        )
        if existing_elo is not None:
            print(f"{print_prefix}✓ Updated {label}: {int(round(existing_elo))} -> {int(round(elo_value))}")
        else:
            print(f"{print_prefix}✓ Saved {label} into checkpoint metadata: {int(round(elo_value))}")
    elif error:
        print(f"{print_prefix}! {error}")


def _time_budget_key(seconds):
    return format(float(seconds), ".9g")
