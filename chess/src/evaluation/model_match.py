"""Shared model-vs-model match entry point used by training and the play UI."""

from __future__ import annotations

import copy

from src.training.rl.trainer import (
    _eval_uses_central_inference,
    _resolve_eval_batch_games,
    _resolve_eval_workers,
    evaluate_models,
)


def _state_shapes(model):
    if model is None:
        return {}
    return {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
        if not key.endswith(("coord_x", "coord_y"))
    }


def _side_search_config(config, model, simulations):
    from src.model import config_for_model_spec

    model_spec = getattr(model, "model_spec", None)
    side_config = (
        config_for_model_spec(config, model_spec)
        if isinstance(model_spec, dict)
        else copy.deepcopy(config)
    )
    side_config.setdefault("model", {})
    side_config["model"]["history_positions"] = int(
        getattr(model, "history_positions", side_config["model"].get("history_positions", 0)) or 0
    )
    rl_cfg = side_config.setdefault("reinforcement_learning", {})
    rl_cfg["mcts_simulations"] = max(1, int(simulations))
    rl_cfg["eval_mcts_simulations_multiplier"] = 1.0
    rl_cfg["mcts_dynamic_budget_enabled"] = False
    return side_config


def run_model_match(
    model1,
    model2,
    config,
    device,
    *,
    num_games,
    game_index_offset=0,
    use_mcts_model1=True,
    use_mcts_model2=True,
    simulations_model1=192,
    simulations_model2=192,
    max_moves=220,
    stop_event=None,
    progress_callback=None,
    runtime_callback=None,
    show_progress=False,
    verbose=False,
):
    """Run a fair alternating-colour match through the canonical eval engine.

    The same batched MCTS, process workers and central GPU inference path used by
    RL evaluation are reused here. Different architectures safely fall back to
    the in-process executor because the central server owns one architecture.
    """
    model1_spec = getattr(model1, "model_spec", None)
    model2_spec = getattr(model2, "model_spec", None)
    if isinstance(model1_spec, dict):
        from src.model import config_for_model_spec

        match_config = config_for_model_spec(config, model1_spec)
    else:
        # Legacy in-memory callers without model_spec retain the old inference.
        from src.ui.game_setup import _infer_architecture_from_state_dict

        match_config = copy.deepcopy(config)
        match_config.setdefault("model", {})
        match_config["model"].update(
            _infer_architecture_from_state_dict(model1.state_dict())
        )
    match_config["model"]["print_summary"] = False

    play_cfg = match_config.get("play", {}) or {}
    rl_cfg = match_config.setdefault("reinforcement_learning", {})
    rl_cfg["eval_max_moves"] = max(1, int(max_moves))
    # Paired openings prevent a deterministic 100-game match from replaying
    # the same one or two games and keep colour comparisons meaningful.
    rl_cfg["eval_fixed_openings_enabled"] = True
    rl_cfg["eval_fixed_openings_pair_games"] = True
    rl_cfg["eval_auto_claim_draw"] = True

    configured_workers = play_cfg.get("match_workers", "auto")
    if configured_workers is not None:
        rl_cfg["eval_workers"] = configured_workers
    # Windows process creation is slow when 8-12 workers are started one after
    # another. GUI matches launch independent spawn processes concurrently;
    # RL training keeps its conservative sequential startup contract.
    rl_cfg["eval_parallel_worker_start"] = True
    rl_cfg["eval_parallel_worker_start_threads"] = 4
    configured_batch = play_cfg.get(
        "match_batch_games",
        play_cfg.get("match_active_games_per_worker", "auto"),
    )
    if not (isinstance(configured_batch, str) and configured_batch.strip().lower() in {"auto", "automatic"}):
        try:
            rl_cfg["eval_batch_games"] = max(1, int(configured_batch))
        except (TypeError, ValueError):
            pass

    same_architecture = bool(
        isinstance(model1_spec, dict)
        and isinstance(model2_spec, dict)
        and model1_spec.get("architecture_id") == model2_spec.get("architecture_id")
        and _state_shapes(model1) == _state_shapes(model2)
    )
    if not same_architecture:
        rl_cfg["eval_workers"] = 1
        rl_cfg["eval_central_inference_enabled"] = False

    model1_config = _side_search_config(match_config, model1, simulations_model1)
    model2_config = _side_search_config(match_config, model2, simulations_model2)
    workers = _resolve_eval_workers(match_config, device, int(num_games))
    central = bool(
        same_architecture
        and _eval_uses_central_inference(match_config, device)
        and int(num_games) >= max(1, int(rl_cfg.get("eval_central_inference_min_games", 2) or 2))
    )
    batch_games = int(_resolve_eval_batch_games(match_config, int(num_games)))
    if central:
        central_cfg = match_config.setdefault("central_inference", {})
        # A PID-based cache created a fresh Inductor directory on nearly every
        # play.py launch. Keep one cache namespace per GPU so later matches
        # reuse compiled kernels instead of paying the full compile again.
        device_index = int(getattr(device, "index", 0) or 0)
        central_cfg["compile_cache_rank"] = 970000 + device_index * 10
        rl_cfg["self_play_compile_use_lock"] = True
    runtime = {
        "engine": "shared_eval",
        "workers": int(workers),
        "batch_games": batch_games,
        "central_inference": central,
        "same_architecture": same_architecture,
        "ready": False,
    }
    if runtime_callback is not None:
        runtime_callback(dict(runtime))

    def _on_runtime_update(details):
        runtime.update(dict(details or {}))
        runtime["ready"] = bool(runtime.get("ready", False))
        if runtime_callback is not None:
            runtime_callback(dict(runtime))

    stats = evaluate_models(
        model1,
        model2,
        match_config,
        device,
        num_games=max(0, int(num_games)),
        game_index_offset=int(game_index_offset),
        use_fixed_openings=True,
        model1_mcts_config=model1_config,
        model2_mcts_config=model2_config,
        progress_desc="AI vs AI",
        progress_callback=progress_callback,
        use_mcts_model1=bool(use_mcts_model1),
        use_mcts_model2=bool(use_mcts_model2),
        stop_event=stop_event,
        show_progress=bool(show_progress),
        verbose=bool(verbose),
        runtime_ready_callback=_on_runtime_update,
    )
    stats["runtime"] = runtime
    return stats
