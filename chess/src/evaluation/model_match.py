"""Shared model-vs-model match entry point used by training and the play UI."""

from __future__ import annotations

import copy
import os

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


def prepare_models_for_play_search(model1, model2, device):
    """Prepare supported CUDA play models for channels-last convolution.

    Call once after both checkpoints are loaded and before starting either a
    visible-board search or a background match. This intentionally does not
    belong in :func:`run_model_match`, which may be called after the visible
    game has already started using these same model objects.

    The conversion is a no-op on CPU, for unsupported architectures, and when
    either model is not already on the requested CUDA device. It never moves a
    model between devices or changes its train/eval mode.

    Returns the number of model objects whose parameter layout was converted.
    """
    import torch

    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return 0
    if target_device.type != "cuda" or not torch.cuda.is_available():
        return 0
    if target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())

    converted = 0
    seen = set()
    supported_architectures = {"se_cnn_v9", "nested_cnn_v10"}
    for model in (model1, model2):
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        if not isinstance(model, torch.nn.Module):
            continue
        model_spec = getattr(model, "model_spec", None)
        architecture_id = (
            model_spec.get("architecture_id")
            if isinstance(model_spec, dict)
            else None
        ) or getattr(model, "architecture_id", None)
        if architecture_id not in supported_architectures:
            continue

        parameters = tuple(model.parameters())
        if not parameters or any(parameter.device != target_device for parameter in parameters):
            continue
        convolution_weights = tuple(
            parameter for parameter in parameters if parameter.ndim == 4
        )
        if not convolution_weights or all(
            parameter.is_contiguous(memory_format=torch.channels_last)
            for parameter in convolution_weights
        ):
            continue

        model.to(memory_format=torch.channels_last)
        converted += 1
    return converted


def get_model_match_opening_prefix(config, game_index=0):
    """Return the canonical paired-opening prefix for a background game.

    The play UI's visible game starts from the standard initial position;
    this compatibility helper retains the evaluator's paired-opening lookup
    for background match indices.
    """
    from src.training.rl.trainer import _get_eval_opening_prefix

    match_config = copy.deepcopy(config)
    rl_cfg = match_config.setdefault("reinforcement_learning", {})
    rl_cfg["eval_fixed_openings_enabled"] = True
    rl_cfg["eval_fixed_openings_pair_games"] = True
    return tuple(
        _get_eval_opening_prefix(
            match_config,
            int(game_index),
            enabled_override=True,
        )
    )


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
    max_moves=None,
    stop_event=None,
    progress_callback=None,
    runtime_callback=None,
    show_progress=False,
    verbose=False,
):
    """Run a fair alternating-colour match through the canonical eval engine.

    The same batched MCTS and evaluation rules as RL are reused here. A
    one-worker GUI match evaluates the already-loaded models in-process;
    central GPU inference is reserved for multi-worker batching or an explicit
    play.match_central_inference override. Each side keeps its checkpoint's
    input and search contract, including when the two architectures differ.
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
    # Interactive model matches must finish by chess rules. A finite cap is
    # reserved for explicit smoke/diagnostic runs; RL/Elo keep their own caps.
    rl_cfg["eval_finish_by_rules"] = max_moves is None
    rl_cfg["eval_unresolved_score_neutral"] = False
    if max_moves is not None:
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

    central_preference = play_cfg.get("match_central_inference", "auto")
    if isinstance(central_preference, bool):
        central_mode = "on" if central_preference else "off"
    else:
        central_mode = str(central_preference or "auto").strip().lower()
    if central_mode in {"true", "yes", "on", "enabled"}:
        central_mode = "on"
    elif central_mode in {"false", "no", "off", "disabled"}:
        central_mode = "off"
    elif central_mode in {"auto", "automatic"}:
        central_mode = "auto"
    else:
        raise ValueError(
            "play.match_central_inference must be 'auto', true, or false"
        )
    if central_mode == "on":
        rl_cfg["eval_central_inference_enabled"] = True
    elif central_mode == "off":
        rl_cfg["eval_central_inference_enabled"] = False

    same_architecture = bool(
        isinstance(model1_spec, dict)
        and isinstance(model2_spec, dict)
        and model1_spec.get("architecture_id") == model2_spec.get("architecture_id")
        and _state_shapes(model1) == _state_shapes(model2)
    )
    model_architectures = {
        str(spec.get("architecture_id"))
        for spec in (model1_spec, model2_spec)
        if isinstance(spec, dict) and spec.get("architecture_id")
    }
    known_v9_v10_match = model_architectures == {
        "se_cnn_v9",
        "nested_cnn_v10",
    }
    requested_workers = configured_workers
    auto_workers_requested = (
        requested_workers is None
        or str(requested_workers).strip().lower() in {"auto", "automatic", "0"}
    )
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    cuda_match = str(device_type).lower() == "cuda"
    if not same_architecture:
        if known_v9_v10_match and cuda_match and central_mode != "off":
            if auto_workers_requested:
                if int(num_games) >= 32:
                    # On 12 logical CPUs, the corrected end-to-end 99-game
                    # benchmark measured the four-worker central route at
                    # 85 s versus 174 s for direct inference, with identical
                    # root budgets and selected leaves. Leave two CPUs free
                    # for the play UI and cap the automatic topology at four.
                    cpu_count = max(1, int(os.cpu_count() or 2))
                    rl_cfg["eval_workers"] = min(4, max(1, cpu_count - 2))
                    if central_mode == "auto":
                        rl_cfg["eval_central_inference_enabled"] = True
                else:
                    # The server's cold startup is not amortized by a short
                    # diagnostic match, so keep those on the direct path.
                    rl_cfg["eval_workers"] = 1
                    if central_mode == "auto":
                        rl_cfg["eval_central_inference_enabled"] = False
            else:
                try:
                    requested_worker_count = int(requested_workers)
                except (TypeError, ValueError):
                    requested_worker_count = 1
                if requested_worker_count > 1 and central_mode == "auto":
                    # An explicit worker count opts into CPU MCTS workers;
                    # route their inference through the one CUDA owner.
                    rl_cfg["eval_central_inference_enabled"] = True
                elif requested_worker_count <= 1:
                    rl_cfg["eval_workers"] = 1
        else:
            # Unmeasured architecture pairs, CPU matches, and explicit direct
            # mode retain the safe single-process evaluator.
            rl_cfg["eval_workers"] = 1

    model1_config = _side_search_config(match_config, model1, simulations_model1)
    model2_config = _side_search_config(match_config, model2, simulations_model2)
    workers = _resolve_eval_workers(match_config, device, int(num_games))
    if central_mode == "auto" and workers <= 1:
        # A one-worker GUI run cannot amortize central IPC or model loading.
        # Let the direct batched evaluator use the models already on this GPU.
        rl_cfg["eval_central_inference_enabled"] = False
        workers = _resolve_eval_workers(match_config, device, int(num_games))
    central = bool(
        _eval_uses_central_inference(match_config, device)
        and int(num_games) >= max(1, int(rl_cfg.get("eval_central_inference_min_games", 2) or 2))
    )
    batch_games = int(_resolve_eval_batch_games(match_config, int(num_games)))
    if central:
        central_cfg = match_config.setdefault("central_inference", {})
        central_cfg["use_compile"] = bool(
            play_cfg.get("match_central_inference_compile", False)
        )
        # Conv-BN folding operates on a disposable eager eval copy. Compiled
        # models keep their original parameters so a later checkpoint reload
        # remains compatible with the unfused state dict.
        central_cfg["fold_eval_conv_bn"] = bool(
            play_cfg.get("match_fold_eval_conv_bn", False)
            and not central_cfg["use_compile"]
        )
        # Variable eval batch shapes made cuDNN autotune repeat expensive
        # per-shape work. This is a play-only override; RL/self-play retain the
        # configured training default.
        central_cfg["cudnn_benchmark"] = False
        # Central eval historically uses FP16, while direct play follows the
        # hardware BF16 setting. Allow a play-only override for controlled
        # comparisons, defaulting to the user's existing hardware preference.
        requested_bfloat16 = play_cfg.get(
            "match_central_inference_bfloat16",
            match_config.get("hardware", {}).get("use_bfloat16", False),
        )
        if isinstance(requested_bfloat16, str):
            requested_bfloat16 = requested_bfloat16.strip().lower() in {
                "1", "true", "yes", "on", "enabled"
            }
        central_cfg["use_bfloat16"] = bool(requested_bfloat16)
        # The CUDA-graph path captures only recurrent, exact batch shapes and
        # keeps eager inference for all other shapes. It is enabled only for
        # the two currently validated architectures and long GUI matches, so
        # its capture/startup cost can be amortized. Set
        # play.match_central_cuda_graph=false to force the original path.
        graph_architectures = {"se_cnn_v9", "nested_cnn_v10"}
        graph_supported = model_architectures.issubset(graph_architectures)
        use_cuda_graph_forward = bool(
            play_cfg.get("match_central_cuda_graph", True)
            and graph_supported
            and cuda_match
            and int(num_games) >= 32
            and not central_cfg["use_compile"]
        )
        central_cfg["cuda_graph_forward_enabled"] = use_cuda_graph_forward
        central_cfg["cuda_graph_forward_min_observations"] = max(
            4,
            int(play_cfg.get("match_central_cuda_graph_min_observations", 4) or 4),
        )
        central_cfg["profile_forward_shapes"] = bool(
            play_cfg.get("match_central_forward_shape_profile", False)
        )
        if use_cuda_graph_forward:
            # Hot shapes come from a real 32-game v9/v10 central match. The
            # runner captures no padded shapes and falls back eagerly for any
            # batch outside these bounded sets.
            central_cfg["cuda_graph_forward_hot_shapes"] = {
                "se_cnn_v9": tuple(
                    play_cfg.get(
                        "match_central_cuda_graph_v9_shapes",
                        (2, 5, 6, 8, 11, 14, 17, 18),
                    )
                ),
                "nested_cnn_v10": tuple(
                    play_cfg.get(
                        "match_central_cuda_graph_v10_shapes",
                        (2, 4, 5, 6, 8, 10, 12, 20),
                    )
                ),
            }
            central_cfg["profile_cuda_graphs"] = bool(
                play_cfg.get("match_central_cuda_graph_profile", False)
            )
        # A PID-based cache created a fresh Inductor directory on nearly every
        # play.py launch. Keep one cache namespace per GPU so later matches
        # reuse compiled kernels instead of paying the full compile again.
        device_index = int(getattr(device, "index", 0) or 0)
        central_cfg["compile_cache_rank"] = 970000 + device_index * 10
        rl_cfg["self_play_compile_use_lock"] = True
    central_amp_dtype = None
    if central:
        central_amp_dtype = (
            "bfloat16"
            if match_config.get("central_inference", {}).get("use_bfloat16", False)
            else "float16"
        )
    runtime = {
        "engine": "shared_eval",
        "workers": int(workers),
        "batch_games": batch_games,
        "central_inference": central,
        "central_cudnn_benchmark": False if central else None,
        "central_amp_dtype": central_amp_dtype,
        "central_compile_enabled": bool(
            match_config.get("central_inference", {}).get("use_compile", False)
        ) if central else False,
        "central_fold_eval_conv_bn": bool(
            match_config.get("central_inference", {}).get("fold_eval_conv_bn", False)
        ) if central else False,
        "central_cuda_graph_forward_enabled": bool(
            match_config.get("central_inference", {}).get("cuda_graph_forward_enabled", False)
        ) if central else False,
        "same_architecture": same_architecture,
        "ready": False,
        "status": "starting",
        "server_pids": [],
        "worker_pids": [],
        "termination_mode": (
            "chess_rules" if max_moves is None else "diagnostic_ply_cap"
        ),
        "diagnostic_cap_plies": None if max_moves is None else max(1, int(max_moves)),
        "unresolved_score_policy": "excluded",
    }
    if runtime_callback is not None:
        runtime_callback(dict(runtime))

    def _on_runtime_update(details):
        runtime.update(dict(details or {}))
        runtime["ready"] = bool(runtime.get("ready", False))
        if runtime_callback is not None:
            runtime_callback(dict(runtime))

    try:
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
            runtime_progress_callback=_on_runtime_update,
        )
    except Exception as exc:
        runtime.update({"ready": False, "status": "error", "error": str(exc)})
        if runtime_callback is not None:
            runtime_callback(dict(runtime))
        raise
    if bool(stats.get("cancelled", False)):
        runtime.update({
            "ready": False,
            "status": "cancelled",
            "cancel_reason": str(stats.get("cancel_reason", "stop requested")),
            "cancel_stage": str(stats.get("cancel_stage", "evaluation")),
        })
    elif stats.get("error") or stats.get("infrastructure_error"):
        runtime.update({
            "ready": False,
            "status": "error",
            "error": str(stats.get("error", stats.get("infrastructure_error"))),
        })
    else:
        runtime.update({"ready": False, "status": "completed"})
    if runtime_callback is not None:
        runtime_callback(dict(runtime))
    stats["runtime"] = runtime
    stats["termination_mode"] = (
        "chess_rules" if max_moves is None else "diagnostic_ply_cap"
    )
    stats["diagnostic_cap_plies"] = (
        None if max_moves is None else max(1, int(max_moves))
    )
    stats["unresolved_score_policy"] = "excluded"
    stats.setdefault("terminal_reasons", {})
    # Match summaries must not turn capped/cancelled games into neutral scores.
    # Report the score over games with a real chess result, and retain unresolved
    # counts and the denominator as explicit metadata.
    resolved_games = (
        int(stats.get("wins", 0) or 0)
        + int(stats.get("draws", 0) or 0)
        + int(stats.get("losses", 0) or 0)
    )
    stats["resolved_games"] = resolved_games
    stats["score_rate_denominator"] = "resolved_games"
    if resolved_games:
        stats["score_rate"] = (
            int(stats.get("wins", 0) or 0)
            + 0.5 * int(stats.get("draws", 0) or 0)
        ) / resolved_games
        stats["win_rate"] = int(stats.get("wins", 0) or 0) / resolved_games
        stats["draw_rate"] = int(stats.get("draws", 0) or 0) / resolved_games
        stats["loss_rate"] = int(stats.get("losses", 0) or 0) / resolved_games
    else:
        stats["score_rate"] = None
        stats["win_rate"] = None
        stats["draw_rate"] = None
        stats["loss_rate"] = None
    return stats
