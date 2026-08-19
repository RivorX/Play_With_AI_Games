"""Multiprocessing workers and result streaming for RL self-play."""

import pickle
import random
import threading
import time

import numpy as np
import torch

from src.inference.central import (
    RemoteInferenceClient,
    _build_selfplay_worker_model,
    _central_inference_option,
    _configure_selfplay_worker_runtime,
    _load_worker_model_state,
)
from src.mcts.search import (
    _debug_nested,
    _maybe_compile_selfplay_model,
    _pack_positions_for_transfer,
    _resolve_dynamic_simulation_budget,
    _resolve_replay_max_policy_targets,
)
from src.selfplay.engine import SelfPlayEngine


def _seed_selfplay_worker(rank, task_seed):
    """Seed opening, Gumbel and framework RNGs for one persistent task."""
    worker_seed = int(task_seed) + int(rank) * 100_003
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2 ** 32))
    torch.manual_seed(worker_seed)
    # Workers are intentionally CPU-only. Calling a CUDA RNG API here may
    # initialize a CUDA context in every process and duplicate driver memory.
    return worker_seed


def _build_worker_logger(rank, worker_verbose):
    def _wlog(message):
        if worker_verbose:
            print(f"Worker {rank}: {message}")

    return _wlog


def _create_selfplay_engine(
    model,
    config,
    device,
    num_games,
    wlog,
    opponent_model=None,
    opponent_source_label="current",
    opponent_models_by_label=None,
    opponent_plan_labels=None,
):
    rl_cfg = config.get('reinforcement_learning', {})
    configured_max = max(
        1,
        int(rl_cfg.get('max_batch_games_per_worker', 256)),
    )

    engine = SelfPlayEngine(
        model,
        config,
        device,
        configured_max,
        opponent_model=opponent_model,
        opponent_source_label=opponent_source_label,
        opponent_models_by_label=opponent_models_by_label,
        opponent_plan_labels=opponent_plan_labels,
    )
    actual_max = min(configured_max, num_games)
    wlog(
        "Self-play engine: batch "
        f"(max {configured_max}, actual {actual_max})"
    )
    return engine


def _play_games_with_engine(
    rank,
    engine,
    config,
    num_games,
    result_file_path,
    wlog,
    result_queue=None,
    task_id=None,
    stream_results_to_queue=False,
    game_job_queue=None,
    initial_game_jobs=None,
):
    rl_cfg = config.get('reinforcement_learning', {})
    progress_file_path = result_file_path.replace('.pkl', '.progress')

    def _write_progress(n):
        try:
            with open(progress_file_path, 'w') as _pf:
                _pf.write(str(n))
        except Exception:
            pass

    if hasattr(engine, '_progress_file'):
        engine._progress_file = progress_file_path

    if bool(rl_cfg.get('mcts_dynamic_budget_enabled', False)):
        dynamic_min, dynamic_target, dynamic_max, _dynamic_chunk = (
            _resolve_dynamic_simulation_budget(
                rl_cfg.get('mcts_simulations', 192),
                minimum=rl_cfg.get('mcts_dynamic_budget_min', 64),
                maximum_multiplier=rl_cfg.get(
                    'mcts_dynamic_budget_max_multiplier',
                    5.0 / 3.0,
                ),
            )
        )
        search_budget_text = (
            f"dynamic {dynamic_min}-{dynamic_max}, avg {dynamic_target} sims/move"
        )
    else:
        search_budget_text = f"{int(rl_cfg.get('mcts_simulations', 0))} sims/move"
    tree_mode = (
        "shared tree + reuse"
        if bool(rl_cfg.get('self_play_share_trees', True))
        else "fresh tree after each move"
    )
    wlog(
        f"Playing {num_games} games with Gumbel AlphaZero "
        f"({search_budget_text}; {tree_mode})"
    )

    save_every = rl_cfg.get('self_play_save_every_games_resolved', None)
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
    if stream_results_to_queue:
        save_every = max(
            1,
            int(
                rl_cfg.get(
                    'self_play_queue_chunk_games',
                    rl_cfg.get('max_batch_games_per_worker', 1),
                )
            ),
        )
    elif save_every is None:
        save_mult = rl_cfg.get('self_play_save_every_games', 0)
        try:
            save_mult = float(save_mult)
        except Exception:
            save_mult = 0
        if save_mult and save_mult > 0:
            games_per_iter = rl_cfg.get('games_per_iteration', 0)
            try:
                games_per_iter = int(games_per_iter)
            except Exception:
                games_per_iter = 0
            save_every = max(1, int(round(games_per_iter * save_mult)))
        else:
            save_every = 0

    total_positions = 0
    total_games = 0
    total_dropped_positions = 0
    total_truncated_games = 0
    total_claimable_draw_ended_games = 0
    total_completed_length_sum = 0
    total_truncated_length_sum = 0

    if game_job_queue is not None:
        streamed_positions = 0
        streamed_games = 0

        def _emit_completed_chunk(chunk_positions, chunk_lengths, chunk_job_ids):
            nonlocal streamed_positions, streamed_games
            if len(chunk_job_ids) != len(chunk_lengths):
                raise RuntimeError(
                    "streamed self-play chunk lost its game identity: "
                    f"{len(chunk_job_ids)} ids for {len(chunk_lengths)} games"
                )
            streamed_positions += len(chunk_positions)
            streamed_games += len(chunk_lengths)
            if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(
                        chunk_positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
                    'game_lengths': list(chunk_lengths),
                    'game_job_ids': list(chunk_job_ids),
                    'stats': {},
                })

        positions, game_lengths = engine.play_games(
            0,
            game_job_queue=game_job_queue,
            stream_task_id=task_id,
            initial_game_jobs=initial_game_jobs,
            completed_chunk_callback=_emit_completed_chunk,
            completed_chunk_games=8,
        )
        stats = getattr(engine, 'last_selfplay_stats', {}) or {}
        total_positions = streamed_positions + len(positions)
        total_games = streamed_games + len(game_lengths)
        total_dropped_positions = int(stats.get('dropped_positions', 0))
        total_truncated_games = int(stats.get('truncated_games', 0))
        total_claimable_draw_ended_games = int(stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum = int(stats.get('completed_length_sum', 0))
        total_truncated_length_sum = int(stats.get('truncated_length_sum', 0))
        _write_progress(total_games)
        if stream_results_to_queue and result_queue is not None:
            result_queue.put({
                'type': 'payload',
                'rank': rank,
                'task_id': task_id,
                'positions': _pack_positions_for_transfer(
                    positions,
                    max_policy_targets=replay_max_policy_targets,
                ),
                'game_lengths': list(game_lengths),
                'game_job_ids': [],
                'stats': stats,
            })
        else:
            with open(result_file_path, 'wb') as f:
                pickle.dump((positions, game_lengths, stats), f)
    elif save_every and save_every > 0:
        if not stream_results_to_queue:
            with open(result_file_path, 'wb') as f:
                pass
        games_left = num_games
        while games_left > 0:
            chunk_games = min(save_every, games_left)
            if hasattr(engine, '_progress_file'):
                engine._progress_file = progress_file_path
            if hasattr(engine, '_progress_base'):
                engine._progress_base = total_games
            positions, game_lengths = engine.play_games(chunk_games)
            stats = getattr(engine, 'last_selfplay_stats', {}) or {}

            if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
                    'game_lengths': list(game_lengths),
                    'stats': stats,
                })
            else:
                with open(result_file_path, 'ab') as f:
                    pickle.dump((positions, game_lengths, stats), f)

            total_positions += len(positions)
            total_games += len(game_lengths)
            total_dropped_positions += int(stats.get('dropped_positions', 0))
            total_truncated_games += int(stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(stats.get('truncated_length_sum', 0))
            _write_progress(total_games)
            games_left -= chunk_games
    else:
        if hasattr(engine, '_progress_file'):
            engine._progress_file = progress_file_path
        positions, game_lengths = engine.play_games(num_games)
        stats = getattr(engine, 'last_selfplay_stats', {}) or {}
        total_positions = len(positions)
        total_games = len(game_lengths)
        total_dropped_positions = int(stats.get('dropped_positions', 0))
        total_truncated_games = int(stats.get('truncated_games', 0))
        total_claimable_draw_ended_games = int(stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum = int(stats.get('completed_length_sum', 0))
        total_truncated_length_sum = int(stats.get('truncated_length_sum', 0))
        _write_progress(total_games)

        if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
                    'game_lengths': list(game_lengths),
                    'stats': stats,
                })
        else:
            with open(result_file_path, 'wb') as f:
                pickle.dump((positions, game_lengths, stats), f)

    wlog(f"Generated {total_positions} positions from {total_games} games")
    if total_truncated_games > 0 or total_dropped_positions > 0:
        wlog(
            "Dropped due to truncation: "
            f"{total_dropped_positions} positions across {total_truncated_games} games"
        )
    if total_claimable_draw_ended_games > 0:
        wlog(f"Claimable-draw ended games: {total_claimable_draw_ended_games}")
    if total_games > 0:
        def _format_plies(value):
            return f"{value:.1f} plies (~{value / 2.0:.1f} full moves)"
        completed_games = max(0, total_games - total_truncated_games)
        if completed_games > 0:
            wlog(
                f"Avg completed game length: {_format_plies(total_completed_length_sum / completed_games)}"
            )
        if total_truncated_games > 0:
            wlog(
                f"Avg truncated game length: {_format_plies(total_truncated_length_sum / total_truncated_games)}"
            )
    wlog(f"Saved to {result_file_path}")
    return total_positions, total_games


def persistent_selfplay_worker(
    rank,
    config,
    device_id,
    task_queue,
    result_queue,
    game_queue=None,
    inference_request_queue=None,
    inference_response_queue=None,
    inference_shared_buffer=None,
):
    """
    Persistent worker for self-play on Windows.

    The process is spawned once, then receives lightweight play tasks so we avoid
    repeated interpreter startup, imports, model construction and argument pickling.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)
    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Persistent worker started on {device}")

        central_inference_enabled = inference_request_queue is not None and inference_response_queue is not None
        _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
        central_debug_enabled = bool(
            debug_root_enabled and central_debug_cfg.get(
                'verbose',
                rl_cfg.get('self_play_central_inference_debug', False),
            )
        )
        central_stall_warning_s = float(
            central_debug_cfg.get(
                'stall_warning_s',
                _central_inference_option(config, 'stall_warning_s', 60.0),
            )
        )
        central_transport_dtype = str(
            _central_inference_option(config, 'transport_dtype', 'float16') or 'float16'
        )
        shared_inference_call_lock = threading.Lock()
        model = None if central_inference_enabled else _build_selfplay_worker_model(config, device)
        inference_model = (
            RemoteInferenceClient(
                "learner",
                inference_request_queue,
                inference_response_queue,
                worker_rank=rank,
                timeout_s=float(_central_inference_option(config, 'timeout_s', 0.0)),
                stall_warning_s=central_stall_warning_s,
                debug_enabled=central_debug_enabled,
                transport_dtype=central_transport_dtype,
                shared_buffer=inference_shared_buffer,
                shared_call_lock=shared_inference_call_lock,
            )
            if central_inference_enabled
            else _maybe_compile_selfplay_model(
                model,
                config,
                device,
                rank,
                model_label="learner",
            )
        )
        opponent_model = None
        engine = None
        engine_signature = None

        while True:
            task = task_queue.get()
            if task is None or task.get('cmd') == 'stop':
                wlog("Stopping persistent worker")
                break

            result_file_path = task['result_file_path']
            try:
                runtime_overrides = dict(task.get('rl_runtime_overrides') or {})
                runtime_overrides['hard_start_positions'] = list(
                    task.get('hard_start_positions', []) or []
                )
                if runtime_overrides:
                    config.setdefault('reinforcement_learning', {}).update({
                        key: value
                        for key, value in runtime_overrides.items()
                        if value is not None
                    })
                    rl_cfg = config.get('reinforcement_learning', {})
                worker_seed = _seed_selfplay_worker(
                    rank,
                    runtime_overrides.get(
                        'self_play_task_seed',
                        config.get('seed', 490050),
                    ),
                )
                wlog(f"RNG seed for task: {worker_seed}")
                if central_inference_enabled and central_debug_enabled:
                    opponent_payload_preview = task.get('opponent_payload') or {}
                    preview_labels = sorted({
                        str((entry or {}).get('label') or 'current')
                        for entry in list(opponent_payload_preview.get('pool_entries', []) or [])
                    })
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Worker {rank}: central task started "
                        f"games={int(task['num_games'])}, opponent_models={preview_labels or ['current']}.",
                        flush=True,
                    )
                if not central_inference_enabled:
                    if task.get('model_state') is not None:
                        model_state = task['model_state']
                    else:
                        model_state = torch.load(task['model_state_path'], map_location='cpu')
                    _load_worker_model_state(model, model_state, rank)
                    model.eval()

                opponent_payload = task.get('opponent_payload') or {}
                opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
                opponent_pool_entries = list(opponent_payload.get('pool_entries', []) or [])
                opponent_models_by_label = {}
                opponent_model = None
                opponent_label = str(opponent_payload.get('label') or 'current')
                for entry in opponent_pool_entries:
                    entry_label = str((entry or {}).get('label') or 'current')
                    entry_state = (entry or {}).get('state')
                    if entry_label == 'current':
                        continue
                    if central_inference_enabled:
                        pooled_model = RemoteInferenceClient(
                            entry_label,
                            inference_request_queue,
                            inference_response_queue,
                            worker_rank=rank,
                            timeout_s=float(_central_inference_option(config, 'timeout_s', 0.0)),
                            stall_warning_s=central_stall_warning_s,
                            debug_enabled=central_debug_enabled,
                            transport_dtype=central_transport_dtype,
                            shared_buffer=inference_shared_buffer,
                            shared_call_lock=shared_inference_call_lock,
                        )
                    else:
                        if entry_state is None:
                            continue
                        pooled_model = _build_selfplay_worker_model(config, device)
                        _load_worker_model_state(pooled_model, entry_state, rank)
                        pooled_model.eval()
                        pooled_model = _maybe_compile_selfplay_model(
                            pooled_model,
                            config,
                            device,
                            rank,
                            model_label=f"opponent:{entry_label}",
                        )
                    opponent_models_by_label[entry_label] = pooled_model
                    if opponent_model is None:
                        opponent_model = pooled_model
                    if opponent_label == 'current':
                        opponent_label = entry_label

                engine = _create_selfplay_engine(
                    inference_model,
                    config,
                    device,
                    int(task['num_games']),
                    wlog,
                    opponent_model=opponent_model,
                    opponent_source_label=opponent_label,
                    opponent_models_by_label=opponent_models_by_label,
                    opponent_plan_labels=opponent_plan_labels,
                )
                engine_signature = None

                if hasattr(engine, 'temperature') and task.get('mcts_temperature') is not None:
                    engine.temperature = float(task['mcts_temperature'])
                total_positions, total_games = _play_games_with_engine(
                    rank,
                    engine,
                    config,
                    int(task['num_games']),
                    result_file_path,
                    wlog,
                    result_queue=result_queue,
                    task_id=task.get('task_id'),
                    stream_results_to_queue=bool(task.get('stream_results_to_queue', False)),
                    game_job_queue=(
                        game_queue if bool(task.get('stream_game_queue', False)) else None
                    ),
                    initial_game_jobs=list(task.get('initial_game_jobs', []) or []),
                )
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task['task_id'],
                    'ok': True,
                    'positions': total_positions,
                    'games': total_games,
                })
            except KeyboardInterrupt:
                try:
                    with open(result_file_path, 'wb') as f:
                        pickle.dump(([], []), f)
                finally:
                    result_queue.put({
                        'type': 'result',
                        'rank': rank,
                        'task_id': task.get('task_id'),
                        'ok': False,
                        'interrupt': True,
                    })
                    raise SystemExit(130)
            except Exception as e:
                print(f"ERROR: Worker {rank} failed: {e}")
                import traceback
                traceback.print_exc()
                with open(result_file_path, 'wb') as f:
                    pickle.dump(([], []), f)
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task.get('task_id'),
                    'ok': False,
                    'error': str(e),
                })
    except KeyboardInterrupt:
        raise SystemExit(130)


def run_selfplay_worker(
    rank,
    model_state,
    config,
    device_id,
    num_games,
    result_file_path,
    opponent_payload=None,
):
    """
    One-shot worker function for parallel MCTS self-play.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        worker_seed = _seed_selfplay_worker(
            rank,
            rl_cfg.get('self_play_task_seed', config.get('seed', 490050)),
        )
        wlog(f"RNG seed for task: {worker_seed}")
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Starting on {device}")

        if isinstance(model_state, (str, bytes)):
            model_state = torch.load(model_state, map_location='cpu')

        model = _build_selfplay_worker_model(config, device)
        _load_worker_model_state(model, model_state, rank)
        model = _maybe_compile_selfplay_model(
            model,
            config,
            device,
            rank,
            model_label="learner",
        )
        opponent_model = None
        opponent_label = 'current'
        opponent_models_by_label = {}
        opponent_plan_labels = []
        if isinstance(opponent_payload, dict):
            opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
            for entry in list(opponent_payload.get('pool_entries', []) or []):
                entry_label = str((entry or {}).get('label') or 'current')
                entry_state = (entry or {}).get('state')
                if entry_state is None or entry_label == 'current':
                    continue
                pooled_model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(pooled_model, entry_state, rank)
                pooled_model.eval()
                pooled_model = _maybe_compile_selfplay_model(
                    pooled_model,
                    config,
                    device,
                    rank,
                    model_label=f"opponent:{entry_label}",
                )
                opponent_models_by_label[entry_label] = pooled_model
                if opponent_model is None:
                    opponent_model = pooled_model
                    opponent_label = entry_label

        engine = _create_selfplay_engine(
            model,
            config,
            device,
            num_games,
            wlog,
            opponent_model=opponent_model,
            opponent_source_label=opponent_label,
            opponent_models_by_label=opponent_models_by_label,
            opponent_plan_labels=opponent_plan_labels,
        )
        _play_games_with_engine(rank, engine, config, num_games, result_file_path, wlog)
    except KeyboardInterrupt:
        try:
            with open(result_file_path, 'wb') as f:
                pickle.dump(([], []), f)
        finally:
            raise SystemExit(130)
    except Exception as e:
        print(f"ERROR: Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        with open(result_file_path, 'wb') as f:
            pickle.dump(([], []), f)
