"""
Reinforcement learning training helpers.
"""

import os
import sys
import contextlib
from pathlib import Path

import chess
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path for src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.batch_selfplay import (
    MultiGameBatchMCTS,
    select_move_by_visits,
    _SELFPLAY_OPENING_LINES,
    _RemoteInferenceModel,
    central_inference_server,
)
from src.data import board_to_tensor, move_to_index
from src.model import ChessNet
from src.utils.data_helpers import ACTION_SIZE, build_hflip_inverse_index_map


_HFLIP_INV_INDEX_MAP = None
_HFLIP_FWD_INDEX_MAP = None


def _snapshot_state_dict_cpu_shared(model):
    snapshot = {}
    for key, tensor in model.state_dict().items():
        cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
        cpu_tensor.share_memory_()
        snapshot[key] = cpu_tensor
    return snapshot


def _eval_uses_central_inference(config, device):
    rl_cfg = config.get("reinforcement_learning", {})
    enabled = rl_cfg.get(
        "eval_central_inference_enabled",
        rl_cfg.get("self_play_central_inference_enabled", False),
    )
    return bool(enabled and device.type == "cuda" and torch.cuda.is_available())


def _resolve_configured_self_play_workers(rl_cfg, cpu_budget):
    cpu_budget = max(1, int(cpu_budget))
    raw_workers = rl_cfg.get("self_play_workers", cpu_budget)
    if isinstance(raw_workers, str) and raw_workers.strip().lower() in {"auto", "automatic"}:
        try:
            multiplier = max(0.10, float(rl_cfg.get("self_play_worker_auto_multiplier", 1.0)))
        except Exception:
            multiplier = 1.0
        return max(1, int(np.ceil(float(cpu_budget) * multiplier)))
    try:
        workers = int(raw_workers)
    except Exception:
        workers = cpu_budget
    if workers <= 0:
        try:
            multiplier = max(0.10, float(rl_cfg.get("self_play_worker_auto_multiplier", 1.0)))
        except Exception:
            multiplier = 1.0
        return max(1, int(np.ceil(float(cpu_budget) * multiplier)))
    return max(1, workers)


def _resolve_eval_workers(config, device, num_games):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_workers = rl_cfg.get("eval_workers", None)
    auto_workers = raw_workers is None or str(raw_workers).strip().lower() in {"auto", "automatic", "0"}

    if auto_workers:
        if _eval_uses_central_inference(config, device):
            # CPU workers own MCTS trees, central GPU server owns inference.
            # This mirrors RL self-play and keeps eval from becoming one-core.
            reserve_threads = max(0, int(rl_cfg.get("eval_cpu_threads_to_reserve", 0) or 0))
            cpu_budget = max(1, (os.cpu_count() or 2) - reserve_threads)
            raw_workers = min(
                _resolve_configured_self_play_workers(rl_cfg, cpu_budget),
                cpu_budget,
            )
        elif device.type == "cuda":
            # Without central inference each eval worker would load full CUDA
            # models, so keep the old memory-safe default.
            raw_workers = 1
        else:
            raw_workers = max(1, (os.cpu_count() or 2) - 1)

    try:
        workers = int(raw_workers)
    except Exception:
        workers = 1

    if auto_workers and _eval_uses_central_inference(config, device):
        try:
            min_games_per_worker = max(1, int(rl_cfg.get("eval_min_games_per_worker", 1) or 1))
        except Exception:
            min_games_per_worker = 1
        if min_games_per_worker > 1:
            workers = min(workers, max(1, int(np.ceil(float(num_games) / float(min_games_per_worker)))))

    return max(1, min(int(num_games), workers))


def _build_eval_central_server_config(config):
    """Project eval-specific central inference knobs onto the shared server keys."""
    server_config = dict(config)
    rl_cfg = dict(config.get("reinforcement_learning", {}))
    mappings = {
        "eval_central_inference_flush_ms": "self_play_central_inference_flush_ms",
        "eval_central_inference_max_batch_size": "self_play_central_inference_max_batch_size",
        "eval_central_inference_transport_dtype": "self_play_central_inference_transport_dtype",
        "eval_central_inference_use_compile": "self_play_central_inference_use_compile",
        "eval_central_inference_compile_warmup_batches": "self_play_central_inference_compile_warmup_batches",
        "eval_central_inference_cudnn_benchmark": "self_play_central_inference_cudnn_benchmark",
        "eval_central_inference_cache_enabled": "self_play_central_inference_cache_enabled",
        "eval_central_inference_cache_entries": "self_play_central_inference_cache_entries",
    }
    for eval_key, server_key in mappings.items():
        if eval_key in rl_cfg:
            rl_cfg[server_key] = rl_cfg[eval_key]
    server_config["reinforcement_learning"] = rl_cfg
    return server_config


def _build_eval_mcts_config(config):
    """Build deterministic full-budget eval MCTS settings."""
    eval_config = dict(config)
    rl_cfg = dict(config.get("reinforcement_learning", {}))
    eval_simulations = max(1, int(rl_cfg.get("mcts_simulations", 1) or 1))
    rl_cfg["mcts_scout_simulations"] = eval_simulations
    rl_cfg["mcts_scout_challenge_fraction"] = 0.0
    eval_config["reinforcement_learning"] = rl_cfg
    return eval_config


def _resolve_eval_central_server_count(config, workers):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get("eval_central_inference_servers", "auto")
    if str(raw_value).strip().lower() not in {"auto", "automatic"}:
        try:
            return max(1, int(raw_value))
        except Exception:
            return 1

    workers = max(1, int(workers))
    target_workers = max(3, int(rl_cfg.get("eval_central_inference_auto_workers_per_server", 6) or 6))
    min_servers = max(1, int(rl_cfg.get("eval_central_inference_auto_min_servers", 1) or 1))
    max_servers = max(min_servers, int(rl_cfg.get("eval_central_inference_auto_max_servers", 3) or 3))
    by_workers = max(1, (workers + target_workers - 1) // target_workers)
    by_vram = max_servers
    try:
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / float(1024 ** 3)
        if total_gib < 10.0:
            by_vram = 1
        elif total_gib < 14.0:
            by_vram = min(by_vram, 2)
        elif total_gib < 24.0:
            by_vram = min(by_vram, 3)
    except Exception:
        by_vram = min(by_vram, 2)
    return max(1, min(max(min_servers, by_workers), max_servers, by_vram))


def _resolve_eval_batch_games(config, num_games):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get(
        "eval_batch_games",
        rl_cfg.get("max_batch_games_per_worker", 16),
    )
    try:
        value = int(raw_value)
    except Exception:
        value = 16
    return max(1, min(int(num_games), value))


def _terminate_eval_processes(processes, timeout_s=0.5):
    for proc in list(processes or []):
        with contextlib.suppress(Exception):
            if proc is None or not proc.is_alive():
                continue
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.join(timeout=max(0.0, float(timeout_s)))
        with contextlib.suppress(Exception):
            if proc is not None and proc.is_alive():
                proc.kill()
        with contextlib.suppress(Exception):
            proc.join(timeout=0.2)


def _resolve_eval_max_moves(config):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get("eval_max_moves", 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _resolve_eval_auto_claim_draw(config):
    rl_cfg = config.get("reinforcement_learning", {})
    return bool(rl_cfg.get("eval_auto_claim_draw", False))


def _resolve_eval_claim_draw_after_moves(config):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get("eval_claim_draw_after_moves", _resolve_eval_max_moves(config))
    try:
        return max(0, int(raw_value))
    except Exception:
        return _resolve_eval_max_moves(config)


def _resolve_eval_claim_repetition_after_moves(config):
    rl_cfg = config.get("reinforcement_learning", {})
    default_value = min(_resolve_eval_claim_draw_after_moves(config), 80)
    raw_value = rl_cfg.get("eval_claim_repetition_after_moves", default_value)
    try:
        return max(0, int(raw_value))
    except Exception:
        return default_value


def _resolve_eval_fixed_openings_enabled(config):
    rl_cfg = config.get("reinforcement_learning", {})
    return bool(rl_cfg.get("eval_fixed_openings_enabled", False))


def _resolve_eval_fixed_openings_max_plies(config):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get("eval_fixed_openings_max_plies", 6)
    try:
        return max(0, int(raw_value))
    except Exception:
        return 6


def _resolve_eval_mcts_simulations(config):
    rl_cfg = config.get("reinforcement_learning", {})
    try:
        base_sims = max(1, int(rl_cfg.get("mcts_simulations", 50)))
    except Exception:
        base_sims = 50

    try:
        multiplier = max(0.01, float(rl_cfg.get("eval_mcts_simulations_multiplier", 1.0)))
        return max(1, int(round(base_sims * multiplier)))
    except Exception:
        return base_sims


def _build_eval_stats(wins, draws, losses, unresolved, num_games):
    num_games = int(num_games)
    wins = int(wins)
    draws = int(draws)
    losses = int(losses)
    unresolved = int(unresolved)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unresolved": unresolved,
        "num_games": num_games,
        "score_rate": (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0,
        "win_rate": wins / num_games if num_games > 0 else 0.0,
        "draw_rate": draws / num_games if num_games > 0 else 0.0,
        "loss_rate": losses / num_games if num_games > 0 else 0.0,
        "resolved_games": num_games - unresolved,
    }


def _print_eval_unresolved(label, unresolved, num_games, max_moves):
    if int(unresolved) <= 0:
        return
    print(
        f"{label} unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
        f"{unresolved}/{num_games} -> excluded from draw count"
    )


def _get_eval_opening_prefix(config, game_idx, enabled_override=None):
    enabled = _resolve_eval_fixed_openings_enabled(config) if enabled_override is None else bool(enabled_override)
    if not enabled or not _SELFPLAY_OPENING_LINES:
        return ()

    opening_idx = int(game_idx)
    if bool(config.get("reinforcement_learning", {}).get("eval_fixed_openings_pair_games", False)):
        opening_idx //= 2
    line = _SELFPLAY_OPENING_LINES[opening_idx % len(_SELFPLAY_OPENING_LINES)]
    max_plies = _resolve_eval_fixed_openings_max_plies(config)
    if max_plies <= 0:
        return ()
    return tuple(line[: min(len(line), max_plies)])


def _apply_opening_prefix_for_eval(board, opening_prefix, mcts_white, mcts_black):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if board.is_game_over(claim_draw=False):
            break
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            break
        if move not in board.legal_moves:
            break

        mcts_white.update_history(board)
        mcts_black.update_history(board)
        mcts_white.advance_root(move)
        mcts_black.advance_root(move)
        board.push(move)
        applied += 1
    return applied


def _append_eval_history(board_history, board, config):
    board_history.append(_encode_eval_history_entry(board))
    max_history = int(config.get("model", {}).get("history_positions", 0) or 0) + 10
    if len(board_history) > max_history:
        del board_history[:-max_history]


def _apply_opening_prefix_for_batched_eval(board, board_history, opening_prefix, config):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if board.is_game_over(claim_draw=False):
            break
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            break
        if move not in board.legal_moves:
            break

        _append_eval_history(board_history, board, config)
        board.push(move)
        applied += 1
    return applied


def _encode_eval_history_entry(board):
    return (
        board_to_tensor(board, flip_perspective=False),
        board_to_tensor(board, flip_perspective=True),
    )


def _build_no_mcts_eval_input(board, board_history, config):
    history_positions = int(config.get("model", {}).get("history_positions", 0) or 0)
    current_tensor = board_to_tensor(board)
    if history_positions <= 0:
        return current_tensor

    use_black_pov = (board.turn == chess.BLACK)
    history_slice = list(board_history[-history_positions:]) if board_history else []
    history_tensors = [
        encoded[1] if use_black_pov else encoded[0]
        for encoded in history_slice
    ]
    pad_count = max(0, history_positions - len(history_tensors))
    if pad_count:
        empty = np.zeros((16, 8, 8), dtype=np.float32)
        history_tensors = [empty] * pad_count + history_tensors
    history_tensors.append(current_tensor)
    return np.concatenate(history_tensors, axis=0)


def _select_no_mcts_policy_move(model, board, board_history, config, device):
    legal_moves = tuple(board.legal_moves)
    if not legal_moves:
        return None

    board_np = _build_no_mcts_eval_input(board, board_history, config)
    board_tensor = torch.from_numpy(board_np).unsqueeze(0).to(
        device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
        non_blocking=True,
    )
    legal_indices = torch.tensor(
        [move_to_index(move, board) for move in legal_moves],
        dtype=torch.long,
        device=device,
    )
    use_amp = bool(config.get("hardware", {}).get("use_amp", False) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if config.get("hardware", {}).get("use_bfloat16", False) else torch.float16
    with torch.inference_mode():
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            policy_logits, _value = model(board_tensor, apply_log_softmax=False, policy_only=True)
        legal_logits = policy_logits[0].index_select(0, legal_indices)
        best_idx = int(torch.argmax(legal_logits).item())
    return legal_moves[best_idx]


def _apply_opening_prefix_for_no_mcts_eval(board, board_history, opening_prefix):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if board.is_game_over(claim_draw=False):
            break
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            break
        if move not in board.legal_moves:
            break
        board_history.append(_encode_eval_history_entry(board))
        board.push(move)
        applied += 1
    return applied


def _evaluate_single_game_no_mcts(
    model_white,
    model_black,
    config,
    device,
    max_moves,
    auto_claim_draw,
    claim_draw_after_moves,
    claim_repetition_after_moves=0,
    opening_prefix=(),
):
    board = chess.Board()
    board_history = []
    move_count = _apply_opening_prefix_for_no_mcts_eval(board, board_history, opening_prefix)
    ended_by_auto_claim_draw = False

    while move_count < max_moves:
        if board.is_game_over(claim_draw=False):
            break

        model = model_white if board.turn == chess.WHITE else model_black
        move = _select_no_mcts_policy_move(model, board, board_history, config, device)
        if move is None:
            break

        board_history.append(_encode_eval_history_entry(board))
        board.push(move)
        move_count += 1

        if auto_claim_draw:
            try:
                if move_count >= claim_repetition_after_moves:
                    claim_threefold = getattr(board, "can_claim_threefold_repetition", None)
                    if callable(claim_threefold) and bool(claim_threefold()):
                        ended_by_auto_claim_draw = True
                        break
                if move_count >= claim_draw_after_moves and board.can_claim_draw():
                    ended_by_auto_claim_draw = True
                    break
            except Exception:
                pass

    if ended_by_auto_claim_draw:
        return "1/2-1/2", False

    result = board.result(claim_draw=False)
    return result, (result == "*")


def _evaluate_single_game(
    mcts_white,
    mcts_black,
    sims,
    max_moves,
    auto_claim_draw,
    claim_draw_after_moves,
    claim_repetition_after_moves=0,
    opening_prefix=(),
):
    board = chess.Board()
    move_count = _apply_opening_prefix_for_eval(board, opening_prefix, mcts_white, mcts_black)
    ended_by_auto_claim_draw = False

    while move_count < max_moves:
        if board.is_game_over(claim_draw=False):
            break

        mcts = mcts_white if board.turn == chess.WHITE else mcts_black
        visit_counts = mcts.search(board, num_simulations=sims, add_root_noise=False)
        move, _ = select_move_by_visits(visit_counts, temperature=0)

        mcts_white.update_history(board)
        mcts_black.update_history(board)
        mcts_white.advance_root(move)
        mcts_black.advance_root(move)

        board.push(move)
        move_count += 1

        if auto_claim_draw:
            try:
                if move_count >= claim_repetition_after_moves:
                    claim_threefold = getattr(board, "can_claim_threefold_repetition", None)
                    if callable(claim_threefold) and bool(claim_threefold()):
                        ended_by_auto_claim_draw = True
                        break
                if move_count >= claim_draw_after_moves and board.can_claim_draw():
                    ended_by_auto_claim_draw = True
                    break
            except Exception:
                pass

    mcts_white.reset_tree()
    mcts_black.reset_tree()

    if ended_by_auto_claim_draw:
        return "1/2-1/2", False

    result = board.result(claim_draw=False)
    return result, (result == "*")


def _advance_eval_root(root, move):
    if root is None:
        return None, False
    child = root.get_child_for_move(move)
    if child is None:
        return None, False
    _ = child.board
    return child.detach_as_root(), True


def _evaluate_games_batched(
    model1,
    model2,
    config,
    device,
    game_indices,
    use_fixed_openings=None,
    progress_callback=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
):
    game_indices = list(game_indices or [])
    if not game_indices:
        return _build_eval_stats(0, 0, 0, 0, 0)

    mcts1 = MultiGameBatchMCTS(model1, model1_mcts_config or config, device)
    mcts2 = MultiGameBatchMCTS(model2, model2_mcts_config or config, device)
    sims = _resolve_eval_mcts_simulations(config)
    max_moves = _resolve_eval_max_moves(config)
    auto_claim_draw = _resolve_eval_auto_claim_draw(config)
    claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(config)
    claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(config)
    active_limit = _resolve_eval_batch_games(config, len(game_indices))

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    cursor = 0
    active_games = []

    def _new_game_state(game_idx):
        board = chess.Board()
        board_history = []
        opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
        move_count = _apply_opening_prefix_for_batched_eval(board, board_history, opening_prefix, config)
        return {
            "game_idx": int(game_idx),
            "board": board,
            "board_history": board_history,
            "move_count": int(move_count),
            "model1_as_white": bool(int(game_idx) % 2 == 0),
            "model1_root": None,
            "model1_synced": False,
            "model2_root": None,
            "model2_synced": False,
            "done": bool(board.is_game_over(claim_draw=False) or move_count >= max_moves),
            "auto_claim_draw": False,
        }

    def _fill_active():
        nonlocal cursor
        while cursor < len(game_indices) and len(active_games) < active_limit:
            active_games.append(_new_game_state(game_indices[cursor]))
            cursor += 1

    def _finish_game(gs):
        nonlocal wins, draws, losses, unresolved, completed
        board = gs["board"]
        if bool(gs.get("auto_claim_draw", False)):
            result = "1/2-1/2"
            was_unresolved = False
        else:
            result = board.result(claim_draw=False)
            was_unresolved = (result == "*")

        if was_unresolved:
            unresolved += 1
            game_wins, game_draws, game_losses, game_unresolved = 0, 0, 0, 1
        else:
            game_wins, game_draws, game_losses = _result_for_model1(
                bool(gs.get("model1_as_white", False)),
                result,
            )
            wins += game_wins
            draws += game_draws
            losses += game_losses
            game_unresolved = 0
        completed += 1
        if progress_callback is not None:
            progress_callback(
                1,
                int(gs.get("game_idx", -1)),
                int(game_wins),
                int(game_draws),
                int(game_losses),
                int(game_unresolved),
            )

    def _claim_draw_if_needed(gs):
        if not auto_claim_draw:
            return False
        board = gs["board"]
        try:
            if gs["move_count"] >= claim_repetition_after_moves:
                claim_threefold = getattr(board, "can_claim_threefold_repetition", None)
                if callable(claim_threefold) and bool(claim_threefold()):
                    gs["auto_claim_draw"] = True
                    return True
            if gs["move_count"] >= claim_draw_after_moves and board.can_claim_draw():
                gs["auto_claim_draw"] = True
                return True
        except Exception:
            return False
        return False

    _fill_active()
    while active_games:
        model1_indices = []
        model2_indices = []
        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            if board.is_game_over(claim_draw=False) or gs["move_count"] >= max_moves:
                gs["done"] = True
                continue
            model1_turn = bool(board.turn == chess.WHITE) == bool(gs["model1_as_white"])
            if model1_turn:
                model1_indices.append(idx)
            else:
                model2_indices.append(idx)

        visit_counts_by_index = {}

        def _run_group(indices, mcts, root_key, synced_key):
            if not indices:
                return
            group_states = []
            for idx in indices:
                gs = active_games[idx]
                group_states.append([
                    gs["board"],
                    gs.get(root_key),
                    bool(gs.get(synced_key, False)),
                    gs["board_history"],
                    int(gs.get("move_count", 0) or 0),
                ])
            visit_counts_group = mcts.search_many(
                group_states,
                num_simulations=sims,
                add_root_noise=False,
            )
            for idx, local_state, visit_counts in zip(indices, group_states, visit_counts_group):
                gs = active_games[idx]
                gs[root_key] = local_state[1]
                gs[synced_key] = bool(local_state[2])
                visit_counts_by_index[idx] = visit_counts

        _run_group(model1_indices, mcts1, "model1_root", "model1_synced")
        _run_group(model2_indices, mcts2, "model2_root", "model2_synced")

        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            visit_counts = visit_counts_by_index.get(idx)
            if not visit_counts:
                gs["done"] = True
                continue
            move, _ = select_move_by_visits(visit_counts, temperature=0)

            _append_eval_history(gs["board_history"], board, config)
            gs["model1_root"], gs["model1_synced"] = _advance_eval_root(gs.get("model1_root"), move)
            gs["model2_root"], gs["model2_synced"] = _advance_eval_root(gs.get("model2_root"), move)
            board.push(move)
            gs["move_count"] += 1

            if (
                board.is_game_over(claim_draw=False)
                or gs["move_count"] >= max_moves
                or _claim_draw_if_needed(gs)
            ):
                gs["done"] = True

        next_active = []
        for gs in active_games:
            if gs["done"]:
                _finish_game(gs)
            else:
                next_active.append(gs)
        active_games = next_active
        _fill_active()

    return _build_eval_stats(wins, draws, losses, unresolved, len(game_indices))


def _result_for_model1(model1_as_white, result):
    if result == "1/2-1/2":
        return 0, 1, 0
    if result == "1-0":
        return (1, 0, 0) if model1_as_white else (0, 0, 1)
    if result == "0-1":
        return (1, 0, 0) if not model1_as_white else (0, 0, 1)
    return 0, 0, 0


def _eval_worker(
    rank,
    model1_state,
    model2_state,
    config,
    device_str,
    game_indices,
    result_queue,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
):
    try:
        worker_config = dict(config)
        worker_config["model"] = dict(config.get("model", {}))
        worker_config["model"]["print_summary"] = False

        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        model1 = ChessNet(worker_config).to(device)
        model2 = ChessNet(worker_config).to(device)
        if device.type == "cuda":
            model1 = model1.to(memory_format=torch.channels_last)
            model2 = model2.to(memory_format=torch.channels_last)

        model1.load_state_dict(model1_state)
        model2.load_state_dict(model2_state)
        model1.eval()
        model2.eval()

        stats = _evaluate_games_batched(
            model1,
            model2,
            worker_config,
            device,
            game_indices,
            use_fixed_openings=use_fixed_openings,
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
            }),
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


def _eval_central_worker(
    rank,
    config,
    game_indices,
    request_queue,
    response_receiver,
    result_queue,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
):
    try:
        rl_cfg = config.get("reinforcement_learning", {})
        torch_threads = max(1, int(rl_cfg.get("eval_torch_threads", rl_cfg.get("self_play_torch_threads", 1)) or 1))
        with contextlib.suppress(Exception):
            torch.set_num_threads(torch_threads)
        with contextlib.suppress(Exception):
            torch.set_num_interop_threads(1)

        timeout_s = float(rl_cfg.get(
            "eval_central_inference_timeout_s",
            rl_cfg.get("self_play_central_inference_timeout_s", 0),
        ) or 0)
        stall_warning_s = float(rl_cfg.get(
            "eval_central_inference_stall_warning_s",
            rl_cfg.get("self_play_central_inference_stall_warning_s", 15),
        ) or 0)
        transport_dtype = str(rl_cfg.get(
            "eval_central_inference_transport_dtype",
            rl_cfg.get("self_play_central_inference_transport_dtype", "float16"),
        ) or "float16")
        debug_enabled = bool(rl_cfg.get("eval_central_inference_debug", False))

        model1 = _RemoteInferenceModel(
            "eval_model1",
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
        )
        model2 = _RemoteInferenceModel(
            "eval_model2",
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
        )

        stats = _evaluate_games_batched(
            model1,
            model2,
            config,
            torch.device("cpu"),
            game_indices,
            use_fixed_openings=use_fixed_openings,
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
            }),
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


def _evaluate_models_with_central_inference(
    model1,
    model2,
    config,
    device,
    num_games,
    game_index_offset=0,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    progress_desc="Eval vs best",
):
    workers = _resolve_eval_workers(config, device, num_games)
    server_count = _resolve_eval_central_server_count(config, workers)
    rl_cfg = config.get("reinforcement_learning", {})
    server_config = _build_eval_central_server_config(config)
    max_moves = _resolve_eval_max_moves(config)
    ctx = mp.get_context("spawn")

    result_queue = ctx.Queue()
    request_queues = [ctx.Queue() for _ in range(server_count)]
    control_queues = [ctx.Queue() for _ in range(server_count)]
    server_response_senders = [dict() for _ in range(server_count)]
    worker_response_receivers = {}
    worker_server_idx = {}

    game_indices_per_worker = [[] for _ in range(workers)]
    for offset, game_idx in enumerate(range(game_index_offset, game_index_offset + num_games)):
        game_indices_per_worker[offset % workers].append(game_idx)

    active_worker_ranks = [
        rank for rank, game_indices in enumerate(game_indices_per_worker)
        if game_indices
    ]
    for rank in active_worker_ranks:
        server_idx = int(rank) % int(server_count)
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        worker_response_receivers[rank] = recv_conn
        server_response_senders[server_idx][rank] = send_conn
        worker_server_idx[rank] = server_idx

    model1_state = _snapshot_state_dict_cpu_shared(model1)
    model2_state = _snapshot_state_dict_cpu_shared(model2)

    server_processes = []
    worker_processes = []
    task_id = f"eval_{os.getpid()}_{id(model1)}_{game_index_offset}_{num_games}"
    try:
        for server_idx in range(server_count):
            proc = ctx.Process(
                target=central_inference_server,
                args=(
                    server_config,
                    0,
                    request_queues[server_idx],
                    server_response_senders[server_idx],
                    control_queues[server_idx],
                    9100 + (int(os.getpid()) % 100000) * 10 + int(server_idx),
                ),
            )
            proc.daemon = True
            proc.start()
            server_processes.append(proc)

        load_timeout_s = float(rl_cfg.get(
            "eval_central_inference_load_timeout_s",
            rl_cfg.get("self_play_central_inference_load_timeout_s", 300),
        ) or 300)
        for request_queue in request_queues:
            request_queue.put({
                "cmd": "load_models",
                "task_id": task_id,
                "clear": True,
                "models": [
                    {"label": "eval_model1", "state": model1_state, "state_path": None},
                    {"label": "eval_model2", "state": model2_state, "state_path": None},
                ],
            })
        import time
        deadline = time.time() + max(1.0, load_timeout_s)
        pending_servers = set(range(server_count))
        load_messages = []
        while pending_servers:
            if time.time() >= deadline:
                raise TimeoutError(
                    "Eval central inference did not acknowledge model load "
                    f"from servers {sorted(pending_servers)}."
                )
            for server_idx in list(pending_servers):
                try:
                    message = control_queues[server_idx].get(timeout=0.25)
                except Exception:
                    continue
                if message.get("type") == "models_loaded" and str(message.get("task_id")) == task_id:
                    load_messages.append(dict(message))
                    pending_servers.discard(server_idx)

        load_times = [
            float(message.get("load_s", 0.0) or 0.0)
            for message in load_messages
            if message.get("load_s") is not None
        ]
        model_summary = next(
            (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
            "models ready",
        )
        pid_summary = ",".join(
            str(int(message.get("pid")))
            for message in load_messages
            if message.get("pid") is not None
        )
        load_summary = f", {model_summary}"
        if load_times:
            load_summary += f", load={max(load_times):.2f}s"
        if pid_summary:
            load_summary += f", pids={pid_summary}"
        print(
            "Eval central inference: "
            f"workers={len(active_worker_ranks)}, servers={server_count}, "
            f"batch_games={_resolve_eval_batch_games(config, num_games)}, "
            f"sims={_resolve_eval_mcts_simulations(config)}"
            f"{load_summary}"
        )

        worker_processes_by_rank = {}
        restart_counts = {int(rank): 0 for rank in active_worker_ranks}
        restart_limit = max(0, int(rl_cfg.get("eval_worker_restart_limit", 2) or 0))
        pending_game_indices_by_rank = {
            int(rank): list(game_indices_per_worker[int(rank)])
            for rank in active_worker_ranks
        }
        total_game_count_by_rank = {
            int(rank): len(game_indices_per_worker[int(rank)])
            for rank in active_worker_ranks
        }
        completed_game_indices_by_rank = {int(rank): set() for rank in active_worker_ranks}

        def _start_eval_worker(rank, game_indices):
            proc = ctx.Process(
                target=_eval_central_worker,
                args=(
                    rank,
                    config,
                    list(game_indices),
                    request_queues[worker_server_idx[rank]],
                    worker_response_receivers[rank],
                    result_queue,
                    use_fixed_openings,
                    model1_mcts_config,
                    model2_mcts_config,
                ),
            )
            proc.daemon = True
            proc.start()
            worker_processes_by_rank[int(rank)] = proc
            if proc not in worker_processes:
                worker_processes.append(proc)
            return proc

        for rank in active_worker_ranks:
            _start_eval_worker(rank, game_indices_per_worker[rank])

        wins = 0
        draws = 0
        losses = 0
        unresolved = 0
        completed = 0
        finished_worker_ranks = set()
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game")
        worker_error = None
        try:
            while len(finished_worker_ranks) < len(active_worker_ranks):
                try:
                    message = result_queue.get(timeout=1.0)
                except Exception:
                    dead_workers = [
                        (rank, proc.exitcode)
                        for rank, proc in list(worker_processes_by_rank.items())
                        if (
                            rank not in finished_worker_ranks
                            and proc is not None
                            and not proc.is_alive()
                            and proc.exitcode not in (0, None)
                        )
                    ]
                    dead_servers = [
                        proc.exitcode for proc in server_processes
                        if proc is not None and not proc.is_alive() and proc.exitcode not in (0, None)
                    ]
                    if dead_workers:
                        for rank, exitcode in dead_workers:
                            completed_for_rank = completed_game_indices_by_rank.setdefault(int(rank), set())
                            remaining = [
                                int(game_idx)
                                for game_idx in pending_game_indices_by_rank.get(int(rank), [])
                                if int(game_idx) not in completed_for_rank
                            ]
                            if not remaining:
                                finished_worker_ranks.add(int(rank))
                                continue
                            restart_counts[int(rank)] = int(restart_counts.get(int(rank), 0)) + 1
                            if restart_counts[int(rank)] > restart_limit:
                                worker_error = (
                                    "eval worker exited unexpectedly after "
                                    f"{restart_limit} restart attempt(s): rank={rank}, "
                                    f"exitcode={exitcode}, remaining={len(remaining)}"
                                )
                                _terminate_eval_processes(list(worker_processes_by_rank.values()))
                                break
                            print(
                                "Warning: eval worker "
                                f"{rank} exited with code {exitcode}; "
                                f"completed={len(completed_for_rank)}/"
                                f"{int(total_game_count_by_rank.get(int(rank), 0))}, "
                                f"remaining={len(remaining)}, "
                                f"restart={restart_counts[int(rank)]}/{restart_limit}."
                            )
                            old_proc = worker_processes_by_rank.get(int(rank))
                            if old_proc is not None:
                                with contextlib.suppress(Exception):
                                    old_proc.join(timeout=0.2)
                            pending_game_indices_by_rank[int(rank)] = remaining
                            _start_eval_worker(int(rank), remaining)
                        if worker_error is not None:
                            break
                    if dead_servers:
                        worker_error = f"eval central inference server exited unexpectedly: exitcodes={dead_servers}"
                        _terminate_eval_processes(list(worker_processes_by_rank.values()))
                        break
                    continue
                message_type = message.get("type")
                if message_type == "progress":
                    rank = int(message.get("rank", -1))
                    game_idx = message.get("game_idx", None)
                    counted = False
                    if game_idx is not None:
                        try:
                            game_idx = int(game_idx)
                        except (TypeError, ValueError):
                            game_idx = None
                    if game_idx is not None:
                        completed_for_rank = completed_game_indices_by_rank.setdefault(rank, set())
                        if game_idx not in completed_for_rank:
                            completed_for_rank.add(game_idx)
                            counted = True
                    else:
                        counted = True
                    if counted:
                        completed += int(message.get("completed", 0))
                        wins += int(message.get("wins", 0))
                        draws += int(message.get("draws", 0))
                        losses += int(message.get("losses", 0))
                        unresolved += int(message.get("unresolved", 0))
                        eval_bar.n = min(num_games, completed)
                        eval_bar.refresh()
                elif message_type == "result":
                    rank = int(message.get("rank", -1))
                    finished_worker_ranks.add(rank)
                elif message_type == "interrupt":
                    raise KeyboardInterrupt
                elif message_type == "error":
                    rank = int(message.get("rank", -1))
                    error_text = str(message.get("error", "unknown error"))
                    completed_for_rank = completed_game_indices_by_rank.setdefault(rank, set())
                    remaining = [
                        int(game_idx)
                        for game_idx in pending_game_indices_by_rank.get(rank, [])
                        if int(game_idx) not in completed_for_rank
                    ]
                    restart_counts[rank] = int(restart_counts.get(rank, 0)) + 1
                    if not remaining:
                        finished_worker_ranks.add(rank)
                    elif restart_counts[rank] <= restart_limit:
                        print(
                            "Warning: eval worker "
                            f"{rank} reported error; remaining={len(remaining)}, "
                            f"restart={restart_counts[rank]}/{restart_limit}: {error_text}"
                        )
                        old_proc = worker_processes_by_rank.get(rank)
                        if old_proc is not None:
                            with contextlib.suppress(Exception):
                                old_proc.join(timeout=0.2)
                        pending_game_indices_by_rank[rank] = remaining
                        _start_eval_worker(rank, remaining)
                    else:
                        worker_error = (
                            "eval worker failed after "
                            f"{restart_limit} restart attempt(s): rank={rank}, error={error_text}"
                        )
                        _terminate_eval_processes(list(worker_processes_by_rank.values()))
                        break
        finally:
            eval_bar.close()

        if worker_error is not None:
            raise RuntimeError(f"Eval worker failed: {worker_error}")
        if unresolved > 0:
            _print_eval_unresolved("Eval", unresolved, num_games, max_moves)
        return _build_eval_stats(wins, draws, losses, unresolved, num_games)
    finally:
        _terminate_eval_processes(worker_processes)
        for request_queue in request_queues:
            with contextlib.suppress(Exception):
                request_queue.put({"cmd": "stop"})
        _terminate_eval_processes(server_processes, timeout_s=1.0)
        for queue_obj in list(request_queues) + list(control_queues) + [result_queue]:
            with contextlib.suppress(Exception):
                queue_obj.close()
        for recv_conn in worker_response_receivers.values():
            with contextlib.suppress(Exception):
                recv_conn.close()
        for sender_map in server_response_senders:
            for send_conn in sender_map.values():
                with contextlib.suppress(Exception):
                    send_conn.close()


def _get_hflip_inverse_index_map():
    global _HFLIP_INV_INDEX_MAP
    if _HFLIP_INV_INDEX_MAP is not None:
        return _HFLIP_INV_INDEX_MAP

    inverse_map = build_hflip_inverse_index_map()
    if len(inverse_map) != ACTION_SIZE:
        raise ValueError(f"Horizontal flip map length mismatch: got {len(inverse_map)}, expected {ACTION_SIZE}")
    _HFLIP_INV_INDEX_MAP = torch.from_numpy(inverse_map).long()
    return _HFLIP_INV_INDEX_MAP


def _get_hflip_forward_index_map():
    global _HFLIP_FWD_INDEX_MAP
    if _HFLIP_FWD_INDEX_MAP is not None:
        return _HFLIP_FWD_INDEX_MAP

    inv_map = _get_hflip_inverse_index_map()
    forward_map = torch.empty_like(inv_map)
    forward_map[inv_map] = torch.arange(inv_map.numel(), dtype=inv_map.dtype)
    _HFLIP_FWD_INDEX_MAP = forward_map
    return _HFLIP_FWD_INDEX_MAP


def _maybe_augment_batch(boards, policy_indices, policy_values, policy_mask, config):
    rl_cfg = config.get("reinforcement_learning", {})
    if not rl_cfg.get("use_augmentation", False):
        return boards, policy_indices, policy_values, policy_mask
    if not rl_cfg.get("augment_horizontal_flip", False):
        return boards, policy_indices, policy_values, policy_mask

    prob = rl_cfg.get("augment_prob", 0.5)
    if prob <= 0:
        return boards, policy_indices, policy_values, policy_mask

    batch_size = boards.size(0)
    if batch_size == 0:
        return boards, policy_indices, policy_values, policy_mask

    flip_mask = torch.rand(batch_size) < prob
    if not flip_mask.any():
        return boards, policy_indices, policy_values, policy_mask

    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])

    if policy_indices.numel() > 0:
        forward_map = _get_hflip_forward_index_map()
        active = flip_mask.unsqueeze(1) & policy_mask
        if active.any():
            remapped = forward_map[policy_indices[active].long()].to(dtype=policy_indices.dtype)
            policy_indices[active] = remapped

    return boards, policy_indices, policy_values, policy_mask


def _final_outcome_targets(value_targets, epsilon=1e-6):
    """Map RL value targets to final W/D/L outcomes from the side-to-move POV."""
    target = value_targets.view(-1)
    return torch.where(
        torch.abs(target) <= float(epsilon),
        torch.zeros_like(target),
        torch.sign(target),
    )


def _wdl_targets_from_final_outcome(target_scalar):
    target_win = (target_scalar > 0.0).to(dtype=target_scalar.dtype)
    target_draw = (target_scalar == 0.0).to(dtype=target_scalar.dtype)
    target_loss = (target_scalar < 0.0).to(dtype=target_scalar.dtype)
    return torch.stack((target_win, target_draw, target_loss), dim=1)


def _apply_wdl_label_smoothing(target_wdl, smoothing):
    smoothing = max(0.0, min(0.30, float(smoothing or 0.0)))
    if smoothing <= 0.0:
        return target_wdl
    off_value = smoothing / max(1, target_wdl.size(1) - 1)
    return target_wdl * (1.0 - smoothing) + (1.0 - target_wdl) * off_value


def _legal_only_sparse_policy_loss(
    policy_logits,
    policy_indices,
    policy_values,
    policy_mask,
    legal_indices,
    legal_mask,
):
    batch_size = int(policy_logits.size(0))
    num_classes = int(policy_logits.size(1))
    if policy_indices.numel() == 0:
        return torch.zeros(batch_size, device=policy_logits.device, dtype=policy_logits.dtype)

    if legal_indices is None or legal_indices.numel() == 0:
        legal_indices = policy_indices
        legal_mask = policy_mask

    safe_policy_indices = policy_indices.long().clamp(0, num_classes - 1)
    valid_policy_mask = policy_mask & (policy_indices >= 0) & (policy_indices < num_classes)
    dense_targets = torch.zeros(
        (batch_size, num_classes),
        device=policy_logits.device,
        dtype=policy_logits.float().dtype,
    )
    dense_targets.scatter_add_(
        1,
        safe_policy_indices,
        torch.where(valid_policy_mask, policy_values.float(), torch.zeros_like(policy_values.float())),
    )

    safe_legal_indices = legal_indices.long().clamp(0, num_classes - 1)
    valid_legal_mask = legal_mask & (legal_indices >= 0) & (legal_indices < num_classes)
    legal_logits = torch.gather(policy_logits.float(), 1, safe_legal_indices)
    legal_logits = legal_logits.masked_fill(~valid_legal_mask, -1.0e9)
    legal_log_probs = F.log_softmax(legal_logits, dim=1)

    legal_targets = torch.gather(dense_targets, 1, safe_legal_indices)
    legal_targets = torch.where(valid_legal_mask, legal_targets, torch.zeros_like(legal_targets))
    target_mass = legal_targets.sum(dim=1, keepdim=True)
    legal_targets = torch.where(
        target_mass > 0.0,
        legal_targets / target_mass.clamp_min(1e-8),
        legal_targets,
    )
    return -(legal_targets * legal_log_probs).sum(dim=1).to(dtype=policy_logits.dtype)


def _weighted_mean(losses, weights):
    weights = torch.clamp(weights.to(dtype=losses.dtype), min=0.0)
    weight_total = weights.sum()
    if float(weight_total.detach().item()) <= 0.0:
        return losses.mean()
    return (losses * weights).sum() / weight_total


def train_on_batch_rl(
    model,
    optimizer,
    batch,
    config,
    device,
    scaler,
    metrics_calc=None,
    value_weight_override=None,
    policy_weight_override=None,
    anchor_model=None,
    best_model_anchor=None,
    best_policy_kl_weight_override=None,
    best_value_distill_weight_override=None,
):
    if len(batch) >= 10:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights, moves_left_targets, legal_indices, legal_mask = batch[:10]
    elif len(batch) >= 8:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights, moves_left_targets = batch[:8]
        legal_indices = policy_indices
        legal_mask = policy_mask
    elif len(batch) >= 7:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights = batch[:7]
        moves_left_targets = torch.zeros_like(value_targets, dtype=torch.float32)
        legal_indices = policy_indices
        legal_mask = policy_mask
    else:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights = batch
        value_sample_weights = torch.ones_like(policy_sample_weights, dtype=torch.float32)
        moves_left_targets = torch.zeros_like(value_targets, dtype=torch.float32)
        legal_indices = policy_indices
        legal_mask = policy_mask
    if config["reinforcement_learning"].get("replay_fp16", False):
        boards = boards.float()
        policy_values = policy_values.float()
        value_targets = value_targets.float()
        moves_left_targets = moves_left_targets.float()

    boards, policy_indices, policy_values, policy_mask = _maybe_augment_batch(
        boards, policy_indices, policy_values, policy_mask, config
    )
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_indices = policy_indices.to(device, non_blocking=True)
    policy_values = policy_values.to(device, non_blocking=True)
    policy_mask = policy_mask.to(device, non_blocking=True)
    legal_indices = legal_indices.to(device, non_blocking=True)
    legal_mask = legal_mask.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    policy_sample_weights = policy_sample_weights.to(device, non_blocking=True)
    value_sample_weights = value_sample_weights.to(device, non_blocking=True)
    moves_left_targets = moves_left_targets.to(device, non_blocking=True)
    value_sample_weights = torch.ones_like(value_sample_weights, dtype=torch.float32, device=device)
    effective_policy_mask = policy_mask & (policy_sample_weights.unsqueeze(1) > 0)

    optimizer.zero_grad(set_to_none=True)

    rl_cfg = config.get("reinforcement_learning", {})
    use_amp = config["hardware"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if config["hardware"].get("use_bfloat16", False) else torch.float16

    value_aux_scalar_loss_weight = float(
        rl_cfg.get("value_aux_scalar_loss_weight", 0.25)
    )
    value_std_floor_loss_weight = max(
        0.0,
        float(rl_cfg.get("value_std_floor_loss_weight", 0.0)),
    )
    value_std_floor_target_ratio = max(
        0.0,
        float(rl_cfg.get("value_std_floor_target_ratio", 0.70)),
    )
    moves_left_loss_weight = max(0.0, float(rl_cfg.get("moves_left_loss_weight", 0.05)))
    policy_anchor_kl_weight = max(
        0.0,
        float(rl_cfg.get("policy_anchor_kl_weight", 0.0)),
    )
    best_policy_kl_weight = max(
        0.0,
        float(
            rl_cfg.get("post_promotion_best_policy_kl_weight", 0.0)
            if best_policy_kl_weight_override is None
            else best_policy_kl_weight_override
        ),
    )
    best_value_distill_weight = max(
        0.0,
        float(
            rl_cfg.get("post_promotion_best_value_distill_weight", 0.0)
            if best_value_distill_weight_override is None
            else best_value_distill_weight_override
        ),
    )

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        policy_logits, value_pred, moves_left_pred = model(
            boards,
            apply_log_softmax=False,
            return_moves_left=True,
        )
        policy_pred = F.log_softmax(policy_logits.float(), dim=1)
        policy_anchor_kl_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        best_policy_kl_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        best_value_distill_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        best_policy_pred = None
        best_value_pred = None
        if anchor_model is not None and policy_anchor_kl_weight > 0.0:
            with torch.no_grad():
                anchor_policy_pred, _anchor_value_pred = anchor_model(boards)
            anchor_policy_probs = torch.exp(anchor_policy_pred.detach())
            policy_anchor_kl_loss = (
                anchor_policy_probs * (anchor_policy_pred.detach() - policy_pred)
            ).sum(dim=1).mean()
        if best_model_anchor is not None and (
            best_policy_kl_weight > 0.0 or best_value_distill_weight > 0.0
        ):
            with torch.no_grad():
                best_policy_pred, best_value_pred = best_model_anchor(boards)
            if best_policy_kl_weight > 0.0:
                best_policy_probs = torch.exp(best_policy_pred.detach())
                best_policy_kl_loss = (
                    best_policy_probs * (best_policy_pred.detach() - policy_pred)
                ).sum(dim=1).mean()
        value_pred_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)
        target_value_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)
        value_std_floor_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)

        if policy_indices.numel() == 0:
            policy_loss = torch.zeros(policy_pred.size(0), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            policy_loss = _legal_only_sparse_policy_loss(
                policy_logits,
                policy_indices,
                policy_values,
                effective_policy_mask,
                legal_indices,
                legal_mask,
            )
            rl_cfg = config.get("reinforcement_learning", {})
            if bool(rl_cfg.get("policy_target_confidence_weighting_enabled", False)):
                valid_targets = torch.where(
                    effective_policy_mask,
                    torch.clamp(policy_values, min=0.0),
                    torch.zeros_like(policy_values),
                )
                target_mass = valid_targets.sum(dim=1).clamp_min(1e-8)
                normalized_targets = valid_targets / target_mass.unsqueeze(1)
                target_lengths = effective_policy_mask.sum(dim=1).to(dtype=policy_loss.dtype)
                target_entropy = -(
                    normalized_targets
                    * torch.log(torch.clamp(normalized_targets, min=1e-12))
                ).sum(dim=1)
                max_entropy = torch.log(torch.clamp(target_lengths, min=2.0))
                entropy_confidence = 1.0 - torch.clamp(target_entropy / max_entropy, 0.0, 1.0)
                top1_confidence = normalized_targets.max(dim=1).values.to(dtype=policy_loss.dtype)
                confidence = torch.maximum(entropy_confidence, top1_confidence)
                confidence_power = max(
                    0.05,
                    float(rl_cfg.get("policy_target_confidence_power", 1.0)),
                )
                if confidence_power != 1.0:
                    confidence = torch.pow(torch.clamp(confidence, min=0.0), confidence_power)
                min_weight = max(
                    0.0,
                    min(1.0, float(rl_cfg.get("policy_target_confidence_min_weight", 0.35))),
                )
                confidence_weight = min_weight + (1.0 - min_weight) * confidence
                policy_loss = policy_loss * confidence_weight.to(dtype=policy_loss.dtype)
            policy_loss = policy_loss * policy_sample_weights.to(dtype=policy_loss.dtype)

        target_scalar = _final_outcome_targets(value_targets)
        target_value_std = target_scalar.std(unbiased=False)
        effective_value_sample_weights = value_sample_weights

        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            target_wdl = _wdl_targets_from_final_outcome(target_scalar)
            target_wdl = _apply_wdl_label_smoothing(
                target_wdl,
                rl_cfg.get("value_wdl_label_smoothing", 0.0),
            )

            value_log_probs = F.log_softmax(value_pred, dim=1)
            value_ce_loss = -(target_wdl * value_log_probs).sum(dim=1)
            value_probs = torch.softmax(value_pred, dim=1)
            value_scalar = value_probs[:, 0] - value_probs[:, 2]
            value_scalar_aux_loss = F.smooth_l1_loss(
                value_scalar,
                target_scalar,
                reduction="none",
                beta=0.25,
            )
            value_loss = value_ce_loss + value_aux_scalar_loss_weight * value_scalar_aux_loss
            value_scalar_std = value_scalar.std(unbiased=False)
            value_pred_std = value_scalar_std.detach()
            if value_std_floor_loss_weight > 0.0 and value_scalar.numel() > 1:
                target_std_floor = target_value_std.detach().to(dtype=value_scalar_std.dtype) * value_std_floor_target_ratio
                std_shortfall = torch.relu(target_std_floor - value_scalar_std)
                value_std_floor_loss = std_shortfall * std_shortfall
            if best_value_pred is not None and best_value_distill_weight > 0.0:
                if best_value_pred.dim() == 2 and best_value_pred.size(1) == 3:
                    best_value_probs = torch.softmax(best_value_pred.detach(), dim=1)
                    best_value_scalar = best_value_probs[:, 0] - best_value_probs[:, 2]
                else:
                    best_value_scalar = best_value_pred.detach().squeeze()
                best_value_distill_loss = _weighted_mean(
                    F.smooth_l1_loss(
                        value_scalar,
                        best_value_scalar.to(dtype=value_scalar.dtype),
                        reduction="none",
                        beta=0.20,
                    ),
                    effective_value_sample_weights,
                )
        else:
            value_loss = (value_pred.squeeze() - target_scalar) ** 2
            value_scalar = value_pred.squeeze()
            value_scalar_std = value_scalar.std(unbiased=False)
            value_pred_std = value_scalar_std.detach()
            if value_std_floor_loss_weight > 0.0 and value_scalar.numel() > 1:
                target_std_floor = target_value_std.detach().to(dtype=value_scalar_std.dtype) * value_std_floor_target_ratio
                std_shortfall = torch.relu(target_std_floor - value_scalar_std)
                value_std_floor_loss = std_shortfall * std_shortfall
            if best_value_pred is not None and best_value_distill_weight > 0.0:
                if best_value_pred.dim() == 2 and best_value_pred.size(1) == 3:
                    best_value_probs = torch.softmax(best_value_pred.detach(), dim=1)
                    best_value_scalar = best_value_probs[:, 0] - best_value_probs[:, 2]
                else:
                    best_value_scalar = best_value_pred.detach().squeeze()
                best_value_distill_loss = _weighted_mean(
                    F.smooth_l1_loss(
                        value_scalar,
                        best_value_scalar.to(dtype=value_scalar.dtype),
                        reduction="none",
                        beta=0.20,
                    ),
                    effective_value_sample_weights,
                )

        if policy_loss.numel() == 0:
            policy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            policy_weight_total = policy_sample_weights.to(dtype=policy_loss.dtype).sum()
            if float(policy_weight_total.detach().item()) > 0.0:
                policy_loss = policy_loss.sum() / policy_weight_total
            else:
                policy_loss = policy_loss.sum() * 0.0
        value_loss = _weighted_mean(value_loss, effective_value_sample_weights)
        moves_left_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        if moves_left_loss_weight > 0.0 and moves_left_pred is not None:
            pred_mlh = moves_left_pred.reshape(-1)
            target_mlh = torch.log1p(
                torch.clamp(moves_left_targets.reshape(-1).to(dtype=pred_mlh.dtype), min=0.0)
            )
            moves_left_loss = F.smooth_l1_loss(pred_mlh, target_mlh, beta=0.25)
        policy_weight = (
            float(config["reinforcement_learning"]["policy_loss_weight"])
            if policy_weight_override is None
            else float(policy_weight_override)
        )
        value_weight = (
            float(config["reinforcement_learning"]["value_loss_weight"])
            if value_weight_override is None
            else float(value_weight_override)
        )
        loss = (
            policy_weight * policy_loss
            + value_weight * value_loss
            + moves_left_loss_weight * moves_left_loss
            + value_std_floor_loss_weight * value_std_floor_loss
            + policy_anchor_kl_weight * policy_anchor_kl_loss
            + best_policy_kl_weight * best_policy_kl_loss
            + best_value_distill_weight * best_value_distill_loss
        )

        policy_probs = torch.exp(policy_pred)
        policy_entropy = -(policy_probs * policy_pred).sum(dim=1).mean()
        entropy_weight = config["reinforcement_learning"].get("entropy_weight", 0.0)
        if entropy_weight > 0:
            loss = loss - entropy_weight * policy_entropy

    scaler.scale(loss).backward()

    grad_clip = config["reinforcement_learning"].get("grad_clip", 1.0)
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    if metrics_calc is not None:
        with torch.no_grad():
            if policy_indices.numel() == 0:
                target_moves = torch.zeros(policy_pred.size(0), dtype=torch.long, device=policy_pred.device)
            else:
                best_sparse_idx = policy_values.argmax(dim=1, keepdim=True)
                target_moves = torch.gather(policy_indices.long(), 1, best_sparse_idx).squeeze(1)
            rl_cfg = config.get("reinforcement_learning", {})
            if boards.dim() >= 4 and boards.size(1) > 15:
                fullmove_indices = torch.clamp(
                    torch.round(boards[:, 15, 0, 0].float() * 100.0),
                    min=1.0,
                    max=float(rl_cfg.get("value_phase_max_fullmove", 120)),
                )
            else:
                fullmove_indices = None
            metrics_calc.update(
                policy_pred,
                value_pred,
                target_moves,
                target_scalar.unsqueeze(1),
                move_indices=fullmove_indices,
                value_weight_min=float(rl_cfg.get("value_metric_weight_min", 0.20)),
                value_max_moves=int(rl_cfg.get("value_metric_max_fullmove", 80)),
                value_phase_opening_max=int(rl_cfg.get("value_phase_opening_max_fullmove", 12)),
                value_phase_endgame_min=int(rl_cfg.get("value_phase_endgame_min_fullmove", 40)),
            )

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        float(policy_entropy.detach().item()),
        float(value_pred_std.detach().item()),
        float(target_value_std.detach().item()),
    )


def evaluate_models(
    model1,
    model2,
    config,
    device,
    num_games=100,
    game_index_offset=0,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    progress_desc="Eval vs best",
):
    config = _build_eval_mcts_config(config)
    model1_mcts_config = _build_eval_mcts_config(model1_mcts_config or config)
    model2_mcts_config = _build_eval_mcts_config(model2_mcts_config or config)
    rl_cfg = config.get("reinforcement_learning", {})
    central_min_games = max(1, int(rl_cfg.get("eval_central_inference_min_games", 2) or 2))
    if _eval_uses_central_inference(config, device) and int(num_games) >= central_min_games:
        return _evaluate_models_with_central_inference(
            model1,
            model2,
            config,
            device,
            int(num_games),
            game_index_offset=game_index_offset,
            use_fixed_openings=use_fixed_openings,
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
            progress_desc=progress_desc,
        )

    workers = _resolve_eval_workers(config, device, num_games)
    if workers <= 1:
        max_moves = _resolve_eval_max_moves(config)
        game_indices = list(range(game_index_offset, game_index_offset + num_games))
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game")
        try:
            stats = _evaluate_games_batched(
                model1,
                model2,
                config,
                device,
                game_indices,
                use_fixed_openings=use_fixed_openings,
                progress_callback=lambda completed, *args: eval_bar.update(int(completed)),
                model1_mcts_config=model1_mcts_config,
                model2_mcts_config=model2_mcts_config,
            )
        finally:
            eval_bar.close()

        unresolved = int((stats or {}).get("unresolved", 0))
        if unresolved > 0:
            _print_eval_unresolved("Eval", unresolved, num_games, max_moves)

        return stats

    model1_state = _snapshot_state_dict_cpu_shared(model1)
    model2_state = _snapshot_state_dict_cpu_shared(model2)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    game_indices_per_worker = [[] for _ in range(workers)]
    for offset, game_idx in enumerate(range(game_index_offset, game_index_offset + num_games)):
        game_indices_per_worker[offset % workers].append(game_idx)

    for rank, game_indices in enumerate(game_indices_per_worker):
        if not game_indices:
            continue
        p = ctx.Process(
            target=_eval_worker,
            args=(
                rank,
                model1_state,
                model2_state,
                config,
                str(device),
                game_indices,
                result_queue,
                use_fixed_openings,
                model1_mcts_config,
                model2_mcts_config,
            ),
        )
        p.start()
        processes.append(p)

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game")
    worker_error = None
    try:
        finished_workers = 0
        while finished_workers < len(processes):
            message = result_queue.get()
            message_type = message.get("type")
            if message_type == "progress":
                completed += int(message.get("completed", 0))
                eval_bar.n = min(num_games, completed)
                eval_bar.refresh()
            elif message_type == "result":
                wins += int(message.get("wins", 0))
                draws += int(message.get("draws", 0))
                losses += int(message.get("losses", 0))
                unresolved += int(message.get("unresolved", 0))
                finished_workers += 1
            elif message_type == "interrupt":
                raise KeyboardInterrupt
            elif message_type == "error":
                worker_error = str(message.get("error", "unknown error"))
                _terminate_eval_processes(processes)
                break
    finally:
        eval_bar.close()
        for p in processes:
            with contextlib.suppress(Exception):
                p.join(timeout=0.5)

    if worker_error is not None:
        raise RuntimeError(f"Eval worker failed: {worker_error}")

    if unresolved > 0:
        max_moves = _resolve_eval_max_moves(config)
        _print_eval_unresolved("Eval", unresolved, num_games, max_moves)

    return _build_eval_stats(wins, draws, losses, unresolved, num_games)


def evaluate_models_no_mcts(model1, model2, config, device, num_games=30, game_index_offset=0, use_fixed_openings=None):
    """Policy-head-only evaluation. This is intentionally separate from promotion eval."""
    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    max_moves = _resolve_eval_max_moves(config)
    auto_claim_draw = _resolve_eval_auto_claim_draw(config)
    claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(config)
    claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(config)

    model1.eval()
    model2.eval()
    eval_bar = tqdm(total=num_games, desc="Eval no MCTS", unit="game")
    try:
        for game_idx in range(game_index_offset, game_index_offset + num_games):
            model1_as_white = (game_idx % 2 == 0)
            white_model = model1 if model1_as_white else model2
            black_model = model2 if model1_as_white else model1
            opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
            result, was_unresolved = _evaluate_single_game_no_mcts(
                white_model,
                black_model,
                config,
                device,
                max_moves,
                auto_claim_draw,
                claim_draw_after_moves,
                claim_repetition_after_moves=claim_repetition_after_moves,
                opening_prefix=opening_prefix,
            )
            if was_unresolved:
                unresolved += 1
                eval_bar.update(1)
                continue
            game_wins, game_draws, game_losses = _result_for_model1(model1_as_white, result)
            wins += game_wins
            draws += game_draws
            losses += game_losses
            eval_bar.update(1)
    finally:
        eval_bar.close()

    if unresolved > 0:
        _print_eval_unresolved("No-MCTS eval", unresolved, num_games, max_moves)

    return _build_eval_stats(wins, draws, losses, unresolved, num_games)
