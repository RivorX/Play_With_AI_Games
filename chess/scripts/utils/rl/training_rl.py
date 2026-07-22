"""
Reinforcement learning training helpers.
"""

import os
import sys
import contextlib
import concurrent.futures
import math
import threading
import time
from pathlib import Path

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
    _board_position_key,
    _record_position_count,
    central_inference_server,
)
from src import chess_backend as chess
from src.data import board_to_tensor, move_to_index
from src.model import ChessNet
from src.utils.data_helpers import (
    ACTION_SIZE,
    MAX_LEGAL_MOVES,
    board_to_tensor_pair,
    build_hflip_inverse_index_map,
)
from src.utils.q_delta import USEFUL_SEARCH_Q_DELTA_MIN
from src.utils.shared_inference import create_shared_inference_buffer


_HFLIP_INV_INDEX_MAP = None
_HFLIP_FWD_INDEX_MAP = None
# Self-play persists the correction identity and its bounded replay weight.
# Training uses the metadata only for uptake diagnostics; it must not multiply
# these rows a second time or the actual objective would diverge from the
# replay weights reported in CSV/PNG.


def _gradient_family(name):
    lowered = str(name).lower()
    if lowered.startswith('policy_') or '.policy_' in lowered:
        return 'policy'
    if (
        lowered.startswith('value_')
        or '.value_' in lowered
        or lowered.startswith('moves_left_')
        or '.moves_left_' in lowered
        or lowered.startswith('search_q_')
        or '.search_q_' in lowered
        or lowered.startswith('search_error_')
        or '.search_error_' in lowered
    ):
        return 'value'
    return 'backbone'


def _gradient_family_norms(model):
    sums = {'backbone': None, 'policy': None, 'value': None}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        family = _gradient_family(name)
        squared = parameter.grad.detach().float().pow(2).sum()
        sums[family] = squared if sums[family] is None else sums[family] + squared
    return {
        family: (0.0 if squared is None else float(torch.sqrt(squared).item()))
        for family, squared in sums.items()
    }


def _task_gradient_probe(policy_objective, value_objective, model):
    """Measure task conflict at the shared tower output once per iteration."""
    probe_owner = getattr(model, '_orig_mod', model)
    probe = getattr(getattr(probe_owner, 'final_bn', None), 'weight', None)
    empty = {
        'policy_probe_norm': 0.0,
        'value_probe_norm': 0.0,
        'policy_value_cosine': 0.0,
    }
    if probe is None or not probe.requires_grad:
        return empty
    policy_grad = torch.autograd.grad(
        policy_objective,
        probe,
        retain_graph=True,
        allow_unused=True,
    )[0]
    value_grad = torch.autograd.grad(
        value_objective,
        probe,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if policy_grad is None or value_grad is None:
        return empty
    policy_flat = policy_grad.detach().float().reshape(-1)
    value_flat = value_grad.detach().float().reshape(-1)
    policy_norm = torch.linalg.vector_norm(policy_flat)
    value_norm = torch.linalg.vector_norm(value_flat)
    denominator = policy_norm * value_norm
    cosine = (
        torch.dot(policy_flat, value_flat) / denominator
        if float(denominator.item()) > 0.0
        else torch.zeros((), device=policy_flat.device)
    )
    return {
        'policy_probe_norm': float(policy_norm.item()),
        'value_probe_norm': float(value_norm.item()),
        'policy_value_cosine': float(torch.clamp(cosine, -1.0, 1.0).item()),
    }


def _snapshot_state_dict_cpu_shared(model):
    snapshot = {}
    for key, tensor in model.state_dict().items():
        cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
        cpu_tensor.share_memory_()
        snapshot[key] = cpu_tensor
    return snapshot


def _eval_uses_central_inference(config, device):
    rl_cfg = config.get("reinforcement_learning", {})
    central_cfg = config.get("central_inference", {}) or {}
    enabled = rl_cfg.get(
        "eval_central_inference_enabled",
        rl_cfg.get("self_play_central_inference_enabled", False),
    )
    return bool(enabled and central_cfg.get("enabled", True) and device.type == "cuda" and torch.cuda.is_available())


def _central_inference_config(config):
    return (config or {}).get("central_inference", {}) or {}


def _central_inference_option(config, key, default=None):
    central_cfg = _central_inference_config(config)
    return central_cfg.get(key, default)


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

    # Keep one stable topology across every funnel stage. Scaling workers down
    # from the stage's game count made 40/60/80-game evals use 7/10/12 workers,
    # forcing different batching behavior and making stage timings incomparable.
    return max(1, min(int(num_games), workers))


def _build_eval_mcts_config(config):
    """Evaluation stays deterministic and may lightly trim only easy roots."""
    eval_config = dict(config or {})
    rl_config = dict(eval_config.get('reinforcement_learning', {}) or {})
    # Self-play's zero-sum redistribution is not an evaluation speedup. Eval's
    # dedicated easy-cut mode spends less total compute and never exceeds the
    # configured simulation count on uncertain positions.
    rl_config['mcts_dynamic_budget_enabled'] = False
    eval_config['reinforcement_learning'] = rl_config
    return eval_config


def _resolve_eval_central_server_count(config, workers):
    """Eval targets one concrete CUDA device, so it always owns one server."""
    # More processes on cuda:0 do not add GPU capacity. They duplicate models,
    # split request batches, consume VRAM and compete for the same kernels.
    # Worker count controls CPU-side MCTS concurrency; the GPU stays centralized.
    return 1


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
        # A ply-cap timeout has no chess result. Score it neutrally instead of
        # silently charging the candidate with a loss.
        "score_rate": (wins + 0.5 * (draws + unresolved)) / num_games if num_games > 0 else 0.0,
        "win_rate": wins / num_games if num_games > 0 else 0.0,
        "draw_rate": (draws + unresolved) / num_games if num_games > 0 else 0.0,
        "loss_rate": losses / num_games if num_games > 0 else 0.0,
        "resolved_games": num_games - unresolved,
    }


def _print_eval_unresolved(label, unresolved, num_games, max_moves):
    if int(unresolved) <= 0:
        return
    print(
        f"{label} unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
        f"{unresolved}/{num_games} -> scored as 0.5 and tracked separately"
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


def _append_eval_history(board_history, board, config):
    board_history.append(_encode_eval_history_entry(board))
    max_history = int(config.get("model", {}).get("history_positions", 0) or 0) + 10
    if len(board_history) > max_history:
        del board_history[:-max_history]


def _apply_opening_prefix_for_batched_eval(
    board,
    board_history,
    opening_prefix,
    config,
    position_counts=None,
):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if chess.is_game_over(board, claim_draw=False):
            break
        try:
            move = chess.move_from_uci(uci)
        except Exception:
            break
        if move is None or move not in chess.legal_moves(board):
            break

        _append_eval_history(board_history, board, config)
        chess.apply_move(board, move)
        if position_counts is not None:
            _record_position_count(position_counts, board)
        applied += 1
    return applied


def _encode_eval_history_entry(board):
    return board_to_tensor_pair(board)


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
    moves = _select_no_mcts_policy_moves_batched(
        model,
        [{"board": board, "board_history": board_history}],
        config,
        device,
    )
    return moves[0] if moves else None


def _select_no_mcts_policy_moves_batched(model, game_states, config, device):
    """Select raw-policy moves for many games with one model request."""
    states = list(game_states or [])
    if not states:
        return []

    inputs = []
    legal_moves_by_row = []
    legal_indices_by_row = []
    max_legal = 0
    for state in states:
        board = state["board"]
        inputs.append(_build_no_mcts_eval_input(board, state.get("board_history", []), config))
        legal_moves = tuple(chess.legal_moves(board))
        legal_indices = [move_to_index(move, board) for move in legal_moves]
        legal_moves_by_row.append(legal_moves)
        legal_indices_by_row.append(legal_indices)
        max_legal = max(max_legal, len(legal_indices))

    if max_legal <= 0:
        return [None] * len(states)

    boards_np = np.stack(inputs, axis=0).astype(np.float32, copy=False)
    remote_legal_gather = bool(getattr(model, "supports_remote_legal_gather", False))
    legal_matrix = np.zeros(
        (len(states), max_legal),
        dtype=np.int16 if remote_legal_gather else np.int64,
    )
    for row_idx, legal_indices in enumerate(legal_indices_by_row):
        if legal_indices:
            legal_matrix[row_idx, :len(legal_indices)] = legal_indices

    if remote_legal_gather:
        model_input = boards_np
        model_kwargs = {
            "apply_log_softmax": False,
            "policy_only": True,
            "legal_index_matrix": legal_matrix,
        }
    else:
        model_input = torch.from_numpy(boards_np).to(
            device,
            dtype=torch.float32,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        model_kwargs = {"apply_log_softmax": False, "policy_only": True}

    use_amp = bool(config.get("hardware", {}).get("use_amp", False) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if config.get("hardware", {}).get("use_bfloat16", False) else torch.float16
    with torch.inference_mode():
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            policy_logits, _value = model(model_input, **model_kwargs)
    compact_policy = bool(getattr(model, "last_response_compact_policy", False))
    policy_np = policy_logits.float().cpu().numpy()

    selected = []
    for row_idx, (legal_moves, legal_indices) in enumerate(zip(legal_moves_by_row, legal_indices_by_row)):
        if not legal_moves:
            selected.append(None)
            continue
        if compact_policy:
            legal_scores = policy_np[row_idx, :len(legal_moves)]
        else:
            legal_scores = policy_np[row_idx, np.asarray(legal_indices, dtype=np.int64)]
        selected.append(legal_moves[int(np.argmax(legal_scores))])
    return selected


def _apply_opening_prefix_for_no_mcts_eval(board, board_history, opening_prefix):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if chess.is_game_over(board, claim_draw=False):
            break
        try:
            move = chess.move_from_uci(uci)
        except Exception:
            break
        if move is None or move not in chess.legal_moves(board):
            break
        board_history.append(_encode_eval_history_entry(board))
        chess.apply_move(board, move)
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
    board = chess.new_board()
    board_history = []
    move_count = _apply_opening_prefix_for_no_mcts_eval(board, board_history, opening_prefix)
    ended_by_auto_claim_draw = False

    while move_count < max_moves:
        if chess.is_game_over(board, claim_draw=False):
            break

        model = model_white if board.turn == chess.WHITE else model_black
        move = _select_no_mcts_policy_move(model, board, board_history, config, device)
        if move is None:
            break

        board_history.append(_encode_eval_history_entry(board))
        chess.apply_move(board, move)
        move_count += 1

        if auto_claim_draw:
            if (
                move_count >= claim_repetition_after_moves
                and chess.can_claim_threefold_repetition(board)
            ):
                ended_by_auto_claim_draw = True
                break
            if move_count >= claim_draw_after_moves and chess.can_claim_draw(board):
                ended_by_auto_claim_draw = True
                break

    if ended_by_auto_claim_draw:
        return "1/2-1/2", False

    result = chess.result(board, claim_draw=False)
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
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
):
    game_indices = list(game_indices or [])
    if not game_indices:
        return _build_eval_stats(0, 0, 0, 0, 0)

    model1_search_config = model1_mcts_config or config
    model2_search_config = model2_mcts_config or config
    mcts1 = MultiGameBatchMCTS(model1, model1_search_config, device) if use_mcts_model1 else None
    mcts2 = MultiGameBatchMCTS(model2, model2_search_config, device) if use_mcts_model2 else None
    sims1 = _resolve_eval_mcts_simulations(model1_search_config)
    sims2 = _resolve_eval_mcts_simulations(model2_search_config)
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
    eval_search_profile = {}

    def _record_eval_search(prefix, metadata):
        if not isinstance(metadata, dict):
            return
        budget = int(metadata.get('simulation_budget', 0) or 0)
        requested_budget = int(metadata.get('eval_easy_cut_requested_budget', budget) or budget)
        budget_samples_key = f"eval_mcts_{prefix}_budget_samples"
        budget_sum_key = f"eval_mcts_{prefix}_budget_sum"
        reduced_key = f"eval_mcts_{prefix}_reduced_budget_count"
        eval_search_profile[budget_samples_key] = int(
            eval_search_profile.get(budget_samples_key, 0)
        ) + 1
        eval_search_profile[budget_sum_key] = float(
            eval_search_profile.get(budget_sum_key, 0.0)
        ) + float(budget)
        if budget < requested_budget:
            eval_search_profile[reduced_key] = int(
                eval_search_profile.get(reduced_key, 0)
            ) + 1
        agreement = metadata.get("selected_prior_agree")
        if agreement is None:
            return
        sample_key = f"eval_mcts_{prefix}_move_samples"
        changed_key = f"eval_mcts_{prefix}_changed_count"
        eval_search_profile[sample_key] = int(eval_search_profile.get(sample_key, 0)) + 1
        changed = float(agreement) < 0.5
        if not changed:
            return
        eval_search_profile[changed_key] = int(eval_search_profile.get(changed_key, 0)) + 1
        q_delta = metadata.get("selected_q_delta")
        try:
            q_delta = float(q_delta)
        except (TypeError, ValueError):
            return
        if not math.isfinite(q_delta):
            return
        q_samples_key = f"eval_mcts_{prefix}_changed_q_samples"
        q_sum_key = f"eval_mcts_{prefix}_changed_q_delta_sum"
        eval_search_profile[q_samples_key] = int(eval_search_profile.get(q_samples_key, 0)) + 1
        eval_search_profile[q_sum_key] = float(eval_search_profile.get(q_sum_key, 0.0)) + q_delta
        if q_delta > USEFUL_SEARCH_Q_DELTA_MIN:
            key = f"eval_mcts_{prefix}_higher_q_count"
            eval_search_profile[key] = int(eval_search_profile.get(key, 0)) + 1
        elif q_delta < -USEFUL_SEARCH_Q_DELTA_MIN:
            key = f"eval_mcts_{prefix}_lower_q_count"
            eval_search_profile[key] = int(eval_search_profile.get(key, 0)) + 1

    def _new_game_state(game_idx):
        board = chess.new_board()
        board_history = []
        position_counts = {_board_position_key(board): 1}
        opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
        move_count = _apply_opening_prefix_for_batched_eval(
            board,
            board_history,
            opening_prefix,
            config,
            position_counts,
        )
        return {
            "game_idx": int(game_idx),
            "board": board,
            "board_history": board_history,
            "position_counts": position_counts,
            "move_count": int(move_count),
            "model1_as_white": bool(int(game_idx) % 2 == 0),
            "model1_root": None,
            "model1_synced": False,
            "model2_root": None,
            "model2_synced": False,
            "done": bool(chess.is_game_over(board, claim_draw=False) or move_count >= max_moves),
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
            result = chess.result(board, claim_draw=False)
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
                int(gs.get("move_count", 0) or 0),
                str(result),
                bool(gs.get("model1_as_white", False)),
            )

    def _claim_draw_if_needed(gs):
        if not auto_claim_draw:
            return False
        board = gs["board"]
        if (
            gs["move_count"] >= claim_repetition_after_moves
            and chess.can_claim_threefold_repetition(board)
        ):
            gs["auto_claim_draw"] = True
            return True
        if gs["move_count"] >= claim_draw_after_moves and chess.can_claim_draw(board):
            gs["auto_claim_draw"] = True
            return True
        return False

    _fill_active()
    while active_games and not (stop_event is not None and stop_event.is_set()):
        model1_indices = []
        model2_indices = []
        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            if chess.is_game_over(board, claim_draw=False) or gs["move_count"] >= max_moves:
                gs["done"] = True
                continue
            model1_turn = bool(board.turn == chess.WHITE) == bool(gs["model1_as_white"])
            if model1_turn:
                model1_indices.append(idx)
            else:
                model2_indices.append(idx)

        moves_by_index = {}

        def _run_group(indices, model, search_config, mcts, sims, root_key, synced_key, profile_prefix):
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
                    gs.get("position_counts"),
                ])
            if mcts is not None:
                moves = []
                visit_counts_group, metadata_group = mcts.search_many(
                    group_states,
                    num_simulations=sims,
                    add_root_noise=False,
                    return_search_metadata=True,
                )
                for visit_counts, metadata in zip(visit_counts_group, metadata_group):
                    selected_move = (
                        metadata.get("selected_move_override")
                        if isinstance(metadata, dict)
                        else None
                    )
                    if selected_move in visit_counts:
                        move = selected_move
                    elif visit_counts:
                        move, _ = select_move_by_visits(visit_counts, temperature=0)
                    else:
                        move = None
                    moves.append(move)
                    _record_eval_search(profile_prefix, metadata)
            else:
                moves = _select_no_mcts_policy_moves_batched(
                    model,
                    [
                        {"board": local_state[0], "board_history": local_state[3]}
                        for local_state in group_states
                    ],
                    search_config,
                    device,
                )

            for idx, local_state, move in zip(indices, group_states, moves):
                gs = active_games[idx]
                gs[root_key] = local_state[1]
                gs[synced_key] = bool(local_state[2])
                moves_by_index[idx] = move

        _run_group(
            model1_indices, model1, model1_search_config, mcts1, sims1,
            "model1_root", "model1_synced", "model1",
        )
        _run_group(
            model2_indices, model2, model2_search_config, mcts2, sims2,
            "model2_root", "model2_synced", "model2",
        )

        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            move = moves_by_index.get(idx)
            if move is None:
                gs["done"] = True
                continue

            _append_eval_history(gs["board_history"], board, config)
            gs["model1_root"], gs["model1_synced"] = _advance_eval_root(gs.get("model1_root"), move)
            gs["model2_root"], gs["model2_synced"] = _advance_eval_root(gs.get("model2_root"), move)
            chess.apply_move(board, move)
            gs["move_count"] += 1
            _record_position_count(gs["position_counts"], board)

            if (
                chess.is_game_over(board, claim_draw=False)
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

    stats = _build_eval_stats(wins, draws, losses, unresolved, len(game_indices))
    stats["completed"] = int(completed)
    stats["cancelled"] = bool(
        stop_event is not None
        and stop_event.is_set()
        and completed < len(game_indices)
    )
    profile = {}
    for mcts in (mcts1, mcts2):
        if mcts is None:
            continue
        for key, value in dict(mcts.get_profile_stats() or {}).items():
            if isinstance(value, (int, float)):
                profile[key] = profile.get(key, 0) + value
    for key, value in eval_search_profile.items():
        profile[key] = profile.get(key, 0) + value
    stats["profile"] = profile
    return stats


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
    use_mcts_model1=True,
    use_mcts_model2=True,
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
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0, plies=0, result="*", model1_as_white=True: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
                "plies": int(plies),
                "result": str(result),
                "model1_as_white": bool(model1_as_white),
            }),
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
                "profile": dict(stats.get("profile", {}) or {}),
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
    use_mcts_model1=True,
    use_mcts_model2=True,
    model1_label="eval_model1",
    model2_label="eval_model2",
    shared_buffer=None,
):
    try:
        rl_cfg = config.get("reinforcement_learning", {})
        torch_threads = max(1, int(rl_cfg.get("eval_torch_threads", rl_cfg.get("self_play_torch_threads", 1)) or 1))
        with contextlib.suppress(Exception):
            torch.set_num_threads(torch_threads)
        with contextlib.suppress(Exception):
            torch.set_num_interop_threads(1)

        timeout_s = float(_central_inference_option(config, "timeout_s", 0) or 0)
        stall_warning_s = float(_central_inference_option(config, "stall_warning_s", 15) or 0)
        transport_dtype = str(_central_inference_option(config, "transport_dtype", "float16") or "float16")
        debug_enabled = bool(rl_cfg.get("eval_central_inference_debug", False))
        shared_call_lock = threading.Lock()

        model1 = _RemoteInferenceModel(
            str(model1_label),
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
            shared_buffer=shared_buffer,
            shared_call_lock=shared_call_lock,
        )
        model2 = _RemoteInferenceModel(
            str(model2_label),
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
            shared_buffer=shared_buffer,
            shared_call_lock=shared_call_lock,
        )

        stats = _evaluate_games_batched(
            model1,
            model2,
            config,
            torch.device("cpu"),
            game_indices,
            use_fixed_openings=use_fixed_openings,
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0, plies=0, result="*", model1_as_white=True: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
                "plies": int(plies),
                "result": str(result),
                "model1_as_white": bool(model1_as_white),
            }),
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
                "profile": dict(stats.get("profile", {}) or {}),
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


def _close_eval_central_runtime(runtime):
    if not runtime or runtime.get("closed"):
        return
    if runtime.get("borrowed"):
        # The persistent self-play pool owns this process and all pipe handles.
        return
    runtime["closed"] = True
    request_queues = list(runtime.get("request_queues", []) or [])
    for request_queue in request_queues:
        with contextlib.suppress(Exception):
            request_queue.put({"cmd": "stop"})
    _terminate_eval_processes(runtime.get("server_processes", []), timeout_s=1.0)
    for queue_obj in request_queues + list(runtime.get("control_queues", []) or []):
        with contextlib.suppress(Exception):
            queue_obj.close()
    for recv_conn in dict(runtime.get("worker_response_receivers", {}) or {}).values():
        with contextlib.suppress(Exception):
            recv_conn.close()
    for sender_map in list(runtime.get("server_response_senders", []) or []):
        for send_conn in dict(sender_map or {}).values():
            with contextlib.suppress(Exception):
                send_conn.close()
    runtime.get("worker_shared_buffers", {}).clear()


def _start_eval_central_runtime(
    model1,
    model2,
    config,
    device,
    num_games,
    *,
    verbose=True,
    ready_callback=None,
):
    """Start one reusable GPU inference server for all stages of an eval funnel."""
    startup_t0 = time.perf_counter()
    workers = _resolve_eval_workers(config, device, num_games)
    server_count = _resolve_eval_central_server_count(config, workers)
    ctx = mp.get_context("spawn")
    request_queues = [ctx.Queue() for _ in range(server_count)]
    control_queues = [ctx.Queue() for _ in range(server_count)]
    server_response_senders = [dict() for _ in range(server_count)]
    worker_response_receivers = {}
    worker_server_idx = {}
    for rank in range(workers):
        server_idx = int(rank) % int(server_count)
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        worker_response_receivers[rank] = recv_conn
        server_response_senders[server_idx][rank] = send_conn
        worker_server_idx[rank] = server_idx

    central_cfg = config.get("central_inference", {}) or {}
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    shared_memory_enabled = bool(
        central_cfg.get("shared_memory_enabled", True)
        and str(central_cfg.get("transport_dtype", "float16")).lower() in {"float16", "fp16"}
    )
    worker_shared_buffers = {}
    if shared_memory_enabled:
        raw_capacity = central_cfg.get(
            "shared_memory_slot_batch_size",
            rl_cfg.get("mcts_batch_size", rl_cfg.get("mcts_simulations", 192)),
        )
        if isinstance(raw_capacity, str) and raw_capacity.strip().lower() == "auto":
            raw_capacity = rl_cfg.get("mcts_batch_size", rl_cfg.get("mcts_simulations", 192))
        capacity = max(1, int(raw_capacity or 192))
        slots = max(1, int(central_cfg.get("shared_memory_slots_per_worker", 2) or 2))
        input_planes = 16 * (1 + int(config.get("model", {}).get("history_positions", 0) or 0))
        worker_shared_buffers = {
            int(rank): create_shared_inference_buffer(
                ctx,
                slots=slots,
                capacity=capacity,
                input_planes=input_planes,
                max_legal_moves=MAX_LEGAL_MOVES,
            )
            for rank in range(workers)
        }

    runtime = {
        "ctx": ctx,
        "workers": workers,
        "server_count": server_count,
        "request_queues": request_queues,
        "control_queues": control_queues,
        "server_response_senders": server_response_senders,
        "worker_response_receivers": worker_response_receivers,
        "worker_server_idx": worker_server_idx,
        "worker_shared_buffers": worker_shared_buffers,
        "model_labels": ("eval_model1", "eval_model2"),
        "server_processes": [],
        "closed": False,
        "stage_count": 0,
    }
    try:
        for server_idx in range(server_count):
            configured_cache_rank = _central_inference_option(config, "compile_cache_rank", None)
            if configured_cache_rank is None:
                compile_rank = 9100 + (int(os.getpid()) % 100000) * 10 + int(server_idx)
            else:
                compile_rank = int(configured_cache_rank) + int(server_idx)
            proc = ctx.Process(
                target=central_inference_server,
                args=(
                    config,
                    int(device.index or 0) if device.type == "cuda" else 0,
                    request_queues[server_idx],
                    server_response_senders[server_idx],
                    control_queues[server_idx],
                    compile_rank,
                    {
                        int(rank): worker_shared_buffers[int(rank)]
                        for rank, assigned_server in worker_server_idx.items()
                        if int(assigned_server) == int(server_idx)
                        and int(rank) in worker_shared_buffers
                    },
                ),
            )
            proc.daemon = True
            proc.start()
            runtime["server_processes"].append(proc)
        runtime["server_spawn_s"] = time.perf_counter() - startup_t0

        task_id = f"eval_runtime_{os.getpid()}_{id(model1)}_{id(model2)}"
        snapshot_t0 = time.perf_counter()
        model1_state = _snapshot_state_dict_cpu_shared(model1)
        model2_state = _snapshot_state_dict_cpu_shared(model2)
        runtime["snapshot_s"] = time.perf_counter() - snapshot_t0
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

        load_timeout_s = float(_central_inference_option(config, "load_timeout_s", 300) or 300)
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
        runtime["model_summary"] = model_summary
        runtime["load_s"] = max(load_times) if load_times else 0.0
        runtime["pid_summary"] = pid_summary
        runtime["startup_s"] = time.perf_counter() - startup_t0
        if ready_callback is not None:
            ready_callback({
                "ready": False,
                "models_ready": True,
                "workers": int(workers),
                "server_count": int(server_count),
                "central_inference": True,
                "load_s": float(runtime["load_s"]),
                "startup_s": float(runtime["startup_s"]),
                "snapshot_s": float(runtime.get("snapshot_s", 0.0) or 0.0),
            })
        if verbose:
            print(
                "Eval inference runtime: "
                f"workers={workers}, servers={server_count}, {model_summary}, "
                f"load={runtime['load_s']:.2f}s, pid={pid_summary or '-'}"
            )
        return runtime
    except Exception:
        _close_eval_central_runtime(runtime)
        raise


def _refresh_eval_central_runtime(
    runtime,
    model1,
    model2,
    config,
    *,
    verbose=True,
    ready_callback=None,
):
    """Replace weights while preserving compiled eval models and GPU process."""
    refresh_t0 = time.perf_counter()
    task_id = f"eval_refresh_{os.getpid()}_{time.time_ns()}"
    snapshot_t0 = time.perf_counter()
    model1_state = _snapshot_state_dict_cpu_shared(model1)
    model2_state = _snapshot_state_dict_cpu_shared(model2)
    model1_label, model2_label = runtime.get(
        "model_labels", ("eval_model1", "eval_model2")
    )
    snapshot_s = time.perf_counter() - snapshot_t0
    for request_queue in runtime["request_queues"]:
        request_queue.put({
            "cmd": "load_models",
            "task_id": task_id,
            "clear": True,
            "models": [
                {"label": model1_label, "state": model1_state, "state_path": None},
                {"label": model2_label, "state": model2_state, "state_path": None},
            ],
        })

    load_timeout_s = float(_central_inference_option(config, "load_timeout_s", 300) or 300)
    deadline = time.time() + max(1.0, load_timeout_s)
    pending_servers = set(range(int(runtime["server_count"])))
    load_messages = []
    while pending_servers:
        if time.time() >= deadline:
            raise TimeoutError(
                "Eval central inference did not acknowledge refreshed models "
                f"from servers {sorted(pending_servers)}."
            )
        for server_idx in list(pending_servers):
            try:
                message = runtime["control_queues"][server_idx].get(timeout=0.25)
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
    runtime["model_summary"] = next(
        (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
        "models ready",
    )
    runtime["pid_summary"] = ",".join(
        str(int(message.get("pid")))
        for message in load_messages
        if message.get("pid") is not None
    )
    runtime["load_s"] = max(load_times) if load_times else 0.0
    runtime["snapshot_s"] = snapshot_s
    runtime["startup_s"] = time.perf_counter() - refresh_t0
    if ready_callback is not None:
        ready_callback({
            "ready": False,
            "models_ready": True,
            "workers": int(runtime["workers"]),
            "server_count": int(runtime["server_count"]),
            "central_inference": True,
            "load_s": float(runtime["load_s"]),
            "startup_s": float(runtime["startup_s"]),
            "snapshot_s": float(snapshot_s),
        })
    if verbose:
        print(
            "Eval inference runtime: "
            f"workers={runtime['workers']}, servers={runtime['server_count']}, "
            f"{runtime['model_summary']}, load={runtime['load_s']:.2f}s, "
            f"pid={runtime['pid_summary'] or '-'}"
        )
    return runtime


def prepare_eval_central_runtime(runtime, model1, model2, config, device, num_games):
    """Start once, then refresh weights without recompiling between iterations."""
    if not _eval_uses_central_inference(config, device):
        if runtime is not None:
            _close_eval_central_runtime(runtime)
        return None

    workers = _resolve_eval_workers(config, device, num_games)
    server_count = _resolve_eval_central_server_count(config, workers)
    runtime_workers = int(runtime.get("workers", 0)) if runtime is not None else 0
    worker_topology_matches = bool(
        runtime_workers == int(workers)
        or (
            runtime is not None
            and runtime.get("borrowed")
            and runtime_workers >= int(workers)
        )
    )
    reusable = bool(
        runtime is not None
        and not runtime.get("closed", False)
        and worker_topology_matches
        and int(runtime.get("server_count", 0)) == int(server_count)
        and all(process.is_alive() for process in runtime.get("server_processes", []))
    )
    if not reusable:
        if runtime is not None:
            _close_eval_central_runtime(runtime)
        return _start_eval_central_runtime(model1, model2, config, device, num_games)
    return _refresh_eval_central_runtime(runtime, model1, model2, config)


def close_eval_central_runtime(runtime):
    _close_eval_central_runtime(runtime)


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
    central_runtime=None,
    progress_callback=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
    show_progress=True,
    verbose=True,
    runtime_ready_callback=None,
):
    owns_runtime = central_runtime is None
    rl_cfg = config.get("reinforcement_learning", {})
    max_moves = _resolve_eval_max_moves(config)
    worker_budget = (
        int(central_runtime["workers"])
        if central_runtime is not None
        else _resolve_eval_workers(config, device, num_games)
    )
    workers = max(1, min(int(num_games), int(worker_budget)))
    ctx = central_runtime["ctx"] if central_runtime is not None else mp.get_context("spawn")
    result_queue = ctx.Queue()
    game_indices_per_worker = [[] for _ in range(workers)]
    for offset, game_idx in enumerate(range(game_index_offset, game_index_offset + num_games)):
        game_indices_per_worker[offset % workers].append(game_idx)

    active_worker_ranks = [
        rank for rank, game_indices in enumerate(game_indices_per_worker)
        if game_indices
    ]
    worker_processes = []
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

    def _build_eval_worker(active_runtime, rank, game_indices):
        request_queues = active_runtime["request_queues"]
        worker_response_receivers = active_runtime["worker_response_receivers"]
        worker_server_idx = active_runtime["worker_server_idx"]
        model1_label, model2_label = active_runtime.get(
            "model_labels", ("eval_model1", "eval_model2")
        )
        return ctx.Process(
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
                use_mcts_model1,
                use_mcts_model2,
                model1_label,
                model2_label,
                dict(active_runtime.get("worker_shared_buffers", {}) or {}).get(int(rank)),
            ),
        )

    def _register_started_worker(rank, proc):
        worker_processes_by_rank[int(rank)] = proc
        if proc not in worker_processes:
            worker_processes.append(proc)

    def _start_eval_worker(active_runtime, rank, game_indices):
        proc = _build_eval_worker(active_runtime, rank, game_indices)
        proc.daemon = True
        proc.start()
        _register_started_worker(rank, proc)
        return proc

    def _start_all_workers(active_runtime):
        pending = [
            (int(rank), _build_eval_worker(active_runtime, rank, game_indices_per_worker[rank]))
            for rank in active_worker_ranks
            if int(rank) not in worker_processes_by_rank
        ]
        for _rank, proc in pending:
            proc.daemon = True
        parallel_start = bool(rl_cfg.get("eval_parallel_worker_start", False)) and len(pending) > 1
        if parallel_start:
            start_threads = max(
                1,
                min(len(pending), int(rl_cfg.get("eval_parallel_worker_start_threads", 4) or 4)),
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=start_threads) as executor:
                    futures = [executor.submit(proc.start) for _rank, proc in pending]
                    for future in futures:
                        future.result()
            except Exception:
                _terminate_eval_processes([proc for _rank, proc in pending])
                raise
        else:
            for _rank, proc in pending:
                proc.start()
        for rank, proc in pending:
            _register_started_worker(rank, proc)
        return list(worker_processes)

    runtime = central_runtime
    try:
        if runtime is None:
            runtime = _start_eval_central_runtime(
                model1,
                model2,
                config,
                device,
                num_games,
                verbose=verbose,
                ready_callback=runtime_ready_callback,
            )
        worker_launch_t0 = time.perf_counter()
        _start_all_workers(runtime)
        runtime["worker_launch_s"] = time.perf_counter() - worker_launch_t0
        runtime["startup_s"] = (
            float(runtime.get("startup_s", 0.0) or 0.0)
            + float(runtime["worker_launch_s"])
        )

        server_count = int(runtime["server_count"])
        server_processes = runtime["server_processes"]
        runtime["stage_count"] = int(runtime.get("stage_count", 0)) + 1
        if verbose:
            print(
                "Eval stage: "
                f"workers={len(active_worker_ranks)}, servers={server_count}, "
                f"batch_games={_resolve_eval_batch_games(config, num_games)}, "
                f"sims={_resolve_eval_mcts_simulations(config)}, "
                f"server={'new' if owns_runtime else 'reused'}"
            )

        if runtime_ready_callback is not None:
            runtime_ready_callback({
                "ready": True,
                "models_ready": True,
                "workers": len(active_worker_ranks),
                "server_count": int(server_count),
                "central_inference": True,
                "load_s": float(runtime.get("load_s", 0.0) or 0.0),
                "startup_s": float(runtime.get("startup_s", 0.0) or 0.0),
                "snapshot_s": float(runtime.get("snapshot_s", 0.0) or 0.0),
                "worker_launch_s": float(runtime.get("worker_launch_s", 0.0) or 0.0),
            })

        wins = 0
        draws = 0
        losses = 0
        unresolved = 0
        completed = 0
        finished_worker_ranks = set()
        eval_profile = {}
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
        worker_error = None
        cancelled = False
        try:
            while len(finished_worker_ranks) < len(active_worker_ranks):
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    _terminate_eval_processes(list(worker_processes_by_rank.values()))
                    break
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
                            _start_eval_worker(runtime, int(rank), remaining)
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
                        if progress_callback is not None:
                            progress_callback(
                                int(message.get("completed", 0)),
                                game_idx,
                                int(message.get("wins", 0)),
                                int(message.get("draws", 0)),
                                int(message.get("losses", 0)),
                                int(message.get("unresolved", 0)),
                                int(message.get("plies", 0)),
                                str(message.get("result", "*")),
                                bool(message.get("model1_as_white", True)),
                            )
                elif message_type == "result":
                    rank = int(message.get("rank", -1))
                    for key, value in dict(message.get("profile", {}) or {}).items():
                        if isinstance(value, (int, float)):
                            eval_profile[key] = eval_profile.get(key, 0) + value
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
                        _start_eval_worker(runtime, rank, remaining)
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
        if unresolved > 0 and verbose:
            _print_eval_unresolved("Eval", unresolved, num_games, max_moves)
        stats = _build_eval_stats(wins, draws, losses, unresolved, num_games)
        stats["profile"] = eval_profile
        requests = int(eval_profile.get("central_inference_requests", 0) or 0)
        if requests > 0 and verbose:
            remote_ms = 1000.0 * float(
                eval_profile.get("central_inference_remote_wait_time", 0.0) or 0.0
            ) / requests
            queue_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_queue_wait_time", 0.0) or 0.0
            ) / requests
            service_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_total_time", 0.0) or 0.0
            ) / requests
            forward_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_forward_time", 0.0) or 0.0
            ) / requests
            server_batch = float(
                eval_profile.get("central_inference_server_batch_items", 0.0) or 0.0
            ) / requests
            search_time = float(eval_profile.get("search_many_time", 0.0) or 0.0)
            inference_wait = float(eval_profile.get("nn_inference_time", 0.0) or 0.0)
            wait_share = inference_wait / search_time if search_time > 0.0 else 0.0
            print(
                "Eval pipeline: "
                f"batch={server_batch:.1f} | response={remote_ms:.2f}ms "
                f"(queue/IPC={queue_ms:.2f}ms, server GPU+copies={service_ms:.2f}ms, "
                f"forward async={forward_ms:.2f}ms) | "
                f"MCTS waiting for inference={wait_share:.0%}"
            )
        stats["cancelled"] = bool(cancelled)
        stats["completed"] = int(completed)
        return stats
    finally:
        _terminate_eval_processes(worker_processes)
        with contextlib.suppress(Exception):
            result_queue.close()
        if owns_runtime:
            _close_eval_central_runtime(runtime)


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


def _horizontal_flip_eligible_mask(boards):
    """Return rows where file mirroring is an exact chess symmetry.

    Castling is asymmetric around the a<->h reflection (the king starts on the
    e-file, not on the centre line).  Plane 12 in every 16-plane history frame
    marks castling rights, so exclude any row where that state is visible.
    """
    if boards.dim() != 4 or int(boards.size(1)) < 16:
        return torch.ones(int(boards.size(0)), dtype=torch.bool, device=boards.device)
    castling_planes = boards[:, 12::16]
    has_castling_rights = castling_planes.abs().flatten(1).amax(dim=1) > 0
    return ~has_castling_rights


def _maybe_augment_batch(
    boards,
    policy_indices,
    policy_values,
    policy_mask,
    legal_indices,
    legal_mask,
    config,
):
    rl_cfg = config.get("reinforcement_learning", {})
    if not rl_cfg.get("use_augmentation", False):
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask
    if not rl_cfg.get("augment_horizontal_flip", False):
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    prob = rl_cfg.get("augment_prob", 0.5)
    if prob <= 0:
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    batch_size = boards.size(0)
    if batch_size == 0:
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    flip_mask = (
        torch.rand(batch_size, device=boards.device) < prob
    ) & _horizontal_flip_eligible_mask(boards)
    if not flip_mask.any():
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])

    if policy_indices.numel() > 0:
        forward_map = _get_hflip_forward_index_map().to(device=policy_indices.device)
        active = (
            flip_mask.unsqueeze(1)
            & policy_mask
            & (policy_indices >= 0)
            & (policy_indices < int(ACTION_SIZE))
        )
        if active.any():
            remapped = forward_map[policy_indices[active].long()].to(dtype=policy_indices.dtype)
            policy_indices[active] = remapped

        # Older replay rows use policy support as their legal support and may
        # therefore alias the same tensor.  Do not mirror shared storage twice.
        shares_policy_storage = (
            legal_indices is policy_indices
            or (
                legal_indices is not None
                and legal_indices.numel() > 0
                and policy_indices.numel() > 0
                and legal_indices.data_ptr() == policy_indices.data_ptr()
            )
        )
        if legal_indices is not None and legal_indices.numel() > 0 and not shares_policy_storage:
            active_legal = (
                flip_mask.unsqueeze(1)
                & legal_mask
                & (legal_indices >= 0)
                & (legal_indices < int(ACTION_SIZE))
            )
            if active_legal.any():
                remapped_legal = forward_map[legal_indices[active_legal].long()].to(
                    dtype=legal_indices.dtype
                )
                legal_indices[active_legal] = remapped_legal

    return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask


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


def _value_error_focus_weights(
    base_weights,
    value_scalar,
    priority_targets,
    *,
    focus_fraction=0.25,
    max_multiplier=1.50,
    min_abs_error=0.05,
):
    """Raise value-loss weight for the largest current supervised errors.

    The selection is batch-local and detached from autograd.  This gives the
    trainer a fresh error-prioritized signal without an extra replay inference
    pass or duplicate samples. ``priority_targets`` must be the same reliable
    target optimized by the weighted loss. In RL this is the final W/D/L
    outcome, not the model-generated root-Q auxiliary target.
    """
    weights = torch.clamp(base_weights.reshape(-1).float(), min=0.0)
    focus_mask = torch.zeros_like(weights, dtype=torch.bool)
    abs_errors = torch.zeros_like(weights)
    fraction = max(0.0, min(1.0, float(focus_fraction)))
    multiplier_cap = max(1.0, float(max_multiplier))
    if fraction <= 0.0 or multiplier_cap <= 1.0 or weights.numel() <= 0:
        return weights, focus_mask, abs_errors

    with torch.no_grad():
        valid_mask = torch.isfinite(priority_targets.reshape(-1))
        clipped_targets = torch.clamp(
            priority_targets.reshape(-1).float(),
            -1.0,
            1.0,
        )
        abs_errors[valid_mask] = torch.abs(
            value_scalar.detach().reshape(-1).float()[valid_mask]
            - clipped_targets[valid_mask]
        )
        candidate_mask = valid_mask & (abs_errors >= max(0.0, float(min_abs_error)))
        valid_indices = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
        if valid_indices.numel() <= 0:
            return weights, focus_mask, abs_errors
        focus_count = min(
            int(valid_indices.numel()),
            max(1, int(round(fraction * int(weights.numel())))),
        )
        selected_local = torch.topk(
            abs_errors[valid_indices],
            k=focus_count,
            largest=True,
            sorted=False,
        ).indices
        selected = valid_indices[selected_local]
        focus_mask[selected] = True
        selected_errors = abs_errors[selected]
        error_scale = torch.clamp(selected_errors.max(), min=1e-6)
        relative_error = torch.clamp(selected_errors / error_scale, 0.0, 1.0)
        weights[selected] *= 1.0 + (multiplier_cap - 1.0) * relative_error
    return weights, focus_mask, abs_errors


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


def _legal_only_log_probs(policy_logits, legal_indices, legal_mask):
    """Normalize policy logits over each position's legal action set."""
    num_classes = int(policy_logits.size(1))
    safe_legal_indices = legal_indices.long().clamp(0, num_classes - 1)
    valid_legal_mask = legal_mask & (legal_indices >= 0) & (legal_indices < num_classes)
    legal_logits = torch.gather(policy_logits.float(), 1, safe_legal_indices)
    legal_logits = legal_logits.masked_fill(~valid_legal_mask, -1.0e9)
    legal_log_probs = F.log_softmax(legal_logits, dim=1)
    legal_log_probs = torch.where(
        valid_legal_mask,
        legal_log_probs,
        torch.zeros_like(legal_log_probs),
    )
    return legal_log_probs, valid_legal_mask, safe_legal_indices


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
    collect_gradient_diagnostics=False,
):
    root_q_targets = None
    search_changed_top = None
    search_q_deltas = None
    best_q_targets = None
    played_q_targets = None
    orig_q_targets = None
    policy_kld_targets = None
    search_visits = None
    if len(batch) >= 18:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights, moves_left_targets, legal_indices, legal_mask, root_q_targets, search_changed_top, search_q_deltas, best_q_targets, played_q_targets, orig_q_targets, policy_kld_targets, search_visits = batch[:18]
    elif len(batch) >= 13:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights, moves_left_targets, legal_indices, legal_mask, root_q_targets, search_changed_top, search_q_deltas = batch[:13]
    elif len(batch) >= 11:
        boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights, value_sample_weights, moves_left_targets, legal_indices, legal_mask, root_q_targets = batch[:11]
    elif len(batch) >= 10:
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

    boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask = _maybe_augment_batch(
        boards,
        policy_indices,
        policy_values,
        policy_mask,
        legal_indices,
        legal_mask,
        config,
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
    if root_q_targets is None:
        root_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    root_q_targets = root_q_targets.to(device, non_blocking=True).reshape(-1)
    if search_changed_top is None:
        search_changed_top = torch.zeros_like(policy_sample_weights, dtype=torch.bool)
    if search_q_deltas is None:
        search_q_deltas = torch.full_like(policy_sample_weights, float("nan"), dtype=torch.float32)
    search_changed_top = search_changed_top.to(device, non_blocking=True).reshape(-1).bool()
    search_q_deltas = search_q_deltas.to(device, non_blocking=True).reshape(-1).float()
    if best_q_targets is None:
        best_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if played_q_targets is None:
        played_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if orig_q_targets is None:
        orig_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if policy_kld_targets is None:
        policy_kld_targets = torch.full_like(policy_sample_weights, float("nan"), dtype=torch.float32)
    if search_visits is None:
        search_visits = torch.zeros_like(policy_sample_weights, dtype=torch.int32)
    best_q_targets = best_q_targets.to(device, non_blocking=True).reshape(-1).float()
    played_q_targets = played_q_targets.to(device, non_blocking=True).reshape(-1).float()
    orig_q_targets = orig_q_targets.to(device, non_blocking=True).reshape(-1).float()
    policy_kld_targets = policy_kld_targets.to(device, non_blocking=True).reshape(-1).float()
    search_visits = search_visits.to(device, non_blocking=True).reshape(-1).float()
    effective_policy_mask = policy_mask & (policy_sample_weights.unsqueeze(1) > 0)

    optimizer.zero_grad(set_to_none=True)

    rl_cfg = config.get("reinforcement_learning", {})
    use_amp = config["hardware"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if config["hardware"].get("use_bfloat16", False) else torch.float16

    value_aux_scalar_loss_weight = float(
        rl_cfg.get("value_aux_scalar_loss_weight", 0.25)
    )
    moves_left_loss_weight = max(0.0, float(rl_cfg.get("moves_left_loss_weight", 0.05)))
    search_q_loss_weight = max(0.0, float(rl_cfg.get("search_q_loss_weight", 0.20)))
    search_error_loss_weight = max(
        0.0, float(rl_cfg.get("search_error_loss_weight", 0.05))
    )
    value_error_focus_fraction = max(
        0.0,
        min(1.0, float(rl_cfg.get("value_error_focus_fraction", 0.25))),
    )
    value_error_focus_max_multiplier = max(
        1.0,
        float(rl_cfg.get("value_error_focus_max_multiplier", 1.50)),
    )
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        policy_logits, value_pred, moves_left_pred, search_q_pred, search_error_pred = model(
            boards,
            apply_log_softmax=False,
            return_moves_left=True,
            return_search_aux=True,
        )
        policy_pred = F.log_softmax(policy_logits.float(), dim=1)
        legal_policy_log_probs, valid_legal_mask, safe_legal_indices = _legal_only_log_probs(
            policy_logits,
            legal_indices,
            legal_mask,
        )
        value_pred_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)
        target_value_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)

        policy_rows_mask = effective_policy_mask.any(dim=1)
        correction_policy_mask = (
            policy_rows_mask
            & search_changed_top
            & torch.isfinite(search_q_deltas)
            & (search_q_deltas > USEFUL_SEARCH_Q_DELTA_MIN)
        )
        policy_effective_weights = torch.clamp(
            policy_sample_weights.to(dtype=policy_pred.dtype),
            min=0.0,
        )
        if policy_indices.numel() == 0:
            policy_loss_per_row = torch.zeros(
                policy_pred.size(0),
                device=policy_pred.device,
                dtype=policy_pred.dtype,
            )
        else:
            policy_loss_per_row = _legal_only_sparse_policy_loss(
                policy_logits,
                policy_indices,
                policy_values,
                effective_policy_mask,
                legal_indices,
                legal_mask,
            )

        target_scalar = _final_outcome_targets(value_targets)
        target_value_std = target_scalar.std(unbiased=False)
        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            hard_target_wdl = _wdl_targets_from_final_outcome(target_scalar)
            target_wdl = hard_target_wdl
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
            value_primary_loss_rows = value_ce_loss
            value_scalar_aux_loss_rows = value_scalar_aux_loss
            value_pred_std = value_scalar.std(unbiased=False).detach()
        else:
            hard_target_wdl = None
            value_probs = None
            value_primary_loss_rows = (value_pred.squeeze() - target_scalar) ** 2
            value_scalar_aux_loss_rows = torch.zeros_like(value_primary_loss_rows)
            value_scalar = value_pred.squeeze()
            value_pred_std = value_scalar.std(unbiased=False).detach()

        (
            effective_value_sample_weights,
            value_error_focus_mask,
            value_priority_abs_errors,
        ) = _value_error_focus_weights(
            value_sample_weights,
            value_scalar,
            target_scalar,
            focus_fraction=value_error_focus_fraction,
            max_multiplier=value_error_focus_max_multiplier,
        )

        if policy_loss_per_row.numel() == 0:
            policy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            policy_weight_total = policy_effective_weights.sum()
            if float(policy_weight_total.detach().item()) > 0.0:
                policy_loss = (
                    policy_loss_per_row * policy_effective_weights
                ).sum() / policy_weight_total
            else:
                policy_loss = policy_loss_per_row.sum() * 0.0
        value_primary_loss = _weighted_mean(
            value_primary_loss_rows,
            effective_value_sample_weights,
        )
        value_scalar_aux_loss = _weighted_mean(
            value_scalar_aux_loss_rows,
            effective_value_sample_weights,
        )
        value_loss = (
            value_primary_loss
            + value_aux_scalar_loss_weight * value_scalar_aux_loss
        )
        # Search-Q is a separate local target. Never pull the final-result WDL
        # logits toward a self-generated MCTS estimate.
        root_q_mask = torch.isfinite(root_q_targets)
        search_q_mask = torch.isfinite(best_q_targets)
        search_q_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        search_error_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        search_error_targets = torch.full_like(target_scalar, float("nan"))
        if search_q_mask.any():
            search_q_values = search_q_pred.reshape(-1)
            search_error_values = search_error_pred.reshape(-1)
            bounded_best_q = torch.clamp(
                best_q_targets.to(dtype=search_q_values.dtype), -1.0, 1.0
            )
            visit_confidence = torch.clamp(
                torch.log1p(torch.clamp(search_visits, min=0.0)) / math.log1p(192.0),
                min=0.25,
                max=1.0,
            ).to(dtype=search_q_values.dtype)
            search_q_loss_rows = F.smooth_l1_loss(
                search_q_values[search_q_mask],
                bounded_best_q[search_q_mask],
                beta=0.15,
                reduction="none",
            )
            search_q_loss = _weighted_mean(
                search_q_loss_rows,
                visit_confidence[search_q_mask],
            )
            # This is a reliability label for search, not the current head's
            # instantaneous regression residual. It remains stable as Q learns.
            search_error_targets = torch.abs(
                bounded_best_q - target_scalar.to(dtype=bounded_best_q.dtype)
            ).detach()
            search_error_loss_rows = F.smooth_l1_loss(
                search_error_values[search_q_mask],
                search_error_targets[search_q_mask],
                beta=0.15,
                reduction="none",
            )
            search_error_loss = _weighted_mean(
                search_error_loss_rows,
                visit_confidence[search_q_mask]
                * torch.clamp(
                    value_sample_weights[search_q_mask].to(dtype=visit_confidence.dtype),
                    min=0.0,
                    max=1.0,
                ),
            )
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
        legal_policy_probs = torch.exp(legal_policy_log_probs) * valid_legal_mask
        legal_entropy_per_row = -(
            legal_policy_probs * legal_policy_log_probs
        ).sum(dim=1)
        rows_with_legal_moves = valid_legal_mask.any(dim=1)
        policy_entropy = (
            legal_entropy_per_row[rows_with_legal_moves].mean()
            if rows_with_legal_moves.any()
            else torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        )
        policy_objective = policy_weight * policy_loss
        value_objective = (
            value_weight * value_loss
            + moves_left_loss_weight * moves_left_loss
            + search_q_loss_weight * search_q_loss
            + search_error_loss_weight * search_error_loss
        )
        loss = policy_objective + value_objective

    task_gradient_diagnostics = (
        _task_gradient_probe(policy_objective, value_objective, model)
        if collect_gradient_diagnostics
        else {
            'policy_probe_norm': 0.0,
            'value_probe_norm': 0.0,
            'policy_value_cosine': 0.0,
        }
    )
    scaler.scale(loss).backward()

    grad_clip = config["reinforcement_learning"].get("grad_clip", 1.0)
    scaler.unscale_(optimizer)
    gradient_family_norms = (
        _gradient_family_norms(model)
        if collect_gradient_diagnostics
        else {'backbone': 0.0, 'policy': 0.0, 'value': 0.0}
    )
    clip_limit = float(grad_clip) if float(grad_clip) > 0.0 else float('inf')
    total_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
    total_grad_norm = float(total_grad_norm_tensor.detach().item())
    grad_clip_scale = (
        min(1.0, clip_limit / max(total_grad_norm, 1e-12))
        if math.isfinite(clip_limit)
        else 1.0
    )
    grad_was_clipped = bool(grad_clip_scale < 1.0)

    scaler.step(optimizer)
    scaler.update()
    if search_q_loss_weight > 0.0 and bool(search_q_mask.any().item()):
        model_owner = getattr(model, '_orig_mod', model)
        ready = getattr(model_owner, 'search_value_ready', None)
        if ready is not None:
            ready.fill_(1.0)

    with torch.no_grad():
        if policy_indices.numel() == 0:
            target_moves = torch.zeros(policy_pred.size(0), dtype=torch.long, device=policy_pred.device)
        else:
            best_sparse_idx = policy_values.argmax(dim=1, keepdim=True)
            target_moves = torch.gather(policy_indices.long(), 1, best_sparse_idx).squeeze(1)
        legal_top_slots = legal_policy_log_probs.masked_fill(
            ~valid_legal_mask,
            -1.0e9,
        ).argmax(dim=1, keepdim=True)
        predicted_legal_moves = torch.gather(
            safe_legal_indices.long(),
            1,
            legal_top_slots,
        ).squeeze(1)
        correction_count = int(correction_policy_mask.sum().item())
        policy_row_count = int(policy_rows_mask.sum().item())
        correction_loss_sum = float(
            policy_loss_per_row[correction_policy_mask].sum().detach().item()
        ) if correction_count > 0 else 0.0
        correction_top1_correct = int(
            (predicted_legal_moves[correction_policy_mask] == target_moves[correction_policy_mask]).sum().item()
        ) if correction_count > 0 else 0
        policy_effective_weight_sum = float(policy_effective_weights.sum().detach().item())
        correction_effective_weight_sum = float(
            policy_effective_weights[correction_policy_mask].sum().detach().item()
        ) if correction_count > 0 else 0.0
        value_scalar_f32 = value_scalar.detach().float().reshape(-1)
        search_q_mask_f32 = search_q_mask.detach()
        search_q_pred_f32 = search_q_pred.detach().float().reshape(-1)[search_q_mask_f32]
        search_q_target_f32 = torch.clamp(
            best_q_targets.detach().float().reshape(-1)[search_q_mask_f32],
            -1.0,
            1.0,
        )
        search_error_pred_f32 = search_error_pred.detach().float().reshape(-1)[search_q_mask_f32]
        search_error_target_f32 = search_error_targets.detach().float().reshape(-1)[search_q_mask_f32]
        target_scalar_f32 = target_scalar.detach().float().reshape(-1)
        value_row_count = int(value_scalar_f32.numel())
        value_weight_f32 = effective_value_sample_weights.detach().float().reshape(-1)
        value_error_focus_count = int(value_error_focus_mask.sum().item())
        value_error_focus_weight_sum = float(
            value_weight_f32[value_error_focus_mask].sum().item()
        ) if value_error_focus_count > 0 else 0.0
        value_effective_weight_sum = float(value_weight_f32.sum().item())
        value_error_focus_abs_error_sum = float(
            value_priority_abs_errors[value_error_focus_mask].sum().item()
        ) if value_error_focus_count > 0 else 0.0
        wdl_pred_sums = [0.0, 0.0, 0.0]
        wdl_target_sums = [0.0, 0.0, 0.0]
        wdl_brier_sum = 0.0
        wdl_ece_counts = [0] * 10
        wdl_ece_confidence_sums = [0.0] * 10
        wdl_ece_correct_sums = [0.0] * 10
        value_phase_wdl = {}
        if value_probs is not None and hard_target_wdl is not None:
            value_probs_f32 = value_probs.detach().float()
            hard_target_wdl_f32 = hard_target_wdl.detach().float()
            wdl_pred_sums = value_probs_f32.sum(dim=0).cpu().tolist()
            wdl_target_sums = hard_target_wdl_f32.sum(dim=0).cpu().tolist()
            wdl_brier_sum = float(
                torch.square(value_probs_f32 - hard_target_wdl_f32).sum(dim=1).sum().item()
            )
            confidence, predicted_class = value_probs_f32.max(dim=1)
            target_class = hard_target_wdl_f32.argmax(dim=1)
            correct = predicted_class.eq(target_class).float()
            bin_indices = torch.clamp((confidence * 10.0).long(), min=0, max=9)
            for bin_idx in range(10):
                bin_mask = bin_indices == bin_idx
                if bin_mask.any():
                    wdl_ece_counts[bin_idx] = int(bin_mask.sum().item())
                    wdl_ece_confidence_sums[bin_idx] = float(confidence[bin_mask].sum().item())
                    wdl_ece_correct_sums[bin_idx] = float(correct[bin_mask].sum().item())

            if boards.dim() >= 4 and boards.size(1) > 15:
                fullmoves = torch.clamp(
                    torch.round(boards[:, 15, 0, 0].float() * 100.0),
                    min=1.0,
                    max=float(rl_cfg.get("value_phase_max_fullmove", 120)),
                )
                opening_max = int(rl_cfg.get("value_phase_opening_max_fullmove", 12))
                endgame_min = int(rl_cfg.get("value_phase_endgame_min_fullmove", 40))
                phase_masks = {
                    "opening": fullmoves <= opening_max,
                    "middlegame": (fullmoves > opening_max) & (fullmoves < endgame_min),
                    "endgame": fullmoves >= endgame_min,
                }
                for phase_name, phase_mask in phase_masks.items():
                    phase_count = int(phase_mask.sum().item())
                    value_phase_wdl[phase_name] = {
                        "rows": phase_count,
                        "draw_pred_sum": float(value_probs_f32[phase_mask, 1].sum().item()),
                        "draw_target_sum": float(hard_target_wdl_f32[phase_mask, 1].sum().item()),
                    }
        root_q_row_count = int(root_q_mask.sum().item())
        if root_q_row_count > 0:
            root_q_pred_f32 = value_scalar_f32[root_q_mask]
            root_q_target_f32 = torch.clamp(
                root_q_targets.detach().float().reshape(-1)[root_q_mask],
                -1.0,
                1.0,
            )
            root_q_error_f32 = root_q_pred_f32 - root_q_target_f32
        else:
            root_q_pred_f32 = value_scalar_f32[:0]
            root_q_target_f32 = target_scalar_f32[:0]
            root_q_error_f32 = value_scalar_f32[:0]

    if metrics_calc is not None:
        with torch.no_grad():
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

    policy_diagnostics = {
        "correction_rows": correction_count,
        "policy_rows": policy_row_count,
        "correction_loss_sum": correction_loss_sum,
        "correction_top1_correct": correction_top1_correct,
        "effective_weight_sum": policy_effective_weight_sum,
        "correction_effective_weight_sum": correction_effective_weight_sum,
        "value_primary_loss": float(value_primary_loss.detach().item()),
        "value_scalar_aux_loss": float(value_scalar_aux_loss.detach().item()),
        "moves_left_loss": float(moves_left_loss.detach().item()),
        "search_q_loss": float(search_q_loss.detach().item()),
        "search_error_loss": float(search_error_loss.detach().item()),
        "search_q_rows": int(search_q_mask.sum().item()),
        "search_q_pred_sum": float(search_q_pred_f32.sum().item()),
        "search_q_target_sum": float(search_q_target_f32.sum().item()),
        "search_q_abs_error_sum": float(
            (search_q_pred_f32 - search_q_target_f32).abs().sum().item()
        ),
        "search_error_pred_sum": float(search_error_pred_f32.sum().item()),
        "search_error_target_sum": float(search_error_target_f32.sum().item()),
        "search_error_abs_error_sum": float(
            (search_error_pred_f32 - search_error_target_f32).abs().sum().item()
        ),
        "value_error_focus_rows": value_error_focus_count,
        "value_error_focus_weight_sum": value_error_focus_weight_sum,
        "value_effective_weight_sum": value_effective_weight_sum,
        "value_error_focus_abs_error_sum": value_error_focus_abs_error_sum,
        "wdl_pred_sums": wdl_pred_sums,
        "wdl_target_sums": wdl_target_sums,
        "wdl_brier_sum": wdl_brier_sum,
        "wdl_ece_counts": wdl_ece_counts,
        "wdl_ece_confidence_sums": wdl_ece_confidence_sums,
        "wdl_ece_correct_sums": wdl_ece_correct_sums,
        "value_phase_wdl": value_phase_wdl,
        # Raw sufficient statistics let the iteration logger aggregate exact
        # means/correlation across uneven final batches without retaining any
        # per-position tensors.
        "value_rows": value_row_count,
        "value_pred_sum": float(value_scalar_f32.sum().item()),
        "value_target_sum": float(target_scalar_f32.sum().item()),
        "root_q_rows": root_q_row_count,
        "root_q_pred_sum": float(root_q_pred_f32.sum().item()),
        "root_q_target_sum": float(root_q_target_f32.sum().item()),
        "root_q_abs_error_sum": float(root_q_error_f32.abs().sum().item()),
        "root_q_error_sum": float(root_q_error_f32.sum().item()),
        "root_q_pred_sq_sum": float((root_q_pred_f32 * root_q_pred_f32).sum().item()),
        "root_q_target_sq_sum": float((root_q_target_f32 * root_q_target_f32).sum().item()),
        "root_q_cross_sum": float((root_q_pred_f32 * root_q_target_f32).sum().item()),
        "total_grad_norm": total_grad_norm,
        "grad_clip_scale": grad_clip_scale,
        "grad_was_clipped": 1 if grad_was_clipped else 0,
        "grad_backbone_norm": gradient_family_norms['backbone'],
        "grad_policy_head_norm": gradient_family_norms['policy'],
        "grad_value_head_norm": gradient_family_norms['value'],
        **task_gradient_diagnostics,
    }
    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        float(policy_entropy.detach().item()),
        float(value_pred_std.detach().item()),
        float(target_value_std.detach().item()),
        policy_diagnostics,
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
    central_runtime=None,
    progress_callback=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
    show_progress=True,
    verbose=True,
    runtime_ready_callback=None,
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
            central_runtime=central_runtime,
            progress_callback=progress_callback,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
            stop_event=stop_event,
            show_progress=show_progress,
            verbose=verbose,
            runtime_ready_callback=runtime_ready_callback,
        )

    workers = _resolve_eval_workers(config, device, num_games)
    if workers <= 1:
        if runtime_ready_callback is not None:
            runtime_ready_callback({
                "ready": True,
                "workers": 1,
                "server_count": 0,
                "central_inference": False,
                "load_s": 0.0,
            })
        max_moves = _resolve_eval_max_moves(config)
        game_indices = list(range(game_index_offset, game_index_offset + num_games))
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
        def _on_progress(completed, *details):
            eval_bar.update(int(completed))
            if progress_callback is not None:
                progress_callback(int(completed), *details)
        try:
            stats = _evaluate_games_batched(
                model1,
                model2,
                config,
                device,
                game_indices,
                use_fixed_openings=use_fixed_openings,
                progress_callback=_on_progress,
                model1_mcts_config=model1_mcts_config,
                model2_mcts_config=model2_mcts_config,
                use_mcts_model1=use_mcts_model1,
                use_mcts_model2=use_mcts_model2,
                stop_event=stop_event,
            )
        finally:
            eval_bar.close()

        unresolved = int((stats or {}).get("unresolved", 0))
        if unresolved > 0 and verbose:
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
                use_mcts_model1,
                use_mcts_model2,
            ),
        )
        p.start()
        processes.append(p)

    if runtime_ready_callback is not None:
        runtime_ready_callback({
            "ready": True,
            "workers": len(processes),
            "server_count": 0,
            "central_inference": False,
            "load_s": 0.0,
        })

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    eval_profile = {}
    eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
    worker_error = None
    cancelled = False
    try:
        finished_workers = 0
        while finished_workers < len(processes):
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                _terminate_eval_processes(processes)
                break
            try:
                message = result_queue.get(timeout=1.0)
            except Exception:
                continue
            message_type = message.get("type")
            if message_type == "progress":
                completed += int(message.get("completed", 0))
                eval_bar.n = min(num_games, completed)
                eval_bar.refresh()
                if progress_callback is not None:
                    progress_callback(
                        int(message.get("completed", 0)),
                        message.get("game_idx", None),
                        int(message.get("wins", 0)),
                        int(message.get("draws", 0)),
                        int(message.get("losses", 0)),
                        int(message.get("unresolved", 0)),
                        int(message.get("plies", 0)),
                        str(message.get("result", "*")),
                        bool(message.get("model1_as_white", True)),
                    )
            elif message_type == "result":
                wins += int(message.get("wins", 0))
                draws += int(message.get("draws", 0))
                losses += int(message.get("losses", 0))
                unresolved += int(message.get("unresolved", 0))
                for key, value in dict(message.get("profile", {}) or {}).items():
                    if isinstance(value, (int, float)):
                        eval_profile[key] = eval_profile.get(key, 0) + value
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

    if unresolved > 0 and verbose:
        max_moves = _resolve_eval_max_moves(config)
        _print_eval_unresolved("Eval", unresolved, num_games, max_moves)

    stats = _build_eval_stats(wins, draws, losses, unresolved, num_games)
    stats["profile"] = eval_profile
    stats["cancelled"] = bool(cancelled)
    stats["completed"] = int(completed)
    return stats


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
