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

from src.batch_selfplay import BatchMCTS, select_move_by_visits, _SELFPLAY_OPENING_LINES
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


def _resolve_eval_workers(config, device, num_games):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_workers = rl_cfg.get("eval_workers", None)

    if raw_workers is None:
        if device.type == "cuda":
            # Eval loads full models + MCTS state per worker, so CUDA eval is
            # much more memory-sensitive than self-play. Default to a single
            # worker on GPU unless the user explicitly opts into more.
            raw_workers = 1
        else:
            raw_workers = max(1, (os.cpu_count() or 2) - 1)

    try:
        workers = int(raw_workers)
    except Exception:
        workers = 1

    return max(1, min(int(num_games), workers))


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


def _get_eval_opening_prefix(config, game_idx, enabled_override=None):
    enabled = _resolve_eval_fixed_openings_enabled(config) if enabled_override is None else bool(enabled_override)
    if not enabled or not _SELFPLAY_OPENING_LINES:
        return ()

    line = _SELFPLAY_OPENING_LINES[int(game_idx) % len(_SELFPLAY_OPENING_LINES)]
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
            policy_logits, _value = model(board_tensor, apply_log_softmax=False)
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


def _result_for_model1(model1_as_white, result):
    if result == "1/2-1/2":
        return 0, 1, 0
    if result == "1-0":
        return (1, 0, 0) if model1_as_white else (0, 0, 1)
    if result == "0-1":
        return (1, 0, 0) if not model1_as_white else (0, 0, 1)
    return 0, 0, 0


def _eval_worker(rank, model1_state, model2_state, config, device_str, game_indices, result_queue, use_fixed_openings=None):
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

        mcts1 = BatchMCTS(model1, worker_config, device)
        mcts2 = BatchMCTS(model2, worker_config, device)

        wins = 0
        draws = 0
        losses = 0
        unresolved = 0
        sims = _resolve_eval_mcts_simulations(worker_config)
        max_moves = _resolve_eval_max_moves(worker_config)
        auto_claim_draw = _resolve_eval_auto_claim_draw(worker_config)
        claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(worker_config)
        claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(worker_config)

        for game_idx in game_indices:
            model1_as_white = (game_idx % 2 == 0)
            white_mcts = mcts1 if model1_as_white else mcts2
            black_mcts = mcts2 if model1_as_white else mcts1
            opening_prefix = _get_eval_opening_prefix(worker_config, game_idx, enabled_override=use_fixed_openings)
            result, was_unresolved = _evaluate_single_game(
                white_mcts,
                black_mcts,
                sims,
                max_moves,
                auto_claim_draw,
                claim_draw_after_moves,
                claim_repetition_after_moves=claim_repetition_after_moves,
                opening_prefix=opening_prefix,
            )
            if was_unresolved:
                unresolved += 1
                result_queue.put({"type": "progress", "rank": rank, "completed": 1})
                continue
            game_wins, game_draws, game_losses = _result_for_model1(model1_as_white, result)
            wins += game_wins
            draws += game_draws
            losses += game_losses
            result_queue.put({"type": "progress", "rank": rank, "completed": 1})

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "unresolved": unresolved,
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


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


def train_on_batch_rl(model, optimizer, batch, config, device, scaler, metrics_calc=None, value_weight_override=None):
    boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights = batch
    if config["reinforcement_learning"].get("replay_fp16", False):
        boards = boards.float()
        policy_values = policy_values.float()
        value_targets = value_targets.float()

    boards, policy_indices, policy_values, policy_mask = _maybe_augment_batch(
        boards, policy_indices, policy_values, policy_mask, config
    )
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_indices = policy_indices.to(device, non_blocking=True)
    policy_values = policy_values.to(device, non_blocking=True)
    policy_mask = policy_mask.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    policy_sample_weights = policy_sample_weights.to(device, non_blocking=True)
    effective_policy_mask = policy_mask & (policy_sample_weights.unsqueeze(1) > 0)

    value_target_noise_std = float(config.get("reinforcement_learning", {}).get("value_target_noise_std", 0.0))
    if value_target_noise_std > 0:
        value_targets = torch.clamp(
            value_targets + torch.randn_like(value_targets) * value_target_noise_std,
            min=-1.0,
            max=1.0,
        )

    optimizer.zero_grad(set_to_none=True)

    use_amp = config["hardware"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if config["hardware"].get("use_bfloat16", False) else torch.float16

    value_aux_scalar_loss_weight = float(
        config.get("reinforcement_learning", {}).get("value_aux_scalar_loss_weight", 0.25)
    )

    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards)
        value_pred_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)
        target_value_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)

        if policy_indices.numel() == 0:
            policy_loss = torch.zeros(policy_pred.size(0), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            safe_indices = policy_indices.long().clamp_min(0)
            gathered_log_probs = torch.gather(policy_pred, 1, safe_indices)
            gathered_log_probs = torch.where(effective_policy_mask, gathered_log_probs, torch.zeros_like(gathered_log_probs))
            policy_loss = -(policy_values * gathered_log_probs).sum(dim=1)
            policy_loss = policy_loss * policy_sample_weights.to(dtype=policy_loss.dtype)

        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            target_scalar = value_targets.squeeze()
            target_value_std = target_scalar.std(unbiased=False)
            target_win = torch.clamp(target_scalar, min=0.0, max=1.0)
            target_loss = torch.clamp(-target_scalar, min=0.0, max=1.0)
            target_draw = torch.clamp(1.0 - torch.abs(target_scalar), min=0.0, max=1.0)
            target_wdl = torch.stack((target_win, target_draw, target_loss), dim=1)
            target_wdl = target_wdl / target_wdl.sum(dim=1, keepdim=True).clamp_min(1e-8)

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
            value_scalar_detached = value_scalar.detach()
            value_pred_std = value_scalar_detached.std(unbiased=False)
        else:
            target_scalar = value_targets.squeeze()
            target_value_std = target_scalar.std(unbiased=False)
            value_loss = (value_pred.squeeze() - target_scalar) ** 2
            value_pred_std = value_pred.detach().squeeze().std(unbiased=False)

        if policy_loss.numel() == 0:
            policy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            policy_weight_total = policy_sample_weights.to(dtype=policy_loss.dtype).sum()
            if float(policy_weight_total.detach().item()) > 0.0:
                policy_loss = policy_loss.sum() / policy_weight_total
            else:
                policy_loss = policy_loss.sum() * 0.0
        value_loss = value_loss.mean()
        policy_weight = config["reinforcement_learning"]["policy_loss_weight"]
        value_weight = (
            float(config["reinforcement_learning"]["value_loss_weight"])
            if value_weight_override is None
            else float(value_weight_override)
        )
        loss = policy_weight * policy_loss + value_weight * value_loss

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
            metrics_calc.update(policy_pred, value_pred, target_moves, value_targets.unsqueeze(1))

    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        float(policy_entropy.detach().item()),
        float(value_pred_std.detach().item()),
        float(target_value_std.detach().item()),
    )


def evaluate_models(model1, model2, config, device, num_games=100, game_index_offset=0, use_fixed_openings=None):
    workers = _resolve_eval_workers(config, device, num_games)
    if workers <= 1:
        mcts1 = BatchMCTS(model1, config, device)
        mcts2 = BatchMCTS(model2, config, device)

        wins = 0
        draws = 0
        losses = 0
        unresolved = 0
        sims = _resolve_eval_mcts_simulations(config)
        max_moves = _resolve_eval_max_moves(config)
        auto_claim_draw = _resolve_eval_auto_claim_draw(config)
        claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(config)
        claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(config)

        eval_bar = tqdm(total=num_games, desc="Eval vs best", unit="game")
        try:
            for game_idx in range(game_index_offset, game_index_offset + num_games):
                model1_as_white = (game_idx % 2 == 0)
                white_mcts = mcts1 if model1_as_white else mcts2
                black_mcts = mcts2 if model1_as_white else mcts1
                opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
                result, was_unresolved = _evaluate_single_game(
                    white_mcts,
                    black_mcts,
                    sims,
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
            print(
                f"Eval unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
                f"{unresolved}/{num_games} -> excluded from draw count"
            )

        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "unresolved": unresolved,
            "num_games": num_games,
            "score_rate": (wins + 0.5 * draws) / num_games,
            "win_rate": wins / num_games,
            "draw_rate": draws / num_games,
            "loss_rate": losses / num_games,
            "resolved_games": num_games - unresolved,
        }

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
            ),
        )
        p.start()
        processes.append(p)

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    eval_bar = tqdm(total=num_games, desc="Eval vs best", unit="game")
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
        print(
            f"Eval unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
            f"{unresolved}/{num_games} -> excluded from draw count"
        )

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unresolved": unresolved,
        "num_games": num_games,
        "score_rate": (wins + 0.5 * draws) / num_games,
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games,
        "resolved_games": num_games - unresolved,
    }


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
        print(
            f"No-MCTS eval unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
            f"{unresolved}/{num_games} -> excluded from draw count"
        )

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
