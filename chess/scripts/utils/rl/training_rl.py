"""
Reinforcement learning training helpers
"""

import sys
import os
import queue
from pathlib import Path

import torch
import chess
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

from .replay import PrioritizedReplayBuffer

# Add project root to path for src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.batch_selfplay import BatchMCTS, select_move_by_visits
from src.model import ChessNet, normalize_state_dict_keys
from src.utils.data_helpers import (
    ACTION_SIZE,
    build_hflip_inverse_index_map,
)

_HFLIP_INV_INDEX_MAP = None
_HFLIP_FWD_INDEX_MAP = None


def _resolve_eval_workers(config, device, num_games):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_workers = rl_cfg.get('eval_workers', None)

    if raw_workers is None:
        if device.type == 'cuda':
            raw_workers = rl_cfg.get('self_play_workers_per_gpu', 1)
        else:
            raw_workers = max(1, (os.cpu_count() or 2) - 1)

    try:
        workers = int(raw_workers)
    except Exception:
        workers = 1

    return max(1, min(int(num_games), workers))


def _resolve_eval_max_moves(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('eval_max_moves', 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _eval_worker(
    rank,
    model1_state,
    model2_state,
    config,
    device_str,
    game_indices,
    result_queue,
):
    try:
        worker_config = dict(config)
        worker_config['model'] = dict(config.get('model', {}))
        worker_config['model']['print_summary'] = False

        device = torch.device(device_str)
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')

        model1 = ChessNet(worker_config).to(device)
        model2 = ChessNet(worker_config).to(device)
        if device.type == 'cuda':
            model1 = model1.to(memory_format=torch.channels_last)
            model2 = model2.to(memory_format=torch.channels_last)

        model1.load_state_dict(model1_state)
        model2.load_state_dict(model2_state)
        model1.eval()
        model2.eval()

        mcts1 = BatchMCTS(model1, worker_config, device)
        mcts2 = BatchMCTS(model2, worker_config, device)

        wins = 0.0
        draws = 0.0
        unresolved = 0
        sims = int(worker_config.get('reinforcement_learning', {}).get('eval_mcts_simulations', 50))
        max_moves = _resolve_eval_max_moves(worker_config)

        for game_idx in game_indices:
            board = chess.Board()

            if game_idx % 2 == 0:
                current_mcts = mcts1
                other_mcts = mcts2
            else:
                current_mcts = mcts2
                other_mcts = mcts1

            move_count = 0
            while not board.is_game_over(claim_draw=True) and move_count < max_moves:
                mcts = current_mcts if board.turn == chess.WHITE else other_mcts
                visit_counts = mcts.search(board, num_simulations=sims, add_root_noise=False)
                move, _ = select_move_by_visits(visit_counts, temperature=0)

                mcts1.update_history(board)
                mcts2.update_history(board)
                mcts1.advance_root(move)
                mcts2.advance_root(move)

                board.push(move)
                move_count += 1

            mcts1.reset_tree()
            mcts2.reset_tree()

            result = board.result(claim_draw=True)
            if result == '*':
                unresolved += 1
                draws += 0.5
                result_queue.put({'type': 'progress', 'rank': rank, 'completed': 1})
                continue
            if game_idx % 2 == 0:
                if result == '1-0':
                    wins += 1.0
                elif result == '1/2-1/2':
                    draws += 0.5
            else:
                if result == '0-1':
                    wins += 1.0
                elif result == '1/2-1/2':
                    draws += 0.5

            result_queue.put({'type': 'progress', 'rank': rank, 'completed': 1})

        result_queue.put({'type': 'result', 'rank': rank, 'wins': wins, 'draws': draws, 'unresolved': unresolved})
    except KeyboardInterrupt:
        result_queue.put({'type': 'interrupt', 'rank': rank})
    except Exception as exc:
        result_queue.put({'type': 'error', 'rank': rank, 'error': str(exc)})


def _get_hflip_inverse_index_map():
    """
    Build inverse index map for horizontal flip (a<->h).

    We return inverse_map so that:
        flipped_policy = policy[:, inverse_map]
    which ensures flipped_policy[mirror(move)] = policy[move].
    """
    global _HFLIP_INV_INDEX_MAP
    if _HFLIP_INV_INDEX_MAP is not None:
        return _HFLIP_INV_INDEX_MAP

    inverse_map = build_hflip_inverse_index_map()
    if len(inverse_map) != ACTION_SIZE:
        raise ValueError(
            f"Horizontal flip map length mismatch: got {len(inverse_map)}, expected {ACTION_SIZE}"
        )
    _HFLIP_INV_INDEX_MAP = torch.from_numpy(inverse_map).long()
    return _HFLIP_INV_INDEX_MAP


def _get_hflip_forward_index_map():
    """
    Build forward action index map for horizontal flip (a<->h).

    If `inv_map[mirrored_idx] = original_idx`, then
    `forward_map[original_idx] = mirrored_idx`.
    """
    global _HFLIP_FWD_INDEX_MAP
    if _HFLIP_FWD_INDEX_MAP is not None:
        return _HFLIP_FWD_INDEX_MAP

    inv_map = _get_hflip_inverse_index_map()
    forward_map = torch.empty_like(inv_map)
    forward_map[inv_map] = torch.arange(inv_map.numel(), dtype=inv_map.dtype)
    _HFLIP_FWD_INDEX_MAP = forward_map
    return _HFLIP_FWD_INDEX_MAP


def _maybe_augment_batch(boards, policy_indices, policy_values, policy_mask, config):
    """
    Apply simple symmetry augmentation (horizontal flip).
    """
    rl_cfg = config.get('reinforcement_learning', {})
    if not rl_cfg.get('use_augmentation', False):
        return boards, policy_indices, policy_values, policy_mask
    if not rl_cfg.get('augment_horizontal_flip', False):
        return boards, policy_indices, policy_values, policy_mask

    prob = rl_cfg.get('augment_prob', 0.5)
    if prob <= 0:
        return boards, policy_indices, policy_values, policy_mask

    batch_size = boards.size(0)
    if batch_size == 0:
        return boards, policy_indices, policy_values, policy_mask

    flip_mask = torch.rand(batch_size) < prob
    if not flip_mask.any():
        return boards, policy_indices, policy_values, policy_mask

    # Flip board tensors across files (a<->h)
    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])

    # Remap sparse policy targets to match flipped board.
    if policy_indices.numel() > 0:
        forward_map = _get_hflip_forward_index_map()
        active = flip_mask.unsqueeze(1) & policy_mask
        if active.any():
            remapped = forward_map[policy_indices[active].long()].to(dtype=policy_indices.dtype)
            policy_indices[active] = remapped

    return boards, policy_indices, policy_values, policy_mask

def train_on_batch_rl(
    model,
    optimizer,
    batch,
    indices,
    weights,
    config,
    device,
    scaler,
    replay_buffer,
    metrics_calc=None,
    value_weight_override=None,
):
    """Train on batch with optional prioritized replay and metrics"""
    boards, policy_indices, policy_values, policy_mask, value_targets = batch
    if config['reinforcement_learning'].get('replay_fp16', False):
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

    value_target_noise_std = float(
        config.get('reinforcement_learning', {}).get('value_target_noise_std', 0.0)
    )
    if value_target_noise_std > 0:
        value_targets = torch.clamp(
            value_targets + torch.randn_like(value_targets) * value_target_noise_std,
            min=-1.0,
            max=1.0,
        )
    
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards)
        value_pred_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)

        if policy_indices.numel() == 0:
            policy_loss = torch.zeros(policy_pred.size(0), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            safe_indices = policy_indices.long().clamp_min(0)
            gathered_log_probs = torch.gather(policy_pred, 1, safe_indices)
            gathered_log_probs = torch.where(
                policy_mask,
                gathered_log_probs,
                torch.zeros_like(gathered_log_probs),
            )
            policy_loss = -(policy_values * gathered_log_probs).sum(dim=1)
        
        # Value loss: WDL CE when using WDL head, otherwise MSE
        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            # Targets are scalar {-1, 0, 1} -> map to WDL classes [W, D, L]
            target_scalar = value_targets.squeeze()
            target_classes = torch.zeros_like(target_scalar, dtype=torch.long)
            target_classes[target_scalar > 0.9] = 0
            target_classes[target_scalar < -0.9] = 2
            target_classes[(target_scalar >= -0.9) & (target_scalar <= 0.9)] = 1
            
            value_loss = F.cross_entropy(value_pred, target_classes, reduction='none')
            wdl_probs_detached = torch.softmax(value_pred.detach(), dim=1)
            value_scalar_detached = wdl_probs_detached[:, 0] - wdl_probs_detached[:, 2]
            value_pred_std = value_scalar_detached.std(unbiased=False)
        else:
            value_loss = (value_pred.squeeze() - value_targets.squeeze()) ** 2
            value_pred_std = value_pred.detach().squeeze().std(unbiased=False)
        
        if weights is not None:
            policy_loss = (policy_loss * weights).mean()
            value_loss = (value_loss * weights).mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss = value_loss.mean()
        
        policy_weight = config['reinforcement_learning']['policy_loss_weight']
        if value_weight_override is None:
            value_weight = float(config['reinforcement_learning']['value_loss_weight'])
        else:
            value_weight = float(value_weight_override)
        loss = policy_weight * policy_loss + value_weight * value_loss

        # Entropy regularization (optional)
        policy_probs = torch.exp(policy_pred)
        policy_entropy = -(policy_probs * policy_pred).sum(dim=1).mean()
        entropy_weight = config['reinforcement_learning'].get('entropy_weight', 0.0)
        if entropy_weight > 0:
            loss = loss - entropy_weight * policy_entropy
    
    scaler.scale(loss).backward()
    
    grad_clip = config['reinforcement_learning'].get('grad_clip', 1.0)
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    if isinstance(replay_buffer, PrioritizedReplayBuffer):
        with torch.no_grad():
            if value_pred.dim() == 2 and value_pred.size(1) == 3:
                # Convert WDL logits to scalar for priority
                wdl_probs = torch.softmax(value_pred, dim=1)
                value_scalar = (wdl_probs[:, 0] - wdl_probs[:, 2])
                td_errors = torch.abs(value_scalar - value_targets.squeeze())
            else:
                td_errors = torch.abs(value_pred.squeeze() - value_targets.squeeze())
            td_errors = td_errors.cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)
    
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
    )


def evaluate_models(model1, model2, config, device, num_games=100):
    """
    Evaluate model1 vs model2
    
    🔧 v4.4: Increased from 20→100 games for statistical significance
    """
    workers = _resolve_eval_workers(config, device, num_games)
    if workers <= 1:
        mcts1 = BatchMCTS(model1, config, device)
        mcts2 = BatchMCTS(model2, config, device)

        wins = 0.0
        draws = 0.0
        unresolved = 0
        sims = int(config.get('reinforcement_learning', {}).get('eval_mcts_simulations', 50))
        max_moves = _resolve_eval_max_moves(config)

        eval_bar = tqdm(total=num_games, desc="⚔️ Eval vs best", unit="gra")
        try:
            for game_idx in range(num_games):
                board = chess.Board()

                if game_idx % 2 == 0:
                    current_mcts = mcts1
                    other_mcts = mcts2
                else:
                    current_mcts = mcts2
                    other_mcts = mcts1

                move_count = 0
                while not board.is_game_over(claim_draw=True) and move_count < max_moves:
                    mcts = current_mcts if board.turn == chess.WHITE else other_mcts
                    visit_counts = mcts.search(board, num_simulations=sims, add_root_noise=False)
                    move, _ = select_move_by_visits(visit_counts, temperature=0)

                    mcts1.update_history(board)
                    mcts2.update_history(board)
                    mcts1.advance_root(move)
                    mcts2.advance_root(move)

                    board.push(move)
                    move_count += 1

                mcts1.reset_tree()
                mcts2.reset_tree()

                result = board.result(claim_draw=True)
                if result == '*':
                    unresolved += 1
                    draws += 0.5
                    eval_bar.update(1)
                    continue
                if game_idx % 2 == 0:
                    if result == '1-0':
                        wins += 1.0
                    elif result == '1/2-1/2':
                        draws += 0.5
                else:
                    if result == '0-1':
                        wins += 1.0
                    elif result == '1/2-1/2':
                        draws += 0.5

                eval_bar.update(1)
        finally:
            eval_bar.close()

        if unresolved > 0:
            print(f"Eval unresolved at move cap ({max_moves}): {unresolved}/{num_games} -> counted as draws")
        return (wins + draws) / num_games

    model1_state = normalize_state_dict_keys(model1.state_dict())
    model2_state = normalize_state_dict_keys(model2.state_dict())
    model1_state = {k: v.detach().to(device='cpu', copy=True) for k, v in model1_state.items()}
    model2_state = {k: v.detach().to(device='cpu', copy=True) for k, v in model2_state.items()}

    game_buckets = [[] for _ in range(workers)]
    for idx, game_idx in enumerate(range(num_games)):
        game_buckets[idx % workers].append(game_idx)

    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []
    worker_device = str(device if device.type != 'cuda' or torch.cuda.is_available() else torch.device('cpu'))

    try:
        for rank, game_indices in enumerate(game_buckets):
            if not game_indices:
                continue
            proc = ctx.Process(
                target=_eval_worker,
                args=(rank, model1_state, model2_state, config, worker_device, game_indices, result_queue),
            )
            proc.daemon = True
            proc.start()
            processes.append(proc)

        wins = 0.0
        draws = 0.0
        unresolved = 0
        remaining = len(processes)

        eval_bar = tqdm(total=num_games, desc="⚔️ Eval vs best", unit="gra")
        try:
            while remaining > 0:
                try:
                    message = result_queue.get(timeout=0.2)
                except queue.Empty:
                    alive_remaining = 0
                    for proc in processes:
                        proc.join(timeout=0)
                        if proc.is_alive():
                            alive_remaining += 1
                        elif proc.exitcode not in (0, None):
                            raise RuntimeError(f"Eval worker exited unexpectedly with code {proc.exitcode}")
                    remaining = alive_remaining
                    continue

                msg_type = message.get('type')
                if msg_type == 'progress':
                    eval_bar.update(int(message.get('completed', 0) or 0))
                elif msg_type == 'result':
                    wins += float(message.get('wins', 0.0) or 0.0)
                    draws += float(message.get('draws', 0.0) or 0.0)
                    unresolved += int(message.get('unresolved', 0) or 0)
                    remaining -= 1
                elif msg_type == 'interrupt':
                    raise KeyboardInterrupt
                elif msg_type == 'error':
                    raise RuntimeError(f"Eval worker {message.get('rank')} failed: {message.get('error')}")
        finally:
            eval_bar.close()
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            result_queue.close()
        except Exception:
            pass

        if unresolved > 0:
            max_moves = _resolve_eval_max_moves(config)
            print(f"Eval unresolved at move cap ({max_moves}): {unresolved}/{num_games} -> counted as draws")
        return (wins + draws) / num_games
