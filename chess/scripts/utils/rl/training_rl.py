"""
Reinforcement learning training helpers
"""

import sys
from pathlib import Path

import torch
import chess
import torch.nn.functional as F

from .replay import PrioritizedReplayBuffer

# Add project root to path for src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.mcts import BatchMCTS, select_move_by_visits
from src.utils.data_helpers import (
    ACTION_SIZE,
    build_hflip_inverse_index_map,
)

_HFLIP_INV_INDEX_MAP = None


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


def _maybe_augment_batch(boards, policy_targets, config):
    """
    Apply simple symmetry augmentation (horizontal flip).
    """
    rl_cfg = config.get('reinforcement_learning', {})
    if not rl_cfg.get('use_augmentation', False):
        return boards, policy_targets
    if not rl_cfg.get('augment_horizontal_flip', False):
        return boards, policy_targets

    prob = rl_cfg.get('augment_prob', 0.5)
    if prob <= 0:
        return boards, policy_targets

    batch_size = boards.size(0)
    if batch_size == 0:
        return boards, policy_targets

    flip_mask = torch.rand(batch_size) < prob
    if not flip_mask.any():
        return boards, policy_targets

    # Flip board tensors across files (a<->h)
    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])

    # Remap policy targets to match flipped board
    inv_map = _get_hflip_inverse_index_map()
    policy_targets[flip_mask] = policy_targets[flip_mask][:, inv_map]

    return boards, policy_targets

def train_on_batch_rl(model, optimizer, batch, indices, weights, config, device, scaler, replay_buffer, metrics_calc=None):
    """Train on batch with optional prioritized replay and metrics"""
    boards, policy_targets, value_targets = batch
    if config['reinforcement_learning'].get('replay_fp16', False):
        boards = boards.float()
        policy_targets = policy_targets.float()
        value_targets = value_targets.float()
    boards, policy_targets = _maybe_augment_batch(boards, policy_targets, config)
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_targets = policy_targets.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    use_amp = config['hardware'].get('use_amp', True)
    amp_dtype = torch.bfloat16 if config['hardware'].get('use_bfloat16', False) else torch.float16
    
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        policy_pred, value_pred = model(boards, return_aux=False)
        
        policy_loss = -(policy_targets * policy_pred).sum(dim=1)
        
        # Value loss: WDL CE when using WDL head, otherwise MSE
        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            # Targets are scalar {-1, 0, 1} -> map to WDL classes [W, D, L]
            target_scalar = value_targets.squeeze()
            target_classes = torch.zeros_like(target_scalar, dtype=torch.long)
            target_classes[target_scalar > 0.9] = 0
            target_classes[target_scalar < -0.9] = 2
            target_classes[(target_scalar >= -0.9) & (target_scalar <= 0.9)] = 1
            
            value_loss = F.cross_entropy(value_pred, target_classes, reduction='none')
        else:
            value_loss = (value_pred.squeeze() - value_targets.squeeze()) ** 2
        
        if weights is not None:
            policy_loss = (policy_loss * weights).mean()
            value_loss = (value_loss * weights).mean()
        else:
            policy_loss = policy_loss.mean()
            value_loss = value_loss.mean()
        
        policy_weight = config['reinforcement_learning']['policy_loss_weight']
        value_weight = config['reinforcement_learning']['value_loss_weight']
        loss = policy_weight * policy_loss + value_weight * value_loss

        # Entropy regularization (optional)
        entropy_weight = config['reinforcement_learning'].get('entropy_weight', 0.0)
        if entropy_weight > 0:
            policy_probs = torch.exp(policy_pred)
            policy_entropy = -(policy_probs * policy_pred).sum(dim=1).mean()
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
            target_moves = policy_targets.argmax(dim=1)
            metrics_calc.update(policy_pred, value_pred, target_moves, value_targets.unsqueeze(1))
    
    return loss.item(), policy_loss.item(), value_loss.item()


def evaluate_models(model1, model2, config, device, num_games=100):
    """
    Evaluate model1 vs model2
    
    🔧 v4.4: Increased from 20→100 games for statistical significance
    """
    mcts1 = BatchMCTS(model1, config, device)
    mcts2 = BatchMCTS(model2, config, device)
    
    wins = 0
    draws = 0
    
    for game_idx in range(num_games):
        board = chess.Board()
        
        if game_idx % 2 == 0:
            current_mcts = mcts1
            other_mcts = mcts2
        else:
            current_mcts = mcts2
            other_mcts = mcts1
        
        move_count = 0
        while not board.is_game_over() and move_count < 200:
            mcts = current_mcts if board.turn == chess.WHITE else other_mcts
            visit_counts = mcts.search(board, num_simulations=50)
            move, _ = select_move_by_visits(visit_counts, temperature=0)
            
            # Update history BEFORE making the move (for POV history inputs)
            mcts1.update_history(board)
            mcts2.update_history(board)
            
            board.push(move)
            move_count += 1
        
        mcts1.reset_tree()
        mcts2.reset_tree()
        
        result = board.result()
        if game_idx % 2 == 0:
            if result == '1-0':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
        else:
            if result == '0-1':
                wins += 1
            elif result == '1/2-1/2':
                draws += 0.5
    
    return (wins + draws) / num_games
