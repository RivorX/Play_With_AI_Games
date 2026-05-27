"""
Training metrics for chess AI evaluation
"""

import math

import torch
import torch.nn.functional as F
import numpy as np


class MetricsCalculator:
    """
    Calculate comprehensive training metrics
    
    Metrics:
    - Policy Accuracy (Top-1, Top-3, Top-5)
    - Value MAE (Mean Absolute Error)
    - Prediction Confidence
    - Legal Move Coverage
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.policy_top1_correct = 0
        self.policy_top3_correct = 0
        self.policy_top5_correct = 0
        # ⚡ Running sums instead of lists (avoids .cpu().tolist() GPU sync per batch)
        self.value_abs_error_sum = 0.0
        self.value_abs_error_count = 0
        self.value_abs_error_weighted_sum = 0.0
        self.value_abs_error_weighted_denom = 0.0
        self.value_phase_abs_error_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_phase_abs_error_count = {
            'opening': 0,
            'middlegame': 0,
            'endgame': 0,
        }
        self.value_wdl_correct = 0
        self.value_wdl_total = 0
        self.value_wdl_ce_sum = 0.0
        self.confidence_sum = 0.0
        self.confidence_sq_sum = 0.0
        self.confidence_count = 0
        self.legal_coverages = []
        self.total_samples = 0
    
    def update(self, policy_pred, value_pred, target_move, target_value, legal_moves_mask=None,
               move_indices=None, total_moves=None, value_weight_min=0.1,
               value_weight_min_total_moves=40, value_max_moves=200,
               value_phase_opening_max=12, value_phase_endgame_min=40,
               value_use_game_length=False, policy_is_logits=False):
        """
        Update metrics with batch predictions
        
        Args:
            policy_pred: Policy logits or log-probabilities (B, action_size)
            value_pred: Value predictions (B, 3) - WDL logits OR (B, 1) - scalar
            target_move: Target move indices (B,)
            target_value: Target values (B, 1) or (B,) - scalar values
            legal_moves_mask: Optional binary mask of legal moves (B, action_size)
        """
        batch_size = policy_pred.size(0)
        self.total_samples += batch_size
        
        # Convert policy scores to probabilities only for confidence/coverage.
        # Top-k is invariant to logits vs log-probabilities.
        policy_probs = torch.softmax(policy_pred, dim=1) if policy_is_logits else torch.exp(policy_pred)
        
        # ============================================================
        # POLICY ACCURACY (Top-1, Top-3, Top-5)
        # ============================================================
        
        # Get top-k predictions
        _, top5_indices = torch.topk(policy_pred, k=5, dim=1)
        
        # Check if target is in top-k
        target_expanded = target_move.unsqueeze(1).expand_as(top5_indices)
        
        top1_correct = (top5_indices[:, 0] == target_move).float().sum().item()
        top3_correct = (top5_indices[:, :3] == target_expanded[:, :3]).any(dim=1).float().sum().item()
        top5_correct = (top5_indices == target_expanded).any(dim=1).float().sum().item()
        
        self.policy_top1_correct += top1_correct
        self.policy_top3_correct += top3_correct
        self.policy_top5_correct += top5_correct
        
        # ============================================================
        # VALUE MAE (Mean Absolute Error)
        # 🆕 Handle WDL predictions by converting to scalar
        # ============================================================
        
        # Ensure target_value is 1D without collapsing batch size 1 to a scalar.
        target_value = target_value.reshape(-1)

        # Check if value_pred is WDL (3 logits) or scalar (1 value)
        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            # WDL format: convert to scalar
            # value_pred: (B, 3) logits
            wdl_probs = torch.softmax(value_pred, dim=1)
            # Scalar: W*1.0 + D*0.0 + L*(-1.0)
            value_scalar = (wdl_probs[:, 0] * 1.0 + 
                           wdl_probs[:, 1] * 0.0 + 
                           wdl_probs[:, 2] * (-1.0))

            # WDL accuracy + CE (classification metrics)
            target_classes = torch.zeros_like(target_value, dtype=torch.long)
            target_classes[target_value > 0.9] = 0
            target_classes[target_value < -0.9] = 2
            target_classes[(target_value >= -0.9) & (target_value <= 0.9)] = 1
            pred_classes = torch.argmax(value_pred, dim=1)
            self.value_wdl_correct += (pred_classes == target_classes).sum().item()
            self.value_wdl_total += target_value.numel()
            self.value_wdl_ce_sum += F.cross_entropy(value_pred, target_classes, reduction='sum').item()
        else:
            # Legacy scalar format
            value_scalar = value_pred.squeeze()
        
        value_mae = torch.abs(value_scalar - target_value)
        # ⚡ Running sum instead of list (avoids GPU→CPU sync per batch)
        self.value_abs_error_sum += value_mae.sum().item()
        self.value_abs_error_count += value_mae.numel()
        
        # Weighted MAE (optional)
        if move_indices is not None:
            if move_indices.dim() > 1:
                move_indices = move_indices.squeeze(-1)
            if value_use_game_length and total_moves is not None:
                if total_moves.dim() > 1:
                    total_moves = total_moves.squeeze(-1)
                effective_total = torch.clamp(
                    total_moves.float(),
                    min=value_weight_min_total_moves,
                    max=value_max_moves
                )
                denom = effective_total
            else:
                denom = value_max_moves
            weights = torch.clamp(
                move_indices.float() / denom,
                min=value_weight_min,
                max=1.0
            )
            self.value_abs_error_weighted_sum += (value_mae * weights).sum().item()
            self.value_abs_error_weighted_denom += weights.sum().item()

            phase_move_indices = move_indices.float()
            phase_masks = {
                'opening': phase_move_indices <= float(value_phase_opening_max),
                'middlegame': (
                    (phase_move_indices > float(value_phase_opening_max))
                    & (phase_move_indices < float(value_phase_endgame_min))
                ),
                'endgame': phase_move_indices >= float(value_phase_endgame_min),
            }
            for phase_name, phase_mask in phase_masks.items():
                count = int(phase_mask.sum().item())
                if count <= 0:
                    continue
                phase_errors = value_mae[phase_mask]
                self.value_phase_abs_error_sum[phase_name] += phase_errors.sum().item()
                self.value_phase_abs_error_count[phase_name] += count
        
        # ============================================================
        # PREDICTION CONFIDENCE
        # ============================================================
        
        # Max probability (confidence in best move)
        max_probs = policy_probs.max(dim=1)[0]
        # ⚡ Running sums for mean + std (avoids GPU→CPU sync per batch)
        self.confidence_sum += max_probs.sum().item()
        self.confidence_sq_sum += (max_probs * max_probs).sum().item()
        self.confidence_count += max_probs.numel()
        
        # ============================================================
        # LEGAL MOVE COVERAGE (if legal moves provided)
        # ============================================================
        
        if legal_moves_mask is not None:
            # Sum of probability mass on legal moves
            legal_prob_sum = (policy_probs * legal_moves_mask).sum(dim=1)
            self.legal_coverages.extend(legal_prob_sum.cpu().tolist())
    
    def compute(self):
        """
        Compute final metrics
        
        Returns:
            Dict with all metrics
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {
            # Policy accuracy
            'policy_top1_acc': self.policy_top1_correct / self.total_samples,
            'policy_top3_acc': self.policy_top3_correct / self.total_samples,
            'policy_top5_acc': self.policy_top5_correct / self.total_samples,
            
            # Value metrics
            'value_mae': (self.value_abs_error_sum / self.value_abs_error_count) if self.value_abs_error_count else 0.0,
            'value_mae_weighted': (self.value_abs_error_weighted_sum / self.value_abs_error_weighted_denom)
            if self.value_abs_error_weighted_denom else 0.0,
            'value_wdl_acc': (self.value_wdl_correct / self.value_wdl_total) if self.value_wdl_total else 0.0,
            'value_wdl_ce': (self.value_wdl_ce_sum / self.value_wdl_total) if self.value_wdl_total else 0.0,
            
            # Confidence metrics (Welford: std = sqrt(E[x²] - E[x]²))
            'avg_confidence': (self.confidence_sum / self.confidence_count) if self.confidence_count else 0.0,
            'confidence_std': math.sqrt(max(0.0, self.confidence_sq_sum / self.confidence_count - (self.confidence_sum / self.confidence_count) ** 2)) if self.confidence_count else 0.0,
        }
        for phase_name in ('opening', 'middlegame', 'endgame'):
            count = self.value_phase_abs_error_count.get(phase_name, 0)
            metrics[f'value_mae_{phase_name}'] = (
                self.value_phase_abs_error_sum.get(phase_name, 0.0) / count
                if count else 0.0
            )
            metrics[f'value_samples_{phase_name}'] = count
        
        # Legal move coverage (if available)
        if self.legal_coverages:
            metrics['legal_coverage'] = np.mean(self.legal_coverages)
        
        return metrics


def compute_batch_metrics(policy_pred, value_pred, target_move, target_value):
    """
    Compute metrics for a single batch (lightweight version)
    
    Args:
        policy_pred: Policy logits (B, action_size)
        value_pred: Value predictions (B, 1)
        target_move: Target move indices (B,)
        target_value: Target values (B, 1)
    
    Returns:
        Dict with batch metrics
    """
    batch_size = policy_pred.size(0)
    
    # Top-1 accuracy
    _, top1_pred = policy_pred.max(dim=1)
    top1_acc = (top1_pred == target_move).float().mean().item()
    
    # Top-3 accuracy
    _, top3_indices = torch.topk(policy_pred, k=3, dim=1)
    target_expanded = target_move.unsqueeze(1).expand_as(top3_indices)
    top3_acc = (top3_indices == target_expanded).any(dim=1).float().mean().item()
    
    # Value MAE - handle WDL
    if value_pred.dim() == 2 and value_pred.size(1) == 3:
        # WDL format
        wdl_probs = torch.softmax(value_pred, dim=1)
        value_scalar = (wdl_probs[:, 0] * 1.0 + 
                       wdl_probs[:, 1] * 0.0 + 
                       wdl_probs[:, 2] * (-1.0))
        
        # WDL accuracy + CE
        target_value = target_value.reshape(-1)
        target_classes = torch.zeros_like(target_value, dtype=torch.long)
        target_classes[target_value > 0.9] = 0
        target_classes[target_value < -0.9] = 2
        target_classes[(target_value >= -0.9) & (target_value <= 0.9)] = 1
        pred_classes = torch.argmax(value_pred, dim=1)
        wdl_acc = (pred_classes == target_classes).float().mean().item()
        wdl_ce = F.cross_entropy(value_pred, target_classes).item()
    else:
        value_scalar = value_pred.squeeze()
        wdl_acc = 0.0
        wdl_ce = 0.0
    
    target_value = target_value.reshape(-1)
    
    value_mae = torch.abs(value_scalar - target_value).mean().item()
    
    # Confidence
    policy_probs = torch.exp(policy_pred)
    avg_confidence = policy_probs.max(dim=1)[0].mean().item()
    
    return {
        'policy_top1_acc': top1_acc,
        'policy_top3_acc': top3_acc,
        'value_mae': value_mae,
        'value_wdl_acc': wdl_acc,
        'value_wdl_ce': wdl_ce,
        'avg_confidence': avg_confidence
    }
