"""
Training metrics for chess AI evaluation
"""

import torch
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
        self.value_abs_errors = []
        self.confidences = []
        self.legal_coverages = []
        self.total_samples = 0
    
    def update(self, policy_pred, value_pred, target_move, target_value, legal_moves_mask=None):
        """
        Update metrics with batch predictions
        
        Args:
            policy_pred: Policy logits (B, 4096) - log probabilities
            value_pred: Value predictions (B, 1)
            target_move: Target move indices (B,)
            target_value: Target values (B, 1)
            legal_moves_mask: Optional binary mask of legal moves (B, 4096)
        """
        batch_size = policy_pred.size(0)
        self.total_samples += batch_size
        
        # Convert log probs to probs
        policy_probs = torch.exp(policy_pred)
        
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
        # ============================================================
        
        value_mae = torch.abs(value_pred.squeeze() - target_value.squeeze())
        self.value_abs_errors.extend(value_mae.cpu().tolist())
        
        # ============================================================
        # PREDICTION CONFIDENCE
        # ============================================================
        
        # Max probability (confidence in best move)
        max_probs = policy_probs.max(dim=1)[0]
        self.confidences.extend(max_probs.cpu().tolist())
        
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
            'value_mae': np.mean(self.value_abs_errors) if self.value_abs_errors else 0.0,
            
            # Confidence metrics
            'avg_confidence': np.mean(self.confidences) if self.confidences else 0.0,
            'confidence_std': np.std(self.confidences) if self.confidences else 0.0,
        }
        
        # Legal move coverage (if available)
        if self.legal_coverages:
            metrics['legal_coverage'] = np.mean(self.legal_coverages)
        
        return metrics


def compute_batch_metrics(policy_pred, value_pred, target_move, target_value):
    """
    Compute metrics for a single batch (lightweight version)
    
    Args:
        policy_pred: Policy logits (B, 4096)
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
    
    # Value MAE
    value_mae = torch.abs(value_pred.squeeze() - target_value.squeeze()).mean().item()
    
    # Confidence
    policy_probs = torch.exp(policy_pred)
    avg_confidence = policy_probs.max(dim=1)[0].mean().item()
    
    return {
        'policy_top1_acc': top1_acc,
        'policy_top3_acc': top3_acc,
        'value_mae': value_mae,
        'avg_confidence': avg_confidence
    }