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
    
    def __init__(self, expensive_policy_diagnostics=False):
        # Legal/support policy diagnostics are useful for one-off audits, but
        # mostly useless in the IL hot path: PGN target moves are already legal,
        # and computing legal alternatives can stall DataLoader/GPU throughput.
        self.expensive_policy_diagnostics = bool(expensive_policy_diagnostics)
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.policy_top1_correct = 0
        self.policy_top3_correct = 0
        self.policy_top5_correct = 0
        self.policy_target_mass_top1_sum = 0.0
        self.policy_target_mass_top3_sum = 0.0
        self.policy_target_mass_top5_sum = 0.0
        self.policy_target_mass_count = 0
        self.policy_entropy_sum = 0.0
        self.policy_top1_prob_sum = 0.0
        self.policy_entropy_count = 0
        self.policy_legal_entropy_sum = 0.0
        self.policy_legal_top1_prob_sum = 0.0
        self.policy_legal_top1_margin_sum = 0.0
        self.policy_legal_entropy_count = 0
        self.policy_target_entropy_sum = 0.0
        self.policy_target_top1_mass_sum = 0.0
        self.policy_target_entropy_count = 0
        self.policy_target_support_top1_prob_sum = 0.0
        self.policy_target_support_top1_margin_sum = 0.0
        self.policy_target_support_count = 0
        # ⚡ Running sums instead of lists (avoids .cpu().tolist() GPU sync per batch)
        self.value_abs_error_sum = 0.0
        self.value_abs_error_count = 0
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
        self.value_phase_pred_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_phase_pred_sq_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_phase_target_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_phase_target_sq_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_wdl_correct = 0
        self.value_wdl_total = 0
        self.value_wdl_ce_sum = 0.0
        self.value_phase_wdl_ce_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.value_phase_wdl_ce_weight = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.moves_left_abs_error_sum = 0.0
        self.moves_left_weight_sum = 0.0
        self.moves_left_loss_sum = 0.0
        self.moves_left_loss_weight_sum = 0.0
        self.moves_left_phase_abs_error_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.moves_left_phase_weight_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.moves_left_phase_loss_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.moves_left_phase_loss_weight_sum = {
            'opening': 0.0,
            'middlegame': 0.0,
            'endgame': 0.0,
        }
        self.confidence_sum = 0.0
        self.confidence_sq_sum = 0.0
        self.confidence_count = 0
        self.legal_coverages = []
        self.total_samples = 0
        self.soft_occurrence_sum = 0.0
        self.soft_occurrence_max = 0.0
        self.soft_sample_weight_sum = 0.0
        self.soft_policy_mass_sum = 0.0
        self.soft_policy_mass_min = 1.0
        self.soft_target_count = 0

    def update_soft_target_stats(self, occurrence_count=None, sample_weight=None, policy_mass_kept=None):
        if occurrence_count is None:
            return
        counts = occurrence_count.detach().float().reshape(-1)
        n = int(counts.numel())
        if n <= 0:
            return
        self.soft_occurrence_sum += counts.sum().item()
        self.soft_occurrence_max = max(self.soft_occurrence_max, counts.max().item())
        if sample_weight is not None:
            weights = sample_weight.detach().float().reshape(-1)
            self.soft_sample_weight_sum += weights.sum().item()
        else:
            self.soft_sample_weight_sum += float(n)
        if policy_mass_kept is not None:
            mass = policy_mass_kept.detach().float().reshape(-1)
            self.soft_policy_mass_sum += mass.sum().item()
            self.soft_policy_mass_min = min(self.soft_policy_mass_min, mass.min().item())
        else:
            self.soft_policy_mass_sum += float(n)
        self.soft_target_count += n

    def update_value_phase_loss(self, value_pred, target_value=None, target_wdl=None,
                                move_indices=None, sample_weight=None,
                                value_phase_opening_max=12, value_phase_endgame_min=40):
        if move_indices is None or value_pred is None:
            return
        if value_pred.dim() != 2 or value_pred.size(1) != 3:
            return
        if move_indices.dim() > 1:
            move_indices = move_indices.squeeze(-1)
        move_indices = move_indices.to(device=value_pred.device)

        log_probs = F.log_softmax(value_pred.float(), dim=1)
        if target_wdl is not None:
            target_dist = target_wdl.to(device=value_pred.device, dtype=torch.float32)
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            losses = -(target_dist.to(log_probs.dtype) * log_probs).sum(dim=1)
        elif target_value is not None:
            target_value = target_value.to(device=value_pred.device).reshape(-1)
            target_classes = torch.zeros_like(target_value, dtype=torch.long)
            target_classes[target_value > 0.9] = 0
            target_classes[target_value < -0.9] = 2
            target_classes[(target_value >= -0.9) & (target_value <= 0.9)] = 1
            losses = F.cross_entropy(value_pred.float(), target_classes, reduction='none')
        else:
            return

        if sample_weight is not None:
            weights = sample_weight.to(device=value_pred.device, dtype=losses.dtype).reshape(-1)
        else:
            weights = torch.ones_like(losses)

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
            if not bool(phase_mask.any().item()):
                continue
            phase_weights = weights[phase_mask]
            weight_sum = phase_weights.sum().item()
            if weight_sum <= 0.0:
                continue
            self.value_phase_wdl_ce_sum[phase_name] += (losses[phase_mask] * phase_weights).sum().item()
            self.value_phase_wdl_ce_weight[phase_name] += weight_sum

    def update_moves_left_metrics(self, pred_moves_left_log, target_moves_left_log=None,
                                  target_moves_left=None, move_indices=None,
                                  total_moves=None, sample_weight=None,
                                  value_phase_opening_max=12, value_phase_endgame_min=40):
        if pred_moves_left_log is None:
            return
        pred_log = pred_moves_left_log.reshape(-1).float()
        device = pred_log.device

        if target_moves_left_log is not None:
            target_log = target_moves_left_log.to(device=device, dtype=torch.float32).reshape(-1)
            target_plies = torch.expm1(target_log).clamp_min(0.0)
        elif target_moves_left is not None:
            target_plies = target_moves_left.to(device=device, dtype=torch.float32).reshape(-1).clamp_min(0.0)
            target_log = torch.log1p(target_plies)
        elif total_moves is not None and move_indices is not None:
            target_plies = (
                total_moves.to(device=device, dtype=torch.float32).reshape(-1)
                - move_indices.to(device=device, dtype=torch.float32).reshape(-1)
            ).clamp_min(0.0)
            target_log = torch.log1p(target_plies)
        else:
            return

        pred_plies = torch.expm1(pred_log).clamp_min(0.0)
        abs_error = torch.abs(pred_plies - target_plies)
        log_losses = F.smooth_l1_loss(pred_log, target_log, beta=0.25, reduction='none')
        if sample_weight is not None:
            weights = sample_weight.to(device=device, dtype=abs_error.dtype).reshape(-1)
        else:
            weights = torch.ones_like(abs_error)

        weight_sum = weights.sum().item()
        if weight_sum <= 0.0:
            return
        self.moves_left_abs_error_sum += (abs_error * weights).sum().item()
        self.moves_left_weight_sum += weight_sum
        self.moves_left_loss_sum += (log_losses * weights).sum().item()
        self.moves_left_loss_weight_sum += weight_sum

        if move_indices is None:
            return
        if move_indices.dim() > 1:
            move_indices = move_indices.squeeze(-1)
        phase_move_indices = move_indices.to(device=device, dtype=torch.float32)
        phase_masks = {
            'opening': phase_move_indices <= float(value_phase_opening_max),
            'middlegame': (
                (phase_move_indices > float(value_phase_opening_max))
                & (phase_move_indices < float(value_phase_endgame_min))
            ),
            'endgame': phase_move_indices >= float(value_phase_endgame_min),
        }
        for phase_name, phase_mask in phase_masks.items():
            if not bool(phase_mask.any().item()):
                continue
            phase_weights = weights[phase_mask]
            phase_weight_sum = phase_weights.sum().item()
            if phase_weight_sum <= 0.0:
                continue
            self.moves_left_phase_abs_error_sum[phase_name] += (
                abs_error[phase_mask] * phase_weights
            ).sum().item()
            self.moves_left_phase_weight_sum[phase_name] += phase_weight_sum
            self.moves_left_phase_loss_sum[phase_name] += (
                log_losses[phase_mask] * phase_weights
            ).sum().item()
            self.moves_left_phase_loss_weight_sum[phase_name] += phase_weight_sum
    
    def update(self, policy_pred, value_pred, target_move, target_value, legal_moves_mask=None,
               move_indices=None, total_moves=None, value_weight_min=0.1,
               value_weight_min_total_moves=40, value_max_moves=200,
               value_phase_opening_max=12, value_phase_endgame_min=40,
               value_use_game_length=False, policy_is_logits=False, legal_indices=None,
               target_policy_indices=None, target_policy_values=None,
               target_wdl=None):
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
        
        # Get top-k predictions. IL/RL policy loss is legal-only, so metrics must
        # also rank only legal moves; otherwise high illegal logits create a false
        # low Top-K signal even though gameplay gathers legal logits.
        if legal_indices is not None:
            if legal_indices.dim() == 1:
                legal_indices = legal_indices.unsqueeze(0)
            legal_indices = legal_indices.to(device=policy_pred.device, dtype=torch.long)
            valid_legal = (legal_indices >= 0) & (legal_indices < policy_pred.size(1))
            safe_legal = legal_indices.clamp(0, policy_pred.size(1) - 1)
            legal_scores = torch.gather(policy_pred, 1, safe_legal)
            legal_scores = legal_scores.masked_fill(~valid_legal, -1.0e9)
            topk = min(5, int(legal_scores.size(1)))
            _, legal_top_pos = torch.topk(legal_scores, k=topk, dim=1)
            top5_indices = torch.gather(legal_indices, 1, legal_top_pos)
            legal_probs = torch.softmax(legal_scores.float(), dim=1).masked_fill(~valid_legal, 0.0)
            legal_entropy = -(legal_probs * legal_probs.clamp_min(1.0e-12).log()).sum(dim=1)
            legal_top1_prob = legal_probs.max(dim=1).values
            legal_top2 = torch.topk(legal_probs, k=min(2, legal_probs.size(1)), dim=1).values
            legal_margin = (
                legal_top2[:, 0] - legal_top2[:, 1]
                if legal_top2.size(1) > 1 else legal_top2[:, 0]
            )
            self.policy_entropy_sum += legal_entropy.sum().item()
            self.policy_top1_prob_sum += legal_top1_prob.sum().item()
            self.policy_entropy_count += int(legal_entropy.numel())
            self.policy_legal_entropy_sum += legal_entropy.sum().item()
            self.policy_legal_top1_prob_sum += legal_top1_prob.sum().item()
            self.policy_legal_top1_margin_sum += legal_margin.sum().item()
            self.policy_legal_entropy_count += int(legal_entropy.numel())
            if topk < 5:
                pad = torch.full(
                    (top5_indices.size(0), 5 - topk),
                    -1,
                    device=top5_indices.device,
                    dtype=top5_indices.dtype,
                )
                top5_indices = torch.cat([top5_indices, pad], dim=1)
        else:
            topk = min(5, int(policy_pred.size(1)))
            _, top5_indices = torch.topk(policy_pred, k=topk, dim=1)
            if topk < 5:
                pad = torch.full(
                    (top5_indices.size(0), 5 - topk),
                    -1,
                    device=top5_indices.device,
                    dtype=top5_indices.dtype,
                )
                top5_indices = torch.cat([top5_indices, pad], dim=1)
            full_probs = torch.softmax(policy_pred.float(), dim=1)
            full_entropy = -(full_probs * full_probs.clamp_min(1.0e-12).log()).sum(dim=1)
            full_top1_prob = full_probs.max(dim=1).values
            self.policy_entropy_sum += full_entropy.sum().item()
            self.policy_top1_prob_sum += full_top1_prob.sum().item()
            self.policy_entropy_count += int(full_entropy.numel())
        
        # Check if target is in top-k
        target_expanded = target_move.unsqueeze(1).expand_as(top5_indices)
        
        top1_correct = (top5_indices[:, 0] == target_move).float().sum().item()
        top3_correct = (top5_indices[:, :3] == target_expanded[:, :3]).any(dim=1).float().sum().item()
        top5_correct = (top5_indices == target_expanded).any(dim=1).float().sum().item()
        
        self.policy_top1_correct += top1_correct
        self.policy_top3_correct += top3_correct
        self.policy_top5_correct += top5_correct

        if target_policy_indices is not None and target_policy_values is not None:
            target_policy_indices = target_policy_indices.to(device=policy_pred.device, dtype=torch.long)
            target_policy_values = target_policy_values.to(device=policy_pred.device, dtype=torch.float32)
            valid_targets = (
                (target_policy_indices >= 0)
                & (target_policy_indices < policy_pred.size(1))
                & (target_policy_values > 0.0)
            )
            target_mass = torch.where(
                valid_targets,
                target_policy_values,
                torch.zeros_like(target_policy_values, dtype=torch.float32),
            )
            target_mass = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            target_entropy = -(target_mass * target_mass.clamp_min(1.0e-12).log()).sum(dim=1)
            target_top1_mass = target_mass.max(dim=1).values
            self.policy_target_entropy_sum += target_entropy.sum().item()
            self.policy_target_top1_mass_sum += target_top1_mass.sum().item()
            self.policy_target_entropy_count += int(target_entropy.numel())

            if self.expensive_policy_diagnostics:
                safe_target_indices = target_policy_indices.clamp(0, policy_pred.size(1) - 1)
                support_scores = torch.gather(policy_pred.float(), 1, safe_target_indices)
                support_scores = support_scores.masked_fill(~valid_targets, -1.0e9)
                support_has_two = valid_targets.sum(dim=1) > 1
                support_probs = torch.softmax(support_scores, dim=1).masked_fill(~valid_targets, 0.0)
                support_top2 = torch.topk(support_probs, k=min(2, support_probs.size(1)), dim=1).values
                support_top1_prob = support_top2[:, 0]
                support_margin = torch.where(
                    support_has_two,
                    support_top2[:, 0] - support_top2[:, 1],
                    support_top2[:, 0],
                )
                valid_support_rows = valid_targets.any(dim=1)
                if bool(valid_support_rows.any().item()):
                    self.policy_target_support_top1_prob_sum += support_top1_prob[valid_support_rows].sum().item()
                    self.policy_target_support_top1_margin_sum += support_margin[valid_support_rows].sum().item()
                    self.policy_target_support_count += int(valid_support_rows.sum().item())

            def _mass_in_top(k):
                matches = target_policy_indices.unsqueeze(1) == top5_indices[:, :k].unsqueeze(2)
                covered = (matches & valid_targets.unsqueeze(1)).any(dim=1)
                return torch.where(covered, target_mass, torch.zeros_like(target_mass)).sum(dim=1)

            mass_top1 = _mass_in_top(1)
            mass_top3 = _mass_in_top(3)
            mass_top5 = _mass_in_top(5)
            self.policy_target_mass_top1_sum += mass_top1.sum().item()
            self.policy_target_mass_top3_sum += mass_top3.sum().item()
            self.policy_target_mass_top5_sum += mass_top5.sum().item()
            self.policy_target_mass_count += int(mass_top1.numel())
        
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

            # WDL accuracy + CE. For soft WDL targets, compare against the
            # target distribution directly; thresholding W-L makes almost every
            # softened target look like a draw and destroys the metric.
            if target_wdl is not None:
                target_dist = target_wdl.to(device=value_pred.device, dtype=torch.float32)
                target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
                target_classes = torch.argmax(target_dist, dim=1)
                log_probs = F.log_softmax(value_pred.float(), dim=1)
                wdl_ce = -(target_dist.to(log_probs.dtype) * log_probs).sum(dim=1)
            else:
                target_classes = torch.zeros_like(target_value, dtype=torch.long)
                target_classes[target_value > 0.9] = 0
                target_classes[target_value < -0.9] = 2
                target_classes[(target_value >= -0.9) & (target_value <= 0.9)] = 1
                wdl_ce = F.cross_entropy(value_pred.float(), target_classes, reduction='none')
            pred_classes = torch.argmax(value_pred, dim=1)
            self.value_wdl_correct += (pred_classes == target_classes).sum().item()
            self.value_wdl_total += target_classes.numel()
            self.value_wdl_ce_sum += wdl_ce.sum().item()
        else:
            # Legacy scalar format
            value_scalar = value_pred.squeeze()
        
        value_mae = torch.abs(value_scalar - target_value)
        # ⚡ Running sum instead of list (avoids GPU→CPU sync per batch)
        self.value_abs_error_sum += value_mae.sum().item()
        self.value_abs_error_count += value_mae.numel()
        
        # Phase diagnostics (optional). These do not weight the value loss.
        if move_indices is not None:
            if move_indices.dim() > 1:
                move_indices = move_indices.squeeze(-1)
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
                phase_pred = value_scalar[phase_mask].float()
                phase_target = target_value[phase_mask].float()
                self.value_phase_pred_sum[phase_name] += phase_pred.sum().item()
                self.value_phase_pred_sq_sum[phase_name] += (phase_pred * phase_pred).sum().item()
                self.value_phase_target_sum[phase_name] += phase_target.sum().item()
                self.value_phase_target_sq_sum[phase_name] += (phase_target * phase_target).sum().item()
        
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
            'value_wdl_acc': (self.value_wdl_correct / self.value_wdl_total) if self.value_wdl_total else 0.0,
            'value_wdl_ce': (self.value_wdl_ce_sum / self.value_wdl_total) if self.value_wdl_total else 0.0,
            'moves_left_mae': (
                self.moves_left_abs_error_sum / self.moves_left_weight_sum
                if self.moves_left_weight_sum > 0.0 else 0.0
            ),
            'moves_left_loss': (
                self.moves_left_loss_sum / self.moves_left_loss_weight_sum
                if self.moves_left_loss_weight_sum > 0.0 else 0.0
            ),
            
            # Confidence metrics (Welford: std = sqrt(E[x²] - E[x]²))
            'avg_confidence': (self.confidence_sum / self.confidence_count) if self.confidence_count else 0.0,
            'confidence_std': math.sqrt(max(0.0, self.confidence_sq_sum / self.confidence_count - (self.confidence_sum / self.confidence_count) ** 2)) if self.confidence_count else 0.0,
        }
        if self.policy_target_mass_count:
            metrics.update({
                'policy_target_mass_top1': self.policy_target_mass_top1_sum / self.policy_target_mass_count,
                'policy_target_mass_top3': self.policy_target_mass_top3_sum / self.policy_target_mass_count,
                'policy_target_mass_top5': self.policy_target_mass_top5_sum / self.policy_target_mass_count,
            })
        if self.policy_entropy_count:
            policy_entropy = self.policy_entropy_sum / self.policy_entropy_count
            metrics.update({
                'policy_entropy': policy_entropy,
                'policy_effective_moves': math.exp(policy_entropy),
                'policy_top1_prob': self.policy_top1_prob_sum / self.policy_entropy_count,
            })
        if self.policy_legal_entropy_count:
            legal_entropy = self.policy_legal_entropy_sum / self.policy_legal_entropy_count
            metrics.update({
                'policy_legal_entropy': legal_entropy,
                'policy_legal_effective_moves': math.exp(legal_entropy),
                'policy_legal_top1_prob': self.policy_legal_top1_prob_sum / self.policy_legal_entropy_count,
                'policy_legal_top1_margin': self.policy_legal_top1_margin_sum / self.policy_legal_entropy_count,
            })
        if self.policy_target_entropy_count:
            target_entropy = self.policy_target_entropy_sum / self.policy_target_entropy_count
            metrics.update({
                'policy_target_entropy': target_entropy,
                'policy_target_effective_moves': math.exp(target_entropy),
                'policy_target_top1_mass': self.policy_target_top1_mass_sum / self.policy_target_entropy_count,
            })
        if self.policy_target_support_count:
            metrics.update({
                'policy_target_support_top1_prob': (
                    self.policy_target_support_top1_prob_sum / self.policy_target_support_count
                ),
                'policy_target_support_top1_margin': (
                    self.policy_target_support_top1_margin_sum / self.policy_target_support_count
                ),
            })
        if self.soft_target_count:
            metrics.update({
                'soft_occurrence_avg': self.soft_occurrence_sum / self.soft_target_count,
                'soft_occurrence_max': self.soft_occurrence_max,
                'soft_sample_weight_avg': self.soft_sample_weight_sum / self.soft_target_count,
                'soft_policy_mass_kept_avg': self.soft_policy_mass_sum / self.soft_target_count,
                'soft_policy_mass_kept_min': self.soft_policy_mass_min,
            })
        for phase_name in ('opening', 'middlegame', 'endgame'):
            count = self.value_phase_abs_error_count.get(phase_name, 0)
            metrics[f'value_mae_{phase_name}'] = (
                self.value_phase_abs_error_sum.get(phase_name, 0.0) / count
                if count else 0.0
            )
            ce_weight = self.value_phase_wdl_ce_weight.get(phase_name, 0.0)
            metrics[f'value_wdl_ce_{phase_name}'] = (
                self.value_phase_wdl_ce_sum.get(phase_name, 0.0) / ce_weight
                if ce_weight > 0.0 else 0.0
            )
            mlh_weight = self.moves_left_phase_weight_sum.get(phase_name, 0.0)
            metrics[f'moves_left_mae_{phase_name}'] = (
                self.moves_left_phase_abs_error_sum.get(phase_name, 0.0) / mlh_weight
                if mlh_weight > 0.0 else 0.0
            )
            mlh_loss_weight = self.moves_left_phase_loss_weight_sum.get(phase_name, 0.0)
            metrics[f'moves_left_loss_{phase_name}'] = (
                self.moves_left_phase_loss_sum.get(phase_name, 0.0) / mlh_loss_weight
                if mlh_loss_weight > 0.0 else 0.0
            )
            metrics[f'value_samples_{phase_name}'] = count
            if count:
                pred_mean = self.value_phase_pred_sum.get(phase_name, 0.0) / count
                target_mean = self.value_phase_target_sum.get(phase_name, 0.0) / count
                pred_var = max(
                    0.0,
                    self.value_phase_pred_sq_sum.get(phase_name, 0.0) / count - pred_mean * pred_mean,
                )
                target_var = max(
                    0.0,
                    self.value_phase_target_sq_sum.get(phase_name, 0.0) / count - target_mean * target_mean,
                )
                pred_std = math.sqrt(pred_var)
                target_std = math.sqrt(target_var)
                metrics[f'value_pred_std_{phase_name}'] = pred_std
                metrics[f'value_target_std_{phase_name}'] = target_std
                metrics[f'value_std_ratio_{phase_name}'] = pred_std / target_std if target_std > 1e-8 else 0.0
            else:
                metrics[f'value_pred_std_{phase_name}'] = 0.0
                metrics[f'value_target_std_{phase_name}'] = 0.0
                metrics[f'value_std_ratio_{phase_name}'] = 0.0
        
        # Legal move coverage (if available)
        if self.legal_coverages:
            metrics['legal_coverage'] = np.mean(self.legal_coverages)
        
        return metrics


def compute_batch_metrics(policy_pred, value_pred, target_move, target_value, target_wdl=None):
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
        if target_wdl is not None:
            target_dist = target_wdl.to(device=value_pred.device, dtype=torch.float32)
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            target_classes = torch.argmax(target_dist, dim=1)
            log_probs = F.log_softmax(value_pred.float(), dim=1)
            wdl_ce = -(target_dist.to(log_probs.dtype) * log_probs).sum(dim=1).mean().item()
        else:
            target_classes = torch.zeros_like(target_value, dtype=torch.long)
            target_classes[target_value > 0.9] = 0
            target_classes[target_value < -0.9] = 2
            target_classes[(target_value >= -0.9) & (target_value <= 0.9)] = 1
            wdl_ce = F.cross_entropy(value_pred, target_classes).item()
        pred_classes = torch.argmax(value_pred, dim=1)
        wdl_acc = (pred_classes == target_classes).float().mean().item()
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
