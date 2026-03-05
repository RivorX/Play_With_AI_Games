"""
Custom loss functions
🆕 v4.3: WDL (Win/Draw/Loss) classification + Move-weighted auxiliary losses
🐛 FIXED: Robust shape handling for (B,) and (B, 1) targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingNLLLoss(nn.Module):
    """
    NLL Loss with label smoothing for better generalization
    
    Prevents overconfident predictions by distributing some probability
    mass to non-target classes.
    """
    
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Amount of smoothing (0.0 = no smoothing, 0.1 = standard)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, log_probs, targets):
        """
        Args:
            log_probs: Log probabilities from model (batch_size, num_classes)
            targets: Target class indices (batch_size,)
        
        Returns:
            Smoothed NLL loss
        """
        num_classes = log_probs.size(-1)
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class WDLLoss(nn.Module):
    """
    🆕 Win/Draw/Loss classification loss
    
    MUCH stronger signal than MSE regression on [-1, 0, +1]!
    
    Architecture:
    - Model outputs 3 logits (Win/Draw/Loss)
    - Loss is cross-entropy over these 3 classes
    - At inference, convert back to scalar: W*1.0 + D*0.0 + L*(-1.0)
    
    Benefits:
    - Categorical nature better matches chess outcomes
    - Clearer gradient signal (class boundaries vs continuous regression)
    - Better handles the discrete nature of game results
    - +5-10% stronger value predictions empirically
    """
    
    def __init__(self, label_smoothing=0.0, debug=False):
        """
        Args:
            label_smoothing: Optional smoothing for WDL targets (0.0-0.1)
            debug: If True, print one-time diagnostics about target distribution
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.debug = bool(debug)
    
    def targets_to_classes(self, target_values):
        """
        Convert scalar targets to class indices, with shape handling and diagnostics.
        """
        # Ensure 1D
        if target_values.dim() > 1:
            target_values = target_values.squeeze(-1)

        target_classes = torch.zeros_like(target_values, dtype=torch.long)
        target_classes[target_values > 0.9] = 0   # Win
        target_classes[target_values < -0.9] = 2  # Loss
        target_classes[(target_values >= -0.9) & (target_values <= 0.9)] = 1  # Draw

        # Diagnostic print (once)
        if self.debug and not hasattr(self, '_diagnostic_printed'):
            print(f"\nWDL Loss DIAGNOSTIC:")
            print(f"  Input target_values shape: {target_values.shape}")
            print(f"  Target values range: [{target_values.min().item():.3f}, {target_values.max().item():.3f}]")
            print(f"  Class distribution:")
            print(f"    Win (0):  {(target_classes == 0).sum().item()} ({(target_classes == 0).float().mean().item()*100:.1f}%)")
            print(f"    Draw (1): {(target_classes == 1).sum().item()} ({(target_classes == 1).float().mean().item()*100:.1f}%)")
            print(f"    Loss (2): {(target_classes == 2).sum().item()} ({(target_classes == 2).float().mean().item()*100:.1f}%)")
            print(f"  Sample targets: {target_values[:10].cpu().numpy()}")
            print(f"  Sample classes: {target_classes[:10].cpu().numpy()}\n")
            self._diagnostic_printed = True

        return target_classes

    def forward(self, wdl_logits, target_values):
        """
        Args:
            wdl_logits: Model output (B, 3) - [Win, Draw, Loss] logits
            target_values: Target scalar values - can be either:
                - (B,) 1D tensor in {-1.0, 0.0, +1.0}
                - (B, 1) 2D tensor in {-1.0, 0.0, +1.0}
        
        Returns:
            Cross-entropy loss
        """
        target_classes = self.targets_to_classes(target_values)

        # Compute cross-entropy loss
        if self.label_smoothing > 0:
            return F.cross_entropy(
                wdl_logits, 
                target_classes, 
                label_smoothing=self.label_smoothing
            )
        else:
            return F.cross_entropy(wdl_logits, target_classes)
    
    @staticmethod
    def wdl_to_scalar(wdl_probs):
        """
        Convert WDL probabilities to scalar value
        
        Args:
            wdl_probs: (B, 3) probabilities [Win, Draw, Loss]
        
        Returns:
            (B,) scalar values: W*1.0 + D*0.0 + L*(-1.0)
        """
        # wdl_probs: (B, 3) -> [p_win, p_draw, p_loss]
        return wdl_probs[:, 0] * 1.0 + wdl_probs[:, 1] * 0.0 + wdl_probs[:, 2] * (-1.0)


class CombinedLoss(nn.Module):
    """
    🆕 Combined loss for IL training with all improvements
    
    Components:
    1. Policy: Label-smoothed NLL
    2. Value: WDL classification (Win/Draw/Loss)
    """
    
    def __init__(self, config):
        """
        Args:
            config: Training configuration dict
        """
        super().__init__()
        
        # Extract weights
        self.policy_weight = config['imitation_learning']['policy_loss_weight']
        self.value_weight = config['imitation_learning']['value_loss_weight']
        
        # Loss functions
        label_smoothing = config['imitation_learning'].get('label_smoothing', 0.1)
        self.policy_loss_fn = LabelSmoothingNLLLoss(smoothing=label_smoothing)
        
        # 🆕 WDL loss for value head
        wdl_smoothing = config['imitation_learning'].get('wdl_label_smoothing', 0.0)
        debug_enabled = config.get('debug', {}).get('enabled', False)
        self.value_loss_fn = WDLLoss(label_smoothing=wdl_smoothing, debug=debug_enabled)

        # ?? Value loss weighting by move index (later positions = stronger signal)
        self.value_move_weighting = config['imitation_learning'].get('value_move_weighting', True)
        self.value_move_weight_min = config['imitation_learning'].get('value_move_weight_min', 0.1)
        self.value_move_weight_use_game_length = config['imitation_learning'].get('value_move_weight_use_game_length', False)
        self.value_move_weight_min_total_moves = config['imitation_learning'].get('value_move_weight_min_total_moves', 40)
        self.value_max_moves = config['data'].get('max_moves_per_game', 200)
    
    def forward(self, predictions, targets):
        """
        Compute combined loss
        
        Args:
            predictions: dict with model outputs:
                - 'policy': (B, ACTION_SIZE) log probabilities
                - 'value': (B, 3) WDL logits
            
            targets: dict with ground truth:
                - 'moves': (B,) move indices
                - 'values': (B,) or (B, 1) scalar values in {-1, 0, +1}
                - 'move_indices': (B,) or (B, 1) move numbers (optional, for weighting)
                - 'total_moves': (B,) or (B, 1) total moves per game (optional, for value weighting)
        
        Returns:
            total_loss, loss_dict
        """

        # Policy loss
        policy_loss = self.policy_loss_fn(
            predictions['policy'], 
            targets['moves']
        )
        
        # ?? Value loss (WDL) - handles both (B,) and (B, 1) targets
        move_indices = targets.get('move_indices', None)
        total_moves = targets.get('total_moves', None)
        if self.value_move_weighting and move_indices is not None:
            if move_indices.dim() > 1:
                move_indices = move_indices.squeeze(-1)
            if self.value_move_weight_use_game_length and total_moves is not None:
                if total_moves.dim() > 1:
                    total_moves = total_moves.squeeze(-1)
                effective_total = torch.clamp(
                    total_moves.float(),
                    min=self.value_move_weight_min_total_moves,
                    max=self.value_max_moves
                )
                denom = effective_total
            else:
                denom = self.value_max_moves
            weights = torch.clamp(
                move_indices.float() / denom,
                min=self.value_move_weight_min,
                max=1.0
            )
            target_classes = self.value_loss_fn.targets_to_classes(targets['values'])
            if self.value_loss_fn.label_smoothing > 0:
                value_losses = F.cross_entropy(
                    predictions['value'],
                    target_classes,
                    reduction='none',
                    label_smoothing=self.value_loss_fn.label_smoothing
                )
            else:
                value_losses = F.cross_entropy(
                    predictions['value'],
                    target_classes,
                    reduction='none'
                )
            value_loss = (value_losses * weights).mean()
        else:
            value_loss = self.value_loss_fn(
                predictions['value'],
                targets['values']
            )

        # Combine
        total_loss = (
            self.policy_weight * policy_loss + 
            self.value_weight * value_loss
        )

        loss_dict = {
            'policy': policy_loss.item(),
            'value': value_loss.item()
        }
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
