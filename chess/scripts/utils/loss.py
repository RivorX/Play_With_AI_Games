"""
Custom loss functions
"""

import torch
import torch.nn as nn


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