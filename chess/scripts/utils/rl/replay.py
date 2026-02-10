"""
Replay buffers for RL training
"""

import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Standard replay buffer with uniform sampling
    """
    
    def __init__(self, max_size, use_fp16=False):
        """
        Args:
            max_size: Maximum buffer size
            use_fp16: Store tensors in float16 to reduce RAM
        """
        self.buffer = deque(maxlen=max_size)
        self.use_fp16 = use_fp16

    def _maybe_fp16(self, position):
        if not self.use_fp16:
            return position
        board, policy, value = position
        if torch.is_tensor(board):
            board = board.half().contiguous()
        if torch.is_tensor(policy):
            policy = policy.half().contiguous()
        if torch.is_tensor(value):
            value = value.half().contiguous()
        return (board, policy, value)
    
    def add(self, position):
        """
        Add position to buffer
        
        Args:
            position: Tuple of (board_tensor, policy_target, value_target)
        """
        self.buffer.append(self._maybe_fp16(position))
    
    def sample(self, batch_size):
        """
        Sample batch uniformly
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Tuple of (boards, policies, values) tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        boards = torch.stack([b for b, _, _ in batch])
        policies = torch.stack([p for _, p, _ in batch])
        values = torch.stack([v for _, _, v in batch])
        
        return boards, policies, values
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay
    
    Samples positions based on TD error priority:
    - High error positions → sampled more often
    - Low error positions → sampled less often
    
    Benefits:
    - Faster learning from "difficult" positions
    - Better sample efficiency
    - Improved convergence
    """
    
    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_end=1.0, epsilon=0.01, use_fp16=False):
        """
        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta_start: Initial importance sampling correction
            beta_end: Final beta value
            epsilon: Small constant to avoid zero priority
            use_fp16: Store tensors in float16 to reduce RAM
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self.use_fp16 = use_fp16
        
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        print(f"🎯 Prioritized Replay Buffer:")
        print(f"   Alpha (priority): {alpha}")
        print(f"   Beta (IS correction): {beta_start} → {beta_end}")
        print(f"   Epsilon: {epsilon}")

    def _maybe_fp16(self, position):
        if not self.use_fp16:
            return position
        board, policy, value = position
        if torch.is_tensor(board):
            board = board.half().contiguous()
        if torch.is_tensor(policy):
            policy = policy.half().contiguous()
        if torch.is_tensor(value):
            value = value.half().contiguous()
        return (board, policy, value)
    
    def add(self, position, priority=None):
        """
        Add position with optional initial priority
        
        Args:
            position: Tuple of (board_tensor, policy_target, value_target)
            priority: Initial priority (default: max priority)
        """
        if priority is None:
            # New positions get max priority (will be sampled quickly)
            priority = self.priorities.max() if self.size > 0 else 1.0
        
        position = self._maybe_fp16(position)

        if len(self.buffer) < self.max_size:
            self.buffer.append(position)
        else:
            self.buffer[self.position] = position
        
        self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, beta=None):
        """
        Sample batch with prioritized sampling
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling correction (default: current beta)
        
        Returns:
            batch: List of positions
            indices: Indices of sampled positions
            weights: Importance sampling weights
        """
        if beta is None:
            beta = self.beta
        
        # Get priorities for all positions
        priorities = self.priorities[:self.size]
        
        # Compute sampling probabilities
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Get batch
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled positions
        
        Args:
            indices: Indices of positions to update
            priorities: New priorities (TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def update_beta(self, progress):
        """
        Update beta (importance sampling correction) based on training progress
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def __len__(self):
        return self.size
