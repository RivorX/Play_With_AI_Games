"""
Replay buffers for RL training
"""

import torch
import numpy as np


_DEFAULT_MAX_POLICY_TARGETS = 256


def _pad_sparse_policy_batch(batch, probs_dtype=torch.float32):
    """
    Collate sparse policy targets into padded tensors.

    Each position is expected as:
        (board_tensor, policy_indices, policy_values, value_target)
    where `policy_indices` is 1D int tensor and `policy_values` is 1D float tensor
    of the same length.
    """
    boards = torch.stack([b for b, _, _, _ in batch])
    values = torch.stack([v for _, _, _, v in batch])

    max_len = max((idx.numel() for _, idx, _, _ in batch), default=0)
    if max_len <= 0:
        batch_size = len(batch)
        policy_indices = torch.empty((batch_size, 0), dtype=torch.int16)
        policy_values = torch.empty((batch_size, 0), dtype=probs_dtype)
        policy_mask = torch.empty((batch_size, 0), dtype=torch.bool)
        return boards, policy_indices, policy_values, policy_mask, values

    batch_size = len(batch)
    policy_indices = torch.full((batch_size, max_len), -1, dtype=torch.int16)
    policy_values = torch.zeros((batch_size, max_len), dtype=probs_dtype)
    policy_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for row, (_, indices, probs, _) in enumerate(batch):
        count = int(indices.numel())
        if count <= 0:
            continue
        policy_indices[row, :count] = indices
        policy_values[row, :count] = probs.to(dtype=probs_dtype)
        policy_mask[row, :count] = True

    return boards, policy_indices, policy_values, policy_mask, values


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
        self.max_size = int(max_size)
        self.use_fp16 = use_fp16
        self.size = 0
        self.position = 0

        # Lazy-initialized on first sample add (depends on input planes).
        self._boards = None
        self._values = None
        self._policy_indices = None
        self._policy_values = None
        self._policy_lengths = None

        # Reused scratch buffers for sampled batches (avoid per-step realloc).
        self._scratch = {}

    def _ensure_storage_initialized(self, board):
        if self._boards is not None:
            return

        if not torch.is_tensor(board):
            board = torch.as_tensor(board)

        board_shape = tuple(board.shape)
        board_dtype = torch.float16 if self.use_fp16 else torch.float32
        probs_dtype = torch.float16 if self.use_fp16 else torch.float32

        self._boards = torch.empty((self.max_size, *board_shape), dtype=board_dtype)
        self._values = torch.empty((self.max_size, 1), dtype=board_dtype)
        self._policy_indices = torch.full(
            (self.max_size, _DEFAULT_MAX_POLICY_TARGETS),
            -1,
            dtype=torch.int16,
        )
        self._policy_values = torch.zeros(
            (self.max_size, _DEFAULT_MAX_POLICY_TARGETS),
            dtype=probs_dtype,
        )
        self._policy_lengths = torch.zeros((self.max_size,), dtype=torch.int16)

    def _get_scratch_batch(self, batch_size, max_len):
        key = (int(batch_size), int(max_len))
        scratch = self._scratch.get(key)
        if scratch is not None:
            return scratch

        board_shape = tuple(self._boards.shape[1:])
        board_dtype = self._boards.dtype
        probs_dtype = self._policy_values.dtype

        scratch = {
            'boards': torch.empty((batch_size, *board_shape), dtype=board_dtype),
            'values': torch.empty((batch_size, 1), dtype=board_dtype),
            'policy_indices': torch.full((batch_size, max_len), -1, dtype=torch.int16),
            'policy_values': torch.zeros((batch_size, max_len), dtype=probs_dtype),
            'policy_mask': torch.zeros((batch_size, max_len), dtype=torch.bool),
            'arange': torch.arange(max_len, dtype=torch.int16),
        }
        self._scratch[key] = scratch
        return scratch

    def _maybe_fp16(self, position):
        if not self.use_fp16:
            return position
        board, policy_indices, policy_values, value = position
        if torch.is_tensor(board):
            board = board.half().contiguous()
        if torch.is_tensor(policy_indices):
            policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        if torch.is_tensor(policy_values):
            policy_values = policy_values.half().contiguous()
        if torch.is_tensor(value):
            value = value.half().contiguous()
        return (board, policy_indices, policy_values, value)

    def _store_at_slot(self, slot, position):
        board, policy_indices, policy_values, value = self._maybe_fp16(position)

        board = board.contiguous()
        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        policy_values = policy_values.to(dtype=self._policy_values.dtype).contiguous()
        value = value.reshape(1).to(dtype=self._values.dtype).contiguous()

        count = int(policy_indices.numel())
        if count > _DEFAULT_MAX_POLICY_TARGETS:
            count = _DEFAULT_MAX_POLICY_TARGETS
            policy_indices = policy_indices[:count]
            policy_values = policy_values[:count]

        self._boards[slot].copy_(board)
        self._values[slot].copy_(value)
        self._policy_indices[slot].fill_(-1)
        self._policy_values[slot].zero_()
        if count > 0:
            self._policy_indices[slot, :count].copy_(policy_indices)
            self._policy_values[slot, :count].copy_(policy_values)
        self._policy_lengths[slot] = count

    def _build_batch_from_indices(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        lengths = self._policy_lengths[idx].to(dtype=torch.int16)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0

        scratch = self._get_scratch_batch(batch_size, max_len)
        boards = scratch['boards']
        values = scratch['values']

        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])

        if max_len <= 0:
            policy_indices = scratch['policy_indices'][:, :0]
            policy_values = scratch['policy_values'][:, :0]
            policy_mask = scratch['policy_mask'][:, :0]
            return boards, policy_indices, policy_values, policy_mask, values

        policy_indices = scratch['policy_indices']
        policy_values = scratch['policy_values']
        policy_mask = scratch['policy_mask']

        policy_indices.copy_(self._policy_indices[idx, :max_len])
        policy_values.copy_(self._policy_values[idx, :max_len])

        arange = scratch['arange']
        policy_mask.copy_(arange.unsqueeze(0) < lengths.unsqueeze(1))

        return boards, policy_indices, policy_values, policy_mask, values
    
    def add(self, position):
        """
        Add position to buffer
        
        Args:
            position: Tuple of (board_tensor, policy_indices, policy_values, value_target)
        """
        board = position[0]
        self._ensure_storage_initialized(board)

        slot = self.position
        self._store_at_slot(slot, position)

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample batch uniformly
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Tuple of (boards, policy_indices, policy_values, policy_mask, values) tensors
        """
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = np.random.choice(self.size, batch_size, replace=False)
        return self._build_batch_from_indices(indices)
    
    def __len__(self):
        return self.size


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
        
        self.buffer = []  # Kept for API compatibility; storage moved to preallocated tensors.
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self._insert_steps = np.zeros(max_size, dtype=np.int64)
        self._global_step = 0
        self.position = 0
        self.size = 0

        # Lazy-initialized on first sample add (depends on input planes).
        self._boards = None
        self._values = None
        self._policy_indices = None
        self._policy_values = None
        self._policy_lengths = None

        # Reused scratch buffers for sampled batches (avoid per-step realloc).
        self._scratch = {}
        self.last_sample_age_mean = 0.0
        
        print(f"🎯 Prioritized Replay Buffer:")
        print(f"   Alpha (priority): {alpha}")
        print(f"   Beta (IS correction): {beta_start} → {beta_end}")
        print(f"   Epsilon: {epsilon}")

    def _ensure_storage_initialized(self, board):
        if self._boards is not None:
            return

        if not torch.is_tensor(board):
            board = torch.as_tensor(board)

        board_shape = tuple(board.shape)
        board_dtype = torch.float16 if self.use_fp16 else torch.float32
        probs_dtype = torch.float16 if self.use_fp16 else torch.float32

        self._boards = torch.empty((self.max_size, *board_shape), dtype=board_dtype)
        self._values = torch.empty((self.max_size, 1), dtype=board_dtype)
        self._policy_indices = torch.full(
            (self.max_size, _DEFAULT_MAX_POLICY_TARGETS),
            -1,
            dtype=torch.int16,
        )
        self._policy_values = torch.zeros(
            (self.max_size, _DEFAULT_MAX_POLICY_TARGETS),
            dtype=probs_dtype,
        )
        self._policy_lengths = torch.zeros((self.max_size,), dtype=torch.int16)

    def _get_scratch_batch(self, batch_size, max_len):
        key = (int(batch_size), int(max_len))
        scratch = self._scratch.get(key)
        if scratch is not None:
            return scratch

        board_shape = tuple(self._boards.shape[1:])
        board_dtype = self._boards.dtype
        probs_dtype = self._policy_values.dtype

        scratch = {
            'boards': torch.empty((batch_size, *board_shape), dtype=board_dtype),
            'values': torch.empty((batch_size, 1), dtype=board_dtype),
            'policy_indices': torch.full((batch_size, max_len), -1, dtype=torch.int16),
            'policy_values': torch.zeros((batch_size, max_len), dtype=probs_dtype),
            'policy_mask': torch.zeros((batch_size, max_len), dtype=torch.bool),
            'arange': torch.arange(max_len, dtype=torch.int16),
        }
        self._scratch[key] = scratch
        return scratch

    def _maybe_fp16(self, position):
        if not self.use_fp16:
            return position
        board, policy_indices, policy_values, value = position
        if torch.is_tensor(board):
            board = board.half().contiguous()
        if torch.is_tensor(policy_indices):
            policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        if torch.is_tensor(policy_values):
            policy_values = policy_values.half().contiguous()
        if torch.is_tensor(value):
            value = value.half().contiguous()
        return (board, policy_indices, policy_values, value)

    def _store_at_slot(self, slot, position):
        board, policy_indices, policy_values, value = self._maybe_fp16(position)

        board = board.contiguous()
        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        policy_values = policy_values.to(dtype=self._policy_values.dtype).contiguous()
        value = value.reshape(1).to(dtype=self._values.dtype).contiguous()

        count = int(policy_indices.numel())
        if count > _DEFAULT_MAX_POLICY_TARGETS:
            count = _DEFAULT_MAX_POLICY_TARGETS
            policy_indices = policy_indices[:count]
            policy_values = policy_values[:count]

        self._boards[slot].copy_(board)
        self._values[slot].copy_(value)
        self._policy_indices[slot].fill_(-1)
        self._policy_values[slot].zero_()
        if count > 0:
            self._policy_indices[slot, :count].copy_(policy_indices)
            self._policy_values[slot, :count].copy_(policy_values)
        self._policy_lengths[slot] = count

    def _build_batch_from_indices(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        lengths = self._policy_lengths[idx].to(dtype=torch.int16)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0

        scratch = self._get_scratch_batch(batch_size, max_len)
        boards = scratch['boards']
        values = scratch['values']

        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])

        if max_len <= 0:
            policy_indices = scratch['policy_indices'][:, :0]
            policy_values = scratch['policy_values'][:, :0]
            policy_mask = scratch['policy_mask'][:, :0]
            return boards, policy_indices, policy_values, policy_mask, values

        policy_indices = scratch['policy_indices']
        policy_values = scratch['policy_values']
        policy_mask = scratch['policy_mask']

        policy_indices.copy_(self._policy_indices[idx, :max_len])
        policy_values.copy_(self._policy_values[idx, :max_len])

        arange = scratch['arange']
        policy_mask.copy_(arange.unsqueeze(0) < lengths.unsqueeze(1))

        return boards, policy_indices, policy_values, policy_mask, values
    
    def add(self, position, priority=None):
        """
        Add position with optional initial priority
        
        Args:
            position: Tuple of (board_tensor, policy_indices, policy_values, value_target)
            priority: Initial priority (default: max priority)
        """
        if priority is None:
            # New positions get max priority (will be sampled quickly)
            priority = self.priorities.max() if self.size > 0 else 1.0
        
        board = position[0]
        self._ensure_storage_initialized(board)

        slot = self.position
        self._store_at_slot(slot, position)

        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        
        self.priorities[slot] = priority
        self._insert_steps[slot] = self._global_step
        self._global_step += 1

        self.position = (slot + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, beta=None, age_decay_lambda=0.0):
        """
        Sample batch with prioritized sampling
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling correction (default: current beta)
        
        Returns:
            batch: Padded batch tensors
            indices: Indices of sampled positions
            weights: Importance sampling weights
        """
        if beta is None:
            beta = self.beta
        
        indices, weights = self._sample_prioritized_indices(
            batch_size,
            beta=beta,
            age_decay_lambda=age_decay_lambda,
        )
        return self._build_batch_from_indices(indices), indices, weights

    def _sample_prioritized_indices(self, batch_size, beta=None, age_decay_lambda=0.0):
        if beta is None:
            beta = self.beta

        priorities = self.priorities[:self.size].astype(np.float32, copy=True)
        if age_decay_lambda > 0:
            ages = (self._global_step - self._insert_steps[:self.size]).astype(np.float32)
            priorities *= np.exp(-float(age_decay_lambda) * ages)

        probs = priorities ** self.alpha
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full(self.size, 1.0 / max(1, self.size), dtype=np.float32)
        else:
            probs = probs / probs_sum

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        if indices.size > 0:
            ages = (self._global_step - self._insert_steps[indices]).astype(np.float32)
            self.last_sample_age_mean = float(np.mean(ages))
        else:
            self.last_sample_age_mean = 0.0
        return indices.astype(np.int64, copy=False), weights.astype(np.float32, copy=False)

    def sample_mixed(self, batch_size, uniform_fraction=0.25, beta=None, age_decay_lambda=0.0):
        """
        Sample a mixed batch: part prioritized, part uniform.

        Uniform samples receive weight 1.0, prioritized samples keep standard
        importance-sampling weights.
        """
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        if beta is None:
            beta = self.beta

        uniform_fraction = float(uniform_fraction)
        uniform_fraction = max(0.0, min(1.0, uniform_fraction))

        uniform_count = int(round(batch_size * uniform_fraction))
        uniform_count = min(uniform_count, batch_size, self.size)
        prioritized_count = min(batch_size - uniform_count, self.size - uniform_count)

        prioritized_indices = np.empty(0, dtype=np.int64)
        prioritized_weights = np.empty(0, dtype=np.float32)
        if prioritized_count > 0:
            prioritized_indices, prioritized_weights = self._sample_prioritized_indices(
                prioritized_count,
                beta=beta,
                age_decay_lambda=age_decay_lambda,
            )

        uniform_indices = np.empty(0, dtype=np.int64)
        if uniform_count > 0:
            available = np.setdiff1d(
                np.arange(self.size, dtype=np.int64),
                prioritized_indices,
                assume_unique=False,
            )
            if available.size < uniform_count:
                uniform_count = int(available.size)
            if uniform_count > 0:
                uniform_indices = np.random.choice(available, uniform_count, replace=False)

        indices = np.concatenate([prioritized_indices, uniform_indices])
        if indices.size == 0:
            raise ValueError("Failed to sample any replay indices.")

        weights = np.concatenate([
            prioritized_weights,
            np.ones(uniform_indices.size, dtype=np.float32),
        ])

        if indices.size > 1:
            order = np.random.permutation(indices.size)
            indices = indices[order]
            weights = weights[order]

        if indices.size > 0:
            ages = (self._global_step - self._insert_steps[indices]).astype(np.float32)
            self.last_sample_age_mean = float(np.mean(ages))
        else:
            self.last_sample_age_mean = 0.0

        return self._build_batch_from_indices(indices), indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled positions
        
        Args:
            indices: Indices of positions to update
            priorities: New priorities (TD errors)
        """
        if len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[indices] = priorities + self.epsilon
    
    def update_beta(self, progress):
        """
        Update beta (importance sampling correction) based on training progress
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def __len__(self):
        return self.size
