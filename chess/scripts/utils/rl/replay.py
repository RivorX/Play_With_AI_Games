"""
Replay buffer for RL training.
"""

import numpy as np
import torch


_DEFAULT_MAX_POLICY_TARGETS = 256


class ReplayBuffer:
    """
    Replay buffer backed by preallocated tensors.
    """

    def __init__(
        self,
        max_size,
        use_fp16=False,
        decisive_sampling_fraction=0.0,
        decisive_value_epsilon=0.05,
    ):
        self.max_size = int(max_size)
        self.use_fp16 = bool(use_fp16)
        self.decisive_sampling_fraction = max(0.0, min(1.0, float(decisive_sampling_fraction)))
        self.decisive_value_epsilon = max(0.0, float(decisive_value_epsilon))
        self.size = 0
        self.position = 0

        self._boards = None
        self._values = None
        self._policy_indices = None
        self._policy_values = None
        self._policy_lengths = None
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
            "boards": torch.empty((batch_size, *board_shape), dtype=board_dtype),
            "values": torch.empty((batch_size, 1), dtype=board_dtype),
            "policy_indices": torch.full((batch_size, max_len), -1, dtype=torch.int16),
            "policy_values": torch.zeros((batch_size, max_len), dtype=probs_dtype),
            "policy_mask": torch.zeros((batch_size, max_len), dtype=torch.bool),
            "arange": torch.arange(max_len, dtype=torch.int16),
        }
        self._scratch[key] = scratch
        return scratch

    def _normalize_position(self, position):
        if len(position) < 4:
            raise ValueError("Replay position must contain board, policy indices, policy values, and value target.")
        board, policy_indices, policy_values, value = position[:4]

        if self.use_fp16:
            board = board.half().contiguous()
            policy_values = policy_values.half().contiguous()
            value = value.half().contiguous()
        else:
            board = board.contiguous()
            policy_values = policy_values.contiguous()
            value = value.contiguous()

        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        value = value.reshape(1).contiguous()
        return board, policy_indices, policy_values, value

    def _store_at_slot(self, slot, position):
        board, policy_indices, policy_values, value = self._normalize_position(position)

        count = int(policy_indices.numel())
        if count > _DEFAULT_MAX_POLICY_TARGETS:
            count = _DEFAULT_MAX_POLICY_TARGETS
            policy_indices = policy_indices[:count]
            policy_values = policy_values[:count]

        self._boards[slot].copy_(board)
        self._values[slot].copy_(value.to(dtype=self._values.dtype))
        self._policy_indices[slot].fill_(-1)
        self._policy_values[slot].zero_()
        if count > 0:
            self._policy_indices[slot, :count].copy_(policy_indices)
            self._policy_values[slot, :count].copy_(policy_values.to(dtype=self._policy_values.dtype))
        self._policy_lengths[slot] = count

    def _build_batch_from_indices(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        lengths = self._policy_lengths[idx].to(dtype=torch.int16)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0

        scratch = self._get_scratch_batch(batch_size, max_len)
        boards = scratch["boards"]
        values = scratch["values"]

        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])

        if max_len <= 0:
            policy_indices = scratch["policy_indices"][:, :0]
            policy_values = scratch["policy_values"][:, :0]
            policy_mask = scratch["policy_mask"][:, :0]
            return boards, policy_indices, policy_values, policy_mask, values

        policy_indices = scratch["policy_indices"]
        policy_values = scratch["policy_values"]
        policy_mask = scratch["policy_mask"]

        policy_indices.copy_(self._policy_indices[idx, :max_len])
        policy_values.copy_(self._policy_values[idx, :max_len])
        policy_mask.copy_(scratch["arange"].unsqueeze(0) < lengths.unsqueeze(1))
        return boards, policy_indices, policy_values, policy_mask, values

    def add(self, position):
        board = position[0]
        self._ensure_storage_initialized(board)
        self._store_at_slot(self.position, position)
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_packed_batch(self, boards, policy_indices, policy_values, policy_lengths, values):
        if boards is None or int(boards.shape[0]) <= 0:
            return

        self._ensure_storage_initialized(boards[0])
        boards = boards.to(dtype=self._boards.dtype).contiguous()
        values = values.reshape(-1, 1).to(dtype=self._values.dtype).contiguous()
        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        policy_values = policy_values.to(dtype=self._policy_values.dtype).contiguous()
        policy_lengths = policy_lengths.to(dtype=torch.int16).contiguous()

        max_len = int(policy_indices.shape[1]) if policy_indices.dim() == 2 else 0
        if max_len > _DEFAULT_MAX_POLICY_TARGETS:
            max_len = _DEFAULT_MAX_POLICY_TARGETS
            policy_indices = policy_indices[:, :max_len]
            policy_values = policy_values[:, :max_len]
            policy_lengths = torch.clamp(policy_lengths, max=max_len)

        batch_size = int(boards.shape[0])
        remaining = batch_size
        src_start = 0
        while remaining > 0:
            dst_start = self.position
            count = min(remaining, self.max_size - dst_start)
            src_end = src_start + count
            dst_slice = slice(dst_start, dst_start + count)
            src_slice = slice(src_start, src_end)

            self._boards[dst_slice].copy_(boards[src_slice])
            self._values[dst_slice].copy_(values[src_slice])
            self._policy_indices[dst_slice].fill_(-1)
            self._policy_values[dst_slice].zero_()
            if max_len > 0:
                self._policy_indices[dst_slice, :max_len].copy_(policy_indices[src_slice, :max_len])
                self._policy_values[dst_slice, :max_len].copy_(policy_values[src_slice, :max_len])
            self._policy_lengths[dst_slice].copy_(policy_lengths[src_slice])

            self.position = (dst_start + count) % self.max_size
            self.size = min(self.size + count, self.max_size)
            remaining -= count
            src_start = src_end

    def _sample_indices_uniform(self, batch_size):
        return np.random.choice(self.size, batch_size, replace=False)

    def _sample_indices_with_decisive_bias(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if self.decisive_sampling_fraction <= 0.0:
            return self._sample_indices_uniform(batch_size)

        values = self._values[:self.size].reshape(-1)
        decisive_mask = torch.abs(values.float()) > self.decisive_value_epsilon
        decisive_indices = torch.nonzero(decisive_mask, as_tuple=False).reshape(-1).cpu().numpy()

        if decisive_indices.size <= 0:
            return self._sample_indices_uniform(batch_size)

        draw_indices = torch.nonzero(~decisive_mask, as_tuple=False).reshape(-1).cpu().numpy()
        target_decisive = int(round(batch_size * self.decisive_sampling_fraction))
        target_decisive = max(1, min(int(batch_size), target_decisive))
        decisive_take = min(int(decisive_indices.size), target_decisive)

        chosen_parts = []
        if decisive_take > 0:
            chosen_parts.append(
                np.random.choice(decisive_indices, decisive_take, replace=False)
            )

        remaining = int(batch_size - decisive_take)
        if remaining > 0:
            if draw_indices.size >= remaining:
                chosen_parts.append(
                    np.random.choice(draw_indices, remaining, replace=False)
                )
            else:
                chosen_draws = draw_indices.copy()
                chosen_parts.append(chosen_draws)
                extra_needed = remaining - int(chosen_draws.size)
                remaining_decisive_pool = np.setdiff1d(
                    decisive_indices,
                    chosen_parts[0] if chosen_parts else np.empty(0, dtype=np.int64),
                    assume_unique=False,
                )
                if remaining_decisive_pool.size >= extra_needed:
                    chosen_parts.append(
                        np.random.choice(remaining_decisive_pool, extra_needed, replace=False)
                    )
                else:
                    available_all = np.setdiff1d(
                        np.arange(self.size, dtype=np.int64),
                        np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64),
                        assume_unique=False,
                    )
                    if available_all.size > 0:
                        chosen_parts.append(
                            np.random.choice(
                                available_all,
                                min(extra_needed, int(available_all.size)),
                                replace=False,
                            )
                        )

        indices = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64)
        if indices.size < batch_size:
            fallback_pool = np.setdiff1d(
                np.arange(self.size, dtype=np.int64),
                indices,
                assume_unique=False,
            )
            if fallback_pool.size > 0:
                indices = np.concatenate(
                    [
                        indices,
                        np.random.choice(
                            fallback_pool,
                            min(batch_size - int(indices.size), int(fallback_pool.size)),
                            replace=False,
                        ),
                    ]
                )
        if indices.size > batch_size:
            indices = indices[:batch_size]
        np.random.shuffle(indices)
        return indices

    def sample(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        batch_size = max(1, min(int(batch_size), int(self.size)))
        indices = self._sample_indices_with_decisive_bias(batch_size)
        return self._build_batch_from_indices(indices)

    def __len__(self):
        return self.size
