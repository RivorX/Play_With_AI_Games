"""
Replay buffer for RL training.
"""

import numpy as np
import torch

from src.utils.data_helpers import MAX_LEGAL_MOVES


_DEFAULT_MAX_POLICY_TARGETS = 256
REPLAY_SOURCE_UNKNOWN = 0
REPLAY_SOURCE_LEARNER = 1
REPLAY_SOURCE_FROZEN_BEST = 2
REPLAY_SOURCE_LABELS = {
    REPLAY_SOURCE_UNKNOWN: "unknown",
    REPLAY_SOURCE_LEARNER: "learner",
    REPLAY_SOURCE_FROZEN_BEST: "frozen_best",
}


class ReplayBuffer:
    """
    Replay buffer backed by preallocated tensors.
    """

    def __init__(
        self,
        max_size,
        max_policy_targets=_DEFAULT_MAX_POLICY_TARGETS,
        use_fp16=False,
    ):
        self.max_size = int(max_size)
        self.max_policy_targets = max(1, int(max_policy_targets))
        self.use_fp16 = bool(use_fp16)
        self.decisive_value_epsilon = 0.05
        self.value_balance_epsilon = 0.05
        self.recent_window_fraction = 0.25
        self.size = 0
        self.position = 0

        self._boards = None
        self._values = None
        self._policy_indices = None
        self._policy_values = None
        self._policy_lengths = None
        self._legal_indices = None
        self._legal_lengths = None
        self._importance = None
        self._policy_sample_weights = None
        self._value_sample_weights = None
        self._moves_left = None
        self._insertion_iterations = None
        self._source_codes = None
        self._scratch = {}
        self.current_iteration = 0
        self.last_sample_age_stats = {}
        self.last_sample_ages = np.empty(0, dtype=np.float32)

    def _ordered_indices_oldest_to_newest(self):
        if self.size <= 0:
            return np.empty(0, dtype=np.int64)
        if self.size < self.max_size:
            return np.arange(self.size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self.position, self.max_size, dtype=np.int64),
                np.arange(0, self.position, dtype=np.int64),
            )
        )

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
            (self.max_size, self.max_policy_targets),
            -1,
            dtype=torch.int16,
        )
        self._policy_values = torch.zeros(
            (self.max_size, self.max_policy_targets),
            dtype=probs_dtype,
        )
        self._policy_lengths = torch.zeros((self.max_size,), dtype=torch.int16)
        self._legal_indices = torch.full(
            (self.max_size, MAX_LEGAL_MOVES),
            -1,
            dtype=torch.int16,
        )
        self._legal_lengths = torch.zeros((self.max_size,), dtype=torch.int16)
        self._importance = torch.zeros((self.max_size,), dtype=torch.float32)
        self._policy_sample_weights = torch.ones((self.max_size,), dtype=torch.float32)
        self._value_sample_weights = torch.ones((self.max_size,), dtype=torch.float32)
        self._moves_left = torch.zeros((self.max_size, 1), dtype=torch.float32)
        self._insertion_iterations = torch.zeros((self.max_size,), dtype=torch.int32)
        self._source_codes = torch.zeros((self.max_size,), dtype=torch.int8)

    def set_current_iteration(self, iteration):
        try:
            self.current_iteration = max(0, int(iteration))
        except Exception:
            self.current_iteration = 0

    def _get_scratch_batch(self, batch_size, max_len, max_legal_len):
        key = (int(batch_size), int(max_len), int(max_legal_len))
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
            "legal_indices": torch.full((batch_size, max_legal_len), -1, dtype=torch.int16),
            "legal_mask": torch.zeros((batch_size, max_legal_len), dtype=torch.bool),
            "policy_sample_weights": torch.ones((batch_size,), dtype=torch.float32),
            "value_sample_weights": torch.ones((batch_size,), dtype=torch.float32),
            "moves_left": torch.zeros((batch_size, 1), dtype=torch.float32),
            "arange": torch.arange(max_len, dtype=torch.int16),
            "legal_arange": torch.arange(max_legal_len, dtype=torch.int16),
        }
        self._scratch[key] = scratch
        return scratch

    def _normalize_position(self, position):
        if len(position) < 4:
            raise ValueError("Replay position must contain board, policy indices, policy values, and value target.")
        board, policy_indices, policy_values, value = position[:4]
        importance = float(position[4]) if len(position) > 4 else 0.0
        policy_weight = float(position[5]) if len(position) > 5 else 1.0
        value_weight = float(position[6]) if len(position) > 6 else 1.0
        source_code = int(position[7]) if len(position) > 7 else REPLAY_SOURCE_UNKNOWN
        moves_left = float(position[8]) if len(position) > 8 else 0.0
        legal_indices = position[9] if len(position) > 9 and position[9] is not None else policy_indices
        if self.use_fp16:
            board = board.half().contiguous()
            policy_values = policy_values.half().contiguous()
            value = value.half().contiguous()
        else:
            board = board.contiguous()
            policy_values = policy_values.contiguous()
            value = value.contiguous()

        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        if not torch.is_tensor(legal_indices):
            legal_indices = torch.as_tensor(legal_indices)
        legal_indices = legal_indices.to(dtype=torch.int16).contiguous()
        value = value.reshape(1).contiguous()
        return board, policy_indices, policy_values, value, importance, policy_weight, value_weight, source_code, moves_left, legal_indices

    def _store_at_slot(self, slot, position):
        (
            board,
            policy_indices,
            policy_values,
            value,
            importance,
            policy_weight,
            value_weight,
            source_code,
            moves_left,
            legal_indices,
        ) = self._normalize_position(position)

        count = int(policy_indices.numel())
        if count > self.max_policy_targets:
            count = self.max_policy_targets
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
        legal_count = min(int(legal_indices.numel()), MAX_LEGAL_MOVES)
        self._legal_indices[slot].fill_(-1)
        if legal_count > 0:
            self._legal_indices[slot, :legal_count].copy_(legal_indices[:legal_count])
        self._legal_lengths[slot] = legal_count
        self._importance[slot] = float(importance)
        self._policy_sample_weights[slot] = float(policy_weight)
        self._value_sample_weights[slot] = float(value_weight)
        self._moves_left[slot] = float(moves_left)
        self._insertion_iterations[slot] = int(self.current_iteration)
        self._source_codes[slot] = int(source_code)

    def _build_batch_from_indices(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        lengths = self._policy_lengths[idx].to(dtype=torch.int16)
        legal_lengths = self._legal_lengths[idx].to(dtype=torch.int16)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0
        max_legal_len = int(legal_lengths.max().item()) if batch_size > 0 else 0

        scratch = self._get_scratch_batch(batch_size, max_len, max_legal_len)
        boards = scratch["boards"]
        values = scratch["values"]
        policy_sample_weights = scratch["policy_sample_weights"]
        value_sample_weights = scratch["value_sample_weights"]
        moves_left = scratch["moves_left"]
        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])
        policy_sample_weights.copy_(self._policy_sample_weights[idx])
        value_sample_weights.copy_(self._value_sample_weights[idx])
        moves_left.copy_(self._moves_left[idx])

        if max_len <= 0:
            policy_indices = scratch["policy_indices"][:, :0]
            policy_values = scratch["policy_values"][:, :0]
            policy_mask = scratch["policy_mask"][:, :0]
            legal_indices = scratch["legal_indices"][:, :max_legal_len]
            legal_mask = scratch["legal_mask"][:, :max_legal_len]
            if max_legal_len > 0:
                legal_indices.copy_(self._legal_indices[idx, :max_legal_len])
                legal_mask.copy_(scratch["legal_arange"].unsqueeze(0) < legal_lengths.unsqueeze(1))
            return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights, value_sample_weights, moves_left, legal_indices, legal_mask

        policy_indices = scratch["policy_indices"]
        policy_values = scratch["policy_values"]
        policy_mask = scratch["policy_mask"]
        legal_indices = scratch["legal_indices"]
        legal_mask = scratch["legal_mask"]

        policy_indices.copy_(self._policy_indices[idx, :max_len])
        policy_values.copy_(self._policy_values[idx, :max_len])
        policy_mask.copy_(scratch["arange"].unsqueeze(0) < lengths.unsqueeze(1))
        if max_legal_len > 0:
            legal_indices.copy_(self._legal_indices[idx, :max_legal_len])
            legal_mask.copy_(scratch["legal_arange"].unsqueeze(0) < legal_lengths.unsqueeze(1))
        return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights, value_sample_weights, moves_left, legal_indices, legal_mask

    def add(self, position):
        board = position[0]
        self._ensure_storage_initialized(board)
        self._store_at_slot(self.position, position)
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_packed_batch(
        self,
        boards,
        policy_indices,
        policy_values,
        policy_lengths,
        values,
        importance_scores=None,
        policy_weights=None,
        value_weights=None,
        moves_left=None,
        legal_indices=None,
        legal_lengths=None,
        source_codes=None,
    ):
        if boards is None or int(boards.shape[0]) <= 0:
            return

        self._ensure_storage_initialized(boards[0])
        boards = boards.to(dtype=self._boards.dtype).contiguous()
        values = values.reshape(-1, 1).to(dtype=self._values.dtype).contiguous()
        policy_indices = policy_indices.to(dtype=torch.int16).contiguous()
        policy_values = policy_values.to(dtype=self._policy_values.dtype).contiguous()
        policy_lengths = policy_lengths.to(dtype=torch.int16).contiguous()
        if importance_scores is None:
            importance_scores = torch.zeros((int(boards.shape[0]),), dtype=torch.float32)
        else:
            importance_scores = importance_scores.reshape(-1).to(dtype=torch.float32).contiguous()
        if policy_weights is None:
            policy_weights = torch.ones((int(boards.shape[0]),), dtype=torch.float32)
        else:
            policy_weights = policy_weights.reshape(-1).to(dtype=torch.float32).contiguous()
        if value_weights is None:
            value_weights = torch.ones((int(boards.shape[0]),), dtype=torch.float32)
        else:
            value_weights = value_weights.reshape(-1).to(dtype=torch.float32).contiguous()
        if moves_left is None:
            moves_left = torch.zeros((int(boards.shape[0]), 1), dtype=torch.float32)
        else:
            moves_left = moves_left.reshape(-1, 1).to(dtype=torch.float32).contiguous()
        if legal_indices is None:
            legal_indices = policy_indices
            legal_lengths = policy_lengths
        else:
            legal_indices = legal_indices.to(dtype=torch.int16).contiguous()
            if legal_lengths is None:
                legal_lengths = (legal_indices >= 0).sum(dim=1).to(dtype=torch.int16)
            else:
                legal_lengths = legal_lengths.to(dtype=torch.int16).contiguous()
        if source_codes is None:
            source_codes = torch.zeros((int(boards.shape[0]),), dtype=torch.int8)
        else:
            source_codes = source_codes.reshape(-1).to(dtype=torch.int8).contiguous()
        max_len = int(policy_indices.shape[1]) if policy_indices.dim() == 2 else 0
        if max_len > self.max_policy_targets:
            max_len = self.max_policy_targets
            policy_indices = policy_indices[:, :max_len]
            policy_values = policy_values[:, :max_len]
            policy_lengths = torch.clamp(policy_lengths, max=max_len)
        max_legal_len = int(legal_indices.shape[1]) if legal_indices.dim() == 2 else 0
        if max_legal_len > MAX_LEGAL_MOVES:
            max_legal_len = MAX_LEGAL_MOVES
            legal_indices = legal_indices[:, :max_legal_len]
            legal_lengths = torch.clamp(legal_lengths, max=max_legal_len)

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
            self._legal_indices[dst_slice].fill_(-1)
            if max_legal_len > 0:
                self._legal_indices[dst_slice, :max_legal_len].copy_(legal_indices[src_slice, :max_legal_len])
            self._legal_lengths[dst_slice].copy_(legal_lengths[src_slice])
            self._importance[dst_slice].copy_(importance_scores[src_slice])
            self._policy_sample_weights[dst_slice].copy_(policy_weights[src_slice])
            self._value_sample_weights[dst_slice].copy_(value_weights[src_slice])
            self._moves_left[dst_slice].copy_(moves_left[src_slice])
            self._insertion_iterations[dst_slice].fill_(int(self.current_iteration))
            self._source_codes[dst_slice].copy_(source_codes[src_slice])
            self.position = (dst_start + count) % self.max_size
            self.size = min(self.size + count, self.max_size)
            remaining -= count
            src_start = src_end

    def _sample_indices_uniform(self, batch_size):
        return np.random.choice(self.size, batch_size, replace=False)

    @staticmethod
    def _select_resize_keep_indices(ordered_indices, keep_size):
        keep_size = max(0, int(keep_size))
        if keep_size <= 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(ordered_indices[-keep_size:], dtype=np.int64)

    def select_indices(self, sample_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        sample_size = max(1, min(int(sample_size), int(self.size)))
        return self._sample_indices_uniform(sample_size)

    def sample_from_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size <= 0:
            raise ValueError("Cannot build an empty replay batch.")
        self._record_sample_age_stats(indices)
        return self._build_batch_from_indices(indices)

    def sample(self, batch_size):
        return self.sample_from_indices(self.select_indices(batch_size))

    def _record_sample_age_stats(self, indices):
        if self._insertion_iterations is None:
            self.last_sample_ages = np.empty(0, dtype=np.float32)
            self.last_sample_age_stats = {}
            return
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size <= 0:
            self.last_sample_ages = np.empty(0, dtype=np.float32)
            self.last_sample_age_stats = {}
            return

        idx = torch.as_tensor(indices, dtype=torch.long)
        inserted = self._insertion_iterations[idx].to(dtype=torch.float32).cpu().numpy()
        ages = np.maximum(0.0, float(self.current_iteration) - inserted).astype(np.float32, copy=False)
        self.last_sample_ages = ages
        self.last_sample_age_stats = {
            "avg": float(np.mean(ages)),
            "p10": float(np.percentile(ages, 10)),
            "p50": float(np.percentile(ages, 50)),
            "p90": float(np.percentile(ages, 90)),
        }

    def resize(self, new_max_size):
        new_max_size = max(1, int(new_max_size))
        if new_max_size == self.max_size:
            return False

        if self._boards is None:
            self.max_size = new_max_size
            self.size = min(self.size, self.max_size)
            self.position = min(self.position, max(0, self.max_size - 1))
            self._scratch = {}
            return True

        keep_size = min(int(self.size), new_max_size)
        ordered_indices = self._ordered_indices_oldest_to_newest()
        keep_indices = self._select_resize_keep_indices(ordered_indices, keep_size)

        old_boards = self._boards
        old_values = self._values
        old_policy_indices = self._policy_indices
        old_policy_values = self._policy_values
        old_policy_lengths = self._policy_lengths
        old_legal_indices = self._legal_indices
        old_legal_lengths = self._legal_lengths
        old_importance = self._importance
        old_policy_sample_weights = self._policy_sample_weights
        old_value_sample_weights = self._value_sample_weights
        old_moves_left = self._moves_left
        old_insertion_iterations = self._insertion_iterations
        old_source_codes = self._source_codes

        board_shape = tuple(old_boards.shape[1:])
        board_dtype = old_boards.dtype
        probs_dtype = old_policy_values.dtype

        self.max_size = new_max_size
        self._boards = torch.empty((self.max_size, *board_shape), dtype=board_dtype)
        self._values = torch.empty((self.max_size, 1), dtype=old_values.dtype)
        self._policy_indices = torch.full(
            (self.max_size, self.max_policy_targets),
            -1,
            dtype=old_policy_indices.dtype,
        )
        self._policy_values = torch.zeros(
            (self.max_size, self.max_policy_targets),
            dtype=probs_dtype,
        )
        self._policy_lengths = torch.zeros((self.max_size,), dtype=old_policy_lengths.dtype)
        self._legal_indices = torch.full(
            (self.max_size, MAX_LEGAL_MOVES),
            -1,
            dtype=old_legal_indices.dtype,
        )
        self._legal_lengths = torch.zeros((self.max_size,), dtype=old_legal_lengths.dtype)
        self._importance = torch.zeros((self.max_size,), dtype=old_importance.dtype)
        self._policy_sample_weights = torch.ones((self.max_size,), dtype=old_policy_sample_weights.dtype)
        self._value_sample_weights = torch.ones((self.max_size,), dtype=old_value_sample_weights.dtype)
        self._moves_left = torch.zeros((self.max_size, 1), dtype=old_moves_left.dtype)
        self._insertion_iterations = torch.zeros((self.max_size,), dtype=old_insertion_iterations.dtype)
        self._source_codes = torch.zeros((self.max_size,), dtype=old_source_codes.dtype)

        if keep_size > 0:
            idx = torch.as_tensor(keep_indices, dtype=torch.long)
            self._boards[:keep_size].copy_(old_boards[idx])
            self._values[:keep_size].copy_(old_values[idx])
            self._policy_indices[:keep_size].copy_(old_policy_indices[idx])
            self._policy_values[:keep_size].copy_(old_policy_values[idx])
            self._policy_lengths[:keep_size].copy_(old_policy_lengths[idx])
            self._legal_indices[:keep_size].copy_(old_legal_indices[idx])
            self._legal_lengths[:keep_size].copy_(old_legal_lengths[idx])
            self._importance[:keep_size].copy_(old_importance[idx])
            self._policy_sample_weights[:keep_size].copy_(old_policy_sample_weights[idx])
            self._value_sample_weights[:keep_size].copy_(old_value_sample_weights[idx])
            self._moves_left[:keep_size].copy_(old_moves_left[idx])
            self._insertion_iterations[:keep_size].copy_(old_insertion_iterations[idx])
            self._source_codes[:keep_size].copy_(old_source_codes[idx])

        self.size = keep_size
        self.position = 0 if keep_size >= self.max_size else keep_size
        self._scratch = {}
        return True

    def quality_stats(self, recent_window_fraction=None):
        """Return cheap aggregate stats describing replay target quality."""
        stats = {
            "size": int(self.size),
            "capacity": int(self.max_size),
            "fill_rate": float(self.size) / float(max(1, self.max_size)),
        }
        if self.size <= 0 or self._boards is None:
            return stats

        values = self._values[:self.size].reshape(-1).float()
        policy_lengths = self._policy_lengths[:self.size].to(dtype=torch.float32)
        policy_weights = self._policy_sample_weights[:self.size].float()
        value_weights = self._value_sample_weights[:self.size].float()
        importance = self._importance[:self.size].float()
        source_codes = self._source_codes[:self.size].to(dtype=torch.int16)
        decisive_mask = torch.abs(values) > float(self.decisive_value_epsilon)
        draw_mask = ~decisive_mask
        positive_mask = values > float(self.value_balance_epsilon)
        negative_mask = values < -float(self.value_balance_epsilon)
        neutral_mask = ~(positive_mask | negative_mask)

        def _safe_mean(tensor):
            if tensor.numel() <= 0:
                return 0.0
            return float(tensor.mean().item())

        def _safe_std(tensor):
            if tensor.numel() <= 0:
                return 0.0
            return float(tensor.std(unbiased=False).item())

        def _safe_quantile(tensor, q):
            if tensor.numel() <= 0:
                return 0.0
            try:
                return float(torch.quantile(tensor.float(), float(q)).item())
            except Exception:
                return float(np.quantile(tensor.float().cpu().numpy(), float(q)))

        stats.update({
            "decisive_fraction": float(decisive_mask.float().mean().item()),
            "draw_fraction": float(draw_mask.float().mean().item()),
            "value_mean": _safe_mean(values),
            "value_std": _safe_std(values),
            "value_positive_fraction": float(positive_mask.float().mean().item()),
            "value_neutral_fraction": float(neutral_mask.float().mean().item()),
            "value_negative_fraction": float(negative_mask.float().mean().item()),
            "policy_weight_mean": _safe_mean(policy_weights),
            "policy_weight_p10": _safe_quantile(policy_weights, 0.10),
            "policy_weight_low_fraction": float((policy_weights < 0.75).float().mean().item()),
            "value_weight_mean": _safe_mean(value_weights),
            "value_weight_p10": _safe_quantile(value_weights, 0.10),
            "value_weight_low_fraction": float((value_weights < 0.75).float().mean().item()),
            "policy_target_len_mean": _safe_mean(policy_lengths),
            "policy_target_len_p90": _safe_quantile(policy_lengths, 0.90),
            "importance_mean": _safe_mean(importance),
            "importance_p90": _safe_quantile(importance, 0.90),
        })
        policy_weight_total = max(1e-8, float(policy_weights.sum().item()))
        source_counts = {}
        source_policy_weight_sums = {}
        for source_code, source_label in REPLAY_SOURCE_LABELS.items():
            source_mask = source_codes == int(source_code)
            source_count = int(source_mask.sum().item())
            source_policy_weight_sum = float(policy_weights[source_mask].sum().item()) if source_count > 0 else 0.0
            source_counts[source_label] = source_count
            source_policy_weight_sums[source_label] = source_policy_weight_sum
            stats[f"source_{source_label}_fraction"] = float(source_count) / float(self.size)
            stats[f"source_{source_label}_policy_weight_share"] = source_policy_weight_sum / policy_weight_total
        stats["source_counts"] = source_counts
        stats["source_policy_weight_sums"] = source_policy_weight_sums

        policy_values = self._policy_values[:self.size].float()
        if policy_values.numel() > 0:
            valid_mask = self._policy_indices[:self.size] >= 0
            probs = torch.where(valid_mask, torch.clamp(policy_values, min=0.0), torch.zeros_like(policy_values))
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
            top1 = probs.max(dim=1).values
            non_empty = policy_lengths > 0
            topk_count = min(3, int(probs.shape[1])) if probs.dim() == 2 else 0
            top3 = (
                torch.topk(probs, k=topk_count, dim=1).values.sum(dim=1)
                if topk_count > 0
                else torch.zeros_like(top1)
            )
            entropy_mean = _safe_mean(entropy[non_empty])
            stats["policy_target_entropy_mean"] = _safe_mean(entropy[non_empty])
            stats["policy_target_top1_prob_mean"] = _safe_mean(top1[non_empty])
            stats["policy_target_top3_prob_mean"] = _safe_mean(top3[non_empty])
            stats["policy_target_effective_moves"] = float(np.exp(entropy_mean)) if entropy_mean > 0.0 else 0.0
        else:
            stats["policy_target_entropy_mean"] = 0.0
            stats["policy_target_top1_prob_mean"] = 0.0
            stats["policy_target_top3_prob_mean"] = 0.0
            stats["policy_target_effective_moves"] = 0.0

        recent_fraction = (
            self.recent_window_fraction
            if recent_window_fraction is None
            else max(0.01, min(1.0, float(recent_window_fraction)))
        )
        recent_count = max(1, int(round(float(self.size) * float(recent_fraction))))
        recent_indices = self._ordered_indices_oldest_to_newest()[-recent_count:]
        if recent_indices.size > 0:
            idx = torch.as_tensor(recent_indices, dtype=torch.long)
            recent_values = self._values[idx].reshape(-1).float()
            recent_decisive = torch.abs(recent_values) > float(self.decisive_value_epsilon)
            recent_positive = recent_values > float(self.value_balance_epsilon)
            recent_negative = recent_values < -float(self.value_balance_epsilon)
            recent_neutral = ~(recent_positive | recent_negative)
            recent_weights = self._policy_sample_weights[idx].float()
            stats.update({
                "recent_count": int(recent_indices.size),
                "recent_decisive_fraction": float(recent_decisive.float().mean().item()),
                "recent_draw_fraction": float((~recent_decisive).float().mean().item()),
                "recent_value_mean": _safe_mean(recent_values),
                "recent_value_std": _safe_std(recent_values),
                "recent_value_positive_fraction": float(recent_positive.float().mean().item()),
                "recent_value_neutral_fraction": float(recent_neutral.float().mean().item()),
                "recent_value_negative_fraction": float(recent_negative.float().mean().item()),
                "recent_policy_weight_mean": _safe_mean(recent_weights),
            })
        return stats

    def __len__(self):
        return self.size
