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
        max_policy_targets=_DEFAULT_MAX_POLICY_TARGETS,
        use_fp16=False,
        decisive_sampling_fraction=0.0,
        decisive_value_epsilon=0.05,
        hard_negative_sampling_fraction=0.0,
        hard_negative_min_importance=0.0,
        recent_sampling_fraction=0.0,
        recent_window_fraction=0.25,
        quality_sampling_fraction=0.0,
        quality_min_importance=0.0,
        quality_value_bonus=0.0,
        value_balanced_sampling_fraction=0.0,
        value_balance_epsilon=None,
        weighted_sampling_power=1.0,
        resize_preserve_decisive_fraction=0.0,
        resize_preserve_decisive_min_count=0,
    ):
        self.max_size = int(max_size)
        self.max_policy_targets = max(1, int(max_policy_targets))
        self.use_fp16 = bool(use_fp16)
        self.decisive_sampling_fraction = max(0.0, min(1.0, float(decisive_sampling_fraction)))
        self.decisive_value_epsilon = max(0.0, float(decisive_value_epsilon))
        self.hard_negative_sampling_fraction = max(0.0, min(1.0, float(hard_negative_sampling_fraction)))
        self.hard_negative_min_importance = max(0.0, float(hard_negative_min_importance))
        self.recent_sampling_fraction = max(0.0, min(1.0, float(recent_sampling_fraction)))
        self.recent_window_fraction = max(0.01, min(1.0, float(recent_window_fraction)))
        self.quality_sampling_fraction = max(0.0, min(1.0, float(quality_sampling_fraction)))
        self.quality_min_importance = max(0.0, float(quality_min_importance))
        self.quality_value_bonus = max(0.0, float(quality_value_bonus))
        self.value_balanced_sampling_fraction = max(
            0.0,
            min(1.0, float(value_balanced_sampling_fraction)),
        )
        self.value_balance_epsilon = (
            self.decisive_value_epsilon
            if value_balance_epsilon is None
            else max(0.0, float(value_balance_epsilon))
        )
        self.weighted_sampling_power = max(0.05, min(2.0, float(weighted_sampling_power)))
        self.resize_preserve_decisive_fraction = max(
            0.0,
            min(1.0, float(resize_preserve_decisive_fraction)),
        )
        self.resize_preserve_decisive_min_count = max(0, int(resize_preserve_decisive_min_count))
        self.size = 0
        self.position = 0

        self._boards = None
        self._values = None
        self._policy_indices = None
        self._policy_values = None
        self._policy_lengths = None
        self._importance = None
        self._policy_sample_weights = None
        self._scratch = {}

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
        self._importance = torch.zeros((self.max_size,), dtype=torch.float32)
        self._policy_sample_weights = torch.ones((self.max_size,), dtype=torch.float32)

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
            "policy_sample_weights": torch.ones((batch_size,), dtype=torch.float32),
            "arange": torch.arange(max_len, dtype=torch.int16),
        }
        self._scratch[key] = scratch
        return scratch

    def _normalize_position(self, position):
        if len(position) < 4:
            raise ValueError("Replay position must contain board, policy indices, policy values, and value target.")
        board, policy_indices, policy_values, value = position[:4]
        importance = float(position[4]) if len(position) > 4 else 0.0
        policy_weight = float(position[5]) if len(position) > 5 else 1.0
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
        return board, policy_indices, policy_values, value, importance, policy_weight

    def _store_at_slot(self, slot, position):
        board, policy_indices, policy_values, value, importance, policy_weight = self._normalize_position(position)

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
        self._importance[slot] = float(importance)
        self._policy_sample_weights[slot] = float(policy_weight)

    def _build_batch_from_indices(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        batch_size = int(idx.numel())
        lengths = self._policy_lengths[idx].to(dtype=torch.int16)
        max_len = int(lengths.max().item()) if batch_size > 0 else 0

        scratch = self._get_scratch_batch(batch_size, max_len)
        boards = scratch["boards"]
        values = scratch["values"]
        policy_sample_weights = scratch["policy_sample_weights"]
        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])
        policy_sample_weights.copy_(self._policy_sample_weights[idx])

        if max_len <= 0:
            policy_indices = scratch["policy_indices"][:, :0]
            policy_values = scratch["policy_values"][:, :0]
            policy_mask = scratch["policy_mask"][:, :0]
            return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights

        policy_indices = scratch["policy_indices"]
        policy_values = scratch["policy_values"]
        policy_mask = scratch["policy_mask"]

        policy_indices.copy_(self._policy_indices[idx, :max_len])
        policy_values.copy_(self._policy_values[idx, :max_len])
        policy_mask.copy_(scratch["arange"].unsqueeze(0) < lengths.unsqueeze(1))
        return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights

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

        max_len = int(policy_indices.shape[1]) if policy_indices.dim() == 2 else 0
        if max_len > self.max_policy_targets:
            max_len = self.max_policy_targets
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
            self._importance[dst_slice].copy_(importance_scores[src_slice])
            self._policy_sample_weights[dst_slice].copy_(policy_weights[src_slice])

            self.position = (dst_start + count) % self.max_size
            self.size = min(self.size + count, self.max_size)
            remaining -= count
            src_start = src_end

    def _sample_indices_uniform(self, batch_size):
        return np.random.choice(self.size, batch_size, replace=False)

    def _take_evenly_spaced_positions(self, ordered_positions, target_count):
        ordered_positions = np.asarray(ordered_positions, dtype=np.int64)
        target_count = int(target_count)
        if target_count <= 0 or ordered_positions.size <= 0:
            return np.empty(0, dtype=np.int64)
        if ordered_positions.size <= target_count:
            return ordered_positions
        raw = np.linspace(0, ordered_positions.size - 1, num=target_count)
        chosen = []
        seen = set()
        for pos in raw:
            idx = int(round(float(pos)))
            idx = max(0, min(idx, int(ordered_positions.size) - 1))
            if idx in seen:
                continue
            seen.add(idx)
            chosen.append(int(ordered_positions[idx]))
        if len(chosen) < target_count:
            for idx in range(int(ordered_positions.size)):
                if idx in seen:
                    continue
                seen.add(idx)
                chosen.append(int(ordered_positions[idx]))
                if len(chosen) >= target_count:
                    break
        return np.asarray(chosen[:target_count], dtype=np.int64)

    def _select_resize_keep_indices(self, ordered_indices, keep_size):
        keep_size = max(0, int(keep_size))
        if keep_size <= 0:
            return np.empty(0, dtype=np.int64)
        total_size = int(ordered_indices.size)
        if total_size <= keep_size:
            return ordered_indices[-keep_size:]

        keep_positions = np.arange(total_size - keep_size, total_size, dtype=np.int64)
        if self.resize_preserve_decisive_fraction <= 0.0:
            return ordered_indices[keep_positions]

        ordered_values = self._values[torch.as_tensor(ordered_indices, dtype=torch.long)].reshape(-1)
        decisive_mask = (
            torch.abs(ordered_values.float()) > self.decisive_value_epsilon
        ).cpu().numpy().astype(bool)

        current_decisive_positions = keep_positions[decisive_mask[keep_positions]]
        target_decisive = max(
            self.resize_preserve_decisive_min_count,
            int(round(keep_size * self.resize_preserve_decisive_fraction)),
        )
        target_decisive = max(int(current_decisive_positions.size), target_decisive)
        target_decisive = min(keep_size, int(decisive_mask.sum()), target_decisive)

        extra_needed = target_decisive - int(current_decisive_positions.size)
        if extra_needed <= 0:
            return ordered_indices[keep_positions]

        dropped_history_positions = np.arange(0, total_size - keep_size, dtype=np.int64)
        older_decisive_positions = dropped_history_positions[decisive_mask[dropped_history_positions]]
        replaceable_keep_positions = keep_positions[~decisive_mask[keep_positions]]
        replace_count = min(
            extra_needed,
            int(older_decisive_positions.size),
            int(replaceable_keep_positions.size),
        )
        if replace_count <= 0:
            return ordered_indices[keep_positions]

        preserved_positions = self._take_evenly_spaced_positions(older_decisive_positions, replace_count)
        dropped_keep_positions = replaceable_keep_positions[:replace_count]
        keep_positions = np.setdiff1d(keep_positions, dropped_keep_positions, assume_unique=False)
        keep_positions = np.concatenate([keep_positions, preserved_positions])
        keep_positions.sort()
        return ordered_indices[keep_positions]

    def _sample_without_replacement(self, indices, take, weights=None):
        indices = np.asarray(indices, dtype=np.int64)
        take = int(take)
        if take <= 0 or indices.size <= 0:
            return np.empty(0, dtype=np.int64)
        if indices.size <= take:
            return indices.copy()

        probs = None
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if weights.size == indices.size:
                weights = np.clip(weights, 0.0, None)
                if self.weighted_sampling_power != 1.0:
                    weights = np.power(weights, self.weighted_sampling_power)
                total = float(weights.sum())
                if total > 0.0 and np.isfinite(total):
                    probs = weights / total

        return np.random.choice(indices, take, replace=False, p=probs)

    def _append_selected(self, chosen_parts, selected):
        selected = np.asarray(selected, dtype=np.int64)
        if selected.size <= 0:
            chosen = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64)
            return chosen
        chosen_parts.append(selected)
        return np.concatenate(chosen_parts)

    def _sample_value_balanced_indices(self, all_indices, chosen, values_np, target_count):
        target_count = int(target_count)
        if target_count <= 0:
            return np.empty(0, dtype=np.int64)

        eps = float(self.value_balance_epsilon)
        buckets = [
            all_indices[values_np > eps],
            all_indices[np.abs(values_np) <= eps],
            all_indices[values_np < -eps],
        ]
        buckets = [
            np.setdiff1d(bucket, chosen, assume_unique=False)
            for bucket in buckets
        ]

        selected_parts = []
        selected = np.empty(0, dtype=np.int64)
        base_take = max(1, target_count // 3)
        bucket_order = sorted(range(3), key=lambda idx: int(buckets[idx].size))

        for bucket_idx in bucket_order:
            if selected.size >= target_count:
                break
            bucket = buckets[bucket_idx]
            take = min(int(bucket.size), base_take, target_count - int(selected.size))
            picked = self._sample_without_replacement(bucket, take)
            if picked.size > 0:
                selected_parts.append(picked)
                selected = np.concatenate(selected_parts)

        if selected.size < target_count:
            remaining_pool = np.setdiff1d(
                np.concatenate(buckets) if buckets else np.empty(0, dtype=np.int64),
                selected,
                assume_unique=False,
            )
            extra = self._sample_without_replacement(
                remaining_pool,
                min(target_count - int(selected.size), int(remaining_pool.size)),
            )
            if extra.size > 0:
                selected_parts.append(extra)
                selected = np.concatenate(selected_parts)

        return selected[:target_count]

    def _sample_indices_with_biases(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        all_indices = np.arange(self.size, dtype=np.int64)
        chosen_parts = []
        chosen = np.empty(0, dtype=np.int64)
        values = self._values[:self.size].reshape(-1)
        values_np = values.float().cpu().numpy()

        def _remaining_slots():
            return max(0, int(batch_size) - int(chosen.size))

        if self.value_balanced_sampling_fraction > 0.0:
            balanced_take = min(
                _remaining_slots(),
                int(round(batch_size * self.value_balanced_sampling_fraction)),
            )
            balanced_selected = self._sample_value_balanced_indices(
                all_indices,
                chosen,
                values_np,
                balanced_take,
            )
            chosen = self._append_selected(chosen_parts, balanced_selected)

        if self.recent_sampling_fraction > 0.0:
            ordered_indices = self._ordered_indices_oldest_to_newest()
            recent_window = max(1, int(round(float(self.size) * self.recent_window_fraction)))
            recent_pool = np.setdiff1d(ordered_indices[-recent_window:], chosen, assume_unique=False)
            recent_take = min(
                _remaining_slots(),
                int(round(batch_size * self.recent_sampling_fraction)),
            )
            recent_take = max(0, recent_take)
            if recent_pool.size > 0 and recent_take > 0:
                recent_selected = self._sample_without_replacement(recent_pool, recent_take)
                chosen = self._append_selected(chosen_parts, recent_selected)

        if self.hard_negative_sampling_fraction > 0.0:
            importance = self._importance[:self.size].cpu().numpy()
            important_mask = importance >= self.hard_negative_min_importance
            important_indices = np.setdiff1d(all_indices[important_mask], chosen, assume_unique=False)
            hard_take = min(
                _remaining_slots(),
                int(round(batch_size * self.hard_negative_sampling_fraction)),
            )
            hard_take = max(0, hard_take)
            if important_indices.size > 0 and hard_take > 0:
                hard_weights = importance[important_indices] - self.hard_negative_min_importance + 1e-3
                hard_selected = self._sample_without_replacement(
                    important_indices,
                    hard_take,
                    weights=hard_weights,
                )
                chosen = self._append_selected(chosen_parts, hard_selected)

        decisive_mask = torch.abs(values.float()) > self.decisive_value_epsilon
        decisive_indices = torch.nonzero(decisive_mask, as_tuple=False).reshape(-1).cpu().numpy()

        if self.quality_sampling_fraction > 0.0:
            importance = self._importance[:self.size].cpu().numpy()
            quality_base = np.maximum(0.0, importance - self.quality_min_importance)
            if self.quality_value_bonus > 0.0:
                quality_base = quality_base + self.quality_value_bonus * np.abs(values_np)
            quality_mask = quality_base > 0.0
            quality_indices = np.setdiff1d(all_indices[quality_mask], chosen, assume_unique=False)
            quality_take = min(
                _remaining_slots(),
                int(round(batch_size * self.quality_sampling_fraction)),
            )
            quality_take = max(0, quality_take)
            if quality_indices.size > 0 and quality_take > 0:
                quality_weights = quality_base[quality_indices] + 1e-3
                quality_selected = self._sample_without_replacement(
                    quality_indices,
                    quality_take,
                    weights=quality_weights,
                )
                chosen = self._append_selected(chosen_parts, quality_selected)

        if self.decisive_sampling_fraction > 0.0 and decisive_indices.size > 0:
            target_decisive = min(
                _remaining_slots(),
                int(round(batch_size * self.decisive_sampling_fraction)),
            )
            target_decisive = max(0, target_decisive)
            remaining_decisive_pool = np.setdiff1d(decisive_indices, chosen, assume_unique=False)
            decisive_take = min(int(remaining_decisive_pool.size), target_decisive)
            decisive_selected = self._sample_without_replacement(remaining_decisive_pool, decisive_take)
            chosen = self._append_selected(chosen_parts, decisive_selected)

        indices = np.concatenate(chosen_parts) if chosen_parts else np.empty(0, dtype=np.int64)
        if indices.size < batch_size:
            remaining_pool = np.setdiff1d(all_indices, indices, assume_unique=False)
            fill = self._sample_without_replacement(
                remaining_pool,
                min(int(batch_size) - int(indices.size), int(remaining_pool.size)),
            )
            if fill.size > 0:
                indices = np.concatenate([indices, fill])
        if indices.size > batch_size:
            indices = indices[:batch_size]
        np.random.shuffle(indices)
        return indices

    def sample(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        batch_size = max(1, min(int(batch_size), int(self.size)))
        indices = self._sample_indices_with_biases(batch_size)
        return self._build_batch_from_indices(indices)

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
        old_importance = self._importance
        old_policy_sample_weights = self._policy_sample_weights

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
        self._importance = torch.zeros((self.max_size,), dtype=old_importance.dtype)
        self._policy_sample_weights = torch.ones((self.max_size,), dtype=old_policy_sample_weights.dtype)

        if keep_size > 0:
            idx = torch.as_tensor(keep_indices, dtype=torch.long)
            self._boards[:keep_size].copy_(old_boards[idx])
            self._values[:keep_size].copy_(old_values[idx])
            self._policy_indices[:keep_size].copy_(old_policy_indices[idx])
            self._policy_values[:keep_size].copy_(old_policy_values[idx])
            self._policy_lengths[:keep_size].copy_(old_policy_lengths[idx])
            self._importance[:keep_size].copy_(old_importance[idx])
            self._policy_sample_weights[:keep_size].copy_(old_policy_sample_weights[idx])

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
        importance = self._importance[:self.size].float()
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
            "policy_target_len_mean": _safe_mean(policy_lengths),
            "policy_target_len_p90": _safe_quantile(policy_lengths, 0.90),
            "importance_mean": _safe_mean(importance),
            "importance_p90": _safe_quantile(importance, 0.90),
        })

        policy_values = self._policy_values[:self.size].float()
        if policy_values.numel() > 0:
            valid_mask = self._policy_indices[:self.size] >= 0
            probs = torch.where(valid_mask, torch.clamp(policy_values, min=0.0), torch.zeros_like(policy_values))
            entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
            top1 = probs.max(dim=1).values
            non_empty = policy_lengths > 0
            stats["policy_target_entropy_mean"] = _safe_mean(entropy[non_empty])
            stats["policy_target_top1_prob_mean"] = _safe_mean(top1[non_empty])
        else:
            stats["policy_target_entropy_mean"] = 0.0
            stats["policy_target_top1_prob_mean"] = 0.0

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
