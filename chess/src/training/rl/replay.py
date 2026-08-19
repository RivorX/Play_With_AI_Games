"""
Replay buffer for RL training.
"""

import numpy as np
import torch

from src.models.data.se_cnn_v9.helpers import MAX_LEGAL_MOVES
from src.mcts.q_delta import USEFUL_SEARCH_Q_DELTA_MIN


_DEFAULT_MAX_POLICY_TARGETS = 256
REPLAY_SOURCE_UNKNOWN = 0
REPLAY_SOURCE_LEARNER = 1
REPLAY_SOURCE_FROZEN_BEST = 2
# Source codes remain stable for replay/checkpoint compatibility.
REPLAY_SOURCE_CHAMPION = REPLAY_SOURCE_FROZEN_BEST
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
        self._root_q_targets = None
        self._search_changed_top = None
        self._search_q_deltas = None
        self._best_q_targets = None
        self._played_q_targets = None
        self._orig_q_targets = None
        self._policy_kld_targets = None
        self._search_visits = None
        self._fens = []
        self._history_fens = []
        self._scratch = {}
        self.current_iteration = 0
        self.last_sample_age_stats = {}
        self.last_sample_ages = np.empty(0, dtype=np.float32)
        self.total_overwritten_positions = 0
        self.total_resize_dropped_positions = 0

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
        self._root_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._search_changed_top = torch.zeros((self.max_size,), dtype=torch.bool)
        self._search_q_deltas = torch.full((self.max_size,), float("nan"), dtype=torch.float32)
        self._best_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._played_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._orig_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._policy_kld_targets = torch.full((self.max_size,), float("nan"), dtype=torch.float32)
        self._search_visits = torch.zeros((self.max_size,), dtype=torch.int32)
        self._fens = [None] * self.max_size
        self._history_fens = [None] * self.max_size

    def set_current_iteration(self, iteration):
        try:
            self.current_iteration = max(0, int(iteration))
        except Exception:
            self.current_iteration = 0

    def relabel_iteration_source(self, iteration, source_code):
        """Relabel active rows inserted by one self-play iteration."""
        if self.size <= 0 or self._source_codes is None or self._insertion_iterations is None:
            return 0
        active_size = min(int(self.size), int(self.max_size))
        iteration = int(iteration)
        mask = self._insertion_iterations[:active_size] == iteration
        changed = int(mask.sum().item())
        if changed > 0:
            self._source_codes[:active_size][mask] = int(source_code)
        return changed

    def clear(self):
        """Drop logical contents while retaining the allocated tensor storage."""
        self.size = 0
        self.position = 0
        self.last_sample_age_stats = {}
        self.last_sample_ages = np.empty(0, dtype=np.float32)
        if self._fens:
            self._fens[:] = [None] * len(self._fens)
        if self._history_fens:
            self._history_fens[:] = [None] * len(self._history_fens)

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
            "root_q_targets": torch.full((batch_size, 1), float("nan"), dtype=torch.float32),
            "search_changed_top": torch.zeros((batch_size,), dtype=torch.bool),
            "search_q_deltas": torch.full((batch_size,), float("nan"), dtype=torch.float32),
            "best_q_targets": torch.full((batch_size, 1), float("nan"), dtype=torch.float32),
            "played_q_targets": torch.full((batch_size, 1), float("nan"), dtype=torch.float32),
            "orig_q_targets": torch.full((batch_size, 1), float("nan"), dtype=torch.float32),
            "policy_kld_targets": torch.full((batch_size,), float("nan"), dtype=torch.float32),
            "search_visits": torch.zeros((batch_size,), dtype=torch.int32),
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
        fen = str(position[10]) if len(position) > 10 and position[10] else None
        root_q = float(position[11]) if len(position) > 11 and position[11] is not None else float("nan")
        history_fens = tuple(position[12] or ()) if len(position) > 12 else ()
        search_changed_top = bool(position[13]) if len(position) > 13 else False
        search_q_delta = float(position[14]) if len(position) > 14 and position[14] is not None else float("nan")
        best_q = float(position[15]) if len(position) > 15 and position[15] is not None else float("nan")
        played_q = float(position[16]) if len(position) > 16 and position[16] is not None else float("nan")
        orig_q = float(position[17]) if len(position) > 17 and position[17] is not None else float("nan")
        policy_kld = float(position[18]) if len(position) > 18 and position[18] is not None else float("nan")
        search_visits = int(position[19]) if len(position) > 19 and position[19] is not None else 0
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
        return (
            board, policy_indices, policy_values, value, importance, policy_weight,
            value_weight, source_code, moves_left, legal_indices, fen, root_q, history_fens,
            search_changed_top, search_q_delta,
            best_q, played_q, orig_q, policy_kld, search_visits,
        )

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
            fen,
            root_q,
            history_fens,
            search_changed_top,
            search_q_delta,
            best_q,
            played_q,
            orig_q,
            policy_kld,
            search_visits,
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
        self._root_q_targets[slot] = float(root_q)
        self._search_changed_top[slot] = bool(search_changed_top)
        self._search_q_deltas[slot] = float(search_q_delta)
        self._best_q_targets[slot] = float(best_q)
        self._played_q_targets[slot] = float(played_q)
        self._orig_q_targets[slot] = float(orig_q)
        self._policy_kld_targets[slot] = float(policy_kld)
        self._search_visits[slot] = max(0, int(search_visits))
        self._fens[slot] = fen
        self._history_fens[slot] = tuple(history_fens)

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
        root_q_targets = scratch["root_q_targets"]
        search_changed_top = scratch["search_changed_top"]
        search_q_deltas = scratch["search_q_deltas"]
        best_q_targets = scratch["best_q_targets"]
        played_q_targets = scratch["played_q_targets"]
        orig_q_targets = scratch["orig_q_targets"]
        policy_kld_targets = scratch["policy_kld_targets"]
        search_visits = scratch["search_visits"]
        boards.copy_(self._boards[idx])
        values.copy_(self._values[idx])
        policy_sample_weights.copy_(self._policy_sample_weights[idx])
        value_sample_weights.copy_(self._value_sample_weights[idx])
        moves_left.copy_(self._moves_left[idx])
        root_q_targets.copy_(self._root_q_targets[idx])
        search_changed_top.copy_(self._search_changed_top[idx])
        search_q_deltas.copy_(self._search_q_deltas[idx])
        best_q_targets.copy_(self._best_q_targets[idx])
        played_q_targets.copy_(self._played_q_targets[idx])
        orig_q_targets.copy_(self._orig_q_targets[idx])
        policy_kld_targets.copy_(self._policy_kld_targets[idx])
        search_visits.copy_(self._search_visits[idx])

        if max_len <= 0:
            policy_indices = scratch["policy_indices"][:, :0]
            policy_values = scratch["policy_values"][:, :0]
            policy_mask = scratch["policy_mask"][:, :0]
            legal_indices = scratch["legal_indices"][:, :max_legal_len]
            legal_mask = scratch["legal_mask"][:, :max_legal_len]
            if max_legal_len > 0:
                legal_indices.copy_(self._legal_indices[idx, :max_legal_len])
                legal_mask.copy_(scratch["legal_arange"].unsqueeze(0) < legal_lengths.unsqueeze(1))
            return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights, value_sample_weights, moves_left, legal_indices, legal_mask, root_q_targets, search_changed_top, search_q_deltas, best_q_targets, played_q_targets, orig_q_targets, policy_kld_targets, search_visits

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
        return boards, policy_indices, policy_values, policy_mask, values, policy_sample_weights, value_sample_weights, moves_left, legal_indices, legal_mask, root_q_targets, search_changed_top, search_q_deltas, best_q_targets, played_q_targets, orig_q_targets, policy_kld_targets, search_visits

    def add(self, position):
        board = position[0]
        self._ensure_storage_initialized(board)
        if self.size >= self.max_size:
            self.total_overwritten_positions += 1
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
        fens=None,
        root_q_targets=None,
        history_fens=None,
        search_changed_top=None,
        search_q_deltas=None,
        best_q_targets=None,
        played_q_targets=None,
        orig_q_targets=None,
        policy_kld_targets=None,
        search_visits=None,
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
        batch_size = int(boards.shape[0])
        if root_q_targets is None:
            root_q_targets = torch.full((batch_size, 1), float("nan"), dtype=torch.float32)
        else:
            root_q_targets = root_q_targets.reshape(-1, 1).to(dtype=torch.float32).contiguous()
        if search_changed_top is None:
            search_changed_top = torch.zeros((batch_size,), dtype=torch.bool)
        else:
            search_changed_top = search_changed_top.reshape(-1).to(dtype=torch.bool).contiguous()
        if search_q_deltas is None:
            search_q_deltas = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        else:
            search_q_deltas = search_q_deltas.reshape(-1).to(dtype=torch.float32).contiguous()
        if best_q_targets is None:
            best_q_targets = torch.full((batch_size, 1), float("nan"), dtype=torch.float32)
        else:
            best_q_targets = best_q_targets.reshape(-1, 1).to(dtype=torch.float32).contiguous()
        if played_q_targets is None:
            played_q_targets = torch.full((batch_size, 1), float("nan"), dtype=torch.float32)
        else:
            played_q_targets = played_q_targets.reshape(-1, 1).to(dtype=torch.float32).contiguous()
        if orig_q_targets is None:
            orig_q_targets = torch.full((batch_size, 1), float("nan"), dtype=torch.float32)
        else:
            orig_q_targets = orig_q_targets.reshape(-1, 1).to(dtype=torch.float32).contiguous()
        if policy_kld_targets is None:
            policy_kld_targets = torch.full((batch_size,), float("nan"), dtype=torch.float32)
        else:
            policy_kld_targets = policy_kld_targets.reshape(-1).to(dtype=torch.float32).contiguous()
        if search_visits is None:
            search_visits = torch.zeros((batch_size,), dtype=torch.int32)
        else:
            search_visits = search_visits.reshape(-1).to(dtype=torch.int32).contiguous()
        fens = list(fens or [None] * batch_size)
        history_fens = list(history_fens or [()] * batch_size)
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

        self.total_overwritten_positions += max(
            0,
            int(self.size) + batch_size - int(self.max_size),
        )
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
            self._root_q_targets[dst_slice].copy_(root_q_targets[src_slice])
            self._search_changed_top[dst_slice].copy_(search_changed_top[src_slice])
            self._search_q_deltas[dst_slice].copy_(search_q_deltas[src_slice])
            self._best_q_targets[dst_slice].copy_(best_q_targets[src_slice])
            self._played_q_targets[dst_slice].copy_(played_q_targets[src_slice])
            self._orig_q_targets[dst_slice].copy_(orig_q_targets[src_slice])
            self._policy_kld_targets[dst_slice].copy_(policy_kld_targets[src_slice])
            self._search_visits[dst_slice].copy_(search_visits[src_slice])
            for offset in range(count):
                src_idx = src_start + offset
                dst_idx = dst_start + offset
                self._fens[dst_idx] = str(fens[src_idx]) if src_idx < len(fens) and fens[src_idx] else None
                self._history_fens[dst_idx] = tuple(history_fens[src_idx] or ()) if src_idx < len(history_fens) else ()
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

    def select_iteration_indices(
        self,
        iteration,
        max_count=None,
        *,
        prefer_useful_search_corrections=False,
    ):
        """Return positions inserted by one self-play iteration.

        The optional preference is intentionally limited to the pinned champion
        reservoir. It reserves at most half of a capped selection for positions
        where search actually changed the prior top move to a measurably higher-Q
        move, then fills the remainder uniformly to preserve phase/diversity.
        """
        if self.size <= 0 or self._insertion_iterations is None:
            return np.empty(0, dtype=np.int64)
        inserted = self._insertion_iterations[:self.size].cpu().numpy()
        indices = np.flatnonzero(inserted == int(iteration)).astype(np.int64, copy=False)
        if max_count is not None and indices.size > int(max_count):
            max_count = max(0, int(max_count))
            if max_count <= 0:
                return np.empty(0, dtype=np.int64)
            if prefer_useful_search_corrections:
                idx_tensor = torch.as_tensor(indices, dtype=torch.long)
                changed = self._search_changed_top[idx_tensor].cpu().numpy().astype(bool, copy=False)
                q_delta = self._search_q_deltas[idx_tensor].cpu().numpy()
                useful_mask = changed & np.isfinite(q_delta) & (q_delta > 0.02)
                useful = indices[useful_mask]
                useful_quota = min(useful.size, int(round(0.5 * max_count)))
                selected_useful = (
                    np.random.choice(useful, useful_quota, replace=False).astype(np.int64)
                    if useful_quota > 0 else np.empty(0, dtype=np.int64)
                )
                remaining_pool = indices[~np.isin(indices, selected_useful, assume_unique=False)]
                remaining_count = max_count - selected_useful.size
                selected_other = np.random.choice(
                    remaining_pool,
                    remaining_count,
                    replace=False,
                ).astype(np.int64)
                indices = np.concatenate((selected_useful, selected_other))
                np.random.shuffle(indices)
            else:
                indices = np.random.choice(indices, max_count, replace=False).astype(np.int64)
        return indices

    def copy_indices_to(self, target, indices, source_code=None, chunk_size=4096):
        """Copy selected rows into another ReplayBuffer without Python row decoding."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size <= 0:
            return 0
        copied = 0
        chunk_size = max(1, int(chunk_size))
        for start in range(0, int(indices.size), chunk_size):
            chunk = indices[start:start + chunk_size]
            idx = torch.as_tensor(chunk, dtype=torch.long)
            source_codes = self._source_codes[idx]
            if source_code is not None:
                source_codes = torch.full(
                    (int(idx.numel()),), int(source_code), dtype=torch.int8,
                )
            target.add_packed_batch(
                self._boards[idx],
                self._policy_indices[idx],
                self._policy_values[idx],
                self._policy_lengths[idx],
                self._values[idx],
                importance_scores=self._importance[idx],
                policy_weights=self._policy_sample_weights[idx],
                value_weights=self._value_sample_weights[idx],
                moves_left=self._moves_left[idx],
                legal_indices=self._legal_indices[idx],
                legal_lengths=self._legal_lengths[idx],
                source_codes=source_codes,
                fens=[self._fens[int(row)] for row in chunk],
                root_q_targets=self._root_q_targets[idx],
                history_fens=[self._history_fens[int(row)] for row in chunk],
                search_changed_top=self._search_changed_top[idx],
                search_q_deltas=self._search_q_deltas[idx],
                best_q_targets=self._best_q_targets[idx],
                played_q_targets=self._played_q_targets[idx],
                orig_q_targets=self._orig_q_targets[idx],
                policy_kld_targets=self._policy_kld_targets[idx],
                search_visits=self._search_visits[idx],
            )
            copied += int(idx.numel())
        return copied

    def sample_from_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size <= 0:
            raise ValueError("Cannot build an empty replay batch.")
        self._record_sample_age_stats(indices)
        return self._build_batch_from_indices(indices)

    def sample(self, batch_size):
        return self.sample_from_indices(self.select_indices(batch_size))

    def select_policy_correction_audit_indices(
        self,
        candidate_indices,
        max_count,
        *,
        seed=0,
    ):
        """Select a deterministic cohort of useful corrections seen by training."""
        candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        if candidates.size <= 0 or self.size <= 0:
            return np.empty(0, dtype=np.int64)
        candidates = np.unique(candidates[(candidates >= 0) & (candidates < self.size)])
        if candidates.size <= 0:
            return candidates

        idx = torch.as_tensor(candidates, dtype=torch.long)
        eligible = (
            (self._policy_lengths[idx] > 0)
            & (self._legal_lengths[idx] > 0)
            & (self._policy_sample_weights[idx] > 0.0)
            & self._search_changed_top[idx]
            & torch.isfinite(self._search_q_deltas[idx])
            & (self._search_q_deltas[idx] > USEFUL_SEARCH_Q_DELTA_MIN)
        ).cpu().numpy()
        selected = candidates[eligible]
        max_count = max(0, int(max_count))
        if max_count <= 0 or selected.size <= max_count:
            return selected
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(selected, max_count, replace=False)).astype(np.int64)

    def select_hard_position_indices(self, sample_size, min_age=0):
        """Select reconstructable positions with emphasis on unresolved search corrections."""
        if self.size <= 0 or not self._fens:
            return np.empty(0, dtype=np.int64)
        eligible = np.asarray(
            [idx for idx in range(self.size) if self._fens[idx]],
            dtype=np.int64,
        )
        if eligible.size <= 0:
            return eligible
        if self._insertion_iterations is not None:
            inserted = self._insertion_iterations[:self.size].cpu().numpy()
            ages = np.maximum(0, int(self.current_iteration) - inserted)
            eligible = eligible[ages[eligible] >= max(0, int(min_age))]
        if eligible.size <= 0:
            return eligible
        sample_size = min(max(1, int(sample_size)), int(eligible.size))
        importance = self._importance[:self.size].float().cpu().numpy()
        changed = self._search_changed_top[:self.size].cpu().numpy()
        q_deltas = self._search_q_deltas[:self.size].float().cpu().numpy()
        correction_strength = np.where(
            changed & np.isfinite(q_deltas) & (q_deltas > USEFUL_SEARCH_Q_DELTA_MIN),
            np.clip(q_deltas, 0.0, 0.50) / 0.20,
            0.0,
        )
        # Reanalyse exists primarily to refresh decisions where search corrected
        # the policy. Keep importance for tactical diversity, but make a useful
        # correction several times more likely than an equally important row.
        weights = np.square(np.maximum(0.0, importance[eligible]) + 0.10)
        weights *= 1.0 + 3.0 * np.clip(correction_strength[eligible], 0.0, 1.0)
        weights /= max(1e-12, float(weights.sum()))
        return np.random.choice(eligible, sample_size, replace=False, p=weights).astype(np.int64)

    def hard_positions_for_indices(self, indices):
        result = []
        for idx in np.asarray(indices, dtype=np.int64).tolist():
            result.append({
                "fen": self._fens[int(idx)],
                "history_fens": list(self._history_fens[int(idx)] or ()),
                "importance": float(self._importance[int(idx)].item()),
            })
        return result

    def update_reanalyzed_targets(
        self,
        indices,
        policy_targets,
        root_q_values,
        policy_mix=0.60,
        *,
        policy_weight_values=None,
        search_changed_top_values=None,
        search_q_delta_values=None,
        best_q_values=None,
        orig_q_values=None,
        policy_kld_values=None,
        search_visit_values=None,
    ):
        """Refresh policy and auxiliary search-value targets without changing game outcomes."""
        updated = 0
        mix = max(0.0, min(1.0, float(policy_mix)))
        for row, idx in enumerate(np.asarray(indices, dtype=np.int64).tolist()):
            if idx < 0 or idx >= self.size or row >= len(policy_targets):
                continue
            policy_indices, policy_values = policy_targets[row]
            policy_indices = torch.as_tensor(policy_indices, dtype=torch.int16).reshape(-1)
            policy_values = torch.as_tensor(policy_values, dtype=torch.float32).reshape(-1)
            count = min(policy_indices.numel(), policy_values.numel(), self.max_policy_targets)
            if count <= 0:
                continue
            old_count = int(self._policy_lengths[idx].item())
            merged = {}
            if mix < 1.0 and old_count > 0:
                for move_idx, probability in zip(
                    self._policy_indices[idx, :old_count].tolist(),
                    self._policy_values[idx, :old_count].float().tolist(),
                ):
                    merged[int(move_idx)] = merged.get(int(move_idx), 0.0) + (1.0 - mix) * float(probability)
            for move_idx, probability in zip(policy_indices[:count].tolist(), policy_values[:count].tolist()):
                merged[int(move_idx)] = merged.get(int(move_idx), 0.0) + mix * float(probability)
            ordered = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:self.max_policy_targets]
            mass = sum(max(0.0, value) for _, value in ordered)
            if mass <= 0.0:
                continue
            self._policy_indices[idx].fill_(-1)
            self._policy_values[idx].zero_()
            for col, (move_idx, probability) in enumerate(ordered):
                self._policy_indices[idx, col] = int(move_idx)
                self._policy_values[idx, col] = float(max(0.0, probability) / mass)
            self._policy_lengths[idx] = len(ordered)
            # A full replacement retains the new search decision and its
            # confidence. A mixed target has no unambiguous winner relationship,
            # so it deliberately drops correction metadata.
            full_replacement = mix >= 1.0 - 1e-8
            if (
                full_replacement
                and policy_weight_values is not None
                and row < len(policy_weight_values)
            ):
                self._policy_sample_weights[idx] = max(
                    0.0, float(policy_weight_values[row])
                )
            else:
                self._policy_sample_weights[idx] = 1.0
            if (
                full_replacement
                and search_changed_top_values is not None
                and row < len(search_changed_top_values)
            ):
                self._search_changed_top[idx] = bool(search_changed_top_values[row])
            else:
                self._search_changed_top[idx] = False
            if (
                full_replacement
                and search_q_delta_values is not None
                and row < len(search_q_delta_values)
            ):
                self._search_q_deltas[idx] = float(search_q_delta_values[row])
            else:
                self._search_q_deltas[idx] = float("nan")
            self._played_q_targets[idx] = float("nan")
            self._orig_q_targets[idx] = float("nan")
            self._policy_kld_targets[idx] = float("nan")
            self._search_visits[idx] = 0
            if row < len(root_q_values):
                self._root_q_targets[idx] = float(root_q_values[row])
            if best_q_values is not None and row < len(best_q_values):
                self._best_q_targets[idx] = float(best_q_values[row])
            else:
                self._best_q_targets[idx] = float("nan")
            if orig_q_values is not None and row < len(orig_q_values):
                self._orig_q_targets[idx] = float(orig_q_values[row])
            if policy_kld_values is not None and row < len(policy_kld_values):
                self._policy_kld_targets[idx] = float(policy_kld_values[row])
            if search_visit_values is not None and row < len(search_visit_values):
                self._search_visits[idx] = max(0, int(search_visit_values[row]))
            updated += 1
        return updated

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

        old_size = int(self.size)
        keep_size = min(old_size, new_max_size)
        self.total_resize_dropped_positions += max(0, old_size - keep_size)
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
        old_root_q_targets = self._root_q_targets
        old_search_changed_top = self._search_changed_top
        old_search_q_deltas = self._search_q_deltas
        old_best_q_targets = self._best_q_targets
        old_played_q_targets = self._played_q_targets
        old_orig_q_targets = self._orig_q_targets
        old_policy_kld_targets = self._policy_kld_targets
        old_search_visits = self._search_visits
        old_fens = self._fens
        old_history_fens = self._history_fens

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
        self._root_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._search_changed_top = torch.zeros((self.max_size,), dtype=torch.bool)
        self._search_q_deltas = torch.full((self.max_size,), float("nan"), dtype=torch.float32)
        self._best_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._played_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._orig_q_targets = torch.full((self.max_size, 1), float("nan"), dtype=torch.float32)
        self._policy_kld_targets = torch.full((self.max_size,), float("nan"), dtype=torch.float32)
        self._search_visits = torch.zeros((self.max_size,), dtype=torch.int32)
        self._fens = [None] * self.max_size
        self._history_fens = [None] * self.max_size

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
            self._root_q_targets[:keep_size].copy_(old_root_q_targets[idx])
            self._search_changed_top[:keep_size].copy_(old_search_changed_top[idx])
            self._search_q_deltas[:keep_size].copy_(old_search_q_deltas[idx])
            self._best_q_targets[:keep_size].copy_(old_best_q_targets[idx])
            self._played_q_targets[:keep_size].copy_(old_played_q_targets[idx])
            self._orig_q_targets[:keep_size].copy_(old_orig_q_targets[idx])
            self._policy_kld_targets[:keep_size].copy_(old_policy_kld_targets[idx])
            self._search_visits[:keep_size].copy_(old_search_visits[idx])
            for dst_idx, src_idx in enumerate(keep_indices.tolist()):
                self._fens[dst_idx] = old_fens[int(src_idx)]
                self._history_fens[dst_idx] = old_history_fens[int(src_idx)]

        self.size = keep_size
        self.position = 0 if keep_size >= self.max_size else keep_size
        self._scratch = {}
        return True

    def storage_rotation_stats(self):
        """Return exact FIFO/resize losses accumulated by this buffer."""
        return {
            "overwritten_positions": int(self.total_overwritten_positions),
            "resize_dropped_positions": int(self.total_resize_dropped_positions),
            "evicted_positions": int(
                self.total_overwritten_positions + self.total_resize_dropped_positions
            ),
        }

    def quality_stats(self, recent_window_fraction=None):
        """Return cheap aggregate stats describing replay target quality."""
        stats = {
            "size": int(self.size),
            "capacity": int(self.max_size),
            "fill_rate": float(self.size) / float(max(1, self.max_size)),
            **self.storage_rotation_stats(),
        }
        if self.size <= 0 or self._boards is None:
            return stats

        values = self._values[:self.size].reshape(-1).float()
        policy_lengths = self._policy_lengths[:self.size].to(dtype=torch.float32)
        policy_weights = self._policy_sample_weights[:self.size].float()
        value_weights = self._value_sample_weights[:self.size].float()
        importance = self._importance[:self.size].float()
        root_q_targets = self._root_q_targets[:self.size].reshape(-1).float()
        root_q_mask = torch.isfinite(root_q_targets)
        best_q_targets = self._best_q_targets[:self.size].reshape(-1).float()
        best_q_mask = torch.isfinite(best_q_targets)
        played_q_targets = self._played_q_targets[:self.size].reshape(-1).float()
        played_q_mask = torch.isfinite(played_q_targets)
        orig_q_targets = self._orig_q_targets[:self.size].reshape(-1).float()
        orig_q_mask = torch.isfinite(orig_q_targets)
        policy_kld_targets = self._policy_kld_targets[:self.size].float()
        policy_kld_mask = torch.isfinite(policy_kld_targets)
        deblunder_mask = value_weights < 0.999
        search_changed_top = self._search_changed_top[:self.size]
        search_q_deltas = self._search_q_deltas[:self.size].float()
        policy_rows_mask = policy_weights > 0.0
        correction_mask = (
            policy_rows_mask
            & search_changed_top
            & torch.isfinite(search_q_deltas)
            & (search_q_deltas > USEFUL_SEARCH_Q_DELTA_MIN)
        )
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
            "root_q_coverage": float(root_q_mask.float().mean().item()),
            "root_q_mean": _safe_mean(root_q_targets[root_q_mask]),
            "root_q_std": _safe_std(root_q_targets[root_q_mask]),
            "best_q_coverage": float(best_q_mask.float().mean().item()),
            "best_q_mean": _safe_mean(best_q_targets[best_q_mask]),
            "played_q_coverage": float(played_q_mask.float().mean().item()),
            "orig_q_coverage": float(orig_q_mask.float().mean().item()),
            "policy_kld_coverage": float(policy_kld_mask.float().mean().item()),
            "policy_kld_mean": _safe_mean(policy_kld_targets[policy_kld_mask]),
            "deblunder_value_fraction": float(deblunder_mask.float().mean().item()),
            "value_weight_p10": _safe_quantile(value_weights, 0.10),
            "value_weight_low_fraction": float((value_weights < 0.75).float().mean().item()),
            "policy_target_len_mean": _safe_mean(policy_lengths),
            "policy_target_len_p90": _safe_quantile(policy_lengths, 0.90),
            "importance_mean": _safe_mean(importance),
            "importance_p90": _safe_quantile(importance, 0.90),
            "policy_correction_fraction": (
                float(correction_mask.sum().item()) / float(max(1, policy_rows_mask.sum().item()))
            ),
            "policy_correction_q_delta_mean": _safe_mean(search_q_deltas[correction_mask]),
            "policy_correction_weight_share": (
                float(policy_weights[correction_mask].sum().item())
                / max(1e-8, float(policy_weights.sum().item()))
            ),
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
