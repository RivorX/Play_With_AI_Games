"""
🎯 BATCH SELF-PLAY WITH FULL MCTS - AlphaZero Style
Plays multiple games using MCTS for move selection

✅ CORRECT IMPLEMENTATION:
- Uses MCTS for all move selections (not raw network)
- Training targets = MCTS visit distributions
- High-quality training data
"""

import torch
import chess
import numpy as np
import math
import pickle
from src.data import board_to_tensor, move_to_index


_EMPTY_HISTORY_TENSOR = np.zeros((16, 8, 8), dtype=np.float32)


def _copy_board_fast(board):
    """Copy board state without move stack/history baggage."""
    try:
        return board.copy(stack=False)
    except TypeError:
        return board.copy()


def _board_position_key(board):
    """
    Fast board identity for tree reuse.

    Prefer python-chess internal transposition key when available; fallback to
    full FEN only if needed.
    """
    key_fn = getattr(board, "_transposition_key", None)
    if callable(key_fn):
        try:
            return key_fn()
        except Exception:
            pass
    return board.fen()


_SELFPLAY_OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"),
    ("e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5"),
    ("d2d4", "d7d5", "g1f3", "g8f6", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"),
    ("g1f3", "d7d5", "d2d4", "g8f6", "c2c4", "e7e6"),
    ("g1f3", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"),
    ("e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"),
    ("d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6"),
    ("e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"),
)

# Keep eval fixed openings relatively balanced, but let self-play sample from
# a broader, sharper pool so RL escapes quiet draw-heavy shells more often.
_SELFPLAY_SHARP_OPENING_LINES = (
    ("e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5"),
    ("e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "d7d5"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f1c4"),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b4", "c2c3"),
    ("e2e4", "e7e5", "d2d4", "e5d4", "c2c3", "d4c3", "f1c4"),
    ("e2e4", "c7c5", "d2d4", "c5d4", "c2c3", "d4c3", "b1c3"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"),
    ("e2e4", "c7c5", "b1c3", "b8c6", "f2f4", "g7g6", "g1f3", "f8g7"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4"),
    ("e2e4", "d7d5", "e4d5", "g8f6", "c2c4", "e7e6", "b1c3", "e6d5"),
    ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6"),
    ("d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f4"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4"),
    ("d2d4", "f7f5", "e2e4", "f5e4", "b1c3", "g8f6", "c1g5"),
    ("d2d4", "g8f6", "c2c4", "e7e5", "d4e5", "f6g4"),
)
_SELFPLAY_SELFPLAY_OPENING_LINES = _SELFPLAY_OPENING_LINES + _SELFPLAY_SHARP_OPENING_LINES


class MCTSEdgeStats:
    """Compact array-backed storage for all child-edge statistics of one node."""

    def __init__(self):
        self.moves = ()
        self.nodes = []
        self.priors = np.empty(0, dtype=np.float32)
        self.base_priors = np.empty(0, dtype=np.float32)
        self.visit_counts = np.empty(0, dtype=np.int32)
        self.total_counts = np.empty(0, dtype=np.float32)
        self.value_sums = np.empty(0, dtype=np.float32)
        self.virtual_losses = np.empty(0, dtype=np.int16)
        self.virtual_losses_f32 = np.empty(0, dtype=np.float32)
        self.explored_flags = np.empty(0, dtype=np.bool_)
        self.selection_scores = np.empty(0, dtype=np.float32)
        self.ucb_buffer = np.empty(0, dtype=np.float32)
        self._move_to_child = None

    def reset(self, legal_moves, legal_priors):
        legal_moves = tuple(legal_moves)
        legal_priors = np.asarray(legal_priors, dtype=np.float32)
        child_count = len(legal_moves)

        self.moves = legal_moves
        self.nodes = []
        self.priors = legal_priors.copy()
        self.base_priors = legal_priors.copy()
        self.visit_counts = np.zeros(child_count, dtype=np.int32)
        self.total_counts = np.zeros(child_count, dtype=np.float32)
        self.value_sums = np.zeros(child_count, dtype=np.float32)
        self.virtual_losses = np.zeros(child_count, dtype=np.int16)
        self.virtual_losses_f32 = np.zeros(child_count, dtype=np.float32)
        self.explored_flags = np.zeros(child_count, dtype=np.bool_)
        self.selection_scores = np.empty(child_count, dtype=np.float32)
        self.ucb_buffer = np.empty(child_count, dtype=np.float32)
        self._move_to_child = None

    def add_child(self, child):
        self.nodes.append(child)

    def iter_nodes(self):
        return zip(self.moves, self.nodes)

    def get_child_for_move(self, move):
        move_to_child = self._move_to_child
        if move_to_child is None:
            move_to_child = {child_move: child for child_move, child in zip(self.moves, self.nodes)}
            self._move_to_child = move_to_child
        return move_to_child.get(move)

    def visit_dict(self):
        return {
            move: int(visits)
            for move, visits in zip(self.moves, self.visit_counts)
        }

    def __len__(self):
        return len(self.nodes)


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, board=None, parent=None, move=None, prior=0.0, copy_board=True):
        if board is not None:
            self._board = _copy_board_fast(board) if copy_board else board
        else:
            self._board = None
        self.parent = parent
        self.move = move
        self.prior = prior
        self.base_prior = prior

        self.edges = MCTSEdgeStats()
        self.parent_edge_index = -1
        self._root_visit_count = 0
        self._root_value_sum = 0.0
        self.expanded = False
        self._root_virtual_loss = 0
        self._root_is_explored = False
        self._explored_prior_sum = 0.0

        self._position_key_cache = None
        self._is_game_over = None
        self._board_tensor = None
        self._legal_moves = None
        self._legal_indices = None

    @property
    def board(self):
        if self._board is None:
            self._board = _copy_board_fast(self.parent.board)
            self._board.push(self.move)
        return self._board

    @property
    def is_game_over(self):
        if self._is_game_over is None:
            # In self-play we do not want "claimable draw" states (threefold/50-move
            # claims) to look terminal to MCTS by default, otherwise the search
            # eagerly settles for sterile repetitions.
            self._is_game_over = self.board.is_game_over(claim_draw=False)
        return self._is_game_over

    def is_leaf(self):
        return not self.expanded

    def _has_parent_edge(self):
        return self.parent is not None and self.parent_edge_index >= 0

    @property
    def visit_count(self):
        if self._has_parent_edge():
            return int(self.parent.edges.visit_counts[int(self.parent_edge_index)])
        return int(self._root_visit_count)

    @visit_count.setter
    def visit_count(self, value):
        value = int(value)
        if self._has_parent_edge():
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
            edges.visit_counts[idx] = value
            edges.total_counts[idx] = float(value + int(edges.virtual_losses[idx]))
        else:
            self._root_visit_count = value

    @property
    def value_sum(self):
        if self._has_parent_edge():
            return float(self.parent.edges.value_sums[int(self.parent_edge_index)])
        return float(self._root_value_sum)

    @value_sum.setter
    def value_sum(self, value):
        value = float(value)
        if self._has_parent_edge():
            self.parent.edges.value_sums[int(self.parent_edge_index)] = value
        else:
            self._root_value_sum = value

    @property
    def virtual_loss(self):
        if self._has_parent_edge():
            return int(self.parent.edges.virtual_losses[int(self.parent_edge_index)])
        return int(self._root_virtual_loss)

    @virtual_loss.setter
    def virtual_loss(self, value):
        value = int(value)
        if self._has_parent_edge():
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
            edges.virtual_losses[idx] = value
            edges.virtual_losses_f32[idx] = float(value)
            edges.total_counts[idx] = float(int(edges.visit_counts[idx]) + value)
        else:
            self._root_virtual_loss = value

    @property
    def _is_explored(self):
        if self._has_parent_edge():
            return bool(self.parent.edges.explored_flags[int(self.parent_edge_index)])
        return bool(self._root_is_explored)

    @_is_explored.setter
    def _is_explored(self, value):
        value = bool(value)
        if self._has_parent_edge():
            self.parent.edges.explored_flags[int(self.parent_edge_index)] = value
        else:
            self._root_is_explored = value

    def _set_explored(self, explored):
        explored = bool(explored)
        if self._is_explored == explored:
            return
        if self.parent is not None:
            if explored:
                self.parent._explored_prior_sum += float(self.prior)
            else:
                self.parent._explored_prior_sum -= float(self.prior)
                if self.parent._explored_prior_sum < 0.0:
                    self.parent._explored_prior_sum = 0.0
        self._is_explored = explored

    def add_virtual_loss(self, n=1):
        was_explored = (self.visit_count + self.virtual_loss) > 0
        self.virtual_loss += n
        if not was_explored and (self.visit_count + self.virtual_loss) > 0:
            self._set_explored(True)

    def remove_virtual_loss(self, n=1):
        was_explored = (self.visit_count + self.virtual_loss) > 0
        self.virtual_loss -= n
        if was_explored and (self.visit_count + self.virtual_loss) <= 0:
            self._set_explored(False)

    def expand_children(self, legal_moves, legal_priors):
        self.edges.reset(legal_moves, legal_priors)

        for idx, (move, prior) in enumerate(zip(self.edges.moves, self.edges.priors)):
            child = MCTSNode(
                board=None,
                parent=self,
                move=move,
                prior=float(prior),
                copy_board=False,
            )
            child.base_prior = float(self.edges.base_priors[idx])
            child.parent_edge_index = idx
            self.edges.add_child(child)

        self.expanded = True

    def iter_child_nodes(self):
        return self.edges.iter_nodes()

    def get_child_for_move(self, move):
        return self.edges.get_child_for_move(move)

    def child_visit_dict(self):
        return self.edges.visit_dict()

    def get_position_key(self):
        if self._position_key_cache is None:
            self._position_key_cache = _board_position_key(self.board)
        return self._position_key_cache

    def get_legal_moves_and_indices(self):
        if self._legal_moves is None or self._legal_indices is None:
            legal_moves = tuple(self.board.legal_moves)
            legal_indices = np.fromiter(
                (move_to_index(move, self.board) for move in legal_moves),
                dtype=np.int32,
                count=len(legal_moves),
            )
            self._legal_moves = legal_moves
            self._legal_indices = legal_indices
        return self._legal_moves, self._legal_indices


def _build_sparse_policy_target_from_visits(visit_counts, board):
    """
    Convert MCTS visit counts into sparse policy targets.

    Returns:
        (indices, probs)
        indices: int16 tensor of action indices
        probs: float32 tensor of normalized visit probabilities
    """
    if not visit_counts:
        return (
            torch.empty(0, dtype=torch.int16),
            torch.empty(0, dtype=torch.float32),
        )

    moves = list(visit_counts.keys())
    indices = np.empty(len(moves), dtype=np.int16)
    visits = np.empty(len(moves), dtype=np.float32)

    for i, move in enumerate(moves):
        indices[i] = move_to_index(move, board)
        visits[i] = float(visit_counts[move])

    total_visits = float(visits.sum())
    if total_visits > 0:
        probs = visits / total_visits
    else:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)

    return (
        torch.from_numpy(indices),
        torch.from_numpy(probs.astype(np.float32, copy=False)),
    )


def _pack_positions_for_transfer(positions):
    """
    Pack self-play positions into batched tensors for queue transport.
    """
    if not positions:
        return None

    batch_size = len(positions)
    boards = torch.stack([pos[0] for pos in positions]).contiguous()
    values = torch.stack([pos[3] for pos in positions]).contiguous()
    max_len = max(int(pos[1].numel()) for pos in positions)

    policy_indices = torch.full((batch_size, max_len), -1, dtype=torch.int16)
    policy_values = torch.zeros((batch_size, max_len), dtype=torch.float32)
    policy_lengths = torch.zeros((batch_size,), dtype=torch.int16)

    for row_idx, pos in enumerate(positions):
        _, indices, probs, _ = pos[:4]
        count = int(indices.numel())
        if count <= 0:
            continue
        policy_indices[row_idx, :count] = indices.to(dtype=torch.int16)
        policy_values[row_idx, :count] = probs.to(dtype=torch.float32)
        policy_lengths[row_idx] = count

    return {
        'boards': boards,
        'policy_indices': policy_indices,
        'policy_values': policy_values,
        'policy_lengths': policy_lengths,
        'values': values,
        'num_positions': batch_size,
    }


def _move_selection_temperature(board, base_temperature, threshold_fullmoves):
    """
    Use exploratory sampling for the first N full moves of the game.
    """
    if threshold_fullmoves is None:
        return base_temperature
    return base_temperature if board.fullmove_number <= threshold_fullmoves else 0.01


def _select_move_from_visits_safe(visit_counts, temperature):
    """
    Select move from visit counts with numeric safeguards.
    """
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        return moves[int(np.argmax(visits))]

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        return moves[np.random.randint(len(moves))]

    probs = visits_temp / total
    probs = np.clip(probs, 0.0, 1.0)
    probs_sum = float(probs.sum())
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        return moves[np.random.randint(len(moves))]

    probs /= probs_sum
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


def select_move_by_visits(visit_counts, temperature=1.0):
    """Compatibility helper used by UI/eval code paths."""
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        best_idx = int(np.argmax(visits))
        return moves[best_idx], visits

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
    else:
        probs = visits_temp / total
        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
        else:
            probs = probs / probs_sum

    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx], probs


def _resolve_selfplay_max_moves(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('self_play_max_moves', 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _build_history_tensor_from_encoded(
    turn,
    encoded_history,
    history_count,
    history_positions,
    empty_history_tensor,
):
    """Build model input from cached history tensors without FEN parsing."""
    use_black_pov = (turn == chess.BLACK)
    available = len(encoded_history)
    if available <= 0:
        return empty_history_tensor

    current_idx = max(0, min(int(history_count), available - 1))
    current_encoded = encoded_history[current_idx]
    current_tensor = current_encoded[1] if use_black_pov else current_encoded[0]

    if history_positions <= 0:
        return current_tensor

    end = max(0, min(int(history_count), available))
    start = max(0, end - int(history_positions))
    history_slice = encoded_history[start:end]

    history_tensors = []
    for encoded in history_slice:
        history_tensors.append(encoded[1] if use_black_pov else encoded[0])

    pad_count = max(0, int(history_positions) - len(history_tensors))
    if pad_count:
        history_tensors = [empty_history_tensor] * pad_count + history_tensors

    history_tensors.append(current_tensor)
    return np.concatenate(history_tensors, axis=0)


# =============================================================================
# TRUE BATCH MCTS (MULTI-GAME) FOR BETTER GPU UTILIZATION
# =============================================================================

class MultiGameBatchMCTS:
    """
    Batch MCTS that searches multiple games in parallel and batches
    leaf evaluations across all games for higher GPU utilization.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.c_puct = config['reinforcement_learning']['mcts_c_puct']
        self.use_dynamic_c_puct = bool(
            config['reinforcement_learning'].get('mcts_dynamic_c_puct', True)
        )
        self.c_puct_base = float(
            config['reinforcement_learning'].get('mcts_c_puct_base', 19652)
        )
        self.c_puct_init = float(
            config['reinforcement_learning'].get('mcts_c_puct_init', self.c_puct)
        )
        c_puct_max_raw = config['reinforcement_learning'].get('mcts_c_puct_max', None)
        self.c_puct_max = float(c_puct_max_raw) if c_puct_max_raw is not None else None
        self.use_fpu = bool(config['reinforcement_learning'].get('mcts_use_fpu', True))
        self.fpu_reduction = float(
            config['reinforcement_learning'].get('mcts_fpu_reduction', 0.30)
        )
        self.fpu_absolute = config['reinforcement_learning'].get('mcts_fpu_absolute', None)
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)

        # History configuration (POV)
        self.history_positions = config['model'].get('history_positions', 0)

        # Tree reuse
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)

        # Dirichlet noise params
        self.dirichlet_alpha = config['reinforcement_learning'].get('mcts_dirichlet_alpha', 0.3)
        self.dirichlet_weight = config['reinforcement_learning'].get('mcts_dirichlet_weight', 0.0)

        # Inference optimization (AMP on GPU)
        self.use_amp = config.get('hardware', {}).get('use_amp', False) and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
        self._empty_history_tensor = np.zeros((16, 8, 8), dtype=np.float32)
        self._legal_index_scratch = {}

    @staticmethod
    def _is_encoded_history_entry(entry):
        return (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], np.ndarray)
            and isinstance(entry[1], np.ndarray)
        )

    def _encode_history_entry(self, board_or_fen):
        """
        Convert history item to cached tensors for both POVs:
        (white_pov_tensor, black_pov_tensor).
        """
        if self._is_encoded_history_entry(board_or_fen):
            return board_or_fen
        if isinstance(board_or_fen, str):
            board_obj = chess.Board(board_or_fen)
        else:
            board_obj = board_or_fen
        if not isinstance(board_obj, chess.Board):
            raise TypeError(f"Unsupported history entry type: {type(board_or_fen)}")

        return (
            board_to_tensor(board_obj, flip_perspective=False),
            board_to_tensor(board_obj, flip_perspective=True),
        )

    def _build_history_tensor(self, current_board, board_history, current_tensor=None):
        """
        Build tensor with history using POV-aware board_to_tensor.

        Args:
            current_board: chess.Board for current position
            board_history: list of previous boards (from real game history)
        """
        if self.history_positions == 0:
            if current_tensor is not None:
                return current_tensor
            return board_to_tensor(current_board)

        history_tensors = []
        use_black_pov = current_board.turn == chess.BLACK
        if board_history:
            history_boards = board_history[-self.history_positions:]
            for hist_entry in history_boards:
                encoded = self._encode_history_entry(hist_entry)
                hist_tensor = encoded[1] if use_black_pov else encoded[0]
                history_tensors.append(hist_tensor)

        pad_count = max(0, self.history_positions - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        if current_tensor is None:
            current_tensor = board_to_tensor(current_board)
        history_tensors.append(current_tensor)

        return np.concatenate(history_tensors, axis=0)

    @staticmethod
    def _current_tensor_for_node(node):
        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = board_to_tensor(node.board)
            node._board_tensor = cached
        return cached

    def _get_legal_index_scratch(self, batch_size, max_legal_count):
        key = (int(batch_size), int(max_legal_count))
        scratch = self._legal_index_scratch.get(key)
        if scratch is None:
            scratch = np.empty((batch_size, max_legal_count), dtype=np.int64)
            self._legal_index_scratch[key] = scratch
        return scratch

    def _select_child(self, node):
        """Select child with highest UCB score (optimized)"""
        if not node.edges.nodes:
            return None

        # Precalculate parent term to avoid doing it for every child
        parent_visits = node.visit_count + node.virtual_loss
        parent_sqrt = math.sqrt(parent_visits + 1)
        if self.use_dynamic_c_puct:
            c_puct = math.log((parent_visits + self.c_puct_base + 1.0) / self.c_puct_base) + self.c_puct_init
            if self.c_puct_max is not None:
                c_puct = min(c_puct, self.c_puct_max)
        else:
            c_puct = self.c_puct

        fpu_value = 0.0
        if self.use_fpu:
            if self.fpu_absolute is not None:
                fpu_value = float(self.fpu_absolute)
            elif parent_visits > 0:
                parent_q = (node.value_sum - node.virtual_loss) / parent_visits
                explored_prior = min(1.0, max(0.0, float(node._explored_prior_sum)))
                unexplored_prior = max(0.0, 1.0 - explored_prior)
                fpu_value = parent_q - self.fpu_reduction * math.sqrt(unexplored_prior)

        edges = node.edges
        cv = edges.total_counts
        q_values = edges.selection_scores
        explored_mask = edges.explored_flags
        q_values.fill(fpu_value)
        if np.any(explored_mask):
            q_values[explored_mask] = (
                edges.value_sums[explored_mask] - edges.virtual_losses_f32[explored_mask]
            ) / cv[explored_mask]

        u_values = edges.ucb_buffer
        np.multiply(edges.priors, c_puct * parent_sqrt, out=u_values)
        u_values /= (1.0 + cv)
        q_values += u_values
        best_idx = int(np.argmax(q_values))
        return edges.nodes[best_idx]

    def _apply_root_noise(self, node):
        """Apply fresh Dirichlet noise to an already expanded root node."""
        if self.dirichlet_weight <= 0 or not node.edges.nodes:
            return

        edges = node.edges
        children = edges.nodes
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        mix = self.dirichlet_weight

        for idx, (child, noise_value) in enumerate(zip(children, noise)):
            new_prior = (1.0 - mix) * float(edges.base_priors[idx]) + mix * float(noise_value)
            if child._is_explored:
                node._explored_prior_sum += float(new_prior - child.prior)
            edges.priors[idx] = new_prior
            child.prior = new_prior

    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def search_many(self, game_states, num_simulations, add_root_noise=False):
        """
        Run MCTS for multiple games and batch leaf evaluations across games.

        Args:
            game_states: list of dicts with keys: board, root, board_history
            num_simulations: simulations per game
        Returns:
            List[Dict[chess.Move, int]] visit counts for each game in order
        """
        if not game_states:
            return []

        # Initialize / reuse roots per game
        for gs in game_states:
            board = gs['board']
            root = gs.get('root', None)
            root_synced = bool(gs.get('_root_synced', False))

            if self.reuse_tree and root is not None:
                if root_synced:
                    gs['_root_synced'] = False
                else:
                    target_key = _board_position_key(board)
                    if root.get_position_key() == target_key:
                        pass
                    else:
                        for child in root.edges.nodes:
                            if child.get_position_key() == target_key:
                                _ = child.board  # Ensure board is instantiated
                                root = child
                                root.parent = None
                                root.parent_edge_index = -1
                                break
                        else:
                            root = MCTSNode(board)
            else:
                root = MCTSNode(board)
                gs['_root_synced'] = False

            gs['root'] = root
            if add_root_noise and root.expanded:
                self._apply_root_noise(root)
                gs['_needs_root_noise_on_expand'] = False
            else:
                gs['_needs_root_noise_on_expand'] = bool(add_root_noise and not root.expanded)

        remaining = [num_simulations] * len(game_states)
        total_remaining = num_simulations * len(game_states)
        game_ptr = 0

        while total_remaining > 0:
            # Cap per-game quota per round so that backprop happens before all
            # simulations of a single game are consumed  (fixes: when
            # eval_batch_size >= num_simulations, all sims land in one batch,
            # root is never traversed after expansion, children stay at 0 visits).
            max_remaining_any_game = max(remaining) if remaining else 1
            slots_per_game = max(1, min(
                self.eval_batch_size // max(1, len(game_states)),
                max_remaining_any_game // 4,
            ))
            batch_size = min(self.eval_batch_size, total_remaining,
                             slots_per_game * len(game_states))
            leaf_nodes = []
            search_paths = []
            leaf_game_indices = []
            selected_this_batch = [0] * len(game_states)

            for _ in range(batch_size):
                # Find next game with remaining sims
                found = False
                for _ in range(len(game_states)):
                    current_gs = game_states[game_ptr]
                    if (
                        remaining[game_ptr] > 0
                        and not (
                            selected_this_batch[game_ptr] > 0
                            and current_gs['root'] is not None
                            and not current_gs['root'].expanded
                        )
                    ):
                        found = True
                        break
                    game_ptr = (game_ptr + 1) % len(game_states)

                if not found:
                    break

                gs_idx = game_ptr
                gs = game_states[gs_idx]

                node = gs['root']
                search_path = [node]
                node.add_virtual_loss()

                while not node.is_leaf():
                    node = self._select_child(node)
                    node.add_virtual_loss()
                    search_path.append(node)

                search_paths.append(search_path)
                leaf_nodes.append(node)
                leaf_game_indices.append(gs_idx)
                selected_this_batch[gs_idx] += 1

                remaining[gs_idx] -= 1
                total_remaining -= 1
                game_ptr = (game_ptr + 1) % len(game_states)

            if not leaf_nodes:
                break

            values = self._batch_expand_and_evaluate(leaf_nodes, leaf_game_indices, game_states)

            for search_path, value in zip(search_paths, values):
                self._backpropagate(search_path, value)
                for node in search_path:
                    node.remove_virtual_loss()

        return [
            gs['root'].child_visit_dict()
            for gs in game_states
        ]

    def _batch_expand_and_evaluate(self, nodes, game_indices, game_states):
        """
        Batch expansion + evaluation for leaf nodes across games.
        """
        # The same leaf can appear multiple times in one batch.
        # Evaluate each unique node once and fan-out value to duplicates.
        node_occurrences = {}
        unique_entries = []
        for idx, node in enumerate(nodes):
            node_id = id(node)
            if node_id not in node_occurrences:
                node_occurrences[node_id] = [idx]
                unique_entries.append((node, game_indices[idx]))
            else:
                node_occurrences[node_id].append(idx)

        terminal_values = {}
        non_terminal_nodes = []
        non_terminal_game_indices = []

        for node, gi in unique_entries:
            if node.is_game_over:
                result = node.board.result(claim_draw=False)
                if result == '1-0':
                    value = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif result == '0-1':
                    value = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    value = 0.0
                terminal_values[id(node)] = value
            else:
                non_terminal_nodes.append(node)
                non_terminal_game_indices.append(gi)

        values_by_node_id = {}

        if non_terminal_nodes:
            legal_moves_per_node = []
            legal_indices_per_node = []
            legal_counts = []
            max_legal_count = 0
            for node in non_terminal_nodes:
                legal_moves, legal_indices = node.get_legal_moves_and_indices()
                legal_moves_per_node.append(legal_moves)
                legal_indices_per_node.append(legal_indices)
                legal_count = len(legal_indices)
                legal_counts.append(legal_count)
                if legal_count > max_legal_count:
                    max_legal_count = legal_count

            boards_np = np.stack(
                [
                    self._build_history_tensor(
                        node.board,
                        game_states[gi]['board_history'],
                        current_tensor=self._current_tensor_for_node(node),
                    )
                    for node, gi in zip(non_terminal_nodes, non_terminal_game_indices)
                ],
                axis=0,
            )
            board_tensors = torch.from_numpy(boards_np).to(
                self.device,
                memory_format=torch.channels_last,
                non_blocking=True,
            )

            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        policy_logits_batch, values_batch = self.model(
                            board_tensors,
                            apply_log_softmax=False,
                        )
                else:
                    policy_logits_batch, values_batch = self.model(
                        board_tensors,
                        apply_log_softmax=False,
                    )
                
                # 🔥 OPTIMIZATION: Transfer entire batch to CPU at once, not row by row
                if values_batch.dim() == 2 and values_batch.shape[1] == 3:
                    # WDL output: compute softmax on GPU before transfer
                    wdl_probs = torch.softmax(values_batch, dim=1)
                    values_batch = (wdl_probs[:, 0] - wdl_probs[:, 2])

                # Gather only legal move logits on GPU before transferring to CPU.
                if max_legal_count > 0:
                    legal_index_matrix = self._get_legal_index_scratch(
                        len(non_terminal_nodes),
                        max_legal_count,
                    )
                    legal_index_matrix.fill(0)
                    for row_idx, legal_indices in enumerate(legal_indices_per_node):
                        legal_count = legal_counts[row_idx]
                        if legal_count:
                            legal_index_matrix[row_idx, :legal_count] = legal_indices
                    legal_index_tensor = torch.from_numpy(legal_index_matrix).to(
                        self.device,
                        non_blocking=True,
                    )
                    legal_logits_batch = torch.gather(policy_logits_batch, 1, legal_index_tensor)
                    legal_logits_batch = legal_logits_batch.to(dtype=torch.float16).cpu().numpy()
                else:
                    legal_logits_batch = None
                values_batch = values_batch.float().cpu().numpy()

            for idx, node in enumerate(non_terminal_nodes):
                value = float(values_batch[idx])

                legal_moves = legal_moves_per_node[idx]
                legal_count = legal_counts[idx]

                if legal_moves:
                    legal_logits = legal_logits_batch[idx, :legal_count].astype(np.float32, copy=False)
                    legal_probs = np.exp(legal_logits - legal_logits.max())
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                else:
                    legal_probs = np.array([])

                gi = non_terminal_game_indices[idx]
                add_noise = (
                    game_states[gi].get('_needs_root_noise_on_expand', False)
                    and node is game_states[gi]['root']
                )

                if add_noise and len(legal_moves) > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
                    legal_probs = (1 - self.dirichlet_weight) * legal_probs + self.dirichlet_weight * noise
                    game_states[gi]['_needs_root_noise_on_expand'] = False

                # Expand only if not already expanded (avoid overwriting priors in same batch)
                if not node.expanded:
                    node.expand_children(legal_moves, legal_probs)

                values_by_node_id[id(node)] = value

        all_values = [0.0] * len(nodes)
        for node_id, indices in node_occurrences.items():
            value = values_by_node_id.get(node_id, terminal_values.get(node_id, 0.0))
            for idx in indices:
                all_values[idx] = value

        return all_values


class BatchMCTS:
    """
    Single-game compatibility wrapper over MultiGameBatchMCTS.

    Keeps the old API (`search`, `advance_root`, `update_history`, `reset_tree`)
    while using the same shared multi-game MCTS core.
    """

    def __init__(self, model, config, device):
        self.config = config
        self._multi = MultiGameBatchMCTS(model, config, device)
        self.root = None
        self._root_synced = False
        self.board_history = []

    def search(self, board, num_simulations, temperature=1.0, add_root_noise=False):
        del temperature  # visit selection temperature is handled by caller

        game_state = {
            'board': board,
            'root': self.root,
            '_root_synced': self._root_synced,
            'board_history': self.board_history,
        }
        visit_counts = self._multi.search_many(
            [game_state],
            num_simulations=num_simulations,
            add_root_noise=add_root_noise,
        )
        self.root = game_state.get('root')
        self._root_synced = bool(game_state.get('_root_synced', False))
        return visit_counts[0] if visit_counts else {}

    def advance_root(self, move):
        if self.root is None:
            return
        child = self.root.get_child_for_move(move)
        if child is None:
            self.root = None
            self._root_synced = False
            return
        _ = child.board
        child.parent = None
        child.parent_edge_index = -1
        self.root = child
        self._root_synced = True

    def update_history(self, board):
        self.board_history.append(self._multi._encode_history_entry(board))
        max_history = int(self.config.get('model', {}).get('history_positions', 0)) + 10
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]

    def reset_tree(self):
        self.root = None
        self._root_synced = False
        self.board_history = []


class BatchSelfPlayMCTSBatch:
    """
    True batch self-play: multiple games in parallel, shared GPU eval batches.
    """

    def __init__(
        self,
        model,
        config,
        device,
        max_batch_games_per_worker,
        opponent_model=None,
        opponent_source_label="current",
        opponent_models_by_label=None,
        opponent_plan_labels=None,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.opponent_model = opponent_model
        self.opponent_source_label = str(opponent_source_label or "current")
        rl_cfg = config.get('reinforcement_learning', {})

        self.mcts = MultiGameBatchMCTS(model, config, device)
        self.opponent_models_by_label = {}
        if isinstance(opponent_models_by_label, dict):
            for label, opp_model in opponent_models_by_label.items():
                if opp_model is not None:
                    self.opponent_models_by_label[str(label)] = opp_model
        elif opponent_model is not None:
            self.opponent_models_by_label[self.opponent_source_label] = opponent_model
        self.opponent_mcts_by_label = {
            str(label): MultiGameBatchMCTS(opp_model, config, device)
            for label, opp_model in self.opponent_models_by_label.items()
            if opp_model is not None
        }
        self.opponent_mcts = next(iter(self.opponent_mcts_by_label.values()), None)
        self.opponent_plan_labels = list(opponent_plan_labels or [])

        if device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.num_simulations = int(config['reinforcement_learning']['mcts_simulations'])
        self.temp_threshold = config['reinforcement_learning']['mcts_temperature_threshold']
        self.temperature = config['reinforcement_learning'].get('mcts_temperature', 1.0)
        self.max_moves = _resolve_selfplay_max_moves(config)
        self.auto_claim_draw = bool(rl_cfg.get('self_play_auto_claim_draw', False))
        self.claim_draw_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_draw_after_moves', self.max_moves)),
        )
        self.claim_repetition_after_moves = max(
            0,
            int(rl_cfg.get('self_play_claim_repetition_after_moves', min(self.claim_draw_after_moves, 80))),
        )
        self.progress_report_interval_games = max(
            1,
            int(rl_cfg.get('self_play_progress_interval_games', 1)),
        )
        self.opening_diversity_enabled = bool(
            rl_cfg.get('self_play_opening_diversity_enabled', False)
        )
        self.opening_diversity_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_opening_diversity_fraction', 0.5))),
        )
        self.opening_diversity_min_plies = max(
            0,
            int(rl_cfg.get('self_play_opening_diversity_min_plies', 2)),
        )
        self.opening_diversity_max_plies = max(
            self.opening_diversity_min_plies,
            int(rl_cfg.get('self_play_opening_diversity_max_plies', 6)),
        )
        self.adjudication_enabled = bool(rl_cfg.get('self_play_adjudication_enabled', False))
        self.adjudication_min_moves = max(
            0,
            int(rl_cfg.get('self_play_adjudication_min_moves', 80)),
        )
        self.adjudication_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_adjudication_threshold', 0.92))),
        )
        self.adjudication_patience = max(
            1,
            int(rl_cfg.get('self_play_adjudication_patience', 6)),
        )
        self.resignation_enabled = bool(rl_cfg.get('self_play_resignation_enabled', False))
        self.resignation_min_moves = max(
            0,
            int(rl_cfg.get('self_play_resignation_min_moves', 60)),
        )
        self.resignation_threshold = min(
            0.999,
            max(0.0, float(rl_cfg.get('self_play_resignation_threshold', 0.92))),
        )
        self.resignation_patience = max(
            1,
            int(rl_cfg.get('self_play_resignation_patience', 3)),
        )
        self.resignation_disable_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_resignation_disable_fraction', 0.10))),
        )
        self.randomize_learner_color = bool(rl_cfg.get('self_play_randomize_learner_color', True))
        self.replay_dynamic_cap_enabled = bool(rl_cfg.get('replay_dynamic_cap_enabled', False))
        self.replay_cap_fraction_decisive = float(rl_cfg.get('replay_cap_fraction_decisive', 0.60))
        self.replay_cap_fraction_draw = float(rl_cfg.get('replay_cap_fraction_draw', 0.15))
        self.replay_cap_min_positions = int(rl_cfg.get('replay_cap_min_positions', 16))
        self.replay_cap_max_positions = int(rl_cfg.get('replay_cap_max_positions', 120))
        
        self.max_positions_per_game = max(
            0,
            int(rl_cfg.get('replay_max_positions_per_game', 32)),
        )

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0
        self._progress_base = 0
        self._plan_cursor = 0

    def _compute_draw_value_target(self, move_count):
        del move_count
        # Keep draw targets centered at 0. Non-zero draw values bias both sides in
        # the same direction and break the zero-sum calibration expected by MCTS.
        return 0.0

    def _select_top_scored_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        del history_len
        positions = np.linspace(0, len(candidates) - 1, num=effective_cap)
        selected_ids = {int(round(pos)) for pos in positions}
        selected = []
        for idx, item in enumerate(candidates):
            if idx in selected_ids:
                selected.append(item)
        return selected

    def _allocate_draw_stratified_caps(self, buckets, effective_cap):
        bucket_count = len(buckets)
        allocations = [0] * bucket_count
        if effective_cap <= 0 or bucket_count <= 0:
            return allocations

        bucket_sizes = [len(bucket) for bucket in buckets]
        non_empty = [idx for idx, size in enumerate(bucket_sizes) if size > 0]
        if not non_empty:
            return allocations

        if effective_cap < len(non_empty):
            desired = []
            for idx in range(bucket_count):
                if bucket_sizes[idx] <= 0:
                    desired.append(0.0)
                elif idx == bucket_count - 1:
                    desired.append(1.0)
                elif idx == bucket_count - 2:
                    desired.append(0.75)
                else:
                    desired.append(0.5)
        else:
            desired = [0.20, 0.35, 0.45]

        remaining = int(effective_cap)
        for idx in non_empty:
            allocations[idx] = 1
            remaining -= 1

        if remaining <= 0:
            return allocations

        desired_counts = [float(effective_cap) * desired[idx] for idx in range(bucket_count)]
        remainders = []
        for idx in range(bucket_count):
            if bucket_sizes[idx] <= allocations[idx]:
                continue
            extra = int(math.floor(max(0.0, desired_counts[idx] - allocations[idx])))
            if extra > 0:
                grant = min(extra, bucket_sizes[idx] - allocations[idx], remaining)
                allocations[idx] += grant
                remaining -= grant
            remainder = max(0.0, desired_counts[idx] - allocations[idx])
            remainders.append((remainder, idx))

        while remaining > 0:
            progressed = False
            remainders.sort(key=lambda pair: (-pair[0], pair[1]))
            for _, idx in remainders:
                if remaining <= 0:
                    break
                if allocations[idx] >= bucket_sizes[idx]:
                    continue
                allocations[idx] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

        if remaining > 0:
            fill_order = sorted(
                non_empty,
                key=lambda idx: (
                    -bucket_sizes[idx],
                    -desired_counts[idx],
                    -idx,
                ),
            )
            while remaining > 0:
                progressed = False
                for idx in fill_order:
                    if remaining <= 0:
                        break
                    if allocations[idx] >= bucket_sizes[idx]:
                        continue
                    allocations[idx] += 1
                    remaining -= 1
                    progressed = True
                if not progressed:
                    break

        return allocations

    def _select_draw_candidates_stratified(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)

        if effective_cap < 3:
            return self._select_top_scored_candidates(candidates, effective_cap, history_len)

        denom = float(max(1, history_len - 1))
        buckets = [[], [], []]
        for item in candidates:
            progress = float(item['history_idx']) / denom
            if progress < (1.0 / 3.0):
                buckets[0].append(item)
            elif progress < (2.0 / 3.0):
                buckets[1].append(item)
            else:
                buckets[2].append(item)

        allocations = self._allocate_draw_stratified_caps(buckets, effective_cap)
        selected = []
        selected_ids = set()
        for bucket, bucket_cap in zip(buckets, allocations):
            for item in self._select_top_scored_candidates(bucket, bucket_cap, history_len):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        if len(selected) < effective_cap:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_ids
            ]
            for item in self._select_top_scored_candidates(
                leftovers,
                effective_cap - len(selected),
                history_len,
            ):
                history_idx = int(item['history_idx'])
                if history_idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(history_idx)

        return selected

    def _select_history_indices_to_keep(self, candidates, history_len, is_decisive=False):
        total_candidates = len(candidates)
        if total_candidates <= 0:
            return [], 0, 0

        filtered = list(candidates)
        curriculum_dropped = 0

        # Determine effective cap limits
        if self.replay_dynamic_cap_enabled:
            fraction = self.replay_cap_fraction_decisive if is_decisive else self.replay_cap_fraction_draw
            effective_cap = int(math.ceil(len(filtered) * fraction))
            effective_cap = max(self.replay_cap_min_positions, effective_cap)
            effective_cap = min(self.replay_cap_max_positions, effective_cap)
        else:
            effective_cap = self.max_positions_per_game

        if effective_cap <= 0 or len(filtered) <= effective_cap:
            selected = filtered
            cap_dropped = 0
        else:
            if is_decisive:
                selected = self._select_top_scored_candidates(filtered, effective_cap, history_len)
            else:
                selected = self._select_draw_candidates_stratified(filtered, effective_cap, history_len)
            cap_dropped = len(filtered) - len(selected)

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected, int(curriculum_dropped), int(cap_dropped)

    def _build_candidate_positions(self, gs, outcome, draw_value_target=0.0):
        candidate_positions = []
        is_draw = (outcome == 0.0)
        for history_idx, history_entry in enumerate(gs['game_history']):
            history_count, policy_indices, policy_values, turn = history_entry
            if is_draw:
                value = draw_value_target
            else:
                value = outcome if turn == chess.WHITE else -outcome
            candidate_positions.append({
                'history_idx': history_idx,
                'history_count': history_count,
                'policy_indices': policy_indices,
                'policy_values': policy_values,
                'turn': turn,
                'value': value,
            })
        return candidate_positions

    def _append_selected_positions_from_game(
        self,
        positions,
        gs,
        outcome,
        *,
        history_positions,
    ):
        history_len = len(gs['game_history'])
        candidate_positions = self._build_candidate_positions(gs, outcome, draw_value_target=0.0)
        selected_positions, curriculum_dropped, cap_dropped = self._select_history_indices_to_keep(
            candidate_positions,
            history_len,
            is_decisive=(outcome != 0.0),
        )

        for item in selected_positions:
            board_tensor_np = _build_history_tensor_from_encoded(
                turn=item['turn'],
                encoded_history=gs['board_history'],
                history_count=item['history_count'],
                history_positions=history_positions,
                empty_history_tensor=_EMPTY_HISTORY_TENSOR,
            )
            board_tensor = torch.from_numpy(board_tensor_np)
            positions.append((
                board_tensor,
                item['policy_indices'],
                item['policy_values'],
                torch.tensor([item['value']], dtype=torch.float32),
            ))

        return history_len, int(curriculum_dropped), int(cap_dropped)

    def _should_auto_claim_draw(self, board, move_count):
        if not self.auto_claim_draw:
            return False
        try:
            if move_count >= self.claim_repetition_after_moves:
                claim_threefold = getattr(board, 'can_claim_threefold_repetition', None)
                if callable(claim_threefold) and bool(claim_threefold()):
                    return True
            if move_count < self.claim_draw_after_moves:
                return False
            return bool(board.can_claim_draw())
        except Exception:
            return False

    def _sample_opening_prefix(self):
        if not self.opening_diversity_enabled:
            return ()
        if self.opening_diversity_fraction <= 0.0:
            return ()
        if np.random.random() > self.opening_diversity_fraction:
            return ()

        line = _SELFPLAY_SELFPLAY_OPENING_LINES[
            int(np.random.randint(0, len(_SELFPLAY_SELFPLAY_OPENING_LINES)))
        ]
        max_plies = min(len(line), self.opening_diversity_max_plies)
        min_plies = min(max_plies, self.opening_diversity_min_plies)
        if max_plies <= 0:
            return ()
        if min_plies >= max_plies:
            plies = max_plies
        else:
            plies = int(np.random.randint(min_plies, max_plies + 1))
        return line[:plies]

    def _maybe_adjudicate_game(self, gs, root, board):
        if not self.adjudication_enabled or root is None:
            return None
        if gs['move_count'] < self.adjudication_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        white_value = root_value if board.turn == chess.WHITE else -root_value
        threshold = self.adjudication_threshold

        if white_value >= threshold:
            gs['white_advantage_streak'] = int(gs.get('white_advantage_streak', 0)) + 1
            gs['black_advantage_streak'] = 0
            if gs['white_advantage_streak'] >= self.adjudication_patience:
                return '1-0'
        elif white_value <= -threshold:
            gs['black_advantage_streak'] = int(gs.get('black_advantage_streak', 0)) + 1
            gs['white_advantage_streak'] = 0
            if gs['black_advantage_streak'] >= self.adjudication_patience:
                return '0-1'
        else:
            gs['white_advantage_streak'] = 0
            gs['black_advantage_streak'] = 0

        return None

    def _maybe_resign_game(self, gs, root, board):
        if not self.resignation_enabled or gs.get('resignation_disabled', False):
            return None
        if root is None or gs['move_count'] < self.resignation_min_moves:
            return None
        visits = int(getattr(root, 'visit_count', 0) or 0)
        if visits <= 0:
            return None

        root_value = float(root.value_sum / max(1, visits))
        if root_value <= -self.resignation_threshold:
            gs['resign_streak'] = int(gs.get('resign_streak', 0)) + 1
            if gs['resign_streak'] >= self.resignation_patience:
                return '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            gs['resign_streak'] = 0
        return None

    def _uses_learner_model(self, gs, board):
        opponent_label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
        if opponent_label not in self.opponent_mcts_by_label:
            return True
        learner_color = gs.get('learner_color', chess.WHITE)
        return bool(board.turn == learner_color)

    def _get_game_opponent_mcts(self, gs):
        opponent_label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
        return self.opponent_mcts_by_label.get(opponent_label)

    def _apply_opening_prefix(self, gs):
        prefix = self._sample_opening_prefix()
        if not prefix:
            return

        board = gs['board']
        for uci in prefix:
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                break
            if move not in board.legal_moves:
                break

            gs['board_history'].append(self.mcts._encode_history_entry(board))
            board.push(move)
            gs['move_count'] += 1

            if board.is_game_over(claim_draw=False) or gs['move_count'] >= self.max_moves:
                gs['done'] = True
                break

    def play_games(self, num_games):
        all_positions = []
        game_lengths = []
        self._games_completed = 0
        self._plan_cursor = 0
        total_dropped_positions = 0
        total_truncated_games = 0
        total_claimable_draw_ended_games = 0
        total_completed_length_sum = 0
        total_truncated_length_sum = 0
        total_completed_white_wins = 0
        total_completed_black_wins = 0
        total_completed_draws = 0
        total_decisive_games = 0
        total_decisive_length_sum = 0
        total_curriculum_dropped_positions = 0
        total_cap_dropped_positions = 0
        total_resigned_games = 0
        opponent_source_counts = {}
        opponent_source_results = {}

        games_left = num_games
        batch_idx = 0

        while games_left > 0:
            batch_size = min(self.max_batch_games_per_worker, games_left)
            batch_idx += 1

            batch_plan_labels = list(self.opponent_plan_labels[self._plan_cursor:self._plan_cursor + batch_size])
            self._plan_cursor += batch_size
            positions, lengths, batch_stats = self._play_batch(batch_size, batch_plan_labels=batch_plan_labels)
            all_positions.extend(positions)
            game_lengths.extend(lengths)
            total_dropped_positions += int(batch_stats.get('dropped_positions', 0))
            total_truncated_games += int(batch_stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(batch_stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(batch_stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(batch_stats.get('truncated_length_sum', 0))
            total_completed_white_wins += int(batch_stats.get('completed_white_wins', 0))
            total_completed_black_wins += int(batch_stats.get('completed_black_wins', 0))
            total_completed_draws += int(batch_stats.get('completed_draws', 0))
            total_decisive_games += int(batch_stats.get('decisive_games', 0))
            total_decisive_length_sum += int(batch_stats.get('decisive_length_sum', 0))
            total_curriculum_dropped_positions += int(batch_stats.get('curriculum_dropped_positions', 0))
            total_cap_dropped_positions += int(batch_stats.get('cap_dropped_positions', 0))
            total_resigned_games += int(batch_stats.get('resigned_games', 0))
            for label, count in dict(batch_stats.get('opponent_source_counts', {}) or {}).items():
                opponent_source_counts[str(label)] = int(opponent_source_counts.get(str(label), 0)) + int(count)
            for label, stats in dict(batch_stats.get('opponent_source_results', {}) or {}).items():
                result_stats = opponent_source_results.setdefault(
                    str(label),
                    {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
                )
                result_stats['wins'] += int((stats or {}).get('wins', 0))
                result_stats['draws'] += int((stats or {}).get('draws', 0))
                result_stats['losses'] += int((stats or {}).get('losses', 0))
                result_stats['games'] += int((stats or {}).get('games', 0))

            games_left -= batch_size

        total_games = len(game_lengths)
        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        self.last_selfplay_stats = {
            'truncated_games': total_truncated_games,
            'completed_games': max(0, total_games - total_truncated_games),
            'dropped_positions': int(total_dropped_positions),
            'claimable_draw_ended_games': int(total_claimable_draw_ended_games),
            'completed_length_sum': int(total_completed_length_sum),
            'truncated_length_sum': int(total_truncated_length_sum),
            'completed_white_wins': int(total_completed_white_wins),
            'completed_black_wins': int(total_completed_black_wins),
            'completed_draws': int(total_completed_draws),
            'decisive_games': int(total_decisive_games),
            'decisive_length_sum': int(total_decisive_length_sum),
            'curriculum_dropped_positions': int(total_curriculum_dropped_positions),
            'cap_dropped_positions': int(total_cap_dropped_positions),
            'resigned_games': int(total_resigned_games),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
            'total_games': int(total_games),
        }
        return all_positions, game_lengths

    def _report_progress(self):
        if self._progress_file is None:
            return
        try:
            with open(self._progress_file, 'w') as _pf:
                _pf.write(str(self._progress_base + self._games_completed))
        except Exception:
            pass

    def _mark_game_completed(self, gs):
        if gs.get('_completion_reported', False):
            return
        gs['_completion_reported'] = True
        self._games_completed += 1
        if (
            self._games_completed == 1
            or (self._games_completed % self.progress_report_interval_games) == 0
        ):
            self._report_progress()

    def _play_batch(self, batch_size, batch_plan_labels=None):
        max_moves = self.max_moves

        game_states = []
        for _game_idx in range(batch_size):
            game_opponent_label = (
                str(batch_plan_labels[_game_idx])
                if batch_plan_labels is not None and _game_idx < len(batch_plan_labels)
                else self.opponent_source_label
            )
            has_frozen_opponent = game_opponent_label in self.opponent_mcts_by_label
            game_states.append({
                'board': chess.Board(),
                'board_history': [],
                'root': None,
                '_root_synced': False,
                'opponent_root': None,
                '_opponent_root_synced': False,
                'game_history': [],
                'move_count': 0,
                'done': False,
                'white_advantage_streak': 0,
                'black_advantage_streak': 0,
                'adjudicated_result': None,
                'resigned_result': None,
                'resign_streak': 0,
                'resignation_disabled': bool(np.random.random() < self.resignation_disable_fraction),
                'opponent_source_label': game_opponent_label,
                'learner_color': (
                    chess.WHITE
                    if not has_frozen_opponent or not self.randomize_learner_color
                    else (chess.WHITE if np.random.random() < 0.5 else chess.BLACK)
                ),
                '_completion_reported': False,
            })
            self._apply_opening_prefix(game_states[-1])

        while True:
            active_indices = [i for i, gs in enumerate(game_states) if not gs['done']]
            if not active_indices:
                break

            visit_counts_by_index = {}

            def _run_search_for_indices(indices, mcts_ref, root_key, synced_key):
                if not indices:
                    return
                group_states = []
                for i in indices:
                    gs = game_states[i]
                    group_states.append({
                        'board': gs['board'],
                        'root': gs.get(root_key),
                        '_root_synced': bool(gs.get(synced_key, False)),
                        'board_history': gs['board_history'],
                    })
                visit_counts_list_group = mcts_ref.search_many(
                    group_states,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                )
                for gs_idx, local_state, visit_counts in zip(indices, group_states, visit_counts_list_group):
                    gs = game_states[gs_idx]
                    gs[root_key] = local_state.get('root')
                    gs[synced_key] = bool(local_state.get('_root_synced', False))
                    visit_counts_by_index[gs_idx] = visit_counts

            if not self.opponent_mcts_by_label:
                _run_search_for_indices(active_indices, self.mcts, 'root', '_root_synced')
            else:
                learner_indices = [
                    i for i in active_indices
                    if self._uses_learner_model(game_states[i], game_states[i]['board'])
                ]
                learner_index_set = set(learner_indices)
                opponent_indices = [
                    i for i in active_indices
                    if i not in learner_index_set
                ]
                _run_search_for_indices(learner_indices, self.mcts, 'root', '_root_synced')
                grouped_opponent_indices = {}
                for idx in opponent_indices:
                    label = str(game_states[idx].get('opponent_source_label', self.opponent_source_label) or "current")
                    grouped_opponent_indices.setdefault(label, []).append(idx)
                for label, indices in grouped_opponent_indices.items():
                    opponent_mcts = self.opponent_mcts_by_label.get(label)
                    if opponent_mcts is None:
                        continue
                    _run_search_for_indices(indices, opponent_mcts, 'opponent_root', '_opponent_root_synced')

            for idx in active_indices:
                gs = game_states[idx]
                board = gs['board']
                visit_counts = visit_counts_by_index.get(idx, None)
                if visit_counts is None:
                    continue

                if self.temp_threshold > 0 and gs['move_count'] < self.temp_threshold:
                    temperature = self.temperature
                else:
                    temperature = 0.0
                learner_turn = self._uses_learner_model(gs, board)
                game_opponent_mcts = self._get_game_opponent_mcts(gs)
                root_key = 'root' if learner_turn or game_opponent_mcts is None else 'opponent_root'
                synced_key = '_root_synced' if learner_turn or game_opponent_mcts is None else '_opponent_root_synced'
                root = gs.get(root_key)
                adjudicated_result = self._maybe_adjudicate_game(gs, root, board)
                if adjudicated_result is not None:
                    gs['done'] = True
                    gs['adjudicated_result'] = adjudicated_result
                    gs['root'] = None
                    gs['opponent_root'] = None
                    gs['_root_synced'] = False
                    gs['_opponent_root_synced'] = False
                    self._mark_game_completed(gs)
                    continue

                resigned_result = self._maybe_resign_game(gs, root, board)
                if resigned_result is not None:
                    gs['done'] = True
                    gs['resigned_result'] = resigned_result
                    gs['root'] = None
                    gs['opponent_root'] = None
                    gs['_root_synced'] = False
                    gs['_opponent_root_synced'] = False
                    self._mark_game_completed(gs)
                    continue

                move = self._select_move_from_visits(visit_counts, temperature)
                if learner_turn or game_opponent_mcts is None:
                    policy_indices, policy_values = _build_sparse_policy_target_from_visits(visit_counts, board)
                    history_count = len(gs['board_history'])
                    gs['game_history'].append((
                        history_count,
                        policy_indices,
                        policy_values,
                        board.turn,
                    ))

                # Update history BEFORE making the move
                # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
                gs['board_history'].append(self.mcts._encode_history_entry(board))

                # Reuse selected subtree directly to skip FEN-matching next turn.
                if game_opponent_mcts is None and root is not None:
                    next_root = root.get_child_for_move(move)
                else:
                    next_root = None
                if next_root is not None:
                    _ = next_root.board
                    next_root.parent = None
                    next_root.parent_edge_index = -1
                    gs[root_key] = next_root
                    gs[synced_key] = True
                else:
                    gs['root'] = None
                    gs['_root_synced'] = False
                    gs['opponent_root'] = None
                    gs['_opponent_root_synced'] = False

                board.push(move)
                gs['move_count'] += 1

                forced_game_over = board.is_game_over(claim_draw=False)
                auto_claim_draw = self._should_auto_claim_draw(board, gs['move_count'])
                if forced_game_over or auto_claim_draw or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    gs['ended_by_auto_claim_draw'] = auto_claim_draw
                    # Free up memory immediately
                    gs['root'] = None
                    gs['opponent_root'] = None
                    gs['_root_synced'] = False
                    gs['_opponent_root_synced'] = False
                    self._mark_game_completed(gs)

        positions = []
        game_lengths = []
        dropped_positions = 0
        truncated_games = 0
        claimable_draw_ended_games = 0
        adjudicated_games = 0
        resigned_games = 0
        completed_length_sum = 0
        truncated_length_sum = 0
        completed_white_wins = 0
        completed_black_wins = 0
        completed_draws = 0
        learner_wins = 0
        learner_draws = 0
        learner_losses = 0
        opponent_source_counts = {}
        opponent_source_results = {}
        decisive_games = 0
        decisive_length_sum = 0
        curriculum_dropped_positions = 0
        cap_dropped_positions = 0
        history_positions = int(self.config.get('model', {}).get('history_positions', 0))

        for gs in game_states:
            board = gs['board']
            opponent_label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
            opponent_source_counts[opponent_label] = int(opponent_source_counts.get(opponent_label, 0)) + 1
            ended_by_claimable_draw = (
                bool(gs.get('ended_by_auto_claim_draw', False))
            )
            if ended_by_claimable_draw:
                claimable_draw_ended_games += 1
            adjudicated_result = gs.get('adjudicated_result', None)
            if adjudicated_result is not None:
                adjudicated_games += 1
            resigned_result = gs.get('resigned_result', None)
            if resigned_result is not None:
                resigned_games += 1
            result = (
                resigned_result
                if resigned_result is not None
                else (
                    adjudicated_result
                    if adjudicated_result is not None
                    else ('1/2-1/2' if ended_by_claimable_draw else board.result(claim_draw=False))
                )
            )
            game_ply_len = int(gs.get('move_count', len(gs['game_history'])))
            if result == '*':
                truncated_games += 1
                stored_history_len = len(gs['game_history'])
                dropped_positions += stored_history_len
                truncated_length_sum += game_ply_len
                game_lengths.append(game_ply_len)
                continue
            if result == '1-0':
                outcome = 1.0
                completed_white_wins += 1
                decisive_games += 1
            elif result == '0-1':
                outcome = -1.0
                completed_black_wins += 1
                decisive_games += 1
            else:
                outcome = 0.0
                completed_draws += 1

            if opponent_label in self.opponent_mcts_by_label:
                learner_color = gs.get('learner_color', chess.WHITE)
                learner_outcome = outcome if learner_color == chess.WHITE else -outcome
                result_stats = opponent_source_results.setdefault(
                    opponent_label,
                    {'wins': 0, 'draws': 0, 'losses': 0, 'games': 0},
                )
                if learner_outcome > 0.0:
                    learner_wins += 1
                    result_stats['wins'] += 1
                elif learner_outcome < 0.0:
                    learner_losses += 1
                    result_stats['losses'] += 1
                else:
                    learner_draws += 1
                    result_stats['draws'] += 1
                result_stats['games'] += 1
            completed_length_sum += game_ply_len
            if outcome != 0.0:
                decisive_length_sum += game_ply_len
            _, game_curriculum_dropped, game_cap_dropped = self._append_selected_positions_from_game(
                positions,
                gs,
                outcome,
                history_positions=history_positions,
            )
            curriculum_dropped_positions += game_curriculum_dropped
            cap_dropped_positions += game_cap_dropped

            game_lengths.append(game_ply_len)

        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        return positions, game_lengths, {
            'total_games': int(len(game_states)),
            'truncated_games': int(truncated_games),
            'dropped_positions': int(dropped_positions),
            'claimable_draw_ended_games': int(claimable_draw_ended_games),
            'adjudicated_games': int(adjudicated_games),
            'resigned_games': int(resigned_games),
            'completed_length_sum': int(completed_length_sum),
            'truncated_length_sum': int(truncated_length_sum),
            'completed_white_wins': int(completed_white_wins),
            'completed_black_wins': int(completed_black_wins),
            'completed_draws': int(completed_draws),
            'learner_wins': int(learner_wins),
            'learner_draws': int(learner_draws),
            'learner_losses': int(learner_losses),
            'decisive_games': int(decisive_games),
            'decisive_length_sum': int(decisive_length_sum),
            'curriculum_dropped_positions': int(curriculum_dropped_positions),
            'cap_dropped_positions': int(cap_dropped_positions),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
        }

    def _select_move_from_visits(self, visit_counts, temperature):
        return _select_move_from_visits_safe(visit_counts, temperature)


# ============================================================================
# PARALLEL WORKER FOR MULTIPROCESSING
# ============================================================================

def _build_worker_logger(rank, worker_verbose):
    def _wlog(message):
        if worker_verbose:
            print(f"Worker {rank}: {message}")

    return _wlog


def _configure_selfplay_worker_runtime(config, device_id):
    rl_cfg = config.get('reinforcement_learning', {})
    self_play_threads = rl_cfg.get('self_play_torch_threads', None)

    if self_play_threads is not None:
        try:
            self_play_threads = int(self_play_threads)
            if self_play_threads > 0:
                torch.set_num_threads(self_play_threads)
                try:
                    torch.set_num_interop_threads(max(1, min(4, self_play_threads)))
                except Exception:
                    pass
        except Exception:
            pass

    if device_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    return device


def _build_selfplay_worker_model(config, device):
    from src.model import ChessNet

    worker_config = dict(config)
    worker_config['model'] = {**config.get('model', {}), 'print_summary': False}
    model = ChessNet(worker_config).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def _load_worker_model_state(model, model_state, rank):
    from src.model import normalize_state_dict_keys, transfer_matching_weights

    # Accept both raw state_dict and full training checkpoints.
    if isinstance(model_state, dict):
        if 'model_state_dict' in model_state and isinstance(model_state.get('model_state_dict'), dict):
            model_state = model_state['model_state_dict']
        elif 'state_dict' in model_state and isinstance(model_state.get('state_dict'), dict):
            model_state = model_state['state_dict']

    if model_state:
        model_state = {
            k: v for k, v in model_state.items()
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        }
        model_state = normalize_state_dict_keys(
            model_state,
            target_keys=set(model.state_dict().keys()),
        )

    # Prefer strict load; when architectures differ, transfer only matching tensors.
    try:
        model.load_state_dict(model_state, strict=True)
        return
    except Exception:
        report = transfer_matching_weights(model, model_state)
        extra_missing = [
            k for k in report.get('missing_keys', [])
            if not (k.endswith('coord_x') or k.endswith('coord_y'))
        ]
        unexpected = report.get('unexpected_keys', [])
        shape_mismatch = report.get('shape_mismatch', [])

        if extra_missing or unexpected or shape_mismatch:
            print(
                f"⚠️ Worker {rank}: state_dict mismatch -> transfer fallback. "
                f"Matched={report.get('matched_tensors', 0)}/{report.get('total_tensors', 0)} "
                f"({report.get('match_ratio', 0.0) * 100:.1f}%), "
                f"Missing={extra_missing}, Unexpected={unexpected}, "
                f"ShapeMismatch={len(shape_mismatch)}"
            )


def _create_selfplay_engine(
    model,
    config,
    device,
    num_games,
    wlog,
    opponent_model=None,
    opponent_source_label="current",
    opponent_models_by_label=None,
    opponent_plan_labels=None,
):
    rl_cfg = config.get('reinforcement_learning', {})
    use_batch = rl_cfg.get('use_batch_selfplay', False)
    max_batch_games = rl_cfg.get('max_batch_games_per_worker', 256)

    if use_batch:
        configured_max = max_batch_games
        mode_label = "batch"
    else:
        configured_max = 1
        mode_label = "single-via-batch"

    engine = BatchSelfPlayMCTSBatch(
        model,
        config,
        device,
        configured_max,
        opponent_model=opponent_model,
        opponent_source_label=opponent_source_label,
        opponent_models_by_label=opponent_models_by_label,
        opponent_plan_labels=opponent_plan_labels,
    )
    actual_max = min(configured_max, num_games)
    wlog(
        f"Self-play engine: {mode_label} "
        f"(max {configured_max}, actual {actual_max})"
    )
    return engine


def _play_games_with_engine(
    rank,
    engine,
    config,
    num_games,
    result_file_path,
    wlog,
    result_queue=None,
    task_id=None,
    stream_results_to_queue=False,
):
    rl_cfg = config.get('reinforcement_learning', {})
    progress_file_path = result_file_path.replace('.pkl', '.progress')

    def _write_progress(n):
        try:
            with open(progress_file_path, 'w') as _pf:
                _pf.write(str(n))
        except Exception:
            pass

    if hasattr(engine, '_progress_file'):
        engine._progress_file = progress_file_path

    wlog(
        f"Playing {num_games} games with MCTS "
        f"({config['reinforcement_learning']['mcts_simulations']} sims/move)"
    )

    save_every = rl_cfg.get('self_play_save_every_games_resolved', None)
    if stream_results_to_queue:
        save_every = max(
            1,
            int(
                rl_cfg.get(
                    'self_play_queue_chunk_games',
                    rl_cfg.get('max_batch_games_per_worker', 1),
                )
            ),
        )
    elif save_every is None:
        save_mult = rl_cfg.get('self_play_save_every_games', 0)
        try:
            save_mult = float(save_mult)
        except Exception:
            save_mult = 0
        if save_mult and save_mult > 0:
            games_per_iter = rl_cfg.get('games_per_iteration', 0)
            try:
                games_per_iter = int(games_per_iter)
            except Exception:
                games_per_iter = 0
            save_every = max(1, int(round(games_per_iter * save_mult)))
        else:
            save_every = 0

    total_positions = 0
    total_games = 0
    total_dropped_positions = 0
    total_truncated_games = 0
    total_claimable_draw_ended_games = 0
    total_completed_length_sum = 0
    total_truncated_length_sum = 0

    if save_every and save_every > 0:
        if not stream_results_to_queue:
            with open(result_file_path, 'wb') as f:
                pass
        games_left = num_games
        while games_left > 0:
            chunk_games = min(save_every, games_left)
            if hasattr(engine, '_progress_file'):
                engine._progress_file = progress_file_path
            if hasattr(engine, '_progress_base'):
                engine._progress_base = total_games
            positions, game_lengths = engine.play_games(chunk_games)
            stats = getattr(engine, 'last_selfplay_stats', {}) or {}

            if stream_results_to_queue and result_queue is not None:
                result_queue.put({
                    'type': 'payload',
                    'rank': rank,
                    'task_id': task_id,
                    'positions': _pack_positions_for_transfer(positions),
                    'game_lengths': list(game_lengths),
                    'stats': stats,
                })
            else:
                with open(result_file_path, 'ab') as f:
                    pickle.dump((positions, game_lengths, stats), f)

            total_positions += len(positions)
            total_games += len(game_lengths)
            total_dropped_positions += int(stats.get('dropped_positions', 0))
            total_truncated_games += int(stats.get('truncated_games', 0))
            total_claimable_draw_ended_games += int(stats.get('claimable_draw_ended_games', 0))
            total_completed_length_sum += int(stats.get('completed_length_sum', 0))
            total_truncated_length_sum += int(stats.get('truncated_length_sum', 0))
            _write_progress(total_games)
            games_left -= chunk_games
    else:
        if hasattr(engine, '_progress_file'):
            engine._progress_file = progress_file_path
        positions, game_lengths = engine.play_games(num_games)
        stats = getattr(engine, 'last_selfplay_stats', {}) or {}
        total_positions = len(positions)
        total_games = len(game_lengths)
        total_dropped_positions = int(stats.get('dropped_positions', 0))
        total_truncated_games = int(stats.get('truncated_games', 0))
        total_claimable_draw_ended_games = int(stats.get('claimable_draw_ended_games', 0))
        total_completed_length_sum = int(stats.get('completed_length_sum', 0))
        total_truncated_length_sum = int(stats.get('truncated_length_sum', 0))
        _write_progress(total_games)

        if stream_results_to_queue and result_queue is not None:
            result_queue.put({
                'type': 'payload',
                'rank': rank,
                'task_id': task_id,
                'positions': _pack_positions_for_transfer(positions),
                'game_lengths': list(game_lengths),
                'stats': stats,
            })
        else:
            with open(result_file_path, 'wb') as f:
                pickle.dump((positions, game_lengths, stats), f)

    wlog(f"Generated {total_positions} positions from {total_games} games")
    if total_truncated_games > 0 or total_dropped_positions > 0:
        wlog(
            "Dropped due to truncation: "
            f"{total_dropped_positions} positions across {total_truncated_games} games"
        )
    if total_claimable_draw_ended_games > 0:
        wlog(f"Claimable-draw ended games: {total_claimable_draw_ended_games}")
    if total_games > 0:
        def _format_plies(value):
            return f"{value:.1f} plies (~{value / 2.0:.1f} full moves)"
        completed_games = max(0, total_games - total_truncated_games)
        if completed_games > 0:
            wlog(
                f"Avg completed game length: {_format_plies(total_completed_length_sum / completed_games)}"
            )
        if total_truncated_games > 0:
            wlog(
                f"Avg truncated game length: {_format_plies(total_truncated_length_sum / total_truncated_games)}"
            )
    wlog(f"Saved to {result_file_path}")
    return total_positions, total_games


def persistent_selfplay_worker(rank, config, device_id, task_queue, result_queue):
    """
    Persistent worker for self-play on Windows.

    The process is spawned once, then receives lightweight play tasks so we avoid
    repeated interpreter startup, imports, model construction and argument pickling.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Persistent worker started on {device}")

        model = _build_selfplay_worker_model(config, device)
        inference_model = model
        opponent_model = None
        engine = None
        engine_signature = None

        while True:
            task = task_queue.get()
            if task is None or task.get('cmd') == 'stop':
                wlog("Stopping persistent worker")
                break

            result_file_path = task['result_file_path']
            try:
                if task.get('model_state') is not None:
                    model_state = task['model_state']
                else:
                    model_state = torch.load(task['model_state_path'], map_location='cpu')
                _load_worker_model_state(model, model_state, rank)
                model.eval()

                opponent_payload = task.get('opponent_payload') or {}
                opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
                opponent_pool_entries = list(opponent_payload.get('pool_entries', []) or [])
                opponent_models_by_label = {}
                opponent_model = None
                opponent_label = str(opponent_payload.get('label') or 'current')
                for entry in opponent_pool_entries:
                    entry_label = str((entry or {}).get('label') or 'current')
                    entry_state = (entry or {}).get('state')
                    if entry_state is None or entry_label == 'current':
                        continue
                    pooled_model = _build_selfplay_worker_model(config, device)
                    _load_worker_model_state(pooled_model, entry_state, rank)
                    pooled_model.eval()
                    opponent_models_by_label[entry_label] = pooled_model
                    if opponent_model is None:
                        opponent_model = pooled_model
                        opponent_label = entry_label

                engine = _create_selfplay_engine(
                    inference_model,
                    config,
                    device,
                    int(task['num_games']),
                    wlog,
                    opponent_model=opponent_model,
                    opponent_source_label=opponent_label,
                    opponent_models_by_label=opponent_models_by_label,
                    opponent_plan_labels=opponent_plan_labels,
                )
                engine_signature = None

                if hasattr(engine, 'temperature') and task.get('mcts_temperature') is not None:
                    engine.temperature = float(task['mcts_temperature'])

                total_positions, total_games = _play_games_with_engine(
                    rank,
                    engine,
                    config,
                    int(task['num_games']),
                    result_file_path,
                    wlog,
                    result_queue=result_queue,
                    task_id=task.get('task_id'),
                    stream_results_to_queue=bool(task.get('stream_results_to_queue', False)),
                )
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task['task_id'],
                    'ok': True,
                    'positions': total_positions,
                    'games': total_games,
                })
            except KeyboardInterrupt:
                try:
                    with open(result_file_path, 'wb') as f:
                        pickle.dump(([], []), f)
                finally:
                    result_queue.put({
                        'type': 'result',
                        'rank': rank,
                        'task_id': task.get('task_id'),
                        'ok': False,
                        'interrupt': True,
                    })
                    raise SystemExit(130)
            except Exception as e:
                print(f"❌ Worker {rank} failed: {e}")
                import traceback
                traceback.print_exc()
                with open(result_file_path, 'wb') as f:
                    pickle.dump(([], []), f)
                result_queue.put({
                    'type': 'result',
                    'rank': rank,
                    'task_id': task.get('task_id'),
                    'ok': False,
                    'error': str(e),
                })
    except KeyboardInterrupt:
        raise SystemExit(130)


def play_games_mcts_worker(
    rank,
    model_state,
    config,
    device_id,
    num_games,
    result_file_path,
    opponent_payload=None,
):
    """
    🚀 One-shot worker function for parallel MCTS self-play.
    """
    rl_cfg = config.get('reinforcement_learning', {})
    worker_verbose = bool(rl_cfg.get('self_play_worker_verbose', False))
    wlog = _build_worker_logger(rank, worker_verbose)

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        wlog(f"Starting on {device}")

        if isinstance(model_state, (str, bytes)):
            model_state = torch.load(model_state, map_location='cpu')

        model = _build_selfplay_worker_model(config, device)
        _load_worker_model_state(model, model_state, rank)
        opponent_model = None
        opponent_label = 'current'
        opponent_models_by_label = {}
        opponent_plan_labels = []
        if isinstance(opponent_payload, dict):
            opponent_plan_labels = list(opponent_payload.get('plan_labels', []) or [])
            for entry in list(opponent_payload.get('pool_entries', []) or []):
                entry_label = str((entry or {}).get('label') or 'current')
                entry_state = (entry or {}).get('state')
                if entry_state is None or entry_label == 'current':
                    continue
                pooled_model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(pooled_model, entry_state, rank)
                pooled_model.eval()
                opponent_models_by_label[entry_label] = pooled_model
                if opponent_model is None:
                    opponent_model = pooled_model
                    opponent_label = entry_label

        engine = _create_selfplay_engine(
            model,
            config,
            device,
            num_games,
            wlog,
            opponent_model=opponent_model,
            opponent_source_label=opponent_label,
            opponent_models_by_label=opponent_models_by_label,
            opponent_plan_labels=opponent_plan_labels,
        )
        _play_games_with_engine(rank, engine, config, num_games, result_file_path, wlog)
    except KeyboardInterrupt:
        try:
            with open(result_file_path, 'wb') as f:
                pickle.dump(([], []), f)
        finally:
            raise SystemExit(130)
    except Exception as e:
        print(f"❌ Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        with open(result_file_path, 'wb') as f:
            pickle.dump(([], []), f)


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================================================

def play_games_batch_worker_safe(rank, model_state, config, device_id, num_games, result_file_path):
    """
    ✅ Backwards compatible wrapper that uses PROPER MCTS
    
    This replaces the old fast-but-wrong version
    """
    return play_games_mcts_worker(rank, model_state, config, device_id, num_games, result_file_path)


# For backwards compatibility
BatchSelfPlay = BatchSelfPlayMCTSBatch
MCTS = BatchMCTS
