"""
BATCH SELF-PLAY WITH FULL MCTS - AlphaZero Style
Plays multiple games using MCTS for move selection
CORRECT IMPLEMENTATION:
- Uses MCTS for all move selections (not raw network)
- Training targets = MCTS visit distributions
- High-quality training data
"""

import os
import tempfile
import torch
import chess
import numpy as np
import math
import pickle
import time
import logging
import queue
import hashlib
from collections import OrderedDict
from pathlib import Path
from src.data import board_to_tensor, move_to_index
from src.utils.data_helpers import _move_to_index_cached


_EMPTY_HISTORY_TENSOR = np.zeros((16, 8, 8), dtype=np.float32)
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

_SYZYGY_ORACLE_CACHE = {}
_SELFPLAY_COMPILE_LOCKFILE = "selfplay_torch_compile.lock"
_DEFAULT_REPLAY_MAX_POLICY_TARGETS = 256
_SCOUT_LOCKED_MARGIN = 8
_Q_DELTA_HIST_MIN = -2.0
_Q_DELTA_HIST_MAX = 2.0
_Q_DELTA_HIST_BINS = 200
_REPLAY_SOURCE_UNKNOWN = 0
_REPLAY_SOURCE_LEARNER = 1
_REPLAY_SOURCE_FROZEN_BEST = 2
_REPLAY_SOURCE_FROZEN_ANCHOR = 3
_REPLAY_SOURCE_FROZEN_RECENT = 4
_REPLAY_SOURCE_OTHER = 5


def _replay_source_code(learner_turn, game_opponent_mcts, opponent_label):
    if learner_turn or game_opponent_mcts is None:
        return _REPLAY_SOURCE_LEARNER
    label = str(opponent_label or "")
    if label == "best":
        return _REPLAY_SOURCE_FROZEN_BEST
    if label == "anchor":
        return _REPLAY_SOURCE_FROZEN_ANCHOR
    if label.startswith("recent_"):
        return _REPLAY_SOURCE_FROZEN_RECENT
    return _REPLAY_SOURCE_OTHER


def _q_delta_histogram(values):
    if not values:
        return [0] * _Q_DELTA_HIST_BINS
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return [0] * _Q_DELTA_HIST_BINS
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return [0] * _Q_DELTA_HIST_BINS
    arr = np.clip(arr, _Q_DELTA_HIST_MIN, _Q_DELTA_HIST_MAX)
    hist, _ = np.histogram(arr, bins=_Q_DELTA_HIST_BINS, range=(_Q_DELTA_HIST_MIN, _Q_DELTA_HIST_MAX))
    return [int(v) for v in hist.tolist()]


def _q_delta_percentile_from_histogram(hist, percentile):
    counts = np.asarray(hist or [], dtype=np.float64)
    if counts.size <= 0 or float(counts.sum()) <= 0.0:
        return 0.0
    total = float(counts.sum())
    threshold = max(1.0, math.ceil((float(percentile) / 100.0) * total))
    index = int(np.searchsorted(np.cumsum(counts), threshold, side='left'))
    index = max(0, min(index, counts.size - 1))
    width = (_Q_DELTA_HIST_MAX - _Q_DELTA_HIST_MIN) / float(counts.size)
    bin_low = float(_Q_DELTA_HIST_MIN + index * width)
    bin_high = float(bin_low + width)
    if bin_low <= 0.0 <= bin_high:
        return 0.0
    return float(bin_low + 0.5 * width)


def _debug_scope(config, name):
    root = config.get('debug', {}) or {}
    if not isinstance(root, dict):
        return {}, {}, False
    scoped = root.get(name, {}) or {}
    if not isinstance(scoped, dict):
        scoped = {}
    return root, scoped, bool(root.get('enabled', False))


def _debug_bool(config, scope_name, key, default=False):
    root, scoped, enabled = _debug_scope(config, scope_name)
    return bool(enabled and scoped.get(key, root.get(key, default)))


def _debug_nested(config, scope_name, nested_name):
    root, scoped, enabled = _debug_scope(config, scope_name)
    nested = scoped.get(nested_name, {}) or {}
    if not isinstance(nested, dict):
        nested = {}
    return root, nested, enabled


class _InductorSMWarningFilter(logging.Filter):
    """Drop noisy Inductor warning for low-SM GPUs during self-play."""

    _needle = "Not enough SMs to use max_autotune_gemm mode"

    def filter(self, record):
        try:
            return self._needle not in record.getMessage()
        except Exception:
            return True


class _SyzygyOracle:
    def __init__(self, paths, max_pieces=6):
        self.max_pieces = max(2, int(max_pieces))
        self.paths = tuple(str(p) for p in paths)
        self._tb = None

        if not self.paths:
            return

        try:
            import chess.syzygy
        except Exception:
            return

        tablebase = None
        for idx, path in enumerate(self.paths):
            try:
                if idx == 0:
                    tablebase = chess.syzygy.open_tablebase(path)
                else:
                    tablebase.add_directory(path)
            except Exception:
                continue
        self._tb = tablebase

    @property
    def enabled(self):
        return self._tb is not None

    def can_probe(self, board):
        return self.enabled and len(board.piece_map()) <= self.max_pieces

    def probe_wdl(self, board):
        if not self.can_probe(board):
            return None
        try:
            return int(self._tb.probe_wdl(board))
        except Exception:
            return None

    def result_for_board(self, board):
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if wdl > 0:
            return '1-0' if board.turn == chess.WHITE else '0-1'
        if wdl < 0:
            return '0-1' if board.turn == chess.WHITE else '1-0'
        return '1/2-1/2'


def _resolve_syzygy_paths(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_paths = rl_cfg.get('syzygy_paths', []) or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]

    chess_root = Path(__file__).resolve().parents[1]
    resolved = []
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        path_str = str(raw_path).strip()
        if not path_str:
            continue
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = chess_root / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_dir():
            resolved.append(str(candidate))
    return tuple(resolved)


def _get_syzygy_oracle(config):
    rl_cfg = config.get('reinforcement_learning', {})
    if not bool(rl_cfg.get('syzygy_enabled', False)):
        return None
    paths = _resolve_syzygy_paths(config)
    if not paths:
        return None
    max_pieces = int(rl_cfg.get('syzygy_max_pieces', 6) or 6)
    cache_key = (paths, max_pieces)
    oracle = _SYZYGY_ORACLE_CACHE.get(cache_key)
    if oracle is None:
        oracle = _SyzygyOracle(paths, max_pieces=max_pieces)
        _SYZYGY_ORACLE_CACHE[cache_key] = oracle
    return oracle if oracle.enabled else None


class _SelfPlayCompileLock:
    def __init__(self, lock_path, timeout_s=900.0, poll_s=0.2, stale_after_s=1800.0):
        self.lock_path = Path(lock_path)
        self.timeout_s = max(1.0, float(timeout_s))
        self.poll_s = max(0.05, float(poll_s))
        self.stale_after_s = max(self.timeout_s, float(stale_after_s))
        self._fd = None

    def __enter__(self):
        start_t = time.perf_counter()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = f"{os.getpid()} {time.time():.6f}\n".encode("ascii", errors="ignore")
                os.write(self._fd, payload)
                return self
            except FileExistsError:
                try:
                    stat = self.lock_path.stat()
                    age_s = max(0.0, time.time() - float(stat.st_mtime))
                    if age_s > self.stale_after_s:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                if (time.perf_counter() - start_t) >= self.timeout_s:
                    raise TimeoutError(f"Timed out waiting for compile lock: {self.lock_path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
        finally:
            self._fd = None
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass


def _resolve_selfplay_compile_enabled(config, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return False
    rl_cfg = config.get('reinforcement_learning', {})
    if 'self_play_use_compile' in rl_cfg:
        return bool(rl_cfg.get('self_play_use_compile', False))
    return bool(config.get('hardware', {}).get('use_compile', False))


def _resolve_selfplay_compile_use_lock(config):
    rl_cfg = config.get('reinforcement_learning', {})
    if 'self_play_compile_use_lock' in rl_cfg:
        return bool(rl_cfg.get('self_play_compile_use_lock', False))
    # Default to parallel compile across workers.
    return False


def _configure_selfplay_compile_cache(rank, device):
    if device.type != 'cuda':
        return None
    root = Path(tempfile.gettempdir()) / "play_with_ai_games" / "torch_compile_cache"
    worker_dir = root / f"cuda_worker_{int(rank)}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(worker_dir / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(worker_dir / "triton")
    return root


def _configure_inductor_for_selfplay(config):
    rl_cfg = config.get('reinforcement_learning', {})

    # Allow explicit control of Inductor compile worker threads.
    compile_threads_raw = rl_cfg.get('self_play_compile_threads', None)
    if compile_threads_raw is not None:
        try:
            compile_threads = int(compile_threads_raw)
            if compile_threads > 0:
                os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
        except Exception:
            pass

    # Silence the low-SM max_autotune warning in self-play logs by default.
    suppress_sm_warning = bool(
        rl_cfg.get('self_play_suppress_inductor_sm_warning', True)
    )
    if not suppress_sm_warning:
        return

    try:
        from torch._inductor import config as inductor_config
        inductor_config.max_autotune_gemm = False
    except Exception:
        pass

    try:
        logger = logging.getLogger("torch._inductor.utils")
        has_filter = any(
            isinstance(existing, _InductorSMWarningFilter)
            for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(_InductorSMWarningFilter())
    except Exception:
        pass


def _maybe_compile_selfplay_model(model, config, device, rank, model_label="learner"):
    if not _resolve_selfplay_compile_enabled(config, device):
        return model
    if not hasattr(torch, "compile"):
        return model

    rl_cfg = config.get('reinforcement_learning', {})
    compile_root = _configure_selfplay_compile_cache(rank, device)
    lock_path = (
        compile_root / _SELFPLAY_COMPILE_LOCKFILE
        if compile_root is not None
        else Path(tempfile.gettempdir()) / _SELFPLAY_COMPILE_LOCKFILE
    )
    timeout_s = float(rl_cfg.get('self_play_compile_lock_timeout_s', 900.0) or 900.0)
    use_compile_lock = _resolve_selfplay_compile_use_lock(config)

    history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
    expected_input_planes = 16 * (1 + history_positions)
    use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
    use_bfloat16 = bool(config.get('hardware', {}).get('use_bfloat16', False))
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    if bool(rl_cfg.get('self_play_compile_reset_dynamo', False)):
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    try:
        def _compile_and_warmup(target_model):
            compiled_model = torch.compile(target_model, mode='default', dynamic=True)
            dummy_input = torch.zeros(
                1,
                expected_input_planes,
                8,
                8,
                device=device,
                dtype=torch.float32,
            )
            if device.type == 'cuda':
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                with torch.autocast(
                    device_type='cuda',
                    enabled=use_amp,
                    dtype=amp_dtype,
                ):
                    compiled_model(dummy_input, apply_log_softmax=False)
            return compiled_model

        if use_compile_lock:
            with _SelfPlayCompileLock(lock_path, timeout_s=timeout_s):
                compiled = _compile_and_warmup(model)
        else:
            compiled = _compile_and_warmup(model)

        return compiled
    except Exception as exc:
        print(
            f"WARNING: Self-play worker {rank}: torch.compile skipped for {model_label} "
            f"({type(exc).__name__}: {exc})"
        )
        return model


def _copy_board_fast(board):
    """Copy board state without move stack/history baggage."""
    try:
        return board.copy(stack=False)
    except TypeError:
        return board.copy()


def _board_position_key(board):
    """
    Fast board identity for tree reuse.
    """
    return board._transposition_key()


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

    __slots__ = (
        "moves",
        "nodes",
        "priors",
        "base_priors",
        "visit_counts",
        "total_counts",
        "value_sums",
        "virtual_losses",
        "virtual_losses_f32",
        "explored_flags",
        "selection_scores",
        "ucb_buffer",
        "_move_to_child",
    )

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
        self.nodes = [None] * child_count
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

    def iter_nodes(self):
        for idx, move in enumerate(self.moves):
            child = self.nodes[idx]
            if child is not None:
                yield move, child

    def _get_move_index(self, move):
        move_to_child = self._move_to_child
        if move_to_child is None:
            move_to_child = {child_move: idx for idx, child_move in enumerate(self.moves)}
            self._move_to_child = move_to_child
        return move_to_child.get(move)

    def get_or_create_child(self, parent, idx):
        idx = int(idx)
        child = self.nodes[idx]
        if child is not None:
            return child

        child = MCTSNode(
            board=None,
            parent=parent,
            move=self.moves[idx],
            prior=float(self.priors[idx]),
            copy_board=False,
        )
        child.base_prior = float(self.base_priors[idx])
        child.parent_edge_index = idx
        self.nodes[idx] = child
        return child

    def get_child_for_move(self, parent, move):
        idx = self._get_move_index(move)
        if idx is None:
            return None
        return self.get_or_create_child(parent, idx)

    def visit_dict(self):
        return {
            move: int(visits)
            for move, visits in zip(self.moves, self.visit_counts)
        }

    def __len__(self):
        return len(self.moves)


class MCTSNode:
    """Node in the MCTS tree."""

    __slots__ = (
        "_board",
        "parent",
        "move",
        "prior",
        "base_prior",
        "edges",
        "parent_edge_index",
        "_root_visit_count",
        "_root_value_sum",
        "expanded",
        "_root_virtual_loss",
        "_root_is_explored",
        "_explored_prior_sum",
        "_position_key_cache",
        "_is_game_over",
        "_board_tensor",
        "_history_tensor_pair",
        "_legal_moves",
        "_legal_indices",
        "selection_prior_temperature",
        "selection_prior_uniform_mix",
    )

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
        self._history_tensor_pair = None
        self._legal_moves = None
        self._legal_indices = None
        self.selection_prior_temperature = 1.0
        self.selection_prior_uniform_mix = 0.0

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

    def detach_as_root(self):
        """Promote this child to root while preserving edge-owned search stats."""
        if self._board is None and self.parent is not None:
            _ = self.board
        if self._has_parent_edge():
            idx = int(self.parent_edge_index)
            edges = self.parent.edges
            self._root_visit_count = int(edges.visit_counts[idx])
            self._root_value_sum = float(edges.value_sums[idx])
            self._root_virtual_loss = 0
            self._root_is_explored = bool(edges.explored_flags[idx])
        self.parent = None
        self.parent_edge_index = -1
        return self

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
        self.expanded = True
        self._board_tensor = None
        self._legal_moves = None
        self._legal_indices = None
        self._position_key_cache = None

    def iter_child_nodes(self):
        return self.edges.iter_nodes()

    def get_child_for_move(self, move):
        return self.edges.get_child_for_move(self, move)

    def child_visit_dict(self):
        return self.edges.visit_dict()

    def get_position_key(self):
        if self._position_key_cache is None:
            self._position_key_cache = _board_position_key(self.board)
        return self._position_key_cache

    def get_legal_moves_and_indices(self):
        if self._legal_moves is None or self._legal_indices is None:
            board = self.board
            is_black_turn = board.turn == chess.BLACK
            legal_moves = tuple(board.legal_moves)
            legal_indices = np.empty(len(legal_moves), dtype=np.int32)
            move_to_idx = _move_to_index_cached
            for idx, move in enumerate(legal_moves):
                legal_indices[idx] = move_to_idx(
                    move.from_square,
                    move.to_square,
                    move.promotion or 0,
                    is_black_turn,
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


def _pack_positions_for_transfer(positions, max_policy_targets=None):
    """
    Pack self-play positions into batched tensors for queue transport.
    """
    if not positions:
        return None

    batch_size = len(positions)
    boards = torch.stack([pos[0] for pos in positions]).contiguous()
    values = torch.stack([pos[3] for pos in positions]).contiguous()
    max_len = max(int(pos[1].numel()) for pos in positions)
    if max_policy_targets is not None:
        max_len = min(max_len, max(1, int(max_policy_targets)))

    policy_indices = torch.full((batch_size, max_len), -1, dtype=torch.int16)
    policy_values = torch.zeros((batch_size, max_len), dtype=torch.float32)
    policy_lengths = torch.zeros((batch_size,), dtype=torch.int16)
    importance_scores = torch.zeros((batch_size,), dtype=torch.float32)
    policy_weights = torch.ones((batch_size,), dtype=torch.float32)
    value_weights = torch.ones((batch_size,), dtype=torch.float32)
    source_codes = torch.zeros((batch_size,), dtype=torch.int8)

    for row_idx, pos in enumerate(positions):
        _, indices, probs, _ = pos[:4]
        count = int(indices.numel())
        if max_policy_targets is not None:
            count = min(count, max(1, int(max_policy_targets)))
        importance_scores[row_idx] = float(pos[4]) if len(pos) > 4 else 0.0
        policy_weights[row_idx] = float(pos[5]) if len(pos) > 5 else 1.0
        value_weights[row_idx] = float(pos[6]) if len(pos) > 6 else 1.0
        source_codes[row_idx] = int(pos[7]) if len(pos) > 7 else _REPLAY_SOURCE_UNKNOWN
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
        'importance_scores': importance_scores,
        'policy_weights': policy_weights,
        'value_weights': value_weights,
        'source_codes': source_codes,
        'num_positions': batch_size,
    }


def _resolve_replay_max_policy_targets(config):
    rl_cfg = config.get('reinforcement_learning', {})
    raw_value = rl_cfg.get('replay_max_policy_targets', None)
    if raw_value is None:
        raw_value = rl_cfg.get('policy_target_pruning_max_moves', _DEFAULT_REPLAY_MAX_POLICY_TARGETS)
    try:
        return max(1, int(raw_value))
    except Exception:
        return _DEFAULT_REPLAY_MAX_POLICY_TARGETS


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
        rl_cfg = config['reinforcement_learning']

        self.c_puct_base = float(
            rl_cfg['mcts_c_puct_base']
        )
        self.c_puct_init = float(
            rl_cfg['mcts_c_puct_init']
        )
        self.c_puct_max = float(rl_cfg['mcts_c_puct_max'])
        self.use_fpu = bool(rl_cfg.get('mcts_use_fpu', True))
        self.fpu_reduction = float(
            rl_cfg.get('mcts_fpu_reduction', 0.30)
        )
        self.fpu_absolute = rl_cfg.get('mcts_fpu_absolute', None)
        self.q_value_scale = 0.0
        self.q_selection_weight = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_q_selection_weight', 1.0)),
        )
        self.q_selection_floor = 0.0
        self.q_centered = bool(config['reinforcement_learning'].get('mcts_q_centered', False))
        self.q_min_child_visits = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_q_min_child_visits', 0.0)),
        )
        self.q_parent_visit_warmup = max(
            1.0,
            float(config['reinforcement_learning'].get('mcts_q_parent_visit_warmup', 1.0)),
        )
        self.q_tanh_scale = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_q_tanh_scale', 0.0)),
        )
        self.q_max_abs = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_q_max_abs', 0.0)),
        )
        self.q_min_fullmove = max(
            1,
            int(config['reinforcement_learning'].get('mcts_q_min_fullmove', 1)),
        )
        self.q_full_weight_fullmove = max(
            self.q_min_fullmove,
            int(config['reinforcement_learning'].get('mcts_q_full_weight_fullmove', self.q_min_fullmove)),
        )
        self.eval_batch_size = config['reinforcement_learning'].get('mcts_batch_size', 32)
        self.scout_simulations = max(1, int(rl_cfg['mcts_scout_simulations']))
        self.scout_check_interval = max(1, int(rl_cfg['mcts_scout_check_interval']))
        self.scout_low_branching_moves = max(2, int(rl_cfg['mcts_scout_low_branching_moves']))
        self.scout_endgame_piece_count = max(2, int(rl_cfg['mcts_scout_endgame_piece_count']))
        self.scout_easy_top_visit_prob = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_easy_top_visit_prob'])),
        )
        self.scout_easy_visit_gap = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_easy_visit_gap'])),
        )
        self.scout_easy_max_entropy = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_easy_max_entropy'])),
        )
        self.scout_easy_min_explored_prior_mass = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_easy_min_explored_prior_mass'])),
        )
        self.scout_easy_min_visited_moves = max(1, int(rl_cfg['mcts_scout_easy_min_visited_moves']))
        self.scout_challenge_fraction = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_challenge_fraction'])),
        )
        self.scout_challenge_budget_multiplier = max(
            1.0,
            float(rl_cfg['mcts_scout_challenge_budget_multiplier']),
        )
        self.scout_challenge_min_score = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_challenge_min_score'])),
        )
        self.scout_challenge_prior_margin_ref = max(
            1e-6,
            float(rl_cfg['mcts_scout_challenge_prior_margin_ref']),
        )
        self.scout_challenge_prior_top_ref = max(
            1e-6,
            float(rl_cfg['mcts_scout_challenge_prior_top_ref']),
        )
        self.scout_challenge_prior_temperature = max(
            1.0,
            float(rl_cfg['mcts_scout_challenge_prior_temperature']),
        )
        self.scout_challenge_uniform_mix = max(
            0.0,
            min(0.50, float(rl_cfg['mcts_scout_challenge_uniform_mix'])),
        )
        self.scout_challenge_min_budget_fraction = max(
            0.0,
            min(1.0, float(rl_cfg['mcts_scout_challenge_min_budget_fraction'])),
        )
        self.scout_sharp_capture_min_gain = max(
            0.0,
            float(rl_cfg['mcts_scout_sharp_capture_min_gain']),
        )
        # History configuration (POV)
        self.history_positions = config['model'].get('history_positions', 0)
        self.history_storage_dtype = (
            np.float16
            if bool(config['reinforcement_learning'].get('self_play_history_fp16', True))
            else np.float32
        )

        # Tree reuse
        self.reuse_tree = config['reinforcement_learning'].get('mcts_reuse_tree', True)
        self.cache_node_tensors = False
        self.cache_history_tensors = bool(
            config['reinforcement_learning'].get('mcts_cache_history_tensors', True)
        )

        # Dirichlet noise params
        self.dirichlet_alpha = config['reinforcement_learning'].get('mcts_dirichlet_alpha', 0.3)
        self.dirichlet_weight = config['reinforcement_learning'].get('mcts_dirichlet_weight', 0.0)
        self.tactical_priors_enabled = bool(
            config['reinforcement_learning'].get('mcts_tactical_priors_enabled', False)
        )
        self.tactical_capture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_capture_bonus', 0.0)),
        )
        self.tactical_winning_capture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_winning_capture_bonus', 0.0)),
        )
        self.tactical_check_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_check_bonus', 0.0)),
        )
        self.tactical_promotion_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_promotion_bonus', 0.0)),
        )
        self.tactical_recapture_bonus = max(
            0.0,
            float(config['reinforcement_learning'].get('mcts_tactical_recapture_bonus', 0.0)),
        )

        # Inference optimization (AMP on GPU)
        self.use_amp = config.get('hardware', {}).get('use_amp', False) and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
        self._empty_history_tensor = _EMPTY_HISTORY_TENSOR.astype(self.history_storage_dtype, copy=False)
        self._board_planes = int(self._empty_history_tensor.shape[0])
        self._history_planes = int(self.history_positions) * self._board_planes
        self._input_planes = self._board_planes + self._history_planes
        self._legal_index_scratch = {}
        self._legal_index_tensor_scratch = {}
        self._board_input_scratch = {}
        self._board_input_tensor_scratch = {}
        self._use_pinned_staging = bool(
            config.get('hardware', {}).get('pin_memory', True)
            and self.device.type == 'cuda'
            and torch.cuda.is_available()
        )
        # Keep lightweight performance counters always on so details/performance CSV/PNG
        # stays useful even when verbose RL debug profiling is disabled. Detailed MCTS
        # stage timers are opt-in because they run inside the tight search loop.
        self.profile_enabled = True
        self.profile_detail_enabled = _debug_bool(
            config,
            'rl',
            'profile_mcts_detail',
            _debug_bool(config, 'rl', 'profile_training', False),
        )
        debug_root = config.get('debug', {}) or {}
        debug_rl = debug_root.get('rl', {}) or {}
        try:
            self.profile_sample_rate = int(debug_rl.get('profile_mcts_sample_rate', debug_root.get('profile_mcts_sample_rate', 64)))
        except (TypeError, ValueError):
            self.profile_sample_rate = 64
        self.profile_sample_rate = max(0, int(self.profile_sample_rate))
        self._profile_sample_counts = {}
        self.cuda_stage_profile_enabled = bool(
            self.profile_detail_enabled
            and _debug_bool(config, 'rl', 'profile_training', False)
        )
        self._profile_stats = {}

        self._tactical_capture_enabled = (
            self.tactical_capture_bonus > 0.0
            or self.tactical_winning_capture_bonus > 0.0
        )
        self._tactical_promotion_enabled = self.tactical_promotion_bonus > 0.0
        self._tactical_check_enabled = self.tactical_check_bonus > 0.0
        self._tactical_recapture_enabled = self.tactical_recapture_bonus > 0.0
        self.reset_profile_stats()

    def reset_profile_stats(self):
        self._profile_sample_counts = {}
        self._profile_stats = {
            'search_many_time': 0.0,
            'search_many_calls': 0,
            'search_root_setup_time': 0.0,
            'search_selection_time': 0.0,
            'search_backprop_time': 0.0,
            'search_scout_classify_time': 0.0,
            'search_metadata_time': 0.0,
            'batch_expand_eval_time': 0.0,
            'batch_expand_eval_calls': 0,
            'batch_expand_dedup_terminal_time': 0.0,
            'batch_expand_legal_moves_time': 0.0,
            'batch_expand_tensor_pack_time': 0.0,
            'batch_expand_history_time': 0.0,
            'batch_expand_input_pack_time': 0.0,
            'batch_expand_legal_index_pack_time': 0.0,
            'batch_expand_cpu_policy_time': 0.0,
            'batch_expand_value_fanout_time': 0.0,
            'board_to_tensor_time': 0.0,
            'board_to_tensor_calls': 0,
            'nn_inference_time': 0.0,
            'nn_inference_calls': 0,
            'nn_inference_batch_items': 0,
            'nn_h2d_time': 0.0,
            'nn_gpu_forward_time': 0.0,
            'nn_gpu_postprocess_time': 0.0,
            'nn_d2h_time': 0.0,
            'nn_legal_move_items': 0,
        }

    def _profile_add(self, key, value):
        self._profile_stats[key] = float(self._profile_stats.get(key, 0.0)) + float(value)

    def _profile_inc(self, key, value=1):
        self._profile_stats[key] = int(self._profile_stats.get(key, 0)) + int(value)

    def _profile_sample_begin(self, key):
        if self.profile_detail_enabled:
            return time.perf_counter(), 1
        if not self.profile_enabled or self.profile_sample_rate <= 0:
            return None, 0
        count = int(self._profile_sample_counts.get(key, 0)) + 1
        self._profile_sample_counts[key] = count
        if count % self.profile_sample_rate != 0:
            return None, 0
        return time.perf_counter(), self.profile_sample_rate

    def _profile_sample_finish(self, key, start_time, scale):
        if start_time is None or scale <= 0:
            return
        self._profile_add(f'{key}_time', (time.perf_counter() - start_time) * float(scale))
        self._profile_inc(f'{key}_calls', int(scale))

    def get_profile_stats(self):
        return dict(self._profile_stats)

    def _board_to_tensor_profiled(self, board, flip_perspective=None, storage_dtype=None):
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor')
        tensor = board_to_tensor(board, flip_perspective=flip_perspective)
        self._profile_sample_finish('board_to_tensor', sample_t0, sample_scale)
        if storage_dtype is not None:
            tensor = tensor.astype(storage_dtype, copy=False)
        return tensor

    def _history_tensor_pair_for_board(self, board):
        storage_dtype = self.history_storage_dtype
        sample_t0, sample_scale = self._profile_sample_begin('board_to_tensor_pair')
        white_tensor = board_to_tensor(board, flip_perspective=False).astype(storage_dtype, copy=False)
        black_tensor = board_to_tensor(board, flip_perspective=True).astype(storage_dtype, copy=False)
        if sample_t0 is not None and sample_scale > 0:
            self._profile_add('board_to_tensor_time', (time.perf_counter() - sample_t0) * float(sample_scale))
            self._profile_inc('board_to_tensor_calls', int(sample_scale) * 2)
        return (white_tensor, black_tensor)

    def _history_tensor_pair_for_node(self, node):
        cached = getattr(node, "_history_tensor_pair", None)
        if cached is None or not self.cache_history_tensors:
            cached = self._history_tensor_pair_for_board(node.board)
            if self.cache_history_tensors:
                node._history_tensor_pair = cached
        return cached

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

        return self._history_tensor_pair_for_board(board_obj)

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
            return self._board_to_tensor_profiled(current_board)

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
            current_tensor = self._board_to_tensor_profiled(current_board)
        history_tensors.append(current_tensor)

        return np.concatenate(history_tensors, axis=0)

    def _history_tensor_for_entry(self, entry, use_black_pov):
        if self._is_encoded_history_entry(entry):
            return entry[1] if use_black_pov else entry[0]

        if isinstance(entry, MCTSNode):
            encoded = self._history_tensor_pair_for_node(entry)
            return encoded[1] if use_black_pov else encoded[0]

        if isinstance(entry, chess.Board):
            return self._board_to_tensor_profiled(
                entry,
                flip_perspective=use_black_pov,
                storage_dtype=self.history_storage_dtype,
            )

        encoded = self._encode_history_entry(entry)
        return encoded[1] if use_black_pov else encoded[0]

    def _collect_node_history_entries(self, node, board_history):
        """
        Return the last history positions for a search leaf.

        board_history contains real-game positions before the current root.
        For deeper MCTS leaves, parent nodes represent the simulated positions
        immediately before the leaf and must be part of the NN history planes.
        """
        if self.history_positions <= 0:
            return []

        simulated_history = []
        ancestor = node.parent
        while ancestor is not None and len(simulated_history) < self.history_positions:
            simulated_history.append(ancestor)
            ancestor = ancestor.parent
        if simulated_history:
            simulated_history.reverse()

        keep_from_game = max(0, int(self.history_positions) - len(simulated_history))
        real_history = list(board_history[-keep_from_game:]) if board_history and keep_from_game > 0 else []
        return real_history + simulated_history

    def _build_history_prefix_for_node(self, node, board_history):
        if self.history_positions <= 0:
            return None

        use_black_pov = (node.board.turn == chess.BLACK)
        history_entries = self._collect_node_history_entries(node, board_history)
        history_tensors = [
            self._history_tensor_for_entry(entry, use_black_pov)
            for entry in history_entries
        ]

        pad_count = max(0, int(self.history_positions) - len(history_tensors))
        if pad_count:
            history_tensors = [self._empty_history_tensor] * pad_count + history_tensors

        return np.concatenate(history_tensors, axis=0) if history_tensors else None

    def _current_tensor_for_node(self, node):
        if not self.cache_node_tensors:
            return self._board_to_tensor_profiled(node.board)

        cached = getattr(node, "_board_tensor", None)
        if cached is None:
            cached = self._board_to_tensor_profiled(node.board)
            node._board_tensor = cached
        return cached

    def _get_legal_index_scratch(self, batch_size, max_legal_count, dtype=np.int64):
        dtype = np.dtype(dtype)
        key = (int(batch_size), int(max_legal_count), dtype.str)
        scratch = self._legal_index_scratch.get(key)
        if scratch is None:
            scratch = np.empty((batch_size, max_legal_count), dtype=dtype)
            self._legal_index_scratch[key] = scratch
        return scratch

    def _get_legal_index_source_tensor(self, legal_index_matrix):
        if not self._use_pinned_staging:
            return torch.from_numpy(legal_index_matrix)

        key = tuple(int(dim) for dim in legal_index_matrix.shape)
        scratch = self._legal_index_tensor_scratch.get(key)
        if scratch is None:
            scratch = torch.empty(key, dtype=torch.long, pin_memory=True)
            self._legal_index_tensor_scratch[key] = scratch
        scratch.copy_(torch.from_numpy(legal_index_matrix))
        return scratch

    def _get_board_input_scratch(self, batch_size):
        key = int(batch_size)
        scratch = self._board_input_scratch.get(key)
        if scratch is None:
            if self._use_pinned_staging:
                tensor_scratch = torch.empty(
                    (batch_size, self._input_planes, 8, 8),
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self._board_input_tensor_scratch[key] = tensor_scratch
                scratch = tensor_scratch.numpy()
            else:
                scratch = np.empty((batch_size, self._input_planes, 8, 8), dtype=np.float32)
            self._board_input_scratch[key] = scratch
        return scratch

    def _get_board_input_source_tensor(self, boards_np):
        if not self._use_pinned_staging:
            return torch.from_numpy(boards_np)

        key = int(boards_np.shape[0])
        tensor_scratch = self._board_input_tensor_scratch.get(key)
        if tensor_scratch is None:
            raise RuntimeError(f"Missing pinned board input scratch for batch size {key}")
        return tensor_scratch

    @staticmethod
    def _captured_piece_value(board, move):
        if board.is_en_passant(move):
            return _PIECE_VALUES[chess.PAWN]
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None:
            return 0.0
        return float(_PIECE_VALUES.get(captured_piece.piece_type, 0.0))

    @staticmethod
    def _moving_piece_value(board, move):
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is None:
            return 0.0
        return float(_PIECE_VALUES.get(moving_piece.piece_type, 0.0))

    @staticmethod
    def _move_may_give_check_fast(board, move):
        """Cheap geometric prefilter before the expensive python-chess gives_check()."""
        moved_piece = board.piece_at(move.from_square)
        if moved_piece is None:
            return False

        king_square = board.king(not board.turn)
        if king_square is None:
            return False

        target_square = move.to_square
        piece_type = move.promotion or moved_piece.piece_type

        from_row = chess.square_rank(target_square)
        from_col = chess.square_file(target_square)
        king_row = chess.square_rank(king_square)
        king_col = chess.square_file(king_square)
        dr = king_row - from_row
        dc = king_col - from_col
        abs_dr = abs(dr)
        abs_dc = abs(dc)

        if piece_type == chess.KNIGHT:
            return (abs_dr, abs_dc) in {(1, 2), (2, 1)}
        if piece_type == chess.BISHOP:
            return abs_dr == abs_dc
        if piece_type == chess.ROOK:
            return dr == 0 or dc == 0
        if piece_type == chess.QUEEN:
            return dr == 0 or dc == 0 or abs_dr == abs_dc
        if piece_type == chess.KING:
            return max(abs_dr, abs_dc) == 1
        if piece_type == chess.PAWN:
            if moved_piece.color == chess.WHITE:
                return dr == 1 and abs_dc == 1
            return dr == -1 and abs_dc == 1
        return False

    def _tactical_prior_multipliers(self, board, legal_moves):
        if not self.tactical_priors_enabled or not legal_moves:
            return np.ones((len(legal_moves),), dtype=np.float32)

        if not (
            self._tactical_capture_enabled
            or self._tactical_promotion_enabled
            or self._tactical_check_enabled
            or self._tactical_recapture_enabled
        ):
            return np.ones((len(legal_moves),), dtype=np.float32)

        multipliers = np.ones((len(legal_moves),), dtype=np.float32)
        last_move = board.peek() if (self._tactical_recapture_enabled and board.move_stack) else None

        for idx, move in enumerate(legal_moves):
            bonus = 0.0
            if self._tactical_capture_enabled and board.is_capture(move):
                bonus += self.tactical_capture_bonus
                gain = self._captured_piece_value(board, move) - self._moving_piece_value(board, move)
                if gain > 0.0:
                    bonus += self.tactical_winning_capture_bonus * min(1.0, gain / 4.0)
            if self._tactical_promotion_enabled and move.promotion is not None:
                bonus += self.tactical_promotion_bonus

            should_eval_check = (
                self._tactical_check_enabled
                and (
                    bonus > 0.0
                    or (
                        not self._tactical_capture_enabled
                        and not self._tactical_promotion_enabled
                        and not self._tactical_recapture_enabled
                    )
                )
            )
            if should_eval_check:
                try:
                    if self._move_may_give_check_fast(board, move) and board.gives_check(move):
                        bonus += self.tactical_check_bonus
                except Exception:
                    pass
            if self._tactical_recapture_enabled and last_move is not None and move.to_square == last_move.to_square:
                bonus += self.tactical_recapture_bonus
            multipliers[idx] = 1.0 + float(bonus)

        return multipliers

    def _effective_q_selection_weight(self, node):
        effective_q_weight = float(self.q_selection_weight) + float(self.q_value_scale)
        if self.q_selection_floor > 0.0:
            effective_q_weight = max(effective_q_weight, float(self.q_selection_floor))
        if effective_q_weight <= 0.0:
            return 0.0
        if self.q_min_fullmove <= 1:
            return effective_q_weight

        try:
            fullmove_number = int(getattr(node.board, 'fullmove_number', 1) or 1)
        except Exception:
            fullmove_number = 1
        if fullmove_number < self.q_min_fullmove:
            return 0.0
        if self.q_full_weight_fullmove > self.q_min_fullmove:
            phase_span = float(self.q_full_weight_fullmove - self.q_min_fullmove)
            phase_progress = float(fullmove_number - self.q_min_fullmove) / phase_span
            effective_q_weight *= max(0.0, min(1.0, phase_progress))
        return effective_q_weight

    def _select_child(self, node):
        """Select child with highest UCB score (optimized)"""
        edges = node.edges
        if not edges.moves:
            return None
        if len(edges.moves) == 1:
            return edges.get_or_create_child(node, 0)

        # Precalculate parent term to avoid doing it for every child
        node_visit_count = node.visit_count
        node_virtual_loss = node.virtual_loss
        parent_visits = node_visit_count + node_virtual_loss
        parent_sqrt = math.sqrt(parent_visits + 1)
        c_puct = math.log((parent_visits + self.c_puct_base + 1.0) / self.c_puct_base) + self.c_puct_init
        c_puct = min(c_puct, self.c_puct_max)
        effective_q_weight = self._effective_q_selection_weight(node)

        cv = edges.total_counts
        q_values = edges.selection_scores
        if effective_q_weight <= 0.0:
            q_values.fill(0.0)
        else:
            fpu_value = 0.0
            if self.use_fpu:
                if self.fpu_absolute is not None:
                    fpu_value = float(self.fpu_absolute)
                elif parent_visits > 0:
                    parent_q = (node.value_sum - node_virtual_loss) / parent_visits
                    explored_prior = float(node._explored_prior_sum)
                    if explored_prior < 0.0:
                        explored_prior = 0.0
                    elif explored_prior > 1.0:
                        explored_prior = 1.0
                    unexplored_prior = 1.0 - explored_prior
                    if unexplored_prior < 0.0:
                        unexplored_prior = 0.0
                    fpu_value = parent_q - self.fpu_reduction * math.sqrt(unexplored_prior)

            explored_mask = edges.explored_flags
            if self.q_centered:
                transformed_q = q_values
                # Keep the configured FPU baseline for unvisited / low-visit
                # children, then overwrite only stable children with centered Q.
                # Previously centered-Q mode zeroed the whole vector, which made
                # mcts_use_fpu effectively meaningless whenever q_centered=true.
                transformed_q.fill(float(fpu_value) * float(effective_q_weight))
                stable_mask = explored_mask
                if self.q_min_child_visits > 0.0:
                    stable_mask = explored_mask & (cv >= float(self.q_min_child_visits))
                if stable_mask.any():
                    stable_q = (
                        -edges.value_sums[stable_mask] - edges.virtual_losses_f32[stable_mask]
                    ) / cv[stable_mask]
                    stable_counts = cv[stable_mask]
                    q_center = float(np.average(stable_q, weights=np.maximum(stable_counts, 1.0)))
                    centered_q = stable_q - q_center
                    if self.q_tanh_scale > 0.0:
                        centered_q = np.tanh(centered_q / float(self.q_tanh_scale))
                    parent_gate = min(1.0, float(parent_visits) / float(self.q_parent_visit_warmup))
                    if self.q_min_child_visits > 0.0:
                        visit_gate = np.minimum(1.0, stable_counts / float(self.q_min_child_visits))
                    else:
                        visit_gate = 1.0
                    centered_q = centered_q * float(effective_q_weight) * float(parent_gate) * visit_gate
                    if self.q_max_abs > 0.0:
                        centered_q = np.clip(centered_q, -float(self.q_max_abs), float(self.q_max_abs))
                    transformed_q[stable_mask] = centered_q.astype(np.float32, copy=False)
            else:
                q_values.fill(fpu_value)
                if explored_mask.any():
                    # Edge value_sums live on child nodes, so they are from the child
                    # side-to-move perspective. The parent must negate them when
                    # deciding which move is good for the current player.
                    q_values[explored_mask] = (
                        -edges.value_sums[explored_mask] - edges.virtual_losses_f32[explored_mask]
                    ) / cv[explored_mask]
                if effective_q_weight != 1.0:
                    q_values *= float(effective_q_weight)

        u_values = edges.ucb_buffer
        prior_temperature = float(getattr(node, 'selection_prior_temperature', 1.0) or 1.0)
        prior_uniform_mix = float(getattr(node, 'selection_prior_uniform_mix', 0.0) or 0.0)
        if prior_temperature > 1.0001 or prior_uniform_mix > 1e-8:
            np.maximum(edges.priors, 1e-12, out=u_values)
            if prior_temperature > 1.0001:
                np.power(u_values, 1.0 / prior_temperature, out=u_values)
            prior_sum = float(u_values.sum())
            if prior_sum > 0.0:
                u_values /= prior_sum
            else:
                u_values.fill(1.0 / float(max(1, len(edges.moves))))
            if prior_uniform_mix > 1e-8 and len(edges.moves) > 0:
                uniform = 1.0 / float(len(edges.moves))
                np.multiply(u_values, 1.0 - prior_uniform_mix, out=u_values)
                u_values += prior_uniform_mix * uniform
            u_values *= c_puct * parent_sqrt
        else:
            np.multiply(edges.priors, c_puct * parent_sqrt, out=u_values)
        np.divide(u_values, (1.0 + cv), out=u_values)
        np.add(q_values, u_values, out=q_values)
        best_idx = int(q_values.argmax())
        return edges.get_or_create_child(node, best_idx)

    def _apply_root_noise(self, node):
        """Apply fresh Dirichlet noise to an already expanded root node."""
        if self.dirichlet_weight <= 0 or not node.edges.moves:
            return

        edges = node.edges
        child_count = len(edges.moves)
        noise = np.random.dirichlet([self.dirichlet_alpha] * child_count)
        mix = self.dirichlet_weight

        for idx, noise_value in enumerate(noise):
            old_prior = float(edges.priors[idx])
            new_prior = (1.0 - mix) * float(edges.base_priors[idx]) + mix * float(noise_value)
            edges.priors[idx] = new_prior
            if edges.explored_flags[idx]:
                node._explored_prior_sum += float(new_prior - old_prior)
            child = edges.nodes[idx]
            if child is not None:
                child.prior = new_prior

    def _backpropagate(self, search_path, value):
        """Backpropagate value"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _summarize_root_search(self, root, simulation_budget, initial_root_visits=0):
        budget = max(1, int(simulation_budget))
        total_root_visits = 0 if root is None else int(getattr(root, 'visit_count', 0) or 0)
        used = max(0, total_root_visits - max(0, int(initial_root_visits)))
        summary = {
            'simulations_used': used,
            'simulation_budget': budget,
            'top_visit_prob': 0.0,
            'visit_gap': 0.0,
            'visit_entropy': 1.0,
            'prior_mcts_agree': None,
            'prior_top_visit_prob': None,
            'prior_top_visit_rank': None,
            'mcts_top_prior_prob': None,
            'mcts_q_delta': None,
            'mcts_changed_to_lower_q': None,
            'mcts_policy_kl': None,
            'explored_prior_mass': 0.0,
            'unexplored_prior_mass': 1.0,
            'visited_move_count': 0,
            'legal_move_count': 0,
            'root_value': 0.0,
            'stopped_early': used < budget,
            'scout_stop_reason': 'budget' if used >= budget else 'unknown',
            'policy_weight': float(max(0.0, min(1.0, used / float(budget)))),
        }
        if root is None or not root.expanded or root.edges is None:
            return summary

        all_visits = root.edges.visit_counts.astype(np.float32, copy=False)
        priors = root.edges.base_priors.astype(np.float32, copy=False)
        legal_count = int(all_visits.size)
        visited_mask = all_visits > 0.0
        visited_count = int(visited_mask.sum())
        prior_total = float(priors.sum()) if priors.size == all_visits.size else 0.0
        explored_prior_mass = (
            float(priors[visited_mask].sum()) / prior_total
            if prior_total > 0.0 and priors.size == all_visits.size
            else 0.0
        )
        explored_prior_mass = float(max(0.0, min(1.0, explored_prior_mass)))
        summary['legal_move_count'] = legal_count
        summary['visited_move_count'] = visited_count
        summary['explored_prior_mass'] = explored_prior_mass
        summary['unexplored_prior_mass'] = float(max(0.0, 1.0 - explored_prior_mass))

        visits = all_visits[visited_mask]
        total = float(visits.sum())
        if total <= 0.0:
            return summary

        root_value = 0.0
        root_visits = int(getattr(root, 'visit_count', 0) or 0)
        if root_visits > 0:
            root_value = float(root.value_sum / max(1, root_visits))
            root_value = float(max(-1.0, min(1.0, root_value)))

        summary['root_value'] = root_value

        visits.sort()
        top = float(visits[-1])
        second = float(visits[-2]) if visits.size > 1 else 0.0
        probs = visits / total
        entropy = 0.0
        if probs.size > 1:
            entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
            entropy /= float(np.log(probs.size))

        summary['top_visit_prob'] = top / total
        summary['visit_gap'] = max(0.0, (top - second) / total)
        summary['visit_entropy'] = float(max(0.0, min(1.0, entropy)))

        if all_visits.size > 0 and priors.size == all_visits.size and prior_total > 0.0:
            prior_probs = priors / prior_total
            visit_probs = all_visits / max(1e-8, float(all_visits.sum()))
            prior_top_idx = int(np.argmax(prior_probs))
            mcts_top_idx = int(np.argmax(all_visits))
            summary['prior_mcts_agree'] = 1.0 if prior_top_idx == mcts_top_idx else 0.0
            summary['prior_top_visit_prob'] = float(visit_probs[prior_top_idx])
            summary['mcts_top_prior_prob'] = float(prior_probs[mcts_top_idx])
            rank_order = np.argsort(-all_visits)
            rank_matches = np.where(rank_order == prior_top_idx)[0]
            if rank_matches.size > 0:
                summary['prior_top_visit_rank'] = int(rank_matches[0]) + 1
            active = all_visits > 0.0
            if active.any():
                kl_terms = visit_probs[active] * (
                    np.log(np.clip(visit_probs[active], 1e-12, 1.0))
                    - np.log(np.clip(prior_probs[active], 1e-12, 1.0))
                )
                summary['mcts_policy_kl'] = float(max(0.0, float(kl_terms.sum())))
            if all_visits[prior_top_idx] > 0.0 and all_visits[mcts_top_idx] > 0.0:
                prior_q = -float(root.edges.value_sums[prior_top_idx]) / max(1.0, float(all_visits[prior_top_idx]))
                mcts_q = -float(root.edges.value_sums[mcts_top_idx]) / max(1.0, float(all_visits[mcts_top_idx]))
                q_delta = float(max(-2.0, min(2.0, mcts_q - prior_q)))
                summary['mcts_q_delta'] = q_delta
                summary['mcts_changed_to_lower_q'] = (
                    1.0
                    if prior_top_idx != mcts_top_idx and q_delta < -0.02
                    else 0.0
                )
        return summary

    def _stable_fraction_for_root(self, root):
        if root is None:
            return 1.0
        try:
            key = root.get_position_key()
        except Exception:
            key = None
        if key is None:
            return 1.0
        digest = hashlib.blake2b(str(key).encode('utf-8'), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder='little', signed=False)
        return float(value) / float((1 << 64) - 1)

    def _set_root_challenge_selection(self, root, enabled):
        if root is None:
            return
        if enabled:
            root.selection_prior_temperature = float(self.scout_challenge_prior_temperature)
            root.selection_prior_uniform_mix = float(self.scout_challenge_uniform_mix)
        else:
            root.selection_prior_temperature = 1.0
            root.selection_prior_uniform_mix = 0.0

    def _scout_prior_uncertainty_score(self, root, move_count=None):
        if root is None or not root.expanded or root.edges is None:
            return 0.0
        priors = np.asarray(root.edges.base_priors, dtype=np.float32)
        if priors.size <= 1:
            return 0.0
        prior_total = float(priors.sum())
        if prior_total <= 0.0:
            return 0.0

        prior_probs = priors / prior_total
        sorted_priors = np.sort(prior_probs)
        top = float(sorted_priors[-1])
        second = float(sorted_priors[-2]) if sorted_priors.size > 1 else 0.0
        margin = max(0.0, top - second)
        entropy = 0.0
        if prior_probs.size > 1:
            entropy = float(-(prior_probs * np.log(np.clip(prior_probs, 1e-8, 1.0))).sum())
            entropy /= float(np.log(prior_probs.size))
        entropy = float(max(0.0, min(1.0, entropy)))

        legal_count = int(prior_probs.size)
        profile = self._scout_root_profile(root, move_count=move_count)
        if bool(profile.get('forced', False)):
            return 0.0

        margin_score = 1.0 - min(1.0, margin / float(self.scout_challenge_prior_margin_ref))
        top_score = 1.0 - min(1.0, top / float(self.scout_challenge_prior_top_ref))
        branching_span = max(1, 28 - int(self.scout_low_branching_moves))
        branching_score = max(
            0.0,
            min(1.0, float(legal_count - int(self.scout_low_branching_moves)) / float(branching_span)),
        )
        tactic_score = 1.0 if bool(profile.get('in_check', False) or profile.get('sharp_tactic', False)) else 0.0
        score = (
            0.42 * margin_score
            + 0.24 * entropy
            + 0.18 * top_score
            + 0.10 * branching_score
            + 0.06 * tactic_score
        )
        return float(max(0.0, min(1.0, score)))

    def _move_is_budget_sensitive_tactic(self, board, move, legal_count=None, move_count=None):
        if move.promotion is not None:
            return True
        is_capture = False
        try:
            is_capture = bool(board.is_capture(move))
            if is_capture:
                gain = self._captured_piece_value(board, move) - self._moving_piece_value(board, move)
                if gain >= float(self.scout_sharp_capture_min_gain):
                    return True
        except Exception:
            pass
        try:
            gives_check = bool(self._move_may_give_check_fast(board, move) and board.gives_check(move))
        except Exception:
            return False
        if not gives_check:
            return False
        if is_capture:
            return True
        if legal_count is not None and int(legal_count) <= max(8, int(self.scout_low_branching_moves) * 2):
            return True
        return False

    def _scout_root_profile(self, root, move_count=None):
        legal_count = 0
        piece_count = None
        in_check = False
        sharp_tactic = False
        board = None if root is None else getattr(root, 'board', None)
        if board is not None:
            try:
                in_check = bool(board.is_check())
            except Exception:
                in_check = False
            try:
                piece_count = len(board.piece_map())
            except Exception:
                piece_count = None

        moves = []
        if root is not None and root.expanded and root.edges is not None:
            moves = list(root.edges.moves) if root.edges.moves is not None else []
        elif board is not None:
            try:
                moves = list(board.legal_moves)
            except Exception:
                moves = []
        legal_count = len(moves)
        if board is not None and moves:
            try:
                sharp_tactic = any(
                    self._move_is_budget_sensitive_tactic(
                        board,
                        move,
                        legal_count=legal_count,
                        move_count=move_count,
                    )
                    for move in moves
                )
            except Exception:
                sharp_tactic = False
        return {
            'legal_count': int(legal_count),
            'forced': bool(legal_count <= 1 and legal_count > 0),
            'low_branching': bool(1 < legal_count <= self.scout_low_branching_moves),
            'endgame': bool(piece_count is not None and piece_count <= self.scout_endgame_piece_count),
            'in_check': bool(in_check),
            'sharp_tactic': bool(sharp_tactic),
        }

    def _scout_target_ready(self, root, summary):
        if root is None or not root.expanded or root.edges is None:
            return False
        legal_count = int(summary.get('legal_move_count', 0) or 0)
        if legal_count <= 1:
            return True

        visited_count = int(summary.get('visited_move_count', 0) or 0)
        required_visited = min(max(1, legal_count), int(self.scout_easy_min_visited_moves))
        if visited_count < required_visited:
            return False

        if visited_count >= legal_count:
            return True
        return float(summary.get('explored_prior_mass', 0.0) or 0.0) >= float(self.scout_easy_min_explored_prior_mass)

    def _scout_root_is_easy(self, root, summary):
        if root is None or not root.expanded or root.edges is None:
            return False
        legal_count = int(summary.get('legal_move_count', 0) or 0)
        if legal_count <= 1:
            return True
        if not self._scout_target_ready(root, summary):
            return False
        return bool(
            float(summary.get('top_visit_prob', 0.0) or 0.0) >= float(self.scout_easy_top_visit_prob)
            and float(summary.get('visit_gap', 0.0) or 0.0) >= float(self.scout_easy_visit_gap)
            and float(summary.get('visit_entropy', 1.0) or 1.0) <= float(self.scout_easy_max_entropy)
        )

    def _scout_challenge_score(self, root, summary, move_count=None):
        prior_score = self._scout_prior_uncertainty_score(root, move_count=move_count)
        top_visit_prob = float(summary.get('top_visit_prob', 0.0) or 0.0)
        visit_gap = float(summary.get('visit_gap', 0.0) or 0.0)
        visit_entropy = float(summary.get('visit_entropy', 1.0) or 1.0)
        top_uncertainty = max(
            0.0,
            min(1.0, (float(self.scout_easy_top_visit_prob) - top_visit_prob) / max(1e-6, self.scout_easy_top_visit_prob)),
        )
        gap_uncertainty = max(
            0.0,
            min(1.0, (float(self.scout_easy_visit_gap) - visit_gap) / max(1e-6, self.scout_easy_visit_gap)),
        )
        entropy_pressure = max(
            0.0,
            min(1.0, (visit_entropy - float(self.scout_easy_max_entropy)) / max(1e-6, 1.0 - self.scout_easy_max_entropy)),
        )
        try:
            prior_rank = int(summary.get('prior_top_visit_rank', 1) or 1)
            rank_pressure = max(0.0, min(1.0, float(prior_rank - 1) / 4.0))
        except Exception:
            rank_pressure = 0.0
        q_delta = float(summary.get('mcts_q_delta', 0.0) or 0.0)
        q_pressure = max(0.0, min(1.0, q_delta / 0.12))
        profile = self._scout_root_profile(root, move_count=move_count)
        tactic_pressure = 1.0 if bool(profile.get('in_check', False) or profile.get('sharp_tactic', False)) else 0.0
        score = (
            0.28 * prior_score
            + 0.22 * top_uncertainty
            + 0.18 * gap_uncertainty
            + 0.12 * entropy_pressure
            + 0.10 * rank_pressure
            + 0.06 * q_pressure
            + 0.04 * tactic_pressure
        )
        return float(max(0.0, min(1.0, score)))

    def _classify_scout_root(self, root, summary, move_count=None):
        profile = self._scout_root_profile(root, move_count=move_count)
        if bool(profile.get('forced', False)):
            return 'easy', 0.0
        if self._scout_root_is_easy(root, summary) and not bool(
            profile.get('in_check', False) or profile.get('sharp_tactic', False)
        ):
            return 'easy', 0.0
        challenge_score = self._scout_challenge_score(root, summary, move_count=move_count)
        stable_fraction = self._stable_fraction_for_root(root)
        if (
            challenge_score >= float(self.scout_challenge_min_score)
            and stable_fraction < float(self.scout_challenge_fraction)
        ):
            return 'challenge', challenge_score
        return 'normal', challenge_score

    def _scout_challenge_minimum(self, simulation_budget):
        return max(
            1,
            int(round(float(max(1, int(simulation_budget))) * float(self.scout_challenge_min_budget_fraction))),
        )

    def _scout_locked_stop_reason(self, root, summary, simulation_budget):
        if root is None or not root.expanded or root.edges is None:
            return None
        visits = root.edges.visit_counts.astype(np.int32, copy=False)
        visits = visits[visits > 0]
        if visits.size <= 0:
            return None
        visits.sort()
        top = int(visits[-1])
        second = int(visits[-2]) if visits.size > 1 else 0
        used = int(summary.get('simulations_used', 0) or 0)
        remaining = max(0, int(simulation_budget) - used)
        if (
            (top - second) > (remaining + _SCOUT_LOCKED_MARGIN)
            and float(summary.get('top_visit_prob', 0.0) or 0.0) >= max(0.0, self.scout_easy_top_visit_prob - 0.06)
            and float(summary.get('visit_gap', 0.0) or 0.0) >= max(0.0, self.scout_easy_visit_gap * 0.75)
            and float(summary.get('visit_entropy', 1.0) or 1.0) <= min(1.0, self.scout_easy_max_entropy + 0.08)
        ):
            return 'locked'
        return None

    def search_many(self, game_states, num_simulations, add_root_noise=False, return_search_metadata=False):
        """
        Run MCTS for multiple games and batch leaf evaluations across games.

        Args:
            game_states: list of dicts with keys: board, root, board_history
            num_simulations: simulations per game
        Returns:
            List[Dict[chess.Move, int]] visit counts for each game in order
        """
        if not game_states:
            if return_search_metadata:
                return [], []
            return []
        perf_counter = time.perf_counter
        profile_detail = self.profile_detail_enabled
        search_t0 = perf_counter()

        game_count = len(game_states)
        boards = [None] * game_count
        roots = [None] * game_count
        initial_root_visits = [0] * game_count
        root_synced_flags = [False] * game_count
        needs_root_noise_on_expand = [False] * game_count
        scout_stop_reasons = [None] * game_count
        move_counts = [0] * game_count
        board_histories = [None] * game_count
        is_mapping_state = [False] * game_count

        # Initialize / reuse roots per game.
        # Supports both dict states and packed list states:
        # [board, root, root_synced, board_history].
        root_setup_t0 = perf_counter() if profile_detail else None
        for idx, gs in enumerate(game_states):
            if isinstance(gs, dict):
                is_mapping_state[idx] = True
                board = gs['board']
                root = gs.get('root', None)
                root_synced = bool(gs.get('_root_synced', False))
                board_history = gs.get('board_history', [])
                move_count = int(gs.get('move_count', len(getattr(board, 'move_stack', []) or [])) or 0)
            else:
                board = gs[0]
                root = gs[1] if len(gs) > 1 else None
                root_synced = bool(gs[2]) if len(gs) > 2 else False
                board_history = gs[3] if len(gs) > 3 else []
                move_count = int(gs[4]) if len(gs) > 4 else len(getattr(board, 'move_stack', []) or [])

            boards[idx] = board
            board_histories[idx] = board_history
            move_counts[idx] = max(0, int(move_count))

            if self.reuse_tree and root is not None:
                if root_synced:
                    root_synced = False
                else:
                    target_key = _board_position_key(board)
                    if root.get_position_key() == target_key:
                        pass
                    else:
                        for move in root.edges.moves:
                            child = root.get_child_for_move(move)
                            if child is not None and child.get_position_key() == target_key:
                                _ = child.board  # Ensure board is instantiated
                                root = child.detach_as_root()
                                break
                        else:
                            root = MCTSNode(board)
            else:
                root = MCTSNode(board)
                root_synced = False

            roots[idx] = root
            initial_root_visits[idx] = 0 if root is None else int(getattr(root, 'visit_count', 0) or 0)
            root_synced_flags[idx] = bool(root_synced)
            self._set_root_challenge_selection(root, False)
            if add_root_noise and root.expanded:
                self._apply_root_noise(root)
                needs_root_noise_on_expand[idx] = False
            else:
                needs_root_noise_on_expand[idx] = bool(add_root_noise and not root.expanded)
        if profile_detail:
            self._profile_add('search_root_setup_time', perf_counter() - root_setup_t0)

        base_budget = max(1, int(num_simulations))
        scout_budget = min(base_budget, max(1, int(self.scout_simulations)))
        challenge_budget = max(
            base_budget,
            int(round(float(base_budget) * float(self.scout_challenge_budget_multiplier))),
        )
        simulation_budgets = [base_budget] * game_count
        simulation_budget_extended = [False] * game_count
        scout_challenge_flags = [False] * game_count
        scout_challenge_scores = [None] * game_count
        scout_classes = [None] * game_count

        remaining = list(simulation_budgets)
        total_remaining = int(sum(remaining))
        game_ptr = 0

        while total_remaining > 0:
            # Cap per-game quota per round so that backprop happens before all
            # simulations of a single game are consumed  (fixes: when
            # eval_batch_size >= num_simulations, all sims land in one batch,
            # root is never traversed after expansion, children stay at 0 visits).
            max_remaining_any_game = max(remaining) if remaining else 1
            slots_per_game = max(1, min(
                self.eval_batch_size // max(1, game_count),
                max_remaining_any_game // 4,
                self.scout_check_interval,
            ))
            batch_size = min(self.eval_batch_size, total_remaining,
                             slots_per_game * game_count)
            leaf_nodes = []
            search_paths = []
            leaf_game_indices = []
            selected_this_batch = [0] * game_count

            selection_t0 = perf_counter() if profile_detail else None
            for _ in range(batch_size):
                # Find next game with remaining sims
                found = False
                for _ in range(game_count):
                    current_root = roots[game_ptr]
                    if (
                        remaining[game_ptr] > 0
                        and not (
                            selected_this_batch[game_ptr] > 0
                            and current_root is not None
                            and not current_root.expanded
                        )
                    ):
                        found = True
                        break
                    game_ptr = (game_ptr + 1) % game_count

                if not found:
                    break

                gs_idx = game_ptr
                node = roots[gs_idx]
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
                game_ptr = (game_ptr + 1) % game_count
            if profile_detail:
                self._profile_add('search_selection_time', perf_counter() - selection_t0)

            if not leaf_nodes:
                break

            values = self._batch_expand_and_evaluate(
                leaf_nodes,
                leaf_game_indices,
                board_histories,
                roots,
                needs_root_noise_on_expand,
            )

            backprop_t0 = perf_counter() if profile_detail else None
            for search_path, value in zip(search_paths, values):
                self._backpropagate(search_path, value)
                for node in search_path:
                    node.remove_virtual_loss()
            if profile_detail:
                self._profile_add('search_backprop_time', perf_counter() - backprop_t0)

            scout_t0 = perf_counter() if profile_detail else None
            for idx, root in enumerate(roots):
                if remaining[idx] <= 0:
                    continue
                if selected_this_batch[idx] <= 0:
                    continue
                if root is None or not root.expanded:
                    continue
                used = max(
                    0,
                    int(getattr(root, 'visit_count', 0) or 0) - int(initial_root_visits[idx]),
                )
                if used < scout_budget:
                    continue
                check_interval = max(1, int(self.scout_check_interval))
                if used % check_interval != 0 and remaining[idx] > check_interval:
                    continue
                summary = self._summarize_root_search(
                    root,
                    simulation_budgets[idx],
                    initial_root_visits=initial_root_visits[idx],
                )
                scout_class = scout_classes[idx]
                if scout_class is None:
                    scout_class, challenge_score = self._classify_scout_root(
                        root,
                        summary,
                        move_count=move_counts[idx],
                    )
                    scout_classes[idx] = scout_class
                    scout_challenge_scores[idx] = float(challenge_score)
                    if scout_class == 'easy':
                        scout_stop_reasons[idx] = 'scout_easy'
                        total_remaining -= int(remaining[idx])
                        remaining[idx] = 0
                        continue
                    if scout_class == 'challenge':
                        added = max(0, int(challenge_budget) - int(simulation_budgets[idx]))
                        if added > 0:
                            simulation_budgets[idx] = int(challenge_budget)
                            simulation_budget_extended[idx] = True
                            scout_challenge_flags[idx] = True
                            self._set_root_challenge_selection(root, True)
                            remaining[idx] += int(added)
                            total_remaining += int(added)
                        continue
                    continue

                stop_reason = None
                if scout_class == 'challenge':
                    if used >= self._scout_challenge_minimum(simulation_budgets[idx]):
                        stop_reason = self._scout_locked_stop_reason(
                            root,
                            summary,
                            simulation_budgets[idx],
                        )
                elif self._scout_root_is_easy(root, summary):
                    stop_reason = 'scout_normal_easy'
                if stop_reason is None:
                    continue
                scout_stop_reasons[idx] = stop_reason
                total_remaining -= int(remaining[idx])
                remaining[idx] = 0
            if profile_detail:
                self._profile_add('search_scout_classify_time', perf_counter() - scout_t0)

        # Sync roots back to caller-provided state containers.
        for idx, gs in enumerate(game_states):
            if is_mapping_state[idx]:
                gs['root'] = roots[idx]
                gs['_root_synced'] = bool(root_synced_flags[idx])
                gs['_needs_root_noise_on_expand'] = bool(needs_root_noise_on_expand[idx])
            else:
                if len(gs) > 1:
                    gs[1] = roots[idx]
                if len(gs) > 2:
                    gs[2] = bool(root_synced_flags[idx])

        result = [
            (root.child_visit_dict() if root is not None else {})
            for root in roots
        ]
        search_metadata = []
        metadata_t0 = perf_counter() if profile_detail else None
        for idx, root in enumerate(roots):
            metadata = self._summarize_root_search(
                root,
                simulation_budgets[idx],
                initial_root_visits=initial_root_visits[idx],
            )
            metadata['simulation_budget_extra'] = bool(simulation_budget_extended[idx])
            metadata['simulation_budget_scout_challenge'] = bool(scout_challenge_flags[idx])
            if scout_challenge_scores[idx] is None:
                scout_challenge_scores[idx] = self._scout_prior_uncertainty_score(
                    root,
                    move_count=move_counts[idx],
                )
            metadata['scout_challenge_score'] = float(scout_challenge_scores[idx] or 0.0)
            metadata['scout_challenge_applied'] = 1.0 if bool(scout_challenge_flags[idx]) else 0.0
            metadata['move_count'] = int(move_counts[idx])
            stop_reason = scout_stop_reasons[idx]
            if stop_reason is None:
                stop_reason = 'budget' if not metadata.get('stopped_early', False) else 'unknown'
            metadata['scout_stop_reason'] = stop_reason
            search_metadata.append(metadata)
        if profile_detail:
            self._profile_add('search_metadata_time', perf_counter() - metadata_t0)
        self._profile_add('search_many_time', perf_counter() - search_t0)
        self._profile_inc('search_many_calls', 1)
        if return_search_metadata:
            return result, search_metadata
        return result

    def _batch_expand_and_evaluate(
        self,
        nodes,
        game_indices,
        board_histories,
        roots,
        needs_root_noise_on_expand,
    ):
        """
        Batch expansion + evaluation for leaf nodes across games.
        """
        perf_counter = time.perf_counter
        profile_detail = self.profile_detail_enabled
        profile_timing = self.profile_enabled
        eval_t0 = perf_counter() if profile_timing else None
        # The same leaf can appear multiple times in one batch.
        # Evaluate each unique node once and fan-out value to duplicates.
        dedup_t0 = perf_counter() if profile_timing else None
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
        if profile_timing:
            self._profile_add('batch_expand_dedup_terminal_time', perf_counter() - dedup_t0)

        values_by_node_id = {}

        if non_terminal_nodes:
            history_prefix_cache = {}

            def _get_history_prefix_for_node(game_idx, node):
                if self.history_positions <= 0:
                    return None

                cache_key = (int(game_idx), id(node))
                if cache_key in history_prefix_cache:
                    return history_prefix_cache[cache_key]

                cached = self._build_history_prefix_for_node(node, board_histories[game_idx])
                history_prefix_cache[cache_key] = cached
                return cached

            legal_moves_per_node = []
            legal_indices_per_node = []
            legal_counts = []
            max_legal_count = 0
            legal_moves_t0 = perf_counter() if profile_timing else None
            for node in non_terminal_nodes:
                legal_moves, legal_indices = node.get_legal_moves_and_indices()
                legal_moves_per_node.append(legal_moves)
                legal_indices_per_node.append(legal_indices)
                legal_count = len(legal_indices)
                legal_counts.append(legal_count)
                if legal_count > max_legal_count:
                    max_legal_count = legal_count
            if profile_timing:
                self._profile_add('batch_expand_legal_moves_time', perf_counter() - legal_moves_t0)

            batch_n = len(non_terminal_nodes)
            boards_np = self._get_board_input_scratch(batch_n)
            tensor_pack_t0 = perf_counter() if profile_timing else None
            for row_idx, (node, gi) in enumerate(zip(non_terminal_nodes, non_terminal_game_indices)):
                current_tensor = self._current_tensor_for_node(node)
                history_t0 = perf_counter() if profile_detail else None
                history_prefix = _get_history_prefix_for_node(gi, node)
                if profile_detail:
                    self._profile_add('batch_expand_history_time', perf_counter() - history_t0)
                input_pack_t0 = perf_counter() if profile_detail else None
                if history_prefix is None:
                    boards_np[row_idx, :, :, :] = current_tensor
                else:
                    boards_np[row_idx, :self._history_planes, :, :] = history_prefix
                    boards_np[row_idx, self._history_planes:, :, :] = current_tensor
                if profile_detail:
                    self._profile_add('batch_expand_input_pack_time', perf_counter() - input_pack_t0)
            if profile_timing:
                self._profile_add('batch_expand_tensor_pack_time', perf_counter() - tensor_pack_t0)

            use_cuda_stage_timing = (
                self.cuda_stage_profile_enabled
                and self.device.type == 'cuda'
                and torch.cuda.is_available()
            )

            def _cuda_event():
                if not use_cuda_stage_timing:
                    return None
                return torch.cuda.Event(enable_timing=True)

            remote_legal_gather = bool(getattr(self.model, 'supports_remote_legal_gather', False))
            h2d_start = _cuda_event()
            h2d_end = _cuda_event()
            if remote_legal_gather:
                board_tensors = boards_np
            else:
                if h2d_start is not None:
                    h2d_start.record()
                board_tensors = self._get_board_input_source_tensor(boards_np).to(
                    self.device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
                if h2d_end is not None:
                    h2d_end.record()

            legal_index_matrix = None
            if max_legal_count > 0:
                legal_pack_t0 = perf_counter() if profile_timing else None
                legal_index_matrix = self._get_legal_index_scratch(
                    len(non_terminal_nodes),
                    max_legal_count,
                    dtype=np.int16 if remote_legal_gather else np.int64,
                )
                legal_index_matrix.fill(0)
                for row_idx, legal_indices in enumerate(legal_indices_per_node):
                    legal_count = legal_counts[row_idx]
                    if legal_count:
                        legal_index_matrix[row_idx, :legal_count] = legal_indices
                if profile_timing:
                    self._profile_add('batch_expand_legal_index_pack_time', perf_counter() - legal_pack_t0)

            inference_t0 = perf_counter()
            h2d_legal_start = None
            h2d_legal_end = None
            wdl_start = None
            wdl_end = None
            gather_start = None
            gather_end = None
            forward_start = _cuda_event()
            forward_end = _cuda_event()
            if forward_start is not None:
                forward_start.record()
            model_kwargs = {'apply_log_softmax': False}
            if (
                legal_index_matrix is not None
                and remote_legal_gather
            ):
                model_kwargs['legal_index_matrix'] = legal_index_matrix
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                        policy_logits_batch, values_batch = self.model(
                            board_tensors,
                            **model_kwargs,
                        )
                else:
                    policy_logits_batch, values_batch = self.model(
                        board_tensors,
                        **model_kwargs,
                    )
                if forward_end is not None:
                    forward_end.record()
                compact_policy_response = bool(getattr(self.model, 'last_response_compact_policy', False))
                
                if values_batch.dim() == 2 and values_batch.shape[1] == 3:
                    wdl_start = _cuda_event()
                    wdl_end = _cuda_event()
                    if wdl_start is not None:
                        wdl_start.record()
                    wdl_probs = torch.softmax(values_batch, dim=1)
                    values_batch = (wdl_probs[:, 0] - wdl_probs[:, 2])
                    if wdl_end is not None:
                        wdl_end.record()

                # Gather only legal move logits on GPU before transferring to CPU.
                if max_legal_count > 0:
                    if compact_policy_response:
                        legal_logits_batch = policy_logits_batch.to(dtype=torch.float16).cpu().numpy()
                        d2h_legal_start = None
                        d2h_legal_end = None
                    else:
                        h2d_legal_start = _cuda_event()
                        h2d_legal_end = _cuda_event()
                        if h2d_legal_start is not None:
                            h2d_legal_start.record()
                        legal_index_tensor = self._get_legal_index_source_tensor(legal_index_matrix).to(
                            self.device,
                            non_blocking=True,
                        )
                        if h2d_legal_end is not None:
                            h2d_legal_end.record()
                        gather_start = _cuda_event()
                        gather_end = _cuda_event()
                        if gather_start is not None:
                            gather_start.record()
                        legal_logits_batch = torch.gather(policy_logits_batch, 1, legal_index_tensor)
                        if gather_end is not None:
                            gather_end.record()
                        d2h_legal_start = _cuda_event()
                        d2h_legal_end = _cuda_event()
                        if d2h_legal_start is not None:
                            d2h_legal_start.record()
                        legal_logits_batch = legal_logits_batch.to(dtype=torch.float16).cpu().numpy()
                        if d2h_legal_end is not None:
                            d2h_legal_end.record()
                else:
                    legal_logits_batch = None
                    d2h_legal_start = None
                    d2h_legal_end = None
                d2h_values_start = _cuda_event()
                d2h_values_end = _cuda_event()
                if d2h_values_start is not None:
                    d2h_values_start.record()
                values_batch = values_batch.float().cpu().numpy()
                if d2h_values_end is not None:
                    d2h_values_end.record()
            self._profile_inc('nn_inference_calls', 1)
            self._profile_inc('nn_inference_batch_items', batch_n)
            self._profile_inc('nn_legal_move_items', sum(legal_counts))
            self._profile_add('nn_inference_time', perf_counter() - inference_t0)
            server_batch_size = int(getattr(self.model, 'last_server_batch_size', 0) or 0)
            if server_batch_size > 0:
                self._profile_inc('central_inference_requests', 1)
                self._profile_inc('central_inference_server_batch_items', server_batch_size)
                self._profile_add(
                    'central_inference_remote_wait_time',
                    float(getattr(self.model, 'last_remote_wait_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_request_put_time',
                    float(getattr(self.model, 'last_request_put_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_queue_wait_time',
                    float(getattr(self.model, 'last_server_queue_wait_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_total_time',
                    float(getattr(self.model, 'last_server_total_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_concat_time',
                    float(getattr(self.model, 'last_server_concat_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_h2d_time',
                    float(getattr(self.model, 'last_server_h2d_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_forward_time',
                    float(getattr(self.model, 'last_server_forward_s', 0.0) or 0.0),
                )
                self._profile_add(
                    'central_inference_server_d2h_time',
                    float(getattr(self.model, 'last_server_d2h_s', 0.0) or 0.0),
                )
            if profile_detail:
                if use_cuda_stage_timing:
                    torch.cuda.synchronize(self.device)

                    def _elapsed_s(start_event, end_event):
                        if start_event is None or end_event is None:
                            return 0.0
                        return max(0.0, float(start_event.elapsed_time(end_event)) / 1000.0)

                    h2d_time = _elapsed_s(h2d_start, h2d_end)
                    h2d_time += _elapsed_s(h2d_legal_start, h2d_legal_end)
                    gpu_postprocess_time = (
                        _elapsed_s(wdl_start, wdl_end)
                        + _elapsed_s(gather_start, gather_end)
                    )
                    d2h_time = _elapsed_s(d2h_legal_start, d2h_legal_end) + _elapsed_s(d2h_values_start, d2h_values_end)
                    self._profile_add('nn_h2d_time', h2d_time)
                    self._profile_add('nn_gpu_forward_time', _elapsed_s(forward_start, forward_end))
                    self._profile_add('nn_gpu_postprocess_time', gpu_postprocess_time)
                    self._profile_add('nn_d2h_time', d2h_time)

            cpu_policy_t0 = perf_counter() if profile_timing else None
            for idx, node in enumerate(non_terminal_nodes):
                value = float(values_batch[idx])

                legal_moves = legal_moves_per_node[idx]
                legal_count = legal_counts[idx]

                if legal_moves:
                    legal_logits = legal_logits_batch[idx, :legal_count].astype(np.float32, copy=False)
                    legal_probs = np.exp(legal_logits - legal_logits.max())
                    legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                    tactical_multipliers = self._tactical_prior_multipliers(node.board, legal_moves)
                    if tactical_multipliers.size == legal_probs.size:
                        legal_probs = legal_probs * tactical_multipliers
                        legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
                else:
                    legal_probs = np.array([])

                gi = non_terminal_game_indices[idx]
                add_noise = (
                    needs_root_noise_on_expand[gi]
                    and node is roots[gi]
                )

                # Expand only if not already expanded (avoid overwriting priors in same batch).
                # Keep base_priors as the clean model prior; root noise belongs only to
                # search-time priors so MCTS-vs-prior telemetry remains interpretable.
                if not node.expanded:
                    node.expand_children(legal_moves, legal_probs)
                    if add_noise and len(legal_moves) > 0:
                        self._apply_root_noise(node)
                        needs_root_noise_on_expand[gi] = False

                values_by_node_id[id(node)] = value
            if profile_timing:
                self._profile_add('batch_expand_cpu_policy_time', perf_counter() - cpu_policy_t0)

        fanout_t0 = perf_counter() if profile_timing else None
        all_values = [0.0] * len(nodes)
        for node_id, indices in node_occurrences.items():
            value = values_by_node_id.get(node_id, terminal_values.get(node_id, 0.0))
            for idx in indices:
                all_values[idx] = value
        if profile_timing:
            self._profile_add('batch_expand_value_fanout_time', perf_counter() - fanout_t0)

        if profile_timing:
            self._profile_add('batch_expand_eval_time', perf_counter() - eval_t0)
            self._profile_inc('batch_expand_eval_calls', 1)
        return all_values


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
        self.mcts_phase_opening_max_fullmove = max(
            1,
            int(rl_cfg.get('value_phase_opening_max_fullmove', 12)),
        )
        self.mcts_phase_endgame_min_fullmove = max(
            self.mcts_phase_opening_max_fullmove + 1,
            int(rl_cfg.get('value_phase_endgame_min_fullmove', 40)),
        )

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
        self._warned_missing_opponent_labels = set()

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
        self.syzygy = _get_syzygy_oracle(config)
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
        self.policy_target_pruning_enabled = bool(
            rl_cfg.get('policy_target_pruning_enabled', False)
        )
        self.policy_target_pruning_keep_top_n = max(
            1,
            int(rl_cfg.get('policy_target_pruning_keep_top_n', 2)),
        )
        self.policy_target_pruning_max_moves = max(
            self.policy_target_pruning_keep_top_n,
            int(rl_cfg.get('policy_target_pruning_max_moves', 12)),
        )
        self.policy_target_pruning_min_prob = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_pruning_min_prob', 0.02))),
        )
        self.policy_target_pruning_min_fraction_of_max = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_pruning_min_fraction_of_max', 0.12))),
        )
        self.policy_target_pruning_keep_mass = max(
            0.05,
            min(1.0, float(rl_cfg.get('policy_target_pruning_keep_mass', 0.92))),
        )
        self.policy_target_quality_weighting_enabled = bool(
            rl_cfg.get('policy_target_quality_weighting_enabled', False)
        )
        self.policy_target_quality_min_weight = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_quality_min_weight', 0.35))),
        )
        self.policy_target_quality_entropy_threshold = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_quality_entropy_threshold', 0.72))),
        )
        self.policy_target_quality_top1_threshold = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_quality_top1_threshold', 0.48))),
        )
        self.policy_target_uptake_gate_enabled = bool(
            rl_cfg.get('policy_target_uptake_gate_enabled', False)
        )
        self.policy_target_uptake_unchanged_low_kl_max = max(
            0.0,
            float(rl_cfg.get('policy_target_uptake_unchanged_low_kl_max', 0.16)),
        )
        self.policy_target_uptake_unchanged_low_kl_weight = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_uptake_unchanged_low_kl_weight', 0.22))),
        )
        self.policy_target_uptake_unchanged_weight = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_uptake_unchanged_weight', 0.50))),
        )
        self.policy_target_search_change_weighting_enabled = bool(
            rl_cfg.get('policy_target_search_change_weighting_enabled', False)
        )
        self.policy_target_search_change_min_q_delta = float(
            rl_cfg.get('policy_target_search_change_min_q_delta', 0.04)
        )
        self.policy_target_search_change_q_delta_ref = max(
            1e-6,
            float(rl_cfg.get('policy_target_search_change_q_delta_ref', 0.30)),
        )
        self.policy_target_search_change_bonus = max(
            0.0,
            float(rl_cfg.get('policy_target_search_change_bonus', 0.20)),
        )
        self.policy_target_search_change_q_bonus = max(
            0.0,
            float(rl_cfg.get('policy_target_search_change_q_bonus', 0.30)),
        )
        self.policy_target_search_change_rank_bonus = max(
            0.0,
            float(rl_cfg.get('policy_target_search_change_rank_bonus', 0.15)),
        )
        self.policy_target_search_change_allow_q_neutral = bool(
            rl_cfg.get('policy_target_search_change_allow_q_neutral', False)
        )
        self.policy_target_search_change_min_top_visit_prob = max(
            0.0,
            min(1.0, float(rl_cfg.get('policy_target_search_change_min_top_visit_prob', 0.55))),
        )
        self.policy_target_search_change_min_prior_rank = max(
            1,
            int(rl_cfg.get('policy_target_search_change_min_prior_rank', 2)),
        )
        self.policy_target_search_change_neutral_min_q_delta = float(
            rl_cfg.get('policy_target_search_change_neutral_min_q_delta', -0.02)
        )
        self.policy_target_search_change_neutral_bonus = max(
            0.0,
            float(rl_cfg.get('policy_target_search_change_neutral_bonus', 0.0)),
        )
        self.policy_target_search_change_neutral_rank_bonus = max(
            0.0,
            float(rl_cfg.get('policy_target_search_change_neutral_rank_bonus', 0.0)),
        )
        self.policy_target_search_change_max_weight = max(
            1.0,
            float(rl_cfg.get('policy_target_search_change_max_weight', 1.55)),
        )
        self.mcts_good_target_min_top_visit_prob = 0.55
        self.mcts_good_target_min_visit_gap = 0.12
        self.store_frozen_best_positions = bool(
            rl_cfg.get('self_play_store_frozen_best_positions', True)
        )
        self.store_frozen_anchor_positions = bool(
            rl_cfg.get('self_play_store_frozen_anchor_positions', True)
        )
        self.store_frozen_recent_positions = False
        self.frozen_opponent_policy_weight = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_frozen_opponent_policy_weight', 0.65))),
        )
        self.frozen_anchor_policy_weight = max(
            0.0,
            min(1.0, float(rl_cfg.get('self_play_frozen_anchor_policy_weight', 0.75))),
        )
        self.replay_importance_gating_enabled = bool(
            rl_cfg.get('replay_importance_gating_enabled', False)
        )
        self.replay_importance_gate_fraction_decisive = max(
            0.0,
            min(1.0, float(rl_cfg.get('replay_importance_gate_fraction_decisive', 1.0))),
        )
        self.replay_importance_gate_fraction_draw = max(
            0.0,
            min(1.0, float(rl_cfg.get('replay_importance_gate_fraction_draw', 1.0))),
        )
        self.replay_importance_gate_min_positions = max(
            0,
            int(rl_cfg.get('replay_importance_gate_min_positions', 0)),
        )
        self.replay_importance_top_fraction = max(
            0.0,
            min(1.0, float(rl_cfg.get('replay_importance_top_fraction', 0.70))),
        )
        self.value_target_weighting_enabled = bool(
            rl_cfg.get('value_target_weighting_enabled', False)
        )
        self.value_target_weight_min = max(
            0.0,
            min(1.0, float(rl_cfg.get('value_target_weight_min', 0.35))),
        )
        self.value_target_weight_draw_min = max(
            0.0,
            min(1.0, float(rl_cfg.get('value_target_weight_draw_min', self.value_target_weight_min))),
        )
        self.value_target_weight_power = max(
            0.1,
            float(rl_cfg.get('value_target_weight_power', 1.75)),
        )
        
        self.max_positions_per_game = max(
            0,
            int(rl_cfg.get('replay_max_positions_per_game', 32)),
        )

        self.max_batch_games_per_worker = max(1, int(max_batch_games_per_worker))
        self._progress_file = None  # Set externally to enable progress reporting
        self._games_completed = 0
        self._progress_base = 0
        self._plan_cursor = 0
        # Performance CSV/PNG should not depend on debug.rl.profile_training.
        # That flag controls console/debug verbosity, not collection of cheap counters.
        self.profile_enabled = True
        self._profile_stats = {}
        self.reset_profile_stats()

    def reset_profile_stats(self):
        self._profile_stats = {
            'move_selection_time': 0.0,
            'move_selection_calls': 0,
            'adjudication_time': 0.0,
            'adjudication_calls': 0,
            'syzygy_time': 0.0,
            'syzygy_calls': 0,
            'policy_target_build_time': 0.0,
            'policy_target_build_calls': 0,
            'policy_target_postgame_time': 0.0,
            'policy_target_postgame_calls': 0,
        }

    def _profile_add(self, key, value):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = float(self._profile_stats.get(key, 0.0)) + float(value)

    def _profile_inc(self, key, value=1):
        if not self.profile_enabled:
            return
        self._profile_stats[key] = int(self._profile_stats.get(key, 0)) + int(value)

    def _merge_profile_stats(self, target, source):
        for key, value in dict(source or {}).items():
            if isinstance(value, (int, np.integer)):
                target[key] = int(target.get(key, 0)) + int(value)
            else:
                target[key] = float(target.get(key, 0.0)) + float(value)

    def _aggregate_engine_profile_stats(self):
        aggregated = dict(self._profile_stats)
        self._merge_profile_stats(
            aggregated,
            {f"learner_mcts_{k}": v for k, v in self.mcts.get_profile_stats().items()},
        )
        for label, opponent_mcts in self.opponent_mcts_by_label.items():
            self._merge_profile_stats(
                aggregated,
                {f"opponent_mcts_{label}_{k}": v for k, v in opponent_mcts.get_profile_stats().items()},
            )
        search_many_time = float(aggregated.get('learner_mcts_search_many_time', 0.0))
        batch_expand_time = float(aggregated.get('learner_mcts_batch_expand_eval_time', 0.0))
        batch_expand_calls = int(aggregated.get('learner_mcts_batch_expand_eval_calls', 0) or 0)
        board_tensor_time = float(aggregated.get('learner_mcts_board_to_tensor_time', 0.0))
        board_tensor_calls = int(aggregated.get('learner_mcts_board_to_tensor_calls', 0) or 0)
        nn_time = float(aggregated.get('learner_mcts_nn_inference_time', 0.0))
        nn_calls = int(aggregated.get('learner_mcts_nn_inference_calls', 0) or 0)
        nn_batch_items = int(aggregated.get('learner_mcts_nn_inference_batch_items', 0) or 0)
        nn_h2d_time = float(aggregated.get('learner_mcts_nn_h2d_time', 0.0))
        nn_gpu_forward_time = float(aggregated.get('learner_mcts_nn_gpu_forward_time', 0.0))
        nn_gpu_postprocess_time = float(aggregated.get('learner_mcts_nn_gpu_postprocess_time', 0.0))
        nn_d2h_time = float(aggregated.get('learner_mcts_nn_d2h_time', 0.0))
        nn_legal_move_items = int(aggregated.get('learner_mcts_nn_legal_move_items', 0) or 0)
        central_requests = int(aggregated.get('learner_mcts_central_inference_requests', 0) or 0)
        central_batch_items = int(aggregated.get('learner_mcts_central_inference_server_batch_items', 0) or 0)
        central_request_put_time = float(aggregated.get('learner_mcts_central_inference_request_put_time', 0.0) or 0.0)
        central_remote_wait_time = float(aggregated.get('learner_mcts_central_inference_remote_wait_time', 0.0) or 0.0)
        central_server_queue_wait_time = float(aggregated.get('learner_mcts_central_inference_server_queue_wait_time', 0.0) or 0.0)
        central_server_total_time = float(aggregated.get('learner_mcts_central_inference_server_total_time', 0.0) or 0.0)
        central_server_concat_time = float(aggregated.get('learner_mcts_central_inference_server_concat_time', 0.0) or 0.0)
        central_server_h2d_time = float(aggregated.get('learner_mcts_central_inference_server_h2d_time', 0.0) or 0.0)
        central_server_forward_time = float(aggregated.get('learner_mcts_central_inference_server_forward_time', 0.0) or 0.0)
        central_server_d2h_time = float(aggregated.get('learner_mcts_central_inference_server_d2h_time', 0.0) or 0.0)
        for label in self.opponent_mcts_by_label.keys():
            prefix = f"opponent_mcts_{label}_"
            search_many_time += float(aggregated.get(prefix + 'search_many_time', 0.0))
            batch_expand_time += float(aggregated.get(prefix + 'batch_expand_eval_time', 0.0))
            batch_expand_calls += int(aggregated.get(prefix + 'batch_expand_eval_calls', 0) or 0)
            board_tensor_time += float(aggregated.get(prefix + 'board_to_tensor_time', 0.0))
            board_tensor_calls += int(aggregated.get(prefix + 'board_to_tensor_calls', 0) or 0)
            nn_time += float(aggregated.get(prefix + 'nn_inference_time', 0.0))
            nn_calls += int(aggregated.get(prefix + 'nn_inference_calls', 0) or 0)
            nn_batch_items += int(aggregated.get(prefix + 'nn_inference_batch_items', 0) or 0)
            nn_h2d_time += float(aggregated.get(prefix + 'nn_h2d_time', 0.0))
            nn_gpu_forward_time += float(aggregated.get(prefix + 'nn_gpu_forward_time', 0.0))
            nn_gpu_postprocess_time += float(aggregated.get(prefix + 'nn_gpu_postprocess_time', 0.0))
            nn_d2h_time += float(aggregated.get(prefix + 'nn_d2h_time', 0.0))
            nn_legal_move_items += int(aggregated.get(prefix + 'nn_legal_move_items', 0) or 0)
            central_requests += int(aggregated.get(prefix + 'central_inference_requests', 0) or 0)
            central_batch_items += int(aggregated.get(prefix + 'central_inference_server_batch_items', 0) or 0)
            central_request_put_time += float(aggregated.get(prefix + 'central_inference_request_put_time', 0.0) or 0.0)
            central_remote_wait_time += float(aggregated.get(prefix + 'central_inference_remote_wait_time', 0.0) or 0.0)
            central_server_queue_wait_time += float(aggregated.get(prefix + 'central_inference_server_queue_wait_time', 0.0) or 0.0)
            central_server_total_time += float(aggregated.get(prefix + 'central_inference_server_total_time', 0.0) or 0.0)
            central_server_concat_time += float(aggregated.get(prefix + 'central_inference_server_concat_time', 0.0) or 0.0)
            central_server_h2d_time += float(aggregated.get(prefix + 'central_inference_server_h2d_time', 0.0) or 0.0)
            central_server_forward_time += float(aggregated.get(prefix + 'central_inference_server_forward_time', 0.0) or 0.0)
            central_server_d2h_time += float(aggregated.get(prefix + 'central_inference_server_d2h_time', 0.0) or 0.0)
        aggregated['mcts_search_many_time'] = float(search_many_time)
        aggregated['mcts_batch_expand_eval_time'] = float(batch_expand_time)
        aggregated['mcts_batch_expand_eval_calls'] = int(batch_expand_calls)
        aggregated['mcts_board_to_tensor_time'] = float(board_tensor_time)
        aggregated['mcts_board_to_tensor_calls'] = int(board_tensor_calls)
        aggregated['mcts_nn_inference_time'] = float(nn_time)
        aggregated['mcts_nn_inference_calls'] = int(nn_calls)
        aggregated['mcts_nn_inference_batch_items'] = int(nn_batch_items)
        aggregated['mcts_nn_h2d_time'] = float(nn_h2d_time)
        aggregated['mcts_nn_gpu_forward_time'] = float(nn_gpu_forward_time)
        aggregated['mcts_nn_gpu_postprocess_time'] = float(nn_gpu_postprocess_time)
        aggregated['mcts_nn_d2h_time'] = float(nn_d2h_time)
        aggregated['mcts_nn_legal_move_items'] = int(nn_legal_move_items)
        aggregated['mcts_central_inference_requests'] = int(central_requests)
        aggregated['mcts_central_inference_server_batch_items'] = int(central_batch_items)
        aggregated['mcts_central_inference_request_put_time'] = float(central_request_put_time)
        aggregated['mcts_central_inference_remote_wait_time'] = float(central_remote_wait_time)
        aggregated['mcts_central_inference_server_queue_wait_time'] = float(central_server_queue_wait_time)
        aggregated['mcts_central_inference_server_total_time'] = float(central_server_total_time)
        aggregated['mcts_central_inference_server_concat_time'] = float(central_server_concat_time)
        aggregated['mcts_central_inference_server_h2d_time'] = float(central_server_h2d_time)
        aggregated['mcts_central_inference_server_forward_time'] = float(central_server_forward_time)
        aggregated['mcts_central_inference_server_d2h_time'] = float(central_server_d2h_time)
        extra_mcts_time_metrics = [
            'search_root_setup_time',
            'search_selection_time',
            'search_backprop_time',
            'search_scout_classify_time',
            'search_metadata_time',
            'batch_expand_dedup_terminal_time',
            'batch_expand_legal_moves_time',
            'batch_expand_tensor_pack_time',
            'batch_expand_history_time',
            'batch_expand_input_pack_time',
            'batch_expand_legal_index_pack_time',
            'batch_expand_cpu_policy_time',
            'batch_expand_value_fanout_time',
        ]
        for metric in extra_mcts_time_metrics:
            total = float(aggregated.get(f'learner_mcts_{metric}', 0.0) or 0.0)
            for label in self.opponent_mcts_by_label.keys():
                total += float(aggregated.get(f'opponent_mcts_{label}_{metric}', 0.0) or 0.0)
            aggregated[f'mcts_{metric}'] = float(total)
        aggregated['average_batch_size'] = float(nn_batch_items / nn_calls) if nn_calls > 0 else 0.0
        aggregated['central_average_batch_size'] = (
            float(central_batch_items / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_remote_wait_ms_per_request'] = (
            1000.0 * float(central_remote_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_request_put_ms_per_request'] = (
            1000.0 * float(central_request_put_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_queue_wait_ms_per_request'] = (
            1000.0 * float(central_server_queue_wait_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_forward_ms_per_request'] = (
            1000.0 * float(central_server_forward_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_h2d_ms_per_request'] = (
            1000.0 * float(central_server_h2d_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_d2h_ms_per_request'] = (
            1000.0 * float(central_server_d2h_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_concat_ms_per_request'] = (
            1000.0 * float(central_server_concat_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['central_server_total_ms_per_request'] = (
            1000.0 * float(central_server_total_time / central_requests) if central_requests > 0 else 0.0
        )
        aggregated['average_legal_moves_per_position'] = (
            float(nn_legal_move_items / nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        aggregated['average_legal_moves_per_batch'] = (
            float(nn_legal_move_items / nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_batch_ms'] = (
            1000.0 * float(nn_time) / float(nn_calls) if nn_calls > 0 else 0.0
        )
        aggregated['inference_time_per_position_ms'] = (
            1000.0 * float(nn_time) / float(nn_batch_items) if nn_batch_items > 0 else 0.0
        )
        gpu_utilization_pct = 100.0 * float(nn_time) / max(1e-8, float(search_many_time))
        aggregated['gpu_utilization_pct'] = float(max(0.0, min(100.0, gpu_utilization_pct)))
        return aggregated

    def _prune_policy_target_visits(self, visit_counts):
        if not self.policy_target_pruning_enabled or not visit_counts:
            return visit_counts

        items = sorted(
            ((move, float(count)) for move, count in visit_counts.items() if float(count) > 0.0),
            key=lambda pair: (-pair[1], pair[0].uci()),
        )
        if not items:
            return visit_counts

        total = float(sum(count for _, count in items))
        if total <= 0.0:
            return visit_counts

        max_count = float(items[0][1])
        kept = []
        cumulative = 0.0
        for idx, (move, count) in enumerate(items):
            prob = count / total
            keep = idx < self.policy_target_pruning_keep_top_n
            if not keep and len(kept) < self.policy_target_pruning_max_moves:
                if (
                    prob >= self.policy_target_pruning_min_prob
                    and count >= max_count * self.policy_target_pruning_min_fraction_of_max
                    and cumulative < self.policy_target_pruning_keep_mass
                ):
                    keep = True
            if keep:
                kept.append((move, count))
                cumulative += prob
            if len(kept) >= self.policy_target_pruning_max_moves:
                break

        if not kept:
            kept = items[:1]
        return {move: int(max(1.0, round(count))) for move, count in kept}

    @staticmethod
    def _policy_target_quality_from_visits(visit_counts):
        if not visit_counts:
            return 0.0, 1.0
        counts = np.asarray(
            [float(count) for count in visit_counts.values() if float(count) > 0.0],
            dtype=np.float64,
        )
        if counts.size <= 0:
            return 0.0, 1.0
        total = float(counts.sum())
        if total <= 0.0 or not np.isfinite(total):
            return 0.0, 1.0
        probs = counts / total
        top1 = float(probs.max())
        if probs.size <= 1:
            return top1, 0.0
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        entropy = entropy / float(np.log(probs.size))
        return top1, float(max(0.0, min(1.0, entropy)))

    def _policy_target_quality_weight(self, top1_prob, normalized_entropy):
        if not self.policy_target_quality_weighting_enabled:
            return 1.0
        top1_threshold = float(self.policy_target_quality_top1_threshold)
        entropy_threshold = float(self.policy_target_quality_entropy_threshold)
        top1_score = (
            1.0
            if top1_threshold <= 0.0
            else max(0.0, min(1.0, float(top1_prob) / top1_threshold))
        )
        entropy_score = (
            1.0
            if entropy_threshold >= 1.0
            else max(0.0, min(1.0, (1.0 - float(normalized_entropy)) / (1.0 - entropy_threshold)))
        )
        confidence = max(top1_score, entropy_score)
        min_weight = float(self.policy_target_quality_min_weight)
        return float(min_weight + (1.0 - min_weight) * confidence)

    def _policy_target_search_change_weight(self, search_metadata):
        if not self.policy_target_search_change_weighting_enabled:
            return 1.0
        if not isinstance(search_metadata, dict):
            return 1.0
        try:
            changed_top = float(search_metadata.get('prior_mcts_agree', 1.0)) < 0.5
        except (TypeError, ValueError):
            changed_top = False
        if not changed_top:
            return 1.0

        rank_strength = 0.0
        prior_rank = 1
        try:
            prior_rank = int(search_metadata.get('prior_top_visit_rank', 1) or 1)
            rank_strength = max(0.0, min(1.0, float(prior_rank - 1) / 4.0))
        except (TypeError, ValueError):
            prior_rank = 1
            rank_strength = 0.0

        q_delta = search_metadata.get('mcts_q_delta', None)
        q_delta_is_good = False
        try:
            if q_delta is not None:
                q_delta = float(q_delta)
                q_delta_is_good = (
                    math.isfinite(q_delta)
                    and q_delta >= float(self.policy_target_search_change_min_q_delta)
                )
        except (TypeError, ValueError):
            q_delta = None
            q_delta_is_good = False

        if q_delta_is_good:
            q_strength = max(0.0, min(1.0, float(q_delta) / float(self.policy_target_search_change_q_delta_ref)))
            weight = (
                1.0
                + float(self.policy_target_search_change_bonus)
                + float(self.policy_target_search_change_q_bonus) * q_strength
                + float(self.policy_target_search_change_rank_bonus) * rank_strength
            )
            return float(max(1.0, min(float(self.policy_target_search_change_max_weight), weight)))

        if not self.policy_target_search_change_allow_q_neutral:
            return 1.0
        if q_delta is not None and math.isfinite(float(q_delta)):
            if float(q_delta) < float(self.policy_target_search_change_neutral_min_q_delta):
                return 1.0

        try:
            top_visit_prob = float(search_metadata.get('top_visit_prob', 0.0) or 0.0)
        except (TypeError, ValueError):
            top_visit_prob = 0.0
        if (
            not math.isfinite(top_visit_prob)
            or top_visit_prob < float(self.policy_target_search_change_min_top_visit_prob)
            or prior_rank < int(self.policy_target_search_change_min_prior_rank)
        ):
            return 1.0

        weight = (
            1.0
            + float(self.policy_target_search_change_neutral_bonus)
            + float(self.policy_target_search_change_neutral_rank_bonus) * rank_strength
        )
        return float(max(1.0, min(float(self.policy_target_search_change_max_weight), weight)))

    def _policy_target_uptake_weight(self, search_metadata):
        if not self.policy_target_uptake_gate_enabled or not isinstance(search_metadata, dict):
            return 1.0
        try:
            changed_top = float(search_metadata.get('prior_mcts_agree', 1.0)) < 0.5
        except (TypeError, ValueError):
            changed_top = False
        if changed_top:
            return 1.0
        try:
            policy_kl = float(search_metadata.get('mcts_policy_kl', 0.0) or 0.0)
        except (TypeError, ValueError):
            policy_kl = 0.0
        if policy_kl <= float(self.policy_target_uptake_unchanged_low_kl_max):
            return float(self.policy_target_uptake_unchanged_low_kl_weight)
        return float(self.policy_target_uptake_unchanged_weight)

    def _mcts_phase_for_board(self, board):
        try:
            fullmove_number = int(getattr(board, 'fullmove_number', 1) or 1)
        except (TypeError, ValueError):
            fullmove_number = 1
        if fullmove_number <= int(self.mcts_phase_opening_max_fullmove):
            return 'opening'
        if fullmove_number >= int(self.mcts_phase_endgame_min_fullmove):
            return 'endgame'
        return 'middlegame'

    def _should_store_policy_position(self, learner_turn, game_opponent_mcts, opponent_label):
        if learner_turn or game_opponent_mcts is None:
            return True
        label = str(opponent_label or "")
        if label == "best":
            return bool(self.store_frozen_best_positions)
        if label == "anchor":
            return bool(self.store_frozen_anchor_positions)
        if label.startswith("recent_"):
            return bool(self.store_frozen_recent_positions)
        return False

    @staticmethod
    def _candidate_importance(item):
        try:
            return float(item.get('importance_score', 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _select_evenly_spaced_candidates(candidates, effective_cap):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        ordered = sorted(candidates, key=lambda item: int(item['history_idx']))
        positions = np.linspace(0, len(ordered) - 1, num=effective_cap)
        selected_history = []
        seen_history = set()
        for pos in positions:
            idx = int(round(float(pos)))
            idx = max(0, min(idx, len(ordered) - 1))
            history_idx = int(ordered[idx]['history_idx'])
            if history_idx in seen_history:
                continue
            seen_history.add(history_idx)
            selected_history.append(history_idx)
        if len(selected_history) < effective_cap:
            for item in ordered:
                history_idx = int(item['history_idx'])
                if history_idx in seen_history:
                    continue
                seen_history.add(history_idx)
                selected_history.append(history_idx)
                if len(selected_history) >= effective_cap:
                    break
        selected_lookup = set(selected_history[:effective_cap])
        return [item for item in ordered if int(item['history_idx']) in selected_lookup]

    def _select_top_scored_candidates(self, candidates, effective_cap, history_len):
        if effective_cap <= 0 or len(candidates) <= effective_cap:
            return list(candidates)
        del history_len
        ranked = sorted(
            candidates,
            key=lambda item: (-self._candidate_importance(item), int(item['history_idx'])),
        )
        top_quota = int(round(effective_cap * self.replay_importance_top_fraction))
        top_quota = max(1, min(int(effective_cap), top_quota))

        selected = []
        selected_history = set()
        for item in ranked:
            history_idx = int(item['history_idx'])
            if history_idx in selected_history:
                continue
            selected.append(item)
            selected_history.add(history_idx)
            if len(selected) >= top_quota:
                break

        remaining = int(effective_cap - len(selected))
        if remaining > 0:
            leftovers = [
                item for item in candidates
                if int(item['history_idx']) not in selected_history
            ]
            for item in self._select_evenly_spaced_candidates(leftovers, remaining):
                history_idx = int(item['history_idx'])
                if history_idx in selected_history:
                    continue
                selected.append(item)
                selected_history.add(history_idx)
                if len(selected) >= effective_cap:
                    break

        selected.sort(key=lambda item: int(item['history_idx']))
        return selected[:effective_cap]

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

        if self.replay_importance_gating_enabled and len(filtered) > 0:
            gate_fraction = (
                self.replay_importance_gate_fraction_decisive
                if is_decisive
                else self.replay_importance_gate_fraction_draw
            )
            if gate_fraction < 1.0:
                gate_keep = int(math.ceil(len(filtered) * gate_fraction))
                gate_keep = max(self.replay_importance_gate_min_positions, gate_keep)
                gate_keep = min(len(filtered), gate_keep)
                if gate_keep < len(filtered):
                    if is_decisive:
                        filtered = self._select_top_scored_candidates(filtered, gate_keep, history_len)
                    else:
                        filtered = self._select_draw_candidates_stratified(filtered, gate_keep, history_len)
                    curriculum_dropped = total_candidates - len(filtered)

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

    def _build_history_selection_candidates(self, gs):
        candidates = []
        for history_idx, history_entry in enumerate(gs['game_history']):
            importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
            candidates.append({
                'history_idx': history_idx,
                'importance_score': importance_score,
            })
        return candidates

    def _build_training_position_from_history_entry(
        self,
        gs,
        history_idx,
        history_len,
        outcome,
        draw_value_target=0.0,
    ):
        history_entry = gs['game_history'][int(history_idx)]
        history_count, policy_indices, policy_values, turn = history_entry[:4]
        importance_score = float(history_entry[4]) if len(history_entry) > 4 else 0.0
        policy_weight = float(history_entry[6]) if len(history_entry) > 6 else 1.0
        source_code = int(history_entry[7]) if len(history_entry) > 7 else _REPLAY_SOURCE_UNKNOWN

        if outcome == 0.0:
            value = draw_value_target
        else:
            signed_outcome = outcome if turn == chess.WHITE else -outcome
            value = float(signed_outcome)
        value_weight = self._value_target_weight(int(history_idx), int(history_len), outcome)

        return {
            'history_idx': int(history_idx),
            'history_count': history_count,
            'policy_indices': policy_indices,
            'policy_values': policy_values,
            'turn': turn,
            'value': value,
            'importance_score': importance_score,
            'policy_weight': policy_weight,
            'value_weight': value_weight,
            'source_code': source_code,
        }

    def _value_target_weight(self, history_idx, history_len, outcome):
        if not self.value_target_weighting_enabled:
            return 1.0
        history_len = max(1, int(history_len))
        progress = max(0.0, min(1.0, (int(history_idx) + 1) / float(history_len)))
        min_weight = (
            self.value_target_weight_draw_min
            if float(outcome) == 0.0
            else self.value_target_weight_min
        )
        shaped = progress ** float(self.value_target_weight_power)
        return float(min_weight + (1.0 - min_weight) * shaped)

    def _compute_position_importance(self, board, move, visit_counts, root):
        importance = 1.0

        if visit_counts:
            visits = np.asarray(list(visit_counts.values()), dtype=np.float32)
            total_visits = float(visits.sum())
            if total_visits > 0.0:
                probs = visits / total_visits
                top_prob = float(probs.max())
                entropy = 0.0
                if probs.size > 1:
                    entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
                    entropy /= float(np.log(probs.size))
                importance += 0.35 * (1.0 - top_prob)
                importance += 0.30 * entropy

        if root is not None:
            root_visits = int(getattr(root, 'visit_count', 0) or 0)
            if root_visits > 0:
                root_value = float(root.value_sum / max(1, root_visits))
                importance += 0.30 * abs(root_value)

        if board.is_capture(move):
            importance += 0.30
            gain = self.mcts._captured_piece_value(board, move) - self.mcts._moving_piece_value(board, move)
            if gain > 0.0:
                importance += 0.20 * min(1.0, gain / 4.0)
        if move.promotion is not None:
            importance += 0.30
        try:
            if board.gives_check(move):
                importance += 0.15
        except Exception:
            pass
        if board.move_stack:
            last_move = board.peek()
            if last_move is not None and move.to_square == last_move.to_square:
                importance += 0.10

        return float(importance)

    def _append_selected_positions_from_game(
        self,
        positions,
        gs,
        outcome,
        *,
        history_positions,
    ):
        postgame_t0 = time.perf_counter() if self.profile_enabled else None
        history_len = len(gs['game_history'])
        selection_candidates = self._build_history_selection_candidates(gs)
        selected_candidates, curriculum_dropped, cap_dropped = self._select_history_indices_to_keep(
            selection_candidates,
            history_len,
            is_decisive=(outcome != 0.0),
        )

        for candidate in selected_candidates:
            item = self._build_training_position_from_history_entry(
                gs,
                candidate['history_idx'],
                history_len,
                outcome,
                draw_value_target=0.0,
            )
            board_tensor_np = _build_history_tensor_from_encoded(
                turn=item['turn'],
                encoded_history=gs['board_history'],
                history_count=item['history_count'],
                history_positions=history_positions,
                empty_history_tensor=_EMPTY_HISTORY_TENSOR,
            )
            if board_tensor_np.dtype != np.float32:
                board_tensor_np = board_tensor_np.astype(np.float32, copy=False)
            board_tensor = torch.from_numpy(board_tensor_np)
            positions.append((
                board_tensor,
                item['policy_indices'],
                item['policy_values'],
                torch.tensor([item['value']], dtype=torch.float32),
                float(item.get('importance_score', 0.0)),
                float(item.get('policy_weight', 1.0)),
                float(item.get('value_weight', 1.0)),
                int(item.get('source_code', _REPLAY_SOURCE_UNKNOWN)),
            ))
        if self.profile_enabled:
            self._profile_add('policy_target_postgame_time', time.perf_counter() - postgame_t0)
            self._profile_inc('policy_target_postgame_calls', 1)
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

    @staticmethod
    def _clear_game_search_state(gs):
        gs['root'] = None
        gs['opponent_root'] = None
        gs['_root_synced'] = False
        gs['_opponent_root_synced'] = False

    @staticmethod
    def _advance_search_root(root, move):
        if root is None:
            return None, False
        child = root.get_child_for_move(move)
        if child is None:
            return None, False
        _ = child.board
        return child.detach_as_root(), True

    def _maybe_finish_with_syzygy(self, gs, board):
        if self.syzygy is None or board.is_game_over(claim_draw=False):
            return None
        if not self.syzygy.can_probe(board):
            return None
        gs['syzygy_probe_positions'] = int(gs.get('syzygy_probe_positions', 0)) + 1
        wdl = self.syzygy.probe_wdl(board)
        if wdl is None:
            return None
        gs['syzygy_probe_hits'] = int(gs.get('syzygy_probe_hits', 0)) + 1

        if wdl > 0:
            result = '1-0' if board.turn == chess.WHITE else '0-1'
        elif wdl < 0:
            result = '0-1' if board.turn == chess.WHITE else '1-0'
        else:
            result = '1/2-1/2'

        gs['done'] = True
        gs['syzygy_result'] = result
        self._clear_game_search_state(gs)
        self._mark_game_completed(gs)
        return result

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
        self.reset_profile_stats()
        self.mcts.reset_profile_stats()
        for opponent_mcts in self.opponent_mcts_by_label.values():
            opponent_mcts.reset_profile_stats()
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
        total_syzygy_ended_games = 0
        total_syzygy_probe_positions = 0
        total_syzygy_probe_hits = 0
        total_search_simulations_used = 0
        total_search_simulations_budget = 0
        total_search_extra_budget_samples = 0
        total_search_samples = 0
        search_simulations_used_samples = []
        search_simulations_budget_samples = []
        total_scout_stopped_early = 0
        scout_stop_reasons = {}
        total_mcts_quality_stats = {
            'mcts_prior_agreement_samples': 0.0,
            'mcts_prior_agreement_sum': 0.0,
            'mcts_prior_changed_count': 0.0,
            'mcts_prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'mcts_policy_kl_sum': 0.0,
            'mcts_top_visit_prob_sum': 0.0,
            'mcts_visit_gap_sum': 0.0,
            'mcts_visit_entropy_sum': 0.0,
            'mcts_good_target_count': 0.0,
            'mcts_explored_prior_mass_sum': 0.0,
            'mcts_visited_move_count_sum': 0.0,
            'mcts_legal_move_count_sum': 0.0,
            'mcts_visit_coverage_ratio_sum': 0.0,
            'mcts_q_delta_samples': 0.0,
            'mcts_q_delta_sum': 0.0,
            'mcts_q_delta_values': [],
            'mcts_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_changed_to_lower_q_count': 0.0,
            'mcts_changed_to_higher_q_count': 0.0,
            'mcts_changed_q_delta_samples': 0.0,
            'mcts_changed_q_delta_sum': 0.0,
            'mcts_changed_q_delta_values': [],
            'mcts_changed_q_delta_hist': [0] * _Q_DELTA_HIST_BINS,
            'mcts_search_change_count': 0.0,
            'mcts_search_change_weight_sum': 0.0,
            'mcts_scout_challenge_eligible_count': 0.0,
            'mcts_scout_challenge_used_count': 0.0,
            'mcts_scout_challenge_changed_count': 0.0,
            'mcts_scout_challenge_score_sum': 0.0,
            'mcts_policy_uptake_samples': 0.0,
            'mcts_policy_uptake_weight_sum': 0.0,
            'mcts_policy_uptake_low_count': 0.0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            total_mcts_quality_stats[f'mcts_phase_{phase}_samples'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_changed_count'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_scout_challenge_eligible_count'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_scout_challenge_used_count'] = 0.0
            total_mcts_quality_stats[f'mcts_phase_{phase}_scout_challenge_changed_count'] = 0.0
        opponent_source_counts = {}
        opponent_source_results = {}

        batch_plan_labels = list(self.opponent_plan_labels[self._plan_cursor:self._plan_cursor + num_games])
        self._plan_cursor += num_games
        positions, lengths, batch_stats = self._play_batch(num_games, batch_plan_labels=batch_plan_labels)
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
        total_syzygy_ended_games += int(batch_stats.get('syzygy_ended_games', 0))
        total_syzygy_probe_positions += int(batch_stats.get('syzygy_probe_positions', 0))
        total_syzygy_probe_hits += int(batch_stats.get('syzygy_probe_hits', 0))
        total_search_simulations_used += int(batch_stats.get('search_simulations_used_sum', 0))
        total_search_simulations_budget += int(batch_stats.get('search_simulations_budget_sum', 0))
        total_search_extra_budget_samples += int(batch_stats.get('search_extra_budget_samples', 0))
        total_search_samples += int(batch_stats.get('search_samples', 0))
        search_simulations_used_samples.extend(list(batch_stats.get('search_simulations_used_samples', []) or []))
        search_simulations_budget_samples.extend(list(batch_stats.get('search_simulations_budget_samples', []) or []))
        total_scout_stopped_early += int(batch_stats.get('scout_stopped_early', 0))
        for reason, count in dict(batch_stats.get('scout_stop_reasons', {}) or {}).items():
            scout_stop_reasons[str(reason)] = int(scout_stop_reasons.get(str(reason), 0)) + int(count)
        for key in total_mcts_quality_stats:
            if key in {'mcts_q_delta_values', 'mcts_changed_q_delta_values'}:
                total_mcts_quality_stats[key].extend(list(batch_stats.get(key, []) or []))
            elif key in {'mcts_q_delta_hist', 'mcts_changed_q_delta_hist'}:
                source_hist = list(batch_stats.get(key, []) or [])
                if len(source_hist) == _Q_DELTA_HIST_BINS:
                    for idx, count in enumerate(source_hist):
                        total_mcts_quality_stats[key][idx] += int(count or 0)
            else:
                total_mcts_quality_stats[key] += float(batch_stats.get(key, 0.0) or 0.0)
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

        total_games = len(game_lengths)
        source_label = self.opponent_source_label
        if len(opponent_source_counts) > 1:
            source_label = 'mixed'
        elif len(opponent_source_counts) == 1:
            source_label = next(iter(opponent_source_counts.keys()))
        avg_search_simulations_used = (
            float(total_search_simulations_used) / float(total_search_samples)
            if total_search_samples > 0
            else 0.0
        )
        p10_search_simulations_used = (
            float(np.percentile(np.asarray(search_simulations_used_samples, dtype=np.float32), 10))
            if search_simulations_used_samples
            else 0.0
        )
        budget_arr = (
            np.asarray(search_simulations_budget_samples, dtype=np.float32)
            if search_simulations_budget_samples
            else None
        )
        p10_search_simulations_budget = float(np.percentile(budget_arr, 10)) if budget_arr is not None else 0.0
        p90_search_simulations_budget = float(np.percentile(budget_arr, 90)) if budget_arr is not None else 0.0
        min_search_simulations_budget = float(np.min(budget_arr)) if budget_arr is not None else 0.0
        max_search_simulations_budget = float(np.max(budget_arr)) if budget_arr is not None else 0.0
        mcts_quality_samples = int(total_mcts_quality_stats['mcts_prior_agreement_samples'])
        mcts_q_delta_samples = int(total_mcts_quality_stats['mcts_q_delta_samples'])
        mcts_q_delta_values = list(total_mcts_quality_stats.get('mcts_q_delta_values', []) or [])
        mcts_q_delta_hist = list(total_mcts_quality_stats.get('mcts_q_delta_hist', []) or [])
        mcts_changed_q_delta_samples = int(total_mcts_quality_stats['mcts_changed_q_delta_samples'])
        mcts_changed_q_delta_values = list(total_mcts_quality_stats.get('mcts_changed_q_delta_values', []) or [])
        mcts_changed_q_delta_hist = list(total_mcts_quality_stats.get('mcts_changed_q_delta_hist', []) or [])
        mcts_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_samples', 0.0) or 0.0)
            changed = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_changed_count', 0.0) or 0.0)
            challenge_eligible = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_scout_challenge_eligible_count', 0.0) or 0.0)
            challenge_used = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_scout_challenge_used_count', 0.0) or 0.0)
            challenge_changed = float(total_mcts_quality_stats.get(f'mcts_phase_{phase}_scout_challenge_changed_count', 0.0) or 0.0)
            mcts_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            mcts_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            mcts_phase_stats[f'mcts_phase_{phase}_scout_challenge_eligible_count'] = int(challenge_eligible)
            mcts_phase_stats[f'mcts_phase_{phase}_scout_challenge_used_count'] = int(challenge_used)
            mcts_phase_stats[f'mcts_phase_{phase}_scout_challenge_changed_count'] = int(challenge_changed)
            mcts_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
            mcts_phase_stats[f'mcts_scout_challenge_eligible_{phase}_rate'] = (
                challenge_eligible / samples if samples > 0.0 else 0.0
            )
            mcts_phase_stats[f'mcts_scout_challenge_used_{phase}_rate'] = (
                challenge_used / samples if samples > 0.0 else 0.0
            )
            mcts_phase_stats[f'mcts_scout_challenge_changed_{phase}_rate'] = (
                challenge_changed / challenge_used if challenge_used > 0.0 else 0.0
            )
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
            'syzygy_ended_games': int(total_syzygy_ended_games),
            'syzygy_probe_positions': int(total_syzygy_probe_positions),
            'syzygy_probe_hits': int(total_syzygy_probe_hits),
            'opponent_source': source_label,
            'opponent_source_counts': opponent_source_counts,
            'opponent_source_results': opponent_source_results,
            'search_simulations_used_avg': float(avg_search_simulations_used),
            'search_simulations_budget_avg': (
                float(total_search_simulations_budget) / float(total_search_samples)
                if total_search_samples > 0
                else 0.0
            ),
            'search_simulations_budget_p10': float(p10_search_simulations_budget),
            'search_simulations_budget_p90': float(p90_search_simulations_budget),
            'search_simulations_budget_min': float(min_search_simulations_budget),
            'search_simulations_budget_max': float(max_search_simulations_budget),
            'search_simulations_used_p10': float(p10_search_simulations_used),
            'search_simulations_used_sum': int(total_search_simulations_used),
            'search_simulations_budget_sum': int(total_search_simulations_budget),
            'search_simulations_used_samples': list(search_simulations_used_samples),
            'search_simulations_budget_samples': list(search_simulations_budget_samples),
            'search_extra_budget_samples': int(total_search_extra_budget_samples),
            'search_extra_budget_rate': (
                float(total_search_extra_budget_samples) / float(total_search_samples)
                if total_search_samples > 0
                else 0.0
            ),
            'search_samples': int(total_search_samples),
            'scout_stopped_early': int(total_scout_stopped_early),
            'scout_stop_rate': (
                float(total_scout_stopped_early) / float(total_search_samples)
                if total_search_samples > 0
                else 0.0
            ),
            'scout_stop_reasons': scout_stop_reasons,
            'mcts_prior_agreement_samples': int(mcts_quality_samples),
            'mcts_prior_agreement_sum': float(total_mcts_quality_stats['mcts_prior_agreement_sum']),
            'mcts_prior_agreement_rate': (
                float(total_mcts_quality_stats['mcts_prior_agreement_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_count': int(total_mcts_quality_stats['mcts_prior_changed_count']),
            'mcts_prior_changed_rate': (
                float(total_mcts_quality_stats['mcts_prior_changed_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_search_change_count': int(total_mcts_quality_stats['mcts_search_change_count']),
            'mcts_search_change_weight_sum': float(total_mcts_quality_stats['mcts_search_change_weight_sum']),
            'mcts_search_change_rate': (
                float(total_mcts_quality_stats['mcts_search_change_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_search_change_weight_mean': (
                float(total_mcts_quality_stats['mcts_search_change_weight_sum'])
                / float(total_mcts_quality_stats['mcts_search_change_count'])
                if int(total_mcts_quality_stats['mcts_search_change_count']) > 0
                else 1.0
            ),
            'mcts_scout_challenge_eligible_count': int(total_mcts_quality_stats['mcts_scout_challenge_eligible_count']),
            'mcts_scout_challenge_eligible_rate': (
                float(total_mcts_quality_stats['mcts_scout_challenge_eligible_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_scout_challenge_used_count': int(total_mcts_quality_stats['mcts_scout_challenge_used_count']),
            'mcts_scout_challenge_used_rate': (
                float(total_mcts_quality_stats['mcts_scout_challenge_used_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_scout_challenge_changed_count': int(total_mcts_quality_stats['mcts_scout_challenge_changed_count']),
            'mcts_scout_challenge_changed_rate': (
                float(total_mcts_quality_stats['mcts_scout_challenge_changed_count'])
                / max(1.0, float(total_mcts_quality_stats['mcts_scout_challenge_used_count']))
                if int(total_mcts_quality_stats['mcts_scout_challenge_used_count']) > 0
                else 0.0
            ),
            'mcts_scout_challenge_score_sum': float(total_mcts_quality_stats['mcts_scout_challenge_score_sum']),
            'mcts_scout_challenge_score_mean': (
                float(total_mcts_quality_stats['mcts_scout_challenge_score_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            **mcts_phase_stats,
            'mcts_policy_uptake_samples': int(total_mcts_quality_stats['mcts_policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(total_mcts_quality_stats['mcts_policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(total_mcts_quality_stats['mcts_policy_uptake_weight_sum'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(total_mcts_quality_stats['mcts_policy_uptake_low_count'])
                / float(total_mcts_quality_stats['mcts_policy_uptake_samples'])
                if int(total_mcts_quality_stats['mcts_policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_prior_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(total_mcts_quality_stats['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_prior_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(total_mcts_quality_stats['mcts_policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(total_mcts_quality_stats['mcts_policy_kl_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(total_mcts_quality_stats['mcts_top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(total_mcts_quality_stats['mcts_top_visit_prob_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(total_mcts_quality_stats['mcts_visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(total_mcts_quality_stats['mcts_visit_gap_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(total_mcts_quality_stats['mcts_visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(total_mcts_quality_stats['mcts_visit_entropy_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(total_mcts_quality_stats['mcts_good_target_count']),
            'mcts_good_target_rate': (
                float(total_mcts_quality_stats['mcts_good_target_count']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(total_mcts_quality_stats['mcts_explored_prior_mass_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(total_mcts_quality_stats['mcts_visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(total_mcts_quality_stats['mcts_visited_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(total_mcts_quality_stats['mcts_legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(total_mcts_quality_stats['mcts_legal_move_count_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(total_mcts_quality_stats['mcts_visit_coverage_ratio_sum']) / float(mcts_quality_samples)
                if mcts_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': int(mcts_q_delta_samples),
            'mcts_q_delta_sum': float(total_mcts_quality_stats['mcts_q_delta_sum']),
            'mcts_q_delta_hist': mcts_q_delta_hist,
            'mcts_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_q_delta_sum']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 10)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 10))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 50)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 50))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_q_delta_hist, 90)
                if sum(mcts_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_q_delta_values, dtype=np.float32), 90))
                if mcts_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(total_mcts_quality_stats['mcts_changed_to_lower_q_count']),
            'mcts_changed_to_lower_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_q_delta_samples)
                if mcts_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(total_mcts_quality_stats['mcts_changed_to_higher_q_count']),
            'mcts_changed_q_delta_samples': int(mcts_changed_q_delta_samples),
            'mcts_changed_q_delta_sum': float(total_mcts_quality_stats['mcts_changed_q_delta_sum']),
            'mcts_changed_q_delta_hist': mcts_changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(total_mcts_quality_stats['mcts_changed_q_delta_sum']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 10)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 10))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 50)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 50))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(mcts_changed_q_delta_hist, 90)
                if sum(mcts_changed_q_delta_hist or []) > 0
                else float(np.percentile(np.asarray(mcts_changed_q_delta_values, dtype=np.float32), 90))
                if mcts_changed_q_delta_values
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_higher_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(total_mcts_quality_stats['mcts_changed_to_lower_q_count']) / float(mcts_changed_q_delta_samples)
                if mcts_changed_q_delta_samples > 0
                else 0.0
            ),
            'total_games': int(total_games),
            'profile': self._aggregate_engine_profile_stats(),
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

        total_games_to_play = max(0, int(batch_size))
        active_limit = min(self.max_batch_games_per_worker, total_games_to_play)
        games_started = 0
        game_states = []
        completed_game_states = []

        def _new_game_state(game_idx):
            game_opponent_label = (
                str(batch_plan_labels[game_idx])
                if batch_plan_labels is not None and game_idx < len(batch_plan_labels)
                else self.opponent_source_label
            )
            opponent_mcts = self.opponent_mcts_by_label.get(game_opponent_label)
            if game_opponent_label != "current" and opponent_mcts is None:
                if game_opponent_label not in self._warned_missing_opponent_labels:
                    print(
                        "Warning: opponent plan references "
                        f"'{game_opponent_label}', but this worker has no model for it. "
                        "Falling back to current."
                    )
                    self._warned_missing_opponent_labels.add(game_opponent_label)
                game_opponent_label = "current"
                opponent_mcts = None
            has_frozen_opponent = opponent_mcts is not None
            gs = {
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
                'syzygy_result': None,
                'syzygy_probe_positions': 0,
                'syzygy_probe_hits': 0,
                'resigned_result': None,
                'resign_streak': 0,
                'resignation_disabled': bool(np.random.random() < self.resignation_disable_fraction),
                'opponent_source_label': game_opponent_label,
                'opponent_mcts': opponent_mcts,
                'has_frozen_opponent': has_frozen_opponent,
                'learner_color': (
                    chess.WHITE
                    if not has_frozen_opponent or not self.randomize_learner_color
                    else (chess.WHITE if np.random.random() < 0.5 else chess.BLACK)
                ),
                '_completion_reported': False,
                '_finalized_for_batch': False,
            }
            self._apply_opening_prefix(gs)
            return gs

        def _fill_active_slots():
            nonlocal games_started
            while games_started < total_games_to_play and len(game_states) < active_limit:
                game_states.append(_new_game_state(games_started))
                games_started += 1

        def _retire_completed_games():
            if not game_states:
                return
            active = []
            for gs in game_states:
                if gs.get('done', False):
                    if not gs.get('_finalized_for_batch', False):
                        gs['_finalized_for_batch'] = True
                        completed_game_states.append(gs)
                else:
                    active.append(gs)
            game_states[:] = active

        _fill_active_slots()

        # Initialize search statistics used across this batch
        search_stats = {
            'sim_used': 0,
            'sim_budget': 0,
            'extra_budget_samples': 0,
            'samples': 0,
            'samples_list': [],
            'budget_samples_list': [],
            'stopped_early': 0,
            'stop_reasons': {},
        }
        target_quality = {
            'samples': 0,
            'changed': 0,
            'agreement_sum': 0.0,
            'prior_top_visit_prob_sum': 0.0,
            'mcts_top_prior_prob_sum': 0.0,
            'policy_kl_sum': 0.0,
            'top_visit_prob_sum': 0.0,
            'visit_gap_sum': 0.0,
            'visit_entropy_sum': 0.0,
            'good_target_count': 0,
            'explored_prior_mass_sum': 0.0,
            'visited_move_count_sum': 0.0,
            'legal_move_count_sum': 0.0,
            'visit_coverage_ratio_sum': 0.0,
            'q_comparable': 0,
            'q_delta_sum': 0.0,
            'q_delta_values': [],
            'changed_to_lower_q': 0,
            'changed_to_higher_q': 0,
            'changed_q_comparable': 0,
            'changed_q_delta_sum': 0.0,
            'changed_q_delta_values': [],
            'search_change_count': 0,
            'search_change_weight_sum': 0.0,
            'policy_uptake_samples': 0,
            'policy_uptake_weight_sum': 0.0,
            'policy_uptake_low_count': 0,
            'scout_challenge_eligible_count': 0,
            'scout_challenge_applied_count': 0,
            'scout_challenge_changed_count': 0,
            'scout_challenge_score_sum': 0.0,
        }
        for phase in ('opening', 'middlegame', 'endgame'):
            target_quality[f'{phase}_samples'] = 0
            target_quality[f'{phase}_changed_count'] = 0
            target_quality[f'{phase}_scout_challenge_eligible_count'] = 0
            target_quality[f'{phase}_scout_challenge_used_count'] = 0
            target_quality[f'{phase}_scout_challenge_changed_count'] = 0

        def _accumulate_target_quality(search_metadata, board):
            if not isinstance(search_metadata, dict):
                return
            agree = search_metadata.get('prior_mcts_agree', None)
            if agree is None:
                return
            target_quality['samples'] += 1
            agree_value = float(agree)
            changed_top = agree_value < 0.5
            challenge_applied = float(search_metadata.get('scout_challenge_applied', 0.0) or 0.0) >= 0.5
            challenge_score = float(search_metadata.get('scout_challenge_score', 0.0) or 0.0)
            mcts_for_challenge_config = getattr(self, 'mcts', None)
            challenge_fraction = float(
                getattr(mcts_for_challenge_config, 'scout_challenge_fraction', 0.0) or 0.0
            )
            challenge_min_score = float(
                getattr(mcts_for_challenge_config, 'scout_challenge_min_score', 1.0) or 1.0
            )
            challenge_eligible = (
                challenge_fraction > 0.0
                and challenge_score >= challenge_min_score
            )
            phase = self._mcts_phase_for_board(board)
            target_quality[f'{phase}_samples'] += 1
            if changed_top:
                target_quality[f'{phase}_changed_count'] += 1
            if challenge_eligible:
                target_quality[f'{phase}_scout_challenge_eligible_count'] += 1
            if challenge_applied:
                target_quality[f'{phase}_scout_challenge_used_count'] += 1
                if changed_top:
                    target_quality[f'{phase}_scout_challenge_changed_count'] += 1
            target_quality['agreement_sum'] += agree_value
            if changed_top:
                target_quality['changed'] += 1
            for meta_key, sum_key in [
                ('prior_top_visit_prob', 'prior_top_visit_prob_sum'),
                ('mcts_top_prior_prob', 'mcts_top_prior_prob_sum'),
                ('mcts_policy_kl', 'policy_kl_sum'),
                ('top_visit_prob', 'top_visit_prob_sum'),
                ('visit_gap', 'visit_gap_sum'),
                ('visit_entropy', 'visit_entropy_sum'),
                ('explored_prior_mass', 'explored_prior_mass_sum'),
            ]:
                value = search_metadata.get(meta_key, None)
                if value is not None:
                    target_quality[sum_key] += float(value)
            target_quality['scout_challenge_applied_count'] += int(challenge_applied)
            target_quality['scout_challenge_eligible_count'] += int(challenge_eligible)
            if challenge_applied and changed_top:
                target_quality['scout_challenge_changed_count'] += 1
            target_quality['scout_challenge_score_sum'] += challenge_score
            try:
                top_visit_prob = float(search_metadata.get('top_visit_prob', 0.0) or 0.0)
                visit_gap = float(search_metadata.get('visit_gap', 0.0) or 0.0)
            except (TypeError, ValueError):
                top_visit_prob = 0.0
                visit_gap = 0.0
            if (
                top_visit_prob >= float(self.mcts_good_target_min_top_visit_prob)
                and visit_gap >= float(self.mcts_good_target_min_visit_gap)
            ):
                target_quality['good_target_count'] += 1
            visited_count = search_metadata.get('visited_move_count', None)
            legal_count = search_metadata.get('legal_move_count', None)
            if visited_count is not None and legal_count is not None:
                visited_count = float(visited_count)
                legal_count = float(legal_count)
                target_quality['visited_move_count_sum'] += float(visited_count)
                target_quality['legal_move_count_sum'] += float(legal_count)
                if legal_count > 0:
                    target_quality['visit_coverage_ratio_sum'] += float(visited_count) / float(legal_count)
            q_delta = search_metadata.get('mcts_q_delta', None)
            if q_delta is not None:
                target_quality['q_comparable'] += 1
                q_delta_value = float(q_delta)
                target_quality['q_delta_sum'] += q_delta_value
                target_quality['q_delta_values'].append(q_delta_value)
                if changed_top:
                    target_quality['changed_q_comparable'] += 1
                    target_quality['changed_q_delta_sum'] += q_delta_value
                    target_quality['changed_q_delta_values'].append(q_delta_value)
                    if q_delta_value < -0.02:
                        target_quality['changed_to_lower_q'] += 1
                    elif q_delta_value > 0.02:
                        target_quality['changed_to_higher_q'] += 1

        while len(completed_game_states) < total_games_to_play:
            active_indices = []
            learner_indices = []
            grouped_opponent_indices = {}
            for i, gs in enumerate(game_states):
                if gs['done']:
                    continue
                board = gs['board']
                syzygy_t0 = time.perf_counter() if self.profile_enabled else None
                self._maybe_finish_with_syzygy(gs, board)
                if self.profile_enabled:
                    self._profile_add('syzygy_time', time.perf_counter() - syzygy_t0)
                    self._profile_inc('syzygy_calls', 1)
                if gs['done']:
                    continue

                active_indices.append(i)
                if not gs.get('has_frozen_opponent', False):
                    gs['_learner_turn_cache'] = True
                    learner_indices.append(i)
                    continue

                learner_turn = bool(board.turn == gs.get('learner_color', chess.WHITE))
                gs['_learner_turn_cache'] = learner_turn
                if learner_turn:
                    learner_indices.append(i)
                else:
                    label = str(gs.get('opponent_source_label', self.opponent_source_label) or "current")
                    grouped_opponent_indices.setdefault(label, []).append(i)

            if not active_indices:
                _retire_completed_games()
                _fill_active_slots()
                if not game_states:
                    break
                continue

            visit_counts_by_index = {}

            def _run_search_for_indices(indices, mcts_ref, root_key, synced_key):
                if not indices:
                    return
                group_states = []
                for i in indices:
                    gs = game_states[i]
                    group_states.append([
                        gs['board'],
                        gs.get(root_key),
                        bool(gs.get(synced_key, False)),
                        gs['board_history'],
                        int(gs.get('move_count', 0) or 0),
                    ])
                visit_counts_list_group, search_metadata_group = mcts_ref.search_many(
                    group_states,
                    num_simulations=self.num_simulations,
                    add_root_noise=True,
                    return_search_metadata=True,
                )
                for gs_idx, local_state, visit_counts, search_metadata in zip(
                    indices,
                    group_states,
                    visit_counts_list_group,
                    search_metadata_group,
                ):
                    gs = game_states[gs_idx]
                    gs[root_key] = local_state[1]
                    gs[synced_key] = bool(local_state[2])
                    visit_counts_by_index[gs_idx] = (visit_counts, search_metadata)
                    used = int(search_metadata.get('simulations_used', 0)) if isinstance(search_metadata, dict) else 0
                    search_stats['sim_used'] += used
                    if isinstance(search_metadata, dict):
                        budget = int(search_metadata.get('simulation_budget', self.num_simulations) or self.num_simulations)
                        search_stats['sim_budget'] += budget
                        search_stats['budget_samples_list'].append(budget)
                        if bool(search_metadata.get('simulation_budget_extra', False)):
                            search_stats['extra_budget_samples'] += 1
                    else:
                        search_stats['sim_budget'] += int(self.num_simulations)
                        search_stats['budget_samples_list'].append(int(self.num_simulations))
                    search_stats['samples'] += 1
                    search_stats['samples_list'].append(used)
                    if isinstance(search_metadata, dict):
                        reason = str(search_metadata.get('scout_stop_reason', 'unknown') or 'unknown')
                        search_stats['stop_reasons'][reason] = int(search_stats['stop_reasons'].get(reason, 0)) + 1
                        if bool(search_metadata.get('stopped_early', False)):
                            search_stats['stopped_early'] += 1

            if not self.opponent_mcts_by_label:
                _run_search_for_indices(active_indices, self.mcts, 'root', '_root_synced')
            else:
                _run_search_for_indices(learner_indices, self.mcts, 'root', '_root_synced')
                for label, indices in grouped_opponent_indices.items():
                    opponent_mcts = self.opponent_mcts_by_label.get(label)
                    if opponent_mcts is None:
                        continue
                    _run_search_for_indices(indices, opponent_mcts, 'opponent_root', '_opponent_root_synced')

            for idx in active_indices:
                gs = game_states[idx]
                board = gs['board']
                visit_payload = visit_counts_by_index.get(idx, None)
                if visit_payload is None:
                    continue
                visit_counts, search_metadata = visit_payload

                if self.temp_threshold > 0 and gs['move_count'] < self.temp_threshold:
                    temperature = self.temperature
                else:
                    temperature = 0.0
                learner_turn = bool(gs.get('_learner_turn_cache', True))
                game_opponent_mcts = gs.get('opponent_mcts')
                root_key = 'root' if learner_turn or game_opponent_mcts is None else 'opponent_root'
                synced_key = '_root_synced' if learner_turn or game_opponent_mcts is None else '_opponent_root_synced'
                root = gs.get(root_key)
                adjudication_t0 = time.perf_counter() if self.profile_enabled else None
                adjudicated_result = self._maybe_adjudicate_game(gs, root, board)
                if self.profile_enabled:
                    self._profile_add('adjudication_time', time.perf_counter() - adjudication_t0)
                    self._profile_inc('adjudication_calls', 1)
                if adjudicated_result is not None:
                    gs['adjudicated_result'] = adjudicated_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                resigned_result = self._maybe_resign_game(gs, root, board)
                if resigned_result is not None:
                    gs['resigned_result'] = resigned_result
                    gs['done'] = True
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
                    continue

                move_t0 = time.perf_counter() if self.profile_enabled else None
                move = self._select_move_from_visits(visit_counts, temperature)
                if self.profile_enabled:
                    self._profile_add('move_selection_time', time.perf_counter() - move_t0)
                    self._profile_inc('move_selection_calls', 1)
                store_policy_position = self._should_store_policy_position(
                    learner_turn,
                    game_opponent_mcts,
                    gs.get('opponent_source_label', self.opponent_source_label),
                )
                if store_policy_position:
                    _accumulate_target_quality(search_metadata, board)
                    policy_t0 = time.perf_counter() if self.profile_enabled else None
                    raw_visit_counts = visit_counts
                    top1, entropy = self._policy_target_quality_from_visits(raw_visit_counts)
                    target_quality_weight = self._policy_target_quality_weight(top1, entropy)
                    visit_counts = self._prune_policy_target_visits(visit_counts)
                    policy_indices, policy_values = _build_sparse_policy_target_from_visits(visit_counts, board)
                    history_count = len(gs['board_history'])
                    importance_score = self._compute_position_importance(board, move, visit_counts, root)
                    root_value = 0.0
                    policy_weight = float(search_metadata.get('policy_weight', 1.0)) if isinstance(search_metadata, dict) else 1.0
                    policy_weight *= float(target_quality_weight)
                    policy_uptake_weight = float(self._policy_target_uptake_weight(search_metadata))
                    policy_weight *= policy_uptake_weight
                    target_quality['policy_uptake_samples'] += 1
                    target_quality['policy_uptake_weight_sum'] += policy_uptake_weight
                    if policy_uptake_weight < 0.999:
                        target_quality['policy_uptake_low_count'] += 1
                    change_weight = self._policy_target_search_change_weight(search_metadata)
                    policy_weight *= float(change_weight)
                    if change_weight > 1.0:
                        importance_score *= min(float(change_weight), 1.35)
                        target_quality['search_change_count'] += 1
                        target_quality['search_change_weight_sum'] += float(change_weight)
                    if not learner_turn and game_opponent_mcts is not None:
                        if str(gs.get('opponent_source_label', '')) == "anchor":
                            policy_weight *= float(self.frozen_anchor_policy_weight)
                        else:
                            policy_weight *= float(self.frozen_opponent_policy_weight)
                    replay_source_code = _replay_source_code(
                        learner_turn,
                        game_opponent_mcts,
                        gs.get('opponent_source_label', self.opponent_source_label),
                    )
                    if root is not None:
                        root_visits = int(getattr(root, 'visit_count', 0) or 0)
                        if root_visits > 0:
                            root_value = float(root.value_sum / max(1, root_visits))
                    gs['game_history'].append((
                        history_count,
                        policy_indices,
                        policy_values,
                        board.turn,
                        importance_score,
                        root_value,
                        policy_weight,
                        replay_source_code,
                    ))
                    if self.profile_enabled:
                        self._profile_add('policy_target_build_time', time.perf_counter() - policy_t0)
                        self._profile_inc('policy_target_build_calls', 1)

                # Update history BEFORE making the move
                # Store cached tensors for both POVs to avoid repeated FEN parse + tensor rebuild.
                gs['board_history'].append(self.mcts._encode_history_entry(board))

                # Keep both side-specific MCTS trees synchronized with the
                # actual game line. Mixed-opponent self-play otherwise starts
                # every move from a fresh root and produces noisier targets.
                for candidate_root_key, candidate_synced_key in [
                    ('root', '_root_synced'),
                    ('opponent_root', '_opponent_root_synced'),
                ]:
                    next_root, next_synced = self._advance_search_root(
                        gs.get(candidate_root_key),
                        move,
                    )
                    gs[candidate_root_key] = next_root
                    gs[candidate_synced_key] = next_synced

                board.push(move)
                gs['move_count'] += 1

                syzygy_t0 = time.perf_counter() if self.profile_enabled else None
                if self._maybe_finish_with_syzygy(gs, board) is not None:
                    if self.profile_enabled:
                        self._profile_add('syzygy_time', time.perf_counter() - syzygy_t0)
                        self._profile_inc('syzygy_calls', 1)
                    continue
                if self.profile_enabled:
                    self._profile_add('syzygy_time', time.perf_counter() - syzygy_t0)
                    self._profile_inc('syzygy_calls', 1)

                forced_game_over = board.is_game_over(claim_draw=False)
                auto_claim_draw = self._should_auto_claim_draw(board, gs['move_count'])
                if forced_game_over or auto_claim_draw or gs['move_count'] >= max_moves:
                    gs['done'] = True
                    gs['ended_by_auto_claim_draw'] = auto_claim_draw
                    # Free up memory immediately
                    self._clear_game_search_state(gs)
                    self._mark_game_completed(gs)
            _retire_completed_games()
            _fill_active_slots()

        positions = []
        game_lengths = []
        dropped_positions = 0
        truncated_games = 0
        claimable_draw_ended_games = 0
        adjudicated_games = 0
        syzygy_ended_games = 0
        syzygy_probe_positions = 0
        syzygy_probe_hits = 0
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

        for gs in completed_game_states:
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
            syzygy_result = gs.get('syzygy_result', None)
            if syzygy_result is not None:
                syzygy_ended_games += 1
            syzygy_probe_positions += int(gs.get('syzygy_probe_positions', 0) or 0)
            syzygy_probe_hits += int(gs.get('syzygy_probe_hits', 0) or 0)
            resigned_result = gs.get('resigned_result', None)
            if resigned_result is not None:
                resigned_games += 1
            result = (
                resigned_result
                if resigned_result is not None
                else (
                    adjudicated_result
                    if adjudicated_result is not None
                    else (
                        syzygy_result
                        if syzygy_result is not None
                        else ('1/2-1/2' if ended_by_claimable_draw else board.result(claim_draw=False))
                    )
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
        target_quality_samples = int(target_quality['samples'])
        target_quality_q_samples = int(target_quality['q_comparable'])
        target_quality_changed_q_samples = int(target_quality['changed_q_comparable'])
        q_delta_values = list(target_quality.get('q_delta_values', []) or [])
        q_delta_hist = _q_delta_histogram(q_delta_values)
        changed_q_delta_values = list(target_quality.get('changed_q_delta_values', []) or [])
        changed_q_delta_hist = _q_delta_histogram(changed_q_delta_values)
        target_phase_stats = {}
        for phase in ('opening', 'middlegame', 'endgame'):
            samples = float(target_quality.get(f'{phase}_samples', 0) or 0)
            changed = float(target_quality.get(f'{phase}_changed_count', 0) or 0)
            challenge_eligible = float(target_quality.get(f'{phase}_scout_challenge_eligible_count', 0) or 0)
            challenge_used = float(target_quality.get(f'{phase}_scout_challenge_used_count', 0) or 0)
            challenge_changed = float(target_quality.get(f'{phase}_scout_challenge_changed_count', 0) or 0)
            target_phase_stats[f'mcts_phase_{phase}_samples'] = int(samples)
            target_phase_stats[f'mcts_phase_{phase}_changed_count'] = int(changed)
            target_phase_stats[f'mcts_phase_{phase}_scout_challenge_eligible_count'] = int(challenge_eligible)
            target_phase_stats[f'mcts_phase_{phase}_scout_challenge_used_count'] = int(challenge_used)
            target_phase_stats[f'mcts_phase_{phase}_scout_challenge_changed_count'] = int(challenge_changed)
            target_phase_stats[f'mcts_changed_{phase}_rate'] = changed / samples if samples > 0.0 else 0.0
            target_phase_stats[f'mcts_scout_challenge_eligible_{phase}_rate'] = (
                challenge_eligible / samples if samples > 0.0 else 0.0
            )
            target_phase_stats[f'mcts_scout_challenge_used_{phase}_rate'] = (
                challenge_used / samples if samples > 0.0 else 0.0
            )
            target_phase_stats[f'mcts_scout_challenge_changed_{phase}_rate'] = (
                challenge_changed / challenge_used if challenge_used > 0.0 else 0.0
            )
        return positions, game_lengths, {
            'total_games': int(len(completed_game_states)),
            'truncated_games': int(truncated_games),
            'dropped_positions': int(dropped_positions),
            'claimable_draw_ended_games': int(claimable_draw_ended_games),
            'adjudicated_games': int(adjudicated_games),
            'syzygy_ended_games': int(syzygy_ended_games),
            'syzygy_probe_positions': int(syzygy_probe_positions),
            'syzygy_probe_hits': int(syzygy_probe_hits),
            'search_simulations_used_sum': int(search_stats['sim_used']),
            'search_simulations_budget_sum': int(search_stats['sim_budget']),
            'search_samples': int(search_stats['samples']),
            'search_simulations_used_samples': list(search_stats['samples_list']),
            'search_simulations_budget_samples': list(search_stats['budget_samples_list']),
            'search_extra_budget_samples': int(search_stats['extra_budget_samples']),
            'search_extra_budget_rate': (
                float(search_stats['extra_budget_samples']) / float(search_stats['samples'])
                if int(search_stats['samples']) > 0
                else 0.0
            ),
            'scout_stopped_early': int(search_stats['stopped_early']),
            'scout_stop_reasons': dict(search_stats['stop_reasons']),
            'mcts_prior_agreement_samples': target_quality_samples,
            'mcts_prior_agreement_sum': float(target_quality['agreement_sum']),
            'mcts_prior_changed_count': int(target_quality['changed']),
            'mcts_prior_agreement_rate': (
                float(target_quality['agreement_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_prior_changed_rate': (
                float(target_quality['changed']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_search_change_count': int(target_quality['search_change_count']),
            'mcts_search_change_weight_sum': float(target_quality['search_change_weight_sum']),
            'mcts_search_change_rate': (
                float(target_quality['search_change_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_search_change_weight_mean': (
                float(target_quality['search_change_weight_sum']) / float(target_quality['search_change_count'])
                if int(target_quality['search_change_count']) > 0
                else 1.0
            ),
            'mcts_scout_challenge_eligible_count': int(target_quality['scout_challenge_eligible_count']),
            'mcts_scout_challenge_eligible_rate': (
                float(target_quality['scout_challenge_eligible_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_scout_challenge_used_count': int(target_quality['scout_challenge_applied_count']),
            'mcts_scout_challenge_used_rate': (
                float(target_quality['scout_challenge_applied_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_scout_challenge_changed_count': int(target_quality['scout_challenge_changed_count']),
            'mcts_scout_challenge_changed_rate': (
                float(target_quality['scout_challenge_changed_count'])
                / max(1.0, float(target_quality['scout_challenge_applied_count']))
                if int(target_quality['scout_challenge_applied_count']) > 0
                else 0.0
            ),
            'mcts_scout_challenge_score_sum': float(target_quality['scout_challenge_score_sum']),
            'mcts_scout_challenge_score_mean': (
                float(target_quality['scout_challenge_score_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            **target_phase_stats,
            'mcts_policy_uptake_samples': int(target_quality['policy_uptake_samples']),
            'mcts_policy_uptake_weight_sum': float(target_quality['policy_uptake_weight_sum']),
            'mcts_policy_uptake_low_count': int(target_quality['policy_uptake_low_count']),
            'mcts_policy_uptake_weight_mean': (
                float(target_quality['policy_uptake_weight_sum']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 1.0
            ),
            'mcts_policy_uptake_low_rate': (
                float(target_quality['policy_uptake_low_count']) / float(target_quality['policy_uptake_samples'])
                if int(target_quality['policy_uptake_samples']) > 0
                else 0.0
            ),
            'mcts_prior_top_visit_prob_sum': float(target_quality['prior_top_visit_prob_sum']),
            'mcts_prior_top_visit_prob_mean': (
                float(target_quality['prior_top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_prior_prob_sum': float(target_quality['mcts_top_prior_prob_sum']),
            'mcts_top_prior_prob_mean': (
                float(target_quality['mcts_top_prior_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_policy_kl_sum': float(target_quality['policy_kl_sum']),
            'mcts_policy_kl_mean': (
                float(target_quality['policy_kl_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_top_visit_prob_sum': float(target_quality['top_visit_prob_sum']),
            'mcts_top_visit_prob_mean': (
                float(target_quality['top_visit_prob_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_gap_sum': float(target_quality['visit_gap_sum']),
            'mcts_visit_gap_mean': (
                float(target_quality['visit_gap_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_entropy_sum': float(target_quality['visit_entropy_sum']),
            'mcts_visit_entropy_mean': (
                float(target_quality['visit_entropy_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_good_target_count': int(target_quality['good_target_count']),
            'mcts_good_target_rate': (
                float(target_quality['good_target_count']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_explored_prior_mass_sum': float(target_quality['explored_prior_mass_sum']),
            'mcts_explored_prior_mass_mean': (
                float(target_quality['explored_prior_mass_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visited_move_count_sum': float(target_quality['visited_move_count_sum']),
            'mcts_visited_move_count_mean': (
                float(target_quality['visited_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_legal_move_count_sum': float(target_quality['legal_move_count_sum']),
            'mcts_legal_move_count_mean': (
                float(target_quality['legal_move_count_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_visit_coverage_ratio_sum': float(target_quality['visit_coverage_ratio_sum']),
            'mcts_visit_coverage_ratio_mean': (
                float(target_quality['visit_coverage_ratio_sum']) / float(target_quality_samples)
                if target_quality_samples > 0
                else 0.0
            ),
            'mcts_q_delta_samples': target_quality_q_samples,
            'mcts_q_delta_sum': float(target_quality['q_delta_sum']),
            'mcts_q_delta_values': q_delta_values,
            'mcts_q_delta_hist': q_delta_hist,
            'mcts_q_delta_mean': (
                float(target_quality['q_delta_sum']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_q_delta_p10': (
                _q_delta_percentile_from_histogram(q_delta_hist, 10)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p50': (
                _q_delta_percentile_from_histogram(q_delta_hist, 50)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_q_delta_p90': (
                _q_delta_percentile_from_histogram(q_delta_hist, 90)
                if sum(q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_count': int(target_quality['changed_to_lower_q']),
            'mcts_changed_to_lower_q_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_q_samples)
                if target_quality_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_count': int(target_quality['changed_to_higher_q']),
            'mcts_changed_q_delta_samples': int(target_quality_changed_q_samples),
            'mcts_changed_q_delta_sum': float(target_quality['changed_q_delta_sum']),
            'mcts_changed_q_delta_values': changed_q_delta_values,
            'mcts_changed_q_delta_hist': changed_q_delta_hist,
            'mcts_changed_q_delta_mean': (
                float(target_quality['changed_q_delta_sum']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p10': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 10)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p50': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 50)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_q_delta_p90': (
                _q_delta_percentile_from_histogram(changed_q_delta_hist, 90)
                if sum(changed_q_delta_hist or []) > 0
                else 0.0
            ),
            'mcts_changed_to_higher_q_rate': (
                float(target_quality['changed_to_higher_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
            'mcts_changed_to_lower_q_when_changed_rate': (
                float(target_quality['changed_to_lower_q']) / float(target_quality_changed_q_samples)
                if target_quality_changed_q_samples > 0
                else 0.0
            ),
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

    _configure_inductor_for_selfplay(config)

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
    from src.model import normalize_state_dict_keys

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

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing = [
        key for key in missing
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    unexpected = [
        key for key in unexpected
        if not (key.endswith('coord_x') or key.endswith('coord_y'))
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Worker {rank}: incompatible state_dict. "
            f"Missing={missing}, Unexpected={unexpected}"
        )


class _RemoteInferenceModel:
    """Small model-like proxy used by MCTS workers with central GPU inference."""

    def __init__(
        self,
        model_label,
        request_queue,
        response_queue,
        worker_rank,
        timeout_s=120.0,
        stall_warning_s=60.0,
        debug_enabled=False,
        transport_dtype="float16",
    ):
        self.model_label = str(model_label or "learner")
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_rank = int(worker_rank)
        timeout_s = float(timeout_s)
        self.timeout_s = None if timeout_s <= 0.0 else max(1.0, timeout_s)
        self.stall_warning_s = max(0.0, float(stall_warning_s))
        self.debug_enabled = bool(debug_enabled)
        self._request_counter = 0
        self._training = False
        self.last_server_batch_size = 0
        self._printed_first_request = False
        self._printed_first_response = False
        self.supports_remote_legal_gather = True
        self.last_response_compact_policy = False
        self.last_request_put_s = 0.0
        self.last_remote_wait_s = 0.0
        self.last_server_queue_wait_s = 0.0
        self.last_server_total_s = 0.0
        self.last_server_concat_s = 0.0
        self.last_server_h2d_s = 0.0
        self.last_server_forward_s = 0.0
        self.last_server_d2h_s = 0.0
        self.last_server_send_s = 0.0
        transport_dtype = str(transport_dtype or "float16").lower()
        self.transport_dtype = "float32" if transport_dtype in {"float32", "fp32"} else "float16"
        self._transport_torch_dtype = torch.float32 if self.transport_dtype == "float32" else torch.float16
        self._transport_np_dtype = np.float32 if self.transport_dtype == "float32" else np.float16

    @property
    def training(self):
        return self._training

    def eval(self):
        self._training = False
        return self

    def train(self, mode=True):
        self._training = bool(mode)
        return self

    def to(self, *args, **kwargs):
        return self

    def _receive_response(self, wait_s):
        if hasattr(self.response_queue, "poll") and hasattr(self.response_queue, "recv"):
            if not self.response_queue.poll(wait_s):
                raise queue.Empty()
            return self.response_queue.recv()
        return self.response_queue.get(timeout=wait_s)

    def _ts(self):
        return time.strftime("%H:%M:%S")

    def __call__(self, board_tensors, apply_log_softmax=False, legal_index_matrix=None, **_kwargs):
        if apply_log_softmax:
            raise ValueError("Remote inference proxy only supports raw policy logits.")
        self._request_counter += 1
        request_id = f"{os.getpid()}_{id(self)}_{self._request_counter}"
        if isinstance(board_tensors, torch.Tensor):
            boards_np = np.array(
                board_tensors.detach().to('cpu', dtype=self._transport_torch_dtype).contiguous().numpy(),
                dtype=self._transport_np_dtype,
                copy=True,
                order="C",
            )
        else:
            boards_np = np.array(board_tensors, dtype=self._transport_np_dtype, copy=True, order="C")
        request = {
            "cmd": "infer",
            "rank": self.worker_rank,
            "request_id": request_id,
            "model_label": self.model_label,
            "boards": boards_np,
        }
        if legal_index_matrix is not None:
            request["legal_index_matrix"] = np.array(legal_index_matrix, dtype=np.int16, copy=True, order="C")
        request["transport_dtype"] = self.transport_dtype
        request["queued_at"] = time.time()
        put_t0 = time.perf_counter()
        self.request_queue.put(request)
        self.last_request_put_s = time.perf_counter() - put_t0
        if self.debug_enabled and not self._printed_first_request:
            self._printed_first_request = True
            print(
                f"[{self._ts()}] Central inference: worker {self.worker_rank} sent first "
                f"{self.model_label} request ({int(boards_np.shape[0])} positions).",
                flush=True,
            )
        started_at = time.time()
        deadline = None if self.timeout_s is None else started_at + self.timeout_s
        next_warning_at = (
            started_at + self.stall_warning_s
            if self.stall_warning_s > 0.0
            else None
        )
        while True:
            now = time.time()
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(f"Central inference timed out for model '{self.model_label}'.")
                wait_s = min(max(remaining, 0.01), 1.0)
            else:
                wait_s = 1.0
            if self.debug_enabled and next_warning_at is not None and now >= next_warning_at:
                waited_s = now - started_at
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} still waiting "
                    f"{waited_s:.0f}s for model '{self.model_label}'.",
                    flush=True,
                )
                next_warning_at = now + self.stall_warning_s
            try:
                response = self._receive_response(wait_s)
            except queue.Empty:
                continue
            if response.get("request_id") != request_id:
                response_request_id = str(response.get("request_id") or "")
                if not response_request_id.startswith(f"{os.getpid()}_"):
                    continue
                raise RuntimeError(
                    "Central inference response routing failed: "
                    f"worker {self.worker_rank} received response for another request."
                )
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "central inference failed"))
            self.last_remote_wait_s = time.time() - started_at
            self.last_server_batch_size = int(response.get("server_batch_size", 0) or 0)
            self.last_response_compact_policy = bool(response.get("compact_policy", False))
            self.last_server_queue_wait_s = float(response.get("server_queue_wait_s", 0.0) or 0.0)
            self.last_server_total_s = float(response.get("server_total_time_s", 0.0) or 0.0)
            self.last_server_concat_s = float(response.get("server_concat_time_s", 0.0) or 0.0)
            self.last_server_h2d_s = float(response.get("server_h2d_time_s", 0.0) or 0.0)
            self.last_server_forward_s = float(response.get("server_forward_time_s", 0.0) or 0.0)
            self.last_server_d2h_s = float(response.get("server_d2h_time_s", 0.0) or 0.0)
            self.last_server_send_s = float(response.get("server_send_time_s", 0.0) or 0.0)
            if self.debug_enabled and not self._printed_first_response:
                self._printed_first_response = True
                print(
                    f"[{self._ts()}] Central inference: worker {self.worker_rank} received first "
                    f"{self.model_label} response (server_batch={self.last_server_batch_size}).",
                    flush=True,
                )
            policy_dtype = np.float16 if self.last_response_compact_policy else np.float32
            policy = torch.from_numpy(np.array(response["policy_logits"], dtype=policy_dtype, copy=True, order="C"))
            value = torch.from_numpy(np.array(response["value_logits"], dtype=np.float32, copy=True, order="C"))
            return policy, value


def central_inference_server(config, device_id, request_queue, response_queues, control_queue=None, server_rank=None):
    """Own GPU inference and batch requests coming from self-play workers."""
    rl_cfg = config.get('reinforcement_learning', {})
    _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
    max_batch = max(1, int(rl_cfg.get('self_play_central_inference_max_batch_size', rl_cfg.get('mcts_batch_size', 256))))
    flush_ms = max(0.0, float(rl_cfg.get('self_play_central_inference_flush_ms', 5.0)))
    debug_enabled = bool(
        debug_root_enabled and central_debug_cfg.get(
            'verbose',
            rl_cfg.get('self_play_central_inference_debug', False),
        )
    )
    quiet_startup = True
    cache_enabled = bool(rl_cfg.get('self_play_central_inference_cache_enabled', True))
    cache_max_entries = max(0, int(rl_cfg.get('self_play_central_inference_cache_entries', 4096)))
    central_use_compile = bool(rl_cfg.get('self_play_central_inference_use_compile', False))
    transport_dtype = str(rl_cfg.get('self_play_central_inference_transport_dtype', 'float16') or 'float16').lower()
    transport_np_dtype = np.float32 if transport_dtype in {'float32', 'fp32'} else np.float16

    try:
        device = _configure_selfplay_worker_runtime(config, device_id)
        try:
            compile_rank = int(server_rank)
        except Exception:
            compile_rank = 900000 + int(os.getpid())
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = bool(
                rl_cfg.get('self_play_central_inference_cudnn_benchmark', False)
            )
        models = {}
        caches = {}
        compiled_base_models = {}
        compiled_wrappers = {}

        def _cache_key(model_label, board_np):
            digest = hashlib.blake2b(np.ascontiguousarray(board_np).view(np.uint8), digest_size=16).digest()
            return (str(model_label), digest)

        def _ts():
            return time.strftime("%H:%M:%S")

        if not quiet_startup:
            print(
                f"[{_ts()}] Central inference: server ready on {device} "
                f"(pid={os.getpid()}, compile_rank={compile_rank}, max_batch={max_batch}, flush={flush_ms:.1f}ms, "
                f"transport={transport_np_dtype.__name__}, "
                f"compile={'on' if central_use_compile else 'off'}, "
                f"cudnn.benchmark={torch.backends.cudnn.benchmark if device.type == 'cuda' else 'n/a'}).",
                flush=True,
            )

        def _warmup_central_model(model, label):
            if not central_use_compile or device.type != 'cuda':
                return None
            raw_batches = rl_cfg.get('self_play_central_inference_compile_warmup_batches', [1, 32, 128, max_batch])
            try:
                warmup_batches = [
                    max(1, int(batch_size))
                    for batch_size in list(raw_batches or [])
                ]
            except Exception:
                warmup_batches = [1, 32, 128, max_batch]
            warmup_batches = sorted({batch_size for batch_size in warmup_batches if batch_size > 0})
            if not warmup_batches:
                return None
            history_positions = int(config.get('model', {}).get('history_positions', 0) or 0)
            input_planes = 16 * (1 + history_positions)
            use_amp = bool(config.get('hardware', {}).get('use_amp', False))
            amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
            dummy_dtype = torch.float16 if use_amp and transport_np_dtype == np.float16 else torch.float32
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warming compiled model '{label}' "
                    f"for batches={warmup_batches}...",
                    flush=True,
                )
            warmup_t0 = time.perf_counter()
            with torch.inference_mode():
                for batch_size in warmup_batches:
                    dummy_input = torch.zeros(
                        batch_size,
                        input_planes,
                        8,
                        8,
                        device=device,
                        dtype=dummy_dtype,
                    ).to(memory_format=torch.channels_last)
                    with torch.autocast(device_type='cuda', enabled=use_amp, dtype=amp_dtype):
                        model(dummy_input, apply_log_softmax=False)
                    torch.cuda.synchronize(device)
            warmup_s = time.perf_counter() - warmup_t0
            if debug_enabled:
                print(
                    f"[{_ts()}] Central inference: warmup for '{label}' done "
                    f"in {warmup_s:.2f}s.",
                    flush=True,
                )
            return warmup_s

        def _load_model(label, state):
            label = str(label or "learner")
            load_info = {
                "label": label,
                "compiled": bool(central_use_compile),
                "reused": False,
                "warmup_s": None,
            }
            if debug_enabled:
                print(f"[{_ts()}] Central inference: loading model '{label}' on {device}...", flush=True)
            if central_use_compile:
                if label in compiled_wrappers and label in compiled_base_models:
                    base_model = compiled_base_models[label]
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = compiled_wrappers[label]
                    load_info["reused"] = True
                    if debug_enabled:
                        print(f"[{_ts()}] Central inference: reused compiled model '{label}'.", flush=True)
                else:
                    base_model = _build_selfplay_worker_model(config, device)
                    _load_worker_model_state(base_model, state, rank=-1)
                    base_model.eval()
                    model = _maybe_compile_selfplay_model(
                        base_model,
                        config,
                        device,
                        rank=compile_rank,
                        model_label=f"central:{label}",
                    )
                    compiled_base_models[label] = base_model
                    if model is not base_model:
                        load_info["warmup_s"] = _warmup_central_model(model, label)
                        compiled_wrappers[label] = model
                    else:
                        load_info["compiled"] = False
            else:
                model = _build_selfplay_worker_model(config, device)
                _load_worker_model_state(model, state, rank=-1)
                model.eval()
            models[label] = model
            caches[label] = OrderedDict()
            if debug_enabled:
                print(f"[{_ts()}] Central inference: model '{label}' ready.", flush=True)
            return load_info

        def _response_queue_for(req):
            if isinstance(response_queues, dict):
                rank = int(req.get("rank", -1))
                response_queue = response_queues.get(rank)
                if response_queue is None:
                    raise RuntimeError(f"Central inference has no response channel for worker {rank}.")
                return response_queue
            return response_queues

        def _send_response(response_channel, response):
            if hasattr(response_channel, "send"):
                response_channel.send(response)
            else:
                response_channel.put(response)

        def _put_response(req, payload):
            if payload.get("ok", False):
                if "policy_logits" in payload:
                    payload["policy_logits"] = np.array(
                        payload["policy_logits"],
                        dtype=np.float16,
                        copy=True,
                        order="C",
                    )
                if "value_logits" in payload:
                    payload["value_logits"] = np.array(
                        payload["value_logits"],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
            response = {
                "request_id": req.get("request_id"),
                "rank": int(req.get("rank", -1)),
            }
            response.update(payload)
            _send_response(_response_queue_for(req), response)

        def _infer_group(model_label, requests):
            group_t0 = time.perf_counter()
            concat_time = 0.0
            h2d_time = 0.0
            forward_time = 0.0
            d2h_time = 0.0
            send_time = 0.0
            model_label = str(model_label)
            model = models.get(model_label)
            if model is None:
                raise RuntimeError(f"Central inference has no model loaded for label '{model_label}'.")

            concat_t0 = time.perf_counter()
            use_amp = bool(config.get('hardware', {}).get('use_amp', False) and device.type == 'cuda')
            amp_dtype = torch.bfloat16 if config.get('hardware', {}).get('use_bfloat16', False) else torch.float16
            input_np_dtype = transport_np_dtype if use_amp else np.float32
            board_batches = [np.asarray(req["boards"], dtype=input_np_dtype) for req in requests]
            batch_sizes = [int(batch.shape[0]) for batch in board_batches]
            boards = np.concatenate(board_batches, axis=0)
            total_n = int(boards.shape[0])
            response_started_at = time.time()
            queue_waits = [
                max(0.0, response_started_at - float(req.get("queued_at", response_started_at) or response_started_at))
                for req in requests
            ]
            legal_index_batches = []
            compact_policy = all(req.get("legal_index_matrix") is not None for req in requests)
            max_legal_cols = 0
            if compact_policy:
                for req, batch_size in zip(requests, batch_sizes):
                    legal_idx = np.asarray(req.get("legal_index_matrix"), dtype=np.int64)
                    if legal_idx.ndim != 2 or int(legal_idx.shape[0]) != batch_size:
                        compact_policy = False
                        legal_index_batches = []
                        break
                    legal_index_batches.append(legal_idx)
                    max_legal_cols = max(max_legal_cols, int(legal_idx.shape[1]))
            if compact_policy and max_legal_cols > 0:
                legal_indices = np.zeros((total_n, max_legal_cols), dtype=np.int64)
                cursor = 0
                for legal_idx, batch_size in zip(legal_index_batches, batch_sizes):
                    legal_indices[cursor:cursor + batch_size, :legal_idx.shape[1]] = legal_idx
                    cursor += batch_size
            else:
                compact_policy = False
                legal_indices = None
            concat_time += time.perf_counter() - concat_t0
            cache = caches.setdefault(model_label, OrderedDict())
            use_cache_for_group = bool(cache_enabled and cache_max_entries > 0 and not compact_policy)

            policy_out = None
            value_out = None
            uncached_indices = None
            uncached_boards = boards
            if use_cache_for_group:
                policy_out = [None] * total_n
                value_out = [None] * total_n
                uncached_indices = []
                uncached_board_list = []
                for idx in range(total_n):
                    key = _cache_key(model_label, boards[idx])
                    if key in cache:
                        policy_np, value_np = cache.pop(key)
                        cache[key] = (policy_np, value_np)
                        policy_out[idx] = policy_np
                        value_out[idx] = value_np
                    else:
                        uncached_indices.append(idx)
                        uncached_board_list.append(boards[idx])
                uncached_boards = np.stack(uncached_board_list, axis=0) if uncached_board_list else None

            policy_np_direct = None
            value_np_direct = None
            if uncached_boards is not None and int(uncached_boards.shape[0]) > 0:
                h2d_t0 = time.perf_counter()
                tensor = torch.from_numpy(uncached_boards).to(
                    device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
                legal_index_tensor = None
                if compact_policy:
                    if use_cache_for_group:
                        uncached_legal_indices = legal_indices[np.asarray(uncached_indices, dtype=np.int64)]
                    else:
                        uncached_legal_indices = legal_indices
                    legal_index_tensor = torch.from_numpy(uncached_legal_indices).to(device, non_blocking=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                h2d_time += time.perf_counter() - h2d_t0
                with torch.inference_mode():
                    forward_t0 = time.perf_counter()
                    if use_amp and device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=amp_dtype):
                            policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    else:
                        policy_logits, value_logits = model(tensor, apply_log_softmax=False)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    forward_time += time.perf_counter() - forward_t0
                    d2h_t0 = time.perf_counter()
                    if compact_policy and legal_index_tensor is not None:
                        policy_logits = torch.gather(policy_logits, 1, legal_index_tensor)
                    policy_np_batch = policy_logits.to(dtype=torch.float16).cpu().numpy().copy()
                    value_np_batch = value_logits.float().cpu().numpy().copy()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    d2h_time += time.perf_counter() - d2h_t0
                if use_cache_for_group:
                    for local_idx, global_idx in enumerate(uncached_indices):
                        policy_np = policy_np_batch[local_idx]
                        value_np = value_np_batch[local_idx]
                        policy_out[global_idx] = policy_np
                        value_out[global_idx] = value_np
                        key = _cache_key(model_label, boards[global_idx])
                        cache[key] = (policy_np, value_np)
                        while len(cache) > cache_max_entries:
                            cache.popitem(last=False)
                else:
                    policy_np_direct = policy_np_batch
                    value_np_direct = value_np_batch

            cursor = 0
            for req_idx, (req, batch_size) in enumerate(zip(requests, batch_sizes)):
                if use_cache_for_group:
                    policy_slice = np.stack(policy_out[cursor:cursor + batch_size], axis=0)
                    value_slice = np.stack(value_out[cursor:cursor + batch_size], axis=0)
                else:
                    policy_slice = policy_np_direct[cursor:cursor + batch_size]
                    value_slice = value_np_direct[cursor:cursor + batch_size]
                cursor += batch_size
                send_t0 = time.perf_counter()
                _put_response(req, {
                    "ok": True,
                    "policy_logits": policy_slice,
                    "value_logits": value_slice,
                    "server_batch_size": total_n,
                    "compact_policy": bool(compact_policy),
                    "server_queue_wait_s": float(queue_waits[req_idx]) if req_idx < len(queue_waits) else 0.0,
                    "server_total_time_s": float(time.perf_counter() - group_t0),
                    "server_concat_time_s": float(concat_time),
                    "server_h2d_time_s": float(h2d_time),
                    "server_forward_time_s": float(forward_time),
                    "server_d2h_time_s": float(d2h_time),
                    "server_send_time_s": float(send_time),
                })
                send_time += time.perf_counter() - send_t0
            return {
                "total_time": time.perf_counter() - group_t0,
                "concat_time": concat_time,
                "h2d_time": h2d_time,
                "forward_time": forward_time,
                "d2h_time": d2h_time,
                "send_time": send_time,
                "compact_policy": bool(compact_policy),
                "positions": int(total_n),
                "requests": int(len(requests)),
            }

        pending = []
        pending_positions = 0
        pending_started_at = None
        first_infer_seen = False
        processed_requests = 0
        processed_positions = 0
        last_debug_print = time.perf_counter()
        last_batch_positions = 0
        last_batch_time_s = 0.0
        last_stage_stats = {}
        interval_infer_time = 0.0
        interval_idle_wait_time = 0.0
        interval_batch_wait_time = 0.0
        interval_batches = 0

        def _request_positions(req):
            boards = req.get("boards")
            shape = getattr(boards, "shape", None)
            if shape:
                return int(shape[0])
            return int(np.asarray(boards).shape[0])

        def _handle_request_item(item):
            nonlocal first_infer_seen, pending_started_at, pending_positions
            cmd = item.get("cmd")
            if cmd == "stop":
                return "stop"
            if cmd == "load_models":
                task_id = item.get("task_id")
                if bool(item.get("clear", False)):
                    models.clear()
                    caches.clear()
                loaded_models = []
                load_t0 = time.perf_counter()
                for entry in list(item.get("models", []) or []):
                    state = entry.get("state")
                    state_path = entry.get("state_path")
                    if state is None and state_path:
                        state = torch.load(state_path, map_location='cpu')
                    if state is not None:
                        loaded_models.append(_load_model(entry.get("label", "learner"), state))
                if loaded_models:
                    def _format_loaded_model(info):
                        label = str(info.get("label", "model"))
                        if info.get("reused"):
                            return f"{label}(reused)"
                        warmup_s = info.get("warmup_s")
                        if warmup_s is not None:
                            return f"{label}(warmup {float(warmup_s):.2f}s)"
                        if info.get("compiled"):
                            return f"{label}(compiled)"
                        return label

                    model_summary = ", ".join(_format_loaded_model(info) for info in loaded_models)
                    if not quiet_startup:
                        print(
                            f"[{_ts()}] Central inference: models ready on {device}: "
                            f"{model_summary} ({time.perf_counter() - load_t0:.2f}s).",
                            flush=True,
                        )
                if control_queue is not None:
                    control_queue.put({
                        "type": "models_loaded",
                        "task_id": task_id,
                        "labels": sorted(models.keys()),
                        "pid": int(os.getpid()),
                        "server_rank": int(compile_rank),
                        "device": str(device),
                        "max_batch": int(max_batch),
                        "flush_ms": float(flush_ms),
                        "transport": str(transport_np_dtype.__name__),
                        "compile": bool(central_use_compile),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device.type == 'cuda' else None,
                        "model_summary": model_summary if loaded_models else "",
                        "load_s": float(time.perf_counter() - load_t0),
                    })
                return "control"
            if cmd == "infer":
                if pending_started_at is None:
                    pending_started_at = time.perf_counter()
                item_positions = _request_positions(item)
                if debug_enabled and not first_infer_seen:
                    first_infer_seen = True
                    print(
                        f"[{_ts()}] Central inference: received first inference request "
                        f"from worker {int(item.get('rank', -1))} "
                        f"({item_positions} positions).",
                        flush=True,
                    )
                pending.append(item)
                pending_positions += item_positions
                return "infer"
            return "ignored"

        while True:
            if pending_started_at is None:
                timeout_s = max(0.001, flush_ms / 1000.0)
            else:
                elapsed_s = time.perf_counter() - pending_started_at
                timeout_s = max(0.0, (flush_ms / 1000.0) - elapsed_s)
            get_t0 = time.perf_counter()
            try:
                item = request_queue.get(timeout=max(0.001, timeout_s))
            except queue.Empty:
                item = None
            get_wait_s = time.perf_counter() - get_t0
            if pending_started_at is None:
                interval_idle_wait_time += get_wait_s
            else:
                interval_batch_wait_time += get_wait_s

            if item is not None:
                item_status = _handle_request_item(item)
                if item_status == "stop":
                    break
                if item_status == "control":
                    continue

            while pending and pending_positions < max_batch:
                try:
                    drained_item = request_queue.get_nowait()
                except queue.Empty:
                    break
                item_status = _handle_request_item(drained_item)
                if item_status == "stop":
                    return
                if item_status == "control":
                    continue

            should_flush = (
                pending
                and (
                    pending_positions >= max_batch
                    or (
                        pending_started_at is not None
                        and (time.perf_counter() - pending_started_at) >= (flush_ms / 1000.0)
                    )
                    or item is None
                )
            )
            if not should_flush:
                continue

            grouped = {}
            for req in pending:
                grouped.setdefault(str(req.get("model_label", "learner")), []).append(req)
            pending = []
            pending_positions = 0
            pending_started_at = None
            grouped_items = sorted(
                grouped.items(),
                key=lambda item_pair: sum(_request_positions(req) for req in item_pair[1]),
                reverse=True,
            )
            for model_label, requests in grouped_items:
                try:
                    infer_t0 = time.perf_counter()
                    stage_stats = _infer_group(model_label, requests)
                    last_batch_time_s = time.perf_counter() - infer_t0
                    last_batch_positions = int(stage_stats.get("positions", 0) or 0)
                    last_stage_stats = dict(stage_stats)
                    interval_infer_time += last_batch_time_s
                    interval_batches += 1
                    processed_requests += len(requests)
                    processed_positions += last_batch_positions
                except Exception as exc:
                    for req in requests:
                        _put_response(req, {
                            "ok": False,
                            "error": str(exc),
                        })
            if debug_enabled and (time.perf_counter() - last_debug_print) >= 5.0:
                interval_s = max(1e-9, time.perf_counter() - last_debug_print)
                last_debug_print = time.perf_counter()
                pos_per_s = (
                    float(last_batch_positions) / max(1e-9, float(last_batch_time_s))
                    if last_batch_positions > 0
                    else 0.0
                )
                active_pct = 100.0 * float(interval_infer_time) / interval_s
                idle_pct = 100.0 * float(interval_idle_wait_time) / interval_s
                batch_wait_pct = 100.0 * float(interval_batch_wait_time) / interval_s
                print(
                    f"[{_ts()}] Central inference: heartbeat "
                    f"requests={processed_requests}, positions={processed_positions}, "
                    f"last_batch={last_batch_positions} pos/{last_batch_time_s:.3f}s "
                    f"({pos_per_s:.1f} pos/s), "
                    f"server_active={active_pct:.1f}%, idle_no_requests={idle_pct:.1f}%, "
                    f"batch_wait={batch_wait_pct:.1f}%, batches={interval_batches}, "
                    f"compact={bool(last_stage_stats.get('compact_policy', False))}, "
                    f"h2d={float(last_stage_stats.get('h2d_time', 0.0)):.3f}s, "
                    f"fwd={float(last_stage_stats.get('forward_time', 0.0)):.3f}s, "
                    f"d2h={float(last_stage_stats.get('d2h_time', 0.0)):.3f}s, "
                    f"send={float(last_stage_stats.get('send_time', 0.0)):.3f}s, "
                    f"last_flush_models={len(grouped)}, pending={len(pending)}.",
                    flush=True,
                )
                interval_infer_time = 0.0
                interval_idle_wait_time = 0.0
                interval_batch_wait_time = 0.0
                interval_batches = 0
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Central inference server failed: {exc}")
        import traceback
        traceback.print_exc()


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
    replay_max_policy_targets = _resolve_replay_max_policy_targets(config)
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
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
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
                    'positions': _pack_positions_for_transfer(
                        positions,
                        max_policy_targets=replay_max_policy_targets,
                    ),
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


def persistent_selfplay_worker(
    rank,
    config,
    device_id,
    task_queue,
    result_queue,
    inference_request_queue=None,
    inference_response_queue=None,
):
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

        central_inference_enabled = inference_request_queue is not None and inference_response_queue is not None
        _, central_debug_cfg, debug_root_enabled = _debug_nested(config, 'rl', 'central_inference')
        central_debug_enabled = bool(
            debug_root_enabled and central_debug_cfg.get(
                'verbose',
                rl_cfg.get('self_play_central_inference_debug', False),
            )
        )
        central_stall_warning_s = float(
            central_debug_cfg.get(
                'stall_warning_s',
                rl_cfg.get('self_play_central_inference_stall_warning_s', 60.0),
            )
        )
        central_transport_dtype = str(
            rl_cfg.get('self_play_central_inference_transport_dtype', 'float16') or 'float16'
        )
        model = None if central_inference_enabled else _build_selfplay_worker_model(config, device)
        inference_model = (
            _RemoteInferenceModel(
                "learner",
                inference_request_queue,
                inference_response_queue,
                worker_rank=rank,
                timeout_s=float(rl_cfg.get('self_play_central_inference_timeout_s', 0.0)),
                stall_warning_s=central_stall_warning_s,
                debug_enabled=central_debug_enabled,
                transport_dtype=central_transport_dtype,
            )
            if central_inference_enabled
            else _maybe_compile_selfplay_model(
                model,
                config,
                device,
                rank,
                model_label="learner",
            )
        )
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
                runtime_overrides = dict(task.get('rl_runtime_overrides') or {})
                if runtime_overrides:
                    config.setdefault('reinforcement_learning', {}).update({
                        key: value
                        for key, value in runtime_overrides.items()
                        if value is not None
                    })
                    rl_cfg = config.get('reinforcement_learning', {})
                if central_inference_enabled and central_debug_enabled:
                    opponent_payload_preview = task.get('opponent_payload') or {}
                    preview_labels = sorted({
                        str((entry or {}).get('label') or 'current')
                        for entry in list(opponent_payload_preview.get('pool_entries', []) or [])
                    })
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Worker {rank}: central task started "
                        f"games={int(task['num_games'])}, opponent_models={preview_labels or ['current']}.",
                        flush=True,
                    )
                if not central_inference_enabled:
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
                    if entry_label == 'current':
                        continue
                    if central_inference_enabled:
                        pooled_model = _RemoteInferenceModel(
                            entry_label,
                            inference_request_queue,
                            inference_response_queue,
                            worker_rank=rank,
                            timeout_s=float(rl_cfg.get('self_play_central_inference_timeout_s', 0.0)),
                            stall_warning_s=central_stall_warning_s,
                            debug_enabled=central_debug_enabled,
                            transport_dtype=central_transport_dtype,
                        )
                    else:
                        if entry_state is None:
                            continue
                        pooled_model = _build_selfplay_worker_model(config, device)
                        _load_worker_model_state(pooled_model, entry_state, rank)
                        pooled_model.eval()
                        pooled_model = _maybe_compile_selfplay_model(
                            pooled_model,
                            config,
                            device,
                            rank,
                            model_label=f"opponent:{entry_label}",
                        )
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
                if task.get('mcts_q_selection_weight') is not None:
                    q_selection_weight = max(0.0, float(task['mcts_q_selection_weight']))
                    opponent_mcts = list(
                        (getattr(engine, 'opponent_mcts_by_label', {}) or {}).values()
                    )
                    for mcts_obj in [getattr(engine, 'mcts', None)] + opponent_mcts:
                        if mcts_obj is not None and hasattr(mcts_obj, 'q_selection_weight'):
                            mcts_obj.q_selection_weight = q_selection_weight
                if task.get('mcts_scout_challenge_fraction') is not None:
                    challenge_fraction = max(0.0, min(1.0, float(task['mcts_scout_challenge_fraction'])))
                    opponent_mcts = list(
                        (getattr(engine, 'opponent_mcts_by_label', {}) or {}).values()
                    )
                    for mcts_obj in [getattr(engine, 'mcts', None)] + opponent_mcts:
                        if mcts_obj is not None and hasattr(mcts_obj, 'scout_challenge_fraction'):
                            mcts_obj.scout_challenge_fraction = challenge_fraction

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
                print(f"ERROR: Worker {rank} failed: {e}")
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
    One-shot worker function for parallel MCTS self-play.
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
        model = _maybe_compile_selfplay_model(
            model,
            config,
            device,
            rank,
            model_label="learner",
        )
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
                pooled_model = _maybe_compile_selfplay_model(
                    pooled_model,
                    config,
                    device,
                    rank,
                    model_label=f"opponent:{entry_label}",
                )
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
        print(f"ERROR: Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        with open(result_file_path, 'wb') as f:
            pickle.dump(([], []), f)


from src.mcts_compat import BatchMCTS, MCTS, select_move_by_visits
