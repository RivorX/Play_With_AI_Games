"""
Elo Estimation via Stockfish matches

Plays fast games against Stockfish at various UCI_Elo levels
and computes the model's estimated Elo from match results.

Uses python-chess's chess.engine module for Stockfish communication.
Auto-downloads Stockfish if not found on the system.
"""

import contextlib
import io
import math
import os
import platform
import subprocess
import shutil
import stat
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

try:
    import urllib.request
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

# Imports are resolved at runtime (script_dir-based sys.path setup in train_il.py)
from src.utils.data_helpers import board_to_tensor, move_to_index
from src.batch_selfplay import MCTS, MultiGameBatchMCTS, select_move_by_visits
from utils.shared.central_inference_session import CentralInferenceSession, snapshot_model_state_cpu


def _build_elo_mcts_config(config: dict, elo_config: dict | None = None) -> dict:
    """Build deterministic MCTS settings for Stockfish Elo checks."""
    eval_config = dict(config)
    elo_config = dict(elo_config or {})
    rl_cfg = dict(config.get("reinforcement_learning", {}))
    elo_early_stop = bool(elo_config.get("elo_mcts_search_early_stop_enabled", False))
    rl_cfg["mcts_search_early_stop_enabled"] = elo_early_stop
    if elo_early_stop:
        early_stop_overrides = {
            "elo_mcts_search_early_stop_min_top_visit_prob": "mcts_search_early_stop_min_top_visit_prob",
            "elo_mcts_search_early_stop_min_visit_gap": "mcts_search_early_stop_min_visit_gap",
            "elo_mcts_search_early_stop_max_visit_entropy": "mcts_search_early_stop_max_visit_entropy",
            "elo_mcts_search_early_stop_min_budget_fraction": "mcts_search_early_stop_min_budget_fraction",
            "elo_mcts_search_early_stop_min_explored_prior_mass": "mcts_search_early_stop_min_explored_prior_mass",
            "elo_mcts_search_early_stop_min_visited_moves": "mcts_search_early_stop_min_visited_moves",
        }
        for elo_key, mcts_key in early_stop_overrides.items():
            if elo_key in elo_config:
                rl_cfg[mcts_key] = elo_config[elo_key]
    rl_cfg["mcts_adaptive_search_enabled"] = False
    eval_config["reinforcement_learning"] = rl_cfg
    return eval_config


# ---------------------------------------------------------------------------
# Stockfish auto-download
# ---------------------------------------------------------------------------

# Default Stockfish version to download
_SF_VERSION = "stockfish-windows-x86-64-avx2"  # Most modern Windows CPUs
_SF_TAG = "sf_17.1"  # GitHub release tag
_SF_REPO = "official-stockfish/Stockfish"


def _available_cpu_count() -> int:
    """Best-effort count of CPUs actually available to this process."""
    try:
        process_cpu_count = getattr(os, "process_cpu_count", None)
        if callable(process_cpu_count):
            value = process_cpu_count()
            if value is not None:
                return max(1, int(value))
    except Exception:
        pass

    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass

    if platform.system().lower() == "windows":
        try:
            import ctypes

            current_process = ctypes.windll.kernel32.GetCurrentProcess()
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            ok = ctypes.windll.kernel32.GetProcessAffinityMask(
                current_process,
                ctypes.byref(process_mask),
                ctypes.byref(system_mask),
            )
            if ok:
                mask_value = int(process_mask.value)
                if mask_value > 0:
                    return max(1, mask_value.bit_count())
        except Exception:
            pass

    return max(1, int(os.cpu_count() or 1))


def _format_worker_game_counts(counts: list[int]) -> str:
    if not counts:
        return "none"

    total = int(sum(counts))
    active_counts = [int(count) for count in counts if int(count) > 0]
    active = len(active_counts)
    if active_counts:
        avg_games = total / max(1, active)
        min_games = min(active_counts)
        max_games = max(active_counts)
    else:
        avg_games = 0.0
        min_games = 0
        max_games = 0

    preview_limit = 32
    preview_parts = [f"w{idx:02d}={int(count)}" for idx, count in enumerate(counts[:preview_limit])]
    if len(counts) > preview_limit:
        preview_parts.append(f"...+{len(counts) - preview_limit} workers")

    return (
        f"total={total}, active={active}/{len(counts)}, "
        f"avg={avg_games:.1f}, min={min_games}, max={max_games}, "
        f"counts=[{', '.join(preview_parts)}]"
    )


def _get_stockfish_download_info() -> tuple[str, str]:
    """
    Return (download_url, expected_binary_name) for the current platform.
    Supports Windows, Linux, macOS (x86_64 + ARM).
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    tag = _SF_TAG
    base = f"https://github.com/{_SF_REPO}/releases/download/{tag}"

    if system == "windows":
        if "64" in machine or machine in ("amd64", "x86_64"):
            name = "stockfish-windows-x86-64-avx2"
        else:
            name = "stockfish-windows-x86-64"  # fallback
        return f"{base}/{name}.zip", f"{name}/stockfish/{name}.exe"

    elif system == "linux":
        if machine in ("x86_64", "amd64"):
            name = "stockfish-linux-x86-64-avx2"
        elif "aarch64" in machine or "arm" in machine:
            name = "stockfish-linux-aarch64"
        else:
            name = "stockfish-linux-x86-64"
        return f"{base}/{name}.tar", f"{name}/stockfish/{name}"

    elif system == "darwin":
        if "arm" in machine or machine == "aarch64":
            name = "stockfish-macos-m1-apple-silicon"
        else:
            name = "stockfish-macos-x86-64-avx2"
        return f"{base}/{name}.tar", f"{name}/stockfish/{name}"

    else:
        raise RuntimeError(f"Unsupported platform: {system} / {machine}")


def _download_stockfish(dest_dir: Path) -> Path:
    """
    Download Stockfish binary to dest_dir and return the path to the executable.
    Uses only stdlib (urllib) â€” no extra dependencies.
    """
    if not _HAS_URLLIB:
        raise RuntimeError("urllib not available â€” cannot auto-download Stockfish.")

    url, inner_path = _get_stockfish_download_info()
    archive_name = url.rsplit("/", 1)[-1]
    is_zip = archive_name.endswith(".zip")

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"  \u2b07\ufe0f  Downloading Stockfish from {url} ...")
    # Download to memory
    req = urllib.request.Request(url, headers={"User-Agent": "ChessAI-EloEstimator/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"  \u2705 Downloaded {len(data) / 1024 / 1024:.1f} MB")

    # Extract
    if is_zip:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest_dir)
    else:
        # .tar file
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(dest_dir)

    binary_path = dest_dir / inner_path
    if not binary_path.exists():
        # Try to locate any stockfish binary in extracted files
        for p in dest_dir.rglob("stockfish*"):
            if p.is_file() and p.stat().st_size > 1_000_000:  # > 1MB = likely binary
                binary_path = p
                break

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Could not locate Stockfish binary after extraction. "
            f"Expected: {dest_dir / inner_path}"
        )

    # Make executable on Unix
    if platform.system().lower() != "windows":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print(f"  \u2705 Stockfish ready: {binary_path}")
    return binary_path


def ensure_stockfish(configured_path: str = "stockfish") -> str:
    """
    Ensure Stockfish is available.  Resolution order:

    1. If configured_path is an absolute path to an existing file â†’ use it.
    2. If configured_path is on PATH (e.g. "stockfish") â†’ use it.
    3. Check project-local cache  (chess/engines/stockfish*).
    4. Auto-download from GitHub releases â†’ cache locally.

    Returns the path (str) to the Stockfish executable.
    """
    # 1. Explicit absolute/relative path
    p = Path(configured_path)
    if p.is_file():
        return str(p)

    # 2. On system PATH?
    found = shutil.which(configured_path)
    if found:
        return found

    # 3. Project-local cache
    # chess/engines/ lives next to chess/scripts/, chess/src/, etc.
    script_dir = Path(__file__).resolve().parent
    chess_dir = script_dir.parents[2]  # utils/shared â†’ scripts/utils/shared â†’ chess/
    engines_dir = chess_dir / "engines"

    # Look for existing cached binary
    if engines_dir.exists():
        for candidate in engines_dir.rglob("stockfish*"):
            if candidate.is_file() and candidate.stat().st_size > 1_000_000:
                print(f"  \u265a Using cached Stockfish: {candidate}")
                return str(candidate)

    # 4. Download
    print("  \u265a Stockfish not found â€” downloading automatically...")
    try:
        binary = _download_stockfish(engines_dir)
        return str(binary)
    except Exception as e:
        print(f"  \u26a0\ufe0f  Failed to auto-download Stockfish: {e}")
        return configured_path  # Return original (will fail later with friendly message)


# ---------------------------------------------------------------------------
# Elo math helpers
# ---------------------------------------------------------------------------

def _expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score of player A against player B (logistic model)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _performance_rating(opponent_elos: list[float], scores: list[float]) -> float | None:
    """
    Compute performance rating via MLE (maximum-likelihood estimate).

    Given a list of opponent Elo values and per-game scores (1/0.5/0),
    find the rating R that maximises the likelihood of the observed results.

    Returns None if the data is degenerate (all wins or all losses at every
    level, making the MLE unbounded).
    """
    total_score = sum(scores)
    n = len(scores)
    if n == 0:
        return None
    score_pct = total_score / n
    # Edge cases: perfect score or zero score â†’ cap at Â±800 from avg opponent
    avg_opp = sum(opponent_elos) / len(opponent_elos)
    if score_pct <= 0.0:
        return avg_opp - 800
    if score_pct >= 1.0:
        return avg_opp + 800

    # Binary search for R that gives the observed score
    lo, hi = avg_opp - 1000, avg_opp + 1000
    for _ in range(64):
        mid = (lo + hi) / 2.0
        expected = sum(_expected_score(mid, opp) for opp in opponent_elos) / n
        if expected < score_pct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _elo_standard_error(opponent_elos: list[float], estimated_elo: float | None) -> float | None:
    """Approximate rating standard error from logistic Fisher information."""
    if estimated_elo is None or not opponent_elos:
        return None
    scale = math.log(10.0) / 400.0
    info = 0.0
    for opp in opponent_elos:
        p = _expected_score(float(estimated_elo), float(opp))
        info += (scale * scale) * max(1e-6, p * (1.0 - p))
    if info <= 0.0:
        return None
    return 1.0 / math.sqrt(info)


# ---------------------------------------------------------------------------
# AI move selection (raw network or MCTS)
# ---------------------------------------------------------------------------

class _ModelPlayer:
    """Thin wrapper that picks moves using raw network or MCTS."""

    def __init__(self, model, config, device, use_mcts: bool = False, simulations: int = 100):
        self.model = model
        self.config = config
        self.device = device
        self.use_mcts = use_mcts
        self.simulations = simulations
        self.history_positions = int(config.get('model', {}).get('history_positions', 0))
        self.use_amp = bool(
            config.get('hardware', {}).get('use_amp', True) and device.type == 'cuda'
        )
        self.amp_dtype = (
            torch.bfloat16
            if config.get('hardware', {}).get('use_bfloat16', False)
            else torch.float16
        )
        # Running board history (chess.Board copies)
        self.board_history: list[chess.Board] = []
        # MCTS instance (lazy init)
        self.mcts = None
        if use_mcts:
            self.mcts = MCTS(model, config, device)

    def reset(self):
        self.board_history = []
        if self.mcts is not None:
            self.mcts.reset_tree()

    def record_state(self, board: chess.Board):
        """Call BEFORE making a move to keep board history."""
        self.board_history.append(board.copy())
        max_keep = self.history_positions + 4
        if len(self.board_history) > max_keep:
            self.board_history = self.board_history[-max_keep:]
        if self.mcts is not None:
            self.mcts.update_history(board)

    def _build_input_tensor(self, board: chess.Board) -> np.ndarray:
        """Build (C, 8, 8) input tensor including history planes."""
        tensors: list[np.ndarray] = []
        flip_history = (board.turn == chess.BLACK)

        if self.history_positions > 0 and self.board_history:
            # Collect up to history_positions past boards (most recent last)
            history = self.board_history[-(self.history_positions):]
            for hb in history:
                tensors.append(board_to_tensor(hb, flip_perspective=flip_history))
            # Pad with zeros if not enough history
            while len(tensors) < self.history_positions:
                tensors.insert(0, np.zeros((16, 8, 8), dtype=np.float32))

        # Current board
        tensors.append(board_to_tensor(board))
        return np.concatenate(tensors, axis=0)

    @torch.inference_mode()
    def best_move(self, board: chess.Board) -> chess.Move | None:
        """Return the best legal move (raw network or MCTS)."""
        if self.use_mcts and self.mcts is not None:
            return self._best_move_mcts(board)
        else:
            return self._best_move_raw(board)
    
    def _best_move_raw(self, board: chess.Board) -> chess.Move | None:
        """Raw network only (fast)."""
        inp = self._build_input_tensor(board)
        t = (
            torch.from_numpy(inp)
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last)
        )
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            policy_logits, _ = self.model(
                t,
                apply_log_softmax=False,
            )

        policy = policy_logits.squeeze(0).float().cpu().numpy()

        best_move = None
        best_score = -float('inf')
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            if idx is not None and 0 <= idx < len(policy):
                if policy[idx] > best_score:
                    best_score = policy[idx]
                    best_move = move
        return best_move
    
    def _best_move_mcts(self, board: chess.Board) -> chess.Move | None:
        """MCTS search (slower but stronger)."""
        visit_counts = self.mcts.search(board, self.simulations, temperature=0.0)
        if not visit_counts:
            return self._best_move_raw(board)  # Fallback
        move, _ = select_move_by_visits(visit_counts, temperature=0.0)
        return move

    def on_move_played(self, move: chess.Move):
        """Keep MCTS tree synchronized with the actual played move."""
        if self.mcts is not None:
            self.mcts.advance_root(move)


class _BatchedModelPlayer:
    """Batch model-side move selection across many independent Elo games."""

    def __init__(self, model, config, device, use_mcts: bool = False, simulations: int = 100):
        self.model = model
        self.config = config
        self.device = device
        self.use_mcts = use_mcts
        self.simulations = simulations
        self.history_positions = int(config.get('model', {}).get('history_positions', 0))
        self.use_amp = bool(
            config.get('hardware', {}).get('use_amp', True) and device.type == 'cuda'
        )
        self.amp_dtype = (
            torch.bfloat16
            if config.get('hardware', {}).get('use_bfloat16', False)
            else torch.float16
        )
        self.multi_mcts = MultiGameBatchMCTS(model, config, device) if use_mcts else None

    def create_state(self) -> dict:
        return {
            "board_history": [],
            "root": None,
            "_root_synced": False,
        }

    def reset_state(self, state: dict):
        state["board_history"] = []
        state["root"] = None
        state["_root_synced"] = False

    def record_state(self, state: dict, board: chess.Board):
        history = state.setdefault("board_history", [])
        history.append(self._encode_history_entry(board))
        max_keep = self.history_positions + 10
        if len(history) > max_keep:
            state["board_history"] = history[-max_keep:]

    @staticmethod
    def _encode_history_entry(board: chess.Board):
        return (
            board_to_tensor(board, flip_perspective=False),
            board_to_tensor(board, flip_perspective=True),
        )

    def _build_input_tensor(self, board: chess.Board, state: dict) -> np.ndarray:
        tensors: list[np.ndarray] = []
        use_black_pov = board.turn == chess.BLACK
        history = state.get("board_history", [])

        if self.history_positions > 0 and history:
            recent_history = history[-self.history_positions:]
            for hist_entry in recent_history:
                hist_tensor = hist_entry[1] if use_black_pov else hist_entry[0]
                tensors.append(hist_tensor)
            while len(tensors) < self.history_positions:
                tensors.insert(0, np.zeros((16, 8, 8), dtype=np.float32))

        tensors.append(board_to_tensor(board))
        return np.concatenate(tensors, axis=0)

    @torch.inference_mode()
    def best_moves(self, boards: list[chess.Board], states: list[dict]) -> list[chess.Move | None]:
        if not boards:
            return []
        if self.use_mcts and self.multi_mcts is not None:
            return self._best_moves_mcts(boards, states)
        return self._best_moves_raw(boards, states)

    def _best_moves_raw(self, boards: list[chess.Board], states: list[dict]) -> list[chess.Move | None]:
        inputs = np.stack(
            [self._build_input_tensor(board, state) for board, state in zip(boards, states)],
            axis=0,
        )
        tensors = torch.from_numpy(inputs).to(
            self.device,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            policy_logits, _ = self.model(
                tensors,
                apply_log_softmax=False,
            )

        policy_batch = policy_logits.float().cpu().numpy()
        moves: list[chess.Move | None] = []
        for row_idx, board in enumerate(boards):
            best_move = None
            best_score = -float('inf')
            policy = policy_batch[row_idx]
            for move in board.legal_moves:
                idx = move_to_index(move, board)
                if idx is not None and 0 <= idx < len(policy) and policy[idx] > best_score:
                    best_score = policy[idx]
                    best_move = move
            moves.append(best_move)
        return moves

    def _best_moves_mcts(self, boards: list[chess.Board], states: list[dict]) -> list[chess.Move | None]:
        game_states = []
        for board, state in zip(boards, states):
            game_states.append(
                {
                    "board": board,
                    "root": state.get("root"),
                    "_root_synced": bool(state.get("_root_synced", False)),
                    "board_history": state.get("board_history", []),
                }
            )

        visit_counts_list = self.multi_mcts.search_many(
            game_states,
            num_simulations=self.simulations,
            add_root_noise=False,
        )

        moves: list[chess.Move | None] = []
        raw_fallback_indices: list[int] = []
        for idx, visit_counts in enumerate(visit_counts_list):
            state = states[idx]
            gs = game_states[idx]
            state["root"] = gs.get("root")
            state["_root_synced"] = bool(gs.get("_root_synced", False))
            if visit_counts:
                move, _ = select_move_by_visits(visit_counts, temperature=0.0)
                moves.append(move)
            else:
                moves.append(None)
                raw_fallback_indices.append(idx)

        if raw_fallback_indices:
            fallback_moves = self._best_moves_raw(
                [boards[idx] for idx in raw_fallback_indices],
                [states[idx] for idx in raw_fallback_indices],
            )
            for idx, move in zip(raw_fallback_indices, fallback_moves):
                moves[idx] = move

        return moves

    def on_move_played(self, state: dict, move: chess.Move):
        if self.multi_mcts is None:
            return
        root = state.get("root")
        if root is None:
            return
        child = root.get_child_for_move(move)
        if child is None:
            state["root"] = None
            state["_root_synced"] = False
            return
        _ = child.board
        state["root"] = child.detach_as_root()
        state["_root_synced"] = True


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class EloEstimator:
    """
    Estimate model Elo by playing rapid games against Stockfish.

    Stockfish must be installed and its path provided (or on PATH).
    Uses UCI_LimitStrength + UCI_Elo for calibrated opponent levels.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        stockfish_path: str = "stockfish",
        stop_event=None,
        elo_config: dict | None = None,
    ):
        self.model = model
        self.elo_config = dict(elo_config or {})
        self.config = _build_elo_mcts_config(config, self.elo_config)
        self.device = device
        self.stockfish_path = stockfish_path
        self.stop_event = stop_event
        self._error_counters: dict[str, int] = {}
        self._thread_local = threading.local()
        self._worker_engines: list[chess.engine.SimpleEngine] = []
        self._worker_engines_lock = threading.Lock()
        self._central_inference_session = None
        self._local_cancel_event = threading.Event()
        self._printed_central_mcts_clients = False
        self._last_elo_batch_stats: dict | None = None

    def _is_cancelled(self) -> bool:
        return self._local_cancel_event.is_set() or (self.stop_event is not None and self.stop_event.is_set())

    def _request_cancel(self):
        self._local_cancel_event.set()
        if self.stop_event is not None:
            with contextlib.suppress(Exception):
                self.stop_event.set()

    def _log_limited(self, key: str, message: str):
        """Log noisy worker errors with suppression after N repeats."""
        limit = int(self.elo_config.get("max_error_logs_per_type", 8) or 8)
        limit = max(1, min(100, limit))

        count = int(self._error_counters.get(key, 0)) + 1
        self._error_counters[key] = count
        if count <= limit:
            print(message)
            return
        if count == (limit + 1):
            print(f"  Warning: {key}: too many repeats, suppressing further logs.")

    def _stockfish_popen_kwargs(self):
        """Build process kwargs for Stockfish engine launches."""
        kwargs = {}
        if platform.system().lower() != "windows":
            return kwargs

        flags = 0
        if bool(self.elo_config.get("stockfish_hide_window", True)):
            flags |= int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))

        priority = str(self.elo_config.get("stockfish_priority", "below_normal")).strip().lower()
        if priority in {"idle", "low"}:
            flags |= int(getattr(subprocess, "IDLE_PRIORITY_CLASS", 0x00000040))
        elif priority in {"below", "below_normal", "background"}:
            flags |= int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000))
        elif priority in {"normal", "default", ""}:
            pass
        else:
            # Unknown value -> keep safe default (below-normal background engine).
            flags |= int(getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000))

        if flags:
            kwargs["creationflags"] = flags
        return kwargs

    def _configure_stockfish_engine(self, engine: chess.engine.SimpleEngine, level: int):
        """Apply UCI options for a Stockfish worker before a game/level batch."""
        options = {"UCI_LimitStrength": True, "UCI_Elo": int(level)}

        threads_raw = self.elo_config.get("stockfish_threads", 1)
        try:
            threads = max(1, int(threads_raw))
        except (TypeError, ValueError):
            threads = 1
        if "Threads" in engine.options:
            options["Threads"] = int(threads)

        hash_raw = self.elo_config.get("stockfish_hash_mb", 64)
        try:
            hash_mb = max(1, int(hash_raw))
        except (TypeError, ValueError):
            hash_mb = 64
        if "Hash" in engine.options:
            options["Hash"] = int(hash_mb)

        engine.configure(options)

    def _resolve_stockfish_threads(self) -> int:
        try:
            return max(1, int(self.elo_config.get("stockfish_threads", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def _resolve_stockfish_hash_mb(self) -> int:
        try:
            return max(1, int(self.elo_config.get("stockfish_hash_mb", 64) or 64))
        except (TypeError, ValueError):
            return 64

    def _resolve_workers(self, requested_workers: int, total_games: int) -> int:
        """
        Resolve worker count with optional training-friendly CPU reservation.
        """
        cpu_total = _available_cpu_count()
        stockfish_threads = self._resolve_stockfish_threads()
        try:
            requested = int(requested_workers or 0)
        except (TypeError, ValueError):
            requested = 0
        if requested <= 0:
            try:
                reserve_auto = int(self.elo_config.get("auto_worker_reserve_cpus", 2) or 0)
            except (TypeError, ValueError):
                reserve_auto = 2
            reserve_auto = max(0, min(cpu_total - 1, reserve_auto))
            available_threads = max(1, cpu_total - reserve_auto)
            requested = max(1, available_threads // stockfish_threads)

        effective = max(1, requested)
        prioritize_training = bool(self.elo_config.get("prioritize_training", False))

        if prioritize_training:
            reserve_loader = 0
            if bool(self.elo_config.get("reserve_dataloader_workers", True)):
                reserve_loader = int(self.config.get("hardware", {}).get("num_workers", 0) or 0)
                reserve_loader = max(0, reserve_loader)

            free_threads = max(1, cpu_total - reserve_loader)
            free_util = float(self.elo_config.get("free_threads_utilization", 1.0) or 1.0)
            free_util = max(0.10, min(1.00, free_util))
            cap_threads = max(1, int(free_threads * free_util))
            cap = max(1, cap_threads // stockfish_threads)

            effective = min(effective, cap)

            if effective < requested:
                print(
                    "  Info: Elo workers capped for training priority: "
                    f"{requested} -> {effective} "
                    f"(cpu={cpu_total}, free={free_threads}, util={free_util:.2f}, "
                    f"reserved_loader={reserve_loader})."
                )

        effective = min(effective, max(1, int(total_games)))
        return max(1, effective)

    def _start_central_inference_for_elo(self, workers: int):
        if self._central_inference_session is not None:
            return self._central_inference_session
        session = CentralInferenceSession(
            config=self.config,
            device=self.device,
            workers=workers,
            model_state=snapshot_model_state_cpu(self.model),
            options=self.elo_config,
            option_prefix="eval_elo",
            model_label="learner",
            rank_base=730000,
        )
        session.start()
        self._central_inference_session = session
        print(f"  Info: Elo central inference: {session.describe()}")
        return session

    def _stop_central_inference_for_elo(self):
        session = self._central_inference_session
        self._central_inference_session = None
        if session is not None:
            session.close()

    def _build_central_remote_model_for_thread(self):
        session = self._central_inference_session
        if not session:
            return None
        return session.remote_model_for_current_thread()

    def _resolve_central_client_groups(self, workers: int, task_count: int) -> tuple[int, list[int]]:
        workers = max(1, int(workers))
        task_count = max(1, int(task_count))
        raw_groups = self.elo_config.get("eval_elo_central_client_groups", "auto")
        if isinstance(raw_groups, str) and raw_groups.strip().lower() == "auto":
            target_groups = max(1, int(self.elo_config.get("eval_elo_central_target_client_groups", 6) or 6))
            min_groups = max(1, int(self.elo_config.get("eval_elo_central_min_client_groups", 4) or 4))
            max_groups = max(min_groups, int(self.elo_config.get("eval_elo_central_max_client_groups", 8) or 8))
            group_count = max(min_groups, min(max_groups, target_groups, workers, task_count))
        else:
            try:
                group_count = max(1, int(raw_groups))
            except (TypeError, ValueError):
                group_count = max(1, min(4, workers, task_count))
        group_count = max(1, min(workers, task_count, group_count))
        base = workers // group_count
        extra = workers % group_count
        group_workers = [base + (1 if idx < extra else 0) for idx in range(group_count)]
        return group_count, group_workers

    def _resolve_central_chunk_target(self) -> int:
        try:
            return max(1, int(self.elo_config.get("eval_elo_central_target_chunks_per_group", 2) or 2))
        except (TypeError, ValueError):
            return 2

    def _resolve_central_max_chunk_games(self) -> int:
        try:
            return max(0, int(self.elo_config.get("eval_elo_central_max_chunk_games", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_central_chunk_size(
        *,
        remaining: int,
        group_count: int,
        group_worker_count: int,
        games_per_worker_chunk: int,
        target_chunks_per_group: int,
        max_chunk_games: int,
    ) -> int:
        remaining = max(0, int(remaining))
        if remaining <= 0:
            return 0
        group_count = max(1, int(group_count))
        group_worker_count = max(1, int(group_worker_count))
        games_per_worker_chunk = max(1, int(games_per_worker_chunk))
        target_chunks_per_group = max(1, int(target_chunks_per_group))

        base_chunk = max(1, group_worker_count * games_per_worker_chunk)
        fair_chunk = max(
            1,
            int(math.ceil(remaining / float(max(1, group_count * target_chunks_per_group)))),
        )
        if remaining >= group_worker_count:
            fair_chunk = max(fair_chunk, group_worker_count)

        chunk_size = min(base_chunk, fair_chunk)
        if max_chunk_games > 0:
            chunk_size = min(chunk_size, int(max_chunk_games))
        return max(1, min(remaining, int(chunk_size)))

    @staticmethod
    def _merge_batched_stats(stats_items: list[dict]) -> dict:
        merged = {
            "model_move_calls": 0,
            "model_move_positions": 0,
            "model_move_time_s": 0.0,
            "stockfish_move_time_s": 0.0,
            "worker_game_counts": [],
        }
        for stats in stats_items:
            merged["model_move_calls"] += int(stats.get("model_move_calls", 0) or 0)
            merged["model_move_positions"] += int(stats.get("model_move_positions", 0) or 0)
            merged["model_move_time_s"] += float(stats.get("model_move_time_s", 0.0) or 0.0)
            merged["stockfish_move_time_s"] += float(stats.get("stockfish_move_time_s", 0.0) or 0.0)
            merged["worker_game_counts"].extend(list(stats.get("worker_game_counts", []) or []))
        return merged

    def _estimate_central_batched_groups(
        self,
        tasks: list[tuple[int, int, bool]],
        workers: int,
        simulations: int,
        stockfish_time_limit: float,
        max_moves: int,
        stockfish_path: str,
        progress_bar,
    ) -> tuple[list[float], list[float]]:
        group_count, group_workers = self._resolve_central_client_groups(workers, len(tasks))
        games_per_worker_chunk = max(
            1,
            int(self.elo_config.get("eval_elo_central_games_per_worker_chunk", 20) or 20),
        )
        target_chunks_per_group = self._resolve_central_chunk_target()
        max_chunk_games = self._resolve_central_max_chunk_games()
        pending_tasks = deque(tasks)
        pending_lock = threading.Lock()

        if not self._printed_central_mcts_clients:
            self._printed_central_mcts_clients = True
            print(
                "  Info: Elo central MCTS clients: "
                f"groups={group_count}, stockfish_workers={workers}, "
                f"active_games/client={group_workers}, "
                f"games/worker_chunk={games_per_worker_chunk}, "
                f"target_chunks/group={target_chunks_per_group}, "
                f"max_chunk={'auto' if max_chunk_games <= 0 else max_chunk_games}"
            )

        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        stats_items: list[dict] = []

        def _next_chunk(group_worker_count: int):
            chunk = []
            with pending_lock:
                remaining = len(pending_tasks)
                if remaining <= 0:
                    return chunk
                chunk_size = self._resolve_central_chunk_size(
                    remaining=remaining,
                    group_count=group_count,
                    group_worker_count=group_worker_count,
                    games_per_worker_chunk=games_per_worker_chunk,
                    target_chunks_per_group=target_chunks_per_group,
                    max_chunk_games=max_chunk_games,
                )
                while pending_tasks and len(chunk) < chunk_size:
                    chunk.append(pending_tasks.popleft())
            return chunk

        def _run_group(group_idx: int):
            remote_model = self._build_central_remote_model_for_thread()
            if remote_model is None:
                raise RuntimeError("Central inference session is not available for Elo MCTS group.")
            group_elos: list[float] = []
            group_scores: list[float] = []
            group_stats_items: list[dict] = []
            while not self._is_cancelled():
                chunk = _next_chunk(group_workers[group_idx])
                if not chunk:
                    break
                chunk_stats = {}
                elos, scores = self._estimate_batched_games(
                    tasks=chunk,
                    workers=min(group_workers[group_idx], len(chunk)),
                    use_mcts=True,
                    simulations=simulations,
                    stockfish_time_limit=stockfish_time_limit,
                    max_moves=max_moves,
                    stockfish_path=stockfish_path,
                    progress_bar=progress_bar,
                    model_override=remote_model,
                    device_override=torch.device("cpu"),
                    stats_out=chunk_stats,
                )
                group_elos.extend(elos)
                group_scores.extend(scores)
                group_stats_items.append(chunk_stats)
            return group_elos, group_scores, self._merge_batched_stats(group_stats_items)

        executor = ThreadPoolExecutor(max_workers=group_count)
        futures = []
        try:
            futures = [executor.submit(_run_group, idx) for idx in range(group_count)]
            for future in as_completed(futures):
                if self._is_cancelled():
                    break
                elos, scores, group_stats = future.result()
                all_opponent_elos.extend(elos)
                all_scores.extend(scores)
                stats_items.append(group_stats)
        except KeyboardInterrupt:
            self._request_cancel()
            for future in futures:
                future.cancel()
            self._close_worker_engines()
            raise
        finally:
            cancelled = self._is_cancelled()
            if cancelled:
                for future in futures:
                    future.cancel()
                self._close_worker_engines()
                self._stop_central_inference_for_elo()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=False)

        merged_stats = self._merge_batched_stats(stats_items)
        calls = int(merged_stats.get("model_move_calls", 0) or 0)
        positions = int(merged_stats.get("model_move_positions", 0) or 0)
        if calls > 0:
            avg_batch = float(positions) / float(calls)
            print(
                "  Info: Elo model batching: "
                f"avg_batch={avg_batch:.1f}, calls={calls}, "
                f"model_time={float(merged_stats.get('model_move_time_s', 0.0)):.1f}s, "
                f"stockfish_wait={float(merged_stats.get('stockfish_move_time_s', 0.0)):.1f}s"
            )
        worker_counts = list(merged_stats.get("worker_game_counts", []) or [])
        self._last_elo_batch_stats = dict(merged_stats)
        if sum(worker_counts) > 0 and bool(self.elo_config.get("elo_verbose_worker_stats", False)):
            print(f"  Info: Elo games/worker: {_format_worker_game_counts(worker_counts)}")
        return all_opponent_elos, all_scores

    def _register_worker_engine(self, engine: chess.engine.SimpleEngine):
        with self._worker_engines_lock:
            self._worker_engines.append(engine)

    def _force_close_engine(self, engine: chess.engine.SimpleEngine | None):
        if engine is None:
            return

        protocol = None
        transport = None
        proc = None

        with contextlib.suppress(Exception):
            protocol = getattr(engine, "protocol", None)
        if protocol is not None:
            with contextlib.suppress(Exception):
                transport = getattr(protocol, "transport", None)
        if transport is None:
            with contextlib.suppress(Exception):
                transport = getattr(engine, "transport", None)
        if transport is not None:
            with contextlib.suppress(Exception):
                proc = getattr(transport, "_proc", None)
            if proc is None:
                with contextlib.suppress(Exception):
                    proc = getattr(transport, "proc", None)

        with contextlib.suppress(Exception):
            engine.quit()
        with contextlib.suppress(Exception):
            close_fn = getattr(engine, "close", None)
            if callable(close_fn):
                close_fn()
        if protocol is not None:
            with contextlib.suppress(Exception):
                protocol_close = getattr(protocol, "close", None)
                if callable(protocol_close):
                    protocol_close()
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.close()
            with contextlib.suppress(Exception):
                transport.abort()
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=0.2)
            with contextlib.suppress(Exception):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=0.2)

    def _close_worker_engines(self):
        with self._worker_engines_lock:
            engines = self._worker_engines
            self._worker_engines = []
        worker = getattr(self._thread_local, "worker_resources", None)
        if worker is not None:
            with contextlib.suppress(Exception):
                self._thread_local.worker_resources = None
            engine = worker.get("engine")
            if engine is not None:
                engines = list(engines) + [engine]
        seen = set()
        for engine in engines:
            engine_id = id(engine)
            if engine_id in seen:
                continue
            seen.add(engine_id)
            self._force_close_engine(engine)

    def _open_stockfish_engine(self, stockfish_path: str) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(
            stockfish_path,
            **self._stockfish_popen_kwargs(),
        )
        self._register_worker_engine(engine)
        return engine

    def _get_thread_worker_resources(self, use_mcts: bool, simulations: int, stockfish_path: str):
        """Get or create persistent worker-local Stockfish engine and model player."""
        worker = getattr(self._thread_local, "worker_resources", None)
        if worker is not None:
            same_cfg = (
                bool(worker.get("use_mcts", False)) == bool(use_mcts)
                and int(worker.get("simulations", 0)) == int(simulations)
                and str(worker.get("stockfish_path", "")) == str(stockfish_path)
            )
            if same_cfg:
                return worker["engine"], worker["player"]
            self._force_close_engine(worker.get("engine"))

        engine = self._open_stockfish_engine(stockfish_path)
        self._configure_stockfish_engine(engine, level=1320)
        central_model = self._build_central_remote_model_for_thread() if use_mcts else None
        if central_model is not None:
            player = _ModelPlayer(central_model, self.config, torch.device("cpu"), use_mcts, simulations)
        else:
            player = _ModelPlayer(self.model, self.config, self.device, use_mcts, simulations)

        worker = {
            "engine": engine,
            "player": player,
            "use_mcts": bool(use_mcts),
            "simulations": int(simulations),
            "stockfish_path": str(stockfish_path),
        }
        self._thread_local.worker_resources = worker
        return engine, player

    @staticmethod
    def _score_game_result(board: chess.Board, model_is_white: bool) -> float:
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if model_is_white else 0.0
        if result == "0-1":
            return 0.0 if model_is_white else 1.0
        return 0.5

    @staticmethod
    def _build_level_tasks(level: int, start_game_idx: int, count: int) -> list[tuple[int, int, bool]]:
        tasks: list[tuple[int, int, bool]] = []
        for offset in range(max(0, int(count))):
            game_idx = int(start_game_idx) + offset
            tasks.append((int(level), game_idx, (game_idx % 2 == 0)))
        return tasks

    @staticmethod
    def _summarize_scores(scores: list[float]) -> dict:
        wins = sum(1 for s in scores if s == 1.0)
        draws = sum(1 for s in scores if s == 0.5)
        losses = sum(1 for s in scores if s == 0.0)
        total = wins + draws + losses
        score_pct = (wins + 0.5 * draws) / total if total else 0.0
        return {"wins": wins, "draws": draws, "losses": losses, "score": score_pct, "total": total}

    @staticmethod
    def _extend_progress_total(progress_bar, count: int, *, cap: int | None = None, phase: str | None = None):
        if progress_bar is None or count <= 0:
            return
        progress_bar.total = int(progress_bar.total or 0) + int(count)
        postfix = {}
        if cap is not None and cap > 0:
            postfix["cap"] = int(cap)
        if phase:
            postfix["phase"] = str(phase)
        if postfix:
            progress_bar.set_postfix(postfix, refresh=False)
        progress_bar.refresh()

    @staticmethod
    def _interleaved_ladder_order(levels: list[int]) -> list[int]:
        """Probe low/high/mid levels in one wave so adaptive Elo still fills workers."""
        values = [int(level) for level in levels]
        n = len(values)
        if n <= 2:
            return values
        mid = n // 2
        indices: list[int] = []
        seen: set[int] = set()
        for offset in range(n):
            candidates = [offset, n - 1 - offset, mid + offset, mid - offset]
            for idx in candidates:
                if 0 <= idx < n and idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
            if len(indices) >= n:
                break
        return [values[idx] for idx in indices]

    def _run_task_batch(
        self,
        *,
        tasks: list[tuple[int, int, bool]],
        workers: int,
        use_mcts: bool,
        simulations: int,
        stockfish_time_limit: float,
        max_moves: int,
        resolved_path: str,
        batch_model_moves: bool,
        central_inference_enabled: bool,
        progress_bar,
    ) -> tuple[list[float], list[float], bool]:
        """Run an already chosen list of Elo games through the shared executor."""
        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        interrupted_by_user = False

        if not tasks:
            return all_opponent_elos, all_scores, interrupted_by_user

        if central_inference_enabled:
            try:
                all_opponent_elos, all_scores = self._estimate_central_batched_groups(
                    tasks=tasks,
                    workers=workers,
                    simulations=simulations,
                    stockfish_time_limit=stockfish_time_limit,
                    max_moves=max_moves,
                    stockfish_path=resolved_path,
                    progress_bar=progress_bar,
                )
            except KeyboardInterrupt:
                interrupted_by_user = True
                self._request_cancel()
                print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
            finally:
                self._close_worker_engines()
        elif batch_model_moves and workers > 1:
            try:
                all_opponent_elos, all_scores = self._estimate_batched_games(
                    tasks=tasks,
                    workers=workers,
                    use_mcts=use_mcts,
                    simulations=simulations,
                    stockfish_time_limit=stockfish_time_limit,
                    max_moves=max_moves,
                    stockfish_path=resolved_path,
                    progress_bar=progress_bar,
                )
            except KeyboardInterrupt:
                interrupted_by_user = True
                self._request_cancel()
                print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
            finally:
                self._close_worker_engines()
        elif workers > 1:
            parallel_thread_slots: dict[int, int] = {}
            parallel_game_counts: list[int] = []

            def _record_parallel_thread_game(thread_id: int):
                slot = parallel_thread_slots.get(thread_id)
                if slot is None:
                    slot = len(parallel_game_counts)
                    parallel_thread_slots[thread_id] = slot
                    parallel_game_counts.append(0)
                parallel_game_counts[slot] += 1

            def _run_parallel_game(
                level: int,
                model_is_white: bool,
                use_mcts: bool,
                simulations: int,
                stockfish_time_limit: float,
                max_moves: int,
                resolved_path: str,
            ):
                result = self._play_single_game_worker(
                    level,
                    model_is_white,
                    use_mcts,
                    simulations,
                    stockfish_time_limit,
                    max_moves,
                    resolved_path,
                )
                return result, threading.get_ident()

            executor = ThreadPoolExecutor(max_workers=workers)
            futures = {}
            try:
                for level, game_idx, model_is_white in tasks:
                    if self._is_cancelled():
                        break
                    future = executor.submit(
                        _run_parallel_game,
                        level,
                        model_is_white,
                        use_mcts,
                        simulations,
                        stockfish_time_limit,
                        max_moves,
                        resolved_path,
                    )
                    futures[future] = (level, game_idx)

                try:
                    for future in as_completed(futures):
                        if self._is_cancelled():
                            break
                        level, _ = futures[future]
                        try:
                            result, thread_id = future.result()
                            _record_parallel_thread_game(int(thread_id))
                            if result is None:
                                if progress_bar is not None:
                                    progress_bar.update(1)
                                continue
                            all_opponent_elos.append(float(level))
                            all_scores.append(float(result))
                        except Exception as exc:
                            self._log_limited("parallel_game_fail", f"  Warning: game at level {level} failed: {exc}")
                            all_opponent_elos.append(float(level))
                            all_scores.append(0.0)
                        finally:
                            if progress_bar is not None:
                                progress_bar.update(1)
                except KeyboardInterrupt:
                    interrupted_by_user = True
                    self._request_cancel()
                    print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
            finally:
                cancelled = self._is_cancelled() or interrupted_by_user
                if cancelled:
                    for future in futures:
                        future.cancel()
                    self._close_worker_engines()
                    executor.shutdown(wait=False, cancel_futures=True)
                    self._close_worker_engines()
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
                self._close_worker_engines()
                self._last_elo_batch_stats = {"worker_game_counts": list(parallel_game_counts)}
                if sum(parallel_game_counts) > 0 and bool(self.elo_config.get("elo_verbose_worker_stats", False)):
                    print(f"  Info: Elo games/worker: {_format_worker_game_counts(parallel_game_counts)}")
        else:
            player = _ModelPlayer(self.model, self.config, self.device, use_mcts, simulations)
            engine = chess.engine.SimpleEngine.popen_uci(
                resolved_path,
                **self._stockfish_popen_kwargs(),
            )
            try:
                try:
                    last_level = None
                    for level, game_idx, model_is_white in tasks:
                        if self._is_cancelled():
                            break
                        if level != last_level:
                            self._configure_stockfish_engine(engine, level)
                            last_level = level
                        result = self._play_game(
                            engine,
                            player,
                            model_is_white,
                            stockfish_time_limit,
                            max_moves,
                        )
                        if result is not None:
                            all_opponent_elos.append(float(level))
                            all_scores.append(float(result))
                        if progress_bar is not None:
                            progress_bar.update(1)
                except KeyboardInterrupt:
                    interrupted_by_user = True
                    self._request_cancel()
                    print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
            finally:
                self._force_close_engine(engine)

        return all_opponent_elos, all_scores, interrupted_by_user

    def _print_elo_completion_summary(self, result: dict, *, use_mcts: bool):
        if not bool(self.elo_config.get("elo_print_completion_summary", True)):
            return
        total_games = int(result.get("total_games", 0) or 0)
        elapsed = float(result.get("total_time", 0.0) or 0.0)
        rate = (float(total_games) / elapsed) if elapsed > 0.0 else 0.0
        print(
            f"  Elo done: {total_games} games in {elapsed:.1f}s "
            f"({rate:.2f} games/s, adaptive)"
        )
        stats = self._last_elo_batch_stats or {}
        calls = int(stats.get("model_move_calls", 0) or 0)
        positions = int(stats.get("model_move_positions", 0) or 0)
        if use_mcts and calls > 0:
            avg_batch = float(positions) / float(calls)
            print(
                "  Elo model batching: "
                f"avg_batch={avg_batch:.1f}, calls={calls}, "
                f"model_time={float(stats.get('model_move_time_s', 0.0) or 0.0):.1f}s, "
                f"stockfish_wait={float(stats.get('stockfish_move_time_s', 0.0) or 0.0):.1f}s"
            )
        worker_counts = list(stats.get("worker_game_counts", []) or [])
        if worker_counts and bool(self.elo_config.get("elo_print_worker_summary", False)):
            print(f"  Elo workers: {_format_worker_game_counts(worker_counts)}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def estimate(
        self,
        levels: list[int] | None = None,
        games_per_level: int = 4,
        stockfish_time_limit: float = 0.05,
        max_moves: int = 150,
        use_mcts: bool = False,
        simulations: int = 100,
        workers: int = 0,
    ) -> dict:
        """
        Play games against Stockfish at several Elo levels and return
        a performance-rating-based estimate of the model's Elo.
        """
        if levels is None:
            levels = [1000, 1300, 1600, 1900, 2200]

        self._local_cancel_event.clear()
        self._printed_central_mcts_clients = False
        self.model.eval()
        if self._is_cancelled():
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": 0,
                "total_time": 0.0,
                "cancelled": True,
            }

        # Try to open Stockfish (auto-download if needed) - just for validation.
        resolved_path = ensure_stockfish(self.stockfish_path)
        try:
            test_engine = chess.engine.SimpleEngine.popen_uci(
                resolved_path,
                **self._stockfish_popen_kwargs(),
            )
            sf_min_elo = 1320
            sf_max_elo = 3190
            if "UCI_Elo" in test_engine.options:
                opt = test_engine.options["UCI_Elo"]
                if hasattr(opt, "min") and opt.min is not None:
                    sf_min_elo = int(opt.min)
                if hasattr(opt, "max") and opt.max is not None:
                    sf_max_elo = int(opt.max)
            self._force_close_engine(test_engine)
        except FileNotFoundError:
            print(f"  Warning: Stockfish not found at '{resolved_path}' - Elo estimation skipped.")
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": 0,
                "total_time": 0.0,
                "error": "stockfish_not_found",
            }
        except Exception as exc:
            print(f"  Warning: Stockfish error: {exc} - Elo estimation skipped.")
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": 0,
                "total_time": 0.0,
                "error": str(exc),
            }

        # Filter/clamp levels to engine's supported range.
        valid_levels = []
        for lvl in levels:
            clamped = max(sf_min_elo, min(sf_max_elo, lvl))
            if clamped != lvl:
                print(f"  Warning: Clamped level {lvl} -> {clamped} (SF range: {sf_min_elo}-{sf_max_elo})")
            if clamped not in valid_levels:
                valid_levels.append(clamped)
        levels = valid_levels

        if not levels:
            print("  Warning: No valid Elo levels after clamping.")
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": 0,
                "total_time": 0.0,
                "error": "no_valid_levels",
            }

        t0 = time.perf_counter()
        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        results_per_level: dict[int, dict] = {}

        requested_total_games = int(len(levels) * max(1, int(games_per_level)))
        max_total_games = requested_total_games
        try:
            max_total_games = int(self.elo_config.get("adaptive_max_total_games", requested_total_games) or requested_total_games)
        except (TypeError, ValueError):
            max_total_games = requested_total_games
        max_total_games = max(1, min(requested_total_games, max_total_games))

        workers = self._resolve_workers(workers, total_games=max_total_games)
        batch_model_moves = bool(self.elo_config.get("batch_model_moves", True))
        if not use_mcts:
            # Raw NN inference is usually much cheaper than the Stockfish move.
            # Running full games per worker avoids the per-ply batch barrier.
            batch_model_moves = bool(self.elo_config.get("batch_raw_model_moves", batch_model_moves))
        central_inference_enabled = bool(
            use_mcts
            and workers > 1
            and self.device.type == "cuda"
            and self.elo_config.get("eval_elo_central_inference_enabled", False)
        )
        if central_inference_enabled:
            # Central inference batches across independent MCTS workers. Keeping
            # the single-process batched MCTS path would serialize search again.
            batch_model_moves = False
        print(f"  Info: Elo workers resolved to {workers} (available_cpu={_available_cpu_count()}).")
        print(
            "  Info: Stockfish config: "
            f"threads={self._resolve_stockfish_threads()}, "
            f"hash={self._resolve_stockfish_hash_mb()} MB, "
            f"time_limit={float(stockfish_time_limit):.3f}s, "
            f"games=adaptive <= {max_total_games}"
        )
        adaptive_focus_cap = min(
            max(1, int(games_per_level)),
            max(
                1,
                int(self.elo_config.get("adaptive_focus_games_per_level", games_per_level) or games_per_level),
            ),
        )
        print(
            "  Info: Adaptive Elo ladder enabled: "
            f"levels={levels}, max_games={max_total_games}, "
            f"probe={int(self.elo_config.get('adaptive_probe_games_per_level', 8) or 8)}, "
            f"focus<= {adaptive_focus_cap}/level"
        )
        if batch_model_moves and workers > 1:
            model_path = "batched_mcts" if use_mcts else "batched_raw"
            print(f"  Info: Elo model path: {model_path} (shared batch scheduling enabled).")
        elif workers > 1:
            if central_inference_enabled:
                model_path = "batched_mcts_central"
            else:
                model_path = "parallel_mcts" if use_mcts else "parallel_raw"
            print(f"  Info: Elo model path: {model_path} (independent game workers).")
        if self._is_cancelled():
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": 0,
                "total_time": 0.0,
                "cancelled": True,
            }

        progress_mode = str(self.elo_config.get("progress_bar", "sync_only")).strip().lower()
        if progress_mode in {"off", "false", "0", "none", "disabled"}:
            show_progress = False
        elif progress_mode in {"on", "true", "1", "always"}:
            show_progress = True
        else:
            # Default: show only in blocking/sequential contexts.
            show_progress = self.stop_event is None

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=0,
                desc="Elo adaptive",
                unit="game",
                dynamic_ncols=True,
                leave=True,
                smoothing=0.05,
                mininterval=0.5,
                delay=0.5,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar:34}| "
                    "{n_fmt}/{total_fmt} {unit} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )

        interrupted_by_user = False
        played_by_level: dict[int, int] = {int(level): 0 for level in levels}
        scores_by_level: dict[int, list[float]] = {int(level): [] for level in levels}

        def _run_selected_tasks(tasks: list[tuple[int, int, bool]], phase: str = "") -> bool:
            nonlocal interrupted_by_user
            if not tasks:
                return False
            self._extend_progress_total(progress_bar, len(tasks), cap=max_total_games, phase=phase)
            batch_elos, batch_scores, batch_interrupted = self._run_task_batch(
                tasks=tasks,
                workers=min(workers, max(1, len(tasks))),
                use_mcts=use_mcts,
                simulations=simulations,
                stockfish_time_limit=stockfish_time_limit,
                max_moves=max_moves,
                resolved_path=resolved_path,
                batch_model_moves=batch_model_moves,
                central_inference_enabled=central_inference_enabled,
                progress_bar=progress_bar,
            )
            interrupted_by_user = interrupted_by_user or batch_interrupted
            for level_f, score in zip(batch_elos, batch_scores):
                level = int(level_f)
                all_opponent_elos.append(float(level_f))
                all_scores.append(float(score))
                played_by_level[level] = int(played_by_level.get(level, 0)) + 1
                scores_by_level.setdefault(level, []).append(float(score))
            return batch_interrupted or self._is_cancelled()

        try:
            if central_inference_enabled:
                self._start_central_inference_for_elo(workers)
            probe_games = max(2, int(self.elo_config.get("adaptive_probe_games_per_level", 8) or 8))
            probe_games = min(max(1, int(games_per_level)), probe_games)
            focus_games = max(probe_games, int(self.elo_config.get("adaptive_focus_games_per_level", games_per_level) or games_per_level))
            focus_games = min(max(1, int(games_per_level)), focus_games)
            extra_round = max(2, int(self.elo_config.get("adaptive_extra_games_per_level", 8) or 8))
            high_skip = float(self.elo_config.get("adaptive_skip_high_score", 0.92) or 0.92)
            low_stop = float(self.elo_config.get("adaptive_stop_low_score", 0.08) or 0.08)
            focus_min = float(self.elo_config.get("adaptive_focus_min_score", 0.20) or 0.20)
            focus_max = float(self.elo_config.get("adaptive_focus_max_score", 0.80) or 0.80)
            target_focus = max(1, int(self.elo_config.get("adaptive_target_focus_levels", 4) or 4))
            focus_min_games = min(focus_games, max(probe_games * 2, 16))
            target_se = float(self.elo_config.get("adaptive_target_standard_error", 0.0) or 0.0)
            min_games_for_se_stop = max(
                probe_games * target_focus,
                int(self.elo_config.get("adaptive_min_games_for_se_stop", 0) or 0),
            )
            min_batch_games_raw = self.elo_config.get("adaptive_min_batch_games", 0)
            try:
                min_batch_games = int(min_batch_games_raw or 0)
            except (TypeError, ValueError):
                min_batch_games = 0
            if min_batch_games <= 0:
                min_batch_games = max(int(workers) * 2, probe_games * 3)
            min_batch_games = max(probe_games, min(max_total_games, min_batch_games))
            probe_levels_per_wave = max(
                1,
                min(len(levels), int(math.ceil(float(min_batch_games) / float(max(1, probe_games))))),
            )
            total_scheduled = 0

            probe_order = self._interleaved_ladder_order(levels)

            def _score_for_level(level: int) -> float:
                return float(self._summarize_scores(scores_by_level.get(int(level), []))["score"])

            def _played_candidate_levels() -> list[int]:
                return [
                    int(level)
                    for level in levels
                    if played_by_level.get(int(level), 0) > 0
                ]

            def _useful_candidate_count() -> int:
                count = 0
                for level in _played_candidate_levels():
                    score = _score_for_level(level)
                    if focus_min <= score <= focus_max:
                        count += 1
                return count

            def _estimate_precise_enough() -> bool:
                if target_se <= 0.0 or len(all_scores) < min_games_for_se_stop:
                    return False
                estimated = _performance_rating(all_opponent_elos, all_scores)
                se = _elo_standard_error(all_opponent_elos, estimated)
                return bool(se is not None and float(se) <= target_se)

            if bool(self.elo_config.get("elo_verbose_adaptive", False)):
                print(
                    "  Adaptive probe waves: "
                    f"levels/wave={probe_levels_per_wave}, min_batch_games={min_batch_games}, "
                    f"order={probe_order[:min(len(probe_order), probe_levels_per_wave * 2)]}"
                    f"{'...' if len(probe_order) > probe_levels_per_wave * 2 else ''}"
                )
            stop_after_wave = False
            for wave_start in range(0, len(probe_order), probe_levels_per_wave):
                if self._is_cancelled() or total_scheduled >= max_total_games or stop_after_wave:
                    break
                wave_levels = probe_order[wave_start: wave_start + probe_levels_per_wave]
                tasks = []
                for level in wave_levels:
                    games_to_add = min(probe_games, max_total_games - total_scheduled - len(tasks))
                    if games_to_add <= 0:
                        break
                    tasks.extend(self._build_level_tasks(level, played_by_level.get(level, 0), games_to_add))
                total_scheduled += len(tasks)
                if _run_selected_tasks(tasks, phase="probe"):
                    break
                for level in wave_levels:
                    summary = self._summarize_scores(scores_by_level.get(int(level), []))
                    if summary["total"] <= 0:
                        continue
                    if bool(self.elo_config.get("elo_verbose_adaptive", False)):
                        print(
                            f"  Adaptive probe vs SF {level}: "
                            f"W{summary['wins']}/D{summary['draws']}/L{summary['losses']} "
                            f"({summary['score']:.0%})"
                        )
                    if summary["score"] >= high_skip and bool(self.elo_config.get("elo_verbose_adaptive", False)):
                        print(f"  Adaptive ladder: SF {level} is saturated (score >= {high_skip:.0%}); probing higher.")

                # Stop probing above a clearly too-strong level only after we
                # already have enough non-saturated candidates. Otherwise the
                # first interleaved wave can lock focus onto useless extremes
                # such as 1320/2800 and spend most of the budget there.
                for level in sorted(wave_levels):
                    summary = self._summarize_scores(scores_by_level.get(int(level), []))
                    if summary["total"] > 0 and summary["score"] <= low_stop:
                        if _useful_candidate_count() >= target_focus:
                            if bool(self.elo_config.get("elo_verbose_adaptive", False)):
                                print(
                                    f"  Adaptive ladder: stopping above {level} after this wave "
                                    f"(score <= {low_stop:.0%})."
                                )
                            stop_after_wave = True
                        break

            candidate_levels = _played_candidate_levels()
            focus_levels = [
                level
                for level in candidate_levels
                if focus_min <= _score_for_level(level) <= focus_max
            ]
            focus_levels = sorted(
                focus_levels,
                key=lambda lvl: abs(_score_for_level(lvl) - 0.5),
            )
            if len(focus_levels) < target_focus:
                extras = sorted(
                    (
                        level
                        for level in candidate_levels
                        if level not in focus_levels and low_stop < _score_for_level(level) < high_skip
                    ),
                    key=lambda lvl: (
                        abs(_score_for_level(lvl) - 0.5),
                        -lvl if _score_for_level(lvl) >= 0.5 else lvl,
                    ),
                )
                focus_levels.extend(extras[: max(0, target_focus - len(focus_levels))])
            if not focus_levels and candidate_levels:
                focus_levels = sorted(
                    candidate_levels,
                    key=lambda lvl: abs(_score_for_level(lvl) - 0.5),
                )[:1]
            focus_levels = focus_levels[:target_focus]

            if focus_levels and bool(self.elo_config.get("elo_verbose_adaptive", False)):
                print(f"  Adaptive focus levels: {focus_levels}")
            while (
                not self._is_cancelled()
                and total_scheduled < max_total_games
                and any(played_by_level.get(level, 0) < focus_games for level in focus_levels)
            ):
                tasks = []
                for level in focus_levels:
                    if total_scheduled + len(tasks) >= max_total_games:
                        break
                    if played_by_level.get(level, 0) >= focus_games:
                        continue
                    if played_by_level.get(level, 0) >= focus_min_games:
                        score = _score_for_level(level)
                        if score <= low_stop or score >= high_skip:
                            continue
                    remaining_level = focus_games - int(played_by_level.get(level, 0))
                    games_to_add = min(extra_round, remaining_level, max_total_games - total_scheduled - len(tasks))
                    if games_to_add <= 0:
                        continue
                    tasks.extend(self._build_level_tasks(level, played_by_level.get(level, 0), games_to_add))
                if not tasks:
                    break
                total_scheduled += len(tasks)
                if _run_selected_tasks(tasks, phase="focus"):
                    break
                if _estimate_precise_enough():
                    break
                if len(tasks) < min_batch_games:
                    # Final partial focus wave; no need to spin another tiny wave
                    # unless some level still has meaningful capacity and budget.
                    if not any(played_by_level.get(level, 0) < focus_games for level in focus_levels):
                        break
            for level in focus_levels:
                if bool(self.elo_config.get("elo_verbose_adaptive", False)):
                    summary = self._summarize_scores(scores_by_level.get(level, []))
                    print(
                        f"  Adaptive focus vs SF {level}: "
                        f"W{summary['wins']}/D{summary['draws']}/L{summary['losses']} "
                        f"({summary['score']:.0%}, games={summary['total']})"
                    )
        finally:
            self._stop_central_inference_for_elo()
            if progress_bar is not None:
                progress_bar.total = progress_bar.n
                progress_bar.refresh()
                progress_bar.close()

        if interrupted_by_user:
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": len(all_scores),
                "total_time": time.perf_counter() - t0,
                "cancelled": True,
            }

        if self._is_cancelled():
            return {
                "estimated_elo": None,
                "results": {},
                "total_games": len(all_scores),
                "total_time": time.perf_counter() - t0,
                "cancelled": True,
            }

        for level in levels:
            level_scores = [score for elo, score in zip(all_opponent_elos, all_scores) if elo == level]
            wins = sum(1 for s in level_scores if s == 1.0)
            draws = sum(1 for s in level_scores if s == 0.5)
            losses = sum(1 for s in level_scores if s == 0.0)
            total = wins + draws + losses
            score_pct = (wins + 0.5 * draws) / total if total else 0.0
            if total <= 0:
                continue
            results_per_level[level] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": score_pct,
                "games": total,
            }

        elapsed = time.perf_counter() - t0
        estimated_elo = _performance_rating(all_opponent_elos, all_scores)
        elo_se = _elo_standard_error(all_opponent_elos, estimated_elo)
        elo_ci95 = None
        if estimated_elo is not None and elo_se is not None:
            elo_ci95 = [round(estimated_elo - 1.96 * elo_se), round(estimated_elo + 1.96 * elo_se)]

        result = {
            "estimated_elo": round(estimated_elo) if estimated_elo is not None else None,
            "results": results_per_level,
            "total_games": len(all_scores),
            "total_time": elapsed,
            "adaptive": True,
            "levels_requested": [int(x) for x in levels],
            "games_per_level_requested": int(games_per_level),
            "actual_games_per_level": {int(k): int(v) for k, v in played_by_level.items() if int(v) > 0},
        }
        if elo_se is not None:
            result["elo_std_error"] = round(float(elo_se), 1)
        if elo_ci95 is not None:
            result["elo_ci95"] = elo_ci95
        self._print_elo_completion_summary(result, use_mcts=use_mcts)
        return result
    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _play_single_game_worker(
        self,
        level: int,
        model_is_white: bool,
        use_mcts: bool,
        simulations: int,
        sf_time_limit: float,
        max_moves: int,
        stockfish_path: str,
    ) -> float | None:
        """
        Worker function for parallel game execution.
        Creates own Stockfish engine and plays one game.
        
        Returns: score (1.0=win, 0.5=draw, 0.0=loss) from model's perspective.
        """
        if self._is_cancelled():
            return None

        # Each worker needs its own engine and player
        try:
            engine, player = self._get_thread_worker_resources(
                use_mcts,
                simulations,
                stockfish_path,
            )
            self._configure_stockfish_engine(engine, level)
        except Exception as exc:
            if self._is_cancelled():
                return None
            self._log_limited("stockfish_open_fail", f"  Warning: worker failed to open Stockfish: {exc}")
            return 0.0  # Count as loss

        try:
            result = self._play_game(engine, player, model_is_white, sf_time_limit, max_moves)
        except Exception as exc:
            if self._is_cancelled():
                result = None
            else:
                self._log_limited("stockfish_game_fail", f"  Warning: game failed: {exc}")
                result = 0.0

        return result

    def _estimate_batched_games(
        self,
        tasks: list[tuple[int, int, bool]],
        workers: int,
        use_mcts: bool,
        simulations: int,
        stockfish_time_limit: float,
        max_moves: int,
        stockfish_path: str,
        progress_bar,
        model_override=None,
        device_override=None,
        stats_out: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        max_half_moves = max(1, int(max_moves)) * 2

        batched_player = _BatchedModelPlayer(
            model_override if model_override is not None else self.model,
            self.config,
            device_override if device_override is not None else self.device,
            use_mcts=use_mcts,
            simulations=simulations,
        )

        worker_slots = max(1, workers)
        worker_game_counts = [0 for _ in range(worker_slots)]
        idle_engines: list[tuple[int, chess.engine.SimpleEngine]] = []
        for worker_slot in range(worker_slots):
            idle_engines.append((worker_slot, self._open_stockfish_engine(stockfish_path)))

        pending_tasks = deque(tasks)
        active_games: list[dict] = []
        model_move_calls = 0
        model_move_positions = 0
        model_move_time_s = 0.0
        stockfish_move_time_s = 0.0

        def finalize_game(game: dict, score: float | None = None):
            if score is None:
                score = self._score_game_result(game["board"], bool(game["model_is_white"]))
            all_opponent_elos.append(float(game["level"]))
            all_scores.append(float(score))
            worker_slot = int(game.get("worker_slot", -1))
            if 0 <= worker_slot < len(worker_game_counts):
                worker_game_counts[worker_slot] += 1
            idle_engines.append((worker_slot, game["engine"]))
            if progress_bar is not None:
                progress_bar.update(1)

        def launch_next_game(worker_slot: int, engine: chess.engine.SimpleEngine):
            while pending_tasks and not self._is_cancelled():
                level, _, model_is_white = pending_tasks.popleft()
                try:
                    self._configure_stockfish_engine(engine, level)
                except Exception as exc:
                    self._log_limited(
                        "stockfish_config_fail",
                        f"  Warning: failed to configure Stockfish at level {level}: {exc}",
                    )
                    all_opponent_elos.append(float(level))
                    all_scores.append(0.0)
                    if 0 <= worker_slot < len(worker_game_counts):
                        worker_game_counts[worker_slot] += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                    self._force_close_engine(engine)
                    engine = self._open_stockfish_engine(stockfish_path)
                    continue

                state = batched_player.create_state()
                batched_player.reset_state(state)
                active_games.append(
                    {
                        "level": level,
                        "model_is_white": bool(model_is_white),
                        "board": chess.Board(),
                        "state": state,
                        "engine": engine,
                        "worker_slot": int(worker_slot),
                        "ply_count": 0,
                    }
                )
                return
            idle_engines.append((worker_slot, engine))

        while idle_engines and pending_tasks and not self._is_cancelled():
            worker_slot, engine = idle_engines.pop()
            launch_next_game(worker_slot, engine)

        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            while active_games and not self._is_cancelled():
                still_active: list[dict] = []
                for game in active_games:
                    if game["ply_count"] >= max_half_moves or game["board"].is_game_over(claim_draw=True):
                        finalize_game(game)
                    else:
                        still_active.append(game)
                active_games = still_active

                while idle_engines and pending_tasks and not self._is_cancelled():
                    worker_slot, engine = idle_engines.pop()
                    launch_next_game(worker_slot, engine)
                if not active_games or self._is_cancelled():
                    continue

                model_turn_games: list[dict] = []
                stockfish_turn_games: list[dict] = []
                for game in active_games:
                    batched_player.record_state(game["state"], game["board"])
                    model_turn = (game["board"].turn == chess.WHITE) == game["model_is_white"]
                    if model_turn:
                        model_turn_games.append(game)
                    else:
                        stockfish_turn_games.append(game)

                stockfish_futures = {}
                stockfish_t0 = time.perf_counter()
                if stockfish_turn_games:
                    stockfish_futures = {
                        executor.submit(
                            game["engine"].play,
                            game["board"].copy(stack=False),
                            chess.engine.Limit(time=stockfish_time_limit),
                        ): game
                        for game in stockfish_turn_games
                    }

                if model_turn_games:
                    model_t0 = time.perf_counter()
                    moves = batched_player.best_moves(
                        [game["board"] for game in model_turn_games],
                        [game["state"] for game in model_turn_games],
                    )
                    model_move_time_s += time.perf_counter() - model_t0
                    model_move_calls += 1
                    model_move_positions += len(model_turn_games)
                    for game, move in zip(model_turn_games, moves):
                        if move is None or move not in game["board"].legal_moves:
                            move = next(iter(game["board"].legal_moves), None)
                        if move is None:
                            game["ply_count"] = max_half_moves
                            continue
                        game["board"].push(move)
                        batched_player.on_move_played(game["state"], move)
                        game["ply_count"] += 1

                if stockfish_futures:
                    for future in as_completed(stockfish_futures):
                        game = stockfish_futures[future]
                        try:
                            result = future.result()
                            move = result.move
                        except Exception as exc:
                            self._log_limited(
                                "stockfish_batch_play_fail",
                                f"  Warning: Stockfish move failed at level {game['level']}: {exc}",
                            )
                            move = None
                        if move is None or move not in game["board"].legal_moves:
                            game["ply_count"] = max_half_moves
                            continue
                        game["board"].push(move)
                        batched_player.on_move_played(game["state"], move)
                        game["ply_count"] += 1
                    stockfish_move_time_s += time.perf_counter() - stockfish_t0

                finished_games: list[dict] = []
                still_active = []
                for game in active_games:
                    if game["ply_count"] >= max_half_moves or game["board"].is_game_over(claim_draw=True):
                        finished_games.append(game)
                    else:
                        still_active.append(game)
                active_games = still_active
                for game in finished_games:
                    finalize_game(game)

                while idle_engines and pending_tasks and not self._is_cancelled():
                    worker_slot, engine = idle_engines.pop()
                    launch_next_game(worker_slot, engine)
        finally:
            if self._is_cancelled():
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=False)

        if stats_out is not None:
            stats_out["model_move_calls"] = int(model_move_calls)
            stats_out["model_move_positions"] = int(model_move_positions)
            stats_out["model_move_time_s"] = float(model_move_time_s)
            stats_out["stockfish_move_time_s"] = float(stockfish_move_time_s)
            stats_out["worker_game_counts"] = list(worker_game_counts)
        else:
            self._last_elo_batch_stats = {
                "model_move_calls": int(model_move_calls),
                "model_move_positions": int(model_move_positions),
                "model_move_time_s": float(model_move_time_s),
                "stockfish_move_time_s": float(stockfish_move_time_s),
                "worker_game_counts": list(worker_game_counts),
            }

        if (
            model_move_calls > 0
            and stats_out is None
            and bool(self.elo_config.get("elo_verbose_batch_stats", False))
        ):
            avg_model_batch = float(model_move_positions) / float(model_move_calls)
            print(
                "  Info: Elo model batching: "
                f"avg_batch={avg_model_batch:.1f}, calls={model_move_calls}, "
                f"model_time={model_move_time_s:.1f}s, stockfish_wait={stockfish_move_time_s:.1f}s"
            )
        if (
            sum(worker_game_counts) > 0
            and stats_out is None
            and bool(self.elo_config.get("elo_verbose_worker_stats", False))
        ):
            print(f"  Info: Elo games/worker: {_format_worker_game_counts(worker_game_counts)}")

        return all_opponent_elos, all_scores

    def _play_game(
        self,
        engine: chess.engine.SimpleEngine,
        player: _ModelPlayer,
        model_is_white: bool,
        sf_time_limit: float,
        max_moves: int,
    ) -> float | None:
        """
        Play a single game. Returns score from the model perspective:
        1.0 = win, 0.5 = draw, 0.0 = loss. Returns None when cancelled.
        """
        if self._is_cancelled():
            return None

        board = chess.Board()
        player.reset()

        for _ in range(max_moves * 2):  # max half-moves
            if self._is_cancelled():
                return None
            if board.is_game_over(claim_draw=True):
                break

            # Keep the model-side history aligned with the real game, not only
            # with plies where the model is to move.
            player.record_state(board)

            model_turn = (board.turn == chess.WHITE) == model_is_white

            if model_turn:
                # Model plays
                move = player.best_move(board)
                if move is None or move not in board.legal_moves:
                    # Fallback: first legal move
                    move = next(iter(board.legal_moves), None)
                    if move is None:
                        break
            else:
                # Stockfish plays
                result = engine.play(board, chess.engine.Limit(time=sf_time_limit))
                move = result.move
                if move is None:
                    break

            board.push(move)
            player.on_move_played(move)

        # Determine result
        return self._score_game_result(board, model_is_white)


# ---------------------------------------------------------------------------
# Convenience wrapper for the training loop
# ---------------------------------------------------------------------------

def estimate_model_elo(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    elo_config: dict | None = None,
    stop_event=None,
) -> dict:
    """
    High-level function called from the training loop.

    Args:
        model: ChessNet in eval mode.
        config: Full training config dict.
        device: torch device.
        elo_config: elo_estimation section of config (or auto-read from config).

    Returns:
        dict  (see EloEstimator.estimate)
    """
    if elo_config is None:
        elo_config = config.get("elo_estimation", {})

    if not elo_config.get("enabled", False):
        return {"estimated_elo": None, "results": {}, "total_games": 0, "total_time": 0.0,
                "skipped": True}

    stockfish_path = elo_config.get("stockfish_path", "stockfish")
    levels = elo_config.get("levels", [1000, 1300, 1600, 1900, 2200])
    games_per_level = elo_config.get("games_per_level", 4)
    sf_time = elo_config.get("stockfish_time_limit", 0.05)
    max_moves = elo_config.get("max_moves", 150)
    use_mcts = elo_config.get("use_mcts", False)
    simulations = elo_config.get("mcts_simulations", 100)
    workers = elo_config.get("workers", 0)

    estimator = EloEstimator(
        model,
        config,
        device,
        stockfish_path,
        stop_event=stop_event,
        elo_config=elo_config,
    )
    return estimator.estimate(
        levels=levels,
        games_per_level=games_per_level,
        stockfish_time_limit=sf_time,
        max_moves=max_moves,
        use_mcts=use_mcts,
        simulations=simulations,
        workers=workers,
    )

