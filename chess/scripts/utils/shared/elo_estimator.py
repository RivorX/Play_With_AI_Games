"""
Elo Estimation via Stockfish matches

Plays fast games against Stockfish at various UCI_Elo levels
and computes the model's estimated Elo from match results.

Uses python-chess's chess.engine module for Stockfish communication.
Auto-downloads Stockfish if not found on the system.
"""

import contextlib
import io
import os
import platform
import subprocess
import shutil
import stat
import threading
import time
import zipfile
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
    Uses only stdlib (urllib) — no extra dependencies.
    """
    if not _HAS_URLLIB:
        raise RuntimeError("urllib not available — cannot auto-download Stockfish.")

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

    1. If configured_path is an absolute path to an existing file → use it.
    2. If configured_path is on PATH (e.g. "stockfish") → use it.
    3. Check project-local cache  (chess/engines/stockfish*).
    4. Auto-download from GitHub releases → cache locally.

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
    chess_dir = script_dir.parents[2]  # utils/shared → scripts/utils/shared → chess/
    engines_dir = chess_dir / "engines"

    # Look for existing cached binary
    if engines_dir.exists():
        for candidate in engines_dir.rglob("stockfish*"):
            if candidate.is_file() and candidate.stat().st_size > 1_000_000:
                print(f"  \u265a Using cached Stockfish: {candidate}")
                return str(candidate)

    # 4. Download
    print("  \u265a Stockfish not found — downloading automatically...")
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
    # Edge cases: perfect score or zero score → cap at ±800 from avg opponent
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
        child.parent = None
        child.parent_edge_index = -1
        state["root"] = child
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
        self.config = config
        self.device = device
        self.stockfish_path = stockfish_path
        self.stop_event = stop_event
        self.elo_config = dict(elo_config or {})
        self._error_counters: dict[str, int] = {}
        self._thread_local = threading.local()
        self._worker_engines: list[chess.engine.SimpleEngine] = []
        self._worker_engines_lock = threading.Lock()

    def _is_cancelled(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

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

        tasks = []
        for level in levels:
            for game_idx in range(games_per_level):
                model_is_white = (game_idx % 2 == 0)
                tasks.append((level, game_idx, model_is_white))

        workers = self._resolve_workers(workers, total_games=len(tasks))
        batch_model_moves = bool(self.elo_config.get("batch_model_moves", True))
        print(f"  Info: Elo workers resolved to {workers} (available_cpu={_available_cpu_count()}).")
        print(
            "  Info: Stockfish config: "
            f"threads={self._resolve_stockfish_threads()}, "
            f"hash={self._resolve_stockfish_hash_mb()} MB, "
            f"time_limit={float(stockfish_time_limit):.3f}s, "
            f"games={len(tasks)}"
        )
        if batch_model_moves and workers > 1:
            model_path = "batched_mcts" if use_mcts else "batched_raw"
            print(f"  Info: Elo model path: {model_path} (shared batch scheduling enabled).")
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
                total=len(tasks),
                desc="Elo games",
                unit="game",
                dynamic_ncols=True,
                leave=True,
            )

        interrupted_by_user = False

        try:
            if batch_model_moves and workers > 1:
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
                    if self.stop_event is not None:
                        with contextlib.suppress(Exception):
                            self.stop_event.set()
                    print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
                finally:
                    self._close_worker_engines()
            elif workers > 1:
                executor = ThreadPoolExecutor(max_workers=workers)
                futures = {}
                try:
                    for level, game_idx, model_is_white in tasks:
                        if self._is_cancelled():
                            break
                        future = executor.submit(
                            self._play_single_game_worker,
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
                                result = future.result()
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
                        if self.stop_event is not None:
                            with contextlib.suppress(Exception):
                                self.stop_event.set()
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
            else:
                player = _ModelPlayer(self.model, self.config, self.device, use_mcts, simulations)
                engine = chess.engine.SimpleEngine.popen_uci(
                    resolved_path,
                    **self._stockfish_popen_kwargs(),
                )
                try:
                    try:
                        for level in levels:
                            if self._is_cancelled():
                                break
                            self._configure_stockfish_engine(engine, level)
                            for game_idx in range(games_per_level):
                                if self._is_cancelled():
                                    break
                                model_is_white = (game_idx % 2 == 0)
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
                        if self.stop_event is not None:
                            with contextlib.suppress(Exception):
                                self.stop_event.set()
                        print("\nCtrl+C detected during Elo estimation. Cancelling remaining games...")
                finally:
                    self._force_close_engine(engine)
        finally:
            if progress_bar is not None:
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
            results_per_level[level] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": score_pct,
            }

        elapsed = time.perf_counter() - t0
        estimated_elo = _performance_rating(all_opponent_elos, all_scores)

        return {
            "estimated_elo": round(estimated_elo) if estimated_elo is not None else None,
            "results": results_per_level,
            "total_games": len(all_scores),
            "total_time": elapsed,
        }
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
    ) -> tuple[list[float], list[float]]:
        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        max_half_moves = max(1, int(max_moves)) * 2

        batched_player = _BatchedModelPlayer(
            self.model,
            self.config,
            self.device,
            use_mcts=use_mcts,
            simulations=simulations,
        )

        idle_engines: list[chess.engine.SimpleEngine] = []
        for _ in range(max(1, workers)):
            idle_engines.append(self._open_stockfish_engine(stockfish_path))

        pending_tasks = list(tasks)
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
            idle_engines.append(game["engine"])
            if progress_bar is not None:
                progress_bar.update(1)

        def launch_next_game(engine: chess.engine.SimpleEngine):
            while pending_tasks and not self._is_cancelled():
                level, _, model_is_white = pending_tasks.pop(0)
                try:
                    self._configure_stockfish_engine(engine, level)
                except Exception as exc:
                    self._log_limited(
                        "stockfish_config_fail",
                        f"  Warning: failed to configure Stockfish at level {level}: {exc}",
                    )
                    all_opponent_elos.append(float(level))
                    all_scores.append(0.0)
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
                        "ply_count": 0,
                    }
                )
                return
            idle_engines.append(engine)

        while idle_engines and pending_tasks and not self._is_cancelled():
            launch_next_game(idle_engines.pop())

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
                    launch_next_game(idle_engines.pop())
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
                    launch_next_game(idle_engines.pop())
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

        if model_move_calls > 0:
            avg_model_batch = float(model_move_positions) / float(model_move_calls)
            print(
                "  Info: Elo model batching: "
                f"avg_batch={avg_model_batch:.1f}, calls={model_move_calls}, "
                f"model_time={model_move_time_s:.1f}s, stockfish_wait={stockfish_move_time_s:.1f}s"
            )

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

