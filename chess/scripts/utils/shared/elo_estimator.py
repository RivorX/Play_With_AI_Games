"""
Elo Estimation via Stockfish matches

Plays fast games against Stockfish at various UCI_Elo levels
and computes the model's estimated Elo from match results.

Uses python-chess's chess.engine module for Stockfish communication.
Auto-downloads Stockfish if not found on the system.
"""

import io
import os
import platform
import shutil
import stat
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

try:
    import urllib.request
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

# Imports are resolved at runtime (script_dir-based sys.path setup in train_il.py)
from src.utils.data_helpers import board_to_tensor, move_to_index
from src.mcts import MCTS, select_move_by_visits


# ---------------------------------------------------------------------------
# Stockfish auto-download
# ---------------------------------------------------------------------------

# Default Stockfish version to download
_SF_VERSION = "stockfish-windows-x86-64-avx2"  # Most modern Windows CPUs
_SF_TAG = "sf_17.1"  # GitHub release tag
_SF_REPO = "official-stockfish/Stockfish"


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

    def _build_input_tensor(self, board: chess.Board) -> np.ndarray:
        """Build (C, 8, 8) input tensor including history planes."""
        tensors: list[np.ndarray] = []

        if self.history_positions > 0 and self.board_history:
            # Collect up to history_positions past boards (most recent last)
            history = self.board_history[-(self.history_positions):]
            for hb in history:
                tensors.append(board_to_tensor(hb))
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
            policy_logits, _ = self.model(t)

        policy = policy_logits.squeeze(0).cpu().numpy()

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
    ):
        self.model = model
        self.config = config
        self.device = device
        self.stockfish_path = stockfish_path

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

        Args:
            levels: Stockfish UCI_Elo levels to test against.
            games_per_level: Number of games per level (half as white, half as black).
            stockfish_time_limit: Seconds per Stockfish move.
            max_moves: Maximum full moves before declaring a draw.
            use_mcts: Enable MCTS for model moves (slower but stronger).
            simulations: MCTS simulations per move (if use_mcts=True).
            workers: Number of parallel game workers (0=auto, 1=sequential).

        Returns:
            dict with keys:
                estimated_elo (float | None)
                results (dict[int, dict])   – per-level {wins, draws, losses, score}
                total_games (int)
                total_time (float)          – wall-clock seconds
        """
        if levels is None:
            levels = [1000, 1300, 1600, 1900, 2200]

        # Auto-detect workers
        if workers == 0 or workers is None:
            workers = max(1, os.cpu_count() - 2)
        workers = max(1, int(workers))

        self.model.eval()

        # Try to open Stockfish (auto-download if needed) - just for validation
        resolved_path = ensure_stockfish(self.stockfish_path)
        try:
            test_engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
            # Auto-detect engine Elo limits from UCI options
            sf_min_elo = 1320
            sf_max_elo = 3190
            if "UCI_Elo" in test_engine.options:
                opt = test_engine.options["UCI_Elo"]
                if hasattr(opt, "min") and opt.min is not None:
                    sf_min_elo = int(opt.min)
                if hasattr(opt, "max") and opt.max is not None:
                    sf_max_elo = int(opt.max)
            test_engine.quit()
        except FileNotFoundError:
            print(f"  ⚠️  Stockfish not found at '{resolved_path}' — Elo estimation skipped.")
            return {"estimated_elo": None, "results": {}, "total_games": 0, "total_time": 0.0,
                    "error": "stockfish_not_found"}
        except Exception as e:
            print(f"  ⚠️  Stockfish error: {e} — Elo estimation skipped.")
            return {"estimated_elo": None, "results": {}, "total_games": 0, "total_time": 0.0,
                    "error": str(e)}

        # Filter/clamp levels to engine's supported range
        valid_levels = []
        for lvl in levels:
            clamped = max(sf_min_elo, min(sf_max_elo, lvl))
            if clamped != lvl:
                print(f"  ⚠️  Clamped level {lvl} → {clamped} (SF range: {sf_min_elo}-{sf_max_elo})")
            if clamped not in valid_levels:
                valid_levels.append(clamped)
        levels = valid_levels

        if not levels:
            print("  ⚠️  No valid Elo levels after clamping.")
            return {"estimated_elo": None, "results": {}, "total_games": 0,
                    "total_time": 0.0, "error": "no_valid_levels"}

        t0 = time.perf_counter()
        all_opponent_elos: list[float] = []
        all_scores: list[float] = []
        results_per_level: dict[int, dict] = {}

        # Build task list: (level, game_idx, model_is_white)
        tasks = []
        for level in levels:
            for game_idx in range(games_per_level):
                model_is_white = (game_idx % 2 == 0)
                tasks.append((level, game_idx, model_is_white))

        # Execute games in parallel
        if workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._play_single_game_worker,
                        level, model_is_white, use_mcts, simulations,
                        stockfish_time_limit, max_moves, resolved_path
                    ): (level, game_idx)
                    for level, game_idx, model_is_white in tasks
                }

                for future in as_completed(futures):
                    level, _ = futures[future]
                    try:
                        result = future.result()
                        all_opponent_elos.append(float(level))
                        all_scores.append(result)
                    except Exception as exc:
                        print(f"  ⚠️  Game at level {level} failed: {exc}")
                        # Count as loss
                        all_opponent_elos.append(float(level))
                        all_scores.append(0.0)
        else:
            # Sequential execution (original behavior)
            player = _ModelPlayer(self.model, self.config, self.device, use_mcts, simulations)
            engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
            
            try:
                for level in levels:
                    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": level})
                    
                    for game_idx in range(games_per_level):
                        model_is_white = (game_idx % 2 == 0)
                        result = self._play_game(
                            engine, player, model_is_white,
                            stockfish_time_limit, max_moves,
                        )
                        all_opponent_elos.append(float(level))
                        all_scores.append(result)
            finally:
                engine.quit()

        # Aggregate results per level
        for level in levels:
            level_scores = [
                score for elo, score in zip(all_opponent_elos, all_scores) if elo == level
            ]
            wins = sum(1 for s in level_scores if s == 1.0)
            draws = sum(1 for s in level_scores if s == 0.5)
            losses = sum(1 for s in level_scores if s == 0.0)
            total = wins + draws + losses
            score_pct = (wins + 0.5 * draws) / total if total else 0.0
            results_per_level[level] = {
                "wins": wins, "draws": draws, "losses": losses,
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
    ) -> float:
        """
        Worker function for parallel game execution.
        Creates own Stockfish engine and plays one game.
        
        Returns: score (1.0=win, 0.5=draw, 0.0=loss) from model's perspective.
        """
        # Each worker needs its own engine and player
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": level})
        except Exception as e:
            print(f"  ⚠️  Worker failed to open Stockfish: {e}")
            return 0.0  # Count as loss
        
        player = _ModelPlayer(self.model, self.config, self.device, use_mcts, simulations)
        
        try:
            result = self._play_game(engine, player, model_is_white, sf_time_limit, max_moves)
        except Exception as e:
            print(f"  ⚠️  Game failed: {e}")
            result = 0.0
        finally:
            engine.quit()
        
        return result

    def _play_game(
        self,
        engine: chess.engine.SimpleEngine,
        player: _ModelPlayer,
        model_is_white: bool,
        sf_time_limit: float,
        max_moves: int,
    ) -> float:
        """
        Play a single game.  Returns score from the model's perspective:
        1.0 = win, 0.5 = draw, 0.0 = loss.
        """
        board = chess.Board()
        player.reset()

        for _ in range(max_moves * 2):  # max half-moves
            if board.is_game_over(claim_draw=True):
                break

            model_turn = (board.turn == chess.WHITE) == model_is_white

            if model_turn:
                # Model plays
                player.record_state(board)
                move = player.best_move(board)
                if move is None or move not in board.legal_moves:
                    # Fallback: random legal move
                    moves = list(board.legal_moves)
                    if not moves:
                        break
                    move = moves[0]
            else:
                # Stockfish plays
                result = engine.play(board, chess.engine.Limit(time=sf_time_limit))
                move = result.move
                if move is None:
                    break

            board.push(move)

        # Determine result
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1.0 if model_is_white else 0.0
        elif result == "0-1":
            return 0.0 if model_is_white else 1.0
        else:
            return 0.5


# ---------------------------------------------------------------------------
# Convenience wrapper for the training loop
# ---------------------------------------------------------------------------

def estimate_model_elo(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    elo_config: dict | None = None,
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

    estimator = EloEstimator(model, config, device, stockfish_path)
    return estimator.estimate(
        levels=levels,
        games_per_level=games_per_level,
        stockfish_time_limit=sf_time,
        max_moves=max_moves,
        use_mcts=use_mcts,
        simulations=simulations,
        workers=workers,
    )
