"""
UCI adapter for Chess AI models.

Implements a minimal-but-practical UCI command set:
- uci / isready / ucinewgame / quit / stop
- position startpos [moves ...]
- position fen <fen> [moves ...]
- go [movetime|wtime btime winc binc movestogo]
- setoption name UseMCTS|Simulations|MoveOverhead|Temperature
"""

import argparse
import copy
import sys
from pathlib import Path

import chess
import torch
import yaml


script_dir = Path(__file__).resolve().parent
chess_dir = script_dir.parents[2]
sys.path.insert(0, str(chess_dir))

from src.mcts import MCTS, select_move_by_visits
from src.model import ChessNet
from src.utils.data_helpers import board_to_tensor, move_to_index


class UCIChessEngine:
    """UCI front-end for ChessNet."""

    def __init__(self, config, checkpoint_path, device, use_mcts=True, simulations=None):
        self.config = copy.deepcopy(config)
        self.device = device
        self.version = self.config.get("model", {}).get("version", "v?.?")
        self.history_positions = int(self.config.get("model", {}).get("history_positions", 0))

        self.board = chess.Board()
        self.board_history = []
        self.stop_requested = False

        self.use_mcts = bool(use_mcts)
        self.simulations = int(
            simulations
            if simulations is not None
            else self.config.get("reinforcement_learning", {}).get("mcts_simulations", 200)
        )
        self.move_overhead_ms = 50
        self.temperature = 0.0

        self.use_amp = bool(
            self.config.get("hardware", {}).get("use_amp", True) and self.device.type == "cuda"
        )
        use_bf16 = bool(self.config.get("hardware", {}).get("use_bfloat16", False))
        self.amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.model = ChessNet(self.config).to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"info string loaded model: {checkpoint_path}", flush=True)
        else:
            print("info string warning: checkpoint not found, using untrained weights", flush=True)

        self.model.eval()
        self.mcts = MCTS(self.model, self.config, self.device) if self.use_mcts else None

    def _send(self, text):
        print(text, flush=True)

    @staticmethod
    def _parse_bool(value):
        return str(value).strip().lower() in {"1", "true", "on", "yes"}

    def _ensure_mcts(self):
        if self.use_mcts and self.mcts is None:
            self.mcts = MCTS(self.model, self.config, self.device)

    def _record_pre_move_state(self):
        self.board_history.append(self.board.copy())
        max_history = self.history_positions + 20
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]
        if self.mcts:
            self.mcts.update_history(self.board)

    def _new_game(self):
        self.board = chess.Board()
        self.board_history = []
        self.stop_requested = False
        if self.mcts:
            self.mcts.reset_tree()

    def _build_history_tensor(self):
        if self.history_positions <= 0:
            return board_to_tensor(self.board)

        tensors = []
        if self.board_history:
            for hist_board in self.board_history[-self.history_positions:]:
                hist_tensor = board_to_tensor(
                    hist_board,
                    flip_perspective=(self.board.turn == chess.BLACK),
                )
                tensors.append(hist_tensor)

        while len(tensors) < self.history_positions:
            tensors.insert(0, torch.zeros(16, 8, 8).numpy())

        tensors.append(board_to_tensor(self.board))

        import numpy as np

        return np.concatenate(tensors, axis=0)

    def _best_network_move(self):
        board_tensor = (
            torch.FloatTensor(self._build_history_tensor())
            .unsqueeze(0)
            .to(self.device, memory_format=torch.channels_last)
        )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                policy_log_probs, _ = self.model(board_tensor, return_aux=False)
            policy = torch.exp(policy_log_probs).cpu().numpy()[0]

        best_move = None
        best_score = -1.0
        for move in self.board.legal_moves:
            idx = move_to_index(move, self.board)
            score = float(policy[idx])
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _parse_go_args(self, args):
        parsed = {
            "wtime": None,
            "btime": None,
            "winc": 0,
            "binc": 0,
            "movestogo": None,
            "movetime": None,
            "depth": None,
        }
        i = 0
        while i < len(args):
            key = args[i]
            if key in parsed and i + 1 < len(args):
                try:
                    parsed[key] = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            i += 1
        return parsed

    def _compute_simulations(self, go_args):
        sims = max(1, int(self.simulations))
        target_ms = None

        if go_args["movetime"] is not None:
            target_ms = max(50, go_args["movetime"] - self.move_overhead_ms)
        elif go_args["wtime"] is not None and go_args["btime"] is not None:
            remaining = go_args["wtime"] if self.board.turn == chess.WHITE else go_args["btime"]
            increment = go_args["winc"] if self.board.turn == chess.WHITE else go_args["binc"]
            moves_to_go = go_args["movestogo"] or 30
            alloc = remaining / max(1, moves_to_go) + 0.5 * increment
            target_ms = max(50, int(alloc) - self.move_overhead_ms)

        if target_ms is not None:
            # Scale around a 1-second baseline.
            sims = int(round(self.simulations * (target_ms / 1000.0)))

        return max(8, min(5000, sims))

    def _best_mcts_move(self, go_args):
        self._ensure_mcts()
        sims = self._compute_simulations(go_args)
        self._send(f"info string mcts simulations {sims}")
        visit_counts = self.mcts.search(self.board, sims, temperature=self.temperature)
        if not visit_counts:
            return None
        move, _ = select_move_by_visits(visit_counts, temperature=self.temperature)
        return move

    def _set_position(self, args):
        if not args:
            return

        self.board_history = []
        if self.mcts:
            self.mcts.reset_tree()

        idx = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif args[0] == "fen":
            if len(args) < 7:
                self._send("info string invalid fen command")
                return
            fen = " ".join(args[1:7])
            try:
                self.board = chess.Board(fen)
            except ValueError:
                self._send("info string invalid fen")
                self.board = chess.Board()
                return
            idx = 7
        else:
            self._send("info string unsupported position command")
            return

        if idx < len(args) and args[idx] == "moves":
            for move_uci in args[idx + 1 :]:
                try:
                    move = chess.Move.from_uci(move_uci)
                except ValueError:
                    self._send(f"info string invalid move format: {move_uci}")
                    break
                if move not in self.board.legal_moves:
                    self._send(f"info string illegal move in position: {move_uci}")
                    break
                self._record_pre_move_state()
                self.board.push(move)

    def _set_option(self, args):
        name = ""
        value = ""
        i = 0
        while i < len(args):
            token = args[i]
            if token == "name":
                i += 1
                start = i
                while i < len(args) and args[i] != "value":
                    i += 1
                name = " ".join(args[start:i])
            elif token == "value":
                value = " ".join(args[i + 1 :])
                break
            else:
                i += 1

        key = name.strip().lower()
        if key == "usemcts":
            self.use_mcts = self._parse_bool(value)
            if self.use_mcts:
                self._ensure_mcts()
            self._send(f"info string UseMCTS set to {self.use_mcts}")
        elif key == "simulations":
            try:
                self.simulations = max(1, int(value))
                self._send(f"info string Simulations set to {self.simulations}")
            except ValueError:
                self._send("info string invalid Simulations value")
        elif key == "moveoverhead":
            try:
                self.move_overhead_ms = max(0, int(value))
                self._send(f"info string MoveOverhead set to {self.move_overhead_ms}")
            except ValueError:
                self._send("info string invalid MoveOverhead value")
        elif key == "temperature":
            try:
                self.temperature = max(0.0, float(value))
                self._send(f"info string Temperature set to {self.temperature}")
            except ValueError:
                self._send("info string invalid Temperature value")

    def _go(self, args):
        if self.board.is_game_over():
            self._send("bestmove 0000")
            return

        go_args = self._parse_go_args(args)
        move = None
        if self.use_mcts:
            move = self._best_mcts_move(go_args)
        if move is None:
            move = self._best_network_move()

        if move is None:
            self._send("bestmove 0000")
        else:
            self._send(f"bestmove {move.uci()}")

    def _uci(self):
        self._send(f"id name ChessAI-UCI {self.version}")
        self._send("id author ChessAI")
        self._send(f"option name UseMCTS type check default {'true' if self.use_mcts else 'false'}")
        self._send(f"option name Simulations type spin default {self.simulations} min 1 max 5000")
        self._send(
            f"option name MoveOverhead type spin default {self.move_overhead_ms} min 0 max 2000"
        )
        self._send("option name Temperature type spin default 0 min 0 max 2")
        self._send("uciok")

    def loop(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            args = parts[1:]

            if cmd == "uci":
                self._uci()
            elif cmd == "isready":
                self._send("readyok")
            elif cmd == "ucinewgame":
                self._new_game()
            elif cmd == "position":
                self._set_position(args)
            elif cmd == "go":
                self._go(args)
            elif cmd == "setoption":
                self._set_option(args)
            elif cmd == "stop":
                self.stop_requested = True
            elif cmd == "d":
                self._send(f"info string fen {self.board.fen()}")
            elif cmd == "quit":
                break


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = copy.deepcopy(config)
    config.setdefault("model", {})
    config["model"]["print_summary"] = False
    return config


def resolve_model_path(base_dir, config, override):
    if override:
        p = Path(override)
        if not p.is_absolute():
            if p.exists():
                return p.resolve()
            p = base_dir / p
        return p
    best_rel = config.get("paths", {}).get("best_model_il", "models/best_model_il.pt")
    return base_dir / best_rel


def main():
    parser = argparse.ArgumentParser(description="Run Chess AI as a UCI engine.")
    parser.add_argument(
        "--config",
        default=str(chess_dir / "config" / "config.yaml"),
        help="Path to config YAML.",
    )
    parser.add_argument("--model", default="", help="Checkpoint path (.pt).")
    parser.add_argument("--device", default="", help="Device override (e.g. cuda or cpu).")
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS and use network only.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Override MCTS simulations per move.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    base_dir = chess_dir
    model_path = resolve_model_path(base_dir, config, args.model)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get("hardware", {}).get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    engine = UCIChessEngine(
        config=config,
        checkpoint_path=model_path,
        device=device,
        use_mcts=not args.no_mcts,
        simulations=args.simulations,
    )
    engine.loop()


if __name__ == "__main__":
    main()
