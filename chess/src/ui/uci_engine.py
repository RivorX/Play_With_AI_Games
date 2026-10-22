"""
UCI adapter for Chess AI models.

Implements a minimal-but-practical UCI command set:
- uci / isready / ucinewgame / quit / stop
- position startpos [moves ...]
- position fen <fen> [moves ...]
- go [movetime|nodes|wtime btime winc binc movestogo]
- setoption name UseMCTS|Simulations|MoveOverhead
"""

import argparse
import copy
import sys
from pathlib import Path

import torch


script_dir = Path(__file__).resolve().parent
chess_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(chess_dir))

from src.mcts.single_game import SingleGameMCTS
from src.game import backend as chess
from src.config import default_config_path, load_project_config
from src.model import load_model
from src.models.registry import MODEL_REGISTRY
from src.models.data.registry import get_input_encoder
from src.inference.prediction import predict_raw
from src.mcts.search import _board_position_key, _position_counts_from_board
from src.models.data.se_cnn_v9.helpers import board_to_tensor, move_to_index
from src.common.runtime import legacy_architecture_scoped_file, model_alias_file


class UCIChessEngine:
    """UCI front-end for registered chess models."""

    def __init__(self, config, checkpoint_path, device, use_mcts=True, simulations=None):
        self.config = copy.deepcopy(config)
        self.device = device
        self.version = self.config.get("model", {}).get("version", "v?.?")
        self.history_positions = int(self.config.get("model", {}).get("history_positions", 0))

        self.board = chess.new_board()
        self.board_history = []
        self._repetition_history_known = True
        self._position_base = ("startpos",)
        self._position_moves = ()
        self.position_counts = {_board_position_key(self.board): 1}
        self.stop_requested = False

        self.use_mcts = bool(use_mcts)
        self.simulations = int(
            simulations
            if simulations is not None
            else self.config.get("reinforcement_learning", {}).get("mcts_simulations", 200)
        )
        self._simulations_explicit = simulations is not None
        self.move_overhead_ms = 50

        self.use_amp = bool(
            self.config.get("hardware", {}).get("use_amp", True) and self.device.type == "cuda"
        )
        use_bf16 = bool(self.config.get("hardware", {}).get("use_bfloat16", False))
        self.amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        if checkpoint_path and checkpoint_path.exists():
            self.model = load_model(
                checkpoint_path, self.config, self.device, strict=True
            )
            print(f"info string loaded model: {checkpoint_path}", flush=True)
        else:
            self.model = load_model(None, self.config, self.device, strict=True)
            print("info string warning: checkpoint not found, using untrained weights", flush=True)
        self.history_positions = int(
            self.model.model_spec["input_spec"]["history_positions"]
        )

        self.model.eval()
        self.mcts = (
            SingleGameMCTS(self.model, self.config, self.device)
            if self.use_mcts
            else None
        )

    def _send(self, text):
        print(text, flush=True)

    @staticmethod
    def _parse_bool(value):
        return str(value).strip().lower() in {"1", "true", "on", "yes"}

    def _ensure_mcts(self):
        if self.use_mcts and self.mcts is None:
            self.mcts = SingleGameMCTS(self.model, self.config, self.device)
            for history_board in self.board_history:
                key = _board_position_key(history_board)
                historical_counts = (
                    _position_counts_from_board(history_board)
                    if self._repetition_history_known
                    else None
                )
                self.mcts.update_history(
                    history_board,
                    repetition_count=(
                        max(0, int((historical_counts or {}).get(key, 1)) - 1)
                        if self._repetition_history_known
                        and self.mcts._multi._path_dependent_repetition
                        else None
                    ),
                )

    def _record_pre_move_state(self):
        self.board_history.append(chess.copy_board(self.board))
        if getattr(self.model, "input_encoder_id", "planes_history_v1") != "planes_history_v2":
            max_history = self.history_positions + 20
            if len(self.board_history) > max_history:
                self.board_history = self.board_history[-max_history:]
        if self.mcts:
            key = _board_position_key(self.board)
            self.mcts.update_history(
                self.board,
                repetition_count=(
                    max(0, int(self.position_counts.get(key, 1)) - 1)
                    if self._repetition_history_known
                    and self.mcts._multi._path_dependent_repetition
                    else None
                ),
            )

    def _new_game(self):
        self.board = chess.new_board()
        self.board_history = []
        self._repetition_history_known = True
        self._position_base = ("startpos",)
        self._position_moves = ()
        self.position_counts = {_board_position_key(self.board): 1}
        self.stop_requested = False
        if self.mcts:
            self.mcts.reset_tree()

    def _build_history_tensor(self):
        encoder_id = str(getattr(self.model, "input_encoder_id", "planes_history_v1"))
        if encoder_id == "planes_history_v2":
            encoder = get_input_encoder(encoder_id)
            history = self.board_history[-self.history_positions:] if self.history_positions else []
            repetition_counts = None
            if self._repetition_history_known:
                seen = {}
                all_boards = list(self.board_history) + [self.board]
                all_counts = []
                for history_board in all_boards:
                    key = _board_position_key(history_board)
                    all_counts.append(int(seen.get(key, 0)))
                    seen[key] = int(seen.get(key, 0)) + 1
                repetition_counts = (
                    all_counts[-(len(history) + 1):-1] + [all_counts[-1]]
                    if history
                    else [all_counts[-1]]
                )
            return encoder.encode_history(
                self.board,
                history_boards=history,
                history_positions=self.history_positions,
                repetition_counts=repetition_counts,
            )

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
        policy_logits, _ = predict_raw(
            self.model,
            board_tensor,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
        )
        policy = policy_logits.float().cpu().numpy()[0]

        best_move = None
        best_score = -float("inf")
        for move in chess.legal_moves(self.board):
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
            "nodes": None,
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

    def _compute_search_budget(self, go_args):
        if go_args.get("nodes") is not None:
            return {
                "budget_mode": "simulations",
                "num_simulations": max(1, min(5000, int(go_args["nodes"]))),
                "time_limit_seconds": None,
            }

        target_ms = None
        if go_args.get("movetime") is not None:
            target_ms = max(1, int(go_args["movetime"]) - self.move_overhead_ms)
        elif go_args.get("wtime") is not None and go_args.get("btime") is not None:
            remaining = go_args["wtime"] if self.board.turn == chess.WHITE else go_args["btime"]
            increment = go_args["winc"] if self.board.turn == chess.WHITE else go_args["binc"]
            moves_to_go = go_args["movestogo"] or 30
            alloc = remaining / max(1, moves_to_go) + 0.5 * increment
            target_ms = max(1, int(alloc) - self.move_overhead_ms)

        if target_ms is not None:
            return {
                "budget_mode": "time",
                "num_simulations": max(1, int(self.simulations)),
                "time_limit_seconds": float(target_ms) / 1000.0,
            }
        if self._simulations_explicit:
            return {
                "budget_mode": "simulations",
                "num_simulations": max(1, min(5000, int(self.simulations))),
                "time_limit_seconds": None,
            }
        return {
            "budget_mode": None,
            "num_simulations": max(1, int(self.simulations)),
            "time_limit_seconds": None,
        }

    def _best_mcts_move(self, go_args):
        self._ensure_mcts()
        budget = self._compute_search_budget(go_args)
        if budget["budget_mode"] == "time":
            self._send(f"info string mcts time {budget['time_limit_seconds']:.3f}s")
        elif budget["budget_mode"] == "simulations":
            self._send(f"info string mcts simulations {budget['num_simulations']}")
        else:
            inference_cfg = self.config.get("inference", {}) or {}
            if str(inference_cfg.get("budget_mode", "simulations")).strip().lower() == "time":
                self._send(
                    "info string mcts time "
                    f"{float(inference_cfg.get('move_time_seconds', 0.2)):.3f}s"
                )
            else:
                self._send(f"info string mcts simulations {budget['num_simulations']}")
        search_result = self.mcts.search(
            self.board,
            budget["num_simulations"],
            position_counts=self.position_counts,
            repetition_known=self._repetition_history_known,
            budget_mode=budget["budget_mode"],
            time_limit_seconds=budget["time_limit_seconds"],
            include_diagnostics=False,
        )
        if not search_result:
            return None
        return search_result.selected_move

    def _set_position(self, args):
        if not args:
            return
        idx = 0
        if args[0] == "startpos":
            base_board = chess.new_board()
            base_key = ("startpos",)
            repetition_history_known = True
            idx = 1
        elif args[0] == "fen":
            if len(args) < 7:
                self._send("info string invalid fen command")
                return
            fen = " ".join(args[1:7])
            try:
                base_board = chess.board_from_fen(fen)
                base_key = ("fen", fen)
                repetition_history_known = False
            except ValueError:
                self._send("info string invalid fen")
                return
            idx = 7
        else:
            self._send("info string unsupported position command")
            return

        move_tokens = []
        if idx < len(args):
            if args[idx] != "moves":
                self._send("info string invalid position command")
                return
            move_tokens = list(args[idx + 1 :])

        # Validate the complete supplied line before changing live state.
        probe = chess.copy_board(base_board)
        for move_uci in move_tokens:
            try:
                move = chess.move_from_uci(move_uci)
            except ValueError:
                self._send(f"info string invalid move format: {move_uci}")
                return
            if move is None or move not in chess.legal_moves(probe):
                self._send(f"info string illegal move in position: {move_uci}")
                return
            chess.apply_move(probe, move)

        requested_moves = tuple(move_tokens)
        can_extend_known_startpos = bool(
            repetition_history_known
            and base_key == ("startpos",)
            and self._repetition_history_known
            and self._position_base == base_key
            and len(requested_moves) >= len(self._position_moves)
            and requested_moves[: len(self._position_moves)] == self._position_moves
        )

        if can_extend_known_startpos:
            suffix = requested_moves[len(self._position_moves) :]
            for move_uci in suffix:
                move = chess.move_from_uci(move_uci)
                self._record_pre_move_state()
                chess.apply_move(self.board, move)
                key = _board_position_key(self.board)
                self.position_counts[key] = int(self.position_counts.get(key, 0)) + 1
                if self.mcts is not None and self.mcts.root is not None:
                    self.mcts.advance_root(
                        move,
                        board=self.board,
                        position_counts=self.position_counts,
                        repetition_known=True,
                    )
        else:
            # FEN positions do not provide complete repetition provenance, so
            # they always start a fresh search session. A changed/branched
            # startpos line also discards the old root.
            if self.mcts:
                self.mcts.reset_tree()
            self.board = base_board
            self.board_history = []
            self._repetition_history_known = repetition_history_known
            self.position_counts = {_board_position_key(self.board): 1}
            for move_uci in requested_moves:
                move = chess.move_from_uci(move_uci)
                self._record_pre_move_state()
                chess.apply_move(self.board, move)
                key = _board_position_key(self.board)
                self.position_counts[key] = int(self.position_counts.get(key, 0)) + 1

        self._position_base = base_key
        self._position_moves = requested_moves

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
                self._simulations_explicit = True
                self._send(f"info string Simulations set to {self.simulations}")
            except ValueError:
                self._send("info string invalid Simulations value")
        elif key == "moveoverhead":
            try:
                self.move_overhead_ms = max(0, int(value))
                self._send(f"info string MoveOverhead set to {self.move_overhead_ms}")
            except ValueError:
                self._send("info string invalid MoveOverhead value")

    def _go(self, args):
        if chess.is_game_over(self.board, claim_draw=True):
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

        if self.mcts is not None:
            self.mcts.close()


def load_config(config_path):
    config = load_project_config(config_path=config_path)
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
    architecture_id = str(config.get("model", {}).get("architecture_id", "se_cnn_v9"))
    configured = base_dir / best_rel
    best_path = model_alias_file(configured, architecture_id, "il")
    if best_path.is_file():
        return best_path
    return legacy_architecture_scoped_file(configured, architecture_id)


def main():
    parser = argparse.ArgumentParser(description="Run Chess AI as a UCI engine.")
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--architecture",
        choices=tuple(sorted(MODEL_REGISTRY)),
        default=None,
        help="Select a model profile when using a new or untrained model.",
    )
    parser.add_argument("--model", default="", help="Checkpoint path (.pt).")
    parser.add_argument("--device", default="", help="Device override (e.g. cuda or cpu).")
    parser.add_argument("--no-mcts", action="store_true", help="Disable MCTS and use network only.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="Use a fixed MCTS simulation budget unless UCI go supplies movetime/nodes.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_project_config(
        architecture_id=args.architecture,
        config_path=config_path,
    )
    config = copy.deepcopy(config)
    config.setdefault("model", {})
    config["model"]["print_summary"] = False
    if args.architecture and config.get("model", {}).get("architecture_id") != args.architecture:
        raise ValueError(
            f"Active config model profile conflicts with --architecture {args.architecture!r}."
        )
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
