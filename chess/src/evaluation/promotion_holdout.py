"""Load or build the rotating promotion-confirmation opening bank on demand."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from src.game import backend as chess

_BANK_SCHEMA_VERSION = 2
_BANK_FAMILY_COUNT = 24
_OPENINGS_PER_FAMILY = 256
_BANK_SUITE_COUNT = 64
_OPENINGS_PER_SUITE = 96
_PLIES_PER_OPENING = 8
_CACHE_PATH = Path(__file__).resolve().parents[2] / "logs" / "promotion_holdout_bank.json"
_PGN_PATH = Path(__file__).resolve().parents[2] / "data" / "lichess_elite_2025-11.pgn"

_BANK_RECORDS: tuple[dict, ...] | None = None
_PROMOTION_HOLDOUT_BANK: tuple[tuple[tuple[str, ...], ...], ...] | None = None


def _position_key(line: tuple[str, ...]) -> int:
    """Replay a UCI line and return the evaluator's position key."""
    board = chess.new_board()
    for uci in line:
        move = chess.move_from_uci(uci)
        if move is None or move not in chess.legal_moves(board):
            raise ValueError(f"Illegal UCI move {uci!r} after prefix {line!r}")
        chess.apply_move(board, move)
    return chess.position_key(board)


def _canonical_opening_lines() -> tuple[tuple[str, ...], ...]:
    from src.mcts.search import _EVAL_OPENING_LINES

    return tuple(tuple(line) for line in _EVAL_OPENING_LINES)


def _suite_digest(opening_lines: tuple[tuple[str, ...], ...]) -> str:
    return hashlib.sha256(repr(opening_lines).encode("utf-8")).hexdigest()


def _family_half_difference(opening_lines: tuple[tuple[str, ...], ...]) -> int:
    first_half = Counter(tuple(line[:2]) for line in opening_lines[:48])
    second_half = Counter(tuple(line[:2]) for line in opening_lines[48:])
    families = first_half.keys() | second_half.keys()
    return max(
        (abs(first_half[family] - second_half[family]) for family in families),
        default=0,
    )


def _opening_visitor(pgn):
    """Return a parser visitor that retains only the opening prefix."""
    class Visitor(pgn.BaseVisitor):
        def __init__(self):
            self.moves = []
            self.final_board = None

        def begin_variation(self):
            return pgn.SKIP

        def begin_parse_san(self, board, san):
            if len(self.moves) == _PLIES_PER_OPENING:
                return pgn.SKIP

        def visit_move(self, board, move):
            self.moves.append(move)

        def visit_board(self, board):
            if len(self.moves) == _PLIES_PER_OPENING and self.final_board is None:
                self.final_board = board.copy()

        def result(self):
            return tuple(self.moves), self.final_board

    return Visitor


def _build_bank() -> dict:
    """Read only until 24 opening families each have 256 unique positions."""
    if not _PGN_PATH.is_file():
        raise FileNotFoundError(
            f"Promotion confirmation needs a cached bank or source PGN: {_PGN_PATH}"
        )

    import chess as python_chess
    import chess.pgn as pgn

    def position_fen(board) -> tuple[str, ...]:
        # Ignore move clocks, which are not part of the evaluation position.
        return tuple(board.fen(en_passant="fen").split()[:4])

    canonical_fens = set()
    for line in _canonical_opening_lines():
        board = python_chess.Board()
        for uci in line:
            board.push_uci(uci)
        canonical_fens.add(position_fen(board))

    families: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    seen_fens = set(canonical_fens)
    seen_positions = set()
    ready_count = 0
    opening_visitor = _opening_visitor(pgn)

    with _PGN_PATH.open("r", encoding="utf-8", errors="replace") as pgn_file:
        while opening_data := pgn.read_game(pgn_file, Visitor=opening_visitor):
            moves, board = opening_data
            if len(moves) != _PLIES_PER_OPENING or board is None:
                continue

            opening = tuple(move.uci() for move in moves)
            fen = position_fen(board)
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            # FEN is the cheap primary dedupe; the runtime key also merges
            # equivalent en-passant encodings before a candidate is retained.
            position = _position_key(opening)
            if position in seen_positions:
                continue
            seen_positions.add(position)

            family = (opening[0], opening[1])
            family_openings = families.setdefault(family, [])
            if len(family_openings) < _OPENINGS_PER_FAMILY:
                family_openings.append(opening)
                if len(family_openings) == _OPENINGS_PER_FAMILY:
                    ready_count += 1
                    if ready_count == _BANK_FAMILY_COUNT:
                        break

    ready_families = sorted(
        family for family, openings in families.items()
        if len(openings) == _OPENINGS_PER_FAMILY
    )
    if len(ready_families) != _BANK_FAMILY_COUNT:
        raise ValueError(
            f"PGN supplied only {len(ready_families)} of {_BANK_FAMILY_COUNT} "
            f"opening families with {_OPENINGS_PER_FAMILY} unique positions each"
        )

    suites = []
    for suite_index in range(_BANK_SUITE_COUNT):
        first_half, second_half = [], []
        for family in ready_families:
            start = suite_index * 4
            first_half.extend(families[family][start : start + 2])
            second_half.extend(families[family][start + 2 : start + 4])
        openings = tuple(first_half + second_half)
        digest = _suite_digest(openings)
        suites.append(
            {
                "index": suite_index,
                "suite_id": f"promotion-holdout-bank-v2-suite{suite_index:02d}-{digest[:12]}",
                "sha256": digest,
                "openings": openings,
            }
        )
    return {"schema_version": _BANK_SCHEMA_VERSION, "suites": suites}


def _write_cache(data: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_CACHE_PATH.parent,
            prefix=f"{_CACHE_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            temp_path = Path(cache_file.name)
            json.dump(data, cache_file, separators=(",", ":"))
        os.replace(temp_path, _CACHE_PATH)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _validate_bank_data(records: tuple[dict, ...], bank: tuple) -> None:
    if len(bank) != _BANK_SUITE_COUNT:
        raise ValueError(
            f"Expected {_BANK_SUITE_COUNT} promotion holdout suites, "
            f"got {len(bank)}"
        )
    if len(records) != len(bank):
        raise ValueError("Promotion holdout bank record and suite counts differ")

    canonical_keys = {
        _position_key(line) for line in _canonical_opening_lines()
    }
    all_final_keys = set()
    for index, (record, opening_lines) in enumerate(
        zip(records, bank)
    ):
        if record.get("index") != index:
            raise ValueError(f"Promotion holdout suite index mismatch at {index}")
        if len(opening_lines) != _OPENINGS_PER_SUITE:
            raise ValueError(
                f"Promotion holdout suite {index} has {len(opening_lines)} lines, "
                f"expected {_OPENINGS_PER_SUITE}"
            )

        digest = _suite_digest(opening_lines)
        expected_id = f"promotion-holdout-bank-v2-suite{index:02d}-{digest[:12]}"
        if record.get("sha256") != digest or record.get("suite_id") != expected_id:
            raise ValueError(f"Promotion holdout suite {index} has invalid content identity")
        if _family_half_difference(opening_lines) > 1:
            raise ValueError(
                f"Promotion holdout suite {index} is not stratified across its two halves"
            )

        suite_keys = set()
        for line_index, line in enumerate(opening_lines):
            if len(line) != _PLIES_PER_OPENING:
                raise ValueError(
                    f"Suite {index} line {line_index} has {len(line)} plies, "
                    f"expected {_PLIES_PER_OPENING}"
                )
            if any(not isinstance(move, str) or not move for move in line):
                raise ValueError(
                    f"Suite {index} line {line_index} contains an invalid UCI move"
                )
            key = _position_key(line)
            if key in suite_keys:
                raise ValueError(
                    f"Duplicate final position in promotion holdout suite {index}"
                )
            if key in canonical_keys:
                raise ValueError(
                    f"Promotion holdout suite {index} overlaps canonical eval positions"
                )
            if key in all_final_keys:
                raise ValueError(
                    f"Duplicate final position across promotion holdout suites at suite {index}"
                )
            suite_keys.add(key)
            all_final_keys.add(key)


def _ensure_bank() -> None:
    global _BANK_RECORDS, _PROMOTION_HOLDOUT_BANK
    if _BANK_RECORDS is not None:
        return

    generated = not _CACHE_PATH.is_file()
    if generated:
        data = _build_bank()
    else:
        with _CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            data = json.load(cache_file)
    if data.get("schema_version") != _BANK_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported promotion holdout bank schema: {data.get('schema_version')!r}"
        )
    records = tuple(data["suites"])
    bank = tuple(
        tuple(tuple(line) for line in record["openings"])
        for record in records
    )
    _validate_bank_data(records, bank)
    if generated:
        _write_cache(data)
    _BANK_RECORDS = records
    _PROMOTION_HOLDOUT_BANK = bank


def promotion_holdout_suite_for_eval(
    eval_index: int,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    """Return the rotating suite for a nonnegative persisted promotion counter."""
    if isinstance(eval_index, bool) or not isinstance(eval_index, int):
        raise TypeError(f"eval_index must be an int, got {type(eval_index).__name__}")
    if eval_index < 0:
        raise ValueError(f"eval_index must be nonnegative, got {eval_index}")
    _ensure_bank()
    assert _BANK_RECORDS is not None and _PROMOTION_HOLDOUT_BANK is not None
    suite_index = eval_index % len(_PROMOTION_HOLDOUT_BANK)
    return _BANK_RECORDS[suite_index]["suite_id"], _PROMOTION_HOLDOUT_BANK[suite_index]


def validate_promotion_holdout_bank() -> bool:
    """Load the ignored cache or build and validate it from the local PGN."""
    _ensure_bank()
    return True
