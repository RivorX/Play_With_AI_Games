"""Load or build the rotating promotion-confirmation opening bank on demand."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path

from src.game import backend as chess

_BANK_SCHEMA_VERSION = 3
_BANK_SUITE_ID_VERSION = 2
_BANK_FAMILY_COUNT = 24
_OPENINGS_PER_FAMILY = 256
_BANK_SUITE_COUNT = 64
_OPENINGS_PER_SUITE = 96
_PLIES_PER_OPENING = 8
_LINEAGE_TEXT_EXTENSIONS = {".csv", ".json", ".yaml", ".yml", ".log", ".txt"}
_CACHE_PATH = Path(__file__).resolve().parents[2] / "logs" / "promotion_holdout_bank.json"
_PGN_PATH = Path(__file__).resolve().parents[2] / "data" / "lichess_elite_2025-11.pgn"

_BANK_RECORDS: tuple[dict, ...] | None = None
_PROMOTION_HOLDOUT_BANK: tuple[tuple[tuple[str, ...], ...], ...] | None = None
_LEDGER_DEFAULT_PATH = _CACHE_PATH.parent / "promotion_holdout_allocations.sqlite3"
_ROOT_SIGNATURE_CACHE: dict[tuple[str, int], tuple[str, ...]] = {}


class _CanonicalOverlapError(ValueError):
    """A well-formed cached bank predates the current canonical eval set."""

    def __init__(self, overlaps: tuple[dict, ...]):
        self.overlaps = overlaps
        first = overlaps[0]
        super().__init__(
            f"Promotion holdout suite {first['suite_index']} overlaps canonical eval positions"
        )


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


def _canonical_position_keys() -> set[int]:
    return {_position_key(line) for line in _canonical_opening_lines()}


def _canonical_position_fingerprint() -> str:
    keys = sorted(_canonical_position_keys())
    encoded = json.dumps(keys, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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
    canonical_keys = set()
    for line in _canonical_opening_lines():
        board = python_chess.Board()
        for uci in line:
            board.push_uci(uci)
        canonical_fens.add(position_fen(board))
        canonical_keys.add(_position_key(line))

    families: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    seen_fens = set(canonical_fens)
    # Match the runtime validator's equivalence relation as well as the FEN
    # filter, including en-passant encodings that FEN treats differently.
    seen_positions = set(canonical_keys)
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
    return {
        "schema_version": _BANK_SCHEMA_VERSION,
        "canonical_eval_sha256": _canonical_position_fingerprint(),
        "suites": suites,
    }


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

    canonical_keys = _canonical_position_keys()
    all_final_keys = set()
    canonical_overlaps = []
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
        expected_id = (
            f"promotion-holdout-bank-v{_BANK_SUITE_ID_VERSION}-"
            f"suite{index:02d}-{digest[:12]}"
        )
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
                canonical_overlaps.append(
                    {"suite_index": index, "line_index": line_index, "position_key": key}
                )
            if key in all_final_keys:
                raise ValueError(
                    f"Duplicate final position across promotion holdout suites at suite {index}"
                )
            suite_keys.add(key)
            all_final_keys.add(key)

    if canonical_overlaps:
        raise _CanonicalOverlapError(tuple(canonical_overlaps))


def _known_suite_usage(suite_ids: tuple[str, ...]) -> list[dict]:
    """Find persisted references to old suite identities for cache lineage."""
    roots = (_CACHE_PATH.parent, _CACHE_PATH.parents[1] / "experiments")
    pattern = re.compile("|".join(re.escape(value) for value in suite_ids))
    references: dict[str, list[dict]] = {suite_id: [] for suite_id in suite_ids}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _LINEAGE_TEXT_EXTENSIONS:
                continue
            if path == _CACHE_PATH or ".archive-" in path.name:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as source:
                    for line_number, line in enumerate(source, start=1):
                        for match in pattern.finditer(line):
                            references[match.group(0)].append(
                                {
                                    "path": path.relative_to(_CACHE_PATH.parents[2]).as_posix(),
                                    "line": line_number,
                                }
                            )
            except OSError:
                continue
    return [
        {"suite_id": suite_id, "references": refs}
        for suite_id, refs in references.items()
        if refs
    ]


def _write_new_file(path: Path, payload: bytes, expected_sha256: str) -> None:
    """Create an immutable archive file, accepting only an identical retry."""
    try:
        with path.open("xb") as destination:
            destination.write(payload)
    except FileExistsError:
        existing_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing_sha256 != expected_sha256:
            raise RuntimeError(f"Promotion holdout archive collision at {path}")


def _archive_stale_cache(
    data: dict,
    original_bytes: bytes,
    *,
    refresh_reason: str,
    overlaps: tuple[dict, ...] = (),
) -> dict:
    cache_sha256 = hashlib.sha256(original_bytes).hexdigest()
    archive_name = f"{_CACHE_PATH.stem}.archive-{cache_sha256}.json"
    manifest_name = f"{_CACHE_PATH.stem}.archive-{cache_sha256}.lineage.json"
    archive_path = _CACHE_PATH.with_name(archive_name)
    manifest_path = _CACHE_PATH.with_name(manifest_name)
    suites = data.get("suites", [])
    suite_records = [
        {
            "index": record.get("index"),
            "suite_id": record.get("suite_id"),
            "sha256": record.get("sha256"),
        }
        for record in suites
        if isinstance(record, dict)
    ]
    known_usage = _known_suite_usage(
        tuple(record["suite_id"] for record in suite_records if record["suite_id"])
    )
    overlap_records = []
    if overlaps:
        canonical_indices_by_key = {}
        for index, line in enumerate(_canonical_opening_lines()):
            canonical_indices_by_key.setdefault(_position_key(line), []).append(index)
        for overlap in overlaps:
            suite_index = overlap["suite_index"]
            line_index = overlap["line_index"]
            overlap_records.append(
                {
                    "suite_index": suite_index,
                    "suite_id": suites[suite_index].get("suite_id"),
                    "line_index": line_index,
                    "canonical_line_indices": canonical_indices_by_key.get(
                        overlap["position_key"], []
                    ),
                }
            )

    manifest = {
        "manifest_schema_version": 1,
        "refresh_reason": refresh_reason,
        "archived_cache": {
            "path": archive_name,
            "sha256": cache_sha256,
            "schema_version": data.get("schema_version"),
        },
        "suite_id_semantics": "content digest; identical openings retain the same suite_id",
        "suites": suite_records,
        "canonical_overlaps": overlap_records,
        "known_usage": known_usage,
        "known_usage_scan": {
            "roots": ["chess/logs", "chess/experiments"],
            "extensions": sorted(_LINEAGE_TEXT_EXTENSIONS),
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    _write_new_file(archive_path, original_bytes, cache_sha256)
    _write_new_file(manifest_path, manifest_bytes, manifest_sha256)
    return {
        "cache_sha256": cache_sha256,
        "archive_path": archive_name,
        "manifest_path": manifest_name,
        "manifest_sha256": manifest_sha256,
    }


def _rebuild_stale_cache(
    old_data: dict,
    original_bytes: bytes,
    *,
    refresh_reason: str,
    overlaps: tuple[dict, ...] = (),
) -> dict:
    replacement = _build_bank()
    records = tuple(replacement["suites"])
    bank = tuple(
        tuple(tuple(line) for line in record["openings"])
        for record in records
    )
    _validate_bank_data(records, bank)
    replacement["supersedes"] = _archive_stale_cache(
        old_data,
        original_bytes,
        refresh_reason=refresh_reason,
        overlaps=overlaps,
    )
    _write_cache(replacement)
    return replacement


def _ensure_bank() -> None:
    global _BANK_RECORDS, _PROMOTION_HOLDOUT_BANK
    if _BANK_RECORDS is not None:
        return

    generated = not _CACHE_PATH.is_file()
    original_bytes = None
    if generated:
        data = _build_bank()
    else:
        original_bytes = _CACHE_PATH.read_bytes()
        data = json.loads(original_bytes.decode("utf-8"))
    schema_version = data.get("schema_version")
    if schema_version not in (2, _BANK_SCHEMA_VERSION):
        raise RuntimeError(
            f"Unsupported promotion holdout bank schema: {schema_version!r}"
        )
    records = tuple(data["suites"])
    bank = tuple(
        tuple(tuple(line) for line in record["openings"])
        for record in records
    )
    stale_reason = None
    stale_overlaps = ()
    try:
        _validate_bank_data(records, bank)
    except _CanonicalOverlapError as error:
        stale_reason = "canonical_position_overlap"
        stale_overlaps = error.overlaps
    if (
        not generated
        and stale_reason is None
        and schema_version == _BANK_SCHEMA_VERSION
        and data.get("canonical_eval_sha256") != _canonical_position_fingerprint()
    ):
        stale_reason = "canonical_eval_fingerprint_changed"
    if stale_reason is not None:
        assert original_bytes is not None
        data = _rebuild_stale_cache(
            data,
            original_bytes,
            refresh_reason=stale_reason,
            overlaps=stale_overlaps,
        )
        records = tuple(data["suites"])
        bank = tuple(
            tuple(tuple(line) for line in record["openings"])
            for record in records
        )
    elif generated:
        _validate_bank_data(records, bank)
        _write_cache(data)
    _BANK_RECORDS = records
    _PROMOTION_HOLDOUT_BANK = bank


def promotion_holdout_suite_for_eval(
    eval_index: int,
    *,
    strict: bool = False,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    """Return a bank suite for a nonnegative persisted promotion counter.

    Legacy callers keep the historical rotation. Promotion training passes
    ``strict=True`` so a stale/reset counter cannot silently reuse an earlier
    confirmation suite after the bank is exhausted.
    """
    if isinstance(eval_index, bool) or not isinstance(eval_index, int):
        raise TypeError(f"eval_index must be an int, got {type(eval_index).__name__}")
    if eval_index < 0:
        raise ValueError(f"eval_index must be nonnegative, got {eval_index}")
    _ensure_bank()
    assert _BANK_RECORDS is not None and _PROMOTION_HOLDOUT_BANK is not None
    if strict and eval_index >= len(_PROMOTION_HOLDOUT_BANK):
        raise IndexError(
            f"promotion confirmation suite index {eval_index} exceeds bank capacity "
            f"{len(_PROMOTION_HOLDOUT_BANK)}"
        )
    suite_index = eval_index % len(_PROMOTION_HOLDOUT_BANK)
    return _BANK_RECORDS[suite_index]["suite_id"], _PROMOTION_HOLDOUT_BANK[suite_index]


def promotion_holdout_suite_capacity() -> int:
    """Return the number of validated unique suites in the local bank."""
    _ensure_bank()
    assert _PROMOTION_HOLDOUT_BANK is not None
    return len(_PROMOTION_HOLDOUT_BANK)


def opening_model_root_signature(
    opening_line: tuple[str, ...] | list[str], *, history_positions: int = 4
) -> str:
    """Hash the exact encoded root input after an opening prefix.

    The signature includes the four most recent pre-move history boards and
    the current board, encoded from the current side-to-move perspective in
    the same order as RL evaluation. Move counters and en-passant metadata are
    included through the encoder planes.
    """
    import hashlib

    import numpy as np

    from src.models.data.se_cnn_v9.helpers import (
        board_to_tensor,
        board_to_tensor_pair,
    )

    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")
    board = chess.new_board()
    board_history = []
    for uci in tuple(opening_line or ()):
        move = chess.move_from_uci(uci)
        if move is None or move not in chess.legal_moves(board):
            raise ValueError(f"Illegal UCI move {uci!r} in opening root signature")
        board_history.append(board_to_tensor_pair(board))
        chess.apply_move(board, move)

    use_black_pov = board.turn == chess.BLACK
    selected_history = [
        encoded[1] if use_black_pov else encoded[0]
        for encoded in board_history[-history_positions:]
    ] if history_positions else []
    pad_count = max(0, history_positions - len(selected_history))
    if pad_count:
        selected_history = [
            np.zeros((16, 8, 8), dtype=np.float32) for _ in range(pad_count)
        ] + selected_history
    selected_history.append(board_to_tensor(board))
    root_input = np.concatenate(selected_history, axis=0)
    return hashlib.sha256(root_input.tobytes(order="C")).hexdigest()


def validate_promotion_holdout_bank() -> bool:
    """Load the ignored cache or build and validate it from the local PGN."""
    _ensure_bank()
    return True


def _holdout_ledger_connect(db_path=None):
    path = Path(db_path) if db_path is not None else _LEDGER_DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    connection.execute("PRAGMA busy_timeout=30000")
    deadline = time.monotonic() + 30.0
    delay = 0.005
    while True:
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                connection.close()
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.1)
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS used_suites (
            suite_id TEXT PRIMARY KEY, suite_index INTEGER NOT NULL,
            token TEXT NOT NULL, state TEXT NOT NULL, provenance TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS used_roots (
            root_signature TEXT PRIMARY KEY, suite_id TEXT NOT NULL,
            provenance TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reservations (
            token TEXT PRIMARY KEY, run_identity TEXT NOT NULL,
            attempt_index INTEGER NOT NULL, mode TEXT NOT NULL,
            state TEXT NOT NULL, detail_json TEXT NOT NULL
        );
        """
    )
    return connection


def _historical_suite_exposures() -> tuple[list[dict], list[dict]]:
    """Return only suite identities with persisted evidence games were started."""
    _ensure_bank()
    assert _BANK_RECORDS is not None and _PROMOTION_HOLDOUT_BANK is not None
    known: dict[str, tuple[tuple[str, ...], ...]] = {
        str(record["suite_id"]): lines
        for record, lines in zip(_BANK_RECORDS, _PROMOTION_HOLDOUT_BANK)
    }
    roots = (_CACHE_PATH.parent, _CACHE_PATH.parents[1] / "experiments")
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("promotion_holdout_bank.archive-*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for record in data.get("suites", []):
                    if record.get("suite_id") and record.get("openings"):
                        known[str(record["suite_id"])] = tuple(
                            tuple(line) for line in record["openings"]
                        )
            except (OSError, ValueError, TypeError):
                continue

    exposures: dict[str, dict] = {}

    def record(suite_id, path, games=0, consumed=False):
        suite_id = str(suite_id or "")
        if suite_id not in known:
            return
        entry = exposures.setdefault(suite_id, {
            "suite_id": suite_id, "openings": known[suite_id], "provenance": []
        })
        proof = {"path": path, "games": int(games or 0), "suite_consumed": bool(consumed)}
        if proof not in entry["provenance"]:
            entry["provenance"].append(proof)

    def walk(value, path, inherited_suite_ids=()):
        if isinstance(value, dict):
            current_ids = list(inherited_suite_ids)
            if value.get("suite_id"):
                current_ids.append(str(value["suite_id"]))
            if isinstance(value.get("suite_ids"), (tuple, list)):
                current_ids.extend(str(item) for item in value["suite_ids"])
            games = value.get("games", value.get("confirmation_games", 0))
            try:
                started = int(games or 0) > 0
            except (TypeError, ValueError):
                started = False
            consumed = value.get("suite_consumed") is True
            if started or consumed:
                for suite_id in current_ids:
                    record(suite_id, path, games, consumed)
            for child in value.values():
                walk(child, path, tuple(dict.fromkeys(current_ids)))
        elif isinstance(value, list):
            for child in value:
                walk(child, path, inherited_suite_ids)

    reference_rows = _known_suite_usage(tuple(known))
    candidate_paths = sorted({
        str(reference["path"])
        for row in reference_rows
        for reference in row["references"]
        if Path(str(reference["path"])).suffix.lower() in {".json", ".csv"}
    })
    for relative_path in candidate_paths:
        path = _CACHE_PATH.parents[2] / Path(relative_path)
        if not path.is_file():
            continue
        rel = Path(relative_path).as_posix()
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                walk(payload, rel)
                # The E5 arena schema persists the suite identity in `protocol`
                # and actual per-arm game totals beside it. Join only this known
                # structured result shape, never its prior-suite declarations.
                if isinstance(payload, dict):
                    protocol = payload.get("protocol")
                    arms = payload.get("arms")
                    if isinstance(protocol, dict) and protocol.get("suite_id") and isinstance(arms, dict):
                        arm_games = 0
                        for arm in arms.values():
                            if isinstance(arm, dict):
                                try:
                                    arm_games += max(0, int(arm.get("games", 0) or 0))
                                except (TypeError, ValueError):
                                    pass
                        if arm_games > 0:
                            record(protocol["suite_id"], rel, arm_games, False)
            else:
                import csv

                with path.open("r", encoding="utf-8-sig", newline="") as source:
                    for row in csv.DictReader(source):
                        for cell in row.values():
                            if isinstance(cell, str) and cell.lstrip().startswith(("{", "[")):
                                try:
                                    walk(json.loads(cell), rel)
                                except (ValueError, TypeError):
                                    pass
                        games = row.get("games") or row.get("confirmation_games") or 0
                        try:
                            started = int(games or 0) > 0
                        except (TypeError, ValueError):
                            started = False
                        consumed = str(row.get("suite_consumed", "")).lower() == "true"
                        suite_id = row.get("suite_id") or row.get("promotion_confirmation_suite_id")
                        if suite_id and (started or consumed):
                            record(suite_id, rel, games, consumed)
        except (OSError, ValueError, TypeError):
            continue
    return list(exposures.values()), reference_rows


def _holdout_bank_identity() -> str:
    _ensure_bank()
    assert _BANK_RECORDS is not None
    payload = json.dumps(
        [(record["suite_id"], record["sha256"]) for record in _BANK_RECORDS],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _suite_root_signatures(openings):
    key = (_suite_digest(tuple(tuple(line) for line in openings)), 4)
    cached = _ROOT_SIGNATURE_CACHE.get(key)
    if cached is None:
        cached = tuple(opening_model_root_signature(line) for line in openings)
        _ROOT_SIGNATURE_CACHE[key] = cached
    return cached


def promotion_holdout_ledger_status(
    *, db_path=None, reserved_indices=(), preferred_start_index=0
) -> dict:
    """Seed structured historical exposure and return truly available capacity."""
    _ensure_bank()
    assert _BANK_RECORDS is not None and _PROMOTION_HOLDOUT_BANK is not None
    exposures, lineage_refs = _historical_suite_exposures()
    exposure_ids = {item["suite_id"] for item in exposures}
    current_suite_ids = {record["suite_id"] for record in _BANK_RECORDS}
    unverified = [
        {"suite_id": item["suite_id"], "references": item["references"]}
        for item in lineage_refs
        if item["suite_id"] in current_suite_ids
        and item["suite_id"] not in exposure_ids
    ]
    connection = _holdout_ledger_connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for exposure in exposures:
            provenance = json.dumps(exposure["provenance"], sort_keys=True)
            connection.execute(
                "INSERT OR IGNORE INTO used_suites VALUES (?, ?, ?, ?, ?)",
                (exposure["suite_id"], -1, "historical", "burned", provenance),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO used_roots VALUES (?, ?, ?)",
                [(sig, exposure["suite_id"], provenance)
                 for sig in _suite_root_signatures(exposure["openings"])],
            )
        reserved = {int(index) for index in reserved_indices}
        occupied_ids = {row[0] for row in connection.execute("SELECT suite_id FROM used_suites")}
        occupied_roots = {row[0] for row in connection.execute("SELECT root_signature FROM used_roots")}
        capacity = len(_BANK_RECORDS)
        start_index = int(preferred_start_index)
        if not 0 <= start_index < capacity:
            raise ValueError(f"preferred holdout start index {start_index} is outside bank capacity {capacity}")
        order = tuple(range(start_index, capacity)) + tuple(range(0, start_index))
        available = []
        for index in order:
            record, openings = _BANK_RECORDS[index], _PROMOTION_HOLDOUT_BANK[index]
            if index in reserved or record["suite_id"] in occupied_ids:
                continue
            if occupied_roots.intersection(_suite_root_signatures(openings)):
                continue
            available.append(index)
        connection.commit()
        return {
            "capacity": len(_BANK_RECORDS),
            "remaining": len(available),
            "available_indices": available,
            "historical_suite_count": len(exposures),
            "unverified_lineage": unverified,
            "bank_identity": _holdout_bank_identity(),
            "ledger_path": str(Path(db_path) if db_path is not None else _LEDGER_DEFAULT_PATH),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reserve_promotion_holdout_suites(
    *, run_identity, attempt_index, mode, count, reserved_indices=(),
    preferred_start_index=0, db_path=None
) -> dict:
    """Atomically mark suites and all their model roots burned before evaluation."""
    count = int(count)
    if count < 1:
        raise ValueError("suite reservation count must be positive")
    status = promotion_holdout_ledger_status(
        db_path=db_path, reserved_indices=reserved_indices,
        preferred_start_index=preferred_start_index,
    )
    if status["remaining"] < count:
        raise RuntimeError(
            "promotion confirmation holdout exhausted: "
            f"need {count} suite(s), only {status['remaining']} remain"
        )
    _ensure_bank()
    assert _BANK_RECORDS is not None and _PROMOTION_HOLDOUT_BANK is not None
    connection = _holdout_ledger_connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        occupied_ids = {row[0] for row in connection.execute("SELECT suite_id FROM used_suites")}
        occupied_roots = {row[0] for row in connection.execute("SELECT root_signature FROM used_roots")}
        chosen = []
        reserved = {int(index) for index in reserved_indices}
        for index in status["available_indices"]:
            record, openings = _BANK_RECORDS[index], _PROMOTION_HOLDOUT_BANK[index]
            suite_id = str(record["suite_id"])
            if index in reserved or suite_id in occupied_ids:
                continue
            signatures = _suite_root_signatures(openings)
            if occupied_roots.intersection(signatures):
                continue
            chosen.append((index, suite_id, openings, signatures))
            occupied_roots.update(signatures)
            occupied_ids.add(suite_id)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise RuntimeError(
                f"promotion confirmation holdout exhausted during reservation: need {count}, have {len(chosen)}"
            )
        bank_id = status["bank_identity"]
        token = hashlib.sha256(
            f"{run_identity}:{attempt_index}:{mode}:{bank_id}:".encode("utf-8") + os.urandom(16)
        ).hexdigest()
        result = []
        for index, suite_id, openings, signatures in chosen:
            provenance = json.dumps({"kind": "atomic_reservation", "token": token}, sort_keys=True)
            connection.execute(
                "INSERT INTO used_suites VALUES (?, ?, ?, ?, ?)",
                (suite_id, index, token, "reserved_burned", provenance),
            )
            connection.executemany(
                "INSERT INTO used_roots VALUES (?, ?, ?)",
                [(sig, suite_id, provenance) for sig in signatures],
            )
            result.append({"suite_index": index, "suite_id": suite_id, "openings": openings})
        connection.execute(
            "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)",
            (token, str(run_identity), int(attempt_index), str(mode), "reserved_burned",
             json.dumps({
                 "bank_identity": bank_id,
                 "suites": [{key: item[key] for key in ("suite_index", "suite_id")} for item in result],
             }, sort_keys=True)),
        )
        connection.commit()
        return {"token": token, "suites": result, "bank_identity": bank_id}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def finish_promotion_holdout_reservation(token, *, state, detail=None, db_path=None):
    """Persist outcome; burned suites are intentionally never released."""
    connection = _holdout_ledger_connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            "UPDATE reservations SET state=?, detail_json=? WHERE token=?",
            (str(state), json.dumps(detail or {}, sort_keys=True), str(token)),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown holdout reservation {token}")
        connection.execute("UPDATE used_suites SET state=? WHERE token=?", (str(state), str(token)))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
