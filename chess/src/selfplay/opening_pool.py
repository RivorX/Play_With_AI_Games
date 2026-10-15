"""Immutable, preflight-validated self-play opening pools.

Pool construction and full selector/holdout exclusion checks are CPU-only and
run before training creates a model. Self-play workers then validate the pool
file, its digest, schema, legality, and balance once per process.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from src.game import backend as chess


POOL_SCHEMA_VERSION = 2
POOL_FAMILY_COUNT = 24
POOL_LINES_PER_FAMILY = 32
POOL_LINE_PLIES = 8
POOL_ROOT_DEPTHS = (6, 7, 8)
DEFAULT_POOL_PATH = Path(__file__).resolve().parents[2] / "logs" / "selfplay_opening_pool_v2.json"
DEFAULT_PGN_PATH = Path(__file__).resolve().parents[2] / "data" / "lichess_elite_2025-11.pgn"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_POOL_CACHE: dict[tuple, tuple[tuple[str, ...], ...]] = {}


def resolve_repository_path(path: str | os.PathLike) -> Path:
    """Resolve relative config and builder paths from the repository root."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _canonical_json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pool_content_sha256(data: dict) -> str:
    payload = {key: value for key, value in data.items() if key != "pool_sha256"}
    return _canonical_json_sha256(payload)


def _family_id(prefix_uci: tuple[str, str]) -> str:
    return hashlib.sha256(" ".join(prefix_uci).encode("ascii")).hexdigest()[:12]


def _line_root_identities(line: tuple[str, ...], history_positions: int) -> dict[int, tuple[str, int]]:
    """Return exact v9 encoded roots and backend position keys at plies 6-8."""
    from src.evaluation.promotion_holdout import opening_model_root_signature

    board = chess.new_board()
    result = {}
    for ply, uci in enumerate(line, start=1):
        move = chess.move_from_uci(uci)
        if move is None or move not in chess.legal_moves(board):
            raise ValueError(f"Illegal UCI move {uci!r} at ply {ply} in line {line!r}")
        chess.apply_move(board, move)
        if ply in POOL_ROOT_DEPTHS:
            prefix = tuple(line[:ply])
            result[ply] = (
                opening_model_root_signature(
                    prefix,
                    history_positions=history_positions,
                ),
                int(chess.position_key(board)),
            )
    return result


def _root_count(roots: dict[int, set]) -> int:
    return sum(len(values) for values in roots.values())


def _root_digest(roots: dict[int, set]) -> str:
    flattened = [
        (depth, value)
        for depth in sorted(roots)
        for value in sorted(roots[depth])
    ]
    return _canonical_json_sha256({"roots": flattened})


def _identity_digest(identities: set) -> str:
    return _canonical_json_sha256({"roots": sorted(identities)})


def _terminal_root_identity(line: tuple[str, ...], history_positions: int) -> tuple[str, int]:
    from src.evaluation.promotion_holdout import opening_model_root_signature

    board = chess.new_board()
    for ply, uci in enumerate(line, start=1):
        move = chess.move_from_uci(uci)
        if move is None or move not in chess.legal_moves(board):
            raise ValueError(f"Illegal UCI move {uci!r} at ply {ply} in line {line!r}")
        chess.apply_move(board, move)
    return (
        opening_model_root_signature(line, history_positions=history_positions),
        int(chess.position_key(board)),
    )


def _resolve_selector_terminal_lines(config=None) -> tuple[tuple[str, ...], ...]:
    from src.training.rl.trainer import (
        _resolve_eval_fixed_openings_max_plies,
        _resolve_eval_opening_lines,
    )

    if config is None:
        config = {
            "reinforcement_learning": {
                "eval_opening_lines": None,
                "eval_fixed_openings_max_plies": 8,
            }
        }
    lines = _resolve_eval_opening_lines(config)
    max_plies = _resolve_eval_fixed_openings_max_plies(config)
    if max_plies <= 0:
        return ()
    return tuple(
        tuple(line[: min(len(line), max_plies)])
        for line in lines
        if line[: min(len(line), max_plies)]
    )


def _collect_exclusion_context(history_positions: int, selector_config=None) -> dict:
    """Build terminal-root and 6–8-ply prefix exclusions for protected suites."""
    from src.evaluation.promotion_holdout import (
        promotion_holdout_suite_capacity,
        promotion_holdout_suite_for_eval,
        validate_promotion_holdout_bank,
    )
    from src.mcts.search import _EVAL_OPENING_SUITE_ID

    if validate_promotion_holdout_bank() is not True:
        raise RuntimeError("Promotion holdout bank did not validate")
    holdout_capacity = int(promotion_holdout_suite_capacity())
    if holdout_capacity != 64:
        raise ValueError(f"Expected all 64 promotion holdout suites, got {holdout_capacity}")

    selector_lines = _resolve_selector_terminal_lines(selector_config)
    selector_terminal_signatures = set()
    selector_terminal_keys = set()
    selector_prefix_signatures: dict[int, set[str]] = defaultdict(set)
    selector_prefix_keys: dict[int, set[int]] = defaultdict(set)
    for line in selector_lines:
        signature, position_key = _terminal_root_identity(line, history_positions)
        selector_terminal_signatures.add(signature)
        selector_terminal_keys.add(position_key)
        identities = _line_root_identities(line, history_positions)
        for depth, (signature, position_key) in identities.items():
            selector_prefix_signatures[depth].add(signature)
            selector_prefix_keys[depth].add(position_key)

    holdout_terminal_signatures = set()
    holdout_terminal_keys = set()
    holdout_prefix_signatures: dict[int, set[str]] = defaultdict(set)
    holdout_prefix_keys: dict[int, set[int]] = defaultdict(set)
    suite_manifest = []
    opening_count = 0
    for suite_index in range(holdout_capacity):
        suite_id, openings = promotion_holdout_suite_for_eval(suite_index, strict=True)
        suite_openings = tuple(tuple(line) for line in openings)
        suite_manifest.append(
            {
                "index": suite_index,
                "suite_id": str(suite_id),
                "openings_sha256": _canonical_json_sha256(
                    {"openings": [list(line) for line in suite_openings]}
                ),
            }
        )
        opening_count += len(suite_openings)
        for line in suite_openings:
            signature, position_key = _terminal_root_identity(line, history_positions)
            holdout_terminal_signatures.add(signature)
            holdout_terminal_keys.add(position_key)
            identities = _line_root_identities(line, history_positions)
            for depth, (signature, position_key) in identities.items():
                holdout_prefix_signatures[depth].add(signature)
                holdout_prefix_keys[depth].add(position_key)

    selector_prefix_root_count = _root_count(selector_prefix_signatures)
    selector_prefix_key_count = _root_count(selector_prefix_keys)
    promotion_prefix_root_count = _root_count(holdout_prefix_signatures)
    promotion_prefix_key_count = _root_count(holdout_prefix_keys)
    protected_signatures = set(selector_terminal_signatures)
    protected_signatures.update(holdout_terminal_signatures)
    protected_keys = set(selector_terminal_keys)
    protected_keys.update(holdout_terminal_keys)
    for depth in POOL_ROOT_DEPTHS:
        protected_signatures.update(selector_prefix_signatures[depth])
        protected_signatures.update(holdout_prefix_signatures[depth])
        protected_keys.update(selector_prefix_keys[depth])
        protected_keys.update(holdout_prefix_keys[depth])

    promotion_bank_sha256 = _canonical_json_sha256({"suites": suite_manifest})
    proof = {
        "selector_suite_id": str(_EVAL_OPENING_SUITE_ID),
        "selector_line_count": len(selector_lines),
        "selector_root_count": len(selector_terminal_signatures),
        "selector_root_sha256": _identity_digest(selector_terminal_signatures),
        "selector_position_key_count": len(selector_terminal_keys),
        "selector_position_key_sha256": _identity_digest(selector_terminal_keys),
        "selector_prefix_root_count": selector_prefix_root_count,
        "selector_prefix_root_sha256": _root_digest(selector_prefix_signatures),
        "selector_prefix_position_key_count": selector_prefix_key_count,
        "selector_prefix_position_key_sha256": _root_digest(selector_prefix_keys),
        "promotion_suite_count": holdout_capacity,
        "promotion_opening_count": opening_count,
        "promotion_bank_sha256": promotion_bank_sha256,
        "promotion_root_count": len(holdout_terminal_signatures),
        "promotion_root_sha256": _identity_digest(holdout_terminal_signatures),
        "promotion_position_key_count": len(holdout_terminal_keys),
        "promotion_position_key_sha256": _identity_digest(holdout_terminal_keys),
        "promotion_prefix_root_count": promotion_prefix_root_count,
        "promotion_prefix_root_sha256": _root_digest(holdout_prefix_signatures),
        "promotion_prefix_position_key_count": promotion_prefix_key_count,
        "promotion_prefix_position_key_sha256": _root_digest(holdout_prefix_keys),
        "signature_kind": "se_cnn_v9_encoded_history_plus_current",
        "position_key_kind": "src.game.backend.position_key",
        "root_depths": list(POOL_ROOT_DEPTHS),
    }
    return {
        "proof": proof,
        "signatures": protected_signatures,
        "position_keys": protected_keys,
    }


def _validate_exclusion_proof(data: dict, context: dict) -> None:
    recorded = data.get("exclusion")
    if not isinstance(recorded, dict):
        raise ValueError("Opening pool exclusion manifest must be an object")
    expected = context["proof"]
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(
                f"Opening pool exclusion manifest mismatch for {key}: "
                f"expected {value!r}, got {recorded.get(key)!r}"
            )
    rejected = recorded.get("excluded_candidate_line_count")
    if isinstance(rejected, bool) or not isinstance(rejected, int) or rejected < 0:
        raise ValueError("Opening pool excluded_candidate_line_count is invalid")

    for family in data["families"]:
        for line_value in family["openings"]:
            line = tuple(line_value)
            roots = _line_root_identities(line, int(data["history_positions"]))
            for depth, (signature, position_key) in roots.items():
                if signature in context["signatures"]:
                    raise ValueError(
                        f"Opening pool overlaps a protected encoded root at ply {depth}"
                    )
                if position_key in context["position_keys"]:
                    raise ValueError(
                        f"Opening pool overlaps a protected position key at ply {depth}"
                    )


def _validate_pool_data(
    data: dict,
    *,
    history_positions: int,
    expected_sha256: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(data, dict):
        raise ValueError("Opening pool JSON root must be an object")
    required_keys = {
        "schema_version",
        "source",
        "line_plies",
        "history_positions",
        "family_count",
        "lines_per_family",
        "families",
        "exclusion",
        "pool_sha256",
    }
    if set(data) != required_keys:
        missing = sorted(required_keys - set(data))
        extra = sorted(set(data) - required_keys)
        raise ValueError(f"Opening pool JSON keys mismatch; missing={missing}, extra={extra}")
    if data.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported opening pool schema: {data.get('schema_version')!r}")
    if data.get("line_plies") != POOL_LINE_PLIES:
        raise ValueError(f"Opening pool lines must contain exactly {POOL_LINE_PLIES} plies")
    if data.get("history_positions") != int(history_positions):
        raise ValueError(
            "Opening pool history_positions does not match the model config: "
            f"{data.get('history_positions')!r} != {int(history_positions)}"
        )
    if data.get("family_count") != POOL_FAMILY_COUNT:
        raise ValueError(f"Expected {POOL_FAMILY_COUNT} opening families")
    if data.get("lines_per_family") != POOL_LINES_PER_FAMILY:
        raise ValueError(f"Expected {POOL_LINES_PER_FAMILY} lines per opening family")

    source = data.get("source")
    if not isinstance(source, dict) or set(source) != {"path", "sha256", "bytes"}:
        raise ValueError("Opening pool source manifest has an invalid shape")
    if (
        not isinstance(source.get("path"), str)
        or not source["path"]
        or not isinstance(source.get("sha256"), str)
        or len(source["sha256"]) != 64
        or isinstance(source.get("bytes"), bool)
        or not isinstance(source.get("bytes"), int)
        or source["bytes"] <= 0
    ):
        raise ValueError("Opening pool source manifest contains invalid values")

    digest = data.get("pool_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Opening pool content SHA-256 is invalid")
    actual_digest = _pool_content_sha256(data)
    if digest != actual_digest:
        raise ValueError(
            f"Opening pool checksum mismatch: recorded {digest}, calculated {actual_digest}"
        )
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise ValueError(
            "Opening pool changed after preflight: "
            f"expected {expected_sha256}, got {digest}"
        )

    families = data.get("families")
    if not isinstance(families, list) or len(families) != POOL_FAMILY_COUNT:
        raise ValueError(f"Opening pool must contain exactly {POOL_FAMILY_COUNT} families")

    pool_lines = []
    seen_prefixes = set()
    seen_openings = set()
    seen_signatures = {depth: set() for depth in POOL_ROOT_DEPTHS}
    seen_position_keys = {depth: set() for depth in POOL_ROOT_DEPTHS}
    for family_index, family in enumerate(families):
        if not isinstance(family, dict) or set(family) != {
            "family_id",
            "prefix_uci",
            "openings",
        }:
            raise ValueError(f"Opening pool family {family_index} has an invalid shape")
        prefix_value = family.get("prefix_uci")
        if (
            not isinstance(prefix_value, list)
            or len(prefix_value) != 2
            or any(not isinstance(move, str) or not move for move in prefix_value)
        ):
            raise ValueError(f"Opening pool family {family_index} has an invalid prefix")
        prefix = tuple(prefix_value)
        if prefix in seen_prefixes:
            raise ValueError(f"Duplicate opening family prefix at family {family_index}")
        seen_prefixes.add(prefix)
        if family.get("family_id") != _family_id(prefix):
            raise ValueError(f"Opening pool family {family_index} has an invalid family_id")
        openings = family.get("openings")
        if not isinstance(openings, list) or len(openings) != POOL_LINES_PER_FAMILY:
            raise ValueError(
                f"Opening family {family_index} must contain {POOL_LINES_PER_FAMILY} lines"
            )
        for line_index, line_value in enumerate(openings):
            if (
                not isinstance(line_value, list)
                or len(line_value) != POOL_LINE_PLIES
                or any(not isinstance(move, str) or not move for move in line_value)
            ):
                raise ValueError(
                    f"Opening family {family_index} line {line_index} must have "
                    f"{POOL_LINE_PLIES} UCI moves"
                )
            line = tuple(line_value)
            if line[:2] != prefix:
                raise ValueError(
                    f"Opening family {family_index} line {line_index} has a mismatched prefix"
                )
            if line in seen_openings:
                raise ValueError(f"Duplicate opening line at family {family_index}, line {line_index}")
            seen_openings.add(line)
            identities = _line_root_identities(line, int(history_positions))
            for depth, (signature, position_key) in identities.items():
                if signature in seen_signatures[depth]:
                    raise ValueError(f"Duplicate encoded opening root at ply {depth}")
                if position_key in seen_position_keys[depth]:
                    raise ValueError(f"Duplicate opening position key at ply {depth}")
                seen_signatures[depth].add(signature)
                seen_position_keys[depth].add(position_key)
            pool_lines.append(line)

    if len(pool_lines) != POOL_FAMILY_COUNT * POOL_LINES_PER_FAMILY:
        raise ValueError("Opening pool total line count is invalid")
    exclusion = data.get("exclusion")
    if not isinstance(exclusion, dict):
        raise ValueError("Opening pool exclusion manifest must be an object")
    pool_sha = exclusion.get("promotion_bank_sha256")
    if not isinstance(pool_sha, str) or len(pool_sha) != 64:
        raise ValueError("Opening pool promotion-bank manifest digest is invalid")
    return tuple(pool_lines)


def _read_pool(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configured self-play opening pool is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Configured self-play opening pool is unreadable or corrupt: {path}: {exc}") from exc
    return data


def preflight_opening_pool(
    path: str | os.PathLike,
    *,
    history_positions: int,
    config=None,
) -> str:
    """Strictly validate the pool and current selector plus all64 holdout roots."""
    resolved = resolve_repository_path(path)
    data = _read_pool(resolved)
    _validate_pool_data(data, history_positions=int(history_positions))
    context = _collect_exclusion_context(
        int(history_positions),
        selector_config=config,
    )
    _validate_exclusion_proof(data, context)
    return str(data["pool_sha256"])


def load_opening_pool(
    path: str | os.PathLike,
    *,
    history_positions: int,
    expected_sha256: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Load the preflighted pool once in a self-play worker process."""
    resolved = resolve_repository_path(path)
    stat = resolved.stat() if resolved.exists() else None
    cache_key = (
        str(resolved),
        int(history_positions),
        str(expected_sha256 or ""),
        stat.st_size if stat is not None else -1,
        stat.st_mtime_ns if stat is not None else -1,
    )
    cached = _POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    data = _read_pool(resolved)
    lines = _validate_pool_data(
        data,
        history_positions=int(history_positions),
        expected_sha256=expected_sha256,
    )
    _POOL_CACHE.clear()
    _POOL_CACHE[cache_key] = lines
    return lines


def _write_pool_immutable(data: dict, output_path: Path) -> None:
    """Publish a completed cache atomically without ever replacing an artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(data, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, output_path)
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite immutable opening pool: {output_path}"
            ) from None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _source_manifest(source_path: Path) -> dict:
    digest = hashlib.sha256()
    byte_count = 0
    with source_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    try:
        display_path = str(source_path.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        display_path = str(source_path)
    return {"path": display_path, "sha256": digest.hexdigest(), "bytes": byte_count}


def _opening_visitor(pgn):
    class Visitor(pgn.BaseVisitor):
        def __init__(self):
            self.moves = []

        def begin_variation(self):
            return pgn.SKIP

        def begin_parse_san(self, board, san):
            if len(self.moves) >= POOL_LINE_PLIES:
                return pgn.SKIP

        def visit_move(self, board, move):
            self.moves.append(move)

        def result(self):
            return tuple(self.moves)

    return Visitor


def _scan_pgn(source_path: Path, *, history_positions: int, context: dict) -> tuple[list, int]:
    import chess.pgn as pgn

    families: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    used_signatures = {depth: set() for depth in POOL_ROOT_DEPTHS}
    used_position_keys = {depth: set() for depth in POOL_ROOT_DEPTHS}
    excluded_candidate_line_count = 0
    visitor = _opening_visitor(pgn)

    with source_path.open("r", encoding="utf-8", errors="replace") as pgn_stream:
        while True:
            game = pgn.read_game(pgn_stream, Visitor=visitor)
            if game is None:
                break
            moves = game
            if len(moves) != POOL_LINE_PLIES:
                continue
            line = tuple(move.uci() for move in moves)
            family_prefix = (line[0], line[1])
            family_lines = families.setdefault(family_prefix, [])
            if len(family_lines) >= POOL_LINES_PER_FAMILY:
                continue

            roots = _line_root_identities(line, history_positions)
            overlaps_protected = any(
                signature in context["signatures"]
                or position_key in context["position_keys"]
                for depth, (signature, position_key) in roots.items()
            )
            if overlaps_protected:
                excluded_candidate_line_count += 1
                continue
            overlaps_pool = any(
                signature in used_signatures[depth]
                or position_key in used_position_keys[depth]
                for depth, (signature, position_key) in roots.items()
            )
            if overlaps_pool:
                continue

            family_lines.append(line)
            for depth, (signature, position_key) in roots.items():
                used_signatures[depth].add(signature)
                used_position_keys[depth].add(position_key)

            ready_families = sum(
                len(openings) == POOL_LINES_PER_FAMILY
                for openings in families.values()
            )
            if ready_families >= POOL_FAMILY_COUNT:
                break

    selected_prefixes = sorted(
        prefix
        for prefix, openings in families.items()
        if len(openings) == POOL_LINES_PER_FAMILY
    )
    if len(selected_prefixes) < POOL_FAMILY_COUNT:
        raise ValueError(
            f"PGN supplied only {len(selected_prefixes)} of {POOL_FAMILY_COUNT} "
            f"families with {POOL_LINES_PER_FAMILY} unique non-overlapping roots each"
        )
    return [
        (prefix, tuple(families[prefix]))
        for prefix in selected_prefixes[:POOL_FAMILY_COUNT]
    ], excluded_candidate_line_count


def build_opening_pool_data(
    pgn_path: str | os.PathLike,
    *,
    history_positions: int = 4,
) -> dict:
    """Build the deterministic balanced pool from a local elite PGN."""
    source_path = resolve_repository_path(pgn_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Self-play opening source PGN is missing: {source_path}")
    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")

    context = _collect_exclusion_context(history_positions)
    selected_families, excluded_line_count = _scan_pgn(
        source_path,
        history_positions=history_positions,
        context=context,
    )
    families = [
        {
            "family_id": _family_id(prefix),
            "prefix_uci": list(prefix),
            "openings": [list(line) for line in openings],
        }
        for prefix, openings in selected_families
    ]
    exclusion = dict(context["proof"])
    exclusion["excluded_candidate_line_count"] = int(excluded_line_count)
    data = {
        "schema_version": POOL_SCHEMA_VERSION,
        "source": _source_manifest(source_path),
        "line_plies": POOL_LINE_PLIES,
        "history_positions": history_positions,
        "family_count": POOL_FAMILY_COUNT,
        "lines_per_family": POOL_LINES_PER_FAMILY,
        "families": families,
        "exclusion": exclusion,
    }
    data["pool_sha256"] = _pool_content_sha256(data)
    _validate_pool_data(data, history_positions=history_positions)
    _validate_exclusion_proof(data, context)
    return data


def write_opening_pool_immutable(data: dict, output_path: str | os.PathLike) -> Path:
    """Write a new immutable pool artifact, refusing every existing target."""
    resolved = resolve_repository_path(output_path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite immutable opening pool: {resolved}")
    _write_pool_immutable(data, resolved)
    return resolved

