"""Build the immutable v9 self-play opening pool from a local PGN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CHESS_ROOT = Path(__file__).resolve().parents[1]
if str(CHESS_ROOT) not in sys.path:
    sys.path.insert(0, str(CHESS_ROOT))

from src.selfplay.opening_pool import (  # noqa: E402
    DEFAULT_PGN_PATH,
    DEFAULT_POOL_PATH,
    build_opening_pool_data,
    load_opening_pool,
    resolve_repository_path,
    write_opening_pool_immutable,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_POOL_PATH),
        help="new immutable JSON artifact path (relative paths use the repo root)",
    )
    parser.add_argument(
        "--pgn",
        default=str(DEFAULT_PGN_PATH),
        help="source PGN path (relative paths use the repo root)",
    )
    parser.add_argument(
        "--history-positions",
        type=int,
        default=4,
        help="v9 history frames represented in each protected-root signature",
    )
    args = parser.parse_args(argv)

    output_path = resolve_repository_path(args.output)
    source_path = resolve_repository_path(args.pgn)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable opening pool: {output_path}")

    data = build_opening_pool_data(
        source_path,
        history_positions=args.history_positions,
    )
    written_path = write_opening_pool_immutable(data, output_path)
    lines = load_opening_pool(
        written_path,
        history_positions=args.history_positions,
        expected_sha256=data["pool_sha256"],
    )
    print(f"opening_pool_path={written_path}")
    print(f"pool_sha256={data['pool_sha256']}")
    print(f"families={data['family_count']} lines_per_family={data['lines_per_family']} total_lines={len(lines)}")
    print(f"source={data['source']['path']} bytes={data['source']['bytes']} sha256={data['source']['sha256']}")
    print(
        "excluded_roots="
        f"selector:{data['exclusion']['selector_root_count']} "
        f"promotion:{data['exclusion']['promotion_root_count']} "
        f"candidate_lines:{data['exclusion']['excluded_candidate_line_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

