"""
Download local Syzygy WDL tablebases into chess/data/... using config defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_project_config

from src.common.syzygy import (
    describe_syzygy_status,
    ensure_syzygy_tables,
    get_project_chess_dir,
)


def _load_config(chess_dir: Path) -> dict:
    return load_project_config()


def main() -> None:
    chess_dir = get_project_chess_dir()
    config = _load_config(chess_dir)

    parser = argparse.ArgumentParser(description="Download Syzygy WDL tablebases.")
    parser.add_argument(
        "--preset",
        choices=("wdl_345", "wdl_345_6"),
        default=None,
        help="Preset to download. Default comes from config/default.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Only print current Syzygy status and exit.",
    )
    args = parser.parse_args()

    if args.preset:
        config.setdefault("reinforcement_learning", {})["syzygy_auto_download_preset"] = args.preset

    if args.status:
        status = describe_syzygy_status(config, chess_dir=chess_dir)
        print(f"Syzygy paths: {', '.join(str(p) for p in status['paths'])}")
        print(f"WDL files found: {status['wdl_files']}")
        return

    info = ensure_syzygy_tables(config, chess_dir=chess_dir, force=bool(args.force))
    destination = info.get("destination")
    if destination is not None:
        print(f"Syzygy destination: {destination}")
    print(f"Preset: {info.get('preset', 'n/a')}")
    print(f"Downloaded files: {int(info.get('downloaded_files', 0))}")
    print(f"Downloaded size: {float(info.get('downloaded_bytes', 0)) / 1024 / 1024:.1f} MB")
    print(f"WDL files now available: {int(info.get('total_wdl_files_now', 0))}")


if __name__ == "__main__":
    main()
