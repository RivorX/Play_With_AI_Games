"""
Download local Syzygy WDL tablebases into chess/data/... using config defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from utils.shared.syzygy_manager import (
    describe_syzygy_status,
    ensure_syzygy_tables,
    get_project_chess_dir,
)


def _load_config(chess_dir: Path) -> dict:
    config_path = chess_dir / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj) or {}


def main() -> None:
    chess_dir = get_project_chess_dir()
    config = _load_config(chess_dir)

    parser = argparse.ArgumentParser(description="Download Syzygy WDL tablebases.")
    parser.add_argument(
        "--preset",
        choices=("wdl_345", "wdl_345_6"),
        default=None,
        help="Preset to download. Default comes from config.yaml.",
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
