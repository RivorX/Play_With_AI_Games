"""List chess model checkpoints with key metadata.

Usage:
  python chess/scripts/list_models.py

This script is intentionally argument-free.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add utils to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from src.models.catalog import (
    load_checkpoint_metadata,
    print_model_table,
    sort_entries_by_folder_and_version,
)


def main() -> None:
    if len(sys.argv) > 1:
        print("This script does not accept arguments.")
        print("Run: python chess/scripts/list_models.py")
        sys.exit(2)

    script_dir = Path(__file__).parent
    chess_dir = script_dir.parent
    models_dir = chess_dir / "models"

    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        sys.exit(1)

    paths = sorted(models_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        print(f"No .pt files found under: {models_dir}")
        sys.exit(0)

    entries = [load_checkpoint_metadata(path, models_dir) for path in paths]
    entries = sort_entries_by_folder_and_version(entries)
    print_model_table(
        entries,
        title="Model Checkpoints",
        show_folder=True,
        show_version=True,
        show_modified=True,
        show_swa=True,
        show_opt=True,
        group_by_folder=True,
    )


if __name__ == "__main__":
    main()
