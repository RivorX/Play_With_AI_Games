"""Executable UCI adapter; implementation lives in ``src.ui.uci_engine``."""

from pathlib import Path
import sys


CHESS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CHESS_DIR))

from src.ui.uci_engine import main


if __name__ == "__main__":
    main()
