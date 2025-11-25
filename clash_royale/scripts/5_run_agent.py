import sys
import os

# Dodaj katalog nadrzędny do ścieżki, aby widzieć moduł 'utils'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.agent import run_agent

if __name__ == "__main__":
    run_agent()
