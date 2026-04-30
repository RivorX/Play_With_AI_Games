"""Compatibility MCTS API for UI, UCI, Elo and single-game eval paths."""

import numpy as np


def select_move_by_visits(visit_counts, temperature=1.0):
    """Select a move from an MCTS visit-count map and return move plus probabilities/visits."""
    moves = list(visit_counts.keys())
    visits = np.fromiter(visit_counts.values(), dtype=np.float64, count=len(moves))

    if temperature == 0 or len(moves) == 1:
        best_idx = int(np.argmax(visits))
        return moves[best_idx], visits

    visits_temp = visits ** (1.0 / temperature)
    total = float(visits_temp.sum())
    if total <= 0 or not np.isfinite(total):
        probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
    else:
        probs = visits_temp / total
        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = float(probs.sum())
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float64)
        else:
            probs = probs / probs_sum

    idx = int(np.random.choice(len(moves), p=probs))
    return moves[idx], probs


class BatchMCTS:
    """
    Single-game compatibility wrapper over MultiGameBatchMCTS.

    Keeps the old API (`search`, `advance_root`, `update_history`, `reset_tree`)
    while using the same shared multi-game MCTS core.
    """

    def __init__(self, model, config, device):
        from src.batch_selfplay import MultiGameBatchMCTS

        self.config = config
        self._multi = MultiGameBatchMCTS(model, config, device)
        self.root = None
        self._root_synced = False
        self.board_history = []

    def search(self, board, num_simulations, temperature=1.0, add_root_noise=False):
        del temperature

        game_state = {
            'board': board,
            'root': self.root,
            '_root_synced': self._root_synced,
            'board_history': self.board_history,
        }
        visit_counts = self._multi.search_many(
            [game_state],
            num_simulations=num_simulations,
            add_root_noise=add_root_noise,
        )
        self.root = game_state.get('root')
        self._root_synced = bool(game_state.get('_root_synced', False))
        return visit_counts[0] if visit_counts else {}

    def advance_root(self, move):
        if self.root is None:
            return
        child = self.root.get_child_for_move(move)
        if child is None:
            self.root = None
            self._root_synced = False
            return
        _ = child.board
        child.parent = None
        child.parent_edge_index = -1
        self.root = child
        self._root_synced = True

    def update_history(self, board):
        self.board_history.append(self._multi._encode_history_entry(board))
        max_history = int(self.config.get('model', {}).get('history_positions', 0)) + 10
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]

    def reset_tree(self):
        self.root = None
        self._root_synced = False
        self.board_history = []


MCTS = BatchMCTS
