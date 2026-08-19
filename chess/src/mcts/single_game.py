"""Single-game API over the canonical batched MCTS engine."""

from src.mcts.result import SearchResult


class SingleGameMCTS:
    """
    Single-game interface over MultiGameBatchMCTS.

    Play, UCI and Elo use this focused API while sharing the production
    multi-root search implementation with RL.
    """

    def __init__(self, model, config, device):
        from src.mcts.search import MultiGameBatchMCTS

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
            '_native_tree_key': id(self),
        }
        search_results = self._multi.search_results_many(
            [game_state],
            num_simulations=num_simulations,
            add_root_noise=add_root_noise,
        )
        self.root = game_state.get('root')
        self._root_synced = bool(game_state.get('_root_synced', False))
        return search_results[0] if search_results else SearchResult.from_legacy_result({})

    def advance_root(self, move):
        if self.root is None:
            return
        child = self.root.get_child_for_move(move)
        if child is None:
            self.root = None
            self._root_synced = False
            return
        _ = child.board
        self.root = child.detach_as_root()
        self._root_synced = True

    def update_history(self, board):
        self.board_history.append(self._multi._encode_history_entry(board))
        max_history = int(self.config.get('model', {}).get('history_positions', 0)) + 10
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]

    def reset_tree(self):
        self._multi.release_native_tree(id(self))
        self.root = None
        self._root_synced = False
        self.board_history = []
