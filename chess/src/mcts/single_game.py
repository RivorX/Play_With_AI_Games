"""Single-game API over the canonical batched MCTS engine."""

import math


class SingleGameMCTS:
    """
    Single-game interface over MultiGameBatchMCTS.

    Play, UCI and Elo use this focused API while sharing the production
    multi-root search implementation with RL.
    """

    def __init__(self, model, config, device, *, single_root_batch_cap=None):
        from src.mcts.search import MultiGameBatchMCTS

        if single_root_batch_cap is not None:
            if isinstance(single_root_batch_cap, bool):
                raise ValueError("single_root_batch_cap must be a non-negative integer")
            try:
                numeric_cap = float(single_root_batch_cap)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "single_root_batch_cap must be a non-negative integer"
                ) from exc
            if (
                not math.isfinite(numeric_cap)
                or numeric_cap < 0
                or not numeric_cap.is_integer()
            ):
                raise ValueError("single_root_batch_cap must be a non-negative integer")

            # Keep this override local to the single-game wrapper. In
            # particular, don't mutate the shared runtime config used by
            # multi-root RL, UCI, or Elo evaluation.
            config = dict(config)
            reinforcement_learning = dict(
                config.get("reinforcement_learning", {}) or {}
            )
            reinforcement_learning["mcts_single_root_batch_cap"] = int(numeric_cap)
            config["reinforcement_learning"] = reinforcement_learning

        self._multi = MultiGameBatchMCTS(model, config, device)
        # MultiGameBatchMCTS resolves architecture and encoder settings from
        # this model's checkpoint spec. Keep history trimming on that same
        # per-model config when two architectures share a game.
        self.config = self._multi.config
        self.root = None
        self._root_synced = False
        self.board_history = []

    def search(self, board, num_simulations, add_root_noise=False):
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
        if len(search_results) != 1:
            raise RuntimeError(
                "SingleGameMCTS expected exactly one result from the canonical "
                f"batched search, received {len(search_results)}"
            )
        return search_results[0]

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

    def update_history(self, board, repetition_count=None):
        self.board_history.append(
            self._multi._encode_history_entry(
                board,
                repetition_count=repetition_count,
            )
        )
        max_history = int(self.config.get('model', {}).get('history_positions', 0)) + 10
        if len(self.board_history) > max_history:
            self.board_history = self.board_history[-max_history:]

    def reset_tree(self):
        self._multi.release_native_tree(id(self))
        self.root = None
        self._root_synced = False
        self.board_history = []
