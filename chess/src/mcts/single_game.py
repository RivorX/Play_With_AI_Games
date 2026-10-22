"""Single-game API over the canonical batched MCTS engine."""

import math


class SingleGameMCTS:
    """
    Single-game interface over MultiGameBatchMCTS.

    Play, UCI and Elo use this focused API while sharing the production
    multi-root search implementation with RL.
    """

    def __init__(
        self, model, config, device, *, single_root_batch_cap=None,
        local_cuda_graph_forward=None,
    ):
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

        self._multi = MultiGameBatchMCTS(
            model, config, device,
            local_cuda_graph_forward=local_cuda_graph_forward,
        )
        # MultiGameBatchMCTS resolves architecture and encoder settings from
        # this model's checkpoint spec. Keep history trimming on that same
        # per-model config when two architectures share a game.
        self.config = self._multi.config
        from src.mcts.session import SearchSession

        self._session = SearchSession(self._multi, tree_key=id(self))
        self.board_history = []

    @property
    def root(self):
        return self._session.root

    @root.setter
    def root(self, value):
        self._session.root = value

    @property
    def _root_synced(self):
        return self._session.root_synced

    @_root_synced.setter
    def _root_synced(self, value):
        self._session._root_synced = bool(value)

    @staticmethod
    def _board_repetition_context(board, position_counts, repetition_known):
        from src.game import backend as chess
        from src.mcts.search import _position_counts_from_board

        if position_counts is not None:
            # A caller-supplied map is complete only when its owner says so.
            return position_counts, bool(repetition_known)
        counts = _position_counts_from_board(board)
        if repetition_known is None:
            repetition_known = bool(
                int(getattr(board, 'halfmove_clock', 0)) == 0
                or bool(chess.move_history(board))
            )
        return counts, bool(repetition_known)

    def search(
        self,
        board,
        num_simulations=None,
        add_root_noise=False,
        *,
        position_counts=None,
        repetition_known=None,
        budget_mode=None,
        time_limit_seconds=None,
        include_diagnostics=True,
    ):
        if num_simulations is None:
            num_simulations = int(
                (self.config.get("reinforcement_learning", {}) or {}).get(
                    "mcts_simulations", 192
                )
            )
        position_counts, repetition_known = self._board_repetition_context(
            board,
            position_counts,
            repetition_known,
        )
        game_state = self._session.make_state(
            board,
            board_history=self.board_history,
            position_counts=position_counts,
            repetition_known=repetition_known,
        )
        try:
            search_results = self._multi.search_results_many(
                [game_state],
                num_simulations=num_simulations,
                add_root_noise=add_root_noise,
                budget_mode=budget_mode,
                time_limit_seconds=time_limit_seconds,
                include_diagnostics=include_diagnostics,
            )
        except BaseException:
            # search_many deliberately clears roots if model generation changes
            # mid-search. Reflect that invalidation in the owning session too.
            self._session.commit(game_state)
            raise
        self._session.commit(game_state)
        if len(search_results) != 1:
            raise RuntimeError(
                "SingleGameMCTS expected exactly one result from the canonical "
                f"batched search, received {len(search_results)}"
            )
        return search_results[0]

    def advance_root(
        self,
        move,
        *,
        board=None,
        position_counts=None,
        repetition_known=None,
    ):
        """Advance after the caller recorded the pre-move history frame.

        ``board`` is the authoritative post-move board. If omitted, the
        searched child board is used. Repetition counts are reconstructed only
        from that board's move stack; a FEN alone never proves full history.
        """
        root = self.root
        child = root.get_child_for_move(move) if root is not None else None
        post_move_board = (
            board
            if board is not None
            else (child.board if child is not None else (root.board if root is not None else None))
        )
        if post_move_board is None:
            return self._session.advance(
                move,
                board=None,
                board_history=self.board_history,
                position_counts=position_counts,
                repetition_known=False,
            )
        position_counts, repetition_known = self._board_repetition_context(
            post_move_board,
            position_counts,
            repetition_known,
        )
        return self._session.advance(
            move,
            board=post_move_board,
            board_history=self.board_history,
            position_counts=position_counts,
            repetition_known=repetition_known,
        )

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
        self._session.clear()
        self.board_history = []

    def close(self):
        try:
            self._session.close()
        finally:
            self._multi.clear_local_cuda_graph_forward()
