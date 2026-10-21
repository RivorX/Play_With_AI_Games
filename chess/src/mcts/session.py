"""Per-game lifetime wrapper around the existing persistent MCTS tree.

The session owns only the current Python root, the stable key used by the
engine's native forest, and root-sync bookkeeping. The caller remains the
source of truth for the board, encoded history, repetition counts, and model
identity.
"""

from __future__ import annotations

import sys


class SearchSession:
    """Keep one game's tree attached across searches and actual moves.

    ``advance`` must be called with the authoritative *post-move* context. It
    checks that context against the selected child, then promotes the child
    with the prior evaluator-generation token. The next search checks that
    token against the current model generation before accepting reuse. A
    successful structural promotion stores the complete observed root anchor,
    allowing any number of actual plies between searches without retaining
    caller history on every node. The owner must serialize search, commit,
    advance, clear, and close for a given session; the native search forest is
    not safe for concurrent mutation under one tree key.
    """

    _STATE_KEYS = frozenset({
        'board',
        'root',
        '_root_synced',
        'board_history',
        'position_counts',
        'position_history_complete',
        '_native_tree_key',
    })

    def __init__(self, engine, *, tree_key=None):
        self._engine = engine
        self._tree_key = object() if tree_key is None else tree_key
        try:
            hash(self._tree_key)
        except TypeError as exc:
            raise TypeError('tree_key must be hashable') from exc
        self._root = None
        self._root_synced = False
        self._closed = False

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._ensure_open()
        self._root = value

    @property
    def tree_key(self):
        return self._tree_key

    @property
    def root_synced(self):
        return self._root_synced

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('SearchSession is closed')

    def make_state(
        self,
        board,
        *,
        board_history,
        position_counts,
        repetition_known,
        **overrides,
    ):
        """Build the canonical mapping accepted by ``search_many``.

        The history and count objects are passed through by reference and are
        never retained by the session.
        """
        self._ensure_open()
        reserved = self._STATE_KEYS.intersection(overrides)
        if reserved:
            names = ', '.join(sorted(reserved))
            raise TypeError(f'cannot override SearchSession state fields: {names}')
        state = {
            'board': board,
            'root': self._root,
            '_root_synced': bool(self._root_synced),
            'board_history': board_history,
            'position_counts': position_counts,
            'position_history_complete': bool(repetition_known),
            '_native_tree_key': self._tree_key,
        }
        state.update(overrides)
        return state

    def commit(self, state):
        """Capture the root and sync state after a completed engine search."""
        self._ensure_open()
        if not isinstance(state, dict):
            raise TypeError('SearchSession.commit expects a mapping search state')
        if state.get('_native_tree_key') is not self._tree_key and (
            state.get('_native_tree_key') != self._tree_key
        ):
            raise ValueError('search state belongs to a different SearchSession')
        self._root = state.get('root')
        self._root_synced = bool(state.get('_root_synced', False))
        return self._root

    def advance(
        self,
        move,
        *,
        board,
        board_history,
        position_counts,
        repetition_known,
    ):
        """Promote a searched move using authoritative post-move context.

        Returns ``True`` only when the searched child and exact next-root
        provenance match the saved anchor. This is a structural promotion;
        the saved evaluator-generation token is checked against the current
        model at the next search before reuse is accepted. On unknown context
        or a missing child, the Python root and native tree are released and
        the method returns ``False``.
        """
        self._ensure_open()
        root = self._root
        if root is None:
            return self._drop()

        child = root.get_child_for_move(move)
        if child is None:
            return self._drop()

        _ = child.board
        anchor = getattr(root, '_reuse_context_anchor', None)
        advance_context = getattr(anchor, 'advance_to_child', None)
        expected = advance_context(child.board) if callable(advance_context) else None
        # Do not rescan local model tensors for each real ply. The saved token
        # remains attached to the old evaluations; the next search compares it
        # with its once-per-batch current token before any reuse is accepted.
        generation = getattr(anchor, 'evaluation_generation', None)
        observed = self._engine._build_root_reuse_context(
            board,
            board_history,
            position_counts,
            bool(repetition_known),
            generation,
            evaluation_generation_is_canonical=True,
        )
        if expected is None or observed is None or not expected.matches(observed):
            return self._drop()

        sync_native_parent = getattr(
            self._engine,
            '_sync_native_node_for_promotion',
            None,
        )
        if callable(sync_native_parent):
            sync_native_parent(root, self._tree_key)

        promoted = child.detach_as_root()
        # The observed context contains the current frame that the one-step
        # derived anchor cannot infer without the caller's updated history.
        # Storing it lets a second move be validated before another search.
        promoted._reuse_context_anchor = observed
        self._root = promoted
        self._root_synced = True
        return True

    def _drop(self):
        self._root = None
        self._root_synced = False
        self._engine.release_native_tree(self._tree_key)
        return False

    def clear(self):
        """Release this game's tree while keeping its stable native key."""
        self._ensure_open()
        self._root = None
        self._root_synced = False
        self._engine.release_native_tree(self._tree_key)

    def close(self):
        """Release the tree and permanently close the session."""
        if self._closed:
            return
        primary_error = sys.exception()
        self._root = None
        self._root_synced = False
        self._closed = True
        try:
            self._engine.release_native_tree(self._tree_key)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            try:
                primary_error.add_note(
                    f"SearchSession native cleanup also failed: {cleanup_error}"
                )
            except Exception:
                pass

    def __del__(self):
        """Best-effort native cleanup when an owning loop exits exceptionally.

        Normal callers should use ``close`` or ``clear`` at a deterministic
        lifecycle boundary. This fallback covers exceptional exits such as a
        self-play loop unwinding before its usual cleanup path.
        """
        try:
            if not getattr(self, '_closed', True):
                self.close()
        except Exception:
            # Destructors can run during interpreter teardown, when the engine
            # or its native runtime may already be partially unavailable.
            pass

