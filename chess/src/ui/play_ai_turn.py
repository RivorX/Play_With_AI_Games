"""Background computation for one GUI AI turn.

The worker owns only the copied position and the selected search object. It
never mutates the Pygame UI or the live chess.Board; the main thread validates
the turn generation and FEN before applying its result.
"""

from __future__ import annotations

import math
import threading
import time


class PlayAITurn:
    def __init__(
        self,
        *,
        generation,
        board_fen,
        compute,
        min_visible_move_seconds=0.0,
        clock=None,
    ):
        self.generation = int(generation)
        self.board_fen = str(board_fen)
        self.compute = compute
        self.min_visible_move_seconds = float(min_visible_move_seconds)
        if (
            not math.isfinite(self.min_visible_move_seconds)
            or self.min_visible_move_seconds < 0
        ):
            raise ValueError(
                "min_visible_move_seconds must be a finite non-negative number"
            )
        self._clock = clock if clock is not None else time.monotonic
        self.compute_started_at = None
        self.compute_finished_at = None
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"play-ai-turn-{self.generation}",
            daemon=True,
        )

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        self.compute_started_at = self._clock()
        try:
            self.result = self.compute()
        except Exception as exc:
            self.error = exc
        finally:
            self.compute_finished_at = self._clock()
            self.done.set()

    def invalidate(self):
        self.cancelled.set()

    def wait(self, timeout=None):
        if self.thread.ident is None:
            return True
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    @property
    def is_done(self):
        return self.done.is_set() and not self.thread.is_alive()

    @property
    def compute_duration_seconds(self):
        """Elapsed time spent in the actual compute call, excluding UI pacing."""
        if self.compute_started_at is None:
            return 0.0
        finished_at = self.compute_finished_at
        if finished_at is None:
            finished_at = self._clock()
        return max(0.0, finished_at - self.compute_started_at)

    @property
    def presentation_remaining_seconds(self):
        """Time until a completed fast move may be shown in the GUI."""
        if self.compute_started_at is None or self.min_visible_move_seconds <= 0:
            return 0.0
        deadline = self.compute_started_at + self.min_visible_move_seconds
        return max(0.0, deadline - self._clock())

    @property
    def is_ready_to_apply(self):
        """Whether GUI may consume the result without blocking its event loop.

        Errors are ready immediately so the GUI can report them without an
        artificial delay. Successful results observe the visible-move minimum.
        """
        if not self.is_done:
            return False
        return self.error is not None or self.presentation_remaining_seconds <= 0.0
