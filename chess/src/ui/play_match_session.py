"""Thread-safe lifecycle and durable results for a GUI model match.

The play window only observes immutable snapshots. Evaluation callbacks update
this session, so an old match can finish cancelling without mutating a restarted
game's UI state.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path


_VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
_RUNTIME_STATUSES = {
    "starting",
    "starting_server",
    "snapshotting_models",
    "loading_models",
    "models_loaded",
    "starting_workers",
    "running",
    "cancelling",
    "cancelled",
    "completed",
    "error",
}


def _json_safe_runtime(details):
    safe = {}
    for key, value in dict(details or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = str(value)
    return safe


class PlayMatchSession:
    """Own one cancellable evaluator run and its resolved game results."""

    def __init__(
        self,
        *,
        total_games,
        model1_name,
        model2_name,
        output_dir,
        match_settings=None,
    ):
        self.total_games = max(1, int(total_games))
        self.model1_name = str(model1_name or "model 1")
        self.model2_name = str(model2_name or "model 2")
        self.output_dir = Path(output_dir)
        self.match_settings = dict(match_settings or {})
        self.session_id = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.result_path = None
        self.raw_stats = None
        self._finalized = False
        self._seen_background_indices = set()
        self._background_result = None
        background_total = max(0, self.total_games - 1)
        self.stats = {
            "session_id": self.session_id,
            "total": self.total_games,
            "attempted": 0,
            "completed": 0,
            "unresolved": 0,
            "invalid_results": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "model1_wins": 0,
            "model2_wins": 0,
            "model1_score_x2": 0,
            "model2_score_x2": 0,
            "model1_as_white": 0,
            "model1_as_black": 0,
            "background_attempted": 0,
            "background_completed": 0,
            "background_unresolved": 0,
            "background_invalid_results": 0,
            "background_total": background_total,
            "background_done": False,
            "background_starting": background_total > 0,
            "background_workers": 0,
            "active_games_per_worker": 0,
            "central_inference": False,
            "central_compile_enabled": False,
            "model_summary": None,
            "match_engine": "shared_eval",
            "runtime_load_s": 0.0,
            "runtime_model_load_s": 0.0,
            "runtime_stage_started_at": time.perf_counter(),
            "runtime_status": "starting" if background_total else "visible_running",
            "status": "starting" if background_total else "visible_running",
            "error": None,
            "cancel_reason": None,
            "started_at": time.perf_counter(),
            "finished_at": None,
            "started_at_unix": time.time(),
            "finished_at_unix": None,
            "plies_total": 0,
            "visible_game": None,
            "visible_game_finished": False,
            "background_remaining": background_total,
        }

    def snapshot(self):
        with self.lock:
            result = dict(self.stats)
        result["runtime_stage_elapsed_s"] = max(
            0.0,
            time.perf_counter()
            - float(result.get("runtime_stage_started_at", result.get("started_at", 0.0)) or 0.0),
        )
        result["total"] = self.total_games
        result["model1_name"] = self.model1_name
        result["model2_name"] = self.model2_name
        result["result_path"] = str(self.result_path) if self.result_path else None
        result["match_settings"] = dict(self.match_settings)
        return result

    def record_visible_game(
        self,
        *,
        result,
        plies,
        model1_as_white,
        winner_model,
        final_fen,
        moves_uci,
        moves_san,
        mcts_searches=None,
    ):
        """Include the visible first game in the same aggregate as the batch."""
        result = str(result or "*")
        visible_game = {
            "game_index": 0,
            "result": result,
            "plies": max(0, int(plies or 0)),
            "model1_as_white": bool(model1_as_white),
            "winner_model": str(winner_model) if winner_model else None,
            "final_fen": str(final_fen or ""),
            "moves_uci": [str(move) for move in (moves_uci or [])],
            "moves_san": [str(move) for move in (moves_san or [])],
        }
        if mcts_searches is not None:
            # Optional, bounded diagnostics for the visible game only. Keep old
            # readers and callers compatible when no search records are passed.
            visible_game["mcts_searches"] = list(mcts_searches)
        with self.lock:
            if self.stats["visible_game_finished"]:
                return False
            self.stats["visible_game"] = visible_game
            self.stats["visible_game_finished"] = True
            self._record_result_locked(
                result=result,
                model1_as_white=model1_as_white,
                plies=plies,
                unresolved=int(result not in _VALID_RESULTS),
                visible=True,
            )
        self._finalize_if_ready()
        return True

    def record_visible_error(
        self, error, *, final_fen, moves_uci, moves_san, mcts_searches=None
    ):
        """Finish a failed first game without inventing a chess result."""
        message = str(error or "AI failed before the visible game finished")
        with self.lock:
            if self.stats["visible_game_finished"]:
                return False
            self.stats["visible_game"] = {
                "game_index": 0,
                "result": "*",
                "plies": len(moves_uci or []),
                "model1_as_white": True,
                "winner_model": None,
                "final_fen": str(final_fen or ""),
                "moves_uci": [str(move) for move in (moves_uci or [])],
                "moves_san": [str(move) for move in (moves_san or [])],
                "error": message,
            }
            if mcts_searches is not None:
                self.stats["visible_game"]["mcts_searches"] = list(mcts_searches)
            self.stats["visible_game_finished"] = True
            self.stats["error"] = message
            self._record_result_locked(
                result="*", model1_as_white=True, plies=len(moves_uci or []),
                unresolved=1, visible=True,
            )
        self._finalize_if_ready()
        return True

    def record_visible_cancel(
        self, reason, *, final_fen, moves_uci, moves_san, mcts_searches=None
    ):
        """Count an abandoned visible game as unresolved, without inventing a result."""
        message = str(reason or "visible game cancelled")
        with self.lock:
            if self.stats["visible_game_finished"]:
                return False
            self.stats["visible_game"] = {
                "game_index": 0,
                "result": "*",
                "plies": len(moves_uci or []),
                "model1_as_white": True,
                "winner_model": None,
                "final_fen": str(final_fen or ""),
                "moves_uci": [str(move) for move in (moves_uci or [])],
                "moves_san": [str(move) for move in (moves_san or [])],
                "cancelled": True,
                "cancel_reason": message,
            }
            if mcts_searches is not None:
                self.stats["visible_game"]["mcts_searches"] = list(mcts_searches)
            self.stats["visible_game_finished"] = True
            self.stats["visible_game_cancelled"] = True
            self.stats["cancel_reason"] = message
            self._record_result_locked(
                result="*", model1_as_white=True, plies=len(moves_uci or []),
                unresolved=1, visible=True,
            )
        self._finalize_if_ready()
        return True

    def _record_result_locked(
        self, *, result, model1_as_white, plies, unresolved=0, invalid=0, visible=False
    ):
        """Update match totals; caller owns ``self.lock``."""
        self.stats["attempted"] += 1
        result = str(result or "*")
        if visible:
            self.stats["background_remaining"] = max(
                0,
                int(self.stats["background_total"])
                - int(self.stats["background_attempted"]),
            )
        else:
            self.stats["background_attempted"] += 1
            self.stats["background_remaining"] = max(
                0,
                int(self.stats["background_total"])
                - int(self.stats["background_attempted"]),
            )
        self.stats["background_starting"] = False

        if unresolved > 0 or result not in _VALID_RESULTS:
            if invalid > 0 or result not in _VALID_RESULTS and result not in {"*", "<ai-error>"}:
                self.stats["invalid_results"] += max(1, int(invalid or 0))
                if not visible:
                    self.stats["background_invalid_results"] += max(1, int(invalid or 0))
            else:
                self.stats["unresolved"] += max(1, int(unresolved or 0))
                if not visible:
                    self.stats["background_unresolved"] += max(1, int(unresolved or 0))
            return

        self.stats["completed"] += 1
        if not visible:
            self.stats["background_completed"] += 1
        self.stats["plies_total"] += max(0, int(plies or 0))
        model1_as_white = bool(model1_as_white)
        if model1_as_white:
            self.stats["model1_as_white"] += 1
        else:
            self.stats["model1_as_black"] += 1

        if result == "1-0":
            self.stats["white_wins"] += 1
        elif result == "0-1":
            self.stats["black_wins"] += 1
        else:
            self.stats["draws"] += 1

        model1_won = (result == "1-0" and model1_as_white) or (result == "0-1" and not model1_as_white)
        model2_won = (result == "0-1" and model1_as_white) or (result == "1-0" and not model1_as_white)
        if model1_won:
            self.stats["model1_wins"] += 1
            self.stats["model1_score_x2"] += 2
        elif model2_won:
            self.stats["model2_wins"] += 1
            self.stats["model2_score_x2"] += 2
        else:
            self.stats["model1_score_x2"] += 1
            self.stats["model2_score_x2"] += 1

        if not visible:
            bg_wins_key = "background_model1_wins" if model1_won else "background_model2_wins" if model2_won else "background_draws"
            self.stats[bg_wins_key] = int(self.stats.get(bg_wins_key, 0)) + 1

    def finish_without_runner(self):
        """Persist a one-game match whose visible game is the full match."""
        self.raw_stats = {"visible_game_only": True}
        self._finish_background({"completed": 0})

    def update_runtime(self, details):
        details = _json_safe_runtime(details)
        with self.lock:
            for key in (
                "workers",
                "batch_games",
                "central_inference",
                "engine",
                "startup_s",
                "load_s",
                "server_spawn_s",
                "snapshot_s",
                "worker_launch_s",
                "central_compile_enabled",
                "model_summary",
                "active_games",
                "completed_games",
                "plies_processed",
                "nn_batches",
                "search_batches",
                "runtime_activity",
                "inference_side",
                "inference_batch_size",
                "activity_elapsed_s",
            ):
                if key in details:
                    target = {
                        "workers": "background_workers",
                        "batch_games": "active_games_per_worker",
                        "engine": "match_engine",
                        "startup_s": "runtime_load_s",
                        "load_s": "runtime_model_load_s",
                    }.get(key, key)
                    self.stats[target] = details[key]

            status = str(details.get("status", self.stats["runtime_status"]))
            if status != str(self.stats.get("runtime_status", "")):
                self.stats["runtime_stage_started_at"] = time.perf_counter()
            if details.get("error"):
                self.stats["error"] = str(details["error"])
            if details.get("cancel_reason"):
                self.stats["cancel_reason"] = str(details["cancel_reason"])
            if status in _RUNTIME_STATUSES:
                self.stats["runtime_status"] = status
                if status == "running":
                    self.stats["status"] = "running"
                elif status in {
                    "starting",
                    "starting_server",
                    "snapshotting_models",
                    "loading_models",
                    "models_loaded",
                    "starting_workers",
                }:
                    self.stats["status"] = status
                elif status == "cancelling":
                    self.stats["status"] = "cancelling"
                elif status == "error":
                    self.stats["status"] = "error"
                elif status == "cancelled":
                    self.stats["status"] = "cancelling"
                elif status == "completed" and self.stats["status"] not in {"error", "cancelling"}:
                    self.stats["status"] = "running"

            self.stats["background_starting"] = self.stats["status"] in {
                "starting",
                "starting_server",
                "snapshotting_models",
                "loading_models",
                "models_loaded",
                "starting_workers",
            }

    def record_progress(
        self,
        _completed=None,
        _game_idx=None,
        wins=0,
        draws=0,
        losses=0,
        unresolved=0,
        plies=0,
        result="*",
        model1_as_white=True,
    ):
        """Record only a resolved chess result; unresolved games stay separate."""
        try:
            unresolved = int(unresolved or 0)
        except (TypeError, ValueError):
            unresolved = 0
        result = str(result or "*")
        with self.lock:
            if _game_idx is not None:
                try:
                    game_idx = int(_game_idx)
                except (TypeError, ValueError):
                    game_idx = None
                if game_idx is not None:
                    if game_idx in self._seen_background_indices:
                        return
                    self._seen_background_indices.add(game_idx)
            self._record_result_locked(
                result=result,
                model1_as_white=model1_as_white,
                plies=plies,
                unresolved=unresolved,
                invalid=1 if unresolved <= 0 and result not in _VALID_RESULTS else 0,
            )
            if self.stats["status"] not in {"cancelling", "error"}:
                self.stats["status"] = "running"

    def request_stop(self):
        if self.done_event.is_set():
            return
        self.stop_event.set()
        with self.lock:
            if not self.stats["background_done"]:
                self.stats["status"] = "cancelling"
                self.stats["runtime_status"] = "cancelling"
                self.stats["background_starting"] = False
        self._finalize_if_ready()

    def start(self, runner):
        if self.thread is not None:
            raise RuntimeError("This match session has already been started.")
        self.thread = threading.Thread(
            target=self._run,
            args=(runner,),
            name=f"play-model-match-{self.session_id[:8]}",
            daemon=True,
        )
        self.thread.start()

    def _run(self, runner):
        try:
            self.raw_stats = runner(
                stop_event=self.stop_event,
                progress_callback=self.record_progress,
                runtime_callback=self.update_runtime,
            ) or {}
            self._finish_from_result(self.raw_stats)
        except Exception as exc:
            if self.stop_event.is_set():
                self._finish_from_result({"cancelled": True, "cancel_reason": str(exc)})
            else:
                self._finish_from_result({"error": f"{type(exc).__name__}: {exc}"})

    def _finish_from_result(self, result):
        self.raw_stats = dict(result or {})
        self._finish_background(self.raw_stats)

    def _finish_background(self, result):
        result = dict(result or {})
        with self.lock:
            self._background_result = result
            self._merge_background_result_locked(result)
            self.stats["background_done"] = True
            self.stats["background_starting"] = False
            if result.get("error"):
                self.stats["error"] = str(result["error"])
            if result.get("cancel_reason"):
                self.stats["cancel_reason"] = str(result["cancel_reason"])
            if not self.stats["visible_game_finished"] and not self.stats.get("error"):
                self.stats["status"] = "visible_running"
        self._finalize_if_ready()

    def _merge_background_result_locked(self, result):
        """Reconcile a runner's final totals with per-game callbacks idempotently."""
        def _count(key):
            try:
                return max(0, int(result.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        for source_key, target_key in (
            ("wins", "background_model1_wins"),
            ("draws", "background_draws"),
            ("losses", "background_model2_wins"),
        ):
            target = _count(source_key)
            current = int(self.stats.get(target_key, 0) or 0)
            if target > current:
                delta = target - current
                self.stats[target_key] = target
                aggregate_key = {
                    "background_model1_wins": "model1_wins",
                    "background_model2_wins": "model2_wins",
                    "background_draws": "draws",
                }[target_key]
                self.stats[aggregate_key] += delta
                if aggregate_key == "model1_wins":
                    self.stats["model1_score_x2"] += 2 * delta
                elif aggregate_key == "model2_wins":
                    self.stats["model2_score_x2"] += 2 * delta
                else:
                    self.stats["model1_score_x2"] += delta
                    self.stats["model2_score_x2"] += delta
                self.stats["completed"] += delta
                self.stats["background_completed"] += delta

        for source_key, target_key, aggregate_key in (
            ("unresolved", "background_unresolved", "unresolved"),
            ("invalid_results", "background_invalid_results", "invalid_results"),
        ):
            target = _count(source_key)
            current = int(self.stats.get(target_key, 0) or 0)
            if target > current:
                delta = target - current
                self.stats[target_key] = target
                self.stats[aggregate_key] += delta

        target_attempted = max(
            int(self.stats["background_attempted"]),
            _count("completed"),
            int(self.stats["background_completed"])
            + int(self.stats["background_unresolved"])
            + int(self.stats["background_invalid_results"]),
        )
        missing = max(0, target_attempted - int(self.stats["background_attempted"]))
        if missing:
            self.stats["background_attempted"] = target_attempted
            self.stats["attempted"] += missing
            self.stats["background_unresolved"] += missing
            self.stats["unresolved"] += missing
        self.stats["background_remaining"] = max(
            0,
            int(self.stats["background_total"])
            - int(self.stats["background_attempted"]),
        )

    def _finalize_if_ready(self):
        with self.lock:
            if self._finalized or not self.stats["background_done"] or not self.stats["visible_game_finished"]:
                return False
            self._finalized = True
            attempted = max(
                int(self.stats["attempted"]),
                int(self.stats["completed"])
                + int(self.stats["unresolved"])
                + int(self.stats["invalid_results"]),
            )
            self.stats["attempted"] = attempted
            if self.stats.get("error") or (self._background_result or {}).get("error"):
                status = "error"
            elif self.stop_event.is_set() or (self._background_result or {}).get("cancelled") or self.stats.get("visible_game_cancelled"):
                status = "cancelled"
            elif attempted >= self.total_games:
                status = "completed"
            else:
                status = "incomplete"
            self.stats["status"] = status
            self.stats["runtime_status"] = status if status in _RUNTIME_STATUSES else "error"
            self.stats["background_remaining"] = max(
                0,
                int(self.stats["background_total"])
                - int(self.stats["background_attempted"]),
            )
            self.stats["finished_at"] = time.perf_counter()
            self.stats["finished_at_unix"] = time.time()
            if not self.stats.get("cancel_reason") and (self._background_result or {}).get("cancel_reason"):
                self.stats["cancel_reason"] = str(self._background_result["cancel_reason"])

        self._persist_result()
        self.done_event.set()
        return True

    def _persist_result(self):
        temp_path = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            started = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.snapshot()["started_at_unix"]))
            result_path = self.output_dir / f"play_match_{started}_{self.session_id[:8]}.json"
            temp_path = result_path.with_suffix(".json.tmp")
            payload = self.snapshot()
            payload["raw_stats"] = self.raw_stats
            payload["raw_stats_scope"] = "background_games_after_visible_first_game"
            with temp_path.open("w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2, default=str)
                output.write("\n")
            os.replace(temp_path, result_path)
            self.result_path = result_path
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            with self.lock:
                self.stats["persistence_error"] = f"{type(exc).__name__}: {exc}"
            self.result_path = None

    def wait(self, timeout=None):
        """Wait for evaluator-owned processes to drain; return whether done."""
        if self.thread is None:
            return True
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    @property
    def is_done(self):
        thread = self.thread
        return self.done_event.is_set() and (thread is None or not thread.is_alive())

