"""Pure adaptive Elo scheduling shared by raw-NN and MCTS evaluators.

This module knows which Stockfish levels and color-paired games to schedule.
It deliberately knows nothing about Torch, chess engines, NN inference or MCTS.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from src.evaluation.elo_rating import (
    elo_fit_diagnostics,
    expected_score,
    performance_rating,
)


GameTask = tuple[int, int, bool]  # Stockfish Elo, game index, model plays white


def build_stockfish_elo_levels(
    minimum: int,
    maximum: int,
    *,
    initial_elo: float | None = None,
) -> list[int]:
    """Build a compact ladder from the UCI_Elo range reported by Stockfish."""
    minimum = int(minimum)
    maximum = int(maximum)
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    if maximum == minimum:
        return [minimum]

    span = maximum - minimum
    coarse_step = 100 if span <= 800 else 200
    levels = {minimum, maximum}
    first_aligned = int(math.ceil(minimum / coarse_step) * coarse_step)
    levels.update(range(first_aligned, maximum + 1, coarse_step))

    try:
        center = float(initial_elo) if initial_elo is not None else None
    except (TypeError, ValueError):
        center = None
    if center is not None and math.isfinite(center):
        center = max(float(minimum), min(float(maximum), center))
        # Dense local opponents give much more information than another sweep
        # over saturated 0%/100% levels. Wider offsets still recover if strength
        # changed substantially since the previous checkpoint measurement.
        for offset in (0, -100, 100, -200, 200, -400, 400, -600, 600, -800, 800):
            levels.add(max(minimum, min(maximum, int(round(center + offset)))))
    return sorted(levels)


class AdaptiveEloSchedule:
    """Stateful, inference-agnostic probe/focus/precision game selector."""

    def __init__(
        self,
        levels: Iterable[int],
        config: dict,
        *,
        max_total_games: int,
        workers: int,
        probe_games: int,
        focus_games: int,
        extra_round: int,
    ):
        self.levels = [int(level) for level in levels]
        self.config = dict(config or {})
        self.max_total_games = max(1, int(max_total_games))
        self.probe_games = max(2, int(probe_games))
        self.focus_games = max(self.probe_games, int(focus_games))
        self.extra_round = max(2, int(extra_round))
        self.target_focus = max(1, int(self.config.get("adaptive_target_focus_levels", 4) or 4))
        self.high_skip = float(self.config.get("adaptive_skip_high_score", 0.92) or 0.92)
        self.low_stop = float(self.config.get("adaptive_stop_low_score", 0.08) or 0.08)
        self.focus_min = float(self.config.get("adaptive_focus_min_score", 0.20) or 0.20)
        self.focus_max = float(self.config.get("adaptive_focus_max_score", 0.80) or 0.80)
        self.focus_min_games = min(self.focus_games, max(self.probe_games * 2, 16))
        self.target_se = float(self.config.get("adaptive_target_standard_error", 0.0) or 0.0)
        self.min_games_for_se_stop = max(
            self.probe_games * self.target_focus,
            int(self.config.get("adaptive_min_games_for_se_stop", 0) or 0),
        )
        min_batch = int(self.config.get("adaptive_min_batch_games", 0) or 0)
        if min_batch <= 0:
            min_batch = max(int(workers) * 2, self.probe_games * 3)
        self.min_batch_games = max(self.probe_games, min(self.max_total_games, min_batch))
        self.probe_levels_per_wave = max(
            1,
            min(len(self.levels), int(math.ceil(self.min_batch_games / max(1, self.probe_games)))),
        )
        initial = self.config.get("adaptive_initial_elo")
        try:
            initial = float(initial) if initial is not None else None
        except (TypeError, ValueError):
            initial = None
        self.initial_elo = initial if initial is not None and math.isfinite(initial) else None
        self.paired_games = bool(self.config.get("paired_openings_enabled", True))

        self.played_by_level = {level: 0 for level in self.levels}
        self.scheduled_by_level = {level: 0 for level in self.levels}
        self.scores_by_level: dict[int, list[float]] = {level: [] for level in self.levels}
        self.all_opponent_elos: list[float] = []
        self.all_scores: list[float] = []
        self.rating_levels: list[int] = []
        self.total_scheduled = 0

    @staticmethod
    def summarize(scores: list[float]) -> dict:
        wins = sum(score == 1.0 for score in scores)
        draws = sum(score == 0.5 for score in scores)
        losses = sum(score == 0.0 for score in scores)
        total = wins + draws + losses
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "total": total,
            "score": (wins + 0.5 * draws) / total if total else 0.0,
        }

    @staticmethod
    def _interleaved_order(levels: list[int]) -> list[int]:
        if len(levels) <= 2:
            return list(levels)
        # Breadth-first interval bisection brackets an unknown model quickly:
        # midpoint, lower midpoint, upper midpoint, then progressively finer
        # gaps. It avoids spending the first wave only on saturated extremes.
        order: list[int] = []
        intervals = [(0, len(levels) - 1)]
        seen: set[int] = set()
        while intervals:
            next_intervals = []
            for lo, hi in intervals:
                if lo > hi:
                    continue
                middle = (lo + hi) // 2
                if middle not in seen:
                    seen.add(middle)
                    order.append(levels[middle])
                if lo <= middle - 1:
                    next_intervals.append((lo, middle - 1))
                if middle + 1 <= hi:
                    next_intervals.append((middle + 1, hi))
            intervals = next_intervals
        return order

    @property
    def probe_order(self) -> list[int]:
        if self.initial_elo is None:
            return self._interleaved_order(self.levels)
        center = float(self.initial_elo)
        nearest = min(self.levels, key=lambda level: (abs(float(level) - center), level))
        lower = sorted(
            (level for level in self.levels if level < nearest),
            key=lambda level: abs(float(level) - center),
        )
        upper = sorted(
            (level for level in self.levels if level > nearest),
            key=lambda level: abs(float(level) - center),
        )
        order = [nearest]
        first_side, second_side = (upper, lower) if float(nearest) < center else (lower, upper)
        for index in range(max(len(lower), len(upper))):
            if index < len(first_side):
                order.append(first_side[index])
            if index < len(second_side):
                order.append(second_side[index])
        return order

    def probe_waves(self):
        order = self.probe_order
        for start in range(0, len(order), self.probe_levels_per_wave):
            yield order[start : start + self.probe_levels_per_wave]

    def _build_level_tasks(self, level: int, count: int) -> list[GameTask]:
        game_index = max(0, self.scheduled_by_level.get(int(level), 0))
        count = max(0, min(int(count), self.max_total_games - self.total_scheduled))
        if self.paired_games:
            if game_index % 2:
                game_index += 1
            count -= count % 2
        return [(int(level), game_index + offset, (game_index + offset) % 2 == 0) for offset in range(count)]

    def _commit_tasks(self, tasks: list[GameTask]) -> list[GameTask]:
        self.total_scheduled += len(tasks)
        for level, game_index, _ in tasks:
            self.scheduled_by_level[int(level)] = max(
                self.scheduled_by_level.get(int(level), 0),
                int(game_index) + 1,
            )
        return tasks

    def probe_tasks(self, wave_levels: Iterable[int]) -> list[GameTask]:
        tasks: list[GameTask] = []
        for level in wave_levels:
            remaining = self.max_total_games - self.total_scheduled - len(tasks)
            if remaining <= 0:
                break
            tasks.extend(self._build_level_tasks(level, min(self.probe_games, remaining)))
        return self._commit_tasks(tasks)

    def record_batch(self, opponent_elos: list[float], scores: list[float]):
        for opponent, score in zip(opponent_elos, scores):
            level = int(round(float(opponent)))
            self.all_opponent_elos.append(float(opponent))
            self.all_scores.append(float(score))
            self.played_by_level[level] = self.played_by_level.get(level, 0) + 1
            self.scores_by_level.setdefault(level, []).append(float(score))

    def score(self, level: int) -> float:
        return float(self.summarize(self.scores_by_level.get(int(level), []))["score"])

    def games(self, level: int) -> int:
        return int(self.played_by_level.get(int(level), 0) or 0)

    def played_levels(self) -> list[int]:
        return [level for level in self.levels if self.games(level) > 0]

    def useful_candidate_count(self) -> int:
        return sum(self.focus_min <= self.score(level) <= self.focus_max for level in self.played_levels())

    def should_stop_probing(self, wave_levels: Iterable[int]) -> bool:
        enough = self.useful_candidate_count() >= self.target_focus
        if self.initial_elo is not None and enough:
            return True
        return enough and any(
            self.summarize(self.scores_by_level.get(int(level), []))["total"] > 0
            and self.score(level) <= self.low_stop
            for level in sorted(int(level) for level in wave_levels)
        )

    def rating_observations(self) -> tuple[list[float], list[float]]:
        if not self.rating_levels:
            return list(self.all_opponent_elos), list(self.all_scores)
        selected = set(self.rating_levels)
        pairs = [
            (opponent, score)
            for opponent, score in zip(self.all_opponent_elos, self.all_scores)
            if int(round(opponent)) in selected
        ]
        if not pairs:
            return list(self.all_opponent_elos), list(self.all_scores)
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def current_rating(self) -> float | None:
        opponents, scores = self.rating_observations()
        return performance_rating(opponents, scores)

    def current_standard_error(self) -> float | None:
        opponents, scores = self.rating_observations()
        rating = performance_rating(opponents, scores)
        return elo_fit_diagnostics(opponents, scores, rating).get("standard_error")

    def precise_enough(self) -> bool:
        _, scores = self.rating_observations()
        if self.target_se <= 0.0 or len(scores) < self.min_games_for_se_stop:
            return False
        standard_error = self.current_standard_error()
        return standard_error is not None and float(standard_error) <= self.target_se

    def extend_game_budget(self, new_max_total_games: int) -> bool:
        """Increase the global budget without resetting accumulated evidence."""
        new_limit = max(self.max_total_games, int(new_max_total_games))
        if new_limit <= self.max_total_games:
            return False
        self.max_total_games = new_limit
        return True

    def _stable_score(self, level: int) -> float:
        observed = self.score(level)
        games = self.games(level)
        center = self.current_rating()
        expected = 0.5 if center is None else expected_score(center, float(level))
        if games <= 0:
            return expected
        prior_games = max(0.0, min(16.0, float(self.config.get("adaptive_focus_score_prior_games", 4.0) or 4.0)))
        return (observed * games + expected * prior_games) / max(1.0, games + prior_games)

    def _focus_sort_key(self, level: int):
        score = self._stable_score(level)
        center = self.current_rating()
        center_penalty = 0.0 if center is None else abs(float(level) - center) / 400.0
        return center_penalty + abs(score - 0.5), self.games(level), abs(score - 0.5)

    def select_rating_levels(self) -> list[int]:
        candidates = self.played_levels()
        selected = [level for level in candidates if self.focus_min <= self._stable_score(level) <= self.focus_max]
        selected.sort(key=self._focus_sort_key)
        if len(selected) < self.target_focus:
            extras = sorted(
                (
                    level
                    for level in candidates
                    if level not in selected and self.low_stop < self._stable_score(level) < self.high_skip
                ),
                key=self._focus_sort_key,
            )
            selected.extend(extras[: max(0, self.target_focus - len(selected))])
        if not selected and candidates:
            selected = sorted(candidates, key=self._focus_sort_key)[:1]
        self.rating_levels = selected[: self.target_focus]
        return list(self.rating_levels)

    def focus_tasks(self) -> list[GameTask]:
        tasks: list[GameTask] = []
        for level in self.rating_levels:
            if self.total_scheduled + len(tasks) >= self.max_total_games or self.games(level) >= self.focus_games:
                continue
            if self.games(level) >= self.focus_min_games:
                score = self.score(level)
                if score <= self.low_stop or score >= self.high_skip:
                    continue
            count = min(
                self.extra_round,
                self.focus_games - self.games(level),
                self.max_total_games - self.total_scheduled - len(tasks),
            )
            tasks.extend(self._build_level_tasks(level, count))
        return self._commit_tasks(tasks)

    def focus_complete(self) -> bool:
        return not any(self.games(level) < self.focus_games for level in self.rating_levels)

    def precision_levels(self) -> list[int]:
        candidates = [
            level for level in self.rating_levels if self.low_stop < self._stable_score(level) < self.high_skip
        ] or list(self.rating_levels) or self.played_levels()
        return sorted(candidates, key=self._focus_sort_key)[: self.target_focus]

    def precision_tasks(self) -> tuple[list[int], list[GameTask]]:
        levels = self.precision_levels()
        tasks: list[GameTask] = []
        for level in levels:
            count = min(
                self.extra_round,
                self.max_total_games - self.total_scheduled - len(tasks),
            )
            if count <= 0:
                break
            tasks.extend(self._build_level_tasks(level, count))
        return levels, self._commit_tasks(tasks)
