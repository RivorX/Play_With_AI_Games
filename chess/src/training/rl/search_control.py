"""Regret-guided opening archive used by RL self-play."""

from __future__ import annotations

import math

import numpy as np


SEARCH_CONTROL_STATE_VERSION = 2
MIN_REGRET_SUFFIX_SAMPLES = 8


def fen_is_exact_restart(fen):
    """A zero halfmove clock makes FEN-only repetition reconstruction exact."""
    try:
        parts = str(fen or "").split()
        return len(parts) >= 6 and int(parts[4]) == 0
    except (TypeError, ValueError):
        return False


def suffix_regret_targets(
    game_history,
    outcome,
    *,
    min_suffix_samples=MIN_REGRET_SUFFIX_SAMPLES,
):
    """Return RGSC Eq. 2 regret for every trajectory row.

    Stored ``played_q`` is from the side-to-move point of view, therefore the
    terminal result is signed separately for every row. Missing/zero-visit Q
    estimates do not enter the suffix average.
    """
    regrets = [float("nan")] * len(game_history)
    min_suffix_samples = max(1, int(min_suffix_samples))
    error_sum = 0.0
    valid_count = 0
    for idx in range(len(game_history) - 1, -1, -1):
        entry = game_history[idx]
        turn = bool(entry[3]) if len(entry) > 3 else True
        played_q = float(entry[14]) if len(entry) > 14 else float("nan")
        visits = int(entry[17]) if len(entry) > 17 else 0
        if math.isfinite(played_q) and visits > 0:
            target = float(outcome) if turn else -float(outcome)
            error_sum += (max(-1.0, min(1.0, played_q)) - target) ** 2
            valid_count += 1
        # A terminal one-row suffix can hit the mathematical ceiling of 4.0
        # from one noisy Q estimate. Requiring a short trajectory window keeps
        # the archive score informative instead of filling it with identical
        # last-ply maxima.
        if valid_count >= min_suffix_samples:
            regrets[idx] = error_sum / float(valid_count)
    return regrets


class RegretOpeningArchive:
    """Fixed-capacity, deduplicated prioritized regret buffer (PRB)."""

    def __init__(self, capacity=800, rank_alpha=0.50, ema_alpha=0.50):
        self.capacity = max(1, int(capacity))
        self.rank_alpha = max(0.0, float(rank_alpha))
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._entries = {}
        self._key_to_id = {}
        self._next_id = 1
        self.last_restore_status = "new"

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _key(fen, history_fens):
        return str(fen), tuple(str(item) for item in (history_fens or ()))

    def add_candidates(self, candidates, iteration):
        added = updated = rejected = replaced = 0
        candidates = list(candidates or ())
        if len(candidates) > 1:
            # Candidate order follows replay/game insertion order. A stable
            # per-iteration shuffle prevents equal-score admission from always
            # retaining the same tail of that order while remaining resumable.
            rng = np.random.default_rng(0x5EA2C + int(iteration) * 1_000_003)
            candidates = [candidates[int(idx)] for idx in rng.permutation(len(candidates))]
        for candidate in candidates:
            fen = candidate.get("fen")
            regret = float(candidate.get("regret", float("nan")))
            if not fen_is_exact_restart(fen) or not math.isfinite(regret) or regret <= 0.0:
                rejected += 1
                continue
            history_fens = tuple(candidate.get("history_fens", ()) or ())
            key = self._key(fen, history_fens)
            existing_id = self._key_to_id.get(key)
            if existing_id is not None:
                entry = self._entries[existing_id]
                entry["regret"] = (
                    (1.0 - self.ema_alpha) * float(entry["regret"])
                    + self.ema_alpha * regret
                )
                entry["updated_iteration"] = int(iteration)
                entry["last_seen_iteration"] = int(iteration)
                updated += 1
                continue
            if len(self._entries) >= self.capacity:
                lowest_id, lowest = min(
                    self._entries.items(),
                    key=lambda item: (
                        float(item[1]["regret"]),
                        int(item[1].get("last_seen_iteration", item[1].get("updated_iteration", 0))),
                        int(item[1].get("last_sampled_iteration", -1)),
                        int(item[0]),
                    ),
                )
                if regret < float(lowest["regret"]):
                    rejected += 1
                    continue
                self._key_to_id.pop(self._key(lowest["fen"], lowest["history_fens"]), None)
                del self._entries[lowest_id]
                replaced += 1
            entry_id = self._next_id
            self._next_id += 1
            self._entries[entry_id] = {
                "id": entry_id,
                "fen": str(fen),
                "history_fens": history_fens,
                "regret": regret,
                "inserted_iteration": int(iteration),
                "updated_iteration": int(iteration),
                "last_seen_iteration": int(iteration),
                "last_sampled_iteration": -1,
                "replays": 0,
            }
            self._key_to_id[key] = entry_id
            added += 1
        return {
            "added": added,
            "updated": updated,
            "rejected": rejected,
            "replaced": replaced,
        }

    def update_replayed(self, updates, iteration):
        updated = 0
        for entry_id, regret in (updates or {}).items():
            entry = self._entries.get(int(entry_id))
            regret = float(regret)
            if entry is None or not math.isfinite(regret) or regret < 0.0:
                continue
            entry["regret"] = (
                (1.0 - self.ema_alpha) * float(entry["regret"])
                + self.ema_alpha * regret
            )
            entry["updated_iteration"] = int(iteration)
            entry["replays"] = int(entry.get("replays", 0)) + 1
            updated += 1
        return updated

    def sample(self, count, *, rng=None, iteration=None):
        count = min(max(0, int(count)), len(self._entries))
        if count <= 0:
            return []
        rng = rng if rng is not None else np.random
        entries = sorted(self._entries.values(), key=lambda item: int(item["id"]))
        scores = np.asarray([max(1e-12, float(item["regret"])) for item in entries])
        # Raw regret is bounded and often tied. Magnitude softmax previously
        # turned temperature=0.1 into score**10 and collapsed sampling. Average
        # ranks are scale-invariant and give exact ties identical probability.
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(entries), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and scores[order[end]] == scores[order[start]]:
                end += 1
            average_rank = 0.5 * ((start + 1) + end)
            ranks[order[start:end]] = average_rank
            start = end
        weights = np.power(ranks, -self.rank_alpha)
        weights /= float(weights.sum())
        selected = rng.choice(len(entries), size=count, replace=False, p=weights)
        if iteration is not None:
            for idx in np.asarray(selected).reshape(-1):
                entries[int(idx)]["last_sampled_iteration"] = int(iteration)
        return [
            {
                "fen": entries[int(idx)]["fen"],
                "history_fens": list(entries[int(idx)]["history_fens"]),
                "archive_id": int(entries[int(idx)]["id"]),
                "regret": float(entries[int(idx)]["regret"]),
            }
            for idx in np.asarray(selected).reshape(-1)
        ]

    def stats(self, sampled=None):
        scores = np.asarray(
            [float(entry["regret"]) for entry in self._entries.values()], dtype=np.float64
        )
        sampled_scores = np.asarray(
            [float(entry.get("regret", float("nan"))) for entry in (sampled or ())],
            dtype=np.float64,
        )
        sampled_scores = sampled_scores[np.isfinite(sampled_scores)]
        tie_fraction = 0.0
        if scores.size:
            _, counts = np.unique(np.round(scores, decimals=4), return_counts=True)
            tie_fraction = float(counts.max()) / float(scores.size)
        return {
            "search_control_archive_size": len(self),
            "search_control_archive_capacity": int(self.capacity),
            "search_control_regret_tie_fraction": tie_fraction,
            "search_control_regret_mean": float(scores.mean()) if scores.size else 0.0,
            "search_control_regret_p10": float(np.percentile(scores, 10)) if scores.size else 0.0,
            "search_control_regret_p50": float(np.percentile(scores, 50)) if scores.size else 0.0,
            "search_control_regret_p90": float(np.percentile(scores, 90)) if scores.size else 0.0,
            "search_control_sampled_regret_mean": (
                float(sampled_scores.mean()) if sampled_scores.size else 0.0
            ),
            "search_control_sampled_regret_p90": (
                float(np.percentile(sampled_scores, 90)) if sampled_scores.size else 0.0
            ),
        }

    def state_dict(self):
        return {
            "format_version": SEARCH_CONTROL_STATE_VERSION,
            "capacity": self.capacity,
            "rank_alpha": self.rank_alpha,
            "ema_alpha": self.ema_alpha,
            "next_id": self._next_id,
            "entries": [dict(entry) for entry in self._entries.values()],
        }

    def load_state_dict(self, state):
        if not state:
            return 0
        version = int(state.get("format_version", 0) or 0)
        if version not in (1, SEARCH_CONTROL_STATE_VERSION):
            raise ValueError("Unsupported search-control archive state version.")
        self._entries = {}
        self._key_to_id = {}
        raw_entries = list(state.get("entries", ()) or ())
        sanitized = []
        for raw in raw_entries:
            entry = dict(raw)
            entry_id = int(entry["id"])
            entry["history_fens"] = tuple(entry.get("history_fens", ()) or ())
            regret = float(entry.get("regret", float("nan")))
            if (
                not fen_is_exact_restart(entry.get("fen"))
                or not math.isfinite(regret)
                or regret <= 0.0
            ):
                continue
            entry["regret"] = regret
            entry["last_seen_iteration"] = int(
                entry.get("last_seen_iteration", entry.get("updated_iteration", 0))
            )
            entry["last_sampled_iteration"] = int(entry.get("last_sampled_iteration", -1))
            sanitized.append(entry)
        legacy_scores = np.asarray(
            [float(entry["regret"]) for entry in sanitized], dtype=np.float64
        )
        legacy_at_ceiling = (
            version == 1
            and legacy_scores.size > 1
            and float(np.ptp(legacy_scores)) <= 1e-6
            and float(np.percentile(legacy_scores, 50)) >= 4.0 - 1e-6
        )
        if legacy_at_ceiling:
            self._next_id = 1
            self.last_restore_status = "legacy_saturated_reset"
            return 0
        sanitized.sort(
            key=lambda entry: (
                float(entry["regret"]),
                int(entry.get("last_seen_iteration", 0)),
                int(entry["id"]),
            ),
            reverse=True,
        )
        for entry in sanitized[: self.capacity]:
            entry_id = int(entry["id"])
            self._entries[entry_id] = entry
            self._key_to_id[self._key(entry["fen"], entry["history_fens"])] = entry_id
        self._next_id = max(
            int(state.get("next_id", 1) or 1),
            max(self._entries, default=0) + 1,
        )
        self.last_restore_status = "legacy_migrated" if version == 1 else "restored"
        return len(self)
