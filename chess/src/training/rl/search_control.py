"""Regret-guided opening archive used by RL self-play."""

from __future__ import annotations

import math

import numpy as np


SEARCH_CONTROL_STATE_VERSION = 1


def fen_is_exact_restart(fen):
    """A zero halfmove clock makes FEN-only repetition reconstruction exact."""
    try:
        parts = str(fen or "").split()
        return len(parts) >= 6 and int(parts[4]) == 0
    except (TypeError, ValueError):
        return False


def suffix_regret_targets(game_history, outcome):
    """Return RGSC Eq. 2 regret for every trajectory row.

    Stored ``played_q`` is from the side-to-move point of view, therefore the
    terminal result is signed separately for every row. Missing/zero-visit Q
    estimates do not enter the suffix average.
    """
    regrets = [float("nan")] * len(game_history)
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
        if valid_count > 0:
            regrets[idx] = error_sum / float(valid_count)
    return regrets


class RegretOpeningArchive:
    """Fixed-capacity, deduplicated prioritized regret buffer (PRB)."""

    def __init__(self, capacity=128, temperature=0.10, ema_alpha=0.50):
        self.capacity = max(1, int(capacity))
        self.temperature = max(1e-3, float(temperature))
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._entries = {}
        self._key_to_id = {}
        self._next_id = 1

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _key(fen, history_fens):
        return str(fen), tuple(str(item) for item in (history_fens or ()))

    def add_candidates(self, candidates, iteration):
        added = updated = rejected = 0
        for candidate in candidates or ():
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
                if regret > float(entry["regret"]):
                    entry["regret"] = regret
                    entry["updated_iteration"] = int(iteration)
                    updated += 1
                continue
            if len(self._entries) >= self.capacity:
                lowest_id, lowest = min(
                    self._entries.items(), key=lambda item: (float(item[1]["regret"]), int(item[0]))
                )
                if regret <= float(lowest["regret"]):
                    rejected += 1
                    continue
                self._key_to_id.pop(self._key(lowest["fen"], lowest["history_fens"]), None)
                del self._entries[lowest_id]
            entry_id = self._next_id
            self._next_id += 1
            self._entries[entry_id] = {
                "id": entry_id,
                "fen": str(fen),
                "history_fens": history_fens,
                "regret": regret,
                "inserted_iteration": int(iteration),
                "updated_iteration": int(iteration),
                "replays": 0,
            }
            self._key_to_id[key] = entry_id
            added += 1
        return {"added": added, "updated": updated, "rejected": rejected}

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

    def sample(self, count, *, rng=None):
        count = min(max(0, int(count)), len(self._entries))
        if count <= 0:
            return []
        rng = rng if rng is not None else np.random
        entries = sorted(self._entries.values(), key=lambda item: int(item["id"]))
        scores = np.asarray([max(1e-12, float(item["regret"])) for item in entries])
        log_weights = np.log(scores) / self.temperature
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        weights /= float(weights.sum())
        selected = rng.choice(len(entries), size=count, replace=False, p=weights)
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
        return {
            "search_control_archive_size": len(self),
            "search_control_regret_mean": float(scores.mean()) if scores.size else 0.0,
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
            "temperature": self.temperature,
            "ema_alpha": self.ema_alpha,
            "next_id": self._next_id,
            "entries": [dict(entry) for entry in self._entries.values()],
        }

    def load_state_dict(self, state):
        if not state:
            return 0
        if int(state.get("format_version", 0) or 0) != SEARCH_CONTROL_STATE_VERSION:
            raise ValueError("Unsupported search-control archive state version.")
        self.capacity = max(1, int(state.get("capacity", self.capacity)))
        self.temperature = max(1e-3, float(state.get("temperature", self.temperature)))
        self.ema_alpha = max(0.0, min(1.0, float(state.get("ema_alpha", self.ema_alpha))))
        self._entries = {}
        self._key_to_id = {}
        for raw in state.get("entries", ()):
            entry = dict(raw)
            entry_id = int(entry["id"])
            entry["history_fens"] = tuple(entry.get("history_fens", ()) or ())
            if not fen_is_exact_restart(entry.get("fen")):
                continue
            self._entries[entry_id] = entry
            self._key_to_id[self._key(entry["fen"], entry["history_fens"])] = entry_id
        self._next_id = max(
            int(state.get("next_id", 1) or 1),
            max(self._entries, default=0) + 1,
        )
        return len(self)
