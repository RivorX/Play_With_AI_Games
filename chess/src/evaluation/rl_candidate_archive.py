"""Bounded, immutable archive for selector-qualified RL checkpoints.

Archived candidates are diagnostic artifacts. Their MCTS selector evidence
does not promote them or replace the independent confirmation contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping

import torch


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_run_identity(manifest_path: str | os.PathLike) -> dict:
    """Freeze checkpoint, source, seed, and config identities from run manifest."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("effective_config", {})
    return {
        "run_manifest_path": str(path),
        "run_manifest_sha256_at_start": sha256_file(path),
        "initial_checkpoint_path": manifest.get("checkpoint_path"),
        "initial_checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "seed": manifest.get("seed"),
        "git_commit": manifest.get("git_commit"),
        "source_tree_sha256": manifest.get("source_tree_sha256"),
        "effective_config_sha256": json_sha256(config),
    }


def restore_candidate_archive_identity(
    manifest_path: str | os.PathLike,
    *,
    saved_identity: Mapping | None = None,
    index_path: str | os.PathLike | None = None,
    allow_source_amendment: bool = False,
    source_amendment_reason: str | None = None,
) -> dict:
    """Reuse the original archive identity on resume after validating the run.

    Run manifests gain mutable runtime fields and resume manifests reference the
    latest checkpoint, so their whole-file hash and initial checkpoint hash are
    expected to differ. Stable source/config/seed fields must still match.
    """
    current = immutable_run_identity(manifest_path)
    prior = dict(saved_identity or {})
    if not prior and index_path is not None:
        path = Path(index_path)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            prior = dict(payload.get("run_identity", {}) or {})
    if not prior:
        return current
    mismatches = {
        key: (prior.get(key), current.get(key))
        for key in ("seed", "git_commit", "source_tree_sha256", "effective_config_sha256")
        if prior.get(key) != current.get(key)
    }
    if set(mismatches) == {"source_tree_sha256"} and allow_source_amendment:
        reason = str(source_amendment_reason or "").strip()
        if not reason:
            raise ValueError(
                "A non-empty reason is required for an RL resume source amendment."
            )
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        amendment_path = manifest_path.with_name(
            f"{manifest_path.stem}_source_amendments.jsonl"
        )
        amendment = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason,
            "changed_fields": ["source_tree_sha256"],
            "prior_identity_sha256": json_sha256(prior),
            "prior_source_tree_sha256": prior["source_tree_sha256"],
            "current_source_tree_sha256": current["source_tree_sha256"],
            "current_source_tree_file_count": manifest.get("source_tree_file_count"),
            "current_git_commit": current.get("git_commit"),
            "current_seed": current.get("seed"),
            "current_effective_config_sha256": current.get("effective_config_sha256"),
            "current_manifest_path": str(manifest_path),
            "current_manifest_sha256": current.get("run_manifest_sha256_at_start"),
            "continuation_checkpoint_path": current.get("initial_checkpoint_path"),
            "continuation_checkpoint_sha256": current.get("initial_checkpoint_sha256"),
            "archive_identity_preserved": True,
            "archive_identity": prior,
        }
        line = json.dumps(
            amendment, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ) + "\n"
        with amendment_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return prior

    for key in (
        "effective_config_sha256",
        "seed",
        "git_commit",
        "source_tree_sha256",
    ):
        if key not in mismatches:
            continue
        archived_value, current_value = mismatches[key]
        if prior.get(key) != current.get(key):
            raise RuntimeError(
                "RL candidate archive cannot resume with a different "
                f"{key}: archived={archived_value!r}, current={current_value!r}"
            )
    return prior


def model_state_sha256(model) -> str:
    """Hash a model state dict independently of device and checkpoint metadata."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        raw = value.reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


class MctsCandidateArchive:
    """Keep the latest top-K MCTS-qualified checkpoints and every outcome.

    The JSON index records every selector-qualified iteration. At most ``top_k``
    checkpoint files are retained; evicted history remains as metadata only.
    The run identity is frozen at construction, and each checkpoint's SHA is
    checked before the outcome record can be updated.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        directory: str | os.PathLike,
        *,
        top_k: int = 3,
        run_identity: Mapping | None = None,
    ):
        self.directory = Path(directory).resolve()
        self.index_path = self.directory / "index.json"
        self.top_k = max(1, int(top_k))
        self.run_identity = dict(run_identity or {})
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.index_path.is_file():
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", 0)) != self.SCHEMA_VERSION:
                raise RuntimeError("Unsupported RL candidate archive schema")
            if payload.get("run_identity") != self.run_identity:
                raise RuntimeError(
                    "RL candidate archive run identity differs from the current run"
                )
            self.records = dict(payload.get("records", {}))
        else:
            self.records = {}
            self._write_index()

    def _write_index(self):
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "top_k": self.top_k,
            "run_identity": self.run_identity,
            "records": self.records,
        }
        temp_path = self.index_path.with_name(
            f"{self.index_path.name}.tmp-{os.getpid()}"
        )
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temp_path, self.index_path)

    def _retained(self):
        return [
            record for record in self.records.values()
            if record.get("retained") and record.get("checkpoint_path")
        ]

    def consider(
        self,
        *,
        iteration: int,
        mcts_score_rate: float,
        mcts_lower_bound: float,
        mcts_score_se: float | None,
        mcts_games: int,
        unique_openings: int,
        simulation_budget: int,
        opponent_identity: Mapping,
        checkpoint_metadata: Mapping,
        write_checkpoint: Callable[[Path], None],
    ) -> dict:
        iteration = int(iteration)
        record_id = f"iteration_{iteration:04d}"
        if record_id in self.records:
            existing = self.records[record_id]
            if existing.get("retained"):
                return {"status": "already_archived", **existing}

        candidate = {
            "iteration": iteration,
            "retained": False,
            "status": "selector-qualified; archive decision pending",
            "mcts_score_rate": float(mcts_score_rate),
            "mcts_lower_bound": float(mcts_lower_bound),
            "mcts_score_se": None if mcts_score_se is None else float(mcts_score_se),
            "mcts_games": int(mcts_games),
            "mcts_unique_openings": int(unique_openings),
            "mcts_simulations": int(simulation_budget),
            "opponent_identity": dict(opponent_identity or {}),
            "run_identity": self.run_identity,
            "checkpoint_metadata": dict(checkpoint_metadata or {}),
        }
        retained = self._retained()
        evicted = None
        can_retain = len(retained) < self.top_k
        if not can_retain:
            # Evaluations may have different opponents after promotion. Keep
            # the latest three qualified checkpoints rather than ranking scores
            # across those incomparable baselines.
            oldest = min(retained, key=lambda item: int(item.get("iteration", 0)))
            can_retain = iteration > int(oldest.get("iteration", 0))
            if can_retain:
                evicted = oldest

        if can_retain:
            checkpoint_path = self.directory / f"candidate_{iteration:04d}.pt"
            write_checkpoint(checkpoint_path)
            candidate["checkpoint_path"] = checkpoint_path.name
            candidate["checkpoint_sha256"] = sha256_file(checkpoint_path)
            candidate["retained"] = True
            candidate["status"] = "selector-qualified; awaiting downstream gates"
            if evicted is not None:
                evicted_id = f"iteration_{int(evicted['iteration']):04d}"
                old_path = self.directory / str(evicted.get("checkpoint_path", ""))
                # Only unlink a file owned by this archive directory.
                try:
                    if old_path.resolve().parent == self.directory and old_path.is_file():
                        old_path.unlink()
                except OSError:
                    pass
                old_record = self.records[evicted_id]
                old_record.update({
                    "retained": False,
                    "status": f"evicted: superseded by {record_id} under latest-{self.top_k} retention",
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                })
        else:
            candidate["status"] = f"selector-qualified; not retained outside latest-{self.top_k}"
        self.records[record_id] = candidate
        self._write_index()
        return {"status": candidate["status"], **candidate}

    def record_outcomes(self, iteration: int, outcomes: Mapping) -> bool:
        record_id = f"iteration_{int(iteration):04d}"
        record = self.records.get(record_id)
        if record is None:
            return False
        relative = record.get("checkpoint_path")
        if record.get("retained") and relative:
            checkpoint_path = (self.directory / str(relative)).resolve()
            if checkpoint_path.parent != self.directory or not checkpoint_path.is_file():
                raise RuntimeError("Archived RL candidate checkpoint is missing or escaped its archive")
            if sha256_file(checkpoint_path) != record.get("checkpoint_sha256"):
                raise RuntimeError("Archived RL candidate checkpoint changed after capture")
        record["outcomes"] = dict(outcomes or {})
        if outcomes and outcomes.get("promoted"):
            record["status"] = "promoted after all configured gates"
        elif outcomes:
            record["status"] = str(outcomes.get("promotion_status") or "blocked after selector")
        self._write_index()
        return True

    def retained_count(self) -> int:
        return len(self._retained())
