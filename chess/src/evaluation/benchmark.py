"""Reusable, paired full-game benchmarks for Elo-compatible MCTS runtimes.

This module shares the same Elo configuration builder, ``EloEstimator`` runtime
factory, ``_BatchedModelPlayer``, ``MultiGameBatchMCTS`` and ``SearchSession``
used by ``scripts/eval_elo.py``. Local CUDA inference stays on one owner thread;
multi-worker CUDA runs use Elo's central GPU server and per-thread CPU clients.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import chess
import chess.pgn
import torch

from src.config.normalization import normalize_rl_config
from src.evaluation.elo_estimator import (
    _BatchedModelPlayer,
    _EVAL_OPENING_LINES,
    _EVAL_OPENING_SUITE_ID,
    _build_elo_mcts_config,
)
from src.evaluation.elo_runner import (
    build_eval_elo_runtime_config,
    create_elo_estimator,
)


_SCHEMA_VERSION = 1
_OPENING_SUITE_NAME = "elo_first_10"
_ARTIFACT_NAMES = (
    "benchmark_manifest.json",
    "benchmark_progress.json",
    "benchmark_games.pgn",
    "benchmark_summary.json",
)
_SEARCH_OVERRIDE_ALIASES = {
    "search_efficiency_feedback_enabled": "mcts_search_efficiency_feedback_enabled",
    "dynamic_budget_enabled": "mcts_dynamic_budget_enabled",
    "dynamic_budget_strategy": "mcts_dynamic_budget_strategy",
    "dynamic_budget_min": "mcts_dynamic_budget_min",
    "dynamic_budget_max_multiplier": "mcts_dynamic_budget_max_multiplier",
    "gumbel_root_schedule": "mcts_gumbel_root_schedule",
    "gumbel_reuse_q_scale": "mcts_gumbel_reuse_q_scale",
    "gumbel_c_visit": "mcts_gumbel_c_visit",
    "gumbel_c_scale": "mcts_gumbel_c_scale",
    "gumbel_q_range_floor": "mcts_gumbel_q_range_floor",
    "gumbel_q_scale_visit_knee": "mcts_gumbel_q_scale_visit_knee",
    "gumbel_q_scale_visit_exponent": "mcts_gumbel_q_scale_visit_exponent",
    "reuse_tree": "mcts_reuse_tree",
    "tree_reuse_visit_credit_enabled": "mcts_tree_reuse_visit_credit_enabled",
    "local_cuda_graph_forward": "mcts_local_cuda_graph_forward",
}
_CENTRAL_INFERENCE_OVERRIDE_KEYS = {
    "cuda_graph_forward_enabled",
    "cuda_graph_forward_min_observations",
    "cuda_graph_forward_hot_shapes",
    "profile_cuda_graphs",
}
_CODE_FINGERPRINT_FILES = (
    "src/evaluation/benchmark.py",
    "src/evaluation/elo_runner.py",
    "src/evaluation/elo_estimator.py",
    "src/evaluation/central_inference.py",
    "src/mcts/search.py",
    "src/mcts/native.py",
    "src/inference/central.py",
    "src/mcts/cpp/mcts_kernels.cpp",
    "src/config/normalization.py",
    "src/training/rl/trainer.py",
    "scripts/eval_elo.py",
)


def _as_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _as_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _jsonable(value: Any):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_fingerprints() -> dict[str, str]:
    chess_root = Path(__file__).resolve().parents[2]
    hashes = {}
    for relative in _CODE_FINGERPRINT_FILES:
        source = chess_root / relative
        if source.is_file():
            hashes[relative] = _sha256_file(source)
    return hashes


def _normalised_config(config: Mapping[str, Any]) -> dict:
    resolved = copy.deepcopy(dict(config or {}))
    normalize_rl_config(resolved)
    resolved.setdefault("reinforcement_learning", {})
    return resolved


def _normalise_search_override_key(key: str) -> str:
    key = str(key).strip()
    if key.startswith("reinforcement_learning."):
        key = key[len("reinforcement_learning."):]
    if key.startswith("search."):
        key = key[len("search."):]
    if key in _SEARCH_OVERRIDE_ALIASES:
        return _SEARCH_OVERRIDE_ALIASES[key]
    if key.startswith("gumbel_"):
        return f"mcts_{key}"
    if key.startswith("dynamic_budget_"):
        return f"mcts_{key}"
    if key.startswith("mcts_"):
        return key
    raise ValueError(
        f"Unsupported search override {key!r}; use a canonical mcts_* key or a supported search key"
    )


def _apply_search_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict:
    resolved = _normalised_config(config)
    if not overrides:
        return resolved
    rl_config = dict(resolved.get("reinforcement_learning", {}) or {})
    for key, value in dict(overrides).items():
        canonical = _normalise_search_override_key(str(key))
        if canonical not in rl_config:
            raise ValueError(f"Unknown normalized search setting: {canonical}")
        rl_config[canonical] = copy.deepcopy(value)
    resolved["reinforcement_learning"] = rl_config
    normalize_rl_config(resolved)
    return resolved


def _apply_central_inference_overrides(
    config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict, dict]:
    """Apply benchmark-only central CUDA graph settings and resolve exclusivity."""
    resolved = _normalised_config(config)
    if not overrides:
        return resolved, {}
    if not isinstance(overrides, Mapping):
        raise ValueError("central_inference_overrides must be a mapping")
    unknown = set(overrides) - _CENTRAL_INFERENCE_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            "unsupported central_inference_overrides: " + ", ".join(sorted(map(str, unknown)))
        )
    effective = copy.deepcopy(dict(overrides))
    for key in ("cuda_graph_forward_enabled", "profile_cuda_graphs"):
        if key in effective and type(effective[key]) is not bool:
            raise ValueError(f"central_inference_overrides.{key} must be true or false")
    if "cuda_graph_forward_min_observations" in effective:
        effective["cuda_graph_forward_min_observations"] = _as_positive_int(
            effective["cuda_graph_forward_min_observations"],
            "central_inference_overrides.cuda_graph_forward_min_observations",
        )
    if "cuda_graph_forward_hot_shapes" in effective:
        shapes = effective["cuda_graph_forward_hot_shapes"]
        if not isinstance(shapes, Mapping):
            raise ValueError("central_inference_overrides.cuda_graph_forward_hot_shapes must be a mapping")
        normalized_shapes = {}
        for architecture, sizes in shapes.items():
            if isinstance(sizes, (str, bytes)) or not isinstance(sizes, Sequence) or not sizes:
                raise ValueError(f"hot shapes for {architecture!r} must be a non-empty sequence")
            normalized_shapes[str(architecture)] = [
                _as_positive_int(size, f"hot shape for {architecture}") for size in sizes
            ]
        effective["cuda_graph_forward_hot_shapes"] = normalized_shapes

    central = dict(resolved.get("central_inference", {}) or {})
    central.update(copy.deepcopy(effective))
    if effective.get("cuda_graph_forward_enabled"):
        # The central server cannot run CUDA graph capture and torch.compile
        # at once. Record the effective backend directly in the server config.
        central["use_compile"] = False
    resolved["central_inference"] = central
    return resolved, effective


def _variant_overrides_to_search(variant: Mapping[str, Any]) -> dict:
    raw = variant.get("config_overrides", {}) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("variant config_overrides must be a mapping")
    if not raw:
        return {}
    if set(raw) - {"reinforcement_learning"}:
        raise ValueError(
            "config_overrides may only change reinforcement_learning search settings; "
            "put common model/central-inference settings in config before calling the benchmark"
        )
    rl_override = raw.get("reinforcement_learning", {}) or {}
    if not isinstance(rl_override, Mapping):
        raise ValueError("reinforcement_learning config_overrides must be a mapping")
    search_override = rl_override.get("search", {}) or {}
    if not isinstance(search_override, Mapping):
        raise ValueError("reinforcement_learning.search override must be a mapping")
    direct_override = {key: value for key, value in rl_override.items() if key != "search"}
    if any(not str(key).startswith("mcts_") for key in direct_override):
        raise ValueError(
            "direct reinforcement_learning config_overrides must use canonical mcts_* keys"
        )
    combined = dict(search_override)
    combined.update(direct_override)
    return combined


def _resolve_profiles(
    simulations: Sequence[int],
    variants: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None,
    search_overrides: Mapping[str, Any] | None,
) -> list[dict]:
    if variants is None:
        profiles = [
            {"name": f"mcts-{_as_positive_int(value, 'simulation budget')}", "simulations": int(value), "search_overrides": {}}
            for value in simulations
        ]
    else:
        if isinstance(variants, Mapping):
            source = []
            for name, specification in variants.items():
                if not isinstance(specification, Mapping):
                    raise ValueError(f"variant {name!r} must be a mapping")
                source.append({"name": str(name), **dict(specification)})
        else:
            source = list(variants)
        profiles = []
        for index, raw in enumerate(source):
            if not isinstance(raw, Mapping):
                raise ValueError("each variant must be a mapping")
            simulations_value = raw.get("simulations")
            if simulations_value is None:
                raise ValueError(f"variant {index} is missing simulations")
            profile_name = str(raw.get("name") or f"mcts-{simulations_value}").strip()
            if not profile_name:
                raise ValueError(f"variant {index} has an empty name")
            variant_search = _variant_overrides_to_search(raw)
            profiles.append({
                "name": profile_name,
                "simulations": _as_positive_int(simulations_value, "variant simulations"),
                "search_overrides": variant_search,
            })
    if not profiles:
        raise ValueError("at least one benchmark simulation profile is required")
    names = [profile["name"] for profile in profiles]
    if len(set(names)) != len(names):
        raise ValueError("benchmark variant names must be unique")
    common_search = dict(search_overrides or {})
    for profile in profiles:
        combined = dict(common_search)
        combined.update(profile["search_overrides"])
        profile["search_overrides"] = combined
    return profiles


def _opening_lines(opening_suite: str, opening_max_plies: int) -> list[list[str]]:
    suite = str(opening_suite or _OPENING_SUITE_NAME).strip().lower()
    if suite not in {_OPENING_SUITE_NAME, "elo_first10", "elo-first-10"}:
        raise ValueError(f"unsupported opening suite {opening_suite!r}; expected {_OPENING_SUITE_NAME!r}")
    max_plies = _as_nonnegative_int(opening_max_plies, "opening_max_plies")
    return [list(line[:max_plies]) for line in _EVAL_OPENING_LINES[:10]]


def build_benchmark_plan(
    *,
    baseline: int = 192,
    simulations: Sequence[int] = (384, 768, 1536),
    games_per_profile: int = 20,
    opening_suite: str = _OPENING_SUITE_NAME,
    opening_max_plies: int = 8,
    variants: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    search_overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Build deterministic full-game keys and shared opening/color pairings."""
    baseline = _as_positive_int(baseline, "baseline")
    games_per_profile = _as_positive_int(games_per_profile, "games_per_profile")
    if games_per_profile % 2:
        raise ValueError("games_per_profile must be even so each opening is played with both colors")
    profiles = _resolve_profiles(simulations, variants, search_overrides)
    openings = _opening_lines(opening_suite, opening_max_plies)
    if len(openings) != 10:
        raise ValueError("the Elo first-ten opening suite must contain exactly ten positions")

    games = []
    for profile in profiles:
        for local_index in range(games_per_profile):
            opening_index = (local_index // 2) % len(openings)
            games.append({
                "game_id": f"{profile['name']}:{local_index:03d}",
                "profile_name": profile["name"],
                "profile_simulations": int(profile["simulations"]),
                "baseline_simulations": baseline,
                "profile_game_index": int(local_index),
                "opening_pair_index": int(local_index // 2),
                "opening_index": int(opening_index),
                "opening_uci": list(openings[opening_index]),
                "profile_is_white": bool(local_index % 2 == 0),
            })

    return {
        "schema_version": _SCHEMA_VERSION,
        "opening_suite": _OPENING_SUITE_NAME,
        "opening_suite_id": f"{_EVAL_OPENING_SUITE_ID}:first-10",
        "opening_max_plies": int(opening_max_plies),
        "openings": openings,
        "baseline_simulations": baseline,
        "games_per_profile": games_per_profile,
        "profiles": profiles,
        "games": games,
    }


def _scenario_project_config(config, profile_search_overrides):
    return _apply_search_overrides(config, profile_search_overrides)


def _build_runtime_scenarios(
    config: Mapping[str, Any],
    *,
    device_type: str,
    baseline: int,
    profiles: list[dict],
    requested_workers,
    central_inference_overrides: Mapping[str, Any] | None = None,
) -> dict:
    normalized, effective_central_overrides = _apply_central_inference_overrides(
        config, central_inference_overrides
    )
    elo_config = dict(normalized.get("elo_estimation", {}) or {})
    scenarios = [{
        "name": "baseline",
        "simulations": int(baseline),
        "project_config": _scenario_project_config(
            normalized,
            profiles[0].get("common_search_overrides", {}),
        ),
    }]
    for profile in profiles:
        scenarios.append({
            "name": profile["name"],
            "simulations": int(profile["simulations"]),
            "project_config": _scenario_project_config(
                normalized,
                profile["search_overrides"],
            ),
        })

    # Baseline receives common overrides only; variant-specific overrides
    # remain local to that variant.
    common_search = profiles[0].get("common_search_overrides", {})
    scenarios[0]["project_config"] = _scenario_project_config(normalized, common_search)
    for scenario in scenarios:
        runtime = build_eval_elo_runtime_config(
            scenario["project_config"],
            elo_config,
            device=SimpleNamespace(type=str(device_type)),
            use_mcts=True,
            simulations=scenario["simulations"],
            budget_mode="simulations",
            workers=requested_workers,
        )
        scenario["elo_runtime_config"] = runtime
        if effective_central_overrides.get("cuda_graph_forward_enabled"):
            # Central CUDA graphs and torch.compile are mutually exclusive.
            runtime["eval_elo_central_inference_use_compile"] = False
        scenario["search_runtime_config"] = _build_elo_mcts_config(
            scenario["project_config"], runtime
        )
    return {
        "scenarios": scenarios,
        "requested_workers": scenarios[0]["elo_runtime_config"].get("workers", 0),
        "device_type": str(device_type),
    }


def resolve_benchmark_runtime_plan(
    config: Mapping[str, Any],
    *,
    device_type: str,
    baseline: int = 192,
    simulations: Sequence[int] = (384, 768, 1536),
    games_per_profile: int = 20,
    workers=None,
    opening_suite: str = _OPENING_SUITE_NAME,
    variants: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    search_overrides: Mapping[str, Any] | None = None,
    central_inference_overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Resolve Elo search/backend settings without loading weights or a GPU."""
    normalized, effective_central_overrides = _apply_central_inference_overrides(
        config, central_inference_overrides
    )
    elo_config = dict(normalized.get("elo_estimation", {}) or {})
    opening_max_plies = _as_nonnegative_int(
        elo_config.get("paired_openings_max_plies", 8),
        "paired_openings_max_plies",
    )
    plan = build_benchmark_plan(
        baseline=baseline,
        simulations=simulations,
        games_per_profile=games_per_profile,
        opening_suite=opening_suite,
        opening_max_plies=opening_max_plies,
        variants=variants,
        search_overrides=search_overrides,
    )
    profiles = plan["profiles"]
    common_search = dict(search_overrides or {})
    for profile in profiles:
        profile["common_search_overrides"] = common_search
    runtime = _build_runtime_scenarios(
        normalized,
        device_type=device_type,
        baseline=plan["baseline_simulations"],
        profiles=profiles,
        requested_workers=workers,
        central_inference_overrides=effective_central_overrides,
    )
    scenarios = runtime["scenarios"]
    requested = int(runtime["requested_workers"] or 0)
    central_possible = bool(
        str(device_type) == "cuda"
        and requested > 1
        and scenarios[0]["elo_runtime_config"].get("eval_elo_central_inference_enabled", False)
        and (
            not scenarios[0]["elo_runtime_config"].get("mcts_dynamic_budget_enabled", False)
            or str(scenarios[0]["elo_runtime_config"].get("mcts_dynamic_budget_strategy", "feedback")) == "feedback"
        )
    )
    server_cfg = dict((scenarios[0]["search_runtime_config"].get("central_inference", {}) or {}))
    local_graph = (scenarios[0]["search_runtime_config"].get("reinforcement_learning", {}) or {}).get(
        "mcts_local_cuda_graph_forward"
    )
    return {
        "plan": plan,
        "requested_workers": requested,
        "effective_workers": None if requested <= 0 else min(requested, len(plan["games"])),
        "worker_resolution": "runtime Elo resolver" if requested <= 0 else "explicit/configured",
        "inference_route": (
            "central_gpu_with_remote_cpu_workers"
            if central_possible
            else "local_single_owner"
            if str(device_type) == "cuda"
            else "local_cpu_workers"
        ),
        "central_inference_requested": bool(
            scenarios[0]["elo_runtime_config"].get("eval_elo_central_inference_enabled", False)
        ),
        "central_use_compile": bool(server_cfg.get("use_compile", False)),
        "central_cuda_graph_forward_enabled": bool(server_cfg.get("cuda_graph_forward_enabled", False)),
        "central_cuda_graph_hot_shapes": _jsonable(server_cfg.get("cuda_graph_forward_hot_shapes", {})),
        "central_cuda_graph_min_observations": int(
            server_cfg.get("cuda_graph_forward_min_observations", 4) or 4
        ),
        "central_profile_cuda_graphs": bool(server_cfg.get("profile_cuda_graphs", False)),
        "central_inference_overrides": _jsonable(effective_central_overrides),
        "local_cuda_graph_policy": "auto" if local_graph is None else bool(local_graph),
        "scenarios": [
            {
                "name": item["name"],
                "simulations": item["simulations"],
                "elo_runtime_config": _jsonable(item["elo_runtime_config"]),
                "search_runtime_config": _jsonable(item["search_runtime_config"]),
            }
            for item in scenarios
        ],
    }


def _manifest_fingerprint(manifest_material: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(manifest_material).encode("utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_pgn(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            pgn = str(record.get("pgn", "")).strip()
            if pgn:
                handle.write(pgn)
                handle.write("\n\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _profile_summary(plan: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[dict]:
    summaries = []
    for profile in plan["profiles"]:
        games = [record for record in records if record.get("profile_name") == profile["name"]]
        wins = sum(record.get("score") == 1.0 for record in games)
        draws = sum(record.get("score") == 0.5 for record in games)
        losses = sum(record.get("score") == 0.0 for record in games)
        scored = sum(float(record.get("score", 0.0)) for record in games)
        total = len(games)
        profile_fresh_sims = sum(
            int(record.get("profile_fresh_simulations", 0) or 0) for record in games
        )
        profile_search_calls = sum(
            int(record.get("profile_search_calls", 0) or 0) for record in games
        )
        baseline_fresh_sims = sum(
            int(record.get("baseline_fresh_simulations", 0) or 0) for record in games
        )
        baseline_search_calls = sum(
            int(record.get("baseline_search_calls", 0) or 0) for record in games
        )
        summaries.append({
            "name": profile["name"],
            "simulations": int(profile["simulations"]),
            "baseline_simulations": int(plan["baseline_simulations"]),
            "games_completed": total,
            "games_expected": int(plan["games_per_profile"]),
            "wins": int(wins),
            "draws": int(draws),
            "losses": int(losses),
            "score_points": scored,
            "score_fraction": (scored / total) if total else None,
            "mean_fresh_simulations_profile": (
                profile_fresh_sims / profile_search_calls
                if profile_search_calls else None
            ),
            "mean_fresh_simulations_baseline": (
                baseline_fresh_sims / baseline_search_calls
                if baseline_search_calls else None
            ),
        })
    return summaries


def _summary_payload(manifest, progress, plan, runtime_info, paths):
    records = sorted(
        list(progress.get("completed_games", {}).values()),
        key=lambda record: int(record.get("plan_index", 0)),
    )
    status = str(progress.get("status", "partial"))
    summary_records = [{key: value for key, value in record.items() if key != "pgn"} for record in records]
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "created_at_utc": manifest.get("created_at_utc"),
        "updated_at_utc": progress.get("updated_at_utc"),
        "fingerprint": manifest["fingerprint"],
        "checkpoint": manifest["checkpoint"],
        "plan": {
            "baseline_simulations": plan["baseline_simulations"],
            "games_per_profile": plan["games_per_profile"],
            "opening_suite_id": plan["opening_suite_id"],
            "opening_max_plies": plan["opening_max_plies"],
            "profiles": plan["profiles"],
            "games_expected": len(plan["games"]),
        },
        "runtime": runtime_info,
        "profiles": _profile_summary(plan, records),
        "games_completed": len(records),
        "games_expected": len(plan["games"]),
        "games": summary_records,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def _make_game_pgn(board, task, result, elapsed_seconds, search_stats):
    game = chess.pgn.Game.from_board(board)
    profile_name = str(task["profile_name"])
    profile_label = f"{profile_name} ({task['profile_simulations']} sims)"
    baseline_label = f"baseline ({task['baseline_simulations']} sims)"
    game.headers["Event"] = "Gumbel simulation budget benchmark"
    game.headers["Site"] = "local"
    game.headers["Round"] = f"{profile_name}-{int(task['profile_game_index']) + 1}"
    game.headers["White"] = profile_label if task["profile_is_white"] else baseline_label
    game.headers["Black"] = baseline_label if task["profile_is_white"] else profile_label
    game.headers["Result"] = result
    game.headers["OpeningSuite"] = _EVAL_OPENING_SUITE_ID
    game.headers["OpeningIndex"] = str(int(task["opening_index"]))
    game.headers["ProfileFreshSims"] = str(int(search_stats["profile_fresh_simulations"]))
    game.headers["BaselineFreshSims"] = str(int(search_stats["baseline_fresh_simulations"]))
    game.headers["ElapsedSeconds"] = f"{float(elapsed_seconds):.3f}"
    return game.accept(chess.pgn.StringExporter(headers=True, variations=True, comments=True))


def _play_full_game(task, profile_player, baseline_player, cancel_event):
    started = time.perf_counter()
    board = chess.Board()
    states = {
        "profile": profile_player.create_state(),
        "baseline": baseline_player.create_state(),
    }
    profile_white = bool(task["profile_is_white"])
    players = {"profile": profile_player, "baseline": baseline_player}
    try:
        for uci in task["opening_uci"]:
            if cancel_event.is_set():
                return None
            if board.is_game_over(claim_draw=True):
                break
            move = chess.Move.from_uci(str(uci))
            if move not in board.legal_moves:
                raise ValueError(f"opening suite contains illegal move {uci!r}")
            for key, player in players.items():
                player.record_state(states[key], board)
            board.push(move)
            for key, player in players.items():
                player.on_move_played(states[key], move)

        while not board.is_game_over(claim_draw=True):
            if cancel_event.is_set():
                return None
            side_key = "profile" if (board.turn == chess.WHITE) == profile_white else "baseline"
            player = players[side_key]
            move = player.best_moves([board], [states[side_key]])[0]
            if move is None or move not in board.legal_moves:
                raise RuntimeError(
                    f"{side_key} player returned an illegal or missing move at ply {board.ply()}"
                )
            for key, state_player in players.items():
                state_player.record_state(states[key], board)
            board.push(move)
            for key, state_player in players.items():
                state_player.on_move_played(states[key], move)

        result = board.result(claim_draw=True)
        if result not in {"1-0", "0-1", "1/2-1/2"}:
            raise RuntimeError(f"game ended without a terminal chess result: {result!r}")
        score = (
            1.0 if (result == "1-0") == profile_white else 0.0
        ) if result in {"1-0", "0-1"} else 0.5
        elapsed = time.perf_counter() - started
        profile_state = states["profile"]
        baseline_state = states["baseline"]
        search_stats = {
            "profile_search_calls": int(profile_state.get("mcts_search_calls", 0) or 0),
            "profile_fresh_simulations": int(profile_state.get("mcts_fresh_simulations", 0) or 0),
            "baseline_search_calls": int(baseline_state.get("mcts_search_calls", 0) or 0),
            "baseline_fresh_simulations": int(baseline_state.get("mcts_fresh_simulations", 0) or 0),
        }
        pgn = _make_game_pgn(board, task, result, elapsed, search_stats)
        return {
            **dict(task),
            "result": result,
            "score": float(score),
            "plies": int(board.ply()),
            "elapsed_seconds": float(elapsed),
            **search_stats,
            "invalid_move_fallbacks": 0,
            "pgn": pgn,
        }
    finally:
        for key, player in players.items():
            state = states.get(key)
            if state is not None:
                player.close_state(state)


def _player_graph_stats(player):
    engine = getattr(player, "multi_mcts", None)
    getter = getattr(engine, "local_cuda_graph_forward_stats", None)
    return _jsonable(getter()) if callable(getter) else None


def _dispose_players(players):
    for player in players.values():
        engine = getattr(player, "multi_mcts", None)
        if engine is None:
            continue
        clear_graph = getattr(engine, "clear_local_cuda_graph_forward", None)
        if callable(clear_graph):
            clear_graph()
        forest = getattr(engine, "_native_forest", None)
        if forest is not None:
            forest.close()
            engine._native_forest = None


def run_benchmark(
    config: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    device: torch.device,
    checkpoint: Path,
    baseline: int = 192,
    simulations: Sequence[int] = (384, 768, 1536),
    games_per_profile: int = 20,
    workers: int | None = None,
    output_dir: Path,
    resume: bool = False,
    opening_suite: str = _OPENING_SUITE_NAME,
    variants: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] | None = None,
    search_overrides: Mapping[str, Any] | None = None,
    central_inference_overrides: Mapping[str, Any] | None = None,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run paired full games and checkpoint terminal results for safe resume.

    ``model`` and ``device`` must come from the same checkpoint loader and
    device selection used by ``eval_elo``. A local CUDA model is accessed from
    one thread only. Multi-worker CUDA runs follow Elo's central-inference
    policy and each worker receives its own CPU ``RemoteInferenceClient`` and
    independent per-game/per-side ``SearchSession`` objects.
    """
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {Path(name).stem.removeprefix("benchmark_"): output_dir / name for name in _ARTIFACT_NAMES}
    manifest_path = output_dir / "benchmark_manifest.json"
    progress_path = output_dir / "benchmark_progress.json"
    pgn_path = output_dir / "benchmark_games.pgn"
    summary_path = output_dir / "benchmark_summary.json"

    normalized, effective_central_overrides = _apply_central_inference_overrides(
        config, central_inference_overrides
    )
    elo_config = dict(normalized.get("elo_estimation", {}) or {})
    opening_max_plies = _as_nonnegative_int(
        elo_config.get("paired_openings_max_plies", 8),
        "paired_openings_max_plies",
    )
    plan = build_benchmark_plan(
        baseline=baseline,
        simulations=simulations,
        games_per_profile=games_per_profile,
        opening_suite=opening_suite,
        opening_max_plies=opening_max_plies,
        variants=variants,
        search_overrides=search_overrides,
    )
    # Ensure a variant's local overrides do not affect the baseline while all
    # profiles still inherit the shared search_overrides.
    common_search = dict(search_overrides or {})
    baseline_config = _apply_search_overrides(normalized, common_search)

    provisional_runtime = build_eval_elo_runtime_config(
        baseline_config,
        elo_config,
        device=device,
        use_mcts=True,
        simulations=plan["baseline_simulations"],
        budget_mode="simulations",
        workers=workers,
    )
    worker_resolver = create_elo_estimator(
        model,
        baseline_config,
        device,
        provisional_runtime,
    )
    effective_workers = worker_resolver._resolve_workers(
        provisional_runtime.get("workers", 0),
        len(plan["games"]),
    )
    runtime_plan = resolve_benchmark_runtime_plan(
        normalized,
        device_type=getattr(device, "type", str(device)),
        baseline=plan["baseline_simulations"],
        simulations=simulations,
        games_per_profile=plan["games_per_profile"],
        workers=effective_workers,
        opening_suite=opening_suite,
        variants=variants,
        search_overrides=common_search,
        central_inference_overrides=effective_central_overrides,
    )
    resolved_scenarios = {
        str(scenario["name"]): scenario
        for scenario in runtime_plan["scenarios"]
    }
    runtime_configs = {
        name: dict(scenario["elo_runtime_config"])
        for name, scenario in resolved_scenarios.items()
    }
    scenario_search_configs = {
        name: dict(scenario["search_runtime_config"])
        for name, scenario in resolved_scenarios.items()
    }
    effective_workers = int(runtime_plan["effective_workers"] or 1)
    central_enabled = runtime_plan["inference_route"] == "central_gpu_with_remote_cpu_workers"
    if getattr(device, "type", None) == "cuda" and effective_workers > 1 and not central_enabled:
        raise ValueError(
            "Multi-worker local CUDA benchmarking is unsafe; enable the Elo central inference path "
            "or set workers=1."
        )
    runtime_info = {
        "device": str(device),
        "device_type": getattr(device, "type", str(device)),
        "requested_workers": workers,
        "resolved_workers": int(effective_workers),
        "inference_route": runtime_plan["inference_route"],
        "local_cuda_graph_policy": runtime_plan["local_cuda_graph_policy"],
        "elo_central_inference_enabled": runtime_plan["central_inference_requested"],
        "central_inference_overrides": runtime_plan["central_inference_overrides"],
        "central_inference": None,
        "local_cuda_graph_stats": {},
    }
    manifest_material = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "base_configs": scenario_search_configs,
        "elo_runtime_configs": runtime_configs,
        "plan": plan,
        "runtime_policy": runtime_plan,
        "effective_workers": effective_workers,
        "device_type": runtime_info["device_type"],
        "model_spec": getattr(model, "model_spec", None),
        "code_sha256": _code_fingerprints(),
    }
    fingerprint = _manifest_fingerprint(manifest_material)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "fingerprint": fingerprint,
        "checkpoint": {"path": str(checkpoint), "sha256": manifest_material["checkpoint_sha256"]},
        "code_sha256": manifest_material["code_sha256"],
        "runtime_policy": runtime_plan,
        "plan": plan,
    }

    artifact_paths = (manifest_path, progress_path, pgn_path, summary_path)
    if resume:
        if not manifest_path.exists() and not progress_path.exists():
            raise FileNotFoundError(
                "resume requested but no benchmark_manifest.json/benchmark_progress.json exists"
            )
        if manifest_path.exists() != progress_path.exists():
            raise ValueError("resume requires both benchmark_manifest.json and benchmark_progress.json")
        if manifest_path.exists():
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prior_manifest.get("fingerprint") != fingerprint:
                raise ValueError("resume refused: checkpoint, effective config, opening plan, or code changed")
            prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if prior_progress.get("fingerprint") != fingerprint:
                raise ValueError("resume refused: progress fingerprint does not match manifest")
            completed = dict(prior_progress.get("completed_games", {}) or {})
    else:
        existing = [str(path) for path in artifact_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "benchmark output already exists; choose a new output_dir or pass resume=True: "
                + ", ".join(existing)
            )
        completed = {}
        _atomic_write_json(manifest_path, manifest)

    plan_game_ids = {str(task["game_id"]) for task in plan["games"]}
    unknown_ids = set(completed) - plan_game_ids
    if unknown_ids:
        raise ValueError(f"resume progress contains unknown game IDs: {sorted(unknown_ids)[:3]}")

    progress = {
        "schema_version": _SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "status": "running",
        "updated_at_utc": _utc_now(),
        "completed_games": completed,
        "games_completed": len(completed),
        "games_expected": len(plan["games"]),
    }
    local_cancel = threading.Event()
    external_cancel = cancel_event
    context_lock = threading.Lock()
    thread_local = threading.local()
    worker_contexts = []
    manager = create_elo_estimator(
        model,
        baseline_config,
        device,
        runtime_configs["baseline"],
    )
    central_session = None
    if central_enabled:
        central_session = manager._start_central_inference_for_elo(effective_workers)
        session_state = getattr(central_session, "_session", {}) or {}
        server_config = dict(session_state.get("server_config", {}) or {})
        central_config = dict(server_config.get("central_inference", {}) or {})
        runtime_info["central_inference"] = {
            "description": central_session.describe(),
            "servers": int(session_state.get("server_count", 0) or 0),
            "workers": int(central_session.workers),
            "models_loaded": _jsonable(session_state.get("load_messages", [])),
            "use_compile_configured": bool(central_config.get("use_compile", False)),
            "cuda_graph_forward_configured": bool(central_config.get("cuda_graph_forward_enabled", False)),
            "cuda_graph_hot_shapes": _jsonable(central_config.get("cuda_graph_forward_hot_shapes", {})),
            "cuda_graph_min_observations": int(
                central_config.get("cuda_graph_forward_min_observations", 4) or 4
            ),
            "profile_cuda_graphs": bool(central_config.get("profile_cuda_graphs", False)),
            "cuda_graph_effective_enabled": bool(
                central_config.get("cuda_graph_forward_enabled", False)
                and not central_config.get("use_compile", False)
                and getattr(device, "type", None) == "cuda"
            ),
            "cuda_graph_capture_stats_available": False,
            "cuda_graph_capture_stats": [],
        }

    def get_worker_context():
        cached = getattr(thread_local, "context", None)
        if cached is not None:
            return cached
        if central_session is not None:
            worker_model = central_session.remote_model_for_current_thread()
            worker_device = torch.device("cpu")
            if worker_model is None:
                raise RuntimeError("central Elo inference did not provide a worker-local model proxy")
        else:
            worker_model = model
            worker_device = device
        players = {}
        for key, scenario_config in scenario_search_configs.items():
            profile_simulations = (
                plan["baseline_simulations"] if key == "baseline"
                else next(item["simulations"] for item in plan["profiles"] if item["name"] == key)
            )
            players[key] = _BatchedModelPlayer(
                worker_model,
                scenario_config,
                worker_device,
                use_mcts=True,
                simulations=int(profile_simulations),
            )
        cached = {"model": worker_model, "device": worker_device, "players": players}
        thread_local.context = cached
        with context_lock:
            worker_contexts.append(cached)
        return cached

    def run_one(task):
        if local_cancel.is_set() or (external_cancel is not None and external_cancel.is_set()):
            return None
        context = get_worker_context()
        return _play_full_game(
            task,
            context["players"][task["profile_name"]],
            context["players"]["baseline"],
            local_cancel if external_cancel is None else _CombinedCancelEvent(local_cancel, external_cancel),
        )

    def save_progress(status: str):
        progress["status"] = status
        progress["updated_at_utc"] = _utc_now()
        progress["games_completed"] = len(progress["completed_games"])
        _atomic_write_json(progress_path, progress)
        ordered = sorted(
            progress["completed_games"].values(),
            key=lambda record: int(record.get("plan_index", 0)),
        )
        _write_pgn(pgn_path, ordered)
        summary = _summary_payload(manifest, progress, plan, runtime_info, {
            "manifest": manifest_path,
            "progress": progress_path,
            "pgn": pgn_path,
            "summary": summary_path,
        })
        _atomic_write_json(summary_path, summary)
        if callable(progress_callback):
            progress_callback({
                "status": status,
                "games_completed": progress["games_completed"],
                "games_expected": progress["games_expected"],
                "profiles": summary["profiles"],
            })

    pending_tasks = [
        {**task, "plan_index": index}
        for index, task in enumerate(plan["games"])
        if task["game_id"] not in completed
    ]
    error = None
    last_checkpoint = time.monotonic()
    interrupted = False
    try:
        if effective_workers <= 1:
            for task in pending_tasks:
                if local_cancel.is_set() or (external_cancel is not None and external_cancel.is_set()):
                    interrupted = True
                    break
                record = run_one(task)
                if record is None:
                    interrupted = True
                    break
                completed[record["game_id"]] = record
                progress["completed_games"] = completed
                save_progress("running")
                last_checkpoint = time.monotonic()
        else:
            executor = ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="mcts-benchmark")
            futures: dict[Future, dict] = {executor.submit(run_one, task): task for task in pending_tasks}
            try:
                while futures:
                    if external_cancel is not None and external_cancel.is_set():
                        local_cancel.set()
                        interrupted = True
                        for future in futures:
                            future.cancel()
                    done, _ = wait(
                        tuple(futures),
                        timeout=max(0.1, 60.0 - (time.monotonic() - last_checkpoint)),
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        save_progress("running")
                        last_checkpoint = time.monotonic()
                        continue
                    for future in done:
                        task = futures.pop(future)
                        if future.cancelled():
                            interrupted = True
                            continue
                        record = future.result()
                        if record is None:
                            interrupted = True
                            continue
                        record["plan_index"] = int(task["plan_index"])
                        completed[record["game_id"]] = record
                        progress["completed_games"] = completed
                        save_progress("running")
                        last_checkpoint = time.monotonic()
            except KeyboardInterrupt:
                interrupted = True
                local_cancel.set()
                for future in futures:
                    future.cancel()
                # Running games check cancellation between moves; wait for them
                # to finish without converting an interrupted game to a draw.
                done, _ = wait(tuple(futures), timeout=None)
                for future in done:
                    if future.cancelled():
                        continue
                    record = future.result()
                    if record is not None:
                        record["plan_index"] = int(futures[future]["plan_index"])
                        completed[record["game_id"]] = record
                progress["completed_games"] = completed
            except BaseException:
                # A failed future must stop in-flight games before shutdown;
                # otherwise executor.shutdown waits for every game to finish.
                local_cancel.set()
                for future in futures:
                    future.cancel()
                done, _ = wait(tuple(futures), timeout=None)
                for future in done:
                    if future.cancelled():
                        continue
                    try:
                        record = future.result()
                    except BaseException:
                        continue
                    if record is not None:
                        record["plan_index"] = int(futures[future]["plan_index"])
                        completed[record["game_id"]] = record
                progress["completed_games"] = completed
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        if external_cancel is not None and external_cancel.is_set():
            interrupted = True
    except KeyboardInterrupt:
        interrupted = True
        local_cancel.set()
    except Exception as exc:
        local_cancel.set()
        error = f"{type(exc).__name__}: {exc}"
    finally:
        graph_stats = {}
        with context_lock:
            contexts = list(worker_contexts)
        for context_index, context in enumerate(contexts):
            graph_stats[str(context_index)] = {
                name: _player_graph_stats(player)
                for name, player in context["players"].items()
            }
            _dispose_players(context["players"])
        runtime_info["local_cuda_graph_stats"] = graph_stats
        if central_session is not None:
            manager._stop_central_inference_for_elo()
            diagnostics = list(getattr(central_session, "_last_diagnostic_messages", []) or [])
            if runtime_info.get("central_inference") is not None:
                graph_messages = [
                    message for message in diagnostics
                    if message.get("type") == "forward_shape_diagnostics"
                ]
                runtime_info["central_inference"]["diagnostic_messages"] = _jsonable(diagnostics)
                runtime_info["central_inference"]["cuda_graph_capture_stats"] = _jsonable([
                    message.get("cuda_graph_stats", {}) for message in graph_messages
                ])
                runtime_info["central_inference"]["cuda_graph_capture_stats_available"] = bool(
                    graph_messages
                )

    is_complete = len(completed) == len(plan["games"]) and error is None
    progress["completed_games"] = completed
    progress["error"] = error
    progress["status"] = "complete" if is_complete else "partial"
    if interrupted:
        progress["interrupted"] = True
    save_progress(progress["status"])
    summary = _summary_payload(manifest, progress, plan, runtime_info, {
        "manifest": manifest_path,
        "progress": progress_path,
        "pgn": pgn_path,
        "summary": summary_path,
    })
    if error is not None:
        summary["error"] = error
        _atomic_write_json(summary_path, summary)
        raise RuntimeError(f"benchmark stopped with an error after saving partial progress: {error}")
    return summary


class _CombinedCancelEvent:
    """Small event facade so game loops honor both internal and caller stops."""

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def is_set(self):
        return bool(self.first.is_set() or self.second.is_set())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
