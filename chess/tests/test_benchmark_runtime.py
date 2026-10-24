import torch
import json
import time

from src.evaluation import elo_runner
from src.evaluation import benchmark as benchmark_module
from src.evaluation.benchmark import (
    build_benchmark_plan,
    resolve_benchmark_runtime_plan,
)
from src.mcts.search import MultiGameBatchMCTS


def _runtime_config():
    return {
        "model": {},
        "reinforcement_learning": {
            "mcts_dynamic_budget_enabled": False,
            "mcts_gumbel_q_scale_visit_exponent": 1.0,
            "mcts_local_cuda_graph_forward": None,
        },
        "central_inference": {"use_compile": True},
        "elo_estimation": {"mcts_eval_workers": 4},
    }


def test_plan_pairs_ten_openings_across_profiles_and_colors():
    plan = build_benchmark_plan()
    assert len(plan["games"]) == 60
    assert len(plan["profiles"]) == 3
    for profile in plan["profiles"]:
        games = [game for game in plan["games"] if game["profile_name"] == profile["name"]]
        assert len(games) == 20
        for pair_start in range(0, len(games), 2):
            first, second = games[pair_start:pair_start + 2]
            assert first["opening_uci"] == second["opening_uci"]
            assert first["profile_is_white"] is True
            assert second["profile_is_white"] is False

    runtime = resolve_benchmark_runtime_plan(
        _runtime_config(),
        device_type="cuda",
        workers=4,
    )
    assert runtime["inference_route"] == "central_gpu_with_remote_cpu_workers"
    assert runtime["central_use_compile"] is True
    assert runtime["local_cuda_graph_policy"] == "auto"


def test_nested_variant_override_reaches_actual_mcts_engine():
    runtime = resolve_benchmark_runtime_plan(
        _runtime_config(),
        device_type="cpu",
        workers=1,
        variants=[{
            "name": "soft-q",
            "simulations": 384,
            "config_overrides": {
                "reinforcement_learning": {
                    "search": {"gumbel_q_scale_visit_exponent": 0.5},
                },
            },
        }],
    )
    engine_config = runtime["scenarios"][1]["search_runtime_config"]
    assert engine_config["reinforcement_learning"]["mcts_gumbel_q_scale_visit_exponent"] == 0.5
    engine = MultiGameBatchMCTS(torch.nn.Identity(), engine_config, torch.device("cpu"))
    try:
        assert engine.gumbel_q_scale_visit_exponent == 0.5
    finally:
        engine.clear_local_cuda_graph_forward()


def test_central_cuda_graph_override_disables_compile_and_is_reported():
    runtime = resolve_benchmark_runtime_plan(
        _runtime_config(),
        device_type="cuda",
        workers=4,
        central_inference_overrides={
            "cuda_graph_forward_enabled": True,
            "cuda_graph_forward_min_observations": 4,
            "cuda_graph_forward_hot_shapes": {"se_cnn_v9": [2, 5, 6]},
            "profile_cuda_graphs": True,
        },
    )
    assert runtime["inference_route"] == "central_gpu_with_remote_cpu_workers"
    assert runtime["central_use_compile"] is False
    assert runtime["central_cuda_graph_forward_enabled"] is True
    assert runtime["central_cuda_graph_min_observations"] == 4
    assert runtime["central_profile_cuda_graphs"] is True
    assert runtime["central_cuda_graph_hot_shapes"] == {"se_cnn_v9": [2, 5, 6]}
    assert runtime["scenarios"][0]["elo_runtime_config"][
        "eval_elo_central_inference_use_compile"
    ] is False


def test_profile_summary_uses_weighted_fresh_simulations_per_search():
    plan = {
        "baseline_simulations": 192,
        "games_per_profile": 2,
        "profiles": [
            {"name": "weighted", "simulations": 384},
            {"name": "no-searches", "simulations": 768},
        ],
    }
    records = [
        {
            "profile_name": "weighted",
            "score": 1.0,
            "profile_fresh_simulations": 100,
            "profile_search_calls": 2,
            "baseline_fresh_simulations": 60,
            "baseline_search_calls": 3,
        },
        {
            "profile_name": "weighted",
            "score": 0.0,
            "profile_fresh_simulations": 300,
            "profile_search_calls": 6,
            "baseline_fresh_simulations": 180,
            "baseline_search_calls": 3,
        },
        {
            "profile_name": "no-searches",
            "score": 0.5,
            "profile_fresh_simulations": 0,
            "profile_search_calls": 0,
            "baseline_fresh_simulations": 0,
            "baseline_search_calls": 0,
        },
    ]

    weighted, no_searches = benchmark_module._profile_summary(plan, records)
    assert weighted["mean_fresh_simulations_profile"] == 50.0
    assert weighted["mean_fresh_simulations_baseline"] == 40.0
    assert no_searches["mean_fresh_simulations_profile"] is None
    assert no_searches["mean_fresh_simulations_baseline"] is None


def test_resume_requires_a_prior_manifest_and_progress(tmp_path, monkeypatch):
    class FakeEstimator:
        def _resolve_workers(self, requested, total_games):
            return min(max(1, int(requested or 1)), total_games)

    monkeypatch.setattr(
        benchmark_module,
        "create_elo_estimator",
        lambda *args, **kwargs: FakeEstimator(),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    try:
        benchmark_module.run_benchmark(
            _runtime_config(),
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            checkpoint=checkpoint,
            baseline=192,
            simulations=(384,),
            games_per_profile=2,
            workers=1,
            output_dir=output,
            resume=True,
        )
    except FileNotFoundError as exc:
        assert "resume requested" in str(exc)
    else:
        raise AssertionError("resume without a prior benchmark should fail")
    assert not (output / "benchmark_manifest.json").exists()


def test_parallel_failure_cancels_running_games_before_executor_shutdown(tmp_path, monkeypatch):
    class FakeEstimator:
        def _resolve_workers(self, requested, total_games):
            return 2

    class FakePlayer:
        multi_mcts = None

        def __init__(self, *args, **kwargs):
            pass

    def failing_game(task, profile_player, baseline_player, cancel_event):
        if task["profile_game_index"] == 0:
            raise ValueError("intentional worker failure")
        while not cancel_event.is_set():
            time.sleep(0.005)
        return None

    monkeypatch.setattr(
        benchmark_module,
        "create_elo_estimator",
        lambda *args, **kwargs: FakeEstimator(),
    )
    monkeypatch.setattr(benchmark_module, "_BatchedModelPlayer", FakePlayer)
    monkeypatch.setattr(benchmark_module, "_play_full_game", failing_game)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "benchmark"
    started = time.monotonic()
    try:
        benchmark_module.run_benchmark(
            _runtime_config(),
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            checkpoint=checkpoint,
            baseline=192,
            simulations=(384,),
            games_per_profile=2,
            workers=2,
            output_dir=output,
        )
    except RuntimeError as exc:
        assert "intentional worker failure" in str(exc)
    else:
        raise AssertionError("the worker failure should abort the benchmark")
    assert time.monotonic() - started < 5.0
    progress = json.loads((output / "benchmark_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "partial"
    assert "intentional worker failure" in progress["error"]


def test_run_elo_check_preserves_estimator_arguments(monkeypatch):
    captured = {}

    class FakeEstimator:
        def estimate(self, **kwargs):
            captured["estimate"] = kwargs
            return {"sentinel": True}

    def factory(model, config, device, elo_config, *, stop_event=None):
        captured["factory"] = (model, config, device, dict(elo_config), stop_event)
        return FakeEstimator()

    monkeypatch.setattr(elo_runner, "create_elo_estimator", factory)
    model = object()
    config = {"reinforcement_learning": {}}
    device = torch.device("cpu")
    elo_config = {
        "games_per_level": 7,
        "stockfish_time_limit": 0.11,
        "max_moves": 90,
        "use_mcts": True,
        "mcts_simulations": 384,
        "workers": 3,
    }
    stop_event = object()
    result = elo_runner.run_elo_check(
        model,
        config,
        device,
        elo_config,
        stop_event=stop_event,
    )
    assert result == {"sentinel": True}
    factory_args = captured["factory"]
    assert factory_args[:3] == (model, config, device)
    assert factory_args[3]["games_per_level"] == 7
    assert factory_args[3]["stockfish_time_limit"] == 0.11
    assert factory_args[3]["mcts_gumbel_root_schedule"] == "standard"
    assert factory_args[3]["mcts_dynamic_budget_strategy"] == "feedback"
    assert factory_args[4] is stop_event
    assert captured["estimate"] == {
        "levels": None,
        "games_per_level": 7,
        "stockfish_time_limit": 0.11,
        "max_moves": 90,
        "use_mcts": True,
        "simulations": 384,
        "workers": 3,
    }
