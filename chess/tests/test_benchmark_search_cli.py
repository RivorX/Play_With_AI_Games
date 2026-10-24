import importlib.util
import io
from pathlib import Path
from unittest.mock import patch

import pytest


CHESS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = CHESS_DIR / "scripts" / "benchmark_search.py"
SPEC = importlib.util.spec_from_file_location("benchmark_search_cli", SCRIPT)
benchmark_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_cli)


def test_default_benchmark_plan_is_three_paired_twenty_game_profiles():
    options = benchmark_cli.load_benchmark_spec()
    assert options["baseline_simulations"] == 192
    assert options["simulations"] == [384, 768, 1536]
    assert options["games_per_profile"] == 20
    assert options["opening_suite"] == "elo_first_10"
    assert options["workers"] == 4
    assert options["variants"] is None
    assert options["search_overrides"] == {}
    assert options["central_inference_overrides"]["cuda_graph_forward_enabled"] is True
    assert "logs" in str(options["output_dir"])
    assert "benchmark" in str(options["output_dir"])


def test_variant_profiles_preserve_overrides_and_allow_same_budget():
    import yaml

    path = CHESS_DIR / "config" / "benchmark.yaml"
    source = yaml.safe_dump({
            "benchmark": {
                "variants": [
                    {"name": "soft-a", "simulations": 384, "config_overrides": {"reinforcement_learning": {"search": {"gumbel_q_scale_visit_exponent": 0.5}}}},
                    {"name": "soft-b", "simulations": 384, "config_overrides": {"reinforcement_learning": {"search": {"gumbel_c_scale": 0.08}}}},
                ],
                "search_overrides": {"gumbel_c_scale": 0.1},
            }
        })
    with patch.object(Path, "open", return_value=io.StringIO(source)):
        options = benchmark_cli.load_benchmark_spec(path)
    assert [variant["name"] for variant in options["variants"]] == ["soft-a", "soft-b"]
    assert options["simulations"] == [384, 384]
    assert options["search_overrides"] == {"gumbel_c_scale": 0.1}
    resolved = benchmark_cli.dry_run_summary(options, benchmark_cli._runtime_config(options))
    assert resolved["profiles"][0]["search_overrides"] == {
        "gumbel_c_scale": 0.1,
        "gumbel_q_scale_visit_exponent": 0.5,
    }


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"benchmark": {"simulations": [192]}}, "must not repeat baseline"),
        ({"benchmark": {"simulations": [384, 384]}}, "must be unique"),
        ({"benchmark": {"games_per_profile": 0}}, "positive integer"),
    ],
)
def test_benchmark_spec_rejects_invalid_budgets_and_game_counts(value, expected):
    import yaml

    path = CHESS_DIR / "config" / "benchmark.yaml"
    with patch.object(Path, "open", return_value=io.StringIO(yaml.safe_dump(value))):
        with pytest.raises(ValueError, match=expected):
            benchmark_cli.load_benchmark_spec(path)


def test_dry_run_resolves_protocol_without_starting_benchmark(capsys):
    assert benchmark_cli.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"total_games": 60' in output
    assert '"workers": 4' in output
    assert '"central_cuda_graph": true' in output
    assert '"central_compile": false' in output
    assert '"inference_route": "central_gpu_with_remote_cpu_workers"' in output


def test_result_summary_reports_win_and_score_percentages_concisely():
    rendered = benchmark_cli.format_result_summary(
        {
            "status": "complete",
            "games_completed": 20,
            "games_expected": 20,
            "profiles": [{
                "name": "mcts-384", "simulations": 384,
                "games_completed": 20, "games_expected": 20,
                "wins": 8, "draws": 4, "losses": 8,
                "score_fraction": 0.5,
                "mean_fresh_simulations_profile": 360,
                "mean_fresh_simulations_baseline": 190,
            }],
        },
        CHESS_DIR / "logs" / "benchmark",
    )
    assert "W-D-L 8-4-8 (20/20)" in rendered
    assert "win 40.0%, score 50.0%" in rendered
    assert "fresh sims/move 360 vs 190" in rendered
    assert "{\"status\"" not in rendered


def test_result_summary_does_not_report_rates_before_any_game_finishes():
    rendered = benchmark_cli.format_result_summary({
        "status": "running",
        "games_completed": 0,
        "games_expected": 20,
        "profiles": [{"name": "mcts-384", "simulations": 384, "games_completed": 0, "games_expected": 20}],
    })
    assert "win n/a, score n/a" in rendered
    assert "win 0.0%" not in rendered
