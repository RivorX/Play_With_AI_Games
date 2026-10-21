"""
Reinforcement learning training helpers.
"""

import os
import sys
import contextlib
import concurrent.futures
import copy
import math
import threading
import time
from multiprocessing.connection import wait as wait_process_connections
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path for src imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.mcts.search import (
    MultiGameBatchMCTS,
    _EVAL_OPENING_LINES,
    _board_position_key,
    _position_counts_from_board,
    _record_position_count,
)
from src.mcts.session import SearchSession
from src.inference.central import RemoteInferenceClient, central_inference_server
from src.game import backend as chess
from src.model import config_for_model_spec, create_model, model_spec_from_config
from src.models.data.se_cnn_v9.helpers import (
    ACTION_SIZE,
    MAX_LEGAL_MOVES,
    board_to_tensor,
    board_to_tensor_pair,
    build_hflip_inverse_index_map,
    move_to_index,
)
from src.models.data.registry import get_input_encoder
from src.mcts.q_delta import (
    RELIABLE_POLICY_TARGET_GAP_MIN,
    RELIABLE_POLICY_TARGET_TOP1_MIN,
    USEFUL_SEARCH_Q_DELTA_MIN,
)
from src.inference.shared_memory import create_shared_inference_buffer
from src.training.soft_policy import (
    legal_masked_soft_policy_loss_from_prepared,
    prepare_legal_masked_soft_policy_target,
)
from .batch import ReplayBatch


_HFLIP_INV_INDEX_MAP = None
_HFLIP_FWD_INDEX_MAP = None
# CE learns the complete soft Gumbel target, but a deeply wrong prior can lower
# KL without ever changing its top move. This bounded term acts only on measured
# higher-Q search corrections and is normalized by every policy row, preventing
# the correction cohort from taking over the gradient as it did in RL56.
_POLICY_CORRECTION_RANK_MARGIN = 0.10
# RL66 left useful-correction top-1 near 14% while correction rows retained their
# natural replay share. The all-policy-row denominator below keeps this stronger
# margin term bounded without restoring the stacked row multipliers removed in RL57.
_POLICY_CORRECTION_RANK_WEIGHT = 0.75
_POLICY_CORRECTION_Q_DELTA_FULL_STRENGTH = 0.20


_TRANSIENT_CUDA_ERROR_MARKERS = (
    "cuda error: out of memory",
    "cuda out of memory",
    "cudaerroroutofmemory",
    "cudaerrormemoryallocation",
    "cuda driver error",
    "driver shutting down",
    "cuda error: initialization error",
    "cuda error: unknown error",
    "cuda error: unspecified launch failure",
    "unspecified launch failure",
    "device is busy or unavailable",
    "the launch timed out",
)


class _EvalCancelled(Exception):
    """Internal signal used to unwind cancellable evaluation startup."""

    def __init__(self, stage):
        self.stage = str(stage or "evaluation")
        super().__init__(f"Evaluation cancelled during {self.stage}.")


def _eval_stop_requested(stop_event):
    return bool(stop_event is not None and stop_event.is_set())


def _notify_eval_runtime(callback, **details):
    if callback is None:
        return
    try:
        callback(dict(details))
    except Exception:
        # Runtime telemetry must never abort or strand the evaluation itself.
        pass


def _raise_if_eval_stopped(stop_event, stage):
    if _eval_stop_requested(stop_event):
        raise _EvalCancelled(stage)


@contextlib.contextmanager
def _freeze_batchnorm_running_stats(model, enabled):
    """Use stored BatchNorm statistics for one forward without freezing affine weights.

    Dropout and other modules keep their original mode. BatchNorm affine
    parameters remain differentiable; only running-stat updates are disabled.
    """
    if not enabled:
        yield
        return

    batchnorm_modes = tuple(
        (module, bool(module.training))
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    try:
        for module, _ in batchnorm_modes:
            module.training = False
        yield
    finally:
        for module, was_training in batchnorm_modes:
            module.training = was_training


def _training_model_owner(model):
    """Unwrap common training wrappers when inspecting model capabilities/config."""
    owner = model
    seen = set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        nested = getattr(owner, "_orig_mod", None)
        if nested is None:
            nested = getattr(owner, "module", None)
        if nested is None or nested is owner:
            break
        owner = nested
    return owner


def _model_supports_training_aux(model):
    return bool(getattr(_training_model_owner(model), "supports_training_aux", False))


def is_transient_cuda_error(error):
    """Recognize CUDA runtime/driver failures that merit a fresh server process."""
    text = str(error or "").strip().lower()
    return bool(text and any(marker in text for marker in _TRANSIENT_CUDA_ERROR_MARKERS))


def _cpu_only_eval_worker_config(config):
    """Remove CUDA-capable local paths from a remote-inference eval worker."""
    worker_config = dict(config or {})
    worker_config["hardware"] = {
        **dict(worker_config.get("hardware", {}) or {}),
        "device": "cpu",
        "use_amp": False,
        "use_bfloat16": False,
        "use_compile": False,
        "pin_memory": False,
    }
    worker_config["reinforcement_learning"] = {
        **dict(worker_config.get("reinforcement_learning", {}) or {}),
        "self_play_device": "cpu",
    }
    return worker_config


def _gradient_family(name):
    lowered = str(name).lower()
    while True:
        if lowered.startswith('_orig_mod.'):
            lowered = lowered[len('_orig_mod.'):]
        elif lowered.startswith('module.'):
            lowered = lowered[len('module.'):]
        else:
            break
    if (
        lowered.startswith((
            'policy_', 'bn_policy_', 'soft_policy_', 'bn_soft_policy_'
        ))
        or any(token in lowered for token in (
            '.policy_', '.bn_policy_', '.soft_policy_', '.bn_soft_policy_'
        ))
    ):
        return 'policy'
    if (
        lowered.startswith('value_')
        or lowered.startswith('bn_value_')
        or '.value_' in lowered
        or '.bn_value_' in lowered
        or lowered.startswith('moves_left_')
        or lowered.startswith('bn_moves_left_')
        or '.moves_left_' in lowered
        or lowered.startswith('search_q_')
        or lowered.startswith('bn_search_q_')
        or '.search_q_' in lowered
    ):
        return 'value'
    return 'backbone'


def _gradient_family_norms(model):
    sums = {'backbone': None, 'policy': None, 'value': None}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        family = _gradient_family(name)
        squared = parameter.grad.detach().float().pow(2).sum()
        sums[family] = squared if sums[family] is None else sums[family] + squared
    return {
        family: (0.0 if squared is None else float(torch.sqrt(squared).item()))
        for family, squared in sums.items()
    }


def _task_gradient_probe(policy_objective, value_objective, model):
    """Measure policy/value gradient interaction on one shared trunk parameter."""
    probe_owner = _training_model_owner(model)
    probe = None
    probe_layer = getattr(probe_owner, 'final_bn', None)
    if probe_layer is None:
        probe_layer = getattr(probe_owner, 'final_norm', None)
    if probe_layer is None:
        tower = getattr(probe_owner, 'residual_tower', None)
        if tower:
            last_block = tower[-1]
            probe = getattr(getattr(last_block, 'expand', None), 'weight', None)
    else:
        probe = getattr(probe_layer, 'weight', None)
    empty = {
        'policy_probe_norm': 0.0,
        'value_probe_norm': 0.0,
        'policy_value_cosine': 0.0,
    }
    if probe is None or not probe.requires_grad:
        return empty
    policy_grad = torch.autograd.grad(
        policy_objective,
        probe,
        retain_graph=True,
        allow_unused=True,
    )[0]
    value_grad = torch.autograd.grad(
        value_objective,
        probe,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if policy_grad is None or value_grad is None:
        return empty
    policy_flat = policy_grad.detach().float().reshape(-1)
    value_flat = value_grad.detach().float().reshape(-1)
    policy_norm = torch.linalg.vector_norm(policy_flat)
    value_norm = torch.linalg.vector_norm(value_flat)
    denominator = policy_norm * value_norm
    cosine = (
        torch.dot(policy_flat, value_flat) / denominator
        if float(denominator.item()) > 0.0
        else torch.zeros((), device=policy_flat.device)
    )
    return {
        'policy_probe_norm': float(policy_norm.item()),
        'value_probe_norm': float(value_norm.item()),
        'policy_value_cosine': float(torch.clamp(cosine, -1.0, 1.0).item()),
    }


def _snapshot_state_dict_cpu_shared(model):
    from src.inference.cuda_graph_forward import cuda_model_setup_lock

    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        model_device = torch.device("cpu")
    with cuda_model_setup_lock(model_device):
        snapshot = {}
        for key, tensor in model.state_dict().items():
            cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
            cpu_tensor.share_memory_()
            snapshot[key] = cpu_tensor
        return snapshot


def _eval_uses_central_inference(config, device):
    rl_cfg = config.get("reinforcement_learning", {})
    central_cfg = config.get("central_inference", {}) or {}
    enabled = rl_cfg.get(
        "eval_central_inference_enabled",
        rl_cfg.get("self_play_central_inference_enabled", False),
    )
    return bool(enabled and central_cfg.get("enabled", True) and device.type == "cuda" and torch.cuda.is_available())


def _central_inference_config(config):
    return (config or {}).get("central_inference", {}) or {}


def _central_inference_option(config, key, default=None):
    central_cfg = _central_inference_config(config)
    return central_cfg.get(key, default)


def _resolve_configured_self_play_workers(rl_cfg, cpu_budget):
    cpu_budget = max(1, int(cpu_budget))
    raw_workers = rl_cfg.get("self_play_workers", cpu_budget)
    if isinstance(raw_workers, str) and raw_workers.strip().lower() in {"auto", "automatic"}:
        try:
            multiplier = max(0.10, float(rl_cfg.get("self_play_worker_auto_multiplier", 1.0)))
        except Exception:
            multiplier = 1.0
        return max(1, math.ceil(float(cpu_budget) * multiplier))
    try:
        workers = int(raw_workers)
    except Exception:
        workers = cpu_budget
    if workers <= 0:
        try:
            multiplier = max(0.10, float(rl_cfg.get("self_play_worker_auto_multiplier", 1.0)))
        except Exception:
            multiplier = 1.0
        return max(1, math.ceil(float(cpu_budget) * multiplier))
    return max(1, workers)


def _resolve_eval_workers(config, device, num_games):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_workers = rl_cfg.get("eval_workers", None)
    auto_workers = raw_workers is None or str(raw_workers).strip().lower() in {"auto", "automatic", "0"}

    if auto_workers:
        if _eval_uses_central_inference(config, device):
            # CPU workers own MCTS trees, central GPU server owns inference.
            # This mirrors RL self-play and keeps eval from becoming one-core.
            reserve_threads = max(0, int(rl_cfg.get("eval_cpu_threads_to_reserve", 0) or 0))
            cpu_budget = max(1, (os.cpu_count() or 2) - reserve_threads)
            raw_workers = min(
                _resolve_configured_self_play_workers(rl_cfg, cpu_budget),
                cpu_budget,
            )
        elif device.type == "cuda":
            # Without central inference each eval worker would load full CUDA
            # models, so keep the old memory-safe default.
            raw_workers = 1
        else:
            raw_workers = max(1, (os.cpu_count() or 2) - 1)

    try:
        workers = int(raw_workers)
    except Exception:
        workers = 1

    # Keep one stable topology across every funnel stage. Scaling workers down
    # from the stage's game count made 40/60/80-game evals use 7/10/12 workers,
    # forcing different batching behavior and making stage timings incomparable.
    return max(1, min(int(num_games), workers))


def _build_eval_mcts_config(config):
    """Return an exact-budget deterministic evaluation config."""
    eval_config = dict(config or {})
    rl_config = dict(eval_config.get('reinforcement_learning', {}) or {})
    # Neither self-play redistribution nor the retired easy-root cut may alter
    # either side's advertised evaluation budget.
    rl_config['mcts_dynamic_budget_enabled'] = False
    rl_config['eval_mcts_dynamic_budget_enabled'] = False
    eval_config['reinforcement_learning'] = rl_config
    return eval_config


def _resolve_eval_central_server_count(config, workers):
    """Eval targets one concrete CUDA device, so it always owns one server."""
    # More processes on cuda:0 do not add GPU capacity. They duplicate models,
    # split request batches, consume VRAM and compete for the same kernels.
    # Worker count controls CPU-side MCTS concurrency; the GPU stays centralized.
    return 1


def _resolve_eval_batch_games(config, num_games):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get(
        "eval_batch_games",
        rl_cfg.get("max_batch_games_per_worker", 16),
    )
    try:
        value = int(raw_value)
    except Exception:
        value = 16
    return max(1, min(int(num_games), value))


def _terminate_eval_processes(processes, timeout_s=0.5, graceful_timeout_s=0.0):
    """Stop only these evaluation-owned processes with one bounded grace period.

    Joining each process for the full timeout made menu cancellation scale with
    the worker count. Signal all owned children first, wait on their sentinels
    together, and escalate only the survivors.
    """
    unique = []
    seen = set()
    for proc in list(processes or []):
        if proc is None or id(proc) in seen:
            continue
        seen.add(id(proc))
        unique.append(proc)

    alive = []
    for proc in unique:
        try:
            if proc.is_alive():
                alive.append(proc)
        except Exception:
            continue

    def _wait_bounded(candidates, total_timeout):
        deadline = time.monotonic() + max(0.0, float(total_timeout))
        pending = list(candidates)
        while pending:
            pending = [proc for proc in pending if _process_is_alive(proc)]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sentinels = []
            for proc in pending:
                with contextlib.suppress(Exception):
                    sentinels.append(proc.sentinel)
            if sentinels:
                with contextlib.suppress(Exception):
                    wait_process_connections(sentinels, timeout=min(remaining, 0.05))
            else:
                time.sleep(min(remaining, 0.01))
        return [proc for proc in candidates if _process_is_alive(proc)]

    def _process_is_alive(proc):
        try:
            return bool(proc.is_alive())
        except Exception:
            return False

    if graceful_timeout_s > 0.0:
        alive = _wait_bounded(alive, graceful_timeout_s)
    for proc in alive:
        with contextlib.suppress(Exception):
            proc.terminate()

    pending = _wait_bounded(alive, timeout_s)
    for proc in pending:
        with contextlib.suppress(Exception):
            proc.kill()
    # Windows may return from TerminateProcess before the process handle becomes
    # signaled, especially while a CUDA context is tearing down. Keep the drain
    # bounded, but wait long enough that a subsequent match cannot overlap it.
    survivors = _wait_bounded(pending, 1.0)

    for proc in unique:
        with contextlib.suppress(Exception):
            proc.join(timeout=0)
        if not _process_is_alive(proc):
            with contextlib.suppress(Exception):
                proc.close()
    return survivors


def _resolve_eval_max_moves(config):
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    if bool(rl_cfg.get("eval_finish_by_rules", False)):
        return None
    raw_value = rl_cfg.get("eval_max_moves", 300)
    try:
        return max(1, int(raw_value))
    except Exception:
        return 300


def _eval_finish_by_rules(config):
    return bool(
        (config.get("reinforcement_learning", {}) or {}).get(
            "eval_finish_by_rules", False
        )
    )


def _eval_ply_cap_reached(move_count, max_moves):
    return max_moves is not None and int(move_count) >= int(max_moves)


def _eval_game_is_over(board, config):
    return chess.is_game_over(board, claim_draw=_eval_finish_by_rules(config))


def _eval_terminal_reason(board, config, *, cap_reached=False):
    if cap_reached and not _eval_game_is_over(board, config):
        return "diagnostic_ply_cap"
    if board in chess.CHECKMATE:
        return "checkmate"
    if board in chess.STALEMATE:
        return "stalemate"
    if board in chess.INSUFFICIENT_MATERIAL:
        return "insufficient_material"
    if board in chess.FORCED_DRAW:
        return "automatic_draw"
    if _eval_finish_by_rules(config) and board in chess.DRAW:
        return "claimable_draw"
    return "unresolved"


def _merge_eval_terminal_reasons(target, source):
    for reason, count in dict(source or {}).items():
        try:
            count = int(count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            target[str(reason)] = int(target.get(str(reason), 0)) + count


def _resolve_eval_auto_claim_draw(config):
    rl_cfg = config.get("reinforcement_learning", {})
    return bool(rl_cfg.get("eval_auto_claim_draw", False))


def _resolve_eval_unresolved_score_neutral(config):
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    return bool(rl_cfg.get("eval_unresolved_score_neutral", True))


def _resolve_eval_claim_draw_after_moves(config):
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    eval_max_moves = _resolve_eval_max_moves(config)
    default_value = 0 if eval_max_moves is None else eval_max_moves
    raw_value = rl_cfg.get("eval_claim_draw_after_moves", default_value)
    try:
        return max(0, int(raw_value))
    except Exception:
        return default_value


def _resolve_eval_claim_repetition_after_moves(config):
    rl_cfg = config.get("reinforcement_learning", {})
    default_value = min(_resolve_eval_claim_draw_after_moves(config), 80)
    raw_value = rl_cfg.get("eval_claim_repetition_after_moves", default_value)
    try:
        return max(0, int(raw_value))
    except Exception:
        return default_value


def _resolve_eval_fixed_openings_enabled(config):
    rl_cfg = config.get("reinforcement_learning", {})
    return bool(rl_cfg.get("eval_fixed_openings_enabled", False))


def _resolve_eval_fixed_openings_max_plies(config):
    rl_cfg = config.get("reinforcement_learning", {})
    raw_value = rl_cfg.get("eval_fixed_openings_max_plies", 6)
    try:
        return max(0, int(raw_value))
    except Exception:
        return 6


def _resolve_eval_opening_lines(config):
    """Return the configured held-out suite, or the canonical promotion suite.

    The config form is intentionally plain nested lists so it survives Windows
    spawn when evaluation workers reconstruct their configuration.
    """
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    configured = rl_cfg.get("eval_opening_lines")
    if configured is None:
        return _EVAL_OPENING_LINES
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("reinforcement_learning.eval_opening_lines must be a non-empty list of UCI move lists")

    lines = []
    for line in configured:
        if isinstance(line, (str, bytes)) or not isinstance(line, (list, tuple)) or not line:
            raise ValueError("each eval_opening_lines entry must be a non-empty UCI move list")
        moves = tuple(str(move).strip() for move in line)
        if any(not move for move in moves):
            raise ValueError("eval_opening_lines cannot contain empty moves")
        lines.append(moves)
    return tuple(lines)


def _resolve_eval_mcts_simulations(config):
    rl_cfg = config.get("reinforcement_learning", {})
    try:
        base_sims = max(1, int(rl_cfg.get("mcts_simulations", 50)))
    except Exception:
        base_sims = 50

    try:
        multiplier = max(0.01, float(rl_cfg.get("eval_mcts_simulations_multiplier", 1.0)))
        return max(1, int(round(base_sims * multiplier)))
    except Exception:
        return base_sims


def _build_eval_stats(wins, draws, losses, unresolved, num_games):
    num_games = int(num_games)
    wins = int(wins)
    draws = int(draws)
    losses = int(losses)
    unresolved = int(unresolved)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unresolved": unresolved,
        "num_games": num_games,
        # A ply-cap timeout has no chess result. Score it neutrally instead of
        # silently charging the candidate with a loss.
        "score_rate": (wins + 0.5 * (draws + unresolved)) / num_games if num_games > 0 else 0.0,
        "win_rate": wins / num_games if num_games > 0 else 0.0,
        "draw_rate": (draws + unresolved) / num_games if num_games > 0 else 0.0,
        "loss_rate": losses / num_games if num_games > 0 else 0.0,
        "resolved_games": num_games - unresolved,
    }


def _print_eval_unresolved(
    label,
    unresolved,
    num_games,
    max_moves,
    *,
    neutral_score=True,
):
    if int(unresolved) <= 0:
        return
    if max_moves is None:
        print(
            f"{label} unresolved without a ply cap: {unresolved}/{num_games}; "
            "the evaluator did not produce a rules-based result"
        )
        return
    score_note = (
        "scored as 0.5 and tracked separately"
        if neutral_score
        else "excluded from the match score"
    )
    print(
        f"{label} unresolved at ply cap ({max_moves}, ~{max_moves / 2.0:.1f} full moves): "
        f"{unresolved}/{num_games} -> {score_note}"
    )


def _get_eval_opening_prefix(config, game_idx, enabled_override=None):
    enabled = _resolve_eval_fixed_openings_enabled(config) if enabled_override is None else bool(enabled_override)
    opening_lines = _resolve_eval_opening_lines(config)
    if not enabled or not opening_lines:
        return ()

    opening_idx = int(game_idx)
    if bool(config.get("reinforcement_learning", {}).get("eval_fixed_openings_pair_games", False)):
        opening_idx //= 2
    line = opening_lines[opening_idx % len(opening_lines)]
    max_plies = _resolve_eval_fixed_openings_max_plies(config)
    if max_plies <= 0:
        return ()
    return tuple(line[: min(len(line), max_plies)])


def _append_eval_history(
    board_history,
    board,
    config,
    repetition_count=None,
    max_history_positions=None,
):
    board_history.append(
        _encode_eval_history_entry(
            board,
            config,
            repetition_count=repetition_count,
        )
    )
    history_positions = (
        config.get("model", {}).get("history_positions", 0)
        if max_history_positions is None
        else max_history_positions
    )
    max_history = int(history_positions or 0) + 10
    if len(board_history) > max_history:
        del board_history[:-max_history]


def _apply_opening_prefix_for_batched_eval(
    board,
    board_histories,
    opening_prefix,
    history_configs,
    position_counts=None,
    max_history_positions=None,
):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if chess.is_game_over(board, claim_draw=False):
            break
        try:
            move = chess.move_from_uci(uci)
        except Exception:
            break
        if move is None or move not in chess.legal_moves(board):
            break

        repetition_count = (
            max(0, int(position_counts.get(_board_position_key(board), 0)) - 1)
            if position_counts is not None
            else None
        )
        seen_histories = set()
        for board_history, history_config in zip(board_histories, history_configs):
            # Both sides can share the same encoder contract. In that case the
            # cached tensors are identical, so encode each visited position once.
            if id(board_history) in seen_histories:
                continue
            seen_histories.add(id(board_history))
            _append_eval_history(
                board_history,
                board,
                history_config,
                repetition_count=repetition_count,
                max_history_positions=max_history_positions,
            )
        chess.apply_move(board, move)
        if position_counts is not None:
            _record_position_count(position_counts, board)
        applied += 1
    return applied


def _encode_eval_history_entry(board, config=None, repetition_count=None):
    model_cfg = (config or {}).get('model', {}) or {}
    encoder = get_input_encoder(
        str(model_cfg.get('input_encoder_id', 'planes_history_v1'))
    )
    if encoder.ENCODER_ID == 'planes_history_v2':
        return encoder.encode_pair(
            board,
            repetition_count=repetition_count,
        )
    return board_to_tensor_pair(board)


def _build_no_mcts_eval_input(
    board,
    board_history,
    config,
    repetition_count=None,
):
    model_cfg = config.get('model', {}) or {}
    history_positions = int(model_cfg.get("history_positions", 0) or 0)
    encoder = get_input_encoder(
        str(model_cfg.get('input_encoder_id', 'planes_history_v1'))
    )
    if encoder.ENCODER_ID == 'planes_history_v2':
        repetition_mode = encoder.resolve_repetition_plane_mode(
            model_config=model_cfg,
        )
        repetition_count = encoder.repetition_count_for_mode(
            repetition_mode,
            repetition_count,
        )
        current_tensor = encoder.encode(
            board,
            repetition_count=repetition_count,
        )
    else:
        current_tensor = board_to_tensor(board)
    if history_positions <= 0:
        return current_tensor

    use_black_pov = (board.turn == chess.BLACK)
    history_slice = list(board_history[-history_positions:]) if board_history else []
    history_tensors = []
    for encoded in history_slice:
        # Evaluation stores one history cache per model, so these are already
        # encoded with this model's encoder contract. Older v10 caches can
        # still contain exact repetition counts even when the checkpoint was
        # trained with unknown (-1) repetition planes; sanitize a copy so the
        # raw-policy path matches MCTS without mutating shared history.
        history_tensor = encoded[1] if use_black_pov else encoded[0]
        if (
            encoder.ENCODER_ID == 'planes_history_v2'
            and repetition_mode != 'exact'
        ):
            history_tensor = np.array(history_tensor, copy=True)
            history_tensor[12].fill(-1.0)
        history_tensors.append(history_tensor)
    if encoder.ENCODER_ID == 'planes_history_v2':
        from src.models.data.nested_cnn_v10.helpers import stack_encoded_history

        return stack_encoded_history(
            current_tensor,
            history_tensors,
            history_positions,
            dtype=np.float32,
        )
    pad_count = max(0, history_positions - len(history_tensors))
    if pad_count:
        empty = np.zeros((16, 8, 8), dtype=np.float32)
        history_tensors = [empty] * pad_count + history_tensors
    history_tensors.append(current_tensor)
    return np.concatenate(history_tensors, axis=0)


def _select_no_mcts_policy_move(
    model,
    board,
    board_history,
    config,
    device,
    position_counts=None,
    position_history_complete=None,
):
    moves = _select_no_mcts_policy_moves_batched(
        model,
        [{
            "board": board,
            "board_history": board_history,
            "position_counts": position_counts,
            "position_history_complete": position_history_complete,
        }],
        config,
        device,
    )
    return moves[0] if moves else None


def _select_no_mcts_policy_moves_batched(model, game_states, config, device):
    """Select raw-policy moves for many games with one model request."""
    states = list(game_states or [])
    if not states:
        return []

    inputs = []
    legal_moves_by_row = []
    legal_indices_by_row = []
    max_legal = 0
    for state in states:
        board = state["board"]
        position_counts = state.get('position_counts')
        provenance_known = state.get(
            'position_history_complete',
            position_counts is not None
            or int(board.halfmove_clock) == 0
            or bool(chess.move_history(board)),
        )
        if provenance_known:
            known_counts = (
                position_counts
                if isinstance(position_counts, dict)
                else _position_counts_from_board(board)
            )
            repetition_count = max(
                0,
                int(known_counts.get(_board_position_key(board), 0)) - 1,
            )
        else:
            repetition_count = None
        inputs.append(_build_no_mcts_eval_input(
            board,
            state.get("board_history", []),
            config,
            repetition_count=repetition_count,
        ))
        legal_moves = tuple(chess.legal_moves(board))
        legal_indices = [move_to_index(move, board) for move in legal_moves]
        legal_moves_by_row.append(legal_moves)
        legal_indices_by_row.append(legal_indices)
        max_legal = max(max_legal, len(legal_indices))

    if max_legal <= 0:
        return [None] * len(states)

    boards_np = np.stack(inputs, axis=0).astype(np.float32, copy=False)
    remote_legal_gather = bool(getattr(model, "supports_remote_legal_gather", False))
    legal_matrix = np.zeros(
        (len(states), max_legal),
        dtype=np.int16 if remote_legal_gather else np.int64,
    )
    for row_idx, legal_indices in enumerate(legal_indices_by_row):
        if legal_indices:
            legal_matrix[row_idx, :len(legal_indices)] = legal_indices

    if remote_legal_gather:
        model_input = boards_np
        model_kwargs = {
            "apply_log_softmax": False,
            "policy_only": True,
            "legal_index_matrix": legal_matrix,
        }
    else:
        model_input = torch.from_numpy(boards_np).to(
            device,
            dtype=torch.float32,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
        model_kwargs = {"apply_log_softmax": False, "policy_only": True}

    use_amp = bool(config.get("hardware", {}).get("use_amp", False) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if config.get("hardware", {}).get("use_bfloat16", False) else torch.float16
    with torch.inference_mode():
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            policy_logits, _value = model(model_input, **model_kwargs)
    compact_policy = bool(getattr(model, "last_response_compact_policy", False))
    policy_np = policy_logits.float().cpu().numpy()

    selected = []
    for row_idx, (legal_moves, legal_indices) in enumerate(zip(legal_moves_by_row, legal_indices_by_row)):
        if not legal_moves:
            selected.append(None)
            continue
        if compact_policy:
            legal_scores = policy_np[row_idx, :len(legal_moves)]
        else:
            legal_scores = policy_np[row_idx, np.asarray(legal_indices, dtype=np.int64)]
        selected.append(legal_moves[int(np.argmax(legal_scores))])
    return selected


def _apply_opening_prefix_for_no_mcts_eval(
    board,
    board_history,
    opening_prefix,
    config=None,
    position_counts=None,
):
    if not opening_prefix:
        return 0

    applied = 0
    for uci in opening_prefix:
        if chess.is_game_over(board, claim_draw=False):
            break
        try:
            move = chess.move_from_uci(uci)
        except Exception:
            break
        if move is None or move not in chess.legal_moves(board):
            break
        repetition_count = (
            max(0, int(position_counts.get(_board_position_key(board), 0)) - 1)
            if position_counts is not None
            else None
        )
        board_history.append(_encode_eval_history_entry(
            board,
            config,
            repetition_count=repetition_count,
        ))
        chess.apply_move(board, move)
        if position_counts is not None:
            _record_position_count(position_counts, board)
        applied += 1
    return applied


def _evaluate_single_game_no_mcts(
    model_white,
    model_black,
    config,
    device,
    max_moves,
    auto_claim_draw,
    claim_draw_after_moves,
    claim_repetition_after_moves=0,
    opening_prefix=(),
):
    board = chess.new_board()
    board_history = []
    position_counts = {_board_position_key(board): 1}
    move_count = _apply_opening_prefix_for_no_mcts_eval(
        board,
        board_history,
        opening_prefix,
        config,
        position_counts,
    )
    ended_by_auto_claim_draw = False

    while max_moves is None or move_count < max_moves:
        if _eval_game_is_over(board, config):
            break

        model = model_white if board.turn == chess.WHITE else model_black
        move = _select_no_mcts_policy_move(
            model,
            board,
            board_history,
            config,
            device,
            position_counts=position_counts,
            position_history_complete=True,
        )
        if move is None:
            break

        repetition_count = max(
            0,
            int(position_counts.get(_board_position_key(board), 0)) - 1,
        )
        board_history.append(_encode_eval_history_entry(
            board,
            config,
            repetition_count=repetition_count,
        ))
        chess.apply_move(board, move)
        _record_position_count(position_counts, board)
        move_count += 1

        if auto_claim_draw:
            if (
                move_count >= claim_repetition_after_moves
                and chess.can_claim_threefold_repetition(board)
            ):
                ended_by_auto_claim_draw = True
                break
            if int(board.halfmove_clock) >= 100:
                ended_by_auto_claim_draw = True
                break
            if move_count >= claim_draw_after_moves and chess.can_claim_draw(board):
                ended_by_auto_claim_draw = True
                break

    if ended_by_auto_claim_draw:
        return "1/2-1/2", False

    result = chess.result(board, claim_draw=_eval_finish_by_rules(config))
    return result, (result == "*")


def _evaluate_games_batched(
    model1,
    model2,
    config,
    device,
    game_indices,
    use_fixed_openings=None,
    progress_callback=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
    runtime_progress_callback=None,
):
    game_indices = list(game_indices or [])
    if not game_indices:
        return _build_eval_stats(0, 0, 0, 0, 0)

    model1_search_config = model1_mcts_config or config
    model2_search_config = model2_mcts_config or config
    mcts1 = MultiGameBatchMCTS(model1, model1_search_config, device) if use_mcts_model1 else None
    mcts2 = MultiGameBatchMCTS(model2, model2_search_config, device) if use_mcts_model2 else None
    sims1 = _resolve_eval_mcts_simulations(model1_search_config)
    sims2 = _resolve_eval_mcts_simulations(model2_search_config)
    max_moves = _resolve_eval_max_moves(config)
    finish_by_rules = _eval_finish_by_rules(config)
    auto_claim_draw = _resolve_eval_auto_claim_draw(config)
    claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(config)
    claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(config)
    active_limit = _resolve_eval_batch_games(config, len(game_indices))

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    terminal_reasons = {}
    cursor = 0
    active_games = []
    eval_search_profile = {}
    runtime_started_at = time.perf_counter()
    runtime_last_emit = runtime_started_at
    runtime_plies_processed = 0
    runtime_nn_batches = 0
    runtime_search_batches = 0

    def _emit_runtime_progress(
        *, force=False, activity=None, inference_side=None, inference_batch_size=None
    ):
        nonlocal runtime_last_emit
        if runtime_progress_callback is None:
            return
        now = time.perf_counter()
        if not force and now - runtime_last_emit < 0.75:
            return
        runtime_last_emit = now
        _notify_eval_runtime(
            runtime_progress_callback,
            status="running",
            ready=True,
            active_games=int(len(active_games)),
            completed_games=int(completed),
            plies_processed=int(runtime_plies_processed),
            nn_batches=int(runtime_nn_batches),
            search_batches=int(runtime_search_batches),
            runtime_activity=str(activity or "advancing_games"),
            inference_side=inference_side,
            inference_batch_size=(
                None if inference_batch_size is None else int(inference_batch_size)
            ),
            activity_elapsed_s=float(now - runtime_started_at),
        )

    def _record_eval_search(prefix, metadata):
        if not isinstance(metadata, dict):
            return
        budget = int(metadata.get('simulation_budget', 0) or 0)
        requested_budget = int(metadata.get('eval_easy_cut_requested_budget', budget) or budget)
        budget_samples_key = f"eval_mcts_{prefix}_budget_samples"
        budget_sum_key = f"eval_mcts_{prefix}_budget_sum"
        reduced_key = f"eval_mcts_{prefix}_reduced_budget_count"
        eval_search_profile[budget_samples_key] = int(
            eval_search_profile.get(budget_samples_key, 0)
        ) + 1
        eval_search_profile[budget_sum_key] = float(
            eval_search_profile.get(budget_sum_key, 0.0)
        ) + float(budget)
        if budget < requested_budget:
            eval_search_profile[reduced_key] = int(
                eval_search_profile.get(reduced_key, 0)
            ) + 1
        agreement = metadata.get("selected_prior_agree")
        if agreement is None:
            return
        sample_key = f"eval_mcts_{prefix}_move_samples"
        changed_key = f"eval_mcts_{prefix}_changed_count"
        eval_search_profile[sample_key] = int(eval_search_profile.get(sample_key, 0)) + 1
        changed = float(agreement) < 0.5
        if not changed:
            return
        eval_search_profile[changed_key] = int(eval_search_profile.get(changed_key, 0)) + 1
        q_delta = metadata.get("selected_q_delta")
        try:
            q_delta = float(q_delta)
        except (TypeError, ValueError):
            return
        if not math.isfinite(q_delta):
            return
        q_samples_key = f"eval_mcts_{prefix}_changed_q_samples"
        q_sum_key = f"eval_mcts_{prefix}_changed_q_delta_sum"
        eval_search_profile[q_samples_key] = int(eval_search_profile.get(q_samples_key, 0)) + 1
        eval_search_profile[q_sum_key] = float(eval_search_profile.get(q_sum_key, 0.0)) + q_delta
        if q_delta > USEFUL_SEARCH_Q_DELTA_MIN:
            key = f"eval_mcts_{prefix}_higher_q_count"
            eval_search_profile[key] = int(eval_search_profile.get(key, 0)) + 1
        elif q_delta < -USEFUL_SEARCH_Q_DELTA_MIN:
            key = f"eval_mcts_{prefix}_lower_q_count"
            eval_search_profile[key] = int(eval_search_profile.get(key, 0)) + 1

    max_history_positions = max(
        int(model1_search_config.get("model", {}).get("history_positions", 0) or 0),
        int(model2_search_config.get("model", {}).get("history_positions", 0) or 0),
    )

    shared_history_cache = (
        model1_search_config.get('model', {}).get('input_encoder_id', 'planes_history_v1')
        == model2_search_config.get('model', {}).get('input_encoder_id', 'planes_history_v1')
        and int(model1_search_config.get('model', {}).get('history_positions', 0) or 0)
        == int(model2_search_config.get('model', {}).get('history_positions', 0) or 0)
    )

    def _new_game_state(game_idx):
        board = chess.new_board()
        model1_history = []
        model2_history = model1_history if shared_history_cache else []
        position_counts = {_board_position_key(board): 1}
        opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
        move_count = _apply_opening_prefix_for_batched_eval(
            board,
            [model1_history, model2_history],
            opening_prefix,
            [model1_search_config, model2_search_config],
            position_counts,
            max_history_positions=max_history_positions,
        )
        return {
            "game_idx": int(game_idx),
            "board": board,
            "model1_board_history": model1_history,
            "model2_board_history": model2_history,
            "position_counts": position_counts,
            "position_history_complete": True,
            "move_count": int(move_count),
            "model1_as_white": bool(int(game_idx) % 2 == 0),
            "model1_session": (
                SearchSession(mcts1, tree_key=object()) if mcts1 is not None else None
            ),
            "model2_session": (
                SearchSession(mcts2, tree_key=object()) if mcts2 is not None else None
            ),
            "done": bool(
                _eval_game_is_over(board, config)
                or _eval_ply_cap_reached(move_count, max_moves)
            ),
            "auto_claim_draw": False,
        }

    def _fill_active():
        nonlocal cursor
        while cursor < len(game_indices) and len(active_games) < active_limit:
            active_games.append(_new_game_state(game_indices[cursor]))
            cursor += 1

    def _finish_game(gs):
        nonlocal wins, draws, losses, unresolved, completed
        for key in ("model1_session", "model2_session"):
            session = gs.get(key)
            if session is not None:
                session.close()
                gs[key] = None
        board = gs["board"]
        if bool(gs.get("auto_claim_draw", False)):
            result = "1/2-1/2"
            was_unresolved = False
        else:
            result = chess.result(board, claim_draw=finish_by_rules)
            was_unresolved = (result == "*")

        reason = (
            "auto_claimed_draw"
            if bool(gs.get("auto_claim_draw", False))
            else _eval_terminal_reason(
                board,
                config,
                cap_reached=_eval_ply_cap_reached(
                    gs.get("move_count", 0), max_moves
                ),
            )
        )
        terminal_reasons[reason] = int(terminal_reasons.get(reason, 0)) + 1

        if was_unresolved:
            unresolved += 1
            game_wins, game_draws, game_losses, game_unresolved = 0, 0, 0, 1
        else:
            game_wins, game_draws, game_losses = _result_for_model1(
                bool(gs.get("model1_as_white", False)),
                result,
            )
            wins += game_wins
            draws += game_draws
            losses += game_losses
            game_unresolved = 0
        completed += 1
        if progress_callback is not None:
            progress_callback(
                1,
                int(gs.get("game_idx", -1)),
                int(game_wins),
                int(game_draws),
                int(game_losses),
                int(game_unresolved),
                int(gs.get("move_count", 0) or 0),
                str(result),
                bool(gs.get("model1_as_white", False)),
            )

    def _claim_draw_if_needed(gs):
        if not auto_claim_draw:
            return False
        board = gs["board"]
        if (
            gs["move_count"] >= claim_repetition_after_moves
            and chess.can_claim_threefold_repetition(board)
        ):
            gs["auto_claim_draw"] = True
            return True
        if int(board.halfmove_clock) >= 100:
            gs["auto_claim_draw"] = True
            return True
        if gs["move_count"] >= claim_draw_after_moves and chess.can_claim_draw(board):
            gs["auto_claim_draw"] = True
            return True
        return False

    _fill_active()
    while active_games and not (stop_event is not None and stop_event.is_set()):
        model1_indices = []
        model2_indices = []
        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            if _eval_game_is_over(board, config) or _eval_ply_cap_reached(
                gs["move_count"], max_moves
            ):
                gs["done"] = True
                continue
            model1_turn = bool(board.turn == chess.WHITE) == bool(gs["model1_as_white"])
            if model1_turn:
                model1_indices.append(idx)
            else:
                model2_indices.append(idx)

        moves_by_index = {}

        def _run_group(
            indices, model, search_config, mcts, sims, session_key, history_key,
            profile_prefix,
        ):
            nonlocal runtime_nn_batches, runtime_search_batches
            if not indices:
                return
            group_states = []
            for idx in indices:
                gs = active_games[idx]
                board_history = gs[history_key]
                position_counts = gs.get("position_counts")
                repetition_known = bool(
                    gs.get("position_history_complete", True)
                )
                if mcts is not None:
                    session = gs.get(session_key)
                    if session is None:
                        raise RuntimeError(
                            f"Missing search session {session_key!r} for active evaluation game"
                        )
                    state = session.make_state(
                        gs["board"],
                        board_history=board_history,
                        position_counts=position_counts,
                        repetition_known=repetition_known,
                        simulation_budget_override=None,
                    )
                else:
                    state = {
                        "board": gs["board"],
                        "board_history": board_history,
                        "position_counts": position_counts,
                        "position_history_complete": repetition_known,
                    }
                group_states.append(state)
            _emit_runtime_progress(
                force=(runtime_nn_batches + runtime_search_batches == 0),
                activity="mcts_search" if mcts is not None else "nn_forward",
                inference_side=str(profile_prefix),
                inference_batch_size=len(indices),
            )
            if mcts is not None:
                runtime_search_batches += 1
                moves = []
                search_results = mcts.search_results_many(
                    group_states,
                    num_simulations=sims,
                    add_root_noise=False,
                )
                for search_result in search_results:
                    moves.append(search_result.selected_move)
                    _record_eval_search(profile_prefix, search_result.metadata)
            else:
                runtime_nn_batches += 1
                moves = _select_no_mcts_policy_moves_batched(
                    model,
                    [
                        {
                            "board": local_state["board"],
                            "board_history": local_state["board_history"],
                            "position_counts": local_state.get("position_counts"),
                            "position_history_complete": local_state.get(
                                "position_history_complete", True
                            ),
                        }
                        for local_state in group_states
                    ],
                    search_config,
                    device,
                )

            for idx, local_state, move in zip(indices, group_states, moves):
                gs = active_games[idx]
                if mcts is not None:
                    gs[session_key].commit(local_state)
                moves_by_index[idx] = move

        _run_group(
            model1_indices, model1, model1_search_config, mcts1, sims1,
            "model1_session", "model1_board_history", "model1",
        )
        if _eval_stop_requested(stop_event):
            break
        _run_group(
            model2_indices, model2, model2_search_config, mcts2, sims2,
            "model2_session", "model2_board_history", "model2",
        )
        if _eval_stop_requested(stop_event):
            break

        for idx, gs in enumerate(active_games):
            if gs["done"]:
                continue
            board = gs["board"]
            move = moves_by_index.get(idx)
            if move is None:
                gs["done"] = True
                continue

            repetition_count = max(
                0,
                int(gs['position_counts'].get(_board_position_key(board), 0)) - 1,
            )
            _append_eval_history(
                gs["model1_board_history"],
                board,
                model1_search_config,
                repetition_count=repetition_count,
                max_history_positions=max_history_positions,
            )
            if not shared_history_cache:
                _append_eval_history(
                    gs["model2_board_history"],
                    board,
                    model2_search_config,
                    repetition_count=repetition_count,
                    max_history_positions=max_history_positions,
                )
            chess.apply_move(board, move)
            gs["move_count"] += 1
            runtime_plies_processed += 1
            _record_position_count(gs["position_counts"], board)
            repetition_known = bool(gs.get("position_history_complete", True))
            for session_key, history_key in (
                ("model1_session", "model1_board_history"),
                ("model2_session", "model2_board_history"),
            ):
                session = gs.get(session_key)
                if session is not None:
                    session.advance(
                        move,
                        board=board,
                        board_history=gs[history_key],
                        position_counts=gs["position_counts"],
                        repetition_known=repetition_known,
                    )

            if (
                _eval_game_is_over(board, config)
                or _eval_ply_cap_reached(gs["move_count"], max_moves)
                or _claim_draw_if_needed(gs)
            ):
                gs["done"] = True

        next_active = []
        for gs in active_games:
            if gs["done"]:
                _finish_game(gs)
            else:
                next_active.append(gs)
        active_games = next_active
        _fill_active()
        _emit_runtime_progress()

    for gs in active_games:
        for session_key in ("model1_session", "model2_session"):
            session = gs.get(session_key)
            if session is not None:
                session.close()
                gs[session_key] = None

    stats = _build_eval_stats(wins, draws, losses, unresolved, len(game_indices))
    stats["completed"] = int(completed)
    stats["terminal_reasons"] = terminal_reasons
    stats["cancelled"] = bool(
        stop_event is not None
        and stop_event.is_set()
        and completed < len(game_indices)
    )
    if stats["cancelled"]:
        stats["cancel_reason"] = "stop requested during game evaluation"
        stats["cancel_stage"] = "game evaluation"
    profile = {}
    for mcts in (mcts1, mcts2):
        if mcts is None:
            continue
        for key, value in dict(mcts.get_profile_stats() or {}).items():
            if isinstance(value, (int, float)):
                profile[key] = profile.get(key, 0) + value
    for key, value in eval_search_profile.items():
        profile[key] = profile.get(key, 0) + value
    stats["profile"] = profile
    return stats


def _result_for_model1(model1_as_white, result):
    if result == "1/2-1/2":
        return 0, 1, 0
    if result == "1-0":
        return (1, 0, 0) if model1_as_white else (0, 0, 1)
    if result == "0-1":
        return (1, 0, 0) if not model1_as_white else (0, 0, 1)
    return 0, 0, 0


def _eval_worker(
    rank,
    model1_state,
    model2_state,
    config,
    device_str,
    game_indices,
    result_queue,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
):
    try:
        worker_config = dict(config)
        worker_config["model"] = dict(config.get("model", {}))
        worker_config["model"]["print_summary"] = False

        device = torch.device(device_str)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        model1 = create_model(worker_config).to(device)
        model2 = create_model(worker_config).to(device)
        if device.type == "cuda":
            model1 = model1.to(memory_format=torch.channels_last)
            model2 = model2.to(memory_format=torch.channels_last)

        model1.load_state_dict(model1_state)
        model2.load_state_dict(model2_state)
        model1.eval()
        model2.eval()

        stats = _evaluate_games_batched(
            model1,
            model2,
            worker_config,
            device,
            game_indices,
            use_fixed_openings=use_fixed_openings,
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0, plies=0, result="*", model1_as_white=True: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
                "plies": int(plies),
                "result": str(result),
                "model1_as_white": bool(model1_as_white),
            }),
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
                "terminal_reasons": dict(stats.get("terminal_reasons", {}) or {}),
                "profile": dict(stats.get("profile", {}) or {}),
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


def _eval_central_worker(
    rank,
    config,
    game_indices,
    request_queue,
    response_receiver,
    result_queue,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    model1_label="eval_model1",
    model2_label="eval_model2",
    shared_buffer=None,
    model1_spec=None,
    model2_spec=None,
):
    try:
        # These processes own only boards and MCTS trees.  The one central
        # server process is the sole CUDA owner during evaluation.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        worker_config = _cpu_only_eval_worker_config(config)
        rl_cfg = worker_config.get("reinforcement_learning", {})
        torch_threads = max(1, int(rl_cfg.get("eval_torch_threads", rl_cfg.get("self_play_torch_threads", 1)) or 1))
        with contextlib.suppress(Exception):
            torch.set_num_threads(torch_threads)
        with contextlib.suppress(Exception):
            torch.set_num_interop_threads(1)

        timeout_s = float(_central_inference_option(config, "timeout_s", 0) or 0)
        stall_warning_s = float(_central_inference_option(config, "stall_warning_s", 15) or 0)
        transport_dtype = str(_central_inference_option(config, "transport_dtype", "float16") or "float16")
        debug_enabled = bool(rl_cfg.get("eval_central_inference_debug", False))
        shared_call_lock = threading.Lock()
        model1_search_config = model1_mcts_config or worker_config
        model2_search_config = model2_mcts_config or worker_config
        if isinstance(model1_spec, dict):
            model1_search_config = config_for_model_spec(
                model1_search_config, model1_spec
            )
        if isinstance(model2_spec, dict):
            model2_search_config = config_for_model_spec(
                model2_search_config, model2_spec
            )

        model1 = RemoteInferenceClient(
            str(model1_label),
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
            shared_buffer=shared_buffer,
            shared_call_lock=shared_call_lock,
            model_spec=model1_spec,
        )
        model2 = RemoteInferenceClient(
            str(model2_label),
            request_queue,
            response_receiver,
            worker_rank=int(rank),
            timeout_s=timeout_s,
            stall_warning_s=stall_warning_s,
            debug_enabled=debug_enabled,
            transport_dtype=transport_dtype,
            shared_buffer=shared_buffer,
            shared_call_lock=shared_call_lock,
            model_spec=model2_spec,
        )

        stats = _evaluate_games_batched(
            model1,
            model2,
            worker_config,
            torch.device("cpu"),
            game_indices,
            use_fixed_openings=use_fixed_openings,
            progress_callback=lambda completed, game_idx=None, wins=0, draws=0, losses=0, unresolved=0, plies=0, result="*", model1_as_white=True: result_queue.put({
                "type": "progress",
                "rank": rank,
                "completed": int(completed),
                "game_idx": None if game_idx is None else int(game_idx),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
                "unresolved": int(unresolved),
                "plies": int(plies),
                "result": str(result),
                "model1_as_white": bool(model1_as_white),
            }),
            model1_mcts_config=model1_search_config,
            model2_mcts_config=model2_search_config,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
        )

        result_queue.put(
            {
                "type": "result",
                "rank": rank,
                "wins": int(stats.get("wins", 0)),
                "draws": int(stats.get("draws", 0)),
                "losses": int(stats.get("losses", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
                "terminal_reasons": dict(stats.get("terminal_reasons", {}) or {}),
                "profile": dict(stats.get("profile", {}) or {}),
            }
        )
    except KeyboardInterrupt:
        result_queue.put({"type": "interrupt", "rank": rank})
    except Exception as exc:
        result_queue.put({"type": "error", "rank": rank, "error": str(exc)})


def _close_eval_central_runtime(runtime):
    if not runtime or runtime.get("closed"):
        return
    if runtime.get("borrowed"):
        # The persistent self-play pool owns this process and all pipe handles.
        return
    runtime["closed"] = True
    request_queues = list(runtime.get("request_queues", []) or [])
    for request_queue in request_queues:
        with contextlib.suppress(Exception):
            request_queue.put({"cmd": "stop"})
    profile_forward_shapes = bool(runtime.get("profile_forward_shapes", False))
    profile_cuda_graphs = bool(runtime.get("profile_cuda_graphs", False))
    if profile_forward_shapes or profile_cuda_graphs:
        # The opt-in batch-shape diagnostic needs one clean server shutdown to
        # publish its aggregate through the control queue. Keep the normal
        # menu/evaluation shutdown path unchanged and bound this diagnostic
        # grace period to two seconds total.
        _terminate_eval_processes(
            runtime.get("server_processes", []),
            timeout_s=0.2,
            graceful_timeout_s=2.0,
        )
        histograms = []
        for control_queue in list(runtime.get("control_queues", []) or []):
            while True:
                try:
                    message = control_queue.get_nowait()
                except Exception:
                    break
                if message and message.get("type") == "forward_shape_diagnostics":
                    histograms.append({
                        "server_rank": int(message.get("server_rank", -1)),
                        "histogram": dict(message.get("histogram", {}) or {}),
                        "cuda_graph_stats": dict(message.get("cuda_graph_stats", {}) or {}),
                    })
        runtime["forward_shape_histograms"] = histograms
    else:
        _terminate_eval_processes(runtime.get("server_processes", []), timeout_s=0.2)
    for queue_obj in request_queues + list(runtime.get("control_queues", []) or []):
        with contextlib.suppress(Exception):
            queue_obj.close()
    for recv_conn in dict(runtime.get("worker_response_receivers", {}) or {}).values():
        with contextlib.suppress(Exception):
            recv_conn.close()
    for sender_map in list(runtime.get("server_response_senders", []) or []):
        for send_conn in dict(sender_map or {}).values():
            with contextlib.suppress(Exception):
                send_conn.close()
    runtime.get("worker_shared_buffers", {}).clear()


def _eval_central_model_specs(model1, model2, config):
    """Return checkpoint-specific contracts used by the two eval labels."""
    specs = {}
    for label, model in (("eval_model1", model1), ("eval_model2", model2)):
        spec = getattr(model, "model_spec", None)
        if not isinstance(spec, dict):
            spec = model_spec_from_config(config)
        specs[label] = copy.deepcopy(spec)
    return specs


def _start_eval_central_runtime(
    model1,
    model2,
    config,
    device,
    num_games,
    *,
    verbose=False,
    ready_callback=None,
    stop_event=None,
):
    """Start one reusable GPU inference server for all stages of an eval funnel."""
    startup_t0 = time.perf_counter()
    _raise_if_eval_stopped(stop_event, "central server startup")
    workers = _resolve_eval_workers(config, device, num_games)
    server_count = _resolve_eval_central_server_count(config, workers)
    _notify_eval_runtime(
        ready_callback,
        status="starting_server",
        ready=False,
        workers=int(workers),
        server_count=int(server_count),
        central_inference=True,
        central_compile_enabled=bool(_central_inference_option(config, "use_compile", False)),
    )
    ctx = mp.get_context("spawn")
    request_queues = [ctx.Queue() for _ in range(server_count)]
    control_queues = [ctx.Queue() for _ in range(server_count)]
    server_response_senders = [dict() for _ in range(server_count)]
    worker_response_receivers = {}
    worker_server_idx = {}
    for rank in range(workers):
        server_idx = int(rank) % int(server_count)
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        worker_response_receivers[rank] = recv_conn
        server_response_senders[server_idx][rank] = send_conn
        worker_server_idx[rank] = server_idx

    central_cfg = config.get("central_inference", {}) or {}
    rl_cfg = config.get("reinforcement_learning", {}) or {}
    model_specs = _eval_central_model_specs(model1, model2, config)
    shared_memory_enabled = bool(
        central_cfg.get("shared_memory_enabled", True)
        and str(central_cfg.get("transport_dtype", "float16")).lower() in {"float16", "fp16"}
    )
    worker_shared_buffers = {}
    if shared_memory_enabled:
        raw_capacity = central_cfg.get(
            "shared_memory_slot_batch_size",
            rl_cfg.get("mcts_batch_size", rl_cfg.get("mcts_simulations", 192)),
        )
        if isinstance(raw_capacity, str) and raw_capacity.strip().lower() == "auto":
            raw_capacity = rl_cfg.get("mcts_batch_size", rl_cfg.get("mcts_simulations", 192))
        capacity = max(1, int(raw_capacity or 192))
        slots = max(1, int(central_cfg.get("shared_memory_slots_per_worker", 2) or 2))
        input_planes = max(
            int(spec["input_spec"]["input_planes"])
            for spec in model_specs.values()
        )
        worker_shared_buffers = {
            int(rank): create_shared_inference_buffer(
                ctx,
                slots=slots,
                capacity=capacity,
                input_planes=input_planes,
                max_legal_moves=MAX_LEGAL_MOVES,
            )
            for rank in range(workers)
        }

    runtime = {
        "ctx": ctx,
        "workers": workers,
        "server_count": server_count,
        "request_queues": request_queues,
        "control_queues": control_queues,
        "server_response_senders": server_response_senders,
        "worker_response_receivers": worker_response_receivers,
        "worker_server_idx": worker_server_idx,
        "worker_shared_buffers": worker_shared_buffers,
        "model_labels": ("eval_model1", "eval_model2"),
        "model_specs": model_specs,
        "server_processes": [],
        "closed": False,
        "stage_count": 0,
        "profile_forward_shapes": bool(
            (config.get("central_inference", {}) or {}).get("profile_forward_shapes", False)
        ),
        "profile_cuda_graphs": bool(
            (config.get("central_inference", {}) or {}).get("profile_cuda_graphs", False)
        ),
    }
    try:
        for server_idx in range(server_count):
            _raise_if_eval_stopped(stop_event, "central server startup")
            configured_cache_rank = _central_inference_option(config, "compile_cache_rank", None)
            if configured_cache_rank is None:
                compile_rank = (
                    910000
                    + int(device.index or 0) * 100
                    + int(server_idx)
                )
            else:
                compile_rank = int(configured_cache_rank) + int(server_idx)
            server_device = (
                int(device.index or 0) if device.type == "cuda" else "cpu"
            )
            proc = ctx.Process(
                target=central_inference_server,
                args=(
                    config,
                    server_device,
                    request_queues[server_idx],
                    server_response_senders[server_idx],
                    control_queues[server_idx],
                    compile_rank,
                    {
                        int(rank): worker_shared_buffers[int(rank)]
                        for rank, assigned_server in worker_server_idx.items()
                        if int(assigned_server) == int(server_idx)
                        and int(rank) in worker_shared_buffers
                    },
                ),
            )
            proc.daemon = True
            proc.start()
            runtime["server_processes"].append(proc)
        runtime["server_spawn_s"] = time.perf_counter() - startup_t0

        _raise_if_eval_stopped(stop_event, "model snapshot")
        _notify_eval_runtime(
            ready_callback,
            status="snapshotting_models",
            ready=False,
            workers=int(workers),
            server_count=int(server_count),
            central_inference=True,
            central_compile_enabled=bool(_central_inference_option(config, "use_compile", False)),
            server_spawn_s=float(runtime.get("server_spawn_s", 0.0) or 0.0),
            server_pids=[int(proc.pid) for proc in runtime["server_processes"] if proc.pid is not None],
        )
        task_id = f"eval_runtime_{os.getpid()}_{id(model1)}_{id(model2)}"
        snapshot_t0 = time.perf_counter()
        model1_state = _snapshot_state_dict_cpu_shared(model1)
        model2_state = _snapshot_state_dict_cpu_shared(model2)
        runtime["snapshot_s"] = time.perf_counter() - snapshot_t0
        _raise_if_eval_stopped(stop_event, "model snapshot")
        _notify_eval_runtime(
            ready_callback,
            status="loading_models",
            ready=False,
            workers=int(workers),
            server_count=int(server_count),
            central_inference=True,
            central_compile_enabled=bool(_central_inference_option(config, "use_compile", False)),
            server_spawn_s=float(runtime.get("server_spawn_s", 0.0) or 0.0),
            snapshot_s=float(runtime["snapshot_s"]),
            server_pids=[int(proc.pid) for proc in runtime["server_processes"] if proc.pid is not None],
        )
        for request_queue in request_queues:
            request_queue.put({
                "cmd": "load_models",
                "task_id": task_id,
                "clear": True,
                "models": [
                    {
                        "label": "eval_model1",
                        "state": model1_state,
                        "state_path": None,
                        "model_spec": model_specs["eval_model1"],
                    },
                    {
                        "label": "eval_model2",
                        "state": model2_state,
                        "state_path": None,
                        "model_spec": model_specs["eval_model2"],
                    },
                ],
            })

        load_timeout_s = float(_central_inference_option(config, "load_timeout_s", 300) or 300)
        deadline = time.time() + max(1.0, load_timeout_s)
        pending_servers = set(range(server_count))
        load_messages = []
        while pending_servers:
            _raise_if_eval_stopped(stop_event, "central model loading")
            if time.time() >= deadline:
                raise TimeoutError(
                    "Eval central inference did not acknowledge model load "
                    f"from servers {sorted(pending_servers)}."
                )
            for server_idx in list(pending_servers):
                try:
                    message = control_queues[server_idx].get(timeout=0.1)
                except Exception:
                    continue
                if message.get("type") == "models_loaded" and str(message.get("task_id")) == task_id:
                    load_messages.append(dict(message))
                    pending_servers.discard(server_idx)

        load_times = [
            float(message.get("load_s", 0.0) or 0.0)
            for message in load_messages
            if message.get("load_s") is not None
        ]
        model_summary = next(
            (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
            "models ready",
        )
        pid_summary = ",".join(
            str(int(message.get("pid")))
            for message in load_messages
            if message.get("pid") is not None
        )
        runtime["model_summary"] = model_summary
        actual_cudnn_benchmark = (
            bool(load_messages[0].get("cudnn_benchmark"))
            if load_messages and load_messages[0].get("cudnn_benchmark") is not None
            else None
        )
        runtime["central_cudnn_benchmark"] = actual_cudnn_benchmark
        runtime["load_s"] = max(load_times) if load_times else 0.0
        runtime["pid_summary"] = pid_summary
        runtime["startup_s"] = time.perf_counter() - startup_t0
        _notify_eval_runtime(
            ready_callback,
            status="models_loaded",
            ready=False,
            models_ready=True,
            workers=int(workers),
            server_count=int(server_count),
            central_inference=True,
            central_compile_enabled=any(bool(message.get("compile")) for message in load_messages),
            central_cudnn_benchmark=actual_cudnn_benchmark,
            model_summary=model_summary,
            server_spawn_s=float(runtime.get("server_spawn_s", 0.0) or 0.0),
            load_s=float(runtime["load_s"]),
            startup_s=float(runtime["startup_s"]),
            snapshot_s=float(runtime.get("snapshot_s", 0.0) or 0.0),
            server_pids=[int(message["pid"]) for message in load_messages if message.get("pid") is not None],
        )
        if verbose:
            print(
                "Eval inference runtime: "
                f"workers={workers}, servers={server_count}, {model_summary}, "
                f"load={runtime['load_s']:.2f}s, pid={pid_summary or '-'}"
            )
        return runtime
    except Exception:
        _close_eval_central_runtime(runtime)
        raise


def _refresh_eval_central_runtime(
    runtime,
    model1,
    model2,
    config,
    *,
    verbose=False,
    ready_callback=None,
):
    """Replace weights while preserving compiled eval models and GPU process."""
    refresh_t0 = time.perf_counter()
    task_id = f"eval_refresh_{os.getpid()}_{time.time_ns()}"
    snapshot_t0 = time.perf_counter()
    model1_state = _snapshot_state_dict_cpu_shared(model1)
    model2_state = _snapshot_state_dict_cpu_shared(model2)
    model1_label, model2_label = runtime.get(
        "model_labels", ("eval_model1", "eval_model2")
    )
    model_specs = _eval_central_model_specs(model1, model2, config)
    snapshot_s = time.perf_counter() - snapshot_t0
    for request_queue in runtime["request_queues"]:
        request_queue.put({
            "cmd": "load_models",
            "task_id": task_id,
            "clear": True,
            "models": [
                {
                    "label": model1_label,
                    "state": model1_state,
                    "state_path": None,
                    "model_spec": model_specs[model1_label],
                },
                {
                    "label": model2_label,
                    "state": model2_state,
                    "state_path": None,
                    "model_spec": model_specs[model2_label],
                },
            ],
        })

    load_timeout_s = float(_central_inference_option(config, "load_timeout_s", 300) or 300)
    deadline = time.time() + max(1.0, load_timeout_s)
    pending_servers = set(range(int(runtime["server_count"])))
    load_messages = []
    while pending_servers:
        if time.time() >= deadline:
            raise TimeoutError(
                "Eval central inference did not acknowledge refreshed models "
                f"from servers {sorted(pending_servers)}."
            )
        for server_idx in list(pending_servers):
            try:
                message = runtime["control_queues"][server_idx].get(timeout=0.25)
            except Exception:
                continue
            if message.get("type") == "models_loaded" and str(message.get("task_id")) == task_id:
                load_messages.append(dict(message))
                pending_servers.discard(server_idx)

    load_times = [
        float(message.get("load_s", 0.0) or 0.0)
        for message in load_messages
        if message.get("load_s") is not None
    ]
    runtime["model_summary"] = next(
        (str(message.get("model_summary")) for message in load_messages if message.get("model_summary")),
        "models ready",
    )
    runtime["pid_summary"] = ",".join(
        str(int(message.get("pid")))
        for message in load_messages
        if message.get("pid") is not None
    )
    runtime["load_s"] = max(load_times) if load_times else 0.0
    actual_cudnn_benchmark = (
        bool(load_messages[0].get("cudnn_benchmark"))
        if load_messages and load_messages[0].get("cudnn_benchmark") is not None
        else None
    )
    runtime["central_cudnn_benchmark"] = actual_cudnn_benchmark
    runtime["model_specs"] = model_specs
    runtime["snapshot_s"] = snapshot_s
    runtime["startup_s"] = time.perf_counter() - refresh_t0
    if ready_callback is not None:
        ready_callback({
            "ready": False,
            "models_ready": True,
            "workers": int(runtime["workers"]),
            "server_count": int(runtime["server_count"]),
            "central_inference": True,
            "central_cudnn_benchmark": actual_cudnn_benchmark,
            "load_s": float(runtime["load_s"]),
            "startup_s": float(runtime["startup_s"]),
            "snapshot_s": float(snapshot_s),
        })
    if verbose:
        print(
            "Eval inference runtime: "
            f"workers={runtime['workers']}, servers={runtime['server_count']}, "
            f"{runtime['model_summary']}, load={runtime['load_s']:.2f}s, "
            f"pid={runtime['pid_summary'] or '-'}"
        )
    return runtime


def prepare_eval_central_runtime(runtime, model1, model2, config, device, num_games):
    """Start once, then refresh weights without recompiling between iterations."""
    if not _eval_uses_central_inference(config, device):
        if runtime is not None:
            _close_eval_central_runtime(runtime)
        return None

    workers = _resolve_eval_workers(config, device, num_games)
    server_count = _resolve_eval_central_server_count(config, workers)
    desired_model_specs = _eval_central_model_specs(model1, model2, config)
    runtime_workers = int(runtime.get("workers", 0)) if runtime is not None else 0
    worker_topology_matches = bool(
        runtime_workers == int(workers)
        or (
            runtime is not None
            and runtime.get("borrowed")
            and runtime_workers >= int(workers)
        )
    )
    reusable = bool(
        runtime is not None
        and not runtime.get("closed", False)
        and runtime.get("model_specs") == desired_model_specs
        and worker_topology_matches
        and int(runtime.get("server_count", 0)) == int(server_count)
        and all(process.is_alive() for process in runtime.get("server_processes", []))
    )
    if not reusable:
        if runtime is not None:
            _close_eval_central_runtime(runtime)
        return _start_eval_central_runtime(model1, model2, config, device, num_games)
    return _refresh_eval_central_runtime(runtime, model1, model2, config)


def close_eval_central_runtime(runtime):
    _close_eval_central_runtime(runtime)


def _evaluate_models_with_central_inference(
    model1,
    model2,
    config,
    device,
    num_games,
    game_index_offset=0,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    progress_desc="Eval vs best",
    central_runtime=None,
    progress_callback=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
    show_progress=True,
    verbose=False,
    runtime_ready_callback=None,
):
    owns_runtime = central_runtime is None
    rl_cfg = config.get("reinforcement_learning", {})
    max_moves = _resolve_eval_max_moves(config)
    worker_budget = (
        int(central_runtime["workers"])
        if central_runtime is not None
        else _resolve_eval_workers(config, device, num_games)
    )
    workers = max(1, min(int(num_games), int(worker_budget)))
    ctx = central_runtime["ctx"] if central_runtime is not None else mp.get_context("spawn")
    result_queue = ctx.Queue()
    game_indices_per_worker = [[] for _ in range(workers)]
    for offset, game_idx in enumerate(range(game_index_offset, game_index_offset + num_games)):
        game_indices_per_worker[offset % workers].append(game_idx)

    active_worker_ranks = [
        rank for rank, game_indices in enumerate(game_indices_per_worker)
        if game_indices
    ]
    worker_processes = []
    worker_processes_by_rank = {}
    restart_counts = {int(rank): 0 for rank in active_worker_ranks}
    restart_limit = max(0, int(rl_cfg.get("eval_worker_restart_limit", 2) or 0))
    pending_game_indices_by_rank = {
        int(rank): list(game_indices_per_worker[int(rank)])
        for rank in active_worker_ranks
    }
    total_game_count_by_rank = {
        int(rank): len(game_indices_per_worker[int(rank)])
        for rank in active_worker_ranks
    }
    completed_game_indices_by_rank = {int(rank): set() for rank in active_worker_ranks}

    def _build_eval_worker(active_runtime, rank, game_indices):
        request_queues = active_runtime["request_queues"]
        worker_response_receivers = active_runtime["worker_response_receivers"]
        worker_server_idx = active_runtime["worker_server_idx"]
        model1_label, model2_label = active_runtime.get(
            "model_labels", ("eval_model1", "eval_model2")
        )
        return ctx.Process(
            target=_eval_central_worker,
            args=(
                rank,
                config,
                list(game_indices),
                request_queues[worker_server_idx[rank]],
                worker_response_receivers[rank],
                result_queue,
                use_fixed_openings,
                model1_mcts_config,
                model2_mcts_config,
                use_mcts_model1,
                use_mcts_model2,
                model1_label,
                model2_label,
                dict(active_runtime.get("worker_shared_buffers", {}) or {}).get(int(rank)),
                dict(active_runtime.get("model_specs", {}) or {}).get(model1_label),
                dict(active_runtime.get("model_specs", {}) or {}).get(model2_label),
            ),
        )

    def _register_started_worker(rank, proc):
        worker_processes_by_rank[int(rank)] = proc
        if proc not in worker_processes:
            worker_processes.append(proc)

    def _start_eval_worker(active_runtime, rank, game_indices):
        proc = _build_eval_worker(active_runtime, rank, game_indices)
        proc.daemon = True
        proc.start()
        _register_started_worker(rank, proc)
        return proc

    def _start_all_workers(active_runtime):
        _raise_if_eval_stopped(stop_event, "worker startup")
        pending = [
            (int(rank), _build_eval_worker(active_runtime, rank, game_indices_per_worker[rank]))
            for rank in active_worker_ranks
            if int(rank) not in worker_processes_by_rank
        ]
        for _rank, proc in pending:
            proc.daemon = True
        parallel_start = bool(rl_cfg.get("eval_parallel_worker_start", False)) and len(pending) > 1
        if parallel_start:
            start_threads = max(
                1,
                min(len(pending), int(rl_cfg.get("eval_parallel_worker_start_threads", 4) or 4)),
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=start_threads) as executor:
                    futures = [executor.submit(proc.start) for _rank, proc in pending]
                    for future in futures:
                        future.result()
                _raise_if_eval_stopped(stop_event, "worker startup")
            except Exception:
                _terminate_eval_processes([proc for _rank, proc in pending])
                raise
        else:
            for _rank, proc in pending:
                _raise_if_eval_stopped(stop_event, "worker startup")
                proc.start()
        for rank, proc in pending:
            _register_started_worker(rank, proc)
        return list(worker_processes)

    runtime = central_runtime
    try:
        if runtime is None:
            try:
                runtime = _start_eval_central_runtime(
                    model1,
                    model2,
                    config,
                    device,
                    num_games,
                    verbose=verbose,
                    ready_callback=runtime_ready_callback,
                    stop_event=stop_event,
                )
            except _EvalCancelled as exc:
                stats = _build_eval_stats(0, 0, 0, 0, num_games)
                stats.update({
                    "cancelled": True,
                    "cancel_reason": str(exc),
                    "cancel_stage": exc.stage,
                    "completed": 0,
                })
                return stats
        _raise_if_eval_stopped(stop_event, "worker startup")
        _notify_eval_runtime(
            runtime_ready_callback,
            status="starting_workers",
            ready=False,
            workers=len(active_worker_ranks),
            server_count=int(runtime.get("server_count", 0) or 0),
            central_inference=True,
            server_pids=[int(proc.pid) for proc in runtime.get("server_processes", []) if proc.pid is not None],
            central_cudnn_benchmark=runtime.get("central_cudnn_benchmark"),
        )
        worker_launch_t0 = time.perf_counter()
        _start_all_workers(runtime)
        runtime["worker_launch_s"] = time.perf_counter() - worker_launch_t0
        runtime["startup_s"] = (
            float(runtime.get("startup_s", 0.0) or 0.0)
            + float(runtime["worker_launch_s"])
        )

        server_count = int(runtime["server_count"])
        server_processes = runtime["server_processes"]
        runtime["stage_count"] = int(runtime.get("stage_count", 0)) + 1
        if verbose:
            print(
                "Eval stage: "
                f"workers={len(active_worker_ranks)}, servers={server_count}, "
                f"batch_games={_resolve_eval_batch_games(config, num_games)}, "
                f"sims={_resolve_eval_mcts_simulations(config)}, "
                f"server={'new' if owns_runtime else 'reused'}"
            )

        _notify_eval_runtime(
            runtime_ready_callback,
            status="running",
            ready=True,
            models_ready=True,
            workers=len(active_worker_ranks),
            server_count=int(server_count),
            central_inference=True,
            load_s=float(runtime.get("load_s", 0.0) or 0.0),
            startup_s=float(runtime.get("startup_s", 0.0) or 0.0),
            snapshot_s=float(runtime.get("snapshot_s", 0.0) or 0.0),
            worker_launch_s=float(runtime.get("worker_launch_s", 0.0) or 0.0),
            server_pids=[int(proc.pid) for proc in server_processes if proc.pid is not None],
            worker_pids=[int(proc.pid) for proc in worker_processes if proc.pid is not None],
            central_cudnn_benchmark=runtime.get("central_cudnn_benchmark"),
        )

        wins = 0
        draws = 0
        losses = 0
        unresolved = 0
        completed = 0
        terminal_reasons = {}
        finished_worker_ranks = set()
        eval_profile = {}
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
        worker_error = None
        infrastructure_error = None
        cancelled = False
        try:
            while len(finished_worker_ranks) < len(active_worker_ranks):
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    _notify_eval_runtime(
                        runtime_ready_callback,
                        status="cancelling",
                        ready=False,
                        cancel_stage="game evaluation",
                        completed=int(completed),
                        requested_games=int(num_games),
                        server_pids=[int(proc.pid) for proc in server_processes if proc.pid is not None],
                        worker_pids=[int(proc.pid) for proc in worker_processes if proc.pid is not None],
                    )
                    _terminate_eval_processes(list(worker_processes_by_rank.values()))
                    break
                try:
                    message = result_queue.get(timeout=0.1)
                except Exception:
                    dead_workers = [
                        (rank, proc.exitcode)
                        for rank, proc in list(worker_processes_by_rank.items())
                        if (
                            rank not in finished_worker_ranks
                            and proc is not None
                            and not proc.is_alive()
                            and proc.exitcode not in (0, None)
                        )
                    ]
                    dead_servers = [
                        proc.exitcode for proc in server_processes
                        if proc is not None and not proc.is_alive() and proc.exitcode not in (0, None)
                    ]
                    if dead_workers:
                        for rank, exitcode in dead_workers:
                            completed_for_rank = completed_game_indices_by_rank.setdefault(int(rank), set())
                            remaining = [
                                int(game_idx)
                                for game_idx in pending_game_indices_by_rank.get(int(rank), [])
                                if int(game_idx) not in completed_for_rank
                            ]
                            if not remaining:
                                finished_worker_ranks.add(int(rank))
                                continue
                            restart_counts[int(rank)] = int(restart_counts.get(int(rank), 0)) + 1
                            if restart_counts[int(rank)] > restart_limit:
                                worker_error = (
                                    "eval worker exited unexpectedly after "
                                    f"{restart_limit} restart attempt(s): rank={rank}, "
                                    f"exitcode={exitcode}, remaining={len(remaining)}"
                                )
                                _terminate_eval_processes(list(worker_processes_by_rank.values()))
                                break
                            print(
                                "Warning: eval worker "
                                f"{rank} exited with code {exitcode}; "
                                f"completed={len(completed_for_rank)}/"
                                f"{int(total_game_count_by_rank.get(int(rank), 0))}, "
                                f"remaining={len(remaining)}, "
                                f"restart={restart_counts[int(rank)]}/{restart_limit}."
                            )
                            old_proc = worker_processes_by_rank.get(int(rank))
                            if old_proc is not None:
                                _terminate_eval_processes([old_proc], timeout_s=0.05)
                            pending_game_indices_by_rank[int(rank)] = remaining
                            _start_eval_worker(runtime, int(rank), remaining)
                        if worker_error is not None:
                            break
                    if dead_servers:
                        infrastructure_error = (
                            "eval central inference server exited unexpectedly: "
                            f"exitcodes={dead_servers}"
                        )
                        _terminate_eval_processes(list(worker_processes_by_rank.values()))
                        break
                    continue
                message_type = message.get("type")
                if message_type == "progress":
                    rank = int(message.get("rank", -1))
                    game_idx = message.get("game_idx", None)
                    counted = False
                    if game_idx is not None:
                        try:
                            game_idx = int(game_idx)
                        except (TypeError, ValueError):
                            game_idx = None
                    if game_idx is not None:
                        completed_for_rank = completed_game_indices_by_rank.setdefault(rank, set())
                        if game_idx not in completed_for_rank:
                            completed_for_rank.add(game_idx)
                            counted = True
                    else:
                        counted = True
                    if counted:
                        completed += int(message.get("completed", 0))
                        wins += int(message.get("wins", 0))
                        draws += int(message.get("draws", 0))
                        losses += int(message.get("losses", 0))
                        unresolved += int(message.get("unresolved", 0))
                        eval_bar.n = min(num_games, completed)
                        eval_bar.refresh()
                        if progress_callback is not None:
                            progress_callback(
                                int(message.get("completed", 0)),
                                game_idx,
                                int(message.get("wins", 0)),
                                int(message.get("draws", 0)),
                                int(message.get("losses", 0)),
                                int(message.get("unresolved", 0)),
                                int(message.get("plies", 0)),
                                str(message.get("result", "*")),
                                bool(message.get("model1_as_white", True)),
                            )
                elif message_type == "result":
                    rank = int(message.get("rank", -1))
                    _merge_eval_terminal_reasons(
                        terminal_reasons, message.get("terminal_reasons")
                    )
                    for key, value in dict(message.get("profile", {}) or {}).items():
                        if isinstance(value, (int, float)):
                            eval_profile[key] = eval_profile.get(key, 0) + value
                    finished_worker_ranks.add(rank)
                elif message_type == "interrupt":
                    raise KeyboardInterrupt
                elif message_type == "error":
                    rank = int(message.get("rank", -1))
                    error_text = str(message.get("error", "unknown error"))
                    if is_transient_cuda_error(error_text):
                        # Every worker talks to the same GPU server. Restarting
                        # individual CPU workers cannot repair its CUDA context
                        # and previously multiplied the same failure 12 times.
                        infrastructure_error = error_text
                        _terminate_eval_processes(list(worker_processes_by_rank.values()))
                        break
                    completed_for_rank = completed_game_indices_by_rank.setdefault(rank, set())
                    remaining = [
                        int(game_idx)
                        for game_idx in pending_game_indices_by_rank.get(rank, [])
                        if int(game_idx) not in completed_for_rank
                    ]
                    restart_counts[rank] = int(restart_counts.get(rank, 0)) + 1
                    if not remaining:
                        finished_worker_ranks.add(rank)
                    elif restart_counts[rank] <= restart_limit:
                        print(
                            "Warning: eval worker "
                            f"{rank} reported error; remaining={len(remaining)}, "
                            f"restart={restart_counts[rank]}/{restart_limit}: {error_text}"
                        )
                        old_proc = worker_processes_by_rank.get(rank)
                        if old_proc is not None:
                            _terminate_eval_processes([old_proc], timeout_s=0.05)
                        pending_game_indices_by_rank[rank] = remaining
                        _start_eval_worker(runtime, rank, remaining)
                    else:
                        worker_error = (
                            "eval worker failed after "
                            f"{restart_limit} restart attempt(s): rank={rank}, error={error_text}"
                        )
                        _terminate_eval_processes(list(worker_processes_by_rank.values()))
                        break
        finally:
            eval_bar.close()

        if infrastructure_error is not None:
            stats = _build_eval_stats(0, 0, 0, 0, 0)
            stats.update({
                "infrastructure_error": str(infrastructure_error),
                "error": str(infrastructure_error),
                "transient_cuda_error": bool(is_transient_cuda_error(infrastructure_error)),
                "completed_before_failure": int(completed),
                "requested_games": int(num_games),
                "profile": eval_profile,
            })
            return stats
        if worker_error is not None:
            raise RuntimeError(f"Eval worker failed: {worker_error}")
        if unresolved > 0 and verbose:
            _print_eval_unresolved(
                "Eval",
                unresolved,
                num_games,
                max_moves,
                neutral_score=_resolve_eval_unresolved_score_neutral(config),
            )
        stats = _build_eval_stats(wins, draws, losses, unresolved, num_games)
        stats["profile"] = eval_profile
        stats["terminal_reasons"] = terminal_reasons
        requests = int(eval_profile.get("central_inference_requests", 0) or 0)
        if requests > 0 and verbose:
            remote_ms = 1000.0 * float(
                eval_profile.get("central_inference_remote_wait_time", 0.0) or 0.0
            ) / requests
            queue_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_queue_wait_time", 0.0) or 0.0
            ) / requests
            service_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_total_time", 0.0) or 0.0
            ) / requests
            forward_ms = 1000.0 * float(
                eval_profile.get("central_inference_server_forward_time", 0.0) or 0.0
            ) / requests
            server_batch = float(
                eval_profile.get("central_inference_server_batch_items", 0.0) or 0.0
            ) / requests
            gpu_batch_fill = float(
                eval_profile.get("central_inference_gpu_batch_fill_sum", 0.0) or 0.0
            ) / requests
            descriptor_ms = 1000.0 * float(
                eval_profile.get(
                    "central_inference_server_descriptor_queue_wait_time", 0.0
                ) or 0.0
            ) / requests
            coalesce_ms = 1000.0 * float(
                eval_profile.get(
                    "central_inference_server_batch_coalesce_wait_time", 0.0
                ) or 0.0
            ) / requests
            shared_slot_ms = 1000.0 * float(
                eval_profile.get("central_inference_shared_slot_wait_time", 0.0) or 0.0
            ) / requests
            search_time = float(eval_profile.get("search_many_time", 0.0) or 0.0)
            inference_wait = float(eval_profile.get("nn_inference_time", 0.0) or 0.0)
            wait_share = inference_wait / search_time if search_time > 0.0 else 0.0
            print(
                "Eval pipeline: "
                f"batch={server_batch:.1f} ({gpu_batch_fill:.0%} fill) | "
                f"response={remote_ms:.2f}ms "
                f"(queue/IPC={queue_ms:.2f}ms: descriptor={descriptor_ms:.2f}, "
                f"coalesce={coalesce_ms:.2f}; slot={shared_slot_ms:.2f}ms; "
                f"server GPU+copies={service_ms:.2f}ms, "
                f"forward async={forward_ms:.2f}ms) | "
                f"MCTS waiting for inference={wait_share:.0%}"
            )
        stats["cancelled"] = bool(cancelled)
        if cancelled:
            stats["cancel_reason"] = "stop requested during game evaluation"
            stats["cancel_stage"] = "game evaluation"
        stats["completed"] = int(completed)
        return stats
    except _EvalCancelled as exc:
        stats = _build_eval_stats(0, 0, 0, 0, num_games)
        stats.update({
            "cancelled": True,
            "cancel_reason": str(exc),
            "cancel_stage": exc.stage,
            "completed": 0,
        })
        return stats
    finally:
        _terminate_eval_processes(
            worker_processes,
            timeout_s=0.1,
            graceful_timeout_s=0.4,
        )
        with contextlib.suppress(Exception):
            result_queue.cancel_join_thread()
        with contextlib.suppress(Exception):
            result_queue.close()
        if owns_runtime:
            _close_eval_central_runtime(runtime)
            if (
                isinstance(runtime, dict)
                and (
                    bool(runtime.get("profile_forward_shapes", False))
                    or bool(runtime.get("profile_cuda_graphs", False))
                )
                and isinstance(locals().get("stats"), dict)
            ):
                diagnostic_results = list(runtime.get("forward_shape_histograms", []) or [])
                eval_profile = stats.setdefault("profile", {})
                if bool(runtime.get("profile_forward_shapes", False)):
                    eval_profile["central_forward_shape_histogram"] = [
                        {"server_rank": item["server_rank"], "histogram": item["histogram"]}
                        for item in diagnostic_results
                    ]
                if bool(runtime.get("profile_cuda_graphs", False)):
                    eval_profile["central_cuda_graph_stats"] = [
                        {"server_rank": item["server_rank"], "stats": item["cuda_graph_stats"]}
                        for item in diagnostic_results
                    ]


def _get_hflip_inverse_index_map():
    global _HFLIP_INV_INDEX_MAP
    if _HFLIP_INV_INDEX_MAP is not None:
        return _HFLIP_INV_INDEX_MAP

    inverse_map = build_hflip_inverse_index_map()
    if len(inverse_map) != ACTION_SIZE:
        raise ValueError(f"Horizontal flip map length mismatch: got {len(inverse_map)}, expected {ACTION_SIZE}")
    _HFLIP_INV_INDEX_MAP = torch.from_numpy(inverse_map).long()
    return _HFLIP_INV_INDEX_MAP


def _get_hflip_forward_index_map():
    global _HFLIP_FWD_INDEX_MAP
    if _HFLIP_FWD_INDEX_MAP is not None:
        return _HFLIP_FWD_INDEX_MAP

    inv_map = _get_hflip_inverse_index_map()
    forward_map = torch.empty_like(inv_map)
    forward_map[inv_map] = torch.arange(inv_map.numel(), dtype=inv_map.dtype)
    _HFLIP_FWD_INDEX_MAP = forward_map
    return _HFLIP_FWD_INDEX_MAP


def _horizontal_flip_eligible_mask(boards, *, input_encoder_id=None, history_positions=None):
    """Return rows where file mirroring is an exact chess symmetry.

    Castling is asymmetric around the a<->h reflection (the king starts on the
    e-file, not on the centre line).  V9 stores castling in plane 12 of every
    16-plane frame; v10 stores current castling rights in metadata planes
    13:17 after H chronological 13-plane history frames and the current
    13-plane state.  Select offsets from the encoder contract, never by
    assuming one layout for both profiles.
    """
    if boards.dim() != 4:
        return torch.ones(int(boards.size(0)), dtype=torch.bool, device=boards.device)
    if input_encoder_id == "planes_history_v2":
        if history_positions is None:
            raise ValueError("v10 augmentation requires history_positions from the model input spec")
        castling_start = int(history_positions) * 13 + 13
        castling_end = castling_start + 4
        if int(boards.size(1)) < castling_end:
            raise ValueError(
                f"v10 augmentation expected castling planes {castling_start}:{castling_end}, "
                f"got {int(boards.size(1))} input planes"
            )
        castling_planes = boards[:, castling_start:castling_end]
    else:
        if int(boards.size(1)) < 16:
            return torch.ones(int(boards.size(0)), dtype=torch.bool, device=boards.device)
        castling_planes = boards[:, 12::16]
    has_castling_rights = castling_planes.abs().flatten(1).amax(dim=1) > 0
    return ~has_castling_rights


def _maybe_augment_batch(
    boards,
    policy_indices,
    policy_values,
    policy_mask,
    legal_indices,
    legal_mask,
    config,
    *,
    input_encoder_id=None,
    history_positions=None,
):
    rl_cfg = config.get("reinforcement_learning", {})
    if not rl_cfg.get("use_augmentation", False):
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask
    if not rl_cfg.get("augment_horizontal_flip", False):
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    prob = rl_cfg.get("augment_prob", 0.5)
    if prob <= 0:
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    batch_size = boards.size(0)
    if batch_size == 0:
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    flip_mask = (
        torch.rand(batch_size, device=boards.device) < prob
    ) & _horizontal_flip_eligible_mask(
        boards,
        input_encoder_id=input_encoder_id,
        history_positions=history_positions,
    )
    if not flip_mask.any():
        return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask

    boards[flip_mask] = torch.flip(boards[flip_mask], dims=[3])

    if policy_indices.numel() > 0:
        forward_map = _get_hflip_forward_index_map().to(device=policy_indices.device)
        active = (
            flip_mask.unsqueeze(1)
            & policy_mask
            & (policy_indices >= 0)
            & (policy_indices < int(ACTION_SIZE))
        )
        if active.any():
            remapped = forward_map[policy_indices[active].long()].to(dtype=policy_indices.dtype)
            policy_indices[active] = remapped

        # Older replay rows use policy support as their legal support and may
        # therefore alias the same tensor.  Do not mirror shared storage twice.
        shares_policy_storage = (
            legal_indices is policy_indices
            or (
                legal_indices is not None
                and legal_indices.numel() > 0
                and policy_indices.numel() > 0
                and legal_indices.data_ptr() == policy_indices.data_ptr()
            )
        )
        if legal_indices is not None and legal_indices.numel() > 0 and not shares_policy_storage:
            active_legal = (
                flip_mask.unsqueeze(1)
                & legal_mask
                & (legal_indices >= 0)
                & (legal_indices < int(ACTION_SIZE))
            )
            if active_legal.any():
                remapped_legal = forward_map[legal_indices[active_legal].long()].to(
                    dtype=legal_indices.dtype
                )
                legal_indices[active_legal] = remapped_legal

    return boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask


def _final_outcome_targets(value_targets, epsilon=1e-6):
    """Map RL value targets to final W/D/L outcomes from the side-to-move POV."""
    target = value_targets.view(-1)
    return torch.where(
        torch.abs(target) <= float(epsilon),
        torch.zeros_like(target),
        torch.sign(target),
    )


def _wdl_targets_from_final_outcome(target_scalar):
    target_win = (target_scalar > 0.0).to(dtype=target_scalar.dtype)
    target_draw = (target_scalar == 0.0).to(dtype=target_scalar.dtype)
    target_loss = (target_scalar < 0.0).to(dtype=target_scalar.dtype)
    return torch.stack((target_win, target_draw, target_loss), dim=1)


def _legal_only_sparse_policy_loss(
    policy_logits,
    policy_indices,
    policy_values,
    policy_mask,
    legal_indices,
    legal_mask,
):
    batch_size = int(policy_logits.size(0))
    num_classes = int(policy_logits.size(1))
    if policy_indices.numel() == 0:
        return torch.zeros(batch_size, device=policy_logits.device, dtype=policy_logits.dtype)

    if legal_indices is None or legal_indices.numel() == 0:
        legal_indices = policy_indices
        legal_mask = policy_mask

    safe_policy_indices = policy_indices.long().clamp(0, num_classes - 1)
    valid_policy_mask = policy_mask & (policy_indices >= 0) & (policy_indices < num_classes)
    dense_targets = torch.zeros(
        (batch_size, num_classes),
        device=policy_logits.device,
        dtype=policy_logits.float().dtype,
    )
    dense_targets.scatter_add_(
        1,
        safe_policy_indices,
        torch.where(valid_policy_mask, policy_values.float(), torch.zeros_like(policy_values.float())),
    )

    safe_legal_indices = legal_indices.long().clamp(0, num_classes - 1)
    valid_legal_mask = legal_mask & (legal_indices >= 0) & (legal_indices < num_classes)
    legal_logits = torch.gather(policy_logits.float(), 1, safe_legal_indices)
    legal_logits = legal_logits.masked_fill(~valid_legal_mask, -1.0e9)
    legal_log_probs = F.log_softmax(legal_logits, dim=1)

    legal_targets = torch.gather(dense_targets, 1, safe_legal_indices)
    legal_targets = torch.where(valid_legal_mask, legal_targets, torch.zeros_like(legal_targets))
    target_mass = legal_targets.sum(dim=1, keepdim=True)
    legal_targets = torch.where(
        target_mass > 0.0,
        legal_targets / target_mass.clamp_min(1e-8),
        legal_targets,
    )
    return -(legal_targets * legal_log_probs).sum(dim=1).to(dtype=policy_logits.dtype)


def _legal_only_log_probs(policy_logits, legal_indices, legal_mask):
    """Normalize policy logits over each position's legal action set."""
    num_classes = int(policy_logits.size(1))
    safe_legal_indices = legal_indices.long().clamp(0, num_classes - 1)
    valid_legal_mask = legal_mask & (legal_indices >= 0) & (legal_indices < num_classes)
    legal_logits = torch.gather(policy_logits.float(), 1, safe_legal_indices)
    legal_logits = legal_logits.masked_fill(~valid_legal_mask, -1.0e9)
    legal_log_probs = F.log_softmax(legal_logits, dim=1)
    legal_log_probs = torch.where(
        valid_legal_mask,
        legal_log_probs,
        torch.zeros_like(legal_log_probs),
    )
    return legal_log_probs, valid_legal_mask, safe_legal_indices


def _weighted_mean(losses, weights):
    losses = losses.reshape(-1)
    weights = torch.clamp(
        weights.reshape(-1).to(device=losses.device, dtype=losses.dtype),
        min=0.0,
    )
    weight_total = weights.sum()
    # Keep zero-weight batches at zero without synchronizing a CUDA scalar to
    # Python. For every positive sum this is the same weighted mean.
    denominator = torch.where(
        weight_total > 0.0,
        weight_total,
        torch.ones_like(weight_total),
    )
    return (losses * weights).sum() / denominator


def _useful_policy_correction_rank_loss(
    legal_log_probs,
    valid_legal_mask,
    safe_legal_indices,
    target_moves,
    correction_mask,
    search_q_deltas,
    policy_weights,
):
    """Give a verified MCTS winner a small margin over the best alternative."""
    batch_size = int(legal_log_probs.size(0))
    zero = legal_log_probs.sum() * 0.0
    per_row = torch.zeros(
        batch_size,
        device=legal_log_probs.device,
        dtype=legal_log_probs.dtype,
    )
    if batch_size <= 0:
        return zero, per_row, torch.zeros_like(correction_mask)

    target_slots = valid_legal_mask & safe_legal_indices.eq(target_moves.reshape(-1, 1))
    other_slots = valid_legal_mask & ~target_slots
    valid_rows = (
        correction_mask & target_slots.any(dim=1) & other_slots.any(dim=1)
        & torch.isfinite(search_q_deltas)
    )
    negative_inf = torch.finfo(legal_log_probs.dtype).min
    target_scores = legal_log_probs.masked_fill(~target_slots, negative_inf).max(dim=1).values
    strongest_other = legal_log_probs.masked_fill(~other_slots, negative_inf).max(dim=1).values
    per_row[valid_rows] = F.relu(
        float(_POLICY_CORRECTION_RANK_MARGIN)
        + strongest_other[valid_rows]
        - target_scores[valid_rows]
    )
    q_strength = torch.clamp(
        search_q_deltas[valid_rows].to(dtype=legal_log_probs.dtype)
        / float(_POLICY_CORRECTION_Q_DELTA_FULL_STRENGTH),
        min=0.0,
        max=1.0,
    )
    effective_weights = torch.clamp(
        policy_weights[valid_rows].to(dtype=legal_log_probs.dtype), min=0.0
    ) * q_strength
    # Include every policy row in the denominator. A small correction cohort can
    # no longer receive the full strength of an independently averaged loss.
    denominator = torch.clamp(policy_weights.sum(), min=1e-8)
    # Missing Q on unrelated replay rows is normal. Never multiply their NaN
    # by zero: that poisons both the objective and rank-loss gradients.
    loss = (per_row[valid_rows] * effective_weights).sum() / denominator + zero
    return loss, per_row, valid_rows


def _reliable_sparse_policy_target_mask(policy_values, policy_mask):
    """Return rows whose normalized soft target has a decisive winner."""
    batch_size = int(policy_values.size(0))
    if batch_size <= 0 or policy_values.dim() != 2 or policy_values.size(1) <= 0:
        return torch.zeros(
            batch_size,
            device=policy_values.device,
            dtype=torch.bool,
        )
    values = torch.where(
        policy_mask,
        torch.clamp(policy_values.float(), min=0.0),
        torch.zeros_like(policy_values, dtype=torch.float32),
    )
    mass = values.sum(dim=1, keepdim=True)
    normalized = values / mass.clamp_min(1e-8)
    top2 = torch.topk(normalized, k=min(2, normalized.size(1)), dim=1).values
    top1 = top2[:, 0]
    second = top2[:, 1] if top2.size(1) > 1 else torch.zeros_like(top1)
    return (
        (mass.reshape(-1) > 0.0)
        & (top1 >= float(RELIABLE_POLICY_TARGET_TOP1_MIN))
        & ((top1 - second) >= float(RELIABLE_POLICY_TARGET_GAP_MIN))
    )


def _search_q_auxiliary_loss(
    search_q_pred,
    root_q_targets,
    search_visits,
    value_sample_weights,
    *,
    reference_visits=320,
):
    """Distil the searched *state* value into the auxiliary value head.

    ``best_q`` is an action-value selected with an optimistic max and is not a
    valid leaf-state evaluator.  The auxiliary head is consumed as a scalar
    value for the whole position, so its matching target is the MCTS root-Q.
    Replay value weights remain part of this objective; previously they were
    accidentally ignored by the search-Q loss.
    """
    predictions = search_q_pred.reshape(-1)
    targets = root_q_targets.reshape(-1).to(dtype=predictions.dtype)
    visits = search_visits.reshape(-1).to(dtype=predictions.dtype)
    sample_weights = value_sample_weights.reshape(-1).to(dtype=predictions.dtype)
    valid_mask = (
        torch.isfinite(targets)
        & torch.isfinite(visits)
        & (visits > 0.0)
        & torch.isfinite(sample_weights)
        & (sample_weights > 0.0)
    )
    bounded_targets = torch.where(
        valid_mask.any(),
        torch.clamp(targets, -1.0, 1.0),
        torch.zeros_like(predictions),
    )
    reference_visits = max(1.0, float(reference_visits))
    visit_confidence = torch.clamp(
        torch.log1p(torch.clamp(visits, min=0.0)) / math.log1p(reference_visits),
        min=0.25,
        max=1.0,
    )
    effective_weights = torch.clamp(sample_weights, min=0.0) * visit_confidence
    loss_rows = F.smooth_l1_loss(
        predictions[valid_mask],
        bounded_targets[valid_mask],
        beta=0.15,
        reduction="none",
    )
    loss = _weighted_mean(loss_rows, effective_weights[valid_mask])
    return loss, valid_mask, bounded_targets


def _value_search_consistency_loss(
    value_scalar,
    root_q_targets,
    search_visits,
    value_sample_weights,
    *,
    reference_visits=320,
):
    """Give the production WDL value a small, confidence-weighted search signal.

    Final W/D/L remains the primary target.  A scalar root-Q cannot reconstruct
    a W/D/L distribution, so this auxiliary constrains only ``P(win)-P(loss)``
    and never fabricates draw labels from search.
    """
    predictions = value_scalar.reshape(-1)
    targets = root_q_targets.reshape(-1).to(dtype=predictions.dtype)
    visits = search_visits.reshape(-1).to(dtype=predictions.dtype)
    sample_weights = value_sample_weights.reshape(-1).to(dtype=predictions.dtype)
    valid_mask = (
        torch.isfinite(predictions)
        & torch.isfinite(targets)
        & torch.isfinite(visits)
        & (visits > 0.0)
        & torch.isfinite(sample_weights)
        & (sample_weights > 0.0)
    )
    bounded_targets = torch.where(
        valid_mask.any(),
        torch.clamp(targets, -1.0, 1.0),
        torch.zeros_like(predictions),
    )
    reference_visits = max(1.0, float(reference_visits))
    visit_confidence = torch.clamp(
        torch.log1p(torch.clamp(visits, min=0.0)) / math.log1p(reference_visits),
        min=0.25,
        max=1.0,
    )
    effective_weights = torch.clamp(sample_weights, min=0.0) * visit_confidence
    loss_rows = F.smooth_l1_loss(
        predictions[valid_mask],
        bounded_targets[valid_mask],
        beta=0.15,
        reduction="none",
    )
    loss = _weighted_mean(loss_rows, effective_weights[valid_mask])
    return loss, valid_mask, bounded_targets


def _moves_left_auxiliary_loss(predictions, targets, sample_weights):
    """Train MLH only on replay rows that actually contain the target."""
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1).to(dtype=predictions.dtype)
    sample_weights = sample_weights.reshape(-1).to(dtype=predictions.dtype)
    valid_mask = (
        torch.isfinite(predictions)
        & torch.isfinite(targets)
        & (targets >= 0.0)
        & torch.isfinite(sample_weights)
        & (sample_weights > 0.0)
    )
    target_log_plies = torch.log1p(targets[valid_mask])
    loss_rows = F.smooth_l1_loss(
        predictions[valid_mask],
        target_log_plies,
        beta=0.25,
        reduction="none",
    )
    return _weighted_mean(loss_rows, sample_weights[valid_mask]), valid_mask


def train_on_batch_rl(
    model,
    optimizer,
    batch,
    config,
    device,
    scaler,
    metrics_calc=None,
    value_weight_override=None,
    policy_weight_override=None,
    collect_gradient_diagnostics=False,
):
    replay_batch = ReplayBatch.from_tuple(batch)
    (
        boards,
        policy_indices,
        policy_values,
        policy_mask,
        value_targets,
        policy_sample_weights,
        value_sample_weights,
        moves_left_targets,
        legal_indices,
        legal_mask,
        root_q_targets,
        search_changed_top,
        search_q_deltas,
        best_q_targets,
        played_q_targets,
        orig_q_targets,
        policy_kld_targets,
        search_visits,
    ) = (
        replay_batch.boards,
        replay_batch.policy_indices,
        replay_batch.policy_values,
        replay_batch.policy_mask,
        replay_batch.value_targets,
        replay_batch.policy_sample_weights,
        replay_batch.value_sample_weights,
        replay_batch.moves_left_targets,
        replay_batch.legal_indices,
        replay_batch.legal_mask,
        replay_batch.root_q_targets,
        replay_batch.search_changed_top,
        replay_batch.search_q_deltas,
        replay_batch.best_q_targets,
        replay_batch.played_q_targets,
        replay_batch.orig_q_targets,
        replay_batch.policy_kld_targets,
        replay_batch.search_visits,
    )
    if config["reinforcement_learning"].get("replay_fp16", False):
        boards = boards.float()
        policy_values = policy_values.float()
        value_targets = value_targets.float()
        moves_left_targets = moves_left_targets.float()

    boards, policy_indices, policy_values, policy_mask, legal_indices, legal_mask = _maybe_augment_batch(
        boards,
        policy_indices,
        policy_values,
        policy_mask,
        legal_indices,
        legal_mask,
        config,
        input_encoder_id=(
            getattr(model, "input_encoder_id", None)
            or (getattr(model, "model_spec", {}) or {}).get("input_spec", {}).get("encoder_id")
            or config.get("model", {}).get("input_encoder_id")
        ),
        history_positions=(
            (getattr(model, "model_spec", {}) or {}).get("input_spec", {}).get("history_positions")
            if (getattr(model, "model_spec", {}) or {}).get("input_spec")
            else config.get("model", {}).get("history_positions", 0)
        ),
    )
    boards = boards.to(device, memory_format=torch.channels_last, non_blocking=True)
    policy_indices = policy_indices.to(device, non_blocking=True)
    policy_values = policy_values.to(device, non_blocking=True)
    policy_mask = policy_mask.to(device, non_blocking=True)
    legal_indices = legal_indices.to(device, non_blocking=True)
    legal_mask = legal_mask.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    policy_sample_weights = policy_sample_weights.to(device, non_blocking=True)
    value_sample_weights = value_sample_weights.to(device, non_blocking=True)
    moves_left_targets = moves_left_targets.to(device, non_blocking=True)
    if root_q_targets is None:
        root_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    root_q_targets = root_q_targets.to(device, non_blocking=True).reshape(-1)
    if search_changed_top is None:
        search_changed_top = torch.zeros_like(policy_sample_weights, dtype=torch.bool)
    if search_q_deltas is None:
        search_q_deltas = torch.full_like(policy_sample_weights, float("nan"), dtype=torch.float32)
    search_changed_top = search_changed_top.to(device, non_blocking=True).reshape(-1).bool()
    search_q_deltas = search_q_deltas.to(device, non_blocking=True).reshape(-1).float()
    if best_q_targets is None:
        best_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if played_q_targets is None:
        played_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if orig_q_targets is None:
        orig_q_targets = torch.full_like(value_targets, float("nan"), dtype=torch.float32)
    if policy_kld_targets is None:
        policy_kld_targets = torch.full_like(policy_sample_weights, float("nan"), dtype=torch.float32)
    if search_visits is None:
        search_visits = torch.zeros_like(policy_sample_weights, dtype=torch.int32)
    best_q_targets = best_q_targets.to(device, non_blocking=True).reshape(-1).float()
    played_q_targets = played_q_targets.to(device, non_blocking=True).reshape(-1).float()
    orig_q_targets = orig_q_targets.to(device, non_blocking=True).reshape(-1).float()
    policy_kld_targets = policy_kld_targets.to(device, non_blocking=True).reshape(-1).float()
    search_visits = search_visits.to(device, non_blocking=True).reshape(-1).float()
    effective_policy_mask = policy_mask

    optimizer.zero_grad(set_to_none=True)

    rl_cfg = config.get("reinforcement_learning", {})
    v10_training_aux = _model_supports_training_aux(model)
    model_cfg = dict(
        getattr(_training_model_owner(model), "model_config", None)
        or config.get("model", {})
        or {}
    )
    bn_head_loss_weight = min(
        1.0,
        max(0.0, float(model_cfg.get("bn_head_loss_weight", 0.8))),
    )
    main_head_loss_weight = 1.0 - bn_head_loss_weight
    soft_policy_temperature = float(model_cfg.get("soft_policy_temperature", 4.0))
    soft_policy_loss_weight = max(
        0.0,
        float(model_cfg.get("soft_policy_loss_weight", 0.25)),
    )
    use_amp = config["hardware"].get("use_amp", True)
    amp_dtype = torch.bfloat16 if config["hardware"].get("use_bfloat16", False) else torch.float16

    moves_left_loss_weight = max(0.0, float(rl_cfg.get("moves_left_loss_weight", 0.05)))
    search_q_loss_weight = max(0.0, float(rl_cfg.get("search_q_loss_weight", 0.0)))
    value_scalar_aux_loss_weight = max(
        0.0,
        float(rl_cfg.get("value_scalar_aux_loss_weight", 0.0)),
    )
    value_search_consistency_loss_weight = max(
        0.0,
        float(rl_cfg.get("value_search_consistency_loss_weight", 0.0)),
    )
    freeze_batchnorm_stats = bool(rl_cfg.get("freeze_batchnorm_stats", False))
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        with _freeze_batchnorm_running_stats(model, freeze_batchnorm_stats):
            if v10_training_aux:
                training_outputs = model(
                    boards,
                    apply_log_softmax=False,
                    return_moves_left=True,
                    return_search_aux=True,
                    return_training_aux=True,
                )
                if not isinstance(training_outputs, dict):
                    raise RuntimeError(
                        "A model advertising supports_training_aux must return a mapping."
                    )
                try:
                    (
                        policy_logits,
                        value_pred,
                        moves_left_pred,
                        search_q_pred,
                    ) = training_outputs["main"]
                    (
                        bn_policy_logits,
                        bn_value_pred,
                        bn_moves_left_pred,
                        bn_search_q_pred,
                    ) = training_outputs["bn"]
                    soft_policy_logits = training_outputs["soft_policy"]
                    bn_soft_policy_logits = training_outputs["bn_soft_policy"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "v10 training auxiliaries must contain main, bn, soft_policy, "
                        "and bn_soft_policy outputs."
                    ) from exc
            else:
                policy_logits, value_pred, moves_left_pred, search_q_pred = model(
                    boards,
                    apply_log_softmax=False,
                    return_moves_left=True,
                    return_search_aux=True,
                )
        policy_pred = F.log_softmax(policy_logits.float(), dim=1)
        legal_policy_log_probs, valid_legal_mask, safe_legal_indices = _legal_only_log_probs(
            policy_logits,
            legal_indices,
            legal_mask,
        )
        value_pred_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)
        target_value_std = torch.tensor(0.0, device=policy_pred.device, dtype=policy_pred.dtype)

        policy_rows_mask = effective_policy_mask.any(dim=1)
        reliable_policy_target_mask = _reliable_sparse_policy_target_mask(
            policy_values,
            effective_policy_mask,
        )
        correction_policy_mask = (
            policy_rows_mask
            & torch.isfinite(policy_sample_weights.reshape(-1))
            & (policy_sample_weights.reshape(-1) > 0.0)
            & search_changed_top
            & torch.isfinite(search_q_deltas)
            & (search_q_deltas > USEFUL_SEARCH_Q_DELTA_MIN)
            & reliable_policy_target_mask
        )
        # Keep the exact stored Gumbel improved-policy distribution. Direct CE
        # is the complete policy objective; do not add a second winner-only loss.
        policy_training_values = policy_values
        if policy_indices.numel() == 0:
            target_moves = torch.zeros(
                policy_pred.size(0),
                dtype=torch.long,
                device=policy_pred.device,
            )
        else:
            best_sparse_idx = policy_training_values.argmax(dim=1, keepdim=True)
            target_moves = torch.gather(
                policy_indices.long(),
                1,
                best_sparse_idx,
            ).squeeze(1)
        finite_policy_weights = torch.where(
            torch.isfinite(policy_sample_weights.reshape(-1)),
            torch.clamp(policy_sample_weights.reshape(-1), min=0.0),
            torch.zeros_like(policy_sample_weights.reshape(-1)),
        ).to(dtype=policy_pred.dtype)
        policy_effective_weights = (
            policy_rows_mask.to(dtype=policy_pred.dtype) * finite_policy_weights
        )
        if policy_indices.numel() == 0:
            policy_loss_per_row = torch.zeros(
                policy_pred.size(0),
                device=policy_pred.device,
                dtype=policy_pred.dtype,
            )
        else:
            policy_loss_per_row = _legal_only_sparse_policy_loss(
                policy_logits,
                policy_indices,
                policy_training_values,
                effective_policy_mask,
                legal_indices,
                legal_mask,
            )
        (
            policy_correction_rank_loss,
            policy_correction_rank_loss_per_row,
            policy_correction_rank_mask,
        ) = _useful_policy_correction_rank_loss(
            legal_policy_log_probs,
            valid_legal_mask,
            safe_legal_indices,
            target_moves,
            correction_policy_mask,
            search_q_deltas,
            policy_effective_weights,
        )
        target_scalar = _final_outcome_targets(value_targets)
        target_value_std = target_scalar.std(unbiased=False)
        if value_pred.dim() == 2 and value_pred.size(1) == 3:
            hard_target_wdl = _wdl_targets_from_final_outcome(target_scalar)
            target_wdl = hard_target_wdl

            value_log_probs = F.log_softmax(value_pred, dim=1)
            value_ce_loss = -(target_wdl * value_log_probs).sum(dim=1)
            value_probs = torch.softmax(value_pred, dim=1)
            value_scalar = value_probs[:, 0] - value_probs[:, 2]
            value_primary_loss_rows = value_ce_loss
            value_pred_std = value_scalar.std(unbiased=False).detach()
        else:
            hard_target_wdl = None
            value_probs = None
            value_primary_loss_rows = (value_pred.squeeze() - target_scalar) ** 2
            value_scalar = value_pred.squeeze()
            value_pred_std = value_scalar.std(unbiased=False).detach()

        raw_value_sample_weights = value_sample_weights.reshape(-1).float()
        effective_value_sample_weights = torch.where(
            torch.isfinite(raw_value_sample_weights),
            torch.clamp(raw_value_sample_weights, min=0.0),
            torch.zeros_like(raw_value_sample_weights),
        )
        if policy_loss_per_row.numel() == 0:
            policy_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        else:
            policy_weight_total = policy_effective_weights.sum()
            if float(policy_weight_total.detach().item()) > 0.0:
                policy_loss = (
                    policy_loss_per_row * policy_effective_weights
                ).sum() / policy_weight_total
            else:
                policy_loss = policy_loss_per_row.sum() * 0.0
        value_primary_loss = _weighted_mean(
            value_primary_loss_rows,
            effective_value_sample_weights,
        )
        value_scalar_aux_loss = torch.zeros(
            (), device=value_primary_loss.device, dtype=value_primary_loss.dtype
        )
        if hard_target_wdl is not None and value_scalar_aux_loss_weight > 0.0:
            # WDL cross-entropy learns calibrated class probabilities, while
            # this small scalar term directly trains the W-L quantity consumed
            # by production MCTS.  IL already uses the same contract; RL had
            # silently dropped it and therefore optimized only the class CE.
            value_scalar_aux_loss = _weighted_mean(
                F.smooth_l1_loss(
                    value_scalar,
                    target_scalar.to(dtype=value_scalar.dtype),
                    beta=0.25,
                    reduction="none",
                ),
                effective_value_sample_weights,
            )
        value_loss = (
            value_primary_loss
            + value_scalar_aux_loss_weight * value_scalar_aux_loss
        )
        search_reference_visits = max(
            192,
            int(round(
                float(rl_cfg.get("mcts_simulations", 192) or 192)
                * float(rl_cfg.get("mcts_dynamic_budget_max_multiplier", 1.0) or 1.0)
            )),
        )
        value_search_consistency_loss, _, _ = _value_search_consistency_loss(
            value_scalar,
            root_q_targets,
            search_visits,
            effective_value_sample_weights,
            reference_visits=search_reference_visits,
        )
        # Keep a separate Q head for diagnostics/distillation. Production MCTS
        # still consumes only WDL; the small consistency term above is the only
        # route by which searched state value regularizes that production head.
        root_q_mask = torch.isfinite(root_q_targets) & (search_visits > 0.0)
        search_q_loss, search_q_mask, bounded_root_q = _search_q_auxiliary_loss(
            search_q_pred,
            root_q_targets,
            search_visits,
            effective_value_sample_weights,
            reference_visits=search_reference_visits,
        )
        moves_left_loss = torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        if moves_left_loss_weight > 0.0 and moves_left_pred is not None:
            moves_left_loss, _ = _moves_left_auxiliary_loss(
                moves_left_pred,
                moves_left_targets,
                effective_value_sample_weights,
            )
        policy_weight = (
            float(config["reinforcement_learning"]["policy_loss_weight"])
            if policy_weight_override is None
            else float(policy_weight_override)
        )
        value_weight = (
            float(config["reinforcement_learning"]["value_loss_weight"])
            if value_weight_override is None
            else float(value_weight_override)
        )
        bn_policy_loss = torch.zeros_like(policy_loss)
        bn_policy_correction_rank_loss = torch.zeros_like(policy_correction_rank_loss)
        bn_value_primary_loss = torch.zeros_like(value_primary_loss)
        bn_value_scalar_aux_loss = torch.zeros_like(value_scalar_aux_loss)
        bn_value_loss = torch.zeros_like(value_loss)
        bn_value_search_consistency_loss = torch.zeros_like(value_search_consistency_loss)
        bn_moves_left_loss = torch.zeros_like(moves_left_loss)
        bn_search_q_loss = torch.zeros_like(search_q_loss)
        soft_policy_main_loss = torch.zeros_like(policy_loss)
        soft_policy_bn_loss = torch.zeros_like(policy_loss)
        soft_policy_loss = torch.zeros_like(policy_loss)
        soft_policy_valid_rows = torch.zeros_like(policy_rows_mask)
        if v10_training_aux:
            bn_policy_log_probs, bn_valid_legal_mask, bn_safe_legal_indices = (
                _legal_only_log_probs(bn_policy_logits, legal_indices, legal_mask)
            )
            if policy_indices.numel() == 0:
                bn_policy_loss_per_row = torch.zeros(
                    policy_pred.size(0),
                    device=policy_pred.device,
                    dtype=policy_pred.dtype,
                )
            else:
                bn_policy_loss_per_row = _legal_only_sparse_policy_loss(
                    bn_policy_logits,
                    policy_indices,
                    policy_training_values,
                    effective_policy_mask,
                    legal_indices,
                    legal_mask,
                )
            bn_policy_loss = _weighted_mean(
                bn_policy_loss_per_row,
                policy_effective_weights,
            )
            (
                bn_policy_correction_rank_loss,
                _,
                _,
            ) = _useful_policy_correction_rank_loss(
                bn_policy_log_probs,
                bn_valid_legal_mask,
                bn_safe_legal_indices,
                target_moves,
                correction_policy_mask,
                search_q_deltas,
                policy_effective_weights,
            )

            if bn_value_pred.dim() == 2 and bn_value_pred.size(1) == 3:
                bn_value_probs = torch.softmax(bn_value_pred, dim=1)
                bn_value_scalar = bn_value_probs[:, 0] - bn_value_probs[:, 2]
                bn_value_primary_rows = -(
                    hard_target_wdl * F.log_softmax(bn_value_pred, dim=1)
                ).sum(dim=1)
            else:
                bn_value_probs = None
                bn_value_scalar = bn_value_pred.squeeze()
                bn_value_primary_rows = (bn_value_scalar - target_scalar) ** 2
            bn_value_primary_loss = _weighted_mean(
                bn_value_primary_rows,
                effective_value_sample_weights,
            )
            if hard_target_wdl is not None and value_scalar_aux_loss_weight > 0.0:
                bn_value_scalar_aux_loss = _weighted_mean(
                    F.smooth_l1_loss(
                        bn_value_scalar,
                        target_scalar.to(dtype=bn_value_scalar.dtype),
                        beta=0.25,
                        reduction="none",
                    ),
                    effective_value_sample_weights,
                )
            bn_value_loss = (
                bn_value_primary_loss
                + value_scalar_aux_loss_weight * bn_value_scalar_aux_loss
            )
            bn_value_search_consistency_loss, _, _ = _value_search_consistency_loss(
                bn_value_scalar,
                root_q_targets,
                search_visits,
                effective_value_sample_weights,
                reference_visits=search_reference_visits,
            )
            if bn_moves_left_pred is not None and moves_left_loss_weight > 0.0:
                bn_moves_left_loss, _ = _moves_left_auxiliary_loss(
                    bn_moves_left_pred,
                    moves_left_targets,
                    effective_value_sample_weights,
                )
            if bn_search_q_pred is not None and search_q_loss_weight > 0.0:
                bn_search_q_loss, _, _ = _search_q_auxiliary_loss(
                    bn_search_q_pred,
                    root_q_targets,
                    search_visits,
                    effective_value_sample_weights,
                    reference_visits=search_reference_visits,
                )

            soft_row_mask = policy_rows_mask & (policy_effective_weights > 0.0)
            # Replay stores legality as padded (index,slot-mask) pairs, whereas
            # the shared helper accepts a list of usable indices. Drop masked
            # slots before passing support so an inactive non-negative index
            # cannot enter the softmax denominator.
            soft_legal_indices = torch.where(
                legal_mask.bool(),
                legal_indices,
                torch.full_like(legal_indices, -1),
            )
            prepared_soft_target = prepare_legal_masked_soft_policy_target(
                soft_policy_logits,
                (policy_indices, policy_values),
                row_weights=policy_effective_weights,
                temperature=soft_policy_temperature,
                row_mask=soft_row_mask,
                legal_indices=soft_legal_indices,
                min_support=2,
            )
            soft_policy_main_loss, _, soft_policy_valid_rows = (
                legal_masked_soft_policy_loss_from_prepared(
                    soft_policy_logits, prepared_soft_target
                )
            )
            soft_policy_bn_loss, _, bn_soft_policy_valid_rows = (
                legal_masked_soft_policy_loss_from_prepared(
                    bn_soft_policy_logits, prepared_soft_target
                )
            )
            soft_policy_loss = (
                main_head_loss_weight * soft_policy_main_loss
                + bn_head_loss_weight * soft_policy_bn_loss
            )
            soft_policy_valid_rows = soft_policy_valid_rows & bn_soft_policy_valid_rows

        legal_policy_probs = torch.exp(legal_policy_log_probs) * valid_legal_mask
        legal_entropy_per_row = -(
            legal_policy_probs * legal_policy_log_probs
        ).sum(dim=1)
        rows_with_legal_moves = valid_legal_mask.any(dim=1)
        policy_entropy = (
            legal_entropy_per_row[rows_with_legal_moves].mean()
            if rows_with_legal_moves.any()
            else torch.zeros((), device=policy_pred.device, dtype=policy_pred.dtype)
        )
        main_policy_objective = policy_weight * (
            policy_loss
            + float(_POLICY_CORRECTION_RANK_WEIGHT)
            * policy_correction_rank_loss
        )
        main_value_objective = (
            value_weight * value_loss
            + value_search_consistency_loss_weight * value_search_consistency_loss
            + moves_left_loss_weight * moves_left_loss
            + search_q_loss_weight * search_q_loss
        )
        if v10_training_aux:
            bn_policy_objective = policy_weight * (
                bn_policy_loss
                + float(_POLICY_CORRECTION_RANK_WEIGHT)
                * bn_policy_correction_rank_loss
            )
            bn_value_objective = (
                value_weight * bn_value_loss
                + value_search_consistency_loss_weight
                * bn_value_search_consistency_loss
                + moves_left_loss_weight * bn_moves_left_loss
                + search_q_loss_weight * bn_search_q_loss
            )
            policy_objective = (
                main_head_loss_weight * main_policy_objective
                + bn_head_loss_weight * bn_policy_objective
            )
            # Keep the task-gradient probe aligned with the policy gradient
            # actually used by the optimizer. The soft-policy auxiliary is
            # policy-only and contributes to `loss` below; omitting it here
            # understated the policy norm and distorted the policy/value
            # cosine in RL diagnostics.
            policy_diagnostic_objective = (
                policy_objective + soft_policy_loss_weight * soft_policy_loss
            )
            value_objective = (
                main_head_loss_weight * main_value_objective
                + bn_head_loss_weight * bn_value_objective
            )
            loss = (
                policy_objective
                + value_objective
                + soft_policy_loss_weight * soft_policy_loss
            )
        else:
            policy_objective = main_policy_objective
            policy_diagnostic_objective = policy_objective
            value_objective = main_value_objective
            loss = policy_objective + value_objective

    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError(
            'Non-finite RL loss before backward; check policy/value targets and weights.'
        )

    if collect_gradient_diagnostics:
        task_gradient_diagnostics = _task_gradient_probe(
            policy_diagnostic_objective,
            value_objective,
            model,
        )
    else:
        task_gradient_diagnostics = {
            'policy_probe_norm': 0.0,
            'value_probe_norm': 0.0,
            'policy_value_cosine': 0.0,
        }
    scaler.scale(loss).backward()

    grad_clip = config["reinforcement_learning"].get("grad_clip", 1.0)
    scaler.unscale_(optimizer)
    gradient_family_norms = (
        _gradient_family_norms(model)
        if collect_gradient_diagnostics
        else {'backbone': 0.0, 'policy': 0.0, 'value': 0.0}
    )
    clip_limit = float(grad_clip) if float(grad_clip) > 0.0 else float('inf')
    total_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
    total_grad_norm = float(total_grad_norm_tensor.detach().item())
    grad_clip_scale = (
        min(1.0, clip_limit / max(total_grad_norm, 1e-12))
        if math.isfinite(clip_limit)
        else 1.0
    )
    grad_was_clipped = bool(grad_clip_scale < 1.0)

    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        legal_top_slots = legal_policy_log_probs.masked_fill(
            ~valid_legal_mask,
            -1.0e9,
        ).argmax(dim=1, keepdim=True)
        predicted_legal_moves = torch.gather(
            safe_legal_indices.long(),
            1,
            legal_top_slots,
        ).squeeze(1)
        correction_count = int(correction_policy_mask.sum().item())
        policy_row_count = int(policy_rows_mask.sum().item())
        correction_loss_sum = float(
            policy_loss_per_row[correction_policy_mask].sum().detach().item()
        ) if correction_count > 0 else 0.0
        correction_rank_count = int(policy_correction_rank_mask.sum().item())
        correction_rank_loss_sum = float(
            policy_correction_rank_loss_per_row[policy_correction_rank_mask]
            .sum()
            .detach()
            .item()
        ) if correction_rank_count > 0 else 0.0
        # These logits come from the forward that produced this optimizer step.
        # Keep the cheap batch diagnostic explicitly pre-step; the fixed audit
        # is the authoritative before/after absorption measurement.
        correction_top1_correct = int(
            (predicted_legal_moves[correction_policy_mask] == target_moves[correction_policy_mask]).sum().item()
        ) if correction_count > 0 else 0
        policy_effective_weight_sum = float(policy_effective_weights.sum().detach().item())
        correction_effective_weight_sum = float(
            policy_effective_weights[correction_policy_mask].sum().detach().item()
        ) if correction_count > 0 else 0.0
        value_scalar_f32 = value_scalar.detach().float().reshape(-1)
        search_q_mask_f32 = search_q_mask.detach()
        search_q_pred_f32 = search_q_pred.detach().float().reshape(-1)[search_q_mask_f32]
        search_q_target_f32 = bounded_root_q.detach().float().reshape(-1)[search_q_mask_f32]
        target_scalar_f32 = target_scalar.detach().float().reshape(-1)
        value_row_count = int(value_scalar_f32.numel())
        wdl_pred_sums = [0.0, 0.0, 0.0]
        wdl_target_sums = [0.0, 0.0, 0.0]
        wdl_brier_sum = 0.0
        wdl_ece_counts = [0] * 10
        wdl_ece_confidence_sums = [0.0] * 10
        wdl_ece_correct_sums = [0.0] * 10
        value_phase_wdl = {}
        if value_probs is not None and hard_target_wdl is not None:
            value_probs_f32 = value_probs.detach().float()
            hard_target_wdl_f32 = hard_target_wdl.detach().float()
            wdl_pred_sums = value_probs_f32.sum(dim=0).cpu().tolist()
            wdl_target_sums = hard_target_wdl_f32.sum(dim=0).cpu().tolist()
            wdl_brier_sum = float(
                torch.square(value_probs_f32 - hard_target_wdl_f32).sum(dim=1).sum().item()
            )
            confidence, predicted_class = value_probs_f32.max(dim=1)
            target_class = hard_target_wdl_f32.argmax(dim=1)
            correct = predicted_class.eq(target_class).float()
            bin_indices = torch.clamp((confidence * 10.0).long(), min=0, max=9)
            for bin_idx in range(10):
                bin_mask = bin_indices == bin_idx
                if bin_mask.any():
                    wdl_ece_counts[bin_idx] = int(bin_mask.sum().item())
                    wdl_ece_confidence_sums[bin_idx] = float(confidence[bin_mask].sum().item())
                    wdl_ece_correct_sums[bin_idx] = float(correct[bin_mask].sum().item())

            if boards.dim() >= 4 and boards.size(1) > 15:
                fullmoves = torch.clamp(
                    torch.round(boards[:, 15, 0, 0].float() * 100.0),
                    min=1.0,
                    max=float(rl_cfg.get("value_phase_max_fullmove", 120)),
                )
                opening_max = int(rl_cfg.get("value_phase_opening_max_fullmove", 12))
                endgame_min = int(rl_cfg.get("value_phase_endgame_min_fullmove", 40))
                phase_masks = {
                    "opening": fullmoves <= opening_max,
                    "middlegame": (fullmoves > opening_max) & (fullmoves < endgame_min),
                    "endgame": fullmoves >= endgame_min,
                }
                for phase_name, phase_mask in phase_masks.items():
                    phase_count = int(phase_mask.sum().item())
                    value_phase_wdl[phase_name] = {
                        "rows": phase_count,
                        "draw_pred_sum": float(value_probs_f32[phase_mask, 1].sum().item()),
                        "draw_target_sum": float(hard_target_wdl_f32[phase_mask, 1].sum().item()),
                    }
        root_q_row_count = int(root_q_mask.sum().item())
        if root_q_row_count > 0:
            root_q_pred_f32 = value_scalar_f32[root_q_mask]
            root_q_target_f32 = torch.clamp(
                root_q_targets.detach().float().reshape(-1)[root_q_mask],
                -1.0,
                1.0,
            )
            root_q_error_f32 = root_q_pred_f32 - root_q_target_f32
        else:
            root_q_pred_f32 = value_scalar_f32[:0]
            root_q_target_f32 = target_scalar_f32[:0]
            root_q_error_f32 = value_scalar_f32[:0]

    if metrics_calc is not None:
        with torch.no_grad():
            rl_cfg = config.get("reinforcement_learning", {})
            if boards.dim() >= 4 and boards.size(1) > 15:
                fullmove_indices = torch.clamp(
                    torch.round(boards[:, 15, 0, 0].float() * 100.0),
                    min=1.0,
                    max=float(rl_cfg.get("value_phase_max_fullmove", 120)),
                )
            else:
                fullmove_indices = None
            metrics_calc.update(
                policy_pred,
                value_pred,
                target_moves,
                target_scalar.unsqueeze(1),
                move_indices=fullmove_indices,
                value_weight_min=float(rl_cfg.get("value_metric_weight_min", 0.20)),
                value_max_moves=int(rl_cfg.get("value_metric_max_fullmove", 80)),
                value_phase_opening_max=int(rl_cfg.get("value_phase_opening_max_fullmove", 12)),
                value_phase_endgame_min=int(rl_cfg.get("value_phase_endgame_min_fullmove", 40)),
                legal_indices=legal_indices,
                target_policy_indices=policy_indices,
                target_policy_values=policy_training_values,
                target_wdl=hard_target_wdl,
            )
            metrics_calc.update_value_phase_loss(
                value_pred,
                target_value=target_scalar.unsqueeze(1),
                target_wdl=hard_target_wdl,
                move_indices=fullmove_indices,
                sample_weight=effective_value_sample_weights,
                value_phase_opening_max=int(
                    rl_cfg.get("value_phase_opening_max_fullmove", 12)
                ),
                value_phase_endgame_min=int(
                    rl_cfg.get("value_phase_endgame_min_fullmove", 40)
                ),
            )

    policy_diagnostics = {
        "correction_rows": correction_count,
        "policy_rows": policy_row_count,
        "correction_loss_sum": correction_loss_sum,
        "correction_rank_rows": correction_rank_count,
        "correction_rank_loss_sum": correction_rank_loss_sum,
        "correction_rank_objective": float(policy_correction_rank_loss.detach().item()),
        "correction_rank_weight": float(_POLICY_CORRECTION_RANK_WEIGHT),
        "bn_head_loss_weight": bn_head_loss_weight if v10_training_aux else 0.0,
        "bn_policy_loss": float(bn_policy_loss.detach().item()),
        "bn_policy_correction_rank_loss": float(
            bn_policy_correction_rank_loss.detach().item()
        ),
        "bn_value_primary_loss": float(bn_value_primary_loss.detach().item()),
        "bn_value_loss": float(bn_value_loss.detach().item()),
        "bn_value_scalar_aux_loss": float(bn_value_scalar_aux_loss.detach().item()),
        "bn_value_search_consistency_loss": float(
            bn_value_search_consistency_loss.detach().item()
        ),
        "bn_moves_left_loss": float(bn_moves_left_loss.detach().item()),
        "bn_search_q_loss": float(bn_search_q_loss.detach().item()),
        "soft_policy_loss": float(soft_policy_loss.detach().item()),
        "soft_policy_main_loss": float(soft_policy_main_loss.detach().item()),
        "soft_policy_bn_loss": float(soft_policy_bn_loss.detach().item()),
        "soft_policy_valid_rows": int(soft_policy_valid_rows.sum().item()),
        "soft_policy_loss_weight": soft_policy_loss_weight if v10_training_aux else 0.0,
        "soft_policy_temperature": soft_policy_temperature if v10_training_aux else 0.0,
        "correction_top1_correct": correction_top1_correct,
        "effective_weight_sum": policy_effective_weight_sum,
        "correction_effective_weight_sum": correction_effective_weight_sum,
        "value_primary_loss": float(value_primary_loss.detach().item()),
        "value_scalar_aux_loss": float(value_scalar_aux_loss.detach().item()),
        "value_scalar_aux_weight": float(value_scalar_aux_loss_weight),
        "value_search_consistency_loss": float(
            value_search_consistency_loss.detach().item()
        ),
        "moves_left_loss": float(moves_left_loss.detach().item()),
        "search_q_loss": float(search_q_loss.detach().item()),
        "search_q_weight": float(search_q_loss_weight),
        "search_q_rows": int(search_q_mask.sum().item()),
        "search_q_pred_sum": float(search_q_pred_f32.sum().item()),
        "search_q_target_sum": float(search_q_target_f32.sum().item()),
        "search_q_abs_error_sum": float(
            (search_q_pred_f32 - search_q_target_f32).abs().sum().item()
        ),
        "wdl_pred_sums": wdl_pred_sums,
        "wdl_target_sums": wdl_target_sums,
        "wdl_brier_sum": wdl_brier_sum,
        "wdl_ece_counts": wdl_ece_counts,
        "wdl_ece_confidence_sums": wdl_ece_confidence_sums,
        "wdl_ece_correct_sums": wdl_ece_correct_sums,
        "value_phase_wdl": value_phase_wdl,
        # Raw sufficient statistics let the iteration logger aggregate exact
        # means/correlation across uneven final batches without retaining any
        # per-position tensors.
        "value_rows": value_row_count,
        "value_pred_sum": float(value_scalar_f32.sum().item()),
        "value_target_sum": float(target_scalar_f32.sum().item()),
        "root_q_rows": root_q_row_count,
        "root_q_pred_sum": float(root_q_pred_f32.sum().item()),
        "root_q_target_sum": float(root_q_target_f32.sum().item()),
        "root_q_abs_error_sum": float(root_q_error_f32.abs().sum().item()),
        "root_q_error_sum": float(root_q_error_f32.sum().item()),
        "root_q_pred_sq_sum": float((root_q_pred_f32 * root_q_pred_f32).sum().item()),
        "root_q_target_sq_sum": float((root_q_target_f32 * root_q_target_f32).sum().item()),
        "root_q_cross_sum": float((root_q_pred_f32 * root_q_target_f32).sum().item()),
        "total_grad_norm": total_grad_norm,
        "grad_clip_scale": grad_clip_scale,
        "grad_was_clipped": 1 if grad_was_clipped else 0,
        "grad_backbone_norm": gradient_family_norms['backbone'],
        "grad_policy_head_norm": gradient_family_norms['policy'],
        "grad_value_head_norm": gradient_family_norms['value'],
        **task_gradient_diagnostics,
    }
    return (
        loss.item(),
        policy_loss.item(),
        value_loss.item(),
        float(policy_entropy.detach().item()),
        float(value_pred_std.detach().item()),
        float(target_value_std.detach().item()),
        policy_diagnostics,
    )


def evaluate_models(
    model1,
    model2,
    config,
    device,
    num_games=100,
    game_index_offset=0,
    use_fixed_openings=None,
    model1_mcts_config=None,
    model2_mcts_config=None,
    progress_desc="Eval vs best",
    central_runtime=None,
    progress_callback=None,
    use_mcts_model1=True,
    use_mcts_model2=True,
    stop_event=None,
    show_progress=True,
    verbose=False,
    runtime_ready_callback=None,
    runtime_progress_callback=None,
):
    if _eval_stop_requested(stop_event):
        stats = _build_eval_stats(0, 0, 0, 0, num_games)
        stats.update({
            "cancelled": True,
            "cancel_reason": "stop requested before evaluation startup",
            "cancel_stage": "startup",
            "completed": 0,
        })
        _notify_eval_runtime(
            runtime_ready_callback,
            status="cancelled",
            ready=False,
            cancel_stage="startup",
            completed=0,
            requested_games=int(num_games),
        )
        return stats
    config = _build_eval_mcts_config(config)
    model1_mcts_config = _build_eval_mcts_config(model1_mcts_config or config)
    model2_mcts_config = _build_eval_mcts_config(model2_mcts_config or config)
    rl_cfg = config.get("reinforcement_learning", {})
    central_min_games = max(1, int(rl_cfg.get("eval_central_inference_min_games", 2) or 2))
    if _eval_uses_central_inference(config, device) and int(num_games) >= central_min_games:
        return _evaluate_models_with_central_inference(
            model1,
            model2,
            config,
            device,
            int(num_games),
            game_index_offset=game_index_offset,
            use_fixed_openings=use_fixed_openings,
            model1_mcts_config=model1_mcts_config,
            model2_mcts_config=model2_mcts_config,
            progress_desc=progress_desc,
            central_runtime=central_runtime,
            progress_callback=progress_callback,
            use_mcts_model1=use_mcts_model1,
            use_mcts_model2=use_mcts_model2,
            stop_event=stop_event,
            show_progress=show_progress,
            verbose=verbose,
            runtime_ready_callback=runtime_ready_callback,
        )

    workers = _resolve_eval_workers(config, device, num_games)
    if workers <= 1:
        _notify_eval_runtime(
            runtime_ready_callback,
            status="running",
            ready=True,
            workers=1,
            server_count=0,
            central_inference=False,
            load_s=0.0,
        )
        max_moves = _resolve_eval_max_moves(config)
        game_indices = list(range(game_index_offset, game_index_offset + num_games))
        eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
        def _on_progress(completed, *details):
            eval_bar.update(int(completed))
            if progress_callback is not None:
                progress_callback(int(completed), *details)
        try:
            stats = _evaluate_games_batched(
                model1,
                model2,
                config,
                device,
                game_indices,
                use_fixed_openings=use_fixed_openings,
                progress_callback=_on_progress,
                model1_mcts_config=model1_mcts_config,
                model2_mcts_config=model2_mcts_config,
                use_mcts_model1=use_mcts_model1,
                use_mcts_model2=use_mcts_model2,
                stop_event=stop_event,
                runtime_progress_callback=runtime_progress_callback,
            )
        finally:
            eval_bar.close()

        unresolved = int((stats or {}).get("unresolved", 0))
        if unresolved > 0 and verbose:
            _print_eval_unresolved(
                "Eval",
                unresolved,
                num_games,
                max_moves,
                neutral_score=_resolve_eval_unresolved_score_neutral(config),
            )

        _notify_eval_runtime(
            runtime_ready_callback,
            status="cancelled" if bool((stats or {}).get("cancelled", False)) else "completed",
            ready=False,
            completed=int((stats or {}).get("completed", 0) or 0),
            requested_games=int(num_games),
            cancel_stage="game evaluation" if bool((stats or {}).get("cancelled", False)) else None,
        )

        return stats

    if _eval_stop_requested(stop_event):
        stats = _build_eval_stats(0, 0, 0, 0, num_games)
        stats.update({
            "cancelled": True,
            "cancel_reason": "stop requested before model snapshot",
            "cancel_stage": "model snapshot",
            "completed": 0,
        })
        return stats
    _notify_eval_runtime(
        runtime_ready_callback,
        status="snapshotting_models",
        ready=False,
        workers=int(workers),
        central_inference=False,
    )
    model1_state = _snapshot_state_dict_cpu_shared(model1)
    model2_state = _snapshot_state_dict_cpu_shared(model2)
    if _eval_stop_requested(stop_event):
        stats = _build_eval_stats(0, 0, 0, 0, num_games)
        stats.update({
            "cancelled": True,
            "cancel_reason": "stop requested during model snapshot",
            "cancel_stage": "model snapshot",
            "completed": 0,
        })
        return stats
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    game_indices_per_worker = [[] for _ in range(workers)]
    for offset, game_idx in enumerate(range(game_index_offset, game_index_offset + num_games)):
        game_indices_per_worker[offset % workers].append(game_idx)

    try:
        for rank, game_indices in enumerate(game_indices_per_worker):
            if not game_indices:
                continue
            _raise_if_eval_stopped(stop_event, "worker startup")
            p = ctx.Process(
                target=_eval_worker,
                args=(
                    rank,
                    model1_state,
                    model2_state,
                    config,
                    str(device),
                    game_indices,
                    result_queue,
                    use_fixed_openings,
                    model1_mcts_config,
                    model2_mcts_config,
                    use_mcts_model1,
                    use_mcts_model2,
                ),
            )
            processes.append(p)
            p.start()
    except _EvalCancelled as exc:
        _terminate_eval_processes(processes, timeout_s=0.2)
        with contextlib.suppress(Exception):
            result_queue.close()
        stats = _build_eval_stats(0, 0, 0, 0, num_games)
        stats.update({
            "cancelled": True,
            "cancel_reason": str(exc),
            "cancel_stage": exc.stage,
            "completed": 0,
        })
        return stats
    except Exception:
        _terminate_eval_processes(processes, timeout_s=0.2)
        with contextlib.suppress(Exception):
            result_queue.close()
        raise

    _notify_eval_runtime(
        runtime_ready_callback,
        status="running",
        ready=True,
        workers=len(processes),
        server_count=0,
        central_inference=False,
        load_s=0.0,
    )

    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    completed = 0
    terminal_reasons = {}
    eval_profile = {}
    eval_bar = tqdm(total=num_games, desc=progress_desc, unit="game", disable=not show_progress)
    worker_error = None
    cancelled = False
    try:
        finished_workers = 0
        while finished_workers < len(processes):
            if _eval_stop_requested(stop_event):
                cancelled = True
                _terminate_eval_processes(processes)
                break
            try:
                message = result_queue.get(timeout=0.1)
            except Exception:
                continue
            message_type = message.get("type")
            if message_type == "progress":
                completed += int(message.get("completed", 0))
                eval_bar.n = min(num_games, completed)
                eval_bar.refresh()
                if progress_callback is not None:
                    progress_callback(
                        int(message.get("completed", 0)),
                        message.get("game_idx", None),
                        int(message.get("wins", 0)),
                        int(message.get("draws", 0)),
                        int(message.get("losses", 0)),
                        int(message.get("unresolved", 0)),
                        int(message.get("plies", 0)),
                        str(message.get("result", "*")),
                        bool(message.get("model1_as_white", True)),
                    )
            elif message_type == "result":
                wins += int(message.get("wins", 0))
                draws += int(message.get("draws", 0))
                losses += int(message.get("losses", 0))
                unresolved += int(message.get("unresolved", 0))
                _merge_eval_terminal_reasons(
                    terminal_reasons, message.get("terminal_reasons")
                )
                for key, value in dict(message.get("profile", {}) or {}).items():
                    if isinstance(value, (int, float)):
                        eval_profile[key] = eval_profile.get(key, 0) + value
                finished_workers += 1
            elif message_type == "interrupt":
                raise KeyboardInterrupt
            elif message_type == "error":
                worker_error = str(message.get("error", "unknown error"))
                _terminate_eval_processes(processes)
                break
    finally:
        eval_bar.close()
        _terminate_eval_processes(
            processes,
            timeout_s=0.1,
            graceful_timeout_s=0.4,
        )
        with contextlib.suppress(Exception):
            result_queue.cancel_join_thread()
        with contextlib.suppress(Exception):
            result_queue.close()

    if worker_error is not None:
        raise RuntimeError(f"Eval worker failed: {worker_error}")

    if unresolved > 0 and verbose:
        max_moves = _resolve_eval_max_moves(config)
        _print_eval_unresolved(
            "Eval",
            unresolved,
            num_games,
            max_moves,
            neutral_score=_resolve_eval_unresolved_score_neutral(config),
        )

    stats = _build_eval_stats(wins, draws, losses, unresolved, num_games)
    stats["profile"] = eval_profile
    stats["terminal_reasons"] = terminal_reasons
    stats["cancelled"] = bool(cancelled)
    if cancelled:
        stats["cancel_reason"] = "stop requested during game evaluation"
        stats["cancel_stage"] = "game evaluation"
    stats["completed"] = int(completed)
    _notify_eval_runtime(
        runtime_ready_callback,
        status="cancelled" if cancelled else "completed",
        ready=False,
        completed=int(completed),
        requested_games=int(num_games),
        cancel_stage="game evaluation" if cancelled else None,
    )
    return stats


def evaluate_models_no_mcts(
    model1,
    model2,
    config,
    device,
    num_games=30,
    game_index_offset=0,
    use_fixed_openings=None,
    progress_callback=None,
):
    """Policy-head-only evaluation. This is intentionally separate from promotion eval."""
    wins = 0
    draws = 0
    losses = 0
    unresolved = 0
    max_moves = _resolve_eval_max_moves(config)
    auto_claim_draw = _resolve_eval_auto_claim_draw(config)
    claim_draw_after_moves = _resolve_eval_claim_draw_after_moves(config)
    claim_repetition_after_moves = _resolve_eval_claim_repetition_after_moves(config)

    model1.eval()
    model2.eval()
    eval_bar = tqdm(total=num_games, desc="Eval no MCTS", unit="game")
    try:
        for game_idx in range(game_index_offset, game_index_offset + num_games):
            model1_as_white = (game_idx % 2 == 0)
            white_model = model1 if model1_as_white else model2
            black_model = model2 if model1_as_white else model1
            opening_prefix = _get_eval_opening_prefix(config, game_idx, enabled_override=use_fixed_openings)
            result, was_unresolved = _evaluate_single_game_no_mcts(
                white_model,
                black_model,
                config,
                device,
                max_moves,
                auto_claim_draw,
                claim_draw_after_moves,
                claim_repetition_after_moves=claim_repetition_after_moves,
                opening_prefix=opening_prefix,
            )
            if was_unresolved:
                unresolved += 1
                game_wins = game_draws = game_losses = 0
                game_unresolved = 1
            else:
                game_wins, game_draws, game_losses = _result_for_model1(model1_as_white, result)
                wins += game_wins
                draws += game_draws
                losses += game_losses
                game_unresolved = 0
            if progress_callback is not None:
                progress_callback(
                    game_idx - game_index_offset + 1,
                    game_idx,
                    game_wins,
                    game_draws,
                    game_losses,
                    game_unresolved,
                    0,
                    result,
                    model1_as_white,
                )
            eval_bar.update(1)
    finally:
        eval_bar.close()

    if unresolved > 0:
        _print_eval_unresolved(
            "No-MCTS eval",
            unresolved,
            num_games,
            max_moves,
            neutral_score=_resolve_eval_unresolved_score_neutral(config),
        )

    return _build_eval_stats(wins, draws, losses, unresolved, num_games)
