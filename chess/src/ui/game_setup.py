"""Model loading and pre-game setup helpers for local GUI play."""

import copy
import csv
import ctypes
from datetime import datetime
import math
import os
from pathlib import Path
import sys

import chess
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import yaml

# Add utils to path for model_catalog import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from src.models.catalog import format_elo_stat_parts, format_elo_summary, load_checkpoint_metadata
from src.common.logger_helpers import _parse_mcts_elo_by_simulations
from src.ui.gui_helpers import create_piece_surfaces
from src.ui.play_home_layout import build_home_layout, build_picker_layout, ui_scale_for_viewport
from src.ui.theme import (
    ACCENT as _ACCENT,
    ACCENT_HOVER as _ACCENT_BORDER,
    ACCENT_HOVER as _ACCENT_HOVER,
    BORDER_STRONG as _BORDER_STRONG,
    BORDER as _BORDER,
    DANGER as _DANGER,
    MUTED as _MUTED,
    SUBTLE as _SUBTLE,
    SURFACE as _PANEL_BG,
    SURFACE_RAISED as _SURFACE_RAISED,
    TEXT as _TEXT,
    WARNING as _WARNING,
    build_background,
    lerp_color as _lerp_color,
)


def _enable_windows_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_enable_windows_dpi_awareness()


_CATEGORY_STYLE = {
    "best": {"label": "BEST MODELS", "fill": (32, 54, 48), "border": (105, 196, 171)},
    "il": {"label": "IMITATION LEARNING", "fill": (31, 46, 49), "border": (94, 157, 157)},
    "rl": {"label": "REINFORCEMENT LEARNING", "fill": (52, 46, 36), "border": (204, 162, 99)},
    "other": {"label": "OTHER MODELS", "fill": (37, 43, 45), "border": (119, 139, 138)},
}
_SETUP_PIECE_SURFACES = {}
_SETUP_PIECE_ICON_CACHE = {}
_PANEL_SURFACE_CACHE = {}


def _position_native_window(width, height):
    """Move the current window to the top-left corner with an exact size."""
    window_handle = _get_native_window_handle()
    if not window_handle:
        return False

    try:
        user32 = ctypes.windll.user32
        swp_framechanged = 0x0020
        swp_showwindow = 0x0040
        user32.SetWindowPos(
            int(window_handle),
            0,
            0,
            0,
            int(width),
            int(height),
            swp_framechanged | swp_showwindow,
        )
        return True
    except Exception:
        return False


def _get_native_window_handle():
    try:
        wm_info = pygame.display.get_wm_info()
    except Exception:
        return None
    return wm_info.get("window")


def _maximize_native_window():
    """Maximize the current window using the native window handle."""
    window_handle = _get_native_window_handle()
    if not window_handle:
        return False

    try:
        ctypes.windll.user32.ShowWindow(int(window_handle), 3)
        return True
    except Exception:
        return False


def _draw_panel(screen, rect, fill=None, border=None, radius=18, shadow=True, fill_alpha=None):
    fill = fill or _PANEL_BG
    border = border or _BORDER
    if rect.width <= 0 or rect.height <= 0:
        return
    alpha = 255 if fill_alpha is None else int(max(0, min(255, fill_alpha)))
    radius = max(0, min(int(radius), rect.width // 2, rect.height // 2))
    if shadow:
        shadow_rect = rect.move(0, 4)
        shadow_key = (rect.size, radius, "shadow")
        shadow_surface = _PANEL_SURFACE_CACHE.get(shadow_key)
        if shadow_surface is None:
            shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                shadow_surface,
                (0, 0, 0, 42),
                shadow_surface.get_rect(),
                border_radius=radius,
            )
            _PANEL_SURFACE_CACHE[shadow_key] = shadow_surface
        screen.blit(shadow_surface, shadow_rect.topleft)

    fill_key = (rect.size, radius, tuple(fill), alpha)
    fill_surface = _PANEL_SURFACE_CACHE.get(fill_key)
    if fill_surface is None:
        fill_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            fill_surface,
            (*fill, alpha),
            fill_surface.get_rect(),
            border_radius=radius,
        )
        _PANEL_SURFACE_CACHE[fill_key] = fill_surface
    screen.blit(fill_surface, rect.topleft)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)
    if len(_PANEL_SURFACE_CACHE) > 512:
        _PANEL_SURFACE_CACHE.clear()


def _get_setup_piece_surfaces(square_size):
    cache_key = max(1, int(square_size))
    cached = _SETUP_PIECE_SURFACES.get(cache_key)
    if cached is None:
        cached = create_piece_surfaces(cache_key)
        _SETUP_PIECE_SURFACES[cache_key] = cached
    return cached


def _scaled_setup_piece_icon(symbol, target_size, fill_ratio=0.82):
    cache_key = (str(symbol), int(target_size), float(fill_ratio))
    cached = _SETUP_PIECE_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached
    surface = _get_setup_piece_surfaces(target_size).get(symbol)
    if surface is None:
        return None
    bbox = surface.get_bounding_rect(min_alpha=1)
    if bbox.width <= 0 or bbox.height <= 0:
        result = pygame.transform.smoothscale(surface, (target_size, target_size))
        _SETUP_PIECE_ICON_CACHE[cache_key] = result
        return result
    cropped = surface.subsurface(bbox).copy()
    target_inner = max(1, int(target_size * fill_ratio))
    scale = min(target_inner / cropped.get_width(), target_inner / cropped.get_height())
    new_w = max(1, int(cropped.get_width() * scale))
    new_h = max(1, int(cropped.get_height() * scale))
    scaled = pygame.transform.smoothscale(cropped, (new_w, new_h))
    canvas = pygame.Surface((target_size, target_size), pygame.SRCALPHA)
    canvas.blit(scaled, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    _SETUP_PIECE_ICON_CACHE[cache_key] = canvas
    return canvas


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _visible_move_delay(value, default=1.0):
    """Normalize the per-visible-move pacing choice to the supported UI range."""
    try:
        delay = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(delay):
        return float(default)
    return min(5.0, max(0.0, delay))


def _infer_architecture_from_state_dict(state_dict):
    inferred = {}
    if not isinstance(state_dict, dict):
        return inferred

    filters = None
    final_bn_weight = state_dict.get("final_bn.weight")
    if final_bn_weight is not None and getattr(final_bn_weight, "ndim", 0) >= 1:
        filters = int(final_bn_weight.shape[0])
        inferred["filters"] = filters

    block_ids = set()
    for key in state_dict.keys():
        if not key.startswith("residual_tower."):
            continue
        parts = key.split(".")
        if len(parts) >= 2:
            block_id = _safe_int(parts[1], default=None)
            if block_id is not None:
                block_ids.add(block_id)
    if block_ids:
        inferred["num_residual_blocks"] = max(block_ids) + 1

    input_planes = None
    coord_weight = state_dict.get("conv_block.0.conv.weight")
    plain_weight = state_dict.get("conv_block.0.weight")
    if coord_weight is not None and getattr(coord_weight, "ndim", 0) == 4:
        inferred["use_coord_conv"] = True
        input_planes = int(coord_weight.shape[1]) - 2
    elif plain_weight is not None and getattr(plain_weight, "ndim", 0) == 4:
        inferred["use_coord_conv"] = False
        input_planes = int(plain_weight.shape[1])

    if input_planes is not None and input_planes > 0 and input_planes % 16 == 0:
        inferred["history_positions"] = max(0, (input_planes // 16) - 1)

    policy_conv_weight = state_dict.get("policy_conv.weight")
    if policy_conv_weight is not None and getattr(policy_conv_weight, "ndim", 0) == 4:
        inferred["policy_head_channels"] = int(policy_conv_weight.shape[0])
        inferred["policy_head_conv_filters"] = int(policy_conv_weight.shape[0])
        in_per_group = int(policy_conv_weight.shape[1])
        if filters and in_per_group > 0 and filters % in_per_group == 0:
            inferred["policy_head_conv_groups"] = int(filters // in_per_group)

    policy_global_fc = state_dict.get("policy_global_fc.weight")
    if policy_global_fc is not None and getattr(policy_global_fc, "ndim", 0) == 2:
        inferred["policy_head_global_dim"] = int(policy_global_fc.shape[0])

    policy_fc1 = state_dict.get("policy_fc1.weight")
    if policy_fc1 is not None and getattr(policy_fc1, "ndim", 0) == 2:
        inferred["policy_head_hidden_dim"] = int(policy_fc1.shape[0])

    value_conv_weight = state_dict.get("value_conv.weight")
    if value_conv_weight is not None and getattr(value_conv_weight, "ndim", 0) == 4:
        inferred["value_head_filters"] = int(value_conv_weight.shape[0])

    value_fc1 = state_dict.get("value_fc1.weight")
    if value_fc1 is not None and getattr(value_fc1, "ndim", 0) == 2:
        inferred["value_hidden_dim"] = int(value_fc1.shape[0])
    moves_left_fc1 = state_dict.get("moves_left_fc1.weight")
    if moves_left_fc1 is not None and getattr(moves_left_fc1, "ndim", 0) == 2:
        inferred["moves_left_hidden_dim"] = int(moves_left_fc1.shape[0])

    se_fc1_keys = [key for key in state_dict.keys() if ".se.fc1.weight" in key]
    se_fc_keys = [key for key in state_dict.keys() if ".se.fc.weight" in key]
    use_se_blocks = bool(se_fc1_keys or se_fc_keys)
    inferred["use_se_blocks"] = use_se_blocks
    if use_se_blocks:
        inferred["use_se_bottleneck"] = bool(se_fc1_keys)
        if se_fc1_keys:
            fc1_weight = state_dict.get(se_fc1_keys[0])
            if fc1_weight is not None and getattr(fc1_weight, "ndim", 0) == 4:
                in_ch = int(fc1_weight.shape[1])
                mid = int(fc1_weight.shape[0])
                if in_ch > 0 and mid > 0:
                    inferred["se_reduction"] = max(1, in_ch // mid)
    inferred["use_layer_scale"] = any(".layer_scale.gamma" in key for key in state_dict.keys())

    return inferred


class ModelCompatibilityError(RuntimeError):
    """Raised when checkpoint tensors do not match the reconstructed model."""

    def __init__(self, checkpoint_path, mismatches):
        self.checkpoint_path = Path(checkpoint_path)
        self.mismatches = list(mismatches)
        details = ", ".join(
            f"{key}: {checkpoint_shape} -> {runtime_shape}"
            for key, checkpoint_shape, runtime_shape in self.mismatches[:3]
        )
        super().__init__(
            f"Checkpoint '{self.checkpoint_path.name}' uses an incompatible model architecture"
            + (f" ({details})" if details else ".")
        )


def load_model_from_checkpoint(checkpoint_path, config, device, verbose=False):
    """Load a model checkpoint and switch model to eval mode."""
    if checkpoint_path.exists():
        from src.model import (
            create_model,
            load_checkpoint_file,
            normalize_state_dict_keys,
            resolve_model_spec,
        )

        checkpoint = load_checkpoint_file(checkpoint_path, device)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing a valid model_state_dict."
            )

        model_spec = resolve_model_spec(config, checkpoint)
        runtime_config = copy.deepcopy(config)
        runtime_config["model"]["print_summary"] = bool(verbose)

        model = create_model(runtime_config, model_spec=model_spec).to(device)
        model_state = normalize_state_dict_keys(
            model_state, target_keys=set(model.state_dict())
        )
        # CoordConv coord_x/coord_y buffers can trigger overlap copy errors on load.
        # They are deterministic and rebuilt in model init, so skipping is safe.
        coordconv_buffer_suffixes = ("coord_x", "coord_y")
        skipped_coord_buffers = [
            key for key in model_state.keys() if key.endswith(coordconv_buffer_suffixes)
        ]
        if skipped_coord_buffers:
            model_state = {
                key: value
                for key, value in model_state.items()
                if not key.endswith(coordconv_buffer_suffixes)
            }

        runtime_state = model.state_dict()
        shape_mismatches = []
        for key, checkpoint_value in model_state.items():
            runtime_value = runtime_state.get(key)
            checkpoint_shape = getattr(checkpoint_value, "shape", None)
            runtime_shape = getattr(runtime_value, "shape", None)
            if runtime_value is not None and checkpoint_shape is not None and runtime_shape is not None:
                checkpoint_shape = tuple(int(dim) for dim in checkpoint_shape)
                runtime_shape = tuple(int(dim) for dim in runtime_shape)
                if checkpoint_shape != runtime_shape:
                    shape_mismatches.append((key, checkpoint_shape, runtime_shape))
        if shape_mismatches:
            raise ModelCompatibilityError(checkpoint_path, shape_mismatches)

        incompatible = model.load_state_dict(model_state, strict=False)
        missing_keys = [
            key
            for key in list(getattr(incompatible, "missing_keys", []) or [])
            if not key.endswith(coordconv_buffer_suffixes)
        ]
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])

        if verbose:
            print(f"Loaded model: {checkpoint_path.name}")
            model_cfg = getattr(model, "model_config", None) or runtime_config.get("model", {})
            architecture_id = str(getattr(model, "architecture_id", "se_cnn_v9"))
            blocks = model_cfg.get("outer_blocks", model_cfg.get("num_residual_blocks"))
            channels = model_cfg.get("trunk_channels", model_cfg.get("filters"))
            print(
                f"  Arch: {architecture_id}, blocks={blocks}, channels={channels}, "
                f"history={getattr(model, 'history_positions', model_cfg.get('history_positions'))}, "
                f"input_planes={int(getattr(model, 'input_planes', 0))}"
            )

            if missing_keys or unexpected_keys:
                print(
                    "  State dict compatibility: "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)} (loaded with strict=False)"
                )
                if missing_keys:
                    print(f"    missing sample: {missing_keys[:3]}")
                if unexpected_keys:
                    print(f"    unexpected sample: {unexpected_keys[:3]}")
            elif skipped_coord_buffers:
                print(
                    f"  State dict compatibility: skipped {len(skipped_coord_buffers)} CoordConv buffers "
                    "(coord_x/coord_y)."
                )

            if "val_policy_loss" in checkpoint:
                val_loss = checkpoint.get("val_loss", checkpoint.get("loss"))
                if isinstance(val_loss, (int, float)):
                    print(f"  Val Loss: {float(val_loss):.4f}")
                print(f"  Policy Loss: {checkpoint['val_policy_loss']:.4f}")
                print(f"  Value Loss: {checkpoint['val_value_loss']:.4f}")
            elif "score_rate" in checkpoint or "win_rate" in checkpoint:
                score_rate = checkpoint.get("score_rate", checkpoint.get("win_rate"))
                if isinstance(score_rate, (int, float)):
                    print(f"  Score Rate: {float(score_rate):.2%}")
                true_win_rate = checkpoint.get("eval_true_win_rate", checkpoint.get("true_win_rate", checkpoint.get("win_rate")))
                if isinstance(true_win_rate, (int, float)):
                    print(f"  True Win Rate: {float(true_win_rate):.2%}")
    else:
        runtime_config = copy.deepcopy(config)
        runtime_config.setdefault("model", {})
        runtime_config["model"]["print_summary"] = bool(verbose)
        model = model_class(runtime_config).to(device)
        print(f"Model file not found: {checkpoint_path}")
        print("Using untrained model.")

    model.eval()
    return model


def _discover_models(base_dir, config):
    models_dir = base_dir / config["paths"]["models_dir"]
    all_models = list(models_dir.glob("*.pt"))
    all_models.extend((models_dir / "IL").rglob("*.pt"))
    all_models.extend((models_dir / "RL").rglob("*.pt"))
    all_models.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return models_dir, all_models


def _group_models(all_models):
    grouped = {"best": [], "il": [], "rl": [], "other": []}

    for model_path in all_models:
        parts = [part.lower() for part in model_path.parts]
        try:
            models_idx = parts.index("models")
            rel_parts = parts[models_idx + 1 :]
        except ValueError:
            rel_parts = [model_path.name.lower()]

        if len(rel_parts) <= 1:
            grouped["best"].append(model_path)
        elif rel_parts[0] == "il":
            grouped["il"].append(model_path)
        elif rel_parts[0] == "rl":
            grouped["rl"].append(model_path)
        else:
            grouped["other"].append(model_path)

    return grouped


def _default_model(grouped, architecture_id=None, metadata_cache=None):
    if architecture_id:
        metadata_cache = metadata_cache or {}
        for key in ("best", "il", "rl", "other"):
            for model_path in grouped[key]:
                metadata = metadata_cache.get(model_path) or {}
                if metadata.get("architecture_id") == architecture_id:
                    return model_path
        # Do not default a newly selected architecture to a different model.
        return None
    for key in ("best", "il", "rl", "other"):
        if grouped[key]:
            return grouped[key][0]
    return None


def _resolve_games_dir(base_dir, config):
    logs_dir = base_dir / config.get("paths", {}).get("logs_dir", "logs")
    games_dir_cfg = config.get("paths", {}).get("games_dir")
    if games_dir_cfg:
        games_dir = Path(games_dir_cfg)
        if not games_dir.is_absolute():
            games_dir = base_dir / games_dir
    else:
        games_dir = logs_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    return games_dir


def write_setup_log(base_dir, config, setup):
    """Write selected play settings to a single file and return file path."""
    games_dir = _resolve_games_dir(base_dir, config)
    now = datetime.now()
    output = games_dir / "play_setup.yaml"

    # Cleanup legacy per-run setup files and optional PGN logs.
    for legacy in games_dir.glob("play_setup_*.yaml"):
        try:
            legacy.unlink()
        except OSError:
            pass
    legacy_preferences = games_dir / "play_preferences.yaml"
    if legacy_preferences.exists():
        try:
            legacy_preferences.unlink()
        except OSError:
            pass
    save_games_pgn = bool((config or {}).get("play", {}).get("save_games_pgn", False))
    if not save_games_pgn:
        for pgn_file in games_dir.glob("*.pgn"):
            try:
                pgn_file.unlink()
            except OSError:
                pass

    payload = {
        "updated_at": now.isoformat(timespec="seconds"),
        "game_mode": setup.get("game_mode"),
        "human_color": setup.get("human_color_name"),
        "use_mcts": bool(setup.get("use_mcts", False)),
        "use_mcts_white": bool(setup.get("use_mcts_white", setup.get("use_mcts", False))),
        "use_mcts_black": bool(setup.get("use_mcts_black", setup.get("use_mcts", False))),
        "mcts_simulations": setup.get("mcts_simulations"),
        "mcts_simulations_white": setup.get("mcts_simulations_white"),
        "mcts_simulations_black": setup.get("mcts_simulations_black"),
        "match_games": setup.get("match_games"),
        "min_visible_move_seconds": _visible_move_delay(setup.get("min_visible_move_seconds", 1.0)),
        "model_ai": str(setup.get("model1_path")) if setup.get("model1_path") else None,
        "model_white": str(setup.get("model_white")) if setup.get("model_white") else None,
        "model_black": str(setup.get("model_black")) if setup.get("model_black") else None,
        "source": "setup_window",
    }

    with open(output, "w", encoding="utf-8", newline="\n") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=False)

    return output


def _resolve_play_preferences_path(base_dir, config):
    # Keep one shared settings file in logs/games.
    return _resolve_games_dir(base_dir, config) / "play_setup.yaml"


def load_play_preferences(base_dir, config):
    path = _resolve_play_preferences_path(base_dir, config)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def save_play_preferences(base_dir, config, preferences):
    path = _resolve_play_preferences_path(base_dir, config)
    existing_payload = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as file_obj:
                existing_payload = yaml.safe_load(file_obj) or {}
            if not isinstance(existing_payload, dict):
                existing_payload = {}
        except Exception:
            existing_payload = {}
    sims = _safe_int((preferences or {}).get("mcts_simulations"), default=100)
    sims_white = _safe_int((preferences or {}).get("mcts_simulations_white"), default=sims)
    sims_black = _safe_int((preferences or {}).get("mcts_simulations_black"), default=sims)
    if sims is None:
        sims = 100
    if sims_white is None:
        sims_white = sims
    if sims_black is None:
        sims_black = sims
    match_games = _safe_int((preferences or {}).get("match_games"), default=1)
    if match_games is None:
        match_games = 1
    min_visible_move_seconds = _visible_move_delay(
        (preferences or {}).get("min_visible_move_seconds", 1.0)
    )
    payload = dict(existing_payload)
    payload.update(
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "use_mcts": bool((preferences or {}).get("use_mcts", False)),
            "use_mcts_white": bool((preferences or {}).get("use_mcts_white", (preferences or {}).get("use_mcts", False))),
            "use_mcts_black": bool((preferences or {}).get("use_mcts_black", (preferences or {}).get("use_mcts", False))),
            "mcts_simulations": max(1, int(sims)),
            "mcts_simulations_white": max(1, int(sims_white)),
            "mcts_simulations_black": max(1, int(sims_black)),
            "match_games": max(1, int(match_games)),
            "min_visible_move_seconds": min_visible_move_seconds,
        }
    )
    for key in ("game_mode", "human_color", "model_ai", "model_white", "model_black"):
        if key in (preferences or {}):
            payload[key] = (preferences or {}).get(key)
    with open(path, "w", encoding="utf-8", newline="\n") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=False)
    return path


def _resolve_saved_model_path(saved_value, models_dir, all_models):
    if not saved_value:
        return None
    try:
        saved_path = Path(str(saved_value))
    except Exception:
        return None

    candidates = []
    if saved_path.is_absolute():
        candidates.append(saved_path)
    else:
        candidates.append((models_dir / saved_path).resolve())
        candidates.append((models_dir / str(saved_value).replace("/", "\\")).resolve())

    all_model_set = set(all_models)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in all_model_set:
            return resolved
    return None


def _short_model_path(path, models_dir, max_len=66):
    if path is None:
        return "None"
    rel = str(path.relative_to(models_dir)).replace("\\", "/")
    if len(rel) > max_len:
        rel = "..." + rel[-(max_len - 3) :]
    return rel


def _model_origin_label(model_path, models_dir, metadata=None):
    """Return the checkpoint's directory origin relative to the models root.

    Root-level aliases have no parent directory, so a small set of established
    canonical names and explicit catalog metadata provide their source. Generic
    filename fragments such as ``*_il_*`` are deliberately not treated as proof.
    """
    if model_path is None:
        return None

    path = Path(model_path)
    metadata = metadata if isinstance(metadata, dict) else {}

    def from_relative_path(relative_path):
        try:
            relative = Path(str(relative_path).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) <= 1:
                return None
            directories = relative.parts[:-1]
            if directories[0].lower() in {"il", "rl"}:
                directories = (directories[0].upper(), *directories[1:])
            return "/".join(directories)
        except (TypeError, ValueError, OSError):
            return None

    # The actual checkpoint location is authoritative whenever it is under the
    # configured models root.
    try:
        relative = path.resolve().relative_to(Path(models_dir).resolve())
    except (ValueError, OSError, RuntimeError):
        relative = None
    if relative is not None and len(relative.parts) > 1:
        return from_relative_path(relative)

    # Catalog metadata can recover the original models-relative location for
    # callers holding a resolved alias or a path outside the configured root.
    metadata_origin = from_relative_path(metadata.get("path_rel"))
    if metadata_origin:
        return metadata_origin

    stage = None
    alias_kind = str(metadata.get("alias_kind") or "").strip().casefold()
    if alias_kind in {
        "v10 il best",
        "v10 il swa",
        "legacy v10 il best",
        "legacy v10 il swa",
    }:
        stage = "IL"
    elif alias_kind in {"v10 rl best", "legacy v10 rl best"}:
        stage = "RL"

    folder = str(metadata.get("folder") or "").strip().upper()
    if stage is None and folder in {"IL", "RL"}:
        stage = folder
    rl_run_version = str(metadata.get("rl_run_version") or "").strip()
    if stage is None and rl_run_version:
        stage = "RL"

    # These exact root aliases are part of the project's model contract. Do
    # not infer a source from arbitrary substrings in other filenames.
    canonical_aliases = {
        "best_model_il": "IL",
        "best_model_il_swa": "IL",
        "best_model_il_nested_cnn_v10": "IL",
        "best_model_il_swa_nested_cnn_v10": "IL",
        "best_model_v10_il": "IL",
        "best_model_v10_il_swa": "IL",
        "best_model_rl": "RL",
        "best_model_rl_nested_cnn_v10": "RL",
        "best_model_v10_rl": "RL",
    }
    if stage is None:
        stage = canonical_aliases.get(path.stem.casefold())

    if stage == "RL" and rl_run_version:
        if rl_run_version.casefold() == "rl":
            return "RL"
        display_version = (
            f"RL{rl_run_version[2:]}"
            if rl_run_version.casefold().startswith("rl")
            else rl_run_version
        )
        return f"RL/{display_version}"
    return stage


def _model_browser_label(model_path, models_dir, metadata=None):
    """Show directory origin before a concise checkpoint name in the picker."""
    if model_path is None:
        return "None"
    origin = _model_origin_label(model_path, models_dir, metadata)
    if origin:
        return f"{origin} · {Path(model_path).name}"
    try:
        return _short_model_path(Path(model_path), Path(models_dir), max_len=240)
    except (ValueError, OSError):
        return Path(model_path).name


def _format_model_entry(path, models_dir, max_len=66, metadata=None):
    """Format model entry with metadata (version, top1, elo).
    
    Returns:
        tuple: (line1_text, line2_text) for two-line display
    """
    rel = _short_model_path(path, models_dir, max_len=max_len)
    
    # Load checkpoint metadata
    if metadata is None:
        metadata = load_checkpoint_metadata(path, models_dir)
    
    # Build second line with key metrics
    parts = []
    
    version = metadata.get("version")
    if version:
        parts.append(f"v{version}" if not str(version).startswith("v") else str(version))
    
    epoch = metadata.get("epoch")
    if epoch is not None:
        parts.append(f"ep{epoch + 1}")
    
    top1 = metadata.get("top1")
    if top1 is not None:
        parts.append(f"Top1:{top1 * 100:.1f}%")
    
    elo_summary = format_elo_summary(metadata)
    if elo_summary != "n/a":
        parts.append(elo_summary)
    
    size_mb = metadata.get("size_mb", 0.0)
    parts.append(f"{size_mb:.1f}MB")
    
    if metadata.get("swa"):
        parts.append("[SWA]")
    
    line2 = " | ".join(parts) if parts else "No metadata"
    
    return rel, line2


def _draw_button(
    screen,
    rect,
    text,
    font,
    hovered=False,
    active=False,
    disabled=False,
    danger=False,
):
    if disabled:
        fill = (35, 44, 48)
        text_color = _SUBTLE
        border = (53, 66, 70)
    elif danger:
        fill = _DANGER if active else (91, 51, 49)
        text_color = _TEXT
        border = _DANGER if active else (126, 75, 70)
    elif active:
        fill = (37, 90, 76)
        text_color = (239, 247, 239)
        border = _ACCENT
    elif hovered:
        fill = _SURFACE_RAISED
        text_color = _TEXT
        border = _BORDER_STRONG
    else:
        fill = (27, 37, 42)
        text_color = _TEXT
        border = _BORDER

    radius = max(6, min(10, rect.height // 3))
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)
    label_text = str(text) if str(text) in ("+", "-", "−", "×", "X", "<", ">") else _fit_text(font, text, max(12, rect.width - 20))
    label = font.render(label_text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_home_player_card(
    screen,
    card_rect,
    model_rect,
    side_label,
    model_path,
    models_dir,
    metadata_cache,
    fonts,
    scale,
    side_color,
    use_mcts=None,
    simulations=None,
    budget_mode="simulations",
    move_time_seconds=1.0,
    mcts_rect=None,
    budget_minus_rect=None,
    budget_value_rect=None,
    budget_plus_rect=None,
    is_human=False,
    hovered=False,
    mouse_pos=None,
):
    """Draw one focused player/model profile and return its clickable model controls."""
    px = lambda value: max(1, int(round(value * scale)))
    accent = _ACCENT if side_color == chess.WHITE else _WARNING
    fill = (23, 34, 38) if side_color == chess.WHITE else (34, 31, 26)
    border = accent if hovered else (61, 79, 82)
    _draw_panel(screen, card_rect, fill=fill, border=border, radius=px(14), shadow=False)

    inset = px(18)
    header = fonts["card_label"].render(side_label.upper(), True, accent)
    screen.blit(header, (card_rect.left + inset, card_rect.top + px(14)))

    if is_human:
        _draw_panel(screen, model_rect, fill=(28, 40, 44), border=(61, 79, 82), radius=px(12), shadow=False)
        player_title = fonts["card_name"].render("Local player", True, _TEXT)
        screen.blit(player_title, (model_rect.left + px(16), model_rect.top + max(1, (model_rect.height - player_title.get_height()) // 2)))
        hint = fonts["card_meta"].render("Moves are entered on the board", True, _MUTED)
        screen.blit(hint, (model_rect.left + px(16), model_rect.bottom + px(8)))
        return {"model": model_rect}

    _draw_panel(screen, model_rect, fill=(30, 43, 48), border=(69, 91, 94), radius=px(12), shadow=False)
    icon_size = min(px(56), max(px(32), model_rect.height - px(20)))
    icon_rect = pygame.Rect(model_rect.left + px(12), model_rect.top + (model_rect.height - icon_size) // 2, icon_size, icon_size)
    symbol = "K" if side_color == chess.WHITE else "k"
    icon = _scaled_setup_piece_icon(symbol, icon_size, fill_ratio=0.84)
    if icon is not None:
        screen.blit(icon, icon_rect)

    text_left = icon_rect.right + px(12)
    text_width = max(1, model_rect.right - text_left - px(80))
    if model_path is None:
        model_name = "Choose a model"
        model_meta = "Open the checkpoint browser"
    else:
        metadata = metadata_cache.get(model_path) or {}
        model_name = metadata.get("model_name") or Path(model_path).name
        origin = _model_origin_label(model_path, models_dir, metadata)
        details = _format_model_entry(model_path, models_dir, max_len=256, metadata=metadata)[1]
        model_meta = " | ".join(part for part in (origin, details) if part)
    name_surface = fonts["card_name"].render(
        _fit_model_identity(fonts["card_name"], model_name, text_width), True, _TEXT
    )
    screen.blit(name_surface, (text_left, model_rect.top + px(11)))
    meta_y = model_rect.top + px(15) + name_surface.get_height()
    meta_surface = fonts["card_meta"].render(_fit_text(fonts["card_meta"], model_meta, text_width), True, _MUTED)
    screen.blit(meta_surface, (text_left, meta_y))
    change = fonts["card_meta"].render("CHANGE", True, accent)
    screen.blit(change, change.get_rect(midright=(model_rect.right - px(14), model_rect.centery)))

    stats_y = model_rect.bottom + px(14)
    stat_gap = px(8)
    stat_width = (card_rect.width - 2 * inset - 2 * stat_gap) // 3
    metadata = (metadata_cache or {}).get(model_path) or {}
    top1 = metadata.get("top1")
    try:
        top1 = f"{float(top1) * 100:.1f}%" if top1 is not None else "n/a"
    except (TypeError, ValueError):
        top1 = "n/a"
    stats = (
        ("TOP 1", top1),
        ("NN ELO", _fmt_float(metadata.get("elo_nn", metadata.get("elo")), "{:.0f}")),
        ("MCTS ELO", _fmt_float(metadata.get("elo_mcts"), "{:.0f}")),
    )
    stat_h = px(54)
    for index, (label, value) in enumerate(stats):
        stat_rect = pygame.Rect(card_rect.left + inset + index * (stat_width + stat_gap), stats_y, stat_width, stat_h)
        pygame.draw.rect(screen, (18, 27, 31), stat_rect, border_radius=px(9))
        pygame.draw.rect(screen, (47, 65, 69), stat_rect, width=max(1, px(1)), border_radius=px(9))
        label_surface = fonts["card_meta"].render(_fit_text(fonts["card_meta"], label, stat_rect.width - px(12)), True, _MUTED)
        value_surface = fonts["card_name"].render(_fit_text(fonts["card_name"], value, stat_rect.width - px(12)), True, _TEXT)
        screen.blit(label_surface, (stat_rect.left + px(8), stat_rect.top + px(5)))
        screen.blit(value_surface, (stat_rect.left + px(8), stat_rect.top + px(23)))

    if mcts_rect is not None:
        search_label = fonts["card_meta"].render("MOVE SEARCH", True, _MUTED)
        screen.blit(search_label, (mcts_rect.left, mcts_rect.top - search_label.get_height() - px(4)))
        enabled = bool(use_mcts)
        _draw_button(
            screen,
            mcts_rect,
            "MCTS ON · POLICY + VALUE" if enabled else "MCTS OFF · NEURAL NET ONLY",
            fonts["meta"],
            hovered=bool(mouse_pos and mcts_rect.collidepoint(mouse_pos)),
            active=enabled,
        )
        if budget_minus_rect and budget_value_rect and budget_plus_rect:
            fixed_budget = str(budget_mode).strip().lower() != "time"
            _draw_button(screen, budget_minus_rect, "−", fonts["h2"], hovered=bool(mouse_pos and budget_minus_rect.collidepoint(mouse_pos)), disabled=not enabled or not fixed_budget)
            budget_text = (
                f"{float(move_time_seconds):g}s per move"
                if enabled and not fixed_budget
                else f"{int(simulations or 0)} sims per move" if enabled
                else "MCTS budget inactive"
            )
            budget_surface = fonts["small"].render(_fit_text(fonts["small"], budget_text, budget_value_rect.width - px(14)), True, _TEXT if enabled else _SUBTLE)
            pygame.draw.rect(screen, (18, 26, 30), budget_value_rect, border_radius=px(9))
            pygame.draw.rect(screen, (54, 72, 76) if enabled else (39, 51, 55), budget_value_rect, width=max(1, px(1)), border_radius=px(9))
            screen.blit(budget_surface, budget_surface.get_rect(center=budget_value_rect.center))
            _draw_button(screen, budget_plus_rect, "+", fonts["h2"], hovered=bool(mouse_pos and budget_plus_rect.collidepoint(mouse_pos)), disabled=not enabled or not fixed_budget)

    return {
        "model": model_rect,
        "mcts": mcts_rect,
        "budget_minus": budget_minus_rect,
        "budget_value": budget_value_rect,
        "budget_plus": budget_plus_rect,
    }


def _fit_text(font, text, max_width):
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed and font.size(trimmed + suffix)[0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + suffix) if trimmed else suffix


def _fit_model_identity(font, text, max_width):
    """Keep both ends of long model names visible so related checkpoints remain distinguishable."""
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text

    ellipsis = "…"
    if font.size(ellipsis)[0] > max_width:
        return _fit_text(font, text, max_width)

    low, high = 0, max(0, len(text) - 1)
    best = ellipsis
    while low <= high:
        kept = (low + high) // 2
        left_count = (kept + 1) // 2
        right_count = kept // 2
        candidate = text[:left_count] + ellipsis + (text[-right_count:] if right_count else "")
        if font.size(candidate)[0] <= max_width:
            best = candidate
            low = kept + 1
        else:
            high = kept - 1
    return best


def show_model_load_error(checkpoint_path, error):
    """Show a blocking, styled load error and return to model selection."""
    screen = pygame.display.get_surface()
    if screen is None:
        screen = pygame.display.set_mode((900, 560), pygame.RESIZABLE)
    pygame.display.set_caption("Chess AI - Model compatibility")
    clock = pygame.time.Clock()

    is_compatibility_error = isinstance(error, ModelCompatibilityError)
    title_text = "Model is incompatible" if is_compatibility_error else "Model could not be loaded"
    body_text = (
        "This checkpoint uses a different network architecture than the current version."
        if is_compatibility_error
        else "The checkpoint is damaged, incomplete, or unsupported by the current version."
    )
    mismatch_rows = list(getattr(error, "mismatches", []) or [])[:3]

    while True:
        width, height = screen.get_size()
        screen.blit(build_background(width, height), (0, 0))

        card_w = min(720, max(560, width - 64))
        card_h = min(390, max(330, height - 80))
        card = pygame.Rect((width - card_w) // 2, (height - card_h) // 2, card_w, card_h)
        _draw_panel(screen, card, fill=(22, 29, 40), border=(105, 137, 181), radius=20, shadow=True)
        pygame.draw.rect(
            screen,
            _DANGER,
            pygame.Rect(card.left + 24, card.top, card.width - 48, 3),
            border_radius=2,
        )

        title_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        body_font = pygame.font.SysFont("Segoe UI", 17)
        meta_font = pygame.font.SysFont("Segoe UI", 14)
        button_font = pygame.font.SysFont("Segoe UI", 16)

        icon_center = (card.left + 48, card.top + 48)
        pygame.draw.circle(screen, (74, 39, 45), icon_center, 22)
        pygame.draw.circle(screen, _DANGER, icon_center, 22, width=1)
        warning = title_font.render("!", True, (255, 224, 226))
        screen.blit(warning, warning.get_rect(center=icon_center))

        screen.blit(title_font.render(title_text, True, _TEXT), (card.left + 82, card.top + 28))
        model_name = Path(checkpoint_path).name
        screen.blit(body_font.render(_fit_text(body_font, model_name, card.width - 64), True, (210, 221, 237)), (card.left + 32, card.top + 88))
        screen.blit(body_font.render(_fit_text(body_font, body_text, card.width - 64), True, _MUTED), (card.left + 32, card.top + 122))

        details_top = card.top + 164
        if mismatch_rows:
            details_rect = pygame.Rect(card.left + 32, details_top, card.width - 64, 94)
            pygame.draw.rect(screen, (17, 22, 31), details_rect, border_radius=12)
            pygame.draw.rect(screen, (68, 84, 108), details_rect, width=1, border_radius=12)
            screen.blit(meta_font.render("Incompatible layers", True, (163, 180, 204)), (details_rect.left + 14, details_rect.top + 10))
            row_y = details_rect.top + 34
            for key, checkpoint_shape, runtime_shape in mismatch_rows:
                row = f"{key}: saved {checkpoint_shape}, current {runtime_shape}"
                screen.blit(meta_font.render(_fit_text(meta_font, row, details_rect.width - 28), True, (203, 213, 228)), (details_rect.left + 14, row_y))
                row_y += 18
        else:
            reason = _fit_text(meta_font, str(error), card.width - 64)
            screen.blit(meta_font.render(reason, True, (203, 213, 228)), (card.left + 32, details_top))

        hint_text = _fit_text(
            meta_font,
            "Choose another checkpoint or use a model trained with the current architecture.",
            card.width - 64,
        )
        hint = meta_font.render(hint_text, True, _MUTED)
        screen.blit(hint, (card.left + 32, card.bottom - 86))

        button = pygame.Rect(card.right - 210, card.bottom - 58, 178, 40)
        mouse_pos = pygame.mouse.get_pos()
        _draw_button(screen, button, "Back to setup", button_font, hovered=button.collidepoint(mouse_pos), active=True)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and button.collidepoint(event.pos):
                return True
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(720, event.w), max(540, event.h)), pygame.RESIZABLE)
        clock.tick(60)


def _fmt_float(value, pattern):
    try:
        if value is None:
            return "n/a"
        return pattern.format(float(value))
    except Exception:
        return "n/a"


def _get_model_info_content(model_path, models_dir, metadata_cache):
    if model_path is None:
        return None, None, None, [], [("NN", "n/a"), ("MCTS", "n/a"), ("Top1", "n/a")]
    metadata = metadata_cache.get(model_path)
    if metadata is None:
        metadata = load_checkpoint_metadata(model_path, models_dir)
        metadata_cache[model_path] = metadata

    name = metadata.get("model_name") or model_path.name
    rel = metadata.get("path_rel") or _short_model_path(model_path, models_dir, max_len=64)

    epoch = metadata.get("epoch")
    epoch_value = str(int(epoch) + 1) if epoch is not None else "n/a"
    top1 = metadata.get("top1")
    top1_value = f"{float(top1) * 100:.2f}%" if top1 is not None else "n/a"
    version = metadata.get("version")
    version_value = f"v{version}" if version and not str(version).startswith("v") else (str(version) if version else "n/a")
    modified = metadata.get("modified") or "n/a"

    rows = [
        ("Version", version_value),
        ("Epoch", epoch_value),
        ("Val loss", _fmt_float(metadata.get("val_loss"), "{:.4f}")),
        ("Policy loss", _fmt_float(metadata.get("policy_loss"), "{:.4f}")),
        ("Modified", modified),
    ]

    elo_stats = format_elo_stat_parts(metadata)
    if len(elo_stats) >= 2:
        top_stats = elo_stats[:2] + [("Top1", top1_value)]
    else:
        top_stats = elo_stats + [("Top1", top1_value), ("Epoch", epoch_value)]
    top_stats = top_stats[:3]
    return metadata, name, rel, rows, top_stats


def _elo_progress_series(metadata):
    """Return one latest Elo pair per changing training checkpoint."""
    series = {"nn": [], "mcts": {}}
    progress = list(metadata.get("elo_progress") or [])
    for chronological_index, point in enumerate(progress):
        try:
            nn_elo = float(point.get("nn"))
        except (TypeError, ValueError):
            nn_elo = None
        if nn_elo is not None:
            series["nn"].append((chronological_index, nn_elo))
        for simulations, entry in _parse_mcts_elo_by_simulations(
            point.get("mcts_by_simulations")
        ).items():
            series["mcts"].setdefault(int(simulations), []).append(
                (chronological_index, float(entry["elo"]))
            )

    latest_index = max(0, len(progress) - 1)
    if not series["nn"] and metadata.get("elo_nn") is not None:
        series["nn"].append((latest_index, float(metadata["elo_nn"])))
    if not series["mcts"]:
        by_simulations = _parse_mcts_elo_by_simulations(
            metadata.get("elo_mcts_by_simulations")
        )
        if not by_simulations and metadata.get("elo_mcts") is not None:
            simulations = _safe_int(metadata.get("elo_mcts_simulations"), 0)
            if simulations > 0:
                by_simulations = {
                    simulations: {"elo": float(metadata["elo_mcts"])}
                }
        for simulations, entry in by_simulations.items():
            series["mcts"][int(simulations)] = [
                (latest_index, float(entry["elo"]))
            ]
    return series


def _checkpoint_stage_label(model_path):
    if model_path is None:
        return None
    path = Path(model_path)
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    if "rl" in parts or "_rl" in stem or stem.startswith("rl_"):
        return "RL"
    if "il" in parts or "_il" in stem or stem.startswith("il_"):
        return "IL"
    return None


def _read_training_elo_progress(csv_path, stage):
    """Read actual training checkpoints, not repeated evaluations of one file."""
    step_column = "iteration" if stage == "RL" else "epoch"
    points_by_step = {}
    try:
        with Path(csv_path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            rows = csv.DictReader(
                line.replace("\x00", "")
                for line in handle
                if not line.startswith("#")
            )
            for row_index, row in enumerate(rows):
                raw_step = str(row.get(step_column, "") or "").strip()
                try:
                    step = int(raw_step)
                except (TypeError, ValueError):
                    continue

                def optional_float(*keys):
                    for key in keys:
                        try:
                            value = str(row.get(key, "") or "").strip()
                            if value:
                                return float(value)
                        except (TypeError, ValueError):
                            continue
                    return None

                nn_elo = optional_float("estimated_elo_nn", "estimated_elo")
                mcts_elo = optional_float("estimated_elo_mcts")
                mcts_by_simulations = _parse_mcts_elo_by_simulations(
                    row.get("estimated_elo_mcts_by_simulations")
                )
                if not mcts_by_simulations and mcts_elo is not None:
                    simulations = optional_float("estimated_elo_mcts_simulations")
                    if simulations is not None and int(simulations) > 0:
                        mcts_by_simulations = {
                            int(simulations): {"elo": float(mcts_elo)}
                        }
                if nn_elo is None and not mcts_by_simulations:
                    continue

                # Appended/resumed CSVs can contain the same checkpoint more
                # than once. Later rows are the newer estimate for that model.
                point = points_by_step.setdefault(
                    step,
                    {
                        "step": step,
                        "nn": None,
                        "mcts": None,
                        "mcts_by_simulations": {},
                        "row": row_index,
                    },
                )
                if nn_elo is not None:
                    point["nn"] = nn_elo
                if mcts_elo is not None:
                    point["mcts"] = mcts_elo
                if mcts_by_simulations:
                    point["mcts_by_simulations"].update(mcts_by_simulations)
                point["row"] = row_index
    except (OSError, csv.Error):
        return []

    return [points_by_step[step] for step in sorted(points_by_step)]


def _training_elo_progress(model_path, metadata, logs_csv_dir, progress_cache):
    """Resolve the latest IL/RL run and overlay the checkpoint's newest Elo."""
    stage = _checkpoint_stage_label(model_path)
    version = str((metadata or {}).get("version") or "").strip()
    if stage is None or not version:
        return []

    cache_key = (stage, version)
    if cache_key not in progress_cache:
        if stage == "RL":
            patterns = (f"RL_{version}_*.csv", f"rl_training_{version}_*.csv")
        else:
            patterns = (f"IL_{version}_*.csv", f"il_training_{version}_*.csv")

        candidates = []
        for pattern in patterns:
            candidates.extend(Path(logs_csv_dir).glob(pattern))
        candidates = [
            path
            for path in set(candidates)
            if not path.stem.endswith(("_performance", "_data_quality"))
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        progress = []
        for candidate in candidates:
            progress = _read_training_elo_progress(candidate, stage)
            if progress:
                break
        progress_cache[cache_key] = progress

    progress = [dict(point) for point in progress_cache[cache_key]]
    latest_nn = (metadata or {}).get("elo_nn")
    latest_mcts = (metadata or {}).get("elo_mcts")
    latest_mcts_by_simulations = _parse_mcts_elo_by_simulations(
        (metadata or {}).get("elo_mcts_by_simulations")
    )
    if latest_nn is None and latest_mcts is None and not latest_mcts_by_simulations:
        return progress

    target_index = None
    # Prefer the row whose recorded result already identifies this checkpoint.
    for index in range(len(progress) - 1, -1, -1):
        point = progress[index]
        if (
            latest_nn is not None
            and point.get("nn") is not None
            and abs(float(point["nn"]) - float(latest_nn)) < 0.5
        ) or (
            latest_mcts is not None
            and point.get("mcts") is not None
            and abs(float(point["mcts"]) - float(latest_mcts)) < 0.5
        ):
            target_index = index
            break

    checkpoint_step = _safe_int((metadata or {}).get("epoch"))
    if target_index is None and checkpoint_step is not None:
        target_index = next(
            (index for index, point in enumerate(progress) if point["step"] == checkpoint_step),
            None,
        )
    if target_index is None and progress:
        target_index = len(progress) - 1
    if target_index is None:
        target_index = 0
        progress.append(
            {
                "step": checkpoint_step if checkpoint_step is not None else 0,
                "nn": None,
                "mcts": None,
                "mcts_by_simulations": {},
                "row": 0,
            }
        )

    if latest_nn is not None:
        progress[target_index]["nn"] = float(latest_nn)
    if latest_mcts is not None:
        progress[target_index]["mcts"] = float(latest_mcts)
    if latest_mcts_by_simulations:
        progress[target_index]["mcts_by_simulations"] = latest_mcts_by_simulations
    return progress


def _draw_model_info_panel(
    screen,
    rect,
    model_path,
    models_dir,
    metadata_cache,
    font_h2,
    font_text,
    font_meta,
    scale=1.0,
):
    px = lambda value: max(1, int(round(value * scale)))
    _draw_panel(screen, rect, fill=(23, 32, 37), border=_BORDER, radius=px(14), shadow=False)
    inner_left = rect.left + px(14)
    inner_width = max(1, rect.width - px(28))

    if model_path is None:
        title = font_h2.render("Model details", True, _TEXT)
        screen.blit(title, (inner_left, rect.top + px(14)))
        screen.blit(
            font_meta.render("Choose a checkpoint to inspect its metrics.", True, _MUTED),
            (inner_left, rect.top + px(48)),
        )
        return

    metadata, name, _rel, _rows, top_stats = _get_model_info_content(
        model_path, models_dir, metadata_cache
    )
    origin = _model_origin_label(model_path, models_dir, metadata)
    version = str((metadata or {}).get("version") or "").strip()
    epoch = _safe_int((metadata or {}).get("epoch"))
    identity = "  ·  ".join(
        part for part in (
            origin,
            f"v{version}" if version and not version.startswith("v") else version,
            f"epoch {epoch + 1}" if epoch is not None else None,
        ) if part
    ) or "Checkpoint"
    identity_surface = font_meta.render(_fit_text(font_meta, identity.upper(), inner_width), True, _ACCENT)
    screen.blit(identity_surface, (inner_left, rect.top + px(12)))
    name_rect = pygame.Rect(inner_left, rect.top + px(31), inner_width, font_text.get_height() + px(3))
    screen.blit(
        font_text.render(_fit_model_identity(font_text, name, name_rect.width), True, _TEXT),
        (name_rect.left, name_rect.top),
    )

    stat_gap = px(8)
    stat_y = name_rect.bottom + px(8)
    stat_h = px(54)
    stat_w = max(1, (inner_width - 2 * stat_gap) // 3)
    for idx, (label, value) in enumerate(top_stats[:3]):
        stat_rect = pygame.Rect(inner_left + idx * (stat_w + stat_gap), stat_y, stat_w, stat_h)
        pygame.draw.rect(screen, (18, 26, 30), stat_rect, border_radius=px(9))
        pygame.draw.rect(screen, (45, 61, 65), stat_rect, width=max(1, px(1)), border_radius=px(9))
        label_surface = font_meta.render(_fit_text(font_meta, label.upper(), stat_w - px(14)), True, _MUTED)
        value_surface = font_h2.render(_fit_text(font_h2, str(value), stat_w - px(14)), True, _TEXT)
        screen.blit(label_surface, (stat_rect.left + px(8), stat_rect.top + px(6)))
        screen.blit(value_surface, (stat_rect.left + px(8), stat_rect.top + px(25)))

    chart_top = stat_y + stat_h + 10
    chart_rect = pygame.Rect(inner_left, chart_top, inner_width, max(px(78), rect.bottom - chart_top - px(12)))
    pygame.draw.rect(screen, (18, 26, 30), chart_rect, border_radius=px(10))
    pygame.draw.rect(screen, (42, 58, 62), chart_rect, width=max(1, px(1)), border_radius=px(10))
    chart_title = "RATING HISTORY"
    title_surface = font_meta.render(chart_title, True, _MUTED)
    screen.blit(title_surface, (chart_rect.left + px(10), chart_rect.top + px(7)))

    series = _elo_progress_series(metadata or {})
    mcts_simulations = sorted(series["mcts"])[-3:]
    legend_text = "NN  ·  MCTS" if mcts_simulations else "NN"
    legend = font_meta.render(legend_text, True, _SUBTLE)
    if legend.get_width() < chart_rect.width - title_surface.get_width() - px(28):
        screen.blit(legend, (chart_rect.right - legend.get_width() - px(10), chart_rect.top + px(7)))

    samples = list(series["nn"])
    for simulations in mcts_simulations:
        samples.extend(series["mcts"][simulations])
    values = [value for _history_index, value in samples]
    plot_rect = pygame.Rect(
        chart_rect.left + px(30),
        chart_rect.top + px(27),
        max(1, chart_rect.width - px(40)),
        max(1, chart_rect.height - px(35)),
    )
    if not values or plot_rect.width < 8 or plot_rect.height < 8:
        message = font_meta.render("No Elo measurements yet", True, _SUBTLE)
        screen.blit(message, message.get_rect(center=plot_rect.center))
        return

    low, high = min(values), max(values)
    padding = max(15.0, (high - low) * 0.12)
    low -= padding
    high += padding
    max_history_index = max(history_index for history_index, _value in samples)
    for fraction in (0.0, 0.5, 1.0):
        grid_y = plot_rect.bottom - int(fraction * plot_rect.height)
        pygame.draw.line(screen, (37, 49, 53), (plot_rect.left, grid_y), (plot_rect.right, grid_y), 1)

    plotted_series = [(series["nn"], _ACCENT)]
    mcts_colors = (_WARNING, (186, 155, 111), (148, 133, 107))
    plotted_series.extend(
        (series["mcts"][simulations], mcts_colors[index % len(mcts_colors)])
        for index, simulations in enumerate(mcts_simulations)
    )
    old_clip = screen.get_clip()
    screen.set_clip(old_clip.clip(plot_rect))
    for points, color in plotted_series:
        if not points:
            continue
        coords = []
        for history_index, value in points:
            x = plot_rect.centerx if max_history_index <= 0 else plot_rect.left + int(
                history_index * plot_rect.width / max_history_index
            )
            y = plot_rect.bottom - int((value - low) * plot_rect.height / max(1.0, high - low))
            coords.append((x, y))
        if len(coords) > 1:
            pygame.draw.lines(screen, color, False, coords, 2)
        if coords:
            pygame.draw.circle(screen, color, coords[-1], 3)
    screen.set_clip(old_clip)


def _draw_model_browser(
    screen,
    panel_rect,
    grouped,
    models_dir,
    metadata_cache,
    selected_model,
    expanded,
    scroll_offset,
    font_header,
    font_text,
    font_meta,
    mouse_pos,
    sort_key="model",
    sort_desc=False,
    scale=1.0,
):
    actions = []
    px = lambda value, minimum=1: max(minimum, int(round(value * scale)))

    _draw_panel(screen, panel_rect, fill=(21, 30, 34), border=_BORDER, radius=14, shadow=False)

    pad = px(10)
    header_h = px(30)
    content_y = header_h + px(8)
    visible_keys = [key for key in ("best", "il", "rl", "other") if grouped[key]]

    if not visible_keys:
        empty = font_text.render("No models available in this directory.", True, _MUTED)
        screen.blit(empty, (panel_rect.left + 16, panel_rect.top + 16))
        return actions, 0

    def fmt_version(meta):
        version = meta.get("version")
        if version is None:
            return "n/a"
        version = str(version)
        return version if version.startswith("v") else f"v{version}"

    def fmt_top1(meta):
        top1 = meta.get("top1")
        if top1 is None:
            return "n/a"
        try:
            return f"{float(top1) * 100:.1f}%"
        except Exception:
            return "n/a"

    def fmt_elo(meta):
        summary = format_elo_summary(meta)
        if summary == "n/a":
            return summary
        return (
            summary.replace("NN ", "N ")
            .replace("MCTS ", "M ")
            .replace("Elo ", "")
        )

    header_rect = pygame.Rect(panel_rect.left + pad, panel_rect.top + pad, panel_rect.width - 2 * pad - px(10), header_h)
    pygame.draw.rect(screen, (18, 26, 30), header_rect, border_radius=px(9))
    pygame.draw.rect(screen, (48, 65, 69), header_rect, width=px(1), border_radius=px(9))
    metric_gap = px(10)
    def _visible_model_paths():
        paths = []
        for category_key in visible_keys:
            if category_key == "best" or expanded.get(category_key, False):
                paths.extend(grouped.get(category_key, []))
        return paths

    visible_metas = [metadata_cache.get(path) or {} for path in _visible_model_paths()]

    def _column_width(label, values, min_w, max_w, pad_w=18):
        widest = font_meta.size(str(label))[0]
        for value in values:
            widest = max(widest, font_meta.size(str(value))[0])
        return min(px(max_w), max(px(min_w), widest + px(pad_w)))

    metric_w = {
        "ver": _column_width("Ver", [fmt_version(meta) for meta in visible_metas], 54, 132),
        "top1": _column_width("Top1", [fmt_top1(meta) for meta in visible_metas], 54, 86),
        "elo": _column_width("Elo", [fmt_elo(meta) for meta in visible_metas], 108, 184),
    }
    min_metric_w = {"ver": px(54), "top1": px(54), "elo": px(108)}
    min_model_w = px(28)
    max_metric_total = max(
        sum(min_metric_w.values()),
        header_rect.width - px(20) - min_model_w - 2 * metric_gap,
    )
    overflow = sum(metric_w.values()) - max_metric_total
    for key in ("ver", "elo", "top1"):
        if overflow <= 0:
            break
        shrink = min(overflow, metric_w[key] - min_metric_w[key])
        metric_w[key] -= shrink
        overflow -= shrink
    metrics_right = header_rect.right - px(10)
    elo_left = metrics_right - metric_w["elo"]
    top1_left = elo_left - metric_gap - metric_w["top1"]
    ver_left = top1_left - metric_gap - metric_w["ver"]

    model_col_rect = pygame.Rect(header_rect.left + px(8), header_rect.top + px(4), max(0, ver_left - header_rect.left - px(14)), header_h - px(8))
    ver_col_rect = pygame.Rect(ver_left, header_rect.top + px(4), metric_w["ver"], header_h - px(8))
    top1_col_rect = pygame.Rect(top1_left, header_rect.top + px(4), metric_w["top1"], header_h - px(8))
    elo_col_rect = pygame.Rect(elo_left, header_rect.top + px(4), metric_w["elo"], header_h - px(8))

    def _sort_label(label, key):
        if sort_key != key:
            return label
        return f"{label} {'v' if sort_desc else '^'}"

    headers = [
        (_sort_label("Model", "model"), model_col_rect, "model"),
        ("Ver", ver_col_rect, None),
        (_sort_label("Top1", "top1"), top1_col_rect, "top1"),
        (_sort_label("Elo", "elo"), elo_col_rect, "elo"),
    ]
    for label, col_rect, sortable_key in headers:
        if col_rect.width <= px(8):
            continue
        color = _ACCENT_HOVER if sortable_key and sort_key == sortable_key else _MUTED
        text = font_meta.render(_fit_text(font_meta, label, max(px(8), col_rect.width - px(4))), True, color)
        screen.blit(text, text.get_rect(center=col_rect.center))
        if sortable_key:
            actions.append((col_rect, "sort", sortable_key))
    for sx in (ver_left - 6, top1_left - 6, elo_left - 6):
        if header_rect.left + px(6) < sx < header_rect.right - px(6):
            pygame.draw.line(screen, (48, 62, 66), (sx, header_rect.top + px(5)), (sx, header_rect.bottom - px(5)), px(1))

    # Clip scrolling content so partially-visible rows are cropped cleanly.
    content_clip = pygame.Rect(
        panel_rect.left + pad,
        header_rect.bottom + px(6),
        panel_rect.width - 2 * pad - px(10),
        max(1, panel_rect.bottom - pad - (header_rect.bottom + px(6))),
    )

    def _display_name_for_category(path, category_key):
        return _model_browser_label(path, models_dir, metadata_cache.get(path))

    def _numeric_meta(path, field):
        meta = metadata_cache.get(path) or {}
        value = meta.get(field)
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _sorted_paths(category_key):
        paths = list(grouped.get(category_key, []))
        if not paths:
            return paths

        if sort_key == "model":
            paths.sort(
                key=lambda p: _display_name_for_category(p, category_key).lower(),
                reverse=bool(sort_desc),
            )
            return paths

        if sort_key in ("top1", "elo"):
            fallback = float("-inf") if sort_desc else float("inf")
            metric_field = "top1" if sort_key == "top1" else "elo"
            def metric_or_fallback(path):
                metric_value = _numeric_meta(path, metric_field)
                return metric_value if metric_value is not None else fallback
            paths.sort(
                key=lambda p: (
                    metric_or_fallback(p),
                    _display_name_for_category(p, category_key).lower(),
                ),
                reverse=bool(sort_desc),
            )
            return paths

        return paths

    def draw_header(key):
        nonlocal content_y
        style = _CATEGORY_STYLE[key]
        label = style["label"]
        count = len(grouped[key])
        collapsible = key != "best"
        marker = "v" if expanded.get(key, False) else ">"
        header_text = f"{marker}  {label}" if collapsible else label

        row_h = px(38)
        y = panel_rect.top + pad + content_y - scroll_offset
        rect = pygame.Rect(panel_rect.left + pad, y, panel_rect.width - 2 * pad - px(10), row_h)
        if rect.bottom >= content_clip.top and rect.top <= content_clip.bottom:
            hovered = rect.collidepoint(mouse_pos)
            fill = style["fill"] if not hovered else tuple(min(255, c + 14) for c in style["fill"])
            pygame.draw.rect(screen, fill, rect, border_radius=9)
            pygame.draw.rect(screen, style["border"], rect, width=1, border_radius=9)
            stripe = pygame.Rect(rect.left + px(7), rect.top + px(5), px(7), rect.height - px(10))
            pygame.draw.rect(screen, style["border"], stripe, border_radius=px(2))
            screen.blit(font_header.render(header_text, True, _TEXT), (rect.left + px(20), rect.top + px(7)))

            count_text = str(count)
            chip_w = max(px(28), font_meta.size(count_text)[0] + px(16))
            count_chip = pygame.Rect(rect.right - chip_w - px(8), rect.top + px(6), chip_w, rect.height - px(12))
            pygame.draw.rect(screen, (25, 30, 39), count_chip, border_radius=px(6))
            pygame.draw.rect(screen, style["border"], count_chip, width=px(1), border_radius=px(6))
            count_label = font_meta.render(count_text, True, _TEXT)
            screen.blit(count_label, count_label.get_rect(center=count_chip.center))
        if collapsible and count > 0:
            hit_rect = pygame.Rect(rect.left, rect.top - 2, rect.width, rect.height + 4).clip(content_clip)
            if hit_rect.width > 0 and hit_rect.height > 0:
                actions.append((hit_rect, "toggle", key))
        content_y += row_h + px(8)

    model_row_idx = 0

    def draw_model_row(path, key):
        nonlocal content_y, model_row_idx
        style = _CATEGORY_STYLE.get(key, {"border": _BORDER})
        row_h = px(44)
        y = panel_rect.top + pad + content_y - scroll_offset
        item_indent = px(16)
        rect = pygame.Rect(
            panel_rect.left + pad + item_indent,
            y,
            panel_rect.width - 2 * pad - px(10) - item_indent,
            row_h,
        )

        active = path == selected_model
        if rect.bottom >= content_clip.top and rect.top <= content_clip.bottom:
            hovered = rect.collidepoint(mouse_pos)
            if active:
                fill = (34, 65, 57)
                border = _ACCENT
                text_color = _TEXT
                meta_color = (188, 218, 204)
                accent_bar = _ACCENT
            else:
                zebra_fill = (25, 35, 39) if model_row_idx % 2 == 0 else (23, 32, 36)
                fill = zebra_fill if not hovered else (31, 44, 48)
                border = style["border"] if hovered else (42, 58, 62)
                text_color = _TEXT
                meta_color = _MUTED
                accent_bar = style["border"] if hovered else (66, 83, 86)

            pygame.draw.rect(screen, fill, rect, border_radius=9)
            pygame.draw.rect(screen, border, rect, width=1, border_radius=9)
            left_bar = pygame.Rect(rect.left + px(8), rect.top + px(8), px(4), rect.height - px(16))
            pygame.draw.rect(screen, accent_bar, left_bar, border_radius=px(2))

            meta = metadata_cache.get(path) or {}
            ver = fmt_version(meta)
            top1 = fmt_top1(meta)
            elo = fmt_elo(meta)

            row_metrics_right = rect.right - px(10)
            row_elo_left = row_metrics_right - metric_w["elo"]
            row_top1_left = row_elo_left - metric_gap - metric_w["top1"]
            row_ver_left = row_top1_left - metric_gap - metric_w["ver"]

            sep_color = (77, 112, 102) if active else (43, 59, 62)
            for sx in (row_ver_left - 6, row_top1_left - 6, row_elo_left - 6):
                if rect.left + px(10) < sx < rect.right - px(10):
                    pygame.draw.line(screen, sep_color, (sx, rect.top + px(8)), (sx, rect.bottom - px(8)), px(1))

            name_x = rect.left + px(20)
            name_max_w = max(0, row_ver_left - name_x - px(12))
            raw_name = _display_name_for_category(path, key)
            if name_max_w > 24:
                name = _fit_model_identity(font_text, raw_name, name_max_w)
                screen.blit(font_text.render(name, True, text_color), (name_x, rect.top + px(11)))

            ver_text = font_meta.render(_fit_text(font_meta, ver, max(px(8), metric_w["ver"] - px(6))), True, meta_color)
            top1_text = font_meta.render(_fit_text(font_meta, top1, max(px(8), metric_w["top1"] - px(6))), True, meta_color)
            elo_text = font_meta.render(_fit_text(font_meta, elo, max(px(8), metric_w["elo"] - px(6))), True, meta_color)
            ver_rect = pygame.Rect(row_ver_left, rect.top + px(10), metric_w["ver"], rect.height - px(20))
            top1_rect = pygame.Rect(row_top1_left, rect.top + px(10), metric_w["top1"], rect.height - px(20))
            elo_rect = pygame.Rect(row_elo_left, rect.top + px(10), metric_w["elo"], rect.height - px(20))
            screen.blit(ver_text, ver_text.get_rect(center=ver_rect.center))
            screen.blit(top1_text, top1_text.get_rect(center=top1_rect.center))
            screen.blit(elo_text, elo_text.get_rect(center=elo_rect.center))

            hit_rect = rect.clip(content_clip)
            if hit_rect.width > 0 and hit_rect.height > 0:
                actions.append((hit_rect, "select", path))

        model_row_idx += 1
        content_y += row_h + px(5)

    previous_clip = screen.get_clip()
    screen.set_clip(content_clip)
    for idx, key in enumerate(visible_keys):
        if idx > 0:
            content_y += px(4)
        draw_header(key)
        if key == "best" or expanded.get(key, False):
            for path in _sorted_paths(key):
                draw_model_row(path, key)
            content_y += px(4)
    screen.set_clip(previous_clip)

    content_height = content_y + px(8)
    viewport = content_clip.height
    max_scroll = max(0, content_height - viewport)

    if max_scroll > 0:
        track = pygame.Rect(panel_rect.right - px(12), content_clip.top, px(6), content_clip.height)
        pygame.draw.rect(screen, (35, 47, 50), track, border_radius=px(3))
        thumb_h = max(px(28), int(track.height * (viewport / max(content_height, 1))))
        thumb_h = min(track.height, thumb_h)
        travel = track.height - thumb_h
        ratio = 0.0 if max_scroll <= 0 else (scroll_offset / max_scroll)
        thumb_y = track.top + int(travel * max(0.0, min(1.0, ratio)))
        thumb = pygame.Rect(track.left, thumb_y, track.width, thumb_h)
        pygame.draw.rect(screen, (92, 128, 120), thumb, border_radius=px(3))

    return actions, max_scroll


def _selection_to_result(
    game_mode,
    human_color,
    selected_models,
    opponent_model,
    use_mcts,
    use_mcts_white=None,
    use_mcts_black=None,
    mcts_simulations=None,
    mcts_simulations_white=None,
    mcts_simulations_black=None,
    match_games=1,
    min_visible_move_seconds=1.0,
):
    if use_mcts_white is None:
        use_mcts_white = use_mcts
    if use_mcts_black is None:
        use_mcts_black = use_mcts
    if game_mode == "human_vs_human":
        return {
            "game_mode": game_mode,
            "human_color": human_color,
            "human_color_name": "white" if human_color == chess.WHITE else "black",
            # Keep user preference unchanged; this mode just does not use AI search.
            "use_mcts": bool(use_mcts),
            "use_mcts_white": bool(use_mcts_white),
            "use_mcts_black": bool(use_mcts_black),
            "mcts_simulations": int(mcts_simulations) if mcts_simulations is not None else None,
            "mcts_simulations_white": int(mcts_simulations_white) if mcts_simulations_white is not None else None,
            "mcts_simulations_black": int(mcts_simulations_black) if mcts_simulations_black is not None else None,
            "match_games": int(match_games),
            "min_visible_move_seconds": _visible_move_delay(min_visible_move_seconds),
            "model1_path": None,
            "model2_path": None,
            "model_white": None,
            "model_black": None,
        }

    if game_mode == "human_vs_ai":
        ai_side = "black" if human_color == chess.WHITE else "white"
        model_white = opponent_model if ai_side == "white" else None
        model_black = opponent_model if ai_side == "black" else None
        return {
            "game_mode": game_mode,
            "human_color": human_color,
            "human_color_name": "white" if human_color == chess.WHITE else "black",
            "use_mcts": bool(use_mcts),
            "use_mcts_white": bool(use_mcts_white),
            "use_mcts_black": bool(use_mcts_black),
            "mcts_simulations": int(mcts_simulations) if mcts_simulations is not None else None,
            "mcts_simulations_white": int(mcts_simulations_white) if mcts_simulations_white is not None else None,
            "mcts_simulations_black": int(mcts_simulations_black) if mcts_simulations_black is not None else None,
            "match_games": int(match_games),
            "min_visible_move_seconds": _visible_move_delay(min_visible_move_seconds),
            "model1_path": opponent_model,
            "model2_path": None,
            "model_white": model_white,
            "model_black": model_black,
        }

    return {
        "game_mode": game_mode,
        "human_color": chess.WHITE,
        "human_color_name": "white",
        "use_mcts": bool(use_mcts),
        "use_mcts_white": bool(use_mcts_white),
        "use_mcts_black": bool(use_mcts_black),
        "mcts_simulations": int(mcts_simulations) if mcts_simulations is not None else None,
        "mcts_simulations_white": int(mcts_simulations_white) if mcts_simulations_white is not None else None,
        "mcts_simulations_black": int(mcts_simulations_black) if mcts_simulations_black is not None else None,
        "match_games": int(match_games),
        "min_visible_move_seconds": _visible_move_delay(min_visible_move_seconds),
        "model1_path": selected_models["white"],
        "model2_path": selected_models["black"],
        "model_white": selected_models["white"],
        "model_black": selected_models["black"],
    }


def _can_start(game_mode, has_models, selected_models, opponent_model):
    if game_mode == "human_vs_human":
        return True
    if not has_models:
        return False
    if game_mode == "human_vs_ai":
        return opponent_model is not None
    return selected_models["white"] is not None and selected_models["black"] is not None


def select_models(
    base_dir,
    config,
    default_use_mcts=True,
    initial_window_size=None,
    initial_window_maximized=False,
):
    """Choose game mode/models from an in-window setup UI.

    Returns:
        dict or None: selection payload used by play.py.
    """
    models_dir, all_models = _discover_models(base_dir, config)
    grouped = _group_models(all_models)
    has_models = len(all_models) > 0
    metadata_cache = {}
    progress_cache = {}
    logs_csv_dir = (
        base_dir
        / config.get("paths", {}).get("logs_dir", "logs")
        / "csv"
    )
    for model_path in all_models:
        try:
            metadata = load_checkpoint_metadata(model_path, models_dir)
            metadata["elo_progress"] = _training_elo_progress(
                model_path,
                metadata,
                logs_csv_dir,
                progress_cache,
            )
            metadata_cache[model_path] = metadata
        except Exception:
            metadata_cache[model_path] = {}

    design_width = 1360
    design_height = 900
    base_width = design_width
    base_height = design_height
    canvas_min_width = 720
    canvas_min_height = 540
    min_width = 720
    min_height = 540
    display_flags = pygame.RESIZABLE

    def _get_default_window_size():
        try:
            info = pygame.display.Info()
            screen_width = int(getattr(info, "current_w", 0) or 0)
            screen_height = int(getattr(info, "current_h", 0) or 0)
        except Exception:
            screen_width = 0
            screen_height = 0

        if screen_width <= 0 or screen_height <= 0:
            return design_width, design_height

        width = max(min_width, min(screen_width, int(screen_width * 0.92)))
        height = max(min_height, min(screen_height, int(screen_height * 0.92)))
        return width, height

    default_window_width, default_window_height = _get_default_window_size()
    start_width = default_window_width
    start_height = default_window_height
    if isinstance(initial_window_size, (list, tuple)) and len(initial_window_size) == 2:
        iw = _safe_int(initial_window_size[0], default=default_window_width)
        ih = _safe_int(initial_window_size[1], default=default_window_height)
        start_width = max(min_width, int(iw or default_window_width))
        start_height = max(min_height, int(ih or default_window_height))

    existing_surface = pygame.display.get_surface()
    if existing_surface is not None and existing_surface.get_size() == (start_width, start_height):
        screen = existing_surface
    else:
        screen = pygame.display.set_mode((start_width, start_height), display_flags)
    pygame.display.set_caption("Chess AI Setup")
    try:
        pygame.display.set_window_minimum_size((min_width, min_height))
    except Exception:
        pass

    canvas = pygame.Surface((base_width, base_height))
    background = build_background(base_width, base_height)
    page_background = background
    page_background_height = base_height

    viewport_rect = pygame.Rect(0, 0, base_width, base_height)
    viewport_scale_x = 1.0
    viewport_scale_y = 1.0
    maximized = False
    restore_window_size = (start_width, start_height)
    page_scroll_y = 0
    page_height = base_height
    ui_scale = 1.0
    font_title = None
    font_h2 = None
    font_text = None
    font_meta = None
    font_small = None
    font_tiny = None
    font_card_label = None
    font_card_name = None
    font_card_meta = None

    def refresh_typography():
        nonlocal ui_scale, font_title, font_h2, font_text, font_meta, font_small, font_tiny, font_card_label, font_card_name, font_card_meta
        ui_scale = ui_scale_for_viewport(base_width, base_height, design_width, design_height)
        font_title = pygame.font.SysFont("Segoe UI", max(28, int(round(46 * ui_scale))), bold=True)
        font_h2 = pygame.font.SysFont("Segoe UI", max(18, int(round(24 * ui_scale))), bold=True)
        font_text = pygame.font.SysFont("Segoe UI", max(14, int(round(19 * ui_scale))))
        font_meta = pygame.font.SysFont("Segoe UI", max(12, int(round(15 * ui_scale))))
        font_small = pygame.font.SysFont("Segoe UI", max(13, int(round(17 * ui_scale))))
        font_tiny = pygame.font.SysFont("Segoe UI", max(11, int(round(14 * ui_scale))))
        font_card_label = pygame.font.SysFont("Bahnschrift", max(16, int(round(20 * ui_scale))), bold=True)
        font_card_name = pygame.font.SysFont("Segoe UI", max(15, int(round(21 * ui_scale))), bold=True)
        font_card_meta = pygame.font.SysFont("Segoe UI", max(11, int(round(14 * ui_scale))))

    def resize_canvas(width, height):
        nonlocal base_width, base_height, canvas, background, page_background, page_background_height, page_height, page_scroll_y
        base_width = max(canvas_min_width, int(width))
        base_height = max(canvas_min_height, int(height))
        canvas = pygame.Surface((base_width, base_height))
        background = build_background(base_width, base_height)
        page_background = background
        page_background_height = base_height
        page_height = max(base_height, page_height)
        page_scroll_y = 0
        refresh_typography()

    def get_page_background(height):
        nonlocal page_background, page_background_height
        if height == base_height:
            return background
        if page_background is None or page_background_height != height or page_background.get_width() != base_width:
            page_background = build_background(base_width, height)
            page_background_height = height
        return page_background

    def clamp_page_scroll():
        nonlocal page_scroll_y
        max_scroll = max(0, page_height - base_height)
        page_scroll_y = max(0, min(page_scroll_y, max_scroll))
        return max_scroll

    def update_viewport():
        nonlocal viewport_rect, viewport_scale_x, viewport_scale_y
        win_w, win_h = screen.get_size()
        scale = min(win_w / max(1, base_width), win_h / max(1, base_height))
        viewport_w = max(1, int(round(base_width * scale)))
        viewport_h = max(1, int(round(base_height * scale)))
        viewport_rect = pygame.Rect((win_w - viewport_w) // 2, (win_h - viewport_h) // 2, viewport_w, viewport_h)
        viewport_scale_x = viewport_w / max(1, base_width)
        viewport_scale_y = viewport_h / max(1, base_height)

    def window_to_ui(pos):
        if not viewport_rect.collidepoint(pos):
            return None
        x = int((pos[0] - viewport_rect.left) / max(0.0001, viewport_scale_x))
        y = int((pos[1] - viewport_rect.top) / max(0.0001, viewport_scale_y)) + page_scroll_y
        x = max(0, min(base_width - 1, x))
        y = max(0, min(page_height - 1, y))
        return x, y

    def set_window_size(width, height):
        nonlocal screen, maximized, restore_window_size
        width = max(min_width, int(width))
        height = max(min_height, int(height))
        screen = pygame.display.set_mode((width, height), display_flags)
        restore_window_size = (width, height)
        maximized = False
        pygame.event.pump()
        _position_native_window(width, height)
        resize_canvas(width, height)
        update_viewport()

    def sync_window_surface():
        nonlocal screen
        current_surface = pygame.display.get_surface()
        if current_surface is not None:
            screen = current_surface
        resize_canvas(*screen.get_size())
        update_viewport()

    def set_window_maximized(enabled=True):
        nonlocal screen, maximized, restore_window_size
        if enabled:
            if not maximized:
                restore_window_size = screen.get_size()
            maximized = bool(_maximize_native_window())
            pygame.event.pump()
            sync_window_surface()
        else:
            screen = pygame.display.set_mode(restore_window_size, display_flags)
            pygame.event.pump()
            _position_native_window(*restore_window_size)
            maximized = False
        resize_canvas(*screen.get_size())
        update_viewport()

    resize_canvas(*screen.get_size())
    update_viewport()
    if bool(initial_window_maximized):
        set_window_maximized(True)

    def _with_window_state(payload):
        if payload is None:
            return None
        payload["window_size"] = [int(screen.get_width()), int(screen.get_height())]
        payload["window_maximized"] = bool(maximized)
        payload["window_fullscreen"] = False
        return payload

    clock = pygame.time.Clock()

    saved_preferences = load_play_preferences(base_dir, config)
    saved_game_mode = str(saved_preferences.get("game_mode") or "").strip().lower()
    if saved_game_mode not in ("human_vs_ai", "ai_vs_ai", "human_vs_human"):
        saved_game_mode = "human_vs_ai" if has_models else "human_vs_human"
    if not has_models and saved_game_mode != "human_vs_human":
        saved_game_mode = "human_vs_human"
    game_mode = saved_game_mode

    saved_human_color = str(saved_preferences.get("human_color") or "").strip().lower()
    human_color = chess.BLACK if saved_human_color == "black" else chess.WHITE

    use_mcts = bool(default_use_mcts and has_models)
    if default_use_mcts and isinstance(saved_preferences.get("use_mcts"), bool):
        use_mcts = bool(saved_preferences.get("use_mcts", False) and has_models)
    use_mcts_white = bool(saved_preferences.get("use_mcts_white", use_mcts) and has_models)
    use_mcts_black = bool(saved_preferences.get("use_mcts_black", use_mcts) and has_models)
    workspace_tab = "home"
    picker_side = None
    min_visible_move_seconds = _visible_move_delay(
        saved_preferences.get("min_visible_move_seconds", 1.0)
    )
    mcts_simulations = _safe_int(
        (config or {}).get("reinforcement_learning", {}).get("mcts_simulations"),
        default=100,
    )
    saved_sims = _safe_int(saved_preferences.get("mcts_simulations"), default=None)
    if saved_sims is not None:
        mcts_simulations = saved_sims
    if mcts_simulations is None:
        mcts_simulations = 100
    mcts_simulations = max(16, min(2000, int(mcts_simulations)))
    mcts_simulations_white = _safe_int(saved_preferences.get("mcts_simulations_white"), default=mcts_simulations)
    mcts_simulations_black = _safe_int(saved_preferences.get("mcts_simulations_black"), default=mcts_simulations)
    mcts_simulations_white = max(16, min(2000, int(mcts_simulations_white)))
    mcts_simulations_black = max(16, min(2000, int(mcts_simulations_black)))
    mcts_step = 16
    rl_search_cfg = (config or {}).get("reinforcement_learning", {}) or {}
    mcts_budget_mode = str(rl_search_cfg.get("mcts_budget_mode", "simulations")).strip().lower()
    if mcts_budget_mode not in {"simulations", "time"}:
        mcts_budget_mode = "simulations"
    try:
        mcts_move_time_seconds = max(0.001, float(rl_search_cfg.get("mcts_move_time_seconds", 1.0)))
    except (TypeError, ValueError, OverflowError):
        mcts_move_time_seconds = 1.0
    match_games = _safe_int(saved_preferences.get("match_games"), default=1)
    match_games = max(1, min(500, int(match_games or 1)))
    match_games_step = 2
    default_model = _default_model(
        grouped,
        architecture_id=str((config or {}).get("model", {}).get("architecture_id", "se_cnn_v9")),
        metadata_cache=metadata_cache,
    )
    selected_models = {
        "white": _resolve_saved_model_path(saved_preferences.get("model_white"), models_dir, all_models) or default_model,
        "black": _resolve_saved_model_path(saved_preferences.get("model_black"), models_dir, all_models) or default_model,
    }
    opponent_model = (
        _resolve_saved_model_path(saved_preferences.get("model_ai"), models_dir, all_models)
        or _resolve_saved_model_path(saved_preferences.get("model_black"), models_dir, all_models)
        or _resolve_saved_model_path(saved_preferences.get("model_white"), models_dir, all_models)
        or default_model
    )

    expanded = {
        "white": {"il": False, "rl": False, "other": False},
        "black": {"il": False, "rl": False, "other": False},
        "opponent": {"il": False, "rl": False, "other": False},
    }
    scroll = {"white": 0, "black": 0, "opponent": 0}
    sort_key = "model"
    sort_desc = False
    last_saved_preferences = {
        "use_mcts": bool(use_mcts),
        "use_mcts_white": bool(use_mcts_white),
        "use_mcts_black": bool(use_mcts_black),
        "mcts_simulations": int(mcts_simulations),
        "mcts_simulations_white": int(mcts_simulations_white),
        "mcts_simulations_black": int(mcts_simulations_black),
        "match_games": int(match_games),
        "min_visible_move_seconds": float(min_visible_move_seconds),
    }

    def persist_ui_preferences(force=False):
        nonlocal last_saved_preferences
        current = {
            "game_mode": game_mode,
            "human_color": "white" if human_color == chess.WHITE else "black",
            "use_mcts": bool(use_mcts),
            "use_mcts_white": bool(use_mcts_white),
            "use_mcts_black": bool(use_mcts_black),
            "mcts_simulations": int(max(1, mcts_simulations)),
            "mcts_simulations_white": int(max(1, mcts_simulations_white)),
            "mcts_simulations_black": int(max(1, mcts_simulations_black)),
            "match_games": int(max(1, match_games)),
            "min_visible_move_seconds": float(min_visible_move_seconds),
            "model_ai": str(opponent_model) if opponent_model else None,
            "model_white": str(selected_models["white"]) if selected_models["white"] else None,
            "model_black": str(selected_models["black"]) if selected_models["black"] else None,
        }
        if not force and current == last_saved_preferences:
            return
        try:
            save_play_preferences(base_dir, config, current)
            last_saved_preferences = current
        except Exception:
            pass

    def apply_sort(selected_key):
        nonlocal sort_key, sort_desc
        if selected_key not in ("model", "top1", "elo"):
            return
        if sort_key == selected_key:
            sort_desc = not sort_desc
        else:
            sort_key = selected_key
            sort_desc = selected_key in ("top1", "elo")

    font_cache = {}
    overlay_cache = {}
    while True:
        mouse_pos = window_to_ui(pygame.mouse.get_pos()) or (-9999, -9999)
        scale = ui_scale_for_viewport(base_width, base_height, design_width, design_height)
        layout = build_home_layout(base_width, base_height, scale, game_mode)
        scale_key = round(scale, 3)
        if scale_key not in font_cache:
            font_cache[scale_key] = {
                "title": pygame.font.SysFont("Segoe UI", max(28, int(43 * scale)), bold=True),
                "h2": pygame.font.SysFont("Segoe UI", max(18, int(24 * scale)), bold=True),
                "text": pygame.font.SysFont("Segoe UI", max(14, int(18 * scale))),
                "meta": pygame.font.SysFont("Segoe UI", max(12, int(15 * scale))),
                "small": pygame.font.SysFont("Segoe UI", max(13, int(16 * scale))),
                "card_label": pygame.font.SysFont("Bahnschrift", max(15, int(18 * scale)), bold=True),
                "card_name": pygame.font.SysFont("Segoe UI", max(15, int(20 * scale)), bold=True),
                "card_meta": pygame.font.SysFont("Segoe UI", max(11, int(13 * scale))),
            }
        fonts = font_cache[scale_key]
        px = lambda value, minimum=1: max(minimum, int(round(value * scale)))
        mouse_inside = mouse_pos[0] >= 0
        picker_layout = None
        picker_actions = []
        picker_max_scroll = 0
        settings_panel = None
        settings_close = None
        settings_games = []
        settings_paces = []
        settings_budgets = []
        canvas.blit(background, (0, 0))

        # App header and mode navigation.
        _draw_panel(canvas, layout.header, fill=(19, 27, 32), border=(44, 60, 66), radius=px(14), shadow=False)
        brand_rect = pygame.Rect(layout.header.left + px(18), layout.header.top + px(14), px(52), layout.header.height - px(28))
        pygame.draw.rect(canvas, (34, 77, 67), brand_rect, border_radius=px(11))
        brand = fonts["card_label"].render("C", True, _ACCENT)
        canvas.blit(brand, brand.get_rect(center=brand_rect.center))
        title_x = brand_rect.right + px(14)
        canvas.blit(fonts["h2"].render("CHESS  /  PLAY", True, _TEXT), (title_x, layout.header.top + px(14)))
        canvas.blit(fonts["meta"].render("Build a match from your models", True, _MUTED), (title_x, layout.header.top + px(44)))
        _draw_button(canvas, layout.options_button, "Settings", fonts["small"], hovered=mouse_inside and layout.options_button.collidepoint(mouse_pos), active=workspace_tab == "settings")

        mode_names = {
            "human_vs_ai": ("Play vs AI", "One model against you"),
            "ai_vs_ai": ("Model arena", "Compare two models"),
            "human_vs_human": ("Local game", "Two players, one board"),
        }
        for mode, rect in layout.mode_buttons.items():
            label, hint = mode_names[mode]
            active = game_mode == mode
            _draw_panel(canvas, rect, fill=(35, 67, 59) if active else (23, 32, 37), border=_ACCENT if active else (44, 59, 64), radius=px(11), shadow=False)
            canvas.blit(fonts["card_label"].render(label, True, _ACCENT if active else _TEXT), (rect.left + px(16), rect.top + px(8)))
            canvas.blit(fonts["card_meta"].render(hint, True, _MUTED), (rect.left + px(16), rect.top + px(33)))

        # Focused player composer. Its heading is fully inside the workspace card.
        _draw_panel(canvas, layout.workspace, fill=(22, 31, 36), border=(47, 63, 68), radius=px(15), shadow=False)
        section = "PLAYER ASSIGNMENT" if game_mode == "human_vs_human" else "MODEL ASSIGNMENT"
        canvas.blit(fonts["meta"].render(section, True, _ACCENT), layout.workspace_title.topleft)
        description = {
            "human_vs_ai": "Choose your side and the model you want to face.",
            "ai_vs_ai": "Choose a model and search profile for each side.",
            "human_vs_human": "Set up a local game on this board.",
        }[game_mode]
        canvas.blit(fonts["card_meta"].render(_fit_text(fonts["card_meta"], description, layout.workspace_title.width), True, _MUTED), (layout.workspace_title.left, layout.workspace_title.top + px(20)))

        white_human = game_mode == "human_vs_human" or (game_mode == "human_vs_ai" and human_color == chess.WHITE)
        black_human = game_mode == "human_vs_human" or (game_mode == "human_vs_ai" and human_color == chess.BLACK)
        white_model = None if white_human else (opponent_model if game_mode == "human_vs_ai" else selected_models["white"])
        black_model = None if black_human else (opponent_model if game_mode == "human_vs_ai" else selected_models["black"])
        if game_mode == "human_vs_ai":
            use_mcts = bool(use_mcts_black if human_color == chess.WHITE else use_mcts_white)
        elif game_mode == "ai_vs_ai":
            use_mcts = bool(use_mcts_white and use_mcts_black)
        white_controls = None if white_human else {"mcts": layout.white_mcts_button, "minus": layout.white_budget_minus, "value": layout.white_budget_value, "plus": layout.white_budget_plus}
        black_controls = None if black_human else {"mcts": layout.black_mcts_button, "minus": layout.black_budget_minus, "value": layout.black_budget_value, "plus": layout.black_budget_plus}
        _draw_home_player_card(
            canvas, layout.white_card, layout.white_model_button, "White", white_model, models_dir, metadata_cache,
            fonts, scale, chess.WHITE, use_mcts=use_mcts_white and not white_human,
            simulations=mcts_simulations_white,
            budget_mode=mcts_budget_mode,
            move_time_seconds=mcts_move_time_seconds,
            mcts_rect=white_controls["mcts"] if white_controls else None,
            budget_minus_rect=white_controls["minus"] if white_controls else None,
            budget_value_rect=white_controls["value"] if white_controls else None,
            budget_plus_rect=white_controls["plus"] if white_controls else None,
            is_human=white_human, hovered=mouse_inside and layout.white_card.collidepoint(mouse_pos), mouse_pos=mouse_pos,
        )
        _draw_home_player_card(
            canvas, layout.black_card, layout.black_model_button, "Black", black_model, models_dir, metadata_cache,
            fonts, scale, chess.BLACK, use_mcts=use_mcts_black and not black_human,
            simulations=mcts_simulations_black,
            budget_mode=mcts_budget_mode,
            move_time_seconds=mcts_move_time_seconds,
            mcts_rect=black_controls["mcts"] if black_controls else None,
            budget_minus_rect=black_controls["minus"] if black_controls else None,
            budget_value_rect=black_controls["value"] if black_controls else None,
            budget_plus_rect=black_controls["plus"] if black_controls else None,
            is_human=black_human, hovered=mouse_inside and layout.black_card.collidepoint(mouse_pos), mouse_pos=mouse_pos,
        )
        _draw_panel(canvas, layout.versus, fill=(18, 26, 30), border=(48, 64, 68), radius=px(16), shadow=False)
        versus = fonts["card_label"].render("VS", True, _WARNING)
        canvas.blit(versus, versus.get_rect(center=layout.versus.center))

        actions = {
            "ai_vs_ai": ("Swap sides", "Use White for both", "Use Black for both"),
            "human_vs_ai": ("Switch sides", "Play as White", "Play as Black"),
            "human_vs_human": ("Local play", "White moves first", "Ready to play"),
        }[game_mode]
        for index, (rect, label) in enumerate(zip((layout.swap_button, layout.copy_white_button, layout.copy_black_button), actions)):
            disabled = game_mode == "human_vs_human"
            active = game_mode == "human_vs_ai" and ((index == 1 and human_color == chess.WHITE) or (index == 2 and human_color == chess.BLACK))
            _draw_button(canvas, rect, label, fonts["small"], hovered=not disabled and mouse_inside and rect.collidepoint(mouse_pos), disabled=disabled, active=active)

        # Persistent match controls and start action.
        _draw_panel(canvas, layout.runtime, fill=(19, 27, 32), border=(47, 63, 68), radius=px(14), shadow=False)
        canvas.blit(fonts["card_meta"].render("MATCH LENGTH" if game_mode == "ai_vs_ai" else "GAMES", True, _MUTED), layout.match_label.topleft)
        _draw_button(canvas, layout.match_minus, "-", fonts["h2"], hovered=mouse_inside and layout.match_minus.collidepoint(mouse_pos), disabled=match_games <= 1)
        _draw_panel(canvas, layout.match_value, fill=(24, 34, 39), border=(48, 64, 68), radius=px(8), shadow=False)
        game_count = fonts["small"].render(str(match_games), True, _TEXT)
        canvas.blit(game_count, game_count.get_rect(center=layout.match_value.center))
        _draw_button(canvas, layout.match_plus, "+", fonts["h2"], hovered=mouse_inside and layout.match_plus.collidepoint(mouse_pos), disabled=match_games >= 500)

        pace_enabled = game_mode != "human_vs_human"
        canvas.blit(fonts["card_meta"].render("MINIMUM VISIBLE AI MOVE", True, _MUTED), layout.pace_label.topleft)
        for value, rect in layout.pace_buttons:
            label = "No wait" if value == 0 else f"{value:g}s"
            _draw_button(canvas, rect, label, fonts["small"], hovered=pace_enabled and mouse_inside and rect.collidepoint(mouse_pos), active=abs(min_visible_move_seconds - value) < 0.01, disabled=not pace_enabled)
        note = "Visible games only; batch evaluation is unaffected." if pace_enabled else "No AI moves in local mode."
        note_surface = fonts["card_meta"].render(_fit_text(fonts["card_meta"], note, layout.pace_label.width), True, _SUBTLE)
        canvas.blit(note_surface, (layout.pace_label.left, layout.pace_buttons[0][1].bottom + px(6)))
        can_start = _can_start(game_mode, has_models, selected_models, opponent_model)
        _draw_button(canvas, layout.start_button, "Start match", fonts["card_label"], hovered=can_start and mouse_inside and layout.start_button.collidepoint(mouse_pos), active=can_start, disabled=not can_start)
        _draw_button(canvas, layout.cancel_button, "Back", fonts["small"], hovered=mouse_inside and layout.cancel_button.collidepoint(mouse_pos))

        if workspace_tab == "picker" and picker_side is not None:
            picker_layout = build_picker_layout(base_width, base_height, scale)
            overlay = overlay_cache.get((base_width, base_height))
            if overlay is None:
                overlay = pygame.Surface((base_width, base_height), pygame.SRCALPHA)
                overlay.fill((5, 9, 12, 190))
                overlay_cache[(base_width, base_height)] = overlay
            canvas.blit(overlay, (0, 0))
            _draw_panel(canvas, picker_layout.panel, fill=(23, 32, 37), border=(72, 98, 100), radius=px(16), shadow=True)
            canvas.blit(fonts["h2"].render("Choose a model", True, _TEXT), picker_layout.title.topleft)
            side_name = {"white": "White", "black": "Black", "opponent": "AI opponent"}.get(picker_side, "AI")
            subtitle = fonts["meta"].render(f"Selecting for {side_name} · select a checkpoint to assign it", True, _MUTED)
            canvas.blit(subtitle, picker_layout.subtitle.topleft)
            _draw_button(canvas, picker_layout.close_button, "X", fonts["h2"], hovered=mouse_inside and picker_layout.close_button.collidepoint(mouse_pos))
            selected = selected_models.get(picker_side) if game_mode == "ai_vs_ai" else opponent_model
            key = picker_side if picker_side in expanded else "opponent"
            picker_actions, picker_max_scroll = _draw_model_browser(
                canvas, picker_layout.browser, grouped, models_dir, metadata_cache, selected,
                expanded[key], scroll[key], fonts["card_label"], fonts["small"], fonts["card_meta"], mouse_pos,
                sort_key=sort_key, sort_desc=sort_desc, scale=scale,
            )
            _draw_model_info_panel(canvas, picker_layout.details, selected, models_dir, metadata_cache, fonts["h2"], fonts["small"], fonts["card_meta"], scale=scale)
        elif workspace_tab == "settings":
            modal_w = min(base_width - px(40), px(900))
            modal_h = min(base_height - px(40), px(610))
            settings_panel = pygame.Rect((base_width - modal_w) // 2, (base_height - modal_h) // 2, modal_w, modal_h)
            overlay = overlay_cache.get((base_width, base_height))
            if overlay is None:
                overlay = pygame.Surface((base_width, base_height), pygame.SRCALPHA)
                overlay.fill((5, 9, 12, 190))
                overlay_cache[(base_width, base_height)] = overlay
            canvas.blit(overlay, (0, 0))
            _draw_panel(canvas, settings_panel, fill=(23, 32, 37), border=(72, 98, 100), radius=px(16), shadow=True)
            canvas.blit(fonts["h2"].render("Match settings", True, _TEXT), (settings_panel.left + px(28), settings_panel.top + px(22)))
            settings_close = pygame.Rect(settings_panel.right - px(58), settings_panel.top + px(18), px(36), px(36))
            _draw_button(canvas, settings_close, "X", fonts["h2"], hovered=mouse_inside and settings_close.collidepoint(mouse_pos))
            canvas.blit(fonts["meta"].render("Choose presets here, or adjust each side directly on the home screen.", True, _MUTED), (settings_panel.left + px(28), settings_panel.top + px(63)))
            gap = px(10)
            col_w = (settings_panel.width - 2 * px(56) - 3 * gap) // 4
            y = settings_panel.top + px(120)
            canvas.blit(fonts["card_label"].render("MATCH LENGTH", True, _ACCENT), (settings_panel.left + px(28), y))
            y += px(34)
            for index, value in enumerate((1, 10, 50, 100)):
                rect = pygame.Rect(settings_panel.left + px(28) + index * (col_w + gap), y, col_w, px(44))
                settings_games.append((value, rect))
                label = "1 game" if value == 1 else f"{value} games"
                _draw_button(canvas, rect, label, fonts["small"], hovered=mouse_inside and rect.collidepoint(mouse_pos), active=match_games == value)
            y += px(82)
            canvas.blit(fonts["card_label"].render("VISIBLE MOVE PACE", True, _ACCENT), (settings_panel.left + px(28), y))
            y += px(34)
            for index, (value, label) in enumerate(((0.0, "No wait"), (0.2, "0.2 sec"), (0.6, "0.6 sec"), (1.0, "1 sec"))):
                rect = pygame.Rect(settings_panel.left + px(28) + index * (col_w + gap), y, col_w, px(44))
                settings_paces.append((value, rect))
                _draw_button(canvas, rect, label, fonts["small"], hovered=pace_enabled and mouse_inside and rect.collidepoint(mouse_pos), active=abs(min_visible_move_seconds - value) < 0.01, disabled=not pace_enabled)
            if pace_enabled and mcts_budget_mode != "time":
                y += px(82)
                canvas.blit(fonts["card_label"].render("SEARCH BUDGET PRESETS", True, _ACCENT), (settings_panel.left + px(28), y))
                y += px(34)
                for index, (value, label) in enumerate(((64, "Fast · 64"), (100, "Balanced · 100"), (200, "Strong · 200"), (400, "Deep · 400"))):
                    rect = pygame.Rect(settings_panel.left + px(28) + index * (col_w + gap), y, col_w, px(44))
                    settings_budgets.append((value, rect))
                    active = mcts_simulations_white == value and mcts_simulations_black == value
                    _draw_button(canvas, rect, label, fonts["small"], hovered=mouse_inside and rect.collidepoint(mouse_pos), active=active)
            note = fonts["card_meta"].render("MCTS off uses the neural-network policy; its search budget is inactive.", True, _SUBTLE)
            canvas.blit(note, (settings_panel.left + px(28), settings_panel.bottom - px(42)))

        if screen.get_size() == canvas.get_size():
            screen.blit(canvas, (0, 0))
        else:
            screen.fill((0, 0, 0))
            screen.blit(pygame.transform.smoothscale(canvas, viewport_rect.size), viewport_rect.topleft)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return _with_window_state(None)
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(min_width, event.w), max(min_height, event.h)), display_flags)
                resize_canvas(*screen.get_size())
                update_viewport()
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if workspace_tab in ("picker", "settings"):
                        workspace_tab, picker_side = "home", None
                    else:
                        return _with_window_state(None)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and workspace_tab == "home" and can_start:
                    use_mcts = bool(use_mcts_white and use_mcts_black)
                    result = _selection_to_result(game_mode, human_color, selected_models, opponent_model, use_mcts,
                        use_mcts_white=use_mcts_white, use_mcts_black=use_mcts_black,
                        mcts_simulations=mcts_simulations, mcts_simulations_white=mcts_simulations_white,
                        mcts_simulations_black=mcts_simulations_black, match_games=match_games,
                        min_visible_move_seconds=min_visible_move_seconds)
                    persist_ui_preferences()
                    return _with_window_state(result)
                continue
            if event.type == pygame.MOUSEWHEEL and workspace_tab == "picker" and picker_layout and picker_layout.browser.collidepoint(mouse_pos):
                key = picker_side if picker_side in scroll else "opponent"
                scroll[key] = max(0, min(picker_max_scroll, scroll[key] - event.y * px(54)))
                continue
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = window_to_ui(event.pos)
                if click is None:
                    continue
                if event.button in (4, 5) and workspace_tab == "picker" and picker_layout and picker_layout.browser.collidepoint(click):
                    key = picker_side if picker_side in scroll else "opponent"
                    scroll[key] = max(0, min(picker_max_scroll, scroll[key] + (px(54) if event.button == 5 else -px(54))))
                    continue
                if event.button != 1:
                    continue
                if workspace_tab == "picker":
                    if not picker_layout or picker_layout.close_button.collidepoint(click) or not picker_layout.panel.collidepoint(click):
                        workspace_tab, picker_side = "home", None
                        continue
                    key = picker_side if picker_side in expanded else "opponent"
                    for rect, action, value in picker_actions:
                        if rect.collidepoint(click):
                            if action == "toggle":
                                expanded[key][value] = not expanded[key].get(value, False)
                                scroll[key] = 0
                            elif action == "sort":
                                apply_sort(value)
                            elif action == "select":
                                if game_mode == "ai_vs_ai":
                                    selected_models[picker_side] = value
                                    if picker_side == "white":
                                        opponent_model = value
                                else:
                                    opponent_model = value
                                workspace_tab, picker_side = "home", None
                                persist_ui_preferences()
                            break
                    continue
                if workspace_tab == "settings":
                    if not settings_panel or settings_close.collidepoint(click) or not settings_panel.collidepoint(click):
                        workspace_tab = "home"
                        continue
                    changed = False
                    for value, rect in settings_games:
                        if rect.collidepoint(click):
                            match_games, changed = value, True
                            break
                    if not changed and pace_enabled:
                        for value, rect in settings_paces:
                            if rect.collidepoint(click):
                                min_visible_move_seconds, changed = _visible_move_delay(value), True
                                break
                    if not changed and mcts_budget_mode != "time":
                        for value, rect in settings_budgets:
                            if rect.collidepoint(click):
                                mcts_simulations = mcts_simulations_white = mcts_simulations_black = value
                                changed = True
                                break
                    if changed:
                        persist_ui_preferences()
                    continue

                if layout.options_button.collidepoint(click):
                    workspace_tab = "settings"
                    continue
                chosen_mode = next((mode for mode, rect in layout.mode_buttons.items() if rect.collidepoint(click)), None)
                if chosen_mode:
                    if chosen_mode != "human_vs_ai" or has_models:
                        game_mode = chosen_mode
                        persist_ui_preferences()
                    continue
                if layout.start_button.collidepoint(click) and can_start:
                    use_mcts = bool(use_mcts_white and use_mcts_black)
                    result = _selection_to_result(game_mode, human_color, selected_models, opponent_model, use_mcts,
                        use_mcts_white=use_mcts_white, use_mcts_black=use_mcts_black,
                        mcts_simulations=mcts_simulations, mcts_simulations_white=mcts_simulations_white,
                        mcts_simulations_black=mcts_simulations_black, match_games=match_games,
                        min_visible_move_seconds=min_visible_move_seconds)
                    persist_ui_preferences()
                    return _with_window_state(result)
                if layout.cancel_button.collidepoint(click):
                    return _with_window_state(None)
                if layout.match_minus.collidepoint(click):
                    match_games = max(1, match_games - match_games_step)
                    persist_ui_preferences()
                    continue
                if layout.match_plus.collidepoint(click):
                    match_games = min(500, match_games + match_games_step)
                    persist_ui_preferences()
                    continue
                pace_value = next((value for value, rect in layout.pace_buttons if rect.collidepoint(click)), None)
                if pace_value is not None and pace_enabled:
                    min_visible_move_seconds = _visible_move_delay(pace_value)
                    persist_ui_preferences()
                    continue
                if layout.swap_button.collidepoint(click) and game_mode != "human_vs_human":
                    if game_mode == "human_vs_ai":
                        human_color = chess.BLACK if human_color == chess.WHITE else chess.WHITE
                    else:
                        selected_models["white"], selected_models["black"] = selected_models["black"], selected_models["white"]
                    persist_ui_preferences()
                    continue
                if layout.copy_white_button.collidepoint(click) and game_mode != "human_vs_human":
                    if game_mode == "human_vs_ai": human_color = chess.WHITE
                    else: selected_models["black"] = selected_models["white"]; opponent_model = selected_models["white"]
                    persist_ui_preferences()
                    continue
                if layout.copy_black_button.collidepoint(click) and game_mode != "human_vs_human":
                    if game_mode == "human_vs_ai": human_color = chess.BLACK
                    else: selected_models["white"] = selected_models["black"]; opponent_model = selected_models["black"]
                    persist_ui_preferences()
                    continue
                if game_mode == "ai_vs_ai":
                    if layout.white_model_button.collidepoint(click):
                        picker_side, workspace_tab = "white", "picker"
                        continue
                    if layout.black_model_button.collidepoint(click):
                        picker_side, workspace_tab = "black", "picker"
                        continue
                elif game_mode == "human_vs_ai":
                    ai_side = "black" if human_color == chess.WHITE else "white"
                    model_rect = layout.black_model_button if ai_side == "black" else layout.white_model_button
                    if model_rect.collidepoint(click):
                        picker_side, workspace_tab = "opponent", "picker"
                        continue
                if white_controls:
                    if white_controls["mcts"].collidepoint(click): use_mcts_white = not use_mcts_white
                    elif white_controls["minus"].collidepoint(click) and use_mcts_white and mcts_budget_mode != "time": mcts_simulations_white = max(16, mcts_simulations_white - mcts_step)
                    elif white_controls["plus"].collidepoint(click) and use_mcts_white and mcts_budget_mode != "time": mcts_simulations_white = min(2000, mcts_simulations_white + mcts_step)
                    else: white_controls = None
                    if white_controls:
                        use_mcts = bool(use_mcts_black if game_mode == "human_vs_ai" and human_color == chess.WHITE else use_mcts_white if game_mode == "human_vs_ai" else use_mcts_white and use_mcts_black)
                        persist_ui_preferences()
                        continue
                if black_controls:
                    if black_controls["mcts"].collidepoint(click): use_mcts_black = not use_mcts_black
                    elif black_controls["minus"].collidepoint(click) and use_mcts_black and mcts_budget_mode != "time": mcts_simulations_black = max(16, mcts_simulations_black - mcts_step)
                    elif black_controls["plus"].collidepoint(click) and use_mcts_black and mcts_budget_mode != "time": mcts_simulations_black = min(2000, mcts_simulations_black + mcts_step)
                    else: black_controls = None
                    if black_controls:
                        use_mcts = bool(use_mcts_black if game_mode == "human_vs_ai" and human_color == chess.WHITE else use_mcts_white if game_mode == "human_vs_ai" else use_mcts_white and use_mcts_black)
                        persist_ui_preferences()
                        continue

        clock.tick(60)
