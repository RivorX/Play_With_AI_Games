"""Model loading and pre-game setup helpers for local GUI play."""

import copy
import ctypes
from datetime import datetime
import os
from pathlib import Path
import sys

import chess
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import torch
import yaml

# Add utils to path for model_catalog import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from utils.shared.model_catalog import format_elo_stat_parts, format_elo_summary, load_checkpoint_metadata
from utils.ui.gui_helpers import create_piece_surfaces, start_piece_asset_prefetch
from utils.ui.theme import (
    ACCENT as _ACCENT,
    ACCENT_HOVER as _ACCENT_BORDER,
    BORDER as _BORDER,
    DANGER as _DANGER,
    MUTED as _MUTED,
    SURFACE as _PANEL_BG,
    TEXT as _TEXT,
    build_background,
    draw_card,
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
    "best": {"label": "BEST models", "fill": (40, 82, 67), "border": (112, 199, 155)},
    "il": {"label": "IL models", "fill": (39, 64, 102), "border": (122, 168, 244)},
    "rl": {"label": "RL models", "fill": (95, 69, 38), "border": (214, 162, 93)},
    "other": {"label": "Other models", "fill": (63, 63, 82), "border": (152, 156, 181)},
}
_SETUP_PIECE_SURFACES = {}


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
    if fill_alpha is None:
        draw_card(screen, rect, fill=fill, border=border, radius=radius, shadow=shadow)
    else:
        if shadow:
            shadow_rect = rect.move(0, 6)
            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, (0, 0, 0, 48), shadow_surface.get_rect(), border_radius=radius)
            screen.blit(shadow_surface, shadow_rect.topleft)
        fill_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(fill_surface, (*fill, int(max(0, min(255, fill_alpha)))), fill_surface.get_rect(), border_radius=radius)
        screen.blit(fill_surface, rect.topleft)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)


def _get_setup_piece_surfaces(square_size):
    cache_key = max(1, int(square_size))
    cached = _SETUP_PIECE_SURFACES.get(cache_key)
    if cached is None:
        cached = create_piece_surfaces(cache_key)
        _SETUP_PIECE_SURFACES[cache_key] = cached
    return cached


def _scaled_setup_piece_icon(symbol, target_size, fill_ratio=0.82):
    surface = _get_setup_piece_surfaces(target_size).get(symbol)
    if surface is None:
        return None
    bbox = surface.get_bounding_rect(min_alpha=1)
    if bbox.width <= 0 or bbox.height <= 0:
        return pygame.transform.smoothscale(surface, (target_size, target_size))
    cropped = surface.subsurface(bbox).copy()
    target_inner = max(1, int(target_size * fill_ratio))
    scale = min(target_inner / cropped.get_width(), target_inner / cropped.get_height())
    new_w = max(1, int(cropped.get_width() * scale))
    new_h = max(1, int(cropped.get_height() * scale))
    scaled = pygame.transform.smoothscale(cropped, (new_w, new_h))
    canvas = pygame.Surface((target_size, target_size), pygame.SRCALPHA)
    canvas.blit(scaled, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


_MODEL_ARCH_KEYS = [
    "version",
    "filters",
    "num_residual_blocks",
    "dropout",
    "history_positions",
    "use_se_blocks",
    "use_se_bottleneck",
    "se_reduction",
    "drop_path_rate",
    "use_coord_conv",
    "use_layer_scale",
    "layer_scale_init",
    "policy_head_channels",
    "policy_head_conv_filters",
    "policy_head_conv_groups",
    "policy_head_global_dim",
    "policy_head_hidden_dim",
    "value_head_filters",
    "value_hidden_dim",
    "moves_left_hidden_dim",
]


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _build_checkpoint_model_config(config, checkpoint):
    model_cfg = dict((config or {}).get("model", {}) or {})
    checkpoint_arch = checkpoint.get("model_architecture")
    if isinstance(checkpoint_arch, dict):
        for key in _MODEL_ARCH_KEYS:
            if key in checkpoint_arch:
                model_cfg[key] = checkpoint_arch[key]
        if "use_se_blocks" not in model_cfg and "use_se2d_blocks" in checkpoint_arch:
            model_cfg["use_se_blocks"] = checkpoint_arch["use_se2d_blocks"]

    for key in _MODEL_ARCH_KEYS:
        if key in checkpoint:
            model_cfg[key] = checkpoint[key]
    if "use_se_blocks" not in model_cfg and "use_se2d_blocks" in checkpoint:
        model_cfg["use_se_blocks"] = checkpoint["use_se2d_blocks"]

    inferred = _infer_architecture_from_state_dict(checkpoint.get("model_state_dict"))
    for key, value in inferred.items():
        model_cfg[key] = value

    return model_cfg


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


def load_model_from_checkpoint(checkpoint_path, config, device, model_class, verbose=False):
    """Load a model checkpoint and switch model to eval mode."""
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing a valid model_state_dict."
            )

        model_config = _build_checkpoint_model_config(config, checkpoint)
        runtime_config = copy.deepcopy(config)
        runtime_config.setdefault("model", {})
        runtime_config["model"].update(model_config)
        runtime_config["model"]["print_summary"] = bool(verbose)

        model = model_class(runtime_config).to(device)
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
            print(
                f"  Arch: blocks={runtime_config['model'].get('num_residual_blocks')}, "
                f"filters={runtime_config['model'].get('filters')}, "
                f"history={runtime_config['model'].get('history_positions')}, "
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
    all_models.extend(models_dir.glob("IL/*.pt"))
    all_models.extend(models_dir.glob("RL/*.pt"))
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


def _default_model(grouped):
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
        fill = (52, 60, 73)
        text_color = (138, 149, 167)
        border = (82, 94, 114)
    elif danger:
        fill = _DANGER if active else (152, 73, 73)
        text_color = _TEXT
        border = (224, 136, 136)
    elif active:
        fill = _ACCENT
        text_color = (247, 250, 255)
        border = _ACCENT_BORDER
    elif hovered:
        fill = (45, 56, 72)
        text_color = _TEXT
        border = (126, 149, 183)
    else:
        fill = (32, 41, 55)
        text_color = _TEXT
        border = _BORDER

    shadow = rect.move(0, 2)
    pygame.draw.rect(screen, (7, 10, 16), shadow, border_radius=10)
    pygame.draw.rect(screen, fill, rect, border_radius=10)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
    label_text = _fit_text(font, text, max(24, rect.width - 16))
    label = font.render(label_text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_selection_card(
    screen,
    rect,
    title,
    model_path,
    models_dir,
    font_title,
    font_text,
    font_meta,
    metadata_cache=None,
    active=False,
    hovered=False,
    side_color=None,
):
    fill = (24, 32, 44)
    border = _BORDER
    side_accent = (210, 220, 234) if side_color == chess.WHITE else (113, 139, 178)
    if hovered:
        fill = _lerp_color(fill, (255, 255, 255), 0.05)
        border = _lerp_color(border, _ACCENT_BORDER, 0.45)
    if active:
        border = _ACCENT_BORDER
        fill = _lerp_color(fill, _ACCENT, 0.13)
        glow_rect = rect.inflate(10, 10)
        glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*_ACCENT, 38), glow_surface.get_rect(), border_radius=18)
        screen.blit(glow_surface, glow_rect.topleft)

    _draw_panel(screen, rect, fill=fill, border=border, radius=14, shadow=False)
    if active:
        pygame.draw.rect(screen, _ACCENT_BORDER, rect, width=2, border_radius=14)
        pygame.draw.rect(screen, _ACCENT, pygame.Rect(rect.left, rect.top + 14, 3, rect.height - 28), border_radius=2)

    label_surface = font_title.render(title.replace(" model", "").upper(), True, side_accent)
    screen.blit(label_surface, (rect.left + 14, rect.top + 12))
    if active:
        badge_text = font_meta.render("ACTIVE", True, _TEXT)
        badge_rect = pygame.Rect(0, 0, badge_text.get_width() + 18, max(22, badge_text.get_height() + 8))
        badge_rect.topright = (rect.right - 12, rect.top + 9)
        pygame.draw.rect(screen, _ACCENT, badge_rect, border_radius=badge_rect.height // 2)
        screen.blit(badge_text, badge_text.get_rect(center=badge_rect.center))

    if model_path is None:
        line1 = "No model selected"
        line2 = "Choose one from the browser below"
    else:
        metadata = (metadata_cache or {}).get(model_path)
        line1, line2 = _format_model_entry(
            model_path,
            models_dir,
            max_len=56,
            metadata=metadata,
        )

    content_top = rect.top + max(40, font_title.get_height() + 24)
    icon_size = max(48, min(58, rect.bottom - content_top - 12))
    icon_rect = pygame.Rect(rect.left + 12, content_top, icon_size, icon_size)
    if side_color in (chess.WHITE, chess.BLACK):
        symbol = "K" if side_color == chess.WHITE else "k"
        piece_surface = _scaled_setup_piece_icon(symbol, icon_size, fill_ratio=0.86)
        if piece_surface is not None:
            screen.blit(piece_surface, piece_surface.get_rect(center=icon_rect.center))

    text_left = icon_rect.right + 12 if side_color in (chess.WHITE, chess.BLACK) else rect.left + 14
    text_width = max(40, rect.right - text_left - 14)
    line1_y = content_top + 3
    line2_y = line1_y + font_text.get_height() + 7
    screen.blit(font_text.render(_fit_text(font_text, line1, text_width), True, _TEXT), (text_left, line1_y))
    screen.blit(font_meta.render(_fit_text(font_meta, line2, text_width), True, _MUTED), (text_left, line2_y))


def _selection_card_height(font_label, font_text, font_meta):
    """Resolve selection-card height from actual typography metrics."""
    header_h = max(40, font_label.get_height() + 24)
    body_h = max(70, font_text.get_height() + font_meta.get_height() + 22)
    return header_h + body_h


def _fit_text(font, text, max_width):
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed and font.size(trimmed + suffix)[0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + suffix) if trimmed else suffix


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


def _model_info_panel_height(model_path, models_dir, metadata_cache, font_h2, font_text, font_meta):
    if model_path is None:
        return 110
    _, _, _, rows, _ = _get_model_info_content(model_path, models_dir, metadata_cache)
    row_h = max(24, font_meta.get_height() + 10)
    header_h = 52 + font_text.get_height() + 14
    stat_block_h = 52 + 16
    table_height = 8 + len(rows) * row_h + 8
    return header_h + stat_block_h + table_height + 12


def _draw_model_info_panel(
    screen,
    rect,
    model_path,
    models_dir,
    metadata_cache,
    font_h2,
    font_text,
    font_meta,
):
    _draw_panel(screen, rect, fill=(24, 31, 41), border=_BORDER, radius=16, shadow=False, fill_alpha=210)
    header_chip = pygame.Rect(rect.left + 16, rect.top + 14, 82, 22)
    pygame.draw.rect(screen, (18, 24, 33), header_chip, border_radius=11)
    pygame.draw.rect(screen, (83, 104, 132), header_chip, width=1, border_radius=11)
    chip_text = font_meta.render("DETAILS", True, (211, 224, 243))
    screen.blit(chip_text, chip_text.get_rect(center=header_chip.center))

    if model_path is None:
        screen.blit(font_text.render("No model selected", True, _MUTED), (rect.left + 16, rect.top + 54))
        screen.blit(font_meta.render("Click a model on the left list.", True, _MUTED), (rect.left + 16, rect.top + 80))
        return

    _, name, _rel, rows, top_stats = _get_model_info_content(model_path, models_dir, metadata_cache)
    text_left = rect.left + 16
    text_width = max(40, rect.width - 32)
    name_y = rect.top + 52
    screen.blit(font_text.render(_fit_text(font_text, name, text_width), True, _TEXT), (text_left, name_y))

    stat_gap = 10
    stat_w = (rect.width - 32 - stat_gap * 2) // 3
    stat_y = name_y + font_text.get_height() + 14
    for idx, (label, value) in enumerate(top_stats):
        stat_rect = pygame.Rect(rect.left + 16 + idx * (stat_w + stat_gap), stat_y, stat_w, 52)
        pygame.draw.rect(screen, (18, 24, 33), stat_rect, border_radius=12)
        pygame.draw.rect(screen, (70, 88, 113), stat_rect, width=1, border_radius=12)
        label_text = _fit_text(font_meta, label, max(8, stat_rect.width - 24))
        value_text = _fit_text(font_h2, str(value), max(8, stat_rect.width - 24))
        screen.blit(font_meta.render(label_text, True, (149, 166, 191)), (stat_rect.left + 12, stat_rect.top + 10))
        value_surface = font_h2.render(value_text, True, _TEXT)
        screen.blit(value_surface, (stat_rect.left + 12, stat_rect.top + 24))

    row_h = max(24, font_meta.get_height() + 10)
    table_y = stat_y + 52 + 16
    table_height = 8 + len(rows) * row_h + 8
    table_rect = pygame.Rect(rect.left + 12, table_y, rect.width - 24, table_height)
    pygame.draw.rect(screen, (19, 24, 33), table_rect, border_radius=12)
    pygame.draw.rect(screen, (67, 83, 108), table_rect, width=1, border_radius=12)

    y = table_rect.top + 8
    label_w = 96
    for idx, (label, value) in enumerate(rows):
        stripe = pygame.Rect(table_rect.left + 6, y - 1, table_rect.width - 12, row_h - 2)
        stripe_fill = (24, 31, 42) if idx % 2 == 0 else (29, 37, 49)
        pygame.draw.rect(screen, stripe_fill, stripe, border_radius=8)
        label_text = _fit_text(font_meta, label, label_w)
        value_text = _fit_text(font_meta, value, table_rect.width - label_w - 28)
        screen.blit(font_meta.render(label_text, True, (149, 166, 191)), (table_rect.left + 16, y + 5))
        screen.blit(font_meta.render(value_text, True, _TEXT), (table_rect.left + 18 + label_w, y + 5))
        y += row_h


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
):
    actions = []

    _draw_panel(screen, panel_rect, fill=(22, 29, 39), border=_BORDER, radius=16, shadow=False, fill_alpha=200)

    pad = 10
    header_h = 30
    content_y = header_h + 8
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

    header_rect = pygame.Rect(panel_rect.left + pad, panel_rect.top + pad, panel_rect.width - 2 * pad - 10, header_h)
    pygame.draw.rect(screen, (16, 21, 30), header_rect, border_radius=11)
    pygame.draw.rect(screen, (90, 108, 136), header_rect, width=1, border_radius=10)
    metric_gap = 10
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
        return min(max_w, max(min_w, widest + pad_w))

    metric_w = {
        "ver": _column_width("Ver", [fmt_version(meta) for meta in visible_metas], 64, 132),
        "top1": _column_width("Top1", [fmt_top1(meta) for meta in visible_metas], 58, 86),
        "elo": _column_width("Elo", [fmt_elo(meta) for meta in visible_metas], 128, 184),
    }
    min_metric_w = {"ver": 54, "top1": 54, "elo": 108}
    min_model_w = 28
    max_metric_total = max(
        sum(min_metric_w.values()),
        header_rect.width - 20 - min_model_w - 2 * metric_gap,
    )
    overflow = sum(metric_w.values()) - max_metric_total
    for key in ("ver", "elo", "top1"):
        if overflow <= 0:
            break
        shrink = min(overflow, metric_w[key] - min_metric_w[key])
        metric_w[key] -= shrink
        overflow -= shrink
    metrics_right = header_rect.right - 10
    elo_left = metrics_right - metric_w["elo"]
    top1_left = elo_left - metric_gap - metric_w["top1"]
    ver_left = top1_left - metric_gap - metric_w["ver"]

    model_col_rect = pygame.Rect(header_rect.left + 8, header_rect.top + 4, max(0, ver_left - header_rect.left - 14), header_h - 8)
    ver_col_rect = pygame.Rect(ver_left, header_rect.top + 4, metric_w["ver"], header_h - 8)
    top1_col_rect = pygame.Rect(top1_left, header_rect.top + 4, metric_w["top1"], header_h - 8)
    elo_col_rect = pygame.Rect(elo_left, header_rect.top + 4, metric_w["elo"], header_h - 8)

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
        if col_rect.width <= 8:
            continue
        color = (196, 210, 232) if sortable_key and sort_key == sortable_key else (166, 181, 205)
        text = font_meta.render(_fit_text(font_meta, label, max(8, col_rect.width - 4)), True, color)
        screen.blit(text, text.get_rect(center=col_rect.center))
        if sortable_key:
            actions.append((col_rect, "sort", sortable_key))
    for sx in (ver_left - 6, top1_left - 6, elo_left - 6):
        if header_rect.left + 6 < sx < header_rect.right - 6:
            pygame.draw.line(screen, (74, 90, 114), (sx, header_rect.top + 5), (sx, header_rect.bottom - 5), 1)

    # Clip scrolling content so partially-visible rows are cropped cleanly.
    content_clip = pygame.Rect(
        panel_rect.left + pad,
        header_rect.bottom + 6,
        panel_rect.width - 2 * pad - 10,
        max(1, panel_rect.bottom - pad - (header_rect.bottom + 6)),
    )

    def _display_name_for_category(path, category_key):
        rel = _short_model_path(path, models_dir, max_len=240)
        if category_key in ("il", "rl"):
            prefix = f"{category_key}/"
            if rel.lower().startswith(prefix):
                return rel[len(prefix):]
        return rel

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

        row_h = 38
        y = panel_rect.top + pad + content_y - scroll_offset
        rect = pygame.Rect(panel_rect.left + pad, y, panel_rect.width - 2 * pad - 10, row_h)
        if rect.bottom >= content_clip.top and rect.top <= content_clip.bottom:
            hovered = rect.collidepoint(mouse_pos)
            fill = style["fill"] if not hovered else tuple(min(255, c + 14) for c in style["fill"])
            pygame.draw.rect(screen, fill, rect, border_radius=10)
            pygame.draw.rect(screen, style["border"], rect, width=1, border_radius=10)
            stripe = pygame.Rect(rect.left + 7, rect.top + 5, 7, rect.height - 10)
            pygame.draw.rect(screen, style["border"], stripe, border_radius=2)
            screen.blit(font_header.render(header_text, True, _TEXT), (rect.left + 20, rect.top + 7))

            count_text = str(count)
            chip_w = max(28, font_meta.size(count_text)[0] + 16)
            count_chip = pygame.Rect(rect.right - chip_w - 8, rect.top + 6, chip_w, rect.height - 12)
            pygame.draw.rect(screen, (25, 30, 39), count_chip, border_radius=6)
            pygame.draw.rect(screen, style["border"], count_chip, width=1, border_radius=6)
            count_label = font_meta.render(count_text, True, (224, 236, 251))
            screen.blit(count_label, count_label.get_rect(center=count_chip.center))
        if collapsible and count > 0:
            hit_rect = pygame.Rect(rect.left, rect.top - 2, rect.width, rect.height + 4).clip(content_clip)
            if hit_rect.width > 0 and hit_rect.height > 0:
                actions.append((hit_rect, "toggle", key))
        content_y += row_h + 8

    model_row_idx = 0

    def draw_model_row(path, key):
        nonlocal content_y, model_row_idx
        style = _CATEGORY_STYLE.get(key, {"border": _BORDER})
        row_h = 44
        y = panel_rect.top + pad + content_y - scroll_offset
        item_indent = 16
        rect = pygame.Rect(
            panel_rect.left + pad + item_indent,
            y,
            panel_rect.width - 2 * pad - 10 - item_indent,
            row_h,
        )

        active = path == selected_model
        if rect.bottom >= content_clip.top and rect.top <= content_clip.bottom:
            hovered = rect.collidepoint(mouse_pos)
            if active:
                fill = (55, 106, 184)
                border = (170, 211, 255)
                text_color = (248, 250, 255)
                meta_color = (226, 236, 255)
                accent_bar = (201, 226, 255)
            else:
                zebra_fill = (26, 34, 45) if model_row_idx % 2 == 0 else (23, 31, 42)
                fill = zebra_fill if not hovered else (34, 45, 58)
                border = style["border"] if hovered else _BORDER
                text_color = _TEXT
                meta_color = (186, 201, 222)
                accent_bar = style["border"] if hovered else (88, 106, 130)

            shadow = rect.move(0, 2)
            pygame.draw.rect(screen, (10, 14, 19), shadow, border_radius=12)
            pygame.draw.rect(screen, fill, rect, border_radius=10)
            pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
            left_bar = pygame.Rect(rect.left + 8, rect.top + 8, 4, rect.height - 16)
            pygame.draw.rect(screen, accent_bar, left_bar, border_radius=2)

            meta = metadata_cache.get(path) or {}
            ver = fmt_version(meta)
            top1 = fmt_top1(meta)
            elo = fmt_elo(meta)

            row_metrics_right = rect.right - 10
            row_elo_left = row_metrics_right - metric_w["elo"]
            row_top1_left = row_elo_left - metric_gap - metric_w["top1"]
            row_ver_left = row_top1_left - metric_gap - metric_w["ver"]

            sep_color = (164, 196, 236) if active else (78, 94, 116)
            for sx in (row_ver_left - 6, row_top1_left - 6, row_elo_left - 6):
                if rect.left + 10 < sx < rect.right - 10:
                    pygame.draw.line(screen, sep_color, (sx, rect.top + 8), (sx, rect.bottom - 8), 1)

            name_x = rect.left + 20
            name_max_w = max(0, row_ver_left - name_x - 12)
            raw_name = _display_name_for_category(path, key)
            if name_max_w > 24:
                name = _fit_text(font_text, raw_name, name_max_w)
                screen.blit(font_text.render(name, True, text_color), (name_x, rect.top + 11))

            ver_text = font_meta.render(_fit_text(font_meta, ver, max(8, metric_w["ver"] - 6)), True, meta_color)
            top1_text = font_meta.render(_fit_text(font_meta, top1, max(8, metric_w["top1"] - 6)), True, meta_color)
            elo_text = font_meta.render(_fit_text(font_meta, elo, max(8, metric_w["elo"] - 6)), True, meta_color)
            ver_rect = pygame.Rect(row_ver_left, rect.top + 10, metric_w["ver"], rect.height - 20)
            top1_rect = pygame.Rect(row_top1_left, rect.top + 10, metric_w["top1"], rect.height - 20)
            elo_rect = pygame.Rect(row_elo_left, rect.top + 10, metric_w["elo"], rect.height - 20)
            screen.blit(ver_text, ver_text.get_rect(center=ver_rect.center))
            screen.blit(top1_text, top1_text.get_rect(center=top1_rect.center))
            screen.blit(elo_text, elo_text.get_rect(center=elo_rect.center))

            hit_rect = rect.clip(content_clip)
            if hit_rect.width > 0 and hit_rect.height > 0:
                actions.append((hit_rect, "select", path))

        model_row_idx += 1
        content_y += row_h + 5

    previous_clip = screen.get_clip()
    screen.set_clip(content_clip)
    for idx, key in enumerate(visible_keys):
        if idx > 0:
            content_y += 4
        draw_header(key)
        if key == "best" or expanded.get(key, False):
            for path in _sorted_paths(key):
                draw_model_row(path, key)
            content_y += 4
    screen.set_clip(previous_clip)

    content_height = content_y + 8
    viewport = content_clip.height
    max_scroll = max(0, content_height - viewport)

    if max_scroll > 0:
        track = pygame.Rect(panel_rect.right - 12, content_clip.top, 6, content_clip.height)
        pygame.draw.rect(screen, (39, 48, 62), track, border_radius=3)
        thumb_h = max(28, int(track.height * (viewport / max(content_height, 1))))
        thumb_h = min(track.height, thumb_h)
        travel = track.height - thumb_h
        ratio = 0.0 if max_scroll <= 0 else (scroll_offset / max_scroll)
        thumb_y = track.top + int(travel * max(0.0, min(1.0, ratio)))
        thumb = pygame.Rect(track.left, thumb_y, track.width, thumb_h)
        pygame.draw.rect(screen, (138, 162, 194), thumb, border_radius=3)

    return actions, max_scroll


def _model_browser_panel_height(grouped, expanded_state):
    visible_keys = [key for key in ("best", "il", "rl", "other") if grouped[key]]
    if not visible_keys:
        return 76

    pad = 10
    header_h = 30
    content_y = header_h + 8
    header_row_h = 38
    model_row_h = 44

    for idx, key in enumerate(visible_keys):
        if idx > 0:
            content_y += 4
        content_y += header_row_h + 8
        if key == "best" or expanded_state.get(key, False):
            content_y += len(grouped[key]) * (model_row_h + 5)
            content_y += 4

    content_height = content_y + 8
    return content_height + (2 * pad + header_h + 6)


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
    for model_path in all_models:
        try:
            metadata_cache[model_path] = load_checkpoint_metadata(model_path, models_dir)
        except Exception:
            metadata_cache[model_path] = {}

    design_width = 1360
    design_height = 900
    base_width = design_width
    base_height = design_height
    canvas_min_width = 980
    canvas_min_height = 660
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

    def _get_display_window_size():
        try:
            user32 = ctypes.windll.user32
            screen_width = int(user32.GetSystemMetrics(0) or 0)
            screen_height = int(user32.GetSystemMetrics(1) or 0)
        except Exception:
            try:
                info = pygame.display.Info()
                screen_width = int(getattr(info, "current_w", 0) or 0)
                screen_height = int(getattr(info, "current_h", 0) or 0)
            except Exception:
                screen_width = 0
                screen_height = 0

        if screen_width <= 0 or screen_height <= 0:
            return default_window_width, default_window_height

        return (
            max(min_width, screen_width),
            max(min_height, screen_height),
        )

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
        responsive_scale = min(base_width / design_width, base_height / design_height) * 0.90
        ui_scale = max(0.72, min(1.04, responsive_scale))
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
            maximized_size = _get_display_window_size()
            screen = pygame.display.set_mode(maximized_size, display_flags)
            pygame.event.pump()
            _position_native_window(*maximized_size)
            _maximize_native_window()
            maximized = True
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
    workspace_tab = "models"
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
    match_games = _safe_int(saved_preferences.get("match_games"), default=1)
    match_games = max(1, min(500, int(match_games or 1)))
    match_games_step = 2
    active_numeric_field = None
    numeric_input_text = ""
    mcts_profiles = [
        ("Fast", 64),
        ("Balanced", 100),
        ("Strong", 200),
        ("Ultra", 400),
    ]

    default_model = _default_model(grouped)
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
    active_side = "white"

    browser_rect = pygame.Rect(36, 420, 860, 360)
    browser_actions = []
    browser_max_scroll = 0

    last_saved_preferences = {
        "use_mcts": bool(use_mcts),
        "use_mcts_white": bool(use_mcts_white),
        "use_mcts_black": bool(use_mcts_black),
        "mcts_simulations": int(mcts_simulations),
        "mcts_simulations_white": int(mcts_simulations_white),
        "mcts_simulations_black": int(mcts_simulations_black),
        "match_games": int(match_games),
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

    def begin_numeric_input(field_key, current_value):
        nonlocal active_numeric_field, numeric_input_text
        active_numeric_field = field_key
        numeric_input_text = str(int(current_value))

    def cancel_numeric_input():
        nonlocal active_numeric_field, numeric_input_text
        active_numeric_field = None
        numeric_input_text = ""

    def commit_numeric_input():
        nonlocal active_numeric_field, numeric_input_text
        nonlocal mcts_simulations, mcts_simulations_white, mcts_simulations_black, match_games
        if active_numeric_field is None:
            return
        raw = numeric_input_text.strip()
        try:
            value = int(raw) if raw else None
        except ValueError:
            value = None

        if active_numeric_field == "match_games":
            if value is not None:
                match_games = max(1, min(500, value))
        elif active_numeric_field == "mcts":
            if value is not None:
                mcts_simulations = max(16, min(2000, value))
        elif active_numeric_field == "mcts_white":
            if value is not None:
                mcts_simulations_white = max(16, min(2000, value))
        elif active_numeric_field == "mcts_black":
            if value is not None:
                mcts_simulations_black = max(16, min(2000, value))

        active_numeric_field = None
        numeric_input_text = ""
        persist_ui_preferences()

    def numeric_text(field_key, current_value):
        if active_numeric_field != field_key:
            return str(int(current_value))
        caret = "|" if (pygame.time.get_ticks() // 450) % 2 == 0 else ""
        return f"{numeric_input_text}{caret}"

    def draw_numeric_value(rect, field_key, current_value, font):
        active = active_numeric_field == field_key
        border = (118, 165, 234) if active else (94, 112, 140)
        fill = (22, 31, 45) if active else (19, 25, 35)
        pygame.draw.rect(canvas, fill, rect, border_radius=10)
        pygame.draw.rect(canvas, border, rect, width=2 if active else 1, border_radius=10)
        label = font.render(numeric_text(field_key, current_value), True, _TEXT)
        canvas.blit(label, label.get_rect(center=rect.center))

    def apply_sort(selected_key):
        nonlocal sort_key, sort_desc
        if selected_key not in ("model", "top1", "elo"):
            return
        if sort_key == selected_key:
            sort_desc = not sort_desc
        else:
            sort_key = selected_key
            sort_desc = selected_key in ("top1", "elo")

    while True:
        mapped_mouse = window_to_ui(pygame.mouse.get_pos())
        mouse_pos = mapped_mouse if mapped_mouse is not None else (-9999, -9999)

        scale_px = lambda value, minimum=1: max(minimum, int(round(value * ui_scale)))
        outer_pad = scale_px(24)
        content_pad = scale_px(40)
        small_gap = scale_px(14)
        section_gap = scale_px(18)
        tiny_gap = scale_px(10)
        header_radius = scale_px(22)
        current_selected_model = None
        current_browser_height = None
        if workspace_tab == "models" and game_mode != "human_vs_human":
            if game_mode == "ai_vs_ai":
                current_selected_model = selected_models.get(active_side)
                current_browser_height = _model_browser_panel_height(grouped, expanded.get(active_side, {}))
            else:
                current_selected_model = opponent_model
                current_browser_height = _model_browser_panel_height(grouped, expanded.get("opponent", {}))
        predicted_workspace_label_y = scale_px(258 if game_mode != "human_vs_ai" else 340)
        predicted_tabs_y = predicted_workspace_label_y + section_gap
        predicted_content_top = predicted_tabs_y + scale_px(46)
        footer_gap = scale_px(20)
        footer_stack_height = scale_px(78)
        footer_top = None
        if workspace_tab == "settings":
            workspace_min_height = scale_px(420 if game_mode == "ai_vs_ai" else 360)
            layout_height = max(base_height, predicted_content_top + workspace_min_height + scale_px(92))
            footer_top = layout_height - scale_px(78)
        elif game_mode == "ai_vs_ai":
            card_h = _selection_card_height(font_card_label, font_small, font_tiny)
            section_panel_height = max(
                current_browser_height or 0,
                _model_info_panel_height(
                    current_selected_model,
                    models_dir,
                    metadata_cache,
                    font_small,
                    font_small,
                    font_meta,
                ),
            )
            footer_top = predicted_content_top + card_h + scale_px(54) + section_panel_height + footer_gap
            workspace_min_height = max(scale_px(280), footer_top - predicted_content_top + scale_px(14))
            layout_height = max(base_height, footer_top + footer_stack_height)
        elif game_mode == "human_vs_ai":
            card_h = _selection_card_height(font_card_label, font_small, font_tiny)
            section_panel_height = max(
                current_browser_height or 0,
                _model_info_panel_height(
                    current_selected_model,
                    models_dir,
                    metadata_cache,
                    font_small,
                    font_small,
                    font_meta,
                ),
            )
            footer_top = predicted_content_top + card_h + scale_px(20) + section_panel_height + footer_gap
            workspace_min_height = max(scale_px(240), footer_top - predicted_content_top + scale_px(14))
            layout_height = max(base_height, footer_top + footer_stack_height)
        else:
            workspace_min_height = scale_px(220)
            footer_top = predicted_content_top + scale_px(142) + footer_gap
            layout_height = max(base_height, footer_top + footer_stack_height)
        page_height = layout_height
        max_page_scroll = clamp_page_scroll()
        canvas = pygame.Surface((base_width, page_height))
        canvas.blit(get_page_background(page_height), (0, 0))

        hero_rect = pygame.Rect(outer_pad, scale_px(16), base_width - outer_pad * 2, scale_px(118))
        _draw_panel(canvas, hero_rect, fill=(20, 27, 37), border=(108, 130, 166), radius=header_radius, shadow=True)
        hero_band = pygame.Rect(hero_rect.left, hero_rect.top, hero_rect.width, scale_px(10))
        pygame.draw.rect(canvas, _ACCENT, hero_band, border_top_left_radius=header_radius, border_top_right_radius=header_radius)
        title = font_title.render("Chess AI Setup", True, _TEXT)
        subtitle = font_small.render("Choose a mode, assign the models, and start playing.", True, _MUTED)
        canvas.blit(title, (content_pad + scale_px(2), scale_px(28)))
        canvas.blit(subtitle, (content_pad + scale_px(4), scale_px(84)))
        settings_w = scale_px(136)
        settings_h = scale_px(38)
        settings_icon_rect = pygame.Rect(base_width - content_pad - settings_w, scale_px(30), settings_w, settings_h)
        _draw_button(
            canvas,
            settings_icon_rect,
            "Settings",
            font_meta,
            hovered=settings_icon_rect.collidepoint(mouse_pos),
            active=workspace_tab == "settings",
        )

        mode_label = font_small.render("Mode", True, _MUTED)
        canvas.blit(mode_label, (content_pad, scale_px(156)))

        mode_gap = scale_px(14)
        mode_btn_w = scale_px(232)
        mode_btn_h = scale_px(52)
        mode_y = scale_px(182)
        mode_rects = {
            "human_vs_ai": pygame.Rect(content_pad, mode_y, mode_btn_w, mode_btn_h),
            "ai_vs_ai": pygame.Rect(content_pad + mode_btn_w + mode_gap, mode_y, mode_btn_w, mode_btn_h),
            "human_vs_human": pygame.Rect(content_pad + (mode_btn_w + mode_gap) * 2, mode_y, mode_btn_w, mode_btn_h),
        }
        _draw_button(
            canvas,
            mode_rects["human_vs_ai"],
            "Human vs AI",
            font_text,
            hovered=mode_rects["human_vs_ai"].collidepoint(mouse_pos),
            active=game_mode == "human_vs_ai",
            disabled=not has_models,
        )
        _draw_button(
            canvas,
            mode_rects["ai_vs_ai"],
            "AI vs AI",
            font_text,
            hovered=mode_rects["ai_vs_ai"].collidepoint(mouse_pos),
            active=game_mode == "ai_vs_ai",
            disabled=not has_models,
        )
        _draw_button(
            canvas,
            mode_rects["human_vs_human"],
            "Human vs Human",
            font_text,
            hovered=mode_rects["human_vs_human"].collidepoint(mouse_pos),
            active=game_mode == "human_vs_human",
        )
        color_white_rect = pygame.Rect(content_pad, scale_px(274), scale_px(172), scale_px(44))
        color_black_rect = pygame.Rect(color_white_rect.right + tiny_gap, scale_px(274), scale_px(172), scale_px(44))
        white_card_rect = None
        black_card_rect = None
        copy_white_rect = None
        copy_black_rect = None
        swap_rect = None
        settings_panel_rect = None
        mcts_toggle_rect = None
        mcts_minus_rect = None
        mcts_value_rect = None
        mcts_plus_rect = None
        mcts_profile_buttons = []
        mcts_white_toggle_rect = None
        mcts_black_toggle_rect = None
        mcts_white_minus_rect = None
        mcts_white_value_rect = None
        mcts_white_plus_rect = None
        mcts_black_minus_rect = None
        mcts_black_value_rect = None
        mcts_black_plus_rect = None
        match_games_minus_rect = None
        match_games_value_rect = None
        match_games_plus_rect = None
        content_top = scale_px(356)
        content_bottom = layout_height - scale_px(92)
        content_height = max(scale_px(180), content_bottom - content_top)
        browser_key = None
        browser_actions = []
        browser_max_scroll = 0
        info_rect = None

        if game_mode == "human_vs_ai":
            canvas.blit(font_h2.render("Human color", True, _TEXT), (content_pad, scale_px(246)))
            _draw_button(
                canvas,
                color_white_rect,
                "White",
                font_text,
                hovered=color_white_rect.collidepoint(mouse_pos),
                active=human_color == chess.WHITE,
            )
            _draw_button(
                canvas,
                color_black_rect,
                "Black",
                font_text,
                hovered=color_black_rect.collidepoint(mouse_pos),
                active=human_color == chess.BLACK,
            )

        workspace_label_y = scale_px(258 if game_mode != "human_vs_ai" else 340)
        tabs_y = workspace_label_y + section_gap
        if workspace_tab == "settings":
            canvas.blit(
                font_meta.render("Settings active. Click Settings again to return to model selection.", True, _MUTED),
                (content_pad, tabs_y + tiny_gap),
            )
        content_top = tabs_y + scale_px(46)
        content_bottom = max(content_top + scale_px(180), footer_top - scale_px(14))
        content_height = max(scale_px(180), content_bottom - content_top)
        workspace_rect = pygame.Rect(outer_pad, content_top - section_gap, base_width - outer_pad * 2, content_height + scale_px(28))
        _draw_panel(canvas, workspace_rect, fill=(18, 24, 34), border=(86, 105, 136), radius=header_radius, shadow=True, fill_alpha=188)

        if workspace_tab == "settings":
            settings_panel_rect = pygame.Rect(content_pad, content_top, base_width - content_pad * 2, max(scale_px(220), content_height - scale_px(10)))
            _draw_panel(canvas, settings_panel_rect, fill=(23, 30, 41), border=(102, 126, 162), radius=scale_px(18), shadow=False, fill_alpha=220)

            canvas.blit(font_h2.render("Search & Runtime Settings", True, _TEXT), (settings_panel_rect.left + scale_px(24), settings_panel_rect.top + scale_px(20)))
            canvas.blit(
                font_small.render("Dial in the engine once here and keep the rest of the setup clean.", True, _MUTED),
                (settings_panel_rect.left + scale_px(24), settings_panel_rect.top + scale_px(52)),
            )

            mcts_toggle_rect = pygame.Rect(settings_panel_rect.left + scale_px(24), settings_panel_rect.top + scale_px(86), scale_px(230), scale_px(46))
            _draw_button(
                canvas,
                mcts_toggle_rect,
                f"All sides: MCTS {'ON' if use_mcts else 'OFF'}" if game_mode == "ai_vs_ai" else f"MCTS: {'ON' if use_mcts else 'OFF'}",
                font_text,
                hovered=mcts_toggle_rect.collidepoint(mouse_pos),
                active=use_mcts,
                disabled=game_mode == "human_vs_human",
            )

            if game_mode == "ai_vs_ai":
                match_label_x = mcts_toggle_rect.right + scale_px(28)
                match_label_y = mcts_toggle_rect.top - scale_px(2)
                canvas.blit(font_meta.render("Match games", True, _MUTED), (match_label_x, match_label_y))
                match_controls_y = mcts_toggle_rect.top + scale_px(18)
                match_games_minus_rect = pygame.Rect(match_label_x, match_controls_y, scale_px(40), scale_px(34))
                match_games_value_rect = pygame.Rect(match_games_minus_rect.right + scale_px(8), match_controls_y, scale_px(96), scale_px(34))
                match_games_plus_rect = pygame.Rect(match_games_value_rect.right + scale_px(8), match_controls_y, scale_px(40), scale_px(34))
                _draw_button(
                    canvas,
                    match_games_minus_rect,
                    "-",
                    font_h2,
                    hovered=match_games_minus_rect.collidepoint(mouse_pos),
                    disabled=match_games <= 1,
                )
                draw_numeric_value(match_games_value_rect, "match_games", match_games, font_h2)
                _draw_button(
                    canvas,
                    match_games_plus_rect,
                    "+",
                    font_h2,
                    hovered=match_games_plus_rect.collidepoint(mouse_pos),
                )

            if game_mode == "human_vs_human":
                canvas.blit(
                    font_meta.render("MCTS is disabled in Human vs Human mode.", True, (221, 181, 128)),
                    (mcts_toggle_rect.right + scale_px(16), mcts_toggle_rect.top + scale_px(14)),
                )

            sims_title_y = settings_panel_rect.top + scale_px(150)
            if game_mode == "ai_vs_ai":
                canvas.blit(font_small.render("MCTS simulations per side", True, _TEXT), (settings_panel_rect.left + scale_px(24), sims_title_y))
                canvas.blit(font_meta.render("White and Black can use different search modes and budgets.", True, _MUTED), (settings_panel_rect.left + scale_px(24), sims_title_y + scale_px(24)))

                controls_y = settings_panel_rect.top + scale_px(198)
                card_gap = scale_px(18)
                card_w = (settings_panel_rect.width - scale_px(48) - card_gap) // 2
                white_card = pygame.Rect(settings_panel_rect.left + scale_px(24), controls_y, card_w, scale_px(146))
                black_card = pygame.Rect(white_card.right + card_gap, controls_y, card_w, scale_px(146))
                for card_rect, label_text, sims_value, side_uses_mcts, is_white in (
                    (white_card, "White search", mcts_simulations_white, use_mcts_white, True),
                    (black_card, "Black search", mcts_simulations_black, use_mcts_black, False),
                ):
                    pygame.draw.rect(canvas, (18, 24, 33), card_rect, border_radius=12)
                    pygame.draw.rect(canvas, (70, 84, 104), card_rect, width=1, border_radius=12)
                    canvas.blit(font_small.render(label_text, True, _TEXT), (card_rect.left + scale_px(14), card_rect.top + scale_px(12)))
                    toggle_rect = pygame.Rect(card_rect.left + scale_px(14), card_rect.top + scale_px(42), card_rect.width - scale_px(28), scale_px(34))
                    _draw_button(
                        canvas,
                        toggle_rect,
                        f"MCTS: {'ON' if side_uses_mcts else 'OFF'}",
                        font_meta,
                        hovered=toggle_rect.collidepoint(mouse_pos),
                        active=side_uses_mcts,
                    )
                    controls_row_y = card_rect.top + scale_px(90)
                    minus_rect = pygame.Rect(card_rect.left + scale_px(14), controls_row_y, scale_px(40), scale_px(38))
                    value_rect = pygame.Rect(card_rect.left + scale_px(62), controls_row_y, card_rect.width - scale_px(124), scale_px(38))
                    plus_rect = pygame.Rect(card_rect.right - scale_px(54), controls_row_y, scale_px(40), scale_px(38))
                    _draw_button(
                        canvas,
                        minus_rect,
                        "-",
                        font_h2,
                        hovered=minus_rect.collidepoint(mouse_pos),
                    )
                    draw_numeric_value(value_rect, "mcts_white" if is_white else "mcts_black", sims_value, font_h2)
                    _draw_button(
                        canvas,
                        plus_rect,
                        "+",
                        font_h2,
                        hovered=plus_rect.collidepoint(mouse_pos),
                    )
                    if is_white:
                        mcts_white_toggle_rect = toggle_rect
                        mcts_white_minus_rect = minus_rect
                        mcts_white_value_rect = value_rect
                        mcts_white_plus_rect = plus_rect
                    else:
                        mcts_black_toggle_rect = toggle_rect
                        mcts_black_minus_rect = minus_rect
                        mcts_black_value_rect = value_rect
                        mcts_black_plus_rect = plus_rect
                preset_y = controls_y + scale_px(160)
            else:
                canvas.blit(font_small.render("MCTS simulations per move", True, _TEXT), (settings_panel_rect.left + scale_px(24), sims_title_y))
                profile_text = "Speed profile: Fast" if mcts_simulations <= 80 else (
                    "Speed profile: Balanced" if mcts_simulations <= 140 else (
                        "Speed profile: Strong" if mcts_simulations <= 300 else "Speed profile: Ultra"
                    )
                )
                canvas.blit(font_meta.render(profile_text, True, _MUTED), (settings_panel_rect.left + scale_px(24), sims_title_y + scale_px(24)))

                controls_y = settings_panel_rect.top + scale_px(198)
                mcts_minus_rect = pygame.Rect(settings_panel_rect.left + scale_px(24), controls_y, scale_px(44), scale_px(42))
                mcts_value_rect = pygame.Rect(settings_panel_rect.left + scale_px(76), controls_y, scale_px(168), scale_px(42))
                mcts_plus_rect = pygame.Rect(settings_panel_rect.left + scale_px(252), controls_y, scale_px(44), scale_px(42))
                _draw_button(
                    canvas,
                    mcts_minus_rect,
                    "-",
                    font_h2,
                    hovered=mcts_minus_rect.collidepoint(mouse_pos),
                )
                draw_numeric_value(mcts_value_rect, "mcts", mcts_simulations, font_h2)
                _draw_button(
                    canvas,
                    mcts_plus_rect,
                    "+",
                    font_h2,
                    hovered=mcts_plus_rect.collidepoint(mouse_pos),
                )
                preset_y = controls_y + scale_px(62)

            preset_gap = scale_px(12)
            preset_width = (settings_panel_rect.width - scale_px(48) - preset_gap * (len(mcts_profiles) - 1)) // len(mcts_profiles)
            for idx, (label, value) in enumerate(mcts_profiles):
                rect = pygame.Rect(
                    settings_panel_rect.left + scale_px(24) + idx * (preset_width + preset_gap),
                    preset_y,
                    preset_width,
                    scale_px(38),
                )
                _draw_button(
                    canvas,
                    rect,
                    f"{label} ({value})",
                    font_meta,
                    hovered=rect.collidepoint(mouse_pos),
                    active=(
                        mcts_simulations == value
                        if game_mode != "ai_vs_ai"
                        else mcts_simulations_white == value and mcts_simulations_black == value
                    ),
                )
                mcts_profile_buttons.append((rect, value))

            help_rows = [
                f"- Current step: +/-{mcts_step} sims.",
                "- Fast/Balanced are better for many games and quick testing.",
                "- Strong/Ultra give better move quality but each move is slower.",
            ]
            info_box = pygame.Rect(settings_panel_rect.left + scale_px(24), preset_y + scale_px(56), settings_panel_rect.width - scale_px(48), scale_px(92))
            pygame.draw.rect(canvas, (18, 24, 33), info_box, border_radius=12)
            pygame.draw.rect(canvas, (70, 84, 104), info_box, width=1, border_radius=12)
            y_row = info_box.top + scale_px(12)
            for row in help_rows:
                canvas.blit(font_meta.render(_fit_text(font_meta, row, info_box.width - scale_px(24)), True, _MUTED), (info_box.left + scale_px(12), y_row))
                y_row += max(scale_px(18), font_meta.get_height() + scale_px(6))
        elif game_mode != "human_vs_human":
            if game_mode == "ai_vs_ai":
                canvas.blit(font_h2.render("Model Assignment", True, _TEXT), (content_pad, content_top - scale_px(34)))
                cards_gap = scale_px(14)
                card_h = _selection_card_height(font_card_label, font_small, font_tiny)
                row_w = base_width - content_pad * 2
                action_labels = ("Swap", "White -> both", "Black -> both")
                action_inner_pad_x = scale_px(12)
                action_button_text_w = max(font_meta.size(label)[0] for label in action_labels)
                action_button_w = max(scale_px(92), action_button_text_w + scale_px(24))
                center_panel_w = action_button_w + action_inner_pad_x * 2
                card_w = max(scale_px(260), (row_w - center_panel_w - cards_gap * 2) // 2)
                total_used = card_w * 2 + center_panel_w + cards_gap * 2
                row_left = content_pad + max(0, (row_w - total_used) // 2)
                white_card_rect = pygame.Rect(row_left, content_top, card_w, card_h)
                action_panel_rect = pygame.Rect(white_card_rect.right + cards_gap, content_top, center_panel_w, card_h)
                black_card_rect = pygame.Rect(action_panel_rect.right + cards_gap, content_top, card_w, card_h)
                _draw_selection_card(
                    canvas,
                    white_card_rect,
                    "White model",
                    selected_models["white"],
                    models_dir,
                    font_card_label,
                    font_card_name,
                    font_card_meta,
                    metadata_cache=metadata_cache,
                    active=active_side == "white",
                    hovered=white_card_rect.collidepoint(mouse_pos),
                    side_color=chess.WHITE,
                )
                _draw_selection_card(
                    canvas,
                    black_card_rect,
                    "Black model",
                    selected_models["black"],
                    models_dir,
                    font_card_label,
                    font_card_name,
                    font_card_meta,
                    metadata_cache=metadata_cache,
                    active=active_side == "black",
                    hovered=black_card_rect.collidepoint(mouse_pos),
                    side_color=chess.BLACK,
                )
                _draw_panel(canvas, action_panel_rect, fill=(21, 28, 38), border=(72, 91, 118), radius=16, shadow=False, fill_alpha=212)

                inner_pad_x = action_inner_pad_x
                button_gap = scale_px(8)
                button_h = scale_px(28)
                button_w = action_panel_rect.width - inner_pad_x * 2
                buttons_total_h = button_h * 3 + button_gap * 2
                first_button_y = action_panel_rect.centery - buttons_total_h // 2
                swap_rect = pygame.Rect(action_panel_rect.left + inner_pad_x, first_button_y, button_w, button_h)
                copy_white_rect = pygame.Rect(action_panel_rect.left + inner_pad_x, swap_rect.bottom + button_gap, button_w, button_h)
                copy_black_rect = pygame.Rect(action_panel_rect.left + inner_pad_x, copy_white_rect.bottom + button_gap, button_w, button_h)
                _draw_button(
                    canvas,
                    swap_rect,
                    "Swap",
                    font_meta,
                    hovered=swap_rect.collidepoint(mouse_pos),
                )
                _draw_button(
                    canvas,
                    copy_white_rect,
                    "White -> both",
                    font_meta,
                    hovered=copy_white_rect.collidepoint(mouse_pos),
                    disabled=selected_models["white"] is None,
                )
                _draw_button(
                    canvas,
                    copy_black_rect,
                    "Black -> both",
                    font_meta,
                    hovered=copy_black_rect.collidepoint(mouse_pos),
                    disabled=selected_models["black"] is None,
                )

                browser_key = active_side
                browser_top = max(white_card_rect.bottom, action_panel_rect.bottom, black_card_rect.bottom) + scale_px(24)
                browser_h = max(scale_px(180), _model_browser_panel_height(grouped, expanded[browser_key]))
                browser_w = int((base_width - content_pad * 2) * 0.67)
                info_w = base_width - content_pad * 2 - browser_w - scale_px(16)
                browser_rect = pygame.Rect(content_pad, browser_top, browser_w, browser_h)
                info_rect = pygame.Rect(browser_rect.right + scale_px(12), browser_top, info_w, 0)
            else:
                ai_side_name = "Black" if human_color == chess.WHITE else "White"
                header = f"Opponent model ({ai_side_name} AI side)"
                canvas.blit(font_h2.render(header, True, _TEXT), (content_pad, content_top - scale_px(34)))
                opponent_card_rect = pygame.Rect(content_pad, content_top, base_width - content_pad * 2, _selection_card_height(font_card_label, font_small, font_tiny))
                _draw_selection_card(
                    canvas,
                    opponent_card_rect,
                    "Selected opponent",
                    opponent_model,
                    models_dir,
                    font_card_label,
                    font_card_name,
                    font_card_meta,
                    metadata_cache=metadata_cache,
                    active=True,
                    hovered=opponent_card_rect.collidepoint(mouse_pos),
                    side_color=chess.BLACK if human_color == chess.WHITE else chess.WHITE,
                )
                browser_key = "opponent"
                browser_top = opponent_card_rect.bottom + scale_px(20)
                browser_h = max(scale_px(180), _model_browser_panel_height(grouped, expanded[browser_key]))
                browser_w = int((base_width - content_pad * 2) * 0.67)
                info_w = base_width - content_pad * 2 - browser_w - scale_px(16)
                browser_rect = pygame.Rect(content_pad, browser_top, browser_w, browser_h)
                info_rect = pygame.Rect(browser_rect.right + scale_px(12), browser_top, info_w, 0)

            selected_model = selected_models[active_side] if browser_key != "opponent" else opponent_model
            info_required_h = _model_info_panel_height(
                selected_model,
                models_dir,
                metadata_cache,
                font_small,
                font_small,
                font_meta,
            )
            info_rect.height = info_required_h
            section_bottom = browser_top + max(browser_rect.height, info_rect.height)
            if section_bottom > content_bottom:
                extra_needed = section_bottom - content_bottom
                content_bottom += extra_needed
                content_height += extra_needed
                workspace_rect.height += extra_needed
            browser_actions, browser_max_scroll = _draw_model_browser(
                screen=canvas,
                panel_rect=browser_rect,
                grouped=grouped,
                models_dir=models_dir,
                metadata_cache=metadata_cache,
                selected_model=selected_model,
                expanded=expanded[browser_key],
                scroll_offset=scroll[browser_key],
                font_header=font_text,
                font_text=font_text,
                font_meta=font_meta,
                mouse_pos=mouse_pos,
                sort_key=sort_key,
                sort_desc=sort_desc,
            )
            _draw_model_info_panel(
                screen=canvas,
                rect=info_rect,
                model_path=selected_model,
                models_dir=models_dir,
                metadata_cache=metadata_cache,
                font_h2=font_small,
                font_text=font_small,
                font_meta=font_meta,
            )

            if scroll[browser_key] > browser_max_scroll:
                scroll[browser_key] = browser_max_scroll
        else:
            info_box = pygame.Rect(content_pad, content_top, base_width - content_pad * 2, scale_px(142))
            _draw_panel(canvas, info_box, fill=(24, 31, 42), border=_BORDER, radius=scale_px(18), shadow=False, fill_alpha=204)
            canvas.blit(font_h2.render("Human vs Human", True, _TEXT), (content_pad + scale_px(20), info_box.top + scale_px(24)))
            canvas.blit(
                font_small.render("No model selection required. Press Start to open the board.", True, _MUTED),
                (content_pad + scale_px(20), info_box.top + scale_px(68)),
            )
            canvas.blit(
                font_meta.render("Open Settings if you want to preconfigure MCTS before switching back to AI modes.", True, _MUTED),
                (content_pad + scale_px(20), info_box.top + scale_px(98)),
            )

        if not has_models:
            warn = "No model files found. Only Human vs Human is available."
            canvas.blit(font_small.render(warn, True, (224, 176, 108)), (content_pad, footer_top - scale_px(48)))

        can_start = _can_start(game_mode, has_models, selected_models, opponent_model)

        footer_rect = pygame.Rect(outer_pad, footer_top, base_width - outer_pad * 2, scale_px(54))
        _draw_panel(canvas, footer_rect, fill=(17, 23, 31), border=(56, 72, 93), radius=scale_px(18), shadow=False, fill_alpha=210)
        footer_hint = "Selected setup is ready to launch." if can_start else "Complete the required selections to start."
        footer_color = (184, 220, 255) if can_start else _MUTED
        canvas.blit(font_meta.render(_fit_text(font_meta, footer_hint, footer_rect.width - scale_px(340)), True, footer_color), (content_pad, footer_rect.top + scale_px(18)))

        start_rect = pygame.Rect(base_width - content_pad - scale_px(302), footer_top + scale_px(8), scale_px(156), scale_px(40))
        cancel_rect = pygame.Rect(base_width - content_pad - scale_px(136), footer_top + scale_px(8), scale_px(136), scale_px(40))
        _draw_button(
            canvas,
            start_rect,
            "Start Match" if game_mode == "ai_vs_ai" and match_games > 1 else "Start Game",
            font_text,
            hovered=start_rect.collidepoint(mouse_pos),
            active=can_start,
            disabled=not can_start,
        )
        _draw_button(
            canvas,
            cancel_rect,
            "Cancel",
            font_text,
            hovered=cancel_rect.collidepoint(mouse_pos),
            danger=True,
        )

        screen.fill((10, 12, 18))
        visible_rect = pygame.Rect(0, page_scroll_y, base_width, base_height)
        visible_surface = canvas.subsurface(visible_rect)
        if viewport_rect.size == (base_width, base_height):
            screen.blit(visible_surface, viewport_rect.topleft)
        else:
            scaled = pygame.transform.smoothscale(visible_surface, viewport_rect.size)
            screen.blit(scaled, viewport_rect.topleft)
        if max_page_scroll > 0:
            track_margin = 10
            track_w = 8
            track_h = max(60, viewport_rect.height - 2 * track_margin)
            track = pygame.Rect(
                viewport_rect.right - track_margin - track_w,
                viewport_rect.top + track_margin,
                track_w,
                track_h,
            )
            pygame.draw.rect(screen, (32, 40, 54), track, border_radius=4)
            thumb_h = max(32, int(track.height * (base_height / max(page_height, 1))))
            thumb_h = min(track.height, thumb_h)
            travel = max(0, track.height - thumb_h)
            ratio = 0.0 if max_page_scroll <= 0 else (page_scroll_y / max_page_scroll)
            thumb_y = track.top + int(travel * ratio)
            thumb = pygame.Rect(track.left, thumb_y, track.width, thumb_h)
            pygame.draw.rect(screen, (118, 165, 234), thumb, border_radius=4)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                persist_ui_preferences(force=True)
                return None

            if event.type == pygame.VIDEORESIZE:
                if maximized:
                    sync_window_surface()
                else:
                    set_window_size(event.w, event.h)
                continue

            if event.type == pygame.KEYDOWN:
                if active_numeric_field is not None:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        commit_numeric_input()
                    elif event.key == pygame.K_ESCAPE:
                        cancel_numeric_input()
                    elif event.key == pygame.K_BACKSPACE:
                        numeric_input_text = numeric_input_text[:-1]
                    elif event.key == pygame.K_DELETE:
                        numeric_input_text = ""
                    elif event.unicode and event.unicode.isdigit() and len(numeric_input_text) < 5:
                        numeric_input_text += event.unicode
                    continue
                if event.key == pygame.K_ESCAPE:
                    persist_ui_preferences(force=True)
                    return None
                if event.key == pygame.K_PAGEUP:
                    page_scroll_y = max(0, page_scroll_y - scale_px(220))
                    continue
                if event.key == pygame.K_PAGEDOWN:
                    page_scroll_y = min(max_page_scroll, page_scroll_y + scale_px(220))
                    continue
                if event.key == pygame.K_HOME:
                    page_scroll_y = 0
                    continue
                if event.key == pygame.K_END:
                    page_scroll_y = max_page_scroll
                    continue
                if event.key == pygame.K_RETURN and can_start:
                    persist_ui_preferences(force=True)
                    return _with_window_state(
                        _selection_to_result(
                            game_mode,
                            human_color,
                            selected_models,
                            opponent_model,
                            use_mcts,
                            use_mcts_white,
                            use_mcts_black,
                            mcts_simulations,
                            mcts_simulations_white,
                            mcts_simulations_black,
                            match_games,
                        )
                    )

            if event.type == pygame.MOUSEWHEEL:
                if (
                    workspace_tab == "models"
                    and game_mode != "human_vs_human"
                    and browser_key
                    and browser_rect.collidepoint(mouse_pos)
                    and browser_max_scroll > 0
                ):
                    scroll[browser_key] -= event.y * 30
                    if scroll[browser_key] < 0:
                        scroll[browser_key] = 0
                    if scroll[browser_key] > browser_max_scroll:
                        scroll[browser_key] = browser_max_scroll
                elif max_page_scroll > 0:
                    page_scroll_y -= event.y * scale_px(56)
                    page_scroll_y = max(0, min(max_page_scroll, page_scroll_y))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                ui_pos = window_to_ui(event.pos)
                if ui_pos is None:
                    continue

                if workspace_tab == "settings":
                    numeric_fields = [
                        ("match_games", match_games_value_rect, match_games),
                        ("mcts", mcts_value_rect, mcts_simulations),
                        ("mcts_white", mcts_white_value_rect, mcts_simulations_white),
                        ("mcts_black", mcts_black_value_rect, mcts_simulations_black),
                    ]
                    for field_key, rect, current_value in numeric_fields:
                        if rect and rect.collidepoint(ui_pos):
                            begin_numeric_input(field_key, current_value)
                            break
                    else:
                        if active_numeric_field is not None:
                            commit_numeric_input()
                    if active_numeric_field is not None and any(
                        rect and rect.collidepoint(ui_pos) for _, rect, _ in numeric_fields
                    ):
                        continue
                elif active_numeric_field is not None:
                    commit_numeric_input()

                if cancel_rect.collidepoint(ui_pos):
                    persist_ui_preferences(force=True)
                    return None

                if start_rect.collidepoint(ui_pos) and can_start:
                    persist_ui_preferences(force=True)
                    return _with_window_state(
                        _selection_to_result(
                            game_mode,
                            human_color,
                            selected_models,
                            opponent_model,
                            use_mcts,
                            use_mcts_white,
                            use_mcts_black,
                            mcts_simulations,
                            mcts_simulations_white,
                            mcts_simulations_black,
                            match_games,
                        )
                    )

                if mode_rects["human_vs_human"].collidepoint(ui_pos):
                    game_mode = "human_vs_human"
                elif has_models and mode_rects["human_vs_ai"].collidepoint(ui_pos):
                    game_mode = "human_vs_ai"
                elif has_models and mode_rects["ai_vs_ai"].collidepoint(ui_pos):
                    game_mode = "ai_vs_ai"

                if game_mode == "human_vs_ai":
                    if color_white_rect.collidepoint(ui_pos):
                        human_color = chess.WHITE
                    elif color_black_rect.collidepoint(ui_pos):
                        human_color = chess.BLACK

                if settings_icon_rect and settings_icon_rect.collidepoint(ui_pos):
                    workspace_tab = "models" if workspace_tab == "settings" else "settings"
                    continue

                if workspace_tab == "settings":
                    if mcts_toggle_rect and mcts_toggle_rect.collidepoint(ui_pos) and game_mode != "human_vs_human":
                        use_mcts = not use_mcts
                        if game_mode == "ai_vs_ai":
                            use_mcts_white = bool(use_mcts)
                            use_mcts_black = bool(use_mcts)
                        persist_ui_preferences()
                        continue
                    if mcts_white_toggle_rect and mcts_white_toggle_rect.collidepoint(ui_pos):
                        use_mcts_white = not use_mcts_white
                        persist_ui_preferences()
                        continue
                    if mcts_black_toggle_rect and mcts_black_toggle_rect.collidepoint(ui_pos):
                        use_mcts_black = not use_mcts_black
                        persist_ui_preferences()
                        continue
                    if mcts_minus_rect and mcts_minus_rect.collidepoint(ui_pos):
                        mcts_simulations = max(16, mcts_simulations - mcts_step)
                        persist_ui_preferences()
                        continue
                    if mcts_plus_rect and mcts_plus_rect.collidepoint(ui_pos):
                        mcts_simulations = min(2000, mcts_simulations + mcts_step)
                        persist_ui_preferences()
                        continue
                    if mcts_white_minus_rect and mcts_white_minus_rect.collidepoint(ui_pos):
                        mcts_simulations_white = max(16, mcts_simulations_white - mcts_step)
                        persist_ui_preferences()
                        continue
                    if mcts_white_plus_rect and mcts_white_plus_rect.collidepoint(ui_pos):
                        mcts_simulations_white = min(2000, mcts_simulations_white + mcts_step)
                        persist_ui_preferences()
                        continue
                    if mcts_black_minus_rect and mcts_black_minus_rect.collidepoint(ui_pos):
                        mcts_simulations_black = max(16, mcts_simulations_black - mcts_step)
                        persist_ui_preferences()
                        continue
                    if mcts_black_plus_rect and mcts_black_plus_rect.collidepoint(ui_pos):
                        mcts_simulations_black = min(2000, mcts_simulations_black + mcts_step)
                        persist_ui_preferences()
                        continue
                    if match_games_minus_rect and match_games_minus_rect.collidepoint(ui_pos):
                        match_games = max(1, match_games - match_games_step)
                        persist_ui_preferences()
                        continue
                    if match_games_plus_rect and match_games_plus_rect.collidepoint(ui_pos):
                        match_games = min(500, match_games + match_games_step)
                        persist_ui_preferences()
                        continue
                    for rect, value in mcts_profile_buttons:
                        if rect.collidepoint(ui_pos):
                            if game_mode == "ai_vs_ai":
                                mcts_simulations_white = int(value)
                                mcts_simulations_black = int(value)
                            else:
                                mcts_simulations = int(value)
                            persist_ui_preferences()
                            break
                    continue

                if workspace_tab == "models" and game_mode == "ai_vs_ai":
                    if white_card_rect and white_card_rect.collidepoint(ui_pos):
                        active_side = "white"
                    elif black_card_rect and black_card_rect.collidepoint(ui_pos):
                        active_side = "black"
                    elif swap_rect and swap_rect.collidepoint(ui_pos):
                        selected_models["white"], selected_models["black"] = (
                            selected_models["black"],
                            selected_models["white"],
                        )
                        continue
                    elif (
                        copy_white_rect
                        and copy_white_rect.collidepoint(ui_pos)
                        and selected_models["white"] is not None
                    ):
                        selected_models["black"] = selected_models["white"]
                        continue
                    elif (
                        copy_black_rect
                        and copy_black_rect.collidepoint(ui_pos)
                        and selected_models["black"] is not None
                    ):
                        selected_models["white"] = selected_models["black"]
                        continue

                    for rect, action, payload in browser_actions:
                        if rect.collidepoint(ui_pos):
                            if action == "toggle":
                                expanded[active_side][payload] = not expanded[active_side][payload]
                            elif action == "select":
                                selected_models[active_side] = payload
                            elif action == "sort":
                                apply_sort(payload)
                            break
                elif workspace_tab == "models" and game_mode == "human_vs_ai":
                    for rect, action, payload in browser_actions:
                        if rect.collidepoint(ui_pos):
                            if action == "toggle":
                                expanded["opponent"][payload] = not expanded["opponent"][payload]
                            elif action == "select":
                                opponent_model = payload
                            elif action == "sort":
                                apply_sort(payload)
                            break

        clock.tick(60)
