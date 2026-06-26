"""Model loading and pre-game setup helpers for local GUI play."""

import copy
import ctypes
from datetime import datetime
import math
from pathlib import Path
import sys

import chess
import pygame
import torch
import yaml

# Add utils to path for model_catalog import
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from utils.shared.model_catalog import load_checkpoint_metadata
from utils.ui.gui_helpers import create_piece_surfaces, start_piece_asset_prefetch


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


_SETUP_BG = (9, 12, 18)
_PANEL_BG = (19, 25, 35)
_CARD_BG = (29, 37, 50)
_TEXT = (246, 249, 255)
_MUTED = (180, 194, 214)
_ACCENT = (82, 155, 255)
_ACCENT_BORDER = (145, 193, 255)
_BORDER = (96, 118, 150)
_DANGER = (177, 83, 83)
_GLOW = (55, 109, 205)
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
def _lerp_color(color_a, color_b, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(color_a[idx] + (color_b[idx] - color_a[idx]) * t) for idx in range(3))


def _draw_panel(screen, rect, fill=None, border=None, radius=18, shadow=True, fill_alpha=None):
    fill = fill or _PANEL_BG
    border = border or _BORDER
    if shadow:
        shadow_rect = rect.move(0, 8)
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 54), shadow_surface.get_rect(), border_radius=radius)
        screen.blit(shadow_surface, shadow_rect.topleft)
    if fill_alpha is None:
        pygame.draw.rect(screen, fill, rect, border_radius=radius)
    else:
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
    
    elo = metadata.get("elo")
    if elo is not None:
        parts.append(f"Elo:{int(round(float(elo)))}")
    
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


def _draw_chip(screen, rect, text, font, tone="neutral"):
    if tone == "accent":
        fill = (46, 84, 136)
        border = (132, 190, 255)
        text_color = (238, 246, 255)
    elif tone == "ok":
        fill = (44, 96, 76)
        border = (118, 199, 159)
        text_color = (234, 249, 240)
    else:
        fill = (35, 44, 58)
        border = (103, 127, 161)
        text_color = (230, 238, 249)
    pygame.draw.rect(screen, fill, rect, border_radius=12)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=12)
    label_text = _fit_text(font, text, max(24, rect.width - 16))
    label = font.render(label_text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_gear_button(screen, rect, hovered=False, active=False, disabled=False):
    if disabled:
        fill = (56, 63, 74)
        border = (82, 92, 107)
        icon = (137, 146, 161)
    elif active:
        fill = _ACCENT
        border = _ACCENT_BORDER
        icon = (245, 250, 255)
    elif hovered:
        fill = (45, 56, 72)
        border = (126, 149, 183)
        icon = (240, 246, 255)
    else:
        fill = (32, 41, 55)
        border = _BORDER
        icon = (223, 232, 246)

    shadow = rect.move(0, 2)
    pygame.draw.rect(screen, (7, 10, 16), shadow, border_radius=10)
    pygame.draw.rect(screen, fill, rect, border_radius=10)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=10)

    cx, cy = rect.center
    radius_outer = max(9, min(rect.width, rect.height) // 2 - 7)
    radius_inner = max(6, radius_outer - 3)
    teeth = 8
    points = []
    for idx in range(teeth * 2):
        angle = -math.pi / 2 + idx * (math.pi / teeth)
        radius = radius_outer if idx % 2 == 0 else radius_inner
        points.append((cx + int(math.cos(angle) * radius), cy + int(math.sin(angle) * radius)))

    pygame.draw.polygon(screen, icon, points)
    hole_r = max(3, radius_inner - 4)
    pygame.draw.circle(screen, fill, (cx, cy), hole_r)
    pygame.draw.circle(screen, icon, (cx, cy), hole_r, width=2)
    pygame.draw.circle(screen, icon, (cx, cy), max(2, hole_r // 3))


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
    if side_color == chess.WHITE:
        fill = (238, 240, 244)
        border = (168, 177, 192)
        title_color = (20, 24, 31)
        meta_color = (58, 65, 78)
        line3_color = (48, 56, 68)
        top_band_fill = (221, 226, 233)
        top_band_text_color = (12, 17, 24)
    elif side_color == chess.BLACK:
        fill = (19, 24, 33)
        border = (86, 101, 126)
        title_color = (241, 245, 251)
        meta_color = (176, 189, 208)
        line3_color = (231, 239, 250)
        top_band_fill = (6, 8, 12)
        top_band_text_color = (244, 247, 252)
    else:
        fill = (22, 29, 39)
        border = _BORDER
        title_color = _TEXT
        meta_color = _MUTED
        line3_color = (216, 228, 244)
        top_band_fill = _lerp_color(fill, border, 0.18)
        top_band_text_color = (214, 226, 244)

    if hovered:
        fill = _lerp_color(fill, (255, 255, 255), 0.06 if side_color == chess.WHITE else 0.08)
        border = _lerp_color(border, _ACCENT_BORDER, 0.35)
    if active:
        border = (134, 188, 255) if side_color != chess.WHITE else (105, 142, 204)
        fill = _lerp_color(fill, (82, 155, 255), 0.10 if side_color == chess.WHITE else 0.18)
        glow_rect = rect.inflate(10, 10)
        glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        glow_color = (82, 155, 255, 56) if side_color != chess.WHITE else (72, 124, 214, 44)
        pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=18)
        screen.blit(glow_surface, glow_rect.topleft)

    _draw_panel(screen, rect, fill=fill, border=border, radius=14, shadow=False)
    if active:
        pygame.draw.rect(screen, _ACCENT_BORDER, rect, width=2, border_radius=14)
    top_band_h = max(34, font_title.get_height() + 14)
    top_band = pygame.Rect(rect.left + 1, rect.top + 1, rect.width - 2, top_band_h)
    pygame.draw.rect(screen, top_band_fill, top_band, border_top_left_radius=14, border_top_right_radius=14)
    band_label = font_title.render(title.upper(), True, top_band_text_color)
    screen.blit(band_label, (rect.left + 16, top_band.centery - band_label.get_height() // 2))
    if active:
        badge_text = font_meta.render("SELECTED", True, (246, 250, 255))
        badge_w = badge_text.get_width() + 18
        badge_h = max(22, badge_text.get_height() + 8)
        badge_rect = pygame.Rect(rect.right - badge_w - 14, top_band.centery - badge_h // 2, badge_w, badge_h)
        badge_fill = (57, 111, 196) if side_color == chess.WHITE else (70, 144, 240)
        pygame.draw.rect(screen, badge_fill, badge_rect, border_radius=badge_h // 2)
        pygame.draw.rect(screen, _ACCENT_BORDER, badge_rect, width=1, border_radius=badge_h // 2)
        screen.blit(badge_text, badge_text.get_rect(center=badge_rect.center))

    if model_path is None:
        line1 = "No model selected"
        line2 = "Choose one from the list below"
        line3 = "Waiting for selection"
    else:
        metadata = (metadata_cache or {}).get(model_path)
        line1, line2 = _format_model_entry(
            model_path,
            models_dir,
            max_len=56,
            metadata=metadata,
        )
        version = metadata.get("version") if isinstance(metadata, dict) else None
        elo = metadata.get("elo") if isinstance(metadata, dict) else None
        size_mb = metadata.get("size_mb") if isinstance(metadata, dict) else None
        line3_parts = []
        if version is not None:
            version = str(version)
            line3_parts.append(version if version.startswith("v") else f"v{version}")
        if elo is not None:
            try:
                line3_parts.append(f"Elo {int(round(float(elo)))}")
            except Exception:
                pass
        if size_mb is not None:
            try:
                line3_parts.append(f"{float(size_mb):.1f} MB")
            except Exception:
                pass
        line3 = "  |  ".join(line3_parts) if line3_parts else "Checkpoint metadata ready"

    content_top = top_band.bottom + 12
    icon_size = max(72, min(104, rect.bottom - content_top - 12))
    icon_rect = pygame.Rect(rect.left + 14, content_top, icon_size, icon_size)
    if side_color in (chess.WHITE, chess.BLACK):
        symbol = "K" if side_color == chess.WHITE else "k"
        piece_surface = _scaled_setup_piece_icon(symbol, icon_size, fill_ratio=0.94)
        if piece_surface is not None:
            screen.blit(piece_surface, piece_surface.get_rect(center=icon_rect.center))

    text_left = icon_rect.right + 12 if side_color in (chess.WHITE, chess.BLACK) else rect.left + 14
    text_right_pad = 14
    text_width = max(40, rect.right - text_left - text_right_pad)
    line1_y = content_top + 2
    line2_y = line1_y + font_text.get_height() + 4
    line3_y = line2_y + font_meta.get_height() + 4
    screen.blit(font_text.render(_fit_text(font_text, line1, text_width), True, title_color), (text_left, line1_y))
    screen.blit(font_meta.render(_fit_text(font_meta, line2, text_width), True, meta_color), (text_left, line2_y))
    screen.blit(font_meta.render(_fit_text(font_meta, line3, text_width), True, line3_color), (text_left, line3_y))


def _selection_card_height(font_label, font_text, font_meta):
    """Resolve selection-card height from actual typography metrics."""
    top_band_h = max(34, font_label.get_height() + 14)
    text_block_h = font_text.get_height() + 4 + font_meta.get_height() + 4 + font_meta.get_height()
    return top_band_h + 12 + max(72, text_block_h + 8) + 12


def _fit_text(font, text, max_width):
    text = str(text)
    if font.size(text)[0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed and font.size(trimmed + suffix)[0] > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + suffix) if trimmed else suffix


def _fmt_float(value, pattern):
    try:
        if value is None:
            return "n/a"
        return pattern.format(float(value))
    except Exception:
        return "n/a"


def _get_model_info_content(model_path, models_dir, metadata_cache):
    if model_path is None:
        return None, None, None, [], [("Elo", "n/a"), ("Top1", "n/a"), ("Epoch", "n/a")]
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
    elo = metadata.get("elo")
    elo_value = f"{int(round(float(elo)))}" if elo is not None else "n/a"
    version = metadata.get("version")
    version_value = f"v{version}" if version and not str(version).startswith("v") else (str(version) if version else "n/a")
    modified = metadata.get("modified") or "n/a"

    rows = [
        ("Version", version_value),
        ("Epoch", epoch_value),
        ("Top1", top1_value),
        ("Val loss", _fmt_float(metadata.get("val_loss"), "{:.4f}")),
        ("Policy", _fmt_float(metadata.get("policy_loss"), "{:.4f}")),
        ("Elo", elo_value),
        ("Size", f"{float(metadata.get('size_mb', 0.0)):.1f} MB"),
        ("SWA", "yes" if metadata.get("swa") else "no"),
        ("Optimizer", "yes" if metadata.get("optimizer") else "no"),
        ("Modified", modified),
    ]

    top_stats = [
        ("Elo", elo_value),
        ("Top1", top1_value),
        ("Epoch", epoch_value),
    ]
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
    header_chip = pygame.Rect(rect.left + 16, rect.top + 14, 108, 22)
    pygame.draw.rect(screen, (18, 24, 33), header_chip, border_radius=11)
    pygame.draw.rect(screen, (83, 104, 132), header_chip, width=1, border_radius=11)
    chip_text = font_meta.render("MODEL INFO", True, (211, 224, 243))
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
        screen.blit(font_meta.render(label, True, (149, 166, 191)), (stat_rect.left + 12, stat_rect.top + 10))
        value_surface = font_h2.render(str(value), True, _TEXT)
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

    header_rect = pygame.Rect(panel_rect.left + pad, panel_rect.top + pad, panel_rect.width - 2 * pad - 10, header_h)
    pygame.draw.rect(screen, (16, 21, 30), header_rect, border_radius=11)
    pygame.draw.rect(screen, (90, 108, 136), header_rect, width=1, border_radius=10)
    metric_gap = 10
    metric_w = {"ver": 56, "top1": 84, "elo": 64}
    metrics_right = header_rect.right - 10
    elo_left = metrics_right - metric_w["elo"]
    top1_left = elo_left - metric_gap - metric_w["top1"]
    ver_left = top1_left - metric_gap - metric_w["ver"]

    model_col_rect = pygame.Rect(header_rect.left + 8, header_rect.top + 4, max(60, ver_left - header_rect.left - 14), header_h - 8)
    ver_col_rect = pygame.Rect(ver_left, header_rect.top + 4, metric_w["ver"], header_h - 8)
    top1_col_rect = pygame.Rect(top1_left, header_rect.top + 4, metric_w["top1"], header_h - 8)
    elo_col_rect = pygame.Rect(elo_left, header_rect.top + 4, metric_w["elo"], header_h - 8)

    def _sort_label(label, key):
        if sort_key != key:
            return label
        return f"{label} {'▼' if sort_desc else '▲'}"

    headers = [
        (_sort_label("Model", "model"), model_col_rect, "model"),
        ("Ver", ver_col_rect, None),
        (_sort_label("Top1", "top1"), top1_col_rect, "top1"),
        (_sort_label("Elo", "elo"), elo_col_rect, "elo"),
    ]
    for label, col_rect, sortable_key in headers:
        color = (196, 210, 232) if sortable_key and sort_key == sortable_key else (166, 181, 205)
        text = font_meta.render(label, True, color)
        screen.blit(text, text.get_rect(center=col_rect.center))
        if sortable_key:
            actions.append((col_rect, "sort", sortable_key))
    for sx in (ver_left - 6, top1_left - 6, elo_left - 6):
        pygame.draw.line(screen, (74, 90, 114), (sx, header_rect.top + 5), (sx, header_rect.bottom - 5), 1)

    # Clip scrolling content so partially-visible rows are cropped cleanly.
    content_clip = pygame.Rect(
        panel_rect.left + pad,
        header_rect.bottom + 6,
        panel_rect.width - 2 * pad - 10,
        max(1, panel_rect.bottom - pad - (header_rect.bottom + 6)),
    )

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
        elo = meta.get("elo")
        if elo is None:
            return "n/a"
        try:
            return str(int(round(float(elo))))
        except Exception:
            return "n/a"

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
                pygame.draw.line(screen, sep_color, (sx, rect.top + 8), (sx, rect.bottom - 8), 1)

            name_x = rect.left + 20
            name_max_w = max(80, row_ver_left - name_x - 12)
            raw_name = _display_name_for_category(path, key)
            name = _fit_text(font_text, raw_name, name_max_w)
            screen.blit(font_text.render(name, True, text_color), (name_x, rect.top + 11))

            ver_text = font_meta.render(ver, True, meta_color)
            top1_text = font_meta.render(top1, True, meta_color)
            elo_text = font_meta.render(elo, True, meta_color)
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

    def _build_background(width, height):
        surf = pygame.Surface((width, height))
        denom = max(1, height - 1)
        top_color = (13, 18, 26)
        bottom_color = (6, 9, 15)
        for y in range(height):
            t = y / denom
            color = _lerp_color(top_color, bottom_color, t)
            pygame.draw.line(surf, color, (0, y), (width, y))

        glow = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*_GLOW, 44), (int(width * 0.18), int(height * 0.12)), int(min(width, height) * 0.24))
        pygame.draw.circle(glow, (40, 166, 132, 22), (int(width * 0.88), int(height * 0.18)), int(min(width, height) * 0.18))
        surf.blit(glow, (0, 0))

        grid_color = (255, 255, 255, 14)
        grid = pygame.Surface((width, height), pygame.SRCALPHA)
        step = 48
        for x in range(0, width, step):
            pygame.draw.line(grid, grid_color, (x, 0), (x, height))
        for y in range(0, height, step):
            pygame.draw.line(grid, grid_color, (0, y), (width, y))
        surf.blit(grid, (0, 0))
        return surf

    canvas = pygame.Surface((base_width, base_height))
    background = _build_background(base_width, base_height)
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
        background = _build_background(base_width, base_height)
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
            page_background = _build_background(base_width, height)
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
        subtitle = font_small.render("Launch a polished match: choose mode, tune settings, assign models.", True, _MUTED)
        hint = font_meta.render(
            "Enter start  |  Esc cancel",
            True,
            _MUTED,
        )
        canvas.blit(title, (content_pad + scale_px(2), scale_px(28)))
        canvas.blit(subtitle, (content_pad + scale_px(4), scale_px(82)))
        canvas.blit(hint, (content_pad + scale_px(4), scale_px(106)))
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
        if game_mode == "ai_vs_ai":
            if use_mcts_white and use_mcts_black:
                ready_label = "Both sides: MCTS"
                ready_tone = "ok"
            elif (not use_mcts_white) and (not use_mcts_black):
                ready_label = "Both sides: Network-only"
                ready_tone = "neutral"
            else:
                ready_label = "Mixed search ready"
                ready_tone = "neutral"
        else:
            ready_label = "MCTS ready" if use_mcts else "Network-only ready"
            ready_tone = "ok" if use_mcts else "neutral"
        ready_chip_w = scale_px(208)
        ready_chip_h = scale_px(28)
        _draw_chip(
            canvas,
            pygame.Rect(base_width - content_pad - ready_chip_w, scale_px(194), ready_chip_w, ready_chip_h),
            ready_label,
            font_meta,
            tone=ready_tone,
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
        mcts_plus_rect = None
        mcts_profile_buttons = []
        mcts_white_toggle_rect = None
        mcts_black_toggle_rect = None
        mcts_white_minus_rect = None
        mcts_white_plus_rect = None
        mcts_black_minus_rect = None
        mcts_black_plus_rect = None
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
                font_meta.render("Settings workspace active. Click the gear again to return to model selection.", True, _MUTED),
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
                f"MCTS: {'ON' if use_mcts else 'OFF'}",
                font_text,
                hovered=mcts_toggle_rect.collidepoint(mouse_pos),
                active=use_mcts,
                disabled=game_mode == "human_vs_human",
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
                    pygame.draw.rect(canvas, (19, 25, 35), value_rect, border_radius=10)
                    pygame.draw.rect(canvas, (94, 112, 140), value_rect, width=1, border_radius=10)
                    value_label = font_h2.render(str(sims_value), True, _TEXT)
                    canvas.blit(value_label, value_label.get_rect(center=value_rect.center))
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
                        mcts_white_plus_rect = plus_rect
                    else:
                        mcts_black_toggle_rect = toggle_rect
                        mcts_black_minus_rect = minus_rect
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
                value_rect = pygame.Rect(settings_panel_rect.left + scale_px(76), controls_y, scale_px(168), scale_px(42))
                mcts_plus_rect = pygame.Rect(settings_panel_rect.left + scale_px(252), controls_y, scale_px(44), scale_px(42))
                _draw_button(
                    canvas,
                    mcts_minus_rect,
                    "-",
                    font_h2,
                    hovered=mcts_minus_rect.collidepoint(mouse_pos),
                )
                pygame.draw.rect(canvas, (19, 25, 35), value_rect, border_radius=10)
                pygame.draw.rect(canvas, (94, 112, 140), value_rect, width=1, border_radius=10)
                value_label = font_h2.render(str(mcts_simulations), True, _TEXT)
                canvas.blit(value_label, value_label.get_rect(center=value_rect.center))
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
                action_labels = ("<-->", "W -> Both", "B -> Both")
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
                    "<-->",
                    font_meta,
                    hovered=swap_rect.collidepoint(mouse_pos),
                )
                _draw_button(
                    canvas,
                    copy_white_rect,
                    "W -> Both",
                    font_meta,
                    hovered=copy_white_rect.collidepoint(mouse_pos),
                    disabled=selected_models["white"] is None,
                )
                _draw_button(
                    canvas,
                    copy_black_rect,
                    "B -> Both",
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
            "Start Game",
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
