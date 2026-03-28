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


_SETUP_BG = (12, 16, 22)
_PANEL_BG = (24, 31, 42)
_CARD_BG = (32, 40, 54)
_TEXT = (241, 245, 251)
_MUTED = (151, 165, 184)
_ACCENT = (82, 155, 255)
_ACCENT_BORDER = (145, 193, 255)
_BORDER = (78, 96, 122)
_DANGER = (177, 83, 83)
_GLOW = (43, 92, 173)
_CATEGORY_STYLE = {
    "best": {"label": "BEST models", "fill": (44, 82, 69), "border": (99, 182, 140)},
    "il": {"label": "IL models", "fill": (44, 64, 98), "border": (108, 149, 221)},
    "rl": {"label": "RL models", "fill": (89, 67, 42), "border": (191, 145, 87)},
    "other": {"label": "Other models", "fill": (62, 61, 77), "border": (135, 137, 162)},
}


def _maximize_native_window():
    """Ask the OS to maximize the current window when available."""
    try:
        wm_info = pygame.display.get_wm_info()
    except Exception:
        return False

    window_handle = wm_info.get("window")
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
            elif "win_rate" in checkpoint:
                print(f"  Win Rate: {checkpoint['win_rate']:.2%}")
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
        name = model_path.name.lower()
        parts = [part.lower() for part in model_path.parts]

        if "best_model" in name:
            grouped["best"].append(model_path)
        elif "il" in parts:
            grouped["il"].append(model_path)
        elif "rl" in parts:
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
        fill = (46, 53, 64)
        text_color = (114, 124, 139)
        border = (66, 76, 92)
    elif danger:
        fill = _DANGER if active else (144, 73, 73)
        text_color = _TEXT
        border = (214, 121, 121)
    elif active:
        fill = _ACCENT
        text_color = (247, 250, 255)
        border = _ACCENT_BORDER
    elif hovered:
        fill = (41, 50, 64)
        text_color = _TEXT
        border = (109, 128, 157)
    else:
        fill = (30, 38, 50)
        text_color = _TEXT
        border = _BORDER

    shadow = rect.move(0, 2)
    pygame.draw.rect(screen, (9, 12, 18), shadow, border_radius=10)
    pygame.draw.rect(screen, fill, rect, border_radius=10)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=10)
    label = font.render(text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_chip(screen, rect, text, font, tone="neutral"):
    if tone == "accent":
        fill = (42, 77, 124)
        border = (116, 177, 255)
        text_color = (229, 241, 255)
    elif tone == "ok":
        fill = (43, 89, 72)
        border = (106, 184, 147)
        text_color = (227, 246, 236)
    else:
        fill = (32, 39, 52)
        border = (87, 108, 138)
        text_color = (220, 228, 241)
    pygame.draw.rect(screen, fill, rect, border_radius=12)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=12)
    label = font.render(text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_gear_button(screen, rect, hovered=False, active=False, disabled=False):
    if disabled:
        fill = (54, 60, 69)
        border = (68, 77, 89)
        icon = (118, 127, 141)
    elif active:
        fill = _ACCENT
        border = _ACCENT_BORDER
        icon = (245, 250, 255)
    elif hovered:
        fill = (41, 50, 64)
        border = (109, 128, 157)
        icon = (234, 241, 252)
    else:
        fill = (30, 38, 50)
        border = _BORDER
        icon = (210, 221, 240)

    shadow = rect.move(0, 2)
    pygame.draw.rect(screen, (9, 12, 18), shadow, border_radius=10)
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
):
    if active:
        fill = (28, 49, 78)
        border = (134, 188, 255)
        title_color = (242, 248, 255)
    elif hovered:
        fill = (29, 38, 50)
        border = (102, 123, 151)
        title_color = _TEXT
    else:
        fill = (22, 29, 39)
        border = _BORDER
        title_color = _TEXT

    _draw_panel(screen, rect, fill=fill, border=border, radius=14, shadow=False)
    top_band_h = max(24, font_meta.get_height() + 10)
    top_band = pygame.Rect(rect.left + 1, rect.top + 1, rect.width - 2, top_band_h)
    pygame.draw.rect(screen, _lerp_color(fill, border, 0.18), top_band, border_top_left_radius=14, border_top_right_radius=14)
    chip_text = title.upper()
    chip_pad_x = 12
    chip_h = max(20, font_meta.get_height() + 8)
    chip_w = min(rect.width - 28, max(96, font_meta.size(chip_text)[0] + chip_pad_x * 2))
    label_chip = pygame.Rect(rect.left + 14, top_band.bottom + 8, chip_w, chip_h)
    pygame.draw.rect(screen, (18, 24, 33), label_chip, border_radius=10)
    pygame.draw.rect(screen, _lerp_color(border, (255, 255, 255), 0.15), label_chip, width=1, border_radius=10)
    chip_label = font_meta.render(chip_text, True, (208, 222, 242))
    screen.blit(chip_label, chip_label.get_rect(center=label_chip.center))

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

    text_left = rect.left + 14
    text_right_pad = 14
    text_width = max(40, rect.right - text_left - text_right_pad)
    line1_y = label_chip.bottom + 10
    line2_y = line1_y + font_text.get_height() + 4
    line3_y = line2_y + font_meta.get_height() + 4
    screen.blit(font_text.render(_fit_text(font_text, line1, text_width), True, title_color), (text_left, line1_y))
    screen.blit(font_meta.render(_fit_text(font_meta, line2, text_width), True, _MUTED), (text_left, line2_y))
    screen.blit(font_meta.render(_fit_text(font_meta, line3, text_width), True, (216, 228, 244)), (text_left, line3_y))


def _selection_card_height(font_text, font_meta):
    """Resolve selection-card height from actual typography metrics."""
    top_band_h = max(24, font_meta.get_height() + 10)
    chip_h = max(20, font_meta.get_height() + 8)
    return top_band_h + 8 + chip_h + 10 + font_text.get_height() + 4 + font_meta.get_height() + 4 + font_meta.get_height() + 14


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

    metadata = metadata_cache.get(model_path)
    if metadata is None:
        metadata = load_checkpoint_metadata(model_path, models_dir)
        metadata_cache[model_path] = metadata

    name = metadata.get("model_name") or model_path.name
    rel = metadata.get("path_rel") or _short_model_path(model_path, models_dir, max_len=64)
    title_y = rect.top + 48
    screen.blit(font_text.render(_fit_text(font_text, name, rect.width - 32), True, _TEXT), (rect.left + 16, title_y))
    screen.blit(font_meta.render(_fit_text(font_meta, rel, rect.width - 32), True, _MUTED), (rect.left + 16, title_y + 24))

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
    stat_gap = 10
    stat_w = (rect.width - 32 - stat_gap * 2) // 3
    stat_y = rect.top + 98
    for idx, (label, value) in enumerate(top_stats):
        stat_rect = pygame.Rect(rect.left + 16 + idx * (stat_w + stat_gap), stat_y, stat_w, 52)
        pygame.draw.rect(screen, (18, 24, 33), stat_rect, border_radius=12)
        pygame.draw.rect(screen, (70, 88, 113), stat_rect, width=1, border_radius=12)
        screen.blit(font_meta.render(label, True, (149, 166, 191)), (stat_rect.left + 12, stat_rect.top + 10))
        value_surface = font_h2.render(str(value), True, _TEXT)
        screen.blit(value_surface, (stat_rect.left + 12, stat_rect.top + 24))

    table_rect = pygame.Rect(rect.left + 12, rect.top + 164, rect.width - 24, rect.height - 178)
    pygame.draw.rect(screen, (19, 24, 33), table_rect, border_radius=12)
    pygame.draw.rect(screen, (67, 83, 108), table_rect, width=1, border_radius=12)

    row_h = 26
    y = table_rect.top + 8
    label_w = 96
    for idx, (label, value) in enumerate(rows):
        if y + row_h > table_rect.bottom - 4:
            break
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

    base_width = 1360
    base_height = 900
    min_width = 980
    min_height = 660
    display_flags = pygame.RESIZABLE

    def _get_display_window_size():
        try:
            info = pygame.display.Info()
            screen_width = int(getattr(info, "current_w", 0) or 0)
            screen_height = int(getattr(info, "current_h", 0) or 0)
        except Exception:
            screen_width = 0
            screen_height = 0

        if screen_width <= 0 or screen_height <= 0:
            return base_width, base_height

        return (
            max(min_width, screen_width),
            max(min_height, screen_height),
        )

    start_width = base_width
    start_height = base_height
    if isinstance(initial_window_size, (list, tuple)) and len(initial_window_size) == 2:
        iw = _safe_int(initial_window_size[0], default=base_width)
        ih = _safe_int(initial_window_size[1], default=base_height)
        start_width = max(min_width, int(iw or base_width))
        start_height = max(min_height, int(ih or base_height))

    screen = pygame.display.set_mode((start_width, start_height), display_flags)
    pygame.display.set_caption("Chess AI Setup")
    try:
        pygame.display.set_window_minimum_size((min_width, min_height))
    except Exception:
        pass

    def _build_background(width, height):
        surf = pygame.Surface((width, height))
        denom = max(1, height - 1)
        top_color = (15, 20, 28)
        bottom_color = (8, 11, 17)
        for y in range(height):
            t = y / denom
            color = _lerp_color(top_color, bottom_color, t)
            pygame.draw.line(surf, color, (0, y), (width, y))

        glow = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*_GLOW, 44), (int(width * 0.18), int(height * 0.12)), int(min(width, height) * 0.24))
        pygame.draw.circle(glow, (40, 166, 132, 22), (int(width * 0.88), int(height * 0.18)), int(min(width, height) * 0.18))
        surf.blit(glow, (0, 0))

        grid_color = (255, 255, 255, 10)
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

    viewport_rect = pygame.Rect(0, 0, base_width, base_height)
    viewport_scale_x = 1.0
    viewport_scale_y = 1.0
    maximized = False
    restore_window_size = (start_width, start_height)

    def resize_canvas(width, height):
        nonlocal base_width, base_height, canvas, background
        base_width = max(min_width, int(width))
        base_height = max(min_height, int(height))
        canvas = pygame.Surface((base_width, base_height))
        background = _build_background(base_width, base_height)

    def update_viewport():
        nonlocal viewport_rect, viewport_scale_x, viewport_scale_y
        win_w, win_h = screen.get_size()
        viewport_rect = pygame.Rect(0, 0, win_w, win_h)
        viewport_scale_x = win_w / max(1, base_width)
        viewport_scale_y = win_h / max(1, base_height)

    def window_to_ui(pos):
        if not viewport_rect.collidepoint(pos):
            return None
        x = int((pos[0] - viewport_rect.left) / max(0.0001, viewport_scale_x))
        y = int((pos[1] - viewport_rect.top) / max(0.0001, viewport_scale_y))
        x = max(0, min(base_width - 1, x))
        y = max(0, min(base_height - 1, y))
        return x, y

    def set_window_size(width, height):
        nonlocal screen, maximized, restore_window_size
        width = max(min_width, int(width))
        height = max(min_height, int(height))
        screen = pygame.display.set_mode((width, height), display_flags)
        restore_window_size = (width, height)
        maximized = False
        resize_canvas(width, height)
        update_viewport()

    def toggle_maximized():
        nonlocal screen, maximized, restore_window_size
        if not maximized:
            restore_window_size = screen.get_size()
            screen = pygame.display.set_mode(_get_display_window_size(), display_flags)
            pygame.event.pump()
            _maximize_native_window()
            maximized = True
        else:
            screen = pygame.display.set_mode(restore_window_size, display_flags)
            maximized = False
        resize_canvas(*screen.get_size())
        update_viewport()

    resize_canvas(*screen.get_size())
    update_viewport()
    if bool(initial_window_maximized):
        toggle_maximized()

    def _with_window_state(payload):
        if payload is None:
            return None
        payload["window_size"] = [int(screen.get_width()), int(screen.get_height())]
        payload["window_maximized"] = bool(maximized)
        return payload

    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("Segoe UI", 46, bold=True)
    font_h2 = pygame.font.SysFont("Segoe UI", 24, bold=True)
    font_text = pygame.font.SysFont("Segoe UI", 19)
    font_meta = pygame.font.SysFont("Segoe UI", 15)
    font_small = pygame.font.SysFont("Segoe UI", 17)
    font_tiny = pygame.font.SysFont("Segoe UI", 14)

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
        canvas.blit(background, (0, 0))

        hero_rect = pygame.Rect(24, 16, base_width - 48, 118)
        _draw_panel(canvas, hero_rect, fill=(22, 29, 39), border=(84, 103, 131), radius=22, shadow=True)
        hero_band = pygame.Rect(hero_rect.left, hero_rect.top, hero_rect.width, 10)
        pygame.draw.rect(canvas, _ACCENT, hero_band, border_top_left_radius=22, border_top_right_radius=22)
        title = font_title.render("Chess AI Setup", True, _TEXT)
        subtitle = font_small.render("Launch a polished match: choose mode, tune settings, assign models.", True, _MUTED)
        hint = font_meta.render(
            "Enter start  |  Esc cancel  |  F11 maximize/restore",
            True,
            _MUTED,
        )
        canvas.blit(title, (42, 28))
        canvas.blit(subtitle, (44, 82))
        canvas.blit(hint, (44, 106))
        settings_icon_rect = pygame.Rect(base_width - 176, 30, 136, 38)
        _draw_button(
            canvas,
            settings_icon_rect,
            "Settings",
            font_meta,
            hovered=settings_icon_rect.collidepoint(mouse_pos),
            active=workspace_tab == "settings",
        )

        mode_label = font_small.render("Mode", True, _MUTED)
        canvas.blit(mode_label, (40, 156))

        mode_gap = 14
        mode_btn_w = 232
        mode_rects = {
            "human_vs_ai": pygame.Rect(40, 182, mode_btn_w, 52),
            "ai_vs_ai": pygame.Rect(40 + mode_btn_w + mode_gap, 182, mode_btn_w, 52),
            "human_vs_human": pygame.Rect(40 + (mode_btn_w + mode_gap) * 2, 182, mode_btn_w, 52),
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
        _draw_chip(
            canvas,
            pygame.Rect(base_width - 248, 194, 208, 28),
            ready_label,
            font_meta,
            tone=ready_tone,
        )

        color_white_rect = pygame.Rect(40, 274, 172, 44)
        color_black_rect = pygame.Rect(222, 274, 172, 44)
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
        content_top = 356
        content_bottom = base_height - 92
        content_height = max(180, content_bottom - content_top)
        browser_key = None
        browser_actions = []
        browser_max_scroll = 0

        if game_mode == "human_vs_ai":
            canvas.blit(font_h2.render("Human color", True, _TEXT), (40, 246))
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

        workspace_label_y = 258 if game_mode != "human_vs_ai" else 340
        tabs_y = workspace_label_y + 18
        if workspace_tab == "settings":
            canvas.blit(
                font_meta.render("Settings workspace active. Click the gear again to return to model selection.", True, _MUTED),
                (40, tabs_y + 10),
            )
        content_top = tabs_y + 46
        content_bottom = base_height - 92
        content_height = max(180, content_bottom - content_top)
        workspace_rect = pygame.Rect(24, content_top - 18, base_width - 48, content_height + 28)
        _draw_panel(canvas, workspace_rect, fill=(19, 26, 35), border=(61, 76, 98), radius=22, shadow=True, fill_alpha=176)

        if workspace_tab == "settings":
            settings_panel_rect = pygame.Rect(40, content_top, base_width - 80, content_height - 10)
            _draw_panel(canvas, settings_panel_rect, fill=(24, 31, 42), border=(78, 100, 128), radius=18, shadow=False, fill_alpha=208)

            canvas.blit(font_h2.render("Search & Runtime Settings", True, _TEXT), (settings_panel_rect.left + 24, settings_panel_rect.top + 20))
            canvas.blit(
                font_small.render("Dial in the engine once here and keep the rest of the setup clean.", True, _MUTED),
                (settings_panel_rect.left + 24, settings_panel_rect.top + 52),
            )

            mcts_toggle_rect = pygame.Rect(settings_panel_rect.left + 24, settings_panel_rect.top + 86, 230, 46)
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
                    (mcts_toggle_rect.right + 16, mcts_toggle_rect.top + 14),
                )

            sims_title_y = settings_panel_rect.top + 150
            if game_mode == "ai_vs_ai":
                canvas.blit(font_small.render("MCTS simulations per side", True, _TEXT), (settings_panel_rect.left + 24, sims_title_y))
                canvas.blit(font_meta.render("White and Black can use different search modes and budgets.", True, _MUTED), (settings_panel_rect.left + 24, sims_title_y + 24))

                controls_y = settings_panel_rect.top + 198
                card_gap = 18
                card_w = (settings_panel_rect.width - 48 - card_gap) // 2
                white_card = pygame.Rect(settings_panel_rect.left + 24, controls_y, card_w, 146)
                black_card = pygame.Rect(white_card.right + card_gap, controls_y, card_w, 146)
                for card_rect, label_text, sims_value, side_uses_mcts, is_white in (
                    (white_card, "White search", mcts_simulations_white, use_mcts_white, True),
                    (black_card, "Black search", mcts_simulations_black, use_mcts_black, False),
                ):
                    pygame.draw.rect(canvas, (18, 24, 33), card_rect, border_radius=12)
                    pygame.draw.rect(canvas, (70, 84, 104), card_rect, width=1, border_radius=12)
                    canvas.blit(font_small.render(label_text, True, _TEXT), (card_rect.left + 14, card_rect.top + 12))
                    toggle_rect = pygame.Rect(card_rect.left + 14, card_rect.top + 42, card_rect.width - 28, 34)
                    _draw_button(
                        canvas,
                        toggle_rect,
                        f"MCTS: {'ON' if side_uses_mcts else 'OFF'}",
                        font_meta,
                        hovered=toggle_rect.collidepoint(mouse_pos),
                        active=side_uses_mcts,
                    )
                    controls_row_y = card_rect.top + 90
                    minus_rect = pygame.Rect(card_rect.left + 14, controls_row_y, 40, 38)
                    value_rect = pygame.Rect(card_rect.left + 62, controls_row_y, card_rect.width - 124, 38)
                    plus_rect = pygame.Rect(card_rect.right - 54, controls_row_y, 40, 38)
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
                preset_y = controls_y + 160
            else:
                canvas.blit(font_small.render("MCTS simulations per move", True, _TEXT), (settings_panel_rect.left + 24, sims_title_y))
                profile_text = "Speed profile: Fast" if mcts_simulations <= 80 else (
                    "Speed profile: Balanced" if mcts_simulations <= 140 else (
                        "Speed profile: Strong" if mcts_simulations <= 300 else "Speed profile: Ultra"
                    )
                )
                canvas.blit(font_meta.render(profile_text, True, _MUTED), (settings_panel_rect.left + 24, sims_title_y + 24))

                controls_y = settings_panel_rect.top + 198
                mcts_minus_rect = pygame.Rect(settings_panel_rect.left + 24, controls_y, 44, 42)
                value_rect = pygame.Rect(settings_panel_rect.left + 76, controls_y, 168, 42)
                mcts_plus_rect = pygame.Rect(settings_panel_rect.left + 252, controls_y, 44, 42)
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
                preset_y = controls_y + 62

            preset_gap = 12
            preset_width = (settings_panel_rect.width - 48 - preset_gap * (len(mcts_profiles) - 1)) // len(mcts_profiles)
            for idx, (label, value) in enumerate(mcts_profiles):
                rect = pygame.Rect(
                    settings_panel_rect.left + 24 + idx * (preset_width + preset_gap),
                    preset_y,
                    preset_width,
                    38,
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
            info_box = pygame.Rect(settings_panel_rect.left + 24, preset_y + 56, settings_panel_rect.width - 48, 92)
            pygame.draw.rect(canvas, (18, 24, 33), info_box, border_radius=12)
            pygame.draw.rect(canvas, (70, 84, 104), info_box, width=1, border_radius=12)
            y_row = info_box.top + 12
            for row in help_rows:
                canvas.blit(font_meta.render(row, True, _MUTED), (info_box.left + 12, y_row))
                y_row += 24
        elif game_mode != "human_vs_human":
            if game_mode == "ai_vs_ai":
                canvas.blit(font_h2.render("Model Assignment", True, _TEXT), (40, content_top - 34))
                cards_gap = 14
                card_h = _selection_card_height(font_small, font_tiny)
                card_w = (base_width - 80 - cards_gap) // 2
                white_card_rect = pygame.Rect(40, content_top, card_w, card_h)
                black_card_rect = pygame.Rect(40 + card_w + cards_gap, content_top, card_w, card_h)
                _draw_selection_card(
                    canvas,
                    white_card_rect,
                    "White model",
                    selected_models["white"],
                    models_dir,
                    font_small,
                    font_small,
                    font_tiny,
                    metadata_cache=metadata_cache,
                    active=active_side == "white",
                    hovered=white_card_rect.collidepoint(mouse_pos),
                )
                _draw_selection_card(
                    canvas,
                    black_card_rect,
                    "Black model",
                    selected_models["black"],
                    models_dir,
                    font_small,
                    font_small,
                    font_tiny,
                    metadata_cache=metadata_cache,
                    active=active_side == "black",
                    hovered=black_card_rect.collidepoint(mouse_pos),
                )

                active_text = "Now choosing for: WHITE" if active_side == "white" else "Now choosing for: BLACK"
                active_text_y = white_card_rect.bottom + 12
                canvas.blit(font_small.render(active_text, True, (184, 220, 255)), (40, active_text_y))

                btn_y = white_card_rect.bottom + 8
                copy_black_rect = pygame.Rect(base_width - 214, btn_y, 174, 36)
                copy_white_rect = pygame.Rect(copy_black_rect.left - 172, btn_y, 166, 34)
                swap_rect = pygame.Rect(copy_white_rect.left - 146, btn_y, 136, 34)
                _draw_button(
                    canvas,
                    swap_rect,
                    "Swap W/B",
                    font_meta,
                    hovered=swap_rect.collidepoint(mouse_pos),
                )
                _draw_button(
                    canvas,
                    copy_white_rect,
                    "White -> Both",
                    font_meta,
                    hovered=copy_white_rect.collidepoint(mouse_pos),
                    disabled=selected_models["white"] is None,
                )
                _draw_button(
                    canvas,
                    copy_black_rect,
                    "Black -> Both",
                    font_meta,
                    hovered=copy_black_rect.collidepoint(mouse_pos),
                    disabled=selected_models["black"] is None,
                )

                browser_key = active_side
                browser_top = white_card_rect.bottom + 54
                browser_h = max(180, content_bottom - browser_top)
                browser_w = int((base_width - 80) * 0.67)
                info_w = base_width - 80 - browser_w - 16
                browser_rect = pygame.Rect(40, browser_top, browser_w, browser_h)
                info_rect = pygame.Rect(browser_rect.right + 12, browser_top, info_w, browser_h)
            else:
                ai_side_name = "Black" if human_color == chess.WHITE else "White"
                header = f"Opponent model ({ai_side_name} AI side)"
                canvas.blit(font_h2.render(header, True, _TEXT), (40, content_top - 34))
                opponent_card_rect = pygame.Rect(40, content_top, base_width - 80, _selection_card_height(font_small, font_tiny))
                _draw_selection_card(
                    canvas,
                    opponent_card_rect,
                    "Selected opponent",
                    opponent_model,
                    models_dir,
                    font_small,
                    font_small,
                    font_tiny,
                    metadata_cache=metadata_cache,
                    active=True,
                    hovered=opponent_card_rect.collidepoint(mouse_pos),
                )
                browser_key = "opponent"
                browser_top = opponent_card_rect.bottom + 20
                browser_h = max(180, content_bottom - browser_top)
                browser_w = int((base_width - 80) * 0.67)
                info_w = base_width - 80 - browser_w - 16
                browser_rect = pygame.Rect(40, browser_top, browser_w, browser_h)
                info_rect = pygame.Rect(browser_rect.right + 12, browser_top, info_w, browser_h)

            selected_model = selected_models[active_side] if browser_key != "opponent" else opponent_model
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
            info_box = pygame.Rect(40, content_top, base_width - 80, 142)
            _draw_panel(canvas, info_box, fill=(24, 31, 42), border=_BORDER, radius=18, shadow=False, fill_alpha=204)
            canvas.blit(font_h2.render("Human vs Human", True, _TEXT), (60, info_box.top + 24))
            canvas.blit(
                font_small.render("No model selection required. Press Start to open the board.", True, _MUTED),
                (60, info_box.top + 68),
            )
            canvas.blit(
                font_meta.render("Open Settings if you want to preconfigure MCTS before switching back to AI modes.", True, _MUTED),
                (60, info_box.top + 98),
            )

        if not has_models:
            warn = "No model files found. Only Human vs Human is available."
            canvas.blit(font_small.render(warn, True, (224, 176, 108)), (40, base_height - 126))

        can_start = _can_start(game_mode, has_models, selected_models, opponent_model)

        footer_rect = pygame.Rect(24, base_height - 78, base_width - 48, 54)
        _draw_panel(canvas, footer_rect, fill=(17, 23, 31), border=(56, 72, 93), radius=18, shadow=False, fill_alpha=210)
        footer_hint = "Selected setup is ready to launch." if can_start else "Complete the required selections to start."
        footer_color = (184, 220, 255) if can_start else _MUTED
        canvas.blit(font_meta.render(footer_hint, True, footer_color), (40, footer_rect.top + 18))

        start_rect = pygame.Rect(base_width - 342, base_height - 70, 156, 40)
        cancel_rect = pygame.Rect(base_width - 176, base_height - 70, 136, 40)
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
        if viewport_rect.size == (base_width, base_height):
            screen.blit(canvas, viewport_rect.topleft)
        else:
            scaled = pygame.transform.smoothscale(canvas, viewport_rect.size)
            screen.blit(scaled, viewport_rect.topleft)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                persist_ui_preferences(force=True)
                return None

            if event.type == pygame.VIDEORESIZE:
                set_window_size(event.w, event.h)
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    persist_ui_preferences(force=True)
                    return None
                if event.key == pygame.K_F11:
                    toggle_maximized()
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

            if event.type == pygame.MOUSEWHEEL and workspace_tab == "models" and game_mode != "human_vs_human":
                if browser_key and browser_rect.collidepoint(mouse_pos):
                    scroll[browser_key] -= event.y * 30
                    if scroll[browser_key] < 0:
                        scroll[browser_key] = 0
                    if scroll[browser_key] > browser_max_scroll:
                        scroll[browser_key] = browser_max_scroll

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
