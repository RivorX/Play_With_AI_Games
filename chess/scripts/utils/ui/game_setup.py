"""Model loading and pre-game setup helpers for local GUI play."""

from datetime import datetime
from pathlib import Path

import chess
import pygame
import torch
import yaml


_SETUP_BG = (18, 22, 28)
_PANEL_BG = (31, 37, 46)
_CARD_BG = (42, 49, 61)
_TEXT = (236, 241, 248)
_MUTED = (150, 160, 176)
_ACCENT = (79, 137, 224)
_ACCENT_BORDER = (124, 170, 236)
_BORDER = (78, 92, 112)
_DANGER = (170, 76, 76)
_CATEGORY_STYLE = {
    "best": {"label": "BEST models", "fill": (52, 84, 70), "border": (95, 154, 126)},
    "il": {"label": "IL models", "fill": (56, 72, 103), "border": (104, 132, 188)},
    "rl": {"label": "RL models", "fill": (88, 68, 46), "border": (157, 122, 84)},
    "other": {"label": "Other models", "fill": (68, 66, 80), "border": (120, 117, 137)},
}


def load_model_from_checkpoint(checkpoint_path, config, device, model_class):
    """Load a model checkpoint and switch model to eval mode."""
    model = model_class(config).to(device)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model: {checkpoint_path.name}")

        if "val_policy_loss" in checkpoint:
            print(f"  Val Loss: {checkpoint.get('loss', 'N/A'):.4f}")
            print(f"  Policy Loss: {checkpoint['val_policy_loss']:.4f}")
            print(f"  Value Loss: {checkpoint['val_value_loss']:.4f}")
        elif "win_rate" in checkpoint:
            print(f"  Win Rate: {checkpoint['win_rate']:.2%}")
    else:
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
    """Write selected play settings to logs/games and return file path."""
    games_dir = _resolve_games_dir(base_dir, config)
    now = datetime.now()
    output = games_dir / f"play_setup_{now.strftime('%Y%m%d_%H%M%S')}.yaml"

    payload = {
        "timestamp": now.isoformat(timespec="seconds"),
        "game_mode": setup.get("game_mode"),
        "human_color": setup.get("human_color_name"),
        "use_mcts": bool(setup.get("use_mcts", False)),
        "model_ai": str(setup.get("model1_path")) if setup.get("model1_path") else None,
        "model_white": str(setup.get("model_white")) if setup.get("model_white") else None,
        "model_black": str(setup.get("model_black")) if setup.get("model_black") else None,
        "source": "setup_window",
    }

    with open(output, "w", encoding="utf-8", newline="\n") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False, allow_unicode=False)

    return output


def _format_model_entry(path, models_dir, max_len=66):
    rel = str(path.relative_to(models_dir)).replace("\\", "/")
    if len(rel) > max_len:
        rel = "..." + rel[-(max_len - 3) :]
    size_mb = path.stat().st_size / (1024**2)
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return rel, f"{size_mb:.1f} MB | {stamp}"


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
        fill = (54, 60, 69)
        text_color = (118, 127, 141)
        border = (68, 77, 89)
    elif danger:
        fill = _DANGER if active else (145, 68, 68)
        text_color = _TEXT
        border = (197, 95, 95)
    elif active:
        fill = _ACCENT
        text_color = (247, 250, 255)
        border = _ACCENT_BORDER
    elif hovered:
        fill = (56, 66, 80)
        text_color = _TEXT
        border = _BORDER
    else:
        fill = _PANEL_BG
        text_color = _TEXT
        border = _BORDER

    pygame.draw.rect(screen, fill, rect, border_radius=8)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=8)
    label = font.render(text, True, text_color)
    screen.blit(label, label.get_rect(center=rect.center))


def _draw_model_browser(
    screen,
    panel_rect,
    grouped,
    models_dir,
    selected_model,
    expanded,
    scroll_offset,
    font_header,
    font_text,
    font_meta,
    mouse_pos,
):
    actions = []

    pygame.draw.rect(screen, _PANEL_BG, panel_rect, border_radius=10)
    pygame.draw.rect(screen, _BORDER, panel_rect, width=2, border_radius=10)

    pad = 10
    content_y = 0
    visible_keys = [key for key in ("best", "il", "rl", "other") if grouped[key]]

    if not visible_keys:
        empty = font_text.render("No models available in this directory.", True, _MUTED)
        screen.blit(empty, (panel_rect.left + 16, panel_rect.top + 16))
        return actions, 0

    def draw_header(key):
        nonlocal content_y
        style = _CATEGORY_STYLE[key]
        label = style["label"]
        count = len(grouped[key])
        collapsible = key != "best"
        marker = "v" if expanded.get(key, False) else ">"
        header_text = f"{marker} {label} ({count})" if collapsible else f"{label} ({count})"

        row_h = 32
        y = panel_rect.top + pad + content_y - scroll_offset
        rect = pygame.Rect(panel_rect.left + pad, y, panel_rect.width - 2 * pad, row_h)
        if y + row_h >= panel_rect.top + 6 and y <= panel_rect.bottom - 6:
            pygame.draw.rect(screen, style["fill"], rect, border_radius=6)
            pygame.draw.rect(screen, style["border"], rect, width=1, border_radius=6)
            stripe = pygame.Rect(rect.left + 6, rect.top + 4, 5, rect.height - 8)
            pygame.draw.rect(screen, style["border"], stripe, border_radius=2)
            screen.blit(font_header.render(header_text, True, _TEXT), (rect.left + 16, rect.top + 5))
        if collapsible and count > 0:
            actions.append((rect, "toggle", key))
        content_y += row_h + 6

    def draw_model_row(path, indent):
        nonlocal content_y
        row_h = 42
        y = panel_rect.top + pad + content_y - scroll_offset
        rect = pygame.Rect(
            panel_rect.left + pad + indent,
            y,
            panel_rect.width - 2 * pad - indent,
            row_h,
        )

        active = path == selected_model
        if y + row_h >= panel_rect.top + 6 and y <= panel_rect.bottom - 6:
            if active:
                fill = _ACCENT
                border = _ACCENT_BORDER
                text_color = (248, 250, 255)
                meta_color = (226, 236, 255)
            else:
                fill = _CARD_BG if not rect.collidepoint(mouse_pos) else (62, 73, 89)
                border = _BORDER
                text_color = _TEXT
                meta_color = _MUTED

            pygame.draw.rect(screen, fill, rect, border_radius=6)
            pygame.draw.rect(screen, border, rect, width=1, border_radius=6)

            line1, line2 = _format_model_entry(path, models_dir)
            screen.blit(font_text.render(line1, True, text_color), (rect.left + 8, rect.top + 4))
            screen.blit(font_meta.render(line2, True, meta_color), (rect.left + 8, rect.top + 22))

            actions.append((rect, "select", path))

        content_y += row_h + 4

    for key in visible_keys:
        draw_header(key)
        if key == "best" or expanded.get(key, False):
            indent = 10 if key == "best" else 22
            for path in grouped[key]:
                draw_model_row(path, indent=indent)

    content_height = content_y + 8
    viewport = panel_rect.height - 2 * pad
    max_scroll = max(0, content_height - viewport)
    return actions, max_scroll


def _selection_to_result(game_mode, human_color, selected_models, opponent_model, use_mcts):
    if game_mode == "human_vs_human":
        return {
            "game_mode": game_mode,
            "human_color": human_color,
            "human_color_name": "white" if human_color == chess.WHITE else "black",
            "use_mcts": False,
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


def select_models(base_dir, config, default_use_mcts=True):
    """Choose game mode/models from an in-window setup UI.

    Returns:
        dict or None: selection payload used by play.py.
    """
    models_dir, all_models = _discover_models(base_dir, config)
    grouped = _group_models(all_models)
    has_models = len(all_models) > 0

    screen = pygame.display.set_mode((1060, 760))
    pygame.display.set_caption("Chess AI Setup")

    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("Segoe UI", 40, bold=True)
    font_h2 = pygame.font.SysFont("Segoe UI", 24, bold=True)
    font_text = pygame.font.SysFont("Segoe UI", 19)
    font_meta = pygame.font.SysFont("Segoe UI", 15)
    font_small = pygame.font.SysFont("Segoe UI", 17)

    game_mode = "human_vs_ai" if has_models else "human_vs_human"
    human_color = chess.WHITE
    use_mcts = bool(default_use_mcts and has_models)

    default_model = _default_model(grouped)
    selected_models = {
        "white": default_model,
        "black": default_model,
    }
    opponent_model = default_model

    expanded = {
        "white": {"il": False, "rl": False, "other": False},
        "black": {"il": False, "rl": False, "other": False},
        "opponent": {"il": False, "rl": False, "other": False},
    }
    scroll = {"white": 0, "black": 0, "opponent": 0}
    active_tab = "white"

    browser_rect = pygame.Rect(36, 288, 988, 382)
    browser_actions = []
    browser_max_scroll = 0

    while True:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(_SETUP_BG)

        title = font_title.render("Chess AI Setup", True, _TEXT)
        subtitle = font_small.render("Select mode, models and search settings", True, _MUTED)
        screen.blit(title, (34, 16))
        screen.blit(subtitle, (36, 66))

        mode_rects = {
            "human_vs_ai": pygame.Rect(36, 108, 230, 50),
            "ai_vs_ai": pygame.Rect(280, 108, 230, 50),
            "human_vs_human": pygame.Rect(524, 108, 230, 50),
        }
        _draw_button(
            screen,
            mode_rects["human_vs_ai"],
            "Human vs AI",
            font_text,
            hovered=mode_rects["human_vs_ai"].collidepoint(mouse_pos),
            active=game_mode == "human_vs_ai",
            disabled=not has_models,
        )
        _draw_button(
            screen,
            mode_rects["ai_vs_ai"],
            "AI vs AI",
            font_text,
            hovered=mode_rects["ai_vs_ai"].collidepoint(mouse_pos),
            active=game_mode == "ai_vs_ai",
            disabled=not has_models,
        )
        _draw_button(
            screen,
            mode_rects["human_vs_human"],
            "Human vs Human",
            font_text,
            hovered=mode_rects["human_vs_human"].collidepoint(mouse_pos),
            active=game_mode == "human_vs_human",
        )

        mcts_label = "MCTS: ON" if use_mcts else "MCTS: OFF"
        mcts_rect = pygame.Rect(772, 108, 252, 50)
        _draw_button(
            screen,
            mcts_rect,
            mcts_label,
            font_text,
            hovered=mcts_rect.collidepoint(mouse_pos),
            active=use_mcts,
            disabled=game_mode == "human_vs_human",
        )

        color_white_rect = pygame.Rect(36, 176, 160, 44)
        color_black_rect = pygame.Rect(206, 176, 160, 44)
        if game_mode == "human_vs_ai":
            screen.blit(font_h2.render("Human color", True, _TEXT), (36, 148))
            _draw_button(
                screen,
                color_white_rect,
                "White",
                font_text,
                hovered=color_white_rect.collidepoint(mouse_pos),
                active=human_color == chess.WHITE,
            )
            _draw_button(
                screen,
                color_black_rect,
                "Black",
                font_text,
                hovered=color_black_rect.collidepoint(mouse_pos),
                active=human_color == chess.BLACK,
            )

        tab_white = None
        tab_black = None
        browser_key = None

        if game_mode != "human_vs_human":
            if game_mode == "ai_vs_ai":
                tab_white = pygame.Rect(36, 238, 180, 40)
                tab_black = pygame.Rect(226, 238, 180, 40)
                _draw_button(
                    screen,
                    tab_white,
                    "White model",
                    font_text,
                    hovered=tab_white.collidepoint(mouse_pos),
                    active=active_tab == "white",
                )
                _draw_button(
                    screen,
                    tab_black,
                    "Black model",
                    font_text,
                    hovered=tab_black.collidepoint(mouse_pos),
                    active=active_tab == "black",
                )
                browser_key = active_tab
            else:
                ai_side_name = "Black" if human_color == chess.WHITE else "White"
                header = f"Opponent model ({ai_side_name} AI)"
                screen.blit(font_h2.render(header, True, _TEXT), (36, 244))
                browser_key = "opponent"

            note_text = "Best models are always visible. IL/RL/Other are collapsed by default."
            screen.blit(font_small.render(note_text, True, _MUTED), (422, 249))

            if game_mode == "human_vs_ai":
                side_note = "Only the opponent model is selected in this mode."
                screen.blit(font_small.render(side_note, True, _MUTED), (36, 678))

            selected_model = selected_models[active_tab] if browser_key != "opponent" else opponent_model
            browser_actions, browser_max_scroll = _draw_model_browser(
                screen=screen,
                panel_rect=browser_rect,
                grouped=grouped,
                models_dir=models_dir,
                selected_model=selected_model,
                expanded=expanded[browser_key],
                scroll_offset=scroll[browser_key],
                font_header=font_text,
                font_text=font_text,
                font_meta=font_meta,
                mouse_pos=mouse_pos,
            )

            if scroll[browser_key] > browser_max_scroll:
                scroll[browser_key] = browser_max_scroll

        if not has_models:
            warn = "No model files found. Only Human vs Human is available."
            screen.blit(font_small.render(warn, True, (224, 176, 108)), (36, 676))

        can_start = _can_start(game_mode, has_models, selected_models, opponent_model)
        start_rect = pygame.Rect(724, 702, 144, 42)
        cancel_rect = pygame.Rect(880, 702, 144, 42)
        _draw_button(
            screen,
            start_rect,
            "Start",
            font_text,
            hovered=start_rect.collidepoint(mouse_pos),
            active=can_start,
            disabled=not can_start,
        )
        _draw_button(
            screen,
            cancel_rect,
            "Cancel",
            font_text,
            hovered=cancel_rect.collidepoint(mouse_pos),
            danger=True,
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN and can_start:
                    return _selection_to_result(
                        game_mode,
                        human_color,
                        selected_models,
                        opponent_model,
                        use_mcts,
                    )

            if event.type == pygame.MOUSEWHEEL and game_mode != "human_vs_human":
                if browser_rect.collidepoint(mouse_pos):
                    scroll[browser_key] -= event.y * 30
                    if scroll[browser_key] < 0:
                        scroll[browser_key] = 0
                    if scroll[browser_key] > browser_max_scroll:
                        scroll[browser_key] = browser_max_scroll

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_rect.collidepoint(event.pos):
                    return None

                if start_rect.collidepoint(event.pos) and can_start:
                    return _selection_to_result(
                        game_mode,
                        human_color,
                        selected_models,
                        opponent_model,
                        use_mcts,
                    )

                if mode_rects["human_vs_human"].collidepoint(event.pos):
                    game_mode = "human_vs_human"
                elif has_models and mode_rects["human_vs_ai"].collidepoint(event.pos):
                    game_mode = "human_vs_ai"
                elif has_models and mode_rects["ai_vs_ai"].collidepoint(event.pos):
                    game_mode = "ai_vs_ai"

                if mcts_rect.collidepoint(event.pos) and game_mode != "human_vs_human":
                    use_mcts = not use_mcts

                if game_mode == "human_vs_ai":
                    if color_white_rect.collidepoint(event.pos):
                        human_color = chess.WHITE
                    elif color_black_rect.collidepoint(event.pos):
                        human_color = chess.BLACK

                if game_mode == "ai_vs_ai":
                    if tab_white and tab_white.collidepoint(event.pos):
                        active_tab = "white"
                    elif tab_black and tab_black.collidepoint(event.pos):
                        active_tab = "black"

                    for rect, action, payload in browser_actions:
                        if rect.collidepoint(event.pos):
                            if action == "toggle":
                                expanded[active_tab][payload] = not expanded[active_tab][payload]
                            elif action == "select":
                                selected_models[active_tab] = payload
                            break
                elif game_mode == "human_vs_ai":
                    for rect, action, payload in browser_actions:
                        if rect.collidepoint(event.pos):
                            if action == "toggle":
                                expanded["opponent"][payload] = not expanded["opponent"][payload]
                            elif action == "select":
                                opponent_model = payload
                            break

        clock.tick(60)
