"""Responsive geometry for the local chess home and model picker."""

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class HomeLayout:
    width: int
    height: int
    scale: float
    header: pygame.Rect
    options_button: pygame.Rect
    mode_buttons: dict
    mode_descriptions: dict
    workspace: pygame.Rect
    workspace_title: pygame.Rect
    white_card: pygame.Rect
    black_card: pygame.Rect
    versus: pygame.Rect
    swap_button: pygame.Rect
    copy_white_button: pygame.Rect
    copy_black_button: pygame.Rect
    white_model_button: pygame.Rect
    black_model_button: pygame.Rect
    white_mcts_button: pygame.Rect
    black_mcts_button: pygame.Rect
    white_budget_minus: pygame.Rect
    white_budget_value: pygame.Rect
    white_budget_plus: pygame.Rect
    black_budget_minus: pygame.Rect
    black_budget_value: pygame.Rect
    black_budget_plus: pygame.Rect
    human_white_button: pygame.Rect
    human_black_button: pygame.Rect
    runtime: pygame.Rect
    match_label: pygame.Rect
    match_minus: pygame.Rect
    match_value: pygame.Rect
    match_plus: pygame.Rect
    pace_label: pygame.Rect
    pace_buttons: tuple
    summary: pygame.Rect
    start_button: pygame.Rect
    cancel_button: pygame.Rect


@dataclass(frozen=True)
class PickerLayout:
    overlay: pygame.Rect
    panel: pygame.Rect
    title: pygame.Rect
    subtitle: pygame.Rect
    close_button: pygame.Rect
    browser: pygame.Rect
    details: pygame.Rect


def ui_scale_for_viewport(width, height, design_width=1360, design_height=900):
    """Scale text and controls with display resolution, including native 4K."""
    width = max(1, int(width))
    height = max(1, int(height))
    scale = min(width / max(1, design_width), height / max(1, design_height)) * 0.92
    return max(0.72, min(2.4, scale))


def _rect(left, top, width, height):
    return pygame.Rect(int(round(left)), int(round(top)), max(1, int(round(width))), max(1, int(round(height))))


def build_home_layout(width, height, scale, game_mode):
    """Create a centered player composer and persistent runtime/start rail."""
    width = max(1, int(width))
    height = max(1, int(height))
    s = float(scale)
    px = lambda value: max(1, int(round(value * s)))
    margin_x = max(px(40), int(round(width * 0.04)))
    gap = px(16)
    inner_left = margin_x
    inner_width = max(1, width - 2 * margin_x)

    header = _rect(margin_x, px(16), inner_width, px(88))
    options = _rect(header.right - px(166), header.top + (header.height - px(42)) // 2, px(146), px(42))

    nav_y = header.bottom + px(18)
    nav_h = px(62)
    nav_gap = px(12)
    nav_w = (inner_width - 2 * nav_gap) // 3
    mode_buttons = {
        "human_vs_ai": _rect(inner_left, nav_y, nav_w, nav_h),
        "ai_vs_ai": _rect(inner_left + nav_w + nav_gap, nav_y, nav_w, nav_h),
        "human_vs_human": _rect(inner_left + 2 * (nav_w + nav_gap), nav_y, nav_w, nav_h),
    }
    mode_descriptions = {
        "human_vs_ai": "Take one side against a model",
        "ai_vs_ai": "Compare two selected models",
        "human_vs_human": "Play locally with another person",
    }

    runtime_h = px(146)
    runtime_bottom_margin = max(px(18), int(round(height * 0.025)))
    runtime = _rect(margin_x, height - runtime_bottom_margin - runtime_h, inner_width, runtime_h)
    workspace_top = mode_buttons["ai_vs_ai"].bottom + px(18)
    workspace_bottom = runtime.top - px(16)
    workspace = _rect(margin_x, workspace_top, inner_width, max(px(280), workspace_bottom - workspace_top))
    title = _rect(workspace.left + px(20), workspace.top + px(16), workspace.width - px(40), px(36))

    action_row_h = px(42)
    action_bottom = workspace.bottom - px(14)
    action_top = action_bottom - action_row_h
    cards_top = title.bottom + px(16)
    cards_bottom = action_top - px(14)
    card_h = max(px(230), cards_bottom - cards_top)

    center_w = max(px(76), int(round(inner_width * 0.085)))
    card_gap = max(px(12), int(round(inner_width * 0.018)))
    card_w = max(1, (inner_width - center_w - 2 * card_gap) // 2)
    cluster_w = 2 * card_w + center_w + 2 * card_gap
    cluster_left = inner_left + max(0, (inner_width - cluster_w) // 2)
    white_card = _rect(cluster_left, cards_top, card_w, card_h)
    versus = _rect(white_card.right + card_gap, cards_top + max(0, (card_h - px(72)) // 2), center_w, px(72))
    black_card = _rect(versus.right + card_gap, cards_top, card_w, card_h)

    inset = px(18)
    title_h = px(28)
    model_h = px(76)
    model_top = white_card.top + inset + title_h + px(10)
    white_model = _rect(white_card.left + inset, model_top, white_card.width - 2 * inset, model_h)
    black_model = _rect(black_card.left + inset, model_top, black_card.width - 2 * inset, model_h)

    control_left_pad = px(18)
    control_right_pad = px(18)
    control_bottom = white_card.bottom - px(16)
    budget_h = px(38)
    budget_top = control_bottom - budget_h
    toggle_h = px(38)
    toggle_top = budget_top - px(12) - toggle_h
    # Keep search controls close to the model metrics on tall/high-DPI screens.
    stats_bottom = model_top + model_h + px(10) + px(54)
    toggle_top = min(toggle_top, stats_bottom + px(34))
    budget_top = toggle_top + toggle_h + px(12)
    white_mcts = _rect(white_card.left + control_left_pad, toggle_top, white_card.width - control_left_pad - control_right_pad, toggle_h)
    black_mcts = _rect(black_card.left + control_left_pad, toggle_top, black_card.width - control_left_pad - control_right_pad, toggle_h)

    budget_side_w = px(44)
    budget_value_w = max(1, white_card.width - 2 * control_left_pad - 2 * budget_side_w - px(16))
    white_budget_minus = _rect(white_card.left + control_left_pad, budget_top, budget_side_w, budget_h)
    white_budget_value = _rect(white_budget_minus.right + px(8), budget_top, budget_value_w, budget_h)
    white_budget_plus = _rect(white_budget_value.right + px(8), budget_top, budget_side_w, budget_h)
    black_budget_minus = _rect(black_card.left + control_left_pad, budget_top, budget_side_w, budget_h)
    black_budget_value = _rect(black_budget_minus.right + px(8), budget_top, budget_value_w, budget_h)
    black_budget_plus = _rect(black_budget_value.right + px(8), budget_top, budget_side_w, budget_h)

    action_gap = px(12)
    action_w = (inner_width - 2 * action_gap) // 3
    swap = _rect(inner_left, action_top, action_w, action_row_h)
    copy_white = _rect(swap.right + action_gap, action_top, action_w, action_row_h)
    copy_black = _rect(copy_white.right + action_gap, action_top, action_w, action_row_h)

    human_w = _rect(inner_left + px(24), white_card.top + white_card.height // 2, max(1, white_card.width - px(48)), px(38))
    human_b = _rect(human_w.left, human_w.bottom + px(12), human_w.width, px(38))

    runtime_pad = px(18)
    summary_w = int(inner_width * 0.27)
    pace_w = int(inner_width * 0.40)
    actions_w = max(1, inner_width - 2 * runtime_pad - summary_w - pace_w - 2 * gap)
    summary = _rect(runtime.left + runtime_pad, runtime.top + runtime_pad, summary_w, runtime.height - 2 * runtime_pad)
    pace_area = _rect(summary.right + gap, summary.top, pace_w, summary.height)
    action_area = _rect(pace_area.right + gap, summary.top, actions_w, summary.height)
    match_label = _rect(summary.left, summary.top, summary.width, px(22))
    match_controls_y = match_label.bottom + px(6)
    match_minus = _rect(summary.left, match_controls_y, px(40), px(38))
    match_value = _rect(match_minus.right + px(7), match_controls_y, px(76), px(38))
    match_plus = _rect(match_value.right + px(7), match_controls_y, px(40), px(38))
    pace_label = _rect(pace_area.left, pace_area.top, pace_area.width, px(22))
    pace_gap = px(8)
    pace_top = pace_label.bottom + px(6)
    pace_button_w = (pace_area.width - 3 * pace_gap) // 4
    pace_values = (0.0, 0.2, 0.6, 1.0)
    pace_buttons = tuple(
        _rect(pace_area.left + index * (pace_button_w + pace_gap), pace_top, pace_button_w, px(38))
        for index in range(4)
    )

    start_h = px(46)
    start_w = min(action_area.width, px(198))
    cancel_w = min(action_area.width, px(104))
    start_y = action_area.top + max(0, (action_area.height - start_h) // 2)
    start = _rect(action_area.left, start_y, start_w, start_h)
    cancel = _rect(start.right + px(8), start_y, min(cancel_w, max(1, action_area.right - (start.right + px(8)))), start_h)

    return HomeLayout(
        width, height, s, header, options, mode_buttons, mode_descriptions, workspace, title,
        white_card, black_card, versus, swap, copy_white, copy_black, white_model, black_model,
        white_mcts, black_mcts, white_budget_minus, white_budget_value, white_budget_plus,
        black_budget_minus, black_budget_value, black_budget_plus, human_w, human_b,
        runtime, match_label, match_minus, match_value, match_plus, pace_label,
        tuple(zip(pace_values, pace_buttons)), summary, start, cancel,
    )


def build_picker_layout(width, height, scale):
    """Size a centered picker modal within the current visible canvas."""
    width = max(1, int(width))
    height = max(1, int(height))
    s = float(scale)
    px = lambda value: max(1, int(round(value * s)))
    overlay = pygame.Rect(0, 0, width, height)
    pad = max(px(24), int(round(min(width, height) * 0.045)))
    max_width = px(1380)
    max_height = px(690)
    panel_width = min(width - 2 * pad, max_width)
    panel_height = min(height - 2 * pad, max_height)
    panel = _rect((width - panel_width) // 2, (height - panel_height) // 2, panel_width, panel_height)
    title = _rect(panel.left + px(28), panel.top + px(20), panel.width - px(100), px(36))
    subtitle = _rect(title.left, title.bottom + px(4), title.width, px(24))
    close = _rect(panel.right - px(56), panel.top + px(18), px(36), px(36))
    content_top = subtitle.bottom + px(16)
    content_bottom = panel.bottom - px(24)
    content_height = max(1, content_bottom - content_top)
    gap = px(18)
    details_width = min(px(340), int(panel.width * 0.32))
    browser_width = max(1, panel.width - 2 * px(28) - gap - details_width)
    browser = _rect(panel.left + px(28), content_top, browser_width, content_height)
    details = _rect(browser.right + gap, content_top, details_width, content_height)
    return PickerLayout(overlay, panel, title, subtitle, close, browser, details)
