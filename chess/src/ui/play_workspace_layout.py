"""Responsive geometry for the two-pane chess play workspace."""

import pygame
import chess

from .play_visuals import fit_middle_text, fit_text
from .gui_helpers import get_game_mode_labels
from .theme import ACCENT, BORDER, MUTED, SURFACE, SURFACE_RAISED, TEXT, WARNING


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720


def workspace_scale(width, height):
    """Scale typography and spacing with the native canvas, including 4K."""
    width = max(1, int(width))
    height = max(1, int(height))
    return max(0.76, min(3.0, min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)))


def compute_workspace_layout(width, height):
    """Return measured-by-scale geometry for header, board, player strips and inspector."""
    width = max(1, int(width))
    height = max(1, int(height))
    scale = workspace_scale(width, height)
    density = max(0.82, min(2.2, 1.0 + 0.6 * (scale - 1.0)))
    margin_x = max(12, round(18 * scale))
    margin_y = max(10, round(14 * scale))
    header_height = max(40, round(50 * scale))
    header_gap = max(8, round(12 * scale))
    panel_gap = max(10, round(14 * scale))

    content = pygame.Rect(
        margin_x,
        margin_y + header_height + header_gap,
        max(1, width - margin_x * 2),
        max(1, height - margin_y * 2 - header_height - header_gap),
    )
    desired_inspector_width = max(round(360 * scale), round(content.width * 0.40))
    max_inspector_width = max(round(320 * scale), round(content.width * 0.45))
    inspector_width = min(desired_inspector_width, max_inspector_width)
    min_board_width = round(300 * scale)
    if content.width - inspector_width - panel_gap < min_board_width:
        inspector_width = max(round(230 * scale), content.width - panel_gap - min_board_width)
    inspector_width = max(1, min(content.width, inspector_width))

    board_area = pygame.Rect(
        content.left,
        content.top,
        max(1, content.width - inspector_width - panel_gap),
        content.height,
    )
    inspector = pygame.Rect(
        board_area.right + panel_gap,
        content.top,
        inspector_width,
        content.height,
    )

    strip_height = max(38, round(56 * scale))
    strip_gap = max(6, round(8 * scale))
    frame_padding = max(5, round(6 * scale))
    board_size = min(
        board_area.width - frame_padding * 2,
        content.height - strip_height * 2 - strip_gap * 2 - frame_padding * 2,
    )
    board_size = max(8, (max(8, int(board_size)) // 8) * 8)
    board_size = min(board_size, board_area.width - frame_padding * 2)
    board_size = max(8, (board_size // 8) * 8)
    board_left = board_area.left + (board_area.width - board_size) // 2
    board_top = content.top + strip_height + strip_gap + frame_padding
    board = pygame.Rect(board_left, board_top, board_size, board_size)
    board_frame = board.inflate(frame_padding * 2, frame_padding * 2)
    top_player = pygame.Rect(board_frame.left, board_frame.top - strip_gap - strip_height, board_frame.width, strip_height)
    bottom_player = pygame.Rect(board_frame.left, board_frame.bottom + strip_gap, board_frame.width, strip_height)

    inspector_pad_x = max(8, round(10 * density))
    tab_top = inspector.top + max(5, round(7 * density))
    tab_height = max(22, round(26 * density))
    detail_button_width = max(92, round(112 * density))
    tabs = {
        "match": pygame.Rect(
            inspector.right - inspector_pad_x - detail_button_width,
            tab_top,
            detail_button_width,
            tab_height,
        ),
    }

    section_gap = max(3, round(5 * density))
    summary_height = max(22, round(27 * density))
    match_summary = pygame.Rect(
        inspector.left + inspector_pad_x,
        tabs["match"].bottom + section_gap,
        inspector.width - inspector_pad_x * 2,
        summary_height,
    )
    inspector_body_top = match_summary.bottom + section_gap
    inspector_body = pygame.Rect(
        inspector.left + inspector_pad_x,
        inspector_body_top,
        inspector.width - inspector_pad_x * 2,
        max(1, inspector.bottom - inspector_pad_x - inspector_body_top),
    )

    return {
        "scale": scale,
        "density": density,
        "margin_x": margin_x,
        "margin_y": margin_y,
        "header": pygame.Rect(margin_x, margin_y, width - margin_x * 2, header_height),
        "content": content,
        "board_area": board_area,
        "board": board,
        "board_frame": board_frame,
        "top_player": top_player,
        "bottom_player": bottom_player,
        "inspector": inspector,
        "tabs": tabs,
        "match_summary": match_summary,
        "inspector_body_without_summary": pygame.Rect(
            inspector.left + inspector_pad_x,
            tabs["match"].bottom + section_gap,
            inspector.width - inspector_pad_x * 2,
            max(1, inspector.bottom - inspector_pad_x - (tabs["match"].bottom + section_gap)),
        ),
        "inspector_body_with_summary": inspector_body,
        "header_control_height": max(30, round(36 * scale)),
        "header_control_gap": max(5, round(8 * scale)),
    }


def _add_button(gui, action, label, enabled, active, rect, danger=False):
    hovered = bool(enabled and rect.collidepoint(gui.mouse_canvas_pos))
    if hovered:
        gui.hover_any_button = True
    hover_t = gui._update_button_hover(action, hovered)
    if rect.height <= round(30 * gui.workspace_layout.get("density", 1.0)):
        if not enabled:
            fill, border, text_color = (35, 45, 48), BORDER, MUTED
        elif danger:
            fill, border, text_color = (93, 51, 51), (166, 103, 95), (246, 231, 222)
        elif active:
            fill, border, text_color = (47, 93, 83), ACCENT, TEXT
        else:
            fill, border, text_color = (37, 57, 61), BORDER, TEXT
        if enabled:
            fill = gui._lerp_color(fill, tuple(min(255, c + 20) for c in fill), hover_t)
            border = gui._lerp_color(border, tuple(min(255, c + 28) for c in border), hover_t)
        button = rect.move(0, -int(round(1 * hover_t)))
        pygame.draw.rect(gui.canvas, (9, 14, 17), button.move(0, 1), border_radius=max(3, round(4 * gui.workspace_layout.get("density", 1.0))))
        pygame.draw.rect(gui.canvas, fill, button, border_radius=max(3, round(4 * gui.workspace_layout.get("density", 1.0))))
        pygame.draw.rect(gui.canvas, border, button, width=1, border_radius=max(3, round(4 * gui.workspace_layout.get("density", 1.0))))
        text = gui.inspector_tiny_font.render(fit_text(gui.inspector_tiny_font, label, button.width - 8), True, text_color)
        gui.canvas.blit(text, text.get_rect(center=button.center))
    else:
        gui._draw_action_button(rect, label, enabled=enabled, active=active, danger=danger, hover_t=hover_t)
    gui.action_buttons.append({"action": action, "rect": rect, "enabled": enabled})


def _activity_text(gui):
    if gui._history_preview_active():
        suffix = " · non-rated analysis" if gui.manual_analysis_mode else " · live position preserved"
        return gui._history_preview_status() + suffix
    if gui.ai_status_error:
        return "Search failed · restart to retry"
    if gui.pending_manual_move is not None:
        return "Applying selected move after current search drains"
    if gui.manual_analysis_mode:
        return "Analysis · non-rated · AI paused" if gui.ai_paused else "Analysis · non-rated"
    if gui.ai_paused:
        return "Match paused" if gui._match_enabled() else "Game paused"
    turn = gui.ai_turn
    if turn is not None:
        if turn.error is not None:
            return "Search failed · restart to retry"
        if turn.is_done and not turn.is_ready_to_apply:
            return f"Move ready · appears in {turn.presentation_remaining_seconds:.1f}s"
        if turn.is_done:
            return "Move ready"
        side = "White" if gui.board.turn == chess.WHITE else "Black"
        return f"{side} searching · {gui._format_duration(gui._current_thinking_elapsed())}"
    if gui.ai_thinking and gui.ai_thinking_started_at is not None:
        side = "White" if gui.board.turn == chess.WHITE else "Black"
        return f"{side} starting search"
    return f"{'White' if gui.board.turn == chess.WHITE else 'Black'} to move"


def _draw_header(gui):
    header = gui.header_rect
    scale = gui.ui_scale
    pad = max(8, round(12 * scale))
    pygame.draw.line(gui.canvas, BORDER, header.bottomleft, header.bottomright, max(1, round(scale)))

    brand = gui.medium_font.render("CHESS", True, ACCENT)
    mode_text, perspective = get_game_mode_labels(gui.game_mode, gui.human_color)
    mode = gui.small_font.render(mode_text, True, TEXT)
    subtitle = gui.tiny_font.render(perspective, True, MUTED)
    gui.canvas.blit(brand, (header.left + pad, header.centery - brand.get_height() // 2))
    mode_x = header.left + pad + brand.get_width() + max(10, round(14 * scale))
    mode_y = header.centery - mode.get_height() // 2
    gui.canvas.blit(mode, (mode_x, mode_y))
    subtitle_x = mode_x + mode.get_width() + max(8, round(12 * scale))
    subtitle_y = header.centery - subtitle.get_height() // 2
    gui.canvas.blit(subtitle, (subtitle_x, subtitle_y))

    actions = [
        ("menu", "Menu", True, False, False),
        ("restart", "Restart", True, False, False),
        ("flip", "Flip", True, gui.flipped, False),
    ]
    if gui._history_preview_active():
        actions.append(("live_board", "Back to live", True, True, False))
    if gui.game_mode == "human_vs_ai" and gui.mcts_enabled:
        actions.append(("toggle_mcts", f"MCTS {'On' if gui.use_mcts else 'Off'}", not gui.ai_thinking, gui.use_mcts, False))
    if getattr(gui, "manual_analysis_mode", False):
        can_resume = bool(gui._can_resume_ai())
        actions.append(("resume_ai", "Resume AI", can_resume, False, False))
    elif gui.game_mode == "ai_vs_ai":
        actions.append(("pause", "Resume" if gui.ai_paused else "Pause", True, gui.ai_paused, False))

    button_gap = gui.workspace_layout["header_control_gap"]
    button_height = gui.workspace_layout["header_control_height"]
    right = header.right - pad
    for action, label, enabled, active, danger in reversed(actions):
        button_width = max(round(72 * scale), gui.small_font.size(label)[0] + round(28 * scale))
        rect = pygame.Rect(right - button_width, header.centery - button_height // 2, button_width, button_height)
        _add_button(gui, action, label, enabled, active, rect, danger=danger)
        right = rect.left - button_gap

    status_x = subtitle_x + subtitle.get_width() + max(12, round(18 * scale))
    status = fit_text(gui.tiny_font, _activity_text(gui), max(0, right - status_x))
    if status:
        color = (255, 154, 132) if gui.ai_status_error else WARNING if gui.ai_thinking or gui.ai_paused else MUTED
        rendered = gui.tiny_font.render(status, True, color)
        gui.canvas.blit(rendered, (status_x, header.centery - rendered.get_height() // 2))


def _draw_player_strip(gui, rect, color):
    info = gui._side_info(color)
    display_board = gui._display_board()
    preview_active = gui._history_preview_active()
    active = display_board.turn == color
    gui._draw_card(rect, fill=SURFACE_RAISED if active else SURFACE, border=ACCENT if active else BORDER)
    scale = gui.ui_scale
    pad_x = max(8, round(12 * scale))
    pad_y = max(4, round(6 * scale))
    side = "WHITE" if color == chess.WHITE else "BLACK"
    if preview_active:
        status = "PREVIEW"
    elif active and gui._pending_ai_choice_matches_display(color):
        choice = gui.pending_ai_choice
        if gui.ai_paused:
            status = f"PAUSED · {choice.get('san', '')}"
        elif gui.ai_turn is not None and not gui.ai_turn.is_ready_to_apply:
            status = f"MOVE IN {gui.ai_turn.presentation_remaining_seconds:.1f}s"
        else:
            status = f"PLAYS {choice.get('san', '')}"
    elif active and gui.ai_paused:
        status = "PAUSED"
    elif active and gui.ai_thinking:
        if gui.ai_turn is not None and gui.ai_turn.is_done and not gui.ai_turn.is_ready_to_apply:
            status = f"MOVE IN {gui.ai_turn.presentation_remaining_seconds:.1f}s"
        else:
            status = "THINKING"
    else:
        status = "TO MOVE" if active else ""
    title = gui.tiny_font.render(side + (f" · {status}" if status else ""), True, ACCENT if active else MUTED)
    gui.canvas.blit(title, (rect.left + pad_x, rect.top + pad_y))

    mode = info.get("search_mode", "LOCAL PLAYER") if not info["is_human"] else "LOCAL PLAYER"
    mode_width = gui.tiny_font.size(mode)[0] + max(12, round(16 * scale))
    badge = pygame.Rect(
        rect.right - pad_x - mode_width,
        rect.top + pad_y,
        mode_width,
        gui.tiny_font.get_height() + max(4, round(6 * scale)),
    )
    pygame.draw.rect(gui.canvas, (25, 48, 44), badge, border_radius=max(6, round(8 * scale)))
    pygame.draw.rect(gui.canvas, BORDER, badge, width=1, border_radius=max(6, round(8 * scale)))
    mode_surface = gui.tiny_font.render(mode, True, ACCENT)
    gui.canvas.blit(mode_surface, mode_surface.get_rect(center=badge.center))

    name_top = rect.top + pad_y + gui.tiny_font.get_height() + max(1, round(2 * scale))
    name_width = max(12, badge.left - rect.left - pad_x - max(8, round(10 * scale)))
    model_name = gui._match_model_label(info.get("name", "Player"))
    name = gui.small_font.render(fit_middle_text(gui.small_font, model_name, name_width), True, TEXT)
    gui.canvas.blit(name, (rect.left + pad_x, name_top))


def _draw_match_summary(gui, rect):
    state = gui._match_snapshot()
    completed = int(state.get("completed", 0) or 0)
    attempted = int(state.get("attempted", 0) or 0)
    total = max(1, int(state.get("total", gui.match_total_games) or gui.match_total_games))
    status = str(state.get("status", "starting"))
    visible_finished = bool(state.get("visible_game_finished", gui.visible_match_finished))
    visible_game = dict(state.get("visible_game") or {})
    background_total = max(0, int(state.get("background_total", total - 1) or 0))
    background_attempted = max(0, int(state.get("background_attempted", 0) or 0))
    background_completed = max(0, int(state.get("background_completed", 0) or 0))
    background_done = bool(state.get("background_done", total <= 1))
    active_games = max(0, int(state.get("active_games", 0) or 0))
    background_starting = bool(state.get("background_starting", False))
    if gui._visible_match_active():
        side = "White" if gui.board.turn == chess.WHITE else "Black"
        if total <= 1:
            line = f"Game 1 live · {side} to move · 1/1"
        elif background_done:
            line = f"Game 1 live · {side} to move · {completed}/{total} done"
        elif background_attempted or active_games:
            line = f"Game 1 · {side} to move · background {background_completed}/{background_total} · {active_games} active"
        elif background_starting:
            line = f"Game 1 · {side} to move · starting {total - 1} more"
        else:
            line = f"Game 1 · {side} to move · {total - 1} queued"
    elif visible_finished and status not in {"completed", "cancelled", "error", "incomplete"}:
        result = str(visible_game.get("result", gui.visible_match_result or "*"))
        winner = visible_game.get("winner_model") or gui.visible_match_winner
        result_label = "draw" if result == "1/2-1/2" else f"{gui._compact_model_label(winner)} won" if winner else "finished"
        line = f"Game 1 {result_label} · {completed}/{total} resolved"
    elif status == "completed":
        line = f"Match complete · {completed}/{total} resolved · {int(state.get('draws', 0) or 0)} draws"
    elif status in {"cancelled", "incomplete", "error"}:
        line = f"Match {status} · {completed}/{total} resolved"
    elif status in {"starting", "loading_models", "starting_workers", "stopping_previous"}:
        line = f"Preparing match · {float(state.get('runtime_stage_elapsed_s', 0.0) or 0.0):.1f}s"
    else:
        line = f"Match · {completed}/{total} · {active_games} active · {max(0, total - attempted)} queued"

    density = gui.workspace_layout.get("density", 1.0)
    pad = max(2, round(4 * density))
    line_surface = gui.inspector_tiny_font.render(fit_text(gui.inspector_tiny_font, line, rect.width - 2 * pad), True, TEXT)
    gui.canvas.blit(line_surface, (rect.left + pad, rect.centery - line_surface.get_height() // 2 - 1))
    bar_height = max(2, round(2 * density))
    bar = pygame.Rect(rect.left + pad, rect.bottom - bar_height - 1, max(1, rect.width - 2 * pad), bar_height)
    pygame.draw.rect(gui.canvas, (42, 57, 59), bar, border_radius=bar_height // 2)
    progress = min(1.0, max(0.0, attempted / float(total)))
    if gui._visible_match_active() and attempted == 0:
        progress = 1.0 / float(total)
    if progress > 0:
        fill = pygame.Rect(bar.left, bar.top, max(1, round(bar.width * progress)), bar.height)
        pygame.draw.rect(gui.canvas, ACCENT, fill, border_radius=bar_height // 2)


def _draw_tabs(gui):
    rect = gui.inspector_tab_rects["match"]
    in_match_details = gui.inspector_tab == "match"
    label = "Back to Play" if in_match_details else "Match details"
    action = "tab_moves" if in_match_details else "tab_match"
    hovered = rect.collidepoint(gui.mouse_canvas_pos)
    if hovered:
        gui.hover_any_button = True
    radius = max(4, round(5 * gui.workspace_layout.get("density", 1.0)))
    pygame.draw.rect(gui.canvas, (36, 55, 51) if hovered or in_match_details else (22, 32, 36), rect, border_radius=radius)
    pygame.draw.rect(gui.canvas, ACCENT if hovered or in_match_details else BORDER, rect, width=1, border_radius=radius)
    text = gui.inspector_tiny_font.render(fit_text(gui.inspector_tiny_font, label, rect.width - 12), True, TEXT if hovered or in_match_details else MUTED)
    gui.canvas.blit(text, text.get_rect(center=rect.center))
    title = gui.inspector_tiny_font.render("MATCH" if in_match_details else "PLAY", True, ACCENT if in_match_details else MUTED)
    gui.canvas.blit(title, (gui.right_panel_rect.left + max(8, round(10 * gui.workspace_layout.get("density", 1.0))), rect.centery - title.get_height() // 2))
    gui.action_buttons.append({"action": action, "rect": rect, "enabled": True})


def _draw_moves(gui, rect):
    scale = gui.workspace_layout["density"]
    font = gui.inspector_tiny_font
    small_font = gui.inspector_small_font
    pad = max(5, round(7 * scale))
    gui._draw_card(rect, fill=SURFACE, border=BORDER)
    title = small_font.render("History", True, TEXT)
    count = font.render(f"{len(gui.move_history)} plies", True, MUTED)
    gui.canvas.blit(title, (rect.left + pad, rect.top + pad))
    nav_h = max(20, round(23 * scale))
    nav_gap = max(2, round(3 * scale))
    button_labels = ("‹", "›")
    nav_widths = [max(round(24 * scale), font.size(label)[0] + round(8 * scale)) for label in button_labels]
    nav_total = sum(nav_widths) + nav_gap * (len(button_labels) - 1)
    count_gap = max(4, round(6 * scale))
    count_x = rect.right - pad - nav_total - count_gap - count.get_width()
    available_for_nav = max(1, rect.right - pad - max(rect.left + pad + title.get_width() + count_gap + count.get_width(), count_x + count.get_width() + count_gap))
    if nav_total > available_for_nav:
        nav_widths = [max(1, (available_for_nav - nav_gap) // 2)] * 2
        nav_total = sum(nav_widths) + nav_gap * (len(button_labels) - 1)
        count_x = rect.right - pad - nav_total - count_gap - count.get_width()
    if count_x >= rect.left + pad + title.get_width() + count_gap:
        gui.canvas.blit(count, (count_x, rect.top + pad + max(1, (title.get_height() - count.get_height()) // 2)))
    nav_x = rect.right - pad - nav_total
    nav_y = rect.top + pad + max(0, (title.get_height() - nav_h) // 2)
    preview = gui._history_preview_active()
    current_ply = gui.selected_history_ply if preview else len(gui.move_history)
    current_ply = max(0, min(len(gui.move_history), int(current_ply or 0)))
    nav_actions = (("history_prev", "‹", current_ply > 0), ("history_next", "›", current_ply < len(gui.move_history)))
    for index, (action, label, enabled) in enumerate(nav_actions):
        button_rect = pygame.Rect(nav_x, nav_y, nav_widths[index], nav_h)
        _add_button(gui, action, label, enabled, action == "live_board" and preview, button_rect)
        nav_x = button_rect.right + nav_gap

    footer_h = max(22, round(26 * scale))
    footer_y = rect.bottom - footer_h - max(3, round(4 * scale))
    list_top = rect.top + pad + max(title.get_height(), nav_h) + max(2, round(3 * scale))
    list_rect = pygame.Rect(rect.left + max(2, round(3 * scale)), list_top, rect.width - max(4, round(6 * scale)), max(1, footer_y - list_top))
    gui.history_rect = list_rect
    gui._draw_card(list_rect, fill=(19, 28, 32), border=BORDER)
    row_top = list_rect.top + max(20, round(22 * scale))
    num_w = max(round(20 * scale), font.size("#")[0] + round(7 * scale))
    col_gap = max(2, round(4 * scale))
    pad_w = max(4, round(6 * scale))
    move_w = max(1, (list_rect.width - 2 * pad_w - num_w - 2 * col_gap) // 2)
    num_x = list_rect.left + pad_w
    white_x = num_x + num_w + col_gap
    black_x = white_x + move_w + col_gap
    for label, x, width in (("#", num_x, num_w), ("WHITE", white_x, move_w), ("BLACK", black_x, move_w)):
        rendered = font.render(fit_text(font, label, width), True, MUTED)
        gui.canvas.blit(rendered, (x, list_rect.top + max(2, round(3 * scale))))

    rows = [
        (index // 2 + 1, gui.move_san_history[index], gui.move_san_history[index + 1] if index + 1 < len(gui.move_san_history) else "")
        for index in range(0, len(gui.move_san_history), 2)
    ]
    row_h = max(round(18 * scale), font.get_height() + max(3, round(4 * scale)))
    visible = max(1, (list_rect.bottom - row_top - pad_w) // row_h)
    max_scroll = max(0, len(rows) - visible)
    gui.history_scroll_rows = max(0, min(gui.history_scroll_rows, max_scroll))
    start = max(0, len(rows) - visible - gui.history_scroll_rows)
    end = len(rows) - gui.history_scroll_rows if gui.history_scroll_rows else len(rows)
    gui.history_entry_buttons = []
    previous_clip = gui.canvas.get_clip()
    gui.canvas.set_clip(list_rect.clip(gui.canvas.get_rect()))
    for row_idx, (move_no, white_move, black_move) in enumerate(rows[start:end]):
        y = row_top + row_idx * row_h
        row_rect = pygame.Rect(list_rect.left + 4, y, list_rect.width - 8, row_h - 3)
        white_ply = (move_no - 1) * 2 + 1
        black_ply = min(len(gui.move_history), white_ply + 1)
        if move_no % 2 == 1:
            pygame.draw.rect(gui.canvas, (24, 35, 39), row_rect, border_radius=max(3, round(4 * scale)))
        selected_ply = gui.selected_history_ply
        selected_move_x = white_x if selected_ply == white_ply else black_x if selected_ply == black_ply else None
        if selected_move_x is not None:
            selected_rect = pygame.Rect(selected_move_x - 3, y + 1, move_w + 2, row_h - 4)
            pygame.draw.rect(gui.canvas, (43, 77, 68), selected_rect, border_radius=max(3, round(4 * scale)))
        y_text = y + max(1, round(2 * scale))
        gui.canvas.blit(font.render(f"{move_no:>2}", True, MUTED), (num_x, y_text))
        gui.canvas.blit(font.render(fit_text(font, white_move, move_w - 4), True, TEXT), (white_x, y_text))
        gui.canvas.blit(font.render(fit_text(font, black_move, move_w - 4), True, TEXT), (black_x, y_text))
        gui.history_entry_buttons.append({"rect": pygame.Rect(white_x - 2, y, move_w, row_h - 1), "ply": white_ply})
        if black_move:
            gui.history_entry_buttons.append({"rect": pygame.Rect(black_x - 2, y, move_w, row_h - 1), "ply": black_ply})
    gui.canvas.set_clip(previous_clip)
    if not rows:
        empty = font.render("Moves will appear here as the game progresses.", True, MUTED)
        gui.canvas.blit(empty, empty.get_rect(center=list_rect.center))

    button_h = max(20, round(23 * scale))
    undo_enabled = bool(gui.move_history) and gui.pending_manual_move is None
    rewind_enabled = gui.selected_history_ply is not None and gui.pending_manual_move is None
    undo_width = max(round(50 * scale), font.size("Undo")[0] + round(14 * scale))
    _add_button(gui, "undo", "Undo", undo_enabled, False, pygame.Rect(rect.left + pad, footer_y, undo_width, button_h))
    if gui.selected_history_ply is not None:
        branch_label = "Continue here"
        branch_width = max(round(92 * scale), font.size(branch_label)[0] + round(14 * scale))
        branch_rect = pygame.Rect(rect.right - pad - branch_width, footer_y, branch_width, button_h)
        _add_button(gui, "rewind", branch_label, rewind_enabled, False, branch_rect)


def _draw_analysis(gui, rect):
    gui.analysis_entry_buttons = []
    gui.analysis_rect = rect.copy()
    density = gui.workspace_layout["density"]
    pad = max(5, round(7 * density))
    display_board = gui._display_board()
    color = display_board.turn
    path_ply = gui.selected_history_ply if gui._history_preview_active() else len(gui.move_history)
    displayed_fen = display_board.fen()
    candidate = gui.selected_analysis_candidate or {}
    selected_matches = (
        (candidate.get("color"), candidate.get("move")) == gui.selected_analysis_move
        and candidate.get("fen") == displayed_fen
        and candidate.get("history_key") == gui._history_path_key(path_ply)
    )
    selected_san = candidate.get("san") if selected_matches else None
    button_h = max(20, round(23 * density)) if selected_san else 0
    card_height = max(1, rect.height - (button_h + max(3, round(4 * density)) if button_h else 0))
    card_rect = pygame.Rect(rect.left, rect.top, rect.width, card_height)
    if gui._side_has_visible_analysis(color):
        previous_clip = gui.canvas.get_clip()
        gui.canvas.set_clip(card_rect.clip(gui.canvas.get_rect()))
        row_limit = max(1, int(getattr(gui, "analysis_display_rows", 10)))
        gui._draw_analysis_card(card_rect, color, active=True, row_limit=row_limit)
        gui.canvas.set_clip(previous_clip)
    else:
        gui._draw_card(card_rect, fill=SURFACE_RAISED, border=BORDER)
        message = "NN candidates loading" if gui._history_preview_active() else "Candidates appear after search"
        detail = gui.inspector_tiny_font.render(fit_text(gui.inspector_tiny_font, message, card_rect.width - 2 * pad), True, MUTED)
        gui.canvas.blit(detail, (card_rect.left + pad, card_rect.centery - detail.get_height() // 2))

    if selected_san and gui.pending_manual_move is None and any(
        (entry.get("color"), entry.get("move")) == gui.selected_analysis_move
        and entry.get("fen") == displayed_fen
        and entry.get("history_key") == gui._history_path_key(path_ply)
        for entry in gui.analysis_entry_buttons
    ):
        label = f"Play {selected_san}"
        button_w = min(rect.width - 2 * pad, max(round(84 * density), gui.inspector_tiny_font.size(label)[0] + round(18 * density)))
        button_rect = pygame.Rect(rect.right - pad - button_w, rect.bottom - button_h, button_w, button_h)
        _add_button(gui, "play_candidate", label, True, False, button_rect)


def draw_workspace(gui):
    """Render the two-pane play workspace and refresh all interactive regions."""
    gui.action_buttons = []
    gui.history_entry_buttons = []
    gui.analysis_entry_buttons = []
    gui.hover_any_button = False
    _draw_header(gui)
    top_color = chess.WHITE if gui.flipped else chess.BLACK
    bottom_color = chess.BLACK if top_color == chess.WHITE else chess.WHITE
    _draw_player_strip(gui, gui.top_player_rect, top_color)
    _draw_player_strip(gui, gui.bottom_player_rect, bottom_color)

    gui._draw_card(gui.right_panel_rect, fill=SURFACE, border=BORDER)
    _draw_tabs(gui)
    show_match_summary = gui._match_enabled() and gui.inspector_tab != "match"
    summary_rect = gui.workspace_layout["match_summary"] if show_match_summary else None
    gui.match_summary_rect = summary_rect
    if summary_rect is not None:
        _draw_match_summary(gui, summary_rect)
        gui.match_panel_rect = summary_rect
    else:
        gui.match_panel_rect = None
    if gui.inspector_tab in {"moves", "analysis"}:
        body_key = "inspector_body_with_summary" if gui.match_summary_rect is not None else "inspector_body_without_summary"
        body = gui.workspace_layout[body_key]
        gui.inspector_body_rect = body
        scale = gui.ui_scale
        split_gap = max(6, round(8 * scale))
        history_height = max(1, int((body.height - split_gap) * 0.48))
        history_rect = pygame.Rect(body.left, body.top, body.width, history_height)
        choices_rect = pygame.Rect(body.left, history_rect.bottom + split_gap, body.width,
                                   max(1, body.bottom - history_rect.bottom - split_gap))
        gui.history_panel_rect = history_rect
        gui.choices_panel_rect = choices_rect
        _draw_moves(gui, history_rect)
        _draw_analysis(gui, choices_rect)
    else:
        body = gui.workspace_layout["inspector_body_without_summary"]
        gui.inspector_body_rect = body
        gui.match_panel_rect = body
        if gui._match_enabled():
            gui._draw_match_panel(body)
        else:
            gui._draw_card(body, fill=SURFACE, border=BORDER)
            title = gui.inspector_small_font.render("Model match", True, TEXT)
            detail = gui.inspector_tiny_font.render("Select two models to compare games and results.", True, MUTED)
            offset = max(8, round(10 * gui.workspace_layout["density"]))
            gui.canvas.blit(title, (body.left + offset, body.top + offset))
            gui.canvas.blit(detail, (body.left + offset, body.top + round(34 * gui.workspace_layout["density"])))
