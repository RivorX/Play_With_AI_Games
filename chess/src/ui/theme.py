"""Shared visual language for the local chess setup and game windows."""

import pygame


BG_TOP = (16, 22, 27)
BG_BOTTOM = (12, 17, 21)
SURFACE = (25, 34, 40)
SURFACE_RAISED = (31, 43, 50)
SURFACE_SOFT = (20, 28, 33)
BORDER = (53, 70, 76)
BORDER_STRONG = (77, 101, 104)
TEXT = (236, 240, 235)
MUTED = (151, 168, 171)
SUBTLE = (110, 128, 132)
ACCENT = (105, 196, 171)
ACCENT_HOVER = (135, 215, 189)
SUCCESS = (113, 193, 148)
WARNING = (225, 178, 104)
DANGER = (202, 111, 104)


def lerp_color(color_a, color_b, t):
    """Interpolate two RGB colors."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(color_a[idx] + (color_b[idx] - color_a[idx]) * t) for idx in range(3))


def build_background(width, height):
    """Create the quiet gradient used behind both application views."""
    width = max(1, int(width))
    height = max(1, int(height))
    surface = pygame.Surface((width, height))
    for y in range(height):
        color = lerp_color(BG_TOP, BG_BOTTOM, y / max(1, height - 1))
        pygame.draw.line(surface, color, (0, y), (width, y))

    return surface


def draw_card(surface, rect, fill=SURFACE, border=BORDER, radius=14, shadow=False, accent=None):
    """Draw a consistent bordered surface, optionally with an active accent."""
    if shadow:
        shadow_rect = rect.move(0, 6)
        shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 48), shadow_surface.get_rect(), border_radius=radius)
        surface.blit(shadow_surface, shadow_rect.topleft)
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=1, border_radius=radius)
    if accent is not None:
        accent_rect = pygame.Rect(rect.left, rect.top + radius, 3, max(1, rect.height - radius * 2))
        pygame.draw.rect(surface, accent, accent_rect, border_radius=2)
