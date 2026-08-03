"""Shared visual language for the local chess setup and game windows."""

import pygame


BG_TOP = (14, 19, 28)
BG_BOTTOM = (7, 10, 16)
SURFACE = (22, 29, 40)
SURFACE_RAISED = (28, 37, 50)
SURFACE_SOFT = (18, 24, 33)
BORDER = (70, 88, 116)
BORDER_STRONG = (94, 118, 154)
TEXT = (240, 245, 252)
MUTED = (150, 166, 190)
SUBTLE = (112, 128, 151)
ACCENT = (84, 149, 235)
ACCENT_HOVER = (105, 168, 247)
SUCCESS = (82, 184, 135)
WARNING = (235, 184, 92)
DANGER = (196, 91, 98)


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

    glow = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.circle(
        glow,
        (*ACCENT, 18),
        (int(width * 0.12), int(height * 0.06)),
        max(180, int(min(width, height) * 0.42)),
    )
    pygame.draw.circle(
        glow,
        (66, 184, 158, 10),
        (int(width * 0.92), int(height * 0.82)),
        max(160, int(min(width, height) * 0.34)),
    )
    surface.blit(glow, (0, 0))
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
