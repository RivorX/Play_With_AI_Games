"""Small, reusable geometry and text helpers for the Pygame play screen."""


def fit_text(font, text, max_width, suffix="..."):
    """Return text that fits a pixel width, using a measured ellipsis if needed."""
    text = str(text)
    max_width = max(0, int(max_width))
    if font.size(text)[0] <= max_width:
        return text
    if not suffix or font.size(suffix)[0] > max_width:
        return ""

    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if font.size(text[:middle] + suffix)[0] <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def fit_middle_text(font, text, max_width, marker="..."):
    """Ellipsize a long identifier in the middle so its distinguishing tail survives."""
    text = str(text)
    max_width = max(0, int(max_width))
    if font.size(text)[0] <= max_width:
        return text
    if not marker or font.size(marker)[0] > max_width:
        return ""

    low = 0
    high = len(text)
    while low < high:
        keep = (low + high + 1) // 2
        head = (keep + 1) // 2
        tail = keep - head
        candidate = text[:head] + marker + (text[-tail:] if tail else "")
        if font.size(candidate)[0] <= max_width:
            low = keep
        else:
            high = keep - 1

    head = (low + 1) // 2
    tail = low - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def centered_rect(pygame, bounds, preferred_width, preferred_height, inset=14):
    """Create a centered rectangle fully contained by ``bounds``."""
    inset = max(0, int(inset))
    width = max(1, min(int(preferred_width), int(bounds.width) - 2 * inset))
    height = max(1, min(int(preferred_height), int(bounds.height) - 2 * inset))
    return pygame.Rect(
        int(bounds.left) + (int(bounds.width) - width) // 2,
        int(bounds.top) + (int(bounds.height) - height) // 2,
        width,
        height,
    )

