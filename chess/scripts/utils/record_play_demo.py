"""Record a real Chess AI GUI session as an animated GIF.

The recorder launches ``play.py`` and captures its window until the player
closes Play. It is intended for short README demos, for example while choosing
AI vs AI and observing the first moves of a match.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageGrab


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAY_SCRIPT = REPO_ROOT / "chess" / "scripts" / "play.py"
WINDOW_TITLE_PREFIX = "Chess AI "


def _enable_dpi_awareness() -> None:
    """Keep Win32 window coordinates aligned with Pillow screen pixels."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _find_play_window_rect() -> tuple[int, int, int, int] | None:
    """Return the visible Chess AI window rectangle, including its title bar."""
    user32 = ctypes.windll.user32
    found: list[tuple[int, int, int, int]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title_length = user32.GetWindowTextLengthW(hwnd)
        if title_length <= 0:
            return True
        title = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title, len(title))
        if not title.value.startswith(WINDOW_TITLE_PREFIX):
            return True

        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width > 0 and height > 0:
                found.append((rect.left, rect.top, rect.right, rect.bottom))
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else None


def _scaled_size(size: tuple[int, int], max_width: int) -> tuple[int, int]:
    width, height = size
    if max_width <= 0 or width <= max_width:
        return size
    target_height = max(1, round(height * max_width / width))
    return max_width, target_height


def _write_gif(
    frame_paths: list[Path], output_path: Path, fps: float
) -> None:
    if not frame_paths:
        raise RuntimeError("Nie udało się przechwycić żadnej klatki.")

    duration_ms = max(1, round(1000 / fps))
    first_frame = Image.open(frame_paths[0])

    def remaining_frames():
        for path in frame_paths[1:]:
            with Image.open(path) as frame:
                yield frame.copy()

    try:
        first_frame.save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=remaining_frames(),
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=True,
        )
    finally:
        first_frame.close()


def record_play_demo(output_path: Path, fps: float, max_width: int) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("Ten recorder używa Windows desktop capture i wymaga Windows.")

    _enable_dpi_awareness()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="play-recording-", dir=output_path.parent))
    frame_paths: list[Path] = []
    play_process: subprocess.Popen | None = None

    try:
        print("Uruchamiam Chess AI Play…")
        play_process = subprocess.Popen(
            [sys.executable, str(PLAY_SCRIPT)], cwd=REPO_ROOT
        )

        print(
            "Nagrywanie wystartuje po pojawieniu się okna. W GUI wybierz AI vs AI, "
            "uruchom partię, a potem zamknij Play — GIF zapisze się automatycznie."
        )
        window_rect: tuple[int, int, int, int] | None = None
        while play_process.poll() is None and window_rect is None:
            window_rect = _find_play_window_rect()
            if window_rect is None:
                time.sleep(0.1)

        if window_rect is None:
            raise RuntimeError("Play zakończył się, zanim pojawiło się okno do nagrania.")

        source_size = (
            window_rect[2] - window_rect[0],
            window_rect[3] - window_rect[1],
        )
        target_size = _scaled_size(source_size, max_width)
        interval_s = 1.0 / fps
        next_frame_at = time.monotonic()
        print(f"Nagrywam {source_size[0]}×{source_size[1]} → {target_size[0]}×{target_size[1]} @ {fps:g} FPS")

        while play_process.poll() is None:
            now = time.monotonic()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
                continue

            current_rect = _find_play_window_rect() or window_rect
            screenshot = ImageGrab.grab(bbox=current_rect).convert("RGB")
            if screenshot.size != target_size:
                screenshot = screenshot.resize(target_size, Image.Resampling.LANCZOS)

            frame_path = temporary_dir / f"frame-{len(frame_paths):06d}.png"
            screenshot.save(frame_path, format="PNG")
            screenshot.close()
            frame_paths.append(frame_path)

            next_frame_at += interval_s
            if next_frame_at <= time.monotonic():
                next_frame_at = time.monotonic() + interval_s

        print(f"Play zamknięty. Kod zakończenia: {play_process.returncode}. Zapisuję GIF…")
        _write_gif(frame_paths, output_path, fps)
        return output_path
    except KeyboardInterrupt:
        if play_process is not None and play_process.poll() is None:
            play_process.terminate()
        raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uruchom play.py i nagraj całą sesję GUI do krótkiego GIF-a."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "chess" / "docs" / "play-ai-vs-ai.gif",
        help="docelowy GIF (domyślnie: chess/docs/play-ai-vs-ai.gif)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="liczba klatek na sekundę; 6 jest dobrym kompromisem dla README",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="maksymalna szerokość GIF-a w pikselach; 0 zachowuje rozmiar okna",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps musi być większe od zera")
    if args.max_width < 0:
        parser.error("--max-width nie może być ujemne")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    saved_path = record_play_demo(output_path.resolve(), args.fps, args.max_width)
    print(f"Gotowe: {saved_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
