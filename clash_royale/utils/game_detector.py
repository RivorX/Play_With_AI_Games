import cv2
import numpy as np
import os
from typing import Optional, Tuple

class GameDetector:
    """
    Detektor stanu gry Clash Royale.
    Wykrywa: obszar gry, ikonę chatu, napis WINNER.
    """
    def __init__(self, icons_path: str):
        self.icons_path = icons_path
        self.chat_template = self._load_icon('chat.png')
        self.winner_blue_template = self._load_icon('winner_blue.png')
        self.winner_red_template = self._load_icon('winner_red.png')
        self.cached_game_region = None

    def _load_icon(self, filename: str) -> Optional[np.ndarray]:
        path = os.path.join(self.icons_path, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                print(f"  ✓ Załadowano: {filename} ({img.shape[1]}x{img.shape[0]})")
                return img
        print(f"  ✗ Brak ikony: {filename}")
        return None

    def detect_game_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        if self.cached_game_region is not None:
            return self.cached_game_region
        h, w = frame.shape[:2]
        region = self._detect_game_boundaries(frame)
        if region is None:
            region = self._detect_by_black_bars(frame)
        if region is None:
            region = (0, 0, w, h)
        self.cached_game_region = region
        return region

    def _detect_game_boundaries(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        col_means = np.mean(gray, axis=0)
        col_diff = np.abs(np.diff(col_means))
        kernel_size = max(3, w // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        col_diff_smooth = np.convolve(col_diff, np.ones(kernel_size)/kernel_size, mode='same')
        threshold = np.mean(col_diff_smooth) + 0.5 * np.std(col_diff_smooth)
        peaks = []
        margin = int(w * 0.05)
        for i in range(margin, len(col_diff_smooth) - margin):
            if (col_diff_smooth[i] > threshold and 
                col_diff_smooth[i] > col_diff_smooth[i-1] and 
                col_diff_smooth[i] > col_diff_smooth[i+1]):
                peaks.append((i, col_diff_smooth[i]))
        if len(peaks) < 2:
            return None
        peaks.sort(key=lambda x: x[1], reverse=True)
        x1 = min(peaks[0][0], peaks[1][0])
        x2 = max(peaks[0][0], peaks[1][0])
        game_w = x2 - x1
        if game_w < w * 0.3:
            return None
        aspect = game_w / h
        if aspect < 0.35 or aspect > 0.75:
            return None
        return (x1, 0, game_w, h)

    def _detect_by_black_bars(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        BLACK_THRESHOLD = 15
        col_means = np.mean(gray, axis=0)
        non_black_cols = np.where(col_means > BLACK_THRESHOLD)[0]
        if len(non_black_cols) == 0:
            return None
        x_start = non_black_cols[0]
        x_end = non_black_cols[-1] + 1
        game_w = x_end - x_start
        if game_w < w * 0.3:
            return None
        return (x_start, 0, game_w, h)

    def detect_chat_icon(self, frame: np.ndarray, threshold: float = 0.65) -> bool:
        if self.chat_template is None:
            return False
        h, w = frame.shape[:2]
        search_region = frame[int(h * 0.7):, :int(w * 0.35)]
        if search_region.shape[0] < 20 or search_region.shape[1] < 20:
            return False
        best_match = 0
        for scale in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            scaled_template = cv2.resize(
                self.chat_template, 
                None, 
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            if (scaled_template.shape[0] > search_region.shape[0] or 
                scaled_template.shape[1] > search_region.shape[1]):
                continue
            if scaled_template.shape[0] < 5 or scaled_template.shape[1] < 5:
                continue
            result = cv2.matchTemplate(search_region, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_match = max(best_match, max_val)
        return best_match > threshold

    def detect_winner(self, frame: np.ndarray, threshold: float = 0.55) -> Optional[str]:
        h, w = frame.shape[:2]
        y_end = int(h * 0.25)
        top_region = frame[0:y_end, :]
        if top_region.shape[0] < 30:
            return None
        blue_score = 0
        if self.winner_blue_template is not None:
            blue_score = self._match_winner_template(top_region, self.winner_blue_template)
        red_score = 0
        if self.winner_red_template is not None:
            red_score = self._match_winner_template(top_region, self.winner_red_template)
        if blue_score > threshold and blue_score > red_score:
            return 'blue'
        elif red_score > threshold and red_score > blue_score:
            return 'red'
        return None

    def _match_winner_template(self, image: np.ndarray, template: np.ndarray) -> float:
        best_match = 0
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        image_edges = cv2.Canny(image_gray, 50, 150)
        template_edges = cv2.Canny(template_gray, 50, 150)
        for scale in [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            scaled_template = cv2.resize(
                template_edges, 
                None, 
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            if (scaled_template.shape[0] > image_edges.shape[0] or 
                scaled_template.shape[1] > image_edges.shape[1]):
                continue
            if scaled_template.shape[0] < 8 or scaled_template.shape[1] < 8:
                continue
            result = cv2.matchTemplate(image_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_match = max(best_match, max_val)
            scaled_template_gray = cv2.resize(
                template_gray, 
                None, 
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            if (scaled_template_gray.shape[0] <= image_gray.shape[0] and 
                scaled_template_gray.shape[1] <= image_gray.shape[1]):
                result2 = cv2.matchTemplate(image_gray, scaled_template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val2, _, _ = cv2.minMaxLoc(result2)
                best_match = max(best_match, max_val2)
        return best_match
