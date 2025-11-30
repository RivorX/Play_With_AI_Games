"""
Segmentacja gier Clash Royale z wideo.

Funkcjonalności:
1. Dynamiczne wykrywanie obszaru gry (wycinanie letterbox)
2. Wykrywanie ikony chatu (oznacza że gra trwa)
3. Wykrywanie napisu WINNER z rozróżnieniem koloru:
   - NIEBIESKI = my wygraliśmy (zachowujemy)
   - CZERWONY = przeciwnik wygrał (odrzucamy)

OPTYMALIZACJE WYDAJNOŚCI:
- Używa seek zamiast sekwencyjnego czytania klatek (2-3x szybciej)
- Analizuje co 1 sekundę zamiast każdej klatki (30x mniej obliczeń)
- Multiprocessing dla wielu wideo równolegle
"""

import cv2
import numpy as np
import os
import yaml
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class GameSegment:
    """Reprezentuje wykryty segment gry."""
    start_frame: int
    end_frame: int
    fps: float
    we_won: bool  # True = niebieski winner, False = czerwony winner
    game_region: Tuple[int, int, int, int]  # x, y, w, h - obszar gry
    
    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps
    
    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps
    
    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps


class GameDetector:
    """
    Detektor stanu gry Clash Royale.
    Wykrywa: obszar gry, ikonę chatu, napis WINNER.
    
    JAK DZIAŁA WYKRYWANIE REGIONU GRY:
    ===================================
    Problem: YouTuberzy dodają kamerki/overlaye Z BOKU ekranu gry.
from clash_royale.utils.game_detector import GameDetector

    
    Rozwiązanie:
    1. Szukamy PIONOWYCH granic gry (lewa/prawa krawędź)
    2. Używamy PEŁNEJ wysokości - NIE przycinamy góry ani dołu!
    3. Nie wymuszamy proporcji - różne telefony mają różne proporcje (9:16, 9:19.5, 9:20)
    
    Metody detekcji:
    1. _detect_game_boundaries - szuka pionowych krawędzi przez gradient jasności
    2. _detect_by_black_bars - fallback dla czarnych pasków letterbox
    """
    
    def __init__(self, icons_path: str):
        self.icons_path = icons_path
        
        # Załaduj wzorce
        self.chat_template = self._load_icon('chat.png')
        self.winner_blue_template = self._load_icon('winner_blue.png')
        self.winner_red_template = self._load_icon('winner_red.png')
        
        # Cache dla wykrytego regionu gry
        self.cached_game_region = None
        
    def _load_icon(self, filename: str) -> Optional[np.ndarray]:
        """Ładuje ikonę w kolorze."""
        path = os.path.join(self.icons_path, filename)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                print(f"  ✓ Załadowano: {filename} ({img.shape[1]}x{img.shape[0]})")
                return img
        print(f"  ✗ Brak ikony: {filename}")
        return None
    
    def detect_game_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Wykrywa obszar gry Clash Royale na klatce.
        
        KLUCZOWE: Używamy PEŁNEJ wysokości! Nie przycinamy góry ani dołu!
        Tam są ważne elementy: pasek HP, timer, karty, elixir, "Next:".
        
        Returns:
            (x, y, width, height) - prostokąt z grą
        """
        if self.cached_game_region is not None:
            return self.cached_game_region
        
        h, w = frame.shape[:2]
        
        # Metoda 1: Szukaj pionowych granic przez gradient
        region = self._detect_game_boundaries(frame)
        
        # Metoda 2: Fallback - czarne paski (letterbox)
        if region is None:
            region = self._detect_by_black_bars(frame)
        
        # Metoda 3: Ostateczność - cała klatka
        if region is None:
            region = (0, 0, w, h)
        
        self.cached_game_region = region
        return region
    
    def _detect_game_boundaries(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Wykrywa region gry szukając PIONOWYCH granic.
        
        Algorytm:
        1. Analizuj pionowe gradienty jasności (gdzie są ostre krawędzie)
        2. Znajdź dwie wyraźne pionowe linie (lewa i prawa granica gry)
        3. Użyj PEŁNEJ wysokości klatki (y=0, h=pełna wysokość)
        4. NIE wymuszaj proporcji - zachowaj pełną wysokość!
        """
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Oblicz średnią jasność każdej kolumny
        col_means = np.mean(gray, axis=0)
        
        # Oblicz gradient (różnica między sąsiednimi kolumnami)
        col_diff = np.abs(np.diff(col_means))
        
        # Wygładź gradient żeby usunąć szum
        kernel_size = max(3, w // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        col_diff_smooth = np.convolve(col_diff, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Znajdź szczyty gradientu (potencjalne granice)
        threshold = np.mean(col_diff_smooth) + 0.5 * np.std(col_diff_smooth)
        peaks = []
        
        # Ignoruj krawędzie (pierwsze/ostatnie 5% obrazu)
        margin = int(w * 0.05)
        
        for i in range(margin, len(col_diff_smooth) - margin):
            if (col_diff_smooth[i] > threshold and 
                col_diff_smooth[i] > col_diff_smooth[i-1] and 
                col_diff_smooth[i] > col_diff_smooth[i+1]):
                peaks.append((i, col_diff_smooth[i]))
        
        if len(peaks) < 2:
            return None
        
        # Sortuj według siły gradientu (najsilniejsze granice)
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Weź dwa najsilniejsze szczyty jako granice
        x1 = min(peaks[0][0], peaks[1][0])
        x2 = max(peaks[0][0], peaks[1][0])
        
        game_w = x2 - x1
        
        # Minimalna szerokość to 30% całego obrazu
        if game_w < w * 0.3:
            return None
        
        # Sprawdź czy proporcje są sensowne (szerokość/wysokość między 0.4 a 0.7)
        aspect = game_w / h
        if aspect < 0.35 or aspect > 0.75:
            # Proporcje nie pasują do Clash Royale, spróbuj czarnych pasków
            return None
        
        # Zwróć region z PEŁNĄ wysokością!
        return (x1, 0, game_w, h)
    
    def _detect_by_black_bars(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Fallback: Wykrywa region przez usunięcie czarnych pasków (letterbox).
        
        Szuka tylko POZIOMYCH czarnych pasków (z boków).
        Zachowuje PEŁNĄ wysokość - nie przycina góry ani dołu!
        """
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        BLACK_THRESHOLD = 15
        
        # Znajdź nie-czarne KOLUMNY (szukamy pionowych granic)
        col_means = np.mean(gray, axis=0)
        non_black_cols = np.where(col_means > BLACK_THRESHOLD)[0]
        
        if len(non_black_cols) == 0:
            return None
        
        x_start = non_black_cols[0]
        x_end = non_black_cols[-1] + 1
        game_w = x_end - x_start
        
        # Sprawdź minimalna szerokość (30% obrazu)
        if game_w < w * 0.3:
            return None
        
        # Zwróć z PEŁNĄ wysokością (y=0, h=pełna wysokość)
        return (x_start, 0, game_w, h)
    
    def detect_game_region_multi_frame(self, cap: cv2.VideoCapture, 
                                        num_samples: int = 5) -> Tuple[int, int, int, int]:
        """
        Wykrywa region gry analizując wiele klatek.
        Bardziej niezawodne niż pojedyncza klatka.
        """
        original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        candidates = []
        
        # Próbkuj klatki z różnych miejsc wideo
        sample_positions = np.linspace(total_frames * 0.1, total_frames * 0.5, num_samples).astype(int)
        
        for pos in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Resetuj cache dla każdej próby
            self.cached_game_region = None
            region = self.detect_game_region(frame)
            
            if region:
                candidates.append(region)
        
        # Przywróć pozycję
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        self.cached_game_region = None
        
        if not candidates:
            return (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Użyj mediany dla stabilności
        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        ws = [c[2] for c in candidates]
        hs = [c[3] for c in candidates]
        
        result = (
            int(np.median(xs)),
            int(np.median(ys)),
            int(np.median(ws)),
            int(np.median(hs))
        )
        
        self.cached_game_region = result
        return result
    
    def crop_to_game(self, frame: np.ndarray) -> np.ndarray:
        """Przycina klatkę do obszaru gry."""
        x, y, w, h = self.detect_game_region(frame)
        return frame[y:y+h, x:x+w]
    
    def detect_chat_icon(self, frame: np.ndarray, threshold: float = 0.65) -> bool:
        """
        Wykrywa ikonę chatu na ekranie.
        Chat widoczny = gra się toczy.
        
        Szuka w dolnej części ekranu (lewy dolny róg).
        """
        if self.chat_template is None:
            return False
        
        h, w = frame.shape[:2]
        
        # Szukaj w lewej dolnej ćwiartce (tam jest chat)
        search_region = frame[int(h * 0.7):, :int(w * 0.35)]
        
        if search_region.shape[0] < 20 or search_region.shape[1] < 20:
            return False
        
        # Wieloskalowe template matching - MAŁE skale dla małych ikon!
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
        """
        Wykrywa ekran końca gry i określa kto wygrał.
        
        Używa template matching z obrazkami winner_blue.png i winner_red.png.
        Szuka napisu "WINNER" w górnej części ekranu.
        
        Returns:
            'blue' - MY wygraliśmy (cyjanowy napis WINNER)
            'red' - PRZECIWNIK wygrał (różowy napis WINNER)
            None - brak ekranu końca gry (normalny gameplay)
        """
        h, w = frame.shape[:2]
        
        # Region górny gdzie jest napis WINNER (0-25% wysokości)
        y_end = int(h * 0.25)
        top_region = frame[0:y_end, :]
        
        if top_region.shape[0] < 30:
            return None
        
        # Sprawdź dopasowanie dla niebieskiego WINNER
        blue_score = 0
        if self.winner_blue_template is not None:
            blue_score = self._match_winner_template(top_region, self.winner_blue_template)
        
        # Sprawdź dopasowanie dla czerwonego WINNER  
        red_score = 0
        if self.winner_red_template is not None:
            red_score = self._match_winner_template(top_region, self.winner_red_template)
        
        # Wybierz lepsze dopasowanie jeśli przekracza próg
        if blue_score > threshold and blue_score > red_score:
            return 'blue'
        elif red_score > threshold and red_score > blue_score:
            return 'red'
        
        return None
    
    def _match_winner_template(self, image: np.ndarray, template: np.ndarray) -> float:
        """
        Dopasowuje template WINNER z tolerancją kolorów.
        Używa edge detection aby być odpornym na różnice kolorystyczne.
        """
        best_match = 0
        
        # Konwertuj do skali szarości (niezależne od koloru)
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        # Wykryj krawędzie (kształt napisu)
        image_edges = cv2.Canny(image_gray, 50, 150)
        template_edges = cv2.Canny(template_gray, 50, 150)
        
        # Wieloskalowe dopasowanie
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
            
            # Template matching na krawędziach
            result = cv2.matchTemplate(image_edges, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_match = max(best_match, max_val)
            
            # Dodatkowo sprawdź oryginalny obraz (dla pewności)
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
    
    def _multi_scale_match(self, image: np.ndarray, template: np.ndarray) -> float:
        """Template matching w wielu skalach."""
        best_match = 0
        
        # Małe skale - ikony są duże, gra jest mała
        for scale in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
            scaled = cv2.resize(
                template, 
                None, 
                fx=scale, 
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
            
            if (scaled.shape[0] > image.shape[0] or 
                scaled.shape[1] > image.shape[1]):
                continue
            
            if scaled.shape[0] < 8 or scaled.shape[1] < 8:
                continue
            
            result = cv2.matchTemplate(image, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_match = max(best_match, max_val)
        
        return best_match


class GameSegmenter:
    """
    Segmentuje wideo na poszczególne gry.
    Zapisuje klatki jako sekwencję obrazów PNG.
    
    CZASY GRY W CLASH ROYALE:
    =========================
    - Normalna gra: 3 minuty (180s)
    - Overtime: +2 minuty (120s) 
    - Sudden Death: +2 minuty (120s)
    - MAKSYMALNIE: ~7 minut (420s)
    
    YouTuberzy czasami skipują końcówkę, więc jeśli gra trwa > 420s
    bez wykrycia WINNER, zapisujemy i tak (prawdopodobnie wygrana).
    """
    
    # Maksymalny czas gry w Clash Royale (7 minut + bufor)
    MAX_GAME_DURATION = 450  # 7.5 minuty
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        icons_path = self.config['paths'].get('icons_assets', 'clash_royale/assets/icons')
        print(f"\nŁadowanie ikon z: {icons_path}")
        self.detector = GameDetector(icons_path)

        # Parametry segmentacji
        self.min_game_duration = 30   # Minimalna długość gry w sekundach
        self.max_game_duration = self.MAX_GAME_DURATION  # Maksymalna długość gry
        self.chat_timeout = 10        # Sekundy bez chatu = koniec gry (zwiększone)
        self.target_fps = 5           # Docelowa ilość klatek na sekundę
        self.winner_cooldown = 15     # Sekundy cooldownu po wykryciu WINNER
        self.analysis_interval = 1.0  # Sekundy między analizami (szybsze = 0.5, wolniejsze = 1.0)

        # Przełącznik: czy zapisywać przegrane gry (czerwony winner)
        self.save_lost_games = self.config.get('game', {}).get('save_lost_games', False)
        
    def find_games(self, video_path: str) -> List[GameSegment]:
        """
        Znajduje wszystkie segmenty gier w wideo.
        
        Logika:
        1. Gra ZACZYNA się gdy pojawia się chat (i minął cooldown po poprzednim WINNER)
        2. Gra KOŃCZY się gdy:
           - Wykryjemy WINNER (niebieski lub czerwony)
           - Chat zniknie na > chat_timeout sekund
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ Nie można otworzyć: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"\nAnalizuję: {os.path.basename(video_path)}")
        print(f"  Czas trwania: {duration:.1f}s ({total_frames} klatek @ {fps:.1f} FPS)")
        
        # Wykryj region gry używając wielu klatek (bardziej niezawodne)
        print(f"  Wykrywam region gry (analiza wielu klatek)...")
        game_region = self.detector.detect_game_region_multi_frame(cap, num_samples=5)
        x, y, w, h = game_region
        aspect = w / h if h > 0 else 0
        print(f"  Region gry: {w}x{h} @ ({x}, {y}) [proporcje: {aspect:.3f}, oczekiwane: 0.5625]")
        
        segments = []
        
        # Stan automatu
        in_game = False
        game_start = None
        last_chat_frame = None
        winner_color = None
        last_winner_frame = None  # Cooldown po wykryciu WINNER
        
        # Sprawdzaj co analysis_interval sekund (domyślnie 1s dla szybkości)
        check_interval = max(1, int(fps * self.analysis_interval))
        
        # Liczba klatek cooldownu po WINNER
        winner_cooldown_frames = int(self.winner_cooldown * fps)
        
        # Używaj seek zamiast czytania każdej klatki (DUŻO szybsze!)
        frame_idx = 0
        
        while frame_idx < total_frames:
            # Skocz do konkretnej klatki zamiast czytać wszystkie
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Przytnij do regionu gry
            game_frame = frame[y:y+h, x:x+w]
            
            current_time = frame_idx / fps
            
            # Sprawdź czy jesteśmy w cooldownie po WINNER
            in_cooldown = (last_winner_frame is not None and 
                           frame_idx - last_winner_frame < winner_cooldown_frames)
            
            # Sprawdź chat (ale nie wykrywaj nowej gry podczas cooldownu)
            has_chat = self.detector.detect_chat_icon(game_frame)
            
            # Sprawdź winner (tylko gdy jesteśmy w grze)
            winner = None
            if in_game and not in_cooldown:
                winner = self.detector.detect_winner(game_frame)
            
            if has_chat:
                last_chat_frame = frame_idx
                
                if not in_game and not in_cooldown:
                    # START GRY (tylko jeśli nie w cooldownie)
                    in_game = True
                    game_start = frame_idx
                    winner_color = None
                    print(f"  ▶ START GRY @ {current_time:.1f}s")
            
            # Wykryto winner
            if winner is not None and in_game:
                winner_color = winner
                color_name = "NIEBIESKI (my)" if winner == 'blue' else "CZERWONY (przeciwnik)"
                print(f"  🏆 WINNER {color_name} @ {current_time:.1f}s")
                
                # Koniec gry po wykryciu winnera
                self._end_game(
                    segments, game_start, frame_idx, fps, 
                    winner_color, game_region
                )
                
                # Ustaw cooldown
                last_winner_frame = frame_idx
                in_game = False
                game_start = None
                winner_color = None
            
            # Sprawdź timeout chatu (gra przerwana)
            elif in_game and last_chat_frame is not None:
                frames_without_chat = frame_idx - last_chat_frame
                seconds_without_chat = frames_without_chat / fps
                
                # Oblicz aktualny czas trwania gry
                current_game_duration = (frame_idx - game_start) / fps if game_start else 0
                
                if seconds_without_chat > self.chat_timeout:
                    # Jeśli gra trwała wystarczająco długo, może YouTuber skipnął ending
                    if current_game_duration >= self.max_game_duration:
                        print(f"  ⏹ MAX CZAS (gra trwała {current_game_duration:.1f}s, skip endingu?) @ {current_time:.1f}s")
                        # Zapisujemy jako wygraną (zakładamy że YouTuber pokazuje głównie wygrane)
                        self._end_game(
                            segments, game_start, frame_idx, fps, 
                            'blue', game_region  # Zakładamy wygraną
                        )
                    else:
                        print(f"  ⏹ TIMEOUT (brak chatu {seconds_without_chat:.1f}s) @ {current_time:.1f}s")
                        # Nie zapisujemy - za krótka gra bez winnera
                    
                    in_game = False
                    game_start = None
                    winner_color = None
            
            # Sprawdź czy gra nie trwa za długo (zapobiega nieskończonej grze)
            elif in_game and game_start is not None:
                current_game_duration = (frame_idx - game_start) / fps
                if current_game_duration > self.max_game_duration + 60:  # +60s bufor
                    print(f"  ⚠ Gra przekroczyła max czas ({current_game_duration:.1f}s) - wymuszam koniec")
                    self._end_game(
                        segments, game_start, frame_idx, fps, 
                        'blue', game_region  # Zakładamy wygraną
                    )
                    in_game = False
                    game_start = None
                    winner_color = None
            
            # Postęp co 10%
            progress = (frame_idx / total_frames) * 100
            if (frame_idx // check_interval) % max(1, (total_frames // check_interval) // 10) == 0:
                print(f"  Postęp: {progress:.0f}%", end='\r')
            
            # Następna klatka do analizy
            frame_idx += check_interval
        
        cap.release()
        print(f"  Postęp: 100%")
        
        # Jeśli gra trwała do końca (nie zakończona)
        if in_game and game_start is not None:
            print(f"  ⏹ KONIEC WIDEO (gra nie zakończona)")
            # Nie zapisujemy - nie ma winnera
        
        return segments
    
    def _end_game(self, segments: List[GameSegment], 
                  start_frame: int, end_frame: int, fps: float,
                  winner_color: Optional[str], game_region: Tuple[int, int, int, int]):
        """Zapisuje zakończoną grę do listy segmentów."""

        duration = (end_frame - start_frame) / fps

        if duration < self.min_game_duration:
            print(f"    → Pomijam (za krótka: {duration:.1f}s < {self.min_game_duration}s)")
            return

        if winner_color is None:
            print(f"    → Pomijam (brak winnera)")
            return

        we_won = (winner_color == 'blue')

        # Nowa logika: zapisywanie przegranych gier jeśli przełącznik aktywny
        if not we_won and not self.save_lost_games:
            print(f"    → Pomijam (przegraliśmy - czerwony winner)")
            return

        segment = GameSegment(
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            we_won=we_won,
            game_region=game_region
        )

        segments.append(segment)
        if we_won:
            print(f"    ✓ Znaleziono wygraną grę ({duration:.1f}s) - do ekstrakcji")
        else:
            print(f"    ✓ Znaleziono przegraną grę ({duration:.1f}s) - do ekstrakcji")
    
    def extract_segment(self, video_path: str, segment: GameSegment, 
                        output_dir: str, segment_idx: int) -> str:
        """
        Wycina segment gry i zapisuje jako sekwencję obrazów PNG.
        Przycina do obszaru gry i skaluje do docelowej rozdzielczości.
        
        Struktura wyjściowa:
        output_dir/
            video_name_game1_WIN/
                frame_00000.png
                frame_00001.png
                ...
                metadata.yaml
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        folder_name = f"{base_name}_game{segment_idx + 1}_WIN"
        output_folder = os.path.join(output_dir, folder_name)
        
        # Utwórz folder dla sekwencji klatek
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n  Wycinam: {folder_name}/")
        print(f"    Czas: {segment.start_sec:.1f}s - {segment.end_sec:.1f}s ({segment.duration_sec:.1f}s)")
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
        
        # Docelowa rozdzielczość
        target_width = self.config['game']['screen_width']
        target_height = self.config['game']['screen_height']
        
        # Oblicz interwał klatek dla docelowego FPS
        frame_interval = max(1, int(round(segment.fps / self.target_fps)))
        
        x, y, w, h = segment.game_region
        current_frame = segment.start_frame
        frames_written = 0
        
        while current_frame <= segment.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (current_frame - segment.start_frame) % frame_interval == 0:
                # Przytnij do regionu gry
                cropped = frame[y:y+h, x:x+w]
                
                # Skaluj do docelowej rozdzielczości
                resized = cv2.resize(cropped, (target_width, target_height))
                
                # Zapisz jako PNG
                frame_filename = f"frame_{frames_written:05d}.png"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                
                frames_written += 1
            
            current_frame += 1
        
        cap.release()
        
        # Zapisz metadane
        metadata = {
            'source_video': os.path.basename(video_path),
            'segment_index': segment_idx + 1,
            'start_time_sec': segment.start_sec,
            'end_time_sec': segment.end_sec,
            'duration_sec': segment.duration_sec,
            'original_fps': segment.fps,
            'target_fps': self.target_fps,
            'frames_count': frames_written,
            'resolution': {
                'width': target_width,
                'height': target_height
            },
            'game_region': {
                'x': x,
                'y': y,
                'width': w,
                'height': h
            },
            'we_won': segment.we_won
        }
        
        metadata_path = os.path.join(output_folder, 'metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"    ✓ Zapisano: {frames_written} klatek PNG + metadata.yaml")
        
        return output_folder


def load_processed_archive(archive_path: str) -> set:
    """Wczytuje listę już przetworzonych plików wideo."""
    if os.path.exists(archive_path):
        with open(archive_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_to_archive(archive_path: str, video_filename: str):
    """Dodaje plik wideo do archiwum przetworzonych."""
    with open(archive_path, 'a', encoding='utf-8') as f:
        f.write(video_filename + '\n')


def process_single_video(args: Tuple[str, str, str, str, int, int]) -> dict:
    """
    Przetwarza pojedyncze wideo. Używane przez multiprocessing.
    
    Args:
        args: (video_path, config_path, segmented_dir, archive_path, video_idx, total_videos)
    
    Returns:
        dict z wynikami przetwarzania
    """
    video_path, config_path, segmented_dir, archive_path, video_idx, total_videos = args
    video_file = os.path.basename(video_path)
    
    result = {
        'video_file': video_file,
        'success': False,
        'segments_found': 0,
        'extracted_files': [],
        'error': None
    }
    
    try:
        print(f"\n[{video_idx}/{total_videos}] Przetwarzam: {video_file}")
        
        segmenter = GameSegmenter(config_path)
        
        # FAZA 1: Analiza wideo
        segments = segmenter.find_games(video_path)
        result['segments_found'] = len(segments)
        
        if not segments:
            print(f"  ℹ Brak wygranych gier do wyodrębnienia.")
            save_to_archive(archive_path, video_file)
            result['success'] = True
            return result
        
        # FAZA 2: Ekstrakcja segmentów
        print(f"\n📁 FAZA 2: Ekstrakcja {len(segments)} wygranych gier jako PNG...")
        
        for idx, segment in enumerate(segments):
            output_path = segmenter.extract_segment(
                video_path, segment, segmented_dir, idx
            )
            result['extracted_files'].append(output_path)
        
        save_to_archive(archive_path, video_file)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def main():
    config_path = "clash_royale/config/config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_videos']
    segmented_dir = config['paths'].get('segmented_games', 'clash_royale/data/segmented_games')
    
    # Plik archiwum przetworzonych nagrań
    archive_path = os.path.join(segmented_dir, 'processed_archive.txt')
    
    os.makedirs(segmented_dir, exist_ok=True)
    
    if not os.path.exists(raw_dir):
        print(f"✗ Katalog nie istnieje: {raw_dir}")
        return
    
    # Wczytaj archiwum przetworzonych
    processed_videos = load_processed_archive(archive_path)
    
    # Znajdź oryginalne wideo (bez _game w nazwie)
    video_extensions = ('.mp4', '.webm', '.mkv', '.avi')
    all_videos = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(video_extensions) and '_game' not in f
    ]
    
    # Filtruj już przetworzone
    videos = [v for v in all_videos if v not in processed_videos]
    
    if not all_videos:
        print("✗ Brak plików wideo do przetworzenia!")
        return
    
    # Liczba procesów = liczba rdzeni CPU (minus 1 dla systemu)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Znaleziono {len(all_videos)} wideo w folderze")
    if processed_videos:
        print(f"Już przetworzonych: {len(processed_videos)} (pominięte)")
    print(f"Do przetworzenia: {len(videos)}")
    print(f"Liczba procesów: {num_workers}")
    print(f"Segmenty będą zapisane w: {segmented_dir}")
    print(f"Archiwum przetworzonych: {archive_path}")
    print("="*60)
    
    if not videos:
        print("\n✓ Wszystkie nagrania zostały już przetworzone!")
        print("  Aby przetworzyć ponownie, usuń plik: processed_archive.txt")
        return
    
    total_wins = 0
    extracted_files = []
    
    # Przygotuj argumenty dla każdego wideo
    tasks = [
        (os.path.join(raw_dir, video_file), config_path, segmented_dir, archive_path, idx, len(videos))
        for idx, video_file in enumerate(videos, 1)
    ]
    
    # Jeśli tylko 1 wideo, przetwarzaj sekwencyjnie (bez overhead multiprocessing)
    if len(videos) == 1:
        result = process_single_video(tasks[0])
        if result['success']:
            total_wins += len(result['extracted_files'])
            extracted_files.extend(result['extracted_files'])
    else:
        # Przetwarzaj równolegle
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_video, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    total_wins += len(result['extracted_files'])
                    extracted_files.extend(result['extracted_files'])
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("PODSUMOWANIE")
    print("="*60)
    print(f"Przetworzono wideo: {len(videos)}")
    print(f"Wyodrębniono wygranych gier: {total_wins}")
    print(f"Segmenty zapisane w: {segmented_dir}")
    
    if extracted_files:
        print(f"\nWyodrębnione foldery z klatkami:")
        for f in extracted_files:
            print(f"  • {os.path.basename(f)}/")
    
    print("\nKolejny krok: Uruchom '4_process_data.py' aby przetworzyć dane")


if __name__ == "__main__":
    main()