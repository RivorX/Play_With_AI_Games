import cv2
import numpy as np
import os
import yaml
import torch
import concurrent.futures
import sys

# Dodaj katalog nadrzędny do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class VideoProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.card_templates = self._load_card_templates()
        self.current_video_name = "video"
        
    def _load_card_templates(self):
        """Ładuje obrazy kart do wykrywania."""
        templates = {}
        assets_path = self.config['paths']['card_assets']
        if not os.path.exists(assets_path):
            print(f"Ostrzeżenie: Brak folderu {assets_path}. Utwórz go i dodaj ikony kart.")
            return templates
            
        print("Ładowanie szablonów kart...")
        for img_name in os.listdir(assets_path):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                name = os.path.splitext(img_name)[0]
                img = cv2.imread(os.path.join(assets_path, img_name), 0) # Grayscale
                if img is not None:
                    # Skalujemy szablony do małego rozmiaru, bo na wideo karty są małe
                    # Wartość 30x30 jest przykładowa, trzeba dobrać eksperymentalnie
                    # Ale do wyszukiwania ręki na pełnym obrazie mogą być potrzebne większe
                    templates[name] = img 
        return templates

    def find_elixir_bar(self, frame):
        """
        Szuka paska eliksiru (fioletowy kolor) aby zlokalizować planszę.
        Zwraca (x, y, w, h) paska lub None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Zakres koloru fioletowego/różowego (Eliksir)
        lower_pink = np.array([140, 100, 100])
        upper_pink = np.array([170, 255, 255])
        
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Szukamy konturów
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_rect = None
        max_area = 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Pasek eliksiru jest poziomy i na dole planszy
            aspect_ratio = w / float(h)
            
            if area > 500 and aspect_ratio > 3: # Zakładamy że jest szeroki
                if area > max_area:
                    max_area = area
                    best_rect = (x, y, w, h)
                    
        return best_rect

    def find_hand_region(self, frame):
        """Szuka obszaru ręki na podstawie kart (tylko w dolnej połowie ekranu)."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Ograniczamy szukanie do dolnej połowy ekranu
        search_y_start = int(h * 0.5)
        search_roi = gray[search_y_start:, :]
        
        best_val = 0
        best_loc = None
        best_template_size = None
        
        # Sprawdzamy kilka popularnych kart jako "kotwice"
        check_cards = ['knight', 'archers', 'giant', 'fireball', 'arrows', 'zap', 'log', 'ice-spirit']
        
        scale_search = 0.5
        small_roi = cv2.resize(search_roi, (0,0), fx=scale_search, fy=scale_search)
        
        for name in check_cards:
            if name not in self.card_templates: continue
            
            template = self.card_templates[name]
            # Skalujemy template do rozmiaru na małym obrazie
            target_h = int(h * 0.12 * scale_search) 
            if target_h < 10: continue
            
            scale = target_h / template.shape[0]
            small_template = cv2.resize(template, (0,0), fx=scale, fy=scale)
            
            res = cv2.matchTemplate(small_roi, small_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                # Przeliczamy współrzędne na oryginał (uwzględniając offset Y)
                best_loc = (int(max_loc[0]/scale_search), int(max_loc[1]/scale_search) + search_y_start)
                
        if best_val > 0.60: # Zwiększony próg pewności
            print(f"Znaleziono kartę (kotwicę): {best_val:.2f} w {best_loc}")
            card_w = int(h * 0.12 * (3.0/4.0)) 
            card_h = int(h * 0.12)
            
            # Szacujemy obszar ręki (4 karty szerokości)
            hand_w = card_w * 4.5
            hand_h = card_h * 1.2
            
            # Centrujemy wokół znalezionej karty
            hand_x = max(0, best_loc[0] - hand_w // 2)
            hand_y = max(0, best_loc[1] - card_h * 0.1)
            
            if hand_x + hand_w > w: hand_x = w - hand_w
            if hand_y + hand_h > h: hand_y = h - hand_h
            
            return (hand_x, hand_y, hand_w, hand_h)
            
        return None

    def find_game_layout(self, frame):
        """
        Niezależnie szuka planszy i ręki. Obsługuje układ pionowy i split-screen.
        """
        h, w = frame.shape[:2]
        
        # 1. Znajdź rękę (Hand ROI)
        hand_rect = self.find_hand_region(frame)
        if hand_rect:
            hx, hy, hw, hh = hand_rect
            self.config['video_layout']['hand_roi'] = [hx/w, hy/h, hw/w, hh/h]
            print(f"Znaleziono RĘKĘ: {self.config['video_layout']['hand_roi']}")
        else:
            # print("Nie znaleziono ręki - pomijam próbę.") 
            return False

        # 2. Znajdź planszę (Board ROI) - szukamy paska eliksiru
        elixir_rect = self.find_elixir_bar(frame)
        
        if elixir_rect:
            ex, ey, ew, eh = elixir_rect
            print(f"Znaleziono PASEK ELIKSIRU: {elixir_rect}")
            
            board_w = ew
            board_h = board_w * (16/9)
            board_x = ex
            board_y = ey - board_h
            if board_y < 0: board_y = 0
            
            self.config['video_layout']['board_roi'] = [board_x/w, board_y/h, board_w/w, board_h/h]
            print(f"Znaleziono PLANSZĘ (wg eliksiru): {self.config['video_layout']['board_roi']}")
            return True
            
        # Fallback: Analiza układu na podstawie pozycji ręki
        hx, hy, hw, hh = hand_rect
        hand_center_x = hx + hw / 2
        
        # Sprawdź czy ręka jest po prawej stronie (Split Screen)
        if hand_center_x > w * 0.6:
            print("Wykryto układ SPLIT SCREEN (Ręka po prawej).")
            # Zakładamy że plansza jest po lewej stronie
            # Zazwyczaj zajmuje lewą połowę ekranu, wycentrowana w pionie lub na dole
            # Spróbujmy standardowego układu 9:16 wpasowanego w lewą połowę
            
            # Szacujemy szerokość planszy jako ~45-50% szerokości ekranu
            board_w = w * 0.48 
            board_h = board_w * (16/9)
            
            board_x = 0 # Od lewej krawędzi
            board_y = (h - board_h) / 2 # Wycentrowana w pionie
            
            # Jeśli wychodzi poza ekran, dociągnij do dołu (częsty układ)
            if board_y + board_h > h:
                board_y = h - board_h
            if board_y < 0: board_y = 0
            
            self.config['video_layout']['board_roi'] = [board_x/w, board_y/h, board_w/w, board_h/h]
            print(f"Zgaduję PLANSZĘ (Split Screen): {self.config['video_layout']['board_roi']}")
            return True
            
        else:
            # Standardowy układ (Ręka na środku/dole)
            print("Wykryto układ STANDARDOWY (Ręka na środku).")
            board_w = hw
            board_h = board_w * (16/9)
            board_x = hx
            board_y = hy - board_h
            if board_y < 0: board_y = 0
            
            self.config['video_layout']['board_roi'] = [board_x/w, board_y/h, board_w/w, board_h/h]
            print(f"Zgaduję PLANSZĘ (Standard): {self.config['video_layout']['board_roi']}")
            return True

    def get_roi(self, frame, roi_config):
        """Wycina fragment obrazu na podstawie znormalizowanych współrzędnych."""
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi_config
        
        x = int(rx * w)
        y = int(ry * h)
        width = int(rw * w)
        height = int(rh * h)
        
        # Zabezpieczenia
        if x < 0: x = 0
        if y < 0: y = 0
        if x + width > w: width = w - x
        if y + height > h: height = h - y
        
        return frame[y:y+height, x:x+width]

    def identify_hand_cards(self, hand_img):
        """
        Rozpoznaje karty w ręce gracza.
        Zwraca listę 4 kart (lub None jeśli puste).
        """
        if hand_img.size == 0: return [None]*4

        # Podziel rękę na 4 sloty
        h, w = hand_img.shape[:2]
        slot_width = w // 4
        
        detected_cards = []
        
        gray_hand = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        for i in range(4):
            slot_img = gray_hand[:, i*slot_width:(i+1)*slot_width]
            if slot_img.size == 0:
                detected_cards.append(None)
                continue

            best_match = None
            best_val = 0
            
            # Sprawdź każdą kartę z bazy
            for name, template in self.card_templates.items():
                # Skalujemy template do rozmiaru slotu
                # Template powinien być nieco mniejszy niż slot
                target_h = int(slot_img.shape[0] * 0.9)
                if target_h <= 0: continue
                
                scale = target_h / template.shape[0]
                resized_template = cv2.resize(template, (0,0), fx=scale, fy=scale)
                
                if resized_template.shape[0] > slot_img.shape[0] or resized_template.shape[1] > slot_img.shape[1]:
                    continue
                    
                res = cv2.matchTemplate(slot_img, resized_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                if max_val > best_val:
                    best_val = max_val
                    best_match = name
            
            # Próg detekcji
            if best_val > 0.5: # Nieco niższy próg
                detected_cards.append(best_match)
            else:
                detected_cards.append(None) 
                
        return detected_cards

    def save_debug_frame(self, frame, frame_idx, detected_cards, action):
        """Zapisuje klatkę z narysowanymi obszarami i detekcją."""
        debug_dir = "clash_royale/debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_img = frame.copy()
        h, w = frame.shape[:2]
        
        # Rysuj Board ROI
        bx, by, bw, bh = self.config['video_layout']['board_roi']
        cv2.rectangle(debug_img, 
                     (int(bx*w), int(by*h)), 
                     (int((bx+bw)*w), int((by+bh)*h)), 
                     (0, 255, 0), 2)
        cv2.putText(debug_img, "BOARD", (int(bx*w), int(by*h)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Rysuj Hand ROI
        hx, hy, hw, hh = self.config['video_layout']['hand_roi']
        cv2.rectangle(debug_img, 
                     (int(hx*w), int(hy*h)), 
                     (int((hx+hw)*w), int((hy+hh)*h)), 
                     (0, 0, 255), 2)
        cv2.putText(debug_img, "HAND", (int(hx*w), int(hy*h)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Wypisz wykryte karty
        info_text = f"Cards: {detected_cards}"
        cv2.putText(debug_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if action:
            action_text = f"ACTION: Slot {action[0]} -> ({action[1]:.2f}, {action[2]:.2f})"
            cv2.putText(debug_img, action_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Narysuj punkt akcji na planszy
            # action[1], action[2] są znormalizowane względem planszy
            board_abs_x = int(bx*w + action[1]*bw*w)
            board_abs_y = int(by*h + action[2]*bh*h)
            cv2.circle(debug_img, (board_abs_x, board_abs_y), 10, (0, 255, 255), -1)

        # Użyj nazwy wideo w nazwie pliku
        safe_name = "".join([c for c in self.current_video_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        cv2.imwrite(os.path.join(debug_dir, f"{safe_name}_frame_{frame_idx:06d}.jpg"), debug_img)

    def detect_card_play(self, frame, prev_frame, current_hand, prev_hand):
        """
        Wykrywa zagranie:
        1. Karta znika z ręki (jest w prev_hand, nie ma w current_hand).
        2. Na planszy pojawia się zmiana (deployment).
        """
        if prev_hand is None or current_hand is None:
            return None
            
        played_card_idx = -1
        played_card_name = None
        
        # Sprawdź czy jakaś karta zniknęła
        for i in range(4):
            if prev_hand[i] is not None and current_hand[i] is None:
                # Karta była, a teraz jej nie ma (puste miejsce)
                played_card_idx = i
                played_card_name = prev_hand[i]
                break
        
        if played_card_idx != -1:
            # Karta została zagrana! Teraz szukamy GDZIE.
            # Analiza różnicy klatek na planszy
            board_roi = self.config['video_layout']['board_roi']
            board_curr = self.get_roi(frame, board_roi)
            board_prev = self.get_roi(prev_frame, board_roi)
            
            if board_curr.size == 0 or board_prev.size == 0: return None

            # Oblicz różnicę
            diff = cv2.absdiff(board_curr, board_prev)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Znajdź środek ciężkości zmian
            M = cv2.moments(thresh)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Normalizacja współrzędnych względem planszy
                h, w = board_curr.shape[:2]
                norm_x = cX / w
                norm_y = cY / h
                
                print(f"Wykryto zagranie: {played_card_name} (Slot {played_card_idx}) na ({norm_x:.2f}, {norm_y:.2f})")
                return (played_card_idx, norm_x, norm_y)
                
        return None

    def process_video(self, video_path):
        self.current_video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        data = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        target_fps = 5
        frame_interval = int(round(fps / target_fps))
        if frame_interval < 1: frame_interval = 1
        
        prev_frame = None
        prev_hand = None
        frame_count = 0
        
        print(f"[{self.current_video_name}] Rozpoczynam przetwarzanie (FPS: {fps})")
        
        game_started = False
        empty_hand_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if not game_started:
                # Skanujemy co 1 sekundę (ok. 30 klatek) w poszukiwaniu planszy
                if frame_count % 30 == 0:
                    # Próbujemy znaleźć układ
                    # Używamy try-except żeby nie wywaliło procesu przy błędzie detekcji
                    try:
                        if self.find_game_layout(frame):
                            print(f"[{self.current_video_name}] START GRY: Wykryto układ w klatce {frame_count}")
                            game_started = True
                            prev_frame = frame
                            empty_hand_frames = 0
                    except Exception as e:
                        print(f"[{self.current_video_name}] Błąd detekcji układu: {e}")
                continue
            
            # --- Gra trwa ---
            
            if frame_count % frame_interval == 0:
                # 1. Wytnij obszary zainteresowania
                try:
                    board_roi = self.config['video_layout']['board_roi']
                    hand_roi = self.config['video_layout']['hand_roi']
                    
                    board_img = self.get_roi(frame, board_roi)
                    hand_img = self.get_roi(frame, hand_roi)
                    
                    # 2. Rozpoznaj karty w ręce
                    current_hand = self.identify_hand_cards(hand_img)
                    
                    # Sprawdzenie czy gra się nie skończyła (pusta ręka przez dłuższy czas)
                    if all(c is None for c in current_hand):
                        empty_hand_frames += 1
                    else:
                        empty_hand_frames = 0
                    
                    # Jeśli przez 30 sekund (150 próbek) nie ma kart, kończymy
                    if empty_hand_frames > 150:
                        print(f"[{self.current_video_name}] KONIEC GRY: Brak kart przez 30s (klatka {frame_count})")
                        break

                    # 3. Resize planszy do modelu (State)
                    if board_img.size > 0:
                        state_img = cv2.resize(board_img, (270, 480))
                    else:
                        state_img = np.zeros((480, 270, 3), dtype=np.uint8)
                    
                    action = None
                    if prev_frame is not None:
                        # 4. Wykryj akcję
                        action = self.detect_card_play(frame, prev_frame, current_hand, prev_hand)
                        
                        if action:
                            data.append((state_img, action))
                        elif len(data) % 10 == 0:
                             data.append((state_img, (4, 0.5, 0.5))) # 4 = wait

                    # LOGOWANIE DEBUGOWE (co 20 przetworzonych klatek lub przy akcji)
                    if action or (len(data) > 0 and len(data) % 20 == 0):
                        self.save_debug_frame(frame, frame_count, current_hand, action)

                    prev_frame = frame
                    prev_hand = current_hand
                except Exception as e:
                    print(f"[{self.current_video_name}] Błąd przetwarzania klatki {frame_count}: {e}")
            
            if frame_count % 2000 == 0:
                print(f"[{self.current_video_name}] Przetworzono {frame_count} klatek...")
                
        cap.release()
        
        if not game_started:
            print(f"[{self.current_video_name}] POMINIĘTO: Nie wykryto rozgrywki w całym pliku.")
            return None
            
        # Minimalna długość gry (np. 30 sekund = 150 próbek przy 5 FPS)
        if len(data) < 150:
            print(f"[{self.current_video_name}] POMINIĘTO: Zbyt krótka rozgrywka ({len(data)} próbek).")
            return None
            
        return data

    def save_dataset(self, data, output_name):
        # Zapisz jako plik .pt (PyTorch) lub .npz
        out_path = os.path.join(self.config['paths']['processed_data'], output_name)
        torch.save(data, out_path)
        print(f"Zapisano dataset: {out_path}")

def process_single_video(video_path):
    """Funkcja pomocnicza dla multiprocessing."""
    # Tworzymy nową instancję procesora dla każdego procesu
    # aby uniknąć problemów z współdzieleniem stanu (np. config layout)
    proc = VideoProcessor("clash_royale/config/config.yaml")
    return proc.process_video(video_path)

if __name__ == "__main__":
    # Wczytaj config tylko po to by pobrać ścieżki
    with open("clash_royale/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_videos']
    all_data = []
    
    if not os.path.exists(raw_dir):
        print(f"Katalog {raw_dir} nie istnieje!")
        exit()
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(".mp4")]
    video_paths = [os.path.join(raw_dir, f) for f in files]
    
    print(f"Znaleziono {len(files)} plików wideo do przetworzenia.")
    
    # Używamy ProcessPoolExecutor do równoległego przetwarzania
    # Liczba procesów = liczba rdzeni CPU (domyślnie)
    successful_count = 0
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_single_video, video_paths)
        
        for video_data in results:
            if video_data is not None:
                all_data.extend(video_data)
                successful_count += 1
            else:
                failed_count += 1
        
    print(f"Przetwarzanie zakończone. Sukces: {successful_count}/{len(video_paths)} wideo.")
    print(f"Łącznie zebrano {len(all_data)} próbek treningowych.")
    
    if len(all_data) > 0:
        # Zapisz dataset używając instancji procesora (może być nowa)
        processor = VideoProcessor("clash_royale/config/config.yaml")
        processor.save_dataset(all_data, "dataset.pt")
    else:
        print("Brak danych do zapisania.")
