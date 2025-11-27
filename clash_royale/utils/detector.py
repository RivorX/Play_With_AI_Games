import cv2
import numpy as np
import os

class StateDetector:
    def __init__(self, config):
        self.config = config
        # Pozycje paska eliksiru (do dostosowania pod rozdzielczość 540x960)
        # Zakładamy, że pasek jest na dole. Te wartości trzeba dobrać eksperymentalnie robiąc screenshot.
        self.elixir_y = 930 
        self.elixir_start_x = 150
        self.elixir_end_x = 450
        self.elixir_step = (self.elixir_end_x - self.elixir_start_x) / 10
        
        # Załaduj ikonę czatu
        self.chat_icon = self._load_chat_icon()
        
    def _load_chat_icon(self):
        """Ładuje ikonę czatu do template matching."""
        chat_path = os.path.join(self.config['paths'].get('icons_assets', 'clash_royale/assets/icons'), 'chat.png')
        if os.path.exists(chat_path):
            icon = cv2.imread(chat_path, cv2.IMREAD_GRAYSCALE)
            if icon is not None:
                print(f"Załadowano ikonę czatu z {chat_path}")
                return icon
        print(f"Ostrzeżenie: Brak ikony czatu w {chat_path}")
        return None

    def get_elixir_level(self, frame):
        """
        Analizuje pasek eliksiru i zwraca szacowaną ilość (0-10).
        Sprawdza kolor pikseli w miejscach gdzie powinny być segmenty eliksiru.
        """
        # Kolor eliksiru to zazwyczaj fioletowy/różowy
        # W przestrzeni HSV fiolet to około 130-160 Hue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        elixir = 0
        for i in range(1, 11):
            # Sprawdź punkt odpowiadający i-temu eliksirowi
            x = int(self.elixir_start_x + (i - 0.5) * self.elixir_step)
            y = self.elixir_y
            
            if y >= frame.shape[0] or x >= frame.shape[1]:
                continue

            pixel = hsv[y, x]
            
            # Prosty próg na kolor fioletowy (Hue: 120-170, Saturation: >50, Value: >50)
            if 120 < pixel[0] < 170 and pixel[1] > 50 and pixel[2] > 50:
                elixir = i
            else:
                # Jeśli ten segment nie świeci, to kolejne też nie powinny (chyba że ładuje się)
                # Ale dla bezpieczeństwa sprawdzamy wszystkie lub przerywamy
                break
                
        return elixir

    def get_available_cards(self, frame):
        """
        Zwraca listę dostępnych kart (ich nazwy) w ręce (sloty 0-3).
        Wymaga template matchingu z załadowanymi assetami.
        """
        # Tu można dodać logikę rozpoznawania kart w 4 slotach na dole ekranu
        # Na razie zwracamy placeholder
        return ["knight", "archers", "giant", "fireball"] # Przykładowa ręka

    def detect_chat_icon(self, frame):
        """
        Wykrywa czy ikona czatu jest widoczna na ekranie.
        Ikona czatu widoczna = gra się toczy.
        
        Szuka w lewym dolnym rogu ekranu.
        """
        if self.chat_icon is None:
            return False
        
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Szukaj w lewym dolnym rogu (tam jest ikona czatu)
        search_region = gray[int(height*0.75):, :int(width*0.3)]
        
        # Przeskaluj ikonę do odpowiedniego rozmiaru
        # Ikona czatu jest mała, ok. 8-12% szerokości ekranu
        target_size = int(width * 0.15)
        if target_size < 20:
            target_size = 20
        
        scale = target_size / max(self.chat_icon.shape)
        scaled_icon = cv2.resize(self.chat_icon, None, fx=scale, fy=scale)
        
        if scaled_icon.shape[0] > search_region.shape[0] or scaled_icon.shape[1] > search_region.shape[1]:
            return False
        
        result = cv2.matchTemplate(search_region, scaled_icon, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max_val > 0.6
    
    def detect_winner(self, frame):
        """
        Wykrywa napis 'Winner!' na ekranie i określa pozycję.
        
        Returns:
            'top' - Winner u góry (przeciwnik wygrał)
            'bottom' - Winner u dołu (my wygraliśmy)
            None - Brak napisu Winner
        """
        height, width = frame.shape[:2]
        
        # Konwersja do HSV - Winner jest złoty/żółty
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Zakres koloru złotego/żółtego (Winner banner)
        lower_gold = np.array([15, 100, 100])
        upper_gold = np.array([45, 255, 255])
        
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Podziel ekran na górną i dolną połowę
        mid_y = height // 2
        
        top_region = mask[:mid_y, :]
        bottom_region = mask[mid_y:, :]
        
        top_gold = cv2.countNonZero(top_region)
        bottom_gold = cv2.countNonZero(bottom_region)
        
        # Próg - musi być znacząca ilość złotego
        threshold = (width * mid_y) * 0.02  # 2% obszaru
        
        if top_gold > threshold and top_gold > bottom_gold * 1.5:
            # Winner u góry - przeciwnik wygrał
            return 'top'
        elif bottom_gold > threshold and bottom_gold > top_gold * 1.5:
            # Winner u dołu - my wygraliśmy
            return 'bottom'
        
        # Dodatkowe sprawdzenie - szukamy wzorca tekstu 'Winner'
        # Używamy detekcji konturów w środkowej części
        center_y_start = int(height * 0.3)
        center_y_end = int(height * 0.7)
        center_mask = mask[center_y_start:center_y_end, :]
        
        gold_in_center = cv2.countNonZero(center_mask)
        center_threshold = (width * (center_y_end - center_y_start)) * 0.03
        
        if gold_in_center > center_threshold:
            # Jest dużo złotego w środku - sprawdź który gracz
            # Znajdź środek masy złotych pikseli
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                center_y = int(moments['m01'] / moments['m00'])
                if center_y < mid_y:
                    return 'top'
                else:
                    return 'bottom'
        
        return None

    def check_victory(self, frame):
        """
        Sprawdza czy na ekranie jest napis Victory / Korony.
        Dla uproszczenia sprawdzamy obecność dużej ilości koloru złotego/żółtego w górnej części ekranu
        lub konkretnego wzorca (jeśli mamy asset 'victory.png').
        """
        # Konwersja do HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Zakres koloru złotego/żółtego (Victory banner)
        lower_gold = np.array([20, 100, 100])
        upper_gold = np.array([40, 255, 255])
        
        # Maska
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Sprawdzamy czy w środkowej części ekranu jest dużo złotego
        height, width = frame.shape[:2]
        center_region = mask[int(height*0.2):int(height*0.5), int(width*0.2):int(width*0.8)]
        
        gold_pixels = cv2.countNonZero(center_region)
        total_pixels = center_region.size
        
        ratio = gold_pixels / total_pixels
        
        # Jeśli więcej niż 5% środka ekranu jest złote, to prawdopodobnie Victory
        # (To jest heurystyka, lepiej użyć template matching napisu "VICTORY")
        return ratio > 0.05
