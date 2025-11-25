import cv2
import numpy as np

class StateDetector:
    def __init__(self, config):
        self.config = config
        # Pozycje paska eliksiru (do dostosowania pod rozdzielczość 540x960)
        # Zakładamy, że pasek jest na dole. Te wartości trzeba dobrać eksperymentalnie robiąc screenshot.
        self.elixir_y = 930 
        self.elixir_start_x = 150
        self.elixir_end_x = 450
        self.elixir_step = (self.elixir_end_x - self.elixir_start_x) / 10

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
