import mss
import numpy as np
import cv2
import time
import subprocess
from ppadb.client import Client as AdbClient

class EmulatorInterface:
    def __init__(self, config):
        self.config = config
        self.sct = mss.mss()
        
        # Konfiguracja obszaru zrzutu ekranu (monitor 1, lub konkretne okno - tutaj uproszczone)
        # W prawdziwym rozwiązaniu warto znaleźć okno po nazwie (np. win32gui na Windows)
        self.monitor = {
            "top": config['game']['crop_top'], 
            "left": config['game']['crop_left'], 
            "width": config['game']['screen_width'], 
            "height": config['game']['screen_height']
        }
        
        # Połączenie z ADB (wymaga włączonego debugowania USB w emulatorze)
        try:
            self.client = AdbClient(host="127.0.0.1", port=5037)
            self.device = self.client.devices()[0]
            print(f"Połączono z urządzeniem: {self.device.serial}")
        except Exception as e:
            print(f"Błąd połączenia ADB: {e}. Upewnij się, że emulator działa i ADB jest włączone.")
            self.device = None

    def capture_screen(self):
        """Pobiera klatkę z ekranu i zwraca jako tablicę numpy (BGR)."""
        screenshot = np.array(self.sct.grab(self.monitor))
        # Usuń kanał alpha
        frame = screenshot[:, :, :3]
        return frame

    def tap(self, x, y):
        """Symuluje dotknięcie ekranu w punkcie (x, y) przez ADB."""
        if self.device:
            self.device.shell(f"input tap {x} {y}")
        else:
            print(f"Symulacja kliknięcia: {x}, {y} (Brak ADB)")

    def swipe(self, x1, y1, x2, y2, duration=300):
        """Symuluje przesunięcie karty."""
        if self.device:
            self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
        else:
            print(f"Symulacja swipe: {x1},{y1} -> {x2},{y2}")

    def play_card(self, card_index, x, y):
        """
        Zagrywa kartę z ręki na planszę.
        Zakładamy stałe pozycje kart w ręce (na dole ekranu).
        """
        # Przykładowe współrzędne kart w ręce (do dostosowania w configu)
        hand_y = 1600 
        hand_xs = [300, 500, 700, 900] # Przykładowe X dla 4 kart
        
        if 0 <= card_index < 4:
            card_x = hand_xs[card_index]
            print(f"Zagrywanie karty {card_index} na pozycję {x}, {y}")
            self.swipe(card_x, hand_y, x, y)
