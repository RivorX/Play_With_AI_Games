import sys
import os
# Dodaj ścieżkę do utils (jeśli uruchamiane bezpośrednio, ale teraz jest importowane)
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
import time
import cv2
import numpy as np
from utils.model import ClashRoyaleAgent
from utils.capture import EmulatorInterface
from utils.detector import StateDetector

def run_agent():
    # Ścieżka do configu musi być relatywna do miejsca uruchomienia skryptu (root projektu)
    # lub absolutna. Zakładamy uruchamianie z root.
    config_path = "clash_royale/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicjalizacja emulatora i detektora
    emu = EmulatorInterface(config)
    detector = StateDetector(config)
    
    # Ładowanie modelu
    model = ClashRoyaleAgent().to(device)
    try:
        model.load_state_dict(torch.load(config['paths']['model_save']))
        print("Załadowano wagi modelu.")
    except:
        print("Nie znaleziono wag modelu, używam losowych (nie zadziała dobrze!).")
    
    model.eval()
    
    print("Start bota... Naciśnij Ctrl+C aby przerwać.")
    
    try:
        while True:
            # 1. Obserwacja
            frame = emu.capture_screen()
            
            # Wykryj stan gry (eliksir, karty)
            current_elixir = detector.get_elixir_level(frame)
            hand_cards = detector.get_available_cards(frame)
            
            # Preprocessing dla modelu (270x480)
            img_resized = cv2.resize(frame, (270, 480))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            elixir_tensor = torch.tensor([[current_elixir]], dtype=torch.float).to(device)
            
            # 2. Decyzja
            with torch.no_grad():
                card_logits, pos_coords, _ = model(img_tensor, elixir_tensor)
                
            # Wybór akcji
            card_idx = torch.argmax(card_logits, dim=1).item()
            
            # Jeśli model wybrał kartę (indeks 0-3), a nie czekanie (indeks 4)
            if card_idx < 4:
                # Sprawdź czy nas stać na tę kartę
                card_name = hand_cards[card_idx]
                card_cost = config['cards'].get(card_name, 3) # Domyślny koszt 3 jak nie znajdzie
                
                if current_elixir >= card_cost:
                    x_norm, y_norm = pos_coords[0].cpu().numpy()
                    
                    # Denormalizacja współrzędnych do ekranu gry
                    game_x = int(x_norm * config['game']['screen_width'])
                    game_y = int(y_norm * config['game']['screen_height'])
                    
                    # 3. Wykonanie
                    print(f"Akcja: Karta {card_name} ({card_cost} eliksiru) na ({game_x}, {game_y}) [Eliksir: {current_elixir}]")
                    emu.play_card(card_idx, game_x, game_y)
                    
                    # Czekaj chwilę na odnowienie eliksiru (prosta heurystyka)
                    time.sleep(2) 
                else:
                    # print(f"Brak eliksiru na {card_name}. Mam {current_elixir}, potrzeba {card_cost}")
                    time.sleep(0.1)
            else:
                # print(f"Czekam... [Eliksir: {current_elixir}]")
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("Zatrzymano bota.")

if __name__ == "__main__":
    run_agent()
