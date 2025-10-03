import os
import yaml
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import cv2
from stable_baselines3 import PPO
import torch
from pathlib import Path
import easyocr

# Wczytaj konfigurację
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

agario_dir = os.path.abspath(os.path.join(script_dir, '..'))
models_dir = os.path.join(agario_dir, config['paths']['models_dir'])

def find_available_models():
    """Znajdź wszystkie dostępne modele w folderze models/"""
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.zip'):
            model_path = os.path.join(models_dir, file)
            # Pobierz info o pliku
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            modified = os.path.getmtime(model_path)
            modified_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(modified))
            
            models.append({
                'name': file,
                'path': model_path,
                'size_mb': size_mb,
                'modified': modified_str
            })
    
    # Sortuj po dacie modyfikacji (najnowsze pierwsze)
    models.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)
    return models

def select_model():
    """Interaktywny wybór modelu"""
    print("\n" + "="*60)
    print("AGAR.IO MODEL TESTER")
    print("="*60)
    
    models = find_available_models()
    
    if not models:
        print(f"\n❌ Brak modeli w folderze: {models_dir}")
        print("   Najpierw wytrenuj model używając: python agario/scripts/train.py")
        return None
    
    print(f"\n📁 Znaleziono {len(models)} model(i):\n")
    
    for i, model in enumerate(models, 1):
        print(f"  [{i}] {model['name']}")
        print(f"      Rozmiar: {model['size_mb']:.2f} MB")
        print(f"      Zmodyfikowano: {model['modified']}")
        print()
    
    while True:
        try:
            choice = input(f"Wybierz model (1-{len(models)}) lub 'q' aby wyjść: ").strip()
            
            if choice.lower() == 'q':
                print("Anulowano.")
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected = models[choice_idx]
                print(f"\n✅ Wybrano: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ Nieprawidłowy wybór. Podaj liczbę 1-{len(models)}")
        except ValueError:
            print("❌ Nieprawidłowy format. Podaj liczbę lub 'q'")
        except KeyboardInterrupt:
            print("\n\nAnulowano.")
            return None

# Custom env podobny do train, ale z real browser
class RealAgarIoEnv:
    def __init__(self, driver, reader, game_area=None):
        self.driver = driver
        self.reader = reader
        self.frame_history = config['environment']['frame_history']
        self.screen_size = config['environment']['screen_size']  # [W, H] = [256, 192]
        # Historia: (T, C, H, W) = (4, 3, 192, 256)
        screen_w, screen_h = self.screen_size
        self.history = np.zeros((self.frame_history, 3, screen_h, screen_w))
        
        # Dynamiczny obszar gry (wykryty przez OCR)
        self.game_area = game_area  # (x1, y1, x2, y2) lub None
        self.debug_frame_saved = False  # Flag dla zapisu debug frame
    
    def get_obs(self, save_debug=False):
        """Pobierz screenshot i przetwórz do formatu modelu"""
        screenshot = self.driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        # Crop obszaru gry (musi być wykryty!)
        if not self.game_area:
            raise ValueError("Obszar gry nie został wykryty! Nie można kontynuować.")
        
        x1, y1, x2, y2 = self.game_area
        crop = img[y1:y2, x1:x2]
        
        # Zapisz debug frame (tylko raz)
        if save_debug and not self.debug_frame_saved:
            debug_dir = os.path.join(agario_dir, 'datasets')
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, 'test_crop_debug.png'), crop)
            print(f"   💾 Zapisano debug crop: datasets/test_crop_debug.png")
            self.debug_frame_saved = True
        
        # Resize do rozmiaru treningowego
        # screen_size = [W, H] = [256, 192]
        # cv2.resize przyjmuje (width, height)
        screen_w, screen_h = self.screen_size
        crop_resized = cv2.resize(crop, (screen_w, screen_h))  # (W=256, H=192)
        
        # BGR -> RGB i normalizacja
        crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # (H, W, 3) -> (3, H, W)
        crop = np.transpose(crop, (2, 0, 1))
        
        # Aktualizuj historię
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = crop
        
        # Zwróć bez batch dimension - model.predict sam go doda
        # Format: (T=4, C=3, H=192, W=256)
        return self.history
    
    def act(self, action):
        """Wykonaj akcję w grze"""
        # action shape: (1, 3) lub (3,)
        if action.ndim == 2:
            action = action[0]  # Usuń batch dim
        
        dx = float(action[0])
        dy = float(action[1])
        split_eject = float(action[2])
        
        # Ruch myszy (symulacja - wymaga pyautogui)
        try:
            import pyautogui
            # Skala ruchu (dostosuj do szybkości gry)
            scale = 50
            pyautogui.moveRel(int(dx * scale), int(dy * scale), duration=0.05)
            
            # Split / Eject
            if split_eject > 0.7:
                pyautogui.press('w')  # Split
            elif split_eject > 0.3:
                pyautogui.press('space')  # Eject
        except ImportError:
            print("⚠️  pyautogui nie zainstalowane - ruch myszy wyłączony")
            print("   Zainstaluj: pip install pyautogui")

def setup_chrome_with_adblock():
    """Skonfiguruj Chrome z blokowaniem reklam"""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Blokowanie reklam - wbudowane filtry Chrome
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,  # Blokuj powiadomienia
        "profile.managed_default_content_settings.images": 1,  # Pozwól na obrazy
    })
    
    # Dodatkowo: Blokuj niektóre domeny reklamowe
    chrome_options.add_argument("--disable-popup-blocking")
    
    # User agent (nie wygląda jak bot)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    return chrome_options

def detect_game_area(driver, reader, max_attempts=3):
    """
    Wykryj obszar planszy gry używając OCR (Score i Leaderboard).
    Zwraca: (x1, y1, x2, y2) lub None jeśli nie wykryto
    """
    for attempt in range(max_attempts):
        print(f"   Próba {attempt+1}/{max_attempts}...", end='\r')
        
        screenshot = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # OCR z niższym threshold
        result = reader.readtext(img_rgb, detail=1, paragraph=False)
        
        score_bbox = None
        leaderboard_bbox = None
        
        # Debug: pokaż wszystkie wykryte teksty
        if attempt == 0:
            print(f"\n   [OCR] Wykryto {len(result)} tekstów:")
            for bbox, text, conf in result[:10]:  # Pokaż pierwsze 10
                print(f"      '{text}' (conf: {conf:.2f})")
        
        for bbox, text, conf in result:
            txt = text.lower()
            # Bardziej elastyczne dopasowanie
            if ('score' in txt or 'scor' in txt) and conf > 0.2:
                score_bbox = bbox
                if attempt == 0:
                    print(f"   ✅ Znaleziono 'Score': '{text}' (conf: {conf:.2f})")
            if ('leaderboard' in txt or 'leader' in txt) and conf > 0.2:
                leaderboard_bbox = bbox
                if attempt == 0:
                    print(f"   ✅ Znaleziono 'Leaderboard': '{text}' (conf: {conf:.2f})")
        
        if score_bbox and leaderboard_bbox:
            # Wyznacz obszar gry między Score (góra) a Leaderboard (prawo)
            x1 = int(min(p[0] for p in score_bbox))
            y2 = int(max(p[1] for p in score_bbox))
            x2 = int(max(p[0] for p in leaderboard_bbox))
            y1 = int(min(p[1] for p in leaderboard_bbox))
            
            # Walidacja
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img.shape[1])
            y2 = min(y2, img.shape[0])
            
            # Sprawdź czy rozmiar jest sensowny
            width = x2 - x1
            height = y2 - y1
            if width > 100 and height > 100:  # Min 100x100 px
                print(f"\n   ✅ Obszar: {width}x{height} px")
                
                # Zapisz debug screenshot z zaznaczonym obszarem
                debug_img = img.copy()
                # Zaznacz Score bbox (czerwony)
                score_pts = np.array(score_bbox, np.int32)
                cv2.polylines(debug_img, [score_pts], isClosed=True, color=(0,0,255), thickness=3)
                # Zaznacz Leaderboard bbox (zielony)
                leader_pts = np.array(leaderboard_bbox, np.int32)
                cv2.polylines(debug_img, [leader_pts], isClosed=True, color=(0,255,0), thickness=3)
                # Zaznacz wykryty obszar gry (niebieski prostokąt)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255,0,0), 3)
                # Zapisz
                debug_dir = os.path.join(agario_dir, 'datasets')
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, 'game_area_detection_debug.png'), debug_img)
                print(f"   💾 Zapisano debug: datasets/game_area_detection_debug.png")
                
                return (x1, y1, x2, y2)
        
        time.sleep(0.5)  # Czekaj przed kolejną próbą
    
    print("\n   ⚠️  Nie udało się wykryć obu elementów")
    return None

def is_in_game(driver, reader):
    """
    Sprawdź czy jesteśmy w grze (vs menu).
    Zwraca: True jeśli wykryto 'Score', False jeśli w menu
    """
    try:
        # Sprawdź canvas
        canvas = driver.find_elements(By.ID, "canvas")
        if not canvas or not canvas[0].is_displayed():
            return False
        
        # Sprawdź Score przez OCR
        screenshot = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = reader.readtext(img_rgb, detail=0, paragraph=False)
        
        for text in result:
            if 'score' in text.lower():
                return True
        
        return False
    except:
        return False

def wait_for_game_start(driver, reader, max_wait=60):
    """
    Czeka aż użytkownik przejdzie przez menu i rozpocznie grę.
    Używa OCR do wykrycia 'Score' w grze.
    Zwraca: (success, game_area) - game_area to (x1,y1,x2,y2) lub None
    """
    print(f"\n⏳ Czekam na start gry (max {max_wait}s)...")
    print("   👉 Kliknij 'Play' w przeglądarce gdy będziesz gotowy!")
    print("   💡 TIP: Zamknij reklamy jeśli się pojawią")
    
    start_time = time.time()
    last_update = 0
    
    while time.time() - start_time < max_wait:
        try:
            # Sprawdź czy jesteśmy w grze (wykryj Score)
            if is_in_game(driver, reader):
                print("\n✅ Gra wykryta (Score znaleziony) - poczekaj na stabilizację...")
                time.sleep(3)  # Poczekaj aż UI się ustabilizuje
                
                # Teraz wykryj obszar gry (Score powinien być już widoczny)
                print("🎯 Wykrywanie obszaru gry...")
                game_area = detect_game_area(driver, reader, max_attempts=5)
                
                if game_area:
                    x1, y1, x2, y2 = game_area
                    print(f"✅ Wykryto obszar gry: ({x1}, {y1}) -> ({x2}, {y2})")
                    return True, game_area
                else:
                    print("❌ Nie udało się wykryć Score i Leaderboard!")
                    print("   Spróbuj ponownie - upewnij się że gra jest w pełni załadowana.")
                    return False, None
            
            # Progress bar co sekundę
            elapsed = int(time.time() - start_time)
            if elapsed > last_update:
                remaining = max_wait - elapsed
                print(f"   Oczekiwanie... ({elapsed}s / {max_wait}s, pozostało: {remaining}s)", end='\r')
                last_update = elapsed
            
            time.sleep(0.5)
            
        except Exception as e:
            pass  # Ignoruj błędy, po prostu czekaj
    
    print(f"\n⚠️  Upłynął limit czasu ({max_wait}s)")
    resp = input("Kontynuować mimo to? [Y/n]: ").strip()
    return (resp.lower() not in ('n', 'no'), None)

def test_model(model_path):
    """Testuj model na prawdziwym Agar.io"""
    print("\n" + "="*60)
    print("ROZPOCZYNANIE TESTU")
    print("="*60)
    
    # Wczytaj model
    print(f"\n📦 Ładowanie modelu: {os.path.basename(model_path)}")
    try:
        model = PPO.load(model_path)
        model.policy.eval()
        print("✅ Model załadowany")
    except Exception as e:
        print(f"❌ Błąd ładowania modelu: {e}")
        return
    
    # Inicjalizuj OCR
    print("\n🔍 Inicjalizacja OCR (EasyOCR)...")
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        print("✅ OCR gotowy (GPU)")
    except:
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ OCR gotowy (CPU)")
    
    # Uruchom przeglądarkę z AdBlock
    print("\n🌐 Uruchamianie Chrome z blokowaniem reklam...")
    try:
        chrome_options = setup_chrome_with_adblock()
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://agar.io")
        print("✅ Przeglądarka uruchomiona")
        
        # Czekaj na start gry (wykrywa również obszar gry)
        game_started, game_area = wait_for_game_start(driver, reader, max_wait=60)
        
        if not game_started:
            print("❌ Anulowano przez użytkownika")
            driver.quit()
            return
            
    except Exception as e:
        print(f"❌ Błąd uruchamiania przeglądarki: {e}")
        print("   Upewnij się że masz zainstalowany ChromeDriver")
        return
    
    # Testowanie
    env = RealAgarIoEnv(driver, reader, game_area)
    
    print("\n🎮 ROZPOCZYNAM TEST (Ctrl+C aby zatrzymać)")
    print("="*60)
    
    start_time = time.time()
    step_count = 0
    last_score_check = time.time()
    
    try:
        while True:
            # Co 2 sekundy sprawdź czy Score jest widoczny
            if time.time() - last_score_check >= 2.0:
                if not is_in_game(driver, reader):
                    print("\n\n⚠️  Gra zakończona (brak Score przez 2s) - zatrzymuję test")
                    break
                last_score_check = time.time()
            
            # Pobierz obserwację (zapisz debug w pierwszej iteracji)
            obs = env.get_obs(save_debug=(step_count == 0))
            
            # Przewidź akcję
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
            
            # Debug pierwszej iteracji
            if step_count == 0:
                print(f"\n[DEBUG] action type: {type(action)}")
                print(f"[DEBUG] action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
                print(f"[DEBUG] action value: {action}\n")
            
            # Wykonaj akcję
            env.act(action)
            
            step_count += 1
            elapsed = time.time() - start_time
            
            # Status co 10 kroków
            if step_count % 10 == 0:
                fps = step_count / elapsed
                # Bezpieczne wyświetlanie akcji
                if hasattr(action, 'shape') and len(action.shape) > 0:
                    if action.shape[0] >= 3:
                        action_str = f"[{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:.2f}]"
                    else:
                        action_str = str(action)
                else:
                    action_str = str(action)
                print(f"Krok: {step_count:4d} | Czas: {elapsed:6.1f}s | FPS: {fps:5.1f} | Akcja: {action_str}", end='\r')
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test zatrzymany przez użytkownika")
    except Exception as e:
        print(f"\n\n❌ Błąd podczas testu: {e}")
    finally:
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("PODSUMOWANIE")
        print("="*60)
        print(f"Czas testu: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Liczba kroków: {step_count}")
        print(f"Średnie FPS: {step_count/elapsed:.1f}")
        print("\n🔚 Zamykam przeglądarkę...")
        driver.quit()
        print("✅ Test zakończony")

def main():
    """Główna funkcja"""
    model_path = select_model()
    
    if model_path:
        test_model(model_path)
    
    print("\nDziękuję za użycie testera! 👋")

if __name__ == "__main__":
    main()