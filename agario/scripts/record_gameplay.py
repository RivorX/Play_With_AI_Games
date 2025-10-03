import os
import time
import json
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pyautogui  # Dla capture ekranu, ale tu używamy Selenium screenshots
from datetime import datetime
import yaml
import easyocr


# Wczytaj konfigurację względnie do lokalizacji skryptu
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Katalog nagrań względny do lokalizacji skryptu

agario_dir = os.path.abspath(os.path.join(script_dir, '..'))
RECORDINGS_DIR = os.path.join(agario_dir, config['dataset']['recordings_dir'])
FPS = config['dataset']['fps']
SESSION_LENGTH = config['dataset']['session_length']

# Zwiększona rozdzielczość do nauki (np. 256x256)


def setup_browser():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--app=https://agar.io")
    driver = webdriver.Chrome(options=chrome_options)
    time.sleep(5)  # Czekaj na load
    try:
        driver.find_element(By.ID, "qa-allow").click()
    except:
        pass
    return driver

def record_session(driver, session_id):
    session_dir = os.path.join(RECORDINGS_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    frames = []
    actions = []
    
    start_time = time.time()
    frame_interval = 1.0 / FPS
    last_frame_time = start_time
    
    # Inicjalizacja OCR
    reader = easyocr.Reader(['en'], gpu=True)

    # Początkowe wykrywanie bboxów dla Score i Leaderboard
    print("Skanowanie ekranu w poszukiwaniu 'Score' i 'Leaderboard'...")
    score_bbox = None
    leaderboard_bbox = None
    while score_bbox is None or leaderboard_bbox is None:
        screenshot = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = reader.readtext(img_rgb, detail=1, paragraph=False)
        for bbox, text, conf in result:
            txt = text.lower()
            if 'score' in txt:
                score_bbox = bbox
            if 'leaderboard' in txt:
                leaderboard_bbox = bbox
        if score_bbox is None or leaderboard_bbox is None:
            time.sleep(0.5)
    print("Wykryto 'Score' i 'Leaderboard'. Rozpoczynam nagrywanie!")

    # Wyznacz początkowy crop gry
    x1 = min(p[0] for p in score_bbox)
    y2 = max(p[1] for p in score_bbox)
    x2 = max(p[0] for p in leaderboard_bbox)
    y1 = min(p[1] for p in leaderboard_bbox)
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    # Ograniczony obszar wyszukiwania dla score
    score_search_margin_x = 200  # ±200 px w poziomie
    score_search_margin_y = 50   # ±50 px w pionie
    last_score = None  # Fallback dla brakujących odczytów
    none_score_count = 0  # Licznik klatek z None

    while time.time() - start_time < SESSION_LENGTH:
        current_time = time.time()

        # Zrób zrzut ekranu
        screenshot = driver.get_screenshot_as_png()
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)

        # Crop gry (stały rozmiar)
        game_crop = img[y1:y2, x1:x2].copy()

        # Dynamiczne ROI dla score
        score_x1 = max(min(p[0] for p in score_bbox) - x1 - score_search_margin_x, 0)
        score_y1 = max(min(p[1] for p in score_bbox) - y1 - score_search_margin_y, 0)
        score_x2 = min(max(p[0] for p in score_bbox) - x1 + score_search_margin_x, game_crop.shape[1])
        score_y2 = min(max(p[1] for p in score_bbox) - y1 + score_search_margin_y, game_crop.shape[0])
        score_roi = game_crop[score_y1:score_y2, score_x1:score_x2]

        # Preprocessing ROI: binaryzacja
        score_roi_gray = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)
        _, score_roi_bin = cv2.threshold(score_roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        score_roi_rgb = cv2.cvtColor(score_roi_bin, cv2.COLOR_GRAY2RGB)

        # OCR na ROI
        score_val = None
        if score_roi.size > 0:
            score_result = reader.readtext(score_roi_rgb, detail=0, paragraph=False)
            import re
            candidates = []
            for r in score_result:
                digits = re.findall(r'\d+', r)
                if digits:
                    candidates.append(max(digits, key=len))
            
            if candidates:
                try:
                    score_val = int(max(candidates, key=len))
                    last_score = score_val
                    none_score_count = 0  # Reset licznika
                except ValueError:
                    score_val = last_score
                    none_score_count += 1
            else:
                score_val = last_score
                none_score_count += 1

        # Zakończ, jeśli 4 klatki z rzędu ma None
        if none_score_count >= 4:
            print("Wykryto 4 kolejne klatki bez score – gra prawdopodobnie zakończona. Kończę nagrywanie!")
            break

        # Zapisz wynik OCR
        with open(os.path.join(session_dir, 'ocr_results.txt'), 'a') as ocr_file:
            ocr_file.write(f"frame_{len(frames):04d}.png - {score_val}\n")

        # Zapisz frame
            # Skalowanie obrazu jeśli najdłuższa krawędź > 1024
            h, w = game_crop.shape[:2]
            max_edge = max(h, w)
            if max_edge > 1024:
                scale = 1024 / max_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                game_crop_scaled = cv2.resize(game_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                game_crop_scaled = game_crop
            frames.append(game_crop_scaled)
            cv2.imwrite(os.path.join(session_dir, f"frame_{len(frames):04d}.png"), game_crop_scaled)

        # Loguj akcje
        mouse_pos = pyautogui.position()
        if len(actions) > 0:
            prev_pos = actions[-1].get('mouse_pos', mouse_pos)
            dx = mouse_pos[0] - prev_pos[0]
            dy = mouse_pos[1] - prev_pos[1]
        else:
            dx, dy = 0, 0
        keys = {'split': False, 'eject': False}  # Dodaj detekcję klawiszy (np. pyautogui.is_pressed)
        actions.append({
            'time': current_time - start_time,
            'mouse_delta': (dx, dy),
            'keys': keys,
            'mouse_pos': mouse_pos,
            'score': score_val
        })

        time.sleep(max(0, frame_interval - (current_time - last_frame_time)))
        last_frame_time = time.time()
    
    # Zapisz JSON akcji
    with open(os.path.join(session_dir, 'actions.json'), 'w') as f:
        json.dump(actions, f)
    
    print(f"Nagrano sesję {session_id}: {len(frames)} klatek")

    # Analiza wyników OCR
    ocr_path = os.path.join(session_dir, 'ocr_results.txt')
    max_score = None
    score_counts = {}
    if os.path.exists(ocr_path):
        with open(ocr_path, 'r') as f:
            for line in f:
                parts = line.strip().split('-')
                if len(parts) == 2:
                    val = parts[1].strip()
                    try:
                        score = int(val)
                        score_counts[score] = score_counts.get(score, 0) + 1
                    except:
                        continue
        valid_scores = [s for s, cnt in score_counts.items() if cnt >= 5]
        if valid_scores:
            max_score = max(valid_scores)
        with open(os.path.join(session_dir, 'max_score.txt'), 'w') as f:
            f.write(str(max_score) if max_score is not None else "Brak wyniku >= 5x")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test-ocr":
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        # Jeśli podano drugi argument, użyj go jako nazwy pliku
        if len(sys.argv) > 2:
            test_img_path = sys.argv[2]
            if not os.path.isabs(test_img_path):
                test_img_path = os.path.join(agario_dir, "datasets", test_img_path)
        else:
            test_img_path = os.path.join(agario_dir, "datasets", "test.png")
        img = cv2.imread(test_img_path)
        if img is None:
            print(f"Nie można wczytać pliku: {test_img_path}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = reader.readtext(img_rgb, detail=1, paragraph=False)
        print(f"Wyniki OCR na {test_img_path}:")
        score_bbox = None
        leaderboard_bbox = None
        for bbox, text, conf in result:
            print(f"Tekst: {text}, Pewność: {conf}, BBOX: {bbox}")
            txt = text.lower()
            if 'score' in txt:
                score_bbox = bbox
            if 'leaderboard' in txt:
                leaderboard_bbox = bbox
        # Podświetl wszystkie wykryte napisy na obrazku
        for bbox, text, conf in result:
            pts = np.array(bbox, np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
        debug_path = os.path.join(agario_dir, "datasets", "test_ocr_debug.png")
        cv2.imwrite(debug_path, img)
        print(f"Zapisano {debug_path} z podświetleniem wykrytych napisów.")

        # Wycinanie planszy gry na podstawie bboxów
        if score_bbox and leaderboard_bbox:
            x1 = min([p[0] for p in score_bbox])
            y2 = max([p[1] for p in score_bbox])
            x2 = max([p[0] for p in leaderboard_bbox])
            y1 = min([p[1] for p in leaderboard_bbox])
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img.shape[1])
            y2 = min(y2, img.shape[0])
            game_crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(agario_dir, "datasets", "test_game_crop.png")
            cv2.imwrite(crop_path, game_crop)
            print(f"Zapisano wyciętą planszę gry: {crop_path}")
            # OCR tylko na oryginalnym bboxie score
            score_x1 = min([p[0] for p in score_bbox]) - x1
            score_y1 = min([p[1] for p in score_bbox]) - y1
            score_x2 = max([p[0] for p in score_bbox]) - x1
            score_y2 = max([p[1] for p in score_bbox]) - y1
            score_roi = game_crop[score_y1:score_y2, score_x1:score_x2]
            score_val = None
            if score_roi.size > 0:
                score_roi_rgb = cv2.cvtColor(score_roi, cv2.COLOR_BGR2RGB)
                score_result = reader.readtext(score_roi_rgb, detail=0, paragraph=False)
                for r in score_result:
                    try:
                        score_val = int(''.join(filter(str.isdigit, r)))
                        break
                    except Exception:
                        continue
                print(f"Wynik OCR na score: {score_val}")
            # Zapisz wynik OCR do pliku tekstowego
            with open(os.path.join(agario_dir, "datasets", "test_ocr_results.txt"), 'a') as ocr_file:
                ocr_file.write(f"{os.path.basename(test_img_path)} - {score_val}\n")
        else:
            print("Nie wykryto obu napisów: 'Score' i 'Leaderboard'.")
        sys.exit(0)

    # ...standardowe nagrywanie...
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    driver = setup_browser()
    try:
        record_session(driver, session_id)
    finally:
        driver.quit()