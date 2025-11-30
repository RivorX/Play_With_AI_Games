import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import cv2
import numpy as np
from utils.game_detector import GameDetector

# Ścieżki do testowych obrazków
DEBUG_IMG_DIR = 'clash_royale/assets/debug'
ICONS_DIR = 'clash_royale/assets/icons'

# Lista przykładowych plików do testów (możesz rozszerzyć)
test_images = [
    os.path.join(DEBUG_IMG_DIR, f) for f in os.listdir(DEBUG_IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

print(f"Znaleziono {len(test_images)} obrazków do testów.")

detector = GameDetector(ICONS_DIR)

def test_image(img_path):
    print(f"\n--- Test: {os.path.basename(img_path)} ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Nie można wczytać obrazu!")
        return

    # Test wykrywania regionu gry
    region = detector.detect_game_region(img)
    print(f"Region gry: {region}")
    x, y, w, h = region
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    # Test wykrywania chatu
    has_chat = detector.detect_chat_icon(img)
    print(f"Chat widoczny: {has_chat}")

    # Test wykrywania winnera
    winner = detector.detect_winner(img)
    print(f"Winner: {winner}")

    logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    logs_dir = os.path.abspath(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"debug_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, img)
    print(f"Podgląd zapisany: {out_path}")

if __name__ == "__main__":
    for img_path in test_images:
        test_image(img_path)
    print("\nTestowanie zakończone.")
