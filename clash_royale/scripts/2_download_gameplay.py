import os
import sys
# Dodaj ścieżkę do utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import yt_dlp
import cv2
from utils.detector import StateDetector

def download_videos():
    config_path = "clash_royale/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_dir = config['paths']['raw_videos']
    os.makedirs(raw_dir, exist_ok=True)
    
    # Konfiguracja pobierania
    ydl_opts = {
        # Pobieramy jakość zbliżoną do 480p (wysokość <= 480), co pasuje do naszego modelu
        # i oszczędza miejsce.
        # Zmieniono format na 'best' aby uniknąć potrzeby ffmpeg do łączenia audio i wideo
        'format': 'best[height<=480][ext=mp4]/best[height<=480]',
        'outtmpl': os.path.join(raw_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'max_downloads': 5, # Pobierzemy na start 5 filmów
    }
    
    # Przykładowe zapytanie - szukamy gameplayu z konkretną kartą lub ogólnie
    # Można tu wstawić link do playlisty profesjonalnego gracza
    search_query = "ytsearch5:Clash Royale Top Ladder Gameplay" 
    
    print(f"Pobieranie wideo do: {raw_dir}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([search_query])
        
    print("Pobieranie zakończone. Rozpoczynam filtrowanie wygranych...")
    filter_wins(raw_dir, config)

def compress_video(input_path, config):
    print(f"Kompresja i skalowanie: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    target_width = config['game']['screen_width']
    target_height = config['game']['screen_height']
    target_fps = 5
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: original_fps = 30
    
    frame_interval = int(round(original_fps / target_fps))
    if frame_interval < 1: frame_interval = 1
    
    temp_path = input_path.replace(".mp4", "_temp.mp4")
    
    # Używamy mp4v dla kompatybilności
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, target_fps, (target_width, target_height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            resized = cv2.resize(frame, (target_width, target_height))
            out.write(resized)
            
        frame_count += 1
        
    cap.release()
    out.release()
    
    # Podmień plik
    if os.path.exists(input_path):
        os.remove(input_path)
    os.rename(temp_path, input_path)
    print(f"Zakończono kompresję: {input_path}")

def filter_wins(video_dir, config):
    """
    Przegląda pobrane wideo, usuwa przegrane, a wygrane kompresuje.
    """
    detector = StateDetector(config)
    
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue
            
        filepath = os.path.join(video_dir, filename)
        cap = cv2.VideoCapture(filepath)
        
        # Sprawdź ostatnie 10 sekund (zakładamy 30 fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        # Skocz do końcówki
        start_frame = max(0, frame_count - int(10 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        won = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize do analizy (szybciej)
            frame_small = cv2.resize(frame, (270, 480))
            
            if detector.check_victory(frame_small):
                won = True
                break
        
        cap.release()
        
        if won:
            print(f"[ZACHOWANO] Wygrana w: {filename}")
            compress_video(filepath, config)
        else:
            print(f"[USUNIĘTO] Przegrana/Brak detekcji w: {filename}")
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Nie udało się usunąć {filename}: {e}")

if __name__ == "__main__":
    download_videos()
