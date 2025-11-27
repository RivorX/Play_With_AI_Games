import cv2
import numpy as np
import os
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.detector import StateDetector

class GameSegmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Wczytaj ikonę chatu - teraz tylko środek (czarne kropki)
        chat_icon_path = self.config['paths'].get('chat_icon', 'clash_royale/assets/icons/chat.png')
        self.chat_template = cv2.imread(chat_icon_path, 0)  # Grayscale
        
        if self.chat_template is None:
            raise FileNotFoundError(f"Nie znaleziono ikony chatu: {chat_icon_path}")
        
        print(f"Wczytano ikonę chatu: {chat_icon_path} ({self.chat_template.shape})")
        print(f"  Porada: Upewnij się że ikona zawiera TYLKO czarny środek (kropki)")
        
        self.detector = StateDetector(self.config)
    
    def detect_chat_icon(self, frame, threshold=0.8):
        """
        Wykrywa ikonę chatu (czarny środek - kropki) na ekranie.
        Szuka trzech czarnych pikseli w dolnej połowie (bardziej niezawodne niż template matching).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Szukamy w dolnej połowie ekranu (gdzie jest chat)
        h, w = gray.shape[:2]
        search_region = gray[int(h*0.6):, :]  # Od 60% wysokości
        
        if search_region.shape[0] < 20:
            return False
        
        # Próg dla czarnych pikseli (bardzo ciemne - <50)
        BLACK_THRESHOLD = 60
        black_pixels = search_region < BLACK_THRESHOLD
        
        # Szukaj obszaru z dużą koncentracją czarnych pikseli
        # Konwertuj na image ze spójnymi komponentami
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        black_mask = (black_pixels * 255).astype(np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Znajdź kontury czarnych obszarów
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Szukaj konturów o rozsądnej wielkości (chat icon)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Chat icon powinien mieć pół do kilku procent obszaru
            if area > (search_region.shape[0] * search_region.shape[1] * 0.001):  # Min 0.1%
                if area < (search_region.shape[0] * search_region.shape[1] * 0.05):  # Max 5%
                    return True
        
        return False
    
    def find_game_segments(self, video_path):
        """
        Przeszukuje wideo i znajduje segmenty gier.
        Gra zaczyna się gdy pojawia się chat (omijamy menu na początku).
        Gra kończy się gdy:
        - Wykryjemy victory ORAZ chat zniknął (koniec gry)
        - Chat znika na >10 sekund (przerwanie/wyjście)
        
        Zwraca listę dict: {start, end, has_victory, duration_sec}
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analizuję: {os.path.basename(video_path)}")
        print(f"  Klatek: {frame_count}, FPS: {fps}")
        
        segments = []
        in_game = False
        game_start = None
        last_chat_seen = None
        frames_without_chat = 0
        victory_detected = False
        
        frame_idx = 0
        check_interval = int(fps)  # Sprawdzaj co sekundę
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sprawdzaj co sekundę
            if frame_idx % check_interval == 0:
                frame_small = cv2.resize(frame, (270, 480))
                
                has_chat = self.detect_chat_icon(frame_small)
                
                if has_chat:
                    last_chat_seen = frame_idx
                    frames_without_chat = 0
                    victory_detected = False  # Reset victory flag gdy chat widoczny
                    
                    if not in_game:
                        # START GRY - chat się pojawił (wyszliśmy z menu)
                        in_game = True
                        game_start = frame_idx
                        print(f"  [START GRY] Klatka {frame_idx} ({frame_idx/fps:.1f}s) - Chat widoczny")
                else:
                    # Chat nie jest widoczny
                    if in_game:
                        frames_without_chat += check_interval
                        
                        # Sprawdź victory TYLKO gdy nie ma chatu
                        if not victory_detected:
                            has_victory = self.detector.check_victory(frame_small)
                            if has_victory:
                                victory_detected = True
                                print(f"  [VICTORY] Klatka {frame_idx} ({frame_idx/fps:.1f}s)")
                
                # Sprawdź czy gra się skończyła
                if in_game:
                    # Koniec gry gdy:
                    # 1. Wykryliśmy victory I chat zniknął (minęło trochę czasu)
                    # 2. Chat zniknął na >10 sekund (wyjście/przerwanie)
                    if victory_detected and frames_without_chat > 2 * fps:
                        # Victory + 2 sekundy bez chatu = koniec gry
                        print(f"  [KONIEC GRY] Klatka {frame_idx} ({frame_idx/fps:.1f}s) - VICTORY")
                        
                        segments.append({
                            'start': game_start,
                            'end': frame_idx,
                            'has_victory': True,
                            'duration_sec': (frame_idx - game_start) / fps
                        })
                        
                        in_game = False
                        game_start = None
                        frames_without_chat = 0
                        victory_detected = False
                    
                    elif frames_without_chat > 10 * fps:
                        # Brak chatu >10s = przerwanie/wyjście
                        print(f"  [KONIEC GRY] Klatka {frame_idx} ({frame_idx/fps:.1f}s) - BRAK CHATU (>10s)")
                        
                        segments.append({
                            'start': game_start,
                            'end': frame_idx,
                            'has_victory': False,
                            'duration_sec': (frame_idx - game_start) / fps
                        })
                        
                        in_game = False
                        game_start = None
                        frames_without_chat = 0
                        victory_detected = False
            
            frame_idx += 1
            
            # Postęp co 5%
            if frame_idx % max(1, frame_count // 20) == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"  Postęp: {progress:.0f}%")
        
        cap.release()
        
        # Jeśli gra trwała do końca wideo
        if in_game and game_start is not None:
            print(f"  [KONIEC GRY] Koniec wideo (klatka {frame_idx})")
            segments.append({
                'start': game_start,
                'end': frame_idx,
                'has_victory': victory_detected,
                'duration_sec': (frame_idx - game_start) / fps
            })
        
        return segments, fps
    
    def extract_game_segment(self, video_path, segment, segment_idx, fps, output_dir):
        """
        Wycina segment gry i zapisuje jako osobny plik w output_dir.
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Nazwa wyjściowa
        victory_tag = "WIN" if segment['has_victory'] else "NOWIN"
        output_filename = f"{base_name}_game{segment_idx + 1}_{victory_tag}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"  Wycinam grę {segment_idx + 1}: {segment['duration_sec']:.1f}s ({victory_tag})")
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start'])
        
        # Przygotuj VideoWriter
        target_width = self.config['game']['screen_width']
        target_height = self.config['game']['screen_height']
        target_fps = 5
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))
        
        frame_interval = int(round(fps / target_fps))
        if frame_interval < 1:
            frame_interval = 1
        
        current_frame = segment['start']
        frames_written = 0
        
        while current_frame <= segment['end']:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (current_frame - segment['start']) % frame_interval == 0:
                resized = cv2.resize(frame, (target_width, target_height))
                out.write(resized)
                frames_written += 1
            
            current_frame += 1
        
        cap.release()
        out.release()
        
        print(f"    Zapisano: {output_filename} ({frames_written} klatek)")
        
        return output_path, segment['has_victory']

def process_video(video_path, config_path, output_dir):
    """
    Przetwarza jedno wideo: znajduje gry i wycina je do output_dir.
    """
    segmenter = GameSegmenter(config_path)
    
    # 1. Znajdź segmenty gier
    segments, fps = segmenter.find_game_segments(video_path)
    
    if not segments:
        print(f"[BRAK GIER] Nie znaleziono żadnych gier w: {os.path.basename(video_path)}")
        return []
    
    print(f"[ZNALEZIONO] {len(segments)} gier w: {os.path.basename(video_path)}")
    
    # 2. Wytnij każdy segment
    extracted_files = []
    for idx, segment in enumerate(segments):
        # Filtruj zbyt krótkie gry (<30 sekund)
        if segment['duration_sec'] < 30:
            print(f"  Pomijam grę {idx + 1} - zbyt krótka ({segment['duration_sec']:.1f}s)")
            continue
        
        output_path, has_victory = segmenter.extract_game_segment(
            video_path, segment, idx, fps, output_dir
        )
        
        extracted_files.append({
            'path': output_path,
            'has_victory': has_victory,
            'duration': segment['duration_sec']
        })
    
    return extracted_files

if __name__ == "__main__":
    config_path = "clash_royale/config/config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    raw_dir = config['paths']['raw_videos']
    
    # NOWY KATALOG DLA SEGMENTÓW
    segmented_dir = config['paths'].get('segmented_games', 'clash_royale/data/segmented_games')
    os.makedirs(segmented_dir, exist_ok=True)
    
    if not os.path.exists(raw_dir):
        print(f"Katalog {raw_dir} nie istnieje!")
        exit()
    
    # Znajdź wszystkie pliki wideo (bez _game w nazwie - to są oryginały)
    files = [
        f for f in os.listdir(raw_dir) 
        if f.endswith(".mp4") and "_game" not in f
    ]
    
    if not files:
        print("Brak plików wideo do przetworzenia!")
        exit()
    
    print(f"Znaleziono {len(files)} plików wideo do przetworzenia.\n")
    
    all_extracted = []
    total_games = 0
    total_victories = 0
    
    for filename in files:
        video_path = os.path.join(raw_dir, filename)
        
        print(f"\n{'='*60}")
        print(f"Przetwarzam: {filename}")
        print('='*60)
        
        try:
            extracted = process_video(video_path, config_path, segmented_dir)
            
            if extracted:
                all_extracted.extend(extracted)
                total_games += len(extracted)
                total_victories += sum(1 for e in extracted if e['has_victory'])
        
        except Exception as e:
            print(f"\n[BŁĄD] Nie udało się przetworzyć {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("PODSUMOWANIE")
    print('='*60)
    print(f"Przetworzono plików: {len(files)}")
    print(f"Wyodrębniono gier: {total_games}")
    print(f"Z czego wygranych: {total_victories}")
    print(f"Procent wygranych: {(total_victories/total_games*100) if total_games > 0 else 0:.1f}%")
    print(f"\nSegmenty zapisane w: {segmented_dir}")
    
    # Opcjonalnie: usuń pliki bez victory
    if total_games > 0:
        remove_losses = input("\nCzy usunąć pliki bez victory? (y/n): ").lower().strip()
        
        if remove_losses == 'y':
            removed = 0
            for extracted in all_extracted:
                if not extracted['has_victory']:
                    try:
                        os.remove(extracted['path'])
                        removed += 1
                        print(f"Usunięto: {os.path.basename(extracted['path'])}")
                    except Exception as e:
                        print(f"Nie udało się usunąć {extracted['path']}: {e}")
            
            print(f"\nUsunięto {removed} gier bez victory.")
            print(f"Pozostało {total_victories} gier z victory w: {segmented_dir}")