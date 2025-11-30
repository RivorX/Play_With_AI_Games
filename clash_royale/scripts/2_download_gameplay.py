import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import yt_dlp

def download_videos():
    config_path = "clash_royale/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_dir = config['paths']['raw_videos']
    os.makedirs(raw_dir, exist_ok=True)
    
    # Plik archiwum do śledzenia pobranych filmów (zapobiega duplikatom)
    archive_file = os.path.join(raw_dir, 'downloaded_archive.txt')
    
    # Licznik pobranych filmów
    download_count = {'count': 0, 'max': 5}
    
    def progress_hook(d):
        if d['status'] == 'finished':
            download_count['count'] += 1
            filename = os.path.basename(d.get('filename', 'Unknown'))
            print(f"      ✓ Pobrano: {filename}\n")
        elif d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            print(f"\r      Pobieranie: {percent} @ {speed}", end='', flush=True)
        elif d['status'] == 'error':
            print(f"      ✗ Błąd pobierania\n")
    
    # Konfiguracja pobierania
    ydl_opts = {
        # Format: preferuj MP4 do 720p, z fallbackiem
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(raw_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'match_filter': filter_tournament_videos,
        'restrictfilenames': True,
        'windowsfilenames': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'noprogress': True,
        'progress_hooks': [progress_hook],
        # Archiwum zapobiega ponownemu pobieraniu
        'download_archive': archive_file,
        # Timeout i retry
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,
        # Ekstraktory YouTube
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
            }
        },
        # Postprocessor do zapewnienia MP4
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }
    
    # Szukamy więcej filmów niż potrzebujemy (część może być odfiltrowana)
    num_to_search = 15
    search_query = f"ytsearch{num_to_search}:Clash Royale Top Ladder Gameplay 2024"
    
    print(f"Pobieranie wideo do: {raw_dir}")
    print(f"Archiwum pobranych: {archive_file}")
    print("="*60)
    
    # Wczytaj już pobrane filmy
    already_downloaded = set()
    if os.path.exists(archive_file):
        with open(archive_file, 'r') as f:
            already_downloaded = set(line.strip() for line in f if line.strip())
        if already_downloaded:
            print(f"Już pobrano wcześniej: {len(already_downloaded)} filmów")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Najpierw pobierz listę bez pobierania
            print(f"Szukam filmów...")
            info = ydl.extract_info(search_query, download=False)
            
            if not info or 'entries' not in info:
                print("Nie znaleziono żadnych filmów.")
                return
            
            # Filtruj filmy
            videos = []
            for entry in info['entries']:
                if not entry:
                    continue
                    
                video_id = f"youtube {entry.get('id', '')}"
                
                # Sprawdź czy już pobrano
                if video_id in already_downloaded:
                    continue
                
                # Sprawdź filtr turniejowy
                filter_result = filter_tournament_videos(entry, False)
                if filter_result:
                    continue
                    
                videos.append(entry)
                
                if len(videos) >= download_count['max']:
                    break
            
            if not videos:
                print("Wszystkie znalezione filmy już zostały pobrane lub odfiltrowane.")
                print("Spróbuj usunąć plik archiwum, aby pobrać ponownie.")
                return
                
            print(f"Znaleziono {len(videos)} nowych filmów do pobrania\n")
            
            # Pobierz każdy film
            for idx, video in enumerate(videos, 1):
                title = video.get('title', 'Unknown')
                duration = video.get('duration', 0)
                duration_str = f"{duration//60}:{duration%60:02d}" if duration else "N/A"
                
                if len(title) > 45:
                    title = title[:42] + "..."
                    
                print(f"[{idx}/{len(videos)}] {title} ({duration_str})")
                
                try:
                    ydl.download([video['webpage_url']])
                except yt_dlp.utils.DownloadError as e:
                    print(f"      ✗ Błąd: {str(e)[:50]}\n")
                except Exception as e:
                    print(f"      ✗ Nieoczekiwany błąd: {e}\n")
                        
    except Exception as e:
        print(f"\n✗ Błąd podczas pobierania: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Podsumowanie - zlicz wszystkie pliki wideo
    video_extensions = ('.mp4', '.webm', '.mkv', '.avi')
    downloaded = [f for f in os.listdir(raw_dir) 
                  if f.endswith(video_extensions) and not f.startswith('.')]
    
    print("="*60)
    print(f"PODSUMOWANIE:")
    print(f"  Pobrano w tej sesji: {download_count['count']} plików")
    print(f"  Łącznie w folderze: {len(downloaded)} filmów")
    print(f"  Lokalizacja: {raw_dir}")
    print("="*60)
    print("\nKolejny krok: Uruchom '3_segment_games.py' aby pociąć na gry")

def filter_tournament_videos(info_dict, incomplete):
    if incomplete:
        return None
    
    title = info_dict.get('title', '').lower()
    tournament_keywords = [
        'turniej', 'turnament', 'tournament', 
        'championship', 'mistrzostwa', 'competit',
        'finals', 'finał', 'semi-final', 'quarter-final'
    ]
    
    for keyword in tournament_keywords:
        if keyword in title:
            return f"Pomijam - wykryto '{keyword}' w tytule"
    
    return None

if __name__ == "__main__":
    download_videos()