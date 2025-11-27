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
    
    # Ulepszona konfiguracja pobierania
    ydl_opts = {
        # Mniej restrykcyjna selekcja formatu - bierz najlepszy dostępny
        'format': 'best[height<=720]/best',
        'outtmpl': os.path.join(raw_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'max_downloads': 5,
        'match_filter': filter_tournament_videos,
        'restrictfilenames': True,
        'windowsfilenames': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'noprogress': False,
        # Lepsze socket timeout dla stabilniejszego połączenia
        'socket_timeout': 30,
        # Pobieranie fragmentów w parallel
        'concurrent_fragment_downloads': 4,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'skip': ['hls', 'dash']
            }
        },
    }
    
    search_query = "ytsearch5:Clash Royale Top Ladder Gameplay" 
    
    print(f"Pobieranie wideo do: {raw_dir}")
    print("="*60)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            
            if 'entries' in info:
                videos = [entry for entry in info['entries'] if entry]
                print(f"Znaleziono {len(videos)} filmów do pobrania\n")
                
                for idx, video in enumerate(videos[:5], 1):
                    title = video.get('title', 'Unknown')
                    if len(title) > 50:
                        title = title[:47] + "..."
                    print(f"[{idx}/5] Pobieranie: {title}")
                    
                    try:
                        ydl.download([video['webpage_url']])
                        print(f"      ✓ Pobrano\n")
                    except Exception as e:
                        print(f"      ✗ Błąd: {e}\n")
                        
    except yt_dlp.utils.MaxDownloadsReached:
        pass
    except Exception as e:
        print(f"\n✗ Błąd podczas pobierania: {e}")
        return
    
    downloaded = [f for f in os.listdir(raw_dir) if f.endswith(".mp4")]
    print("="*60)
    print(f"PODSUMOWANIE:")
    print(f"  Pobrano: {len(downloaded)} plików")
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