"""
Skrypt do generowania przykładowych torów
"""
import sys
import os

# Dodaj ścieżkę do scripts
sys.path.append(os.path.dirname(__file__))

from track import Track


def create_all_tracks():
    """Tworzy wszystkie przykładowe tory"""
    
    # Ustal bazowy katalog projektu
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tracks_dir = os.path.join(base_dir, 'tracks')
    
    # Upewnij się że katalog istnieje
    os.makedirs(tracks_dir, exist_ok=True)
    
    print("Tworzenie przykładowych torów...")
    
    # Prosty tor
    print("1. Tworzenie prostego toru...")
    track1 = Track.create_simple_track()
    track1.save(tracks_dir)
    
    # Owalny tor
    print("2. Tworzenie owalnego toru...")
    track2 = Track.create_oval_track()
    track2.save(tracks_dir)
    
    # Zygzakowaty tor
    print("3. Tworzenie zygzakowatego toru...")
    track3 = Track.create_zigzag_track()
    track3.save(tracks_dir)
    
    print("\nWszystkie tory zostały utworzone w katalogu 'tracks/'")
    print(f"Utworzono 3 tory:")
    print("  - simple.json")
    print("  - oval.json")
    print("  - zigzag.json")
    print(f"  Lokalizacja: {tracks_dir}")


if __name__ == "__main__":
    create_all_tracks()
