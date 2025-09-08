# Snake — Reinforcement Learning

Projekt polega na uczeniu agenta RL (PPO, Stable Baselines3) do gry w Snake w środowisku opartym o Gymnasium i Pygame.

## Jak uruchomić

1. Przejdź do głównego katalogu repozytorium i aktywuj środowisko:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Zainstaluj wymagane pakiety:
   ```bash
   pip install -r requirements.txt
   pip uninstal torch
   pip install torch>=2.7 --index-url https://download.pytorch.org/whl/cu128
   ```
3. Rozpocznij trening agenta:
   ```bash
   python snake/scripts/train.py
   ```
4. Przetestuj wytrenowanego agenta:
   ```bash
   python snake/scripts/test_snake_model.py
   ```

## Czego się spodziewać
- Podczas treningu agent uczy się zdobywać jak najwięcej punktów, unikając kolizji.
- Wyniki i modele zapisywane są w katalogach `snake/models/` i `snake/logs/`.
- W trakcie testowania pojawi się okno z wizualizacją gry Snake, a agent będzie grał samodzielnie (o ile to włączymy w configu).
- W katalogu `logs/` generowany jest wykres postępu treningu (`training_progress.png`).

## Jak działa środowisko
- Obserwacja agenta to wektor cech (np. mini-mapa wokół głowy, kierunek, odległość do jedzenia).
- Akcje: skręt w lewo, prosto, skręt w prawo.
- Nagrody: za zebranie jabłka, za przeżycie, kara za kolizję lub zbyt długie kręcenie się bez postępu.
- Możesz modyfikować parametry środowiska i modelu w pliku `snake/config/config.yaml`.

## Pliki
- `model.py` — definicja środowiska Snake (Gym)
- `train.py` — trening agenta RL
- `test_snake_model.py` — testowanie wytrenowanego agenta
- `plot_train_progress.py` — generowanie wykresu postępu
- `config/config.yaml` — konfiguracja środowiska, modelu i ścieżek