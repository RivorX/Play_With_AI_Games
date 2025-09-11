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

# Przywracanie modelu z kopii zapasowej

Jeśli chcesz kontynuować trening lub przetestować ostatni zapisany model, a plik modelu został uszkodzony lub chcesz wrócić do poprzedniej wersji:

1. Przejdź do folderu `models`.
2. Znajdź plik z końcówką `.zip.backup` (np. `snake_ppo_model.zip.backup` lub `best_model.zip.backup`).
3. Zmień nazwę pliku, usuwając końcówkę `.backup` (np. `snake_ppo_model.zip.backup` → `snake_ppo_model.zip`).
4. Teraz możesz uruchomić trening lub testowanie – program użyje przywróconego modelu.

To samo dotyczy pliku najlepszego modelu (`best_model.zip.backup`).

# Wyjaśnienie metryk PPO (Stable Baselines3)

Podczas treningu w logach pojawiają się metryki, które pomagają monitorować postęp i stabilność uczenia:

- **approx_kl** — Przybliżona dywergencja KL. Pokazuje, jak bardzo zmieniła się polityka agenta po aktualizacji. Niska wartość = stabilne uczenie.
- **clip_fraction** — Ułamek aktualizacji, które zostały „ucięte” przez mechanizm PPO (clip_range). Wysoka wartość może oznaczać zbyt duże zmiany polityki.
- **clip_range** — Maksymalny zakres zmiany polityki w jednej aktualizacji (ustawienie PPO).
- **entropy_loss** — Entropia polityki. Im bliżej zera, tym bardziej deterministyczne decyzje agenta. Wyższa wartość = więcej eksploracji.
- **explained_variance** — Jak dobrze krytyk (value function) przewiduje nagrody. 0 = brak korelacji, 1 = idealne przewidywanie. Im wyżej, tym lepiej.
- **learning_rate** — Aktualna szybkość uczenia.
- **loss** — Całkowita strata modelu (łączna dla polityki i wartości). Ważny jest trend (czy maleje).
- **n_updates** — Liczba aktualizacji sieci.
- **policy_gradient_loss** — Strata gradientu polityki. Pokazuje, jak mocno aktualizowana jest polityka.
- **value_loss** — Strata funkcji wartości (krytyka). Im niższa, tym lepiej krytyk przewiduje nagrody.

Te wskaźniki pozwalają ocenić, czy agent uczy się stabilnie, nie „wariuje” i czy sieć wartości dobrze przewiduje nagrody.