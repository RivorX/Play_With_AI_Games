# Game AI Lab

Zbiór prostych projektów i eksperymentów związanych z zastosowaniem algorytmów uczenia maszynowego w grach.

W repo znajdziesz m.in. implementację agenta RL dla gry Snake wraz z narzędziami do treningu, testów i wizualizacji wyników.

## Podprojekty

- [snake/](snake/README.md) — agent PPO dla gry Snake (trening, testy, analiza, generowanie GIF)

Każdy podkatalog zawiera własne instrukcje uruchomieniowe i opis konfiguracji.

## Wymagania

- Python 3.10+
- pip
- (opcjonalnie) środowisko wirtualne, np. `venv`

## Szybka instalacja

Przykład (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Jeśli potrzebujesz konkretnej wersji torch (dostosuj do swojej konfiguracji GPU/CPU):
pip uninstall torch
pip install torch>=2.7 --index-url https://download.pytorch.org/whl/cu128
```

Szczegóły dotyczące uruchamiania i konfiguracji znajdziesz w README odpowiedniego podprojektu.