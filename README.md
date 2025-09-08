# Game AI Lab

To repozytorium zawiera różne projekty związane ze sztuczną inteligencją w grach.

## Zawartość

- [snake/](snake/README.md) — Projekt uczenia agenta RL do gry Snake

Każdy podkatalog zawiera własny README z instrukcją uruchomienia i opisem działania.

## Wymagania ogólne
- Python 3.10+
- pip
- Zalecane środowisko wirtualne (venv)

## Instalacja (przykład)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip uninstal torch
pip install torch>=2.7 --index-url https://download.pytorch.org/whl/cu128
```

Szczegóły dotyczące uruchamiania i konfiguracji znajdziesz w README odpowiedniego podprojektu.