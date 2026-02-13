# ♚ Chess AI

Silnik szachowy oparty na deep learning, inspirowany podejściem AlphaZero. Model uczy się grać w szachy poprzez **Imitation Learning** (nauka z partii mistrzów) a następnie **Reinforcement Learning** (samogra z MCTS).

## Architektura

**ChessNet** — sieć konwolucyjna typu pre-activation ResNet:

- **Input**: `16 × (1 + history)` płaszczyzn na planszę 8×8
  - 12 płaszczyzn: figury (6 × gracz bieżący + 6 × przeciwnik) w POV (Point of View)
  - 4 płaszczyzny metadata: roszada, en passant, halfmove clock, fullmove number
  - × `(1 + history_positions)` — historia pozycji dla kontekstu temporalnego
- **Trunk**: 8 bloków rezydualnych z SE2D (Squeeze-and-Excitation), CoordConv, Stochastic Depth, LayerScale
- **Policy head**: Dual-stream (3×3 spatial conv + Global Average Pool → 2-stage FC → 4272 akcji z promocjami)
- **Value head**: WDL (Win/Draw/Loss) — 3 logity zamiast skalarnej wartości


## Struktura projektu

```
chess/
├── config/
│   └── config.yaml           # Cała konfiguracja (model, trening, dane, hardware)
├── data/
│   ├── *.pgn                 # Pliki PGN z partiami (Lichess Elite)
│   └── preprocessing/        # Cache przetworzonych danych (auto-generowany)
├── engines/                  # Stockfish (auto-pobierany przy Elo estimation)
├── logs/                     # Logi treningowe (CSV + PNG wykresy)
│   └── games/                # Zapisane partie (PGN + YAML setup)
├── models/
│   ├── best_model_il.pt      # Najlepszy model IL
│   ├── best_model_rl.pt      # Najlepszy model RL
│   ├── IL/                   # Checkpointy IL (co N epok)
│   └── RL/                   # Checkpointy RL (co N iteracji)
├── scripts/
│   ├── train_il.py           # Trening Imitation Learning
│   ├── train_rl.py           # Trening Reinforcement Learning
│   ├── play.py               # GUI do gry (Pygame)
│   ├── eval_elo.py           # Ewaluacja Elo vs Stockfish
│   └── utils/
│       ├── il/               # Funkcje treningowe IL, loss
│       ├── rl/               # Replay buffer, temperature, trening RL
│       ├── shared/           # Logger, metrics, elo_estimator
│       └── ui/               # GUI helpers, game setup, UCI engine
└── src/
    ├── model.py              # ChessNet (architektura sieci)
    ├── mcts.py               # Monte Carlo Tree Search (batch MCTS)
    ├── data.py               # Pipeline danych (PGN → tensory)
    ├── batch_selfplay.py     # Równoległa samogra MCTS
    └── utils/
        ├── data_helpers.py   # board_to_tensor, move_to_index, ACTION_SIZE
        ├── data_pipeline.py  # Przetwarzanie PGN (4-fazowe)
        └── data_dataset.py   # Dataset + DataLoader
```

## Pipeline treningu

### 1. Imitation Learning (IL)

Nauka z partii silnych graczy (Lichess Elite, Elo ≥ 2300):

```bash
python chess/scripts/train_il.py
```

**Co robi:**
- Parsuje pliki PGN → kompaktowy format binarny (4-fazowy pipeline)
- POV: plansza zawsze z perspektywy grającego
- Sliding window: dynamiczne budowanie historii pozycji
- Filtrowanie: min Elo, deduplikacja pozycji, sampling po progresie gry
- Trening: policy (cross-entropy) + value (WDL cross-entropy)
- SWA (Stochastic Weight Averaging) dla lepszej generalizacji
- Early stopping z patience

**Metryki:** Policy Top-1/3/5 accuracy, Value MAE, WDL accuracy/CE, **estymowane Elo**

**Hiperparametry** (w `config.yaml`):
- `batch_size: 9216`, `learning_rate: 0.0005`, `epochs: 30`
- `label_smoothing: 0.08`, `scheduler: cosine_decay`
- `sliding_window_stride: 2` (co 2. pozycja)

### 2. Reinforcement Learning (RL)

Samogra z MCTS (styl AlphaZero):

```bash
python chess/scripts/train_rl.py
```

**Co robi:**
- Równoległa samogra MCTS (wielu workerów CPU, batch mode)
- Prioritized Experience Replay
- Temperature schedule (wysoka eksploracja → niska)
- Target network (stabilność treningu)
- Augmentacja danych (lustrzane odbicie planszy)
- Ewaluacja: nowy model vs najlepszy, win rate > 55% → zastąpienie

**Hiperparametry:**
- `games_per_iteration: 100`, `mcts_simulations: 200`
- `batch_size: 4096`, `learning_rate: 0.0002`
- `eval_games: 50`, `win_rate_threshold: 0.55`

### 3. Ewaluacja Elo

Porównanie modeli przez grę ze Stockfishem na różnych poziomach:

```bash
# Ewaluacja najlepszego modelu
python chess/scripts/eval_elo.py

# Porównanie wielu checkpointów
python chess/scripts/eval_elo.py --model models/best_model_il.pt models/IL/*.pt

# Szybki test
python chess/scripts/eval_elo.py --model models/IL/*.pt --quick

# Z MCTS (dokładniejsze Elo, ale wolniejsze)
python chess/scripts/eval_elo.py --mcts --simulations 200
```

**Co robi:**
- Gra szybkie partie vs Stockfish na poziomach `[1320, 1500, 1700, 1900, 2200]`
- Oblicza Performance Rating (MLE) z wyników W/D/L
- Stockfish jest auto-pobierany jeśli nie jest zainstalowany
- Przy wielu modelach — tabela porównawcza + zapis do CSV

**Flagi:**
| Flaga | Opis |
|-------|------|
| `--model` | Ścieżki do modeli (wildcards obsługiwane) |
| `--levels` | Poziomy Elo Stockfisha |
| `--games N` | Gier na poziom |
| `--mcts` | Użyj MCTS (silniejsza gra) |
| `--simulations N` | Symulacje MCTS na ruch |
| `--quick` | Tryb szybki (2 gry/lvl, 3 poziomy) |
| `--output plik.csv` | Zapis wyników do CSV |

### 4. Gra (GUI)

Interfejs graficzny Pygame do gry z modelem:

```bash
# Gra
python chess/scripts/play.py

```

**Tryby gry:** Human vs AI, AI vs AI, Human vs Human

**Funkcje:**
- Podświetlanie legalnych ruchów i bić
- Historia ruchów w sidebarze
- MCTS toggle w trakcie gry
- Obsługa promocji
- Auto-zapis partii do PGN

### 5. UCI Engine

Adapter UCI do użycia w GUI szachowych (Arena, CuteChess, itp.):

```bash
python chess/scripts/utils/ui/uci_engine.py
```

**Opcje UCI:** `UseMCTS`, `Simulations`, `MoveOverhead`, `Temperature`

## Dane treningowe

Projekt używa **Lichess Elite Database** — partii z Lichess gdzie obaj gracze mają Elo ≥ 2300:
- Format: PGN
- Ściągnij z: https://database.nikonoel.fr/
- Umieść w: `chess/data/`

## Wymagania

```
torch (CUDA)
python-chess>=1.999
pygame
numpy
pyyaml
tqdm
```

**GPU:** Testowane na RTX 5060 Ti 16GB. Domyślna konfiguracja wymaga ~12-15 GB VRAM (batch 9216, 128 filtrów).

## Szybki start

```bash
# 1. Zainstaluj zależności
pip install -r requirements.txt

# 2. Ściągnij dane PGN do chess/data/

# 3. Trenuj IL
python chess/scripts/train_il.py

# 4. Sprawdź Elo
python chess/scripts/eval_elo.py

# 5. Zagraj
python chess/scripts/play.py
```

## Logi i monitorowanie

Każdy trening generuje:
- **CSV** z metrykami per-epoka (loss, accuracy, MAE, WDL, Elo)
- **PNG** z wykresami postępu (5 wierszy × 2 kolumny)
- Checkpointy modelu co N epok

Pliki w `chess/logs/`:
```
il_training_v5.0_20260212_173156.csv   # Metryki
il_training_v5.0_20260212_173156.png   # Wykresy
elo_comparison_20260214_*.csv          # Porównanie Elo
```
