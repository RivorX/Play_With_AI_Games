# Chess AI

Silnik szachowy oparty na deep learning (CNN) i MCTS, inspirowany AlphaZero.

## Etapy treningu

1. **IL** (Imitation Learning) - nauka z partii mistrzowskich (PGN)
2. **RL** (Reinforcement Learning) - samogra z MCTS i self-play
3. **Evaluation** - pomiar siły gry (Elo vs Stockfish)

## Funkcjonalności

### Model
- **Policy head**: AZ-like classic chess (`8x8x73`)
- **Value head**: WDL classification (Win/Draw/Loss)
- **POV**: wszystkie pozycje z perspektywy gracza na ruchu
- **Historia**: `history_positions` pozycji wstecz (sliding window)
- **Metadane szachowe**: castling, en passant, halfmove, fullmove
- **Promocje**: przestrzeń akcji AlphaZero (`ACTION_SIZE = 4672`)

### Trening
- **SWA** (Stochastic Weight Averaging): uśrednianie wag od epoch 15
- **Elo tracking**: asynchroniczna ewaluacja podczas IL
- **Mixed precision**: AMP (float16/bfloat16)
- **Resume/Transfer**: menu startowe z kompatybilnością checkpointów

### Gra
- **GUI**: pygame interface (Human vs AI, AI vs AI, Human vs Human)
- **MCTS**: opcjonalny (toggle klawiszem `M` w GUI)
- **UCI**: adapter do silników szachowych

## Architektura modelu

`ChessNet` (`chess/src/model.py`) to pre-activation ResNet z:
- wejscie: `16 * (1 + history_positions)` kanalow
- trunk: bloki residualne
- policy head: `1x1 conv + BN + FC` (AZ-like, `8x8x73`)
- value head: 3 klasy WDL

Opcjonalne elementy (zaleznie od `config.yaml`):
- `SEBlock` (Squeeze-and-Excitation)
- `CoordConv2d`
- `LayerScale`
- `Stochastic Depth`

## Struktura katalogow

```text
chess/
|- config/config.yaml
|- data/                     # PGN + preprocessing cache
|- engines/                  # Stockfish cache (auto-download)
|- logs/                     # CSV i PNG z treningu
|- models/
|  |- best_model_il.pt
|  |- best_model_rl.pt
|  |- IL/                    # checkpointy IL
|  |- RL/                    # checkpointy RL
|- scripts/
|  |- train_il.py
|  |- train_rl.py
|  |- eval_elo.py
|  |- download_nikonoel_pgns.py
|  |- play.py
|  |- utils/
|     |- il/
|     |- rl/
|     |- shared/
|     |- ui/
|- src/
   |- model.py
   |- mcts.py
   |- data.py
   |- batch_selfplay.py
   |- utils/
```

## Szybki start

Uruchamiaj z root repo (`Play_With_AI_Games`):

```bash
pip install -r requirements.txt
python chess/scripts/download_nikonoel_pgns.py
python chess/scripts/train_il.py
python chess/scripts/train_rl.py
python chess/scripts/play.py
```

## Pobieranie PGN

Datasety z `https://database.nikonoel.fr` mozna listowac i pobierac skryptem:

```bash
python chess/scripts/download_nikonoel_pgns.py
```

- bez argumentow skrypt przechodzi w tryb interaktywny i pyta, co pokazac / pobrac
- rozpakowane pliki `.pgn` trafiaja zawsze do `chess/data`
- archiwa tymczasowe sa trzymane pod `chess/data/_archives/nikonoel`
- w `config.yaml` `data.max_games` moze byc liczbą albo `"max"` dla calego PGN

## IL (Imitation Learning)

### Start

```bash
python chess/scripts/train_il.py
```

### Workflow

1. **Menu startowe** (jeśli istnieją checkpointy):
   ```
   ━━━ IL STARTUP MENU ━━━
   
   [1] New training from scratch
   [2] Resume full state (optimizer + scheduler + scaler)
   [3] Transfer matching weights only
   
   Select option [1-3]:
   ```

2. **Lista checkpointów** (dla resume/transfer):
   ```
   ID  Epoch  Top1     Val Loss  Compat  Size    Path
   ─────────────────────────────────────────────────────
   1   ep 20  67.84%   0.8234   100.0%   45.2MB  v5.1_epoch_20.pt
   2   ep 15  65.12%   0.8891    98.7%   45.1MB  v5.0_epoch_15.pt
   3   ep 10  62.45%   0.9123    85.3%   38.4MB  v4.9_epoch_10.pt
   ```
   - **Compat**: % kompatybilności architektury (matching tensors)
   - Resume wymaga 100% (strict load), transfer działa z <100%

3. **Resume mode**:
### Start

```bash
python chess/scripts/train_rl.py
```

### Workflow

1. **Inicjalizacja**:
   - Ładuje `best_model_il.pt` (jeśli istnieje)
   - Menu startowe: new/resume/transfer (jak w IL)

2. **Self-play** (parallel MCTS):
   ```
   🎯 Parallel MCTS Self-Play:
      Self-play device: cuda
      Workers: 4
      Games per worker: 25, 25, 25, 25 (balanced)
      Total games: 100
      MCTS simulations: 200
   
   ✅ MCTS Self-play completed:
      Positions: 4,832
      Games: 100
      Self-play time: 45.3s
      Speed: 106.7 positions/s
      Avg game length: 48.3 moves
   ```

3. **Training loop**:
   - Batch sampling z **replay buffer**
   - Policy target: MCTS visit distribution (nie legal moves!)
   - Value target: końcowy wynik partii (główny WDL) + pomocnicza lokalna ocena `root_q`
   - Value error focus: do 25% największych bieżących błędów `|value-root_q|`
     dostaje maksymalnie `1.5x` względnej wagi (bez duplikowania próbek)
   - Jednolity sampling z krótkiego FIFO replayu

4. **Evaluation**:
   - Co `eval_every` iteracji: AI vs Best Model
   - Win rate > threshold → promote current to best
   - Zapis: `best_model_rl.pt`

5. **Fixed MCTS sampling**:
   - Stała temperatura do `mcts_temperature_threshold` plies
   - Deterministyczny wybór później

### Replay Buffer

- **Capacity**: `run.games_per_iteration * replay.buffer_multiplier`
- **FIFO**: stare pozycje wypierane przez nowe
- **Uniform sampling**: każda pozycja w aktywnym FIFO ma równą szansę
- **Error-focused value loss**: sampling pozostaje równomierny, ale trudne
  pozycje z wiarygodnym `root_q` otrzymują umiarkowanie większą wagę value
- **Difficulty-aware MCTS**: trudność pozycji łączy niepewność policy, względną
  różnicę dwóch najlepszych ruchów, branching i niepewność value. Najłatwiejsze
  pozycje PCR dostają 16 symulacji, pełne search'e 64-320, a średnia grupy
  pozostaje dokładnie równa `search.simulations`
- **Shared tree + tree reuse**: w zwykłym guarded-actor self-play obie strony
  używają jednego drzewa. Po ruchu odwiedzony podwęzeł zostaje nowym rootem,
  a niepotrzebni przodkowie i rodzeństwo są od razu zwalniani. Partie dwóch
  różnych checkpointów zachowują osobne drzewa, aby nie mieszać ich priorytetów
  ani ocen pozycji

### Checkpointy

- Po każdej iteracji: najnowszy stan ze stanem optymalizatora w `models/RL/*_latest.pt`
- Po promocji: `best_model_rl.pt` oraz wersjonowany `models/RL/*_best.pt`

### Start

```bash
python chess/scripts/eval_elo.py
```

**Brak argumentów CLI** - wszystko przez interaktywne menu.

### Workflow

1. **Wybór modeli**:
   ```
   ━━━ Model Selection ━━━
   
   Select model scope:
   [1] Best model only (best_model_il.pt + SWA)
   [2] Choose model IDs (custom selection)
   [3] All listed models
   
   Narzędzia

### List Models

```bash
python chess/scripts/list_models.py
```

Wyświetla wszystkie checkpointy z metadanymi:

```
═══════════════════════════════════════════════════════════════════════════
Model Checkpoints
═══════════════════════════════════════════════════════════════════════════
 ID  Folder  Version  Epoch   Top1      ValLoss    PolLoss      Elo   SWA  Opt   SizeMB  Updated           Checkpoint
---- ------- -------- ------ -------- ---------- ---------- -------- ---- ---- ------- ----------------- ---------
-- root --
  1  root    v5.1        20   67.84%     0.8234     0.6123     1847   no   yes   45.2  2026-02-17 14:23  best_model_il.pt
  2  root    v5.1        20   68.12%     0.8156     0.6089     1923   yes  yes   45.3  2026-02-17 14:30  best_model_il_swa.pt
-- IL --
  3  IL      v5.1        20   67.84%     0.8234     0.6123     1847   no   yes   45.2  2026-02-17 14:23  v5.1_epoch_20.pt
  4  IL      v5.1        15   65.12%     0.8891     0.6445     1756   no   yes   45.1  2026-02-17 12:45  v5.1_epoch_15.pt
  5  IL      v5.0        15   62.89%     0.9234     0.6789     1689   no   yes   45.0  2026-02-12 18:34  v5.0_epoch_15.pt
-- RL --
  6  RL      v5.1       143   69.34%     0.7845     0.5923     2034   no   yes   45.4  2026-02-16 22:11  rl_iter_0143.pt
───────────────────────────────────────────────────────────────────────────
Total: 6 | Valid: 6 | With Elo: 6 | With optimizer: 6 | SWA-tagged: 1
═══════════════════════════════════════════════════════════════════════════
```

**Kolumny**:
- **Compat**: % kompatybilności z obecną architekturą
- **SWA**: czy checkpoint powstał z SWA finalization
- **Opt**: czy zawiera optimizer state (resume vs transfer)
- **Elo**: estimated Elo (jeśli był mierzony)

### GUI

```bash
python chess/scripts/play.py
```

**Tryby gry**:
- Human vs AI
- AI vs AI
- Human vs Human

**Klawisze**:
- `M` - toggle MCTS (network-only ↔ MCTS)
- `R` - restart game
- `U` - cofnij ruch

**Flagi**:
- `--no-mcts` - uruchom bez MCTS (tylko raw network)

**Autosave**:
- Zapisuje gry do `chess/games/*.pgn`
- PGN z metadanymi (model, Elo, MCTS settings)

### UCI Adapter

### Struktura

```
chess/logs/
├── il_training_v5.1_20260217_143025.csv    # Metryki IL
├── il_training_v5.1_20260217_143025.png    # Wykresy IL
├── elo_comparison_20260217_153045.csv      # Wyniki Elo
└── debug/
    └── training_profile_*.txt              # Profile (jeśli debug=True)

chess/models/
├── best_model_il.pt                        # Najlepszy IL
├── best_model_il_swa.pt                    # Najlepszy IL SWA
├── best_model_rl.pt                        # Najlepszy RL
├── IL/
│   ├── v5.1_epoch_05.pt
│   ├── v5.1_epoch_10.pt
│   ├── v5.1_epoch_10_swa.pt                # SWA snapshot
│   └── v5.1_epoch_15.pt
└── RL/
    ├── rl_iter_0100.pt
    └── rl_iter_0200.pt
```

### CSV Format (IL)

```csv
epoch,train_loss,val_loss,train_policy,val_policy,train_top1,val_top1,val_mae,lr,estimated_elo
1,2.3456,2.4567,1.8234,1.8923,0.4523,0.4312,0.3456,0.001,
5,1.2345,1.3456,0.9123,0.9456,0.6234,0.6123,0.2345,0.0009,1623
10,0.9876,1.0234,0.7234,0.7456,0.6789,0.6623,0.1987,0.0007,1745
```

### Przerwanie (`Ctrl+C`)

- **Graceful shutdown**: finalizacja SWA, zapis ostatniego checkpointu
- **Cleanup**: usuwa incomplete CSV (jeśli PNG nie powstał)
- **Resume**: możliwe od ostatniego zapisanego epocha
   ```

### Format wyniku

```
Model                    Elo  ±Conf  vs1320  vs1500  vs1700  vs1900  vs2200
─────────────────────────────────────────────────────────────────────────────
v5.1_epoch_20.pt        1847   ±45   6/6     6/6     5/6     3/6     1/6
v5.1_epoch_20_swa.pt    1923   ±38   6/6     6/6     6/6     4/6     2/6
best_model_il.pt        1805   ±52   6/6     6/6     4/6     2/6     1/6
```

### Automatyczna aktualizacja checkpointu

- Zapisuje `estimated_elo` do pliku `.pt`
- Widoczne w `list_models.py` i IL resume menuhutdown
- Finalizacja SWA (jeśli zebrane dane)
- Cleanup incomplete logs

## RL (Reinforcement Learning)

Start:

```bash
python chess/scripts/train_rl.py
```

Najwazniejsze zachowania:
- RL startuje od `best_model_il.pt` (jesli plik istnieje)
- Samogra przez `batch_selfplay` + MCTS worker, zawsze bez zewnętrznego silnika
- Guarded actor gra przeciwko sobie; learner nie generuje danych, dopóki nie
  przejdzie lekkiej bramki non-inferiority względem zaakceptowanego best
- Replay łączy świeże dane actora z przypiętym archiwum ostatniego championa
- Stałe parametry MCTS oraz LR schedule
- Eval vs best model co `eval_every`
- Zapisy:
  - `best_model_rl.pt`
  - `models/RL/*_latest.pt` po każdej iteracji

Stockfish jest wyłącznie niezależnym estymatorem Elo. Nigdy nie jest
przeciwnikiem self-play i nie dostarcza replayu, ruchów nauczyciela ani targetów.

### Logi RL (`schema_version=12`)

- główny CSV: uczenie, eval, lower bound promocji, anchor i Elo,
- `*_data_quality.csv`: replay, champion reservoir, targety i zachowanie MCTS,
- `*_performance.csv`: throughput, czasy etapów, batching, latency i bottleneck.

Metryka ma jednego właściciela: eval nie jest kopiowany do data-quality, a czasy
profilera nie trafiają do głównego CSV. Schematy są zdefiniowane w
`scripts/utils/rl/rl_log_schema.py`.

## Ewaluacja Elo

Start:

```bash
python chess/scripts/eval_elo.py
```

Przy recznym uruchomieniu skrypt pyta, czy test ma byc:
- raw network
- MCTS

Przydatne flagi:
- `--model` (wspiera wildcard)
- `--quick`
- `--mcts` / `--no-mcts`
- `--simulations`
- `--levels`
- `--games`
- `--output`

## GUI i UCI

GUI:

```bash
python chess/scripts/play.py
```

- obsluga Human vs AI / AI vs AI / Human vs Human
- mozliwosc gry z lub bez MCTS (`--no-mcts`)
- podczas Human vs AI mozna przelaczyc tryb klawiszem `M`

UCI adapter:

```bash
python chess/scripts/utils/ui/uci_engine.py
```

## Logi i checkpointy

`chess/logs/`:
- `*.csv` metryki treningu
- `*.png` wykresy treningu

`chess/models/`:
- best modele (`best_model_il.pt`, `best_model_rl.pt`)
- checkpointy etapowe (`models/IL`, `models/RL`)

Przerwanie treningu (`Ctrl+C`):
- IL i RL koncza sie graceful
- jesli dla danego runu nie powstal jeszcze PNG, tymczasowy CSV moze zostac usuniety

## Dane

PGN wrzuc do:
- `chess/data/`

Projekt byl przygotowywany pod zbiory typu Lichess Elite (wysokie Elo).
Filtry jak `min_elo`, sampling i deduplikacje ustawiasz w `chess/config/config.yaml`.

## Konfiguracja

Glowne sekcje:
- `data`
- `model`
- `imitation_learning`
- `reinforcement_learning`
- `elo_estimation`
- `hardware`
- `debug`

Punkt startowy konfiguracji:
- `chess/config/config.yaml`
