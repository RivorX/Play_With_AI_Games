# Solitaire AI (Klondike)

Ten projekt zawiera środowisko Pasjansa (Klondike) oraz skrypty do trenowania agenta AI przy użyciu algorytmu MaskablePPO (Stable Baselines 3).

---

## Przykładowy przebieg (GIF) — szybki podgląd

![Solitaire Run GIF](./docs/solitaire_run.gif)

---

## Struktura

- `config/config.yaml`: Konfiguracja środowiska i modelu.
- `scripts/model.py`: Implementacja środowiska Solitaire (Gymnasium).
- `scripts/cnn.py`: Ekstraktor cech (Feature Extractor) przetwarzający stan gry.
- `scripts/train.py`: Skrypt treningowy.
- `scripts/test_solitaire_model.py`: Wizualizacja rozgrywki agenta w oknie Pygame.
- `scripts/debug/make_gif.py`: Nagrywanie rozgrywki do pliku GIF.
- `models/`: Katalog na zapisane modele.
- `logs/`: Logi treningowe i wykresy postępu.

## Uruchomienie treningu

Aby rozpocząć trening, uruchom skrypt `train.py` z katalogu głównego:

```bash
python ./solitaire/scripts/train.py
```

## Testowanie modelu

### Wizualizacja w oknie Pygame
Aby zobaczyć grę agenta w czasie rzeczywistym:

```bash
python ./solitaire/scripts/test_solitaire_model.py
```

Funkcjonalności:
- Interaktywny wybór modelu (best_model.zip lub solitaire_ppo_model.zip)
- Graficzna wizualizacja stanu gry
- Wyświetlanie podjętych akcji, nagród i wyników
- Komunikaty o przegranej/wygranej/poddaniu się

### Nagrywanie do GIF-a
Aby nagrać rozgrywkę w formacie GIF:

```bash
python ./solitaire/scripts/debug/make_gif.py --episodes 1 --fps 2
```

Parametry:
- `--episodes`: Liczba epizodów do nagrania (default: 1)
- `--fps`: Klatki na sekundę (default: 2)
- `--pause`: Czas pauzy na ostatniej klatce w sekundach (default: 3.0)
- `--out`: Niestandardowa ścieżka wyjściowa GIF-a

Wynik zapisywany jest do `solitaire/logs/solitaire_run.gif`.

## Zasady (Uproszczone)

- **Draw Stock**: Dobieranie karty z talii.
- **Ruchy**:
  - Z talii (Waste) na stół (Tableau).
  - Z talii (Waste) na stosy bazowe (Foundations).
  - Między kolumnami na stole (Tableau -> Tableau).
  - Ze stołu na stosy bazowe (Tableau -> Foundations).
  - Z talii (Stock) na stosy bazowe (Stock -> Foundations).
  - Poddanie się (Surrender) - agent może zakończyć grę, gdy uzna ją za beznadziejną.
- **Cel**: Umieścić wszystkie karty na stosach bazowych (Foundations).

## Architektura Sieci Neuronowej

### Feature Extractor (cnn.py)
Architektura wykorzystuje specjalistyczne podsieci dla każdej części stanu gry:

- **Tableau Network**: Przetwarza kolumny na stole (7×20×4) → 128 cech
- **Foundations Network**: Przetwarza stosy bazowe (4,) → 32 cechy
- **Waste Network**: Przetwarza kartę z talii (3,) → 32 cechy
- **Stock Network**: Przetwarza dostępne karty z talii (52×3) → 64 cechy

Wszystkie cechy łączone są w sieć fuzji (Fusion Network) dającą 256-wymiarową reprezentację stanu.

Każda podsieć zawiera:
- Warstwy liniowe
- **LayerNorm** dla stabilizacji treningu
- Funkcję aktywacji ReLU

### Normalizacja Danych
Wszystkie dane wejściowe są znormalizowane do zakresu [0.0, 1.0]:
- Ranga karty: `rank / 13.0`
- Kolor karty: `suit / 3.0`

To znacznie poprawia stabilność treningu sieci neuronowej.

## Postęp Treningu

Poniższe wykresy pokazują postęp treningu modelu:

### Training Progress
![Training Progress](./docs/training_progress.png)

Wykres zawiera trzy subwykresy pokazujące postęp treningu:

1. **Mean Reward (Średnia Nagroda - górny panel)**:
   - Wykresy pokazują ewolucję średniej nagrody na ostatnie 100 epizodów
   - **Linia niebieska (Raw)**: Surowe wartości
   - **Linia ciemnoniebieska (Rolling Mean)**: Średnia krocząca (wygładzanie)
   - Agent systematycznie uczy się grać, osiągając wyższą średnią nagrodę im dalej w trening
   - Wahania są naturalne w reinforcement learning i wynikają z eksploracji nowych strategii

2. **Mean Episode Length (Średnia Długość Epizodu - środkowy panel)**:
   - Długość epizodów rośnie z czasem treningu:
     - Na początku agent grał ~1-10 kroków (szybko przegrywał)
     - Po optymalizacjach i zwiększeniu kary za poddanie się, agent gra średnio 100-150 kroków
   - Dłuższe epizody oznaczają, że agent bardziej zaangażuje się w grę zamiast szybko się poddawać
   - Zwiększenie wskazuje na lepszą strategię gry i aktywne poszukiwanie rozwiązań

3. **Win Rate (Procent Wygranych - dolny panel)**:
   - Wskaźnik sukcesów na ostatnich 100 epizodach
   - Zaczyna od ~0% i progresywnie wzrasta do ~15-20%
   - Pasjans Klondike jest bardzo trudny (nawet człowiek wygrywa w ~15-20% przypadków)
   - Agent osiąga wyniki porównywalne z człowiekiem, co jest doskonałym rezultatem

## Konfiguracja

Wszystkie parametry są zdefiniowane w `config.yaml`:

### Nagrody (Reward Scaling)
- `move_to_foundation`: +20 (główny cel)
- `flip_tableau_card`: +7 (odkrywanie kart)
- `move_waste_to_tableau`: +2 (przygotowanie)
- `move_tableau_to_tableau`: -0.5 (zniechęcanie do zbyt wielu przesunięć)
- `win_bonus`: +2000 (wygrana gry)
- `invalid_move_penalty`: -0.5 (kara za niewalidny ruch)
- `time_penalty`: -0.1 (kara za każdy krok)
- `recycle_waste_penalty`: -10 (kara za przetasowanie talii)
- `surrender_penalty`: -100 (kara za poddanie się)

### Parametry Treningu
- `n_envs`: 16 (równoległy trening na 16 środowiskach)
- `total_timesteps`: 5,000,000 (całkowita liczba kroków)
- `learning_rate`: 0.0003
- `batch_size`: 2048
- `gamma`: 0.995 (dyskont przyszłych nagród)

## Przykład Rozgrywki

GIF pokazany na początku dokumentacji zawiera przykładową rozgrywkę agenta w akcji:
- Agent wybiera ruchy w oparciu o wytrenowany model neuronowy
- Karty z talii są stopniowo odkrywane
- Karty są przenoszone na stosy bazowe (Foundations) gdy jest to możliwe
- Agent wykorzystuje strategię, aby maksymalizować szanse na zwycięstwo
