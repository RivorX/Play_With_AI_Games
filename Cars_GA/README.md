# Cars_GA - Algorytm Genetyczny dla Samochodów 🏎️🧬

Projekt wykorzystujący algorytm genetyczny do trenowania AI, które uczy się jeździć samochodami po torach wyścigowych.

## Opis

Projekt implementuje:
- **Algorytm genetyczny** - ewolucyjne uczenie AI
- **Sieci neuronowe** - mózg każdego samochodu
- **Fizyka samochodu** - realistyczna symulacja ruchu
- **System torów** - różnorodne tory wyścigowe
- **Edytor torów** - twórz własne tory
- **Wizualizacja** - obserwuj jak AI się uczy

## Instalacja

1. Zainstaluj wymagane pakiety:
```bash
pip install -r requirements.txt
```

## Użycie

### Uruchomienie aplikacji głównej
```bash
python .\Cars_GA\scripts\main.py
```

### Dostępne tryby:

1. **Trenowanie AI** - Obserwuj jak populacja samochodów ewoluuje
2. **Testowanie** - Przetestuj wytrenowany model
3. **Edytor torów** - Stwórz własne tory wyścigowe

## Jak to działa?

### Algorytm Genetyczny:
1. Tworzona jest populacja samochodów z losowymi sieciami neuronowymi
2. Każdy samochód jeździ po torze i zbiera punkty (fitness)
3. Najlepsze samochody są wybierane do reprodukcji
4. Ich "geny" (wagi sieci) są krzyżowane i mutowane
5. Proces powtarza się przez wiele generacji

### Sieć Neuronowa:
- **Wejście**: Odczyty z 5 czujników odległości
- **Ukryte warstwy**: Przetwarzanie informacji
- **Wyjście**: 4 akcje (lewo, prawo, przyspieszenie, hamowanie)

### Funkcja Fitness:
- Punkty za przejechanie checkpointów
- Punkty za przebytą odległość
- Kary za czas i zderzenia

## Struktura Projektu

```
Cars_GA/
├── config/          # Konfiguracja
├── models/          # Zapisane modele
├── tracks/          # Tory wyścigowe
├── logs/            # Logi treningu
├── scripts/         # Kod źródłowy
│   ├── main.py              # Główna aplikacja
│   ├── car.py               # Fizyka samochodu
│   ├── neural_network.py   # Sieć neuronowa
│   ├── genetic_algorithm.py # Algorytm genetyczny
│   ├── track.py             # System torów
│   ├── track_editor.py      # Edytor torów
│   └── utils/               # Narzędzia pomocnicze
└── docs/            # Dokumentacja
```

## Konfiguracja

Edytuj `config/config.yaml` aby zmienić:
- Parametry algorytmu genetycznego
- Strukturę sieci neuronowej
- Parametry fizyki samochodu
- Funkcję fitness

### Ważne parametry:

**Próg akcji** (`car.action_threshold`):
- Wartość 0.3 oznacza że akcja aktywuje się gdy output > 0.3
- Niższy = łatwiej aktywować akcje (ale może być chaotyczne)
- Wyższy = trudniej aktywować (bot może się nie ruszać)

**Początkowa prędkość** (`car.initial_speed`):
- Wartość 1.0 daje samochodom lekki "impuls" startowy
- Ułatwia początek ruchu i naukę kierowania

## Tworzenie Torów

W edytorze torów:
- **LPM** - Rysuj ściany
- **PPM** - Dodaj checkpointy
- **S** - Zapisz tor
- **C** - Wyczyść
- **ESC** - Wyjdź

## Autor

Projekt stworzony z pomocą AI dla nauki algorytmów genetycznych i sieci neuronowych.
