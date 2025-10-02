# README dla projektu Agar.io AI

Ten projekt trenuje model AI do gry w Agar.io, najpierw przez imitację ludzkich rozgrywek (Behavioral Cloning), a potem przez uczenie ze wzmocnieniem (PPO). Model gra autonomicznie w przeglądarce.

## Wymagania
- Python 3.10+
- Zainstaluj zależności:
  ```
  pip install requirements.txt
  ```
- Przeglądarka Chrome + ChromeDriver (automatycznie pobierany przez webdriver-manager).

## Instrukcja uruchamiania

1. **Nagrywanie rozgrywek**:
   - Uruchom: `python agario/scripts/record_gameplay.py`
   - Otworzy się Agar.io w Chrome. Wpisz nick, naciśnij Enter.
   - Graj, używając myszy (ruch), W (split), spacji (eject). Sesja trwa 5 min.
   - Powtórz 5+ razy, by zebrać dane (zapis w datasets/recordings/).

2. **Trening modelu**:
   - Uruchom: `python agario/scripts/train.py`
   - Jeśli model istnieje, odpowiedz na pytania (Y/n) o kontynuację i hiperparametry.
   - Model najpierw trenuje się na nagraniach (BC), potem przez PPO.
   - Wyniki (model, logi, wykres) zapisują się w models/ i logs/.

3. **Testowanie modelu**:
   - Uruchom: `python agario/scripts/test_agario_model.py`
   - Model gra w Agar.io przez 10 min, sterując myszą i klawiszami.
   - Średni score wyświetli się w konsoli.

## Ważne uwagi
- Środowisko w train.py to placeholder. Dla realnego treningu potrzebny wrapper Selenium z detekcją OpenCV (kulki/jedzenie).
- Dostosuj crop ekranu w record_gameplay.py i test_agario_model.py (linie z img[100:h-100, 100:w-100]) do swojego monitora.
- Użyj GPU dla szybszego treningu PPO (sprawdź config.yaml: device: cuda).
- W razie błędów (np. shape tensorów), sprawdź logi lub zapytaj o debug.
- Do testów działania ocr używałem tej komendy `python agario/scripts/record_gameplay.py test-ocr a3.png`, gdzie a3.png to obraz w katalogu agario/datasets.

## Pliki
- config/config.yaml: Konfiguracja (hiperparametry, ścieżki).
- scripts/record_gameplay.py: Nagrywanie rozgrywek.
- scripts/train.py: Trening (BC + PPO).
- scripts/model.py: Custom polityka CnnLstmPolicy.
- scripts/conv_lstm.py: Warstwa ConvLSTM dla modelu.
- scripts/test_agario_model.py: Testowanie modelu w przeglądarce.