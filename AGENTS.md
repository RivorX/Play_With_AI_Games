# Zasady implementacji

- Funkcjonalność zaimplementowana w C++ ma działać wyłącznie przez C++. Nie dodawaj ani nie utrzymuj fallbacków do Pythona dla tych samych operacji.
- Gdy wymagany moduł C++ jest niedostępny lub nie działa, zgłoś czytelny błąd zamiast po cichu uruchamiać implementację w Pythonie.
- Nie dodawaj martwych fallbacków, nieużywanych implementacji zapasowych ani kodu awaryjnego „na przyszłość”.
- Przy zmianach w istniejącym kodzie usuwaj fallbacki do Pythona w zmienianym zakresie. Python może nadal pełnić rolę warstwy sterującej, ale nie zastępować implementacji C++.

# Weryfikacja zmian

- Dla małych, prostych zmian nie dodawaj ani nie uruchamiaj testów. Wystarczy przegląd zmienionego kodu lub diffu.
- Testy stosuj przy większych zmianach lub istotnym ryzyku błędu; dobieraj ich zakres proporcjonalnie do zmiany.
