# Rejestr eksperymentów Chess RL

Ten plik zapisuje **co porównaliśmy, z czym i z jakim skutkiem**. Nowy eksperyment dopisujemy po uzyskaniu wyniku, także gdy wynik jest negatywny. Szczegółowe pomiary mogą pozostać w `chess/logs/analysis/`; ten katalog jest lokalnym, ignorowanym przez Git archiwum, więc najważniejsze liczby i decyzja muszą znaleźć się również tutaj.

## Jak oceniamy wynik

- **Pomogło** — poprawa siły gry w porównaniu z właściwym punktem odniesienia, potwierdzona niezależną areną przy tym samym budżecie. Zapisujemy liczbę gier i przedział ufności.
- **Zaszkodziło** — istotne pogorszenie głównego miernika lub naruszenie ustalonej wcześniej bramki bezpieczeństwa.
- **Nierozstrzygnięte** — wynik obejmuje remis albo mamy tylko pośredni pomiar, np. stratę, KL lub zamrożony replay. Taki wynik nie jest dowodem wzrostu Elo.
- **Odrzucone** — wariant nie przeszedł ustalonych bramek i nie trafił do konfiguracji produkcyjnej. Przy tej decyzji dopisujemy, czy pomógł, zaszkodził lub pozostał nierozstrzygnięty w poszczególnych miernikach.

## Wyniki

| Data / ID | Zmiana i punkt odniesienia | Pomiar | Wniosek i decyzja |
| --- | --- | --- | --- |
| 2026-09-26 / E3-fresh | Korekty z wysokim `Q-delta` wobec niskiego; świeże wyszukiwanie 384 symulacji, 256 pozycji z rozłącznych gier | Zgodność zapisanego najlepszego ruchu z najlepszym ruchem według świeżego Q: Q1 51,6%, Q4 82,8% | **Pomaga ocenić wiarygodność sygnału**, ale nie mierzy Elo ani jakości ruchu poza tym samym modelem i algorytmem. Bez zmiany treningu. Szczegóły: `chess/logs/analysis/E3_fresh384_qdelta_gate_20260926.json`. |
| 2026-09-26 / E3 | Większa masa zwykłego policy CE dla mocnych korekt wobec obecnego ważenia; ten sam checkpoint i 123 kroki na ramię | Mocne korekty: zmiana prawdopodobieństwa celu +0,000153, przedział obejmuje zero. Zwykłe pozycje: −0,002630, przedział całkowicie poniżej zera | **Zaszkodziło zwykłym pozycjom**, nie wykazano korzyści dla mocnych korekt. **Odrzucone**, bez areny. `chess/logs/analysis/E3_correction_weighting_ab_20260926.json`. |
| 2026-09-26 / E3b | Przesunięcie istniejącej masy rank loss ku najmocniejszemu kwartylowi korekt wobec obecnego rank loss; bez zmiany jego całkowitej masy | Udział Q4 w masie rank loss 36,71% → 42,78%; zmiana prawdopodobieństwa celu Q4 −0,000307, 95% CI [−0,000641; −0,000004] | **Zaszkodziło głównemu miernikowi korekt**. **Odrzucone**, bez areny. `chess/logs/analysis/E3b_rank_reallocation_ab_20260926.json`. |
| 2026-09-26 / E4 | Wyłączenie rank loss (`0,75` → `0`) wobec obecnego ustawienia; ten sam checkpoint, replay i 123 kroki na ramię | Q4: zmiana redukcji KL −0,04777, 95% CI [−0,06123; −0,03470]. Zwykłe pozycje: prawdopodobieństwo celu +0,01847 | **Pomogło zwykłym pozycjom, zaszkodziło korektom**. Nie przeszło wcześniej ustalonej bramki korekt, więc **odrzucone**, bez areny. `chess/logs/analysis/E4_rank_scalar_ab_20260926.json`. |

W E3, E3b i E4 porównywano modele po uczeniu na zamrożonym replayu. Wyniki te rozstrzygają opisane bramki pośrednie, **nie** dowodzą zmiany siły gry. Żaden z tych wariantów nie został wdrożony.

## Szablon następnego wpisu

Skopiuj blok i wypełnij po zakończeniu eksperymentu. Cel i bramki warto zapisać **przed** uruchomieniem porównania.

```text
Data / ID:
Pytanie i hipoteza:
Punkt odniesienia (checkpoint, config, commit):
Jedyna zmieniona rzecz:
Dane i budżet (gry, pozycje, symulacje, kroki, seed):
Główny miernik i bramka decyzji ustalone przed testem:
Mierniki bezpieczeństwa:
Wynik z niepewnością / przedziałem ufności:
Arena na niezależnych grach, jeśli wykonana:
Ocena: pomogło / zaszkodziło / nierozstrzygnięte
Decyzja: wdrożone / odrzucone / do dalszego sprawdzenia
Artefakty i uwagi:
```
