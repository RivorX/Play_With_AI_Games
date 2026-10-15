# CNN v9 — SE ResNet

`se_cnn_v9` to dotychczasowy model szachowy. Wspólny **trunk** (wieża konwolucyjna) buduje cechy pozycji, a osobne **głowice** przewidują ruchy i wynik partii. Opis dotyczy domyślnego profilu `v9.9b`.

```text
80 kanałów wejścia → CoordConv 3×3 → 128 kanałów
→ 8 bloków residual z SE → policy i value
```

- **Wejście:** bieżąca plansza i cztery poprzednie, po 16 kanałów na pozycję. Perspektywa jest dostosowana do gracza wykonującego ruch.
- **Trunk:** każdy blok ma dwie konwolucje 3×3 i połączenie residual, które dodaje wejście do przetworzonych cech. Ułatwia to uczenie głębszej sieci.
- **SE:** podsumowuje całą planszę średnią kanałów i uczy się, które kanały wzmacniać lub osłabiać.
- **Policy:** konwolucja 3×3, 128→96, potem 1×1, 96→73; logity są mapowane na 1858 ruchów. MCTS wykorzystuje je jako podpowiedź, co przeszukiwać.
- **Value:** konwolucja 3×3 do 16 kanałów, spłaszczenie i globalna średnia trunku, następnie FC384→WDL3. WDL oznacza prawdopodobieństwa wygranej, remisu i przegranej.
- **Pomocnicze wyjścia:** search-Q i liczba ruchów do końca partii, używane zależnie od trybu treningu/inferencji.

Domyślny model ma **3 125 774 parametry** i około **166,26 mln MAC na pozycję**. MAC liczy tu operacje Conv/Linear standardowego forwardu policy+value, bez opcjonalnych wyjść; nie jest pomiarem czasu GPU.

Profil: [`se_cnn_v9.yaml`](../../config/models/se_cnn_v9.yaml). Następca: [CNN v10](cnn_v10.md). V9 zachowuje własny enkoder i zgodność ze swoimi checkpointami.
