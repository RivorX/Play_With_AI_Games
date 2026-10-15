# CNN v10 — nested bottleneck i global pooling

`nested_cnn_v10` to osobna architektura CNN. Jej celem jest bogatsze przetwarzanie pozycji i globalnego kontekstu przy podobnym budżecie obliczeń do [v9](cnn_v9.md). Wyższa siła gry wymaga potwierdzenia treningiem i areną.

```text
72 kanały wejścia → CoordConv 3×3 → 192 kanały
→ 6 bloków nested bottleneck 192/96 → policy i value
```

- **Wejście:** bieżąca plansza i cztery poprzednie. Każda ma 12 kanałów figur oraz kanał powtórzeń; dodatkowe 7 kanałów opisuje bieżące prawa obu stron do roszady, en passant, licznik półruchów i obszar planszy. Nieznana historia powtórzeń ma wartość `-1`.
- **Nested bottleneck:** blok zwęża cechy z 192 do 96 kanałów przez 1×1, wykonuje dwie wewnętrzne jednostki residual (łącznie cztery 3×3), po czym wraca do 192 kanałów. Szerokie cechy są zachowane przez zewnętrzne połączenie residual.
- **Global pooling:** w blokach 2, 4 i 6 oraz w policy średnie i maksima kanałów sterują dodatkowymi przesunięciami cech. Przekazuje to kontekst całej planszy do lokalnego przetwarzania; mapa 8×8 pozostaje zachowana.
- **Policy:** dwie konwolucje 3×3, 192→96→73, z globalnym kontekstem i mapowaniem na te same 1858 ruchów co v9.
- **Value:** konwolucja 1×1 do 32 kanałów, spłaszczenie oraz średnie i maksima trunku, następnie FC192→WDL3. Pozostają opcjonalne search-Q i moves-left.

| Domyślny profil | Parametry | MAC na pozycję |
| --- | ---: | ---: |
| v9 | 3 125 774 | 166,26 mln |
| v10 | 3 201 262 | 165,35 mln |

MAC obejmuje Conv/Linear standardowego forwardu policy+value, bez opcjonalnych wyjść. Podobna liczba MAC nie gwarantuje podobnego czasu GPU. V10 ma własny kontrakt danych i wymaga własnych wag; można natomiast rozgrywać mecze v9 przeciw v10.

Profil: [`nested_cnn_v10.yaml`](../../config/models/nested_cnn_v10.yaml). Uruchamianie, nazwy checkpointów i ograniczenia: [instrukcja v10](../nested_cnn_v10.md).
