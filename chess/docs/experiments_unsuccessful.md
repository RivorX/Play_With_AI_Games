# Eksperymenty Chess AI — wyniki negatywne i nierozstrzygnięte

Zebrano tu wyniki, które nie uzasadniają adopcji, nie przeszły ustalonych bramek albo pozostawiają efekt nierozstrzygnięty. Przedział obejmujący remis nie dowodzi równoważności. Wynik opisowy nie jest dowodem przyczynowym.

PMCTS usunięto z runtime; wyniki poniżej pozostają wyłącznie historyczne.

## Modele, IL i RL

| Badanie | Warunki i wynik | Decyzja / ograniczenie |
| --- | --- | --- |
| E3-fresh | Na 256 pozycjach z replayu świeży search lepiej odtwarzał target z wysokim Q-delta niż z niskim: 82,8% vs 51,6%; Spearman ρ=0,557. | Wspiera ranking odtwarzalności korekt w tym samym modelu i algorytmie; nie ocenia prawdziwej jakości ruchu ani Elo. |
| E3 / E3b / E4 | Krótkie uczenie na zamrożonym replayu. Wzmocnienie CE nie poprawiło istotnie Q4 i pogorszyło zwykły oraz szeroki holdout; przesunięcie masy rank loss nie przeszło bramki Q4; wyłączenie rank loss pomagało zwykłym pozycjom, lecz szkodziło korektom Q4. | Odrzucone. Brak aren, więc wyniki dotyczą tylko pośrednich metryk. |
| SB2 — trening na 320 symulacjach | Cztery iteracje od wspólnego V9.9b IL. Model z targetami 320 uzyskał 49,87% wobec modelu z targetami 192 w 384 grach MCTS-192 (95% CI 45,83–54,04%). Trening trwał o 64,6% dłużej. | Nie wykazano przewagi ani równoważności; pozostać przy 192. Próba krótka, a część wspólnej reanalizy używała 192. |
| DS1 — udział trudnych startów 20% vs 40% | Sześć iteracji na ramię. Wariant 40% uzyskał 49,22% w 384 grach (95% CI 45,2–53,2%); wyniki dwóch suite'ów miały różny kierunek. | Nie wykazano poprawy; pozostawić 20%. Jedna krótka próba. |
| Policy-target A/B, A3/A4 | Dodatnie wyniki pojedynczych aren uzasadniły replikację. | To nie była adopcja; szczegóły są w dokumencie wyników potwierdzonych. |
| E5 / E6 — replikacja targetu | E5 po 6 iteracjach: 96 gier na ramię, CI obu wyników wobec IL obejmowały 50%. E6 po 70 iteracjach: baseline 54,69%, `challenge_delta` 54,04% wobec IL; head-to-head 50,65% (CI obejmuje 50%). | Brak dowodu, że `challenge_delta` przewyższa baseline. E5 i E6 używały tego samego seeda treningowego. |
| E7 — temperatura 0,95 vs 1,0 | 24 iteracje na ramię. Wynik 1,0 wobec 0,95 wyniósł 50,26% (95% CI 46,35–54,04%); oba endpointy przekroczyły 50% wobec IL w swoich arenach. | Bramka adopcji temperatury 1,0 nie przeszła; nie promować checkpointów z tej próby. |
| E8 / E9 — zgodność wartości z search | E8 przekroczył bramki końcowej areny w jednym seedzie, lecz kandydaci selekcyjni nie przeszli confirmation; E9 wygrał z IL, ale nie potwierdził przewagi nad wcześniejszym best. | Szczegółowe wyniki są powyżej w dokumencie wyników potwierdzonych. Brak kontrolowanego dowodu efektu straty 0,10. |
| E10 — zrównoważona pula otwarć | 24 rodziny po 32 linie; wynik puli vs legacy 50,13% (95% CI 46,22–54,17%), puli vs IL 55,73%, legacy vs IL 52,86%, po 384 gry. | Główna bramka head-to-head nie przeszła; nie przypisywać puli przewagi nad legacy. |
| E11 / E12 / E15 — replay i BatchNorm | Dodatkowe przejścia poprawiały fit train-corrections, lecz holdout był mieszany; w E15 trzy dodatkowe przejścia nie dały rozstrzygającego zysku holdout. Audyt BN nie mierzył siły. | Brak podstaw do zmiany liczby przejść lub BN; transfer korekt do niezależnych gier pozostał nierozstrzygnięty. |
| E13 / E14 — teacher policy | Reservoir z udziałem teacher 25% nie poprawił siły: jego wynik wobec baseline wyniósł 45,44% (95% CI 41,41–49,48%). W eksperymencie zamrożonego replayu udziały 10% i 25% nie przeszły bramek areny; poprawa dopasowania do korekt współwystępowała z pogorszeniem części holdoutu. | Nie adoptować teacher fraction. Pojedynczy seed i szerokie CI nie dowodzą braku małego efektu. |
| E15 / E17 / E19 — replay i learner | E17: wyższy LR i policy-only nie przeszły bramki przyrostu na holdoucie. E19: okno recent2−full dla `p_target` korekt dało +0,000734 (95% CI −0,000972 do +0,002576); latest1 też nierozstrzygnięte, z przekroczonymi marginesami szkody. | Nie zmieniać LR, objective ani retencji na podstawie tych prób. Metryki pośrednie nie mierzą siły. |
| E20 / E21 — absorpcja korekt i regularyzacja | W stałym replayu learner poprawiał dopasowanie korekt treningowych, ale holdout KL pogarszał się w obu seedach. Dłuższy świeży test RL w dwóch seedach nie wykazał istotnej przewagi strategii równomiernego ani tylko pełnego budżetu w MCTS; wyłączenie dropout/drop-path nie dało potwierdzonego wzrostu. | Słabe przenoszenie korekt pozostaje obserwacją, nie rozstrzygniętą przyczyną. Raw-NN i MCTS należy traktować osobno; nie zmieniono domyślnych wag ani budżetów. |

## Search i jakość ruchów

| Badanie | Warunki i wynik | Decyzja / ograniczenie |
| --- | --- | --- |
| E16 — stabilność targetu i koszt | Na 128 wcześniej oglądanych pozycjach top-target 192 zgadzał się z 320 w 87,5% przypadków; 320 było bliżej proxy 512 o 5,47 pp. Warm median czasu: 7,387 s / 13,741 s / 17,778 s dla 192/320/512. | 512 jest proxy, nie prawdą. Brak dowodu jakości ruchu lub poprawy treningu. |
| E18 — ocena ruchów przez Stockfish | 128 wcześniej oglądanych pozycji; różnica oceny wymuszonego ruchu 512−192: średnia +80,75 cp, mediana 0, 95% CI −2,34 do +234,53; 23 lepsze, 7 gorszych i 98 identycznych ocen. | Brak rozstrzygającego dowodu przewagi 512. Fixed-node score nie jest Elo ani ground truth. |
| Dodatkowe passy replayu / value consistency | E11/E14 wykazały lepszy fit do wybranych korekt, ale bez potwierdzenia siły; E8 był pozytywną areną na pojedynczym seedzie, a E9 nie odtworzył przewagi nad poprzednim best. | Nie wyciągać wniosku o skuteczności treningowej z samego policy KL lub correction audit. |
| Równoległe batchowanie MCTS | Cap 16 uzyskał 25,00% wobec B cap0 (64 gry, jednostronna dolna granica 17,97%); cap32 uzyskał 39,06% (CI 28,13–50,78%). | Oba nie przeszły bramki. Pozostawić cap 0. |
| Persistent search memory — jakość | Zimne drzewo uzyskało 21,9% wobec ciepłego w 64 grach; oba ramiona użyły pełnych 192 świeżych symulacji. `total` vs `fresh`: 57,03% (CI 46,875–67,188%). | Ten test nie rozstrzygnął skali Q. Starsza decyzja o domyślnym `fresh` została zastąpiona obecnym zatwierdzonym B `revisiting/total`. |
| Gumbel `revisiting` kontra standard / PMCTS | Mała arena na 12 otwarciach: standard i `revisiting` zremisowały 8/8/8; obie wygrały z PMCTS 21/1/2. Osobny test stałych 192 symulacji wskazał wyższy koszt `revisiting`. | Ta mała arena sama nie rozstrzyga jakości ani kosztu; obecnym zatwierdzonym defaultem jest B `revisiting/total`, zatwierdzony na podstawie osobnego wyniku i optymalizacji. |

## Izolowane próby A–D, 20 października

Ten historyczny zestaw używał tego samego V9.9b IL, 32 sparowanych otwarć i 64 gier na wariant. Punkty odnoszą się do kandydata wobec kontroli; CI klastrowano po otwarciu. Wcześniejsze rekomendacje `standard`/`fresh` z tego zestawu zostały zastąpione przez aktualnie zatwierdzone B `revisiting/total`.

| Wariant | W–D–L | Score, 95% CI | Mediana czasu searchu: kontrola → kandydat | Wniosek |
| --- | ---: | ---: | ---: | --- |
| A — mniejsza waga odziedziczonego Q | 19–26–19 | 50,00% [50,00–50,00] | 98,05 → 97,88 ms | Brak rozróżnienia w sparowanych parach; nie dowodzi równoważności. |
| B — Gumbel `revisiting` | 35–17–12 | 67,97% [58,59–77,34] | około 2,4× wolniej | Dodatni historyczny wynik przy stałej liczbie symulacji. Brak testu równym czasem ogranicza jego interpretację; późniejsza akceptacja B `revisiting/total` zastąpiła wcześniejszą rekomendację `standard`. |
| C — korekta wartości poddrzewa | 23–23–18 | 53,91% [46,88–61,72] | 97,27 → 98,05 ms | Arena nie wykazała przewagi. |
| D — pamięć wartości transpozycji | 19–26–19 | 50,00% [50,00–50,00] | 98,90 → 112,64 ms | 439 086 prób pamięci dało zero trafień; pełna historia nie dała praktycznej korzyści. |

A/B i C/D używały różnych wersji natywnego API; nie są bezpośrednim rankingiem wariantów między sobą. Dokładne klucze kontekstu historycznego nie były dostępne w późniejszym audycie strukturalnych okazji do transpozycji, więc liczby par nie są rzeczywistymi trafieniami pamięci.

## Końcowy ekran arenowy A2 / C030 / C2

Każdy kandydat A2/C030/C2 porównano z aktualnym zatwierdzonym B `revisiting/total`, w 64 grach i przy 192 świeżych symulacjach na ruch, bez kredytu odziedziczonych wizyt. Wszystkie ramiona używały tego samego V9.9b i identycznych 167 tensorów wag. Te przedziały nie wykluczają ani małej korzyści, ani szkody.

| Wariant | W–D–L kandydata | Score | Klastrowy 95% CI | Wynik ekranu |
| --- | ---: | ---: | ---: | --- |
| A2 | 23–15–26 | 47,656% | 38,281–56,250% | Bramka siły nie przeszła. |
| C030 | 20–21–23 | 47,656% | 38,281–57,812% | Nierozstrzygnięte. |
| C2 | 21–16–27 | 45,313% | 36,719–54,688% | Bramka siły nie przeszła; bez potwierdzenia ani przewagi, ani równoważności. |

Audyt tożsamości wykazał identyczne 167 tensorów wag; różnica surowego hasha pliku wynikała z metadanych. Arena nie daje podstaw do awansu żadnego kandydata. A2 ekranowy koszt wyszukiwania jest niewiarygodny z powodu przeplatanych zapisów; oddzielna, stała kohorta 21 korzeni wykazała medianę ilorazu czasu 1,009× dla A2 i 1,009× dla C2. Wcześniejszy, zaakceptowany ekran C030 miał medianę 1,0118×; nie wykonano dla niego powtórki stałej kohorty. Te małe wyniki kosztowe nie są pomiarem Elo na godzinę.

Dla D wykryto 2 pary węzeł–provider między searchami i 12 w obrębie korzenia przy co najmniej 16 wizytach wyjściowych providera. To liczby strukturalne, nie trafienia ani niezależne dowody; dokładne klucze historii nie były dostępne, więc bezpiecznych cache hits nie da się oszacować. W starszym D z rzeczywistym cache wszystkie 439 086 prób chybiły, a koszt wzrósł o 13,9%.

Deterministyczny fixture ABI18 potwierdził natywną korektę: off = 0, C z czynnikiem 0,30 = −0,07, a C2 = −0,03 po backupie native. W **osobnej, stałej kohorcie kosztowej** C2 miał zmiany Q/drzewa w 14 z 21 korzeni; to zgodne z działaniem mechanizmu, ale nie jest pomiarem areny siły i nie ma licznika aktywacji dla gier screenowych. Fixture potwierdził floor peer-support i shrinkage. Zwykły świeży liść wyklucza wkład własnego węzła, więc nie wykazano, że zmiana aktywuje samokorektę. Zera z pythonowej tabeli nie opisują natywnego C++. C2 nie przeszedł bramki siły; nie wykazano przewagi ani podstaw do stwierdzenia, że mechanizm był nieaktywny.

## Optymalizacje wydajności bez potwierdzenia jakości

| Kierunek | Wynik | Granica |
| --- | --- | --- |
| Centralna inferencja — wcześniejsze scalanie D2H | Dwa przebiegi wykazały 1,0–3,5% krótszy self-play wall time i 4,4–8,8% więcej wierszy/s. | Mała próba, nieustalony efekt całego treningu. |
| GPU categorical/byte-pack i D2H | Testowane warianty pakowania wyjść policy/value na GPU i wspólnego transferu D2H były wolniejsze od sekwencyjnej ścieżki. | Odrzucone; nie pozostawiono kodu produkcyjnego. |
| Adapter `predict_single_position` vs surowa inferencja | V9.9b, bieżące wagi, CUDA AMP BF16, wejście FP16 `[1,80,8,8]`, bez compile/graph; 16 przeplatanych powtórzeń z synchronizacją i zwrotem CPU: stary adapter 5,206 ms, obecny 5,393 ms, raw z dokładnym full-softmax WDL i dekodowaniem CPU 5,567 ms. Raw forward-only 4,92 ms pomija dekodowanie i nie jest porównywalnym adapterem. Polityka FP32 dla wszystkich 1858 ruchów i skalar WDL były dokładnie zgodne. | Nie wykazano szybszego adaptera ani poprawy jakości sieci. Zachować raw fast path MCTS z legalnym gather na GPU; adapter kompatybilności korzysta ze wspólnego raw core. Konwersja value obsługuje `[B,3]` WDL, `[B]` i `[B,1]`, zachowując dtype/device. Poprawki walidacji metadanych i przywracania flag train modułów dotyczą poprawności, nie jakości sieci. Dodatkowy GPU smoke MCTS przeszedł na 3 korzeniach V9.9b/API16/GPU Graphs: ruchy i pełne N/Q akcji były dokładnie zgodne, max Q-delta 0, bez błędów capture/replay. To parytet, nie pomiar szybkości ani jakości sieci; n=3 i mieszane czasy nie uzasadniają tezy o przyspieszeniu. |
| Fused output — pierwszy przebieg E2E | Wynik wydawał się szybszy, ale oracle ujawnił błąd offsetu wartości przy ponownym użyciu slotu. | Pierwszy wynik throughput unieważniono; poprawiony wynik jest opisany w dokumencie potwierdzonych rezultatów. |
| Cache wyników NN | Ciepłe cache przyspieszało powtarzane stałe pozycje, ale koszt lookup niwelował oszczędność przy większym budżecie; w części testów targety się różniły. | Lokalny prototyp nie został aktywowany w RL; liczba trafień sama nie dowodzi korzyści. |
| Pełny kod bieżący vs HEAD | Na pojedynczej pozycji różnice czasu zmieniały kierunek dla budżetów 192/4096/8192; ruchy i liczby ewaluacji były identyczne. | Nie wykazano ogólnego przyspieszenia ani regresji. |

## Granice wspólne

Wyniki treningu na zamrożonym replayu nie mierzą Elo. Pojedynczy seed, arena po małej liczbie gier lub nieprzekroczona bramka nie dowodzą odpowiednio powtarzalnej korzyści, równoważności ani szkodliwości poza badaną próbą. Ekrany A2/C030/C2 oraz starsze porównania A–D są odrębnymi eksperymentami i nie należy ich łączyć w jeden ranking.
