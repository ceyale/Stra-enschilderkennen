# Datensätze für Verkehrszeichen-Erkennung

Recherche vom 02.10.2026. Warum dieses Blatt: Die Qualität hängt weniger am Netz als an den
Daten – und **nicht jeder gute Datensatz darf in einer öffentlich gehosteten App benutzt
werden**. Hier steht beides: was es gibt und was davon zu diesem Projekt passt.

## 1. Maßstab: Was dieses Projekt braucht

| Anforderung | Warum |
|---|---|
| **9 grobe Typen** statt 220 Klassen (siehe §4) | Das Netz sagt Position **und** Typ in einem Lauf; mehr Klassen kosten nur Rechenzeit |
| **Szenen mit Boxen**, nicht nur Ausschnitte | Die heutige Messlatte besteht aus GTSRB-Crops – dort steht F1 0,90, auf einem abfotografierten Poster dagegen **0,00** (`PLAN.md` §1) |
| **deutsche Zeichen**, echte Alterung | StVO-Formen (roter Kreis, gelbe Raute, Dreieck) |
| **kleine Schilder im Bild** | 15 px Zeichen in einem 4 096 px breiten Foto sind bei 320 px Eingang 1,2 px |
| **Bilder OHNE Schild** | Ohne sie erfindet das Netz Zeichen auf Postern und Produktfotos |
| **Lizenz für ein öffentliches Produkt** | Die App liegt auf Cloudflare, sie ist öffentlich |

## 2. Katalog – alles, was rechtlich und fachlich in Frage kommt

| Datensatz | Umfang | Klassen | Inhalt | Lizenz | Für uns |
|---|---|---|---|---|---|
| **GTSRB** (nutzen wir) | 51 839 Bilder | 43 | Crops, deutsch, echte Alterung | frei für Forschung | Basis der Positiven |
| **GTSDB** (deutsche Szenen) | 900 Bilder (600/300); COCO-Fassung 545 (383/108/54) | 43 | **deutsche Straßenszenen mit Boxen** | CC BY 4.0 (COCO-Fassung) | **ehrliche Messlatte** – klein, aber echt |
| **GTSIGN-220** | 75 541 Bilder, 1,86 GB | **220 (StVO-konform)** | Crops aus echten Fotos, Zahl vor `_` = Mapillary-Bild-ID | **CC BY-SA 4.0** | Klassenvielfalt, seltene Zeichen |
| **Synset Signset Germany** (Fraunhofer IOSB) | 211 000 Bilder (105 500 unabhängig), 17,6 GB | **211**, davon 43 = „GTSRB-Zwilling" | **synthetisch**: GAN-Texturen (Dreck, Kratzer), Licht/Wetter/Perspektive parametrisiert, + Segmentierung/Maske | **CC BY 4.0** | seltene und neue Zeichen (2020) ohne Fotorechte-Problem |
| **Open Images V7** | ~1,9 Mio. Bilder, Klasse „Traffic sign" | 1 | weltweit, Szenen mit Boxen | Bilder CC BY 2.0 | **harte Negative** + „da ist überhaupt etwas" |
| **RTSD** (Moskau) | 47 639/11 389 Szenen, 80 277/25 232 Zeichen | 205 | russische Szenen mit Boxen, alle Jahreszeiten | **nicht eindeutig dokumentiert** | Formen sehr ähnlich – Lizenz vorher klären |
| **TT100K** | 16 817 Szenen (6 105/7 641/3 071), 2 048², ~15 GB | 221 (chinesisch) | Szenen mit Boxen | **CC BY-NC 2.0** | nur privat/Forschung |

## 3. Was ausdrücklich NICHT geht

* **Mapillary Traffic Sign Dataset (MTSD)** – 105 000 Bilder, 400 Klassen, der größte und
  vielfältigste Detektionsdatensatz überhaupt. Die **Mapillary Object Dataset Research Use
  License** verbietet aber Derivate, die Weitergabe der Dateien und **den Einbau in ein
  öffentlich verfügbares Produkt oder einen öffentlichen Dienst**. Genau das ist unsere App →
  fachlich top, rechtlich nicht benutzbar.
* **TT100K** (CC BY-NC) und **BDD100K** (nur Forschung, außerdem nur **eine** Klasse
  „traffic sign") – beide nicht kommerziell.
* **Mapillary Vistas** – nur Segmentierung, nicht kommerziell.

**Wichtige Unterscheidung:** Die Mapillary-**Fotos** stehen unter CC BY-SA 4.0 – deshalb darf
GTSIGN-220 als Crop-Datensatz daraus existieren. Geschützt ist die **Annotation** des MTSD,
nicht das Bild darunter. (Der SA-Teil von CC BY-SA 4.0 gilt trotzdem: abgeleitete Datensätze
müssen dieselbe Lizenz tragen.)

## 4. Zuordnung auf unsere 9 Typen (StVO)

Das Muster gibt es schon: `tools/gtsrb_dataset.py` bildet die 43 GTSRB-Klassen über `CLASS_MAP`
auf die 9 Typen ab. Dieselbe Regel gilt für die StVO-Nummern:

| Unser Typ | StVO-Nummern | GTSRB-ClassIds (heute) |
|---|---|---|
| `stop` | 206 | 14 |
| `vorfahrtGewaehren` | 205, 208, 201 | 11, 13 |
| `vorfahrtstrasse` | 306 | 12 |
| `warnung` | 101–159 (Dreieck, Spitze oben) | 18–31 |
| `verbot` | 250, 251, 267, 268, 272–282 (roter Kreis) | 0–10, 15, 16 |
| `einfahrtVerboten` | 267 | 17 |
| `gebot` | 209–244 (blauer Kreis) | 33–40 |
| `hinweis` | 301–460 (Richtzeichen) | – (fehlte in GTSRB → synthetisch) |
| `ortstafel` | 310 | – (fehlte in GTSRB → synthetisch) |

„Ende von …" (GTSRB 32, 41, 42) bleibt draußen – `SKIP` im Werkzeug. Der Gewinn der großen
Datensätze liegt also **nicht** in mehr Klassen (wir wollen nur 9), sondern in **mehr
Erscheinungsformen** dieser 9: jede StVO-Nummer innerhalb eines Typs ist ein neues Aussehen
(andere Symbole, andere Zahlen, andere Größen) – genau das, was heute fehlt.

## 5. Was daraus folgt (Reihenfolge nach Wirkung)

1. **GTSDB als Szenen-Messlatte** (`tools/gtsdb_dataset.py`, Aufruf wie `gtsrb_dataset.py`):
   545–900 echte deutsche Straßenszenen mit Boxen. Klein, aber es wäre die **erste Zahl, die
   man dem Nutzer zeigen kann**, ohne sich zu schmücken. Ohne sie bleibt jede Verbesserung
   unsichtbar – heute steht dort 0,90 auf GTSRB-Crops und 0,00 auf dem Poster.
2. **GTSIGN-220 + Synset Signset Germany** für die Erscheinungsformen: 220 StVO-Klassen statt
   12 nutzbarer GTSRB-Klassen. Die Crops laufen durch die **bestehende** Szene-Komposition
   (`tools/synth_data.py`, `place_sign`) – kein neues Netz, keine neue Mathematik, nur ein
   neuer Leser für Ordner-pro-Klasse. Namensnennung: beide Datensätze verlangen sie im README;
   GTSIGN-220 zusätzlich **ShareAlike**.
3. **Open Images „Traffic sign"** (Klasse 1 von 600) für Szenenmaterial und harte Negative –
   deutlich größer als unser `neg`-Split und ohne deutsche Zeichen.
4. **Gegen die kleinen Schilder:** mit Zufallsgröße trainieren (`--zoom` gibt es) **plus**
   Kachelmodus in der App (`PLAN.md` §3.2). Die Messung zeigt: das Netz findet die Zeichen im
   Poster, sobald man es aufteilt (6×6: 53 Treffer, bester Score 0,907).
## 6. Tempo – gemessen, nicht geschätzt (`data/_speed.py`)

x86, ONNX Runtime CPU, Median aus 12 Läufen, das **ausgelieferte** Modell:

| Variante | 256 px | 320 px | 384 px | 448 px |
|---|---|---|---|---|
| offene Achsen (ausgeliefert), 1 Faden | 13,8 ms | 18,6 ms | 25,1 ms | 29,8 ms |
| feste Größe (`--size 320`), 1 Faden | – | **15,3 ms** | – | – |

Mehr Rechenfäden bei 320 px: 1 Faden 15,1 ms · 2 Fäden 13,6 ms · **4 Fäden 9,7 ms** (1,9×).

Daraus die Hebel, in dieser Reihenfolge:

1. **Mehrfäden im Browser** – `src/model.js` setzte `numThreads = 1`, weil GitHub Pages keine
   COOP/COEP-Header schicken kann. **Cloudflare kann es:** die Datei `dist/_headers` mit
   `Cross-Origin-Opener-Policy: same-origin` und `Cross-Origin-Embedder-Policy: require-corp`
   macht `crossOriginIsolated` wahr, dann nimmt die App bis zu 4 Fäden. Geprüft: jsdelivr
   liefert für `ort.min.js` und die `.wasm` sowohl `Cross-Origin-Resource-Policy: cross-origin`
   als auch `Access-Control-Allow-Origin: *` – das CDN-Skript bleibt unter `require-corp`
   ladbar. Größter Hebel, **keine** Qualitätseinbuße.
2. **Feste Eingabegröße** (18 % schneller) – dafür verliert die App die freie Wahl zwischen
   256/320/384/448 px. Mittelweg: zwei statische Modelle (320 für Video, 384 für Foto).
3. **Kleinere Eingabe** – 256 px ist 26 % schneller als 320 px, kostet aber Recall.

Alle Browserzahlen sind Übersetzungen derselben Rechnung in WebAssembly und liegen höher als
diese CPU-Werte (auf einem echten Handy waren es 122 ms je Bild, gemessen mit der Vorfassung);
die **Rangfolge** der Varianten überträgt sich.

## 7. Quellen

* GTSRB/GTSDB: `benchmark.ini.rub.de`; COCO-Fassung `huggingface.co/datasets/keremberke/german-traffic-sign-detection` (CC BY 4.0)
* GTSIGN-220: `huggingface.co/datasets/miriamcarnot/GTSIGN-220` (CC BY-SA 4.0), Carnot et al., IEEE IV 2026
* Synset Signset Germany: `synset.de/datasets/synset-signset-ger` (CC BY 4.0), Sielemann et al., ITSC 2024
* Mapillary MTSD: `research.mapillary.com/publication/eccv20d` (Ertler et al., ECCV 2020) – Research Use License
* TT100K: Zhu et al., CVPR 2016; Übersicht `docs.ultralytics.com/datasets/detect/tt100k` (CC BY-NC 2.0)
* RTSD: `graphics.cs.msu.ru/projects/traffic-sign-recognition.html` – Lizenz dort nicht ausgewiesen
* COOP/COEP für die Auslieferung: `developers.cloudflare.com/workers/static-assets/headers/`
