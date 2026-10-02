# PLAN – Qualität auf echten Fotos (Stand: Block 5, nicht abgeschlossen)

Kurzfassung des aktuellen Umbaus. Gemessene Zahlen stehen in
[`TRAINING.md`](TRAINING.md) §5/§10; dieses Blatt ist die Arbeitsliste.

## 1. Was gemessen wurde (Fotos des Nutzers, `data/_test_bilder.py`)

| Test | Ergebnis |
|---|---|
| `Test/Nothing.jpg` (Zimmer/Monitor, **kein** Schild) | 3 Fehlalarme bei conf 0,25, Boxen **380×305 bis 1971×1811 px** (halbes Bild), Bestwerte `ortstafel` 0,36 / `einfahrtVerboten` 0,32 |
| `Test/Schilder.jpg` (Poster, ~80 Schilder) | **0 Treffer**, Bestwert 0,19 |
| derselbe Checkpoint auf `data/det` val (2 000 Bilder) | P 0,883 / R 0,796 / F1 0,837, Treffer-IoU 0,954 |

**Schluss:** Die Architektur ist nicht das Problem – es fehlt die **Domäne** (echte Fotos ohne
Schild, abfotografierte Anzeige mit Moiré) und die **Auflösung** für kleine Schilder.

### Kachelmessung (dieselben Fotos)

Die Schilder sind im 4096 px breiten Poster ~15 px groß → bei 320 px Eingang 1,2 px.

| Foto | ganzes Bild | 3×3 Kacheln | 6×6 Kacheln |
|---|---|---|---|
| `Schilder.jpg` | 0 | 28 (beste 0,51) | **115 (beste 0,88)** |
| `Nothing.jpg` | 3 FP | 3 FP | 22 FP (beste 0,60) → conf 0,3 lässt nur **5** |

⇒ Kacheln bringen die Erkennung, kosten aber Fehlalarme → braucht höhere Schwelle.

## 2. Was bereits gebaut ist

* `tools/real_negatives.py` – echte Fotos → viele Zufallsausschnitte mit **leerem Label**.
  Trennung ohne Selbstbetrug: `--train-region`/`--test-region` teilen **dasselbe** Foto
  (`Nothing.jpg`: obere Hälfte Training, untere Hälfte Messlatte); coco128 in 114 Training /
  12 gesperrte Testfotos. COCO-Bilder mit Klasse 11 (stop sign) fliegen raus.
* `data/negatives/fetch_coco.ps1` – holt coco128 (128 echte Fotos, CC BY 4.0, 7 MB).
* `tools/synth_data.py` – neue Szene `screen_panel` (Schildertafel/Anzeige: 3–34 Schilder im
  Raster, 15–160 px, Moiré/Gammaschlag/Kanalversatz/Vignette), Helfer `place_sign`
  (Boxrechnung nur an einer Stelle), neue Negativ-Arten `hart:bildschirm` und `hart:tastatur`.
* `tools/synth_missing.py --scene-prob` – Anteil Tafel-Szene.
* `tools/synth_negatives.py` – kennt die neuen Arten über `HARD_KINDS`.
* `.gitignore` – **Modell wird jetzt eingecheckt** (`models/signs-det.pt`, `*.onnx`, `*.json`).
* Datensatz `data/det` (gebaut mit `data/_build2.ps1`): **32 700 Bilder**, 7 700 Negative.
  Train 29 600 = GTSRB 20 000 + Lückenschluss 3 000 (Tafel) + synth. Negative 3 000 +
  **echte Negative 3 600**. Split `neg` 1 100. **`val` 2 000 unverändert** (Messlatte).
* `data/_test_bilder.py --tile N`, `data/_check_synth.py` (`data/_commit5.txt` entfernt).

### Messung 02.10.2026 (Zielvorgabe des Nutzers: `Schilder.jpg` alles, Negative nichts)

Werkzeuge (neu, nur Diagnose, nicht ausgeliefert): `data/_fp_messung.py` (rechnet mit der
**ausgelieferten** `models/signs-det.onnx` und der Mathematik aus `tools/detmath.py` – die
PyTorch-DLLs waren hier zeitweise blockiert, inzwischen laufen sie wieder: gemessen
`torch 2.14.0+cu126`, `cuda verfuegbar: True`, **GTX 1060 6 GB**, also ist Training auch lokal
möglich), `data/_dubletten.py` (Duplikate und Leakage), `data/_schwellen.py` (ein Modelllauf,
mehrere Schwellen).

| Bild | 320 px ganz | 3×3 Kacheln | 6×6 Kacheln | höchster Roh-Score |
|---|---|---|---|---|
| `Test/Schilder.jpg` (~80 Schilder am Monitor) | **0** | 15 | **53** (beste 0,907) | 0,166 / 0,907 |
| `Test/Nothing.jpg` | 1 | 2 | 8 | 0,259 / 0,685 |
| `negative-images/` (15 Bilder) | **35** auf 8 Bildern | 143 auf 14 | 281 auf 15 | 0,596 (ISO-7010-Poster) |

Schwellen (val 1 000 Bilder, unveränderte Messlatte; neg = 1 000 echte Negative):

| conf | F1 val | R val | FP/Bild neg | FP/Bild val |
|---|---|---|---|---|
| **0,25 (App-Stand)** | 0,841 | 0,801 | 0,282 | 0,128 |
| **0,35** | **0,844** | 0,777 | 0,133 | 0,080 |
| 0,40 | 0,843 | 0,764 | 0,096 | 0,059 |
| 0,50 | 0,830 | 0,724 | 0,027 | 0,027 |

⇒ **0,35 halbiert die Fehlalarme ohne F1-Verlust** (kostet 2,4 Punkte Recall); auf den
Nutzerfotos 35 → 11 Fehlalarme, `Nothing.jpg` 1 → 0. `Schilder.jpg` bleibt bei **jeder**
Schwelle 0 – das ist **kein** Schwellenproblem, sondern Auflösung: die Schilder sind im
4 096 px breiten Foto ~15 px groß, bei 320 px Eingang also 1,2 px. Kacheln finden sie
(6×6: 53 Treffer, bester Score 0,907), kosten aber Fehlalarme auf den Negativen.

Duplikate (32 700 Bilder): byte-gleich 4 Paare, **alle innerhalb `train`**, keine über
Splitgrenzen; inhaltlich (dhash ≤ 2 Bit) 378 Gruppen / 1 622 Bilder, darunter 339 fast leere
Negative (dhash 0) – die tragen nichts bei. Kein Testbild ähnelt einem Trainingsbild. Die
4 200 echten Negative stammen aber aus nur **127 Fotos** (je ~33 Ausschnitte): die Vielfalt
ist klein, nicht die Dateizahl.

## 3. Nächste Schritte

0. **Training läuft auf Kaggle** (`kaggle/`, siehe [`kaggle/README.md`](kaggle/README.md)): der
   Entwicklungsrechner kann nicht rechnen, weil die Windows-Anwendungssteuerung (Smart App
   Control) die PyTorch-DLLs blockiert (`WinError 4551`). Der Kernel baut den Datensatz aus
   den Rohdaten, trainiert, exportiert ONNX und misst – mit genau den Aufrufen aus
   `docs/TRAINING.md` §4.3/§5. Rückweg der Ergebnisse: `kaggle\run.ps1 -Step pull -Step install`.
   **Lauf vom 02.10. ist durch und übernommen** (Tesla T4, 70 min, 220 × 150, 29 600 Bilder):
   val **P 0,944 / R 0,865 / F1 0,903**, 384 px 0,906, Fixture und beide Node-Tests grün,
   Deploy auf Cloudflare erledigt – Zahlen in `CHANGELOG.md` 0.4.1. Offen bleibt, die
   Tabellen in `docs/TRAINING.md` §5/§10 auf diesen Lauf nachzuziehen.
1. **Trainingslauf Block 5 ist gelaufen** (02.10.). Offen dazu:
   * Datensatz aufräumen: **339 fast leere Negative** (dhash 0) und 4 byte-gleiche Paare –
     sie kosten Rechenzeit und bringen nichts.
   * Die **Val-Messlatte ist GTSRB-Material** (Crops mit großem Schild). Deshalb steht dort
     0,90, während dieselbe Datei auf echten Szenen 0,00 (Poster) liefert. Für eine ehrliche
     Zahl gehört eine Szene-Messlatte dazu (die Nutzerfotos sind der Anfang).
   * **Schwelle entscheiden:** 0,25 → **0,35** in `src/model.js` halbiert die Fehlalarme
     **ohne F1-Verlust** (Messung oben). Eine Zeile, sofort wirksam.
   * **Negative sammeln** – ja, aber nach **Art**, nicht nach Menge: Anzeigen/Poster mit
     Piktogrammen (die härteste Klasse: rote Kreise, blaue Kreise, gelbe Dreiecke sind genau
     die gesuchten Formen), Produktfotos auf Weiß, Nachtaufnahmen, Büro/Zimmer/Werkstatt.
     Rezeptur mit dem bestehenden Werkzeug (bewiesen, 6 Bilder in 2 s):
     `python tools/real_negatives.py --out data/det --n 3000 --split train --real-train <70 % der Fotos> --montage data/m.png`
     und `--n 600 --split neg --real-test <die übrigen 30 %>` – **getrennte Fotos** für
     Training und Messlatte, sonst betrügt man sich selbst (je Foto ~33 Ausschnitte).
2. **Kachelmodus in die App**: `src/model.js` + `src/app.js` – Foto in N×N überlappende
   Kacheln, Treffer verschieben, NMS über die Kachelgrenzen, **höhere Schwelle** im Kachelmodus
   (Messung: conf 0,3 → 40 statt 115 Treffer auf dem Poster, aber 5 statt 22 FP auf `Nothing`).
3. **Hard-Negative-Mining**: mit dem neuen Modell über den Negativ-Pool laufen und die
   schlimmsten Fehltreffer selbst als Trainingsbilder nachliefern (begrenzter Nutzen war
   gemessen, jetzt aber die echte Baustelle: Riesenboxen auf Anzeigen).
4. **Endabnahme**: `eval_conditions --split val` (unverändert vergleichbar) + `--split neg`
   (echte Negative!) + `--diagnose`, ONNX-Export mit dynamischen Achsen, Fixture, Node-Tests,
   Website bauen, Doku + CHANGELOG nachziehen, veröffentlichen.

## 4. Fallstricke, die nicht vergessen werden dürfen

* `val` **nicht** anfassen – er ist die einzige Zahl, die mit den früheren Läufen vergleichbar ist.
* `Schilder.jpg` **nie** trainieren (nur der untere Streifen von `Nothing.jpg` und 114 der 126
  coco-Fotos sind Trainingsmaterial).
* Kachelmodus muss in **Python und JS identisch** umgesetzt sein (sonst rechnet der Browser
  anders als die Messung) – die Regel „eine Rechnung, zwei Sprachen“ gilt auch hier.
* Alte Checkpoints sind nicht ladbar (Kopfzweige umbenannt); ein Lauf muss von Grund auf neu.
* int8 bleibt verworfen (Qualitätstor), ausgeliefert wird fp32.

## 5. Offene Fragen an den Nutzer

* **Kachelmodus im Video erlaubt?** 6×6 Kacheln = 36 Modellläufe je Bild (bei 150 ms je Lauf
  zu langsam für Live); sinnvoll wäre 2×2 oder nur der Foto-Modus.
* **Sollen falsch erkannte Typen wichtiger sein als verpasste?** Das steuert, ob wir im
  Kachelmodus die Schwelle hochziehen (weniger Fehlalarme, weniger Treffer).