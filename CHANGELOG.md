# Changelog

## 0.4.0 – 2026-09-30
- **Fehlerzerlegung vor dem Umbau (neu: `tools/eval_conditions.py --diagnose`)**: von 656
  verpassten Boxen waren 334 „tief verpasst“ (46 % davon ≤ 32 px Diagonale), 212 falsch
  klassifiziert und 110 falsch lokalisiert; von 226 Fehlalarmen lagen 195 auf echten
  Schildern und nur 31 auf freier Fläche. Die Treffer hatten schon IoU 0,938. Damit war
  klar, wo die Arbeit hingeht – und wo nicht (ein CIoU-Verlust wäre wirkungslos, weil die
  Treffer schon sitzen; eine Klassen-Arbitrierung nach der NMS brachte gemessen nur
  +0,7 % Precision).
- **Vier Erkennungsstufen statt drei** (`stride 4/8/16/32`, `tools/detmath.py`).
  Zusätzlich geändert: die Stufe wird nach der **längsten Objektseite** gewählt
  (`ASSIGN_MAX_SIDE`) statt nach `log2` der Diagonale – die alte Regel schickte ein
  30-px-Schild auf `stride 32` mit nur 10×10 Zellen. Und bei Randlage lernt die
  **Nachbarzelle** mit (`dual`), weil der Offset vorher auf 0,999 festgeklemmt war.
  Kosten: 731 → 878 MFLOPs.
- **Getrennter Kopf (Box/Objektivität gegen Art)** in `tools/hybrid_net.py`: der Kopf
  verwechselte Arten (rotes Dreieck Spitze oben gegen unten 32×). Kostet ~4 MFLOPs.
- **Daten: Klassenausgleich, Roll-Politik, Negative.**
  `tools/gtsrb_dataset.py --balance` (jeder Typ ~3 600 Boxen statt `verbot` 50 %),
  `roll_angle()` dreht **Dreiecke nur ±35°** (stark gedrehte Dreiecke sind widersprüchliche
  Ziele: `warnung` verlor dadurch 18 Recall-Punkte), neu
  **`tools/synth_negatives.py`** für Bilder ohne Schild (3000 train + 500 im neuen Split
  `neg` als Messlatte) und mehr Hintergrundarten.
- **Mehrskaliges Training** (`--zoom`, Bereich 0,7–1,5) und `--obj-norm` in
  `tools/train_det.py`; `--val-split neg` misst während des Trainings Fehlalarme.
- **Ergebnis (2 000 val-Bilder)**: P 0,891 → **0,883**, R 0,739 → **0,796**,
  F1 0,808 → **0,837**; verpasste Boxen 656 → **513**, Fehlalarme 226 → 264. Der Zugewinn
  liegt bei den Typen mit wenigen Beispielen (`stop` R 0,478 → 0,717, `vorfahrtGewaehren`
  0,434 → 0,689, `einfahrtVerboten` 0,532 → 0,806), der Preis bei `verbot` (0,801 → 0,779).
  Alle Tabellen in `docs/TRAINING.md` §5 und §10.
- **Neu: Größe wählbar** – `tools/export_onnx.py --dynamic` (eine Datei für 256/320/384/448 px),
  `src/model.js` nimmt die Größe entgegen, die App bekommt ein Auswahlfeld
  (automatisch: Video 320 px, Foto 384 px). Gemessen: 256 px F1 0,796 / 320 px 0,837 /
  **384 px 0,854 bei 21 % weniger Fehlalarmen**.
- **`models/labels.json`** trägt jetzt `dynamic`; `tools/export_onnx.py --size` erlaubt
  zusätzlich einen statischen Export in einer festen Größe (auf x86 rund 59 % schneller
  als mit offenen Achsen).
- **Checkpoints der Vorfassung sind nicht mehr ladbar** (Kopfzweige umbenannt).

## 0.3.0 – 2026-09-30
- **Erstes echtes Modell trainiert** (bisher gab es nur die Kette dafür): `--preset balanced`
  (1 287 066 Params, 731 MFLOPs) in drei Blöcken über 159 Epochen auf **23 000 Bildern**
  aus GTSRB-Kompositionen plus 3 000 synthetischen Bildern. Gemessen auf **2 000
  Validierungsbildern: P = 0,891 / R = 0,739 / F1 = 0,808**; Export 5,16 MB fp32,
  Parität 1,1 · 10⁻⁴ mit 4/4 Erkennungen IoU ≥ 0,95, ONNX Runtime CPU (x86, 1 Thread,
  320×320) 7,2 ms je Bild. Alle Zahlen und die Schwachstellen in `docs/TRAINING.md` §5.
- **GTSRB-Lücke geschlossen:** GTSRB enthält keine blauen Hinweiszeichen (Z 3xx) und keine
  gelbe Ortstafel (Z 310) – zwei der neun Typen hatten null Trainingsdaten. Neues
  `tools/synth_missing.py` füllt genau diese Klassen synthetisch nach; gemessen danach
  `hinweis` P/R = 0,840/0,781 und `ortstafel` 0,971/0,745 (vorher: nie vorhergesagt).
- **Neu: `tools/eval_conditions.py`** – Precision/Recall je Bedingung (aus
  `manifest.conditions`) und je Klasse, mit `--json` für die Ablage. Ersetzt den als offen
  markierten Punkt „Auswertung nach Bedingung" und macht Verbesserungen messbar.
- **`tools/train_det.py`:** `--device auto|cpu|cuda` (CUDA wird erkannt, Checkpoints bleiben
  portabel), `--eval-every` wirkt jetzt wirklich (war ein toter Schalter – jede Epoche
  kostete eine volle Auswertung), Geraetewechsel beim `--resume` berücksichtigt.
- **`tools/gtsrb_dataset.py`:** `--gt-zip` für das Test-Set (Bilder und Labels liegen dort in
  **zwei** Archiven; die labelose `GT-final_test.test.csv` im Bildarchiv lieferte vorher
  stillschweigend null Zeilen), `--size`-Default 320 (Modellgröße), Boxen werden korrekt
  durch die Letterbox gerechnet, wenn Datensatzgröße ≠ Trainingsgröße.
- **Doku:** README (KI-Modus mit echtem Rezept und Messwerten), `docs/TRAINING.md`
  (§4.3 Datenweg, §5.1 Trainingsläufe, §5.2 Bedingungs-/Klassentabelle, §6 Parität/int8,
  §8 Status, §9 nächste Schritte), `models/README.md`, Changelog.
- **int8 bleibt verworfen** – jetzt auch mit 200 echten Kalibrierbildern gemessen
  (Abweichung 3,16 · 10¹). Das Qualitätstor greift, ausgeliefert wird fp32.

## 0.2.0 – 2026-09-30
- **KI-Modus (optional):** hybrides Netz mit **CNN-Backbone und Transformer-Stufen**
  (`tools/hybrid_net.py`) sagt **Position und Art** in einem Durchlauf voraus – anchor-free,
  drei Stufen (stride 8/16/32), 9 Typen, 14 Kanäle je Zelle. Globale Attention nur auf
  stride 32, lokale auf stride 16; die feinen Stufen bleiben CNN (Begründung + Messwerte
  in `docs/TRAINING.md`).
- **Werkzeuge:** `tools/synth_data.py` (synthetische Schilder mit genau den Bedingungen, die
  die Heuristik brechen), `tools/train_det.py` (Training mit `--resume`, Precision/Recall bei
  IoU 0,5), `tools/detmath.py` (Letterbox, Zuweisung, Decode, NMS – Vorlage für JS),
  `tools/bench_model.py` (Parameter/FLOPs/Latenz je Variante), `tools/selfcheck.py`
  (Überfittet das Netz ein Bild?), `tools/export_onnx.py` (ONNX + **Paritätscheck** +
  int8 mit Qualitätstor), `tools/export_fixture.py` (Testfixture aus echtem Modelllauf).
- **Browser:** `src/model.js` mit Letterbox, Decodierung, NMS und ONNX Runtime Web
  (WebGPU mit WASM-Rückfall, einthreadig wegen fehlender COOP/COEP-Header auf GitHub Pages).
  `src/app.js` wählt automatisch Modell oder Heuristik und zeigt den Modus im Status.
- **Tests:** `tests/model.test.js` prüft die Modell-Mathematik und vergleicht sie über
  `tests/fixtures/model-out.json` mit der Python-Seite (gleiche Klassen, Boxen, Scores).
- **Grenzen ehrlich dokumentiert:** int8 wurde mit wenigen Kalibrierbildern als unbrauchbar
  gemessen und wird deshalb automatisch verworfen; Laufzeit auf echten Handys und
  Erkennungsqualität auf Straßenbildern sind noch nicht gemessen.
- Doku: neue `docs/TRAINING.md`, README, `docs/DOKUMENTATION.md` (§7/§8) und `models/README.md`
  angepasst. Heuristik unverändert und weiterhin der Standard/Rückfall.

## 0.1.0 – 2026-09-30
- Erste Version: Live-Erkennung von neun Schildtypen über Farbe und Form, Foto-Import, Farbmasken-Ansicht, Stabilisierung über mehrere Bilder.

