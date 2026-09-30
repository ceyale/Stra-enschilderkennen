# Changelog

## 0.2.0 – 2026-09-30
- Erkennung: doppelte Farbflächen desselben Schildtyps werden zusammengefasst (`mergeOverlaps`), Treffer nach Größe sortiert und auf sechs begrenzt.
- Erkennung: schräg gesehene runde Schilder (Ellipsen) werden über einen weiteren Seitenverhältnisbereich erkannt; Achteck-Erkennung prüft jetzt Ober- **und** Unterkante.
- Erkennung: alle Schwellwerte (Farbtöne, Formen, Mindestgrößen) liegen in `CONFIG`; `analyzeShape` fällt bei leerer Silhouette nicht mehr auf Division durch null herein.
- Tracker: `reset()` ergänzt.
- Oberfläche: „Bild speichern“ (PNG) und „Zurücksetzen“; Reichweite der Analyse wählbar (160/240/360 px); Statuszeile mit Analysezeit und Bildrate; Trefferliste mit Anzahl und Sicherheit; Farbmasken werden ohne Müll pro Pixel gezeichnet.
- Foto-Import: EXIF-Drehung von Handyfotos wird berücksichtigt, dasselbe Foto lässt sich erneut wählen.
- Tests: alle neun Schildtypen (Ortstafel fehlte), Negativ-, Ellipsen-, Merge- und Tracker-Fälle; zusätzlich `tests/index.html` (Browser) und `tests/app.test.html` (Oberfläche).

## 0.1.0 – 2026-09-30
- Erste Version: Live-Erkennung von neun Schildtypen über Farbe und Form, Foto-Import, Farbmasken-Ansicht, Stabilisierung über mehrere Bilder.
