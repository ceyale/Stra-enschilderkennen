# Dokumentation

## 1. Idee

Deutsche Verkehrszeichen sind durch **Farbe und Umriss** stark standardisiert. Für einfache Schilder reicht daher klassische Bildverarbeitung ohne KI-Modell:

> Farbflächen finden → Umriss vermessen → aus Farbe + Umriss den Schildtyp ableiten.

Vorteile: keine Bibliotheken, kein Modell-Download, läuft offline auf dem Handy, jeder Schritt ist nachvollziehbar. Nachteil: siehe Abschnitt 7.

## 2. Erkennbare Schilder

| Schildtyp | Farbe | Form | Zusatzkriterium |
|---|---|---|---|
| Stopp (Z 206) | rot | Achteck | Vollfläche (`fill` > 0,6) |
| Vorfahrt gewähren (Z 205) | rot | Dreieck, Spitze unten | – |
| Gefahrzeichen (Z 101 ff.) | rot | Dreieck, Spitze oben | – |
| Verbotszeichen (Z 2xx) | rot | Kreis | Ring (`fill` ≤ 0,65) |
| Einfahrt verboten (Z 267) | rot | Kreis | Vollfläche (`fill` > 0,65) |
| Gebotszeichen (Z 2xx) | blau | Kreis | – |
| Hinweiszeichen (Z 3xx) | blau | Rechteck | – |
| Vorfahrtstraße (Z 306) | gelb | Raute | – |
| Ortstafel (Z 310) | gelb | Rechteck | – |

## 3. Ablauf pro Bild

Alles in `src/detector.js`; die Funktionsnamen entsprechen den Schritten.

1. **Verkleinern.** `app.js` zeichnet das Kamerabild auf ein Canvas mit 240 Pixeln Breite. Das macht die Analyse schnell (ca. 5–15 ms auf üblichen Handys).
2. **Farbklassifikation (`classifyPixel`).** Jeder Pixel wird von RGB nach HSV umgerechnet (Farbton *h* in Grad, Sättigung *s*, Helligkeit *v*). HSV trennt die Farbe von der Helligkeit und ist damit robuster gegen Schatten und Sonne als RGB.
   - Rot: *h* ≤ 14° oder ≥ 345° (Rot liegt am Anfang/Ende des Farbkreises)
   - Gelb: *h* 38°–66°, *v* > 0,45
   - Blau: *h* 200°–255°
   - Voraussetzung immer: *s* ≥ `minSaturation` (Regler „Farbstrenge“) und *v* ≥ 0,22. Graue, weiße und schwarze Pixel fallen so heraus.
3. **Zusammenhängende Flächen (`findComponents`).** Flood-Fill mit 4er-Nachbarschaft sammelt gleichfarbige Pixel zu Flächen. Jede Fläche bekommt einen Rahmen (Bounding Box). Verworfen werden Flächen, die zu klein sind (Kantenlänge < 10 px, Fläche < 40 px), ein unpassendes Seitenverhältnis haben (außerhalb 0,5–2,2) oder fast das ganze Bild füllen (> 90 %).
4. **Umriss vermessen (`analyzeShape`).** Pro Bildzeile zählt nur der äußerste linke und rechte Pixel der Fläche. Dadurch wird ein roter *Ring* zur gefüllten *Silhouette*. Daraus entstehen vier Kennzahlen:

   | Kennzahl | Bedeutung |
   |---|---|
   | `solidity` | Silhouettenfläche ÷ Rahmenfläche |
   | `fill` | tatsächliche Farbpixel ÷ Silhouettenfläche (Ring klein, Vollfläche groß) |
   | `wTop` | mittlere Breite im oberen Viertel ÷ Rahmenbreite |
   | `wBot` | mittlere Breite im unteren Viertel ÷ Rahmenbreite |

   Idealwerte der `solidity`: Kreis π/4 ≈ 0,785 · Achteck ≈ 0,83 · Dreieck und Raute 0,5 · Rechteck 1,0.
5. **Form (`shapeOf`).**
   - `solidity` < 0,65 → Dreieck oder Raute. Ist die Fläche oben **und** unten schmal (`wTop`, `wBot` < 0,3), ist es eine Raute. Sonst entscheidet, ob oben (Spitze oben) oder unten (Spitze unten) die schmalere Seite liegt.
   - `solidity` > 0,92 → Rechteck.
   - Sonst, bei fast quadratischem Rahmen (Seitenverhältnis 0,8–1,25) → Achteck, wenn `wTop` > 0,64 und `solidity` > 0,8 und `fill` > 0,6, sonst Kreis.
6. **Schildtyp (`labelOf`).** Tabelle aus Abschnitt 2.
7. **Sicherheit (`confidence`).** `1 − 3 · |solidity − Idealwert|`, begrenzt auf 0…1. Das ist ein Maß für die Formtreue, **keine** statistische Wahrscheinlichkeit.
8. **Stabilisierung (`createTracker`).** Treffer im Video flackern. Der Tracker ordnet Treffer im nächsten Bild per Überlappung (IoU > 0,25, gleicher Typ) dem vorherigen zu, glättet den Rahmen und zeigt ein Schild erst nach 3 Treffern. Nach 3 Bildern ohne Treffer verschwindet es. Bei Einzelfotos ist der Tracker aus.

## 4. Einstellbare Werte

| Wert | Ort | Wirkung |
|---|---|---|
| Farbstrenge | Regler in der Oberfläche (`minSaturation`) | Höher: weniger Fehltreffer, aber blasse Schilder fehlen. Niedriger: bei Dämmerung oder verblichenen Schildern. |
| Farbmasken zeigen | Häkchen | Legt die erkannten Farbflächen über das Bild. Damit sieht man, warum ein Schild (nicht) erkannt wird. |
| `WORK_W` | `src/app.js` | Analysebreite. Größer erkennt kleinere/entferntere Schilder, kostet Rechenzeit. |
| `INTERVAL_MS` | `src/app.js` | Abstand zwischen zwei Analysen. |
| `CONFIG` | `src/detector.js` | Mindestgrößen, Seitenverhältnis, Helligkeit. |
| Tracker (`minHits`, `maxMiss`) | `createTracker(...)` | Stabilität gegen Reaktionszeit. |

## 5. Neues Schild ergänzen

1. Eintrag in `SIGNS` (`name`, `zeichen`, `hex`, `text`, `note`).
2. Regel in `labelOf` (Farbe + Form + ggf. `fill`).
3. Testfall in `tests/detector.test.js` (Bild mit `render(...)` zeichnen).
4. `node tests/detector.test.js` ausführen.

Für eine neue Form (z. B. Sechseck) `shapeOf` erweitern und passende Kennzahlen wählen.

## 6. Test

`tests/detector.test.js` zeichnet acht Schilder synthetisch (100 × 100 px, graue Fläche als Hintergrund) und prüft, dass genau der erwartete Typ erkannt wird. Er prüft die Logik, **nicht** die Praxistauglichkeit mit echten Kamerabildern.

## 7. Grenzen (bitte ernst nehmen)

- **Keine Zahlen/Symbole.** Ein Tempo-30-Schild wird als „Verbotszeichen“ erkannt, nicht als „30“.
- **Fehltreffer** durch rote/blaue/gelbe Gegenstände mit passender Form (Autos, Warnwesten, Plakate, blauer Himmel bei hoher Farbstärke). Dagegen helfen Regler und Masken-Ansicht.
- **Verpasste Schilder** bei Gegenlicht, Dämmerung, starker Schräglage, Verdeckung, schmutzigen oder stark verblichenen Schildern.
- **Nur frontale Sicht** ist ausgelegt. Schräg gesehene Kreise werden Ellipsen, Rechtecke Trapeze.
  Gemessen an synthetischen Bildern: schon 15–30° Roll führen zu **falschen Typen**
  (Gefahrzeichen → „Vorfahrt gewähren", Raute → „Ortstafel", Stopp → „Einfahrt verboten").
  Der optionale KI-Modus (docs/TRAINING.md) behebt genau diese Fehlerklasse, weil er
  Schräglage und Unschärfe aus den Trainingsdaten lernt.
- **Nur deutsche Schilder.** Andere Länder nutzen teils andere Farben/Formen.
- **Kein Ersatz für Aufmerksamkeit.** Nicht als Fahrassistenz im Straßenverkehr verwenden. Nicht während der Fahrt bedienen.
- Der **Kamerazugriff** verlangt HTTPS oder `localhost`. Bilder werden nicht hochgeladen, alles läuft lokal im Browser.

## 8. Ausbaustufen

1. **Zahlen lesen** (Tempolimit): Innenfläche eines Verbotskreises ausschneiden und mit Tesseract.js oder einem kleinen Ziffern-Modell erkennen – **offen**.
2. **Echtes ML-Modell:** *umgesetzt und trainiert* als hybrides Netz (`tools/`,
   optionaler Modus in `src/model.js`). Es findet **Position und Art in einem Durchlauf**
   (anchor-free, 9 Typen, drei Stufen) statt nur ausgeschnittene Flächen zu klassifizieren
   – der Vorfilter ist damit nicht mehr Voraussetzung. Einzelne tiefe Stufen sind
   Transformer-Blöcke. Trainiert wurde es auf GTSRB-Kompositionen (23 000 Bilder) plus
   synthetischen Bildern für die zwei Typen, die GTSRB nicht enthält; gemessen
   **P = 0,891 / R = 0,739 / F1 = 0,808** auf 2 000 Validierungsbildern. Kosten, Training,
   Auswertung je Bedingung und alle Messwerte stehen in [`TRAINING.md`](TRAINING.md).
   Die Heuristik bleibt als Rückfall erhalten.
3. **Perspektive:** Kanten/Ecken über Konturnäherung bestimmen statt Breitenprofil – **entfällt im
   Modellmodus** (das Netz sieht Schräglage und Roll aus den Trainingsdaten), bleibt aber offen
   für den Heuristik-Pfad.
4. **Verlauf:** erkannte Schilder mit Zeitstempel speichern – **offen**.

