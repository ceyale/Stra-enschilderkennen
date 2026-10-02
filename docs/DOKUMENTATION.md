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

1. **Verkleinern.** `app.js` zeichnet das Kamerabild auf ein Canvas mit der eingestellten Analysebreite (Regler „Reichweite“, 160–360 px). Das macht die Analyse schnell (ca. 5–15 ms auf üblichen Handys).
2. **Farbklassifikation (`classifyPixel`).** Jeder Pixel wird von RGB nach HSV umgerechnet (Farbton *h* in Grad, Sättigung *s*, Helligkeit *v*). HSV trennt die Farbe von der Helligkeit und ist damit robuster gegen Schatten und Sonne als RGB.
   - Rot: *h* ≤ 14° oder ≥ 345° (Rot liegt am Anfang/Ende des Farbkreises)
   - Gelb: *h* 38°–66°, *v* > 0,45
   - Blau: *h* 200°–255°
   - Voraussetzung immer: *s* ≥ `minSaturation` (Regler „Farbstrenge“) und *v* ≥ 0,22. Graue, weiße und schwarze Pixel fallen so heraus.
   - Alle Grenzen (`redHueMax`/`redHueMin`, `yellowHueMin`/`yellowHueMax`, `blueHueMin`/`blueHueMax`, `minValue`, `minValueYellow`) stehen in `CONFIG` und lassen sich dort ändern.
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
   - `solidity` < 0,65 → Dreieck oder Raute. Ist die Fläche oben **und** unten schmal (`wTop`, `wBot` < 0,35), ist es eine Raute. Sonst entscheidet, ob oben (Spitze oben) oder unten (Spitze unten) die schmalere Seite liegt; sind beide gleich breit, ist es keine spitze Form (`null`).
   - `solidity` > 0,92 → Rechteck.
   - Sonst, bei fast quadratischem Rahmen (Seitenverhältnis 0,72–1,38, also auch leicht schräg gesehene Kreise) → Achteck, wenn `wTop` **und** `wBot` > 0,6 und `solidity` > 0,8 und `fill` > 0,65, sonst Kreis.
6. **Schildtyp (`labelOf`).** Tabelle aus Abschnitt 2.
7. **Sicherheit (`confidence`).** `1 − 3 · |solidity − Idealwert|`, begrenzt auf 0…1. Das ist ein Maß für die Formtreue, **keine** statistische Wahrscheinlichkeit.
8. **Doppelte Treffer (`mergeOverlaps`).** Ein Schild ergibt manchmal zwei Farbflächen – wenn ein Mast, ein Schatten oder eine Strebe die Fläche trennt, oder wenn eine Fläche in einer anderen liegt. Treffer desselben Typs mit starker Überlappung (IoU ≥ `mergeIou`) oder Verschachtelung (≥ `mergeContain`) werden zu einem gemeinsamen Rahmen verschmolzen, damit dasselbe Schild nicht zweimal in der Liste steht. Danach wird nach Größe sortiert (größte Schilder zuerst) und auf `maxResults` begrenzt.
9. **Stabilisierung (`createTracker`).** Treffer im Video flackern. Der Tracker ordnet Treffer im nächsten Bild per Überlappung (IoU > 0,25, gleicher Typ) dem vorherigen zu, glättet den Rahmen und zeigt ein Schild erst nach 3 Treffern. Nach 3 Bildern ohne Treffer verschwindet es. Bei Einzelfotos ist der Tracker aus. `reset()` vergisst alle Spuren – nötig nach „Zurücksetzen“, einem neuen Foto oder einem Wechsel der Reichweite, weil die Rahmen in Analysepixeln liegen.

## 4. Einstellbare Werte

| Wert | Ort | Wirkung |
|---|---|---|
| Farbstrenge | Regler in der Oberfläche (`minSaturation`) | Höher: weniger Fehltreffer, aber blasse Schilder fehlen. Niedriger: bei Dämmerung oder verblichenen Schildern. |
| Farbmasken zeigen | Häkchen | Legt die erkannten Farbflächen über das Bild. Damit sieht man, warum ein Schild (nicht) erkannt wird. |
| Reichweite | Auswahlfeld (`workW`, 160/240/360 px) | Analysebreite. Größer erkennt kleinere/entferntere Schilder, kostet Rechenzeit. |
| `INTERVAL_MS` | `src/app.js` | Abstand zwischen zwei Analysen (100 ms). |
| `CONFIG` | `src/detector.js` | Alle Schwellwerte: Farbtöne, Helligkeit, Mindestgrößen, Seitenverhältnisse, Formgrenzen (`maxSolidityTriangle`, `minSolidityRect`, `minWidthNarrow`, `minWidthFlat`, `minFillFull`, `minCircleAspect`, `maxCircleAspect`), Zusammenfassen (`mergeIou`, `mergeContain`) und `maxResults`. |
| Tracker (`minHits`, `maxMiss`) | `createTracker(...)` | Stabilität gegen Flackern; `reset()` löscht alle Spuren. |

## 5. Neues Schild ergänzen

1. Eintrag in `SIGNS` (`name`, `zeichen`, `hex`, `text`, `note`).
2. Regel in `labelOf` (Farbe + Form + ggf. `fill`).
3. Testfall in `tests/detector.test.js` (Bild mit `render(...)` zeichnen).
4. `node tests/detector.test.js` ausführen.

Für eine neue Form (z. B. Sechseck) `shapeOf` erweitern und passende Kennzahlen wählen.

## 6. Test

`tests/detector.test.js` prüft 25 Fälle:

- alle **neun** Schildtypen, synthetisch gezeichnet (100 × 100 px, graue Fläche als Hintergrund),
- Bilder, die **kein** Schild sein dürfen (grau, zu klein, bildfüllend, rotes Quadrat, blaues Dreieck),
- einen schräg gesehenen runden Schild (Ellipse 1,27 : 1),
- das Zusammenfassen doppelter Farbflächen,
- den Tracker (Anzeige erst nach 3 Treffern, Vergessen nach 3 leeren Bildern, Verschmelzen, neue Spur, `reset()`).

```
node tests/detector.test.js        # Node; Rückgabewert 0 = alles bestanden
```

`tests/index.html` führt dieselben Fälle im Browser aus – ohne Node.
`tests/app.test.html` lädt `index.html` in einem Fenster und speist ein gemaltes Foto über den Datei-Knopf ein; damit ist der ganze Weg Foto → Analyse → Liste → Speichern/Zurücksetzen geprüft (braucht `python -m http.server 8000`).

Alle Tests prüfen die Logik, **nicht** die Praxistauglichkeit mit echten Kamerabildern. `detector.js` gibt dafür `mergeOverlaps` und `shapeOf` nach außen, damit die Tests diese Schritte direkt prüfen können.

## 7. Grenzen (bitte ernst nehmen)

- **Keine Zahlen/Symbole.** Ein Tempo-30-Schild wird als „Verbotszeichen“ erkannt, nicht als „30“.
- **Fehltreffer** durch rote/blaue/gelbe Gegenstände mit passender Form (Autos, Warnwesten, Plakate, blauer Himmel bei hoher Farbstärke). Dagegen helfen Regler und Masken-Ansicht.
- **Verpasste Schilder** bei Gegenlicht, Dämmerung, starker Schräglage, Verdeckung, schmutzigen oder stark verblichenen Schildern.
- **Nur leicht schräge Sicht** ist ausgelegt: Kreise werden bis zu einem Seitenverhältnis von ≈ 1,4 noch als Ellipse erkannt, Rechtecke bis ≈ 2,2. Stärkere Schräglage, gedrehte oder verwackelte Schilder fallen durch.
- **Verschmolzene Rahmen sind eine Hülle:** nach `mergeOverlaps` umschließt der Rahmen alle Teile und kann deshalb größer sein als das Schild selbst.
- Die **Sicherheit** in Prozent ist ein Maß für die Formtreue, keine statistische Wahrscheinlichkeit.
- **Nur deutsche Schilder.** Andere Länder nutzen teils andere Farben/Formen.
- **Kein Ersatz für Aufmerksamkeit.** Nicht als Fahrassistenz im Straßenverkehr verwenden. Nicht während der Fahrt bedienen.
- Der **Kamerazugriff** verlangt HTTPS oder `localhost`. Bilder werden nicht hochgeladen, alles läuft lokal im Browser.

## 8. Ausbaustufen

1. **Zahlen lesen** (Tempolimit): Innenfläche eines Verbotskreises ausschneiden und mit Tesseract.js oder einem kleinen Ziffern-Modell erkennen.
2. **Echtes ML-Modell:** Ein auf dem GTSRB-Datensatz trainiertes Netz (TensorFlow.js oder ONNX Runtime Web) klassifiziert die ausgeschnittenen Flächen. Die Farbsuche bleibt als schneller Vorfilter.
3. **Perspektive:** Kanten/Ecken über Konturnäherung bestimmen statt Breitenprofil; die aktuelle Lösung deckt über `minCircleAspect`/`maxCircleAspect` nur leicht schräge Sicht ab.
4. **Verlauf:** erkannte Schilder mit Zeitstempel speichern.
