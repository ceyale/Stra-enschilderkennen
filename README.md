# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera – direkt im Browser, ohne Server, ohne Installation, ohne Bibliotheken.

Erkannt werden Schilder an **Farbe + Form**: unter anderem Stopp, Vorfahrt gewähren, Gefahrzeichen, Verbotszeichen, Einfahrt verboten, Gebotszeichen, Hinweiszeichen, Vorfahrtstraße, Ortstafel und grüne Wegweiser. Ziffern und Großbuchstaben im Schildinneren werden zusätzlich offline ausgelesen und neben dem Erkennungskasten angezeigt. Die Zeichenerkennung ist bewusst zurückhaltend; bei kleinen, schrägen, verdeckten oder unscharfen Schildern kann Text fehlen oder falsch gelesen werden. Für verlässliche Ergebnisse Schild möglichst frontal und groß aufnehmen.

## Schnellstart

**Auf dem Handy (GitHub Pages):**
1. Alle Dateien in ein GitHub-Repository hochladen (Ordnerstruktur beibehalten).
2. *Settings → Pages → Deploy from a branch → `main` / `(root)`* wählen.
3. Die Seite `https://<name>.github.io/<repo>/` am Handy öffnen, „Kamera starten“ tippen und Zugriff erlauben.

**Auf dem PC zum Testen:**
```
python -m http.server 8000
```
Dann `http://localhost:8000` öffnen. `localhost` gilt als sicher, die Webcam funktioniert. Über die WLAN-IP am Handy geht die Kamera **nicht**, dafür braucht der Browser HTTPS.

**Ohne Kamera:** „Foto wählen“ analysiert ein einzelnes Bild.

## Bedienung

| Knopf / Regler | Wirkung |
|---|---|
| Kamera starten / stoppen | Live-Erkennung. Die Kamera läuft nur, solange der Knopf „Kamera stoppen“ heißt. |
| Foto wählen | Ein einzelnes Bild analysieren (ohne Stabilisierung, jeder Treffer zählt sofort). Handyfotos werden anhand ihrer EXIF-Drehung aufrecht analysiert. |
| Bild speichern | Das angezeigte Bild samt Rahmen und Sicherheit als PNG sichern. |
| Zurücksetzen | Kamera aus, Foto weg, Trefferliste leer. |
| Farbstrenge | Höher = strenger: weniger Fehltreffer, aber blasse Schilder fehlen. |
| Reichweite | Analysebreite (`nah` 160 px / `mittel` 240 px / `fern` 360 px). Größer erkennt kleinere und weiter entfernte Schilder, kostet aber Rechenzeit. |
| Farbmasken zeigen | Blendet ein, welche Pixel als rot, blau oder gelb gezählt wurden – damit sieht man, warum ein Schild (nicht) erkannt wird. |

## Tests

```
node tests/detector.test.js
```
Ohne Node: `tests/index.html` im Browser öffnen – derselbe Test, nur mit Ergebnis im Fenster.
Der Oberflächen-Test braucht einen lokalen Server (Dateien dürfen sonst nicht auf das Testfenster zugreifen):
```
python -m http.server 8000
```
→ `http://localhost:8000/tests/app.test.html`

## Projektstruktur

| Datei | Inhalt |
|---|---|
| `index.html` | Seitengerüst |
| `src/detector.js` | Erkennung (ohne Browser-Abhängigkeit, auch in Node nutzbar) |
| `src/app.js` | Kamera, Foto-Import, Anzeige |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Tests mit synthetisch gezeichneten Schildern (`node tests/detector.test.js`) |
| `tests/index.html` | Dieselben Tests im Browser, ohne Node |
| `tests/app.test.html` | Oberflächen-Test: speist ein gemaltes Foto in die Seite ein (braucht den lokalen Server) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |

Lizenz: MIT.
