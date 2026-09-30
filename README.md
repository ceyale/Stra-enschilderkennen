# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera – direkt im Browser, ohne Server, ohne Installation, ohne Bibliotheken.

Erkannt werden Schilder an **Farbe + Form**: Stopp, Vorfahrt gewähren, Gefahrzeichen, Verbotszeichen, Einfahrt verboten, Gebotszeichen, Hinweiszeichen, Vorfahrtstraße, Ortstafel. Zahlen und Symbole (z. B. „30“) werden **nicht** gelesen.

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

## Projektstruktur

| Datei | Inhalt |
|---|---|
| `index.html` | Seitengerüst |
| `src/detector.js` | Erkennung (ohne Browser-Abhängigkeit, auch in Node nutzbar) |
| `src/app.js` | Kamera, Foto-Import, Anzeige |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Test mit synthetisch gezeichneten Schildern (`node tests/detector.test.js`) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |

Lizenz: MIT.
