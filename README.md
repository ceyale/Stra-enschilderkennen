# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera – direkt im Browser,
ohne Server, ohne Installation.

Standardmäßig arbeitet eine **Heuristik ohne jede Bibliothek** (Farbe + Form, `src/detector.js`).
Optional rechnet ein **kleines trainiertes Netz** (CNN mit Transformer-Stufen, ONNX Runtime
Web, `src/model.js`), das Position und Art in einem Durchlauf findet – siehe `docs/TRAINING.md`.
Ist kein Modell vorhanden oder die Laufzeit nicht verfügbar, bleibt die Heuristik aktiv.

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

## KI-Modus (optional)

Ein kleines Netz sagt **Position und Art** in einem Durchlauf voraus (anchor-free, drei
Stufen, 9 Typen) – trainiert mit `tools/`, ausgeliefert als ONNX und im Browser gerechnet.
Einzelne tiefe Stufen sind **Transformer-Blöcke** statt CNN-Schichten; mehr dazu und alle
gemessenen Zahlen in [`docs/TRAINING.md`](docs/TRAINING.md).

```
python tools/synth_data.py --out data/synth --n 400     # synthetische Daten (kein Datensatz nötig)
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8
python tools/export_onnx.py --ckpt models/signs-det.pt --int8 --calib-n 200
```

Danach liegen `models/signs-det.onnx`, `models/labels.json` und `models/manifest.json`
bereit und die App nutzt den Modellpfad. Fehlen sie, läuft alles wie vorher (Heuristik) –
das steht dann im Status unter dem Titel.

## Hosting auf Cloudflare Pages

Cloudflare liefert `<projektname>.pages.dev` aus – für dieses Projekt ist der Name
**`schilderscanner-r7k4m2`** vorgesehen, die Adresse wäre also
`https://schilderscanner-r7k4m2.pages.dev`.

```
python tools/build_site.py                 # baut dist/ (nur index.html, src/, models/)
python tools/build_site.py --deploy        # lädt dist/ hoch (braucht Cloudflare-Zugang)
```

Für `--deploy` müssen zwei Umgebungsvariablen gesetzt sein:

| Variable | Woher |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Dashboard → My Profile → API Tokens → Create Token (Recht „Cloudflare Pages: Edit") |
| `CLOUDFLARE_ACCOUNT_ID` | Dashboard → Übersicht, rechte Spalte |

Fehlen sie, bricht das Skript **vor** dem Upload mit einer Meldung ab – es wird nichts
hochgeladen und nichts halb angelegt. Ohne `npx`/`wrangler` auf dem Rechner geht es
alternativ über das Cloudflare-Dashboard: *Workers & Pages → Create → Pages → Upload
assets* und den Inhalt von `dist/` hineinziehen.

Zwei Dinge, die man beim Hosten kennen sollte:

* **`dist/` ist Absicht:** Cloudflare lädt den ganzen Ausgabeordner hoch. `tools/`,
  `data/` und `.git/` haben im Netz nichts zu suchen.
* **Kamera braucht HTTPS** – `pages.dev` liefert HTTPS, damit funktioniert
  `getUserMedia` auch am Handy (im WLAN über die lokale IP nicht).

## Projektstruktur

| Datei | Inhalt |
|---|---|
| `index.html` | Seitengerüst |
| `src/detector.js` | Heuristik-Erkennung (ohne Browser-Abhängigkeit, auch in Node nutzbar) |
| `src/model.js` | KI-Modus: Letterbox, Dekodierung, NMS, ONNX Runtime Web (Mathematik Node-testbar) |
| `src/app.js` | Kamera, Foto-Import, Anzeige, Wahl zwischen Modell und Heuristik |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Test der Heuristik mit synthetisch gezeichneten Schildern |
| `tests/model.test.js` | Test der Modell-Mathematik inkl. Gegenprobe gegen die Python-Seite |
| `tests/fixtures/` | Fixture mit echten ONNX-Ausgaben für diese Gegenprobe |
| `tools/` | Python-Werkzeuge: Modell, Training, Export, Daten (nicht ausgeliefert) |
| `models/` | Zielort der Modell-Dateien (lokal erzeugt, nicht im Git) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |
| `docs/TRAINING.md` | KI-Modus: Architektur, gemessene Kosten, Training, Export, Browser |

Lizenz: MIT.
