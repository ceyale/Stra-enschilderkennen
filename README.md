# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera – direkt im Browser,
ohne Server, ohne Installation.

Standardmäßig arbeitet eine **Heuristik ohne jede Bibliothek** (Farbe + Form, `src/detector.js`).
Optional rechnet ein **kleines trainiertes Netz** (CNN mit CATM-Stufen, ONNX Runtime
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

Ein kleines Netz sagt **Position und Art** in einem Durchlauf voraus (anchor-free, vier
Stufen, 74 Klassen) – trainiert mit `tools/`, ausgeliefert als ONNX und im Browser gerechnet.
Einzelne tiefe Stufen sind **CATM-Blöcke** (Convolutional Additive Token Mixer, CAS-ViT) statt
CNN-Schichten, die Merkmalspyramide ist eine **LGP-FPN**, der Erkennungskopf ein **TGADHead**
(aufgabengeführt und entkoppelt: TDAD + TCN) mit **hierarchischem Klassifikationskopf**
(9 Oberkategorien → 74 Unterarten); ausgeliefert wird ein **INT8**-Modell mit FP32-Rückfall.
Mehr dazu und alle gemessenen Zahlen in [`docs/TRAINING.md`](docs/TRAINING.md).

Der hierarchische Kopf ist der Grund, warum die Anzeige eine Herkunft nennen kann: jede
Erkennung kennt ihre **Familie** (`labels.json` → `hierarchy`), z. B. „Tempolimit → tempo70".
Die Klassenliste aller 74 Zeichen steht in `tools/signmap.py`.

**Aktueller Stand (gemessen, `docs/TRAINING.md` §5):** 1 300 360 Parameter, 878 MFLOPs,
5,23 MB ONNX – auf 2 000 Validierungsbildern **P = 0,883 / R = 0,796 / F1 = 0,837**,
alle neun Typen inklusive `hinweis` und `ortstafel`. Vor der Überarbeitung waren es
0,891 / 0,739 / 0,808 – der Recall ist um **5,7 Punkte** gestiegen, vor allem bei den
Typen mit wenigen Beispielen (`stop` R 0,48 → 0,72, `vorfahrtGewaehren` 0,43 → 0,69).
Die Eingabegröße ist wählbar: 256 px (F1 0,796, Sparmodus), 320 px (0,837, Video),
**384 px (0,854, Foto)** – dieselbe Modell-Datei.

Mit echten Daten (GTSRB, frei für Forschung) statt nur synthetischen:

```
curl -L -C - -o data/gtsrb/GTSRB_Final_Training_Images.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
curl -L -C - -o data/gtsrb/GTSRB_Final_Test_Images.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
curl -L -C - -o data/gtsrb/GTSRB_Final_Test_GT.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

python tools/gtsrb_dataset.py --zip data/gtsrb/GTSRB_Final_Training_Images.zip \
    --out data/det --n 20000 --size 320 --balance           # train (Typen ausgeglichen)
python tools/gtsrb_dataset.py --zip data/gtsrb/GTSRB_Final_Test_Images.zip \
    --gt-zip data/gtsrb/GTSRB_Final_Test_GT.zip \
    --out data/det --n 1800 --size 320 --split val          # val (Labels im Extra-Archiv!)
python tools/synth_missing.py --out data/det --n 3000        # hinweis + ortstafel fehlen in GTSRB
python tools/synth_missing.py --out data/det --n 200 --split val
python tools/synth_negatives.py --out data/det --n 3000 --split train   # Bilder OHNE Schild
python tools/synth_negatives.py --out data/det --n 500  --split neg     # Messlatte dafür

python tools/train_det.py --data data/det --size 320 --preset balanced \
    --batch 16 --epochs 220 --steps 100 --lr 1.5e-3 --degrade 0.6 --zoom 0.5 \
    --workers 6 --eval-every 10 --save-every 10               # --device auto erkennt CUDA
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --diagnose
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --split neg
python tools/export_onnx.py --ckpt models/signs-det.pt --dynamic   # eine Datei für 256…448 px
```

Ohne echten Datensatz geht es auch rein synthetisch (Rauchtest der ganzen Kette):

```
python tools/synth_data.py --out data/synth --n 400
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8
python tools/export_onnx.py --ckpt models/signs-det.pt --int8 --calib-n 200
```

> **Auf diesem Rechner rechnet Kaggle.** Die Windows-Anwendungssteuerung (*Smart App
> Control*) blockiert hier die PyTorch-DLLs (`WinError 4551`), deshalb läuft der Trainingslauf
> als Kaggle-Kernel: [`kaggle/README.md`](kaggle/README.md). Die Aufrufe sind dieselben wie
> oben – der Kernel baut den Datensatz, trainiert, exportiert und wertet aus; die Ergebnisse
> kommen mit `kaggle\run.ps1 -Step pull` und `-Step install` zurück nach `models/` und
> `tests/fixtures/`.

Danach liegen `models/signs-det.onnx`, `models/labels.json` und `models/manifest.json`
bereit und die App nutzt den Modellpfad. Fehlen sie, läuft alles wie vorher (Heuristik) –
das steht dann im Status unter dem Titel. int8 wird vom Qualitätstor **verworfen**
(siehe `docs/TRAINING.md` §6), ausgeliefert wird fp32.

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

### Sofort-Vorschau mit zufälliger Adresse (ohne Konto)

Wer nur schnell eine Adresse braucht, lässt Wrangler einen **temporären Zugang** anlegen und
die Seite als Worker mit statischen Dateien ausliefern:

```
python tools/build_site.py                  # dist/ bauen (muss das aktuelle Modell enthalten)
npx --yes wrangler@4.146.0 deploy --assets dist --name schilderscanner-<zufall> `
    --compatibility-date 2026-10-02 --temporary
```

Der Name wird zufällig gewählt, die Adresse ergibt sich als
`https://<name>.<konto>.workers.dev` – am 02.10.2026 gemessen:
`https://schilderscanner-hdtjal.tartan-twist.workers.dev` (Deploy 17 s nach 13 s Upload).

Vier Dinge, die dabei aufgefallen sind (gemessen, nicht vermutet):

* Wrangler verlangt **innerhalb von 60 Minuten** einen Klick auf die Claim-URL aus dem
  Protokoll. Danach ist der Zugang weg und die Adresse mit ihm – dauerhaft ist es nur mit
  eigenem Konto (`--deploy`, siehe oben).
* `urllib` (Python) bekommt von der Adresse **403** – der Standard-User-Agent wird
  abgewiesen, `curl` und Browser kommen durch. Für Skripte einen User-Agent mitschicken.
* `/index.html` antwortet mit **307** auf `/`; die Startseite liefert `/`.
* Die Adresse ist nur so gut wie `dist/`: Fehlt `models/signs-det.onnx`, läuft die Seite im
  Heuristik-Modus (der Status unter dem Titel sagt es).

Drei Dinge, die man beim Hosten kennen sollte:

* **`dist/` ist Absicht:** Cloudflare lädt den ganzen Ausgabeordner hoch. `tools/`,
  `data/` und `.git/` haben im Netz nichts zu suchen.
* **`_headers` gehört dazu:** Die Datei setzt `Cross-Origin-Opener-Policy` und
  `Cross-Origin-Embedder-Policy`. Erst dann meldet der Browser `crossOriginIsolated` und
  `src/model.js` darf mehrere Rechenfäden benutzen – gemessen 18,6 ms (1 Faden) gegen
  9,7 ms (4 Fäden) je Bild (`docs/DATENSAETZE.md` §6). Auf GitHub Pages fehlen diese
  Header, dort läuft dieselbe Rechnung mit einem Faden weiter.
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
| `tools/` | Python-Werkzeuge: Modell, Training (`train_det.py`), Klassen (`signmap.py`), Daten (`crops_dataset.py`, `gtsdb_dataset.py`, `gtsrb_dataset.py`, `synth_data.py`), Auswertung (`eval_conditions.py`), Export, Doku-Hilfen (nicht ausgeliefert) |
| `kaggle/` | Trainingslauf auf Kaggle (Datensatz bauen + trainieren + exportieren), siehe [`kaggle/README.md`](kaggle/README.md) – nötig, weil die Windows-Anwendungssteuerung PyTorch lokal blockiert |
| `models/` | Zielort der Modell-Dateien (lokal erzeugt, nicht im Git) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |
| `docs/TRAINING.md` | KI-Modus: Architektur, gemessene Kosten, Training, Export, Browser |
| `docs/DATENSAETZE.md` | Datensatz-Recherche: Umfang, **Lizenz**, StVO-Zuordnung auf die 74 Klassen, was eingebaut wurde, Tempo-Messung |
| `data/` | Rohdaten und Diagnose-Werkzeuge (`_speed.py`, `_fp_messung.py`, `_dubletten.py`, `_kaggle_probe.py`) – nicht im Git, nicht ausgeliefert |

Lizenz: MIT.
