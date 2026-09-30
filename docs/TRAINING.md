# Training und KI-Modus

Dieses Dokument beschreibt den optionalen KI-Modus: ein kleines hybrides Netz
(CNN + Transformer-Stufen), das in **einem** Durchlauf die **Position** (Box) und die
**Art** (9 Schildtypen) vorhersagt. Trainiert wird mit Python (Werkzeuge in `tools/`),
ausgeliefert wird nur `models/signs-det.onnx` und im Browser gerechnet
(`onnxruntime-web`, siehe `src/model.js`).

Die Heuristik in `src/detector.js` bleibt vollständig erhalten und ist der **Rückfall**:
Fehlt das Modell oder kann die Laufzeit es nicht laden, arbeitet die App wie bisher.

## 1. Architektur

`tools/hybrid_net.py`, Klasse `HybridNano`. Eingang fest **1 × 3 × 320 × 320**
(Letterbox, Grau 114), Ausgang **drei** Tensoren `os8`, `os16`, `os32`.

| Teil | Aufbau |
|---|---|
| Stem | 3×3 Conv stride 2 → 16 Kanäle (auf /2) |
| Stufe /4 | Inverted-Residual-Block (MobileNet-Stil), 32 Kanäle |
| Stufe /8 → `p3` | 2 × IR-Block, 64 Kanäle |
| Stufe /16 → `p4` | 2 × IR-Block, 128 Kanäle |
| Stufe /32 → `p5` | 2 × IR-Block, 256 Kanäle |
| **Transformer** | `p5` global (10×10 = 100 Tokens), optional `p4` (20×20 = 400) lokal/windowed oder global |
| Fusion | FPN-lite: 1×1 lateral + Nearest-Upsample + Add + 3×3 Fuse |
| Köpfe | je Stufe: 3×3 Depthwise → 1×1 → 1×1 auf **14 Kanäle** |

Kopf-Layout je Zelle: `[tx, ty, tw, th, obj, cls0..cls8]`, Dekodierung (Python wie JS):

```
cx = (gx + sigmoid(tx)) * stride        cy = (gy + sigmoid(ty)) * stride
w  = exp(clip(tw, ±8)) * stride         h  = exp(clip(th, ±8)) * stride
score = sigmoid(obj) * max_j sigmoid(cls_j)
```

**Warum Transformer nur in den tiefen Stufen?** Attention kostet O(N²) in der Tokenzahl
N = (H/stride)·(W/stride). Bei 320×320 ergibt das:

| Stufe | Raster | Tokens | Nachbarschaft bei globaler Attention |
|---|---|---|---|
| /8 (`p3`) | 40×40 | 1600 | 2 560 000 Paare → unbezahlbar |
| /16 (`p4`) | 20×20 | 400 | 160 000 Paare → nur lokal sinnvoll |
| /32 (`p5`) | 10×10 | 100 | 10 000 Paare → praktisch gratis |

Deshalb: **globaler** Block nur auf `p5` (dort kann das Netz Beziehungen über das ganze
Bild sehen – Schild + Mast + Umgebung, Verdecker einordnen), **lokale** (windowed)
Blöcke mittig, und die feinen Stufen bleiben CNN (dort steckt die Detailinformation für
kleine Schilder, und eine Attention wäre dort am teuersten).

Der Block selbst (`HybridEncoder`) ist MobileViT-artig und exportfreundlich:
Pre-Norm, MHSA per MatMul/Softmax (kein Sonderop), **CPE** = Depthwise-Conv als
Positionsersatz (damit funktioniert derselbe Block bei jeder Auflösung),
FFN als 1×1 → Depthwise 3×3 → 1×1, LayerScale (γ = 0,01) und Restverbindungen.

## 2. Gemessene Kosten je Variante

`python tools/bench_model.py --iters 3` auf dieser Maschine (CPU, PyTorch 2.13, 320×320).
Params und MFLOPs sind exakt, die Millisekunden streuen auf dieser CPU um bis zu ±30 %
(Auslastung), sie sind eine Größenordnung, kein Messprotokoll.

| Variante | Params | int8-Datei | MFLOPs | ms (4 Threads) | ms (1 Thread) |
|---|---|---|---|---|---|
| cnn-only | 750 k | ~0,75 MB | **615** | 16–25 | 22–27 |
| **p5-global** (Standard) | 1 287 k | ~1,29 MB | **731** | 22–28 | 25–29 |
| p5g + p4 windowed | 1 424 k | ~1,42 MB | 844 | 34 | 31 |
| p4 global + p5 global | 1 424 k | ~1,42 MB | 921 | 24 | 29 |
| p5g + p4w, TokenNorm=LayerNorm | 1 424 k | ~1,42 MB | 844 | 25 | 33 |
| p5g + p4w + p3w | 1 460 k | ~1,46 MB | 965 | 44 | 43 |
| p5g + Hardswish | 1 287 k | ~1,29 MB | 731 | 21 | 22 |

Ablesbare Erkenntnisse:

* Der **`p5`-Transformer kostet ~19 % Mehrrechnung** (615 → 731 MFLOPs) und liegt bei
  den Parametern in derselben Größenordnung wie ein „Nano"-Detektor (1,3 M).
* **Windowed ist in FLOPs billiger, aber auf dieser CPU nicht schneller** — die
  Fenster-Reshapes kosten mehr als die reine Attention (844 MFLOPs, aber 34 ms statt 24 ms
  für `p4` global). Auf WebGPU/WASM neu messen, bevor man sich festlegt.
* **`p3` windowed ist der Kostentreiber** (1600 Tokens): +43 % Rechenzeit für den
  kleinsten erwartbaren Nutzen → nicht verwenden.
* **Hardswish** spart hier ~6 % Zeit bei identischen FLOPs (auf ARM meist mehr).
* Auflösung schlägt alles andere: 192² statt 320² ist rund **⅓ der Rechnung**
  (Attention-Skalierung: 192 px → 36 Tokens auf p5, 320 px → 100, 416 px → 169).

## 3. Echtzeit-Budget

Gemessene Referenz: **ONNX Runtime CPU (x86, 1 Thread), 128×128: 1,1–1,5 ms je Bild**
(`tools/export_onnx.py --bench`). Das ist nicht die Handy-Zahl, sondern zeigt nur, dass
der Graph schlank ist. Für den Browser gilt:

| Weg | Erwartung | Maßnahme |
|---|---|---|
| WebGPU (Chrome/Android 12+, Safari/iOS 26+) | deutlich unter 10 ms | 320 px fahren, `enableGraphCapture` möglich |
| WASM einthreadig (GitHub Pages, kein COOP/COEP) | Größenordnung 10² ms | Eingabe auf 192–224 px, Analyse alle 200 ms |
| kein WebGPU, Modell fehlt, offline | – | Heuristik (5–15 ms bei 240 px) bleibt aktiv |

Wichtig, gemessen in der Doku von ONNX Runtime Web: **Mehrthread-WASM gibt es nur mit
`crossOriginIsolated`** (COOP/COEP-Header), und GitHub Pages erlaubt keine eigenen
Header. `src/model.js` setzt deshalb `ort.env.wasm.numThreads = 1`. Zusätzlich:
`ort.env.wasm.proxy` (Worker) **lässt sich nicht mit WebGPU kombinieren** — wir nehmen
WebGPU, sonst einthreadiges WASM.

## 4. Daten

### 4.1 Format (gilt für echte und synthetische Daten)

```
data/det/images/<id>.jpg            # Vollbilder
data/det/labels/<id>.txt            # YOLO: "<klasse> <cx> <cy> <w> <h>", alles 0..1
data/det/manifest.json              # Herkunft, Lizenz, Szene, Bedingungen, Split
```

`manifest.json` je Eintrag:
`{id, src, license, scene, conditions[], split, width, height, boxes}`.
Split **nach Szene/Fahrt**, nie nach Einzelbild – sonst stehen dieselben Laternen in
Training und Validierung und die Zahlen sind wertlos.

Klassenreihenfolge = die Keys aus `SIGNS` in `src/detector.js`:
`stop, vorfahrtGewaehren, warnung, verbot, einfahrtVerboten, gebot, hinweis,
vorfahrtstrasse, ortstafel` (+ implizit „kein Schild" über `obj`).
Damit sind die `label`-Werte im Modellpfad dieselben Strings wie in der Heuristik –
`createTracker`, `drawBoxes`, `renderList` und `SIGNS` bleiben unverändert.

### 4.2 Fremddaten (Lizenzen vor Nutzung prüfen!)

| Datensatz | Umfang | Nutzen | Lizenz |
|---|---|---|---|
| GTSRB | 39 209 + 12 630 Crops, 43 Klassen | Klassifikator-Vorwärmung | frei für Forschung |
| GTSDB | 900 Bilder, 852 Schilder, Boxen | Tag/frontal-Grundrauschen | frei für Forschung |
| TT100K | 100 000 Bilder, ~26 000–30 000 Schilder, 221 Klassen, „large variations in illuminance and weather" | Schlechtwetter/Verdeckung | **CC-BY-NC** (kommerziell nur auf Anfrage) |
| Mapillary MTSD | 105 830 Bilder, 354 154 Boxen, 400 Klassen, Attribute `occluded`/`dummy`/`ambiguous` | Schräglage, Verdeckung, Welt | **nicht frei kommerziell** (eigene Lizenz klären) |
| INTSD | 11 016 Bilder, 14 044 Schilder, 41 Klassen, Nacht, Glare/Blur/Noise + Distraktoren | Nacht/Blendung/harte Negative | vor Nutzung prüfen |

Für ein MIT-Repo heißt das: eigene Bilder + GTSRB/GTSDB sind unkritisch,
TT100K/MTSD/INTSD erst nach Klärung. Für den Detektor ist ein 1-Klassen-Remap trivial
(alle Boxen → eine Klasse), die Typentscheidung kommt aus dem Kopf.

### 4.3 So wird hier tatsächlich trainiert: GTSRB + eigene Hintergründe

GTSRB liefert **echte** deutsche Schilder als Ausschnitte (mit ROI und Klasse), aber keine
Szenen. Das Netz braucht aber Vollbilder mit Box. `tools/gtsrb_dataset.py` löst das:
echten Ausschnitt auf einen erzeugten Hintergrund kleben (Asphalt, Himmel, Störobjekte),
dazu Roll-Verdrehung, Verdeckung, Blendung, Unschärfe, Rauschen – und die Box exakt
mitschreiben. Ergebnis: **echte Schilderoptik in harten Bedingungen, lizenzfrei**
(GTSRB ist für Forschung frei).

```
curl -L -C - -o data/gtsrb/gtsrb-train.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
curl -L -C - -o data/gtsrb/gtsrb-test.zip  \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
python tools/gtsrb_dataset.py --zip data/gtsrb/gtsrb-train.zip --out data/det --n 20000 --size 256 --split train
python tools/gtsrb_dataset.py --zip data/gtsrb/gtsrb-test.zip  --out data/det --n 600   --size 256 --split val
```

Die 43 GTSRB-Klassen werden auf die 9 Typen abgebildet (`CLASS_MAP` in
`tools/gtsrb_dataset.py`): Tempolimits/Überholverbote → `verbot`, Kl. 17 →
`einfahrtVerboten`, Kl. 13/11 → `vorfahrtGewaehren`, Kl. 14 → `stop`, Kl. 18–31 →
`warnung`, Kl. 12 → `vorfahrtstrasse`, Kl. 33–40 → `gebot`.
**Verworfen** werden 32/41/42 („Ende von …", grau/weiß) – sie gehören nicht zu den neun
Typen und würden die Objektivität verwässern. `hinweis` und `ortstafel` kommen aus dem
synthetischen Erzeuger, weil GTSRB sie nicht enthält.

Kein Datenleck: Zug und Validierung kommen aus **verschiedenen Archiven** (Trainings-Zip
bzw. Test-Zip), nicht aus denselben Aufnahmen.

### 4.4 Eigene Bilder (der eigentliche Aufwand)


Ziel: 3 000–5 000 Frames mit ~6 000–10 000 Boxen **plus ≥ 1 000 Frames ohne Schild**
(gegen Fehltreffer). Pro Fahrt 10–20 s Video plus Standbilder. Bedingungen **je Bild**
taggen, sonst ist keine Auswertung nach Bedingung möglich:

frontal · schräg (Roll 15–60°) · gegenlicht · dämmerung · nacht · regen · unscharf ·
klein/weit · verdeckt · verblichen/verschmutzt · baustelle · stoerer (rote/blaue/gelbe
Nicht-Schilder: Plakate, Jacken, Fahrzeuge).

### 4.5 Synthetische Daten

`tools/synth_data.py` erzeugt Bilder mit genau den Bedingungen, die die Heuristik
nachweislich brechen (siehe `docs/DOKUMENTATION.md`, Abschnitt „Grenzen"): Dunkelheit,
Entsättigung, Bewegungsunschärfe, Roll-Verdrehung, Verdeckung, Blendung, Rauschen,
bunte Störer. Zweck: Trainings-/Exportkette und Tests ohne echten Datensatz prüfbar.

```
python tools/synth_data.py --out data/synth --n 200
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8   # on-the-fly
```

## 5. Training

```
python tools/train_det.py --data data/det --epochs 60 --steps 200 --batch 16 \
    --size 320 --lr 2e-3 --tr-global p5 --tr-window - --degrade 0.6 \
    --out models/signs-det.pt
```

| Schalter | Wirkung |
|---|---|
| `--tr-global p5,p4` | Transformer-Stufen (global). `-` schaltet ab (reines CNN) |
| `--tr-window p4` | lokale Transformer-Stufen |
| `--act silu\|hardswish` | Hardswish war in der Messung ~6 % schneller |
| `--norm gn\|ln` | GroupNorm (schnell) oder LayerNorm je Token (genauer, teurer) |
| `--degrade 0…1` | Anteil künstlich verschlechterter Bilder |
| `--resume models/signs-det.pt` | weitertrainieren (Architektur + Auflösung kommen aus dem Checkpoint) |

Verlust (`DetLoss`): Objektivität als **fokale** BCE, Klasse als Cross-Entropy mit
Label-Smoothing (nur positive Zellen), Box als L1 auf `(tx,ty,tw,th)` mit
Größengewichtung (`2 − Fläche/Bildfläche`, begrenzt auf 0,5…2).

Wichtig und gemessen: die Objektivität wird **durch die Anzahl positiver Zellen**
geteilt (RetinaNet-Normierung), **nicht** durch alle Zellen. Mit „Mittelwert über alle
Zellen" war der Lernschritt für die wenigen Schilder um Faktor ~16 000 verdünnt – der
Verlust bewegte sich kaum und die Validierung fand nichts. Nach der Umstellung fiel
`obj` in 14 Schritten von 13,7 auf 2,0.

Zielwerte je Objekt: **genau eine Zelle** auf der Stufe, deren stride der Objektgröße am
nächsten liegt (`argmin |log2(diag/stride)|`). Dadurch bleibt der Offset `tx,ty` in
[0,1) und passt exakt zur Decode-Formel.

### 5.1 Selbsttest: kann das Netz ein Bild überfitten?

```
python tools/selfcheck.py
```

Er trainiert 120 Schritte auf **einem** synthetischen Bild. Ergebnis auf dieser Maschine:
nach 80 Schritten Klasse `ortstafel` mit Score 0,32, nach 120 Schritten Score 0,59 und
Box `(77, 45, 139, 86)` gegen GT `(77, 46, 134, 83)` → **IoU ≈ 0,90**.
Damit ist die Kette Zuweisung → Verlust → Dekodierung nachweislich konsistent.
Fällt dieser Test unter IoU 0,5, ist etwas an Zuweisung oder Dekodierung kaputt –
**nicht** die Datenmenge.


## 6. Export, Paritätscheck, Quantisierung

```
python tools/export_onnx.py --ckpt models/signs-det.pt --out models/signs-det.onnx --int8 --calib-n 200
python tools/export_fixture.py --ckpt models/signs-det.pt        # Testfixture für tests/model.test.js
```

Der Export erzeugt `models/signs-det.onnx` (Eingang `images`, Ausgänge `os8`, `os16`,
`os32`), `models/labels.json` (Klassen, Größe, Stufen, Kanal-Layout – das liest
`src/model.js`) und `models/manifest.json` (Architektur, Parameter, Trainingsmetriken,
Parität, Größen, **sha256**, Torch-Version).

Der **Paritätscheck** im Export ist Pflicht, nicht Deko: PyTorch und ONNX Runtime
rechnen dieselben Bilder, verglichen werden rohe Tensoren **und** die fertigen
Erkennungen. Gemessen auf dieser Maschine (Beispielmodell, 128 px):

* maximale Tensorabweichung **1,5 · 10⁻⁵**
* Erkennungen: gleiche Anzahl, alle Paare mit **IoU ≥ 0,95** → Export ist korrekt

**int8 ist nicht automatisch gut.** Mit nur 6 synthetischen Kalibrierbildern war das
quantisierte Modell unbrauchbar: maximale Abweichung **4,12**, **0 von 1** Erkennungen
identisch. `tools/export_onnx.py` hat deshalb ein **Qualitätstor**: Ist die Parität
schlechter als 0,5 bzw. gehen Erkennungen verloren, wird die int8-Datei **verworfen**
und fp32 bleibt aktiv (mit Hinweis im Log). Rezepte für einen zweiten Versuch:
mehr Kalibrierbilder aus echten Frames (`--calib-n 200`), `QuantFormat.QOperator`
statt QDQ, oder fp16 statt int8.

Warum das wichtig ist: Größe und Latenz sind verlockend (1,47 MB statt 5,16 MB), aber
ein stillschweigend verschlechtertes Modell wäre im Feld schwer zu finden.

## 7. Browser-Seite

`index.html` lädt `onnxruntime-web@1.30.0` von jsDelivr; die WebAssembly-Dateien holt
sich ONNX Runtime Web aus demselben Verzeichnis. `src/model.js`

* liest `models/labels.json`,
* setzt `ort.env.wasm.numThreads = 1` (GitHub Pages kann COOP/COEP nicht setzen),
* probiert `executionProviders: ['webgpu','wasm']` und fällt auf `['wasm']` zurück,
* dekodiert und unterdrückt Doppel (NMS je Klasse), rechnet die Letterbox zurück,
* liefert dieselbe Struktur `{label, x, y, w, h, conf, …}` wie `detector.js`.

`src/app.js` entscheidet: Modell geladen → Modellpfad, sonst Heuristik. Im Modellpfad
wird das **Quellbild** (Video/Photo) in ein 320×320-Letterbox-Canvas gezeichnet – das
Netz sieht also echte 320 px Detail statt der auf 240 px verkleinerten Heuristik-Eingabe.
Nebeneffekte, die man kennen sollte:

* `conf` ist im Modellpfad eine **echte Score-Zahl** (sigmoid(obj)·max sigmoid(cls)),
  in der Heuristik war es Formtreue – die Anzeige in Prozent ist also jetzt
  vergleichbar mit „Sicherheit", nicht mehr mit „wie rund war es".
* Die **Farbmasken-Ansicht** gibt es nur in der Heuristik (das Netz hat keine Maske).
* `createTracker` (3 Treffer bis Anzeige) bleibt unverändert; mit einem besseren
  Modell kann man `minHits` senken, das ist aber eine eigene Messung.
* Erstes Laden braucht Netz (CDN) + Modell (~1,5–5 MB). Danach liegt beides im
  HTTP-Cache; für echten Offline-Betrieb kann man das Modell in IndexedDB legen.

## 8. Was ist tatsächlich verifiziert?

| Punkt | Status |
|---|---|
| Modell, Training, Verlust, Zuweisung | **verifiziert** (`tools/selfcheck.py`: IoU ≈ 0,90 auf einem Bild) |
| Parameter/FLOPs je Transformer-Variante | **gemessen** (`tools/bench_model.py`) |
| ONNX-Export = PyTorch-Ausgabe | **verifiziert** (1,5 · 10⁻⁵, IoU ≥ 0,95) |
| JavaScript-Dekodierung = Python-Dekodierung | **verifiziert** (`tests/model.test.js`, Fixture mit echten ONNX-Tensoren) |
| Letterbox/NMS/Rückrechnung in JS | **verifiziert** (16 Checks, `node tests/model.test.js`) |
| Laufzeit auf **echten Handys** (WebGPU/WASM) | **nicht gemessen** – benötigt Gerätetest |
| Erkennungsqualität auf echten Straßenbildern | **nicht gemessen** – dafür fehlen die Daten (Abschnitt 4.3) |
| int8 brauchbar | **widerlegt** für 6 Kalibrierbilder, Tor verwirft es |

## 9. Grenzen und nächste Schritte

1. **Zahlen/Symbole** („30", Pfeile) liest auch dieses Netz nicht – dafür braucht es
   einen zweiten, kleinen Klassifikator auf dem Innenausschnitt oder OCR.
2. **Rechenzeit auf dem Handy** zuerst messen: 320 px auf WebGPU, 192–224 px im
   WASM-Rückfall. Die WEBGPU-Verfügbarkeit ist gut (Android 12+, iOS 26), aber
   nicht überall.
3. **Auswertung nach Bedingung** fehlt noch: ein Skript, das `data/det` mit
   `manifest.conditions` gruppiert und Recall/Fehltreffer je Gruppe ausgibt. Ohne das
   ist jede Verbesserung nur ein Gefühl.
4. **Verlust verfeinern**: CIoU statt L1 für die Box, Hard-Negative-Mining für die
   Objektivität – beides typische +2…5 Punkte Recall bei kleinen Schildern.
5. **Lizenzen** (Abschnitt 4.2) klären, bevor Fremddaten in ein veröffentlichtes Modell
   einfließen.
6. **Modellversionierung** über `models/manifest.json` (sha256 + Metriken) beibehalten:
   ohne Trainingsdaten-Hash ist ein Modell nicht reproduzierbar.

