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
(Letterbox, Grau 114), Ausgang **vier** Tensoren `os4`, `os8`, `os16`, `os32`
(ein Ausgang je Erkennungsstufe, fein → grob).

| Teil | Aufbau |
|---|---|
| Stem | 3×3 Conv stride 2 → 16 Kanäle (auf /2) |
| Stufe /4 → `p2` | Inverted-Residual-Block (MobileNet-Stil), 32 Kanäle |
| Stufe /8 → `p3` | 2 × IR-Block, 64 Kanäle |
| Stufe /16 → `p4` | 2 × IR-Block, 128 Kanäle |
| Stufe /32 → `p5` | 2 × IR-Block, 256 Kanäle |
| **Transformer** | `p5` global (10×10 = 100 Tokens), optional `p4` (20×20 = 400) lokal/windowed oder global |
| Fusion | FPN-lite: 1×1 lateral + Nearest-Upsample + Add + 3×3 Fuse, von grob nach fein über alle Stufen |
| Köpfe | je Stufe: 3×3 Depthwise → 1×1 (gemeinsamer Stamm), dann **zwei 1×1-Zweige** – 5 Kanäle (`tx,ty,tw,th,obj`) und 9 Kanäle (`cls0..cls8`) |

Kopf-Layout je Zelle: `[tx, ty, tw, th, obj, cls0..cls8]` (14 Kanäle), Dekodierung
(Python wie JS):

```
cx = (gx + sigmoid(tx)) * stride        cy = (gy + sigmoid(ty)) * stride
w  = exp(clip(tw, ±8)) * stride         h  = exp(clip(th, ±8)) * stride
score = sigmoid(obj) * max_j sigmoid(cls_j)
```

**Warum vier Stufen und warum ein getrennter Kopf?** Beides kam aus der Messung, nicht aus
dem Gefühl (Zahlen in §10). Von 656 verpassten Boxen waren 334 „tief verpasst“, davon
46 % kleiner als 32 px Diagonale – der Recall für kleine Schilder lag bei etwa **0,29**
gegen 0,74 im Mittel. Eine Ursache war die alte Zuordnungsregel (nächste Stufe in `log2`
der Diagonale): sie schickte ein 30-px-Schild auf `stride 32`, wo nur 10×10 Zellen zur
Verfügung stehen. Dazu kamen **212 falsch klassifizierte Boxen** (32 % aller FN) und 195
der 226 Fehlalarme auf echten Schildern – der Kopf verwechselte Arten (rotes Dreieck
Spitze oben gegen Spitze unten: 32×). Deshalb: `stride 4` dazu und der Kopf in zwei
getrennte Zweige (Ort/Objekt gegen Art).

Die Zuordnung der Objekte zu den Stufen läuft über die **längste Objektseite**
(`ASSIGN_MAX_SIDE = (16, 40, 96, ∞)` px für stride 4/8/16/32, siehe `tools/detmath.py`);
außerdem lernt bei Randlage die Nachbarzelle dasselbe Objekt mit (`dual`). Die alte
Ein-Zellen-Regel presste den Offset auf 0,999 fest – bei großen Schildern (Median der
Lokalisierungsfehler: 125 px Diagonale) musste also eine einzige Zelle die Box mehrere
Zellen weit regressieren.

**Warum Transformer nur in den tiefen Stufen?** Attention kostet O(N²) in der Tokenzahl
N = (H/stride)·(W/stride). Bei 320×320 ergibt das:

| Stufe | Raster | Tokens | Nachbarschaft bei globaler Attention |
|---|---|---|---|
| /4 (`p2`) | 80×80 | 6400 | 40 960 000 Paare → unbezahlbar |
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

`python tools/bench_model.py --iters 4` auf dieser Maschine (CPU, PyTorch 2.14, 320×320).
Params und MFLOPs sind exakt, die Millisekunden streuen auf dieser CPU stark
(Auslastung) – sie sind eine Größenordnung, kein Messprotokoll.

Fertige Größen (`--preset`), alle mit **vier** Stufen (stride 4/8/16/32):

| Preset | Params | int8-Datei | MFLOPs | ms (1 Thread) |
|---|---|---|---|---|
| `fast` | 537 k | ~0,54 MB | 433 | 24 |
| **`balanced`** (ausgeliefert) | 1 300 k | ~1,30 MB | **878** | 34–60 |
| `quality` | 1 575 k | ~1,57 MB | 1 180 | 66–169 |

Transformer-Varianten (Basis = vier Stufen, sonst `balanced`):

| Variante | Params | int8-Datei | MFLOPs | ms (4 Threads) | ms (1 Thread) |
|---|---|---|---|---|---|
| cnn-only | 764 k | ~0,76 MB | **762** | 37 | 36 |
| **p5-global** (Standard) | 1 300 k | ~1,30 MB | **878** | 35 | 35–57 |
| p5g + p4 windowed | 1 438 k | ~1,44 MB | 991 | 54 | 55–96 |
| p4 global + p5 global | 1 438 k | ~1,44 MB | 1 068 | 51 | 70 |
| p5g + p4w, TokenNorm=LayerNorm | 1 438 k | ~1,44 MB | 991 | 65 | 55 |
| p5g + p4w + p3w | 1 473 k | ~1,47 MB | 1 112 | 56 | 52 |
| p5g + Hardswish | 1 300 k | ~1,30 MB | 878 | 34 | 34 |

Ablesbare Erkenntnisse:

* Der **`p5`-Transformer kostet ~15 % Mehrrechnung** (762 → 878 MFLOPs) und liegt bei den
  Parametern in derselben Größenordnung wie ein „Nano“-Detektor (1,3 M).
* **Die vierte Stufe (stride 4) ist der eigentliche Zuwachs:** sie hob die Rechnung von
  731 auf 878 MFLOPs (+20 %) und den kleinen-Schilder-Recall deutlich – siehe §10.
* **Windowed ist in FLOPs billiger, aber auf dieser CPU nicht schneller** — die
  Fenster-Reshapes kosten mehr als die reine Attention. Auf WebGPU/WASM neu messen,
  bevor man sich festlegt.
* **`p3` windowed ist der Kostentreiber** (1600 Tokens): +27 % FLOPs für den kleinsten
  erwartbaren Nutzen → nicht verwenden.
* **Hardswish** spart hier ~6 % Zeit bei identischen FLOPs (auf ARM meist mehr).
* Auflösung schlägt alles andere: **256 px sind 64 % der Rechnung von 320 px**, 384 px
  sind 144 %, 448 px 196 %. Weil mit `--zoom` trainiert wurde, sind das reine
  Einstellungen am ausgelieferten Modell (§3).

## 3. Echtzeit-Budget

Zwei Messpunkte, die zusammen das Budget ergeben:

* **Im Browser, `wasm` (Einfachthread), am echten Gerät: 122 ms** je Bild bei 320 px für
  die Vorfassung mit 731 MFLOPs – also **~0,167 ms je MFLOP** (Vorbereitung Letterbox:
  2 ms). Das ist die belastbare Zahl, alles andere hier ist Umrechnung.
* **ONNX Runtime CPU (x86, 1 Thread)** als Referenz für den Graphen: 7,2 ms bei 320 px
  (Vorfassung, 731 MFLOPs), 10–18 ms für das ausgelieferte Modell (878 MFLOPs).

Was die Eingabegröße auf den val-Bildern kostet und bringt (2 000 Bilder, conf 0,25):

| Eingabe | MFLOPs | P | R | F1 | fp/Bild | Browser (grob) |
|---|---|---|---|---|---|---|
| 256 px | 562 | 0,859 | 0,742 | 0,796 | 0,152 | ~110 ms |
| **320 px** (Standard) | 878 | 0,883 | 0,796 | 0,837 | 0,132 | ~150–230 ms |
| **384 px** | 1 264 | **0,907** | **0,808** | **0,854** | **0,104** | ~210–330 ms |
| 448 px | 1 721 | – | – | – | – | ~290–450 ms |

Die Spanne kommt von der Frage, wie stark offene Höhe/Breite (dynamischer Export) im WASM
kosten: auf x86 gemessen **+59 %** (17,7 ms statisch gegen 28,2 ms dynamisch bei 320 px),
im WASM unbekannt. Ausgeliefert wird deshalb die **dynamische** Datei (eine Datei für alle
Größen, die App kann umschalten) – wenn Live-Video zu langsam ist, gibt es zwei Wege:
in der App **256 px** wählen, oder statisch neu exportieren:

```
python tools/export_onnx.py --ckpt models/signs-det.pt --size 384   # ohne --dynamic
```

Statisch 384 px kostete auf x86 **16,7 ms** – also so viel wie statisch 320 px, aber mit
der besseren Qualität. Die App zeigt die gemessene Zeit im Status an
(`KI-Modell (wasm, 384 px) · 213 ms (Vorbereitung 2 ms)`).

| Weg | Erwartung | Maßnahme |
|---|---|---|
| WebGPU (Chrome/Android 12+, Safari/iOS 26+) | deutlich unter 10 ms | 320 px fahren, `enableGraphCapture` möglich |
| WASM einthreadig (GitHub Pages, kein COOP/COEP) | 10² ms | Größe wählen: 256 px = Sparmodus, 384 px = Foto |
| kein WebGPU, Modell fehlt, offline | – | Heuristik (5–15 ms bei 240 px) bleibt aktiv |

**Offen:** warum die App auf `wasm` zurückfällt, obwohl `webgpu` zuerst versucht wird.
Wenn WebGPU trägt, ist deutlich mehr Rechnung bezahlbar.

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
curl -L -C - -o data/gtsrb/GTSRB_Final_Training_Images.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
curl -L -C - -o data/gtsrb/GTSRB_Final_Test_Images.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
curl -L -C - -o data/gtsrb/GTSRB_Final_Test_GT.zip \
  https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

python tools/gtsrb_dataset.py --zip data/gtsrb/GTSRB_Final_Training_Images.zip \
    --out data/det --n 20000 --size 320 --seed 0 --balance
python tools/gtsrb_dataset.py --zip data/gtsrb/GTSRB_Final_Test_Images.zip \
    --gt-zip data/gtsrb/GTSRB_Final_Test_GT.zip --out data/det --n 1800 --size 320 \
    --split val --seed 1
```

**`--balance` (nur für `--split train`)** zieht die Schilder je Typ gleich häufig, weil
GTSRB zur Hälfte aus Tempolimits besteht (alle → `verbot`). Gemessen vor dem Umbau: `verbot`
stellte **50 % aller Validierungsboxen**, während `vorfahrtGewaehren` nur **R = 0,43** und
`stop` nur **R = 0,48** erreichten. Nach dem Ausgleich hat jeder Typ ~3 600 Boxen
(`stop` 3 625, `verbot` 3 591 statt ~200 bzw. ~15 000). Die **Validierung bleibt absichtlich
unausgeglichen** – sie ist die Messlatte und soll GTSRB widerspiegeln, nicht das Training.

**Roll-Verdrehung ist formabhängig** (`roll_angle()` in `tools/synth_data.py`): Kreise,
Rechtecke und Rauten werden weiterhin bis ±70° gedreht (dort bricht die Heuristik), rote
**Dreiecke nur bis ±35°**. Grund: ein Dreieck bei ~65° ist von seinem Gegenstück („Vorfahrt
gewähren“ gegen „Gefahrzeichen“) nicht mehr zu unterscheiden – solche Bilder sind
widersprüchliche Lernziele. Messung dazu: `warnung` verlor **18 Recall-Punkte** auf den
stark gedrehten Bildern (0,832 → 0,655), und die Verwechslung `vorfahrtGewaehren → warnung`
war mit 32 Fällen der häufigste Fehler überhaupt.

**`--gt-zip` ist beim Test-Set Pflicht:** die Bilder liegen in
`GTSRB_Final_Test_Images.zip`, die Labels in einem **eigenen** Archiv
(`GTSRB_Final_Test_GT.zip`, `GT-final_test.csv`). Die Datei
`GT-final_test.test.csv` im Bildarchiv hat **keine** `ClassId`-Spalte und liefert
stillschweigend null Zeilen – `read_rows()` prüft deshalb die Kopfzeile und nimmt nur
Annotationen mit Klassen.

Die 43 GTSRB-Klassen werden auf die 9 Typen abgebildet (`CLASS_MAP` in
`tools/gtsrb_dataset.py`): Tempolimits/Überholverbote → `verbot`, Kl. 17 →
`einfahrtVerboten`, Kl. 13/11 → `vorfahrtGewaehren`, Kl. 14 → `stop`, Kl. 18–31 →
`warnung`, Kl. 12 → `vorfahrtstrasse`, Kl. 33–40 → `gebot`.
**Verworfen** werden 32/41/42 („Ende von …", grau/weiß) – sie gehören nicht zu den neun
Typen und würden die Objektivität verwässern.

**GTSRB hat zwei der neun Typen nicht:** blaue Hinweiszeichen (Z 3xx) und gelbe Ortstafel
(Z 310). Ohne Nachschub sagt das Netz diese Klassen nie voraus – im KI-Modus wären blaue
und gelbe Rechtecke unerreichbar. Beide sind geometrisch einfach, deshalb füllt
`tools/synth_missing.py` die Lücke mit denselben Kacheln, die auch `synth_data.py` nutzt,
und hängt sie in den bestehenden Datensatz (`src = "synthetisch (Lueckenschluss)"`):

```
python tools/synth_missing.py --out data/det --n 3000              # train
python tools/synth_missing.py --out data/det --n 200 --split val   # Messlatte dazu
```

Gemessen danach auf den beiden synthetischen Typen: `hinweis` P=0,840/R=0,781,
`ortstafel` P=0,971/R=0,745 – der Lückenschluss wirkt.

**Bilder ohne Schild** erzeugt `tools/synth_negatives.py` – zwei Sorten aus derselben
Hintergrundverteilung wie die Positivbilder (sonst lernt das Netz „Hintergrund“ statt
„Schild“): reiner Hintergrund (`leer`) und **harte Negative** (`hart:*`), also
schildähnliche Störer: Rückleuchten, Ampel, Baustelle, Leuchtreklame, graue
Schildrückseite, rote Flagge, gelbe Richtungstafel, farbige Kreise. Labels sind bewusst
leer (0 Byte).

```
python tools/synth_negatives.py --out data/det --n 3000 --split train
python tools/synth_negatives.py --out data/det --n 500  --split neg    # Messlatte
```

Der Split **`neg`** ist die Gegenprobe: nur Bilder ohne Schild. Dort ist **`fp/Bild`** die
Kennzahl (`--split neg` in `tools/eval_conditions.py`). Ohne diesen Split wäre „keine
Fehlalarme auf freier Fläche“ nicht messbar, weil jedes Validierungsbild ein Schild enthält.
Ehrlich dazugesagt: gemessen lagen nur **31 der 226 Fehlalarme auf freiem Hintergrund**
(14 %), **195 dagegen auf echten Schildern** – die Negative sind also eine Absicherung
gegen Rückleuchten, Ampel & Co., aber nicht das Hauptproblem (das war die Klassentrennung,
siehe §1 und §9.4).

Kein Datenleck: Zug und Validierung kommen aus **verschiedenen Archiven** (Trainings-Zip
bzw. Test-Zip), nicht aus denselben Aufnahmen. Der Validierungssatz des Test-Archivs ist
das letzte 15-%-Stück des Dateinamenslaufs (zusammenhängende Aufnahmen bleiben zusammen).

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

Drei Bausteine kamen beim Umbau dazu:

* **`compose_negative()` / `hard_negative_sample()`** – Bilder ganz ohne Schild
  (siehe §4.3), inklusive schildähnlicher Störer und weicher Kanten, damit sie nicht
  trivial zu trennen sind.
* **mehr Hintergrundarten** (`random_background`: asphalt, sky, busy, **wand, gruen,
  nacht**) – Positiv- und Negativbilder ziehen aus derselben Verteilung.
* **`roll_angle()`** – formabhängige Roll-Verdrehung (Dreiecke nur ±35°, siehe §4.3).

```
python tools/synth_data.py --out data/synth --n 200
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8   # on-the-fly
```

## 5. Training

```
python tools/train_det.py --data data/det --size 320 --preset balanced \
    --batch 16 --epochs 220 --steps 100 --lr 1.5e-3 --degrade 0.6 --zoom 0.5 \
    --workers 6 --eval-every 10 --save-every 10 \
    --out models/signs-det.pt --seed 7
```

| Schalter | Wirkung |
|---|---|
| `--preset fast\|balanced\|quality` | Größe komplett aus `tools/hybrid_net.py` (überschreibt Breiten/Tiefen) |
| `--device auto\|cpu\|cuda` | `auto` nimmt CUDA, wenn vorhanden – sonst läuft dasselbe Kommando auf der CPU |
| `--tr-global p5,p4` | Transformer-Stufen (global). `-` schaltet ab (reines CNN) |
| `--tr-window p4` | lokale Transformer-Stufen |
| `--act silu\|hardswish` | Hardswish war in der Messung ~6 % schneller |
| `--norm gn\|ln` | GroupNorm (schnell) oder LayerNorm je Token (genauer, teurer) |
| `--degrade 0…1` | Anteil künstlich verschlechterter Bilder (zusätzlich zur Erzeugung) |
| **`--zoom 0…1`** | Anteil Trainingsbilder mit **Skalenschnitt** (Mehrskaligkeit), Bereich `--zoom-lo`/`--zoom-hi` (Standard 0,7–1,5) |
| **`--obj-norm pos\|sqrt`** | Normierung des Objektivitätsverlusts. `pos` = RetinaNet-Rezept (durch die Zahl positiver Zellen); `sqrt` ist ruhiger, wenn viele Negative im Batch liegen |
| **`--val-split val\|neg`** | worauf während des Trainings gemessen wird (`neg` = nur Hintergrund, dann ist `fp/Bild` die Kennzahl) |
| `--eval-every N` | Auswertung nur alle N Epochen (die letzte Epoche wird immer gemessen) |
| `--resume models/signs-det.pt` | weitertrainieren (Architektur + Auflösung kommen aus dem Checkpoint) |
| `--steps`, `--epochs` | Bilddurchläufe: eine Epoche sieht `steps × batch` Bilder |

**`--zoom` ist der Grund, warum die Eingabegröße im Browser frei bleibt.** Ohne
Skalenschnitt ist das Netz auf genau die Skalen festgelegt, die im Datensatz vorkommen;
mit ihm lernt es dieselben Gewichte bei anderen Größen – 256 px (schnell), 320 px
(Standard) oder 448 px (Standbild) sind damit eine Einstellung statt drei Modelle.

> **Alte Checkpoints sind nicht mehr ladbar.** Der Umbau auf vier Stufen und zwei
> Kopfzweige hat die Parameternamen im Kopf geändert (`pred` → `obj` + `cls`). Ein
> Trainingslauf muss von Grund auf neu starten; `--resume` funktioniert nur noch für
> Checkpoints ab diesem Umbau. Die Zahlen der Vorfassung (drei Stufen, ein Kopfzweig)
> stehen in §5.1 und §9 als Vergleich, das neue Werkzeug kann sie nicht neu berechnen.

### 5.1 Was hier tatsächlich gelaufen ist

`--preset balanced`, 320×320, GPU der Klasse GTX 1060: **rund 20 s je Epoche**
(100 Schritte × 16 Bilder, inkl. Nachladen und Augmentation in 6 Ladeprozessen),
Auswertung über 2 000 Validierungsbilder ~40 s.

**Vorfassung** (drei Stufen, ein Kopfzweig, 1 287 066 Parameter, 731 MFLOPs):

| Block | Daten | Epochen | val bei Epoche | P | R | F1 |
|---|---|---|---|---|---|---|
| 1 | 20 000 GTSRB-Kompositionen | 0–79 | 9 → 79 | 0,680 → **0,885** | 0,154 → **0,736** | 0,251 → **0,804** |
| 2 | dieselben Daten (Feinschliff) | 80–119 | 79 → 119 | 0,885 → 0,880 | 0,736 → 0,742 | 0,804 → 0,805 |
| 3 | + 3 000 Bilder `hinweis`/`ortstafel` | 120–159 | 159 | **0,891** | **0,739** | **0,808** |

Ablauf der Kurve aus Block 1 (jede Auswertung über die vollen 1 800 Bilder):
R = 0,154 (Ep. 9) → 0,342 (14) → 0,594 (29) → 0,651 (39) → 0,695 (54) → 0,721 (69) →
0,736 (79); P blieb dabei zwischen 0,84 und 0,89. **Block 2 hat nichts mehr gebracht**
(Plateau, Daten ausgeschöpft) – die 3 000 synthetischen Bilder in Block 3 dagegen schon:
erst mit ihnen existieren `hinweis` und `ortstafel` im Modell.

**Aktuelle Fassung** (vier Stufen, getrennter Kopf, 1 300 360 Parameter, 878 MFLOPs) –
ein Lauf von Grund auf, 220 Epochen in 1 h 15 min:

| Block | Daten | Epochen | val bei Epoche | P | R | F1 |
|---|---|---|---|---|---|---|
| 4 | 26 000 Bilder (20 000 ausgeglichene GTSRB + 3 000 Lückenschluss + 3 000 Negative) | 0–219 | 9 → 219 | 0,811 → **0,883** | 0,029 → **0,796** | 0,056 → **0,837** |

Der Verlauf zeigt, warum Geduld nötig war – die ausgeglichenen Klassen und die
Mehrskaligkeit brauchen länger, holen aber weiter aus:

| Epoche | 9 | 19 | 29 | 39 | 49 | 59 | 79 | 99 | 149 | 189 | 219 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **neu** R | 0,029 | 0,183 | 0,329 | 0,483 | 0,588 | 0,669 | 0,692 | 0,752 | 0,780 | 0,793 | **0,796** |
| Vorfassung R | 0,154 | 0,447 | 0,594 | 0,651 | 0,684 | 0,712 | 0,736 | 0,742 | 0,738 | – | 0,739 |

**Ergebnis:** Recall **+5,7 Punkte** (0,739 → 0,796), F1 **+2,9 Punkte** (0,808 → 0,837),
Precision −0,8 Punkte (0,891 → 0,883). Verpasste Boxen: 656 → **513** (−22 %),
Fehlalarme 226 → 264 (+17 %). Der Zugewinn liegt genau dort, wo er geplant war
(Klassen mit wenigen Beispielen, §5.2), der Preis bei `verbot` – der Klasse, die vorher
die Hälfte aller Boxen stellte.

### 5.2 Auswertung nach Bedingung und Klasse

`tools/eval_conditions.py` gruppiert die Validierungsbilder über `manifest.conditions`
und zählt tp/fp/fn je Gruppe:

```
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --diagnose --json data/eval_val4.json
```

Stand des ausgelieferten Modells (2 000 val-Bilder, 2 509 Boxen, `conf` 0,25, IoU 0,5):
P 0,883 / R 0,796 / F1 0,837.

| Bedingung | Bilder | Boxen | P | R |
|---|---|---|---|---|
| ohne Bedingung | 140 | 144 | 0,906 | 0,868 |
| dunkel | 412 | 525 | 0,903 | 0,798 |
| rauschen | 673 | 848 | 0,881 | 0,800 |
| roll | 1 066 | 1 436 | 0,879 | 0,792 |
| unschaerfe5 | 132 | 159 | 0,855 | 0,818 |
| verblichen | 636 | 797 | 0,886 | 0,793 |
| hell | 413 | 501 | 0,864 | 0,798 |
| verdeckt | 601 | 762 | 0,862 | 0,790 |
| blendung | 497 | 639 | 0,865 | 0,773 |
| unschaerfe3 | 132 | 173 | 0,840 | 0,792 |
| unschaerfe2 | 113 | 140 | 0,851 | 0,814 |
| unschaerfe4 | 112 | 132 | 0,835 | 0,765 |
| unschaerfe6 | 109 | 148 | 0,877 | 0,723 |
| **mehrere** | 444 | 888 | 0,860 | **0,682** |

Vergleich zur Vorfassung (dieselben Bilder): Unschärfe war dort 0,69–0,72, Blendung 0,71,
Rauschen 0,72, „mehrere“ 0,634. **Alles besser, „mehrere“ bleibt die härteste Bedingung**
(zwei Schilder im Bild sind meist klein und liegen nah beieinander).

Je Klasse (dieselbe Messung) – hier liegt der eigentliche Umbaugewinn:

| Klasse | Boxen | P | R | R Vorfassung |
|---|---|---|---|---|
| ortstafel | 137 | 0,956 | **0,942** | 0,745 |
| hinweis | 128 | 0,622 | **0,938** | 0,781 |
| gebot | 339 | 0,863 | 0,891 | 0,838 |
| vorfahrtstrasse | 91 | 0,902 | **0,813** | 0,615 |
| einfahrtVerboten | 62 | 0,758 | **0,806** | 0,532 |
| verbot | 1 115 | 0,981 | 0,779 | 0,801 |
| warnung | 395 | 0,863 | 0,719 | 0,704 |
| stop | 46 | 0,805 | **0,717** | 0,478 |
| vorfahrtGewaehren | 196 | 0,758 | **0,689** | 0,434 |

Lesbare Erkenntnisse:

* **Alle Klassen mit wenigen Beispielen haben massiv zugelegt**: `stop` +24 Punkte,
  `einfahrtVerboten` +27, `vorfahrtGewaehren` +26, `vorfahrtstrasse` +20. Das ist der
  Klassenausgleich im Training (§4.3) – vorher stellte `verbot` die Hälfte aller Boxen.
* **`verbot` hat 2 Punkte verloren** (0,801 → 0,779) und ist der Preis dafür. Weil
  `verbot` aber 44 % der val-Boxen stellt, wirkt sich das auf die Gesamtzahl aus.
* **`hinweis` hat die schwächste Precision (0,622, 73 Fehlalarme)** – blaue Rechtecke
  werden zu oft als Hinweiszeichen gedeutet (Plakate, Schilder, Werbung). Das ist der
  nächste Kandidat für harte Negative.
* Die Dreiecks-Verwechslung ist kleiner, aber nicht weg: `vorfahrtGewaehren → warnung`
  war 32× der häufigste Fehler, jetzt 14× (dafür ist `verbot → gebot` mit 31× der neue
  häufigste). Details in §10.2.
* Unschärfe, Dunkelheit, Blendung und Rauschen brechen das Modell weiterhin **nicht** –
  genau die Bedingungen, an denen die Heuristik scheitert.

Verlust (`DetLoss`): Objektivität als **fokale** BCE, Klasse als Cross-Entropy mit
Label-Smoothing (nur positive Zellen), Box als L1 auf `(tx,ty,tw,th)` mit
Größengewichtung (`2 − Fläche/Bildfläche`, begrenzt auf 0,5…2).

| Klasse | Boxen | P | R | Bemerkung |
|---|---|---|---|---|
| ortstafel | 137 | 0,971 | 0,745 | rein synthetisch gelernt |
| verbot | 1 115 | 0,934 | 0,801 | größte Klasse |
| gebot | 339 | 0,928 | 0,838 | |
| vorfahrtstrasse | 91 | 0,903 | 0,615 | wenige Beispiele |
| vorfahrtGewaehren | 196 | 0,876 | **0,434** | Verwechslung mit `warnung` |
| einfahrtVerboten | 62 | 0,846 | 0,532 | wenige Beispiele |
| hinweis | 128 | 0,840 | 0,781 | rein synthetisch gelernt |
| warnung | 395 | 0,768 | 0,704 | 84 FP – nimmt die Dreiecke an |
| stop | 46 | 0,667 | 0,478 | nur 46 val-Boxen |

Lesbare Erkenntnisse:

* **`vorfahrtGewaehren` (Spitze unten) gegen `warnung` (Spitze oben) ist der Hauptfehler.**
  Im Datensatz stecken Roll-Verdrehungen bis ±60° – dort ist „Spitze oben/unten" teilweise
  nicht mehr entscheidbar. Wer das braucht, muss die Roll-Verteilung begrenzen oder
  `vorfahrtGewaehren` über die Innenfläche (weißes Dreieck) unterscheiden lernen.
* **„mehrere" ist die härteste Bedingung** (R 0,63): zwei Schilder im Bild sind meist
  klein und liegen nah beieinander.
* **Kleine Klassen bleiben schwach** (`stop` 46, `einfahrtVerboten` 62 Boxen) – hier
  fehlen schlicht Beispiele; mehr Kompositionen dieser Klassen wären der billigste Schritt.
* Unschärfe, Dunkelheit, Blendung und Rauschen brechen das Modell **nicht** mehr
  (R 0,69–0,76) – genau die Bedingungen, an denen die Heuristik scheitert.

Verlust (`DetLoss`): Objektivität als **fokale** BCE, Klasse als Cross-Entropy mit
Label-Smoothing (nur positive Zellen), Box als L1 auf `(tx,ty,tw,th)` mit
Größengewichtung (`2 − Fläche/Bildfläche`, begrenzt auf 0,5…2).

Wichtig und gemessen: die Objektivität wird **durch die Anzahl positiver Zellen**
geteilt (RetinaNet-Normierung), **nicht** durch alle Zellen. Mit „Mittelwert über alle
Zellen" war der Lernschritt für die wenigen Schilder um Faktor ~16 000 verdünnt – der
Verlust bewegte sich kaum und die Validierung fand nichts. Nach der Umstellung fiel
`obj` in 14 Schritten von 13,7 auf 2,0.

Zielwerte je Objekt: **eine Zelle** auf der Stufe, die zur **längsten Objektseite** passt
(`ASSIGN_MAX_SIDE` in `tools/detmath.py`: < 16 px → stride 4, < 40 → stride 8, < 96 →
stride 16, sonst stride 32). Liegt der Objektmittelpunkt im äußeren Viertel seiner Zelle,
lernt zusätzlich die **Nachbarzelle** dasselbe Objekt (`dual`) – das behebt den auf 0,999
festgeklemmten Offset der alten Ein-Zellen-Regel. Die vorherige Regel
(`argmin |log2(diag/stride)|`) schickte ein 30-px-Schild auf stride 32 und damit auf ein
10×10-Raster; sie ist der Grund für den kleinen-Schilder-Recall von ~0,29.

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
python tools/export_onnx.py --ckpt models/signs-det.pt --out models/signs-det.onnx --dynamic --int8 --calib-n 200
python tools/export_fixture.py --ckpt models/signs-det.pt        # Testfixture für tests/model.test.js
```

Der Export erzeugt `models/signs-det.onnx` (Eingang `images`, **ein Ausgang je Stufe**:
`os4`, `os8`, `os16`, `os32`), `models/labels.json` (Klassen, Größe, Stufen, Kanal-Layout –
das liest `src/model.js`) und `models/manifest.json` (Architektur, Parameter,
Trainingsmetriken, Parität, Größen, **sha256**, Torch-Version).

`--dynamic` lässt Höhe und Breite offen. Dann kann **ein** Modell mit 256, 320, 384 oder
448 px laufen, und die App bekommt eine Qualitäts-/Tempowahl (§3). Die Gegenprobe dazu ist
im Export eingebaut: dasselbe Bild wird zusätzlich in `--check-size` durch ONNX geschickt
und die Ausgangsformen werden gegen `size/stride` geprüft. Gemessen:

```
[dynamisch] Eingang 384x384 -> Formen [[1,14,96,96], [1,14,48,48], [1,14,24,24], [1,14,12,12]]  passend=True  max. Abweichung=9,54e-06
```

Der **Paritätscheck** im Export ist Pflicht, nicht Deko: PyTorch und ONNX Runtime rechnen
dieselben Bilder, verglichen werden rohe Tensoren **und** die fertigen Erkennungen.
Gemessen am **ausgelieferten** Modell (E219, dynamisch):

* maximale Tensorabweichung **2,1 · 10⁻⁴**
* Erkennungen: 9 gegen 9, **alle** Paare mit **IoU ≥ 0,95** → Export ist korrekt
* Dateigröße **5,23 MB** (fp32), ONNX Runtime CPU (x86, 1 Thread), 320×320: **10–18 ms**
  je Bild (Referenzmessung des Exports; die Streuung kommt von der Auslastung der Maschine,
  nicht vom Graphen). Statisch exportiert sind es 17,7 ms bei 320 px und 16,7 ms bei
  384 px – die offenen Achsen kosten auf x86 rund 59 % (§3).

**int8 ist nicht automatisch gut.** Mit 6 synthetischen Kalibrierbildern war das
quantisierte Modell unbrauchbar: maximale Abweichung **4,12**. Auch mit **200 echten**
Kalibrierbildern aus `data/det` bleibt es falsch (Abweichung **3,16 · 10¹**,
0 von 2 Erkennungen identisch) – gemessen am gleichen Checkpoint. `tools/export_onnx.py`
hat deshalb ein **Qualitätstor**: Ist die Parität schlechter als 0,5 bzw. gehen
Erkennungen verloren, wird die int8-Datei **verworfen** und fp32 bleibt aktiv (mit
Hinweis im Log). Rezepte für einen zweiten Versuch: `QuantFormat.QOperator` statt QDQ,
`QuantType.QUInt8`, fp16 statt int8, oder die LayerNorm-Ersatzpfade vorher
quantisierungsfreundlich umbauen (die Warnungen des Quantisierers zeigen genau dorthin:
`Expected bias '/p5/norm1/op/Constant_2_output_0' to be an initializer`).

Warum das wichtig ist: Größe und Latenz sind verlockend (1,47 MB statt 5,23 MB), aber
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
wird das **Quellbild** (Video/Photo) in ein Letterbox-Canvas gezeichnet – das Netz sieht
also echte 320 px Detail statt der auf 240 px verkleinerten Heuristik-Eingabe. Die
**Eingabegröße** ist wählbar (Auswahlfeld „KI-Eingang“, erscheint nur im KI-Modus):

* **automatisch** – Video in Modellgröße (320 px), **Foto in 384 px** (dort zählt Tempo
  nicht, und 384 px bringt gemessene +2,5 Punkte Recall bei −21 % Fehlalarmen, §3)
* **256 px** als Sparmodus für langsame Geräte, **448 px** für ein scharfes Einzelbild
* Der Status zeigt die tatsächlich benutzte Größe und die gemessene Zeit:
  `KI-Modell (wasm, 384 px) · 213 ms (Vorbereitung 2 ms)`

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
| Modell, Training, Verlust, Zuweisung | **verifiziert** (`tools/selfcheck.py`: nach 120 Schritten wird das eine Bild gefunden) |
| **Training auf echten Daten** (GTSRB, 26 000 + 2 000 Bilder) | **gelaufen** – P = 0,883 / R = 0,796 / F1 = 0,837 auf den val-Bildern (§5.1); Vorfassung 0,891 / 0,739 / 0,808 |
| **Auswertung nach Bedingung und Klasse** | **gemessen** (`tools/eval_conditions.py --diagnose`, §5.2/§10) – schlechtester Fall `mehrere` R = 0,682, schwächste Precision `hinweis` 0,622 |
| **Fehlalarme auf Bildern ohne Schild** | **gemessen** (Split `neg`, 500 Bilder): 0,011 je Bild auf reinem Hintergrund, 0,155 auf schildähnlichen Störern |
| **Eingabegröße 256/320/384 px** | **gemessen** (§3): F1 0,796 / 0,837 / 0,854 – als App-Einstellung verfügbar |
| **Alle neun Typen im Modell** | **verifiziert** – `hinweis`/`ortstafel` rein synthetisch gelernt (§4.3) |
| Parameter/FLOPs je Variante | **gemessen** (`tools/bench_model.py`) |
| ONNX-Export = PyTorch-Ausgabe | **verifiziert** (2,1 · 10⁻⁴, 9/9 Erkennungen IoU ≥ 0,95; dynamische Achsen zusätzlich bei 384 px geprüft) |
| JavaScript-Dekodierung = Python-Dekodierung | **verifiziert** (`node tests/model.test.js`, Fixture aus dem **ausgelieferten** Modell) |
| Letterbox/NMS/Rückrechnung in JS | **verifiziert** (`node tests/model.test.js`, 32 Prüfungen) |
| Laufzeit im **Browser** (wasm, echtes Gerät) | **gemessen für die Vorfassung**: 122 ms bei 320 px. Für die neue Fassung steht die Gerätezahl noch aus (die App zeigt sie an) |
| Laufzeit auf **echten Handys** (WebGPU/WASM) | **nicht gemessen** – benötigt Gerätetest |
| Erkennungsqualität auf echten **Straßenbildern** | **nicht gemessen** – der val-Satz besteht aus GTSRB-Ausschnitten auf erzeugten Hintergründen, nicht aus ganzen Szenen |
| int8 brauchbar | **widerlegt** – 6 synthetische *und* 200 echte Kalibrierbilder, Tor verwirft es |

## 9. Grenzen und nächste Schritte

1. **Zahlen/Symbole** („30", Pfeile) liest auch dieses Netz nicht – dafür braucht es
   einen zweiten, kleinen Klassifikator auf dem Innenausschnitt oder OCR.
2. **Rechenzeit im Browser**: gemessen wurde `wasm` (Einfachthread) mit **122 ms** für die
   Vorfassung (731 MFLOPs) – das sind ~0,167 ms je MFLOP. Die neue Fassung hat 878 MFLOPs,
   also grob 150 ms; die Zahl am echten Gerät steht noch aus (Anzeige in der App).
   Offene Frage: warum greift der Rückfall auf `wasm`, obwohl `webgpu` zuerst versucht
   wird – wenn WebGPU läuft, ist deutlich mehr Rechnung bezahlbar.
3. ~~**Auswertung nach Bedingung** fehlt noch~~ → **erledigt**: `tools/eval_conditions.py`
   (Tabelle je Bedingung und je Klasse, `--diagnose` für die Fehlerzerlegung, `--json` für
   die Ablage in `data/`).
4. **Dreiecks-Verwechslung** (`vorfahrtGewaehren` R=0,43 gegen `warnung`) – **angegangen**
   mit drei Maßnahmen: Roll-Verdrehung für Dreiecke auf ±35° begrenzt (die alte
   Augmentation erzeugte widersprüchliche Ziele), Klassen ausgeglichen, Kopf in zwei
   Zweige getrennt. Ergebnisse in §5.1/§10. Bleibt offen: die Innenfläche (weißes Dreieck)
   als zusätzliches Merkmal, und eine Entscheidungsschicht, die zwei Dreiecks-Klassen auf
   derselben Stelle gegeneinander abwägt – gemessen brachte eine Arbitrierung nach NMS
   allerdings nur **+0,7 % Precision**, deshalb nicht eingebaut.
5. ~~**Kleine Klassen auffüllen**~~ → **erledigt** über `--balance` in
   `tools/gtsrb_dataset.py`: jeder Typ hat jetzt ~3 600 Trainingsboxen (`stop` 3 625 statt
   ~200, `verbot` 3 591 statt ~15 000). Die Validierung bleibt absichtlich unausgeglichen.
6. **Verlust und Zuweisung**: `dual` (Nachbarzelle bei Randlage) und die Zuordnung nach
   längster Seite sind **eingebaut**. Ein **CIoU-Verlust ist gemessen widerlegt** – die
   Treffer haben schon eine mittlere IoU von 0,938, dort ist nichts zu holen.
   **Hard-Negative-Mining bleibt offen**: das Netz könnte über den negativen Pool laufen
   und die besten Fehltreffer selbst als Trainingsbilder nachliefern. Der gemessene Nutzen
   ist aber begrenzt, weil nur 31 der 226 Fehlalarme auf freier Fläche lagen.
   Ebenfalls gemessen: **mehr Epochen allein bringen nichts** (Block 2 in §5.1 war ein
   Plateau über 40 Epochen).
7. **Eigene Szenen** (§4.4) sind weiterhin der eigentliche Qualitätssprung: die val-Zahlen
   hier stammen aus komponierten Bildern, nicht aus Kamerafahrten.
8. **Lizenzen** (Abschnitt 4.2) klären, bevor Fremddaten in ein veröffentlichtes Modell
   einfließen.
9. **Modellversionierung** über `models/manifest.json` (sha256 + Metriken) beibehalten:
   ohne Trainingsdaten-Hash ist ein Modell nicht reproduzierbar. Für die Daten gehört
   zusätzlich der GTSRB-Archiv-Hash dazu (die Archive sind unveränderlich, der Hash
   steht auf der ERDA-Seite).
10. **Kompatibilität**: Checkpoints der Vorfassung (drei Stufen, ein Kopfzweig) sind nicht
    mehr ladbar (siehe §5). Wer den Vergleich reproduzieren will, muss die Vorfassung aus
    der Versionsgeschichte holen und dort trainieren.

## 10. Fehlerzerlegung: warum Boxen verpasst oder erfunden werden

Eine Gesamtzahl versteckt die Ursache. `tools/eval_conditions.py --diagnose` teilt deshalb
jede verpasste Box ein (keine Erkennung in der Nähe = Objektivität, Box sitzt falsch,
falsche Klasse) und jeden Fehlalarm (auf einem echten Schild oder auf freier Fläche):

```
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --diagnose
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --split neg
```

### 10.1 Vorfassung (drei Stufen, ein Kopfzweig) – die Zahlen, die den Umbau ausgelöst haben

Gemessen mit 2 000 val-Bildern (2 509 Boxen), conf 0,25, IoU 0,5:
TP 1 853, FP 226, FN 656.

| Ursache | Anzahl | Anteil | Größe (Median) |
|---|---|---|---|
| verpasst (Objektivität) | 334 | 51 % der FN | 35 px – **46 % davon ≤ 32 px** |
| falsche Klasse | **212** | **32 % der FN** | – |
| Box sitzt falsch (IoU 0,1–0,5) | 110 | 17 % der FN | 125 px (große Schilder) |
| FP auf echtem Schild | **195** | **86 % der FP** | – |
| FP auf freier Fläche | 31 | 14 % der FP | – |

Weiteres aus derselben Messung:

* **Treffer-IoU im Mittel 0,938** (84 % über 0,9) – die Lokalisierung *gefundener* Schilder
  ist nicht das Problem, ein CIoU-Verlust hätte hier nichts zu holen.
* Verwechslungen (`gt → vorhergesagt`): `vorfahrtGewaehren → warnung` **32×** (häufigster
  Fehler), `warnung → verbot` 12×, `verbot → warnung` 11×.
* Nach Roll getrennt (1 000 Bilder): gesamt R 0,771 → **0,729** mit Roll; `warnung`
  0,832 → **0,655**.
* Klassenerinnerung: `vorfahrtGewaehren` **0,434**, `stop` 0,583, `einfahrtVerboten` 0,559,
  `vorfahrtstrasse` 0,632, `warnung` 0,737, `verbot` 0,792, `gebot` 0,876 – und `verbot`
  stellte **50 % aller Boxen** (GTSRB besteht zur Hälfte aus Tempolimits).
* Schwellenwanderung (600 Bilder): conf 0,15 → P 0,812 / R 0,808; conf 0,25 → 0,896 / 0,753;
  conf 0,35 → 0,938 / 0,675. F1 ist dabei fast konstant – die Schwelle ist ein
  Recall/Precision-Regler, kein Qualitätsgewinn.
* **Nachverarbeitung hilft nicht:** eine Klassen-Arbitrierung nach der NMS (zwei Klassen auf
  derselben Stelle → nur die beste) brachte **+0,7 % Precision**; klassenloses NMS zerstört
  das Ergebnis (P 0,023). Die Fehlalarme sind also keine Duplikate auf identischer Stelle,
  sondern echte Fehlentscheidungen des Kopfes.

Daraus folgten genau die vier Umbauten: Zuordnung nach längster Seite plus `stride 4`
(kleine Schilder), getrennter Kopf (Arten), Klassenausgleich, begrenzte Roll-Verdrehung für
Dreiecke. Die Negative gegen freie Fläche wurden trotzdem ergänzt (§4.3) – nur mit dem
klaren Wissen, dass sie 14 % der Fehlalarme adressieren, nicht 86 %.

### 10.2 Aktuelle Fassung

Dieselbe Messung, dieselben 2 000 val-Bilder, dieselbe Schwelle. Zum Vergleich die
Vorfassung daneben (TP 1 996, FP 264, FN 513):

| Ursache | Anzahl | Anteil | vorher | Größe (Median) |
|---|---|---|---|---|
| verpasst (Objektivität) | 266 | 51,9 % | 334 | 30 px |
| falsche Klasse | 160 | 31,2 % | 212 | – |
| Box sitzt falsch (IoU 0,1–0,5) | 87 | 17,0 % | 110 | 125 px |
| FP auf echtem Schild | 180 | 68,2 % | 195 | – |
| FP auf freier Fläche | 84 | 31,8 % | 31 | – |

* **Lokalisierung deutlich besser:** Treffer-IoU im Mittel **0,954** (vorher 0,938),
  nur noch **10** Treffer unter 0,7 (vorher 31). Das ist der Effekt von `stride 4` plus
  der Nachbarzellen-Zuweisung – nicht von einem neuen Boxverlust (der blieb L1).
* **Alle drei Fehlerarten sind gesunken**: −68 verpasst, −52 falsche Klasse, −23 falsche Box.
* **Fehlalarme haben sich verschoben, nicht verringert:** auf freier Fläche 31 → 84. Das
  ausgeglichene Training macht seltene Typen empfindlicher, und die schwächste Precision
  hat jetzt `hinweis` (0,622) – blaue Rechtecke. Auf dem **negativen Split** (500 Bilder
  ohne jedes Schild, von `tools/synth_negatives.py`):

  | Art der Fläche | Bilder | Fehlalarme | je Bild |
  |---|---|---|---|
  | nur Hintergrund (`leer`) | 274 | 3 | **0,011** |
  | harte Störer (`hart:*`) | 226 | 35 | 0,155 |
  | **gesamt** | 500 | 38 | **0,076** |

  Reiner Hintergrund ist also praktisch fehlalarmfrei (3 von 274), während die bewusst
  schildähnlichen Störer weiter treffen – am stärksten `hart:kreise` (13 von 30) und
  `hart:bake` (5 von 28). Ziel war ≤ 0,05 gesamt; erreicht ist 0,076. Der ehrliche
  nächste Schritt wäre Hard-Negative-Mining genau auf diesen Bildern.
* **Verwechslungen:** `vorfahrtGewaehren → warnung` von 32 auf **14** gefallen (die
  Roll-Begrenzung wirkt), dafür ist `verbot → gebot` mit 31 der neue Spitzenreiter –
  rote Kreise werden mit blauen verwechselt. Auch das ist eine Folge des Ausgleichs
  (`gebot` bekommt mehr Gewicht) und der nächste Ansatzpunkt.
* **Fehlalarme sind selten sicher:** 25 von 264 liegen über Score 0,6, der Rest darunter
  (87 bei 0,25–0,30, 93 bei 0,30–0,40, 59 bei 0,40–0,60). Eine höhere Schwelle in der
  App verschiebt das Verhältnis also wirksam – auf Kosten von Recall (§10.1).

Reproduzieren:

```
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --diagnose
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --split neg
python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --size 384   # größerer Eingang
```



