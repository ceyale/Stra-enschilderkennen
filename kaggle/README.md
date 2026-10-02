# Training auf Kaggle

Hier liegt alles, was das Training auf Kaggle ausfuehrt: der Kernel (Datensatz bauen,
trainieren, exportieren, messen), die Metadaten fuer Datensatz und Kernel und ein Skript,
das die Reihenfolge abarbeitet.

## Warum ueberhaupt Kaggle

Der Entwicklungsrechner kann **nicht** rechnen: die Windows-Anwendungssteuerung
(*Smart App Control*) blockiert die DLLs von PyTorch, der Import endet mit

```
OSError: [WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert.
Error loading "...\torch\lib\torch_global_deps.dll"
```

Nachgeprueft ist die Richtlinie aktiv (`HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`,
`VerifiedAndReputablePolicyState = 1`). Ohne `torch` laeuft **kein** Schritt der Kette -
auch der Datensatz nicht, denn `tools/hybrid_net.py` liefert `SIGN_LABELS` und importiert
dafuer PyTorch. Deshalb rechnet Kaggle; der Rechenweg bleibt aber derselbe: `train_kernel.py`
ruft `tools/gtsrb_dataset.py`, `tools/synth_missing.py`, `tools/synth_negatives.py`,
`tools/real_negatives.py`, `tools/train_det.py`, `tools/export_onnx.py`,
`tools/export_fixture.py` und `tools/eval_conditions.py` mit genau den Aufrufen aus
[`docs/TRAINING.md`](../docs/TRAINING.md) §4.3 und §5 auf.

## Was auf Kaggle liegt

| Kaggle-Objekt | Inhalt | erzeugt von |
|---|---|---|
| Datensatz `raphbre/schilder-det-raw` (privat) | `tools/` (Code aus diesem Repo), `gtsrb/*.zip` (GTSRB Training/Test/Test-GT), `negatives/coco128.zip`, `user/Nothing.jpg` | `run.ps1 -Step stage` + `-Step dataset` |
| Kernel `raphbre/schilder-scanner-detektor-trainieren-gtsrb` (privat, GPU) | `train_kernel.py` | `run.ps1 -Step push` |

Weil der Code **im Datensatz** liegt, braucht der Kernel kein Netz (`enable_internet: false`).
Aendert sich etwas in `tools/`, muss der Datensatz neu hochgeladen werden
(`run.ps1 -Step stage -NeueVersion` bzw. `-Step dataset -NeueVersion`).

## Ablauf

```powershell
.\kaggle\run.ps1 -Step raw        # fehlende Rohdaten holen (GTSRB-Test-GT, coco128)
.\kaggle\run.ps1 -Step stage      # Upload-Verzeichnis anlegen (harte Verknuepfungen)
.\kaggle\run.ps1 -Step probe      # Kernel-Befehle pruefen (ohne Rechnen, ohne Netz)
.\kaggle\run.ps1 -Step dataset    # Datensatz anlegen (einmalig, 356 MB)
.\kaggle\run.ps1 -Step push       # Kernel starten (12-h-Reissleine gesetzt)
.\kaggle\run.ps1 -Step status     # laeuft es? (KernelWorkerStatus.RUNNING/COMPLETE/ERROR)
.\kaggle\run.ps1 -Step logs       # Protokoll (die API liefert es erst NACH dem Lauf)
.\kaggle\run.ps1 -Step warten     # bis zum Ende warten, dann pull + install in einem Zug
.\kaggle\run.ps1 -Step pull       # Ergebnisse holen (400 MB; -NurModelle laesst das ZIP weg)
.\kaggle\run.ps1 -Step install    # models/ + tests/fixtures/ uebernehmen
```

Live mitlesen geht nur im Browser
(`https://www.kaggle.com/code/raphbre/schilder-scanner-detektor-trainieren-gtsrb`):
`kaggle kernels logs` gibt **waehrend** des Laufs nichts aus, sondern erst danach (gemessen -
solange der Lauf lief, kam eine leere Antwort, direkt nach dem Abbruch das vollstaendige
Protokoll als JSON). `-Step status` ist die schnelle Frage „laeuft es noch?".

`-Step alle` macht `raw -> stage -> probe -> dataset -> push`. Das Staging liegt
absichtlich ausserhalb des Repos (`%TEMP%\schilder-kaggle`), damit die 365 MB GTSRB-ZIPs
nicht ein zweites Mal im OneDrive-Ordner landen; verknuepft wird hart (`HardLink`).

## Rezept (im Kernel oben einstellbar)

`DATEN` (Block 5 aus [`PLAN.md`](../PLAN.md)): 20 000 ausgeglichene GTSRB-Kompositionen,
1 800 GTSRB-Testbilder als Messlatte, 3 000 synthetische Bilder fuer `hinweis`/`ortstafel`
(+200 in der Messlatte), 3 000 synthetische und 3 600 echte Negative; Split `neg` mit
500 + 600 Bildern. Ergebnis: **29 600** Trainingsbilder, **2 000** val, **1 100** neg.
Der Kernel prueft diese Zahlen im Manifest und bricht ab, wenn sie abweichen.


## Was am Ende herauskommt

Alles landet in `/kaggle/working` und ist der Ausgang des Kernels:

| Datei | Bedeutung |
|---|---|
| `models/signs-det.pt` | Checkpoint (Ausgangspunkt fuer `--resume`, Export, weitere Messungen) |
| `models/signs-det.onnx` | ausgeliefertes Modell, dynamische Hoehe/Breite (256…448 px) - geht in `src/model.js` |
| `models/labels.json`, `models/manifest.json` | Klassen/Eingabegroesse fuer den Browser, Paritaet + sha256 |
| `berichte/val.json`, `neg.json`, `val384.json` | Precision/Recall je Bedingung und Klasse, inkl. Fehlerzerlegung |
| `berichte/model-out.json` | Fixture fuer `tests/model.test.js` (Gegenprobe Modell <-> `src/model.js`) |
| `berichte/protokoll.txt` | das vollstaendige Protokoll aller Werkzeugaufrufe samt Trainingskurve |
| `data/det.zip` | der gebaute Datensatz (32 700 Bilder) - dieselbe Fassung, die hier rechnen wuerde |
| `kaggle_report.json` | Kurzfassung: Rezept, Dauer je Schritt, Kennzahlen, Trainingskurve |

`-Step install` kopiert `models/*` nach `models/` und die Fixture nach
`tests/fixtures/model-out.json`. Danach ist die Kette lokal pruefbar:

```powershell
node tests/model.test.js
node tests/detector.test.js
python tools/build_site.py
```

`-Step install -DatenErsetzen` entpackt zusaetzlich `data/det.zip` nach `data/det`. Der
lokale Stand dort ist ein **aelterer, unvollstaendiger Aufbau** (20 600 Bilder, `val` nur
600 statt 2 000, keine echten Negative) - der Kaggle-Lauf baut den vollstaendigen Stand.
Die Bilder sind deterministisch (feste Saaten in den Werkzeugen), der Aufbau ist also
wiederholbar.

## Zeitplan und Grenzen

| Schritt | Dauer (Richtwert) |
|---|---|
| Datensatz bauen (31 700 Bilder + Zippen) | 15-25 min |
| Training (220 x 100 x 16, P100/T4) | 60-90 min, Auswertung alle 10 Epochen kostet mit |
| Export + Paritaet + drei Auswertungen | 5-10 min |
| **Summe** | **rund 2 h** (Kaggle-Grenze: 12 h je Lauf, 30 h GPU je Woche) |

Der Kernel bricht **ohne GPU** sofort ab - ein CPU-Lauf dauerte Stunden und verbrauchte die
Wochenquote, ohne dass das Ergebnis anders waere. Fehlt die GPU im Lauf, stimmt etwas an
der Kernel-Einstellung: in `kernel-metadata.json` muss `"enable_gpu": "true"` stehen.

## Fallstricke (gemessen, nicht vermutet)

* **`-r zip` ist beim Upload Pflicht.** Ohne das (`-r skip`, die Vorgabe) landen
  Unterordner **nicht** im Datensatz: ein Testdatensatz mit `a.txt` und `sub/b.txt` enthielt
  nach dem Hochladen nur `a.txt`. Mit `-r zip` bleibt die Struktur erhalten (`sub/b.txt`).
* **Kaggle entpackt hochgeladene ZIPs beim Anlegen des Datensatzes.** Aus `gtsrb-train.zip`
  wird der Ordner `gtsrb/gtsrb-train/GTSRB/Final_Training/Images/…`, aus `coco128.zip` wird
  `negatives/coco128/coco128/…`. Die Werkzeuge brauchen aber ein Archiv (`zipfile` + ROI-CSV
  im selben Zugriff), deshalb packt `archiv_bereitstellen()` im Kernel den entpackten Ordner
  wieder zu `gtsrb/train.zip` - der Inhalt bleibt byte-gleich, der gebaute Datensatz ist
  derselbe. Wer den Datensatz lieber unverpackt hochladen will, muss in `train_kernel.py`
  nichts aendern: das Skript akzeptiert beide Formen (ZIP *oder* Ordner).
* **Der Kernelname kommt aus dem Titel (Slug), nicht aus `id`.** Bei `kaggle kernels push`
  gewinnt der Titel: er wird zu `…-detektor-trainieren-gtsrb`, die Warnung „title does not
  resolve to the specified id" erscheint, wenn beides auseinandergeht. Deshalb steht in
  `kernel-metadata.json` die ID genau so, wie der Titel sie ergibt.
* **Der Titel darf hoechstens 50 Zeichen lang sein** - laenger lehnt die API mit
  `400 Bad Request … SaveKernel` ab (der erste Versuch scheiterte genau daran).
* **Der Eingang liegt nicht immer direkt unter `/kaggle/input/<slug>`.** Im ersten Lauf lag
  er unter `/kaggle/input/datasets/<besitzer>/<slug>` - der Kernel brach mit „kein
  Rohdaten-Datensatz gefunden (tools/train_det.py fehlt); in /kaggle/input liegt: datasets"
  ab. `eingang_finden()` durchsucht deshalb vier Ebenen und erkennt den Datensatz am
  Merkmal `tools/train_det.py` statt am Namen. Wer ein anderes Einhaengeverzeichnis hat,
  setzt `SCHILDER_INPUT`.
* **Die Dateinamen der Rohdaten** muessen so bleiben, wie der Kernel sie sucht
  (`gtsrb/gtsrb-train.zip`, `gtsrb/gtsrb-test.zip`, `gtsrb/gtsrb-test-gt.zip`,
  `negatives/coco128.zip`, `user/Nothing.jpg`) - die Werkzeuge erwarten genau diese Pfade.
* **`val` nie anfassen.** Der Kernel wertet auf `val` (2 000 Bilder) aus, damit die Zahlen
  mit den veroeffentlichten vergleichbar bleiben.
* **`--obj-norm sqrt`** waere bei 22 % Negativen im Batch der ruhigere Objektivitaetsverlust,
  ist hier aber **ungemessen**. Wer es versucht, muss `TRAINING` im Kernel anpassen und
  den Datensatz **nicht** neu hochladen (nur der Code aendert sich - der Kernelcode liegt
  im Kernel, nicht im Datensatz; `-Step push` genuegt).
* **Kein Netz, keine Ueberraschung:** `enable_internet` bleibt aus, damit der Lauf sich in
  zwei Wochen noch genauso wiederholt - alle Quellen stecken im Datensatz.

## Schnellere Wiederholungen (optional)

`data/det.zip` aus dem Kernelausgang ist ein fertiger Datensatz. Wer ihn als eigenen
Kaggle-Datensatz hochlaedt (`schilderdet`), kann den Aufbau (15-25 min je Lauf) sparen:
dann `dataset_sources` in `kernel-metadata.json` um `raphbre/schilderdet` ergaenzen und im
Kernel statt `datensatz_bauen()` den vorhandenen Ordner aus `/kaggle/input` verwenden.
Das ist bewusst **noch nicht** eingebaut - es kostet einen weiteren Upload von 400 MB und
war fuer den ersten Lauf nicht noetig.

`TRAINING` ist der veroeffentlichte Lauf: `--preset balanced`, 320 px, 220 Epochen zu
100 Schritten (1 600 Bilder je Epoche), `lr 1.5e-3`, `--degrade 0.6`, `--zoom 0.5`,
`--obj-norm pos`, Auswertung alle 10 Epochen. Andere Werte sind erlaubt - aber dann sind
die Zahlen nicht mehr mit `docs/TRAINING.md` §5 vergleichbar. Der val-Split bleibt
absichtlich unberuehrt; er ist die einzige vergleichbare Messlatte.
