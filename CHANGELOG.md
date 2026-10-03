# Changelog

## 0.8.0 – 2026-10-03
- **Effizienterer Kopf: TCN-Engstelle statt vier voller 1×1-Faltungen.** Erst wurde gemessen,
  **wo** die Rechnung steckt (`tools/flops_wo.py`, Preset `breit`, 384 px): Backbone 31 %,
  CATM-Blöcke 27 %, **die vier Köpfe 33 %** – teuerster Einzelposten war die feinste Stufe
  (`heads.0`, 96×96 Zellen: 206 MFLOPs), und darin der **TCN** mit vier vollen 1×1-Faltungen
  `mid→mid` (Querterm **und** Tor je Richtung). Jetzt laufen beide über **eine gemeinsame
  Verengung auf mid/2**:
  * Parameter **4C² → 2C² + 2C** (halbiert), Rechnung in diesem Block **−62 %**
  * das Tor ist dadurch einkanalig (räumlich veränderlicher Skalar statt C Gewichte je Pixel) –
    „wie sehr darf der Ort hier den Art-Merkmalen trauen?" ist eine Frage je *Pixel*, die
    Kanalauswahl trifft schon die Engstelle
  * die beiden Richtungen lesen jetzt die **Eingangs**merkmale des anderen Zweigs, sind also
    unabhängig und können sich nicht in einer Schleife aufschaukeln
  Gemessen **am ganzen Netz**: **2016 → 1866 MFLOPs (−7,4 %)**, **2785k → 2611k Parameter
  (−174k)**, Kopf p2 **206 → 169 MFLOPs**, Laufzeit 158 → 112 ms; `balanced` 1874k/1324 →
  1836k/1276. Der Ausgangsvertrag bleibt (geprüft: `[[1,79,96,96],[1,79,48,48],[1,79,24,24],
  [1,79,12,12]]`).
- **Eine Sparmaßnahme wurde gemessen und VERWORFEN** – dilatierte 3×3 (d=2) statt 5×5 im
  GranularPerception-Zweig: im Zweig −64 % Rechnung bei gleichem Empfangsfeld, im **Netz** aber
  nur **−0,9 %** (1998 statt 2016 MFLOPs) und auf x86 **+12 % langsamer** (171 statt 153 ms),
  weil dilatierte Faltungen dort schlechter vektorisieren. Messung: `tools/gp_messung.py`,
  Ergebnis als Kommentar im Quelltext – damit die nächste Entscheidung nicht geraten wird.
- **Zwei Doku-Dateien waren durch mein eigenes Werkzeug beschädigt** (`Get-Content -Raw` +
  `Set-Content -Encoding utf8` liest UTF-8 als ANSI → aus jedem Umlaut wird ein Byte-Paar
  (mojibake-beispiel), 239 Stellen
  in `CHANGELOG.md`, dazu `docs/TRAINING.md`): repariert mit `tools/reparatur_mojibake.py`,
  und `tools/mojibake_pruefen.py` prüft seitdem **alle** versionierten Textdateien
  (Ergebnis: 47 Dateien, 0 Funde). Das Werkzeug überspringt eine Zeile, die sich nicht
  eindeutig zurückrechnen lässt, statt den Rest mitzunehmen – der erste Reparaturversuch
  scheiterte genau daran und ließ die Dateien kaputt.
- **fp16 für WebGPU** (`tools/export_onnx.py --fp16`, im Kaggle-Lauf jetzt mitgeschaltet):
  halbe Datei für den Weg, der sie nativ rechnen kann. Gemessen am Artefakt (320 px, fp32
  5,23 MB): fp16 **2,66 MB**, Ein-/Ausgänge bleiben **fp32** (`keep_io_types` – der Browser
  muss seine Tensoren nicht umrechnen, und die Kanalverträge `os4/os8/os16/os32` bleiben wie
  erwartet), maximale Tensorabweichung **8,9 Â· 10â»³**. Auf ONNX Runtime **CPU** ist fp16
  langsamer (11,6 gegen 8,3 ms) – genau deshalb wird sie nicht als Ersatz ausgeliefert.
  Wie bei int8 gilt das **Qualitätstor**: hält die Parität nicht, wird die Datei verworfen
  und im Log steht der Grund. `models/labels.json`/`manifest.json` führen sie in `files`
  (mit sha256); `kaggle/train_kernel.py` meldet ihre Größe im Ergebnis.
- **`src/model.js` wählt die Fassung nach der Ausführungsart, nicht global** – die
  Reihenfolge ist jetzt begründet und getestet statt „Hauptsache irgendein Rückfall":
  mit `navigator.gpu` zuerst **fp16/webgpu**, dann **int8**, dann **fp32**; ohne WebGPU
  direkt **int8 → fp32**. Der Status zeigt den Klartext (`fp16 webgpu`, `int8 wasm ×4`, …).
- **Zweiter Lehrer (optional): `kelvinandreas/vit-traffic-sign-GTSRB`** – der GTSRB-ViT
  (43 Klassen, Acc 0,985 / F1 0,985, **MIT**) ist auf seinen Klassen genauer als der
  Hauptlehrer, deckt aber nur **36 unserer 74** Typen ab (GTSIGN-220: 68). Er bleibt deshalb
  zweiter Lehrer und wird **0,5/0,5 gemittelt – gefiltert**: nur bei Boxen, deren
  **Grundwahrheitsklasse** zu seinen 36 gehört. Ungefiltert schriebe ein Lehrer, der
  „tempo40" nicht kennt, seine Meinung „tempo30" als Lernziel in den Cache.
  Nachgewiesen in `tests/teacher_probe.py` (Lehrer durch feste Zahlen ersetzt, ohne Netz):
  2 von 4 Boxen gemittelt, `tempo50` → 0,5·0,8 + 0,5·0,9, `tempo40` unverändert.
  Rezeptschalter `LEHRER["zweiter"]` (Standard **aus**: erst muss der Hauptlehrer allein
  messbar wirken; ein zweites 86-Mio.-Modell verdoppelt außerdem die Cachedauer).
  Lizenz-Hinweis: bei zwei Lehrern sind **beide** Lizenzen zu nennen – der Lauf warnt selbst.
- **Rezeptprüfung erweitert** (`tests/kernel_probe.py`, Teil 4): der Exportbefehl muss
  `--int8` **und** `--fp16` tragen, `tools/export_onnx.py` muss beide Flags kennen (ein
  unbekanntes Flag beendet argparse mit Code 2 – der Lauf hätte danach kein Modell) und
  `src/model.js` muss beide Fassungen auch **laden** (sonst liegt eine Datei im Ordner, die
  niemand benutzt). Alle vier Teile grün.
- **Zwei stille Fehler dabei gefunden** (beide von den neuen Prüfungen aufgedeckt, nicht im
  Feld): das Klassenfeld der Labeldatei steht in **Spalte 0** (`cls cx cy w h`), nicht ab
  Spalte 5 – die Filterung hätte sonst nie gegriffen; und der Cache wird als **float16**
  gespeichert, weshalb ein Vergleich „0,8 exakt" fehlschlägt (0,79980…) – die Prüfung
  vergleicht jetzt mit passender Toleranz.

## 0.7.0 – 2026-10-03
- **Wissens-Distillation von einem ViT-Lehrer** – der wirksamste Einzelhebel gegen den
  eigentlichen Fehler des v6-Laufs (Boxen gelernt, Arten nicht: Klassifikationsverlust blieb
  bei 3,4, Zufall wäre ln 74 ≈ 4,3; 195 von 226 Fehlalarmen lagen auf *echten* Schildern mit
  falscher Klasse).
  * **Lehrer:** `vit_gtsign_all_classes` aus dem GTSIGN-220-Datensatz – `google/vit-base-patch16-224`,
    86 Mio. Parameter, feinjustiert auf **220 deutsche StVO-Klassen**, veröffentlicht mit
    Accuracy 0,973 / P 0,911 / R 0,930 (Werte aus `eval_results.json` des Repos gelesen, nicht
    geschätzt). Lizenz **CC BY-SA 4.0** – die Weitergabe-Bedingung steht im Bericht
    (`kaggle_report.json → lehrer.lizenz`) und in `docs/TRAINING.md`.
  * **Warum genau dieser Lehrer:** ein auf COCO trainierter Detektor kennt nur „stop sign"
    (eine Klasse). Dieser kennt dieselben Zeichen wie wir, nur feiner geteilt (220 statt 74) –
    deshalb ist seine Zuordnung ein **Nachschlagen der StVO-Nummer** und keine Vermutung:
    **211 von 220** Lehrer-Klassen lassen sich zuordnen, sie decken **68 der 74** unserer
    Klassen ab. Ohne Lehrer bleiben: `tempo110`, `zone20`, `mindestgeschwindigkeit`,
    `gebotLinks`, `gebotGeradeaus`, `umleitung`.
  * **Neu `tools/teacher.py`:** rechnet die Lehrer-Verteilung **einmal je Grundwahrheitsbox**
    (Ausschnitt +25 % Rand, damit Achtkant-Ecken und Dreiecksspitzen nicht abgeschnitten
    werden), marginalisiert auf unsere 74 Klassen (Summe der Feinklassen – nicht Maximum, sonst
    ginge Masse verloren) und legt sie als `data/teacher/teacher_<split>.npz` (+ `.json` mit
    Modell, Lizenz, Zuordnung) ab. Nur der **Trainings**split – `val`/`neg` bleiben Messlatte.
  * **Neu `tools/train_det.py --teacher --distill --temperature`:** KL(Lehrer‖Schüler) auf den
    positiven Zellen, mit **T²** skaliert (Hinton u. a. 2015 – ohne T² wäre die Wirkung bei
    T=2 nur ein Viertel und die Gewichtsangabe bedeutete etwas anderes als sie sagt). Die
    Zuordnung Zelle→Box läuft über die neue `boxidx`-Ebene in `tools/detmath.py`, damit zwei
    Schilder derselben Klasse ihre **eigene** Verteilung bekommen können.
  * **Gewicht gemessen statt geraten:** eine positive Zelle trägt KL×T² von **2–9** bei
    (Prüfung mit künstlichem Ziel: 9,02), während der Klassifikationsverlust bei 0,1–0,3 und der
    Box-Verlust bei ~5 liegt. Deshalb **0,25** und nicht 1,0 – mit 1,0 hätte der Lehrer die
    übrigen Verluste überstimmt.
  * **Zwei Fehler dabei gefunden und behoben** (beide vor dem Kaggle-Lauf, mit Nachweis):
    `ViTForImageClassification.from_pretrained(<repo-URL>)` scheitert, weil das Modell in einem
    **Dataset**-Repo liegt (`hf_hub_download(repo_type="dataset")` nötig); und das Repo liefert
    **keine** Bildvorverarbeitung (`preprocessor_config.json` fehlt, 404) – sie kommt jetzt von
    `google/vit-base-patch16-224` mit festen Werten als letzter Stufe.
  * **Regression im eigenen Umbau gefunden und behoben:** ein zu früh gesetztes `continue` in
    `collate()` übersprang die *gesamte* Zielzuweisung, sobald kein Lehrer-Cache vorlag
    (`pos = 0`). Die Ziele werden jetzt **vor** allem Lehrer-Zeug gesetzt. Beweis:
    `pos je Stufe [0, 6, 2, 8]`, `obj je Stufe [0.0, 6.0, 2.0, 8.0]`.
- **Lokale Rezeptprüfung erweitert** (`tests/kernel_probe.py`, Teil 3): sie prüft jetzt auch,
  dass der Cache gefunden wird, `--teacher/--distill/--temperature` im Trainingsbefehl stehen
  (und *ohne* Cache nicht), dass das Gewicht im sinnvollen Bereich liegt und jeder Eingabepfad
  unter `WORK` existiert. Alle drei Teile grün: `alle Pruefungen bestanden`.

## 0.6.0 – 2026-10-03
- **Erkennungskopf ersetzt: TGADHead** (`tools/hybrid_net.py`) – aufgabengeführter,
  entkoppelter Kopf aus Zuo, Liu, Chen, Fu, Wang, *„TGADHead: An efficient and accurate
  task-guided attention-decoupled head for single-stage object detection"*, Knowledge-Based
  Systems **302:112349 (2024)**. Zwei Bausteine:
  * **TDAD** (Task Decoupled Attention Distributor): zwei aufgabenspezifische
    Aufmerksamkeits-Wahrnehmungen. Der **Ort**-Zweig gewichtet über die *Zellen* (Tiefenconv
    3×3 → 1 Kanal → Sigmoid), der **Art**-Zweig über die *Kanäle* (globaler Mittelwert → 1×1
    → Sigmoid). Getrennt, weil eine gemeinsame Gewichtung scharf-lokal und weich-global
    gleichzeitig sein müsste – zusammen mittelt sich das zu etwas, das keinem von beiden dient.
  * **TCN** (Task Correlation Network): Austausch *zwischen* den Zweigen
    (`ort += gate(1×1(art))`, `art += gate(1×1(ort))`, Tor startet bei Sigmoid 0,27). Der
    Abstract begründet ihn damit, dass genaue Ortung und hoher Klassenscore in bestehenden
    Detektoren auseinandergehen – genau das war hier gemessen: 195 von 226 Fehlalarmen lagen
    auf echten Schildern.
  * **Herkunft, offen benannt:** es gibt **keine öffentliche Referenzumsetzung**; der Code ist
    nach dem im Abstract beschriebenen Aufbau geschrieben und **nicht aus dem Paper
    abgetippt** – Abweichungen im Detail sind möglich. Steht so im Quelltext. Gemessen
    (Preset `breit`, 320 px): **2,79 Mio. Parameter** gegen 2,20 Mio. des Vorgängers, also
    +0,59 Mio. für Aufgabentrennung und Austausch.
- **Hierarchischer Klassifikationskopf**: statt einer flachen Entscheidung über 74 Klassen
  jetzt zwei Stufen – **9 Oberkategorien** (Familie: Form/Farbe) und **74 Unterkategorien**
  (Symbol im Inneren), verrechnet als `logit_k = super[familie(k)] + sub[k]`, in
  Log-Wahrscheinlichkeiten also `P(k) = P(Familie)·P(k|Familie)`.
  * Die Hierarchie steht in `tools/signmap.py` bei den Klassen (**eine** Taxonomie) und ist
    verteilt: **74 = 5 + 18 + 3 + 7 + 19 + 7 + 4 + 2 + 9** (Vorfahrt, Tempo, Überholen, Verbot,
    Warnung, Gebot, Rad/Fuß, Zone, Hinweis). `signmap.pruefen()` erzwingt jetzt, dass jede
    Unterkategorie genau einer Familie zugeordnet ist – ein Tippfehler wäre sonst erst als
    Klasse aufgefallen, die nie erkannt wird.
  * **Der Ausgangsvertrag bleibt unverändert** (79 Kanäle, `os4/os8/os16/os32`): die Summe
    steht im selben Tensor, `tools/detmath.py` und `src/model.js` sind nicht angefasst.
  * Die **Familie wird zusätzlich direkt überwacht** (`--hier-aux 0.3`, Hilfsverlust auf
    denselben positiven Zellen). Ohne ihn bekäme die grobe Entscheidung nur mittelbar ein
    Signal.
- **Focal Loss im Klassifikationskopf** (`tools/train_det.py`) mit **Klassengewichten** und
  **Label-Smoothing**:
  * `γ=2`, `α=0,25`; Klassen `w_c = (1/häufigkeit_c)^0,5`, auf Mittelwert 1 normiert
    (gedämpft – `1/f` hätte den Kopf in die Gegenrichtung kippen lassen), Smoothing `0,075`.
  * **α wirkt hier nur als konstanter Faktor** (der Kopf rechnet ausschließlich auf positiven
    Zellen, es gibt keine Negativklasse auszugleichen) und würde den Kopf still auf ein
    Viertel drosseln. Deshalb gleicht `--cls-w 4.0` das aus (0,25 × 4,0 = 1,0 wie vorher) –
    im Kernel mit Begründung hinterlegt.
  * Die Häufigkeiten kommen aus den **Labeldateien des Trainingssplits**, nicht aus `val`
    (sonst steckte eine Messlatte im Verlust). Gemessen am lokalen Datensatz:
    `tempo10=10175` ist die häufigste, Gewichte `0,65 … 1,11`.
- **Trainingsauflösung 320 → 384 px** (`kaggle/train_kernel.py`, `DATEN` *und* `TRAINING`).
  Grund: der Median der verpassten Objekte liegt bei 35 px Diagonale – bei 320 px ist das auf
  `stride 32` noch **ein** Pixel breit. Datensatz *und* Training müssen dieselbe Zahl
  benutzen; `--size` wird jetzt auch an `synth_negatives.py` und `real_negatives.py`
  durchgereicht (vorher schrieben die mit dem 320er-Default). Die zweite Messgröße wandert
  entsprechend auf **448 px**.
  * Dazu in `tools/detmath.py`: `ASSIGN_MAX_SIDE` wird mit `size/320` **skaliert**. Die
    Grenzen sind absolute Pixel aus der 320-px-Zeit; ohne die Umrechnung landete dasselbe
    Schild bei 384 px eine Stufe feiner, und der Offset klemmt am Zellenrand.
    Bei `size == 320` ist das Verhalten unverändert.
- **Augmentierung zurückgenommen**: `zoom 0,5 → 0,3`, `degrade 0,6 → 0,4`. Aggressiver
  Skalenschnitt verkleinert kleine Schilder weiter, statt sie näherzubringen; starke Störung
  löscht auf einem 20-px-Schild das Symbol, nicht nur dessen Kontrast – was übrig bleibt, ist
  Rauschen mit einer Box daran.
- **Schwellen-Suche in der Auswertung** (`tools/eval_conditions.py --sweep`): `conf`
  0,15…0,60 × NMS-IoU 0,40…0,70, Optimum **nach F1** (auf einem Split ohne Boxen nach
  `fp/Bild`). Das Netz läuft **einmal**, je `conf` einmal `decode_level`, je NMS-Paar nur
  NMS + Zuordnung – deshalb 70 Kombinationen in Sekunden statt 70 Netzlaufzeiten. Rohausgaben
  als `float16` (`--sweep-limit 400`, rund 0,8 GB). Ergebnis nach `berichte/sweep.json` und
  damit in `kaggle_report.json`; zusätzlich wird gegen den bisherigen Wert
  (conf 0,25 / NMS 0,45) gerechnet, damit die Verbesserung dasteht.

- **ONNX-Export: der Fehler aus dem Kaggle-Lauf ist behoben und darf sich nicht wiederholen.**
  Im Lauf vom 03.10. brach der Export mit `Reshape … Input shape:{1,128,24,24}, requested
  shape:{1,128,4,5,4,5}` ab, und der Kernel schlug den Schritt als **Ganzes** fehl
  („übersprungen") – 2,5 Stunden Training ohne auslieferbares Modell. Zwei Ursachen, zwei
  Maßnahmen:
  * Der `Reshape` mit fest eingebauter Fenstergröße stammte aus dem **alten
    Aufmerksamkeitsblock**. Mit CATM existiert diese Stelle nicht mehr – die Fensterteilung
    ist ersatzlos entfallen (belegt: kein `win`/`pad` mehr im Baum, `py_compile` und Export
    gegen einen frischen Checkpoint laufen durch).
  * Unabhängig davon ist der Export jetzt **abgesichert**: schlägt die Gegenprobe der
    dynamischen Achsen fehl oder wirft sie, wird **statisch** exportiert, `labels.json`
    bekommt `dynamic:false` **plus** `dynamic_fallback` mit Grund und fester Größe, ebenso
    `manifest.json`. Das Werkzeug endet mit Code 0 statt 1. Ein Modell, das nur seine
    Trainingsgröße kann, ist ungleich besser als keines.
  * `src/model.js`/`src/app.js` respektieren das: bei `labels.dynamic === false` werden
    `inputSize` und die Größenauswahl **ignoriert** und fest auf `labels.size` gerechnet –
    sonst würde in 384 px geletterboxt und in 320 px gerechnet, ohne dass ein Fehler auftritt.
  * Beide Wege sind durchgespielt: normal **Parität 4,77e-06**, dynamisch 384 px →
    Formen `[96,48,24,12]` passend; künstlich erzwungener Fehler → Rückfall auf statisch,
    Parität 3,81e-06, `dynamic:false` im `labels.json`.
- **Verifiziert in diesem Schritt** (jeweils gemessen, nicht angenommen):
  Kopfvarianten bauen und liefern **79 Kanäle** (`tgadhier` 2 785 132 / `tgadflach`
  2 780 776 / `plainhier` 2 195 212 Parameter bei 320 px); Rauchtest mit echten Daten
  (Loss-Zerlegung zeigt `fam`); Schwellen-Suche end-to-end gegen 15 Bilder inkl. JSON;
  `signmap.pruefen()` ohne Fehler; `node tests/model.test.js` und `detector.test.js` ohne FAIL.
  * Details: `data/det` lokal = 29 600 train / 2 000 val / 1 100 neg (GTSRB);
    `int8` aus dem neuen Kopf: **7,57 MB → 2,46 MB**, Abweichung 1,94e-02, **keine**
    „Expected bias"-Warnung mehr.

## 0.5.0 – 2026-10-03 (in Arbeit)
- **Architektur umgebaut** (`tools/hybrid_net.py`): die fensterbasierte Selbstattention ist
  durch den **CATM** (Convolutional Additive Token Mixer) aus CAS-ViT ersetzt
  (arXiv:2408.03703) – `out = proj(dwc(q + k) * v)`, additiv statt `q@k`, ohne Softmax, ohne
  Fenster. Die FPN-lite ist durch eine **LGP-FPN** ersetzt (granulare Wahrnehmung mit
  Tiefenconvs 3×3 **und** 5×5 additiv, plus Kontextbezug über den globalen Mittelwert); die
  seitlichen Verbindungen bleiben 1×1. Gemessen (Preset `breit`, 320×320): **2,20 Mio.
  Parameter / 1 057 MFLOPs** gegen 1,88 Mio. / 1 180 vorher – mehr Parameter, **10 % weniger
  Rechnung**. CATM sitzt jetzt auch auf `p3` (bisher CNN), weil er pro Zelle konstant teuer
  ist; genau dort landen kleine Schilder. Konfigurationsfeld `catm` (vorher `tr_global`/
  `tr_window`/`tr_heads`/`win`), CLI `--catm p5,p4,p3`.
  * **Herkunft, offen benannt:** CATM ist eine Übernahme der Referenzumsetzung. Für die
    **LGP-FPN** (Yan Zhang u. a., „A lightweight granular perception feature pyramid network
    with context-awareness for small traffic sign detection", Expert Systems with
    Applications 317:131885, 2026) ist **keine Referenzumsetzung öffentlich**; die Umsetzung
    hier folgt dem Namen und der Aufgabenstellung und ist im Quelltext als solche
    gekennzeichnet.
- **INT8 im Frontend**: `tools/export_onnx.py --int8` wird jetzt im Kaggle-Lauf benutzt
  (`--calib-data data/det --calib-n 200`, QDQ, Kalibrierung auf den echten Bildern des
  Datensatzes). Davor wird **Conv+BN verschmolzen** (`quant_pre_process`) – ohne diesen Schritt
  warnte der Quantisierer bei jeder Faltung („Expected bias … to be an initializer"), weil das
  Netz mit `bias=False` gebaut ist. Gemessen an einem Prüfmodell: **8,75 MB → 2,67 MB (−69 %)**,
  max. Tensorabweichung 1,06Â·10â»². `src/model.js` lädt INT8 zuerst und **fällt auf FP32
  zurück**, wenn es scheitert; welches läuft, steht im Status (`int8 wasm ×4`).
- **Drei Fehler behoben**, die den Kaggle-Lauf vom 03.10. **nach 28 Minuten**
  Datensatzvorbereitung beendet haben (`kaggle/train_kernel.py`, `tools/real_negatives.py`):
  * Der Katalogordner heißt `kataloge/`, im Werkzeugaufruf stand aber `catalogs/` –
    `crops_dataset.py` brach mit `FileNotFoundError: catalogs/GTSIGN-220.zip` ab, obwohl der
    Download einwandfrei gelaufen war. Der Pfad wird jetzt geprüft; fehlt eine Quelle, entfällt
    **sie einzeln** (mit Hinweiszeile) statt den ganzen Lauf zu beenden.
  * **GTSRB wurde gepackt und nie benutzt**: 39 253 Bilder in 280 MB, 3 Minuten Packzeit – und
    kein `--gtsrb` im Trainingsaufruf, 12 nutzbare Klassen lagen brach. Jetzt:
    `--gtsrb gtsrb/train.zip` im Training (die Test-Messlatte bleibt draußen, damit die
    Bewertung nicht wieder auf GTSRB-Material aufgeht).
  * **Zwei Negative-Aufrufe auf denselben Split ersetzten einander** (gleicher `src`): der
    zweite Aufruf löschte die 3 600 echten Negative des ersten, und beide schrieben dieselben
    Dateinamen (`train_r000000 …`) – im Manifest standen danach zwei Einträge für eine Datei.
    Genau so entstehen doppelte Bilder. Neu: `--src` kennzeichnet die Herkunft, die Dateinamen
    tragen sie mit.
  * Die erwarteten Bildzahlen stehen **nicht mehr fest im Skript**, sondern werden aus dem
    Rezept gerechnet – fehlende Wahlquellen (etwa die eigenen Negative, wenn ihr Upload
    scheitert) beenden den Lauf nicht mehr. Nachgerechnet und geprüft mit
    `python tests/kernel_probe.py`: 36 283 / 2 662 / 1 500.
  * Eine **leere Quelle** sprengte den Datensatzbau: `compose()` zieht jede Quelle mit ihrem
    Gewicht, und `rng.randrange(0)` wirft `ValueError`. Jetzt wird sie mit Warnzeile
    ausgelassen (`tools/crops_dataset.py`). Auslöser kann ein falsches Archiv sein – die
    GTSRB-**Test-Labels** enthalten keine Bilder und liegen trotzdem als `*.zip` bereit
    (gemessen: `gtsrb-test-gt.zip` → 0 Bilder, `GTSRB_Final_Training_Images.zip` → 39 209
    Bilder / 36 Klassen).
- **74 Klassen statt 9** (neu: [`tools/signmap.py`](tools/signmap.py)). Angelpunkt ist die
  Zuordnung: GTSRB liefert eine `ClassId`, GTSIGN-220 die **StVO-Nummer** (`274-70`), Synset
  Signset Germany den **deutschen Namen** (`Geschwindigkeit70`), GTSDB englische
  Kategorienamen – vier Schreibweisen, eine Klassenliste. Geprüft mit
  `python tools/signmap.py`: **GTSIGN deckt 68 der 74 Klassen ab, Synset 73** (nur
  `ortstafel` fehlt dort), und die 9 GTSIGN-Katalogzeilen, die übrig bleiben
  (Absperrschranke, Leitplatte, Grünpfeilschild, Abschleppzone), sind Ausstattung und keine
  Zeichentypen – sie werden verworfen statt in einen Sammeltopf geworfen.
- **Vier Kataloge im Training** (neu: [`tools/crops_dataset.py`](tools/crops_dataset.py),
  [`tools/gtsdb_dataset.py`](tools/gtsdb_dataset.py)): GTSRB (39 209 Ausschnitte),
  GTSIGN-220 (71 264 von 75 541), Synset Signset Germany (**streamend** von HuggingFace,
  kein 17,6-GB-Upload) und **GTSDB** (383 echte Szenen zum Training). Dazu **Open Images**:
  4 000 Fotos *ohne* Verkehrszeichen-Annotation, in **drei getrennte Töpfe** gelegt - als
  echte Umgebung für die Komposition und als Fehlalarm-Gegenprobe.
- **Upsampling je Quelle** (`--weight`, z. B. `gtsign=2`): der Anteil wird über die *Quelle*
  gesteuert, nicht über die Bildzahl. Vorher war ein kleiner, sauber annotierter Katalog in
  der Menge eines großen unsichtbaren. `--balance` zieht zusätzlich seltene Klassen häufiger.
- **Ehrliche Messlatte**: die Auswertung läuft jetzt auf Bildern, die im Training **nicht**
  vorkommen - GTSIGN-`val` (7 038 Ausschnitte), Synset-`validation` und GTSDB
  `valid`+`test` (162 **echte Szenen**, Schild klein im Bild). Die Überschneidung der
  GTSIGN-Split-Listen ist **geprüft 0** (`data/_kaggle_probe.py`). Die frühere val-Zahl
  0,903 stammt aus GTSRB-Material und ist mit dieser Messlatte **nicht vergleichbar**; der
  Vergleich mit dem alten Checkpoint ist deshalb abgeschaltet (`ALT_VERGLEICH = False`) -
  ein 9-Klassen-Netz kann die neuen 74 Klassen nicht ausgeben.
- **Tempo im Browser**, gemessen statt geschätzt:
  * Die Umrechnung der Bilddaten läuft über eine 256er-Tabelle statt Division je Pixel:
    **5,09 ms → 2,18 ms** je 320-px-Bild (**57 %**, node-Referenzmessung).
  * Ausgabenamen werden einmal aufgelöst statt je Bild über `map`.
  * Was schon stand und jetzt auch dokumentiert ist: COOP/COEP über `dist/_headers` macht
    `crossOriginIsolated` wahr → 4 Rechenfäden statt 1 (**18,6 ms → 9,7 ms** je Bild).
- **Absichtlich nicht genommen:** Mapillary MTSD (105 000 Bilder, 400 Klassen - fachlich der
  beste, aber die Research-Use-Lizenz verbietet den Einbau in ein öffentliches Produkt),
  TT100K und BDD100K (CC BY-**NC**). Begründung und Prüfweg in
  [`docs/DATENSAETZE.md`](docs/DATENSAETZE.md) §3.
- **Architektur an die Klassenzahl angepasst** (neues Preset `breit` in `tools/hybrid_net.py`):
  mit 74 statt 9 Klassen entscheidet sich die Art im **Kopf**, also bekommt der
  Klassenzweig den vollen Merkmalsvorrat. Warum das genau `mid = Rumpfbreite` bedeutet: der
  Kopf beginnt mit einer **Tiefenconvolution** (`cba(cin, mid, g=mid)`), deshalb muss `mid`
  die Rumpfbreite *teilen* – und die Rumpfbreite selbst ist der größte zulässige Wert. Dazu
  ein Block mehr in den tiefen Stufen (`depth=(1,2,3,3)`, dort stehen nur 20×20 und 10×10
  Zellen, kostet also fast nichts) und **lokale** Attention auf p4 statt der globalen aus
  `quality` (Fenster 5×5 gegen ~256× teurere globale Attention auf stride 8).
  Gemessen: **1,88 Mio. Parameter / 1 180 MFLOPs** gegen 1,32 Mio. / 916 bei `balanced`
  (+42 % / +29 %). Die Zeit ist da: der letzte Lauf brauchte **70 von 540** Kaggle-Minuten.
- **Was der Lauf noch nicht zeigt:** ob 74 Klassen und das breitere Netz die Qualität halten.
  Das ist eine Messung, keine Zusage – die Zahlen kommen mit `kaggle\run.ps1 -Step pull` und
  gehören dann in `models/README.md`. Bekannt und gemeldet: von 383 GTSDB-Trainingsbildern
  hatten **50** nur Zeichen unter 6 px nach dem Einpassen auf 320 px.

## 0.4.1 – 2026-10-02
- **Kaggle-Lauf Block 5 ausgewertet und übernommen** (Tesla T4, 220 Epochen × 150 Schritte,
  29 600 Trainingsbilder mit 3 600 echten Negativen und den neuen Tafel-Szenen, 70 min):
  auf den unveränderten 2 000 val-Bildern **P = 0,944 / R = 0,865 / F1 = 0,903**
  (vorher 0,883 / 0,796 / 0,837) und bei 384 px Eingabe **0,934 / 0,879 / F1 = 0,906**.
  Parität PyTorch â†” ONNX 3,5 Â· 10â»â´ mit 23/23 Erkennungen IoU â‰¥ 0,95, ONNX Runtime CPU
  (x86, 1 Thread, 320×320) 6,9 ms je Bild. Zahlen kommen mit
  `kaggle\run.ps1 -Step pull` / `-Step install` nach `models/` und `tests/fixtures/`.
  Die Tabellen in `docs/TRAINING.md` §5/§10 zeigen noch **den Stand der Vorfassung** –
  das Nachziehen ist offen (PLAN.md §3.4).
- **Ausgeliefert und im Netz geprüft:** neue Modell-Dateien in `models/`, Fixture in
  `tests/fixtures/model-out.json` (beide Node-Tests grün), `dist/` gebaut und auf Cloudflare
  gelegt. Der Weg ohne Konto (temporärer Wrangler-Zugang, zufälliger Name) steht mit den
  gemessenen Eigenheiten in der README (`workers.dev`-Adresse, 60-Minuten-Claim,
  403 für `urllib`, 307 auf `/index.html`).
- **Bekannt und unverändert:** auf den beiden Nutzerfotos bleibt es schwach –
  `Test/Schilder.jpg` (Poster, Schilder ~15 px) liefert **0** Treffer, `Test/Nothing.jpg`
  **1–2** schwache Fehlalarme (0,26–0,36). Die Domänen- und Auflösungslücke aus PLAN.md §1
  ist mit diesem Lauf **nicht** geschlossen; 1 100 Negativbilder ergeben 0,25 Fehlalarme je
  Bild (277 Treffer auf 1 100 Bildern).

## 0.4.0 – 2026-09-30
- **Fehlerzerlegung vor dem Umbau (neu: `tools/eval_conditions.py --diagnose`)**: von 656
  verpassten Boxen waren 334 „tief verpasst“ (46 % davon ≤ 32 px Diagonale), 212 falsch
  klassifiziert und 110 falsch lokalisiert; von 226 Fehlalarmen lagen 195 auf echten
  Schildern und nur 31 auf freier Fläche. Die Treffer hatten schon IoU 0,938. Damit war
  klar, wo die Arbeit hingeht – und wo nicht (ein CIoU-Verlust wäre wirkungslos, weil die
  Treffer schon sitzen; eine Klassen-Arbitrierung nach der NMS brachte gemessen nur
  +0,7 % Precision).
- **Vier Erkennungsstufen statt drei** (`stride 4/8/16/32`, `tools/detmath.py`).
  Zusätzlich geändert: die Stufe wird nach der **längsten Objektseite** gewählt
  (`ASSIGN_MAX_SIDE`) statt nach `log2` der Diagonale – die alte Regel schickte ein
  30-px-Schild auf `stride 32` mit nur 10×10 Zellen. Und bei Randlage lernt die
  **Nachbarzelle** mit (`dual`), weil der Offset vorher auf 0,999 festgeklemmt war.
  Kosten: 731 → 878 MFLOPs.
- **Getrennter Kopf (Box/Objektivität gegen Art)** in `tools/hybrid_net.py`: der Kopf
  verwechselte Arten (rotes Dreieck Spitze oben gegen unten 32×). Kostet ~4 MFLOPs.
- **Daten: Klassenausgleich, Roll-Politik, Negative.**
  `tools/gtsrb_dataset.py --balance` (jeder Typ ~3 600 Boxen statt `verbot` 50 %),
  `roll_angle()` dreht **Dreiecke nur ±35°** (stark gedrehte Dreiecke sind widersprüchliche
  Ziele: `warnung` verlor dadurch 18 Recall-Punkte), neu
  **`tools/synth_negatives.py`** für Bilder ohne Schild (3000 train + 500 im neuen Split
  `neg` als Messlatte) und mehr Hintergrundarten.
- **Mehrskaliges Training** (`--zoom`, Bereich 0,7–1,5) und `--obj-norm` in
  `tools/train_det.py`; `--val-split neg` misst während des Trainings Fehlalarme.
- **Ergebnis (2 000 val-Bilder)**: P 0,891 → **0,883**, R 0,739 → **0,796**,
  F1 0,808 → **0,837**; verpasste Boxen 656 → **513**, Fehlalarme 226 → 264. Der Zugewinn
  liegt bei den Typen mit wenigen Beispielen (`stop` R 0,478 → 0,717, `vorfahrtGewaehren`
  0,434 → 0,689, `einfahrtVerboten` 0,532 → 0,806), der Preis bei `verbot` (0,801 → 0,779).
  Alle Tabellen in `docs/TRAINING.md` §5 und §10.
- **Neu: Größe wählbar** – `tools/export_onnx.py --dynamic` (eine Datei für 256/320/384/448 px),
  `src/model.js` nimmt die Größe entgegen, die App bekommt ein Auswahlfeld
  (automatisch: Video 320 px, Foto 384 px). Gemessen: 256 px F1 0,796 / 320 px 0,837 /
  **384 px 0,854 bei 21 % weniger Fehlalarmen**.
- **`models/labels.json`** trägt jetzt `dynamic`; `tools/export_onnx.py --size` erlaubt
  zusätzlich einen statischen Export in einer festen Größe (auf x86 rund 59 % schneller
  als mit offenen Achsen).
- **Checkpoints der Vorfassung sind nicht mehr ladbar** (Kopfzweige umbenannt).

## 0.3.0 – 2026-09-30
- **Erstes echtes Modell trainiert** (bisher gab es nur die Kette dafür): `--preset balanced`
  (1 287 066 Params, 731 MFLOPs) in drei Blöcken über 159 Epochen auf **23 000 Bildern**
  aus GTSRB-Kompositionen plus 3 000 synthetischen Bildern. Gemessen auf **2 000
  Validierungsbildern: P = 0,891 / R = 0,739 / F1 = 0,808**; Export 5,16 MB fp32,
  Parität 1,1 Â· 10â»â´ mit 4/4 Erkennungen IoU â‰¥ 0,95, ONNX Runtime CPU (x86, 1 Thread,
  320×320) 7,2 ms je Bild. Alle Zahlen und die Schwachstellen in `docs/TRAINING.md` §5.
- **GTSRB-Lücke geschlossen:** GTSRB enthält keine blauen Hinweiszeichen (Z 3xx) und keine
  gelbe Ortstafel (Z 310) – zwei der neun Typen hatten null Trainingsdaten. Neues
  `tools/synth_missing.py` füllt genau diese Klassen synthetisch nach; gemessen danach
  `hinweis` P/R = 0,840/0,781 und `ortstafel` 0,971/0,745 (vorher: nie vorhergesagt).
- **Neu: `tools/eval_conditions.py`** – Precision/Recall je Bedingung (aus
  `manifest.conditions`) und je Klasse, mit `--json` für die Ablage. Ersetzt den als offen
  markierten Punkt „Auswertung nach Bedingung" und macht Verbesserungen messbar.
- **`tools/train_det.py`:** `--device auto|cpu|cuda` (CUDA wird erkannt, Checkpoints bleiben
  portabel), `--eval-every` wirkt jetzt wirklich (war ein toter Schalter – jede Epoche
  kostete eine volle Auswertung), Geraetewechsel beim `--resume` berücksichtigt.
- **`tools/gtsrb_dataset.py`:** `--gt-zip` für das Test-Set (Bilder und Labels liegen dort in
  **zwei** Archiven; die labelose `GT-final_test.test.csv` im Bildarchiv lieferte vorher
  stillschweigend null Zeilen), `--size`-Default 320 (Modellgröße), Boxen werden korrekt
  durch die Letterbox gerechnet, wenn Datensatzgröße ≠ Trainingsgröße.
- **Doku:** README (KI-Modus mit echtem Rezept und Messwerten), `docs/TRAINING.md`
  (§4.3 Datenweg, §5.1 Trainingsläufe, §5.2 Bedingungs-/Klassentabelle, §6 Parität/int8,
  §8 Status, §9 nächste Schritte), `models/README.md`, Changelog.
- **int8 bleibt verworfen** – jetzt auch mit 200 echten Kalibrierbildern gemessen
  (Abweichung 3,16 · 10¹). Das Qualitätstor greift, ausgeliefert wird fp32.

## 0.2.0 – 2026-09-30
- **KI-Modus (optional):** hybrides Netz mit **CNN-Backbone und Transformer-Stufen**
  (`tools/hybrid_net.py`) sagt **Position und Art** in einem Durchlauf voraus – anchor-free,
  drei Stufen (stride 8/16/32), 9 Typen, 14 Kanäle je Zelle. Globale Attention nur auf
  stride 32, lokale auf stride 16; die feinen Stufen bleiben CNN (Begründung + Messwerte
  in `docs/TRAINING.md`).
- **Werkzeuge:** `tools/synth_data.py` (synthetische Schilder mit genau den Bedingungen, die
  die Heuristik brechen), `tools/train_det.py` (Training mit `--resume`, Precision/Recall bei
  IoU 0,5), `tools/detmath.py` (Letterbox, Zuweisung, Decode, NMS – Vorlage für JS),
  `tools/bench_model.py` (Parameter/FLOPs/Latenz je Variante), `tools/selfcheck.py`
  (Überfittet das Netz ein Bild?), `tools/export_onnx.py` (ONNX + **Paritätscheck** +
  int8 mit Qualitätstor), `tools/export_fixture.py` (Testfixture aus echtem Modelllauf).
- **Browser:** `src/model.js` mit Letterbox, Decodierung, NMS und ONNX Runtime Web
  (WebGPU mit WASM-Rückfall, einthreadig wegen fehlender COOP/COEP-Header auf GitHub Pages).
  `src/app.js` wählt automatisch Modell oder Heuristik und zeigt den Modus im Status.
- **Tests:** `tests/model.test.js` prüft die Modell-Mathematik und vergleicht sie über
  `tests/fixtures/model-out.json` mit der Python-Seite (gleiche Klassen, Boxen, Scores).
- **Grenzen ehrlich dokumentiert:** int8 wurde mit wenigen Kalibrierbildern als unbrauchbar
  gemessen und wird deshalb automatisch verworfen; Laufzeit auf echten Handys und
  Erkennungsqualität auf Straßenbildern sind noch nicht gemessen.
- Doku: neue `docs/TRAINING.md`, README, `docs/DOKUMENTATION.md` (§7/§8) und `models/README.md`
  angepasst. Heuristik unverändert und weiterhin der Standard/Rückfall.

## 0.1.0 – 2026-09-30
- Erste Version: Live-Erkennung von neun Schildtypen über Farbe und Form, Foto-Import, Farbmasken-Ansicht, Stabilisierung über mehrere Bilder.

