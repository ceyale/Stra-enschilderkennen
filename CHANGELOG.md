# Changelog

## 0.8.0 â€“ 2026-10-03
- **fp16 fÃ¼r WebGPU** (`tools/export_onnx.py --fp16`, im Kaggle-Lauf jetzt mitgeschaltet):
  halbe Datei fÃ¼r den Weg, der sie nativ rechnen kann. Gemessen am Artefakt (320 px, fp32
  5,23 MB): fp16 **2,66 MB**, Ein-/AusgÃ¤nge bleiben **fp32** (`keep_io_types` â€“ der Browser
  muss seine Tensoren nicht umrechnen, und die KanalvertrÃ¤ge `os4/os8/os16/os32` bleiben wie
  erwartet), maximale Tensorabweichung **8,9 Â· 10â»Â³**. Auf ONNX Runtime **CPU** ist fp16
  langsamer (11,6 gegen 8,3 ms) â€“ genau deshalb wird sie nicht als Ersatz ausgeliefert.
  Wie bei int8 gilt das **QualitÃ¤tstor**: hÃ¤lt die ParitÃ¤t nicht, wird die Datei verworfen
  und im Log steht der Grund. `models/labels.json`/`manifest.json` fÃ¼hren sie in `files`
  (mit sha256); `kaggle/train_kernel.py` meldet ihre GrÃ¶ÃŸe im Ergebnis.
- **`src/model.js` wÃ¤hlt die Fassung nach der AusfÃ¼hrungsart, nicht global** â€“ die
  Reihenfolge ist jetzt begrÃ¼ndet und getestet statt â€žHauptsache irgendein RÃ¼ckfall":
  mit `navigator.gpu` zuerst **fp16/webgpu**, dann **int8**, dann **fp32**; ohne WebGPU
  direkt **int8 â†’ fp32**. Der Status zeigt den Klartext (`fp16 webgpu`, `int8 wasm Ã—4`, â€¦).
- **Zweiter Lehrer (optional): `kelvinandreas/vit-traffic-sign-GTSRB`** â€“ der GTSRB-ViT
  (43 Klassen, Acc 0,985 / F1 0,985, **MIT**) ist auf seinen Klassen genauer als der
  Hauptlehrer, deckt aber nur **36 unserer 74** Typen ab (GTSIGN-220: 68). Er bleibt deshalb
  zweiter Lehrer und wird **0,5/0,5 gemittelt â€“ gefiltert**: nur bei Boxen, deren
  **Grundwahrheitsklasse** zu seinen 36 gehÃ¶rt. Ungefiltert schriebe ein Lehrer, der
  â€žtempo40" nicht kennt, seine Meinung â€žtempo30" als Lernziel in den Cache.
  Nachgewiesen in `tests/teacher_probe.py` (Lehrer durch feste Zahlen ersetzt, ohne Netz):
  2 von 4 Boxen gemittelt, `tempo50` â†’ 0,5Â·0,8 + 0,5Â·0,9, `tempo40` unverÃ¤ndert.
  Rezeptschalter `LEHRER["zweiter"]` (Standard **aus**: erst muss der Hauptlehrer allein
  messbar wirken; ein zweites 86-Mio.-Modell verdoppelt auÃŸerdem die Cachedauer).
  Lizenz-Hinweis: bei zwei Lehrern sind **beide** Lizenzen zu nennen â€“ der Lauf warnt selbst.
- **RezeptprÃ¼fung erweitert** (`tests/kernel_probe.py`, Teil 4): der Exportbefehl muss
  `--int8` **und** `--fp16` tragen, `tools/export_onnx.py` muss beide Flags kennen (ein
  unbekanntes Flag beendet argparse mit Code 2 â€“ der Lauf hÃ¤tte danach kein Modell) und
  `src/model.js` muss beide Fassungen auch **laden** (sonst liegt eine Datei im Ordner, die
  niemand benutzt). Alle vier Teile grÃ¼n.
- **Zwei stille Fehler dabei gefunden** (beide von den neuen PrÃ¼fungen aufgedeckt, nicht im
  Feld): das Klassenfeld der Labeldatei steht in **Spalte 0** (`cls cx cy w h`), nicht ab
  Spalte 5 â€“ die Filterung hÃ¤tte sonst nie gegriffen; und der Cache wird als **float16**
  gespeichert, weshalb ein Vergleich â€ž0,8 exakt" fehlschlÃ¤gt (0,79980â€¦) â€“ die PrÃ¼fung
  vergleicht jetzt mit passender Toleranz.

## 0.7.0 â€“ 2026-10-03
- **Wissens-Distillation von einem ViT-Lehrer** â€“ der wirksamste Einzelhebel gegen den
  eigentlichen Fehler des v6-Laufs (Boxen gelernt, Arten nicht: Klassifikationsverlust blieb
  bei 3,4, Zufall wÃ¤re ln 74 â‰ˆ 4,3; 195 von 226 Fehlalarmen lagen auf *echten* Schildern mit
  falscher Klasse).
  * **Lehrer:** `vit_gtsign_all_classes` aus dem GTSIGN-220-Datensatz â€“ `google/vit-base-patch16-224`,
    86 Mio. Parameter, feinjustiert auf **220 deutsche StVO-Klassen**, verÃ¶ffentlicht mit
    Accuracy 0,973 / P 0,911 / R 0,930 (Werte aus `eval_results.json` des Repos gelesen, nicht
    geschÃ¤tzt). Lizenz **CC BY-SA 4.0** â€“ die Weitergabe-Bedingung steht im Bericht
    (`kaggle_report.json â†’ lehrer.lizenz`) und in `docs/TRAINING.md`.
  * **Warum genau dieser Lehrer:** ein auf COCO trainierter Detektor kennt nur â€žstop sign"
    (eine Klasse). Dieser kennt dieselben Zeichen wie wir, nur feiner geteilt (220 statt 74) â€“
    deshalb ist seine Zuordnung ein **Nachschlagen der StVO-Nummer** und keine Vermutung:
    **211 von 220** Lehrer-Klassen lassen sich zuordnen, sie decken **68 der 74** unserer
    Klassen ab. Ohne Lehrer bleiben: `tempo110`, `zone20`, `mindestgeschwindigkeit`,
    `gebotLinks`, `gebotGeradeaus`, `umleitung`.
  * **Neu `tools/teacher.py`:** rechnet die Lehrer-Verteilung **einmal je Grundwahrheitsbox**
    (Ausschnitt +25 % Rand, damit Achtkant-Ecken und Dreiecksspitzen nicht abgeschnitten
    werden), marginalisiert auf unsere 74 Klassen (Summe der Feinklassen â€“ nicht Maximum, sonst
    ginge Masse verloren) und legt sie als `data/teacher/teacher_<split>.npz` (+ `.json` mit
    Modell, Lizenz, Zuordnung) ab. Nur der **Trainings**split â€“ `val`/`neg` bleiben Messlatte.
  * **Neu `tools/train_det.py --teacher --distill --temperature`:** KL(Lehrerâ€–SchÃ¼ler) auf den
    positiven Zellen, mit **TÂ²** skaliert (Hinton u. a. 2015 â€“ ohne TÂ² wÃ¤re die Wirkung bei
    T=2 nur ein Viertel und die Gewichtsangabe bedeutete etwas anderes als sie sagt). Die
    Zuordnung Zelleâ†’Box lÃ¤uft Ã¼ber die neue `boxidx`-Ebene in `tools/detmath.py`, damit zwei
    Schilder derselben Klasse ihre **eigene** Verteilung bekommen kÃ¶nnen.
  * **Gewicht gemessen statt geraten:** eine positive Zelle trÃ¤gt KLÃ—TÂ² von **2â€“9** bei
    (PrÃ¼fung mit kÃ¼nstlichem Ziel: 9,02), wÃ¤hrend der Klassifikationsverlust bei 0,1â€“0,3 und der
    Box-Verlust bei ~5 liegt. Deshalb **0,25** und nicht 1,0 â€“ mit 1,0 hÃ¤tte der Lehrer die
    Ã¼brigen Verluste Ã¼berstimmt.
  * **Zwei Fehler dabei gefunden und behoben** (beide vor dem Kaggle-Lauf, mit Nachweis):
    `ViTForImageClassification.from_pretrained(<repo-URL>)` scheitert, weil das Modell in einem
    **Dataset**-Repo liegt (`hf_hub_download(repo_type="dataset")` nÃ¶tig); und das Repo liefert
    **keine** Bildvorverarbeitung (`preprocessor_config.json` fehlt, 404) â€“ sie kommt jetzt von
    `google/vit-base-patch16-224` mit festen Werten als letzter Stufe.
  * **Regression im eigenen Umbau gefunden und behoben:** ein zu frÃ¼h gesetztes `continue` in
    `collate()` Ã¼bersprang die *gesamte* Zielzuweisung, sobald kein Lehrer-Cache vorlag
    (`pos = 0`). Die Ziele werden jetzt **vor** allem Lehrer-Zeug gesetzt. Beweis:
    `pos je Stufe [0, 6, 2, 8]`, `obj je Stufe [0.0, 6.0, 2.0, 8.0]`.
- **Lokale RezeptprÃ¼fung erweitert** (`tests/kernel_probe.py`, Teil 3): sie prÃ¼ft jetzt auch,
  dass der Cache gefunden wird, `--teacher/--distill/--temperature` im Trainingsbefehl stehen
  (und *ohne* Cache nicht), dass das Gewicht im sinnvollen Bereich liegt und jeder Eingabepfad
  unter `WORK` existiert. Alle drei Teile grÃ¼n: `alle Pruefungen bestanden`.

## 0.6.0 â€“ 2026-10-03
- **Erkennungskopf ersetzt: TGADHead** (`tools/hybrid_net.py`) â€“ aufgabengefÃ¼hrter,
  entkoppelter Kopf aus Zuo, Liu, Chen, Fu, Wang, *â€žTGADHead: An efficient and accurate
  task-guided attention-decoupled head for single-stage object detection"*, Knowledge-Based
  Systems **302:112349 (2024)**. Zwei Bausteine:
  * **TDAD** (Task Decoupled Attention Distributor): zwei aufgabenspezifische
    Aufmerksamkeits-Wahrnehmungen. Der **Ort**-Zweig gewichtet Ã¼ber die *Zellen* (Tiefenconv
    3Ã—3 â†’ 1 Kanal â†’ Sigmoid), der **Art**-Zweig Ã¼ber die *KanÃ¤le* (globaler Mittelwert â†’ 1Ã—1
    â†’ Sigmoid). Getrennt, weil eine gemeinsame Gewichtung scharf-lokal und weich-global
    gleichzeitig sein mÃ¼sste â€“ zusammen mittelt sich das zu etwas, das keinem von beiden dient.
  * **TCN** (Task Correlation Network): Austausch *zwischen* den Zweigen
    (`ort += gate(1Ã—1(art))`, `art += gate(1Ã—1(ort))`, Tor startet bei Sigmoid 0,27). Der
    Abstract begrÃ¼ndet ihn damit, dass genaue Ortung und hoher Klassenscore in bestehenden
    Detektoren auseinandergehen â€“ genau das war hier gemessen: 195 von 226 Fehlalarmen lagen
    auf echten Schildern.
  * **Herkunft, offen benannt:** es gibt **keine Ã¶ffentliche Referenzumsetzung**; der Code ist
    nach dem im Abstract beschriebenen Aufbau geschrieben und **nicht aus dem Paper
    abgetippt** â€“ Abweichungen im Detail sind mÃ¶glich. Steht so im Quelltext. Gemessen
    (Preset `breit`, 320 px): **2,79 Mio. Parameter** gegen 2,20 Mio. des VorgÃ¤ngers, also
    +0,59 Mio. fÃ¼r Aufgabentrennung und Austausch.
- **Hierarchischer Klassifikationskopf**: statt einer flachen Entscheidung Ã¼ber 74 Klassen
  jetzt zwei Stufen â€“ **9 Oberkategorien** (Familie: Form/Farbe) und **74 Unterkategorien**
  (Symbol im Inneren), verrechnet als `logit_k = super[familie(k)] + sub[k]`, in
  Log-Wahrscheinlichkeiten also `P(k) = P(Familie)Â·P(k|Familie)`.
  * Die Hierarchie steht in `tools/signmap.py` bei den Klassen (**eine** Taxonomie) und ist
    verteilt: **74 = 5 + 18 + 3 + 7 + 19 + 7 + 4 + 2 + 9** (Vorfahrt, Tempo, Ãœberholen, Verbot,
    Warnung, Gebot, Rad/FuÃŸ, Zone, Hinweis). `signmap.pruefen()` erzwingt jetzt, dass jede
    Unterkategorie genau einer Familie zugeordnet ist â€“ ein Tippfehler wÃ¤re sonst erst als
    Klasse aufgefallen, die nie erkannt wird.
  * **Der Ausgangsvertrag bleibt unverÃ¤ndert** (79 KanÃ¤le, `os4/os8/os16/os32`): die Summe
    steht im selben Tensor, `tools/detmath.py` und `src/model.js` sind nicht angefasst.
  * Die **Familie wird zusÃ¤tzlich direkt Ã¼berwacht** (`--hier-aux 0.3`, Hilfsverlust auf
    denselben positiven Zellen). Ohne ihn bekÃ¤me die grobe Entscheidung nur mittelbar ein
    Signal.
- **Focal Loss im Klassifikationskopf** (`tools/train_det.py`) mit **Klassengewichten** und
  **Label-Smoothing**:
  * `Î³=2`, `Î±=0,25`; Klassen `w_c = (1/hÃ¤ufigkeit_c)^0,5`, auf Mittelwert 1 normiert
    (gedÃ¤mpft â€“ `1/f` hÃ¤tte den Kopf in die Gegenrichtung kippen lassen), Smoothing `0,075`.
  * **Î± wirkt hier nur als konstanter Faktor** (der Kopf rechnet ausschlieÃŸlich auf positiven
    Zellen, es gibt keine Negativklasse auszugleichen) und wÃ¼rde den Kopf still auf ein
    Viertel drosseln. Deshalb gleicht `--cls-w 4.0` das aus (0,25 Ã— 4,0 = 1,0 wie vorher) â€“
    im Kernel mit BegrÃ¼ndung hinterlegt.
  * Die HÃ¤ufigkeiten kommen aus den **Labeldateien des Trainingssplits**, nicht aus `val`
    (sonst steckte eine Messlatte im Verlust). Gemessen am lokalen Datensatz:
    `tempo10=10175` ist die hÃ¤ufigste, Gewichte `0,65 â€¦ 1,11`.
- **TrainingsauflÃ¶sung 320 â†’ 384 px** (`kaggle/train_kernel.py`, `DATEN` *und* `TRAINING`).
  Grund: der Median der verpassten Objekte liegt bei 35 px Diagonale â€“ bei 320 px ist das auf
  `stride 32` noch **ein** Pixel breit. Datensatz *und* Training mÃ¼ssen dieselbe Zahl
  benutzen; `--size` wird jetzt auch an `synth_negatives.py` und `real_negatives.py`
  durchgereicht (vorher schrieben die mit dem 320er-Default). Die zweite MessgrÃ¶ÃŸe wandert
  entsprechend auf **448 px**.
  * Dazu in `tools/detmath.py`: `ASSIGN_MAX_SIDE` wird mit `size/320` **skaliert**. Die
    Grenzen sind absolute Pixel aus der 320-px-Zeit; ohne die Umrechnung landete dasselbe
    Schild bei 384 px eine Stufe feiner, und der Offset klemmt am Zellenrand.
    Bei `size == 320` ist das Verhalten unverÃ¤ndert.
- **Augmentierung zurÃ¼ckgenommen**: `zoom 0,5 â†’ 0,3`, `degrade 0,6 â†’ 0,4`. Aggressiver
  Skalenschnitt verkleinert kleine Schilder weiter, statt sie nÃ¤herzubringen; starke StÃ¶rung
  lÃ¶scht auf einem 20-px-Schild das Symbol, nicht nur dessen Kontrast â€“ was Ã¼brig bleibt, ist
  Rauschen mit einer Box daran.
- **Schwellen-Suche in der Auswertung** (`tools/eval_conditions.py --sweep`): `conf`
  0,15â€¦0,60 Ã— NMS-IoU 0,40â€¦0,70, Optimum **nach F1** (auf einem Split ohne Boxen nach
  `fp/Bild`). Das Netz lÃ¤uft **einmal**, je `conf` einmal `decode_level`, je NMS-Paar nur
  NMS + Zuordnung â€“ deshalb 70 Kombinationen in Sekunden statt 70 Netzlaufzeiten. Rohausgaben
  als `float16` (`--sweep-limit 400`, rund 0,8 GB). Ergebnis nach `berichte/sweep.json` und
  damit in `kaggle_report.json`; zusÃ¤tzlich wird gegen den bisherigen Wert
  (conf 0,25 / NMS 0,45) gerechnet, damit die Verbesserung dasteht.

- **ONNX-Export: der Fehler aus dem Kaggle-Lauf ist behoben und darf sich nicht wiederholen.**
  Im Lauf vom 03.10. brach der Export mit `Reshape â€¦ Input shape:{1,128,24,24}, requested
  shape:{1,128,4,5,4,5}` ab, und der Kernel schlug den Schritt als **Ganzes** fehl
  (â€žÃ¼bersprungen") â€“ 2,5 Stunden Training ohne auslieferbares Modell. Zwei Ursachen, zwei
  MaÃŸnahmen:
  * Der `Reshape` mit fest eingebauter FenstergrÃ¶ÃŸe stammte aus dem **alten
    Aufmerksamkeitsblock**. Mit CATM existiert diese Stelle nicht mehr â€“ die Fensterteilung
    ist ersatzlos entfallen (belegt: kein `win`/`pad` mehr im Baum, `py_compile` und Export
    gegen einen frischen Checkpoint laufen durch).
  * UnabhÃ¤ngig davon ist der Export jetzt **abgesichert**: schlÃ¤gt die Gegenprobe der
    dynamischen Achsen fehl oder wirft sie, wird **statisch** exportiert, `labels.json`
    bekommt `dynamic:false` **plus** `dynamic_fallback` mit Grund und fester GrÃ¶ÃŸe, ebenso
    `manifest.json`. Das Werkzeug endet mit Code 0 statt 1. Ein Modell, das nur seine
    TrainingsgrÃ¶ÃŸe kann, ist ungleich besser als keines.
  * `src/model.js`/`src/app.js` respektieren das: bei `labels.dynamic === false` werden
    `inputSize` und die GrÃ¶ÃŸenauswahl **ignoriert** und fest auf `labels.size` gerechnet â€“
    sonst wÃ¼rde in 384 px geletterboxt und in 320 px gerechnet, ohne dass ein Fehler auftritt.
  * Beide Wege sind durchgespielt: normal **ParitÃ¤t 4,77e-06**, dynamisch 384 px â†’
    Formen `[96,48,24,12]` passend; kÃ¼nstlich erzwungener Fehler â†’ RÃ¼ckfall auf statisch,
    ParitÃ¤t 3,81e-06, `dynamic:false` im `labels.json`.
- **Verifiziert in diesem Schritt** (jeweils gemessen, nicht angenommen):
  Kopfvarianten bauen und liefern **79 KanÃ¤le** (`tgadhier` 2 785 132 / `tgadflach`
  2 780 776 / `plainhier` 2 195 212 Parameter bei 320 px); Rauchtest mit echten Daten
  (Loss-Zerlegung zeigt `fam`); Schwellen-Suche end-to-end gegen 15 Bilder inkl. JSON;
  `signmap.pruefen()` ohne Fehler; `node tests/model.test.js` und `detector.test.js` ohne FAIL.
  * Details: `data/det` lokal = 29 600 train / 2 000 val / 1 100 neg (GTSRB);
    `int8` aus dem neuen Kopf: **7,57 MB â†’ 2,46 MB**, Abweichung 1,94e-02, **keine**
    â€žExpected bias"-Warnung mehr.

## 0.5.0 â€“ 2026-10-03 (in Arbeit)
- **Architektur umgebaut** (`tools/hybrid_net.py`): die fensterbasierte Selbstattention ist
  durch den **CATM** (Convolutional Additive Token Mixer) aus CAS-ViT ersetzt
  (arXiv:2408.03703) â€“ `out = proj(dwc(q + k) * v)`, additiv statt `q@k`, ohne Softmax, ohne
  Fenster. Die FPN-lite ist durch eine **LGP-FPN** ersetzt (granulare Wahrnehmung mit
  Tiefenconvs 3Ã—3 **und** 5Ã—5 additiv, plus Kontextbezug Ã¼ber den globalen Mittelwert); die
  seitlichen Verbindungen bleiben 1Ã—1. Gemessen (Preset `breit`, 320Ã—320): **2,20 Mio.
  Parameter / 1 057 MFLOPs** gegen 1,88 Mio. / 1 180 vorher â€“ mehr Parameter, **10 % weniger
  Rechnung**. CATM sitzt jetzt auch auf `p3` (bisher CNN), weil er pro Zelle konstant teuer
  ist; genau dort landen kleine Schilder. Konfigurationsfeld `catm` (vorher `tr_global`/
  `tr_window`/`tr_heads`/`win`), CLI `--catm p5,p4,p3`.
  * **Herkunft, offen benannt:** CATM ist eine Ãœbernahme der Referenzumsetzung. FÃ¼r die
    **LGP-FPN** (Yan Zhang u. a., â€žA lightweight granular perception feature pyramid network
    with context-awareness for small traffic sign detection", Expert Systems with
    Applications 317:131885, 2026) ist **keine Referenzumsetzung Ã¶ffentlich**; die Umsetzung
    hier folgt dem Namen und der Aufgabenstellung und ist im Quelltext als solche
    gekennzeichnet.
- **INT8 im Frontend**: `tools/export_onnx.py --int8` wird jetzt im Kaggle-Lauf benutzt
  (`--calib-data data/det --calib-n 200`, QDQ, Kalibrierung auf den echten Bildern des
  Datensatzes). Davor wird **Conv+BN verschmolzen** (`quant_pre_process`) â€“ ohne diesen Schritt
  warnte der Quantisierer bei jeder Faltung (â€žExpected bias â€¦ to be an initializer"), weil das
  Netz mit `bias=False` gebaut ist. Gemessen an einem PrÃ¼fmodell: **8,75 MB â†’ 2,67 MB (âˆ’69 %)**,
  max. Tensorabweichung 1,06Â·10â»Â². `src/model.js` lÃ¤dt INT8 zuerst und **fÃ¤llt auf FP32
  zurÃ¼ck**, wenn es scheitert; welches lÃ¤uft, steht im Status (`int8 wasm Ã—4`).
- **Drei Fehler behoben**, die den Kaggle-Lauf vom 03.10. **nach 28 Minuten**
  Datensatzvorbereitung beendet haben (`kaggle/train_kernel.py`, `tools/real_negatives.py`):
  * Der Katalogordner heiÃŸt `kataloge/`, im Werkzeugaufruf stand aber `catalogs/` â€“
    `crops_dataset.py` brach mit `FileNotFoundError: catalogs/GTSIGN-220.zip` ab, obwohl der
    Download einwandfrei gelaufen war. Der Pfad wird jetzt geprÃ¼ft; fehlt eine Quelle, entfÃ¤llt
    **sie einzeln** (mit Hinweiszeile) statt den ganzen Lauf zu beenden.
  * **GTSRB wurde gepackt und nie benutzt**: 39 253 Bilder in 280 MB, 3 Minuten Packzeit â€“ und
    kein `--gtsrb` im Trainingsaufruf, 12 nutzbare Klassen lagen brach. Jetzt:
    `--gtsrb gtsrb/train.zip` im Training (die Test-Messlatte bleibt drauÃŸen, damit die
    Bewertung nicht wieder auf GTSRB-Material aufgeht).
  * **Zwei Negative-Aufrufe auf denselben Split ersetzten einander** (gleicher `src`): der
    zweite Aufruf lÃ¶schte die 3 600 echten Negative des ersten, und beide schrieben dieselben
    Dateinamen (`train_r000000 â€¦`) â€“ im Manifest standen danach zwei EintrÃ¤ge fÃ¼r eine Datei.
    Genau so entstehen doppelte Bilder. Neu: `--src` kennzeichnet die Herkunft, die Dateinamen
    tragen sie mit.
  * Die erwarteten Bildzahlen stehen **nicht mehr fest im Skript**, sondern werden aus dem
    Rezept gerechnet â€“ fehlende Wahlquellen (etwa die eigenen Negative, wenn ihr Upload
    scheitert) beenden den Lauf nicht mehr. Nachgerechnet und geprÃ¼ft mit
    `python tests/kernel_probe.py`: 36 283 / 2 662 / 1 500.
  * Eine **leere Quelle** sprengte den Datensatzbau: `compose()` zieht jede Quelle mit ihrem
    Gewicht, und `rng.randrange(0)` wirft `ValueError`. Jetzt wird sie mit Warnzeile
    ausgelassen (`tools/crops_dataset.py`). AuslÃ¶ser kann ein falsches Archiv sein â€“ die
    GTSRB-**Test-Labels** enthalten keine Bilder und liegen trotzdem als `*.zip` bereit
    (gemessen: `gtsrb-test-gt.zip` â†’ 0 Bilder, `GTSRB_Final_Training_Images.zip` â†’ 39 209
    Bilder / 36 Klassen).
- **74 Klassen statt 9** (neu: [`tools/signmap.py`](tools/signmap.py)). Angelpunkt ist die
  Zuordnung: GTSRB liefert eine `ClassId`, GTSIGN-220 die **StVO-Nummer** (`274-70`), Synset
  Signset Germany den **deutschen Namen** (`Geschwindigkeit70`), GTSDB englische
  Kategorienamen â€“ vier Schreibweisen, eine Klassenliste. GeprÃ¼ft mit
  `python tools/signmap.py`: **GTSIGN deckt 68 der 74 Klassen ab, Synset 73** (nur
  `ortstafel` fehlt dort), und die 9 GTSIGN-Katalogzeilen, die Ã¼brig bleiben
  (Absperrschranke, Leitplatte, GrÃ¼npfeilschild, Abschleppzone), sind Ausstattung und keine
  Zeichentypen â€“ sie werden verworfen statt in einen Sammeltopf geworfen.
- **Vier Kataloge im Training** (neu: [`tools/crops_dataset.py`](tools/crops_dataset.py),
  [`tools/gtsdb_dataset.py`](tools/gtsdb_dataset.py)): GTSRB (39 209 Ausschnitte),
  GTSIGN-220 (71 264 von 75 541), Synset Signset Germany (**streamend** von HuggingFace,
  kein 17,6-GB-Upload) und **GTSDB** (383 echte Szenen zum Training). Dazu **Open Images**:
  4 000 Fotos *ohne* Verkehrszeichen-Annotation, in **drei getrennte TÃ¶pfe** gelegt - als
  echte Umgebung fÃ¼r die Komposition und als Fehlalarm-Gegenprobe.
- **Upsampling je Quelle** (`--weight`, z. B. `gtsign=2`): der Anteil wird Ã¼ber die *Quelle*
  gesteuert, nicht Ã¼ber die Bildzahl. Vorher war ein kleiner, sauber annotierter Katalog in
  der Menge eines groÃŸen unsichtbaren. `--balance` zieht zusÃ¤tzlich seltene Klassen hÃ¤ufiger.
- **Ehrliche Messlatte**: die Auswertung lÃ¤uft jetzt auf Bildern, die im Training **nicht**
  vorkommen - GTSIGN-`val` (7 038 Ausschnitte), Synset-`validation` und GTSDB
  `valid`+`test` (162 **echte Szenen**, Schild klein im Bild). Die Ãœberschneidung der
  GTSIGN-Split-Listen ist **geprÃ¼ft 0** (`data/_kaggle_probe.py`). Die frÃ¼here val-Zahl
  0,903 stammt aus GTSRB-Material und ist mit dieser Messlatte **nicht vergleichbar**; der
  Vergleich mit dem alten Checkpoint ist deshalb abgeschaltet (`ALT_VERGLEICH = False`) -
  ein 9-Klassen-Netz kann die neuen 74 Klassen nicht ausgeben.
- **Tempo im Browser**, gemessen statt geschÃ¤tzt:
  * Die Umrechnung der Bilddaten lÃ¤uft Ã¼ber eine 256er-Tabelle statt Division je Pixel:
    **5,09 ms â†’ 2,18 ms** je 320-px-Bild (**57 %**, node-Referenzmessung).
  * Ausgabenamen werden einmal aufgelÃ¶st statt je Bild Ã¼ber `map`.
  * Was schon stand und jetzt auch dokumentiert ist: COOP/COEP Ã¼ber `dist/_headers` macht
    `crossOriginIsolated` wahr â†’ 4 RechenfÃ¤den statt 1 (**18,6 ms â†’ 9,7 ms** je Bild).
- **Absichtlich nicht genommen:** Mapillary MTSD (105 000 Bilder, 400 Klassen - fachlich der
  beste, aber die Research-Use-Lizenz verbietet den Einbau in ein Ã¶ffentliches Produkt),
  TT100K und BDD100K (CC BY-**NC**). BegrÃ¼ndung und PrÃ¼fweg in
  [`docs/DATENSAETZE.md`](docs/DATENSAETZE.md) Â§3.
- **Architektur an die Klassenzahl angepasst** (neues Preset `breit` in `tools/hybrid_net.py`):
  mit 74 statt 9 Klassen entscheidet sich die Art im **Kopf**, also bekommt der
  Klassenzweig den vollen Merkmalsvorrat. Warum das genau `mid = Rumpfbreite` bedeutet: der
  Kopf beginnt mit einer **Tiefenconvolution** (`cba(cin, mid, g=mid)`), deshalb muss `mid`
  die Rumpfbreite *teilen* â€“ und die Rumpfbreite selbst ist der grÃ¶ÃŸte zulÃ¤ssige Wert. Dazu
  ein Block mehr in den tiefen Stufen (`depth=(1,2,3,3)`, dort stehen nur 20Ã—20 und 10Ã—10
  Zellen, kostet also fast nichts) und **lokale** Attention auf p4 statt der globalen aus
  `quality` (Fenster 5Ã—5 gegen ~256Ã— teurere globale Attention auf stride 8).
  Gemessen: **1,88 Mio. Parameter / 1 180 MFLOPs** gegen 1,32 Mio. / 916 bei `balanced`
  (+42 % / +29 %). Die Zeit ist da: der letzte Lauf brauchte **70 von 540** Kaggle-Minuten.
- **Was der Lauf noch nicht zeigt:** ob 74 Klassen und das breitere Netz die QualitÃ¤t halten.
  Das ist eine Messung, keine Zusage â€“ die Zahlen kommen mit `kaggle\run.ps1 -Step pull` und
  gehÃ¶ren dann in `models/README.md`. Bekannt und gemeldet: von 383 GTSDB-Trainingsbildern
  hatten **50** nur Zeichen unter 6 px nach dem Einpassen auf 320 px.

## 0.4.1 â€“ 2026-10-02
- **Kaggle-Lauf Block 5 ausgewertet und Ã¼bernommen** (Tesla T4, 220 Epochen Ã— 150 Schritte,
  29 600 Trainingsbilder mit 3 600 echten Negativen und den neuen Tafel-Szenen, 70 min):
  auf den unverÃ¤nderten 2 000 val-Bildern **P = 0,944 / R = 0,865 / F1 = 0,903**
  (vorher 0,883 / 0,796 / 0,837) und bei 384 px Eingabe **0,934 / 0,879 / F1 = 0,906**.
  ParitÃ¤t PyTorch â†” ONNX 3,5 Â· 10â»â´ mit 23/23 Erkennungen IoU â‰¥ 0,95, ONNX Runtime CPU
  (x86, 1 Thread, 320Ã—320) 6,9 ms je Bild. Zahlen kommen mit
  `kaggle\run.ps1 -Step pull` / `-Step install` nach `models/` und `tests/fixtures/`.
  Die Tabellen in `docs/TRAINING.md` Â§5/Â§10 zeigen noch **den Stand der Vorfassung** â€“
  das Nachziehen ist offen (PLAN.md Â§3.4).
- **Ausgeliefert und im Netz geprÃ¼ft:** neue Modell-Dateien in `models/`, Fixture in
  `tests/fixtures/model-out.json` (beide Node-Tests grÃ¼n), `dist/` gebaut und auf Cloudflare
  gelegt. Der Weg ohne Konto (temporÃ¤rer Wrangler-Zugang, zufÃ¤lliger Name) steht mit den
  gemessenen Eigenheiten in der README (`workers.dev`-Adresse, 60-Minuten-Claim,
  403 fÃ¼r `urllib`, 307 auf `/index.html`).
- **Bekannt und unverÃ¤ndert:** auf den beiden Nutzerfotos bleibt es schwach â€“
  `Test/Schilder.jpg` (Poster, Schilder ~15 px) liefert **0** Treffer, `Test/Nothing.jpg`
  **1â€“2** schwache Fehlalarme (0,26â€“0,36). Die DomÃ¤nen- und AuflÃ¶sungslÃ¼cke aus PLAN.md Â§1
  ist mit diesem Lauf **nicht** geschlossen; 1 100 Negativbilder ergeben 0,25 Fehlalarme je
  Bild (277 Treffer auf 1 100 Bildern).

## 0.4.0 â€“ 2026-09-30
- **Fehlerzerlegung vor dem Umbau (neu: `tools/eval_conditions.py --diagnose`)**: von 656
  verpassten Boxen waren 334 â€žtief verpasstâ€œ (46 % davon â‰¤ 32 px Diagonale), 212 falsch
  klassifiziert und 110 falsch lokalisiert; von 226 Fehlalarmen lagen 195 auf echten
  Schildern und nur 31 auf freier FlÃ¤che. Die Treffer hatten schon IoU 0,938. Damit war
  klar, wo die Arbeit hingeht â€“ und wo nicht (ein CIoU-Verlust wÃ¤re wirkungslos, weil die
  Treffer schon sitzen; eine Klassen-Arbitrierung nach der NMS brachte gemessen nur
  +0,7 % Precision).
- **Vier Erkennungsstufen statt drei** (`stride 4/8/16/32`, `tools/detmath.py`).
  ZusÃ¤tzlich geÃ¤ndert: die Stufe wird nach der **lÃ¤ngsten Objektseite** gewÃ¤hlt
  (`ASSIGN_MAX_SIDE`) statt nach `log2` der Diagonale â€“ die alte Regel schickte ein
  30-px-Schild auf `stride 32` mit nur 10Ã—10 Zellen. Und bei Randlage lernt die
  **Nachbarzelle** mit (`dual`), weil der Offset vorher auf 0,999 festgeklemmt war.
  Kosten: 731 â†’ 878 MFLOPs.
- **Getrennter Kopf (Box/ObjektivitÃ¤t gegen Art)** in `tools/hybrid_net.py`: der Kopf
  verwechselte Arten (rotes Dreieck Spitze oben gegen unten 32Ã—). Kostet ~4 MFLOPs.
- **Daten: Klassenausgleich, Roll-Politik, Negative.**
  `tools/gtsrb_dataset.py --balance` (jeder Typ ~3 600 Boxen statt `verbot` 50 %),
  `roll_angle()` dreht **Dreiecke nur Â±35Â°** (stark gedrehte Dreiecke sind widersprÃ¼chliche
  Ziele: `warnung` verlor dadurch 18 Recall-Punkte), neu
  **`tools/synth_negatives.py`** fÃ¼r Bilder ohne Schild (3000 train + 500 im neuen Split
  `neg` als Messlatte) und mehr Hintergrundarten.
- **Mehrskaliges Training** (`--zoom`, Bereich 0,7â€“1,5) und `--obj-norm` in
  `tools/train_det.py`; `--val-split neg` misst wÃ¤hrend des Trainings Fehlalarme.
- **Ergebnis (2 000 val-Bilder)**: P 0,891 â†’ **0,883**, R 0,739 â†’ **0,796**,
  F1 0,808 â†’ **0,837**; verpasste Boxen 656 â†’ **513**, Fehlalarme 226 â†’ 264. Der Zugewinn
  liegt bei den Typen mit wenigen Beispielen (`stop` R 0,478 â†’ 0,717, `vorfahrtGewaehren`
  0,434 â†’ 0,689, `einfahrtVerboten` 0,532 â†’ 0,806), der Preis bei `verbot` (0,801 â†’ 0,779).
  Alle Tabellen in `docs/TRAINING.md` Â§5 und Â§10.
- **Neu: GrÃ¶ÃŸe wÃ¤hlbar** â€“ `tools/export_onnx.py --dynamic` (eine Datei fÃ¼r 256/320/384/448 px),
  `src/model.js` nimmt die GrÃ¶ÃŸe entgegen, die App bekommt ein Auswahlfeld
  (automatisch: Video 320 px, Foto 384 px). Gemessen: 256 px F1 0,796 / 320 px 0,837 /
  **384 px 0,854 bei 21 % weniger Fehlalarmen**.
- **`models/labels.json`** trÃ¤gt jetzt `dynamic`; `tools/export_onnx.py --size` erlaubt
  zusÃ¤tzlich einen statischen Export in einer festen GrÃ¶ÃŸe (auf x86 rund 59 % schneller
  als mit offenen Achsen).
- **Checkpoints der Vorfassung sind nicht mehr ladbar** (Kopfzweige umbenannt).

## 0.3.0 â€“ 2026-09-30
- **Erstes echtes Modell trainiert** (bisher gab es nur die Kette dafÃ¼r): `--preset balanced`
  (1 287 066 Params, 731 MFLOPs) in drei BlÃ¶cken Ã¼ber 159 Epochen auf **23 000 Bildern**
  aus GTSRB-Kompositionen plus 3 000 synthetischen Bildern. Gemessen auf **2 000
  Validierungsbildern: P = 0,891 / R = 0,739 / F1 = 0,808**; Export 5,16 MB fp32,
  ParitÃ¤t 1,1 Â· 10â»â´ mit 4/4 Erkennungen IoU â‰¥ 0,95, ONNX Runtime CPU (x86, 1 Thread,
  320Ã—320) 7,2 ms je Bild. Alle Zahlen und die Schwachstellen in `docs/TRAINING.md` Â§5.
- **GTSRB-LÃ¼cke geschlossen:** GTSRB enthÃ¤lt keine blauen Hinweiszeichen (Z 3xx) und keine
  gelbe Ortstafel (Z 310) â€“ zwei der neun Typen hatten null Trainingsdaten. Neues
  `tools/synth_missing.py` fÃ¼llt genau diese Klassen synthetisch nach; gemessen danach
  `hinweis` P/R = 0,840/0,781 und `ortstafel` 0,971/0,745 (vorher: nie vorhergesagt).
- **Neu: `tools/eval_conditions.py`** â€“ Precision/Recall je Bedingung (aus
  `manifest.conditions`) und je Klasse, mit `--json` fÃ¼r die Ablage. Ersetzt den als offen
  markierten Punkt â€žAuswertung nach Bedingung" und macht Verbesserungen messbar.
- **`tools/train_det.py`:** `--device auto|cpu|cuda` (CUDA wird erkannt, Checkpoints bleiben
  portabel), `--eval-every` wirkt jetzt wirklich (war ein toter Schalter â€“ jede Epoche
  kostete eine volle Auswertung), Geraetewechsel beim `--resume` berÃ¼cksichtigt.
- **`tools/gtsrb_dataset.py`:** `--gt-zip` fÃ¼r das Test-Set (Bilder und Labels liegen dort in
  **zwei** Archiven; die labelose `GT-final_test.test.csv` im Bildarchiv lieferte vorher
  stillschweigend null Zeilen), `--size`-Default 320 (ModellgrÃ¶ÃŸe), Boxen werden korrekt
  durch die Letterbox gerechnet, wenn DatensatzgrÃ¶ÃŸe â‰  TrainingsgrÃ¶ÃŸe.
- **Doku:** README (KI-Modus mit echtem Rezept und Messwerten), `docs/TRAINING.md`
  (Â§4.3 Datenweg, Â§5.1 TrainingslÃ¤ufe, Â§5.2 Bedingungs-/Klassentabelle, Â§6 ParitÃ¤t/int8,
  Â§8 Status, Â§9 nÃ¤chste Schritte), `models/README.md`, Changelog.
- **int8 bleibt verworfen** â€“ jetzt auch mit 200 echten Kalibrierbildern gemessen
  (Abweichung 3,16 Â· 10Â¹). Das QualitÃ¤tstor greift, ausgeliefert wird fp32.

## 0.2.0 â€“ 2026-09-30
- **KI-Modus (optional):** hybrides Netz mit **CNN-Backbone und Transformer-Stufen**
  (`tools/hybrid_net.py`) sagt **Position und Art** in einem Durchlauf voraus â€“ anchor-free,
  drei Stufen (stride 8/16/32), 9 Typen, 14 KanÃ¤le je Zelle. Globale Attention nur auf
  stride 32, lokale auf stride 16; die feinen Stufen bleiben CNN (BegrÃ¼ndung + Messwerte
  in `docs/TRAINING.md`).
- **Werkzeuge:** `tools/synth_data.py` (synthetische Schilder mit genau den Bedingungen, die
  die Heuristik brechen), `tools/train_det.py` (Training mit `--resume`, Precision/Recall bei
  IoU 0,5), `tools/detmath.py` (Letterbox, Zuweisung, Decode, NMS â€“ Vorlage fÃ¼r JS),
  `tools/bench_model.py` (Parameter/FLOPs/Latenz je Variante), `tools/selfcheck.py`
  (Ãœberfittet das Netz ein Bild?), `tools/export_onnx.py` (ONNX + **ParitÃ¤tscheck** +
  int8 mit QualitÃ¤tstor), `tools/export_fixture.py` (Testfixture aus echtem Modelllauf).
- **Browser:** `src/model.js` mit Letterbox, Decodierung, NMS und ONNX Runtime Web
  (WebGPU mit WASM-RÃ¼ckfall, einthreadig wegen fehlender COOP/COEP-Header auf GitHub Pages).
  `src/app.js` wÃ¤hlt automatisch Modell oder Heuristik und zeigt den Modus im Status.
- **Tests:** `tests/model.test.js` prÃ¼ft die Modell-Mathematik und vergleicht sie Ã¼ber
  `tests/fixtures/model-out.json` mit der Python-Seite (gleiche Klassen, Boxen, Scores).
- **Grenzen ehrlich dokumentiert:** int8 wurde mit wenigen Kalibrierbildern als unbrauchbar
  gemessen und wird deshalb automatisch verworfen; Laufzeit auf echten Handys und
  ErkennungsqualitÃ¤t auf StraÃŸenbildern sind noch nicht gemessen.
- Doku: neue `docs/TRAINING.md`, README, `docs/DOKUMENTATION.md` (Â§7/Â§8) und `models/README.md`
  angepasst. Heuristik unverÃ¤ndert und weiterhin der Standard/RÃ¼ckfall.

## 0.1.0 â€“ 2026-09-30
- Erste Version: Live-Erkennung von neun Schildtypen Ã¼ber Farbe und Form, Foto-Import, Farbmasken-Ansicht, Stabilisierung Ã¼ber mehrere Bilder.

