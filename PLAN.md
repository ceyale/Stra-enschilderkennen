# PLAN – Qualität auf echten Fotos (Stand: Block 5, nicht abgeschlossen)

Kurzfassung des aktuellen Umbaus. Gemessene Zahlen stehen in
[`TRAINING.md`](TRAINING.md) §5/§10; dieses Blatt ist die Arbeitsliste.

## 1. Was gemessen wurde (Fotos des Nutzers, `data/_test_bilder.py`)

| Test | Ergebnis |
|---|---|
| `Test/Nothing.jpg` (Zimmer/Monitor, **kein** Schild) | 3 Fehlalarme bei conf 0,25, Boxen **380×305 bis 1971×1811 px** (halbes Bild), Bestwerte `ortstafel` 0,36 / `einfahrtVerboten` 0,32 |
| `Test/Schilder.jpg` (Poster, ~80 Schilder) | **0 Treffer**, Bestwert 0,19 |
| derselbe Checkpoint auf `data/det` val (2 000 Bilder) | P 0,883 / R 0,796 / F1 0,837, Treffer-IoU 0,954 |

**Schluss:** Die Architektur ist nicht das Problem – es fehlt die **Domäne** (echte Fotos ohne
Schild, abfotografierte Anzeige mit Moiré) und die **Auflösung** für kleine Schilder.

### Kachelmessung (dieselben Fotos)

Die Schilder sind im 4096 px breiten Poster ~15 px groß → bei 320 px Eingang 1,2 px.

| Foto | ganzes Bild | 3×3 Kacheln | 6×6 Kacheln |
|---|---|---|---|
| `Schilder.jpg` | 0 | 28 (beste 0,51) | **115 (beste 0,88)** |
| `Nothing.jpg` | 3 FP | 3 FP | 22 FP (beste 0,60) → conf 0,3 lässt nur **5** |

⇒ Kacheln bringen die Erkennung, kosten aber Fehlalarme → braucht höhere Schwelle.

## 2. Was bereits gebaut ist

* `tools/real_negatives.py` – echte Fotos → viele Zufallsausschnitte mit **leerem Label**.
  Trennung ohne Selbstbetrug: `--train-region`/`--test-region` teilen **dasselbe** Foto
  (`Nothing.jpg`: obere Hälfte Training, untere Hälfte Messlatte); coco128 in 114 Training /
  12 gesperrte Testfotos. COCO-Bilder mit Klasse 11 (stop sign) fliegen raus.
* `data/negatives/fetch_coco.ps1` – holt coco128 (128 echte Fotos, CC BY 4.0, 7 MB).
* `tools/synth_data.py` – neue Szene `screen_panel` (Schildertafel/Anzeige: 3–34 Schilder im
  Raster, 15–160 px, Moiré/Gammaschlag/Kanalversatz/Vignette), Helfer `place_sign`
  (Boxrechnung nur an einer Stelle), neue Negativ-Arten `hart:bildschirm` und `hart:tastatur`.
* `tools/synth_missing.py --scene-prob` – Anteil Tafel-Szene.
* `tools/synth_negatives.py` – kennt die neuen Arten über `HARD_KINDS`.
* `.gitignore` – **Modell wird jetzt eingecheckt** (`models/signs-det.pt`, `*.onnx`, `*.json`).
* Datensatz `data/det` (gebaut mit `data/_build2.ps1`): **32 700 Bilder**, 7 700 Negative.
  Train 29 600 = GTSRB 20 000 + Lückenschluss 3 000 (Tafel) + synth. Negative 3 000 +
  **echte Negative 3 600**. Split `neg` 1 100. **`val` 2 000 unverändert** (Messlatte).
* `data/_test_bilder.py --tile N`, `data/_check_synth.py` (`data/_commit5.txt` entfernt).

## 3. Nächste Schritte

1. **Trainingslauf Block 5** (Rezept wie `data/train_signs4.ps1`, aber auf dem neuen
   Datensatz): 220 Epochen, ~1,3 h bei 29 600 Bildern.
   *Hinweis:* Der Vorabtest mit 600 Schritten (`data/_probe2.ps1`) zeigte val `tp=0` – das ist
   normal für so wenige Schritte (der Vorlauf hatte bei Epoche 9 R = 0,029) und **kein Urteil**.
2. **Kachelmodus in die App**: `src/model.js` + `src/app.js` – Foto in N×N überlappende
   Kacheln, Treffer verschieben, NMS über die Kachelgrenzen, **höhere Schwelle** im Kachelmodus
   (Messung: conf 0,3 → 40 statt 115 Treffer auf dem Poster, aber 5 statt 22 FP auf `Nothing`).
3. **Hard-Negative-Mining**: mit dem neuen Modell über den Negativ-Pool laufen und die
   schlimmsten Fehltreffer selbst als Trainingsbilder nachliefern (begrenzter Nutzen war
   gemessen, jetzt aber die echte Baustelle: Riesenboxen auf Anzeigen).
4. **Endabnahme**: `eval_conditions --split val` (unverändert vergleichbar) + `--split neg`
   (echte Negative!) + `--diagnose`, ONNX-Export mit dynamischen Achsen, Fixture, Node-Tests,
   Website bauen, Doku + CHANGELOG nachziehen, veröffentlichen.

## 4. Fallstricke, die nicht vergessen werden dürfen

* `val` **nicht** anfassen – er ist die einzige Zahl, die mit den früheren Läufen vergleichbar ist.
* `Schilder.jpg` **nie** trainieren (nur der untere Streifen von `Nothing.jpg` und 114 der 126
  coco-Fotos sind Trainingsmaterial).
* Kachelmodus muss in **Python und JS identisch** umgesetzt sein (sonst rechnet der Browser
  anders als die Messung) – die Regel „eine Rechnung, zwei Sprachen“ gilt auch hier.
* Alte Checkpoints sind nicht ladbar (Kopfzweige umbenannt); ein Lauf muss von Grund auf neu.
* int8 bleibt verworfen (Qualitätstor), ausgeliefert wird fp32.

## 5. Offene Fragen an den Nutzer

* **Kachelmodus im Video erlaubt?** 6×6 Kacheln = 36 Modellläufe je Bild (bei 150 ms je Lauf
  zu langsam für Live); sinnvoll wäre 2×2 oder nur der Foto-Modus.
* **Sollen falsch erkannte Typen wichtiger sein als verpasste?** Das steuert, ob wir im
  Kachelmodus die Schwelle hochziehen (weniger Fehlalarme, weniger Treffer).