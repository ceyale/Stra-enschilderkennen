# models/

Hier liegen die **ausgelieferten** Modell-Dateien. Sie werden lokal erzeugt und sind
absichtlich nicht im Git (siehe `.gitignore`).

| Datei | erzeugt von | gelesen von |
|---|---|---|
| `signs-det.onnx` (oder `signs-det-int8.onnx`) | `tools/export_onnx.py` | `src/model.js` |
| `labels.json` | `tools/export_onnx.py` | `src/model.js` (Klassen, Eingabegröße, Stufen) |
| `manifest.json` | `tools/export_onnx.py` | Menschen (Metriken, Parität, sha256) |

Fehlen diese Dateien, läuft die App im **Heuristik-Modus** (`src/detector.js`) und zeigt
das im Status an – sie bricht nicht.

## Stand des hier erzeugten Modells

`hybridnano_gP5_w-_silu` (`--preset balanced`), **vier** Erkennungsstufen
(stride 4/8/16/32), 1 300 360 Parameter, 878 MFLOPs, **5,23 MB fp32** – trainiert in
einem Lauf über 220 Epochen auf **26 000 Bildern** (20 000 ausgeglichene
GTSRB-Kompositionen, 3 000 synthetische Bilder für `hinweis`/`ortstafel`, 3 000 Bilder
**ohne** Schild).

| Kennzahl | Wert |
|---|---|
| Precision / Recall / F1 (2 000 val-Bilder, conf 0,25, IoU 0,5, 320 px) | **0,883 / 0,796 / 0,837** |
| dasselbe Modell bei 384 px Eingabe | **0,907 / 0,808 / 0,854** (fp/Bild 0,104) |
| Fehlalarme je Bild (2 000 val-Bilder) | 0,132 |
| Fehlalarme auf Bildern ohne Schild (Split `neg`, 500 Bilder) | 0,076 gesamt, davon 0,011 auf reinem Hintergrund |
| Treffer-IoU (Mittel) | 0,954 |
| ONNX ↔ PyTorch: max. Tensorabweichung, Trefferquote | 2,1 · 10⁻⁴, 9/9 Erkennungen mit IoU ≥ 0,95 |
| Laufzeit ONNX Runtime CPU (x86, 1 Thread, 320×320) | 10–18 ms je Bild (dynamische Achsen) |
| Laufzeit im Browser (wasm, echtes Gerät, 320 px) | Vorfassung gemessen: 122 ms – neue Zahl steht aus (die App zeigt sie an) |
| int8 | **verworfen** (Qualitätstor, Abweichung 3,16 · 10¹ auch mit 200 echten Kalibrierbildern) |
| Tests | `node tests/model.test.js` (Fixture aus genau diesem Modell) und `node tests/detector.test.js` grün |

Details, Verlauf, Klassentabelle und die Fehlerzerlegung (z. B. `hinweis` hat die
schwächste Precision): `docs/TRAINING.md` §5 und §10.

Kurzweg zu einem ersten Modell:

```
python tools/synth_data.py --out data/synth --n 400
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8
python tools/export_onnx.py --ckpt models/signs-det.pt --int8 --calib-n 200
```

Mit echten Bildern: `data/det` nach `docs/TRAINING.md` §4.3 aufbauen (GTSRB holen,
`--gt-zip` nicht vergessen, Lücken mit `tools/synth_missing.py` füllen) und
`--data data/det` verwenden. `examples/` enthält den Stand des Rauchtests
(schwach trainiert, nur als Beleg für Export und Paritätscheck).
