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

`hybridnano_gP5_w-_silu` (`--preset balanced`), 1 287 066 Parameter, 731 MFLOPs,
**5,16 MB fp32** – trainiert in drei Blöcken (159 Epochen) auf 23 000 Bildern aus
GTSRB-Kompositionen plus 3 000 synthetischen Bildern für `hinweis`/`ortstafel`.

| Kennzahl | Wert |
|---|---|
| Precision / Recall / F1 (2 000 val-Bilder, conf 0,25, IoU 0,5) | **0,891 / 0,739 / 0,808** |
| ONNX ↔ PyTorch: max. Tensorabweichung, Trefferquote | 1,1 · 10⁻⁴, 4/4 Erkennungen mit IoU ≥ 0,95 |
| Laufzeit ONNX Runtime CPU (x86, 1 Thread, 320×320) | 7,2 ms je Bild |
| int8 | **verworfen** (Qualitätstor, Abweichung 3,16 · 10¹ auch mit 200 echten Kalibrierbildern) |
| Tests | `node tests/model.test.js` (Fixture aus genau diesem Modell) und `node tests/detector.test.js` grün |

Details, Verlauf und die schwachen Stellen (z.B. `vorfahrtGewaehren` gegen `warnung`):
`docs/TRAINING.md` §5.

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
