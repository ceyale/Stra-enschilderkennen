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

Kurzweg zu einem ersten Modell (synthetische Daten, kein Datensatz nötig):

```
python tools/synth_data.py --out data/synth --n 400
python tools/train_det.py --data synth --epochs 30 --steps 100 --batch 8
python tools/export_onnx.py --ckpt models/signs-det.pt --int8 --calib-n 200
```

Mit echten Bildern: `data/det` nach `docs/TRAINING.md` aufbauen und
`--data data/det` verwenden. `examples/` enthält den Stand des Rauchtests
(schwach trainiert, nur als Beleg für Export und Paritätscheck).
