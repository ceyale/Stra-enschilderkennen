"""
tools/export_fixture.py - Testfixture erzeugen: echte Modellausgabe + erwartete Erkennungen.

Damit wird die JavaScript-Seite (src/model.js) gegen die Python-Seite (tools/detmath.py)
festgenagelt: das Fixture enthaelt die rohen ONNX-Tensoren und die daraus in Python
berechneten Erkennungen. tests/model.test.js dekodiert dieselben Tensoren in JS und
muss dieselben Boxen erhalten - sonst laeuft der Browser anders als das Training.

Enthalten ist bewusst ein NICHT quadratisches Bild, damit auch Letterbox und
Rueckrechnung geprueft werden.

Aufruf: python tools/export_fixture.py --ckpt models/signs-det.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import detmath as dm
import synth_data as sd
from detmath import LETTERBOX_GREY
from hybrid_net import SIGN_LABELS
from export_onnx import load_model, ort_run, ort_session


def make_input(size: int, seed: int, ratio: float = 0.75):
    """Synthetisches Bild, beschnitten auf ein anderes Seitenverhaeltnis, dann Letterbox."""
    arr, _, _ = sd.compose_sample(random.Random(seed), size, degrade_prob=0.7)
    h, w = arr.shape[:2]
    arr = arr[:, : int(w * ratio)]
    boxed, s, px, py = dm.letterbox_array(arr, size)
    return boxed, arr.shape[1], arr.shape[0], s, px, py


def main() -> None:
    ap = argparse.ArgumentParser(description="Testfixture aus echtem Modelllauf erzeugen")
    ap.add_argument("--ckpt", default="models/signs-det.pt")
    ap.add_argument("--onnx", default="models/signs-det.onnx")
    ap.add_argument("--out", default="tests/fixtures/model-out.json")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    model, cfg, ck = load_model(args.ckpt)
    size = int(ck.get("size", 320))
    if not Path(args.onnx).exists():
        raise SystemExit(f"{args.onnx} fehlt - erst python tools/export_onnx.py laufen lassen")
    sess = ort_session(Path(args.onnx))

    cases = []
    for k, seed in enumerate((11, 12)):
        boxed, src_w, src_h, s, px, py = make_input(size, seed)
        x = np.ascontiguousarray(boxed.transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        raw = ort_run(sess, x)
        tensors = [{"dims": list(r.shape), "data": [round(float(v), 5) for v in r.reshape(-1)]}
                   for r in raw]
        dets = dm.decode_multi([r[0] for r in raw], dm.LEVELS, args.conf, args.iou)
        image_coords = []
        for d in dets:
            back = dm.boxes_from_letterbox(np.array([[d["x0"], d["y0"], d["x1"], d["y1"]]]), s, px, py)[0]
            image_coords.append({"labelIdx": d["label_idx"], "score": round(float(d["score"]), 5),
                                 "x0": round(float(back[0]), 3), "y0": round(float(back[1]), 3),
                                 "x1": round(float(back[2]), 3), "y1": round(float(back[3]), 3)})
        cases.append({
            "source": {"width": int(src_w), "height": int(src_h)},
            "letterbox": {"s": round(float(s), 6), "padX": int(px), "padY": int(py)},
            "tensors": tensors,
            "expected_model_coords": [{"labelIdx": d["label_idx"], "score": round(float(d["score"]), 5),
                                       "x0": round(float(d["x0"]), 3), "y0": round(float(d["y0"]), 3),
                                       "x1": round(float(d["x1"]), 3), "y1": round(float(d["y1"]), 3)} for d in dets],
            "expected_image_coords": image_coords,
        })

    fixture = {"arch": cfg.name(), "size": size, "levels": list(dm.LEVELS), "classes": SIGN_LABELS,
               "conf": args.conf, "iou": args.iou, "note": "rohe ONNX-Ausgaben + Python-Dekodierung",
               "cases": cases}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    n = sum(len(c["expected_model_coords"]) for c in cases)
    print(f"[fixture] {out}  {out.stat().st_size/1024:.0f} KB  Faelle={len(cases)}  "
          f"erwartete Erkennungen={n}  (Grauwert Rand={LETTERBOX_GREY})")


if __name__ == "__main__":
    main()
