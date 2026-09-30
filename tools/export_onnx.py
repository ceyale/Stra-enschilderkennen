"""
tools/export_onnx.py - Trainiertes Netz nach ONNX ausgeben und die Ausgabe pruefen.

Schritte:
  1. Checkpoint laden (models/signs-det.pt oder --ckpt), Netz rekonstruieren.
  2. ONNX-Export mit statischer Eingabe (1,3,size,size) und drei Ausgaengen os8/os16/os32.
  3. Paritaetscheck: PyTorch vs. ONNX Runtime auf denselben Bildern - sowohl rohe
     Tensoren (max. Abweichung) als auch die fertigen Erkennungen (Klassen + Boxen).
     Ohne diesen Schritt kann der Browser still andere Ergebnisse liefern als das Training.
  4. Optional statische int8-Quantisierung mit Kalibrierbildern.
  5. models/labels.json + models/manifest.json schreiben (Klassen, Groesse, Levels,
     Decode-Konstanten, Groesse, sha256, Metriken) - das liest src/model.js.

Aufruf:
    python tools/export_onnx.py --ckpt models/signs-det.pt --out models/signs-det.onnx
    python tools/export_onnx.py --ckpt models/smoke.pt --int8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import detmath as dm
import synth_data as sd
from hybrid_net import SIGN_LABELS, NetCfg, build_model, n_params


def load_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = NetCfg(**{k: v for k, v in ck["cfg"].items() if k in NetCfg.__dataclass_fields__})
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def calib_batches(size: int, n: int, seed: int = 1234, data_dir: str | None = None):
    """Kalibrierbilder fuer die int8-Quantisierung.

    Wichtig: auf der Verteilungs des Einsatzes kalibrieren. Mit --calib-data nimmt der
    Kalibrierer echte Bilder aus dem Detektor-Datensatz (data/det), sonst synthetische.
    """
    if data_dir:
        root = Path(data_dir)
        items = sorted((root / "images").glob("*.jpg")) + sorted((root / "images").glob("*.png"))
        if not items:
            raise SystemExit(f"keine Bilder in {root/'images'}")
        rng = random.Random(seed)
        for i in range(n):
            arr = np.asarray(Image.open(items[rng.randrange(len(items))]).convert("RGB"), dtype=np.uint8)
            boxed, _, _, _ = dm.letterbox_array(arr, size)
            yield np.ascontiguousarray(boxed.transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        return
    rng = random.Random(seed)
    for _ in range(n):
        arr, _, _ = sd.compose_sample(rng, size, degrade_prob=0.8)
        x = np.ascontiguousarray(arr.transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        yield x


class CalibReader:
    def __init__(self, size: int, n: int, data_dir: str | None = None):
        self.it = calib_batches(size, n, data_dir=data_dir)

    def get_next(self):
        try:
            return {"images": next(self.it)}
        except StopIteration:
            return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(model, size: int, out_path: Path) -> None:
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["images"], output_names=["os8", "os16", "os32"],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )


def ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_run(sess, x: np.ndarray) -> list[np.ndarray]:
    names = [o.name for o in sess.get_outputs()]
    return sess.run(names, {"images": x})


def parity_check(model, sess, size: int, n: int = 6, seed: int = 99) -> dict:
    """PyTorch vs. ONNX: rohe Tensoren UND fertige Erkennungen vergleichen."""
    rng = random.Random(seed)
    max_diff = 0.0
    n_ref = n_onnx = n_match = 0
    for _ in range(n):
        arr, _, _ = sd.compose_sample(rng, size, degrade_prob=0.8)
        x = np.ascontiguousarray(arr.transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        with torch.no_grad():
            ref = [o.numpy()[0] for o in model(torch.from_numpy(x))]
        got = [o[0] for o in ort_run(sess, x)]
        max_diff = max(max_diff, max(float(np.abs(a - b).max()) for a, b in zip(ref, got)))
        d_ref = dm.decode_multi(ref, dm.LEVELS, 0.3)
        d_onnx = dm.decode_multi(got, dm.LEVELS, 0.3)
        n_ref += len(d_ref)
        n_onnx += len(d_onnx)
        for a in d_ref:
            best = 0.0
            for b in d_onnx:
                if a["label_idx"] != b["label_idx"]:
                    continue
                iou = dm.box_iou(np.array([[a["x0"], a["y0"], a["x1"], a["y1"]]]),
                                 np.array([[b["x0"], b["y0"], b["x1"], b["y1"]]]))[0, 0]
                best = max(best, float(iou))
            if best >= 0.95:
                n_match += 1
    return {"max_tensor_diff": max_diff, "det_ref": n_ref, "det_onnx": n_onnx, "matched_iou95": n_match}


def quantize_int8(src: Path, dst: Path, size: int, n_calib: int, data_dir: str | None = None) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(str(src), str(dst), CalibReader(size, n_calib, data_dir),
                    quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Netz nach ONNX exportieren und pruefen")
    ap.add_argument("--ckpt", default="models/signs-det.pt")
    ap.add_argument("--out", default="models/signs-det.onnx")
    ap.add_argument("--int8", action="store_true", help="zusaetzlich statisch quantisieren")
    ap.add_argument("--calib-n", type=int, default=8, help="Kalibrierbilder fuer int8 (mehr = besser)")
    ap.add_argument("--calib-data", default="", help="Ordner mit echten Bildern (z.B. data/det) fuer die Kalibrierung")
    ap.add_argument("--parity-n", type=int, default=6)
    ap.add_argument("--bench", type=int, default=10, help="Laeufe fuer die Laufzeitmessung (CPU-Referenz)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_model(args.ckpt)
    size = int(ck.get("size", 320))
    print(f"[ckpt] {args.ckpt}  cfg={cfg.name()}  size={size}  params={n_params(model)}  "
          f"epoche={ck.get('epoch')}  metriken={ck.get('metrics')}")

    export_onnx(model, size, out_path)
    print(f"[export] {out_path}  {out_path.stat().st_size/1e6:.2f} MB")

    sess = ort_session(out_path)
    par = parity_check(model, sess, size, args.parity_n)
    print(f"[paritaet] max. Tensorabweichung={par['max_tensor_diff']:.2e}  "
          f"Erkennungen torch={par['det_ref']} onnx={par['det_onnx']} "
          f"davon IoU>=0.95 {par['matched_iou95']}")

    x = np.ascontiguousarray(np.zeros((1, 3, size, size), np.float32))
    t0 = time.perf_counter()
    for _ in range(args.bench):
        ort_run(sess, x)
    ms = (time.perf_counter() - t0) / args.bench * 1000.0
    print(f"[laufzeit] ONNX Runtime CPU (x86, 1 Thread): {ms:.1f} ms je Bild bei {size}x{size}")

    files = {"onnx": out_path.name, "fp32_bytes": out_path.stat().st_size}
    if args.int8:
        q = out_path.with_name(out_path.stem + "-int8.onnx")
        quantize_int8(out_path, q, size, args.calib_n, args.calib_data or None)
        sess_q = ort_session(q)
        par_q = parity_check(model, sess_q, size, max(2, args.parity_n // 2))
        ok = (par_q["max_tensor_diff"] <= 0.5 and par_q["matched_iou95"] >= par_q["det_ref"]) \
            if par_q["det_ref"] else par_q["max_tensor_diff"] <= 0.5
        print(f"[int8] {q}  {q.stat().st_size/1e6:.2f} MB  "
              f"max. Tensorabweichung={par_q['max_tensor_diff']:.2e}  "
              f"Erkennungen matched={par_q['matched_iou95']}/{par_q['det_ref']}  "
              f"-> {'VERWENDET' if ok else 'VERWORFEN (Paritaet zu schlecht)'}")
        if ok:
            files["int8"] = q.name
            files["int8_bytes"] = q.stat().st_size
        else:
            print("[int8] Hinweis: mehr Kalibrierbilder (--calib-n 200 aus echten Frames), "
                  "QuantFormat.QOperator oder fp16 versuchen. Solange bleibt fp32 aktiv.")
            q.unlink(missing_ok=True)

    labels = {"format": "signs-det/1", "classes": SIGN_LABELS, "size": size, "levels": list(dm.LEVELS),
              "channels": dm.N_CH, "layout": "tx,ty,tw,th,obj,cls0..cls8",
              "score": "sigmoid(obj)*max(sigmoid(cls))", "files": files}
    (out_path.parent / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {"arch": cfg.as_dict(), "params": n_params(model), "size": size, "classes": SIGN_LABELS,
                "trained_epoch": ck.get("epoch"), "train_metrics": ck.get("metrics"),
                "parity": par, "ort_cpu_ms": round(ms, 2), "files": files,
                "sha256": {k: sha256(out_path.parent / v) for k, v in files.items() if k.endswith("onnx")},
                "torch": torch.__version__}
    (out_path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[labels] {out_path.parent/'labels.json'}  [manifest] {out_path.parent/'manifest.json'}")


if __name__ == "__main__":
    main()
