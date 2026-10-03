"""
tools/export_onnx.py - Trainiertes Netz nach ONNX ausgeben und die Ausgabe pruefen.

Schritte:
  1. Checkpoint laden (models/signs-det.pt oder --ckpt), Netz rekonstruieren.
  2. ONNX-Export mit statischer Eingabe (1,3,size,size) und einem Ausgang je Stufe
   (os4/os8/os16/os32, siehe tools/detmath.py LEVELS).
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
import signmap
import synth_data as sd
from hybrid_net import N_CLASSES, SIGN_LABELS, NetCfg, build_model, n_params


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


def export_onnx(model, size: int, out_path: Path, dynamic: bool = False) -> None:
    """Exportieren; die Ausgangsnamen folgen den Stufen (os4, os8, os16, os32).

    src/model.js liest sie ueber labels.levels -> out['os' + stride]. Das Namensschema
    haengt also an tools/detmath.py (LEVELS) und nicht an einer Liste hier.

    dynamic=True laesst Hoehe und Breite offen (Vielfache von 32): EIN Modell fuer alle
    Eingabegroessen. Das ist messbar sinnvoll - auf den val-Bildern bringt 384 px gegenueber
    320 px +2.5 Punkte Recall und 21 % weniger Fehlalarme, 256 px ist umgekehrt der
    Sparmodus fuer langsame Geraete (docs/TRAINING.md 3).
    """
    dummy = torch.zeros(1, 3, size, size)
    axes = {"images": {0: "batch", 2: "height", 3: "width"}} if dynamic else None
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["images"], output_names=[f"os{s}" for s in dm.LEVELS],
        opset_version=17, dynamo=False, do_constant_folding=True, dynamic_axes=axes,
    )


def ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_run(sess, x: np.ndarray) -> list[np.ndarray]:
    names = [o.name for o in sess.get_outputs()]
    return sess.run(names, {"images": x})


def dynamic_check(model, sess, size: int, other: int, seed: int = 5) -> dict:
    """Gegenprobe fuer dynamische Achsen: dasselbe Bild in anderer Groesse durch ONNX.

    Ohne diese Pruefung faellt eine offene Hoehe/Breite erst im Browser auf. Verglichen
    werden die Ausgangsformen (muessen zu size/stride passen) und die Tensorabweichung
    gegen PyTorch bei derselben Groesse.
    """
    rng = random.Random(seed)
    arr, _, _ = sd.compose_sample(rng, size, degrade_prob=0.5)
    from PIL import Image

    small = np.asarray(Image.fromarray(arr).resize((other, other), Image.BILINEAR))
    x = np.ascontiguousarray(small.transpose(2, 0, 1))[None].astype(np.float32) / 255.0
    with torch.no_grad():
        ref = [o.numpy()[0] for o in model(torch.from_numpy(x))]
    got = ort_run(sess, x)
    shapes = [list(o.shape) for o in got]                # (1, 14, H/stride, W/stride)
    diff = max(float(np.abs(a - b[0]).max()) for a, b in zip(ref, got))
    ok = all(s[2] == other // lv and s[3] == other // lv for s, lv in zip(shapes, dm.LEVELS))
    return {"size": other, "shapes": shapes, "max_diff": diff, "forme_ok": ok}


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
    """Statische int8-Quantisierung (QDQ) mit Kalibrierbildern.

    Vorverarbeitung zuerst: Conv+BatchNorm verschmelzen und die Formen ableiten. Ohne diesen
    Schritt warnt der Quantisierer bei JEDER Faltung unseres Netzes
    ("Expected bias 'onnx::Conv_1304' to be an initializer"), weil hier alle Faltungen mit
    bias=False gebaut und die Verschiebung von der BatchNorm beigesteuert wird. Nach dem
    Verschmelzen traegt die Faltung die Verschiebung selbst - das ist genau die Form, fuer die
    die QDQ-Kalibrierung gedacht ist, und verbessert die Genauigkeit des int8-Modells.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quelle = src
    vor = dst.with_name(dst.stem + "-pre.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        quant_pre_process(str(src), str(vor), skip_symbolic_shape=False)
        quelle = vor
    except Exception as fehler:                      # noqa: BLE001 - Vorverarbeitung ist Wahl
        print(f"[int8] Vorverarbeitung uebersprungen ({type(fehler).__name__}: {fehler})")

    try:
        quantize_static(str(quelle), str(dst), CalibReader(size, n_calib, data_dir),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
    finally:
        vor.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Netz nach ONNX exportieren und pruefen")
    ap.add_argument("--ckpt", default="models/signs-det.pt")
    ap.add_argument("--out", default="models/signs-det.onnx")
    ap.add_argument("--int8", action="store_true", help="zusaetzlich statisch quantisieren")
    ap.add_argument("--calib-n", type=int, default=8, help="Kalibrierbilder fuer int8 (mehr = besser)")
    ap.add_argument("--calib-data", default="", help="Ordner mit echten Bildern (z.B. data/det) fuer die Kalibrierung")
    ap.add_argument("--parity-n", type=int, default=6)
    ap.add_argument("--size", type=int, default=0,
                    help="Exportgroesse abweichend vom Checkpoint (0 = wie trainiert)")
    ap.add_argument("--dynamic", action="store_true",
                    help="Hoehe/Breite offenlassen (Vielfache von 32) - ein Modell fuer 256/320/384/448 px")
    ap.add_argument("--check-size", type=int, default=384,
                    help="Groesse fuer die Gegenprobe der dynamischen Achsen")
    ap.add_argument("--bench", type=int, default=10, help="Laeufe fuer die Laufzeitmessung (CPU-Referenz)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model, cfg, ck = load_model(args.ckpt)
    size = int(args.size or ck.get("size", 320))
    print(f"[ckpt] {args.ckpt}  cfg={cfg.name()}  size={size}  params={n_params(model)}  "
          f"epoche={ck.get('epoch')}  metriken={ck.get('metrics')}")

    export_onnx(model, size, out_path, dynamic=args.dynamic)
    print(f"[export] {out_path}  {out_path.stat().st_size/1e6:.2f} MB"
          + ("  (dynamische Hoehe/Breite)" if args.dynamic else ""))

    sess = ort_session(out_path)
    par = parity_check(model, sess, size, args.parity_n)
    print(f"[paritaet] max. Tensorabweichung={par['max_tensor_diff']:.2e}  "
          f"Erkennungen torch={par['det_ref']} onnx={par['det_onnx']} "
          f"davon IoU>=0.95 {par['matched_iou95']}")

    dyn, rueckfall, grund = None, False, ""
    if args.dynamic:
        # Die Gegenprobe darf den Export NICHT mitreissen. Genau das ist im Kaggle-Lauf vom
        # 03.10. passiert: ein Reshape im damaligen Aufmerksamkeitsblock hatte die
        # Fenstergroesse fest eingebaut (4x5), das Netz stiess bei 384 px auf eine andere
        # Zellenzahl - die Ausnahme flog bis in main() durch, das Werkzeug endete mit Code 1,
        # und der Kernel SCHLUG DEN EXPORT ALS GANZEN FEHL ("uebersprungen"). Ergebnis: 2,5
        # Stunden Training, kein auslieferbares Modell.
        # Jetzt gilt: laesst sich Hoehe/Breite nicht oeffnen, wird STATISCH exportiert. Ein
        # Modell, das nur seine Trainingsgroesse kann, ist ungleich besser als keines -
        # src/model.js liest labels.dynamic und skaliert dann fest auf labels.size.
        try:
            dyn = dynamic_check(model, sess, size, args.check_size)
            print(f"[dynamisch] Eingang {args.check_size}x{args.check_size} -> "
                  f"Formen {dyn['shapes']}  passend={dyn['forme_ok']}  "
                  f"max. Tensorabweichung={dyn['max_diff']:.2e}")
        except Exception as fehler:                   # noqa: BLE001 - Rueckfall statt Abbruch
            grund = f"{type(fehler).__name__}: {str(fehler)[:160]}"
            dyn = None
        if dyn is not None and not dyn["forme_ok"]:
            grund = f"Ausgangsformen passen nicht zu {args.check_size} px"
            dyn = None
        if dyn is None:
            rueckfall, args.dynamic = True, False
            print(f"[dynamisch] FEHLGESCHLAGEN ({grund})")
            print(f"[rueckfall] statischer Export auf {size} px (fest)")
            out_path.unlink(missing_ok=True)
            export_onnx(model, size, out_path, dynamic=False)
            sess = ort_session(out_path)
            par = parity_check(model, sess, size, args.parity_n)
            print(f"[rueckfall] {out_path}  {out_path.stat().st_size/1e6:.2f} MB  "
                  f"Paritaet={par['max_tensor_diff']:.2e}")

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
              "channels": dm.N_CH, "layout": f"tx,ty,tw,th,obj,cls0..cls{N_CLASSES - 1}",
              "score": "sigmoid(obj)*max(sigmoid(cls))", "dynamic": bool(args.dynamic),
              # Anzeige-Informationen je Klasse. Ohne sie kennt die Oberflaeche nur die neun
              # Heuristik-Typen und koennte die uebrigen Klassen nicht benennen.
              "info": {name: signmap.INFO[name] for name in SIGN_LABELS},
              # Hierarchie (tools/signmap.py): Oberkategorie je Klasse und die Familien mit
              # ihren Unterkategorien. Der Browser BRAUCHT sie nicht - der Ausgangstensor
              # traegt die fertige Klasse. Sie steht hier, damit die Herkunft einer
              # Erkennung nachvollziehbar bleibt (z.B. "Tempolimit -> tempo70").
              "hierarchy": {"super": {name: signmap.SUPER_LABELS[signmap.SUPER_OF[i]]
                                      for i, name in enumerate(SIGN_LABELS)},
                            "groups": {f: list(signmap.SUPER_GRUPPEN[f])
                                       for f in signmap.SUPER_LABELS}},
              "files": files}
    if rueckfall:
        # Nicht verschweigen: die Anzeige und jedes spaetere Debugging haengt daran, ob das
        # Modell andere Eingabegroessen annehmen kann. Grund im Klartext dazu.
        labels["dynamic_fallback"] = {"grund": grund, "feste_groesse": size}
    (out_path.parent / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {"arch": cfg.as_dict(), "params": n_params(model), "size": size, "classes": SIGN_LABELS,
                "trained_epoch": ck.get("epoch"), "train_metrics": ck.get("metrics"),
                "parity": par, "dynamic": dyn, "ort_cpu_ms": round(ms, 2), "files": files,
                "dynamic_fallback": ({"grund": grund, "feste_groesse": size} if rueckfall else None),
                "sha256": {k: sha256(out_path.parent / v) for k, v in files.items()
                           if k in ("onnx", "int8")},
                "torch": torch.__version__}
    (out_path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[labels] {out_path.parent/'labels.json'}  [manifest] {out_path.parent/'manifest.json'}")


if __name__ == "__main__":
    main()
