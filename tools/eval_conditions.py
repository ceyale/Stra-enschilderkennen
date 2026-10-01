"""
tools/eval_conditions.py - Precision/Recall je Bedingung und je Klasse.

Warum: eine Gesamtzahl versteckt, WORAN es scheitert. Das Manifest von
tools/gtsrb_dataset.py und tools/synth_data.py traegt je Bild die Bedingungen, unter
denen es entstanden ist ("dunkel", "roll", "verdeckt", "mehrere", ...). Dieses Skript
gruppiert die Bilder danach und zaehlt je Gruppe tp/fp/fn - damit ist eine Verbesserung
messbar statt gefuehlt (docs/TRAINING.md 9.3).

Bewertet wird der PyTorch-Checkpoint (wie im Training). Fuer den Browser gilt derselbe
Wert, solange tools/export_onnx.py eine gute Paritaet meldet.

Aufruf:
    python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det
    python tools/eval_conditions.py --ckpt models/signs-det.pt --limit 200 --json data/eval.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import detmath as dm
import train_det as td
from export_onnx import load_model
from hybrid_net import SIGN_LABELS


def per_class_counts(dets: list[dict], boxes: np.ndarray, labels: np.ndarray,
                     iou_thres: float = 0.5) -> dict[int, dict]:
    """tp/fp/fn je Klasse. Eine Erkennung der Klasse c, die keine c-Box trifft, ist ein FP -
    auch dann, wenn an der Stelle ein Schild einer anderen Klasse steht (uebliche Sicht)."""
    out: dict[int, dict] = {}
    boxes = np.asarray(boxes).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1)
    for c in range(len(SIGN_LABELS)):
        d_c = [d for d in dets if d["label_idx"] == c]
        m = labels == c
        if not d_c and not bool(m.any()):
            continue
        tp, fp, fn = dm.match_counts(d_c, boxes[m], labels[m], iou_thres=iou_thres)
        out[c] = {"det": len(d_c), "gt": int(m.sum()), "tp": tp, "fp": fp, "fn": fn}
    return out


def _bump(g: dict, det: int, gt: int, tp: int, fp: int, fn: int) -> None:
    g["img"] += 1
    g["det"] += det
    g["gt"] += gt
    g["tp"] += tp
    g["fp"] += fp
    g["fn"] += fn


def _prf(g: dict) -> dict:
    p = g["tp"] / max(g["tp"] + g["fp"], 1)
    r = g["tp"] / max(g["tp"] + g["fn"], 1)
    return {**g, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def table(titel: str, rows: list[tuple[str, dict]]) -> None:
    """Kleine Tabelle im Stil der Doku (schmal genug fuer Markdown)."""
    print(f"\n{titel}")
    print(f"{'Gruppe':<22}{'Bilder':>7}{'Boxen':>7}{'Erk.':>7}{'tp':>6}{'fp':>6}{'fn':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    for name, g in rows:
        print(f"{name:<22}{g['img']:>7}{g['gt']:>7}{g['det']:>7}{g['tp']:>6}{g['fp']:>6}{g['fn']:>6}"
              f"{g['precision']:>8.3f}{g['recall']:>8.3f}{g['f1']:>8.3f}")



def main() -> None:
    ap = argparse.ArgumentParser(description="Precision/Recall je Bedingung und je Klasse")
    ap.add_argument("--ckpt", default="models/signs-det.pt")
    ap.add_argument("--data", default="data/det")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Bilder (0 = alle)")
    ap.add_argument("--json", default="", help="Ergebnis zusaetzlich als JSON ablegen")
    args = ap.parse_args()

    model, cfg, ck = load_model(args.ckpt)
    size = int(ck.get("size", 320))
    device = td.device_from_args(args)
    model.to(device).eval()
    loader, ds = td.make_loader(args.data, size, args.split, args.batch, False, 0, 0, 0.0, args.workers)
    if not ds.items:
        raise SystemExit(f"{args.data}: keine Bilder fuer split={args.split}")

    alle: dict = defaultdict(int)
    je_bedingung: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    je_klasse: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    seen = 0
    with torch.no_grad():
        for x, _tgt, meta in loader:
            outs = [o.cpu().numpy() for o in model(x.to(device))]
            for bi in range(len(meta)):
                if args.limit and seen >= args.limit:
                    break
                idx = seen
                seen += 1
                dets = dm.decode_multi([o[bi] for o in outs], dm.LEVELS, args.conf, iou_thres=0.45)
                boxes, labels = meta[bi]
                boxes, labels = boxes.numpy(), labels.numpy()
                tp, fp, fn = dm.match_counts(dets, boxes, labels, iou_thres=args.iou)
                _bump(alle, len(dets), len(labels), tp, fp, fn)
                for tag in (ds.items[idx][2].get("conditions") or ["ohne Bedingung"]):
                    _bump(je_bedingung[tag], len(dets), len(labels), tp, fp, fn)
                for c, g in per_class_counts(dets, boxes, labels, args.iou).items():
                    _bump(je_klasse[c], g["det"], g["gt"], g["tp"], g["fp"], g["fn"])

    if not seen:
        raise SystemExit("keine Bilder ausgewertet")
    gesamt = _prf(alle)
    print(f"[auswertung] {Path(args.ckpt).name}  cfg={cfg.name()}  daten={args.data}/{args.split}  "
          f"Bilder={seen}  conf={args.conf}  IoU={args.iou}  geraet={device}")
    table("Alle Bilder", [("alle", gesamt)])
    table("Je Bedingung (ein Bild kann mehrere tragen)",
          sorted(((k, _prf(v)) for k, v in je_bedingung.items()), key=lambda kv: -kv[1]["fn"]))
    table("Je Klasse", [(SIGN_LABELS[c], _prf(v)) for c, v in sorted(je_klasse.items())])

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "ckpt": args.ckpt, "cfg": cfg.name(), "data": args.data, "split": args.split,
            "size": size, "conf": args.conf, "iou": args.iou, "bilder": seen, "alle": gesamt,
            "bedingungen": {k: _prf(v) for k, v in je_bedingung.items()},
            "klassen": {SIGN_LABELS[c]: _prf(v) for c, v in sorted(je_klasse.items())},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
