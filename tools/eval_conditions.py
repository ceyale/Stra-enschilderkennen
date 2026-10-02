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
    python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --diagnose
    python tools/eval_conditions.py --ckpt models/signs-det.pt --data data/det --split neg
    python tools/eval_conditions.py --ckpt models/signs-det.pt --limit 200 --json data/eval.json

--diagnose zerlegt zusaetzlich jede verpasste Box (Objektivitaet / falsche Box / falsche
Klasse) und jeden Fehlalarm (auf echtem Schild / auf Hintergrund), siehe diagnose().
--split neg wertet den Split aus, der NUR Hintergrundbilder ohne Schild enthaelt - dort ist
"fp/Bild" die Kennzahl (erzeugt von tools/synth_negatives.py).
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
    # Fehlalarme je Bild: auf dem negativen Split (nur Hintergrund) die einzige sinnvolle Zahl.
    return {**g, "precision": p, "recall": r, "fp_bild": g["fp"] / max(g["img"], 1),
            "f1": 2 * p * r / max(p + r, 1e-9)}


def table(titel: str, rows: list[tuple[str, dict]]) -> None:
    """Kleine Tabelle im Stil der Doku (schmal genug fuer Markdown)."""
    print(f"\n{titel}")
    print(f"{'Gruppe':<22}{'Bilder':>7}{'Boxen':>7}{'Erk.':>7}{'tp':>6}{'fp':>6}{'fn':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'fp/Bild':>9}")
    for name, g in rows:
        print(f"{name:<22}{g['img']:>7}{g['gt']:>7}{g['det']:>7}{g['tp']:>6}{g['fp']:>6}{g['fn']:>6}"
              f"{g['precision']:>8.3f}{g['recall']:>8.3f}{g['f1']:>8.3f}{g['fp_bild']:>9.3f}")



def diagnose(model, loader, ds, device, conf: float, iou: float) -> dict:
    """Fehlerzerlegung: WARUM eine Box verpasst oder erfunden wird.

    Jede verpasste Box wird eingeteilt in
      obj_miss  - keine Erkennung in der Naehe       (die Objektivitaet hat versagt)
      loc       - Erkennung da, aber IoU < iou       (die Box sitzt falsch)
      cls_wrong - Box sitzt richtig, Klasse falsch   (der Kopf verwechselt Arten)
    und jeder Fehlalarm in
      fremd_gt    - liegt auf einem echten Schild    (meist die falsche Klasse)
      hintergrund - freie Flaeche                    (der klassische Fehlalarm)
    Dazu mittlere IoU der Treffer, Groesse der verpassten Boxen und die Verwechslungs-
    matrix. Genau diese Zerlegung hat die Reihenfolge der Arbeiten bestimmt (Abschnitt 10).
    """
    fn_kind = {"obj_miss": 0, "loc": 0, "cls_wrong": 0}
    fp_kind = {"fremd_gt": 0, "hintergrund": 0}
    fp_scores: list[float] = []
    tp_ious: list[float] = []
    gt_diag: list[float] = []
    groesse: dict[str, list[float]] = {"obj_miss": [], "loc": []}
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    with torch.no_grad():
        for x, _tgt, meta in loader:
            outs = [o.cpu().numpy() for o in model(x.to(device))]
            for bi in range(len(meta)):
                dets = dm.decode_multi([o[bi] for o in outs], dm.LEVELS, conf, iou_thres=0.45)
                gb, gl = meta[bi]
                gb, gl = gb.numpy().reshape(-1, 4), gl.numpy().reshape(-1)
                for b in gb:
                    gt_diag.append(float(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))))
                if len(gb):
                    db = np.array([[d["x0"], d["y0"], d["x1"], d["y1"]] for d in dets]).reshape(-1, 4)
                    M = dm.box_iou(db, gb)
                else:
                    M = np.zeros((len(dets), 0))
                used = np.zeros(len(gb), bool)
                for k in sorted(range(len(dets)), key=lambda z: -dets[z]["score"]):
                    same = [i for i in range(len(gb)) if not used[i] and gl[i] == dets[k]["label_idx"]]
                    if same:
                        j = same[int(np.argmax(M[k, same]))]
                        if M[k, j] >= iou:
                            used[j] = True
                            tp_ious.append(float(M[k, j]))
                            continue
                    if len(gb) and float(M[k].max()) >= iou:
                        fp_kind["fremd_gt"] += 1
                    else:
                        fp_kind["hintergrund"] += 1
                    fp_scores.append(float(dets[k]["score"]))
                for i in range(len(gb)):
                    if used[i]:
                        continue
                    best = float(M[:, i].max()) if len(dets) else 0.0
                    b = gb[i]
                    dg = float(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0)))
                    if best >= iou:
                        fn_kind["cls_wrong"] += 1
                        k = int(np.argmax(M[:, i]))
                        confusion[(SIGN_LABELS[int(gl[i])], SIGN_LABELS[dets[k]["label_idx"]])] += 1
                    elif best >= 0.1:
                        fn_kind["loc"] += 1
                        groesse["loc"].append(dg)
                    else:
                        fn_kind["obj_miss"] += 1
                        groesse["obj_miss"].append(dg)
    tp_a, gt_a, fs = np.array(tp_ious), np.array(gt_diag), np.array(fp_scores)
    buckets: dict[str, int] = {}
    for lo, hi in ((0.25, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)):
        m = (fs >= lo) & (fs < hi) if len(fs) else np.zeros(0, bool)
        buckets[f"{lo:.2f}-{hi:.2f}"] = int(m.sum())
    return {
        "fn": fn_kind,
        "fp": fp_kind,
        "fp_score": buckets,
        "tp_iou_mittel": float(tp_a.mean()) if len(tp_a) else 0.0,
        "tp_iou_unter_07": int((tp_a < 0.7).sum()) if len(tp_a) else 0,
        "gt_boxen": int(len(gt_a)),
        "gt_unter_32px": int((gt_a <= 32).sum()) if len(gt_a) else 0,
        "gt_diag_median": float(np.median(gt_a)) if len(gt_a) else 0.0,
        "verpasst_median_px": {k: (float(np.median(v)) if v else None) for k, v in groesse.items()},
        "verpasst_unter_32px": {k: int(sum(1 for x in v if x <= 32)) for k, v in groesse.items()},
        "verwechslungen": [{"gt": a, "vorhergesagt": b, "n": n} for (a, b), n in
                           sorted(confusion.items(), key=lambda kv: -kv[1])[:12]],
    }


def print_diagnose(d: dict) -> None:
    """Die Zerlegung als Tabelle (Form wie in docs/TRAINING.md 9.3)."""
    f, p = d["fn"], d["fp"]
    ges_f, ges_p = max(sum(f.values()), 1), max(sum(p.values()), 1)
    print("\nWarum Fehler entstehen (Diagnose)")
    print(f"{'Ursache':<26}{'Anzahl':>8}{'Anteil':>9}{'Groesse median':>16}")
    for key, text in (("obj_miss", "verpasst (Objektivitaet)"),
                      ("loc", "Box sitzt falsch"),
                      ("cls_wrong", "falsche Klasse")):
        med = d["verpasst_median_px"].get(key)
        print(f"{text:<26}{f[key]:>8}{f[key] / ges_f:>9.1%}"
              f"{('-' if med is None else f'{med:.0f} px'):>16}")
    for key, text in (("fremd_gt", "FP auf echtem Schild"), ("hintergrund", "FP auf Hintergrund")):
        print(f"{text:<26}{p[key]:>8}{p[key] / ges_p:>9.1%}{'-':>16}")
    print(f"{'Treffer-IoU (Mittel)':<26}{d['tp_iou_mittel']:>8.3f}"
          f"   davon unter 0.7: {d['tp_iou_unter_07']}")
    print(f"{'GT-Boxen <= 32 px':<26}{d['gt_unter_32px']:>8} von {d['gt_boxen']}")
    if any(d["fp_score"].values()):
        print("Fehlalarme nach Score: " + ", ".join(f"{k}: {v}" for k, v in d["fp_score"].items()))
    if d["verwechslungen"]:
        print("Verwechslungen (gt -> vorhergesagt):")
        for e in d["verwechslungen"]:
            print(f"   {e['gt']:<20} -> {e['vorhergesagt']:<20}{e['n']:>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Precision/Recall je Bedingung und je Klasse")
    ap.add_argument("--ckpt", default="models/signs-det.pt")
    ap.add_argument("--data", default="data/det")
    ap.add_argument("--split", default="val", choices=["train", "val", "neg"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Bilder (0 = alle)")
    ap.add_argument("--size", type=int, default=0,
                    help="Eingabegroesse abweichend vom Checkpoint (0 = wie trainiert); "
                         "damit ist messbar, was 256 oder 384 px kosten")
    ap.add_argument("--json", default="", help="Ergebnis zusaetzlich als JSON ablegen")
    ap.add_argument("--diagnose", action="store_true",
                    help="zusaetzlich zerlegen, WARUM Boxen verpasst/erfunden werden")
    args = ap.parse_args()

    model, cfg, ck = load_model(args.ckpt)
    size = int(args.size or ck.get("size", 320))
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

    diag = None
    if args.diagnose:
        diag = diagnose(model, loader, ds, device, args.conf, args.iou)
        print_diagnose(diag)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "ckpt": args.ckpt, "cfg": cfg.name(), "data": args.data, "split": args.split,
            "size": size, "conf": args.conf, "iou": args.iou, "bilder": seen, "alle": gesamt,
            "bedingungen": {k: _prf(v) for k, v in je_bedingung.items()},
            "klassen": {SIGN_LABELS[c]: _prf(v) for c, v in sorted(je_klasse.items())},
            **({"diagnose": diag} if diag else {}),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[json] {out}")


if __name__ == "__main__":
    main()
