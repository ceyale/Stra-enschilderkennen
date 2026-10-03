"""
tools/detmath.py - Gemeinsame Mathematik fuer Training, Auswertung und Browser.

WICHTIG: Die Funktionen hier sind die Vorlage fuer src/model.js. Aendert sich hier
etwas (Letterbox, Decode-Formel, NMS), muss es dort wortgleich nachgezogen werden -
sonst rechnet der Browser anders als das Training.

Konventionen:
  * Boxen intern immer als xyxy in PIXELN des Modell-Eingangs (320x320, nach Letterbox).
  * Kopf-Ausgang je Zelle: [tx, ty, tw, th, obj, cls0..clsN-1] (N siehe tools/signmap.py).
  * Decode: cx = (gx + sigmoid(tx)) * stride, w = exp(tw) * stride.
"""
from __future__ import annotations

import math

import numpy as np

LEVELS = (4, 8, 16, 32)       # stride der vier Erkennungsstufen (fein -> grob)
# Klassenzahl an EINER Stelle: tools/signmap.py. Vorher stand hier eine zweite 9 - die
# haette nach der Klassenerweiterung stillschweigend falsch weitergerechnet.
import signmap

N_CLASSES = signmap.N_LABELS
N_CH = 4 + 1 + N_CLASSES
LETTERBOX_GREY = 114          # Randfarbe beim Einpassen (muss in JS identisch sein)

# Zuordnung der Objekte zu den Stufen, nach laengster Objektseite (FCOS-Idee, auf die
# 320-px-Eingabe gerechnet): Stufe i bekommt Objekte bis zu dieser Seitenlaenge.
# Die frueher benutzte Regel "naechste Stufe in log2(diagonale)" schickte ein 30-px-Schild
# auf stride 32 - dort stehen nur 10x10 Zellen zur Verfuegung. Gemessen lag der Recall fuer
# Schilder unter 32 px Diagonale bei etwa 0.29 gegen 0.74 im Mittel (docs/TRAINING.md 10).
ASSIGN_MAX_SIDE = (16.0, 40.0, 96.0, float("inf"))


def letterbox_params(w: int, h: int, size: int) -> tuple[float, float, float]:
    """Skalierung und Rand fuer das Einpassen in ein Quadrat (gleiche Formel in JS!)."""
    s = min(size / float(w), size / float(h))
    nw, nh = int(round(w * s)), int(round(h * s))
    return s, (size - nw) / 2.0, (size - nh) / 2.0


def letterbox_array(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Bild (H,W,3) auf size x size einpassen, mit grauem Rand (114) wie in YOLO."""
    from PIL import Image

    h, w = img.shape[:2]
    s, px, py = letterbox_params(w, h, size)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((size, size, 3), LETTERBOX_GREY, dtype=np.uint8)
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    x0, y0 = int(round(px)), int(round(py))
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, s, float(x0), float(y0)


def xywhn_to_xyxy(boxes: np.ndarray, size: int) -> np.ndarray:
    """Normalisierte cx,cy,w,h (0..1) -> Pixel xyxy im size x size Bild."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, bw, bh = b[:, 0] * size, b[:, 1] * size, b[:, 2] * size, b[:, 3] * size
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def boxes_to_letterbox(boxes_xyxy: np.ndarray, s: float, pad_x: float, pad_y: float) -> np.ndarray:
    out = np.asarray(boxes_xyxy, dtype=np.float32).copy().reshape(-1, 4) * s
    out[:, [0, 2]] += pad_x
    out[:, [1, 3]] += pad_y
    return out


def boxes_from_letterbox(boxes_xyxy: np.ndarray, s: float, pad_x: float, pad_y: float) -> np.ndarray:
    out = np.asarray(boxes_xyxy, dtype=np.float32).copy().reshape(-1, 4)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    return out / s


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU-Matrix zwischen zwei Sätzen xyxy-Boxen. Rueckgabe (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(dets: list[dict], iou_thres: float = 0.45) -> list[dict]:
    """Non-Maximum-Suppression: pro Klasse, hoechster Score zuerst (wie src/model.js)."""
    keep: list[dict] = []
    for cls in {d["label_idx"] for d in dets}:
        group = sorted((d for d in dets if d["label_idx"] == cls), key=lambda d: -d["score"])
        while group:
            best = group.pop(0)
            keep.append(best)
            if group:
                ious = box_iou(np.array([[best["x0"], best["y0"], best["x1"], best["y1"]]]),
                               np.array([[d["x0"], d["y0"], d["x1"], d["y1"]] for d in group]))[0]
                group = [d for d, i in zip(group, ious) if i <= iou_thres]
    return sorted(keep, key=lambda d: -d["score"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def assign_targets(boxes_xyxy: np.ndarray, labels: np.ndarray, size: int,
                   levels: tuple = LEVELS, max_side: tuple = ASSIGN_MAX_SIDE,
                   dual: bool = True, ref_size: int = 320) -> list[dict]:
    """Zielwerte fuer den Verlust: je Objekt eine Zelle, bei Randlage die Nachbarzelle mit.

    Zwei Aenderungen gegenueber der ersten Fassung, beide aus Messungen begruendet:
      * Stufe nach laengster Seite statt nach log2 der Diagonale (siehe ASSIGN_MAX_SIDE).
      * `dual`: liegt der Objektmittelpunkt im aeusseren Viertel seiner Zelle, lernt die
        Nachbarzelle dasselbe Objekt mit. Die alte Ein-Zellen-Regel presste den Offset auf
        0.999 fest - bei grossen Schildern (Median der Lokalisierungsfehler: 125 px) musste
        die Box also aus einer einzigen Zelle heraus mehrere Zellen weit regressiert werden.

    `ref_size` ist die Eingabegroesse, auf die ASSIGN_MAX_SIDE gemessen wurde. Die Grenzen
    sind ABSOLUTE Pixel und muessen mit der Eingabe mitwachsen: ohne die Umrechnung landet
    dasselbe Schild bei 384 px eine Stufe feiner als bei 320 px. Die Zellen der feineren
    Stufe sind kleiner als das Schild - der Offset klemmt dann am Zellenrand, und die
    Stufen sind ungleich ausgelastet. Bei size == ref_size bleibt alles wie bisher.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    faktor = float(size) / float(ref_size)
    out = []
    for s in levels:
        g = size // s
        out.append({
            "stride": s,
            "obj": np.zeros((g, g), dtype=np.float32),
            "cls": np.zeros((g, g), dtype=np.int64),
            "box": np.zeros((g, g, 4), dtype=np.float32),
            "pos": np.zeros((g, g), dtype=bool),
            # Welche BOX (Index in `boxes_xyxy`) diese Zelle traegt, -1 = keine. Braucht nur
            # die Wissens-Distillation (tools/teacher.py): die Lehrer-Verteilung haengt am
            # Objekt, nicht an der Klasse - zwei Schilder derselben Klasse koennen eine
            # verschiedene Verteilung haben, und die Zelle muss ihre eigene bekommen.
            "boxidx": np.full((g, g), -1, dtype=np.int64),
        })
    for box_nr, ((x0, y0, x1, y1), lab) in enumerate(zip(boxes, labels)):
        w, h = float(x1 - x0), float(y1 - y0)
        cx, cy = float((x0 + x1) / 2), float((y0 + y1) / 2)
        side = max(w, h)
        idx = len(levels) - 1                      # groebste Stufe als Rueckfall
        for i, lim in enumerate(tuple(max_side)[:len(levels)]):
            if side < lim * faktor:                # Grenzen mit der Eingabe skalieren
                idx = i                            # kleinste passende Stufe zuerst
                break
        s = levels[idx]
        g = size // s
        gx = int(min(g - 1, max(0, math.floor(cx / s))))
        gy = int(min(g - 1, max(0, math.floor(cy / s))))
        xs, ys = [gx], [gy]
        if dual:
            fx, fy = cx / s - gx, cy / s - gy
            if fx < 0.25 and gx > 0:
                xs.append(gx - 1)
            elif fx > 0.75 and gx < g - 1:
                xs.append(gx + 1)
            if fy < 0.25 and gy > 0:
                ys.append(gy - 1)
            elif fy > 0.75 and gy < g - 1:
                ys.append(gy + 1)
        t = out[idx]
        for cell_x in xs:
            for cell_y in ys:
                t["pos"][cell_y, cell_x] = True
                t["obj"][cell_y, cell_x] = 1.0
                t["cls"][cell_y, cell_x] = lab
                t["boxidx"][cell_y, cell_x] = box_nr
                t["box"][cell_y, cell_x] = [
                    np.clip(cx / s - cell_x, 0.0, 0.999), np.clip(cy / s - cell_y, 0.0, 0.999),
                    math.log(max(w / s, 1e-3)), math.log(max(h / s, 1e-3)),
                ]
    return out


def decode_level(out: np.ndarray, stride: int, conf_thres: float = 0.25) -> list[dict]:
    """Kopf-Ausgang (14,H,W) -> Erkennungen (xyxy in Pixeln der Modell-Eingabe).

    Identisch in src/model.js implementiert (dort ueber Float32Array).
    """
    c, gh, gw = out.shape
    assert c == N_CH, f"erwartete {N_CH} Kanaele, bekam {c}"
    tx = sigmoid(out[0]); ty = sigmoid(out[1])
    tw = np.exp(np.clip(out[2], -8, 8)); th = np.exp(np.clip(out[3], -8, 8))
    obj = sigmoid(out[4])
    cls = sigmoid(out[5:])
    best = cls.max(axis=0)
    lab = cls.argmax(axis=0)
    score = obj * best
    ys, xs = np.nonzero(score >= conf_thres)
    dets = []
    for gy, gx in zip(ys.tolist(), xs.tolist()):
        cx = (gx + tx[gy, gx]) * stride
        cy = (gy + ty[gy, gx]) * stride
        w = tw[gy, gx] * stride
        h = th[gy, gx] * stride
        dets.append({"label_idx": int(lab[gy, gx]), "score": float(score[gy, gx]),
                     "cx": cx, "cy": cy, "w": w, "h": h,
                     "x0": cx - w / 2, "y0": cy - h / 2, "x1": cx + w / 2, "y1": cy + h / 2})
    return dets


def decode_multi(outputs: list[np.ndarray], levels: tuple = LEVELS,
                 conf_thres: float = 0.25, iou_thres: float = 0.45) -> list[dict]:
    dets: list[dict] = []
    for out, s in zip(outputs, levels):
        dets += decode_level(out, s, conf_thres)
    return nms(dets, iou_thres)


def match_counts(dets: list[dict], gt_boxes: np.ndarray, gt_labels: np.ndarray,
                 iou_thres: float = 0.5) -> tuple[int, int, int]:
    """Greedy-Zuordnung fuer Precision/Recall: Rueckgabe (richtig, falsch, verpasst)."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    used = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0
    for d in sorted(dets, key=lambda x: -x["score"]):
        cand = [i for i in range(len(gt_boxes))
                if not used[i] and gt_labels[i] == d["label_idx"]]
        if not cand:
            fp += 1
            continue
        ious = box_iou(np.array([[d["x0"], d["y0"], d["x1"], d["y1"]]]),
                       gt_boxes[cand])[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thres:
            used[cand[j]] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~used).sum())
