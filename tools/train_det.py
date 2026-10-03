"""
tools/train_det.py - Training des hybriden Nano-Netzes (Ort + Art in einem Durchlauf).

Daten:
  * --data <ordner>   erwartet images/, labels/ (YOLO-TXT) und manifest.json
                      (so schreibt es tools/synth_data.py; echte Daten ebenso)
  * --data synth      erzeugt die Bilder on-the-fly mit tools/synth_data.py
                      (Rauchtest ohne echten Datensatz)

Beispiele:
  python tools/train_det.py --data synth --epochs 1 --steps 40 --batch 4 --smoke
  python tools/train_det.py --data data/det --epochs 60 --batch 16 --out models/signs-det.pt
  python tools/train_det.py --resume models/signs-det.pt --epochs 2    # weitertrainieren

Das Modell lernt Position UND Art gleichzeitig: der Kopf gibt je Zelle Box-Regression,
Objektivitaet und 9 Klassen aus (siehe tools/hybrid_net.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import detmath as dm
import synth_data as sd
from hybrid_net import N_CH, N_CLASSES, SIGN_LABELS, NetCfg, build_model, n_params, preset_cfg


class SignDataset(torch.utils.data.Dataset):
    """Echte Bilder (images/ + labels/ + manifest.json) oder synthetisch on-the-fly.

    `zoom` schaltet die Mehrskaligkeit ein: ein zufaelliger Ausschnitt (oder eine mit Grau
    114 erweiterte Flaeche) wird auf `size` gezogen. Gemessen war der Recall fuer Schilder
    unter 32 px Diagonale nur etwa 0,29; ausserdem waeren ohne Mehrskaligkeit 256/320/448 px
    im Browser drei verschiedene Modelle statt einer Einstellung.
    """

    def __init__(self, root: str, size: int = 320, split: str = "train",
                 synth_len: int = 200, seed: int = 0, degrade_prob: float = 0.6,
                 zoom: float = 0.0, zoom_lo: float = 0.7, zoom_hi: float = 1.5):
        self.size, self.split, self.seed = size, split, seed
        self.degrade_prob = degrade_prob
        # Mehrskaligkeit nur im Training - die Validierung soll dieselbe Messlatte bleiben.
        self.zoom = zoom if split == "train" else 0.0
        self.zoom_lo, self.zoom_hi = zoom_lo, zoom_hi
        self.items: list[tuple[Path, Path, dict]] = []
        self.manifest: dict = {}
        self.synth_len = synth_len
        p = Path(root)
        if root != "synth" and (p / "manifest.json").exists():
            self.manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            for e in self.manifest["images"]:
                if e.get("split", "train") != split:
                    continue
                img = p / "images" / f"{e['id']}.jpg"
                lab = p / "labels" / f"{e['id']}.txt"
                if img.exists():
                    self.items.append((img, lab, e))
        if not self.items:
            self.manifest = {"classes": SIGN_LABELS, "source": "synthetisch (on-the-fly)"}

    def __len__(self) -> int:
        return len(self.items) if self.items else self.synth_len

    def _load_synth(self, i: int):
        rng = random.Random(self.seed + i + (0 if self.split == "train" else 10_000_000))
        arr, boxes, _ = sd.compose_sample(rng, self.size, degrade_prob=self.degrade_prob)
        xyxy, labels = [], []
        for b in boxes:
            xyxy.append([(b["cx"] - b["w"] / 2) * self.size, (b["cy"] - b["h"] / 2) * self.size,
                         (b["cx"] + b["w"] / 2) * self.size, (b["cy"] + b["h"] / 2) * self.size])
            labels.append(SIGN_LABELS.index(b["label"]))
        return np.asarray(arr, dtype=np.uint8), np.asarray(xyxy, np.float32), np.asarray(labels, np.int64)

    def _load_real(self, i: int):
        img_path, lab_path, _ = self.items[i]
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        arr = np.asarray(img, dtype=np.uint8)
        arr, s, px, py = dm.letterbox_array(arr, self.size)
        xyxy, labels = [], []
        if lab_path.exists():
            for line in lab_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]))
                # Gespeicherte Boxen sind normalisiert auf das DATENSATZ-Bild, das Netz
                # sieht aber das Letterbox-Ergebnis - ohne diese Umrechnung waeren die
                # Ziele verschoben, sobald Datensatzgroesse != --size ist.
                box = np.array([[(cx - bw / 2) * w0, (cy - bh / 2) * h0,
                                 (cx + bw / 2) * w0, (cy + bh / 2) * h0]], np.float32)
                xyxy.append(dm.boxes_to_letterbox(box, s, px, py)[0])
                labels.append(cls)
        return arr, np.asarray(xyxy, np.float32).reshape(-1, 4), np.asarray(labels, np.int64)

    def _zoom_sample(self, arr, xyxy, labels, rng):
        """Zufaelliger Ausschnitt (oder graue Erweiterung) und zurueck auf `size`.

        z > 1 zoomt hinein (Schild wird groesser, Ausschnitt wandert), z < 1 heraus
        (Rand in Letterbox-Grau 114 - dieselbe Farbe, die im Datensatz ohnehin vorkommt).
        Anschliessend werden die Boxen mitgezogen; beschnittene Schilder behalten ihren
        sichtbaren Teil, fast unsichtbare fliegen raus.
        """
        size = self.size
        z = rng.uniform(self.zoom_lo, self.zoom_hi)
        xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        win = size / z                                   # Fenstergroesse in Originalpixeln
        if win > size:                                   # herauszoomen -> grauer Rand
            pad = int(round(win))
            canvas = np.full((pad, pad, 3), dm.LETTERBOX_GREY, dtype=np.uint8)
            off = (pad - size) // 2
            canvas[off:off + size, off:off + size] = arr
            arr, xyxy, src = canvas, xyxy + off, pad
        else:
            src = size
        win_px = max(8, min(src, int(round(win))))
        x0 = int(round(rng.uniform(0, max(0.0, src - win_px))))
        y0 = int(round(rng.uniform(0, max(0.0, src - win_px))))
        crop = np.ascontiguousarray(arr[y0:y0 + win_px, x0:x0 + win_px])
        out = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
        if len(xyxy) == 0:
            return out, xyxy, np.asarray(labels, np.int64)
        scale = size / float(win_px)
        boxes = (xyxy - np.array([x0, y0, x0, y0], np.float32)) * scale
        keep_b, keep_l = [], []
        for box, lab in zip(boxes, labels):
            vw = min(float(box[2]), size) - max(float(box[0]), 0.0)
            vh = min(float(box[3]), size) - max(float(box[1]), 0.0)
            if vw < 6 or vh < 6:                         # Rest zu klein fuer ein Ziel
                continue
            area = max((float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])), 1.0)
            if vw * vh / area < 0.3:                     # fast ganz ausserhalb: verwerfen
                continue
            keep_b.append([max(float(box[0]), 0.0), max(float(box[1]), 0.0),
                           min(float(box[2]), size), min(float(box[3]), size)])
            keep_l.append(lab)
        return out, np.asarray(keep_b, np.float32).reshape(-1, 4), np.asarray(keep_l, np.int64)

    def __getitem__(self, i: int):
        if self.items:
            arr, xyxy, labels = self._load_real(i)
        else:
            arr, xyxy, labels = self._load_synth(i)
        if self.split == "train":
            # Augmentationen bei JEDEM Aufruf neu ziehen, nicht je Bildindex: sonst saehe
            # dasselbe Bild in jeder Epoche genau dieselbe Verzerrung (der Sampler mischt
            # die Reihenfolge, der Seed bleibt trotzdem reproduzierbar).
            rng = random.Random(random.randrange(1 << 30))
            # Zielbedingungen auch bei echten Bildern nachbilden (siehe synth_data.degrade)
            if self.items and rng.random() < self.degrade_prob:
                arr, _ = sd.degrade(arr, rng, strength=0.8)
            if self.zoom and rng.random() < self.zoom:
                arr, xyxy, labels = self._zoom_sample(arr, xyxy, labels, rng)
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
        return x, torch.from_numpy(np.asarray(xyxy, np.float32)), torch.from_numpy(np.asarray(labels, np.int64))


def collate(batch, size: int):
    xs = torch.stack([b[0] for b in batch])
    targets = {"obj": [], "cls": [], "box": [], "pos": []}
    for lvl in range(len(dm.LEVELS)):
        g = size // dm.LEVELS[lvl]
        targets["obj"].append(torch.zeros(len(batch), g, g))
        targets["cls"].append(torch.zeros(len(batch), g, g, dtype=torch.long))
        targets["box"].append(torch.zeros(len(batch), g, g, 4))
        targets["pos"].append(torch.zeros(len(batch), g, g, dtype=torch.bool))
    meta = []
    for bi, (_, boxes, labels) in enumerate(batch):
        meta.append((boxes, labels))
        if len(boxes) == 0:
            continue
        for lvl, t in enumerate(dm.assign_targets(boxes.numpy(), labels.numpy(), size)):
            targets["obj"][lvl][bi] = torch.from_numpy(t["obj"])
            targets["cls"][lvl][bi] = torch.from_numpy(t["cls"])
            targets["box"][lvl][bi] = torch.from_numpy(t["box"])
            targets["pos"][lvl][bi] = torch.from_numpy(t["pos"])
    return xs, targets, meta


class DetLoss(nn.Module):
    """Verlust: Objektivitaet (fokal), Art (CE mit Label-Smoothing), Box (L1, groessengewichtet).

    Bewusst ohne IoU/DFL-Verlust. Der Kommentar "CIoU waere der naechste Genauigkeitsschritt"
    war eine Vermutung - gemessen wurde sie widerlegt: die 1853 Treffer auf 2000 val-Bildern
    haben im Mittel IoU 0.938 (84 % ueber 0.9). Der Verlust ist also nicht die Baustelle;
    die Fehler liegen in der Klassentrennung (32 % der FN) und bei kleinen Schildern.

    obj_norm steuert die Normierung des Objektivitaetsverlusts: "pos" teilt durch die Zahl
    der positiven Zellen (RetinaNet-Rezept), "sqrt" durch deren Wurzel. Mit vielen
    Negativbildern im Batch waechst der Objektivitaetsanteil stark - dann kann "sqrt"
    ruhiger trainieren.
    """

    def __init__(self, size: int = 320, alpha: float = 0.25, gamma: float = 2.0,
                 w_obj: float = 1.0, w_cls: float = 1.0, w_box: float = 5.0, smooth: float = 0.05,
                 obj_norm: str = "pos"):
        super().__init__()
        self.size, self.alpha, self.gamma = size, alpha, gamma
        self.w_obj, self.w_cls, self.w_box, self.smooth = w_obj, w_cls, w_box, smooth
        self.obj_norm = obj_norm

    def forward(self, preds: list[torch.Tensor], targets: dict) -> tuple[torch.Tensor, dict]:
        zero = torch.zeros((), dtype=torch.float32)
        obj_t, cls_t, box_t = zero, zero.clone(), zero.clone()
        n_pos = 0
        for lvl, p in enumerate(preds):
            t_obj = targets["obj"][lvl]
            pos = targets["pos"][lvl]
            logit = p[:, 4]
            bce = F.binary_cross_entropy_with_logits(logit, t_obj, reduction="none")
            pr = torch.sigmoid(logit)
            pt = pr * t_obj + (1 - pr) * (1 - t_obj)
            at = self.alpha * t_obj + (1 - self.alpha) * (1 - t_obj)
            obj_t = obj_t + (at * (1 - pt) ** self.gamma * bce).sum()
            if not bool(pos.any()):
                continue
            flat = pos.reshape(pos.shape[0], -1).reshape(-1)
            n_pos += int(flat.sum())
            cls_logits = p[:, 5:].permute(0, 2, 3, 1).reshape(-1, N_CLASSES)[flat]
            cls_target = targets["cls"][lvl].reshape(-1)[flat]
            cls_t = cls_t + F.cross_entropy(cls_logits, cls_target,
                                            label_smoothing=self.smooth, reduction="sum")
            pb = p[:, :4].permute(0, 2, 3, 1).reshape(-1, 4)[flat]
            tb = targets["box"][lvl].reshape(-1, 4)[flat]
            pred = torch.cat([torch.sigmoid(pb[:, :2]), pb[:, 2:]], dim=1)
            wh_px = torch.exp(tb[:, 2:]) * dm.LEVELS[lvl]
            gain = (2.0 - (wh_px[:, 0] * wh_px[:, 1]) / (self.size * self.size)).clamp(0.5, 2.0)
            box_t = box_t + (F.l1_loss(pred, tb, reduction="none").sum(-1) * gain).sum()
        n = float(n_pos)
        if self.obj_norm == "sqrt":
            n = math.sqrt(max(n, 1.0))
        n = max(n, 1.0)
        # Normierung wie in RetinaNet: durch die Anzahl positiver Zellen teilen, nicht
        # durch alle Zellen. Sonst waeren die wenigen Schilder im Lernschritt verdunnt.
        obj, cls, box = obj_t / n, cls_t / n, box_t / n
        loss = self.w_obj * obj + self.w_cls * cls + self.w_box * box
        return loss, {"obj": float(obj.detach()), "cls": float(cls.detach()),
                      "box": float(box.detach()), "n_pos": n_pos}


METRIC_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "fp_bild")


@torch.no_grad()
def evaluate(model: nn.Module, loader, conf: float = 0.25, iou: float = 0.5) -> dict:
    """Precision/Recall bei IoU 0.5 - die Kennzahl, die im Browser zaehlt."""
    model.eval()
    dev = next(model.parameters()).device
    tp = fp = fn = 0
    for x, _, meta in loader:
        outs = [o.cpu().numpy() for o in model(x.to(dev))]
        for bi in range(x.shape[0]):
            dets = dm.decode_multi([o[bi] for o in outs], dm.LEVELS, conf, iou_thres=0.45)
            boxes, labels = meta[bi]
            a, b, c = dm.match_counts(dets, boxes.numpy(), labels.numpy(), iou_thres=iou)
            tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    # Fehlalarme je Bild: die Kennzahl, die im Alltag stoert - auf dem negativen Split
    # (nur Hintergrund, kein Schild) ist sie die einzige sinnvolle Zahl.
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
            "fp_bild": fp / max(int(len(loader.dataset)), 1),
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def parse_list(value: str) -> tuple[str, ...]:
    if value.strip().lower() in ("", "-", "none", "keine"):
        return ()
    return tuple(s.strip() for s in value.split(",") if s.strip())


def device_from_args(args) -> torch.device:
    """Rechengeraet: 'auto' nimmt CUDA, wenn vorhanden, sonst die CPU.

    Dieselbe Datei funktioniert damit auf einem Laptop ohne GPU; der Checkpoint bleibt
    portabel, weil export_onnx.py immer mit map_location='cpu' laedt.
    """
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def cfg_from_args(args) -> NetCfg:
    # Mit --preset bestimmt das Preset die Groesse VOLLSTAENDIG. Sonst wuerden die
    # CLI-Defaults (z.B. --act silu) die Preset-Werte (hardswish) still ueberschreiben
    # und die Variante waere langsamer als gemessen.
    if getattr(args, "preset", ""):
        return preset_cfg(args.preset)
    return NetCfg(catm=parse_list(args.catm), act=args.act,
                  tr_ffn=args.ffn, tr_norm=args.norm)


def make_loader(root: str, size: int, split: str, batch: int, shuffle: bool,
                n_synth: int, seed: int, degrade: float, workers: int = 0,
                zoom: float = 0.0, zoom_lo: float = 0.7, zoom_hi: float = 1.5):
    ds = SignDataset(root, size, split, synth_len=n_synth, seed=seed, degrade_prob=degrade,
                     zoom=zoom, zoom_lo=zoom_lo, zoom_hi=zoom_hi)
    # functools.partial statt lambda: Lambdas sind unter Windows nicht picklebar und
    # lassen den Worker-Start mit EOFError abbrechen.
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        persistent_workers=workers > 0, collate_fn=partial(collate, size=size),
        drop_last=False), ds


def main() -> None:
    ap = argparse.ArgumentParser(description="hybrides Nano-Netz trainieren (Ort + Art)")
    ap.add_argument("--data", default="synth", help="Ordner mit images/labels/manifest.json oder 'synth'")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="Schritte je Epoche")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="Rechengeraet; 'auto' = CUDA wenn vorhanden, sonst CPU")
    ap.add_argument("--workers", type=int, default=2, help="Ladeprozesse (0 = im Hauptprozess)")
    ap.add_argument("--catm", default="p5,p4",
                    help="Stufen mit CATM (Convolutional Additive Token Mixer), z.B. p5,p4 "
                         "oder p5,p4,p3 oder - fuer reines CNN")
    ap.add_argument("--act", default="silu", choices=["silu", "hardswish"])
    ap.add_argument("--preset", default="", choices=["", "fast", "balanced", "quality", "breit"],
                    help="Größenvariante aus tools/hybrid_net.py (überschreibt Breiten/Tiefen)")
    ap.add_argument("--ffn", type=float, default=2.0, help="FFN-Expansion im CATM-Block")
    ap.add_argument("--norm", default="gn", choices=["gn", "ln"])
    ap.add_argument("--degrade", type=float, default=0.6, help="Anteil kuenstlich verschlechterter Bilder")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="Anteil Trainingsbilder mit Skalenschnitt (Mehrskaligkeit, z.B. 0.5)")
    ap.add_argument("--zoom-lo", type=float, default=0.7, help="kleinster Zoomfaktor (unter 1 = heraus)")
    ap.add_argument("--zoom-hi", type=float, default=1.5, help="groesster Zoomfaktor")
    ap.add_argument("--obj-norm", default="pos", choices=["pos", "sqrt"],
                    help="Normierung des Objektivitaetsverlusts (bei vielen Negativen: sqrt)")
    ap.add_argument("--val-split", default="val", choices=["val", "neg"],
                    help="Split fuer die Auswertung; 'neg' ist die Gegenprobe auf Fehlalarme")
    ap.add_argument("--synth-train", type=int, default=400)
    ap.add_argument("--synth-val", type=int, default=60)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--box-w", type=float, default=5.0, help="Gewicht des Box-Verlusts")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=1, help="Auswertung alle N Epochen")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--out", default="models/signs-det.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = device_from_args(args)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg, start_epoch, ckpt = cfg_from_args(args), 0, None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = NetCfg(**{k: v for k, v in ckpt["cfg"].items() if k in NetCfg.__dataclass_fields__})
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        args.size = int(ckpt.get("size", args.size))     # gleiche Auflösung weiter trainieren
        print(f"[resume] {args.resume} -> Epoche {start_epoch}, Konfiguration aus Checkpoint, "
              f"size={args.size}")
    model = build_model(cfg)
    crit = DetLoss(size=args.size, w_box=args.box_w, obj_norm=args.obj_norm)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            # Zustand auf das aktuelle Geraet ziehen (Checkpoint von CUDA, Lauf auf CPU o. umgekehrt)
            for st in opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
    model.to(device)

    loader, train_ds = make_loader(args.data, args.size, "train", args.batch, True,
                                   args.synth_train, args.seed, args.degrade, args.workers,
                                   args.zoom, args.zoom_lo, args.zoom_hi)
    vloader, val_ds = make_loader(args.data, args.size, args.val_split, args.batch, False,
                                  args.synth_val, args.seed + 999, 0.0, args.workers)
    total = max(1, args.epochs * args.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total, eta_min=args.lr * 0.05, last_epoch=start_epoch * args.steps - 1)

    print(f"[modell] {cfg.name()}  params={n_params(model)}  daten={train_ds.manifest.get('source')} "
          f"train={len(train_ds)} {args.val_split}={len(val_ds)} geraet={device} "
          f"zoom={args.zoom} obj_norm={args.obj_norm}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = list(ckpt.get("history", [])) if ckpt else []
    last_measured = None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.perf_counter(), None
        for step, (x, tgt, meta) in enumerate(loader):
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            tgt = {k: [t.to(device, non_blocking=True) for t in v] for k, v in tgt.items()}
            preds = model(x)
            loss, parts = crit(preds, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run = parts
            if step % args.log_every == 0:
                print(f"  e{epoch} {step:4d}/{args.steps} loss={float(loss.detach()):7.3f} "
                      f"obj={parts['obj']:.3f} cls={parts['cls']:.3f} box={parts['box']:.3f} "
                      f"pos={parts['n_pos']} n={len(x)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        # Auswertung ist teuer (jede val-Bild durch das Netz + NMS in numpy): nur alle
        # --eval-every Epochen und immer in der letzten - sonst steht im Log nichts Belastbares.
        due = args.eval_every <= 1 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if due:
            metrics = evaluate(model, vloader, args.conf, args.iou)
            last_measured = metrics
            print(f"[epoche {epoch}] P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} "
                  f"fp/Bild={metrics['fp_bild']:.3f}",
                  flush=True)
        else:
            metrics = {k: None for k in METRIC_KEYS}
            nxt = min(((epoch // args.eval_every) + 1) * args.eval_every - 1, args.epochs - 1)
            print(f"[epoche {epoch}] ohne Auswertung (naechste in Epoche {nxt})", flush=True)
        history.append({"epoch": epoch, "loss": None if run is None else run, "eval": due, **metrics})
        if (epoch + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg.as_dict(),
                        "epoch": epoch, "size": args.size, "history": history,
                        "classes": SIGN_LABELS, "metrics": last_measured or metrics}, args.out)
            print(f"[checkpoint] {args.out} gespeichert (Epoche {epoch})")

    gemessen = [h for h in history if h.get("eval", True)]
    print(json.dumps({"cfg": cfg.name(), "params": n_params(model), "out": args.out,
                      "letzte_metriken": gemessen[-1] if gemessen else None}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
