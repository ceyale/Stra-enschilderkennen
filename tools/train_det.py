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
import signmap as sm
import synth_data as sd
from hybrid_net import (N_CH, N_CLASSES, N_SUPER, SIGN_LABELS, NetCfg, build_model,
                        n_params, preset_cfg)


class SignDataset(torch.utils.data.Dataset):
    """Echte Bilder (images/ + labels/ + manifest.json) oder synthetisch on-the-fly.

    `zoom` schaltet die Mehrskaligkeit ein: ein zufaelliger Ausschnitt (oder eine mit Grau
    114 erweiterte Flaeche) wird auf `size` gezogen. Gemessen war der Recall fuer Schilder
    unter 32 px Diagonale nur etwa 0,29; ausserdem waeren ohne Mehrskaligkeit 256/320/448 px
    im Browser drei verschiedene Modelle statt einer Einstellung.
    """

    def __init__(self, root: str, size: int = 320, split: str = "train",
                 synth_len: int = 200, seed: int = 0, degrade_prob: float = 0.6,
                 zoom: float = 0.0, zoom_lo: float = 0.7, zoom_hi: float = 1.5,
                 teacher_dir: str = ""):
        self.size, self.split, self.seed = size, split, seed
        self.degrade_prob = degrade_prob
        # Mehrskaligkeit nur im Training - die Validierung soll dieselbe Messlatte bleiben.
        self.zoom = zoom if split == "train" else 0.0
        self.zoom_lo, self.zoom_hi = zoom_lo, zoom_hi
        self.items: list[tuple[Path, Path, dict]] = []
        self.manifest: dict = {}
        self.synth_len = synth_len
        # Wissens-Distillation: Lehrer-Verteilungen je Grundwahrheitsbox (tools/teacher.py).
        # Gespeichert wird NICHT nach Bildindex, sondern nach Bildnamen - der Lehrer-Cache
        # ueberspringt Bilder ohne Labeldatei, die Liste hier nicht. Ein Indexvergleich
        # wuerde also still verrutschen, sobald ein Bild fehlt.
        self.t_index: dict[str, tuple[int, int]] = {}
        self.t_logits = None
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
        if teacher_dir and self.split == "train" and self.items:
            self._lehrer_laden(Path(teacher_dir))

    def _lehrer_laden(self, ordner: Path) -> None:
        """Lehrer-Cache laden und den Bildnamen zuordnen.

        Der Cache haelt (logits, offset): Bild k besitzt die Boxen offset[k]..offset[k+1].
        Die `ids`-Liste im JSON ist die Bruecke - ueber den Bildnamen, nicht ueber den
        Index (siehe __init__). Stimmt die Boxzahl eines Bildes nicht mit der Labeldatei
        ueberein, wird der Eintrag verworfen statt falsch zugeordnet.
        """
        pfad = ordner / f"teacher_{self.split}.npz"
        json_pfad = ordner / f"teacher_{self.split}.json"
        if not pfad.exists() or not json_pfad.exists():
            print(f"[lehrer] kein Cache unter {pfad} - Training ohne Distillation")
            return
        daten = np.load(pfad)
        self.t_logits = daten["logits"].astype(np.float32)   # float16 gespeichert (Groesse)
        offset, ids = daten["offset"], json.loads(json_pfad.read_text(encoding="utf-8"))["ids"]
        self.t_index = {stem: (int(offset[i]), int(offset[i + 1])) for i, stem in enumerate(ids)}
        print(f"[lehrer] Cache {pfad.name}: {len(ids)} Bilder, {self.t_logits.shape[0]} Boxen, "
              f"{self.t_logits.shape[1]} Klassen, Temperatur "
              f"{json.loads(json_pfad.read_text(encoding='utf-8'))['temperatur']}")

    def lehrer_zeilen(self, i: int) -> tuple[int, int] | None:
        """(von, bis) in t_logits fuer Bild i - oder None, wenn es keinen Eintrag hat."""
        if self.t_logits is None or not self.items:
            return None
        return self.t_index.get(self.items[i][0].stem)

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
        zeilen = self.lehrer_zeilen(i) if self.t_logits is not None else None
        # Das Augmentieren (Zoom) aendert die Boxliste - die Lehrer-Verteilungen sind an die
        # urspruenglichen Boxen gebunden und wuerden dann falsch zugeordnet. Bei Zoom wird
        # dieses Bild deshalb OHNE Distillation gerechnet (nicht mit falscher Zuordnung).
        if zeilen is not None and len(xyxy) == zeilen[1] - zeilen[0]:
            lehrer = torch.from_numpy(self.t_logits[zeilen[0]:zeilen[1]])
        else:
            lehrer = torch.zeros((0, N_CLASSES), dtype=torch.float32)
        return (x, torch.from_numpy(np.asarray(xyxy, np.float32)),
                torch.from_numpy(np.asarray(labels, np.int64)), lehrer)


def collate(batch, size: int):
    xs = torch.stack([b[0] for b in batch])
    targets = {"obj": [], "cls": [], "box": [], "pos": [], "tcls": []}
    for lvl in range(len(dm.LEVELS)):
        g = size // dm.LEVELS[lvl]
        targets["obj"].append(torch.zeros(len(batch), g, g))
        targets["cls"].append(torch.zeros(len(batch), g, g, dtype=torch.long))
        targets["box"].append(torch.zeros(len(batch), g, g, 4))
        targets["pos"].append(torch.zeros(len(batch), g, g, dtype=torch.bool))
        # Lehrer-Verteilung je ZELLE (Distillation). Der Index ist die Boxnummer des Bildes;
        # -1 bedeutet "keine" und wird im Verlust verworfen (Zellen mit Hintergrund).
        targets["tcls"].append(torch.full((len(batch), g, g), -1, dtype=torch.long))
    # Lehrer-Verteilungen als GEPOLSTERTER Block (B, max_n, C): die Boxzahl ist je Bild
    # verschieden, und der Zugriff im Verlust laeuft ueber die Boxnummer der Zelle
    # (targets["tcls"]). Gepolsterte Zeilen werden nie ausgewaehlt - boxidx zeigt immer auf
    # eine echte Box des Bildes - deshalb braucht es keine Maske, nur den Nullzeilen-Sinn
    # "keine Lehre".
    max_n = max((b[3].shape[0] for b in batch), default=0)
    lehrer = torch.zeros(len(batch), max_n, N_CLASSES)
    meta = []
    for bi, (_, boxes, labels, lb) in enumerate(batch):
        meta.append((boxes, labels))
        # ZIELE ZUERST - unabhängig vom Lehrer. (Ein früher `continue` an dieser Stelle hat
        # vorübergehend die gesamte Zielzuweisung übersprungen: pos blieb 0, der Verlust
        # bestand nur aus Objektivität. Das darf nicht an einer Distillations-Bedingung hängen.)
        if len(boxes):
            for lvl, t in enumerate(dm.assign_targets(boxes.numpy(), labels.numpy(), size)):
                targets["obj"][lvl][bi] = torch.from_numpy(t["obj"])
                targets["cls"][lvl][bi] = torch.from_numpy(t["cls"])
                targets["box"][lvl][bi] = torch.from_numpy(t["box"])
                targets["pos"][lvl][bi] = torch.from_numpy(t["pos"])
                idx = torch.from_numpy(t["boxidx"])
                idx = torch.where(idx < lehrer.shape[1], idx, torch.full_like(idx, -1))
                targets["tcls"][lvl][bi] = idx
        # Lehrer-Verteilungen: nur wenn der Cache zu DIESEM Bild passt (gleiche Boxzahl).
        # Fehlt er oder weicht die Zahl ab, bleiben die Zeilen Null - der Verlust springt sie
        # ueber (siehe kd_verlust). Lieber keine Lehre als eine verschobene.
        if lb.shape[0] and lb.shape[0] == len(boxes):
            lehrer[bi, :lb.shape[0]] = lb
    return xs, targets, meta, lehrer


def class_frequencies(ds, n_klassen: int = N_CLASSES) -> np.ndarray:
    """Anzahl der Boxen je Klasse im Trainingssplit (aus den YOLO-Labeldateien).

    Grundlage der Klassengewichte. Gezaehlt werden Boxen, nicht Bilder: entscheidend ist,
    wie oft der Kopf eine Klasse ueberhaupt zu sehen bekommt.
    """
    zaehler = np.zeros(n_klassen, dtype=np.float64)
    for _bild, lab, _eintrag in ds.items:
        if not lab.exists():
            continue
        for zeile in lab.read_text(encoding="utf-8").splitlines():
            teile = zeile.split()
            if len(teile) >= 5:
                k = int(float(teile[0]))
                if 0 <= k < n_klassen:
                    zaehler[k] += 1.0
    return zaehler


def super_frequencies(cls_counts: np.ndarray) -> np.ndarray:
    """Boxen je FAMILIE (Summe ihrer Unterkategorien) - Grundlage der Familien-Gewichte."""
    fam = np.zeros(N_SUPER, dtype=np.float64)
    for k, n in enumerate(cls_counts):
        f = sm.SUPER_OF[k]
        if f >= 0:
            fam[f] += n
    return fam


def class_weights_from(counts: np.ndarray) -> np.ndarray:
    """w_c = (1/haeufigkeit_c)^0.5, auf Mittelwert 1 normiert.

    Gedaempft, nicht 1/f: die seltenste Klasse waere sonst einige hundert Mal so schwer wie
    die haeufigste und der Kopf kippte in die Gegenrichtung (er sagte dann bevorzugt die
    seltenen Klassen und produzierte genau die Fehlalarme, die er loswerden soll). Die
    Wurzel daempft das, und die Normierung auf 1 haelt den Anteil des Kopfes am
    Gesamtverlust unveraendert - sonst aendert sich mit den Gewichten auch die Balance
    zwischen Objektivitaet, Klasse und Box.

    Die Laenge richtet sich nach der Eingabe (nicht nach N_CLASSES): dieselbe Funktion
    rechnet die Gewichte der 74 Unterkategorien UND die der 9 Familien.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    w = np.ones(len(counts), dtype=np.float64)
    da = counts > 0
    if da.any():
        w[da] = (1.0 / counts[da]) ** 0.5
        w[da] /= w[da].mean()
    return w


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
                        alpha: float = 1.0, weight: torch.Tensor | None = None,
                        smoothing: float = 0.0) -> torch.Tensor:
    """Mehrklassen-Focal-Loss (Summe) fuer die Klassifikation der positiven Zellen.

    Warum Focal statt Kreuzentropie: bei der Fehlerzerlegung waren 98 % aller Fehlalarme
    echte Schilder mit FALSCHER Klasse (tools/eval_conditions.py --diagnose). Der Kopf
    findet die Boxen also, entscheidet sich aber falsch - und die Kreuzentropie behandelt
    leichte und schwere Faelle gleich. Der Faktor (1 - p_t)^gamma zieht die schweren nach
    vorn, ohne die leichten ganz wegzunehmen (gamma=0 ist wieder Kreuzentropie).

    alpha wirkt hier NUR als konstanter Faktor: der Klassifikationskopf rechnet
    ausschliesslich auf positiven Zellen, es gibt also keine Negativzelle auszugleichen -
    genau das macht alpha im RetinaNet-Rezept. Der dortige Wert 0,25 drosselt damit den
    Anteil des Kopfes am Gesamtverlust auf ein Viertel; kaggle/train_kernel.py gleicht das
    mit --cls-w 4.0 wieder aus (0,25 x 4,0 = 1,0). Ohne diesen Ausgleich wuerde die
    Aenderung den Kopf schwaecht, den sie staerken soll.

    smoothing verteilt einen kleinen Teil der Zielmasse gleichmaessig auf alle Klassen
    (Label-Smoothing). Das haelt die Logits endlich und bremst die Ueberzeugung bei
    aehnlichen Zeichen - die Verwechslungsmatrix zeigte genau dort die Fehler.
    """
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        echt = torch.zeros_like(logp)
        echt.scatter_(1, target[:, None], 1.0)
        if smoothing > 0.0:
            echt = echt * (1.0 - smoothing) + smoothing / logp.shape[1]
    p = logp.exp()
    pt = (echt * p).sum(dim=-1).clamp(1e-7, 1.0)
    verlust = -(1.0 - pt).pow(gamma) * (echt * logp).sum(dim=-1)
    if alpha != 1.0:
        verlust = verlust * alpha
    if weight is not None:
        verlust = verlust * weight.to(target.device)[target]
    return verlust.sum()


def kd_verlust(schueler_logits: torch.Tensor, lehrer_p: torch.Tensor,
               temperatur: float = 2.0) -> torch.Tensor:
    """Wissens-Distillation: KL(Lehrer||Schueler) auf den positiven Zellen, Summe.

    Warum als KL gegen die LEHRER-VERTEILUNG und nicht als hartes Ziel: genau darum geht es
    bei der Distillation. Der Lehrer weiss mehr als die eine wahre Klasse - er kennt die
    Aehnlichkeiten (40er und 50er Schild sehen fast gleich aus, ein Aufhebungszeichen ist
    "Tempo ohne Zahl"). Diese Information steckt in den relativen Wahrscheinlichkeiten und
    geht bei einem harten Ziel verloren.

    Der Faktor T^2 hebt die Verkleinerung der Gradienten durch die Temperatur wieder auf
    (Hinton u. a., 2015) - ohne ihn waere die Distillation mit T=2 nur ein Viertel so stark
    wie beabsichtigt, und der Gewichts-Parameter im Rezept bedeutet etwas anderes als er sagt.

    Uebersprungen werden Zeilen, deren Summe 0 ist: das sind Bilder, fuer die kein
    Lehrer-Cache vorlag (siehe collate). Eine Nullzeile als Ziel waere ein Ziel, das
    "keine Klasse" behauptet - das waere falsche Lehre.
    """
    gueltig = lehrer_p.sum(dim=-1) > 0.0
    if not bool(gueltig.any()):
        return torch.zeros((), dtype=torch.float32, device=schueler_logits.device)
    ziel = lehrer_p[gueltig]
    # log q: die Wahrscheinlichkeiten des Lehrers sind bereits normiert (Softmax seiner
    # Logits, marginalisiert) - deshalb nur der Logarithmus, kein erneutes Softmax.
    log_q = torch.log(ziel.clamp_min(1e-9))
    log_p = F.log_softmax(schueler_logits[gueltig] / temperatur, dim=-1)
    return (ziel * (log_q - log_p)).sum() * (temperatur ** 2)


class DetLoss(nn.Module):
    """Verlust: Objektivitaet (fokal), Art (fokal, klassengewichtet, geglaettet), Box (L1, groessengewichtet).

    Bewusst ohne IoU/DFL-Verlust. Der Kommentar "CIoU waere der naechste Genauigkeitsschritt"
    war eine Vermutung - gemessen wurde sie widerlegt: die 1853 Treffer auf 2000 val-Bildern
    haben im Mittel IoU 0.938 (84 % ueber 0.9). Der Verlust ist also nicht die Baustelle;
    die Fehler liegen in der Klassentrennung (32 % der FN) und bei kleinen Schildern.

    Der ART-Verlust ist seit dem 03.10. ein Focal-Loss mit Klassengewichten
    (siehe focal_cross_entropy) - die Fehlerzerlegung hatte gezeigt, dass 98 % aller
    Fehlalarme echte Schilder mit falscher Klasse sind. Eine Auswertung ist in
    docs/TRAINING.md Abschnitt 10 nachgezogen.

    obj_norm steuert die Normierung des Objektivitaetsverlusts: "pos" teilt durch die Zahl
    der positiven Zellen (RetinaNet-Rezept), "sqrt" durch deren Wurzel. Mit vielen
    Negativbildern im Batch waechst der Objektivitaetsanteil stark - dann kann "sqrt"
    ruhiger trainieren.
    """

    def __init__(self, size: int = 320, alpha: float = 0.25, gamma: float = 2.0,
                 w_obj: float = 1.0, w_cls: float = 1.0, w_box: float = 5.0, smooth: float = 0.05,
                 obj_norm: str = "pos", cls_gamma: float = 2.0, cls_alpha: float = 1.0,
                 class_weights: np.ndarray | None = None, hier_aux: float = 0.0,
                 super_of: np.ndarray | None = None,
                 super_weights: np.ndarray | None = None,
                 distill: float = 0.0, temper: float = 2.0):
        super().__init__()
        self.size, self.alpha, self.gamma = size, alpha, gamma
        self.w_obj, self.w_cls, self.w_box, self.smooth = w_obj, w_cls, w_box, smooth
        self.obj_norm = obj_norm
        # alpha/gamma oben gehoeren zur OBJEKTIVITAET (binaer, dort gleicht alpha die
        # Uebermacht der Hintergrundzellen aus). cls_alpha/cls_gamma gehoeren zur ART.
        self.cls_gamma, self.cls_alpha = cls_gamma, cls_alpha
        self.class_weights = None if class_weights is None else torch.as_tensor(
            np.asarray(class_weights, dtype=np.float32))
        # Hilfsverlust auf der FAMILIE (nur hierarchischer Kopf, siehe TGADHead). Ohne ihn
        # lernt der Familienkopf nur mittelbar: sein Logit geht in die Summe ein, ein
        # eigenes Ziel hat er nicht. Mit ihm bekommt die grobe Entscheidung ein eigenes,
        # groeberes Signal - auch dann, wenn die Unterart noch falsch liegt.
        self.hier_aux = hier_aux
        self.super_of = None if super_of is None else torch.as_tensor(
            np.asarray(super_of, dtype=np.int64))
        self.super_weights = None if super_weights is None else torch.as_tensor(
            np.asarray(super_weights, dtype=np.float32))
        # Wissens-Distillation: Gewicht des Lehrer-Verlusts und die Temperatur. 0 schaltet
        # sie ab (dann wird auch kein Lehrer-Block gerechnet, siehe will_kd).
        self.distill, self.temper = distill, temper

    @property
    def will_kd(self) -> bool:
        """Braucht der Verlauf einen Lehrer-Block? (sonst ueberspringt der Lader ihn)"""
        return self.distill > 0.0

    @property
    def will_aux(self) -> bool:
        """Braucht der Verlust den Familien-Ausgang? (dann want_aux=True im Vorwaertslauf)"""
        return self.hier_aux > 0.0 and self.super_of is not None

    def forward(self, preds, targets: dict, lehrer=None) -> tuple[torch.Tensor, dict]:
        # Der hierarchische Verlustlauf liefert ein Paar (Ausgaenge, Familien-Logits);
        # die Auswertung ruft das Netz ohne want_aux und bekommt nur die Ausgaenge.
        familie = None
        if isinstance(preds, tuple):
            preds, familie = preds
        zero = torch.zeros((), dtype=torch.float32)
        obj_t, cls_t, box_t = zero, zero.clone(), zero.clone()
        fam_t = zero.clone()
        kd_t = zero.clone()
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
            cls_t = cls_t + focal_cross_entropy(cls_logits, cls_target, gamma=self.cls_gamma,
                                                alpha=self.cls_alpha, weight=self.class_weights,
                                                smoothing=self.smooth)
            if familie is not None and self.super_of is not None:
                # Familie = Ziel-Familie der Wahren Klasse. flat waehlt dieselben Zellen aus
                # wie oben, damit der Hilfsverlust genau auf den positiven Zellen wirkt.
                fam_logits = familie[lvl].permute(0, 2, 3, 1).reshape(-1, N_SUPER)[flat]
                fam_target = self.super_of.to(cls_target.device)[cls_target]
                fam_t = fam_t + focal_cross_entropy(fam_logits, fam_target, gamma=self.cls_gamma,
                                                    alpha=self.cls_alpha,
                                                    weight=self.super_weights,
                                                    smoothing=self.smooth)
            pb = p[:, :4].permute(0, 2, 3, 1).reshape(-1, 4)[flat]
            tb = targets["box"][lvl].reshape(-1, 4)[flat]
            pred = torch.cat([torch.sigmoid(pb[:, :2]), pb[:, 2:]], dim=1)
            wh_px = torch.exp(tb[:, 2:]) * dm.LEVELS[lvl]
            gain = (2.0 - (wh_px[:, 0] * wh_px[:, 1]) / (self.size * self.size)).clamp(0.5, 2.0)
            box_t = box_t + (F.l1_loss(pred, tb, reduction="none").sum(-1) * gain).sum()
            if lehrer is not None and self.will_kd and "tcls" in targets:
                # Lehrer-Verteilung DIESER Zelle: tcls haelt die Boxnummer, `lehrer` ist der
                # gepolsterte Block (B, max_n, C). Der Zugriff laeuft ueber (Bild, Zelle),
                # nicht ueber die flache Position - sonst wuerde die Zuordnung bei
                # ungleichen Bildgroessen im Batch verrutschen.
                b_idx, c_idx = pos.reshape(pos.shape[0], -1).nonzero(as_tuple=True)
                bnr = targets["tcls"][lvl].reshape(pos.shape[0], -1)[b_idx, c_idx]
                gueltig = (bnr >= 0) & (bnr < lehrer.shape[1])
                if bool(gueltig.any()):
                    kd_t = kd_t + kd_verlust(cls_logits[gueltig], lehrer[b_idx[gueltig],
                                                                       bnr[gueltig]],
                                              self.temper)
        n = float(n_pos)
        if self.obj_norm == "sqrt":
            n = math.sqrt(max(n, 1.0))
        n = max(n, 1.0)
        # Normierung wie in RetinaNet: durch die Anzahl positiver Zellen teilen, nicht
        # durch alle Zellen. Sonst waeren die wenigen Schilder im Lernschritt verdunnt.
        obj, cls, box = obj_t / n, cls_t / n, box_t / n
        fam = fam_t / n
        kd = kd_t / n
        loss = (self.w_obj * obj + self.w_cls * cls + self.w_box * box
                + self.hier_aux * fam + self.distill * kd)
        return loss, {"obj": float(obj.detach()), "cls": float(cls.detach()),
                      "box": float(box.detach()), "fam": float(fam.detach()),
                      "kd": float(kd.detach()), "n_pos": n_pos}


METRIC_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "fp_bild")


@torch.no_grad()
def evaluate(model: nn.Module, loader, conf: float = 0.25, iou: float = 0.5) -> dict:
    """Precision/Recall bei IoU 0.5 - die Kennzahl, die im Browser zaehlt."""
    model.eval()
    dev = next(model.parameters()).device
    tp = fp = fn = 0
    for x, _, meta, _ in loader:
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
    # Kopf und Hierarchie sind davon AUSGENOMMEN: sie sind keine Groesse, sondern eine
    # Bauart - und werden in beiden Zweigen ausdruecklich durchgereicht.
    kopf = dict(head=args.head, hier=not args.no_hier)
    if getattr(args, "preset", ""):
        return preset_cfg(args.preset, **kopf)
    return NetCfg(catm=parse_list(args.catm), act=args.act,
                  tr_ffn=args.ffn, tr_norm=args.norm, **kopf)


def make_loader(root: str, size: int, split: str, batch: int, shuffle: bool,
                n_synth: int, seed: int, degrade: float, workers: int = 0,
                zoom: float = 0.0, zoom_lo: float = 0.7, zoom_hi: float = 1.5,
                teacher_dir: str = ""):
    ds = SignDataset(root, size, split, synth_len=n_synth, seed=seed, degrade_prob=degrade,
                     zoom=zoom, zoom_lo=zoom_lo, zoom_hi=zoom_hi, teacher_dir=teacher_dir)
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
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal-gamma im KLASSIFIKATIONSkopf (0 = reine Kreuzentropie)")
    ap.add_argument("--focal-alpha", type=float, default=1.0,
                    help="konstanter Faktor des Klassifikations-Focal-Loss; 0,25 ist der "
                         "RetinaNet-Wert und drosselt den Kopf auf ein Viertel - dann mit "
                         "--cls-w 4.0 ausgleichen (tools/train_det.py focal_cross_entropy)")
    ap.add_argument("--cls-w", type=float, default=1.0, help="Gewicht des Klassifikationsverlusts")
    ap.add_argument("--smooth", type=float, default=0.05,
                    help="Label-Smoothing im Klassifikationskopf (0 = aus)")
    ap.add_argument("--no-class-weights", action="store_true",
                    help="Klassengewichte (1/haeufigkeit)^0.5 abschalten")
    ap.add_argument("--head", default="tgad", choices=["tgad", "plain"],
                    help="Erkennungskopf: tgad = TGADHead (aufgabengefuehrt, entkoppelt), "
                         "plain = Vorgaengerkopf mit gemeinsamem Stamm")
    ap.add_argument("--no-hier", action="store_true",
                    help="flacher Klassifikationskopf statt Ober-/Unterkategorien")
    ap.add_argument("--hier-aux", type=float, default=0.3,
                    help="Gewicht des Hilfsverlusts auf der FAMILIE (0 = aus)")
    ap.add_argument("--teacher", default="", help="Ordner mit dem Lehrer-Cache "
                    "(tools/teacher.py) - aktiviert die Wissens-Distillation")
    ap.add_argument("--distill", type=float, default=0.0,
                    help="Gewicht des Lehrer-Verlusts (KL gegen die Lehrer-Verteilung, mit T^2 "
                         "skaliert); 0 = aus. Gemessen traegt eine positive Zelle 2-9 bei, "
                         "waehrend der Klassifikationsverlust bei 0,1-0,3 liegt - deshalb sind "
                         "kleine Werte richtig (das Rezept nutzt 0,25), nicht 1")
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="Temperatur der Distillation (Hinton u. a.; 2 ist der Standardwert)")
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

    lehrer_dir = args.teacher if args.distill > 0 else ""
    if args.distill > 0 and not args.teacher:
        print("[lehrer] --distill ohne --teacher: Distillation bleibt aus (kein Cache angegeben)")
    if args.teacher and args.distill <= 0:
        print("[lehrer] --teacher ohne --distill: Distillation bleibt aus (Gewicht 0)")
    loader, train_ds = make_loader(args.data, args.size, "train", args.batch, True,
                                   args.synth_train, args.seed, args.degrade, args.workers,
                                   args.zoom, args.zoom_lo, args.zoom_hi, lehrer_dir)
    vloader, val_ds = make_loader(args.data, args.size, args.val_split, args.batch, False,
                                  args.synth_val, args.seed + 999, 0.0, args.workers)

    # Klassengewichte aus dem TRAININGSsplit (nicht aus val - sonst waere das eine
    # versteckte Messlatte im Verlust). Gezaehlt werden die Boxen der Labeldateien.
    cls_weights = None
    haeufig = np.zeros(N_CLASSES, dtype=np.float64)   # immer definiert: der Familien-
    if not args.no_class_weights:                     # Hilfsverlust braucht sie auch
        haeufig = class_frequencies(train_ds)
        if haeufig.sum() > 0:
            cls_weights = class_weights_from(haeufig)
            selten = np.argsort(haeufig)[:6]
            print("[klassen] haeufigste: " + ", ".join(
                f"{SIGN_LABELS[k]}={int(haeufig[k])}" for k in np.argsort(-haeufig)[:6]))
            print("[klassen] seltenste:  " + ", ".join(
                f"{SIGN_LABELS[k]}={int(haeufig[k])} -> w={cls_weights[k]:.2f}" for k in selten
                if haeufig[k] > 0))
            print(f"[klassen] Gewichte (1/haeufigkeit)^0.5, Mittelwert 1: "
                  f"min={cls_weights[haeufig > 0].min():.2f} max={cls_weights[haeufig > 0].max():.2f}")
    # Der Hilfsverlust der Hierarchie braucht die Ziel-Familien je Klasse. Sie kommen aus
    # tools/signmap.py; die Familien-Gewichte werden wie die Klassengewichte aus den
    # HAEUFIGKEITEN gerechnet (Summe der Unterkategorien), nicht geraten.
    hier_aux = args.hier_aux if (cfg.hier and args.head == "tgad") else 0.0
    super_of = np.asarray(sm.SUPER_OF, dtype=np.int64)
    fam_weights = None
    if hier_aux > 0 and not args.no_class_weights:
        fam_weights = class_weights_from(super_frequencies(haeufig)) if haeufig.sum() > 0 else None
    crit = DetLoss(size=args.size, w_box=args.box_w, w_cls=args.cls_w, obj_norm=args.obj_norm,
                   cls_gamma=args.focal_gamma, cls_alpha=args.focal_alpha,
                   class_weights=cls_weights, smooth=args.smooth,
                   hier_aux=hier_aux, super_of=super_of, super_weights=fam_weights,
                   distill=(args.distill if lehrer_dir else 0.0), temper=args.temperature)
    if args.hier_aux > 0 and hier_aux == 0.0:
        print("[hierarchie] Hilfsverlust abgeschaltet - er wirkt nur mit dem "
              "hierarchischen TGADHead (--head tgad ohne --no-hier)")

    total = max(1, args.epochs * args.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total, eta_min=args.lr * 0.05, last_epoch=start_epoch * args.steps - 1)

    print(f"[modell] {cfg.name()}  params={n_params(model)}  daten={train_ds.manifest.get('source')} "
          f"train={len(train_ds)} {args.val_split}={len(val_ds)} geraet={device} "
          f"zoom={args.zoom} obj_norm={args.obj_norm} kopf={args.head} "
          f"hier={'aus' if args.no_hier else 'an'} "
          f"cls=fokal(gamma={args.focal_gamma} alpha={args.focal_alpha} "
          f"w={args.cls_w} smooth={args.smooth} gewichte={'an' if cls_weights is not None else 'aus'}) "
          f"fam-hilfsverlust={hier_aux} "
          f"distill={crit.distill} (T={crit.temper}, cache={'an' if lehrer_dir else 'aus'})"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = list(ckpt.get("history", [])) if ckpt else []
    last_measured = None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, run = time.perf_counter(), None
        for step, (x, tgt, meta, lehrer) in enumerate(loader):
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            tgt = {k: [t.to(device, non_blocking=True) for t in v] for k, v in tgt.items()}
            lehrer = lehrer.to(device, non_blocking=True) if crit.will_kd else None
            preds = model.forward_aux(x, want_aux=crit.will_aux)
            loss, parts = crit(preds, tgt, lehrer)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run = parts
            if step % args.log_every == 0:
                print(f"  e{epoch} {step:4d}/{args.steps} loss={float(loss.detach()):7.3f} "
                      f"obj={parts['obj']:.3f} cls={parts['cls']:.3f} box={parts['box']:.3f} "
                      f"fam={parts['fam']:.3f} kd={parts['kd']:.3f} "
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
