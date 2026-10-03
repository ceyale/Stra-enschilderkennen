"""
tools/gtsdb_dataset.py - echte deutsche Strassenszenen als Detektor-Datensatz.

Warum diese Quelle anders ist als alle anderen: GTSIGN-220, Synset und GTSRB liefern
AUSSCHNITTE (das Zeichen fuellt das Bild). GTSDB liefert fertige Szenen - 1 360 x 800, das
Schild darin klein und schief, mehrere Zeichen pro Bild. Genau diese Bedingung war die
Bruchstelle: die alte Messlatte bestand aus GTSRB-Ausschnitten, dort stand F1 0,90, waehrend
dasselbe Modell auf einem abfotografierten Poster 0,00 lieferte (PLAN.md §1).

Die Dateien liegen im COCO-Format vor (Roboflow-Ausgabe):

    <ordner>/_annotations.coco.json     Bilder, Kategorien, Boxen
    <ordner>/*.jpg

Aufruf:
    python tools/gtsdb_dataset.py --coco data/gtsdb/train --out data/det --split train
    python tools/gtsdb_dataset.py --coco data/gtsdb/valid --out data/det --split val

Wichtig: Training und Messlatte muessen aus VERSCHIEDENEN Splits kommen (train gegen
valid/test), sonst misst man Gelerntes. Das ist hier automatisch so, weil das Werkzeug den
Split als Argument nimmt und die Dateien getrennt liegen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import detmath as dm
import signmap


def lesen(coco_dir: Path) -> tuple[dict, list[dict]]:
    """COCO-JSON lesen und je Bild die zugeordneten Boxen sammeln."""
    pfad = coco_dir / "_annotations.coco.json"
    if not pfad.exists():
        raise SystemExit(f"{pfad} fehlt - ist {coco_dir} eine entpackte COCO-Fassung?")
    daten = json.loads(pfad.read_text(encoding="utf-8"))
    kategorien = {int(c["id"]): str(c["name"]) for c in daten["categories"]}
    nach_bild: dict[int, list[dict]] = {}
    for a in daten["annotations"]:
        nach_bild.setdefault(int(a["image_id"]), []).append(a)
    bilder = []
    for b in daten["images"]:
        boxen = []
        for a in nach_bild.get(int(b["id"]), []):
            label = signmap.label_for_gtsdb(kategorien.get(int(a["category_id"]), ""))
            if label:
                boxen.append((label, a["bbox"]))
        bilder.append({"datei": Path(b["file_name"]).name, "boxen": boxen})
    return kategorien, bilder


def bild_und_boxen(pfad: Path, boxen: list, size: int, min_px: float) -> tuple:
    """Ein Bild auf `size` einpassen und die Boxen mitrechnen (gleiche Mathematik wie ueberall)."""
    with Image.open(pfad) as f:
        arr = np.asarray(f.convert("RGB"), dtype=np.uint8)
    arr, s, px, py = dm.letterbox_array(arr, size)
    raus, labels = [], []
    for label, (x, y, w, h) in boxen:
        x0, y0 = x * s + px, y * s + py
        x1, y1 = (x + w) * s + px, (y + h) * s + py
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(size), x1), min(float(size), y1)
        if x1 - x0 < min_px or y1 - y0 < min_px:
            continue                    # nach dem Verkleinern nicht mehr erkennbar
        raus.append({"label": label, "cx": (x0 + x1) / 2 / size, "cy": (y0 + y1) / 2 / size,
                     "w": (x1 - x0) / size, "h": (y1 - y0) / size})
        labels.append(label)
    return arr, raus, labels


def build(coco_dir: str | Path, out_dir: str | Path, split: str, size: int = 320,
          min_px: float = 6.0, limit: int = 0) -> dict:
    coco_dir, out = Path(coco_dir), Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    kategorien, bilder = lesen(coco_dir)
    # Merkung je QUELLORDNER, nicht nur je Split: valid und test landen beide im Split val,
    # und ohne diesen Unterschied wuerde der zweite Aufruf die Bilder des ersten ersetzen.
    merkung = f"gtsdb-{coco_dir.name}"
    zaehler: dict[str, int] = {}
    ohne_box, geschrieben, leergefallen = 0, 0, 0
    entries: list[dict] = []
    for i, b in enumerate(bilder[:limit] if limit else bilder):
        quelle = coco_dir / b["datei"]
        if not quelle.exists():
            continue
        if not b["boxen"]:
            ohne_box += 1
        arr, boxen, _ = bild_und_boxen(quelle, b["boxen"], size, min_px)
        if b["boxen"] and not boxen:
            leergefallen += 1               # alle Zeichen zu klein fuer das Netz
        stem = f"{split}_g{i:05d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=90)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(
            f"{signmap.CLASS_ID[x['label']]} {x['cx']:.6f} {x['cy']:.6f} {x['w']:.6f} {x['h']:.6f}"
            for x in boxen) + "\n", encoding="utf-8")
        for x in boxen:
            zaehler[x["label"]] = zaehler.get(x["label"], 0) + 1
        entries.append({"id": stem, "src": "GTSDB", "license": "CC BY 4.0 (Roboflow-Fassung)",
                        "scene": merkung, "conditions": ["echteSzene"], "split": split,
                        "width": size, "height": size, "boxes": len(boxen)})
        geschrieben += 1

    pfad = out / "manifest.json"
    alt = json.loads(pfad.read_text(encoding="utf-8"))["images"] if pfad.exists() else []
    gemischt = [e for e in alt if not (e.get("scene") == merkung and e.get("split") == split)]
    gemischt += entries
    pfad.write_text(json.dumps(
        {"format": "yolo-txt", "classes": signmap.LABELS, "source": "tools/gtsdb_dataset.py",
         "images": gemischt}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"neu": geschrieben, "klassen": len(zaehler), "label": zaehler,
            "ohne_box": ohne_box, "leergefallen": leergefallen,
            "bilder_gesamt": len(gemischt), "kategorien": len(kategorien)}


def main() -> None:
    ap = argparse.ArgumentParser(description="GTSDB-Szenen -> Detektor-Datensatz (YOLO)")
    ap.add_argument("--coco", required=True, help="Ordner mit _annotations.coco.json")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--min-px", type=float, default=6.0,
                    help="kleinste Kastenlaenge nach dem Einpassen (darunter faellt die Box weg)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    info = build(args.coco, args.out, args.split, args.size, args.min_px, args.limit)
    print(f"[gtsdb] {info['neu']} echte Szenen ({args.split}) nach {args.out}, "
          f"{info['klassen']} Klassen belegt, {info['ohne_box']} ohne Schild")
    if info["leergefallen"]:
        print(f"[hinweis] {info['leergefallen']} Bilder hatten nur zu kleine Zeichen "
              f"(< {args.min_px} px nach Einpassen auf {args.size})")
    print(f"[gtsdb] manifest: {info['bilder_gesamt']} Bilder insgesamt")


if __name__ == "__main__":
    main()