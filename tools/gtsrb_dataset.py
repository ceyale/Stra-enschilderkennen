"""
tools/gtsrb_dataset.py - Echten Detektor-Datensatz aus GTSRB bauen.

Warum so? GTSRB liefert 39 209 **echte** deutsche Schilder als Ausschnitte (mit ROI und
Klasse), aber keine Umgebungen. Fuer ein Netz, das Position UND Art finden soll, braucht
es aber Vollbilder mit Box. Deshalb: echten Ausschnitt auf einen Hintergrund kleben
(Asphalt, Himmel, Stoerer, Verdeckung, Verschlechterungen aus tools/synth_data.py) und die
Box exakt mitschreiben. Ergebnis: echte Schilderoptik in echten Bedingungen, lizenzfrei.

Die GTSRB-Klassen werden auf die 9 Typen dieses Projekts abgebildet (siehe CLASS_MAP).
Nicht abbildbare Klassen (Ende-Schilder) werden verworfen - das Netz soll sie nicht
lernen, sonst ziehen sie die Objektivitaet ins Falsche.

Aufruf:
    python tools/gtsrb_dataset.py --zip data/gtsrb/gtsrb-train.zip --out data/det --n 20000
    python tools/gtsrb_dataset.py --zip data/gtsrb/gtsrb-test.zip  --out data/det --n 4000 --split val
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import synth_data as sd
from hybrid_net import SIGN_LABELS

# GTSRB-ClassId -> Label dieses Projekts. Alles Fehlende fliegt raus.
CLASS_MAP: dict[int, str] = {
    **{c: "verbot" for c in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16]},   # Tempolimits, Ueberholverbot, Fahrverbote
    11: "vorfahrtGewaehren",   # Vorfahrt an der naechsten Kreuzung (rotes Dreieck, Spitze unten)
    12: "vorfahrtstrasse",     # Vorfahrtstrasse (gelbe Raute)
    13: "vorfahrtGewaehren",   # Vorfahrt gewaehren
    14: "stop",                # Stopp
    17: "einfahrtVerboten",    # Einfahrt verboten (roter Vollkreis)
    **{c: "warnung" for c in range(18, 32)},                                # Gefahrzeichen (Dreieck, Spitze oben)
    **{c: "gebot" for c in [33, 34, 35, 36, 37, 38, 39, 40]},               # Gebotszeichen (blauer Kreis)
}
# 32, 41, 42 = "Ende von ..." (grau/weiss) -> nicht Teil der 9 Typen
SKIP = {32, 41, 42}


def read_rows(zf: zipfile.ZipFile) -> list[dict]:
    """Alle GT-*.csv aus dem Archiv lesen (Spalten: Filename, Roi.X1..Y2, ClassId).

    Beide Archivformen von GTSRB werden unterstuetzt: Trainings-Zip mit Klassenordnern
    (…/Images/00000/00000_00000.ppm) und Test-Zip ohne (…/Final_Test/Images/00000.ppm).
    """
    names = set(zf.namelist())
    rows: list[dict] = []

    def resolve(csv_dir: str, filename: str) -> str | None:
        for cand in (f"{csv_dir}/{filename}", f"{csv_dir}/Images/{filename}",
                     f"{csv_dir.rsplit('/', 1)[0]}/Images/{filename}"):
            if cand in names:
                return cand
        return None

    for name in zf.namelist():
        base = name.split("/")[-1]
        if not (base.startswith("GT-") and base.endswith(".csv")):
            continue
        folder = name.rsplit("/", 1)[0]
        text = zf.read(name).decode("utf-8", "replace")
        for row in csv.DictReader(io.StringIO(text), delimiter=";"):
            if "ClassId" not in row:          # Test-Zip liefert keine Labels -> unbrauchbar
                continue
            cls = int(row["ClassId"])
            if cls in SKIP or cls not in CLASS_MAP:
                continue
            path = resolve(folder, row["Filename"])
            if path is None:
                continue
            rows.append({
                "path": path,
                "x1": int(row["Roi.X1"]), "y1": int(row["Roi.Y1"]),
                "x2": int(row["Roi.X2"]), "y2": int(row["Roi.Y2"]),
                "cls": cls, "label": CLASS_MAP[cls],
            })
    return rows


def background_pool(rng: random.Random, size: int, n: int = 40) -> list[np.ndarray]:
    """Hintergruende einmal vorberechnen - Rauschen pro Bild waere der Flaschenhals."""
    return [sd.random_background(rng, size) for _ in range(n)]


def compose(zf: zipfile.ZipFile, picks: list[dict], rng: random.Random, size: int,
            bgs: list[np.ndarray], degrade_prob: float, occlude_prob: float):
    """Ein Vollbild aus echten GTSRB-Ausschnitten bauen; Rueckgabe (Bild, Boxen, Tags).

    picks = 1 oder 2 Zeilen (zwei Schilder im Bild sind fuer die Objektivitaet wichtig,
    weil die Heuristik dort doppelte Treffer liefert).
    """
    canvas = Image.fromarray(bgs[rng.randrange(len(bgs))].copy())
    tags: list[str] = []
    boxes: list[dict] = []
    if len(picks) > 1:
        tags.append("mehrere")

    for row in picks:
        with zf.open(row["path"]) as fh:
            sign = Image.open(io.BytesIO(fh.read())).convert("RGB")
        sign = sign.crop((row["x1"], row["y1"], row["x2"] + 1, row["y2"] + 1))
        side = max(8, int(size * rng.uniform(0.035, 0.6)))        # Anteil am Bild: klein bis gross
        ratio = sign.height / max(1, sign.width)
        sign = sign.resize((side, max(6, int(side * ratio))), Image.LANCZOS)
        angle = rng.uniform(-25, 25) if rng.random() < 0.8 else rng.uniform(-60, 60)
        sign = sign.rotate(angle, resample=Image.BICUBIC, expand=True)
        if abs(angle) > 15:
            tags.append("roll")                                   # Roll: Bruchstelle der Heuristik
        px = rng.randrange(max(1, size - sign.width))
        py = rng.randrange(max(1, size - sign.height))
        canvas.paste(sign, (px, py))
        boxes.append({"label": row["label"], "cx": (px + sign.width / 2) / size,
                      "cy": (py + sign.height / 2) / size,
                      "w": sign.width / size, "h": sign.height / size})

    arr = np.asarray(canvas).copy()
    if rng.random() < degrade_prob:
        arr, t2 = sd.degrade(arr, rng)
        tags += t2
    if rng.random() < occlude_prob:
        arr, t2 = sd.add_occluder(arr, rng)
        tags += t2
    return arr, boxes, sorted(set(tags))


def build(zip_path: str, out_dir: str, n: int, size: int, split: str, seed: int = 0,
          degrade_prob: float = 0.85, occlude_prob: float = 0.3, val_share: float = 0.15) -> dict:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    zf = zipfile.ZipFile(zip_path)
    rows = read_rows(zf)
    if not rows:
        raise SystemExit(f"keine verwertbaren Annotationen in {zip_path}")
    # Kein Datenleck UND keine Klassenlücke: je Klasse die letzten val_share der
    # Aufnahmen (zusammenhaengende Sequenzen) abtrennen. Ein Split ueber die gesamte
    # Liste wuerde nur die letzten Klassen in die Validierung legen.
    rows.sort(key=lambda r: r["path"])
    if split == "val":
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["path"].rsplit("/", 2)[-2], []).append(r)
        val_rows: list[dict] = []
        for key in sorted(groups):
            g = groups[key]
            val_rows += g[int(len(g) * (1 - val_share)):]
        rows = val_rows
    if not rows:
        raise SystemExit(f"keine verwertbaren Zeilen in {zip_path}")
    bgs = background_pool(rng, size)
    entries, tags_all = [], {}
    for i in range(n):
        picks = [rows[rng.randrange(len(rows))] for _ in range(2 if rng.random() < 0.25 else 1)]
        arr, boxes, tags = compose(zf, picks, rng, size, bgs, degrade_prob, occlude_prob)
        stem = f"{split}_{i:06d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=86)
        lines = [f"{SIGN_LABELS.index(b['label'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
                 for b in boxes]
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for t in tags:
            tags_all[t] = tags_all.get(t, 0) + 1
        entries.append({"id": stem, "src": "GTSRB", "license": "frei fuer Forschung",
                        "scene": picks[0]["path"].rsplit("/", 2)[-2], "conditions": tags,
                        "split": split, "width": size, "height": size,
                        "boxes": len(boxes), "gtsrb_class": picks[0]["cls"]})
    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))["images"] if manifest_path.exists() else []
    merged = [e for e in old if e.get("split") != split] + entries
    manifest_path.write_text(json.dumps(
        {"format": "yolo-txt", "classes": SIGN_LABELS, "source": "tools/gtsrb_dataset.py",
         "images": merged}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"neu": len(entries), "klassen": len({r["cls"] for r in rows}), "bedingungen": tags_all}


def main() -> None:
    ap = argparse.ArgumentParser(description="GTSRB -> Detektor-Datensatz (echte Schilder, eigene Hintergruende)")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade", type=float, default=0.85)
    ap.add_argument("--occlude", type=float, default=0.3)
    args = ap.parse_args()
    info = build(args.zip, args.out, args.n, args.size, args.split, args.seed, args.degrade, args.occlude)
    print(f"[daten] {info['neu']} Bilder ({args.split}) nach {args.out} geschrieben, "
          f"{info['klassen']} GTSRB-Klassen genutzt")
    print("[daten] Bedingungen:", ", ".join(f"{k}={v}" for k, v in sorted(info["bedingungen"].items())))


if __name__ == "__main__":
    main()
