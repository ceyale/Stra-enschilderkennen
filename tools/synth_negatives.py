"""
tools/synth_negatives.py - Bilder OHNE jedes Schild (leeres Label) in einen Datensatz haengen.

Warum: Gemessen hatte das Netz 226 Fehlalarme auf 2000 val-Bildern. Davon lagen 195 auf
echten Schildern (falsche Klasse), aber 31 auf reinem Hintergrund - es hatte nie gelernt,
was KEIN Schild ist. Zusaetzlich stabilisiert eine gesunde Negativquote die
Objektivitaet: ohne Negative gibt es im Training kein einziges wirklich leeres Ziel.

Zwei Sorten, beide aus derselben Hintergrundverteilung wie die Positivbilder
(tools/synth_data.py), damit das Netz nicht "Hintergrund" statt "Schild" lernt:
  * leer  - nur Hintergrund (Asphalt, Himmel, Wand, Gruen, Nacht, Stoererflaeche)
  * hart  - schildaehnliche Stoerer: Rueckleuchten, Ampel, Baustelle, Leuchtreklame,
            graue Schildrueckseite, rote Flagge, gelbe Richtungstafel, farbige Kreise

Die Bilder landen in einen BESTEHENDEN Datensatzordner (images/, labels/, manifest.json).
Wiederholte Aufrufe ersetzen die zuvor von diesem Skript erzeugten Eintraege.
Labels sind bewusst LEER (0 Byte) - das uebliche YOLO-Zeichen fuer "kein Objekt".

Aufruf:
    python tools/synth_negatives.py --out data/det --n 2500 --split train
    python tools/synth_negatives.py --out data/det --n 400 --split neg   # Messlatte
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

import synth_data as sd
from hybrid_net import SIGN_LABELS

SRC = "synthetisch (Negativ)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bilder ohne Schild (Negative) erzeugen")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--split", default="train", choices=["train", "val", "neg"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--hard-prob", type=float, default=0.45,
                    help="Anteil mit schildaehnlichem Stoerer (statt nur Hintergrund)")
    ap.add_argument("--degrade", type=float, default=0.6)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    kinds: dict[str, int] = {}
    entries = []
    for i in range(args.n):
        arr, boxes, tags = sd.compose_negative(rng, args.size, hard_prob=args.hard_prob,
                                               degrade_prob=args.degrade)
        stem = f"{args.split}_n{i:06d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=86)
        # Leeres Label = kein Objekt im Bild (YOLO-Konvention).
        (out / "labels" / f"{stem}.txt").write_text("\n".join(
            f"{SIGN_LABELS.index(b['label'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in boxes) + "\n", encoding="utf-8")
        hard = next((t for t in tags if t.startswith("hart:")), "leer")
        kinds[hard] = kinds.get(hard, 0) + 1
        entries.append({"id": stem, "src": SRC, "license": "eigene Quelle (generiert)",
                        "scene": "synth", "conditions": tags, "split": args.split,
                        "width": args.size, "height": args.size, "boxes": len(boxes),
                        "negativ": True, "art": hard})

    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
        {"format": "yolo-txt", "classes": SIGN_LABELS, "source": "tools/synth_negatives.py",
         "images": []}
    manifest["images"] = [e for e in manifest.get("images", [])
                          if not (e.get("src") == SRC and e.get("split") == args.split)] + entries
    manifest["negatives"] = {**manifest.get("negatives", {}), args.split:
                             {"n": args.n, "hart": args.hard_prob, "arten": kinds}}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[negative] {len(entries)} Bilder ohne Schild nach {out} ({args.split}); "
          + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"[negative] manifest: {len(manifest['images'])} Bilder insgesamt, "
          f"davon {sum(1 for e in manifest['images'] if e.get('negativ'))} Negative")


if __name__ == "__main__":
    main()
