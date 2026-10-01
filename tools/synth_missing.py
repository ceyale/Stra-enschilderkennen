"""
tools/synth_missing.py - synthetische Bilder fuer Typen, die GTSRB NICHT enthaelt.

Warum: GTSRB hat keine blauen Hinweiszeichen (Z 3xx) und keine gelbe Ortstafel (Z 310).
Ohne Nachschub kennt das Netz diese beiden der neun Typen nie - im KI-Modus waeren
Rechtecke aus blau/gelb also grundsaetzlich unerreichbar. Beide sind geometrisch einfach,
deshalb genuegen die synthetischen Kacheln aus tools/synth_data.py.

Die Bilder werden in einen BESTEHENDEN Datensatzordner gehaengt (images/, labels/,
manifest.json) - dorthin, wo tools/gtsrb_dataset.py gebaut hat. Wiederholte Aufrufe
ersetzen die zuvor von diesem Skript erzeugten Eintraege, statt sie zu haeufen.

Aufruf:
    python tools/synth_missing.py --out data/det --n 3000                 # train
    python tools/synth_missing.py --out data/det --n 200 --split val     # Messlatte dazu
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

import synth_data as sd
from gtsrb_dataset import CLASS_MAP
from hybrid_net import SIGN_LABELS

SRC = "synthetisch (Lueckenschluss)"


def missing_labels() -> list[str]:
    """Typen, die in GTSRB keine Entsprechung haben (siehe CLASS_MAP)."""
    covered = set(CLASS_MAP.values())
    return [name for name in SIGN_LABELS if name not in covered]


def main() -> None:
    ap = argparse.ArgumentParser(description="Typen, die GTSRB nicht hat, synthetisch ergaenzen")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--degrade", type=float, default=0.85)
    ap.add_argument("--occlude", type=float, default=0.3)
    args = ap.parse_args()

    fehlend = missing_labels()
    if not fehlend:
        raise SystemExit("GTSRB deckt alle Typen ab - nichts zu ergaenzen")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    tags_all: dict[str, int] = {}
    entries = []
    for i in range(args.n):
        arr, boxes, tags = sd.compose_sample(rng, args.size, degrade_prob=args.degrade,
                                            occlude_prob=args.occlude, labels=fehlend)
        stem = f"{args.split}_m{i:06d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=86)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(
            f"{SIGN_LABELS.index(b['label'])} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in boxes) + "\n", encoding="utf-8")
        for t in tags:
            tags_all[t] = tags_all.get(t, 0) + 1
        entries.append({"id": stem, "src": SRC, "license": "eigene Quelle (generiert)",
                        "scene": "synth", "conditions": tags, "split": args.split,
                        "width": args.size, "height": args.size, "boxes": len(boxes)})

    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
        {"format": "yolo-txt", "classes": SIGN_LABELS, "source": "tools/synth_missing.py",
         "images": []}
    manifest["images"] = [e for e in manifest.get("images", [])
                          if not (e.get("src") == SRC and e.get("split") == args.split)] + entries
    manifest["missing_synth"] = {"classes": fehlend, "n": args.n, "split": args.split}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[lueckenschluss] {len(entries)} Bilder fuer {', '.join(fehlend)} nach {out} "
          f"({args.split}); Bedingungen: " + ", ".join(f"{k}={v}" for k, v in sorted(tags_all.items())))
    print(f"[lueckenschluss] manifest: {len(manifest['images'])} Bilder insgesamt")


if __name__ == "__main__":
    main()
