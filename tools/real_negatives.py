"""
tools/real_negatives.py - Bilder OHNE Schild aus ECHTEN Fotos in den Datensatz haengen.

Warum: Mit data/_test_bilder.py gemessen lieferte ein echtes Zimmer-/Monitorfoto (Test/
Nothing.jpg) zwei bis drei Fehlalarme mit Boxen von HALBER Bildgroesse, obwohl dort kein
einziges Schild steht. Die synthetischen Negative (tools/synth_negatives.py) halfen dagegen
nicht, weil sie nicht aus der Verteilung des Einsatzes stammen: das Netz kannte kein echtes
Foto ohne Schild. Aus jedem echten Foto werden hier viele Zufallsausschnitte (mit
Verschlechterungen) erzeugt - jeder Ausschnitt ist ein vollstaendiges Trainingsbild mit
leerem Label.

Quellen:
  * --real-train / --real-test  eigene Fotos (z. B. Test/Nothing.jpg)
  * --coco                      coco128 (128 echte Fotos, CC BY 4.0), geholt von
                                data/negatives/fetch_coco.ps1
Bilder, die in COCO mit Klasse 11 (stop sign) beschriftet sind, werden verworfen - sonst
wuerde ein echtes Stoppschild als "kein Objekt" gelernt.

Aufteilung ohne Selbstbetrug: --train-region und --test-region erlauben es, aus DEMSELBEN
Foto verschiedene Bildbereiche zu verwenden (train = obere Haelfte, Test = untere Haelfte).
Bei coco128 bleiben die letzten --coco-test-n Bilder (nach Name sortiert) IMMER aus dem
Training draussen und werden nur im Split neg benutzt.

Aufruf:
    python tools/real_negatives.py --out data/det --n 3600 --split train \
        --real-train Test/Nothing.jpg --train-region 0,0,1,0.5 \
        --coco data/negatives/coco128 --coco-test-n 12 --user-share 0.55
    python tools/real_negatives.py --out data/det --n 600 --split neg \
        --real-test Test/Nothing.jpg --test-region 0,0.5,1,1 \
        --coco data/negatives/coco128 --coco-test-n 12 --user-share 0.35
    python tools/real_negatives.py --out data/det --n 24 --split neg --montage data/x.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import detmath as dm
from hybrid_net import SIGN_LABELS

SRC = "echt (Negativ)"
COCO_STOP_KLASSE = 11          # COCO-Index 11 = "stop sign"; Bilder damit fliegen raus
BILD_ENDUNGEN = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _region(spec: str, w: int, h: int) -> tuple[int, int, int, int]:
    """'x0,y0,x1,y1' als Anteile 0..1 -> Pixelbereich (fuer Train/Test-Trennung am Foto)."""
    if not spec:
        return 0, 0, w, h
    x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def coco_bilder(root: Path) -> tuple[list[Path], list[str]]:
    """Bilder + Namen aus einem coco128-Ordner, Stoppschild-Bilder aussortiert."""
    img_dir = root / "images" / "train2017"
    lab_dir = root / "labels" / "train2017"
    if not img_dir.exists():
        img_dir, lab_dir = root / "images", root / "labels"
    bilder, verworfen = [], []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in BILD_ENDUNGEN:
            continue
        lab = lab_dir / (p.stem + ".txt")
        if lab.exists():
            klassen = {int(z.split()[0]) for z in lab.read_text(encoding="utf-8").splitlines()
                       if z.strip()}
            if COCO_STOP_KLASSE in klassen:
                verworfen.append(p.name)
                continue
        bilder.append(p)
    return bilder, verworfen


def ausschnitt(arr: np.ndarray, rng: random.Random, vollbild_prob: float,
               lo: float, hi: float) -> np.ndarray:
    """Zufaelliges Fenster aus dem Bild (oder das ganze Bild)."""
    h, w = arr.shape[:2]
    if rng.random() < vollbild_prob:
        return arr
    fw, fh = max(24, int(w * rng.uniform(lo, hi))), max(24, int(h * rng.uniform(lo, hi)))
    fw, fh = min(fw, w), min(fh, h)
    x0 = rng.randrange(0, w - fw + 1)
    y0 = rng.randrange(0, h - fh + 1)
    return arr[y0:y0 + fh, x0:x0 + fw]


def verschlechtere(img: Image.Image, rng: random.Random, size: int) -> tuple[Image.Image, list[str]]:
    """Handyfoto-typische Verschlechterungen: Licht, Farbe, Wackler, Rauschen, Rand."""
    tags: list[str] = []
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.6:
        k = rng.uniform(0.4, 1.3)
        img = ImageEnhance.Brightness(img).enhance(k)
        tags.append("dunkel" if k < 0.7 else ("hell" if k > 1.1 else "licht"))
    if rng.random() < 0.45:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.2, 1.3))
        tags.append("verblichen")
    if rng.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.8)))
        tags.append("unschaerfe")
    if rng.random() < 0.25:
        a = np.asarray(img, dtype=np.float32)
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 14), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        tags.append("rauschen")
    if rng.random() < 0.3:
        # Grauer Rand wie beim Letterbox im Browser (gleiche Farbe 114) - auch der Rand darf
        # kein Objekt werden.
        f = rng.uniform(0.55, 0.9)
        nw, nh = max(32, int(size * f)), max(32, int(size * f))
        klein = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (size, size), (dm.LETTERBOX_GREY,) * 3)
        canvas.paste(klein, (rng.randrange(0, size - nw + 1), rng.randrange(0, size - nh + 1)))
        img = canvas
        tags.append("rand")
    return img, tags


def main() -> None:
    ap = argparse.ArgumentParser(description="echte Fotos als Negative (leeres Label)")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--split", default="train", choices=["train", "neg"])
    # Herkunftskennzeichen im Manifest. Es entscheidet, WELCHE frueheren Eintraege dieser
    # Aufruf ersetzt: gleicher src UND gleicher split = Ersetzen, sonst Hinzufuegen. Zwei
    # Negative-Aufrufe auf denselben Split addieren sich also nur mit unterschiedlichem src -
    # ohne eigene Kennung haette der zweite Aufruf die Bilder des ersten stillschweigend
    # geloescht (so stand der Lauf vom 03.10. bei 32 683 statt 36 283 Trainingsbildern).
    ap.add_argument("--src", default=SRC,
                    help="Herkunftskennzeichen im Manifest (Standard: 'echt (Negativ)')")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--real-train", nargs="*", default=[], help="eigene Fotos fuers Training")
    ap.add_argument("--real-test", nargs="*", default=[], help="eigene Fotos fuer die Messlatte")
    ap.add_argument("--train-region", default="", help="x0,y0,x1,y1 als Anteile (Trainingsbereich)")
    ap.add_argument("--test-region", default="", help="x0,y0,x1,y1 als Anteile (Testbereich)")
    ap.add_argument("--coco", default="", help="coco128-Ordner")
    ap.add_argument("--coco-test-n", type=int, default=12)
    ap.add_argument("--user-share", type=float, default=0.5, help="Anteil eigener Fotos")
    ap.add_argument("--vollbild", type=float, default=0.25, help="Anteil ganzer Bilder (ohne Ausschnitt)")
    ap.add_argument("--fenster", type=float, nargs=2, default=[0.35, 1.0],
                    help="Fenstergroesse als Anteil (min max)")
    ap.add_argument("--montage", default="", help="Sichtpruefung als Bilddatei ablegen")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kennung = "".join(z for z in args.src.lower() if z.isalnum())[:8] or "neg"
    nutzer_pfade = args.real_train if args.split == "train" else args.real_test
    region = args.train_region if args.split == "train" else args.test_region
    nutzer: list[tuple[Path, tuple[int, int, int, int], str]] = []
    for p in nutzer_pfade:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"Bild fehlt: {path}")
        im = Image.open(path).convert("RGB")
        nutzer.append((path, _region(region, *im.size), f"echt:{path.stem.lower()}"))
    coco: list[tuple[Path, tuple[int, int, int, int], str]] = []
    if args.coco:
        bilder, verworfen = coco_bilder(Path(args.coco))
        test_namen = set(p.stem for p in bilder[-args.coco_test_n:]) if args.coco_test_n else set()
        auswahl = [p for p in bilder if (p.stem in test_namen) == (args.split == "neg")]
        coco = [(p, (0, 0, 0, 0), "echt:coco") for p in auswahl]
        print(f"[coco] {len(bilder)} Fotos, {len(verworfen)} mit Stoppschild verworfen; "
              f"fuer split={args.split}: {len(coco)} Fotos; gesperrte Testfotos (immer nur im "
              f"Split neg): {len(test_namen)} (z. B. {sorted(test_namen)[:2]})")
    if not nutzer and not coco:
        raise SystemExit("keine Quelle: --real-train/--real-test oder --coco angeben")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    zellen: list[Image.Image] = []
    arten: dict[str, int] = {}
    for i in range(args.n):
        if nutzer and (not coco or rng.random() < args.user_share):
            path, reg, art = rng.choice(nutzer)
        else:
            path, _, art = rng.choice(coco)
            reg = (0, 0, *Image.open(path).size)
        im = Image.open(path).convert("RGB").crop(reg)
        crop = ausschnitt(np.asarray(im), rng, args.vollbild, *args.fenster)
        bild = Image.fromarray(crop).resize((args.size, args.size), Image.BICUBIC)
        bild, tags = verschlechtere(bild, rng, args.size)
        # Der Dateiname traegt die Herkunft mit: ohne sie schreiben zwei Aufrufe auf denselben
        # Split dieselben Namen (train_r000000 ...) und ueberschreiben einander, waehrend das
        # Manifest beide Eintraege behaelt - zwei Eintraege, eine Datei. Genau so entstehen
        # doppelte Bilder im Datensatz.
        stem = f"{args.split}_{kennung}_r{i:06d}"
        bild.save(out / "images" / f"{stem}.jpg", quality=rng.randrange(58, 92))
        # Leeres Label: kein Objekt im Bild (YOLO-Konvention, siehe tools/synth_negatives.py).
        (out / "labels" / f"{stem}.txt").write_text("\n", encoding="utf-8")
        if len(zellen) < 24:
            zellen.append(bild.copy())
        arten[art] = arten.get(art, 0) + 1
        entries.append({"id": stem, "src": args.src,
                        "license": "COCO 2017 (CC BY 4.0) / eigene Aufnahme",
                        "scene": "echt", "conditions": sorted(set(["negativ", "echt"] + tags)),
                        "split": args.split, "width": args.size, "height": args.size,
                        "boxes": 0, "negativ": True, "art": art, "quelle": path.name})

    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
        {"format": "yolo-txt", "classes": SIGN_LABELS, "source": "tools/real_negatives.py",
         "images": []}
    manifest["images"] = [e for e in manifest.get("images", [])
                          if not (e.get("src") == args.src
                                  and e.get("split") == args.split)] + entries
    # Schluessel aus Split UND Kennung: sonst ueberschreibt der zweite Aufruf auf denselben
    # Split die Aufzeichnung des ersten.
    manifest["echte_negative"] = {**manifest.get("echte_negative", {}),
                                  f"{args.split}|{args.src}":
                                  {"n": args.n, "arten": arten, "coco_test_n": args.coco_test_n,
                                   "region": region}}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.montage:
        Path(args.montage).parent.mkdir(parents=True, exist_ok=True)
        montage(zellen).save(args.montage)
        print(f"[montage] {args.montage}")
    print(f"[echt-negativ] {len(entries)} Bilder ohne Schild nach {out} ({args.split}); "
          + ", ".join(f"{k}={v}" for k, v in sorted(arten.items())))
    print(f"[echt-negativ] manifest: {len(manifest['images'])} Bilder insgesamt, "
          f"davon {sum(1 for e in manifest['images'] if e.get('negativ'))} Negative")


def montage(zellen: list[Image.Image], spalten: int = 6) -> Image.Image:
    """Sichtpruefung: die ersten Negative als Bildtafel (--montage)."""
    w, h = zellen[0].size
    reihen = (len(zellen) + spalten - 1) // spalten
    tafel = Image.new("RGB", (spalten * w, reihen * h), (10, 10, 10))
    for i, z in enumerate(zellen):
        tafel.paste(z, ((i % spalten) * w, (i // spalten) * h))
    return tafel


if __name__ == "__main__":
    main()