"""
tools/synth_data.py - Synthetische Schilder fuer Training, Rauchtest und Export-Pruefung.

Zweck:
  1. Die Trainings-/Export-Kette ohne den echten Datensatz testbar machen
     (data/det/ wird erst mit eigenen Fotos gefuellt, siehe docs/TRAINING.md).
  2. Gezielt die Bedingungen erzeugen, die die Heuristik nachweislich brechen:
     Dunkelheit, Entsaettigung, Bewegungsunschaefe, Roll-Verdrehung, Verdeckung, Blendung.
  3. Feste Beispiele fuer den Paritaetscheck (PyTorch vs. ONNX) liefern.

Aufruf:
    python tools/synth_data.py --out data/synth --n 200
Erzeugt data/synth/images/*.jpg, data/synth/labels/*.txt (YOLO-Format) und manifest.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from hybrid_net import SIGN_LABELS

CLASS_ID = {name: i for i, name in enumerate(SIGN_LABELS)}
RED = (200, 30, 40)
BLUE = (20, 90, 180)
YELLOW = (245, 195, 0)
WHITE = (245, 245, 240)

# Roll-Politik (siehe docs/TRAINING.md 9.4): ein rotes Dreieck ist bei starker Drehung
# nicht mehr von seinem Gegenstueck zu unterscheiden - die Spitze zeigt dann zur Seite.
# Solche Bilder waeren widerspruechliche Lernziele, deshalb bleiben Dreiecke mild.
TRIANGLE_LABELS = ("warnung", "vorfahrtGewaehren")
ROLL_MILD = 35.0     # Dreiecke: bis hierher ist "Spitze oben/unten" eindeutig
ROLL_STARK = 70.0    # Kreise, Rechtecke, Rauten: bleiben auch stark gedreht eindeutig


def roll_angle(rng: random.Random, label: str) -> float:
    """Roll-Winkel fuer ein Schild.

    Gemessen: `warnung` verliert 18 Recall-Punkte, wenn bis +-70 Grad gedreht wird,
    `vorfahrtGewaehren` bleibt dagegen auch mild schlecht. Deshalb wird nach Form
    entschieden: Dreiecke mild, alles andere weiterhin stark (die Heuristik bricht
    genau an starker Drehung - dort soll das Netz stark bleiben).
    """
    if label in TRIANGLE_LABELS:
        return rng.uniform(-ROLL_MILD, ROLL_MILD)
    if rng.random() < 0.3:
        return rng.uniform(-ROLL_STARK, ROLL_STARK)
    return rng.uniform(-ROLL_MILD, ROLL_MILD)



def _poly_regular(cx: float, cy: float, r: float, n: int, rot: float = 0.0) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def sign_patch(name: str, size: int = 128) -> Image.Image:
    """Ein Schild als RGBA-Kachel (transparenter Rand). Geometrie = deutsche Zeichen."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = size / 2, size * 0.40

    def disc(radius: float, color, outline=None, width: int = 6):
        d.ellipse([c - radius, c - radius, c + radius, c + radius], fill=color, outline=outline, width=width)

    if name == "stop":
        d.polygon(_poly_regular(c, c, r * 1.05, 8, math.pi / 8), fill=RED, outline=WHITE)
        d.text((c - 26, c - 8), "STOP", fill=WHITE)
    elif name == "vorfahrtGewaehren":
        tri = [(c, c + r * 1.15), (c - r * 1.15, c - r * 0.85), (c + r * 1.15, c - r * 0.85)]
        d.polygon(tri, fill=RED, outline=WHITE)
        inner = [(c, c + r * 0.65), (c - r * 0.65, c - r * 0.5), (c + r * 0.65, c - r * 0.5)]
        d.polygon(inner, fill=WHITE)
    elif name == "warnung":
        tri = [(c, c - r * 1.15), (c - r * 1.15, c + r * 0.85), (c + r * 1.15, c + r * 0.85)]
        d.polygon(tri, fill=RED, outline=WHITE)
        inner = [(c, c - r * 0.6), (c - r * 0.6, c + r * 0.45), (c + r * 0.6, c + r * 0.45)]
        d.polygon(inner, fill=WHITE)
        d.polygon([(c, c - r * 0.35), (c - r * 0.28, c + r * 0.3), (c + r * 0.28, c + r * 0.3)], fill=(40, 40, 40))
    elif name == "verbot":
        disc(r, RED)
        disc(r * 0.75, WHITE)
        d.text((c - 12, c - 10), "30", fill=(40, 40, 40))
    elif name == "einfahrtVerboten":
        disc(r, RED)
        d.rectangle([c - r * 0.62, c - r * 0.16, c + r * 0.62, c + r * 0.16], fill=WHITE)
    elif name == "gebot":
        disc(r, BLUE)
        d.polygon([(c, c - r * 0.6), (c - r * 0.45, c + r * 0.1), (c - r * 0.15, c + r * 0.1),
                   (c - r * 0.15, c + r * 0.55), (c + r * 0.15, c + r * 0.55), (c + r * 0.15, c + r * 0.1),
                   (c + r * 0.45, c + r * 0.1)], fill=WHITE)
    elif name == "hinweis":
        d.rounded_rectangle([c - r * 0.95, c - r * 0.7, c + r * 0.95, c + r * 0.7], radius=8, fill=BLUE, outline=WHITE)
        d.rectangle([c - r * 0.28, c - r * 0.28, c + r * 0.28, c + r * 0.28], fill=WHITE)
    elif name == "vorfahrtstrasse":
        d.polygon([(c, c - r), (c + r, c), (c, c + r), (c - r, c)], fill=YELLOW, outline=WHITE)
        d.polygon([(c, c - r * 0.72), (c + r * 0.72, c), (c, c + r * 0.72), (c - r * 0.72, c)], fill=YELLOW)
    elif name == "ortstafel":
        d.rectangle([c - r * 1.05, c - r * 0.62, c + r * 1.05, c + r * 0.62], fill=YELLOW, outline=(40, 40, 40))
        for k in range(2):
            d.rectangle([c - r * 0.7, c - r * 0.28 + k * r * 0.42, c + r * 0.7, c - r * 0.12 + k * r * 0.42], fill=(40, 40, 40))
    else:
        raise ValueError(f"unbekanntes Schild: {name}")
    return img


def random_background(rng: random.Random, size: int, kind: str | None = None) -> np.ndarray:
    """Hintergrund ohne Schild: Asphalt, Himmel, Wand, Gruen, Nacht oder Stoererflaeche.

    Positiv- und Negativbilder ziehen aus DERSELBEN Verteilung (siehe
    tools/synth_negatives.py) - sonst koennte das Netz "Hintergrund" statt "Schild"
    lernen und die Negative wuerden die Positiven kaputt machen.
    """
    kind = kind or rng.choice(["asphalt", "asphalt", "sky", "busy", "wand", "gruen", "nacht"])
    rng_np = np.random.default_rng(rng.randrange(1 << 30))
    if kind == "asphalt":
        base = rng.uniform(70, 140)
        img = np.full((size, size, 3), base) + rng_np.normal(0, 9, (size, size, 3))
        if rng.random() < 0.4:                                  # Fahrbahnmarkierung
            y = rng.randrange(size)
            img[y:y + max(3, size // 40), :, :] = 235
    elif kind == "sky":
        g = np.linspace(150, 235, size)[:, None]
        img = np.stack([g * 0.7, g * 0.85, g * 1.0], axis=-1) * np.ones((1, size, 1))
        img = img + rng_np.normal(0, 4, (size, size, 3))
    elif kind == "wand":                                        # Beton, Putz, Fassade
        base = rng.uniform(95, 205)
        img = np.full((size, size, 3), base) + rng_np.normal(0, 6, (size, size, 3))
        if rng.random() < 0.5:                                  # Fugen / Steinreihen
            step = rng.randrange(max(6, size // 12), max(8, size // 5))
            img[::step, :, :] -= 18
            img[:, ::step, :] -= 12
    elif kind == "gruen":                                       # Hecke, Blaetter, Wiese
        img = np.stack([np.full((size, size), rng.uniform(30, 90)),
                        np.full((size, size), rng.uniform(70, 150)),
                        np.full((size, size), rng.uniform(20, 70))], axis=-1)
        img = img + rng_np.normal(0, 28, (size, size, 3))
    elif kind == "nacht":                                       # dunkel mit Lichtpunkten
        img = np.full((size, size, 3), rng.uniform(8, 45)) + rng_np.normal(0, 7, (size, size, 3))
        yy, xx = np.mgrid[0:size, 0:size]
        for _ in range(rng.randrange(1, 5)):
            cy, cx = rng.randrange(size), rng.randrange(size)
            rad = max(2.0, float(rng.randrange(max(3, size // 80), max(5, size // 25))))
            blob = np.clip(1 - np.hypot(xx - cx, yy - cy) / rad, 0, 1) ** 2
            img = img + blob[..., None] * np.array([245.0, 225.0, 170.0]) * rng.uniform(0.4, 1.0)
    else:                                                       # Stoerer in Schildfarben
        img = np.full((size, size, 3), rng.uniform(60, 150)) + rng_np.normal(0, 8, (size, size, 3))
        for _ in range(rng.randrange(2, 6)):
            col = rng.choice([RED, BLUE, YELLOW, (240, 240, 235), (60, 60, 60)])
            x0, y0 = rng.randrange(size), rng.randrange(size)
            w, h = rng.randrange(size // 8, size // 2), rng.randrange(size // 8, size // 2)
            img[y0:y0 + h, x0:x0 + w] = np.array(col)
    return np.clip(img, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Die gemessenen Schwachstellen der Heuristik gezielt nachbilden."""
    tags: list[str] = []
    if rng.random() < 0.5:                                    # Helligkeit / Gegenlicht
        k = rng.uniform(0.35, 1.25) if rng.random() < 0.7 else rng.uniform(0.2, 0.45)
        img = np.clip(np.asarray(img, dtype=np.float32) * k, 0, 255).astype(np.uint8)
        tags.append("dunkel" if k < 0.6 else "hell")
    if rng.random() < 0.4:                                    # Entsaettigung / verblichen
        a = np.asarray(img, dtype=np.float32)
        g = a.mean(axis=2, keepdims=True)
        f = rng.uniform(0.15, 0.7)
        img = np.clip(g + (a - g) * f, 0, 255).astype(np.uint8)
        tags.append("verblichen")
    if rng.random() < 0.35:                                   # Bewegungsunschaere
        r = rng.randrange(2, 7)
        img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(r))).astype(np.uint8)
        tags.append(f"unschaerfe{r}")
    if rng.random() < 0.3:                                    # Blendung
        a = np.asarray(img, dtype=np.float32)
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        rad = rng.uniform(0.15, 0.45) * max(h, w)
        blob = np.clip(1 - np.hypot(xx - cx, yy - cy) / rad, 0, 1) ** 2
        img = np.clip(a + blob[..., None] * 165.0, 0, 255).astype(np.uint8)
        tags.append("blendung")
    if rng.random() < 0.4:                                    # Rauschen (Nachtsensor)
        a = np.asarray(img, dtype=np.float32)
        n = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12 * strength, a.shape)
        img = np.clip(a + n, 0, 255).astype(np.uint8)
        tags.append("rauschen")
    return img, tags


def add_occluder(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, list[str]]:
    """Verdecker (Ast, Pfosten, LKW) ueber einen Teil des Bildes."""
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    col = rng.choice([(45, 40, 35), (70, 65, 60), (30, 45, 30), (90, 90, 95)])
    x0, y0 = rng.randrange(w), rng.randrange(h)
    bw, bh = rng.randrange(w // 12, w // 4), rng.randrange(h // 12, h // 3)
    a[y0:y0 + bh, x0:x0 + bw] = np.array(col, dtype=np.uint8)
    return a, ["verdeckt"]


# Stoerer, die KEINE Schilder sind, aber wie welche aussehen. Die gemessenen Fehlalarme
# des Netzes entstehen auf rot/rund/leuchtend - ohne solche Bilder kennt es sie nicht.
HARD_KINDS = ("rueckleuchten", "ampel", "bake", "werbung", "rueckseite", "flagge",
              "pfeiltafel", "kreise")


def hard_negative_sample(rng: random.Random, size: int, degrade_prob: float = 0.6
                         ) -> tuple[np.ndarray, list[str]]:
    """Ein Bild OHNE Schild, aber mit schildaehnlichem Stoerer (leeres Label).

    Beispiele: Rueckleuchten, Ampel, Baustellenbake, Leuchtreklame, graue Schildrueckseite,
    rote Flagge (Wimpel/Dreieck), gelbe Richtungstafel, farbige Kreise.
    """
    kind = rng.choice(HARD_KINDS)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def circle(cx: float, cy: float, r: float, col) -> None:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    def rect(x: float, y: float, w: float, h: float, col, rad: int = 0) -> None:
        box = [x, y, x + w, y + h]
        if rad:
            d.rounded_rectangle(box, radius=rad, fill=col)
        else:
            d.rectangle(box, fill=col)

    def place(w: int, h: int) -> tuple[int, int]:
        return rng.randrange(max(1, size - w)), rng.randrange(max(1, size - h))

    if kind == "rueckleuchten":                       # Autoheck: zwei rote Leuchten
        w, h = rng.randrange(size // 3, size // 2), rng.randrange(size // 5, size // 3)
        x, y = place(w, h)
        rect(x, y, w, h, (35, 35, 40), rad=8)
        r = max(2, h // 5)
        for i, cx in enumerate((x + w // 5, x + w - w // 5)):
            circle(cx, y + h // 3, r, (215, 25, 30))
            circle(cx, y + 2 * h // 3, r, (230, 120, 40) if i else (200, 30, 30))
    elif kind == "ampel":                             # Ampel: dunkler Kasten, drei Kreise
        w, h = size // 8, size // 3
        x, y = place(w, h)
        rect(x, y, w, h, (28, 30, 32), rad=6)
        lit = rng.randrange(3)
        for k, col in enumerate([(220, 40, 40), (235, 200, 40), (60, 200, 90)]):
            circle(x + w // 2, y + h * (2 * k + 1) // 6, max(2, w // 3),
                   col if k == lit else (58, 54, 54))
    elif kind == "bake":                              # Baustellenbake: orange/weiss gestreift
        w, h = size // 7, size // 3
        x, y = place(w, h)
        nb = 6
        for k in range(nb):
            spread = int(k * w / nb)
            col = (235, 120, 20) if k % 2 == 0 else (240, 240, 235)
            d.rectangle([x - spread, y + k * h // nb, x + w + spread, y + (k + 1) * h // nb], fill=col)
    elif kind == "werbung":                           # Leuchtreklame: bunte Flaechen
        for _ in range(rng.randrange(2, 5)):
            w, h = rng.randrange(size // 6, size // 3), rng.randrange(size // 10, size // 5)
            x, y = place(w, h)
            rect(x, y, w, h, rng.choice([(235, 40, 60), (250, 200, 30), (40, 120, 230),
                                         (250, 250, 245)]), rad=4)
    elif kind == "rueckseite":                        # Rueckseite eines Schildes: graues Achteck
        w, h = rng.randrange(size // 5, size // 3), rng.randrange(size // 5, size // 3)
        x, y = place(w, h)
        d.polygon(_poly_regular(x + w / 2, y + h / 2, w / 2, 8, math.pi / 8), fill=(150, 150, 152))
        if rng.random() < 0.5:
            circle(x + w // 2, y + h // 2, max(2, w // 8), (120, 120, 122))
    elif kind == "flagge":                            # rote Flagge / Wimpel (Dreieck!)
        w, h = rng.randrange(size // 4, size // 2), rng.randrange(size // 4, size // 2)
        x, y = place(w, h)
        col = rng.choice([(200, 25, 35), (215, 60, 30), (30, 90, 175)])
        if rng.random() < 0.5:
            d.polygon([(x, y), (x + w, y + h // 2), (x, y + h)], fill=col)
        else:
            d.polygon([(x, y), (x + w, y), (x + w // 2, y + h)], fill=col)
        d.line([(x, y), (x, y + h)], fill=(70, 70, 70), width=max(2, size // 160))
    elif kind == "pfeiltafel":                        # gelbe Tafel mit schwarzem Pfeil
        w = rng.randrange(size // 2, int(size * 0.8))
        h = rng.randrange(max(8, size // 12), max(12, size // 5))
        x, y = place(w, h)
        rect(x, y, w, h, (240, 195, 10), rad=3)
        d.line([(x + w // 8, y + h // 2), (x + w - w // 8, y + h // 2)],
               fill=(30, 30, 30), width=max(3, h // 6))
        d.polygon([(x + w - w // 8, y + h // 2 - h // 4), (x + w - w // 12, y + h // 2),
                   (x + w - w // 8, y + h // 2 + h // 4)], fill=(30, 30, 30))
    else:                                             # Kreise in Schildfarben (Baelle, Lichter)
        for _ in range(rng.randrange(2, 6)):
            r = max(3, rng.randrange(max(4, size // 14), max(6, size // 6)))
            x, y = place(2 * r, 2 * r)
            circle(x + r, y + r, r, rng.choice([(205, 35, 40), (30, 95, 185),
                                                (240, 200, 30), (235, 235, 230)]))

    # Ohne weiche Kanten waere der Stoerer zu leicht von echten Schildern zu trennen.
    if rng.random() < 0.45:
        layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.2)))
    canvas = Image.fromarray(random_background(rng, size)).convert("RGBA")
    canvas.alpha_composite(layer)
    arr = np.asarray(canvas.convert("RGB")).copy()
    tags = ["negativ", "hart:" + kind]
    if rng.random() < degrade_prob:
        arr, t2 = degrade(arr, rng)
        tags += t2
    return arr, sorted(set(tags))


def compose_negative(rng: random.Random, size: int = 320, hard_prob: float = 0.45,
                     degrade_prob: float = 0.6, occlude_prob: float = 0.15):
    """Ein Trainingsbild OHNE Schild: leere Boxliste, nur Bedingungen.

    Gleiche Rueckgabe wie compose_sample (Bild, Boxen, Tags), damit die Datensatz-Werkzeuge
    beide Faelle gleich behandeln koennen (siehe tools/synth_negatives.py).
    """
    if rng.random() < hard_prob:
        arr, tags = hard_negative_sample(rng, size, degrade_prob)
        return arr, [], tags
    arr = random_background(rng, size)
    tags = ["negativ", "leer"]
    if rng.random() < degrade_prob:
        arr, t2 = degrade(arr, rng)
        tags += t2
    if rng.random() < occlude_prob:
        arr, t2 = add_occluder(arr, rng)
        tags += t2
    return arr, [], sorted(set(tags))


def compose_sample(rng: random.Random, size: int = 320, n_signs: int | None = None,
                   degrade_prob: float = 0.75, occlude_prob: float = 0.25,
                   labels: list[str] | None = None):
    """Ein Trainingsbild bauen: Schild(er) auf Hintergrund + Verschlechterungen.

    labels beschraenkt die Auswahl (z.B. auf Typen, die im echten Datensatz fehlen - siehe
    tools/synth_missing.py); ohne Angabe wird aus allen neun Typen gezogen.
    """
    pool = labels or SIGN_LABELS
    n = n_signs if n_signs is not None else rng.choice([1, 1, 1, 2, 3])
    names = rng.sample(pool, min(n, len(pool)))
    img = Image.fromarray(random_background(rng, size))
    boxes, tags = [], []
    for name in names:
        tile = sign_patch(name)
        scale = rng.uniform(0.14, 0.75) if rng.random() < 0.75 else rng.uniform(0.07, 0.16)
        side = max(12, int(size * scale))
        t = tile.resize((side, side), Image.LANCZOS)
        angle = roll_angle(rng, name)
        t = t.rotate(angle, resample=Image.BICUBIC, expand=True)
        if abs(angle) >= 20:
            tags.append("roll")          # Roll-Verdrehung: Bruchstelle der Heuristik
        fg = np.asarray(t)[..., 3] > 40
        if not fg.any():
            continue
        ys, xs = np.nonzero(fg)
        bh, bw = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        px, py = rng.randrange(max(1, size - bw)), rng.randrange(max(1, size - bh))
        img.paste(t, (px, py), t)
        boxes.append({"label": name, "cx": (px + xs.min() + bw / 2) / size,
                      "cy": (py + ys.min() + bh / 2) / size, "w": bw / size, "h": bh / size})
    arr = np.asarray(img)[..., :3].copy()
    if rng.random() < degrade_prob:
        arr, t2 = degrade(arr, rng)
        tags += t2
    if rng.random() < occlude_prob:
        arr, t2 = add_occluder(arr, rng)
        tags += t2
    return arr, boxes, sorted(set(tags))


def write_dataset(out_dir: str | Path, n: int, size: int = 320, seed: int = 0) -> dict:
    """Datensatz schreiben: images/*.jpg + labels/*.txt (YOLO) + manifest.json."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries = []
    for i in range(n):
        arr, boxes, tags = compose_sample(rng, size)
        stem = f"synth_{i:05d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=88)
        lines = [f"{CLASS_ID[b['label']]} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes]
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries.append({"id": stem, "src": "synthetisch", "license": "eigene Quelle (generiert)",
                        "scene": "synth", "conditions": tags, "split": "train" if i % 10 else "val",
                        "width": size, "height": size, "boxes": len(boxes)})
    manifest = {"format": "yolo-txt", "classes": SIGN_LABELS, "source": "tools/synth_data.py",
                "seed": seed, "size": size, "images": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="synthetische Schilderdaten erzeugen")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = write_dataset(args.out, args.n, args.size, args.seed)
    tags: dict[str, int] = {}
    for e in m["images"]:
        for t in e["conditions"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"{len(m['images'])} Bilder in {args.out} geschrieben.")
    print("Bedingungen:", ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))


if __name__ == "__main__":
    main()
