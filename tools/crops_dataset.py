"""
tools/crops_dataset.py - Detektor-Datensatz aus den grossen Katalogen bauen.

Warum ein eigener Bauer neben tools/gtsrb_dataset.py: GTSRB liefert nur 43 Klassen als
Ausschnitte. GTSIGN-220 (75 541 Crops, 220 StVO-Klassen) und Synset Signset Germany
(211 Klassen) liefern dieselbe Art Daten, aber in anderen Verpackungen - und erst zusammen
kommt die Klassenliste aus tools/signmap.py zusammen. Der Kern der Arbeit ist ueberall
gleich: echten Ausschnitt auf einen Hintergrund kleben, Box mitschreiben.

Upsampling je Quelle (--weight) ist hier ausdruecklich ein eigener Hebel: Gerechnet wird
NICHT nach Bildzahl, sondern die QUELLE wird gezogen. Damit bestimmt der Gewichtsfaktor
direkt den Anteil im Training - ein kleiner, hochwertiger Katalog wird also nicht von einem
grossen erdrueckt. Ohne Angabe zieht jede Quelle gleich haeufig.

Aufruf (Beispiel, wie es kaggle/train_kernel.py aufruft):

    python tools/crops_dataset.py --out data/det --n 30000 --split train ^
        --gtsrb data/gtsrb/GTSRB_Final_Training_Images.zip ^
        --gtsign data/gtsign/GTSIGN-220.zip ^
        --synset data/synset-crops ^
        --weight gtsign=2 --weight synset=2 --balance
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import signmap
import synth_data as sd
from gtsrb_dataset import read_rows as gtsrb_rows

LABELS = signmap.LABELS


# ---------------------------------------------------------------------------
# Quellen: jede liefert (Schluessel, Label) - der Schluessel holt spaeter den Ausschnitt.
# Der Unterschied zwischen den Quellen steckt nur im Lesen, nicht im Komponieren.
# ---------------------------------------------------------------------------
@dataclass
class Quelle:
    name: str
    schluessel: list = field(default_factory=list)   # Handhabe (Zip-Eintrag, Bildpfad, ...)
    labels: list[str] = field(default_factory=list)
    gewicht: float = 1.0
    lizenz: str = ""
    haltung: object = None                           # offenes ZipFile
    verworfen: int = 0                               # Eintraege ohne Label in unserer Liste

    def __len__(self) -> int:
        return len(self.schluessel)

    def hole(self, rng: random.Random) -> tuple[Image.Image, str]:
        """Einen zufaelligen Ausschnitt liefern (RGB, Label)."""
        i = rng.randrange(len(self.schluessel))
        return self.bild(i), self.labels[i]

    def bild(self, i: int) -> Image.Image:            # in den Unterklassen gefuellt
        raise NotImplementedError

    def zaehlung(self) -> dict[str, int]:
        z: dict[str, int] = {}
        for name in self.labels:
            z[name] = z.get(name, 0) + 1
        return z


def _stv_tabelle(csv_pfad: Path) -> dict[str, str]:
    """GTSIGN-Ordner (Class_ID als dreistellige Zahl) -> Label."""
    if not csv_pfad.exists():
        raise SystemExit(f"StVO-Tabelle fehlt: {csv_pfad} "
                         f"(GTSIGN-220: class_descriptions_and_stvo.csv)")
    tab: dict[str, str] = {}
    for r in csv.DictReader(csv_pfad.open(encoding="utf-8")):
        label = signmap.label_for_stvo(r["StVO_Sign_Number"])
        if label:
            tab[f"{int(r['Class_ID']):03d}"] = label
    return tab


class GtsignQuelle(Quelle):
    """GTSIGN-220: <Wurzel>/<Class_ID dreistellig>/<name>.jpg, Label aus der StVO-Nummer."""

    def __init__(self, zip_pfad: Path, csv_pfad: Path, gewicht: float = 1.0):
        super().__init__(name="GTSIGN-220", gewicht=gewicht, lizenz="CC BY-SA 4.0",
                         haltung=zipfile.ZipFile(zip_pfad))
        tab = _stv_tabelle(csv_pfad)
        for eintrag in self.haltung.namelist():
            if eintrag.endswith("/") or not eintrag.lower().endswith(".jpg"):
                continue
            teile = eintrag.split("/")
            if len(teile) < 3:
                continue
            label = tab.get(teile[-2])
            if not label:
                self.verworfen += 1
                continue
            self.schluessel.append(eintrag)
            self.labels.append(label)

    def bild(self, i: int) -> Image.Image:
        with self.haltung.open(self.schluessel[i]) as f:
            return Image.open(f).convert("RGB")


class GtsrbQuelle(Quelle):
    """GTSRB: ROI aus der GT-CSV, Label aus der ClassId."""

    def __init__(self, zip_pfad: Path, gt_zip: Path | None = None, gewicht: float = 1.0):
        super().__init__(name="GTSRB", gewicht=gewicht, lizenz="frei fuer Forschung",
                         haltung=zipfile.ZipFile(zip_pfad))
        for r in gtsrb_rows(self.haltung, zipfile.ZipFile(gt_zip) if gt_zip else None,
                            labeler=signmap.label_for_gtsrb, skip=frozenset()):
            label = r["label"]
            if not label:
                self.verworfen += 1
                continue
            self.schluessel.append((r["path"], (r["x1"], r["y1"], r["x2"], r["y2"])))
            self.labels.append(label)

    def bild(self, i: int) -> Image.Image:
        pfad, roi = self.schluessel[i]
        with self.haltung.open(pfad) as f:
            img = Image.open(f).convert("RGB")
        return img.crop(roi)


class OrdnerQuelle(Quelle):
    """Ordner mit Unterordnern je Label (so werden Synset-Ausschnitte zwischengelagert).

    Struktur: <Wurzel>/<label>/<datei>.jpg. Die Ordnernamen sind die aus tools/signmap.py,
    damit keine zweite Namensliste entsteht; unbekannte Ordner werden gemeldet, nicht still
    verschluckt.
    """

    def __init__(self, wurzel: Path, gewicht: float = 1.0, lizenz: str = "",
                 name: str | None = None):
        super().__init__(name=name or wurzel.name, gewicht=gewicht, lizenz=lizenz)
        self.unbekannt: list[str] = []
        for unter in sorted(p for p in wurzel.iterdir() if p.is_dir()):
            if unter.name not in signmap.CLASS_ID:
                self.unbekannt.append(unter.name)
                continue
            for bild in sorted(unter.iterdir()):
                if bild.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.schluessel.append(bild)
                    self.labels.append(unter.name)

    def bild(self, i: int) -> Image.Image:
        return Image.open(self.schluessel[i]).convert("RGB")


class SzenenQuelle:
    """Echte Fotos als Hintergrund (z.B. Open Images).

    Zwei Aufgaben: (1) der Komposition eine echte Umgebung geben statt nur synthetischer
    Flaechen, (2) Negative liefern - echte Szenen ohne deutsches Schild sind die Gegenprobe,
    die im Betrieb zaehlt.
    """

    def __init__(self, wurzel: Path, name: str = "Szenen", lizenz: str = ""):
        self.name, self.lizenz = name, lizenz
        self.pfade = sorted(p for p in wurzel.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.pfade)

    def hintergrund(self, rng: random.Random, size: int) -> np.ndarray | None:
        """Ein Foto mittig auf `size` zuschneiden (Laplace-Fehler werden gemeldet)."""
        if not self.pfade:
            return None
        i = rng.randrange(len(self.pfade))
        if i not in self._cache:
            with Image.open(self.pfade[i]) as f:
                img = f.convert("RGB")
                w, h = img.size
                kante = min(w, h)                       # quadratisch beschneiden statt verzerren
                img = img.crop(((w - kante) // 2, (h - kante) // 2,
                                (w + kante) // 2, (h + kante) // 2))
                self._cache[i] = np.asarray(img.resize((size, size), Image.BILINEAR),
                                            dtype=np.uint8)
            if len(self._cache) > 64:                   # Speicher begrenzen
                self._cache.pop(next(iter(self._cache)))
        return self._cache[i].copy()


def _item_gewichte(quelle: Quelle) -> list[float]:
    """Klassenausgleich innerhalb einer Quelle: seltene Label haeufiger ziehen.

    Nur eine Gewichtung, keine Vervielfachung: der Datensatz waechst nicht, die Auswahl
    verschiebt sich. Genau das ist der Sinn (siehe --balance in tools/gtsrb_dataset.py).
    """
    zaehl = quelle.zaehlung()
    return [1.0 / zaehl[name] for name in quelle.labels]


def compose(quellen: list[Quelle], rng: random.Random, size: int,
            szenen: SzenenQuelle | None, degrade_prob: float, occlude_prob: float,
            ausgleich: bool, szenen_prob: float) -> tuple:
    """Ein Trainingsbild mit echten Schildausschnitten bauen."""
    arr = None
    if szenen is not None and rng.random() < szenen_prob:
        arr = szenen.hintergrund(rng, size)            # echtes Foto als Umgebung
    if arr is None:
        arr = sd.random_background(rng, size)          # synthetische Flaeche wie bisher
    img = Image.fromarray(arr)
    n = rng.choice([1, 1, 1, 2, 3])
    boxes, tags, herkunft = [], [], []
    for _ in range(n):
        quelle = rng.choices(quellen, weights=[q.gewicht for q in quellen])[0]
        if ausgleich:
            i = rng.choices(range(len(quelle)), weights=_item_gewichte(quelle))[0]
            crop, label = quelle.bild(i), quelle.labels[i]
        else:
            crop, label = quelle.hole(rng)
        skala = rng.uniform(0.14, 0.75) if rng.random() < 0.75 else rng.uniform(0.06, 0.16)
        box = sd.place_crop(img, crop, rng, int(max(12, size * skala)), tags=tags)
        if box:
            box["label"] = label
            boxes.append(box)
            herkunft.append(quelle.name)
    arr = np.asarray(img)[..., :3].copy()
    if rng.random() < degrade_prob:
        arr, t2 = sd.degrade(arr, rng)
        tags += t2
    if rng.random() < occlude_prob:
        arr, t2 = sd.add_occluder(arr, rng)
        tags += t2
    return arr, boxes, sorted(set(tags)), herkunft


def zahlen(quellen: list[Quelle], szenen: SzenenQuelle | None) -> None:
    """Vor dem Bauen zeigen, was jede Quelle beisteuert - sonst raet man am Gewicht."""
    print("[quellen] Gewicht  Bilder  Klassen  verworfen  Name")
    for q in quellen:
        print(f"          {q.gewicht:7.1f}  {len(q):6d}  {len(q.zaehlung()):7d}  "
              f"{q.verworfen:9d}  {q.name}")
        if getattr(q, "unbekannt", None):
            print(f"          [hinweis] unbekannte Labelordner in {q.name}: "
                  f"{', '.join(q.unbekannt[:8])}")
    if szenen is not None:
        print(f"          {'-':>7}  {len(szenen):6d}  {'-':>7}  {'-':>9}  "
              f"{szenen.name} (Hintergrund/Negative)")


def build(out_dir: str | Path, n: int, size: int, split: str, seed: int,
          quellen: list[Quelle], szenen: SzenenQuelle | None = None,
          neg_share: float = 0.0, balance: bool = False, degrade_prob: float = 0.85,
          occlude_prob: float = 0.3, scene_prob: float = 0.5) -> dict:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries: list[dict] = []
    tag_zaehler: dict[str, int] = {}
    label_zaehler: dict[str, int] = {}
    quelle_zaehler: dict[str, int] = {}
    for i in range(n):
        negativ = (szenen is not None and rng.random() < neg_share) or not quellen
        if negativ:
            # Echtes Foto OHNE Schild: leere Boxliste, aber dieselben Bedingungen wie sonst
            arr = szenen.hintergrund(rng, size) if szenen is not None else None
            if arr is None:
                continue
            boxes, tags, herkunft = [], ["negativ", "echteSzene"], [szenen.name]
            if rng.random() < degrade_prob:
                arr, t2 = sd.degrade(arr, rng)
                tags += t2
        else:
            arr, boxes, tags, herkunft = compose(quellen, rng, size, szenen, degrade_prob,
                                                 occlude_prob, balance, scene_prob)
        stem = f"{split}_{i:06d}"
        Image.fromarray(arr).save(out / "images" / f"{stem}.jpg", quality=86)
        (out / "labels" / f"{stem}.txt").write_text("\n".join(
            f"{signmap.CLASS_ID[b['label']]} {b['cx']:.6f} {b['cy']:.6f} "
            f"{b['w']:.6f} {b['h']:.6f}" for b in boxes) + "\n", encoding="utf-8")
        for t in tags:
            tag_zaehler[t] = tag_zaehler.get(t, 0) + 1
        for b in boxes:
            label_zaehler[b["label"]] = label_zaehler.get(b["label"], 0) + 1
        for h in set(herkunft):
            quelle_zaehler[h] = quelle_zaehler.get(h, 0) + 1
        entries.append({"id": stem, "src": "/".join(sorted(set(herkunft))),
                        "license": "siehe docs/DATENSAETZE.md", "scene": "katalog",
                        "conditions": tags, "split": split, "width": size, "height": size,
                        "boxes": len(boxes), "balanced": bool(balance)})

    manifest_path = out / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))["images"] \
        if manifest_path.exists() else []
    merged = [e for e in old if e.get("split") != split] + entries
    manifest_path.write_text(json.dumps(
        {"format": "yolo-txt", "classes": LABELS, "source": "tools/crops_dataset.py",
         "images": merged}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"neu": len(entries), "bedingungen": tag_zaehler, "label": label_zaehler,
            "herkunft": quelle_zaehler, "bilder_gesamt": len(merged)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Detektor-Datensatz aus GTSRB/GTSIGN/Synset bauen")
    ap.add_argument("--out", default="data/det")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade", type=float, default=0.85)
    ap.add_argument("--occlude", type=float, default=0.3)
    ap.add_argument("--scene-prob", type=float, default=0.5,
                    help="Anteil Bilder, deren Hintergrund ein ECHTES Foto ist (--scenes)")
    ap.add_argument("--gtsrb", default="", help="GTSRB-Bildarchiv (.zip)")
    ap.add_argument("--gtsrb-gt", default="", help="eigenes Label-Archiv (Test-Set)")
    ap.add_argument("--gtsign", default="", help="GTSIGN-220.zip")
    ap.add_argument("--gtsign-csv", default="data/gtsign/class_descriptions_and_stvo.csv")
    ap.add_argument("--synset", default="", help="Ordner mit Synset-Ausschnitten (Labelordner)")
    ap.add_argument("--crops", action="append", default=[],
                    help="weitere Quelle im Format NAME=ORDNER (mehrfach moeglich)")
    ap.add_argument("--scenes", default="", help="Ordner mit echten Fotos (Hintergrund/Negative)")
    ap.add_argument("--neg-share", type=float, default=0.0,
                    help="Anteil Bilder ohne Schild aus --scenes (0 = keine)")
    ap.add_argument("--weight", action="append", default=[],
                    help="Upsampling je Quelle, z.B. --weight synset=3 (mehrfach moeglich)")
    ap.add_argument("--balance", action="store_true",
                    help="seltene Klassen innerhalb jeder Quelle haeufiger ziehen")
    ap.add_argument("--nur-zaehlen", action="store_true",
                    help="Quellen auflisten und beenden (kein Bild schreiben)")
    args = ap.parse_args()

    gewichte: dict[str, float] = {}
    for w in args.weight:
        if "=" not in w:
            raise SystemExit(f"--weight braucht die Form name=zahl, bekam: {w}")
        k, v = w.split("=", 1)
        gewichte[k.strip().lower()] = float(v)

    quellen: list[Quelle] = []
    if args.gtsrb:
        quellen.append(GtsrbQuelle(Path(args.gtsrb),
                                   Path(args.gtsrb_gt) if args.gtsrb_gt else None,
                                   gewicht=gewichte.get("gtsrb", 1.0)))
    if args.gtsign:
        quellen.append(GtsignQuelle(Path(args.gtsign), Path(args.gtsign_csv),
                                    gewicht=gewichte.get("gtsign", 1.0)))
    if args.synset:
        quellen.append(OrdnerQuelle(Path(args.synset), gewicht=gewichte.get("synset", 1.0),
                                    lizenz="CC BY 4.0", name="Synset-Signset"))
    for eintrag in args.crops:
        if "=" not in eintrag:
            raise SystemExit(f"--crops braucht die Form NAME=ORDNER, bekam: {eintrag}")
        name, ordner = eintrag.split("=", 1)
        quellen.append(OrdnerQuelle(Path(ordner), gewicht=gewichte.get(name.lower(), 1.0),
                                    name=name))
    if not quellen:
        # Reine Negativ-Splits brauchen keinen Katalog: --scenes mit --neg-share 1.0
        # baut Bilder ohne Schild (die Gegenprobe auf echten Fotos).
        if not (args.scenes and args.neg_share >= 1.0):
            raise SystemExit("keine Quelle angegeben (--gtsrb / --gtsign / --synset / --crops); "
                             "nur ein reiner Negativ-Split darf ohne Quelle laufen "
                             "(--scenes mit --neg-share 1.0)")

    szenen = SzenenQuelle(Path(args.scenes)) if args.scenes else None
    # Eine Quelle ohne ein einziges brauchbares Bild darf den Lauf nicht sprengen: compose()
    # zieht jede Quelle mit ihrem Gewicht, und rng.randrange(0) wirft dann ValueError. Ursache
    # ist meist ein falsches Archiv - die GTSRB-Test-LABELS enthalten zum Beispiel gar keine
    # Bilder und stehen trotzdem als *.zip bereit.
    leer = [q for q in quellen if not len(q)]
    if leer:
        for q in leer:
            print(f"[warnung] Quelle ohne Bilder, wird ausgelassen: {q.name} "
                  f"({q.verworfen} Eintraege ohne passendes Label)")
        quellen = [q for q in quellen if len(q)]
        if not quellen and not (args.scenes and args.neg_share >= 1.0):
            raise SystemExit("alle uebergebenen Quellen sind leer - siehe Warnungen oben")
    zahlen(quellen, szenen)
    if args.nur_zaehlen:
        return
    info = build(args.out, args.n, args.size, args.split, args.seed, quellen, szenen,
                 args.neg_share, args.balance, args.degrade, args.occlude, args.scene_prob)
    print(f"[daten] {info['neu']} Bilder ({args.split}) nach {args.out}, "
          f"{info['bilder_gesamt']} im Manifest")
    print("[daten] Herkunft:", ", ".join(f"{k}={v}" for k, v in sorted(info["herkunft"].items())))
    print("[daten] Bedingungen:", ", ".join(f"{k}={v}" for k, v in sorted(info["bedingungen"].items())))
    leer = [l for l in LABELS if l not in info["label"]]
    print(f"[daten] Klassen mit Beispielen: {len(info['label'])}/{len(LABELS)}")
    if leer:
        print(f"[warnung] ohne Beispiel in diesem Split ({len(leer)}): {', '.join(leer)}")


if __name__ == "__main__":
    main()