"""
tools/teacher.py - Lehrer-Modell (Distillation) fuer den Schilder-Detektor.

WARUM EIN LEHRER? Gemessen (v6-Lauf, 03.10.): der Detektor hat die BOXEN gelernt
(Objektivitaet 253 -> 0,2, Box 18,7 -> 0,4) und die ART nicht (Klassifikationsverlust blieb
bei 3,4 - Zufall bei 74 Klassen waere ln 74 = 4,3). Genau das passt zur Fehlerzerlegung:
195 von 226 Fehlalarmen lagen auf ECHTEN Schildern, also auf einer richtigen Box mit
falscher Klasse. Ein Lehrer, der die ART besser weiss als unser Netz, ist deshalb der
wirksamste einzelne Hebel - nicht eine groessere Schuelerarchitektur.

WELCHER LEHRER: `vit_gtsign_all_classes` aus dem GTSIGN-220-Datensatz (Carnot u. a., IV 2026).
    * google/vit-base-patch16-224, 12 Layer, 768 breit, patch 16, 86 Mio. Parameter
    * auf 220 DEUTSCHEN StVO-Klassen trainiert (dieselben Zeichen, nur feiner geteilt)
    * gemessene Guete des veroeffentlichten Modells: Accuracy 0,973, P 0,911, R 0,930
    * Download: 344 MB (model.safetensors)
Das ist inhaltlich der beste verfuegbare Lehrer fuer DIESE Aufgabe: ein auf COCO
trainierter Detektor kennt nur "stop sign" (eine Klasse), dieser kennt 220 StVO-Zeichen.
Aus demselben Grund ist er auch der einzige, dessen Klassen sich verlustfrei auf unsere 74
abbilden lassen - ueber die StVO-Nummer, die beide Seiten fuehren (tools/signmap.py).

LIZENZ (wichtig, weil ShareAlike): das Modell stammt aus einem Datensatz, der unter
CC BY-SA 4.0 steht (Ableitung von Mapillary-Bildern). Wer das hier erzeugte Schuelermodell
veroeffentlicht, muss diese Herkunft nennen und die ShareAlike-Bedingung beachten. Steht
auch in docs/TRAINING.md.

WAS HIER ENTSTEHT: keine neuen Gewichte, sondern ein CACHE der Lehrer-Verteilungen. Fuer
jede Grundwahrheitsbox wird der Ausschnitt dem Lehrer vorgelegt; seine 220
Klassenwahrscheinlichkeiten werden auf unsere 74 MARGINALISIERT (aufsummiert, nicht
Maximum - mehrere StVO-Nummern koennen auf denselben unserer Typen fallen, und die
Wahrscheinlichkeit dieses Typs ist die Summe seiner Feinklassen).

Aufruf:
    python tools/teacher.py --mapping                       # Zuordnung pruefen (ohne Download)
    python tools/teacher.py --data data/det --out data/teacher --split train val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "tools"))

import signmap  # noqa: E402

# Der Lehrer und seine Herkunft - an EINER Stelle, damit Lizenz und Name nicht driften.
LEHRER_REPO = "miriamcarnot/GTSIGN-220"
LEHRER_PFAD = "trained_models/vit_gtsign_all_classes"
LEHRER_URL = (f"https://huggingface.co/datasets/{LEHRER_REPO}/resolve/main/"
              f"{LEHRER_PFAD}")
LEHRER_NAME = "vit-base-patch16-224 / GTSIGN-220 all classes"
# Das Basismodell, von dem feinjustiert wurde. Es liefert die Bildvorverarbeitung, weil das
# Lehrer-Repo keine eigene mitliefert (gemessen: kein preprocessor_config.json).
LEHRER_BASIS = "google/vit-base-patch16-224"
LEHRER_LIZENZ = "CC BY-SA 4.0 (GTSIGN-220, Carnot u. a.; Ableitung von Mapillary CC BY-SA 4.0)"
LEHRER_GUETE = {"accuracy": 0.9727, "precision": 0.9112, "recall": 0.9304}
LEHRER_STVO_CSV = "class_descriptions_and_stvo.csv"
N_LEHRER = 220          # Klassen des Lehrers
TEMPERATUR = 2.0        # Standard fuer Wissens-Distillation (Hinton u. a.)

# ---------------------------------------------------------------------------
# Zweiter Lehrer (optional): GTSRB-ViT. Wird mit dem ersten GEMITTELT (je 0,5).
#
# Warum ueberhaupt ein zweiter - und warum der erste trotzdem der Hauptlehrer bleibt:
#
#                                | Klassen | deckt von unseren 74 | Guete (veroeffentlicht) | Lizenz
#   vit_gtsign_all_classes       |  220    |        68            | Acc 0,973 / P 0,911     | CC BY-SA 4.0
#   vit-traffic-sign-GTSRB       |   43    |        36            | Acc 0,985 / F1 0,985    | MIT
#
# Auf den ersten Blick ist der GTSRB-ViT der bessere Lehrer: genauere veroeffentlichte Werte,
# freie Lizenz, und seine Trainingsbilder sind echte Fotos (GTSRB), nicht nur Mapillary. Der
# entscheidende Punkt ist aber die ABDECKUNG: er kennt nur die 43 GTSRB-Zeichen, also 36
# unserer 74 Typen - alle uebrigen Tempostufen (5, 10, 40, 90, 110, 130), Zonen, Andreaskreuz,
# Rad-/Gehwege, Einbahnstrasse, Parken, Autobahn ... kennt er gar nicht. Der GTSIGN-220-Lehrer
# deckt 68 von 74 ab, weil er nach StVO-Nummer feiner geteilt ist.
#
# Deshalb: GTSIGN-220 bleibt der Hauptlehrer. Der GTSRB-ViT kommt als ZWEITER dazu und wird
# gleichgewichtet gemittelt. Auf den 36 gemeinsamen Klassen entscheidet damit der genauere
# mit, und fuer die uebrigen 32 gilt allein der Hauptlehrer - die Mittelung verschlechtert
# dort nichts, weil ein Lehrer, der ein Zeichen nicht kennt, keine Masse beitraegt.
# Einschalten: tools/teacher.py --zweiter (bzw. LEHRER["zweiter"] im Kaggle-Rezept).
# ---------------------------------------------------------------------------
ZWEITLEHRER_REPO = "kelvinandreas/vit-traffic-sign-GTSRB"
ZWEITLEHRER_NAME = "vit-base-patch16-224-in21k / GTSRB (43 Klassen)"
ZWEITLEHRER_LIZENZ = "MIT (Kelvin Andreas; feinjustiert von google/vit-base-patch16-224-in21k)"
ZWEITLEHRER_GUETE = {"accuracy": 0.9846, "precision": 0.9853, "recall": 0.9846, "f1": 0.9846}
N_ZWEITLEHRER = 43
ZWEITLEHRER_GEWICHT = 0.5       # Gewicht des zweiten Lehrers in der Mittelung


def lehrer_mapping(csv_pfad: Path) -> np.ndarray:
    """Lehrer-Klasse (0..219) -> unsere Klasse (0..73) oder -1.

    Der Lehrer teilt feiner: seine Klasse 83 ist "Z 274-70", unsere "tempo70". Beide Seiten
    fuehren dieselbe StVO-Nummer, deshalb ist die Zuordnung keine Vermutung, sondern ein
    Nachschlagen. Klassen ohne Treffer (z. B. Zusatzzeichen, die wir nicht fuehren) fallen
    heraus - sie duerfen nicht auf einen Sammeltopf, sonst lehrt der Lehrer Falsches.
    """
    import csv

    tab = np.full(N_LEHRER, -1, dtype=np.int64)
    if not csv_pfad.exists():
        raise SystemExit(f"StVO-Tabelle fehlt: {csv_pfad}")
    for r in csv.DictReader(csv_pfad.open(encoding="utf-8")):
        lehrer_id = int(r["Class_ID"])
        if not 0 <= lehrer_id < N_LEHRER:
            continue
        unser = signmap.label_for_stvo(r["StVO_Sign_Number"])
        if unser:
            tab[lehrer_id] = signmap.CLASS_ID[unser]
    return tab


def zweitlehrer_mapping() -> np.ndarray:
    """GTSRB-Klasse (0..42) -> unsere Klasse (0..73) oder -1.

    Die Zuordnung steht schon in tools/signmap.py (GTSRB_MAP, Quelle 1) - dort direkt auf
    unsere NAMEN, nicht ueber die StVO-Nummer. Das ist Absicht: die GTSRB-Namen sind eindeutig
    englisch, waehrend eine StVO-Zahl hier geraten waere (Z 282 "Ende aller Streckenverbote"
    gegen Z 280 "Ende des Ueberholverbots" ist genau so ein Fall).
    """
    tab = np.full(N_ZWEITLEHRER, -1, dtype=np.int64)
    for i in range(N_ZWEITLEHRER):
        name = signmap.label_for_gtsrb(i)
        if name:
            tab[i] = signmap.CLASS_ID[name]
    return tab


def zweitlehrer_laden(geraet: str):
    """Zweiten Lehrer samt Vorverarbeitung laden.

    Anders als beim Hauptlehrer liegt dieser in einem normalen MODELL-Repo - dort gibt es
    preprocessor_config.json, deshalb genuegt `from_pretrained(repo)` (gemessen: die
    Dateiliste des Repos enthaelt config.json, model.safetensors, preprocessor_config.json).
    """
    from transformers import ViTForImageClassification, ViTImageProcessor

    proc = ViTImageProcessor.from_pretrained(ZWEITLEHRER_REPO)
    modell = ViTForImageClassification.from_pretrained(ZWEITLEHRER_REPO).to(geraet).eval()
    return modell, proc


def zweitlehrer_klassen(map2: np.ndarray) -> set[int]:
    """Unsere Klassen, die der zweite Lehrer ueberhaupt kennt (Sonst keine Mittelung)."""
    return {int(i) for i in map2[map2 >= 0]}


def teacher_logits_fuer_boxen(bild_pfad: Path, boxen: list[list[float]], modell, processor,
                              mapping: np.ndarray, geraet, rand: float = 0.25,
                              img_size: int = 224) -> np.ndarray:
    """Lehrer-Verteilungen fuer die GT-Boxen EINES Bildes -> (n_boxen, 74) float32.

    Der Ausschnitt wird um `rand` vergroessert (ein Viertel der Boxgroesse je Seite): der
    Lehrer ist auf Schildausschnitten trainiert und braucht etwas Rand, sonst schneidet die
    Box die Ecken eines Achtkants oder die Spitze eines Dreiecks ab - genau die Formmerkmale,
    an denen seine Entscheidung haengt.
    """
    import torch
    from PIL import Image

    if not boxen:
        return np.zeros((0, signmap.N_LABELS), dtype=np.float32)
    with Image.open(bild_pfad) as f:
        bild = f.convert("RGB")
    w, h = bild.size
    ausschnitte = []
    for x0, y0, x1, y1 in boxen:
        mx, my = (x1 - x0) * rand, (y1 - y0) * rand
        a, b = int(max(0, x0 - mx)), int(max(0, y0 - my))
        c, d = int(min(w, x1 + mx)), int(min(h, y1 + my))
        if c - a < 2 or d - b < 2:
            ausschnitte.append(Image.new("RGB", (img_size, img_size), (114, 114, 114)))
            continue
        ausschnitte.append(bild.crop((a, b, c, d)))
    stappel = processor(images=ausschnitte, return_tensors="pt")["pixel_values"].to(geraet)
    with torch.no_grad():
        logits = modell(pixel_values=stappel).logits      # (n, 220)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
    # Marginalisieren: mehrere Lehrer-Klassen koennen auf denselben unserer Typen fallen
    # (z. B. Z 205 und Z 208 beide auf "vorfahrtGewaehren"). Die Wahrscheinlichkeit unseres
    # Typs ist die SUMME seiner Feinklassen - nicht das Maximum, sonst ginge Masse verloren
    # und die Verteilung summierte sich nicht mehr auf 1.
    aus = np.zeros((len(ausschnitte), signmap.N_LABELS), dtype=np.float32)
    for k in np.nonzero(mapping >= 0)[0]:
        aus[:, mapping[k]] += p[:, k]
    return aus


def stvo_csv(pfad: str = "") -> Path:
    """Pfad der StVO-Tabelle: uebergebener Wert, sonst der Standardplatz unter data/gtsign."""
    return Path(pfad) if pfad else ROOT / "data" / "gtsign" / LEHRER_STVO_CSV


def bild_prozessor():
    """Bildvorverarbeitung des Lehrers - mit Rueckfall, weil sie im Repo NICHT liegt.

    Gemessen (03.10.2026): unter `trained_models/vit_gtsign_all_classes/` liegen nur
    config.json, model.safetensors und Trainer-Zustand - **kein preprocessor_config.json**.
    `ViTImageProcessor.from_pretrained(<repo>)` scheitert deshalb mit OSError. Drei Stufen,
    jede dokumentiert:

      1. das Repo selbst (falls die Vorverarbeitung spaeter nachgereicht wird),
      2. `google/vit-base-patch16-224` - das Basismodell, von dem feinjustiert wurde; seine
         Vorverarbeitung ist damit nachweislich die richtige und nicht geraten,
      3. feste Werte (224 px, rescale 1/255, Mittel 0.5, Streuung 0.5, bikubisch) - die
         Standardwerte genau dieses Basismodells, als letzte Absicherung ohne Netz.
    """
    from transformers import ViTImageProcessor

    for quelle in (LEHRER_URL, LEHRER_BASIS):
        try:
            proc = ViTImageProcessor.from_pretrained(quelle)
            print(f"[lehrer] Bildvorverarbeitung von {quelle}", flush=True)
            return proc
        except Exception as fehler:
            print(f"[lehrer] Vorverarbeitung von {quelle} nicht ladbar ({type(fehler).__name__})",
                  flush=True)
    print("[lehrer] Vorverarbeitung fest gesetzt (224 px, /255, Mittel 0.5, Streuung 0.5)",
          flush=True)
    return ViTImageProcessor(size={"height": 224, "width": 224},
                             resample=3,       # PIL.Image.BICUBIC - Standard dieses ViT
                             rescale_factor=1 / 255.0,
                             image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])


def lehrer_laden(geraet: str):
    """Das Lehrer-Modell holen und laden.

    WARUM NICHT einfach `from_pretrained(URL)`: derselbe Fehler ist lokal passiert - das
    Modell liegt in einem **Dataset**-Repo, und `from_pretrained` erwartet ohne `repo_type`
    ein Modell-Repo. Genau diese Fehlermeldung kam: "Repo id must be in the form
    'repo_name' or 'namespace/repo_name'". `hf_hub_download(..., repo_type="dataset")` holt
    die zwei Dateien stattdessen in den HF-Zwischenspeicher; sie landen im selben Ordner und
    lassen sich von dort ganz normal laden.
    """
    from huggingface_hub import hf_hub_download
    from transformers import ViTForImageClassification

    pfade = [hf_hub_download(repo_id=LEHRER_REPO, filename=f"{LEHRER_PFAD}/{name}",
                             repo_type="dataset")
             for name in ("config.json", "model.safetensors")]
    print(f"[lehrer] Gewichte: {pfade[1]} "
          f"({Path(pfade[1]).stat().st_size / 1e6:.0f} MB)", flush=True)
    return ViTForImageClassification.from_pretrained(str(Path(pfade[0]).parent)).to(geraet).eval()


def cache_bauen(data: Path, out: Path, split: str, geraet: str = "auto",
                limit: int = 0, csv_pfad: str = "", zweiter: bool = False) -> dict:
    """Lehrer-Cache fuer einen Split bauen (einmalig, im Training nur noch laden).

    `zweiter=True` mittelt den GTSRB-ViT dazu - aber NUR fuer Boxen, deren Grundwahrheit er
    ueberhaupt kennt (36 unserer 74 Typen). Warum so eng: ein Lehrer, der ein Zeichen nicht
    kennt, antwortet trotzdem - er sagt dann z. B. bei "tempo40" (kennt er nicht) "tempo30".
    Ungefiltert gemittelt wuerde er damit Falsches in den Cache schreiben. Mit dem Filter
    kann er nur dort mitreden, wo er recht haben KANN.
    """
    import torch

    mapping = lehrer_mapping(stvo_csv(csv_pfad))
    if (mapping >= 0).sum() == 0:
        raise SystemExit("Zuordnung leer - StVO-Spalte in der Tabelle pruefen")
    geraet = ("cuda" if torch.cuda.is_available() else "cpu") if geraet == "auto" else geraet
    print(f"[lehrer] {LEHRER_NAME}")
    print(f"[lehrer] Lizenz: {LEHRER_LIZENZ}")
    print(f"[lehrer] Zuordnung: {(mapping >= 0).sum()} von {N_LEHRER} Lehrer-Klassen auf "
          f"{len(set(mapping[mapping >= 0].tolist()))} unserer {signmap.N_LABELS}")

    processor = bild_prozessor()
    modell = lehrer_laden(geraet)

    map2 = None
    modell2 = proc2 = None
    bekannt2: set[int] = set()
    if zweiter:
        map2 = zweitlehrer_mapping()
        bekannt2 = zweitlehrer_klassen(map2)
        print(f"[lehrer] zweiter: {ZWEITLEHRER_NAME} (Gewicht {ZWEITLEHRER_GEWICHT})")
        print(f"[lehrer] zweiter: Lizenz {ZWEITLEHRER_LIZENZ}")
        print(f"[lehrer] zweiter: {len(bekannt2)} unserer Klassen mitgemittelt - "
              f"nur wo er das Zeichen kennt")
        modell2, proc2 = zweitlehrer_laden(geraet)

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    eintraege = [e for e in manifest["images"] if e.get("split", "train") == split]
    if limit:
        eintraege = eintraege[:limit]
    ids: list[str] = []
    zaehler: list[int] = []
    bloecke: list[np.ndarray] = []
    zweit_zahlen: list[tuple[int, int]] = []      # (gemittelte Boxen, Boxen) je Bild
    for nr, e in enumerate(eintraege, 1):
        stem = e["id"]
        lab, bild = data / "labels" / f"{stem}.txt", data / "images" / f"{stem}.jpg"
        if not lab.exists() or not bild.exists():
            continue
        h, w = e.get("height", 0), e.get("width", 0)
        boxen, gt_klassen = [], []
        for zeile in lab.read_text(encoding="utf-8").splitlines():
            t = zeile.split()
            if len(t) < 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in t[:5])
            # Klassenindex steht in Spalte 0 (YOLO-Format "cls cx cy w h" - genauso liest es
            # tools/train_det.py). Er wird NUR fuer den Filter gebraucht (kennt der zweite
            # Lehrer dieses Zeichen?), NICHT als Lernziel - das bleibt die Lehrer-Verteilung.
            gt_klassen.append(int(float(t[0])))
            boxen.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
        ids.append(stem)
        zaehler.append(len(boxen))
        if boxen:
            p = teacher_logits_fuer_boxen(bild, boxen, modell, processor, mapping, geraet)
            if modell2 is not None:
                p2 = teacher_logits_fuer_boxen(bild, boxen, modell2, proc2, map2, geraet)
                w2 = ZWEITLEHRER_GEWICHT
                gemittelt = 0
                for j, gt in enumerate(gt_klassen):
                    if gt in bekannt2:
                        p[j] = (1.0 - w2) * p[j] + w2 * p2[j]
                        gemittelt += 1
                zweit_zahlen.append((gemittelt, len(boxen)))
            bloecke.append(p)
        if nr % 2000 == 0:
            print(f"[lehrer]   {nr}/{len(eintraege)}", flush=True)

    logits = (np.concatenate(bloecke) if bloecke
              else np.zeros((0, signmap.N_LABELS), dtype=np.float32))
    offset = np.concatenate([[0], np.cumsum(zaehler)]).astype(np.int64)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"teacher_{split}.npz", logits=logits.astype(np.float16),
                        offset=offset, zaehler=np.asarray(zaehler, dtype=np.int32))
    (out / f"teacher_{split}.json").write_text(json.dumps({
        "lehrer": LEHRER_NAME, "pfad": LEHRER_PFAD, "lizenz": LEHRER_LIZENZ,
        "guete_veroeffentlicht": LEHRER_GUETE, "temperatur": TEMPERATUR,
        "klassen_lehrer": N_LEHRER, "klassen_schueler": signmap.N_LABELS,
        "zuordnung_lehrer_zu_schueler": mapping.tolist(),
        # Zweiter Lehrer (mittelte mit) - samt Gewicht und der Zahl der Boxen, die er
        # ueberhaupt bewerten durfte. Ohne diese Zahl waere spaeter nicht nachvollziehbar,
        # wie viel vom Cache wirklich aus zwei Lehrern stammt.
        "zweiter": (None if map2 is None else {
            "name": ZWEITLEHRER_NAME, "repo": ZWEITLEHRER_REPO, "lizenz": ZWEITLEHRER_LIZENZ,
            "guete_veroeffentlicht": ZWEITLEHRER_GUETE, "gewicht": ZWEITLEHRER_GEWICHT,
            "klassen": N_ZWEITLEHRER, "bekannte_klassen": sorted(bekannt2),
            "zuordnung_zweiter_zu_schueler": map2.tolist(),
            "boxen_gemittelt": int(sum(a for a, _ in zweit_zahlen)),
            "boxen_gesamt": int(sum(b for _, b in zweit_zahlen)),
        }),
        "ids": ids, "boxen_gesamt": int(sum(zaehler)), "bilder": len(ids),
        "hinweis": "Reihenfolge der Boxen = Reihenfolge der Zeilen der Labeldatei",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"bilder": len(ids), "boxen": int(sum(zaehler)),
            "klassen_belegt": int(len(set(mapping[mapping >= 0].tolist()))),
            "zweit_boxen": int(sum(a for a, _ in zweit_zahlen))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Lehrer-Verteilungen (Distillation) cachen")
    ap.add_argument("--data", default="data/det")
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--csv", default="", help="StVO-Tabelle (Default: data/gtsign/<datei>)")
    ap.add_argument("--split", nargs="+", default=["train"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Bilder (Test)")
    ap.add_argument("--zweiter", action="store_true",
                    help="den GTSRB-ViT dazu mitteln (nur wo er das Zeichen kennt)")
    ap.add_argument("--mapping", action="store_true", help="nur die Zuordnung zeigen")
    args = ap.parse_args()

    if args.mapping:
        mapping = lehrer_mapping(stvo_csv(args.csv))
        print(f"[zuordnung] Hauptlehrer: {(mapping >= 0).sum()} von {N_LEHRER} Klassen "
              f"zugeordnet ({LEHRER_NAME})")
        for unser_idx in range(signmap.N_LABELS):
            lehrer = [i for i in range(N_LEHRER) if mapping[i] == unser_idx]
            if lehrer:
                print(f"  {signmap.LABELS[unser_idx]:26s} <- {len(lehrer):3d} "
                      f"(z. B. {lehrer[:6]})")
        leer = [signmap.LABELS[i] for i in range(signmap.N_LABELS)
                if not (mapping == i).any()]
        print(f"[zuordnung] ohne Lehrer: {leer if leer else 'keine'}")

        map2 = zweitlehrer_mapping()
        bekannt = zweitlehrer_klassen(map2)
        print(f"[zuordnung] Zweitlehrer: {(map2 >= 0).sum()} von {N_ZWEITLEHRER} Klassen "
              f"zugeordnet ({ZWEITLEHRER_NAME})")
        for unser_idx in range(signmap.N_LABELS):
            gtsrb = [i for i in range(N_ZWEITLEHRER) if map2[i] == unser_idx]
            if gtsrb:
                print(f"  {signmap.LABELS[unser_idx]:26s} <- {gtsrb}")
        print(f"[zuordnung] Zweitlehrer deckt {len(bekannt)} unserer {signmap.N_LABELS} "
              f"ab; ohne ihn bleibt der Hauptlehrer allein")
        return

    for split in args.split:
        info = cache_bauen(Path(args.data), Path(args.out), split, args.device, args.limit,
                           args.csv, args.zweiter)
        print(f"[lehrer] {split}: {info['bilder']} Bilder, {info['boxen']} Boxen, "
              f"{info['klassen_belegt']} unserer Klassen belegt")
        if args.zweiter:
            print(f"[lehrer] {split}: zweiter Lehrer bei {info['zweit_boxen']} von "
                  f"{info['boxen']} Boxen mitgemittelt")


if __name__ == "__main__":
    main()