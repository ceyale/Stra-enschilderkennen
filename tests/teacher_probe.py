"""Prueft die LEHRER-Mittelung in tools/teacher.py, ohne Netz und ohne 86-Mio.-Modelle.

Warum als eigenes Skript: die Mittelung des zweiten Lehrers ist die einzige Stelle, an der
ein Fehler STILL waere - es kommt eine Datei heraus, das Training laeuft, und der Cache
enthaelt trotzdem Unsinn. Genau zwei Dinge muessen stimmen:

  1. Ein Lehrer, der ein Zeichen NICHT kennt, darf es nicht mitbewerten. Sonst schreibt er
     fuer "tempo40" (kennt er nicht) seine Meinung "tempo30" in den Cache - ein bekannt
     falsches Lernziel.
  2. Wo er es kennt, muss exakt gewichtet gemittelt werden (0,5 / 0,5), und die Begleitdatei
     muss festhalten, bei wie vielen Boxen das der Fall war.

Beide Lehrer werden hier durch feste Zahlen ersetzt (`teacher_logits_fuer_boxen` wird
ausgetauscht) - geprueft wird also die Rechnung und die Filterung, nicht der ViT selbst.

Aufruf:  python tests/teacher_probe.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import signmap  # noqa: E402
import teacher  # noqa: E402

OUT = ROOT / "data" / "_teacherprobe"
FEHLER: list[str] = []
# Der Cache wird als float16 gespeichert (teacher.py: .astype(np.float16)) - 0,8 ist dort
# 0,79980... Deshalb wird mit dieser Toleranz verglichen und nicht mit 1e-6.
F16 = 2e-3


def fehler(text: str) -> None:
    print(f"  [FEHLER] {text}")
    FEHLER.append(text)


def datensatz_bauen(ordner: Path) -> Path:
    """Mini-Datensatz: 2 Bilder mit je 2 Boxen (dieselbe Klasse zweimal)."""
    from PIL import Image

    (ordner / "images").mkdir(parents=True, exist_ok=True)
    (ordner / "labels").mkdir(parents=True, exist_ok=True)
    # Eine Klasse, die BEIDE Lehrer kennen (tempo50), und eine, die nur der Hauptlehrer kennt
    # (tempo40) - genau der Unterschied, um den es hier geht.
    for id_name, cls in (("bild1", signmap.CLASS_ID["tempo50"]),
                         ("bild2", signmap.CLASS_ID["tempo40"])):
        Image.new("RGB", (320, 320), (128, 128, 128)).save(ordner / "images" / f"{id_name}.jpg")
        (ordner / "labels" / f"{id_name}.txt").write_text(
            "".join(f"{cls} 0.5 0.5 0.2 0.2\n" for _ in range(2)), encoding="utf-8")
    (ordner / "manifest.json").write_text(json.dumps({"images": [
        {"id": "bild1", "split": "train", "height": 320, "width": 320},
        {"id": "bild2", "split": "train", "height": 320, "width": 320}]}), encoding="utf-8")
    return ordner


def verteilung(klasse: int, sicher: float):
    """Ersatz fuer teacher_logits_fuer_boxen: setzt alle Masse auf EINE Klasse."""
    def fn(bild_pfad, boxen, modell, processor, mapping, geraet, **rest):
        p = np.full((len(boxen), signmap.N_LABELS), 1e-6, dtype=np.float32)
        p[:, klasse] = sicher
        return p
    return fn


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    arbeit = OUT / "arbeit"
    shutil.rmtree(arbeit, ignore_errors=True)
    det = datensatz_bauen(arbeit / "det")
    csv = arbeit / "stvo.csv"
    # Der Hauptlehrer kennt hier genau eine Klasse: unsere "tempo50" (StVO 274-50).
    csv.write_text("Class_ID,StVO_Sign_Number\n0,274-50\n", encoding="utf-8")

    print("[1] Zuordnung des zweiten Lehrers (echte Tabelle aus tools/signmap.py)")
    map2 = teacher.zweitlehrer_mapping()
    bekannt = teacher.zweitlehrer_klassen(map2)
    tempo50, tempo40 = signmap.CLASS_ID["tempo50"], signmap.CLASS_ID["tempo40"]
    tempo30 = signmap.CLASS_ID["tempo30"]
    print(f"    GTSRB-Klassen zugeordnet: {int((map2 >= 0).sum())}/{teacher.N_ZWEITLEHRER}, "
          f"deckt {len(bekannt)} unserer {signmap.N_LABELS}")
    if int((map2 >= 0).sum()) != teacher.N_ZWEITLEHRER:
        fehler("nicht alle GTSRB-Klassen sind zugeordnet")
    if tempo50 not in bekannt or tempo40 in bekannt:
        fehler(f"Abdeckung stimmt nicht: tempo50 drin={tempo50 in bekannt}, "
               f"tempo40 drin={tempo40 in bekannt}")

    # Beide Lehrer durch feste Zahlen ersetzen. Der Zweitlehrer sagt absichtlich etwas
    # ANDERES (tempo30 mit 0,9), damit die Mittelung ueberhaupt sichtbar wird.
    haupt = verteilung(tempo50, 0.8)
    zweit = verteilung(tempo30, 0.9)

    def beide(bild_pfad, boxen, modell, processor, mapping, geraet, **rest):
        if modell == "Zweitlehrer":
            return zweit(bild_pfad, boxen, modell, processor, mapping, geraet)
        return haupt(bild_pfad, boxen, modell, processor, mapping, geraet)

    echt = teacher.teacher_logits_fuer_boxen
    teacher.lehrer_mapping = lambda pfad: np.array(
        [signmap.CLASS_ID["tempo50"]] + [-1] * (teacher.N_LEHRER - 1))
    teacher.bild_prozessor = lambda: None
    teacher.lehrer_laden = lambda geraet: "Hauptlehrer"
    teacher.zweitlehrer_laden = lambda geraet: ("Zweitlehrer", None)
    teacher.teacher_logits_fuer_boxen = haupt

    print("[2] ohne zweiten Lehrer: unveraendert")
    info = teacher.cache_bauen(det, arbeit / "cache1", "train", "cpu", 0, str(csv), False)
    z1 = np.load(arbeit / "cache1" / "teacher_train.npz")["logits"].astype(np.float32)
    kopf1 = json.loads((arbeit / "cache1" / "teacher_train.json").read_text(encoding="utf-8"))
    print(f"    {info['bilder']} Bilder, {info['boxen']} Boxen, zweiter={kopf1['zweiter']}")
    if kopf1["zweiter"] is not None:
        fehler("ohne --zweiter steht trotzdem ein zweiter Lehrer im Kopf")
    if abs(float(z1[0, tempo50]) - 0.8) > F16:
        fehler(f"Cache ohne Mittelung falsch: {z1[0, tempo50]:.4f} statt 0,8")

    print("[3] mit zweitem Lehrer: nur wo er das Zeichen kennt")
    teacher.teacher_logits_fuer_boxen = beide
    info2 = teacher.cache_bauen(det, arbeit / "cache2", "train", "cpu", 0, str(csv), True)
    z2 = np.load(arbeit / "cache2" / "teacher_train.npz")["logits"].astype(np.float32)
    kopf2 = json.loads((arbeit / "cache2" / "teacher_train.json").read_text(encoding="utf-8"))
    zw = kopf2["zweiter"] or {}
    print(f"    {info2['bilder']} Bilder, {info2['boxen']} Boxen, gemittelt bei "
          f"{zw.get('boxen_gemittelt')} von {zw.get('boxen_gesamt')}")
    w = teacher.ZWEITLEHRER_GEWICHT
    erwartet = (1 - w) * 1e-6 + w * 0.9        # bild1 (tempo50) -> beide Lehrer
    if abs(float(z2[0, tempo30]) - erwartet) > F16:
        fehler(f"Mittelung falsch: tempo30 {z2[0, tempo30]:.4f} statt {erwartet:.4f}")
    if abs(float(z2[2, tempo50]) - 0.8) > F16 or float(z2[2, tempo30]) > 1e-4:
        fehler(f"bild2 (tempo40, kennt der Zweitlehrer NICHT) wurde mitgemittelt: "
               f"tempo50={z2[2, tempo50]:.4f} tempo30={z2[2, tempo30]:.4f}")
    if zw.get("boxen_gemittelt") != 2 or zw.get("boxen_gesamt") != 4:
        fehler(f"Zaehler im Kopf falsch: {zw.get('boxen_gemittelt')} von "
               f"{zw.get('boxen_gesamt')} (erwartet 2 von 4)")
    if not zw.get("lizenz"):
        fehler("Lizenz des zweiten Lehrers fehlt im Kopf")

    teacher.teacher_logits_fuer_boxen = echt
    print()
    if FEHLER:
        print(f"[ergebnis] {len(FEHLER)} Fehler:")
        for f in FEHLER:
            print(f"  - {f}")
        return 1
    print("[ergebnis] alle Pruefungen bestanden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())