"""Zwei Pruefungen VOR dem teuren Kaggle-Lauf (nur lokal, nicht im Auslieferordner).

Teil 1 - Werkzeugebene: tools/real_negatives.py muss zwei Negative-Aufrufe auf denselben
         Split ADDIEREN (eigenes --src) und fuer beide eindeutige Dateinamen schreiben.
         Vorher loeschte der zweite Aufruf die Bilder des ersten stillschweigend.
Teil 2 - Rezepturebene: die Befehlskette aus kaggle/train_kernel.py im Probelauf (PROBE=1).
         Geprueft wird, dass JEDER Eingabepfad, der an ein Werkzeug geht, unter WORK auch
         wirklich liegt. Genau das fehlte beim Lauf vom 03.10.: dort stand "catalogs/" statt
         "kataloge/" und der Lauf brach nach 28 Minuten Datensatzvorbereitung ab.

Teil 3 - Wissens-Distillation: der Lehrer-Cache wird gefunden und --teacher/--distill/
         --temperature stehen im Trainingsbefehl (mit Cache) bzw. eben nicht (ohne).
Teil 4 - Auslieferung: der Exportbefehl traegt --int8 und --fp16, tools/export_onnx.py kennt
         beide Flags, und src/model.js laedt beide Fassungen. Genau hier laege sonst eine
         Datei, die geschrieben, aber nie benutzt wird.

Aufruf:  python tests/kernel_probe.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "_kernel_probe"
# Parameter, die EINGABEN bezeichnen - nur fuer diese Pfade wird die Existenz geprueft.
EINGABE_FLAGS = {"--gtsign", "--gtsign-csv", "--gtsrb", "--gtsrb-gt", "--synset", "--scenes",
                 "--coco", "--real-train", "--real-test", "--csv", "--teacher"}
# NICHT in dieser Menge: --data. Bei tools/train_det.py und tools/teacher.py ist das der
# DATENSATZORDNER data/det - er entsteht erst durch den Datensatzschritt (crops_dataset).
# Eine Existenzpruefung wuerde hier also grundsaetzlich falsch anschlagen; geprueft wird der
# Pfad deshalb ueber die Zeichenkette (siehe teil3).
FEHLER: list[str] = []


def fehler(text: str) -> None:
    print(f"  [FEHLER] {text}")
    FEHLER.append(text)


def teil1() -> None:
    """real_negatives: zwei Aufrufe, ein Split - additiv und ohne Namenskollision."""
    print("[teil 1] tools/real_negatives.py: additiv statt ersetzend")
    arbeit = OUT / "teil1"
    shutil.rmtree(arbeit, ignore_errors=True)
    coco = arbeit / "coco128"
    (coco / "images").mkdir(parents=True)
    (coco / "labels").mkdir(parents=True)
    from PIL import Image
    for i in range(6):
        Image.new("RGB", (96, 96), (200, 200, 200)).save(coco / "images" / f"{i:04d}.jpg")
        (coco / "labels" / f"{i:04d}.txt").write_text("", encoding="utf-8")
    det = arbeit / "det"
    aufrufe = [["--n", "4", "--split", "train"],
               ["--n", "4", "--split", "train", "--src", "eigene (Negativ)"]]
    for nr, extra in enumerate(aufrufe, 1):
        cmd = [sys.executable, str(ROOT / "tools" / "real_negatives.py"),
               "--out", str(det), "--coco", str(coco), "--coco-test-n", "0",
               "--size", "96", *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(ROOT))
        if r.returncode != 0:
            fehler(f"Aufruf {nr} endete mit Code {r.returncode}: {(r.stderr or '')[-300:]}")
            return
    manifest = json.loads((det / "manifest.json").read_text(encoding="utf-8"))
    eintraege = [e for e in manifest["images"] if e.get("split") == "train"]
    dateien = sorted(p.name for p in (det / "images").glob("*.jpg"))
    kennungen = {e["src"] for e in eintraege}
    print(f"         Manifest-Eintraege (train): {len(eintraege)}  "
          f"Bilddateien: {len(dateien)}  Kennungen: {sorted(kennungen)}")
    if len(eintraege) != 8:
        fehler(f"erwartet 8 Eintraege (4+4), gefunden {len(eintraege)} - "
               f"der zweite Aufruf hat den ersten ersetzt")
    if len(dateien) != 8:
        fehler(f"erwartet 8 Bilddateien, gefunden {len(dateien)} - Dateien ueberschreiben sich")
    if kennungen != {"echt (Negativ)", "eigene (Negativ)"}:
        fehler(f"Kennungen stimmen nicht: {sorted(kennungen)}")
    ids = [e["id"] for e in eintraege]
    if len(set(ids)) != len(ids):
        fehler("doppelte IDs im Manifest")
    for e in eintraege:
        if not (det / "images" / f"{e['id']}.jpg").exists():
            fehler(f"Manifest-Eintrag ohne Datei: {e['id']}")
    if not FEHLER:
        print("         ok - 4 + 4 Negative, 8 eigene Dateien, keine Dublette")


def _work_vorbereiten(work: Path, eingang: Path) -> None:
    """Die Ordner anlegen, die der Kernel VOR dem Datensatzbau erzeugt (mit Blinddateien)."""
    for rel in ("kataloge", "gtsrb", "crops/synset-train/verbot", "crops/synset-val/verbot",
                "oi-train", "oi-val", "oi-neg", "negatives/coco128/images",
                "negatives/coco128/labels", "negatives/eigene", "user", "models", "berichte",
                # gtsdb_bereitstellen legt diese drei Ordner an (train/valid/test aus der
                # COCO-Fassung) - ohne sie im Aufbau schlaegt die Pfadpruefung falsch an.
                "gtsdb-coco/train", "gtsdb-coco/valid", "gtsdb-coco/test"):
        (work / rel).mkdir(parents=True, exist_ok=True)
    (work / "kataloge" / "GTSIGN-220.zip").write_bytes(b"PK\x03\x04blind")
    (work / "kataloge" / "class_descriptions_and_stvo.csv").write_text(
        "Class_ID,StVO_Sign_Number\n001,101\n", encoding="utf-8")
    (work / "gtsrb" / "train.zip").write_bytes(b"PK\x03\x04blind")
    for name in ("train", "val", "neg"):
        (work / f"oi-{name}" / "00000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (work / "user" / "Nothing.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (work / "negatives" / "eigene" / "foto1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (work / "negatives" / "coco128" / "images" / "0000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (work / "negatives" / "coco128" / "labels" / "0000.txt").write_text("", encoding="utf-8")
    eingang.mkdir(parents=True, exist_ok=True)


def _kernel_laden(work: Path, eingang: Path):
    """train_kernel.py mit umgebogenen Pfaden laden (WORK/EINGANG kommen aus der Umgebung)."""
    os.environ["SCHILDER_WORK"] = str(work)
    os.environ["SCHILDER_INPUT"] = str(eingang)
    os.environ["SCHILDER_PROBE"] = "1"
    spec = importlib.util.spec_from_file_location("tk_probe",
                                                  ROOT / "kaggle" / "train_kernel.py")
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


def _befehle(k, work: Path, mit_eigenen: bool) -> str:
    """datensatz_bauen im Probelauf: die Befehle werden gedruckt, nicht ausgefuehrt."""
    p = k.Protokoll(work / "berichte" / "protokoll.txt")
    eigene = (["negatives/eigene/foto1.jpg"], []) if mit_eigenen else ([], [])
    # gtsdb-Ergebnis nachstellen: so liefert es gtsdb_bereitstellen im Echtlauf (Bildzahlen).
    ein = {"coco128": "negatives/coco128", "eigene": eigene,
           "gtsdb": {"gefunden": True, "teile": {"train": 383, "valid": 108, "test": 54}}}
    puffer = io.StringIO()
    with contextlib.redirect_stdout(puffer):
        dauer, erwartet = k.datensatz_bauen(p, ein)
    p.datei.close()
    return puffer.getvalue()


def _erwartet_zeile(text: str) -> dict:
    """Die Zeile "[daten] erwartet aus dem Rezept: train=..., val=..., neg=..." auswerten."""
    for z in text.splitlines():
        if z.startswith("[daten] erwartet aus dem Rezept:"):
            rest = z.split(":", 1)[1].replace(" ", "")
            return {k: int(v) for k, v in (t.split("=") for t in rest.split(",") if t)}
    return {}


def teil2() -> None:
    """Rezepturebene: jeder Eingabepfad muss unter WORK liegen."""
    print("[teil 2] kaggle/train_kernel.py: Befehlskette im Probelauf")
    arbeit = OUT / "teil2"
    shutil.rmtree(arbeit, ignore_errors=True)
    work, eingang = arbeit / "working", arbeit / "input"
    _work_vorbereiten(work, eingang)
    k = _kernel_laden(work, eingang)
    text = _befehle(k, work, mit_eigenen=True)

    if "catalogs" in text:
        fehler("immer noch 'catalogs/' in den Befehlen - es muss 'kataloge/' heissen")
    for erwartet in ("--gtsign kataloge/GTSIGN-220.zip",
                     "--gtsign-csv kataloge/class_descriptions_and_stvo.csv",
                     "--gtsrb gtsrb/train.zip",
                     "--synset crops/synset-train",
                     "--scenes oi-train",
                     "--scenes oi-neg",
                     "--src eigene (Negativ)"):
        if erwartet not in text:
            fehler(f"fehlt in der Befehlskette: {erwartet}")

    # Der Kern dieser Pruefung: jeder Eingabepfad der Reihe nach.
    geprueft = 0
    zeilen = [z for z in text.splitlines() if z.startswith("[befehl]")]
    for zeile in zeilen:
        teile = zeile.split()[2:]      # "[befehl]" und das Python-Programm abschneiden
        for i, tok in enumerate(teile):
            if tok in EINGABE_FLAGS and i + 1 < len(teile):
                pfad = teile[i + 1]
                geprueft += 1
                if not (work / pfad).exists():
                    fehler(f"{tok} {pfad} liegt nicht unter WORK")
    print(f"         {len(zeilen)} Werkzeugaufrufe, {geprueft} Eingabepfade geprueft")
    if geprueft < 10:
        fehler(f"zu wenige Eingabepfade erkannt ({geprueft}) - Pruefung greift nicht")

    # Die erwarteten Bildzahlen: 30000 Katalog + 500 synthetisch + 3600 echt + 383 Szenen
    # + 1800 eigene = 36283; val = 2500 + 108 + 54 = 2662; neg = 1000 + 500 = 1500.
    soll = {"train": 36283, "val": 2662, "neg": 1500}
    ist = _erwartet_zeile(text)
    print(f"         erwartet (gedruckt): {ist}")
    if ist != soll:
        fehler(f"Erwartung falsch gerechnet: {ist} statt {soll}")

    # Gegenprobe: fehlt ein Ordner, darf der Lauf NICHT abbrechen, sondern die Quelle
    # auslassen (und der reine Negativ-Schritt muss verschwinden).
    vorher = len(FEHLER)
    shutil.rmtree(work / "oi-neg", ignore_errors=True)
    text2 = _befehle(k, work, mit_eigenen=False)
    if "negative messlatte" in text2:
        fehler("der Negativ-Schritt laeuft ohne Szenen - crops_dataset bricht dort ab")
    if "[hinweis] Open Images (Negative)" not in text2:
        fehler("fehlende Szenen werden nicht gemeldet")
    if len(FEHLER) == vorher:
        print("         ok - fehlende Quelle wird gemeldet statt den Lauf zu beenden")

    if not FEHLER:
        print("         ok - Katalog unter kataloge/, GTSRB dabei, Negative additiv")


def _protokoll_text(k, work: Path, fn, *args) -> str:
    """Eine Kernel-Funktion im Probelauf aufrufen und alles mitgelesen zurueckgeben."""
    p = k.Protokoll(work / "berichte" / "protokoll.txt")
    puffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(puffer):
            fn(p, *args)
    finally:
        p.datei.close()
    return puffer.getvalue()


def teil3() -> None:
    """Wissens-Distillation: kommt der Cache an, und traegt das Training den Lehrer mit?

    Warum als eigene Pruefung: die Distillation haengt an VIER Stellen zusammen - Cache-Pfad,
    StVO-Tabelle, Trainingsflags und Gewicht. Ein Fehler an einer Stelle faellt im Echtlauf
    erst nach einer Stunde Datensatzbau auf. Geprueft wird beides: MIT Cache (die Flags
    muessen im Trainingsbefehl stehen) und OHNE (dann darf kein --teacher auftauchen).
    """
    print("[teil 3] Wissens-Distillation: Cache und Trainingsflags")
    arbeit = OUT / "teil3"
    shutil.rmtree(arbeit, ignore_errors=True)
    work, eingang = arbeit / "working", arbeit / "input"
    _work_vorbereiten(work, eingang)
    k = _kernel_laden(work, eingang)

    # (a) ohne Cache: keine Spur von Distillation
    text_aus = _protokoll_text(k, work, k.trainieren, False)
    if "--teacher" in text_aus or "--distill" in text_aus:
        fehler("ohne Lehrer-Cache steht trotzdem --teacher/--distill im Trainingsbefehl")

    # (b) mit Cache (Blinddatei genuegt - im Probelauf wird nichts gerechnet)
    (work / "data" / "teacher").mkdir(parents=True, exist_ok=True)
    ziel = work / "data" / "teacher" / f"teacher_{k.LEHRER['split']}.npz"
    ziel.write_bytes(b"PK\x03\x04blind")
    hat = _protokoll_text(k, work, k.lehrer_cache, work / "input", True)
    text_mit = _protokoll_text(k, work, k.trainieren, True)

    for erwartet in ("--teacher data/teacher",
                     f"--distill {k.LEHRER['distill']}",
                     f"--temperature {k.LEHRER['temperature']}"):
        if erwartet not in text_mit:
            fehler(f"fehlt im Trainingsbefehl: {erwartet}")
    if "[lehrer] Cache bereit" not in hat:
        fehler("lehrer_cache meldet den vorhandenen Cache nicht als bereit")
    # Das Gewicht: gemessen traegt eine positive Zelle KL x T^2 von 2-9 bei, der
    # Klassifikationsverlust liegt bei 0,1-0,3. Ein Gewicht ab 1,0 wuerde ihn ueberstimmen.
    if not 0.0 < k.LEHRER["distill"] <= 0.5:
        fehler(f"Distillationsgewicht {k.LEHRER['distill']} ausserhalb des sinnvollen "
               "Bereichs (0..0,5)")
    if not text_mit.split("[befehl]")[1].count("--data data/det"):
        fehler("--data data/det fehlt im Trainingsbefehl")

    # Der Eingabepfad-Check aus Teil 2 gilt auch fuer diese Befehle.
    for zeile in [z for z in text_mit.splitlines() if z.startswith("[befehl]")]:
        teile = zeile.split()[2:]
        for i, tok in enumerate(teile):
            if tok in EINGABE_FLAGS and i + 1 < len(teile) and not (work / teile[i + 1]).exists():
                fehler(f"{tok} {teile[i + 1]} liegt nicht unter WORK")
    if not FEHLER:
        print("         ok - Cache gefunden, --teacher/--distill/--temperature gesetzt, "
              "Gewicht im sinnvollen Bereich")


def teil4() -> None:
    """Auslieferung: traegt der Export die Fassungen (int8/fp16), und kennt sie der Browser?

    Zwei Fehler, die im Echtlauf erst nach 60 Minuten auffallen wuerden und beide still sind:
    (a) ein Flag, das tools/export_onnx.py nicht kennt (dann bricht der Export ab und der
    Kaggle-Lauf hat kein Modell), (b) eine Datei, die zwar geschrieben, aber von src/model.js
    gar nicht geladen wird - dann laege im Ergebnisordner eine halbe Datei, die niemand nutzt.
    """
    print("[teil 4] Auslieferung: int8- und fp16-Fassung im Export und im Browser")
    arbeit = OUT / "teil4"
    shutil.rmtree(arbeit, ignore_errors=True)
    work, eingang = arbeit / "working", arbeit / "input"
    _work_vorbereiten(work, eingang)
    k = _kernel_laden(work, eingang)
    text = _protokoll_text(k, work, k.exportieren)

    werkzeug = (ROOT / "tools" / "export_onnx.py").read_text(encoding="utf-8")
    seite = (ROOT / "src" / "model.js").read_text(encoding="utf-8")
    # (a) Jedes Flag muss im Befehl stehen UND vom Werkzeug gekannt werden. Ein Flag, das
    #     argparse nicht kennt, beendet den Export mit Code 2 - und der Lauf hat kein Modell.
    for flag in ("--dynamic", "--int8", "--fp16"):
        if flag not in text:
            fehler(f"der Exportbefehl traegt {flag} nicht")
        if f'"{flag}"' not in werkzeug:
            fehler(f"tools/export_onnx.py kennt {flag} nicht - der Export wuerde abbrechen")
    # (b) Jede geschriebene Fassung muss src/model.js auch laden, sonst liegt sie ungenutzt
    #     im Modellordner (und niemand merkt es, weil die Seite ja laeuft).
    for schluessel in ("int8", "fp16", "onnx"):
        if f"'{schluessel}'" not in seite and f"'{schluessel}':" not in seite:
            fehler(f"src/model.js laedt die {schluessel}-Fassung nicht "
                   f"(eine geschriebene Datei, die niemand benutzt)")
    if "--calib-data" not in text:
        fehler("der Export kalibriert nicht auf den echten Bildern (--calib-data fehlt)")
    if not FEHLER:
        print("         ok - --int8/--fp16 im Export, beide Fassungen kennt src/model.js")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    zeige = "--zeigen" in sys.argv
    if zeige:
        arbeit = OUT / "zeigen"
        shutil.rmtree(arbeit, ignore_errors=True)
        work, eingang = arbeit / "working", arbeit / "input"
        _work_vorbereiten(work, eingang)
        k = _kernel_laden(work, eingang)
        print(_befehle(k, work, mit_eigenen=True))
        return 0
    teil1()
    teil2()
    teil3()
    teil4()
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
