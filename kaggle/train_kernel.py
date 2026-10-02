"""
kaggle/train_kernel.py - Ein Kaggle-Lauf: Datensatz bauen, trainieren, exportieren, messen.

Warum dieses Skript existiert: Auf dem Entwicklungsrechner blockiert die Windows-
Anwendungssteuerung (Smart App Control, `WinError 4551` beim Laden von
`torch_global_deps.dll`) die PyTorch-DLLs - dort laeuft weder Training noch Export, und
ohne `hybrid_net` (es liefert `SIGN_LABELS`) laesst sich auch kein Datensatz bauen. Der
Rechenweg bleibt trotzdem derselbe: dieses Skript ruft genau dieselben Werkzeuge mit
denselben Aufrufen auf, die in `docs/TRAINING.md` §4.3 (Daten) und §5 (Training) stehen -
nur auf einer anderen Maschine.

Eingaben (Kaggle-Datensatz `raphbre/schilder-det-raw`, gebaut von `kaggle/run.ps1 -Step stage`):

    tools/                 der Werkzeugkasten dieses Repos (Kernel braucht kein Netz)
    gtsrb/*.zip            GTSRB Final Training Images / Final Test Images / Final Test GT
    negatives/coco128.zip  128 echte Fotos (CC BY 4.0) fuer echte Negative
    user/Nothing.jpg       eigenes Foto ohne Schild (obere Haelfte Training, untere Messlatte)

Ausgaben (in /kaggle/working, abholbar mit `kaggle/run.ps1 -Step pull`):

    models/signs-det.pt        Checkpoint (Ausgangspunkt fuer Export und weitere Laeufe)
    models/signs-det.onnx      ausgeliefertes Modell (dynamische Hoehe/Breite)
    models/labels.json         Klassen/Eingabegroesse fuer src/model.js
    models/manifest.json       Metriken, Paritaet, sha256
    berichte/val.json          Auswertung der Messlatte (2 000 Bilder, Split val)
    berichte/neg.json          Gegenprobe: Fehlalarme (Split neg)
    berichte/val384.json       dieselbe Messlatte bei 384 px (Fotomodus)
    berichte/val_block4.json   dieselbe Messlatte mit dem ALTEN Checkpoint (Vergleich)
    berichte/model-out.json    Fixture fuer tests/model.test.js (aus genau diesem Modell)
    berichte/protokoll.txt     vollstaendiges Protokoll aller Werkzeugaufrufe
    data/det.zip               der gebaute Datensatz (fuer weitere Laeufe ohne Neuaufbau)
    kaggle_report.json         Kurzfassung: Dauer je Schritt, Kennzahlen, Wege

Zwei Dinge, die der Kernel aus gemessenem Grund selbst erledigt: er stellt `onnxruntime`
sicher (Kaggle bringt es nicht mit - der erste Lauf starb deswegen nach dem Training) und
er laesst keinen Nebenschritt den Lauf entwerten (Export, Auswertung und Fixture laufen mit
fatal=False; das Training selbst bleibt toedlich, weil ohne Checkpoint nichts zu retten ist).

Schnelltest ohne Rechnen (prueft nur Wege und Befehle): Umgebungsvariable SCHILDER_PROBE=1
setzen - dann wird kein Schritt ausgefuehrt, aber alles aufgelistet.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Das Rezept. Alle Zahlen stehen hier zusammen, damit ein Lauf nachvollziehbar
# bleibt; die Vorgaben sind genau die, mit denen die Zahlen in docs/TRAINING.md
# §5 und models/README.md gemessen wurden (Block 4), zusaetzlich die echten
# Negative aus Block 5 (29 600 statt 26 000 Trainingsbilder).
# ---------------------------------------------------------------------------
DATEN = dict(
    n_train=20000,        # GTSRB-Kompositionen, Typen ausgeglichen (--balance)
    n_val=1800,           # GTSRB-Testbilder als Messlatte (bleibt unausgeglichen)
    n_fehlend=3000,       # Lueckenschluss: hinweis + ortstafel (GTSRB hat sie nicht)
    n_fehlend_val=200,    # dieselbe Luecke in der Messlatte (1800 + 200 = 2 000)
    n_neg_train=3000,     # synthetische Bilder ohne Schild
    n_neg_neg=500,        # synthetische Negative als Messlatte
    n_echt_train=3600,    # echte Fotos ohne Schild (coco128 + eigenes Foto)
    n_echt_neg=600,       # deren gesperrter Teil als Messlatte
    coco_test_n=12,       # die letzten 12 coco-Fotos sind nur im Split neg
    user_share_train=0.55,
    user_share_neg=0.35,
    size=320,
)
TRAINING = dict(
    preset="balanced",
    size=320,
    batch=16,
    epochs=220,
    steps=150,            # 150 x 16 = 2 400 Bilder je Epoche; mehr Abdeckung der 29 600 Bilder
    lr=1.5e-3,
    degrade=0.6,
    zoom=0.5,             # Mehrskaligkeit - Grund fuer die freie Eingabegroesse im Browser
    workers=4,            # Kaggle hat 4 vCPU
    eval_every=10,
    save_every=10,
    seed=7,
    obj_norm="pos",       # "sqrt" waere ruhiger bei so vielen Negativen - ungemessen
    val_split="val",
)
AUSWERTUNG = dict(conf=0.25, iou=0.5, iou_det=0.45, zweite_groesse=384)
AUFRAEUMEN = True         # Bildordner nach dem Zippen loeschen (sonst 65 000 Ausgabedateien)
ERWARTET = {"train": 29600, "val": 2000, "neg": 1100}   # PLAN.md, Block 5
FEHLER: list[str] = []        # Schritte, die trotz "nicht toedlich" schiefgingen (fuer den Bericht)


class Protokoll:
    """Jede Zeile eines Werkzeugs geht auf die Konsole UND in eine Datei.

    Kaggle zeigt die Ausgabe zwar live, aber beim Herunterladen der Ergebnisse ist die
    Datei das einzige, was bleibt - und die Trainingskurve ist der eigentliche Beleg.
    """

    def __init__(self, pfad: Path):
        self.pfad = pfad
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        self.datei = self.pfad.open("a", encoding="utf-8")

    def zeile(self, text: str = "") -> None:
        print(text, flush=True)
        self.datei.write(text + "\n")
        self.datei.flush()


def python(*args: str) -> list[str]:
    """Werkzeugaufruf in der Sprache dieses Repos: python tools/<werkzeug> <argumente>."""
    return [sys.executable, *args]


def lauf(p: Protokoll, cmd: list[str], was: str, fatal: bool = True) -> float:
    """Ein Werkzeug aufrufen, Ausgabe mitschreiben, Dauer zurueckgeben.

    cwd ist /kaggle/working: relativ zum Arbeitsverzeichnis liegen tools/, data/, models/
    und berichte/. Weil Python das Verzeichnis des aufgerufenen Skripts an den Suchpfad
    haengt, finden die Werkzeuge ihre Nachbarmodule (detmath, synth_data, hybrid_net) von
    allein - dieselben Aufrufe wie in der Doku, nur mit relativen Wegen.

    fatal=False fuer Schritte, die den Lauf nicht wertlos machen duerfen (Export, Auswertung,
    Fixture). Warum das wichtig ist: der erste Kaggle-Lauf hatte nach 43 Minuten Training
    fertig - und der anschliessende Export riss alles mit, weil onnxruntime fehlte; die
    Ausgabe eines abgebrochenen Laufs ist weg. Ein einzelner Nebenschritt darf so etwas
    nicht mehr ausloesen.
    """
    p.zeile()
    p.zeile("=" * 78)
    p.zeile(f"[schritt] {was}")
    p.zeile(f"[befehl]  {' '.join(cmd)}")
    p.zeile("=" * 78)
    if PROBE:
        return 0.0
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(WORK), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    assert proc.stdout is not None
    for aus in proc.stdout:
        p.zeile(aus.rstrip("\n"))
    code = proc.wait()
    dauer = time.perf_counter() - t0
    if code != 0:
        p.zeile(f"[fehler] {was} endete mit Code {code}")
        if fatal:
            raise SystemExit(f"{was} fehlgeschlagen (Code {code}) - siehe berichte/protokoll.txt")
        FEHLER.append(was)
        p.zeile(f"[weiter] {was} uebersprungen - alle anderen Schritte laufen weiter")
        return dauer
    p.zeile(f"[dauer]  {was}: {dauer/60:.1f} min")
    return dauer

import zipfile
from pathlib import Path

WORK = Path(os.environ.get("SCHILDER_WORK", "/kaggle/working"))
EINGANG = Path(os.environ.get("SCHILDER_INPUT", "/kaggle/input"))
PROBE = os.environ.get("SCHILDER_PROBE", "") not in ("", "0")



def eingang_finden() -> Path:
    """Den Rohdaten-Datensatz im Kernel finden - egal in welcher Einhaengung.

    Kaggle legt Datensaetze je nach Fassung direkt unter /kaggle/input/<slug> oder unter
    /kaggle/input/datasets/<besitzer>/<slug> ab (beides gemessen: der erste Lauf brach ab,
    weil nur die oberste Ebene durchsucht wurde und dort "datasets" stand). Gesucht wird
    nicht nach einem Namen, sondern nach dem Erkennungsmerkmal: ein Ordner, in dem
    tools/train_det.py (oder tools.zip) liegt. Die Suche ist auf vier Ebenen begrenzt -
    tiefer liegt nichts Sinnvolles, und ein vollstaendiger Durchlauf ueber 50 000 Dateien
    waere unnoetig langsam.
    """
    if not EINGANG.exists():
        raise SystemExit(f"{EINGANG} fehlt - laeuft das hier ueberhaupt auf Kaggle?")
    gesehen: list[Path] = []
    for tiefe in range(1, 5):
        for kandidat in sorted(EINGANG.glob("/".join(["*"] * tiefe))):
            if not kandidat.is_dir():
                continue
            gesehen.append(kandidat)
            if (kandidat / "tools" / "train_det.py").exists() or (kandidat / "tools.zip").exists():
                return kandidat
    raise SystemExit("kein Rohdaten-Datensatz gefunden (tools/train_det.py fehlt); in "
                     f"{EINGANG} liegt: "
                     + ", ".join(p.relative_to(EINGANG).as_posix() for p in gesehen[:20]))


def geraet(p: Protokoll) -> str:
    """Rechengeraet melden - und auf CPU gar nicht erst anfangen.

    Ein CPU-Lauf ueber 22 000 Schritte dauerte Stunden und verbrauchte die Wochenquote,
    ohne dass das Ergebnis anders waere. Fehlt die GPU, ist das ein Fehler in der
    Kernel-Einstellung (enable_gpu), kein Grund zum Durchhalten.
    """
    if PROBE:
        return "probe"
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        p.zeile(f"[geraet] CUDA: {name}  torch={torch.__version__}")
        return name
    p.zeile("[geraet] KEINE CUDA-GPU - Abbruch. In kaggle/kernel-metadata.json muss "
            '"enable_gpu": true stehen (und der Kernel braucht einen GPU-Kontingent-Rest).')
    raise SystemExit("ohne GPU nicht starten")


def entpacken(p: Protokoll, quelle: Path, ziel: Path, was: str) -> None:
    """Ein ZIP im Kernel entpacken (Python, nicht `unzip`: das Werkzeug ist nicht garantiert)."""
    if PROBE:
        p.zeile(f"[probe] entpacken {quelle} -> {ziel}")
        return
    ziel.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(quelle) as z:
        z.extractall(ziel)
    p.zeile(f"[entpackt] {was}: {quelle.name} -> {ziel}")



def zip_aus_ordner(p: Protokoll, quelle: Path, ziel: Path, was: str) -> None:
    """Einen Ordner so packen, wie das Original-Archiv aufgebaut war (Wege bleiben erhalten)."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    t0, anzahl = time.perf_counter(), 0
    with zipfile.ZipFile(ziel, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for datei in sorted(quelle.rglob("*")):
            if datei.is_file():
                z.write(datei, datei.relative_to(quelle).as_posix())
                anzahl += 1
    p.zeile(f"[gepackt] {was}: {anzahl} Dateien -> {ziel.name} "
            f"({ziel.stat().st_size/1e6:.1f} MB, {time.perf_counter()-t0:.0f} s)")


def archiv_bereitstellen(p: Protokoll, gtsrb_roh: Path, basis: str, zielname: str, was: str) -> None:
    """Ein GTSRB-Archiv in WORK/gtsrb bereitstellen - egal in welcher Form es ankommt.

    Gemessen (kaggle/README.md): Kaggle entpackt hochgeladene ZIPs beim Anlegen des
    Datensatzes - aus gtsrb-train.zip wird der Ordner gtsrb-train/. Die Werkzeuge brauchen
    aber ein Archiv (zipfile + ROI-CSV im selben Zugriff). Deshalb wird hier EINE Form
    hergestellt: liegt das ZIP noch da, wird es kopiert; liegt nur der entpackte Ordner,
    wird er neu gepackt. Der Inhalt bleibt byte-gleich, der gebaute Datensatz also derselbe.
    """
    ziel = WORK / "gtsrb" / zielname
    zip_roh, ordner_roh = gtsrb_roh / f"{basis}.zip", gtsrb_roh / basis
    if PROBE:
        p.zeile(f"[probe] {was}: {'ZIP kopieren' if zip_roh.exists() else 'Ordner neu packen'} "
                f"-> gtsrb/{zielname}")
        return
    if zip_roh.exists():
        shutil.copy2(zip_roh, ziel)
        p.zeile(f"[kopiert] {zip_roh.name} -> gtsrb/{zielname} ({ziel.stat().st_size/1e6:.1f} MB)")
    elif ordner_roh.exists():
        zip_aus_ordner(p, ordner_roh, ziel, was)
    else:
        raise SystemExit(f"{was}: weder {zip_roh} noch {ordner_roh} im Rohdaten-Datensatz")


def coco_bereitstellen(p: Protokoll, roh: Path) -> str:
    """coco128 als Ordner unter WORK/negatives bereitstellen; Pfad relativ zurueckgeben.

    tools/real_negatives.py will einen Ordner (--coco). Kaggle liefert ihn entpackt, moeglich
    ist aber auch das ZIP; je nach Entpacktiefe liegt "images/train2017" eine Ebene hoeher
    oder tiefer. Deshalb wird gesucht statt geraten.
    """
    ziel = WORK / "negatives" / "coco128"
    if PROBE:
        p.zeile(f"[probe] coco128 bereitstellen -> {ziel}")
        return "negatives/coco128"
    for kandidat in (roh / "negatives" / "coco128", roh / "negatives" / "coco128" / "coco128",
                     roh / "negatives"):
        if (kandidat / "images" / "train2017").exists():
            shutil.copytree(kandidat, ziel, dirs_exist_ok=True)
            fotos = len(list((ziel / "images" / "train2017").glob("*")))
            p.zeile(f"[kopiert] coco128: {fotos} Fotos aus {kandidat} -> negatives/coco128")
            return "negatives/coco128"
    zip_roh = roh / "negatives" / "coco128.zip"
    if zip_roh.exists():
        entpacken(p, zip_roh, WORK / "negatives", "coco128 (echte Negative)")
        if not (ziel / "images" / "train2017").exists():   # ZIP enthaelt den Ordner coco128/
            raise SystemExit(f"coco128 unerwartet aufgebaut: {ziel}")
        return "negatives/coco128"
    raise SystemExit("coco128 fehlt im Rohdaten-Datensatz (weder Ordner noch ZIP gefunden)")


def onnxruntime_sichern(p: Protokoll) -> bool:
    """onnxruntime bereitstellen - ohne diese Bibliothek gibt es keinen Export und keine Fixture.

    Gemessen: Kaggle bringt onnxruntime **nicht** mit. Der erste Lauf starb nach 43 Minuten
    Training im anschliessenden Export an `ModuleNotFoundError: No module named
    'onnxruntime'` - und weil der Lauf mit Fehler endete, war die ganze Ausgabe weg.
    Ebenfalls gemessen: ein Kernel mit `enable_internet: true` darf es installieren (Probe:
    `pip install onnxruntime` -> 1.30.0, dieselbe Fassung wie lokal). Liegt im Eingang ein
    Ordner `wheels/`, wird von dort installiert - dann braucht der Lauf kein Netz.
    """
    if PROBE:
        return True
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime
        p.zeile(f"[onnxruntime] vorhanden: {onnxruntime.__version__}")
        return True
    wheels = next((k for k in sorted(EINGANG.rglob("wheels")) if k.is_dir()), None)
    befehl = [sys.executable, "-m", "pip", "install", "-q", "--no-input"]
    if wheels is not None:
        befehl += ["--no-index", "--find-links", str(wheels), "onnxruntime"]
        p.zeile(f"[onnxruntime] fehlt - Installation aus {wheels} (ohne Netz)")
    else:
        befehl += ["onnxruntime"]
        p.zeile("[onnxruntime] fehlt - Installation von PyPI (Kernel braucht Internet)")
    ergebnis = subprocess.run(befehl, capture_output=True, text=True)
    if ergebnis.returncode != 0:
        p.zeile("[onnxruntime] Installation fehlgeschlagen: "
                + (ergebnis.stderr or ergebnis.stdout or "")[-500:])
        return False
    try:
        import onnxruntime
        p.zeile(f"[onnxruntime] installiert: {onnxruntime.__version__}")
        return True
    except Exception as fehler:                        # pragma: no cover - nur auf Kaggle
        p.zeile(f"[onnxruntime] weiterhin nicht ladbar: {fehler}")
        return False


def alt_bereitstellen(p: Protokoll, roh: Path) -> str:
    """Den Vergleichs-Checkpoint bereitstellen - den veroeffentlichten Block-4-Stand.

    Wozu: die Messlatte ist zwischen Block 4 und Block 5 **nicht** dieselbe. Die 200
    synthetischen val-Bilder entstehen mit dem neuen Tafel-Szenario (3-34 Schilder je Bild)
    und tragen viel mehr Boxen - 2 509 gegenueber 3 763 im Grundwert. Zwei Laeufe auf
    verschiedenen Messlatten zu vergleichen waere wertlos, deshalb wird der ALTE Checkpoint
    auf DERSELBEN (neuen) Messlatte mitgemessen. Fehlt die Datei, entfaellt die Gegenprobe.
    """
    ziel, quelle = WORK / "alt" / "vergleich.pt", roh / "alt" / "vergleich.pt"
    if PROBE:
        p.zeile(f"[probe] Vergleichs-Checkpoint {quelle} -> {ziel}")
        return "alt/vergleich.pt"
    if not quelle.exists():
        p.zeile(f"[hinweis] kein Vergleichs-Checkpoint ({quelle} fehlt) - die Gegenprobe "
                "alter/neuer Checkpoint auf derselben Messlatte entfaellt")
        return ""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(quelle, ziel)
    p.zeile(f"[kopiert] {quelle.name} -> alt/vergleich.pt ({ziel.stat().st_size/1e6:.1f} MB)")
    return "alt/vergleich.pt"


def einrichten(p: Protokoll, roh: Path) -> dict:
    """Arbeitsverzeichnis herrichten: Code bereitstellen, Rohdaten normalisieren.

    Der Kaggle-Eingang ist schreibgeschuetzt und liegt unter einem anderen Pfad als in der
    Doku - deshalb wird kopiert statt gearbeitet. Die Werkzeuge werden relativ aufgerufen
    (tools/...), damit der Aufruf mit docs/TRAINING.md §4.3/§5 Buchstabe fuer Buchstabe
    zusammenfaellt. Die Rohdaten kommen je nach Kaggle-Verhalten als ZIP oder schon entpackt
    an - in Form bringt sie archiv_bereitstellen() bzw. coco_bereitstellen().
    """
    tools = WORK / "tools"
    if tools.exists():
        shutil.rmtree(tools)
    if (roh / "tools").exists():
        shutil.copytree(roh / "tools", tools,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif (roh / "tools.zip").exists():
        entpacken(p, roh / "tools.zip", WORK, "Werkzeuge")
    else:
        raise SystemExit(f"kein tools/ und kein tools.zip in {roh}")
    p.zeile(f"[kopiert] Werkzeuge -> {tools}")

    gtsrb_roh = roh / "gtsrb"
    (WORK / "gtsrb").mkdir(parents=True, exist_ok=True)
    archiv_bereitstellen(p, gtsrb_roh, "gtsrb-train", "train.zip", "GTSRB Trainingsbilder")
    archiv_bereitstellen(p, gtsrb_roh, "gtsrb-test", "test.zip", "GTSRB Testbilder")
    # Die Labels des Test-Sets liegen in einem eigenen Archiv (GT-final_test.csv).
    archiv_bereitstellen(p, gtsrb_roh, "gtsrb-test-gt", "test-gt.zip", "GTSRB Test-Labels")

    coco = coco_bereitstellen(p, roh)

    nutzer = WORK / "user"
    nutzer.mkdir(parents=True, exist_ok=True)
    foto = roh / "user" / "Nothing.jpg"
    if not foto.exists():
        raise SystemExit(f"{foto} fehlt im Rohdaten-Datensatz")
    shutil.copy2(foto, nutzer / "Nothing.jpg")
    p.zeile(f"[kopiert] {foto.name} -> user/Nothing.jpg (eigenes Foto ohne Schild)")

    (WORK / "models").mkdir(exist_ok=True)
    (WORK / "berichte").mkdir(exist_ok=True)
    return {"tools": str(tools), "gtsrb": str(WORK / "gtsrb"), "coco128": coco,
            "foto": str(nutzer / "Nothing.jpg"), "alt": alt_bereitstellen(p, roh)}



def datensatz_bauen(p: Protokoll) -> dict:
    """Den Detektor-Datensatz bauen - dieselben acht Aufrufe wie docs/TRAINING.md §4.3/§4.3+.

    Reihenfolge und Zahlen sind der Block-5-Stand aus PLAN.md §2: 29 600 Trainingsbilder
    (20 000 GTSRB + 3 000 Lueckenschluss + 3 000 synthetische + 3 600 echte Negative),
    Messlatte 2 000 (val) und 1 100 (neg). Die Werkzeuge haengen sich an einen BESTEHENDEN
    Ordner (manifest.json), deshalb ist die Reihenfolge bindend: erst GTSRB, dann alles
    Synthetische, dann die echten Negative.
    """
    d, t = DATEN, TRAINING
    dauer: dict[str, float] = {}
    schritte = [
        ("gtsrb train", python("tools/gtsrb_dataset.py", "--zip", "gtsrb/train.zip",
                               "--out", "data/det", "--n", str(d["n_train"]),
                               "--size", str(d["size"]), "--seed", "0", "--balance")),
        ("gtsrb val", python("tools/gtsrb_dataset.py", "--zip", "gtsrb/test.zip",
                             "--gt-zip", "gtsrb/test-gt.zip", "--out", "data/det",
                             "--n", str(d["n_val"]), "--size", str(d["size"]),
                             "--split", "val", "--seed", "1")),
        ("lueckenschluss train", python("tools/synth_missing.py", "--out", "data/det",
                                        "--n", str(d["n_fehlend"]))),
        ("lueckenschluss val", python("tools/synth_missing.py", "--out", "data/det",
                                      "--n", str(d["n_fehlend_val"]), "--split", "val")),
        ("synthetische negative", python("tools/synth_negatives.py", "--out", "data/det",
                                         "--n", str(d["n_neg_train"]), "--split", "train")),
        ("synthetische negative (Messlatte)", python("tools/synth_negatives.py", "--out", "data/det",
                                                     "--n", str(d["n_neg_neg"]), "--split", "neg")),
        # Echte Negative: Trennung ohne Selbstbetrug - vom eigenen Foto nur die obere
        # Haelfte, die untere bleibt fuer den Split neg gesperrt (PLAN.md §2).
        ("echte negative", python("tools/real_negatives.py", "--out", "data/det",
                                  "--n", str(d["n_echt_train"]), "--split", "train",
                                  "--real-train", "user/Nothing.jpg",
                                  "--train-region", "0,0,1,0.5",
                                  "--coco", "negatives/coco128",
                                  "--coco-test-n", str(d["coco_test_n"]),
                                  "--user-share", str(d["user_share_train"]))),
        ("echte negative (Messlatte)", python("tools/real_negatives.py", "--out", "data/det",
                                              "--n", str(d["n_echt_neg"]), "--split", "neg",
                                              "--real-test", "user/Nothing.jpg",
                                              "--test-region", "0,0.5,1,1",
                                              "--coco", "negatives/coco128",
                                              "--coco-test-n", str(d["coco_test_n"]),
                                              "--user-share", str(d["user_share_neg"]))),
    ]
    for was, cmd in schritte:
        dauer[was] = lauf(p, cmd, f"Datensatz: {was}")
    return dauer


def zahlen_pruefen(p: Protokoll) -> dict:
    """Die Bildzahlen des Manifests gegen die Erwartung stellen.

    Ein Tippfehler im Rezept faellt sonst erst nach zwei Stunden Training auf - und dann
    sieht man nur eine schlechtere Zahl, nicht die Ursache. Deshalb wird hier hart geprueft.
    """
    manifest = json.loads((WORK / "data" / "det" / "manifest.json").read_text(encoding="utf-8"))
    zahlen: dict[str, int] = {}
    for e in manifest["images"]:
        split = e.get("split", "train")
        zahlen[split] = zahlen.get(split, 0) + 1
    p.zeile(f"[daten] {len(manifest['images'])} Bilder im Manifest: "
            + ", ".join(f"{k}={v}" for k, v in sorted(zahlen.items())))
    falsch = {k: (zahlen.get(k, 0), v) for k, v in ERWARTET.items() if zahlen.get(k, 0) != v}
    if falsch:
        p.zeile("[daten] ERWARTUNG TRIFFT NICHT ZU: "
                + ", ".join(f"{k}: {ist} statt {soll}" for k, (ist, soll) in falsch.items()))
        raise SystemExit("Datensatz stimmt nicht mit PLAN.md (Block 5) ueberein")
    p.zeile("[daten] Bildzahlen stimmen mit PLAN.md (Block 5) ueberein")
    return zahlen


def trainieren(p: Protokoll) -> float:
    """Das Netz trainieren - Aufruf wie docs/TRAINING.md §5, Wort fuer Wort."""
    t = TRAINING
    cmd = python("tools/train_det.py", "--data", "data/det", "--size", str(t["size"]),
                 "--preset", t["preset"], "--batch", str(t["batch"]),
                 "--epochs", str(t["epochs"]), "--steps", str(t["steps"]),
                 "--lr", str(t["lr"]), "--degrade", str(t["degrade"]), "--zoom", str(t["zoom"]),
                 "--workers", str(t["workers"]), "--eval-every", str(t["eval_every"]),
                 "--save-every", str(t["save_every"]), "--obj-norm", t["obj_norm"],
                 "--val-split", t["val_split"], "--out", "models/signs-det.pt",
                 "--seed", str(t["seed"]))
    return lauf(p, cmd, f"Training: {t['epochs']} Epochen x {t['steps']} Schritte "
                        f"x {t['batch']} Bilder (Merksatz: 1 Epoche = {t['steps']*t['batch']} Bilder)")


def exportieren(p: Protokoll) -> float:
    """ONNX-Export mit dynamischen Achsen: EINE Datei fuer 256/320/384/448 px.

    Der Export prueft sich selbst (Paritaet gegen PyTorch) und schreibt labels.json +
    manifest.json dazu - genau die drei Dateien, die src/model.js erwartet. fatal=False:
    fehlt onnxruntime, soll der Lauf trotzdem seine Messwerte abliefern.
    """
    return lauf(p, python("tools/export_onnx.py", "--ckpt", "models/signs-det.pt",
                          "--out", "models/signs-det.onnx", "--dynamic"),
                "Export nach ONNX (dynamische Hoehe/Breite)", fatal=False)


def auswerten(p: Protokoll, ein: dict, mit_fixture: bool = True) -> dict:
    """Die Zahlen holen, die ueber die Qualitaet entscheiden - plus die Fixture fuer JS.

    val      = Messlatte (2 000 Bilder, unausgeglichen) - das ist die Zahl, die mit den
               veroeffentlichten Werten verglichen wird.
    neg      = Bilder ohne Schild: dort ist fp/Bild die Kennzahl (weniger ist besser).
    384 px   = derselbe Checkpoint bei anderer Eingabegroesse (Fotomodus der App).
    Vergleich = der ALTE Checkpoint (Block 4) auf derselben Messlatte. Noetig, weil die
               Messlatte sich geaendert hat (siehe alt_bereitstellen): nur dieser Vergleich
               sagt, was der neue Trainingslauf wirklich gebracht hat.
    Fixture  = rohe ONNX-Ausgaben fuer tests/model.test.js (braucht onnxruntime).

    Alle Schritte laufen mit fatal=False: ein misslungener Nebenschritt soll die 45 Minuten
    Training nicht entwerten.
    """
    a = AUSWERTUNG
    gemein = ["--data", "data/det", "--conf", str(a["conf"]), "--iou", str(a["iou"]),
              "--workers", "4"]
    schritte = [
        ("Messlatte val", "tools/eval_conditions.py",
         ["--ckpt", "models/signs-det.pt", *gemein, "--split", "val", "--diagnose",
          "--json", "berichte/val.json"]),
        ("Gegenprobe neg", "tools/eval_conditions.py",
         ["--ckpt", "models/signs-det.pt", *gemein, "--split", "neg",
          "--json", "berichte/neg.json"]),
        (f"Messlatte {a['zweite_groesse']} px", "tools/eval_conditions.py",
         ["--ckpt", "models/signs-det.pt", *gemein, "--split", "val",
          "--size", str(a["zweite_groesse"]), "--json", "berichte/val384.json"]),
    ]
    if ein.get("alt"):
        schritte.append(("Vergleich alter Checkpoint (dieselbe Messlatte)",
                         "tools/eval_conditions.py",
                         ["--ckpt", ein["alt"], *gemein, "--split", "val",
                          "--json", "berichte/val_block4.json"]))
    if mit_fixture:
        schritte.append(("Fixture fuer die JS-Tests", "tools/export_fixture.py",
                         ["--ckpt", "models/signs-det.pt", "--onnx", "models/signs-det.onnx",
                          "--out", "berichte/model-out.json"]))
    dauer: dict[str, float] = {}
    for was, werkzeug, args in schritte:
        dauer[was] = lauf(p, python(werkzeug, *args), f"Auswertung: {was}", fatal=False)
    return dauer


def packen(p: Protokoll) -> dict:
    """Den gebauten Datensatz als EIN Archiv ablegen.

    Warum ueberhaupt: Kaggle reicht die Ausgabe eines Kernels als Dateiliste weiter - 32 700
    Bilder waeren 65 400 Einzeldateien. Als ein ZIP sind es eine Datei von rund 400 MB; sie
    dient zugleich als Vorlage, um spaeter einen Kaggle-Datensatz daraus zu bauen und sich
    den Aufbau (etwa 20 Minuten je Lauf) zu sparen.
    """
    quelle, ziel = WORK / "data" / "det", WORK / "data" / "det.zip"
    if PROBE:
        p.zeile(f"[probe] packen: {quelle} -> {ziel}")
        return {}
    t0, anzahl = time.perf_counter(), 0
    with zipfile.ZipFile(ziel, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for datei in sorted(quelle.rglob("*")):
            if not datei.is_file():
                continue
            # JPEG laesst sich nicht weiter verkleinern - Speichern ist schneller und
            # genauso klein; die Labels (Text) dagegen schon.
            art = zipfile.ZIP_STORED if datei.suffix.lower() in (".jpg", ".jpeg", ".png") \
                else zipfile.ZIP_DEFLATED
            z.write(datei, datei.relative_to(WORK).as_posix(), compress_type=art)
            anzahl += 1
    mb = ziel.stat().st_size / 1e6
    p.zeile(f"[gepackt] {anzahl} Dateien -> data/det.zip ({mb:.0f} MB, {time.perf_counter()-t0:.0f} s)")
    if AUFRAEUMEN:
        # Nur der Datensatz bleibt (als Archiv); die Zwischenstaende wuerden sonst als
        # Kernelausgabe mitgeschleppt - mehrere hundert MB und zehntausende Dateien.
        for name in ("data/det", "gtsrb", "negatives", "user"):
            ordner = WORK / name
            if ordner.exists():
                shutil.rmtree(ordner)
        p.zeile("[aufgeraeumt] data/det/, gtsrb/, negatives/, user/ entfernt - det.zip bleibt")
    return {"zip_mb": round(mb, 1), "dateien": anzahl}


def trainingskurve() -> list[dict]:
    """Die gemessenen val-Punkte aus dem Checkpoint - der Verlauf ist der eigentliche Beleg.

    Ohne diese Liste sieht man nur die letzte Zahl; erst die Kurve zeigt, ob ein Lauf noch
    gestiegen waere (docs/TRAINING.md §5.1) oder ob er schon platt ist.
    """
    if PROBE:
        return []
    import torch
    ck = torch.load(WORK / "models" / "signs-det.pt", map_location="cpu", weights_only=False)
    return [{"epoche": h["epoch"], "precision": h.get("precision"), "recall": h.get("recall"),
             "f1": h.get("f1"), "tp": h.get("tp"), "fp": h.get("fp"), "fn": h.get("fn")}
            for h in ck.get("history", []) if h.get("eval") and h.get("f1") is not None]


def kurzfassung(p: Protokoll, geraetname: str, ein: dict, dauer: dict, zahlen: dict,
                archiv: dict) -> dict:
    """kaggle_report.json: Rezept, Dauer, Kennzahlen und Wege an einer Stelle.

    Die abgelegten Auswertungen (berichte/*.json) sind die Wahrheit; diese Datei ist die
    erste Seite davon - damit man nach dem Herunterladen nicht in fuenf Dateien suchen muss.
    """
    def lade(pfad: Path):
        return json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else None

    berichte = {name: lade(WORK / "berichte" / name)
                for name in ("val.json", "neg.json", "val384.json", "val_block4.json",
                             "model-out.json")}
    dateien = {f.name: f.stat().st_size for f in sorted((WORK / "models").glob("*")) if f.is_file()}
    report = {
        "stand": time.strftime("%Y-%m-%d %H:%M"),
        "geraet": geraetname,
        "eingang": ein,
        "rezept": {"daten": DATEN, "training": TRAINING, "auswertung": AUSWERTUNG},
        "bilder": zahlen,
        "archiv": archiv,
        "dauer_min": {k: round(v / 60, 1) for k, v in dauer.items()},
        "kennzahlen": {k: (v or {}).get("alle") for k, v in berichte.items() if v},
        "modelldateien": dateien,
        "fehlgeschlagen": FEHLER,
        "trainingskurve": trainingskurve(),
    }
    (WORK / "kaggle_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    p.zeile("[bericht] kaggle_report.json geschrieben")
    return report


def main() -> None:
    t0 = time.perf_counter()
    p = Protokoll(WORK / "berichte" / "protokoll.txt")
    p.zeile(f"[start] Schilder-Detektor, Kaggle-Lauf {time.strftime('%Y-%m-%d %H:%M:%S')}"
            + ("   (PROBE: es wird nichts gerechnet)" if PROBE else ""))
    geraetename = geraet(p)
    # Zuerst onnxruntime sicherstellen: es entscheidet ueber Export, Paritaetsprobe und Fixture.
    # Ohne diese Vorbereitung riss der erste Lauf die gesamte Ausgabe mit (43 min Training weg).
    hat_ort = onnxruntime_sichern(p)
    roh = eingang_finden()
    p.zeile(f"[eingang] Rohdaten-Datensatz: {roh}")

    dauer: dict[str, float] = {"einrichten": 0.0}
    ein = einrichten(p, roh)
    dauer.update(datensatz_bauen(p))
    zahlen = {} if PROBE else zahlen_pruefen(p)
    dauer["training"] = trainieren(p)

    if hat_ort:
        dauer["export"] = exportieren(p)
    else:
        FEHLER.append("Export (onnxruntime fehlt)")
        p.zeile("[warnung] ONNX-Export uebersprungen - ohne onnxruntime keine Paritaetsprobe "
                "und damit kein labels.json/manifest.json")
    dauer.update(auswerten(p, ein, mit_fixture=hat_ort))
    archiv = packen(p)
    report = kurzfassung(p, geraetename, ein, dauer, zahlen, archiv)

    gesamt = (time.perf_counter() - t0) / 60
    p.zeile()
    p.zeile(f"[fertig] Gesamtdauer {gesamt:.0f} min auf {geraetename}")
    for name, blatt in (("val (Messlatte)", "val.json"), ("neg (Fehlalarme)", "neg.json"),
                        ("val 384 px", "val384.json"),
                        ("Vergleich Block 4", "val_block4.json")):
        daten = report["kennzahlen"].get(blatt)
        if daten:
            p.zeile(f"[ergebnis] {name:18s} P={daten['precision']:.3f} R={daten['recall']:.3f} "
                    f"F1={daten['f1']:.3f}  tp={daten['tp']} fp={daten['fp']} fn={daten['fn']}")
    p.zeile(f"[ergebnis] Modell: models/signs-det.onnx "
            f"{report['modelldateien'].get('signs-det.onnx', 0)/1e6:.2f} MB, "
            f"Checkpoint {report['modelldateien'].get('signs-det.pt', 0)/1e6:.1f} MB")
    if FEHLER:
        p.zeile()
        p.zeile("[warnung] nicht gelaufen: " + ", ".join(FEHLER))
        p.zeile("[warnung] die uebrigen Ergebnisse liegen in /kaggle/working "
                "(Checkpoint, Datensatz, Messwerte, Protokoll)")
    p.datei.close()


if __name__ == "__main__":
    main()

