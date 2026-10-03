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
    # Klassen: 74 statt 9 (tools/signmap.py). Quellen: GTSRB + GTSIGN-220 + Synset Signset
    # Germany + Open Images. --weight gleicht die Quellen an: GTSIGN und Synset sind kleiner,
    # aber sauberer annotiert, und ohne den Faktor wuerde die Menge der GTSRB-Bilder sie
    # erdruecken (siehe tools/crops_dataset.py).
    n_train=30000,
    n_val=2500,
    n_neg=1000,               # echte Negative (Open-Images-Fotos ohne Verkehrszeichen)
    n_neg_synth=500,          # dazu synthetische Negative als Gegenprobe
    n_echt_train=3600,        # echte Fotos ohne Schild aus Block 5 (coco128 + eigenes Foto)
    n_eigene_train=1800,      # die selbst gesammelten Negative des Nutzers (70 % der Fotos)
    n_eigene_neg=400,         # deren gesperrte 30 % als Messlatte
    eigene_share=0.75,        # davon 75 % eigene Fotos, 25 % coco128 als Streuung
    coco_test_n=12,           # die letzten 12 coco-Fotos bleiben fuer die Messlatte gesperrt
    user_share_train=0.55,    # Anteil des eigenen Fotos, nur die obere Haelfte
    size=384,                 # TRAININGS- und Datensatzgroesse (vorher 320). Warum:
                              # der Median der verpassten Objekte lag bei 35 px Bilddiagonale -
                              # bei 320 px Eingang ist ein 35-px-Schild nach dem Downsampling
                              # auf stride 32 nur noch EIN Pixel breit. Bei 384 px sind es
                              # 1,2 - und die feinste Stufe (stride 4) sieht viermal so viele
                              # Bildpunkte. Datensatz und Training MUESSEN dieselbe Zahl
                              # benutzen: crops_dataset schreibt die Bilder in dieser Groesse,
                              # train_det letterboxt darauf (--size).
                              # Achtung Werkzeuge: tools/crops_dataset.py --size 384,
                              # tools/gtsdb_dataset.py --size 384 (beide aus diesem Wert).
    weights=("gtsign=2", "synset=2"),
    gtsign_train_max=24000,   # Ausschnitte aus dem GTSIGN-Trainingssplit
    gtsign_val_max=6000,      # Ausschnitte aus dem GTSIGN-Validierungssplit (fremde Messlatte)
    synset_train_max=12000,   # Synset-Ausschnitte (Trainingssplit der Karte)
    synset_val_max=3000,      # Synset-Ausschnitte (Validierungssplit der Karte)
    oi_max=4000,              # Open-Images-Fotos ohne Verkehrszeichen
    neg_share_train=0.12,     # Anteil echter Fotos ohne Schild im Training
    neg_share_val=0.20,       # derselbe Anteil in der Messlatte
)
TRAINING = dict(
    # "breit" statt "balanced": 74 Klassen entscheiden sich im Kopf, deshalb dort mehr
    # Kapazitaet (mid = Rumpfbreite) und ein Block mehr in den tiefen Stufen.
    # Gemessen nach dem CATM/LGP-FPN-Umbau (tools/bench_model.py --preset):
    #   2,20 Mio. Parameter / 1 057 MFLOPs gegen 1,88 Mio. / 1 180 vorher.
    # Also MEHR Parameter, aber 10 % WENIGER Rechnung - CATM kommt ohne das N x N-Feld der
    # Selbstattention aus, und die Fensteraufteilung auf p4 ist ersatzlos entfallen. Neu dazu
    # kommt CATM auf p3 (dort ist die Rechnung pro Zelle konstant).
    preset="breit",
    size=384,             # siehe DATEN["size"] - beide MUESSEN gleich sein
    batch=16,
    epochs=220,
    steps=150,            # 150 x 16 = 2 400 Bilder je Epoche
    lr=1.5e-3,
    # Augmentierung ZURUECKGENOMMEN (vorher degrade 0.6, zoom 0.5):
    #  * Zoom verstärkt genau das Problem, das er loesen soll: ein aggressiver Skalenschnitt
    #    schrumpft kleine Schilder weiter, statt sie dem Netz naeher zu bringen. Gemessen
    #    (data/_fp_messung.py, 6x6-Kacheln auf dem Poster) kann das Netz 15-px-Schilder
    #    erkennen - es sieht sie nur im Trainingsbild zu selten in brauchbarer Groesse.
    #  * Dieselbe Ueberlegung fuer degrade: starke Stoerung auf einem 20-px-Schild loescht
    #    das Symbol, nicht nur dessen Kontrast. Was uebrig bleibt, ist Rauschen mit einer
    #    Box daran - und genau das erzeugt Fehlalarme.
    # Der Rest der Augmentierung (Zoom-Bereich 0.7..1.5, Helligkeit/Kontrast in synth_data)
    # bleibt: die Zielbedingungen (Nacht, Regen, Bewegung) muessen weiter abgedeckt sein.
    degrade=0.4,
    zoom=0.3,
    workers=4,            # Kaggle hat 4 vCPU
    eval_every=10,
    save_every=10,
    seed=7,
    obj_norm="pos",
    val_split="val",
    # Verlust und Kopf (siehe tools/train_det.py):
    #  * focal_gamma 2 / focal_alpha 0.25: Focal Loss im Klassifikationskopf. 98 % aller
    #    Fehlalarme waren echte Schilder mit FALSCHER Klasse - der Kopf findet, entscheidet
    #    aber falsch, und die Kreuzentropie gewichtet leichte und schwere Faelle gleich.
    #  * cls_w 4.0 gleicht das alpha=0.25 wieder aus: der Klassifikationskopf rechnet nur auf
    #    positiven Zellen, dort ist alpha ein KONSTANTER Faktor und wuerde den Kopf sonst
    #    still auf ein Viertel drosseln (4.0 x 0.25 = 1.0 wie vorher).
    #  * smooth 0.075: Label-Smoothing haelt die Logits endlich und bremst die Ueberzeugung
    #    bei aehnlichen Zeichen (Verwechslungsmatrix: Geschwister waren die Hauptfehler).
    #  * hier_aux 0.3: die FAMILIE (9 Oberkategorien) wird direkt ueberwacht. Der
    #    hierarchische Kopf bekaeme sonst nur mittelbar ein Signal.
    focal_gamma=2.0,
    focal_alpha=0.25,
    cls_w=4.0,
    smooth=0.075,
    hier_aux=0.3,
)
# Auswertung: conf und NMS werden NICHT mehr fest angenommen. Der Sweep (--sweep) sucht das
# beste Paar nach F1 ueber 0,15..0,60 (conf) x 0,40..0,70 (NMS-IoU) und kostet kein
# Neutraining - das Netz laeuft einmal, die Nachbearbeitung danach beliebig oft. Die hier
# eingetragenen Werte bleiben als Startwert der uebrigen Schritte stehen (und als
# Vergleichswert im Bericht); der Sweep schreibt sein Optimum nach berichte/sweep.json.
AUSWERTUNG = dict(conf=0.25, iou=0.5, iou_det=0.45, zweite_groesse=448,
                  sweep_limit=400)
# int8-Quantisierung: QDQ mit Kalibrierung auf den ECHTEN Bildern des Datensatzes - die
# Verteilung des Einsatzes entscheidet ueber die Skalen, nicht eine synthetische. 200 Bilder
# sind gemessen ausreichend; mehr kostet nur Zeit. Das Werkzeug verwirft die int8-Datei
# SELBST, wenn die Paritaet gegen PyTorch zu schlecht ist - dann bleibt fp32 aktiv und
# src/model.js faellt ohnehin darauf zurueck.
QUANT = dict(n_calib=200, calib_data="data/det")
AUFRAEUMEN = True         # Bildordner nach dem Zippen loeschen (sonst 65 000 Ausgabedateien)
SYNSET_KONFIG = "Cycles"  # die Synset-Fassung mit Pfadverfolgung (GTSRB-Zwilling inklusive)
SYNSET_REPO = "FraunhoferIOSB/Synset-Signset-Germany"   # wird gestreamt, nicht hochgeladen
# Werkzeuge und Kataloge kommen aus dem Netz (enable_internet ist an). Damit muss bei einer
# Code-Aenderung NICHT der 700-MB-Datensatz neu hochgeladen werden - nur dieses Skript
# (50 KB). kaggle/run.ps1 -Step push setzt WERKZEUGE_SHA auf den aktuellen Commit.
REPO_SLUG = "ceyale/Stra-enschilderkennen"
WERKZEUGE_SHA = "4ccf69edc8f81bb55eb1ac14696de9d24ff14fc9"
HUGGING = "https://huggingface.co/datasets/miriamcarnot/GTSIGN-220/resolve/main/"
# Der veroeffentlichte Vergleichs-Checkpoint hat 9 Klassen, dieser Lauf trainiert 74
# (tools/signmap.py). Die Gegenprobe "alter gegen neuer Checkpoint auf derselben Messlatte"
# ist damit nicht moeglich - das alte Netz kann die neuen Klassen nicht ausgeben. Erst ein
# naechster Lauf auf derselben Taxonomie kann diesen Vergleich wieder fuehren.
ALT_VERGLEICH = False
# Erwartete Bildzahlen stehen absichtlich NICHT hier: datensatz_bauen rechnet sie aus dem
# Rezept (siehe dort). Wahlquellen duerfen fehlen, ohne dass die Pruefung den Lauf beendet.
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
    kandidaten: list[Path] = []
    for tiefe in range(1, 5):
        for kandidat in sorted(EINGANG.glob("/".join(["*"] * tiefe))):
            if not kandidat.is_dir():
                continue
            gesehen.append(kandidat)
            if (kandidat / "tools" / "train_det.py").exists() or (kandidat / "tools.zip").exists():
                kandidaten.append(kandidat)
    # Ein Ordner mit `gtsrb` oder `kataloge` ist eindeutig der Datensatz. Ohne diese
    # Bevorzugung kann ein Arbeitsordner gewinnen, in den nur die Werkzeuge kopiert wurden -
    # gemessen bei der lokalen Probe, wo der Probe-Ordner alphabetisch vor dem Datensatz lag.
    for kandidat in kandidaten:
        if (kandidat / "gtsrb").exists() or (kandidat / "kataloge").exists():
            return kandidat
    if kandidaten:
        return kandidaten[0]
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
    # Kein harter Abbruch: coco128 ist nur Streuung neben den eigenen Fotos. Fehlt es, bauen
    # die Negative-Aufrufe eben nur aus den eigenen Aufnahmen - besser als ein Lauf, der
    # daran scheitert (tools/real_negatives.py behandelt --coco ohnehin als Wahl).
    p.zeile("[hinweis] coco128 nicht gefunden (weder Ordner noch ZIP) - die Negative kommen "
            "dann allein aus den eigenen Fotos")
    return ""


def datasets_sichern(p: Protokoll) -> bool:
    """`datasets` (HuggingFace) bereitstellen - es traegt das Synset-Streaming.

    Dieselbe Lehre wie bei onnxruntime: was der Kernel nicht mitbringt, muss er sich selbst
    holen. Ohne `datasets` faellt die Synset-Quelle aus (die anderen drei bleiben), aber ein
    Lauf, der eine ganze Quelle verliert, ist es nicht wert - der Aufruf kostet Sekunden.
    """
    try:
        import datasets  # noqa: F401
        p.zeile("[lib] datasets ist vorhanden")
        return True
    except ImportError:
        pass
    if PROBE:
        p.zeile("[probe] pip install datasets (waere noetig)")
        return True
    p.zeile("[lib] datasets fehlt - pip install datasets (HuggingFace-Streaming fuer Synset)")
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "pyarrow"],
                        capture_output=True, text=True)
    if rc.returncode != 0:
        p.zeile(f"[lib] Installation fehlgeschlagen: {rc.stderr.strip()[-300:]}")
        p.zeile("[warnung] Synset Signset Germany wird uebersprungen")
        return False
    p.zeile("[lib] datasets installiert")
    return True


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


def _katalog_split(pfad: Path, split: str) -> list[str]:
    """Eine Split-Liste eines Katalogs lesen (Kopfzeile wird erkannt und uebersprungen)."""
    zeilen = [z.strip() for z in pfad.read_text(encoding="utf-8").splitlines() if z.strip()]
    if zeilen and zeilen[0].lower().startswith("image_path"):
        zeilen = zeilen[1:]
    if not zeilen:
        raise SystemExit(f"{pfad}: leerer Split '{split}'")
    return zeilen


def _stride_plan(gesamt: int, max_bilder: int) -> tuple[int, int]:
    """Wie oft muss man springen, um aus `gesamt` hoechstens `max_bilder` zu bekommen?

    Der Sprung ist Absicht: eine einfache Grenze ("die ersten 12 000") wuerde bei einer
    nach Klassen sortierten Liste nur die ersten Klassen liefern. Gleichmaessiges Springer
    trifft dagegen jede Klasse - auch bei Kleinstklassen mit wenigen Bildern.
    """
    if max_bilder <= 0 or max_bilder >= gesamt:
        return 1, gesamt
    schritt = max(1, gesamt // max_bilder)
    return schritt, min(gesamt, len(range(0, gesamt, schritt)))


def feste_ausschnitte(p: Protokoll, zip_pfad: Path, csv_pfad: Path, split_datei: Path,
                      ziel: Path, max_bilder: int, lang: int = 384) -> int:
    """Schildausschnitte EINES Splits als <ziel>/<label>/*.jpg bereitstellen.

    Wozu ueberhaupt entpacken: tools/crops_dataset.py komponiert Bilder, und aus einem ZIP
    heraus ist das langsam (jedes Bild einzeln entpacken). Einmal entpackt kostet es
    Sekunden - dafuer setzt die Trennung von Training und Messlatte auf der SPLIT-LISTE des
    Katalogs auf. Nur so besteht die Messlatte aus Bildern, die im Training nicht vorkommen
    (PLAN.md: "getrennte Fotos, sonst betruegt man sich selbst").

    Verkleinert wird auf `lang` px: das Modell sieht 320 px, ein 384-px-Ausschnitt verliert
    dabei nichts Sichtbares und spart beim Schreiben Stunden.
    """
    if PROBE:
        p.zeile(f"[probe] Ausschnitte {zip_pfad.name} <- {split_datei.name} -> {ziel} "
                f"(max {max_bilder})")
        return 0
    import csv as _csv
    import zipfile

    from PIL import Image

    sys.path.insert(0, str(WORK / "tools"))
    import signmap

    tab: dict[str, str] = {}
    for r in _csv.DictReader(csv_pfad.open(encoding="utf-8")):
        label = signmap.label_for_stvo(r["StVO_Sign_Number"])
        if label:
            tab[f"{int(r['Class_ID']):03d}"] = label
    if not tab:
        raise SystemExit(f"{csv_pfad}: keine zuordenbare Klasse gefunden")

    gewuenscht = {z.replace("\\", "/") for z in _katalog_split(split_datei, "split")}
    ziel.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_pfad) as z:
        paare: list[tuple[str, str, str]] = []      # (Zip-Eintrag, Klasse, Label)
        for name in z.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            teile = name.split("/")
            if len(teile) < 3:
                continue
            klasse = teile[-2]
            rel = "/".join(teile[-2:])
            if rel not in gewuenscht or klasse not in tab:
                continue
            paare.append((name, klasse, tab[klasse]))
        paare.sort()
        schritt, erwartet = _stride_plan(len(paare), max_bilder)
        p.zeile(f"[ausschnitte] {ziel.name}: {len(paare)} Bilder im Split, "
                f"jedes {schritt}. -> {erwartet}")
        geschrieben = 0
        for name, klasse, label in paare[::schritt]:
            ordner = ziel / label
            ordner.mkdir(exist_ok=True)
            with z.open(name) as f:
                bild = Image.open(f).convert("RGB")
            if max(bild.size) != lang:
                bild = bild.resize((lang, lang), Image.BILINEAR)
            bild.save(ordner / f"{geschrieben:07d}.jpg", quality=88)
            geschrieben += 1
            if geschrieben % 5000 == 0:
                p.zeile(f"[ausschnitte]   {geschrieben}/{erwartet}")
    leer = [d.name for d in ziel.iterdir() if d.is_dir() and not any(d.iterdir())]
    if leer:
        p.zeile(f"[ausschnitte] ohne Bild geblieben: {', '.join(leer)}")
    return geschrieben


def synset_ausschnitte(p: Protokoll, roh: Path, ziel: Path, konfig: str, split: str,
                       max_bilder: int, lang: int = 384) -> int:
    """Synset Signset Germany in <ziel>/<label>/*.jpg umwandeln (nur der gewaehlte Split).

    STREAMEND gelesen: die Parquet-Datei enthaelt 105 500 Bilder in EINER Datei - sie
    vollstaendig zu laden sprengt den Arbeitsspeicher der Kaggle-Maschine. Der Split kommt
    aus der Karte des Datensatzes (train/validation), damit Trainings- und Messbilder nicht
    dasselbe Zeichen zeigen. Beim Schreiben wird verkleinert (siehe feste_ausschnitte).

    Woher gelesen wird: liegt im Rohdaten-Datensatz ein Ordner `synset/` (die HuggingFace-
    Ablage des Datensatzes), wird der genommen - dann braucht der Lauf kein Netz. Sonst
    wird direkt von HuggingFace gestreamt: 17,6 GB in den Kaggle-Datensatz zu legen waere
    weder beim Hochladen noch beim Speicher vertretbar, und gestreamt werden nur die
    Zeilengruppen gelesen, die man wirklich braucht.
    """
    if PROBE:
        p.zeile(f"[probe] Synset {konfig}/{split} -> {ziel} (max {max_bilder})")
        return 0
    lokal = roh / "synset"
    quelle = str(lokal) if lokal.exists() else SYNSET_REPO
    p.zeile(f"[synset] Quelle: {quelle}")
    try:
        from datasets import load_dataset
    except ImportError as fehler:
        p.zeile(f"[synset] 'datasets' fehlt ({fehler}) - Quelle entfaellt")
        return 0
    from PIL import Image

    sys.path.insert(0, str(WORK / "tools"))
    import signmap

    p.zeile(f"[synset] Durchlauf 1/2: Klassenverzeichnis von {konfig}/{split}")
    try:
        kopf = load_dataset(quelle, konfig, split=split, streaming=True)
        zuordnung: dict[int, str] = {}
        gesamt = 0
        for r in kopf.select_columns(["label", "class_name"]):
            l = int(r["label"])
            zuordnung.setdefault(l, str(r["class_name"]))
            gesamt += 1
    except Exception as fehler:
        p.zeile(f"[synset] nicht lesbar ({type(fehler).__name__}: {fehler}) - Quelle entfaellt")
        return 0
    if not gesamt:
        p.zeile("[synset] Split leer")
        return 0
    schritt, erwartet = _stride_plan(gesamt, max_bilder)
    p.zeile(f"[synset] {gesamt} Bilder, {len(zuordnung)} Klassen; Durchlauf 2/2: "
            f"jedes {schritt}. Bild (gleichmaessig ueber alle Klassen)")
    ziel.mkdir(parents=True, exist_ok=True)
    daten = load_dataset(str(quelle), konfig, split=split, streaming=True)
    geschrieben, unbekannt = 0, 0
    for i, r in enumerate(daten):
        if i % schritt or geschrieben >= erwartet:
            continue
        label = signmap.label_for_synset(zuordnung.get(int(r["label"]), ""))
        if not label:
            unbekannt += 1
            continue
        ordner = ziel / label
        ordner.mkdir(exist_ok=True)
        bild = r["image"]
        if max(bild.size) != lang:
            bild = bild.resize((lang, lang), Image.BILINEAR)
        bild.convert("RGB").save(ordner / f"{i:07d}.jpg", quality=88)
        geschrieben += 1
        if geschrieben % 2000 == 0:
            p.zeile(f"[synset]   {geschrieben}/{erwartet}")
    if unbekannt:
        p.zeile(f"[synset] {unbekannt} Bild(er) ohne Label in unserer Liste - uebersprungen")
    return geschrieben


# Open Images V7: die dichte Fassung liegt als "2018_04" weiterhin zum Abruf bereit
# (die Pfade unter /v7/ antworten mit 403 - gemessen). Verkehrszeichen ist /m/01mqdt.
OI_BASIS = "https://storage.googleapis.com/openimages/2018_04"
OI_FOTO = "https://open-images-dataset.s3.amazonaws.com"
OI_VERKEHRSZEICHEN = "/m/01mqdt"


def oi_bereitstellen(p: Protokoll, roh: Path, ziel: Path, max_fotos: int,
                     split: str = "validation") -> int:
    """Fotos aus Open Images holen, auf denen KEIN Verkehrszeichen annotiert ist.

    Zwei Aufgaben auf einmal: sie liefern echte Umgebungen fuer die Komposition (statt nur
    synthetischer Flaechen) und sie sind die ehrliche Gegenprobe auf Fehlalarme - echte
    Fotos ohne deutsches Schild. Bilder MIT Verkehrszeichen werden uebersprungen: sie waeren
    als Negativ falsch beschriftet (im Foto steckt ein Zeichen, nur kein deutsches).

    Vorbereitete Fotos (Ordner `oi-negatives/` im Rohdaten-Datensatz) haben Vorrang: dann
    braucht der Kernel kein Netz fuer die Bilder und der Lauf ist reproduzierbar.
    """
    if PROBE:
        p.zeile(f"[probe] Open Images -> {ziel} (max {max_fotos})")
        return 0
    ziel.mkdir(parents=True, exist_ok=True)
    vorbereitet = roh / "oi-negatives"
    if vorbereitet.exists():
        fotos = sorted(q for q in vorbereitet.iterdir()
                       if q.suffix.lower() in (".jpg", ".jpeg", ".png"))[:max_fotos]
        for i, f in enumerate(fotos):
            shutil.copy2(f, ziel / f"{i:05d}.jpg")
        p.zeile(f"[oi] {len(fotos)} vorbereitete Fotos aus {vorbereitet} uebernommen")
        return len(fotos)

    import csv as _csv
    import urllib.request

    p.zeile(f"[oi] hole Annotationen ({split})")
    mit_zeichen: set[str] = set()
    alle: set[str] = set()
    try:
        adresse = f"{OI_BASIS}/{split}/{split}-annotations-bbox.csv"
        p.zeile(f"[oi] hole {adresse}")
        with urllib.request.urlopen(adresse, timeout=120) as antwort:
            zeilen = (z.decode("utf-8", "replace") for z in antwort)
            for r in _csv.DictReader(zeilen):
                bild, label = r["ImageID"], r["LabelName"]
                alle.add(bild)
                if label == OI_VERKEHRSZEICHEN:
                    mit_zeichen.add(bild)
    except Exception as fehler:
        p.zeile(f"[oi] Annotationen nicht abrufbar ({type(fehler).__name__}: {fehler}) - "
                "Quelle entfaellt")
        return 0
    kandidaten = sorted(alle - mit_zeichen)
    p.zeile(f"[oi] {len(alle)} Fotos im Split {split}, davon {len(mit_zeichen)} mit "
            f"Verkehrszeichen; {len(kandidaten)} ohne")
    geholt, versuche = 0, 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def hole(bild: str) -> bytes | None:
        try:
            with urllib.request.urlopen(f"{OI_FOTO}/{split}/{bild}.jpg", timeout=60) as f:
                return f.read()
        except Exception:
            return None                                  # einzelne Ausfaelle sind normal

    # Acht Faehden: bei rund 0,3 s je Foto sind 4 000 Bilder sonst 20 Minuten Wartezeit.
    with ThreadPoolExecutor(max_workers=8) as pool:
        auftraege = {pool.submit(hole, bild): bild for bild in kandidaten[: max_fotos * 2]}
        for auftrag in as_completed(auftraege):
            versuche += 1
            if geholt >= max_fotos:
                for rest in auftraege:              # Schluessel sind die Auftraege selbst
                    rest.cancel()
                break
            rohdaten = auftrag.result()
            if rohdaten is None:
                continue
            (ziel / f"{geholt:05d}.jpg").write_bytes(rohdaten)
            geholt += 1
            if geholt % 500 == 0:
                p.zeile(f"[oi]   {geholt}/{max_fotos}")
    if geholt < max_fotos:
        p.zeile(f"[oi] nur {geholt} von {max_fotos} Fotos geladen ({versuche} Versuche)")
    return geholt


def oi_aufteilen(p: Protokoll, quelle: Path, anteile: tuple = (0.60, 0.15, 0.25)) -> dict:
    """Die Open-Images-Fotos in drei Toepfe trennen (Training / Messlatte / Fehlalarme).

    Das ist die wichtigste Vorsichtsmassnahme dieses Blocks: ein FOTO, dessen Ausschnitte im
    Training lagen, darf nicht in der Messlatte liegen - sonst misst man Gelerntes. Getrennt
    werden deshalb die Fotos, nicht die fertigen Bilder.
    """
    fotos = sorted(quelle.glob("*.jpg"))
    if not fotos:
        return {"train": 0, "val": 0, "neg": 0}
    g1 = int(len(fotos) * anteile[0])
    g2 = g1 + int(len(fotos) * anteile[1])
    toepfe = {"train": fotos[:g1], "val": fotos[g1:g2], "neg": fotos[g2:]}
    zahlen: dict[str, int] = {}
    for name, liste in toepfe.items():
        ziel = WORK / f"oi-{name}"
        ziel.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(liste):
            shutil.copy2(f, ziel / f"{i:05d}.jpg")
        zahlen[name] = len(liste)
    p.zeile("[oi] Fotos getrennt (kein Leck zwischen Training und Messlatte): "
            + ", ".join(f"{k}={v}" for k, v in zahlen.items()))
    return zahlen


def gtsdb_bereitstellen(p: Protokoll, roh: Path, ziel: Path) -> dict:
    """GTSDB (echte deutsche Szenen, COCO-Fassung) holen und entpacken.

    Nur 72 MB - deshalb wird zur Laufzeit geladen statt in den Kaggle-Datensatz gelegt.
    Liegt im Rohdaten-Datensatz schon ein Ordner `gtsdb-coco/`, wird der genommen.

    Das ist die einzige Quelle mit fertigen Szenen (Schild klein im Bild). Sie liefert damit
    die ehrliche Messlatte, die das Projekt bisher nicht hatte (PLAN.md §1).
    """
    if PROBE:
        p.zeile(f"[probe] GTSDB -> {ziel}")
        return {"gefunden": True, "teile": {"train": 0, "valid": 0, "test": 0}}
    vorbereitet = roh / "gtsdb-coco"
    teile = ("train", "valid", "test")
    if vorbereitet.exists():
        for name in teile:
            if (vorbereitet / name).exists():
                shutil.copytree(vorbereitet / name, ziel / name, dirs_exist_ok=True)
        p.zeile(f"[gtsdb] vorbereitete Szenen aus {vorbereitet} uebernommen")
        # Echte Bildzahlen statt blosser Ja/Nein-Werte: aus ihnen wird unten die erwartete
        # Groesse der Messlatte berechnet (die Szenen zaehlen als eigene Bilder dazu).
        return {"gefunden": True,
                "teile": {n: len(list((ziel / n).glob("*.jpg"))) for n in teile}}

    import urllib.request
    import zipfile

    ziel.mkdir(parents=True, exist_ok=True)
    geholt = {}
    for name in teile:
        adresse = (f"https://huggingface.co/datasets/keremberke/"
                   f"german-traffic-sign-detection/resolve/main/data/{name}.zip")
        zieldatei = ziel / f"{name}.zip"
        try:
            p.zeile(f"[gtsdb] hole {adresse}")
            with urllib.request.urlopen(adresse, timeout=120) as f:
                zieldatei.write_bytes(f.read())
            with zipfile.ZipFile(zieldatei) as z:
                z.extractall(ziel / name)
            zieldatei.unlink()
            geholt[name] = len(list((ziel / name).glob("*.jpg")))
        except Exception as fehler:
            p.zeile(f"[gtsdb] {name} nicht ladbar ({type(fehler).__name__}: {fehler})")
            geholt[name] = 0
    p.zeile("[gtsdb] Szenen: " + ", ".join(f"{k}={v}" for k, v in geholt.items()))
    return {"gefunden": False, "teile": geholt}


def eigene_negative(p: Protokoll, roh: Path, anteil_train: float = 0.7) -> tuple:
    """Die selbst gesammelten Negative des Nutzers in Training und Messlatte teilen.

    Warum das wichtig ist: auf genau diesen Motiven (Anzeigen, Poster, Produktfotos) hatte
    die Vorfassung ihre Fehlalarme. Sie muessen deshalb ins TRAINING - und ein Teil von
    ihnen in die Messlatte, sonst hat man fuer diese Bildart gar keine Prufung.

    Getrennt wird nach FOTO, nicht nach Ausschnitt: je Foto entstehen rund 33 Ausschnitte,
    dieselbe Aufnahme in beiden Toepfen waere Selbstbetrug (PLAN.md).
    """
    ordner = roh / "negatives" / "eigene"
    fotos = sorted(q.name for q in ordner.glob("*.jpg")) if ordner.exists() else []
    if not fotos:
        p.zeile(f"[hinweis] keine eigenen Negative in {ordner} - Schritt entfaellt")
        return [], []
    grenze = max(1, int(len(fotos) * anteil_train))
    ziel = WORK / "negatives" / "eigene"
    ziel.mkdir(parents=True, exist_ok=True)
    for name in fotos:
        shutil.copy2(ordner / name, ziel / name)
    pfad = "negatives/eigene/"
    train = [pfad + n for n in fotos[:grenze]]
    mess = [pfad + n for n in fotos[grenze:]]
    p.zeile(f"[eigene] {len(fotos)} Fotos getrennt: {len(train)} Training, {len(mess)} Messlatte")
    return train, mess


def werkzeuge_holen(p: Protokoll, tools: Path) -> None:
    """Die Werkzeuge aus dem oeffentlichen Repo holen - festgenagelt auf WERKZEUGE_SHA.

    Warum ueberhaupt, wo sie doch im Rohdaten-Datensatz liegen: der Datensatz muss dann bei
    JEDER Code-Aenderung neu hochgeladen werden (700 MB, gemessen mehrfach gescheitert). Die
    Werkzeuge sind 215 KB - die kommen billiger und zuverlaessiger aus dem Netz, das der
    Kernel ohnehin hat (enable_internet). Der Datensatz liefert nur noch die GROSSEN
    Rohdaten (GTSRB 350 MB), die sich nicht sinnvoll erneut laden lassen.

    WERKZEUGE_SHA wird von kaggle/run.ps1 -Step push auf den aktuellen Commit gesetzt, damit
    Kernel und Werkzeuge niemals auseinanderlaufen (sonst trainiert man Code, der nicht
    geprueft wurde). Faellt das Netz aus, bleibt die Kopie aus dem Datensatz liegen.
    """
    if PROBE:
        p.zeile(f"[probe] Werkzeuge von GitHub (Commit {WERKZEUGE_SHA[:7]})")
        return
    import urllib.request
    import zipfile

    adresse = f"https://codeload.github.com/{REPO_SLUG}/zip/{WERKZEUGE_SHA}"
    zieldatei = WORK / "werkzeuge.zip"
    try:
        p.zeile(f"[github] hole {adresse}")
        with urllib.request.urlopen(adresse, timeout=180) as f:
            zieldatei.write_bytes(f.read())
        with zipfile.ZipFile(zieldatei) as z:
            namen = [n for n in z.namelist() if "/tools/" in n and n.endswith(".py")]
            if not namen:
                raise SystemExit("kein tools/*.py im Archiv")
            tools.mkdir(parents=True, exist_ok=True)
            for name in namen:
                ziel = tools / Path(name).name
                ziel.write_bytes(z.read(name))
        zieldatei.unlink()
        p.zeile(f"[github] {len(namen)} Werkzeuge uebernommen (Commit {WERKZEUGE_SHA[:7]})")
    except SystemExit:
        raise
    except Exception as fehler:
        p.zeile(f"[github] nicht erreichbar ({type(fehler).__name__}: {fehler}) - "
                "es bleibt bei der Kopie aus dem Rohdaten-Datensatz")


def gtsign_bereitstellen(p: Protokoll, roh: Path, ziel: Path) -> list[str]:
    """GTSIGN-220 (+ StVO-Tabelle und Split-Listen) bereitstellen.

    Bevorzugt aus dem Rohdaten-Datensatz, sonst direkt von HuggingFace (375 MB). Die
    Split-Listen sind der eigentliche Wert: ohne sie gaebe es keine Trennung von Training und
    Messlatte - und ohne diese Trennung misst man Gelerntes (PLAN.md).
    """
    dateien = {"GTSIGN-220.zip": (HUGGING + "GTSIGN-220.zip", 300_000_000),
               "class_descriptions_and_stvo.csv": (HUGGING + "class_descriptions_and_stvo.csv", 20_000)}
    for split in ("train", "val"):
        dateien[f"gtsign_splits/{split}.txt"] = (HUGGING + f"splits/{split}.txt", 100_000)
    vorhanden = roh / "kataloge"
    fehlen: list[str] = []
    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "gtsign_splits").mkdir(exist_ok=True)
    for name, (adresse, mindest) in dateien.items():
        zieldatei = ziel / name
        quelle = vorhanden / name
        if quelle.exists() and quelle.stat().st_size >= mindest:
            shutil.copy2(quelle, zieldatei)
            continue
        if PROBE:
            p.zeile(f"[probe] GTSIGN {name} von {adresse}")
            continue
        try:
            import urllib.request
            p.zeile(f"[gtsign] hole {adresse}")
            with urllib.request.urlopen(adresse, timeout=300) as f:
                zieldatei.write_bytes(f.read())
            if zieldatei.stat().st_size < mindest:
                raise SystemExit(f"{name} zu klein ({zieldatei.stat().st_size} B) - "
                                 "HuggingFace liefert bei Abbruch eine Fehlerseite")
        except SystemExit:
            raise
        except Exception as fehler:
            fehlen.append(f"{name} ({type(fehler).__name__})")
    if fehlen and not PROBE:
        p.zeile(f"[warnung] GTSIGN unvollstaendig: {', '.join(fehlen)} - Quelle entfaellt")
    elif not PROBE:
        p.zeile("[gtsign] GTSIGN-220 + StVO-Tabelle + Split-Listen bereit")
    return fehlen


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

    # Werkzeuge auf den Stand festnageln, der zum Kernel gehoert (siehe werkzeuge_holen).
    werkzeuge_holen(p, tools)

    # --- Die neuen Kataloge (tools/crops_dataset.py) -------------------------------------
    # Alles, was der Kernel an Zusatzdaten braucht, kommt aus dem Rohdaten-Datensatz. Fehlt
    # ein Katalog, wird er gemeldet und uebersprungen - der Lauf faellt nicht um.
    katalog = roh / "kataloge"
    gtsign_zip = WORK / "kataloge" / "GTSIGN-220.zip"
    fehlend = gtsign_bereitstellen(p, roh, WORK / "kataloge")
    if fehlend:
        p.zeile(f"[hinweis] GTSIGN-220 nicht verfuegbar ({', '.join(fehlend)}) - Quelle entfaellt")
    crops = {"gtsign_train": 0, "gtsign_val": 0, "synset_train": 0, "synset_val": 0}
    if not fehlend:
        crops["gtsign_train"] = feste_ausschnitte(
            p, gtsign_zip, WORK / "kataloge" / "class_descriptions_and_stvo.csv",
            WORK / "kataloge" / "gtsign_splits" / "train.txt", WORK / "crops" / "gtsign-train",
            DATEN["gtsign_train_max"])
        crops["gtsign_val"] = feste_ausschnitte(
            p, gtsign_zip, WORK / "kataloge" / "class_descriptions_and_stvo.csv",
            WORK / "kataloge" / "gtsign_splits" / "val.txt", WORK / "crops" / "gtsign-val",
            DATEN["gtsign_val_max"])
    crops["synset_train"] = synset_ausschnitte(p, roh, WORK / "crops" / "synset-train",
                                               SYNSET_KONFIG, "train",
                                               DATEN["synset_train_max"])
    crops["synset_val"] = synset_ausschnitte(p, roh, WORK / "crops" / "synset-val",
                                             SYNSET_KONFIG, "validation",
                                             DATEN["synset_val_max"])
    oi_bereitstellen(p, roh, WORK / "oi-alle", DATEN["oi_max"])
    oi = oi_aufteilen(p, WORK / "oi-alle")
    if not PROBE and sum(oi.values()) < 100:
        # Frueh und deutlich statt kryptisch: ohne diese Fotos kann die Messlatte 'neg'
        # (zu zwei Dritteln daraus gebaut) nicht entstehen, und zahlen_pruefen wuerde den
        # Lauf erst nach dem Datensatzbau abbrechen - mit einer Zahl statt einer Ursache.
        p.zeile("[fehler] Open Images lieferte keine Fotos. Der Datensatz kann so nicht "
                "gebaut werden. Netz im Kernel pruefen (enable_internet) oder einen Ordner "
                "oi-negatives/ in den Rohdaten-Datensatz legen.")
        raise SystemExit("Open Images fehlt - siehe Protokoll")
    gtsdb = gtsdb_bereitstellen(p, roh, WORK / "gtsdb-coco")

    return {"tools": str(tools), "gtsrb": str(WORK / "gtsrb"), "coco128": coco,
            "foto": str(nutzer / "Nothing.jpg"),
            "alt": alt_bereitstellen(p, roh) if ALT_VERGLEICH else "",
            "crops": crops, "oi": oi, "gtsdb": gtsdb,
            "eigene": eigene_negative(p, roh)}



def datensatz_bauen(p: Protokoll, ein: dict) -> tuple[dict, dict]:
    """Den Detektor-Datensatz bauen - jetzt aus vier Quellen mit Quellen-Upsampling.

    Aufbau (PLAN.md, Block 6): erst die Kataloge fuer Training und Messlatte getrennt, dann
    die synthetischen Negative (Negativbilder OHNE Schild sind im Betrieb die halbe Miete),
    dann die echten Negative aus Open Images. Die Werkzeuge haengen sich an einen
    BESTEHENDEN Ordner (manifest.json), deshalb ist die Reihenfolge bindend.

    Der entscheidende Unterschied zu Block 5: die Messlatte wird aus den SPLITS der Kataloge
    gebaut (GTSIGN-val, Synset-validation), nicht nur aus GTSRB. Vorher war sie GTSRB-Material
    mit grossen Zeichen - dort stand 0,90, waehrend dasselbe Modell auf echten Szenen 0,00
    lieferte (PLAN.md §1). Ein Split des Katalogs kommt NIE auch ins Training: sonst misst
    man dasselbe Bild, das man gelernt hat.
    """
    d = DATEN
    dauer: dict[str, float] = {}
    # Upsampling der Quellen: --weight gilt JE QUELLE, nicht je Bild. GTSIGN und Synset sind
    # kleiner, aber sauberer annotiert; ohne Faktor wuerde die Menge GTSRB/Synset die
    # GTSIGN-Bilder erdruecken.
    g_gewichte = [arg for w in d["weights"] for arg in ("--weight", w)]
    # coco128 nur anhängen, wenn es wirklich bereitgestellt werden konnte (coco_bereitstellen
    # liefert dann einen Pfad, sonst "").
    coco_pfad = ein.get("coco128") or ""
    coco_args = (["--coco", coco_pfad, "--coco-test-n", str(d["coco_test_n"])]
                 if coco_pfad else [])
    # Nur was WIRKLICH da ist, wird als Quelle uebergeben. Alle Reader werfen bei fehlendem
    # Pfad sofort (GtsignQuelle: zipfile.FileNotFoundError, OrdnerQuelle/SzenenQuelle:
    # FileNotFoundError aus iterdir) - eine fehlende Quelle soll aber nur DIESE Quelle
    # kosten, nicht den ganzen Lauf.
    def da(relativ: str) -> bool:
        return (WORK / relativ).exists()

    def fehlt(was: str, relativ: str) -> list[str]:
        p.zeile(f"[hinweis] {was} fehlt ({relativ}) - Quelle entfaellt")
        return []

    # ACHTUNG: hier stand einmal "catalogs/" mit c. gtsign_bereitstellen legt den Ordner aber
    # als WORK/kataloge an, und die Werkzeuge laufen mit cwd=WORK. Der Lauf vom 03.10. brach
    # deshalb NACH 28 Minuten Datensatzvorbereitung mit
    # FileNotFoundError: catalogs/GTSIGN-220.zip ab - genau das verhindert der Vergleich unten.
    kataloge = (["--gtsign", "kataloge/GTSIGN-220.zip",
                 "--gtsign-csv", "kataloge/class_descriptions_and_stvo.csv"]
                if da("kataloge/GTSIGN-220.zip") and da("kataloge/class_descriptions_and_stvo.csv")
                else fehlt("GTSIGN-220", "kataloge/GTSIGN-220.zip"))
    # GTSRB stand bis zum 03.10. nur im Vorbereitungsschritt: 39 253 Bilder wurden gepackt
    # (280 MB, 3 min) und dann nie als --gtsrb uebergeben - 12 nutzbare Klassen lagen brach.
    # Die Messlatte des Test-Sets bleibt dagegen absichtlich draussen: sie bestand frueher NUR
    # aus GTSRB-Material und liess die Bewertung mit 0,90 gut aussehen, waehrend dasselbe
    # Modell auf echten Szenen 0,00 lieferte (PLAN.md §1). Gelernt wird daraus trotzdem.
    gtsrb_train = (["--gtsrb", "gtsrb/train.zip"] if da("gtsrb/train.zip")
                   else fehlt("GTSRB", "gtsrb/train.zip"))
    syn_train_q = (["--synset", "crops/synset-train"] if da("crops/synset-train")
                   else fehlt("Synset (Training)", "crops/synset-train"))
    syn_val_q = (["--synset", "crops/synset-val"] if da("crops/synset-val")
                 else fehlt("Synset (Messlatte)", "crops/synset-val"))
    szene_train_q = (["--scenes", "oi-train"] if da("oi-train")
                     else fehlt("Open Images (Training)", "oi-train"))
    szene_val_q = (["--scenes", "oi-val"] if da("oi-val")
                   else fehlt("Open Images (Messlatte)", "oi-val"))
    # Der reine Negativ-Split BRAUCHT die Szenen (er besteht nur aus ihnen), deshalb wird der
    # Schritt unten ganz weggelassen, wenn der Ordner fehlt - sonst bricht der Lauf dort ab.
    szene_neg_q = (["--scenes", "oi-neg"] if da("oi-neg")
                   else fehlt("Open Images (Negative)", "oi-neg"))

    schritte = [
        # 1. Training: die Kataloge + echte Fotos aus Open Images als Umgebung und Negative
        ("kataloge train", python("tools/crops_dataset.py", "--out", "data/det",
                                  "--n", str(d["n_train"]), "--split", "train",
                                  "--size", str(d["size"]), "--seed", "0", "--balance",
                                  *kataloge, *gtsrb_train,
                                  *syn_train_q,
                                  *szene_train_q,
                                  "--neg-share", str(d["neg_share_train"]),
                                  *g_gewichte)),
        # 2. Messlatte: die SPLITS der Kataloge - Bilder, die im Training nicht vorkommen
        ("kataloge val", python("tools/crops_dataset.py", "--out", "data/det",
                                "--n", str(d["n_val"]), "--split", "val",
                                "--size", str(d["size"]), "--seed", "1",
                                *kataloge,
                                *syn_val_q,
                                *szene_val_q,
                                "--neg-share", str(d["neg_share_val"]))),
        # 3. Synthetische Negative (gezeichnete Stoererflaechen, Anzeigen, Nacht)
        ("synthetische negative", python("tools/synth_negatives.py", "--out", "data/det",
                                         "--n", str(d["n_neg_synth"]), "--split", "train",
                                         "--size", str(d["size"]))),
        # 4. Echte Negative aus Block 5 (eigenes Foto + coco128) - andere Umgebung als OI
        ("echte negative", python("tools/real_negatives.py", "--out", "data/det",
                                   "--n", str(d["n_echt_train"]), "--split", "train",
                                   "--size", str(d["size"]),
                                   "--real-train", "user/Nothing.jpg",
                                   "--train-region", "0,0,1,0.5",
                                   *coco_args,
                                   "--user-share", str(d["user_share_train"]))),
        # 5. Die Gegenprobe auf Fehlalarme: nur Bilder OHNE Schild, aus dem dritten OI-Topf.
        #    Ohne diese Szenen ist der Schritt sinnlos (er besteht nur aus ihnen) - dann faellt
        #    er weg, statt mit FileNotFoundError den Lauf zu beenden.
        *([("negative messlatte", python("tools/crops_dataset.py", "--out", "data/det",
                                        "--n", str(d["n_neg"]), "--split", "neg",
                                        "--size", str(d["size"]), "--seed", "2",
                                        *szene_neg_q, "--neg-share", "1.0"))]
          if szene_neg_q else []),
        ("synthetische negative (Messlatte)", python("tools/synth_negatives.py", "--out", "data/det",
                                                     "--n", str(d["n_neg_synth"]), "--split", "neg",
                                                     "--size", str(d["size"]))),
        # 6. GTSDB: die einzigen ECHTEN Szenen (Schild klein im Bild). Trainingssplit ins
        #    Training, valid+test in die Messlatte - getrennte Ordner, also kein Leck.
        ("gtsdb train", python("tools/gtsdb_dataset.py", "--coco", "gtsdb-coco/train",
                               "--out", "data/det", "--split", "train",
                               "--size", str(d["size"]))),
        ("gtsdb val", python("tools/gtsdb_dataset.py", "--coco", "gtsdb-coco/valid",
                             "--out", "data/det", "--split", "val",
                             "--size", str(d["size"]))),
        ("gtsdb test", python("tools/gtsdb_dataset.py", "--coco", "gtsdb-coco/test",
                              "--out", "data/det", "--split", "val",
                              "--size", str(d["size"]))),
    ]

    # 7. Die selbst gesammelten Negative des Nutzers. Sie sind der unmittelbare Grund fuer
    #    diesen Block (16 Fehlalarme auf genau solchen Motiven) und brauchen einen EIGENEN
    #    Aufruf: --train-region oben wuerde sie auf die obere Bildhaelfte beschneiden, und
    #    das ist nur fuer user/Nothing.jpg gewollt (dort ist die untere Haelfte fuer die
    #    Messlatte reserviert).
    eigene_train, eigene_neg = ein.get("eigene", ([], []))
    if eigene_train:
        schritte.append(("eigene negative (Training)",
                         python("tools/real_negatives.py", "--out", "data/det",
                                "--n", str(d["n_eigene_train"]), "--split", "train",
                                "--size", str(d["size"]),
                                # Eigenes Kennzeichen: sonst haette dieser Aufruf die 3 600
                                # echten Negative des Schritts darueber geloescht (gleicher
                                # src "echt (Negativ)" und gleicher Split = Ersetzen).
                                "--src", "eigene (Negativ)",
                                "--real-train", *eigene_train,
                                *coco_args,
                                "--user-share", str(d["eigene_share"]))))
    if eigene_neg:
        schritte.append(("eigene negative (Messlatte)",
                         python("tools/real_negatives.py", "--out", "data/det",
                                "--n", str(d["n_eigene_neg"]), "--split", "neg",
                                "--size", str(d["size"]),
                                "--src", "eigene (Negativ)",
                                "--real-test", *eigene_neg)))
    for was, cmd in schritte:
        dauer[was] = lauf(p, cmd, f"Datensatz: {was}")

    # Die erwarteten Bildzahlen werden aus dem Rezept SELBST gerechnet, nicht fest
    # eingetragen. Grund: die eigenen Negative des Nutzers liegen nur dann im Rohdatensatz,
    # wenn ihr Upload geklappt hat - im Lauf vom 03.10. fehlten sie, und eine feste Zahl
    # haette den Lauf nach 30 Minuten Datensatzvorbereitung beendet, obwohl alles in Ordnung
    # war. Geprueft wird damit weiterhin genau das, was schiefgehen kann: dass ein Schritt
    # weniger Bilder schreibt, als er zugesagt hat.
    teile = (ein.get("gtsdb") or {}).get("teile") or {}
    gtsdb_train = int(teile.get("train") or 0)
    gtsdb_val = int(teile.get("valid") or 0) + int(teile.get("test") or 0)
    erwartet = {
        "train": (d["n_train"] + d["n_neg_synth"] + d["n_echt_train"] + gtsdb_train
                  + (d["n_eigene_train"] if eigene_train else 0)),
        "val": d["n_val"] + gtsdb_val,
        "neg": (d["n_neg"] if szene_neg_q else 0) + d["n_neg_synth"]
               + (d["n_eigene_neg"] if eigene_neg else 0),
    }
    p.zeile("[daten] erwartet aus dem Rezept: "
            + ", ".join(f"{k}={v}" for k, v in erwartet.items()))
    return dauer, erwartet


def zahlen_pruefen(p: Protokoll, erwartet: dict) -> dict:
    """Die Bildzahlen des Manifests gegen die Erwartung stellen.

    Ein Tippfehler im Rezept faellt sonst erst nach zwei Stunden Training auf - und dann
    sieht man nur eine schlechtere Zahl, nicht die Ursache. Deshalb wird hier hart geprueft.
    Die Erwartung kommt aus datensatz_bauen und wird dort aus dem Rezept gerechnet; fehlt
    eine Wahlquelle (etwa die eigenen Negative, wenn ihr Upload nicht geklappt hat), passt
    sie sich an, statt den Lauf zu beenden.
    """
    manifest = json.loads((WORK / "data" / "det" / "manifest.json").read_text(encoding="utf-8"))
    zahlen: dict[str, int] = {}
    for e in manifest["images"]:
        split = e.get("split", "train")
        zahlen[split] = zahlen.get(split, 0) + 1
    p.zeile(f"[daten] {len(manifest['images'])} Bilder im Manifest: "
            + ", ".join(f"{k}={v}" for k, v in sorted(zahlen.items())))
    # Toleranz statt harter Gleichheit: das Rezept ist eine Summe aus sieben Aufrufen, und
    # einzelne Bilder fallen regelmaessig weg (ein Negativ ohne Hintergrundfoto, ein Zeichen
    # kleiner als die Mindestkante). Ein Tippfehler im Rezept verschiebt die Zahl dagegen um
    # Prozent, nicht um Promille - die Pruefung greift also weiterhin, toetet den Lauf aber
    # nicht mehr wegen ein paar Bildern.
    grenze = 0.02
    falsch, ungenau = {}, {}
    for k, soll in erwartet.items():
        ist = zahlen.get(k, 0)
        if ist == soll:
            continue
        (falsch if soll and abs(ist - soll) / soll > grenze else ungenau)[k] = (ist, soll)
    for k, (ist, soll) in ungenau.items():
        p.zeile(f"[daten] {k}: {ist} statt {soll} erwartet "
                f"({100 * (ist - soll) / max(soll, 1):+.1f} %) - im Rahmen")
    if falsch:
        p.zeile("[daten] ERWARTUNG TRIFFT NICHT ZU: "
                + ", ".join(f"{k}: {ist} statt {soll}" for k, (ist, soll) in falsch.items()))
        raise SystemExit("Datensatz weicht stark vom Rezept ab (siehe DATEN in train_kernel.py)")
    p.zeile("[daten] Bildzahlen stimmen zum Rezept")
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
                 # Verlust und Kopf: Focal im Klassifikationskopf, Klassengewichte,
                 # Label-Smoothing, hierarchischer Kopf mit direkt ueberwachter Familie.
                 "--focal-gamma", str(t["focal_gamma"]),
                 "--focal-alpha", str(t["focal_alpha"]),
                 "--cls-w", str(t["cls_w"]),
                 "--smooth", str(t["smooth"]),
                 "--hier-aux", str(t["hier_aux"]),
                 "--seed", str(t["seed"]))
    return lauf(p, cmd, f"Training: {t['epochs']} Epochen x {t['steps']} Schritte "
                        f"x {t['batch']} Bilder (Merksatz: 1 Epoche = {t['steps']*t['batch']} Bilder)")


def exportieren(p: Protokoll) -> float:
    """ONNX-Export mit dynamischen Achsen: EINE Datei fuer 256/320/384/448 px - plus int8.

    Der Export prueft sich selbst (Paritaet gegen PyTorch) und schreibt labels.json +
    manifest.json dazu - genau die drei Dateien, die src/model.js erwartet. fatal=False:
    fehlt onnxruntime, soll der Lauf trotzdem seine Messwerte abliefern.

    Das int8-Modell ist das, was ausgeliefert wird (src/model.js nimmt labels.files.int8 zuerst).
    Gemessen an einem Pruefmodell: 8,75 MB -> 2,67 MB (-69 %), max. Tensorabweichung 1,06e-02.
    """
    q = QUANT
    return lauf(p, python("tools/export_onnx.py", "--ckpt", "models/signs-det.pt",
                          "--out", "models/signs-det.onnx", "--dynamic",
                          "--int8", "--calib-data", q["calib_data"],
                          "--calib-n", str(q["n_calib"])),
                "Export nach ONNX (dynamische Hoehe/Breite) + int8", fatal=False)


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
        # Schwellen-Suche: conf 0,15..0,60 x NMS-IoU 0,40..0,70, Optimum nach F1. Kostet kein
        # Neutraining, sondern nur die Nachbearbeitung - das Netz laeuft einmal. Ergebnis nach
        # berichte/sweep.json; der Bericht uebernimmt es als "sweep.json".
        ("Schwellen-Suche conf/NMS", "tools/eval_conditions.py",
         ["--ckpt", "models/signs-det.pt", *gemein, "--split", "val", "--sweep",
          "--sweep-limit", str(a["sweep_limit"]), "--json", "berichte/sweep.json"]),
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
                             "sweep.json", "model-out.json")}
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
    hat_ds = datasets_sichern(p)
    roh = eingang_finden()
    p.zeile(f"[eingang] Rohdaten-Datensatz: {roh}")

    dauer: dict[str, float] = {"einrichten": 0.0}
    ein = einrichten(p, roh)
    dauer_bauen, erwartet = datensatz_bauen(p, ein)
    dauer.update(dauer_bauen)
    zahlen = {} if PROBE else zahlen_pruefen(p, erwartet)
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
                        (f"val {AUSWERTUNG['zweite_groesse']} px", "val384.json"),
                        ("Vergleich Block 4", "val_block4.json"),
                        ("Schwellen-Suche", "sweep.json")):
        daten = report["kennzahlen"].get(blatt)
        if daten:
            p.zeile(f"[ergebnis] {name:18s} P={daten['precision']:.3f} R={daten['recall']:.3f} "
                    f"F1={daten['f1']:.3f}  tp={daten['tp']} fp={daten['fp']} fn={daten['fn']}")
    p.zeile(f"[ergebnis] Modell: models/signs-det.onnx "
            f"{report['modelldateien'].get('signs-det.onnx', 0)/1e6:.2f} MB, "
            f"int8 {report['modelldateien'].get('signs-det-int8.onnx', 0)/1e6:.2f} MB "
            f"(das liefert der Browser aus), "
            f"Checkpoint {report['modelldateien'].get('signs-det.pt', 0)/1e6:.1f} MB")
    if FEHLER:
        p.zeile()
        p.zeile("[warnung] nicht gelaufen: " + ", ".join(FEHLER))
        p.zeile("[warnung] die uebrigen Ergebnisse liegen in /kaggle/working "
                "(Checkpoint, Datensatz, Messwerte, Protokoll)")
    p.datei.close()


if __name__ == "__main__":
    main()

