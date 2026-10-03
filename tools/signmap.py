"""
tools/signmap.py - Eine Taxonomie fuer alle Datenquellen (mehr als die alten 9 Typen).

Warum diese Datei der Angelpunkt ist: Die drei grossen Quellen benennen ihre Klassen
unterschiedlich - GTSRB eine ClassId 0..42, GTSIGN-220 die StVO-Nummer ("274-70"),
Synset Signset Germany den deutschen Namen ("Geschwindigkeit70"). Ein Detektor kann aber
nur EINE Klassenliste haben. Diese Datei fuehrt alle drei auf dieselben LABELS zusammen.

Zwei unabhaengige deutsche Kataloge muessen dabei auf denselben Namen landen - das ist die
eigentliche Pruefung der Zuordnung:

    python tools/signmap.py --gtsign-csv data/gtsign/class_descriptions_and_stvo.csv
    python tools/signmap.py --synset-json data/synset-klassen.json

Ausgegeben wird je Quelle, wie viele Klassen auf welches Label fallen und was uebrig
bleibt. Alles, was auf None faellt, wird verworfen (nicht "sonstiges" - sonst lernt das Netz
einen Sammeltopf, der im Betrieb nichts bedeutet).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Die Klassenliste. Reihenfolge = Klassenindex im Modell UND Zeilenindex im YOLO-Label.
# Aufbau: erst die haeufigen und verkehrlich wichtigen Zeichen, dann die selteneren.
# Wer hier etwas umsortiert, muss neu trainieren - die Gewichte haengen an der Reihenfolge.
# ---------------------------------------------------------------------------
LABELS: list[str] = [
    # Vorfahrt / Halt
    "stop", "vorfahrtGewaehren", "vorfahrtstrasse", "vorfahrtstrasseEnde", "andreaskreuz",
    # Tempolimits (Z 274) - der haeufigste Zeichentyp ueberhaupt
    "tempo5", "tempo10", "tempo20", "tempo30", "tempo40", "tempo50", "tempo60", "tempo70",
    "tempo80", "tempo90", "tempo100", "tempo110", "tempo120", "tempo130",
    "tempoEnde", "zone20", "zone30", "zoneEnde", "mindestgeschwindigkeit",
    # Ueberholverbot
    "ueberholverbot", "ueberholverbotKfz", "ueberholverbotEnde",
    # Verbote
    "einfahrtVerboten", "verbotFahrzeuge", "verbotFussgaenger", "verbotRadverkehr",
    "verbotWenden", "halteverbot", "verbotSonstiges",
    # Gefahrzeichen: die Form ist gleich, unterschieden wird das Symbol im Inneren
    "warnGefahrstelle", "warnKreuzung", "warnKurve", "warnDoppelkurve", "warnVerengung",
    "warnUnebeneFahrbahn", "warnSchleuder", "warnGlaette", "warnArbeitsstelle",
    "warnLichtzeichen", "warnFussgaenger", "warnKinder", "warnRadverkehr",
    "warnTiereWildwechsel",
    "warnGegenverkehr", "warnStau", "warnSteinschlag", "warnBahnuebergang", "warnSonstiges",
    # Gebotszeichen
    "gebotRechts", "gebotLinks", "gebotGeradeaus", "gebotGeradeausSeitlich",
    "gebotVorbeifahrt", "kreisverkehr",
    # Rad- und Fusswege
    "radweg", "gehweg", "gemeinsamerGehUndRadweg", "getrennterRadGehweg",
    # Zonen
    "fussgaengerzone", "fahrradstrasse",
    # Hinweiszeichen
    "einbahnstrasse", "haltestelle", "parken", "autobahn", "tunnel", "sackgasse",
    "umleitung", "ortstafel",
    # Sammelklasse fuer beschriftete Hinweiszeichen (Informationsstelle, Erste Hilfe, ...)
    "hinweisSonstiges",
]
N_LABELS = len(LABELS)
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# Hierarchie: Ober- und Unterkategorien (fuer den hierarchischen Klassifikationskopf).
#
# Wozu ueberhaupt: die 74 Klassen sind keine gleichartige Liste, sondern neun Familien,
# zwischen denen der Kopf GANZ unterschiedlich leicht unterscheidet. Gemessen (PLAN.md)
# lagen die Fehler genau zwischen Geschwistern - "rotes Dreieck Spitze oben" gegen
# "Spitze unten", tempo70 gegen tempo80. Ein flacher Kopf muss alle 74 Entscheidungen
# auf einmal treffen; ein hierarchischer erst die Familie (Form/Farbe - das ist grob und
# robust) und danach das Symbol IM Inneren (fein, aber nur noch innerhalb der Familie).
#
# Wirkung auf die Ausgabe: KEINE. Der Kopf gibt weiterhin N_LABELS Kanaele aus, nur ist
# der Wert je Klasse die SUMME aus Familien-Logit und Unter-Logit. In Log-Wahrscheinlich-
# keiten ist das genau P(Klasse) = P(Familie) * P(Klasse | Familie) - dieselbe Rechnung,
# nur in zwei Schritten gelernt. Damit bleiben tools/detmath.py und src/model.js
# unveraendert (dort steht weiter "sigmoid(obj) * max(sigmoid(cls))").
#
# Die Reihenfolge der Familien ist die Reihenfolge im Modell (9 Ausgaenge im Familienkopf).
# Wer hier etwas aendert, muss neu trainieren - wie bei LABELS haengen die Gewichte daran.
# ---------------------------------------------------------------------------
SUPER_LABELS: list[str] = [
    "vorfahrt",    # Vorfahrt und Halt
    "tempo",       # Geschwindigkeitsbegrenzungen samt Ende und Zonen 20/30
    "ueberholen",  # Ueberholverbote
    "verbot",      # Verbote
    "warnung",     # Gefahrzeichen (gleiche Form, unterschiedliches Symbol)
    "gebot",       # Gebotszeichen (blauer Kreis, Richtung/Rad/Mindestgeschwindigkeit)
    "radfuss",     # Rad- und Fusswege
    "zone",        # Zonen
    "hinweis",     # Hinweiszeichen
]
N_SUPER = len(SUPER_LABELS)
SUPER_ID: dict[str, int] = {name: i for i, name in enumerate(SUPER_LABELS)}

# Zuordnung Unterkategorie -> Oberkategorie. Bewusst ueber NAMEN und nicht ueber Indizes:
# ein Tippfehler faellt in pruefen() auf, ein falscher Index nicht. Die Familien sind an
# der Verkehrsbedeutung und an der Form ausgerichtet (das ist es, was das Netz sieht).
SUPER_GRUPPEN: dict[str, tuple[str, ...]] = {
    "vorfahrt": ("stop", "vorfahrtGewaehren", "vorfahrtstrasse", "vorfahrtstrasseEnde",
                 "andreaskreuz"),
    "tempo": ("tempo5", "tempo10", "tempo20", "tempo30", "tempo40", "tempo50", "tempo60",
              "tempo70", "tempo80", "tempo90", "tempo100", "tempo110", "tempo120", "tempo130",
              "tempoEnde", "zone20", "zone30", "zoneEnde"),
    "ueberholen": ("ueberholverbot", "ueberholverbotKfz", "ueberholverbotEnde"),
    "verbot": ("einfahrtVerboten", "verbotFahrzeuge", "verbotFussgaenger",
               "verbotRadverkehr", "verbotWenden", "halteverbot", "verbotSonstiges"),
    "warnung": ("warnGefahrstelle", "warnKreuzung", "warnKurve", "warnDoppelkurve",
                "warnVerengung", "warnUnebeneFahrbahn", "warnSchleuder", "warnGlaette",
                "warnArbeitsstelle", "warnLichtzeichen", "warnFussgaenger", "warnKinder",
                "warnRadverkehr", "warnTiereWildwechsel", "warnGegenverkehr", "warnStau",
                "warnSteinschlag", "warnBahnuebergang", "warnSonstiges"),
    # Mindestgeschwindigkeit ist ein Gebotszeichen (blauer Kreis, Z 275) - es gehoert
    # deshalb hierher und nicht zu den Tempolimits, obwohl es von Geschwindigkeit handelt.
    "gebot": ("gebotRechts", "gebotLinks", "gebotGeradeaus", "gebotGeradeausSeitlich",
              "gebotVorbeifahrt", "kreisverkehr", "mindestgeschwindigkeit"),
    "radfuss": ("radweg", "gehweg", "gemeinsamerGehUndRadweg", "getrennterRadGehweg"),
    "zone": ("fussgaengerzone", "fahrradstrasse"),
    "hinweis": ("einbahnstrasse", "haltestelle", "parken", "autobahn", "tunnel",
                "sackgasse", "umleitung", "ortstafel", "hinweisSonstiges"),
}

# Unterkategorie -> Index der Oberkategorie, in der Reihenfolge von LABELS. Das ist die
# Tabelle, die der Klassifikationskopf beim Bauen liest (tools/hybrid_net.py).
SUPER_OF: list[int] = [-1] * N_LABELS
for _familie, _mitglieder in SUPER_GRUPPEN.items():
    for _name in _mitglieder:
        if _name in CLASS_ID:
            SUPER_OF[CLASS_ID[_name]] = SUPER_ID[_familie]
del _familie, _mitglieder, _name

# ---------------------------------------------------------------------------
# Anzeige-Informationen fuer die Web-Oberflaeche (src/detector.js, SIGNS).
# Je Zeile: (Label, Anzeigename, StVO-Nummer, Farbe, Kurznotiz).
# Muss zu LABELS passen - die Pruefung dazu steht am Ende dieser Datei (pruefen()).
# ---------------------------------------------------------------------------
INFO_ROWS: list[tuple] = [
    ("stop", "Stopp", "Z 206", "#d3232f", "Roter Achtkant: Halt!"),
    ("vorfahrtGewaehren", "Vorfahrt gewaehren", "Z 205/208", "#d3232f", "Dreieck mit Spitze unten."),
    ("vorfahrtstrasse", "Vorfahrtstrasse", "Z 306", "#f2c200", "Gelbe Raute mit weissem Rand."),
    ("vorfahrtstrasseEnde", "Ende Vorfahrtstrasse", "Z 307", "#e8e8e8", "Raute, grau durchgestrichen."),
    ("andreaskreuz", "Andreaskreuz", "Z 201", "#d3232f", "Bahnuebergang, weisses Kreuz."),
    ("tempo5", "Tempolimit 5", "Z 274-5", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo10", "Tempolimit 10", "Z 274-10", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo20", "Tempolimit 20", "Z 274-20", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo30", "Tempolimit 30", "Z 274-30", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo40", "Tempolimit 40", "Z 274-40", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo50", "Tempolimit 50", "Z 274-50", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo60", "Tempolimit 60", "Z 274-60", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo70", "Tempolimit 70", "Z 274-70", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo80", "Tempolimit 80", "Z 274-80", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo90", "Tempolimit 90", "Z 274-90", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo100", "Tempolimit 100", "Z 274-100", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo110", "Tempolimit 110", "Z 274-110", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo120", "Tempolimit 120", "Z 274-120", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempo130", "Tempolimit 130", "Z 274-130", "#d3232f", "Roter Ring, Zahl im Inneren."),
    ("tempoEnde", "Ende Tempolimit", "Z 278 ff.", "#e8e8e8", "Graue Zahl, schraeg durchgestrichen."),
    ("zone20", "Zone 20", "Z 274.1-20", "#d3232f", "Rechteck mit Zonenhinweis."),
    ("zone30", "Zone 30", "Z 274.1-30", "#d3232f", "Rechteck mit Zonenhinweis."),
    ("zoneEnde", "Ende Zone", "Z 274.2", "#e8e8e8", "Zonenende, grau."),
    ("mindestgeschwindigkeit", "Mindestgeschwindigkeit", "Z 275", "#1467b8", "Blauer Kreis, weisse Zahl."),
    ("ueberholverbot", "Ueberholverbot", "Z 276", "#d3232f", "Zwei Autos, eines rot."),
    ("ueberholverbotKfz", "Ueberholverbot Kfz", "Z 277", "#d3232f", "Lkw ueberholt Pkw, rot."),
    ("ueberholverbotEnde", "Ende Ueberholverbot", "Z 280/281", "#e8e8e8", "Graue Autos."),
    ("einfahrtVerboten", "Einfahrt verboten", "Z 267", "#d3232f", "Roter Vollkreis mit weissem Balken."),
    ("verbotFahrzeuge", "Verbot fuer Fahrzeuge", "Z 250/251/253", "#d3232f", "Roter Ring, Fahrzeugsymbol."),
    ("verbotFussgaenger", "Verbot fuer Fussgaenger", "Z 259", "#d3232f", "Roter Ring mit Fussgaenger."),
    ("verbotRadverkehr", "Verbot fuer Radverkehr", "Z 254", "#d3232f", "Roter Ring mit Fahrrad."),
    ("verbotWenden", "Wendeverbot", "Z 272", "#d3232f", "Roter Ring, Wendepfeil durchgestrichen."),
    ("halteverbot", "Halteverbot", "Z 283 ff.", "#1467b8", "Blauer Kreis, rot durchgestrichen."),
    ("verbotSonstiges", "Verbot (sonstiges)", "Z 2xx", "#d3232f", "Roter Ring, seltenes Symbol."),
    ("warnGefahrstelle", "Gefahrstelle", "Z 101", "#d3232f", "Dreieck mit Ausrufezeichen."),
    ("warnKreuzung", "Kreuzung/Einmuendung", "Z 102", "#d3232f", "Dreieck mit Kreuz."),
    ("warnKurve", "Kurve", "Z 103", "#d3232f", "Dreieck mit Kurvenpfeil."),
    ("warnDoppelkurve", "Doppelkurve", "Z 105", "#d3232f", "Dreieck mit S-Kurve."),
    ("warnVerengung", "Verengte Fahrbahn", "Z 120/121", "#d3232f", "Dreieck, Fahrbahn wird schmaler."),
    ("warnUnebeneFahrbahn", "Unebene Fahrbahn", "Z 112", "#d3232f", "Dreieck mit Bodenwelle."),
    ("warnSchleuder", "Schleudergefahr", "Z 114", "#d3232f", "Dreieck mit Schleuderspur."),
    ("warnGlaette", "Glaette", "Z 101-51/52", "#d3232f", "Dreieck mit Schneeflocke."),
    ("warnArbeitsstelle", "Arbeitsstelle", "Z 123", "#d3232f", "Dreieck mit Baustelle."),
    ("warnLichtzeichen", "Lichtzeichenanlage", "Z 131", "#d3232f", "Dreieck mit Ampel."),
    ("warnFussgaenger", "Fussgaenger", "Z 101-11/133", "#d3232f", "Dreieck mit Fussgaenger."),
    ("warnKinder", "Kinder", "Z 136", "#d3232f", "Dreieck mit laufenden Kindern."),
    ("warnRadverkehr", "Radverkehr", "Z 138", "#d3232f", "Dreieck mit Fahrrad."),
    ("warnTiereWildwechsel", "Tiere/Wildwechsel", "Z 142", "#d3232f", "Dreieck mit Tier."),
    ("warnGegenverkehr", "Gegenverkehr", "Z 125", "#d3232f", "Dreieck mit zwei Pfeilen."),
    ("warnStau", "Stau", "Z 124", "#d3232f", "Dreieck mit Fahrzeugreihe."),
    ("warnSteinschlag", "Steinschlag", "Z 101-15", "#d3232f", "Dreieck mit Steinen."),
    ("warnBahnuebergang", "Bahnuebergang", "Z 151", "#d3232f", "Dreieck mit Schranke."),
    ("warnSonstiges", "Gefahrzeichen (sonstiges)", "Z 1xx", "#d3232f", "Dreieck, seltenes Symbol."),
    ("gebotRechts", "Vorgeschrieben rechts", "Z 209-20", "#1467b8", "Blauer Kreis, Pfeil rechts."),
    ("gebotLinks", "Vorgeschrieben links", "Z 209-10", "#1467b8", "Blauer Kreis, Pfeil links."),
    ("gebotGeradeaus", "Vorgeschrieben geradeaus", "Z 209-30", "#1467b8", "Blauer Kreis, Pfeil geradeaus."),
    ("gebotGeradeausSeitlich", "Geradeaus oder abbiegen", "Z 214", "#1467b8", "Blauer Kreis, zwei Pfeile."),
    ("gebotVorbeifahrt", "Vorgeschriebene Vorbeifahrt", "Z 222", "#1467b8", "Blauer Kreis, Pfeil am Hindernis."),
    ("kreisverkehr", "Kreisverkehr", "Z 215", "#1467b8", "Blauer Kreis mit Kreispfeilen."),
    ("radweg", "Radweg", "Z 237", "#1467b8", "Blauer Kreis mit Fahrrad."),
    ("gehweg", "Gehweg", "Z 239", "#1467b8", "Blauer Kreis mit Fussgaengern."),
    ("gemeinsamerGehUndRadweg", "Geh- und Radweg", "Z 240", "#1467b8", "Blauer Kreis, Fahrrad und Fussgaenger."),
    ("getrennterRadGehweg", "Getrennter Rad-/Gehweg", "Z 241", "#1467b8", "Blauer Kreis, geteilt."),
    ("fussgaengerzone", "Fussgaengerzone", "Z 242.1", "#1467b8", "Rechteck, Fussgaenger und Kind."),
    ("fahrradstrasse", "Fahrradstrasse", "Z 244.1", "#1467b8", "Rechteck, Fahrrad."),
    ("einbahnstrasse", "Einbahnstrasse", "Z 220", "#1467b8", "Blauer Pfeil im Rechteck."),
    ("haltestelle", "Haltestelle", "Z 224", "#f2c200", "Gelber Balken mit H."),
    ("parken", "Parken", "Z 314", "#1467b8", "Blaues Rechteck mit P."),
    ("autobahn", "Autobahn/Kraftfahrstrasse", "Z 330/331", "#1467b8", "Blau, weisse Strasse."),
    ("tunnel", "Tunnel", "Z 327", "#1467b8", "Blau, Tunneleinfahrt."),
    ("sackgasse", "Sackgasse", "Z 357", "#1467b8", "Blau, T-foermige Strasse."),
    ("umleitung", "Umleitung", "Z 4xx", "#f2c200", "Gelber Wegweiser mit Pfeil."),
    ("ortstafel", "Ortstafel", "Z 310", "#f2c200", "Gelbes Rechteck, Ortsname."),
    ("hinweisSonstiges", "Hinweiszeichen (sonstiges)", "Z 3xx/1xxx", "#1467b8", "Beschriftetes Hinweiszeichen."),
]
INFO: dict[str, dict] = {
    label: {"name": name, "zeichen": zeichen, "hex": hex_, "note": note}
    for label, name, zeichen, hex_, note in INFO_ROWS
}

# ---------------------------------------------------------------------------
# Quelle 1: GTSRB (ClassId 0..42). Die 43 Klassen sind der kleinste gemeinsame Nenner.
# ---------------------------------------------------------------------------
GTSRB_MAP: dict[int, str] = {
    0: "tempo20", 1: "tempo30", 2: "tempo50", 3: "tempo60", 4: "tempo70", 5: "tempo80",
    6: "tempoEnde", 7: "tempo100", 8: "tempo120",
    9: "ueberholverbot", 10: "ueberholverbotKfz",
    11: "vorfahrtGewaehren", 12: "vorfahrtstrasse", 13: "vorfahrtGewaehren", 14: "stop",
    15: "verbotFahrzeuge", 16: "verbotFahrzeuge", 17: "einfahrtVerboten",
    18: "warnGefahrstelle", 19: "warnKurve", 20: "warnKurve", 21: "warnDoppelkurve",
    22: "warnUnebeneFahrbahn", 23: "warnSchleuder", 24: "warnVerengung",
    25: "warnArbeitsstelle", 26: "warnLichtzeichen", 27: "warnFussgaenger",
    28: "warnKinder", 29: "warnRadverkehr", 30: "warnGlaette", 31: "warnTiereWildwechsel",
    32: "tempoEnde",
    33: "gebotRechts", 34: "gebotLinks", 35: "gebotGeradeaus",
    36: "gebotGeradeausSeitlich", 37: "gebotGeradeausSeitlich",
    38: "gebotVorbeifahrt", 39: "gebotVorbeifahrt", 40: "kreisverkehr",
    41: "ueberholverbotEnde", 42: "ueberholverbotEnde",
}

# ---------------------------------------------------------------------------
# Quelle 2: GTSIGN-220 (StVO-Nummern). Vollstaendige Nummer zuerst, dann Hauptnummer.
# Neue Nummern brauchen keine Code-Aenderung, solange die Hauptnummer stimmt - deshalb
# sind die Tabellen unten nach Nummernbereich geordnet, nicht nach Klasse.
# ---------------------------------------------------------------------------
TEMPO_NUMMERN = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130)

STVO_EXAKT: dict[str, str] = {
    # Vorfahrt und Halt
    "201": "andreaskreuz", "205": "vorfahrtGewaehren", "208": "vorfahrtGewaehren",
    "206": "stop", "306": "vorfahrtstrasse", "307": "vorfahrtstrasseEnde",
    # Tempolimits: Ende und Zonen
    "278": "tempoEnde", "279": "tempoEnde", "282": "tempoEnde",
    "274.1": "zone30", "274.1-20": "zone20", "274.1-30": "zone30",
    "274.2": "zoneEnde", "274.2-20": "zoneEnde", "274.2-30": "zoneEnde",
    "275": "mindestgeschwindigkeit",
    # Ueberholverbot
    "276": "ueberholverbot", "277": "ueberholverbotKfz",
    "280": "ueberholverbotEnde", "281": "ueberholverbotEnde",
    # Verbote
    "250": "verbotFahrzeuge", "251": "verbotFahrzeuge", "253": "verbotFahrzeuge",
    "254": "verbotRadverkehr", "259": "verbotFussgaenger", "267": "einfahrtVerboten",
    "272": "verbotWenden",
    # Gebote und Rad-/Fusswege
    "209": "gebotRechts", "209-10": "gebotLinks", "209-20": "gebotRechts",
    "209-30": "gebotGeradeaus", "211": "gebotRechts", "211-10": "gebotLinks",
    "214": "gebotGeradeausSeitlich", "214-10": "gebotGeradeausSeitlich",
    "214-30": "gebotGeradeausSeitlich", "215": "kreisverkehr", "222": "gebotVorbeifahrt",
    "237": "radweg", "239": "gehweg", "240": "gemeinsamerGehUndRadweg",
    "241": "getrennterRadGehweg", "242.1": "fussgaengerzone", "244.1": "fahrradstrasse",
    # Richtzeichen
    "220": "einbahnstrasse", "220-10": "einbahnstrasse", "220-20": "einbahnstrasse",
    "224": "haltestelle", "224-51": "haltestelle", "310": "ortstafel", "314": "parken",
    "327": "tunnel", "330": "autobahn", "331": "autobahn", "333": "autobahn",
    "334": "autobahn", "357": "sackgasse", "358": "sackgasse", "359": "sackgasse",
    "360": "sackgasse",
}

# Gefahrzeichen: Hauptnummer -> Warnklasse (Symbol im Dreieck).
STVO_WARN: dict[str, str] = {
    "102": "warnKreuzung", "103": "warnKurve", "105": "warnDoppelkurve",
    "112": "warnUnebeneFahrbahn", "114": "warnSchleuder", "116": "warnGlaette",
    "120": "warnVerengung", "121": "warnVerengung", "123": "warnArbeitsstelle",
    "124": "warnStau", "125": "warnGegenverkehr", "131": "warnLichtzeichen",
    "133": "warnFussgaenger", "136": "warnKinder", "138": "warnRadverkehr",
    "142": "warnTiereWildwechsel", "151": "warnBahnuebergang",
    # Z 101 mit Unternummer: die Aufstellungsseite (rechts/links) ist unerheblich
    "101-11": "warnFussgaenger", "101-21": "warnFussgaenger",
    "101-12": "warnTiereWildwechsel", "101-22": "warnTiereWildwechsel",
    "101-13": "warnTiereWildwechsel", "101-23": "warnTiereWildwechsel",
    "101-14": "warnTiereWildwechsel", "101-24": "warnTiereWildwechsel",
    "101-15": "warnSteinschlag", "101-25": "warnSteinschlag",
    "101-51": "warnGlaette", "101-52": "warnGlaette",
}

# Halteverbote stehen unter mehreren Nummern (Z 283 ff. und Zusatzzeichen 1053-*).
STVO_HALTEVERBOT = {"283", "286", "290", "292", "293", "294", "295", "296", "297",
                    "299", "299.1", "1053-31", "1053-33", "1053-34", "1053-35"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def label_for_gtsrb(class_id: int) -> str | None:
    """GTSRB-ClassId (0..42) -> Label dieses Projekts."""
    return GTSRB_MAP.get(int(class_id))


def _stvo_einzeln(token: str) -> str | None:
    """EIN Token wie '274-70', '101-11' oder '1053-34' -> Label (oder None)."""
    if not token or token == "-":
        return None
    if token in STVO_EXAKT:
        return STVO_EXAKT[token]
    if token in STVO_WARN:
        return STVO_WARN[token]
    if token.startswith("274"):
        # Z 274 mit Unternummer: der Zahlenwert steht hinter dem letzten Bindestrich.
        rest = token.split("-")[-1]
        if rest.isdigit() and int(rest) in TEMPO_NUMMERN:
            return f"tempo{int(rest)}"
        return None
    if token in STVO_HALTEVERBOT:
        return "halteverbot"
    haupt = token.split("-")[0]
    if haupt in STVO_EXAKT:
        return STVO_EXAKT[haupt]
    if haupt in STVO_WARN:
        return STVO_WARN[haupt]
    if haupt in STVO_HALTEVERBOT:
        return "halteverbot"
    if not haupt.split(".")[0].isdigit():
        return None
    nummer = int(haupt.split(".")[0])
    if nummer == 101:
        return "warnGefahrstelle"
    if 102 <= nummer <= 159:
        return "warnSonstiges"
    if 200 <= nummer <= 299:
        return "verbotSonstiges"
    if 300 <= nummer <= 499:
        return "hinweisSonstiges"
    if nummer >= 1000:
        return "hinweisSonstiges"      # Zusatzzeichen (Z 1xxx)
    return None


def label_for_stvo(number: str) -> str | None:
    """StVO-Nummer(nfeld) aus GTSIGN-220 -> Label. Mehrere Nummern sind mit ';' getrennt.

    Die erste zuordenbare Nummer gewinnt: die Katalogzeile nennt zuerst das eigentliche
    Zeichen, danach Aufstellungsvarianten (rechts/links) und Zusatzzeichen.
    """
    for token in re.split(r"[;,]", str(number or "")):
        treffer = _stvo_einzeln(_norm(token))
        if treffer:
            return treffer
    return None


# ---------------------------------------------------------------------------
# Quelle 3: Synset Signset Germany (deutsche Klassennamen aus der Karte).
# Erkannt wird erkannt: die Namen sind systematisch gebildet, deshalb genuegen wenige
# Praefix-Regeln - und die decken dann auch Namen ab, die es heute noch nicht gibt.
# ---------------------------------------------------------------------------
SYNSET_EXAKT: dict[str, str] = {
    "Stop": "stop", "Vorfahrt": "vorfahrtGewaehren", "Vorfahrtsstrasse": "vorfahrtstrasse",
    "VorfahrtGewaehren": "vorfahrtGewaehren", "VorrangDesGegenverkehrs": "vorfahrtGewaehren",
    "VorrangVorGegenverkehr": "vorfahrtGewaehren", "Andreaskreuz": "andreaskreuz",
    "Ueberholverbot": "ueberholverbot",
    "UeberholverbotKraftfahrzeuge": "ueberholverbotKfz",
    "VerbotUeberholenEinspurigeFahrzeuge": "ueberholverbot",
    "VerbotDerEinfahrt": "einfahrtVerboten", "VerbotFussgaenger": "verbotFussgaenger",
    "VerbotRadverkehr": "verbotRadverkehr", "VerbotWenden": "verbotWenden",
    "VerbotFahrzeugeAllerArt": "verbotFahrzeuge", "VerbotKraftfahrzeuge": "verbotFahrzeuge",
    "VerbotKraftwagen": "verbotFahrzeuge",
    "VerbotFuerKraftfahrzeugeUeberDreiKommaFuenfTonnen": "verbotFahrzeuge",
    "KFZZulaessigeGesamtmasse3Komma5TOhnePfeilsymbol": "verbotFahrzeuge",
    "AbsolutesHalteverbot": "halteverbot", "EingeschraenktesHalteverbot": "halteverbot",
    "BeginnEingeschraenktesHalteverbotZone": "halteverbot",
    "EndeEingeschraenktesHalteverbotZone": "halteverbot",
    "Radweg": "radweg", "Gehweg": "gehweg", "GemeinsamerGehUndRadweg": "gemeinsamerGehUndRadweg",
    "Kreisverkehr": "kreisverkehr", "Haltestelle": "haltestelle", "Parken": "parken",
    "Parkhaus": "parken", "Taxenstand": "parken", "Tunnel": "tunnel",
    "Autobahn": "autobahn", "Kraftfahrstrasse": "autobahn", "EndeAutbahn": "autobahn",
    "EndeKraftfahrstrasse": "autobahn", "EndeUmleitung": "umleitung",
    "AnkuendigungOderFortsetzungUmleitungOhnePfeilsymbol": "umleitung",
    "EndeDerUmleitungsankuendigung": "umleitung",
    "Gefahrenstelle": "warnGefahrstelle", "KreuzungOderEinmuendung": "warnKreuzung",
    "UnebeneFahrbahn": "warnUnebeneFahrbahn", "SchleuderOderRutschgefahr": "warnSchleuder",
    "VerengteFahrbahn": "warnVerengung", "Arbeitsstelle": "warnArbeitsstelle",
    "Lichtzeichenanlage": "warnLichtzeichen", "Stau": "warnStau",
    "Gegenverkehr": "warnGegenverkehr", "Bahnuebergang": "warnBahnuebergang",
    "SchneeOderEisglaette": "warnGlaette",
    "Verkehrshelfer": "hinweisSonstiges", "Reitweg": "hinweisSonstiges",
    "NothalteUndPannenbucht": "hinweisSonstiges", "Ufer": "warnSonstiges",
    "SplittSchotter": "warnGlaette", "Gefaelle10": "warnSonstiges",
    "Steigung10": "warnSonstiges", "UnzureichendesLichtraumprofil": "warnSonstiges",
    "BeweglicheBruecke": "warnSonstiges",
    "EndeFussgaengerzone": "hinweisSonstiges", "EndeFahrradstrasse": "hinweisSonstiges",
    "EndeFahrradzone": "hinweisSonstiges",
    # Beginn-Zeichen der Zonen (die Praefix-Regeln greifen hier nicht, weil "Beginn" davor steht)
    "BeginnFussgaengerzone": "fussgaengerzone", "BeginnFahrradstrasse": "fahrradstrasse",
    "BeginnFahrradzone": "fahrradstrasse",
    # Warnzeichen Radverkehr (das Pflichtzeichen heisst dagegen schlicht "Radweg")
    "RadverkehrRechts": "warnRadverkehr", "RadverkehrLinks": "warnRadverkehr",
}

# Praefix-Regeln in Pruefreihenfolge. Die Reihenfolge ist Teil der Logik: "EndeGeschwindigkeit"
# muss vor "Geschwindigkeit" stehen, sonst wird aus dem Ende ein Limit.
SYNSET_PRAEFIX: list[tuple] = [
    ("EndeGeschwindigkeit", None),          # Sonderfall: Zahlenwert wird abgeschnitten
    ("EndeSaemtlicherBegrenzungen", "zoneEnde"),
    ("EndeZone", "zoneEnde"),
    ("VerlaufVorfahrtsstrasse", "vorfahrtstrasse"),
    ("EndeVorfahrtsstrasse", "vorfahrtstrasseEnde"),
    ("EndeUeberholverbot", "ueberholverbotEnde"),
    ("VorgeschriebeneFahrtrichtung", None),
    ("VorgeschriebeneVorbeifahrt", "gebotVorbeifahrt"),
    ("VorgeschriebeneMindestgeschwindigkeit", "mindestgeschwindigkeit"),
    ("Verbot", None),
    ("Fussgaengerueberweg", "warnFussgaenger"),
    ("Fussgaenger", "warnFussgaenger"),
    ("Kinder", "warnKinder"),
    ("Wildwechsel", "warnTiereWildwechsel"),
    ("Viehtrieb", "warnTiereWildwechsel"),
    ("Reiter", "warnTiereWildwechsel"),
    ("Amphibienwanderung", "warnTiereWildwechsel"),
    ("Steinschlag", "warnSteinschlag"),
    ("Flugbetrieb", "warnSonstiges"),
    ("Seitenwind", "warnSonstiges"),
    ("Kurve", "warnKurve"),
    ("Doppelkurve", "warnDoppelkurve"),
    ("EinseitigVerengteFahrbahn", "warnVerengung"),
    ("Umleitungs", "umleitung"),
    ("Richtungstafeln", "umleitung"),
    ("Einbahnstrasse", "einbahnstrasse"),
    ("Sackgasse", "sackgasse"),
    ("GetrennterRadUndGehweg", "getrennterRadGehweg"),
    ("Fussgaengerzone", "fussgaengerzone"),
    ("Fahrradstrasse", "fahrradstrasse"),
    ("Fahrradzone", "fahrradstrasse"),
]

# Richtungsvarianten bei Z 209/211/214: der Rest hinter dem Praefix ist die Richtung.
FAHRTRICHTUNG: dict[str, str] = {
    "Rechts": "gebotRechts", "Links": "gebotLinks", "Geradeaus": "gebotGeradeaus",
    "GeradeausOderRechts": "gebotGeradeausSeitlich", "GeradeausOderLinks": "gebotGeradeausSeitlich",
    "RechtsOderLinks": "gebotGeradeausSeitlich", "HierRechts": "gebotRechts",
    "HierLinks": "gebotLinks", "": "gebotRechts",
}


def label_for_synset(class_name: str) -> str | None:
    """Deutscher Klassenname aus Synset Signset Germany -> Label dieses Projekts."""
    name = (class_name or "").strip()
    if not name:
        return None
    if name in SYNSET_EXAKT:
        return SYNSET_EXAKT[name]
    # Zahlenwerte zuerst: Geschwindigkeit70 -> tempo70, EndeGeschwindigkeit80 -> tempoEnde
    m = re.fullmatch(r"Geschwindigkeit(\d+)", name)
    if m:
        return f"tempo{int(m.group(1))}" if int(m.group(1)) in TEMPO_NUMMERN else None
    if re.fullmatch(r"EndeGeschwindigkeit\d+", name):
        return "tempoEnde"
    m = re.fullmatch(r"BeginnZone(\d+)", name)
    if m:
        return f"zone{m.group(1)}" if int(m.group(1)) in (20, 30) else None
    m = re.fullmatch(r"EndeZone(\d+)", name)
    if m:
        return "zoneEnde"
    # Richtungszeichen: Praefix abschneiden und den Rest in der Tabelle nachschlagen
    for praefix, ziel in SYNSET_PRAEFIX:
        if not name.startswith(praefix):
            continue
        if ziel:
            return ziel
        rest = name[len(praefix):]
        if praefix == "Verbot":
            return "verbotSonstiges"            # Verbot eines seltenen Fahrzeugtyps
        return FAHRTRICHTUNG.get(rest, "gebotRechts")
    if name.startswith("Verbot"):
        return "verbotSonstiges"
    if name.startswith("Ende"):
        return "hinweisSonstiges"               # "Ende von ..." ohne eigene Klasse
    return "hinweisSonstiges"                   # beschriftete Hinweiszeichen (Z 3xx/1xxx)


# ---------------------------------------------------------------------------
# Selbstpruefung: LABELS, INFO und die Zuordnungen muessen zusammenpassen.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Quelle 4: GTSDB (deutsche Strassenszenen, COCO-Fassung von Roboflow). Die Kategorie-
# namen sind englisch beschreibend - deshalb eine eigene Tabelle. Diese Quelle ist die
# einzige mit ECHTEN Szenen: hier stehen die Schilder klein im Bild, nicht als Ausschnitt.
# ---------------------------------------------------------------------------
GTSDB_NAME: dict[str, str] = {
    "animals": "warnTiereWildwechsel", "construction": "warnArbeitsstelle",
    "cycles crossing": "warnRadverkehr", "danger": "warnGefahrstelle",
    "no entry": "einfahrtVerboten", "pedestrian crossing": "warnFussgaenger",
    "school crossing": "warnKinder", "snow": "warnGlaette", "stop": "stop",
    "bend": "warnKurve", "bend left": "warnKurve", "bend right": "warnKurve",
    "give way": "vorfahrtGewaehren", "go left": "gebotLinks",
    "go left or straight": "gebotGeradeausSeitlich", "go right": "gebotRechts",
    "go right or straight": "gebotGeradeausSeitlich", "go straight": "gebotGeradeaus",
    "keep left": "gebotVorbeifahrt", "keep right": "gebotVorbeifahrt",
    "no overtaking": "ueberholverbot", "no overtaking -trucks-": "ueberholverbotKfz",
    "no traffic both ways": "verbotFahrzeuge", "no trucks": "verbotFahrzeuge",
    "priority at next intersection": "warnKreuzung", "priority road": "vorfahrtstrasse",
    "restriction ends": "tempoEnde", "restriction ends 80": "tempoEnde",
    "restriction ends -overtaking-": "ueberholverbotEnde",
    "restriction ends -overtaking -trucks--": "ueberholverbotEnde",
    "road narrows": "warnVerengung", "roundabout": "kreisverkehr",
    "slippery road": "warnSchleuder",
    **{f"speed limit {v}": f"tempo{v}" for v in TEMPO_NUMMERN},
    "traffic signal": "warnLichtzeichen", "uneven road": "warnUnebeneFahrbahn",
}


def label_for_gtsdb(category_name: str) -> str | None:
    """Kategoriename der COCO-Fassung von GTSDB -> Label dieses Projekts."""
    return GTSDB_NAME.get((category_name or "").strip().lower())


def pruefen() -> list[str]:
    """Widersprueche in dieser Datei finden (leere Liste = alles in Ordnung)."""
    fehler: list[str] = []
    if len(set(LABELS)) != len(LABELS):
        fehler.append("LABELS enthaelt Doppelte")
    if [r[0] for r in INFO_ROWS] != LABELS:
        fehlt = [x for x in LABELS if x not in {r[0] for r in INFO_ROWS}]
        extra = [r[0] for r in INFO_ROWS if r[0] not in CLASS_ID]
        fehler.append(f"INFO_ROWS passt nicht zu LABELS (fehlt: {fehlt}, extra: {extra})")
    for quelle, ziele in (("GTSRB", GTSRB_MAP.values()), ("StVO", STVO_EXAKT.values()),
                          ("StVO-Warnung", STVO_WARN.values()),
                          ("Synset", SYNSET_EXAKT.values()),
                          ("Synset-Praefix", [z for _, z in SYNSET_PRAEFIX if z]),
                          ("Fahrtrichtung", FAHRTRICHTUNG.values()),
                          ("GTSDB", GTSDB_NAME.values())):
        unbekannt = sorted({z for z in ziele if z not in CLASS_ID})
        if unbekannt:
            fehler.append(f"{quelle} verweist auf unbekannte Labels: {unbekannt}")
    # Hierarchie (tools/signmap.py oben): jede Unterkategorie genau EINER Oberkategorie
    # zugeordnet. Ohne diese Pruefung faellt ein Tippfehler in SUPER_GRUPPEN erst beim
    # Training auf - dann als eine Klasse, die nie richtig erkannt wird, weil ihr
    # Familien-Logit nie ein Ziel bekommt (-1).
    ohne = [LABELS[i] for i, v in enumerate(SUPER_OF) if v < 0]
    if ohne:
        fehler.append(f"SUPER_GRUPPEN deckt diese Labels nicht ab: {ohne}")
    doppelt = sorted({n for g in SUPER_GRUPPEN.values() for n in g
                      if sum(g2.count(n) for g2 in SUPER_GRUPPEN.values()) > 1})
    if doppelt:
        fehler.append(f"SUPER_GRUPPEN enthaelt Labels mehrfach: {doppelt}")
    if len({v for v in SUPER_OF if v >= 0}) != N_SUPER:
        fehler.append("mindestens eine Oberkategorie hat keine Unterkategorie")
    return fehler


def bericht_gtsign(csv_pfad: Path) -> None:
    rows = list(csv.DictReader(csv_pfad.open(encoding="utf-8")))
    treffer: dict[str, int] = {}
    offen: list[str] = []
    for r in rows:
        label = label_for_stvo(r["StVO_Sign_Number"])
        if label:
            treffer[label] = treffer.get(label, 0) + 1
        else:
            offen.append(f"  {r['Class_ID']:>4}  {r['StVO_Sign_Number'][:28]:28s} "
                         f"{r['Description'][:44]}")
    print(f"[GTSIGN] {len(rows)} Katalogzeilen -> {len(treffer)} von {N_LABELS} Labels belegt")
    print(f"[GTSIGN] {len(offen)} Zeilen ohne Label (werden verworfen):")
    print("\n".join(offen) if offen else "  (keine)")


def bericht_synset(synset_pfad: Path) -> None:
    namen = json.loads(synset_pfad.read_text(encoding="utf-8"))
    if isinstance(namen, dict):
        namen = list(namen.values())
    treffer: dict[str, int] = {}
    offen: list[str] = []
    for name in namen:
        label = label_for_synset(name)
        if label:
            treffer[label] = treffer.get(label, 0) + 1
        else:
            offen.append(f"  {name}")
    belegt = {l: 0 for l in LABELS}
    for name in namen:
        b = label_for_synset(name)
        if b:
            belegt[b] += 1
    leer = [l for l, v in belegt.items() if v == 0]
    print(f"[Synset] {len(namen)} Klassennamen -> {len(treffer)} von {N_LABELS} Labels belegt")
    print(f"[Synset] {len(offen)} Namen ohne Label (werden verworfen):")
    print("\n".join(offen) if offen else "  (keine)")
    print(f"[Synset] Labels ohne Synset-Daten ({len(leer)}): {', '.join(leer)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Klassenzuordnung der Datenquellen pruefen")
    ap.add_argument("--gtsign-csv", default="data/gtsign/class_descriptions_and_stvo.csv")
    ap.add_argument("--synset-json", default="data/synset-klassen.json")
    ap.add_argument("--labels", action="store_true", help="nur die Klassenliste ausgeben")
    args = ap.parse_args()

    fehler = pruefen()
    if fehler:
        for f in fehler:
            print(f"[fehler] {f}")
        raise SystemExit(1)
    print(f"[ok] {N_LABELS} Klassen, INFO_ROWS vollstaendig, alle Zuordnungen bekannt")
    if args.labels:
        for i, name in enumerate(LABELS):
            print(f"{i:3d}  {name:26s} {INFO[name]['name']}")
        return
    p = Path(args.gtsign_csv)
    if p.exists():
        bericht_gtsign(p)
    else:
        print(f"[hinweis] {p} fehlt - GTSIGN-Bericht uebersprungen")
    p = Path(args.synset_json)
    if p.exists():
        bericht_synset(p)
    else:
        print(f"[hinweis] {p} fehlt - Synset-Bericht uebersprungen")


if __name__ == "__main__":
    main()