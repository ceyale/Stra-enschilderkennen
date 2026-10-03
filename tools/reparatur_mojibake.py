"""Repariert doppelt kodierte Umlaute in Textdateien (Mojibake).

Hintergrund: ein PowerShell-`Get-Content -Raw` + `Set-Content -Encoding utf8` hat UTF-8-Dateien
als ANSI gelesen und als UTF-8 zurueckgeschrieben ("außerdem" -> "auÃŸerdem"). Diese Reparatur
macht genau das rueckgaengig: Datei als UTF-8 lesen, die Zeichen als Windows-1252-Bytes
deuten, diese Bytes als UTF-8 lesen.

Aufruf:  python tools/reparatur_mojibake.py --pfad CHANGELOG.md docs/TRAINING.md [--trocken]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def reparieren(text: str) -> str:
    """Mojibake rueckgaengig machen - ZEILENWEISE, damit eine unreparierbare Zeile nicht alles
    blockiert (der erste Versuch scheiterte genau daran: ein Sonderzeichen ausserhalb cp1252
    liess `encode('cp1252')` fuer die GANZE Datei werfen, und die Datei blieb kaputt)."""
    # Die haeufigsten Faelle zusaetzlich als feste Zuordnung: sie greift auch dann, wenn die
    # Byte-Runde wegen eines einzelnen Zeichens nicht moeglich ist.
    fest = {"ÃŸ": "ß", "Ã¤": "ä", "Ã¶": "ö", "Ã¼": "ü", "Ã„": "Ä", "Ã–": "Ö", "Ãœ": "Ü",
            "â€ž": "„", "â€œ": "“", "â€“": "–", "â€”": "—", "â€¦": "…", "â‚¬": "€",
            "â€š": "‚", "â€¹": "‹", "Â°": "°", "Âµ": "µ", "Â²": "²", "Â³": "³"}
    aus = []
    for zeile in text.split("\n"):
        neu = zeile
        try:
            neu = zeile.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            for alt, gut in fest.items():
                neu = neu.replace(alt, gut)
        aus.append(neu)
    return "\n".join(aus)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfad", nargs="+", required=True)
    ap.add_argument("--trocken", action="store_true", help="nur zeigen, nicht schreiben")
    args = ap.parse_args()
    for p in args.pfad:
        f = Path(p)
        roh = f.read_text(encoding="utf-8")
        neu = reparieren(roh)
        geaendert = neu != roh
        print(f"{f}: {'repariert' if geaendert else 'unveraendert'} "
              f"({len(roh)} -> {len(neu)} Zeichen)")
        if geaendert and not args.trocken:
            f.write_text(neu, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())