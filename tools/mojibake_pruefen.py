"""Prueft alle Textdateien im Repo auf doppelt kodierte Umlaute (Mojibake).

Anlass: ein PowerShell-`Get-Content -Raw` + `Set-Content -Encoding utf8` hat zwei Doku-Dateien
beschaedigt (UTF-8 als ANSI gelesen). Diese Pruefung findet solche Stellen in ALLEN Dateien -
auch in denen, die man gerade nicht angeschaut hat. Reparatur: tools/reparatur_mojibake.py.

Aufruf:  python tools/mojibake_pruefen.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Zeichenfolgen, die in korrektem UTF-8 NICHT vorkommen: die typischen Byte-Paare eines
# UTF-8-Zeichens, das als Windows-1252 gelesen und erneut als UTF-8 geschrieben wurde.
SPUREN = ["\u00c3", "\u00e2\u20ac", "\u00c2\u00b0", "\u00e2\u0082\u00ac"]
ENDUNGEN = {".md", ".py", ".js", ".json", ".html", ".css", ".ps1", ".txt", ".yml", ".yaml"}


def dateien() -> list[Path]:
    aus = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8")
    return [ROOT / z for z in aus.stdout.splitlines() if Path(z).suffix.lower() in ENDUNGEN]


def main() -> int:
    fehler = 0
    geprueft = 0
    for f in dateien():
        if f.suffix.lower() not in ENDUNGEN or not f.exists():
            continue
        geprueft += 1
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"[FEHLER] {f.relative_to(ROOT)}: kein gueltiges UTF-8 ({e})")
            fehler += 1
            continue
        for spur in SPUREN:
            if spur in text:
                stellen = text.count(spur)
                print(f"[FEHLER] {f.relative_to(ROOT)}: {stellen}x Mojibake "
                      f"({spur!r}) - tools/reparatur_mojibake.py hilft")
                fehler += 1
                break
    print(f"[ergebnis] {geprueft} Dateien geprueft, {fehler} mit Mojibake")
    return 1 if fehler else 0


if __name__ == "__main__":
    raise SystemExit(main())