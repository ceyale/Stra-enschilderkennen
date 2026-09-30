"""
tools/build_site.py - Auslieferung nach dist/ sammeln und optional zu Cloudflare Pages schieben.

Warum ein Zwischenschritt? Cloudflare Pages laedt den GANZEN Ausgabeordner hoch. Das Repo
enthaelt aber tools/, data/, tests/ und .git/ - das gehoert nicht ins Netz. dist/ enthaelt
deshalb nur: index.html, src/, models/ (nur die noetigen Modelldateien).

    python tools/build_site.py                      # nur dist/ bauen (Groesse anzeigen)
    python tools/build_site.py --check              # zusaetzlich pruefen, ob ein Modell dabei ist
    python tools/build_site.py --deploy             # zusaetzlich hochladen (braucht wrangler + Token)

Fuer --deploy werden zwei Umgebungsvariablen gebraucht (Cloudflare-Dashboard -> API-Token):
    CLOUDFLARE_API_TOKEN   Token mit Recht "Cloudflare Pages: Edit"
    CLOUDFLARE_ACCOUNT_ID  Kontokennung (rechte Spalte der Uebersicht)
Ohne diese beiden bricht wrangler mit einer Anmeldung ab - der Befehl wird hier nicht
simuliert, sondern mit klarer Meldung beendet.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Zufaelliger Projektname -> die Seite landet unter https://<name>.pages.dev
DEFAULT_PROJECT = "schilderscanner-r7k4m2"


def collect(dist: Path) -> list[tuple[Path, int]]:
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    copied: list[tuple[Path, int]] = []

    for name in ("index.html",):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dist / name)
            copied.append((Path(name), (dist / name).stat().st_size))

    for folder, patterns in (("src", None), ("models", ("*.onnx", "labels.json"))):
        target = dist / folder
        for path in sorted((ROOT / folder).glob("*")):
            if not path.is_file():
                continue
            if patterns and not any(path.match(p) for p in patterns):
                continue
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
            copied.append((Path(folder) / path.name, (target / path.name).stat().st_size))
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="Website-Auslieferung bauen/pruefen/deployen")
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--deploy", action="store_true", help="mit wrangler pages deploy hochladen")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    dist = ROOT / args.dist
    copied = collect(dist)
    total = sum(size for _, size in copied)
    print(f"[build] {len(copied)} Dateien, {total/1e6:.2f} MB -> {dist}")
    for name, size in copied:
        print(f"  {size/1e3:9.1f} kB  {name}")

    model = next((n for n, _ in copied if str(n).endswith(".onnx")), None)
    if model is None:
        print("[warnung] Kein ONNX-Modell in dist/ - die Seite laeuft im Heuristik-Modus.")
    if args.check:
        sys.exit(0 if model else 2)
    if not args.deploy:
        return

    token, account = os.environ.get("CLOUDFLARE_API_TOKEN"), os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        print("[deploy] abgebrochen: CLOUDFLARE_API_TOKEN und CLOUDFLARE_ACCOUNT_ID muessen gesetzt sein.")
        print("         Token: Cloudflare-Dashboard -> My Profile -> API Tokens -> Create Token")
        print("                Vorlage 'Edit Cloudflare Workers' bzw. Recht 'Cloudflare Pages: Edit'")
        sys.exit(3)
    cmd = ["npx", "--yes", "wrangler@latest", "pages", "deploy", str(dist),
           "--project-name", args.project_name, "--branch", args.branch]
    print("[deploy] " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
