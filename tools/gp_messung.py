"""Vergleicht die 5x5-Tiefenconv gegen die dilatierte 3x3 im GranularPerception-Zweig.

Nur zum Messen (nicht ausgeliefert). Die beiden Varianten sind baugleich bis auf den zweiten
Koernungs-Zweig; gemessen werden Parameter, MFLOPs und Laufzeit auf derselben Maschine.

Aufruf:  python tools/gp_messung.py [--size 384]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import hybrid_net as hn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402


def varianten(netz: hn.HybridNano, k: int, d: int) -> None:
    """Den zweiten Zweig aller GranularPerception-Module umbauen (k/d wie angegeben)."""
    for m in netz.modules():
        if isinstance(m, hn.GranularPerception):
            m.dw5 = hn.cba(m.dw5[0].in_channels, m.dw5[0].out_channels, k, 1,
                           g=m.dw5[0].in_channels, act="silu", d=d)


def messen(preset: str, size: int, k: int, d: int, iters: int) -> tuple:
    cfg = hn.preset_cfg(preset)
    netz = hn.build_model(cfg).eval()
    varianten(netz, k, d)
    x = torch.zeros(1, 3, size, size)
    with FlopCounterMode(display=False) as z:
        with torch.no_grad():
            netz(x)
    flops = z.get_total_flops() / 1e6
    with torch.no_grad():
        netz(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            netz(x)
        ms = (time.perf_counter() - t0) / iters * 1000
    return hn.n_params(netz), flops, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    print(f"Eingabe {args.size}x{args.size}, {args.iters} Durchlaeufe")
    print("| Zweig | Params | MFLOPs | ms |")
    print("|---|---|---|---|")
    ergebnis = {}
    for name, k, d in (("5x5 (vorher)", 5, 1), ("3x3 d=2 (jetzt)", 3, 2)):
        p, f, ms = messen("breit", args.size, k, d, args.iters)
        ergebnis[name] = (p, f, ms)
        print(f"| {name} | {p/1e3:.0f}k | {f:.0f} | {ms:.1f} |")
    alt, neu = ergebnis["5x5 (vorher)"], ergebnis["3x3 d=2 (jetzt)"]
    print()
    print(f"MFLOPs: {alt[1]:.0f} -> {neu[1]:.0f}  ({(neu[1]/alt[1]-1)*100:+.1f} %)")
    print(f"Parameter: {alt[0]/1e3:.0f}k -> {neu[0]/1e3:.0f}k "
          f"({(neu[0]-alt[0])/1e3:+.0f}k, die 3x3 hat weniger Gewichte)")
    print(f"Laufzeit: {alt[2]:.1f} -> {neu[2]:.1f} ms ({(neu[2]/alt[2]-1)*100:+.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())