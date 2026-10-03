"""Zeigt, WO die Rechnung im Netz steckt (FLOPs je Modul) - Grundlage fuer Sparmassnahmen.

Ohne diese Messung wird an der falschen Stelle gespart: die dilatierte 3x3 im
GranularPerception-Zweig kostete nur 0,9 % der Gesamtrechnung (data/_gp_messung.py),
die Kopf-Staemme und die CATM-Blocks dagegen liegen um ein Vielfaches darueber.

Aufruf:  python tools/flops_wo.py [--preset breit] [--size 384]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import hybrid_net as hn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="breit")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    netz = hn.build_model(hn.preset_cfg(args.preset)).eval()
    x = torch.zeros(1, 3, args.size, args.size)

    def merker(formen: dict, name: str):
        """Pre-Hook-Fabrik: merkt die Eingangsform. Muss None zurueckgeben (siehe oben)."""
        def hook(mod, inp) -> None:
            formen.setdefault(name, tuple(inp[0].shape))
            return None
        return hook

    formen: dict[str, tuple] = {}
    # Auch die Kindmodule von ModuleList/ModuleDict messen (heads.0, lgp.1, ...) - genau dort
    # liegt der groesste Posten, und ein Modul wie 'heads' laesst sich nicht direkt aufrufen.
    ziele = [(n, m) for n, m in netz.named_modules()
             if n and n.count(".") <= 1 and not isinstance(m, (torch.nn.ModuleList,
                                                              torch.nn.ModuleDict))]
    ruten = [m.register_forward_pre_hook(merker(formen, n)) for n, m in ziele]
    with torch.no_grad():
        netz(x)
    for r in ruten:
        r.remove()

    with FlopCounterMode(display=False) as z:
        with torch.no_grad():
            netz(x)
    gesamt = z.get_total_flops() / 1e6

    print(f"Preset {args.preset}, {args.size} px: **{gesamt:.0f} MFLOPs**, "
          f"{hn.n_params(netz)/1e3:.0f}k Parameter")
    print("| Modul | Eingang | Params | MFLOPs | Anteil |")
    print("|---|---|---|---|---|")
    zeilen = []
    for name, mod in ziele:
        form = formen.get(name)
        if form is None:
            continue
        try:
            with FlopCounterMode(display=False) as zz:
                with torch.no_grad():
                    mod(torch.zeros(*form))
            f = zz.get_total_flops() / 1e6
        except Exception:
            continue
        zeilen.append((f, name, form, hn.n_params(mod)))
    for f, name, form, p in sorted(zeilen, reverse=True)[: args.top]:
        print(f"| {name} | {form[1]}x{form[2]} | {p/1e3:.0f}k | {f:.0f} | {f/gesamt*100:.1f} % |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())