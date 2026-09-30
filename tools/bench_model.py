"""
tools/bench_model.py - Was kostet der Transformer-Einbau? (Parameter, FLOPs, Latenz)

Beantwortet die Frage "Transformer ja, aber in Echtzeit" mit Messwerten statt Gefuehl.
Gemessen wird auf dieser Maschine (CPU, PyTorch). Die Zahlen sind ein OBERER Rahmen
fuer das Handy: ONNX Runtime Web mit WebGPU ist meist schneller als CPU-torch,
ein einthreadiges WASM auf ARM ist langsamer. Deshalb wird zusaetzlich mit
1 Thread gemessen (Naeherung fuer einthreadiges WASM auf dem Handy).

Aufruf:  python tools/bench_model.py            (Standardvergleich)
         python tools/bench_model.py --size 256 (andere Eingabegroesse)
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from hybrid_net import PRESETS, NetCfg, build_model, n_params, preset_cfg

# Varianten: zeigt, was der Transformer an welcher Stelle kostet
VARIANTS = {
    "cnn-only":        dict(tr_global=(), tr_window=()),
    "p5-global":       dict(tr_global=("p5",), tr_window=()),
    "p5g+p4window":    dict(tr_global=("p5",), tr_window=("p4",)),
    "p4g+p5g":         dict(tr_global=("p4", "p5"), tr_window=()),
    "p5g+p4w+lnnorm":  dict(tr_global=("p5",), tr_window=("p4",), tr_norm="ln"),
    "p5g+p4w+p3w":     dict(tr_global=("p5",), tr_window=("p4", "p3")),
    "p5g-hardswish":   dict(tr_global=("p5",), tr_window=(), act="hardswish"),
}


def measure(cfg: NetCfg, size: int = 320, iters: int = 15, threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    model = build_model(cfg).eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        model(x)                                    # Warmlauf
        with FlopCounterMode(display=False) as fc:
            model(x)
        mflops = fc.get_total_flops() / 1e6
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        ms = (time.perf_counter() - t0) / iters * 1000.0
    params = n_params(model)
    return {
        "params": params,
        "mb_int8": params / 1e6,                    # 1 Byte je Gewicht (int8) + kleiner Aufschlag
        "mflops": mflops,
        "ms": ms,
        "threads": threads,
    }


def preset_rows(preset: str, size: int, iters: int) -> dict:
    return measure(preset_cfg(preset), size, iters, threads=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kosten der Transformer-Varianten messen")
    ap.add_argument("--size", type=int, default=320, help="Eingabegroesse (Default 320)")
    ap.add_argument("--iters", type=int, default=5, help="Durchlaeufe je Messung (Default 5)")
    ap.add_argument("--only", default="", help="nur Varianten mit diesem Teilstring messen")
    ap.add_argument("--preset", action="store_true", help="die Presets fast/balanced/quality vergleichen")
    ap.add_argument("--scaling", action="store_true", help="zusaetzlich die Attention-Skalierung messen")
    args = ap.parse_args()

    print(f"Eingabe {args.size}x{args.size}, CPU-PyTorch {torch.__version__}")
    if args.preset:
        print("| Preset | Params | int8-Datei | MFLOPs | ms (1 Thread) |")
        print("|---|---|---|---|---|")
        for name in PRESETS:
            r = preset_rows(name, args.size, args.iters)
            print(f"| {name} | {r['params']/1e3:.0f}k | ~{r['mb_int8']:.2f} MB | "
                  f"{r['mflops']:.0f} | {r['ms']:.1f} |")
        return

    print("| Variante | Params | int8-Datei | MFLOPs | ms (4 Threads) | ms (1 Thread) |")
    print("|---|---|---|---|---|---|")
    for name, kw in VARIANTS.items():
        if args.only and args.only not in name:
            continue
        cfg = NetCfg(**kw)
        r4 = measure(cfg, args.size, args.iters, threads=4)
        r1 = measure(cfg, args.size, max(3, args.iters // 2), threads=1)
        print(f"| {name} | {r4['params']/1e3:.0f}k | ~{r4['mb_int8']:.2f} MB | "
              f"{r4['mflops']:.0f} | {r4['ms']:.1f} | {r1['ms']:.1f} |")

    if not args.scaling:
        return
    # Skalierung der Attention: gleiche Variante, andere Eingabegroesse
    print("\nSkalierung der Attention (gleiche Variante, andere Eingabegroesse):")
    for size in (192, 320, 416):
        a = measure(NetCfg(tr_global=(), tr_window=()), size, 4)
        b = measure(NetCfg(tr_global=("p5",), tr_window=("p4",)), size, 4)
        print(f"  {size:3d}px: cnn-only {a['ms']:6.1f} ms | +p5g+p4w {b['ms']:6.1f} ms "
              f"(+{100*(b['ms']/a['ms']-1):5.1f} %) | Tokens p4={(size//16)**2:4d} p5={(size//32)**2:3d}")


if __name__ == "__main__":
    main()
