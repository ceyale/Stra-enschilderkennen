"""Wegwerf-Test: kann das Netz ein einzelnes Bild ueberfitten?

Beweist, dass Zuweisung (assign_targets) -> Verlust -> Decode zusammenpassen.
Wenn das klappt, ist der Rest eine Frage von Rechenzeit und Datenmenge.
"""
import random

import numpy as np
import torch

import detmath as dm
import synth_data as sd
import train_det as td
from hybrid_net import SIGN_LABELS, NetCfg, build_model

size = 128
torch.manual_seed(0)
rng = random.Random(3)
arr, boxes, tags = sd.compose_sample(rng, size, degrade_prob=0.0)
xyxy = np.array([[(b["cx"] - b["w"] / 2) * size, (b["cy"] - b["h"] / 2) * size,
                  (b["cx"] + b["w"] / 2) * size, (b["cy"] + b["h"] / 2) * size] for b in boxes], np.float32)
labels = np.array([SIGN_LABELS.index(b["label"]) for b in boxes], np.int64)
print("GT:", [(b["label"], round((b["cx"] - b["w"] / 2) * size), round((b["cy"] - b["h"] / 2) * size),
               round((b["cx"] + b["w"] / 2) * size), round((b["cy"] + b["h"] / 2) * size)) for b in boxes], tags)

x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div(255.0)[None]
_, tgt, _ = td.collate([(x[0], torch.from_numpy(xyxy), torch.from_numpy(labels))], size)

model = build_model(NetCfg(tr_global=("p5",), tr_window=()))
crit = td.DetLoss(size=size)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for i in range(120):
    model.train()
    loss, parts = crit(model(x), tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if i % 20 == 0 or i == 119:
        model.eval()
        with torch.no_grad():
            outs = [o.numpy()[0] for o in model(x)]
        dets = dm.decode_multi(outs, dm.LEVELS, 0.25, 0.45)
        top = dets[0] if dets else None
        info = "-" if top is None else (f"{SIGN_LABELS[top['label_idx']]} {top['score']:.2f} "
                                        f"({top['x0']:.0f},{top['y0']:.0f},{top['x1']:.0f},{top['y1']:.0f})")
        print(f"i={i:3d} loss={float(loss.detach()):7.3f} obj={parts['obj']:.3f} cls={parts['cls']:.3f} "
              f"box={parts['box']:.3f} npos={parts['n_pos']} top={info}", flush=True)
