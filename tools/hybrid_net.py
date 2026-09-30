"""
tools/hybrid_net.py - Hybrides Nano-Netz fuer Schildererkennung.

Aufgabe: in EINEM Vorwaertslauf Position (Bounding Box) und Art (9 Schildtypen)
finden - anchor-free, einstufig, drei Aufloesungsstufen (stride 8 / 16 / 32)
bei fester Eingabe 320x320.

Warum Transformer nur in den TIEFEN Stufen?
    Attention kostet O(N^2) in der Tokenzahl N = (H/stride) * (W/stride).
    320x320 ergibt:  stride 8 -> 40x40 = 1600 Tokens
                     stride 16 -> 20x20 =  400 Tokens
                     stride 32 -> 10x10 =  100 Tokens
    Ein globaler Block auf stride 8 waere ~256x teurer als auf stride 32.
    Deshalb: globaler Transformer NUR auf stride 32 (dort praktisch gratis),
    lokale (windowed) Bloecke auf stride 16, und die feinen Stufen bleiben CNN.
    Gemessene Kosten pro Variante: siehe tools/bench_model.py / docs/TRAINING.md

Kein gelerntes Positions-Embedding: die Position liefert eine Depthwise-Convolution
(CPE, "conditional positional encoding" im Geist von CvT/LeViT). Damit funktioniert
derselbe Block bei jeder Aufloesung und die Export-Graphen bleiben statisch.

Dieses Modul ist reines Training/Export-Werkzeug (Python, PyTorch) und wird NICHT
an den Browser ausgeliefert. Ausgeliefert wird nur models/signs-det.onnx.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reihenfolge = Klassenreihenfolge im Modell. Muss identisch zu den Keys in
# src/detector.js (SIGNS) bleiben; export_onnx.py schreibt sie nach models/labels.json.
SIGN_LABELS = [
    "stop", "vorfahrtGewaehren", "warnung", "verbot", "einfahrtVerboten",
    "gebot", "hinweis", "vorfahrtstrasse", "ortstafel",
]
N_CLASSES = len(SIGN_LABELS)          # 9
# Kanalreihenfolge im Kopf-Ausgang: [tx, ty, tw, th, obj, cls0..cls8]
N_CH = 4 + 1 + N_CLASSES              # 14


@dataclass
class NetCfg:
    """Architektur-Schalter. Alles, was die Laufzeit beeinflusst, ist hier einstellbar."""
    width: tuple = (32, 64, 128, 256)     # Kanaele nach stride 4/8/16/32
    depth: tuple = (1, 2, 2, 2)           # IR-Bloecke pro Stufe
    act: str = "silu"                     # "silu" (genauer) | "hardswish" (auf ARM schneller)
    tr_global: tuple = ("p5",)            # globale Transformer-Stufen
    tr_window: tuple = ("p4",)            # windowed Transformer-Stufen
    tr_heads: int = 4
    tr_ffn: float = 2.0                   # FFN-Expansion im Transformer-Block
    tr_norm: str = "gn"                   # "gn" = GroupNorm(1,C) schnell | "ln" = LayerNorm je Token
    win: int = 5                          # Fenstergroesse fuer windowed Attention
    levels: tuple = (8, 16, 32)           # Erkennungsstufen (stride) fuer die Koepfe
    mid: tuple = (32, 64, 128)            # Kopf-Breite je Stufe

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["name"] = self.name()
        return d

    def name(self) -> str:
        tg = "".join(s.upper() for s in self.tr_global) or "-"
        tw = "".join(s.upper() for s in self.tr_window) or "-"
        return f"hybridnano_g{tg}_w{tw}_{self.act}"


# Fertige Groessen: schneller heisst hier schmaler UND flacher UND weniger Transformer.
# Gemessen mit tools/bench_model.py --preset <name> (siehe docs/TRAINING.md).
PRESETS: dict[str, dict] = {
    # kleinste Variante: ~1/4 der Rechnung von "balanced", Hardswish, ein schmaler Kopf
    "fast": dict(width=(24, 48, 96, 192), depth=(1, 1, 1, 1), mid=(24, 48, 96),
                 tr_global=("p5",), tr_window=(), act="hardswish"),
    # Standard: volle Breiten, 2 Bloecke je tiefer Stufe, globaler Block nur auf p5
    "balanced": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 64, 128),
                     tr_global=("p5",), tr_window=(), act="silu"),
    # mehr Kontext (p4 zusaetzlich), dafuer teurer
    "quality": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 64, 128),
                    tr_global=("p5", "p4"), tr_window=("p4",), act="silu"),
}


def preset_cfg(preset: str, **overrides) -> "NetCfg":
    """Konfiguration aus einem Preset bauen; einzelne Felder per Schluesselwort ueberschreiben."""
    if preset not in PRESETS:
        raise ValueError(f"unbekanntes Preset {preset!r}; bekannt: {', '.join(PRESETS)}")
    base = dict(PRESETS[preset])
    base.update(overrides)
    cfg = NetCfg(**base)
    for dim in (cfg.width[3], cfg.width[2]):
        if dim % cfg.tr_heads:
            raise ValueError(f"Kanalzahl {dim} muss durch tr_heads={cfg.tr_heads} teilbar sein")
    return cfg


def act_layer(name: str, inplace: bool = True) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "hardswish":
        return nn.Hardswish(inplace=inplace)
    raise ValueError(f"unbekannte Aktivierung: {name}")


def cba(cin: int, cout: int, k: int = 1, s: int = 1, g: int = 1,
        act: str = "silu", bn: bool = True) -> nn.Sequential:
    """Conv -> BatchNorm -> Aktivierung (die Grundzelle des CNN-Teils)."""
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(cout))
    if act:
        layers.append(act_layer(act))
    return nn.Sequential(*layers)


class IRBlock(nn.Module):
    """Inverted-Residual-Block (MobileNetV2/V3-Stil): expand -> depthwise -> project.

    Traegt den CNN-Teil: billig in FLOPs, gutmuetig beim Quantisieren.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1, expand: float = 2.0, act: str = "silu"):
        super().__init__()
        mid = max(8, int(round(cin * expand / 8)) * 8)
        self.res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            cba(cin, mid, 1, 1, act=act),
            cba(mid, mid, 3, stride, g=mid, act=act),
            nn.Conv2d(mid, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.res else y


class TokenNorm(nn.Module):
    """Normalisierung fuer Transformer-Bloecke auf Merkmalskarten.

    "gn": GroupNorm(1, C) - normalisiert ueber Kanaele UND Ort, ohne Permute,
          dafuer nicht exakt "pro Token". Schnell und exportfreundlich.
    "ln": echtes LayerNorm je Token (B,H,W,C). Genauer, kostet Permutes/Transposes.
    """

    def __init__(self, dim: int, kind: str = "gn"):
        super().__init__()
        self.kind = kind
        self.op = nn.GroupNorm(1, dim) if kind == "gn" else nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gn":
            return self.op(x)
        x = x.permute(0, 2, 3, 1)
        x = self.op(x)
        return x.permute(0, 3, 1, 2)


class Attention(nn.Module):
    """Mehrkopf-Selbstattention auf (B, C, H, W).

    window = 0  -> global (alle Zellen sehen alle Zellen)
    window > 0  -> lokal in Fenstern (win x win), Kosten steigen nur linear mit H*W
    Bewusst per MatMul/Softmax statt F.scaled_dot_product_attention geschrieben:
    dieser Graph ist als ONNX stabil und laeuft in ONNX Runtime Web ohne Sonderop.
    """

    def __init__(self, dim: int, heads: int = 4, window: int = 0):
        super().__init__()
        if dim % heads:
            raise ValueError("dim muss durch heads teilbar sein")
        self.dim, self.h, self.win = dim, heads, window
        self.scale = float((dim // heads) ** -0.5)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    @staticmethod
    def _to_win(t: torch.Tensor, w: int) -> torch.Tensor:      # (B,C,H,W) -> (B*G, C, w*w)
        B, C, H, W = t.shape
        nh, nw = H // w, W // w
        return t.view(B, C, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).reshape(B * nh * nw, C, w * w)

    @staticmethod
    def _from_win(t: torch.Tensor, w: int, B: int, C: int, H: int, W: int) -> torch.Tensor:
        nh, nw = H // w, W // w
        return t.view(B, nh, nw, C, w, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, C, N = q.shape
        d = C // self.h
        q = q.view(B, self.h, d, N).permute(0, 1, 3, 2)        # (B,h,N,d)
        k = k.view(B, self.h, d, N)                            # (B,h,d,N)
        v = v.view(B, self.h, d, N).permute(0, 1, 3, 2)        # (B,h,N,d)
        att = torch.softmax(q @ k * self.scale, dim=-1)        # (B,h,N,N)
        return (att @ v).permute(0, 1, 3, 2).reshape(B, C, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (-H) % self.win if self.win else 0
        pad_w = (-W) % self.win if self.win else 0
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]
        q, k, v = self.qkv(x).chunk(3, dim=1)
        if self.win:
            o = self._attend(self._to_win(q, self.win), self._to_win(k, self.win), self._to_win(v, self.win))
            y = self._from_win(o, self.win, B, C, Hp, Wp)
        else:
            y = self._attend(q.flatten(2), k.flatten(2), v.flatten(2)).view(B, C, Hp, Wp)
        return self.proj(y)[:, :, :H, :W]


class HybridEncoder(nn.Module):
    """Ein Hybrid-Block: CPE + Attention + FFN (MobileViT-artig).

    Pre-Norm, LayerScale (kleines gamma) und Restverbindungen - damit trainieren
    solche Hybridnetze stabil, ohne dass eine Stufe die Vortrainingsgewichte zerstoert.
    CPE = Depthwise 3x3 auf dem Attention-Ausgang (ersetzt Positions-Embeddings).
    """

    def __init__(self, dim: int, cfg: NetCfg, window: int = 0):
        super().__init__()
        self.norm1 = TokenNorm(dim, cfg.tr_norm)
        self.attn = Attention(dim, cfg.tr_heads, window)
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gamma1 = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))
        self.norm2 = TokenNorm(dim, cfg.tr_norm)
        mid = max(8, int(round(dim * cfg.tr_ffn / 8)) * 8)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, mid, 1, bias=False), nn.BatchNorm2d(mid), act_layer(cfg.act),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False), nn.BatchNorm2d(mid), act_layer(cfg.act),
            nn.Conv2d(mid, dim, 1, bias=False), nn.BatchNorm2d(dim),
        )
        self.gamma2 = nn.Parameter(torch.full((1, dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.cpe(self.attn(self.norm1(x)))
        return x + self.gamma2 * self.ffn(self.norm2(x))


class Head(nn.Module):
    """Erkennungskopf: Depthwise 3x3 -> 1x1 -> Vorhersage (14 Kanaele).

    Ausgabe je Zelle: [tx, ty, tw, th, obj, cls0..cls8]
    Dekodierung (identisch in Python und JavaScript, siehe src/model.js):
        cx = (gx + sigmoid(tx)) * stride      cy = (gy + sigmoid(ty)) * stride
        w  = exp(tw) * stride                 h  = exp(th) * stride
        score = sigmoid(obj) * max_j sigmoid(cls_j)
    """

    def __init__(self, cin: int, mid: int, act: str = "silu"):
        super().__init__()
        self.stem = nn.Sequential(
            cba(cin, mid, 3, 1, g=mid, act=act),
            cba(mid, mid, 1, 1, act=act),
        )
        self.pred = nn.Conv2d(mid, N_CH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.stem(x))


class HybridNano(nn.Module):
    """Ein Netz fuer Ort und Art: CNN-Backbone, optional Transformer-Stufen, FPN-lite, 3 Koepfe.

    Eingang: (B, 3, 320, 320), Werte 0..1 (RGB).
    Ausgang: 3 Tensoren in der Reihenfolge stride 8, 16, 32 (fein -> grob),
             je (B, 14, H/stride, W/stride). Reihenfolge wie LEVELS in tools/detmath.py
             und wie die ONNX-Ausgaenge os8/os16/os32 (siehe tools/export_onnx.py).
    """

    def __init__(self, cfg: NetCfg | None = None):
        super().__init__()
        self.cfg = cfg or NetCfg()
        c = self.cfg
        w = c.width
        self.stem = cba(3, 16, 3, 2, act=c.act)                     # /2
        self.stage1 = self._stage(16, w[0], c.depth[0], c.act)      # /4
        self.stage2 = self._stage(w[0], w[1], c.depth[1], c.act)    # /8   -> p3
        self.stage3 = self._stage(w[1], w[2], c.depth[2], c.act)    # /16  -> p4
        self.stage4 = self._stage(w[2], w[3], c.depth[3], c.act)    # /32  -> p5

        # Transformer-Ersatz: tief = global, mittig = lokal (windowed)
        self.tr = nn.ModuleDict()
        if "p5" in c.tr_global:
            self.tr["p5"] = HybridEncoder(w[3], c, window=0)
        if "p4" in c.tr_global:
            self.tr["p4"] = HybridEncoder(w[2], c, window=0)
        if "p4" in c.tr_window:
            self.tr["p4w"] = HybridEncoder(w[2], c, window=c.win)
        if "p3" in c.tr_window:
            self.tr["p3w"] = HybridEncoder(w[1], c, window=c.win)

        # FPN-lite: Querverbindungen + Fusion (billig, hebt kleine Schilder)
        self.lat5 = cba(w[3], w[2], 1, 1, act="")
        self.lat4 = cba(w[2], w[1], 1, 1, act="")
        self.fuse4 = cba(w[2], w[2], 3, 1, act=c.act)
        self.fuse3 = cba(w[1], w[1], 3, 1, act=c.act)

        # Stufe i sitzt auf dem Merkmal mit width[i+1] Kanaelen (p3=w[1], p4=w[2], p5=w[3])
        self.heads = nn.ModuleList([Head(w[i + 1], c.mid[i], c.act) for i in range(3)])
        self._init_weights()

    @staticmethod
    def _stage(cin: int, cout: int, depth: int, act: str) -> nn.Sequential:
        layers = [IRBlock(cin, cout, stride=2, act=act)]
        layers += [IRBlock(cout, cout, stride=1, act=act) for _ in range(depth - 1)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Objektivitaet zu Beginn selten (Prior ~1%): stabilisiert die ersten Epochen
        for head in self.heads:
            nn.init.constant_(head.pred.bias[4], -4.6)
            nn.init.constant_(head.pred.bias[5:], -2.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c8 = self.stage2(self.stage1(x))          # /8
        c16 = self.stage3(c8)                     # /16
        c32 = self.stage4(c16)                    # /32
        # Ersetzen statt anhaengen: die Stufe wird durch den Transformer-Block gefuehrt,
        # das CNN liefert weiterhin die feinen Stufen (stride 8 und kleiner).
        for name, feat in (("p5", "c32"), ("p4", "c16"), ("p4w", "c16"), ("p3w", "c8")):
            if name not in self.tr:
                continue
            if feat == "c32":
                c32 = self.tr[name](c32)
            elif feat == "c16":
                c16 = self.tr[name](c16)
            else:
                c8 = self.tr[name](c8)
        p5 = c32
        p4 = self.fuse4(c16 + F.interpolate(self.lat5(p5), size=c16.shape[-2:], mode="nearest"))
        p3 = self.fuse3(c8 + F.interpolate(self.lat4(p4), size=c8.shape[-2:], mode="nearest"))
        # Reihenfolge fein -> grob (stride 8, 16, 32): identisch zu tools/detmath.LEVELS
        return [self.heads[0](p3), self.heads[1](p4), self.heads[2](p5)]


def build_model(cfg: NetCfg | None = None) -> HybridNano:
    """Modell bauen; die Konfiguration ist ueber CLI-Flags der Tools einstellbar."""
    return HybridNano(cfg)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for cfg in (NetCfg(tr_global=(), tr_window=()), NetCfg(), NetCfg(tr_window=(), tr_global=("p4", "p5"))):
        m = build_model(cfg).eval()
        with torch.no_grad():
            outs = m(torch.zeros(1, 3, 320, 320))
        shapes = " | ".join(str(tuple(o.shape)) for o in outs)
        print(f"{cfg.name():34s} params={n_params(m):8d}  out: {shapes}")
