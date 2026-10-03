"""
tools/hybrid_net.py - Hybrides Nano-Netz fuer Schildererkennung.

Aufgabe: in EINEM Vorwaertslauf Position (Bounding Box) und Art (9 Schildtypen)
finden - anchor-free, einstufig, vier Aufloesungsstufen (stride 4 / 8 / 16 / 32)
bei fester Eingabe 320x320.

Warum vier Stufen? Gemessen war der Recall fuer Schilder unter 32 px Diagonale nur
etwa 0.29 gegen 0.74 im Mittel - und die alte Zuordnung (naechste Stufe in log2)
schickte ein 30-px-Schild auf stride 32, wo nur 10x10 Zellen zur Verfuegung stehen.
Stride 4 kostet rund ein Viertel mehr Rechnung (siehe tools/bench_model.py), deshalb
wird dort nur die feinste Stufe bedient (siehe tools/detmath.py, ASSIGN_MAX_SIDE).

Warum Transformer nur in den TIEFEN Stufen?
    Attention kostet O(N^2) in der Tokenzahl N = (H/stride) * (W/stride).
    320x320 ergibt:  stride 4  -> 80x80 = 6400 Tokens
                     stride 8  -> 40x40 = 1600 Tokens
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

# Reihenfolge = Klassenreihenfolge im Modell. Sie steht in tools/signmap.py, weil dort auch
# die Zuordnung der drei Datenquellen liegt (GTSRB-ClassId, StVO-Nummer, Synset-Name) - zwei
# Listen waeren zwei Wahrheiten. src/detector.js kennt die neun Grundtypen weiterhin als
# Heuristik; fuer die Anzeige aller Klassen schickt export_onnx.py die Namen in labels.json mit.
import signmap

SIGN_LABELS = signmap.LABELS
N_CLASSES = len(SIGN_LABELS)          # 74 (siehe tools/signmap.py)
# Kanalreihenfolge im Kopf-Ausgang: [tx, ty, tw, th, obj, cls0..cls73]
N_CH = 4 + 1 + N_CLASSES              # 79


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
    levels: tuple = (4, 8, 16, 32)        # Erkennungsstufen (stride) fuer die Koepfe
    mid: tuple = (32, 32, 64, 128)        # Kopfbreite je Stufe (gleiche Reihenfolge)

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
    "fast": dict(width=(24, 48, 96, 192), depth=(1, 1, 1, 1), mid=(24, 24, 48, 96),
                 tr_global=("p5",), tr_window=(), act="hardswish"),
    # Standard: volle Breiten, 2 Bloecke je tiefer Stufe, globaler Block nur auf p5
    "balanced": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 32, 64, 128),
                     tr_global=("p5",), tr_window=(), act="silu"),
    # mehr Kontext (p4 zusaetzlich), dafuer teurer
    "quality": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 32, 64, 128),
                    tr_global=("p5", "p4"), tr_window=("p4",), act="silu"),
    # Fuer die erweiterte Taxonomie (74 Klassen, tools/signmap.py). Warum genau so:
    #  * mid = Rumpfbreite. Der Kopf beginnt mit einer TIEFENCONVOLUTION (cba(cin, mid, g=mid)),
    #    deshalb muss mid die Rumpfbreite TEILEN - und der groesste zulaessige Wert ist die
    #    Rumpfbreite selbst. Damit bekommt der Klassenzweig den vollen Merkmalsvorrat statt
    #    der Haelfte. Genau dort faellt die Entscheidung (74 Klassen statt 9).
    #  * depth=(1,2,3,3): ein Block mehr in den tiefen Stufen. Dort stehen wenige Zellen
    #    (20x20 und 10x10), ein Block kostet also fast nichts, vergroessert aber das
    #    receptive Feld - noetig, um ein Schild von seiner Umgebung zu trennen.
    #  * tr_window=("p4",) statt tr_global=("p5","p4") wie in "quality": lokale Attention ist
    #    billig (Fenster 5x5), globale auf stride 8 waere rund 256x teurer (siehe Kopf dieser
    #    Datei). Gemessen: 1,88 Mio. Parameter / 1 180 MFLOPs gegen 1,32 Mio. / 916 bei
    #    "balanced" (+42 % / +29 %) - der letzte Lauf brauchte 70 von 540 Kaggle-Minuten.
    "breit": dict(width=(32, 64, 128, 256), depth=(1, 2, 3, 3), mid=(32, 64, 128, 256),
                  tr_global=("p5",), tr_window=("p4",), act="silu"),
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
    """Erkennungskopf: gemeinsamer Stamm, dann zwei getrennte Zweige (Box/Objekt, Art).

    Ausgabe je Zelle bleibt [tx, ty, tw, th, obj, cls0..cls8] - diese Reihenfolge ist der
    Vertrag mit tools/detmath.py und src/model.js und darf sich nicht aendern.

    Warum getrennte Zweige? Gemessen waren von 656 verpassten Boxen 212 falsch
    klassifiziert, und 195 der 226 Fehlalarme lagen auf echten Schildern: der Kopf
    verwechselt Arten (rotes Dreieck Spitze oben gegen Spitze unten 32x). Ortstreue (Box)
    und Invarianz (Art) sind gegensaetzliche Aufgaben, ein gemeinsamer Kanalvorrat fuer
    beide bremst. Die Trennung kostet rund 4 MFLOPs (siehe tools/bench_model.py).
    """

    def __init__(self, cin: int, mid: int, act: str = "silu"):
        super().__init__()
        self.stem = nn.Sequential(
            cba(cin, mid, 3, 1, g=mid, act=act),
            cba(mid, mid, 1, 1, act=act),
        )
        self.obj = nn.Conv2d(mid, 5, 1)                 # tx, ty, tw, th, obj
        self.cls = nn.Conv2d(mid, N_CLASSES, 1)         # cls0..cls8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        return torch.cat([self.obj(y), self.cls(y)], dim=1)


class HybridNano(nn.Module):
    """Ein Netz fuer Ort und Art: CNN-Backbone, optional Transformer-Stufen, FPN-lite, je Stufe ein Kopf.

    Eingang: (B, 3, 320, 320), Werte 0..1 (RGB).
    Ausgang: ein Tensor je Stufe, fein -> grob, je (B, 14, H/stride, W/stride).
             Reihenfolge wie LEVELS in tools/detmath.py und wie die ONNX-Ausgaenge
             os4/os8/os16/os32 (siehe tools/export_onnx.py).
    """

    def __init__(self, cfg: NetCfg | None = None):
        super().__init__()
        self.cfg = cfg or NetCfg()
        c = self.cfg
        w = c.width
        if len(c.mid) != len(c.levels) or len(w) not in (len(c.levels), len(c.levels) + 1):
            raise ValueError("width/mid/levels passen nicht zusammen: mid und levels gleich lang, "
                             "width gleich lang oder genau ein Eintrag mehr")
        # Abgriffe = die letzten len(levels) Rueckgrat-Stufen. Eine aeltere Konfiguration mit
        # drei Stufen (8/16/32) laeuft damit unveraendert durch denselben Code.
        self.taps = list(range(len(w) - len(c.levels), len(w)))
        self.stem = cba(3, 16, 3, 2, act=c.act)                     # /2
        self.stage1 = self._stage(16, w[0], c.depth[0], c.act)      # /4   -> p2
        self.stage2 = self._stage(w[0], w[1], c.depth[1], c.act)    # /8   -> p3
        self.stage3 = self._stage(w[1], w[2], c.depth[2], c.act)    # /16  -> p4
        self.stage4 = self._stage(w[2], w[3], c.depth[3], c.act)    # /32  -> p5

        # Transformer-Bloecke: tief = global, mittig = lokal (windowed). tr_plan haelt die
        # Reihenfolge fest, in der die Abgriffe bearbeitet werden (grob -> fein).
        self.tr, self.tr_plan = nn.ModuleDict(), []
        for name, stride in (("p5", 32), ("p4", 16), ("p3", 8), ("p2", 4)):
            if stride not in c.levels:
                continue
            pos = c.levels.index(stride)
            if name in c.tr_global:
                self.tr[name] = HybridEncoder(w[self.taps[pos]], c, window=0)
                self.tr_plan.append((pos, name))
            if name in c.tr_window:
                self.tr[name + "w"] = HybridEncoder(w[self.taps[pos]], c, window=c.win)
                self.tr_plan.append((pos, name + "w"))

        # FPN-lite: lat[i] holt die groebere Stufe hoch, fuse[i] glaettet die Summe.
        # Die groebste Stufe wird nicht gefiltert - sie hat nichts ueber sich.
        self.lat = nn.ModuleList([cba(w[t + 1], w[t], 1, 1, act="") for t in self.taps[:-1]])
        self.fuse = nn.ModuleList([cba(w[t], w[t], 3, 1, act=c.act) for t in self.taps[:-1]])
        self.heads = nn.ModuleList([Head(w[t], c.mid[i], c.act)
                                    for i, t in enumerate(self.taps)])
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
        # Objektivitaet zu Beginn selten (Prior ~1 %), Klassen zunaechst unsicher:
        # stabilisiert die ersten Epochen. Die beiden Kopfzweige werden getrennt vorbelegt.
        for head in self.heads:
            nn.init.constant_(head.obj.bias[4], -4.6)
            nn.init.constant_(head.cls.bias, -2.0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = [self.stage1(x)]                    # /4  (p2)
        feats.append(self.stage2(feats[-1]))        # /8  (p3)
        feats.append(self.stage3(feats[-1]))        # /16 (p4)
        feats.append(self.stage4(feats[-1]))        # /32 (p5)
        # Transformer-Bloecke anwenden (Reihenfolge wie beim Bau, grob -> fein)
        for pos, name in self.tr_plan:
            tap = self.taps[pos]
            feats[tap] = self.tr[name](feats[tap])
        # FPN-lite von grob nach fein: jede Stufe bekommt die hochgezogene groebere Stufe
        out: list[torch.Tensor | None] = [None] * len(self.taps)
        out[-1] = feats[self.taps[-1]]
        for i in range(len(self.taps) - 2, -1, -1):
            tap = self.taps[i]
            up = F.interpolate(self.lat[i](out[i + 1]), size=feats[tap].shape[-2:],
                               mode="nearest")
            out[i] = self.fuse[i](feats[tap] + up)
        # Reihenfolge fein -> grob (stride 4, 8, 16, 32): identisch zu tools/detmath.LEVELS
        return [head(o) for head, o in zip(self.heads, out)]


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
