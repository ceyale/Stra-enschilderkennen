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

Token-Mischer: CATM statt Selbstattention
    In den tiefen/mittleren Stufen sitzt jetzt der Convolutional Additive Token Mixer
    (CATM) aus CAS-ViT (arXiv:2408.03703, Zhang u. a., "Convolutional Additive
    Self-attention Vision Transformers for Efficient Mobile Applications"). Er ersetzt die
    beiden frueheren Aufmerksamkeitsvarianten (global auf p5, Fenster 5x5 auf p4) durch
    EINE gemeinsame Zelle:

        q, k, v = 1x1(x)                 # ein gemeinsamer 1x1-Projektor auf 3*dim
        q = ChannelOperation(SpatialOperation(q))
        k = ChannelOperation(SpatialOperation(k))
        y = 3x3-Tiefenconv(proj) (dwc(q + k) * v)

    Warum das hier besser passt als Attention:
      * ADDITIV statt multiplikativ: (q + k) gewichtet Merkmale, ohne dass ein
        N x N-Aehnlichkeitsfeld entsteht. Kein Softmax, kein MatMul ueber Tokens.
      * Damit faellt die O(N^2)-Kurve weg, die vorher die Stufenwahl bestimmt hat: derselbe
        Block ist auf stride 8 (1600 Tokens), 16 (400) und 32 (100) gleich teuer pro Zelle.
        Die frueher noetige Fensteraufteilung (win=5) ist ersatzlos entfallen.
      * Rein konvolutional -> BELIEBIGE Eingabegroessen. Die Fensterteilung musste vorher
        auf (-H) % win auffuellen; dieser Zwang und die damit verbundenen Randbedingungen im
        ONNX-Graph sind hier verschwunden, dynamische Hoehe/Breite bleibt sauber erhalten.
      * SpatialOperation ist ein Aufmerksamkeitsgewicht ueber den ORT (dwc + 1 Kanal +
        Sigmoid), ChannelOperation eines ueber die KANAELE (globaler Mittelwert + 1x1 +
        Sigmoid). Zusammen also Ort- und Kanalgewichtung - genau die zwei Achsen, die eine
        Merkmalskarte hat.

Objektivitaet: Vor jeder CATM-Zelle steht - wie in CAS-ViT - eine lokale Wahrnehmung
(LocalIntegration, Tiefenconv 3x3 zwischen zwei 1x1) als Restzweig. Pre-Norm und
LayerScale (kleines gamma) bleiben aus dem Vorgaenger erhalten, weil sie auch hier das
Anlaufen stabilisieren.

Merkmalsverarbeitung: LGP-FPN statt FPN-lite
    Statt "1x1 hochziehen + 3x3 glaetten" (FPN-lite) sitzt jetzt eine leichtgewichtige
    granulare Wahrnehmungs-Pyramide (LGP-FPN). Aufbau je Stufe:
      * GranularPerception: dieselbe Stufe parallel mit Tiefenconvs mehrerer Koernungen
        (3x3 und 5x5), additiv zusammengefuehrt (1x1). "Granular" heisst hier: ein Merkmal
        wird gleichzeitig in mehreren Kornungen betrachtet - das ist fuer kleine Schilder
        entscheidend, weil deren Umriss bei 3x3 noch Form ist und bei 5x5 schon Umgebung.
      * ContextAware: globaler Mittelwert je Kanal + 1x1, additiv zurueck - der Kontextbezug
        aus dem Namen des Verfahrens.
      * Seitliche Verbindungen sind 1x1 (leichtgewichtig), es gibt keine 3x3-Faltung im
        Top-down-Weg.

Hinweis zur Herkunft der LGP-FPN: das Verfahren stammt aus Yan Zhang u. a., "A lightweight
granular perception feature pyramid network with context-awareness for small traffic sign
detection", Expert Systems with Applications 317:131885 (2026). Eine Referenzumsetzung ist
NICHT oeffentlich; die Umsetzung hier folgt dem Namen und der Aufgabenstellung (leicht-
gewichtig, granulare Mehrkornung, Kontextbezug) und ist als solche gekennzeichnet - sie ist
keine 1:1-Uebernahme des Originalmoduls.

Kein gelerntes Positions-Embedding: die Position liefert weiterhin eine Depthwise-Convolution
(CPE, "conditional positional encoding" im Geist von CvT/LeViT).

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
    catm: tuple = ("p5", "p4")            # Stufen mit CATM-Block (AdditiveTokenMixer)
    tr_ffn: float = 2.0                   # FFN-Expansion im CATM-Block
    tr_norm: str = "gn"                   # "gn" = GroupNorm(1,C) schnell | "ln" = LayerNorm je Token
    lgp_ctx: bool = True                  # Kontextzweig der LGP-FPN (globale Sicht)
    levels: tuple = (4, 8, 16, 32)        # Erkennungsstufen (stride) fuer die Koepfe
    mid: tuple = (32, 32, 64, 128)        # Kopfbreite je Stufe (gleiche Reihenfolge)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["name"] = self.name()
        return d

    def name(self) -> str:
        catm = "".join(s.upper() for s in self.catm) or "-"
        return f"hybridnano_catm{catm}_lgp_{self.act}"


# Fertige Groessen: schneller heisst hier schmaler UND flacher UND weniger CATM-Stufen.
# Gemessen mit tools/bench_model.py --preset <name> (siehe docs/TRAINING.md).
# catm nennt die STUFEN mit CATM-Block. Weil CATM pro Zelle konstant teuer ist (kein
# N x N-Feld, siehe Kopf dieser Datei), ist auch stride 8 (p3) vertretbar - stride 4 (p2)
# bleibt trotzdem aussen vor: dort sind es 6400 Zellen.
PRESETS: dict[str, dict] = {
    # kleinste Variante: ~1/4 der Rechnung von "balanced", Hardswish, ein schmaler Kopf
    "fast": dict(width=(24, 48, 96, 192), depth=(1, 1, 1, 1), mid=(24, 24, 48, 96),
                 catm=("p5",), act="hardswish"),
    # Standard: volle Breiten, 2 Bloecke je tiefer Stufe, CATM auf p5 und p4
    "balanced": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 32, 64, 128),
                     catm=("p5", "p4"), act="silu"),
    # mehr Kontext: CATM zusaetzlich auf p3 (mittlere Aufloesung), dafuer teurer
    "quality": dict(width=(32, 64, 128, 256), depth=(1, 2, 2, 2), mid=(32, 32, 64, 128),
                    catm=("p5", "p4", "p3"), act="silu"),
    # Fuer die erweiterte Taxonomie (74 Klassen, tools/signmap.py). Warum genau so:
    #  * mid = Rumpfbreite. Der Kopf beginnt mit einer TIEFENCONVOLUTION (cba(cin, mid, g=mid)),
    #    deshalb muss mid die Rumpfbreite TEILEN - und der groesste zulaessige Wert ist die
    #    Rumpfbreite selbst. Damit bekommt der Klassenzweig den vollen Merkmalsvorrat statt
    #    der Haelfte. Genau dort faellt die Entscheidung (74 Klassen statt 9).
    #  * depth=(1,2,3,3): ein Block mehr in den tiefen Stufen. Dort stehen wenige Zellen
    #    (20x20 und 10x10), ein Block kostet also fast nichts, vergroessert aber das
    #    receptive Feld - noetig, um ein Schild von seiner Umgebung zu trennen.
    #  * catm=("p5","p4","p3"): p3 ist NEU gegenueber dem Vorgaenger. Vorher war dort keine
    #    Aufmerksamkeit moeglich (Fenster/global), jetzt kostet CATM pro Zelle gleich viel -
    #    und p3 ist genau die Stufe, auf der kleine Schilder (unter 32 px) landen.
    "breit": dict(width=(32, 64, 128, 256), depth=(1, 2, 3, 3), mid=(32, 64, 128, 256),
                  catm=("p5", "p4", "p3"), act="silu"),
}


def preset_cfg(preset: str, **overrides) -> "NetCfg":
    """Konfiguration aus einem Preset bauen; einzelne Felder per Schluesselwort ueberschreiben."""
    if preset not in PRESETS:
        raise ValueError(f"unbekanntes Preset {preset!r}; bekannt: {', '.join(PRESETS)}")
    base = dict(PRESETS[preset])
    base.update(overrides)
    cfg = NetCfg(**base)
    # Keine Teilbarkeitspruefung mehr: CATM hat keine Koepfe mehr (frueher tr_heads), die eine
    # Kanalzahl teilten. Der einzige harte Zwang kommt aus tools/detmath.py (Stufen 4/8/16/32).
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


class SpatialOperation(nn.Module):
    """Ortgewichtung: ein Aufmerksamkeitsgewicht ueber (H, W), je Kanal gemeinsam.

    Tiefenconv 3x3 -> BN -> ReLU -> 1x1 auf EINEN Kanal -> Sigmoid. Das Ergebnis ist eine
    Gewichtskarte, die auf alle Kanaele multipliziert wird. Uebernommen aus CAS-ViT
    (detection/model/rcvit.py, SpatialOperation) - dort genau so definiert.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class ChannelOperation(nn.Module):
    """Kanalgewichtung: ein Aufmerksamkeitsgewicht ueber die Kanaele, je Bild gemeinsam.

    Globaler Mittelwert -> 1x1 -> Sigmoid (SE-artig, ohne Engstelle). Uebernommen aus
    CAS-ViT (ChannelOperation). Der globale Mittelwert ist als ONNX unkritisch und
    funktioniert bei dynamischer Hoehe/Breite ohne Sonderfall.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class CATM(nn.Module):
    """Convolutional Additive Token Mixer (CAS-ViT, arXiv:2408.03703).

    Uebernommen aus der Referenzumsetzung (detection/model/rcvit.py, AdditiveTokenMixer),
    nur Schreibweise und Kommentare angepasst. Der Kern ist die ADDITIVE Verknuepfung:

        out = proj(dwc(q + k) * v)

    Statt eines N x N-Aehnlichkeitsfeldes (q @ k, Softmax) wird hier ADDiert. Deshalb gibt es
    weder Softmax noch MatMul ueber Tokens noch Fenster - und damit auch kein Auffuellen auf
    ein Vielfaches der Fenstergroesse. Der Graph ist rein konvolutional und bleibt bei
    beliebiger Hoehe/Breite gueltig.

    Ablauf: ein gemeinsamer 1x1-Projektor erzeugt q, k, v. q und k laufen durch
    (Ortgewichtung, Kanalgewichtung), werden addiert, mit v multipliziert, durch eine
    Tiefenconv 3x3 und zuletzt durch eine 3x3-Tiefenconv als Ausgangsprojektion geschickt.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=False)
        self.oper_q = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.oper_k = nn.Sequential(SpatialOperation(dim), ChannelOperation(dim))
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        return self.proj(self.dwc(q + k) * v)


class LocalIntegration(nn.Module):
    """Lokale Wahrnehmung als Restzweig (CAS-ViT, LocalIntegration).

    1x1 -> Aktivierung -> Tiefenconv 3x3 -> Aktivierung -> 1x1. Das ist der Zweig, der in
    CAS-ViT VOR jedem Token-Mischer liegt (x = x + local(x)): er sammelt die Nachbarschaft,
    bevor der Mischer globale/kanalweise Gewichte setzt.
    """

    def __init__(self, dim: int, ratio: float = 1.0, act: str = "silu"):
        super().__init__()
        mid = max(8, int(round(dim * ratio / 8)) * 8)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), act_layer(act),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False), nn.BatchNorm2d(mid),
            act_layer(act),
            nn.Conv2d(mid, dim, 1, 1, 0, bias=False), nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CatmBlock(nn.Module):
    """Ein Block mit CATM statt Selbstattention: lokale Wahrnehmung + Mischer + FFN.

    Aufbau wie AdditiveBlock in CAS-ViT:

        x = x + local(x)
        x = x + drop_path(mix(norm1(x)))
        x = x + drop_path(ffn(norm2(x)))

    Beibehalten aus dem Vorgaenger (HybridEncoder) sind Pre-Norm, LayerScale (kleines gamma)
    und die optionale CPE-Tiefenconv. Die CPE traegt weiter die Ortsinformation bei: die
    Gewichtungen in CATM sind ortsabhaengig, aber translationsinvariant - ohne die CPE
    koennte der Block eine Position nicht von einer gleich aussehenden anderen unterscheiden.
    """

    def __init__(self, dim: int, cfg: NetCfg):
        super().__init__()
        self.local = LocalIntegration(dim, 1.0, cfg.act)
        self.norm1 = TokenNorm(dim, cfg.tr_norm)
        self.mix = CATM(dim)
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
        x = x + self.local(x)
        x = x + self.gamma1 * self.cpe(self.mix(self.norm1(x)))
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


class GranularPerception(nn.Module):
    """Granulare Wahrnehmung: dieselbe Stufe in mehreren Koernungen gleichzeitig.

    Zwei parallele Tiefenconvs (3x3 und 5x5) auf demselben Eingang, beide mit BN und
    Aktivierung, danach additiv zusammengefuehrt und mit 1x1 gemischt. Warum das fuer
    Verkehrszeichen hilft: ein Schild von 15-30 px ist bei 3x3 noch reine Form (Kreis,
    Dreieck, Raute), bei 5x5 dagegen schon Form MIT Umgebung. Beides gleichzeitig zu sehen
    ist genau die Information, die zwischen "roter Kreis" und "rotes Rad am Auto"
    unterscheidet.

    Leichtgewichtig ist das, weil beide Zweige TIEFENconvs sind (groups=dim): die Rechnung
    waechst mit der Kernelbreite, nicht mit dim^2. Der 3x3-Zweig ist restverbunden, damit die
    Stufe nicht schlechter wird als ohne dieses Modul.
    """

    def __init__(self, dim: int, act: str = "silu"):
        super().__init__()
        self.dw3 = cba(dim, dim, 3, 1, g=dim, act=act)
        self.dw5 = cba(dim, dim, 5, 1, g=dim, act=act)
        self.mix = cba(dim, dim, 1, 1, act="")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(self.dw3(x) + self.dw5(x))


class ContextAware(nn.Module):
    """Kontextbezug: globale Sicht je Kanal, additiv zurueckgelegt.

    Globaler Mittelwert -> 1x1 -> addieren. Das ist der Kontextzweig aus dem Namen des
    Verfahrens ("with context-awareness") und entspricht der ChannelOperation aus CAS-ViT,
    nur ohne Sigmoid: hier soll der Kontext ADDIERT werden, nicht als Tor dienen.

    Der globale Mittelwert ist als ONNX unproblematisch und bleibt bei dynamischer Hoehe und
    Breite gueltig (AdaptiveAvgPool auf 1x1).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(self.pool(x))


class HybridNano(nn.Module):
    """Ein Netz fuer Ort und Art: CNN-Backbone, CATM-Stufen, LGP-FPN, je Stufe ein Kopf.

    Eingang: (B, 3, 320, 320), Werte 0..1 (RGB).
    Ausgang: ein Tensor je Stufe, fein -> grob, je (B, 79, H/stride, W/stride).
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

        # CATM-Bloecke auf den gewaehlten Stufen. catm_plan haelt die Reihenfolge fest
        # (grob -> fein). Anders als die frueheren Fensterbloecke gibt es KEINE Fenstergroesse
        # und keine Auffuellung - der Block ist rein konvolutional.
        self.catm_blocks, self.catm_plan = nn.ModuleDict(), []
        for name, stride in (("p5", 32), ("p4", 16), ("p3", 8), ("p2", 4)):
            if stride not in c.levels or name not in c.catm:
                continue
            pos = c.levels.index(stride)
            self.catm_blocks[name] = CatmBlock(w[self.taps[pos]], c)
            self.catm_plan.append((pos, name))

        # LGP-FPN: seitliche 1x1-Verbindungen (leichtgewichtig) und je zusammengefuehrter
        # Stufe eine granulare Wahrnehmung mit optionalem Kontextbezug. Die groebste Stufe hat
        # nichts ueber sich und bekommt deshalb nur diese Nachbearbeitung.
        self.lat = nn.ModuleList([cba(w[t + 1], w[t], 1, 1, act="") for t in self.taps[:-1]])
        self.lgp = nn.ModuleList([
            nn.Sequential(GranularPerception(w[t], c.act),
                          *((ContextAware(w[t]),) if c.lgp_ctx else ()))
            for t in self.taps[:-1]])
        self.lgp_top = nn.Sequential(
            GranularPerception(w[self.taps[-1]], c.act),
            *((ContextAware(w[self.taps[-1]]),) if c.lgp_ctx else ()))
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
        # CATM-Bloecke anwenden (Reihenfolge wie beim Bau, grob -> fein)
        for pos, name in self.catm_plan:
            tap = self.taps[pos]
            feats[tap] = self.catm_blocks[name](feats[tap])
        # LGP-FPN von grob nach fein: seitliche 1x1-Verbindung hochziehen, addieren, dann auf
        # der zusammengefuehrten Stufe granulare Wahrnehmung (und Kontextbezug).
        out: list[torch.Tensor | None] = [None] * len(self.taps)
        out[-1] = self.lgp_top(feats[self.taps[-1]])
        for i in range(len(self.taps) - 2, -1, -1):
            tap = self.taps[i]
            up = F.interpolate(self.lat[i](out[i + 1]), size=feats[tap].shape[-2:],
                               mode="nearest")
            out[i] = self.lgp[i](feats[tap] + up)
        # Reihenfolge fein -> grob (stride 4, 8, 16, 32): identisch zu tools/detmath.LEVELS
        return [head(o) for head, o in zip(self.heads, out)]


def build_model(cfg: NetCfg | None = None) -> HybridNano:
    """Modell bauen; die Konfiguration ist ueber CLI-Flags der Tools einstellbar."""
    return HybridNano(cfg)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Drei Varianten zum Vergleich: ohne CATM (reines CNN), Standard, und CATM bis p3.
    for cfg in (NetCfg(catm=()), NetCfg(), NetCfg(catm=("p5", "p4", "p3"))):
        m = build_model(cfg).eval()
        with torch.no_grad():
            outs = m(torch.zeros(1, 3, 320, 320))
        shapes = " | ".join(str(tuple(o.shape)) for o in outs)
        print(f"{cfg.name():36s} params={n_params(m):8d}  out: {shapes}")
