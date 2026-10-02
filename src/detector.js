/**
 * detector.js – Kernlogik der Schilderkennung.
 * Keine DOM-Abhängigkeit → im Browser (window.SignDetector) und in Node (require) nutzbar.
 *
 * Ablauf (Details: docs/DOKUMENTATION.md):
 *   1. buildMask       Pixel nach Farbe klassifizieren (rot / blau / gelb) über HSV
 *   2. findComponents  zusammenhängende Farbflächen finden (Flood-Fill) und grob filtern
 *   3. analyzeShape    Silhouette der Fläche vermessen (Füllgrad, Breitenprofil)
 *   4. shapeOf/labelOf Form + Farbe → Schildtyp
 *   5. mergeOverlaps   doppelte Treffer desselben Typs zusammenfassen
 *   6. createTracker   Treffer über mehrere Bilder stabilisieren (gegen Flackern)
 */
(function (root) {
  'use strict';

  const NONE = 0, RED = 1, BLUE = 2, YELLOW = 3, GREEN = 4, WHITE = 5;

  /** Schwellwerte. Alle Größen in Pixeln des verkleinerten Analysebilds. */
  const CONFIG = {
    minSaturation: 0.5,   // Farbsättigung ab der ein Pixel zählt (0..1). Höher = strenger.
    minValue: 0.22,       // Helligkeit, darunter wird ein Pixel ignoriert
    minValueYellow: 0.45, // Gelb braucht mehr Helligkeit, sonst wirken braune Flächen gelb
    redHueMax: 14,        // Rot liegt am Rand des Farbkreises → zwei Grenzen
    redHueMin: 345,
    yellowHueMin: 38,
    yellowHueMax: 66,
    blueHueMin: 200,
    blueHueMax: 255,
    greenHueMin: 80,
    greenHueMax: 165,
    minBox: 10,           // kleinste Kantenlänge einer Fläche
    minArea: 40,          // kleinste Pixelanzahl einer Fläche
    maxFrameShare: 0.9,   // Flächen, die fast das ganze Bild füllen, sind kein Schild
    minAspect: 0.5,       // Breite/Höhe, erlaubter Bereich für jede Fläche
    maxAspect: 2.2,
    maxSolidityTriangle: 0.65, // darunter: Dreieck oder Raute
    minSolidityRect: 0.92,     // darüber: Rechteck
    minWidthNarrow: 0.35,      // oben *und* unten so schmal → Raute
    minWidthFlat: 0.6,         // oben *und* unten so breit → Achteck
    minSolidityOctagon: 0.8,
    minFillFull: 0.65,         // ab hier gilt ein roter Kreis als Vollfläche (Einfahrt verboten)
    minCircleAspect: 0.72,     // schräg gesehene Kreise werden Ellipsen → etwas Toleranz
    maxCircleAspect: 1.38,
    mergeIou: 0.35,            // so stark überlappende Treffer sind ein Schild
    mergeContain: 0.8,         // so stark verschachtelte Treffer sind ein Schild
    maxResults: 6              // höchstens so viele Schilder pro Bild melden
  };

  /** Katalog der erkennbaren Schilder (Zeichen-Nummern nach StVO). */
  const SIGNS = {
    stop:              { name: 'Stopp',             zeichen: 'Z 206',     hex: '#d3232f', text: '#fff',    note: 'Roter Achtkant: Halt! Vorfahrt gewähren.' },
    vorfahrtGewaehren: { name: 'Vorfahrt gewähren', zeichen: 'Z 205',     hex: '#d3232f', text: '#fff',    note: 'Rotes Dreieck mit der Spitze nach unten.' },
    warnung:           { name: 'Gefahrzeichen',     zeichen: 'Z 101 ff.', hex: '#d3232f', text: '#fff',    note: 'Rotes Dreieck, Spitze oben. Das Symbol im Inneren wird nicht gelesen.' },
    verbot:            { name: 'Verbotszeichen',    zeichen: 'Z 2xx',     hex: '#d3232f', text: '#fff',    note: 'Roter Ring, z. B. Tempolimit (Z 274). Die Zahl wird nicht gelesen.' },
    einfahrtVerboten:  { name: 'Einfahrt verboten', zeichen: 'Z 267',     hex: '#d3232f', text: '#fff',    note: 'Roter Vollkreis mit weißem Balken.' },
    gebot:             { name: 'Gebotszeichen',     zeichen: 'Z 2xx',     hex: '#1467b8', text: '#fff',    note: 'Blauer Kreis, z. B. vorgeschriebene Fahrtrichtung oder Radweg.' },
    hinweis:           { name: 'Hinweiszeichen',    zeichen: 'Z 3xx',     hex: '#1467b8', text: '#fff',    note: 'Blaues Rechteck, z. B. Parkplatz (Z 314).' },
    vorfahrtstrasse:   { name: 'Vorfahrtstraße',    zeichen: 'Z 306',     hex: '#f2c200', text: '#1e2329', note: 'Gelbe Raute mit weißem Rand.' },
    ortstafel:         { name: 'Ortstafel',         zeichen: 'Z 310',     hex: '#f2c200', text: '#1e2329', note: 'Gelbes Rechteck am Ortseingang.' }
    ,wegweiser:        { name: 'Wegweiser',         zeichen: 'Z 4xx',     hex: '#1467b8', text: '#fff',    note: 'Richtungs- oder Wegweiserschild.' }
    ,vorfahrt:         { name: 'Vorfahrt an der nächsten Kreuzung', zeichen: 'Z 301', hex: '#d3232f', text: '#fff', note: 'Vorfahrt an der nächsten Kreuzung.' }
    ,parkverbot:       { name: 'Halt-/Parkverbot', zeichen: 'Z 2xx', hex: '#d3232f', text: '#fff', note: 'Rundes Verbotsschild mit blauem Feld.' }
    ,gruenHinweis:     { name: 'Grünes Wegweiserschild', zeichen: 'Z 4xx', hex: '#16834a', text: '#fff', note: 'Grünes rechteckiges Wegweiserschild.' }
  };

  /** Schritt 1: RGB → Farbklasse. h in Grad (0..360), s und v in 0..1. */
  function classifyPixel(r, g, b, cfg) {
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const v = max / 255;
    if (d === 0 || v < cfg.minValue) return NONE;
    const s = d / max;
    if (s < cfg.minSaturation) return NONE;
    let h;
    if (max === r) h = 60 * (((g - b) / d + 6) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h <= cfg.redHueMax || h >= cfg.redHueMin) return RED;
    if (h >= cfg.yellowHueMin && h <= cfg.yellowHueMax && v > cfg.minValueYellow) return YELLOW;
    if (h >= cfg.blueHueMin && h <= cfg.blueHueMax) return BLUE;
    if (h >= cfg.greenHueMin && h <= cfg.greenHueMax) return GREEN;
    return NONE;
  }

  /** Maske für das ganze Bild (RGBA-Array wie von getImageData). */
  function buildMask(data, w, h, cfg) {
    const mask = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) mask[i] = classifyPixel(data[p], data[p + 1], data[p + 2], cfg);
    return mask;
  }

  /**
   * Schritt 2: Flood-Fill (4er-Nachbarschaft, gleiche Farbklasse).
   * Die Warteschlange `queue` enthält danach alle Pixel; jede Fläche belegt einen
   * zusammenhängenden Abschnitt [start, end) – so braucht man keine Extra-Listen.
   */
  function findComponents(mask, w, h, cfg) {
    const n = w * h, queue = new Int32Array(n), seen = new Uint8Array(n), comps = [];
    let tail = 0;
    for (let i = 0; i < n; i++) {
      const c = mask[i];
      if (!c || seen[i]) continue;
      const start = tail;
      let head = tail, x0 = w, x1 = 0, y0 = h, y1 = 0;
      queue[tail++] = i; seen[i] = 1;
      while (head < tail) {
        const p = queue[head++], x = p % w, y = (p / w) | 0;
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0     && !seen[p - 1] && mask[p - 1] === c) { seen[p - 1] = 1; queue[tail++] = p - 1; }
        if (x < w - 1 && !seen[p + 1] && mask[p + 1] === c) { seen[p + 1] = 1; queue[tail++] = p + 1; }
        if (y > 0     && !seen[p - w] && mask[p - w] === c) { seen[p - w] = 1; queue[tail++] = p - w; }
        if (y < h - 1 && !seen[p + w] && mask[p + w] === c) { seen[p + w] = 1; queue[tail++] = p + w; }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, area = tail - start, aspect = bw / bh;
      if (area < cfg.minArea || bw < cfg.minBox || bh < cfg.minBox) continue;   // zu klein
      if (aspect < cfg.minAspect || aspect > cfg.maxAspect) continue;            // falsches Seitenverhältnis
      if (bw > w * cfg.maxFrameShare || bh > h * cfg.maxFrameShare) continue;    // füllt das Bild
      comps.push({ color: c, x: x0, y: y0, w: bw, h: bh, area, start, end: tail });
    }
    return { comps, queue };
  }

  /**
   * Schritt 3: Silhouette vermessen. Pro Zeile zählt nur der äußerste linke und rechte
   * Pixel – so werden Ringe (rote Ränder) zur gefüllten Form.
   *   solidity = Silhouettenfläche / Rahmenfläche   (Kreis .785, Dreieck/Raute .5, Achteck .83, Rechteck 1)
   *   fill     = echte Farbpixel / Silhouettenfläche (Ring niedrig, Vollfläche hoch)
   *   wTop/wBot = mittlere Breite im oberen/unteren Viertel relativ zur Rahmenbreite
   * Gibt null zurück, wenn keine Silhouette messbar war (dann ist die Fläche unbrauchbar).
   */
  function analyzeShape(c, queue, w) {
    const rowMin = new Int16Array(c.h).fill(32767), rowMax = new Int16Array(c.h).fill(-1);
    for (let k = c.start; k < c.end; k++) {
      const p = queue[k], x = p % w, y = ((p / w) | 0) - c.y;
      if (x < rowMin[y]) rowMin[y] = x;
      if (x > rowMax[y]) rowMax[y] = x;
    }
    const nq = Math.max(1, Math.round(c.h / 4));
    let sil = 0, top = 0, bot = 0;
    for (let y = 0; y < c.h; y++) {
      const width = rowMax[y] >= 0 ? rowMax[y] - rowMin[y] + 1 : 0;
      sil += width;
      if (y < nq) top += width;
      if (y >= c.h - nq) bot += width;
    }
    if (!sil) return null;                                     // keine Fläche messbar
    return { solidity: sil / (c.w * c.h), fill: c.area / sil, wTop: top / nq / c.w, wBot: bot / nq / c.w };
  }

  /** Schritt 4a: Form bestimmen. */
  function shapeOf(s, aspect, cfg) {
    if (s.solidity < cfg.maxSolidityTriangle) {                        // Dreieck oder Raute (beide ≈ 0.5)
      if (s.wTop < cfg.minWidthNarrow && s.wBot < cfg.minWidthNarrow) return 'diamond';  // schmal oben und unten
      if (Math.abs(s.wTop - s.wBot) < 0.15) return null;               // überall gleich breit → keine Spitze
      return s.wTop < s.wBot ? 'triangleUp' : 'triangleDown';
    }
    if (s.solidity > cfg.minSolidityRect) return 'rect';
    if (aspect >= cfg.minCircleAspect && aspect <= cfg.maxCircleAspect) {   // rund oder achteckig
      const octagon = s.wTop > cfg.minWidthFlat && s.wBot > cfg.minWidthFlat   // Achteck hat oben und unten
        && s.solidity > cfg.minSolidityOctagon && s.fill > cfg.minFillFull;    // eine breite, flache Kante
      return octagon ? 'octagon' : 'circle';
    }
    return null;
  }

  /** Schritt 4b: Farbe + Form → Schildtyp (oder null). */
  function labelOf(color, shape, fill, cfg) {
    if (color === RED) {
      if (shape === 'octagon') return 'stop';
      if (shape === 'triangleDown') return 'vorfahrtGewaehren';
      if (shape === 'triangleUp') return 'warnung';
      if (shape === 'circle') return fill > cfg.minFillFull ? 'einfahrtVerboten' : 'verbot';
      if (shape === 'diamond') return 'vorfahrt';
    } else if (color === BLUE) {
      if (shape === 'circle') return 'gebot';
      if (shape === 'rect') return 'hinweis';
    } else if (color === YELLOW) {
      if (shape === 'diamond') return 'vorfahrtstrasse';
      if (shape === 'rect') return 'ortstafel';
    }
    if (color === GREEN && shape === 'rect') return 'gruenHinweis';
    return null;
  }

  // 5x7-Fontmuster für eine robuste, offlinefähige OCR auf Schildtext.
  const GLYPHS = {
    '0':'01110100011001110101110011000101110','1':'00100011000010000100001000010001110',
    '2':'01110100010000100010001000100011111','3':'11110000010000101110000010000111110',
    '4':'00010001100101010010111110001000010','5':'11111100001000011110000010000111110',
    '6':'00110010001000011110100011000101110','7':'11111000010001000100010000100001000',
    '8':'01110100011000101110100011000101110','9':'01110100011000101111000010001001100',
    'A':'01110100011000111111100011000110001','B':'11110100011000111110100011000111110',
    'C':'01111100001000010000100001000001111','D':'11110100011000110001100011000111110',
    'E':'11111100001000011110100001000011111','F':'11111100001000011110100001000010000',
    'G':'01111100001000010111100011000101111','H':'10001100011000111111100011000110001',
    'I':'01110001000010000100001000010001110','J':'00111000100001000010100101001001100',
    'K':'10001100101010011000101001001010001','L':'10000100001000010000100001000011111',
    'M':'10001110111010110101100011000110001','N':'10001110011010110011100011000110001',
    'O':'01110100011000110001100011000101110','P':'11110100011000111110100001000010000',
    'Q':'01110100011000110001101011001001101','R':'11110100011000111110101001001010001',
    'S':'01111100001000001110000010000111110','T':'11111001000010000100001000010000100',
    'U':'10001100011000110001100011000101110','V':'10001100011000110001100010101000100',
    'W':'10001100011000110101101011101110001','X':'10001100010101000100010101000110001',
    'Y':'10001100010101000100001000010000100','Z':'11111000010001000100010001000011111'
  };

  function readText(data, iw, ih, box) {
    const x0 = Math.max(0, box.x + Math.round(box.w * 0.13)), x1 = Math.min(iw, box.x + box.w - Math.round(box.w * 0.13));
    const y0 = Math.max(0, box.y + Math.round(box.h * 0.18)), y1 = Math.min(ih, box.y + box.h - Math.round(box.h * 0.18));
    if (x1 - x0 < 12 || y1 - y0 < 12) return { text: '', textConf: 0 };
    // Vordergrund ist die seltenere helle/dunkle Pixelgruppe; farbige Grundflächen werden ignoriert.
    const lum = new Uint8Array((x1-x0)*(y1-y0)); let bright=0;
    for(let y=y0;y<y1;y++) for(let x=x0;x<x1;x++) { const p=(y*iw+x)*4, l=.299*data[p]+.587*data[p+1]+.114*data[p+2]; lum[(y-y0)*(x1-x0)+x-x0]=l; if(l>150) bright++; }
    const threshold=bright < lum.length*.5 ? 150 : 95, mask=new Uint8Array(lum.length);
    for(let i=0;i<lum.length;i++) mask[i]=bright < lum.length*.5 ? +(lum[i]>threshold) : +(lum[i]<threshold);
    const ww=x1-x0, hh=y1-y0, seen=new Uint8Array(mask.length), glyphs=[];
    for(let i=0;i<mask.length;i++) if(mask[i]&&!seen[i]) {
      const q=[i]; seen[i]=1; let minx=ww,maxx=0,miny=hh,maxy=0;
      for(let j=0;j<q.length;j++){const p=q[j],x=p%ww,y=(p/ww)|0; minx=Math.min(minx,x);maxx=Math.max(maxx,x);miny=Math.min(miny,y);maxy=Math.max(maxy,y);
        for(const n of [x? p-1:-1,x<ww-1?p+1:-1,y?p-ww:-1,y<hh-1?p+ww:-1]) if(n>=0&&mask[n]&&!seen[n]){seen[n]=1;q.push(n);}}
      if(q.length>=3 && maxy-miny>=Math.max(5,hh*.18)) glyphs.push({minx,maxx,miny,maxy});
    }
    glyphs.sort((a,b)=>a.minx-b.minx); if(glyphs.length>18) return {text:'',textConf:0};
    let out='', scores=[];
    for(const g of glyphs){const gw=g.maxx-g.minx+1, gh=g.maxy-g.miny+1; let best='?', bs=0;
      for(const [ch,pat] of Object.entries(GLYPHS)){let hit=0,total=35;
        for(let yy=0;yy<7;yy++)for(let xx=0;xx<5;xx++){const sx=g.minx+Math.min(gw-1,Math.floor((xx+.5)*gw/5)),sy=g.miny+Math.min(gh-1,Math.floor((yy+.5)*gh/7)); if(mask[sy*ww+sx]===+(pat[yy*5+xx]==='1'))hit++;}
        const score=hit/total;if(score>bs){bs=score;best=ch;}}
      if(bs>=.56){out+=best;scores.push(bs);} else if(out) out+=' ';
    }
    out=out.trim(); return {text:out,textConf:scores.length?scores.reduce((a,b)=>a+b,0)/scores.length:0};
  }

  /** Sicherheit 0..1: wie nah liegt die gemessene Füllung am Idealwert der Form? */
  const IDEAL = { circle: 0.785, octagon: 0.828, triangleUp: 0.5, triangleDown: 0.5, diamond: 0.5, rect: 1 };
  const confidence = (shape, solidity) => Math.max(0, Math.min(1, 1 - Math.abs(solidity - IDEAL[shape]) * 3));

  /** Ein Bild analysieren. `data` ist RGBA (ImageData.data). */
  function detect(data, w, h, options) {
    const cfg = Object.assign({}, CONFIG, options);
    const mask = buildMask(data, w, h, cfg);
    const { comps, queue } = findComponents(mask, w, h, cfg);
    let detections = [];
    for (const c of comps) {
      const s = analyzeShape(c, queue, w);
      if (!s) continue;
      const shape = shapeOf(s, c.w / c.h, cfg);
      const label = shape && labelOf(c.color, shape, s.fill, cfg);
      if (label) detections.push(Object.assign({ label, shape, x: c.x, y: c.y, w: c.w, h: c.h, conf: confidence(shape, s.solidity) }, readText(data,w,h,c)));
    }
    detections = mergeOverlaps(detections, cfg);
    detections.sort((a, b) => b.w * b.h - a.w * a.h || b.conf - a.conf);   // größte Schilder zuerst
    return { detections: detections.slice(0, cfg.maxResults), mask, count: detections.length };
  }

  /**
   * Schritt 5: Doppelte Treffer zusammenfassen. Ein Schild liefert manchmal zwei Flächen
   * (z. B. weil ein Mast, ein Schatten oder eine Strebe die Farbfläche trennt, oder weil
   * eine Fläche in einer anderen liegt). Ohne diesen Schritt stünde dasselbe Schild
   * zweimal im Ergebnis. Verschmolzen wird nur, was denselben Typ hat.
   */
  function overlapRatio(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    if (inter <= 0) return { iou: 0, contain: 0 };
    const aa = a.w * a.h, ba = b.w * b.h;
    return { iou: inter / (aa + ba - inter), contain: inter / Math.min(aa, ba) };
  }

  function mergeOverlaps(dets, cfg) {
    const out = [], rest = dets.slice();
    while (rest.length) {
      const a = rest.shift();
      for (let changed = true; changed;) {
        changed = false;
        for (let i = 0; i < rest.length; i++) {
          const b = rest[i];
          if (b.label !== a.label) continue;
          const o = overlapRatio(a, b);
          if (o.iou < cfg.mergeIou && o.contain < cfg.mergeContain) continue;
          const x2 = Math.max(a.x + a.w, b.x + b.w), y2 = Math.max(a.y + a.h, b.y + b.h);
          a.x = Math.min(a.x, b.x); a.y = Math.min(a.y, b.y);
          a.w = x2 - a.x; a.h = y2 - a.y;
          a.conf = Math.max(a.conf, b.conf);
          rest.splice(i, 1); i--; changed = true;
        }
      }
      out.push(a);
    }
    return out;
  }

  /** Schritt 6: Treffer über Bilder hinweg verfolgen. Ein Schild gilt erst nach `minHits` Treffern als sicher. */
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  function createTracker(minHits = 3, maxMiss = 3) {
    let tracks = [], nextId = 1;
    return {
      update(dets) {
        tracks.forEach(t => { t.matched = false; });
        for (const d of dets) {
          let best = null, bestIou = 0.25;
          for (const t of tracks) {
            if (t.matched || t.label !== d.label) continue;
            const o = iou(t, d);
            if (o > bestIou) { bestIou = o; best = t; }
          }
          if (best) {
            best.matched = true; best.hits++; best.miss = 0;
            for (const k of ['x', 'y', 'w', 'h']) best[k] = best[k] * 0.5 + d[k] * 0.5;   // Box glätten
            best.conf = best.conf * 0.7 + d.conf * 0.3;
          } else tracks.push(Object.assign({ id: nextId++, hits: 1, miss: 0, matched: true }, d));
        }
        tracks.forEach(t => { if (!t.matched) t.miss++; });
        tracks = tracks.filter(t => t.miss <= maxMiss);
        return tracks.filter(t => t.hits >= minHits);
      },
      /** Alles vergessen – z. B. nach einem Auflösungswechsel oder einem neuen Foto. */
      reset() { tracks = []; nextId = 1; }
    };
  }

  const api = { CONFIG, SIGNS, detect, createTracker, mergeOverlaps, shapeOf, RED, BLUE, YELLOW, GREEN, readText };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SignDetector = api;
})(typeof window !== 'undefined' ? window : globalThis);
