/**
 * model.js – optionaler KI-Modus: hybrides CNN+Transformer-Netz über ONNX Runtime Web.
 *
 * Das Netz sagt in EINEM Durchlauf Position (Box) UND Art (9 Schildtypen) voraus
 * (anchor-free, vier Stufen stride 4/8/16/32, 14 Kanäle je Zelle). Die Stufenliste steht
 * in `models/labels.json` (`levels`) – dieses Skript hier ist absichtlich generisch
 * darüber, damit eine geänderte Stufenzahl keine JS-Änderung erfordert. Trainiert wird es
 * mit tools/train_det.py, exportiert mit tools/export_onnx.py.
 *
 * Aufbau dieser Datei:
 *   1. Reine Mathematik: letterbox, decode, NMS – KEINE Laufzeit-Abhängigkeit,
 *      läuft in Node (tests/model.test.js) und muss wortgleich zu tools/detmath.py sein.
 *   2. Laufzeit: ONNX Runtime Web laden und die Mathematik darum herum aufrufen.
 *      Wird nur benutzt, wenn `onnxruntime.min.js` eingebunden ist; fehlt sie,
 *      bleibt der Heuristik-Pfad in detector.js aktiv (kein Bruch, nur weniger Qualität).
 *
 * Kanalreihenfolge je Zelle: [tx, ty, tw, th, obj, cls0..cls8] (siehe models/labels.json)
 *   cx = (gx + sigmoid(tx)) * stride      cy = (gy + sigmoid(ty)) * stride
 *   w  = exp(clip(tw, -8, 8)) * stride    h  = exp(clip(th, -8, 8)) * stride
 *   score = sigmoid(obj) * max_j sigmoid(cls_j)
 */
(function (root) {
  'use strict';

  const DEFAULTS = { conf: 0.25, iou: 0.45, labelsUrl: 'models/labels.json' };

  /** Letterbox-Kennzahlen – identisch zu tools/detmath.py (letterbox_params). */
  function letterboxParams(w, h, size) {
    const s = Math.min(size / w, size / h);
    const nw = Math.round(w * s), nh = Math.round(h * s);
    return { s, nw, nh, padX: Math.round((size - nw) / 2), padY: Math.round((size - nh) / 2) };
  }

  const sigmoid = x => 1 / (1 + Math.exp(-x));
  const expClipped = x => Math.exp(Math.max(-8, Math.min(8, x)));

  /** IoU zweier Boxen (x0,y0,x1,y1). */
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
    const iy = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
    const inter = ix * iy;
    const union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
    return union > 0 ? inter / union : 0;
  }

  /**
   * Kopf-Ausgang einer Stufe auswerten.
   * @param {Float32Array} data  flacher Tensor (14, H, W)
   * @param {number} gh, gw      Rastergröße
   * @param {number} stride      8, 16 oder 32
   * @param {number} nCls        Anzahl Klassen
   */
  function decodeLevel(data, gh, gw, stride, nCls, conf) {
    const plane = gh * gw, out = [];
    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        const i = gy * gw + gx;
        const obj = sigmoid(data[4 * plane + i]);
        if (obj < conf) continue;                    // billiger Vorfilter vor der Klassenrunde
        let best = -1, bestIdx = 0;
        for (let c = 0; c < nCls; c++) {
          const p = sigmoid(data[(5 + c) * plane + i]);
          if (p > best) { best = p; bestIdx = c; }
        }
        const score = obj * best;
        if (score < conf) continue;
        const cx = (gx + sigmoid(data[i])) * stride;
        const cy = (gy + sigmoid(data[plane + i])) * stride;
        const w = expClipped(data[2 * plane + i]) * stride;
        const h = expClipped(data[3 * plane + i]) * stride;
        out.push({ labelIdx: bestIdx, score: score,
                   x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2 });
      }
    }
    return out;
  }

  /** NMS pro Klasse – identisch zu tools/detmath.py (nms). */
  function nms(dets, iouThres) {
    const keep = [];
    const classes = [...new Set(dets.map(d => d.labelIdx))];
    for (const cls of classes) {
      const group = dets.filter(d => d.labelIdx === cls).sort((a, b) => b.score - a.score);
      while (group.length) {
        const best = group.shift();
        keep.push(best);
        for (let i = group.length - 1; i >= 0; i--) if (iou(best, group[i]) > iouThres) group.splice(i, 1);
      }
    }
    return keep.sort((a, b) => b.score - a.score);
  }

  /** Alle Stufen auswerten und unterdrücken. `tensors` = je Stufe ein Tensor {data, dims}. */
  function decodeAll(tensors, levels, nCls, conf, iouThres) {
    let dets = [];
    for (let i = 0; i < tensors.length; i++) {
      const t = tensors[i];
      const dims = t.dims;                        // [1, 14, H, W]
      const gh = dims[dims.length - 2], gw = dims[dims.length - 1];
      dets = dets.concat(decodeLevel(t.data, gh, gw, levels[i], nCls, conf));
    }
    return nms(dets, iouThres);
  }

  /** Modell-Koordinaten (Letterbox-Quadrat) zurück ins Analysenbild rechnen.
   *  Bewusst ohne Rundung: die Math-Schicht bleibt exakt (prüfbar gegen Python),
   *  gerundet wird erst beim Zeichnen. */
  function toImageCoords(dets, lb, labels) {
    return dets.map(d => ({
      label: labels[d.labelIdx],
      x: (d.x0 - lb.padX) / lb.s,
      y: (d.y0 - lb.padY) / lb.s,
      w: (d.x1 - d.x0) / lb.s,
      h: (d.y1 - d.y0) / lb.s,
      conf: Math.max(0, Math.min(1, d.score)),
      shape: 'cnn', viaModel: true,
    }));
  }

  const math = { letterboxParams, sigmoid, iou, decodeLevel, nms, decodeAll, toImageCoords, DEFAULTS };
  if (typeof module !== 'undefined' && module.exports) module.exports = math;
  root.SignModelMath = math;

  /* ------------------------------------------------------------------ *
   * Laufzeit (ONNX Runtime Web). Nur aktiv, wenn `ort` global vorhanden ist.
   * ------------------------------------------------------------------ */

  /**
   * Modell laden.
   * @returns {Promise<{ok:boolean, reason?:string, backend?:string, labels?:object,
   *                    detect?:function}>}
   */
  async function load(opts) {
    const o = Object.assign({}, DEFAULTS, opts || {});
    if (typeof root.ort === 'undefined' || !root.ort.InferenceSession) {
      return { ok: false, reason: 'onnxruntime-web ist nicht geladen (Skript fehlt oder offline)' };
    }
    let labels;
    try {
      const res = await fetch(o.labelsUrl, { cache: 'no-cache' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      labels = await res.json();
    } catch (err) {
      return { ok: false, reason: 'models/labels.json fehlt (' + err.message + ') – noch kein Modell trainiert?' };
    }

    const ort = root.ort;
    // GitHub Pages erlaubt keine COOP/COEP-Header -> kein Mehrthread-WASM.
    // Einthreadig bleiben die Ergebnisse reproduzierbar (siehe docs/TRAINING.md).
    ort.env.wasm.numThreads = 1;
    if (o.wasmPaths && ort.env.wasm) ort.env.wasm.wasmPaths = o.wasmPaths;

    const file = labels.files.int8 || labels.files.onnx;
    const url = o.modelUrl || ('models/' + file);
    let session = null, backend = '';
    for (const eps of [['webgpu', 'wasm'], ['wasm']]) {
      try {
        session = await ort.InferenceSession.create(url, {
          executionProviders: eps,
          graphOptimizationLevel: 'all',
        });
        backend = eps.join('→');
        break;
      } catch (err) {
        if (eps.length === 1) return { ok: false, reason: 'Modell nicht ladbar: ' + err.message };
      }
    }

    const size = labels.size;
    const nCls = labels.classes.length;
    const conf = o.conf, iouThres = o.iou;
    // Eingabepuffer je Größe einmal anlegen. Das Modell wird mit offener Höhe/Breite
    // exportiert (tools/export_onnx.py --dynamic), deshalb darf der Aufrufer die Größe
    // wählen. Gemessen auf 2000 val-Bildern: 256 px F1 0,796 / 320 px 0,837 / 384 px 0,854
    // – größer ist genauer, kleiner schneller (docs/TRAINING.md §3).
    const bufs = new Map();
    const chwFor = sz => {
      let b = bufs.get(sz);
      if (!b) { b = new Float32Array(3 * sz * sz); bufs.set(sz, b); }
      return b;
    };

    /** Bilddaten eines sz×sz-Letterbox-Canvas auswerten. `inputSize` ist optional. */
    async function detect(imageData, inputSize) {
      const sz = inputSize || size;
      const p0 = sz * sz;
      const chw = chwFor(sz);
      const d = imageData.data;
      for (let i = 0, p = 0; i < p0; i++, p += 4) {
        chw[i] = d[p] / 255;
        chw[p0 + i] = d[p + 1] / 255;
        chw[2 * p0 + i] = d[p + 2] / 255;
      }
      const feeds = {};
      feeds[labels.input || 'images'] = new ort.Tensor('float32', chw, [1, 3, sz, sz]);
      const out = await session.run(feeds);
      const tensors = labels.levels.map(s => out['os' + s]);
      return decodeAll(tensors, labels.levels, nCls, conf, iouThres);
    }

    return { ok: true, backend: backend, labels: labels, size: size, detect: detect,
             session: session };
  }

  /** Letterbox in einen Canvas zeichnen (Grau 114 wie im Training). */
  function drawLetterbox(ctx, source, srcW, srcH, size) {
    const lb = letterboxParams(srcW, srcH, size);
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    if (ctx.imageSmoothingQuality) ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, srcW, srcH, lb.padX, lb.padY, lb.nw, lb.nh);
    return lb;
  }

  root.SignModel = { load: load, math: math, drawLetterbox: drawLetterbox, DEFAULTS: DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
