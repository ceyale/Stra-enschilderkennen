/* Tests der Modell-Mathematik (Letterbox, Decode, NMS, Rückrechnung).
   Muss wortgleich zu tools/detmath.py passen: node tests/model.test.js */
const M = require('../src/model.js');

let failed = 0;
function check(name, ok, extra) {
  if (!ok) failed++;
  console.log((ok ? 'PASS ' : 'FAIL ') + name + (ok ? '' : ' → ' + extra));
}
const near = (a, b, eps) => Math.abs(a - b) <= (eps === undefined ? 1e-6 : eps);

/* ---------- 1. Letterbox: gleiche Formel wie tools/detmath.py ---------- */
const lb1 = M.letterboxParams(1280, 720, 320);                 // Querformat 16:9
check('letterbox 1280x720 -> s=0.25, padY=70',
  near(lb1.s, 0.25) && lb1.nw === 320 && lb1.nh === 180 && lb1.padX === 0 && lb1.padY === 70,
  JSON.stringify(lb1));

const lb2 = M.letterboxParams(640, 480, 320);                  // 4:3
check('letterbox 640x480 -> s=0.5, padY=40',
  near(lb2.s, 0.5) && lb2.nw === 320 && lb2.nh === 240 && lb2.padX === 0 && lb2.padY === 40,
  JSON.stringify(lb2));

const lb3 = M.letterboxParams(720, 1280, 320);                 // Hochformat (Handy)
check('letterbox 720x1280 -> padX=70, padY=0',
  near(lb3.s, 0.25) && lb3.nw === 180 && lb3.nh === 320 && lb3.padX === 70 && lb3.padY === 0,
  JSON.stringify(lb3));

/* ---------- 2. Decode einer Zelle ---------- */
const gh = 2, gw = 2, stride = 8, nCls = 2, plane = gh * gw, nCh = 5 + nCls;
const data = new Float32Array(nCh * plane);
// Wichtig: leere Zellen brauchen einen klar negativen Obj-Logit. Bei Logit 0 waere
// sigmoid(0)=0.5 -> Score 0.25 und damit genau an der Schwelle (echte Modelle haben
// durch den Bias-Prior -4.6 einen Wert von ~0.01).
data.fill(-20, 4 * plane, 5 * plane);
const i = 1 * gw + 0;                                          // Zelle (gy=1, gx=0)
data[0 * plane + i] = 0;                 // tx: sigmoid(0) = 0.5
data[1 * plane + i] = 0;                 // ty: sigmoid(0) = 0.5
data[2 * plane + i] = Math.log(2);       // tw: w = 2 * stride = 16
data[3 * plane + i] = 0;                 // th: h = 1 * stride = 8
data[4 * plane + i] = 10;                // obj: ~1
data[5 * plane + i] = -10;               // Klasse 0
data[6 * plane + i] = 10;                // Klasse 1

const dets = M.decodeLevel(data, gh, gw, stride, nCls, 0.25);
check('decodeLevel findet genau eine Box', dets.length === 1, dets.length);
if (dets.length === 1) {
  const d = dets[0];
  check('Klasse = 1', d.labelIdx === 1, d.labelIdx);
  check('Score ~ 1', d.score > 0.999, d.score);
  check('cx = (0+0.5)*8 = 4', near(d.x0 + (d.x1 - d.x0) / 2, 4, 1e-4), d.x0 + (d.x1 - d.x0) / 2);
  check('cy = (1+0.5)*8 = 12', near(d.y0 + (d.y1 - d.y0) / 2, 12, 1e-4), d.y0 + (d.y1 - d.y0) / 2);
  check('w = exp(log 2)*8 = 16', near(d.x1 - d.x0, 16, 1e-4), d.x1 - d.x0);
  check('h = exp(0)*8 = 8', near(d.y1 - d.y0, 8, 1e-4), d.y1 - d.y0);
  check('Box = (-4, 8, 12, 16)', near(d.x0, -4, 1e-4) && near(d.y0, 8, 1e-4)
    && near(d.x1, 12, 1e-4) && near(d.y1, 16, 1e-4), JSON.stringify(d));
}

/* ---------- 3. Ausgabe der Stufen + NMS ---------- */
const zero = () => new Float32Array(nCh * plane);
const t8 = { data: zero(), dims: [1, nCh, gh, gw] };
t8.data.set(data);                                             // eine Box auf Stufe 8
const t16 = { data: zero(), dims: [1, nCh, 1, 1] };
t16.data[4 * 1 + 0] = -20;                                     // Stufe 16 bleibt leer
const all = M.decodeAll([t8, t16], [8, 16], nCls, 0.25, 0.45);
check('decodeAll nutzt beide Stufen, leere Stufe liefert nichts', all.length === 1, all.length);

const a = { labelIdx: 0, score: 0.9, x0: 0, y0: 0, x1: 10, y1: 10 };
const b = { labelIdx: 0, score: 0.8, x0: 1, y0: 1, x1: 11, y1: 11 };   // IoU > 0.45
const c = { labelIdx: 1, score: 0.7, x0: 0, y0: 0, x1: 10, y1: 10 };   // andere Klasse
check('NMS unterdrueckt Doppel in gleicher Klasse', M.nms([a, b], 0.45).length === 1);
check('NMS behaelt andere Klasse', M.nms([a, c], 0.45).length === 2);

/* ---------- 4. Rueckrechnung ins Bild (Letterbox rueckwaerts) ---------- */
const labels = ['verbot', 'stop'];
const img = M.toImageCoords([{ labelIdx: 0, score: 1, x0: 0, y0: 48, x1: 32, y1: 80 }], lb2, labels);
check('Rueckrechnung: x=0, y=16, w=64, h=64',
  img.length === 1 && img[0].label === 'verbot' && img[0].x === 0 && img[0].y === 16
  && img[0].w === 64 && img[0].h === 64, JSON.stringify(img));

/* ---------- 5. Schwellwert wirkt ---------- */
const weak = zero();
weak.set(data);
weak[4 * plane + i] = -20;                                     // obj praktisch 0
check('Objektschwelle verwirft schwache Zellen',
  M.decodeLevel(weak, gh, gw, stride, nCls, 0.25).length === 0);

/* ---------- 6. Gegenprobe gegen die Python-Seite (echtes Modell, echtes Fixture) ----------
   tests/fixtures/model-out.json enthaelt rohe ONNX-Tensoren + die in Python berechneten
   Erkennungen (tools/export_fixture.py). Weicht die JS-Dekodierung ab, laeuft der Browser
   anders als das Training - genau der Fehler, den dieser Test verhindert. */
const fs = require('fs');
const path = require('path');
const fixturePath = path.join(__dirname, 'fixtures', 'model-out.json');
if (fs.existsSync(fixturePath)) {
  const fx = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  fx.cases.forEach((c, ci) => {
    const tensors = c.tensors.map(t => ({ dims: t.dims, data: Float32Array.from(t.data) }));
    const got = M.decodeAll(tensors, fx.levels, fx.classes.length, fx.conf, fx.iou);
    const want = c.expected_model_coords;
    check(`fixture ${ci}: Anzahl Erkennungen ${want.length}`, got.length === want.length,
      got.length + ' statt ' + want.length);
    if (got.length === want.length) {
      let ok = true, detail = '';
      got.forEach((g, k) => {
        const w = want[k];
        const d = Math.max(Math.abs(g.x0 - w.x0), Math.abs(g.y0 - w.y0),
                           Math.abs(g.x1 - w.x1), Math.abs(g.y1 - w.y1), Math.abs(g.score - w.score));
        if (g.labelIdx !== w.labelIdx || d > 0.01) { ok = false; detail = JSON.stringify({ got: g, want: w }); }
      });
      check(`fixture ${ci}: Klassen, Boxen und Scores identisch zu Python`, ok, detail);
    }
    const lb = M.letterboxParams(c.source.width, c.source.height, fx.size);
    check(`fixture ${ci}: Letterbox wie Python (s, padX, padY)`,
      near(lb.s, c.letterbox.s, 1e-6) && lb.padX === c.letterbox.padX && lb.padY === c.letterbox.padY,
      JSON.stringify({ js: lb, py: c.letterbox }));
    const img = M.toImageCoords(got, lb, fx.classes);
    const wi = c.expected_image_coords;
    let okImg = img.length === wi.length, detailImg = '';
    img.forEach((g, k) => {
      if (k >= wi.length) return;
      const w = wi[k];
      const d = Math.max(Math.abs(g.x - w.x0), Math.abs(g.y - w.y0),
                         Math.abs(g.x + g.w - w.x1), Math.abs(g.y + g.h - w.y1));
      if (g.label !== fx.classes[w.labelIdx] || d > 0.01) {
        okImg = false;
        detailImg = JSON.stringify({ got: g, want: w });
      }
    });
    check(`fixture ${ci}: Rueckrechnung ins Bild wie Python`, okImg, detailImg);
  });
} else {
  console.log('HINWEIS fixtures/model-out.json fehlt – Cross-Language-Test uebersprungen');
  console.log('        (erzeugen mit: python tools/export_fixture.py)');
}

process.exit(failed ? 1 : 0);
