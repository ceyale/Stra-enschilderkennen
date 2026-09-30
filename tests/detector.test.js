/* Test mit synthetisch gezeichneten Schildern: node tests/detector.test.js */
const { detect } = require('../src/detector.js');
const N = 100, RED = [200, 30, 40], BLUE = [20, 90, 180], YEL = [245, 195, 0], WHITE = [255, 255, 255];

function render(colorAt) {                       // colorAt(x, y) → [r,g,b] oder null (Hintergrund grau)
  const d = new Uint8ClampedArray(N * N * 4);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const c = colorAt(x, y) || [128, 128, 128], i = (y * N + x) * 4;
    d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
  }
  return d;
}
const dist = (x, y) => Math.hypot(x - 50, y - 50);
const tri = (x, y, top, bottom, half) => y >= top && y <= bottom && Math.abs(x - 50) <= (y - top) / (bottom - top) * half;
const triRing = (x, y) => tri(x, y, 15, 85, 40) ? (tri(x, y, 32, 76, 25.1) ? WHITE : RED) : null;
const octa = (x, y) => { const a = Math.abs(x - 50), b = Math.abs(y - 50); return a <= 40 && b <= 40 && a + b <= 56.6; };

const cases = {
  verbot:            (x, y) => dist(x, y) <= 40 ? (dist(x, y) > 30 ? RED : WHITE) : null,
  warnung:           triRing,
  vorfahrtGewaehren: (x, y) => triRing(x, N - 1 - y),
  stop:              (x, y) => octa(x, y) ? (Math.abs(y - 50) < 6 && Math.abs(x - 50) < 25 ? WHITE : RED) : null,
  einfahrtVerboten:  (x, y) => dist(x, y) <= 38 ? (Math.abs(y - 50) < 6 && Math.abs(x - 50) < 25 ? WHITE : RED) : null,
  gebot:             (x, y) => dist(x, y) <= 38 ? BLUE : null,
  hinweis:           (x, y) => x >= 20 && x <= 80 && y >= 20 && y <= 80 ? BLUE : null,
  vorfahrtstrasse:   (x, y) => Math.abs(x - 50) + Math.abs(y - 50) <= 40 ? YEL : null
};

let failed = 0;
for (const [expected, fn] of Object.entries(cases)) {
  const got = detect(render(fn), N, N, {}).detections.map(d => d.label);
  const ok = got.length === 1 && got[0] === expected;
  if (!ok) failed++;
  console.log((ok ? 'PASS ' : 'FAIL ') + expected + (ok ? '' : ' → erkannt: [' + got + ']'));
}
process.exit(failed ? 1 : 0);
