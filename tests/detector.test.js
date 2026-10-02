/*
 * Test der Erkennung mit synthetisch gezeichneten Schildern.
 *   Node:    node tests/detector.test.js
 *   Browser: tests/index.html öffnen (dann ist kein Node nötig)
 * Rückgabewert im Node-Betrieb: 0 = alles bestanden, 1 = mindestens ein Fehler.
 */
(function () {
  'use strict';
  const D = (typeof require === 'function')
    ? require('../src/detector.js')
    : (typeof window !== 'undefined' ? window.SignDetector : globalThis.SignDetector);
  const { detect, createTracker, mergeOverlaps, CONFIG } = D;

  const N = 100;
  const RED = [200, 30, 40], BLUE = [20, 90, 180], YEL = [245, 195, 0], WHITE = [255, 255, 255];
  const results = [];
  const check = (name, ok, detail) => results.push({ name, ok: !!ok, detail: detail || '' });

  /** Bild zeichnen: colorAt(x, y) → [r,g,b] oder null (grauer Hintergrund). */
  function render(colorAt, n) {
    n = n || N;
    const d = new Uint8ClampedArray(n * n * 4);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const c = colorAt(x, y) || [128, 128, 128], i = (y * n + x) * 4;
      d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
    }
    return d;
  }
  const found = colorAt => detect(render(colorAt), N, N, {}).detections.map(d => d.label);

  const dist = (x, y) => Math.hypot(x - 50, y - 50);
  const tri = (x, y, top, bottom, half) => y >= top && y <= bottom && Math.abs(x - 50) <= (y - top) / (bottom - top) * half;
  const triRing = (x, y) => tri(x, y, 15, 85, 40) ? (tri(x, y, 32, 76, 25.1) ? WHITE : RED) : null;
  const octa = (x, y) => { const a = Math.abs(x - 50), b = Math.abs(y - 50); return a <= 40 && b <= 40 && a + b <= 56.6; };

  /* ---------- 1. Alle neun Schildtypen ---------- */

  const cases = {
    verbot:            (x, y) => dist(x, y) <= 40 ? (dist(x, y) > 30 ? RED : WHITE) : null,
    warnung:           triRing,
    vorfahrtGewaehren: (x, y) => triRing(x, N - 1 - y),
    stop:              (x, y) => octa(x, y) ? (Math.abs(y - 50) < 6 && Math.abs(x - 50) < 25 ? WHITE : RED) : null,
    einfahrtVerboten:  (x, y) => dist(x, y) <= 38 ? (Math.abs(y - 50) < 6 && Math.abs(x - 50) < 25 ? WHITE : RED) : null,
    gebot:             (x, y) => dist(x, y) <= 38 ? BLUE : null,
    hinweis:           (x, y) => x >= 20 && x <= 80 && y >= 20 && y <= 80 ? BLUE : null,
    vorfahrtstrasse:   (x, y) => Math.abs(x - 50) + Math.abs(y - 50) <= 40 ? YEL : null,
    ortstafel:         (x, y) => x >= 25 && x <= 75 && y >= 32 && y <= 68 ? YEL : null   // fehlte in der ersten Version
  };

  for (const [expected, fn] of Object.entries(cases)) {
    const got = found(fn);
    check('erkennt ' + expected, got.length === 1 && got[0] === expected, 'erkannt: [' + got + ']');
  }

  /* ---------- 2. Fälle, die KEIN Schild sein dürfen ---------- */

  check('graues Bild ergibt keinen Treffer', found(() => null).length === 0);
  check('winziger roter Fleck ist zu klein', found((x, y) => (x < 6 && y < 6 ? RED : null)).length === 0);
  check('bildfüllendes Rot wird verworfen', found(() => RED).length === 0);
  check('rotes Quadrat ist kein Schildtyp', found((x, y) => (x >= 30 && x <= 70 && y >= 30 && y <= 70 ? RED : null)).length === 0);
  // Ein blaues Rechteck ist dagegen ein Hinweiszeichen – das prüft Fall „hinweis“ oben.
  check('blaues Dreieck ist kein Schildtyp',
    found((x, y) => (tri(x, y, 15, 85, 40) ? BLUE : null)).length === 0);

  /* ---------- 3. Schrägsicht: runder Schild als Ellipse ---------- */

  const ellipse = (hx, hy) => (x, y) => {
    const dx = (x - 50) / hx, dy = (y - 50) / hy;
    return dx * dx + dy * dy <= 1;
  };
  check('schräg gesehener runder Schild (Ellipse 1,27:1)',
    found((x, y) => (ellipse(38, 30)(x, y) ? BLUE : null)).join() === 'gebot');

  /* ---------- 4. Doppelte Flächen zusammenfassen ---------- */

  const box = (label, x, y, w, h) => ({ label, shape: 'rect', x, y, w, h, conf: 0.5 });
  check('verschachtelte Treffer werden einer',
    mergeOverlaps([box('hinweis', 10, 10, 60, 60), box('hinweis', 30, 30, 20, 20)], CONFIG).length === 1);
  check('getrennte Treffer bleiben getrennt',
    mergeOverlaps([box('hinweis', 10, 10, 20, 20), box('hinweis', 60, 60, 20, 20)], CONFIG).length === 2);
  check('verschiedene Typen werden nicht verschmolzen',
    mergeOverlaps([box('hinweis', 10, 10, 60, 60), box('gebot', 30, 30, 20, 20)], CONFIG).length === 2);

  // Ein weißer Rand trennt den Rahmen vom Innenfeld – beides gehört zu einem Schild.
  const nested = (x, y) => {
    const outer = x >= 20 && x <= 80 && y >= 20 && y <= 80;
    const brand = x >= 35 && x <= 65 && y >= 35 && y <= 65;
    const inner = x >= 40 && x <= 60 && y >= 40 && y <= 60;
    return inner ? BLUE : (outer ? (brand ? WHITE : BLUE) : null);
  };
  check('durch weißen Rand getrennte Flächen ergeben einen Treffer',
    found(nested).join() === 'hinweis', 'erkannt: [' + found(nested) + ']');

  /* ---------- 5. Stabilisierung über mehrere Bilder ---------- */

  const det = (x, y, label) => ({ label: label || 'gebot', shape: 'circle', x, y, w: 30, h: 30, conf: 0.9 });

  const t1 = createTracker(3, 2);
  const t1a = t1.update([det(10, 10)]), t1b = t1.update([det(10, 10)]), t1c = t1.update([det(10, 10)]);
  check('Tracker zeigt ein Schild erst nach 3 Treffern',
    t1a.length === 0 && t1b.length === 0 && t1c.length === 1, t1a.length + '/' + t1b.length + '/' + t1c.length);
  const t1d = t1.update([]), t1e = t1.update([]), t1f = t1.update([]);
  check('Tracker vergisst ein Schild nach 3 leeren Bildern',
    t1d.length === 1 && t1e.length === 1 && t1f.length === 0, t1d.length + '/' + t1e.length + '/' + t1f.length);

  const t2 = createTracker(2, 2);
  const t2a = t2.update([det(10, 10)]), t2b = t2.update([det(13, 12)]);
  check('Tracker führt leicht verschobene Boxen weiter', t2a.length === 0 && t2b.length === 1);

  const t3 = createTracker(2, 2);
  t3.update([det(10, 10)]);
  check('Tracker startet für weit entfernte Fläche eine neue Spur', t3.update([det(80, 80)]).length === 0);

  const t4 = createTracker(2, 2);
  t4.update([det(10, 10)]); t4.update([det(10, 10)]);
  t4.reset();
  check('Tracker reset() löscht alle Spuren', t4.update([det(10, 10)]).length === 0);

  const t5 = createTracker(2, 2);
  t5.update([det(10, 10, 'gebot')]);
  check('Tracker verwechselt Typen nicht', t5.update([det(10, 10, 'hinweis')]).length === 0);

  /* ---------- Ergebnis melden ---------- */

  const failed = results.filter(r => !r.ok);
  const head = 'Schilder-Scanner: ' + (results.length - failed.length) + '/' + results.length +
    ' Tests bestanden' + (failed.length ? ' – FEHLER' : ' – OK');
  const line = r => (r.ok ? 'PASS ' : 'FAIL ') + r.name + (r.ok || !r.detail ? '' : ' → ' + r.detail);

  if (typeof process !== 'undefined' && process.exit) {
    console.log(head);
    results.forEach(r => console.log(line(r)));
    process.exit(failed.length ? 1 : 0);
  } else {
    document.title = head;
    const pre = document.createElement('pre');
    pre.id = 'result';
    pre.textContent = head + '\n\n' + results.map(line).join('\n');
    document.body.appendChild(pre);
  }
})();

