/**
 * app.js – Oberfläche: Kamera, Foto-Import, Zeichnen der Ergebnisse.
 * Die eigentliche Erkennung passiert in detector.js.
 */
(function () {
  'use strict';
  const D = SignDetector;
  const $ = id => document.getElementById(id);
  const view = $('view'), ctx = view.getContext('2d');           // sichtbares Bild + Rahmen
  const work = document.createElement('canvas');                  // kleines Bild für die Analyse
  const wctx = work.getContext('2d', { willReadFrequently: true });
  const maskCv = document.createElement('canvas');                // Farbmasken-Ansicht (Debug)
  const maskCtx = maskCv.getContext('2d');
  const video = $('video'), list = $('list'), statusEl = $('status'), saveBtn = $('saveBtn');
  const satEl = $('sat'), resEl = $('res'), maskEl = $('maskCb');

  const VIEW_W = 640;        // Breite der Anzeige
  const INTERVAL_MS = 100;   // höchstens 10 Analysen pro Sekunde
  // Farben der Maskenansicht: je Farbklasse 4 Bytes (RGB + reserviertes Alpha).
  const PALETTE = new Uint8Array([0, 0, 0, 0, 211, 35, 47, 0, 20, 103, 184, 0, 242, 194, 0, 0]);

  let stream = null, running = false, still = null, lastRun = 0, dims = '';
  let lastKey = null;                                        // null = Liste neu aufbauen ('' wäre gültig!)
  let tracker = D.createTracker(), current = [], mask = null;
  let workW = +resEl.value;                                  // Breite der Analyse („Reichweite“)
  let lastMs = 0, fps = 0, lastFrame = 0, lastStatus = '';
  let maskImg = null;                                        // wiederverwendeter Puffer der Maskenansicht

  /** Statuszeile sofort setzen (und den Zwischenspeicher mitziehen). */
  function say(text) { lastStatus = text; statusEl.textContent = text; }

  /** Statuszeile aus dem aktuellen Zustand bauen – nur schreiben, wenn sich etwas ändert. */
  function updateStatus() {
    let txt;
    if (running) txt = 'Kamera läuft · ' + Math.round(lastMs) + ' ms je Analyse · ' + fps.toFixed(1) + ' Bilder/s';
    else if (still) txt = 'Foto analysiert · ' + Math.round(lastMs) + ' ms für die Analyse';
    else txt = 'Kamera starten oder ein Foto wählen.';
    if (txt !== lastStatus) say(txt);
  }

  /** Canvas-Größen an die Bildquelle anpassen (Seitenverhältnis beibehalten). */
  function setSize(sw, sh) {
    const key = sw + 'x' + sh + '@' + workW;
    if (dims === key) return;
    dims = key;
    view.width = VIEW_W; view.height = Math.round(VIEW_W * sh / sw);
    work.width = maskCv.width = workW; work.height = maskCv.height = Math.round(workW * sh / sw);
    maskImg = null;                                   // Puffergröße passt nicht mehr zur Maske
  }

  /** Erkennung auf einer Bildquelle ausführen (Video oder Foto). */
  function analyse(source, useTracker) {
    const t0 = performance.now();
    wctx.drawImage(source, 0, 0, work.width, work.height);
    const img = wctx.getImageData(0, 0, work.width, work.height);
    const res = D.detect(img.data, work.width, work.height, { minSaturation: +satEl.value });
    current = useTracker ? tracker.update(res.detections) : res.detections;
    mask = res.mask;
    lastMs = performance.now() - t0;
    saveBtn.disabled = false;
    renderList();
    updateStatus();
  }

  /** Liste unter dem Bild – nur neu aufbauen, wenn sich Typ, Anzahl oder Sicherheit ändern. */
  function renderList() {
    const groups = new Map();                          // ein Eintrag je Schildtyp, auch bei mehreren Funden
    for (const t of current) {
      const g = groups.get(t.label) || { label: t.label, n: 0, conf: 0 };
      g.n++;
      if (t.conf > g.conf) g.conf = t.conf;
      groups.set(t.label, g);
    }
    const items = [...groups.values()].sort((a, b) => b.conf - a.conf || a.label.localeCompare(b.label));
    const key = items.map(g => g.label + 'x' + g.n + '@' + Math.round(g.conf * 100)).join('|');
    if (key === lastKey) return;
    lastKey = key;
    list.innerHTML = items.length
      ? items.map(g => { const s = D.SIGNS[g.label];
          return `<li style="--c:${s.hex}"><b>${s.name}</b><span>${s.zeichen}${g.n > 1 ? ' · ' + g.n + '×' : ''}</span>`
            + `<span class="conf">${Math.round(g.conf * 100)} %</span><p>${s.note}</p></li>`; }).join('')
      : '<li class="leer">Noch kein Schild erkannt. Halte ein Schild ruhig und frontal ins Bild.</li>';
  }

  /** Debug: erkannte Farbflächen halbtransparent einblenden. */
  function drawMask() {
    if (!maskImg || maskImg.width !== maskCv.width || maskImg.height !== maskCv.height) {
      maskImg = maskCtx.createImageData(maskCv.width, maskCv.height);   // Puffer wiederverwenden
    }
    const px = maskImg.data;
    // Kein Array je Pixel: direkt in den Puffer schreiben (sonst bremst die Müllabfuhr die Analyse).
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
      const c = mask[i];
      if (!c) { px[p + 3] = 0; continue; }
      const o = c * 4;
      px[p] = PALETTE[o]; px[p + 1] = PALETTE[o + 1]; px[p + 2] = PALETTE[o + 2]; px[p + 3] = 255;
    }
    maskCtx.putImageData(maskImg, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.globalAlpha = 0.6;
    ctx.drawImage(maskCv, 0, 0, view.width, view.height);
    ctx.globalAlpha = 1; ctx.imageSmoothingEnabled = true;
  }

  /** Rahmen + Beschriftung; Koordinaten vom Analysebild auf die Anzeige hochrechnen. */
  function drawBoxes() {
    const k = view.width / work.width;
    ctx.lineWidth = 3; ctx.textBaseline = 'top';
    ctx.font = '600 15px Bahnschrift, "DIN Alternate", system-ui, sans-serif';
    for (const t of current) {
      const s = D.SIGNS[t.label], x = t.x * k, y = t.y * k, w = t.w * k, h = t.h * k;
      const txt = s.name + ' ' + Math.round(t.conf * 100) + ' %', tw = ctx.measureText(txt).width + 10;
      const ty = y > 22 ? y - 22 : y + h + 2;
      ctx.strokeStyle = s.hex; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = s.hex; ctx.fillRect(x, ty, tw, 20);
      ctx.fillStyle = s.text; ctx.fillText(txt, x + 5, ty + 3);
    }
  }

  const draw = () => { if ($('maskCb').checked && mask) drawMask(); drawBoxes(); };

  /** Kamera-Schleife: jedes Bild anzeigen, aber nur alle INTERVAL_MS analysieren. */
  function frame(now) {
    if (!running) return;
    if (video.videoWidth) {
      if (lastFrame) fps = fps ? fps * 0.8 + (1000 / (now - lastFrame)) * 0.2 : 1000 / (now - lastFrame);
      lastFrame = now;
      setSize(video.videoWidth, video.videoHeight);
      ctx.drawImage(video, 0, 0, view.width, view.height);
      if (now - lastRun >= INTERVAL_MS) { lastRun = now; analyse(video, true); }
      draw();
    }
    requestAnimationFrame(frame);
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      say('Kein Kamerazugriff. Die Seite muss über HTTPS (z. B. GitHub Pages) geöffnet werden.');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream; await video.play();
      still = null; current = []; lastKey = null; lastRun = 0; lastFrame = 0; fps = 0;
      tracker.reset();
      running = true;
      $('camBtn').textContent = 'Kamera stoppen';
      renderList(); updateStatus();
      requestAnimationFrame(frame);
    } catch (err) {
      say('Kamera nicht verfügbar (' + err.name + '). Zugriff im Browser erlauben?');
    }
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (video.srcObject) { video.srcObject = null; video.load(); }   // Kamera wirklich freigeben
    stream = null; $('camBtn').textContent = 'Kamera starten';
  }

  /** Einzelnes Foto analysieren (ohne Tracker, jeder Treffer zählt sofort). */
  function showStill() {
    setSize(still.naturalWidth || still.width, still.naturalHeight || still.height);
    ctx.drawImage(still, 0, 0, view.width, view.height);
    analyse(still, false); draw();
  }

  /**
   * Foto laden. Bevorzugt createImageBitmap, weil Handyfotos damit aufrecht analysiert werden –
   * die Drehung steckt nur in den EXIF-Daten. Kann der Browser das nicht, greift ein <img>.
   */
  async function loadPhoto(file) {
    stopCamera();
    tracker.reset(); current = []; lastKey = null; lastMs = 0;
    if (typeof createImageBitmap === 'function') {
      try {
        still = await createImageBitmap(file, { imageOrientation: 'from-image' });
        showStill(); updateStatus(); return;
      } catch (err) { /* Option nicht unterstützt → Fallback unten */ }
    }
    const img = new Image();
    img.onload = () => { still = img; showStill(); URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(file);
  }

  /** Die Anzeige (Bild mit Rahmen) als PNG sichern. */
  function saveImage() {
    if (!view.toBlob) { say('Speichern unterstützt dieser Browser nicht.'); return; }
    view.toBlob(blob => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'schilder-' + new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19) + '.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    }, 'image/png');
  }

  /** Alles auf Anfang: Kamera aus, Foto weg, Anzeige leer. */
  function reset() {
    stopCamera();
    still = null; mask = null; current = []; lastKey = null; lastMs = 0; fps = 0; dims = '';
    tracker.reset();
    view.width = VIEW_W; view.height = Math.round(VIEW_W * 3 / 4);
    ctx.clearRect(0, 0, view.width, view.height);
    saveBtn.disabled = true;
    renderList();
    say('Kamera starten oder ein Foto wählen.');
  }

  $('camBtn').addEventListener('click', () => {
    if (running) { stopCamera(); updateStatus(); } else startCamera();
  });
  $('saveBtn').addEventListener('click', saveImage);
  $('resetBtn').addEventListener('click', reset);
  $('file').addEventListener('change', e => {
    const f = e.target.files[0];
    e.target.value = '';                       // dasselbe Foto lässt sich danach erneut wählen
    if (f) loadPhoto(f);
  });
  satEl.addEventListener('input', () => {
    $('satOut').textContent = (+satEl.value).toFixed(2);
    if (still) showStill();
  });
  resEl.addEventListener('change', () => {
    workW = +resEl.value; dims = '';           // Boxen liegen in Analysepixeln → Größen neu setzen
    tracker.reset(); lastKey = null;
    if (still) showStill(); else renderList();
  });
  maskEl.addEventListener('input', () => { if (still) showStill(); });
  renderList();
})();
