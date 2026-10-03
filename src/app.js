/**
 * app.js – Oberfläche: Kamera, Foto-Import, Zeichnen der Ergebnisse.
 * Die eigentliche Erkennung passiert in detector.js.
 */
(function () {
  'use strict';
  const D = SignDetector;
  const $ = id => document.getElementById(id);
  const view = $('view'), ctx = view.getContext('2d');           // sichtbares Bild + Rahmen
  const work = document.createElement('canvas');                  // kleines Bild für die Heuristik
  const wctx = work.getContext('2d', { willReadFrequently: true });
  const maskCv = document.createElement('canvas');                // Farbmasken-Ansicht (Debug)
  const modelCv = document.createElement('canvas');               // Eingang des KI-Modells (Letterbox)
  const mctx = modelCv.getContext('2d', { willReadFrequently: true });
  const video = $('video'), list = $('list'), statusEl = $('status');

  const VIEW_W = 640;       // Breite der Anzeige
  const WORK_W = 240;       // Breite der Heuristik: klein = schnell, groß = erkennt weiter entfernte Schilder
  const INTERVAL_MS = 100;  // höchstens 10 Analysen pro Sekunde

  let stream = null, running = false, still = null, lastRun = 0, lastKey = '', dims = '';
  let tracker = D.createTracker(), current = [], mask = null;
  let model = null;                       // gesetzt, wenn ein ONNX-Modell geladen wurde
  let busy = false;                       // verhindert überlappende Modellläufe
  let srcW = 0, srcH = 0, boxScale = 1;   // Maßstab von Erkennung zu Anzeige

  /** Canvas-Größen an die Bildquelle anpassen (Seitenverhältnis beibehalten). */
  function setSize(sw, sh) {
    if (dims === sw + 'x' + sh) return;
    dims = sw + 'x' + sh;
    view.width = VIEW_W; view.height = Math.round(VIEW_W * sh / sw);
    work.width = maskCv.width = WORK_W; work.height = maskCv.height = Math.round(WORK_W * sh / sw);
  }

  /** Erkennung auf einer Bildquelle ausführen (Video oder Foto): KI-Modell oder Heuristik. */
  /** Eingabegröße des Netzes. "auto" = Video in Modellgröße (320), Foto in 384 px –
   *  gemessen auf 2000 val-Bildern: 256 px F1 0,796 / 320 px 0,837 / 384 px 0,854.
   *  Das Modell wird mit offener Höhe/Breite exportiert, deshalb geht jede dieser Größen. */
  function modelInputSize(kind) {
    const v = $('size') ? $('size').value : 'auto';
    if (v && v !== 'auto') return +v;
    return kind === 'foto' ? 384 : ((model && model.size) || 320);
  }

  function analyse(source, useTracker) {
    if (model) return analyseModel(source, useTracker);
    const t0 = performance.now();
    wctx.drawImage(source, 0, 0, work.width, work.height);
    const img = wctx.getImageData(0, 0, work.width, work.height);
    const res = D.detect(img.data, work.width, work.height, { minSaturation: +$('sat').value });
    current = useTracker ? tracker.update(res.detections) : res.detections;
    mask = res.mask;
    srcW = work.width; srcH = work.height; boxScale = view.width / work.width;
    statusEl.textContent = (running ? 'Kamera läuft' : 'Foto analysiert') + ' · Heuristik · '
      + Math.round(performance.now() - t0) + ' ms pro Analyse';
    renderList();
  }

  /** KI-Modus: ein Netz liefert Position UND Art in einem Durchlauf (src/model.js).
   *  Läuft asynchron – deshalb sperrt `busy` in frame() überlappende Läufe. */
  async function analyseModel(source, useTracker) {
    const sw = source.videoWidth || source.naturalWidth;
    const sh = source.videoHeight || source.naturalHeight;
    if (!sw || !sh) return;
    const sz = modelInputSize(useTracker ? 'video' : 'foto');
    if (modelCv.width !== sz) modelCv.width = modelCv.height = sz;
    const t0 = performance.now();
    SignModel.drawLetterbox(mctx, source, sw, sh, sz);                 // Grau 114 wie im Training
    const img = mctx.getImageData(0, 0, sz, sz);
    const tPrep = performance.now() - t0;
    const dets = await model.detect(img, sz);
    const lb = SignModel.math.letterboxParams(sw, sh, sz);
    const mapped = SignModel.math.toImageCoords(dets, lb, model.labels.classes);
    current = useTracker ? tracker.update(mapped) : mapped;
    mask = null;                                                     // Farbmasken gibt es nur in der Heuristik
    srcW = sw; srcH = sh; boxScale = view.width / sw;
    statusEl.textContent = (running ? 'Kamera läuft' : 'Foto analysiert') + ' · KI-Modell (' + model.backend
      + ', ' + sz + ' px) · ' + Math.round(performance.now() - t0) + ' ms (Vorbereitung '
      + Math.round(tPrep) + ' ms)';
    renderList();
    draw();
  }

  /** Anzeige-Informationen zu einem Label.

   * Seit der Klassenerweiterung (74 Klassen) schickt das Modell die Namen in
   * labels.json mit (`info`). detector.js kennt nur die neun Heuristik-Typen - dort
   * wird nachgeschlagen, wenn kein Modell geladen ist.
   */
  function signInfo(label) {
    const ausModell = model && model.labels && model.labels.info && model.labels.info[label];
    const s = ausModell || D.SIGNS[label];
    if (s) return { name: s.name, zeichen: s.zeichen, note: s.note, hex: s.hex,
                    text: s.text || '#fff' };
    return { name: label, zeichen: '', note: '', hex: '#5b6470', text: '#fff' };
  }

  /** Liste unter dem Bild – nur neu aufbauen, wenn sich die Schildtypen ändern. */
  function renderList() {
    const labels = [...new Set(current.map(t => t.label))];
    const key = labels.join();
    if (key === lastKey) return;
    lastKey = key;
    list.innerHTML = labels.length
      ? labels.map(l => { const s = signInfo(l);
          return `<li style="--c:${s.hex}"><b>${s.name}</b><span>${s.zeichen}</span><p>${s.note}</p></li>`; }).join('')
      : '<li class="leer">Noch kein Schild erkannt. Halte ein Schild ruhig und frontal ins Bild.</li>';
  }

  /** Debug: erkannte Farbflächen halbtransparent einblenden. */
  function drawMask() {
    const m = maskCv.getContext('2d'), img = m.createImageData(maskCv.width, maskCv.height);
    const pal = [null, [211, 35, 47], [20, 103, 184], [242, 194, 0]];
    for (let i = 0; i < mask.length; i++) if (mask[i]) img.data.set([...pal[mask[i]], 255], i * 4);
    m.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.globalAlpha = 0.6;
    ctx.drawImage(maskCv, 0, 0, view.width, view.height);
    ctx.globalAlpha = 1; ctx.imageSmoothingEnabled = true;
  }

  /** Rahmen + Beschriftung; Koordinaten der Erkennung auf die Anzeige hochrechnen. */
  function drawBoxes() {
    const k = boxScale;
    ctx.lineWidth = 3; ctx.textBaseline = 'top';
    ctx.font = '600 15px Bahnschrift, "DIN Alternate", system-ui, sans-serif';
    for (const t of current) {
      const s = signInfo(t.label), x = t.x * k, y = t.y * k, w = t.w * k, h = t.h * k;
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
      setSize(video.videoWidth, video.videoHeight);
      ctx.drawImage(video, 0, 0, view.width, view.height);
      if (!busy && now - lastRun >= INTERVAL_MS) {
        lastRun = now;
        busy = true;
        Promise.resolve(analyse(video, true)).catch(() => {}).finally(() => { busy = false; });
      }
      draw();
    }
    requestAnimationFrame(frame);
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      statusEl.textContent = 'Kein Kamerazugriff. Die Seite muss über HTTPS (z. B. GitHub Pages) geöffnet werden.';
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream; await video.play();
      still = null; tracker = D.createTracker(); running = true;
      $('camBtn').textContent = 'Kamera stoppen';
      requestAnimationFrame(frame);
    } catch (err) {
      statusEl.textContent = 'Kamera nicht verfügbar (' + err.name + '). Zugriff im Browser erlauben?';
    }
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; $('camBtn').textContent = 'Kamera starten';
  }

  /** Einzelnes Foto analysieren (ohne Tracker, jeder Treffer zählt sofort). */
  function showStill() {
    setSize(still.naturalWidth, still.naturalHeight);
    ctx.drawImage(still, 0, 0, view.width, view.height);
    analyse(still, false); draw();
  }

  $('camBtn').addEventListener('click', () => (running ? stopCamera() : startCamera()));
  $('file').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    stopCamera();
    const img = new Image();
    img.onload = () => { still = img; lastKey = ''; showStill(); URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(f);
  });
  ['sat', 'maskCb'].forEach(id => $(id).addEventListener('input', () => {
    $('satOut').textContent = (+$('sat').value).toFixed(2);
    if (still) showStill();
  }));
  // Größe des Netz-Eingangs: nur relevant im KI-Modus, deshalb erst dann sichtbar.
  $('size').addEventListener('change', () => { if (still) showStill(); });
  renderList();

  // Optionales KI-Modell: liegen models/labels.json + eine ONNX-Datei bereit und ist
  // onnxruntime-web geladen, rechnet das Netz; sonst bleibt die Heuristik aktiv.
  (async () => {
    if (typeof SignModel === 'undefined') return;
    const res = await SignModel.load();
    if (!res.ok) {
      statusEl.textContent = 'Heuristik-Modus (' + res.reason + ')';
      return;
    }
    modelCv.width = modelCv.height = modelInputSize('video');
    model = res;
    $('sizeRow').hidden = false;                 // Größenwahl ist nur im KI-Modus sinnvoll
    statusEl.textContent = 'KI-Modell bereit (' + res.backend + ', ' + res.size
      + ' px). Kamera starten oder ein Foto wählen.';
  })();
})();
