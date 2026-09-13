/**
 * module-2.js — Reacher-v5 simulation stream
 *
 * Manages the SSE stream to /api/simulation/stream, torque sliders,
 * auto-retry on error, and the fullscreen expand overlay.
 */
'use strict';

(() => {
  const startBtn    = document.getElementById('m2-start-btn');
  const stopBtn     = document.getElementById('m2-stop-btn');
  const expandBtn   = document.getElementById('m2-expand-btn');
  const frameImg    = document.getElementById('m2-frame');
  const status      = document.getElementById('m2-status');
  const sliders     = document.getElementById('m2-sliders');
  const legend      = document.getElementById('m2-legend');
  const t0Input     = document.getElementById('m2-torque0');
  const t1Input     = document.getElementById('m2-torque1');
  const t0Val       = document.getElementById('m2-torque0-val');
  const t1Val       = document.getElementById('m2-torque1-val');
  const overlay     = document.getElementById('m2-overlay');
  const overlayImg  = document.getElementById('m2-overlay-img');
  const overlayClose = document.getElementById('m2-overlay-close');

  let es = null;
  let reconnectTimer = null;
  let isRunning = false;

  function currentAction() {
    return `${parseFloat(t0Input.value).toFixed(3)},${parseFloat(t1Input.value).toFixed(3)}`;
  }

  function openStream() {
    if (es) { es.close(); es = null; }
    clearTimeout(reconnectTimer);
    status.textContent = 'Connecting\u2026';
    es = new EventSource(`/api/simulation/stream?action=${currentAction()}`);

    es.onmessage = (e) => {
      // Frames are tagged "gpu|<b64>" or "software|<b64>" by the server.
      const pipeIdx = e.data.indexOf('|');
      const renderMode = pipeIdx > -1 ? e.data.substring(0, pipeIdx) : 'gpu';
      const b64 = pipeIdx > -1 ? e.data.substring(pipeIdx + 1) : e.data;

      const src = 'data:image/jpeg;base64,' + b64;
      frameImg.src = src;
      if (overlay.style.display === 'flex') overlayImg.src = src;

      // Detect the "Simulation busy" sentinel frame — stop auto-retrying,
      // show a plain stop state so the user clicks Start themselves after
      // stopping any other active stream.
      // The busy frame is small (~4k b64); real status frames are 15k+.
      // We check frame size before updating status so there's no race.
      if (b64.length < 8000) {
        // Small frame = busy sentinel — stop the retry loop immediately
        isRunning = false;
        startBtn.disabled  = false;
        stopBtn.disabled   = true;
        expandBtn.disabled = true;
        sliders.style.display = 'none';
        legend.style.display  = 'none';
        frameImg.src = src;
        status.textContent = 'Simulation busy \u2014 another stream is running. Stop any active simulation, then click Start again.';
        if (es) { es.close(); es = null; }
        return;
      }

      const renderBadge = renderMode === 'software'
        ? '  \u00b7  \u26a0\ufe0f software render (no GPU)'
        : '';
      status.textContent =
        `Streaming at ~20 fps  |  \u03c4\u2081=${t0Input.value}  \u03c4\u2082=${t1Input.value}${renderBadge}`;
    };

    es.onerror = () => {
      status.textContent = 'Stream error \u2014 is the server running?';
      if (es) { es.close(); es = null; }
      if (isRunning) {
        reconnectTimer = setTimeout(openStream, 2000);
      } else {
        startBtn.disabled  = false;
        stopBtn.disabled   = true;
        expandBtn.disabled = true;
        sliders.style.display = 'none';
        legend.style.display  = 'none';
      }
    };
  }

  startBtn.addEventListener('click', () => {
    isRunning = true;
    startBtn.disabled  = false;
    stopBtn.disabled   = false;
    expandBtn.disabled = false;
    sliders.style.display = 'flex';
    legend.style.display  = 'flex';
    openStream();
  });

  stopBtn.addEventListener('click', () => {
    isRunning = false;
    clearTimeout(reconnectTimer);
    if (es) { es.close(); es = null; }
    frameImg.src = '';
    if (overlay.style.display === 'flex') {
      overlay.style.display = 'none';
      overlayImg.src = '';
    }
    status.textContent = 'Simulation stopped.';
    startBtn.disabled  = false;
    stopBtn.disabled   = true;
    expandBtn.disabled = true;
    sliders.style.display = 'none';
    legend.style.display  = 'none';
    t0Input.value = '0'; t0Val.textContent = '0.00';
    t1Input.value = '0'; t1Val.textContent = '0.00';
  });

  // Sliders — debounced stream reconnect
  let debounce = null;
  function onSliderChange() {
    t0Val.textContent = parseFloat(t0Input.value).toFixed(2);
    t1Val.textContent = parseFloat(t1Input.value).toFixed(2);
    if (!isRunning) return;
    clearTimeout(debounce);
    debounce = setTimeout(openStream, 120);
  }
  t0Input.addEventListener('input', onSliderChange);
  t1Input.addEventListener('input', onSliderChange);

  // Expand overlay
  function showOverlay() {
    if (frameImg.src && frameImg.src !== window.location.href) {
      overlayImg.src = frameImg.src;
      overlay.style.display = 'flex';
    }
  }
  function closeOverlay() {
    overlay.style.display = 'none';
    overlayImg.src = '';
  }

  expandBtn.addEventListener('click', showOverlay);
  frameImg.addEventListener('click', () => { if (isRunning) showOverlay(); });
  overlayClose.addEventListener('click', closeOverlay);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeOverlay(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay.style.display === 'flex') closeOverlay();
  });
})();
