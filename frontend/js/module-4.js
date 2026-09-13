/**
 * module-4.js — Perception to Action
 *
 * Uses MediaPipe HandLandmarker (Tasks API) from CDN to track the index
 * fingertip and map its normalised [0,1] position to Reacher joint torques
 * streamed via SSE from the server.
 *
 * Action update design:
 *   - ONE persistent EventSource for the lifetime of the session.
 *   - Finger movement updates are sent via POST /api/simulation/action —
 *     no reconnection, no lock contention, no "Simulation busy" errors.
 *   - Stop calls POST /api/simulation/stop then closes the EventSource.
 */
'use strict';

(() => {
  const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/';
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

  // Throttle action POSTs — send at most once per ACTION_INTERVAL_MS
  const ACTION_INTERVAL_MS = 80;

  let landmarker    = null;
  let latestHands   = null;
  let stream        = null;
  let simEs         = null;   // one persistent EventSource
  let running       = false;
  let rafId         = null;
  let lastActionTs  = 0;      // last time we POSTed an action

  const video    = document.getElementById('m4-video');
  const canvas   = document.getElementById('m4-canvas');
  const simImg   = document.getElementById('m4-sim');
  const ctx      = canvas.getContext('2d');
  const startBtn = document.getElementById('m4-start-btn');
  const stopBtn  = document.getElementById('m4-stop-btn');
  const status   = document.getElementById('m4-status');

  // --- Simulation stream ---------------------------------------------------

  function openSimStream() {
    if (simEs) { simEs.close(); simEs = null; }
    simEs = new EventSource('/api/simulation/stream?action=0.000,0.000');
    simEs.onmessage = (e) => {
      const pipe = e.data.indexOf('|');
      const b64  = pipe > -1 ? e.data.substring(pipe + 1) : e.data;
      simImg.src = 'data:image/jpeg;base64,' + b64;
    };
    simEs.onerror = () => {
      status.textContent = 'Simulation stream error \u2014 click Stop then Start to retry.';
    };
  }

  /**
   * Send an action update to the running simulation.
   * Fire-and-forget — no need to await; errors are silently ignored
   * since the next tick will send a fresh value anyway.
   */
  function sendAction(ax, ay) {
    fetch('/api/simulation/action', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ action: [ax, ay] }),
      signal:  AbortSignal.timeout(300),  // drop stale updates, not critical path
    }).catch(() => {});  // intentional no-op — next frame will retry
  }

  // --- MediaPipe -----------------------------------------------------------

  async function initLandmarker() {
    startBtn.setAttribute('aria-busy', 'true');
    try {
      const { HandLandmarker, FilesetResolver, DrawingUtils } =
        await import(MEDIAPIPE_CDN + 'vision_bundle.mjs');
      const fs = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN + 'wasm');
      const drawingUtils = new DrawingUtils(ctx);

      landmarker = await HandLandmarker.createFromOptions(fs, {
        baseOptions:    { modelAssetPath: MODEL_URL },
        runningMode:    'LIVE_STREAM',
        numHands:       1,
        resultCallback: (result) => {
          const hands = result.hand_landmarks ?? result.landmarks;
          latestHands = (hands && hands.length > 0) ? hands : null;

          if (!latestHands) return;

          const lm8 = latestHands[0][8];   // index fingertip
          const ax  = lm8.x * 2 - 1;
          const ay  = lm8.y * 2 - 1;

          // Throttle action POSTs — no reconnect, just update the shared array
          const now = performance.now();
          if (now - lastActionTs > ACTION_INTERVAL_MS) {
            lastActionTs = now;
            sendAction(ax, ay);
          }

          status.textContent = `Action: [${ax.toFixed(2)}, ${ay.toFixed(2)}]`;
        }
      });

      landmarker._du  = drawingUtils;
      landmarker._HLC = HandLandmarker;
    } finally {
      startBtn.removeAttribute('aria-busy');
    }
  }

  function detect() {
    if (!running || !landmarker) return;

    // Only process frames the browser is actually decoding.
    // videoWidth === 0 means the video element isn't producing pixels yet.
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
      // 1. Draw video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // 2. Overlay landmarks from the PREVIOUS result — same rAF tick as video draw
      //    so the video paint never wipes them out.
      if (latestHands && landmarker._du && landmarker._HLC) {
        for (const hand of latestHands) {
          landmarker._du.drawConnectors(hand, landmarker._HLC.HAND_CONNECTIONS,
            { color: '#38bdf8', lineWidth: 2 });
          landmarker._du.drawLandmarks(hand, { color: '#ffd23c', radius: 4 });
        }
      }

      // 3. Submit for async detection (result arrives in resultCallback ~1 frame later)
      landmarker.detectForVideo(video, performance.now());
    }

    rafId = requestAnimationFrame(detect);
  }

  // --- Start / Stop --------------------------------------------------------

  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      video.srcObject = stream;
      await video.play();

      if (!landmarker) {
        status.textContent = 'Loading hand-tracking model\u2026';
        await initLandmarker();
      }

      running = true;
      stopBtn.disabled = false;

      // Open ONE persistent stream — updates flow through /api/simulation/action
      openSimStream();

      rafId = requestAnimationFrame(detect);
      status.textContent = 'Move your index finger to control the simulation.';
    } catch (err) {
      const m = err.message || '';
      if (m.includes('fetch') || m.includes('Failed to load') || m.includes('import')) {
        status.textContent = 'MediaPipe failed to load \u2014 check internet connection and reload.';
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        status.textContent = 'Camera permission denied \u2014 allow camera in browser settings, then click Start.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        status.textContent = 'No camera found \u2014 connect a webcam or check Windows privacy settings.';
      } else {
        status.textContent = 'Error: ' + m;
      }
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener('click', async () => {
    running = false;

    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }

    // Signal the server worker to stop, then close the SSE connection
    try {
      await fetch('/api/simulation/stop', { method: 'POST', signal: AbortSignal.timeout(2000) });
    } catch (_) {}

    if (simEs) { simEs.close(); simEs = null; }
    simImg.src = '';

    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;

    if (landmarker) {
      try { landmarker.close(); } catch (_) {}
      landmarker = null;
    }

    latestHands  = null;
    lastActionTs = 0;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    status.textContent = 'Stopped.';
    startBtn.disabled = false;
    stopBtn.disabled  = true;
  });
})();
