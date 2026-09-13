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
  let _demoMode     = false;  // true when demo video running (VIDEO mode, synchronous)

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

      // 3. Submit for detection
      if (_demoMode) {
        // VIDEO mode: detectForVideo returns synchronously — process result immediately
        const result = landmarker.detectForVideo(video, performance.now());
        if (result) {
          const hands = result.hand_landmarks ?? result.landmarks;
          latestHands = (hands && hands.length > 0) ? hands : null;
          if (latestHands) {
            const lm8 = latestHands[0][8];
            const ax  = lm8.x * 2 - 1;
            const ay  = lm8.y * 2 - 1;
            const now = performance.now();
            if (now - lastActionTs > ACTION_INTERVAL_MS) {
              lastActionTs = now;
              sendAction(ax, ay);
            }
            status.textContent = `Action: [${ax.toFixed(2)}, ${ay.toFixed(2)}]`;
          } else {
            status.textContent = 'Demo video running \u2014 no hand detected in frame.';
          }
        }
      } else {
        // LIVE_STREAM mode: fire-and-forget, result arrives via resultCallback
        landmarker.detectForVideo(video, performance.now());
      }
    }

    rafId = requestAnimationFrame(detect);
  }

  // --- Start / Stop --------------------------------------------------------

  // Shared start logic — videoSource is either a MediaStream or 'demo'
  async function doStart(videoSource) {
    startBtn.disabled = true;
    const demoBtn = document.getElementById('m4-demo-btn');
    if (demoBtn) demoBtn.disabled = true;

    try {
      if (videoSource === 'demo') {
        // Use looping hand demo video instead of webcam
        stream = 'demo';
        video.srcObject = null;
        video.src = '/assets/fallback_hand_demo.mp4?v=2';   // ?v=2 busts any old cached version
        video.loop = true;
        video.muted = true;
        await video.play();
        status.textContent = 'Loading hand-tracking model\u2026 (demo video)';
      } else {
        stream = videoSource;
        video.srcObject = stream;
        await video.play();
        status.textContent = 'Loading hand-tracking model\u2026';
      }

      if (!landmarker) {
        await initLandmarker();
      }

      // Switch to VIDEO mode for demo path — detectForVideo returns synchronously,
      // which works reliably with a <video src=...> file (no async callback needed).
      if (videoSource === 'demo') {
        await landmarker.setOptions({ runningMode: 'VIDEO' });
        _demoMode = true;
      } else {
        _demoMode = false;
      }

      running = true;
      stopBtn.disabled = false;
      openSimStream();
      rafId = requestAnimationFrame(detect);

      status.textContent = videoSource === 'demo'
        ? 'Demo video running \u2014 watch the arm follow the hand in the video.'
        : 'Move your index finger to control the simulation.';

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
      if (demoBtn) demoBtn.disabled = false;
    }
  }

  startBtn.addEventListener('click', async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      await doStart(mediaStream);
    } catch (err) {
      startBtn.disabled = false;
      const m = err.message || '';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        status.textContent = 'Camera permission denied \u2014 allow camera in browser settings, then click Start.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        status.textContent = 'No camera found \u2014 use "Use demo video" to try without a webcam.';
      } else {
        status.textContent = 'Error: ' + m;
      }
    }
  });

  document.getElementById('m4-demo-btn')?.addEventListener('click', () => {
    doStart('demo');
  });

  stopBtn.addEventListener('click', async () => {
    running = false;

    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }

    try {
      await fetch('/api/simulation/stop', { method: 'POST', signal: AbortSignal.timeout(2000) });
    } catch (_) {}

    if (simEs) { simEs.close(); simEs = null; }
    simImg.src = '';

    // Clean up video source — real stream or demo video
    if (stream === 'demo') {
      video.pause();
      video.src = '';
    } else if (stream) {
      stream.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    stream = null;

    if (landmarker) {
      try { landmarker.close(); } catch (_) {}
      landmarker = null;
    }

    latestHands  = null;
    lastActionTs = 0;
    _demoMode    = false;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    status.textContent = 'Stopped.';
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    const demoBtn = document.getElementById('m4-demo-btn');
    if (demoBtn) demoBtn.disabled = false;
  });
})();
