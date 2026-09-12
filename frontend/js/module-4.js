/**
 * module-4.js — Perception to Action
 *
 * Uses MediaPipe HandLandmarker (Tasks API) from CDN to track the index
 * fingertip and map its normalised [0,1] position to Reacher joint torques
 * streamed via SSE from the server.
 */
'use strict';

(() => {
  const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/';
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

  // Reconnect the SSE stream only when the finger has moved enough AND
  // enough time has elapsed since the last reconnect.
  const RECONNECT_DELTA       = 0.03;
  const RECONNECT_INTERVAL_MS = 150;

  let landmarker    = null;
  let stream        = null;
  let simEs         = null;
  let running       = false;
  let lastLm8       = { x: 0, y: 0 };
  let lastReconnect = 0;

  const video    = document.getElementById('m4-video');
  const canvas   = document.getElementById('m4-canvas');
  const simImg   = document.getElementById('m4-sim');
  const ctx      = canvas.getContext('2d');
  const startBtn = document.getElementById('m4-start-btn');
  const stopBtn  = document.getElementById('m4-stop-btn');
  const status   = document.getElementById('m4-status');

  function openSimStream(actionX, actionY) {
    if (simEs) simEs.close();
    simEs = new EventSource(
      `/api/simulation/stream?action=${actionX.toFixed(3)},${actionY.toFixed(3)}`
    );
    simEs.onmessage = (e) => { simImg.src = 'data:image/jpeg;base64,' + e.data; };
    simEs.onerror   = () => { status.textContent = 'Sim stream error \u2014 is the server running?'; };
  }

  async function initLandmarker() {
    const { HandLandmarker, FilesetResolver } =
      await import(MEDIAPIPE_CDN + 'vision_bundle.mjs');
    const fs = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN + 'wasm');
    landmarker = await HandLandmarker.createFromOptions(fs, {
      baseOptions:    { modelAssetPath: MODEL_URL },
      runningMode:    'LIVE_STREAM',
      numHands:       1,
      resultCallback: (result) => {
        const hands = result.hand_landmarks ?? result.landmarks;
        if (!hands || hands.length === 0) return;

        const lm8 = hands[0][8];   // index fingertip
        const ax  = lm8.x * 2 - 1;
        const ay  = lm8.y * 2 - 1;
        const now = performance.now();
        const dx  = Math.abs(lm8.x - lastLm8.x);
        const dy  = Math.abs(lm8.y - lastLm8.y);

        if (
          (dx > RECONNECT_DELTA || dy > RECONNECT_DELTA) &&
          (now - lastReconnect) > RECONNECT_INTERVAL_MS
        ) {
          lastLm8 = { x: lm8.x, y: lm8.y };
          lastReconnect = now;
          openSimStream(ax, ay);
        }

        status.textContent = `Action: [${ax.toFixed(2)}, ${ay.toFixed(2)}]`;
      }
    });
  }

  function detect() {
    if (!running || !landmarker) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    landmarker.detectForVideo(video, performance.now());
    requestAnimationFrame(detect);
  }

  startBtn.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      video.srcObject = stream;
      await video.play();

      if (!landmarker) {
        status.textContent = 'Loading hand-tracking model\u2026';
        await initLandmarker();
      }

      running = true;
      startBtn.disabled = true;
      stopBtn.disabled  = false;
      openSimStream(0, 0);
      detect();
      status.textContent = 'Move your index finger to control the simulation.';
    } catch (err) {
      status.textContent = 'Error: ' + err.message;
    }
  });

  stopBtn.addEventListener('click', () => {
    running = false;
    if (stream)  { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (simEs)   { simEs.close(); simEs = null; }
    simImg.src = '';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    status.textContent = 'Stopped.';
    startBtn.disabled = false;
    stopBtn.disabled  = true;
  });
})();
