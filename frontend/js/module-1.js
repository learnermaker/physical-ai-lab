/**
 * module-1.js — Perception: webcam capture + MediaPipe hand landmarking
 *
 * Exports (via window.*):
 *   window.webcamManager    — getUserMedia lifecycle, canvas draw loop
 *   window.mediapipeHandler — HandLandmarker init, overlay drawing, angle computation
 *
 * Both are exposed on window so Module 4 can reuse them if needed.
 */
'use strict';

// ---------------------------------------------------------------------------
// MediaPipe CDN constants
// ---------------------------------------------------------------------------
const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/';
const MODEL_URL     = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';


// ---------------------------------------------------------------------------
// webcamManager — getUserMedia lifecycle
// ---------------------------------------------------------------------------
const webcamManager = (() => {
  const video  = document.getElementById('m1-video');
  const canvas = document.getElementById('m1-canvas');
  const ctx    = canvas.getContext('2d');

  let stream   = null;
  let rafId    = null;
  let _onFrame = null;   // callback(ctx, video, timestamp)

  async function start(onFrame) {
    if (stream) return;   // already running
    _onFrame = onFrame;
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();
    _loop();
  }

  function _loop() {
    rafId = requestAnimationFrame((ts) => {
      if (!stream) return;
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (_onFrame) _onFrame(ctx, video, ts);
      }
      _loop();
    });
  }

  function stop() {
    if (rafId)  { cancelAnimationFrame(rafId); rafId = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (ctx)    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function getCanvas() { return canvas; }

  return { start, stop, getCanvas };
})();

window.webcamManager = webcamManager;


// ---------------------------------------------------------------------------
// mediapipeHandler — HandLandmarker init, joint angle computation, overlay
// ---------------------------------------------------------------------------
const mediapipeHandler = (() => {
  let handLandmarker      = null;
  let latestResult        = null;
  let lastTs              = -1;
  let drawingUtils        = null;
  let HandLandmarkerClass = null;

  const stateVec = document.getElementById('m1-state-vector');
  const startBtn = document.getElementById('m1-start-btn');
  const stopBtn  = document.getElementById('m1-stop-btn');
  const statusEl = document.getElementById('m1-status');

  // Angle at vertex V between rays V→A and V→B.
  // atan2(|u×v|, u·v) handles the full [0°, 180°] range without acos issues.
  function computeAngle(ax, ay, vx, vy, bx, by) {
    const ux = ax - vx, uy = ay - vy;
    const wx = bx - vx, wy = by - vy;
    const cross = ux * wy - uy * wx;
    const dot   = ux * wx + uy * wy;
    return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
  }

  async function _init() {
    statusEl.textContent = 'Loading model\u2026';
    const { HandLandmarker, FilesetResolver, DrawingUtils } = await import(
      MEDIAPIPE_CDN + 'vision_bundle.mjs'
    );
    HandLandmarkerClass = HandLandmarker;
    const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN + 'wasm');
    handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions:    { modelAssetPath: MODEL_URL },
      runningMode:    'LIVE_STREAM',
      numHands:       2,
      resultCallback: (result, _image, ts) => {
        latestResult = result;
        _processResult(result);
      }
    });
    const canvas = webcamManager.getCanvas();
    drawingUtils = new DrawingUtils(canvas.getContext('2d'));
    statusEl.textContent = '';
  }

  function _processResult(result) {
    const canvas = webcamManager.getCanvas();
    const ctx    = canvas.getContext('2d');

    if (!result.landmarks || result.landmarks.length === 0) {
      stateVec.textContent = '\u2014 no hand detected \u2014';
      return;
    }

    for (const landmarks of result.landmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        HandLandmarkerClass.HAND_CONNECTIONS,
        { color: '#38bdf8', lineWidth: 2 }
      );
      drawingUtils.drawLandmarks(landmarks, { color: '#5b6af0', radius: 3 });
    }

    const lm = result.landmarks[0];
    const tThumb  = computeAngle(lm[0].x, lm[0].y, lm[1].x, lm[1].y, lm[2].x, lm[2].y);
    const tIndex  = computeAngle(lm[0].x, lm[0].y, lm[5].x, lm[5].y, lm[6].x, lm[6].y);
    const tMiddle = computeAngle(lm[0].x, lm[0].y, lm[9].x, lm[9].y, lm[10].x, lm[10].y);
    stateVec.textContent =
      `[${tThumb.toFixed(1)}, ${tIndex.toFixed(1)}, ${tMiddle.toFixed(1)}]`;
  }

  function onFrame(_ctx, video, ts) {
    if (!handLandmarker) return;
    if (ts <= lastTs) return;
    lastTs = ts;
    handLandmarker.detectForVideo(video, ts);
  }

  async function start() {
    startBtn.disabled = true;
    try {
      await _init();
      await webcamManager.start(onFrame);
      stopBtn.disabled = false;
    } catch (err) {
      // Give actionable guidance based on the error type
      let msg = '';
      const m = err.message || '';
      if (m.includes('fetch') || m.includes('Failed to load') || m.includes('import')) {
        msg = 'MediaPipe failed to load from CDN — check your internet connection and reload the page.';
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied — allow camera access in your browser settings and click Start Camera again.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera found — connect a webcam or check Windows camera privacy settings (Settings \u2192 Privacy \u2192 Camera).';
      } else {
        msg = 'Error: ' + m;
      }
      stateVec.textContent = msg;
      statusEl.textContent = '';
      startBtn.disabled = false;
    }
  }

  function stop() {
    webcamManager.stop();
    stateVec.textContent = '\u2014';
    startBtn.disabled    = false;
    stopBtn.disabled     = true;
    lastTs = -1;
  }

  function getLatestResult() { return latestResult; }

  return { start, stop, getLatestResult };
})();

window.mediapipeHandler = mediapipeHandler;


// ---------------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------------
document.getElementById('m1-start-btn').addEventListener('click', () => mediapipeHandler.start());
document.getElementById('m1-stop-btn').addEventListener('click',  () => mediapipeHandler.stop());
