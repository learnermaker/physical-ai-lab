/**
 * module-1.js — Perception: webcam capture + MediaPipe hand landmarking
 *
 * Exports (via window.*):
 *   window.webcamManager    — getUserMedia lifecycle, canvas draw loop
 *   window.mediapipeHandler — HandLandmarker init, overlay drawing, angle computation
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
  let _onFrame = null;

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

  // startDemo: use the looping hand demo video instead of getUserMedia.
  // MediaPipe reads from the same <video> element — no other code changes needed.
  async function startDemo(onFrame) {
    if (stream) return;
    _onFrame = onFrame;
    // Sentinel: use the string 'demo' so stop() can tell it apart from a real stream
    stream = 'demo';
    video.srcObject = null;
    video.src = '/assets/fallback_hand_demo.mp4?v=3';   // ?v=3 busts any browser cache
    video.loop = true;
    video.muted = true;
    video.playbackRate = 1.0;
    await video.play();
    _loop();
  }

  function _loop() {
    rafId = requestAnimationFrame((ts) => {
      if (!stream) return;
      // HAVE_CURRENT_DATA + videoWidth > 0 ensures the browser is actually
      // decoding frames. display:none can suppress decoding on some engines —
      // we use opacity:0 in CSS instead to keep the video in the render pipeline.
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (_onFrame) _onFrame(ctx, video, ts);
      }
      _loop();
    });
  }

  function stop() {
    if (rafId)  { cancelAnimationFrame(rafId); rafId = null; }
    if (stream === 'demo') {
      // Demo video: pause and reset the src
      video.pause();
      video.src = '';
      video.srcObject = null;
    } else if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    stream = null;
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function getCanvas() { return canvas; }

  return { start, startDemo, stop, getCanvas };
})();

window.webcamManager = webcamManager;


// ---------------------------------------------------------------------------
// mediapipeHandler — HandLandmarker init, joint angle computation, overlay
// ---------------------------------------------------------------------------
const mediapipeHandler = (() => {
  let handLandmarker      = null;   // null until first _init(); null again after stop()
  let latestResult        = null;
  let lastTs              = -1;
  let drawingUtils        = null;
  let HandLandmarkerClass = null;
  let _demoMode           = false;  // true when running demo video (VIDEO mode vs LIVE_STREAM)

  const stateVec = document.getElementById('m1-state-vector');
  const startBtn = document.getElementById('m1-start-btn');
  const stopBtn  = document.getElementById('m1-stop-btn');
  const statusEl = document.getElementById('m1-status');

  function computeAngle(ax, ay, vx, vy, bx, by) {
    const ux = ax - vx, uy = ay - vy;
    const wx = bx - vx, wy = by - vy;
    const cross = ux * wy - uy * wx;
    const dot   = ux * wx + uy * wy;
    return (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
  }

  async function _init() {
    if (handLandmarker) return;   // already initialised — skip reload

    statusEl.textContent = 'Loading model\u2026';
    startBtn.setAttribute('aria-busy', 'true');
    try {
      const { HandLandmarker, FilesetResolver, DrawingUtils } = await import(
        MEDIAPIPE_CDN + 'vision_bundle.mjs'
      );
      HandLandmarkerClass = HandLandmarker;
      const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN + 'wasm');
      handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
        baseOptions:    { modelAssetPath: MODEL_URL },
        runningMode:    'LIVE_STREAM',
        numHands:       2,
        resultCallback: (result) => {
          _processResult(result);
        }
      });
      const canvas = webcamManager.getCanvas();
      drawingUtils = new DrawingUtils(canvas.getContext('2d'));
      statusEl.textContent = '';
    } finally {
      startBtn.removeAttribute('aria-busy');
    }
  }

  function _processResult(result) {
    // Store result — drawing happens in onFrame() so landmarks are always
    // composited on top of the freshly drawn video frame in the same rAF tick.
    latestResult = result;

    if (!result.landmarks || result.landmarks.length === 0) {
      stateVec.textContent = '\u2014 no hand detected \u2014';
      return;
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

    // Draw landmarks from the PREVIOUS result on top of the video frame drawn
    // by webcamManager just before calling this callback — same rAF tick, no wipe.
    if (latestResult && latestResult.landmarks && latestResult.landmarks.length > 0) {
      for (const landmarks of latestResult.landmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          HandLandmarkerClass.HAND_CONNECTIONS,
          { color: '#38bdf8', lineWidth: 2 }
        );
        drawingUtils.drawLandmarks(landmarks, { color: '#5b6af0', radius: 3 });
      }
    }

    if (ts <= lastTs) return;
    lastTs = ts;

    if (_demoMode) {
      // VIDEO mode: detectForVideo returns results synchronously
      const result = handLandmarker.detectForVideo(video, ts);
      if (result) _processResult(result);
    } else {
      // LIVE_STREAM mode: fire-and-forget, result arrives via resultCallback
      handLandmarker.detectForVideo(video, ts);
    }
  }

  async function start() {
    startBtn.disabled = true;
    try {
      await _init();                      // no-op if already loaded
      await webcamManager.start(onFrame);
      stopBtn.disabled = false;
    } catch (err) {
      let msg = '';
      const m = err.message || '';
      if (m.includes('fetch') || m.includes('Failed to load') || m.includes('import')) {
        msg = 'MediaPipe failed to load from CDN \u2014 check your internet connection and reload.';
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = 'Camera permission denied \u2014 allow camera access in browser settings, then click Start again.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = 'No camera found \u2014 connect a webcam or check system camera permissions.';
        // Linux: may need  sudo usermod -aG video $USER  then log out/in
        // Windows: check Settings \u2192 Privacy \u2192 Camera
      } else {
        msg = 'Error: ' + m;
      }
      stateVec.textContent = msg;
      statusEl.textContent = '';
      startBtn.disabled = false;
    }
  }

  // startDemo: load MediaPipe the same way, but feed the looping demo video instead
  async function startDemo() {
    startBtn.disabled = true;
    const demoBtn = document.getElementById('m1-demo-btn');
    if (demoBtn) demoBtn.disabled = true;
    statusEl.textContent = 'Loading model\u2026';
    try {
      await _init();
      // Switch to VIDEO mode for the demo file — detectForVideo returns synchronously,
      // which works reliably with a <video src=...> element (no async callback needed).
      await handLandmarker.setOptions({ runningMode: 'VIDEO' });
      _demoMode = true;
      await webcamManager.startDemo(onFrame);
      stopBtn.disabled = false;
      statusEl.textContent = 'Demo video running — no real webcam needed.';
    } catch (err) {
      stateVec.textContent = 'Demo video failed to load: ' + (err.message || err);
      statusEl.textContent = '';
      startBtn.disabled = false;
      if (demoBtn) demoBtn.disabled = false;
    }
  }

  function stop() {
    // Stop the camera + rAF loop
    webcamManager.stop();

    // Close the MediaPipe landmarker and release WASM/WebGL resources
    if (handLandmarker) {
      try { handLandmarker.close(); } catch (_) {}
      handLandmarker      = null;
      drawingUtils        = null;
      HandLandmarkerClass = null;
    }

    // Clear stale state so re-start is clean
    latestResult = null;
    lastTs       = -1;
    _demoMode    = false;

    stateVec.textContent = '\u2014';
    statusEl.textContent = '';
    startBtn.disabled    = false;
    stopBtn.disabled     = true;
    const demoBtn = document.getElementById('m1-demo-btn');
    if (demoBtn) demoBtn.disabled = false;
  }

  function getLatestResult() { return latestResult; }

  return { start, startDemo, stop, getLatestResult };
})();

window.mediapipeHandler = mediapipeHandler;


// ---------------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------------
document.getElementById('m1-start-btn').addEventListener('click', () => mediapipeHandler.start());
document.getElementById('m1-stop-btn').addEventListener('click',  () => mediapipeHandler.stop());
document.getElementById('m1-demo-btn')?.addEventListener('click', () => mediapipeHandler.startDemo());
