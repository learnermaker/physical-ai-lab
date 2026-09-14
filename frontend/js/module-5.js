/**
 * module-5.js — Foundation Models (Gemini)
 *
 * Camera capture via getUserMedia, frame encoding,
 * and calls to /api/gemini/describe + /api/gemini/action.
 * Reads window.hubConfig.getApiKey() for the client-side key.
 *
 * No-webcam path: three sample images are shown as clickable thumbnails.
 * Selecting one draws it onto the canvas so the full VLA pipeline
 * (Frame → Encode → Prompt → Parse → Act) works without a camera.
 *
 * Fallback transparency:
 *   Every API response includes a _source field ("gemini" | "cache").
 *   The cacheNotice is shown whenever _source === "cache", regardless of
 *   whether the user has a key — the server may return cache even
 *   with a key if the key is invalid or the API is unavailable.
 */
'use strict';

(() => {
  const camBtn       = document.getElementById('m5-cam-btn');
  const describeBtn  = document.getElementById('m5-describe-btn');
  const actionBtn    = document.getElementById('m5-action-btn');
  const video        = document.getElementById('m5-video');
  const canvas       = document.getElementById('m5-canvas');
  const ctx          = canvas.getContext('2d');
  const resultEl     = document.getElementById('m5-result');
  const resultCard   = document.getElementById('m5-result-card');
  const cacheNotice  = document.getElementById('m5-cache-notice');
  const samplesDiv   = document.getElementById('m5-samples');
  const pipelineEl   = document.getElementById('m5-pipeline-step');

  let stream         = null;   // active MediaStream, null when no camera
  let sampleLoaded   = false;  // true when a sample image is on the canvas

  // ── Pipeline step indicator ──────────────────────────────────────────────
  const STEPS = {
    idle:    '',
    capture: '① Capture — frame drawn to canvas',
    encode:  '② Encode — converting to base64 JPEG…',
    send:    '③ Send — POST to /api/gemini/…',
    parse:   '④ Parse — reading JSON response…',
    done:    '⑤ Act — response ready',
  };

  function setStep(key) {
    const el = document.getElementById('m5-pipeline-step');
    if (!el) return;
    if (!key || key === 'idle') {
      el.style.display = 'none';
      el.textContent = '';
    } else {
      el.style.display = '';
      el.textContent = STEPS[key] || key;
    }
  }

  // ── Sample image thumbnails ───────────────────────────────────────────────
  function hideSamples() {
    if (samplesDiv) samplesDiv.style.display = 'none';
  }

  function showSamples() {
    if (samplesDiv) samplesDiv.style.display = '';
  }

  // Highlight the selected thumbnail with an accent border
  function selectThumbnail(btn) {
    document.querySelectorAll('.m5-sample-btn').forEach(b => {
      b.style.borderColor = 'transparent';
    });
    if (btn) btn.style.borderColor = 'var(--accent, #c2410c)';
  }

  // Wire each thumbnail to draw its image on the canvas
  document.querySelectorAll('.m5-sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const src = btn.dataset.src;
      if (!src) return;
      const img = new Image();
      img.crossOrigin = 'anonymous'; // required for getImageData on same-origin images in some browsers
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        sampleLoaded = true;
        selectThumbnail(btn);
        describeBtn.disabled = false;
        actionBtn.disabled   = false;
        setStep('capture');
        // Show which image was selected in the pipeline indicator
        const label = btn.getAttribute('title') || 'sample image';
        pipelineEl.textContent = `① Capture — "${label}" loaded onto canvas`;
      };
      img.onerror = () => {
        resultEl.textContent = `Could not load sample image: ${src}`;
      };
      img.src = src;
    });
  });

  // ── Frame capture ─────────────────────────────────────────────────────────
  function getFrame() {
    // Canvas already has either a live frame (via rAF loop) or a sample image.
    // Encode as JPEG base64 and strip the data-URL prefix.
    setStep('encode');
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
  }

  // ── Result display ────────────────────────────────────────────────────────
  function showResult(data, isText) {
    resultCard.removeAttribute('aria-busy');
    setStep('done');

    const displayData = Object.fromEntries(
      Object.entries(data).filter(([k]) => !k.startsWith('_'))
    );
    resultEl.textContent = isText
      ? (data.text || JSON.stringify(displayData))
      : JSON.stringify(displayData, null, 2);

    const isCache = data._source === 'cache';
    if (isCache) {
      const reason = data._reason || 'unknown';
      const reasonText = reason === 'no_api_key'
        ? 'No API key set — using pre-cached responses.'
        : `Gemini API unavailable — using pre-cached response. (${reason.substring(0, 80)})`;
      cacheNotice.textContent = reasonText;
      cacheNotice.style.display = 'block';
    } else {
      cacheNotice.style.display = 'none';
    }
  }

  // ── API call ──────────────────────────────────────────────────────────────
  async function callApi(endpoint) {
    const frame_b64 = (stream || sampleLoaded) ? getFrame() : '';
    const api_key   = window.hubConfig?.getApiKey() || null;

    resultCard.setAttribute('aria-busy', 'true');
    resultEl.textContent      = '';
    cacheNotice.style.display = 'none';

    setStep('send');
    const res = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame_b64, api_key }),
    });
    setStep('parse');
    return await res.json();
  }

  // ── Camera button ─────────────────────────────────────────────────────────
  camBtn.addEventListener('click', async () => {
    // Stop path
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      camBtn.textContent   = 'Start Camera';
      describeBtn.disabled = true;
      actionBtn.disabled   = true;
      sampleLoaded = false;
      selectThumbnail(null);
      showSamples();
      setStep('idle');
      return;
    }

    // Start path
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      video.srcObject = stream;
      await video.play();
      camBtn.textContent   = 'Stop Camera';
      describeBtn.disabled = false;
      actionBtn.disabled   = false;
      sampleLoaded = false;
      hideSamples();
      selectThumbnail(null);

      // Live rAF loop keeps the canvas current for every API call
      const tick = () => {
        if (stream) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          setStep('capture');
          requestAnimationFrame(tick);
        }
      };
      tick();

    } catch (err) {
      // Camera unavailable — still allow API calls in sample/cache mode
      const isPermission = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
      const isNotFound   = err.name === 'NotFoundError'   || err.name === 'DevicesNotFoundError';
      let guidance = '';
      if (isPermission) {
        guidance = ' Allow camera in browser settings, or pick a sample image below.';
      } else if (isNotFound) {
        guidance = ' No camera detected — pick a sample image below to try the full pipeline.';
      } else {
        guidance = ` ${err.message}. Pick a sample image to continue.`;
      }
      resultEl.textContent = 'Camera unavailable.' + guidance;

      showSamples();
      // Buttons stay disabled until a sample is chosen (to avoid blank-frame confusion)
    }
  });

  // ── Describe / Action buttons ─────────────────────────────────────────────
  describeBtn.addEventListener('click', async () => {
    try {
      showResult(await callApi('/api/gemini/describe'), true);
    } catch (err) {
      resultCard.removeAttribute('aria-busy');
      setStep('idle');
      resultEl.textContent = 'Error: ' + err.message;
    }
  });

  actionBtn.addEventListener('click', async () => {
    try {
      showResult(await callApi('/api/gemini/action'), false);
    } catch (err) {
      resultCard.removeAttribute('aria-busy');
      setStep('idle');
      resultEl.textContent = 'Error: ' + err.message;
    }
  });

  // ── Init — buttons start disabled; enabled on camera start or sample pick ──
  // (Describe/Action buttons already start disabled via HTML disabled attribute)
  // The sample thumbnails are always visible so participants know the option
  // exists immediately without needing to attempt the camera first.
})();
