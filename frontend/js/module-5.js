/**
 * module-5.js — Foundation Models (Gemini)
 *
 * Camera capture via getUserMedia, frame encoding,
 * and calls to /api/gemini/describe + /api/gemini/action.
 * Reads window.hubConfig.getApiKey() for the client-side key.
 *
 * Fallback transparency:
 *   Every API response now includes a _source field ("gemini" | "cache").
 *   The cacheNotice is shown whenever _source === "cache", regardless of
 *   whether the user has a key saved — the server may return cache even
 *   with a key if the key is invalid or the API is unavailable.
 */
'use strict';

(() => {
  const camBtn      = document.getElementById('m5-cam-btn');
  const describeBtn = document.getElementById('m5-describe-btn');
  const actionBtn   = document.getElementById('m5-action-btn');
  const video       = document.getElementById('m5-video');
  const canvas      = document.getElementById('m5-canvas');
  const ctx         = canvas.getContext('2d');
  const resultEl    = document.getElementById('m5-result');
  const spinner     = document.getElementById('m5-spinner');
  const cacheNotice = document.getElementById('m5-cache-notice');
  let stream = null;

  function getFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
  }

  function showResult(data, isText) {
    spinner.style.display = 'none';

    // Show the content (strip internal _source/_reason keys from display)
    const displayData = Object.fromEntries(
      Object.entries(data).filter(([k]) => !k.startsWith('_'))
    );
    resultEl.textContent = isText
      ? (data.text || JSON.stringify(displayData))
      : JSON.stringify(displayData, null, 2);

    // Show fallback notice based on server's _source field, not on key presence.
    // This correctly catches: no key, invalid key, API down, rate limit, etc.
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

  async function callApi(endpoint) {
    // If camera is not running, send empty frame — server will use cache fallback.
    const frame_b64 = stream ? getFrame() : '';
    const api_key   = window.hubConfig?.getApiKey() || null;
    spinner.style.display = 'block';
    resultEl.textContent  = '';
    cacheNotice.style.display = 'none';
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_b64, api_key }),
    });
    return await res.json();
  }

  camBtn.addEventListener('click', async () => {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      camBtn.textContent    = 'Start Camera';
      describeBtn.disabled  = true;
      actionBtn.disabled    = true;
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
      video.srcObject = stream;
      await video.play();
      camBtn.textContent   = 'Stop Camera';
      describeBtn.disabled = false;
      actionBtn.disabled   = false;
      const tick = () => {
        if (stream) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          requestAnimationFrame(tick);
        }
      };
      tick();
    } catch (err) {
      // Camera not available — still allow API calls with empty frame (cache mode)
      const isPermission = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
      const isNotFound   = err.name === 'NotFoundError'   || err.name === 'DevicesNotFoundError';
      let guidance = '';
      if (isPermission) {
        guidance = ' — allow camera in browser settings, or use the buttons below without a frame (cache mode).';
      } else if (isNotFound) {
        guidance = ' — no camera detected. You can still use Describe/Action buttons; responses will come from the pre-cached dataset.';
      } else {
        guidance = ` — ${err.message}. You can still use the API buttons in cache mode.`;
      }
      resultEl.textContent = 'Camera unavailable' + guidance;
      // Enable buttons even without camera — server handles empty frame gracefully
      describeBtn.disabled = false;
      actionBtn.disabled   = false;
    }
  });

  describeBtn.addEventListener('click', async () => {
    try {
      showResult(await callApi('/api/gemini/describe'), true);
    } catch (err) {
      spinner.style.display = 'none';
      resultEl.textContent = 'Error: ' + err.message;
    }
  });

  actionBtn.addEventListener('click', async () => {
    try {
      showResult(await callApi('/api/gemini/action'), false);
    } catch (err) {
      spinner.style.display = 'none';
      resultEl.textContent = 'Error: ' + err.message;
    }
  });
})();
