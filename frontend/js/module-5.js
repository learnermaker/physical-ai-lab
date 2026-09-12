/**
 * module-5.js — Foundation Models (Gemini)
 *
 * Camera capture via getUserMedia, frame encoding,
 * and calls to /api/gemini/describe + /api/gemini/action.
 * Reads window.hubConfig.getApiKey() for the client-side key.
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
    resultEl.textContent = isText
      ? (data.text || JSON.stringify(data))
      : JSON.stringify(data, null, 2);
    cacheNotice.style.display = window.hubConfig?.getApiKey() ? 'none' : 'block';
  }

  async function callApi(endpoint) {
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
      camBtn.textContent   = 'Start Camera';
      describeBtn.disabled = true;
      actionBtn.disabled   = true;
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
      resultEl.textContent = 'Camera error: ' + err.message;
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
