/**
 * hub-config.js — Shared hub infrastructure
 *
 * Exports (via window.*):
 *   window.hubConfig   — API key management + /api/config fetch
 *
 * Also initialises (no export needed):
 *   serverCheck        — polls /api/status, updates all service badges
 *   serverControl      — restart/stop button handlers in the Setup panel
 *   API key UI         — save / clear / test key handlers
 *   sidebar nav        — IntersectionObserver active-link tracking
 *   Module 3 chart     — PPO CartPole training chart + SSE progress
 */
'use strict';

// ---------------------------------------------------------------------------
// hubConfig — fetch /api/config, manage sessionStorage API key
// ---------------------------------------------------------------------------
const hubConfig = (() => {
  const KEY_NAME = 'gemini_api_key';

  function getApiKey() { return sessionStorage.getItem(KEY_NAME) || null; }

  function setApiKey(key) {
    const trimmed = (key || '').trim();
    trimmed ? sessionStorage.setItem(KEY_NAME, trimmed) : sessionStorage.removeItem(KEY_NAME);
  }

  function clearApiKey() { sessionStorage.removeItem(KEY_NAME); }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config', { signal: AbortSignal.timeout(4000) });
      if (!res.ok) return null;
      return await res.json();
    } catch { return null; }
  }

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status', { signal: AbortSignal.timeout(5000) });
      if (!res.ok) return null;
      return await res.json();
    } catch { return null; }
  }

  return { getApiKey, setApiKey, clearApiKey, fetchConfig, fetchStatus };
})();

window.hubConfig = hubConfig;


// ---------------------------------------------------------------------------
// serverCheck — poll /api/status every 10 s, update all service tiles + banner
// ---------------------------------------------------------------------------
const serverCheck = (() => {
  const serverBadge    = document.getElementById('server-badge');
  const geminiSvrBadge = document.getElementById('gemini-server-badge');

  // Service tile elements (Setup panel)
  const tiles = {
    server:      document.getElementById('svc-server'),
    mujoco:      document.getElementById('svc-mujoco'),
    gymnasium:   document.getElementById('svc-gymnasium'),
    sb3:         document.getElementById('svc-sb3'),
    gemini:      document.getElementById('svc-gemini'),
    mediapipe:   document.getElementById('svc-mediapipe'),
    training:    document.getElementById('svc-training'),
  };

  function setTile(id, cls, text) {
    const el = tiles[id];
    if (!el) return;
    const s = el.querySelector('.svc-status');
    if (s) {
      s.className = 'svc-status ' + cls;
      s.textContent = text;
      // Add contextual tooltip for the Gemini tile
      if (id === 'gemini' && cls === 'svc-warn') {
        s.title = 'Cache only — pre-written responses are returned. The model is NOT analysing your camera frame. Set a valid API key (below) to enable live inference.';
      } else if (id === 'gemini' && cls === 'svc-ok') {
        s.title = 'Live key configured — Gemini will analyse real camera frames when called from Module 5.';
      } else if (id === 'gemini' && cls === 'svc-error') {
        s.title = 'No key and no cache — Module 5 Gemini calls will return an error.';
      }
    }
  }

  function setBadge(el, cls, text) {
    if (!el) return;
    el.className = 'badge ' + cls;
    el.textContent = text;
  }

  function disableServerControls() {
    document.querySelectorAll('[data-requires-server]').forEach(el => { el.disabled = true; });
  }

  function enableServerControls() {
    document.querySelectorAll('[data-requires-server]').forEach(el => { el.disabled = false; });
  }

  async function check() {
    const status = await hubConfig.fetchStatus();

    if (!status) {
      document.body.classList.add('server-down');
      setBadge(serverBadge, 'error', 'Unreachable');
      setBadge(geminiSvrBadge, 'warn', 'Unknown');
      setTile('server',    'svc-error', 'Offline');
      setTile('mujoco',    'svc-idle',  '—');
      setTile('gymnasium', 'svc-idle',  '—');
      setTile('sb3',       'svc-idle',  '—');
      setTile('gemini',    'svc-idle',  '—');
      setTile('mediapipe', 'svc-idle',  '—');
      setTile('training',  'svc-idle',  '—');
      disableServerControls();
      return;
    }

    document.body.classList.remove('server-down');
    enableServerControls();
    setBadge(serverBadge, 'ok', 'Connected');

    const dep = status.deps || {};
    setTile('server',    'svc-ok',   'Running');
    setTile('mujoco',    dep['mujoco']            ? 'svc-ok'   : 'svc-error',
                         dep['mujoco']            ? 'OK'       : 'Missing');
    setTile('gymnasium', dep['gymnasium']          ? 'svc-ok'   : 'svc-error',
                         dep['gymnasium']          ? 'OK'       : 'Missing');
    setTile('sb3',       dep['stable_baselines3'] ? 'svc-ok'   : 'svc-error',
                         dep['stable_baselines3'] ? 'OK'       : 'Missing');
    setTile('mediapipe', dep['mediapipe']          ? 'svc-ok'   : 'svc-warn',
                         dep['mediapipe']          ? 'OK'       : 'Not found');

    const geminiOk = status.gemini_key_configured;
    const cacheOk  = status.gemini_cache_loaded;
    setBadge(geminiSvrBadge, geminiOk ? 'ok' : 'warn', geminiOk ? 'Configured' : 'Not configured');
    setTile('gemini',
      geminiOk  ? 'svc-ok'   : cacheOk ? 'svc-warn' : 'svc-error',
      geminiOk  ? 'Live key' : cacheOk ? 'Cache only' : 'No key/cache'
    );

    const active = status.active_training_jobs || 0;
    setTile('training',
      active > 0 ? 'svc-ok'  : 'svc-idle',
      active > 0 ? `${active} running` : 'Idle'
    );
  }

  check();
  setInterval(check, 10_000);
  return { check };
})();


// ---------------------------------------------------------------------------
// serverControl — restart/stop buttons in the Setup server-panel
// ---------------------------------------------------------------------------
(() => {
  const restartBtn = document.getElementById('server-restart-btn');
  const logEl      = document.getElementById('server-log');

  function log(msg) {
    if (!logEl) return;
    logEl.classList.add('visible');
    logEl.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  }

  if (restartBtn) {
    restartBtn.addEventListener('click', async () => {
      restartBtn.disabled = true;
      log('Requesting server status check…');
      // We can't restart the Python process from the browser, but we can
      // immediately re-poll to confirm the server is alive and refresh all tiles.
      const status = await hubConfig.fetchStatus();
      if (status) {
        log('Server is running — all status tiles refreshed.');
        serverCheck.check();
      } else {
        log('Server unreachable. Start it with: python start_hub.py');
      }
      restartBtn.disabled = false;
    });
  }
})();


// ---------------------------------------------------------------------------
// API key input — save / clear / test
// ---------------------------------------------------------------------------
(() => {
  const input    = document.getElementById('api-key-input');
  const saveBtn  = document.getElementById('save-key-btn');
  const clearBtn = document.getElementById('clear-key-btn');
  const testBtn  = document.getElementById('test-key-btn');
  const notice   = document.getElementById('key-saved-notice');
  const testOut  = document.getElementById('gemini-test-result');

  const existing = hubConfig.getApiKey();
  if (existing) { input.value = existing; notice.style.display = 'inline'; }

  saveBtn.addEventListener('click', () => {
    hubConfig.setApiKey(input.value);
    notice.style.display = 'inline';
    setTimeout(() => { notice.style.display = 'none'; }, 3000);
  });

  clearBtn.addEventListener('click', () => {
    hubConfig.clearApiKey();
    input.value = '';
    notice.style.display = 'none';
    testOut.style.display = 'none';
  });

  testBtn.addEventListener('click', async () => {
    const key = input.value.trim();
    if (!key) {
      testOut.style.display = 'block';
      testOut.style.color = 'var(--color-error)';
      testOut.textContent = 'Enter an API key first.';
      return;
    }
    testBtn.disabled = true;
    testOut.style.display = 'block';
    testOut.style.color = 'var(--color-muted)';
    testOut.textContent = 'Calling Gemini\u2026';

    try {
      const res = await fetch('/api/gemini/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_b64: '', api_key: key }),
        signal: AbortSignal.timeout(35000),
      });
      const data = await res.json();
      // Use server's _source field — reliable, unlike a fragile length heuristic.
      // "cache" means the API call failed (bad key, empty frame rejected, rate limit, etc.)
      const isCache = data._source === 'cache';
      if (isCache) {
        testOut.style.color = 'var(--color-warn)';
        testOut.textContent =
          'Cache fallback returned \u2014 the empty test frame was rejected or the key is invalid.\n\n'
          + 'Tip: For a definitive test, start the camera in Module 5, point it at a scene, and press "Describe scene".\n\n'
          + JSON.stringify(data, null, 2);
      } else {
        // _source === "gemini" — API was reached and responded to the call
        hubConfig.setApiKey(key);
        notice.style.display = 'inline';
        setTimeout(() => { notice.style.display = 'none'; }, 3000);
        testOut.style.color = 'var(--color-ok)';
        testOut.textContent = 'Gemini API responded (key accepted):\n\n' + JSON.stringify(data, null, 2);
      }
    } catch (err) {
      testOut.style.color = 'var(--color-error)';
      testOut.textContent = 'Request failed: ' + err.message;
    } finally {
      testBtn.disabled = false;
    }
  });
})();


// ---------------------------------------------------------------------------
// Sidebar active-link tracking (IntersectionObserver)
// ---------------------------------------------------------------------------
(() => {
  const links = document.querySelectorAll('#sidebar-nav a[data-target]');
  const linkMap = new Map();
  links.forEach(a => linkMap.set(a.dataset.target, a));
  let currentActive = null;

  function setActive(id) {
    if (currentActive === id) return;
    if (currentActive) linkMap.get(currentActive)?.classList.remove('active');
    linkMap.get(id)?.classList.add('active');
    currentActive = id;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter(e => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length > 0) setActive(visible[0].target.id);
    },
    { threshold: 0.15 }
  );

  document.querySelectorAll('section[id]').forEach(sec => observer.observe(sec));
  links.forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const target = document.getElementById(a.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
})();


// ---------------------------------------------------------------------------
// Challenge card accordion — open/close on header click
// ---------------------------------------------------------------------------
document.querySelectorAll('.challenge-card .challenge-header').forEach(header => {
  header.addEventListener('click', () => {
    header.closest('.challenge-card').classList.toggle('open');
  });
});


// ---------------------------------------------------------------------------
// Module 3 — RL training chart (Chart.js + SSE progress stream)
// ---------------------------------------------------------------------------
(() => {
  const trainBtn = document.getElementById('m3-train-btn');
  const statusEl = document.getElementById('m3-status');
  if (!trainBtn) return;

  const chartCtx = document.getElementById('m3-chart').getContext('2d');
  const chart = new Chart(chartCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Mean episode reward',
        data: [],
        borderColor: '#5b6af0',
        backgroundColor: 'rgba(91,106,240,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: { title: { display: true, text: 'Timestep' }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: true, text: 'Mean Reward' }, min: 0 }
      },
      plugins: { legend: { display: false } }
    }
  });

  trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    statusEl.textContent = 'Starting training\u2026';
    statusEl.style.color = 'var(--color-muted)';
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();

    try {
      const res = await fetch('/api/training/start', { method: 'POST' });
      const { job_id } = await res.json();
      if (!job_id) throw new Error('No job_id returned');

      statusEl.textContent = `Job ${job_id.substring(0, 8)}\u2026 (50k steps)`;
      const es = new EventSource(`/api/training/progress/${job_id}`);

      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.done) {
          if (data.error) {
            // Training failed — surface the error clearly
            statusEl.textContent = 'Training failed: ' + data.error;
            statusEl.style.color = 'var(--color-error)';
          } else {
            statusEl.textContent = 'Training complete!';
            statusEl.style.color = 'var(--color-ok)';
          }
          es.close();
          trainBtn.disabled = false;
          serverCheck.check();   // refresh training tile
          return;
        }
        if (data.timestep != null && data.mean_reward != null) {
          chart.data.labels.push(data.timestep.toString());
          chart.data.datasets[0].data.push(data.mean_reward);
          chart.update();
          statusEl.style.color = 'var(--color-muted)';
          statusEl.textContent =
            `Step ${data.timestep.toLocaleString()} \u2014 Mean reward: ${data.mean_reward.toFixed(1)}`;
        }
      };

      es.onerror = () => {
        statusEl.textContent = 'Stream error \u2014 training may still be running in background.';
        statusEl.style.color = 'var(--color-warn)';
        es.close();
        trainBtn.disabled = false;
      };
    } catch (err) {
      statusEl.textContent = 'Error: ' + err.message;
      statusEl.style.color = 'var(--color-error)';
      trainBtn.disabled = false;
    }
  });
})();
