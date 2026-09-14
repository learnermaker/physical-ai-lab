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
    const s = el.querySelector('.svc-status, .badge');   // handles both pre- and post-setBadge state
    if (s) {
      s.className = 'svc-status ' + cls;
      s.textContent = text;
      // Update tooltip content for the Gemini tile based on current state
      if (id === 'gemini') {
        if (cls === 'svc-ok') {
          s.setAttribute('data-tooltip', 'Gemini key ready — Module 5 will use live inference.');
        } else if (cls === 'svc-warn') {
          s.setAttribute('data-tooltip', 'No key → cache only. Module 5 returns pre-written responses.');
        } else if (cls === 'svc-error') {
          s.setAttribute('data-tooltip', 'No key and no cache — Module 5 Gemini calls will fail.');
        }
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

    const geminiOk  = status.gemini_key_configured;    // server-side .env key
    const cacheOk   = status.gemini_cache_loaded;
    const browserKey = window.hubConfig?.getApiKey();  // browser sessionStorage key

    // Show the browser key state (what the hub actually uses for Module 5)
    // The server .env key is used by the Python scripts in the terminal.
    if (browserKey) {
      setBadge(geminiSvrBadge, 'ok', 'Key saved (browser)');
      setTile('gemini', 'svc-ok', 'Browser key set');
    } else if (geminiOk) {
      setBadge(geminiSvrBadge, 'ok', 'Configured (.env)');
      setTile('gemini', 'svc-ok', 'Server .env key');
    } else {
      setBadge(geminiSvrBadge, 'warn', 'No key set');
      setTile('gemini', cacheOk ? 'svc-warn' : 'svc-error',
                        cacheOk ? 'Cache only' : 'No key/cache');
    }

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
// ---------------------------------------------------------------------------
// serverControl — status panel buttons
//
// "Refresh status"   — re-polls /api/status and updates all tiles immediately.
//                      Can't restart the Python process from the browser, but
//                      this confirms it's alive and surfaces any new errors.
//
// "Stop simulation"  — calls POST /api/simulation/stop.  This sets _sim_stop
//                      so any stuck worker exits within ~50 ms and releases
//                      the lock.  Useful after a crash or when switching
//                      between Module 2 and Module 4 without pressing Stop.
// ---------------------------------------------------------------------------
(() => {
  const restartBtn = document.getElementById('server-restart-btn');
  const simStopBtn = document.getElementById('sim-stop-btn');
  const logEl      = document.getElementById('server-log');

  function log(msg, isError) {
    if (!logEl) return;
    logEl.classList.add('visible');
    logEl.style.color = isError ? 'var(--color-error, #c0392b)' : '';
    logEl.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  }

  if (restartBtn) {
    restartBtn.addEventListener('click', async () => {
      restartBtn.disabled = true;
      log('Checking server…');
      try {
        const status = await hubConfig.fetchStatus();
        if (status) {
          log('Server is running — status tiles refreshed.');
          serverCheck.check();
        } else {
          log('Server unreachable. Run: python start_hub.py', true);
        }
      } catch {
        log('Server unreachable. Run: python start_hub.py', true);
      } finally {
        restartBtn.disabled = false;
      }
    });
  }

  if (simStopBtn) {
    simStopBtn.addEventListener('click', async () => {
      simStopBtn.disabled = true;
      log('Stopping simulation worker…');
      try {
        const r = await fetch('/api/simulation/stop', {
          method: 'POST',
          signal: AbortSignal.timeout(3000),
        });
        if (r.ok) {
          log('Simulation stopped — lock released. You can now start a fresh simulation.');
        } else {
          log('Stop request failed (server returned ' + r.status + ').', true);
        }
      } catch {
        log('Could not reach server — is it running?', true);
      } finally {
        simStopBtn.disabled = false;
        // Refresh tiles so the status panel reflects the new state
        setTimeout(() => serverCheck.check(), 400);
      }
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
    serverCheck.check();   // update Gemini tile immediately
  });

  clearBtn.addEventListener('click', () => {
    hubConfig.clearApiKey();
    input.value = '';
    notice.style.display = 'none';
    testOut.style.display = 'none';
    serverCheck.check();   // update Gemini tile immediately
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
      // Sends an empty frame — the server now handles this with a text-only prompt
      // ("Reply: API key is valid.") rather than an empty image, so the key is
      // actually validated without needing a camera frame.
      const res = await fetch('/api/gemini/describe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_b64: '', api_key: key }),
        signal: AbortSignal.timeout(35000),
      });
      const data = await res.json();
      const isCache = data._source === 'cache';
      if (isCache) {
        // Key test failed — show reason clearly, skip the confusing cache response text
        const reason = (data._reason || 'unknown');
        const isNoKey = reason === 'no_api_key';
        testOut.style.color = 'var(--color-warn)';

        let msg = '';
        if (isNoKey) {
          msg = 'No key transmitted (internal error — try again).';
        } else if (reason.includes('401') || reason.includes('UNAUTHENTICATED')) {
          msg = 'Key rejected (401) — this is not a valid Gemini API key.\n\nGemini API keys now start with AQ. and are issued at:\nhttps://aistudio.google.com/app/apikey';
        } else if (reason.includes('429') || reason.includes('RESOURCE_EXHAUSTED')) {
          msg = 'Rate limit hit (429) — the key is valid but you\'ve exceeded the free quota.\n\nWait a minute and try again, or check usage at https://aistudio.google.com/';
        } else {
          msg = 'API call failed.\n\nReason: ' + reason.substring(0, 200)
            + '\n\nGet a free key at https://aistudio.google.com/app/apikey';
        }
        testOut.textContent = msg;
      } else {
        // Key is valid — show the response, but strip internal fields
        hubConfig.setApiKey(key);
        notice.style.display = 'inline';
        setTimeout(() => { notice.style.display = 'none'; }, 3000);
        testOut.style.color = 'var(--color-ok)';
        const display = Object.fromEntries(Object.entries(data).filter(([k]) => !k.startsWith('_')));
        testOut.textContent = '\u2714 Key is valid — Gemini responded:\n\n' + JSON.stringify(display, null, 2);
        serverCheck.check();   // update Gemini tile to "Browser key set"
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
    trainBtn.setAttribute('aria-busy', 'true');   // Pico spinner on button
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
          trainBtn.removeAttribute('aria-busy');
          if (data.error) {
            statusEl.textContent = 'Training failed: ' + data.error;
            statusEl.style.color = 'var(--color-error)';
          } else {
            statusEl.textContent = 'Training complete!';
            statusEl.style.color = 'var(--color-ok)';
          }
          es.close();
          trainBtn.disabled = false;
          serverCheck.check();
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
        trainBtn.removeAttribute('aria-busy');
        statusEl.textContent = 'Stream error \u2014 training may still be running in background.';
        statusEl.style.color = 'var(--color-warn)';
        es.close();
        trainBtn.disabled = false;
      };
    } catch (err) {
      trainBtn.removeAttribute('aria-busy');
      statusEl.textContent = 'Error: ' + err.message;
      statusEl.style.color = 'var(--color-error)';
      trainBtn.disabled = false;
    }
  });
})();


// ---------------------------------------------------------------------------
// File opener — auto-wire all <code> elements that reference .py / .ipynb files
// Adds a small [open] button next to each reference; clicking calls the server
// which uses the OS default handler (whatever IDE the user has set up).
// ---------------------------------------------------------------------------
(() => {
  /**
   * Call the server to open a file using the OS default handler.
   * @param {string} relPath  Relative path from repo root, e.g. "modules/01_perception/exercise.py"
   * @param {HTMLElement} btn  The button element — shows brief feedback
   */
  async function openFile(relPath, btn) {
    const original = btn.textContent;
    btn.textContent = '\u23f3';   // hourglass while opening
    btn.disabled = true;
    try {
      const res = await fetch(
        '/api/open-file?path=' + encodeURIComponent(relPath),
        { signal: AbortSignal.timeout(5000) }
      );
      const data = await res.json();
      if (data.opened) {
        btn.textContent = '\u2713';  // checkmark on success
        setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1500);
      } else {
        btn.textContent = '!';
        btn.title = data.error || 'Could not open file';
        setTimeout(() => { btn.textContent = original; btn.disabled = false; btn.title = ''; }, 3000);
      }
    } catch {
      btn.textContent = original;
      btn.disabled = false;
    }
  }

  /**
   * Returns true if the text looks like a file path we can open.
   * Matches: modules/XX_.../something.py, utils/something.py, something.ipynb
   */
  function isOpenablePath(text) {
    return /\.(py|ipynb)$/.test(text.trim()) &&
           /^(modules\/|utils\/|api\/|[a-zA-Z0-9_/.-]+\.(py|ipynb)$)/.test(text.trim());
  }

  function wireFileLinks() {
    document.querySelectorAll('code').forEach(el => {
      const text = el.textContent.trim();
      if (!isOpenablePath(text)) return;
      // Don't wire "Create a new file" instructions — those files don't exist yet
      if (el.hasAttribute('data-no-open')) return;
      // Don't double-wire
      if (el.nextSibling && el.nextSibling.classList?.contains('file-open-btn')) return;

      const btn = document.createElement('button');
      btn.className = 'file-open-btn';
      btn.textContent = 'open';
      btn.title = 'Open in default editor: ' + text;
      btn.setAttribute('aria-label', 'Open ' + text + ' in default editor');
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        openFile(text, btn);
      });

      // Insert immediately after the <code> element
      el.insertAdjacentElement('afterend', btn);
    });
  }

  // Run once DOM is ready, then re-run if content is dynamically injected
  wireFileLinks();
})();
