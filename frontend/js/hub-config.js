/**
 * hub-config.js — Shared hub infrastructure
 *
 * Exports (via window.*):
 *   window.hubConfig   — API key management + /api/config fetch
 *
 * Also initialises (no export needed):
 *   serverCheck        — polls /api/config, updates badges + banner
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

  function getApiKey() {
    return sessionStorage.getItem(KEY_NAME) || null;
  }

  function setApiKey(key) {
    const trimmed = (key || '').trim();
    if (trimmed) {
      sessionStorage.setItem(KEY_NAME, trimmed);
    } else {
      sessionStorage.removeItem(KEY_NAME);
    }
  }

  function clearApiKey() {
    sessionStorage.removeItem(KEY_NAME);
  }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config', { signal: AbortSignal.timeout(4000) });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    }
  }

  return { getApiKey, setApiKey, clearApiKey, fetchConfig };
})();

window.hubConfig = hubConfig;


// ---------------------------------------------------------------------------
// serverCheck — poll /api/config on load, update banner + status badges
// ---------------------------------------------------------------------------
const serverCheck = (() => {
  const banner         = document.getElementById('server-banner');
  const serverBadge    = document.getElementById('server-badge');
  const geminiSvrBadge = document.getElementById('gemini-server-badge');

  function setBadge(el, state, text) {
    el.className = `badge ${state}`;
    el.textContent = text;
  }

  function disableServerControls() {
    document.querySelectorAll('[data-requires-server]').forEach(el => {
      el.disabled = true;
    });
  }

  function enableServerControls() {
    document.querySelectorAll('[data-requires-server]').forEach(el => {
      el.disabled = false;
    });
  }

  async function check() {
    const config = await hubConfig.fetchConfig();
    if (!config) {
      document.body.classList.add('server-down');
      setBadge(serverBadge, 'error', 'Unreachable');
      setBadge(geminiSvrBadge, 'warn', 'Unknown');
      disableServerControls();
    } else {
      document.body.classList.remove('server-down');
      setBadge(serverBadge, 'ok', 'Connected');
      setBadge(geminiSvrBadge,
        config.gemini_key_configured ? 'ok' : 'warn',
        config.gemini_key_configured ? 'Configured' : 'Not configured');
      enableServerControls();
    }
  }

  check();
  setInterval(check, 10_000);

  return { check };
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
  if (existing) {
    input.value = existing;
    notice.style.display = 'inline';
  }

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
      const isCached = !data.text || data.text.length < 5;
      if (isCached) {
        testOut.style.color = 'var(--color-warn)';
        testOut.textContent =
          'Received a cache fallback \u2014 key may be invalid or network unavailable.\n\n' +
          JSON.stringify(data, null, 2);
      } else {
        hubConfig.setApiKey(key);
        notice.style.display = 'inline';
        setTimeout(() => { notice.style.display = 'none'; }, 3000);
        testOut.style.color = 'var(--color-ok)';
        testOut.textContent = 'Gemini responded:\n\n' + JSON.stringify(data, null, 2);
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

  const chartCtx = document.getElementById('m3-chart').getContext('2d');
  const chart = new Chart(chartCtx, {     // Chart is a UMD global from index.html <head>
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
    statusEl.textContent = 'Starting training...';
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();

    try {
      const res = await fetch('/api/training/start', { method: 'POST' });
      const { job_id } = await res.json();
      if (!job_id) throw new Error('No job_id returned');

      statusEl.textContent = `Training job ${job_id.substring(0, 8)}... (50k steps)`;
      const es = new EventSource(`/api/training/progress/${job_id}`);

      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.done) {
          statusEl.textContent = 'Training complete!';
          es.close();
          trainBtn.disabled = false;
          return;
        }
        if (data.timestep != null && data.mean_reward != null) {
          chart.data.labels.push(data.timestep.toString());
          chart.data.datasets[0].data.push(data.mean_reward);
          chart.update();
          statusEl.textContent =
            `Step ${data.timestep.toLocaleString()} \u2014 Mean reward: ${data.mean_reward.toFixed(1)}`;
        }
      };

      es.onerror = () => {
        statusEl.textContent = 'Stream error.';
        es.close();
        trainBtn.disabled = false;
      };
    } catch (err) {
      statusEl.textContent = 'Error: ' + err.message;
      trainBtn.disabled = false;
    }
  });
})();
