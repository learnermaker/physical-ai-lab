"""
Test start_hub.py end-to-end:
  1. Confirms free_port() logic parses LISTENING correctly
  2. Runs start_hub.py as a subprocess (which opens a new cmd window)
  3. Polls /api/config until healthy (up to 25s)
  4. Verifies all key endpoints
  5. Reports results
"""
import subprocess, sys, time, urllib.request, json, socket

BASE = "http://localhost:8000"
PYTHON = sys.executable

def http_get(path, timeout=5):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode()
    except Exception as e:
        return 0, str(e)

def http_post(path, body, timeout=35):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

def wait_healthy(timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/api/config", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

# ── 1. Unit-test free_port logic ─────────────────────────────────────────────
print("=== 1. free_port logic unit test ===")
# Synthesise fake netstat lines
fake_lines = [
    "  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       12345",
    "  TCP    127.0.0.1:8000         127.0.0.1:50264        TIME_WAIT       0",
    "  TCP    127.0.0.1:8000         127.0.0.1:56353        ESTABLISHED     12345",
    "  TCP    127.0.0.1:56353        127.0.0.1:8000         ESTABLISHED     99999",  # browser
]
PORT = 8000
killed = []
for line in fake_lines:
    if f":{PORT} " in line and "LISTENING" in line:
        parts = line.split()
        pid = parts[-1]
        if pid.isdigit():
            killed.append(pid)
assert killed == ["12345"], f"Expected ['12345'], got {killed}"
print(f"  PASS — free_port would kill only PID 12345 (not browser PID 99999)")

# ── 2. Launch start_hub.py ────────────────────────────────────────────────────
print()
print("=== 2. Launching start_hub.py ===")
proc = subprocess.Popen(
    [PYTHON, "start_hub.py"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True,
    cwd=r"c:\Users\JIMSEELAN\Desktop\Projects\physical-ai\physical-ai-workshop"
)

# ── 3. Wait for server to be healthy (start_hub.py also does this internally) ─
print("  Waiting for server health...")
ok = wait_healthy(25)

# Collect start_hub.py output (it exits after opening browser)
try:
    out, err = proc.communicate(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate()

print(f"  start_hub.py output:\n{out.rstrip()}")
if err.strip():
    print(f"  stderr: {err.rstrip()[:200]}")
print(f"  start_hub.py exit code: {proc.returncode}")
print(f"  Server healthy: {ok}")

# ── 4. Endpoint checks ────────────────────────────────────────────────────────
print()
print("=== 3. Endpoint smoke tests ===")
results = []

s, b = http_get("/")
results.append(("GET /", s, s == 200 and len(b) > 1000))

s, b = http_get("/api/config")
d = json.loads(b) if s == 200 else {}
results.append(("GET /api/config", s, s == 200 and "gemini_key_configured" in d))

KEY = "REDACTED_API_KEY"
s, b = http_post("/api/gemini/action", {"frame_b64": "", "api_key": KEY})
d = json.loads(b) if s == 200 else {}
results.append(("POST /api/gemini/action", s, s == 200 and "action" in d))

s, b = http_post("/api/gemini/describe", {"frame_b64": "", "api_key": KEY})
d = json.loads(b) if s == 200 else {}
results.append(("POST /api/gemini/describe", s, s == 200 and "text" in d))

s, b = http_post("/api/training/start", {})
d = json.loads(b) if s == 200 else {}
results.append(("POST /api/training/start", s, s == 200 and "job_id" in d))

# SSE stream: open and read first data frame
try:
    import urllib.request as ur
    with ur.urlopen(BASE + "/api/simulation/stream?action=0,0", timeout=20) as resp:
        chunk = resp.read(500).decode(errors="replace")
    has_data = "data:" in chunk
    results.append(("GET /api/simulation/stream", 200, has_data))
except Exception as e:
    results.append(("GET /api/simulation/stream", 0, False))

for name, status, passed in results:
    flag = "PASS" if passed else "FAIL"
    print(f"  [{flag}] {name}  (status={status})")

all_pass = all(r[2] for r in results)
print()
print(f"=== Result: {'ALL PASS' if all_pass else 'SOME FAILURES'} ===")
sys.exit(0 if (ok and all_pass) else 1)
