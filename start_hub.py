"""
Physical AI Workshop — Hub Launcher
------------------------------------
Starts the API server in a new terminal window and opens the browser.

Usage (after running setup.bat once):
    python start_hub.py

setup.bat and start_hub.bat both call this automatically.
Participants can also run it directly to restart the hub without
re-running the full setup.
"""

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

PORT = 8000
SERVER = Path(__file__).parent / "api" / "server.py"
HEALTH_URL = f"http://localhost:{PORT}/api/config"
# How long to wait for the server to become healthy before opening the browser.
HEALTH_TIMEOUT_S = 20
HEALTH_POLL_S = 0.5


def free_port(port: int) -> None:
    """Kill any process currently *listening* on *port* (Windows only).

    Filters strictly to LISTENING lines so that established connections
    (browser tabs, TIME_WAIT sockets) are never targeted.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        seen_pids: set[str] = set()
        for line in result.stdout.splitlines():
            # Match only the listener: "0.0.0.0:<port>  0.0.0.0:0  LISTENING  <pid>"
            if f":{port} " in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and pid not in seen_pids:
                    seen_pids.add(pid)
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True
                    )
        if seen_pids:
            # Give the OS a moment to release the socket before we bind again.
            time.sleep(0.5)
    except Exception:
        pass  # non-fatal — server will report the conflict if it persists


def launch_server() -> None:
    """Open a new cmd window running the API server with the venv Python."""
    title = "Physical AI Workshop Hub"
    # Use sys.executable so we always use the venv Python, not whatever is on
    # PATH.  /k keeps the window open after the server exits so errors are
    # visible; /d sets the working directory so relative imports in server.py
    # resolve correctly.
    cmd = (
        f'start "{title}" cmd /k '
        f'"{sys.executable}" "{SERVER}"'
    )
    subprocess.Popen(cmd, shell=True, cwd=Path(__file__).parent)


def wait_for_server(timeout: float = HEALTH_TIMEOUT_S) -> bool:
    """Poll /api/config until the server responds 200 or *timeout* elapses.

    Returns True when healthy, False on timeout.
    """
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def main() -> None:
    print()
    print("  Physical AI Workshop — starting hub...")
    print()

    free_port(PORT)
    launch_server()

    print("  Waiting for server to become healthy", end="", flush=True)
    ok = wait_for_server()
    print()  # newline after the dots

    if ok:
        print(f"  [OK] Server healthy at http://localhost:{PORT}")
    else:
        print(f"  [WARN] Server did not respond within {HEALTH_TIMEOUT_S}s.")
        print("         Check the hub window for error details.")
        print("         Opening browser anyway — it may need a manual refresh.")

    webbrowser.open(f"http://localhost:{PORT}")

    print()
    print(f"  Hub:     http://localhost:{PORT}  (separate window)")
    print("  Browser: opening now")
    print()
    print("  Use Chrome or Edge — Firefox is not supported (MediaPipe).")
    print("  Tip: run  python start_hub.py  any time to restart the hub.")
    print()


if __name__ == "__main__":
    main()
