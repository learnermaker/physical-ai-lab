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
    """Kill any process currently *listening* on *port*.

    Cross-platform: uses taskkill on Windows, lsof/kill on Linux/macOS.
    Wrapped in a broad try/except — entirely non-fatal; the server will
    report any remaining conflict on startup.
    """
    try:
        if sys.platform == "win32":
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
                time.sleep(0.5)
        else:
            # Linux / macOS: lsof finds the PID, xargs kill terminates it
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True
            )
            pids = [p.strip() for p in result.stdout.splitlines() if p.strip().isdigit()]
            for pid in pids:
                subprocess.run(["kill", "-9", pid], capture_output=True)
            if pids:
                time.sleep(0.5)
    except Exception:
        pass  # non-fatal — server will report the conflict if it persists


def launch_server() -> None:
    """Start the API server in a detached process.

    On Windows: opens a new console window (CREATE_NEW_CONSOLE).
    On Linux/macOS: detaches from the parent session (start_new_session=True)
    so the server survives even if the launching terminal is closed.
    """
    kwargs: dict = {"cwd": Path(__file__).parent}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000010  # CREATE_NEW_CONSOLE
    else:
        kwargs["start_new_session"] = True    # detach on Linux/macOS
    subprocess.Popen([sys.executable, str(SERVER)], **kwargs)


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
