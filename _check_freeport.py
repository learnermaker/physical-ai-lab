"""Audit the free_port logic in start_hub.py against live netstat output."""
import subprocess

PORT = 8000
result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
print(f"=== Lines matching ':{PORT}' ===")
for line in result.stdout.splitlines():
    if f":{PORT}" in line:
        print(repr(line))

print()
print("=== Current free_port() would kill ===")
for line in result.stdout.splitlines():
    if f":{PORT}" in line and ("LISTENING" in line or "0.0.0.0" in line):
        parts = line.split()
        pid = parts[-1]
        print(f"  pid={pid!r}  [{line.strip()}]")

print()
print("=== Fixed logic (LISTENING only, not 0.0.0.0) ===")
for line in result.stdout.splitlines():
    if f":{PORT}" in line and "LISTENING" in line:
        parts = line.split()
        pid = parts[-1]
        print(f"  pid={pid!r}  [{line.strip()}]")
