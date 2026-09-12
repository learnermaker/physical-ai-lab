import subprocess, sys
r = subprocess.run(
    [sys.executable, "verify_install.py"],
    capture_output=True, text=True,
    cwd=r"c:\Users\JIMSEELAN\Desktop\Projects\physical-ai\physical-ai-workshop"
)
print(r.stdout)
if r.stderr: print("STDERR:", r.stderr[-300:])
print("Exit:", r.returncode)
