"""
tests/test_verify_install.py — Unit tests for verify_install.py

Tests cover Requirements 5.5:
  5.5  verify_install.py prints a correctly formatted summary line.

The script is run as a subprocess because it is a standalone script (not a
module), so import-level side-effects cannot be mocked from the outside.
The test only checks the summary line format — not the pass/fail counts —
because the test environment may lack webcam hardware or optional packages.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "verify_install.py"


def test_summary_line_format():
    """verify_install.py prints a correctly formatted summary line.

    Regardless of which checks pass or fail in the test environment, the
    script must emit a line matching:
        Setup complete: <N>/14 checks passed[...]
    This validates Requirement 5.5 — the script always reports a summary.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    # Exit code 0 (all pass) or 1 (some fail) are both valid outcomes.
    # We do not assert on the exit code — only the output format matters.
    combined = result.stdout + result.stderr
    assert re.search(r"Setup complete: \d+/14 checks passed", combined), (
        f"Expected summary line not found in script output.\n\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
