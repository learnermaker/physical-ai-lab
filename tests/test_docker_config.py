"""
tests/test_docker_config.py — Static contract tests for the Docker workshop deployment.

These tests run WITHOUT Docker. They lock the participant-facing contract so a
future edit cannot silently break it:
  - localhost-only port binding, no privileged mode, no docker socket
  - Dockerfile essentials (3.12 base, headless MuJoCo, non-root, healthcheck)
  - requirements.txt / environment.yml mediapipe consistency (Tasks + Solutions APIs)
  - model asset wiring (download script, gitkeep placeholders, fallback video)
  - startup scripts (Compose v2, health polling, correct URLs)
  - FastAPI server container-readiness (PORT env, 0.0.0.0 bind, health routes)

Docker build / runtime / browser checks remain the job of CI (docker-build.yml)
and verify_docker.py (executed inside the built image).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
yaml = pytest.importorskip("yaml", reason="pyyaml required (present via huggingface_hub)")


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _compose() -> dict:
    return yaml.safe_load((REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Compose contract
# ---------------------------------------------------------------------------

def test_compose_single_service_localhost_only():
    cfg = _compose()
    services = cfg.get("services", {})
    assert list(services.keys()) == ["lab"], "exactly one service named 'lab'"
    ports = [str(p) for p in services["lab"].get("ports", [])]
    assert len(ports) == 1
    assert ports[0].startswith("127.0.0.1:"), "must bind localhost only"
    assert ports[0].endswith(":8000"), "container port must be 8000"
    assert "${HOST_PORT:-8000}" in ports[0], "host port must be HOST_PORT-overridable"


def test_compose_no_privilege_escalation():
    svc = _compose()["services"]["lab"]
    assert svc.get("privileged") is not True
    assert svc.get("network_mode") != "host"
    blob = " ".join(str(v) for v in svc.get("volumes", []))
    assert "docker.sock" not in blob, "must never mount the docker socket"
    assert "init: true" in _read("compose.yaml")


def test_compose_persistence_and_env():
    text = _read("compose.yaml")
    assert ".:/workspace" in text, "repo bind-mount required for participant persistence"
    assert "required: false" in text, ".env must be optional (server works keyless)"
    assert "unless-stopped" in text, "container must survive terminal close / reboot"
    assert "shm_size" in text


# ---------------------------------------------------------------------------
# Dockerfile contract
# ---------------------------------------------------------------------------

def test_dockerfile_base_and_headless_env():
    text = _read("Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "MUJOCO_GL=osmesa" in text
    assert "MPLBACKEND" in text and "Agg" in text


def test_dockerfile_non_root_and_health():
    text = _read("Dockerfile")
    assert re.search(r"^USER\s+\w+", text, re.M) and "USER root" not in text
    assert "EXPOSE 8000" in text and "EXPOSE 8888" not in text
    assert "HEALTHCHECK" in text and "/api/config" in text
    assert 'CMD ["python", "api/server.py"]' in text


def test_dockerfile_no_secrets_and_pinned_torch():
    text = _read("Dockerfile")
    assert "GEMINI_API_KEY" not in text
    assert ".env" not in text
    assert "torch==2.14.0+cpu" in text
    assert "WARNING" in text, "torch fallback must be loud, not silent"


# ---------------------------------------------------------------------------
# Dependency consistency (requirements.txt vs environment.yml)
# ---------------------------------------------------------------------------

def _mediapipe_pin(line_source: str) -> str | None:
    for line in line_source.splitlines():
        if "mediapipe" in line:
            m = re.search(r"mediapipe\s*([<>=!]+[^#\s]*)", line)
            if not m:
                return "UNPARSEABLE:" + line.strip()
            spec = m.group(1)
            exact = re.fullmatch(r"==([0-9.]+)", spec)
            return exact.group(1) if exact else "RANGE:" + spec
    return None


def test_mediapipe_pinned_consistently():
    req_pin = _mediapipe_pin(_read("requirements.txt"))
    env_text = _read("environment.yml")
    env_pin = _mediapipe_pin(env_text)
    assert req_pin == "0.10.21", f"requirements.txt must pin mediapipe==0.10.21, got {req_pin}"
    assert env_pin == "0.10.21", f"environment.yml must match (Tasks + Solutions APIs), got {env_pin}"


def test_environment_yml_has_backend_deps():
    env_text = _read("environment.yml")
    assert "fastapi" in env_text, "conda env needs fastapi for api/server.py"
    assert "uvicorn" in env_text, "conda env needs uvicorn for api/server.py"


# ---------------------------------------------------------------------------
# Model asset wiring
# ---------------------------------------------------------------------------

def test_model_assets_wired():
    dl = _read("download_model.py")
    assert "hand_landmarker" in dl and "HalfCheetah" in dl
    assert (REPO_ROOT / "assets" / ".gitkeep").exists()
    assert (REPO_ROOT / "models" / ".gitkeep").exists()
    assert (REPO_ROOT / "frontend" / "assets" / "fallback_hand_demo.mp4").exists()


def test_dockerignore_keeps_secrets_and_assets_out():
    di = _read(".dockerignore")
    for needle in [".env", ".venv/", "assets/*.task", "assets/*.mp4", "models/*.zip", "__pycache__/"]:
        assert needle in di, f".dockerignore missing {needle}"


# ---------------------------------------------------------------------------
# Startup scripts
# ---------------------------------------------------------------------------

def test_start_scripts_use_compose_v2_and_health():
    for name in ("docker-start.sh", "docker-start.bat"):
        text = _read(name)
        assert "docker compose" in text
        assert "/api/config" in text
        assert "8000" in text


def test_windows_start_script_respects_host_port_env():
    text = _read("docker-start.bat")
    assert 'if "%HOST_PORT%"=="" set "HOST_PORT=8000"' in text


def test_stop_scripts_preserve_data():
    for name in ("docker-stop.sh", "docker-stop.bat"):
        text = _read(name)
        assert "docker compose down" in text
        assert "-v" not in text.replace("docker-compose", ""), f"{name} must not delete volumes"


# ---------------------------------------------------------------------------
# FastAPI container-readiness (static: no server started)
# ---------------------------------------------------------------------------

def test_server_binds_all_interfaces_with_port_env():
    text = _read("api/server.py")
    assert 'os.environ.get("PORT"' in text
    assert 'host="0.0.0.0"' in text


def test_server_health_routes_exist():
    text = _read("api/server.py")
    assert '"/api/config"' in text and '"/api/status"' in text


# ---------------------------------------------------------------------------
# Docs point at the same contract
# ---------------------------------------------------------------------------

def test_docs_reference_single_workflow():
    setup = _read("DOCKER_SETUP.md")
    readme = _read("README.md")
    assert "docker-start.bat" in setup and "docker-start.sh" in setup
    assert "HOST_PORT" in setup
    assert "DOCKER_SETUP.md" in readme
    assert "PHYSICAL_AI_PORT" not in setup, "one port var only (HOST_PORT)"
    assert "PHYSICAL_AI_PORT" not in readme
