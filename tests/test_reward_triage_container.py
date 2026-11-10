"""Container isolation checks (skip if docker unavailable)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker unavailable")
def test_container_blocks_socket_and_env_secrets():
    compose = ROOT / "docker-compose.triage.yml"
    assert compose.exists()
    # Build may be slow; use prebuilt if present, else build once.
    build = subprocess.run(
        ["docker", "compose", "-f", str(compose), "build", "triage-sandbox"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    run = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose),
            "run",
            "--rm",
            "-e",
            "OPENAI_API_KEY=",
            "triage-sandbox",
            "python",
            "-m",
            "reward_triage.container_selftest",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    assert "container_selftest: pass" in run.stdout
