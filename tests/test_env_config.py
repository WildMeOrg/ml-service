"""Tests for env-driven server configuration in app.main.

The same container image must run unmodified across providers: Cloud Run
injects PORT, RunPod/VMs set DEVICE etc. Env vars supply argparse defaults;
explicit CLI flags still win. app.main parses sys.argv at import time, so
each probe runs in a subprocess.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _probe(attr, env=None, argv=None):
    """Import app.main in a subprocess and return getattr(main.args, attr)."""
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("PORT", "HOST", "DEVICE", "WORKERS")}
    child_env.update(env or {})
    code = (
        "import sys; "
        f"sys.argv = ['app.main'] + {argv or []!r}; "
        "from app import main; "
        f"print('PROBE:' + str(getattr(main.args, {attr!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=child_env, cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("PROBE:"):
            return line[len("PROBE:"):]
    raise AssertionError(f"probe output missing: {result.stdout!r}")


def test_port_defaults_to_8888_without_env():
    """Bare-metal default is unchanged (silent breaks forbidden)."""
    assert _probe("port") == "8888"


def test_port_env_supplies_default():
    assert _probe("port", env={"PORT": "7777"}) == "7777"


def test_cli_flag_overrides_port_env():
    assert _probe("port", env={"PORT": "7777"}, argv=["--port", "7010"]) == "7010"


def test_malformed_port_env_falls_back():
    """Providers sometimes inject junk (k8s service links); never crash."""
    assert _probe("port", env={"PORT": "tcp://10.0.0.1:80"}) == "8888"


def test_whitespace_padded_port_env_falls_back():
    """Only bare ASCII digits are accepted, exactly matching the image
    healthcheck's shell validation — the server and probe must never
    disagree about the bound port."""
    assert _probe("port", env={"PORT": " 7777 "}) == "8888"


def test_empty_port_env_falls_back():
    assert _probe("port", env={"PORT": ""}) == "8888"


def test_out_of_range_port_env_falls_back():
    """Ports outside 1-65535 can't serve traffic; fall back like other junk."""
    assert _probe("port", env={"PORT": "0"}) == "8888"
    assert _probe("port", env={"PORT": "65536"}) == "8888"


def test_device_env_supplies_default():
    assert _probe("device", env={"DEVICE": "cpu"}) == "cpu"


def test_host_env_supplies_default():
    assert _probe("host", env={"HOST": "127.0.0.1"}) == "127.0.0.1"


def test_workers_env_supplies_default():
    assert _probe("workers", env={"WORKERS": "2"}) == "2"
