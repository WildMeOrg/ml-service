"""Tests for env-driven server configuration in app.main.

The same container image must run unmodified across providers: Cloud Run
injects PORT, RunPod/VMs set DEVICE etc. Env vars supply argparse defaults;
explicit CLI flags still win. app.main parses sys.argv at import time, so
each probe runs in a subprocess.
"""
import os
import pathlib
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _probe(attr, env=None, argv=None):
    """Import app.main in a subprocess and return getattr(main.args, attr)."""
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("PORT", "HOST", "DEVICE", "WORKERS",
                              "LIMIT_CONCURRENCY")}
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
    """A stale or mis-copied PORT must not crash startup."""
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


def test_limit_concurrency_env_supplies_default():
    """Added after the merge with main, which introduced the flag: a deploy
    knob outside the env contract would leave the image non-portable."""
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "64"}) == "64"


def test_malformed_limit_concurrency_env_falls_back():
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "lots"}) == "32"


def test_limit_concurrency_of_one_falls_back():
    """uvicorn 503s when len(connections) >= limit and counts the connection
    it is serving, so a limit of 1 rejects every request -- /health included,
    which makes the probe kill a healthy server. Floor the env at 2."""
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "1"}) == "32"
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "2"}) == "2"


# --- server/probe agreement -------------------------------------------------
#
# The image healthcheck re-implements _int_env in shell. The two must agree on
# every input or the probe kills a server that bound a different port, so pin
# them against one shared matrix -- reading the shell out of the dockerfile so
# an edit there cannot drift away from the Python.

PORT_MATRIX = ["6050", "8888", "1", "65535", "", "   ", "tcp://10.0.0.1:80",
               " 7777 ", "0", "65536", "07777", "abc", "-1", "1e3"]


def _healthcheck_port_script():
    """The dockerfile HEALTHCHECK's port resolution, with curl swapped out."""
    text = pathlib.Path(REPO_ROOT, "docker", "dockerfile").read_text()
    joined = text.replace("\\\n", " ")
    line = next(l for l in joined.splitlines() if l.startswith("HEALTHCHECK"))
    body = line.split("CMD", 1)[1]
    resolution = body.split("curl", 1)[0].rstrip().rstrip(";")
    assert "case" in resolution and "8888" in resolution, resolution
    return resolution + '; echo "${p}"'


def test_healthcheck_mirrors_int_env_on_every_input():
    script = _healthcheck_port_script()
    for value in PORT_MATRIX:
        shell = subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True,
            env={"PORT": value, "PATH": os.environ.get("PATH", "")}, timeout=30)
        assert shell.returncode == 0, f"PORT={value!r}: {shell.stderr}"
        # Compared numerically: the shell echoes the literal string, so a
        # zero-padded "07777" is the same port as argparse's int 7777.
        probe = _probe("port", env={"PORT": value})
        assert int(shell.stdout.strip()) == int(probe), (
            f"PORT={value!r}: probe resolves {shell.stdout.strip()}, "
            f"server resolves {probe}"
        )
