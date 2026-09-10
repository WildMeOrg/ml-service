"""Tests for env-driven server configuration in app.main.

The same container image must run unmodified across providers: Cloud Run
injects PORT, RunPod/VMs set DEVICE etc. Env vars supply argparse defaults;
explicit CLI flags still win. app.main parses sys.argv at import time, so
each probe runs in a subprocess.
"""
import json
import os
import pathlib
import re
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
    marking a healthy container unhealthy for autoheal to restart. Values
    below 2 are refused, which means the fallback 32, not a clamp to 2."""
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "1"}) == "32"
    assert _probe("limit_concurrency", env={"LIMIT_CONCURRENCY": "2"}) == "2"


# --- server/probe agreement -------------------------------------------------
#
# The image healthcheck re-implements _int_env in shell. The two must agree on
# every input or the probe targets a port the server never bound, so pin them
# against one shared matrix -- reading the shell out of the dockerfile and
# calling the real _int_env, so an edit to either side cannot drift.

PORT_MATRIX = [
    "6050", "8888", "1", "65535",       # valid
    "", "   ", "\t7777\n", " 7777 ",    # empty / whitespace
    "tcp://10.0.0.1:80", "abc", "1e3", "-1", "+7777",  # not a bare integer
    "٧٧٧٧",                              # decimal but not ASCII
    "0", "65536",                        # out of range
    "9" * 5000,                          # past CPython's int-conversion limit
    "0" * 5000 + "7777",                 # ... and numerically in range once parsed
    "07777", "007777",                   # zero-padded, at and over five digits
]


def _int_env_matrix():
    """The real _int_env's verdict on every matrix input, in one subprocess."""
    code = (
        "import json, os, sys; "
        "sys.argv = ['app.main']; "
        "from app import main; "
        "vals = json.loads(sys.stdin.read()); "
        "out = [];\n"
        "for v in vals:\n"
        "    os.environ['PORT'] = v\n"
        "    out.append(main._int_env('PORT', 8888, maximum=65535))\n"
        "print('MATRIX:' + json.dumps(out))"
    )
    child_env = {k: v for k, v in os.environ.items()
                 if k not in ("PORT", "HOST", "DEVICE", "WORKERS",
                              "LIMIT_CONCURRENCY")}
    result = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(PORT_MATRIX),
        capture_output=True, text=True, env=child_env, cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("MATRIX:"):
            return json.loads(line[len("MATRIX:"):])
    raise AssertionError(f"matrix output missing: {result.stdout!r}")


def _healthcheck_command():
    """The dockerfile HEALTHCHECK's shell command, continuations joined."""
    text = pathlib.Path(REPO_ROOT, "docker", "dockerfile").read_text()
    joined = text.replace("\\\n", " ")
    line = next(l for l in joined.splitlines() if l.startswith("HEALTHCHECK"))
    command = line.split("CMD", 1)[1].strip()
    assert "curl" in command, command
    return command


def _probe_target_port(command, stub_dir, value):
    """Run the real healthcheck with a curl stub; return the port it hit."""
    result = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, timeout=30,
        env={"PORT": value, "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}"})
    assert result.returncode == 0, f"PORT={value!r}: {result.stderr}"
    urls = re.findall(r"http://localhost:([^/]*)/health", result.stdout)
    assert len(urls) == 1, f"PORT={value!r}: curl got {result.stdout!r}"
    return urls[0]


def test_healthcheck_probes_the_port_the_server_binds(tmp_path):
    """Runs the whole HEALTHCHECK, not just its port arithmetic: a curl stub
    records the URL actually requested, so hardcoding a port back into the
    curl line would fail here even though the case/range logic still ran."""
    stub = tmp_path / "curl"
    stub.write_text('#!/bin/sh\nfor a in "$@"; do echo "$a"; done\n')
    stub.chmod(0o755)

    command = _healthcheck_command()
    expected = _int_env_matrix()
    for value, want in zip(PORT_MATRIX, expected):
        got = _probe_target_port(command, str(tmp_path), value)
        assert got.isascii() and got.isdecimal(), (
            f"PORT={value!r}: probe built a non-numeric port {got!r}")
        assert int(got) == want, (
            f"PORT={value!r}: probe hits {got}, server binds {want}")
