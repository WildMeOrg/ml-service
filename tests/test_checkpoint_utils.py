"""Tests for app.utils.checkpoint_utils download hardening.

The downloader runs at service startup on serverless GPU platforms, so it
must not hang forever (timeout), must never expose a partially written file
at the cache path (atomic write), and must not collide when two different
URLs share a basename (cache key).
"""
import os

import pytest

from app.utils import checkpoint_utils
from app.utils.checkpoint_utils import download_checkpoint, get_checkpoint_path


class FakeResponse:
    """Minimal stand-in for requests.Response streaming."""

    def __init__(self, chunks, on_chunk=None):
        self._chunks = chunks
        self._on_chunk = on_chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for chunk in self._chunks:
            if self._on_chunk:
                self._on_chunk()
            yield chunk


def _patch_get(monkeypatch, response_factory, calls):
    def fake_get(url, stream=True, **kwargs):
        calls.append({"url": url, "stream": stream, **kwargs})
        return response_factory(url)

    monkeypatch.setattr(checkpoint_utils.requests, "get", fake_get)


def test_download_passes_timeout_to_requests(monkeypatch, tmp_path):
    calls = []
    _patch_get(monkeypatch, lambda url: FakeResponse([b"data"]), calls)

    download_checkpoint("https://example.org/models/a.pt", cache_dir=str(tmp_path))

    assert calls, "requests.get was never called"
    assert calls[0].get("timeout") == checkpoint_utils.DOWNLOAD_TIMEOUT
    assert checkpoint_utils.DOWNLOAD_TIMEOUT >= (5, 60), (
        "timeout must leave room for multi-hundred-MB weight files"
    )


def test_cache_path_absent_until_download_completes(monkeypatch, tmp_path):
    """A concurrent reader must never see a partial file at the cache path."""
    seen = []

    def snapshot():
        seen.extend(p.name for p in tmp_path.iterdir())

    calls = []
    _patch_get(
        monkeypatch,
        lambda url: FakeResponse([b"chunk1", b"chunk2"], on_chunk=snapshot),
        calls,
    )

    local_path = download_checkpoint(
        "https://example.org/models/b.pt", cache_dir=str(tmp_path)
    )

    final_name = os.path.basename(local_path)
    assert final_name not in seen, (
        "final cache path existed while the download was still streaming"
    )
    with open(local_path, "rb") as f:
        assert f.read() == b"chunk1chunk2"


def test_distinct_urls_with_same_basename_do_not_collide(monkeypatch, tmp_path):
    calls = []
    _patch_get(
        monkeypatch,
        lambda url: FakeResponse([url.encode()]),
        calls,
    )

    path_a = download_checkpoint(
        "https://bucket-a.example.org/models/model.pt", cache_dir=str(tmp_path)
    )
    path_b = download_checkpoint(
        "https://bucket-b.example.org/models/model.pt", cache_dir=str(tmp_path)
    )

    assert path_a != path_b
    with open(path_a, "rb") as f:
        assert b"bucket-a" in f.read()
    with open(path_b, "rb") as f:
        assert b"bucket-b" in f.read()


def test_same_url_reuses_cache(monkeypatch, tmp_path):
    calls = []
    _patch_get(monkeypatch, lambda url: FakeResponse([b"payload"]), calls)

    url = "https://example.org/models/c.pt"
    first = download_checkpoint(url, cache_dir=str(tmp_path))
    second = download_checkpoint(url, cache_dir=str(tmp_path))

    assert first == second
    assert len(calls) == 1, "cached checkpoint must not be re-downloaded"


def test_failed_download_leaves_no_files(monkeypatch, tmp_path):
    class ExplodingResponse(FakeResponse):
        def iter_content(self, chunk_size=8192):
            yield b"partial"
            raise IOError("connection dropped")

    calls = []
    _patch_get(monkeypatch, lambda url: ExplodingResponse([]), calls)

    with pytest.raises(Exception):
        download_checkpoint(
            "https://example.org/models/d.pt", cache_dir=str(tmp_path)
        )

    leftovers = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")]
    assert leftovers == [], "failed download left files behind"


def test_download_enforces_total_deadline(monkeypatch, tmp_path):
    """A server drip-feeding or stalling mid-body must not hold startup
    hostage. The socket read timeout only bounds per-read inactivity, so the
    caller must get control back at the wall-clock deadline even while the
    fetch is blocked inside a read."""
    import threading
    import time as real_time

    monkeypatch.setattr(checkpoint_utils, "DOWNLOAD_TOTAL_DEADLINE", 0.2)
    release = threading.Event()

    class StallingResponse(FakeResponse):
        def iter_content(self, chunk_size=8192):
            yield b"first"
            release.wait(10)  # stall well past the deadline
            raise IOError("connection aborted")

        def close(self):
            release.set()

    calls = []
    _patch_get(monkeypatch, lambda url: StallingResponse([]), calls)

    start = real_time.monotonic()
    with pytest.raises(TimeoutError):
        download_checkpoint(
            "https://example.org/models/slow.pt", cache_dir=str(tmp_path)
        )
    elapsed = real_time.monotonic() - start
    assert elapsed < 5, (
        f"deadline must bound wall-clock time even during a stalled read "
        f"(took {elapsed:.1f}s)"
    )

    # The aborted worker cleans up its partial file once unblocked.
    release.set()
    poll_deadline = real_time.monotonic() + 5
    while real_time.monotonic() < poll_deadline:
        leftovers = [
            p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")
        ]
        if not leftovers:
            break
        real_time.sleep(0.05)
    assert leftovers == [], "deadline abort left files behind"


def test_timeout_during_header_wait_never_writes_cache(monkeypatch, tmp_path):
    """If the deadline expires before the server even returns headers, the
    late-arriving worker must not keep downloading and install a checkpoint
    after the caller already raised TimeoutError."""
    import threading
    import time as real_time

    monkeypatch.setattr(checkpoint_utils, "DOWNLOAD_TOTAL_DEADLINE", 0.2)
    gate = threading.Event()

    def stalling_get(url, stream=True, **kwargs):
        gate.wait(10)  # headers arrive only long after the deadline
        return FakeResponse([b"late-body"])

    monkeypatch.setattr(checkpoint_utils.requests, "get", stalling_get)

    with pytest.raises(TimeoutError):
        download_checkpoint(
            "https://example.org/models/late.pt", cache_dir=str(tmp_path)
        )

    # Release the worker and watch: no cache file may ever materialize.
    gate.set()
    watch_until = real_time.monotonic() + 1.5
    while real_time.monotonic() < watch_until:
        files = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")]
        assert all(f.endswith(".part") for f in files), (
            f"late worker installed a cache file after timeout: {files}"
        )
        real_time.sleep(0.05)
    leftovers = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")]
    assert leftovers == [], f"late worker left files behind: {leftovers}"


def test_concurrent_downloader_waits_and_reuses_result(monkeypatch, tmp_path):
    """If another process holds the download lock, we block, then reuse the
    file it produced instead of downloading again."""
    import fcntl
    import threading

    url = "https://example.org/models/e.pt"
    url_hash = checkpoint_utils.hashlib.sha256(url.encode()).hexdigest()
    local_path = tmp_path / f"{url_hash}_e.pt"
    lock_path = tmp_path / f"{url_hash}_e.pt.lock"

    calls = []
    _patch_get(monkeypatch, lambda u: FakeResponse([b"should-not-run"]), calls)

    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    result = {}

    def run():
        result["path"] = download_checkpoint(url, cache_dir=str(tmp_path))

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=0.3)
    assert t.is_alive(), "downloader should be blocked on the lock"

    # Simulate the lock holder finishing its download, then releasing.
    local_path.write_bytes(b"winner")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    t.join(timeout=5)
    assert not t.is_alive()

    assert result["path"] == str(local_path)
    assert local_path.read_bytes() == b"winner"
    assert calls == [], "must reuse the other process's file, not re-download"


def test_get_checkpoint_path_local_passthrough(tmp_path):
    ckpt = tmp_path / "weights.bin"
    ckpt.write_bytes(b"w")
    assert get_checkpoint_path(str(ckpt)) == str(ckpt)


def test_get_checkpoint_path_missing_local_raises():
    with pytest.raises(FileNotFoundError):
        get_checkpoint_path("/nonexistent/path/weights.bin")


def test_get_checkpoint_path_none_returns_none():
    assert get_checkpoint_path(None) is None
