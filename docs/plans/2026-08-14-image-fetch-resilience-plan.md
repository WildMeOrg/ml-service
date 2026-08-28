# Image-Fetch Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slow/stalled Wildbook upstreams must not starve inference or blow memory; fetch failures must map to retry-ladder-correct status codes with the URI logged.

**Architecture:** Per the approved spec `docs/plans/2026-08-14-image-fetch-resilience-design.md`. All fetch machinery lives in `app/utils/image_uri.py` (shared httpx client, admission semaphore, streaming fetch, size/pixel caps, error-mapping helper). A pure-ASGI body-cap middleware lives in `app/middleware.py`. The four image routers acquire an admission slot, fetch, then acquire their (unchanged) inference semaphore.

**Tech Stack:** FastAPI 0.109, httpx 0.26 (`MockTransport` for tests), Pillow 10.1, pytest, Python 3.10.

## Global Constraints

- Python 3.10 syntax only (no `X | Y` in isinstance, `asyncio.wait_for` not `asyncio.timeout`).
- All env defaults exactly as in the spec table: connect 10s, read 30s, total deadline 60s, admission 8, admission wait 20s, image cap 52428800, body cap 4194304, pixels 150000000, `--limit-concurrency` 32.
- Status mapping exactly as the spec's table (400 permanent / 502 / 503 / 504 / 413).
- Every fetch failure log line includes `sanitize_uri_for_logging(uri)`.
- **Line endings:** after every Edit/Write to a tracked file run `perl -i -pe 's/\r\n/\n/g' <file>` and verify `grep -cP '\r$' <file>` prints 0 (WSL /mnt/c flips endings; CRLF diffs are unreviewable).
- Run the full suite with `python3 -m pytest tests/ -q` before each commit; new failures in untouched tests are a stop-and-investigate signal.

---

### Task 1: Fetch state, settings, and lifecycle in `image_uri.py`

**Files:**
- Modify: `app/utils/image_uri.py` (append; keep all existing functions)
- Create: `tests/conftest.py`
- Test: `tests/test_image_fetch_lifecycle.py`

**Interfaces:**
- Produces: `load_fetch_settings() -> dict` (keys: `connect_timeout_s`, `read_timeout_s`, `total_deadline_s`, `admission_limit`, `admission_wait_s`, `max_image_bytes`, `max_pixels`); `init_image_fetch(transport=None) -> None` (sync); `shutdown_image_fetch() -> Coroutine`; `reset_fetch_state_for_tests() -> None`; module globals `_client`, `_admission`, `_settings`; `_ensure_initialized() -> None` (lazy init used by later tasks).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_fetch_lifecycle.py
"""Lifecycle and settings tests for the shared image-fetch state."""
import asyncio
import httpx
import pytest

from app.utils import image_uri


def test_load_fetch_settings_defaults():
    s = image_uri.load_fetch_settings()
    assert s["connect_timeout_s"] == 10.0
    assert s["read_timeout_s"] == 30.0
    assert s["total_deadline_s"] == 60.0
    assert s["admission_limit"] == 8
    assert s["admission_wait_s"] == 20.0
    assert s["max_image_bytes"] == 52428800
    assert s["max_pixels"] == 150000000


def test_load_fetch_settings_env_override(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_READ_TIMEOUT_S", "5")
    monkeypatch.setenv("IMAGE_ADMISSION_LIMIT", "2")
    s = image_uri.load_fetch_settings()
    assert s["read_timeout_s"] == 5.0
    assert s["admission_limit"] == 2


def test_init_creates_client_and_admission():
    image_uri.init_image_fetch()
    assert isinstance(image_uri._client, httpx.AsyncClient)
    assert isinstance(image_uri._admission, asyncio.Semaphore)
    # invariant asserted at init: pool max_connections == admission_limit
    assert image_uri._settings["admission_limit"] == 8


def test_init_accepts_mock_transport():
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    image_uri.init_image_fetch(transport=transport)
    assert image_uri._client is not None


def test_reset_clears_state():
    image_uri.init_image_fetch()
    image_uri.reset_fetch_state_for_tests()
    assert image_uri._client is None
    assert image_uri._admission is None
```

```python
# tests/conftest.py
"""Shared fixtures. The fetch state (httpx client + admission semaphore)
binds to the event loop that first uses it; TestClient creates a fresh loop
per instance, so state must be reset between tests to avoid cross-loop
reuse of asyncio primitives."""
import pytest

from app.utils import image_uri


@pytest.fixture(autouse=True)
def _reset_image_fetch_state():
    image_uri.reset_fetch_state_for_tests()
    yield
    image_uri.reset_fetch_state_for_tests()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_image_fetch_lifecycle.py -v`
Expected: FAIL / ERROR with `AttributeError: module 'app.utils.image_uri' has no attribute 'load_fetch_settings'`

- [ ] **Step 3: Implement**

Append to `app/utils/image_uri.py` (and extend the imports at the top to `import asyncio`, `import logging`, `import os`, `from typing import Optional` — keep existing imports). Add `logger = logging.getLogger(__name__)` after the imports.

```python
# --- Shared fetch state -----------------------------------------------------
# One httpx client + one admission semaphore per worker process. Created
# lazily (first request) or eagerly (app startup); asyncio primitives bind
# to the running event loop, so tests reset between cases (see conftest).

_client: Optional["httpx.AsyncClient"] = None
_admission: Optional[asyncio.Semaphore] = None
_settings: Optional[dict] = None


def load_fetch_settings() -> dict:
    return {
        "connect_timeout_s": float(os.getenv("IMAGE_FETCH_CONNECT_TIMEOUT_S", "10")),
        "read_timeout_s": float(os.getenv("IMAGE_FETCH_READ_TIMEOUT_S", "30")),
        "total_deadline_s": float(os.getenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "60")),
        "admission_limit": int(os.getenv("IMAGE_ADMISSION_LIMIT", "8")),
        "admission_wait_s": float(os.getenv("IMAGE_ADMISSION_WAIT_S", "20")),
        "max_image_bytes": int(os.getenv("IMAGE_FETCH_MAX_BYTES", "52428800")),
        "max_pixels": int(os.getenv("IMAGE_MAX_PIXELS", "150000000")),
    }


async def _log_redirect_hop(response: httpx.Response) -> None:
    if response.is_redirect:
        logger.info(
            "Image fetch redirect: %s -> %s",
            response.request.url.host,
            response.headers.get("location", "?"),
        )


def init_image_fetch(transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
    """Create the shared client + admission semaphore. Idempotent."""
    global _client, _admission, _settings
    if _client is not None:
        return
    _settings = load_fetch_settings()
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=_settings["connect_timeout_s"],
            read=_settings["read_timeout_s"],
            write=10.0,
            pool=None,  # admission semaphore keeps requests <= max_connections
        ),
        limits=httpx.Limits(
            max_connections=_settings["admission_limit"],
            max_keepalive_connections=4,
        ),
        follow_redirects=True,
        max_redirects=3,
        event_hooks={"response": [_log_redirect_hop]},
        transport=transport,
    )
    _admission = asyncio.Semaphore(_settings["admission_limit"])


async def shutdown_image_fetch() -> None:
    global _client, _admission, _settings
    if _client is not None:
        await _client.aclose()
    _client = None
    _admission = None
    _settings = None


def reset_fetch_state_for_tests() -> None:
    """Synchronous reset for test isolation (drops, doesn't close, the client)."""
    global _client, _admission, _settings
    _client = None
    _admission = None
    _settings = None


def _ensure_initialized() -> None:
    if _client is None:
        init_image_fetch()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_image_fetch_lifecycle.py -v` — Expected: all PASS.
Then: `python3 -m pytest tests/ -q` — Expected: no new failures (conftest reset must not break existing tests).

- [ ] **Step 5: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/utils/image_uri.py tests/conftest.py tests/test_image_fetch_lifecycle.py
git add app/utils/image_uri.py tests/conftest.py tests/test_image_fetch_lifecycle.py
git commit -m "feat: shared image-fetch client and admission state with env-tunable settings"
```

---

### Task 2: Streaming URL fetch with size cap + total deadline

**Files:**
- Modify: `app/utils/image_uri.py` (replace the URL branch of `resolve_image_uri`; add `_fetch_url_bytes`, `ImageTooLargeError`, `EmptyImageError`)
- Test: `tests/test_image_fetch_url.py`

**Interfaces:**
- Consumes: `_ensure_initialized`, `_client`, `_settings` (Task 1).
- Produces: `ImageTooLargeError(ValueError)`, `EmptyImageError(ValueError)`; `resolve_image_uri(uri) -> bytes` now streams URLs through the shared client, raises `asyncio.TimeoutError` past the total deadline, keeps raising `ValueError`/httpx errors otherwise (contract unchanged for data URIs).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_fetch_url.py
"""URL-branch tests for resolve_image_uri using httpx.MockTransport.

MockTransport cannot simulate timeouts by sleeping; handlers raise
httpx.ReadTimeout directly (per httpx docs)."""
import asyncio
import httpx
import pytest

from app.utils import image_uri


def _init_with(handler):
    image_uri.init_image_fetch(transport=httpx.MockTransport(handler))


def test_url_fetch_returns_bytes():
    _init_with(lambda req: httpx.Response(200, content=b"jpegbytes"))
    result = asyncio.run(image_uri.resolve_image_uri("https://wb.example/img.jpg"))
    assert result == b"jpegbytes"


def test_url_fetch_declared_oversize_rejected_before_body():
    _init_with(lambda req: httpx.Response(
        200, headers={"content-length": str(60 * 1024 * 1024)}, content=b""))
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/big.jpg"))


def test_url_fetch_undeclared_oversize_rejected_midstream(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "10")
    _init_with(lambda req: httpx.Response(200, content=b"x" * 32))
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/liar.jpg"))


def test_url_fetch_empty_body_rejected():
    _init_with(lambda req: httpx.Response(200, content=b""))
    with pytest.raises(image_uri.EmptyImageError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/empty.jpg"))


def test_url_fetch_read_timeout_propagates():
    def handler(req):
        raise httpx.ReadTimeout("stalled", request=req)
    _init_with(handler)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/slow.jpg"))


def test_url_fetch_total_deadline(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "0.05")

    async def run():
        async def slow_drip():
            # slower than the deadline but each read under the read timeout
            yield b"a"
            await asyncio.sleep(0.2)
            yield b"b"
        def handler(req):
            return httpx.Response(200, content=slow_drip())
        image_uri.init_image_fetch(transport=httpx.MockTransport(handler))
        with pytest.raises(asyncio.TimeoutError):
            await image_uri.resolve_image_uri("https://wb.example/drip.jpg")
    asyncio.run(run())


def test_url_fetch_upstream_error_raises_status_error():
    _init_with(lambda req: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/oops.jpg"))


def test_url_fetch_follows_redirect():
    def handler(req):
        if req.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://cdn.example/b"})
        return httpx.Response(200, content=b"moved")
    _init_with(handler)
    result = asyncio.run(image_uri.resolve_image_uri("https://wb.example/a"))
    assert result == b"moved"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_image_fetch_url.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'ImageTooLargeError'` and assertion failures (old code buffers via throwaway client).

- [ ] **Step 3: Implement**

In `app/utils/image_uri.py`: add the exception classes near the top, add `_fetch_url_bytes`, and change `resolve_image_uri`'s URL branch.

```python
class ImageTooLargeError(ValueError):
    """Image exceeds IMAGE_FETCH_MAX_BYTES or IMAGE_MAX_PIXELS."""


class EmptyImageError(ValueError):
    """Upstream returned a successful but empty body."""


async def _fetch_url_bytes(uri: str) -> bytes:
    """Stream a URL through the shared client, enforcing the byte cap."""
    max_bytes = _settings["max_image_bytes"]
    async with _client.stream("GET", uri) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise ImageTooLargeError(
                f"Declared content-length {declared} exceeds cap {max_bytes}")
        chunks = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise ImageTooLargeError(
                    f"Body exceeded cap {max_bytes} bytes mid-stream")
            chunks.append(chunk)
        if size == 0:
            raise EmptyImageError("Upstream returned an empty body")
        return b"".join(chunks)
```

Replace the `elif uri.startswith(('http://', 'https://')):` block of `resolve_image_uri` with:

```python
    elif uri.startswith(('http://', 'https://')):
        _ensure_initialized()
        return await asyncio.wait_for(
            _fetch_url_bytes(uri), timeout=_settings["total_deadline_s"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_image_fetch_url.py tests/test_image_fetch_lifecycle.py -v` — Expected: all PASS.
Then: `python3 -m pytest tests/ -q` — no new failures.

- [ ] **Step 5: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/utils/image_uri.py tests/test_image_fetch_url.py
git add app/utils/image_uri.py tests/test_image_fetch_url.py
git commit -m "feat: streaming image fetch with size cap and total wall-clock deadline"
```

---

### Task 3: Data-URI encoded-length check, local-path safety, pixel-header cap

**Files:**
- Modify: `app/utils/image_uri.py` (`decode_data_uri`, local-path branch of `resolve_image_uri`; add `check_image_header`)
- Test: `tests/test_image_fetch_guards.py`

**Interfaces:**
- Consumes: `_settings`, `_ensure_initialized`, `ImageTooLargeError` (Tasks 1–2).
- Produces: `check_image_header(data: bytes) -> None` (raises `ValueError` on unparseable header, `ImageTooLargeError` over pixel cap); `decode_data_uri` rejects over-cap encoded payloads; local paths are stat-checked (regular file, size) and read in a threadpool.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_fetch_guards.py
"""Guards: encoded-length cap for data URIs, stat-first local files,
header-parsed pixel cap."""
import asyncio
import base64
import io
import os
import struct
import zlib

import pytest
from PIL import Image

from app.utils import image_uri

# Valid 1x1 red PNG
ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _png_with_dimensions(width, height):
    """Craft a PNG whose IHDR declares width x height (header-only parse)."""
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = bytearray(buf.getvalue())
    # IHDR starts at byte 16: 4-byte width, 4-byte height (big-endian)
    data[16:24] = struct.pack(">II", width, height)
    # fix the IHDR CRC (bytes 29-33 cover chunk type+data at 12..29)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))
    return bytes(data)


def test_check_image_header_accepts_small_png():
    image_uri.init_image_fetch()
    image_uri.check_image_header(ONE_PX_PNG)  # must not raise


def test_check_image_header_rejects_bomb():
    image_uri.init_image_fetch()
    bomb = _png_with_dimensions(20000, 20000)  # 400 MP > 150 MP cap
    with pytest.raises(image_uri.ImageTooLargeError):
        image_uri.check_image_header(bomb)


def test_check_image_header_rejects_non_image():
    image_uri.init_image_fetch()
    with pytest.raises(ValueError):
        image_uri.check_image_header(b"\x00\x00\x00\x18ftypmp42 not an image")


def test_data_uri_encoded_length_cap(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "8")
    image_uri.init_image_fetch()
    big = "data:image/png;base64," + base64.b64encode(b"x" * 64).decode()
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri(big))


def test_local_path_rejects_fifo(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        asyncio.run(image_uri.resolve_image_uri(str(fifo)))


def test_local_path_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "4")
    f = tmp_path / "big.jpg"
    f.write_bytes(b"x" * 64)
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri(str(f)))


def test_local_path_reads_regular_file(tmp_path):
    f = tmp_path / "ok.png"
    f.write_bytes(ONE_PX_PNG)
    assert asyncio.run(image_uri.resolve_image_uri(str(f))) == ONE_PX_PNG
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_image_fetch_guards.py -v`
Expected: FAIL — no `check_image_header`; FIFO test hangs are prevented because the current code `open()`s it only after `exists()`, so if this test hangs instead of failing, that confirms the defect — kill and proceed.

- [ ] **Step 3: Implement**

In `app/utils/image_uri.py`: add imports `import io`, `import stat as stat_module`, `from PIL import Image`, `from fastapi.concurrency import run_in_threadpool`. Immediately after the PIL import add:

```python
# PIL's own decompression-bomb guard is replaced by the explicit
# IMAGE_MAX_PIXELS check in check_image_header(), which runs before any
# decode. Bytes never reach a decoder without passing that check.
Image.MAX_IMAGE_PIXELS = None
```

Add:

```python
def check_image_header(data: bytes) -> None:
    """Reject bytes whose image header is unparseable or declares more
    pixels than IMAGE_MAX_PIXELS. Parses the header only — no pixel decode."""
    _ensure_initialized()
    max_pixels = _settings["max_pixels"]
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception as e:
        raise ValueError(f"Unrecognized image format: {e}")
    if width * height > max_pixels:
        raise ImageTooLargeError(
            f"Image dimensions {width}x{height} exceed {max_pixels} pixel cap")
```

In `decode_data_uri`, after the base64 header check and before decoding, add (note `_ensure_initialized()` first):

```python
    _ensure_initialized()
    # 4/3 base64 expansion + header slack: reject before materializing
    if len(encoded) > _settings["max_image_bytes"] * 4 // 3 + 8:
        raise ImageTooLargeError("Encoded data URI exceeds size cap")
```

Replace the local-path `else:` branch of `resolve_image_uri` with:

```python
    else:
        _ensure_initialized()
        file_path = Path(uri)
        try:
            st = os.stat(file_path)
        except OSError:
            raise ValueError(f"File not found: {uri}")
        if not stat_module.S_ISREG(st.st_mode):
            raise ValueError(f"Not a regular file: {uri}")
        if st.st_size > _settings["max_image_bytes"]:
            raise ImageTooLargeError(
                f"File size {st.st_size} exceeds cap {_settings['max_image_bytes']}")
        return await run_in_threadpool(file_path.read_bytes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_image_fetch_guards.py -v` — all PASS.
Then: `python3 -m pytest tests/ -q` — no new failures.

- [ ] **Step 5: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/utils/image_uri.py tests/test_image_fetch_guards.py
git add app/utils/image_uri.py tests/test_image_fetch_guards.py
git commit -m "feat: pixel-header cap, encoded data-URI cap, stat-first local reads"
```

---

### Task 4: Error-mapping helper and bounded admission slot

**Files:**
- Modify: `app/utils/image_uri.py`
- Test: `tests/test_image_fetch_mapping.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: `fetch_image_for_request(uri: str) -> bytes` (raises `fastapi.HTTPException` per the spec's mapping table, logs every failure with sanitized URI + elapsed ms); `admission_slot()` async context manager (403→ no; raises `HTTPException(503)` when the slot wait exceeds `IMAGE_ADMISSION_WAIT_S`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_fetch_mapping.py
"""fetch_image_for_request must map every failure class to the spec's
status code, and admission_slot must 503 on wait timeout."""
import asyncio
import httpx
import pytest
from fastapi import HTTPException

from app.utils import image_uri
from tests.test_image_fetch_guards import ONE_PX_PNG


def _status_for(handler):
    image_uri.init_image_fetch(transport=httpx.MockTransport(handler))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(image_uri.fetch_image_for_request("https://wb.example/x.jpg"))
    return exc.value.status_code


def test_success_returns_bytes():
    image_uri.init_image_fetch(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=ONE_PX_PNG)))
    data = asyncio.run(image_uri.fetch_image_for_request("https://wb.example/ok.png"))
    assert data == ONE_PX_PNG


def test_read_timeout_maps_504():
    def h(req):
        raise httpx.ReadTimeout("stall", request=req)
    assert _status_for(h) == 504


def test_connect_error_maps_502():
    def h(req):
        raise httpx.ConnectError("refused", request=req)
    assert _status_for(h) == 502


def test_upstream_500_maps_502():
    assert _status_for(lambda r: httpx.Response(500)) == 502


def test_upstream_429_maps_502():
    assert _status_for(lambda r: httpx.Response(429)) == 502


def test_upstream_408_maps_502():
    assert _status_for(lambda r: httpx.Response(408)) == 502


@pytest.mark.parametrize("code", [401, 403, 404, 410, 415])
def test_upstream_permanent_4xx_maps_400(code):
    assert _status_for(lambda r, c=code: httpx.Response(c)) == 400


def test_too_many_redirects_maps_400():
    def h(req):
        return httpx.Response(302, headers={"location": str(req.url)})
    assert _status_for(h) == 400


def test_decoding_error_maps_502():
    def h(req):
        raise httpx.DecodingError("bad encoding", request=req)
    assert _status_for(h) == 502


def test_unparseable_image_maps_400():
    assert _status_for(lambda r: httpx.Response(200, content=b"mp4junk" * 4)) == 400


def test_invalid_data_uri_maps_400():
    image_uri.init_image_fetch()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(image_uri.fetch_image_for_request("data:image/png;base64"))
    assert exc.value.status_code == 400


def test_total_deadline_maps_504(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "0.05")

    async def run():
        async def drip():
            yield b"a"
            await asyncio.sleep(0.2)
            yield b"b"
        image_uri.init_image_fetch(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=drip())))
        with pytest.raises(HTTPException) as exc:
            await image_uri.fetch_image_for_request("https://wb.example/drip.jpg")
        assert exc.value.status_code == 504
    asyncio.run(run())


def test_admission_slot_503_on_wait_timeout(monkeypatch):
    monkeypatch.setenv("IMAGE_ADMISSION_LIMIT", "1")
    monkeypatch.setenv("IMAGE_ADMISSION_WAIT_S", "0.05")

    async def run():
        image_uri.init_image_fetch()
        async with image_uri.admission_slot():
            with pytest.raises(HTTPException) as exc:
                async with image_uri.admission_slot():
                    pass
            assert exc.value.status_code == 503
    asyncio.run(run())


def test_admission_slot_releases_on_exception():
    async def run():
        image_uri.init_image_fetch()
        with pytest.raises(RuntimeError):
            async with image_uri.admission_slot():
                raise RuntimeError("boom")
        # slot must be free again
        async with image_uri.admission_slot():
            pass
    asyncio.run(run())


def test_failure_log_contains_sanitized_uri(caplog):
    def h(req):
        raise httpx.ConnectError("refused", request=req)
    image_uri.init_image_fetch(transport=httpx.MockTransport(h))
    with pytest.raises(HTTPException):
        asyncio.run(image_uri.fetch_image_for_request("https://wb.example/y.jpg"))
    assert any("https://wb.example/y.jpg" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_image_fetch_mapping.py -v`
Expected: FAIL — `no attribute 'fetch_image_for_request'`.

- [ ] **Step 3: Implement**

Append to `app/utils/image_uri.py` (add `import time` and `from contextlib import asynccontextmanager`; import `HTTPException` from fastapi):

```python
@asynccontextmanager
async def admission_slot():
    """Bounded admission: one slot per in-flight image request, held from
    fetch start through inference end. 503 (retryable) if the wait exceeds
    IMAGE_ADMISSION_WAIT_S."""
    _ensure_initialized()
    try:
        await asyncio.wait_for(
            _admission.acquire(), timeout=_settings["admission_wait_s"])
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Service saturated: no admission slot available")
    try:
        yield
    finally:
        _admission.release()


def _fail(status_code: int, uri: str, exc: Exception, started: float):
    logger.warning(
        "Image fetch failed (%d, %s, %.0f ms): %s",
        status_code, type(exc).__name__, (time.monotonic() - started) * 1000,
        sanitize_uri_for_logging(uri))
    raise HTTPException(
        status_code=status_code,
        detail=f"Image fetch failed for {sanitize_uri_for_response(uri)}: "
               f"{type(exc).__name__}")


async def fetch_image_for_request(uri: str) -> bytes:
    """Router entry point: resolve + header-check an image URI, mapping
    every failure to the retry-ladder-correct HTTPException."""
    started = time.monotonic()
    try:
        data = await resolve_image_uri(uri)
        check_image_header(data)
        logger.debug(
            "Image fetched: %d bytes in %.0f ms from %s",
            len(data), (time.monotonic() - started) * 1000,
            sanitize_uri_for_logging(uri))
        return data
    except ValueError as e:               # incl. ImageTooLarge/Empty/bad header
        _fail(400, uri, e, started)
    except asyncio.TimeoutError as e:     # total deadline
        _fail(504, uri, e, started)
    except httpx.TimeoutException as e:   # connect/read/write/pool timeout
        _fail(504, uri, e, started)
    except httpx.TooManyRedirects as e:
        _fail(400, uri, e, started)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (408, 429) or code >= 500:
            _fail(502, uri, e, started)
        _fail(400, uri, e, started)
    except httpx.UnsupportedProtocol as e:
        _fail(400, uri, e, started)
    except httpx.InvalidURL as e:
        _fail(400, uri, e, started)
    except httpx.RequestError as e:       # DNS, refused, reset, TLS, decoding
        _fail(502, uri, e, started)
```

Exception-order constraint (do not reorder): `TimeoutException`, `TooManyRedirects`, `UnsupportedProtocol` are all `RequestError` subclasses — the generic `RequestError` catch must stay last. `InvalidURL` is NOT a `RequestError` in httpx 0.26 but is caught explicitly for safety.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_image_fetch_mapping.py -v` — all PASS.
Then: `python3 -m pytest tests/ -q` — no new failures.

- [ ] **Step 5: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/utils/image_uri.py tests/test_image_fetch_mapping.py
git add app/utils/image_uri.py tests/test_image_fetch_mapping.py
git commit -m "feat: retry-ladder error mapping and bounded admission slots"
```

---

### Task 5: Body-cap ASGI middleware

**Files:**
- Create: `app/middleware.py`
- Test: `tests/test_body_limit_middleware.py`

**Interfaces:**
- Produces: `BodyLimitMiddleware(app, max_bytes: int)` — pure ASGI; 413 on over-cap `Content-Length` or over-cap streamed body. Wired into the app in Task 7.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_body_limit_middleware.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import BodyLimitMiddleware


def _make_app(max_bytes=64):
    app = FastAPI()
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(payload: dict):
        return {"n": len(str(payload))}
    return TestClient(app)


def test_small_body_passes():
    client = _make_app()
    r = client.post("/echo", json={"a": 1})
    assert r.status_code == 200


def test_oversize_content_length_413():
    client = _make_app(max_bytes=8)
    r = client.post("/echo", json={"key": "value-far-over-eight-bytes"})
    assert r.status_code == 413


def test_oversize_chunked_body_413():
    client = _make_app(max_bytes=8)

    def gen():
        yield b'{"key": "'
        yield b"x" * 32
        yield b'"}'
    r = client.post("/echo", content=gen(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_non_http_scope_passthrough():
    # middleware must not touch lifespan scopes: app startup succeeding
    # under TestClient's context manager proves it
    with _make_app() as client:
        assert client.post("/echo", json={"a": 1}).status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_body_limit_middleware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.middleware'`.

- [ ] **Step 3: Implement**

```python
# app/middleware.py
"""Pure-ASGI request-body cap.

Bounds per-request parse-time memory: FastAPI/Pydantic materializes the
whole body before route handlers run, so the cap must sit below them.
Aggregate concurrency is bounded separately by uvicorn --limit-concurrency."""


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)
            # else: response already in flight; connection ends here

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"Request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_body_limit_middleware.py -v` — all PASS.
Then: `python3 -m pytest tests/ -q` — no new failures.

- [ ] **Step 5: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/middleware.py tests/test_body_limit_middleware.py
git add app/middleware.py tests/test_body_limit_middleware.py
git commit -m "feat: ASGI body-size cap middleware (413 over MAX_REQUEST_BODY_BYTES)"
```

---

### Task 6: Router integration (pipeline, predict, classify, extract)

**Files:**
- Modify: `app/routers/pipeline_router.py`, `app/routers/predict_router.py`, `app/routers/classify_router.py`, `app/routers/extract_router.py`
- Modify: `tests/test_pipeline_router_classifier.py`, `tests/test_pipeline_router_theta.py`, `tests/test_pipeline_router_sentinel_species.py` (replace truncated data-URI payloads)
- Test: `tests/test_router_fetch_isolation.py`

**Interfaces:**
- Consumes: `fetch_image_for_request`, `admission_slot` (Task 4).
- Produces: request flow `admission_slot → validate → fetch → inference semaphore → inference` in all four routers.

**The identical restructuring, applied to each router** (shown for pipeline; predict/classify/extract are the same moves on their own semaphore and validation blocks):

1. Change the import line to `from app.utils.image_uri import fetch_image_for_request, admission_slot, sanitize_uri_for_response, sanitize_uri_for_logging` (keep only names the file still uses; drop `resolve_image_uri`).
2. Replace `async with pipeline_semaphore:` (the whole-handler guard) with `async with admission_slot():`.
3. Replace the fetch block

```python
            try:
                image_bytes = await resolve_image_uri(pipeline_request.image_uri)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
```

with

```python
            image_bytes = await fetch_image_for_request(pipeline_request.image_uri)
```

(In classify/extract also delete the dead `with open(file_path, "rb") as f:` remnant lines directly below the old block.)

4. Immediately after the fetch line, open the inference guard and indent everything from the first model call to the `return` one level into it:

```python
            async with pipeline_semaphore:
                ...prediction / classification / extraction / response build...
```

The inference semaphore now wraps only model work; validation and fetch run before it. The `try/except` skeleton of each handler stays where it is (inside `admission_slot()`), so `HTTPException`s from the helper pass through the existing `except HTTPException: raise`.

5. In `pipeline_router.py` delete the now-dead upstream-download handler (the fetch helper maps those):

```python
        except httpx.HTTPStatusError as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error downloading image: {str(e)}"
            )
```

Remove the same `except httpx.HTTPStatusError` block from classify/extract/predict if present, and drop each file's `import httpx` if nothing else in the file uses it.

- [ ] **Step 1: Update existing test payloads (they will otherwise fail the new header check)**

In `tests/test_pipeline_router_classifier.py`, `tests/test_pipeline_router_theta.py`, `tests/test_pipeline_router_sentinel_species.py`, replace every truncated payload — both `"data:image/png;base64,iVBORw0KGgo="` and `"data:image/jpeg;base64,/9j/4AAQSkZJRg=="` — with this valid 1×1 PNG data URI (define once per file as `VALID_PNG_DATA_URI` and reference it):

```python
VALID_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
```

- [ ] **Step 2: Write the failing isolation test**

```python
# tests/test_router_fetch_isolation.py
"""A request that fails at fetch must never touch the inference semaphore,
and mapped statuses must pass through the router unchanged."""
import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.routers import pipeline_router
from app.utils import image_uri


class SpySemaphore:
    def __init__(self):
        self.acquired = False

    async def __aenter__(self):
        self.acquired = True

    async def __aexit__(self, *args):
        return False


def _client(monkeypatch, spy):
    monkeypatch.setattr(pipeline_router, "pipeline_semaphore", spy)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    handler = MagicMock()
    handler.get_model.return_value = MagicMock()
    handler.get_model_info.return_value = {"config": {}}
    handler.list_models.return_value = {}
    app.state.model_handler = handler
    return TestClient(app)


PAYLOAD = {
    "predict_model_id": "p", "classify_model_id": "c",
    "extract_model_id": "e", "image_uri": "https://wb.example/img.jpg",
}


@pytest.mark.parametrize("code", [400, 502, 503, 504])
def test_fetch_failure_status_passes_through_and_skips_inference(
        monkeypatch, code):
    spy = SpySemaphore()

    async def failing_fetch(uri):
        raise HTTPException(status_code=code, detail="mapped upstream failure")
    monkeypatch.setattr(pipeline_router, "fetch_image_for_request", failing_fetch)
    # model-type validation must pass so we reach the fetch
    monkeypatch.setattr(pipeline_router, "isinstance",
                        lambda obj, cls: True, raising=False)

    client = _client(monkeypatch, spy)
    r = client.post("/pipeline/", json=PAYLOAD)
    assert r.status_code == code
    assert spy.acquired is False
```

(If patching `isinstance` proves brittle, instead build the handler mocks with `MagicMock(spec=...)` real model classes exactly as `tests/test_pipeline_router_classifier.py` does — copy its `_make_app_with_models` and pass the spy semaphore.)

- [ ] **Step 3: Run tests to verify current state fails**

Run: `python3 -m pytest tests/test_router_fetch_isolation.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'fetch_image_for_request'` on the router module (not yet imported there).

- [ ] **Step 4: Apply the restructuring to all four routers** (moves 1–5 above, per router)

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: all PASS — including the updated payload tests and the isolation test. Investigate any failure before proceeding; the most likely causes are (a) a payload still using a truncated data URI, (b) an indentation slip while inserting the inference guard.

- [ ] **Step 6: Normalize line endings and commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/routers/pipeline_router.py app/routers/predict_router.py \
  app/routers/classify_router.py app/routers/extract_router.py \
  tests/test_pipeline_router_classifier.py tests/test_pipeline_router_theta.py \
  tests/test_pipeline_router_sentinel_species.py tests/test_router_fetch_isolation.py
git add -A app/routers tests
git commit -m "feat: admission-gated fetch decoupled from inference semaphores in all image routers"
```

---

### Task 7: App wiring, server concurrency limit, docs

**Files:**
- Modify: `app/main.py`
- Modify: `docker/docker-compose.prod.yml` (comment block only)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `init_image_fetch`, `shutdown_image_fetch` (Task 1), `BodyLimitMiddleware` (Task 5).

- [ ] **Step 1: Wire startup/shutdown and middleware in `app/main.py`**

After `app = FastAPI()` add:

```python
from app.middleware import BodyLimitMiddleware
from app.utils import image_uri

MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "4194304"))
app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
```

Inside the existing `startup_event()` (first line of the body):

```python
    image_uri.init_image_fetch()
```

Inside the existing `shutdown_event()` (first line of the body):

```python
    await image_uri.shutdown_image_fetch()
```

Add the CLI arg next to the existing ones and pass it to uvicorn:

```python
parser.add_argument('--limit-concurrency', type=int, default=32,
                   help='Max concurrent connections per worker; excess get 503 '
                        'before their bodies are read (bounds parse-time memory)')
```

```python
    uvicorn.run("app.main:app", host=args.host, port=args.port,
               reload=args.reload, workers=args.workers,
               limit_concurrency=args.limit_concurrency)
```

- [ ] **Step 2: Verify the app boots and the suite passes**

Run: `python3 -c "import app.main" && python3 -m pytest tests/ -q`
Expected: import succeeds (argparse tolerates no args), all tests PASS.

- [ ] **Step 3: Document the knobs in `docker/docker-compose.prod.yml`**

Add directly above the `command:` line (comments only — defaults are in code):

```yaml
    # Image-fetch resilience knobs (defaults in code; override via environment:)
    #   IMAGE_FETCH_CONNECT_TIMEOUT_S=10  IMAGE_FETCH_READ_TIMEOUT_S=30
    #   IMAGE_FETCH_TOTAL_DEADLINE_S=60   IMAGE_ADMISSION_LIMIT=8
    #   IMAGE_ADMISSION_WAIT_S=20         IMAGE_FETCH_MAX_BYTES=52428800
    #   IMAGE_MAX_PIXELS=150000000        MAX_REQUEST_BODY_BYTES=4194304
    # Raise MAX_REQUEST_BODY_BYTES only for installations POSTing data URIs.
    # Recommended nginx back-stop in front of this service: client_max_body_size.
    # --limit-concurrency 32 (in-code default) 503s excess connections per worker.
```

- [ ] **Step 4: Update `CHANGELOG.md`**

Add under a new `## [Unreleased]`-style entry at the top, matching the file's existing format:

```markdown
- Image-fetch resilience (design: docs/plans/2026-08-14-image-fetch-resilience-design.md):
  image downloads now use a shared client with explicit timeouts and a 60s total
  deadline, run outside the inference semaphores behind a bounded admission gate,
  stream with a 50 MB cap and 150 MP header check, and map failures to
  retry-ladder-correct statuses (504/502 retryable, 400 permanent, 503 saturated,
  413 oversized body). Fixes the GiraffeSpotter timeout→500 retry storm
  (2026-08-14) and removes slow-Wildbook head-of-line blocking.
```

- [ ] **Step 5: Normalize line endings, run everything, commit**

```bash
perl -i -pe 's/\r\n/\n/g' app/main.py docker/docker-compose.prod.yml CHANGELOG.md
python3 -m pytest tests/ -q
git add app/main.py docker/docker-compose.prod.yml CHANGELOG.md
git commit -m "feat: wire fetch lifecycle, body-cap middleware, and per-worker connection limit"
```

---

## Verification (whole plan)

- [ ] `python3 -m pytest tests/ -q` — entire suite green.
- [ ] `git diff --ignore-cr-at-eol --stat main...HEAD` — diff matches real changes (no CRLF noise).
- [ ] Manual smoke (optional, needs models): boot with `python3 -m app.main --device cpu`, POST `/pipeline/` with an unreachable `image_uri` (e.g. `https://10.255.255.1/x.jpg`) → expect 504 within ~70s, not 500, and a WARNING log naming the URI.
