# Image-Fetch Resilience Design

**Date:** 2026-08-14
**Status:** Draft — revised after Codex adversarial review (rounds 1–3); pending user review
**Motivation:** Production incident 2026-08-14: GiraffeSpotter stalled serving media,
ml-service's image fetch hit httpx's unconfigured 5s default read timeout, the
`httpx.ReadTimeout` escaped as a 500, and Wildbook's queue re-dispatched the job
every ~6s (retry storm). Because the fetch runs inside the inference semaphore,
each stalled fetch also held one of two per-worker inference slots, throttling
every other installation sharing the service.

## Goals

1. A slow or stalled Wildbook must not consume inference slots, and must not
   degrade service for other installations.
2. Per-worker memory held by in-flight images is bounded by configuration, not
   by upstream behavior.
3. Image-fetch failures surface with HTTP semantics that drive Wildbook's retry
   ladder correctly (retry transient failures with back-off; permanently drop
   hopeless jobs).
4. Every fetch failure logs the (sanitized) `image_uri` so incidents are
   attributable to an asset and host.
5. Timeouts and concurrency are tunable in production via environment variables.

## Non-goals (with rationale)

- **No push-model migration** (Wildbook POSTing image bytes). Pull stays
  primary; base64 data URIs already work as a per-installation escape hatch.
- **No per-host circuit breaker, no internal fetch retries.** Wildbook's
  `MlServiceClient` failure ladder already retries 5xx/timeouts with back-off;
  ml-service retrying on top multiplies load on a struggling Wildbook.
- **No per-origin admission fairness.** Wildbook's FileQueue dispatches IA jobs
  serially (~1 outstanding job per installation), so a single stalled
  installation occupies only 1–2 admission slots of 8, and cannot starve the
  others. **Assumption made explicit:** if Wildbook dispatch ever becomes
  parallel per installation, add per-origin caps to the admission semaphore —
  that is the designed extension point.
- **No local-path root allowlist** (deferred hardening follow-up); local-path
  resolution otherwise gains safety checks (§2).

## Context (verified facts)

- All four routers (`pipeline`, `predict`, `classify`, `extract`) call
  `resolve_image_uri()` **inside** their `asyncio.Semaphore(2)` inference
  guards, and catch only `ValueError` at the call site.
- `resolve_image_uri()` creates a throwaway `httpx.AsyncClient()` per request
  with default timeouts (5s), no redirect following, and full-body buffering.
- Prod runs 4 uvicorn workers; all semaphores/clients are per-worker.
- Wildbook sends `MediaAsset.webURL()` and waits connect 30s / read 120s. Its
  failure ladder: timeout/408 → retry (no back-off increment); 429/5xx →
  retry with back-off increment; other 4xx → non-retryable, job dropped.
- httpx semantics that shape this design: `read` timeout is a **per-read
  inactivity** timeout, not a total deadline; non-streaming requests buffer the
  entire body before returning; `TooManyRedirects` and `DecodingError` are
  `RequestError`s that are **not** `TransportError`s.

## Design

### 1. Shared fetch client, owned by the app lifespan

One `httpx.AsyncClient` per worker, created in the FastAPI lifespan (post
worker start, so no fork/loop hazards) and closed on shutdown. A small module
(`app/utils/image_fetch.py` or extension of `image_uri.py`) holds it behind an
accessor; tests replace it (and the semaphore, §2) per test via a fixture.

Client configuration:

- `timeout=httpx.Timeout(connect=CONNECT_T, read=READ_T, write=10.0, pool=None)`
  (`pool=None` is safe because the admission semaphore keeps concurrent
  requests ≤ `max_connections`; the invariant is asserted at startup).
- `limits=httpx.Limits(max_connections=ADMISSION_LIMIT, max_keepalive_connections=4)`
- `follow_redirects=True`, `max_redirects=3`. Cross-host redirects are allowed
  (S3/CDN-backed asset stores legitimately redirect off-host) but each hop is
  logged. **Explicit behavior change:** `resolve_image_uri()` today returns the
  3xx body; after this change it follows up to 3 hops. Documented, tested.

Environment variables (read once at startup):

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE_FETCH_CONNECT_TIMEOUT_S` | `10` | TCP/TLS connect budget per attempt |
| `IMAGE_FETCH_READ_TIMEOUT_S` | `30` | per-read inactivity budget |
| `IMAGE_FETCH_TOTAL_DEADLINE_S` | `60` | hard wall-clock cap on the whole fetch (all redirects included) |
| `IMAGE_ADMISSION_LIMIT` | `8` | in-flight requests per worker (fetch → inference end) |
| `IMAGE_ADMISSION_WAIT_S` | `20` | max wait for an admission slot before returning 503 |
| `IMAGE_FETCH_MAX_BYTES` | `52428800` (50 MB) | image size cap (streamed, data URI, and local file) |
| `MAX_REQUEST_BODY_BYTES` | `4194304` (4 MB) | ASGI body cap; raise only on installations that POST data URIs |
| `IMAGE_MAX_PIXELS` | `150000000` (150 MP) | decoded-dimension cap, checked from the image header before any decode |
| uvicorn `--limit-concurrency` | `32` | server-level connection cap per worker; excess connections get 503 before their bodies are read |

Budget check: total fetch deadline (60s) + inference must normally fit inside
Wildbook's 120s read timeout; overrun degrades to Wildbook's own timeout,
classified retryable-without-increment. Acceptable.

### 2. Admission semaphore: one bound from fetch start to inference end

`asyncio.Semaphore(IMAGE_ADMISSION_LIMIT)`, created in lifespan alongside the
client (avoids loop-binding surprises), acquired at the top of each router
handler and held until the response is built — covering fetch, the wait for an
inference slot, and inference itself. The inner inference semaphores (2/worker)
are unchanged.

The acquire itself is bounded: `asyncio.wait_for(admission.acquire(),
IMAGE_ADMISSION_WAIT_S)`; on timeout the request returns **503** (retryable
with back-off on the Wildbook side). Without this bound, a worker whose eight
slots are held by slow downloads would queue requests until Wildbook's own
120s timeout fires — an unmapped, uninformative failure.

Why one bound instead of a fetch-only semaphore: releasing after download
would let completed multi-MB images pile up unboundedly while waiting for the
2 inference slots. Holding admission through inference bounds worst-case
per-worker image memory at roughly `2 × ADMISSION_LIMIT × IMAGE_FETCH_MAX_BYTES`
— the factor 2 is honest accounting for the transient during `b"".join(chunks)`
(chunk list + joined copy coexist briefly), and downstream decode (cv2/PIL)
adds its own decoded-pixel allocation on the 2 inference slots. With defaults:
2 × 8 × 50 MB = 800 MB ceiling per worker for **compressed bytes**, against
typical real images of ≤15 MB. Decoded-pixel memory is bounded separately:
before any bytes reach a model, the fetch helper reads the image header via
PIL's lazy `Image.open` (parses dimensions without decoding pixel data) and
rejects images whose `width × height > IMAGE_MAX_PIXELS` (default 150 MP,
generous for camera/drone imagery) → 400 non-retryable — a decompression-bomb
guard the current code lacks. cv2's `OPENCV_IO_MAX_IMAGE_PIXELS` env var is
documented as a backstop for any decode path that bypasses the helper. A
spooled-file handoff was considered and rejected: model decode needs
contiguous bytes in RAM anyway, so spooling removes only the transient join
copy at the cost of a changed downstream interface. The isolation property is
unchanged: a stalled fetch holds an admission slot but **never** an inference
slot, so inference stays saturated with work from healthy installations.

The admission semaphore applies to **all** URI types (URL, data URI, local
path) — queued data-URI requests hold memory too.

**Data URIs.** FastAPI/Pydantic materializes the request body before any
handler code runs, so handler-level checks cannot bound parse-time memory.
Two controls close this:

1. A small ASGI middleware enforces `MAX_REQUEST_BODY_BYTES` (default 4 MB —
   ample for URL-based payloads) → **413** on violation. It rejects on the
   `Content-Length` header when present and counts streamed bytes for chunked
   bodies. Installations that POST data URIs raise the env var deliberately.
2. In the handler, data URIs are rejected by **encoded length**
   (`len(uri) > MAX_BYTES × 4/3 + header slack` → 400) *before* base64
   decoding, so the decoder never materializes an over-cap payload.

A proxy-level cap (nginx `client_max_body_size`) remains a recommended
back-stop, documented in the deploy README, but the middleware makes the bound
enforced rather than advisory. Aggregate parse-time memory (many concurrent
bodies, each under the cap) is bounded server-side with uvicorn's native
`--limit-concurrency 32` in the prod compose command — excess connections
receive 503 before their bodies are read — giving an enforced pre-handler
bound of `32 × MAX_REQUEST_BODY_BYTES = 128 MB` per worker. No custom ASGI
admission gate is built; the native limit is sufficient and simpler.

**Local paths** (dev/test usage; Wildbook never sends them): resolve via
`os.stat` first — regular files only (reject FIFOs/devices, which could block
indefinitely) and `st_size > IMAGE_FETCH_MAX_BYTES` → 400 **before** reading;
then read in a threadpool (`run_in_threadpool`) so the event loop is never
blocked by disk I/O. Restricting paths to an allowlisted root is a separate
hardening follow-up, noted but out of scope here.

### 3. Streaming fetch with size cap and total deadline

The URL branch of `resolve_image_uri()` becomes:

```python
async def _fetch_url(uri: str) -> bytes:
    async with client.stream("GET", uri) as response:
        response.raise_for_status()
        declared = response.headers.get("content-length")
        if declared and int(declared) > MAX_BYTES:
            raise ImageTooLarge(...)
        chunks, size = [], 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_BYTES:
                raise ImageTooLarge(...)   # closes connection via context exit
            chunks.append(chunk)
        return b"".join(chunks)
```

wrapped in `asyncio.wait_for(_fetch_url(uri), IMAGE_FETCH_TOTAL_DEADLINE_S)`
— the total deadline is what defeats slow-drip peers and redirect-chain
budget multiplication; the per-read timeout just fails obvious stalls faster.
Cancellation must release the admission permit (guaranteed by `async with` /
`finally` structure) — covered by a dedicated test.

The cap check counts **decoded** bytes (post `Content-Encoding`), which is the
memory actually held.

### 4. Error mapping

New `async def fetch_image_for_request(uri: str) -> bytes` — the single entry
point for routers. Maps failures to `HTTPException`, logging each with
`sanitize_uri_for_logging(uri)`, elapsed time, and exception class:

| Failure | Response | Wildbook's reaction |
|---|---|---|
| request body over `MAX_REQUEST_BODY_BYTES` (middleware) | 413 | drop job |
| admission wait exceeded `IMAGE_ADMISSION_WAIT_S` | 503 | retry w/ back-off |
| malformed/unsupported URL, invalid data URI, missing/irregular local file | 400 | drop job |
| oversized (declared, streamed, encoded data URI, or local `st_size`) | 400 | drop job |
| decoded dimensions over `IMAGE_MAX_PIXELS`, or unparseable image header | 400 | drop job |
| empty body (0 bytes) | 400 | drop job |
| redirect loop / too many redirects (`TooManyRedirects`) | 400 | drop job |
| upstream 4xx **except 408, 429** (auth, gone, forbidden…) | 400 — permanent for an unchanged URI | drop job |
| upstream 408, 429, any 5xx | 502 | retry w/ back-off |
| total deadline exceeded (`asyncio.TimeoutError`) or httpx `TimeoutException` | 504 | retry w/ back-off |
| `DecodingError` (bad content-encoding) | 502 | retry w/ back-off |
| any other `httpx.RequestError` (DNS, refused, reset, TLS…) | 502 | retry w/ back-off |

Rationale for upstream-4xx → 400: retrying an identical URI against a server
that answered 401/403/404/415… only adds load; Wildbook logs the drop and the
job is inspectable. 408/429 are the transient exceptions.

`resolve_image_uri()` keeps raising raw exceptions (`ValueError` / httpx
errors); only the mapping helper knows HTTP. Non-router callers see two
intentional changes: redirect following (§1) and rejection of
irregular/oversized local files (§2).

### 5. Routers: admission outside, fetch before inference

All four routers restructure identically:

```
POST /… → acquire admission semaphore (held to end of request)
        →   validate models           (registry reads only)
        →   fetch_image_for_request() (streaming, deadline-capped)
        →   acquire inference semaphore
        →     inference
        → response
```

In `pipeline_router` the model validation moves out of the *inference*
semaphore — safe, it only reads the in-memory model registry — while running
inside the admission slot, per the flow above.

Validation runs inside the admission slot; under full saturation a malformed
request waits for a slot before failing — accepted for implementation
simplicity.

### 6. Observability

- Every fetch failure: one WARNING with mapped status, exception class,
  elapsed ms, sanitized URI.
- Every fetch success: DEBUG with elapsed ms and byte count (early signal of a
  degrading Wildbook).
- Every redirect hop: INFO with from-host → to-host.

## Testing

Unit tests (`tests/test_image_fetch.py`) with `httpx.MockTransport` injected
through the lifespan/fixture hook; the fixture replaces **both** the client
and the admission semaphore on the test's event loop:

1. Mapping matrix: handler raises `httpx.ReadTimeout` directly (MockTransport
   does not simulate timeouts by sleeping) → 504; `ConnectError` → 502;
   upstream 500/429/408 → 502; upstream 404/403/401 → 400; `TooManyRedirects`
   → 400; `DecodingError` → 502; empty body → 400.
2. Size cap: declared oversize → 400 before body read; undeclared oversize →
   400 mid-stream; oversized data URI rejected on encoded length before decode
   → 400; local FIFO/oversized file → 400 without reading.
2a. Body cap middleware: over-cap `Content-Length` → 413; over-cap chunked
   body → 413. Admission wait timeout → 503 (semaphore pre-drained in test).
2b. Pixel cap: valid JPEG header declaring > `IMAGE_MAX_PIXELS` → 400 without
   full decode; bytes with no parseable image header (e.g. mp4) → 400.
3. Total deadline: slow-drip handler (async generator yielding with delays) →
   504 at the deadline, admission permit released after cancellation.
4. Redirects: followed to 200 across hosts (≤3 hops); 4th hop → 400.
5. Router level (extend existing per-router tests): each mapped
   `HTTPException` passes through; a request failing at fetch never touches
   the inference semaphore (instrumented semaphore).

## Rollout

1. Land on a branch; full test suite.
2. Deploy to prod — all installations benefit; no Wildbook change required.
3. Watch GiraffeSpotter: stalls should surface as 504s with the offending URI
   logged; other installations' latency unaffected.
4. Wildbook-side note (separate track): GiraffeSpotter's ~6s hot retry loop
   suggests a build predating the current `MlServiceClient` failure ladder;
   upgrading converts storms into back-off retries.

## Relationship to existing work

`fix/reject-undecodable-image-4xx` (undecodable/corrupt media → 4xx) is
complementary: that branch handles bytes that arrive but can't decode; this
design handles bytes that never arrive. The empty-body → 400 rule here closes
the gap between them, and the header-based pixel cap (§2) additionally rejects
non-image payloads (e.g. video files) at fetch time — overlapping that
branch's goal from the other side; the two changes remain independent and
compatible.

## Review log

- **Codex round 1:** 2 Critical, 5 Major, 1 Minor. Adopted: streaming size cap
  (was post-read), total wall-clock deadline (was per-read only), admission
  semaphore held through inference (was fetch-only), upstream-4xx → 400 (was
  404/410 only), broadened `RequestError` mapping, lifespan-owned
  client+semaphore, explicit redirect policy. Rejected: per-origin fairness
  caps — Wildbook FileQueue dispatch is serial per installation (≤1–2
  concurrent jobs each), so one stalled host cannot exhaust 8 admission slots;
  documented as an assumption with a named extension point instead.
- **Codex round 2:** 1 Critical, 3 Major; core structure (admission ordering,
  streaming, deadline, mapping, redirect policy) endorsed; per-origin
  rejection accepted conditional on the serial-dispatch contract. Adopted:
  ASGI body-cap middleware + encoded-length data-URI rejection (was
  advisory proxy note), bounded admission wait → 503 (was unbounded), honest
  2× peak-memory accounting with cap lowered to 50 MB (spooled-file handoff
  rejected — decode needs contiguous RAM regardless), local-path handling
  specified (stat-first, regular files only, threadpool read; root allowlist
  deferred as hardening follow-up). Rejected: admission-before-body-parse —
  disproportionate given the 4 MB default body cap makes parse-time exposure
  negligible.
- **Codex round 3:** 1 Critical, 1 Major. Adopted (both, via the proportionate
  variant): aggregate parse-time memory bounded with uvicorn's native
  `--limit-concurrency 32` (Codex's own alternative to a custom ASGI admission
  gate); decoded-pixel bound via header-parsed `IMAGE_MAX_PIXELS` check → 400
  before any decode, with `OPENCV_IO_MAX_IMAGE_PIXELS` as documented backstop.
