# Deploying ml-service on on-demand GPU (provider-independent)

The service is stateless and GPU-bound, which makes it a clean fit for
serverless / on-demand GPU platforms: you pay per-second only while a GPU is
processing, the platform autoscales across GPUs during bursts and back down to
a warm baseline (or zero) when idle, and **the platform is the load balancer** —
there is no nginx/HAProxy to run yourself.

The point of this directory is to stay provider-independent: one OCI image,
built from the repo's existing `docker/dockerfile` and serving plain HTTP, runs
unchanged on RunPod, Cloud Run, or a plain VM. Each provider gets only a thin
config file here. Nothing in this directory is imported by the application.

## The portability contract

The image makes no provider-specific assumptions. It is configured entirely
through environment variables (`app/main.py`):

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `8888` | HTTP port. Cloud Run injects this; RunPod/VM set it explicitly (the compose file uses `6050`). |
| `HOST` | `0.0.0.0` | Bind address. |
| `DEVICE` | `cuda` | `cuda` / `cpu` / `mps`. |
| `WORKERS` | `1` | **Keep at 1 per GPU.** Scale with replicas, not workers. |
| `MODEL_BASE` | `/datasets` | Prefix for model weights in `app/model_config.json`. A filesystem path **or** an `https://` object-store prefix. |

`MODEL_BASE` replaces the `${MODEL_BASE}` prefix in the model config's weight
paths at startup (`app/utils/config_loader.py`). Weights can then live in:

- a **mounted volume** — `/datasets` (VM bind mount) or `/runpod-volume/models`
  (RunPod network volume), or
- an **object store** — `MODEL_BASE=https://storage.googleapis.com/your-bucket/models`,
  fetched and cached at boot.

URL weights are fetched and cached by `checkpoint_utils.get_checkpoint_path`.
The URL form is the most portable: identical config everywhere, no volume
wiring, and workers need no cloud credentials when the bucket is public.

**`MODEL_BASE` only moves paths that are written as `${MODEL_BASE}/...`.** The
committed `app/model_config.json` is. Leaving the variable unset (or setting it
to `/datasets`) resolves those paths to exactly the filenames they resolved to
before, so an existing deployment that sets nothing is unaffected — and nothing
sets it today, because the variable is new here. Setting it takes effect from
now on. But
production installs bind-mount their own registry over that file (see the
`_note` at the top of it), and **a mounted registry with literal `/datasets/...`
paths ignores `MODEL_BASE` entirely.** Deploying to a provider therefore means
supplying a registry whose paths carry the prefix, not just setting the env var.
Neither provider config in this directory ships one; that is a per-install
decision about which models to load.

## Two load-balancing knobs

1. **Concurrency per replica = 1.** Send the second concurrent request to
   another GPU replica rather than queueing it behind the first. This is a
   measurement, not a guess: an A/B sweep during phase-0 GPU validation
   (RTX A5000, 2026-08) found concurrency 1 beat 2 on every axis — 16.2 vs
   11.4 req/s, p95 859 vs 987 ms, p99 1.25 vs 3.19 s. Two concurrent CUDA
   streams contend rather than help.

   Caveat on scope: that sweep ran against a slim 2-model profile on the spike
   branch. On `main` the in-process limit is `MAX_CONCURRENT_PREDICTIONS = 2`
   in `app/routers/predict_router.py` — a module constant, not env-driven, and
   not yet re-measured for the full 5-model registry. Setting the *platform*
   concurrency to 1 is the safe default either way; it bounds what reaches the
   semaphore rather than contradicting it.

2. **min / max replicas.** `min = 1` keeps a GPU warm; `max = N` is the burst
   ceiling. `min = 0` gives true scale-to-zero — cheapest, but the first
   request after idle pays the full cold start.

> **Do not raise `WORKERS`.** Multiple uvicorn workers on one GPU each load a
> full copy of every model into the same VRAM, which is an OOM risk that grows
> with the size of the registry. The concurrency sweep above measured requests
> against a single worker and does not by itself establish what extra workers
> would do, so treat the VRAM duplication as the reason. One worker per GPU;
> scale via replicas.

## Cold start and health checks

Startup eagerly loads every model in the registry, so a true cold start is tens
of seconds — the image healthcheck allows a 90s `start-period` for the default
5-model config (`docker/dockerfile`). Phase-0 measured 31.5s on RunPod with
cached image layers, but that was a 2-model profile; budget more for the full
registry. A first pull of the image on a datacenter that has never seen it can
take far longer (17.5 GB took over 2 hours once); layers cache per-datacenter
afterwards.

Uvicorn does not accept connections until startup has finished loading models,
so during that window a probe of any path gets a refused connection rather than
a response. What the three endpoints separate is the state *after* the port
opens:

- **`/health`** — liveness. This is what Grafana and the autoheal container
  watch. **Do not point a load balancer at it.** On a GPU deployment (`DEVICE`
  exactly `cuda`) it shells out to `nvidia-smi` on every call, which is wasteful
  at an edge probe interval across every worker; and a worker holding an empty
  registry still gets `status: healthy` with `models_loaded: 0`, so it never
  fails closed on the case that matters.
- **`/readyz`** — readiness. 503 unless every configured model is loaded, 200
  otherwise, and cheap. This is the probe to gate traffic on.
- **`/ping`** — an alias for `/readyz`, present for one reason: **RunPod's
  load-balancer edge health-probes `GET /ping` unconditionally and ignores the
  documented `HEALTH_CHECK_PATH` setting.** Without it the edge gets a 404,
  every worker reports healthy, and every request fails with
  `400 "timed out waiting for worker"`. Never remove it.

## Per-provider files

- **RunPod** — `runpod/endpoint.json`. A record of the validated phase-0
  LOAD_BALANCER endpoint, not an aspirational template. Read its top comment
  before creating an endpoint; the GPU-pool exclusion in particular cannot be
  set at creation time.
- **Cloud Run** — `cloudrun/service.yaml` (declarative) or `cloudrun/deploy.sh`
  (imperative). NVIDIA L4, `min-instances=1`, `concurrency=1`, `timeout=300`,
  `/readyz` startup probe. The two are equivalent; the script uses
  `--startup-probe` and `--no-gpu-zonal-redundancy`. Written and deployed during
  phase-0, but the throughput numbers above were measured on RunPod.

Both consume the same image. Moving providers rebuilds nothing: apply the other
config file and point `MODEL_BASE` at that environment's model store.

## Local parity check

```bash
# Invoked the way the platforms invoke it (env-driven, one worker, CPU):
PORT=6050 DEVICE=cpu WORKERS=1 MODEL_BASE=/datasets python3 -m app.main
curl -f http://localhost:6050/readyz
```
