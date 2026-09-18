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

`MODEL_BASE` is expanded into the model config at startup
(`app/utils/config_loader.py`). Because every model loader now resolves its
path through `checkpoint_utils.get_checkpoint_path`, weights can live in:

- a **mounted volume** — `/datasets` (VM bind mount) or `/runpod-volume/models`
  (RunPod network volume), or
- an **object store** — `MODEL_BASE=https://storage.googleapis.com/your-bucket/models`,
  fetched and cached at boot.

The URL form is the most portable: identical config everywhere, no volume
wiring, and workers need no cloud credentials when the bucket is public. The
default (`/datasets`) is what the committed config and the production compose
file already use, so expansion is a no-op for existing deployments.

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
> full copy of every model into the same VRAM (OOM risk) while the GPU still
> executes serially, so there is no throughput gain. One worker per GPU;
> scale via replicas.

## Cold start and health checks

Startup eagerly loads every model in the registry, so a true cold start is tens
of seconds — the image healthcheck allows a 90s `start-period` for the default
5-model config (`docker/dockerfile`). Phase-0 measured 31.5s on RunPod with
cached image layers, but that was a 2-model profile; budget more for the full
registry. A first pull of the image on a datacenter that has never seen it can
take far longer (17.5 GB took over 2 hours once); layers cache per-datacenter
afterwards.

Three probes, and the difference matters:

- **`/health`** — liveness. Reports `degraded` rather than failing while models
  are still loading, and runs GPU/torch checks. This is what Grafana and the
  autoheal container watch. **Do not** route load-balancer traffic on it: it
  returns 200 before the service can serve a request.
- **`/readyz`** — readiness. 503 until every configured model is loaded, 200
  after. This is the probe a load balancer should gate traffic on.
- **`/ping`** — an alias for `/readyz`, present for one reason: **RunPod's
  load-balancer edge health-probes `GET /ping` unconditionally and ignores the
  documented `HEALTH_CHECK_PATH` setting.** Without it every worker reports
  healthy while every request fails with `400 "timed out waiting for worker"`.
  Never remove it.

## Per-provider files

- **RunPod** — `runpod/endpoint.json`. A record of the validated phase-0
  LOAD_BALANCER endpoint, not an aspirational template. Read its top comment
  before creating an endpoint; the GPU-pool exclusion in particular cannot be
  set at creation time.
- **Cloud Run** — `cloudrun/service.yaml` (declarative) or `cloudrun/deploy.sh`
  (imperative). NVIDIA L4, `min-instances=1`, `concurrency=1`, `timeout=300`,
  startup probe on `/readyz`. Written and deployed during phase-0, but the
  throughput numbers above were measured on RunPod, not here.

Both consume the same image. Moving providers rebuilds nothing: apply the other
config file and point `MODEL_BASE` at that environment's model store.

## Local parity check

```bash
# Invoked the way the platforms invoke it (env-driven, one worker, CPU):
PORT=6050 DEVICE=cpu WORKERS=1 MODEL_BASE=/datasets python3 -m app.main
curl -f http://localhost:6050/readyz
```
