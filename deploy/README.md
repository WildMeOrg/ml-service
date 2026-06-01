# Deploying the ML detector service on on-demand GPU (provider-independent)

This service is **stateless** and GPU-bound, which makes it a clean fit for
serverless / on-demand GPU platforms: you pay per-second only while a GPU is
actually processing, and the platform autoscales across many GPUs during bursts
and back down to a warm baseline (or zero) when idle. **The platform is your
load balancer** — you do not run nginx/HAProxy yourself.

The goal of this directory is to stay **provider-independent**: one OCI image
(built from the repo's existing `docker/dockerfile`), serving plain HTTP, runs
identically on RunPod, Cloud Run, or a plain VM. Each provider gets only a thin
config file here.

## The portability contract (what the container guarantees)

The image makes zero provider-specific assumptions. It is configured entirely
through environment variables:

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `6050` | HTTP port. Cloud Run injects this; RunPod/VM can set it. (`app/main.py`) |
| `HOST` | `0.0.0.0` | Bind address. |
| `DEVICE` | `cuda` | `cuda` / `cpu` / `mps`. |
| `WORKERS` | `1` | **Keep at 1 per GPU.** Scale with replicas, not workers (see below). |
| `MODEL_BASE` | `/datasets` | Prefix for model weights in `app/model_config.json`. Can be a filesystem path **or** an `https://` object-store prefix. |

**Model storage is the one thing to decide per environment.** `MODEL_BASE` is
expanded into `model_config.json` at startup. Because every model loader (incl.
the YOLO detector) resolves paths through `checkpoint_utils.get_checkpoint_path`,
weights can be:

- a **mounted volume** — `MODEL_BASE=/datasets` (VM) or `/runpod-volume/models` (RunPod network volume), or
- an **object-store URL** — `MODEL_BASE=https://storage.googleapis.com/your-bucket/models` (fetched + cached to `/tmp/checkpoints` at boot).

The URL option is the most portable: identical config everywhere, no
provider-specific volume wiring.

## Two load-balancing knobs (the same idea on every platform)

1. **Concurrency per replica = 2** — matches the in-process semaphore
   `MAX_CONCURRENT_PREDICTIONS` in `predict_router.py` / `pipeline_router.py`.
   The platform sends the 3rd concurrent request to another GPU replica.
2. **min / max replicas** — `min=1` keeps a GPU warm (fast first response);
   `max=N` is your burst ceiling. `min=0` = true scale-to-zero (cheapest, but
   the first request after idle pays the cold start).

> **Do not raise `WORKERS`.** Multiple uvicorn workers on one GPU each load a
> full copy of all 5 models into the same VRAM (OOM risk) while the GPU still
> executes serially — no throughput gain. One worker per GPU; scale via replicas.

## Cold start (know this before choosing min=0)

The service eagerly loads 5 models at startup (`startup_event`), and the health
check allows a 90s `start_period`. Real cold start = image pull + load all
models into VRAM (tens of seconds). Mitigations: keep `min`/`activeWorkers >= 1`
(chosen here), and put weights on fast storage near the GPU region.

## Per-provider files

- **Cloud Run** — `cloudrun/service.yaml` (declarative) or `cloudrun/deploy.sh`
  (imperative). NVIDIA L4, `min-instances=1`, `concurrency=2`, `timeout=300`.
- **RunPod** — `runpod/endpoint.json`. HTTP **load-balancing** serverless
  endpoint (not the queue/handler model, which would be RunPod-specific code),
  `activeWorkers=1`, `concurrencyPerWorker=2`, network volume or URL for models.

Both consume the **same image**. To move providers, you rebuild nothing — you
just apply the other config file and point `MODEL_BASE` at that environment's
model store.

## Quick local parity check

```bash
# Runs the same way the platforms invoke it (env-driven, 1 worker, CPU):
PORT=6050 DEVICE=cpu WORKERS=1 MODEL_BASE=/datasets python3 -m app.main
curl -f http://localhost:6050/health
```
