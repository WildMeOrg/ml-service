#!/usr/bin/env bash
# Imperative Cloud Run deploy (alternative to `gcloud run services replace service.yaml`).
# Deploys an already-built-and-pushed image onto an L4 GPU.
#
# Prereqs: gcloud auth, an Artifact Registry repo named "ml-service", and the
# image already pushed to $IMAGE.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${MODEL_BUCKET:?set MODEL_BUCKET (bucket holding the model weights)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-service/ml-service:latest"

# Build + push first. `gcloud builds submit` has no -f flag, so it cannot point
# at docker/dockerfile from the repo root without first copying it to
# ./Dockerfile; a plain local build + push is simpler:
#   docker build -f docker/dockerfile -t "${IMAGE}" .
#   docker push "${IMAGE}"

gcloud run deploy ml-service \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi \
  --concurrency 1 \
  --min-instances 1 \
  --max-instances 5 \
  --no-cpu-throttling \
  --timeout 300 \
  --port 6050 \
  --no-gpu-zonal-redundancy \
  --startup-probe "httpGet.path=/readyz,httpGet.port=6050,initialDelaySeconds=60,periodSeconds=5,failureThreshold=30" \
  --set-env-vars "MODEL_BASE=https://storage.googleapis.com/${BUCKET}/models,WORKERS=1,DEVICE=cuda"

# --startup-probe on /readyz rather than the default TCP check: a TCP probe
# passes as soon as the port is open, which for a worker whose registry loaded
# no models means traffic gets routed to something that cannot serve it.
# --no-gpu-zonal-redundancy is required in projects without zonal-redundant L4
# quota; drop it if yours has the quota.
#
# For true scale-to-zero (cheapest, cold start on first hit): --min-instances 0
