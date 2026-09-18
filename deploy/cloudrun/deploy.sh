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
  --set-env-vars "MODEL_BASE=https://storage.googleapis.com/${BUCKET}/models,WORKERS=1,DEVICE=cuda"

# Two things this imperative path cannot express, both in service.yaml:
#   - the /readyz startup probe (gcloud run deploy has no flag for it, so you
#     get Cloud Run's default TCP check and traffic can reach a worker whose
#     registry failed to load)
#   - gpu-zonal-redundancy-disabled, needed in projects without L4 zonal quota
# Apply service.yaml instead if either matters.
#
# For true scale-to-zero (cheapest, cold start on first hit): --min-instances 0
