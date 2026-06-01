#!/usr/bin/env bash
# Imperative Cloud Run deploy (alternative to `gcloud run services replace service.yaml`).
# Builds the existing Dockerfile, pushes to Artifact Registry, deploys with an L4 GPU.
#
# Prereqs: gcloud auth + an Artifact Registry repo named "ml-service".
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${MODEL_BUCKET:?set MODEL_BUCKET (gs bucket holding model weights)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/ml-service/ml-detector:latest"

# Build with the repo's existing Dockerfile (no Cloud-Run-specific image).
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" -f docker/dockerfile .

gcloud run deploy ml-detector \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 8 --memory 32Gi \
  --concurrency 2 \
  --min-instances 1 \
  --max-instances 10 \
  --no-cpu-throttling \
  --timeout 300 \
  --port 6050 \
  --set-env-vars "PORT=6050,DEVICE=cuda,WORKERS=1,MODEL_BASE=https://storage.googleapis.com/${BUCKET}/models"

# For true scale-to-zero (cheapest, cold-start on first hit): --min-instances 0
