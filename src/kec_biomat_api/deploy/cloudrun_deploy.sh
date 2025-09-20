#!/bin/bash
set -euo pipefail

# Cloud Run deploy script for Darwin API
echo "[INFO] Starting Cloud Run deployment for Darwin API..."

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "[ERROR] GCP_PROJECT_ID environment variable is required." >&2
  exit 1
fi

GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="darwin-kec-biomat"
IMAGE="gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME"

# Build container image
echo "[INFO] Building Docker image with Cloud Build..."
gcloud builds submit --tag "$IMAGE" infra/api

# Deploy to Cloud Run
echo "[INFO] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$GCP_REGION" \
  --platform managed \
  --service-account "$SERVICE_ACCOUNT" \
  --set-env-vars "NAMESPACE=$NAMESPACE" \
  --set-env-vars "BIGQUERY_DATASET=$BIGQUERY_DATASET,BIGQUERY_TABLE=$BIGQUERY_TABLE" \
  --set-env-vars "VERTEX_LOCATION=$VERTEX_LOCATION,VERTEX_TEXT_MODEL=$VERTEX_TEXT_MODEL,VERTEX_EMB_MODEL=$VERTEX_EMB_MODEL" \
  --set-env-vars "GCP_PROJECT_ID=$GCP_PROJECT_ID" \
  --set-secrets "API_KEY=darwin_api_key:latest,OPENAI_API_KEY=openai_api_key:latest"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" --format='value(status.url)')
echo "[SUCCESS] Service deployed!"
echo "[HINT] Access your service at: $SERVICE_URL"
