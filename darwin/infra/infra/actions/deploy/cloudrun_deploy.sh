#!/bin/bash
set -euo pipefail

echo "[INFO] Deploying KEC_BIOMAT Actions service to Cloud Run..."

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "[ERROR] GCP_PROJECT_ID environment variable is required." >&2
  exit 1
fi

GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-kec-actions-api}"
IMAGE="gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME"
BUILD_CONTEXT="server"

# Ensure openapi.yaml is present inside the build context so the Dockerfile can copy it.
if [[ -f "openapi.yaml" ]]; then
  echo "[INFO] Copying openapi.yaml into $BUILD_CONTEXT/ for build context..."
  cp openapi.yaml "$BUILD_CONTEXT/"
  CLEAN_OPENAPI_COPY=1
fi

echo "[INFO] Building container image with Cloud Build (context: $BUILD_CONTEXT)..."
gcloud builds submit "$BUILD_CONTEXT" --tag "$IMAGE"

# Clean up temporary copy if created
if [[ "${CLEAN_OPENAPI_COPY:-0}" -eq 1 ]]; then
  rm -f "$BUILD_CONTEXT/openapi.yaml"
fi

echo "[INFO] Deploying image to Cloud Run service $SERVICE_NAME..."

deploy_args=(
  --image "$IMAGE"
  --region "$GCP_REGION"
  --platform managed
  --allow-unauthenticated
)

if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  deploy_args+=(--service-account "$SERVICE_ACCOUNT")
fi

if [[ -n "${KEC_API_KEY_SECRET:-}" ]]; then
  deploy_args+=(--set-secrets "KEC_API_KEY=${KEC_API_KEY_SECRET}")
elif [[ -n "${KEC_API_KEY:-}" ]]; then
  deploy_args+=(--set-env-vars "KEC_API_KEY=${KEC_API_KEY}")
fi

if [[ -n "${MAX_INSTANCES:-}" ]]; then
  deploy_args+=(--max-instances "$MAX_INSTANCES")
fi

if [[ -n "${MIN_INSTANCES:-}" ]]; then
  deploy_args+=(--min-instances "$MIN_INSTANCES")
fi

gcloud run deploy "$SERVICE_NAME" "${deploy_args[@]}"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" --format='value(status.url)')
echo "[SUCCESS] Service deployed at: $SERVICE_URL"
echo "[NEXT] Ensure the X-API-Key secret is configured before connecting GPT Actions."

