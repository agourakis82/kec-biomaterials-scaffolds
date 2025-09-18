#!/usr/bin/env bash
set -euo pipefail
PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
JOB="darwin-discovery"

gcloud scheduler jobs create http darwin-discovery-hourly \
  --schedule="0 * * * *" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB}:run" \
  --http-method=POST \
  --oauth-service-account-email "$(gcloud config get-value account)" \
  --location="$REGION" || true

echo "Agendado: darwin-discovery-hourly"
