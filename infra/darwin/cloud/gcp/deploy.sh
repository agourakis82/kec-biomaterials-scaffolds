#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="darwin-rag"
IMG="gcr.io/${PROJECT_ID}/${SERVICE}:v$(date +%Y%m%d%H%M%S)"
SA_EMAIL="${GCP_SA_NAME:-darwin-runner}@${PROJECT_ID}.iam.gserviceaccount.com"

# Build & push
gcloud builds submit --tag "$IMG" api/

# Deploy API
gcloud run deploy "$SERVICE" \
  --image "$IMG" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},VECTOR_BACKEND=gcp_bq,DISCOVERY_FROM_SECRET=true" \
  --update-secrets "DARWIN_API_KEY=DARWIN_API_KEY:latest,VERTEX_MODELS_JSON=VERTEX_MODELS_JSON:latest,DISCOVERY_FEEDS_YML=DISCOVERY_FEEDS_YML:latest"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "RUN_URL=${URL}"

# Cloud Run Job para discovery
JOB="darwin-discovery"
gcloud run jobs describe "$JOB" --region "$REGION" >/dev/null 2>&1 || \
  gcloud run jobs create "$JOB" \
    --image "$IMG" \
    --region "$REGION" \
    --service-account "$SA_EMAIL" \
    --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},VECTOR_BACKEND=gcp_bq,DISCOVERY_FROM_SECRET=true" \
    --update-secrets "DARWIN_API_KEY=DARWIN_API_KEY:latest,VERTEX_MODELS_JSON=VERTEX_MODELS_JSON:latest,DISCOVERY_FEEDS_YML=DISCOVERY_FEEDS_YML:latest" \
    --command "python" \
    --args "-c","import requests,os;url=os.environ.get('RUN_URL','${URL}')+'/discovery/run';print(requests.post(url,headers={'X-API-KEY':os.environ['DARWIN_API_KEY']},json={'run_once':True}).text)"

echo "OK. Agende com Cloud Scheduler chamando o Job ${JOB} de hora em hora."
