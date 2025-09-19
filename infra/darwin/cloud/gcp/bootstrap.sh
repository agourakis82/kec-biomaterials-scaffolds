#!/usr/bin/env bash
set -euo pipefail

# ====== EDITAR ======
PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
SA_NAME="${GCP_SA_NAME:-darwin-runner}"
BQ_DATASET="${BQ_DATASET:-darwin_kg}"
BQ_TABLE="${BQ_TABLE:-documents}"
# ====================

gcloud config set project "$PROJECT_ID"

# APIS
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com

# Service Account
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud iam service-accounts create "$SA_NAME" --display-name "DARWIN Runner"

# IAM mínimos
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/run.admin"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/aiplatform.user"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/bigquery.admin"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/logging.logWriter"

# BigQuery dataset & tabela
bq --location="$REGION" mk -d --description "DARWIN KG" "${PROJECT_ID}:${BQ_DATASET}" || true
bq mk --table "${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}" \
  title:STRING,url:STRING,abstract:STRING,content:STRING,source:STRING,created_at:TIMESTAMP,doc_id:STRING

echo "SA_EMAIL=${SA_EMAIL}"
