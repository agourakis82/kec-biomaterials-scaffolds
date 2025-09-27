#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-pcs-helio}"
DATASET="${DATASET:-ragplus}"
TABLE="${TABLE:-kb}"
LOCATION="${LOCATION:-us-central1}"

echo "[BQ] Ensuring dataset ${PROJECT_ID}:${DATASET} in ${LOCATION}"
bq --location="${LOCATION}" mk -d --description "RAG++ Knowledge Base" "${PROJECT_ID}:${DATASET}" 2>/dev/null || echo "[BQ] Dataset exists"

echo "[BQ] Ensuring table ${PROJECT_ID}.${DATASET}.${TABLE}"
bq query --project_id="${PROJECT_ID}" --use_legacy_sql=false "
CREATE TABLE IF NOT EXISTS \`${PROJECT_ID}.${DATASET}.${TABLE}\` (
  id STRING NOT NULL,
  content STRING,
  embedding ARRAY<FLOAT64>,
  source STRING,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
"

echo "[BQ] Done."