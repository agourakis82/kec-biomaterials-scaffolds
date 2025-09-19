#!/bin/bash
# Script to deploy the Next.js UI to Google Cloud Run.

# --- CONFIGURATION ---
# !!! EDIT THESE VARIABLES !!!
export PROJECT_ID="pcs-helio"          # Your Google Cloud project ID
export REGION="us-central1"                # e.g., us-central1
export AR_REPO="kec-biomat-repo"   # e.g., my-app-repo
export UI_SERVICE_NAME="app-agourakis-med-br"      # Name for your Cloud Run service

# These are passed to the Cloud Run service.
# The UI server will use these to proxy requests to the DARWIN backend.
export DARWIN_URL="https://api.agourakis.med.br" # URL of your deployed DARWIN backend
export DARWIN_API_KEY="dk_e99a9e39aafe48362b4a2a2e34f822d3_1758238271"      # Your DARWIN API Key
# --- END CONFIGURATION ---

set -e # Exit immediately if a command exits with a non-zero status.

# Construct the image name
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${UI_SERVICE_NAME}:latest"

echo "--------------------------------------------------"
echo "Starting UI Deployment..."
echo "PROJECT_ID: ${PROJECT_ID}"
echo "REGION: ${REGION}"
echo "UI_SERVICE_NAME: ${UI_SERVICE_NAME}"
echo "IMAGE_NAME: ${IMAGE_NAME}"
echo "--------------------------------------------------"

# 1. Build the Docker image
echo "
[STEP 1/4] Building Docker image..."
docker build -t ${IMAGE_NAME} ./ui

# 2. Configure Docker to use gcloud for authentication
echo "
[STEP 2/4] Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# 3. Push the image to Artifact Registry
echo "
[STEP 3/4] Pushing image to Artifact Registry..."
docker push ${IMAGE_NAME}

# 4. Deploy to Cloud Run
echo "
[STEP 4/4] Deploying to Cloud Run..."
gcloud run deploy ${UI_SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars="DARWIN_URL=${DARWIN_URL}" \
  --set-env-vars="DARWIN_SERVER_KEY=${DARWIN_API_KEY}" \
  --project ${PROJECT_ID}

echo "
--------------------------------------------------"
echo "Deployment successful!"
SERVICE_URL=$(gcloud run services describe ${UI_SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Your UI is available at: ${SERVICE_URL}"
echo "--------------------------------------------------"
