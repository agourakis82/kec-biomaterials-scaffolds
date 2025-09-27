#!/bin/bash
# Script to deploy the Next.js UI to Google Cloud Run.

# --- CONFIGURATION ---
# !!! EDIT THESE VARIABLES !!!
export PROJECT_ID="pcs-helio"          # Your Google Cloud project ID
export REGION="us-central1"                # e.g., us-central1
export AR_REPO="kec-biomat-repo"   # e.g., my-app-repo
export UI_SERVICE_NAME="darwin-frontend-web"      # Name for your Cloud Run service
 
# These are passed to the Cloud Run service.
# The UI server will use these to proxy requests to the DARWIN backend.
export NEXT_PUBLIC_API_BASE_URL="https://api.agourakis.med.br" # URL of your deployed DARWIN backend
# DARWIN_API_KEY is not used by the frontend directly, but might be needed for other services.
# export DARWIN_API_KEY="dk_e99a9e39aafe48362b4a2a2e34f822d3_1758238271"      # Your DARWIN API Key
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
docker build --network=host -t ${IMAGE_NAME} ./ui

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
  --set-env-vars="NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}" \
  --project ${PROJECT_ID}

echo "
--------------------------------------------------"
echo "Deployment successful!"
SERVICE_URL=$(gcloud run services describe ${UI_SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')
echo "Your UI is available at: ${SERVICE_URL}"
echo "--------------------------------------------------"

echo "
[STEP 5/5] Configuring Custom Domain Mapping..."
echo "Attempting to delete existing domain mapping for darwin.agourakis.med.br..."
gcloud alpha run domain-mappings delete --domain darwin.agourakis.med.br --project ${PROJECT_ID} --region ${REGION} --quiet || true
echo "Existing domain mapping deleted or did not exist."

echo "Creating domain mapping for darwin.agourakis.med.br to ${UI_SERVICE_NAME}..."
gcloud alpha run domain-mappings create --service ${UI_SERVICE_NAME} --domain darwin.agourakis.med.br --project ${PROJECT_ID} --region ${REGION}
echo "Domain mapping created. Please ensure your DNS records (CNAME for 'darwin') are updated to point to the Cloud Run service."

echo "
--------------------------------------------------"
echo "IMPORTANT: The previous service 'app-agourakis-med-br' has been kept for rollback purposes."
echo "If you need to revert, you can switch the domain mapping back to 'app-agourakis-med-br' or deploy the old image to it."
echo "--------------------------------------------------"
