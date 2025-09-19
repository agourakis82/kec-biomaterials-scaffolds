#!/usr/bin/env bash
set -euo pipefail

# Fix script for existing DARWIN deployment permissions issues
# This script addresses the warnings from the deployment output

# ====== CONFIGURATION ======
PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
SA_NAME="${GCP_SA_NAME:-darwin-runner}"
SERVICE="darwin-rag"
# ===========================

echo "🔧 Fixing DARWIN deployment permissions..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Account: $SA_NAME"
echo "Service: $SERVICE"
echo ""

gcloud config set project "$PROJECT_ID"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
COMPUTE_SA_EMAIL="862291807500-compute@developer.gserviceaccount.com"

echo "📝 Step 1: Adding missing Cloud Logging permissions..."

# Add logging.logWriter role to the custom service account
echo "   Adding roles/logging.logWriter to $SA_EMAIL"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role "roles/logging.logWriter" || echo "   Role may already exist"

# Also add logging permissions to the default Compute Engine service account
# (in case Cloud Build is using it)
echo "   Adding roles/logging.logWriter to default Compute Engine SA"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${COMPUTE_SA_EMAIL}" \
  --role "roles/logging.logWriter" || echo "   Role may already exist"

echo ""
echo "🌐 Step 2: Fixing Cloud Run IAM policy for public access..."

# Set the Cloud Run service to allow unauthenticated access
echo "   Setting Cloud Run service to allow unauthenticated access"
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member="allUsers" \
  --role="roles/run.invoker" || echo "   Policy may already exist"

echo ""
echo "🔍 Step 3: Verifying service account permissions..."

echo "   Current IAM bindings for $SA_EMAIL:"
gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}"

echo ""
echo "   Current IAM bindings for default Compute Engine SA:"
gcloud projects get-iam-policy "$PROJECT_ID" \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:serviceAccount:${COMPUTE_SA_EMAIL}"

echo ""
echo "🚀 Step 4: Verifying Cloud Run service status..."

SERVICE_URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)' 2>/dev/null || echo "Service not found")
if [ "$SERVICE_URL" != "Service not found" ]; then
    echo "   Service URL: $SERVICE_URL"
    echo "   Service status: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.conditions[0].type)')"
else
    echo "   ⚠️  Service $SERVICE not found in region $REGION"
fi

echo ""
echo "✅ Permission fixes completed!"
echo ""
echo "📋 Summary of changes:"
echo "   ✓ Added roles/logging.logWriter to custom service account"
echo "   ✓ Added roles/logging.logWriter to default Compute Engine service account"
echo "   ✓ Set Cloud Run service to allow public access"
echo ""
echo "🧪 Next steps:"
echo "   1. Test your service at: $SERVICE_URL"
echo "   2. Check Cloud Logging for application logs"
echo "   3. Monitor for any remaining permission issues"
echo ""
echo "💡 If you still see logging issues, ensure your application is configured"
echo "   to use Google Cloud Logging (e.g., google-cloud-logging Python package)"
