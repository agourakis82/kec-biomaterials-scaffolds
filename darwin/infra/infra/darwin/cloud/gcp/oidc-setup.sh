#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIGURATION ======
PROJECT_ID="${GCP_PROJECT_ID:-pcs-helio}"
REGION="${GCP_REGION:-us-central1}"
SA_NAME="${GCP_SA_NAME:-darwin-runner}"
WIP_POOL="${WIP_POOL:-github-oidc-pool}"
WIP_PROVIDER="${WIP_PROVIDER:-github-oidc-provider}"
GH_ORG_REPO="${GH_ORG_REPO:-agourakis82/kec-biomaterials-scaffolds}"
# ===========================

echo "🔧 Setting up OIDC Workload Identity for GitHub Actions..."

gcloud config set project "$PROJECT_ID"

# Enable IAM Service Account Credentials API
gcloud services enable iamcredentials.googleapis.com

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create Workload Identity Pool
echo "📋 Creating Workload Identity Pool: ${WIP_POOL}"
gcloud iam workload-identity-pools create "$WIP_POOL" \
  --location="global" \
  --display-name="GitHub OIDC Pool" \
  --description="Pool for GitHub Actions OIDC authentication" || true

# Create OIDC Provider
echo "🔗 Creating OIDC Provider: ${WIP_PROVIDER}"
gcloud iam workload-identity-pools providers create-oidc "$WIP_PROVIDER" \
  --location="global" \
  --workload-identity-pool="$WIP_POOL" \
  --display-name="GitHub OIDC Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" || true

# Allow GitHub repo to impersonate service account
echo "🔐 Binding GitHub repo ${GH_ORG_REPO} to service account ${SA_EMAIL}"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${WIP_POOL}/attribute.repository/${GH_ORG_REPO}"

# Output the workload identity provider for GitHub Actions
WIP_PROVIDER_FULL="projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${WIP_POOL}/providers/${WIP_PROVIDER}"

echo ""
echo "✅ OIDC Setup Complete!"
echo "📝 Use this in GitHub Actions:"
echo "   workload_identity_provider: ${WIP_PROVIDER_FULL}"
echo "   service_account: ${SA_EMAIL}"
echo ""
