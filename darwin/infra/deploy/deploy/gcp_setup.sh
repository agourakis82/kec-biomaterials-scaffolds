#!/bin/bash
# 🌐 DARWIN GCP Complete Setup Script
# Setup completo do projeto pcs-helio para DARWIN Meta-Research Brain

set -e  # Exit on any error

# Configuration
PROJECT_ID="pcs-helio"
REGION="us-central1"
ZONE="us-central1-a"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 DARWIN GCP Complete Setup${NC}"
echo -e "${PURPLE}============================${NC}"

# Verify prerequisites
echo -e "${YELLOW}📋 Verifying prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Installing...${NC}"
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
    gcloud init
fi

# Set project configuration
echo -e "${YELLOW}🎯 Setting GCP project configuration...${NC}"
gcloud config set project ${PROJECT_ID}
gcloud config set compute/region ${REGION}
gcloud config set compute/zone ${ZONE}

echo -e "${GREEN}✅ Project: ${PROJECT_ID}${NC}"
echo -e "${GREEN}✅ Region: ${REGION}${NC}"
echo -e "${GREEN}✅ Zone: ${ZONE}${NC}"

# Enable required APIs
echo -e "${YELLOW}📡 Enabling required GCP APIs...${NC}"

REQUIRED_APIS=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "secretmanager.googleapis.com"
    "bigquery.googleapis.com"
    "aiplatform.googleapis.com"
    "dns.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
    "redis.googleapis.com"
    "containerregistry.googleapis.com"
    "compute.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    echo -e "${BLUE}   Enabling ${api}...${NC}"
    gcloud services enable ${api} --project=${PROJECT_ID}
done

echo -e "${GREEN}✅ All APIs enabled successfully${NC}"

# Create service accounts
echo -e "${YELLOW}👤 Creating service accounts...${NC}"

# Create service account for Cloud Run
if ! gcloud iam service-accounts describe darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com --project=${PROJECT_ID} &> /dev/null; then
    gcloud iam service-accounts create darwin-runner \
        --display-name="DARWIN Cloud Run Service Account" \
        --description="Service account for DARWIN Cloud Run services" \
        --project=${PROJECT_ID}
    
    echo -e "${GREEN}✅ Service account darwin-runner created${NC}"
else
    echo -e "${YELLOW}⚠️ Service account darwin-runner already exists${NC}"
fi

# Grant necessary roles
echo -e "${YELLOW}🔐 Granting IAM roles...${NC}"

IAM_ROLES=(
    "roles/secretmanager.secretAccessor"
    "roles/bigquery.dataEditor"
    "roles/redis.editor"
    "roles/aiplatform.user"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
)

for role in "${IAM_ROLES[@]}"; do
    echo -e "${BLUE}   Granting ${role}...${NC}"
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="${role}" \
        --quiet
done

echo -e "${GREEN}✅ IAM roles configured${NC}"

# Setup Secret Manager
echo -e "${YELLOW}🔐 Setting up Secret Manager...${NC}"

SECRETS=(
    "openai-api-key"
    "anthropic-api-key"
    "google-ai-api-key"
    "darwin-server-key"
)

for secret in "${SECRETS[@]}"; do
    if ! gcloud secrets describe ${secret} --project=${PROJECT_ID} &> /dev/null; then
        gcloud secrets create ${secret} \
            --replication-policy="automatic" \
            --project=${PROJECT_ID}
        echo -e "${GREEN}✅ Secret ${secret} created${NC}"
    else
        echo -e "${YELLOW}⚠️ Secret ${secret} already exists${NC}"
    fi
done

echo -e "${YELLOW}📝 Remember to add actual secret values:${NC}"
for secret in "${SECRETS[@]}"; do
    echo -e "${BLUE}   echo 'your-${secret}' | gcloud secrets versions add ${secret} --data-file=-${NC}"
done

# Setup BigQuery
echo -e "${YELLOW}📊 Setting up BigQuery datasets...${NC}"

BIGQUERY_DATASETS=(
    "darwin_knowledge:DARWIN Knowledge Base"
    "darwin_analytics:DARWIN Analytics & Metrics"
    "darwin_logs:DARWIN User Activity Logs"
)

for dataset_info in "${BIGQUERY_DATASETS[@]}"; do
    dataset_name="${dataset_info%%:*}"
    dataset_desc="${dataset_info#*:}"
    
    if ! bq show --project_id=${PROJECT_ID} --dataset ${dataset_name} &> /dev/null; then
        bq mk --project_id=${PROJECT_ID} --dataset \
            --description="${dataset_desc}" \
            ${dataset_name}
        echo -e "${GREEN}✅ BigQuery dataset ${dataset_name} created${NC}"
    else
        echo -e "${YELLOW}⚠️ BigQuery dataset ${dataset_name} already exists${NC}"
    fi
done

# Setup Redis Memory Store
echo -e "${YELLOW}🔴 Setting up Redis Memory Store...${NC}"

REDIS_INSTANCE="darwin-cache"

if ! gcloud redis instances describe ${REDIS_INSTANCE} --region=${REGION} --project=${PROJECT_ID} &> /dev/null; then
    gcloud redis instances create ${REDIS_INSTANCE} \
        --size=1 \
        --region=${REGION} \
        --redis-version=redis_6_x \
        --tier=basic \
        --project=${PROJECT_ID} \
        --async
    
    echo -e "${GREEN}✅ Redis instance ${REDIS_INSTANCE} creation started${NC}"
    echo -e "${YELLOW}⏳ Redis instance will be ready in ~5-10 minutes${NC}"
else
    echo -e "${YELLOW}⚠️ Redis instance ${REDIS_INSTANCE} already exists${NC}"
fi

# Setup Cloud Build triggers (optional)
echo -e "${YELLOW}🔄 Setting up Cloud Build configurations...${NC}"

# Create cloudbuild.yaml if it doesn't exist
if [ ! -f "../cloudbuild.yaml" ]; then
    echo -e "${BLUE}   Creating cloudbuild.yaml...${NC}"
    cat > ../cloudbuild.yaml << 'EOF'
steps:
# Build Backend
- name: 'gcr.io/cloud-builders/docker'
  args: [
    'build', 
    '-t', 'gcr.io/pcs-helio/darwin-unified-brain:$COMMIT_SHA',
    '-t', 'gcr.io/pcs-helio/darwin-unified-brain:latest',
    '-f', 'src/kec_unified_api/Dockerfile.production',
    './src/kec_unified_api'
  ]

# Push Backend Image
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/pcs-helio/darwin-unified-brain:latest']

# Deploy Backend
- name: 'gcr.io/cloud-builders/gcloud'
  args: [
    'run', 'deploy', 'darwin-unified-brain',
    '--image=gcr.io/pcs-helio/darwin-unified-brain:latest',
    '--region=us-central1',
    '--platform=managed',
    '--allow-unauthenticated'
  ]

# Build Frontend
- name: 'gcr.io/cloud-builders/npm'
  args: ['install']
  dir: 'ui'

- name: 'gcr.io/cloud-builders/npm'
  args: ['run', 'build']
  dir: 'ui'
  env:
  - 'NODE_ENV=production'
  - 'NEXT_PUBLIC_API_URL=https://api.agourakis.med.br'

- name: 'gcr.io/cloud-builders/docker'
  args: [
    'build',
    '-t', 'gcr.io/pcs-helio/darwin-frontend:latest',
    '-f', 'ui/Dockerfile.production',
    './ui'
  ]

# Push Frontend Image
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/pcs-helio/darwin-frontend:latest']

# Deploy Frontend
- name: 'gcr.io/cloud-builders/gcloud'
  args: [
    'run', 'deploy', 'darwin-frontend',
    '--image=gcr.io/pcs-helio/darwin-frontend:latest',
    '--region=us-central1',
    '--platform=managed',
    '--allow-unauthenticated'
  ]

timeout: 1200s
options:
  logging: CLOUD_LOGGING_ONLY
EOF
    echo -e "${GREEN}✅ cloudbuild.yaml created${NC}"
fi

# Setup monitoring and alerting
echo -e "${YELLOW}📊 Setting up Cloud Monitoring...${NC}"

# Create notification channel (email)
# Note: This would typically require an email to be configured
echo -e "${BLUE}   Monitoring dashboards will be created after first deployment${NC}"

# Display setup summary
echo ""
echo -e "${GREEN}🎉 DARWIN GCP Setup Complete!${NC}"
echo -e "${PURPLE}==============================${NC}"
echo ""
echo -e "${GREEN}✅ Project configured: ${PROJECT_ID}${NC}"
echo -e "${GREEN}✅ APIs enabled: ${#REQUIRED_APIS[@]} services${NC}"
echo -e "${GREEN}✅ Service accounts: darwin-runner${NC}"
echo -e "${GREEN}✅ Secrets created: ${#SECRETS[@]} secrets${NC}"
echo -e "${GREEN}✅ BigQuery datasets: ${#BIGQUERY_DATASETS[@]} datasets${NC}"
echo -e "${GREEN}✅ Redis instance: ${REDIS_INSTANCE} (creating...)${NC}"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo -e "${YELLOW}1. Add API keys to Secret Manager:${NC}"
for secret in "${SECRETS[@]}"; do
    echo -e "${BLUE}   echo 'your-key' | gcloud secrets versions add ${secret} --data-file=-${NC}"
done
echo ""
echo -e "${YELLOW}2. Deploy DARWIN services:${NC}"
echo -e "${BLUE}   ./deploy/gcp_deploy_backend.sh${NC}"
echo -e "${BLUE}   ./deploy/gcp_deploy_frontend.sh${NC}"
echo ""
echo -e "${YELLOW}3. Configure DNS records:${NC}"
echo -e "${BLUE}   api.agourakis.med.br CNAME ghs.googlehosted.com${NC}"
echo -e "${BLUE}   darwin.agourakis.med.br CNAME ghs.googlehosted.com${NC}"
echo ""
echo -e "${YELLOW}4. Monitor deployment:${NC}"
echo -e "${BLUE}   https://console.cloud.google.com/run?project=${PROJECT_ID}${NC}"
echo ""

# Check Redis status
echo -e "${YELLOW}🔍 Checking Redis instance status...${NC}"
REDIS_STATUS=$(gcloud redis instances describe ${REDIS_INSTANCE} --region=${REGION} --project=${PROJECT_ID} --format="value(state)" 2>/dev/null || echo "CREATING")
echo -e "${BLUE}Redis Status: ${REDIS_STATUS}${NC}"

if [ "$REDIS_STATUS" = "READY" ]; then
    REDIS_IP=$(gcloud redis instances describe ${REDIS_INSTANCE} --region=${REGION} --project=${PROJECT_ID} --format="value(host)")
    echo -e "${GREEN}✅ Redis ready at: ${REDIS_IP}:6379${NC}"
fi

echo -e "${GREEN}✅ DARWIN GCP setup completed successfully!${NC}"
echo -e "${PURPLE}Ready for deployment! 🚀${NC}"