# 🚀 DARWIN GCP Deployment Guide

## 📋 Visão Geral

Este guia fornece instruções completas para deployar a plataforma **DARWIN Meta-Research Brain** no Google Cloud Platform (GCP) usando o projeto **pcs-helio**. 

**🎯 Objetivo:** Deploy completo de produção com alta disponibilidade, auto-scaling e monitoramento.

**🏗️ Arquitetura Alvo:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Cloud Run     │    │   Databases     │
│   + SSL         │────│   Frontend      │────│   BigQuery      │
│                 │    │   Backend       │    │   Redis         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        └────── api.agourakis.med.br ──────────────────────┘
```

---

## 📋 Índice

1. [Prerequisites](#prerequisites)
2. [Configuração Inicial do Projeto](#configuração-inicial)
3. [Backend Deployment](#backend-deployment)
4. [Frontend Deployment](#frontend-deployment)  
5. [Database Setup](#database-setup)
6. [Custom Domain & SSL](#custom-domain--ssl)
7. [Monitoring & Logging](#monitoring--logging)
8. [Auto-scaling Configuration](#auto-scaling)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Security & Secrets](#security--secrets)
11. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### 1. Ferramentas Necessárias

```bash
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python 3.11+
sudo apt-get update
sudo apt-get install python3.11 python3.11-pip
```

### 2. Configuração de Projeto GCP

```bash
# Set project
export PROJECT_ID="pcs-helio"
export REGION="us-central1"
export ZONE="us-central1-a"

gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### 3. Verificação de Permissões

```bash
# Check current permissions
gcloud projects get-iam-policy $PROJECT_ID

# Required roles:
# - Cloud Run Admin
# - Cloud Build Editor  
# - Secret Manager Admin
# - BigQuery Admin
# - DNS Administrator
```

---

## ⚙️ Configuração Inicial

### 1. Habilitar APIs Necessárias

```bash
#!/bin/bash
# Script: enable_gcp_apis.sh

PROJECT_ID="pcs-helio"

# Enable required APIs
gcloud services enable run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  bigquery.googleapis.com \
  aiplatform.googleapis.com \
  dns.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  redis.googleapis.com \
  --project=$PROJECT_ID

echo "✅ All APIs enabled successfully"
```

### 2. Setup de IAM e Service Accounts

```bash
#!/bin/bash
# Script: setup_iam.sh

PROJECT_ID="pcs-helio"

# Create service account for Cloud Run
gcloud iam service-accounts create darwin-runner \
    --display-name="DARWIN Cloud Run Service Account" \
    --project=$PROJECT_ID

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/redis.editor"

echo "✅ Service accounts configured"
```

---

## 🔧 Backend Deployment

### 1. Dockerfile de Produção

```dockerfile
# File: src/kec_unified_api/Dockerfile.production
FROM python:3.11-slim

# System dependencies mínimas para produção
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-rag-plus.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-rag-plus.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONPATH="/app"
ENV PORT=8080
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE 8080

# Production command with Gunicorn
CMD exec gunicorn --bind :$PORT \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 60 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info \
    main:app
```

### 2. Script de Deploy Backend

```bash
#!/bin/bash
# File: deploy/gcp_deploy_backend.sh

set -e  # Exit on error

PROJECT_ID="pcs-helio"
SERVICE_NAME="darwin-unified-brain"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting DARWIN Backend Deployment"

# Build Docker image
echo "📦 Building Docker image..."
cd src/kec_unified_api
docker build -f Dockerfile.production -t ${IMAGE_NAME}:latest .

# Push to Google Container Registry
echo "⬆️ Pushing to GCR..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --platform=managed \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --port=8080 \
    --allow-unauthenticated \
    --service-account=darwin-runner@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars="ENVIRONMENT=production,PROJECT_ID=${PROJECT_ID}" \
    --set-secrets="OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,GOOGLE_AI_API_KEY=google-ai-api-key:latest" \
    --concurrency=100 \
    --timeout=300

echo "✅ Backend deployed successfully!"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --format="value(status.url)")

echo "🌐 Backend URL: ${SERVICE_URL}"

# Test health check
echo "🏥 Testing health check..."
curl -f "${SERVICE_URL}/health" && echo "✅ Health check passed!"
```

### 3. Environment Variables para Backend

```bash
# Environment variables for backend Cloud Run
ENVIRONMENT=production
PROJECT_ID=pcs-helio
GOOGLE_CLOUD_PROJECT=pcs-helio

# Database connections
REDIS_URL=redis://redis-instance:6379
BIGQUERY_PROJECT=pcs-helio
BIGQUERY_DATASET=darwin_knowledge

# API Keys (managed via Secret Manager)
# OPENAI_API_KEY - from Secret Manager
# ANTHROPIC_API_KEY - from Secret Manager  
# GOOGLE_AI_API_KEY - from Secret Manager

# Feature flags
ENABLE_RAG_PLUS=true
ENABLE_MULTI_AI=true
ENABLE_KEC_METRICS=true
ENABLE_TREE_SEARCH=true
ENABLE_SCIENTIFIC_DISCOVERY=true
ENABLE_SCORE_CONTRACTS=true
ENABLE_KNOWLEDGE_GRAPH=true

# Performance settings
MAX_WORKERS=4
REQUEST_TIMEOUT=60
CACHE_TTL=3600
```

---

## 🌐 Frontend Deployment

### 1. Dockerfile para Frontend

```dockerfile
# File: ui/Dockerfile.production
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code  
COPY . .

# Build application with production config
RUN NODE_ENV=production npm run build

# Production stage
FROM node:18-alpine AS runner

WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Set user
USER nextjs

# Environment
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:3000/api/health || exit 1

EXPOSE 3000

# Start application
CMD ["node", "server.js"]
```

### 2. Script de Deploy Frontend

```bash
#!/bin/bash
# File: deploy/gcp_deploy_frontend.sh

set -e

PROJECT_ID="pcs-helio"
SERVICE_NAME="darwin-frontend"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🎨 Starting DARWIN Frontend Deployment"

# Build Docker image
echo "📦 Building frontend image..."
cd ui
docker build -f Dockerfile.production -t ${IMAGE_NAME}:latest .

# Push to GCR
echo "⬆️ Pushing to GCR..."
docker push ${IMAGE_NAME}:latest

# Deploy to Cloud Run
echo "🌐 Deploying frontend to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --platform=managed \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=5 \
    --port=3000 \
    --allow-unauthenticated \
    --set-env-vars="NEXT_PUBLIC_API_URL=https://api.agourakis.med.br,NEXT_PUBLIC_ENVIRONMENT=production" \
    --concurrency=50 \
    --timeout=60

echo "✅ Frontend deployed successfully!"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --format="value(status.url)")

echo "🌐 Frontend URL: ${SERVICE_URL}"
```

---

## 🗄️ Database Setup

### 1. BigQuery Configuration

```bash
#!/bin/bash
# File: deploy/setup_bigquery.sh

PROJECT_ID="pcs-helio"

# Create datasets
echo "📊 Creating BigQuery datasets..."

bq mk --project_id=${PROJECT_ID} --dataset \
    --description="DARWIN Knowledge Base" \
    darwin_knowledge

bq mk --project_id=${PROJECT_ID} --dataset \
    --description="DARWIN Analytics & Metrics" \
    darwin_analytics

bq mk --project_id=${PROJECT_ID} --dataset \
    --description="DARWIN User Activity Logs" \
    darwin_logs

# Create tables
echo "🗃️ Creating BigQuery tables..."

# Knowledge base table
bq mk --project_id=${PROJECT_ID} --table \
    darwin_knowledge.scientific_papers \
    schema_papers.json

# Analytics table  
bq mk --project_id=${PROJECT_ID} --table \
    darwin_analytics.kec_analyses \
    schema_kec.json

# User activity table
bq mk --project_id=${PROJECT_ID} --table \
    darwin_logs.user_activities \
    schema_logs.json

echo "✅ BigQuery setup completed"
```

### 2. Redis Memory Store

```bash
#!/bin/bash
# File: deploy/setup_redis.sh

PROJECT_ID="pcs-helio"
REGION="us-central1"
REDIS_INSTANCE="darwin-cache"

echo "🔴 Creating Redis instance..."

gcloud redis instances create ${REDIS_INSTANCE} \
    --size=1 \
    --region=${REGION} \
    --redis-version=redis_6_x \
    --tier=basic \
    --project=${PROJECT_ID}

# Get Redis IP
REDIS_IP=$(gcloud redis instances describe ${REDIS_INSTANCE} \
    --region=${REGION} \
    --project=${PROJECT_ID} \
    --format="value(host)")

echo "🔴 Redis IP: ${REDIS_IP}"
echo "🔴 Redis connection: redis://${REDIS_IP}:6379"
```

---

## 🔐 Security & Secrets

### 1. Secret Manager Setup

```bash
#!/bin/bash
# File: deploy/setup_secrets.sh

PROJECT_ID="pcs-helio"

echo "🔐 Setting up Secret Manager..."

# Create secrets
gcloud secrets create openai-api-key \
    --project=${PROJECT_ID}

gcloud secrets create anthropic-api-key \
    --project=${PROJECT_ID}

gcloud secrets create google-ai-api-key \
    --project=${PROJECT_ID}

gcloud secrets create darwin-server-key \
    --project=${PROJECT_ID}

echo "🔐 Created secret placeholders"
echo "⚠️  Remember to add actual secret values:"
echo "gcloud secrets versions add openai-api-key --data-file=openai_key.txt"
echo "gcloud secrets versions add anthropic-api-key --data-file=anthropic_key.txt"
echo "gcloud secrets versions add google-ai-api-key --data-file=google_key.txt"
echo "gcloud secrets versions add darwin-server-key --data-file=server_key.txt"
```

### 2. IAM Security Policies

```bash
#!/bin/bash
# File: deploy/setup_security.sh

PROJECT_ID="pcs-helio"

# Cloud Run security
echo "🛡️ Configuring Cloud Run security..."

# Backend security
gcloud run services add-iam-policy-binding darwin-unified-brain \
    --region=us-central1 \
    --member=allUsers \
    --role=roles/run.invoker \
    --project=${PROJECT_ID}

# Frontend security
gcloud run services add-iam-policy-binding darwin-frontend \
    --region=us-central1 \
    --member=allUsers \
    --role=roles/run.invoker \
    --project=${PROJECT_ID}

echo "✅ Security policies configured"
```

---

## 🌍 Custom Domain & SSL

### 1. Domain Mapping

```bash
#!/bin/bash
# File: deploy/setup_custom_domain.sh

PROJECT_ID="pcs-helio"
REGION="us-central1"
DOMAIN_BACKEND="api.agourakis.med.br"
DOMAIN_FRONTEND="darwin.agourakis.med.br"

echo "🌐 Setting up custom domains..."

# Map backend domain
gcloud run domain-mappings create \
    --service=darwin-unified-brain \
    --domain=${DOMAIN_BACKEND} \
    --region=${REGION} \
    --project=${PROJECT_ID}

# Map frontend domain  
gcloud run domain-mappings create \
    --service=darwin-frontend \
    --domain=${DOMAIN_FRONTEND} \
    --region=${REGION} \
    --project=${PROJECT_ID}

echo "✅ Domain mappings created"
echo "📋 Configure DNS records:"
echo "api.agourakis.med.br -> ghs.googlehosted.com"
echo "darwin.agourakis.med.br -> ghs.googlehosted.com"
```

### 2. SSL Certificate Setup

```bash
#!/bin/bash
# File: deploy/setup_ssl.sh

PROJECT_ID="pcs-helio"

echo "🔒 Setting up SSL certificates..."

# Create managed SSL certificate for backend
gcloud compute ssl-certificates create darwin-backend-cert \
    --domains=api.agourakis.med.br \
    --global \
    --project=${PROJECT_ID}

# Create managed SSL certificate for frontend
gcloud compute ssl-certificates create darwin-frontend-cert \
    --domains=darwin.agourakis.med.br \
    --global \
    --project=${PROJECT_ID}

echo "✅ SSL certificates created (may take up to 60 minutes to provision)"
```

---

## 📊 Monitoring & Logging

### 1. Cloud Monitoring Setup

```bash
#!/bin/bash
# File: deploy/setup_monitoring.sh

PROJECT_ID="pcs-helio"

echo "📈 Setting up Cloud Monitoring..."

# Create alerting policy for backend health
cat > backend_alert_policy.json << EOF
{
  "displayName": "DARWIN Backend Health Alert",
  "conditions": [
    {
      "displayName": "Backend Error Rate",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" resource.label.service_name=\"darwin-unified-brain\"",
        "comparison": "COMPARISON_GT",
        "thresholdValue": 0.05,
        "duration": "300s"
      }
    }
  ],
  "notificationChannels": [],
  "alertStrategy": {
    "autoClose": "1800s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=backend_alert_policy.json \
    --project=${PROJECT_ID}

echo "✅ Monitoring configured"
```

### 2. Custom Dashboards

```bash
#!/bin/bash
# File: deploy/setup_dashboards.sh

PROJECT_ID="pcs-helio"

echo "📊 Creating monitoring dashboards..."

# Create custom dashboard
cat > darwin_dashboard.json << EOF
{
  "displayName": "DARWIN Platform Dashboard",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Backend Request Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" resource.label.service_name=\"darwin-unified-brain\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE"
                    }
                  }
                }
              }
            ]
          }
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=darwin_dashboard.json \
    --project=${PROJECT_ID}

echo "✅ Dashboard created"
```

---

## ⚡ Auto-scaling Configuration

### 1. Backend Auto-scaling

```yaml
# File: deploy/backend-autoscaling.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: darwin-unified-brain
  annotations:
    run.googleapis.com/ingress: all
    autoscaling.knative.dev/minScale: "1"
    autoscaling.knative.dev/maxScale: "10"
    run.googleapis.com/cpu-throttling: "false"
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/pcs-helio/darwin-unified-brain:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: 4Gi
            cpu: 2000m
```

### 2. Frontend Auto-scaling

```yaml
# File: deploy/frontend-autoscaling.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: darwin-frontend
  annotations:
    run.googleapis.com/ingress: all
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "5"
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "5"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1"
    spec:
      containerConcurrency: 50
      containers:
      - image: gcr.io/pcs-helio/darwin-frontend:latest
        ports:
        - containerPort: 3000
```

---

## 🔄 CI/CD Pipeline

### 1. Cloud Build Configuration

```yaml
# File: cloudbuild.yaml
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
  args: ['push', 'gcr.io/pcs-helio/darwin-unified-brain:$COMMIT_SHA']

- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/pcs-helio/darwin-unified-brain:latest']

# Deploy Backend
- name: 'gcr.io/cloud-builders/gcloud'
  args: [
    'run', 'deploy', 'darwin-unified-brain',
    '--image=gcr.io/pcs-helio/darwin-unified-brain:$COMMIT_SHA',
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
    '-t', 'gcr.io/pcs-helio/darwin-frontend:$COMMIT_SHA',
    '-t', 'gcr.io/pcs-helio/darwin-frontend:latest',
    '-f', 'ui/Dockerfile.production',
    './ui'
  ]

# Push Frontend Image
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/pcs-helio/darwin-frontend:$COMMIT_SHA']

- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/pcs-helio/darwin-frontend:latest']

# Deploy Frontend
- name: 'gcr.io/cloud-builders/gcloud'
  args: [
    'run', 'deploy', 'darwin-frontend',
    '--image=gcr.io/pcs-helio/darwin-frontend:$COMMIT_SHA',
    '--region=us-central1',
    '--platform=managed',
    '--allow-unauthenticated'
  ]

# Run Tests
- name: 'gcr.io/cloud-builders/gcloud'
  args: ['run', 'services', 'list']

timeout: 1200s
options:
  logging: CLOUD_LOGGING_ONLY
```

### 2. Build Triggers

```bash
#!/bin/bash
# File: deploy/setup_build_triggers.sh

PROJECT_ID="pcs-helio"
REPO_NAME="kec-biomaterials-scaffolds"

echo "🔄 Setting up Cloud Build triggers..."

# Create build trigger for main branch
gcloud builds triggers create github \
    --repo-name=${REPO_NAME} \
    --repo-owner=agourakis82 \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --project=${PROJECT_ID}

echo "✅ Build triggers configured"
```

---

## 🚀 Complete Deployment Script

```bash
#!/bin/bash
# File: deploy/full_deployment.sh

set -e

PROJECT_ID="pcs-helio"
REGION="us-central1"

echo "🚀 DARWIN Complete GCP Deployment"
echo "=================================="

# 1. Enable APIs
echo "📡 Enabling GCP APIs..."
./deploy/enable_gcp_apis.sh

# 2. Setup IAM
echo "👤 Configuring IAM..."
./deploy/setup_iam.sh

# 3. Setup Secrets
echo "🔐 Setting up secrets..."
./deploy/setup_secrets.sh

# 4. Setup Databases
echo "🗄️ Setting up databases..."
./deploy/setup_bigquery.sh
./deploy/setup_redis.sh

# 5. Deploy Backend
echo "🔧 Deploying backend..."
./deploy/gcp_deploy_backend.sh

# 6. Deploy Frontend
echo "🎨 Deploying frontend..."
./deploy/gcp_deploy_frontend.sh

# 7. Setup Custom Domains
echo "🌐 Configuring custom domains..."
./deploy/setup_custom_domain.sh

# 8. Setup Monitoring
echo "📊 Setting up monitoring..."
./deploy/setup_monitoring.sh
./deploy/setup_dashboards.sh

# 9. Setup CI/CD
echo "🔄 Configuring CI/CD..."
./deploy/setup_build_triggers.sh

echo ""
echo "✅ DARWIN Deployment Complete!"
echo "================================"
echo "🌐 Backend URL: https://api.agourakis.med.br"
echo "🎨 Frontend URL: https://darwin.agourakis.med.br"
echo "📊 Monitoring: https://console.cloud.google.com/monitoring"
echo ""
echo "⚠️  Next steps:"
echo "1. Configure DNS records"
echo "2. Add API keys to Secret Manager"
echo "3. Test all endpoints"
echo "4. Configure monitoring alerts"
```

---

## 🔧 Troubleshooting

### Common Issues

#### 🔴 "Service does not exist" Error

**Causa:** Service não foi criado corretamente

**Solução:**
```bash
# Check if service exists
gcloud run services list --project=pcs-helio

# Redeploy if necessary
./deploy/gcp_deploy_backend.sh
```

#### 🟡 "Domain mapping failed" Error

**Causa:** DNS não configurado corretamente

**Solução:**
```bash
# Check domain mapping status
gcloud run domain-mappings describe api.agourakis.med.br \
    --region=us-central1 --project=pcs-helio

# Update DNS records
# api.agourakis.med.br CNAME ghs.googlehosted.com
```

#### 🔵 "Cold start timeout" Error

**Causa:** Container demora muito para inicializar

**Solução:**
```bash
# Increase timeout and add min instances
gcloud run services update darwin-unified-brain \
    --timeout=300 \
    --min-instances=1 \
    --project=pcs-helio \
    --region=us-central1
```

### Performance Tuning

```bash
# Optimize backend performance
gcloud run services update darwin-unified-brain \
    --memory=4Gi \
    --cpu=2 \
    --concurrency=100 \
    --max-instances=10 \
    --execution-environment=gen2 \
    --project=pcs-helio

# Optimize frontend performance  
gcloud run services update darwin-frontend \
    --memory=2Gi \
    --cpu=1 \
    --concurrency=50 \
    --max-instances=5 \
    --project=pcs-helio
```

### Monitoring Commands

```bash
# Check service status
gcloud run services describe darwin-unified-brain \
    --region=us-central1 --project=pcs-helio

# View logs
gcloud logs read "resource.type=cloud_run_revision" \
    --project=pcs-helio --limit=50

# Check metrics
gcloud monitoring metrics list \
    --filter="resource.type=cloud_run_revision" \
    --project=pcs-helio
```

---

## 📋 Final Checklist

### Pre-Deployment
- [ ] GCP Project configured (pcs-helio)
- [ ] APIs enabled
- [ ] IAM permissions set
- [ ] Docker images built
- [ ] Secrets configured

### Deployment
- [ ] Backend deployed to Cloud Run
- [ ] Frontend deployed to Cloud Run  
- [ ] BigQuery datasets created
- [ ] Redis instance running
- [ ] Custom domains mapped
- [ ] SSL certificates provisioned

### Post-Deployment
- [ ] Health checks passing
- [ ] DNS records configured
- [ ] Monitoring dashboards active
- [ ] CI/CD pipeline working
- [ ] Load testing completed
- [ ] Documentation updated

---

## 💡 Best Practices

### Security
- Use Secret Manager for all sensitive data
- Enable VPC connector for database access
- Implement proper CORS policies
- Regular security updates

### Performance  
- Use CDN for static assets
- Implement proper caching strategies
- Monitor and optimize cold starts
- Use appropriate instance sizing

### Cost Optimization
- Set appropriate min/max instances
- Use preemptible instances when possible
- Monitor and optimize resource usage
- Implement proper auto-scaling

### Monitoring
- Set up comprehensive alerting
- Monitor key business metrics
- Implement distributed tracing
- Regular performance reviews

---

*DARWIN GCP Deployment Guide v1.0 | © 2024 Agourakis Research Labs*