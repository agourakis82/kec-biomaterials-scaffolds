#!/bin/bash

set -e

# Diretório base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo "gcloud CLI não encontrado. Instale em https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Carregar variáveis do .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env não encontrado. Crie com PROJECT_ID, REGION, etc."
    exit 1
fi

PROJECT_ID=${PROJECT_ID:-pcs-helio}
REGION=${REGION:-us-central1}
BACKEND_SERVICE_NAME="kec-backend"
FRONTEND_SERVICE_NAME="kec-frontend"

echo "Iniciando deploy para GCP Cloud Run..."

# Build e push para backend
echo "Construindo e enviando imagem do backend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$BACKEND_SERVICE_NAME backend/kec_unified_api

# Deploy backend
echo "Deploying backend para Cloud Run..."
gcloud run deploy $BACKEND_SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$BACKEND_SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "REDIS_URL=redis://redis:6379/0,OLLAMA_URL=ollama-url,ENVIRONMENT=production" \
  --update-secrets "API_KEYS=api-keys:latest" \
  --service-account kec-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --max-instances 10

# Build e push para frontend
echo "Construindo e enviando imagem do frontend..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME frontend/ui

# Deploy frontend
echo "Deploying frontend para Cloud Run..."
gcloud run deploy $FRONTEND_SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --port 3000 \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars "NODE_ENV=production,NEXT_PUBLIC_API_URL=https://$BACKEND_SERVICE_NAME-$REGION.run.app" \
  --max-instances 5

echo "Deploy concluído!"
echo "Backend URL: https://$BACKEND_SERVICE_NAME-$REGION.run.app"
echo "Frontend URL: https://$FRONTEND_SERVICE_NAME-$REGION.run.app"
echo "Teste health check: curl https://$BACKEND_SERVICE_NAME-$REGION.run.app/healthz"