#!/bin/bash

# Deploy Simplificado para Diagnóstico
set -e

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🚀 Deploy Simplificado para Diagnóstico${NC}"

# Configurações
PROJECT_ID="pcs-helio"
REGION="us-central1"
SERVICE_NAME="darwin-kec-biomat-simple"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# Build e Deploy
(cd src/kec_biomat_api && gcloud builds submit --tag ${IMAGE_NAME} .)
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated

echo -e "${GREEN}✅ Deploy simplificado concluído!${NC}"