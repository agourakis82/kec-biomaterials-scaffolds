#!/bin/bash

# Deploy Modular Backend - Script Otimizado para Cloud Run
# Deploy completo do backend modular KEC com Vertex AI, BigQuery e Cloud Run

set -e

# Cores para output  
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 KEC Biomaterials - Deploy Modular Backend v2.0${NC}"
echo "=================================================="

# Configurações - usando projeto existente
PROJECT_ID=${1:-"pcs-helio"}
REGION=${2:-"us-central1"}
SERVICE_NAME=${3:-"kec-biomaterials-api"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Verifica autenticação
echo -e "${YELLOW}🔐 Verificando autenticação...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Não autenticado no Google Cloud${NC}"
    echo "Execute: gcloud auth login"
    exit 1
fi

# Define projeto
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✅ Projeto configurado: ${PROJECT_ID}${NC}"

# 1. REMOVER BUILDS INÚTEIS
echo -e "${YELLOW}🗑️  Removendo builds inúteis...${NC}"

# Remove imagens antigas (com tratamento de erro)
echo "Removendo imagens antigas do Container Registry..."
OLD_IMAGES=$(gcloud container images list-tags gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --filter='timestamp.datetime < -P7D' \
  --format='get(digest)' 2>/dev/null || echo "")

if [ ! -z "$OLD_IMAGES" ]; then
    for digest in $OLD_IMAGES; do
        gcloud container images delete gcr.io/${PROJECT_ID}/${SERVICE_NAME}@${digest} --quiet || true
    done
    echo -e "${GREEN}✅ Imagens antigas removidas${NC}"
else
    echo "Nenhuma imagem antiga encontrada"
fi

# 2. CONFIGURAR VERTEX AI E BIGQUERY
echo -e "${YELLOW}🧠 Configurando Vertex AI e BigQuery...${NC}"

# Enable APIs necessárias
gcloud services enable aiplatform.googleapis.com \
    bigquery.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com

echo -e "${GREEN}✅ APIs do Google Cloud habilitadas${NC}"

# 3. CONFIGURAR BIGQUERY
echo -e "${YELLOW}📊 Configurando BigQuery...${NC}"

# Criar dataset se não existir
if ! bq show --dataset ${PROJECT_ID}:kec_knowledge_base >/dev/null 2>&1; then
    bq mk \
        --dataset \
        --location=US \
        --description="KEC Biomaterials Knowledge Base" \
        ${PROJECT_ID}:kec_knowledge_base
    echo -e "${GREEN}✅ Dataset BigQuery criado: kec_knowledge_base${NC}"
else
    echo "Dataset kec_knowledge_base já existe"
fi

# Criar tabelas principais
echo "Configurando tabelas BigQuery..."

# Tabela de documentos científicos
bq mk --table \
    --description="Scientific documents with embeddings" \
    ${PROJECT_ID}:kec_knowledge_base.scientific_documents \
    id:STRING,content:STRING,embedding:REPEATED,source:STRING,metadata:JSON,timestamp:TIMESTAMP 2>/dev/null || echo "Tabela scientific_documents já existe"

# Tabela de conversações  
bq mk --table \
    --description="Conversation history with LLMs" \
    ${PROJECT_ID}:kec_knowledge_base.conversation_history \
    id:STRING,timestamp:TIMESTAMP,llm_provider:STRING,user_message:STRING,assistant_response:STRING,context_type:STRING,relevance_score:FLOAT64 2>/dev/null || echo "Tabela conversation_history já existe"

echo -e "${GREEN}✅ BigQuery configurado${NC}"

# 4. BUILD E DEPLOY
echo -e "${YELLOW}🏗️  Building e deployando aplicação...${NC}"

# Navega para o diretório da API
pushd src/kec_biomat_api

# Build da imagem
echo "Building Docker image..."
gcloud builds submit --config cloudbuild.yaml .

echo -e "${GREEN}✅ Imagem Docker built: ${IMAGE_NAME}${NC}"

# Deploy no Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --concurrency 100 \
    --timeout 300 \
    --set-env-vars "PYTHONPATH=/app/src:/app/external/pcs-meta-repo:/app,KEC_CONFIG_PATH=/app/src/kec_biomat/configs,DARWIN_MEMORY_PATH=/app/src/darwin_core/memory,GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --port 8080

echo -e "${GREEN}✅ Deploy Cloud Run concluído${NC}"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo -e "${BLUE}🌐 Service URL: ${SERVICE_URL}${NC}"

# 5. TESTAR DEPLOYMENT
echo -e "${YELLOW}🧪 Testando deployment...${NC}"

sleep 10  # Aguarda service estar pronto

# Test health
echo "Testing /healthz..."
if curl -f "${SERVICE_URL}/healthz" -s >/dev/null; then
    echo -e "${GREEN}✅ Health check OK${NC}"
else
    echo -e "${YELLOW}⚠️  Health check failed - service pode estar inicializando${NC}"
fi

# Test GPT Actions
echo "Testing GPT Actions..."
if curl -f "${SERVICE_URL}/gpt-actions/system-health" -s >/dev/null; then
    echo -e "${GREEN}✅ GPT Actions OK${NC}"
else
    echo -e "${YELLOW}⚠️  GPT Actions não acessível ainda${NC}"
fi

# 6. GERAR CONFIGURAÇÃO GPT ACTIONS
echo -e "${YELLOW}🤖 Gerando configuração para ChatGPT...${NC}"

popd  # Volta para raiz

cat > gpt_actions_config.json <<EOF
{
  "schema_version": "v1",
  "name_for_model": "kec_biomaterials_api",
  "name_for_human": "KEC Biomaterials Analysis API",
  "description_for_model": "Advanced biomaterials analysis API with RAG++, tree search, memory systems, and scientific discovery. Provides KEC metrics (entropy, curvature, small-world) calculation, knowledge retrieval, and automated research discovery.",
  "description_for_human": "Analyze porous biomaterials using advanced KEC metrics, search scientific knowledge, and discover recent research.",
  "auth": {
    "type": "none"
  },
  "api": {
    "type": "openapi",
    "url": "${SERVICE_URL}/openapi.json"
  },
  "logo_url": "${SERVICE_URL}/static/logo.png",
  "contact_email": "admin@kec-biomaterials.dev",
  "legal_info_url": "${SERVICE_URL}/docs"
}
EOF

echo -e "${GREEN}✅ Configuração GPT Actions salva: gpt_actions_config.json${NC}"

# 7. RESUMO FINAL
echo ""
echo -e "${GREEN}🎉 DEPLOY MODULAR BACKEND COMPLETO!${NC}"
echo ""
echo -e "${BLUE}📋 INFORMAÇÕES DO DEPLOY:${NC}"
echo "• Projeto: ${PROJECT_ID}"
echo "• Service: ${SERVICE_NAME}"
echo "• URL: ${SERVICE_URL}"
echo "• Região: ${REGION}"
echo ""
echo -e "${BLUE}🤖 ENDPOINTS GPT ACTIONS:${NC}"
echo "• Health: ${SERVICE_URL}/gpt-actions/system-health"
echo "• KEC Metrics: ${SERVICE_URL}/gpt-actions/analyze-kec-metrics"  
echo "• RAG Query: ${SERVICE_URL}/gpt-actions/rag-query"
echo "• Project Status: ${SERVICE_URL}/gpt-actions/project-status"
echo "• Discovery: ${SERVICE_URL}/gpt-actions/scientific-discovery"
echo ""
echo -e "${BLUE}📚 DOCUMENTAÇÃO:${NC}"
echo "• OpenAPI: ${SERVICE_URL}/openapi.json"
echo "• Docs: ${SERVICE_URL}/docs"
echo "• Health: ${SERVICE_URL}/healthz"
echo ""
echo -e "${GREEN}✅ Backend modular KEC v2.0 com sistemas de memória deployado!${NC}"