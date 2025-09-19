#!/bin/bash
# Script de configuração rápida para integração ChatGPT Actions
# KEC_BIOMAT API

set -e

echo "🚀 Configuração Rápida - Integração ChatGPT Actions"
echo "=================================================="

# Verificar se gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI não encontrado. Instale em: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Solicitar informações do usuário
read -p "Digite seu GCP Project ID: " GCP_PROJECT_ID
read -p "Digite sua API Key secreta: " KEC_API_KEY
read -p "Região GCP (padrão: us-central1): " GCP_REGION
GCP_REGION=${GCP_REGION:-us-central1}

# Configurar gcloud
echo "🔧 Configurando gcloud..."
gcloud config set project $GCP_PROJECT_ID
gcloud config set region $GCP_REGION

# Ativar APIs necessárias
echo "🔌 Ativando APIs do GCP..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Exportar variáveis
export GCP_PROJECT_ID=$GCP_PROJECT_ID
export KEC_API_KEY=$KEC_API_KEY
export GCP_REGION=$GCP_REGION
export SERVICE_NAME="kec-actions-api"

echo "📝 Variáveis configuradas:"
echo "  GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "  KEC_API_KEY: $KEC_API_KEY"
echo "  GCP_REGION: $GCP_REGION"
echo "  SERVICE_NAME: $SERVICE_NAME"

# Copiar openapi.yaml para o diretório server
echo "📋 Preparando arquivos..."
cp openapi.yaml server/

# Fazer deploy
echo "🚀 Fazendo deploy no Cloud Run..."
bash infra/actions/deploy/cloudrun_deploy.sh

# Obter URL do serviço
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$GCP_REGION --format='value(status.url)')

echo ""
echo "✅ Deploy concluído!"
echo "🌐 URL do serviço: $SERVICE_URL"
echo ""
echo "📋 Próximos passos:"
echo "1. Configure um domínio customizado (opcional):"
echo "   gcloud run domain-mappings create --service=$SERVICE_NAME --domain=SEU_DOMINIO --region=$GCP_REGION"
echo ""
echo "2. Acesse o ChatGPT e crie uma nova Action:"
echo "   - Import from OpenAPI: $SERVICE_URL/openapi.yaml"
echo "   - Authentication: API Key = $KEC_API_KEY"
echo ""
echo "3. Teste a integração:"
echo "   curl -H 'X-API-Key: $KEC_API_KEY' $SERVICE_URL/health"
echo ""
echo "📖 Documentação completa: CHATGPT_INTEGRATION_GUIDE.md"</content>
<parameter name="filePath">/home/agourakis82/workspace/kec-biomaterials-scaffolds/setup_chatgpt_integration.sh