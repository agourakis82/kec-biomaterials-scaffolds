# Guia de Integração ChatGPT Actions - KEC_BIOMAT

## 📋 Pré-requisitos

### 1. Conta Google Cloud Platform (GCP)
- Acesse [Google Cloud Console](https://console.cloud.google.com/)
- Crie um novo projeto ou selecione um existente
- Ative o Cloud Run API e Cloud Build API

### 2. Google Cloud SDK
```bash
# Instalar gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 3. Configurar Projeto GCP
```bash
# Definir projeto padrão
export GCP_PROJECT_ID="seu-projeto-gcp"
gcloud config set project $GCP_PROJECT_ID

# Ativar APIs necessárias
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## 🚀 Deploy da API

### 1. Configurar Variáveis de Ambiente
```bash
# Chave API para proteger os endpoints
export KEC_API_KEY="sua-chave-api-secreta-aqui"

# Configurações opcionais
export GCP_REGION="us-central1"
export SERVICE_NAME="kec-actions-api"
```

### 2. Fazer Deploy no Cloud Run
```bash
# Executar script de deploy
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds
bash infra/actions/deploy/cloudrun_deploy.sh
```

### 3. Verificar Deploy
```bash
# Obter URL do serviço
gcloud run services describe $SERVICE_NAME --region=$GCP_REGION --format='value(status.url)'
```

## 🔗 Configurar Domínio HTTPS (Opcional)

### 1. Configurar Domínio Customizado
```bash
# Mapear domínio para o serviço Cloud Run
gcloud run domain-mappings create \
  --service=$SERVICE_NAME \
  --domain=api.agourakis.med.br \
  --region=$GCP_REGION
```

### 2. Configurar DNS
- Acesse seu provedor de DNS
- Adicione registro CNAME apontando para o domínio fornecido pelo Cloud Run

## 🤖 Configurar ChatGPT Action

### 1. Acessar ChatGPT
- Acesse [ChatGPT](https://chat.openai.com/)
- Vá para "Explore" → "Create a GPT" → "Configure actions"

### 2. Importar OpenAPI Specification
- Selecione "Import from OpenAPI"
- URL: `https://api.agourakis.med.br/openapi.yaml`
- Ou faça upload do arquivo `openapi.yaml`

### 3. Configurar Autenticação
- Authentication Type: "API Key"
- API Key: Seu `KEC_API_KEY`
- Header Name: `X-API-Key`

### 4. Testar Conexão
- ChatGPT irá validar a conexão automaticamente
- Deve mostrar os endpoints disponíveis: `/health`, `/kec/compute`, `/jobs/{id}`

## 🧪 Testes

### 1. Teste Local (antes do deploy)
```bash
# Iniciar servidor local
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds
python -c "
from fastapi.testclient import TestClient
from server.fastapi_app import app

client = TestClient(app)

# Teste health
response = client.get('/health')
print(f'Health: {response.status_code} - {response.json()}')

# Teste compute (com API key)
headers = {'X-API-Key': 'dev-12345'}
data = {'graph_id': 'demo-001', 'sigma_q': False}
response = client.post('/kec/compute', json=data, headers=headers)
print(f'Compute: {response.status_code} - {response.json()}')
"
```

### 2. Teste Produção
```bash
# Health check
curl -s https://api.agourakis.med.br/health

# Compute request
curl -s -H "X-API-Key: $KEC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"demo-001","sigma_q":false}' \
  https://api.agourakis.med.br/kec/compute
```

## 📝 Endpoints Disponíveis

### GET /health
- **Descrição**: Verifica status da API
- **Resposta**: `{"status": "ok"}`

### POST /kec/compute
- **Descrição**: Executa cálculo de métricas KEC
- **Autenticação**: `X-API-Key` obrigatória
- **Request Body**:
  ```json
  {
    "graph_id": "string",
    "sigma_q": false
  }
  ```
- **Resposta**:
  ```json
  {
    "H_spectral": 0.0,
    "k_forman_mean": 0.0,
    "sigma": 0.0,
    "swp": 0.0
  }
  ```

### GET /jobs/{id}
- **Descrição**: Consulta status de job assíncrono
- **Autenticação**: `X-API-Key` obrigatória
- **Resposta**:
  ```json
  {
    "id": "string",
    "status": "completed|running|failed",
    "result": {...}
  }
  ```

## 🔧 Troubleshooting

### Problema: "Could not find a valid URL in servers"
- Verifique se o domínio está acessível via HTTPS
- Confirme se o arquivo `openapi.yaml` está disponível publicamente

### Problema: Erro de autenticação
- Verifique se a `X-API-Key` está correta
- Confirme se o header está sendo enviado como `X-API-Key`

### Problema: Build falha no Docker
- Verifique conectividade de rede
- Tente novamente ou use `--no-cache` no build

### Problema: Serviço não inicia no Cloud Run
- Verifique logs: `gcloud logs read`
- Confirme se todas as variáveis de ambiente estão definidas

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs do Cloud Run
2. Teste localmente primeiro
3. Consulte a documentação do ChatGPT Actions
4. Abra uma issue no repositório

---

**Status**: ✅ Preparado para integração
**Última atualização**: $(date)
**Versão da API**: 2025-09-19</content>
<parameter name="filePath">/home/agourakis82/workspace/kec-biomaterials-scaffolds/CHATGPT_INTEGRATION_GUIDE.md