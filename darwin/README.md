# Darwin KEC Biomaterials Scaffolds - Deploy e Setup

## Visão Geral
Este projeto inclui um backend FastAPI com JAX e AutoGen, e um frontend Next.js. O setup local usa Docker Compose para backend, frontend, Redis e Ollama. O deploy é para GCP Cloud Run.

## Pré-requisitos
- Docker e Docker Compose instalados
- Node.js e npm para frontend
- Python 3.11 para backend
- gcloud CLI para deploy (autenticado com GCP)
- Arquivo .env com variáveis (copiado de .env.example): PROJECT_ID, REGION, API_KEYS, etc.

## Setup Local
1. Clone o repositório e navegue para o diretório `darwin/`.
2. Copie .env.example para .env e edite com suas chaves:
   ```
   cp .env.example .env
   ```
3. Execute o script de setup:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```
   - Instala dependências (pip para backend, npm para frontend).
   - Inicia serviços com `docker-compose.local.yml`: backend (porta 8080), frontend (porta 3000), Redis (6379), Ollama (11434).

4. Verifique health checks:
   ```
curl http://localhost:8090/healthz
   curl http://localhost:3000/api/health
   ```

5. Para parar os serviços:
   ```
   docker compose -f docker-compose.local.yml down
   ```

## Deploy no GCP Cloud Run
1. Configure gcloud:
   ```
   gcloud auth login
   gcloud config set project $PROJECT_ID
   gcloud config set run/region $REGION
   ```

2. Execute o script de deploy:
   ```
   chmod +x deploy.sh
   ./deploy.sh
   ```
   - Constrói e envia imagens para Container Registry.
   - Deploys backend (`kec-backend`) e frontend (`kec-frontend`) para Cloud Run.
   - Configura env vars, portas, memória/CPU, e health checks.

3. URLs após deploy:
   - Backend: https://kec-backend-$REGION.run.app
   - Frontend: https://kec-frontend-$REGION.run.app

4. Teste:
   ```
   curl https://kec-backend-$REGION.run.app/healthz
   ```

## Estrutura de Arquivos Criados
- `setup.sh`: Instala dependências e inicia local.
- `docker-compose.local.yml`: Orquestra serviços locais.
- `deploy.sh`: Deploys para Cloud Run.
- `darwin/backend/kec_unified_api/Dockerfile`: Multi-stage para backend (crie manualmente se necessário).
- Frontend usa `Dockerfile` existente para dev e `Dockerfile.production` para prod.

## Notas
- Para JAX com GPU, ajuste JAX_PLATFORM=cuda no .env e use Dockerfile.gpu se disponível.
- Ollama requer pull de modelos manualmente no container local: `docker exec -it <ollama-container> ollama pull <model>`.
- Para produção, configure secrets no GCP Secret Manager e service accounts.

Para mais detalhes, consulte os scripts e .env.example.
