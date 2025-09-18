# Modo B - Visao Geral Cloud

## Arquitetura
```
+-----------------+        +----------------------+        +-----------------+
|  Cloud Scheduler| -----> | Cloud Run Job (RAG+) | -----> | BigQuery (docs) |
+-----------------+        +----------------------+        +-----------------+
           |                                                      ^
           v                                                      |
   Pub/Sub (futuro)      +----------------------+        +---------------------+
           |             | Cloud Run API (RAG+) | <----> | Vertex AI (LLM/Emb) |
           v             +----------------------+        +---------------------+
     Secret Manager ----------------------------- stores keys & configs
```

## Variaveis Obrigatorias
- `GCP_PROJECT_ID=pcs-helio`
- `GCP_REGION=us-central1`
- `GCP_SA_NAME=darwin-runner`
- `BQ_DATASET=darwin_kg`
- `BQ_TABLE=documents`
- `DARWIN_BASE_URL=https://PREENCHER_DOMINIO`

## Referencias
- Documentacao OpenAPI: `GET /openapi.json` na URL publica da API Cloud Run.
- Acoes ChatGPT devem apontar para o dominio configurado acima.

## Runbook
1. `export GCP_PROJECT_ID=... GCP_REGION=us-central1 GCP_SA_NAME=darwin-runner BQ_DATASET=darwin_kg BQ_TABLE=documents`
2. `bash infra/darwin/cloud/gcp/bootstrap.sh`
3. `bash infra/darwin/cloud/gcp/secrets.sh`
4. (local) `pip install -r infra/darwin/requirements.txt && uvicorn api.app:app --reload`
5. `bash infra/darwin/cloud/gcp/deploy.sh` → copie `RUN_URL=...`
6. `RUN_URL=... DARWIN_API_KEY=... bash infra/darwin/cloud/tests/smoke.sh`
7. `bash infra/darwin/cloud/gcp/schedule.sh` (liga discovery de hora em hora)
8. Atualize ChatGPT Actions com a URL publica e o bearer.
