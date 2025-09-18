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
