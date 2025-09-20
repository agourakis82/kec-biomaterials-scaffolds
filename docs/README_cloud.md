# Cloud – GCP + Cloudflare

Infraestrutura recomendada (GCP):
- Cloud Run: API FastAPI `infra/darwin/api` (Autenticado por OIDC / API Key no gateway).
- BigQuery: armazenamento de eventos/discovery/corpus.
- Vertex AI: modelos/embedding quando aplicável.
- Cloud Scheduler: agendar Discovery periodicamente.

Domínio + SSL (Cloudflare):
- Aponte DNS para o endpoint (inicialmente DNS only) para emissão SSL.
- Após emitir o certificado, habilite proxy (nuvem laranja) e configure SSL Full/Strict.

Implantação (resumo):
1) Configure OIDC do GitHub → GCP (Workload Identity Federation).
2) Configure variáveis/segredos do repositório:
   - `GCP_OIDC` (Workload Identity Provider)
   - `GCP_SERVICE_ACCOUNT`
   - `GCP_PROJECT_ID`, `REGION`, `CLOUD_RUN_SERVICE`
3) Execute o workflow `CI/CD` com `deploy_api=true` (ou use PR com label `deploy-api`).

Notas:
- Não commit secrets. Use Secret Manager/Actions Secrets.
- Ajuste `NEXT_PUBLIC_DARWIN_URL` para o domínio final.
