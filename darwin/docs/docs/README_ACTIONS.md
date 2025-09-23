KEC_BIOMAT — GPT Actions (Caminho A)

Pacote mínimo para integrar `https://api.agourakis.med.br` ao ChatGPT (Actions) via OpenAPI.
Inclui:
- `openapi.yaml` (OpenAPI 3.1, com `servers.url` em HTTPS)
- Servidor FastAPI de referência em `server/fastapi_app.py` (stubs compatíveis)
- `server/requirements.txt`
- Testes via `curl`
- Checklist de prontidão

Requisitos: Python 3.11+.

1) Subir o servidor local (opcional)

```
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
export KEC_API_KEY="dev-12345"  # defina para exigir a chave; se não definir, as rotas protegidas liberam no dev
uvicorn server.fastapi_app:app --reload
```

Verifique docs e schema:
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- OpenAPI YAML: http://127.0.0.1:8000/openapi.yaml

Atalho via Makefile:

```
make actions-run
```

2) Publicar o OpenAPI (produção)

Há duas opções válidas para o ChatGPT Builder (Actions → Import from OpenAPI):
- Disponibilizar arquivo em URL público HTTPS, ex.: `https://api.agourakis.med.br/openapi.yaml` (ideal)
- Fazer upload do arquivo `openapi.yaml` diretamente no Builder

O arquivo fornecido neste repo já define:
```
servers:
  - url: https://api.agourakis.med.br
    description: Production
```
Se o Builder mostrar “Could not find a valid URL in servers”, confirme que o domínio está acessível em HTTPS.

3) Importar no ChatGPT (Actions)

No ChatGPT:
- Create → Build → Actions → Import from OpenAPI
- Selecione a opção de URL (ex.: `https://api.agourakis.med.br/openapi.yaml`) ou faça upload do arquivo
- Confirme que os endpoints aparecem (`/health`, `/kec/compute`, `/jobs/{id}`)
- Em Authentication, selecione “Custom API Key” e configure o header `X-API-Key`

4) Testes rápidos (curl)

Produção (ajuste o domínio se necessário):
```
curl -s https://api.agourakis.med.br/health

curl -s -H "X-API-Key: $KEC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"demo-001","sigma_q":false}' \
  https://api.agourakis.med.br/kec/compute
```

Local (se subiu via `uvicorn`):
```
curl -s http://127.0.0.1:8000/health

curl -s -H "X-API-Key: $KEC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"demo-001","sigma_q":false}' \
  http://127.0.0.1:8000/kec/compute
```

5) Endpoints (stubs)

- `GET /health` → `{ "status": "ok" }`
- `POST /kec/compute` (protegido por `X-API-Key`) → `{ H_spectral, k_forman_mean, sigma, swp }`
- `GET /jobs/{id}` (protegido por `X-API-Key`) → `{ id, status, result? }` (status pseudo-determinístico)

Observação: O servidor em `server/fastapi_app.py` contém comentários indicando onde plugar o cálculo real de KEC.

6) Checklist de prontidão

- `servers.url` em `openapi.yaml` aponta para HTTPS público e válido.
- `GET /health` responde 200 com `{"status":"ok"}`.
- `GET /openapi.json` e/ou arquivo `openapi.yaml` acessível publicamente.
- Header `X-API-Key` aceito nos endpoints protegidos.
- Swagger UI acessível em `/docs` (ambiente de validação).
- CORS conforme necessário (por padrão, liberado para `https://chat.openai.com`).

7) Dicas rápidas de deploy

Veja `deploy-notes.md` para exemplo de Nginx (reverse proxy com TLS), cabeçalhos de segurança, systemd e publicação do `openapi.yaml` em `/openapi.yaml`.

8) Docker + Cloud Run

- Build local da imagem: `make actions-build`
- Teste em container: `make actions-docker-run`
- Deploy Cloud Run: `bash infra/actions/deploy/cloudrun_deploy.sh` (usa `server/Dockerfile`)

Variáveis úteis antes do deploy:

```
export GCP_PROJECT_ID="meu-projeto"
export SERVICE_ACCOUNT="svc-actions@meu-projeto.iam.gserviceaccount.com"  # opcional
export KEC_API_KEY_SECRET="kec_actions_api_key:latest"  # ou defina KEC_API_KEY
export SERVICE_NAME="kec-actions-api"  # opcional
```

O script publica a imagem com Cloud Build e cria/atualiza o serviço Cloud Run apontando para `https://api.agourakis.med.br` (ajuste DNS/SSL conforme `deploy-notes.md`).
