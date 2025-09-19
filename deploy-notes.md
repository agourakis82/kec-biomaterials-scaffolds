Deploy notes — KEC_BIOMAT Actions

Objetivo: publicar o servidor FastAPI atrás de HTTPS (ex.: `https://api.agourakis.med.br`) e expor `openapi.yaml` publicamente para o ChatGPT Builder.

Script pronto (Cloud Run): `infra/actions/deploy/cloudrun_deploy.sh`
- Requer `GCP_PROJECT_ID` e opcionalmente `SERVICE_ACCOUNT`, `KEC_API_KEY_SECRET`, `SERVICE_NAME`
- Usa `server/Dockerfile` para construir e publicar a imagem

1) Processo de aplicação (uvicorn + systemd)

Crie um serviço systemd simples (ajuste caminhos):

```
[Unit]
Description=KEC_BIOMAT FastAPI
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/kec-biomaterials-scaffolds
Environment="KEC_API_KEY=change-me"
ExecStart=/opt/kec-biomaterials-scaffolds/.venv/bin/uvicorn server.fastapi_app:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Ative e inicie:
```
sudo systemctl daemon-reload
sudo systemctl enable kebiomat.service
sudo systemctl start kebiomat.service
```

2) Reverse proxy (Nginx + TLS)

Exemplo de bloco de servidor com Let's Encrypt (ajuste domínio e paths):

```
server {
  listen 443 ssl http2;
  server_name api.agourakis.med.br;

  ssl_certificate /etc/letsencrypt/live/api.agourakis.med.br/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/api.agourakis.med.br/privkey.pem;
  add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
  add_header X-Content-Type-Options nosniff;
  add_header X-Frame-Options DENY;
  add_header Referrer-Policy no-referrer;

  # Sirva o arquivo openapi.yaml de forma estática (opcional)
  location = /openapi.yaml {
    root /opt/kec-biomaterials-scaffolds;  # arquivo em /opt/kec-biomaterials-scaffolds/openapi.yaml
    default_type application/yaml;
    try_files $uri =404;
  }

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}

server {
  listen 80;
  server_name api.agourakis.med.br;
  return 301 https://$host$request_uri;
}
```

3) CORS

No app, o CORS está restrito a `https://chat.openai.com`. Para testar com outra origem (ex.: docs locais), adicione-a em `allow_origins` no `fastapi_app.py`.

4) Segurança

- Use `KEC_API_KEY` no ambiente do processo para exigir a chave em `X-API-Key`.
- Mantenha TLS atualizado e renove certificados com `certbot`/`acme`.
- Garanta que apenas portas 80/443 estejam expostas externamente.

5) Publicação do OpenAPI

- Alternativa A: servir `/openapi.yaml` de forma estática via Nginx (ver acima).
- Alternativa B: utilizar o endpoint `/openapi.yaml` do FastAPI (gera YAML a partir do JSON do app).
- Alternativa C: hospedar o arquivo em um bucket público (ex.: GCS/S3) sob HTTPS.
