# DARWIN - Meta-Research Brain for Biomaterials Science

DARWIN is a hybrid local-first AI platform for biomaterials research, combining FastAPI backend with Next.js frontend, Redis caching, and optional Ollama/Chroma for local AI processing. It supports Cloudflare Tunnel for secure remote access and GCP integration for production workloads.

## Architecture

- **Backend**: FastAPI with RAG++/KEC multi-IA, JAX acceleration, Q1 Scholar plugin
- **Frontend**: Next.js with TypeScript, Tailwind CSS
- **Database**: Redis for caching, ChromaDB for vector storage
- **AI**: Hybrid local/web with Vertex AI and OpenAI fallbacks
- **Deployment**: Docker Compose (local) + Cloud Run (GCP production)

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)
- Cloudflare account (for tunnel)

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd kec-biomaterials-scaffolds
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Start full stack**:
   ```bash
   make compose-up
   # Or: docker compose up -d
   ```

3. **Access services**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8090/healthz
   - Jupyter: http://localhost:8888
   - Redis: localhost:6379 (internal only)

### With Cloudflare Tunnel

1. **Setup tunnel**:
   ```bash
   cloudflared login
   cloudflared tunnel create darwin
   # Update .cloudflared/config.yml with tunnel ID
   ```

2. **Start with tunnel**:
   ```bash
   make up-all
   # Or: make compose-up && make tunnel-up
   ```

3. **Public URLs** (after DNS propagation):
   - Frontend: https://darwin-local.agourakis.med.br
   - Backend: https://api-local.agourakis.med.br/healthz

## Environment Variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `local` | Environment (local/production) |
| `HOST_BACKEND_PORT` | `8090` | Backend external port |
| `HOST_FRONTEND_PORT` | `3000` | Frontend external port |
| `JUPYTER_PORT` | `8888` | Jupyter port |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `CHROMA_PORT` | `8000` | ChromaDB port |

### AI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_AI_DEFAULT_PROVIDER` | `openai` | Primary AI provider |
| `ENABLE_RAG_PLUS` | `true` | Enable RAG++ features |
| `ENABLE_MULTI_AI` | `true` | Enable multi-AI routing |
| `ENABLE_KEC_METRICS` | `true` | Enable KEC analysis |
| `CONFIG_FILE` | `/app/config/production.yaml` | YAML config path |

### Cloudflare Tunnel

| Variable | Default | Description |
|----------|---------|-------------|
| `TUNNEL_CREDENTIALS_FILE` | `/etc/cloudflared/darwin.json` | Tunnel credentials |

## YAML Configuration (production.yaml)

```yaml
ai:
  providers:
    vertex_ai:
      enabled: true
      project_id: "pcs-helio"
      location: "us-central1"
      model: "gemini-1.5-pro"
    openai:
      enabled: true
      model: "gpt-4o"
  routing:
    local_first: true
    timeout_ms: 30000
    retries: 2
    fallback_chain: ["ollama", "vertex_ai", "openai"]

performance:
  jax_platform: "cpu"  # or "cuda" for GPU
  batch_size: 32
  cache_ttl_seconds: 3600

features:
  rag_plus: true
  kec_metrics: true
  scientific_discovery: true
  score_contracts: true
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make compose-up` | Start all services |
| `make compose-down` | Stop all services |
| `make compose-up-gpu` | Start with GPU backend |
| `make tunnel-up` | Start Cloudflare tunnel |
| `make smoke-local` | Run local health checks |
| `make smoke-public` | Run public health checks |
| `make smoke-q1` | Check Q1 Scholar health |

## VSCode Development

### Tasks (Ctrl+Shift+P > Tasks: Run Task)

- `compose:build` - Build all images
- `compose:up` - Start services
- `compose:down` - Stop services
- `compose:logs` - View logs
- `tunnel:up` - Start tunnel
- `smoke:local` - Local tests
- `smoke:public` - Public tests

### Debugging

- Backend: Launch `Debug API (FastAPI)`
- Frontend: Launch `Debug Web (Next.js)`

## Health Checks

### Local
```bash
curl http://localhost:8090/healthz
curl http://localhost:3000/api/health
curl http://localhost:8090/q1-scholar/health
```

### Public (with tunnel)
```bash
curl https://api-local.agourakis.med.br/healthz
curl https://darwin-local.agourakis.med.br/api/health
curl https://api-local.agourakis.med.br/q1-scholar/health
```

## GPU Support

For NVIDIA GPU acceleration:

1. Install NVIDIA Container Toolkit
2. Set `JAX_PLATFORM=cuda` in environment
3. Use `make compose-up-gpu`

## Production Deployment

### GCP Cloud Run

```bash
# Deploy via Cloud Build
gcloud builds submit --config cloudbuild-darwin-complete.yaml

# Or use Terraform
cd infrastructure/terraform
terraform apply
```

### Manual Docker

```bash
# Build and push
docker build -f darwin/backend/kec_unified_api/Dockerfile -t gcr.io/pcs-helio/darwin-backend .
docker build -f darwin/frontend/ui/Dockerfile.production -t gcr.io/pcs-helio/darwin-frontend .

# Deploy to Cloud Run
gcloud run deploy darwin-backend --image gcr.io/pcs-helio/darwin-backend --platform managed
gcloud run deploy darwin-frontend --image gcr.io/pcs-helio/darwin-frontend --platform managed
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Use `HOST_BACKEND_PORT` and `HOST_FRONTEND_PORT`
2. **GPU not detected**: Check NVIDIA drivers and toolkit
3. **Tunnel not working**: Verify Cloudflare credentials and DNS
4. **AI timeouts**: Adjust YAML timeouts or enable fallbacks

### Logs

```bash
# All services
make compose-logs

# Specific service
docker compose logs api

# Tunnel logs
make tunnel-logs
```

## Contributing

1. Create feature branch: `git checkout -b feat/your-feature`
2. Make changes with tests
3. Run smoke tests: `make smoke-all`
4. Submit PR

## License

See LICENSE file.

## Cloudflare Zero Trust (Optional)

For enhanced security and access control, configure Cloudflare Zero Trust:

### Setup Steps

1. **Create Zero Trust Account**:
   ```bash
   # Login to Cloudflare Zero Trust
   cloudflared login
   ```

2. **Create Application**:
   - Go to Zero Trust Dashboard > Access > Applications
   - Create new application with:
     - Application type: Self-hosted
     - Domain: `darwin-local.agourakis.med.br`
     - Policies: Configure user/group access rules

3. **Configure Policies**:
   ```yaml
   # Example policy for development access
   - name: "DARWIN Dev Access"
     include:
       - email_domain: "yourdomain.com"
     require:
       - device_posture: "managed"
   ```

4. **Update Tunnel Config**:
   ```yaml
   # .cloudflared/config.yml
   tunnel: darwin
   credentials-file: /etc/cloudflared/darwin.json

   ingress:
     - hostname: darwin-local.agourakis.med.br
       originRequest:
         noTLSVerify: false
       service: https://localhost:3000
     - hostname: api-local.agourakis.med.br
       originRequest:
         noTLSVerify: false
       service: https://localhost:8090
     - service: http_status:404
   ```

5. **Enable Device Posture** (Recommended):
   - Install WARP on user devices
   - Configure device posture checks
   - Require managed devices for access

### Benefits

- **Zero Trust Security**: Never trust, always verify
- **Device Posture**: Ensure devices meet security requirements
- **Identity-Based Access**: Control access per user/group
- **Audit Logs**: Track all access attempts
- **Conditional Access**: Time-based, location-based policies
