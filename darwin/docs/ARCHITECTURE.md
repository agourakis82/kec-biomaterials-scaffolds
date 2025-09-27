# DARWIN Infrastructure Architecture

**Arquitetura Completa da Infraestrutura DARWIN Production-Ready**

---

## 🏗️ Visão Geral da Arquitetura

A plataforma DARWIN implementa uma arquitetura moderna, escalável e resiliente no Google Cloud Platform, otimizada para pesquisa científica em biomateriais com capacidades de IA avançada.

### 🎯 Princípios Arquiteturais

- **Cloud-Native:** Totalmente containerizada com Cloud Run
- **Microservices:** Componentes desacoplados e independentes
- **Infrastructure as Code:** Terraform para gestão completa
- **Security by Design:** Segurança integrada em todas as camadas
- **Observability:** Monitoramento e logging abrangentes
- **Auto-scaling:** Escalabilidade automática baseada em demanda

---

## 🏛️ Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 CLOUD CDN + LOAD BALANCER                      │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   SSL/TLS       │    │   Cloud Armor   │                    │
│  │  Certificates   │    │   WAF + DDoS    │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              FRONTEND (CLOUD RUN)                          │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   React     │  │   Next.js   │  │     PWA     │        │ │
│  │  │ TypeScript  │  │   Server    │  │  Features   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   APPLICATION LAYER                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               BACKEND (CLOUD RUN)                          │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   FastAPI   │  │     JAX     │  │  Knowledge  │        │ │
│  │  │  RESTful    │  │   Compute   │  │    Graph    │        │ │
│  │  │    API      │  │   Engine    │  │   Engine    │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Multi-AI  │  │    RAG+     │  │   Vector    │        │ │
│  │  │ Orchestrator│  │   Search    │  │   Search    │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     DATA LAYER                                  │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  POSTGRESQL     │  │  REDIS CACHE    │  │ CLOUD STORAGE   │ │
│  │                 │  │                 │  │                 │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │ │
│  │ │   pgvector  │ │  │ │  Session    │ │  │ │   Models    │ │ │
│  │ │  Extensions │ │  │ │   Cache     │ │  │ │   Storage   │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │ │
│  │                 │  │                 │  │                 │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │ │
│  │ │ Knowledge   │ │  │ │   Query     │ │  │ │  Documents  │ │ │
│  │ │   Graph     │ │  │   Cache     │ │  │ │   & Assets   │ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Networking Architecture

### VPC Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                      VPC NETWORK                                │
│                   (darwin-production-vpc)                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    PUBLIC SUBNET                           │ │
│  │                   (10.0.0.0/24)                            │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │ │
│  │  │    Cloud    │    │    Cloud    │    │   Global    │   │ │
│  │  │     NAT     │    │   Router    │    │ Load Balancer│   │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   PRIVATE SUBNET                           │ │
│  │                  (Services Range)                          │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │ │
│  │  │  Cloud Run  │    │   Cloud     │    │   Redis     │   │ │
│  │  │  Services   │    │     SQL     │    │ Memorystore │   │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                VPC CONNECTOR                               │ │
│  │                (10.8.0.0/28)                               │ │
│  │         Cloud Run ↔ VPC Communication                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Firewall Rules

| Rule Name | Direction | Protocol | Ports | Source | Target | Purpose |
|-----------|-----------|----------|-------|--------|---------|---------|
| allow-http | Ingress | TCP | 80 | 0.0.0.0/0 | web-servers | HTTP traffic |
| allow-https | Ingress | TCP | 443 | 0.0.0.0/0 | web-servers | HTTPS traffic |
| allow-health-check | Ingress | TCP | 8080,8000 | GCP Health Check IPs | services | Health checks |
| allow-database | Ingress | TCP | 5432 | backend-services | database | PostgreSQL |
| allow-redis | Ingress | TCP | 6379 | backend-services | redis | Redis cache |
| deny-all | Ingress | All | All | 0.0.0.0/0 | deny-all | Default deny |

---

## 🖥️ Compute Architecture

### Cloud Run Services

#### Backend Service
```yaml
Service: darwin-production-backend
Configuration:
  CPU: 2000m (2 vCPUs)
  Memory: 4Gi
  Min Instances: 2 (always warm)
  Max Instances: 20
  Concurrency: 100
  Timeout: 300s
  
Network:
  VPC Connector: darwin-production-connector
  Egress: private-ranges-only
  
Environment:
  - DATABASE_URL: (from Secret Manager)
  - REDIS_URL: (from Secret Manager)
  - JWT_SECRET: (from Secret Manager)
  - ENVIRONMENT: production
```

#### Frontend Service
```yaml
Service: darwin-production-frontend
Configuration:
  CPU: 1000m (1 vCPU)
  Memory: 2Gi
  Min Instances: 1
  Max Instances: 10
  Concurrency: 80
  Timeout: 60s
  
Network:
  VPC Connector: darwin-production-connector
  Egress: private-ranges-only
  
Environment:
  - NEXT_PUBLIC_API_URL: https://api.agourakis.med.br
  - NEXT_PUBLIC_FRONTEND_URL: https://darwin.agourakis.med.br
  - NODE_ENV: production
```

### Auto-scaling Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUTO-SCALING                              │
│                                                                 │
│  Backend Triggers:                                              │
│  • CPU > 70% → Scale up                                         │
│  • Memory > 80% → Scale up                                      │
│  • Request queue > 50 → Scale up                                │
│  • No traffic for 15min → Scale down                            │
│                                                                 │
│  Frontend Triggers:                                             │
│  • CPU > 60% → Scale up                                         │
│  • Concurrent requests > 60 → Scale up                          │
│  • No traffic for 10min → Scale down                            │
│                                                                 │
│  Scaling Limits:                                                │
│  • Backend: 2-20 instances                                      │
│  • Frontend: 1-10 instances                                     │
│  • Scale-up delay: 60s                                          │
│  • Scale-down delay: 300s                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Data Architecture

### Database Layer

#### PostgreSQL Configuration
```sql
-- Instance: darwin-production-db
-- Tier: db-n1-standard-2 (2 vCPUs, 7.5GB RAM)
-- Storage: 100GB SSD (auto-resize to 500GB)
-- Availability: Regional (High Availability)

-- Extensions Enabled:
CREATE EXTENSION vector;           -- Vector similarity search
CREATE EXTENSION pg_stat_statements; -- Query performance monitoring
CREATE EXTENSION pg_buffercache;   -- Buffer cache monitoring
CREATE EXTENSION pg_trgm;          -- Trigram matching for search
CREATE EXTENSION btree_gin;        -- GIN indexes for complex queries
CREATE EXTENSION uuid-ossp;        -- UUID generation

-- Performance Optimizations:
shared_preload_libraries = 'vector,pg_stat_statements,pg_buffercache'
max_connections = 200
shared_buffers = 512MB
effective_cache_size = 2GB
maintenance_work_mem = 128MB
```

#### Data Schema
```sql
-- Core Tables
users                 -- User management
documents             -- Document storage with embeddings
research_papers       -- Scientific papers with vectors
knowledge_graph       -- Entity relationships
chat_sessions         -- Conversation history

-- Indexes for Performance
idx_documents_embedding    -- Vector similarity search
idx_papers_embedding       -- Research paper vectors
idx_knowledge_embedding    -- Knowledge graph vectors
idx_documents_content_gin  -- Full-text search
idx_papers_abstract_gin    -- Abstract search
```

### Redis Cache Layer

```yaml
Instance: darwin-production-redis
Configuration:
  Memory: 2GB
  Version: REDIS_6_X
  Network: Private VPC
  Auth: Enabled
  SSL: Server-Client encryption
  
Cache Strategy:
  Session Data: TTL 24h
  Query Results: TTL 1h
  Vector Embeddings: TTL 6h
  API Responses: TTL 30min
  
Eviction Policy: allkeys-lru
```

### Storage Architecture

```
Cloud Storage Buckets:
┌─────────────────────────────────────────────────────────────────┐
│  darwin-production-uploads     │ User file uploads              │
│  darwin-production-documents   │ Processed documents            │
│  darwin-production-models      │ AI models and weights          │
│  darwin-production-backups     │ Database and config backups    │
│  darwin-production-logs        │ Application and audit logs     │
│  darwin-production-cdn-assets  │ CDN-optimized static assets    │
│  darwin-production-static      │ Static files and images        │
└─────────────────────────────────────────────────────────────────┘

Lifecycle Policies:
• Uploads: 90 days → NEARLINE → COLDLINE
• Documents: 365 days retention
• Models: Never delete (long-term storage)
• Backups: 90 days → COLDLINE
• CDN Assets: 30 days → DELETE
```

---

## 🔐 Security Architecture

### Identity and Access Management

```
Service Accounts (Least Privilege):
┌─────────────────────────────────────────────────────────────────┐
│  darwin-production-backend-sa:                                 │
│  • cloudsql.client                                             │
│  • redis.editor                                                │
│  • storage.objectAdmin                                         │
│  • secretmanager.secretAccessor                                │
│  • aiplatform.user                                             │
│                                                                 │
│  darwin-production-frontend-sa:                                │
│  • storage.objectViewer (CDN buckets only)                     │
│  • logging.logWriter                                           │
│                                                                 │
│  darwin-production-terraform-sa:                               │
│  • Project Editor (infrastructure management)                  │
│  • Service Account Admin                                       │
│                                                                 │
│  darwin-production-monitoring-sa:                              │
│  • monitoring.admin                                            │
│  • logging.admin                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Encryption Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENCRYPTION LAYERS                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 ENCRYPTION AT REST                         │ │
│  │                                                             │ │
│  │  Database: Customer-managed KMS keys                       │ │
│  │  Storage: Customer-managed KMS keys                        │ │
│  │  Secrets: Automatic encryption                             │ │
│  │  Backups: Customer-managed KMS keys                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                ENCRYPTION IN TRANSIT                       │ │
│  │                                                             │ │
│  │  Client ↔ Load Balancer: TLS 1.2+                         │ │
│  │  Load Balancer ↔ Services: TLS 1.2+                       │ │
│  │  Services ↔ Database: SSL required                         │ │
│  │  Services ↔ Redis: TLS encryption                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              APPLICATION-LEVEL ENCRYPTION                  │ │
│  │                                                             │ │
│  │  JWT Tokens: RS256 signatures                              │ │
│  │  Sensitive Data: AES-256 field-level encryption            │ │
│  │  API Keys: Secure Secret Manager storage                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Network Security

```
Security Layers:
┌─────────────────────────────────────────────────────────────────┐
│                   CLOUD ARMOR (Layer 7)                        │
│  • WAF Rules: SQL injection, XSS protection                    │
│  • Rate Limiting: 1000 req/min per IP                          │
│  • DDoS Protection: Adaptive protection                        │
│  • Geo-blocking: Optional country restrictions                 │
└─────────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────────────────────────────────────────┐
│                  VPC FIREWALL (Layer 4)                        │
│  • Default Deny All                                            │
│  • Explicit Allow Rules:                                       │
│    - HTTP/HTTPS from internet                                  │
│    - Database access from backend only                         │
│    - Redis access from backend only                            │
│    - Health checks from GCP ranges                             │
└─────────────────────────────────────────────────────────────────┘
│
┌─────────────────────────────────────────────────────────────────┐
│                 PRIVATE NETWORKING                              │
│  • Private Google Access: Enabled                              │
│  • Service Networking: Private IP only                         │
│  • VPC Peering: Isolated networks                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Monitoring Architecture

### Observability Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      MONITORING STACK                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 CLOUD MONITORING                           │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Custom      │  │   Alert     │  │   Uptime    │        │ │
│  │  │ Dashboards  │  │  Policies   │  │   Checks    │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  CLOUD LOGGING                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │Application  │  │   Audit     │  │  Security   │        │ │
│  │  │    Logs     │  │    Logs     │  │    Logs     │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   CLOUD TRACE                              │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Distributed │  │  Latency    │  │ Performance │        │ │
│  │  │   Tracing   │  │  Analysis   │  │  Profiling  │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics Monitored

#### 📈 Performance Metrics
- **Response Time:** P50, P95, P99 latencies
- **Throughput:** Requests per second
- **Error Rate:** 4xx/5xx response codes
- **Availability:** Uptime percentage
- **Resource Usage:** CPU, Memory, Disk utilization

#### 🔒 Security Metrics
- **Failed Authentication:** Login attempts
- **Suspicious Activity:** Unusual access patterns
- **Vulnerability Scans:** Container and dependency scans
- **Policy Violations:** Access control violations

#### 💰 Cost Metrics
- **Monthly Spend:** Current vs budgeted
- **Resource Costs:** Per-service breakdown
- **Cost Trends:** Usage patterns and optimization opportunities

---

## 🔄 CI/CD Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       SOURCE CODE                              │
│                    (Git Repository)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD BUILD                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              INFRASTRUCTURE PIPELINE                       │ │
│  │                                                             │ │
│  │  Security Scan → Terraform Plan → Apply → Verify           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                BACKEND PIPELINE                            │ │
│  │                                                             │ │
│  │  Test → Security → Build → Scan → Deploy → Health Check    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               FRONTEND PIPELINE                            │ │
│  │                                                             │ │
│  │  Test → Build → Optimize → Deploy → CDN → Health Check     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INTEGRATION TESTING                            │
│                                                                 │
│  Unit Tests → Integration → Load → Security → E2E → Report     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PRODUCTION DEPLOYMENT                          │
│              https://darwin.agourakis.med.br                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Performance Architecture

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      CACHING LAYERS                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   CLOUD CDN                                │ │
│  │  • Static Assets: 1 year cache                             │ │
│  │  • API Responses: No cache                                 │ │
│  │  • HTML Files: No cache                                    │ │
│  │  • Global Edge Locations                                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                APPLICATION CACHE                           │ │
│  │  • Redis: Query results, sessions                          │ │
│  │  • Memory: In-process caching                              │ │
│  │  • Database: Connection pooling                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 DATABASE CACHE                             │ │
│  │  • PostgreSQL: Shared buffers, query cache                 │ │
│  │  • Read Replicas: Distributed read operations              │ │
│  │  • Connection Pool: Persistent connections                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Load Balancing

```
Request Flow:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Client   │───▶│   Global    │───▶│  Regional   │
│   Request   │    │     LB      │    │   Backend   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │     SSL Termination     │
              │     Security Policies   │
              │     Rate Limiting       │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │    Cloud Run Services   │
              │   (Auto-scaled fleet)   │
              └─────────────────────────┘
```

---

## 🧠 AI/ML Architecture

### JAX Compute Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                      JAX ARCHITECTURE                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 COMPUTE BACKENDS                           │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │     CPU     │  │     GPU     │  │     TPU     │        │ │
│  │  │   Default   │  │  Optional   │  │  Optional   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               VECTOR OPERATIONS                            │ │
│  │                                                             │ │
│  │  • Similarity Search: Cosine, L2, Inner Product            │ │
│  │  • Embeddings: Text, Image, Scientific Data                │ │
│  │  • Knowledge Graph: Vector-based relationships             │ │
│  │  • Multi-modal: Cross-domain vector search                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### AI Model Integration

- **Multi-AI Chat:** OpenAI, Anthropic, Google AI integration
- **Vector Search:** pgvector with optimized indexes
- **Knowledge Graph:** Entity relationships with embeddings
- **RAG+ Engine:** Retrieval-Augmented Generation
- **Scientific Discovery:** Domain-specific AI workflows

---

## 📡 API Architecture

### RESTful API Design

```
API Structure:
/api/
├── health/                 # Health checks
├── auth/                   # Authentication
├── users/                  # User management
├── documents/              # Document operations
├── search/                 # Vector and text search
├── knowledge-graph/        # Knowledge graph operations
├── multi-ai/              # AI model interactions
├── rag/                   # RAG+ operations
├── discovery/             # Scientific discovery
└── admin/                 # Administrative functions

Authentication:
• JWT tokens with RS256 signatures
• Service-to-service: Google Cloud IAM
• API keys for external integrations
• Rate limiting per authenticated user
```

### API Performance

- **Response Times:** < 2s for 95% of requests
- **Throughput:** 1000+ requests/minute
- **Availability:** 99.9% uptime target
- **Caching:** Intelligent query result caching
- **Compression:** gzip for all text responses

---

## 🔄 Backup and Disaster Recovery

### Backup Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                     BACKUP STRATEGY                            │
│                                                                 │
│  Database Backups:                                              │
│  • Automated daily backups at 03:00 UTC                        │
│  • Point-in-time recovery (7 days)                             │
│  • Cross-region backup replication                             │
│  • 30-day retention policy                                     │
│                                                                 │
│  Configuration Backups:                                         │
│  • Terraform state in versioned GCS bucket                     │
│  • Infrastructure code in Git repository                       │
│  • Environment configs versioned                               │
│                                                                 │
│  Application Backups:                                           │
│  • Container images in Container Registry                      │
│  • Static assets replicated across regions                     │
│  • User data in encrypted storage buckets                      │
└─────────────────────────────────────────────────────────────────┘
```

### Disaster Recovery

**Recovery Time Objectives (RTO):**
- **Infrastructure:** 2 hours (Terraform redeploy)
- **Applications:** 30 minutes (Cloud Run auto-healing)
- **Database:** 1 hour (automated failover)
- **DNS/CDN:** 15 minutes (global propagation)

**Recovery Point Objectives (RPO):**
- **Database:** 1 hour (point-in-time recovery)
- **File Storage:** 24 hours (daily snapshots)
- **Configuration:** 0 (Git version control)

---

## 📈 Scalability Architecture

### Horizontal Scaling

```
Scaling Dimensions:
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE SCALING                             │
│                                                                 │
│  Cloud Run Auto-scaling:                                        │
│  • Backend: 2-20 instances                                     │
│  • Frontend: 1-10 instances                                    │
│  • Scaling triggers: CPU, memory, request queue                │ │
│  • Cold start optimization: Min instances always warm          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DATA SCALING                               │
│                                                                 │
│  Database Scaling:                                              │
│  • Primary: High-performance instance                          │
│  • Read Replicas: Auto-created based on load                   │
│  • Connection Pooling: Optimized pool sizes                    │
│  • Storage: Auto-resize based on usage                         │
│                                                                 │
│  Cache Scaling:                                                 │
│  • Redis: Memory auto-scaling                                  │
│  • CDN: Global edge caching                                    │
│  • Application: In-memory caches                               │
└─────────────────────────────────────────────────────────────────┘
```

### Vertical Scaling

- **Database:** db-n1-standard-2 → db-n1-standard-4 (automated)
- **Redis:** 2GB → 4GB memory (as needed)
- **Cloud Run:** CPU/Memory limits adjustable
- **Storage:** Automatic disk resizing

---

## 🔧 DevOps Architecture

### Infrastructure as Code

```
Terraform Modules:
infrastructure/terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Input variables
├── outputs.tf             # Output values
├── versions.tf            # Provider versions
├── backend.tf             # Remote state
└── modules/
    ├── networking/         # VPC, LB, SSL
    ├── backend/           # Cloud Run, DB, Redis
    ├── frontend/          # Cloud Run, CDN
    ├── monitoring/        # Dashboards, alerts
    └── security/          # IAM, KMS, policies
```

### Configuration Management

```
Environment Configs:
config/environments/
├── production.yaml         # Production settings
├── staging.yaml           # Staging settings
├── dev.yaml              # Development settings
└── terraform.tfvars      # Terraform variables
```

---

## 🎯 High Availability

### Multi-Region Setup (Optional)

```
Primary Region: us-central1
└── All services active

Secondary Region: us-east1 (Disaster Recovery)
└── Passive standby

Cross-Region Resources:
• Global Load Balancer
• Cloud CDN (global edge locations)
• Multi-region storage buckets
• Cross-region database backups
```

### Fault Tolerance

- **Load Balancer:** Multi-zone distribution
- **Cloud Run:** Automatic instance replacement
- **Database:** Regional deployment with automatic failover
- **Storage:** Multi-zone replication
- **Monitoring:** Multiple alert channels

---

## 🔍 Troubleshooting Architecture

### Debugging Flow

```
Issue Detection:
┌─────────────────────────────────────────────────────────────────┐
│  Monitoring Alerts → Incident Response → Root Cause Analysis   │
│           │                    │                    │          │
│           ▼                    ▼                    ▼          │
│     Uptime Checks      Performance Metrics    Error Logs       │
│     SLO Violations     Resource Alerts       Audit Trails      │
└─────────────────────────────────────────────────────────────────┘

Debug Resources:
• Cloud Logging: Centralized log aggregation
• Cloud Trace: Distributed tracing
• Cloud Debugger: Live application debugging
• Cloud Profiler: Performance profiling
• Error Reporting: Automatic error detection
```

---

## 🏷️ Resource Naming Convention

### Naming Pattern
```
{project}-{environment}-{component}-{resource-type}

Examples:
• darwin-production-backend-service
• darwin-production-db-instance
• darwin-staging-vpc-network
• darwin-dev-redis-cache
```

### Labels Strategy
```yaml
Common Labels:
  project: darwin
  environment: production|staging|dev
  team: kec-biomaterials
  cost-center: research
  managed-by: terraform
  component: backend|frontend|database|cache
```

---

## 📚 Additional Resources

### 📖 Documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md):** This document
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md):** Problem resolution
- **[API_REFERENCE.md](API_REFERENCE.md):** API documentation
- **[SECURITY.md](docs/SECURITY.md):** Security policies

### 🔗 External Resources
- **Terraform Google Provider:** https://registry.terraform.io/providers/hashicorp/google
- **Cloud Run Documentation:** https://cloud.google.com/run/docs
- **PostgreSQL pgvector:** https://github.com/pgvector/pgvector
- **JAX Documentation:** https://jax.readthedocs.io

---

**Esta arquitetura foi projetada para ser:**
- ✅ **Production-Ready:** Alta disponibilidade e performance
- ✅ **Secure:** Múltiplas camadas de segurança
- ✅ **Scalable:** Auto-scaling em todas as dimensões
- ✅ **Observable:** Monitoramento e alerting completos
- ✅ **Cost-Effective:** Otimizada para custos operacionais
- ✅ **Maintainable:** Infrastructure as Code e automação

Para questões arquiteturais específicas, consulte a equipe de DevOps ou abra uma issue no repositório.