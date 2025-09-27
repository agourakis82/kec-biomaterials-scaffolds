# Análise Completa de Conflitos de Dependências - Backends KEC Biomaterials

## 1. RESUMO EXECUTIVO DOS BACKENDS

### Backends Identificados:
1. **Backend Principal**: `src/kec_biomat_api/` (Ativo - 13 routers)
2. **Darwin RAG+**: `infra/darwin/api/` (Ativo - RAG especializado)  
3. **Actions**: `server/` (Quebrado - spec only)
4. **Ingestão Job**: `kec_biomat_ingest/` (Ativo - pipeline)
5. **Frontend APIs**: `ui/src/app/api/` (Ativo - proxy layer)

### Análise de Dockerfiles:
- **Python Base**: Todos os backends Python usam `python:3.11-slim` ✅ **COMPATÍVEL**
- **Node.js Base**: Frontend usa `node:18-alpine` ✅ **ISOLADO**

## 2. MAPEAMENTO DE DEPENDÊNCIAS POR CATEGORIA

### 2.1 FRAMEWORK WEB & ASGI

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **FastAPI** | `sem versão` | `==0.104.1` | `>=0.115.0,<1.0.0` | `não usa` | `sem versão` |
| **Uvicorn** | `sem versão` | `[standard]==0.24.0` | `[standard]>=0.30.0,<1.0.0` | `não usa` | `sem versão` |
| **Starlette** | `sem versão` | `não usa` | `não usa` | `não usa` | `não usa` |

**🔴 CONFLITO CRÍTICO**: FastAPI e Uvicorn com versões incompatíveis
- Darwin usa FastAPI 0.104.1 (Nov 2023) + Uvicorn 0.24.0
- Actions requer FastAPI >=0.115.0 (Jul 2024) + Uvicorn >=0.30.0
- **Impacto**: Breaking changes entre versões podem quebrar APIs

### 2.2 VALIDAÇÃO & SERIALIZAÇÃO

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **Pydantic** | `sem versão` | `==2.5.0` | `>=2.8.0,<3.0.0` | `não usa` | `não usa` |
| **Pydantic-settings** | `sem versão` | `não usa` | `não usa` | `não usa` | `sem versão` |

**🟡 CONFLITO MÉDIO**: Pydantic v2 minor versions
- Darwin: 2.5.0 (Dec 2023)
- Actions: >=2.8.0 (Mai 2024)
- **Impacto**: Possíveis mudanças em validators e serializers

### 2.3 GOOGLE CLOUD PLATFORM

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **google-cloud-aiplatform** | `sem versão` | `==1.67.1` | `não usa` | `não usa` | `sem versão` |
| **google-cloud-bigquery** | `sem versão` | `==3.25.0` | `não usa` | `não usa` | `sem versão` |
| **google-cloud-secret-manager** | `não usa` | `==2.21.1` | `não usa` | `não usa` | `não usa` |
| **vertexai** | `não usa` | `==1.67.1` | `não usa` | `não usa` | `não usa` |

**🟢 CONFLITO MENOR**: Versões específicas vs. não especificadas
- Darwin tem versões fixadas, outros não
- **Resolução**: Usar versões do Darwin como baseline

### 2.4 AI/ML & VECTOR DATABASES

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **OpenAI** | `não usa` | `==1.3.7` | `não usa` | `não usa` | `não usa` |
| **ChromaDB** | `não usa` | `==0.4.15` | `não usa` | `sem versão` | `não usa` |
| **sentence-transformers** | `sem versão` | `não usa` | `não usa` | `sem versão` | `não usa` |
| **scikit-learn** | `sem versão` | `não usa` | `não usa` | `não usa` | `não usa` |
| **faiss-cpu** | `sem versão` | `não usa` | `não usa` | `não usa` | `não usa` |

**🟡 CONFLITO MÉDIO**: ChromaDB duplicado sem sincronização
- Darwin: versão específica 0.4.15
- Ingestão: sem versão específica
- **Impacto**: Incompatibilidade de schemas de dados

### 2.5 PROCESSAMENTO DE DADOS

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **Pandas** | `sem versão` | `não usa` | `não usa` | `sem versão` | `não usa` |
| **Numpy** | `sem versão` | `==1.24.3` | `não usa` | `não usa` | `não usa` |
| **PyYAML** | `sem versão` | `==6.0.1` | `>=6.0.1,<7.0.0` | `não usa` | `sem versão` |
| **requests** | `sem versão` | `==2.31.0` | `não usa` | `sem versão` | `não usa` |

**🟢 CONFLITO MENOR**: Versões compatíveis mas não sincronizadas

### 2.6 INFRAESTRUTURA & CACHE

| Biblioteca | Backend Principal | Darwin RAG+ | Actions | Ingestão | Raiz |
|------------|-------------------|-------------|---------|----------|------|
| **Redis** | `sem versão` | `não usa` | `não usa` | `não usa` | `não usa` |
| **NetworkX** | `sem versão` | `não usa` | `não usa` | `não usa` | `não usa` |
| **PyJWT** | `sem versão` | `não usa` | `não usa` | `não usa` | `sem versão` |
| **psutil** | `sem versão` | `não usa` | `não usa` | `não usa` | `sem versão` |

**🟢 CONFLITO MENOR**: Dependências específicas do backend principal

## 3. DEPENDÊNCIAS ESPECÍFICAS POR BACKEND

### 3.1 Backend Principal EXCLUSIVAS:
- **LangChain**: `langchain`, `langchain-community`, `langchain-google-vertexai`
- **Document Processing**: `beautifulsoup4`, `html2text`, `lxml`, `markdown`, `pdf2image`, `pytesseract`, `unstructured`
- **Monitoring**: `opentelemetry-*`, `prometheus-fastapi-instrumentator`
- **GraphQL**: `strawberry-graphql[fastapi]`
- **API Docs**: `redoc`
- **Dev Tools**: `black`, `mypy`, `pytest`, `pytest-asyncio`, `ruff`
- **ML/NLP**: `nltk`, `tiktoken`

### 3.2 Darwin RAG+ EXCLUSIVAS:
- **Multipart**: `python-multipart==0.0.6`

### 3.3 Ingestão Job EXCLUSIVAS:
- **Bibliography**: `bibtexparser`
- **Data Processing**: `python-dateutil`, `tqdm`, `pyarrow`

## 4. FRONTEND NODE.JS DEPENDENCIES

### 4.1 Framework & Core:
- **Next.js**: `14.2.0`
- **React**: `18.3.1`
- **TypeScript**: `5.6.2`

### 4.2 UI Components:
- **Radix UI**: Conjunto completo de componentes
- **Tailwind CSS**: `3.4.10`
- **Framer Motion**: `11.11.1`

### 4.3 Specialized:
- **PDF Viewer**: `@react-pdf-viewer/*`
- **Graph Visualization**: `@react-sigma/core`, `graphology`, `sigma`
- **Charts**: `recharts`
- **State Management**: `zustand`
- **Data Fetching**: `@tanstack/react-query`

**🟢 SEM CONFLITOS**: Frontend isolado com dependências estáveis

## 5. MATRIZ DE COMPATIBILIDADE

### 5.1 CONFLITOS CRÍTICOS (🔴)
| Conflito | Descrição | Impacto | Prioridade |
|----------|-----------|---------|------------|
| **FastAPI 0.104.1 vs >=0.115.0** | Breaking changes entre versões | API compatibility | **P0** |
| **Uvicorn 0.24.0 vs >=0.30.0** | ASGI server incompatibilidade | Runtime errors | **P0** |

### 5.2 CONFLITOS MÉDIOS (🟡)
| Conflito | Descrição | Impacto | Prioridade |
|----------|-----------|---------|------------|
| **Pydantic 2.5.0 vs >=2.8.0** | Validator changes | Data validation | **P1** |
| **ChromaDB versioned vs unversioned** | Vector DB schema | Data compatibility | **P1** |

### 5.3 CONFLITOS MENORES (🟢)
| Conflito | Descrição | Impacto | Resolução |
|----------|-----------|---------|-----------|
| **Numpy/PyYAML versions** | Compatible ranges | Low | Unify versions |
| **Missing version specs** | Dependency drift | Medium | Add version pins |

## 6. RECOMENDAÇÕES DE VERSÕES UNIFICADAS

### 6.1 Framework Web:
```txt
fastapi>=0.115.0,<1.0.0  # Usar range do Actions (mais recente)
uvicorn[standard]>=0.30.0,<1.0.0  # Usar range do Actions
pydantic>=2.8.0,<3.0.0  # Usar range do Actions
starlette>=0.38.0,<1.0.0  # Compatível com FastAPI 0.115+
```

### 6.2 Google Cloud:
```txt
google-cloud-aiplatform==1.67.1  # Versão testada do Darwin
google-cloud-bigquery==3.25.0    # Versão testada do Darwin
google-cloud-secret-manager==2.21.1  # Do Darwin
vertexai==1.67.1  # Do Darwin
```

### 6.3 AI/ML:
```txt
openai==1.3.7  # Versão testada do Darwin
chromadb==0.4.15  # Versão testada do Darwin
sentence-transformers>=2.2.0,<3.0.0  # Range estável
```

### 6.4 Data Processing:
```txt
pandas>=2.0.0,<3.0.0  # Range moderno
numpy>=1.24.3,<2.0.0  # Compatibilidade ampla
pyyaml>=6.0.1,<7.0.0  # Range do Actions
requests>=2.31.0,<3.0.0  # Versão segura
```

## 7. CONFIGURAÇÕES DOCKER UNIFICADAS

### 7.1 Dockerfile Base Recomendado:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies unificadas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Environment variables padrão
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Health check padrão
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}
```

## 8. VARIÁVEIS DE AMBIENTE CONFLITANTES

### 8.1 Backend Principal:
```bash
PYTHONPATH="/app/src:/app/external/pcs-meta-repo:/app"
KEC_CONFIG_PATH="/app/src/kec_biomat/configs"
DARWIN_MEMORY_PATH="/app/src/darwin_core/memory"
GOOGLE_CLOUD_PROJECT="pcs-helio"
```

### 8.2 Darwin RAG+:
```bash
PYTHONPATH=/app
PORT=8080
```

### 8.3 Unificação Recomendada:
```bash
PYTHONPATH="/app/src:/app/external:/app"
KEC_CONFIG_PATH="/app/configs"
DARWIN_MEMORY_PATH="/app/data/memory"
GOOGLE_CLOUD_PROJECT="pcs-helio"
PORT=8080
SERVICE_NAME="kec-unified-backend"
ENVIRONMENT="production"
```

## 9. PLANO DE RESOLUÇÃO DE CONFLITOS

### 9.1 Fase 1 - Crítica (P0):
1. **Atualizar Darwin RAG+** para FastAPI >=0.115.0
2. **Atualizar Darwin RAG+** para Uvicorn >=0.30.0
3. **Tester compatibilidade** de APIs existentes
4. **Validar quebras** em endpoints Darwin

### 9.2 Fase 2 - Média (P1):
1. **Sincronizar Pydantic** para >=2.8.0 em todos os backends
2. **Fixar versão ChromaDB** em 0.4.15 no Ingestão
3. **Testar schemas** de dados entre backends
4. **Validar pipelines** de ingestão

### 9.3 Fase 3 - Menor (P2):
1. **Adicionar version pins** em todos os requirements
2. **Padronizar ranges** de versões compatíveis
3. **Unificar Dockerfiles** com base image comum
4. **Consolidar environment variables**

### 9.4 Fase 4 - Validação:
1. **Testes de integração** end-to-end
2. **Performance benchmarks** comparativos
3. **Deploy staging** com backend unificado
4. **Smoke tests** em produção

## 10. RISCOS E MITIGAÇÕES

### 10.1 Riscos Altos:
- **Breaking Changes FastAPI**: APIs podem quebrar
  - **Mitigação**: Testes extensivos + rollback plan
- **ChromaDB Schema Changes**: Dados podem corromper  
  - **Mitigação**: Backup + migration scripts

### 10.2 Riscos Médios:
- **Pydantic Validation**: Modelos podem falhar
  - **Mitigação**: Validation tests + schema versioning
- **Performance Regression**: Overhead de dependências
  - **Mitigação**: Benchmarks + profiling

## CONCLUSÃO

A unificação é **TECNICAMENTE VIÁVEL** mas requer **CUIDADO EXTREMO** nos conflitos críticos P0. Os backends compartilham arquitetura similar (Python 3.11-slim) mas têm versões de dependências incompatíveis que podem quebrar funcionalidades existentes.

**Recomendação**: Executar fases de resolução sequencialmente com testes extensivos em cada etapa.