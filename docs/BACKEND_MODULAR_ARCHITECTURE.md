# 🏗️ Arquitetura Modular Backend KEC Biomaterials

## 📋 Visão Geral

Reestruturação completa do backend em **4 módulos especializados** mantendo compatibilidade com sistema RAG++ em produção no Google Cloud Run.

## 🧬 Módulos Principais

### 1. `darwin_core/` - RAG++, Tree Search & Memory
```
darwin_core/
├── __init__.py
├── rag/
│   ├── rag_plus.py      # Motor RAG++ com Vertex AI
│   ├── iterative.py     # Busca iterativa e refinamento  
│   └── vertex.py        # Integração Google Vertex AI
├── tree_search/
│   ├── puct.py          # PUCT implementation
│   ├── mcts.py          # Monte Carlo Tree Search
│   └── algorithms.py    # Algoritmos complementares
├── memory/
│   ├── session.py       # Gestão de sessões
│   ├── context_cache.py # Cache de contexto
│   └── persistent.py    # Armazenamento persistente
└── discovery/
    └── engine.py        # Motor de descoberta
```

### 2. `kec_biomat/` - Biomaterials Core
```
kec_biomat/
├── __init__.py
├── metrics/
│   ├── kec_metrics.py   # ✅ Migrado (H, κ, σ, ϕ, σ_Q)
│   ├── entropy.py       # Cálculos de entropia
│   └── percolation.py   # Análise de percolação
├── configs/
│   ├── __init__.py      # ✅ Sistema de configuração YAML
│   └── kec_config.yaml  # ✅ Migrado do pack_2025-09-19
├── processing/
│   └── pipeline.py      # ✅ Pipeline completo de processamento
└── tests/
    ├── test_kec_metrics.py   # ✅ Migrado e expandido
    ├── test_config.py        # ✅ Testes de configuração
    └── test_pipeline.py      # ✅ Testes de pipeline
```

### 3. `pcs_helio/` - Advanced Analytics
```
pcs_helio/
├── __init__.py
├── analytics/
│   └── engine.py        # ✅ Motor de analytics avançados
├── integration/
│   └── pcs_meta_bridge.py # ✅ Ponte com pcs-meta-repo
└── services/
    └── helio_service.py # ✅ Serviço integrado
```

### 4. `philosophy/` - Reasoning & Knowledge
```
philosophy/
├── __init__.py
├── reasoning/
│   └── logic_engine.py  # ✅ Motor de raciocínio lógico
└── knowledge/
    └── ontology.py      # ✅ Gestão de conhecimento
```

### 5. `kec_biomat_api/` - FastAPI Gateway (Existente)
```
kec_biomat_api/
├── main.py              # 🔄 REFATORAR para usar novos módulos
├── routers/             # 🔄 LIMPAR e reorganizar
│   ├── rag.py           # → darwin_core.rag
│   ├── memory.py        # → darwin_core.memory  
│   ├── processing.py    # → kec_biomat.processing
│   └── tree_search.py   # → darwin_core.tree_search
├── services/            # 🔄 REFATORAR
│   ├── rag_service.py   # → darwin_core integration
│   └── helio_service.py # → pcs_helio integration
└── [manter]: auth/, cache/, monitoring/, websocket/
```

## 🔗 Integração com pcs-meta-repo

### Git Submodule (Recomendado)
```bash
# Já configurado em .gitmodules
git submodule update --init --recursive
```

### PYTHONPATH Configuration
```bash
# Automaticamente configurado via scripts/setup_backend_modules.sh
export PYTHONPATH="/app/src:/app/external/pcs-meta-repo:$PYTHONPATH"
export KEC_CONFIG_PATH="/app/src/kec_biomat/configs"
export DARWIN_MEMORY_PATH="/app/src/darwin_core/memory"
```

## 📦 Pattern de Imports

```python
# Imports dos novos módulos
from darwin_core.rag import RAGPlusEngine, IterativeSearch
from darwin_core.tree_search import PUCTSearch, MCTSEngine
from darwin_core.memory import SessionManager
from kec_biomat.metrics import compute_kec_metrics
from kec_biomat.configs import load_config
from kec_biomat.processing import KECProcessingPipeline
from pcs_helio.analytics import AnalyticsEngine
from pcs_helio.integration import PCSMetaBridge
from philosophy.reasoning import LogicEngine
```

## 🚀 Refatoração do main.py

### Estrutura Proposta
```python
# src/kec_biomat_api/main.py (refatorado)
from fastapi import FastAPI
from darwin_core.rag import RAGPlusEngine
from darwin_core.tree_search import PUCTSearch
from kec_biomat.processing import KECProcessingPipeline
from pcs_helio.services import HelioService

app = FastAPI(title="KEC Biomaterials API v2.0")

# Inicialização modular
@app.on_event("startup")
async def startup():
    # Inicializa módulos
    app.state.rag_engine = RAGPlusEngine(config)
    app.state.puct_search = PUCTSearch(evaluator, config)
    app.state.kec_pipeline = KECProcessingPipeline()
    app.state.helio_service = HelioService()
    
    # Inicializa serviços
    await app.state.rag_engine.initialize()
    await app.state.helio_service.initialize()
```

## 📊 Migração de Arquivos

### ✅ Concluído
- `kec_biomat_pack_2025-09-19/pipeline/kec_metrics.py` → `src/kec_biomat/metrics/kec_metrics.py`
- `kec_biomat_pack_2025-09-19/configs/kec_config.yaml` → `src/kec_biomat/configs/kec_config.yaml`
- `kec_biomat_pack_2025-09-19/tests/test_kec_metrics.py` → `src/kec_biomat/tests/test_kec_metrics.py`

### 🔄 Próximos Passos
- `kec_biomat_pack_2025-09-19/darwin/memory_kec_v2.json` → `src/darwin_core/memory/`
- Refatorar routers para usar novos módulos
- Atualizar Dockerfile com PYTHONPATH
- Testes de integração end-to-end

## 🎯 Benefícios da Arquitetura

### ✅ Separação de Responsabilidades
- **darwin_core**: RAG++, busca, memória
- **kec_biomat**: Métricas específicas do domínio  
- **pcs_helio**: Analytics avançados
- **philosophy**: Raciocínio e conhecimento

### ✅ Reutilização de Código
- Módulos independentes e testáveis
- APIs claras entre componentes
- Integração via interfaces bem definidas

### ✅ Escalabilidade
- Cada módulo pode evoluir independentemente
- Fácil adição de novos algoritmos
- Integração natural com pcs-meta-repo

### ✅ Manutenibilidade
- Código organizado por domínio
- Testes específicos por módulo
- Configuração centralizada

## 🔧 Deploy em Produção

### Cloud Run (Atual)
O backend continua funcionando normalmente com:
- FastAPI gateway mantido
- Endpoints existentes funcionais
- Migração gradual para novos módulos

### Dockerfile Update
```dockerfile
# Adicionar ao Dockerfile existente
ENV PYTHONPATH="/app/src:/app/external/pcs-meta-repo"
RUN git submodule update --init --recursive
```

## 📈 Performance & Monitoring

### Métricas por Módulo
- **darwin_core**: Latência RAG++, cache hit rate
- **kec_biomat**: Tempo de processamento de métricas
- **pcs_helio**: Throughput de analytics
- **philosophy**: Profundidade de inferência

### Health Checks
```python
# Cada módulo expõe get_status()
GET /api/health/darwin_core
GET /api/health/kec_biomat  
GET /api/health/pcs_helio
GET /api/health/philosophy
```

---

**Esta arquitetura modular transforma o backend em um sistema robusto, escalável e mantível, preservando todas as funcionalidades existentes de RAG++, tree search, memória e PUCT em produção.**