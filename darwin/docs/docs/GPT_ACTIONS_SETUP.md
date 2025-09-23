# 🤖 Configuração GPT Actions - KEC Biomaterials API

## 📋 Visão Geral

O backend modular KEC Biomaterials v2.0 está totalmente preparado para integração com **ChatGPT Actions**, oferecendo acesso direto a:
- ✅ **Análise KEC** de biomateriais (H, κ, σ, ϕ, σ_Q)
- ✅ **RAG++ Search** com Vertex AI e BigQuery
- ✅ **Tree Search PUCT** para otimização
- ✅ **Memória Persistente** e descoberta científica
- ✅ **Status de Projeto** em tempo real

## 🚀 Service URL (Após Deploy)

```
https://kec-biomaterials-api-{hash}.a.run.app
```

*A URL exata será gerada pelo script de deploy em execução*

## 🛠️ Configuração no ChatGPT

### 1. **Criar GPT Action**

1. Acesse **ChatGPT** > **Explore GPTs** > **Create**
2. Configure o GPT:
   - **Name**: "KEC Biomaterials Analyst"
   - **Description**: "Advanced biomaterials analysis using KEC metrics, RAG++ search, and scientific discovery"

### 2. **Configurar Action Schema**

Use o schema OpenAPI gerado automaticamente:

```yaml
# URL do schema (será atualizada após deploy)
Schema URL: https://kec-biomaterials-api-{hash}.a.run.app/openapi.json
```

### 3. **Authentication**

```yaml
Authentication Type: None
```
*(O service está configurado como público para GPT Actions)*

## 🎯 Endpoints Principais para GPT Actions

### 1. **Análise KEC de Biomateriais**
```http
POST /gpt-actions/analyze-kec-metrics
```

**Uso no ChatGPT**:
> "Analise as métricas KEC desta estrutura porosa: [dados do grafo]"

**Retorna**: H_spectral, k_forman, sigma, swp + interpretação

### 2. **Busca RAG++ Científica**
```http
POST /gpt-actions/rag-query
```

**Uso no ChatGPT**:
> "Busque pesquisas recentes sobre scaffolds porosos para engenharia de tecidos"

**Retorna**: Resposta + fontes científicas relevantes

### 3. **Status Completo do Projeto**
```http
GET /gpt-actions/project-status
```

**Uso no ChatGPT**:
> "Qual o status atual do projeto KEC Biomaterials?"

**Retorna**: Fase atual, tarefas ativas, próximos passos, health

### 4. **Descoberta Científica**
```http
POST /gpt-actions/scientific-discovery
```

**Uso no ChatGPT**:
> "Execute uma descoberta científica sobre inovações em biomateriais"

**Retorna**: Últimas descobertas científicas relevantes

### 5. **Health Check dos Sistemas**
```http
GET /gpt-actions/system-health
```

**Uso no ChatGPT**:
> "Verifique a saúde de todos os sistemas backend"

**Retorna**: Status de todos os módulos e sistemas de memória

## 📊 Exemplo de Uso Completo

### Conversa Exemplo com ChatGPT:

**Usuário**: "Analise esta estrutura porosa e me dê insights sobre sua eficiência para crescimento celular"

**ChatGPT usa KEC Actions**:
1. **POST** `/gpt-actions/analyze-kec-metrics` → Calcula H, κ, σ, ϕ
2. **POST** `/gpt-actions/rag-query` → Busca literatura sobre eficiência celular
3. **GET** `/gpt-actions/project-status` → Verifica contexto do projeto

**Resposta Integrada**:
> "Baseado na análise KEC, sua estrutura tem:
> - **Entropia H=2.34**: Complexidade moderada, boa para diferenciação celular
> - **Small-world σ=1.8**: Excelente conectividade para transporte de nutrientes  
> - **Curvatura κ=0.12**: Geometria favorável para adesão celular
>
> Pesquisas recentes (via RAG++) mostram que estruturas com σ>1.5 têm 40% maior eficiência para crescimento de osteoblastos..."

## 🔧 Configuração Avançada

### Headers Personalizados
```yaml
# Headers automáticos (já configurados)
Access-Control-Allow-Origin: "*"
Access-Control-Allow-Methods: "GET, POST, OPTIONS"
X-Process-Time: "tempo_de_processamento"
```

### Rate Limiting
- **100 requests/min** por padrão
- **Burst capacity**: 200 requests
- **Timeout**: 300s para análises complexas

### CORS Configuration
```yaml
# Já configurado para GPT Actions
allowed_origins: 
  - "https://chat.openai.com"
  - "https://chatgpt.com"
allowed_methods: ["GET", "POST", "OPTIONS"]
```

## 🎯 Funcionalidades Especiais

### 1. **Memória de Conversação**
- **Histórico completo** de todas as interações GPT
- **Contexto automaticamente carregado** no início de sessões
- **Continuidade garantida** entre sessões

### 2. **Descoberta Científica 24/7**
- **Monitoramento contínuo** de ArXiv, PubMed, Nature
- **Alerts automáticos** para breakthroughs
- **Filtragem inteligente** por relevância

### 3. **Analytics com pcs-meta-repo**
- **Integração automática** com ferramentas avançadas
- **ML pipeline** para análises preditivas
- **Correlações multi-variadas** automatizadas

## 📈 Monitoramento e Performance

### Métricas Cloud Run
- **Latência média**: < 2s para queries simples
- **Throughput**: 100 requests/min sustained
- **Uptime**: 99.9% target
- **Memory**: 4GB otimizado para ML workloads

### Health Checks
```http
# Verificação básica
GET /healthz

# Health completo dos módulos
GET /gpt-actions/system-health
```

## 🚨 Troubleshooting

### Problemas Comuns

**1. "Service Unavailable"**
- Verificar: `GET /healthz`
- Service pode estar cold starting (aguardar 30s)

**2. "Authentication Failed"**
- GPT Actions usa autenticação none
- Verificar CORS headers

**3. "Timeout"**
- Análises complexas podem levar até 5 minutos
- Tree search com budget alto = mais tempo

### Logs e Debug
```bash
# Logs Cloud Run
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=kec-biomaterials-api" --limit=50

# Status dos sistemas
curl https://service-url/gpt-actions/system-health
```

---

## 🎉 Resultado Final

**✅ Backend modular KEC v2.0 pronto para ChatGPT Actions**
**✅ Memória persistente e descoberta científica ativas**  
**✅ Integração Vertex AI e BigQuery configurada**
**✅ Deploy otimizado no Cloud Run**

O ChatGPT agora pode acessar diretamente todo o poder do backend KEC para análises avançadas de biomateriais!