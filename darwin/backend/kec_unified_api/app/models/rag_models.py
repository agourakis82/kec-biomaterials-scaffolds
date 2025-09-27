"""RAG++ Consolidated Models

Modelos Pydantic consolidados integrando funcionalidades dos backends
Principal e Darwin para sistema RAG++ unificado com Vertex AI.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryDomain(str, Enum):
    """Domínios científicos suportados"""
    BIOMATERIALS = "biomaterials"
    NEUROSCIENCE = "neuroscience"
    PHILOSOPHY = "philosophy"
    QUANTUM = "quantum"
    PSYCHIATRY = "psychiatry"
    CROSS_DOMAIN = "cross_domain"


class SearchMethod(str, Enum):
    """Métodos de busca RAG"""
    SIMPLE = "simple"
    ITERATIVE = "iterative"
    SCIENTIFIC = "scientific"
    CROSS_DOMAIN = "cross_domain"
    DISCOVERY = "discovery"


class SourceType(str, Enum):
    """Tipos de fontes de conhecimento"""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    NATURE = "nature"
    IEEE = "ieee"
    SPRINGER = "springer"
    MANUAL = "manual"
    RSS_FEED = "rss_feed"
    DOI = "doi"


# =============================================================================
# MODELOS BASE CONSOLIDADOS
# =============================================================================

class BaseRAGRequest(BaseModel):
    """Request base para operações RAG"""
    query: str = Field(..., description="Pergunta ou query de pesquisa")
    top_k: Optional[int] = Field(5, description="Número de documentos a recuperar")
    include_sources: bool = Field(True, description="Incluir informações das fontes")


class BaseRAGResponse(BaseModel):
    """Response base para operações RAG"""
    query: str = Field(..., description="Query original")
    answer: str = Field(..., description="Resposta gerada")
    method: str = Field(..., description="Método de processamento usado")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Documentos fonte")
    retrieved_docs: Optional[int] = Field(None, description="Número de documentos recuperados")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da resposta")


# =============================================================================
# MODELOS DO BACKEND PRINCIPAL (migrados)
# =============================================================================

class RAGPlusQuery(BaseRAGRequest):
    """RAG++ query request (do backend principal)"""
    pass


class RAGPlusResponse(BaseRAGResponse):
    """RAG++ query response (do backend principal)"""
    reasoning_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Passos de raciocínio iterativo")
    total_steps: Optional[int] = Field(None, description="Total de passos de raciocínio")


class IterativeRAGRequest(BaseRAGRequest):
    """Request para RAG iterativo com ReAct pattern"""
    max_iterations: int = Field(3, description="Máximo de iterações ReAct")
    reasoning_method: str = Field("react", description="Método de raciocínio (react, cot)")


class IterativeRAGResponse(BaseRAGResponse):
    """Response para RAG iterativo"""
    iterations: List[Dict[str, Any]] = Field(default_factory=list, description="Iterações do processo")
    final_iteration: int = Field(1, description="Iteração final alcançada")
    convergence_reason: str = Field("completed", description="Razão para convergência")


class DiscoveryRequest(BaseModel):
    """Request para descoberta científica"""
    run_once: bool = Field(False, description="Executar apenas uma vez")
    domains: List[QueryDomain] = Field(default_factory=list, description="Domínios para descoberta")
    sources: List[SourceType] = Field(default_factory=list, description="Fontes para monitorar")
    keywords: List[str] = Field(default_factory=list, description="Palavras-chave para filtrar")


class DiscoveryResponse(BaseModel):
    """Response para operações de descoberta"""
    status: str = Field(..., description="Status da operação")
    fetched: int = Field(0, description="Artigos buscados")
    novel: int = Field(0, description="Artigos novos encontrados")
    added: int = Field(0, description="Artigos adicionados à base")
    errors: int = Field(0, description="Erros encontrados")
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentRequest(BaseModel):
    """Request para adicionar documento"""
    content: str = Field(..., description="Conteúdo do documento")
    source: str = Field("", description="Fonte do documento")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados adicionais")
    domain: Optional[QueryDomain] = Field(None, description="Domínio científico")


class ServiceStatus(BaseModel):
    """Status do serviço RAG++"""
    service: str = Field("rag_plus", description="Nome do serviço")
    status: str = Field(..., description="Status do serviço")
    components: Dict[str, Any] = Field(default_factory=dict, description="Status dos componentes")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuração do serviço")
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# MODELOS DO DARWIN (migrados e adaptados)
# =============================================================================

class RAGSearchRequest(BaseModel):
    """RAG search request (do Darwin)"""
    q: str = Field(..., description="Query de busca")
    k: int = Field(4, description="Número de resultados")
    domain: Optional[QueryDomain] = Field(None, description="Domínio científico")


class RAGSearchResponse(BaseModel):
    """RAG search response (do Darwin)"""
    query: str = Field(..., description="Query original")
    answer: str = Field(..., description="Resposta gerada")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Resultados da busca")


class DiscoveryConfig(BaseModel):
    """Configuração para descoberta científica"""
    feeds: List[Dict[str, Any]] = Field(default_factory=list, description="Feeds RSS configurados")
    interval_hours: int = Field(24, description="Intervalo entre descobertas (horas)")
    max_articles_per_feed: int = Field(10, description="Máximo de artigos por feed")
    novelty_threshold: float = Field(0.7, description="Threshold para novidade")


class SourceConfig(BaseModel):
    """Configuração de fonte de conhecimento"""
    name: str = Field(..., description="Nome da fonte")
    type: SourceType = Field(..., description="Tipo da fonte")
    url: str = Field(..., description="URL da fonte")
    enabled: bool = Field(True, description="Fonte habilitada")
    check_interval: int = Field(3600, description="Intervalo de verificação (segundos)")
    domain: Optional[QueryDomain] = Field(None, description="Domínio específico")


class RefinementRequest(BaseModel):
    """Request para refinamento iterativo"""
    original_query: str = Field(..., description="Query original")
    previous_answer: str = Field(..., description="Resposta anterior")
    refinement_instruction: str = Field(..., description="Instrução de refinamento")


# =============================================================================
# MODELOS UNIFICADOS NOVOS
# =============================================================================

class UnifiedRAGRequest(BaseModel):
    """Request RAG unificado consolidando todas as funcionalidades"""
    query: str = Field(..., description="Pergunta de pesquisa")
    method: SearchMethod = Field(SearchMethod.SIMPLE, description="Método de busca")
    domain: Optional[QueryDomain] = Field(None, description="Domínio científico específico")
    top_k: int = Field(5, description="Número de documentos a recuperar")
    max_iterations: Optional[int] = Field(3, description="Máx iterações (para método iterativo)")
    include_sources: bool = Field(True, description="Incluir fontes na resposta")
    cross_domain: bool = Field(False, description="Busca interdisciplinar")
    scientific_validation: bool = Field(True, description="Validação científica de fontes")
    real_time_discovery: bool = Field(False, description="Descoberta em tempo real")


class ScientificSearchRequest(BaseModel):
    """Request para busca científica especializada"""
    query: str = Field(..., description="Query científica")
    domains: List[QueryDomain] = Field(default_factory=list, description="Domínios científicos")
    sources: List[SourceType] = Field(default_factory=list, description="Fontes preferenciais")
    require_doi: bool = Field(False, description="Exigir DOI nos resultados")
    min_impact_factor: Optional[float] = Field(None, description="Fator de impacto mínimo")
    temporal_weight: float = Field(1.0, description="Peso para relevância temporal")
    citation_network: bool = Field(True, description="Analisar rede de citações")


class BiomaterialsQueryRequest(BaseModel):
    """Request específico para queries de biomateriais"""
    query: str = Field(..., description="Query sobre biomateriais")
    scaffold_type: Optional[str] = Field(None, description="Tipo de scaffold específico")
    material_properties: List[str] = Field(default_factory=list, description="Propriedades do material")
    tissue_type: Optional[str] = Field(None, description="Tipo de tecido alvo")
    porosity_range: Optional[Dict[str, float]] = Field(None, description="Faixa de porosidade")
    mechanical_properties: bool = Field(True, description="Incluir propriedades mecânicas")
    biocompatibility: bool = Field(True, description="Incluir biocompatibilidade")


class CrossDomainRequest(BaseModel):
    """Request para queries interdisciplinares"""
    primary_query: str = Field(..., description="Query principal")
    primary_domain: QueryDomain = Field(..., description="Domínio primário")
    secondary_domains: List[QueryDomain] = Field(..., description="Domínios secundários")
    connection_strength: float = Field(0.5, description="Força mínima da conexão interdisciplinar")
    concept_mapping: bool = Field(True, description="Mapear conceitos entre domínios")
    analogical_reasoning: bool = Field(False, description="Raciocínio analógico")


class UnifiedRAGResponse(BaseModel):
    """Response RAG unificado"""
    query: str = Field(..., description="Query original")
    method: SearchMethod = Field(..., description="Método usado")
    domain: Optional[QueryDomain] = Field(None, description="Domínio primário")
    answer: str = Field(..., description="Resposta consolidada")
    confidence_score: float = Field(0.0, description="Score de confiança")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Fontes utilizadas")
    cross_domain_connections: List[Dict[str, Any]] = Field(default_factory=list, description="Conexões interdisciplinares")
    scientific_validation: Dict[str, Any] = Field(default_factory=dict, description="Validação científica")
    reasoning_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Trace do raciocínio")
    knowledge_graph: Optional[Dict[str, Any]] = Field(None, description="Grafo de conhecimento")
    discovery_insights: List[str] = Field(default_factory=list, description="Insights de descoberta")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas de performance")
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# MODELOS DE CONFIGURAÇÃO E MONITORAMENTO
# =============================================================================

class VertexAIConfig(BaseModel):
    """Configuração Vertex AI"""
    project_id: str = Field(..., description="GCP Project ID")
    location: str = Field("us-central1", description="Localização Vertex AI")
    embedding_model: str = Field("text-embedding-gecko", description="Modelo de embeddings")
    chat_model: str = Field("gemini-pro", description="Modelo de chat/geração")
    max_output_tokens: int = Field(1024, description="Máximo tokens de saída")
    temperature: float = Field(0.2, description="Temperatura para geração")


class RAGEngineConfig(BaseModel):
    """Configuração do RAG Engine"""
    vector_backend: str = Field("chroma", description="Backend de vetores")
    embedding_dimension: int = Field(768, description="Dimensão dos embeddings")
    similarity_threshold: float = Field(0.7, description="Threshold de similaridade")
    max_context_length: int = Field(8000, description="Máximo comprimento do contexto")
    chunk_size: int = Field(512, description="Tamanho dos chunks")
    chunk_overlap: int = Field(50, description="Sobreposição entre chunks")


class PerformanceMetrics(BaseModel):
    """Métricas de performance do sistema"""
    query_time_ms: float = Field(0.0, description="Tempo da query (ms)")
    retrieval_time_ms: float = Field(0.0, description="Tempo de recuperação (ms)")
    generation_time_ms: float = Field(0.0, description="Tempo de geração (ms)")
    total_time_ms: float = Field(0.0, description="Tempo total (ms)")
    documents_processed: int = Field(0, description="Documentos processados")
    tokens_generated: int = Field(0, description="Tokens gerados")
    api_calls: int = Field(0, description="Chamadas à API")
    cache_hits: int = Field(0, description="Cache hits")


class HealthStatus(BaseModel):
    """Status de saúde detalhado"""
    healthy: bool = Field(True, description="Status geral de saúde")
    components: Dict[str, bool] = Field(default_factory=dict, description="Status por componente")
    errors: List[str] = Field(default_factory=list, description="Erros encontrados")
    warnings: List[str] = Field(default_factory=list, description="Avisos")
    last_check: datetime = Field(default_factory=datetime.now, description="Último check")
    uptime_seconds: float = Field(0.0, description="Tempo ativo (segundos)")