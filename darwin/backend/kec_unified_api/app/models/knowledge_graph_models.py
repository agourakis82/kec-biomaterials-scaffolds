"""DARWIN Knowledge Graph - Pydantic Models

Modelos completos para o Knowledge Graph interdisciplinar que conecta
biomaterials, neuroscience, philosophy, quantum mechanics e psychiatry.
"""

from typing import Any, Dict, List, Optional, Union, Set, Tuple, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


# ==================== GRAPH TYPE ENUMS ====================

class KnowledgeGraphTypes(str, Enum):
    """Tipos de Knowledge Graph suportados."""
    CITATION_NETWORK = "citation"
    CONCEPT_MAP = "concept"
    METHODOLOGY_GRAPH = "methodology"
    COLLABORATION_NETWORK = "collaboration"
    TEMPORAL_GRAPH = "temporal"
    INTERDISCIPLINARY = "cross_domain"


class ScientificDomains(str, Enum):
    """Domínios científicos do DARWIN."""
    BIOMATERIALS = "biomaterials"
    NEUROSCIENCE = "neuroscience"
    PHILOSOPHY = "philosophy"
    QUANTUM_MECHANICS = "quantum_mechanics"
    PSYCHIATRY = "psychiatry"
    MATHEMATICS = "mathematics"
    INTERDISCIPLINARY = "interdisciplinary"


class NodeTypes(str, Enum):
    """Tipos de nós no Knowledge Graph."""
    PAPER = "paper"
    CONCEPT = "concept"
    AUTHOR = "author"
    METHOD = "method"
    KEYWORD = "keyword"
    INSIGHT = "insight"
    DISCOVERY = "discovery"
    METRIC = "metric"


class EdgeTypes(str, Enum):
    """Tipos de conexões entre nós."""
    CITES = "cites"
    SIMILAR_TO = "similar_to"
    USES_METHOD = "uses_method"
    COLLABORATES = "collaborates"
    BRIDGES_DOMAIN = "bridges_domain"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    SEMANTIC_RELATION = "semantic_relation"


# ==================== NODE MODELS ====================

class GraphNodeBase(BaseModel):
    """Base model para nós do Knowledge Graph."""
    id: str = Field(..., description="Identificador único do nó")
    type: NodeTypes = Field(..., description="Tipo do nó")
    label: str = Field(..., description="Label do nó")
    domain: ScientificDomains = Field(..., description="Domínio científico")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Propriedades específicas")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PaperNode(GraphNodeBase):
    """Nó representando um paper científico."""
    type: NodeTypes = Field(default=NodeTypes.PAPER)
    title: str = Field(..., description="Título do paper")
    authors: List[str] = Field(default_factory=list, description="Lista de autores")
    publication_date: Optional[datetime] = Field(None, description="Data de publicação")
    journal: Optional[str] = Field(None, description="Journal/Conferência")
    doi: Optional[str] = Field(None, description="DOI do paper")
    abstract: Optional[str] = Field(None, description="Abstract")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    citation_count: int = Field(default=0, description="Número de citações")


class ConceptNode(GraphNodeBase):
    """Nó representando um conceito científico."""
    type: NodeTypes = Field(default=NodeTypes.CONCEPT)
    description: Optional[str] = Field(None, description="Descrição do conceito")
    related_papers: List[str] = Field(default_factory=list, description="IDs de papers relacionados")
    semantic_embedding: Optional[List[float]] = Field(None, description="Embedding semântico")
    frequency: int = Field(default=1, description="Frequência de aparição")


class AuthorNode(GraphNodeBase):
    """Nó representando um autor."""
    type: NodeTypes = Field(default=NodeTypes.AUTHOR)
    affiliations: List[str] = Field(default_factory=list, description="Afiliações")
    h_index: Optional[int] = Field(None, description="H-index")
    research_areas: List[str] = Field(default_factory=list, description="Áreas de pesquisa")
    total_citations: int = Field(default=0, description="Total de citações")


class MethodNode(GraphNodeBase):
    """Nó representando uma metodologia."""
    type: NodeTypes = Field(default=NodeTypes.METHOD)
    description: Optional[str] = Field(None, description="Descrição da metodologia")
    application_domains: List[ScientificDomains] = Field(default_factory=list)
    complexity: Optional[str] = Field(None, description="Nível de complexidade")


class InsightNode(GraphNodeBase):
    """Nó representando um insight descoberto."""
    type: NodeTypes = Field(default=NodeTypes.INSIGHT)
    content: str = Field(..., description="Conteúdo do insight")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confiança do insight")
    source_system: Optional[str] = Field(None, description="Sistema que gerou o insight")
    validation_status: Optional[str] = Field(None, description="Status de validação")


# ==================== EDGE MODELS ====================

class GraphEdgeBase(BaseModel):
    """Base model para arestas do Knowledge Graph."""
    id: str = Field(..., description="ID único da aresta")
    source: str = Field(..., description="ID do nó origem")
    target: str = Field(..., description="ID do nó destino")
    type: EdgeTypes = Field(..., description="Tipo da conexão")
    weight: float = Field(default=1.0, ge=0.0, description="Peso da conexão")
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CitationEdge(GraphEdgeBase):
    """Aresta representando citação entre papers."""
    type: EdgeTypes = Field(default=EdgeTypes.CITES)
    citation_context: Optional[str] = Field(None, description="Contexto da citação")
    citation_type: Optional[str] = Field(None, description="Tipo de citação")


class SimilarityEdge(GraphEdgeBase):
    """Aresta representando similaridade entre conceitos."""
    type: EdgeTypes = Field(default=EdgeTypes.SIMILAR_TO)
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Score de similaridade")
    similarity_method: Optional[str] = Field(None, description="Método de cálculo")


class BridgeEdge(GraphEdgeBase):
    """Aresta representando ponte entre domínios."""
    type: EdgeTypes = Field(default=EdgeTypes.BRIDGES_DOMAIN)
    domain_from: ScientificDomains = Field(..., description="Domínio origem")
    domain_to: ScientificDomains = Field(..., description="Domínio destino")
    bridge_strength: float = Field(..., ge=0.0, le=1.0, description="Força da ponte")


# ==================== GRAPH MODELS ====================

class KnowledgeGraphSnapshot(BaseModel):
    """Snapshot completo do Knowledge Graph."""
    id: str = Field(..., description="ID do snapshot")
    graph_type: KnowledgeGraphTypes = Field(..., description="Tipo do grafo")
    nodes: List[Union[PaperNode, ConceptNode, AuthorNode, MethodNode, InsightNode]] = Field(
        default_factory=list, description="Lista de nós"
    )
    edges: List[Union[CitationEdge, SimilarityEdge, BridgeEdge, GraphEdgeBase]] = Field(
        default_factory=list, description="Lista de arestas"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados do grafo")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('nodes')
    def validate_unique_node_ids(cls, v):
        """Valida que todos os nós têm IDs únicos."""
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Node IDs must be unique")
        return v


# ==================== ANALYSIS MODELS ====================

class CentralityAnalysis(BaseModel):
    """Resultado de análise de centralidade."""
    algorithm: str = Field(..., description="Algoritmo usado")
    results: Dict[str, float] = Field(..., description="Node ID -> centrality score")
    top_k: Optional[List[Tuple[str, float]]] = Field(None, description="Top K nodes")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommunityDetection(BaseModel):
    """Resultado de detecção de comunidades."""
    algorithm: str = Field(..., description="Algoritmo usado")
    communities: List[List[str]] = Field(..., description="Lista de comunidades (node IDs)")
    modularity: Optional[float] = Field(None, description="Modularity score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PathAnalysis(BaseModel):
    """Análise de caminhos no grafo."""
    source: str = Field(..., description="Nó origem")
    target: str = Field(..., description="Nó destino")
    shortest_path: Optional[List[str]] = Field(None, description="Caminho mais curto")
    path_length: Optional[int] = Field(None, description="Comprimento do caminho")
    all_paths: Optional[List[List[str]]] = Field(None, description="Todos os caminhos")


class BridgeAnalysis(BaseModel):
    """Análise de conceitos ponte entre domínios."""
    bridge_concepts: List[Dict[str, Any]] = Field(..., description="Conceitos que fazem ponte")
    domain_connections: Dict[str, List[str]] = Field(..., description="Conexões entre domínios")
    interdisciplinary_score: float = Field(..., ge=0.0, le=1.0, description="Score interdisciplinar")


# ==================== TIMELINE MODELS ====================

class ResearchInsight(BaseModel):
    """Insight de pesquisa com timestamp."""
    id: str = Field(..., description="ID único do insight")
    content: str = Field(..., description="Conteúdo do insight")
    source: str = Field(..., description="Fonte do insight (RAG++, Multi-AI, etc.)")
    domains: List[ScientificDomains] = Field(..., description="Domínios relacionados")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nível de confiança")
    related_papers: List[str] = Field(default_factory=list, description="Papers relacionados")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchTimeline(BaseModel):
    """Timeline de insights e descobertas."""
    id: str = Field(..., description="ID da timeline")
    title: str = Field(..., description="Título da timeline")
    insights: List[ResearchInsight] = Field(default_factory=list, description="Lista de insights")
    start_date: datetime = Field(..., description="Data de início")
    end_date: Optional[datetime] = Field(None, description="Data de fim")
    domains_covered: Set[ScientificDomains] = Field(default_factory=set)
    total_insights: int = Field(default=0, description="Total de insights")
    
    @validator('total_insights', always=True)
    def update_total_insights(cls, v, values):
        """Atualiza automaticamente o total de insights."""
        insights = values.get('insights', [])
        return len(insights)


# ==================== SEARCH MODELS ====================

class ConceptSearchQuery(BaseModel):
    """Query para busca de conceitos."""
    query: str = Field(..., description="Texto da busca")
    domains: Optional[List[ScientificDomains]] = Field(None, description="Domínios específicos")
    node_types: Optional[List[NodeTypes]] = Field(None, description="Tipos de nós")
    limit: int = Field(default=10, ge=1, le=100, description="Limite de resultados")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ConceptSearchResult(BaseModel):
    """Resultado de busca de conceitos."""
    node: Union[PaperNode, ConceptNode, AuthorNode, MethodNode, InsightNode] = Field(
        ..., description="Nó encontrado"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Score de relevância")
    explanation: Optional[str] = Field(None, description="Explicação da relevância")


class NavigationQuery(BaseModel):
    """Query para navegação no grafo."""
    source_concept: str = Field(..., description="Conceito de origem")
    target_concept: Optional[str] = Field(None, description="Conceito de destino")
    max_hops: int = Field(default=3, ge=1, le=10, description="Máximo de saltos")
    domains: Optional[List[ScientificDomains]] = Field(None, description="Domínios permitidos")


# ==================== VISUALIZATION MODELS ====================

class GraphVisualizationConfig(BaseModel):
    """Configuração para visualização do grafo."""
    layout: str = Field(default="force-directed", description="Tipo de layout")
    show_labels: bool = Field(default=True, description="Mostrar labels")
    node_size_by: Optional[str] = Field(None, description="Atributo para tamanho do nó")
    edge_width_by: Optional[str] = Field(None, description="Atributo para largura da aresta")
    color_scheme: str = Field(default="domain", description="Esquema de cores")
    filter_domains: Optional[List[ScientificDomains]] = Field(None)
    filter_node_types: Optional[List[NodeTypes]] = Field(None)


class GraphVisualizationData(BaseModel):
    """Dados para visualização do grafo."""
    nodes: List[Dict[str, Any]] = Field(..., description="Dados dos nós para D3.js")
    edges: List[Dict[str, Any]] = Field(..., description="Dados das arestas para D3.js")
    config: GraphVisualizationConfig = Field(..., description="Configuração de visualização")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Estatísticas do grafo")


# ==================== REQUEST/RESPONSE MODELS ====================

class GraphBuildRequest(BaseModel):
    """Request para construção do grafo."""
    graph_types: List[KnowledgeGraphTypes] = Field(..., description="Tipos de grafo a construir")
    domains: Optional[List[ScientificDomains]] = Field(None, description="Domínios específicos")
    include_citations: bool = Field(default=True, description="Incluir rede de citações")
    include_concepts: bool = Field(default=True, description="Incluir mapa conceitual")
    include_timeline: bool = Field(default=True, description="Incluir timeline")
    force_rebuild: bool = Field(default=False, description="Forçar reconstrução")


class GraphStatsResponse(BaseModel):
    """Estatísticas do Knowledge Graph."""
    total_nodes: int = Field(..., description="Total de nós")
    total_edges: int = Field(..., description="Total de arestas")
    nodes_by_type: Dict[str, int] = Field(..., description="Nós por tipo")
    nodes_by_domain: Dict[str, int] = Field(..., description="Nós por domínio")
    edges_by_type: Dict[str, int] = Field(..., description="Arestas por tipo")
    graph_density: float = Field(..., description="Densidade do grafo")
    average_degree: float = Field(..., description="Grau médio dos nós")
    connected_components: int = Field(..., description="Componentes conectados")
    interdisciplinary_connections: int = Field(..., description="Conexões interdisciplinares")
    last_updated: datetime = Field(..., description="Última atualização")


# ==================== INTEGRATION MODELS ====================

class IntegrationSource(BaseModel):
    """Fonte de dados para integração no Knowledge Graph."""
    source_name: str = Field(..., description="Nome da fonte")
    source_type: str = Field(..., description="Tipo da fonte (RAG++, Multi-AI, etc.)")
    data_format: str = Field(..., description="Formato dos dados")
    last_sync: Optional[datetime] = Field(None, description="Última sincronização")
    active: bool = Field(default=True, description="Fonte ativa")


class KnowledgeGraphHealth(BaseModel):
    """Health check do Knowledge Graph."""
    healthy: bool = Field(..., description="Status geral de saúde")
    components: Dict[str, str] = Field(..., description="Status dos componentes")
    graph_stats: Optional[GraphStatsResponse] = Field(None, description="Estatísticas do grafo")
    integration_sources: List[IntegrationSource] = Field(
        default_factory=list, description="Status das fontes de dados"
    )
    last_check: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = Field(default_factory=list, description="Erros encontrados")


# ==================== EXPORT MODELS ====================

__all__ = [
    # Enums
    "KnowledgeGraphTypes", "ScientificDomains", "NodeTypes", "EdgeTypes",
    
    # Node Models
    "GraphNodeBase", "PaperNode", "ConceptNode", "AuthorNode", "MethodNode", "InsightNode",
    
    # Edge Models
    "GraphEdgeBase", "CitationEdge", "SimilarityEdge", "BridgeEdge",
    
    # Graph Models
    "KnowledgeGraphSnapshot",
    
    # Analysis Models
    "CentralityAnalysis", "CommunityDetection", "PathAnalysis", "BridgeAnalysis",
    
    # Timeline Models
    "ResearchInsight", "ResearchTimeline",
    
    # Search Models
    "ConceptSearchQuery", "ConceptSearchResult", "NavigationQuery",
    
    # Visualization Models
    "GraphVisualizationConfig", "GraphVisualizationData",
    
    # Request/Response Models
    "GraphBuildRequest", "GraphStatsResponse",
    
    # Integration Models
    "IntegrationSource", "KnowledgeGraphHealth"
]