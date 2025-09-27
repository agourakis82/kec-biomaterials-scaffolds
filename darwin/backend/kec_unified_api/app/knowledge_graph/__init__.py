"""DARWIN Knowledge Graph - Sistema Épico de Knowledge Graph Interdisciplinar

Sistema completo que conecta automaticamente TODOS os domínios científicos 
(biomaterials, neuroscience, philosophy, quantum, psychiatry) em um grafo 
único de conhecimento com visualização e navegação inteligente.

Componentes principais:
- Graph Builder: Construção automática integrando RAG++, Multi-AI, Scientific Discovery
- Concept Linker: Linking interdisciplinar automático entre conceitos  
- Citation Network: Análise de rede de citações e propagação de influência
- Research Timeline: Timeline de insights e descobertas temporais
- Graph Algorithms: Algoritmos avançados de análise (centralidade, comunidades, caminhos)
- Visualization: Visualização web interativa com D3.js, layouts dinâmicos
- Router: Endpoints completos para análise, navegação e gerenciamento
"""

from typing import Any, Dict, List, Optional

# Import models and enums
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes, KnowledgeGraphTypes
)

# Import logging
from ..core.logging import get_logger

# Core engine
from .engine import CrossDomainKnowledgeGraph

# Main router
from .router import router

# Graph Builder - Construção automática
from .graph_builder import (
    KnowledgeGraphBuilder,
    BuilderConfiguration, 
    INTERDISCIPLINARY_BRIDGES
)

# Concept Linker - Linking interdisciplinar
from .concept_linker import (
    InterdisciplinaryConceptLinker,
    LinkingConfiguration,
    CROSS_DOMAIN_CONCEPT_MAPPINGS,
    UNIVERSAL_CONCEPTS
)

# Citation Network - Análise de citações
from .citation_network import (
    CitationNetwork,
    CitationNetworkConfig,
    CITATION_QUALITY_PATTERNS,
    JOURNAL_IMPACT_FACTORS
)

# Research Timeline - Timeline de insights
from .research_timeline import (
    DARWINResearchTimeline,
    TimelineConfiguration,
    InsightType,
    InsightSource,
    INSIGHT_TYPE_KEYWORDS,
    INTERDISCIPLINARY_COMBINATIONS
)

# Graph Algorithms - Análise avançada
from .graph_algorithms import (
    DARWINGraphAlgorithms,
    GraphAlgorithmConfig,
    CentralityAlgorithm,
    CommunityAlgorithm,
    PathAlgorithm,
    KEC_GRAPH_METRICS,
    TREE_SEARCH_TARGETS
)

# Visualization - Interface web interativa
from .visualization import (
    DARWINGraphVisualization,
    VisualizationConfig,
    LayoutType,
    ColorScheme,
    InteractionMode,
    DOMAIN_COLORS,
    NODE_TYPE_COLORS,
    FORCE_SIMULATION_DEFAULTS
)


# ==================== FACTORY FUNCTIONS ====================

def create_knowledge_graph_system(
    builder_config: Optional[BuilderConfiguration] = None,
    linker_config: Optional[LinkingConfiguration] = None,
    citation_config: Optional[CitationNetworkConfig] = None,
    timeline_config: Optional[TimelineConfiguration] = None,
    algorithms_config: Optional[GraphAlgorithmConfig] = None,
    visualization_config: Optional[VisualizationConfig] = None
) -> Dict[str, Any]:
    """
    Factory function para criar sistema completo de Knowledge Graph.
    
    Retorna todos os componentes inicializados e prontos para uso.
    """
    return {
        "graph_builder": KnowledgeGraphBuilder(builder_config or BuilderConfiguration()),
        "concept_linker": InterdisciplinaryConceptLinker(linker_config or LinkingConfiguration()),
        "citation_network": CitationNetwork(citation_config or CitationNetworkConfig()),
        "research_timeline": DARWINResearchTimeline(timeline_config or TimelineConfiguration()),
        "graph_algorithms": DARWINGraphAlgorithms(algorithms_config or GraphAlgorithmConfig()),
        "graph_visualization": DARWINGraphVisualization(visualization_config or VisualizationConfig()),
        "cross_domain_engine": CrossDomainKnowledgeGraph()
    }


def get_default_configurations() -> Dict[str, Any]:
    """
    Retorna configurações padrão para todos os componentes.
    """
    return {
        "builder": BuilderConfiguration(),
        "linker": LinkingConfiguration(),
        "citation": CitationNetworkConfig(),
        "timeline": TimelineConfiguration(),
        "algorithms": GraphAlgorithmConfig(),
        "visualization": VisualizationConfig()
    }


def get_component_info() -> Dict[str, Any]:
    """
    Retorna informações sobre todos os componentes disponíveis.
    """
    return {
        "components": {
            "graph_builder": {
                "description": "Construção automática do Knowledge Graph integrando múltiplas fontes",
                "features": [
                    "Integration com RAG++, Multi-AI, Scientific Discovery",
                    "Node creation automático (papers, conceitos, autores, métodos)",
                    "Edge creation baseado em citações, similaridade, metodologias",
                    "Cross-domain linking automático",
                    "Incremental building support"
                ]
            },
            "concept_linker": {
                "description": "Sistema de linking interdisciplinar entre conceitos",
                "features": [
                    "Semantic similarity usando TF-IDF e embeddings",
                    "Cross-domain concept mapping predefinido",
                    "Universal concept detection",
                    "Pattern-based relationship extraction",
                    "Clustering temporal de conceitos relacionados"
                ]
            },
            "citation_network": {
                "description": "Análise de rede de citações científicas",
                "features": [
                    "PageRank e métricas de centralidade",
                    "Community detection (Louvain, Spectral)",
                    "Bridge paper detection entre domínios",
                    "Influence propagation analysis",
                    "Citation burst detection",
                    "Temporal evolution analysis"
                ]
            },
            "research_timeline": {
                "description": "Timeline de insights e descobertas de pesquisa",
                "features": [
                    "Tracking automático de insights de todas as fontes",
                    "Temporal clustering de insights relacionados",
                    "Milestone detection automático",
                    "Trend analysis e padrões temporais",
                    "Cross-domain evolution tracking",
                    "Research productivity analytics"
                ]
            },
            "graph_algorithms": {
                "description": "Algoritmos avançados de análise de grafos",
                "features": [
                    "Múltiplos algoritmos de centralidade",
                    "Community detection avançado",
                    "Path analysis comprehensive",
                    "Integration com KEC metrics",
                    "Tree Search PUCT integration",
                    "Spectral analysis e decomposições"
                ]
            },
            "graph_visualization": {
                "description": "Visualização web interativa do Knowledge Graph",
                "features": [
                    "Múltiplos layout algorithms (force-directed, hierarchical)",
                    "Dynamic filtering por domínio, tipo, centralidade",
                    "Interactive features (zoom, pan, drag, hover)",
                    "Export para múltiplos formatos (SVG, PNG, JSON, D3.js)",
                    "Real-time updates e animações",
                    "Responsive design"
                ]
            }
        },
        "interdisciplinary_domains": [domain.value for domain in ScientificDomains],
        "supported_node_types": [node_type.value for node_type in NodeTypes],
        "supported_edge_types": [edge_type.value for edge_type in EdgeTypes],
        "graph_types": [graph_type.value for graph_type in KnowledgeGraphTypes]
    }


# ==================== QUICK ACCESS FUNCTIONS ====================

async def quick_build_interdisciplinary_graph() -> Any:
    """
    Função de acesso rápido para construir grafo interdisciplinar básico.
    """
    system = create_knowledge_graph_system()
    builder = system["graph_builder"]
    
    return await builder.build_complete_graph(
        graph_types=[KnowledgeGraphTypes.INTERDISCIPLINARY],
        domains=None,
        force_rebuild=False
    )


async def quick_visualize_graph(
    layout: LayoutType = LayoutType.FORCE_DIRECTED,
    color_scheme: ColorScheme = ColorScheme.DOMAIN
) -> Any:
    """
    Função de acesso rápido para visualizar grafo.
    """
    system = create_knowledge_graph_system()
    visualization = system["graph_visualization"]
    
    # Assuming snapshot is available
    config = VisualizationConfig(
        layout_type=layout,
        color_scheme=color_scheme
    )
    
    return await visualization.generate_visualization_data(layout)


async def quick_search_concepts(query: str, domains: Optional[List[ScientificDomains]] = None) -> List[Any]:
    """
    Função de acesso rápido para buscar conceitos.
    """
    # Implementation would search in current graph
    # This is a placeholder for the actual search functionality
    return []


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Core engine
    "CrossDomainKnowledgeGraph",
    
    # Main router
    "router",
    
    # Components
    "KnowledgeGraphBuilder",
    "InterdisciplinaryConceptLinker", 
    "CitationNetwork",
    "DARWINResearchTimeline",
    "DARWINGraphAlgorithms",
    "DARWINGraphVisualization",
    
    # Configurations
    "BuilderConfiguration",
    "LinkingConfiguration",
    "CitationNetworkConfig", 
    "TimelineConfiguration",
    "GraphAlgorithmConfig",
    "VisualizationConfig",
    
    # Enums
    "InsightType",
    "InsightSource",
    "CentralityAlgorithm",
    "CommunityAlgorithm", 
    "PathAlgorithm",
    "LayoutType",
    "ColorScheme",
    "InteractionMode",
    
    # Constants
    "INTERDISCIPLINARY_BRIDGES",
    "CROSS_DOMAIN_CONCEPT_MAPPINGS",
    "UNIVERSAL_CONCEPTS",
    "CITATION_QUALITY_PATTERNS",
    "JOURNAL_IMPACT_FACTORS",
    "INSIGHT_TYPE_KEYWORDS", 
    "INTERDISCIPLINARY_COMBINATIONS",
    "KEC_GRAPH_METRICS",
    "TREE_SEARCH_TARGETS",
    "DOMAIN_COLORS",
    "NODE_TYPE_COLORS",
    "FORCE_SIMULATION_DEFAULTS",
    
    # Factory functions
    "create_knowledge_graph_system",
    "get_default_configurations",
    "get_component_info",
    
    # Quick access functions
    "quick_build_interdisciplinary_graph",
    "quick_visualize_graph", 
    "quick_search_concepts"
]


# ==================== MODULE METADATA ====================

__version__ = "1.0.0"
__author__ = "DARWIN Research Team"
__description__ = "Sistema épico de Knowledge Graph interdisciplinar para meta-research científica"
__license__ = "MIT"

# Module-level logger
module_logger = get_logger("knowledge_graph")
module_logger.info("🌐 DARWIN Knowledge Graph Module loaded - Ready for interdisciplinary research!")