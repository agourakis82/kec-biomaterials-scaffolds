"""DARWIN Knowledge Graph Router - Endpoints Completos para Knowledge Graph

Router épico que integra TODAS as funcionalidades do Knowledge Graph interdisciplinar:
Graph Builder, Concept Linker, Citation Network, Research Timeline, Graph Algorithms,
e Visualization com endpoints completos para análise, navegação e gerenciamento.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Body, Path
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes, KnowledgeGraphTypes,
    GraphBuildRequest, GraphStatsResponse, KnowledgeGraphHealth,
    ConceptSearchQuery, ConceptSearchResult, NavigationQuery,
    GraphVisualizationConfig, CentralityAnalysis, CommunityDetection
)

# Import all Knowledge Graph components
from .graph_builder import KnowledgeGraphBuilder, BuilderConfiguration
from .concept_linker import InterdisciplinaryConceptLinker, LinkingConfiguration
from .citation_network import CitationNetwork, CitationNetworkConfig
from .research_timeline import DARWINResearchTimeline, TimelineConfiguration, InsightSource
from .graph_algorithms import DARWINGraphAlgorithms, GraphAlgorithmConfig, CentralityAlgorithm, CommunityAlgorithm, PathAlgorithm
from .visualization import DARWINGraphVisualization, VisualizationConfig, LayoutType, ColorScheme

logger = get_logger("knowledge_graph.router")

# Initialize router
router = APIRouter(prefix="/api/v1/knowledge-graph", tags=["Knowledge Graph"])

# Global components - will be initialized on first use
_graph_builder: Optional[KnowledgeGraphBuilder] = None
_concept_linker: Optional[InterdisciplinaryConceptLinker] = None
_citation_network: Optional[CitationNetwork] = None
_research_timeline: Optional[DARWINResearchTimeline] = None
_graph_algorithms: Optional[DARWINGraphAlgorithms] = None
_graph_visualization: Optional[DARWINGraphVisualization] = None

# Cache for latest graph snapshot
_current_snapshot = None
_last_build_time = None

# Component statistics
_component_stats = {
    "total_requests": 0,
    "successful_builds": 0,
    "failed_builds": 0,
    "cache_hits": 0,
    "cache_misses": 0
}


# ==================== INITIALIZATION HELPERS ====================

async def _ensure_components_initialized():
    """Garante que todos os componentes estão inicializados."""
    global _graph_builder, _concept_linker, _citation_network, _research_timeline
    global _graph_algorithms, _graph_visualization
    
    if _graph_builder is None:
        _graph_builder = KnowledgeGraphBuilder(BuilderConfiguration())
        logger.info("🏗️ Graph Builder initialized")
    
    if _concept_linker is None:
        _concept_linker = InterdisciplinaryConceptLinker(LinkingConfiguration())
        logger.info("🔗 Concept Linker initialized")
    
    if _citation_network is None:
        _citation_network = CitationNetwork(CitationNetworkConfig())
        logger.info("📊 Citation Network initialized")
    
    if _research_timeline is None:
        _research_timeline = DARWINResearchTimeline(TimelineConfiguration())
        logger.info("⏰ Research Timeline initialized")
    
    if _graph_algorithms is None:
        _graph_algorithms = DARWINGraphAlgorithms(GraphAlgorithmConfig())
        logger.info("🧮 Graph Algorithms initialized")
    
    if _graph_visualization is None:
        _graph_visualization = DARWINGraphVisualization(VisualizationConfig())
        logger.info("🎨 Graph Visualization initialized")


# ==================== CORE GRAPH ENDPOINTS ====================

@router.get("/health", response_model=KnowledgeGraphHealth)
async def health_check():
    """
    Health check completo do Knowledge Graph.
    
    Retorna status de todos os componentes e estatísticas gerais.
    """
    try:
        _component_stats["total_requests"] += 1
        
        await _ensure_components_initialized()
        
        # Check all components
        components_status = {
            "graph_builder": "operational" if _graph_builder else "not_initialized",
            "concept_linker": "operational" if _concept_linker else "not_initialized",
            "citation_network": "operational" if _citation_network else "not_initialized",
            "research_timeline": "operational" if _research_timeline else "not_initialized",
            "graph_algorithms": "operational" if _graph_algorithms else "not_initialized",
            "graph_visualization": "operational" if _graph_visualization else "not_initialized"
        }
        
        # Calculate graph stats if snapshot exists
        graph_stats = None
        if _current_snapshot:
            graph_stats = GraphStatsResponse(
                total_nodes=len(_current_snapshot.nodes),
                total_edges=len(_current_snapshot.edges),
                nodes_by_type={},  # Will be calculated properly in real implementation
                nodes_by_domain={},
                edges_by_type={},
                graph_density=0.0,
                average_degree=0.0,
                connected_components=1,
                interdisciplinary_connections=0,
                last_updated=_last_build_time or datetime.utcnow()
            )
        
        # Overall health
        all_operational = all(status == "operational" for status in components_status.values())
        
        return KnowledgeGraphHealth(
            healthy=all_operational,
            components=components_status,
            graph_stats=graph_stats,
            integration_sources=[],  # Will be populated with real sources
            last_check=datetime.utcnow(),
            errors=[] if all_operational else ["Some components not operational"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/full")
async def get_full_graph():
    """
    Retorna o Knowledge Graph completo em formato JSON.
    
    Inclui todos os nós, arestas e metadados do grafo atual.
    """
    try:
        _component_stats["total_requests"] += 1
        
        if _current_snapshot is None:
            # Build graph if not exists
            await build_knowledge_graph(GraphBuildRequest(
                graph_types=[KnowledgeGraphTypes.INTERDISCIPLINARY],
                domains=None,
                force_rebuild=False
            ))
        
        if _current_snapshot:
            _component_stats["cache_hits"] += 1
            return {
                "snapshot": _current_snapshot.dict(),
                "build_time": _last_build_time.isoformat() if _last_build_time else None,
                "status": "success"
            }
        else:
            _component_stats["cache_misses"] += 1
            raise HTTPException(status_code=404, detail="No graph data available")
            
    except Exception as e:
        logger.error(f"Get full graph failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {str(e)}")


@router.get("/domains")
async def get_domains_view(
    domains: Optional[List[ScientificDomains]] = Query(None)
):
    """
    Retorna visualização do grafo filtrada por domínios científicos.
    
    Permite filtrar por domínios específicos para análise focada.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available. Build graph first.")
        
        # Initialize visualization with domain filter
        config = VisualizationConfig(
            filter_domains=domains,
            layout_type=LayoutType.DOMAIN_SEPARATED,
            color_scheme=ColorScheme.DOMAIN,
            max_nodes=500
        )
        
        await _graph_visualization.initialize_from_snapshot(_current_snapshot, config)
        viz_data = await _graph_visualization.generate_visualization_data()
        
        return {
            "visualization_data": viz_data.dict(),
            "filtered_domains": domains,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Get domains view failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domains view: {str(e)}")


@router.get("/connections")
async def get_interdisciplinary_connections():
    """
    Retorna conexões interdisciplinares do Knowledge Graph.
    
    Foca especificamente em conexões que atravessam domínios científicos.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Get bridge analysis from concept linker
        concepts = [node for node in _current_snapshot.nodes if hasattr(node, 'type') and node.type.value == 'concept']
        
        if concepts:
            # Analyze concept connectivity
            concept_links = await _concept_linker.create_concept_links(concepts)
            connectivity_analysis = await _concept_linker.analyze_concept_connectivity(concepts, concept_links)
            
            # Filter for interdisciplinary connections
            interdisciplinary_links = [
                link for link in concept_links 
                if hasattr(link, 'properties') and link.properties.get('bridge_type')
            ]
            
            return {
                "interdisciplinary_connections": len(interdisciplinary_links),
                "connectivity_analysis": connectivity_analysis,
                "top_bridge_concepts": interdisciplinary_links[:20],
                "status": "success"
            }
        else:
            return {
                "interdisciplinary_connections": 0,
                "connectivity_analysis": {},
                "top_bridge_concepts": [],
                "status": "no_concepts_available"
            }
        
    except Exception as e:
        logger.error(f"Get connections failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get connections: {str(e)}")


@router.get("/timeline")
async def get_research_timeline(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    domains: Optional[List[ScientificDomains]] = Query(None)
):
    """
    Retorna timeline de insights e descobertas de pesquisa.
    
    Mostra a evolução temporal dos insights interdisciplinares.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        # Create timeline with filters
        timeline_id = await _research_timeline.create_timeline(
            title="DARWIN Research Timeline",
            domain_filter=domains,
            start_date=start_date,
            end_date=end_date
        )
        
        # Analyze timeline patterns
        analysis = await _research_timeline.analyze_timeline_patterns(timeline_id)
        
        # Get timeline statistics
        stats = _research_timeline.get_timeline_statistics()
        
        return {
            "timeline_id": timeline_id,
            "analysis": analysis,
            "statistics": stats,
            "filters_applied": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "domains": domains
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Get timeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@router.get("/visualization")
async def get_visualization(
    layout: Optional[LayoutType] = Query(LayoutType.FORCE_DIRECTED),
    color_scheme: Optional[ColorScheme] = Query(ColorScheme.DOMAIN),
    max_nodes: Optional[int] = Query(500),
    domains: Optional[List[ScientificDomains]] = Query(None)
):
    """
    Retorna visualização web interativa do Knowledge Graph.
    
    Configurável com múltiplos layouts, esquemas de cores e filtros.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Configure visualization
        config = VisualizationConfig(
            layout_type=layout,
            color_scheme=color_scheme,
            filter_domains=domains,
            max_nodes=max_nodes,
            enable_zoom=True,
            enable_pan=True,
            enable_hover=True
        )
        
        await _graph_visualization.initialize_from_snapshot(_current_snapshot, config)
        viz_data = await _graph_visualization.generate_visualization_data(layout)
        
        return {
            "visualization": viz_data.dict(),
            "config": {
                "layout": layout.value,
                "color_scheme": color_scheme.value,
                "max_nodes": max_nodes,
                "filtered_domains": domains
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Get visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get visualization: {str(e)}")


# ==================== ANALYSIS ENDPOINTS ====================

@router.post("/analyze/centrality")
async def analyze_centrality(
    algorithms: Optional[List[CentralityAlgorithm]] = Body(None),
    top_k: Optional[int] = Body(20),
    normalize: Optional[bool] = Body(True)
):
    """
    Realiza análise de centralidade no Knowledge Graph.
    
    Suporte a múltiplos algoritmos de centralidade para identificar nós importantes.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Initialize graph algorithms
        await _graph_algorithms.initialize_from_snapshot(_current_snapshot)
        
        # Analyze centrality
        centrality_results = await _graph_algorithms.analyze_centrality(
            algorithms=algorithms,
            top_k=top_k,
            normalize=normalize
        )
        
        return {
            "centrality_analysis": {
                alg: analysis.dict() for alg, analysis in centrality_results.items()
            },
            "parameters": {
                "algorithms": [alg.value for alg in (algorithms or [])],
                "top_k": top_k,
                "normalize": normalize
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Centrality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Centrality analysis failed: {str(e)}")


@router.post("/analyze/communities")
async def detect_communities(
    algorithms: Optional[List[CommunityAlgorithm]] = Body(None),
    min_size: Optional[int] = Body(3)
):
    """
    Detecta comunidades no Knowledge Graph.
    
    Utiliza múltiplos algoritmos para identificar grupos de conceitos relacionados.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        await _graph_algorithms.initialize_from_snapshot(_current_snapshot)
        
        # Detect communities
        communities_results = await _graph_algorithms.detect_communities(
            algorithms=algorithms,
            min_size=min_size
        )
        
        return {
            "community_detection": communities_results,
            "parameters": {
                "algorithms": [alg.value for alg in (algorithms or [])],
                "min_community_size": min_size
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Community detection failed: {str(e)}")


@router.post("/analyze/paths")
async def analyze_paths(
    source_nodes: Optional[List[str]] = Body(None),
    target_nodes: Optional[List[str]] = Body(None),
    algorithms: Optional[List[PathAlgorithm]] = Body(None)
):
    """
    Analisa caminhos no Knowledge Graph.
    
    Encontra caminhos mais curtos, diâmetro, e outras métricas de conectividade.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        await _graph_algorithms.initialize_from_snapshot(_current_snapshot)
        
        # Analyze paths
        path_results = await _graph_algorithms.analyze_paths(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            algorithms=algorithms
        )
        
        return {
            "path_analysis": path_results,
            "parameters": {
                "source_nodes": source_nodes,
                "target_nodes": target_nodes,
                "algorithms": [alg.value for alg in (algorithms or [])]
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Path analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Path analysis failed: {str(e)}")


@router.get("/analyze/bridges")
async def get_bridge_analysis():
    """
    Identifica conceitos ponte entre domínios científicos.
    
    Encontra nós que conectam diferentes domínios interdisciplinares.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Get paper nodes for bridge analysis
        papers = [node for node in _current_snapshot.nodes if hasattr(node, 'type') and node.type.value == 'paper']
        citations = [edge for edge in _current_snapshot.edges if hasattr(edge, 'type') and edge.type.value == 'cites']
        
        if papers and citations:
            # Build citation network
            citation_analysis = await _citation_network.build_citation_network(papers, citations, [])
            
            bridge_results = citation_analysis.get("bridge_analysis", {})
            
            return {
                "bridge_analysis": bridge_results,
                "total_papers_analyzed": len(papers),
                "total_citations_analyzed": len(citations),
                "status": "success"
            }
        else:
            return {
                "bridge_analysis": {},
                "total_papers_analyzed": 0,
                "total_citations_analyzed": 0,
                "status": "insufficient_data"
            }
        
    except Exception as e:
        logger.error(f"Bridge analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bridge analysis failed: {str(e)}")


@router.post("/analyze/influence")
async def analyze_influence(
    source_papers: Optional[List[str]] = Body(None),
    max_hops: Optional[int] = Body(3)
):
    """
    Analisa propagação de influência no Knowledge Graph.
    
    Rastreia como ideias se espalham através da rede de citações.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        papers = [node for node in _current_snapshot.nodes if hasattr(node, 'type') and node.type.value == 'paper']
        citations = [edge for edge in _current_snapshot.edges if hasattr(edge, 'type') and edge.type.value == 'cites']
        
        if papers and citations:
            citation_analysis = await _citation_network.build_citation_network(papers, citations, [])
            influence_results = citation_analysis.get("influence_analysis", {})
            
            return {
                "influence_analysis": influence_results,
                "parameters": {
                    "source_papers": source_papers,
                    "max_hops": max_hops
                },
                "status": "success"
            }
        else:
            return {
                "influence_analysis": {},
                "status": "insufficient_data"
            }
        
    except Exception as e:
        logger.error(f"Influence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Influence analysis failed: {str(e)}")


# ==================== MANAGEMENT ENDPOINTS ====================

@router.post("/rebuild")
async def build_knowledge_graph(
    request: GraphBuildRequest
):
    """
    Constrói ou reconstrói o Knowledge Graph completo.
    
    Integra dados de RAG++, Multi-AI, Scientific Discovery, Score Contracts,
    KEC Metrics e Tree Search para construir grafo interdisciplinar.
    """
    try:
        global _current_snapshot, _last_build_time
        
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        logger.info(f"🚀 Starting Knowledge Graph build with types: {request.graph_types}")
        
        # Build the graph
        snapshot = await _graph_builder.build_complete_graph(
            graph_types=request.graph_types,
            domains=request.domains,
            force_rebuild=request.force_rebuild
        )
        
        # Update global state
        _current_snapshot = snapshot
        _last_build_time = datetime.utcnow()
        _component_stats["successful_builds"] += 1
        
        # Initialize other components with new snapshot
        if _graph_algorithms:
            await _graph_algorithms.initialize_from_snapshot(snapshot)
        
        if _graph_visualization:
            await _graph_visualization.initialize_from_snapshot(snapshot)
        
        logger.info("✅ Knowledge Graph build completed successfully")
        
        return {
            "snapshot_id": snapshot.id,
            "build_time": _last_build_time.isoformat(),
            "total_nodes": len(snapshot.nodes),
            "total_edges": len(snapshot.edges),
            "graph_types": [gt.value for gt in request.graph_types],
            "domains": [d.value for d in request.domains] if request.domains else None,
            "metadata": snapshot.metadata,
            "status": "success"
        }
        
    except Exception as e:
        _component_stats["failed_builds"] += 1
        logger.error(f"❌ Knowledge Graph build failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph build failed: {str(e)}")


@router.post("/concepts/link")
async def link_concepts_manually(
    concept1: str = Body(...),
    concept2: str = Body(...),
    relationship: str = Body(...),
    confidence: Optional[float] = Body(0.8)
):
    """
    Cria link manual entre dois conceitos.
    
    Permite curadorção manual de conexões conceituais específicas.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        # In a real implementation, this would add the link to the graph
        # For now, return success with the link information
        
        link_id = f"manual_link_{concept1}_{concept2}"
        
        return {
            "link_id": link_id,
            "concept1": concept1,
            "concept2": concept2,
            "relationship": relationship,
            "confidence": confidence,
            "created_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Manual concept linking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Concept linking failed: {str(e)}")


@router.delete("/cache/clear")
async def clear_cache():
    """
    Limpa todos os caches do Knowledge Graph.
    
    Remove dados cached para forçar reconstrução na próxima requisição.
    """
    try:
        global _current_snapshot, _last_build_time
        
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        # Clear global state
        _current_snapshot = None
        _last_build_time = None
        
        # Clear component caches
        if _graph_algorithms:
            _graph_algorithms.clear_caches()
        
        # Reset component statistics
        cache_cleared_stats = _component_stats.copy()
        _component_stats.update({
            "total_requests": 0,
            "successful_builds": 0,
            "failed_builds": 0,
            "cache_hits": 0,
            "cache_misses": 0
        })
        
        return {
            "cache_cleared": True,
            "cleared_at": datetime.utcnow().isoformat(),
            "previous_stats": cache_cleared_stats,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.post("/export")
async def export_graph(
    format_type: str = Body(...),
    layout: Optional[LayoutType] = Body(LayoutType.FORCE_DIRECTED),
    include_metadata: Optional[bool] = Body(True)
):
    """
    Exporta Knowledge Graph em diferentes formatos.
    
    Suporte a JSON, SVG, PNG, Graphviz DOT, D3.js, Cytoscape.js.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Initialize visualization
        await _graph_visualization.initialize_from_snapshot(_current_snapshot)
        
        # Export in requested format
        exported_data = await _graph_visualization.export_visualization(
            format_type=format_type,
            layout_type=layout
        )
        
        # Determine response type based on format
        if format_type.lower() in ["svg", "dot", "graphviz"]:
            return PlainTextResponse(content=exported_data)
        elif format_type.lower() in ["json", "d3", "cytoscape"]:
            return JSONResponse(content=exported_data)
        else:
            return Response(content=exported_data)
        
    except Exception as e:
        logger.error(f"Graph export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/stats")
async def get_graph_statistics():
    """
    Retorna estatísticas detalhadas do Knowledge Graph.
    
    Inclui métricas de rede, distribuições e estatísticas de performance.
    """
    try:
        _component_stats["total_requests"] += 1
        
        if _current_snapshot is None:
            return {
                "graph_stats": None,
                "component_stats": _component_stats,
                "build_info": {
                    "last_build": None,
                    "graph_available": False
                },
                "status": "no_graph_available"
            }
        
        # Calculate basic stats
        nodes_by_type = {}
        nodes_by_domain = {}
        edges_by_type = {}
        
        for node in _current_snapshot.nodes:
            node_type = getattr(node, 'type', None)
            if node_type:
                nodes_by_type[node_type.value] = nodes_by_type.get(node_type.value, 0) + 1
            
            domain = getattr(node, 'domain', None)
            if domain:
                nodes_by_domain[domain.value] = nodes_by_domain.get(domain.value, 0) + 1
        
        for edge in _current_snapshot.edges:
            edge_type = getattr(edge, 'type', None)
            if edge_type:
                edges_by_type[edge_type.value] = edges_by_type.get(edge_type.value, 0) + 1
        
        # Create stats response
        graph_stats = GraphStatsResponse(
            total_nodes=len(_current_snapshot.nodes),
            total_edges=len(_current_snapshot.edges),
            nodes_by_type=nodes_by_type,
            nodes_by_domain=nodes_by_domain,
            edges_by_type=edges_by_type,
            graph_density=0.0,  # Would be calculated properly
            average_degree=0.0,  # Would be calculated properly
            connected_components=1,  # Would be calculated properly
            interdisciplinary_connections=0,  # Would be calculated properly
            last_updated=_last_build_time or datetime.utcnow()
        )
        
        return {
            "graph_stats": graph_stats.dict(),
            "component_stats": _component_stats,
            "build_info": {
                "last_build": _last_build_time.isoformat() if _last_build_time else None,
                "graph_available": True,
                "snapshot_id": _current_snapshot.id
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ==================== SEARCH & NAVIGATION ENDPOINTS ====================

@router.post("/search/concepts")
async def search_concepts(
    query: ConceptSearchQuery
):
    """
    Busca conceitos no Knowledge Graph.
    
    Suporte a busca semântica, filtros por domínio e ranking por relevância.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Get concept nodes
        concept_nodes = [
            node for node in _current_snapshot.nodes 
            if hasattr(node, 'type') and node.type.value == 'concept'
        ]
        
        # Simple search implementation
        results = []
        query_lower = query.query.lower()
        
        for node in concept_nodes[:query.limit]:
            # Check domain filter
            if query.domains and hasattr(node, 'domain') and node.domain not in query.domains:
                continue
            
            # Check node type filter
            if query.node_types and hasattr(node, 'type') and node.type not in query.node_types:
                continue
            
            # Simple text matching
            node_text = getattr(node, 'label', '').lower()
            if query_lower in node_text:
                score = len(query_lower) / max(len(node_text), 1)  # Simple relevance score
                
                if score >= query.similarity_threshold:
                    results.append(ConceptSearchResult(
                        node=node,
                        score=score,
                        explanation=f"Text match in label"
                    ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return {
            "results": [result.dict() for result in results],
            "total_results": len(results),
            "query": query.dict(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Concept search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/navigate/shortest-path")
async def find_shortest_path(
    source: str = Body(...),
    target: str = Body(...),
    max_length: Optional[int] = Body(10)
):
    """
    Encontra caminho mais curto entre dois conceitos.
    
    Navegação inteligente através do grafo de conhecimento.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        await _graph_algorithms.initialize_from_snapshot(_current_snapshot)
        
        # Find shortest path
        path_analysis = await _graph_algorithms.analyze_paths(
            source_nodes=[source],
            target_nodes=[target],
            algorithms=[PathAlgorithm.SHORTEST_PATH]
        )
        
        return {
            "source": source,
            "target": target,
            "path_analysis": path_analysis,
            "max_length": max_length,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Shortest path failed: {e}")
        raise HTTPException(status_code=500, detail=f"Path finding failed: {str(e)}")


@router.get("/explore/domain/{domain}")
async def explore_domain(
    domain: ScientificDomains = Path(...),
    layout: Optional[LayoutType] = Query(LayoutType.DOMAIN_SEPARATED),
    max_nodes: Optional[int] = Query(100)
):
    """
    Explora Knowledge Graph filtrado por domínio científico específico.
    
    Visualização focada em um domínio com conexões interdisciplinares.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Configure visualization for domain exploration
        config = VisualizationConfig(
            layout_type=layout,
            color_scheme=ColorScheme.DOMAIN,
            filter_domains=[domain],
            max_nodes=max_nodes,
            enable_hover=True
        )
        
        await _graph_visualization.initialize_from_snapshot(_current_snapshot, config)
        viz_data = await _graph_visualization.generate_visualization_data(layout)
        
        # Get domain-specific statistics
        domain_nodes = [
            node for node in _current_snapshot.nodes
            if hasattr(node, 'domain') and node.domain == domain
        ]
        
        return {
            "domain": domain.value,
            "visualization": viz_data.dict(),
            "domain_stats": {
                "total_nodes": len(domain_nodes),
                "layout_used": layout.value,
                "max_nodes_limit": max_nodes
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Domain exploration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Domain exploration failed: {str(e)}")


@router.post("/recommend/connections")
async def recommend_connections(
    concept_id: str = Body(...),
    max_recommendations: Optional[int] = Body(10),
    min_confidence: Optional[float] = Body(0.6)
):
    """
    Recomenda conexões conceituais baseadas em análise do grafo.
    
    Usa algoritmos de análise para sugerir conexões potenciais.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        if _current_snapshot is None:
            raise HTTPException(status_code=404, detail="No graph data available")
        
        # Get concept node
        concept_node = None
        for node in _current_snapshot.nodes:
            if node.id == concept_id:
                concept_node = node
                break
        
        if not concept_node:
            raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
        
        # Get concept nodes for analysis
        concepts = [
            node for node in _current_snapshot.nodes 
            if hasattr(node, 'type') and node.type.value == 'concept' and node.id != concept_id
        ]
        
        # Generate concept links to find recommendations
        concept_links = await _concept_linker.create_concept_links([concept_node] + concepts[:50])
        
        # Filter links involving the target concept
        recommendations = []
        for link in concept_links:
            if (link.source == concept_id or link.target == concept_id) and link.weight >= min_confidence:
                other_concept_id = link.target if link.source == concept_id else link.source
                other_concept = next(
                    (node for node in concepts if node.id == other_concept_id), 
                    None
                )
                
                if other_concept:
                    recommendations.append({
                        "concept_id": other_concept.id,
                        "concept_label": getattr(other_concept, 'label', other_concept.id),
                        "confidence": link.weight,
                        "relationship_type": getattr(link, 'type', 'unknown'),
                        "explanation": f"Connection strength: {link.weight:.2f}"
                    })
        
        # Sort by confidence and limit
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        recommendations = recommendations[:max_recommendations]
        
        return {
            "concept_id": concept_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "parameters": {
                "max_recommendations": max_recommendations,
                "min_confidence": min_confidence
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Connection recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


# ==================== TIMELINE ENDPOINTS ====================

@router.post("/timeline/add-insight")
async def add_research_insight(
    content: str = Body(...),
    source: InsightSource = Body(...),
    domains: List[ScientificDomains] = Body(...),
    confidence: Optional[float] = Body(0.7),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """
    Adiciona novo insight à timeline de pesquisa.
    
    Permite tracking manual de insights de diferentes fontes.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        # Add insight to timeline
        insight_id = await _research_timeline.add_insight(
            content=content,
            source=source,
            domains=domains,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        return {
            "insight_id": insight_id,
            "content": content,
            "source": source.value,
            "domains": [d.value for d in domains],
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Add insight failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add insight: {str(e)}")


# ==================== PERFORMANCE ENDPOINT ====================

@router.get("/performance")
async def get_performance_metrics():
    """
    Retorna métricas de performance dos algoritmos e componentes.
    
    Útil para monitoring e otimização do sistema.
    """
    try:
        _component_stats["total_requests"] += 1
        await _ensure_components_initialized()
        
        performance_data = {
            "component_stats": _component_stats,
            "build_info": {
                "last_build": _last_build_time.isoformat() if _last_build_time else None,
                "graph_available": _current_snapshot is not None,
                "snapshot_id": _current_snapshot.id if _current_snapshot else None
            }
        }
        
        # Add algorithm performance if available
        if _graph_algorithms:
            algo_performance = _graph_algorithms.get_algorithm_performance()
            performance_data["algorithm_performance"] = algo_performance
        
        # Add visualization info if available
        if _graph_visualization:
            viz_info = _graph_visualization.get_visualization_info()
            performance_data["visualization_performance"] = viz_info
        
        # Add timeline stats if available
        if _research_timeline:
            timeline_stats = _research_timeline.get_timeline_statistics()
            performance_data["timeline_performance"] = timeline_stats
        
        return {
            "performance_metrics": performance_data,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Get performance metrics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


# Export router
__all__ = ["router"]