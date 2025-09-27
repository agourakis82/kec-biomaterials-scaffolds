"""RAG++ Router Consolidado

Router FastAPI unificado consolidando todas as funcionalidades dos backends
Principal e Darwin com integração Vertex AI e busca científica avançada.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

# Try to import full services, fallback to simple versions
try:
    from ..services.rag_engine import get_rag_engine, RAGEngine
    RAG_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Full RAG engine not available: {e}")
    RAG_ENGINE_AVAILABLE = False

try:
    from ..services.scientific_search import get_scientific_search_service, ScientificSearchService
    SCIENTIFIC_SEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scientific search not available: {e}")
    SCIENTIFIC_SEARCH_AVAILABLE = False

try:
    from ..services.vertex_ai_client import get_vertex_client, VertexAIClient
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vertex AI not available: {e}")
    VERTEX_AI_AVAILABLE = False

# Import simple fallback
from ..services.simple_rag_engine import get_simple_rag_engine, SimpleRAGEngine
from ..models.rag_models import (
    # Request models
    UnifiedRAGRequest, ScientificSearchRequest, BiomaterialsQueryRequest,
    CrossDomainRequest, DiscoveryRequest, DocumentRequest,
    RAGPlusQuery, IterativeRAGRequest, RAGSearchRequest,
    
    # Response models
    UnifiedRAGResponse, DiscoveryResponse, ServiceStatus,
    RAGPlusResponse, IterativeRAGResponse, RAGSearchResponse,
    HealthStatus, PerformanceMetrics,
    
    # Enums
    SearchMethod, QueryDomain, SourceType
)

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/rag-plus",
    tags=["RAG++ Enhanced Research"],
    responses={
        500: {"description": "Internal server error"},
        422: {"description": "Validation error"}
    }
)

# Dependency injection helpers with fallbacks
async def get_rag_service():
    """Dependency para RAG Engine (com fallback)"""
    try:
        if RAG_ENGINE_AVAILABLE:
            return await get_rag_engine()
        else:
            return await get_simple_rag_engine()
    except Exception as e:
        logger.error(f"Failed to get RAG service: {e}")
        # Try simple engine as last resort
        return await get_simple_rag_engine()

async def get_scientific_service():
    """Dependency para Scientific Search Service"""
    if not SCIENTIFIC_SEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scientific search not available")
    try:
        return await get_scientific_search_service()
    except Exception as e:
        logger.error(f"Failed to get scientific search service: {e}")
        raise HTTPException(status_code=503, detail="Scientific search unavailable")

async def get_vertex_service():
    """Dependency para Vertex AI Client"""
    if not VERTEX_AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vertex AI not available")
    try:
        return await get_vertex_client()
    except Exception as e:
        logger.error(f"Failed to get Vertex AI client: {e}")
        raise HTTPException(status_code=503, detail="Vertex AI unavailable")


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check rápido para RAG++ service.
    
    Retorna status básico de saúde sem testes abrangentes.
    Use /status para informações detalhadas de saúde.
    """
    try:
        # Quick health checks with fallbacks
        rag_service = await get_rag_service()
        rag_health = await rag_service.get_health_status()
        
        vertex_health = {"healthy": False}
        if VERTEX_AI_AVAILABLE:
            try:
                vertex_client = await get_vertex_client()
                vertex_health = await vertex_client.health_check()
            except:
                pass
        
        sci_health = {"healthy": False}
        if SCIENTIFIC_SEARCH_AVAILABLE:
            try:
                scientific_service = await get_scientific_search_service()
                sci_health = await scientific_service.get_health_status()
            except:
                pass
        
        # Aggregate health
        overall_healthy = (
            rag_health.get("healthy", False) and
            vertex_health.get("healthy", False) and
            sci_health.get("healthy", False)
        )
        
        components = {
            "rag_engine": rag_health.get("healthy", False),
            "vertex_ai": vertex_health.get("healthy", False),
            "scientific_search": sci_health.get("healthy", False)
        }
        
        return HealthStatus(
            healthy=overall_healthy,
            components=components,
            errors=[],
            warnings=[] if overall_healthy else ["Some components unhealthy"],
            last_check=datetime.now(),
            uptime_seconds=0.0  # TODO: Track actual uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthStatus(
            healthy=False,
            components={"error": False},
            errors=[str(e)],
            warnings=["Health check failed"],
            last_check=datetime.now(),
            uptime_seconds=0.0
        )


@router.get("/status", response_model=ServiceStatus)
async def get_service_status(
    rag_service = Depends(get_rag_service)
):
    """
    Status abrangente do serviço RAG++.
    
    Retorna informações detalhadas de saúde para todos os componentes
    incluindo BigQuery, modelos Vertex AI e monitoramento de discovery.
    """
    try:
        # Get detailed status from available services
        rag_status = await rag_service.get_health_status()
        
        components = {
            "rag_service": rag_status,
            "scientific_search": SCIENTIFIC_SEARCH_AVAILABLE,
            "vertex_ai": VERTEX_AI_AVAILABLE
        }
        
        # Configuration info (simplified for fallback mode)
        configuration = {
            "mode": "simple" if not RAG_ENGINE_AVAILABLE else "full",
            "rag_engine_available": RAG_ENGINE_AVAILABLE,
            "scientific_search_available": SCIENTIFIC_SEARCH_AVAILABLE,
            "vertex_ai_available": VERTEX_AI_AVAILABLE,
            "document_count": rag_status.get("document_count", 0)
        }
        
        # Overall status
        overall_status = "healthy" if rag_status.get("healthy", False) else "degraded"
        
        return ServiceStatus(
            service="rag_plus_unified",
            status=overall_status,
            components=components,
            configuration=configuration,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/metrics")
async def get_performance_metrics():
    """
    Métricas de performance do sistema RAG++.
    
    Retorna métricas básicas disponíveis no modo atual.
    """
    try:
        if VERTEX_AI_AVAILABLE:
            vertex_client = await get_vertex_client()
            vertex_metrics = vertex_client.get_performance_metrics()
            return vertex_metrics
        else:
            # Return basic metrics in fallback mode
            return {
                "mode": "simple",
                "message": "Limited metrics available in simple mode",
                "vertex_ai_available": False
            }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {
            "error": True,
            "message": str(e)
        }


# =============================================================================
# CORE RAG ENDPOINTS (Migrados do Backend Principal)
# =============================================================================

@router.post("/query", response_model=RAGPlusResponse)
async def query_rag_plus(
    request: RAGPlusQuery,
    rag_service = Depends(get_rag_service)
):
    """
    Query RAG++ com retrieval-augmented generation simples.
    
    Usa busca semântica para encontrar documentos relevantes e gera
    resposta com citações da base de conhecimento DARWIN.
    """
    try:
        logger.info(f"RAG++ query: {request.query[:100]}...")
        start_time = time.time()
        
        # Convert to unified request
        unified_request = UnifiedRAGRequest(
            query=request.query,
            method=SearchMethod.SIMPLE,
            domain=None,
            top_k=request.top_k or 5,
            max_iterations=None,
            include_sources=request.include_sources,
            cross_domain=False,
            scientific_validation=True,
            real_time_discovery=False
        )
        
        # Execute unified query
        result = await rag_service.unified_query(unified_request)
        
        # Convert to legacy format
        response = RAGPlusResponse(
            query=result.query,
            answer=result.answer,
            method=result.method.value,
            sources=result.sources if request.include_sources else [],
            retrieved_docs=len(result.sources),
            reasoning_steps=result.reasoning_trace,
            total_steps=len(result.reasoning_trace) if result.reasoning_trace else 0
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"RAG++ query completed in {elapsed_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG++ query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/iterative", response_model=RAGPlusResponse)
async def query_iterative(
    request: IterativeRAGRequest,
    rag_service = Depends(get_rag_service)
):
    """
    Query RAG++ com raciocínio iterativo ReAct.
    
    Usa loops Thought→Action→Observation para questões complexas
    que requerem raciocínio multi-step e coleta de informações.
    """
    try:
        logger.info(f"RAG++ iterative query: {request.query[:100]}...")
        start_time = time.time()
        
        # Convert to unified request
        unified_request = UnifiedRAGRequest(
            query=request.query,
            method=SearchMethod.ITERATIVE,
            domain=None,
            top_k=request.top_k or 5,
            max_iterations=request.max_iterations,
            include_sources=request.include_sources,
            cross_domain=False,
            scientific_validation=True,
            real_time_discovery=False
        )
        
        # Execute iterative query
        result = await rag_service.unified_query(unified_request)
        
        # Convert to legacy format
        response = RAGPlusResponse(
            query=result.query,
            answer=result.answer,
            method=result.method.value,
            sources=result.sources if request.include_sources else [],
            retrieved_docs=len(result.sources),
            reasoning_steps=result.reasoning_trace,
            total_steps=len(result.reasoning_trace) if result.reasoning_trace else 0
        )
        
        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"RAG++ iterative query completed in {elapsed_time:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG++ iterative query: {e}")
        raise HTTPException(status_code=500, detail=f"Iterative query failed: {str(e)}")


@router.post("/documents")
async def add_document(
    request: DocumentRequest,
    rag_service = Depends(get_rag_service)
):
    """
    Adiciona documento à base de conhecimento RAG++.
    
    Indexa manualmente um documento com conteúdo e metadados
    na base de conhecimento DARWIN para uso em queries futuras.
    """
    try:
        logger.info(f"Adding document with {len(request.content)} characters")
        
        # Prepare metadata
        metadata = request.metadata or {}
        metadata.update({
            "source": request.source or "manual_addition",
            "domain": request.domain.value if request.domain else None,
            "added_via": "api",
            "added_at": datetime.now().isoformat()
        })
        
        # Index document
        doc_id = await rag_service.index_document(request.content, metadata)
        
        return {
            "status": "added",
            "document_id": doc_id,
            "message": "Document added to knowledge base successfully",
            "content_length": len(request.content),
            "metadata_keys": list(metadata.keys())
        }
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    domain: Optional[QueryDomain] = Query(None, description="Domain filter"),
    rag_service = Depends(get_rag_service)
):
    """
    Busca direta na base de conhecimento RAG++.
    
    Executa busca semântica sem geração de resposta,
    retornando documentos mais relevantes com scores de similaridade.
    """
    try:
        logger.info(f"Knowledge base search: {query[:100]}...")
        
        results = await rag_service.search_documents(query, top_k, domain)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "doc_id": result.doc_id,
                "score": result.score,
                "title": result.metadata.get("title", ""),
                "content": result.metadata.get("content", "")[:500] + "...",
                "source": result.metadata.get("source", ""),
                "domain": result.metadata.get("domain", ""),
                "url": result.metadata.get("url", "")
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "domain_filter": domain.value if domain else None
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# =============================================================================
# DARWIN ENDPOINTS (Migrados)
# =============================================================================

@router.post("/search", response_model=RAGSearchResponse)
async def rag_search_darwin(
    request: RAGSearchRequest,
    rag_service = Depends(get_rag_service)
):
    """
    Busca RAG+ compatível com Darwin.
    
    Endpoint de compatibilidade com a API Darwin original
    para busca com geração de resposta.
    """
    try:
        logger.info(f"Darwin RAG search: {request.q}")
        
        # Convert to unified request
        unified_request = UnifiedRAGRequest(
            query=request.q,
            method=SearchMethod.SIMPLE,
            domain=request.domain,
            top_k=request.k,
            max_iterations=None,
            include_sources=True,
            cross_domain=False,
            scientific_validation=True,
            real_time_discovery=False
        )
        
        result = await rag_service.unified_query(unified_request)
        
        # Format results for Darwin compatibility
        formatted_results = []
        for source in result.sources:
            formatted_results.append({
                "doc_id": source["doc_id"],
                "score": source["score"],
                "metadata": {
                    "title": source["title"],
                    "content": source.get("abstract", ""),
                    "source": source["source"],
                    "url": source["url"]
                }
            })
        
        return RAGSearchResponse(
            query=result.query,
            answer=result.answer,
            results=formatted_results
        )
        
    except Exception as e:
        logger.error(f"Darwin RAG search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# =============================================================================
# NOVOS ENDPOINTS UNIFICADOS
# =============================================================================

@router.post("/unified", response_model=UnifiedRAGResponse)
async def unified_query(
    request: UnifiedRAGRequest,
    rag_service = Depends(get_rag_service)
):
    """
    Query RAG++ unificado com todos os métodos.
    
    Endpoint principal que consolida todas as funcionalidades:
    - Busca simples, iterativa, científica e cross-domain
    - Validação científica automática
    - Discovery em tempo real
    - Análise de redes de citação
    """
    try:
        logger.info(f"Unified RAG++ query: {request.query[:100]}... (method: {request.method.value})")
        
        result = await rag_service.unified_query(request)
        return result
        
    except Exception as e:
        logger.error(f"Unified query error: {e}")
        raise HTTPException(status_code=500, detail=f"Unified query failed: {str(e)}")


@router.post("/scientific-search")
async def scientific_search(
    request: ScientificSearchRequest,
    scientific_service: ScientificSearchService = Depends(get_scientific_service)
):
    """
    Busca científica especializada com validação.
    
    Busca avançada com:
    - Resolução automática de DOI
    - Validação de fontes peer-reviewed
    - Análise de redes de citação
    - Filtros por impacto e recência
    """
    try:
        logger.info(f"Scientific search: {request.query[:100]}... (domains: {request.domains})")
        
        result = await scientific_service.scientific_search(request)
        return result
        
    except Exception as e:
        logger.error(f"Scientific search error: {e}")
        raise HTTPException(status_code=500, detail=f"Scientific search failed: {str(e)}")


@router.post("/biomaterials")
async def biomaterials_query(
    request: BiomaterialsQueryRequest,
    scientific_service: ScientificSearchService = Depends(get_scientific_service)
):
    """
    Query especializada para biomateriais.
    
    Busca otimizada para pesquisa em biomateriais com:
    - Filtros específicos de scaffold e material
    - Propriedades mecânicas e biocompatibilidade
    - Integração com bases de dados especializadas
    """
    try:
        logger.info(f"Biomaterials query: {request.query[:100]}...")
        
        result = await scientific_service.biomaterials_search(request)
        return result
        
    except Exception as e:
        logger.error(f"Biomaterials query error: {e}")
        raise HTTPException(status_code=500, detail=f"Biomaterials query failed: {str(e)}")


@router.post("/cross-domain")
async def cross_domain_query(
    request: CrossDomainRequest,
    scientific_service: ScientificSearchService = Depends(get_scientific_service)
):
    """
    Query interdisciplinar entre domínios.
    
    Busca que conecta múltiplos domínios científicos:
    - Análise de conceitos compartilhados
    - Raciocínio analógico entre campos
    - Identificação de oportunidades de colaboração
    """
    try:
        logger.info(f"Cross-domain query: {request.primary_query[:100]}... (domains: {request.secondary_domains})")
        
        result = await scientific_service.cross_domain_search(request)
        return result
        
    except Exception as e:
        logger.error(f"Cross-domain query error: {e}")
        raise HTTPException(status_code=500, detail=f"Cross-domain query failed: {str(e)}")


# =============================================================================
# DISCOVERY ENDPOINTS
# =============================================================================

@router.post("/discovery/run", response_model=DiscoveryResponse)
async def run_discovery(
    request: DiscoveryRequest,
    background_tasks: BackgroundTasks,
    scientific_service: ScientificSearchService = Depends(get_scientific_service)
):
    """
    Executa discovery científico automático.
    
    Monitora feeds RSS configurados, detecta novidades
    e adiciona descobertas relevantes à base de conhecimento.
    """
    try:
        logger.info(f"Running discovery - domains: {request.domains}, run_once: {request.run_once}")
        
        if request.run_once:
            # Execute immediately
            result = await scientific_service.run_discovery(request)
        else:
            # Run in background
            background_tasks.add_task(scientific_service.run_discovery, request)
            result = DiscoveryResponse(
                status="started",
                fetched=0,
                novel=0,
                added=0,
                errors=0,
                timestamp=datetime.now()
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Discovery error: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.post("/discovery/continuous")
async def start_continuous_discovery(
    domains: List[QueryDomain] = Query([], description="Domains to monitor"),
    interval_hours: int = Query(24, description="Check interval in hours"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    scientific_service: ScientificSearchService = Depends(get_scientific_service)
):
    """
    Inicia discovery científico contínuo.
    
    Monitora 24/7 fontes científicas com atualizações automáticas
    da base de conhecimento para descobertas novas.
    """
    try:
        request = DiscoveryRequest(
            run_once=False,
            domains=domains or list(QueryDomain),
            keywords=[]
        )
        
        # Schedule continuous discovery
        background_tasks.add_task(scientific_service.run_discovery, request)
        
        return {
            "status": "started",
            "message": "Continuous discovery monitoring started",
            "domains": [d.value for d in domains] if domains else "all",
            "interval_hours": interval_hours
        }
        
    except Exception as e:
        logger.error(f"Error starting continuous discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start discovery: {str(e)}")


# =============================================================================
# CONFIGURATION & INFO ENDPOINTS
# =============================================================================

@router.get("/config")
async def get_configuration(
    vertex_client: VertexAIClient = Depends(get_vertex_service)
):
    """
    Configuração do serviço RAG++.
    
    Retorna parâmetros de configuração atuais incluindo
    modelos, thresholds e configurações de discovery.
    """
    try:
        return {
            "project_id": vertex_client.config.project_id,
            "location": vertex_client.config.location,
            "embedding_model": vertex_client.config.embedding_model,
            "chat_model": vertex_client.config.chat_model,
            "max_output_tokens": vertex_client.config.max_output_tokens,
            "temperature": vertex_client.config.temperature,
            "supported_domains": [domain.value for domain in QueryDomain],
            "supported_methods": [method.value for method in SearchMethod],
            "supported_sources": [source.value for source in SourceType]
        }
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")


@router.get("/knowledge-graph")
async def get_knowledge_graph(
    domain: Optional[QueryDomain] = Query(None, description="Domain filter"),
    max_nodes: int = Query(50, description="Maximum nodes to return"),
    rag_engine: RAGEngine = Depends(get_rag_service)
):
    """
    Grafo de conhecimento da base RAG++.
    
    Retorna representação em grafo das conexões entre
    documentos, conceitos e domínios na base de conhecimento.
    """
    try:
        # Simplified knowledge graph based on document relationships
        # In a full implementation, this would build a proper graph structure
        
        # Get sample documents
        sample_query = domain.value if domain else "knowledge"
        results = await rag_engine.search_documents(sample_query, max_nodes)
        
        # Build simple node structure
        nodes = []
        edges = []
        
        for i, result in enumerate(results):
            nodes.append({
                "id": result.doc_id,
                "label": result.metadata.get("title", f"Document {i+1}")[:50],
                "type": "document",
                "domain": result.metadata.get("domain", "unknown"),
                "score": result.score
            })
        
        # Simple edges based on domain similarity
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if node1["domain"] == node2["domain"]:
                    edges.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "weight": 0.5,
                        "type": "domain_similarity"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges[:100],  # Limit edges
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "domain_filter": domain.value if domain else None,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph failed: {str(e)}")


@router.get("/sources")
async def list_discovery_sources():
    """
    Lista fontes de discovery configuradas.
    
    Retorna informação sobre feeds RSS e outras fontes
    monitoradas para descobertas científicas.
    """
    try:
        # Get configured sources from scientific service
        scientific_service = await get_scientific_search_service()
        
        sources = []
        for domain, feed_urls in scientific_service.feed_monitor.feeds_config.items():
            for url in feed_urls:
                sources.append({
                    "name": f"{domain.title()} Feed",
                    "type": "rss_feed",
                    "url": url,
                    "domain": domain,
                    "enabled": True,
                    "check_interval": 3600  # 1 hour
                })
        
        return {
            "sources": sources,
            "total_sources": len(sources),
            "enabled_sources": len([s for s in sources if s["enabled"]]),
            "domains_covered": list(set(s["domain"] for s in sources)),
            "discovery_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {str(e)}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception in RAG++ router: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )