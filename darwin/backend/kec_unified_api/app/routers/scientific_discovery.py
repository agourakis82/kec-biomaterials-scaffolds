"""
DARWIN SCIENTIFIC DISCOVERY - Router completo
Endpoints FastAPI para sistema Scientific Discovery automático
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse

from ..models.discovery_models import (
    # Request models
    DiscoveryRequest,
    DiscoveryStartRequest,
    DiscoveryStopRequest,
    InterdisciplinaryDiscoveryRequest,
    BiomaterialsDiscoveryConfig,
    NeuroscienceDiscoveryConfig,
    PhilosophyDiscoveryConfig,
    FeedConfigurationRequest,
    
    # Response models
    DiscoveryResponse,
    DiscoveryResult,
    DiscoveryStatusResponse,
    CrossDomainInsight,
    EmergingTrend,
    NoveltyAnalysisResult,
    PaperMetadata,
    
    # Enums and configs
    ScientificDomain,
    NoveltyLevel,
    NoveltyThreshold,
    FeedConfig,
    DiscoveryStatus
)

from ..services.discovery_engine import get_discovery_engine, DiscoveryEngine, DiscoveryConfig
from ..services.rss_monitor import get_rss_monitor
from ..services.novelty_detector import get_novelty_detector
from ..services.cross_domain_analyzer import get_cross_domain_analyzer

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/discovery", tags=["Scientific Discovery"])

# Dependency para obter discovery engine
def get_discovery_service() -> DiscoveryEngine:
    """Dependency para obter instância do Discovery Engine."""
    return get_discovery_engine()


# =============================================================================
# CORE DISCOVERY ENDPOINTS
# =============================================================================

@router.get("/health", summary="Discovery System Health Check")
async def discovery_health_check():
    """Health check específico do sistema Discovery."""
    try:
        engine = get_discovery_engine()
        rss_monitor = get_rss_monitor()
        novelty_detector = get_novelty_detector()
        cross_domain_analyzer = get_cross_domain_analyzer()
        
        return {
            "status": "healthy",
            "service": "scientific_discovery",
            "timestamp": datetime.now(timezone.utc),
            "components": {
                "discovery_engine": "operational",
                "rss_monitor": "operational",
                "novelty_detector": "operational", 
                "cross_domain_analyzer": "operational"
            },
            "stats": engine.get_status().stats
        }
    except Exception as e:
        logger.error(f"Discovery health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/run", response_model=DiscoveryResponse, summary="Execute Manual Discovery")
async def run_discovery(
    request: DiscoveryRequest,
    background_tasks: BackgroundTasks,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Executa descoberta científica manual.
    
    - **domains**: Lista de domínios científicos para analisar
    - **max_papers**: Número máximo de papers por execução
    - **run_once**: Se deve executar apenas uma vez
    - **filters**: Filtros opcionais para papers
    """
    try:
        logger.info(f"Manual discovery requested for domains: {[d.value for d in request.domains]}")
        
        # Executar discovery assíncronamente
        result = await engine.run_discovery_async(request)
        
        response = DiscoveryResponse(
            status="completed" if result.status == DiscoveryStatus.COMPLETED else "error",
            message=result.error_message or f"Discovery completed successfully",
            run_id=result.run_id,
            added=result.papers_discovered,
            discovered=result.papers_novel,
            processing_time=result.processing_time_seconds
        )
        
        logger.info(f"Manual discovery completed: {response.discovered} novel papers discovered")
        return response
        
    except Exception as e:
        logger.error(f"Manual discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery execution failed: {str(e)}")


@router.post("/start", response_model=DiscoveryResponse, summary="Start Continuous Discovery")
async def start_continuous_discovery(
    request: DiscoveryStartRequest,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Inicia discovery científico contínuo.
    
    - **interval_minutes**: Intervalo em minutos entre execuções
    - **sources**: Configurações de fontes de dados
    - **novelty_threshold**: Thresholds personalizados para novidade
    """
    try:
        success = engine.start_continuous_discovery(request.interval_minutes)
        
        if success:
            logger.info(f"Continuous discovery started with {request.interval_minutes}min interval")
            return DiscoveryResponse(
                status="started",
                message=f"Continuous discovery started with {request.interval_minutes} minute interval",
                next_run=datetime.now(timezone.utc) + timedelta(minutes=request.interval_minutes)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to start continuous discovery")
            
    except Exception as e:
        logger.error(f"Failed to start continuous discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start continuous discovery: {str(e)}")


@router.post("/stop", response_model=DiscoveryResponse, summary="Stop Continuous Discovery")
async def stop_continuous_discovery(
    request: DiscoveryStopRequest,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Para discovery científico contínuo.
    
    - **force**: Se deve forçar parada mesmo com execução em andamento
    """
    try:
        success = engine.stop_continuous_discovery()
        
        if success:
            logger.info("Continuous discovery stopped")
            return DiscoveryResponse(
                status="stopped",
                message="Continuous discovery stopped successfully"
            )
        else:
            return DiscoveryResponse(
                status="not_running",
                message="Continuous discovery was not running"
            )
            
    except Exception as e:
        logger.error(f"Failed to stop continuous discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop continuous discovery: {str(e)}")


@router.get("/status", response_model=DiscoveryStatusResponse, summary="Get Discovery System Status")
async def get_discovery_status(
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Retorna status completo do sistema Discovery.
    
    Inclui informações sobre:
    - Status atual do sistema
    - Número de feeds ativos
    - Estatísticas de execução
    - Última execução
    - Próxima execução programada
    """
    try:
        status = engine.get_status()
        logger.debug(f"Status requested - current status: {status.status}")
        return status
        
    except Exception as e:
        logger.error(f"Failed to get discovery status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/sources", summary="Get Configured Discovery Sources")
async def get_discovery_sources(
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Lista todas as fontes de dados configuradas.
    """
    try:
        feeds = engine.feed_configs
        sources_by_domain = {}
        
        for feed in feeds:
            domain = feed.domain.value
            if domain not in sources_by_domain:
                sources_by_domain[domain] = []
            
            sources_by_domain[domain].append({
                "name": feed.name,
                "url": feed.url,
                "status": feed.status.value,
                "priority": feed.priority,
                "max_entries": feed.max_entries,
                "rate_limit": feed.rate_limit_seconds
            })
        
        return {
            "total_sources": len(feeds),
            "sources_by_domain": sources_by_domain,
            "active_sources": len([f for f in feeds if f.status.value == "active"]),
            "last_updated": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Failed to get discovery sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")


# =============================================================================
# SPECIALIZED DISCOVERY ENDPOINTS
# =============================================================================

@router.post("/interdisciplinary", response_model=Dict[str, Any], summary="Execute Interdisciplinary Discovery")
async def run_interdisciplinary_discovery(
    request: InterdisciplinaryDiscoveryRequest,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Executa discovery focado em conexões interdisciplinares.
    
    - **primary_domain**: Domínio científico principal
    - **secondary_domains**: Domínios secundários para análise de conexões
    - **cross_domain_threshold**: Threshold mínimo para conexões interdisciplinares
    """
    try:
        # Criar request de discovery incluindo todos os domínios
        all_domains = [request.primary_domain] + request.secondary_domains
        
        discovery_request = DiscoveryRequest(
            domains=all_domains,
            max_papers=request.max_papers_per_domain * len(all_domains),
            run_once=True
        )
        
        result = await engine.run_discovery_async(discovery_request)
        
        # Filtrar apenas insights cross-domain relevantes
        relevant_insights = [
            insight for insight in result.cross_domain_insights
            if (insight.primary_domain == request.primary_domain or 
                insight.primary_domain in request.secondary_domains) and
            insight.confidence_score >= request.cross_domain_threshold
        ]
        
        return {
            "status": "completed",
            "primary_domain": request.primary_domain.value,
            "secondary_domains": [d.value for d in request.secondary_domains],
            "papers_analyzed": result.papers_discovered,
            "cross_domain_insights": [
                {
                    "insight_id": insight.insight_id,
                    "domains": [insight.primary_domain.value] + [d.value for d in insight.connected_domains],
                    "confidence_score": insight.confidence_score,
                    "description": insight.insight_description,
                    "potential_applications": insight.potential_applications,
                    "research_gaps": insight.research_gaps_identified
                } for insight in relevant_insights
            ],
            "insight_count": len(relevant_insights),
            "processing_time": result.processing_time_seconds,
            "run_id": result.run_id
        }
        
    except Exception as e:
        logger.error(f"Interdisciplinary discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interdisciplinary discovery failed: {str(e)}")


@router.post("/biomaterials-focus", response_model=Dict[str, Any], summary="Execute Biomaterials-Focused Discovery")
async def run_biomaterials_discovery(
    config: BiomaterialsDiscoveryConfig,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Executa discovery especializado em biomateriais.
    
    - **scaffold_types**: Tipos de scaffolds de interesse
    - **applications**: Aplicações específicas (tissue engineering, drug delivery, etc.)
    - **characterization_methods**: Métodos de caracterização relevantes
    """
    try:
        # Criar filtros específicos para biomateriais
        required_keywords = (config.scaffold_types + 
                           config.applications + 
                           config.characterization_methods)
        
        discovery_request = DiscoveryRequest(
            domains=[ScientificDomain.BIOMATERIALS],
            max_papers=100,
            run_once=True
        )
        
        result = await engine.run_discovery_async(discovery_request)
        
        # Filtrar papers relevantes baseado na configuração
        relevant_papers = []
        for paper in result.papers:
            paper_text = f"{paper.title} {paper.abstract}".lower()
            
            # Verificar se contém termos de interesse
            has_scaffold = any(scaffold.lower() in paper_text for scaffold in config.scaffold_types)
            has_application = any(app.lower() in paper_text for app in config.applications)
            has_method = any(method.lower() in paper_text for method in config.characterization_methods)
            
            if has_scaffold or has_application or has_method:
                relevant_papers.append(paper)
        
        return {
            "status": "completed",
            "focus": "biomaterials",
            "total_papers": result.papers_discovered,
            "relevant_papers": len(relevant_papers),
            "novel_papers": result.papers_novel,
            "scaffold_types_analyzed": config.scaffold_types,
            "applications_analyzed": config.applications,
            "methods_analyzed": config.characterization_methods,
            "top_discoveries": [
                {
                    "title": paper.title,
                    "journal": paper.journal,
                    "doi": paper.doi,
                    "abstract": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                } for paper in relevant_papers[:5]
            ],
            "processing_time": result.processing_time_seconds,
            "run_id": result.run_id
        }
        
    except Exception as e:
        logger.error(f"Biomaterials discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Biomaterials discovery failed: {str(e)}")


# =============================================================================
# INSIGHTS AND TRENDS ENDPOINTS
# =============================================================================

@router.get("/insights/recent", response_model=List[Dict[str, Any]], summary="Get Recent Cross-Domain Insights")
async def get_recent_insights(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of insights to return"),
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Retorna insights interdisciplinares recentes.
    
    - **limit**: Número máximo de insights para retornar
    """
    try:
        insights = engine.get_recent_insights(limit)
        
        return [
            {
                "insight_id": insight.insight_id,
                "primary_domain": insight.primary_domain.value,
                "connected_domains": [d.value for d in insight.connected_domains],
                "connection_strength": insight.connection_strength,
                "confidence_score": insight.confidence_score,
                "description": insight.insight_description,
                "potential_applications": insight.potential_applications,
                "research_gaps": insight.research_gaps_identified,
                "methodology_transfers": insight.methodology_transfers,
                "created_at": insight.created_at,
                "papers_count": len(insight.papers_involved)
            } for insight in insights
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recent insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/trends/emerging", response_model=List[Dict[str, Any]], summary="Get Emerging Scientific Trends")
async def get_emerging_trends(
    domain: Optional[ScientificDomain] = Query(default=None, description="Filter by scientific domain"),
    time_window_days: int = Query(default=90, ge=7, le=365, description="Time window in days for trend analysis"),
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum confidence score"),
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Detecta e retorna trends científicas emergentes.
    
    - **domain**: Domínio científico para filtrar (opcional)
    - **time_window_days**: Janela temporal em dias para análise
    - **min_confidence**: Score mínimo de confiança para trends
    """
    try:
        # Obter papers recentes do cache
        recent_papers = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        
        for paper in engine.paper_cache.values():
            if paper.publication_date and paper.publication_date > cutoff_date:
                if not domain or paper.domain == domain:
                    recent_papers.append(paper)
        
        if len(recent_papers) < 10:
            return []
        
        # Detectar trends emergentes
        novelty_detector = get_novelty_detector()
        trends = novelty_detector.detect_emerging_trends(recent_papers, time_window_days)
        
        # Filtrar por confidence mínima
        filtered_trends = [t for t in trends if t.confidence_score >= min_confidence]
        
        return [
            {
                "trend_id": trend.trend_id,
                "domain": trend.domain.value,
                "trend_name": trend.trend_name,
                "description": trend.description,
                "confidence_score": trend.confidence_score,
                "growth_rate": trend.growth_rate,
                "emergence_date": trend.emergence_date,
                "supporting_papers_count": len(trend.supporting_papers),
                "related_keywords": trend.related_keywords,
                "predicted_impact": trend.predicted_impact,
                "created_at": trend.created_at
            } for trend in filtered_trends
        ]
        
    except Exception as e:
        logger.error(f"Failed to get emerging trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/novelty/scores", response_model=List[Dict[str, Any]], summary="Get Novelty Scores for Recent Papers")
async def get_novelty_scores(
    domain: Optional[ScientificDomain] = Query(default=None, description="Filter by scientific domain"),
    min_score: float = Query(default=0.7, ge=0.0, le=1.0, description="Minimum novelty score"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of results"),
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Retorna scores de novidade para papers recentes.
    
    - **domain**: Filtrar por domínio científico
    - **min_score**: Score mínimo de novidade
    - **limit**: Número máximo de resultados
    """
    try:
        # Obter análises de novidade do cache
        novelty_results = []
        
        for paper_id, novelty_result in engine.novelty_cache.items():
            if novelty_result.overall_score >= min_score:
                paper = engine.paper_cache.get(paper_id)
                if paper and (not domain or paper.domain == domain):
                    novelty_results.append((paper, novelty_result))
        
        # Ordenar por score de novidade
        novelty_results.sort(key=lambda x: x[1].overall_score, reverse=True)
        novelty_results = novelty_results[:limit]
        
        return [
            {
                "paper_id": paper.doc_id,
                "title": paper.title,
                "domain": paper.domain.value if paper.domain else None,
                "journal": paper.journal,
                "publication_date": paper.publication_date,
                "novelty_level": result.novelty_level.value,
                "overall_score": result.overall_score,
                "semantic_score": result.semantic_score,
                "citation_score": result.citation_score,
                "keyword_score": result.keyword_score,
                "temporal_score": result.temporal_score,
                "justification": result.justification,
                "detected_innovations": result.detected_innovations,
                "methodology_novelty": result.methodology_novelty,
                "conceptual_novelty": result.conceptual_novelty
            } for paper, result in novelty_results
        ]
        
    except Exception as e:
        logger.error(f"Failed to get novelty scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get novelty scores: {str(e)}")


@router.get("/connections/cross-domain", response_model=List[Dict[str, Any]], summary="Get Cross-Domain Connections")
async def get_cross_domain_connections(
    domain1: Optional[ScientificDomain] = Query(default=None, description="First domain"),
    domain2: Optional[ScientificDomain] = Query(default=None, description="Second domain"),
    min_strength: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum connection strength"),
    limit: int = Query(default=15, ge=1, le=50, description="Maximum number of connections"),
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Analisa e retorna conexões entre domínios científicos.
    
    - **domain1**, **domain2**: Domínios específicos para analisar conexões
    - **min_strength**: Força mínima da conexão
    - **limit**: Número máximo de conexões
    """
    try:
        cross_domain_analyzer = get_cross_domain_analyzer()
        
        # Se domínios específicos foram fornecidos, analisar apenas essa conexão
        if domain1 and domain2:
            profile1 = cross_domain_analyzer.get_domain_profile(domain1)
            profile2 = cross_domain_analyzer.get_domain_profile(domain2)
            
            if not profile1 or not profile2:
                return []
            
            # Executar análise específica (simplified)
            insights = [insight for insight in engine.insight_cache.values()
                       if ((insight.primary_domain == domain1 and domain2 in insight.connected_domains) or
                           (insight.primary_domain == domain2 and domain1 in insight.connected_domains)) and
                       insight.connection_strength >= min_strength]
        else:
            # Obter todas as conexões
            insights = [insight for insight in engine.insight_cache.values()
                       if insight.connection_strength >= min_strength]
        
        # Ordenar por força de conexão
        insights.sort(key=lambda i: i.connection_strength, reverse=True)
        insights = insights[:limit]
        
        return [
            {
                "insight_id": insight.insight_id,
                "primary_domain": insight.primary_domain.value,
                "connected_domains": [d.value for d in insight.connected_domains],
                "connection_strength": insight.connection_strength,
                "confidence_score": insight.confidence_score,
                "description": insight.insight_description,
                "methodology_transfers": insight.methodology_transfers,
                "conceptual_bridges": insight.conceptual_bridges,
                "research_opportunities": insight.potential_applications,
                "papers_involved": len(insight.papers_involved),
                "created_at": insight.created_at
            } for insight in insights
        ]
        
    except Exception as e:
        logger.error(f"Failed to get cross-domain connections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get connections: {str(e)}")


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@router.post("/feeds/configure", response_model=Dict[str, Any], summary="Configure Custom RSS Feeds")
async def configure_feeds(
    request: FeedConfigurationRequest,
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Configura feeds RSS personalizados para um domínio.
    
    - **domain**: Domínio científico
    - **custom_feeds**: URLs de feeds RSS personalizados
    - **enable_default_feeds**: Se deve manter feeds padrão
    - **priority_adjustment**: Ajustes de prioridade para feeds específicos
    """
    try:
        new_configs = engine.configure_feeds(request.domain, request.custom_feeds)
        
        return {
            "status": "configured",
            "domain": request.domain.value,
            "custom_feeds_added": len(new_configs),
            "custom_feeds": [
                {
                    "name": config.name,
                    "url": config.url,
                    "priority": config.priority,
                    "max_entries": config.max_entries
                } for config in new_configs
            ],
            "total_feeds_for_domain": len([
                f for f in engine.feed_configs 
                if f.domain == request.domain
            ]),
            "configured_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Failed to configure feeds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure feeds: {str(e)}")


# =============================================================================
# ANALYTICS AND MONITORING ENDPOINTS
# =============================================================================

@router.get("/analytics/summary", summary="Get Discovery Analytics Summary")
async def get_analytics_summary(
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Retorna resumo analítico do sistema Discovery.
    """
    try:
        status = engine.get_status()
        high_impact = engine.get_high_impact_discoveries(10)
        
        # Análise por domínio
        papers_by_domain = {}
        for paper in engine.paper_cache.values():
            domain = paper.domain.value if paper.domain else "unknown"
            papers_by_domain[domain] = papers_by_domain.get(domain, 0) + 1
        
        # Top discoveries por score de novidade
        top_novelty = []
        for paper_id, novelty_result in list(engine.novelty_cache.items())[:10]:
            paper = engine.paper_cache.get(paper_id)
            if paper:
                top_novelty.append({
                    "title": paper.title[:60] + "..." if len(paper.title) > 60 else paper.title,
                    "score": novelty_result.overall_score,
                    "level": novelty_result.novelty_level.value,
                    "domain": paper.domain.value if paper.domain else "unknown"
                })
        
        top_novelty.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "system_status": status.status.value,
            "uptime_hours": round(status.uptime_seconds / 3600, 1),
            "total_papers_processed": status.stats.get("total_papers_processed", 0),
            "novel_discoveries": status.stats.get("total_novel_discoveries", 0),
            "cross_domain_insights": status.stats.get("total_cross_domain_insights", 0),
            "high_impact_alerts": len(high_impact),
            "papers_by_domain": papers_by_domain,
            "top_novelty_discoveries": top_novelty[:5],
            "recent_high_impact": high_impact[:3],
            "continuous_mode": status.stats.get("continuous_mode", False),
            "last_successful_run": status.last_run,
            "active_feeds": status.active_feeds,
            "error_count": status.error_count,
            "generated_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/history", response_model=List[Dict[str, Any]], summary="Get Discovery Execution History")
async def get_discovery_history(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of history entries"),
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Retorna histórico de execuções do Discovery.
    
    - **limit**: Número máximo de entradas do histórico
    """
    try:
        history = engine.get_discovery_history(limit)
        
        return [
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "domains_processed": [d.value for d in result.domains_processed],
                "papers_discovered": result.papers_discovered,
                "papers_novel": result.papers_novel,
                "insights_generated": result.insights_generated,
                "processing_time_seconds": result.processing_time_seconds,
                "created_at": result.created_at,
                "error_message": result.error_message,
                "feeds_processed_count": len(result.feeds_processed)
            } for result in history
        ]
        
    except Exception as e:
        logger.error(f"Failed to get discovery history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.delete("/cache/clear", summary="Clear Discovery System Cache")
async def clear_discovery_cache(
    engine: DiscoveryEngine = Depends(get_discovery_service)
):
    """
    Limpa todos os caches do sistema Discovery.
    
    **Atenção**: Esta operação remove todos os dados em cache.
    """
    try:
        engine.clear_caches()
        
        return {
            "status": "cleared",
            "message": "All discovery system caches have been cleared",
            "cleared_at": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")