"""
DARWIN SCIENTIFIC DISCOVERY - Discovery Engine
Engine principal que orquestra todo o sistema de descoberta científica automática
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..models.discovery_models import (
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryStatus,
    DiscoveryStatusResponse,
    FeedConfig,
    NoveltyThreshold,
    PaperMetadata,
    NoveltyAnalysisResult,
    CrossDomainInsight,
    ScientificDomain,
    EmergingTrend,
    FilterConfig,
    DiscoveryError
)

from .rss_monitor import get_rss_monitor, RSSMonitor
from .novelty_detector import get_novelty_detector, NoveltyDetector
from .cross_domain_analyzer import get_cross_domain_analyzer, CrossDomainAnalyzer
from .scheduler import get_scheduler, ScientificDiscoveryScheduler

try:
    from google.cloud import secretmanager
    SECRETMANAGER_AVAILABLE = True
except ImportError:
    secretmanager = None
    SECRETMANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DISCOVERY ENGINE STATE
# =============================================================================

@dataclass
class DiscoverySession:
    """Estado de uma sessão de discovery."""
    session_id: str
    start_time: datetime
    status: DiscoveryStatus
    domains: List[ScientificDomain]
    feeds_processed: List[str] = field(default_factory=list)
    papers_discovered: int = 0
    papers_novel: int = 0
    insights_generated: int = 0
    errors: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiscoveryConfig:
    """Configuração do sistema de discovery."""
    continuous_mode: bool = False
    interval_minutes: int = 120
    novelty_threshold: NoveltyThreshold = field(default_factory=NoveltyThreshold)
    max_papers_per_run: int = 200
    enable_cross_domain: bool = True
    enable_alerts: bool = True
    persist_to_rag: bool = True
    persist_to_knowledge_graph: bool = True
    alert_threshold_score: float = 0.8


# =============================================================================
# DISCOVERY ENGINE CLASS
# =============================================================================

class DiscoveryEngine:
    """
    Engine principal do sistema Scientific Discovery.
    Orquestra RSS Monitor, Novelty Detector e Cross Domain Analyzer.
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """
        Inicializa o Discovery Engine.
        
        Args:
            config: Configuração personalizada do engine
        """
        self.config = config or DiscoveryConfig()
        
        # Componentes principais
        self.rss_monitor: RSSMonitor = get_rss_monitor()
        self.novelty_detector: NoveltyDetector = get_novelty_detector(self.config.novelty_threshold)
        self.cross_domain_analyzer: CrossDomainAnalyzer = get_cross_domain_analyzer()
        
        # Estado do sistema
        self.status = DiscoveryStatus.IDLE
        self.current_session: Optional[DiscoverySession] = None
        self.feed_configs: List[FeedConfig] = []
        
        # Scheduler para discovery contínuo
        self.scheduler: ScientificDiscoveryScheduler = get_scheduler()
        
        # Caches e histórico
        self.discovery_history: List[DiscoveryResult] = []
        self.paper_cache: Dict[str, PaperMetadata] = {}
        self.novelty_cache: Dict[str, NoveltyAnalysisResult] = {}
        self.insight_cache: Dict[str, CrossDomainInsight] = {}
        
        # Alertas importantes
        self.high_impact_discoveries: List[Dict[str, Any]] = []
        
        # Estatísticas globais
        self.global_stats = {
            'total_runs': 0,
            'total_papers_processed': 0,
            'total_novel_discoveries': 0,
            'total_cross_domain_insights': 0,
            'total_emerging_trends': 0,
            'uptime_start': datetime.now(timezone.utc),
            'last_successful_run': None,
            'error_count': 0
        }
        
        # Integração com outros sistemas
        self.rag_engine = None
        self.knowledge_graph = None
        
        logger.info(f"Discovery Engine initialized - secret_manager: {SECRETMANAGER_AVAILABLE}")
    
    def _load_secret(self, secret_name: str) -> Optional[str]:
        """Carrega secret do GCP Secret Manager."""
        if not SECRETMANAGER_AVAILABLE:
            return None
        
        try:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                return None
            
            client = secretmanager.SecretManagerServiceClient()
            resource = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(name=resource)
            return response.payload.data.decode("utf-8")
            
        except Exception as e:
            logger.warning(f"Failed to load secret {secret_name}: {e}")
            return None
    
    def _initialize_feeds(self) -> List[FeedConfig]:
        """Inicializa configurações de feeds RSS."""
        # Tentar carregar de secret manager primeiro
        feeds_config = None
        if os.getenv("DISCOVERY_FROM_SECRET", "false").lower() == "true":
            feeds_yaml = self._load_secret("DISCOVERY_FEEDS_YML")
            if feeds_yaml:
                # Parse YAML config (simplified)
                logger.info("Loaded feeds configuration from secret manager")
        
        # Fallback para feeds padrão
        if not feeds_config:
            self.feed_configs = self.rss_monitor.load_default_feeds()
        
        logger.info(f"Initialized {len(self.feed_configs)} feed configurations")
        return self.feed_configs
    
    def _apply_filters(self, papers: List[PaperMetadata], filters: Optional[FilterConfig]) -> List[PaperMetadata]:
        """Aplica filtros aos papers descobertos."""
        if not filters:
            return papers
        
        filtered_papers = []
        
        for paper in papers:
            # Filtro de data
            if filters.min_publication_date and paper.publication_date:
                if paper.publication_date < filters.min_publication_date:
                    continue
            
            # Filtro de idade
            if paper.publication_date:
                age_days = (datetime.now(timezone.utc) - paper.publication_date).days
                if age_days > filters.max_age_days:
                    continue
            
            # Filtro de domínios excluídos
            if paper.domain in filters.exclude_domains:
                continue
            
            # Filtro de keywords obrigatórias
            if filters.required_keywords:
                paper_text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
                has_required = any(kw.lower() in paper_text for kw in filters.required_keywords)
                if not has_required:
                    continue
            
            # Filtro de keywords excluídas
            if filters.excluded_keywords:
                paper_text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
                has_excluded = any(kw.lower() in paper_text for kw in filters.excluded_keywords)
                if has_excluded:
                    continue
            
            # Filtro de impact factor
            if filters.min_impact_factor and paper.impact_factor:
                if paper.impact_factor < filters.min_impact_factor:
                    continue
            
            filtered_papers.append(paper)
        
        logger.info(f"Filtered {len(papers)} papers to {len(filtered_papers)}")
        return filtered_papers
    
    def _persist_to_rag(self, papers: List[PaperMetadata]) -> int:
        """Persiste papers no sistema RAG."""
        if not self.config.persist_to_rag or not papers:
            return 0
        
        # Integração com RAG engine (placeholder)
        try:
            # TODO: Integrar com RAG engine real
            # self.rag_engine.index_papers(papers)
            logger.info(f"Would persist {len(papers)} papers to RAG system")
            return len(papers)
            
        except Exception as e:
            logger.error(f"Failed to persist to RAG: {e}")
            return 0
    
    def _persist_to_knowledge_graph(self, papers: List[PaperMetadata], insights: List[CrossDomainInsight]) -> bool:
        """Persiste papers e insights no Knowledge Graph."""
        if not self.config.persist_to_knowledge_graph:
            return False
        
        try:
            # TODO: Integrar com Knowledge Graph real
            # self.knowledge_graph.add_papers(papers)
            # self.knowledge_graph.add_insights(insights)
            logger.info(f"Would persist {len(papers)} papers and {len(insights)} insights to Knowledge Graph")
            return True
            
        except Exception as e:
            logger.error(f"Failed to persist to Knowledge Graph: {e}")
            return False
    
    def _generate_alerts(
        self, 
        novel_papers: List[Tuple[PaperMetadata, NoveltyAnalysisResult]],
        insights: List[CrossDomainInsight],
        trends: List[EmergingTrend]
    ):
        """Gera alertas para descobertas importantes."""
        if not self.config.enable_alerts:
            return
        
        high_impact_items = []
        
        # Alertas para papers revolucionários
        revolutionary_papers = [
            (paper, result) for paper, result in novel_papers
            if result.overall_score >= self.config.alert_threshold_score
        ]
        
        for paper, result in revolutionary_papers:
            alert = {
                'type': 'revolutionary_paper',
                'paper_id': paper.doc_id,
                'title': paper.title,
                'domain': paper.domain.value if paper.domain else "unknown",
                'novelty_score': result.overall_score,
                'novelty_level': result.novelty_level.value,
                'justification': result.justification,
                'timestamp': datetime.now(timezone.utc)
            }
            high_impact_items.append(alert)
        
        # Alertas para insights cross-domain importantes
        high_confidence_insights = [
            insight for insight in insights
            if insight.confidence_score >= self.config.alert_threshold_score
        ]
        
        for insight in high_confidence_insights:
            alert = {
                'type': 'cross_domain_insight',
                'insight_id': insight.insight_id,
                'domains': [insight.primary_domain.value] + [d.value for d in insight.connected_domains],
                'confidence_score': insight.confidence_score,
                'description': insight.insight_description,
                'applications': insight.potential_applications,
                'timestamp': datetime.now(timezone.utc)
            }
            high_impact_items.append(alert)
        
        # Alertas para trends emergentes
        high_confidence_trends = [
            trend for trend in trends
            if trend.confidence_score >= 0.7  # Threshold para trends
        ]
        
        for trend in high_confidence_trends:
            alert = {
                'type': 'emerging_trend',
                'trend_id': trend.trend_id,
                'domain': trend.domain.value,
                'trend_name': trend.trend_name,
                'confidence_score': trend.confidence_score,
                'description': trend.description,
                'growth_rate': trend.growth_rate,
                'timestamp': datetime.now(timezone.utc)
            }
            high_impact_items.append(alert)
        
        # Armazenar alertas importantes
        self.high_impact_discoveries.extend(high_impact_items)
        
        # Log dos alertas
        if high_impact_items:
            logger.warning(f"🚨 {len(high_impact_items)} HIGH IMPACT DISCOVERIES detected!")
            for alert in high_impact_items[:3]:  # Log top 3
                logger.warning(f"  - {alert['type'].upper()}: {alert.get('title', alert.get('description', 'N/A'))}")
    
    async def run_discovery_async(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Executa descoberta científica assíncrona."""
        session_id = f"discovery_{int(time.time())}"
        start_time = datetime.now(timezone.utc)
        
        # Criar sessão
        session = DiscoverySession(
            session_id=session_id,
            start_time=start_time,
            status=DiscoveryStatus.RUNNING,
            domains=request.domains
        )
        self.current_session = session
        self.status = DiscoveryStatus.RUNNING
        
        try:
            logger.info(f"🔬 Starting scientific discovery session {session_id}")
            
            # 1. CARREGAR FEEDS CONFIGURADOS
            if not self.feed_configs:
                self.feed_configs = self._initialize_feeds()
            
            # Filtrar feeds por domínios solicitados
            relevant_feeds = [
                feed for feed in self.feed_configs
                if not request.domains or feed.domain in request.domains
            ]
            
            session.feeds_processed = [feed.name for feed in relevant_feeds]
            
            # 2. FETCH RSS FEEDS
            logger.info(f"📡 Fetching {len(relevant_feeds)} RSS feeds...")
            all_papers = await self.rss_monitor.sync_all_feeds_async(relevant_feeds)
            
            # Aplicar filtros
            if request.filters:
                all_papers = self._apply_filters(all_papers, request.filters)
            
            # Limitar número de papers
            if len(all_papers) > request.max_papers:
                all_papers = all_papers[:request.max_papers]
            
            session.papers_discovered = len(all_papers)
            logger.info(f"📄 Discovered {len(all_papers)} papers")
            
            # 3. ANÁLISE DE NOVIDADE
            logger.info("🔍 Analyzing novelty...")
            historical_context = list(self.paper_cache.values())
            novelty_results = await self.novelty_detector.batch_analyze_novelty(all_papers, historical_context)
            
            # Filtrar papers novos significativos
            novel_papers = []
            for paper in all_papers:
                result = next((r for r in novelty_results if r.paper_id == paper.doc_id), None)
                if result and result.overall_score >= self.config.novelty_threshold.semantic_similarity:
                    novel_papers.append((paper, result))
                    self.novelty_cache[paper.doc_id] = result
            
            session.papers_novel = len(novel_papers)
            logger.info(f"✨ Found {len(novel_papers)} novel papers")
            
            # 4. ANÁLISE CROSS-DOMAIN (se habilitada)
            insights = []
            if self.config.enable_cross_domain and len(request.domains) > 1:
                logger.info("🔗 Analyzing cross-domain connections...")
                
                # Organizar papers por domínio
                papers_by_domain = defaultdict(list)
                for paper in all_papers:
                    if paper.domain:
                        papers_by_domain[paper.domain].append(paper)
                
                # Analisar conexões
                self.cross_domain_analyzer.analyze_domain_papers(dict(papers_by_domain))
                insights = await self.cross_domain_analyzer.detect_cross_domain_insights_async()
                
                session.insights_generated = len(insights)
                logger.info(f"🧠 Generated {len(insights)} cross-domain insights")
            
            # 5. DETECÇÃO DE TRENDS EMERGENTES
            trends = []
            if len(all_papers) > 10:  # Mínimo para análise de trends
                trends = self.novelty_detector.detect_emerging_trends(all_papers)
                logger.info(f"📈 Detected {len(trends)} emerging trends")
            
            # 6. PERSISTÊNCIA
            persisted_count = 0
            if self.config.persist_to_rag:
                persisted_count = self._persist_to_rag(all_papers)
            
            if self.config.persist_to_knowledge_graph:
                self._persist_to_knowledge_graph(all_papers, insights)
            
            # 7. GERAR ALERTAS
            self._generate_alerts(novel_papers, insights, trends)
            
            # 8. CRIAR RESULTADO
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = DiscoveryResult(
                run_id=session_id,
                status=DiscoveryStatus.COMPLETED,
                domains_processed=request.domains,
                feeds_processed=session.feeds_processed,
                papers_discovered=session.papers_discovered,
                papers_novel=session.papers_novel,
                insights_generated=session.insights_generated,
                processing_time_seconds=processing_time,
                papers=all_papers,
                novelty_results=novelty_results,
                cross_domain_insights=insights
            )
            
            # Atualizar cache de papers
            for paper in all_papers:
                self.paper_cache[paper.doc_id] = paper
            
            # Armazenar insights
            for insight in insights:
                self.insight_cache[insight.insight_id] = insight
            
            # Atualizar estatísticas globais
            self.global_stats['total_runs'] += 1
            self.global_stats['total_papers_processed'] += len(all_papers)
            self.global_stats['total_novel_discoveries'] += len(novel_papers)
            self.global_stats['total_cross_domain_insights'] += len(insights)
            self.global_stats['total_emerging_trends'] += len(trends)
            self.global_stats['last_successful_run'] = datetime.now(timezone.utc)
            
            # Atualizar sessão e status
            session.status = DiscoveryStatus.COMPLETED
            self.current_session = None
            self.status = DiscoveryStatus.IDLE
            
            # Armazenar no histórico
            self.discovery_history.append(result)
            
            logger.info(f"✅ Discovery session completed in {processing_time:.1f}s - {len(novel_papers)} novel discoveries")
            return result
            
        except Exception as e:
            # Handle errors
            error_msg = f"Discovery session failed: {str(e)}"
            logger.error(error_msg)
            
            session.errors.append(error_msg)
            session.status = DiscoveryStatus.ERROR
            self.status = DiscoveryStatus.ERROR
            self.global_stats['error_count'] += 1
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            error_result = DiscoveryResult(
                run_id=session_id,
                status=DiscoveryStatus.ERROR,
                domains_processed=request.domains,
                feeds_processed=session.feeds_processed,
                papers_discovered=session.papers_discovered,
                papers_novel=session.papers_novel,
                insights_generated=session.insights_generated,
                processing_time_seconds=processing_time,
                error_message=error_msg,
                papers=[],
                novelty_results=[],
                cross_domain_insights=[]
            )
            
            return error_result
    
    def run_discovery(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Executa descoberta científica síncrona."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.run_discovery_async(request))
        finally:
            loop.close()
    
    async def start_continuous_discovery(self, interval_minutes: int = 120) -> bool:
        """Inicia discovery contínuo."""
        try:
            # Usar o scheduler dedicado
            success = await self.scheduler.schedule_continuous_discovery(
                interval_minutes=interval_minutes,
                domains=None,  # Todos os domínios
                max_papers=self.config.max_papers_per_run
            )
            
            if success:
                self.config.continuous_mode = True
                self.config.interval_minutes = interval_minutes
                logger.info(f"🔄 Continuous discovery started - interval: {interval_minutes} minutes")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start continuous discovery: {e}")
            return False
    
    async def stop_continuous_discovery(self) -> bool:
        """Para discovery contínuo."""
        try:
            # Remover job de discovery contínuo
            success = self.scheduler.remove_job("continuous_discovery")
            
            if success:
                self.config.continuous_mode = False
                logger.info("🛑 Continuous discovery stopped")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to stop continuous discovery: {e}")
            return False
    
    # Remove _scheduled_discovery method - now handled by scheduler service
    
    def get_status(self) -> DiscoveryStatusResponse:
        """Retorna status atual do sistema."""
        active_feeds = len([f for f in self.feed_configs if f.status.value == "active"])
        
        uptime = (datetime.now(timezone.utc) - self.global_stats['uptime_start']).total_seconds()
        
        # Get next run from scheduler
        next_run = None
        continuous_job = self.scheduler.get_job_info("continuous_discovery")
        if continuous_job:
            next_run = continuous_job.next_run
        
        return DiscoveryStatusResponse(
            status=self.status,
            current_run_id=self.current_session.session_id if self.current_session else None,
            active_feeds=active_feeds,
            total_papers=len(self.paper_cache),
            last_run=self.global_stats['last_successful_run'],
            next_run=next_run,
            uptime_seconds=uptime,
            error_count=self.global_stats['error_count'],
            stats={
                'total_runs': self.global_stats['total_runs'],
                'total_papers_processed': self.global_stats['total_papers_processed'],
                'total_novel_discoveries': self.global_stats['total_novel_discoveries'],
                'total_cross_domain_insights': self.global_stats['total_cross_domain_insights'],
                'total_emerging_trends': self.global_stats['total_emerging_trends'],
                'high_impact_alerts': len(self.high_impact_discoveries),
                'continuous_mode': self.config.continuous_mode,
                'interval_minutes': self.config.interval_minutes if self.config.continuous_mode else None,
                'scheduler_stats': self.scheduler.get_scheduler_stats()
            }
        )
    
    def get_recent_insights(self, limit: int = 10) -> List[CrossDomainInsight]:
        """Retorna insights recentes."""
        insights = list(self.insight_cache.values())
        insights.sort(key=lambda i: i.created_at, reverse=True)
        return insights[:limit]
    
    def get_high_impact_discoveries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retorna descobertas de alto impacto."""
        discoveries = sorted(
            self.high_impact_discoveries, 
            key=lambda d: d['timestamp'], 
            reverse=True
        )
        return discoveries[:limit]
    
    def configure_feeds(self, domain: ScientificDomain, custom_feeds: List[str]) -> List[FeedConfig]:
        """Configura feeds personalizados para um domínio."""
        new_configs = self.rss_monitor.add_custom_feeds(domain, custom_feeds)
        self.feed_configs.extend(new_configs)
        return new_configs
    
    def get_discovery_history(self, limit: int = 50) -> List[DiscoveryResult]:
        """Retorna histórico de execuções."""
        history = sorted(self.discovery_history, key=lambda r: r.created_at, reverse=True)
        return history[:limit]
    
    def clear_caches(self):
        """Limpa todos os caches."""
        self.paper_cache.clear()
        self.novelty_cache.clear()
        self.insight_cache.clear()
        self.high_impact_discoveries.clear()
        self.discovery_history.clear()
        
        # Limpar caches dos componentes
        self.rss_monitor.clear_processed_cache()
        self.novelty_detector.clear_cache()
        self.cross_domain_analyzer.clear_cache()
        
        logger.info("All discovery engine caches cleared")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Instância global do Discovery Engine
_discovery_engine = None

def get_discovery_engine(config: Optional[DiscoveryConfig] = None) -> DiscoveryEngine:
    """Retorna instância singleton do Discovery Engine."""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = DiscoveryEngine(config)
    return _discovery_engine