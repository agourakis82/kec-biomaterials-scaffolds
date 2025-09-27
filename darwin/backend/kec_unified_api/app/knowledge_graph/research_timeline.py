"""DARWIN Research Timeline - Sistema de Timeline de Insights e Descobertas

Sistema épico que rastreia, organiza e analisa a evolução temporal de todos os insights,
descobertas e marcos de pesquisa interdisciplinar do DARWIN através dos domínios
biomaterials, neuroscience, philosophy, quantum mechanics e psychiatry.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, ResearchInsight, ResearchTimeline,
    KnowledgeGraphTypes
)

logger = get_logger("knowledge_graph.research_timeline")


# ==================== TIMELINE CONFIGURATION ====================

class InsightType(str, Enum):
    """Tipos de insights rastreados na timeline."""
    DISCOVERY = "discovery"              # Nova descoberta científica
    CONNECTION = "connection"            # Conexão interdisciplinar identificada
    BREAKTHROUGH = "breakthrough"        # Breakthrough conceitual importante
    VALIDATION = "validation"           # Validação de hipótese/teoria
    CONTRADICTION = "contradiction"      # Contradição ou refutação
    SYNTHESIS = "synthesis"             # Síntese de múltiplas ideias
    APPLICATION = "application"         # Nova aplicação prática
    METHODOLOGY = "methodology"         # Novo método ou abordagem
    HYPOTHESIS = "hypothesis"           # Nova hipótese gerada
    TREND = "trend"                     # Tendência identificada


class InsightSource(str, Enum):
    """Fontes de insights no DARWIN."""
    RAG_PLUS = "rag_plus"                    # RAG++ Enhanced Research
    MULTI_AI = "multi_ai_hub"                # Multi-AI Hub conversations
    SCIENTIFIC_DISCOVERY = "scientific_discovery"  # Scientific Discovery RSS
    SCORE_CONTRACTS = "score_contracts"      # Score Contracts mathematical analysis
    KEC_METRICS = "kec_metrics"             # KEC Metrics calculations
    TREE_SEARCH = "tree_search_puct"        # Tree Search PUCT optimization
    KNOWLEDGE_GRAPH = "knowledge_graph"      # Knowledge Graph analysis itself
    USER_INTERACTION = "user_interaction"    # User queries and interactions


@dataclass
class TimelineConfiguration:
    """Configuração do sistema de timeline."""
    # Clustering temporal
    enable_temporal_clustering: bool = True
    clustering_time_window_days: int = 30
    min_cluster_size: int = 3
    max_clusters: int = 20
    
    # Análise de tendências
    enable_trend_analysis: bool = True
    trend_detection_window_days: int = 90
    trend_significance_threshold: float = 0.7
    
    # Milestone detection
    enable_milestone_detection: bool = True
    milestone_confidence_threshold: float = 0.8
    milestone_impact_threshold: float = 0.6
    
    # Cross-domain tracking
    track_cross_domain_evolution: bool = True
    cross_domain_connection_bonus: float = 0.2
    
    # Insight quality filtering
    min_insight_confidence: float = 0.3
    enable_quality_filtering: bool = True
    
    # Timeline granularity
    default_time_granularity: str = "daily"  # daily, weekly, monthly
    max_insights_per_timeline: int = 10000


# Keywords para diferentes tipos de insights
INSIGHT_TYPE_KEYWORDS = {
    InsightType.DISCOVERY: {
        "discovered", "found", "identified", "revealed", "uncovered",
        "detected", "observed", "established", "demonstrated"
    },
    InsightType.CONNECTION: {
        "connected", "linked", "related", "associated", "correlated",
        "bridges", "connects", "relates", "joins", "unifies"
    },
    InsightType.BREAKTHROUGH: {
        "breakthrough", "revolutionary", "paradigm", "groundbreaking",
        "innovative", "novel", "unprecedented", "transformative"
    },
    InsightType.VALIDATION: {
        "validates", "confirms", "supports", "proves", "verifies",
        "substantiates", "corroborates", "affirms"
    },
    InsightType.CONTRADICTION: {
        "contradicts", "refutes", "challenges", "disputes", "opposes",
        "conflicts", "argues against", "questions"
    },
    InsightType.SYNTHESIS: {
        "synthesizes", "integrates", "combines", "merges", "unifies",
        "consolidates", "brings together", "harmonizes"
    },
    InsightType.APPLICATION: {
        "applies", "implements", "uses", "employs", "utilizes",
        "practical", "application", "implementation"
    },
    InsightType.METHODOLOGY: {
        "method", "approach", "technique", "procedure", "protocol",
        "methodology", "framework", "algorithm"
    },
    InsightType.HYPOTHESIS: {
        "hypothesis", "proposes", "suggests", "theorizes", "postulates",
        "hypothesizes", "speculates", "conjectures"
    },
    InsightType.TREND: {
        "trend", "pattern", "tendency", "direction", "movement",
        "evolution", "progression", "development"
    }
}

# Domínios interdisciplinares prioritários
INTERDISCIPLINARY_COMBINATIONS = [
    (ScientificDomains.BIOMATERIALS, ScientificDomains.NEUROSCIENCE),
    (ScientificDomains.NEUROSCIENCE, ScientificDomains.PHILOSOPHY),
    (ScientificDomains.PHILOSOPHY, ScientificDomains.QUANTUM_MECHANICS),
    (ScientificDomains.QUANTUM_MECHANICS, ScientificDomains.BIOMATERIALS),
    (ScientificDomains.PSYCHIATRY, ScientificDomains.NEUROSCIENCE),
    (ScientificDomains.MATHEMATICS, ScientificDomains.BIOMATERIALS),
    (ScientificDomains.MATHEMATICS, ScientificDomains.NEUROSCIENCE),
]


class DARWINResearchTimeline:
    """
    Sistema completo de timeline de pesquisa do DARWIN.
    
    Funcionalidades:
    - Tracking automático de insights de todas as fontes
    - Clustering temporal de insights relacionados
    - Detection de milestones de pesquisa importantes
    - Análise de tendências e padrões temporais
    - Cross-domain evolution tracking
    - Timeline visualizations e analytics
    - Milestone notifications e alertas
    - Research progress analytics
    """
    
    def __init__(self, config: Optional[TimelineConfiguration] = None):
        self.config = config or TimelineConfiguration()
        
        # Core timeline data
        self.insights: Dict[str, ResearchInsight] = {}
        self.timelines: Dict[str, ResearchTimeline] = {}
        
        # Temporal indexes
        self.insights_by_date: Dict[datetime.date, List[str]] = defaultdict(list)
        self.insights_by_domain: Dict[ScientificDomains, List[str]] = defaultdict(list)
        self.insights_by_source: Dict[InsightSource, List[str]] = defaultdict(list)
        self.insights_by_type: Dict[InsightType, List[str]] = defaultdict(list)
        
        # Analysis results
        self.temporal_clusters: List[Dict[str, Any]] = []
        self.detected_milestones: List[Dict[str, Any]] = []
        self.trend_analysis: Dict[str, Any] = {}
        self.cross_domain_evolution: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Analytics cache
        self.timeline_statistics: Dict[str, Any] = {}
        
        logger.info("⏰ DARWIN Research Timeline initialized")
    
    # ==================== CORE TIMELINE METHODS ====================
    
    async def add_insight(
        self,
        content: str,
        source: Union[InsightSource, str],
        domains: List[ScientificDomains],
        confidence: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Adiciona novo insight à timeline.
        """
        insight_id = f"insight_{uuid.uuid4().hex[:12]}"
        insight_timestamp = timestamp or datetime.utcnow()
        
        # Detectar tipo de insight automaticamente
        detected_type = await self._detect_insight_type(content)
        
        # Filtrar por qualidade se habilitado
        if self.config.enable_quality_filtering and confidence < self.config.min_insight_confidence:
            logger.debug(f"Insight filtered due to low confidence: {confidence}")
            return insight_id
        
        # Criar insight
        insight = ResearchInsight(
            id=insight_id,
            content=content,
            source=str(source),
            domains=domains,
            confidence=confidence,
            timestamp=insight_timestamp,
            metadata={
                **(metadata or {}),
                "detected_type": detected_type.value if detected_type else None,
                "interdisciplinary": len(domains) > 1,
                "cross_domain_score": await self._calculate_cross_domain_score(domains)
            }
        )
        
        # Armazenar insight
        self.insights[insight_id] = insight
        
        # Atualizar indexes
        await self._update_temporal_indexes(insight)
        
        # Trigger análises se necessário
        if self.config.enable_temporal_clustering:
            await self._trigger_incremental_analysis(insight)
        
        logger.debug(f"📝 Added insight {insight_id}: {content[:100]}...")
        return insight_id
    
    async def create_timeline(
        self,
        title: str,
        domain_filter: Optional[List[ScientificDomains]] = None,
        source_filter: Optional[List[InsightSource]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        type_filter: Optional[List[InsightType]] = None
    ) -> str:
        """
        Cria timeline baseada em filtros específicos.
        """
        timeline_id = f"timeline_{uuid.uuid4().hex[:8]}"
        
        # Filtrar insights baseado nos critérios
        filtered_insights = await self._filter_insights(
            domain_filter=domain_filter,
            source_filter=source_filter,
            start_date=start_date,
            end_date=end_date,
            type_filter=type_filter
        )
        
        # Determinar dates se não especificadas
        if filtered_insights:
            insight_dates = [insight.timestamp for insight in filtered_insights]
            actual_start_date = start_date or min(insight_dates)
            actual_end_date = end_date or max(insight_dates)
        else:
            actual_start_date = start_date or datetime.utcnow()
            actual_end_date = end_date or datetime.utcnow()
        
        # Criar timeline
        timeline = ResearchTimeline(
            id=timeline_id,
            title=title,
            insights=filtered_insights,
            start_date=actual_start_date,
            end_date=actual_end_date,
            domains_covered=set(domain for insight in filtered_insights for domain in insight.domains)
        )
        
        self.timelines[timeline_id] = timeline
        
        logger.info(f"📊 Created timeline '{title}' with {len(filtered_insights)} insights")
        return timeline_id
    
    async def analyze_timeline_patterns(
        self, 
        timeline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisa padrões temporais completos.
        """
        logger.info("🔍 Analyzing timeline patterns...")
        analysis_start = datetime.utcnow()
        
        try:
            # Usar timeline específica ou insights gerais
            if timeline_id and timeline_id in self.timelines:
                insights_to_analyze = self.timelines[timeline_id].insights
            else:
                insights_to_analyze = list(self.insights.values())
            
            if not insights_to_analyze:
                return {"error": "No insights to analyze"}
            
            analysis_results = {}
            
            # 1. Clustering temporal
            if self.config.enable_temporal_clustering:
                clustering_results = await self._perform_temporal_clustering(insights_to_analyze)
                analysis_results["temporal_clusters"] = clustering_results
                self.temporal_clusters = clustering_results.get("clusters", [])
            
            # 2. Milestone detection
            if self.config.enable_milestone_detection:
                milestone_results = await self._detect_research_milestones(insights_to_analyze)
                analysis_results["milestones"] = milestone_results
                self.detected_milestones = milestone_results.get("milestones", [])
            
            # 3. Trend analysis
            if self.config.enable_trend_analysis:
                trend_results = await self._analyze_research_trends(insights_to_analyze)
                analysis_results["trends"] = trend_results
                self.trend_analysis = trend_results
            
            # 4. Cross-domain evolution
            if self.config.track_cross_domain_evolution:
                evolution_results = await self._analyze_cross_domain_evolution(insights_to_analyze)
                analysis_results["cross_domain_evolution"] = evolution_results
                self.cross_domain_evolution = evolution_results
            
            # 5. Estatísticas gerais
            general_stats = await self._calculate_timeline_statistics(insights_to_analyze)
            analysis_results["statistics"] = general_stats
            self.timeline_statistics = general_stats
            
            # 6. Productivity analytics
            productivity_analysis = await self._analyze_research_productivity(insights_to_analyze)
            analysis_results["productivity"] = productivity_analysis
            
            analysis_time = (datetime.utcnow() - analysis_start).total_seconds()
            analysis_results["analysis_metadata"] = {
                "analysis_time_seconds": analysis_time,
                "insights_analyzed": len(insights_to_analyze),
                "timestamp": analysis_start.isoformat()
            }
            
            logger.info(f"✅ Timeline analysis completed in {analysis_time:.2f}s")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Timeline analysis failed: {e}")
            return {"error": str(e)}
    
    # ==================== TEMPORAL CLUSTERING ====================
    
    async def _perform_temporal_clustering(
        self, 
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """
        Agrupa insights temporalmente relacionados.
        """
        if len(insights) < self.config.min_cluster_size:
            return {"clusters": [], "total_clusters": 0}
        
        try:
            # Preparar dados temporais
            temporal_features = []
            insight_metadata = []
            
            for insight in insights:
                # Features temporais e de conteúdo
                timestamp_features = await self._extract_temporal_features(insight)
                content_features = await self._extract_content_features(insight)
                
                combined_features = timestamp_features + content_features
                temporal_features.append(combined_features)
                insight_metadata.append({
                    "insight_id": insight.id,
                    "timestamp": insight.timestamp,
                    "content": insight.content[:200],
                    "domains": [d.value for d in insight.domains]
                })
            
            # Normalizar features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(temporal_features)
            
            # Determinar número de clusters
            n_clusters = min(
                max(len(insights) // 10, 2),
                self.config.max_clusters
            )
            
            # Aplicar KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Organizar clusters
            clusters_data = defaultdict(list)
            for i, (insight, label) in enumerate(zip(insights, cluster_labels)):
                clusters_data[int(label)].append({
                    "insight_id": insight.id,
                    "insight": insight,
                    "cluster_distance": float(np.linalg.norm(
                        normalized_features[i] - kmeans.cluster_centers_[label]
                    ))
                })
            
            # Filtrar clusters pequenos
            valid_clusters = {
                cluster_id: cluster_insights
                for cluster_id, cluster_insights in clusters_data.items()
                if len(cluster_insights) >= self.config.min_cluster_size
            }
            
            # Analisar cada cluster
            analyzed_clusters = []
            for cluster_id, cluster_insights in valid_clusters.items():
                cluster_analysis = await self._analyze_temporal_cluster(cluster_id, cluster_insights)
                analyzed_clusters.append(cluster_analysis)
            
            # Ordenar clusters por relevância
            analyzed_clusters.sort(
                key=lambda x: x.get("significance_score", 0), 
                reverse=True
            )
            
            return {
                "clusters": analyzed_clusters,
                "total_clusters": len(analyzed_clusters),
                "clustering_method": "kmeans_temporal",
                "features_used": ["temporal", "content", "domain"],
                "cluster_quality_score": await self._evaluate_cluster_quality(analyzed_clusters)
            }
            
        except Exception as e:
            logger.error(f"Temporal clustering failed: {e}")
            return {"clusters": [], "error": str(e)}
    
    async def _extract_temporal_features(self, insight: ResearchInsight) -> List[float]:
        """Extrai features temporais de um insight."""
        timestamp = insight.timestamp
        
        # Features temporais básicas
        hour_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        day_of_month = timestamp.day / 31.0
        month_of_year = timestamp.month / 12.0
        
        # Features de recência (relativo ao insight mais recente)
        if self.insights:
            latest_timestamp = max(ins.timestamp for ins in self.insights.values())
            days_since_latest = (latest_timestamp - timestamp).days / 365.0  # Normalizar por ano
        else:
            days_since_latest = 0.0
        
        return [hour_of_day, day_of_week, day_of_month, month_of_year, days_since_latest]
    
    async def _extract_content_features(self, insight: ResearchInsight) -> List[float]:
        """Extrai features de conteúdo de um insight."""
        content = insight.content.lower()
        
        # Features básicas de conteúdo
        content_length = min(len(content) / 1000.0, 1.0)  # Normalizar
        word_count = min(len(content.split()) / 100.0, 1.0)
        
        # Features de domínio
        domain_count = len(insight.domains) / 6.0  # Max 6 domínios possíveis
        is_interdisciplinary = float(len(insight.domains) > 1)
        
        # Features de confiança e qualidade
        confidence = insight.confidence
        
        return [content_length, word_count, domain_count, is_interdisciplinary, confidence]
    
    async def _analyze_temporal_cluster(
        self, 
        cluster_id: int, 
        cluster_insights: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analisa características de um cluster temporal."""
        insights = [item["insight"] for item in cluster_insights]
        
        # Análise temporal
        timestamps = [insight.timestamp for insight in insights]
        time_span = (max(timestamps) - min(timestamps)).days
        
        # Análise de domínios
        all_domains = [domain for insight in insights for domain in insight.domains]
        domain_distribution = Counter(domain.value for domain in all_domains)
        
        # Análise de fontes
        source_distribution = Counter(insight.source for insight in insights)
        
        # Análise de conteúdo
        avg_confidence = sum(insight.confidence for insight in insights) / len(insights)
        total_content_length = sum(len(insight.content) for insight in insights)
        
        # Detectar tema principal
        main_theme = await self._detect_cluster_theme(insights)
        
        # Calcular significance score
        significance_score = await self._calculate_cluster_significance(insights, cluster_insights)
        
        return {
            "cluster_id": cluster_id,
            "size": len(insights),
            "time_span_days": time_span,
            "start_date": min(timestamps).isoformat(),
            "end_date": max(timestamps).isoformat(),
            "domain_distribution": dict(domain_distribution),
            "dominant_domain": domain_distribution.most_common(1)[0][0] if domain_distribution else "unknown",
            "source_distribution": dict(source_distribution),
            "primary_source": source_distribution.most_common(1)[0][0] if source_distribution else "unknown",
            "avg_confidence": avg_confidence,
            "total_content_length": total_content_length,
            "main_theme": main_theme,
            "significance_score": significance_score,
            "interdisciplinary_score": sum(1 for insight in insights if len(insight.domains) > 1) / len(insights),
            "insights": [
                {
                    "id": insight.id,
                    "content": insight.content[:150] + "..." if len(insight.content) > 150 else insight.content,
                    "timestamp": insight.timestamp.isoformat(),
                    "confidence": insight.confidence,
                    "domains": [d.value for d in insight.domains]
                }
                for insight in insights[:10]  # Limitar para performance
            ]
        }
    
    async def _detect_cluster_theme(self, insights: List[ResearchInsight]) -> str:
        """Detecta tema principal de um cluster."""
        # Análise simples baseada em palavras-chave
        all_content = " ".join(insight.content.lower() for insight in insights)
        words = all_content.split()
        
        # Filtrar palavras comuns
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those"
        }
        
        filtered_words = [word for word in words if len(word) > 3 and word not in stopwords]
        
        if not filtered_words:
            return "general_research"
        
        # Contar frequência e pegar palavras mais comuns
        word_freq = Counter(filtered_words)
        top_words = word_freq.most_common(3)
        
        return "_".join(word[0] for word in top_words)
    
    async def _calculate_cluster_significance(
        self, 
        insights: List[ResearchInsight], 
        cluster_data: List[Dict[str, Any]]
    ) -> float:
        """Calcula significance score do cluster."""
        # Fatores de significância
        size_factor = min(len(insights) / 20.0, 1.0)  # Clusters maiores são mais significantes
        confidence_factor = sum(insight.confidence for insight in insights) / len(insights)
        interdisciplinary_factor = sum(1 for insight in insights if len(insight.domains) > 1) / len(insights)
        
        # Coerência temporal (insights mais próximos temporalmente são mais significantes)
        timestamps = [insight.timestamp for insight in insights]
        time_variance = np.var([t.timestamp() for t in timestamps])
        temporal_coherence = 1.0 / (1.0 + time_variance / 1e10)  # Normalizar
        
        # Diversity de fontes (mais fontes = mais significante)
        sources = set(insight.source for insight in insights)
        source_diversity = len(sources) / 6.0  # Max 6 fontes possíveis
        
        # Combinar fatores
        significance = (
            size_factor * 0.3 +
            confidence_factor * 0.25 +
            interdisciplinary_factor * 0.2 +
            temporal_coherence * 0.15 +
            source_diversity * 0.1
        )
        
        return min(significance, 1.0)
    
    async def _evaluate_cluster_quality(self, clusters: List[Dict[str, Any]]) -> float:
        """Avalia qualidade geral do clustering."""
        if not clusters:
            return 0.0
        
        # Métricas de qualidade
        avg_significance = sum(cluster.get("significance_score", 0) for cluster in clusters) / len(clusters)
        size_distribution = [cluster.get("size", 0) for cluster in clusters]
        size_variance = np.var(size_distribution) if size_distribution else 0
        
        # Clusters com sizes similares são preferíveis
        size_balance = 1.0 / (1.0 + size_variance)
        
        # Score final
        quality_score = (avg_significance * 0.7 + size_balance * 0.3)
        return min(quality_score, 1.0)
    
    # ==================== MILESTONE DETECTION ====================
    
    async def _detect_research_milestones(
        self, 
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """Detecta milestones importantes na pesquisa."""
        try:
            milestone_candidates = []
            
            for insight in insights:
                # Calcular score de milestone
                milestone_score = await self._calculate_milestone_score(insight)
                
                if milestone_score >= self.config.milestone_confidence_threshold:
                    milestone_candidates.append({
                        "insight_id": insight.id,
                        "insight": insight,
                        "milestone_score": milestone_score,
                        "milestone_type": await self._classify_milestone_type(insight),
                        "impact_domains": [d.value for d in insight.domains],
                        "timestamp": insight.timestamp,
                        "significance_factors": await self._analyze_milestone_significance(insight)
                    })
            
            # Ordenar por score e filtrar
            milestone_candidates.sort(key=lambda x: x["milestone_score"], reverse=True)
            
            # Agrupar milestones próximos temporalmente
            grouped_milestones = await self._group_temporal_milestones(milestone_candidates)
            
            return {
                "milestones": grouped_milestones,
                "total_milestones": len(grouped_milestones),
                "milestone_density": len(grouped_milestones) / len(insights) if insights else 0,
                "detection_threshold": self.config.milestone_confidence_threshold,
                "top_milestone_domains": self._analyze_milestone_domains(grouped_milestones)
            }
            
        except Exception as e:
            logger.error(f"Milestone detection failed: {e}")
            return {"milestones": [], "error": str(e)}
    
    async def _calculate_milestone_score(self, insight: ResearchInsight) -> float:
        """Calcula score de milestone para um insight."""
        base_confidence = insight.confidence
        
        # Fator de interdisciplinaridade
        interdisciplinary_factor = len(insight.domains) * 0.1
        
        # Fator de conteúdo (palavras-chave importantes)
        content_factor = await self._analyze_milestone_content(insight.content)
        
        # Fator de impacto (baseado em metadata se disponível)
        impact_factor = insight.metadata.get("cross_domain_score", 0.0) * 0.2
        
        # Fator de novidade (insights únicos são mais prováveis de ser milestones)
        novelty_factor = await self._calculate_novelty_factor(insight)
        
        milestone_score = (
            base_confidence * 0.4 +
            content_factor * 0.25 +
            interdisciplinary_factor * 0.15 +
            impact_factor * 0.1 +
            novelty_factor * 0.1
        )
        
        return min(milestone_score, 1.0)
    
    async def _analyze_milestone_content(self, content: str) -> float:
        """Analisa conteúdo para palavras-chave de milestone."""
        content_lower = content.lower()
        milestone_keywords = {
            "breakthrough", "discovery", "revolutionary", "novel", "unprecedented",
            "paradigm", "fundamental", "groundbreaking", "innovative", "transformative",
            "significant", "major", "important", "critical", "key", "essential",
            "advances", "progress", "development", "evolution", "emergence"
        }
        
        keyword_matches = sum(1 for keyword in milestone_keywords if keyword in content_lower)
        content_score = min(keyword_matches * 0.1, 0.8)
        
        # Bonus para conteúdo mais longo e detalhado
        length_bonus = min(len(content) / 500.0 * 0.1, 0.2)
        
        return content_score + length_bonus
    
    async def _classify_milestone_type(self, insight: ResearchInsight) -> str:
        """Classifica tipo de milestone."""
        content_lower = insight.content.lower()
        
        # Mapeamento de palavras-chave para tipos
        type_keywords = {
            "theoretical_breakthrough": ["theory", "theoretical", "paradigm", "framework"],
            "empirical_discovery": ["found", "discovered", "observed", "measured"],
            "methodological_advance": ["method", "technique", "approach", "algorithm"],
            "interdisciplinary_connection": ["connects", "bridges", "links", "relates"],
            "practical_application": ["application", "practical", "implementation", "use"]
        }
        
        type_scores = {}
        for milestone_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                type_scores[milestone_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "general_milestone"
    
    async def _analyze_milestone_significance(self, insight: ResearchInsight) -> Dict[str, Any]:
        """Analisa fatores de significância do milestone."""
        return {
            "interdisciplinary": len(insight.domains) > 1,
            "high_confidence": insight.confidence >= 0.8,
            "cross_domain_score": insight.metadata.get("cross_domain_score", 0.0),
            "content_length": len(insight.content),
            "domains_involved": [d.value for d in insight.domains],
            "source_system": insight.source,
            "novelty_indicators": await self._detect_novelty_indicators(insight.content)
        }
    
    async def _detect_novelty_indicators(self, content: str) -> List[str]:
        """Detecta indicadores de novidade no conteúdo."""
        novelty_patterns = [
            r"\bnew\b", r"\bnovel\b", r"\bfirst\b", r"\bunprecedented\b",
            r"\bnever\s+before\b", r"\boriginal\b", r"\bunique\b",
            r"\binnovative\b", r"\bgroundbreaking\b"
        ]
        
        detected = []
        content_lower = content.lower()
        
        for pattern in novelty_patterns:
            if re.search(pattern, content_lower):
                detected.append(pattern.replace("\\b", "").replace("\\s+", " "))
        
        return detected
    
    async def _calculate_novelty_factor(self, insight: ResearchInsight) -> float:
        """Calcula fator de novidade do insight."""
        # Análise simples baseada em conteúdo único
        content_words = set(insight.content.lower().split())
        
        # Comparar com outros insights
        similarity_scores = []
        for other_insight in self.insights.values():
            if other_insight.id != insight.id:
                other_words = set(other_insight.content.lower().split())
                if content_words and other_words:
                    intersection = len(content_words & other_words)
                    union = len(content_words | other_words)
                    jaccard_similarity = intersection / union if union > 0 else 0
                    similarity_scores.append(jaccard_similarity)
        
        # Novidade é inversamente proporcional à similaridade
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            novelty_factor = 1.0 - avg_similarity
        else:
            novelty_factor = 1.0  # Primeiro insight é completamente novo
        
        return min(max(novelty_factor, 0.0), 1.0)
    
    async def _group_temporal_milestones(
        self, 
        milestone_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Agrupa milestones próximos temporalmente."""
        if not milestone_candidates:
            return []
        
        # Ordenar por timestamp
        sorted_milestones = sorted(milestone_candidates, key=lambda x: x["timestamp"])
        
        grouped = []
        current_group = [sorted_milestones[0]]
        
        for milestone in sorted_milestones[1:]:
            # Verificar se milestone atual está próximo do último no grupo
            time_diff = (milestone["timestamp"] - current_group[-1]["timestamp"]).days
            
            if time_diff <= 7:  # Agrupar milestones até 7 dias de distância
                current_group.append(milestone)
            else:
                # Finalizar grupo atual e começar novo
                if current_group:
                    grouped_milestone = await self._create_grouped_milestone(current_group)
                    grouped.append(grouped_milestone)
                current_group = [milestone]
        
        # Adicionar último grupo
        if current_group:
            grouped_milestone = await self._create_grouped_milestone(current_group)
            grouped.append(grouped_milestone)
        
        return grouped
    
    async def _create_grouped_milestone(
        self, 
        milestone_group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cria milestone agrupado."""
        # Selecionar milestone com maior score como principal
        main_milestone = max(milestone_group, key=lambda x: x["milestone_score"])
        
        # Calcular estatísticas do grupo
        group_score = sum(m["milestone_score"] for m in milestone_group) / len(milestone_group)
        all_domains = set()
        for m in milestone_group:
            all_domains.update(m["impact_domains"])
        
        return {
            "milestone_id": f"milestone_{uuid.uuid4().hex[:8]}",
            "main_insight_id": main_milestone["insight_id"],
            "group_size": len(milestone_group),
            "group_milestone_score": group_score,
            "main_milestone_score": main_milestone["milestone_score"],
            "milestone_type": main_milestone["milestone_type"],
            "timestamp": main_milestone["timestamp"],
            "title": f"Research Milestone: {main_milestone['insight'].content[:100]}...",
            "description": main_milestone["insight"].content,
            "impact_domains": list(all_domains),
            "significance_factors": main_milestone["significance_factors"],
            "related_insights": [
                {
                    "insight_id": m["insight_id"],
                    "score": m["milestone_score"],
                    "content": m["insight"].content[:100] + "..."
                }
                for m in milestone_group
            ]
        }
    
    def _analyze_milestone_domains(self, milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa distribuição de domínios nos milestones."""
        domain_counts = Counter()
        for milestone in milestones:
            for domain in milestone.get("impact_domains", []):
                domain_counts[domain] += 1
        
        return {
            "distribution": dict(domain_counts),
            "most_active_domain": domain_counts.most_common(1)[0] if domain_counts else None,
            "total_domains": len(domain_counts)
        }
    
    # ==================== TREND ANALYSIS ====================
    
    async def _analyze_research_trends(
        self, 
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """Analisa tendências de pesquisa."""
        try:
            # Agrupar insights por janelas temporais
            time_windows = await self._create_time_windows(insights)
            
            # Analisar tendências por domínio
            domain_trends = await self._analyze_domain_trends(time_windows)
            
            # Analisar tendências por fonte
            source_trends = await self._analyze_source_trends(time_windows)
            
            # Analisar tendências de tópicos
            topic_trends = await self._analyze_topic_trends(time_windows)
            
            # Detectar tendências emergentes
            emerging_trends = await self._detect_emerging_trends(time_windows)
            
            return {
                "analysis_period": {
                    "window_days": self.config.trend_detection_window_days,
                    "total_windows": len(time_windows),
                    "start_date": min(w["start_date"] for w in time_windows).isoformat() if time_windows else None,
                    "end_date": max(w["end_date"] for w in time_windows).isoformat() if time_windows else None
                },
                "domain_trends": domain_trends,
                "source_trends": source_trends,
                "topic_trends": topic_trends,
                "emerging_trends": emerging_trends,
                "trend_summary": await self._summarize_trends(domain_trends, source_trends, topic_trends)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    async def _create_time_windows(
        self, 
        insights: List[ResearchInsight]
    ) -> List[Dict[str, Any]]:
        """Cria janelas temporais para análise de tendências."""
        if not insights:
            return []
        
        # Ordenar insights por timestamp
        sorted_insights = sorted(insights, key=lambda x: x.timestamp)
        
        start_date = sorted_insights[0].timestamp
        end_date = sorted_insights[-1].timestamp
        window_delta = timedelta(days=self.config.trend_detection_window_days)
        
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + window_delta, end_date)
            
            # Filtrar insights nesta janela
            window_insights = [
                insight for insight in sorted_insights
                if current_start <= insight.timestamp < current_end
            ]
            
            if window_insights:
                windows.append({
                    "start_date": current_start,
                    "end_date": current_end,
                    "insights": window_insights,
                    "insight_count": len(window_insights)
                })
            
            current_start = current_end
        
        return windows
    
    async def _analyze_domain_trends(
        self, 
        time_windows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analisa tendências por domínio científico."""
        domain_timeline = {}
        
        for window in time_windows:
            window_date = window["start_date"]
            domain_counts = Counter()
            
            for insight in window["insights"]:
                for domain in insight.domains:
                    domain_counts[domain.value] += 1
            
            for domain in ScientificDomains:
                if domain.value not in domain_timeline:
                    domain_timeline[domain.value] = []
                
                domain_timeline[domain.value].append({
                    "date": window_date.isoformat(),
                    "count": domain_counts.get(domain.value, 0),
                    "percentage": (domain_counts.get(domain.value, 0) / window["insight_count"] * 100) if window["insight_count"] > 0 else 0
                })
        
        # Calcular tendências (slope simples)
        domain_trend_analysis = {}
        for domain, timeline in domain_timeline.items():
            if len(timeline) >= 2:
                counts = [point["count"] for point in timeline]
                trend_slope = await self._calculate_trend_slope(counts)
                
                domain_trend_analysis[domain] = {
                    "timeline": timeline,
                    "trend_slope": trend_slope,
                    "trend_direction": "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable",
                    "total_insights": sum(counts),
                    "avg_insights_per_window": sum(counts) / len(counts),
                    "peak_window": max(timeline, key=lambda x: x["count"])["date"] if timeline else None
                }
        
        return domain_trend_analysis
    
    async def _analyze_source_trends(
        self, 
        time_windows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analisa tendências por fonte de insights."""
        source_timeline = {}
        
        for window in time_windows:
            window_date = window["start_date"]
            source_counts = Counter(insight.source for insight in window["insights"])
            
            for source in InsightSource:
                if source.value not in source_timeline:
                    source_timeline[source.value] = []
                
                source_timeline[source.value].append({
                    "date": window_date.isoformat(),
                    "count": source_counts.get(source.value, 0)
                })
        
        # Analisar tendências das fontes
        source_trend_analysis = {}
        for source, timeline in source_timeline.items():
            if len(timeline) >= 2:
                counts = [point["count"] for point in timeline]
                trend_slope = await self._calculate_trend_slope(counts)
                
                source_trend_analysis[source] = {
                    "timeline": timeline,
                    "trend_slope": trend_slope,
                    "trend_direction": "increasing" if trend_slope > 0.1 else "decreasing" if trend_slope < -0.1 else "stable",
                    "total_insights": sum(counts),
                    "productivity_score": sum(counts) / len(timeline)
                }
        
        return source_trend_analysis
    
    async def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calcula slope de tendência usando regressão linear simples."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Calcular slope usando método de mínimos quadrados
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, values))
        sum_xx = sum(x_i * x_i for x_i in x)
        
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    async def _analyze_topic_trends(
        self, 
        time_windows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analisa tendências de tópicos emergentes."""
        # Implementação simplificada - extrair palavras-chave principais
        topic_timeline = defaultdict(lambda: defaultdict(int))
        
        for window in time_windows:
            window_key = window["start_date"].strftime("%Y-%m")
            
            # Extrair palavras-chave de todos os insights na janela
            all_content = " ".join(insight.content.lower() for insight in window["insights"])
            words = re.findall(r'\b[a-z]{4,}\b', all_content)  # Palavras com 4+ caracteres
            
            # Filtrar stopwords
            stopwords = {"this", "that", "with", "from", "they", "were", "been", "have", "their", "would", "there", "could", "other", "more", "very", "what", "know", "just", "time", "about", "after", "first", "well", "years"}
            filtered_words = [w for w in words if w not in stopwords]
            
            # Contar top palavras
            word_counts = Counter(filtered_words)
            for word, count in word_counts.most_common(10):
                topic_timeline[word][window_key] = count
        
        # Analisar tendências dos tópicos
        topic_trends = {}
        for topic, timeline in topic_timeline.items():
            if len(timeline) >= 2:
                sorted_timeline = sorted(timeline.items())
                counts = [count for date, count in sorted_timeline]
                trend_slope = await self._calculate_trend_slope(counts)
                
                if abs(trend_slope) > 0.1:  # Apenas tendências significativas
                    topic_trends[topic] = {
                        "timeline": [{"date": date, "count": count} for date, count in sorted_timeline],
                        "trend_slope": trend_slope,
                        "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
                        "total_mentions": sum(counts),
                        "peak_period": max(sorted_timeline, key=lambda x: x[1])[0]
                    }
        
        # Ordenar por relevância
        sorted_topics = sorted(
            topic_trends.items(),
            key=lambda x: abs(x[1]["trend_slope"]),
            reverse=True
        )
        
        return dict(sorted_topics[:20])  # Top 20 trending topics
    
    async def _detect_emerging_trends(
        self, 
        time_windows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detecta tendências emergentes."""
        emerging = []
        
        if len(time_windows) < 3:
            return emerging
        
        # Analisar últimas vs primeiras janelas
        recent_windows = time_windows[-2:]  # Últimas 2 janelas
        early_windows = time_windows[:2]    # Primeiras 2 janelas
        
        # Comparar domínios
        recent_domains = Counter()
        early_domains = Counter()
        
        for window in recent_windows:
            for insight in window["insights"]:
                for domain in insight.domains:
                    recent_domains[domain.value] += 1
        
        for window in early_windows:
            for insight in window["insights"]:
                for domain in insight.domains:
                    early_domains[domain.value] += 1
        
        # Identificar domínios emergentes
        for domain in recent_domains:
            recent_count = recent_domains[domain]
            early_count = early_domains.get(domain, 0)
            
            # Calcular growth ratio
            if early_count > 0:
                growth_ratio = recent_count / early_count
            else:
                growth_ratio = recent_count if recent_count > 0 else 0
            
            if growth_ratio > 2.0 and recent_count >= 3:  # Threshold para emergência
                emerging.append({
                    "type": "domain_emergence",
                    "domain": domain,
                    "growth_ratio": growth_ratio,
                    "recent_activity": recent_count,
                    "early_activity": early_count,
                    "significance": min(growth_ratio / 5.0, 1.0)
                })
        
        # Ordenar por significância
        emerging.sort(key=lambda x: x["significance"], reverse=True)
        
        return emerging[:10]  # Top 10 emerging trends
    
    async def _summarize_trends(
        self,
        domain_trends: Dict[str, Any],
        source_trends: Dict[str, Any],
        topic_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resume tendências principais."""
        summary = {
            "most_active_domain": None,
            "most_productive_source": None,
            "hottest_topics": [],
            "overall_research_velocity": 0.0,
            "trend_diversity": 0.0
        }
        
        # Domínio mais ativo
        if domain_trends:
            most_active = max(
                domain_trends.items(),
                key=lambda x: x[1].get("total_insights", 0)
            )
            summary["most_active_domain"] = {
                "domain": most_active[0],
                "total_insights": most_active[1]["total_insights"],
                "trend": most_active[1]["trend_direction"]
            }
        
        # Fonte mais produtiva
        if source_trends:
            most_productive = max(
                source_trends.items(),
                key=lambda x: x[1].get("productivity_score", 0)
            )
            summary["most_productive_source"] = {
                "source": most_productive[0],
                "productivity_score": most_productive[1]["productivity_score"],
                "trend": most_productive[1]["trend_direction"]
            }
        
        # Topics mais quentes
        if topic_trends:
            hot_topics = sorted(
                topic_trends.items(),
                key=lambda x: abs(x[1]["trend_slope"]),
                reverse=True
            )[:5]
            
            summary["hottest_topics"] = [
                {
                    "topic": topic,
                    "trend_slope": data["trend_slope"],
                    "direction": data["trend_direction"],
                    "mentions": data["total_mentions"]
                }
                for topic, data in hot_topics
            ]
        
        return summary
    
    # ==================== CROSS-DOMAIN EVOLUTION ====================
    
    async def _analyze_cross_domain_evolution(
        self, 
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """Analisa evolução das conexões cross-domain."""
        try:
            evolution_data = defaultdict(list)
            
            # Agrupar insights interdisciplinares por combinações de domínios
            for insight in insights:
                if len(insight.domains) > 1:
                    # Criar chaves para todas as combinações de domínios
                    domain_combinations = []
                    domains_sorted = sorted([d.value for d in insight.domains])
                    
                    for i, domain1 in enumerate(domains_sorted):
                        for domain2 in domains_sorted[i+1:]:
                            combination_key = f"{domain1}_{domain2}"
                            domain_combinations.append(combination_key)
                    
                    # Adicionar insight a todas as combinações relevantes
                    for combination in domain_combinations:
                        evolution_data[combination].append({
                            "insight_id": insight.id,
                            "timestamp": insight.timestamp,
                            "confidence": insight.confidence,
                            "content": insight.content,
                            "domains": [d.value for d in insight.domains],
                            "source": insight.source
                        })
            
            # Analisar evolução temporal para cada combinação
            evolution_analysis = {}
            for combination, insights_data in evolution_data.items():
                if len(insights_data) >= 2:  # Necessário pelo menos 2 insights para evolução
                    analysis = await self._analyze_combination_evolution(combination, insights_data)
                    evolution_analysis[combination] = analysis
            
            # Identificar as combinações mais ativas
            most_active_combinations = sorted(
                evolution_analysis.items(),
                key=lambda x: x[1]["total_insights"],
                reverse=True
            )[:10]
            
            return {
                "total_combinations": len(evolution_analysis),
                "most_active_combinations": dict(most_active_combinations),
                "evolution_summary": await self._summarize_cross_domain_evolution(evolution_analysis),
                "interdisciplinary_metrics": await self._calculate_interdisciplinary_metrics(insights)
            }
            
        except Exception as e:
            logger.error(f"Cross-domain evolution analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_combination_evolution(
        self,
        combination: str,
        insights_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analisa evolução de uma combinação específica de domínios."""
        # Ordenar insights por timestamp
        sorted_insights = sorted(insights_data, key=lambda x: x["timestamp"])
        
        # Análise temporal
        timestamps = [insight["timestamp"] for insight in sorted_insights]
        time_span = (max(timestamps) - min(timestamps)).days
        
        # Análise de qualidade/confiança
        confidences = [insight["confidence"] for insight in sorted_insights]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_trend = await self._calculate_trend_slope(confidences)
        
        # Análise de fontes
        sources = Counter(insight["source"] for insight in sorted_insights)
        
        # Densidade temporal (insights por unidade de tempo)
        if time_span > 0:
            insight_density = len(sorted_insights) / time_span
        else:
            insight_density = len(sorted_insights)
        
        # Detectar períodos de alta atividade
        activity_periods = await self._detect_activity_periods(sorted_insights)
        
        return {
            "combination": combination,
            "domains": combination.split("_"),
            "total_insights": len(sorted_insights),
            "time_span_days": time_span,
            "first_insight": timestamps[0].isoformat(),
            "latest_insight": timestamps[-1].isoformat(),
            "avg_confidence": avg_confidence,
            "confidence_trend": confidence_trend,
            "insight_density": insight_density,
            "source_distribution": dict(sources),
            "primary_source": sources.most_common(1)[0][0] if sources else "unknown",
            "activity_periods": activity_periods,
            "evolution_score": await self._calculate_evolution_score(
                len(sorted_insights), time_span, avg_confidence, insight_density
            ),
            "recent_insights": [
                {
                    "content": insight["content"][:150] + "...",
                    "timestamp": insight["timestamp"].isoformat(),
                    "confidence": insight["confidence"]
                }
                for insight in sorted_insights[-3:]  # Últimos 3 insights
            ]
        }
    
    async def _detect_activity_periods(
        self,
        sorted_insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detecta períodos de alta atividade."""
        if len(sorted_insights) < 3:
            return []
        
        # Agrupar insights por períodos de 30 dias
        periods = defaultdict(list)
        
        for insight in sorted_insights:
            # Agrupar por mês/ano
            period_key = insight["timestamp"].strftime("%Y-%m")
            periods[period_key].append(insight)
        
        # Identificar períodos com alta atividade (acima da média)
        period_sizes = [len(insights) for insights in periods.values()]
        avg_activity = sum(period_sizes) / len(period_sizes) if period_sizes else 0
        
        high_activity_periods = []
        for period_key, period_insights in periods.items():
            if len(period_insights) > avg_activity * 1.5:  # 50% acima da média
                high_activity_periods.append({
                    "period": period_key,
                    "insight_count": len(period_insights),
                    "activity_ratio": len(period_insights) / avg_activity if avg_activity > 0 else 0,
                    "avg_confidence": sum(i["confidence"] for i in period_insights) / len(period_insights)
                })
        
        # Ordenar por atividade
        high_activity_periods.sort(key=lambda x: x["activity_ratio"], reverse=True)
        
        return high_activity_periods[:5]  # Top 5 períodos
    
    async def _calculate_evolution_score(
        self,
        total_insights: int,
        time_span_days: int,
        avg_confidence: float,
        insight_density: float
    ) -> float:
        """Calcula score de evolução da combinação."""
        # Fatores de evolução
        volume_factor = min(total_insights / 20.0, 1.0)  # Normalizar por 20 insights
        quality_factor = avg_confidence
        consistency_factor = min(insight_density * 10, 1.0)  # Normalizar densidade
        longevity_factor = min(time_span_days / 365.0, 1.0)  # Normalizar por ano
        
        evolution_score = (
            volume_factor * 0.3 +
            quality_factor * 0.3 +
            consistency_factor * 0.25 +
            longevity_factor * 0.15
        )
        
        return min(evolution_score, 1.0)
    
    async def _summarize_cross_domain_evolution(
        self,
        evolution_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resume evolução cross-domain."""
        if not evolution_analysis:
            return {}
        
        # Estatísticas agregadas
        total_insights = sum(analysis["total_insights"] for analysis in evolution_analysis.values())
        avg_evolution_score = sum(analysis["evolution_score"] for analysis in evolution_analysis.values()) / len(evolution_analysis)
        
        # Combinação mais evoluída
        most_evolved = max(evolution_analysis.items(), key=lambda x: x[1]["evolution_score"])
        
        # Combinação mais ativa
        most_active = max(evolution_analysis.items(), key=lambda x: x[1]["total_insights"])
        
        # Tendências de qualidade
        confidence_trends = [analysis["confidence_trend"] for analysis in evolution_analysis.values()]
        overall_confidence_trend = sum(confidence_trends) / len(confidence_trends)
        
        return {
            "total_cross_domain_insights": total_insights,
            "avg_evolution_score": avg_evolution_score,
            "most_evolved_combination": {
                "combination": most_evolved[0],
                "score": most_evolved[1]["evolution_score"],
                "insights": most_evolved[1]["total_insights"]
            },
            "most_active_combination": {
                "combination": most_active[0],
                "insights": most_active[1]["total_insights"],
                "density": most_active[1]["insight_density"]
            },
            "overall_confidence_trend": overall_confidence_trend,
            "quality_direction": "improving" if overall_confidence_trend > 0.05 else "declining" if overall_confidence_trend < -0.05 else "stable"
        }
    
    async def _calculate_interdisciplinary_metrics(
        self,
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """Calcula métricas de interdisciplinaridade."""
        total_insights = len(insights)
        if total_insights == 0:
            return {}
        
        # Contar insights interdisciplinares
        interdisciplinary_insights = [i for i in insights if len(i.domains) > 1]
        interdisciplinary_ratio = len(interdisciplinary_insights) / total_insights
        
        # Diversidade de combinações
        domain_combinations = set()
        for insight in interdisciplinary_insights:
            domains_sorted = sorted([d.value for d in insight.domains])
            for i, domain1 in enumerate(domains_sorted):
                for domain2 in domains_sorted[i+1:]:
                    domain_combinations.add((domain1, domain2))
        
        # Cobertura de domínios
        all_domains = set()
        for insight in insights:
            all_domains.update(d.value for d in insight.domains)
        
        return {
            "interdisciplinary_ratio": interdisciplinary_ratio,
            "total_domain_combinations": len(domain_combinations),
            "domain_coverage": len(all_domains),
            "avg_domains_per_insight": sum(len(i.domains) for i in insights) / total_insights,
            "max_domains_in_single_insight": max(len(i.domains) for i in insights) if insights else 0,
            "unique_combinations": list(domain_combinations)
        }
    
    # ==================== PRODUCTIVITY ANALYTICS ====================
    
    async def _analyze_research_productivity(
        self,
        insights: List[ResearchInsight]
    ) -> Dict[str, Any]:
        """Analisa produtividade de pesquisa."""
        try:
            if not insights:
                return {}
            
            # Agrupar por períodos temporais
            daily_productivity = defaultdict(int)
            weekly_productivity = defaultdict(int)
            monthly_productivity = defaultdict(int)
            source_productivity = defaultdict(int)
            domain_productivity = defaultdict(int)
            
            for insight in insights:
                date = insight.timestamp.date()
                week = date.strftime("%Y-W%U")
                month = date.strftime("%Y-%m")
                
                daily_productivity[date.isoformat()] += 1
                weekly_productivity[week] += 1
                monthly_productivity[month] += 1
                source_productivity[insight.source] += 1
                
                for domain in insight.domains:
                    domain_productivity[domain.value] += 1
            
            # Calcular estatísticas
            daily_values = list(daily_productivity.values())
            weekly_values = list(weekly_productivity.values())
            monthly_values = list(monthly_productivity.values())
            
            productivity_stats = {
                "daily_productivity": {
                    "avg": sum(daily_values) / len(daily_values) if daily_values else 0,
                    "max": max(daily_values) if daily_values else 0,
                    "min": min(daily_values) if daily_values else 0,
                    "peak_day": max(daily_productivity.items(), key=lambda x: x[1])[0] if daily_productivity else None
                },
                "weekly_productivity": {
                    "avg": sum(weekly_values) / len(weekly_values) if weekly_values else 0,
                    "max": max(weekly_values) if weekly_values else 0,
                    "peak_week": max(weekly_productivity.items(), key=lambda x: x[1])[0] if weekly_productivity else None
                },
                "monthly_productivity": {
                    "avg": sum(monthly_values) / len(monthly_values) if monthly_values else 0,
                    "max": max(monthly_values) if monthly_values else 0,
                    "peak_month": max(monthly_productivity.items(), key=lambda x: x[1])[0] if monthly_productivity else None
                },
                "source_productivity": dict(source_productivity),
                "domain_productivity": dict(domain_productivity),
                "total_insights": len(insights),
                "analysis_period_days": (max(i.timestamp for i in insights) - min(i.timestamp for i in insights)).days if insights else 0
            }
            
            return productivity_stats
            
        except Exception as e:
            logger.error(f"Productivity analysis failed: {e}")
            return {"error": str(e)}
    
    # ==================== UTILITY METHODS ====================
    
    async def _filter_insights(
        self,
        domain_filter: Optional[List[ScientificDomains]] = None,
        source_filter: Optional[List[InsightSource]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        type_filter: Optional[List[InsightType]] = None
    ) -> List[ResearchInsight]:
        """Filtra insights baseado em critérios."""
        filtered = list(self.insights.values())
        
        if domain_filter:
            filtered = [
                insight for insight in filtered
                if any(domain in insight.domains for domain in domain_filter)
            ]
        
        if source_filter:
            source_strings = [s.value if isinstance(s, InsightSource) else s for s in source_filter]
            filtered = [
                insight for insight in filtered
                if insight.source in source_strings
            ]
        
        if start_date:
            filtered = [
                insight for insight in filtered
                if insight.timestamp >= start_date
            ]
        
        if end_date:
            filtered = [
                insight for insight in filtered
                if insight.timestamp <= end_date
            ]
        
        if type_filter:
            # Filtrar por tipo detectado (se disponível no metadata)
            filtered = [
                insight for insight in filtered
                if insight.metadata.get("detected_type") in [t.value for t in type_filter]
            ]
        
        return filtered
    
    async def _update_temporal_indexes(self, insight: ResearchInsight):
        """Atualiza indexes temporais."""
        insight_date = insight.timestamp.date()
        self.insights_by_date[insight_date].append(insight.id)
        
        for domain in insight.domains:
            self.insights_by_domain[domain].append(insight.id)
        
        # Converter source para enum se necessário
        if isinstance(insight.source, str):
            try:
                source_enum = InsightSource(insight.source)
                self.insights_by_source[source_enum].append(insight.id)
            except ValueError:
                logger.warning(f"Unknown insight source: {insight.source}")
        
        # Adicionar ao index de tipo se detectado
        detected_type = insight.metadata.get("detected_type")
        if detected_type:
            try:
                type_enum = InsightType(detected_type)
                self.insights_by_type[type_enum].append(insight.id)
            except ValueError:
                pass
    
    async def _trigger_incremental_analysis(self, insight: ResearchInsight):
        """Trigger análises incrementais quando apropriado."""
        # Trigger re-análise a cada 50 insights novos
        if len(self.insights) % 50 == 0:
            logger.info("Triggering incremental timeline analysis...")
            asyncio.create_task(self.analyze_timeline_patterns())
    
    async def _detect_insight_type(self, content: str) -> Optional[InsightType]:
        """Detecta tipo de insight baseado no conteúdo."""
        content_lower = content.lower()
        
        type_scores = {}
        for insight_type, keywords in INSIGHT_TYPE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                type_scores[insight_type] = score
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return InsightType.DISCOVERY  # Default type
    
    async def _calculate_cross_domain_score(self, domains: List[ScientificDomains]) -> float:
        """Calcula score de cross-domain."""
        if len(domains) <= 1:
            return 0.0
        
        base_score = min(len(domains) / 6.0, 1.0)  # Normalizar por 6 domínios max
        
        # Bonus para combinações interdisciplinares prioritárias
        domain_set = set(domains)
        priority_bonus = 0.0
        
        for combo in INTERDISCIPLINARY_COMBINATIONS:
            if set(combo).issubset(domain_set):
                priority_bonus += 0.1
        
        return min(base_score + priority_bonus, 1.0)
    
    def get_timeline_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da timeline."""
        return {
            "total_insights": len(self.insights),
            "total_timelines": len(self.timelines),
            "temporal_clusters": len(self.temporal_clusters),
            "detected_milestones": len(self.detected_milestones),
            "insights_by_domain": {
                domain.value: len(insights) 
                for domain, insights in self.insights_by_domain.items()
            },
            "insights_by_source": {
                source.value: len(insights) 
                for source, insights in self.insights_by_source.items()
            },
            "timeline_span": self._calculate_timeline_span(),
            "latest_statistics": self.timeline_statistics
        }
    
    def _calculate_timeline_span(self) -> Dict[str, Any]:
        """Calcula span temporal da timeline."""
        if not self.insights:
            return {}
        
        timestamps = [insight.timestamp for insight in self.insights.values()]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        return {
            "start_date": start_time.isoformat(),
            "end_date": end_time.isoformat(),
            "total_days": (end_time - start_time).days,
            "total_hours": (end_time - start_time).total_seconds() / 3600
        }


__all__ = [
    "DARWINResearchTimeline",
    "TimelineConfiguration",
    "InsightType",
    "InsightSource",
    "INSIGHT_TYPE_KEYWORDS",
    "INTERDISCIPLINARY_COMBINATIONS"
]