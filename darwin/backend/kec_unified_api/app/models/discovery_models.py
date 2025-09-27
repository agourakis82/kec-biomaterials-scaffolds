"""
DARWIN SCIENTIFIC DISCOVERY - Models consolidados
Modelos Pydantic para sistema Scientific Discovery automático
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class DiscoveryStatus(str, Enum):
    """Status do sistema Discovery."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class ScientificDomain(str, Enum):
    """Domínios científicos suportados."""
    BIOMATERIALS = "biomaterials"
    NEUROSCIENCE = "neuroscience"
    PHILOSOPHY = "philosophy"
    QUANTUM_MECHANICS = "quantum"
    MATHEMATICS = "mathematics"
    PSYCHOLOGY = "psychology"
    COMPUTER_SCIENCE = "computer_science"


class NoveltyLevel(str, Enum):
    """Níveis de novidade científica."""
    LOW = "low"           # Incremental
    MEDIUM = "medium"     # Significativa
    HIGH = "high"         # Breakthrough
    REVOLUTIONARY = "revolutionary"  # Paradigm shift


class FeedStatus(str, Enum):
    """Status de feeds RSS."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


# =============================================================================
# BASE MODELS
# =============================================================================

class TimestampedModel(BaseModel):
    """Base model com timestamps."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class FeedConfig(TimestampedModel):
    """Configuração de feed RSS."""
    name: str = Field(..., description="Nome identificativo do feed")
    url: str = Field(..., description="URL do feed RSS/Atom")
    domain: ScientificDomain = Field(..., description="Domínio científico")
    max_entries: int = Field(default=15, ge=1, le=100, description="Máximo de entradas por sync")
    priority: int = Field(default=1, ge=1, le=10, description="Prioridade do feed (1=baixa, 10=alta)")
    status: FeedStatus = Field(default=FeedStatus.ACTIVE)
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    rate_limit_seconds: int = Field(default=60, ge=10, description="Rate limit entre requests")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class SourceConfig(BaseModel):
    """Configuração de fonte de dados."""
    type: str = Field(..., description="Tipo da fonte (rss, arxiv, pubmed)")
    enabled: bool = Field(default=True)
    config: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)


class NoveltyThreshold(BaseModel):
    """Thresholds para detecção de novidade."""
    semantic_similarity: float = Field(default=0.85, ge=0.0, le=1.0)
    citation_uniqueness: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_emergence: float = Field(default=0.6, ge=0.0, le=1.0)
    cross_domain_score: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_significance: float = Field(default=0.75, ge=0.0, le=1.0)


class FilterConfig(BaseModel):
    """Configuração de filtros de descoberta."""
    min_publication_date: Optional[datetime] = None
    max_age_days: int = Field(default=30, ge=1)
    exclude_domains: List[ScientificDomain] = Field(default_factory=list)
    required_keywords: List[str] = Field(default_factory=list)
    excluded_keywords: List[str] = Field(default_factory=list)
    min_impact_factor: Optional[float] = None


# =============================================================================
# DISCOVERY REQUEST/RESPONSE MODELS
# =============================================================================

class DiscoveryRequest(BaseModel):
    """Request básico para discovery."""
    domains: List[ScientificDomain] = Field(default_factory=lambda: [ScientificDomain.BIOMATERIALS])
    max_papers: int = Field(default=50, ge=1, le=1000)
    run_once: bool = Field(default=False)
    filters: Optional[FilterConfig] = None


class DiscoveryStartRequest(BaseModel):
    """Request para iniciar discovery contínuo."""
    sources: List[SourceConfig] = Field(default_factory=list)
    interval_minutes: int = Field(default=120, ge=15, le=1440)
    novelty_threshold: Optional[NoveltyThreshold] = None


class DiscoveryStopRequest(BaseModel):
    """Request para parar discovery."""
    force: bool = Field(default=False)


class InterdisciplinaryDiscoveryRequest(BaseModel):
    """Request para discovery interdisciplinar."""
    primary_domain: ScientificDomain
    secondary_domains: List[ScientificDomain]
    max_papers_per_domain: int = Field(default=20, ge=1, le=100)
    cross_domain_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    novelty_threshold: Optional[NoveltyThreshold] = None


class BiomaterialsDiscoveryConfig(BaseModel):
    """Configuração específica para discovery de biomateriais."""
    scaffold_types: List[str] = Field(default_factory=lambda: ["hydrogel", "nanofiber", "ceramic", "composite"])
    applications: List[str] = Field(default_factory=lambda: ["tissue_engineering", "drug_delivery", "implants"])
    characterization_methods: List[str] = Field(default_factory=lambda: ["SEM", "mechanical_testing", "cytotoxicity"])
    include_in_vitro: bool = Field(default=True)
    include_in_vivo: bool = Field(default=True)
    min_sample_size: Optional[int] = None


class NeuroscienceDiscoveryConfig(BaseModel):
    """Configuração específica para discovery de neurociência."""
    brain_regions: List[str] = Field(default_factory=lambda: ["cortex", "hippocampus", "cerebellum"])
    techniques: List[str] = Field(default_factory=lambda: ["fMRI", "EEG", "optogenetics", "electrophysiology"])
    model_organisms: List[str] = Field(default_factory=lambda: ["human", "mouse", "rat", "zebrafish"])
    include_computational: bool = Field(default=True)
    include_clinical: bool = Field(default=True)


class PhilosophyDiscoveryConfig(BaseModel):
    """Configuração específica para discovery de filosofia."""
    areas: List[str] = Field(default_factory=lambda: ["philosophy_of_mind", "epistemology", "ethics", "metaphysics"])
    include_applied_ethics: bool = Field(default=True)
    include_philosophy_of_science: bool = Field(default=True)
    min_citation_count: Optional[int] = None


# =============================================================================
# PAPER/ARTICLE MODELS
# =============================================================================

class PaperMetadata(TimestampedModel):
    """Metadados de paper científico."""
    doc_id: str = Field(..., description="ID único do documento")
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    domain: Optional[ScientificDomain] = None
    source_feed: Optional[str] = None
    impact_factor: Optional[float] = None
    citation_count: Optional[int] = None


class NoveltyAnalysisResult(BaseModel):
    """Resultado da análise de novidade."""
    paper_id: str
    novelty_level: NoveltyLevel
    semantic_score: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    keyword_score: float = Field(ge=0.0, le=1.0)
    temporal_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    justification: str
    detected_innovations: List[str] = Field(default_factory=list)
    methodology_novelty: Optional[float] = None
    conceptual_novelty: Optional[float] = None


class CrossDomainInsight(TimestampedModel):
    """Insight interdisciplinar detectado."""
    insight_id: str = Field(..., description="ID único do insight")
    primary_domain: ScientificDomain
    connected_domains: List[ScientificDomain]
    connection_strength: float = Field(ge=0.0, le=1.0)
    papers_involved: List[str] = Field(description="IDs dos papers relacionados")
    insight_description: str
    potential_applications: List[str] = Field(default_factory=list)
    research_gaps_identified: List[str] = Field(default_factory=list)
    methodology_transfers: List[str] = Field(default_factory=list)
    conceptual_bridges: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)


# =============================================================================
# DISCOVERY RESULTS
# =============================================================================

class DiscoveryResult(TimestampedModel):
    """Resultado de uma execução de discovery."""
    run_id: str = Field(..., description="ID único da execução")
    status: DiscoveryStatus
    domains_processed: List[ScientificDomain]
    feeds_processed: List[str]
    papers_discovered: int = Field(ge=0)
    papers_novel: int = Field(ge=0)
    insights_generated: int = Field(ge=0)
    processing_time_seconds: float = Field(ge=0.0)
    error_message: Optional[str] = None
    papers: List[PaperMetadata] = Field(default_factory=list)
    novelty_results: List[NoveltyAnalysisResult] = Field(default_factory=list)
    cross_domain_insights: List[CrossDomainInsight] = Field(default_factory=list)


class DiscoveryResponse(BaseModel):
    """Response básico de discovery."""
    status: str
    message: str
    run_id: Optional[str] = None
    added: int = Field(default=0, ge=0)
    discovered: int = Field(default=0, ge=0)
    processing_time: Optional[float] = None
    next_run: Optional[datetime] = None


class DiscoveryStatusResponse(BaseModel):
    """Response para status do discovery."""
    status: DiscoveryStatus
    current_run_id: Optional[str] = None
    active_feeds: int = Field(ge=0)
    total_papers: int = Field(ge=0)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    uptime_seconds: float = Field(ge=0.0)
    error_count: int = Field(default=0, ge=0)
    stats: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# INSIGHTS AND TRENDS MODELS
# =============================================================================

class EmergingTrend(TimestampedModel):
    """Trend emergente detectado."""
    trend_id: str
    domain: ScientificDomain
    trend_name: str
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_papers: List[str] = Field(default_factory=list)
    emergence_date: datetime
    growth_rate: Optional[float] = None
    predicted_impact: Optional[str] = None
    related_keywords: List[str] = Field(default_factory=list)
    methodological_implications: List[str] = Field(default_factory=list)


class ResearchGap(BaseModel):
    """Gap de pesquisa identificado."""
    gap_id: str
    domain: ScientificDomain
    gap_description: str
    potential_impact: str
    suggested_methodologies: List[str] = Field(default_factory=list)
    related_papers: List[str] = Field(default_factory=list)
    funding_opportunities: List[str] = Field(default_factory=list)


class FeedConfigurationRequest(BaseModel):
    """Request para configurar feeds RSS."""
    domain: ScientificDomain
    custom_feeds: List[str] = Field(default_factory=list)
    enable_default_feeds: bool = Field(default=True)
    priority_adjustment: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# VALIDATION AND ERROR MODELS
# =============================================================================

class ValidationError(BaseModel):
    """Erro de validação."""
    field: str
    message: str
    value: Any


class DiscoveryError(BaseModel):
    """Erro específico do discovery."""
    error_code: str
    error_message: str
    component: str  # rss_monitor, novelty_detector, etc.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    # Enums
    "DiscoveryStatus", "ScientificDomain", "NoveltyLevel", "FeedStatus",
    
    # Base models
    "TimestampedModel",
    
    # Configuration
    "FeedConfig", "SourceConfig", "NoveltyThreshold", "FilterConfig",
    
    # Request models
    "DiscoveryRequest", "DiscoveryStartRequest", "DiscoveryStopRequest",
    "InterdisciplinaryDiscoveryRequest", "BiomaterialsDiscoveryConfig",
    "NeuroscienceDiscoveryConfig", "PhilosophyDiscoveryConfig",
    "FeedConfigurationRequest",
    
    # Paper/Analysis models
    "PaperMetadata", "NoveltyAnalysisResult", "CrossDomainInsight",
    
    # Results
    "DiscoveryResult", "DiscoveryResponse", "DiscoveryStatusResponse",
    
    # Insights
    "EmergingTrend", "ResearchGap",
    
    # Error models
    "ValidationError", "DiscoveryError",
]