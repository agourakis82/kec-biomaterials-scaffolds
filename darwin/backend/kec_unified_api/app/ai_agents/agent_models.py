"""Agent Models - AutoGen Multi-Agent Research Team

🎯 MODELOS PARA SISTEMA REVOLUCIONÁRIO DE AGENTES IA COLABORATIVOS
Estruturas de dados avançadas para coordenar departamento de pesquisa IA completo.

Features:
- 🔬 CollaborativeResearchRequest/Response para pesquisa em equipe
- 🌐 CrossDomainRequest/Response para insights interdisciplinares  
- 🎭 AgentSpecialization para definir expertise de cada agent
- 💡 ResearchInsight para capturar descobertas colaborativas
- 📊 AgentStatus para monitoramento em tempo real
- ⚙️ TeamConfiguration para setup dinâmico da equipe
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

# ==================== ENUMS ====================

class AgentSpecialization(str, Enum):
    """Especializações dos agents de pesquisa."""
    BIOMATERIALS = "biomaterials"
    MATHEMATICS = "mathematics"
    PHILOSOPHY = "philosophy"
    LITERATURE = "literature"
    SYNTHESIS = "synthesis"
    NEUROSCIENCE = "neuroscience"
    QUANTUM_MECHANICS = "quantum_mechanics"
    PSYCHIATRY = "psychiatry"
    PHARMACOLOGY = "pharmacology"
    CLINICAL_MEDICINE = "clinical_medicine"


class AgentStatus(str, Enum):
    """Status dos agents na equipe."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class ResearchPriority(str, Enum):
    """Prioridade da pesquisa."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class InsightType(str, Enum):
    """Tipos de insights gerados."""
    HYPOTHESIS = "hypothesis"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    CONCLUSION = "conclusion"
    RECOMMENDATION = "recommendation"


# ==================== CORE MODELS ====================

class ResearchInsight(BaseModel):
    """Insight individual gerado por agent."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_specialization": "biomaterials",
                "content": "Scaffold porosity optimal range 70-90% for tissue engineering applications",
                "confidence": 0.85,
                "evidence": ["Smith et al. (2023)", "Nature Biomat review"],
                "type": "analysis"
            }
        }
    )
    
    agent_specialization: AgentSpecialization = Field(
        ..., description="Especialização do agent que gerou o insight"
    )
    content: str = Field(
        ..., description="Conteúdo do insight", min_length=10
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança no insight (0-1)"
    )
    evidence: Optional[List[str]] = Field(
        None, description="Evidências que suportam o insight"
    )
    type: InsightType = Field(
        ..., description="Tipo do insight"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp da geração"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadados adicionais"
    )


class AgentConfiguration(BaseModel):
    """Configuração individual de cada agent."""
    model_config = ConfigDict()
    
    name: str = Field(..., description="Nome do agent")
    specialization: AgentSpecialization = Field(..., description="Especialização")
    model_name: str = Field(
        default="gpt-4-turbo", description="Modelo IA a usar"
    )
    system_message: str = Field(..., description="System message do agent")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Criatividade do agent"
    )
    max_tokens: int = Field(
        default=2000, ge=100, le=8000, description="Tokens máximos por resposta"
    )
    expertise_keywords: List[str] = Field(
        default_factory=list, description="Palavras-chave da expertise"
    )
    enabled: bool = Field(default=True, description="Agent ativo na equipe")

    # Campos específicos para Vertex AI
    model_provider: str = Field("openai", description="Provedor do modelo (ex: 'openai', 'vertex_ai')")
    gcp_project_id: Optional[str] = Field(None, description="ID do projeto GCP para Vertex AI")
    gcp_location: Optional[str] = Field(None, description="Localização do GCP para Vertex AI")


class TeamConfiguration(BaseModel):
    """Configuração da equipe completa."""
    model_config = ConfigDict()
    
    team_name: str = Field(
        default="DARWIN Research Team", description="Nome da equipe"
    )
    max_round: int = Field(
        default=10, ge=2, le=50, description="Rodadas máximas de discussão"
    )
    allow_repeat_speaker: bool = Field(
        default=True, description="Permitir agent falar novamente"
    )
    agents: List[AgentConfiguration] = Field(
        ..., description="Configurações dos agents"
    )
    coordinator_config: Optional[Dict[str, Any]] = Field(
        None, description="Configuração específica do coordenador"
    )
    collaboration_rules: Optional[List[str]] = Field(
        None, description="Regras de colaboração da equipe"
    )


# ==================== REQUEST MODELS ====================

class CollaborativeResearchRequest(BaseModel):
    """Request para pesquisa colaborativa."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "research_question": "What are the optimal KEC metrics for scaffold design in bone tissue engineering?",
                "context": "Focus on porosity, mechanical properties and biocompatibility",
                "priority": "high",
                "target_specializations": ["biomaterials", "mathematics"],
                "max_agents": 3,
                "deadline_minutes": 30
            }
        }
    )
    
    research_question: str = Field(
        ..., description="Pergunta de pesquisa principal", min_length=10
    )
    context: Optional[str] = Field(
        None, description="Contexto adicional para a pesquisa"
    )
    priority: ResearchPriority = Field(
        default=ResearchPriority.MEDIUM, description="Prioridade da pesquisa"
    )
    target_specializations: Optional[List[AgentSpecialization]] = Field(
        None, description="Especializations específicas a incluir"
    )
    exclude_specializations: Optional[List[AgentSpecialization]] = Field(
        None, description="Especializations a excluir"
    )
    max_agents: int = Field(
        default=5, ge=2, le=10, description="Número máximo de agents"
    )
    max_rounds: int = Field(
        default=10, ge=2, le=50, description="Rodadas máximas de discussão"
    )
    deadline_minutes: Optional[int] = Field(
        None, ge=1, le=120, description="Deadline em minutos"
    )
    include_synthesis: bool = Field(
        default=True, description="Incluir síntese final"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Parâmetros específicos da pesquisa"
    )


class CrossDomainRequest(BaseModel):
    """Request para análise cross-domain."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_domain": "biomaterials",
                "secondary_domains": ["mathematics", "philosophy"],
                "research_topic": "Consciousness emergence in bio-neural scaffolds",
                "specific_question": "Can KEC metrics predict consciousness-like properties?"
            }
        }
    )
    
    primary_domain: AgentSpecialization = Field(
        ..., description="Domínio principal da análise"
    )
    secondary_domains: List[AgentSpecialization] = Field(
        ..., min_length=1, description="Domínios secundários"
    )
    research_topic: str = Field(
        ..., description="Tópico interdisciplinar", min_length=5
    )
    specific_question: Optional[str] = Field(
        None, description="Pergunta específica cross-domain"
    )
    data_sources: Optional[List[str]] = Field(
        None, description="Fontes de dados específicas"
    )
    methodology_requirements: Optional[List[str]] = Field(
        None, description="Requisitos metodológicos"
    )
    expected_insights: Optional[List[str]] = Field(
        None, description="Insights esperados"
    )


class BiomaterialsAnalysisRequest(BaseModel):
    """Request específico para análise biomaterials."""
    model_config = ConfigDict()
    
    scaffold_data: Optional[Dict[str, Any]] = Field(
        None, description="Dados do scaffold para análise"
    )
    kec_metrics: Optional[Dict[str, float]] = Field(
        None, description="Métricas KEC pré-calculadas"
    )
    material_properties: Optional[Dict[str, Any]] = Field(
        None, description="Propriedades do material"
    )
    application_context: str = Field(
        ..., description="Contexto da aplicação (bone, cartilage, etc.)"
    )
    performance_requirements: Optional[Dict[str, Any]] = Field(
        None, description="Requisitos de performance"
    )


# ==================== RESPONSE MODELS ====================

class CollaborativeResearchResponse(BaseModel):
    """Response da pesquisa colaborativa."""
    model_config = ConfigDict()
    
    research_id: str = Field(..., description="ID único da pesquisa")
    research_question: str = Field(..., description="Pergunta original")
    status: str = Field(..., description="Status da pesquisa")
    participating_agents: List[str] = Field(
        ..., description="Agents que participaram"
    )
    insights: List[ResearchInsight] = Field(
        ..., description="Insights gerados pela equipe"
    )
    synthesis: Optional[str] = Field(
        None, description="Síntese final colaborativa"
    )
    methodology: Optional[str] = Field(
        None, description="Metodologia utilizada"
    )
    conclusions: Optional[List[str]] = Field(
        None, description="Conclusões principais"
    )
    recommendations: Optional[List[str]] = Field(
        None, description="Recomendações"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança geral nos resultados"
    )
    collaboration_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Métricas da colaboração"
    )
    execution_time_seconds: Optional[float] = Field(
        None, description="Tempo de execução"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp da conclusão"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadados adicionais"
    )


class CrossDomainResponse(BaseModel):
    """Response da análise cross-domain."""
    model_config = ConfigDict()
    
    analysis_id: str = Field(..., description="ID único da análise")
    primary_domain: AgentSpecialization = Field(
        ..., description="Domínio principal"
    )
    secondary_domains: List[AgentSpecialization] = Field(
        ..., description="Domínios secundários"
    )
    cross_domain_insights: List[ResearchInsight] = Field(
        ..., description="Insights interdisciplinares"
    )
    domain_connections: Dict[str, Any] = Field(
        ..., description="Conexões entre domínios"
    )
    novel_perspectives: Optional[List[str]] = Field(
        None, description="Perspectivas inovadoras identificadas"
    )
    interdisciplinary_opportunities: Optional[List[str]] = Field(
        None, description="Oportunidades interdisciplinares"
    )
    synthesis_narrative: Optional[str] = Field(
        None, description="Narrativa síntese interdisciplinar"
    )
    confidence_by_domain: Dict[str, float] = Field(
        ..., description="Confiança por domínio"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AgentStatusResponse(BaseModel):
    """Status individual de agent."""
    model_config = ConfigDict()
    
    agent_name: str = Field(..., description="Nome do agent")
    specialization: AgentSpecialization = Field(..., description="Especialização")
    status: AgentStatus = Field(..., description="Status atual")
    current_task: Optional[str] = Field(None, description="Tarefa atual")
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Métricas de performance"
    )
    insights_generated: int = Field(
        default=0, description="Total de insights gerados"
    )
    collaboration_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score de colaboração"
    )
    last_active: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TeamStatusResponse(BaseModel):
    """Status da equipe completa."""
    model_config = ConfigDict()
    
    team_name: str = Field(..., description="Nome da equipe")
    total_agents: int = Field(..., description="Total de agents")
    active_agents: int = Field(..., description="Agents ativos")
    agents_status: List[AgentStatusResponse] = Field(
        ..., description="Status individual dos agents"
    )
    ongoing_researches: int = Field(
        default=0, description="Pesquisas em andamento"
    )
    completed_researches: int = Field(
        default=0, description="Pesquisas completadas"
    )
    team_performance: Optional[Dict[str, Any]] = Field(
        None, description="Métricas de performance da equipe"
    )
    collaboration_network: Optional[Dict[str, Any]] = Field(
        None, description="Rede de colaboração entre agents"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ==================== EXPORT ====================

__all__ = [
    # Enums
    "AgentSpecialization",
    "AgentStatus", 
    "ResearchPriority",
    "InsightType",
    
    # Core Models
    "ResearchInsight",
    "AgentConfiguration",
    "TeamConfiguration",
    
    # Request Models
    "CollaborativeResearchRequest",
    "CrossDomainRequest",
    "BiomaterialsAnalysisRequest",
    
    # Response Models
    "CollaborativeResearchResponse",
    "CrossDomainResponse",
    "AgentStatusResponse",
    "TeamStatusResponse",
]