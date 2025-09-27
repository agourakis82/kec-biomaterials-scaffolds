"""Multi-AI Hub Models - Modelos para orchestração de múltiplas IAs."""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class AIProvider(str, Enum):
    """Provedores de IA disponíveis."""
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    GEMINI = "gemini"


class AIModel(str, Enum):
    """Modelos específicos por provedor."""
    # ChatGPT Models
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    
    # Claude Models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    
    # Gemini Models
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class ScientificDomain(str, Enum):
    """Domínios científicos para roteamento."""
    KEC_ANALYSIS = "kec_analysis"
    MATHEMATICAL_PROOFS = "mathematical_proofs"
    ALGORITHM_DESIGN = "algorithm_design"
    BIOMATERIALS = "biomaterials"
    SCAFFOLD_DESIGN = "scaffold_design"
    MATERIALS_ENGINEERING = "materials_engineering"
    LITERATURE_SEARCH = "literature_search"
    RESEARCH_SYNTHESIS = "research_synthesis"
    ACADEMIC_WRITING = "academic_writing"
    PHILOSOPHY = "philosophy"
    CONSCIOUSNESS = "consciousness"
    ETHICS = "ethics"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    INTERDISCIPLINARY = "interdisciplinary"


class ChatMessage(BaseModel):
    """Mensagem de chat universal."""
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Papel da mensagem"
    )
    content: str = Field(..., description="Conteúdo da mensagem")
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow, description="Timestamp da mensagem"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata adicional"
    )


class ChatRequest(BaseModel):
    """Request de chat universal."""
    message: str = Field(..., description="Mensagem do usuário")
    domain: Optional[ScientificDomain] = Field(
        default=None, description="Domínio científico (para roteamento automático)"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="ID da conversa existente"
    )
    preferred_ai: Optional[AIProvider] = Field(
        default=None, description="IA preferida (override do roteamento)"
    )
    model: Optional[AIModel] = Field(
        default=None, description="Modelo específico (override)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Contexto adicional"
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperatura para geração"
    )
    max_tokens: Optional[int] = Field(
        default=1000, ge=1, le=8000, description="Máximo de tokens"
    )
    include_context: bool = Field(
        default=True, description="Incluir contexto da conversa"
    )


class ChatResponse(BaseModel):
    """Response de chat padronizada."""
    message: str = Field(..., description="Resposta da IA")
    ai_provider: AIProvider = Field(..., description="IA que gerou a resposta")
    model: AIModel = Field(..., description="Modelo específico usado")
    conversation_id: str = Field(..., description="ID da conversa")
    domain: Optional[ScientificDomain] = Field(
        default=None, description="Domínio detectado/usado"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp da resposta"
    )
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Informações de uso (tokens, custo)"
    )
    routing_reason: Optional[str] = Field(
        default=None, description="Razão do roteamento para esta IA"
    )
    confidence_score: Optional[float] = Field(
        default=None, description="Confiança na escolha da IA"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata adicional"
    )


class ConversationCreate(BaseModel):
    """Criação de nova conversa."""
    title: Optional[str] = Field(default=None, description="Título da conversa")
    domain: Optional[ScientificDomain] = Field(
        default=None, description="Domínio principal"
    )
    description: Optional[str] = Field(
        default=None, description="Descrição da conversa"
    )
    preferred_ai: Optional[AIProvider] = Field(
        default=None, description="IA preferida para esta conversa"
    )
    participants: Optional[List[AIProvider]] = Field(
        default=None, description="IAs participantes (multi-AI)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata da conversa"
    )


class ConversationHistory(BaseModel):
    """Histórico completo de conversa."""
    conversation_id: str = Field(..., description="ID único da conversa")
    title: Optional[str] = Field(default=None, description="Título")
    domain: Optional[ScientificDomain] = Field(default=None, description="Domínio")
    messages: List[ChatMessage] = Field(default=[], description="Mensagens")
    participants: List[AIProvider] = Field(
        default=[], description="IAs participantes"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Data de criação"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Última atualização"
    )
    total_messages: int = Field(default=0, description="Total de mensagens")
    total_tokens: Optional[int] = Field(default=None, description="Total de tokens")
    estimated_cost: Optional[float] = Field(default=None, description="Custo estimado")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")


class ContextSyncRequest(BaseModel):
    """Request para sincronização de contexto."""
    conversation_id: str = Field(..., description="ID da conversa")
    context: Dict[str, Any] = Field(..., description="Contexto a sincronizar")
    source_ai: AIProvider = Field(..., description="IA origem do contexto")
    target_ais: Optional[List[AIProvider]] = Field(
        default=None, description="IAs alvo (None = todas)"
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Prioridade da sincronização"
    )


class ModelRecommendation(BaseModel):
    """Recomendação de modelo/IA."""
    recommended_ai: AIProvider = Field(..., description="IA recomendada")
    recommended_model: AIModel = Field(..., description="Modelo recomendado")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança na recomendação"
    )
    reasoning: str = Field(..., description="Explicação da recomendação")
    domain: Optional[ScientificDomain] = Field(
        default=None, description="Domínio detectado"
    )
    alternative_options: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Opções alternativas"
    )


# ==================== SPECIALIZED REQUESTS ====================

class BiomaterialsChatRequest(ChatRequest):
    """Chat especializado para biomateriais."""
    scaffold_type: Optional[str] = Field(
        default=None, description="Tipo de scaffold"
    )
    material_properties: Optional[Dict[str, Any]] = Field(
        default=None, description="Propriedades do material"
    )
    kec_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Métricas KEC relevantes"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Como otimizar porosidade para regeneração óssea?",
                "domain": "biomaterials",
                "scaffold_type": "ceramic",
                "material_properties": {"porosity": 0.7, "pore_size": "100-500um"},
                "kec_metrics": {"h_spectral": 2.34, "h_forman": 1.89}
            }
        }


class MathematicalChatRequest(ChatRequest):
    """Chat especializado para matemática/algoritmos."""
    proof_type: Optional[str] = Field(
        default=None, description="Tipo de prova matemática"
    )
    complexity_level: Optional[Literal["undergraduate", "graduate", "research"]] = Field(
        default=None, description="Nível de complexidade"
    )
    mathematical_domain: Optional[str] = Field(
        default=None, description="Área matemática específica"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Prove que H_spectral é monotônica para grafos bipartidos",
                "domain": "mathematical_proofs",
                "proof_type": "graph_theory",
                "complexity_level": "research",
                "mathematical_domain": "algebraic_topology"
            }
        }


class PhilosophyChatRequest(ChatRequest):
    """Chat especializado para filosofia/ética."""
    philosophical_school: Optional[str] = Field(
        default=None, description="Escola filosófica"
    )
    ethical_framework: Optional[str] = Field(
        default=None, description="Framework ético"
    )
    consciousness_level: Optional[Literal["phenomenological", "computational", "integrated"]] = Field(
        default=None, description="Nível de análise da consciência"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Qual a relação entre consciência e complexidade topológica?",
                "domain": "consciousness",
                "philosophical_school": "phenomenology",
                "consciousness_level": "integrated"
            }
        }


class ResearchChatRequest(ChatRequest):
    """Chat especializado para pesquisa acadêmica."""
    research_field: Optional[str] = Field(
        default=None, description="Campo de pesquisa"
    )
    paper_type: Optional[Literal["review", "empirical", "theoretical", "meta-analysis"]] = Field(
        default=None, description="Tipo de paper"
    )
    citation_style: Optional[Literal["APA", "MLA", "Chicago", "Nature", "IEEE"]] = Field(
        default="APA", description="Estilo de citação"
    )
    target_journal: Optional[str] = Field(
        default=None, description="Journal alvo"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Sintetize literatura sobre scaffolds inteligentes 2020-2024",
                "domain": "research_synthesis",
                "research_field": "biomaterials_engineering",
                "paper_type": "review",
                "citation_style": "Nature"
            }
        }


class CrossDomainChatRequest(ChatRequest):
    """Chat interdisciplinar."""
    primary_domain: ScientificDomain = Field(
        ..., description="Domínio primário"
    )
    secondary_domains: List[ScientificDomain] = Field(
        ..., description="Domínios secundários"
    )
    integration_approach: Optional[Literal["synthesis", "bridge", "fusion"]] = Field(
        default="synthesis", description="Abordagem de integração"
    )
    complexity_preference: Optional[Literal["simple", "moderate", "advanced"]] = Field(
        default="moderate", description="Preferência de complexidade"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Como consciência quântica influencia design de biomateriais?",
                "domain": "interdisciplinary",
                "primary_domain": "biomaterials",
                "secondary_domains": ["consciousness", "quantum_mechanics"],
                "integration_approach": "fusion",
                "complexity_preference": "advanced"
            }
        }


# ==================== ANALYTICS MODELS ====================

class UsageAnalytics(BaseModel):
    """Análise de uso do Multi-AI Hub."""
    total_requests: int = Field(default=0, description="Total de requests")
    requests_by_ai: Dict[AIProvider, int] = Field(
        default={}, description="Requests por IA"
    )
    requests_by_domain: Dict[ScientificDomain, int] = Field(
        default={}, description="Requests por domínio"
    )
    total_tokens: int = Field(default=0, description="Total de tokens")
    estimated_costs: Dict[AIProvider, float] = Field(
        default={}, description="Custos por IA"
    )
    avg_response_time: Dict[AIProvider, float] = Field(
        default={}, description="Tempo médio de resposta por IA"
    )
    success_rate: Dict[AIProvider, float] = Field(
        default={}, description="Taxa de sucesso por IA"
    )
    period_start: datetime = Field(..., description="Início do período")
    period_end: datetime = Field(..., description="Fim do período")


class PerformanceMetrics(BaseModel):
    """Métricas de performance comparativa."""
    ai_provider: AIProvider = Field(..., description="Provedor da IA")
    model: AIModel = Field(..., description="Modelo")
    avg_latency: float = Field(..., description="Latência média (ms)")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Taxa de sucesso")
    quality_score: Optional[float] = Field(
        default=None, ge=0.0, le=10.0, description="Score de qualidade"
    )
    cost_per_request: float = Field(..., description="Custo por request")
    tokens_per_request: float = Field(..., description="Tokens por request")
    domain_specialty: Optional[List[ScientificDomain]] = Field(
        default=None, description="Domínios de especialidade"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp da métrica"
    )


class ExtractedInsight(BaseModel):
    """Insight extraído automaticamente."""
    insight_id: str = Field(..., description="ID único do insight")
    content: str = Field(..., description="Conteúdo do insight")
    source_conversation: str = Field(..., description="ID da conversa origem")
    domains: List[ScientificDomain] = Field(..., description="Domínios relacionados")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança no insight"
    )
    novelty_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Score de novidade"
    )
    related_concepts: Optional[List[str]] = Field(
        default=None, description="Conceitos relacionados"
    )
    potential_applications: Optional[List[str]] = Field(
        default=None, description="Aplicações potenciais"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Data de extração"
    )
    ai_source: AIProvider = Field(..., description="IA que gerou o insight")


# ==================== ROUTING MODELS ====================

class RoutingRule(BaseModel):
    """Regra de roteamento."""
    domain: ScientificDomain = Field(..., description="Domínio científico")
    preferred_ai: AIProvider = Field(..., description="IA preferida")
    model: AIModel = Field(..., description="Modelo específico")
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Threshold de confiança"
    )
    keywords: Optional[List[str]] = Field(
        default=None, description="Palavras-chave para detecção"
    )
    fallback_ai: Optional[AIProvider] = Field(
        default=None, description="IA de fallback"
    )
    reasoning: str = Field(..., description="Justificativa da regra")


class RoutingDecision(BaseModel):
    """Decisão de roteamento."""
    selected_ai: AIProvider = Field(..., description="IA selecionada")
    selected_model: AIModel = Field(..., description="Modelo selecionado")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confiança na decisão"
    )
    reasoning: str = Field(..., description="Explicação da decisão")
    detected_domain: Optional[ScientificDomain] = Field(
        default=None, description="Domínio detectado"
    )
    keywords_matched: Optional[List[str]] = Field(
        default=None, description="Palavras-chave encontradas"
    )
    fallback_used: bool = Field(
        default=False, description="Se usou fallback"
    )
    processing_time_ms: float = Field(..., description="Tempo de processamento")


# ==================== HEALTH CHECK ====================

class MultiAIHealthCheck(BaseModel):
    """Health check do Multi-AI Hub."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Status geral"
    )
    ai_providers: Dict[AIProvider, Dict[str, Any]] = Field(
        ..., description="Status de cada IA"
    )
    routing_engine: Dict[str, Any] = Field(
        ..., description="Status do engine de roteamento"
    )
    context_bridge: Dict[str, Any] = Field(
        ..., description="Status do context bridge"
    )
    conversation_manager: Dict[str, Any] = Field(
        ..., description="Status do conversation manager"
    )
    total_conversations: int = Field(
        default=0, description="Total de conversas ativas"
    )
    uptime_seconds: float = Field(..., description="Tempo ativo")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Última verificação"
    )


__all__ = [
    # Enums
    "AIProvider",
    "AIModel", 
    "ScientificDomain",
    
    # Core Models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ConversationCreate",
    "ConversationHistory",
    "ContextSyncRequest",
    "ModelRecommendation",
    
    # Specialized Requests
    "BiomaterialsChatRequest",
    "MathematicalChatRequest", 
    "PhilosophyChatRequest",
    "ResearchChatRequest",
    "CrossDomainChatRequest",
    
    # Analytics
    "UsageAnalytics",
    "PerformanceMetrics",
    "ExtractedInsight",
    
    # Routing
    "RoutingRule",
    "RoutingDecision",
    
    # Health
    "MultiAIHealthCheck"
]