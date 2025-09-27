"""AI Agents Router - AutoGen Multi-Agent Research Team Endpoints

🎯 ROUTERS REVOLUCIONÁRIOS PARA EQUIPE IA COLABORATIVA
Endpoints épicos para coordenar departamento de pesquisa IA completo com
AutoGen Multi-Agent Research Team para insights interdisciplinares disruptivos.

Features Disruptivas:
- 🤖 /research-team/collaborate - Pesquisa colaborativa épica
- 🌐 /research-team/cross-domain - Análise interdisciplinar revolutionary
- 🧬 /research-team/biomaterials-analysis - Team analysis para biomateriais
- 📊 /research-team/status - Monitoramento da equipe IA
- ⚡ /research-team/agent/{agent}/insight - Insights individuais por agent
- 🎭 /research-team/specializations - Lista de especializations disponíveis

Integration: AutoGen GroupChat + DARWIN Multi-AI Hub + Specialized Agents
"""

import asyncio
import logging
import uuid
import time
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, HTTPException, Response, Query, Path, Depends
from fastapi.responses import JSONResponse

from ..ai_agents import (
    get_research_team,
    is_research_team_available,
    initialize_research_team,
    AUTOGEN_AVAILABLE,
    CollaborativeResearchRequest,
    CollaborativeResearchResponse,
    CrossDomainRequest,
    CrossDomainResponse,
    AgentSpecialization,
    TeamStatusResponse,
    ResearchInsight,
    BiomaterialsAnalysisRequest
)
from ..ai_agents.biomaterials_agent import BiomaterialsAgent
from ..ai_agents.quantum_mechanics_agent import QuantumMechanicsAgent
from ..ai_agents.clinical_psychiatry_agent import ClinicalPsychiatryAgent
from ..ai_agents.pharmacology_agent import PharmacologyAgent
from ..ai_agents.mathematics_agent import MathematicsAgent
from ..ai_agents.philosophy_agent import PhilosophyAgent
from ..ai_agents.literature_agent import LiteratureAgent
from ..ai_agents.synthesis_agent import SynthesisAgent

from ..core.logging import get_logger

logger = get_logger("darwin.ai_agents_router")

router = APIRouter(
    prefix="/research-team",
    tags=["AI Agents Research Team"],
    responses={
        500: {"description": "Internal server error"},
        422: {"description": "Validation error"},
        503: {"description": "Research team unavailable"}
    }
)

# Global agent instances
_agents_pool: Dict[str, Any] = {}

async def get_agents_pool():
    """Obtém pool de agentes especializados."""
    global _agents_pool
    
    if not _agents_pool:
        try:
            _agents_pool = {
                "biomaterials": BiomaterialsAgent(),
                "quantum": QuantumMechanicsAgent(),
                "clinical_psychiatry": ClinicalPsychiatryAgent(),
                "pharmacology": PharmacologyAgent(),
                "mathematics": MathematicsAgent(),
                "philosophy": PhilosophyAgent(),
                "literature": LiteratureAgent(),
                "synthesis": SynthesisAgent()
            }
            logger.info(f"✅ Agent pool initialized with {len(_agents_pool)} specialists")
        except Exception as e:
            logger.error(f"Failed to initialize agent pool: {e}")
            _agents_pool = {}
    
    return _agents_pool

def check_research_team_available():
    """Verifica se research team está disponível."""
    if not is_research_team_available():
        raise HTTPException(
            status_code=503,
            detail="Research Team não está inicializado. Aguarde inicialização do sistema."
        )

# ==================== COLLABORATIVE RESEARCH ENDPOINTS ====================

@router.post("/collaborate", response_model=CollaborativeResearchResponse)
async def collaborative_research(
    request: CollaborativeResearchRequest,
    response: Response
) -> CollaborativeResearchResponse:
    """
    🤖 PESQUISA COLABORATIVA ÉPICA
    
    Coordena equipe completa de agentes IA especializados para resolver
    pergunta de pesquisa através de discussão colaborativa interdisciplinar.
    
    **Agentes Disponíveis:**
    - 🧬 Dr. Biomaterials (scaffolds, tissue engineering, KEC metrics)
    - 🔢 Dr. Mathematics (spectral analysis, graph theory, validation)
    - 🧠 Dr. Philosophy (consciousness, ethics, epistemology)
    - 📚 Dr. Literature (bibliography, synthesis, gaps)
    - 🔬 Dr. Synthesis (integration, frameworks, insights)
    - 🌌 Dr. Quantum (quantum mechanics, quantum biology)
    - 🏥 Dr. Clinical Psychiatry (medicine, psychiatry, diagnostics)
    - 💊 Dr. Pharmacology (precision dosing, quantum pharmacology)
    """
    try:
        logger.info(f"🎯 Iniciando pesquisa colaborativa: {request.research_question}")
        
        # Verificar disponibilidade
        research_team = get_research_team()
        if not research_team:
            # Tentar inicializar se não estiver ready
            try:
                research_team = await initialize_research_team()
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Research Team não pôde ser inicializado: {str(e)}"
                )
        
        # Executar pesquisa colaborativa
        result = await research_team.collaborative_research(request)
        
        # Set response headers
        response.headers["X-Research-ID"] = result.research_id
        response.headers["X-Research-Status"] = result.status
        response.headers["X-Participating-Agents"] = str(len(result.participating_agents))
        response.headers["X-Insights-Generated"] = str(len(result.insights))
        
        if result.execution_time_seconds:
            response.headers["X-Execution-Time"] = f"{result.execution_time_seconds:.2f}s"
        
        if result.confidence_score:
            response.headers["X-Confidence-Score"] = f"{result.confidence_score:.3f}"
        
        logger.info(f"✅ Pesquisa colaborativa concluída: {result.research_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na pesquisa colaborativa: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na pesquisa colaborativa: {str(e)}"
        )


@router.post("/cross-domain", response_model=CrossDomainResponse)
async def cross_domain_research(
    request: CrossDomainRequest,
    response: Response
) -> CrossDomainResponse:
    """
    🌐 ANÁLISE CROSS-DOMAIN REVOLUTIONARY
    
    Coordena agentes de diferentes domínios para análise interdisciplinar
    e descoberta de conexões inovadoras entre campos de conhecimento.
    """
    try:
        logger.info(f"🌐 Iniciando análise cross-domain: {request.research_topic}")
        
        research_team = get_research_team()
        if not research_team:
            research_team = await initialize_research_team()
        
        # Executar análise cross-domain
        result = await research_team.cross_domain_analysis(request)
        
        # Set response headers
        response.headers["X-Analysis-ID"] = result.analysis_id
        response.headers["X-Primary-Domain"] = result.primary_domain.value
        response.headers["X-Secondary-Domains"] = ",".join([d.value for d in result.secondary_domains])
        response.headers["X-Cross-Domain-Insights"] = str(len(result.cross_domain_insights))
        
        logger.info(f"✅ Análise cross-domain concluída: {result.analysis_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise cross-domain: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise cross-domain: {str(e)}"
        )


@router.post("/biomaterials-analysis", response_model=ResearchInsight)
async def biomaterials_team_analysis(
    request: BiomaterialsAnalysisRequest,
    response: Response
) -> ResearchInsight:
    """
    🧬 TEAM ANALYSIS BIOMATERIALS SPECIALIZADA
    
    Análise especializada de biomateriais usando o agent expert
    Dr. Biomaterials com integração KEC metrics e tissue engineering.
    """
    try:
        logger.info(f"🧬 Análise biomaterials especializada para {request.application_context}")
        
        # Obter agent biomaterials
        agents_pool = await get_agents_pool()
        biomaterials_agent = agents_pool.get("biomaterials")
        
        if not biomaterials_agent:
            raise HTTPException(
                status_code=503,
                detail="Biomaterials Agent não está disponível"
            )
        
        # Executar análise especializada
        result = await biomaterials_agent.analyze_scaffold(request)
        
        # Set response headers
        response.headers["X-Agent-Specialization"] = result.agent_specialization.value
        response.headers["X-Analysis-Type"] = "biomaterials_specialized"
        response.headers["X-Application-Context"] = request.application_context
        response.headers["X-Confidence"] = f"{result.confidence:.3f}"
        
        logger.info(f"✅ Análise biomaterials concluída com confiança {result.confidence:.2f}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise biomaterials: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise biomaterials: {str(e)}"
        )


# ==================== INDIVIDUAL AGENT ENDPOINTS ====================

@router.get("/agent/{agent_name}/insight")
async def get_agent_individual_insight(
    agent_name: str = Path(..., description="Nome do agente"),
    research_question: str = Query(..., description="Pergunta de pesquisa"),
    context: Optional[str] = Query(None, description="Contexto adicional"),
    response: Response = None
) -> ResearchInsight:
    """
    ⚡ INSIGHT INDIVIDUAL POR AGENT
    
    Obtém insight especializado de agent individual para pergunta específica.
    
    **Agents Disponíveis:**
    - biomaterials, quantum, clinical_psychiatry, pharmacology,
    - mathematics, philosophy, literature, synthesis
    """
    try:
        logger.info(f"⚡ Gerando insight individual: {agent_name} -> {research_question}")
        
        # Obter agent específico
        agents_pool = await get_agents_pool()
        agent = agents_pool.get(agent_name.lower())
        
        if not agent:
            available_agents = list(agents_pool.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' não encontrado. Disponíveis: {available_agents}"
            )
        
        # Gerar insight colaborativo
        result = await agent.generate_collaborative_insight(research_question, context)
        
        # Set response headers
        response.headers["X-Agent-Name"] = agent.name
        response.headers["X-Agent-Specialization"] = result.agent_specialization.value
        response.headers["X-Insight-Type"] = result.type.value
        response.headers["X-Confidence"] = f"{result.confidence:.3f}"
        
        logger.info(f"✅ Insight individual gerado por {agent_name}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao gerar insight individual: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao gerar insight do agent {agent_name}: {str(e)}"
        )


@router.get("/agent/{agent_name}/expertise")
async def get_agent_expertise(
    agent_name: str = Path(..., description="Nome do agente")
) -> Dict[str, Any]:
    """
    📊 EXPERTISE SUMMARY DO AGENT
    
    Retorna resumo completo da expertise e capabilities do agent.
    """
    try:
        # Obter agent específico
        agents_pool = await get_agents_pool()
        agent = agents_pool.get(agent_name.lower())
        
        if not agent:
            available_agents = list(agents_pool.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' não encontrado. Disponíveis: {available_agents}"
            )
        
        # Obter expertise summary
        expertise = agent.get_expertise_summary()
        
        return {
            "agent_name": agent_name,
            "expertise_summary": expertise,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter expertise: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao obter expertise do agent {agent_name}: {str(e)}"
        )


# ==================== TEAM MANAGEMENT ENDPOINTS ====================

@router.get("/status", response_model=TeamStatusResponse)
async def get_team_status() -> TeamStatusResponse:
    """
    📊 STATUS DA EQUIPE REVOLUTIONARY
    
    Status completo da equipe de pesquisa multi-agent incluindo:
    - Status individual de cada agent
    - Métricas de colaboração
    - Performance da equipe
    - Histórico de pesquisas
    """
    try:
        logger.info("📊 Obtendo status da equipe de pesquisa")
        
        research_team = get_research_team()
        if not research_team:
            # Se não inicializado, retornar status básico
            agents_pool = await get_agents_pool()
            
            return TeamStatusResponse(
                team_name="DARWIN Revolutionary Research Team",
                total_agents=len(agents_pool),
                active_agents=0,
                agents_status=[],
                ongoing_researches=0,
                completed_researches=0,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Obter status completo da equipe
        status = await research_team.get_team_status()
        
        logger.info(f"✅ Status da equipe obtido: {status.active_agents}/{status.total_agents} agents ativos")
        return status
        
    except Exception as e:
        logger.error(f"Erro ao obter status da equipe: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao obter status da equipe: {str(e)}"
        )


@router.get("/specializations")
async def get_available_specializations() -> Dict[str, Any]:
    """
    🎭 ESPECIALIZATIONS DISPONÍVEIS
    
    Lista todas as especializations de agents disponíveis na equipe
    com descrições e capabilities de cada uma.
    """
    try:
        specializations = {}
        
        for spec in AgentSpecialization:
            specializations[spec.value] = {
                "name": spec.value.replace("_", " ").title(),
                "description": _get_specialization_description(spec),
                "keywords": _get_specialization_keywords(spec),
                "agent_available": True  # Por enquanto, todos estão implementados
            }
        
        return {
            "available_specializations": specializations,
            "total_count": len(specializations),
            "autogen_enabled": AUTOGEN_AVAILABLE,
            "team_available": is_research_team_available(),
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar especializações: {e}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao listar especializações disponíveis"
        )


@router.post("/initialize")
async def initialize_team() -> Dict[str, Any]:
    """
    🚀 INICIALIZAR RESEARCH TEAM
    
    Inicializa a equipe de pesquisa multi-agent se ainda não estiver ativa.
    """
    try:
        logger.info("🚀 Solicitação de inicialização da equipe")
        
        if is_research_team_available():
            return {
                "status": "already_initialized",
                "message": "Research Team já está ativo e funcionando",
                "timestamp": datetime.now(timezone.utc)
            }
        
        # Inicializar equipe
        research_team = await initialize_research_team()
        
        return {
            "status": "initialized",
            "message": "Research Team inicializado com sucesso",
            "team_name": research_team.team_name,
            "agents_count": len(research_team.agents) if hasattr(research_team, 'agents') else 8,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Erro na inicialização da equipe: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na inicialização da equipe: {str(e)}"
        )


# ==================== SPECIALIZED ANALYSIS ENDPOINTS ====================

@router.post("/quantum-analysis")
async def quantum_biomaterial_analysis(
    material_properties: Dict[str, Any],
    temperature: float = Query(298.15, ge=0.1, le=1000.0, description="Temperatura em Kelvin"),
    context: Optional[str] = Query(None, description="Contexto da análise"),
    response: Response = None
) -> ResearchInsight:
    """
    🌌 ANÁLISE QUÂNTICA DE BIOMATERIAIS
    
    Análise especializada de efeitos quânticos em biomateriais
    usando Dr. Quantum - expert em mecânica quântica aplicada.
    """
    try:
        logger.info(f"🌌 Análise quântica de biomateriais a T={temperature}K")
        
        agents_pool = await get_agents_pool()
        quantum_agent = agents_pool.get("quantum")
        
        if not quantum_agent:
            raise HTTPException(
                status_code=503,
                detail="Quantum Agent não está disponível"
            )
        
        # Executar análise quântica
        result = await quantum_agent.analyze_quantum_effects_in_biomaterials(
            material_properties, temperature, context
        )
        
        response.headers["X-Analysis-Type"] = "quantum_biomaterials"
        response.headers["X-Temperature"] = f"{temperature}K"
        response.headers["X-Quantum-Agent"] = quantum_agent.name
        
        logger.info("✅ Análise quântica de biomateriais concluída")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise quântica: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise quântica: {str(e)}"
        )


@router.post("/pharmacological-analysis")
async def pharmacological_team_analysis(
    drug_name: str = Query(..., description="Nome do medicamento"),
    patient_genetics: Optional[Dict[str, Any]] = None,
    biomaterial_delivery: Optional[Dict[str, Any]] = None,
    response: Response = None
) -> ResearchInsight:
    """
    💊 ANÁLISE FARMACOLÓGICA TEAM
    
    Análise farmacológica especializada incluindo farmacologia quântica
    e psicofarmacologia de precisão usando Dr. Pharmacology.
    """
    try:
        logger.info(f"💊 Análise farmacológica de {drug_name}")
        
        agents_pool = await get_agents_pool()
        pharmacology_agent = agents_pool.get("pharmacology")
        
        if not pharmacology_agent:
            raise HTTPException(
                status_code=503,
                detail="Pharmacology Agent não está disponível"
            )
        
        # Criar drug profile básico para análise
        from ..ai_agents.pharmacology_agent import DrugProfile, DrugClass, PharmacogeneticProfile
        
        # Drug profile simplificado - em implementação real viria de database
        drug_profile = DrugProfile(
            name=drug_name,
            drug_class=DrugClass.ANTIDEPRESSANTS,  # Default
            mechanism_of_action="To be determined",
            half_life=24.0,
            bioavailability=0.8,
            protein_binding=0.9,
            metabolism_pathway="CYP2D6",
            therapeutic_range=(50.0, 200.0),
            side_effects=["nausea", "headache"],
            contraindications=["hypersensitivity"]
        )
        
        # Converter genetics se fornecida
        genetics = None
        if patient_genetics:
            genetics = PharmacogeneticProfile(
                cyp2d6_phenotype=patient_genetics.get("cyp2d6", "extensive"),
                cyp2c19_phenotype=patient_genetics.get("cyp2c19", "extensive"),
                cyp3a4_activity=patient_genetics.get("cyp3a4", "normal"),
                transporter_variants=patient_genetics.get("transporters", {}),
                receptor_polymorphisms=patient_genetics.get("receptors", {}),
                drug_sensitivities=patient_genetics.get("sensitivities", [])
            )
        
        # Executar análise farmacológica
        result = await pharmacology_agent.pharmacological_analysis(
            drug_profile, genetics, biomaterial_delivery
        )
        
        response.headers["X-Drug-Name"] = drug_name
        response.headers["X-Analysis-Type"] = "pharmacological_team"
        response.headers["X-Has-Genetics"] = str(genetics is not None)
        response.headers["X-Has-Biomaterial"] = str(biomaterial_delivery is not None)
        
        logger.info("✅ Análise farmacológica team concluída")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise farmacológica: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise farmacológica: {str(e)}"
        )


# ==================== HELPER FUNCTIONS ====================

def _get_specialization_description(spec: AgentSpecialization) -> str:
    """Descrição das especializações."""
    descriptions = {
        AgentSpecialization.BIOMATERIALS: "Expert em biomateriais, scaffolds, tissue engineering e KEC metrics",
        AgentSpecialization.MATHEMATICS: "Expert em análise espectral, teoria de grafos e validação matemática",
        AgentSpecialization.PHILOSOPHY: "Expert em filosofia da mente, consciência e epistemologia",
        AgentSpecialization.LITERATURE: "Expert em literatura científica, revisão bibliográfica e síntese",
        AgentSpecialization.SYNTHESIS: "Expert em integração interdisciplinar e síntese de insights",
        AgentSpecialization.QUANTUM_MECHANICS: "Expert em mecânica quântica, física quântica e quantum biology",
        AgentSpecialization.PSYCHIATRY: "Expert em clínica médica, psiquiatria e neuropsiquiatria",
        AgentSpecialization.PHARMACOLOGY: "Expert em farmacologia quântica e psicofarmacologia de precisão"
    }
    return descriptions.get(spec, f"Especialização {spec.value}")

def _get_specialization_keywords(spec: AgentSpecialization) -> List[str]:
    """Keywords das especializações."""
    keywords = {
        AgentSpecialization.BIOMATERIALS: ["scaffold", "porosity", "biocompatibility", "tissue engineering"],
        AgentSpecialization.MATHEMATICS: ["spectral", "eigenvalue", "graph theory", "validation"],
        AgentSpecialization.PHILOSOPHY: ["consciousness", "ethics", "epistemology", "methodology"],
        AgentSpecialization.LITERATURE: ["bibliography", "review", "synthesis", "evidence"],
        AgentSpecialization.SYNTHESIS: ["integration", "interdisciplinary", "framework", "insights"],
        AgentSpecialization.QUANTUM_MECHANICS: ["quantum", "coherence", "tunneling", "quantum biology"],
        AgentSpecialization.PSYCHIATRY: ["psychiatry", "clinical", "diagnosis", "treatment"],
        AgentSpecialization.PHARMACOLOGY: ["pharmacology", "precision dosing", "drug interactions", "quantum effects"]
    }
    return keywords.get(spec, [spec.value])


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def research_team_health() -> Dict[str, Any]:
    """Health check para o sistema de agents."""
    try:
        agents_pool = await get_agents_pool()
        research_team = get_research_team()
        
        return {
            "status": "healthy" if agents_pool else "degraded",
            "autogen_available": AUTOGEN_AVAILABLE,
            "research_team_initialized": research_team is not None,
            "agents_pool_size": len(agents_pool),
            "available_agents": list(agents_pool.keys()),
            "specializations_count": len(AgentSpecialization),
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

# ==================== HEALTH CHECKS ESPECÍFICOS PARA JAX E AUTOGEN ====================

@router.get("/health/jax")
async def jax_health_check() -> Dict[str, Any]:
    """
    🏥 HEALTH CHECK ESPECÍFICO PARA JAX
    
    Verifica status do JAX Engine incluindo:
    - Disponibilidade do JAX
    - Hardware detectado (GPU/TPU/CPU)
    - Status de inicialização
    - Estatísticas de performance
    """
    try:
        from ..performance import JAX_AVAILABLE, GPU_AVAILABLE, TPU_AVAILABLE, DEVICE_TYPE, get_performance_engine
        
        # Tentar obter instância do JAX Engine se disponível
        jax_engine = get_performance_engine()
        
        health_status = {
            "jax_available": JAX_AVAILABLE,
            "gpu_available": GPU_AVAILABLE,
            "tpu_available": TPU_AVAILABLE,
            "device_type": DEVICE_TYPE,
            "engine_initialized": jax_engine.is_initialized if jax_engine else False,
            "status": "healthy" if JAX_AVAILABLE else "degraded",
            "capabilities": {
                "jit_compilation": JAX_AVAILABLE,
                "gpu_acceleration": GPU_AVAILABLE,
                "tpu_acceleration": TPU_AVAILABLE,
                "optax_optimization": False  # Será preenchido abaixo
            },
            "performance_stats": jax_engine.get_performance_summary() if jax_engine else {},
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Adicionar info Optax se disponível
        try:
            from ..performance.jax_kec_engine import OPTAX_AVAILABLE
            health_status["capabilities"]["optax_optimization"] = OPTAX_AVAILABLE
        except:
            health_status["capabilities"]["optax_optimization"] = False
        
        return health_status
        
    except Exception as e:
        logger.error(f"JAX health check failed: {e}")
        return {
            "jax_available": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

@router.get("/health/autogen")
async def autogen_health_check() -> Dict[str, Any]:
    """
    🏥 HEALTH CHECK ESPECÍFICO PARA AUTOGEN
    
    Verifica status do AutoGen Research Team incluindo:
    - Disponibilidade do AutoGen
    - Status de inicialização
    - Agents ativos
    - Estatísticas de colaboração
    """
    try:
        from ..ai_agents import AUTOGEN_AVAILABLE, get_research_team
        
        # Tentar obter instância do Research Team se disponível
        research_team = get_research_team()
        team_status = {}
        if research_team:
            team_status = await research_team.get_team_status()
        
        health_status = {
            "autogen_available": AUTOGEN_AVAILABLE,
            "research_team_initialized": research_team.is_initialized if research_team else False,
            "status": "healthy" if AUTOGEN_AVAILABLE else "degraded",
            "agents_count": team_status.total_agents if hasattr(team_status, 'total_agents') else 0,
            "active_agents": team_status.active_agents if hasattr(team_status, 'active_agents') else 0,
            "completed_researches": team_status.completed_researches if hasattr(team_status, 'completed_researches') else 0,
            "team_performance": team_status.team_performance if hasattr(team_status, 'team_performance') else {},
            "timestamp": datetime.now(timezone.utc)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"AutoGen health check failed: {e}")
        return {
            "autogen_available": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

@router.get("/health/jax-autogen-integration")
async def jax_autogen_integration_health() -> Dict[str, Any]:
    """
    🔄 HEALTH CHECK DA INTEGRAÇÃO JAX + AUTOGEN
    
    Verifica a integração entre JAX e AutoGen para múltiplos agentes IA:
    - Disponibilidade de ambos os sistemas
    - Status de integração
    - Capacidades combinadas
    """
    try:
        # Obter status individual
        jax_health = await jax_health_check()
        autogen_health = await autogen_health_check()
        
        integration_status = {
            "jax_status": jax_health["status"],
            "autogen_status": autogen_health["status"],
            "integration_healthy": jax_health["status"] == "healthy" and autogen_health["status"] == "healthy",
            "combined_capabilities": {
                "jax_ultra_performance": jax_health["jax_available"],
                "autogen_multi_agent": autogen_health["autogen_available"],
                "gpu_acceleration": jax_health.get("gpu_available", False),
                "multi_agent_collaboration": autogen_health["autogen_available"],
                "real_time_optimization": jax_health["capabilities"].get("optax_optimization", False)
            },
            "performance_metrics": {
                "jax_speedup": jax_health.get("performance_stats", {}).get("average_speedup", 0),
                "autogen_collaborations": autogen_health.get("completed_researches", 0)
            },
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Determinar status geral
        if integration_status["integration_healthy"]:
            integration_status["overall_status"] = "healthy"
            integration_status["status_message"] = "JAX + AutoGen integration fully operational"
        else:
            integration_status["overall_status"] = "degraded"
            integration_status["status_message"] = "JAX + AutoGen integration partially degraded"
        
        return integration_status
        
    except Exception as e:
        logger.error(f"JAX+AutoGen integration health check failed: {e}")
        return {
            "integration_healthy": False,
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

# ==================== DIAGNOSTIC ENDPOINTS ====================

@router.get("/diagnostics/jax")
async def jax_diagnostics() -> Dict[str, Any]:
    """
    🔧 DIAGNÓSTICO COMPLETO DO JAX
    
    Retorna informações detalhadas de diagnóstico do JAX Engine:
    - Informações de hardware
    - Versões das bibliotecas
    - Configurações
    - Problemas detectados
    """
    try:
        from ..performance import JAX_AVAILABLE, GPU_AVAILABLE, TPU_AVAILABLE, DEVICE_TYPE, get_performance_engine, get_hardware_info
        
        # Obter informações de hardware
        hardware_info = get_hardware_info()
        
        diagnostics = {
            "hardware": {
                "device_type": DEVICE_TYPE,
                "gpu_count": hardware_info.get("hardware", {}).get("gpus", 0),
                "tpu_count": hardware_info.get("hardware", {}).get("tpus", 0),
                "cpu_count": hardware_info.get("hardware", {}).get("cpus", 0)
            },
            "versions": {
                "jax_available": JAX_AVAILABLE,
                "optax_available": False,  # Será preenchido abaixo
                "python_version": sys.version
            },
            "configuration": {
                "default_device": DEVICE_TYPE,
                "jit_enabled": JAX_AVAILABLE,
                "gpu_acceleration": GPU_AVAILABLE,
                "tpu_acceleration": TPU_AVAILABLE
            },
            "performance": {
                "initialized": False,
                "stats": {}
            },
            "issues_detected": [],
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Adicionar informações do engine se disponível
        try:
            jax_engine = get_performance_engine()
            if jax_engine:
                diagnostics["performance"]["initialized"] = jax_engine.is_initialized
                diagnostics["performance"]["stats"] = jax_engine.get_performance_summary()
                
                # Verificar problemas comuns
                if not JAX_AVAILABLE:
                    diagnostics["issues_detected"].append("JAX not available - using NumPy fallback")
                if not GPU_AVAILABLE and not TPU_AVAILABLE:
                    diagnostics["issues_detected"].append("No GPU/TPU acceleration available")
                if not jax_engine.is_initialized:
                    diagnostics["issues_detected"].append("JAX Engine not initialized")
                    
        except Exception as e:
            diagnostics["issues_detected"].append(f"Error accessing JAX Engine: {str(e)}")
        
        # Adicionar info Optax se disponível
        try:
            from ..performance.jax_kec_engine import OPTAX_AVAILABLE
            diagnostics["versions"]["optax_available"] = OPTAX_AVAILABLE
        except:
            diagnostics["versions"]["optax_available"] = False
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"JAX diagnostics failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

@router.get("/diagnostics/autogen")
async def autogen_diagnostics() -> Dict[str, Any]:
    """
    🔧 DIAGNÓSTICO COMPLETO DO AUTOGEN
    
    Retorna informações detalhadas de diagnóstico do AutoGen:
    - Configuração dos agents
    - Status de colaboração
    - Problemas detectados
    """
    try:
        from ..ai_agents import AUTOGEN_AVAILABLE, get_research_team
        
        diagnostics = {
            "autogen_available": AUTOGEN_AVAILABLE,
            "research_team_initialized": False,
            "agents_configuration": {},
            "collaboration_metrics": {},
            "issues_detected": [],
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Adicionar informações do research team se disponível
        try:
            research_team = get_research_team()
            if research_team:
                diagnostics["research_team_initialized"] = research_team.is_initialized
                
                # Obter status detalhado
                team_status = await research_team.get_team_status()
                if hasattr(team_status, 'total_agents'):
                    diagnostics["agents_configuration"] = {
                        "total_agents": team_status.total_agents,
                        "active_agents": team_status.active_agents,
                        "agents_status": [{
                            "name": agent.agent_name,
                            "specialization": agent.specialization.value,
                            "status": agent.status.value,
                            "insights_generated": agent.insights_generated
                        } for agent in team_status.agents_status] if hasattr(team_status, 'agents_status') else []
                    }
                    diagnostics["collaboration_metrics"] = team_status.team_performance if hasattr(team_status, 'team_performance') else {}
                
                # Verificar problemas comuns
                if not AUTOGEN_AVAILABLE:
                    diagnostics["issues_detected"].append("AutoGen not available - using fallback mode")
                if not research_team.is_initialized:
                    diagnostics["issues_detected"].append("Research Team not initialized")
                if hasattr(team_status, 'active_agents') and team_status.active_agents == 0:
                    diagnostics["issues_detected"].append("No active agents")
                    
        except Exception as e:
            diagnostics["issues_detected"].append(f"Error accessing Research Team: {str(e)}")
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"AutoGen diagnostics failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

# ==================== EXPORTS ====================

__all__ = ["router"]