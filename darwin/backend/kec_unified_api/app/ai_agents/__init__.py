"""AutoGen Multi-Agent Research Team - Sistema Revolucionário

🚀 DEPARTAMENTO DE PESQUISA IA COLABORATIVO
Sistema disruptivo de agentes especializados trabalhando em equipe para resolver
problemas complexos interdisciplinares em biomateriais, matemática, filosofia e mais.

Features Épicas:
- 🎯 Research Team Coordinator com GroupChat collaborative
- 🧬 Biomaterials Expert Agent (especialista em scaffolds e KEC)
- 🔢 Mathematics Expert Agent (análise espectral e teoria de grafos)  
- 🧠 Philosophy Expert Agent (consciência e epistemologia)
- 📚 Literature Expert Agent (revisão e síntese bibliográfica)
- 🔬 Synthesis Agent (integração de insights interdisciplinares)

Tecnologia: AutoGen Framework + Multi-AI Hub Integration
"""

from typing import Optional, Dict, Any, List, Union
import logging

from ..core.logging import get_logger

logger = get_logger("darwin.ai_agents")

# Importações condicionais para AutoGen
try:
    import autogen
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
    logger.info("🎯 AutoGen framework loaded successfully - Multi-Agent Research Team Ready!")
except ImportError as e:
    logger.warning(f"AutoGen não disponível - funcionando sem Multi-Agent Research Team: {e}")
    AUTOGEN_AVAILABLE = False
    # Fallback types
    ConversableAgent = object
    GroupChat = object  
    GroupChatManager = object

# Importar componentes principais
from .research_team import ResearchTeamCoordinator
from .biomaterials_agent import BiomaterialsAgent
from .mathematics_agent import MathematicsAgent
from .philosophy_agent import PhilosophyAgent
from .literature_agent import LiteratureAgent
from .synthesis_agent import SynthesisAgent
from .quantum_mechanics_agent import QuantumMechanicsAgent
from .clinical_psychiatry_agent import ClinicalPsychiatryAgent
from .pharmacology_agent import PharmacologyAgent

# Importar models
from .agent_models import (
    CollaborativeResearchRequest,
    CollaborativeResearchResponse,
    CrossDomainRequest,
    CrossDomainResponse,
    BiomaterialsAnalysisRequest,
    AgentSpecialization,
    ResearchInsight,
    AgentStatus,
    AgentStatusResponse,
    TeamStatusResponse,
    TeamConfiguration
)

# Global research team instance
_research_team: Optional[ResearchTeamCoordinator] = None

async def initialize_research_team() -> ResearchTeamCoordinator:
    """Inicializa equipe de pesquisa multi-agent."""
    global _research_team
    
    if not AUTOGEN_AVAILABLE:
        logger.warning("AutoGen não disponível - Research Team funcionará em modo limitado")
        
    try:
        logger.info("🚀 Inicializando AutoGen Multi-Agent Research Team...")
        
        _research_team = ResearchTeamCoordinator()
        await _research_team.initialize()
        
        logger.info("✅ Research Team inicializado com sucesso - Departamento IA Colaborativo Ativo!")
        return _research_team
        
    except Exception as e:
        logger.error(f"Falha na inicialização do Research Team: {e}")
        raise

async def shutdown_research_team():
    """Shutdown da equipe de pesquisa."""
    global _research_team
    
    if _research_team:
        try:
            await _research_team.shutdown()
            logger.info("🛑 Research Team shutdown complete")
        except Exception as e:
            logger.error(f"Erro no shutdown do Research Team: {e}")
        finally:
            _research_team = None

def get_research_team() -> Optional[ResearchTeamCoordinator]:
    """Retorna instância da equipe de pesquisa."""
    return _research_team

def is_research_team_available() -> bool:
    """Verifica se o research team está disponível."""
    return AUTOGEN_AVAILABLE and _research_team is not None

# Status de disponibilidade
__status__ = {
    "autogen_available": AUTOGEN_AVAILABLE,
    "research_team_ready": False,
    "agents_count": 8,
    "specializations": [
        "biomaterials", "mathematics", "philosophy",
        "literature", "synthesis", "quantum_mechanics",
        "psychiatry", "pharmacology"
    ]
}

# Exports
__all__ = [
    # Core components
    "ResearchTeamCoordinator",
    "BiomaterialsAgent",
    "MathematicsAgent",
    "PhilosophyAgent",
    "LiteratureAgent",
    "SynthesisAgent",
    "QuantumMechanicsAgent",
    "ClinicalPsychiatryAgent",
    "PharmacologyAgent",
    
    # Models
    "CollaborativeResearchRequest",
    "CollaborativeResearchResponse",
    "CrossDomainRequest",
    "CrossDomainResponse",
    "BiomaterialsAnalysisRequest",
    "AgentSpecialization",
    "ResearchInsight",
    "AgentStatus",
    "AgentStatusResponse",
    "TeamStatusResponse",
    "TeamConfiguration",
    
    # Functions
    "initialize_research_team",
    "shutdown_research_team",
    "get_research_team",
    "is_research_team_available",
    
    # Constants
    "AUTOGEN_AVAILABLE",
    "__status__"
]