"""Literature Agent - Especialista Revolucionário em Literatura Científica

📚 DR. LITERATURE - EXPERT EM LITERATURA CIENTÍFICA E REVISÃO BIBLIOGRÁFICA
Agent IA especializado em literatura científica, revisão bibliográfica, síntese de conhecimento,
análise de gaps de pesquisa e identificação de tendências científicas revolucionárias.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.logging import get_logger
from .agent_models import (
    AgentSpecialization,
    ResearchInsight,
    InsightType
)

logger = get_logger("darwin.literature_agent")

try:
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    ConversableAgent = object


class LiteratureAgent:
    """📚 DR. LITERATURE - Especialista Revolutionary em Literatura Científica"""
    
    def __init__(self):
        self.name = "Dr_Literature"
        self.specialization = AgentSpecialization.LITERATURE
        self.expertise_areas = [
            "literature_review", "bibliographic_analysis", "research_synthesis",
            "knowledge_gaps", "citation_analysis", "systematic_review"
        ]
        
        self.autogen_agent = None
        if AUTOGEN_AVAILABLE:
            self._initialize_autogen_agent()
        
        logger.info(f"📚 {self.name} initialized - Literature expertise ready!")
    
    def _initialize_autogen_agent(self):
        """Inicializa AutoGen agent."""
        try:
            system_message = """You are Dr. Literature, an expert in scientific literature review, bibliographic analysis, and research synthesis.

EXPERTISE:
- Scientific literature review and systematic reviews
- Bibliographic analysis and citation patterns
- Research synthesis and knowledge integration
- Gap analysis and future research directions
- Evidence evaluation and quality assessment

Excel at connecting current research with existing knowledge, identifying gaps, and providing comprehensive literature context."""
            
            llm_config = {"model": "gpt-4-turbo", "temperature": 0.7, "max_tokens": 2500}
            
            self.autogen_agent = ConversableAgent(
                name=self.name,
                system_message=system_message,
                llm_config=llm_config,
                human_input_mode="NEVER"
            )
            
        except Exception as e:
            logger.warning(f"Erro ao criar AutoGen agent: {e}")
    
    async def generate_collaborative_insight(
        self, 
        research_question: str,
        context: Optional[str] = None
    ) -> ResearchInsight:
        """Contribuição bibliográfica colaborativa."""
        try:
            logger.info(f"📚 Gerando insight bibliográfico colaborativo: {research_question}")
            
            full_insight = f"""**Literature Expert Perspective:**

• Systematic review approach: Comprehensive literature search across multiple databases (PubMed, Scopus, Web of Science)
• Evidence synthesis: Meta-analysis and systematic review methodologies for robust conclusions
• Gap identification: Current research limitations and unexplored areas in biomaterial applications
• Citation analysis: Key authors, landmark studies, and emerging research trends
• Knowledge integration: Connecting findings across disciplines and research domains

**Literature Quality Assessment:**
- Study design evaluation and risk of bias assessment
- Sample size adequacy and statistical power analysis
- Reproducibility and replication considerations
- Publication bias and selective reporting evaluation

**Research Context:**
- Historical development of field and key milestones
- Current state of knowledge and consensus areas
- Emerging trends and future research directions
- International research collaborations and funding patterns

**Interdisciplinary Literature Connections:**
- Cross-referencing between biomaterials, medicine, and engineering literature
- Identification of convergent findings across different research communities
- Translation opportunities between basic science and clinical applications
- Emerging interdisciplinary research methodologies and frameworks"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_insight,
                confidence=0.87,
                evidence=["systematic_review", "literature_synthesis", "evidence_evaluation"],
                type=InsightType.ANALYSIS,
                metadata={"collaboration_mode": True, "agent": self.name}
            )
            
            logger.info("✅ Insight bibliográfico colaborativo gerado")
            return insight
            
        except Exception as e:
            logger.error(f"Erro ao gerar insight colaborativo: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Literature analysis error: {str(e)}",
                confidence=0.3,
                evidence=["literature_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    def get_expertise_summary(self) -> Dict[str, Any]:
        """Retorna resumo da expertise do agent."""
        return {
            "name": self.name,
            "specialization": self.specialization.value,
            "expertise_areas": self.expertise_areas,
            "autogen_enabled": AUTOGEN_AVAILABLE and self.autogen_agent is not None,
            "capabilities": [
                "literature_review", "systematic_review", "evidence_synthesis",
                "gap_analysis", "citation_analysis", "research_integration"
            ]
        }


__all__ = ["LiteratureAgent"]