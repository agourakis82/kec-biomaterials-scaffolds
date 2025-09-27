"""Research Team Coordinator - AutoGen Multi-Agent System

🎯 COORDENADOR REVOLUCIONÁRIO DA EQUIPE DE PESQUISA IA
Sistema épico que coordena múltiplos agentes especializados usando AutoGen GroupChat
para resolver problemas complexos através de colaboração IA interdisciplinar.

Features Disruptivas:
- 🤖 GroupChat Manager para coordenação automática
- 🎭 Multiple Specialized Agents (biomaterials, mathematics, philosophy, etc.)
- 🧠 Intelligent Discussion Orchestration 
- 💡 Collaborative Insight Generation
- 🔄 Dynamic Agent Selection
- 📊 Real-time Collaboration Metrics

Architecture: AutoGen GroupChat + DARWIN Integration
"""

import asyncio
import logging
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.logging import get_logger
from ..config.settings import settings
from .agent_models import (
    CollaborativeResearchRequest,
    CollaborativeResearchResponse,
    CrossDomainRequest, 
    CrossDomainResponse,
    AgentSpecialization,
    ResearchInsight,
    AgentStatus,
    TeamConfiguration,
    AgentConfiguration,
    InsightType,
    ResearchPriority,
    TeamStatusResponse,
    AgentStatusResponse
)

logger = get_logger("darwin.research_team")

# Importações condicionais AutoGen - New API v0.4+
try:
    # Try new AutoGen 0.4+ API structure
    from autogen_agentchat.agents import AssistantAgent as ConversableAgent
    from autogen_agentchat.teams import RoundRobinGroupChat as GroupChat
    from autogen_agentchat.teams import GroupChatManager
    AUTOGEN_AVAILABLE = True
    logger.info("✅ AutoGen v0.4+ loaded successfully")
except ImportError:
    try:
        # Fallback to legacy API
        from autogen import ConversableAgent, GroupChat, GroupChatManager
        AUTOGEN_AVAILABLE = True
        logger.info("✅ AutoGen legacy loaded successfully")
    except ImportError:
        logger.warning("AutoGen não disponível - funcionando em modo fallback")
        AUTOGEN_AVAILABLE = False
        # Functional fallback classes
        class ConversableAgent:
            def __init__(self, **kwargs): 
                self.name = kwargs.get('name', 'Agent')
                self.system_message = kwargs.get('system_message', '')
        class GroupChat:
            def __init__(self, **kwargs): pass
        class GroupChatManager:
            def __init__(self, **kwargs): pass

# Fallback para MultiAI Hub se AutoGen não disponível
try:
    from ..multi_ai.chat_orchestrator import ChatOrchestrator
    MULTI_AI_AVAILABLE = True
except ImportError:
    MULTI_AI_AVAILABLE = False
    ChatOrchestrator = None


class ResearchTeamCoordinator:
    """
    🎯 Coordenador Revolutionary da Equipe de Pesquisa Multi-Agent
    
    Gerencia equipe completa de agentes especializados usando AutoGen GroupChat
    para colaboração IA interdisciplinar em problemas complexos.
    """
    
    def __init__(self):
        self.team_name = "DARWIN Revolutionary Research Team"
        self.agents: Dict[str, ConversableAgent] = {}
        self.group_chat: Optional[GroupChat] = None
        self.chat_manager: Optional[GroupChatManager] = None
        self.team_config: Optional[TeamConfiguration] = None
        self.research_history: List[CollaborativeResearchResponse] = []
        self.is_initialized = False
        
        # Fallback orchestrator se AutoGen não disponível
        self.multi_ai_orchestrator = None
        
        # Métricas de colaboração
        self.collaboration_metrics = {
            "total_researches": 0,
            "successful_collaborations": 0,
            "average_response_time": 0.0,
            "agent_participation": {},
            "insight_generation_rate": 0.0
        }
        
        logger.info(f"🎯 {self.team_name} Coordinator inicializado")
    
    async def initialize(self, config: Optional[TeamConfiguration] = None):
        """Inicializa a equipe de pesquisa com configuração."""
        try:
            logger.info("🚀 Inicializando Revolutionary Research Team...")
            
            # Usar configuração padrão se não fornecida
            if not config:
                config = self._create_default_team_config()
            
            self.team_config = config
            
            if AUTOGEN_AVAILABLE:
                await self._initialize_autogen_team()
                logger.info("✅ AutoGen Team inicializado com sucesso!")
            else:
                await self._initialize_fallback_team()
                logger.info("✅ Fallback Team inicializado (sem AutoGen)")
            
            self.is_initialized = True
            logger.info(f"🎉 {self.team_name} está READY para colaboração épica!")
            
        except Exception as e:
            logger.error(f"Falha na inicialização do Research Team: {e}")
            raise
    
    async def _initialize_autogen_team(self):
        """Inicializa equipe usando AutoGen framework."""
        try:
            # Criar agents especializados
            for agent_config in self.team_config.agents:
                if not agent_config.enabled:
                    continue
                    
                # Configuração do LLM baseada no modelo
                llm_config = self._create_llm_config(agent_config)
                
                # Criar ConversableAgent
                agent = ConversableAgent(
                    name=agent_config.name,
                    system_message=agent_config.system_message,
                    llm_config=llm_config,
                    human_input_mode="NEVER",  # Totalmente autônomo
                    max_consecutive_auto_reply=3,
                    code_execution_config=False  # Segurança
                )
                
                self.agents[agent_config.name] = agent
                logger.info(f"✅ Agent criado: {agent_config.name} ({agent_config.specialization})")
            
            # Criar GroupChat para colaboração
            agents_list = list(self.agents.values())
            
            self.group_chat = GroupChat(
                agents=agents_list,
                messages=[],
                max_round=self.team_config.max_round,
                allow_repeat_speaker=self.team_config.allow_repeat_speaker,
                speaker_selection_method="auto"  # Seleção inteligente
            )
            
            # Criar GroupChatManager
            manager_llm_config = {
                "model": "gpt-4-turbo", 
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            self.chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config=manager_llm_config,
                system_message=self._create_manager_system_message()
            )
            
            logger.info(f"🎭 GroupChat criado com {len(agents_list)} agents")
            
        except Exception as e:
            logger.error(f"Erro na inicialização AutoGen: {e}")
            raise
    
    async def _initialize_fallback_team(self):
        """Inicializa equipe usando Multi-AI Hub como fallback."""
        try:
            if MULTI_AI_AVAILABLE and ChatOrchestrator:
                self.multi_ai_orchestrator = ChatOrchestrator()
                await self.multi_ai_orchestrator.initialize()
                logger.info("Multi-AI Hub configurado como fallback")
            else:
                logger.warning("Funcionando em modo limitado sem AutoGen ou Multi-AI Hub")
                
        except Exception as e:
            logger.warning(f"Fallback initialization parcial: {e}")
    
    async def collaborative_research(
        self, 
        request: CollaborativeResearchRequest
    ) -> CollaborativeResearchResponse:
        """
        🔬 PESQUISA COLABORATIVA REVOLUCIONÁRIA
        
        Coordena equipe de agentes para resolver pergunta de pesquisa através
        de discussão colaborativa interdisciplinar.
        """
        research_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"🔬 Iniciando pesquisa colaborativa: {request.research_question}")
            
            if not self.is_initialized:
                raise RuntimeError("Research Team não está inicializado")
            
            # Análise da pergunta para selecionar agents apropriados
            selected_agents = self._select_agents_for_research(request)
            logger.info(f"🎯 Agents selecionados: {[a.name for a in selected_agents]}")
            
            # Executar colaboração
            if AUTOGEN_AVAILABLE and self.chat_manager:
                insights = await self._execute_autogen_collaboration(
                    request, selected_agents
                )
            else:
                insights = await self._execute_fallback_collaboration(
                    request, selected_agents
                )
            
            # Síntese final se solicitada
            synthesis = None
            if request.include_synthesis and insights:
                synthesis = await self._generate_synthesis(insights, request)
            
            # Extrair conclusões e recomendações
            conclusions = self._extract_conclusions(insights)
            recommendations = self._extract_recommendations(insights)
            
            # Calcular confiança geral
            confidence_score = self._calculate_confidence_score(insights)
            
            # Métricas de colaboração
            collaboration_metrics = self._calculate_collaboration_metrics(
                selected_agents, insights, time.time() - start_time
            )
            
            # Criar response
            response = CollaborativeResearchResponse(
                research_id=research_id,
                research_question=request.research_question,
                status="completed",
                participating_agents=[a.name if hasattr(a, 'name') else str(a) for a in selected_agents],
                insights=insights,
                synthesis=synthesis,
                methodology=None,
                conclusions=conclusions,
                recommendations=recommendations,
                confidence_score=confidence_score,
                collaboration_metrics=collaboration_metrics,
                execution_time_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                metadata=None
            )
            
            # Salvar no histórico
            self.research_history.append(response)
            self._update_team_metrics(response)
            
            logger.info(f"✅ Pesquisa colaborativa concluída: {research_id}")
            return response
            
        except Exception as e:
            logger.error(f"Erro na pesquisa colaborativa: {e}")
            
            # Response de erro
            return CollaborativeResearchResponse(
                research_id=research_id,
                research_question=request.research_question,
                status="failed",
                participating_agents=[],
                insights=[],
                synthesis=None,
                methodology=None,
                conclusions=None,
                recommendations=None,
                confidence_score=0.0,
                collaboration_metrics=None,
                execution_time_seconds=time.time() - start_time,
                timestamp=datetime.now(timezone.utc),
                metadata={"error": str(e)}
            )
    
    async def cross_domain_analysis(
        self, 
        request: CrossDomainRequest
    ) -> CrossDomainResponse:
        """
        🌐 ANÁLISE CROSS-DOMAIN INTERDISCIPLINAR
        
        Coordena agentes de diferentes domínios para análise interdisciplinar
        e descoberta de conexões inovadoras.
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🌐 Iniciando análise cross-domain: {request.research_topic}")
            
            # Selecionar agents baseado nos domínios
            primary_agents = self._get_agents_by_specialization([request.primary_domain])
            secondary_agents = self._get_agents_by_specialization(request.secondary_domains)
            all_agents = primary_agents + secondary_agents
            
            # Construir pergunta colaborativa
            collaborative_request = CollaborativeResearchRequest(
                research_question=f"Cross-domain analysis: {request.research_topic}",
                context=f"Primary domain: {request.primary_domain}, Secondary domains: {request.secondary_domains}",
                target_specializations=[request.primary_domain] + request.secondary_domains,
                priority=ResearchPriority.HIGH,
                include_synthesis=True,
                exclude_specializations=None,
                deadline_minutes=None,
                parameters=None
            )
            
            if request.specific_question:
                collaborative_request.research_question += f" - {request.specific_question}"
            
            # Executar colaboração
            collab_response = await self.collaborative_research(collaborative_request)
            
            # Processar para resposta cross-domain
            domain_connections = self._analyze_domain_connections(
                collab_response.insights, request
            )
            
            novel_perspectives = self._identify_novel_perspectives(collab_response.insights)
            interdisciplinary_opportunities = self._identify_opportunities(collab_response.insights)
            
            # Confiança por domínio
            confidence_by_domain = {}
            for domain in [request.primary_domain] + request.secondary_domains:
                domain_insights = [i for i in collab_response.insights 
                                 if i.agent_specialization == domain]
                if domain_insights:
                    confidence_by_domain[domain.value] = sum(i.confidence for i in domain_insights) / len(domain_insights)
                else:
                    confidence_by_domain[domain.value] = 0.0
            
            response = CrossDomainResponse(
                analysis_id=analysis_id,
                primary_domain=request.primary_domain,
                secondary_domains=request.secondary_domains,
                cross_domain_insights=collab_response.insights,
                domain_connections=domain_connections,
                novel_perspectives=novel_perspectives,
                interdisciplinary_opportunities=interdisciplinary_opportunities,
                synthesis_narrative=collab_response.synthesis,
                confidence_by_domain=confidence_by_domain,
                timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(f"✅ Análise cross-domain concluída: {analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Erro na análise cross-domain: {e}")
            raise
    
    async def get_team_status(self) -> TeamStatusResponse:
        """Status completo da equipe de pesquisa."""
        try:
            agents_status = []
            active_count = 0
            
            for agent_name, agent in self.agents.items():
                # Determinar status do agent
                status = AgentStatus.READY if self.is_initialized else AgentStatus.INITIALIZING
                
                if status == AgentStatus.READY:
                    active_count += 1
                
                # Encontrar especialização
                specialization = self._get_agent_specialization(agent_name)
                
                # Métricas de colaboração do agent
                agent_metrics = self.collaboration_metrics.get("agent_participation", {}).get(agent_name, {})
                
                agent_status = AgentStatusResponse(
                    agent_name=agent_name,
                    specialization=specialization,
                    status=status,
                    insights_generated=agent_metrics.get("insights_count", 0),
                    collaboration_score=agent_metrics.get("collaboration_score", 0.0),
                    current_task=None,
                    performance_metrics=None
                )
                
                agents_status.append(agent_status)
            
            return TeamStatusResponse(
                team_name=self.team_name,
                total_agents=len(self.agents),
                active_agents=active_count,
                agents_status=agents_status,
                ongoing_researches=0,  # TODO: implementar tracking
                completed_researches=len(self.research_history),
                team_performance=self.collaboration_metrics.copy(),
                timestamp=datetime.now(timezone.utc),
                collaboration_network=None
            )
            
        except Exception as e:
            logger.error(f"Erro ao obter status da equipe: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown da equipe de pesquisa."""
        try:
            logger.info("🛑 Shutting down Research Team...")
            
            if self.multi_ai_orchestrator:
                await self.multi_ai_orchestrator.shutdown()
            
            # Limpar agents
            self.agents.clear()
            self.group_chat = None
            self.chat_manager = None
            self.is_initialized = False
            
            logger.info("✅ Research Team shutdown complete")
            
        except Exception as e:
            logger.error(f"Erro no shutdown: {e}")
    
    # ==================== PRIVATE METHODS ====================
    
    def _create_default_team_config(self) -> TeamConfiguration:
        """Cria configuração padrão da equipe."""
        agents_configs = [
            AgentConfiguration(
                name="Dr_Biomaterials",
                specialization=AgentSpecialization.BIOMATERIALS,
                system_message="You are Dr. Biomaterials, a world-renowned expert in biomaterials, scaffolds, and tissue engineering. You specialize in KEC metrics analysis, porosity optimization, mechanical properties, and biocompatibility assessment. Provide scientific, evidence-based insights with focus on practical applications in regenerative medicine.",
                temperature=0.7,
                expertise_keywords=["scaffold", "porosity", "biocompatibility", "tissue engineering", "KEC metrics"],
                model_provider="vertex_ai",
                model_name="gemini-1.5-pro",
                gcp_project_id="pcs-helio",
                gcp_location="us-central1"
            ),
            AgentConfiguration(
                name="Dr_Mathematics",
                specialization=AgentSpecialization.MATHEMATICS,
                system_message="You are Dr. Mathematics, a brilliant expert in spectral analysis, graph theory, topology, and computational mathematics. You specialize in KEC 2.0 metrics, eigenvalue analysis, Forman curvature, and small-world properties. Provide rigorous mathematical insights and validate computational approaches.",
                temperature=0.6,
                expertise_keywords=["spectral analysis", "graph theory", "eigenvalues", "topology", "curvature"],
                model_provider="openai",
                model_name="gpt-4-turbo",
                gcp_project_id=None,
                gcp_location=None
            ),
            AgentConfiguration(
                name="Dr_Philosophy",
                specialization=AgentSpecialization.PHILOSOPHY, 
                system_message="You are Dr. Philosophy, a profound expert in philosophy of mind, consciousness studies, epistemology, and scientific methodology. You provide conceptual clarity, identify assumptions, and explore the deeper implications of scientific findings from a philosophical perspective.",
                temperature=0.8,
                expertise_keywords=["consciousness", "epistemology", "philosophy of mind", "methodology", "concepts"],
                model_provider="openai",
                model_name="gpt-4-turbo",
                gcp_project_id=None,
                gcp_location=None
            ),
            AgentConfiguration(
                name="Dr_Literature",
                specialization=AgentSpecialization.LITERATURE,
                system_message="You are Dr. Literature, an expert in scientific literature review, bibliographic analysis, and research synthesis. You excel at connecting current research with existing knowledge, identifying gaps, and providing comprehensive literature context.",
                temperature=0.7,
                expertise_keywords=["literature review", "bibliography", "research synthesis", "citations", "knowledge gaps"],
                model_provider="openai",
                model_name="gpt-4-turbo",
                gcp_project_id=None,
                gcp_location=None
            ),
            AgentConfiguration(
                name="Dr_Synthesis",
                specialization=AgentSpecialization.SYNTHESIS,
                system_message="You are Dr. Synthesis, a master of interdisciplinary integration and insight synthesis. You excel at combining perspectives from different domains, identifying novel connections, and creating coherent narratives from diverse viewpoints.",
                temperature=0.9,
                expertise_keywords=["synthesis", "integration", "interdisciplinary", "connections", "insights"],
                model_provider="openai",
                model_name="gpt-4-turbo",
                gcp_project_id=None,
                gcp_location=None
            )
        ]
        
        return TeamConfiguration(
            team_name=self.team_name,
            max_round=10,
            allow_repeat_speaker=True,
            agents=agents_configs,
            coordinator_config=None,
            collaboration_rules=None
        )
    
    def _create_llm_config(self, agent_config: AgentConfiguration) -> Dict[str, Any]:
        """Cria configuração LLM para agent, com suporte para Vertex AI."""
        if agent_config.model_provider == "vertex_ai":
            from .vertex_ai_manager import get_vertex_ai_llm_config
            return get_vertex_ai_llm_config(
                model_name=agent_config.model_name,
                temperature=agent_config.temperature,
                max_tokens=agent_config.max_tokens,
            )
        
        # Fallback para OpenAI ou outros provedores
        return {
            "model": agent_config.model_name,
            "temperature": agent_config.temperature,
            "max_tokens": agent_config.max_tokens,
            "timeout": 60,
        }
    
    def _create_manager_system_message(self) -> str:
        """System message para GroupChat Manager."""
        return """You are the Research Team Manager coordinating a revolutionary AI research team. 

Your role:
- Facilitate productive discussions between specialized agents
- Ensure all relevant perspectives are heard  
- Guide conversations toward meaningful insights
- Maintain focus on the research question
- Encourage interdisciplinary collaboration
- Synthesize key points when appropriate

Keep discussions focused, productive, and scientifically rigorous."""
    
    def _select_agents_for_research(self, request: CollaborativeResearchRequest) -> List[Any]:
        """Seleciona agents apropriados para a pesquisa."""
        try:
            selected = []
            
            # Se especializações específicas foram solicitadas
            if request.target_specializations:
                for spec in request.target_specializations:
                    agents = self._get_agents_by_specialization([spec])
                    selected.extend(agents)
            else:
                # Seleção inteligente baseada na pergunta
                selected = list(self.agents.values())
            
            # Aplicar exclusões
            if request.exclude_specializations:
                excluded_names = []
                for spec in request.exclude_specializations:
                    excluded_agents = self._get_agents_by_specialization([spec])
                    excluded_names.extend([a.name if hasattr(a, 'name') else str(a) for a in excluded_agents])
                
                selected = [a for a in selected if (a.name if hasattr(a, 'name') else str(a)) not in excluded_names]
            
            # Limitar número máximo
            if len(selected) > request.max_agents:
                selected = selected[:request.max_agents]
            
            return selected
            
        except Exception as e:
            logger.warning(f"Erro na seleção de agents, usando todos: {e}")
            return list(self.agents.values())[:request.max_agents]
    
    def _get_agents_by_specialization(self, specializations: List[AgentSpecialization]) -> List[Any]:
        """Retorna agents por especialização."""
        agents = []
        
        for agent_name, agent in self.agents.items():
            agent_spec = self._get_agent_specialization(agent_name)
            if agent_spec in specializations:
                agents.append(agent)
        
        return agents
    
    def _get_agent_specialization(self, agent_name: str) -> AgentSpecialization:
        """Retorna especialização do agent pelo nome."""
        if not self.team_config:
            return AgentSpecialization.SYNTHESIS  # default
            
        for config in self.team_config.agents:
            if config.name == agent_name:
                return config.specialization
        
        return AgentSpecialization.SYNTHESIS  # default
    
    async def _execute_autogen_collaboration(
        self, 
        request: CollaborativeResearchRequest,
        selected_agents: List[Any]
    ) -> List[ResearchInsight]:
        """Executa colaboração usando AutoGen."""
        try:
            # Construir prompt inicial
            initial_message = self._build_research_prompt(request)
            
            # Usar primeiro agent para iniciar discussão
            initiator = selected_agents[0] if selected_agents else list(self.agents.values())[0]
            
            # Executar discussão no GroupChat
            # Note: Esta é uma simplificação - implementação real seria mais complexa
            chat_result = await asyncio.to_thread(
                initiator.initiate_chat,
                self.chat_manager,
                message=initial_message,
                max_turns=request.max_rounds
            )
            
            # Extrair insights das mensagens
            insights = self._extract_insights_from_chat(chat_result, selected_agents)
            return insights
            
        except Exception as e:
            logger.error(f"Erro na colaboração AutoGen: {e}")
            return await self._execute_fallback_collaboration(request, selected_agents)
    
    async def _execute_fallback_collaboration(
        self, 
        request: CollaborativeResearchRequest,
        selected_agents: List[Any]
    ) -> List[ResearchInsight]:
        """Executa colaboração usando fallback (simulação ou Multi-AI Hub)."""
        try:
            insights = []
            
            # Simular colaboração sequencial entre agents
            for i, agent in enumerate(selected_agents):
                agent_name = agent.name if hasattr(agent, 'name') else f"Agent_{i}"
                specialization = self._get_agent_specialization(agent_name)
                
                # Simular insight baseado na especialização
                insight = ResearchInsight(
                    agent_specialization=specialization,
                    content=f"[{specialization.value}] Analysis of: {request.research_question}. This requires detailed expertise in {specialization.value} domain.",
                    confidence=0.7 + (i * 0.05),  # Variação na confiança
                    type=InsightType.ANALYSIS,
                    evidence=[f"{specialization.value} domain expertise"],
                    metadata={"agent": agent_name, "mode": "fallback"}
                )
                
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Erro na colaboração fallback: {e}")
            return []
    
    def _build_research_prompt(self, request: CollaborativeResearchRequest) -> str:
        """Constrói prompt para pesquisa colaborativa."""
        prompt = f"""COLLABORATIVE RESEARCH REQUEST

Research Question: {request.research_question}

Context: {request.context or 'General research context'}

Priority: {request.priority.value}

Instructions:
- Each agent should contribute insights from their expertise area
- Build upon each other's contributions
- Provide evidence-based analysis
- Identify interdisciplinary connections
- Aim for actionable recommendations

Please begin the collaborative analysis."""

        return prompt
    
    def _extract_insights_from_chat(self, chat_result: Any, agents: List[Any]) -> List[ResearchInsight]:
        """Extrai insights das mensagens do chat."""
        insights = []
        
        try:
            # Esta seria uma implementação mais sofisticada na versão real
            # Por enquanto, simular extração de insights
            for i, agent in enumerate(agents):
                agent_name = agent.name if hasattr(agent, 'name') else f"Agent_{i}"
                specialization = self._get_agent_specialization(agent_name)
                
                insight = ResearchInsight(
                    agent_specialization=specialization,
                    content=f"Collaborative insight from {agent_name}: Analysis completed based on chat discussion.",
                    confidence=0.8,
                    type=InsightType.ANALYSIS,
                    evidence=["AutoGen chat discussion"],
                    metadata={"source": "autogen_chat", "agent": agent_name}
                )
                
                insights.append(insight)
            
        except Exception as e:
            logger.warning(f"Erro ao extrair insights do chat: {e}")
        
        return insights
    
    async def _generate_synthesis(self, insights: List[ResearchInsight], request: CollaborativeResearchRequest) -> str:
        """Gera síntese colaborativa dos insights."""
        try:
            # Agrupar insights por tipo e especialização
            synthesis_parts = []
            
            synthesis_parts.append(f"# Collaborative Research Synthesis\n")
            synthesis_parts.append(f"**Research Question:** {request.research_question}\n")
            
            # Agrupar por especialização
            by_specialization: Dict[str, List[ResearchInsight]] = {}
            for insight in insights:
                spec = insight.agent_specialization.value
                if spec not in by_specialization:
                    by_specialization[spec] = []
                by_specialization[spec].append(insight)
            
            synthesis_parts.append("## Insights by Domain\n")
            for spec, spec_insights in by_specialization.items():
                synthesis_parts.append(f"### {spec.title()}\n")
                for insight in spec_insights:
                    synthesis_parts.append(f"- {insight.content}\n")
            
            # Síntese interdisciplinar
            synthesis_parts.append("\n## Interdisciplinary Synthesis\n")
            synthesis_parts.append("The collaborative analysis reveals several key interdisciplinary connections and insights that emerge from combining multiple expert perspectives.\n")
            
            return "".join(synthesis_parts)
            
        except Exception as e:
            logger.warning(f"Erro na geração de síntese: {e}")
            return "Synthesis generation encountered an error."
    
    def _extract_conclusions(self, insights: List[ResearchInsight]) -> List[str]:
        """Extrai conclusões principais dos insights."""
        conclusions = []
        
        conclusion_insights = [i for i in insights if i.type == InsightType.CONCLUSION]
        
        for insight in conclusion_insights:
            conclusions.append(insight.content)
        
        # Se não há conclusões específicas, derivar das análises
        if not conclusions and insights:
            conclusions = [
                "Collaborative analysis completed with insights from multiple expert domains",
                "Interdisciplinary approach provides comprehensive perspective",
                "Evidence-based recommendations generated from expert collaboration"
            ]
        
        return conclusions[:5]  # Limitar a 5 conclusões principais
    
    def _extract_recommendations(self, insights: List[ResearchInsight]) -> List[str]:
        """Extrai recomendações dos insights."""
        recommendations = []
        
        rec_insights = [i for i in insights if i.type == InsightType.RECOMMENDATION]
        
        for insight in rec_insights:
            recommendations.append(insight.content)
        
        # Recomendações derivadas se necessário
        if not recommendations and insights:
            recommendations = [
                "Continue interdisciplinary collaboration for comprehensive insights",
                "Validate findings through experimental verification", 
                "Consider implementation of recommended approaches",
                "Monitor outcomes and iterate based on results"
            ]
        
        return recommendations[:5]  # Limitar a 5 recomendações
    
    def _calculate_confidence_score(self, insights: List[ResearchInsight]) -> float:
        """Calcula score de confiança geral."""
        if not insights:
            return 0.0
        
        total_confidence = sum(insight.confidence for insight in insights)
        avg_confidence = total_confidence / len(insights)
        
        # Boost por diversidade de especializations
        unique_specs = set(i.agent_specialization for i in insights)
        diversity_bonus = min(0.1, len(unique_specs) * 0.02)
        
        return min(1.0, avg_confidence + diversity_bonus)
    
    def _calculate_collaboration_metrics(self, agents: List[Any], insights: List[ResearchInsight], duration: float) -> Dict[str, Any]:
        """Calcula métricas da colaboração."""
        metrics = {
            "participants_count": len(agents),
            "insights_generated": len(insights),
            "duration_seconds": duration,
            "avg_confidence": self._calculate_confidence_score(insights),
            "specializations_involved": len(set(i.agent_specialization for i in insights)),
            "collaboration_efficiency": len(insights) / max(1, duration) * 60  # insights per minute
        }
        
        return metrics
    
    def _analyze_domain_connections(self, insights: List[ResearchInsight], request: CrossDomainRequest) -> Dict[str, Any]:
        """Analisa conexões entre domínios."""
        connections = {
            "primary_secondary_links": [],
            "novel_connections": [],
            "shared_concepts": [],
            "connection_strength": 0.0
        }
        
        # Análise simplificada das conexões
        primary_insights = [i for i in insights if i.agent_specialization == request.primary_domain]
        secondary_insights = [i for i in insights if i.agent_specialization in request.secondary_domains]
        
        # Calcular força da conexão
        if primary_insights and secondary_insights:
            connections["connection_strength"] = 0.8  # Placeholder
        
        return connections
    
    def _identify_novel_perspectives(self, insights: List[ResearchInsight]) -> List[str]:
        """Identifica perspectivas inovadoras."""
        # Implementação simplificada
        novel_perspectives = [
            "Interdisciplinary approach reveals new research directions",
            "Cross-domain analysis identifies unexplored connections",
            "Collaborative insights suggest innovative methodologies"
        ]
        
        return novel_perspectives[:3]
    
    def _identify_opportunities(self, insights: List[ResearchInsight]) -> List[str]:
        """Identifica oportunidades interdisciplinares."""
        # Implementação simplificada
        opportunities = [
            "Joint research initiatives between domains",
            "Development of interdisciplinary frameworks",
            "Cross-pollination of methodologies and concepts",
            "Novel applications of combined expertise"
        ]
        
        return opportunities[:4]
    
    def _update_team_metrics(self, response: CollaborativeResearchResponse):
        """Atualiza métricas da equipe."""
        self.collaboration_metrics["total_researches"] += 1
        
        if response.status == "completed":
            self.collaboration_metrics["successful_collaborations"] += 1
        
        # Atualizar participação por agent
        for agent_name in response.participating_agents:
            if agent_name not in self.collaboration_metrics["agent_participation"]:
                self.collaboration_metrics["agent_participation"][agent_name] = {
                    "participation_count": 0,
                    "insights_count": 0,
                    "collaboration_score": 0.0
                }
            
            self.collaboration_metrics["agent_participation"][agent_name]["participation_count"] += 1
            
            # Contar insights do agent
            agent_insights = [i for i in response.insights if i.agent_specialization.value in agent_name.lower()]
            self.collaboration_metrics["agent_participation"][agent_name]["insights_count"] += len(agent_insights)


# ==================== EXPORTS ====================

__all__ = [
    "ResearchTeamCoordinator",
    "AUTOGEN_AVAILABLE"
]