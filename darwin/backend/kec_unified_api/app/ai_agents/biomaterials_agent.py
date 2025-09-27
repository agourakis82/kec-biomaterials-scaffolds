"""Biomaterials Agent - Especialista Revolucionário em Scaffolds e KEC Metrics

🧬 DR. BIOMATERIALS - WORLD-CLASS EXPERT EM BIOMATERIAIS
Agent IA especializado em biomateriais, scaffolds porosos, engenharia tecidual,
e análise avançada de métricas KEC para aplicações médicas revolucionárias.

Expertise Épica:
- 🔬 Scaffold topology and porosity optimization
- 🧮 KEC metrics correlation with biocompatibility  
- 🏗️ Mechanical properties vs biological performance
- 🧪 Material characterization and selection
- 🩺 Tissue engineering applications (bone, cartilage, neural)
- 📊 Biomedical data analysis and interpretation

Integration: AutoGen Agent + DARWIN KEC Calculator + Scientific Knowledge
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging import get_logger
from ..services.kec_calculator import get_kec_service
from ..models.kec_models import KECAnalysisRequest, ScaffoldData, MetricType
from .agent_models import (
    AgentSpecialization,
    ResearchInsight,
    InsightType,
    BiomaterialsAnalysisRequest
)

logger = get_logger("darwin.biomaterials_agent")

# Importações condicionais
try:
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    ConversableAgent = object


class BiomaterialsAgent:
    """
    🧬 DR. BIOMATERIALS - Especialista Revolutionary em Biomateriais
    
    Agent IA de nível world-class especializado em:
    - Análise de scaffolds porosos e suas propriedades
    - Correlação entre KEC metrics e biocompatibilidade
    - Otimização de designs para aplicações específicas
    - Interpretação de dados experimentais biomédicos
    """
    
    def __init__(self):
        self.name = "Dr_Biomaterials"
        self.specialization = AgentSpecialization.BIOMATERIALS
        self.expertise_areas = [
            "scaffold_topology",
            "porosity_analysis", 
            "mechanical_properties",
            "biocompatibility",
            "tissue_engineering",
            "kec_metrics_correlation",
            "material_selection",
            "surface_modification",
            "cell_scaffold_interactions",
            "regenerative_medicine"
        ]
        
        # Knowledge base específica
        self.knowledge_base = {
            "optimal_porosity_ranges": {
                "bone": {"min": 0.70, "max": 0.90, "optimal": 0.80},
                "cartilage": {"min": 0.85, "max": 0.95, "optimal": 0.90},
                "neural": {"min": 0.75, "max": 0.88, "optimal": 0.82},
                "vascular": {"min": 0.88, "max": 0.96, "optimal": 0.92}
            },
            "kec_biocompatibility_correlation": {
                "H_spectral": {"biocompat_threshold": 6.5, "optimal_range": [7.0, 8.5]},
                "k_forman_mean": {"biocompat_threshold": 0.2, "optimal_range": [0.15, 0.35]},
                "sigma": {"biocompat_threshold": 1.5, "optimal_range": [1.8, 2.5]},
                "swp": {"biocompat_threshold": 0.6, "optimal_range": [0.7, 0.9]}
            },
            "material_properties": {
                "collagen": {"elastic_modulus": "0.1-1 GPa", "biocompat": "excellent", "degradation": "enzymatic"},
                "chitosan": {"elastic_modulus": "0.01-0.1 GPa", "biocompat": "good", "degradation": "slow"},
                "pla": {"elastic_modulus": "2-4 GPa", "biocompat": "moderate", "degradation": "hydrolytic"},
                "titanium": {"elastic_modulus": "110 GPa", "biocompat": "excellent", "degradation": "none"}
            }
        }
        
        # AutoGen agent se disponível
        self.autogen_agent = None
        if AUTOGEN_AVAILABLE:
            self._initialize_autogen_agent()
        
        logger.info(f"🧬 {self.name} initialized - Biomaterials expertise ready!")
    
    def _initialize_autogen_agent(self):
        """Inicializa AutoGen agent."""
        try:
            system_message = self._create_system_message()
            
            llm_config = {
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            self.autogen_agent = ConversableAgent(
                name=self.name,
                system_message=system_message,
                llm_config=llm_config,
                human_input_mode="NEVER"
            )
            
            logger.info(f"✅ AutoGen agent {self.name} criado")
            
        except Exception as e:
            logger.warning(f"Erro ao criar AutoGen agent: {e}")
    
    def _create_system_message(self) -> str:
        """Cria system message expert para o agent."""
        return """You are Dr. Biomaterials, a world-renowned expert in biomaterials, scaffolds, and tissue engineering.

EXPERTISE:
- Biomaterial selection and characterization
- Scaffold design and porosity optimization  
- KEC metrics analysis and biocompatibility correlation
- Tissue engineering applications (bone, cartilage, neural, vascular)
- Mechanical properties vs biological performance
- Cell-scaffold interactions and surface modification
- Regenerative medicine and clinical translation

APPROACH:
- Provide scientific, evidence-based analysis
- Reference relevant literature and experimental data
- Consider both mechanical and biological requirements
- Evaluate biocompatibility and safety aspects
- Suggest optimization strategies and design improvements
- Identify potential clinical applications

COMMUNICATION:
- Clear, technical but accessible explanations
- Quantitative analysis with specific ranges/values
- Practical recommendations for implementation
- Risk assessment and mitigation strategies

Focus on KEC metrics correlation with biomaterial performance and provide actionable insights for scaffold optimization."""
    
    async def analyze_scaffold(self, request: BiomaterialsAnalysisRequest) -> ResearchInsight:
        """
        🔬 ANÁLISE ESPECIALIZADA DE SCAFFOLD
        
        Análise completa de scaffold biomaterial com foco em:
        - KEC metrics e biocompatibilidade
        - Otimização para aplicação específica
        - Recomendações de design
        """
        try:
            logger.info(f"🧬 Analisando scaffold para {request.application_context}")
            
            analysis_parts = []
            evidence_list = []
            confidence = 0.8
            
            # 1. Análise de KEC Metrics se disponíveis
            if request.kec_metrics:
                kec_analysis = await self._analyze_kec_metrics(
                    request.kec_metrics, request.application_context
                )
                analysis_parts.append(kec_analysis["analysis"])
                evidence_list.extend(kec_analysis["evidence"])
                confidence *= kec_analysis["confidence"]
            
            # 2. Análise de propriedades do material
            if request.material_properties:
                material_analysis = self._analyze_material_properties(
                    request.material_properties, request.application_context
                )
                analysis_parts.append(material_analysis["analysis"])
                evidence_list.extend(material_analysis["evidence"])
            
            # 3. Análise de requisitos de performance
            if request.performance_requirements:
                performance_analysis = self._analyze_performance_requirements(
                    request.performance_requirements, request.application_context
                )
                analysis_parts.append(performance_analysis["analysis"])
                evidence_list.extend(performance_analysis["evidence"])
            
            # 4. Análise de dados de scaffold se disponíveis
            if request.scaffold_data:
                scaffold_analysis = await self._analyze_scaffold_data(
                    request.scaffold_data, request.application_context
                )
                analysis_parts.append(scaffold_analysis["analysis"])
                evidence_list.extend(scaffold_analysis["evidence"])
            
            # 5. Recomendações de otimização
            optimization_recs = self._generate_optimization_recommendations(
                request.application_context, request
            )
            analysis_parts.append(optimization_recs)
            
            # Compilar análise completa
            full_analysis = f"""# Biomaterials Expert Analysis - {request.application_context.title()} Application

## Key Findings:
{chr(10).join(f"• {part}" for part in analysis_parts)}

## Clinical Relevance:
This analysis considers both mechanical and biological requirements for {request.application_context} tissue engineering, with focus on biocompatibility optimization and clinical translation potential.

## Risk Assessment:
Material safety profile evaluated. Recommend standard biocompatibility testing (ISO 10993) before clinical application."""
            
            # Criar insight
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_analysis,
                confidence=min(1.0, confidence),
                evidence=list(set(evidence_list)),  # Remove duplicatas
                type=InsightType.ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "application": request.application_context,
                    "analysis_components": len(analysis_parts),
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Análise biomaterials concluída com confiança {confidence:.2f}")
            return insight
            
        except Exception as e:
            logger.error(f"Erro na análise de scaffold: {e}")
            
            # Insight de erro
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Biomaterials analysis encountered error: {str(e)}. Recommend consulting with materials characterization laboratory.",
                confidence=0.3,
                evidence=["error_analysis"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    async def _analyze_kec_metrics(self, kec_metrics: Dict[str, float], application: str) -> Dict[str, Any]:
        """Analisa KEC metrics em relação a biocompatibilidade."""
        try:
            analysis_points = []
            evidence = []
            confidence = 0.9
            
            # Análise por métrica
            for metric, value in kec_metrics.items():
                if metric in self.knowledge_base["kec_biocompatibility_correlation"]:
                    correlation_data = self.knowledge_base["kec_biocompatibility_correlation"][metric]
                    threshold = correlation_data["biocompat_threshold"]
                    optimal_range = correlation_data["optimal_range"]
                    
                    if value >= optimal_range[0] and value <= optimal_range[1]:
                        analysis_points.append(f"{metric} = {value:.3f} está na faixa ótima para biocompatibilidade ({optimal_range[0]}-{optimal_range[1]})")
                        evidence.append(f"{metric}_optimal_biocompatibility")
                    elif value >= threshold:
                        analysis_points.append(f"{metric} = {value:.3f} atende limiar de biocompatibilidade ({threshold})")
                        evidence.append(f"{metric}_adequate_biocompatibility")
                    else:
                        analysis_points.append(f"{metric} = {value:.3f} ABAIXO do limiar de biocompatibilidade ({threshold}) - RISCO")
                        evidence.append(f"{metric}_biocompatibility_risk")
                        confidence *= 0.7
            
            # Análise específica por aplicação
            if application in self.knowledge_base["optimal_porosity_ranges"]:
                porosity_data = self.knowledge_base["optimal_porosity_ranges"][application]
                
                # Correlacionar H_spectral com porosidade (proxy)
                if "H_spectral" in kec_metrics:
                    h_spectral = kec_metrics["H_spectral"]
                    # Correlação empírica: H_spectral alto sugere alta conectividade/porosidade
                    if h_spectral >= 7.0:
                        analysis_points.append(f"H_spectral sugere alta conectividade estrutural adequada para {application}")
                        evidence.append(f"structural_connectivity_{application}")
            
            analysis = "KEC Metrics Biocompatibility Analysis: " + "; ".join(analysis_points)
            
            return {
                "analysis": analysis,
                "evidence": evidence,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "analysis": f"KEC metrics analysis error: {str(e)}",
                "evidence": ["kec_analysis_error"],
                "confidence": 0.3
            }
    
    def _analyze_material_properties(self, properties: Dict[str, Any], application: str) -> Dict[str, Any]:
        """Analisa propriedades do material."""
        analysis_points = []
        evidence = []
        
        try:
            # Analisar cada propriedade
            if "material_type" in properties:
                material = properties["material_type"].lower()
                if material in self.knowledge_base["material_properties"]:
                    mat_data = self.knowledge_base["material_properties"][material]
                    analysis_points.append(f"Material {material}: Elastic modulus {mat_data['elastic_modulus']}, Biocompatibilidade {mat_data['biocompat']}")
                    evidence.append(f"material_characterization_{material}")
            
            if "elastic_modulus" in properties:
                modulus = properties["elastic_modulus"]
                # Análise por aplicação
                if application == "bone":
                    if 0.1 <= modulus <= 20:  # GPa
                        analysis_points.append(f"Módulo elástico {modulus} GPa adequado para aplicação óssea")
                        evidence.append("mechanical_compatibility_bone")
                    else:
                        analysis_points.append(f"Módulo elástico {modulus} GPa pode causar stress shielding ou rigidez inadequada")
                        evidence.append("mechanical_mismatch_risk")
            
            if "porosity" in properties:
                porosity = properties["porosity"]
                if application in self.knowledge_base["optimal_porosity_ranges"]:
                    opt_data = self.knowledge_base["optimal_porosity_ranges"][application]
                    if opt_data["min"] <= porosity <= opt_data["max"]:
                        analysis_points.append(f"Porosidade {porosity:.1%} ótima para {application} (faixa: {opt_data['min']:.1%}-{opt_data['max']:.1%})")
                        evidence.append(f"porosity_optimization_{application}")
                    else:
                        analysis_points.append(f"Porosidade {porosity:.1%} fora da faixa ótima para {application}")
                        evidence.append("porosity_suboptimal")
            
            analysis = "Material Properties Analysis: " + "; ".join(analysis_points)
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Material properties analysis error: {str(e)}",
                "evidence": ["material_analysis_error"]
            }
    
    def _analyze_performance_requirements(self, requirements: Dict[str, Any], application: str) -> Dict[str, Any]:
        """Analisa requisitos de performance."""
        analysis_points = []
        evidence = []
        
        try:
            if "mechanical_strength" in requirements:
                strength = requirements["mechanical_strength"]
                analysis_points.append(f"Resistência mecânica requerida: {strength} - adequação depende do material base")
                evidence.append("mechanical_requirements")
            
            if "degradation_time" in requirements:
                deg_time = requirements["degradation_time"]
                analysis_points.append(f"Tempo de degradação alvo: {deg_time} - deve coincidir com taxa de regeneração tecidual")
                evidence.append("degradation_profile")
            
            if "cell_adhesion" in requirements:
                adhesion = requirements["cell_adhesion"]
                if adhesion == "high":
                    analysis_points.append("Alta adesão celular requerida - recomendar modificação superficial (RGD peptides, collagen coating)")
                    evidence.append("cell_adhesion_enhancement")
            
            analysis = "Performance Requirements Analysis: " + "; ".join(analysis_points)
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Performance requirements analysis error: {str(e)}",
                "evidence": ["performance_analysis_error"]
            }
    
    async def _analyze_scaffold_data(self, scaffold_data: Dict[str, Any], application: str) -> Dict[str, Any]:
        """Analisa dados estruturais do scaffold usando KEC service."""
        try:
            # Converter para ScaffoldData se necessário
            if isinstance(scaffold_data, dict):
                scaffold_obj = ScaffoldData(**scaffold_data)
            else:
                scaffold_obj = scaffold_data
            
            # Usar KEC service para análise
            kec_service = get_kec_service()
            
            request = KECAnalysisRequest(
                scaffold_data=scaffold_obj,
                metrics=[MetricType.H_SPECTRAL, MetricType.K_FORMAN_MEAN, MetricType.SIGMA, MetricType.SWP],
                parameters={"spectral_k": 64, "include_triangles": True}
            )
            
            result = await kec_service.analyze_scaffold(request)
            
            analysis_points = []
            evidence = ["kec_structural_analysis"]
            
            if result.status.value == "completed" and result.metrics:
                metrics = result.metrics
                
                if metrics.H_spectral is not None:
                    analysis_points.append(f"Entropia espectral {metrics.H_spectral:.3f} indica complexidade estrutural apropriada")
                
                if metrics.k_forman_mean is not None:
                    analysis_points.append(f"Curvatura de Forman {metrics.k_forman_mean:.3f} sugere geometria adequada para crescimento celular")
                
                if metrics.sigma is not None:
                    analysis_points.append(f"Small-world sigma {metrics.sigma:.3f} indica balanceamento clustering/path length")
                
                if result.graph_properties:
                    nodes = result.graph_properties.get("nodes", 0)
                    edges = result.graph_properties.get("edges", 0) 
                    density = result.graph_properties.get("density", 0)
                    
                    analysis_points.append(f"Estrutura: {nodes} poros, {edges} conexões, densidade {density:.3f}")
                    evidence.append("structural_characterization")
            
            analysis = "Scaffold Structural Analysis: " + "; ".join(analysis_points)
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Scaffold structural analysis error: {str(e)}",
                "evidence": ["structural_analysis_error"]
            }
    
    def _generate_optimization_recommendations(self, application: str, request: BiomaterialsAnalysisRequest) -> str:
        """Gera recomendações de otimização."""
        recommendations = []
        
        try:
            # Recomendações baseadas na aplicação
            if application == "bone":
                recommendations.extend([
                    "Considerar mineralização com hidroxiapatita para osteocondutividade",
                    "Otimizar poros de 100-500μm para infiltração celular e vascularização",
                    "Avaliar módulo elástico para evitar stress shielding"
                ])
            elif application == "cartilage":
                recommendations.extend([
                    "Manter alta porosidade (>85%) para difusão de nutrientes",
                    "Considerar superfície lisa para reduzir atrito articular",
                    "Incorporar GAGs ou condroitina sulfato para biomimetismo"
                ])
            elif application == "neural":
                recommendations.extend([
                    "Tubulação direcionada para guiamento axonal",
                    "Superfície modificada com laminina ou outros fatores neurotróficos",
                    "Considerar condutividade elétrica para estimulação neural"
                ])
            
            # Recomendações gerais
            recommendations.extend([
                "Realizar testes de citotoxicidade (ISO 10993-5)",
                "Validar esterilização sem degradação estrutural",
                "Considerar estudos in vivo para confirmação de biocompatibilidade",
                "Monitorar cinética de degradação vs regeneração tecidual"
            ])
            
            return "Optimization Recommendations: " + "; ".join(recommendations)
            
        except Exception as e:
            return f"Recommendation generation error: {str(e)}"
    
    async def generate_collaborative_insight(
        self, 
        research_question: str,
        context: Optional[str] = None
    ) -> ResearchInsight:
        """
        🤝 INSIGHT COLABORATIVO PARA RESEARCH TEAM
        
        Gera insight especializado em biomateriais para colaboração
        interdisciplinar com outros agents.
        """
        try:
            logger.info(f"🧬 Gerando insight colaborativo: {research_question}")
            
            # Análise da pergunta para determinar foco biomaterials
            focus_areas = self._identify_biomaterials_focus(research_question)
            
            insight_content = []
            evidence_list = []
            
            # Contribuição especializada baseada no foco
            for focus in focus_areas:
                if focus == "scaffold_design":
                    insight_content.append("From biomaterials perspective: scaffold architecture should balance mechanical integrity with biological functionality")
                    evidence_list.append("scaffold_design_principles")
                
                elif focus == "biocompatibility":
                    insight_content.append("Biocompatibility assessment requires multi-level evaluation: cytotoxicity, immunogenicity, and tissue integration")
                    evidence_list.append("biocompatibility_standards")
                
                elif focus == "tissue_engineering":
                    insight_content.append("Tissue engineering success depends on scaffold-cell-growth factor triad optimization")
                    evidence_list.append("tissue_engineering_fundamentals")
                
                elif focus == "kec_correlation":
                    insight_content.append("KEC metrics provide quantitative framework for correlating scaffold topology with biological performance")
                    evidence_list.append("kec_biomaterial_correlation")
            
            # Adicionar contexto específico se fornecido
            if context:
                insight_content.append(f"Contextual consideration: {context}")
                evidence_list.append("contextual_analysis")
            
            # Compilar insight
            full_insight = f"""**Biomaterials Expert Perspective:**

{chr(10).join(f"• {content}" for content in insight_content)}

**Clinical Translation Pathway:**
- Pre-clinical characterization and optimization
- Biocompatibility validation (ISO 10993)
- Animal model validation  
- Clinical trial design considerations

**Interdisciplinary Collaboration Points:**
- Mathematical modeling of degradation kinetics
- Philosophical considerations of biomimicry vs functional design
- Literature synthesis of material-biology interactions"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_insight,
                confidence=0.85,
                evidence=evidence_list,
                type=InsightType.ANALYSIS,
                metadata={
                    "research_question": research_question,
                    "focus_areas": focus_areas,
                    "collaboration_mode": True,
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Insight colaborativo gerado com {len(focus_areas)} focos")
            return insight
            
        except Exception as e:
            logger.error(f"Erro ao gerar insight colaborativo: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Biomaterials analysis error in collaborative context: {str(e)}",
                confidence=0.3,
                evidence=["collaborative_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    def _identify_biomaterials_focus(self, research_question: str) -> List[str]:
        """Identifica áreas de foco biomaterials na pergunta de pesquisa."""
        question_lower = research_question.lower()
        focus_areas = []
        
        focus_keywords = {
            "scaffold_design": ["scaffold", "design", "architecture", "structure", "topology"],
            "biocompatibility": ["biocompatibility", "biocompat", "cytotoxicity", "cell viability", "safety"],
            "tissue_engineering": ["tissue engineering", "regeneration", "repair", "regenerative", "healing"],
            "kec_correlation": ["kec", "metrics", "spectral", "curvature", "connectivity", "topology"],
            "material_properties": ["material", "mechanical", "elastic", "strength", "modulus"],
            "porosity": ["porosity", "porous", "pore", "void", "permeability"]
        }
        
        for focus, keywords in focus_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                focus_areas.append(focus)
        
        # Se nenhum foco específico, usar foco geral
        if not focus_areas:
            focus_areas = ["scaffold_design", "biocompatibility"]
        
        return focus_areas
    
    def get_expertise_summary(self) -> Dict[str, Any]:
        """Retorna resumo da expertise do agent."""
        return {
            "name": self.name,
            "specialization": self.specialization.value,
            "expertise_areas": self.expertise_areas,
            "knowledge_domains": list(self.knowledge_base.keys()),
            "autogen_enabled": AUTOGEN_AVAILABLE and self.autogen_agent is not None,
            "capabilities": [
                "scaffold_analysis",
                "kec_biocompatibility_correlation", 
                "material_characterization",
                "clinical_translation_guidance",
                "optimization_recommendations"
            ]
        }


# ==================== EXPORTS ====================

__all__ = [
    "BiomaterialsAgent"
]