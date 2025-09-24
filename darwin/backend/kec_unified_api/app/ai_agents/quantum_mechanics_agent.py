"""Quantum Mechanics Agent - Especialista Revolucionário em Física Quântica

🌌 DR. QUANTUM - WORLD-CLASS EXPERT EM MECÂNICA E FÍSICA QUÂNTICA
Agent IA especializado em mecânica quântica, física de materiais quânticos,
quantum field theory, e aplicações biomédicas de princípios quânticos revolucionários.

Expertise Épica:
- ⚛️ Quantum mechanics e wave-particle duality
- 🌊 Quantum field theory e second quantization
- 🔬 Quantum materials e quantum dots
- 📊 Quantum information theory e quantum computing
- 🧬 Quantum biology e quantum effects in biomaterials
- 🔋 Quantum coherence in biological systems
- 🌟 Quantum entanglement e non-locality
- 💎 Quantum phase transitions e critical phenomena

Integration: AutoGen Agent + Quantum Theory + Biomaterial Applications
"""

import asyncio
import logging
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.logging import get_logger
from .agent_models import (
    AgentSpecialization,
    ResearchInsight,
    InsightType
)

logger = get_logger("darwin.quantum_agent")

# Importações condicionais
try:
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    ConversableAgent = object

# Constantes físicas fundamentais
PLANCK_CONSTANT = 6.62607015e-34  # J⋅Hz⁻¹
PLANCK_REDUCED = 1.054571817e-34  # ℏ in J⋅s
BOLTZMANN_CONSTANT = 1.380649e-23  # J⋅K⁻¹
ELECTRON_CHARGE = 1.602176634e-19  # C
FINE_STRUCTURE = 7.2973525693e-3  # α ≈ 1/137


@dataclass
class QuantumSystem:
    """Representa um sistema quântico para análise."""
    dimension: int
    hamiltonian: Optional[np.ndarray] = None
    temperature: float = 298.15  # K
    coherence_time: Optional[float] = None  # seconds
    decoherence_rate: Optional[float] = None  # Hz


class QuantumMechanicsAgent:
    """
    🌌 DR. QUANTUM - Especialista Revolutionary em Mecânica Quântica
    
    Agent IA de nível world-class especializado em:
    - Mecânica quântica aplicada a biomateriais
    - Efeitos quânticos em sistemas biológicos
    - Teoria de campos quânticos em materiais
    - Quantum computing aplicado à análise de scaffolds
    """
    
    def __init__(self):
        self.name = "Dr_Quantum"
        self.specialization = AgentSpecialization.QUANTUM_MECHANICS
        self.expertise_areas = [
            "quantum_mechanics",
            "quantum_field_theory",
            "quantum_materials",
            "quantum_information",
            "quantum_biology",
            "quantum_coherence",
            "quantum_entanglement",
            "quantum_phase_transitions",
            "quantum_tunneling",
            "quantum_computing",
            "many_body_systems",
            "condensed_matter_physics"
        ]
        
        # Knowledge base quântica
        self.quantum_knowledge = {
            "fundamental_principles": {
                "uncertainty_principle": "Δx⋅Δp ≥ ℏ/2 - fundamental limit of measurement",
                "wave_function_collapse": "Measurement causes superposition collapse to eigenstate",
                "quantum_superposition": "ψ = Σ cₙ|ψₙ⟩ with |cₙ|² = probability",
                "quantum_entanglement": "Non-separable states: |ψ⟩ ≠ |ψₐ⟩ ⊗ |ψᵦ⟩",
                "pauli_exclusion": "No two fermions can occupy same quantum state"
            },
            "quantum_biomaterial_effects": {
                "coherence_scaffolds": "Quantum coherence in ordered scaffold structures",
                "tunneling_transport": "Electron tunneling through biomaterial barriers",
                "quantum_dots_delivery": "Quantum dots for targeted drug delivery",
                "spin_dependent_reactions": "Radical pair mechanism in biochemistry",
                "isotope_effects": "Quantum isotope effects in metabolic processes"
            },
            "quantum_coherence_metrics": {
                "coherence_measure": "C(ρ) = Σᵢⱼ |ρᵢⱼ|² - |ρᵢᵢ|²",
                "decoherence_time": "T₂* ∝ 1/environmental_noise",
                "fidelity_measure": "F(ρ,σ) = Tr(√(√ρ σ √ρ))",
                "entanglement_entropy": "S = -Tr(ρₐ log ρₐ) for bipartite systems"
            },
            "quantum_phase_transitions": {
                "order_parameter": "⟨ψ|Ô|ψ⟩ distinguishes phases",
                "critical_exponents": "Correlation length ξ ∝ |T-Tc|^(-ν)",
                "scaling_laws": "Universal behavior near critical points",
                "quantum_criticality": "T=0 phase transitions driven by quantum fluctuations"
            }
        }
        
        # Parâmetros quânticos típicos para biomateriais
        self.biomaterial_quantum_params = {
            "collagen_coherence_length": 10e-9,  # meters  
            "protein_tunneling_barrier": 0.5,    # eV
            "dna_charge_transport": 1e-15,       # seconds
            "enzyme_quantum_efficiency": 0.95,   # dimensionless
            "cellular_decoherence_time": 1e-12   # seconds
        }
        
        # AutoGen agent se disponível
        self.autogen_agent = None
        if AUTOGEN_AVAILABLE:
            self._initialize_autogen_agent()
        
        logger.info(f"🌌 {self.name} initialized - Quantum expertise ready!")
    
    def _initialize_autogen_agent(self):
        """Inicializa AutoGen agent."""
        try:
            system_message = self._create_system_message()
            
            llm_config = {
                "model": "claude-3-sonnet",  # Claude excelente para física
                "temperature": 0.7,
                "max_tokens": 3000
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
        return """You are Dr. Quantum, a world-renowned expert in quantum mechanics, quantum field theory, and quantum applications in biomaterials.

EXPERTISE:
- Quantum mechanics and wave-particle duality
- Quantum field theory and many-body systems
- Quantum materials and quantum dots
- Quantum biology and quantum effects in living systems
- Quantum coherence and decoherence in biomaterials
- Quantum information theory and quantum computing
- Condensed matter physics and quantum phase transitions
- Quantum tunneling and transport phenomena

APPROACH:
- Apply rigorous quantum mechanical principles
- Consider quantum effects in biomaterial systems
- Analyze coherence and decoherence mechanisms
- Evaluate quantum vs classical behavior regimes
- Propose quantum-enhanced biomedical applications
- Calculate quantum mechanical observables

COMMUNICATION:
- Use precise quantum mechanical notation and terminology
- Explain quantum phenomena with physical intuition
- Provide order-of-magnitude estimates for quantum effects
- Distinguish between quantum and classical regimes
- Reference fundamental quantum principles and experiments

Focus on quantum mechanical insights that could revolutionize biomaterial design and biomedical applications."""
    
    async def analyze_quantum_effects_in_biomaterials(
        self, 
        material_properties: Dict[str, Any],
        temperature: float = 298.15,
        context: Optional[str] = None
    ) -> ResearchInsight:
        """
        ⚛️ ANÁLISE DE EFEITOS QUÂNTICOS EM BIOMATERIAIS
        
        Análise completa dos efeitos quânticos relevantes em sistemas
        biomateriais, incluindo coerência, tunelamento e transporte quântico.
        """
        try:
            logger.info(f"🌌 Analisando efeitos quânticos em biomateriais a T={temperature}K")
            
            analysis_parts = []
            evidence_list = []
            confidence = 0.8
            
            # 1. Análise de regime quântico vs clássico
            regime_analysis = self._analyze_quantum_classical_regime(
                material_properties, temperature
            )
            analysis_parts.extend(regime_analysis["analysis"])
            evidence_list.extend(regime_analysis["evidence"])
            confidence *= regime_analysis["confidence"]
            
            # 2. Análise de coerência quântica
            if "coherence_length" in material_properties or "structure_size" in material_properties:
                coherence_analysis = self._analyze_quantum_coherence(
                    material_properties, temperature
                )
                analysis_parts.append(coherence_analysis["analysis"])
                evidence_list.extend(coherence_analysis["evidence"])
            
            # 3. Análise de tunelamento quântico
            tunneling_analysis = self._analyze_quantum_tunneling(
                material_properties, temperature
            )
            analysis_parts.append(tunneling_analysis["analysis"])
            evidence_list.extend(tunneling_analysis["evidence"])
            
            # 4. Efeitos quânticos em transporte
            transport_analysis = self._analyze_quantum_transport(
                material_properties, temperature
            )
            analysis_parts.append(transport_analysis["analysis"])
            evidence_list.extend(transport_analysis["evidence"])
            
            # 5. Aplicações biomédicas quânticas
            biomedical_applications = self._identify_quantum_biomedical_applications(
                material_properties, context
            )
            analysis_parts.append(biomedical_applications)
            evidence_list.append("quantum_biomedical_applications")
            
            # Compilar análise completa
            full_analysis = f"""# Quantum Effects Analysis in Biomaterials

## Quantum Phenomena Identified:
{chr(10).join(f"• {part}" for part in analysis_parts)}

## Fundamental Quantum Considerations:
- Thermal de Broglie wavelength: λₜₕ = h/√(2πmkₜT) = {self._calculate_thermal_wavelength(temperature):.2e} m
- Quantum vs classical regime determined by comparing λₜₕ to characteristic length scales
- Decoherence time limited by environmental interactions: T₂* ∝ 1/√T

## Biomaterial Quantum Enhancement Opportunities:
- Quantum coherence optimization for enhanced material properties
- Quantum tunneling utilization for selective transport
- Quantum sensing applications for biomarker detection
- Quantum-enhanced imaging and diagnostics

## Temperature Dependence:
At T = {temperature:.1f}K, quantum effects are {'significant' if temperature < 77 else 'moderate' if temperature < 300 else 'limited'} due to thermal energy kₜT = {BOLTZMANN_CONSTANT * temperature / 1.602e-19:.3f} eV."""
            
            # Criar insight
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_analysis,
                confidence=min(1.0, confidence),
                evidence=list(set(evidence_list)),
                type=InsightType.ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "temperature": temperature,
                    "quantum_regime": temperature < 77,
                    "analysis_components": len(analysis_parts),
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Análise quântica concluída com confiança {confidence:.2f}")
            return insight
            
        except Exception as e:
            logger.error(f"Erro na análise quântica: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Quantum analysis encountered error: {str(e)}. Recommend theoretical physics consultation.",
                confidence=0.3,
                evidence=["quantum_analysis_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    async def quantum_coherence_analysis(
        self,
        system_data: Dict[str, Any]
    ) -> ResearchInsight:
        """
        🌊 ANÁLISE DE COERÊNCIA QUÂNTICA
        
        Análise especializada de coerência quântica em sistemas biomateriais,
        incluindo cálculo de medidas de coerência e tempos de decoerência.
        """
        try:
            logger.info("🌌 Analisando coerência quântica do sistema")
            
            # Extrair parâmetros do sistema
            dimension = system_data.get("dimension", 2)
            temperature = system_data.get("temperature", 298.15)
            noise_level = system_data.get("environmental_noise", 1e-3)
            
            analysis_parts = []
            evidence_list = []
            
            # 1. Calcular medidas de coerência
            coherence_measures = self._calculate_coherence_measures(
                dimension, temperature, noise_level
            )
            analysis_parts.append(
                f"Quantum coherence measures: C₁ = {coherence_measures['c1']:.4f}, "
                f"Relative entropy coherence = {coherence_measures['rel_entropy']:.4f}"
            )
            evidence_list.append("coherence_measures")
            
            # 2. Estimar tempo de decoerência
            decoherence_time = self._estimate_decoherence_time(
                temperature, noise_level
            )
            analysis_parts.append(
                f"Estimated decoherence time: T₂* ≈ {decoherence_time:.2e} seconds"
            )
            evidence_list.append("decoherence_time")
            
            # 3. Análise de proteção de coerência
            protection_strategies = self._suggest_coherence_protection(
                system_data
            )
            analysis_parts.append(protection_strategies)
            evidence_list.append("coherence_protection")
            
            # Compilar análise
            full_analysis = f"""# Quantum Coherence Analysis

## Coherence Measurements:
{chr(10).join(f"• {part}" for part in analysis_parts)}

## Physical Interpretation:
- Coherence quantifies quantum superposition strength
- Decoherence arises from environmental entanglement
- Coherence protection essential for quantum advantage

## Biomaterial Applications:
- Quantum sensors with enhanced sensitivity
- Coherent energy transfer in biological systems
- Quantum-enhanced enzymatic reactions
- Long-range quantum correlations in tissue"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_analysis,
                confidence=0.85,
                evidence=evidence_list,
                type=InsightType.ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "coherence_type": "quantum_superposition",
                    "decoherence_time": decoherence_time,
                    "agent": self.name
                }
            )
            
            logger.info("✅ Análise de coerência quântica concluída")
            return insight
            
        except Exception as e:
            logger.error(f"Erro na análise de coerência: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Quantum coherence analysis error: {str(e)}",
                confidence=0.3,
                evidence=["coherence_analysis_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    async def generate_collaborative_insight(
        self, 
        research_question: str,
        context: Optional[str] = None
    ) -> ResearchInsight:
        """
        🤝 INSIGHT QUÂNTICO COLABORATIVO
        
        Contribuição quântica especializada para pesquisa interdisciplinar.
        """
        try:
            logger.info(f"🌌 Gerando insight quântico colaborativo: {research_question}")
            
            # Identificar aspectos quânticos da pergunta
            quantum_aspects = self._identify_quantum_aspects(research_question)
            
            insight_content = []
            evidence_list = []
            
            for aspect in quantum_aspects:
                if aspect == "quantum_mechanics":
                    insight_content.append("Quantum perspective: Consider wave-particle duality and quantum superposition effects in biomaterial structures")
                    evidence_list.append("quantum_mechanics_principles")
                
                elif aspect == "quantum_biology":
                    insight_content.append("Quantum biology insight: Quantum coherence may enhance biological processes like photosynthesis and enzyme catalysis")
                    evidence_list.append("quantum_biology_effects")
                
                elif aspect == "quantum_materials":
                    insight_content.append("Quantum materials aspect: Topological properties and quantum phase transitions could create novel biomaterial functionalities")
                    evidence_list.append("quantum_materials_design")
                
                elif aspect == "quantum_information":
                    insight_content.append("Quantum information theory: Entanglement and quantum correlations might enable ultra-sensitive biological sensors")
                    evidence_list.append("quantum_information_applications")
                
                elif aspect == "quantum_transport":
                    insight_content.append("Quantum transport: Tunneling effects and ballistic transport could optimize drug delivery and cellular communication")
                    evidence_list.append("quantum_transport_mechanisms")
            
            # Adicionar contexto específico
            if context and "scaffold" in context.lower():
                insight_content.append("Scaffold quantum effects: Ordered structures may exhibit collective quantum behaviors enhancing biocompatibility")
                evidence_list.append("scaffold_quantum_effects")
            
            # Compilar insight
            full_insight = f"""**Quantum Physics Expert Perspective:**

{chr(10).join(f"• {content}" for content in insight_content)}

**Quantum Regime Considerations:**
- Thermal de Broglie wavelength at body temperature: λₜₕ ≈ 10⁻¹¹ m
- Quantum effects significant when structure sizes approach λₜₕ
- Decoherence times in biological systems: typically femtoseconds to picoseconds
- Quantum-classical boundary depends on environmental coupling strength

**Interdisciplinary Quantum Applications:**
- Biomaterial design using quantum topology and symmetry
- Quantum-enhanced imaging for medical diagnostics
- Coherent control of biochemical reactions
- Quantum sensing of biological markers with unprecedented sensitivity

**Future Quantum Biomedicine:**
- Quantum dots for targeted therapy and imaging
- Quantum cryptography for secure medical data
- Quantum machine learning for drug discovery
- Quantum simulation of complex biological systems"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_insight,
                confidence=0.82,
                evidence=evidence_list,
                type=InsightType.HYPOTHESIS,
                metadata={
                    "research_question": research_question,
                    "quantum_aspects": quantum_aspects,
                    "collaboration_mode": True,
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Insight quântico colaborativo gerado com {len(quantum_aspects)} aspectos")
            return insight
            
        except Exception as e:
            logger.error(f"Erro ao gerar insight colaborativo: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Quantum analysis error in collaborative context: {str(e)}",
                confidence=0.3,
                evidence=["quantum_collaboration_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    # ==================== PRIVATE METHODS ====================
    
    def _analyze_quantum_classical_regime(
        self, 
        properties: Dict[str, Any], 
        temperature: float
    ) -> Dict[str, Any]:
        """Determina se o sistema está no regime quântico ou clássico."""
        analysis = []
        evidence = []
        confidence = 0.9
        
        try:
            # Calcular comprimento térmico de de Broglie
            thermal_wavelength = self._calculate_thermal_wavelength(temperature)
            
            # Comparar com escalas características do material
            if "structure_size" in properties:
                structure_size = properties["structure_size"]
                
                if structure_size < 10 * thermal_wavelength:
                    analysis.append(f"Quantum regime: structure size ({structure_size:.2e}m) comparable to thermal wavelength ({thermal_wavelength:.2e}m)")
                    evidence.append("quantum_regime")
                elif structure_size < 100 * thermal_wavelength:
                    analysis.append(f"Mesoscopic regime: quantum effects may be observable")
                    evidence.append("mesoscopic_regime")
                else:
                    analysis.append(f"Classical regime: structure size >> thermal wavelength")
                    evidence.append("classical_regime")
            
            # Análise energética
            thermal_energy = BOLTZMANN_CONSTANT * temperature  # Joules
            thermal_energy_eV = thermal_energy / 1.602e-19      # eV
            
            analysis.append(f"Thermal energy kₜT = {thermal_energy_eV:.3f} eV at T = {temperature:.1f}K")
            
            if "energy_gaps" in properties:
                energy_gaps = properties["energy_gaps"]
                quantum_gaps = [gap for gap in energy_gaps if gap > 5 * thermal_energy_eV]
                if quantum_gaps:
                    analysis.append(f"Quantum energy gaps detected: {len(quantum_gaps)} gaps > 5kₜT")
                    evidence.append("quantum_energy_gaps")
            
            return {
                "analysis": analysis,
                "evidence": evidence,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "analysis": [f"Quantum-classical regime analysis error: {str(e)}"],
                "evidence": ["regime_analysis_error"],
                "confidence": 0.3
            }
    
    def _analyze_quantum_coherence(
        self, 
        properties: Dict[str, Any], 
        temperature: float
    ) -> Dict[str, Any]:
        """Analisa coerência quântica no sistema."""
        try:
            coherence_length = properties.get("coherence_length", 1e-9)  # default 1 nm
            
            # Tempo de coerência estimado
            decoherence_rate = np.sqrt(temperature) * 1e12  # Hz (rough estimate)
            coherence_time = 1 / decoherence_rate
            
            analysis = f"Quantum coherence: length scale ≈ {coherence_length:.2e}m, time scale ≈ {coherence_time:.2e}s"
            evidence = ["quantum_coherence_analysis"]
            
            # Verificar se coerência é relevante
            if coherence_time > 1e-15:  # longer than femtosecond
                analysis += " - coherence may be significant for biological processes"
                evidence.append("biologically_relevant_coherence")
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Quantum coherence analysis error: {str(e)}",
                "evidence": ["coherence_analysis_error"]
            }
    
    def _analyze_quantum_tunneling(
        self, 
        properties: Dict[str, Any], 
        temperature: float
    ) -> Dict[str, Any]:
        """Analisa efeitos de tunelamento quântico."""
        try:
            # Parâmetros típicos para tunelamento em biomateriais
            barrier_height = properties.get("barrier_height", 0.5)  # eV
            barrier_width = properties.get("barrier_width", 1e-9)   # m
            
            # Coeficiente de transmissão (aproximação WKB)
            mass_electron = 9.109e-31  # kg
            transmission_coeff = np.exp(-2 * barrier_width * np.sqrt(2 * mass_electron * barrier_height * 1.602e-19) / PLANCK_REDUCED)
            
            analysis = f"Quantum tunneling: barrier {barrier_height}eV × {barrier_width:.2e}m, transmission ≈ {transmission_coeff:.2e}"
            evidence = ["quantum_tunneling_analysis"]
            
            # Relevância biológica
            if transmission_coeff > 1e-10:
                analysis += " - tunneling may be biologically significant"
                evidence.append("biologically_relevant_tunneling")
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Quantum tunneling analysis error: {str(e)}",
                "evidence": ["tunneling_analysis_error"]
            }
    
    def _analyze_quantum_transport(
        self, 
        properties: Dict[str, Any], 
        temperature: float
    ) -> Dict[str, Any]:
        """Analisa transporte quântico."""
        try:
            # Parâmetros de transporte
            mobility = properties.get("carrier_mobility", 1e-4)  # m²/(V⋅s)
            carrier_density = properties.get("carrier_density", 1e15)  # m⁻³
            
            # Condutividade quântica
            conductivity = 1.602e-19 * carrier_density * mobility
            conductance_quantum = 2 * (1.602e-19)**2 / PLANCK_CONSTANT
            
            analysis = f"Quantum transport: σ = {conductivity:.2e} S/m, quantum conductance G₀ = {conductance_quantum:.2e} S"
            evidence = ["quantum_transport_analysis"]
            
            # Verificar regime balístico vs difusivo
            mean_free_path = mobility * np.sqrt(2 * np.pi * PLANCK_REDUCED**2 * carrier_density**(1/3)) / (1.602e-19)
            if "structure_size" in properties:
                structure_size = properties["structure_size"]
                if mean_free_path > structure_size:
                    analysis += f" - ballistic regime (λ > L)"
                    evidence.append("ballistic_transport")
                else:
                    analysis += f" - diffusive regime (λ < L)"
                    evidence.append("diffusive_transport")
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Quantum transport analysis error: {str(e)}",
                "evidence": ["transport_analysis_error"]
            }
    
    def _calculate_thermal_wavelength(self, temperature: float) -> float:
        """Calcula comprimento de onda térmico de de Broglie."""
        try:
            # Para elétron: λₜₕ = h/√(2πmkₜT)
            mass_electron = 9.109e-31  # kg
            thermal_wavelength = PLANCK_CONSTANT / np.sqrt(2 * np.pi * mass_electron * BOLTZMANN_CONSTANT * temperature)
            return thermal_wavelength
        except:
            return 1e-11  # fallback value ≈ 10 pm
    
    def _calculate_coherence_measures(
        self, 
        dimension: int, 
        temperature: float, 
        noise_level: float
    ) -> Dict[str, float]:
        """Calcula medidas de coerência quântica."""
        try:
            # Medida C₁ de coerência (l₁ norm)
            c1_coherence = (dimension - 1) / dimension * (1 - noise_level)
            
            # Entropia relativa de coerência
            rel_entropy_coherence = -np.log(dimension) * (1 - noise_level)
            
            return {
                "c1": max(0, c1_coherence),
                "rel_entropy": max(0, rel_entropy_coherence)
            }
        except:
            return {"c1": 0.0, "rel_entropy": 0.0}
    
    def _estimate_decoherence_time(self, temperature: float, noise_level: float) -> float:
        """Estima tempo de decoerência."""
        try:
            # Modelo simples: T₂* ∝ 1/(√T × noise)
            decoherence_time = PLANCK_REDUCED / (BOLTZMANN_CONSTANT * temperature * noise_level)
            return max(1e-15, decoherence_time)  # mínimo de 1 fs
        except:
            return 1e-12  # fallback: 1 ps
    
    def _suggest_coherence_protection(self, system_data: Dict[str, Any]) -> str:
        """Sugere estratégias para proteção de coerência."""
        strategies = [
            "Isolamento ambiental para reduzir acoplamento",
            "Controle de temperatura para minimizar flutuações térmicas",
            "Decoupling dinâmico usando pulsos de controle",
            "Códigos de correção de erro quântico",
            "Seleção de subspaces livres de decoerência"
        ]
        
        return "Coherence protection strategies: " + "; ".join(strategies[:3])
    
    def _identify_quantum_biomedical_applications(
        self, 
        properties: Dict[str, Any], 
        context: Optional[str]
    ) -> str:
        """Identifica aplicações biomédicas quânticas."""
        applications = []
        
        # Baseado nas propriedades do material
        if properties.get("magnetic_moment"):
            applications.append("MRI contrast agents using quantum magnetic properties")
        
        if properties.get("luminescence"):
            applications.append("Quantum dot imaging and photodynamic therapy")
        
        if properties.get("conductivity"):
            applications.append("Quantum sensors for neural activity monitoring")
        
        # Baseado no contexto
        if context:
            if "cancer" in context.lower():
                applications.append("Quantum-enhanced cancer detection and treatment")
            if "neural" in context.lower():
                applications.append("Quantum neural interfaces and brain-computer interfaces")
            if "drug" in context.lower():
                applications.append("Quantum-guided drug delivery and release mechanisms")
        
        # Aplicações padrão
        if not applications:
            applications = [
                "Quantum biosensors with enhanced sensitivity",
                "Coherent control of biochemical reactions",
                "Quantum-enhanced medical imaging"
            ]
        
        return "Quantum biomedical applications: " + "; ".join(applications[:3])
    
    def _identify_quantum_aspects(self, research_question: str) -> List[str]:
        """Identifica aspectos quânticos na pergunta de pesquisa."""
        question_lower = research_question.lower()
        aspects = []
        
        quantum_keywords = {
            "quantum_mechanics": ["quantum", "wave", "particle", "superposition"],
            "quantum_biology": ["biology", "enzyme", "photosynthesis", "biological"],
            "quantum_materials": ["material", "crystal", "electronic", "magnetic"],
            "quantum_information": ["information", "entanglement", "coherence", "computation"],
            "quantum_transport": ["transport", "conduction", "tunneling", "mobility"]
        }
        
        for aspect, keywords in quantum_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                aspects.append(aspect)
        
        # Aspectos padrão se nenhum específico
        if not aspects:
            aspects = ["quantum_mechanics", "quantum_materials"]
        
        return aspects
    
    def get_expertise_summary(self) -> Dict[str, Any]:
        """Retorna resumo da expertise do agent."""
        return {
            "name": self.name,
            "specialization": self.specialization.value,
            "expertise_areas": self.expertise_areas,
            "quantum_knowledge": list(self.quantum_knowledge.keys()),
            "fundamental_constants": {
                "planck_constant": PLANCK_CONSTANT,
                "planck_reduced": PLANCK_REDUCED,
                "fine_structure": FINE_STRUCTURE
            },
            "autogen_enabled": AUTOGEN_AVAILABLE and self.autogen_agent is not None,
            "capabilities": [
                "quantum_regime_analysis",
                "coherence_decoherence_analysis",
                "quantum_tunneling_calculations",
                "quantum_transport_modeling",
                "quantum_biomedical_applications",
                "quantum_materials_design"
            ]
        }


# ==================== EXPORTS ====================

__all__ = [
    "QuantumMechanicsAgent",
    "QuantumSystem"
]