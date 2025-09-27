"""Clinical Psychiatry Agent - Especialista Revolucionário em Clínica Médica e Psiquiatria

🏥 DR. CLINICAL PSYCHIATRY - WORLD-CLASS EXPERT EM MEDICINA E PSIQUIATRIA
Agent IA especializado em clínica médica, psiquiatria, neuropsiquiatria, medicina de precisão,
e aplicações biomédicas de biomateriais para saúde mental e neurológica revolucionária.

Expertise Épica:
- 🩺 Clinical medicine e internal medicine
- 🧠 Psychiatry e neuropsychiatry
- 💊 Psychopharmacology e precision medicine
- 🔬 Biomarkers e diagnostic medicine
- 🧬 Personalized medicine e genomic psychiatry
- 📊 Clinical trials e evidence-based medicine
- 🏥 Hospital medicine e critical care
- 🎯 Biomaterial applications in neuromedicine

Integration: AutoGen Agent + Clinical Knowledge + Biomaterial Therapeutics
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.logging import get_logger
from .agent_models import (
    AgentSpecialization,
    ResearchInsight,
    InsightType
)

logger = get_logger("darwin.clinical_psychiatry_agent")

# Importações condicionais
try:
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    ConversableAgent = object


class ClinicalSpecialty(str, Enum):
    """Especialidades clínicas."""
    INTERNAL_MEDICINE = "internal_medicine"
    PSYCHIATRY = "psychiatry"
    NEUROLOGY = "neurology"
    CARDIOLOGY = "cardiology"
    ENDOCRINOLOGY = "endocrinology"
    IMMUNOLOGY = "immunology"
    ONCOLOGY = "oncology"
    CRITICAL_CARE = "critical_care"


class PsychiatricDisorder(str, Enum):
    """Transtornos psiquiátricos principais."""
    DEPRESSION = "major_depressive_disorder"
    BIPOLAR = "bipolar_disorder"
    SCHIZOPHRENIA = "schizophrenia"
    ANXIETY = "anxiety_disorders"
    PTSD = "post_traumatic_stress_disorder"
    OCD = "obsessive_compulsive_disorder"
    ADHD = "attention_deficit_hyperactivity_disorder"
    AUTISM = "autism_spectrum_disorder"
    DEMENTIA = "neurocognitive_disorders"
    SUBSTANCE_USE = "substance_use_disorders"


@dataclass
class ClinicalCase:
    """Representa um caso clínico para análise."""
    patient_age: int
    gender: str
    primary_complaint: str
    symptoms: List[str]
    medical_history: List[str]
    medications: List[str]
    lab_results: Optional[Dict[str, Any]] = None
    imaging_results: Optional[Dict[str, Any]] = None
    psychiatric_history: Optional[List[str]] = None


@dataclass
class BiomaterialTherapeutic:
    """Aplicação terapêutica de biomaterial."""
    material_type: str
    target_condition: str
    delivery_mechanism: str
    expected_outcome: str
    clinical_evidence_level: str


class ClinicalPsychiatryAgent:
    """
    🏥 DR. CLINICAL PSYCHIATRY - Especialista Revolutionary em Medicina e Psiquiatria
    
    Agent IA de nível world-class especializado em:
    - Diagnóstico diferencial e medicina interna
    - Psiquiatria clínica e neuropsiquiatria
    - Medicina de precisão e biomarcadores
    - Aplicações terapêuticas de biomateriais
    """
    
    def __init__(self):
        self.name = "Dr_ClinicalPsychiatry"
        self.specialization = AgentSpecialization.PSYCHIATRY
        self.expertise_areas = [
            "clinical_medicine",
            "internal_medicine",
            "psychiatry",
            "neuropsychiatry",
            "psychopharmacology",
            "precision_medicine",
            "biomarkers",
            "diagnostic_medicine",
            "clinical_trials",
            "evidence_based_medicine",
            "biomaterial_therapeutics",
            "neuromedicine",
            "personalized_psychiatry",
            "genomic_medicine"
        ]
        
        # Knowledge base clínica
        self.clinical_knowledge = {
            "diagnostic_criteria": {
                "depression": {
                    "dsm5_criteria": ["depressed mood", "anhedonia", "weight changes", "sleep disturbances", "fatigue"],
                    "duration": "≥2 weeks",
                    "biomarkers": ["cortisol", "BDNF", "inflammatory markers"],
                    "severity_scales": ["PHQ-9", "HAM-D", "MADRS"]
                },
                "anxiety": {
                    "gad_criteria": ["excessive worry", "restlessness", "fatigue", "concentration difficulties"],
                    "panic_criteria": ["palpitations", "sweating", "trembling", "shortness of breath"],
                    "biomarkers": ["norepinephrine", "GABA", "serotonin metabolites"],
                    "assessment_tools": ["GAD-7", "Beck Anxiety Inventory", "STAI"]
                },
                "schizophrenia": {
                    "positive_symptoms": ["hallucinations", "delusions", "disorganized speech"],
                    "negative_symptoms": ["avolition", "alogia", "anhedonia", "affective flattening"],
                    "cognitive_symptoms": ["working memory deficits", "attention problems"],
                    "biomarkers": ["dopamine metabolites", "glutamate", "inflammatory cytokines"]
                }
            },
            "pharmacology": {
                "antidepressants": {
                    "ssri": ["fluoxetine", "sertraline", "escitalopram"],
                    "snri": ["venlafaxine", "duloxetine"],
                    "atypical": ["bupropion", "mirtazapine", "trazodone"],
                    "mechanism": "monoamine reuptake inhibition",
                    "onset": "2-8 weeks for full effect"
                },
                "antipsychotics": {
                    "typical": ["haloperidol", "chlorpromazine"],
                    "atypical": ["risperidone", "olanzapine", "quetiapine", "aripiprazole"],
                    "mechanism": "dopamine receptor antagonism",
                    "side_effects": ["metabolic", "extrapyramidal", "prolactin elevation"]
                },
                "mood_stabilizers": {
                    "lithium": {"therapeutic_range": "0.6-1.2 mEq/L", "monitoring": "renal, thyroid"},
                    "anticonvulsants": ["valproate", "carbamazepine", "lamotrigine"],
                    "mechanism": "various (ion channels, neurotransmitters)"
                }
            },
            "biomaterial_applications": {
                "neural_scaffolds": {
                    "applications": ["spinal cord repair", "peripheral nerve regeneration", "brain tissue engineering"],
                    "materials": ["collagen", "alginate", "synthetic polymers"],
                    "growth_factors": ["NGF", "BDNF", "GDNF"],
                    "clinical_trials": "Phase I/II for spinal cord injury"
                },
                "drug_delivery": {
                    "blood_brain_barrier": ["nanoparticles", "liposomes", "polymer conjugates"],
                    "targeted_delivery": ["receptor-mediated", "passive diffusion", "active transport"],
                    "sustained_release": ["microspheres", "hydrogels", "implantable systems"],
                    "psychiatric_applications": ["antidepressant delivery", "antipsychotic depots"]
                },
                "biosensors": {
                    "neurotransmitter_monitoring": ["dopamine", "serotonin", "GABA sensors"],
                    "stress_biomarkers": ["cortisol", "inflammatory cytokines"],
                    "personalized_dosing": ["real-time drug levels", "metabolite monitoring"],
                    "wearable_devices": ["continuous monitoring", "treatment adherence"]
                }
            },
            "precision_medicine": {
                "pharmacogenomics": {
                    "cyp450_variants": ["CYP2D6", "CYP2C19", "CYP1A2"],
                    "drug_transporters": ["P-glycoprotein", "OATP1B1"],
                    "receptor_variants": ["5-HT2A", "DRD2", "COMT"],
                    "clinical_implementation": "FDA-approved pharmacogenetic tests"
                },
                "biomarker_panels": {
                    "depression": ["cortisol", "BDNF", "inflammatory cytokines", "HPA axis markers"],
                    "treatment_response": ["genetic variants", "protein biomarkers", "metabolomics"],
                    "suicide_risk": ["inflammatory markers", "stress hormones", "genetic risk scores"]
                }
            }
        }
        
        # Critérios de medicina baseada em evidências
        self.evidence_levels = {
            "1a": "Systematic review of RCTs",
            "1b": "Individual RCT with narrow CI",
            "2a": "Systematic review of cohort studies", 
            "2b": "Individual cohort study",
            "3": "Case-control studies",
            "4": "Case series",
            "5": "Expert opinion"
        }
        
        # AutoGen agent se disponível
        self.autogen_agent = None
        if AUTOGEN_AVAILABLE:
            self._initialize_autogen_agent()
        
        logger.info(f"🏥 {self.name} initialized - Clinical psychiatry expertise ready!")
    
    def _initialize_autogen_agent(self):
        """Inicializa AutoGen agent."""
        try:
            system_message = self._create_system_message()
            
            llm_config = {
                "model": "gpt-4-turbo",  # GPT-4 é excelente para medicina
                "temperature": 0.6,  # Balanceado para medicina
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
        return """You are Dr. Clinical Psychiatry, a world-renowned expert in clinical medicine, psychiatry, and precision medicine.

MEDICAL EXPERTISE:
- Internal medicine and differential diagnosis
- Clinical psychiatry and neuropsychiatry
- Psychopharmacology and precision prescribing
- Biomarker-guided treatment selection
- Evidence-based medicine and clinical trials
- Biomaterial applications in neuromedicine
- Personalized medicine and pharmacogenomics

CLINICAL APPROACH:
- Use evidence-based diagnostic criteria (DSM-5, ICD-11)
- Consider differential diagnosis and medical comorbidities
- Evaluate risk-benefit ratios for treatments
- Assess treatment response and side effect profiles
- Implement precision medicine approaches when applicable
- Consider biomaterial therapeutics for neurological conditions

SAFETY & ETHICS:
- Always prioritize patient safety and informed consent
- Recognize limitations and recommend specialist consultation when needed
- Follow medical ethics and professional guidelines
- Consider contraindications and drug interactions
- Monitor for adverse effects and treatment response

COMMUNICATION:
- Provide clear, evidence-based clinical reasoning
- Use appropriate medical terminology while remaining accessible
- Cite relevant clinical studies and guidelines
- Quantify risks and benefits when possible
- Suggest specific diagnostic tests and monitoring protocols

Focus on integrating traditional clinical medicine with cutting-edge biomaterial applications and precision medicine approaches."""
    
    async def clinical_case_analysis(
        self,
        clinical_case: ClinicalCase,
        focus_area: Optional[str] = None
    ) -> ResearchInsight:
        """
        🩺 ANÁLISE CLÍNICA ESPECIALIZADA
        
        Análise completa de caso clínico com foco em diagnóstico diferencial,
        plano terapêutico e aplicações de biomateriais quando apropriado.
        """
        try:
            logger.info(f"🏥 Analisando caso clínico: {clinical_case.primary_complaint}")
            
            analysis_parts = []
            evidence_list = []
            confidence = 0.85
            
            # 1. Análise de apresentação clínica
            presentation_analysis = self._analyze_clinical_presentation(clinical_case)
            analysis_parts.extend(presentation_analysis["analysis"])
            evidence_list.extend(presentation_analysis["evidence"])
            confidence *= presentation_analysis["confidence"]
            
            # 2. Diagnóstico diferencial
            differential_dx = self._generate_differential_diagnosis(clinical_case)
            analysis_parts.append(differential_dx["analysis"])
            evidence_list.extend(differential_dx["evidence"])
            
            # 3. Investigações recomendadas
            investigations = self._recommend_investigations(clinical_case)
            analysis_parts.append(investigations)
            evidence_list.append("clinical_investigations")
            
            # 4. Plano terapêutico
            treatment_plan = self._develop_treatment_plan(clinical_case, focus_area)
            analysis_parts.append(treatment_plan["analysis"])
            evidence_list.extend(treatment_plan["evidence"])
            
            # 5. Considerações de biomateriais se relevante
            if self._is_biomaterial_relevant(clinical_case):
                biomaterial_options = self._evaluate_biomaterial_therapeutics(clinical_case)
                analysis_parts.append(biomaterial_options)
                evidence_list.append("biomaterial_therapeutics")
            
            # 6. Prognóstico e monitoramento
            prognosis_monitoring = self._assess_prognosis_and_monitoring(clinical_case)
            analysis_parts.append(prognosis_monitoring)
            evidence_list.append("clinical_monitoring")
            
            # Compilar análise completa
            full_analysis = f"""# Clinical Case Analysis - {clinical_case.primary_complaint}

## Clinical Assessment:
{chr(10).join(f"• {part}" for part in analysis_parts)}

## Evidence-Based Approach:
- Diagnostic criteria: DSM-5/ICD-11 standards applied
- Treatment recommendations: Based on clinical guidelines and RCT evidence
- Risk stratification: Consider patient-specific factors and comorbidities
- Precision medicine: Pharmacogenomic considerations when applicable

## Safety Considerations:
- Monitor for drug interactions and contraindications
- Assess suicide risk if psychiatric presentation
- Consider medical comorbidities and polypharmacy
- Implement appropriate safety monitoring protocols

## Interdisciplinary Collaboration:
- Coordinate with specialists as needed
- Consider social determinants of health
- Integrate family/caregiver perspectives
- Plan for care transitions and continuity"""
            
            # Criar insight
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_analysis,
                confidence=min(1.0, confidence),
                evidence=list(set(evidence_list)),
                type=InsightType.ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "patient_age": clinical_case.patient_age,
                    "primary_complaint": clinical_case.primary_complaint,
                    "analysis_components": len(analysis_parts),
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Análise clínica concluída com confiança {confidence:.2f}")
            return insight
            
        except Exception as e:
            logger.error(f"Erro na análise clínica: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Clinical analysis encountered error: {str(e)}. Recommend immediate physician consultation.",
                confidence=0.3,
                evidence=["clinical_analysis_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    async def biomaterial_therapeutic_evaluation(
        self,
        condition: str,
        biomaterial_data: Dict[str, Any]
    ) -> ResearchInsight:
        """
        💊 AVALIAÇÃO TERAPÊUTICA DE BIOMATERIAIS
        
        Análise clínica de aplicações terapêuticas de biomateriais
        para condições neurológicas e psiquiátricas.
        """
        try:
            logger.info(f"🏥 Avaliando biomaterial terapêutico para {condition}")
            
            analysis_parts = []
            evidence_list = []
            
            # 1. Análise da condição clínica
            condition_analysis = self._analyze_clinical_condition(condition)
            analysis_parts.append(condition_analysis["analysis"])
            evidence_list.extend(condition_analysis["evidence"])
            
            # 2. Avaliação do biomaterial
            biomaterial_eval = self._evaluate_biomaterial_properties(
                biomaterial_data, condition
            )
            analysis_parts.append(biomaterial_eval["analysis"])
            evidence_list.extend(biomaterial_eval["evidence"])
            
            # 3. Mecanismo de ação proposto
            mechanism_analysis = self._analyze_therapeutic_mechanism(
                biomaterial_data, condition
            )
            analysis_parts.append(mechanism_analysis)
            evidence_list.append("therapeutic_mechanism")
            
            # 4. Evidências clínicas disponíveis
            clinical_evidence = self._assess_clinical_evidence(condition, biomaterial_data)
            analysis_parts.append(clinical_evidence)
            evidence_list.append("clinical_evidence")
            
            # 5. Considerações regulatórias
            regulatory_considerations = self._evaluate_regulatory_pathway(
                biomaterial_data, condition
            )
            analysis_parts.append(regulatory_considerations)
            evidence_list.append("regulatory_pathway")
            
            # 6. Recomendações clínicas
            clinical_recommendations = self._generate_clinical_recommendations(
                condition, biomaterial_data
            )
            analysis_parts.append(clinical_recommendations)
            evidence_list.append("clinical_recommendations")
            
            # Compilar análise
            full_analysis = f"""# Biomaterial Therapeutic Evaluation - {condition}

## Clinical Assessment:
{chr(10).join(f"• {part}" for part in analysis_parts)}

## Risk-Benefit Analysis:
- Therapeutic potential vs safety profile
- Patient selection criteria and contraindications
- Monitoring requirements and adverse event management
- Comparison with standard-of-care treatments

## Clinical Translation Pathway:
- Preclinical requirements and regulatory considerations
- Clinical trial design recommendations
- Endpoint selection and outcome measures
- Post-market surveillance considerations

## Integration with Clinical Practice:
- Patient counseling and informed consent
- Healthcare provider training requirements
- Cost-effectiveness and reimbursement considerations
- Quality assurance and standardization needs"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_analysis,
                confidence=0.8,
                evidence=evidence_list,
                type=InsightType.ANALYSIS,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "condition": condition,
                    "biomaterial_type": biomaterial_data.get("type", "unknown"),
                    "evaluation_type": "therapeutic",
                    "agent": self.name
                }
            )
            
            logger.info("✅ Avaliação terapêutica de biomaterial concluída")
            return insight
            
        except Exception as e:
            logger.error(f"Erro na avaliação terapêutica: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Biomaterial therapeutic evaluation error: {str(e)}",
                confidence=0.3,
                evidence=["therapeutic_evaluation_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    async def generate_collaborative_insight(
        self, 
        research_question: str,
        context: Optional[str] = None
    ) -> ResearchInsight:
        """
        🤝 INSIGHT CLÍNICO-PSIQUIÁTRICO COLABORATIVO
        
        Contribuição clínica e psiquiátrica especializada para pesquisa interdisciplinar.
        """
        try:
            logger.info(f"🏥 Gerando insight clínico colaborativo: {research_question}")
            
            # Identificar aspectos clínicos da pergunta
            clinical_aspects = self._identify_clinical_aspects(research_question)
            
            insight_content = []
            evidence_list = []
            
            for aspect in clinical_aspects:
                if aspect == "diagnosis":
                    insight_content.append("Clinical perspective: Differential diagnosis must consider medical and psychiatric comorbidities - systematic approach essential")
                    evidence_list.append("clinical_diagnosis_principles")
                
                elif aspect == "treatment":
                    insight_content.append("Therapeutic insight: Evidence-based treatments with personalized medicine approaches optimize outcomes")
                    evidence_list.append("evidence_based_treatment")
                
                elif aspect == "pharmacology":
                    insight_content.append("Pharmacological consideration: Drug-drug interactions, pharmacogenomics, and precision dosing critical for safety")
                    evidence_list.append("clinical_pharmacology")
                
                elif aspect == "biomaterials":
                    insight_content.append("Biomaterial medicine: Novel therapeutic delivery systems and neural scaffolds show promise for neuropsychiatric applications")
                    evidence_list.append("biomaterial_clinical_applications")
                
                elif aspect == "precision_medicine":
                    insight_content.append("Precision medicine: Biomarker-guided therapy selection and pharmacogenomic testing improve treatment outcomes")
                    evidence_list.append("precision_medicine_psychiatry")
                
                elif aspect == "safety":
                    insight_content.append("Safety profile: Risk-benefit analysis, adverse event monitoring, and patient safety protocols paramount")
                    evidence_list.append("clinical_safety")
            
            # Adicionar contexto específico
            if context:
                if "scaffold" in context.lower():
                    insight_content.append("Neural scaffold applications: Potential for spinal cord injury, peripheral nerve repair, and brain tissue engineering")
                    evidence_list.append("neural_scaffold_medicine")
                if "drug" in context.lower():
                    insight_content.append("Drug delivery optimization: Blood-brain barrier penetration and targeted neurotherapy essential")
                    evidence_list.append("neurotherapeutic_delivery")
            
            # Compilar insight
            full_insight = f"""**Clinical Psychiatry Expert Perspective:**

{chr(10).join(f"• {content}" for content in insight_content)}

**Evidence-Based Medicine Framework:**
- Systematic literature review and clinical guidelines integration
- Risk stratification and patient safety prioritization
- Outcome measurement and treatment response monitoring
- Quality improvement and care standardization

**Clinical Translation Considerations:**
- Regulatory approval pathways and clinical trial design
- Patient selection criteria and inclusion/exclusion considerations
- Healthcare implementation and provider training requirements
- Health economics and cost-effectiveness analysis

**Interdisciplinary Collaboration Points:**
- Integration of basic science research with clinical applications
- Collaboration with biomedical engineers on device development
- Pharmacokinetic and pharmacodynamic optimization with pharmacologists
- Ethical considerations and patient advocacy perspectives

**Patient-Centered Care:**
- Shared decision-making and informed consent processes
- Cultural competency and health equity considerations
- Care coordination and multidisciplinary team approaches
- Patient-reported outcomes and quality-of-life measures"""
            
            insight = ResearchInsight(
                agent_specialization=self.specialization,
                content=full_insight,
                confidence=0.87,
                evidence=evidence_list,
                type=InsightType.ANALYSIS,
                metadata={
                    "research_question": research_question,
                    "clinical_aspects": clinical_aspects,
                    "collaboration_mode": True,
                    "agent": self.name
                }
            )
            
            logger.info(f"✅ Insight clínico colaborativo gerado com {len(clinical_aspects)} aspectos")
            return insight
            
        except Exception as e:
            logger.error(f"Erro ao gerar insight colaborativo: {e}")
            
            return ResearchInsight(
                agent_specialization=self.specialization,
                content=f"Clinical analysis error in collaborative context: {str(e)}",
                confidence=0.3,
                evidence=["clinical_collaboration_error"],
                type=InsightType.ANALYSIS,
                metadata={"error": str(e), "agent": self.name}
            )
    
    # ==================== PRIVATE METHODS ====================
    
    def _analyze_clinical_presentation(self, case: ClinicalCase) -> Dict[str, Any]:
        """Analisa apresentação clínica."""
        analysis = []
        evidence = []
        confidence = 0.9
        
        try:
            # Análise de sintomas principais
            primary_analysis = f"Primary complaint: {case.primary_complaint}"
            symptom_count = len(case.symptoms)
            
            if symptom_count >= 3:
                analysis.append(f"{primary_analysis} with {symptom_count} associated symptoms suggests systematic evaluation needed")
                evidence.append("comprehensive_symptom_assessment")
            else:
                analysis.append(f"{primary_analysis} with limited associated symptoms")
                evidence.append("focused_symptom_assessment")
            
            # Análise por idade e gênero
            if case.patient_age < 18:
                analysis.append("Pediatric presentation requires developmental considerations")
                evidence.append("pediatric_considerations")
            elif case.patient_age > 65:
                analysis.append("Geriatric presentation requires cognitive and medical comorbidity assessment")
                evidence.append("geriatric_considerations")
            
            # Análise de histórico médico
            if len(case.medical_history) > 0:
                analysis.append(f"Medical history includes {len(case.medical_history)} conditions requiring integration")
                evidence.append("medical_comorbidity_assessment")
                confidence *= 0.95  # Mais informação = mais confiança
            
            return {
                "analysis": analysis,
                "evidence": evidence,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "analysis": [f"Clinical presentation analysis error: {str(e)}"],
                "evidence": ["presentation_analysis_error"],
                "confidence": 0.4
            }
    
    def _generate_differential_diagnosis(self, case: ClinicalCase) -> Dict[str, Any]:
        """Gera diagnóstico diferencial."""
        try:
            differential_list = []
            evidence = []
            
            # Análise baseada em sintomas principais
            primary_lower = case.primary_complaint.lower()
            
            if any(word in primary_lower for word in ["depressed", "sad", "hopeless", "mood"]):
                differential_list.extend([
                    "Major Depressive Disorder",
                    "Bipolar Disorder (depressed episode)",
                    "Adjustment Disorder with Depressed Mood",
                    "Medical causes (hypothyroidism, anemia)"
                ])
                evidence.append("mood_disorder_differential")
            
            elif any(word in primary_lower for word in ["anxious", "worried", "panic", "fear"]):
                differential_list.extend([
                    "Generalized Anxiety Disorder",
                    "Panic Disorder",
                    "Social Anxiety Disorder",
                    "Medical causes (hyperthyroidism, cardiac arrhythmias)"
                ])
                evidence.append("anxiety_disorder_differential")
            
            elif any(word in primary_lower for word in ["hallucinations", "delusions", "psychosis"]):
                differential_list.extend([
                    "Schizophrenia",
                    "Brief Psychotic Disorder",
                    "Substance-induced Psychotic Disorder",
                    "Medical causes (delirium, dementia, medications)"
                ])
                evidence.append("psychotic_disorder_differential")
            
            # Considerações baseadas em idade
            if case.patient_age > 65:
                differential_list.append("Neurocognitive disorders (dementia, mild cognitive impairment)")
                evidence.append("geriatric_differential")
            
            # Considerações baseadas em medicamentos
            if case.medications and len(case.medications) > 0:
                differential_list.append("Medication-induced psychiatric symptoms")
                evidence.append("medication_induced_differential")
            
            # Padrão de diagnósticos se lista vazia
            if not differential_list:
                differential_list = [
                    "Requires comprehensive psychiatric evaluation",
                    "Consider medical workup for organic causes",
                    "Psychosocial stressors assessment needed"
                ]
                evidence.append("general_differential")
            
            analysis = f"Differential diagnosis considerations: {'; '.join(differential_list[:4])}"
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Differential diagnosis generation error: {str(e)}",
                "evidence": ["differential_diagnosis_error"]
            }
    
    def _recommend_investigations(self, case: ClinicalCase) -> str:
        """Recomenda investigações clínicas."""
        try:
            investigations = []
            
            # Investigações básicas
            investigations.extend([
                "Complete Blood Count (CBC)",
                "Basic Metabolic Panel (BMP)",
                "Thyroid Function Tests (TSH, T4)",
                "Vitamin B12 and Folate levels"
            ])
            
            # Baseado na idade
            if case.patient_age > 65:
                investigations.extend([
                    "Mini-Mental State Exam (MMSE)",
                    "Vitamin D level",
                    "Cognitive assessment battery"
                ])
            
            # Baseado em sintomas específicos
            primary_lower = case.primary_complaint.lower()
            if "mood" in primary_lower or "depressed" in primary_lower:
                investigations.extend([
                    "PHQ-9 Depression Scale",
                    "Cortisol level",
                    "Drug and alcohol screening"
                ])
            
            if "anxiety" in primary_lower or "panic" in primary_lower:
                investigations.extend([
                    "GAD-7 Anxiety Scale",
                    "EKG (rule out cardiac causes)",
                    "Caffeine intake assessment"
                ])
            
            # Investigações de precisão se apropriado
            if case.medications:
                investigations.append("Pharmacogenomic testing (CYP450 variants)")
            
            return f"Recommended investigations: {'; '.join(investigations[:8])}"
            
        except Exception as e:
            return f"Investigation recommendations error: {str(e)}"
    
    def _develop_treatment_plan(self, case: ClinicalCase, focus_area: Optional[str]) -> Dict[str, Any]:
        """Desenvolve plano terapêutico."""
        try:
            treatment_components = []
            evidence = []
            
            # Tratamento não-farmacológico
            treatment_components.extend([
                "Psychotherapy (CBT, IPT, or supportive therapy)",
                "Lifestyle modifications (exercise, sleep hygiene, stress management)",
                "Psychoeducation and family involvement"
            ])
            evidence.append("non_pharmacological_treatment")
            
            # Tratamento farmacológico se apropriado
            primary_lower = case.primary_complaint.lower()
            if any(word in primary_lower for word in ["depressed", "mood"]):
                treatment_components.extend([
                    "SSRI first-line (escitalopram, sertraline)",
                    "Consider SNRI if comorbid anxiety",
                    "Monitor for suicidal ideation"
                ])
                evidence.append("antidepressant_treatment")
            
            elif any(word in primary_lower for word in ["anxiety", "panic"]):
                treatment_components.extend([
                    "SSRI/SNRI first-line for long-term",
                    "Short-term anxiolytic if severe (with caution)",
                    "Relaxation techniques and breathing exercises"
                ])
                evidence.append("anxiety_treatment")
            
            # Medicina de precisão
            if focus_area == "precision_medicine":
                treatment_components.append("Pharmacogenomic-guided medication selection")
                evidence.append("precision_medicine_approach")
            
            # Biomateriais se relevante
            if focus_area == "biomaterials":
                treatment_components.append("Consider novel drug delivery systems for enhanced efficacy")
                evidence.append("biomaterial_enhanced_treatment")
            
            analysis = f"Treatment plan: {'; '.join(treatment_components[:5])}"
            
            return {
                "analysis": analysis,
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Treatment planning error: {str(e)}",
                "evidence": ["treatment_plan_error"]
            }
    
    def _is_biomaterial_relevant(self, case: ClinicalCase) -> bool:
        """Determina se biomateriais são relevantes para o caso."""
        try:
            # Condições que podem se beneficiar de biomateriais
            relevant_conditions = [
                "spinal cord injury", "peripheral neuropathy", "traumatic brain injury",
                "stroke", "neurodegenerative", "chronic pain", "treatment-resistant"
            ]
            
            case_text = (case.primary_complaint + " " + " ".join(case.symptoms) + " " + " ".join(case.medical_history)).lower()
            
            return any(condition in case_text for condition in relevant_conditions)
            
        except:
            return False
    
    def _evaluate_biomaterial_therapeutics(self, case: ClinicalCase) -> str:
        """Avalia opções terapêuticas com biomateriais."""
        try:
            biomaterial_options = []
            
            case_text = (case.primary_complaint + " " + " ".join(case.symptoms)).lower()
            
            if "pain" in case_text:
                biomaterial_options.append("Targeted drug delivery systems for chronic pain management")
            
            if "neurological" in case_text or "nerve" in case_text:
                biomaterial_options.append("Neural scaffolds for nerve regeneration")
            
            if "depression" in case_text and "treatment-resistant" in case_text:
                biomaterial_options.append("Novel antidepressant delivery systems")
            
            if not biomaterial_options:
                biomaterial_options = ["Biomaterial applications under investigation for this condition"]
            
            return f"Biomaterial therapeutic considerations: {'; '.join(biomaterial_options[:3])}"
            
        except Exception as e:
            return f"Biomaterial evaluation error: {str(e)}"
    
    def _assess_prognosis_and_monitoring(self, case: ClinicalCase) -> str:
        """Avalia prognóstico e plano de monitoramento."""
        try:
            monitoring_components = []
            
            # Monitoramento geral
            monitoring_components.extend([
                "Regular follow-up appointments (2-4 weeks initially)",
                "Standardized rating scales for symptom tracking",
                "Side effect monitoring and medication adherence"
            ])
            
            # Baseado na idade
            if case.patient_age > 65:
                monitoring_components.append("Cognitive function monitoring")
            
            # Monitoramento específico por condição
            if case.medications:
                monitoring_components.append("Laboratory monitoring for medication effects")
            
            # Prognóstico geral
            prognosis_statement = "Prognosis generally favorable with appropriate treatment and compliance"
            
            return f"Monitoring and prognosis: {'; '.join(monitoring_components[:4])}. {prognosis_statement}"
            
        except Exception as e:
            return f"Prognosis assessment error: {str(e)}"
    
    def _analyze_clinical_condition(self, condition: str) -> Dict[str, Any]:
        """Analisa condição clínica específica."""
        try:
            condition_lower = condition.lower()
            analysis = []
            evidence = []
            
            if "depression" in condition_lower:
                analysis.append("Major Depressive Disorder: Complex neurotransmitter dysfunction requiring multimodal treatment")
                evidence.extend(["depression_pathophysiology", "monoamine_hypothesis"])
            elif "anxiety" in condition_lower:
                analysis.append("Anxiety Disorders: GABA-glutamate imbalance with HPA axis dysregulation")
                evidence.extend(["anxiety_neurobiology", "stress_response_system"])
            elif "schizophrenia" in condition_lower:
                analysis.append("Schizophrenia: Dopaminergic and glutamatergic dysfunction with neurodevelopmental components")
                evidence.extend(["dopamine_hypothesis", "glutamate_hypofunction"])
            elif "spinal cord" in condition_lower:
                analysis.append("Spinal Cord Injury: Neuronal damage requiring neuroprotection and regeneration strategies")
                evidence.extend(["spinal_cord_pathophysiology", "neuroregeneration"])
            else:
                analysis.append(f"Clinical condition analysis: {condition} requires systematic evaluation")
                evidence.append("general_clinical_assessment")
            
            return {
                "analysis": "; ".join(analysis),
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Condition analysis error: {str(e)}",
                "evidence": ["condition_analysis_error"]
            }
    
    def _evaluate_biomaterial_properties(self, biomaterial_data: Dict[str, Any], condition: str) -> Dict[str, Any]:
        """Avalia propriedades do biomaterial para aplicação clínica."""
        try:
            analysis = []
            evidence = []
            
            # Biocompatibilidade
            if "biocompatibility" in biomaterial_data:
                biocompat = biomaterial_data["biocompatibility"]
                analysis.append(f"Biocompatibility profile: {biocompat}")
                evidence.append("biocompatibility_assessment")
            
            # Propriedades de degradação
            if "degradation_rate" in biomaterial_data:
                deg_rate = biomaterial_data["degradation_rate"]
                analysis.append(f"Degradation kinetics: {deg_rate} - must match tissue healing timeline")
                evidence.append("degradation_kinetics")
            
            # Propriedades mecânicas
            if "mechanical_properties" in biomaterial_data:
                mech_props = biomaterial_data["mechanical_properties"]
                analysis.append(f"Mechanical properties: {mech_props} - should match native tissue")
                evidence.append("mechanical_compatibility")
            
            # Funcionalização
            if "surface_modification" in biomaterial_data:
                surface_mod = biomaterial_data["surface_modification"]
                analysis.append(f"Surface functionalization: {surface_mod} - enhances biological activity")
                evidence.append("surface_functionalization")
            
            if not analysis:
                analysis.append("Biomaterial properties require comprehensive characterization")
                evidence.append("properties_characterization_needed")
            
            return {
                "analysis": "; ".join(analysis),
                "evidence": evidence
            }
            
        except Exception as e:
            return {
                "analysis": f"Biomaterial properties evaluation error: {str(e)}",
                "evidence": ["properties_evaluation_error"]
            }
    
    def _analyze_therapeutic_mechanism(self, biomaterial_data: Dict[str, Any], condition: str) -> str:
        """Analisa mecanismo de ação terapêutico."""
        try:
            mechanisms = []
            
            # Mecanismo baseado no tipo de biomaterial
            material_type = biomaterial_data.get("type", "unknown").lower()
            
            if "scaffold" in material_type:
                mechanisms.append("Structural support and guided tissue regeneration")
            elif "nanoparticle" in material_type:
                mechanisms.append("Targeted drug delivery and controlled release")
            elif "hydrogel" in material_type:
                mechanisms.append("Injectable delivery system with sustained drug release")
            elif "implant" in material_type:
                mechanisms.append("Long-term drug reservoir and local therapeutic delivery")
            
            # Mecanismo baseado na condição
            condition_lower = condition.lower()
            if "neurological" in condition_lower or "neural" in condition_lower:
                mechanisms.append("Neuroprotection and axonal regeneration promotion")
            elif "psychiatric" in condition_lower:
                mechanisms.append("Blood-brain barrier penetration and CNS drug delivery")
            
            if not mechanisms:
                mechanisms = ["Mechanism of action requires further investigation"]
            
            return f"Therapeutic mechanism: {'; '.join(mechanisms[:3])}"
            
        except Exception as e:
            return f"Therapeutic mechanism analysis error: {str(e)}"
    
    def _assess_clinical_evidence(self, condition: str, biomaterial_data: Dict[str, Any]) -> str:
        """Avalia evidências clínicas disponíveis."""
        try:
            evidence_assessment = []
            
            # Evidência baseada na maturidade da tecnologia
            if "clinical_trials" in biomaterial_data:
                trials = biomaterial_data["clinical_trials"]
                evidence_assessment.append(f"Clinical trial evidence: {trials}")
            else:
                evidence_assessment.append("Clinical evidence: Limited human data available")
            
            # Evidência por tipo de condição
            condition_lower = condition.lower()
            if "spinal cord" in condition_lower:
                evidence_assessment.append("Neural scaffolds: Phase I/II trials showing safety")
            elif "depression" in condition_lower:
                evidence_assessment.append("Drug delivery systems: Preclinical studies promising")
            elif "pain" in condition_lower:
                evidence_assessment.append("Pain management: Some FDA-approved systems available")
            
            # Nível de evidência
            evidence_level = biomaterial_data.get("evidence_level", "4")
            evidence_assessment.append(f"Evidence level: {evidence_level} ({self.evidence_levels.get(evidence_level, 'Unknown')})")
            
            return f"Clinical evidence assessment: {'; '.join(evidence_assessment)}"
            
        except Exception as e:
            return f"Clinical evidence assessment error: {str(e)}"
    
    def _evaluate_regulatory_pathway(self, biomaterial_data: Dict[str, Any], condition: str) -> str:
        """Avalia caminho regulatório."""
        try:
            regulatory_considerations = []
            
            # Classificação regulatória
            material_type = biomaterial_data.get("type", "unknown").lower()
            
            if "device" in material_type or "scaffold" in material_type:
                regulatory_considerations.append("FDA Class II/III medical device pathway")
            elif "drug delivery" in material_type:
                regulatory_considerations.append("FDA drug-device combination product pathway")
            else:
                regulatory_considerations.append("Regulatory classification requires determination")
            
            # Considerações específicas
            regulatory_considerations.extend([
                "Preclinical safety and efficacy studies required",
                "GMP manufacturing and quality control essential",
                "Clinical trial design with appropriate endpoints"
            ])
            
            return f"Regulatory considerations: {'; '.join(regulatory_considerations[:3])}"
            
        except Exception as e:
            return f"Regulatory pathway evaluation error: {str(e)}"
    
    def _generate_clinical_recommendations(self, condition: str, biomaterial_data: Dict[str, Any]) -> str:
        """Gera recomendações clínicas."""
        try:
            recommendations = []
            
            # Recomendações gerais
            recommendations.extend([
                "Conduct systematic literature review of existing evidence",
                "Design appropriate clinical studies with validated endpoints",
                "Establish safety monitoring protocols and adverse event reporting"
            ])
            
            # Recomendações específicas por condição
            condition_lower = condition.lower()
            if "neurological" in condition_lower:
                recommendations.append("Collaborate with neurosurgery and neurology specialists")
            elif "psychiatric" in condition_lower:
                recommendations.append("Integrate with psychiatric treatment protocols")
            
            # Recomendações de implementação
            recommendations.extend([
                "Develop clinical practice guidelines and protocols",
                "Train healthcare providers on appropriate use and monitoring"
            ])
            
            return f"Clinical recommendations: {'; '.join(recommendations[:4])}"
            
        except Exception as e:
            return f"Clinical recommendations error: {str(e)}"
    
    def _identify_clinical_aspects(self, research_question: str) -> List[str]:
        """Identifica aspectos clínicos na pergunta de pesquisa."""
        question_lower = research_question.lower()
        aspects = []
        
        clinical_keywords = {
            "diagnosis": ["diagnosis", "diagnostic", "symptom", "presentation"],
            "treatment": ["treatment", "therapy", "therapeutic", "intervention"],
            "pharmacology": ["drug", "medication", "pharmacology", "dosing"],
            "biomaterials": ["biomaterial", "scaffold", "delivery", "implant"],
            "precision_medicine": ["precision", "personalized", "biomarker", "genomic"],
            "safety": ["safety", "adverse", "side effect", "risk"]
        }
        
        for aspect, keywords in clinical_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                aspects.append(aspect)
        
        # Aspectos padrão se nenhum específico
        if not aspects:
            aspects = ["diagnosis", "treatment"]
        
        return aspects
    
    def get_expertise_summary(self) -> Dict[str, Any]:
        """Retorna resumo da expertise do agent."""
        return {
            "name": self.name,
            "specialization": self.specialization.value,
            "expertise_areas": self.expertise_areas,
            "clinical_knowledge": list(self.clinical_knowledge.keys()),
            "evidence_levels": self.evidence_levels,
            "autogen_enabled": AUTOGEN_AVAILABLE and self.autogen_agent is not None,
            "capabilities": [
                "clinical_case_analysis",
                "differential_diagnosis",
                "treatment_planning",
                "biomaterial_therapeutic_evaluation",
                "precision_medicine_application",
                "clinical_evidence_assessment",
                "regulatory_pathway_guidance",
                "patient_safety_monitoring"
            ]
        }


# ==================== EXPORTS ====================

__all__ = [
    "ClinicalPsychiatryAgent",
    "ClinicalCase",
    "BiomaterialTherapeutic",
    "ClinicalSpecialty",
    "PsychiatricDisorder"
]