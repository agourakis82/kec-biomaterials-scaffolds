"""
DARWIN SCIENTIFIC DISCOVERY - Cross Domain Analyzer
Sistema avançado de análise interdisciplinar para detectar conexões entre domínios científicos
"""

import asyncio
import logging
import re
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from ..models.discovery_models import (
    PaperMetadata,
    CrossDomainInsight,
    ScientificDomain,
    NoveltyLevel
)

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DomainProfile:
    """Perfil de um domínio científico."""
    domain: ScientificDomain
    papers: List[PaperMetadata]
    key_concepts: Set[str]
    methodologies: Set[str]
    vocabulary: Set[str]
    temporal_activity: List[Tuple[datetime, int]]
    concept_vectors: Optional[np.ndarray] = None


@dataclass
class ConceptualBridge:
    """Ponte conceitual entre domínios."""
    concept: str
    primary_domain: ScientificDomain
    connected_domains: List[ScientificDomain]
    strength: float
    evidence_papers: List[str]
    bridge_type: str  # 'methodological', 'conceptual', 'applied'


@dataclass
class InterdisciplinaryOpportunity:
    """Oportunidade de pesquisa interdisciplinar."""
    opportunity_id: str
    domains_involved: List[ScientificDomain]
    description: str
    potential_impact: str
    suggested_approaches: List[str]
    supporting_evidence: List[str]
    confidence_score: float
    research_gaps: List[str]


@dataclass
class CrossDomainTransfer:
    """Transferência de conhecimento entre domínios."""
    source_domain: ScientificDomain
    target_domain: ScientificDomain
    transferred_concept: str
    transfer_type: str  # 'method', 'theory', 'technology', 'application'
    feasibility_score: float
    examples: List[str]
    barriers: List[str]


# =============================================================================
# CROSS DOMAIN ANALYZER CLASS
# =============================================================================

class CrossDomainAnalyzer:
    """
    Analisador avançado de conexões interdisciplinares.
    """
    
    def __init__(self):
        """Inicializa o analisador cross-domain."""
        
        # Profiles por domínio
        self.domain_profiles: Dict[ScientificDomain, DomainProfile] = {}
        
        # Mapping de conexões conhecidas
        self.known_connections = self._initialize_known_connections()
        
        # Bridge patterns especializados
        self.bridge_patterns = self._initialize_bridge_patterns()
        
        # Vectorizer para análise semântica
        self.vectorizer = None
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
        
        # Cache de insights detectados
        self.insight_cache: Dict[str, CrossDomainInsight] = {}
        
        # Estatísticas
        self.stats = {
            'domains_analyzed': 0,
            'bridges_detected': 0,
            'insights_generated': 0,
            'opportunities_identified': 0,
            'last_analysis': None
        }
        
        logger.info(f"Cross Domain Analyzer initialized - sklearn: {SKLEARN_AVAILABLE}, networkx: {NETWORKX_AVAILABLE}")
    
    def _initialize_known_connections(self) -> Dict[Tuple[ScientificDomain, ScientificDomain], List[str]]:
        """Inicializa conexões interdisciplinares conhecidas."""
        return {
            # Biomaterials ↔ Neuroscience
            (ScientificDomain.BIOMATERIALS, ScientificDomain.NEUROSCIENCE): [
                'neural scaffolds', 'bioelectronics', 'neural interfaces',
                'brain implants', 'neural tissue engineering', 'conductive polymers',
                'neural stimulation', 'biocompatibility', 'neural regeneration'
            ],
            
            # Neuroscience ↔ Philosophy
            (ScientificDomain.NEUROSCIENCE, ScientificDomain.PHILOSOPHY): [
                'consciousness', 'free will', 'mind-brain problem', 'qualia',
                'neural correlates of consciousness', 'cognitive architecture',
                'personal identity', 'mental causation', 'embodied cognition',
                'phenomenology', 'philosophy of mind'
            ],
            
            # Philosophy ↔ Quantum
            (ScientificDomain.PHILOSOPHY, ScientificDomain.QUANTUM_MECHANICS): [
                'quantum consciousness', 'measurement problem', 'observer effect',
                'quantum indeterminacy', 'many-worlds interpretation',
                'quantum logic', 'complementarity', 'quantum information',
                'quantum reality', 'copenhagen interpretation'
            ],
            
            # Quantum ↔ Biomaterials
            (ScientificDomain.QUANTUM_MECHANICS, ScientificDomain.BIOMATERIALS): [
                'quantum biology', 'quantum coherence', 'quantum tunneling',
                'photosynthesis', 'quantum effects in proteins',
                'quantum sensing', 'quantum dots', 'spin states',
                'magnetic nanoparticles', 'quantum materials'
            ],
            
            # Mathematics ↔ All domains
            (ScientificDomain.MATHEMATICS, ScientificDomain.BIOMATERIALS): [
                'mathematical modeling', 'network theory', 'graph theory',
                'optimization', 'topology', 'differential equations',
                'statistical mechanics', 'percolation theory'
            ],
            
            (ScientificDomain.MATHEMATICS, ScientificDomain.NEUROSCIENCE): [
                'neural networks', 'dynamical systems', 'information theory',
                'graph theory', 'statistical analysis', 'machine learning',
                'signal processing', 'computational modeling'
            ],
            
            (ScientificDomain.MATHEMATICS, ScientificDomain.PHILOSOPHY): [
                'mathematical logic', 'set theory', 'category theory',
                'foundations of mathematics', 'formal systems',
                'computability theory', 'proof theory'
            ],
            
            (ScientificDomain.MATHEMATICS, ScientificDomain.QUANTUM_MECHANICS): [
                'linear algebra', 'hilbert spaces', 'operator theory',
                'group theory', 'topology', 'differential geometry',
                'probability theory', 'functional analysis'
            ]
        }
    
    def _initialize_bridge_patterns(self) -> Dict[str, List[str]]:
        """Inicializa padrões de pontes conceituais."""
        return {
            'methodological_bridges': [
                r'\b(?:using|via|through|employing)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:from|in)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
                r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:approach|method|technique)\s+(?:adapted|borrowed|transferred)\s+from\s+([a-z]+)\b',
                r'\b(?:inspired by|based on|drawing from)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:in|from)\s+([a-z]+)\b'
            ],
            
            'conceptual_bridges': [
                r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:similar to|analogous to|parallel to)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
                r'\b(?:bridging|connecting|linking)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:and|with)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
                r'\b(?:interdisciplinary|cross-field|multi-domain)\s+(?:approach|study|research)\b'
            ],
            
            'application_bridges': [
                r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:applications?|uses?)\s+in\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
                r'\b(?:applying|implementing|utilizing)\s+([a-z]+(?:\s+[a-z]+){0,2})\s+(?:to|for|in)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
                r'\b([a-z]+)\s+(?:meets|crosses|intersects)\s+([a-z]+)\b'
            ]
        }
    
    def _extract_domain_vocabulary(self, papers: List[PaperMetadata], domain: ScientificDomain) -> Set[str]:
        """Extrai vocabulário especializado de um domínio."""
        vocabulary = set()
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
            
            # Extrair termos técnicos
            technical_terms = re.findall(r'\b[a-z]{4,}\b', text)
            vocabulary.update(technical_terms)
            
            # Extrair compostos técnicos
            compound_terms = re.findall(r'\b[a-z]+-[a-z]+\b', text)
            vocabulary.update(compound_terms)
            
            # Extrair acrônimos e códigos
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
            vocabulary.update([a.lower() for a in acronyms])
        
        # Filtrar termos muito comuns
        common_words = {'study', 'research', 'analysis', 'method', 'results', 'conclusion', 'data', 'system'}
        vocabulary = vocabulary - common_words
        
        return vocabulary
    
    def _extract_methodologies(self, papers: List[PaperMetadata]) -> Set[str]:
        """Extrai metodologias de uma lista de papers."""
        methodologies = set()
        
        method_patterns = [
            r'\b(?:using|via|employing|through)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',
            r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:technique|method|approach|analysis|measurement)\b',
            r'\b(?:performed|conducted|carried out)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'
        ]
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            
            for pattern in method_patterns:
                matches = re.findall(pattern, text)
                methodologies.update(matches)
        
        # Filtrar metodologias relevantes
        filtered_methods = set()
        for method in methodologies:
            if len(method) > 3 and method not in ['data', 'results', 'study']:
                filtered_methods.add(method)
        
        return filtered_methods
    
    def _extract_key_concepts(self, papers: List[PaperMetadata], domain: ScientificDomain) -> Set[str]:
        """Extrai conceitos-chave específicos do domínio."""
        concepts = set()
        
        # Padrões para conceitos importantes
        concept_patterns = [
            r'\b(?:novel|new|innovative)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
            r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:theory|model|framework|paradigm)\b',
            r'\b(?:fundamental|key|essential|critical)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'
        ]
        
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            
            for pattern in concept_patterns:
                matches = re.findall(pattern, text)
                if isinstance(matches[0], tuple) if matches else False:
                    concepts.update([match[0] for match in matches])
                else:
                    concepts.update(matches)
            
            # Adicionar keywords do paper
            concepts.update([kw.lower() for kw in paper.keywords])
        
        return concepts
    
    def _build_domain_profile(self, domain: ScientificDomain, papers: List[PaperMetadata]) -> DomainProfile:
        """Constrói perfil completo de um domínio científico."""
        if not papers:
            return DomainProfile(
                domain=domain,
                papers=[],
                key_concepts=set(),
                methodologies=set(),
                vocabulary=set(),
                temporal_activity=[]
            )
        
        # Extrair características do domínio
        vocabulary = self._extract_domain_vocabulary(papers, domain)
        methodologies = self._extract_methodologies(papers)
        concepts = self._extract_key_concepts(papers, domain)
        
        # Análise temporal
        temporal_activity = []
        papers_by_month = defaultdict(int)
        
        for paper in papers:
            if paper.publication_date:
                month_key = paper.publication_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                papers_by_month[month_key] += 1
        
        temporal_activity = [(date, count) for date, count in sorted(papers_by_month.items())]
        
        # Criar vetores conceituais se sklearn disponível
        concept_vectors = None
        if SKLEARN_AVAILABLE and papers:
            try:
                texts = [f"{p.title} {p.abstract}" for p in papers]
                if self.vectorizer and len(texts) > 1:
                    vectors = self.vectorizer.fit_transform(texts)
                    # Handle sparse matrix conversion safely
                    try:
                        concept_vectors = vectors.toarray()
                    except AttributeError:
                        # Handle scipy sparse matrix
                        import scipy.sparse
                        if scipy.sparse.issparse(vectors):
                            concept_vectors = vectors.todense().A
                        else:
                            concept_vectors = np.array(vectors)
            except Exception as e:
                logger.warning(f"Failed to create concept vectors for {domain}: {e}")
        
        return DomainProfile(
            domain=domain,
            papers=papers,
            key_concepts=concepts,
            methodologies=methodologies,
            vocabulary=vocabulary,
            temporal_activity=temporal_activity,
            concept_vectors=concept_vectors
        )
    
    def _detect_conceptual_bridges(
        self, 
        profile1: DomainProfile, 
        profile2: DomainProfile
    ) -> List[ConceptualBridge]:
        """Detecta pontes conceituais entre dois domínios."""
        bridges = []
        
        try:
            # 1. Conceitos compartilhados
            shared_concepts = profile1.key_concepts & profile2.key_concepts
            
            for concept in shared_concepts:
                evidence_papers = []
                
                # Encontrar papers que mencionam este conceito
                for paper in profile1.papers + profile2.papers:
                    text = f"{paper.title} {paper.abstract}".lower()
                    if concept in text:
                        evidence_papers.append(paper.doc_id)
                
                if len(evidence_papers) >= 2:  # Mínimo de evidência
                    bridge = ConceptualBridge(
                        concept=concept,
                        primary_domain=profile1.domain,
                        connected_domains=[profile2.domain],
                        strength=min(1.0, len(evidence_papers) / 10.0),
                        evidence_papers=evidence_papers[:5],  # Top 5
                        bridge_type='conceptual'
                    )
                    bridges.append(bridge)
            
            # 2. Metodologias transferíveis
            shared_methods = profile1.methodologies & profile2.methodologies
            
            for method in shared_methods:
                evidence_papers = []
                
                for paper in profile1.papers + profile2.papers:
                    text = f"{paper.title} {paper.abstract}".lower()
                    if method in text:
                        evidence_papers.append(paper.doc_id)
                
                if len(evidence_papers) >= 1:
                    bridge = ConceptualBridge(
                        concept=method,
                        primary_domain=profile1.domain,
                        connected_domains=[profile2.domain],
                        strength=min(1.0, len(evidence_papers) / 5.0),
                        evidence_papers=evidence_papers[:3],
                        bridge_type='methodological'
                    )
                    bridges.append(bridge)
            
            # 3. Conexões explícitas conhecidas
            domain_pair = (profile1.domain, profile2.domain)
            reverse_pair = (profile2.domain, profile1.domain)
            
            known_bridges = []
            if domain_pair in self.known_connections:
                known_bridges = self.known_connections[domain_pair]
            elif reverse_pair in self.known_connections:
                known_bridges = self.known_connections[reverse_pair]
            
            for bridge_concept in known_bridges:
                # Verificar evidência nos papers
                evidence_count = 0
                evidence_papers = []
                
                for paper in profile1.papers + profile2.papers:
                    text = f"{paper.title} {paper.abstract}".lower()
                    if bridge_concept.lower() in text:
                        evidence_count += 1
                        evidence_papers.append(paper.doc_id)
                
                if evidence_count > 0:
                    bridge = ConceptualBridge(
                        concept=bridge_concept,
                        primary_domain=profile1.domain,
                        connected_domains=[profile2.domain],
                        strength=min(1.0, evidence_count / 3.0),
                        evidence_papers=evidence_papers[:3],
                        bridge_type='applied'
                    )
                    bridges.append(bridge)
        
        except Exception as e:
            logger.error(f"Error detecting bridges between {profile1.domain} and {profile2.domain}: {e}")
        
        return bridges
    
    def _compute_cross_domain_similarity(self, profile1: DomainProfile, profile2: DomainProfile) -> float:
        """Computa similaridade entre dois domínios."""
        if not profile1.papers or not profile2.papers:
            return 0.0
        
        # Similaridade baseada em conceitos compartilhados
        concepts1 = profile1.key_concepts
        concepts2 = profile2.key_concepts
        
        if not concepts1 or not concepts2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(concepts1 & concepts2)
        union = len(concepts1 | concepts2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Similaridade baseada em vocabulário
        vocab1 = profile1.vocabulary
        vocab2 = profile2.vocabulary
        
        vocab_intersection = len(vocab1 & vocab2)
        vocab_union = len(vocab1 | vocab2)
        
        vocab_jaccard = vocab_intersection / vocab_union if vocab_union > 0 else 0.0
        
        # Similaridade baseada em metodologias
        methods1 = profile1.methodologies
        methods2 = profile2.methodologies
        
        method_intersection = len(methods1 & methods2)
        method_union = len(methods1 | methods2)
        
        method_jaccard = method_intersection / method_union if method_union > 0 else 0.0
        
        # Score combinado ponderado
        similarity = (
            0.5 * jaccard +
            0.3 * vocab_jaccard +
            0.2 * method_jaccard
        )
        
        return similarity
    
    def _generate_cross_domain_insight(
        self, 
        bridges: List[ConceptualBridge],
        profile1: DomainProfile,
        profile2: DomainProfile,
        similarity_score: float
    ) -> Optional[CrossDomainInsight]:
        """Gera insight interdisciplinar baseado nas pontes detectadas."""
        if not bridges:
            return None
        
        try:
            # Organizar bridges por tipo e força
            conceptual_bridges = [b for b in bridges if b.bridge_type == 'conceptual']
            methodological_bridges = [b for b in bridges if b.bridge_type == 'methodological']
            applied_bridges = [b for b in bridges if b.bridge_type == 'applied']
            
            # Gerar descrição do insight
            description_parts = []
            
            if conceptual_bridges:
                top_concepts = sorted(conceptual_bridges, key=lambda b: b.strength, reverse=True)[:3]
                concepts_str = ", ".join([b.concept for b in top_concepts])
                description_parts.append(f"Shared conceptual foundations in {concepts_str}")
            
            if methodological_bridges:
                top_methods = sorted(methodological_bridges, key=lambda b: b.strength, reverse=True)[:2]
                methods_str = ", ".join([b.concept for b in top_methods])
                description_parts.append(f"Transferable methodologies: {methods_str}")
            
            if applied_bridges:
                top_applied = sorted(applied_bridges, key=lambda b: b.strength, reverse=True)[:2]
                applied_str = ", ".join([b.concept for b in top_applied])
                description_parts.append(f"Applied connections through {applied_str}")
            
            description = ". ".join(description_parts)
            
            # Identificar aplicações potenciais
            potential_applications = []
            
            domain_pair = (profile1.domain, profile2.domain)
            if domain_pair == (ScientificDomain.BIOMATERIALS, ScientificDomain.NEUROSCIENCE):
                potential_applications = [
                    "Neural scaffold development",
                    "Bioelectronic interfaces",
                    "Brain-computer implants",
                    "Neural tissue regeneration"
                ]
            elif domain_pair == (ScientificDomain.NEUROSCIENCE, ScientificDomain.PHILOSOPHY):
                potential_applications = [
                    "Consciousness studies",
                    "AI ethics frameworks",
                    "Cognitive architecture models",
                    "Free will research"
                ]
            elif domain_pair == (ScientificDomain.QUANTUM_MECHANICS, ScientificDomain.BIOMATERIALS):
                potential_applications = [
                    "Quantum sensing in biology",
                    "Quantum-enhanced drug delivery",
                    "Coherent biological processes",
                    "Quantum dots for imaging"
                ]
            
            # Identificar lacunas de pesquisa
            research_gaps = [
                f"Limited cross-pollination between {profile1.domain.value} and {profile2.domain.value}",
                "Need for interdisciplinary methodological frameworks",
                "Insufficient collaborative research platforms"
            ]
            
            # Transferências metodológicas
            methodology_transfers = []
            for bridge in methodological_bridges:
                methodology_transfers.append(f"{bridge.concept} from {profile1.domain.value} to {profile2.domain.value}")
            
            # Pontes conceituais
            conceptual_bridges_desc = []
            for bridge in conceptual_bridges:
                conceptual_bridges_desc.append(f"{bridge.concept} bridges {profile1.domain.value}-{profile2.domain.value}")
            
            # Calcular confidence score
            bridge_strength_avg = np.mean([b.strength for b in bridges]) if bridges else 0.0
            evidence_count = sum(len(b.evidence_papers) for b in bridges)
            
            confidence_score = min(1.0, (
                0.4 * bridge_strength_avg +
                0.3 * similarity_score +
                0.3 * min(1.0, evidence_count / 10.0)
            ))
            
            # Criar insight
            insight_id = f"{profile1.domain.value}_{profile2.domain.value}_{int(datetime.now().timestamp())}"
            
            insight = CrossDomainInsight(
                insight_id=insight_id,
                primary_domain=profile1.domain,
                connected_domains=[profile2.domain],
                connection_strength=similarity_score,
                papers_involved=[paper_id for bridge in bridges for paper_id in bridge.evidence_papers],
                insight_description=description,
                potential_applications=potential_applications,
                research_gaps_identified=research_gaps,
                methodology_transfers=methodology_transfers,
                conceptual_bridges=conceptual_bridges_desc,
                confidence_score=float(confidence_score)
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error generating cross-domain insight: {e}")
            return None
    
    def analyze_domain_papers(self, domain_papers: Dict[ScientificDomain, List[PaperMetadata]]):
        """Analisa papers por domínio e constrói profiles."""
        
        self.domain_profiles.clear()
        
        for domain, papers in domain_papers.items():
            if papers:
                logger.info(f"Building profile for {domain.value} with {len(papers)} papers")
                profile = self._build_domain_profile(domain, papers)
                self.domain_profiles[domain] = profile
        
        self.stats['domains_analyzed'] = len(self.domain_profiles)
        logger.info(f"Built profiles for {len(self.domain_profiles)} domains")
    
    def detect_cross_domain_insights(
        self, 
        threshold: float = 0.1
    ) -> List[CrossDomainInsight]:
        """
        Detecta insights interdisciplinares entre todos os domínios.
        
        Args:
            threshold: Threshold mínimo de similaridade para gerar insights
            
        Returns:
            Lista de insights interdisciplinares detectados
        """
        insights = []
        
        # Analisar todas as combinações de domínios
        domain_pairs = list(combinations(self.domain_profiles.keys(), 2))
        
        for domain1, domain2 in domain_pairs:
            try:
                profile1 = self.domain_profiles[domain1]
                profile2 = self.domain_profiles[domain2]
                
                logger.debug(f"Analyzing connection between {domain1.value} and {domain2.value}")
                
                # Detectar pontes conceituais
                bridges = self._detect_conceptual_bridges(profile1, profile2)
                
                if not bridges:
                    continue
                
                # Calcular similaridade
                similarity = self._compute_cross_domain_similarity(profile1, profile2)
                
                if similarity >= threshold:
                    # Gerar insight
                    insight = self._generate_cross_domain_insight(bridges, profile1, profile2, similarity)
                    
                    if insight and insight.confidence_score >= 0.3:  # Threshold de confidence
                        insights.append(insight)
                        self.insight_cache[insight.insight_id] = insight
                        
                        logger.info(f"Generated insight between {domain1.value}-{domain2.value}: {insight.confidence_score:.3f}")
                
                self.stats['bridges_detected'] += len(bridges)
                
            except Exception as e:
                logger.error(f"Error analyzing {domain1.value}-{domain2.value}: {e}")
        
        # Ordenar insights por confidence score
        insights.sort(key=lambda i: i.confidence_score, reverse=True)
        
        self.stats['insights_generated'] = len(insights)
        self.stats['last_analysis'] = datetime.now(timezone.utc)
        
        return insights
    
    async def detect_cross_domain_insights_async(
        self, 
        threshold: float = 0.1
    ) -> List[CrossDomainInsight]:
        """Versão assíncrona da detecção de insights."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.detect_cross_domain_insights, threshold
        )
    
    def identify_research_opportunities(
        self, 
        insights: List[CrossDomainInsight]
    ) -> List[InterdisciplinaryOpportunity]:
        """Identifica oportunidades de pesquisa interdisciplinar baseadas nos insights."""
        opportunities = []
        
        # Agrupar insights por domínios similares
        domain_clusters = defaultdict(list)
        
        for insight in insights:
            domains = tuple(sorted([insight.primary_domain] + insight.connected_domains))
            domain_clusters[domains].append(insight)
        
        # Gerar oportunidades para cada cluster
        for domains, cluster_insights in domain_clusters.items():
            if len(cluster_insights) >= 1:  # Mínimo de insights para oportunidade
                
                # Combinar informações dos insights
                all_applications = []
                all_gaps = []
                all_transfers = []
                
                for insight in cluster_insights:
                    all_applications.extend(insight.potential_applications)
                    all_gaps.extend(insight.research_gaps_identified)
                    all_transfers.extend(insight.methodology_transfers)
                
                # Deduplicate
                unique_applications = list(set(all_applications))
                unique_gaps = list(set(all_gaps))
                unique_transfers = list(set(all_transfers))
                
                # Calcular confidence média
                avg_confidence = np.mean([i.confidence_score for i in cluster_insights])
                
                opportunity = InterdisciplinaryOpportunity(
                    opportunity_id=f"opp_{'_'.join([d.value for d in domains])}_{int(datetime.now().timestamp())}",
                    domains_involved=list(domains),
                    description=f"Cross-domain research opportunity spanning {', '.join([d.value for d in domains])}",
                    potential_impact="High - novel interdisciplinary applications",
                    suggested_approaches=unique_transfers[:5],
                    supporting_evidence=[i.insight_id for i in cluster_insights],
                    confidence_score=float(avg_confidence),
                    research_gaps=unique_gaps[:3]
                )
                
                opportunities.append(opportunity)
        
        self.stats['opportunities_identified'] = len(opportunities)
        
        return opportunities
    
    def suggest_knowledge_transfers(
        self, 
        source_domain: ScientificDomain,
        target_domain: ScientificDomain
    ) -> List[CrossDomainTransfer]:
        """Sugere transferências específicas de conhecimento entre domínios."""
        transfers = []
        
        if source_domain not in self.domain_profiles or target_domain not in self.domain_profiles:
            return transfers
        
        source_profile = self.domain_profiles[source_domain]
        target_profile = self.domain_profiles[target_domain]
        
        # Metodologias do domínio fonte não presentes no alvo
        transferable_methods = source_profile.methodologies - target_profile.methodologies
        
        for method in transferable_methods:
            # Avaliar feasibility
            feasibility = 0.5  # Base score
            
            # Aumentar se há conceitos relacionados
            related_concepts = source_profile.key_concepts & target_profile.key_concepts
            if related_concepts:
                feasibility += 0.3
            
            # Verificar exemplos na literatura
            examples = []
            for paper in source_profile.papers:
                if method in f"{paper.title} {paper.abstract}".lower():
                    examples.append(f"{paper.title[:50]}...")
            
            transfer = CrossDomainTransfer(
                source_domain=source_domain,
                target_domain=target_domain,
                transferred_concept=method,
                transfer_type='method',
                feasibility_score=min(1.0, feasibility),
                examples=examples[:3],
                barriers=[
                    "Different experimental paradigms",
                    "Technical implementation challenges",
                    "Lack of cross-domain expertise"
                ]
            )
            
            transfers.append(transfer)
        
        return transfers[:5]  # Top 5 transfers
    
    def get_domain_profile(self, domain: ScientificDomain) -> Optional[DomainProfile]:
        """Retorna profile de um domínio específico."""
        return self.domain_profiles.get(domain)
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da análise."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Limpa caches do analisador."""
        self.insight_cache.clear()
        self.domain_profiles.clear()
        logger.info("Cross-domain analyzer cache cleared")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Instância global do Cross Domain Analyzer
_cross_domain_analyzer = None

def get_cross_domain_analyzer() -> CrossDomainAnalyzer:
    """Retorna instância singleton do Cross Domain Analyzer."""
    global _cross_domain_analyzer
    if _cross_domain_analyzer is None:
        _cross_domain_analyzer = CrossDomainAnalyzer()
    return _cross_domain_analyzer