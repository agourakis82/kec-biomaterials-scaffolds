"""
DARWIN SCIENTIFIC DISCOVERY - Novelty Detector
Sistema avançado de detecção automática de novidades e insights científicos
"""

import asyncio
import logging
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from ..models.discovery_models import (
    PaperMetadata,
    NoveltyAnalysisResult,
    NoveltyLevel,
    NoveltyThreshold,
    ScientificDomain,
    EmergingTrend
)

# Optional dependencies with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    KMeans = None
    TruncatedSVD = None
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
class SemanticFeatures:
    """Características semânticas de um paper."""
    tfidf_vector: Optional[np.ndarray] = None
    key_concepts: List[str] = None
    methodology_terms: List[str] = None
    novel_terminology: List[str] = None
    citation_context: List[str] = None
    
    def __post_init__(self):
        if self.key_concepts is None:
            self.key_concepts = []
        if self.methodology_terms is None:
            self.methodology_terms = []
        if self.novel_terminology is None:
            self.novel_terminology = []
        if self.citation_context is None:
            self.citation_context = []


@dataclass
class CitationNetwork:
    """Rede de citações para análise de novidade."""
    paper_citations: Dict[str, List[str]]
    citation_graph: Optional[Any] = None  # networkx.Graph se disponível
    centrality_scores: Dict[str, float] = None
    novelty_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.paper_citations is None:
            self.paper_citations = {}
        if self.centrality_scores is None:
            self.centrality_scores = {}
        if self.novelty_scores is None:
            self.novelty_scores = {}


@dataclass
class TemporalTrends:
    """Trends temporais para detecção de novidade."""
    keyword_evolution: Dict[str, List[Tuple[datetime, float]]]
    concept_emergence: Dict[str, datetime]
    methodology_shifts: Dict[str, List[Tuple[datetime, float]]]
    domain_activity: Dict[ScientificDomain, List[Tuple[datetime, int]]]
    
    def __post_init__(self):
        if self.keyword_evolution is None:
            self.keyword_evolution = {}
        if self.concept_emergence is None:
            self.concept_emergence = {}
        if self.methodology_shifts is None:
            self.methodology_shifts = {}
        if self.domain_activity is None:
            self.domain_activity = {}


# =============================================================================
# NOVELTY DETECTOR CLASS
# =============================================================================

class NoveltyDetector:
    """
    Detector avançado de novidades científicas usando múltiplas métricas.
    """
    
    def __init__(self, threshold_config: Optional[NoveltyThreshold] = None):
        """
        Inicializa o detector de novidade.
        
        Args:
            threshold_config: Configuração de thresholds personalizados
        """
        self.thresholds = threshold_config or NoveltyThreshold()
        
        # Caches e histórico
        self.paper_cache: Dict[str, PaperMetadata] = {}
        self.semantic_cache: Dict[str, SemanticFeatures] = {}
        self.citation_network = CitationNetwork(paper_citations={})
        self.temporal_trends = TemporalTrends(
            keyword_evolution={},
            concept_emergence={},
            methodology_shifts={},
            domain_activity={}
        )
        
        # Vocabulários especializados por domínio
        self.domain_vocabularies = self._initialize_domain_vocabularies()
        
        # TF-IDF vectorizer global
        self.tfidf_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8
            )
        
        # Estatísticas
        self.stats = {
            'papers_analyzed': 0,
            'novel_papers_detected': 0,
            'breakthrough_papers': 0,
            'last_analysis': None,
            'avg_novelty_score': 0.0
        }
        
        logger.info(f"Novelty Detector initialized - sklearn: {SKLEARN_AVAILABLE}, networkx: {NETWORKX_AVAILABLE}")
    
    def _initialize_domain_vocabularies(self) -> Dict[ScientificDomain, Dict[str, List[str]]]:
        """Inicializa vocabulários especializados por domínio científico."""
        return {
            ScientificDomain.BIOMATERIALS: {
                'materials': ['hydrogel', 'nanofiber', 'scaffold', 'polymer', 'ceramic', 'composite', 'bioactive'],
                'methods': ['electrospinning', '3d printing', 'sol-gel', 'crosslinking', 'functionalization'],
                'properties': ['biocompatibility', 'biodegradability', 'mechanical', 'porosity', 'surface'],
                'applications': ['tissue engineering', 'drug delivery', 'implant', 'regenerative', 'wound healing']
            },
            ScientificDomain.NEUROSCIENCE: {
                'structures': ['neuron', 'synapse', 'dendrite', 'axon', 'cortex', 'hippocampus', 'cerebellum'],
                'methods': ['fmri', 'eeg', 'optogenetics', 'electrophysiology', 'calcium imaging', 'patch-clamp'],
                'processes': ['plasticity', 'learning', 'memory', 'cognition', 'consciousness', 'attention'],
                'disorders': ['alzheimer', 'parkinson', 'depression', 'schizophrenia', 'autism', 'epilepsy']
            },
            ScientificDomain.PHILOSOPHY: {
                'branches': ['epistemology', 'metaphysics', 'ethics', 'logic', 'aesthetics', 'phenomenology'],
                'concepts': ['consciousness', 'knowledge', 'reality', 'truth', 'existence', 'identity', 'causation'],
                'methods': ['argument', 'proof', 'dialectic', 'analysis', 'synthesis', 'deduction', 'induction'],
                'problems': ['mind-body', 'free will', 'personal identity', 'meaning of life', 'moral responsibility']
            },
            ScientificDomain.QUANTUM_MECHANICS: {
                'phenomena': ['entanglement', 'superposition', 'decoherence', 'tunneling', 'interference'],
                'particles': ['qubit', 'photon', 'electron', 'boson', 'fermion', 'particle'],
                'methods': ['quantum computing', 'quantum cryptography', 'quantum simulation', 'quantum sensing'],
                'theories': ['copenhagen interpretation', 'many-worlds', 'hidden variables', 'quantum field theory']
            },
            ScientificDomain.MATHEMATICS: {
                'areas': ['algebra', 'geometry', 'topology', 'analysis', 'number theory', 'combinatorics'],
                'objects': ['manifold', 'group', 'ring', 'field', 'vector space', 'graph', 'function'],
                'methods': ['proof', 'theorem', 'lemma', 'algorithm', 'optimization', 'approximation'],
                'properties': ['continuity', 'differentiability', 'integrability', 'convergence', 'stability']
            }
        }
    
    def _extract_key_concepts(self, text: str, domain: Optional[ScientificDomain] = None) -> List[str]:
        """Extrai conceitos-chave do texto baseado no domínio."""
        if not text:
            return []
        
        text_lower = text.lower()
        concepts = set()
        
        # Conceitos gerais (independente de domínio)
        general_patterns = [
            r'\b(novel|new|innovative|breakthrough|paradigm|revolutionary)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
            r'\b(first|initial|pioneer|unprecedented)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',
            r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:approach|method|technique|strategy|framework)\b'
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.add(match[1])
                else:
                    concepts.add(match)
        
        # Conceitos específicos do domínio
        if domain and domain in self.domain_vocabularies:
            vocab = self.domain_vocabularies[domain]
            for category, terms in vocab.items():
                for term in terms:
                    if term in text_lower:
                        concepts.add(term)
        
        return list(concepts)
    
    def _extract_methodology_terms(self, text: str, domain: Optional[ScientificDomain] = None) -> List[str]:
        """Extrai termos metodológicos do texto."""
        if not text:
            return []
        
        text_lower = text.lower()
        methodology_terms = set()
        
        # Padrões metodológicos gerais
        method_patterns = [
            r'\b(?:using|via|through|by means of|employing)\s+([a-z]+(?:\s+[a-z]+){0,3})\b',
            r'\b([a-z]+(?:\s+[a-z]+){0,2})\s+(?:analysis|measurement|characterization|evaluation)\b',
            r'\b(?:performed|conducted|carried out)\s+([a-z]+(?:\s+[a-z]+){0,2})\b'
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) > 2:  # Evitar termos muito curtos
                    methodology_terms.add(match)
        
        # Métodos específicos do domínio
        if domain and domain in self.domain_vocabularies:
            vocab = self.domain_vocabularies[domain]
            if 'methods' in vocab:
                for method in vocab['methods']:
                    if method in text_lower:
                        methodology_terms.add(method)
        
        return list(methodology_terms)
    
    def _detect_novel_terminology(self, text: str, historical_terms: Set[str]) -> List[str]:
        """Detecta terminologia potencialmente nova."""
        if not text:
            return []
        
        # Extrair termos técnicos (palavras com padrões específicos)
        technical_patterns = [
            r'\b[A-Z][a-z]+-[A-Z][a-z]+\b',  # Termos compostos com hífen
            r'\b[a-z]+(?:-[a-z]+)+\b',       # Termos multi-hifenizados
            r'\b[a-z]{3,}[A-Z][a-z]{2,}\b',  # CamelCase
            r'\b(?:[A-Z]{2,}|[a-z]+)\d+\b', # Códigos alfanuméricos
        ]
        
        novel_terms = set()
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match.lower() not in historical_terms and len(match) > 3:
                    novel_terms.add(match.lower())
        
        return list(novel_terms)
    
    def _compute_semantic_similarity(self, paper1: PaperMetadata, paper2: PaperMetadata) -> float:
        """Computa similaridade semântica entre dois papers."""
        if not SKLEARN_AVAILABLE:
            return self._simple_similarity(paper1, paper2)
        
        try:
            # Concatenar título e abstract
            text1 = f"{paper1.title} {paper1.abstract}".lower()
            text2 = f"{paper2.title} {paper2.abstract}".lower()
            
            # Vectorização TF-IDF
            if self.tfidf_vectorizer is None:
                return 0.0
            
            # Criar corpus temporário
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calcular similaridade cosseno
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return float(similarity_matrix[0, 1])
            
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return self._simple_similarity(paper1, paper2)
    
    def _simple_similarity(self, paper1: PaperMetadata, paper2: PaperMetadata) -> float:
        """Similaridade simples baseada em palavras-chave e conceitos."""
        # Extrair palavras de títulos e abstracts
        def extract_words(text):
            return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        
        words1 = extract_words(f"{paper1.title} {paper1.abstract}")
        words2 = extract_words(f"{paper2.title} {paper2.abstract}")
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_citation_novelty(self, paper: PaperMetadata, historical_papers: List[PaperMetadata]) -> float:
        """Computa novidade baseada em padrões de citação."""
        if not NETWORKX_AVAILABLE or not historical_papers:
            return 0.5  # Valor neutro sem dados
        
        try:
            # Simular análise de citações baseada em similaridade
            unique_connections = 0
            total_connections = 0
            
            for hist_paper in historical_papers[-100:]:  # Limitar a análise recente
                similarity = self._compute_semantic_similarity(paper, hist_paper)
                
                if similarity > 0.1:  # Threshold mínimo para conexão
                    total_connections += 1
                    if similarity < 0.3:  # Conexão com paper diferente
                        unique_connections += 1
            
            if total_connections == 0:
                return 1.0  # Paper completamente isolado = muito novel
            
            return unique_connections / total_connections
            
        except Exception as e:
            logger.warning(f"Citation novelty computation failed: {e}")
            return 0.5
    
    def _compute_keyword_novelty(self, paper: PaperMetadata, historical_keywords: Set[str]) -> float:
        """Computa novidade baseada em emergência de keywords."""
        # Extrair keywords do paper atual
        paper_text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
        current_keywords = set(re.findall(r'\b[a-z]{3,}\b', paper_text))
        
        # Calcular proporção de keywords novas
        if not current_keywords:
            return 0.0
        
        novel_keywords = current_keywords - historical_keywords
        novelty_ratio = len(novel_keywords) / len(current_keywords)
        
        # Bonus para termos técnicos complexos
        technical_bonus = 0.0
        for keyword in novel_keywords:
            if len(keyword) > 8 or '-' in keyword or keyword.count('_') > 0:
                technical_bonus += 0.1
        
        return min(1.0, novelty_ratio + technical_bonus)
    
    def _compute_temporal_significance(self, paper: PaperMetadata, recent_papers: List[PaperMetadata]) -> float:
        """Computa significância temporal baseada em trends emergentes."""
        if not paper.publication_date or not recent_papers:
            return 0.5
        
        # Analisar papers dos últimos 6 meses
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=180)
        recent = [p for p in recent_papers if p.publication_date and p.publication_date > cutoff_date]
        
        if not recent:
            return 0.8  # Paper em área pouco ativa = potencialmente significativo
        
        # Extrair conceitos do paper atual
        paper_concepts = set(self._extract_key_concepts(
            f"{paper.title} {paper.abstract}", 
            paper.domain
        ))
        
        # Calcular raridade dos conceitos em papers recentes
        concept_frequencies = Counter()
        for rp in recent:
            recent_concepts = self._extract_key_concepts(
                f"{rp.title} {rp.abstract}", 
                rp.domain
            )
            concept_frequencies.update(recent_concepts)
        
        # Calcular score de raridade
        rarity_scores = []
        for concept in paper_concepts:
            frequency = concept_frequencies.get(concept, 0)
            rarity = 1.0 / (1.0 + frequency)  # Mais raro = score maior
            rarity_scores.append(rarity)
        
        return np.mean(rarity_scores) if rarity_scores else 0.5
    
    def _detect_methodological_innovations(self, paper: PaperMetadata, historical_methods: Set[str]) -> List[str]:
        """Detecta inovações metodológicas."""
        current_methods = set(self._extract_methodology_terms(
            f"{paper.title} {paper.abstract}",
            paper.domain
        ))
        
        # Métodos potencialmente novos
        novel_methods = current_methods - historical_methods
        
        # Filtrar por relevância
        significant_methods = []
        for method in novel_methods:
            # Critérios: comprimento, complexidade, padrões específicos
            if (len(method) > 5 or 
                '-' in method or 
                'micro' in method or 
                'nano' in method or 
                '3d' in method or
                'ai' in method or
                'machine learning' in method):
                significant_methods.append(method)
        
        return significant_methods
    
    def _detect_conceptual_innovations(self, paper: PaperMetadata, historical_concepts: Set[str]) -> List[str]:
        """Detecta inovações conceituais."""
        current_concepts = set(self._extract_key_concepts(
            f"{paper.title} {paper.abstract}",
            paper.domain
        ))
        
        # Conceitos potencialmente novos
        novel_concepts = current_concepts - historical_concepts
        
        # Filtrar conceitos significativos
        significant_concepts = []
        innovation_indicators = [
            'novel', 'new', 'innovative', 'breakthrough', 'paradigm',
            'revolutionary', 'unprecedented', 'first-time', 'unique'
        ]
        
        paper_text = f"{paper.title} {paper.abstract}".lower()
        
        for concept in novel_concepts:
            # Verificar se o conceito está associado a indicadores de inovação
            for indicator in innovation_indicators:
                if f"{indicator} {concept}" in paper_text or f"{concept} {indicator}" in paper_text:
                    significant_concepts.append(concept)
                    break
        
        return significant_concepts
    
    def _classify_novelty_level(self, overall_score: float) -> NoveltyLevel:
        """Classifica nível de novidade baseado no score geral."""
        if overall_score >= 0.9:
            return NoveltyLevel.REVOLUTIONARY
        elif overall_score >= 0.8:
            return NoveltyLevel.HIGH
        elif overall_score >= 0.6:
            return NoveltyLevel.MEDIUM
        else:
            return NoveltyLevel.LOW
    
    async def analyze_paper_novelty_async(
        self, 
        paper: PaperMetadata, 
        historical_context: Optional[List[PaperMetadata]] = None
    ) -> NoveltyAnalysisResult:
        """Análise assíncrona de novidade de um paper."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.analyze_paper_novelty, paper, historical_context
        )
    
    def analyze_paper_novelty(
        self, 
        paper: PaperMetadata, 
        historical_context: Optional[List[PaperMetadata]] = None
    ) -> NoveltyAnalysisResult:
        """
        Analisa a novidade científica de um paper.
        
        Args:
            paper: Metadados do paper a analisar
            historical_context: Lista de papers históricos para comparação
            
        Returns:
            Resultado completo da análise de novidade
        """
        try:
            logger.debug(f"Analyzing novelty for paper: {paper.title[:50]}...")
            
            # Contexto histórico
            historical_context = historical_context or list(self.paper_cache.values())
            
            # Extrair características históricas
            historical_keywords = set()
            historical_concepts = set()
            historical_methods = set()
            
            for hp in historical_context:
                hist_text = f"{hp.title} {hp.abstract} {' '.join(hp.keywords)}".lower()
                historical_keywords.update(re.findall(r'\b[a-z]{3,}\b', hist_text))
                historical_concepts.update(self._extract_key_concepts(hist_text, hp.domain))
                historical_methods.update(self._extract_methodology_terms(hist_text, hp.domain))
            
            # 1. ANÁLISE SEMÂNTICA
            semantic_scores = []
            if historical_context:
                for hp in historical_context[-50:]:  # Limitar para performance
                    similarity = self._compute_semantic_similarity(paper, hp)
                    semantic_scores.append(similarity)
            
            semantic_score = 1.0 - (np.mean(semantic_scores) if semantic_scores else 0.0)
            semantic_score = max(0.0, min(1.0, semantic_score))
            
            # 2. ANÁLISE DE CITAÇÕES
            citation_score = self._compute_citation_novelty(paper, historical_context)
            
            # 3. ANÁLISE DE KEYWORDS
            keyword_score = self._compute_keyword_novelty(paper, historical_keywords)
            
            # 4. ANÁLISE TEMPORAL
            temporal_score = self._compute_temporal_significance(paper, historical_context)
            
            # 5. SCORE GERAL PONDERADO
            weights = {
                'semantic': 0.3,
                'citation': 0.25,
                'keyword': 0.25,
                'temporal': 0.2
            }
            
            overall_score = (
                weights['semantic'] * semantic_score +
                weights['citation'] * citation_score +
                weights['keyword'] * keyword_score +
                weights['temporal'] * temporal_score
            )
            
            # 6. DETECÇÃO DE INOVAÇÕES
            methodological_innovations = self._detect_methodological_innovations(paper, historical_methods)
            conceptual_innovations = self._detect_conceptual_innovations(paper, historical_concepts)
            
            detected_innovations = methodological_innovations + conceptual_innovations
            
            # 7. NOVELTY SCORES ESPECÍFICAS
            methodology_novelty = len(methodological_innovations) / max(1, len(historical_methods)) * 10
            methodology_novelty = min(1.0, methodology_novelty)
            
            conceptual_novelty = len(conceptual_innovations) / max(1, len(historical_concepts)) * 10  
            conceptual_novelty = min(1.0, conceptual_novelty)
            
            # 8. CLASSIFICAÇÃO E JUSTIFICATIVA
            novelty_level = self._classify_novelty_level(overall_score)
            
            justification_parts = []
            if semantic_score > self.thresholds.semantic_similarity:
                justification_parts.append(f"High semantic novelty (score: {semantic_score:.3f})")
            if citation_score > self.thresholds.citation_uniqueness:
                justification_parts.append(f"Unique citation patterns (score: {citation_score:.3f})")
            if keyword_score > self.thresholds.keyword_emergence:
                justification_parts.append(f"Novel terminology detected (score: {keyword_score:.3f})")
            if temporal_score > self.thresholds.temporal_significance:
                justification_parts.append(f"Temporally significant (score: {temporal_score:.3f})")
            if detected_innovations:
                justification_parts.append(f"Innovations detected: {', '.join(detected_innovations[:3])}")
            
            justification = "; ".join(justification_parts) if justification_parts else "Standard research contribution"
            
            # Criar resultado
            result = NoveltyAnalysisResult(
                paper_id=paper.doc_id,
                novelty_level=novelty_level,
                semantic_score=semantic_score,
                citation_score=citation_score,
                keyword_score=keyword_score,
                temporal_score=temporal_score,
                overall_score=overall_score,
                justification=justification,
                detected_innovations=detected_innovations,
                methodology_novelty=methodology_novelty,
                conceptual_novelty=conceptual_novelty
            )
            
            # Cache do paper e atualizar estatísticas
            self.paper_cache[paper.doc_id] = paper
            self.stats['papers_analyzed'] += 1
            if novelty_level in [NoveltyLevel.HIGH, NoveltyLevel.REVOLUTIONARY]:
                self.stats['novel_papers_detected'] += 1
            if novelty_level == NoveltyLevel.REVOLUTIONARY:
                self.stats['breakthrough_papers'] += 1
            
            self.stats['avg_novelty_score'] = (
                (self.stats['avg_novelty_score'] * (self.stats['papers_analyzed'] - 1) + overall_score) / 
                self.stats['papers_analyzed']
            )
            self.stats['last_analysis'] = datetime.now(timezone.utc)
            
            logger.info(f"Novelty analysis completed for '{paper.title[:30]}...': {novelty_level.value} ({overall_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Novelty analysis failed for paper {paper.doc_id}: {e}")
            # Retornar resultado padrão em caso de erro
            return NoveltyAnalysisResult(
                paper_id=paper.doc_id,
                novelty_level=NoveltyLevel.LOW,
                semantic_score=0.0,
                citation_score=0.0,
                keyword_score=0.0,
                temporal_score=0.0,
                overall_score=0.0,
                justification=f"Analysis failed: {str(e)}",
                detected_innovations=[]
            )
    
    async def batch_analyze_novelty(
        self, 
        papers: List[PaperMetadata],
        historical_context: Optional[List[PaperMetadata]] = None
    ) -> List[NoveltyAnalysisResult]:
        """Análise em lote de novidade para múltiplos papers."""
        tasks = [
            self.analyze_paper_novelty_async(paper, historical_context) 
            for paper in papers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar exceções
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for paper {papers[i].doc_id}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def detect_emerging_trends(
        self, 
        recent_papers: List[PaperMetadata],
        time_window_days: int = 90
    ) -> List[EmergingTrend]:
        """Detecta trends emergentes baseado em papers recentes."""
        if not recent_papers:
            return []
        
        # Filtrar papers por janela temporal
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        filtered_papers = [
            p for p in recent_papers 
            if p.publication_date and p.publication_date > cutoff_date
        ]
        
        if len(filtered_papers) < 5:  # Mínimo para detectar trends
            return []
        
        trends = []
        
        try:
            # Agrupar por domínio
            domain_papers = defaultdict(list)
            for paper in filtered_papers:
                domain_papers[paper.domain].append(paper)
            
            # Analisar trends por domínio
            for domain, papers in domain_papers.items():
                if len(papers) < 3:
                    continue
                
                # Extrair conceitos temporalmente
                concept_timeline = defaultdict(list)
                
                for paper in papers:
                    concepts = self._extract_key_concepts(
                        f"{paper.title} {paper.abstract}",
                        domain
                    )
                    pub_date = paper.publication_date or datetime.now(timezone.utc)
                    
                    for concept in concepts:
                        concept_timeline[concept].append(pub_date)
                
                # Identificar conceitos emergentes
                for concept, dates in concept_timeline.items():
                    if len(dates) >= 2:  # Mínimo de ocorrências
                        # Calcular growth rate
                        dates.sort()
                        days_span = (dates[-1] - dates[0]).days
                        
                        if days_span > 0:
                            growth_rate = len(dates) / days_span
                            
                            # Threshold para trend emergente
                            if growth_rate > 0.02:  # Ajustável
                                trend = EmergingTrend(
                                    trend_id=f"{domain.value}_{concept}_{int(time.time())}",
                                    domain=domain,
                                    trend_name=concept.title(),
                                    description=f"Emerging concept '{concept}' in {domain.value}",
                                    confidence_score=min(1.0, growth_rate * 10),
                                    supporting_papers=[p.doc_id for p in papers if concept in f"{p.title} {p.abstract}".lower()],
                                    emergence_date=dates[0],
                                    growth_rate=growth_rate,
                                    related_keywords=[concept]
                                )
                                trends.append(trend)
            
            # Ordenar por confidence score
            trends.sort(key=lambda t: t.confidence_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Emerging trends detection failed: {e}")
        
        return trends[:10]  # Top 10 trends
    
    def get_novelty_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do detector de novidade."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Limpa caches do detector."""
        self.paper_cache.clear()
        self.semantic_cache.clear()
        logger.info("Novelty detector cache cleared")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Instância global do Novelty Detector
_novelty_detector = None

def get_novelty_detector(threshold_config: Optional[NoveltyThreshold] = None) -> NoveltyDetector:
    """Retorna instância singleton do Novelty Detector."""
    global _novelty_detector
    if _novelty_detector is None:
        _novelty_detector = NoveltyDetector(threshold_config)
    return _novelty_detector