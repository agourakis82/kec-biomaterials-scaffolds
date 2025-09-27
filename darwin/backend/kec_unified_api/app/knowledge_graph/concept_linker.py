"""DARWIN Concept Linker - Sistema de Linking Interdisciplinar Avançado

Sistema épico que identifica e cria conexões semânticas automáticas entre conceitos
de diferentes domínios científicos (biomaterials, neuroscience, philosophy, quantum, psychiatry).
Utiliza múltiplos algoritmos de similaridade, análise semântica e knowledge graph embeddings.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes,
    ConceptNode, SimilarityEdge, BridgeEdge,
    GraphNodeBase
)

logger = get_logger("knowledge_graph.concept_linker")


# ==================== INTERDISCIPLINARY CONCEPT MAPPINGS ====================

CROSS_DOMAIN_CONCEPT_MAPPINGS = {
    # Biomaterials ↔ Neuroscience
    ("biomaterials", "neuroscience"): {
        "scaffold": ["neural_network_architecture", "synaptic_structure"],
        "biocompatibility": ["neural_compatibility", "synaptic_integration"],
        "surface_modification": ["neural_interface", "receptor_binding"],
        "mechanical_properties": ["neural_elasticity", "membrane_mechanics"],
        "porous_structure": ["neural_connectivity", "synaptic_gaps"],
        "degradation": ["neural_plasticity", "synaptic_pruning"],
        "cell_adhesion": ["neural_adhesion", "synaptic_adhesion"],
        "tissue_engineering": ["neural_engineering", "brain_organoids"]
    },
    
    # Neuroscience ↔ Philosophy  
    ("neuroscience", "philosophy"): {
        "consciousness": ["neural_correlates", "subjective_experience"],
        "free_will": ["neural_determinism", "agency"],
        "memory": ["knowledge", "mental_representation"],
        "perception": ["phenomenology", "qualia"],
        "cognition": ["mind", "rationality"],
        "neural_networks": ["connectionism", "mental_architecture"],
        "plasticity": ["learning", "adaptation"],
        "attention": ["consciousness", "awareness"]
    },
    
    # Philosophy ↔ Quantum Mechanics
    ("philosophy", "quantum_mechanics"): {
        "consciousness": ["quantum_consciousness", "observer_effect"],
        "reality": ["quantum_reality", "measurement_problem"],
        "causality": ["quantum_causation", "nonlocality"],
        "determinism": ["quantum_indeterminacy", "uncertainty_principle"],
        "observation": ["quantum_measurement", "wave_function_collapse"],
        "knowledge": ["quantum_information", "quantum_epistemology"],
        "existence": ["quantum_ontology", "superposition"],
        "identity": ["quantum_identity", "entanglement"]
    },
    
    # Quantum Mechanics ↔ Biomaterials
    ("quantum_mechanics", "biomaterials"): {
        "quantum_coherence": ["biomolecular_coherence", "protein_folding"],
        "entanglement": ["biomolecular_entanglement", "enzyme_catalysis"],
        "superposition": ["biomolecular_states", "conformational_states"],
        "quantum_tunneling": ["biological_tunneling", "electron_transport"],
        "wave_function": ["molecular_orbitals", "electronic_structure"],
        "decoherence": ["biological_decoherence", "thermal_effects"],
        "quantum_information": ["biological_information", "genetic_code"],
        "quantum_field": ["bioelectric_fields", "electromagnetic_biology"]
    },
    
    # Psychiatry ↔ Neuroscience
    ("psychiatry", "neuroscience"): {
        "mental_disorder": ["neural_dysfunction", "brain_pathology"],
        "depression": ["neurotransmitter_imbalance", "neural_circuits"],
        "anxiety": ["amygdala_hyperactivity", "fear_circuits"],
        "schizophrenia": ["dopamine_systems", "neural_connectivity"],
        "therapy": ["neural_plasticity", "brain_stimulation"],
        "medication": ["neuropharmacology", "receptor_targeting"],
        "cognitive_function": ["executive_networks", "prefrontal_cortex"],
        "emotional_regulation": ["limbic_system", "neural_emotion_circuits"]
    },
    
    # Mathematics conecta todos os domínios
    ("mathematics", "biomaterials"): {
        "graph_theory": ["network_topology", "scaffold_architecture"],
        "optimization": ["material_design", "property_optimization"],
        "topology": ["pore_topology", "surface_topology"],
        "statistics": ["material_characterization", "property_analysis"],
        "differential_equations": ["diffusion_models", "degradation_kinetics"],
        "linear_algebra": ["mechanical_tensors", "stress_analysis"]
    },
    
    ("mathematics", "neuroscience"): {
        "graph_theory": ["neural_networks", "connectome"],
        "dynamic_systems": ["neural_dynamics", "oscillations"],
        "information_theory": ["neural_coding", "mutual_information"],
        "probability": ["neural_computation", "bayesian_brain"],
        "topology": ["brain_topology", "neural_manifolds"],
        "optimization": ["learning_algorithms", "synaptic_optimization"]
    },
    
    ("mathematics", "philosophy"): {
        "logic": ["formal_logic", "reasoning"],
        "set_theory": ["conceptual_categories", "classification"],
        "probability": ["epistemic_uncertainty", "belief_updating"],
        "game_theory": ["rational_choice", "decision_theory"],
        "topology": ["conceptual_spaces", "meaning_structure"],
        "information_theory": ["knowledge_representation", "semantic_information"]
    }
}

# Conceitos universais que aparecem em múltiplos domínios
UNIVERSAL_CONCEPTS = {
    "network", "structure", "function", "dynamics", "information", 
    "system", "complexity", "emergence", "pattern", "organization",
    "hierarchy", "interaction", "regulation", "adaptation", "evolution",
    "optimization", "stability", "feedback", "control", "signal",
    "noise", "entropy", "order", "symmetry", "phase_transition"
}

# Padrões textuais para identificação de conceitos relacionados
SEMANTIC_PATTERNS = {
    "causation": [r"\bcause[sd]?\b", r"\bleads? to\b", r"\bresults? in\b", r"\binduces?\b"],
    "correlation": [r"\bcorrelat\w+", r"\bassociat\w+", r"\blink\w+", r"\bconnect\w+"],
    "similarity": [r"\bsimilar\b", r"\banalogous\b", r"\bcomparable\b", r"\bresemble\w+"],
    "opposition": [r"\boppos\w+", r"\bcontrast\w+", r"\bdiff\w+", r"\bunlike\b"],
    "temporal": [r"\bbefore\b", r"\bafter\b", r"\bduring\b", r"\bwhile\b", r"\bthen\b"],
    "spatial": [r"\babove\b", r"\bbelow\b", r"\bnear\b", r"\bwithin\b", r"\bbetween\b"]
}


@dataclass
class LinkingConfiguration:
    """Configuração do sistema de linking."""
    semantic_similarity_threshold: float = 0.6
    cross_domain_bonus: float = 0.2
    universal_concept_bonus: float = 0.15
    pattern_match_bonus: float = 0.1
    frequency_weight: float = 0.05
    enable_clustering: bool = True
    clustering_eps: float = 0.3
    clustering_min_samples: int = 2
    max_links_per_concept: int = 20
    enable_embeddings: bool = True
    embedding_dimension: int = 300


class InterdisciplinaryConceptLinker:
    """
    Sistema avançado de linking automático entre conceitos interdisciplinares.
    
    Funcionalidades:
    - Semantic similarity usando múltiplos algoritmos
    - Cross-domain concept mapping automático
    - Universal concept detection
    - Pattern-based relationship extraction
    - Concept clustering e community detection
    - Knowledge graph embeddings
    - Temporal relationship tracking
    """
    
    def __init__(self, config: Optional[LinkingConfiguration] = None):
        self.config = config or LinkingConfiguration()
        
        # TF-IDF para similaridade semântica básica
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Cache para embeddings e similaridades
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.concept_clusters: Dict[str, int] = {}
        
        # Estatísticas de linking
        self.linking_stats = {
            "total_links_created": 0,
            "cross_domain_links": 0,
            "semantic_links": 0,
            "pattern_links": 0,
            "universal_links": 0
        }
        
        logger.info("🔗 Interdisciplinary Concept Linker initialized")
    
    # ==================== MAIN LINKING METHODS ====================
    
    async def create_concept_links(
        self, 
        concepts: List[ConceptNode],
        existing_links: Optional[List[SimilarityEdge]] = None
    ) -> List[Union[SimilarityEdge, BridgeEdge]]:
        """
        Cria links automáticos entre conceitos usando múltiplos algoritmos.
        """
        logger.info(f"🔗 Creating concept links for {len(concepts)} concepts")
        link_start = datetime.utcnow()
        
        try:
            # Preparar dados dos conceitos
            concept_data = await self._prepare_concept_data(concepts)
            
            # Múltiplas estratégias de linking em paralelo
            linking_tasks = [
                self._create_semantic_links(concept_data),
                self._create_cross_domain_links(concept_data),
                self._create_universal_concept_links(concept_data),
                self._create_pattern_based_links(concept_data),
            ]
            
            # Executar linking strategies
            all_links = []
            link_results = await asyncio.gather(*linking_tasks, return_exceptions=True)
            
            for i, result in enumerate(link_results):
                if isinstance(result, Exception):
                    logger.warning(f"Linking strategy {i} failed: {result}")
                    continue
                if isinstance(result, list):
                    all_links.extend(result)
            
            # Aplicar clustering se habilitado
            if self.config.enable_clustering:
                cluster_links = await self._create_cluster_based_links(concept_data)
                all_links.extend(cluster_links)
            
            # Filtrar e rankear links
            final_links = await self._filter_and_rank_links(all_links, existing_links)
            
            # Atualizar estatísticas
            await self._update_linking_stats(final_links)
            
            link_time = (datetime.utcnow() - link_start).total_seconds()
            logger.info(f"✅ Created {len(final_links)} concept links in {link_time:.2f}s")
            
            return final_links
            
        except Exception as e:
            logger.error(f"❌ Concept linking failed: {e}")
            return []
    
    async def _prepare_concept_data(
        self, 
        concepts: List[ConceptNode]
    ) -> Dict[str, Any]:
        """Prepara dados dos conceitos para linking."""
        concept_texts = []
        concept_domains = []
        concept_frequencies = []
        
        for concept in concepts:
            # Combinar label e descrição para texto completo
            text = f"{concept.label} {concept.description or ''}"
            concept_texts.append(text.lower())
            concept_domains.append(concept.domain)
            concept_frequencies.append(getattr(concept, 'frequency', 1))
        
        # Calcular TF-IDF embeddings se não existirem
        if self.config.enable_embeddings and len(concept_texts) > 1:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(concept_texts)
                
                # Redução de dimensionalidade se necessário
                if tfidf_matrix.shape[1] > self.config.embedding_dimension:
                    pca = PCA(n_components=self.config.embedding_dimension)
                    embeddings = pca.fit_transform(tfidf_matrix.toarray())
                else:
                    embeddings = tfidf_matrix.toarray()
                
                # Cache embeddings
                for i, concept in enumerate(concepts):
                    self.embeddings_cache[concept.id] = embeddings[i]
                    
            except Exception as e:
                logger.warning(f"Failed to compute embeddings: {e}")
                embeddings = None
        else:
            embeddings = None
        
        return {
            "concepts": concepts,
            "texts": concept_texts,
            "domains": concept_domains,
            "frequencies": concept_frequencies,
            "embeddings": embeddings,
            "tfidf_matrix": getattr(self, 'tfidf_matrix', None)
        }
    
    # ==================== SEMANTIC SIMILARITY LINKING ====================
    
    async def _create_semantic_links(
        self, 
        concept_data: Dict[str, Any]
    ) -> List[SimilarityEdge]:
        """Cria links baseados em similaridade semântica."""
        concepts = concept_data["concepts"]
        texts = concept_data["texts"]
        
        if len(concepts) < 2:
            return []
        
        semantic_links = []
        
        try:
            # Calcular matriz de similaridade TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Criar links baseados em similaridade
            for i, concept_i in enumerate(concepts):
                for j, concept_j in enumerate(concepts[i+1:], i+1):
                    similarity = similarity_matrix[i][j]
                    
                    # Aplicar threshold e bônus cross-domain
                    adjusted_similarity = similarity
                    
                    # Bônus para links cross-domain
                    if concept_i.domain != concept_j.domain:
                        adjusted_similarity += self.config.cross_domain_bonus
                    
                    if adjusted_similarity >= self.config.semantic_similarity_threshold:
                        edge = SimilarityEdge(
                            id=f"semantic_{concept_i.id}_{concept_j.id}",
                            source=concept_i.id,
                            target=concept_j.id,
                            weight=adjusted_similarity,
                            similarity_score=similarity,
                            similarity_method="tfidf_cosine_semantic",
                            properties={
                                "base_similarity": similarity,
                                "cross_domain_bonus": self.config.cross_domain_bonus if concept_i.domain != concept_j.domain else 0.0,
                                "domains": [concept_i.domain.value, concept_j.domain.value]
                            }
                        )
                        semantic_links.append(edge)
        
        except Exception as e:
            logger.warning(f"Semantic linking failed: {e}")
            return []
        
        self.linking_stats["semantic_links"] += len(semantic_links)
        logger.info(f"🧠 Created {len(semantic_links)} semantic links")
        return semantic_links
    
    # ==================== CROSS-DOMAIN MAPPING LINKING ====================
    
    async def _create_cross_domain_links(
        self, 
        concept_data: Dict[str, Any]
    ) -> List[BridgeEdge]:
        """Cria links baseados em mapeamentos cross-domain predefinidos."""
        concepts = concept_data["concepts"]
        cross_domain_links = []
        
        # Agrupar conceitos por domínio
        concepts_by_domain = defaultdict(list)
        for concept in concepts:
            concepts_by_domain[concept.domain.value].append(concept)
        
        # Aplicar mapeamentos cross-domain
        for (domain1, domain2), mappings in CROSS_DOMAIN_CONCEPT_MAPPINGS.items():
            if domain1 in concepts_by_domain and domain2 in concepts_by_domain:
                domain1_concepts = concepts_by_domain[domain1]
                domain2_concepts = concepts_by_domain[domain2]
                
                for concept1 in domain1_concepts:
                    for concept2 in domain2_concepts:
                        # Verificar se conceitos fazem match com mapeamentos
                        bridge_strength = await self._calculate_mapping_strength(
                            concept1, concept2, mappings
                        )
                        
                        if bridge_strength > 0.3:  # Threshold para criar bridge
                            edge = BridgeEdge(
                                id=f"bridge_{concept1.id}_{concept2.id}",
                                source=concept1.id,
                                target=concept2.id,
                                weight=bridge_strength,
                                domain_from=concept1.domain,
                                domain_to=concept2.domain,
                                bridge_strength=bridge_strength,
                                properties={
                                    "mapping_type": f"{domain1}_{domain2}",
                                    "bridge_concepts": await self._find_matching_mappings(
                                        concept1, concept2, mappings
                                    )
                                }
                            )
                            cross_domain_links.append(edge)
        
        self.linking_stats["cross_domain_links"] += len(cross_domain_links)
        logger.info(f"🌉 Created {len(cross_domain_links)} cross-domain bridge links")
        return cross_domain_links
    
    async def _calculate_mapping_strength(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode, 
        mappings: Dict[str, List[str]]
    ) -> float:
        """Calcula força de mapping entre dois conceitos."""
        concept1_text = f"{concept1.label} {concept1.description or ''}".lower()
        concept2_text = f"{concept2.label} {concept2.description or ''}".lower()
        
        total_strength = 0.0
        matching_mappings = 0
        
        for source_concept, target_concepts in mappings.items():
            # Verificar se concept1 match com source_concept
            if source_concept.replace('_', ' ') in concept1_text:
                for target_concept in target_concepts:
                    # Verificar se concept2 match com target_concept
                    if target_concept.replace('_', ' ') in concept2_text:
                        # Calcular força baseada na especificidade do match
                        source_strength = len(source_concept) / len(concept1_text)
                        target_strength = len(target_concept) / len(concept2_text)
                        mapping_strength = (source_strength + target_strength) / 2
                        total_strength += mapping_strength
                        matching_mappings += 1
        
        return min(total_strength, 1.0) if matching_mappings > 0 else 0.0
    
    async def _find_matching_mappings(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode, 
        mappings: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """Encontra mapeamentos específicos que fazem match."""
        matches = []
        concept1_text = f"{concept1.label} {concept1.description or ''}".lower()
        concept2_text = f"{concept2.label} {concept2.description or ''}".lower()
        
        for source_concept, target_concepts in mappings.items():
            if source_concept.replace('_', ' ') in concept1_text:
                for target_concept in target_concepts:
                    if target_concept.replace('_', ' ') in concept2_text:
                        matches.append({
                            "source": source_concept,
                            "target": target_concept,
                            "concept1": concept1.label,
                            "concept2": concept2.label
                        })
        
        return matches
    
    # ==================== UNIVERSAL CONCEPT LINKING ====================
    
    async def _create_universal_concept_links(
        self, 
        concept_data: Dict[str, Any]
    ) -> List[SimilarityEdge]:
        """Cria links para conceitos universais."""
        concepts = concept_data["concepts"]
        universal_links = []
        
        # Identificar conceitos universais
        universal_concepts = []
        for concept in concepts:
            concept_text = f"{concept.label} {concept.description or ''}".lower()
            
            for universal_term in UNIVERSAL_CONCEPTS:
                if universal_term in concept_text:
                    universal_concepts.append((concept, universal_term))
                    break
        
        # Criar links entre conceitos universais
        for i, (concept_i, term_i) in enumerate(universal_concepts):
            for concept_j, term_j in universal_concepts[i+1:]:
                # Calcular similaridade baseada em termos universais compartilhados
                shared_terms = self._count_shared_universal_terms(concept_i, concept_j)
                
                if shared_terms > 0:
                    # Base similarity + universal bonus
                    base_similarity = min(shared_terms * 0.3, 0.8)
                    final_similarity = base_similarity + self.config.universal_concept_bonus
                    
                    if final_similarity >= self.config.semantic_similarity_threshold:
                        edge = SimilarityEdge(
                            id=f"universal_{concept_i.id}_{concept_j.id}",
                            source=concept_i.id,
                            target=concept_j.id,
                            weight=final_similarity,
                            similarity_score=base_similarity,
                            similarity_method="universal_concepts",
                            properties={
                                "universal_terms": [term_i, term_j],
                                "shared_terms_count": shared_terms,
                                "universal_bonus": self.config.universal_concept_bonus
                            }
                        )
                        universal_links.append(edge)
        
        self.linking_stats["universal_links"] += len(universal_links)
        logger.info(f"🌐 Created {len(universal_links)} universal concept links")
        return universal_links
    
    def _count_shared_universal_terms(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode
    ) -> int:
        """Conta termos universais compartilhados."""
        text1 = f"{concept1.label} {concept1.description or ''}".lower()
        text2 = f"{concept2.label} {concept2.description or ''}".lower()
        
        terms1 = {term for term in UNIVERSAL_CONCEPTS if term in text1}
        terms2 = {term for term in UNIVERSAL_CONCEPTS if term in text2}
        
        return len(terms1 & terms2)
    
    # ==================== PATTERN-BASED LINKING ====================
    
    async def _create_pattern_based_links(
        self, 
        concept_data: Dict[str, Any]
    ) -> List[SimilarityEdge]:
        """Cria links baseados em padrões textuais."""
        concepts = concept_data["concepts"]
        pattern_links = []
        
        for i, concept_i in enumerate(concepts):
            for concept_j in concepts[i+1:]:
                # Analisar padrões entre conceitos
                pattern_strength = await self._analyze_textual_patterns(concept_i, concept_j)
                
                if pattern_strength > 0.3:
                    final_similarity = pattern_strength + self.config.pattern_match_bonus
                    
                    if final_similarity >= self.config.semantic_similarity_threshold:
                        edge = SimilarityEdge(
                            id=f"pattern_{concept_i.id}_{concept_j.id}",
                            source=concept_i.id,
                            target=concept_j.id,
                            weight=final_similarity,
                            similarity_score=pattern_strength,
                            similarity_method="textual_patterns",
                            properties={
                                "pattern_bonus": self.config.pattern_match_bonus,
                                "detected_patterns": await self._get_detected_patterns(concept_i, concept_j)
                            }
                        )
                        pattern_links.append(edge)
        
        self.linking_stats["pattern_links"] += len(pattern_links)
        logger.info(f"🔍 Created {len(pattern_links)} pattern-based links")
        return pattern_links
    
    async def _analyze_textual_patterns(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode
    ) -> float:
        """Analisa padrões textuais entre conceitos."""
        text1 = f"{concept1.label} {concept1.description or ''}".lower()
        text2 = f"{concept2.label} {concept2.description or ''}".lower()
        
        pattern_strength = 0.0
        pattern_count = 0
        
        for pattern_type, patterns in SEMANTIC_PATTERNS.items():
            for pattern in patterns:
                # Verificar se padrão aparece em ambos os textos
                if re.search(pattern, text1) and re.search(pattern, text2):
                    pattern_strength += 0.2
                    pattern_count += 1
                # Verificar se padrão conecta os textos (em contexto maior)
                combined_text = f"{text1} {text2}"
                if re.search(pattern, combined_text):
                    pattern_strength += 0.1
                    pattern_count += 1
        
        # Normalizar pela quantidade de padrões
        return min(pattern_strength / max(pattern_count, 1), 1.0) if pattern_count > 0 else 0.0
    
    async def _get_detected_patterns(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode
    ) -> List[str]:
        """Obtém padrões detectados entre conceitos."""
        text1 = f"{concept1.label} {concept1.description or ''}".lower()
        text2 = f"{concept2.label} {concept2.description or ''}".lower()
        combined = f"{text1} {text2}"
        
        detected = []
        for pattern_type, patterns in SEMANTIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text1) and re.search(pattern, text2):
                    detected.append(f"{pattern_type}_both")
                elif re.search(pattern, combined):
                    detected.append(f"{pattern_type}_combined")
        
        return detected
    
    # ==================== CLUSTERING BASED LINKING ====================
    
    async def _create_cluster_based_links(
        self, 
        concept_data: Dict[str, Any]
    ) -> List[SimilarityEdge]:
        """Cria links baseados em clustering de conceitos."""
        concepts = concept_data["concepts"]
        embeddings = concept_data.get("embeddings")
        
        if embeddings is None or len(concepts) < 3:
            return []
        
        cluster_links = []
        
        try:
            # Aplicar DBSCAN clustering
            clustering = DBSCAN(
                eps=self.config.clustering_eps,
                min_samples=self.config.clustering_min_samples,
                metric='cosine'
            )
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Agrupar conceitos por cluster
            clusters = defaultdict(list)
            for i, (concept, cluster_label) in enumerate(zip(concepts, cluster_labels)):
                if cluster_label != -1:  # Ignorar outliers
                    clusters[cluster_label].append(concept)
                    self.concept_clusters[concept.id] = cluster_label
            
            # Criar links dentro de cada cluster
            for cluster_id, cluster_concepts in clusters.items():
                if len(cluster_concepts) < 2:
                    continue
                
                # Links all-to-all dentro do cluster
                for i, concept_i in enumerate(cluster_concepts):
                    for concept_j in cluster_concepts[i+1:]:
                        # Calcular similaridade dentro do cluster
                        cluster_similarity = await self._calculate_cluster_similarity(
                            concept_i, concept_j, embeddings, concepts
                        )
                        
                        edge = SimilarityEdge(
                            id=f"cluster_{concept_i.id}_{concept_j.id}",
                            source=concept_i.id,
                            target=concept_j.id,
                            weight=cluster_similarity,
                            similarity_score=cluster_similarity,
                            similarity_method="dbscan_clustering",
                            properties={
                                "cluster_id": cluster_id,
                                "cluster_size": len(cluster_concepts)
                            }
                        )
                        cluster_links.append(edge)
            
            logger.info(f"🔗 Created {len(clusters)} clusters with {len(cluster_links)} intra-cluster links")
            
        except Exception as e:
            logger.warning(f"Clustering-based linking failed: {e}")
            return []
        
        return cluster_links
    
    async def _calculate_cluster_similarity(
        self, 
        concept1: ConceptNode, 
        concept2: ConceptNode,
        embeddings: np.ndarray,
        concepts: List[ConceptNode]
    ) -> float:
        """Calcula similaridade específica para conceitos no mesmo cluster."""
        # Encontrar índices dos conceitos
        concept1_idx = next(i for i, c in enumerate(concepts) if c.id == concept1.id)
        concept2_idx = next(i for i, c in enumerate(concepts) if c.id == concept2.id)
        
        # Calcular similaridade cosine
        embedding1 = embeddings[concept1_idx]
        embedding2 = embeddings[concept2_idx]
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Aplicar bônus para cross-domain no mesmo cluster
        if concept1.domain != concept2.domain:
            similarity += self.config.cross_domain_bonus
        
        return min(max(similarity, 0.0), 1.0)
    
    # ==================== FILTERING AND RANKING ====================
    
    async def _filter_and_rank_links(
        self, 
        all_links: List[Union[SimilarityEdge, BridgeEdge]],
        existing_links: Optional[List] = None
    ) -> List[Union[SimilarityEdge, BridgeEdge]]:
        """Filtra e rankeia links finais."""
        # Remover duplicatas baseadas em source-target pairs
        unique_links = {}
        for link in all_links:
            key = (min(link.source, link.target), max(link.source, link.target))
            
            # Manter o link com maior peso
            if key not in unique_links or link.weight > unique_links[key].weight:
                unique_links[key] = link
        
        # Converter para lista e ordenar por peso
        filtered_links = list(unique_links.values())
        filtered_links.sort(key=lambda x: x.weight, reverse=True)
        
        # Limitar número de links por conceito
        concept_link_count = defaultdict(int)
        final_links = []
        
        for link in filtered_links:
            source_count = concept_link_count[link.source]
            target_count = concept_link_count[link.target]
            
            if (source_count < self.config.max_links_per_concept and 
                target_count < self.config.max_links_per_concept):
                
                final_links.append(link)
                concept_link_count[link.source] += 1
                concept_link_count[link.target] += 1
        
        return final_links
    
    async def _update_linking_stats(
        self, 
        links: List[Union[SimilarityEdge, BridgeEdge]]
    ):
        """Atualiza estatísticas de linking."""
        self.linking_stats["total_links_created"] = len(links)
        
        # Contar por tipo
        for link in links:
            if isinstance(link, BridgeEdge):
                self.linking_stats["cross_domain_links"] += 1
            elif hasattr(link, 'similarity_method'):
                if 'semantic' in link.similarity_method:
                    self.linking_stats["semantic_links"] += 1
                elif 'pattern' in link.similarity_method:
                    self.linking_stats["pattern_links"] += 1
                elif 'universal' in link.similarity_method:
                    self.linking_stats["universal_links"] += 1
    
    # ==================== ANALYSIS AND REPORTING ====================
    
    async def analyze_concept_connectivity(
        self, 
        concepts: List[ConceptNode],
        links: List[Union[SimilarityEdge, BridgeEdge]]
    ) -> Dict[str, Any]:
        """Analisa conectividade dos conceitos."""
        # Construir grafo NetworkX
        G = nx.Graph()
        
        # Adicionar nós
        for concept in concepts:
            G.add_node(concept.id, domain=concept.domain.value, label=concept.label)
        
        # Adicionar arestas
        for link in links:
            G.add_edge(link.source, link.target, weight=link.weight)
        
        # Calcular métricas de conectividade
        try:
            analysis = {
                "total_concepts": len(concepts),
                "total_links": len(links),
                "graph_density": nx.density(G) if len(concepts) > 1 else 0.0,
                "connected_components": nx.number_connected_components(G),
                "average_degree": sum(dict(G.degree()).values()) / len(concepts) if concepts else 0,
                "clustering_coefficient": nx.average_clustering(G) if len(concepts) > 2 else 0.0,
                "diameter": nx.diameter(G) if nx.is_connected(G) else "infinite",
                "linking_stats": self.linking_stats.copy()
            }
            
            # Conceitos mais conectados
            if concepts:
                degree_centrality = nx.degree_centrality(G)
                top_connected = sorted(
                    degree_centrality.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                analysis["most_connected_concepts"] = [
                    {
                        "concept_id": concept_id,
                        "centrality": centrality,
                        "concept_label": next(
                            (c.label for c in concepts if c.id == concept_id), 
                            "Unknown"
                        )
                    }
                    for concept_id, centrality in top_connected
                ]
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Connectivity analysis failed: {e}")
            return {
                "total_concepts": len(concepts),
                "total_links": len(links),
                "error": str(e)
            }
    
    def get_linking_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de linking."""
        return {
            **self.linking_stats,
            "config": {
                "semantic_threshold": self.config.semantic_similarity_threshold,
                "cross_domain_bonus": self.config.cross_domain_bonus,
                "universal_bonus": self.config.universal_concept_bonus,
                "max_links_per_concept": self.config.max_links_per_concept
            },
            "cache_status": {
                "embeddings_cached": len(self.embeddings_cache),
                "similarities_cached": len(self.similarity_cache),
                "concepts_clustered": len(self.concept_clusters)
            }
        }


__all__ = [
    "InterdisciplinaryConceptLinker", 
    "LinkingConfiguration",
    "CROSS_DOMAIN_CONCEPT_MAPPINGS",
    "UNIVERSAL_CONCEPTS"
]