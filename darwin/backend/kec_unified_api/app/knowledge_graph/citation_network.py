"""DARWIN Citation Network - Sistema de Rede de Citações Científicas

Sistema épico que constrói e analisa redes de citações científicas para
identificar influência, propagação de ideias e conexões interdisciplinares
entre papers de biomaterials, neuroscience, philosophy, quantum e psychiatry.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
import math

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes,
    PaperNode, AuthorNode, CitationEdge,
    CommunityDetection, CentralityAnalysis, PathAnalysis
)

logger = get_logger("knowledge_graph.citation_network")


# ==================== CITATION ANALYSIS CONFIGURATIONS ====================

@dataclass
class CitationNetworkConfig:
    """Configuração da Citation Network."""
    # PageRank parameters
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 1000
    pagerank_tol: float = 1e-6
    
    # Community detection
    enable_community_detection: bool = True
    community_algorithm: str = "louvain"  # louvain, spectral, leiden
    min_community_size: int = 3
    
    # Influence analysis
    influence_decay_factor: float = 0.8
    max_influence_hops: int = 4
    temporal_weight_decay: float = 0.9
    
    # Citation quality weighting
    journal_impact_weight: float = 0.3
    author_reputation_weight: float = 0.2
    citation_context_weight: float = 0.2
    temporal_recency_weight: float = 0.15
    cross_domain_bonus: float = 0.15
    
    # Bridge detection
    bridge_threshold: float = 0.3
    interdisciplinary_bonus: float = 0.25
    
    # Burst detection
    enable_burst_detection: bool = True
    burst_window_days: int = 365
    burst_threshold: float = 2.0


# Citation quality indicators
CITATION_QUALITY_PATTERNS = {
    "positive": [
        r"\bbuilds?\s+on\b", r"\bextends?\b", r"\bimproves?\b",
        r"\bvalidates?\b", r"\bconfirms?\b", r"\bsupports?\b",
        r"\bdemonstrates?\b", r"\bshows?\b", r"\bproves?\b"
    ],
    "critical": [
        r"\bcriticizes?\b", r"\bquestions?\b", r"\bchallenges?\b",
        r"\bdisputes?\b", r"\brefutes?\b", r"\bcontradicts?\b"
    ],
    "neutral": [
        r"\bcites?\b", r"\breferences?\b", r"\bmentions?\b",
        r"\bdiscusses?\b", r"\breviews?\b", r"\bcompares?\b"
    ],
    "methodological": [
        r"\busing\b", r"\badapts?\b", r"\bmodifies?\b",
        r"\bapplies?\b", r"\bimplements?\b", r"\bfollows?\b"
    ]
}

# Journal impact factor simulado (em implementação real, viria de database)
JOURNAL_IMPACT_FACTORS = {
    "nature": 49.962,
    "science": 47.728,
    "cell": 41.582,
    "nature_neuroscience": 24.884,
    "nature_materials": 43.841,
    "nature_physics": 20.034,
    "journal_philosophy": 3.2,
    "biomaterials": 12.479,
    "acta_biomaterialia": 9.037,
    "neuroimage": 5.812,
    "consciousness_and_cognition": 2.5,
    "quantum_information": 5.1,
    "biological_psychiatry": 12.095,
    "default": 2.0
}


class CitationNetwork:
    """
    Sistema completo de análise de rede de citações científicas.
    
    Funcionalidades:
    - Construction de rede de citações direcionada
    - PageRank e outras métricas de centralidade
    - Community detection (Louvain, Spectral, Leiden)
    - Bridge paper detection entre domínios
    - Influence propagation analysis
    - Citation burst detection
    - Temporal evolution analysis
    - Cross-domain citation analysis
    """
    
    def __init__(self, config: Optional[CitationNetworkConfig] = None):
        self.config = config or CitationNetworkConfig()
        
        # Core network structures
        self.citation_graph = nx.DiGraph()  # Directed graph for citations
        self.paper_nodes: Dict[str, PaperNode] = {}
        self.author_nodes: Dict[str, AuthorNode] = {}
        self.citation_edges: Dict[str, CitationEdge] = {}
        
        # Analysis results cache
        self.pagerank_scores: Dict[str, float] = {}
        self.communities: Dict[str, int] = {}
        self.bridge_papers: List[Dict[str, Any]] = []
        self.influence_scores: Dict[str, float] = {}
        
        # Temporal data
        self.citation_timeline: Dict[str, List[datetime]] = defaultdict(list)
        self.burst_papers: List[Dict[str, Any]] = []
        
        logger.info("📊 Citation Network initialized")
    
    # ==================== NETWORK CONSTRUCTION ====================
    
    async def build_citation_network(
        self, 
        papers: List[PaperNode],
        citations: List[CitationEdge],
        authors: Optional[List[AuthorNode]] = None
    ) -> Dict[str, Any]:
        """
        Constrói rede de citações completa.
        """
        logger.info(f"🔗 Building citation network: {len(papers)} papers, {len(citations)} citations")
        build_start = datetime.utcnow()
        
        try:
            # 1. Preparar dados
            await self._prepare_network_data(papers, citations, authors or [])
            
            # 2. Construir grafo de citações
            await self._build_citation_graph()
            
            # 3. Calcular métricas de centralidade
            centrality_results = await self._calculate_centrality_metrics()
            
            # 4. Detectar comunidades
            community_results = await self._detect_communities() if self.config.enable_community_detection else {}
            
            # 5. Identificar papers ponte
            bridge_results = await self._identify_bridge_papers()
            
            # 6. Análise de influência e propagação
            influence_results = await self._analyze_influence_propagation()
            
            # 7. Detecção de citation bursts
            burst_results = await self._detect_citation_bursts() if self.config.enable_burst_detection else {}
            
            # 8. Análise temporal
            temporal_results = await self._analyze_temporal_evolution()
            
            # 9. Estatísticas gerais
            network_stats = await self._calculate_network_statistics()
            
            build_time = (datetime.utcnow() - build_start).total_seconds()
            logger.info(f"✅ Citation network built in {build_time:.2f}s")
            
            return {
                "network_stats": network_stats,
                "centrality_analysis": centrality_results,
                "community_detection": community_results,
                "bridge_analysis": bridge_results,
                "influence_analysis": influence_results,
                "burst_analysis": burst_results,
                "temporal_analysis": temporal_results,
                "build_metadata": {
                    "build_time_seconds": build_time,
                    "papers_processed": len(papers),
                    "citations_processed": len(citations),
                    "timestamp": build_start.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Citation network build failed: {e}")
            raise
    
    async def _prepare_network_data(
        self, 
        papers: List[PaperNode],
        citations: List[CitationEdge],
        authors: List[AuthorNode]
    ):
        """Prepara dados para construção da rede."""
        # Cache papers
        for paper in papers:
            self.paper_nodes[paper.id] = paper
        
        # Cache authors
        for author in authors:
            self.author_nodes[author.id] = author
        
        # Cache citations e preparar timeline
        for citation in citations:
            self.citation_edges[citation.id] = citation
            
            # Adicionar ao timeline se paper de destino tem data
            target_paper = self.paper_nodes.get(citation.target)
            if target_paper and target_paper.publication_date:
                self.citation_timeline[citation.target].append(target_paper.publication_date)
        
        logger.info(f"📚 Prepared: {len(self.paper_nodes)} papers, {len(self.citation_edges)} citations")
    
    async def _build_citation_graph(self):
        """Constrói o grafo direcionado de citações."""
        self.citation_graph.clear()
        
        # Adicionar nós (papers)
        for paper_id, paper in self.paper_nodes.items():
            self.citation_graph.add_node(
                paper_id,
                title=paper.title,
                domain=paper.domain.value,
                publication_date=paper.publication_date.isoformat() if paper.publication_date else None,
                citation_count=paper.citation_count,
                journal=paper.journal or "unknown",
                authors=paper.authors
            )
        
        # Adicionar arestas (citations)
        for citation_id, citation in self.citation_edges.items():
            if citation.source in self.paper_nodes and citation.target in self.paper_nodes:
                # Calcular peso da citação
                citation_weight = await self._calculate_citation_weight(citation)
                
                self.citation_graph.add_edge(
                    citation.source,
                    citation.target,
                    weight=citation_weight,
                    citation_type=getattr(citation, 'citation_type', 'unknown'),
                    citation_context=getattr(citation, 'citation_context', ''),
                    edge_id=citation_id
                )
        
        logger.info(f"🔗 Built citation graph: {self.citation_graph.number_of_nodes()} nodes, {self.citation_graph.number_of_edges()} edges")
    
    async def _calculate_citation_weight(self, citation: CitationEdge) -> float:
        """Calcula peso da citação baseado em múltiplos fatores."""
        base_weight = citation.weight
        
        source_paper = self.paper_nodes.get(citation.source)
        target_paper = self.paper_nodes.get(citation.target)
        
        if not source_paper or not target_paper:
            return base_weight
        
        # 1. Journal impact factor
        journal_factor = self._get_journal_impact_factor(source_paper.journal)
        
        # 2. Author reputation (simulado)
        author_factor = await self._calculate_author_reputation_factor(source_paper.authors)
        
        # 3. Citation context quality
        context_factor = await self._analyze_citation_context_quality(
            getattr(citation, 'citation_context', '')
        )
        
        # 4. Temporal recency
        temporal_factor = await self._calculate_temporal_recency_factor(
            source_paper.publication_date, target_paper.publication_date
        )
        
        # 5. Cross-domain bonus
        cross_domain_factor = (
            self.config.cross_domain_bonus 
            if source_paper.domain != target_paper.domain 
            else 0.0
        )
        
        # Combinar fatores
        weighted_score = (
            base_weight +
            journal_factor * self.config.journal_impact_weight +
            author_factor * self.config.author_reputation_weight +
            context_factor * self.config.citation_context_weight +
            temporal_factor * self.config.temporal_recency_weight +
            cross_domain_factor
        )
        
        return max(min(weighted_score, 10.0), 0.1)  # Clamp entre 0.1 e 10.0
    
    def _get_journal_impact_factor(self, journal: Optional[str]) -> float:
        """Obtém fator de impacto do journal."""
        if not journal:
            return JOURNAL_IMPACT_FACTORS["default"]
        
        # Normalizar nome do journal
        journal_key = journal.lower().replace(" ", "_").replace("-", "_")
        
        # Buscar por matches parciais
        for key, impact in JOURNAL_IMPACT_FACTORS.items():
            if key in journal_key or journal_key in key:
                return min(impact / 50.0, 1.0)  # Normalizar para [0,1]
        
        return JOURNAL_IMPACT_FACTORS["default"] / 50.0
    
    async def _calculate_author_reputation_factor(self, authors: List[str]) -> float:
        """Calcula fator de reputação dos autores."""
        if not authors:
            return 0.0
        
        # Em implementação real, consultaria database de autores
        # Por agora, simular baseado na quantidade de autores (colaboração)
        collaboration_bonus = min(len(authors) * 0.1, 0.5)
        
        # Simular h-index médio
        simulated_h_index = sum(len(author) for author in authors) / len(authors) / 10
        h_index_factor = min(simulated_h_index, 1.0)
        
        return (collaboration_bonus + h_index_factor) / 2
    
    async def _analyze_citation_context_quality(self, context: str) -> float:
        """Analisa qualidade do contexto da citação."""
        if not context:
            return 0.5  # Neutral if no context
        
        context_lower = context.lower()
        quality_score = 0.5  # Base neutral
        
        # Analisar padrões de qualidade
        for quality_type, patterns in CITATION_QUALITY_PATTERNS.items():
            pattern_matches = sum(
                1 for pattern in patterns 
                if re.search(pattern, context_lower)
            )
            
            if pattern_matches > 0:
                if quality_type == "positive":
                    quality_score += 0.3 * min(pattern_matches, 2)
                elif quality_type == "critical":
                    quality_score += 0.2 * min(pattern_matches, 1)  # Critical still valuable
                elif quality_type == "methodological":
                    quality_score += 0.25 * min(pattern_matches, 2)
                # Neutral doesn't change score
        
        return min(max(quality_score, 0.0), 1.0)
    
    async def _calculate_temporal_recency_factor(
        self, 
        source_date: Optional[datetime],
        target_date: Optional[datetime]
    ) -> float:
        """Calcula fator de recência temporal."""
        if not source_date or not target_date:
            return 0.5
        
        # Citações são mais valiosas quando são recentes
        time_diff = abs((source_date - target_date).days)
        
        # Decay exponencial
        decay_factor = math.exp(-time_diff / (365 * 2))  # 2 years half-life
        
        return decay_factor
    
    # ==================== CENTRALITY ANALYSIS ====================
    
    async def _calculate_centrality_metrics(self) -> Dict[str, Any]:
        """Calcula múltiplas métricas de centralidade."""
        if self.citation_graph.number_of_nodes() == 0:
            return {}
        
        try:
            centrality_results = {}
            
            # 1. PageRank (mais importante para citações)
            pagerank = nx.pagerank(
                self.citation_graph,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol,
                weight='weight'
            )
            self.pagerank_scores = pagerank
            centrality_results["pagerank"] = CentralityAnalysis(
                algorithm="pagerank",
                results=pagerank,
                top_k=sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            
            # 2. In-degree centrality (papers mais citados)
            in_degree = nx.in_degree_centrality(self.citation_graph)
            centrality_results["in_degree"] = CentralityAnalysis(
                algorithm="in_degree_centrality",
                results=in_degree,
                top_k=sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            
            # 3. Out-degree centrality (papers que mais citam)
            out_degree = nx.out_degree_centrality(self.citation_graph)
            centrality_results["out_degree"] = CentralityAnalysis(
                algorithm="out_degree_centrality",
                results=out_degree,
                top_k=sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            
            # 4. Betweenness centrality (papers ponte)
            betweenness = nx.betweenness_centrality(
                self.citation_graph, 
                weight='weight',
                normalized=True
            )
            centrality_results["betweenness"] = CentralityAnalysis(
                algorithm="betweenness_centrality",
                results=betweenness,
                top_k=sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            
            # 5. Closeness centrality
            try:
                # Para grafo direcionado, usar versão strongly connected components
                if nx.is_strongly_connected(self.citation_graph):
                    closeness = nx.closeness_centrality(self.citation_graph, distance='weight')
                else:
                    # Usar componente maior
                    largest_cc = max(nx.strongly_connected_components(self.citation_graph), key=len)
                    subgraph = self.citation_graph.subgraph(largest_cc)
                    closeness_sub = nx.closeness_centrality(subgraph, distance='weight')
                    closeness = {node: closeness_sub.get(node, 0.0) for node in self.citation_graph.nodes()}
                
                centrality_results["closeness"] = CentralityAnalysis(
                    algorithm="closeness_centrality",
                    results=closeness,
                    top_k=sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:20]
                )
            except Exception as e:
                logger.warning(f"Closeness centrality calculation failed: {e}")
            
            logger.info(f"📊 Calculated {len(centrality_results)} centrality metrics")
            return centrality_results
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            return {}
    
    # ==================== COMMUNITY DETECTION ====================
    
    async def _detect_communities(self) -> Dict[str, Any]:
        """Detecta comunidades na rede de citações."""
        if self.citation_graph.number_of_nodes() < self.config.min_community_size:
            return {}
        
        try:
            community_results = {}
            
            # 1. Louvain algorithm (default)
            if self.config.community_algorithm == "louvain":
                communities = await self._louvain_communities()
                
            # 2. Spectral clustering
            elif self.config.community_algorithm == "spectral":
                communities = await self._spectral_communities()
                
            # 3. Leiden algorithm (se disponível)
            elif self.config.community_algorithm == "leiden":
                communities = await self._leiden_communities()
                
            else:
                logger.warning(f"Unknown community algorithm: {self.config.community_algorithm}")
                return {}
            
            if communities:
                self.communities = communities
                
                # Calcular modularity
                modularity = await self._calculate_modularity(communities)
                
                # Agrupar papers por comunidade
                communities_grouped = defaultdict(list)
                for paper_id, community_id in communities.items():
                    communities_grouped[community_id].append(paper_id)
                
                # Filtrar comunidades pequenas
                valid_communities = {
                    comm_id: papers 
                    for comm_id, papers in communities_grouped.items()
                    if len(papers) >= self.config.min_community_size
                }
                
                community_results = {
                    "algorithm": self.config.community_algorithm,
                    "communities": list(valid_communities.values()),
                    "modularity": modularity,
                    "total_communities": len(valid_communities),
                    "community_stats": await self._analyze_community_characteristics(valid_communities)
                }
                
                logger.info(f"🏘️ Detected {len(valid_communities)} communities with modularity {modularity:.3f}")
            
            return community_results
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
    
    async def _louvain_communities(self) -> Dict[str, int]:
        """Implementa algoritmo de Louvain para detecção de comunidades."""
        try:
            # Converter para grafo não-direcionado para Louvain
            undirected_graph = self.citation_graph.to_undirected()
            
            # Usar NetworkX community detection (implementation simplified)
            import networkx.algorithms.community as nx_community
            communities = nx_community.greedy_modularity_communities(undirected_graph, weight='weight')
            
            # Converter para dict
            paper_to_community = {}
            for i, community in enumerate(communities):
                for paper_id in community:
                    paper_to_community[paper_id] = i
            
            return paper_to_community
            
        except Exception as e:
            logger.warning(f"Louvain algorithm failed: {e}")
            return {}
    
    async def _spectral_communities(self) -> Dict[str, int]:
        """Implementa spectral clustering para comunidades."""
        try:
            # Obter matriz de adjacência
            nodes = list(self.citation_graph.nodes())
            adj_matrix = nx.adjacency_matrix(self.citation_graph, nodelist=nodes, weight='weight')
            
            # Estimar número de clusters
            n_clusters = min(max(len(nodes) // 20, 2), 10)
            
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            # Normalizar matriz para spectral clustering
            normalized_adj = normalize(adj_matrix + adj_matrix.T, norm='l2', axis=1)
            cluster_labels = clustering.fit_predict(normalized_adj)
            
            # Converter para dict
            paper_to_community = {
                nodes[i]: int(cluster_labels[i]) 
                for i in range(len(nodes))
            }
            
            return paper_to_community
            
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            return {}
    
    async def _leiden_communities(self) -> Dict[str, int]:
        """Placeholder para algoritmo de Leiden."""
        # Em implementação real, usaria biblioteca leidenalg
        logger.warning("Leiden algorithm not implemented, falling back to Louvain")
        return await self._louvain_communities()
    
    async def _calculate_modularity(self, communities: Dict[str, int]) -> float:
        """Calcula modularidade da partição de comunidades."""
        try:
            # Converter communities dict para lista de sets
            community_sets = defaultdict(set)
            for paper, comm_id in communities.items():
                community_sets[comm_id].add(paper)
            
            communities_list = list(community_sets.values())
            
            # Usar NetworkX para calcular modularity
            undirected_graph = self.citation_graph.to_undirected()
            modularity = nx.algorithms.community.modularity(
                undirected_graph, 
                communities_list, 
                weight='weight'
            )
            
            return modularity
            
        except Exception as e:
            logger.warning(f"Modularity calculation failed: {e}")
            return 0.0
    
    async def _analyze_community_characteristics(
        self, 
        communities: Dict[int, List[str]]
    ) -> List[Dict[str, Any]]:
        """Analisa características de cada comunidade."""
        community_stats = []
        
        for comm_id, paper_ids in communities.items():
            # Contar domínios na comunidade
            domain_counts = Counter()
            total_citations = 0
            avg_publication_year = []
            journals = Counter()
            
            for paper_id in paper_ids:
                paper = self.paper_nodes.get(paper_id)
                if paper:
                    domain_counts[paper.domain.value] += 1
                    total_citations += paper.citation_count
                    
                    if paper.publication_date:
                        avg_publication_year.append(paper.publication_date.year)
                    
                    if paper.journal:
                        journals[paper.journal] += 1
            
            # Calcular estatísticas
            dominant_domain = domain_counts.most_common(1)[0] if domain_counts else ("unknown", 0)
            avg_year = sum(avg_publication_year) / len(avg_publication_year) if avg_publication_year else 0
            top_journal = journals.most_common(1)[0] if journals else ("unknown", 0)
            
            community_stats.append({
                "community_id": comm_id,
                "size": len(paper_ids),
                "dominant_domain": dominant_domain[0],
                "domain_diversity": len(domain_counts),
                "domain_distribution": dict(domain_counts),
                "total_citations": total_citations,
                "avg_citations_per_paper": total_citations / len(paper_ids) if paper_ids else 0,
                "avg_publication_year": int(avg_year) if avg_year else None,
                "top_journal": top_journal[0],
                "journal_diversity": len(journals),
                "interdisciplinary_score": self._calculate_interdisciplinary_score(domain_counts)
            })
        
        return community_stats
    
    def _calculate_interdisciplinary_score(self, domain_counts: Counter) -> float:
        """Calcula score de interdisciplinaridade da comunidade."""
        if len(domain_counts) <= 1:
            return 0.0
        
        # Shannon entropy normalizado
        total = sum(domain_counts.values())
        entropy = -sum(
            (count / total) * math.log2(count / total) 
            for count in domain_counts.values() if count > 0
        )
        
        max_entropy = math.log2(len(domain_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    # ==================== BRIDGE PAPER DETECTION ====================
    
    async def _identify_bridge_papers(self) -> Dict[str, Any]:
        """Identifica papers que fazem ponte entre domínios."""
        try:
            bridge_candidates = []
            
            # 1. Usar betweenness centrality como base
            if not hasattr(self, 'betweenness_scores'):
                betweenness = nx.betweenness_centrality(self.citation_graph, weight='weight')
            else:
                betweenness = getattr(self, 'betweenness_scores', {})
            
            # 2. Analisar cada paper de alta betweenness
            high_betweenness_papers = [
                (paper_id, score) for paper_id, score in betweenness.items()
                if score > self.config.bridge_threshold
            ]
            
            for paper_id, betweenness_score in high_betweenness_papers:
                paper = self.paper_nodes.get(paper_id)
                if not paper:
                    continue
                
                # Analisar citations in/out por domínio
                citing_domains = await self._analyze_citing_domains(paper_id)
                cited_domains = await self._analyze_cited_domains(paper_id)
                
                # Calcular bridge strength
                bridge_strength = await self._calculate_bridge_strength(
                    paper, citing_domains, cited_domains, betweenness_score
                )
                
                if bridge_strength > self.config.bridge_threshold:
                    bridge_candidates.append({
                        "paper_id": paper_id,
                        "paper_title": paper.title,
                        "paper_domain": paper.domain.value,
                        "bridge_strength": bridge_strength,
                        "betweenness_centrality": betweenness_score,
                        "citing_domains": citing_domains,
                        "cited_domains": cited_domains,
                        "cross_domain_citations": sum(
                            count for domain, count in citing_domains.items()
                            if domain != paper.domain.value
                        ),
                        "interdisciplinary_score": self._calculate_paper_interdisciplinary_score(
                            citing_domains, cited_domains, paper.domain.value
                        )
                    })
            
            # Ordenar por bridge strength
            bridge_candidates.sort(key=lambda x: x["bridge_strength"], reverse=True)
            self.bridge_papers = bridge_candidates[:50]  # Top 50 bridge papers
            
            # Análise agregada
            bridge_analysis = {
                "total_bridge_papers": len(self.bridge_papers),
                "top_bridge_papers": self.bridge_papers[:10],
                "domain_bridge_distribution": self._analyze_bridge_domain_distribution(),
                "interdisciplinary_connections": self._analyze_interdisciplinary_connections(),
                "bridge_statistics": {
                    "avg_bridge_strength": sum(bp["bridge_strength"] for bp in self.bridge_papers) / len(self.bridge_papers) if self.bridge_papers else 0,
                    "avg_betweenness": sum(bp["betweenness_centrality"] for bp in self.bridge_papers) / len(self.bridge_papers) if self.bridge_papers else 0,
                    "total_cross_domain_citations": sum(bp["cross_domain_citations"] for bp in self.bridge_papers)
                }
            }
            
            logger.info(f"🌉 Identified {len(self.bridge_papers)} bridge papers")
            return bridge_analysis
            
        except Exception as e:
            logger.error(f"Bridge paper detection failed: {e}")
            return {}
    
    async def _analyze_citing_domains(self, paper_id: str) -> Dict[str, int]:
        """Analisa domínios dos papers que citam este paper."""
        citing_domains = Counter()
        
        # Papers que citam este (predecessors no grafo direcionado)
        for citing_paper_id in self.citation_graph.predecessors(paper_id):
            citing_paper = self.paper_nodes.get(citing_paper_id)
            if citing_paper:
                citing_domains[citing_paper.domain.value] += 1
        
        return dict(citing_domains)
    
    async def _analyze_cited_domains(self, paper_id: str) -> Dict[str, int]:
        """Analisa domínios dos papers citados por este paper."""
        cited_domains = Counter()
        
        # Papers citados por este (successors no grafo direcionado)
        for cited_paper_id in self.citation_graph.successors(paper_id):
            cited_paper = self.paper_nodes.get(cited_paper_id)
            if cited_paper:
                cited_domains[cited_paper.domain.value] += 1
        
        return dict(cited_domains)
    
    async def _calculate_bridge_strength(
        self,
        paper: PaperNode,
        citing_domains: Dict[str, int],
        cited_domains: Dict[str, int],
        betweenness_score: float
    ) -> float:
        """Calcula força de ponte do paper."""
        paper_domain = paper.domain.value
        
        # Contar conexões cross-domain
        cross_domain_in = sum(
            count for domain, count in citing_domains.items()
            if domain != paper_domain
        )
        
        cross_domain_out = sum(
            count for domain, count in cited_domains.items()
            if domain != paper_domain
        )
        
        total_in = sum(citing_domains.values())
        total_out = sum(cited_domains.values())
        
        # Calcular ratios cross-domain
        cross_domain_in_ratio = cross_domain_in / total_in if total_in > 0 else 0
        cross_domain_out_ratio = cross_domain_out / total_out if total_out > 0 else 0
        
        # Diversidade de domínios
        domain_diversity_in = len([d for d in citing_domains.keys() if d != paper_domain])
        domain_diversity_out = len([d for d in cited_domains.keys() if d != paper_domain])
        
        # Bridge strength combinando múltiplos fatores
        bridge_strength = (
            betweenness_score * 0.4 +  # Posição estrutural
            (cross_domain_in_ratio + cross_domain_out_ratio) / 2 * 0.3 +  # Cross-domain ratio
            min(domain_diversity_in + domain_diversity_out, 5) / 5 * 0.2 +  # Domain diversity
            self.config.interdisciplinary_bonus * min(cross_domain_in + cross_domain_out, 10) / 10 * 0.1  # Volume bonus
        )
        
        return min(bridge_strength, 1.0)
    
    def _calculate_paper_interdisciplinary_score(
        self,
        citing_domains: Dict[str, int],
        cited_domains: Dict[str, int],
        paper_domain: str
    ) -> float:
        """Calcula score interdisciplinar do paper."""
        all_domains = set(citing_domains.keys()) | set(cited_domains.keys())
        other_domains = all_domains - {paper_domain}
        
        if not other_domains:
            return 0.0
        
        # Shannon entropy das conexões por domínio
        all_connections = Counter(citing_domains) + Counter(cited_domains)
        total = sum(all_connections.values())
        
        if total == 0:
            return 0.0
        
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in all_connections.values() if count > 0
        )
        
        max_entropy = math.log2(len(all_connections))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_bridge_domain_distribution(self) -> Dict[str, Any]:
        """Analisa distribuição de domínios nos bridge papers."""
        domain_counts = Counter()
        domain_bridge_strength = defaultdict(list)
        
        for bridge_paper in self.bridge_papers:
            domain = bridge_paper["paper_domain"]
            strength = bridge_paper["bridge_strength"]
            
            domain_counts[domain] += 1
            domain_bridge_strength[domain].append(strength)
        
        distribution = {}
        for domain, count in domain_counts.items():
            strengths = domain_bridge_strength[domain]
            distribution[domain] = {
                "count": count,
                "avg_bridge_strength": sum(strengths) / len(strengths),
                "max_bridge_strength": max(strengths),
                "percentage_of_bridges": count / len(self.bridge_papers) * 100 if self.bridge_papers else 0
            }
        
        return distribution
    
    def _analyze_interdisciplinary_connections(self) -> Dict[str, int]:
        """Analisa conexões interdisciplinares via bridge papers."""
        connections = Counter()
        
        for bridge_paper in self.bridge_papers:
            paper_domain = bridge_paper["paper_domain"]
            
            # Contar conexões para outros domínios
            for domain, count in bridge_paper["citing_domains"].items():
                if domain != paper_domain and count > 0:
                    connection_key = tuple(sorted([paper_domain, domain]))
                    connections[connection_key] += count
            
            for domain, count in bridge_paper["cited_domains"].items():
                if domain != paper_domain and count > 0:
                    connection_key = tuple(sorted([paper_domain, domain]))
                    connections[connection_key] += count
        
        return {f"{d1} <-> {d2}": count for (d1, d2), count in connections.items()}
    
    # ==================== INFLUENCE PROPAGATION ANALYSIS ====================
    
    async def _analyze_influence_propagation(self) -> Dict[str, Any]:
        """Analisa como influência se propaga na rede."""
        try:
            # Usar papers com alto PageRank como sementes de influência
            top_influential = sorted(
                self.pagerank_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            influence_analysis = {
                "top_influential_papers": [],
                "influence_propagation_paths": [],
                "domain_influence_flow": {},
                "temporal_influence_patterns": {}
            }
            
            for paper_id, pagerank_score in top_influential:
                paper = self.paper_nodes.get(paper_id)
                if not paper:
                    continue
                
                # Analisar propagação de influência
                propagation_data = await self._calculate_influence_propagation(paper_id, pagerank_score)
                
                influence_analysis["top_influential_papers"].append({
                    "paper_id": paper_id,
                    "title": paper.title,
                    "domain": paper.domain.value,
                    "pagerank_score": pagerank_score,
                    "direct_influence": propagation_data["direct_influence"],
                    "indirect_influence": propagation_data["indirect_influence"],
                    "influence_reach": propagation_data["influence_reach"],
                    "cross_domain_influence": propagation_data["cross_domain_influence"]
                })
            
            # Analisar fluxo de influência entre domínios
            domain_flow = await self._analyze_domain_influence_flow()
            influence_analysis["domain_influence_flow"] = domain_flow
            
            logger.info("📈 Analyzed influence propagation patterns")
            return influence_analysis
            
        except Exception as e:
            logger.error(f"Influence analysis failed: {e}")
            return {}
    
    async def _calculate_influence_propagation(
        self, 
        source_paper_id: str, 
        initial_influence: float
    ) -> Dict[str, Any]:
        """Calcula propagação de influência de um paper específico."""
        visited = set()
        influence_by_hop = defaultdict(float)
        cross_domain_influence = defaultdict(float)
        
        source_paper = self.paper_nodes.get(source_paper_id)
        source_domain = source_paper.domain.value if source_paper else "unknown"
        
        # BFS com decay de influência
        queue = [(source_paper_id, initial_influence, 0)]  # (paper_id, influence, hop)
        
        while queue:
            current_id, current_influence, hop = queue.pop(0)
            
            if current_id in visited or hop > self.config.max_influence_hops:
                continue
            
            visited.add(current_id)
            influence_by_hop[hop] += current_influence
            
            current_paper = self.paper_nodes.get(current_id)
            if current_paper and current_paper.domain.value != source_domain:
                cross_domain_influence[current_paper.domain.value] += current_influence
            
            # Propagar para papers citados (out-edges)
            for cited_paper_id in self.citation_graph.successors(current_id):
                if cited_paper_id not in visited:
                    # Aplicar decay baseado em peso da aresta e hop count
                    edge_data = self.citation_graph.get_edge_data(current_id, cited_paper_id)
                    edge_weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    
                    decayed_influence = (
                        current_influence * 
                        self.config.influence_decay_factor ** hop * 
                        edge_weight / 10.0  # Normalizar edge weight
                    )
                    
                    if decayed_influence > 0.01:  # Threshold para continuar propagação
                        queue.append((cited_paper_id, decayed_influence, hop + 1))
        
        return {
            "direct_influence": influence_by_hop.get(1, 0.0),
            "indirect_influence": sum(influence_by_hop[h] for h in influence_by_hop if h > 1),
            "influence_reach": len(visited),
            "cross_domain_influence": dict(cross_domain_influence),
            "influence_by_hop": dict(influence_by_hop)
        }
    
    async def _analyze_domain_influence_flow(self) -> Dict[str, Any]:
        """Analisa fluxo de influência entre domínios."""
        domain_flow = defaultdict(lambda: defaultdict(float))
        
        # Para cada aresta, calcular fluxo entre domínios
        for source_id, target_id, edge_data in self.citation_graph.edges(data=True):
            source_paper = self.paper_nodes.get(source_id)
            target_paper = self.paper_nodes.get(target_id)
            
            if source_paper and target_paper:
                source_domain = source_paper.domain.value
                target_domain = target_paper.domain.value
                edge_weight = edge_data.get('weight', 1.0)
                
                # Usar PageRank do source como proxy para influence
                source_influence = self.pagerank_scores.get(source_id, 0.0)
                flow_amount = source_influence * edge_weight
                
                domain_flow[source_domain][target_domain] += flow_amount
        
        # Converter para formato mais legível
        flow_matrix = {}
        for source_domain, targets in domain_flow.items():
            flow_matrix[source_domain] = dict(targets)
        
        return flow_matrix
    
    # ==================== CITATION BURST DETECTION ====================
    
    async def _detect_citation_bursts(self) -> Dict[str, Any]:
        """Detecta citation bursts (sudden popularity spikes)."""
        try:
            burst_papers = []
            current_date = datetime.utcnow()
            
            for paper_id, citation_dates in self.citation_timeline.items():
                if len(citation_dates) < 3:  # Need minimum citations for burst detection
                    continue
                
                # Analisar padrão temporal de citações
                burst_data = await self._analyze_citation_burst_pattern(
                    paper_id, citation_dates, current_date
                )
                
                if burst_data["burst_score"] > self.config.burst_threshold:
                    paper = self.paper_nodes.get(paper_id)
                    
                    burst_papers.append({
                        "paper_id": paper_id,
                        "title": paper.title if paper else "Unknown",
                        "domain": paper.domain.value if paper else "unknown",
                        "burst_score": burst_data["burst_score"],
                        "burst_period": burst_data["burst_period"],
                        "citations_in_burst": burst_data["citations_in_burst"],
                        "total_citations": len(citation_dates),
                        "burst_start": burst_data["burst_start"],
                        "burst_peak": burst_data["burst_peak"]
                    })
            
            # Ordenar por burst score
            burst_papers.sort(key=lambda x: x["burst_score"], reverse=True)
            self.burst_papers = burst_papers[:30]  # Top 30 burst papers
            
            burst_analysis = {
                "total_burst_papers": len(self.burst_papers),
                "top_burst_papers": self.burst_papers[:10],
                "burst_by_domain": self._analyze_bursts_by_domain(),
                "recent_bursts": [
                    bp for bp in self.burst_papers 
                    if bp["burst_start"] and 
                    (current_date - bp["burst_start"]).days < self.config.burst_window_days
                ]
            }
            
            logger.info(f"💥 Detected {len(self.burst_papers)} citation bursts")
            return burst_analysis
            
        except Exception as e:
            logger.error(f"Burst detection failed: {e}")
            return {}
    
    async def _analyze_citation_burst_pattern(
        self, 
        paper_id: str,
        citation_dates: List[datetime],
        current_date: datetime
    ) -> Dict[str, Any]:
        """Analisa padrão de burst para um paper específico."""
        # Ordenar datas
        sorted_dates = sorted(citation_dates)
        
        if len(sorted_dates) < 3:
            return {"burst_score": 0.0}
        
        # Criar janelas temporais para análise
        window_days = self.config.burst_window_days
        windows = []
        
        start_date = sorted_dates[0]
        end_date = min(sorted_dates[-1], current_date)
        
        current_window_start = start_date
        while current_window_start < end_date:
            window_end = current_window_start + timedelta(days=window_days)
            
            # Contar citações nesta janela
            citations_in_window = [
                date for date in sorted_dates
                if current_window_start <= date < window_end
            ]
            
            windows.append({
                "start": current_window_start,
                "end": window_end,
                "citation_count": len(citations_in_window),
                "citations": citations_in_window
            })
            
            current_window_start = window_end
        
        if len(windows) < 2:
            return {"burst_score": 0.0}
        
        # Detectar burst usando variação z-score
        citation_counts = [w["citation_count"] for w in windows]
        mean_citations = sum(citation_counts) / len(citation_counts)
        std_citations = (
            sum((x - mean_citations) ** 2 for x in citation_counts) / len(citation_counts)
        ) ** 0.5
        
        if std_citations == 0:
            return {"burst_score": 0.0}
        
        # Encontrar janela com maior z-score
        max_burst_score = 0.0
        burst_window = None
        
        for i, window in enumerate(windows):
            z_score = (window["citation_count"] - mean_citations) / std_citations
            if z_score > max_burst_score:
                max_burst_score = z_score
                burst_window = window
        
        return {
            "burst_score": max_burst_score,
            "burst_period": f"{burst_window['start'].strftime('%Y-%m')} to {burst_window['end'].strftime('%Y-%m')}" if burst_window else None,
            "citations_in_burst": burst_window["citation_count"] if burst_window else 0,
            "burst_start": burst_window["start"] if burst_window else None,
            "burst_peak": max(burst_window["citations"], key=lambda x: x) if burst_window and burst_window["citations"] else None,
            "baseline_citations_per_window": mean_citations,
            "burst_intensity": max_burst_score / self.config.burst_threshold if self.config.burst_threshold > 0 else 0
        }
    
    def _analyze_bursts_by_domain(self) -> Dict[str, Any]:
        """Analisa distribution de bursts por domínio."""
        domain_bursts = defaultdict(list)
        
        for burst_paper in self.burst_papers:
            domain = burst_paper["domain"]
            domain_bursts[domain].append(burst_paper["burst_score"])
        
        domain_analysis = {}
        for domain, scores in domain_bursts.items():
            domain_analysis[domain] = {
                "count": len(scores),
                "avg_burst_score": sum(scores) / len(scores),
                "max_burst_score": max(scores),
                "percentage_of_bursts": len(scores) / len(self.burst_papers) * 100 if self.burst_papers else 0
            }
        
        return domain_analysis
    
    # ==================== TEMPORAL EVOLUTION ANALYSIS ====================
    
    async def _analyze_temporal_evolution(self) -> Dict[str, Any]:
        """Analisa evolução temporal da rede de citações."""
        try:
            # Agrupar papers por ano de publicação
            papers_by_year = defaultdict(list)
            citations_by_year = defaultdict(int)
            
            for paper in self.paper_nodes.values():
                if paper.publication_date:
                    year = paper.publication_date.year
                    papers_by_year[year].append(paper)
            
            # Contar citações por ano (baseado na data do paper citado)
            for citation in self.citation_edges.values():
                target_paper = self.paper_nodes.get(citation.target)
                if target_paper and target_paper.publication_date:
                    year = target_paper.publication_date.year
                    citations_by_year[year] += 1
            
            # Análise de crescimento
            years = sorted(papers_by_year.keys())
            growth_analysis = await self._analyze_network_growth(years, papers_by_year, citations_by_year)
            
            # Análise de domínios ao longo do tempo
            domain_evolution = await self._analyze_domain_evolution_over_time(papers_by_year)
            
            temporal_analysis = {
                "publication_timeline": {
                    str(year): len(papers) for year, papers in papers_by_year.items()
                },
                "citation_timeline": {
                    str(year): count for year, count in citations_by_year.items()
                },
                "growth_analysis": growth_analysis,
                "domain_evolution": domain_evolution,
                "temporal_statistics": {
                    "earliest_paper": min(years) if years else None,
                    "latest_paper": max(years) if years else None,
                    "peak_publication_year": max(papers_by_year.items(), key=lambda x: len(x[1]))[0] if papers_by_year else None,
                    "peak_citation_year": max(citations_by_year.items(), key=lambda x: x[1])[0] if citations_by_year else None,
                    "total_years_span": max(years) - min(years) + 1 if years else 0
                }
            }
            
            logger.info("📈 Analyzed temporal evolution patterns")
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {}
    
    async def _analyze_network_growth(
        self, 
        years: List[int],
        papers_by_year: Dict[int, List[PaperNode]],
        citations_by_year: Dict[int, int]
    ) -> Dict[str, Any]:
        """Analisa crescimento da rede ao longo do tempo."""
        cumulative_papers = 0
        cumulative_citations = 0
        growth_rates_papers = []
        growth_rates_citations = []
        
        growth_data = []
        
        for i, year in enumerate(years):
            papers_this_year = len(papers_by_year[year])
            citations_this_year = citations_by_year.get(year, 0)
            
            cumulative_papers += papers_this_year
            cumulative_citations += citations_this_year
            
            # Calcular growth rate
            if i > 0:
                prev_cumulative_papers = cumulative_papers - papers_this_year
                prev_cumulative_citations = cumulative_citations - citations_this_year
                
                paper_growth_rate = (
                    (papers_this_year / prev_cumulative_papers * 100) 
                    if prev_cumulative_papers > 0 else 0
                )
                citation_growth_rate = (
                    (citations_this_year / prev_cumulative_citations * 100) 
                    if prev_cumulative_citations > 0 else 0
                )
                
                growth_rates_papers.append(paper_growth_rate)
                growth_rates_citations.append(citation_growth_rate)
            
            growth_data.append({
                "year": year,
                "papers_this_year": papers_this_year,
                "citations_this_year": citations_this_year,
                "cumulative_papers": cumulative_papers,
                "cumulative_citations": cumulative_citations
            })
        
        return {
            "growth_timeline": growth_data,
            "avg_paper_growth_rate": sum(growth_rates_papers) / len(growth_rates_papers) if growth_rates_papers else 0,
            "avg_citation_growth_rate": sum(growth_rates_citations) / len(growth_rates_citations) if growth_rates_citations else 0,
            "peak_growth_year_papers": years[growth_rates_papers.index(max(growth_rates_papers))] if growth_rates_papers else None,
            "peak_growth_year_citations": years[growth_rates_citations.index(max(growth_rates_citations))] if growth_rates_citations else None
        }
    
    async def _analyze_domain_evolution_over_time(
        self, 
        papers_by_year: Dict[int, List[PaperNode]]
    ) -> Dict[str, Any]:
        """Analisa evolução dos domínios ao longo do tempo."""
        domain_by_year = defaultdict(lambda: defaultdict(int))
        
        for year, papers in papers_by_year.items():
            for paper in papers:
                domain_by_year[year][paper.domain.value] += 1
        
        # Converter para formato mais útil
        domain_evolution = {}
        all_years = sorted(papers_by_year.keys())
        all_domains = set()
        
        for year_data in domain_by_year.values():
            all_domains.update(year_data.keys())
        
        for domain in all_domains:
            domain_timeline = []
            for year in all_years:
                count = domain_by_year[year].get(domain, 0)
                domain_timeline.append({"year": year, "count": count})
            domain_evolution[domain] = domain_timeline
        
        return domain_evolution
    
    # ==================== NETWORK STATISTICS ====================
    
    async def _calculate_network_statistics(self) -> Dict[str, Any]:
        """Calcula estatísticas gerais da rede."""
        try:
            stats = {
                "basic_metrics": {
                    "total_papers": len(self.paper_nodes),
                    "total_citations": len(self.citation_edges),
                    "total_authors": len(self.author_nodes),
                    "graph_nodes": self.citation_graph.number_of_nodes(),
                    "graph_edges": self.citation_graph.number_of_edges()
                }
            }
            
            if self.citation_graph.number_of_nodes() > 0:
                # Density
                stats["network_structure"] = {
                    "density": nx.density(self.citation_graph),
                    "is_strongly_connected": nx.is_strongly_connected(self.citation_graph),
                    "number_strongly_connected_components": nx.number_strongly_connected_components(self.citation_graph),
                    "number_weakly_connected_components": nx.number_weakly_connected_components(self.citation_graph)
                }
                
                # Degree statistics
                in_degrees = [d for n, d in self.citation_graph.in_degree()]
                out_degrees = [d for n, d in self.citation_graph.out_degree()]
                
                stats["degree_statistics"] = {
                    "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
                    "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
                    "max_in_degree": max(in_degrees) if in_degrees else 0,
                    "max_out_degree": max(out_degrees) if out_degrees else 0,
                    "in_degree_distribution": dict(Counter(in_degrees)),
                    "out_degree_distribution": dict(Counter(out_degrees))
                }
                
                # Domain statistics
                domain_counts = Counter(
                    paper.domain.value for paper in self.paper_nodes.values()
                )
                
                stats["domain_statistics"] = {
                    "papers_by_domain": dict(domain_counts),
                    "domain_diversity": len(domain_counts),
                    "most_common_domain": domain_counts.most_common(1)[0] if domain_counts else None
                }
                
                # Citation statistics
                citation_counts = [paper.citation_count for paper in self.paper_nodes.values()]
                
                stats["citation_statistics"] = {
                    "total_citation_count": sum(citation_counts),
                    "avg_citations_per_paper": sum(citation_counts) / len(citation_counts) if citation_counts else 0,
                    "max_citations": max(citation_counts) if citation_counts else 0,
                    "citation_distribution": dict(Counter(citation_counts))
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Network statistics calculation failed: {e}")
            return {"error": str(e)}


__all__ = [
    "CitationNetwork", 
    "CitationNetworkConfig",
    "CITATION_QUALITY_PATTERNS",
    "JOURNAL_IMPACT_FACTORS"
]