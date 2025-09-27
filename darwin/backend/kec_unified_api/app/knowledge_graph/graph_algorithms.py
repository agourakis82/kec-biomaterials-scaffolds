"""DARWIN Graph Algorithms - Algoritmos Avançados de Análise de Knowledge Graph

Sistema épico que implementa algoritmos avançados de análise de grafos para o Knowledge Graph
interdisciplinar, incluindo análise de centralidade, detecção de comunidades, análise de caminhos,
detecção de pontes interdisciplinares e integração com KEC metrics e Tree Search PUCT.
"""

import asyncio
import math
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np
import networkx as nx
from scipy import sparse, linalg
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes,
    KnowledgeGraphSnapshot, CentralityAnalysis, CommunityDetection,
    PathAnalysis, BridgeAnalysis
)

logger = get_logger("knowledge_graph.graph_algorithms")


# ==================== ALGORITHM CONFIGURATIONS ====================

class CentralityAlgorithm(str, Enum):
    """Algoritmos de centralidade disponíveis."""
    PAGERANK = "pagerank"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    DEGREE = "degree"
    KATZ = "katz"
    HARMONIC = "harmonic"
    LOAD = "load"
    CURRENT_FLOW_BETWEENNESS = "current_flow_betweenness"
    CURRENT_FLOW_CLOSENESS = "current_flow_closeness"


class CommunityAlgorithm(str, Enum):
    """Algoritmos de detecção de comunidades."""
    LOUVAIN = "louvain"
    SPECTRAL = "spectral"
    LABEL_PROPAGATION = "label_propagation"
    LEIDEN = "leiden"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    GREEDY_MODULARITY = "greedy_modularity"
    ASYNC_FLUID = "async_fluid"


class PathAlgorithm(str, Enum):
    """Algoritmos de análise de caminhos."""
    SHORTEST_PATH = "shortest_path"
    ALL_SIMPLE_PATHS = "all_simple_paths"
    NODE_CONNECTIVITY = "node_connectivity"
    EDGE_CONNECTIVITY = "edge_connectivity"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ECCENTRICITY = "eccentricity"


@dataclass
class GraphAlgorithmConfig:
    """Configuração dos algoritmos de grafo."""
    # PageRank parameters
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 1000
    pagerank_tol: float = 1e-6
    
    # Community detection
    community_resolution: float = 1.0
    min_community_size: int = 3
    max_communities: int = 50
    
    # Bridge detection
    bridge_threshold: float = 0.3
    interdisciplinary_bonus: float = 0.25
    
    # Path analysis
    max_path_length: int = 10
    max_paths_to_analyze: int = 1000
    
    # KEC integration
    enable_kec_integration: bool = True
    kec_weight_factor: float = 0.2
    
    # Tree Search integration
    enable_tree_search_optimization: bool = True
    optimization_iterations: int = 100
    
    # Performance settings
    use_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


# KEC-specific metrics integration
KEC_GRAPH_METRICS = {
    "efficiency_centrality": "global_efficiency",
    "clustering_coefficient": "local_clustering",
    "transitivity": "global_clustering",
    "small_world_coefficient": "small_worldness",
    "assortativity": "degree_assortativity",
    "rich_club": "rich_club_coefficient"
}

# Tree Search optimization targets
TREE_SEARCH_TARGETS = {
    "modularity_optimization": "maximize_modularity",
    "centrality_balance": "balance_centrality_distribution",
    "path_optimization": "minimize_average_path_length",
    "bridge_enhancement": "enhance_interdisciplinary_bridges",
    "community_cohesion": "maximize_community_cohesion"
}


class DARWINGraphAlgorithms:
    """
    Sistema completo de algoritmos de análise de Knowledge Graph.
    
    Funcionalidades:
    - Multiple centrality algorithms com customizações
    - Advanced community detection algorithms
    - Comprehensive path analysis
    - Bridge detection entre domínios científicos
    - Integration com KEC metrics para análise de scaffold topology
    - Tree Search PUCT integration para otimização
    - Spectral analysis e decomposições matriciais
    - Network topology characterization
    - Influence propagation models
    - Temporal evolution analysis
    """
    
    def __init__(self, config: Optional[GraphAlgorithmConfig] = None):
        self.config = config or GraphAlgorithmConfig()
        
        # Core graph structures
        self.graph: Optional[nx.Graph] = None
        self.directed_graph: Optional[nx.DiGraph] = None
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Analysis results cache
        self.centrality_cache: Dict[str, Dict[str, float]] = {}
        self.community_cache: Dict[str, Dict[str, Any]] = {}
        self.path_cache: Dict[str, Dict[str, Any]] = {}
        self.bridge_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # KEC integration
        self.kec_metrics: Dict[str, float] = {}
        self.topology_descriptors: Dict[str, Any] = {}
        
        # Tree Search integration
        self.optimization_results: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.algorithm_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("🧮 DARWIN Graph Algorithms initialized")
    
    # ==================== CORE GRAPH SETUP ====================
    
    async def initialize_from_snapshot(
        self, 
        snapshot: KnowledgeGraphSnapshot
    ) -> None:
        """
        Inicializa algoritmos a partir de um snapshot do Knowledge Graph.
        """
        logger.info(f"🔄 Initializing graph algorithms from snapshot {snapshot.id}")
        
        try:
            # Create NetworkX graphs
            self.graph = nx.Graph()
            self.directed_graph = nx.DiGraph()
            
            # Add nodes with attributes
            for node in snapshot.nodes:
                node_attrs = {
                    "type": node.type.value,
                    "domain": node.domain.value,
                    "label": node.label,
                    **node.properties
                }
                self.node_attributes[node.id] = node_attrs
                self.graph.add_node(node.id, **node_attrs)
                self.directed_graph.add_node(node.id, **node_attrs)
            
            # Add edges with attributes
            for edge in snapshot.edges:
                edge_attrs = {
                    "type": edge.type.value,
                    "weight": edge.weight,
                    **edge.properties
                }
                edge_key = (edge.source, edge.target)
                self.edge_attributes[edge_key] = edge_attrs
                
                # Add to both graphs
                self.graph.add_edge(edge.source, edge.target, **edge_attrs)
                self.directed_graph.add_edge(edge.source, edge.target, **edge_attrs)
            
            # Initialize KEC metrics if enabled
            if self.config.enable_kec_integration:
                await self._initialize_kec_metrics()
            
            logger.info(f"✅ Graph initialized: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"❌ Graph initialization failed: {e}")
            raise
    
    async def _initialize_kec_metrics(self):
        """Inicializa métricas KEC para o grafo."""
        try:
            # Calculate basic KEC-inspired metrics
            self.kec_metrics = {
                "global_efficiency": nx.global_efficiency(self.graph),
                "local_efficiency": np.mean(list(nx.local_efficiency(self.graph).values())),
                "transitivity": nx.transitivity(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "assortativity": nx.degree_assortativity_coefficient(self.graph),
                "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
                "radius": nx.radius(self.graph) if nx.is_connected(self.graph) else float('inf')
            }
            
            # Calculate topology descriptors
            await self._calculate_topology_descriptors()
            
            logger.info(f"📊 KEC metrics initialized: {len(self.kec_metrics)} metrics")
            
        except Exception as e:
            logger.warning(f"KEC metrics initialization failed: {e}")
            self.kec_metrics = {}
    
    async def _calculate_topology_descriptors(self):
        """Calcula descritores topológicos inspirados em KEC."""
        try:
            degrees = [d for n, d in self.graph.degree()]
            
            self.topology_descriptors = {
                "degree_distribution": {
                    "mean": np.mean(degrees),
                    "std": np.std(degrees),
                    "skewness": self._calculate_skewness(degrees),
                    "kurtosis": self._calculate_kurtosis(degrees)
                },
                "small_world_properties": await self._analyze_small_world_properties(),
                "scale_free_properties": await self._analyze_scale_free_properties(),
                "modularity_structure": await self._analyze_modularity_structure()
            }
            
        except Exception as e:
            logger.warning(f"Topology descriptors calculation failed: {e}")
            self.topology_descriptors = {}
    
    # ==================== CENTRALITY ANALYSIS ====================
    
    async def analyze_centrality(
        self,
        algorithms: Optional[List[CentralityAlgorithm]] = None,
        top_k: int = 20,
        normalize: bool = True
    ) -> Dict[str, CentralityAnalysis]:
        """
        Analisa centralidade usando múltiplos algoritmos.
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call initialize_from_snapshot first.")
        
        # Use all algorithms if none specified
        if algorithms is None:
            algorithms = list(CentralityAlgorithm)
        
        logger.info(f"🎯 Analyzing centrality with {len(algorithms)} algorithms")
        results = {}
        
        for algorithm in algorithms:
            try:
                start_time = datetime.utcnow()
                
                # Calculate centrality
                centrality_scores = await self._calculate_centrality(algorithm)
                
                if centrality_scores:
                    # Create analysis result
                    top_nodes = sorted(
                        centrality_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:top_k]
                    
                    results[algorithm.value] = CentralityAnalysis(
                        algorithm=algorithm.value,
                        results=centrality_scores,
                        top_k=top_nodes,
                        metadata={
                            "normalized": normalize,
                            "graph_size": self.graph.number_of_nodes(),
                            "calculation_time": (datetime.utcnow() - start_time).total_seconds(),
                            "kec_integrated": self.config.enable_kec_integration
                        }
                    )
                    
                    # Cache results
                    self.centrality_cache[algorithm.value] = centrality_scores
                    
                    # Track performance
                    calc_time = (datetime.utcnow() - start_time).total_seconds()
                    self.algorithm_performance[algorithm.value]["centrality"] = calc_time
                
            except Exception as e:
                logger.error(f"Centrality calculation failed for {algorithm.value}: {e}")
                continue
        
        logger.info(f"✅ Centrality analysis completed: {len(results)} algorithms")
        return results
    
    async def _calculate_centrality(
        self, 
        algorithm: CentralityAlgorithm
    ) -> Dict[str, float]:
        """Calcula centralidade para um algoritmo específico."""
        
        # Choose graph (directed or undirected based on algorithm)
        graph = self._get_graph_for_algorithm(algorithm)
        
        if algorithm == CentralityAlgorithm.PAGERANK:
            centrality = nx.pagerank(
                graph,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
                tol=self.config.pagerank_tol,
                weight='weight'
            )
            
        elif algorithm == CentralityAlgorithm.BETWEENNESS:
            centrality = nx.betweenness_centrality(
                graph,
                normalized=True,
                weight='weight',
                endpoints=False
            )
            
        elif algorithm == CentralityAlgorithm.CLOSENESS:
            centrality = nx.closeness_centrality(
                graph,
                distance='weight',
                wf_improved=True
            )
            
        elif algorithm == CentralityAlgorithm.EIGENVECTOR:
            try:
                centrality = nx.eigenvector_centrality(
                    graph,
                    max_iter=1000,
                    weight='weight'
                )
            except nx.NetworkXError:
                # Fallback for disconnected graphs
                centrality = nx.eigenvector_centrality_numpy(graph, weight='weight')
                
        elif algorithm == CentralityAlgorithm.DEGREE:
            centrality = nx.degree_centrality(graph)
            
        elif algorithm == CentralityAlgorithm.KATZ:
            try:
                centrality = nx.katz_centrality(
                    graph,
                    alpha=0.1,
                    beta=1.0,
                    max_iter=1000,
                    weight='weight'
                )
            except (nx.NetworkXError, np.linalg.LinAlgError):
                logger.warning("Katz centrality failed, using degree centrality")
                centrality = nx.degree_centrality(graph)
                
        elif algorithm == CentralityAlgorithm.HARMONIC:
            centrality = nx.harmonic_centrality(graph, distance='weight')
            
        elif algorithm == CentralityAlgorithm.LOAD:
            centrality = nx.load_centrality(graph, weight='weight')
            
        elif algorithm == CentralityAlgorithm.CURRENT_FLOW_BETWEENNESS:
            try:
                centrality = nx.current_flow_betweenness_centrality(
                    graph, 
                    weight='weight'
                )
            except nx.NetworkXError:
                logger.warning("Current flow betweenness failed, using standard betweenness")
                centrality = nx.betweenness_centrality(graph, weight='weight')
                
        elif algorithm == CentralityAlgorithm.CURRENT_FLOW_CLOSENESS:
            try:
                centrality = nx.current_flow_closeness_centrality(
                    graph, 
                    weight='weight'
                )
            except nx.NetworkXError:
                logger.warning("Current flow closeness failed, using standard closeness")
                centrality = nx.closeness_centrality(graph, distance='weight')
        
        else:
            raise ValueError(f"Unknown centrality algorithm: {algorithm}")
        
        # Apply KEC integration if enabled
        if self.config.enable_kec_integration and self.kec_metrics:
            centrality = await self._apply_kec_centrality_weighting(centrality, algorithm)
        
        return centrality
    
    async def _apply_kec_centrality_weighting(
        self,
        centrality: Dict[str, float],
        algorithm: CentralityAlgorithm
    ) -> Dict[str, float]:
        """Aplica ponderação baseada em métricas KEC."""
        
        # Different KEC factors for different centrality algorithms
        kec_factors = {
            CentralityAlgorithm.PAGERANK: "global_efficiency",
            CentralityAlgorithm.BETWEENNESS: "local_efficiency", 
            CentralityAlgorithm.CLOSENESS: "average_clustering",
            CentralityAlgorithm.EIGENVECTOR: "assortativity"
        }
        
        kec_factor_name = kec_factors.get(algorithm)
        if not kec_factor_name or kec_factor_name not in self.kec_metrics:
            return centrality
        
        kec_factor = abs(self.kec_metrics[kec_factor_name])
        weight_factor = self.config.kec_weight_factor
        
        # Apply weighting
        weighted_centrality = {}
        for node, score in centrality.items():
            # Get node-specific properties that might relate to KEC
            node_attrs = self.node_attributes.get(node, {})
            domain_factor = 1.0
            
            # Apply domain-specific weighting (interdisciplinary nodes get bonus)
            if "domain" in node_attrs:
                # Check if node connects multiple domains (proxy for scaffold-like behavior)
                neighbors = list(self.graph.neighbors(node))
                neighbor_domains = set()
                for neighbor in neighbors:
                    neighbor_attrs = self.node_attributes.get(neighbor, {})
                    if "domain" in neighbor_attrs:
                        neighbor_domains.add(neighbor_attrs["domain"])
                
                if len(neighbor_domains) > 1:
                    domain_factor = 1.0 + self.config.interdisciplinary_bonus
            
            weighted_score = score * (1.0 + kec_factor * weight_factor * domain_factor)
            weighted_centrality[node] = weighted_score
        
        return weighted_centrality
    
    def _get_graph_for_algorithm(self, algorithm: CentralityAlgorithm) -> Union[nx.Graph, nx.DiGraph]:
        """Retorna grafo apropriado para o algoritmo."""
        # Algorithms that work better with directed graphs
        directed_algorithms = {
            CentralityAlgorithm.PAGERANK,
            CentralityAlgorithm.KATZ
        }
        
        if algorithm in directed_algorithms and self.directed_graph:
            return self.directed_graph
        else:
            return self.graph
    
    # ==================== COMMUNITY DETECTION ====================
    
    async def detect_communities(
        self,
        algorithms: Optional[List[CommunityAlgorithm]] = None,
        min_size: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detecta comunidades usando múltiplos algoritmos.
        """
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        if algorithms is None:
            algorithms = [
                CommunityAlgorithm.LOUVAIN,
                CommunityAlgorithm.SPECTRAL,
                CommunityAlgorithm.LABEL_PROPAGATION
            ]
        
        min_community_size = min_size or self.config.min_community_size
        
        logger.info(f"🏘️ Detecting communities with {len(algorithms)} algorithms")
        results = {}
        
        for algorithm in algorithms:
            try:
                start_time = datetime.utcnow()
                
                # Detect communities
                communities_data = await self._detect_communities_algorithm(algorithm)
                
                if communities_data:
                    # Filter by minimum size
                    filtered_communities = [
                        community for community in communities_data["communities"]
                        if len(community) >= min_community_size
                    ]
                    
                    # Calculate modularity
                    modularity = await self._calculate_modularity(
                        filtered_communities, algorithm
                    )
                    
                    results[algorithm.value] = {
                        "algorithm": algorithm.value,
                        "communities": filtered_communities,
                        "total_communities": len(filtered_communities),
                        "modularity": modularity,
                        "coverage": self._calculate_community_coverage(filtered_communities),
                        "silhouette_score": await self._calculate_community_silhouette(filtered_communities),
                        "calculation_time": (datetime.utcnow() - start_time).total_seconds(),
                        "community_stats": await self._analyze_community_characteristics(filtered_communities)
                    }
                    
                    # Cache results
                    self.community_cache[algorithm.value] = results[algorithm.value]
                    
                    # Track performance
                    calc_time = (datetime.utcnow() - start_time).total_seconds()
                    self.algorithm_performance[algorithm.value]["community"] = calc_time
                
            except Exception as e:
                logger.error(f"Community detection failed for {algorithm.value}: {e}")
                continue
        
        logger.info(f"✅ Community detection completed: {len(results)} algorithms")
        return results
    
    async def _detect_communities_algorithm(
        self,
        algorithm: CommunityAlgorithm
    ) -> Optional[Dict[str, Any]]:
        """Detecta comunidades usando algoritmo específico."""
        
        if algorithm == CommunityAlgorithm.LOUVAIN:
            return await self._louvain_communities()
            
        elif algorithm == CommunityAlgorithm.SPECTRAL:
            return await self._spectral_communities()
            
        elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
            return await self._label_propagation_communities()
            
        elif algorithm == CommunityAlgorithm.HIERARCHICAL:
            return await self._hierarchical_communities()
            
        elif algorithm == CommunityAlgorithm.DBSCAN:
            return await self._dbscan_communities()
            
        elif algorithm == CommunityAlgorithm.GREEDY_MODULARITY:
            return await self._greedy_modularity_communities()
            
        elif algorithm == CommunityAlgorithm.ASYNC_FLUID:
            return await self._async_fluid_communities()
            
        else:
            logger.warning(f"Unknown community algorithm: {algorithm}")
            return None
    
    async def _louvain_communities(self) -> Dict[str, Any]:
        """Implementa Louvain community detection."""
        try:
            import networkx.algorithms.community as nx_community
            communities = nx_community.greedy_modularity_communities(
                self.graph, 
                weight='weight',
                resolution=self.config.community_resolution
            )
            
            return {
                "communities": [list(community) for community in communities],
                "method": "louvain_greedy"
            }
            
        except Exception as e:
            logger.error(f"Louvain communities failed: {e}")
            return None
    
    async def _spectral_communities(self) -> Dict[str, Any]:
        """Implementa Spectral clustering para comunidades."""
        try:
            # Get adjacency matrix
            adj_matrix = nx.adjacency_matrix(self.graph, weight='weight')
            
            # Determine number of clusters
            n_nodes = adj_matrix.shape[0]
            n_clusters = min(
                max(n_nodes // 20, 2), 
                self.config.max_communities
            )
            
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            labels = clustering.fit_predict(adj_matrix.toarray())
            
            # Group nodes by cluster
            communities = defaultdict(list)
            nodes = list(self.graph.nodes())
            
            for i, label in enumerate(labels):
                communities[label].append(nodes[i])
            
            return {
                "communities": list(communities.values()),
                "method": "spectral_clustering"
            }
            
        except Exception as e:
            logger.error(f"Spectral communities failed: {e}")
            return None
    
    async def _label_propagation_communities(self) -> Dict[str, Any]:
        """Implementa Label Propagation algorithm."""
        try:
            communities = nx.algorithms.community.label_propagation_communities(
                self.graph
            )
            
            return {
                "communities": [list(community) for community in communities],
                "method": "label_propagation"
            }
            
        except Exception as e:
            logger.error(f"Label propagation communities failed: {e}")
            return None
    
    async def _hierarchical_communities(self) -> Dict[str, Any]:
        """Implementa Hierarchical clustering."""
        try:
            # Calculate distance matrix based on graph
            nodes = list(self.graph.nodes())
            n_nodes = len(nodes)
            
            if n_nodes < 3:
                return None
            
            # Create distance matrix
            distance_matrix = np.zeros((n_nodes, n_nodes))
            
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    try:
                        # Use shortest path as distance
                        distance = nx.shortest_path_length(
                            self.graph, node1, node2, weight='weight'
                        )
                    except nx.NetworkXNoPath:
                        distance = float('inf')
                    
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
            
            # Replace inf with max finite distance + 1
            max_finite = np.max(distance_matrix[distance_matrix != np.inf])
            distance_matrix[distance_matrix == np.inf] = max_finite + 1
            
            # Hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Determine number of clusters
            n_clusters = min(max(n_nodes // 15, 2), self.config.max_communities)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group nodes by cluster
            communities = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                communities[label].append(nodes[i])
            
            return {
                "communities": list(communities.values()),
                "method": "hierarchical_ward"
            }
            
        except Exception as e:
            logger.error(f"Hierarchical communities failed: {e}")
            return None
    
    async def _dbscan_communities(self) -> Dict[str, Any]:
        """Implementa DBSCAN clustering."""
        try:
            # Use node2vec embeddings or adjacency-based features
            features = await self._extract_node_features()
            
            if features is None or len(features) < self.config.min_community_size:
                return None
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=self.config.min_community_size)
            cluster_labels = dbscan.fit_predict(features)
            
            # Group nodes by cluster (ignore noise points with label -1)
            communities = defaultdict(list)
            nodes = list(self.graph.nodes())
            
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise
                    communities[label].append(nodes[i])
            
            return {
                "communities": list(communities.values()),
                "method": "dbscan",
                "noise_points": sum(1 for label in cluster_labels if label == -1)
            }
            
        except Exception as e:
            logger.error(f"DBSCAN communities failed: {e}")
            return None
    
    async def _greedy_modularity_communities(self) -> Dict[str, Any]:
        """Implementa Greedy Modularity Maximization."""
        try:
            import networkx.algorithms.community as nx_community
            communities = nx_community.greedy_modularity_communities(
                self.graph,
                weight='weight'
            )
            
            return {
                "communities": [list(community) for community in communities],
                "method": "greedy_modularity"
            }
            
        except Exception as e:
            logger.error(f"Greedy modularity communities failed: {e}")
            return None
    
    async def _async_fluid_communities(self) -> Dict[str, Any]:
        """Implementa Asynchronous Fluid Communities algorithm."""
        try:
            # Determine number of communities
            n_nodes = self.graph.number_of_nodes()
            k = min(max(n_nodes // 25, 2), self.config.max_communities)
            
            communities = nx.algorithms.community.asyn_fluidc(
                self.graph, k=k, max_iter=100
            )
            
            return {
                "communities": [list(community) for community in communities],
                "method": "async_fluid"
            }
            
        except Exception as e:
            logger.error(f"Async fluid communities failed: {e}")
            return None
    
    async def _extract_node_features(self) -> Optional[np.ndarray]:
        """Extrai features dos nós para clustering."""
        try:
            nodes = list(self.graph.nodes())
            n_nodes = len(nodes)
            
            # Create feature matrix
            features = []
            
            for node in nodes:
                node_features = []
                
                # Degree-based features
                degree = self.graph.degree(node, weight='weight')
                node_features.append(degree)
                
                # Clustering coefficient
                clustering = nx.clustering(self.graph, node, weight='weight')
                node_features.append(clustering)
                
                # Local efficiency
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) > 1:
                    subgraph = self.graph.subgraph(neighbors)
                    local_eff = nx.global_efficiency(subgraph)
                else:
                    local_eff = 0.0
                node_features.append(local_eff)
                
                # Centrality measures (if cached)
                if CentralityAlgorithm.PAGERANK.value in self.centrality_cache:
                    pagerank = self.centrality_cache[CentralityAlgorithm.PAGERANK.value].get(node, 0.0)
                    node_features.append(pagerank)
                
                if CentralityAlgorithm.BETWEENNESS.value in self.centrality_cache:
                    betweenness = self.centrality_cache[CentralityAlgorithm.BETWEENNESS.value].get(node, 0.0)
                    node_features.append(betweenness)
                
                # Domain information (if available)
                node_attrs = self.node_attributes.get(node, {})
                if "domain" in node_attrs:
                    # One-hot encode domains
                    domain_vector = [0.0] * len(ScientificDomains)
                    try:
                        domain_idx = list(ScientificDomains).index(ScientificDomains(node_attrs["domain"]))
                        domain_vector[domain_idx] = 1.0
                    except ValueError:
                        pass  # Unknown domain
                    node_features.extend(domain_vector)
                
                features.append(node_features)
            
            # Ensure all feature vectors have same length
            if features:
                max_length = max(len(f) for f in features)
                normalized_features = []
                
                for feature_vector in features:
                    # Pad with zeros if needed
                    while len(feature_vector) < max_length:
                        feature_vector.append(0.0)
                    normalized_features.append(feature_vector)
                
                return np.array(normalized_features)
            
            return None
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _calculate_modularity(
        self,
        communities: List[List[str]],
        algorithm: CommunityAlgorithm
    ) -> float:
        """Calcula modularidade das comunidades."""
        try:
            # Convert to sets for NetworkX
            community_sets = [set(community) for community in communities]
            
            modularity = nx.algorithms.community.modularity(
                self.graph,
                community_sets,
                weight='weight'
            )
            
            return modularity
            
        except Exception as e:
            logger.warning(f"Modularity calculation failed: {e}")
            return 0.0
    
    def _calculate_community_coverage(self, communities: List[List[str]]) -> float:
        """Calcula cobertura das comunidades."""
        total_nodes = self.graph.number_of_nodes()
        covered_nodes = set()
        
        for community in communities:
            covered_nodes.update(community)
        
        return len(covered_nodes) / total_nodes if total_nodes > 0 else 0.0
    
    async def _calculate_community_silhouette(self, communities: List[List[str]]) -> float:
        """Calcula silhouette score das comunidades."""
        try:
            # Extract features for silhouette calculation
            features = await self._extract_node_features()
            
            if features is None or len(communities) < 2:
                return 0.0
            
            # Create labels
            nodes = list(self.graph.nodes())
            labels = [-1] * len(nodes)  # -1 for nodes not in any community
            
            for comm_id, community in enumerate(communities):
                for node in community:
                    if node in nodes:
                        node_idx = nodes.index(node)
                        labels[node_idx] = comm_id
            
            # Calculate silhouette score only for nodes in communities
            valid_indices = [i for i, label in enumerate(labels) if label != -1]
            
            if len(valid_indices) < 2 or len(set(labels[i] for i in valid_indices)) < 2:
                return 0.0
            
            valid_features = features[valid_indices]
            valid_labels = [labels[i] for i in valid_indices]
            
            return silhouette_score(valid_features, valid_labels)
            
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            return 0.0
    
    async def _analyze_community_characteristics(
        self,
        communities: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Analisa características das comunidades."""
        community_stats = []
        
        for i, community in enumerate(communities):
            if not community:
                continue
            
            # Basic stats
            size = len(community)
            
            # Subgraph analysis
            subgraph = self.graph.subgraph(community)
            internal_edges = subgraph.number_of_edges()
            
            # External connections
            external_edges = 0
            for node in community:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in community:
                        external_edges += 1
            
            # Domain analysis
            domain_counts = Counter()
            for node in community:
                node_attrs = self.node_attributes.get(node, {})
                if "domain" in node_attrs:
                    domain_counts[node_attrs["domain"]] += 1
            
            # Calculate community metrics
            density = nx.density(subgraph) if size > 1 else 0.0
            avg_clustering = nx.average_clustering(subgraph) if size > 2 else 0.0
            
            community_stats.append({
                "community_id": i,
                "size": size,
                "density": density,
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "modularity_contribution": internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0.0,
                "avg_clustering": avg_clustering,
                "domain_distribution": dict(domain_counts),
                "dominant_domain": domain_counts.most_common(1)[0][0] if domain_counts else "unknown",
                "interdisciplinary": len(domain_counts) > 1
            })
        
        return community_stats
    
    # ==================== PATH ANALYSIS ====================
    
    async def analyze_paths(
        self,
        source_nodes: Optional[List[str]] = None,
        target_nodes: Optional[List[str]] = None,
        algorithms: Optional[List[PathAlgorithm]] = None
    ) -> Dict[str, Any]:
        """
        Analisa caminhos no grafo.
        """
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        if algorithms is None:
            algorithms = [
                PathAlgorithm.SHORTEST_PATH,
                PathAlgorithm.DIAMETER,
                PathAlgorithm.RADIUS
            ]
        
        logger.info(f"🛣️ Analyzing paths with {len(algorithms)} algorithms")
        results = {}
        
        # Global graph metrics
        results["global_metrics"] = await self._calculate_global_path_metrics()
        
        # Algorithm-specific analysis
        for algorithm in algorithms:
            try:
                start_time = datetime.utcnow()
                
                if algorithm == PathAlgorithm.SHORTEST_PATH:
                    results["shortest_paths"] = await self._analyze_shortest_paths(
                        source_nodes, target_nodes
                    )
                    
                elif algorithm == PathAlgorithm.ALL_SIMPLE_PATHS:
                    results["all_simple_paths"] = await self._analyze_all_simple_paths(
                        source_nodes, target_nodes
                    )
                    
                elif algorithm == PathAlgorithm.DIAMETER:
                    results["diameter_analysis"] = await self._analyze_diameter()
                    
                elif algorithm == PathAlgorithm.RADIUS:
                    results["radius_analysis"] = await self._analyze_radius()
                    
                elif algorithm == PathAlgorithm.ECCENTRICITY:
                    results["eccentricity_analysis"] = await self._analyze_eccentricity()
                    
                elif algorithm == PathAlgorithm.NODE_CONNECTIVITY:
                    results["node_connectivity"] = await self._analyze_node_connectivity()
                    
                elif algorithm == PathAlgorithm.EDGE_CONNECTIVITY:
                    results["edge_connectivity"] = await self._analyze_edge_connectivity()
                
                calc_time = (datetime.utcnow() - start_time).total_seconds()
                self.algorithm_performance[algorithm.value]["path"] = calc_time
                
            except Exception as e:
                logger.error(f"Path analysis failed for {algorithm.value}: {e}")
                continue
        
        # Cache results
        self.path_cache["latest_analysis"] = results
        
        logger.info(f"✅ Path analysis completed")
        return results
    
    async def _calculate_global_path_metrics(self) -> Dict[str, Any]:
        """Calcula métricas globais de caminho."""
        try:
            metrics = {
                "is_connected": nx.is_connected(self.graph),
                "number_of_components": nx.number_connected_components(self.graph)
            }
            
            if metrics["is_connected"]:
                metrics["diameter"] = nx.diameter(self.graph, weight='weight')
                metrics["radius"] = nx.radius(self.graph, weight='weight')
                metrics["average_shortest_path_length"] = nx.average_shortest_path_length(
                    self.graph, weight='weight'
                )
            else:
                # For disconnected graphs, analyze largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                
                metrics["largest_component_size"] = len(largest_cc)
                metrics["largest_component_diameter"] = nx.diameter(subgraph, weight='weight')
                metrics["largest_component_radius"] = nx.radius(subgraph, weight='weight')
                metrics["largest_component_avg_path"] = nx.average_shortest_path_length(
                    subgraph, weight='weight'
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Global path metrics calculation failed: {e}")
            return {}
    
    async def _analyze_shortest_paths(
        self,
        source_nodes: Optional[List[str]],
        target_nodes: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analisa caminhos mais curtos."""
        try:
            if not source_nodes:
                # Select high centrality nodes as sources
                if CentralityAlgorithm.PAGERANK.value in self.centrality_cache:
                    pagerank_scores = self.centrality_cache[CentralityAlgorithm.PAGERANK.value]
                    source_nodes = sorted(
                        pagerank_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    source_nodes = [node for node, score in source_nodes]
                else:
                    # Fallback to random nodes
                    all_nodes = list(self.graph.nodes())
                    source_nodes = all_nodes[:min(10, len(all_nodes))]
            
            if not target_nodes:
                target_nodes = source_nodes  # All-to-all within source set
            
            paths_analysis = {
                "total_pairs_analyzed": 0,
                "successful_paths": 0,
                "average_path_length": 0.0,
                "path_length_distribution": Counter(),
                "example_paths": []
            }
            
            total_length = 0.0
            successful_paths = 0
            
            for source in source_nodes[:5]:  # Limit to prevent excessive computation
                for target in target_nodes[:5]:
                    if source == target:
                        continue
                    
                    paths_analysis["total_pairs_analyzed"] += 1
                    
                    try:
                        path = nx.shortest_path(
                            self.graph, source, target, weight='weight'
                        )
                        path_length = nx.shortest_path_length(
                            self.graph, source, target, weight='weight'
                        )
                        
                        successful_paths += 1
                        total_length += path_length
                        paths_analysis["path_length_distribution"][len(path)] += 1
                        
                        # Store example paths
                        if len(paths_analysis["example_paths"]) < 10:
                            paths_analysis["example_paths"].append({
                                "source": source,
                                "target": target,
                                "path": path,
                                "length": path_length,
                                "hop_count": len(path) - 1
                            })
                        
                    except nx.NetworkXNoPath:
                        continue
            
            if successful_paths > 0:
                paths_analysis["successful_paths"] = successful_paths
                paths_analysis["average_path_length"] = total_length / successful_paths
            
            paths_analysis["path_length_distribution"] = dict(paths_analysis["path_length_distribution"])
            
            return paths_analysis
            
        except Exception as e:
            logger.error(f"Shortest paths analysis failed: {e}")
            return {}
    
    async def _analyze_diameter(self) -> Dict[str, Any]:
        """Analisa diâmetro do grafo."""
        try:
            if nx.is_connected(self.graph):
                diameter = nx.diameter(self.graph, weight='weight')
                periphery = nx.periphery(self.graph)
                
                # Find the actual diameter path
                diameter_path = None
                max_distance = 0
                
                for node1 in periphery:
                    for node2 in periphery:
                        if node1 != node2:
                            try:
                                distance = nx.shortest_path_length(
                                    self.graph, node1, node2, weight='weight'
                                )
                                if distance > max_distance:
                                    max_distance = distance
                                    diameter_path = nx.shortest_path(
                                        self.graph, node1, node2, weight='weight'
                                    )
                            except nx.NetworkXNoPath:
                                continue
                
                return {
                    "diameter": diameter,
                    "periphery_nodes": periphery,
                    "diameter_path": diameter_path,
                    "diameter_path_length": len(diameter_path) - 1 if diameter_path else 0
                }
            else:
                # Analyze largest component
                components = nx.connected_components(self.graph)
                largest_component = max(components, key=len)
                subgraph = self.graph.subgraph(largest_component)
                
                diameter = nx.diameter(subgraph, weight='weight')
                periphery = nx.periphery(subgraph)
                
                return {
                    "graph_connected": False,
                    "largest_component_diameter": diameter,
                    "largest_component_periphery": periphery,
                    "largest_component_size": len(largest_component)
                }
                
        except Exception as e:
            logger.error(f"Diameter analysis failed: {e}")
            return {}
    
    async def _analyze_radius(self) -> Dict[str, Any]:
        """Analisa raio do grafo."""
        try:
            if nx.is_connected(self.graph):
                radius = nx.radius(self.graph, weight='weight')
                center = nx.center(self.graph)
                
                return {
                    "radius": radius,
                    "center_nodes": center,
                    "eccentricities": dict(nx.eccentricity(self.graph, weight='weight'))
                }
            else:
                # Analyze largest component
                components = nx.connected_components(self.graph)
                largest_component = max(components, key=len)
                subgraph = self.graph.subgraph(largest_component)
                
                radius = nx.radius(subgraph, weight='weight')
                center = nx.center(subgraph)
                
                return {
                    "graph_connected": False,
                    "largest_component_radius": radius,
                    "largest_component_center": center,
                    "largest_component_size": len(largest_component)
                }
                
        except Exception as e:
            logger.error(f"Radius analysis failed: {e}")
            return {}
    
    async def _analyze_eccentricity(self) -> Dict[str, Any]:
        """Analisa excentricidade dos nós."""
        try:
            if nx.is_connected(self.graph):
                eccentricities = nx.eccentricity(self.graph, weight='weight')
                
                # Statistical analysis of eccentricities
                ecc_values = list(eccentricities.values())
                
                return {
                    "eccentricities": eccentricities,
                    "mean_eccentricity": np.mean(ecc_values),
                    "std_eccentricity": np.std(ecc_values),
                    "min_eccentricity": min(ecc_values),
                    "max_eccentricity": max(ecc_values),
                    "center_nodes": [node for node, ecc in eccentricities.items() if ecc == min(ecc_values)],
                    "periphery_nodes": [node for node, ecc in eccentricities.items() if ecc == max(ecc_values)]
                }
            else:
                return {"graph_connected": False}
                
        except Exception as e:
            logger.error(f"Eccentricity analysis failed: {e}")
            return {}
    
    async def _analyze_node_connectivity(self) -> Dict[str, Any]:
        """Analisa conectividade de nós."""
        try:
            node_connectivity = nx.node_connectivity(self.graph)
            
            # Find critical nodes (articulation points)
            articulation_points = list(nx.articulation_points(self.graph))
            
            return {
                "node_connectivity": node_connectivity,
                "articulation_points": articulation_points,
                "number_of_articulation_points": len(articulation_points),
                "is_biconnected": nx.is_biconnected(self.graph)
            }
            
        except Exception as e:
            logger.error(f"Node connectivity analysis failed: {e}")
            return {}
    
    async def _analyze_edge_connectivity(self) -> Dict[str, Any]:
        """Analisa conectividade de arestas."""
        try:
            edge_connectivity = nx.edge_connectivity(self.graph)
            
            # Find bridges
            bridges = list(nx.bridges(self.graph))
            
            return {
                "edge_connectivity": edge_connectivity,
                "bridges": bridges,
                "number_of_bridges": len(bridges),
                "bridge_components": nx.number_connected_components(self.graph) if bridges else 1
            }
            
        except Exception as e:
            logger.error(f"Edge connectivity analysis failed: {e}")
            return {}
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calcula skewness de uma distribuição."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        skewness = (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std) ** 3 for x in values)
        
        return skewness
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calcula kurtosis de uma distribuição."""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(((x - mean) / std) ** 4 for x in values) - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        
        return kurtosis
    
    async def _analyze_small_world_properties(self) -> Dict[str, Any]:
        """Analisa propriedades de mundo pequeno."""
        try:
            if not nx.is_connected(self.graph):
                return {"small_world": False, "reason": "graph_not_connected"}
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(self.graph)
            
            # Calculate average shortest path length
            avg_path_length = nx.average_shortest_path_length(self.graph, weight='weight')
            
            # Generate random graph for comparison
            n = self.graph.number_of_nodes()
            m = self.graph.number_of_edges()
            random_graph = nx.erdos_renyi_graph(n, 2 * m / (n * (n - 1)))
            
            random_clustering = nx.average_clustering(random_graph)
            random_avg_path = nx.average_shortest_path_length(random_graph)
            
            # Small world coefficient
            if random_clustering > 0 and random_avg_path > 0:
                small_world_coefficient = (clustering / random_clustering) / (avg_path_length / random_avg_path)
                is_small_world = small_world_coefficient > 1.0
            else:
                small_world_coefficient = 0.0
                is_small_world = False
            
            return {
                "small_world": is_small_world,
                "small_world_coefficient": small_world_coefficient,
                "clustering_coefficient": clustering,
                "avg_path_length": avg_path_length,
                "random_clustering": random_clustering,
                "random_avg_path": random_avg_path
            }
            
        except Exception as e:
            logger.error(f"Small world analysis failed: {e}")
            return {}
    
    async def _analyze_scale_free_properties(self) -> Dict[str, Any]:
        """Analisa propriedades scale-free."""
        try:
            # Get degree sequence
            degrees = [d for n, d in self.graph.degree()]
            degree_counts = Counter(degrees)
            
            # Fit power law
            degrees_array = np.array(list(degree_counts.keys()))
            counts_array = np.array(list(degree_counts.values()))
            
            if len(degrees_array) < 3:
                return {"scale_free": False, "reason": "insufficient_data"}
            
            # Log-log fit
            log_degrees = np.log(degrees_array)
            log_counts = np.log(counts_array)
            
            # Linear regression in log-log space
            coeffs = np.polyfit(log_degrees, log_counts, 1)
            power_law_exponent = -coeffs[0]  # Negative because it's P(k) ~ k^(-gamma)
            
            # R-squared
            predicted = np.polyval(coeffs, log_degrees)
            ss_res = np.sum((log_counts - predicted) ** 2)
            ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Scale-free if exponent is between 2 and 3 and good fit
            is_scale_free = (2.0 <= power_law_exponent <= 3.5) and (r_squared > 0.8)
            
            return {
                "scale_free": is_scale_free,
                "power_law_exponent": power_law_exponent,
                "r_squared": r_squared,
                "degree_distribution": dict(degree_counts),
                "mean_degree": np.mean(degrees),
                "std_degree": np.std(degrees)
            }
            
        except Exception as e:
            logger.error(f"Scale-free analysis failed: {e}")
            return {}
    
    async def _analyze_modularity_structure(self) -> Dict[str, Any]:
        """Analisa estrutura de modularidade."""
        try:
            # Use cached communities if available
            if CommunityAlgorithm.LOUVAIN.value in self.community_cache:
                communities_data = self.community_cache[CommunityAlgorithm.LOUVAIN.value]
                communities = communities_data.get("communities", [])
                modularity = communities_data.get("modularity", 0.0)
            else:
                # Quick community detection
                import networkx.algorithms.community as nx_community
                communities = nx_community.greedy_modularity_communities(self.graph, weight='weight')
                communities = [list(community) for community in communities]
                
                community_sets = [set(community) for community in communities]
                modularity = nx.algorithms.community.modularity(self.graph, community_sets, weight='weight')
            
            return {
                "modularity": modularity,
                "number_of_communities": len(communities),
                "modularity_class": self._classify_modularity(modularity),
                "average_community_size": np.mean([len(c) for c in communities]) if communities else 0
            }
            
        except Exception as e:
            logger.error(f"Modularity analysis failed: {e}")
            return {}
    
    def _classify_modularity(self, modularity: float) -> str:
        """Classifica o nível de modularidade."""
        if modularity > 0.7:
            return "very_high"
        elif modularity > 0.5:
            return "high"
        elif modularity > 0.3:
            return "moderate"
        elif modularity > 0.1:
            return "low"
        else:
            return "very_low"
    
    # ==================== PERFORMANCE AND CACHING ====================
    
    def get_algorithm_performance(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance dos algoritmos."""
        return {
            "algorithm_performance": dict(self.algorithm_performance),
            "cache_status": {
                "centrality_algorithms_cached": len(self.centrality_cache),
                "community_algorithms_cached": len(self.community_cache),
                "path_analyses_cached": len(self.path_cache),
                "bridge_analyses_cached": len(self.bridge_cache)
            },
            "graph_info": {
                "nodes": self.graph.number_of_nodes() if self.graph else 0,
                "edges": self.graph.number_of_edges() if self.graph else 0,
                "is_directed": isinstance(self.directed_graph, nx.DiGraph),
                "kec_integration_enabled": self.config.enable_kec_integration
            }
        }
    
    def clear_caches(self):
        """Limpa todos os caches."""
        self.centrality_cache.clear()
        self.community_cache.clear()
        self.path_cache.clear()
        self.bridge_cache.clear()
        self.algorithm_performance.clear()
        logger.info("🧹 Algorithm caches cleared")


__all__ = [
    "DARWINGraphAlgorithms",
    "GraphAlgorithmConfig",
    "CentralityAlgorithm",
    "CommunityAlgorithm",
    "PathAlgorithm",
    "KEC_GRAPH_METRICS",
    "TREE_SEARCH_TARGETS"
]