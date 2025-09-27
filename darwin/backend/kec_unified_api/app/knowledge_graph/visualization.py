"""DARWIN Knowledge Graph Visualization - Sistema de Visualização Web Interativa

Sistema épico que gera visualizações interativas do Knowledge Graph interdisciplinar
com suporte a D3.js, Plotly, layouts dinâmicos, filtros por domínio, zoom/pan,
navegação inteligente e export em múltiplos formatos.
"""

import asyncio
import json
import math
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..core.logging import get_logger
from ..models.knowledge_graph_models import (
    ScientificDomains, NodeTypes, EdgeTypes,
    GraphVisualizationConfig, GraphVisualizationData,
    KnowledgeGraphSnapshot
)

logger = get_logger("knowledge_graph.visualization")


# ==================== VISUALIZATION CONFIGURATIONS ====================

class LayoutType(str, Enum):
    """Tipos de layout para visualização."""
    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    SPRING = "spring"
    KAMADA_KAWAI = "kamada_kawai"
    SPECTRAL = "spectral"
    PLANAR = "planar"
    SHELL = "shell"
    SPIRAL = "spiral"
    TREE = "tree"
    RADIAL = "radial"
    GRID = "grid"
    CLUSTER_BASED = "cluster_based"
    DOMAIN_SEPARATED = "domain_separated"
    TEMPORAL = "temporal"


class ColorScheme(str, Enum):
    """Esquemas de cores para visualização."""
    DOMAIN = "domain"
    NODE_TYPE = "node_type"
    CENTRALITY = "centrality"
    COMMUNITY = "community"
    TEMPORAL = "temporal"
    CUSTOM = "custom"
    GRADIENT = "gradient"
    CATEGORICAL = "categorical"


class InteractionMode(str, Enum):
    """Modos de interação."""
    EXPLORE = "explore"
    NAVIGATE = "navigate"
    ANALYZE = "analyze"
    EDIT = "edit"
    PRESENT = "present"


@dataclass
class VisualizationConfig:
    """Configuração completa de visualização."""
    # Layout configuration
    layout_type: LayoutType = LayoutType.FORCE_DIRECTED
    layout_params: Dict[str, Any] = None
    
    # Visual encoding
    color_scheme: ColorScheme = ColorScheme.DOMAIN
    node_size_attribute: Optional[str] = "degree"
    edge_width_attribute: Optional[str] = "weight"
    
    # Filtering
    show_node_labels: bool = True
    show_edge_labels: bool = False
    filter_domains: Optional[List[ScientificDomains]] = None
    filter_node_types: Optional[List[NodeTypes]] = None
    min_node_degree: int = 0
    max_nodes: int = 1000
    max_edges: int = 5000
    
    # Interaction
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_drag: bool = True
    enable_hover: bool = True
    enable_click: bool = True
    
    # Performance
    use_webgl: bool = True
    enable_clustering: bool = True
    cluster_threshold: int = 100
    
    # Export options
    width: int = 1200
    height: int = 800
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.layout_params is None:
            self.layout_params = {}
        if self.export_formats is None:
            self.export_formats = ["svg", "png", "json"]


# Color palettes for different schemes
DOMAIN_COLORS = {
    ScientificDomains.BIOMATERIALS: "#FF6B6B",
    ScientificDomains.NEUROSCIENCE: "#4ECDC4", 
    ScientificDomains.PHILOSOPHY: "#45B7D1",
    ScientificDomains.QUANTUM_MECHANICS: "#96CEB4",
    ScientificDomains.PSYCHIATRY: "#FFEAA7",
    ScientificDomains.MATHEMATICS: "#DDA0DD",
    ScientificDomains.INTERDISCIPLINARY: "#FFD93D"
}

NODE_TYPE_COLORS = {
    NodeTypes.PAPER: "#E74C3C",
    NodeTypes.CONCEPT: "#3498DB", 
    NodeTypes.AUTHOR: "#2ECC71",
    NodeTypes.METHOD: "#F39C12",
    NodeTypes.KEYWORD: "#9B59B6",
    NodeTypes.INSIGHT: "#1ABC9C",
    NodeTypes.DISCOVERY: "#E67E22",
    NodeTypes.METRIC: "#34495E"
}

# D3.js force simulation parameters
FORCE_SIMULATION_DEFAULTS = {
    "charge_strength": -300,
    "link_distance": 50,
    "link_strength": 1,
    "collision_radius": 5,
    "center_strength": 0.1,
    "alpha": 1.0,
    "alpha_decay": 0.0228,
    "velocity_decay": 0.4
}


class DARWINGraphVisualization:
    """
    Sistema completo de visualização interativa do Knowledge Graph.
    
    Funcionalidades:
    - Multiple layout algorithms (force-directed, hierarchical, etc.)
    - Dynamic filtering por domínio, tipo, centralidade
    - Interactive features (zoom, pan, drag, hover, click)
    - Color encoding baseado em atributos
    - Node/edge sizing baseado em métricas
    - Export para múltiplos formatos (SVG, PNG, JSON, HTML)
    - Cluster-based rendering para performance
    - Real-time updates e animações
    - Integration com D3.js, Plotly, Cytoscape.js
    - Responsive design para múltiplas telas
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Core data structures
        self.graph: Optional[nx.Graph] = None
        self.directed_graph: Optional[nx.DiGraph] = None
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Layout positions cache
        self.layout_positions: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # Filtering state
        self.filtered_nodes: Set[str] = set()
        self.filtered_edges: Set[Tuple[str, str]] = set()
        
        # Visualization data cache
        self.visualization_cache: Dict[str, Dict[str, Any]] = {}
        
        # Interactive state
        self.selected_nodes: Set[str] = set()
        self.highlighted_paths: List[List[str]] = []
        
        # Clustering for performance
        self.node_clusters: Dict[str, int] = {}
        self.cluster_representatives: Dict[int, str] = {}
        
        logger.info("🎨 DARWIN Graph Visualization initialized")
    
    # ==================== CORE SETUP ====================
    
    async def initialize_from_snapshot(
        self,
        snapshot: KnowledgeGraphSnapshot,
        config: Optional[VisualizationConfig] = None
    ) -> None:
        """Inicializa visualização a partir de snapshot do Knowledge Graph."""
        if config:
            self.config = config
            
        logger.info(f"🎨 Initializing visualization from snapshot {snapshot.id}")
        
        try:
            # Create graphs
            self.graph = nx.Graph()
            self.directed_graph = nx.DiGraph()
            
            # Add nodes with attributes
            for node in snapshot.nodes:
                node_attrs = {
                    "id": node.id,
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
                    "id": edge.id,
                    "type": edge.type.value,
                    "weight": edge.weight,
                    **edge.properties
                }
                
                edge_key = (edge.source, edge.target)
                self.edge_attributes[edge_key] = edge_attrs
                
                self.graph.add_edge(edge.source, edge.target, **edge_attrs)
                self.directed_graph.add_edge(edge.source, edge.target, **edge_attrs)
            
            # Apply filtering
            await self._apply_filters()
            
            # Perform clustering if enabled
            if self.config.enable_clustering:
                await self._perform_node_clustering()
            
            logger.info(f"✅ Visualization initialized: {len(self.filtered_nodes)} nodes, {len(self.filtered_edges)} edges")
            
        except Exception as e:
            logger.error(f"❌ Visualization initialization failed: {e}")
            raise
    
    async def _apply_filters(self):
        """Aplica filtros de visualização."""
        # Start with all nodes and edges
        self.filtered_nodes = set(self.graph.nodes())
        self.filtered_edges = set(self.graph.edges())
        
        # Filter by domains
        if self.config.filter_domains:
            domain_strings = [d.value for d in self.config.filter_domains]
            self.filtered_nodes = {
                node for node in self.filtered_nodes
                if self.node_attributes.get(node, {}).get("domain") in domain_strings
            }
        
        # Filter by node types
        if self.config.filter_node_types:
            type_strings = [t.value for t in self.config.filter_node_types]
            self.filtered_nodes = {
                node for node in self.filtered_nodes
                if self.node_attributes.get(node, {}).get("type") in type_strings
            }
        
        # Filter by minimum degree
        if self.config.min_node_degree > 0:
            degrees = dict(self.graph.degree())
            self.filtered_nodes = {
                node for node in self.filtered_nodes
                if degrees.get(node, 0) >= self.config.min_node_degree
            }
        
        # Filter edges to only include those between filtered nodes
        self.filtered_edges = {
            edge for edge in self.filtered_edges
            if edge[0] in self.filtered_nodes and edge[1] in self.filtered_nodes
        }
        
        # Limit size for performance
        if len(self.filtered_nodes) > self.config.max_nodes:
            # Keep nodes with highest degree/centrality
            degrees = dict(self.graph.degree())
            sorted_nodes = sorted(
                self.filtered_nodes,
                key=lambda x: degrees.get(x, 0),
                reverse=True
            )
            self.filtered_nodes = set(sorted_nodes[:self.config.max_nodes])
            
            # Update edges
            self.filtered_edges = {
                edge for edge in self.filtered_edges
                if edge[0] in self.filtered_nodes and edge[1] in self.filtered_nodes
            }
        
        if len(self.filtered_edges) > self.config.max_edges:
            # Keep edges with highest weight
            sorted_edges = sorted(
                self.filtered_edges,
                key=lambda x: self.edge_attributes.get(x, {}).get("weight", 0),
                reverse=True
            )
            self.filtered_edges = set(sorted_edges[:self.config.max_edges])
        
        logger.info(f"🔍 Filters applied: {len(self.filtered_nodes)} nodes, {len(self.filtered_edges)} edges")
    
    # ==================== LAYOUT ALGORITHMS ====================
    
    async def calculate_layout(
        self,
        layout_type: Optional[LayoutType] = None,
        layout_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Calcula posições dos nós usando algoritmo de layout especificado."""
        
        layout = layout_type or self.config.layout_type
        params = {**self.config.layout_params, **(layout_params or {})}
        
        logger.info(f"📐 Calculating layout: {layout.value}")
        
        # Create subgraph with filtered nodes/edges
        subgraph = self.graph.subgraph(self.filtered_nodes).copy()
        
        # Remove isolated nodes for layout calculation
        connected_nodes = set()
        for edge in self.filtered_edges:
            connected_nodes.add(edge[0])
            connected_nodes.add(edge[1])
        
        if not connected_nodes:
            # Create simple grid layout for isolated nodes
            return await self._grid_layout(list(self.filtered_nodes))
        
        # Calculate layout
        try:
            if layout == LayoutType.FORCE_DIRECTED:
                positions = await self._force_directed_layout(subgraph, params)
                
            elif layout == LayoutType.SPRING:
                positions = await self._spring_layout(subgraph, params)
                
            elif layout == LayoutType.HIERARCHICAL:
                positions = await self._hierarchical_layout(subgraph, params)
                
            elif layout == LayoutType.CIRCULAR:
                positions = await self._circular_layout(subgraph, params)
                
            elif layout == LayoutType.KAMADA_KAWAI:
                positions = await self._kamada_kawai_layout(subgraph, params)
                
            elif layout == LayoutType.SPECTRAL:
                positions = await self._spectral_layout(subgraph, params)
                
            elif layout == LayoutType.SHELL:
                positions = await self._shell_layout(subgraph, params)
                
            elif layout == LayoutType.SPIRAL:
                positions = await self._spiral_layout(subgraph, params)
                
            elif layout == LayoutType.RADIAL:
                positions = await self._radial_layout(subgraph, params)
                
            elif layout == LayoutType.CLUSTER_BASED:
                positions = await self._cluster_based_layout(subgraph, params)
                
            elif layout == LayoutType.DOMAIN_SEPARATED:
                positions = await self._domain_separated_layout(subgraph, params)
                
            elif layout == LayoutType.TEMPORAL:
                positions = await self._temporal_layout(subgraph, params)
                
            else:
                logger.warning(f"Unknown layout type: {layout}, using spring layout")
                positions = await self._spring_layout(subgraph, params)
            
            # Cache positions
            self.layout_positions[layout.value] = positions
            
            logger.info(f"✅ Layout calculated: {len(positions)} positions")
            return positions
            
        except Exception as e:
            logger.error(f"Layout calculation failed: {e}")
            # Fallback to simple grid layout
            return await self._grid_layout(list(self.filtered_nodes))
    
    async def _force_directed_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout força-direcionado otimizado para D3.js."""
        
        # Use NetworkX spring layout as base
        k = params.get("k", 1/math.sqrt(graph.number_of_nodes())) if graph.number_of_nodes() > 0 else 1
        iterations = params.get("iterations", 50)
        
        positions = nx.spring_layout(
            graph,
            k=k,
            iterations=iterations,
            weight="weight",
            scale=500  # Scale for web visualization
        )
        
        return positions
    
    async def _spring_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Spring layout com parâmetros customizáveis."""
        
        positions = nx.spring_layout(
            graph,
            k=params.get("k", None),
            pos=params.get("pos", None),
            fixed=params.get("fixed", None),
            iterations=params.get("iterations", 50),
            threshold=params.get("threshold", 1e-4),
            weight="weight",
            scale=params.get("scale", 500),
            center=params.get("center", None),
            dim=2,
            seed=params.get("seed", 42)
        )
        
        return positions
    
    async def _hierarchical_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout hierárquico baseado em tree/DAG."""
        
        try:
            # Try to use hierarchy based on graph structure
            if nx.is_directed_acyclic_graph(self.directed_graph.subgraph(graph.nodes())):
                # Use graphviz layout if available
                try:
                    positions = nx.nx_agraph.graphviz_layout(
                        graph, 
                        prog=params.get("prog", "dot")
                    )
                    return positions
                except:
                    pass
            
            # Fallback: create artificial hierarchy based on centrality
            centrality = nx.degree_centrality(graph)
            
            # Sort nodes by centrality
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Create hierarchical positions
            positions = {}
            levels = params.get("levels", 5)
            width = params.get("width", 800)
            height = params.get("height", 600)
            
            nodes_per_level = len(sorted_nodes) // levels + 1
            
            for i, (node, _) in enumerate(sorted_nodes):
                level = i // nodes_per_level
                pos_in_level = i % nodes_per_level
                total_in_level = min(nodes_per_level, len(sorted_nodes) - level * nodes_per_level)
                
                x = (pos_in_level - total_in_level/2) * (width / max(total_in_level, 1))
                y = level * (height / levels)
                
                positions[node] = (x, y)
            
            return positions
            
        except Exception as e:
            logger.warning(f"Hierarchical layout failed: {e}")
            return await self._spring_layout(graph, {})
    
    async def _circular_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout circular."""
        
        positions = nx.circular_layout(
            graph,
            scale=params.get("scale", 300),
            center=params.get("center", None),
            dim=2
        )
        
        return positions
    
    async def _kamada_kawai_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Kamada-Kawai layout para visualização estética."""
        
        try:
            positions = nx.kamada_kawai_layout(
                graph,
                dist=params.get("dist", None),
                pos=params.get("pos", None),
                weight="weight",
                scale=params.get("scale", 500),
                center=params.get("center", None),
                dim=2
            )
            return positions
        except:
            # Fallback to spring layout
            return await self._spring_layout(graph, params)
    
    async def _spectral_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Spectral layout baseado em eigenvetores."""
        
        try:
            positions = nx.spectral_layout(
                graph,
                weight="weight",
                scale=params.get("scale", 500),
                center=params.get("center", None),
                dim=2
            )
            return positions
        except:
            return await self._spring_layout(graph, params)
    
    async def _shell_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Shell layout com múltiplas camadas."""
        
        # Create shells based on domains or node types
        shells = []
        nodes_by_attribute = defaultdict(list)
        
        attribute = params.get("shell_attribute", "domain")
        
        for node in graph.nodes():
            attr_value = self.node_attributes.get(node, {}).get(attribute, "unknown")
            nodes_by_attribute[attr_value].append(node)
        
        shells = list(nodes_by_attribute.values())
        
        try:
            positions = nx.shell_layout(
                graph,
                nlist=shells if len(shells) > 1 else None,
                scale=params.get("scale", 500),
                center=params.get("center", None),
                dim=2
            )
            return positions
        except:
            return await self._circular_layout(graph, params)
    
    async def _spiral_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout em espiral baseado em ordem customizada."""
        
        # Sort nodes by some attribute (centrality, domain, etc.)
        sort_attribute = params.get("sort_by", "degree")
        
        if sort_attribute == "degree":
            node_values = dict(graph.degree())
        elif sort_attribute == "domain":
            node_values = {
                node: self.node_attributes.get(node, {}).get("domain", "z")
                for node in graph.nodes()
            }
        else:
            node_values = {node: 0 for node in graph.nodes()}
        
        sorted_nodes = sorted(node_values.items(), key=lambda x: x[1], reverse=True)
        
        # Create spiral positions
        positions = {}
        scale = params.get("scale", 300)
        
        for i, (node, _) in enumerate(sorted_nodes):
            angle = i * 0.5  # Spiral parameter
            radius = scale * math.sqrt(i / len(sorted_nodes))
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            positions[node] = (x, y)
        
        return positions
    
    async def _radial_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout radial com nó central."""
        
        # Find center node (highest centrality)
        centrality = nx.degree_centrality(graph)
        center_node = max(centrality.items(), key=lambda x: x[1])[0]
        
        # Calculate distances from center
        distances = nx.single_source_shortest_path_length(graph, center_node)
        
        positions = {}
        scale = params.get("scale", 200)
        
        # Group nodes by distance
        nodes_by_distance = defaultdict(list)
        for node, distance in distances.items():
            nodes_by_distance[distance].append(node)
        
        # Position center
        positions[center_node] = (0, 0)
        
        # Position other nodes in concentric circles
        for distance, nodes in nodes_by_distance.items():
            if distance == 0:
                continue
                
            radius = distance * scale
            angle_step = 2 * math.pi / len(nodes) if len(nodes) > 0 else 0
            
            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[node] = (x, y)
        
        return positions
    
    async def _cluster_based_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout baseado em clusters detectados."""
        
        # Use cached clusters if available
        if not self.node_clusters:
            await self._perform_node_clustering()
        
        positions = {}
        scale = params.get("scale", 400)
        cluster_separation = params.get("cluster_separation", 200)
        
        # Group nodes by cluster
        nodes_by_cluster = defaultdict(list)
        for node in graph.nodes():
            cluster = self.node_clusters.get(node, 0)
            nodes_by_cluster[cluster].append(node)
        
        # Position each cluster
        num_clusters = len(nodes_by_cluster)
        
        for i, (cluster_id, cluster_nodes) in enumerate(nodes_by_cluster.items()):
            # Cluster center position
            cluster_angle = i * 2 * math.pi / num_clusters
            cluster_center_x = cluster_separation * math.cos(cluster_angle)
            cluster_center_y = cluster_separation * math.sin(cluster_angle)
            
            # Create subgraph for cluster
            cluster_subgraph = graph.subgraph(cluster_nodes)
            
            # Layout nodes within cluster
            if len(cluster_nodes) > 1:
                cluster_positions = nx.spring_layout(
                    cluster_subgraph,
                    scale=scale / (2 * max(1, math.sqrt(num_clusters))),
                    iterations=30
                )
                
                # Offset by cluster center
                for node, (x, y) in cluster_positions.items():
                    positions[node] = (x + cluster_center_x, y + cluster_center_y)
            else:
                positions[cluster_nodes[0]] = (cluster_center_x, cluster_center_y)
        
        return positions
    
    async def _domain_separated_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout com separação por domínios científicos."""
        
        positions = {}
        
        # Group nodes by domain
        nodes_by_domain = defaultdict(list)
        for node in graph.nodes():
            domain = self.node_attributes.get(node, {}).get("domain", "unknown")
            nodes_by_domain[domain].append(node)
        
        # Assign positions to domain centers
        domain_centers = {}
        domains = list(nodes_by_domain.keys())
        
        if len(domains) == 1:
            domain_centers[domains[0]] = (0, 0)
        else:
            for i, domain in enumerate(domains):
                angle = i * 2 * math.pi / len(domains)
                radius = params.get("domain_separation", 300)
                domain_centers[domain] = (
                    radius * math.cos(angle),
                    radius * math.sin(angle)
                )
        
        # Layout nodes within each domain
        for domain, nodes in nodes_by_domain.items():
            center_x, center_y = domain_centers[domain]
            
            if len(nodes) == 1:
                positions[nodes[0]] = (center_x, center_y)
            else:
                # Create domain subgraph
                domain_subgraph = graph.subgraph(nodes)
                
                # Use spring layout for domain
                domain_positions = nx.spring_layout(
                    domain_subgraph,
                    scale=params.get("domain_scale", 150),
                    iterations=30
                )
                
                # Offset by domain center
                for node, (x, y) in domain_positions.items():
                    positions[node] = (x + center_x, y + center_y)
        
        return positions
    
    async def _temporal_layout(
        self, 
        graph: nx.Graph, 
        params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Layout baseado em informação temporal."""
        
        positions = {}
        
        # Extract temporal information
        node_times = {}
        for node in graph.nodes():
            # Try to get timestamp from node attributes
            attrs = self.node_attributes.get(node, {})
            
            # Look for various timestamp fields
            timestamp = attrs.get("timestamp") or attrs.get("created_at") or attrs.get("publication_date")
            
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        timestamp = None
                
                if timestamp:
                    node_times[node] = timestamp
        
        if not node_times:
            # Fallback to spring layout
            return await self._spring_layout(graph, params)
        
        # Sort nodes by time
        sorted_nodes = sorted(node_times.items(), key=lambda x: x[1])
        
        # Create timeline layout
        width = params.get("width", 800)
        height = params.get("height", 600)
        
        # Group nodes by time periods
        time_groups = defaultdict(list)
        
        if sorted_nodes:
            min_time = sorted_nodes[0][1]
            max_time = sorted_nodes[-1][1]
            time_span = (max_time - min_time).total_seconds()
            
            # Create time buckets
            num_buckets = params.get("time_buckets", 10)
            bucket_size = time_span / num_buckets if time_span > 0 else 1
            
            for node, timestamp in sorted_nodes:
                bucket = int((timestamp - min_time).total_seconds() / bucket_size) if bucket_size > 0 else 0
                bucket = min(bucket, num_buckets - 1)  # Clamp to valid range
                time_groups[bucket].append(node)
        
        # Position nodes
        for bucket, nodes in time_groups.items():
            x_position = (bucket / max(len(time_groups) - 1, 1)) * width - width/2
            
            # Arrange nodes vertically within time bucket
            for i, node in enumerate(nodes):
                y_offset = (i - len(nodes)/2) * (height / max(len(nodes), 1))
                positions[node] = (x_position, y_offset)
        
        return positions
    
    async def _grid_layout(self, nodes: List[str]) -> Dict[str, Tuple[float, float]]:
        """Simple grid layout for isolated nodes."""
        positions = {}
        
        if not nodes:
            return positions
        
        # Calculate grid dimensions
        n_nodes = len(nodes)
        cols = math.ceil(math.sqrt(n_nodes))
        rows = math.ceil(n_nodes / cols)
        
        spacing = 100
        
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            
            x = (col - cols/2) * spacing
            y = (row - rows/2) * spacing
            
            positions[node] = (x, y)
        
        return positions
    
    # ==================== NODE CLUSTERING ====================
    
    async def _perform_node_clustering(self):
        """Realiza clustering dos nós para otimização de performance."""
        
        if len(self.filtered_nodes) < self.config.cluster_threshold:
            # No clustering needed
            return
        
        logger.info("🔗 Performing node clustering for performance optimization")
        
        try:
            # Extract features for clustering
            features = await self._extract_clustering_features()
            
            if features is None:
                return
            
            # Determine number of clusters
            n_clusters = min(
                max(len(self.filtered_nodes) // 50, 2),
                20  # Max 20 clusters
            )
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Assign clusters to nodes
            nodes_list = list(self.filtered_nodes)
            for i, node in enumerate(nodes_list):
                self.node_clusters[node] = int(cluster_labels[i])
            
            # Select representative nodes for each cluster
            cluster_centers = kmeans.cluster_centers_
            for cluster_id in range(n_clusters):
                cluster_nodes = [
                    node for i, node in enumerate(nodes_list) 
                    if cluster_labels[i] == cluster_id
                ]
                
                if cluster_nodes:
                    # Find node closest to cluster center
                    cluster_features = features[[
                        i for i, node in enumerate(nodes_list) 
                        if cluster_labels[i] == cluster_id
                    ]]
                    
                    distances = np.linalg.norm(
                        cluster_features - cluster_centers[cluster_id], 
                        axis=1
                    )
                    
                    representative_idx = np.argmin(distances)
                    representative_node = [
                        node for i, node in enumerate(nodes_list) 
                        if cluster_labels[i] == cluster_id
                    ][representative_idx]
                    
                    self.cluster_representatives[cluster_id] = representative_node
            
            logger.info(f"✅ Node clustering completed: {n_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Node clustering failed: {e}")
    
    async def _extract_clustering_features(self) -> Optional[np.ndarray]:
        """Extrai features para clustering dos nós."""
        
        try:
            nodes = list(self.filtered_nodes)
            features = []
            
            for node in nodes:
                node_features = []
                
                # Structural features
                degree = self.graph.degree(node)
                node_features.append(degree)
                
                # Centrality features (if available from previous calculations)
                node_features.append(self.graph.degree(node, weight="weight") if "weight" in self.graph[node] else degree)
                
                # Domain encoding
                domain = self.node_attributes.get(node, {}).get("domain", "unknown")
                domain_vector = [0.0] * len(ScientificDomains)
                
                for i, d in enumerate(ScientificDomains):
                    if d.value == domain:
                        domain_vector[i] = 1.0
                        break
                
                node_features.extend(domain_vector)
                
                # Type encoding
                node_type = self.node_attributes.get(node, {}).get("type", "unknown")
                type_vector = [0.0] * len(NodeTypes)
                
                for i, t in enumerate(NodeTypes):
                    if t.value == node_type:
                        type_vector[i] = 1.0
                        break
                
                node_features.extend(type_vector)
                
                features.append(node_features)
            
            if features:
                # Normalize features
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(features)
                return normalized_features
            
            return None
            
        except Exception as e:
            logger.error(f"Feature extraction for clustering failed: {e}")
            return None
    
    # ==================== VISUALIZATION DATA GENERATION ====================
    
    async def generate_visualization_data(
        self,
        layout_type: Optional[LayoutType] = None
    ) -> GraphVisualizationData:
        """Gera dados completos para visualização web."""
        
        logger.info("🎨 Generating visualization data")
        
        try:
            # Calculate layout positions
            positions = await self.calculate_layout(layout_type)
            
            # Generate node data
            nodes_data = await self._generate_nodes_data(positions)
            
            # Generate edge data
            edges_data = await self._generate_edges_data()
            
            # Calculate statistics
            stats = await self._calculate_visualization_statistics()
            
            # Create visualization data
            viz_data = GraphVisualizationData(
                nodes=nodes_data,
                edges=edges_data,
                config=self.config,
                stats=stats
            )
            
            # Cache the result
            cache_key = f"{layout_type.value if layout_type else 'default'}_{hash(str(self.config))}"
            self.visualization_cache[cache_key] = viz_data.dict()
            
            logger.info(f"✅ Visualization data generated: {len(nodes_data)} nodes, {len(edges_data)} edges")
            return viz_data
            
        except Exception as e:
            logger.error(f"❌ Visualization data generation failed: {e}")
            raise
    
    async def _generate_nodes_data(
        self, 
        positions: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """Gera dados dos nós para visualização."""
        
        nodes_data = []
        
        # Calculate node sizes
        node_sizes = await self._calculate_node_sizes()
        
        # Calculate node colors
        node_colors = await self._calculate_node_colors()
        
        for node in self.filtered_nodes:
            if node not in positions:
                continue
                
            x, y = positions[node]
            attrs = self.node_attributes.get(node, {})
            
            node_data = {
                "id": node,
                "x": float(x),
                "y": float(y),
                "size": node_sizes.get(node, 5),
                "color": node_colors.get(node, "#999999"),
                "label": attrs.get("label", node),
                "type": attrs.get("type", "unknown"),
                "domain": attrs.get("domain", "unknown"),
                "degree": self.graph.degree(node),
                "cluster": self.node_clusters.get(node, 0),
                
                # Additional attributes
                **{k: v for k, v in attrs.items() if k not in ["id", "label", "type", "domain"]}
            }
            
            # Add interactivity properties
            if self.config.enable_hover:
                node_data["tooltip"] = await self._generate_node_tooltip(node, attrs)
            
            if self.config.enable_click:
                node_data["clickable"] = True
            
            nodes_data.append(node_data)
        
        return nodes_data
    
    async def _generate_edges_data(self) -> List[Dict[str, Any]]:
        """Gera dados das arestas para visualização."""
        
        edges_data = []
        
        # Calculate edge widths
        edge_widths = await self._calculate_edge_widths()
        
        # Calculate edge colors
        edge_colors = await self._calculate_edge_colors()
        
        for edge in self.filtered_edges:
            source, target = edge
            
            if source not in self.filtered_nodes or target not in self.filtered_nodes:
                continue
                
            attrs = self.edge_attributes.get(edge, {})
            
            edge_data = {
                "id": f"{source}-{target}",
                "source": source,
                "target": target,
                "width": edge_widths.get(edge, 1),
                "color": edge_colors.get(edge, "#cccccc"),
                "weight": attrs.get("weight", 1),
                "type": attrs.get("type", "unknown"),
                
                # Additional attributes
                **{k: v for k, v in attrs.items() if k not in ["id", "weight", "type"]}
            }
            
            # Add label if enabled
            if self.config.show_edge_labels:
                edge_data["label"] = attrs.get("label", f"{source}-{target}")
            
            edges_data.append(edge_data)
        
        return edges_data
    
    async def _calculate_node_sizes(self) -> Dict[str, float]:
        """Calcula tamanhos dos nós baseado no atributo configurado."""
        
        node_sizes = {}
        
        size_attribute = self.config.node_size_attribute
        
        if size_attribute == "degree":
            # Size by degree
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            
            for node in self.filtered_nodes:
                degree = degrees.get(node, 0)
                # Scale between 3 and 20
                size = 3 + (degree / max_degree) * 17
                node_sizes[node] = size
                
        elif size_attribute == "centrality":
            # Size by centrality (would need pre-calculated values)
            # For now, use degree as fallback
            return await self._calculate_node_sizes()  # Recursive call with degree
            
        else:
            # Fixed size
            for node in self.filtered_nodes:
                node_sizes[node] = 8  # Default size
        
        return node_sizes
    
    async def _calculate_node_colors(self) -> Dict[str, str]:
        """Calcula cores dos nós baseado no esquema configurado."""
        
        node_colors = {}
        
        color_scheme = self.config.color_scheme
        
        if color_scheme == ColorScheme.DOMAIN:
            # Color by domain
            for node in self.filtered_nodes:
                domain = self.node_attributes.get(node, {}).get("domain", "unknown")
                
                # Try to match to ScientificDomains enum
                color = "#999999"  # Default
                for d in ScientificDomains:
                    if d.value == domain:
                        color = DOMAIN_COLORS.get(d, "#999999")
                        break
                
                node_colors[node] = color
                
        elif color_scheme == ColorScheme.NODE_TYPE:
            # Color by node type
            for node in self.filtered_nodes:
                node_type = self.node_attributes.get(node, {}).get("type", "unknown")
                
                # Try to match to NodeTypes enum
                color = "#999999"  # Default
                for t in NodeTypes:
                    if t.value == node_type:
                        color = NODE_TYPE_COLORS.get(t, "#999999")
                        break
                
                node_colors[node] = color
                
        elif color_scheme == ColorScheme.COMMUNITY:
            # Color by cluster/community
            cluster_colors = {}
            unique_clusters = set(self.node_clusters.values())
            
            # Generate colors for clusters
            for i, cluster in enumerate(unique_clusters):
                hue = (i * 137.5) % 360  # Golden angle for good color distribution
                cluster_colors[cluster] = f"hsl({hue}, 70%, 60%)"
            
            for node in self.filtered_nodes:
                cluster = self.node_clusters.get(node, 0)
                node_colors[node] = cluster_colors.get(cluster, "#999999")
                
        else:
            # Default gray
            for node in self.filtered_nodes:
                node_colors[node] = "#999999"
        
        return node_colors
    
    async def _calculate_edge_widths(self) -> Dict[Tuple[str, str], float]:
        """Calcula larguras das arestas baseado no atributo configurado."""
        
        edge_widths = {}
        
        width_attribute = self.config.edge_width_attribute
        
        if width_attribute == "weight":
            # Width by weight
            weights = {edge: attrs.get("weight", 1) for edge, attrs in self.edge_attributes.items()}
            max_weight = max(weights.values()) if weights else 1
            
            for edge in self.filtered_edges:
                weight = weights.get(edge, 1)
                # Scale between 1 and 8
                width = 1 + (weight / max_weight) * 7
                edge_widths[edge] = width
        else:
            # Fixed width
            for edge in self.filtered_edges:
                edge_widths[edge] = 2  # Default width
        
        return edge_widths
    
    async def _calculate_edge_colors(self) -> Dict[Tuple[str, str], str]:
        """Calcula cores das arestas."""
        
        edge_colors = {}
        
        for edge in self.filtered_edges:
            # Color by edge type or default
            edge_type = self.edge_attributes.get(edge, {}).get("type", "unknown")
            
            # Simple color mapping for edge types
            type_colors = {
                "cites": "#ff7f0e",
                "similar_to": "#2ca02c", 
                "bridges_domain": "#d62728",
                "collaborates": "#9467bd",
                "uses_method": "#8c564b"
            }
            
            edge_colors[edge] = type_colors.get(edge_type, "#cccccc")
        
        return edge_colors
    
    async def _generate_node_tooltip(self, node: str, attrs: Dict[str, Any]) -> str:
        """Gera tooltip para um nó."""
        
        tooltip_lines = [
            f"<strong>{attrs.get('label', node)}</strong>",
            f"Type: {attrs.get('type', 'Unknown')}",
            f"Domain: {attrs.get('domain', 'Unknown')}",
            f"Degree: {self.graph.degree(node)}"
        ]
        
        # Add domain-specific information
        if "description" in attrs:
            tooltip_lines.append(f"Description: {attrs['description'][:100]}...")
        
        if "confidence" in attrs:
            tooltip_lines.append(f"Confidence: {attrs['confidence']:.2f}")
        
        return "<br>".join(tooltip_lines)
    
    async def _calculate_visualization_statistics(self) -> Dict[str, Any]:
        """Calcula estatísticas da visualização."""
        
        # Domain distribution
        domain_counts = Counter()
        for node in self.filtered_nodes:
            domain = self.node_attributes.get(node, {}).get("domain", "unknown")
            domain_counts[domain] += 1
        
        # Type distribution
        type_counts = Counter()
        for node in self.filtered_nodes:
            node_type = self.node_attributes.get(node, {}).get("type", "unknown")
            type_counts[node_type] += 1
        
        # Edge type distribution
        edge_type_counts = Counter()
        for edge in self.filtered_edges:
            edge_type = self.edge_attributes.get(edge, {}).get("type", "unknown")
            edge_type_counts[edge_type] += 1
        
        # Degree statistics
        degrees = [self.graph.degree(node) for node in self.filtered_nodes]
        
        return {
            "total_nodes": len(self.filtered_nodes),
            "total_edges": len(self.filtered_edges),
            "domain_distribution": dict(domain_counts),
            "type_distribution": dict(type_counts),
            "edge_type_distribution": dict(edge_type_counts),
            "degree_stats": {
                "min": min(degrees) if degrees else 0,
                "max": max(degrees) if degrees else 0,
                "mean": sum(degrees) / len(degrees) if degrees else 0,
                "median": sorted(degrees)[len(degrees)//2] if degrees else 0
            },
            "clusters": len(set(self.node_clusters.values())),
            "layout_type": self.config.layout_type.value,
            "color_scheme": self.config.color_scheme.value
        }
    
    # ==================== EXPORT FUNCTIONS ====================
    
    async def export_visualization(
        self,
        format_type: str,
        layout_type: Optional[LayoutType] = None,
        **kwargs
    ) -> Union[str, bytes, Dict[str, Any]]:
        """Exporta visualização em formato especificado."""
        
        logger.info(f"📤 Exporting visualization in {format_type} format")
        
        # Generate visualization data
        viz_data = await self.generate_visualization_data(layout_type)
        
        if format_type.lower() == "json":
            return viz_data.dict()
            
        elif format_type.lower() == "d3":
            return await self._export_d3_format(viz_data, **kwargs)
            
        elif format_type.lower() == "cytoscape":
            return await self._export_cytoscape_format(viz_data, **kwargs)
            
        elif format_type.lower() == "graphviz":
            return await self._export_graphviz_format(viz_data, **kwargs)
            
        elif format_type.lower() == "svg":
            return await self._export_svg_format(viz_data, **kwargs)
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    async def _export_d3_format(
        self, 
        viz_data: GraphVisualizationData, 
        **kwargs
    ) -> Dict[str, Any]:
        """Exporta em formato otimizado para D3.js."""
        
        return {
            "nodes": viz_data.nodes,
            "links": [
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "value": edge.get("weight", 1),
                    "width": edge.get("width", 1),
                    "color": edge.get("color", "#cccccc"),
                    **{k: v for k, v in edge.items() if k not in ["source", "target", "id"]}
                }
                for edge in viz_data.edges
            ],
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "force_params": FORCE_SIMULATION_DEFAULTS,
                **kwargs
            },
            "metadata": viz_data.stats
        }
    
    async def _export_cytoscape_format(
        self, 
        viz_data: GraphVisualizationData, 
        **kwargs
    ) -> Dict[str, Any]:
        """Exporta em formato Cytoscape.js."""
        
        elements = []
        
        # Add nodes
        for node in viz_data.nodes:
            elements.append({
                "data": {
                    "id": node["id"],
                    "label": node.get("label", node["id"]),
                    "type": node.get("type", "unknown"),
                    "domain": node.get("domain", "unknown"),
                    **{k: v for k, v in node.items() if k not in ["id", "x", "y", "size", "color"]}
                },
                "position": {
                    "x": node["x"],
                    "y": node["y"]
                },
                "style": {
                    "background-color": node.get("color", "#999999"),
                    "width": node.get("size", 8) * 2,
                    "height": node.get("size", 8) * 2
                }
            })
        
        # Add edges
        for edge in viz_data.edges:
            elements.append({
                "data": {
                    "id": edge["id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "weight": edge.get("weight", 1),
                    **{k: v for k, v in edge.items() if k not in ["id", "source", "target", "width", "color"]}
                },
                "style": {
                    "line-color": edge.get("color", "#cccccc"),
                    "width": edge.get("width", 2)
                }
            })
        
        return {
            "elements": elements,
            "style": kwargs.get("style", []),
            "layout": kwargs.get("layout", {"name": "preset"}),
            "config": {
                "container_width": self.config.width,
                "container_height": self.config.height
            }
        }
    
    async def _export_graphviz_format(
        self, 
        viz_data: GraphVisualizationData, 
        **kwargs
    ) -> str:
        """Exporta em formato DOT (Graphviz)."""
        
        lines = ["digraph KnowledgeGraph {"]
        lines.append('  rankdir="TB";')
        lines.append('  node [shape=circle];')
        
        # Add nodes
        for node in viz_data.nodes:
            attributes = [
                f'label="{node.get("label", node["id"])}"',
                f'color="{node.get("color", "#999999")}"',
                f'width={node.get("size", 8) / 10}'
            ]
            lines.append(f'  "{node["id"]}" [{", ".join(attributes)}];')
        
        # Add edges
        for edge in viz_data.edges:
            attributes = [
                f'color="{edge.get("color", "#cccccc")}"',
                f'penwidth={edge.get("width", 2)}'
            ]
            lines.append(f'  "{edge["source"]}" -> "{edge["target"]}" [{", ".join(attributes)}];')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    async def _export_svg_format(
        self, 
        viz_data: GraphVisualizationData, 
        **kwargs
    ) -> str:
        """Exporta em formato SVG."""
        
        width = self.config.width
        height = self.config.height
        
        # Calculate bounding box of positions
        if viz_data.nodes:
            min_x = min(node["x"] for node in viz_data.nodes)
            max_x = max(node["x"] for node in viz_data.nodes)
            min_y = min(node["y"] for node in viz_data.nodes)
            max_y = max(node["y"] for node in viz_data.nodes)
            
            # Add padding
            padding = 50
            viewbox_x = min_x - padding
            viewbox_y = min_y - padding
            viewbox_width = max_x - min_x + 2 * padding
            viewbox_height = max_y - min_y + 2 * padding
        else:
            viewbox_x, viewbox_y = 0, 0
            viewbox_width, viewbox_height = width, height
        
        lines = [
            f'<svg width="{width}" height="{height}" viewBox="{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}" xmlns="http://www.w3.org/2000/svg">',
            '  <defs>',
            '    <style>',
            '      .node-label { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }',
            '      .edge { stroke-opacity: 0.6; }',
            '      .node { stroke: #000; stroke-width: 1px; }',
            '    </style>',
            '  </defs>'
        ]
        
        # Add edges first (so they appear behind nodes)
        for edge in viz_data.edges:
            source_node = next(n for n in viz_data.nodes if n["id"] == edge["source"])
            target_node = next(n for n in viz_data.nodes if n["id"] == edge["target"])
            
            lines.append(
                f'  <line x1="{source_node["x"]}" y1="{source_node["y"]}" '
                f'x2="{target_node["x"]}" y2="{target_node["y"]}" '
                f'stroke="{edge.get("color", "#cccccc")}" '
                f'stroke-width="{edge.get("width", 2)}" class="edge" />'
            )
        
        # Add nodes
        for node in viz_data.nodes:
            # Node circle
            lines.append(
                f'  <circle cx="{node["x"]}" cy="{node["y"]}" '
                f'r="{node.get("size", 8)}" '
                f'fill="{node.get("color", "#999999")}" class="node" />'
            )
            
            # Node label (if enabled)
            if self.config.show_node_labels:
                lines.append(
                    f'  <text x="{node["x"]}" y="{node["y"] + node.get("size", 8) + 15}" '
                    f'class="node-label">{node.get("label", node["id"])}</text>'
                )
        
        lines.append('</svg>')
        
        return "\n".join(lines)
    
    # ==================== UTILITY METHODS ====================
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o estado da visualização."""
        
        return {
            "config": {
                "layout_type": self.config.layout_type.value,
                "color_scheme": self.config.color_scheme.value,
                "node_size_attribute": self.config.node_size_attribute,
                "edge_width_attribute": self.config.edge_width_attribute,
                "max_nodes": self.config.max_nodes,
                "max_edges": self.config.max_edges
            },
            "data_status": {
                "total_nodes": len(self.node_attributes),
                "total_edges": len(self.edge_attributes),
                "filtered_nodes": len(self.filtered_nodes),
                "filtered_edges": len(self.filtered_edges),
                "clusters": len(set(self.node_clusters.values())) if self.node_clusters else 0
            },
            "layout_cache": list(self.layout_positions.keys()),
            "visualization_cache": list(self.visualization_cache.keys()),
            "performance": {
                "clustering_enabled": self.config.enable_clustering,
                "webgl_enabled": self.config.use_webgl,
                "cluster_threshold": self.config.cluster_threshold
            }
        }


__all__ = [
    "DARWINGraphVisualization",
    "VisualizationConfig", 
    "LayoutType",
    "ColorScheme",
    "InteractionMode",
    "DOMAIN_COLORS",
    "NODE_TYPE_COLORS",
    "FORCE_SIMULATION_DEFAULTS"
]