"""
KEC Biomat Processing Pipeline
=============================

Pipeline completo para processamento de imagens de biomateriais e cálculo de métricas KEC 2.0.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import networkx as nx
import logging
import time
from pathlib import Path

from ..configs import KECConfig, load_config
from ..metrics.kec_metrics import compute_kec_metrics

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuração específica para processamento."""
    input_path: str
    output_path: str
    config_path: Optional[str] = None
    save_intermediates: bool = True
    parallel_processing: bool = False
    verbose: bool = True

@dataclass 
class ProcessingResults:
    """Resultados do pipeline de processamento."""
    kec_metrics: Dict[str, float]
    graph: nx.Graph
    image_metadata: Dict[str, Any]
    processing_time: float
    config_used: KECConfig
    intermediates: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class KECProcessingPipeline:
    """
    Pipeline completo KEC 2.0 para análise de biomateriais porosos.
    
    Etapas:
    1. Carregamento e pré-processamento de imagens
    2. Segmentação (Otsu local + filtros)
    3. Extração de grafo (PoreSpy ou método customizado)
    4. Cálculo de métricas KEC (H, κ, σ, ϕ, σ_Q)
    5. Análise estatística e reporting
    """
    
    def __init__(self, config: Optional[KECConfig] = None):
        self.config = config or KECConfig()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configura logging do pipeline."""
        if self.config.seed:
            np.random.seed(self.config.seed)
            
    async def process_image(self, 
                          image_data: Union[np.ndarray, str, Path],
                          processing_config: Optional[ProcessingConfig] = None) -> ProcessingResults:
        """
        Processa uma imagem completa através do pipeline KEC.
        
        Args:
            image_data: Array numpy, caminho da imagem, ou objeto Path
            processing_config: Configuração de processamento
            
        Returns:
            ProcessingResults com métricas e metadados
        """
        
        start_time = time.time()
        results = ProcessingResults(
            kec_metrics={},
            graph=nx.Graph(),
            image_metadata={},
            processing_time=0.0,
            config_used=self.config
        )
        
        try:
            # 1. Carregamento da imagem
            logger.info("Iniciando pipeline KEC 2.0...")
            image, metadata = await self._load_image(image_data)
            results.image_metadata = metadata
            
            if processing_config and processing_config.save_intermediates:
                results.intermediates["original_image"] = image.copy()
            
            # 2. Pré-processamento
            logger.info("Aplicando pré-processamento...")
            preprocessed = await self._preprocess_image(image)
            
            if processing_config and processing_config.save_intermediates:
                results.intermediates["preprocessed_image"] = preprocessed.copy()
            
            # 3. Segmentação
            logger.info(f"Executando segmentação: {self.config.segmentation_method}")
            segmented = await self._segment_image(preprocessed)
            
            if processing_config and processing_config.save_intermediates:
                results.intermediates["segmented_image"] = segmented.copy()
            
            # 4. Extração de grafo
            logger.info("Extraindo grafo da estrutura porosa...")
            graph = await self._extract_graph(segmented)
            results.graph = graph
            
            # 5. Cálculo de métricas KEC
            logger.info("Calculando métricas KEC 2.0...")
            metrics = await self._compute_kec_metrics(graph)
            results.kec_metrics = metrics
            
            # 6. Análise adicional
            if self.config.null_models:
                logger.info("Calculando modelos nulos...")
                null_stats = await self._compute_null_models(graph)
                results.intermediates["null_models"] = null_stats
            
            processing_time = time.time() - start_time
            results.processing_time = processing_time
            
            logger.info(f"Pipeline concluído em {processing_time:.2f}s")
            logger.info(f"Métricas KEC: H={metrics.get('H_spectral', 0):.3f}, "
                       f"κ_mean={metrics.get('k_forman_mean', 0):.3f}, "
                       f"σ={metrics.get('sigma', 0):.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"Erro no pipeline: {str(e)}"
            logger.error(error_msg)
            results.errors.append(error_msg)
            results.processing_time = time.time() - start_time
            return results
    
    async def _load_image(self, image_data: Union[np.ndarray, str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Carrega imagem de diferentes fontes."""
        
        if isinstance(image_data, np.ndarray):
            metadata = {
                "shape": image_data.shape,
                "dtype": str(image_data.dtype),
                "source": "numpy_array"
            }
            return image_data, metadata
        
        # Para desenvolvimento, retorna imagem sintética
        logger.warning("Carregamento de arquivo não implementado - usando imagem sintética")
        
        # Gera imagem sintética para demonstração
        size = 256
        image = np.random.rand(size, size) > 0.3  # Estrutura porosa sintética
        
        metadata = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "source": "synthetic",
            "size": size
        }
        
        return image.astype(np.uint8), metadata
    
    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica pré-processamento conforme configuração."""
        
        processed = image.copy()
        
        if self.config.bilateral_filter:
            logger.debug("Aplicando filtro bilateral...")
            # TODO: Implementar filtro bilateral real
            # Por enquanto, apenas suavização simples
            from scipy.ndimage import gaussian_filter
            processed = gaussian_filter(processed.astype(float), sigma=1.0)
        
        return processed
    
    async def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """Executa segmentação conforme método configurado."""
        
        if self.config.segmentation_method == "otsu_local":
            return await self._otsu_local_segmentation(image)
        elif self.config.segmentation_method == "otsu_global":
            return await self._otsu_global_segmentation(image)
        else:
            logger.warning(f"Método {self.config.segmentation_method} não implementado, usando threshold simples")
            return (image > np.mean(image)).astype(np.uint8)
    
    async def _otsu_local_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Segmentação Otsu local (implementação simplificada)."""
        
        try:
            from skimage.filters import threshold_local
            threshold = threshold_local(image, block_size=35, offset=0.01)
            segmented = (image > threshold).astype(np.uint8)
            return segmented
        except ImportError:
            logger.warning("scikit-image não disponível, usando threshold global")
            return await self._otsu_global_segmentation(image)
    
    async def _otsu_global_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Segmentação Otsu global (implementação simplificada)."""
        
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(image)
            segmented = (image > threshold).astype(np.uint8)
            return segmented
        except ImportError:
            # Fallback: threshold no percentil 50
            threshold = np.percentile(image, 50)
            return (image > threshold).astype(np.uint8)
    
    async def _extract_graph(self, segmented_image: np.ndarray) -> nx.Graph:
        """Extrai grafo da imagem segmentada."""
        
        if self.config.use_porespy:
            return await self._extract_graph_porespy(segmented_image)
        else:
            return await self._extract_graph_simple(segmented_image)
    
    async def _extract_graph_porespy(self, image: np.ndarray) -> nx.Graph:
        """Extração usando PoreSpy (se disponível)."""
        
        try:
            import porespy as ps
            
            # Análise de network usando PoreSpy
            snow = ps.networks.snow_partitioning(image)
            net = ps.networks.extract_pore_network(snow)
            
            # Converte para NetworkX
            G = nx.Graph()
            
            # Adiciona nós (poros)
            for i, pore in enumerate(net['pore.coords']):
                G.add_node(i, 
                          pos=tuple(pore),
                          volume=net['pore.volume'][i] if 'pore.volume' in net else 1.0)
            
            # Adiciona arestas (gargantas)
            conns = net['throat.conns']
            for i, (p1, p2) in enumerate(conns):
                weight_data = {}
                for throat_prop in self.config.throat_weights:
                    if f'throat.{throat_prop}' in net:
                        weight_data[throat_prop] = net[f'throat.{throat_prop}'][i]
                
                G.add_edge(p1, p2, **weight_data)
            
            logger.info(f"Grafo PoreSpy: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
            return G
            
        except ImportError:
            logger.warning("PoreSpy não disponível, usando extração simples")
            return await self._extract_graph_simple(image)
        except Exception as e:
            logger.error(f"Erro no PoreSpy: {e}, fallback para método simples")
            return await self._extract_graph_simple(image)
    
    async def _extract_graph_simple(self, image: np.ndarray) -> nx.Graph:
        """Extração simples baseada em conectividade de pixels."""
        
        # Método simplificado: cada região conectada vira um nó
        from scipy.ndimage import label
        
        # Identifica componentes conectados
        labeled, num_features = label(image)
        
        G = nx.Graph()
        
        # Adiciona nós para cada região
        for region_id in range(1, num_features + 1):
            region_mask = (labeled == region_id)
            centroid = np.mean(np.where(region_mask), axis=1)
            volume = np.sum(region_mask)
            
            G.add_node(region_id, 
                      pos=tuple(centroid),
                      volume=float(volume))
        
        # Conecta regiões adjacentes
        # Implementação simplificada: conecta por proximidade
        nodes = list(G.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                pos1 = np.array(data1['pos'])
                pos2 = np.array(data2['pos'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # Conecta se distância for menor que threshold
                if distance < np.sqrt(image.size) / 10:  # threshold adaptativo
                    G.add_edge(node1, node2, distance=distance)
        
        logger.info(f"Grafo simples: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
        return G
    
    async def _compute_kec_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """Calcula métricas KEC 2.0 usando o módulo de métricas."""
        
        if graph.number_of_nodes() == 0:
            logger.warning("Grafo vazio - retornando métricas zeradas")
            return {
                "H_spectral": 0.0,
                "k_forman_mean": 0.0,
                "k_forman_p05": 0.0,
                "k_forman_p50": 0.0,
                "k_forman_p95": 0.0,
                "sigma": 0.0,
                "swp": 0.0
            }
        
        # Usa função principal do módulo de métricas
        metrics = compute_kec_metrics(
            graph,
            spectral_k=self.config.k_eigs,
            include_triangles=self.config.include_triangles,
            n_random=self.config.n_random,
            sigma_q=self.config.sigma_Q
        )
        
        return metrics
    
    async def _compute_null_models(self, graph: nx.Graph) -> Dict[str, Any]:
        """Computa modelos nulos para comparação estatística."""
        
        if graph.number_of_nodes() < 3:
            return {"error": "Grafo muito pequeno para modelos nulos"}
        
        null_results = {
            "erdos_renyi": [],
            "configuration_model": [],
            "watts_strogatz": []
        }
        
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        # Gera modelos nulos
        for i in range(min(self.config.preserve_degree_nulls, 10)):  # Limita para performance
            try:
                # Erdős-Rényi
                G_er = nx.erdos_renyi_graph(n, m / (n * (n-1) / 2))
                er_metrics = compute_kec_metrics(G_er, sigma_q=False)
                null_results["erdos_renyi"].append(er_metrics)
                
                # Configuration model (preserva distribuição de graus)
                if m > 0:
                    degrees = [d for _, d in graph.degree()]
                    if sum(degrees) % 2 == 0:  # Soma deve ser par
                        G_conf = nx.configuration_model(degrees)
                        G_conf = nx.Graph(G_conf)  # Remove multi-edges
                        conf_metrics = compute_kec_metrics(G_conf, sigma_q=False)
                        null_results["configuration_model"].append(conf_metrics)
                
            except Exception as e:
                logger.debug(f"Erro no modelo nulo {i}: {e}")
        
        return null_results
    
    def get_summary_statistics(self, results: ProcessingResults) -> Dict[str, Any]:
        """Gera estatísticas resumidas dos resultados."""
        
        summary = {
            "processing_time": results.processing_time,
            "graph_nodes": results.graph.number_of_nodes(),
            "graph_edges": results.graph.number_of_edges(),
            "kec_metrics": results.kec_metrics.copy(),
            "errors": len(results.errors),
            "config_hash": hash(str(results.config_used.__dict__))
        }
        
        # Adiciona estatísticas do grafo
        if results.graph.number_of_nodes() > 0:
            degrees = [d for _, d in results.graph.degree()]
            summary["graph_stats"] = {
                "avg_degree": np.mean(degrees),
                "degree_std": np.std(degrees),
                "density": nx.density(results.graph),
                "is_connected": nx.is_connected(results.graph)
            }
        
        return summary