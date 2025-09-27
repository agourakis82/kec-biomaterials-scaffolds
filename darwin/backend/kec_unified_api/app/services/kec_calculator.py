"""KEC Calculator Service

Serviço principal para cálculo de métricas KEC (Knowledge Exchange Coefficient)
para scaffolds biomateriais e análise de grafos.

Migrado e adaptado de:
- backup_old_backends/kec_biomat_api/services/score_contracts.py
- backup_old_backends/kec_biomat/metrics/kec_metrics.py
"""

import asyncio
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import networkx as nx
from numpy.typing import ArrayLike

from ..models.kec_models import (
    ExecutionStatus,
    MetricType,
    KECMetricsResult,
    ContractExecutionRequest,
    ContractExecutionResponse,
    KECAnalysisRequest,
    KECAnalysisResponse,
    ScaffoldData,
    GraphData,
)
from ..core.logging import get_logger

logger = get_logger("kec.calculator")

# Importações opcionais com fallbacks
try:
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY = True
except ImportError:
    logger.warning("SciPy não disponível, usando fallbacks numéricos")
    eigsh = None
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    logger.warning("Pandas não disponível")
    HAS_PANDAS = False


# ==================== CORE KEC ALGORITHMS ====================

class KECAlgorithms:
    """Implementação das métricas KEC 2.0."""
    
    @staticmethod
    def _normalized_laplacian(G: nx.Graph):
        """Retorna Laplaciano normalizado."""
        try:
            return nx.normalized_laplacian_matrix(G)
        except Exception:
            # Fallback manual
            A = nx.to_numpy_array(G)
            d = np.sum(A, axis=1)
            D_inv_sqrt = np.diag(1.0/np.sqrt(np.maximum(d, 1e-12)))
            L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
            return L
    
    @staticmethod
    def spectral_entropy(G: nx.Graph, k_eigs: int = 64, tol: float = 1e-8) -> float:
        """
        Entropia espectral (von Neumann) baseada no Laplaciano normalizado.
        
        S(ρ) = - sum_i p_i log(p_i), onde ρ ~ L / Tr(L)
        """
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
            
        try:
            L = KECAlgorithms._normalized_laplacian(G)
            
            # Calcula autovalores
            if HAS_SCIPY and eigsh is not None and hasattr(L, "shape") and min(L.shape) > 128:
                k = min(max(2, k_eigs), n-1)
                vals = np.abs(eigsh(L, k=k, which="LM", return_eigenvectors=False, tol=tol))
                # Aproxima com top-k; normalizar
                s = np.sum(vals)
                p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
            else:
                # Pequeno: autovalores completos
                if hasattr(L, "toarray"):
                    L = L.toarray()
                vals = np.linalg.eigvalsh(L)
                vals = np.abs(vals)
                s = np.sum(vals)
                p = (vals / s) if s > 0 else np.ones_like(vals)/len(vals)
                
        except Exception as e:
            logger.warning(f"Erro no cálculo espectral, usando fallback: {e}")
            # Fallback: grau como proxy
            deg = np.array([d for _, d in G.degree()], dtype=float)
            p = deg / np.sum(deg) if deg.sum() > 0 else np.ones(len(deg))/len(deg)
            
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))
    
    @staticmethod
    def forman_curvature_stats(G: nx.Graph, include_triangles: bool = True) -> Dict[str, float]:
        """
        Curvatura de Forman por aresta. Retorna estatísticas globais.
        
        F(u,v) ≈ deg(u) + deg(v) - 2 - t(u,v), onde t=triângulos incidentes.
        """
        if G.number_of_edges() == 0:
            return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
            
        deg = dict(G.degree())
        
        def edge_triangles(u, v):
            if not include_triangles:
                return 0
            Nu = set(G.neighbors(u)) - {v}
            Nv = set(G.neighbors(v)) - {u}
            return len(Nu & Nv)
        
        vals = []
        for u, v in G.edges():
            t = edge_triangles(u, v)
            f = (deg[u] + deg[v] - 2) - t
            vals.append(f)
            
        a = np.array(vals, dtype=float)
        return {
            "mean": float(np.mean(a)),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }
    
    @staticmethod
    def small_world_sigma(G: nx.Graph, n_random: int = 20) -> float:
        """Humphries & Gurney σ = (C/C_rand) / (L/L_rand)."""
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0
            
        try:
            C = nx.average_clustering(G)
            # Usar maior componente conectado
            largest_cc = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc).copy()
            L = nx.average_shortest_path_length(G_main)
            
            # Redes aleatórias equivalentes
            Cr, Lr = 0.0, 0.0
            for _ in range(max(1, n_random)):
                Gr = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
                Cr += nx.average_clustering(Gr)
                # Usar maior componente
                if nx.is_connected(Gr):
                    Lr += nx.average_shortest_path_length(Gr)
                else:
                    largest_cc_r = max(nx.connected_components(Gr), key=len)
                    Gr_main = Gr.subgraph(largest_cc_r)
                    Lr += nx.average_shortest_path_length(Gr_main)
                    
            Cr /= max(1, n_random)
            Lr /= max(1, n_random)
            
            if Cr == 0 or Lr == 0:
                return 0.0
            return float((C/Cr) / (L/Lr))
            
        except Exception as e:
            logger.warning(f"Erro no cálculo small-world sigma: {e}")
            return 0.0
    
    @staticmethod
    def small_world_propensity(G: nx.Graph, n_random: int = 20) -> float:
        """
        SWP (Muldoon et al., 2016): 0..1, corrige viés de densidade.
        Implementação simplificada suficiente para ranking/comparação.
        """
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return 0.0
            
        try:
            C = nx.average_clustering(G)
            n = G.number_of_nodes()
            k_bar = float(np.mean([d for _, d in G.degree()]))
            
            # Estimativas aproximadas para clustering
            C_latt = min(1.0, (3*(k_bar-2)) / (4*(k_bar-1)+1e-9)) if k_bar >= 2 else 0.0
            C_rand = k_bar / (n-1) if n > 1 else 0.0
            
            # Normalização clustering
            C_norm = (C - C_rand) / (C_latt - C_rand + 1e-12)
            C_norm = float(np.clip(C_norm, 0.0, 1.0))
            
            # Path length normalizado
            largest_cc = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc).copy()
            L = nx.average_shortest_path_length(G_main)
            
            L_latt = n / (2*max(1.0, k_bar))  # proxy lattice
            L_rand = (np.log(n) / np.log(max(2.0, k_bar))) if k_bar > 1 else L
            L_norm = (L_rand - L) / (L_rand - L_latt + 1e-12)
            L_norm = float(np.clip(L_norm, 0.0, 1.0))
            
            # SWP como média dos dois normalizados
            return float((C_norm + L_norm)/2.0)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo SWP: {e}")
            return 0.0
    
    @staticmethod
    def quantum_coherence_sigma(G: nx.Graph, k_eigs: int = 64, tol: float = 1e-8) -> float:
        """
        σ_Q = S(diag ρ) − S(ρ), onde ρ deriva do Laplaciano normalizado.
        
        diag ρ ~ distribuição de graus normalizada (proxy).
        """
        n = G.number_of_nodes()
        if n == 0:
            return 0.0
            
        # S(ρ): entropia espectral
        S_spectral = KECAlgorithms.spectral_entropy(G, k_eigs=k_eigs, tol=tol)
        
        # S(diag ρ): graus normalizados
        deg = np.array([d for _, d in G.degree()], dtype=float)
        p = deg/deg.sum() if deg.sum() > 0 else np.ones(n)/n
        p = p[p > 1e-15]
        S_diag = float(-np.sum(p*np.log(p)))
        
        return float(max(0.0, S_diag - S_spectral))


# ==================== GRAPH UTILITIES ====================

class GraphUtils:
    """Utilitários para conversão e manipulação de grafos."""
    
    @staticmethod
    def scaffold_to_networkx(scaffold_data: ScaffoldData) -> nx.Graph:
        """Converte dados de scaffold para NetworkX Graph."""
        G = nx.Graph()
        
        # Adicionar nós
        for node in scaffold_data.nodes:
            node_id = node.get("id", len(G.nodes))
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})
        
        # Adicionar arestas
        for edge in scaffold_data.edges:
            source = edge.get("source")
            target = edge.get("target")
            if source is not None and target is not None:
                weight = edge.get("weight", 1.0)
                G.add_edge(source, target, weight=weight, 
                          **{k: v for k, v in edge.items() if k not in ["source", "target", "weight"]})
        
        return G
    
    @staticmethod
    def graph_data_to_networkx(graph_data: GraphData) -> nx.Graph:
        """Converte GraphData para NetworkX Graph."""
        G = nx.Graph()
        
        # Método 1: Matriz de adjacência
        if graph_data.adjacency_matrix:
            adj_matrix = np.array(graph_data.adjacency_matrix)
            G = nx.from_numpy_array(adj_matrix)
            
        # Método 2: Lista de arestas
        elif graph_data.edge_list:
            for edge in graph_data.edge_list:
                if len(edge) >= 2:
                    G.add_edge(edge[0], edge[1])
        
        # Adicionar atributos de nós
        if graph_data.node_attributes:
            for node_id, attrs in graph_data.node_attributes.items():
                if G.has_node(node_id):
                    G.nodes[node_id].update(attrs)
        
        # Adicionar atributos de arestas
        if graph_data.edge_attributes:
            for edge_key, attrs in graph_data.edge_attributes.items():
                # Assumindo edge_key como "u-v" ou tupla
                if isinstance(edge_key, str) and "-" in edge_key:
                    u, v = map(int, edge_key.split("-"))
                    if G.has_edge(u, v):
                        G.edges[u, v].update(attrs)
        
        return G
    
    @staticmethod
    def get_graph_properties(G: nx.Graph) -> Dict[str, Any]:
        """Extrai propriedades básicas do grafo."""
        properties = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
        }
        
        if G.number_of_nodes() > 0:
            properties.update({
                "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
                "clustering_coefficient": nx.average_clustering(G),
                "number_of_components": nx.number_connected_components(G),
            })
            
            if nx.is_connected(G) and G.number_of_nodes() > 1:
                properties["diameter"] = nx.diameter(G)
                properties["average_path_length"] = nx.average_shortest_path_length(G)
        
        return properties


# ==================== SANDBOX EXECUTION ====================

@dataclass
class ContractInput:
    """Input para execução de contrato."""
    contract_type: str
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    timeout_seconds: float = 30.0


class KECSandbox:
    """Sandbox seguro para execução de análises KEC."""
    
    def __init__(self):
        self._history: List[ContractExecutionResponse] = []
        self._algorithms = KECAlgorithms()
        self._graph_utils = GraphUtils()
    
    def get_available_contracts(self) -> List[Dict[str, Any]]:
        """Lista contratos/métricas disponíveis."""
        return [
            {
                "type": "delta_kec_v1",
                "name": "Delta KEC v1",
                "schema": {
                    "type": "object",
                    "properties": {
                        "source_entropy": {"type": "number"},
                        "target_entropy": {"type": "number"},
                        "mutual_information": {"type": "number"}
                    }
                }
            },
            {
                "type": "kec_spectral",
                "name": "KEC Spectral Analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "graph_data": {"type": "object"},
                        "k_eigenvalues": {"type": "integer", "default": 64}
                    }
                }
            },
            {
                "type": "kec_full",
                "name": "KEC Full Analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scaffold_data": {"type": "object"},
                        "metrics": {"type": "array"},
                        "parameters": {"type": "object"}
                    }
                }
            }
        ]
    
    def get_contract_schema(self, contract_type: str) -> Optional[Dict[str, Any]]:
        """Schema para tipo específico de contrato."""
        contracts = self.get_available_contracts()
        for contract in contracts:
            if contract["type"] == contract_type:
                return contract["schema"]
        return None
    
    def get_execution_history(self, contract_type: Optional[str] = None, 
                            limit: int = 50) -> List[ContractExecutionResponse]:
        """Histórico de execuções."""
        history = self._history
        if contract_type:
            history = [h for h in history if h.contract_type == contract_type]
        return history[-limit:]
    
    async def execute_contract(self, contract_input: ContractInput) -> ContractExecutionResponse:
        """Executa contrato no sandbox."""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            async def _execute():
                if contract_input.contract_type == "delta_kec_v1":
                    return await self._execute_delta_kec_v1(contract_input)
                elif contract_input.contract_type == "kec_spectral":
                    return await self._execute_kec_spectral(contract_input)
                elif contract_input.contract_type == "kec_full":
                    return await self._execute_kec_full(contract_input)
                else:
                    raise ValueError(f"Tipo de contrato desconhecido: {contract_input.contract_type}")
            
            result = await asyncio.wait_for(_execute(), timeout=contract_input.timeout_seconds)
            
            response = ContractExecutionResponse(
                execution_id=execution_id,
                contract_type=contract_input.contract_type,
                status=ExecutionStatus.COMPLETED,
                score=result.get("score"),
                confidence=result.get("confidence", 0.85),
                metadata=result.get("metadata", {"sandbox": True}),
                execution_time_ms=(time.time() - start_time) * 1000.0,
                timestamp=datetime.now(timezone.utc)
            )
            
        except asyncio.TimeoutError:
            response = ContractExecutionResponse(
                execution_id=execution_id,
                contract_type=contract_input.contract_type,
                status=ExecutionStatus.TIMEOUT,
                error_message="Execução excedeu timeout",
                execution_time_ms=(time.time() - start_time) * 1000.0,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Erro na execução do contrato: {e}")
            response = ContractExecutionResponse(
                execution_id=execution_id,
                contract_type=contract_input.contract_type,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000.0,
                timestamp=datetime.now(timezone.utc)
            )
        
        self._history.append(response)
        return response
    
    async def _execute_delta_kec_v1(self, contract_input: ContractInput) -> Dict[str, Any]:
        """Executa Delta KEC v1."""
        data = contract_input.data
        source_entropy = float(data.get("source_entropy", 0))
        target_entropy = float(data.get("target_entropy", 0))
        
        # Cálculo Delta KEC simplificado
        delta = target_entropy - source_entropy
        score = max(-3.0, min(3.0, delta))
        
        return {
            "score": score,
            "confidence": 0.75,
            "metadata": {
                "delta_kec": delta,
                "source_entropy": source_entropy,
                "target_entropy": target_entropy
            }
        }
    
    async def _execute_kec_spectral(self, contract_input: ContractInput) -> Dict[str, Any]:
        """Executa análise espectral KEC."""
        data = contract_input.data
        graph_data = GraphData(**data.get("graph_data", {}))
        k_eigs = data.get("k_eigenvalues", 64)
        
        G = self._graph_utils.graph_data_to_networkx(graph_data)
        H_spectral = self._algorithms.spectral_entropy(G, k_eigs=k_eigs)
        
        return {
            "score": H_spectral,
            "confidence": 0.9,
            "metadata": {
                "H_spectral": H_spectral,
                "graph_nodes": G.number_of_nodes(),
                "graph_edges": G.number_of_edges()
            }
        }
    
    async def _execute_kec_full(self, contract_input: ContractInput) -> Dict[str, Any]:
        """Executa análise KEC completa."""
        data = contract_input.data
        parameters = contract_input.parameters or {}
        
        # Converter dados para grafo
        if "scaffold_data" in data:
            scaffold = ScaffoldData(**data["scaffold_data"])
            G = self._graph_utils.scaffold_to_networkx(scaffold)
        elif "graph_data" in data:
            graph_data = GraphData(**data["graph_data"])
            G = self._graph_utils.graph_data_to_networkx(graph_data)
        else:
            raise ValueError("Dados de grafo ou scaffold necessários")
        
        # Calcular métricas
        metrics_requested = data.get("metrics", ["H_spectral", "k_forman_mean", "sigma", "swp"])
        results = {}
        
        if "H_spectral" in metrics_requested:
            results["H_spectral"] = self._algorithms.spectral_entropy(
                G, k_eigs=parameters.get("spectral_k", 64)
            )
        
        if any(m.startswith("k_forman") for m in metrics_requested):
            forman_stats = self._algorithms.forman_curvature_stats(
                G, include_triangles=parameters.get("include_triangles", True)
            )
            results.update(forman_stats)
        
        if "sigma" in metrics_requested:
            results["sigma"] = self._algorithms.small_world_sigma(
                G, n_random=parameters.get("n_random", 20)
            )
        
        if "swp" in metrics_requested:
            results["swp"] = self._algorithms.small_world_propensity(
                G, n_random=parameters.get("n_random", 20)
            )
        
        if "sigma_Q" in metrics_requested:
            results["sigma_Q"] = self._algorithms.quantum_coherence_sigma(
                G, k_eigs=parameters.get("spectral_k", 64)
            )
        
        # Score agregado (média das métricas normalizadas)
        scores = [v for v in results.values() if isinstance(v, (int, float)) and not math.isnan(v)]
        aggregate_score = np.mean(scores) if scores else 0.0
        
        return {
            "score": aggregate_score,
            "confidence": 0.9,
            "metadata": {
                **results,
                "graph_properties": self._graph_utils.get_graph_properties(G)
            }
        }


# ==================== MAIN SERVICE ====================

class KECCalculatorService:
    """Serviço principal para cálculos KEC."""
    
    def __init__(self):
        self.sandbox = KECSandbox()
        self.algorithms = KECAlgorithms()
        self.graph_utils = GraphUtils()
        logger.info("KEC Calculator Service inicializado")
    
    async def analyze_scaffold(self, request: KECAnalysisRequest) -> KECAnalysisResponse:
        """Análise completa de scaffold."""
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Converter para grafo
            if request.scaffold_data:
                G = self.graph_utils.scaffold_to_networkx(request.scaffold_data)
            elif request.graph_data:
                G = self.graph_utils.graph_data_to_networkx(request.graph_data)
            else:
                raise ValueError("scaffold_data ou graph_data necessários")
            
            # Parâmetros
            params = request.parameters or {}
            
            # Calcular métricas solicitadas
            results = KECMetricsResult()
            
            for metric in request.metrics:
                if metric == MetricType.H_SPECTRAL:
                    results.H_spectral = self.algorithms.spectral_entropy(
                        G, k_eigs=params.get("spectral_k", 64)
                    )
                elif metric == MetricType.K_FORMAN_MEAN:
                    forman_stats = self.algorithms.forman_curvature_stats(
                        G, include_triangles=params.get("include_triangles", True)
                    )
                    results.k_forman_mean = forman_stats["mean"]
                    results.k_forman_p05 = forman_stats["p05"]
                    results.k_forman_p50 = forman_stats["p50"]
                    results.k_forman_p95 = forman_stats["p95"]
                elif metric == MetricType.SIGMA:
                    results.sigma = self.algorithms.small_world_sigma(
                        G, n_random=params.get("n_random", 20)
                    )
                elif metric == MetricType.SWP:
                    results.swp = self.algorithms.small_world_propensity(
                        G, n_random=params.get("n_random", 20)
                    )
                elif metric == MetricType.SIGMA_Q:
                    results.sigma_Q = self.algorithms.quantum_coherence_sigma(
                        G, k_eigs=params.get("spectral_k", 64)
                    )
            
            execution_time = (time.time() - start_time) * 1000.0
            
            return KECAnalysisResponse(
                analysis_id=analysis_id,
                status=ExecutionStatus.COMPLETED,
                metrics=results,
                graph_properties=self.graph_utils.get_graph_properties(G),
                execution_time_ms=execution_time,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de scaffold: {e}")
            return KECAnalysisResponse(
                analysis_id=analysis_id,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def execute_contract(self, request: ContractExecutionRequest) -> ContractExecutionResponse:
        """Executa contrato via sandbox."""
        contract_input = ContractInput(
            contract_type=request.contract_type,
            data=request.data,
            parameters=request.parameters,
            timeout_seconds=request.timeout_seconds or 30.0
        )
        
        return await self.sandbox.execute_contract(contract_input)
    
    def get_available_contracts(self) -> List[Dict[str, Any]]:
        """Lista contratos disponíveis."""
        return self.sandbox.get_available_contracts()
    
    def get_execution_history(self, contract_type: Optional[str] = None, 
                            limit: int = 50) -> List[ContractExecutionResponse]:
        """Histórico de execuções."""
        return self.sandbox.get_execution_history(contract_type, limit)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Status de saúde do serviço."""
        try:
            # Teste simples
            G = nx.path_graph(10)
            H = self.algorithms.spectral_entropy(G, k_eigs=8)
            
            return {
                "status": "healthy",
                "message": "KEC Calculator operacional",
                "components": {
                    "algorithms": "operational",
                    "sandbox": "operational", 
                    "networkx": "operational",
                    "scipy": "operational" if HAS_SCIPY else "fallback",
                    "pandas": "operational" if HAS_PANDAS else "unavailable"
                },
                "test_result": {
                    "spectral_entropy": H,
                    "graph_nodes": 10
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Erro no KEC Calculator: {str(e)}",
                "components": {
                    "algorithms": "error",
                    "sandbox": "error"
                }
            }


# ==================== SINGLETON INSTANCE ====================

_kec_service: Optional[KECCalculatorService] = None

def get_kec_service() -> KECCalculatorService:
    """Retorna instância singleton do serviço KEC."""
    global _kec_service
    if _kec_service is None:
        _kec_service = KECCalculatorService()
    return _kec_service


# ==================== EXPORTS ====================

__all__ = [
    "KECCalculatorService",
    "KECAlgorithms", 
    "GraphUtils",
    "KECSandbox",
    "get_kec_service",
]