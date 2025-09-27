"""KEC Metrics Models - Pydantic v2

Modelos para análise de métricas KEC (Knowledge Exchange Coefficient)
para scaffolds biomateriais e análise de grafos.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class MetricType(str, Enum):
    """Tipos de métricas KEC disponíveis."""
    H_SPECTRAL = "H_spectral"
    K_FORMAN_MEAN = "k_forman_mean"
    SIGMA = "sigma"
    SWP = "swp"
    SIGMA_Q = "sigma_Q"
    DELTA_KEC_V1 = "delta_kec_v1"
    ZUCO_READING_V1 = "zuco_reading_v1"
    EDITORIAL_V1 = "editorial_v1"


class ExecutionStatus(str, Enum):
    """Status de execução das métricas."""
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    PENDING = "pending"
    RUNNING = "running"


# ==================== REQUEST MODELS ====================

class ScaffoldData(BaseModel):
    """Dados de scaffold para análise."""
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}],
                "edges": [{"source": 1, "target": 2, "weight": 1.0}],
                "properties": {"porosity": 0.85, "pore_size": 100.0}
            }
        }
    )
    
    nodes: List[Dict[str, Any]] = Field(
        ..., 
        description="Lista de nós do grafo com propriedades",
        min_length=1
    )
    edges: List[Dict[str, Any]] = Field(
        ..., 
        description="Lista de arestas do grafo",
        min_length=0
    )
    properties: Optional[Dict[str, Any]] = Field(
        None, 
        description="Propriedades físicas do scaffold"
    )


class GraphData(BaseModel):
    """Dados de grafo genérico para análise."""
    model_config = ConfigDict(extra="allow")
    
    graph_id: str = Field(..., description="Identificador único do grafo")
    adjacency_matrix: Optional[List[List[float]]] = Field(
        None, description="Matriz de adjacência"
    )
    edge_list: Optional[List[List[int]]] = Field(
        None, description="Lista de arestas como pares [source, target]"
    )
    node_attributes: Optional[Dict[str, Any]] = Field(
        None, description="Atributos dos nós"
    )
    edge_attributes: Optional[Dict[str, Any]] = Field(
        None, description="Atributos das arestas"
    )


class ContractExecutionRequest(BaseModel):
    """Request para execução de contratos/métricas."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "contract_type": "delta_kec_v1",
                "data": {"source_entropy": 8.5, "target_entropy": 7.2},
                "parameters": {"alpha": 0.7, "beta": 0.3},
                "timeout_seconds": 30.0
            }
        }
    )
    
    contract_type: str = Field(..., description="Tipo de contrato/métrica")
    data: Dict[str, Any] = Field(..., description="Dados de entrada")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Parâmetros de configuração"
    )
    timeout_seconds: Optional[float] = Field(
        30.0, ge=1.0, le=300.0, description="Timeout em segundos"
    )


class KECAnalysisRequest(BaseModel):
    """Request para análise KEC de scaffolds."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scaffold_data": {
                    "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
                    "edges": [{"source": 1, "target": 2}]
                },
                "metrics": ["H_spectral", "k_forman_mean", "sigma", "swp"],
                "parameters": {"spectral_k": 64, "include_triangles": True}
            }
        }
    )
    
    scaffold_data: Optional[ScaffoldData] = Field(
        None, description="Dados do scaffold"
    )
    graph_data: Optional[GraphData] = Field(
        None, description="Dados do grafo"
    )
    metrics: List[MetricType] = Field(
        default=[MetricType.H_SPECTRAL, MetricType.K_FORMAN_MEAN, 
                MetricType.SIGMA, MetricType.SWP],
        description="Lista de métricas a calcular"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Parâmetros específicos da análise"
    )


class BatchExecuteRequest(BaseModel):
    """Request para execução em lote."""
    model_config = ConfigDict()
    
    contracts: List[ContractExecutionRequest] = Field(
        ..., max_length=10, description="Lista de contratos para executar"
    )


class SpectralAnalysisRequest(BaseModel):
    """Request para análise espectral específica."""
    model_config = ConfigDict()
    
    graph_data: GraphData = Field(..., description="Dados do grafo")
    k_eigenvalues: Optional[int] = Field(
        64, ge=1, le=1000, description="Número de autovalores"
    )
    tolerance: Optional[float] = Field(
        1e-8, gt=0, description="Tolerância numérica"
    )


class TopologyAnalysisRequest(BaseModel):
    """Request para análise topológica."""
    model_config = ConfigDict()
    
    graph_data: GraphData = Field(..., description="Dados do grafo")
    include_triangles: Optional[bool] = Field(
        True, description="Incluir triângulos na curvatura"
    )
    n_random_graphs: Optional[int] = Field(
        20, ge=1, le=100, description="Número de grafos aleatórios para comparação"
    )


# ==================== RESPONSE MODELS ====================

class KECMetricsResult(BaseModel):
    """Resultado das métricas KEC."""
    model_config = ConfigDict()
    
    H_spectral: Optional[float] = Field(None, description="Entropia espectral")
    k_forman_mean: Optional[float] = Field(None, description="Curvatura de Forman média")
    k_forman_p05: Optional[float] = Field(None, description="Curvatura Forman percentil 5")
    k_forman_p50: Optional[float] = Field(None, description="Curvatura Forman mediana")
    k_forman_p95: Optional[float] = Field(None, description="Curvatura Forman percentil 95")
    sigma: Optional[float] = Field(None, description="Small-world sigma")
    swp: Optional[float] = Field(None, description="Small-World Propensity")
    sigma_Q: Optional[float] = Field(None, description="Coerência quântica sigma")


class ContractExecutionResponse(BaseModel):
    """Response de execução de contrato."""
    model_config = ConfigDict()
    
    execution_id: str = Field(..., description="ID único da execução")
    contract_type: str = Field(..., description="Tipo de contrato executado")
    status: ExecutionStatus = Field(..., description="Status da execução")
    score: Optional[float] = Field(None, description="Score resultante")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confiança")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados adicionais")
    error_message: Optional[str] = Field(None, description="Mensagem de erro se aplicável")
    execution_time_ms: Optional[float] = Field(None, ge=0, description="Tempo de execução")
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp da execução"
    )


class KECAnalysisResponse(BaseModel):
    """Response da análise KEC."""
    model_config = ConfigDict()
    
    analysis_id: str = Field(..., description="ID único da análise")
    status: ExecutionStatus = Field(..., description="Status da análise")
    metrics: Optional[KECMetricsResult] = Field(None, description="Resultados das métricas")
    graph_properties: Optional[Dict[str, Any]] = Field(
        None, description="Propriedades do grafo analisado"
    )
    execution_time_ms: Optional[float] = Field(None, ge=0, description="Tempo de execução")
    error_message: Optional[str] = Field(None, description="Mensagem de erro")
    timestamp: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp da análise"
    )


class BatchExecuteResponse(BaseModel):
    """Response de execução em lote."""
    model_config = ConfigDict()
    
    results: List[ContractExecutionResponse] = Field(
        ..., description="Resultados individuais"
    )
    total_count: int = Field(..., ge=0, description="Total de execuções")
    success_count: int = Field(..., ge=0, description="Execuções bem-sucedidas")
    error_count: int = Field(..., ge=0, description="Execuções com erro")


class AvailableMetric(BaseModel):
    """Métrica disponível para análise."""
    model_config = ConfigDict()
    
    type: str = Field(..., description="Tipo da métrica")
    name: str = Field(..., description="Nome legível")
    description: str = Field(..., description="Descrição da métrica")
    schema: Dict[str, Any] = Field(..., description="Schema de entrada")
    category: Optional[str] = Field(None, description="Categoria da métrica")


class AlgorithmInfo(BaseModel):
    """Informações sobre algoritmo disponível."""
    model_config = ConfigDict()
    
    name: str = Field(..., description="Nome do algoritmo")
    description: str = Field(..., description="Descrição do algoritmo")
    parameters: Dict[str, Any] = Field(..., description="Parâmetros aceitos")
    complexity: Optional[str] = Field(None, description="Complexidade computacional")
    references: Optional[List[str]] = Field(None, description="Referências bibliográficas")


class HealthStatus(BaseModel):
    """Status de saúde do serviço."""
    model_config = ConfigDict()
    
    status: str = Field(..., description="Status geral")
    message: str = Field(..., description="Mensagem do status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp do check"
    )
    components: Dict[str, str] = Field(
        default_factory=dict, description="Status dos componentes"
    )
    available_metrics: Optional[int] = Field(None, description="Métricas disponíveis")
    uptime_seconds: Optional[float] = Field(None, description="Uptime em segundos")


class HistoryEntry(BaseModel):
    """Entrada do histórico de execuções."""
    model_config = ConfigDict()
    
    execution_id: str = Field(..., description="ID da execução")
    timestamp: datetime = Field(..., description="Timestamp")
    metric_type: str = Field(..., description="Tipo de métrica")
    status: ExecutionStatus = Field(..., description="Status")
    execution_time_ms: Optional[float] = Field(None, description="Tempo de execução")
    result_summary: Optional[Dict[str, Any]] = Field(None, description="Resumo do resultado")


# ==================== LEGACY COMPATIBILITY ====================

class ComputeRequest(BaseModel):
    """Request compatível com API legada."""
    model_config = ConfigDict()
    
    graph_id: str = Field(..., description="ID do grafo")
    sigma_q: bool = Field(False, description="Incluir sigma quântico")


class ComputeResponse(BaseModel):
    """Response compatível com API legada."""
    model_config = ConfigDict()
    
    H_spectral: float = Field(..., description="Entropia espectral")
    k_forman_mean: float = Field(..., description="Curvatura de Forman média")
    sigma: float = Field(..., description="Small-world sigma")
    swp: float = Field(..., description="Small-World Propensity")


# ==================== EXPORT ====================

__all__ = [
    # Enums
    "MetricType",
    "ExecutionStatus",
    
    # Request Models
    "ScaffoldData",
    "GraphData",
    "ContractExecutionRequest", 
    "KECAnalysisRequest",
    "BatchExecuteRequest",
    "SpectralAnalysisRequest",
    "TopologyAnalysisRequest",
    
    # Response Models
    "KECMetricsResult",
    "ContractExecutionResponse",
    "KECAnalysisResponse", 
    "BatchExecuteResponse",
    "AvailableMetric",
    "AlgorithmInfo",
    "HealthStatus",
    "HistoryEntry",
    
    # Legacy Compatibility
    "ComputeRequest",
    "ComputeResponse",
]