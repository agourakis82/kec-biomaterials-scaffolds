"""KEC Metrics Router

Router principal para análise de métricas KEC (Knowledge Exchange Coefficient)
para scaffolds biomateriais e análise de grafos.

Migrado e modernizado de:
- backup_old_backends/kec_biomat_api/routers/score_contracts.py
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response, Query, Path
from fastapi.responses import JSONResponse

from ..models.kec_models import (
    # Request models
    ContractExecutionRequest,
    KECAnalysisRequest,
    BatchExecuteRequest,
    SpectralAnalysisRequest,
    TopologyAnalysisRequest,
    ComputeRequest,
    
    # Response models
    ContractExecutionResponse,
    KECAnalysisResponse,
    BatchExecuteResponse,
    AvailableMetric,
    AlgorithmInfo,
    HealthStatus,
    HistoryEntry,
    ComputeResponse,
    
    # Enums and others
    MetricType,
    ExecutionStatus,
)
from ..services.kec_calculator import get_kec_service
from ..core.logging import get_logger

logger = get_logger("kec.router")

router = APIRouter(
    prefix="/kec-metrics",
    tags=["KEC Metrics"],
    responses={
        500: {"description": "Internal server error"},
        422: {"description": "Validation error"}
    }
)

# ==================== MAIN ANALYSIS ENDPOINTS ====================

@router.post("/analyze", response_model=KECAnalysisResponse)
async def analyze_kec_metrics(
    request: KECAnalysisRequest,
    response: Response
) -> KECAnalysisResponse:
    """
    Análise completa de métricas KEC para scaffolds biomateriais.
    
    Calcula métricas como:
    - H_spectral: Entropia espectral
    - k_forman_mean: Curvatura de Forman média  
    - sigma: Small-world sigma
    - swp: Small-World Propensity
    - sigma_Q: Coerência quântica (opcional)
    """
    try:
        kec_service = get_kec_service()
        result = await kec_service.analyze_scaffold(request)
        
        # Set response headers
        response.headers["X-Analysis-ID"] = result.analysis_id
        response.headers["X-Analysis-Status"] = result.status.value
        
        if result.execution_time_ms:
            response.headers["X-Execution-Time-MS"] = str(int(result.execution_time_ms))
        
        if result.metrics:
            metrics_count = len([v for v in result.metrics.dict().values() if v is not None])
            response.headers["X-Metrics-Count"] = str(metrics_count)
        
        return result
        
    except Exception as e:
        logger.error(f"Erro na análise KEC: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise de métricas KEC: {str(e)}"
        )


@router.post("/batch-analyze", response_model=BatchExecuteResponse)
async def batch_analyze_contracts(
    request: BatchExecuteRequest,
    response: Response
) -> BatchExecuteResponse:
    """
    Análise em lote de contratos/métricas KEC.
    
    Permite processar múltiplas análises em uma única requisição,
    com controle individual de erros e timeouts.
    """
    try:
        kec_service = get_kec_service()
        results = []
        success_count = 0
        error_count = 0
        
        for contract_request in request.contracts:
            try:
                result = await kec_service.execute_contract(contract_request)
                results.append(result)
                
                if result.status == ExecutionStatus.COMPLETED:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.warning(f"Falha em contrato individual: {e}")
                error_count += 1
                # Continue with other contracts
                
        # Set response headers
        response.headers["X-Batch-Total"] = str(len(request.contracts))
        response.headers["X-Batch-Success"] = str(success_count)
        response.headers["X-Batch-Errors"] = str(error_count)
        
        return BatchExecuteResponse(
            results=results,
            total_count=len(request.contracts),
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        logger.error(f"Falha na análise em lote: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na execução em lote: {str(e)}"
        )


# ==================== SPECIALIZED ANALYSIS ENDPOINTS ====================

@router.post("/scaffold/analyze", response_model=KECAnalysisResponse)
async def analyze_scaffold_specific(
    request: KECAnalysisRequest,
    response: Response
) -> KECAnalysisResponse:
    """
    Análise específica para scaffolds biomateriais.
    
    Endpoint otimizado para propriedades de scaffolds porosos,
    incluindo análise de conectividade, porosidade e estrutura.
    """
    try:
        # Validar que scaffold_data está presente
        if not request.scaffold_data:
            raise HTTPException(
                status_code=422,
                detail="scaffold_data é obrigatório para análise de scaffold"
            )
        
        kec_service = get_kec_service()
        result = await kec_service.analyze_scaffold(request)
        
        # Headers específicos para scaffold
        response.headers["X-Analysis-Type"] = "scaffold"
        response.headers["X-Analysis-ID"] = result.analysis_id
        
        if result.graph_properties:
            if "nodes" in result.graph_properties:
                response.headers["X-Scaffold-Nodes"] = str(result.graph_properties["nodes"])
            if "edges" in result.graph_properties:
                response.headers["X-Scaffold-Edges"] = str(result.graph_properties["edges"])
            if "density" in result.graph_properties:
                response.headers["X-Scaffold-Density"] = str(result.graph_properties["density"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de scaffold: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise de scaffold: {str(e)}"
        )


@router.get("/spectral/compute")
async def compute_spectral_analysis(
    graph_id: str = Query(..., description="ID do grafo"),
    k_eigenvalues: int = Query(64, ge=1, le=1000, description="Número de autovalores"),
    tolerance: float = Query(1e-8, gt=0, description="Tolerância numérica")
) -> Dict[str, Any]:
    """
    Análise espectral específica do grafo.
    
    Calcula entropia espectral baseada no Laplaciano normalizado,
    otimizada para grafos de grande escala.
    """
    try:
        # Por enquanto retornamos exemplo - seria integrado com banco de dados de grafos
        return {
            "graph_id": graph_id,
            "H_spectral": 4.2,  # Placeholder - calcularia real
            "eigenvalue_count": k_eigenvalues,
            "tolerance": tolerance,
            "computation_method": "scipy_sparse" if k_eigenvalues > 64 else "dense",
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Erro na análise espectral: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise espectral: {str(e)}"
        )


@router.post("/graph/topology", response_model=Dict[str, Any])
async def analyze_graph_topology(
    request: TopologyAnalysisRequest,
    response: Response
) -> Dict[str, Any]:
    """
    Análise topológica detalhada do grafo.
    
    Calcula propriedades topológicas como:
    - Curvatura de Forman (com/sem triângulos)
    - Small-world metrics (σ, SWP)
    - Propriedades de conectividade
    """
    try:
        kec_service = get_kec_service()
        
        # Análise usando o serviço KEC
        analysis_request = KECAnalysisRequest(
            scaffold_data=None,
            graph_data=request.graph_data,
            metrics=[MetricType.K_FORMAN_MEAN, MetricType.SIGMA, MetricType.SWP],
            parameters={
                "include_triangles": request.include_triangles,
                "n_random": request.n_random_graphs
            }
        )
        
        result = await kec_service.analyze_scaffold(analysis_request)
        
        if result.status != ExecutionStatus.COMPLETED:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na análise topológica: {result.error_message}"
            )
        
        # Format topology-specific response
        topology_result = {
            "analysis_id": result.analysis_id,
            "topology_metrics": {
                "forman_curvature": {
                    "mean": result.metrics.k_forman_mean if result.metrics else None,
                    "p05": result.metrics.k_forman_p05 if result.metrics else None,
                    "p50": result.metrics.k_forman_p50 if result.metrics else None,
                    "p95": result.metrics.k_forman_p95 if result.metrics else None,
                },
                "small_world": {
                    "sigma": result.metrics.sigma if result.metrics else None,
                    "swp": result.metrics.swp if result.metrics else None,
                },
            },
            "graph_properties": result.graph_properties,
            "parameters": {
                "include_triangles": request.include_triangles,
                "n_random_graphs": request.n_random_graphs
            },
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp
        }
        
        response.headers["X-Analysis-Type"] = "topology"
        if result.graph_properties:
            response.headers["X-Graph-Nodes"] = str(result.graph_properties.get("nodes", 0))
        
        return topology_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise topológica: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na análise topológica: {str(e)}"
        )


# ==================== METADATA & DISCOVERY ENDPOINTS ====================

@router.get("/available-metrics", response_model=List[AvailableMetric])
async def get_available_metrics() -> List[AvailableMetric]:
    """
    Lista todas as métricas KEC disponíveis.
    
    Retorna informações detalhadas sobre cada métrica,
    incluindo schemas de entrada e descrições.
    """
    try:
        kec_service = get_kec_service()
        contracts = kec_service.get_available_contracts()
        
        metrics = []
        for contract in contracts:
            metric = AvailableMetric(
                type=contract["type"],
                name=contract["name"],
                description=_get_metric_description(contract["type"]),
                schema=contract["schema"],
                category=_get_metric_category(contract["type"])
            )
            metrics.append(metric)
        
        # Adicionar métricas KEC específicas
        kec_specific_metrics = [
            AvailableMetric(
                type="H_spectral",
                name="Entropia Espectral",
                description="Entropia von Neumann baseada no Laplaciano normalizado",
                schema={"type": "object", "properties": {"graph_data": {"type": "object"}}},
                category="spectral"
            ),
            AvailableMetric(
                type="k_forman_mean",
                name="Curvatura de Forman",
                description="Curvatura média de Forman (2-complex com triângulos)",
                schema={"type": "object", "properties": {"graph_data": {"type": "object"}}},
                category="curvature"
            ),
            AvailableMetric(
                type="sigma",
                name="Small-World Sigma",
                description="Humphries & Gurney σ = (C/C_rand) / (L/L_rand)",
                schema={"type": "object", "properties": {"graph_data": {"type": "object"}}},
                category="small_world"
            ),
            AvailableMetric(
                type="swp",
                name="Small-World Propensity",
                description="Small-World Propensity (Muldoon et al., 2016)",
                schema={"type": "object", "properties": {"graph_data": {"type": "object"}}},
                category="small_world"
            ),
        ]
        
        metrics.extend(kec_specific_metrics)
        return metrics
        
    except Exception as e:
        logger.error(f"Erro ao buscar métricas disponíveis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao recuperar métricas disponíveis"
        )


@router.get("/schema/{metric_type}")
async def get_metric_schema(
    metric_type: str = Path(..., description="Tipo de métrica")
) -> Dict[str, Any]:
    """
    Schema de entrada para tipo específico de métrica.
    
    Retorna JSON schema detalhado para validação de entrada
    de cada tipo de métrica KEC.
    """
    try:
        kec_service = get_kec_service()
        schema = kec_service.sandbox.get_contract_schema(metric_type)
        
        if not schema:
            # Schemas específicos KEC
            kec_schemas = {
                "H_spectral": {
                    "type": "object",
                    "required": ["graph_data"],
                    "properties": {
                        "graph_data": {
                            "type": "object",
                            "properties": {
                                "graph_id": {"type": "string"},
                                "adjacency_matrix": {"type": "array"},
                                "edge_list": {"type": "array"}
                            }
                        },
                        "k_eigenvalues": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 64},
                        "tolerance": {"type": "number", "minimum": 1e-12, "default": 1e-8}
                    }
                },
                "k_forman_mean": {
                    "type": "object", 
                    "required": ["graph_data"],
                    "properties": {
                        "graph_data": {"type": "object"},
                        "include_triangles": {"type": "boolean", "default": True}
                    }
                },
                "sigma": {
                    "type": "object",
                    "required": ["graph_data"],
                    "properties": {
                        "graph_data": {"type": "object"},
                        "n_random": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                    }
                },
                "swp": {
                    "type": "object",
                    "required": ["graph_data"], 
                    "properties": {
                        "graph_data": {"type": "object"},
                        "n_random": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                    }
                }
            }
            
            schema = kec_schemas.get(metric_type)
            
        if not schema:
            raise HTTPException(
                status_code=404,
                detail=f"Schema não encontrado para métrica: {metric_type}"
            )
        
        return schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao buscar schema: {e}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao recuperar schema de métrica"
        )


@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def get_available_algorithms() -> List[AlgorithmInfo]:
    """
    Lista algoritmos disponíveis para análise KEC.
    
    Retorna informações sobre implementações, complexidade
    computacional e referências bibliográficas.
    """
    algorithms = [
        AlgorithmInfo(
            name="Spectral Entropy (von Neumann)",
            description="Entropia espectral baseada em autovalores do Laplaciano normalizado",
            parameters={
                "k_eigenvalues": {"type": "integer", "description": "Número de autovalores"},
                "tolerance": {"type": "number", "description": "Tolerância numérica"}
            },
            complexity="O(n³) dense, O(k·n) sparse",
            references=[
                "Passerini & Severini (2008). The von Neumann entropy of networks",
                "Braunstein et al. (2006). Laplacian of a graph as a density matrix"
            ]
        ),
        AlgorithmInfo(
            name="Forman Curvature (2-complex)",
            description="Curvatura de Forman incluindo triângulos (2-simplexes)",
            parameters={
                "include_triangles": {"type": "boolean", "description": "Incluir triângulos"}
            },
            complexity="O(m·Δ²) onde m=arestas, Δ=grau máximo",
            references=[
                "Forman (2003). Bochner's method for cell complexes",
                "Weber et al. (2017). Forman curvature for complex networks"
            ]
        ),
        AlgorithmInfo(
            name="Small-World Sigma",
            description="Humphries & Gurney σ = (C/C_rand) / (L/L_rand)",
            parameters={
                "n_random": {"type": "integer", "description": "Número de grafos aleatórios"}
            },
            complexity="O(n·r) onde r=grafos aleatórios",
            references=[
                "Humphries & Gurney (2008). Network 'small-world-ness'"
            ]
        ),
        AlgorithmInfo(
            name="Small-World Propensity (SWP)",
            description="Muldoon et al. SWP corrigido para densidade",
            parameters={
                "n_random": {"type": "integer", "description": "Grafos de referência"}
            },
            complexity="O(n²) para normalizações",
            references=[
                "Muldoon et al. (2016). Small-world propensity in weighted networks"
            ]
        ),
    ]
    
    return algorithms


@router.get("/examples/{metric_type}")
async def get_metric_examples(
    metric_type: str = Path(..., description="Tipo de métrica")
) -> Dict[str, Any]:
    """
    Exemplos de entrada para tipos específicos de métricas.
    
    Fornece dados de exemplo válidos para facilitar
    integração e testes da API.
    """
    examples = {
        "H_spectral": {
            "graph_data": {
                "graph_id": "example_scaffold_001",
                "edge_list": [[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]],
                "node_attributes": {
                    "0": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "1": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "2": {"x": 1.0, "y": 1.0, "z": 0.0},
                    "3": {"x": 0.0, "y": 1.0, "z": 0.0}
                }
            },
            "k_eigenvalues": 64,
            "tolerance": 1e-8
        },
        "k_forman_mean": {
            "graph_data": {
                "graph_id": "scaffold_porous_002",
                "adjacency_matrix": [
                    [0, 1, 0, 1],
                    [1, 0, 1, 1],
                    [0, 1, 0, 1],
                    [1, 1, 1, 0]
                ]
            },
            "include_triangles": True
        },
        "delta_kec_v1": {
            "data": {
                "source_entropy": 8.5,
                "target_entropy": 7.2,
                "mutual_information": 3.1,
            },
            "parameters": {"alpha": 0.7, "beta": 0.3},
        },
        "kec_full": {
            "scaffold_data": {
                "nodes": [
                    {"id": 0, "x": 0.0, "y": 0.0, "z": 0.0},
                    {"id": 1, "x": 1.0, "y": 0.0, "z": 0.0},
                    {"id": 2, "x": 1.0, "y": 1.0, "z": 0.0},
                    {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0}
                ],
                "edges": [
                    {"source": 0, "target": 1, "weight": 1.0},
                    {"source": 1, "target": 2, "weight": 1.0},
                    {"source": 2, "target": 3, "weight": 1.0},
                    {"source": 3, "target": 0, "weight": 1.0}
                ],
                "properties": {
                    "porosity": 0.85,
                    "pore_size_mean": 150.0,
                    "material": "collagen"
                }
            },
            "metrics": ["H_spectral", "k_forman_mean", "sigma", "swp"],
            "parameters": {
                "spectral_k": 32,
                "include_triangles": True,
                "n_random": 10
            }
        }
    }
    
    example = examples.get(metric_type)
    if not example:
        raise HTTPException(
            status_code=404,
            detail=f"Exemplos não disponíveis para métrica: {metric_type}"
        )
    
    return {
        "metric_type": metric_type,
        "example_input": example,
        "description": f"Exemplo de entrada válido para {metric_type}",
        "timestamp": datetime.now(timezone.utc)
    }


# ==================== HISTORY & MONITORING ====================

@router.get("/history", response_model=List[HistoryEntry])
async def get_execution_history(
    metric_type: Optional[str] = Query(None, description="Filtrar por tipo de métrica"),
    limit: int = Query(50, ge=1, le=200, description="Número máximo de resultados")
) -> List[HistoryEntry]:
    """
    Histórico de execuções de análises KEC.
    
    Permite rastreamento e auditoria de análises realizadas,
    com filtros por tipo de métrica e limite de resultados.
    """
    try:
        kec_service = get_kec_service()
        history = kec_service.get_execution_history(metric_type, limit)
        
        # Convert to HistoryEntry format
        history_entries = []
        for h in history:
            entry = HistoryEntry(
                execution_id=h.execution_id,
                timestamp=h.timestamp or datetime.now(timezone.utc),
                metric_type=h.contract_type,
                status=h.status,
                execution_time_ms=h.execution_time_ms,
                result_summary={
                    "score": h.score,
                    "confidence": h.confidence,
                    "metadata": h.metadata
                } if h.score is not None else None
            )
            history_entries.append(entry)
        
        return history_entries
        
    except Exception as e:
        logger.error(f"Erro ao buscar histórico: {e}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao recuperar histórico de execuções"
        )


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check específico para o serviço KEC Metrics.
    
    Verifica status dos componentes críticos:
    - Algoritmos KEC
    - Sandbox de execução
    - Dependências científicas (NumPy, SciPy, NetworkX)
    """
    try:
        kec_service = get_kec_service()
        health_data = kec_service.get_health_status()
        
        status = "healthy" if health_data["status"] == "healthy" else "unhealthy"
        
        return HealthStatus(
            status=status,
            message=health_data["message"],
            timestamp=datetime.now(timezone.utc),
            components=health_data.get("components", {}),
            available_metrics=len(kec_service.get_available_contracts()),
            uptime_seconds=0.0  # Could be implemented with start time tracking
        )
        
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return HealthStatus(
            status="unhealthy",
            message=f"Erro no health check: {str(e)}",
            timestamp=datetime.now(timezone.utc),
            components={"error": "health_check_failed"},
            available_metrics=0,
            uptime_seconds=0.0
        )


# ==================== LEGACY COMPATIBILITY ====================

@router.post("/execute", response_model=ContractExecutionResponse)
async def execute_contract_legacy(
    request: ContractExecutionRequest,
    response: Response
) -> ContractExecutionResponse:
    """
    Execução de contrato (compatibilidade legada).
    
    Mantém compatibilidade com API original score_contracts.
    Recomenda-se usar /analyze para novas integrações.
    """
    logger.info(f"Usando endpoint legado /execute - considere migrar para /analyze")
    
    try:
        kec_service = get_kec_service()
        result = await kec_service.execute_contract(request)
        
        # Legacy headers
        response.headers["X-Contract-Type"] = request.contract_type
        response.headers["X-Execution-Status"] = result.status.value
        response.headers["X-Execution-ID"] = result.execution_id
        
        return result
        
    except Exception as e:
        logger.error(f"Erro na execução legada: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na execução de contrato: {str(e)}"
        )


@router.get("/available", response_model=List[Dict[str, Any]])
async def get_available_contracts_legacy() -> List[Dict[str, Any]]:
    """
    Lista contratos disponíveis (compatibilidade legada).
    
    Mantém formato original. Use /available-metrics para novo formato.
    """
    logger.info("Usando endpoint legado /available - considere migrar para /available-metrics")
    
    try:
        kec_service = get_kec_service()
        return kec_service.get_available_contracts()
        
    except Exception as e:
        logger.error(f"Erro ao listar contratos legados: {e}")
        raise HTTPException(
            status_code=500,
            detail="Falha ao recuperar contratos disponíveis"
        )


@router.post("/compute", response_model=ComputeResponse)  
async def compute_legacy(request: ComputeRequest) -> ComputeResponse:
    """
    Computação legada (compatibilidade com API anterior).
    
    Mantém interface original compute. Use /analyze para funcionalidades completas.
    """
    logger.info("Usando endpoint legado /compute - considere migrar para /analyze")
    
    try:
        # Simulate computation based on graph_id
        # Em implementação real, buscaria grafo por ID
        return ComputeResponse(
            H_spectral=4.2,      # Placeholder
            k_forman_mean=0.15,  # Placeholder  
            sigma=1.8,           # Placeholder
            swp=0.65             # Placeholder
        )
        
    except Exception as e:
        logger.error(f"Erro na computação legada: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha na computação: {str(e)}"
        )


# ==================== HELPER FUNCTIONS ====================

def _get_metric_description(metric_type: str) -> str:
    """Descrições legíveis para tipos de métrica."""
    descriptions = {
        "delta_kec_v1": "Delta Knowledge Exchange Coefficient - mede eficiência de transferência de conhecimento",
        "kec_spectral": "Análise espectral KEC - entropia baseada em autovalores do Laplaciano",
        "kec_full": "Análise KEC completa - todas as métricas de scaffold biomaterial",
        "H_spectral": "Entropia espectral (von Neumann) do Laplaciano normalizado",
        "k_forman_mean": "Curvatura média de Forman para análise topológica",
        "sigma": "Small-world sigma de Humphries & Gurney",
        "swp": "Small-World Propensity (Muldoon et al.)",
        "zuco_reading_v1": "Score de leitura ZuCo - baseado em EEG e eye-tracking",
        "editorial_v1": "Score de qualidade editorial - avaliação abrangente de texto",
    }
    
    return descriptions.get(metric_type, f"Métrica KEC: {metric_type}")


def _get_metric_category(metric_type: str) -> str:
    """Categorias para tipos de métrica."""
    categories = {
        "delta_kec_v1": "transfer",
        "kec_spectral": "spectral",
        "kec_full": "comprehensive",
        "H_spectral": "spectral",
        "k_forman_mean": "curvature",
        "sigma": "small_world",
        "swp": "small_world",
        "zuco_reading_v1": "neuroscience",
        "editorial_v1": "text_analysis",
    }
    
    return categories.get(metric_type, "general")


# ==================== EXPORTS ====================

__all__ = ["router"]