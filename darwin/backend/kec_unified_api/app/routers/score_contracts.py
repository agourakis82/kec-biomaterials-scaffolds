"""Score Contracts Router - Sistema Completo de Contratos

Router principal com todos os endpoints para execução de contratos matemáticos
em sandbox seguro. Migrado e expandido do backend principal.

Feature crítica #5 para mestrado - Router completo funcional.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from fastapi.responses import JSONResponse

from ..models.contract_models import (
    ContractType,
    ContractExecutionRequest,
    ContractExecutionResponse,
    BatchExecuteRequest,
    BatchExecuteResponse,
    AvailableContract,
    ContractSchema,
    ExecutionHistory,
    ContractValidationResult,
    ContractPerformanceMetrics,
    SystemHealthStatus,
    ContractAPIResponse,
    SecurityLevel,
    
    # Specialized contract models
    DeltaKECContract,
    ZuCoReadingContract,
    EditorialContract,
    BiomaterialsContract,
    NetworkAnalysisContract,
    SpectralAnalysisContract
)
from ..services.contract_executor import get_contract_executor, ContractExecutor
from ..services.contract_validator import get_contract_validator
from ..services.mathematical_contracts import (
    execute_mathematical_contract,
    get_available_mathematical_contracts,
    validate_contract_input
)
from ..services.sandbox_manager import get_sandbox_manager

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/contracts",
    tags=["Score Contracts"],
    responses={404: {"description": "Not found"}}
)


# ============================================================================
# CORE ENDPOINTS - MIGRADOS DO BACKEND PRINCIPAL
# ============================================================================

@router.post("/execute", response_model=ContractExecutionResponse)
async def execute_contract(
    request: ContractExecutionRequest,
    response: Response,
    user_id: Optional[str] = Query(None, description="User ID for tracking")
) -> ContractExecutionResponse:
    """
    Executa contrato individual em sandbox seguro.
    
    Endpoint principal migrado do backend com sandbox execution avançado.
    """
    try:
        # Validação pré-execução
        validator = get_contract_validator()
        validation_result = validator.validate_contract_request(request)
        
        if not validation_result.is_valid:
            error_messages = [error.error_message for error in validation_result.errors]
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed: {'; '.join(error_messages)}"
            )
        
        # Executa contrato
        executor = await get_contract_executor()
        result = await executor.execute_contract(request, user_id)
        
        # Headers de resposta
        response.headers["X-Contract-Type"] = request.contract_type.value
        response.headers["X-Execution-Status"] = result.status.value
        response.headers["X-Execution-ID"] = result.execution_id
        
        if result.execution_time_ms:
            response.headers["X-Execution-Time-MS"] = str(int(result.execution_time_ms))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Contract execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Contract execution failed: {str(e)}"
        )


@router.post("/batch-execute", response_model=BatchExecuteResponse)
async def batch_execute_contracts(
    request: BatchExecuteRequest,
    response: Response,
    user_id: Optional[str] = Query(None, description="User ID for tracking")
) -> BatchExecuteResponse:
    """
    Executa múltiplos contratos em lote.
    
    Endpoint migrado com execução paralela opcional e controle de erros.
    """
    try:
        if not request.contracts:
            raise HTTPException(status_code=400, detail="No contracts provided")
        
        if len(request.contracts) > 50:
            raise HTTPException(status_code=400, detail="Too many contracts (max 50)")
        
        # Executa batch
        executor = await get_contract_executor()
        result = await executor.execute_batch(request, user_id)
        
        # Headers de resposta
        response.headers["X-Batch-Total"] = str(result.total_count)
        response.headers["X-Batch-Success"] = str(result.success_count)
        response.headers["X-Batch-Errors"] = str(result.error_count)
        response.headers["X-Batch-ID"] = result.batch_id
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch execution failed: {str(e)}"
        )


@router.get("/available", response_model=List[AvailableContract])
async def get_available_contracts() -> List[AvailableContract]:
    """
    Lista contratos disponíveis.
    
    Endpoint migrado com contratos expandidos incluindo biomateriais.
    """
    try:
        executor = await get_contract_executor()
        contracts_data = executor.get_available_contracts()
        
        available_contracts = []
        for contract in contracts_data:
            available_contracts.append(AvailableContract(
                type=ContractType(contract['type']),
                name=contract['name'],
                description=contract['description'],
                version=contract.get('version', '1.0.0'),
                contract_schema=contract['schema'],
                output_schema=contract.get('output_schema', {}),
                example_input=_get_contract_example(contract['type']),
                supported_features=contract.get('supported_features', []),
                resource_requirements=contract.get('resource_requirements', {})
            ))
        
        return available_contracts
        
    except Exception as e:
        logger.error(f"Failed to get available contracts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve available contracts"
        )


@router.get("/schema/{contract_type}")
async def get_contract_schema(contract_type: str) -> Dict[str, Any]:
    """
    Obtém schema de entrada para tipo de contrato.
    
    Endpoint migrado com schemas expandidos.
    """
    try:
        # Converte string para enum
        try:
            contract_type_enum = ContractType(contract_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown contract type: {contract_type}. "
                f"Available types: {[t.value for t in ContractType]}"
            )
        
        executor = await get_contract_executor()
        schema = executor.get_contract_schema(contract_type_enum)
        
        if not schema:
            raise HTTPException(
                status_code=404,
                detail=f"Schema not found for contract type: {contract_type}"
            )
        
        return schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get contract schema: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve contract schema"
        )


@router.get("/history", response_model=List[ExecutionHistory])
async def get_execution_history(
    contract_type: Optional[str] = Query(None, description="Filter by contract type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
) -> List[ExecutionHistory]:
    """
    Obtém histórico de execuções.
    
    Endpoint migrado com filtros expandidos.
    """
    try:
        executor = await get_contract_executor()
        history = executor.get_execution_history(limit)
        
        # Filtra por tipo de contrato se especificado
        if contract_type:
            try:
                contract_type_enum = ContractType(contract_type)
                history = [h for h in history if h.contract_type == contract_type_enum]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid contract type: {contract_type}"
                )
        
        # Filtra por user_id se especificado
        if user_id:
            history = [h for h in history if h.user_id == user_id]
        
        return history
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve execution history"
        )


@router.get("/examples/{contract_type}")
async def get_contract_examples(contract_type: str) -> Dict[str, Any]:
    """
    Obtém exemplos de entrada para tipo de contrato.
    
    Endpoint migrado com exemplos expandidos para todos os contratos.
    """
    try:
        contract_type_enum = ContractType(contract_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown contract type: {contract_type}"
        )
    
    example = _get_contract_example(contract_type)
    if not example:
        raise HTTPException(
            status_code=404,
            detail=f"No examples available for contract type: {contract_type}"
        )
    
    return example


@router.get("/health")
async def contracts_health() -> Dict[str, Any]:
    """
    Health check do sistema Score Contracts.
    
    Endpoint migrado com verificações expandidas.
    """
    try:
        executor = await get_contract_executor()
        health_status = executor.get_system_health()
        sandbox_manager = get_sandbox_manager()
        
        return {
            "status": health_status.status,
            "message": "Score Contracts system operational",
            "timestamp": health_status.timestamp.isoformat(),
            "available_contracts": len(health_status.available_contracts),
            "contract_types": [ct.value for ct in health_status.available_contracts],
            "active_executions": health_status.active_executions,
            "queued_executions": health_status.queued_executions,
            "sandbox_status": health_status.sandbox_status,
            "resource_usage": health_status.resource_usage,
            "uptime_seconds": health_status.uptime_seconds,
            "security_info": sandbox_manager.get_security_info()
        }
        
    except Exception as e:
        logger.error(f"Contracts health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Score contracts error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "available_contracts": 0,
            "contract_types": [],
            "error": str(e)
        }


# ============================================================================
# SPECIALIZED ENDPOINTS - NOVOS PARA BIOMATERIAIS
# ============================================================================

@router.post("/biomaterials/scaffold", response_model=ContractExecutionResponse)
async def analyze_biomaterials_scaffold(
    scaffold_data: BiomaterialsContract,
    response: Response,
    user_id: Optional[str] = Query(None, description="User ID for tracking")
) -> ContractExecutionResponse:
    """
    Análise especializada de scaffold biomaterial.
    
    Endpoint novo para análise completa de scaffolds com métricas KEC.
    """
    try:
        # Converte para request padrão
        request = ContractExecutionRequest(
            contract_type=ContractType.BIOMATERIALS_SCAFFOLD,
            data=scaffold_data.dict(),
            timeout_seconds=60.0  # Análise mais complexa
        )
        
        return await execute_contract(request, response, user_id)
        
    except Exception as e:
        logger.error(f"Biomaterials scaffold analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scaffold analysis failed: {str(e)}"
        )


@router.post("/network/topology", response_model=ContractExecutionResponse)
async def analyze_network_topology(
    network_data: NetworkAnalysisContract,
    response: Response,
    user_id: Optional[str] = Query(None, description="User ID for tracking")
) -> ContractExecutionResponse:
    """
    Análise topológica de rede complexa.
    
    Endpoint novo para análise de redes com métricas topológicas.
    """
    try:
        # Converte para request padrão
        request = ContractExecutionRequest(
            contract_type=ContractType.NETWORK_TOPOLOGY,
            data=network_data.dict(),
            timeout_seconds=45.0
        )
        
        return await execute_contract(request, response, user_id)
        
    except Exception as e:
        logger.error(f"Network topology analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network analysis failed: {str(e)}"
        )


@router.post("/spectral/analysis", response_model=ContractExecutionResponse)
async def analyze_spectral_properties(
    spectral_data: SpectralAnalysisContract,
    response: Response,
    user_id: Optional[str] = Query(None, description="User ID for tracking")
) -> ContractExecutionResponse:
    """
    Análise espectral de grafo.
    
    Endpoint novo para análise espectral com eigenvalues e spectral gap.
    """
    try:
        # Converte para request padrão
        request = ContractExecutionRequest(
            contract_type=ContractType.SPECTRAL_ANALYSIS,
            data=spectral_data.dict(),
            timeout_seconds=50.0
        )
        
        return await execute_contract(request, response, user_id)
        
    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Spectral analysis failed: {str(e)}"
        )


# ============================================================================
# VALIDATION AND SECURITY ENDPOINTS
# ============================================================================

@router.post("/validate", response_model=ContractValidationResult)
async def validate_contract_request(
    request: ContractExecutionRequest
) -> ContractValidationResult:
    """
    Validação pré-execução de contrato.
    
    Endpoint novo para validação robusta antes da execução.
    """
    try:
        validator = get_contract_validator()
        validation_result = validator.validate_contract_request(request)
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Contract validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


@router.get("/security/policy")
async def get_security_policy() -> Dict[str, Any]:
    """
    Obtém política de segurança atual.
    
    Endpoint novo para transparência das configurações de segurança.
    """
    try:
        sandbox_manager = get_sandbox_manager()
        security_info = sandbox_manager.get_security_info()
        
        return {
            "security_policy": security_info,
            "allowed_contract_types": [ct.value for ct in ContractType],
            "security_levels": [sl.value for sl in SecurityLevel],
            "validation_enabled": True,
            "sandbox_isolation": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get security policy: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve security policy"
        )


@router.get("/performance", response_model=Dict[str, ContractPerformanceMetrics])
async def get_performance_metrics() -> Dict[str, ContractPerformanceMetrics]:
    """
    Obtém métricas de performance dos contratos.
    
    Endpoint novo para monitoramento de performance.
    """
    try:
        executor = await get_contract_executor()
        metrics = executor.get_performance_metrics()
        
        # Converte keys de enum para string
        return {ct.value: metrics[ct] for ct in metrics}
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance metrics"
        )


# ============================================================================
# LEGACY COMPATIBILITY ENDPOINTS
# ============================================================================

@router.post("/execute-legacy")
async def execute_contract_legacy(
    contract_type: str,
    data: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = 30.0
) -> Dict[str, Any]:
    """
    Endpoint legacy para compatibilidade com sistema anterior.
    
    Mantém API original para não quebrar integrações existentes.
    """
    try:
        # Converte para novo formato
        try:
            contract_type_enum = ContractType(contract_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown contract type: {contract_type}"
            )
        
        request = ContractExecutionRequest(
            contract_type=contract_type_enum,
            data=data,
            parameters=parameters,
            timeout_seconds=timeout_seconds
        )
        
        # Executa usando novo sistema
        response_obj = Response()
        result = await execute_contract(request, response_obj)
        
        # Converte para formato legacy
        return {
            "execution_id": result.execution_id,
            "contract_type": result.contract_type.value,
            "status": result.status.value,
            "score": result.score,
            "confidence": result.confidence,
            "metadata": result.metadata,
            "error_message": result.error_message,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy contract execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Contract execution failed: {str(e)}"
        )


# ============================================================================
# MATHEMATICAL CONTRACTS DIRECT ACCESS
# ============================================================================

@router.post("/math/delta-kec", response_model=ContractAPIResponse)
async def execute_delta_kec_direct(
    kec_data: DeltaKECContract,
    parameters: Optional[Dict[str, Any]] = None
) -> ContractAPIResponse:
    """
    Execução direta do Delta KEC v1.
    
    Bypass do sandbox para execução rápida do contrato core do mestrado.
    """
    try:
        result = execute_mathematical_contract(
            "delta_kec_v1",
            kec_data.dict(),
            parameters
        )
        
        return ContractAPIResponse(
            success=True,
            data=result,
            metadata={"execution_method": "direct_mathematical"}
        )
        
    except Exception as e:
        logger.error(f"Direct Delta KEC execution failed: {e}")
        return ContractAPIResponse(
            success=False,
            error=f"Delta KEC execution failed: {str(e)}",
            metadata={"execution_method": "direct_mathematical"}
        )


@router.post("/math/batch", response_model=ContractAPIResponse)
async def execute_mathematical_batch(
    contracts: List[Dict[str, Any]],
    parallel: bool = Query(False, description="Execute in parallel")
) -> ContractAPIResponse:
    """
    Execução em lote de contratos matemáticos diretos.
    
    Endpoint para execução rápida sem sandbox para análises matemáticas.
    """
    try:
        if len(contracts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many contracts for mathematical batch (max 100)"
            )
        
        results = []
        
        if parallel:
            import asyncio
            tasks = []
            for contract in contracts:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        execute_mathematical_contract,
                        contract.get('type'),
                        contract.get('data', {}),
                        contract.get('parameters')
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processa exceções
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    results[i] = {
                        'error': str(result),
                        'contract_index': i
                    }
        else:
            # Execução sequencial
            for i, contract in enumerate(contracts):
                try:
                    result = execute_mathematical_contract(
                        contract.get('type'),
                        contract.get('data', {}),
                        contract.get('parameters')
                    )
                    results.append(result)
                except Exception as e:
                    results.append({
                        'error': str(e),
                        'contract_index': i
                    })
        
        success_count = sum(1 for r in results if 'error' not in r)
        
        return ContractAPIResponse(
            success=True,
            data={
                'results': results,
                'total_count': len(contracts),
                'success_count': success_count,
                'error_count': len(contracts) - success_count
            },
            metadata={
                'execution_method': 'mathematical_batch',
                'parallel_execution': parallel
            }
        )
        
    except Exception as e:
        logger.error(f"Mathematical batch execution failed: {e}")
        return ContractAPIResponse(
            success=False,
            error=f"Batch execution failed: {str(e)}",
            metadata={'execution_method': 'mathematical_batch'}
        )


# ============================================================================
# MONITORING AND ADMIN ENDPOINTS
# ============================================================================

@router.get("/admin/active-executions")
async def get_active_executions() -> Dict[str, Any]:
    """
    Lista execuções ativas no sistema.
    
    Endpoint administrativo para monitoramento.
    """
    try:
        sandbox_manager = get_sandbox_manager()
        active_executions = sandbox_manager.get_active_executions()
        
        return {
            "active_executions": active_executions,
            "count": len(active_executions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active executions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve active executions"
        )


@router.post("/admin/cancel-execution/{execution_id}")
async def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """
    Cancela execução ativa.
    
    Endpoint administrativo para cancelamento de execuções.
    """
    try:
        sandbox_manager = get_sandbox_manager()
        cancelled = sandbox_manager.cancel_execution(execution_id)
        
        if cancelled:
            return {
                "success": True,
                "message": f"Execution {execution_id} cancelled",
                "execution_id": execution_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found or already completed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel execution: {str(e)}"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_contract_example(contract_type: str) -> Optional[Dict[str, Any]]:
    """Obtém exemplo de entrada para tipo de contrato."""
    
    examples = {
        "delta_kec_v1": {
            "data": {
                "source_entropy": 8.5,
                "target_entropy": 7.2,
                "mutual_information": 3.1
            },
            "parameters": {
                "alpha": 0.7,
                "beta": 0.3
            }
        },
        "zuco_reading_v1": {
            "data": {
                "eeg_features": {
                    "theta_power": 12.5,
                    "alpha_power": 8.3,
                    "beta_power": 15.7,
                    "gamma_power": 9.2
                },
                "eye_tracking_features": {
                    "avg_fixation_duration": 245.0,
                    "avg_saccade_velocity": 180.0,
                    "regression_rate": 0.15,
                    "reading_speed_wpm": 220.0
                }
            },
            "parameters": {
                "eeg_weight": 0.6,
                "et_weight": 0.4
            }
        },
        "editorial_v1": {
            "data": {
                "text_metrics": {
                    "readability_score": 85.0,
                    "grammar_score": 92.0,
                    "vocabulary_diversity": 0.75,
                    "coherence_score": 88.0,
                    "originality_score": 78.0
                }
            },
            "parameters": {
                "weights": {
                    "readability": 0.2,
                    "grammar": 0.25,
                    "vocabulary": 0.15,
                    "coherence": 0.25,
                    "originality": 0.15
                }
            }
        },
        "biomaterials_scaffold": {
            "data": {
                "scaffold_structure": {
                    "porosity": 0.65,
                    "connectivity": 0.8,
                    "pore_size_distribution": [10, 15, 25, 30, 20],
                    "surface_area_ratio": 1.5
                },
                "material_properties": {
                    "young_modulus": 1500.0,
                    "tensile_strength": 75.0,
                    "biocompatibility_index": 0.85,
                    "degradation_rate": 0.05
                },
                "pore_network": {
                    "clustering_coefficient": 0.7,
                    "average_path_length": 2.5
                }
            },
            "parameters": {
                "analysis_depth": "full",
                "include_mechanical": True
            }
        },
        "network_topology": {
            "data": {
                "adjacency_matrix": [
                    [0, 1, 1, 0],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [0, 1, 1, 0]
                ],
                "node_attributes": {
                    "0": {"type": "hub"},
                    "1": {"type": "connector"},
                    "2": {"type": "connector"},
                    "3": {"type": "peripheral"}
                },
                "analysis_type": "full"
            },
            "parameters": {
                "centrality_measures": ["degree", "betweenness"],
                "community_detection": True
            }
        },
        "spectral_analysis": {
            "data": {
                "graph_laplacian": [
                    [2, -1, -1, 0],
                    [-1, 3, -1, -1],
                    [-1, -1, 3, -1],
                    [0, -1, -1, 2]
                ],
                "eigenvalue_analysis": True,
                "community_detection": True
            },
            "parameters": {
                "precision": "high",
                "include_eigenvectors": False
            }
        }
    }
    
    return examples.get(contract_type)


def _get_contract_description(contract_type: str) -> str:
    """Obtém descrição humanizada do contrato."""
    
    descriptions = {
        "delta_kec_v1": "Delta Knowledge Exchange Coefficient - measures knowledge transfer efficiency between systems",
        "zuco_reading_v1": "ZuCo Reading Score - EEG and eye-tracking based reading comprehension analysis",
        "editorial_v1": "Editorial Quality Score - comprehensive text quality assessment with multiple dimensions",
        "biomaterials_scaffold": "Biomaterials Scaffold Analysis - mathematical analysis of scaffold structures with KEC metrics",
        "network_topology": "Network Topology Analysis - comprehensive topological analysis of complex networks",
        "spectral_analysis": "Spectral Graph Analysis - eigenvalue and spectral gap analysis of graph structures"
    }
    
    return descriptions.get(contract_type, f"Score contract: {contract_type}")


# ============================================================================
# NOTE: Exception handlers são registrados na aplicação principal (main.py)
# APIRouter não suporta exception_handler decorator
# ============================================================================