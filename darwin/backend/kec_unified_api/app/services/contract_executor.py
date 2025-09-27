"""Contract Executor - Pipeline de Execução Principal

Sistema principal de execução de contratos que orquestra sandbox, validação,
contratos matemáticos e logging. Pipeline completo para execução segura.

Feature crítica #5 para mestrado - Executor de contratos com pipeline avançado.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..models.contract_models import (
    ContractType,
    ContractExecutionRequest,
    ContractExecutionResponse,
    BatchExecuteRequest,
    BatchExecuteResponse,
    ExecutionStatus,
    ContractPriority,
    SecurityLevel,
    ExecutionHistory,
    ContractPerformanceMetrics,
    SystemHealthStatus
)
from .sandbox_manager import (
    SandboxManager,
    get_sandbox_manager,
    create_default_security_policy,
    create_high_security_policy,
    SandboxSecurityError,
    SandboxTimeoutError,
    SandboxResourceError,
    SandboxValidationError
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION EXCEPTIONS
# ============================================================================

class ContractExecutionError(Exception):
    """Erro base de execução de contrato."""
    pass


class ContractNotFoundError(ContractExecutionError):
    """Contrato não encontrado."""
    pass


class ContractValidationError(ContractExecutionError):
    """Erro de validação de contrato."""
    pass


class ExecutionQueueFullError(ContractExecutionError):
    """Fila de execução cheia."""
    pass


# ============================================================================
# EXECUTION CONTEXT
# ============================================================================

class ExecutionContext:
    """Contexto de execução de contrato."""
    
    def __init__(
        self,
        execution_id: str,
        contract_type: ContractType,
        request_data: ContractExecutionRequest,
        user_id: Optional[str] = None
    ):
        self.execution_id = execution_id
        self.contract_type = contract_type
        self.request_data = request_data
        self.user_id = user_id
        self.start_time = time.time()
        self.status = ExecutionStatus.PENDING
        self.metadata: Dict[str, Any] = {}
        self.result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        self.validation_errors: List[str] = []
        
    @property
    def execution_time_ms(self) -> float:
        """Tempo de execução em milissegundos."""
        return (time.time() - self.start_time) * 1000
    
    def to_response(self) -> ContractExecutionResponse:
        """Converte contexto para response."""
        return ContractExecutionResponse(
            execution_id=self.execution_id,
            contract_type=self.contract_type,
            status=self.status,
            score=self.result.get('score') if self.result else None,
            confidence=self.result.get('confidence') if self.result else None,
            metadata=self.result.get('metadata', {}) if self.result else None,
            error_message=self.error_message,
            execution_time_ms=self.execution_time_ms,
            timestamp=datetime.now(),
            sandbox_info=self.metadata.get('sandbox_info')
        )


# ============================================================================
# CONTRACT REGISTRY
# ============================================================================

class ContractRegistry:
    """Registro de contratos disponíveis."""
    
    def __init__(self):
        self._contracts: Dict[ContractType, Dict[str, Any]] = {}
        self._initialize_default_contracts()
    
    def _initialize_default_contracts(self):
        """Inicializa contratos padrão."""
        
        # Delta KEC v1
        self._contracts[ContractType.DELTA_KEC_V1] = {
            'name': 'Delta Knowledge Exchange Coefficient v1',
            'description': 'Calcula coeficiente de transferência de conhecimento entre sistemas',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'source_entropy': {'type': 'number', 'minimum': 0},
                    'target_entropy': {'type': 'number', 'minimum': 0},
                    'mutual_information': {'type': 'number', 'minimum': 0}
                },
                'required': ['source_entropy', 'target_entropy']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number'},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'delta_kec': {'type': 'number'},
                    'normalized_score': {'type': 'number'}
                }
            },
            'contract_code': self._get_delta_kec_code(),
            'supported_features': ['entropy_analysis', 'information_theory'],
            'resource_requirements': {
                'cpu_intensive': False,
                'memory_intensive': False,
                'network_access': False
            }
        }
        
        # ZuCo Reading v1
        self._contracts[ContractType.ZUCO_READING_V1] = {
            'name': 'ZuCo Reading Comprehension Analysis v1',
            'description': 'Análise de compreensão de leitura baseada em EEG e eye tracking',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'eeg_features': {
                        'type': 'object',
                        'properties': {
                            'theta_power': {'type': 'number'},
                            'alpha_power': {'type': 'number'},
                            'beta_power': {'type': 'number'},
                            'gamma_power': {'type': 'number'}
                        },
                        'required': ['theta_power', 'alpha_power', 'beta_power', 'gamma_power']
                    },
                    'eye_tracking_features': {
                        'type': 'object',
                        'properties': {
                            'avg_fixation_duration': {'type': 'number'},
                            'avg_saccade_velocity': {'type': 'number'},
                            'regression_rate': {'type': 'number'},
                            'reading_speed_wpm': {'type': 'number'}
                        }
                    }
                },
                'required': ['eeg_features', 'eye_tracking_features']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'reading_difficulty': {'type': 'string'},
                    'cognitive_load': {'type': 'number'}
                }
            },
            'contract_code': self._get_zuco_reading_code(),
            'supported_features': ['eeg_analysis', 'eye_tracking', 'cognitive_assessment'],
            'resource_requirements': {
                'cpu_intensive': True,
                'memory_intensive': False,
                'network_access': False
            }
        }
        
        # Editorial v1
        self._contracts[ContractType.EDITORIAL_V1] = {
            'name': 'Editorial Quality Assessment v1',
            'description': 'Avaliação abrangente de qualidade editorial de textos',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'text_metrics': {
                        'type': 'object',
                        'properties': {
                            'readability_score': {'type': 'number'},
                            'grammar_score': {'type': 'number'},
                            'vocabulary_diversity': {'type': 'number'},
                            'coherence_score': {'type': 'number'},
                            'originality_score': {'type': 'number'}
                        }
                    }
                },
                'required': ['text_metrics']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number', 'minimum': 0, 'maximum': 100},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'quality_category': {'type': 'string'},
                    'recommendations': {'type': 'array'}
                }
            },
            'contract_code': self._get_editorial_code(),
            'supported_features': ['text_analysis', 'quality_assessment'],
            'resource_requirements': {
                'cpu_intensive': False,
                'memory_intensive': False,
                'network_access': False
            }
        }
        
        # Biomaterials Scaffold
        self._contracts[ContractType.BIOMATERIALS_SCAFFOLD] = {
            'name': 'Biomaterials Scaffold Analysis',
            'description': 'Análise matemática completa de scaffolds biomateriais',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'scaffold_structure': {'type': 'object'},
                    'material_properties': {'type': 'object'},
                    'pore_network': {'type': 'object'}
                },
                'required': ['scaffold_structure', 'material_properties']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number'},
                    'kec_metrics': {'type': 'object'},
                    'biocompatibility_index': {'type': 'number'},
                    'mechanical_properties': {'type': 'object'}
                }
            },
            'contract_code': self._get_biomaterials_code(),
            'supported_features': ['network_analysis', 'kec_metrics', 'biocompatibility'],
            'resource_requirements': {
                'cpu_intensive': True,
                'memory_intensive': True,
                'network_access': False
            }
        }
        
        # Network Topology
        self._contracts[ContractType.NETWORK_TOPOLOGY] = {
            'name': 'Network Topology Analysis',
            'description': 'Análise topológica avançada de redes complexas',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'adjacency_matrix': {'type': 'array'},
                    'node_attributes': {'type': 'object'},
                    'analysis_type': {'type': 'string'}
                },
                'required': ['adjacency_matrix']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number'},
                    'network_metrics': {'type': 'object'},
                    'community_structure': {'type': 'object'},
                    'centrality_measures': {'type': 'object'}
                }
            },
            'contract_code': self._get_network_topology_code(),
            'supported_features': ['graph_analysis', 'centrality', 'community_detection'],
            'resource_requirements': {
                'cpu_intensive': True,
                'memory_intensive': True,
                'network_access': False
            }
        }
        
        # Spectral Analysis
        self._contracts[ContractType.SPECTRAL_ANALYSIS] = {
            'name': 'Spectral Analysis',
            'description': 'Análise espectral avançada de grafos e redes',
            'version': '1.0.0',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'graph_laplacian': {'type': 'array'},
                    'eigenvalue_analysis': {'type': 'boolean'},
                    'community_detection': {'type': 'boolean'}
                },
                'required': ['graph_laplacian']
            },
            'output_schema': {
                'type': 'object',
                'properties': {
                    'score': {'type': 'number'},
                    'spectral_features': {'type': 'object'},
                    'eigenvalues': {'type': 'array'},
                    'spectral_gap': {'type': 'number'}
                }
            },
            'contract_code': self._get_spectral_analysis_code(),
            'supported_features': ['eigenvalue_analysis', 'spectral_clustering'],
            'resource_requirements': {
                'cpu_intensive': True,
                'memory_intensive': True,
                'network_access': False
            }
        }
    
    def get_contract_info(self, contract_type: ContractType) -> Optional[Dict[str, Any]]:
        """Obtém informações do contrato."""
        return self._contracts.get(contract_type)
    
    def get_available_contracts(self) -> List[Dict[str, Any]]:
        """Lista todos os contratos disponíveis."""
        return [
            {
                'type': contract_type.value,
                'name': info['name'],
                'description': info['description'],
                'version': info['version'],
                'schema': info['input_schema'],
                'output_schema': info['output_schema'],
                'supported_features': info['supported_features']
            }
            for contract_type, info in self._contracts.items()
        ]
    
    def get_contract_code(self, contract_type: ContractType) -> Optional[str]:
        """Obtém código do contrato."""
        info = self._contracts.get(contract_type)
        return info['contract_code'] if info else None
    
    # Métodos de geração de código de contratos
    
    def _get_delta_kec_code(self) -> str:
        """Código do contrato Delta KEC v1."""
        return '''
import math

def execute_contract(input_data, parameters=None):
    """Contrato Delta KEC v1 - Core do mestrado."""
    
    source_entropy = float(input_data.get('source_entropy', 0))
    target_entropy = float(input_data.get('target_entropy', 0))
    mutual_info = float(input_data.get('mutual_information', 0))
    
    # Parâmetros
    alpha = parameters.get('alpha', 0.7) if parameters else 0.7
    beta = parameters.get('beta', 0.3) if parameters else 0.3
    
    # Calcula Delta KEC
    delta_kec = target_entropy - source_entropy
    
    # Normaliza considerando informação mútua
    if mutual_info > 0:
        normalized_delta = delta_kec / (1 + mutual_info)
    else:
        normalized_delta = delta_kec
    
    # Score final ponderado
    score = alpha * normalized_delta + beta * mutual_info
    
    # Confidence baseada na magnitude dos valores
    total_entropy = source_entropy + target_entropy
    confidence = min(0.95, total_entropy / (total_entropy + 1)) if total_entropy > 0 else 0.1
    
    result = {
        'score': float(score),
        'confidence': float(confidence),
        'delta_kec': float(delta_kec),
        'normalized_score': float(normalized_delta),
        'metadata': {
            'alpha': alpha,
            'beta': beta,
            'total_entropy': total_entropy
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''
    
    def _get_zuco_reading_code(self) -> str:
        """Código do contrato ZuCo Reading v1."""
        return '''
import math

def execute_contract(input_data, parameters=None):
    """Contrato ZuCo Reading v1 - EEG + Eye Tracking."""
    
    eeg = input_data['eeg_features']
    et = input_data['eye_tracking_features']
    
    # Weights
    eeg_weight = parameters.get('eeg_weight', 0.6) if parameters else 0.6
    et_weight = parameters.get('et_weight', 0.4) if parameters else 0.4
    
    # EEG Score (normalizado)
    eeg_total = eeg['theta_power'] + eeg['alpha_power'] + eeg['beta_power'] + eeg['gamma_power']
    eeg_balance = 1.0 - abs(0.25 - eeg['alpha_power']/eeg_total) if eeg_total > 0 else 0
    
    # Eye Tracking Score
    fixation_score = min(1.0, et['avg_fixation_duration'] / 300.0)  # 300ms baseline
    saccade_score = min(1.0, et['avg_saccade_velocity'] / 200.0)   # 200°/s baseline
    regression_penalty = 1.0 - min(0.5, et.get('regression_rate', 0))
    
    et_score = (fixation_score + saccade_score + regression_penalty) / 3.0
    
    # Score final
    score = eeg_weight * eeg_balance + et_weight * et_score
    
    # Cognitive load estimation
    cognitive_load = (eeg['gamma_power'] / eeg_total) if eeg_total > 0 else 0.5
    
    # Reading difficulty
    if score > 0.8:
        difficulty = "easy"
    elif score > 0.6:
        difficulty = "moderate"
    elif score > 0.4:
        difficulty = "hard"
    else:
        difficulty = "very_hard"
    
    # Confidence
    confidence = min(0.95, score * 0.8 + 0.2)
    
    result = {
        'score': float(score),
        'confidence': float(confidence),
        'reading_difficulty': difficulty,
        'cognitive_load': float(cognitive_load),
        'metadata': {
            'eeg_score': float(eeg_balance),
            'et_score': float(et_score),
            'eeg_weight': eeg_weight,
            'et_weight': et_weight
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''
    
    def _get_editorial_code(self) -> str:
        """Código do contrato Editorial v1."""
        return '''
def execute_contract(input_data, parameters=None):
    """Contrato Editorial v1 - Análise de qualidade textual."""
    
    metrics = input_data['text_metrics']
    
    # Weights padrão
    default_weights = {
        'readability': 0.2,
        'grammar': 0.25,
        'vocabulary': 0.15,
        'coherence': 0.25,
        'originality': 0.15
    }
    
    weights = parameters.get('weights', default_weights) if parameters else default_weights
    
    # Normaliza métricas (assumindo escala 0-100)
    normalized_metrics = {
        'readability': metrics.get('readability_score', 0) / 100.0,
        'grammar': metrics.get('grammar_score', 0) / 100.0,
        'vocabulary': metrics.get('vocabulary_diversity', 0),  # Já 0-1
        'coherence': metrics.get('coherence_score', 0) / 100.0,
        'originality': metrics.get('originality_score', 0) / 100.0
    }
    
    # Score ponderado
    score = sum(normalized_metrics[key] * weights.get(key, 0) for key in normalized_metrics)
    score *= 100  # Converte para escala 0-100
    
    # Quality category
    if score >= 90:
        category = "excellent"
    elif score >= 80:
        category = "very_good"
    elif score >= 70:
        category = "good"
    elif score >= 60:
        category = "fair"
    else:
        category = "poor"
    
    # Recommendations
    recommendations = []
    if normalized_metrics['readability'] < 0.7:
        recommendations.append("Improve text readability")
    if normalized_metrics['grammar'] < 0.8:
        recommendations.append("Review grammar and style")
    if normalized_metrics['vocabulary'] < 0.6:
        recommendations.append("Increase vocabulary diversity")
    if normalized_metrics['coherence'] < 0.7:
        recommendations.append("Enhance text coherence")
    if normalized_metrics['originality'] < 0.5:
        recommendations.append("Add more original content")
    
    # Confidence baseada na variância das métricas
    values = list(normalized_metrics.values())
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    confidence = max(0.1, min(0.95, 1.0 - variance))
    
    result = {
        'score': float(score),
        'confidence': float(confidence),
        'quality_category': category,
        'recommendations': recommendations,
        'metadata': {
            'normalized_metrics': normalized_metrics,
            'weights_used': weights,
            'variance': float(variance)
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''
    
    def _get_biomaterials_code(self) -> str:
        """Código do contrato Biomaterials Scaffold."""
        return '''
import math

def execute_contract(input_data, parameters=None):
    """Contrato Biomaterials Scaffold - Análise de scaffolds."""
    
    structure = input_data['scaffold_structure']
    materials = input_data['material_properties']
    pore_network = input_data.get('pore_network', {})
    
    # Métricas KEC básicas
    porosity = structure.get('porosity', 0.5)
    connectivity = structure.get('connectivity', 0.7)
    pore_size_dist = structure.get('pore_size_distribution', [])
    
    # Propriedades do material
    young_modulus = materials.get('young_modulus', 1000)  # MPa
    tensile_strength = materials.get('tensile_strength', 50)  # MPa
    biocompatibility = materials.get('biocompatibility_index', 0.8)
    
    # KEC Metrics calculation (simplified)
    # Entropia espectral baseada na distribuição de poros
    if pore_size_dist:
        total_pores = sum(pore_size_dist)
        if total_pores > 0:
            probs = [p/total_pores for p in pore_size_dist if p > 0]
            spectral_entropy = -sum(p * math.log(p) for p in probs)
        else:
            spectral_entropy = 0
    else:
        spectral_entropy = porosity  # Approximation
    
    # Curvatura de Forman aproximada
    forman_curvature = connectivity * (1 - porosity)
    
    # Small-world propensity (simplified)
    clustering_coeff = pore_network.get('clustering_coefficient', connectivity)
    path_length = pore_network.get('average_path_length', 1/connectivity if connectivity > 0 else 10)
    swp = clustering_coeff / path_length if path_length > 0 else 0
    
    # Score final baseado em múltiplos critérios
    structural_score = (porosity * 0.3 + connectivity * 0.4 + spectral_entropy * 0.3)
    mechanical_score = min(1.0, (young_modulus / 2000.0 + tensile_strength / 100.0) / 2)
    bio_score = biocompatibility
    topology_score = min(1.0, (forman_curvature + swp) / 2)
    
    final_score = (structural_score * 0.3 + mechanical_score * 0.2 + 
                   bio_score * 0.3 + topology_score * 0.2)
    
    # Biocompatibility index
    bio_index = biocompatibility * structural_score
    
    # Confidence
    confidence = min(0.95, (structural_score + bio_score) / 2)
    
    result = {
        'score': float(final_score),
        'confidence': float(confidence),
        'kec_metrics': {
            'spectral_entropy': float(spectral_entropy),
            'forman_curvature': float(forman_curvature),
            'small_world_propensity': float(swp)
        },
        'biocompatibility_index': float(bio_index),
        'mechanical_properties': {
            'structural_score': float(structural_score),
            'mechanical_score': float(mechanical_score)
        },
        'metadata': {
            'porosity': porosity,
            'connectivity': connectivity,
            'topology_score': float(topology_score)
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''
    
    def _get_network_topology_code(self) -> str:
        """Código do contrato Network Topology."""
        return '''
import math

def execute_contract(input_data, parameters=None):
    """Contrato Network Topology - Análise topológica."""
    
    adj_matrix = input_data['adjacency_matrix']
    node_attrs = input_data.get('node_attributes', {})
    analysis_type = input_data.get('analysis_type', 'full')
    
    n = len(adj_matrix)
    if n == 0:
        return {'error': 'Empty adjacency matrix'}
    
    # Calcula métricas básicas da rede
    total_edges = sum(sum(row) for row in adj_matrix) / 2  # Para grafo não-direcionado
    density = (2 * total_edges) / (n * (n - 1)) if n > 1 else 0
    
    # Degree distribution
    degrees = [sum(row) for row in adj_matrix]
    avg_degree = sum(degrees) / n if n > 0 else 0
    max_degree = max(degrees) if degrees else 0
    
    # Clustering coefficient aproximado
    clustering_sum = 0
    for i in range(n):
        neighbors = [j for j in range(n) if adj_matrix[i][j] > 0 and i != j]
        if len(neighbors) > 1:
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = sum(
                adj_matrix[j][k] for j in neighbors for k in neighbors if j < k
            )
            clustering_sum += actual_edges / possible_edges if possible_edges > 0 else 0
    
    avg_clustering = clustering_sum / n if n > 0 else 0
    
    # Centrality measures (degree centrality)
    degree_centrality = [deg / (n - 1) for deg in degrees] if n > 1 else [0] * n
    
    # Network efficiency aproximada
    # Simplified: inverse of average degree
    efficiency = avg_degree / max_degree if max_degree > 0 else 0
    
    # Score baseado em múltiplas métricas
    connectivity_score = min(1.0, density * 2)  # Penaliza redes muito esparsas
    efficiency_score = efficiency
    clustering_score = avg_clustering
    
    final_score = (connectivity_score * 0.4 + efficiency_score * 0.3 + clustering_score * 0.3)
    
    # Detecção de comunidades simplificada (baseada em clustering)
    num_communities = max(1, int(n * (1 - avg_clustering) / 2))
    community_structure = {
        'num_communities': num_communities,
        'modularity': avg_clustering,  # Approximation
        'community_sizes': [n // num_communities] * num_communities
    }
    
    result = {
        'score': float(final_score),
        'confidence': 0.8,  # Fixed confidence for this simplified implementation
        'network_metrics': {
            'density': float(density),
            'average_degree': float(avg_degree),
            'clustering_coefficient': float(avg_clustering),
            'efficiency': float(efficiency),
            'num_nodes': n,
            'num_edges': int(total_edges)
        },
        'community_structure': community_structure,
        'centrality_measures': {
            'degree_centrality': [float(dc) for dc in degree_centrality[:5]],  # Top 5
            'max_degree_centrality': float(max(degree_centrality)) if degree_centrality else 0
        },
        'metadata': {
            'analysis_type': analysis_type,
            'connectivity_score': float(connectivity_score),
            'efficiency_score': float(efficiency_score),
            'clustering_score': float(clustering_score)
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''
    
    def _get_spectral_analysis_code(self) -> str:
        """Código do contrato Spectral Analysis."""
        return '''
import math

def execute_contract(input_data, parameters=None):
    """Contrato Spectral Analysis - Análise espectral."""
    
    laplacian = input_data['graph_laplacian']
    eigenvalue_analysis = input_data.get('eigenvalue_analysis', True)
    community_detection = input_data.get('community_detection', False)
    
    n = len(laplacian)
    if n == 0:
        return {'error': 'Empty Laplacian matrix'}
    
    # Simplified eigenvalue estimation
    # Real implementation would use numpy.linalg.eigvals
    # Here we approximate using matrix properties
    
    # Trace (sum of eigenvalues)
    trace = sum(laplacian[i][i] for i in range(n))
    
    # Approximate eigenvalues using Gershgorin circle theorem
    eigenvalue_estimates = []
    for i in range(n):
        center = laplacian[i][i]
        radius = sum(abs(laplacian[i][j]) for j in range(n) if i != j)
        eigenvalue_estimates.extend([center - radius, center + radius])
    
    eigenvalue_estimates.sort()
    
    # Spectral gap (difference between two smallest non-zero eigenvalues)
    # In real implementation: second smallest eigenvalue
    spectral_gap = eigenvalue_estimates[2] if len(eigenvalue_estimates) > 2 else 0
    
    # Spectral features
    spectral_radius = max(eigenvalue_estimates) if eigenvalue_estimates else 0
    algebraic_connectivity = spectral_gap  # Approximation
    
    # Score baseado na conectividade algébrica e gap espectral
    connectivity_score = min(1.0, algebraic_connectivity / (n / 4)) if n > 0 else 0
    gap_score = min(1.0, spectral_gap / spectral_radius) if spectral_radius > 0 else 0
    
    final_score = (connectivity_score * 0.6 + gap_score * 0.4)
    
    # Community detection based on spectral gap
    if community_detection:
        # Simplified: larger spectral gap suggests fewer communities
        estimated_communities = max(1, int(n / (spectral_gap + 1)))
    else:
        estimated_communities = None
    
    # Confidence baseada na magnitude do spectral gap
    confidence = min(0.95, spectral_gap / (spectral_radius + 0.1))
    
    result = {
        'score': float(final_score),
        'confidence': float(confidence),
        'spectral_features': {
            'spectral_gap': float(spectral_gap),
            'spectral_radius': float(spectral_radius),
            'algebraic_connectivity': float(algebraic_connectivity),
            'trace': float(trace)
        },
        'eigenvalues': [float(e) for e in eigenvalue_estimates[:10]],  # Top 10
        'metadata': {
            'matrix_size': n,
            'connectivity_score': float(connectivity_score),
            'gap_score': float(gap_score),
            'estimated_communities': estimated_communities
        }
    }
    
    return result

# Execução principal
result = execute_contract(input_data, input_data.get('parameters'))
'''


# ============================================================================
# EXECUTION QUEUE
# ============================================================================

class ExecutionQueue:
    """Fila de execução com prioridade para contratos."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Tuple[ContractPriority, float, ExecutionContext]] = []
        self.lock = asyncio.Lock()
        
    async def enqueue(self, context: ExecutionContext) -> None:
        """Adiciona contexto à fila."""
        async with self.lock:
            if len(self.queue) >= self.max_size:
                raise ExecutionQueueFullError("Execution queue is full")
            
            priority_value = self._get_priority_value(context.request_data.priority)
            self.queue.append((context.request_data.priority, time.time(), context))
            
            # Ordena por prioridade (maior prioridade primeiro) e timestamp
            self.queue.sort(key=lambda x: (self._get_priority_value(x[0]), x[1]))
    
    async def dequeue(self) -> Optional[ExecutionContext]:
        """Remove e retorna próximo contexto da fila."""
        async with self.lock:
            if not self.queue:
                return None
            return self.queue.pop(0)[2]
    
    async def size(self) -> int:
        """Retorna tamanho da fila."""
        async with self.lock:
            return len(self.queue)
    
    def _get_priority_value(self, priority: ContractPriority) -> int:
        """Converte prioridade em valor numérico."""
        priority_values = {
            ContractPriority.LOW: 1,
            ContractPriority.NORMAL: 2,
            ContractPriority.HIGH: 3,
            ContractPriority.URGENT: 4
        }
        return priority_values.get(priority, 2)


# ============================================================================
# CONTRACT EXECUTOR
# ============================================================================

class ContractExecutor:
    """Executor principal de contratos com pipeline completo."""
    
    def __init__(self, max_concurrent: int = 5, max_queue_size: int = 1000):
        self.max_concurrent = max_concurrent
        self.registry = ContractRegistry()
        self.sandbox_manager = get_sandbox_manager()
        self.execution_queue = ExecutionQueue(max_queue_size)
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionHistory] = []
        self.performance_metrics: Dict[ContractType, ContractPerformanceMetrics] = {}
        self.is_running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"ContractExecutor initialized with max_concurrent={max_concurrent}")
    
    async def start(self) -> None:
        """Inicia o executor."""
        if self.is_running:
            return
        
        self.is_running = True
        asyncio.create_task(self._execution_worker())
        logger.info("ContractExecutor started")
    
    async def stop(self) -> None:
        """Para o executor."""
        self.is_running = False
        
        # Aguarda execuções ativas terminarem
        while self.active_executions:
            await asyncio.sleep(0.1)
        
        logger.info("ContractExecutor stopped")
    
    async def execute_contract(
        self,
        request: ContractExecutionRequest,
        user_id: Optional[str] = None
    ) -> ContractExecutionResponse:
        """Executa contrato individual."""
        
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(execution_id, request.contract_type, request, user_id)
        
        try:
            # Valida contrato
            contract_info = self.registry.get_contract_info(request.contract_type)
            if not contract_info:
                raise ContractNotFoundError(f"Contract type not found: {request.contract_type}")
            
            # Enfileira para execução
            await self.execution_queue.enqueue(context)
            
            # Aguarda execução
            while context.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                await asyncio.sleep(0.1)
            
            return context.to_response()
            
        except Exception as e:
            context.status = ExecutionStatus.FAILED
            context.error_message = str(e)
            logger.error(f"Contract execution failed: {e}")
            return context.to_response()
    
    async def execute_batch(
        self,
        request: BatchExecuteRequest,
        user_id: Optional[str] = None
    ) -> BatchExecuteResponse:
        """Executa múltiplos contratos em lote."""
        
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        if request.parallel_execution:
            # Execução paralela
            tasks = [
                self.execute_contract(contract_req, user_id)
                for contract_req in request.contracts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processa resultados
            responses = []
            for result in results:
                if isinstance(result, Exception):
                    responses.append(ContractExecutionResponse(
                        execution_id=str(uuid.uuid4()),
                        contract_type=ContractType.DELTA_KEC_V1,  # Fallback
                        status=ExecutionStatus.FAILED,
                        error_message=str(result),
                        execution_time_ms=0.0,
                        timestamp=datetime.now()
                    ))
                else:
                    responses.append(result)
        else:
            # Execução sequencial
            responses = []
            for contract_req in request.contracts:
                try:
                    response = await self.execute_contract(contract_req, user_id)
                    responses.append(response)
                    
                    # Para na primeira falha se configurado
                    if request.stop_on_error and response.status == ExecutionStatus.FAILED:
                        break
                        
                except Exception as e:
                    error_response = ContractExecutionResponse(
                        execution_id=str(uuid.uuid4()),
                        contract_type=contract_req.contract_type,
                        status=ExecutionStatus.FAILED,
                        error_message=str(e),
                        execution_time_ms=0.0,
                        timestamp=datetime.now()
                    )
                    responses.append(error_response)
                    
                    if request.stop_on_error:
                        break
        
        # Calcula estatísticas do lote
        total_count = len(responses)
        success_count = sum(1 for r in responses if r.status == ExecutionStatus.COMPLETED)
        error_count = sum(1 for r in responses if r.status == ExecutionStatus.FAILED)
        timeout_count = sum(1 for r in responses if r.status == ExecutionStatus.TIMEOUT)
        
        total_execution_time = (time.time() - start_time) * 1000
        
        return BatchExecuteResponse(
            batch_id=batch_id,
            results=responses,
            total_count=total_count,
            success_count=success_count,
            error_count=error_count,
            timeout_count=timeout_count,
            total_execution_time_ms=total_execution_time,
            batch_metadata=request.batch_metadata
        )
    
    async def _execution_worker(self) -> None:
        """Worker que processa fila de execução."""
        while self.is_running:
            try:
                context = await self.execution_queue.dequeue()
                if context is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Executa com semáforo para limitar concorrência
                async with self._semaphore:
                    await self._execute_context(context)
                    
            except Exception as e:
                logger.error(f"Execution worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_context(self, context: ExecutionContext) -> None:
        """Executa contexto de contrato."""
        
        context.status = ExecutionStatus.RUNNING
        self.active_executions[context.execution_id] = context
        
        try:
            # Obtém código do contrato
            contract_code = self.registry.get_contract_code(context.contract_type)
            if not contract_code:
                raise ContractNotFoundError(f"Contract code not found: {context.contract_type}")
            
            # Executa no sandbox
            result = await self.sandbox_manager.execute_contract(
                contract_code=contract_code,
                input_data=context.request_data.data,
                contract_type=context.contract_type,
                execution_id=context.execution_id
            )
            
            context.result = result
            context.status = ExecutionStatus.COMPLETED
            
        except SandboxTimeoutError as e:
            context.status = ExecutionStatus.TIMEOUT
            context.error_message = str(e)
            
        except (SandboxSecurityError, SandboxValidationError) as e:
            context.status = ExecutionStatus.FAILED
            context.error_message = f"Security/Validation error: {e}"
            
        except Exception as e:
            context.status = ExecutionStatus.FAILED
            context.error_message = f"Execution error: {e}"
            logger.error(f"Contract execution failed: {e}")
            
        finally:
            # Remove das execuções ativas
            self.active_executions.pop(context.execution_id, None)
            
            # Registra no histórico
            self._record_execution_history(context)
            
            # Atualiza métricas de performance
            self._update_performance_metrics(context)
    
    def _record_execution_history(self, context: ExecutionContext) -> None:
        """Registra execução no histórico."""
        
        history_entry = ExecutionHistory(
            execution_id=context.execution_id,
            contract_type=context.contract_type,
            status=context.status,
            score=context.result.get('score') if context.result else None,
            execution_time_ms=context.execution_time_ms,
            timestamp=datetime.now(),
            user_id=context.user_id,
            resource_usage=context.metadata.get('resource_usage'),
            error_type=type(context.error_message).__name__ if context.error_message else None
        )
        
        self.execution_history.append(history_entry)
        
        # Limita histórico
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-5000:]
    
    def _update_performance_metrics(self, context: ExecutionContext) -> None:
        """Atualiza métricas de performance."""
        
        contract_type = context.contract_type
        
        if contract_type not in self.performance_metrics:
            self.performance_metrics[contract_type] = ContractPerformanceMetrics(
                contract_type=contract_type,
                avg_execution_time_ms=0.0,
                success_rate=0.0,
                error_rate=0.0,
                timeout_rate=0.0,
                avg_memory_usage_mb=0.0,
                avg_cpu_usage_percent=0.0,
                total_executions=0,
                period_start=datetime.now(),
                period_end=datetime.now()
            )
        
        metrics = self.performance_metrics[contract_type]
        
        # Atualiza estatísticas incrementalmente
        total_prev = metrics.total_executions
        total_new = total_prev + 1
        
        # Média ponderada do tempo de execução
        new_exec_time = context.execution_time_ms
        metrics.avg_execution_time_ms = (
            (metrics.avg_execution_time_ms * total_prev + new_exec_time) / total_new
        )
        
        # Atualiza taxas
        success_count = int(metrics.success_rate * total_prev)
        error_count = int(metrics.error_rate * total_prev)
        timeout_count = int(metrics.timeout_rate * total_prev)
        
        if context.status == ExecutionStatus.COMPLETED:
            success_count += 1
        elif context.status == ExecutionStatus.TIMEOUT:
            timeout_count += 1
        else:
            error_count += 1
        
        metrics.success_rate = success_count / total_new
        metrics.error_rate = error_count / total_new
        metrics.timeout_rate = timeout_count / total_new
        metrics.total_executions = total_new
        metrics.period_end = datetime.now()
    
    # Métodos de consulta
    
    def get_available_contracts(self) -> List[Dict[str, Any]]:
        """Lista contratos disponíveis."""
        return self.registry.get_available_contracts()
    
    def get_contract_schema(self, contract_type: ContractType) -> Optional[Dict[str, Any]]:
        """Obtém schema de contrato."""
        info = self.registry.get_contract_info(contract_type)
        return info['input_schema'] if info else None
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionHistory]:
        """Obtém histórico de execuções."""
        return self.execution_history[-limit:]
    
    def get_performance_metrics(self) -> Dict[ContractType, ContractPerformanceMetrics]:
        """Obtém métricas de performance."""
        return self.performance_metrics.copy()
    
    def get_system_health(self) -> SystemHealthStatus:
        """Obtém status de saúde do sistema."""
        
        active_count = len(self.active_executions)
        queue_size = asyncio.create_task(self.execution_queue.size())
        
        # Calcula uso de recursos aproximado
        cpu_usage = min(100.0, (active_count / self.max_concurrent) * 100)
        memory_usage = active_count * 50.0  # Aproximação: 50MB por execução
        
        return SystemHealthStatus(
            status="healthy" if active_count < self.max_concurrent else "busy",
            timestamp=datetime.now(),
            available_contracts=[ct for ct in ContractType],
            sandbox_status="operational",
            active_executions=active_count,
            queued_executions=0,  # Seria await queue_size se fosse async
            resource_usage={
                'cpu_percent': cpu_usage,
                'memory_mb': memory_usage,
                'concurrent_limit': self.max_concurrent
            },
            last_error=None,
            uptime_seconds=time.time()  # Simplified
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# Instância global do executor
_contract_executor: Optional[ContractExecutor] = None


async def get_contract_executor() -> ContractExecutor:
    """Obtém instância global do executor."""
    global _contract_executor
    
    if _contract_executor is None:
        _contract_executor = ContractExecutor()
        await _contract_executor.start()
    
    return _contract_executor


async def initialize_contract_executor(
    max_concurrent: int = 5,
    max_queue_size: int = 1000
) -> ContractExecutor:
    """Inicializa executor com configurações específicas."""
    global _contract_executor
    
    if _contract_executor and _contract_executor.is_running:
        await _contract_executor.stop()
    
    _contract_executor = ContractExecutor(max_concurrent, max_queue_size)
    await _contract_executor.start()
    
    return _contract_executor