"""Score Contracts Models

Modelos Pydantic completos para o sistema Score Contracts com sandbox execution seguro.
Migrado e expandido de backup_old_backends/kec_biomat_api/routers/score_contracts.py

Feature crítica #5 para mestrado - Sistema matemático avançado.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# ============================================================================
# CORE CONTRACT ENUMS
# ============================================================================

class ContractType(str, Enum):
    """Tipos de contratos disponíveis no sistema."""
    
    # Contratos originais migrados
    DELTA_KEC_V1 = "delta_kec_v1"
    ZUCO_READING_V1 = "zuco_reading_v1"
    EDITORIAL_V1 = "editorial_v1"
    
    # Novos contratos especializados para biomateriais
    BIOMATERIALS_SCAFFOLD = "biomaterials_scaffold"
    NETWORK_TOPOLOGY = "network_topology"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    
    # Contratos matemáticos avançados
    KEC_ENTROPY_ANALYSIS = "kec_entropy_analysis"
    FORMAN_CURVATURE = "forman_curvature"
    SMALL_WORLD_METRICS = "small_world_metrics"


class ExecutionStatus(str, Enum):
    """Status de execução de contratos."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ContractPriority(str, Enum):
    """Prioridade de execução de contratos."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class SecurityLevel(str, Enum):
    """Nível de segurança do sandbox."""
    
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    MAXIMUM = "maximum"


# ============================================================================
# CORE CONTRACT MODELS
# ============================================================================

class ContractExecutionRequest(BaseModel):
    """Request para execução de contrato individual."""
    
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    data: Dict[str, Any] = Field(..., description="Dados de entrada")
    parameters: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[float] = 30.0
    priority: ContractPriority = ContractPriority.NORMAL
    security_level: SecurityLevel = SecurityLevel.STANDARD
    metadata: Optional[Dict[str, Any]] = None


class ContractExecutionResponse(BaseModel):
    """Response da execução de contrato."""
    
    execution_id: str = Field(..., description="ID único da execução")
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    status: ExecutionStatus = Field(..., description="Status da execução")
    score: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: Optional[datetime] = None
    sandbox_info: Optional[Dict[str, Any]] = None


class BatchExecuteRequest(BaseModel):
    """Request para execução em lote de contratos."""
    
    contracts: List[ContractExecutionRequest] = Field(..., max_length=50, description="Lista de contratos")
    parallel_execution: bool = Field(False, description="Execução paralela")
    stop_on_error: bool = Field(False, description="Parar ao encontrar erro")
    batch_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados do lote")


class BatchExecuteResponse(BaseModel):
    """Response da execução em lote."""
    
    batch_id: str = Field(..., description="ID do lote")
    results: List[ContractExecutionResponse] = Field(..., description="Resultados individuais")
    total_count: int = Field(..., description="Total de contratos")
    success_count: int = Field(..., description="Contratos executados com sucesso")
    error_count: int = Field(..., description="Contratos com erro")
    timeout_count: int = Field(0, description="Contratos com timeout")
    total_execution_time_ms: float = Field(..., description="Tempo total de execução")
    batch_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados do lote")


class AvailableContract(BaseModel):
    """Informações de contrato disponível."""
    
    type: ContractType = Field(..., description="Tipo do contrato")
    name: str = Field(..., description="Nome do contrato")
    description: str = Field(..., description="Descrição detalhada")
    version: str = Field(..., description="Versão do contrato")
    contract_schema: Dict[str, Any] = Field(..., description="Schema de entrada")
    output_schema: Dict[str, Any] = Field(..., description="Schema de saída")
    example_input: Optional[Dict[str, Any]] = Field(None, description="Exemplo de entrada")
    supported_features: List[str] = Field(default_factory=list, description="Features suportadas")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Requisitos de recursos")


class ContractSchema(BaseModel):
    """Schema detalhado de um contrato."""
    
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    input_schema: Dict[str, Any] = Field(..., description="Schema JSON de entrada")
    output_schema: Dict[str, Any] = Field(..., description="Schema JSON de saída")
    parameter_schema: Optional[Dict[str, Any]] = Field(None, description="Schema dos parâmetros")
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Regras de validação")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Exemplos de uso")


class ExecutionHistory(BaseModel):
    """Histórico de execução de contratos."""
    
    execution_id: str = Field(..., description="ID da execução")
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    status: ExecutionStatus = Field(..., description="Status final")
    score: Optional[float] = Field(None, description="Score obtido")
    execution_time_ms: float = Field(..., description="Tempo de execução")
    timestamp: datetime = Field(..., description="Timestamp da execução")
    user_id: Optional[str] = Field(None, description="ID do usuário")
    resource_usage: Optional[Dict[str, Any]] = Field(None, description="Uso de recursos")
    error_type: Optional[str] = Field(None, description="Tipo do erro")


# ============================================================================
# SPECIALIZED CONTRACT MODELS
# ============================================================================

class DeltaKECContract(BaseModel):
    """Contrato Delta KEC v1 - Core do mestrado."""
    
    source_entropy: float = Field(..., description="Entropia da fonte")
    target_entropy: float = Field(..., description="Entropia do alvo")
    mutual_information: Optional[float] = Field(None, description="Informação mútua")
    temporal_window: Optional[float] = Field(None, description="Janela temporal")
    normalization_method: str = Field("standard", description="Método de normalização")
    
    @validator('source_entropy', 'target_entropy')
    def validate_entropy(cls, v):
        if v < 0:
            raise ValueError('Entropy values must be non-negative')
        return v


class ZuCoReadingContract(BaseModel):
    """Contrato ZuCo Reading v1 - EEG + Eye Tracking."""
    
    eeg_features: Dict[str, float] = Field(..., description="Features EEG")
    eye_tracking_features: Dict[str, float] = Field(..., description="Features eye tracking")
    text_features: Optional[Dict[str, Any]] = Field(None, description="Features do texto")
    participant_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados participante")
    
    @validator('eeg_features')
    def validate_eeg_features(cls, v):
        required_features = ['theta_power', 'alpha_power', 'beta_power', 'gamma_power']
        if not all(feature in v for feature in required_features):
            raise ValueError(f'EEG features must include: {required_features}')
        return v


class EditorialContract(BaseModel):
    """Contrato Editorial v1 - Análise de qualidade textual."""
    
    text_metrics: Dict[str, float] = Field(..., description="Métricas textuais")
    linguistic_features: Optional[Dict[str, Any]] = Field(None, description="Features linguísticas")
    style_analysis: Optional[Dict[str, Any]] = Field(None, description="Análise de estilo")
    domain_context: Optional[str] = Field(None, description="Contexto do domínio")


class BiomaterialsContract(BaseModel):
    """Contrato para análise de scaffolds biomateriais."""
    
    scaffold_structure: Dict[str, Any] = Field(..., description="Estrutura do scaffold")
    material_properties: Dict[str, float] = Field(..., description="Propriedades do material")
    pore_network: Optional[Dict[str, Any]] = Field(None, description="Rede de poros")
    mechanical_properties: Optional[Dict[str, float]] = Field(None, description="Propriedades mecânicas")
    biocompatibility_data: Optional[Dict[str, Any]] = Field(None, description="Dados biocompatibilidade")


class NetworkAnalysisContract(BaseModel):
    """Contrato para análise topológica de redes."""
    
    adjacency_matrix: List[List[float]] = Field(..., description="Matriz de adjacência")
    node_attributes: Optional[Dict[str, Any]] = Field(None, description="Atributos dos nós")
    edge_weights: Optional[Dict[str, float]] = Field(None, description="Pesos das arestas")
    analysis_type: str = Field("full", description="Tipo de análise")
    
    @validator('adjacency_matrix')
    def validate_adjacency_matrix(cls, v):
        if not v:
            raise ValueError('Adjacency matrix cannot be empty')
        n = len(v)
        if not all(len(row) == n for row in v):
            raise ValueError('Adjacency matrix must be square')
        return v


class SpectralAnalysisContract(BaseModel):
    """Contrato para análise espectral avançada."""
    
    graph_laplacian: List[List[float]] = Field(..., description="Laplaciano do grafo")
    spectral_features: Optional[Dict[str, float]] = Field(None, description="Features espectrais")
    eigenvalue_analysis: bool = Field(True, description="Análise de autovalores")
    community_detection: bool = Field(False, description="Detecção de comunidades")


# ============================================================================
# SECURITY MODELS
# ============================================================================

class SandboxConfig(BaseModel):
    """Configuração do sandbox de execução."""
    
    cpu_limit_percent: float = Field(10.0, ge=1.0, le=50.0, description="Limite de CPU %")
    memory_limit_mb: int = Field(512, ge=64, le=2048, description="Limite de memória MB")
    execution_timeout_seconds: float = Field(30.0, ge=1.0, le=300.0, description="Timeout execução")
    network_isolation: bool = Field(True, description="Isolamento de rede")
    file_system_access: SecurityLevel = Field(SecurityLevel.MINIMAL, description="Acesso filesystem")
    allowed_imports: List[str] = Field(default_factory=list, description="Imports permitidos")
    forbidden_operations: List[str] = Field(default_factory=list, description="Operações proibidas")


class ExecutionLimits(BaseModel):
    """Limites de execução de contratos."""
    
    max_memory_bytes: int = Field(512 * 1024 * 1024, description="Memória máxima")
    max_cpu_time_seconds: float = Field(30.0, description="Tempo CPU máximo")
    max_wall_time_seconds: float = Field(60.0, description="Tempo wall máximo")
    max_file_size_bytes: int = Field(10 * 1024 * 1024, description="Tamanho arquivo máximo")
    max_processes: int = Field(1, description="Número máximo de processos")
    max_threads: int = Field(4, description="Número máximo de threads")


class SecurityPolicy(BaseModel):
    """Política de segurança do sistema."""
    
    security_level: SecurityLevel = Field(..., description="Nível de segurança")
    sandbox_config: SandboxConfig = Field(..., description="Configuração sandbox")
    execution_limits: ExecutionLimits = Field(..., description="Limites de execução")
    audit_logging: bool = Field(True, description="Log de auditoria")
    encryption_required: bool = Field(False, description="Criptografia obrigatória")
    user_isolation: bool = Field(True, description="Isolamento por usuário")


class ResourceQuota(BaseModel):
    """Quotas de recursos por usuário/contrato."""
    
    daily_executions: int = Field(1000, description="Execuções diárias")
    hourly_executions: int = Field(100, description="Execuções por hora")
    concurrent_executions: int = Field(5, description="Execuções concorrentes")
    max_batch_size: int = Field(50, description="Tamanho máximo lote")
    total_cpu_seconds: float = Field(3600.0, description="CPU seconds totais")
    total_memory_gb_seconds: float = Field(1800.0, description="Memória GB*seconds")


# ============================================================================
# VALIDATION AND ERROR MODELS
# ============================================================================

class ContractValidationError(BaseModel):
    """Erro de validação de contrato."""
    
    error_type: str = Field(..., description="Tipo do erro")
    error_message: str = Field(..., description="Mensagem do erro")
    field_path: Optional[str] = Field(None, description="Caminho do campo")
    invalid_value: Optional[Any] = Field(None, description="Valor inválido")
    suggestions: List[str] = Field(default_factory=list, description="Sugestões de correção")


class ContractValidationResult(BaseModel):
    """Resultado da validação de contrato."""
    
    is_valid: bool = Field(..., description="Contrato é válido")
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    errors: List[ContractValidationError] = Field(default_factory=list, description="Lista de erros")
    warnings: List[str] = Field(default_factory=list, description="Lista de warnings")
    validation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados validação")


# ============================================================================
# PERFORMANCE AND MONITORING MODELS
# ============================================================================

class ContractPerformanceMetrics(BaseModel):
    """Métricas de performance de contratos."""
    
    contract_type: ContractType = Field(..., description="Tipo do contrato")
    avg_execution_time_ms: float = Field(..., description="Tempo médio execução")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Taxa de sucesso")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Taxa de erro")
    timeout_rate: float = Field(..., ge=0.0, le=1.0, description="Taxa de timeout")
    avg_memory_usage_mb: float = Field(..., description="Uso médio memória")
    avg_cpu_usage_percent: float = Field(..., description="Uso médio CPU")
    total_executions: int = Field(..., description="Total de execuções")
    period_start: datetime = Field(..., description="Início do período")
    period_end: datetime = Field(..., description="Fim do período")


class SystemHealthStatus(BaseModel):
    """Status de saúde do sistema Score Contracts."""
    
    status: str = Field(..., description="Status geral")
    timestamp: datetime = Field(..., description="Timestamp do check")
    available_contracts: List[ContractType] = Field(..., description="Contratos disponíveis")
    sandbox_status: str = Field(..., description="Status do sandbox")
    active_executions: int = Field(..., description="Execuções ativas")
    queued_executions: int = Field(..., description="Execuções na fila")
    resource_usage: Dict[str, float] = Field(..., description="Uso de recursos")
    last_error: Optional[str] = Field(None, description="Último erro")
    uptime_seconds: float = Field(..., description="Uptime em segundos")


# ============================================================================
# RESPONSE WRAPPER MODELS  
# ============================================================================

class ContractAPIResponse(BaseModel):
    """Response wrapper padrão para APIs de contratos."""
    
    success: bool = Field(..., description="Operação bem-sucedida")
    data: Optional[Any] = Field(None, description="Dados da resposta")
    error: Optional[str] = Field(None, description="Mensagem de erro")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")
    execution_info: Optional[Dict[str, Any]] = Field(None, description="Info da execução")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp resposta")


# ============================================================================
# BACKWARD COMPATIBILITY MODELS
# ============================================================================

class LegacyContractInput(BaseModel):
    """Modelo para compatibilidade com sistema anterior."""
    
    contract_type: str = Field(..., description="Tipo do contrato (string)")
    data: Dict[str, Any] = Field(..., description="Dados de entrada")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parâmetros")
    timeout_seconds: Optional[float] = Field(30.0, description="Timeout")


class LegacyContractOutput(BaseModel):
    """Output para compatibilidade com sistema anterior."""
    
    execution_id: str = Field(..., description="ID da execução")
    contract_type: str = Field(..., description="Tipo do contrato")
    status: str = Field(..., description="Status da execução")
    score: Optional[float] = Field(None, description="Score")
    confidence: Optional[float] = Field(None, description="Confiança")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadados")
    error_message: Optional[str] = Field(None, description="Mensagem de erro")
    execution_time_ms: Optional[float] = Field(None, description="Tempo execução")
    timestamp: Optional[str] = Field(None, description="Timestamp")