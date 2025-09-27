"""Contract Validator - Sistema de Validação Robusta

Sistema de validação em múltiplas camadas para contratos:
- Schema validation (Pydantic)
- Security validation (Code safety)
- Resource validation (Limites)
- Dependency validation (Imports)
- Input validation (Data constraints)

Feature crítica #5 para mestrado - Validação robusta e segura.
"""

import ast
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json

from ..models.contract_models import (
    ContractType,
    ContractExecutionRequest,
    ContractValidationError,
    ContractValidationResult,
    SecurityLevel,
    SandboxConfig
)

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Módulos permitidos por nível de segurança
SECURITY_LEVEL_IMPORTS = {
    SecurityLevel.MINIMAL: {
        "math", "statistics", "itertools", "collections"
    },
    SecurityLevel.STANDARD: {
        "math", "numpy", "scipy", "networkx", "pandas", "statistics",
        "itertools", "collections", "json", "datetime", "uuid",
        "functools", "operator", "copy", "re", "string"
    },
    SecurityLevel.STRICT: {
        "math", "numpy", "scipy", "networkx", "pandas", "statistics",
        "itertools", "collections", "json", "datetime", "uuid",
        "functools", "operator", "copy", "re", "string", "random",
        "sympy", "mpmath", "sklearn"
    },
    SecurityLevel.MAXIMUM: {
        "math", "numpy", "scipy", "networkx", "pandas", "statistics",
        "itertools", "collections", "json", "datetime", "uuid",
        "functools", "operator", "copy", "re", "string", "random",
        "sympy", "mpmath", "sklearn", "matplotlib", "seaborn"
    }
}

# Operações sempre proibidas independente do nível
ALWAYS_FORBIDDEN = {
    "eval", "exec", "compile", "__import__", "open", "file",
    "subprocess", "os", "sys", "socket", "urllib", "requests",
    "pickle", "marshal", "shelve", "dbm", "sqlite3",
    "exit", "quit", "help", "input", "raw_input",
    "reload", "vars", "globals", "locals", "dir",
    # Operações de reflexão perigosas
    "getattr", "setattr", "delattr", "hasattr",
    "isinstance", "issubclass", "callable",
    # Operações de arquivo
    "write", "read", "seek", "tell", "close"
}

# Padrões perigosos de código
DANGEROUS_PATTERNS = [
    r'__[a-zA-Z_]+__',  # Métodos mágicos
    r'import\s+os',      # Import direto de os
    r'import\s+sys',     # Import direto de sys
    r'from\s+os',        # From os import
    r'from\s+sys',       # From sys import
    r'\.system\(',       # Chamadas system
    r'\.popen\(',        # Chamadas popen
    r'\.spawn\(',        # Chamadas spawn
    r'\.fork\(',         # Chamadas fork
    r'\.kill\(',         # Chamadas kill
    r'subprocess\.',     # Uso de subprocess
    r'eval\s*\(',        # Chamadas eval
    r'exec\s*\(',        # Chamadas exec
    r'compile\s*\(',     # Chamadas compile
    r'open\s*\(',        # Abertura de arquivos
    r'file\s*\(',        # Operações de arquivo
    r'with\s+open',      # Context manager com open
]


# ============================================================================
# AST ANALYZER
# ============================================================================

class ASTSecurityAnalyzer(ast.NodeVisitor):
    """Analisador AST para detectar código perigoso."""
    
    def __init__(self, allowed_imports: Set[str], forbidden_ops: Set[str]):
        self.allowed_imports = allowed_imports
        self.forbidden_ops = forbidden_ops
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.imports_found: Set[str] = set()
        self.function_calls: List[str] = []
        
    def visit_Import(self, node: ast.Import) -> None:
        """Analisa imports simples."""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            self.imports_found.add(module_name)
            
            if module_name not in self.allowed_imports:
                self.errors.append(f"Forbidden import: {module_name}")
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Analisa imports from."""
        if node.module:
            module_name = node.module.split('.')[0]
            self.imports_found.add(module_name)
            
            if module_name not in self.allowed_imports:
                self.errors.append(f"Forbidden import from: {module_name}")
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Analisa chamadas de função."""
        # Extrai nome da função
        func_name = self._get_func_name(node.func)
        if func_name:
            self.function_calls.append(func_name)
            
            # Verifica operações proibidas
            if func_name in self.forbidden_ops:
                self.errors.append(f"Forbidden function call: {func_name}")
            
            # Verifica padrões perigosos específicos
            if func_name in ['eval', 'exec', 'compile']:
                self.errors.append(f"Dangerous function call: {func_name}")
            
            if func_name == 'open' or 'open' in func_name.lower():
                self.errors.append("File operations are not allowed")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Analisa acessos a atributos."""
        attr_name = node.attr
        
        # Verifica atributos perigosos
        if attr_name.startswith('__') and attr_name.endswith('__'):
            self.warnings.append(f"Magic method access: {attr_name}")
        
        # Verifica acessos a módulos do sistema
        if isinstance(node.value, ast.Name):
            if node.value.id in ['os', 'sys', 'subprocess']:
                self.errors.append(f"System module access: {node.value.id}.{attr_name}")
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """Analisa nomes/identificadores."""
        name = node.id
        
        # Verifica identificadores proibidos
        if name in self.forbidden_ops:
            self.warnings.append(f"Forbidden identifier: {name}")
        
        self.generic_visit(node)
    
    def _get_func_name(self, node: ast.AST) -> Optional[str]:
        """Extrai nome da função de um nó AST."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_func_name(node.func)
        return None


# ============================================================================
# SCHEMA VALIDATORS
# ============================================================================

class SchemaValidator:
    """Validador de schemas JSON para contratos."""
    
    @staticmethod
    def validate_delta_kec_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida schema do Delta KEC v1."""
        errors = []
        
        # Campos obrigatórios
        if 'source_entropy' not in data:
            errors.append("Missing required field: source_entropy")
        elif not isinstance(data['source_entropy'], (int, float)):
            errors.append("source_entropy must be a number")
        elif data['source_entropy'] < 0:
            errors.append("source_entropy must be non-negative")
        
        if 'target_entropy' not in data:
            errors.append("Missing required field: target_entropy")
        elif not isinstance(data['target_entropy'], (int, float)):
            errors.append("target_entropy must be a number")
        elif data['target_entropy'] < 0:
            errors.append("target_entropy must be non-negative")
        
        # Campos opcionais
        if 'mutual_information' in data:
            if not isinstance(data['mutual_information'], (int, float)):
                errors.append("mutual_information must be a number")
            elif data['mutual_information'] < 0:
                errors.append("mutual_information must be non-negative")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_zuco_reading_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida schema do ZuCo Reading v1."""
        errors = []
        
        # EEG features obrigatórias
        if 'eeg_features' not in data:
            errors.append("Missing required field: eeg_features")
        else:
            eeg = data['eeg_features']
            required_bands = ['theta_power', 'alpha_power', 'beta_power', 'gamma_power']
            
            for band in required_bands:
                if band not in eeg:
                    errors.append(f"Missing EEG band: {band}")
                elif not isinstance(eeg[band], (int, float)):
                    errors.append(f"EEG {band} must be a number")
                elif eeg[band] < 0:
                    errors.append(f"EEG {band} must be non-negative")
        
        # Eye tracking features
        if 'eye_tracking_features' not in data:
            errors.append("Missing required field: eye_tracking_features")
        else:
            et = data['eye_tracking_features']
            numeric_fields = ['avg_fixation_duration', 'avg_saccade_velocity', 'reading_speed_wpm']
            
            for field in numeric_fields:
                if field in et:
                    if not isinstance(et[field], (int, float)):
                        errors.append(f"Eye tracking {field} must be a number")
                    elif et[field] < 0:
                        errors.append(f"Eye tracking {field} must be non-negative")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_biomaterials_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida schema do Biomaterials Scaffold."""
        errors = []
        
        # Scaffold structure
        if 'scaffold_structure' not in data:
            errors.append("Missing required field: scaffold_structure")
        else:
            structure = data['scaffold_structure']
            
            if 'porosity' in structure:
                porosity = structure['porosity']
                if not isinstance(porosity, (int, float)):
                    errors.append("porosity must be a number")
                elif not 0 <= porosity <= 1:
                    errors.append("porosity must be between 0 and 1")
            
            if 'connectivity' in structure:
                connectivity = structure['connectivity']
                if not isinstance(connectivity, (int, float)):
                    errors.append("connectivity must be a number")
                elif not 0 <= connectivity <= 1:
                    errors.append("connectivity must be between 0 and 1")
        
        # Material properties
        if 'material_properties' not in data:
            errors.append("Missing required field: material_properties")
        else:
            materials = data['material_properties']
            
            numeric_props = ['young_modulus', 'tensile_strength', 'biocompatibility_index']
            for prop in numeric_props:
                if prop in materials:
                    value = materials[prop]
                    if not isinstance(value, (int, float)):
                        errors.append(f"{prop} must be a number")
                    elif value < 0:
                        errors.append(f"{prop} must be non-negative")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_network_topology_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida schema do Network Topology."""
        errors = []
        
        # Adjacency matrix
        if 'adjacency_matrix' not in data:
            errors.append("Missing required field: adjacency_matrix")
        else:
            matrix = data['adjacency_matrix']
            
            if not isinstance(matrix, list):
                errors.append("adjacency_matrix must be a list")
            elif not matrix:
                errors.append("adjacency_matrix cannot be empty")
            else:
                n = len(matrix)
                for i, row in enumerate(matrix):
                    if not isinstance(row, list):
                        errors.append(f"Row {i} of adjacency_matrix must be a list")
                        break
                    elif len(row) != n:
                        errors.append("adjacency_matrix must be square")
                        break
                    
                    for j, val in enumerate(row):
                        if not isinstance(val, (int, float)):
                            errors.append(f"adjacency_matrix[{i}][{j}] must be a number")
                        elif val < 0:
                            errors.append(f"adjacency_matrix[{i}][{j}] must be non-negative")
        
        return len(errors) == 0, errors


# ============================================================================
# RESOURCE VALIDATOR
# ============================================================================

class ResourceValidator:
    """Validador de requisitos de recursos."""
    
    def __init__(self):
        self.max_matrix_size = 1000
        self.max_data_size_mb = 10
        self.max_list_length = 10000
        
    def validate_resource_requirements(self, data: Dict[str, Any], contract_type: ContractType) -> Tuple[bool, List[str]]:
        """Valida requisitos de recursos para execução."""
        errors = []
        
        # Calcula tamanho dos dados
        data_size = self._estimate_data_size(data)
        if data_size > self.max_data_size_mb * 1024 * 1024:
            errors.append(f"Data size too large: {data_size/1024/1024:.1f}MB > {self.max_data_size_mb}MB")
        
        # Valida específico por tipo de contrato
        if contract_type in [ContractType.NETWORK_TOPOLOGY, ContractType.SPECTRAL_ANALYSIS]:
            # Verifica tamanho da matriz
            matrix_field = 'adjacency_matrix' if contract_type == ContractType.NETWORK_TOPOLOGY else 'graph_laplacian'
            
            if matrix_field in data:
                matrix = data[matrix_field]
                if isinstance(matrix, list) and len(matrix) > self.max_matrix_size:
                    errors.append(f"Matrix too large: {len(matrix)} > {self.max_matrix_size}")
        
        # Valida listas grandes
        for key, value in data.items():
            if isinstance(value, list) and len(value) > self.max_list_length:
                errors.append(f"List {key} too long: {len(value)} > {self.max_list_length}")
        
        return len(errors) == 0, errors
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estima tamanho dos dados em bytes."""
        try:
            return len(json.dumps(data, default=str).encode('utf-8'))
        except Exception:
            return sys.getsizeof(data)


# ============================================================================
# CODE SECURITY ANALYZER
# ============================================================================

class CodeSecurityAnalyzer:
    """Analisador de segurança de código."""
    
    def __init__(self, security_level: SecurityLevel):
        self.security_level = security_level
        self.allowed_imports = SECURITY_LEVEL_IMPORTS.get(security_level, set())
        self.forbidden_ops = ALWAYS_FORBIDDEN
        
    def analyze_code_security(self, code: str) -> Tuple[bool, List[str], List[str]]:
        """Analisa segurança do código."""
        errors = []
        warnings = []
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Análise AST
            analyzer = ASTSecurityAnalyzer(self.allowed_imports, self.forbidden_ops)
            analyzer.visit(tree)
            
            errors.extend(analyzer.errors)
            warnings.extend(analyzer.warnings)
            
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors, warnings
        
        # Análise de padrões regex
        pattern_errors = self._check_dangerous_patterns(code)
        errors.extend(pattern_errors)
        
        # Análise de complexidade
        complexity_warnings = self._check_code_complexity(code)
        warnings.extend(complexity_warnings)
        
        return len(errors) == 0, errors, warnings
    
    def _check_dangerous_patterns(self, code: str) -> List[str]:
        """Verifica padrões perigosos usando regex."""
        errors = []
        
        for pattern in DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                errors.append(f"Dangerous pattern detected: {pattern}")
        
        return errors
    
    def _check_code_complexity(self, code: str) -> List[str]:
        """Verifica complexidade do código."""
        warnings = []
        
        lines = code.split('\n')
        
        # Muito longo
        if len(lines) > 200:
            warnings.append("Code is very long (>200 lines)")
        
        # Muitos loops aninhados
        nested_loops = 0
        for line in lines:
            if 'for ' in line or 'while ' in line:
                nested_loops += line.count('    ')  # Indentation level
        
        if nested_loops > 10:
            warnings.append("Code has deeply nested loops")
        
        # Muitas variáveis
        variables = set()
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                var_name = line.split('=')[0].strip().split()[-1]
                if var_name.isidentifier():
                    variables.add(var_name)
        
        if len(variables) > 50:
            warnings.append("Code has many variables (>50)")
        
        return warnings


# ============================================================================
# DEPENDENCY VALIDATOR
# ============================================================================

class DependencyValidator:
    """Validador de dependências de contratos."""
    
    def __init__(self):
        self.available_modules = self._get_available_modules()
    
    def validate_dependencies(self, contract_type: ContractType, imports: Set[str]) -> Tuple[bool, List[str]]:
        """Valida se dependências estão disponíveis."""
        errors = []
        
        # Verifica se módulos estão instalados
        for module in imports:
            if module not in self.available_modules:
                errors.append(f"Required module not available: {module}")
        
        # Validações específicas por tipo
        if contract_type == ContractType.BIOMATERIALS_SCAFFOLD:
            required = {'numpy', 'networkx'}
            missing = required - imports.intersection(self.available_modules)
            if missing:
                errors.append(f"Biomaterials contract requires: {missing}")
        
        elif contract_type == ContractType.SPECTRAL_ANALYSIS:
            required = {'numpy', 'scipy'}
            missing = required - imports.intersection(self.available_modules)
            if missing:
                errors.append(f"Spectral analysis requires: {missing}")
        
        return len(errors) == 0, errors
    
    def _get_available_modules(self) -> Set[str]:
        """Obtém lista de módulos disponíveis."""
        available = set()
        
        # Módulos built-in sempre disponíveis
        builtin_modules = {
            'math', 'statistics', 'itertools', 'collections',
            'json', 'datetime', 'uuid', 'functools', 'operator',
            'copy', 're', 'string', 'random'
        }
        available.update(builtin_modules)
        
        # Tenta importar módulos científicos
        scientific_modules = ['numpy', 'scipy', 'networkx', 'pandas', 'sympy', 'mpmath', 'sklearn']
        
        for module in scientific_modules:
            try:
                __import__(module)
                available.add(module)
            except ImportError:
                pass
        
        return available


# ============================================================================
# MAIN VALIDATOR
# ============================================================================

class ContractValidator:
    """Validador principal de contratos."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.schema_validator = SchemaValidator()
        self.resource_validator = ResourceValidator()
        self.dependency_validator = DependencyValidator()
        self.security_analyzer = CodeSecurityAnalyzer(security_level)
        
        logger.info(f"ContractValidator initialized with security level: {security_level}")
    
    def validate_contract_request(self, request: ContractExecutionRequest) -> ContractValidationResult:
        """Valida completamente uma request de execução."""
        
        errors = []
        warnings = []
        validation_metadata = {
            'contract_type': request.contract_type.value,
            'security_level': self.security_level.value,
            'validation_timestamp': 0.0  # Placeholder
        }
        
        try:
            # === VALIDAÇÃO DE SCHEMA ===
            
            schema_valid, schema_errors = self._validate_input_schema(request.contract_type, request.data)
            if not schema_valid:
                errors.extend([
                    ContractValidationError(
                        error_type="schema_error",
                        error_message=error,
                        field_path="data",
                        invalid_value=request.data,
                        suggestions=["Check input data format and required fields"]
                    ) for error in schema_errors
                ])
            
            # === VALIDAÇÃO DE RECURSOS ===
            
            resource_valid, resource_errors = self.resource_validator.validate_resource_requirements(
                request.data, request.contract_type
            )
            if not resource_valid:
                errors.extend([
                    ContractValidationError(
                        error_type="resource_error",
                        error_message=error,
                        field_path="data",
                        invalid_value=None,
                        suggestions=["Reduce data size or matrix dimensions"]
                    ) for error in resource_errors
                ])
            
            # === VALIDAÇÃO DE PARÂMETROS ===
            
            if request.parameters:
                param_valid, param_errors = self._validate_parameters(request.contract_type, request.parameters)
                if not param_valid:
                    errors.extend([
                        ContractValidationError(
                            error_type="parameter_error",
                            error_message=error,
                            field_path="parameters",
                            invalid_value=request.parameters,
                            suggestions=["Check parameter names and value ranges"]
                        ) for error in param_errors
                    ])
            
            # === VALIDAÇÃO DE TIMEOUT ===
            
            if request.timeout_seconds:
                if request.timeout_seconds < 1.0 or request.timeout_seconds > 300.0:
                    errors.append(ContractValidationError(
                        error_type="timeout_error",
                        error_message="Timeout must be between 1 and 300 seconds",
                        field_path="timeout_seconds",
                        invalid_value=request.timeout_seconds,
                        suggestions=["Set timeout between 1 and 300 seconds"]
                    ))
            
            validation_metadata.update({
                'schema_valid': schema_valid,
                'resource_valid': resource_valid,
                'total_errors': len(errors),
                'total_warnings': len(warnings)
            })
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            errors.append(ContractValidationError(
                error_type="validation_exception",
                error_message=f"Validation failed: {str(e)}",
                field_path="request",
                invalid_value=None,
                suggestions=["Check request format and try again"]
            ))
        
        return ContractValidationResult(
            is_valid=len(errors) == 0,
            contract_type=request.contract_type,
            errors=errors,
            warnings=warnings,
            validation_metadata=validation_metadata
        )
    
    def validate_contract_code(self, code: str, contract_type: ContractType) -> Tuple[bool, List[str], List[str]]:
        """Valida código de contrato personalizado."""
        
        # Análise de segurança
        is_secure, sec_errors, sec_warnings = self.security_analyzer.analyze_code_security(code)
        
        # Validação de dependências
        analyzer = ASTSecurityAnalyzer(
            self.security_analyzer.allowed_imports,
            self.security_analyzer.forbidden_ops
        )
        
        try:
            tree = ast.parse(code)
            analyzer.visit(tree)
            
            dep_valid, dep_errors = self.dependency_validator.validate_dependencies(
                contract_type, analyzer.imports_found
            )
            
            if not dep_valid:
                sec_errors.extend(dep_errors)
                is_secure = False
                
        except SyntaxError as e:
            sec_errors.append(f"Syntax error: {e}")
            is_secure = False
        
        return is_secure, sec_errors, sec_warnings
    
    def _validate_input_schema(self, contract_type: ContractType, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida schema de entrada por tipo de contrato."""
        
        validators = {
            ContractType.DELTA_KEC_V1: self.schema_validator.validate_delta_kec_schema,
            ContractType.ZUCO_READING_V1: self.schema_validator.validate_zuco_reading_schema,
            ContractType.BIOMATERIALS_SCAFFOLD: self.schema_validator.validate_biomaterials_schema,
            ContractType.NETWORK_TOPOLOGY: self.schema_validator.validate_network_topology_schema,
        }
        
        validator_func = validators.get(contract_type)
        if validator_func:
            return validator_func(data)
        
        # Para contratos sem validador específico, apenas verifica que não está vazio
        if not data:
            return False, ["Input data cannot be empty"]
        
        return True, []
    
    def _validate_parameters(self, contract_type: ContractType, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida parâmetros por tipo de contrato."""
        errors = []
        
        if contract_type == ContractType.DELTA_KEC_V1:
            # Valida alpha e beta
            if 'alpha' in parameters:
                alpha = parameters['alpha']
                if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
                    errors.append("alpha must be a number between 0 and 1")
            
            if 'beta' in parameters:
                beta = parameters['beta']
                if not isinstance(beta, (int, float)) or not 0 <= beta <= 1:
                    errors.append("beta must be a number between 0 and 1")
        
        elif contract_type == ContractType.ZUCO_READING_V1:
            # Valida pesos EEG e ET
            if 'eeg_weight' in parameters:
                weight = parameters['eeg_weight']
                if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                    errors.append("eeg_weight must be a number between 0 and 1")
            
            if 'et_weight' in parameters:
                weight = parameters['et_weight']
                if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                    errors.append("et_weight must be a number between 0 and 1")
        
        elif contract_type == ContractType.EDITORIAL_V1:
            # Valida pesos das dimensões
            if 'weights' in parameters:
                weights = parameters['weights']
                if not isinstance(weights, dict):
                    errors.append("weights must be a dictionary")
                else:
                    valid_dimensions = {'readability', 'grammar', 'vocabulary', 'coherence', 'originality'}
                    for dim, weight in weights.items():
                        if dim not in valid_dimensions:
                            errors.append(f"Invalid weight dimension: {dim}")
                        elif not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                            errors.append(f"Weight {dim} must be a number between 0 and 1")
        
        return len(errors) == 0, errors


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_validator(security_level: SecurityLevel = SecurityLevel.STANDARD) -> ContractValidator:
    """Cria validador com nível de segurança específico."""
    return ContractValidator(security_level)


def validate_contract_request_quick(request: ContractExecutionRequest) -> bool:
    """Validação rápida para verificação inicial."""
    try:
        validator = create_validator()
        result = validator.validate_contract_request(request)
        return result.is_valid
    except Exception:
        return False


# Instância global do validador
_contract_validator: Optional[ContractValidator] = None


def get_contract_validator() -> ContractValidator:
    """Obtém instância global do validador."""
    global _contract_validator
    
    if _contract_validator is None:
        _contract_validator = create_validator()
    
    return _contract_validator


def initialize_contract_validator(security_level: SecurityLevel) -> ContractValidator:
    """Inicializa validador com nível de segurança específico."""
    global _contract_validator
    
    _contract_validator = create_validator(security_level)
    return _contract_validator