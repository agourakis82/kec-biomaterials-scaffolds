"""Sandbox Manager - Sistema de Execução Segura

Sistema de sandbox para execução segura de contratos matemáticos com isolamento
completo de processos, recursos limitados e validação rigorosa.

Feature crítica #5 para mestrado - Security absoluta do sandbox.
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import psutil
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from ..models.contract_models import (
    ContractType,
    ExecutionStatus,
    SandboxConfig,
    ExecutionLimits,
    SecurityLevel,
    SecurityPolicy
)

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

# Whitelist de módulos permitidos no sandbox
ALLOWED_IMPORTS = {
    "math", "numpy", "scipy", "networkx", "pandas", "statistics", 
    "itertools", "collections", "json", "datetime", "uuid",
    "functools", "operator", "copy", "re", "string", "random",
    # Específicos para análise matemática
    "sympy", "mpmath", "sklearn", "matplotlib", "seaborn"
}

# Operações proibidas no sandbox
FORBIDDEN_OPERATIONS = {
    "eval", "exec", "compile", "__import__", "open", "file",
    "subprocess", "os", "sys", "socket", "urllib", "requests",
    "pickle", "marshal", "shelve", "dbm", "sqlite3",
    # Operações de sistema
    "exit", "quit", "help", "input", "raw_input",
    "reload", "vars", "globals", "locals", "dir"
}

# Limites padrão de recursos por tipo de contrato
DEFAULT_RESOURCE_LIMITS = {
    ContractType.DELTA_KEC_V1: {
        "cpu_seconds": 10.0,
        "memory_mb": 256,
        "wall_time_seconds": 30.0
    },
    ContractType.BIOMATERIALS_SCAFFOLD: {
        "cpu_seconds": 30.0,
        "memory_mb": 512,
        "wall_time_seconds": 60.0
    },
    ContractType.NETWORK_TOPOLOGY: {
        "cpu_seconds": 20.0,
        "memory_mb": 384,
        "wall_time_seconds": 45.0
    },
    ContractType.SPECTRAL_ANALYSIS: {
        "cpu_seconds": 25.0,
        "memory_mb": 448,
        "wall_time_seconds": 50.0
    }
}


# ============================================================================
# SECURITY EXCEPTIONS
# ============================================================================

class SandboxSecurityError(Exception):
    """Erro de segurança no sandbox."""
    pass


class SandboxTimeoutError(Exception):
    """Timeout na execução do sandbox."""
    pass


class SandboxResourceError(Exception):
    """Erro de recursos no sandbox."""
    pass


class SandboxValidationError(Exception):
    """Erro de validação no sandbox."""
    pass


# ============================================================================
# PROCESS MONITOR
# ============================================================================

class ProcessMonitor:
    """Monitor de processo para controle de recursos."""
    
    def __init__(self, pid: int, limits: ExecutionLimits):
        self.pid = pid
        self.limits = limits
        self.start_time = time.time()
        self.max_memory_used = 0
        self.cpu_time_used = 0.0
        
    def check_limits(self) -> None:
        """Verifica se os limites estão sendo respeitados."""
        try:
            process = psutil.Process(self.pid)
            
            # Verifica memória
            memory_info = process.memory_info()
            memory_bytes = memory_info.rss
            self.max_memory_used = max(self.max_memory_used, memory_bytes)
            
            if memory_bytes > self.limits.max_memory_bytes:
                raise SandboxResourceError(
                    f"Memory limit exceeded: {memory_bytes} > {self.limits.max_memory_bytes}"
                )
            
            # Verifica tempo CPU
            cpu_times = process.cpu_times()
            self.cpu_time_used = cpu_times.user + cpu_times.system
            
            if self.cpu_time_used > self.limits.max_cpu_time_seconds:
                raise SandboxResourceError(
                    f"CPU time limit exceeded: {self.cpu_time_used} > {self.limits.max_cpu_time_seconds}"
                )
            
            # Verifica tempo wall
            wall_time = time.time() - self.start_time
            if wall_time > self.limits.max_wall_time_seconds:
                raise SandboxTimeoutError(
                    f"Wall time limit exceeded: {wall_time} > {self.limits.max_wall_time_seconds}"
                )
            
            # Verifica número de processos/threads
            num_threads = process.num_threads()
            if num_threads > self.limits.max_threads:
                raise SandboxResourceError(
                    f"Thread limit exceeded: {num_threads} > {self.limits.max_threads}"
                )
                
        except psutil.NoSuchProcess:
            # Processo terminou
            pass
        except psutil.AccessDenied:
            logger.warning(f"Access denied monitoring process {self.pid}")


# ============================================================================
# CODE VALIDATOR
# ============================================================================

class CodeValidator:
    """Validador de código para sandbox."""
    
    def __init__(self, allowed_imports: Set[str], forbidden_ops: Set[str]):
        self.allowed_imports = allowed_imports
        self.forbidden_ops = forbidden_ops
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Valida código antes da execução."""
        errors = []
        
        try:
            # Compila o código para verificar sintaxe
            compile(code, '<sandbox>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors
        
        # Verifica operações proibidas
        for forbidden_op in self.forbidden_ops:
            if forbidden_op in code:
                errors.append(f"Forbidden operation detected: {forbidden_op}")
        
        # Verifica imports (básico)
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Análise básica de imports
                if 'import ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        module_name = parts[1].split('.')[0]
                        if module_name not in self.allowed_imports:
                            errors.append(f"Forbidden import: {module_name}")
        
        return len(errors) == 0, errors


# ============================================================================
# SANDBOX EXECUTOR
# ============================================================================

class SandboxExecutor:
    """Executor seguro para contratos no sandbox."""
    
    def __init__(self, config: SandboxConfig, limits: ExecutionLimits):
        self.config = config
        self.limits = limits
        self.validator = CodeValidator(
            allowed_imports=set(config.allowed_imports) if config.allowed_imports else ALLOWED_IMPORTS,
            forbidden_ops=set(config.forbidden_operations) if config.forbidden_operations else FORBIDDEN_OPERATIONS
        )
        
    async def execute_safe(
        self, 
        contract_code: str, 
        input_data: Dict[str, Any],
        contract_type: ContractType
    ) -> Dict[str, Any]:
        """Executa código de contrato em ambiente seguro."""
        
        # Gera ID único para execução
        execution_id = str(uuid.uuid4())
        logger.info(f"Starting sandbox execution {execution_id} for {contract_type}")
        
        # Valida código
        is_valid, validation_errors = self.validator.validate_code(contract_code)
        if not is_valid:
            raise SandboxValidationError(f"Code validation failed: {validation_errors}")
        
        # Cria ambiente isolado
        with tempfile.TemporaryDirectory(prefix=f"sandbox_{execution_id}_") as temp_dir:
            try:
                result = await self._execute_in_subprocess(
                    contract_code, input_data, temp_dir, execution_id, contract_type
                )
                logger.info(f"Sandbox execution {execution_id} completed successfully")
                return result
                
            except Exception as e:
                logger.error(f"Sandbox execution {execution_id} failed: {e}")
                raise
    
    async def _execute_in_subprocess(
        self,
        code: str,
        input_data: Dict[str, Any],
        temp_dir: str,
        execution_id: str,
        contract_type: ContractType
    ) -> Dict[str, Any]:
        """Executa código em subprocess isolado."""
        
        # Cria script de execução
        script_content = self._generate_execution_script(code, input_data)
        script_path = Path(temp_dir) / "contract_script.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Cria comando do subprocess
        python_cmd = [
            sys.executable,
            "-u",  # unbuffered output
            "-S",  # don't add user site directory
            str(script_path)
        ]
        
        # Configura ambiente limitado
        env = self._create_limited_environment()
        
        start_time = time.time()
        process = None
        monitor = None
        
        try:
            # Inicia processo
            process = await asyncio.create_subprocess_exec(
                *python_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=temp_dir,
                preexec_fn=self._setup_process_limits
            )
            
            # Inicia monitoramento
            monitor = ProcessMonitor(process.pid, self.limits)
            monitor_task = asyncio.create_task(self._monitor_process(monitor))
            
            # Espera execução com timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.limits.max_wall_time_seconds
                )
            except asyncio.TimeoutError:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                raise SandboxTimeoutError("Process execution timed out")
            
            # Cancela monitoramento
            monitor_task.cancel()
            
            execution_time = time.time() - start_time
            
            # Verifica resultado
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise SandboxSecurityError(f"Process failed with return code {process.returncode}: {error_msg}")
            
            # Processa saída
            output = stdout.decode('utf-8', errors='ignore')
            result = self._parse_execution_output(output)
            
            # Adiciona metadados de execução
            result.update({
                'execution_id': execution_id,
                'execution_time_seconds': execution_time,
                'max_memory_used_bytes': monitor.max_memory_used if monitor else 0,
                'cpu_time_used_seconds': monitor.cpu_time_used if monitor else 0,
                'contract_type': contract_type.value
            })
            
            return result
            
        except Exception as e:
            # Cleanup do processo se ainda estiver rodando
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except:
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass
            raise e
    
    async def _monitor_process(self, monitor: ProcessMonitor) -> None:
        """Task de monitoramento contínuo do processo."""
        try:
            while True:
                monitor.check_limits()
                await asyncio.sleep(0.1)  # Check every 100ms
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Process monitoring error: {e}")
            raise
    
    def _setup_process_limits(self) -> None:
        """Configura limites de recursos no processo filho."""
        try:
            # Limite de memória virtual
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.limits.max_memory_bytes, self.limits.max_memory_bytes)
            )
            
            # Limite de tempo CPU
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.limits.max_cpu_time_seconds), int(self.limits.max_cpu_time_seconds))
            )
            
            # Limite de arquivos abertos
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            
            # Limite de processos
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            
        except Exception as e:
            logger.warning(f"Failed to set process limits: {e}")
    
    def _create_limited_environment(self) -> Dict[str, str]:
        """Cria ambiente limitado para execução."""
        env = {
            'PATH': '/usr/bin:/bin',
            'PYTHONPATH': '',
            'HOME': '/tmp',
            'USER': 'sandbox',
            'SHELL': '/bin/false',
            'LANG': 'C.UTF-8'
        }
        
        # Remove variáveis perigosas
        dangerous_vars = [
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONSTARTUP',
            'PYTHONHOME', 'PYTHONEXECUTABLE'
        ]
        
        for var in dangerous_vars:
            env.pop(var, None)
        
        return env
    
    def _generate_execution_script(self, code: str, input_data: Dict[str, Any]) -> str:
        """Gera script de execução com wrapper de segurança."""
        
        # Serializa dados de entrada de forma segura
        import json
        input_json = json.dumps(input_data, default=str)
        
        script_template = f'''#!/usr/bin/env python3
# Sandbox execution script
import sys
import json
import traceback

# Restricted builtins
restricted_builtins = {{
    name: getattr(__builtins__, name) 
    for name in dir(__builtins__) 
    if name not in {repr(FORBIDDEN_OPERATIONS)}
}}

# Override builtins
__builtins__ = restricted_builtins

def safe_print(*args, **kwargs):
    """Safe print function."""
    print(*args, **kwargs)

# Input data
input_data = {input_json}

# Contract execution wrapper
try:
    # Create execution namespace
    namespace = {{
        '__builtins__': restricted_builtins,
        'input_data': input_data,
        'print': safe_print,
        'result': None
    }}
    
    # Execute contract code
    exec("""
{code}
""", namespace)
    
    # Extract result
    result = namespace.get('result')
    if result is None:
        result = {{'error': 'No result returned by contract'}}
    
    # Output result
    print("SANDBOX_RESULT_START")
    print(json.dumps(result, default=str))
    print("SANDBOX_RESULT_END")
    
except Exception as e:
    error_result = {{
        'error': str(e),
        'error_type': type(e).__name__,
        'traceback': traceback.format_exc()
    }}
    print("SANDBOX_RESULT_START")
    print(json.dumps(error_result, default=str))
    print("SANDBOX_RESULT_END")
'''
        
        return script_template
    
    def _parse_execution_output(self, output: str) -> Dict[str, Any]:
        """Parse da saída da execução do sandbox."""
        try:
            # Procura pelos marcadores de resultado
            start_marker = "SANDBOX_RESULT_START"
            end_marker = "SANDBOX_RESULT_END"
            
            start_idx = output.find(start_marker)
            end_idx = output.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                return {
                    'error': 'Invalid output format',
                    'raw_output': output
                }
            
            result_json = output[start_idx + len(start_marker):end_idx].strip()
            
            import json
            result = json.loads(result_json)
            
            return result
            
        except Exception as e:
            return {
                'error': f'Failed to parse output: {e}',
                'raw_output': output
            }


# ============================================================================
# SANDBOX MANAGER
# ============================================================================

class SandboxManager:
    """Gerenciador principal do sistema de sandbox."""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.security_policy = security_policy
        self.active_executions: Dict[str, SandboxExecutor] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        logger.info(f"SandboxManager initialized with security level: {security_policy.security_level}")
    
    async def execute_contract(
        self,
        contract_code: str,
        input_data: Dict[str, Any],
        contract_type: ContractType,
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executa contrato no sandbox."""
        
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        # Cria executor para esta execução
        executor = SandboxExecutor(
            config=self.security_policy.sandbox_config,
            limits=self._get_limits_for_contract(contract_type)
        )
        
        # Registra execução ativa
        self.active_executions[execution_id] = executor
        
        start_time = time.time()
        status = ExecutionStatus.RUNNING
        
        try:
            result = await executor.execute_safe(contract_code, input_data, contract_type)
            status = ExecutionStatus.COMPLETED
            return result
            
        except SandboxTimeoutError as e:
            status = ExecutionStatus.TIMEOUT
            raise e
        except (SandboxSecurityError, SandboxValidationError) as e:
            status = ExecutionStatus.FAILED
            raise e
        except Exception as e:
            status = ExecutionStatus.FAILED
            logger.error(f"Unexpected error in sandbox execution {execution_id}: {e}")
            raise SandboxSecurityError(f"Sandbox execution failed: {e}")
            
        finally:
            # Remove execução ativa
            self.active_executions.pop(execution_id, None)
            
            # Registra no histórico
            execution_time = time.time() - start_time
            self._record_execution_history(
                execution_id, contract_type, status, execution_time
            )
    
    def _get_limits_for_contract(self, contract_type: ContractType) -> ExecutionLimits:
        """Obtém limites de recursos para tipo de contrato."""
        
        # Usa limites específicos se configurados
        if hasattr(self.security_policy, 'contract_limits'):
            contract_limits = getattr(self.security_policy, 'contract_limits', {})
            if contract_type in contract_limits:
                limits_dict = contract_limits[contract_type]
                return ExecutionLimits(**limits_dict)
        
        # Usa limites padrão baseados no tipo
        default_limits = DEFAULT_RESOURCE_LIMITS.get(contract_type, DEFAULT_RESOURCE_LIMITS[ContractType.DELTA_KEC_V1])
        
        return ExecutionLimits(
            max_memory_bytes=default_limits["memory_mb"] * 1024 * 1024,
            max_cpu_time_seconds=default_limits["cpu_seconds"],
            max_wall_time_seconds=default_limits["wall_time_seconds"],
            max_file_size_bytes=10 * 1024 * 1024,  # 10MB
            max_processes=1,
            max_threads=4
        )
    
    def _record_execution_history(
        self,
        execution_id: str,
        contract_type: ContractType,
        status: ExecutionStatus,
        execution_time: float
    ) -> None:
        """Registra execução no histórico."""
        
        history_entry = {
            'execution_id': execution_id,
            'contract_type': contract_type.value,
            'status': status.value,
            'execution_time_seconds': execution_time,
            'timestamp': time.time()
        }
        
        self.execution_history.append(history_entry)
        
        # Limita tamanho do histórico
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]
    
    def get_active_executions(self) -> List[str]:
        """Retorna lista de execuções ativas."""
        return list(self.active_executions.keys())
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de execuções."""
        return self.execution_history[-limit:]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancela execução ativa."""
        if execution_id in self.active_executions:
            # TODO: Implementar cancelamento real do processo
            logger.info(f"Cancelling execution {execution_id}")
            return True
        return False
    
    def get_security_info(self) -> Dict[str, Any]:
        """Retorna informações de segurança do sandbox."""
        return {
            'security_level': self.security_policy.security_level.value,
            'sandbox_config': self.security_policy.sandbox_config.dict(),
            'execution_limits': self.security_policy.execution_limits.dict(),
            'active_executions': len(self.active_executions),
            'total_executions_in_history': len(self.execution_history)
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_security_policy() -> SecurityPolicy:
    """Cria política de segurança padrão."""
    
    sandbox_config = SandboxConfig(
        cpu_limit_percent=10.0,
        memory_limit_mb=512,
        execution_timeout_seconds=30.0,
        network_isolation=True,
        file_system_access=SecurityLevel.MINIMAL,
        allowed_imports=list(ALLOWED_IMPORTS),
        forbidden_operations=list(FORBIDDEN_OPERATIONS)
    )
    
    execution_limits = ExecutionLimits(
        max_memory_bytes=512 * 1024 * 1024,
        max_cpu_time_seconds=30.0,
        max_wall_time_seconds=60.0,
        max_file_size_bytes=10 * 1024 * 1024,
        max_processes=1,
        max_threads=4
    )
    
    return SecurityPolicy(
        security_level=SecurityLevel.STANDARD,
        sandbox_config=sandbox_config,
        execution_limits=execution_limits,
        audit_logging=True,
        encryption_required=False,
        user_isolation=True
    )


def create_high_security_policy() -> SecurityPolicy:
    """Cria política de segurança rigorosa."""
    
    sandbox_config = SandboxConfig(
        cpu_limit_percent=5.0,
        memory_limit_mb=256,
        execution_timeout_seconds=15.0,
        network_isolation=True,
        file_system_access=SecurityLevel.MINIMAL,
        allowed_imports=["math", "statistics", "itertools", "collections"],
        forbidden_operations=list(FORBIDDEN_OPERATIONS) + ["print", "input"]
    )
    
    execution_limits = ExecutionLimits(
        max_memory_bytes=256 * 1024 * 1024,
        max_cpu_time_seconds=15.0,
        max_wall_time_seconds=30.0,
        max_file_size_bytes=1 * 1024 * 1024,
        max_processes=1,
        max_threads=2
    )
    
    return SecurityPolicy(
        security_level=SecurityLevel.MAXIMUM,
        sandbox_config=sandbox_config,
        execution_limits=execution_limits,
        audit_logging=True,
        encryption_required=True,
        user_isolation=True
    )


# Instância global do sandbox manager
_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Obtém instância global do sandbox manager."""
    global _sandbox_manager
    
    if _sandbox_manager is None:
        policy = create_default_security_policy()
        _sandbox_manager = SandboxManager(policy)
    
    return _sandbox_manager


def initialize_sandbox_manager(security_policy: SecurityPolicy) -> SandboxManager:
    """Inicializa sandbox manager com política específica."""
    global _sandbox_manager
    
    _sandbox_manager = SandboxManager(security_policy)
    return _sandbox_manager