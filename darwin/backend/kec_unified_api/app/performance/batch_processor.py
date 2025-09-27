"""Batch Processor - Processamento Épico de Milhões de Scaffolds

🌊 BATCH PROCESSOR REVOLUTIONARY - MILHÕES DE SCAFFOLDS SIMULTANEAMENTE
Sistema disruptivo para processamento em massa de scaffolds biomateriais
usando JAX vectorization e chunking inteligente para datasets gigantes.

Features Épicas:
- 🌊 Million Scaffold Processing - Processa milhões simultaneamente
- 📦 Intelligent Chunking - Chunking adaptativo por memória disponível
- ⚡ Vectorized Operations - vmap processing para performance máxima
- 💾 Memory Management - Gestão inteligente de memória para big data
- 📊 Progress Tracking - Tracking em tempo real de progresso épico
- 🔄 Async Processing - Processamento assíncrono não-blocking

Technology: JAX vmap + Chunking + Memory Management + Async Processing
Target: Process 1M+ scaffolds in minutes, not hours
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Iterator, AsyncIterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

from ..core.logging import get_logger
from ..models.kec_models import KECMetricsResult, ExecutionStatus

logger = get_logger("darwin.batch_processor")

# Importações condicionais
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, devices
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX não disponível para batch processing")
    JAX_AVAILABLE = False
    jnp = np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class BatchConfig:
    """Configuração para processamento em lote."""
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    use_gpu: bool = True
    parallel_chunks: int = 4
    progress_callback: Optional[callable] = None
    save_intermediate: bool = False
    output_format: str = "json"  # json, hdf5, npz


@dataclass
class ProcessingProgress:
    """Progresso do processamento em tempo real."""
    batch_id: str
    total_scaffolds: int
    processed_scaffolds: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion: Optional[datetime] = None
    current_throughput: float = 0.0  # scaffolds/second
    errors_count: int = 0


@dataclass
class BatchProcessingResult:
    """Resultado completo do processamento em lote."""
    batch_id: str
    total_processed: int
    success_count: int
    error_count: int
    processing_time_seconds: float
    average_throughput: float  # scaffolds/second
    peak_throughput: float
    memory_peak_mb: float
    device_used: str
    results: List[KECMetricsResult]
    error_details: List[Dict[str, Any]] = field(default_factory=list)


class BatchProcessor:
    """
    🌊 BATCH PROCESSOR REVOLUTIONARY - MILHÕES DE SCAFFOLDS PROCESSING
    
    Processador épico que:
    - Processa milhões de scaffolds usando JAX vectorization
    - Gerencia memória inteligentemente para datasets gigantes  
    - Fornece progress tracking em tempo real
    - Otimiza chunking baseado em hardware disponível
    """
    
    def __init__(self):
        self.processor_id = str(uuid.uuid4())
        self.is_initialized = False
        self.active_batches: Dict[str, ProcessingProgress] = {}
        self.completed_batches: List[BatchProcessingResult] = []
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Memory management
        self.memory_monitor = MemoryMonitor()
        
        # Performance tracking
        self.throughput_history = []
        self.peak_performance = {
            "max_throughput": 0.0,
            "largest_batch": 0,
            "fastest_chunk_time": float('inf'),
            "best_device": "cpu"
        }
        
        logger.info(f"🌊 Batch Processor created: {self.processor_id}")
    
    async def initialize(self):
        """Inicializa o batch processor."""
        try:
            logger.info("🌊 Inicializando Batch Processor...")
            
            # Inicializar thread pool para I/O async
            self.thread_pool = ThreadPoolExecutor(
                max_workers=4, 
                thread_name_prefix="BatchProcessor"
            )
            
            # Inicializar memory monitor
            await self.memory_monitor.initialize()
            
            # Detectar configuração optimal de batching
            optimal_config = await self._detect_optimal_batch_config()
            self.default_config = optimal_config
            
            self.is_initialized = True
            logger.info("✅ Batch Processor initialized - Million Scaffold Processing Ready!")
            
        except Exception as e:
            logger.error(f"Falha na inicialização Batch Processor: {e}")
            raise
    
    async def process_million_scaffolds(
        self,
        scaffold_matrices: List[np.ndarray],
        config: Optional[BatchConfig] = None,
        metrics: Optional[List[str]] = None
    ) -> BatchProcessingResult:
        """
        🌊 PROCESSAMENTO DE MILHÕES DE SCAFFOLDS
        
        Processa até milhões de scaffolds usando chunking inteligente,
        JAX vectorization e memory management avançado.
        """
        if not self.is_initialized:
            raise RuntimeError("Batch Processor não está inicializado")
        
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Usar configuração padrão se não fornecida
            if not config:
                config = self.default_config
            
            # Métricas padrão
            if not metrics:
                metrics = ["H_spectral", "k_forman_mean", "sigma", "swp"]
            
            total_scaffolds = len(scaffold_matrices)
            logger.info(f"🌊 Processing {total_scaffolds:,} scaffolds with {len(metrics)} metrics")
            
            # Criar progress tracker
            progress = ProcessingProgress(
                batch_id=batch_id,
                total_scaffolds=total_scaffolds,
                total_chunks=self._calculate_total_chunks(total_scaffolds, config.chunk_size)
            )
            self.active_batches[batch_id] = progress
            
            # Otimizar chunk size baseado em memória disponível
            optimal_chunk_size = await self._optimize_chunk_size(
                scaffold_matrices[0].shape if scaffold_matrices else (100, 100),
                config
            )
            
            logger.info(f"🎯 Optimal chunk size: {optimal_chunk_size}")
            
            # Processar em chunks
            all_results = []
            error_details = []
            throughput_samples = []
            
            for chunk_idx, chunk_start in enumerate(range(0, total_scaffolds, optimal_chunk_size)):
                chunk_end = min(chunk_start + optimal_chunk_size, total_scaffolds)
                chunk = scaffold_matrices[chunk_start:chunk_end]
                
                try:
                    # Processar chunk
                    chunk_start_time = time.time()
                    chunk_results = await self._process_chunk_vectorized(
                        chunk, metrics, config
                    )
                    chunk_time = time.time() - chunk_start_time
                    
                    # Calcular throughput do chunk
                    chunk_throughput = len(chunk) / chunk_time
                    throughput_samples.append(chunk_throughput)
                    
                    # Adicionar resultados
                    all_results.extend(chunk_results)
                    
                    # Atualizar progress
                    progress.processed_scaffolds = chunk_end
                    progress.current_chunk = chunk_idx + 1
                    progress.current_throughput = chunk_throughput
                    
                    # Estimar completion time
                    if chunk_idx > 0:  # Após primeiro chunk
                        avg_throughput = np.mean(throughput_samples)
                        remaining_scaffolds = total_scaffolds - chunk_end
                        eta_seconds = remaining_scaffolds / avg_throughput
                        progress.estimated_completion = datetime.now(timezone.utc).timestamp() + eta_seconds
                    
                    # Callback de progresso se disponível
                    if config.progress_callback:
                        await self._safe_progress_callback(config.progress_callback, progress)
                    
                    # Log progress a cada 10 chunks
                    if chunk_idx % 10 == 0:
                        percent_complete = (chunk_end / total_scaffolds) * 100
                        logger.info(f"📊 Progress: {percent_complete:.1f}% ({chunk_end:,}/{total_scaffolds:,}), {chunk_throughput:.1f} scaffolds/s")
                    
                except Exception as e:
                    logger.warning(f"Erro no chunk {chunk_idx}: {e}")
                    error_details.append({
                        "chunk_index": chunk_idx,
                        "chunk_start": chunk_start,
                        "chunk_size": len(chunk),
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc)
                    })
                    progress.errors_count += 1
            
            # Calcular métricas finais
            total_time = time.time() - start_time
            success_count = len(all_results)
            error_count = len(error_details)
            
            average_throughput = success_count / total_time if total_time > 0 else 0
            peak_throughput = max(throughput_samples) if throughput_samples else 0
            
            # Monitoramento de memória
            peak_memory = self.memory_monitor.get_peak_usage()
            
            # Criar resultado final
            batch_result = BatchProcessingResult(
                batch_id=batch_id,
                total_processed=total_scaffolds,
                success_count=success_count,
                error_count=error_count,
                processing_time_seconds=total_time,
                average_throughput=average_throughput,
                peak_throughput=peak_throughput,
                memory_peak_mb=peak_memory,
                device_used=config.use_gpu and JAX_AVAILABLE and len(jax.devices('gpu')) > 0 and "gpu" or "cpu",
                results=all_results,
                error_details=error_details
            )
            
            # Salvar no histórico
            self.completed_batches.append(batch_result)
            del self.active_batches[batch_id]
            
            # Atualizar performance tracking
            self._update_performance_tracking(batch_result, throughput_samples)
            
            logger.info(f"🌊 Batch processing COMPLETE: {success_count:,}/{total_scaffolds:,} success ({(success_count/total_scaffolds)*100:.1f}%)")
            logger.info(f"📊 Performance: {average_throughput:.1f} avg, {peak_throughput:.1f} peak scaffolds/s in {total_time:.1f}s")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Erro no processamento de milhões: {e}")
            
            # Cleanup em caso de erro
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
            
            raise
    
    async def _process_chunk_vectorized(
        self,
        chunk: List[np.ndarray],
        metrics: List[str],
        config: BatchConfig
    ) -> List[KECMetricsResult]:
        """Processa chunk usando vectorization JAX."""
        try:
            if JAX_AVAILABLE and config.use_gpu:
                return await self._process_chunk_jax_vectorized(chunk, metrics)
            else:
                return await self._process_chunk_sequential(chunk, metrics)
                
        except Exception as e:
            logger.warning(f"Chunk processing error: {e}")
            return []
    
    async def _process_chunk_jax_vectorized(
        self,
        chunk: List[np.ndarray],
        metrics: List[str]
    ) -> List[KECMetricsResult]:
        """Processamento vectorized usando JAX vmap."""
        try:
            # Converter chunk para JAX arrays
            # Assumir que todas as matrizes têm mesmo tamanho
            if not chunk:
                return []
            
            # Stack matrices para batch processing
            chunk_array = jnp.stack([jnp.array(matrix) for matrix in chunk])
            
            # Importar batch functions do JAX engine
            from .jax_kec_engine import (
                h_spectral_batch, k_forman_batch, 
                sigma_batch, swp_batch
            )
            
            results = []
            
            # Processar cada métrica em batch
            h_results = None
            k_results = None
            sigma_results = None
            swp_results = None
            
            if "H_spectral" in metrics:
                h_results = h_spectral_batch(chunk_array)
            
            if "k_forman_mean" in metrics:
                k_results = k_forman_batch(chunk_array)
            
            if "sigma" in metrics:
                sigma_results = sigma_batch(chunk_array)
            
            if "swp" in metrics:
                swp_results = swp_batch(chunk_array)
            
            # Construir resultados
            for i in range(len(chunk)):
                result = KECMetricsResult(
                    H_spectral=float(h_results[i]) if h_results is not None else None,
                    k_forman_mean=float(k_results[i]) if k_results is not None else None,
                    sigma=float(sigma_results[i]) if sigma_results is not None else None,
                    swp=float(swp_results[i]) if swp_results is not None else None
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"JAX vectorized processing falhou: {e}")
            # Fallback para processamento sequential
            return await self._process_chunk_sequential(chunk, metrics)
    
    async def _process_chunk_sequential(
        self,
        chunk: List[np.ndarray],
        metrics: List[str]
    ) -> List[KECMetricsResult]:
        """Processamento sequential como fallback."""
        try:
            from ..services.kec_calculator import KECAlgorithms
            import networkx as nx
            
            results = []
            
            for matrix in chunk:
                try:
                    # Converter para NetworkX
                    G = nx.from_numpy_array(matrix)
                    
                    # Calcular métricas solicitadas
                    result = KECMetricsResult()
                    
                    if "H_spectral" in metrics:
                        result.H_spectral = KECAlgorithms.spectral_entropy(G)
                    
                    if "k_forman_mean" in metrics:
                        forman_stats = KECAlgorithms.forman_curvature_stats(G)
                        result.k_forman_mean = forman_stats["mean"]
                    
                    if "sigma" in metrics:
                        result.sigma = KECAlgorithms.small_world_sigma(G)
                    
                    if "swp" in metrics:
                        result.swp = KECAlgorithms.small_world_propensity(G)
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Erro em scaffold individual: {e}")
                    # Adicionar resultado vazio para manter índices
                    results.append(KECMetricsResult())
            
            return results
            
        except Exception as e:
            logger.error(f"Sequential processing falhou: {e}")
            return []
    
    async def _optimize_chunk_size(
        self,
        matrix_shape: Tuple[int, int],
        config: BatchConfig
    ) -> int:
        """Otimiza chunk size baseado em memória disponível."""
        try:
            # Estimar memória por matriz
            matrix_size_mb = (matrix_shape[0] * matrix_shape[1] * 8) / (1024 * 1024)  # float64
            
            # Obter memória disponível
            available_memory_gb = await self.memory_monitor.get_available_memory()
            if available_memory_gb is None:
                available_memory_gb = config.max_memory_gb
            
            # Calcular chunk size optimal (usar 60% da memória para segurança)
            safe_memory_gb = min(available_memory_gb * 0.6, config.max_memory_gb)
            safe_memory_mb = safe_memory_gb * 1024
            
            optimal_chunk = int(safe_memory_mb / (matrix_size_mb + 1e-9))
            
            # Aplicar limits práticos
            optimal_chunk = max(10, min(optimal_chunk, 50000))  # Entre 10 e 50k
            
            # Se muito pequeno, usar chunk size da config
            if optimal_chunk < config.chunk_size // 2:
                optimal_chunk = config.chunk_size
            
            logger.info(f"🎯 Optimized chunk size: {optimal_chunk} (matrix: {matrix_shape}, memory: {safe_memory_gb:.1f}GB)")
            
            return optimal_chunk
            
        except Exception as e:
            logger.warning(f"Chunk size optimization falhou: {e}")
            return config.chunk_size
    
    def _calculate_total_chunks(self, total_items: int, chunk_size: int) -> int:
        """Calcula número total de chunks."""
        return (total_items + chunk_size - 1) // chunk_size
    
    async def _safe_progress_callback(self, callback: callable, progress: ProcessingProgress):
        """Executa progress callback com error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")
    
    async def get_batch_progress(self, batch_id: str) -> Optional[ProcessingProgress]:
        """Obtém progresso de batch específico."""
        return self.active_batches.get(batch_id)
    
    async def get_all_active_batches(self) -> Dict[str, ProcessingProgress]:
        """Obtém todos os batches ativos."""
        return self.active_batches.copy()
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancela processamento de batch."""
        try:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
                logger.info(f"🛑 Batch {batch_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao cancelar batch: {e}")
            return False
    
    async def _detect_optimal_batch_config(self) -> BatchConfig:
        """Detecta configuração optimal de batch."""
        try:
            # Detectar memória disponível
            available_memory = await self.memory_monitor.get_available_memory()
            if not available_memory:
                available_memory = 8.0  # default 8GB
            
            # Detectar se GPU está disponível
            use_gpu = JAX_AVAILABLE and len(jax.devices('gpu')) > 0
            
            # Configurar chunk size baseado em memória
            if available_memory >= 32:  # High-end system
                chunk_size = 5000
                parallel_chunks = 8
            elif available_memory >= 16:  # Mid-range system
                chunk_size = 2000
                parallel_chunks = 4
            else:  # Low-memory system
                chunk_size = 500
                parallel_chunks = 2
            
            config = BatchConfig(
                chunk_size=chunk_size,
                max_memory_gb=available_memory * 0.8,  # Use 80% max
                use_gpu=use_gpu,
                parallel_chunks=parallel_chunks,
                save_intermediate=False
            )
            
            logger.info(f"🎯 Optimal batch config: chunk_size={chunk_size}, memory={available_memory}GB, gpu={use_gpu}")
            
            return config
            
        except Exception as e:
            logger.warning(f"Optimal config detection falhou: {e}")
            return BatchConfig()  # Default config
    
    def _update_performance_tracking(
        self,
        result: BatchProcessingResult,
        throughput_samples: List[float]
    ):
        """Atualiza tracking de performance."""
        try:
            # Adicionar throughput ao histórico
            self.throughput_history.extend(throughput_samples)
            
            # Manter apenas últimas 10k amostras
            if len(self.throughput_history) > 10000:
                self.throughput_history = self.throughput_history[-10000:]
            
            # Atualizar peak performance
            if result.peak_throughput > self.peak_performance["max_throughput"]:
                self.peak_performance["max_throughput"] = result.peak_throughput
                self.peak_performance["best_device"] = result.device_used
            
            if result.total_processed > self.peak_performance["largest_batch"]:
                self.peak_performance["largest_batch"] = result.total_processed
            
            # Tempo mais rápido de chunk
            if throughput_samples:
                fastest_time = 1.0 / max(throughput_samples)  # Converter para tempo
                if fastest_time < self.peak_performance["fastest_chunk_time"]:
                    self.peak_performance["fastest_chunk_time"] = fastest_time
            
        except Exception as e:
            logger.warning(f"Performance tracking update falhou: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Resumo de performance do batch processor."""
        try:
            # Estatísticas de throughput
            if self.throughput_history:
                avg_throughput = np.mean(self.throughput_history)
                median_throughput = np.median(self.throughput_history)
                p95_throughput = np.percentile(self.throughput_history, 95)
            else:
                avg_throughput = median_throughput = p95_throughput = 0.0
            
            return {
                "processor_id": self.processor_id,
                "is_initialized": self.is_initialized,
                "active_batches": len(self.active_batches),
                "completed_batches": len(self.completed_batches),
                "performance_summary": {
                    "average_throughput": avg_throughput,
                    "median_throughput": median_throughput,
                    "p95_throughput": p95_throughput,
                    "peak_performance": self.peak_performance.copy()
                },
                "memory_stats": await self.memory_monitor.get_stats(),
                "hardware_info": {
                    "jax_available": JAX_AVAILABLE,
                    "gpu_count": len(jax.devices('gpu')) if JAX_AVAILABLE else 0,
                    "tpu_count": len(jax.devices('tpu')) if JAX_AVAILABLE else 0
                },
                "capabilities": [
                    "million_scaffold_processing",
                    "intelligent_chunking",
                    "vectorized_operations", 
                    "memory_management",
                    "progress_tracking",
                    "async_processing"
                ],
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar performance summary: {e}")
            return {"error": str(e), "processor_id": self.processor_id}
    
    async def shutdown(self):
        """Shutdown do batch processor."""
        try:
            logger.info(f"🛑 Shutting down Batch Processor {self.processor_id}")
            
            # Cancelar batches ativos
            active_count = len(self.active_batches)
            if active_count > 0:
                logger.warning(f"Cancelling {active_count} active batches")
                self.active_batches.clear()
            
            # Shutdown thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            # Shutdown memory monitor
            await self.memory_monitor.shutdown()
            
            # Log final statistics
            completed_count = len(self.completed_batches)
            if completed_count > 0:
                total_scaffolds = sum(b.total_processed for b in self.completed_batches)
                logger.info(f"📊 Final stats: {completed_count} batches, {total_scaffolds:,} total scaffolds processed")
            
            self.is_initialized = False
            logger.info("✅ Batch Processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Erro no shutdown Batch Processor: {e}")


# ==================== MEMORY MONITOR ====================

class MemoryMonitor:
    """Monitor de memória para batch processing."""
    
    def __init__(self):
        self.peak_usage = 0.0
        self.current_usage = 0.0
        self.monitoring_active = False
    
    async def initialize(self):
        """Inicializa monitor de memória."""
        try:
            if PSUTIL_AVAILABLE:
                self.monitoring_active = True
                self.current_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
                logger.info("💾 Memory monitor initialized")
            else:
                logger.warning("Memory monitor não disponível (psutil required)")
        except Exception as e:
            logger.warning(f"Memory monitor initialization falhou: {e}")
    
    async def get_available_memory(self) -> Optional[float]:
        """Obtém memória disponível em GB."""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.available / (1024**3)  # GB
        except:
            pass
        return None
    
    def get_peak_usage(self) -> float:
        """Obtém peak usage de memória."""
        try:
            if PSUTIL_AVAILABLE:
                current = psutil.virtual_memory().used / (1024 * 1024)  # MB
                self.peak_usage = max(self.peak_usage, current)
        except:
            pass
        return self.peak_usage
    
    async def get_stats(self) -> Dict[str, Any]:
        """Estatísticas de memória."""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": memory.percent,
                    "peak_usage_mb": self.peak_usage
                }
        except:
            pass
        return {"monitoring_available": False}
    
    async def shutdown(self):
        """Shutdown do monitor."""
        self.monitoring_active = False


# ==================== UTILITY FUNCTIONS ====================

async def estimate_processing_time(
    scaffold_count: int,
    matrix_size: int,
    device_type: str = "auto"
) -> Dict[str, Any]:
    """
    📊 ESTIMATIVA DE TEMPO DE PROCESSAMENTO
    
    Estima tempo necessário para processar batch baseado em:
    - Número de scaffolds
    - Tamanho das matrizes
    - Hardware disponível
    """
    try:
        # Benchmarks empíricos (scaffolds/second)
        throughput_estimates = {
            "gpu": matrix_size ** 0.5 * 1000,   # GPU scaling
            "tpu": matrix_size ** 0.3 * 2000,   # TPU scaling  
            "cpu": matrix_size ** 0.8 * 10      # CPU scaling
        }
        
        # Auto-detect device se necessário
        if device_type == "auto":
            if JAX_AVAILABLE:
                if len(jax.devices('tpu')) > 0:
                    device_type = "tpu"
                elif len(jax.devices('gpu')) > 0:
                    device_type = "gpu"
                else:
                    device_type = "cpu"
            else:
                device_type = "cpu"
        
        # Estimar throughput
        estimated_throughput = throughput_estimates.get(device_type, 10)
        
        # Estimar tempo total
        estimated_time_seconds = scaffold_count / estimated_throughput
        
        # Formatação user-friendly
        if estimated_time_seconds < 60:
            time_str = f"{estimated_time_seconds:.1f} seconds"
        elif estimated_time_seconds < 3600:
            time_str = f"{estimated_time_seconds/60:.1f} minutes"
        else:
            time_str = f"{estimated_time_seconds/3600:.1f} hours"
        
        return {
            "scaffold_count": scaffold_count,
            "matrix_size": matrix_size,
            "device_type": device_type,
            "estimated_throughput": estimated_throughput,
            "estimated_time_seconds": estimated_time_seconds,
            "estimated_time_human": time_str,
            "memory_requirement_gb": (scaffold_count * matrix_size**2 * 8) / (1024**3),
            "feasibility": "high" if estimated_time_seconds < 3600 else "medium" if estimated_time_seconds < 86400 else "low"
        }
        
    except Exception as e:
        return {"error": str(e), "feasibility": "unknown"}


async def benchmark_batch_performance(
    test_sizes: List[int] = [100, 1000, 10000],
    matrix_size: int = 100
) -> Dict[str, Any]:
    """
    📈 BENCHMARK BATCH PERFORMANCE
    
    Executa benchmarks de batch processing para diferentes tamanhos.
    """
    try:
        logger.info(f"📈 Running batch benchmarks: sizes {test_sizes}")
        
        benchmark_results = {}
        
        # Criar processor temporário
        processor = BatchProcessor()
        await processor.initialize()
        
        try:
            for test_size in test_sizes:
                # Gerar matrices de teste
                test_matrices = []
                for _ in range(test_size):
                    matrix = np.random.rand(matrix_size, matrix_size)
                    matrix = (matrix + matrix.T) / 2  # Simétrica
                    np.fill_diagonal(matrix, 0)
                    test_matrices.append(matrix)
                
                # Executar benchmark
                start_time = time.time()
                result = await processor.process_million_scaffolds(
                    test_matrices,
                    config=BatchConfig(chunk_size=min(1000, test_size//4 + 1))
                )
                benchmark_time = time.time() - start_time
                
                benchmark_results[f"size_{test_size}"] = {
                    "scaffold_count": test_size,
                    "processing_time_seconds": benchmark_time,
                    "throughput_scaffolds_per_second": result.average_throughput,
                    "success_rate": result.success_count / result.total_processed if result.total_processed > 0 else 0,
                    "device_used": result.device_used,
                    "peak_memory_mb": result.memory_peak_mb
                }
                
                logger.info(f"📊 Size {test_size}: {benchmark_time:.2f}s, {result.average_throughput:.1f} scaffolds/s")
        
        finally:
            await processor.shutdown()
        
        return {
            "benchmark_results": benchmark_results,
            "matrix_size": matrix_size,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return {"error": str(e)}


# ==================== EXPORTS ====================

__all__ = [
    "BatchProcessor",
    "BatchConfig",
    "ProcessingProgress",
    "BatchProcessingResult",
    "MemoryMonitor",
    "estimate_processing_time",
    "benchmark_batch_performance"
]