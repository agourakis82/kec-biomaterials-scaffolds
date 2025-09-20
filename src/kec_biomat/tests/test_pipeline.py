"""
Testes para pipeline de processamento KEC
========================================
"""

import numpy as np
import pytest

from ..processing import KECProcessingPipeline, ProcessingConfig, ProcessingResults
from ..configs import KECConfig


def test_processing_pipeline():
    """Teste básico do pipeline de processamento."""
    
    # Configuração simples
    config = KECConfig()
    config.seed = 42
    
    pipeline = KECProcessingPipeline(config)
    
    # Imagem sintética pequena
    test_image = np.random.rand(64, 64) > 0.4
    
    # Configuração de processamento
    proc_config = ProcessingConfig(
        input_path="test",
        output_path="test_out",
        save_intermediates=True
    )
    
    # Executa pipeline (modo async simulado)
    import asyncio
    results = asyncio.run(pipeline.process_image(test_image, proc_config))
    
    # Verifica resultados
    assert isinstance(results, ProcessingResults)
    assert "H_spectral" in results.kec_metrics
    assert results.graph.number_of_nodes() >= 0
    assert results.processing_time > 0
    assert len(results.errors) == 0


if __name__ == "__main__":
    test_processing_pipeline()
    print("✅ Teste de pipeline passou!")