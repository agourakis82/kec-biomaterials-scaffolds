"""
KEC Biomat Tests
===============

Testes para o módulo kec_biomat incluindo métricas, processamento e configurações.
"""

# Importações principais para facilitar testes
from .test_kec_metrics import test_kec_metrics_basic, test_entropy_calculation
from .test_config import test_config_loading
from .test_pipeline import test_processing_pipeline

__all__ = [
    "test_kec_metrics_basic",
    "test_entropy_calculation", 
    "test_config_loading",
    "test_processing_pipeline"
]