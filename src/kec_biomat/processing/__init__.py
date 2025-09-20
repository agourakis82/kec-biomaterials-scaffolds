"""
KEC Biomat Processing Pipeline
=============================

Pipeline de processamento para análise de biomateriais porosos.
Integra segmentação, extração de grafos, cálculo de métricas e modelagem.
"""

from .pipeline import KECProcessingPipeline, ProcessingConfig, ProcessingResults

__all__ = ["KECProcessingPipeline", "ProcessingConfig", "ProcessingResults"]