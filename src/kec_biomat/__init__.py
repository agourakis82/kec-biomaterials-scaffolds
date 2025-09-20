"""
KEC Biomat - Biomaterials Analysis Core
=======================================

Módulo principal para análise de biomateriais porosos usando métricas KEC 2.0:
- H: Entropia espectral (von Neumann)
- κ: Curvatura de Forman (2-complex)
- σ/ϕ: Small-world metrics (sigma e SWP)
- σ_Q: Coerência quântica (opcional)

Migrado e consolidado de kec_biomat_pack_2025-09-19 e kec_biomat_api.logic
"""

__version__ = "2.0.0"
__author__ = "KEC Biomaterials Team"

from .metrics import kec_metrics, entropy, percolation
from .processing import pipeline
from .configs import load_config

__all__ = [
    "kec_metrics",
    "entropy", 
    "percolation",
    "pipeline",
    "load_config"
]