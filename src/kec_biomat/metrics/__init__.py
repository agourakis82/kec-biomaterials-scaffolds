"""
KEC Biomat Metrics Module
=========================

Métricas para análise de biomateriais porosos:
- kec_metrics: Implementação KEC 2.0 (H, κ, σ, ϕ, σ_Q)
- entropy: Cálculos de entropia 
- percolation: Análise de percolação
"""

from .kec_metrics import compute_kec_metrics, spectral_entropy, forman_curvature_stats
from .kec_metrics import small_world_sigma, small_world_propensity, quantum_coherence_sigma

# Re-export para manter compatibilidade
__all__ = [
    "compute_kec_metrics",
    "spectral_entropy", 
    "forman_curvature_stats",
    "small_world_sigma",
    "small_world_propensity", 
    "quantum_coherence_sigma"
]