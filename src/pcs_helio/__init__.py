"""
PCS Helio - Advanced Analytics Module
====================================

Módulo para análises avançadas, integração com pcs-meta-repo e funcionalidades
de analytics de próxima geração para o backend KEC.

Componentes:
- analytics: Motor de analytics avançados
- integration: Ponte com pcs-meta-repo
- services: Serviços especializados do Helio
"""

__version__ = "1.0.0"
__author__ = "PCS Helio Team"

from .analytics import AnalyticsEngine, AnalyticsConfig
from .integration import PCSMetaBridge
from .services import HelioService

__all__ = [
    "AnalyticsEngine",
    "AnalyticsConfig", 
    "PCSMetaBridge",
    "HelioService"
]