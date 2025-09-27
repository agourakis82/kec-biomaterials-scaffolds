"""DARWIN Monitoring Module - Sistema Épico de Observabilidade

🚀 MONITORING REVOLUTIONARY - OBSERVABILIDADE BEYOND STATE-OF-THE-ART
Módulo completo de monitoring para o sistema DARWIN AutoGen + JAX.

Exports principais:
- DarwinMonitoring: Engine principal de monitoring
- get_monitoring(): Factory function para instância global
- record_performance(): Helper para registrar performance JAX
- log_epic_event(): Helper para logging estruturado

Technology: Real-time metrics + Structured logging + Intelligent alerting
"""

from .darwin_monitoring import (
    DarwinMonitoring,
    PerformanceMetric,
    LogEntry,
    AlertCondition,
    Alert,
    get_monitoring,
    record_performance,
    log_epic_event
)

__all__ = [
    "DarwinMonitoring",
    "PerformanceMetric",
    "LogEntry", 
    "AlertCondition",
    "Alert",
    "get_monitoring",
    "record_performance",
    "log_epic_event"
]