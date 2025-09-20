"""
Philosophy Module - Reasoning & Knowledge Systems
================================================

Módulo para sistemas de raciocínio lógico, gestão de conhecimento e 
inferência filosófica aplicada ao domínio científico.

Componentes:
- reasoning: Motor de raciocínio lógico
- knowledge: Sistema de gestão de conhecimento e ontologias
"""

__version__ = "1.0.0"
__author__ = "Philosophy Team"

from .reasoning import LogicEngine, ReasoningConfig
from .knowledge import KnowledgeBase, OntologyManager

__all__ = [
    "LogicEngine",
    "ReasoningConfig",
    "KnowledgeBase", 
    "OntologyManager"
]