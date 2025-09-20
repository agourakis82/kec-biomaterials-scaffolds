"""
RAG (Retrieval-Augmented Generation Plus) Module
================================================

Módulo responsável por funcionalidades RAG++, busca iterativa e integração com Vertex AI.

Componentes:
- rag_plus: Implementação principal do RAG++
- iterative: Busca iterativa e refinamento
- vertex: Integração com Google Vertex AI
"""

from .rag_plus import RAGPlusEngine
from .iterative import IterativeSearch
from .vertex import VertexAIIntegration

__all__ = ["RAGPlusEngine", "IterativeSearch", "VertexAIIntegration"]