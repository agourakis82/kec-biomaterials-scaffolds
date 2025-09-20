"""
Darwin Core - RAG++, Tree Search & Memory System
===============================================

Módulo principal para funcionalidades de RAG++ (Retrieval-Augmented Generation Plus),
sistemas de busca em árvore (Tree Search), gerenciamento de memória e discovery engine.

Componentes:
- rag: RAG++ implementation, iterative search, Vertex AI integration
- tree_search: PUCT, Monte Carlo Tree Search, algoritmos de busca
- memory: Session management, context caching, persistent storage
- discovery: Discovery engine para análise exploratória
"""

__version__ = "1.0.0"
__author__ = "KEC Biomaterials Team"

# Importações principais para facilitar o uso
from .rag import rag_plus, iterative, vertex
from .tree_search import puct, mcts, algorithms
from .memory import session, context_cache, persistent
from .discovery import engine

__all__ = [
    "rag_plus",
    "iterative", 
    "vertex",
    "puct",
    "mcts",
    "algorithms",
    "session",
    "context_cache",
    "persistent",
    "engine"
]