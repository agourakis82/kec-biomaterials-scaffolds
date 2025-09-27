"""Simple RAG Engine para testes funcionais

Versão simplificada do RAG Engine que funciona sem dependências opcionais
para permitir testes básicos dos endpoints.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.rag_models import (
    UnifiedRAGRequest, UnifiedRAGResponse, SearchMethod,
    QueryDomain, PerformanceMetrics
)

logger = logging.getLogger(__name__)


class SimpleQueryResult:
    """Resultado de busca simplificado"""
    
    def __init__(self, doc_id: str, score: float, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata


class SimpleVectorStore:
    """Vector store em memória simplificado"""
    
    def __init__(self):
        self.documents = []
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Adiciona documento"""
        doc = {
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata
        }
        # Remove documento existente com mesmo ID
        self.documents = [d for d in self.documents if d["doc_id"] != doc_id]
        self.documents.append(doc)
    
    def search(self, query: str, top_k: int = 5) -> List[SimpleQueryResult]:
        """Busca simples por palavra-chave"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            content = doc["content"].lower()
            metadata = doc["metadata"]
            title = metadata.get("title", "").lower()
            
            # Score simples baseado em ocorrências
            score = 0.0
            
            # Busca na content
            if query_lower in content:
                score += 0.5
                
            # Busca no título (peso maior)
            if query_lower in title:
                score += 0.3
                
            # Busca por palavras individuais
            query_words = query_lower.split()
            for word in query_words:
                if word in content:
                    score += 0.1
                if word in title:
                    score += 0.2
            
            if score > 0:
                results.append(SimpleQueryResult(doc["doc_id"], score, metadata))
        
        # Ordena por score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_document_count(self) -> int:
        return len(self.documents)


class SimpleRAGEngine:
    """RAG Engine simplificado para testes"""
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.performance_metrics = {
            "total_queries": 0,
            "total_documents": 0,
            "avg_response_time": 0.0
        }
        
        # Adiciona alguns documentos de exemplo
        self._add_sample_documents()
    
    def _add_sample_documents(self):
        """Adiciona documentos de exemplo"""
        sample_docs = [
            {
                "content": "Biomaterial scaffolds with controlled porosity are essential for tissue engineering applications. The pore size and interconnectivity affect cell migration and nutrient transport.",
                "metadata": {
                    "title": "Porous Scaffolds in Tissue Engineering",
                    "domain": "biomaterials",
                    "source": "sample_data",
                    "url": "https://example.com/doc1"
                }
            },
            {
                "content": "Neural networks in the brain exhibit small-world properties with high clustering and short path lengths. This topology enables efficient information processing.",
                "metadata": {
                    "title": "Small-World Networks in Neuroscience",
                    "domain": "neuroscience", 
                    "source": "sample_data",
                    "url": "https://example.com/doc2"
                }
            },
            {
                "content": "Quantum entanglement is a phenomenon where particles remain connected regardless of distance. This forms the basis for quantum computing applications.",
                "metadata": {
                    "title": "Quantum Entanglement and Computing",
                    "domain": "quantum",
                    "source": "sample_data", 
                    "url": "https://example.com/doc3"
                }
            }
        ]
        
        for i, doc in enumerate(sample_docs):
            doc_id = f"sample_doc_{i+1}"
            self.vector_store.add_document(doc_id, doc["content"], doc["metadata"])
    
    async def initialize(self):
        """Inicialização básica"""
        logger.info("Simple RAG Engine initialized")
    
    async def index_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Indexa documento"""
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", hashlib.md5(content.encode()).hexdigest())
        
        full_metadata = {
            "content": content,
            "indexed_at": datetime.now().isoformat(),
            **metadata
        }
        
        self.vector_store.add_document(doc_id, content, full_metadata)
        self.performance_metrics["total_documents"] = self.vector_store.get_document_count()
        
        logger.info(f"Document indexed: {doc_id}")
        return doc_id
    
    async def search_documents(self, query: str, top_k: int = 5, domain: Optional[QueryDomain] = None) -> List[SimpleQueryResult]:
        """Busca documentos"""
        results = self.vector_store.search(query, top_k * 2)  # Busca mais para filtrar
        
        # Filtra por domínio se especificado
        if domain:
            results = [r for r in results if r.metadata.get("domain") == domain.value]
        
        return results[:top_k]
    
    async def generate_answer(self, query: str, context_results: List[SimpleQueryResult]) -> str:
        """Gera resposta simples baseada no contexto"""
        if not context_results:
            return f"I don't have specific information about '{query}' in my knowledge base."
        
        # Resposta simples baseada no melhor resultado
        best_result = context_results[0]
        title = best_result.metadata.get("title", "Unknown")
        content = best_result.metadata.get("content", "")[:200]
        
        answer = f"Based on the document '{title}', here's what I found:\n\n{content}..."
        
        if len(context_results) > 1:
            answer += f"\n\nI found {len(context_results)} related documents in total."
        
        return answer
    
    async def unified_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query unificado simplificado"""
        start_time = time.time()
        
        try:
            # Busca documentos
            results = await self.search_documents(
                request.query, 
                request.top_k, 
                request.domain
            )
            
            # Gera resposta
            answer = await self.generate_answer(request.query, results)
            
            # Prepara sources
            sources = []
            if request.include_sources:
                sources = [
                    {
                        "doc_id": r.doc_id,
                        "title": r.metadata.get("title", ""),
                        "score": r.score,
                        "url": r.metadata.get("url", ""),
                        "domain": r.metadata.get("domain", ""),
                        "source": r.metadata.get("source", "")
                    }
                    for r in results
                ]
            
            elapsed_time = (time.time() - start_time) * 1000
            self.performance_metrics["total_queries"] += 1
            
            return UnifiedRAGResponse(
                query=request.query,
                method=request.method,
                domain=request.domain,
                answer=answer,
                confidence_score=min(results[0].score, 1.0) if results else 0.0,
                sources=sources,
                cross_domain_connections=[],
                scientific_validation={},
                reasoning_trace=[],
                knowledge_graph=None,
                discovery_insights=[],
                performance_metrics={
                    "query_time_ms": elapsed_time,
                    "documents_found": len(results),
                    "total_documents": self.vector_store.get_document_count()
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in unified query: {e}")
            return UnifiedRAGResponse(
                query=request.query,
                method=request.method,
                domain=request.domain,
                answer=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                sources=[],
                cross_domain_connections=[],
                scientific_validation={},
                reasoning_trace=[],
                knowledge_graph=None,
                discovery_insights=[],
                performance_metrics={"error": True},
                timestamp=datetime.now()
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Status de saúde"""
        return {
            "healthy": True,
            "components": {
                "vector_store": True,
                "simple_engine": True
            },
            "metrics": self.performance_metrics,
            "document_count": self.vector_store.get_document_count()
        }


# Instância global
_simple_rag_engine: Optional[SimpleRAGEngine] = None


async def get_simple_rag_engine() -> SimpleRAGEngine:
    """Obtém instância do RAG Engine simples"""
    global _simple_rag_engine
    
    if _simple_rag_engine is None:
        _simple_rag_engine = SimpleRAGEngine()
        await _simple_rag_engine.initialize()
    
    return _simple_rag_engine