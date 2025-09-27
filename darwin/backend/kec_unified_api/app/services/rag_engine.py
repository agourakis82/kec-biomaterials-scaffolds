"""RAG Engine Consolidado

Engine RAG++ unificado consolidando funcionalidades dos backends
Principal e Darwin com integração Vertex AI e busca científica avançada.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

# Setup logger early
logger = logging.getLogger(__name__)

# Importações opcionais
try:
    import chromadb
    from chromadb.api.models import Collection
    CHROMADB_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # Handle both ImportError and NumPy compatibility issues
    chromadb = None
    Collection = None
    CHROMADB_AVAILABLE = False
    logger.warning(f"ChromaDB not available: {e}")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from .vertex_ai_client import VertexAIClient, get_vertex_client
from ..models.rag_models import (
    QueryDomain, SearchMethod, SourceType,
    UnifiedRAGRequest, UnifiedRAGResponse,
    ScientificSearchRequest, BiomaterialsQueryRequest,
    CrossDomainRequest, PerformanceMetrics,
    RAGEngineConfig
)

# Logger already defined at module level


class DocumentMetadata:
    """Wrapper para metadados de documento"""
    
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.title = metadata.get("title", "")
        self.source = metadata.get("source", "")
        self.url = metadata.get("url", "")
        self.domain = metadata.get("domain", "")
        self.created_at = metadata.get("created_at", datetime.now().isoformat())
        self.doi = metadata.get("doi", "")
        self.authors = metadata.get("authors", [])
        self.abstract = metadata.get("abstract", "")
        self.citations = metadata.get("citations", 0)
        self.impact_factor = metadata.get("impact_factor", 0.0)
        self.raw_metadata = metadata


class QueryResult:
    """Resultado de busca RAG"""
    
    def __init__(self, doc_id: str, score: float, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata
        self.document = DocumentMetadata(doc_id, metadata.get("content", ""), metadata)


class VectorStore:
    """Abstração para diferentes backends de vector store"""
    
    def __init__(self, backend: str = "memory", collection_name: str = "rag_documents"):
        self.backend = backend
        self.collection_name = collection_name
        self.memory_store = []
        self.chroma_collection = None
        
        if backend == "chroma" and CHROMADB_AVAILABLE:
            self._init_chroma()
    
    def _init_chroma(self):
        """Inicializa ChromaDB"""
        try:
            client = chromadb.PersistentClient(path="./data/chroma")
            self.chroma_collection = client.get_or_create_collection(self.collection_name)
            logger.info(f"ChromaDB initialized - collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.backend = "memory"
    
    async def add_document(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Adiciona documento ao vector store"""
        if self.backend == "chroma" and self.chroma_collection:
            try:
                self.chroma_collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.error(f"ChromaDB add error: {e}")
                # Fallback to memory
                self._add_to_memory(doc_id, embedding, metadata)
        else:
            self._add_to_memory(doc_id, embedding, metadata)
    
    def _add_to_memory(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Adiciona documento ao store em memória"""
        doc = {
            "doc_id": doc_id,
            "embedding": embedding,
            "metadata": metadata
        }
        
        # Remove documento existente com mesmo ID
        self.memory_store = [d for d in self.memory_store if d["doc_id"] != doc_id]
        self.memory_store.append(doc)
    
    async def search(self, embedding: List[float], top_k: int = 5) -> List[QueryResult]:
        """Busca documentos similares"""
        if self.backend == "chroma" and self.chroma_collection:
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k
                )
                
                query_results = []
                ids = results.get("ids", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for doc_id, metadata, distance in zip(ids, metadatas, distances):
                    score = 1.0 - distance  # Convert distance to similarity
                    query_results.append(QueryResult(doc_id, score, metadata or {}))
                
                return query_results
                
            except Exception as e:
                logger.error(f"ChromaDB search error: {e}")
                # Fallback to memory search
        
        return self._search_memory(embedding, top_k)
    
    def _search_memory(self, embedding: List[float], top_k: int) -> List[QueryResult]:
        """Busca em memória usando similaridade cosseno"""
        if not self.memory_store or not NUMPY_AVAILABLE:
            return []
        
        query_vector = np.array(embedding)
        scored_docs = []
        
        for doc in self.memory_store:
            doc_vector = np.array(doc["embedding"])
            
            # Similaridade cosseno
            dot_product = np.dot(query_vector, doc_vector)
            norm_product = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                scored_docs.append((similarity, doc))
        
        # Ordena por similaridade
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append(QueryResult(doc["doc_id"], float(score), doc["metadata"]))
        
        return results
    
    def get_document_count(self) -> int:
        """Retorna número de documentos indexados"""
        if self.backend == "chroma" and self.chroma_collection:
            try:
                return self.chroma_collection.count()
            except:
                pass
        return len(self.memory_store)


class ReActReasoner:
    """Implementação do padrão ReAct para raciocínio iterativo"""
    
    def __init__(self, vertex_client: VertexAIClient):
        self.vertex_client = vertex_client
        self.max_iterations = 5
    
    async def reason(self, query: str, context_retriever, max_iterations: int = None) -> Dict[str, Any]:
        """Executa loop ReAct para raciocínio iterativo"""
        max_iterations = max_iterations or self.max_iterations
        iterations = []
        
        current_query = query
        
        for step in range(1, max_iterations + 1):
            logger.info(f"ReAct iteration {step}: {current_query[:100]}...")
            
            # THOUGHT: Análise da situação
            thought_prompt = f"""
            Analyze this research question step by step:
            Question: {current_query}
            
            Think about:
            1. What specific information do I need to answer this?
            2. What search strategy would be most effective?
            3. Are there any domain-specific considerations?
            
            Provide a clear thought process:
            """
            
            thought = await self.vertex_client.generate_text(thought_prompt)
            
            # ACTION: Busca por informações
            search_results = await context_retriever(current_query)
            
            # OBSERVATION: Análise dos resultados
            if search_results:
                context = self._format_context(search_results)
                observation_prompt = f"""
                Based on the search results below, what can I conclude about the question: {current_query}
                
                Search Results:
                {context}
                
                Observation and analysis:
                """
                
                observation = await self.vertex_client.generate_text(observation_prompt)
            else:
                observation = "No relevant documents found for this query."
            
            iteration = {
                "step": step,
                "query": current_query,
                "thought": thought,
                "action": f"search_documents({current_query})",
                "observation": observation,
                "results_count": len(search_results) if search_results else 0
            }
            
            iterations.append(iteration)
            
            # Decide se precisa continuar
            if search_results and step > 1:
                # Avalia se tem informação suficiente
                continue_prompt = f"""
                Given the information gathered so far, can I provide a comprehensive answer to: {query}
                
                Information gathered:
                {observation}
                
                Answer with only YES or NO:
                """
                
                should_continue = await self.vertex_client.generate_text(continue_prompt)
                
                if "NO" in should_continue.upper():
                    # Refina a query para próxima iteração
                    refine_prompt = f"""
                    The current search didn't fully answer: {query}
                    
                    Based on what I learned: {observation}
                    
                    Generate a more specific follow-up question to get missing information:
                    """
                    
                    current_query = await self.vertex_client.generate_text(refine_prompt)
                else:
                    break
            elif not search_results:
                # Tenta reformular a query se não achou nada
                current_query = await self._reformulate_query(current_query)
        
        return {
            "iterations": iterations,
            "final_iteration": len(iterations),
            "convergence_reason": "max_iterations" if len(iterations) >= max_iterations else "sufficient_information"
        }
    
    async def _reformulate_query(self, query: str) -> str:
        """Reformula query que não retornou resultados"""
        prompt = f"""
        This search query returned no results: {query}
        
        Suggest a broader, more general version that might find relevant documents:
        """
        
        return await self.vertex_client.generate_text(prompt)
    
    def _format_context(self, results: List[QueryResult]) -> str:
        """Formata resultados para contexto"""
        context_parts = []
        
        for i, result in enumerate(results[:5], 1):
            title = result.metadata.get("title", f"Document {i}")
            content = result.metadata.get("content", "")[:500]
            
            context_parts.append(f"{i}. {title}\nScore: {result.score:.3f}\n{content}...\n")
        
        return "\n---\n".join(context_parts)


class RAGEngine:
    """Engine RAG++ consolidado com todas as funcionalidades"""
    
    def __init__(self, config: Optional[RAGEngineConfig] = None):
        self.config = config or RAGEngineConfig()
        self.vertex_client: Optional[VertexAIClient] = None
        self.vector_store = VectorStore(
            backend=self.config.vector_backend,
            collection_name="rag_unified"
        )
        self.react_reasoner: Optional[ReActReasoner] = None
        self.performance_tracker = defaultdict(float)
        
        # Cache para otimização
        self._embedding_cache = {}
        self._response_cache = {}
    
    async def initialize(self):
        """Inicializa o RAG Engine"""
        try:
            self.vertex_client = await get_vertex_client()
            self.react_reasoner = ReActReasoner(self.vertex_client)
            logger.info("RAG Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {e}")
            raise
    
    async def index_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Indexa documento na base de conhecimento"""
        if not self.vertex_client:
            await self.initialize()
        
        start_time = time.time()
        
        # Gera ID único se não fornecido
        doc_id = metadata.get("doc_id") if metadata else None
        if not doc_id:
            doc_id = hashlib.md5(content.encode()).hexdigest()
        
        # Prepara metadados
        doc_metadata = metadata or {}
        doc_metadata.update({
            "content": content,
            "doc_id": doc_id,
            "indexed_at": datetime.now().isoformat(),
            "content_length": len(content)
        })
        
        try:
            # Gera embedding
            embedding = await self.vertex_client.get_embedding(content)
            
            # Adiciona ao vector store
            await self.vector_store.add_document(doc_id, embedding, doc_metadata)
            
            # Persiste metadados se configurado para BigQuery
            if hasattr(self.vertex_client, 'store_document'):
                await self.vertex_client.store_document(
                    doc_metadata, 
                    "rag_documents", 
                    "document_metadata"
                )
            
            elapsed_time = (time.time() - start_time) * 1000
            self.performance_tracker["indexing_time_ms"] += elapsed_time
            
            logger.info(f"Document indexed - ID: {doc_id}, Time: {elapsed_time:.2f}ms")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            raise
    
    async def search_documents(self, query: str, top_k: int = 5, domain: Optional[QueryDomain] = None) -> List[QueryResult]:
        """Busca documentos relevantes"""
        if not self.vertex_client:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Gera embedding da query
            cache_key = f"embed:{hashlib.md5(query.encode()).hexdigest()}"
            
            if cache_key in self._embedding_cache:
                query_embedding = self._embedding_cache[cache_key]
            else:
                query_embedding = await self.vertex_client.get_embedding(query)
                self._embedding_cache[cache_key] = query_embedding
            
            # Busca no vector store
            results = await self.vector_store.search(query_embedding, top_k * 2)  # Busca mais para filtrar
            
            # Filtra por domínio se especificado
            if domain:
                results = [r for r in results if r.metadata.get("domain") == domain.value]
            
            # Retorna top_k resultados
            results = results[:top_k]
            
            elapsed_time = (time.time() - start_time) * 1000
            self.performance_tracker["search_time_ms"] += elapsed_time
            
            logger.debug(f"Found {len(results)} documents in {elapsed_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def generate_answer(self, query: str, context_results: List[QueryResult], method: SearchMethod = SearchMethod.SIMPLE) -> str:
        """Gera resposta baseada no contexto recuperado"""
        if not self.vertex_client:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepara contexto
            context_blocks = []
            for i, result in enumerate(context_results, 1):
                title = result.metadata.get("title", f"Document {i}")
                content = result.metadata.get("content", "")
                source = result.metadata.get("source", "Unknown")
                
                block = f"""
                Source {i}: {title}
                From: {source}
                Score: {result.score:.3f}
                Content: {content[:800]}...
                """
                context_blocks.append(block)
            
            context = "\n---\n".join(context_blocks) if context_blocks else "No supporting documents found."
            
            # Prompt específico por método
            if method == SearchMethod.SCIENTIFIC:
                prompt = self._create_scientific_prompt(query, context)
            elif method == SearchMethod.CROSS_DOMAIN:
                prompt = self._create_cross_domain_prompt(query, context)
            else:
                prompt = self._create_standard_prompt(query, context)
            
            # Gera resposta
            answer = await self.vertex_client.generate_text(prompt)
            
            elapsed_time = (time.time() - start_time) * 1000
            self.performance_tracker["generation_time_ms"] += elapsed_time
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}"
    
    async def unified_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query unificado consolidando todos os métodos"""
        start_time = time.time()
        
        try:
            if request.method == SearchMethod.ITERATIVE:
                return await self._iterative_query(request)
            elif request.method == SearchMethod.SCIENTIFIC:
                return await self._scientific_query(request)
            elif request.method == SearchMethod.CROSS_DOMAIN:
                return await self._cross_domain_query(request)
            else:
                return await self._simple_query(request)
                
        except Exception as e:
            logger.error(f"Unified query error: {e}")
            
            # Resposta de fallback
            return UnifiedRAGResponse(
                query=request.query,
                method=request.method,
                domain=request.domain,
                answer=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                sources=[],
                performance_metrics={"error": True, "error_message": str(e)},
                total_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _simple_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query RAG simples"""
        # Busca documentos
        results = await self.search_documents(
            request.query, 
            request.top_k, 
            request.domain
        )
        
        # Gera resposta
        answer = await self.generate_answer(request.query, results, request.method)
        
        # Prepara sources se solicitado
        sources = []
        if request.include_sources:
            sources = [self._result_to_source(r) for r in results]
        
        return UnifiedRAGResponse(
            query=request.query,
            method=request.method,
            domain=request.domain,
            answer=answer,
            confidence_score=self._calculate_confidence(results),
            sources=sources,
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _iterative_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query RAG iterativo com ReAct"""
        if not self.react_reasoner:
            return await self._simple_query(request)
        
        # Define function de busca para ReAct
        async def context_retriever(query: str) -> List[QueryResult]:
            return await self.search_documents(query, request.top_k, request.domain)
        
        # Executa ReAct reasoning
        react_result = await self.react_reasoner.reason(
            request.query, 
            context_retriever,
            request.max_iterations or 3
        )
        
        # Busca final com query refinada
        final_query = react_result["iterations"][-1]["query"] if react_result["iterations"] else request.query
        final_results = await self.search_documents(final_query, request.top_k, request.domain)
        
        # Gera resposta final
        final_answer = await self.generate_answer(request.query, final_results, request.method)
        
        sources = []
        if request.include_sources:
            sources = [self._result_to_source(r) for r in final_results]
        
        return UnifiedRAGResponse(
            query=request.query,
            method=request.method,
            domain=request.domain,
            answer=final_answer,
            confidence_score=self._calculate_confidence(final_results),
            sources=sources,
            reasoning_trace=react_result["iterations"],
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _scientific_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query científico especializado"""
        # Busca com validação científica
        results = await self.search_documents(request.query, request.top_k * 2, request.domain)
        
        # Filtra resultados com validação científica
        if request.scientific_validation:
            validated_results = []
            for result in results:
                if self._validate_scientific_source(result):
                    validated_results.append(result)
            results = validated_results[:request.top_k]
        
        # Gera resposta científica
        answer = await self.generate_answer(request.query, results, SearchMethod.SCIENTIFIC)
        
        sources = []
        if request.include_sources:
            sources = [self._result_to_source(r) for r in results]
        
        # Validação científica adicional
        scientific_validation = {
            "peer_reviewed_sources": sum(1 for r in results if r.metadata.get("peer_reviewed", False)),
            "doi_available": sum(1 for r in results if r.metadata.get("doi")),
            "avg_impact_factor": np.mean([r.metadata.get("impact_factor", 0) for r in results]) if results else 0
        }
        
        return UnifiedRAGResponse(
            query=request.query,
            method=request.method,
            domain=request.domain,
            answer=answer,
            confidence_score=self._calculate_confidence(results),
            sources=sources,
            scientific_validation=scientific_validation,
            performance_metrics=self._get_performance_metrics()
        )
    
    async def _cross_domain_query(self, request: UnifiedRAGRequest) -> UnifiedRAGResponse:
        """Query cross-domain interdisciplinar"""
        # Implementação simplificada - busca em domínios relacionados
        all_results = []
        
        # Busca no domínio primário
        if request.domain:
            primary_results = await self.search_documents(request.query, request.top_k, request.domain)
            all_results.extend(primary_results)
        
        # Busca em outros domínios
        other_domains = [d for d in QueryDomain if d != request.domain][:2]  # Limita a 2 outros domínios
        
        for domain in other_domains:
            domain_results = await self.search_documents(request.query, 2, domain)  # Menos resultados por domínio
            all_results.extend(domain_results)
        
        # Ordena por relevância
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:request.top_k]
        
        # Gera resposta cross-domain
        answer = await self.generate_answer(request.query, final_results, SearchMethod.CROSS_DOMAIN)
        
        sources = []
        if request.include_sources:
            sources = [self._result_to_source(r) for r in final_results]
        
        # Analisa conexões cross-domain
        cross_domain_connections = self._analyze_cross_domain_connections(final_results)
        
        return UnifiedRAGResponse(
            query=request.query,
            method=request.method,
            domain=request.domain,
            answer=answer,
            confidence_score=self._calculate_confidence(final_results),
            sources=sources,
            cross_domain_connections=cross_domain_connections,
            performance_metrics=self._get_performance_metrics()
        )
    
    def _create_standard_prompt(self, query: str, context: str) -> str:
        """Cria prompt padrão para geração"""
        return f"""
        You are DARWIN, a research assistant specializing in biomaterials and scientific research.
        
        Answer the following question using the provided context. Be factual, concise, and cite relevant sources.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
    
    def _create_scientific_prompt(self, query: str, context: str) -> str:
        """Cria prompt para respostas científicas"""
        return f"""
        You are DARWIN, a scientific research assistant. Provide a rigorous, evidence-based answer.
        
        Guidelines:
        - Use scientific terminology appropriately
        - Cite specific studies and sources
        - Mention limitations and uncertainties
        - Include quantitative data when available
        
        Scientific Context:
        {context}
        
        Research Question: {query}
        
        Scientific Analysis:
        """
    
    def _create_cross_domain_prompt(self, query: str, context: str) -> str:
        """Cria prompt para análise cross-domain"""
        return f"""
        You are DARWIN, analyzing connections across scientific domains.
        
        The context includes information from multiple research domains. Identify:
        - Common principles across domains
        - Analogies and parallels
        - Potential cross-domain applications
        - Interdisciplinary insights
        
        Multi-domain Context:
        {context}
        
        Question: {query}
        
        Interdisciplinary Analysis:
        """
    
    def _validate_scientific_source(self, result: QueryResult) -> bool:
        """Valida se fonte é cientificamente confiável"""
        metadata = result.metadata
        
        # Critérios de validação
        has_doi = bool(metadata.get("doi"))
        is_peer_reviewed = metadata.get("peer_reviewed", False)
        has_impact_factor = metadata.get("impact_factor", 0) > 0
        is_recent = True  # Simplificado - validar data
        
        # Score mínimo de relevância
        min_score = result.score > 0.5
        
        return min_score and (has_doi or is_peer_reviewed or has_impact_factor)
    
    def _analyze_cross_domain_connections(self, results: List[QueryResult]) -> List[Dict[str, Any]]:
        """Analisa conexões entre domínios"""
        connections = []
        domains_found = set()
        
        for result in results:
            domain = result.metadata.get("domain")
            if domain:
                domains_found.add(domain)
        
        # Identifica conexões potenciais
        for domain in domains_found:
            domain_results = [r for r in results if r.metadata.get("domain") == domain]
            if domain_results:
                connections.append({
                    "domain": domain,
                    "document_count": len(domain_results),
                    "avg_relevance": np.mean([r.score for r in domain_results]),
                    "top_concepts": [r.metadata.get("title", "")[:50] for r in domain_results[:2]]
                })
        
        return connections
    
    def _calculate_confidence(self, results: List[QueryResult]) -> float:
        """Calcula score de confiança da resposta"""
        if not results:
            return 0.0
        
        # Baseado na qualidade e quantidade dos resultados
        avg_score = np.mean([r.score for r in results])
        result_count_factor = min(len(results) / 5, 1.0)  # Máximo boost para 5+ resultados
        
        confidence = avg_score * result_count_factor
        return min(confidence, 1.0)
    
    def _result_to_source(self, result: QueryResult) -> Dict[str, Any]:
        """Converte QueryResult para formato de source"""
        return {
            "doc_id": result.doc_id,
            "title": result.metadata.get("title", ""),
            "source": result.metadata.get("source", ""),
            "url": result.metadata.get("url", ""),
            "score": result.score,
            "domain": result.metadata.get("domain", ""),
            "doi": result.metadata.get("doi", ""),
            "abstract": result.metadata.get("abstract", "")[:200]
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance"""
        return {
            "indexing_time_ms": self.performance_tracker.get("indexing_time_ms", 0),
            "search_time_ms": self.performance_tracker.get("search_time_ms", 0),
            "generation_time_ms": self.performance_tracker.get("generation_time_ms", 0),
            "document_count": self.vector_store.get_document_count(),
            "cache_size": len(self._embedding_cache)
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Verifica saúde do RAG Engine"""
        try:
            # Testa componentes principais
            vertex_health = await self.vertex_client.health_check() if self.vertex_client else {"healthy": False}
            
            return {
                "healthy": vertex_health.get("healthy", False),
                "components": {
                    "vertex_ai": vertex_health.get("healthy", False),
                    "vector_store": self.vector_store.get_document_count() >= 0,
                    "react_reasoner": self.react_reasoner is not None,
                    "embedding_cache": len(self._embedding_cache) >= 0
                },
                "metrics": self._get_performance_metrics(),
                "document_count": self.vector_store.get_document_count()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "components": {"error": True}
            }


# Instância global singleton
_rag_engine: Optional[RAGEngine] = None


async def get_rag_engine() -> RAGEngine:
    """Obtém instância global do RAG Engine"""
    global _rag_engine
    
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        await _rag_engine.initialize()
    
    return _rag_engine


async def cleanup_rag_engine():
    """Limpa instância global do RAG Engine"""
    global _rag_engine
    _rag_engine = None