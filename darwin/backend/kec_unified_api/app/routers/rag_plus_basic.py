"""RAG++ Basic Router

Router RAG++ básico funcional sem dependências opcionais,
usando apenas funcionalidades core para demonstração.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/rag-plus",
    tags=["RAG++ Enhanced Research"],
)

# Basic models
class BasicRAGRequest(BaseModel):
    query: str = Field(..., description="Pergunta de pesquisa")
    top_k: int = Field(5, description="Número de documentos")
    include_sources: bool = Field(True, description="Incluir fontes")

class BasicRAGResponse(BaseModel):
    query: str = Field(..., description="Query original")
    answer: str = Field(..., description="Resposta gerada")
    method: str = Field("basic", description="Método usado")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class BasicHealthStatus(BaseModel):
    healthy: bool = Field(True)
    components: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field("")
    timestamp: datetime = Field(default_factory=datetime.now)

# In-memory knowledge base para demonstração
SAMPLE_KNOWLEDGE_BASE = [
    {
        "doc_id": "bio_001",
        "title": "Biomaterial Scaffolds for Tissue Engineering",
        "content": "Biomaterial scaffolds with controlled porosity are essential for tissue engineering applications. The pore size and interconnectivity significantly affect cell migration, nutrient transport, and tissue regeneration. Scaffolds with porosity between 70-90% and pore sizes of 100-500 μm show optimal performance for bone tissue engineering.",
        "domain": "biomaterials",
        "source": "Scientific Literature",
        "url": "https://example.com/bio001",
        "keywords": ["scaffold", "porosity", "tissue engineering", "biomaterial"]
    },
    {
        "doc_id": "neuro_001", 
        "title": "Neural Network Topology and Small-World Properties",
        "content": "Neural networks in the brain exhibit small-world network properties characterized by high clustering coefficients and short path lengths. This topology enables efficient information processing and robust connectivity. The balance between local clustering and global connectivity is crucial for cognitive functions.",
        "domain": "neuroscience",
        "source": "Neuroscience Research",
        "url": "https://example.com/neuro001",
        "keywords": ["neural network", "small-world", "topology", "connectivity"]
    },
    {
        "doc_id": "quantum_001",
        "title": "Quantum Entanglement and Information Processing",
        "content": "Quantum entanglement represents a fundamental quantum mechanical phenomenon where particles remain correlated regardless of spatial separation. This property forms the foundation for quantum computing, quantum cryptography, and quantum communication protocols. Understanding entanglement is crucial for developing quantum technologies.",
        "domain": "quantum",
        "source": "Physics Literature", 
        "url": "https://example.com/quantum001",
        "keywords": ["quantum", "entanglement", "information", "computing"]
    },
    {
        "doc_id": "phil_001",
        "title": "Epistemology and Scientific Knowledge",
        "content": "Epistemology examines the nature, sources, and limits of knowledge. In scientific contexts, epistemological considerations address how we acquire reliable knowledge about the natural world. The relationship between observation, theory, and truth remains a central philosophical concern in understanding scientific methodology.",
        "domain": "philosophy",
        "source": "Philosophy Literature",
        "url": "https://example.com/phil001", 
        "keywords": ["epistemology", "knowledge", "science", "methodology"]
    }
]

def simple_search(query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Busca simples por palavra-chave"""
    query_lower = query.lower()
    results = []
    
    for doc in SAMPLE_KNOWLEDGE_BASE:
        # Filtra por domínio se especificado
        if domain and doc.get("domain") != domain:
            continue
            
        # Calcula score simples
        score = 0.0
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        
        # Busca exata na query
        if query_lower in content_lower:
            score += 0.5
        if query_lower in title_lower:
            score += 0.3
            
        # Busca por palavras individuais
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 2:  # Ignore palavras muito pequenas
                if word in content_lower:
                    score += 0.1
                if word in title_lower:
                    score += 0.2
                if word in str(doc.get("keywords", [])).lower():
                    score += 0.15
        
        if score > 0:
            result = doc.copy()
            result["score"] = round(score, 3)
            results.append(result)
    
    # Ordena por score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def generate_basic_answer(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """Gera resposta básica baseada no contexto"""
    if not context_docs:
        return f"Não encontrei informações específicas sobre '{query}' na base de conhecimento. Você pode tentar reformular sua pergunta ou adicionar mais documentos à base."
    
    # Usa o melhor resultado
    best_doc = context_docs[0]
    title = best_doc["title"]
    content = best_doc["content"]
    score = best_doc["score"]
    
    # Resposta básica estruturada
    answer = f"Com base no documento '{title}' (relevância: {score:.1%}), posso fornecer as seguintes informações:\n\n"
    
    # Extrai informação relevante do conteúdo
    sentences = content.split('. ')
    relevant_sentences = []
    
    query_words = query.lower().split()
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in query_words if len(word) > 2):
            relevant_sentences.append(sentence.strip())
    
    if relevant_sentences:
        answer += '. '.join(relevant_sentences[:2]) + "."
    else:
        answer += content[:300] + "..."
    
    if len(context_docs) > 1:
        answer += f"\n\nEncontrei {len(context_docs)} documentos relevantes no total."
    
    return answer

# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/health", response_model=BasicHealthStatus)
async def health_check():
    """Health check para RAG++ básico"""
    return BasicHealthStatus(
        healthy=True,
        components={
            "basic_rag": True,
            "knowledge_base": True,
            "sample_documents": len(SAMPLE_KNOWLEDGE_BASE)
        },
        message="RAG++ basic mode operational"
    )

@router.post("/query", response_model=BasicRAGResponse)
async def query_basic_rag(request: BasicRAGRequest):
    """
    Query RAG++ básico com busca por palavra-chave.
    
    Versão simplificada que funciona com base de conhecimento em memória
    e busca por palavra-chave para demonstrar funcionalidade.
    """
    try:
        logger.info(f"Basic RAG query: {request.query[:100]}...")
        start_time = time.time()
        
        # Busca documentos
        results = simple_search(request.query, request.top_k)
        
        # Gera resposta
        answer = generate_basic_answer(request.query, results)
        
        # Prepara sources se solicitado
        sources = []
        if request.include_sources:
            sources = [
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"], 
                    "score": doc["score"],
                    "domain": doc["domain"],
                    "url": doc["url"],
                    "source": doc["source"]
                }
                for doc in results
            ]
        
        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"Basic RAG query completed in {elapsed_time:.2f}ms")
        
        return BasicRAGResponse(
            query=request.query,
            answer=answer,
            method="basic_keyword_search",
            sources=sources,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in basic RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    domain: Optional[str] = Query(None, description="Domain filter")
):
    """
    Busca direta na base de conhecimento básica.
    
    Retorna documentos relevantes sem geração de resposta.
    """
    try:
        results = simple_search(query, top_k, domain)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "score": doc["score"],
                "domain": doc["domain"],
                "content_preview": doc["content"][:200] + "...",
                "url": doc["url"]
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "domain_filter": domain,
            "available_domains": list(set(doc["domain"] for doc in SAMPLE_KNOWLEDGE_BASE))
        }
        
    except Exception as e:
        logger.error(f"Error in basic search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/status")
async def get_basic_status():
    """Status básico do serviço RAG++"""
    return {
        "service": "rag_plus_basic",
        "status": "operational", 
        "mode": "basic",
        "components": {
            "knowledge_base": "operational",
            "search_engine": "operational",
            "answer_generation": "operational"
        },
        "configuration": {
            "total_documents": len(SAMPLE_KNOWLEDGE_BASE),
            "available_domains": list(set(doc["domain"] for doc in SAMPLE_KNOWLEDGE_BASE)),
            "search_method": "keyword_based"
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/knowledge-graph")
async def get_basic_knowledge_graph(
    domain: Optional[str] = Query(None, description="Domain filter"),
    max_nodes: int = Query(20, description="Maximum nodes")
):
    """Grafo de conhecimento básico"""
    try:
        # Filtra documentos por domínio se especificado
        docs = SAMPLE_KNOWLEDGE_BASE
        if domain:
            docs = [doc for doc in docs if doc["domain"] == domain]
        
        # Cria nodes básicos
        nodes = []
        for doc in docs[:max_nodes]:
            nodes.append({
                "id": doc["doc_id"],
                "label": doc["title"][:30],
                "type": "document",
                "domain": doc["domain"],
                "keywords": doc.get("keywords", [])
            })
        
        # Cria edges básicos baseados em domínios compartilhados
        edges = []
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if node1["domain"] == node2["domain"]:
                    edges.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "weight": 0.5,
                        "type": "domain_similarity"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "domains": list(set(node["domain"] for node in nodes)),
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph failed: {str(e)}")

@router.get("/config")
async def get_basic_config():
    """Configuração básica do serviço"""
    return {
        "mode": "basic",
        "version": "1.0.0-basic",
        "knowledge_base": {
            "total_documents": len(SAMPLE_KNOWLEDGE_BASE),
            "domains": list(set(doc["domain"] for doc in SAMPLE_KNOWLEDGE_BASE))
        },
        "capabilities": {
            "keyword_search": True,
            "basic_answer_generation": True,
            "domain_filtering": True,
            "knowledge_graph": True,
            "vertex_ai": False,
            "advanced_rag": False,
            "scientific_validation": False
        },
        "supported_domains": ["biomaterials", "neuroscience", "quantum", "philosophy"],
        "timestamp": datetime.now().isoformat()
    }

@router.post("/documents")
async def add_document_basic(
    content: str = Query(..., description="Document content"),
    title: str = Query(..., description="Document title"),
    domain: str = Query("general", description="Document domain"),
    source: str = Query("manual", description="Document source")
):
    """Adiciona documento à base básica (simulado)"""
    try:
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Simula adição (na implementação real, persistiria)
        new_doc = {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "domain": domain,
            "source": source,
            "url": f"manual/{doc_id}",
            "added_at": datetime.now().isoformat()
        }
        
        return {
            "status": "added",
            "document_id": doc_id,
            "message": f"Document '{title}' added successfully (simulated)",
            "content_length": len(content),
            "domain": domain
        }
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")