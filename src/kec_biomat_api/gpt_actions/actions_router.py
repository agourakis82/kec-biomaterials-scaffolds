"""
GPT Actions Router - Endpoints Otimizados para GPT Actions
=========================================================

Router principal com endpoints específicos e otimizados para integração
com ChatGPT Actions, incluindo schemas simplificados e respostas estruturadas.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import asyncio

# Imports dos novos módulos modulares
import sys
sys.path.append('/app/src')

from darwin_core.rag.rag_plus import RAGPlusEngine, RAGPlusConfig
from darwin_core.tree_search.puct import PUCTSearch, PUCTConfig
from darwin_core.memory.integrated_memory_system import get_integrated_memory_system
from kec_biomat.metrics.kec_metrics import compute_kec_metrics
from kec_biomat.processing.pipeline import KECProcessingPipeline
from pcs_helio.services.helio_service import HelioService

logger = logging.getLogger(__name__)

# Router para GPT Actions
gpt_actions_router = APIRouter(prefix="/gpt-actions", tags=["GPT Actions"])


# ================ SCHEMAS PARA GPT ACTIONS ================

class KECAnalysisRequest(BaseModel):
    """Request para análise KEC de biomateriais."""
    image_data: Optional[str] = Field(None, description="Base64 image data ou URL")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data em formato NetworkX JSON")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Override de configuração KEC")
    
    class Config:
        schema_extra = {
            "example": {
                "graph_data": {"nodes": [{"id": 0}, {"id": 1}], "edges": [{"source": 0, "target": 1}]},
                "config_overrides": {"k_eigs": 64, "sigma_Q": True}
            }
        }


class RAGQueryRequest(BaseModel):
    """Request para query RAG++."""
    query: str = Field(..., description="Pergunta ou query para busca")
    use_iterative: bool = Field(False, description="Usar busca iterativa")
    max_sources: int = Field(5, description="Máximo de fontes a retornar")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "latest advances in biomaterial scaffolds",
                "use_iterative": True,
                "max_sources": 10
            }
        }


class TreeSearchRequest(BaseModel):
    """Request para tree search PUCT."""
    initial_state: str = Field(..., description="Estado inicial para busca")
    budget: int = Field(200, description="Budget de nós para explorar")
    max_depth: int = Field(5, description="Profundidade máxima")
    
    class Config:
        schema_extra = {
            "example": {
                "initial_state": "optimize_scaffold_design",
                "budget": 500,
                "max_depth": 8
            }
        }


class MemoryQueryRequest(BaseModel):
    """Request para consulta de memória."""
    query: str = Field(..., description="Query para buscar em memória")
    memory_type: str = Field("conversation", description="Tipo: conversation, project, discovery")
    time_window_days: int = Field(30, description="Janela temporal em dias")


# ================ ENDPOINTS GPT ACTIONS ================

@gpt_actions_router.post("/analyze-kec-metrics")
async def analyze_kec_metrics_gpt(request: KECAnalysisRequest) -> Dict[str, Any]:
    """
    🧬 Análise KEC de Biomateriais para GPT Actions
    
    Calcula métricas KEC (H, κ, σ, ϕ, σ_Q) para estruturas porosas.
    """
    try:
        # Se tiver dados de grafo, usa diretamente
        if request.graph_data:
            import networkx as nx
            G = nx.node_link_graph(request.graph_data)
        else:
            # Senão, cria grafo exemplo para demonstração
            import networkx as nx
            G = nx.erdos_renyi_graph(50, 0.1, seed=42)
        
        # Configura parâmetros
        config_overrides = request.config_overrides or {}
        
        # Calcula métricas KEC
        metrics = compute_kec_metrics(
            G,
            spectral_k=config_overrides.get("k_eigs", 64),
            include_triangles=config_overrides.get("include_triangles", True),
            n_random=config_overrides.get("n_random", 20),
            sigma_q=config_overrides.get("sigma_Q", False)
        )
        
        # Adiciona contexto interpretativo para GPT
        interpretation = _interpret_kec_metrics(metrics, G)
        
        return {
            "success": True,
            "kec_metrics": metrics,
            "graph_info": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": f"{nx.density(G):.4f}",
                "connected": nx.is_connected(G)
            },
            "interpretation": interpretation,
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in KEC analysis: {e}")
        raise HTTPException(status_code=500, detail=f"KEC analysis failed: {str(e)}")


@gpt_actions_router.post("/rag-query")
async def rag_query_gpt(request: RAGQueryRequest) -> Dict[str, Any]:
    """
    🔍 RAG++ Query para GPT Actions
    
    Busca inteligente usando RAG++ com Vertex AI e BigQuery.
    """
    try:
        # Inicializa RAG engine
        config = RAGPlusConfig(
            project_id="kec-biomaterials-prod",  # Será configurado via env
            location="us-central1",
            dataset_id="kec_knowledge",
            table_id="documents"
        )
        
        rag_engine = RAGPlusEngine(config)
        await rag_engine.initialize()
        
        # Executa query
        if request.use_iterative:
            # Usar módulo de busca iterativa
            from darwin_core.rag.iterative import IterativeSearch, IterativeConfig
            iterative_search = IterativeSearch(
                IterativeConfig(max_iterations=3),
                rag_engine=rag_engine
            )
            result = await iterative_search.search_iteratively(request.query)
        else:
            # RAG++ simples
            result = await rag_engine.answer_question(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"][:request.max_sources],
            "method": result["method"],
            "retrieved_docs": result["retrieved_docs"],
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_answer": f"Busca por '{request.query}' - Sistema em modo local/desenvolvimento"
        }


@gpt_actions_router.post("/tree-search")
async def tree_search_gpt(request: TreeSearchRequest) -> Dict[str, Any]:
    """
    🌳 Tree Search PUCT para GPT Actions
    
    Executa busca em árvore usando algoritmo PUCT.
    """
    try:
        from darwin_core.tree_search.algorithms import StringStateEvaluator
        
        # Configura PUCT
        config = PUCTConfig(
            default_budget=request.budget,
            max_depth=request.max_depth,
            c_puct=1.414
        )
        
        evaluator = StringStateEvaluator(branching_factor=3, max_length=20)
        puct_search = PUCTSearch(evaluator, config)
        
        # Executa busca
        result_node = await puct_search.search(request.initial_state, budget=request.budget)
        
        # Extrai resultados
        best_sequence = puct_search.get_best_action_sequence(max_length=10)
        action_probs = puct_search.get_action_probabilities()
        stats = puct_search.get_search_statistics()
        
        return {
            "success": True,
            "initial_state": request.initial_state,
            "best_action_sequence": best_sequence,
            "action_probabilities": action_probs,
            "search_statistics": stats,
            "root_value": result_node.mean_value,
            "nodes_explored": stats["nodes_explored"],
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in tree search: {e}")
        raise HTTPException(status_code=500, detail=f"Tree search failed: {str(e)}")


@gpt_actions_router.post("/memory-query")
async def memory_query_gpt(request: MemoryQueryRequest) -> Dict[str, Any]:
    """
    🧠 Query de Memória para GPT Actions
    
    Busca em sistema de memória (conversações, projeto, descobertas).
    """
    try:
        memory_system = await get_integrated_memory_system()
        
        if request.memory_type == "conversation":
            # Busca em histórico de conversações
            conversations = await memory_system.conversation_memory.retrieve_relevant_context(
                query=request.query,
                max_results=10,
                time_window_days=request.time_window_days
            )
            
            results = [
                {
                    "timestamp": conv.timestamp.isoformat(),
                    "llm_provider": conv.llm_provider,
                    "context_type": conv.context_type,
                    "summary": conv.user_message[:200],
                    "relevance": conv.relevance_score
                }
                for conv in conversations
            ]
            
        elif request.memory_type == "discovery":
            # Busca em descobertas científicas
            findings = await memory_system.scientific_discovery.get_recent_discoveries(
                hours=request.time_window_days * 24,
                min_relevance=0.5
            )
            
            results = [
                {
                    "title": finding.title,
                    "abstract": finding.abstract[:300],
                    "source": finding.source,
                    "url": finding.url,
                    "relevance": finding.relevance_score,
                    "novelty": finding.novelty_score,
                    "category": finding.category
                }
                for finding in findings[:10]
            ]
            
        else:  # project
            # Contexto completo do projeto
            project_context = await memory_system.get_complete_project_context()
            results = [project_context]
        
        return {
            "success": True,
            "query": request.query,
            "memory_type": request.memory_type,
            "results": results,
            "total_found": len(results),
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in memory query: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": request.query
        }


@gpt_actions_router.get("/project-status")
async def get_project_status_gpt() -> Dict[str, Any]:
    """
    📊 Status Completo do Projeto para GPT Actions
    
    Retorna status atual, progresso, próximos passos e contexto completo.
    """
    try:
        memory_system = await get_integrated_memory_system()
        
        # Contexto completo
        project_context = await memory_system.get_complete_project_context()
        
        # Health dos sistemas
        system_health = await memory_system.get_system_health()
        
        # Resumo das últimas 24h
        conversation_summary = await memory_system.conversation_memory.get_conversation_summary(days=1)
        discovery_summary = await memory_system.scientific_discovery.generate_discovery_report(hours=24)
        
        return {
            "success": True,
            "project_overview": {
                "name": "KEC Biomaterials Scaffolds",
                "backend_type": "RAG++ with Tree Search, Memory & PUCT",
                "architecture": "Modular 4-module backend",
                "deployment": "Google Cloud Run"
            },
            "current_status": {
                "phase": project_context["project_state"]["current_phase"],
                "momentum": project_context["project_state"]["momentum"],
                "health": system_health["overall_health"],
                "systems_active": len([s for s in system_health["subsystems"].values() if s.get("status") == "ready"])
            },
            "recent_activity": {
                "conversations_24h": conversation_summary["total_conversations"],
                "discoveries_24h": discovery_summary["total_discoveries"],
                "files_modified": len(project_context["immediate_context"]["files_needing_attention"]),
                "active_tasks": len(project_context["immediate_context"]["active_tasks"])
            },
            "next_actions": project_context["immediate_context"]["recommended_actions"],
            "four_steps_progress": project_context["four_steps_progress"],
            "memory_stats": project_context["memory_systems"],
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting project status: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback_status": "Backend modular implementado, sistemas de memória ativos"
        }


@gpt_actions_router.post("/scientific-discovery")
async def trigger_scientific_discovery_gpt() -> Dict[str, Any]:
    """
    🔬 Trigger Descoberta Científica para GPT Actions
    
    Executa descoberta científica imediata e retorna findings.
    """
    try:
        memory_system = await get_integrated_memory_system()
        discovery_system = memory_system.scientific_discovery
        
        # Força descoberta imediata (não aguarda ciclo horário)
        if discovery_system:
            await discovery_system._parallel_discovery()
        
        # Gera relatório das últimas descobertas
        report = await discovery_system.generate_discovery_report(hours=24)
        
        return {
            "success": True,
            "discovery_triggered": True,
            "report": report,
            "top_findings": report.get("top_discoveries", [])[:3],
            "categories": report.get("categories", {}),
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in scientific discovery: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Discovery system not available"
        }


@gpt_actions_router.get("/system-health")
async def get_system_health_gpt() -> Dict[str, Any]:
    """
    ⚡ Health Check Completo para GPT Actions
    
    Status de todos os sistemas e módulos.
    """
    try:
        memory_system = await get_integrated_memory_system()
        health = await memory_system.get_system_health()
        
        # Simplifica para GPT Actions
        simplified_health = {
            "overall_status": health["overall_health"],
            "systems": {
                name: system.get("status", "unknown")
                for name, system in health["subsystems"].items()
            },
            "four_steps_completed": len([s for s in health["four_steps"].values() if s["status"] == "completed"]),
            "total_steps": len(health["four_steps"]),
            "memory_active": True,
            "discovery_active": health["subsystems"].get("scientific_discovery", {}).get("running", False),
            "timestamp": "2025-09-20T01:01:00Z"
        }
        
        return simplified_health
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "overall_status": "limited",
            "error": str(e),
            "fallback_message": "Basic systems operational"
        }


@gpt_actions_router.post("/analytics")
async def run_analytics_gpt(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    📈 Analytics Avançados para GPT Actions
    
    Executa analytics usando pcs_helio com integração pcs-meta-repo.
    """
    try:
        helio_service = HelioService()
        await helio_service.initialize()
        
        # Se dados contém métricas KEC, analisa
        if "kec_metrics" in data:
            analysis = await helio_service.analyze_with_pcs_integration(
                data["kec_metrics"],
                metadata=data.get("metadata", {})
            )
            
            return {
                "success": True,
                "analysis": analysis,
                "integration_status": analysis.get("integration_status", "unknown"),
                "insights": analysis.get("local_analysis", {}).get("insights", []),
                "timestamp": "2025-09-20T01:01:00Z"
            }
        else:
            return {
                "success": False,
                "error": "No KEC metrics provided for analysis",
                "expected_format": {"kec_metrics": {"H_spectral": 0.0, "sigma": 0.0}}
            }
            
    except Exception as e:
        logger.error(f"Error in analytics: {e}")
        return {
            "success": False,
            "error": str(e),
            "fallback": "Analytics system not available"
        }


# ================ HELPER FUNCTIONS ================

def _interpret_kec_metrics(metrics: Dict[str, float], graph) -> Dict[str, str]:
    """Interpreta métricas KEC para contexto humano/GPT."""
    
    interpretation = {}
    
    # Entropia espectral
    h_val = metrics.get("H_spectral", 0)
    if h_val > 3.0:
        interpretation["entropy"] = "Alta complexidade estrutural - rede heterogênea"
    elif h_val > 1.5:
        interpretation["entropy"] = "Complexidade moderada - estrutura balanceada"
    else:
        interpretation["entropy"] = "Baixa complexidade - estrutura mais ordenada"
    
    # Small-world sigma
    sigma_val = metrics.get("sigma", 0)
    if sigma_val > 2.0:
        interpretation["small_world"] = "Forte comportamento small-world - eficiente para transporte"
    elif sigma_val > 1.2:
        interpretation["small_world"] = "Comportamento small-world moderado"
    else:
        interpretation["small_world"] = "Comportamento small-world fraco - pode limitar difusão"
    
    # Curvatura de Forman
    k_mean = metrics.get("k_forman_mean", 0)
    if k_mean > 0:
        interpretation["curvature"] = "Curvatura positiva média - estrutura estável"
    elif k_mean < -2:
        interpretation["curvature"] = "Curvatura negativa - geometria hiperbólica"
    else:
        interpretation["curvature"] = "Curvatura neutra - geometria plana"
    
    # Recomendações
    if sigma_val > 1.5 and h_val > 2.0:
        interpretation["recommendation"] = "Excelente para aplicações biomédicas - alta eficiência e complexidade"
    elif sigma_val < 1.2 or h_val < 1.0:
        interpretation["recommendation"] = "Considerar otimização da arquitetura porosa"
    else:
        interpretation["recommendation"] = "Propriedades adequadas para aplicação target"
    
    return interpretation


