"""
GPT Actions Schemas - Schemas OpenAPI para ChatGPT Actions
=========================================================

Schemas otimizados e simplificados para integração perfeita com ChatGPT Actions.
"""

from pydantic import BaseModel
from typing import Dict, List, Any
from enum import Enum


class ActionType(str, Enum):
    """Tipos de ações disponíveis."""

    ANALYZE_KEC = "analyze_kec_metrics"
    RAG_QUERY = "rag_query"
    TREE_SEARCH = "tree_search"
    MEMORY_QUERY = "memory_query"
    PROJECT_STATUS = "project_status"
    DISCOVERY = "scientific_discovery"
    ANALYTICS = "analytics"


class KECMetricsResponse(BaseModel):
    """Response para análise KEC."""

    success: bool
    kec_metrics: Dict[str, float]
    graph_info: Dict[str, Any]
    interpretation: Dict[str, str]
    timestamp: str


class RAGResponse(BaseModel):
    """Response para query RAG++."""

    success: bool
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    method: str
    retrieved_docs: int
    timestamp: str


class TreeSearchResponse(BaseModel):
    """Response para tree search."""

    success: bool
    initial_state: str
    best_action_sequence: List[str]
    action_probabilities: Dict[str, float]
    search_statistics: Dict[str, Any]
    timestamp: str


class ProjectStatusResponse(BaseModel):
    """Response para status do projeto."""

    success: bool
    project_overview: Dict[str, str]
    current_status: Dict[str, Any]
    recent_activity: Dict[str, int]
    next_actions: List[str]
    four_steps_progress: Dict[str, Any]
    timestamp: str


class GPTActionSchemas:
    """Container para todos os schemas GPT Actions."""

    # Request schemas
    KECAnalysisRequest = "KECAnalysisRequest"
    RAGQueryRequest = "RAGQueryRequest"
    TreeSearchRequest = "TreeSearchRequest"
    MemoryQueryRequest = "MemoryQueryRequest"

    # Response schemas
    KECMetricsResponse = KECMetricsResponse
    RAGResponse = RAGResponse
    TreeSearchResponse = TreeSearchResponse
    ProjectStatusResponse = ProjectStatusResponse

    @staticmethod
    def get_openapi_schema() -> Dict[str, Any]:
        """Retorna schema OpenAPI otimizado para GPT Actions."""

        return {
            "openapi": "3.0.0",
            "info": {
                "title": "KEC Biomaterials API for GPT Actions",
                "description": "Advanced biomaterials analysis with RAG++, Tree Search, and Memory Systems",
                "version": "2.0.0",
            },
            "servers": [
                {
                    "url": "https://kec-biomaterials-api-prod.cloudfunctions.net",
                    "description": "Production Cloud Run",
                }
            ],
            "paths": {
                "/gpt-actions/analyze-kec-metrics": {
                    "post": {
                        "summary": "Analyze Biomaterial KEC Metrics",
                        "description": "Calculate H (entropy), κ (curvature), σ/ϕ (small-world) metrics for porous structures",
                        "operationId": "analyzeKECMetrics",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "graph_data": {
                                                "type": "object",
                                                "description": "NetworkX graph in JSON format",
                                            },
                                            "config_overrides": {
                                                "type": "object",
                                                "description": "KEC configuration overrides",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "KEC metrics calculated successfully",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "success": {"type": "boolean"},
                                                "kec_metrics": {"type": "object"},
                                                "interpretation": {"type": "object"},
                                            },
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
                "/gpt-actions/rag-query": {
                    "post": {
                        "summary": "RAG++ Knowledge Query",
                        "description": "Search knowledge base using advanced RAG++ with iterative refinement",
                        "operationId": "ragQuery",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["query"],
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "Search query or question",
                                            },
                                            "use_iterative": {
                                                "type": "boolean",
                                                "default": False,
                                                "description": "Use iterative search refinement",
                                            },
                                            "max_sources": {
                                                "type": "integer",
                                                "default": 5,
                                                "description": "Maximum sources to return",
                                            },
                                        },
                                    }
                                }
                            },
                        },
                    }
                },
                "/gpt-actions/project-status": {
                    "get": {
                        "summary": "Get Complete Project Status",
                        "description": "Current project status, progress, next steps, and system health",
                        "operationId": "getProjectStatus",
                        "responses": {
                            "200": {
                                "description": "Project status retrieved successfully"
                            }
                        },
                    }
                },
                "/gpt-actions/scientific-discovery": {
                    "post": {
                        "summary": "Trigger Scientific Discovery",
                        "description": "Execute immediate scientific discovery across multiple sources",
                        "operationId": "triggerDiscovery",
                    }
                },
                "/gpt-actions/system-health": {
                    "get": {
                        "summary": "System Health Check",
                        "description": "Health status of all backend modules and systems",
                        "operationId": "getSystemHealth",
                    }
                },
            },
        }
