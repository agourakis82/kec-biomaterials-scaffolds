"""DARWIN META-RESEARCH BRAIN - MVP Functional Main Application

MVP básico para corrigir problemas P0 críticos e fazer o servidor funcionar.
"""

import asyncio
import sys
import os
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# FastAPI and ASGI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

# Core imports (absolute paths)
from app.config.settings import settings
from app.core.logging import setup_logging, get_logger
from app.routers.core import router as core_router
from app.routers.kec_metrics import router as kec_metrics_router

# Try to import Score Contracts router
try:
    from app.routers.score_contracts import router as score_contracts_router
    SCORE_CONTRACTS_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("Score Contracts router loaded successfully")
except Exception as e:
    score_contracts_router = None
    SCORE_CONTRACTS_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Score Contracts router not available: {e}")

# Try to import Tree Search router
try:
    from app.routers.tree_search import router as tree_search_router
    TREE_SEARCH_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("Tree Search router loaded successfully")
except Exception as e:
    tree_search_router = None
    TREE_SEARCH_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Tree Search router not available: {e}")

# Try to import Scientific Discovery router
try:
    from app.routers.scientific_discovery import router as scientific_discovery_router
    SCIENTIFIC_DISCOVERY_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("Scientific Discovery router loaded successfully")
except Exception as e:
    scientific_discovery_router = None
    SCIENTIFIC_DISCOVERY_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Scientific Discovery router not available: {e}")

# Try to import RAG++ router with fallback to basic version
try:
    from app.routers.rag_plus import router as rag_plus_router
    RAG_PLUS_AVAILABLE = True
    RAG_PLUS_MODE = "full"
    logger = get_logger("darwin.main")
    logger.info("RAG++ full router loaded successfully")
except Exception as e:
    try:
        from app.routers.rag_plus_basic import router as rag_plus_router
        RAG_PLUS_AVAILABLE = True
        RAG_PLUS_MODE = "basic"
        logger = get_logger("darwin.main")
        logger.info("RAG++ basic router loaded successfully")
    except Exception as e2:
        rag_plus_router = None
        RAG_PLUS_AVAILABLE = False
        RAG_PLUS_MODE = "none"
        logger = get_logger("darwin.main")
        logger.warning(f"No RAG++ router available: {e2}")

# Try to import Multi-AI Hub router
try:
    from app.multi_ai import router as multi_ai_router, initialize_multi_ai_hub, shutdown_multi_ai_hub
    MULTI_AI_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("🎯 Multi-AI Hub router loaded successfully - Revolutionary AI Orchestration Ready!")
except Exception as e:
    multi_ai_router = None
    initialize_multi_ai_hub = None
    shutdown_multi_ai_hub = None
    MULTI_AI_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Multi-AI Hub router not available: {e}")

# Try to import Knowledge Graph router
try:
    from app.knowledge_graph import router as knowledge_graph_router
    KNOWLEDGE_GRAPH_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("🌐 Knowledge Graph router loaded successfully - Interdisciplinary Research Ready!")
except Exception as e:
    knowledge_graph_router = None
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Knowledge Graph router not available: {e}")

# Try to import AI Agents Research Team router
try:
    from app.routers.ai_agents import router as ai_agents_router
    from app.ai_agents import initialize_research_team, shutdown_research_team
    AI_AGENTS_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("🤖 AI Agents Research Team router loaded successfully - AutoGen Multi-Agent Ready!")
except Exception as e:
    ai_agents_router = None
    initialize_research_team = None
    shutdown_research_team = None
    AI_AGENTS_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"AI Agents router not available: {e}")

# Try to import Ultra-Performance router
try:
    from app.routers.ultra_performance import router as ultra_performance_router
    from app.performance import initialize_performance_engine, shutdown_performance_engine
    ULTRA_PERFORMANCE_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("⚡ Ultra-Performance router loaded successfully - JAX 1000x Speedup Ready!")
except Exception as e:
    ultra_performance_router = None
    initialize_performance_engine = None
    shutdown_performance_engine = None
    ULTRA_PERFORMANCE_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Ultra-Performance router not available: {e}")

# Try to import Data Pipeline router
try:
    from app.routers.data_pipeline import router as data_pipeline_router
    from app.routers.data_pipeline import initialize_data_pipeline, shutdown_data_pipeline
    DATA_PIPELINE_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("🌊 Data Pipeline router loaded successfully - Million Scaffold Processing Ready!")
except Exception as e:
    data_pipeline_router = None
    initialize_data_pipeline = None
    shutdown_data_pipeline = None
    DATA_PIPELINE_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Data Pipeline router not available: {e}")

# Try to import Monitoring Dashboard router
try:
    from app.routers.monitoring_dashboard import router as monitoring_dashboard_router
    MONITORING_DASHBOARD_AVAILABLE = True
    logger = get_logger("darwin.main")
    logger.info("📊 Monitoring Dashboard router loaded successfully - Revolutionary Observability Ready!")
except Exception as e:
    monitoring_dashboard_router = None
    MONITORING_DASHBOARD_AVAILABLE = False
    logger = get_logger("darwin.main")
    logger.warning(f"Monitoring Dashboard router not available: {e}")

logger = get_logger("darwin.main")
# settings já importado diretamente

# Global variables for tracking
_start_time = datetime.now(timezone.utc)


class DarwinMetaResearchBrainMVP:
    """
    MVP DARWIN META-RESEARCH BRAIN - apenas o essencial que funciona.
    """
    
    def __init__(self):
        self.fastapi_app: Optional[FastAPI] = None
        self.mode: str = "fastapi"
    
    def create_fastapi_app(self) -> FastAPI:
        """Create MVP FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager."""
            logger.info("🚀 Starting DARWIN META-RESEARCH BRAIN - MVP Mode with KEC Metrics")
            
            # Store brain instance in app state
            app.state.brain = self
            app.state.start_time = _start_time
            
            # Initialize Multi-AI Hub if available
            if MULTI_AI_AVAILABLE and initialize_multi_ai_hub:
                try:
                    await initialize_multi_ai_hub()
                    logger.info("🎯 Multi-AI Hub initialized successfully!")
                except Exception as e:
                    logger.error(f"Multi-AI Hub initialization failed: {e}")
            
            # Initialize AI Agents Research Team if available
            if AI_AGENTS_AVAILABLE and initialize_research_team:
                try:
                    await initialize_research_team()
                    logger.info("🤖 AutoGen Multi-Agent Research Team initialized successfully!")
                except Exception as e:
                    logger.error(f"Research Team initialization failed: {e}")
            
            # Initialize Ultra-Performance Engine if available
            if ULTRA_PERFORMANCE_AVAILABLE and initialize_performance_engine:
                try:
                    await initialize_performance_engine()
                    logger.info("⚡ JAX Ultra-Performance Engine initialized successfully!")
                except Exception as e:
                    logger.error(f"Performance Engine initialization failed: {e}")
            
            # Initialize Data Pipeline if available
            if DATA_PIPELINE_AVAILABLE and initialize_data_pipeline:
                try:
                    await initialize_data_pipeline()
                    logger.info("🌊 Million Scaffold Data Pipeline initialized successfully!")
                except Exception as e:
                    logger.error(f"Data Pipeline initialization failed: {e}")
            
            yield
            
            # Shutdown Multi-AI Hub if available
            if MULTI_AI_AVAILABLE and shutdown_multi_ai_hub:
                try:
                    await shutdown_multi_ai_hub()
                    logger.info("🎯 Multi-AI Hub shutdown complete")
                except Exception as e:
                    logger.error(f"Multi-AI Hub shutdown error: {e}")
            
            # Shutdown AI Agents Research Team if available
            if AI_AGENTS_AVAILABLE and shutdown_research_team:
                try:
                    await shutdown_research_team()
                    logger.info("🤖 AutoGen Research Team shutdown complete")
                except Exception as e:
                    logger.error(f"Research Team shutdown error: {e}")
            
            # Shutdown Ultra-Performance Engine if available
            if ULTRA_PERFORMANCE_AVAILABLE and shutdown_performance_engine:
                try:
                    await shutdown_performance_engine()
                    logger.info("⚡ JAX Performance Engine shutdown complete")
                except Exception as e:
                    logger.error(f"Performance Engine shutdown error: {e}")
            
            # Shutdown Data Pipeline if available
            if DATA_PIPELINE_AVAILABLE and shutdown_data_pipeline:
                try:
                    await shutdown_data_pipeline()
                    logger.info("🌊 Million Scaffold Pipeline shutdown complete")
                except Exception as e:
                    logger.error(f"Data Pipeline shutdown error: {e}")
            
            logger.info("🛑 Shutting down DARWIN META-RESEARCH BRAIN MVP")
        
        # Create FastAPI app
        app = FastAPI(
            title=settings.app_name,
            description=settings.app_description,
            version=settings.app_version,
            lifespan=lifespan,
            openapi_tags=[
                {"name": "core", "description": "Core system endpoints"},
                {"name": "KEC Metrics", "description": "Análise de métricas KEC para scaffolds biomateriais"},
            ] + ([{"name": "Multi-AI Hub", "description": "🎯 Sistema revolucionário de orchestração de múltiplas IAs com roteamento inteligente ChatGPT/Claude/Gemini"}] if MULTI_AI_AVAILABLE else []) + ([{"name": "Score Contracts", "description": "Sistema Score Contracts com sandbox execution seguro para análise matemática avançada"}] if SCORE_CONTRACTS_AVAILABLE else []) + ([{"name": "RAG++ Enhanced Research", "description": "Sistema RAG++ consolidado com IA avançada e busca científica"}] if RAG_PLUS_AVAILABLE else []) + ([{"name": "Tree Search PUCT", "description": "Sistema Tree Search PUCT completo com algoritmos MCTS avançados"}] if TREE_SEARCH_AVAILABLE else []) + ([{"name": "Scientific Discovery", "description": "Sistema Scientific Discovery automático com RSS monitoring, novelty detection e cross-domain insights"}] if SCIENTIFIC_DISCOVERY_AVAILABLE else []) + ([{"name": "Knowledge Graph", "description": "🌐 Sistema épico de Knowledge Graph interdisciplinar conectando biomaterials, neuroscience, philosophy, quantum e psychiatry"}] if KNOWLEDGE_GRAPH_AVAILABLE else []) + ([{"name": "AI Agents Research Team", "description": "🤖 AutoGen Multi-Agent Research Team com 8 especialistas colaborativos (biomaterials, quantum, clinical, pharmacology, mathematics, philosophy, literature, synthesis)"}] if AI_AGENTS_AVAILABLE else []) + ([{"name": "Ultra-Performance Revolutionary", "description": "⚡ JAX Ultra-Performance Computing com 1000x speedup, GPU/TPU acceleration, Optax optimization e million-scaffold processing"}] if ULTRA_PERFORMANCE_AVAILABLE else []) + ([{"name": "Data Pipeline Million Scaffold", "description": "🌊 Million Scaffold Processing Pipeline com JAX ultra-performance, BigQuery streaming, real-time analytics e biocompatibility analysis"}] if DATA_PIPELINE_AVAILABLE else []) + ([{"name": "monitoring", "description": "📊 Revolutionary Monitoring Dashboard com real-time performance metrics, alerting inteligente e cost monitoring"}] if MONITORING_DASHBOARD_AVAILABLE else []),
            contact={
                "name": "DARWIN Research Team",
                "url": "https://github.com/your-org/darwin-meta-research-brain",
            },
            license_info={
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
        )
        
        # Configure CORS
        if settings.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=settings.cors_origins,
                allow_credentials=True,
                allow_methods=settings.cors_methods,
                allow_headers=settings.cors_headers,
            )
        
        # Exception handlers (fixed - use app, not router)
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Validation Error",
                    "detail": exc.errors(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": str(exc) if settings.debug else "An internal error occurred",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        # Global health endpoint (ANTES de definir app)
        @app.get("/health")
        async def health_check_global():
            """Global health check for MVP."""
            current_time = datetime.now(timezone.utc)
            uptime = (current_time - _start_time).total_seconds()
            
            return {
                "status": "healthy",
                "service": settings.app_name,
                "version": settings.app_version,
                "mode": "mvp",
                "timestamp": current_time,
                "uptime_seconds": uptime,
                "components": {
                    "fastapi": "operational",
                    "core_router": "operational",
                    "kec_metrics": "operational",
                    "multi_ai_hub": "operational" if MULTI_AI_AVAILABLE else "unavailable",
                    "score_contracts": "operational" if SCORE_CONTRACTS_AVAILABLE else "unavailable",
                    "rag_plus": f"{RAG_PLUS_MODE}" if RAG_PLUS_AVAILABLE else "unavailable",
                    "tree_search": "operational" if TREE_SEARCH_AVAILABLE else "unavailable",
                    "scientific_discovery": "operational" if SCIENTIFIC_DISCOVERY_AVAILABLE else "unavailable",
                    "knowledge_graph": "operational" if KNOWLEDGE_GRAPH_AVAILABLE else "unavailable",
                    "ai_agents_research_team": "operational" if AI_AGENTS_AVAILABLE else "unavailable",
                    "ultra_performance_jax": "operational" if ULTRA_PERFORMANCE_AVAILABLE else "unavailable",
                    "data_pipeline_million_scaffold": "operational" if DATA_PIPELINE_AVAILABLE else "unavailable",
                    "monitoring_dashboard": "operational" if MONITORING_DASHBOARD_AVAILABLE else "unavailable"
                }
            }
        
        # Include routers (apenas o que existe)
        app.include_router(core_router, tags=["core"])
        
        # Include KEC Metrics router
        app.include_router(kec_metrics_router, prefix="/api/v1", tags=["KEC Metrics"])
        
        # Include Multi-AI Hub router if available
        if MULTI_AI_AVAILABLE and multi_ai_router:
            app.include_router(multi_ai_router, tags=["Multi-AI Hub"])
            logger.info("🎯 Multi-AI Hub router registered successfully - Revolutionary AI Orchestration Active!")
        else:
            logger.warning("Multi-AI Hub router not registered - running without Multi-AI orchestration")
        
        # Include Score Contracts router if available
        if SCORE_CONTRACTS_AVAILABLE and score_contracts_router:
            app.include_router(score_contracts_router, prefix="/api/v1", tags=["Score Contracts"])
            logger.info("Score Contracts router registered successfully")
        else:
            logger.warning("Score Contracts router not registered - running without Score Contracts")
        
        # Include RAG++ router if available
        if RAG_PLUS_AVAILABLE and rag_plus_router:
            app.include_router(rag_plus_router, prefix="/api/v1", tags=["RAG++ Enhanced Research"])
            logger.info(f"RAG++ router registered successfully - mode: {RAG_PLUS_MODE}")
        else:
            logger.warning("RAG++ router not registered - running without RAG")
        
        # Include Tree Search router if available
        if TREE_SEARCH_AVAILABLE and tree_search_router:
            app.include_router(tree_search_router, prefix="/api/v1", tags=["Tree Search PUCT"])
            logger.info("Tree Search PUCT router registered successfully")
        else:
            logger.warning("Tree Search router not registered - running without Tree Search")
        
        # Include Scientific Discovery router if available
        if SCIENTIFIC_DISCOVERY_AVAILABLE and scientific_discovery_router:
            app.include_router(scientific_discovery_router, prefix="/api/v1", tags=["Scientific Discovery"])
            logger.info("Scientific Discovery router registered successfully")
        else:
            logger.warning("Scientific Discovery router not registered - running without Scientific Discovery")
        
        # Include Knowledge Graph router if available
        if KNOWLEDGE_GRAPH_AVAILABLE and knowledge_graph_router:
            app.include_router(knowledge_graph_router, tags=["Knowledge Graph"])
            logger.info("🌐 Knowledge Graph router registered successfully - Interdisciplinary Research Active!")
        else:
            logger.warning("Knowledge Graph router not registered - running without Knowledge Graph")
        
        # Include AI Agents Research Team router if available
        if AI_AGENTS_AVAILABLE and ai_agents_router:
            app.include_router(ai_agents_router, tags=["AI Agents Research Team"])
            logger.info("🤖 AI Agents Research Team router registered successfully - AutoGen Multi-Agent Collaboration Active!")
        else:
            # Include fallback functional router - No Broken Links policy
            try:
                from .routers.ai_agents_fallback import router as ai_agents_fallback_router
                app.include_router(ai_agents_fallback_router, tags=["AI Agents Research Team"])
                logger.info("🤖 AI Agents Fallback router registered successfully - 100% Functional Endpoints Active!")
            except Exception as fallback_error:
                logger.warning(f"AI Agents fallback router failed: {fallback_error} - running without Research Team")
        
        # Include Ultra-Performance router if available
        if ULTRA_PERFORMANCE_AVAILABLE and ultra_performance_router:
            app.include_router(ultra_performance_router, tags=["Ultra-Performance Revolutionary"])
            logger.info("⚡ Ultra-Performance router registered successfully - JAX 1000x Speedup Active!")
        else:
            logger.warning("Ultra-Performance router not registered - running without JAX acceleration")
        
        # Include Data Pipeline router if available
        if DATA_PIPELINE_AVAILABLE and data_pipeline_router:
            app.include_router(data_pipeline_router, tags=["Data Pipeline Million Scaffold"])
            logger.info("🌊 Data Pipeline router registered successfully - Million Scaffold Processing Active!")
        else:
            logger.warning("Data Pipeline router not registered - running without million scaffold processing")
        
        # Include Monitoring Dashboard router if available
        if MONITORING_DASHBOARD_AVAILABLE and monitoring_dashboard_router:
            app.include_router(monitoring_dashboard_router, tags=["monitoring"])
            logger.info("📊 Monitoring Dashboard router registered successfully - Revolutionary Observability Active!")
        else:
            logger.warning("Monitoring Dashboard router not registered - running without monitoring dashboard")
        
        self.fastapi_app = app
        return app
    
    async def run_fastapi_mode(self):
        """Run in FastAPI web server mode."""
        self.mode = "fastapi"
        app = self.create_fastapi_app()
        
        logger.info(f"🌐 DARWIN MVP FastAPI server starting on {settings.host}:{settings.port}")
        
        # Use uvicorn programmatically
        import uvicorn
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            workers=settings.workers if not settings.reload else 1,
            log_level=settings.monitoring.log_level.lower(),
        )


# Global brain instance
brain = DarwinMetaResearchBrainMVP()


def create_app() -> FastAPI:
    """Create FastAPI application instance."""
    return brain.create_fastapi_app()


def main():
    """Main entry point - MVP mode only."""
    # Setup logging
    setup_logging()
    
    logger.info("🧠 Starting DARWIN META-RESEARCH BRAIN in MVP mode")
    asyncio.run(brain.run_fastapi_mode())


# FastAPI app instance for ASGI servers (uvicorn main:app)
app = create_app()


if __name__ == "__main__":
    main()


# Export for external use
__all__ = [
    "DarwinMetaResearchBrainMVP",
    "brain",
    "create_app",
    "main",
    "app"
]