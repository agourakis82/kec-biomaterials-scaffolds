"""Main FastAPI application factory for KEC Biomaterials API v2.0 - Modular Backend."""

from contextlib import asynccontextmanager
import sys
import os

# Add modular backend to path
sys.path.insert(0, '/app/src')

from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import APIKeyMiddleware
from .cache import cache_manager
from .config import settings
from .documentation import get_documentation_manager, setup_documentation_routes
from .errors import (
    APIError,
    api_error_handler,
    general_exception_handler,
    validation_error_handler,
)
from .custom_logging import RequestLoggingMiddleware, get_logger
from .monitoring import initialize_monitoring, shutdown_monitoring
from .processing import start_processing, stop_processing
from .rate_limit import add_rate_limiting
from .routers import (
    admin,
    core,
    data,
    monitoring,
    notebooks,
    rag,
    rag_plus,
    memory,
    tree_search,
    score_contracts,
)
from .routers.processing import router as processing_router

# NEW: GPT Actions integration
from .gpt_actions import gpt_actions_router

# NEW: Modular backend initialization
from darwin_core.memory.integrated_memory_system import get_integrated_memory_system

logger = get_logger("main")


class ASGILoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs the ASGI scope for incoming requests.

    Logs `path`, `raw_path` and request headers at INFO level with a clear
    marker so it can be filtered in Cloud Run logs for debugging ingress/path
    normalization issues (e.g. missing /healthz).
    """

    async def dispatch(self, request, call_next):
        try:
            scope = request.scope
            path = scope.get("path")
            raw_path = scope.get("raw_path")
            # Collect a few headers to avoid huge logs
            headers = {k.decode(): v.decode() for k, v in scope.get("headers", []) if k.decode().lower() in ("host", "x-forwarded-for", "x-forwarded-host", "x-appengine-country")}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ASGI scope extraction failed", exc_info=exc)
            return await call_next(request)

        logger.info(
            f"ASGI_SCOPE_INSTRUMENT path={path} raw_path={raw_path} headers={headers}"
        )

        # Continue request processing
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with modular backend initialization.

    Handles startup and shutdown events for the FastAPI application.
    """
    logger.info("🚀 Starting KEC Biomaterials API v2.0 - Modular Backend")

    # Initialize cache manager
    await cache_manager.initialize()
    logger.info("Cache manager initialized")

    # Initialize monitoring system
    await initialize_monitoring()
    logger.info("Monitoring system initialized")

    # Initialize processing system
    await start_processing()
    logger.info("Processing system initialized")

    # NEW: Initialize modular backend systems
    try:
        logger.info("🧠 Initializing Integrated Memory System...")
        memory_system = await get_integrated_memory_system()
        app.state.memory_system = memory_system
        logger.info("✅ Integrated Memory System initialized")
        
        # Store startup context for session continuity
        startup_context = await memory_system.get_complete_project_context()
        logger.info(f"📋 Project context loaded - Phase: {startup_context['project_state']['current_phase']}")
        
    except Exception as e:
        logger.error(f"❌ Error initializing modular backend: {e}")
        logger.info("⚠️  Continuing with basic functionality")

    # Log configuration details
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Authentication required: {settings.API_KEY_REQUIRED}")
    logger.info(
        f"Rate limiting enabled: {settings.RATE_LIMIT_REQUESTS_PER_MINUTE} req/min"
    )
    logger.info(f"Cache enabled: {getattr(settings, 'CACHE_ENABLED', True)}")
    logger.info("🎯 GPT Actions endpoints: /gpt-actions/*")

    yield

    logger.info("🛑 Shutting down KEC Biomaterials API v2.0")

    # NEW: Shutdown modular systems
    try:
        if hasattr(app.state, 'memory_system'):
            # Stop scientific discovery if running
            if app.state.memory_system.scientific_discovery:
                await app.state.memory_system.scientific_discovery.stop_continuous_discovery()
            logger.info("🧠 Memory systems shutdown")
    except Exception as e:
        logger.error(f"Error shutting down memory systems: {e}")

    # Shutdown processing system
    await stop_processing()
    logger.info("Processing system shutdown")

    # Shutdown monitoring system
    await shutdown_monitoring()
    logger.info("Monitoring system shutdown")

    # Shutdown cache manager
    await cache_manager.shutdown()
    logger.info("Cache manager shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with OpenAPI configuration
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "admin",
                "description": "Administrative endpoints for health and version",
            },
            {
                "name": "rag",
                "description": "Retrieval-Augmented Generation for documents",
            },
            {
                "name": "RAG++ Enhanced Research",
                "description": "Advanced RAG with iterative reasoning and discovery",
            },
            {
                "name": "data",
                "description": "Data access endpoints for AG5 and HELIO datasets",
            },
            {
                "name": "notebooks",
                "description": "Jupyter notebook management endpoints",
            },
        ],
        contact={
            "name": "PCS-HELIO Project",
            "url": settings.repo_url,
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Add CORS middleware if enabled
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled", extra={"origins": settings.cors_origins or ["*"]})

    # Instrumentation: log ASGI scope early for debugging ingress/path issues
    app.add_middleware(ASGILoggingMiddleware)

    # Add middleware pipeline in reverse order (last added = first executed)

    # 1. Request logging (outermost - logs everything)
    app.add_middleware(
        RequestLoggingMiddleware,
        logger_name="http.access",
        exempt_paths=["/healthz", "/ping"],
    )

    # 2. Rate limiting (before authentication to limit all requests)
    add_rate_limiting(
        app,
        exclude_paths=["/", "/healthz", "/ping", "/docs", "/redoc", "/openapi.json", "/openapi.yaml"],
        header_prefix="X-RateLimit",
    )

    # 3. Authentication (innermost - only applies to protected endpoints)
    app.add_middleware(
        APIKeyMiddleware,
        exempt_paths=["/", "/healthz", "/ping", "/docs", "/redoc", "/openapi.json"],
        require_auth=False,  # Let individual endpoints decide
    )

    # Add exception handlers for comprehensive error handling
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Include routers
    app.include_router(core.router)  # Core endpoints at root level
    app.include_router(admin.router)
    app.include_router(rag.router)
    app.include_router(rag_plus.router)  # RAG++ Enhanced Research endpoints
    app.include_router(data.router)
    app.include_router(notebooks.router)
    app.include_router(monitoring.router, prefix="/monitoring")  # Monitoring endpoints
    app.include_router(processing_router)  # Processing endpoints
    app.include_router(memory.router, prefix="", tags=["Memory"])  # Mount memory router at root
    app.include_router(tree_search.router)
    app.include_router(score_contracts.router)
    
    
    # NEW: GPT Actions router for ChatGPT integration
    app.include_router(gpt_actions_router, tags=["GPT Actions"])
    logger.info("🤖 GPT Actions router mounted at /gpt-actions")

    # Setup documentation routes
    setup_documentation_routes(app)

    # Customize OpenAPI schema
    doc_manager = get_documentation_manager(app)
    app.openapi = doc_manager.get_custom_openapi

    # Expose .well-known manifests for ChatGPT Actions / Gemini Extensions
    try:
        from .openapi_config import setup_well_known_routes

        setup_well_known_routes(app)
    except Exception:
        logger.warning(".well-known routes not configured (openapi_config missing)")

    # Mount static files if directory exists
    if settings.static_path.exists():
        app.mount("/static", StaticFiles(directory=settings.static_path), name="static")
        logger.info("Static files mounted", extra={"path": str(settings.static_path)})

    # POST /rag/index alias for /rag-plus/documents
    @app.post("/rag/index")
    async def rag_index_alias(request: Request):
        """
        Alias for /rag-plus/documents. Forwards request to the same handler.
        Preserves Authorization header.
        """
        from .routers import rag_plus  # Avoid circular import
        # Get the service instance
        service = await rag_plus.get_rag_plus_service()
        body = await request.json()
        # Call the add_document function directly
        response = await rag_plus.add_document(
            request=rag_plus.DocumentAdd(**body),
            service=service
        )
        return response

    # GET /healthz endpoint
    @app.get("/healthz")
    async def healthz():
        namespace = os.environ.get("NAMESPACE", "KEC_BIOMAT_V1")
        return {"status": "ok", "namespace": namespace}

    logger.info("FastAPI application created", extra={"title": settings.api_title})
    return app


# Create application instance
app = create_app()
