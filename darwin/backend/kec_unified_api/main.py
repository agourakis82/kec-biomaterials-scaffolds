from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn

from app.config.settings import settings  # [python.import()](darwin/backend/kec_unified_api/app/config/settings.py:1)
from app.routers.kec_metrics import router as kec_router  # [python.import()](darwin/backend/kec_unified_api/app/routers/kec_metrics.py:1)
from app.routers.rag_plus_basic import router as rag_basic_router  # [python.import()](darwin/backend/kec_unified_api/app/routers/rag_plus_basic.py:1)
from app.routers.scientific_discovery import router as discovery_router  # [python.import()](darwin/backend/kec_unified_api/app/routers/scientific_discovery.py:1)
from app.routers.data_pipeline import (  # [python.import()](darwin/backend/kec_unified_api/app/routers/data_pipeline.py:1)
    router as pipeline_router,
    initialize_data_pipeline,
    shutdown_data_pipeline,
)
# from app.routers.ai_agents import router as ai_agents_router  # [python.import()](darwin/backend/kec_unified_api/app/routers/ai_agents.py:1)
from app.plugins.q1_scholar import q1_scholar_router  # [python.import()](darwin/backend/kec_unified_api/app/plugins/q1_scholar/__init__.py:1)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
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

# Register routers
app.include_router(kec_router)
app.include_router(rag_basic_router)
app.include_router(discovery_router)
app.include_router(pipeline_router)
app.include_router(q1_scholar_router)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.env,
        "cors": {
            "enabled": settings.cors_enabled,
            "origins": settings.cors_origins,
        },
    }

@app.get("/healthz")
async def healthz():
    # Mirror /health for k8s-style probes
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.env,
        "ready": True,
    }

@app.get("/readyz")
async def readyz():
    # Lightweight readiness probe; extend with real checks if needed
    return {
        "status": "ready",
        "app": settings.app_name,
        "env": settings.env,
        "dependencies": {
            "pipeline_initialized": True  # placeholder: assume ready
        }
    }

@app.get("/")
async def root_info():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "api_base": os.getenv("API_BASE_URL", None),
        "frontend_base": os.getenv("FRONTEND_BASE_URL", None),
    }

@app.on_event("startup")
async def on_startup():
    try:
        await initialize_data_pipeline()
    except Exception:
        # Keep app booting even if pipeline init fails
        pass

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await shutdown_data_pipeline()
    except Exception:
        pass

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", os.getenv("PORT", settings.port)))
    uvicorn.run("main:app", host=settings.host, port=port, reload=settings.reload)