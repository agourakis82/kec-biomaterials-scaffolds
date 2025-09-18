#!/usr/bin/env python3
"""
DARWIN Mode B Cloud API - Main FastAPI application
Exposes RAG+ endpoints with GCP Vertex AI and BigQuery integration
"""

import os
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import rag, discovery, deps
from .providers_gcp import get_vertex_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DARWIN Research API",
    version="1.0.0",
    description="DARWIN Mode B Cloud API exposing retrieval-augmented endpoints for biomaterials research.",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from X-API-KEY header or Bearer token"""
    expected_key = os.getenv("DARWIN_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Request/Response models
class RAGSearchRequest(BaseModel):
    q: str
    k: int = 4

class RAGSearchResponse(BaseModel):
    answer: str
    results: List[Dict[str, Any]]

class DiscoveryRequest(BaseModel):
    run_once: bool = False

class DiscoveryResponse(BaseModel):
    status: str
    added: int
    message: str

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "darwin-rag"}

# OpenAPI endpoint
@app.get("/openapi.json")
async def get_openapi():
    """Return OpenAPI schema"""
    return app.openapi()

# Static files for plugin
@app.get("/.well-known/ai-plugin.json")
async def get_ai_plugin():
    """Return AI plugin manifest for ChatGPT Actions"""
    run_url = os.getenv("RUN_URL", "https://darwin-rag-placeholder.run.app")
    return {
        "schema_version": "v1",
        "name_for_human": "DARWIN Research API",
        "name_for_model": "darwin_research",
        "description_for_model": "Access DARWIN's retrieval-augmented research tools for biomaterials.",
        "description_for_human": "DARWIN RAG+ assistant for biomaterials discovery.",
        "auth": {
            "type": "service_http",
            "authorization_type": "bearer",
            "verification_tokens": {}
        },
        "api": {
            "type": "openapi",
            "url": f"{run_url}/openapi.json",
            "is_user_authenticated": True
        },
        "logo_url": f"{run_url}/static/logo.png",
        "contact_email": "demetrios@agourakis.med.br",
        "legal_info_url": f"{run_url}/terms"
    }

# RAG+ Search endpoint
@app.post("/rag-plus/search", response_model=RAGSearchResponse)
async def rag_search(
    request: RAGSearchRequest,
    api_key: str = Depends(verify_api_key)
):
    """Search knowledge base via RAG+"""
    try:
        logger.info(f"RAG search query: {request.q}")
        
        # Initialize RAG engine
        engine = rag.RAGEngine()
        
        # Perform RAG search with answer generation
        result = engine.answer(request.q, top_k=request.k)
        
        return RAGSearchResponse(
            answer=result["answer"],
            results=result["results"]
        )
        
    except Exception as e:
        logger.error(f"RAG search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Discovery endpoint
@app.post("/discovery/run", response_model=DiscoveryResponse)
async def run_discovery(
    request: DiscoveryRequest = DiscoveryRequest(),
    api_key: str = Depends(verify_api_key)
):
    """Trigger discovery ingest job"""
    try:
        logger.info(f"Discovery run requested, run_once={request.run_once}")
        
        # Run discovery
        result = discovery.run_discovery(run_once=request.run_once)
        
        return DiscoveryResponse(
            status="success",
            added=result.get("added", 0),
            message=result.get("message", "Discovery completed")
        )
        
    except Exception as e:
        logger.error(f"Discovery error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
