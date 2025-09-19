import os
import hashlib
import random
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import yaml


# --- Models -----------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class ComputeRequest(BaseModel):
    graph_id: str = Field(..., description="Identifier of the graph to compute KEC metrics for")
    sigma_q: bool = Field(False, description="Whether to enable the sigma quality variant")


class ComputeResponse(BaseModel):
    H_spectral: float
    k_forman_mean: float
    sigma: float
    swp: float


class JobStatusResponse(BaseModel):
    id: str
    status: str
    result: Optional[ComputeResponse] = None


# --- App setup ---------------------------------------------------------------

app = FastAPI(
    title="KEC_BIOMAT API",
    version="2025-09-19",
    description=(
        "Stub API for KEC_BIOMAT to validate ChatGPT Actions integration.\n"
        "Protects selected routes with X-API-Key header."
    ),
)

# Restrictive CORS: allow only ChatGPT UI origin by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth dependency ---------------------------------------------------------

def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Verify X-API-Key header against env var KEC_API_KEY if it is set.

    - If KEC_API_KEY is defined (non-empty), require an exact match.
    - If not defined, allow requests (useful for local dev), but keep the header optional.
    """
    expected = os.environ.get("KEC_API_KEY")
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Utility -----------------------------------------------------------------

def _deterministic_metrics(seed_text: str) -> ComputeResponse:
    """Generate deterministic stub metrics based on the input string.

    This is a placeholder where the real KEC computation should be plugged in.
    """
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return ComputeResponse(
        H_spectral=round(rng.uniform(0.5, 2.5), 6),
        k_forman_mean=round(rng.uniform(0.0, 1.0), 6),
        sigma=round(rng.uniform(0.0, 1.0), 6),
        swp=round(rng.uniform(0.0, 100.0), 6),
    )


# --- Routes ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/kec/compute",
    response_model=ComputeResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Compute KEC metrics for a graph",
)
async def compute_kec(payload: ComputeRequest) -> ComputeResponse:
    # TODO: Replace with the real KEC computation using graph_id and sigma_q
    seed_text = f"{payload.graph_id}|sigma_q={payload.sigma_q}"
    return _deterministic_metrics(seed_text)


@app.get(
    "/jobs/{id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Get computation job status",
)
async def get_job_status(id: str) -> JobStatusResponse:
    # Stub: Rotate a deterministic status based on id; include result only when 'done'.
    status_choices = ["queued", "running", "done", "error"]
    idx = int(hashlib.md5(id.encode("utf-8")).hexdigest(), 16) % len(status_choices)
    status = status_choices[idx]
    result = _deterministic_metrics(id) if status == "done" else None
    return JobStatusResponse(id=id, status=status, result=result)


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def ai_plugin_manifest():
    """Serve ChatGPT Actions plugin manifest."""
    return {
        "schema_version": "v1",
        "name_for_human": "KEC_BIOMAT API",
        "name_for_model": "kec_biomat",
        "description_for_human": "Compute KEC (Kirchhoff-Escher-Connectivity) metrics for biomaterial scaffolds.",
        "description_for_model": "API for computing KEC metrics including spectral radius, Forman curvature, and percolation properties of biomaterial scaffold graphs.",
        "auth": {
            "type": "service_http",
            "instructions": "",
            "authorization_type": "bearer",
            "verification_tokens": {
                "openai": os.environ.get("OPENAI_VERIFICATION_TOKEN", "dummy_token")
            }
        },
        "api": {
            "type": "openapi",
            "url": "https://api.agourakis.med.br/openapi.yaml",
            "is_user_authenticated": False
        },
        "logo_url": "https://api.agourakis.med.br/static/logo.png",
        "contact_email": "support@agourakis.med.br",
        "legal_info_url": "https://api.agourakis.med.br/legal"
    }


@app.get("/.well-known/gemini-extension.json", include_in_schema=False)
async def gemini_extension_manifest():
    """Serve Gemini Extensions manifest."""
    return {
        "name": "KEC_BIOMAT API",
        "description": "Compute KEC metrics for biomaterial scaffolds",
        "version": "1.0",
        "api_spec_url": "https://api.agourakis.med.br/openapi.yaml",
        "auth_config": {
            "auth_type": "API_KEY",
            "api_key_config": {
                "api_key_name": "X-API-Key",
                "api_key_location": "HEADER"
            }
        },
        "logo_url": "https://api.agourakis.med.br/static/logo.png",
        "contact_email": "support@agourakis.med.br"
    }


# Notes:
# - To run locally: `uvicorn server.fastapi_app:app --reload`
# - Adjust CORS origins as needed (e.g., to your domain during development).
# - Set env var KEC_API_KEY to enforce API key verification.
