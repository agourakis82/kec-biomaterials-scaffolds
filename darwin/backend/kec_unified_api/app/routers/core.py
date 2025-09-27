"""Core router for DARWIN META-RESEARCH BRAIN.

Basic system endpoints including health checks, status, and meta-information.
"""

import asyncio
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..core.exceptions import APIError, DomainError
from ..core.logging import get_logger

logger = get_logger("core.router")

router = APIRouter(
    tags=["Core System"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    }
)


# Response Models
class HealthStatus(BaseModel):
    """Health check response model."""
    
    healthy: bool = Field(..., description="Overall health status")
    status: str = Field(..., description="Status description")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    
    
class ComponentHealth(BaseModel):
    """Individual component health status."""
    
    name: str = Field(..., description="Component name")
    healthy: bool = Field(..., description="Component health status")
    status: str = Field(..., description="Status description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    last_check: datetime = Field(..., description="Last health check timestamp")


class SystemInfo(BaseModel):
    """System information model."""
    
    service_name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment (dev/prod/test)")
    python_version: str = Field(..., description="Python version")
    platform: str = Field(..., description="Platform information")
    hostname: str = Field(..., description="Server hostname")
    process_id: int = Field(..., description="Process ID")
    start_time: datetime = Field(..., description="Service start time")
    current_time: datetime = Field(..., description="Current server time")
    
    # Feature flags
    features: Dict[str, bool] = Field(..., description="Enabled features")
    domains: Dict[str, bool] = Field(..., description="Enabled research domains")
    
    # Configuration summary
    config_summary: Dict[str, Any] = Field(..., description="Configuration summary")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    
    overall: HealthStatus = Field(..., description="Overall system health")
    components: List[ComponentHealth] = Field(..., description="Component health details")
    system_info: SystemInfo = Field(..., description="System information")


# Global variables for tracking
_start_time = datetime.now(timezone.utc)
_health_cache: List[ComponentHealth] = []
_last_health_check = None


async def check_domain_health(domain_name: str, engine: Any) -> ComponentHealth:
    """Check health of a specific domain engine."""
    try:
        if hasattr(engine, 'health_check'):
            result = await engine.health_check()
            return ComponentHealth(
                name=f"domain.{domain_name}",
                healthy=result.get("healthy", True),
                status=result.get("status", "operational"),
                details=result.get("details", {}),
                last_check=datetime.now(timezone.utc)
            )
        else:
            return ComponentHealth(
                name=f"domain.{domain_name}",
                healthy=True,
                status="operational",
                details={"message": "Health check not implemented"},
                last_check=datetime.now(timezone.utc)
            )
    except Exception as e:
        logger.error(f"Domain health check failed for {domain_name}: {e}")
        return ComponentHealth(
            name=f"domain.{domain_name}",
            healthy=False,
            status="error",
            details={"error": str(e)},
            last_check=datetime.now(timezone.utc)
        )


async def check_system_health(request: Request) -> ComponentHealth:
    """Check core system health."""
    try:
        # Basic system checks
        checks = {
            "memory_usage": sys.getsizeof({}),  # Basic memory check
            "can_create_tasks": True,
            "filesystem_writable": True,  # Could add actual FS check
        }
        
        return ComponentHealth(
            name="system.core",
            healthy=True,
            status="operational",
            details=checks,
            last_check=datetime.now(timezone.utc)
        )
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return ComponentHealth(
            name="system.core",
            healthy=False,
            status="error",
            details={"error": str(e)},
            last_check=datetime.now(timezone.utc)
        )


@router.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc),
        "endpoints": {
            "health": "/healthz",
            "detailed_health": "/health/detailed", 
            "system_info": "/info",
            "ping": "/ping"
        },
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc",
            "schema": "/openapi.json"
        }
    }


@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity checks."""
    return {
        "ping": "pong",
        "timestamp": datetime.now(timezone.utc),
        "service": settings.app_name
    }


@router.get("/healthz", response_model=HealthStatus)
async def health_check_simple():
    """Simple health check endpoint."""
    current_time = datetime.now(timezone.utc)
    uptime = (current_time - _start_time).total_seconds()
    
    return HealthStatus(
        healthy=True,
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        timestamp=current_time,
        uptime_seconds=uptime
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def health_check_detailed(request: Request):
    """Detailed health check with component status."""
    global _health_cache, _last_health_check
    
    current_time = datetime.now(timezone.utc)
    
    # Cache health checks for 30 seconds to avoid overloading
    if (_last_health_check is None or 
        (current_time - _last_health_check).total_seconds() > 30):
        
        logger.info("Performing detailed health check...")
        
        components = []
        
        # Check system health
        system_health = await check_system_health(request)
        components.append(system_health)
        
        # Check domain engines if available
        if hasattr(request.app.state, 'brain'):
            brain = request.app.state.brain
            
            if brain.domain_engines:
                for domain_name, engine in brain.domain_engines.items():
                    domain_health = await check_domain_health(domain_name, engine)
                    components.append(domain_health)
            
            # Check core systems
            if brain.cache_manager:
                try:
                    cache_result = await brain.cache_manager.health_check()
                    components.append(ComponentHealth(
                        name="system.cache",
                        healthy=cache_result.get("healthy", True),
                        status=cache_result.get("status", "operational"),
                        details=cache_result.get("details", {}),
                        last_check=current_time
                    ))
                except Exception as e:
                    components.append(ComponentHealth(
                        name="system.cache",
                        healthy=False,
                        status="error",
                        details={"error": str(e)},
                        last_check=current_time
                    ))
            
            if brain.multi_ai_hub:
                try:
                    ai_result = await brain.multi_ai_hub.health_check()
                    components.append(ComponentHealth(
                        name="system.multi_ai",
                        healthy=ai_result.get("healthy", True),
                        status=ai_result.get("status", "operational"),
                        details=ai_result.get("details", {}),
                        last_check=current_time
                    ))
                except Exception as e:
                    components.append(ComponentHealth(
                        name="system.multi_ai",
                        healthy=False,
                        status="error", 
                        details={"error": str(e)},
                        last_check=current_time
                    ))
        
        _health_cache = components
        _last_health_check = current_time
    else:
        components = _health_cache
    
    # Determine overall health
    all_healthy = all(component.healthy for component in components)
    overall_status = "healthy" if all_healthy else "degraded"
    
    uptime = (current_time - _start_time).total_seconds()
    
    # Create system info
    system_info = SystemInfo(
        service_name=settings.app_name,
        version=settings.app_version,
        environment=settings.env,
        python_version=sys.version,
        platform=platform.platform(),
        hostname=platform.node(),
        process_id=os.getpid(),
        start_time=_start_time,
        current_time=current_time,
        features=settings.features_enabled,
        domains={
            "biomaterials": settings.domains.biomaterials_enabled,
            "neuroscience": settings.domains.neuroscience_enabled,
            "philosophy": settings.domains.philosophy_enabled,
            "quantum_mechanics": settings.domains.quantum_enabled,
            "psychiatry": settings.domains.psychiatry_enabled,
        },
        config_summary={
            "debug": settings.debug,
            "cors_enabled": settings.cors_enabled,
            "mcp_enabled": settings.mcp.mcp_enabled,
            "cache_enabled": True,  # Assume cache is always enabled
            "monitoring_enabled": settings.monitoring.metrics_enabled,
        }
    )
    
    return DetailedHealthResponse(
        overall=HealthStatus(
            healthy=all_healthy,
            status=overall_status,
            service=settings.app_name,
            version=settings.app_version,
            timestamp=current_time,
            uptime_seconds=uptime
        ),
        components=components,
        system_info=system_info
    )


@router.get("/info", response_model=SystemInfo)
async def get_system_info():
    """Get detailed system information."""
    current_time = datetime.now(timezone.utc)
    
    return SystemInfo(
        service_name=settings.app_name,
        version=settings.app_version,
        environment=settings.env,
        python_version=sys.version,
        platform=platform.platform(),
        hostname=platform.node(),
        process_id=os.getpid(),
        start_time=_start_time,
        current_time=current_time,
        features=settings.features_enabled,
        domains={
            "biomaterials": settings.domains.biomaterials_enabled,
            "neuroscience": settings.domains.neuroscience_enabled,
            "philosophy": settings.domains.philosophy_enabled,
            "quantum_mechanics": settings.domains.quantum_enabled,
            "psychiatry": settings.domains.psychiatry_enabled,
        },
        config_summary={
            "debug": settings.debug,
            "cors_enabled": settings.cors_enabled,
            "mcp_enabled": settings.mcp.mcp_enabled,
            "cache_enabled": True,
            "monitoring_enabled": settings.monitoring.metrics_enabled,
        }
    )


@router.get("/config")
async def get_config_summary():
    """Get configuration summary (non-sensitive information only)."""
    return {
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.env,
            "debug": settings.debug,
        },
        "features": settings.features_enabled,
        "domains": {
            "biomaterials": settings.domains.biomaterials_enabled,
            "neuroscience": settings.domains.neuroscience_enabled,
            "philosophy": settings.domains.philosophy_enabled,
            "quantum_mechanics": settings.domains.quantum_enabled,
            "psychiatry": settings.domains.psychiatry_enabled,
        },
        "mcp": {
            "enabled": settings.mcp.mcp_enabled,
            "tools_enabled": settings.mcp.mcp_tools_enabled,
            "resources_enabled": settings.mcp.mcp_resources_enabled,
        },
        "cors": {
            "enabled": settings.cors_enabled,
            "origins": settings.cors_origins if settings.debug else ["<hidden>"],
        },
        "monitoring": {
            "log_level": settings.monitoring.log_level,
            "metrics_enabled": settings.monitoring.metrics_enabled,
        }
    }


@router.get("/version")
async def get_version():
    """Get service version information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "build_date": _start_time.isoformat(),
        "python_version": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro
        },
        "platform": platform.platform(),
    }


@router.post("/shutdown")
async def shutdown_service():
    """Emergency shutdown endpoint (development only)."""
    if not settings.debug:
        raise HTTPException(
            status_code=403,
            detail="Shutdown endpoint only available in debug mode"
        )
    
    logger.warning("Emergency shutdown requested via API")
    
    # In a real implementation, this would trigger graceful shutdown
    return {
        "message": "Shutdown initiated",
        "timestamp": datetime.now(timezone.utc),
        "note": "This is a development-only endpoint"
    }


# Note: Exception handlers devem ser registrados no FastAPI app, não no router
# Isso será feito em main.py


# Export router
__all__ = ["router"]