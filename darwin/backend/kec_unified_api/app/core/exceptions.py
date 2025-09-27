"""Exception handling for DARWIN META-RESEARCH BRAIN."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Any, Dict

class APIError(Exception):
    """Base API error class."""
    def __init__(self, message: str, code: str = "API_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)

class DomainError(APIError):
    """Domain-specific error."""
    def __init__(self, message: str, domain: str):
        self.domain = domain
        super().__init__(message, f"DOMAIN_ERROR_{domain.upper()}")

class MCPError(APIError):
    """MCP-related error."""
    def __init__(self, message: str):
        super().__init__(message, "MCP_ERROR")

# Exception handlers
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": exc.message,
            "code": exc.code,
            "type": "api_error"
        }
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "type": "general_error"
        }
    )

async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Validation error",
            "type": "validation_error"
        }
    )

__all__ = [
    "APIError", 
    "DomainError", 
    "MCPError",
    "api_error_handler",
    "general_exception_handler", 
    "validation_error_handler"
]