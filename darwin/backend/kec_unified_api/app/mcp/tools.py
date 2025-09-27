"""MCP Tools registration for DARWIN."""

from ..core.logging import get_logger

logger = get_logger("mcp.tools")


def register_mcp_tools(server, brain):
    """Register MCP tools with the server."""
    logger.info("Registering MCP tools...")
    # Tools will be implemented here
    logger.info("✅ MCP tools registered")


__all__ = ["register_mcp_tools"]