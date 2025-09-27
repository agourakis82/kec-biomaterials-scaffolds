"""MCP Resources registration for DARWIN."""

from ..core.logging import get_logger

logger = get_logger("mcp.resources")


def register_mcp_resources(server, brain):
    """Register MCP resources with the server."""
    logger.info("Registering MCP resources...")
    # Resources will be implemented here
    logger.info("✅ MCP resources registered")


__all__ = ["register_mcp_resources"]