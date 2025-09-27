"""Auth manager stub for DARWIN."""

from typing import Any, Dict

class AuthManager:
    """Basic auth manager."""
    
    def __init__(self):
        self.enabled = False
    
    async def initialize(self):
        """Initialize auth."""
        self.enabled = True
    
    async def shutdown(self):
        """Shutdown auth."""
        self.enabled = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for auth."""
        return {
            "healthy": self.enabled,
            "status": "operational" if self.enabled else "offline",
        }

__all__ = ["AuthManager"]