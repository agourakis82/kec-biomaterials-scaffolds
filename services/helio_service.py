"""Minimal HELIO data service stub."""

from typing import Any, Dict, List, Optional


class _HELIOService:
    async def get_summaries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return []

    async def health_check(self) -> Dict[str, str]:
        return {"status": "ready"}


helio_service = _HELIOService()

