"""Minimal notebook service stub."""

from typing import Dict, List


class _NotebookService:
    async def list_notebooks(self) -> List[Dict[str, str]]:
        return []

    async def health_check(self) -> Dict[str, str]:
        return {"status": "ready"}


notebook_service = _NotebookService()

