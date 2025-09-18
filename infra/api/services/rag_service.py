"""Minimal RAG service stub for admin health aggregation."""

from typing import Dict


class _RagService:
    async def health_check(self) -> Dict[str, str]:
        return {"rag_service": "ready"}


rag_service = _RagService()

