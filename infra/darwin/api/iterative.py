"""Iterative refinement utilities for DARWIN RAG."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rag import RAGEngine


class IterativeRAG:
    """Simple iterative RAG loop that re-prompts based on prior answers."""

    def __init__(self, engine: Optional[RAGEngine] = None) -> None:
        self.engine = engine or RAGEngine()

    def query(self, question: str, max_iters: int = 3, top_k: int = 4) -> Dict[str, Any]:
        iterations: List[Dict[str, Any]] = []
        follow_up = question
        for step in range(1, max_iters + 1):
            result = self.engine.answer(follow_up, top_k=top_k)
            iterations.append({"iteration": step, **result})
            if not result.get("results"):
                break
            top_hit = result["results"][0]
            follow_up = (
                "Refine the previous answer using the highlighted passage. "
                f"Document ID: {top_hit['doc_id']}. Original question: {question}"
            )
        return {"question": question, "iterations": iterations}
