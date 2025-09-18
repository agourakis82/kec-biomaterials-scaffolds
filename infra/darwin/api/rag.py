"""Retrieval-augmented generation helpers for DARWIN."""
from __future__ import annotations

import os
import textwrap
import uuid
from typing import Any, Dict, Iterable, List

from . import deps, providers_gcp

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")

_vertex_ready = False


def _use_gcp() -> bool:
    return bool(GCP_PROJECT_ID)


def _ensure_vertex() -> None:
    global _vertex_ready
    if not _use_gcp() or _vertex_ready:
        return
    providers_gcp.init_vertex(GCP_PROJECT_ID, GCP_REGION)
    _vertex_ready = True


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if _use_gcp():
        _ensure_vertex()
        return providers_gcp.embed_texts(texts)
    return deps.embed_texts_openai(texts)


def _chat_complete(prompt: str) -> str:
    if _use_gcp():
        _ensure_vertex()
        return providers_gcp.chat_complete(prompt)
    return deps.chat_complete_openai(prompt)


def _persist_metadata_if_needed(row: Dict[str, Any]) -> None:
    if _use_gcp() and VECTOR_BACKEND.lower() == "gcp_bq" and BQ_DATASET and BQ_TABLE:
        providers_gcp.bq_upsert_document(BQ_DATASET, BQ_TABLE, row)


class RAGEngine:
    """Minimal RAG engine supporting Mode A/B."""

    def __init__(self) -> None:
        self.vector_store = deps.get_vector_store()

    def index(self, content: str, metadata: Dict[str, Any]) -> str:
        doc_id = metadata.get("doc_id") or str(uuid.uuid4())
        full_metadata = dict(metadata)
        full_metadata.setdefault("content", content)
        embedding = _embed_texts([content])[0]
        self.vector_store.add(doc_id, embedding, full_metadata)
        _persist_metadata_if_needed(
            {
                "doc_id": doc_id,
                "title": full_metadata.get("title", ""),
                "url": full_metadata.get("url", ""),
                "abstract": full_metadata.get("abstract", ""),
                "content": content,
                "source": full_metadata.get("source", "manual"),
            }
        )
        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[deps.QueryResult]:
        embedding = _embed_texts([query])[0]
        return self.vector_store.query(embedding, top_k=top_k)

    def answer(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        results = self.search(query, top_k=top_k)
        context_blocks = []
        for idx, result in enumerate(results, start=1):
            content = result.metadata.get("content") or result.metadata.get("text") or ""
            title = result.metadata.get("title") or f"Fragment {idx}"
            block = textwrap.dedent(
                f"""Title: {title}\nScore: {result.score:.3f}\nContent:\n{content}\n"""
            )
            context_blocks.append(block)
        context = "\n---\n".join(context_blocks) if context_blocks else "No supporting passages available."
        prompt = textwrap.dedent(
            f"""
            You are DARWIN, a research assistant.
            Using the provided context, answer the question factually and concisely.
            Context:\n{context}

            Question: {query}
            """
        ).strip()
        completion = _chat_complete(prompt)
        return {
            "query": query,
            "answer": completion,
            "results": [
                {
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }


def persist_metadata(row: Dict[str, Any]) -> None:
    """Public helper for other modules to push metadata into BigQuery when enabled."""
    _persist_metadata_if_needed(row)
