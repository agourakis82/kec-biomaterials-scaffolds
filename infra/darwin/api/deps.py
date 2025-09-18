"""Shared dependencies and helpers for the DARWIN cloud API."""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional dependency for local mode A
    import chromadb
    from chromadb.api.models import Collection
except ImportError:  # pragma: no cover - optional
    chromadb = None
    Collection = None

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "darwin_rag")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


@dataclass
class QueryResult:
    doc_id: str
    score: float
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    """Very small in-memory vector store used for Mode B (GCP BigQuery)."""

    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, doc_id: str, embedding: Iterable[float], metadata: Dict[str, Any]) -> None:
        vector = _to_vector(embedding)
        item = {"doc_id": doc_id, "embedding": vector, "metadata": metadata}
        with self._lock:
            self._items.append(item)

    def query(self, embedding: Iterable[float], top_k: int = 5) -> List[QueryResult]:
        vector = _to_vector(embedding)
        with self._lock:
            if not self._items:
                return []
            scored: List[QueryResult] = []
            for item in self._items:
                score = _cosine_similarity(vector, item["embedding"])
                scored.append(QueryResult(doc_id=item["doc_id"], score=score, metadata=dict(item["metadata"])) )
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[:top_k]


class ChromaVectorStore:
    """Wrapper around ChromaDB for backwards compatibility (Mode A)."""

    def __init__(self) -> None:  # pragma: no cover - requires chromadb
        if chromadb is None:
            raise RuntimeError("Chroma is not available. Install chromadb or switch VECTOR_BACKEND")
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection: Collection = self._client.get_or_create_collection(CHROMA_COLLECTION)

    def add(self, doc_id: str, embedding: Iterable[float], metadata: Dict[str, Any]) -> None:  # pragma: no cover - requires chromadb
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[list(map(float, embedding))],
            metadatas=[metadata],
        )

    def query(self, embedding: Iterable[float], top_k: int = 5) -> List[QueryResult]:  # pragma: no cover
        results = self._collection.query(embeddings=[list(map(float, embedding))], n_results=top_k)
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        scored: List[QueryResult] = []
        for doc_id, metadata, distance in zip(ids, metadatas, distances):
            score = 1.0 - float(distance)
            scored.append(QueryResult(doc_id=str(doc_id), score=score, metadata=dict(metadata or {})))
        return scored


_vector_store: Optional[Any] = None
_vector_lock = threading.Lock()


def _to_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(map(float, values)), dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("Embedding must be a 1D vector")
    return vector


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_vector_store() -> Any:
    """Return the configured vector store, respecting Mode B constraints."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    with _vector_lock:
        if _vector_store is not None:
            return _vector_store
        backend = VECTOR_BACKEND.lower()
        if backend == "gcp_bq":
            store: Any = InMemoryVectorStore()
        else:
            if chromadb is not None:
                try:
                    store = ChromaVectorStore()
                except Exception:  # pragma: no cover - fallback safety
                    store = InMemoryVectorStore()
            else:
                store = InMemoryVectorStore()
        _vector_store = store
        return _vector_store


def reset_vector_store() -> None:
    global _vector_store
    with _vector_lock:
        _vector_store = None


def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI or fallback deterministic vectors."""
    try:  # pragma: no cover - requires openai package/runtime
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        return [list(item.embedding) for item in response.data]
    except Exception:
        return [deterministic_embedding(text) for text in texts]


def chat_complete_openai(prompt: str) -> str:
    """Create a text completion using OpenAI if available, otherwise fallback."""
    try:  # pragma: no cover - requires openai package/runtime
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=512,
            temperature=0.2,
        )
        return response.output_text
    except Exception:
        return fallback_completion(prompt)


def deterministic_embedding(text: str, dim: int = 128) -> List[float]:
    """Create a deterministic pseudo-embedding for offline/development usage."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeated = (digest * ((dim // len(digest)) + 1))[:dim]
    vector = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
    norm = np.linalg.norm(vector) or 1.0
    return (vector / norm).tolist()


def fallback_completion(prompt: str) -> str:
    """Return a lightweight completion when OpenAI is unavailable."""
    snippet = prompt.strip().splitlines()[-1] if prompt.strip() else ""
    return f"[offline-mode] Unable to reach OpenAI. Last prompt line: {snippet[:80]}"
