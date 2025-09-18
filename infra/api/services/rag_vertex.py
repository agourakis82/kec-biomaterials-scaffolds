"""Vertex-ready RAG backends (Engine + Vector Search).

This module defines a unified interface that selects between
Vertex RAG Engine and Vector Search based on configuration.

Network calls are stubbed for local dev/tests; production code
should plug in the Vertex SDK clients where indicated.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.settings import get_settings

# Lazy imports for GCP clients to avoid mandatory dependency at import time
_AIPLATFORM = None
_VERTEXAI = None

def _lazy_import_vertex():
    global _AIPLATFORM, _VERTEXAI
    if _AIPLATFORM is None:
        try:
            from google.cloud import aiplatform as _gcaip  # type: ignore
            _AIPLATFORM = _gcaip
        except Exception:  # pragma: no cover - optional dep
            _AIPLATFORM = False
    if _VERTEXAI is None:
        try:
            from vertexai.preview.language_models import TextEmbeddingModel as _vem  # type: ignore
            _VERTEXAI = _vem
        except Exception:  # pragma: no cover - optional dep
            _VERTEXAI = False


@dataclass
class RagSource:
    title: str
    snippet: str
    url_or_doi: Optional[str]
    score: float


@dataclass
class RagResult:
    backend_type: str
    cache_hit: bool
    text: str
    sources: List[RagSource]


class _SimpleLru:
    """Tiny async-safe LRU with TTL for query caching."""

    def __init__(self, max_size: int = 256, ttl_s: int = 300) -> None:
        self.max_size = max_size
        self.ttl_s = ttl_s
        self._store: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expires = item
            if time.time() > expires:
                await self.delete(key)
                return None
            # refresh order
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._store:
                self._order.remove(key)
            self._store[key] = (value, time.time() + self.ttl_s)
            self._order.append(key)
            while len(self._order) > self.max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
            if key in self._order:
                self._order.remove(key)


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split()).strip()


class VertexVectorBackend:
    """Adapter to Vertex Vector Search.

    Replace stubbed sections with Vertex SDK calls.
    """

    def __init__(self, project_id: str, location: str, index_id: str, endpoint_id: str):
        self.project_id = project_id
        self.location = location
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.emb_model = get_settings().VERTEX_EMB_MODEL
        self._cache = _SimpleLru(max_size=256, ttl_s=300)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Try Vertex embeddings when configured & libs available
        _lazy_import_vertex()
        s = get_settings()
        if _AIPLATFORM and _VERTEXAI and s.PROJECT_ID and s.LOCATION and s.VERTEX_EMB_MODEL:
            try:  # pragma: no cover - requires cloud
                _AIPLATFORM.init(project=s.PROJECT_ID, location=s.LOCATION)
                model = _VERTEXAI.from_pretrained(s.VERTEX_EMB_MODEL)
                # vertex returns list of Embedding objects with .values
                embeddings = model.get_embeddings(texts)
                return [e.values for e in embeddings]  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback deterministic embedding for dev/tests
        return [[(hash(t) % 997) / 997.0 for _ in range(64)] for t in texts]

    async def retrieve(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> RagResult:
        key = f"vs:{_norm_query(query)}:{k}:{bool(filter)}"
        cached = await self._cache.get(key)
        if cached:
            return RagResult(
                backend_type="vector",
                cache_hit=True,
                text="",
                sources=cached,
            )

        # Attempt Vertex Vector Search when configured
        _lazy_import_vertex()
        s = get_settings()
        if (
            _AIPLATFORM
            and s.PROJECT_ID
            and s.LOCATION
            and self.endpoint_id
            and self.index_id
        ):
            try:  # pragma: no cover - requires cloud
                _AIPLATFORM.init(project=s.PROJECT_ID, location=s.LOCATION)
                # Embed query
                qvec = (await self.embed([query]))[0]
                # Build resource name and query
                endpoint_name = f"projects/{s.PROJECT_ID}/locations/{s.LOCATION}/indexEndpoints/{self.endpoint_id}"
                endpoint = _AIPLATFORM.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
                resp = endpoint.find_neighbors(
                    deployed_index_id=self.index_id,
                    queries=[qvec],
                    num_neighbors=max(1, min(k, 20)),
                )
                neighbors = resp[0].neighbors if resp else []  # type: ignore[attr-defined]
                sources: List[RagSource] = []
                for i, nb in enumerate(neighbors):
                    # name or datapoint id depending on SDK version
                    nid = getattr(nb, "datapoint_id", None) or getattr(nb, "id", f"doc-{i}")
                    distance = getattr(nb, "distance", 0.0)
                    score = max(0.0, 1.0 - float(distance))
                    sources.append(
                        RagSource(
                            title=f"Result {i+1}",
                            snippet=f"Vector match for '{query}'",
                            url_or_doi=str(nid),
                            score=score,
                        )
                    )
                await self._cache.set(key, sources)
                return RagResult(backend_type="vector", cache_hit=False, text="", sources=sources)
            except Exception:
                # fall through to stubbed response
                pass

        # Fallback stubbed response
        sources = [
            RagSource(
                title=f"Doc {i+1} for '{query}'",
                snippet=f"Snippet about {query} (vector backend)",
                url_or_doi=f"https://example.org/{i+1}",
                score=1.0 - i * 0.1,
            )
            for i in range(max(1, min(k, 20)))
        ]
        await self._cache.set(key, sources)
        return RagResult(backend_type="vector", cache_hit=False, text="", sources=sources)


class VertexRagEngine:
    """Adapter to Vertex RAG Engine (managed ingestion + caching)."""

    def __init__(self, project_id: str, location: str, corpus_id: str):
        self.project_id = project_id
        self.location = location
        self.corpus_id = corpus_id
        self._cache = _SimpleLru(max_size=256, ttl_s=300)

    async def retrieve(self, query: str, k: int = 5) -> RagResult:
        key = f"re:{_norm_query(query)}:{k}"
        cached = await self._cache.get(key)
        if cached:
            return RagResult(
                backend_type="engine",
                cache_hit=True,
                text="",
                sources=cached,
            )

        # TODO: Integrate with Vertex RAG Engine once corpus is configured.
        # Fallback synthetic response
        sources = [
            RagSource(
                title=f"Engine doc {i+1} for '{query}'",
                snippet=f"Managed chunk for {query} with citations",
                url_or_doi=f"doi:10.1234/{i+1:04d}",
                score=1.0 - i * 0.1,
            )
            for i in range(max(1, min(k, 20)))
        ]
        await self._cache.set(key, sources)
        return RagResult(backend_type="engine", cache_hit=False, text="", sources=sources)

    async def answer(self, query: str, k: int = 5) -> RagResult:
        # Simple compose: retrieve then synthesize text (LLM call would go here)
        retrieved = await self.retrieve(query, k)
        text = (
            f"Answer for '{query}' grounded on {len(retrieved.sources)} sources."
        )
        return RagResult(
            backend_type=retrieved.backend_type,
            cache_hit=retrieved.cache_hit,
            text=text,
            sources=retrieved.sources,
        )


class _CompositeRag:
    """Facade that exposes `retrieve` and `answer` consistently."""

    def __init__(self) -> None:
        s = get_settings()
        if s.RAG_CORPUS_ID:
            self._backend = VertexRagEngine(s.PROJECT_ID, s.LOCATION, s.RAG_CORPUS_ID)
        else:
            self._backend = VertexVectorBackend(
                s.PROJECT_ID, s.LOCATION, s.VECTOR_INDEX_ID, s.VECTOR_ENDPOINT_ID
            )

    @property
    def backend(self) -> Any:
        return self._backend

    async def retrieve(self, query: str, k: int = 5) -> RagResult:
        # For vector backend, map to RagResult shape
        if isinstance(self._backend, VertexVectorBackend):
            vs = await self._backend.retrieve(query, k)
            return RagResult(
                backend_type=vs.backend_type,
                cache_hit=vs.cache_hit,
                text="",
                sources=vs.sources,
            )
        # Engine already returns RagResult
        return await self._backend.retrieve(query, k)

    async def answer(self, query: str, k: int = 5) -> RagResult:
        if hasattr(self._backend, "answer"):
            return await self._backend.answer(query, k)  # type: ignore
        retrieved = await self._backend.retrieve(query, k)  # type: ignore
        text = f"Answer for '{query}' grounded on {len(retrieved.sources)} sources."
        return RagResult(
            backend_type=retrieved.backend_type,
            cache_hit=retrieved.cache_hit,
            text=text,
            sources=retrieved.sources,
        )

    async def health_check(self) -> bool:
        # Simple health: settings presence implies healthy
        s = get_settings()
        return bool(s.PROJECT_ID or s.RAG_CORPUS_ID or s.VECTOR_INDEX_ID)


_singleton: Optional[_CompositeRag] = None


def get_rag() -> _CompositeRag:
    """Factory returning a singleton RAG adapter based on settings."""
    global _singleton
    if _singleton is None:
        _singleton = _CompositeRag()
    return _singleton
