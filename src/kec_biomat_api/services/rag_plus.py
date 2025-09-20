"""RAG++ service integrating Vertex embeddings and BigQuery storage (optional).

This implementation uses configuration from infra.api.config.Settings via
services.settings.get_settings(). All cloud calls are guarded and fallback to
safe stubs in local development.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from kec_biomat_api.config import settings

# Lazy import clients to keep local dev light
_AIPLATFORM = None
_VERTEXAI = None
_BIGQUERY = None


def _lazy_imports():
    global _AIPLATFORM, _VERTEXAI, _BIGQUERY
    if _AIPLATFORM is None:
        try:
            from google.cloud import aiplatform as _gcaip  # type: ignore
            _AIPLATFORM = _gcaip
        except Exception:
            _AIPLATFORM = False
    if _VERTEXAI is None:
        try:
            from vertexai.preview.language_models import TextEmbeddingModel as _vem  # type: ignore
            _VERTEXAI = _vem
        except Exception:
            _VERTEXAI = False
    if _BIGQUERY is None:
        try:
            from google.cloud import bigquery as _gbq  # type: ignore
            _BIGQUERY = _gbq
        except Exception:
            _BIGQUERY = False


@dataclass
class _Config:
    project_id: str
    location: str
    dataset_id: str
    table_id: str
    embedding_model: str
    generation_model: str
    novelty_threshold: float
    max_iterations: int
    top_k_retrieval: int
    discovery_enabled: bool
    discovery_interval: int


class _Source:
    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.type = "rss"
        self.enabled = False
        self.check_interval = 3600


class DarwinRAGPlusService:
    def __init__(self) -> None:
        s = settings
        self.config = _Config(
            project_id=s.PROJECT_ID,
            location=s.LOCATION,
            dataset_id=getattr(s, "RAG_PLUS_DATASET_ID", ""),
            table_id=getattr(s, "RAG_PLUS_TABLE_ID", ""),
            embedding_model=s.VERTEX_EMB_MODEL,
            generation_model="gemini-1.5-flash",
            novelty_threshold=0.3,
            max_iterations=5,
            top_k_retrieval=5,
            discovery_enabled=False,
            discovery_interval=3600,
        )
        self.sources = [_Source("arXiv", "https://arxiv.org/rss/cs.AI")]

    def _init_vertex(self) -> bool:
        _lazy_imports()
        s = settings
        if _AIPLATFORM and _VERTEXAI and s.PROJECT_ID and s.LOCATION:
            try:
                _AIPLATFORM.init(project=s.PROJECT_ID, location=s.LOCATION)
                return True
            except Exception:
                return False
        return False

    def _bq_client(self):
        _lazy_imports()
        if _BIGQUERY and self.config.project_id:
            try:
                return _BIGQUERY.Client(project=self.config.project_id)
            except Exception:
                return None
        return None

    async def get_embedding(self, text: str) -> List[float]:
        if self._init_vertex() and self.config.embedding_model:
            try:  # pragma: no cover - requires cloud
                model = _VERTEXAI.from_pretrained(self.config.embedding_model)
                emb = model.get_embeddings([text])[0]
                return emb.values  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback small deterministic vector
        return [(hash(text) % 1000) / 1000.0 for _ in range(64)]

    async def answer_question(self, query: str) -> Dict[str, Any]:
        # Simple pipeline: get top_k docs (keyword fallback) and return stubbed summary
        docs = await self.query_knowledge_base(query, top_k=self.config.top_k_retrieval)
        answer = f"Found {len(docs)} relevant document(s) for: {query}"
        return {
            "answer": answer,
            "sources": docs,
            "method": "simple",
            "retrieved_docs": len(docs),
        }

    async def answer_question_iterative(self, query: str) -> Dict[str, Any]:
        docs = await self.query_knowledge_base(query, top_k=self.config.top_k_retrieval)
        steps = [{"thought": "retrieve", "action": "search_bigquery", "k": len(docs)}]
        answer = f"Iterative reasoning produced {len(docs)} sources for: {query}"
        return {
            "answer": answer,
            "sources": docs,
            "method": "iterative",
            "retrieved_docs": len(docs),
            "reasoning_steps": steps,
            "total_steps": len(steps),
        }

    async def discover_new_knowledge(self) -> Dict[str, int]:
        # Placeholder discovery job
        return {"fetched": 0, "novel": 0, "added": 0, "errors": 0}

    async def get_service_status(self) -> Dict[str, Any]:
        components: Dict[str, Any] = {}
        bq_ok = bool(self._bq_client() and self.config.dataset_id and self.config.table_id)
        components["bigquery"] = "ok" if bq_ok else "disabled"
        components["vertex_embeddings"] = "ok" if self._init_vertex() else "disabled"
        return {
            "service": "rag_plus",
            "status": "ready",
            "components": components,
            "configuration": self.config.__dict__,
            "timestamp": "",
        }

    async def start_continuous_discovery(self) -> None:
        self.config.discovery_enabled = True

    async def stop_continuous_discovery(self) -> None:
        self.config.discovery_enabled = False

    async def add_document(
        self,
        doc_id: str,
        content: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        discovery_type: str = "manual",
    ) -> bool:
        client = self._bq_client()
        if not client or not self.config.dataset_id or not self.config.table_id:
            return True  # No-op in local dev
        try:  # pragma: no cover - requires cloud
            emb = await self.get_embedding(content)
            row = {
                "id": doc_id,
                "content": content,
                "embedding": emb,
                "source": source,
                "metadata": metadata or {},
            }
            table = f"{self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}"
            errors = client.insert_rows_json(table, [row])  # type: ignore[arg-type]
            return not errors
        except Exception:
            return False

    async def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        client = self._bq_client()
        if not client or not self.config.dataset_id or not self.config.table_id:
            # Local fallback: return empty results
            return []
        try:  # pragma: no cover - requires cloud
            # Keyword fallback with simple scoring
            table = f"{self.config.project_id}.{self.config.dataset_id}.{self.config.table_id}"
            sql = f"""
            SELECT id, content, source, metadata
            FROM `{table}`
            WHERE LOWER(content) LIKE LOWER(@q)
            ORDER BY RAND()
            LIMIT @k
            """
            job = client.query(sql, job_config=_BIGQUERY.QueryJobConfig(
                query_parameters=[
                    _BIGQUERY.ScalarQueryParameter("q", "STRING", f"%{query}%"),
                    _BIGQUERY.ScalarQueryParameter("k", "INT64", top_k),
                ]
            ))
            rows = list(job.result())
            return [
                {
                    "title": r.get("id"),
                    "snippet": (r.get("content") or "")[:200],
                    "url_or_doi": r.get("source"),
                    "score": 0.5,
                }
                for r in rows
            ]
        except Exception:
            return []


_SERVICE = DarwinRAGPlusService()


async def get_rag_plus_service() -> DarwinRAGPlusService:
    return _SERVICE
