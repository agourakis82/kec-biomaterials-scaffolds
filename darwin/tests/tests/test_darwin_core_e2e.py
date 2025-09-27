import asyncio
import os
import time
from typing import Any, List, Tuple

import pytest

# Core modules (src is added to sys.path by tests/conftest.py)
from darwin_core.rag.rag_plus import RAGPlusConfig, RAGPlusEngine
from darwin_core.rag.iterative import IterativeConfig, IterativeSearch
from darwin_core.tree_search.puct import PUCTConfig, PUCTSearch
from darwin_core.memory.integrated_memory_system import get_integrated_memory_system

# API (FastAPI)
from fastapi.testclient import TestClient
from src.kec_biomat_api.main import create_app
from src.kec_biomat_api.config import CONFIG as API_SETTINGS


@pytest.mark.asyncio
async def test_core_ragplus_embedding_perf_and_dims():
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    assert await engine.initialize() is True

    text = "biomaterials scaffolds porous structure" * 8
    t0 = time.perf_counter()
    emb = await engine.get_embedding(text)
    dt = time.perf_counter() - t0

    assert isinstance(emb, list)
    assert len(emb) in (64, 768)  # fallback (64) or cloud (768)
    assert dt < 2.0


@pytest.mark.asyncio
async def test_core_iterative_search_integration_with_rag_engine():
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    await engine.initialize()

    it = IterativeSearch(IterativeConfig(max_iterations=3), rag_engine=engine)
    result = await it.search_iteratively("scaffold porosity and tissue ingrowth")

    assert isinstance(result, dict)
    for key in ("answer", "sources", "method", "iterations", "final_score", "converged"):
        assert key in result
    assert result["method"] == "iterative_search"
    assert isinstance(result["sources"], list)


@pytest.mark.asyncio
async def test_core_puct_basic_search_and_stats():
    class MockEvaluator:
        async def expand(self, state: str) -> List[Tuple[str, str, float]]:
            # 2 actions per node until depth runs out
            return [
                ("a", f"{state}a", 0.5),
                ("b", f"{state}b", 0.5),
            ]

        async def rollout(self, state: str, max_steps: int = 5) -> float:
            # Deterministic score based on state length
            return min(len(state) / 10.0, 1.0)

    puct = PUCTSearch(MockEvaluator(), PUCTConfig(default_budget=60, max_depth=6))
    root = await puct.search("S", budget=60, time_limit=1.5)

    stats = puct.get_search_statistics()
    assert root is not None
    assert stats["nodes_count"] > 1
    assert stats["nodes_explored"] > 0
    assert stats["search_time"] >= 0.0


@pytest.mark.asyncio
async def test_core_memory_integrated_system_context():
    ims = await get_integrated_memory_system()
    ctx = await ims.get_complete_project_context()

    # Sanity checks on structure
    assert "session_context" in ctx
    assert "project_state" in ctx
    assert "immediate_context" in ctx
    assert "four_steps_progress" in ctx
    assert "memory_systems" in ctx


def test_api_smoke_healthz_and_rag_plus_endpoints():
    app = create_app()
    client = TestClient(app)

    # Health
    r = client.get("/healthz")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"

    # RAG++ status (should work without API key by default)
    r = client.get("/rag-plus/status")
    assert r.status_code == 200
    j = r.json()
    assert "service" in j and "components" in j

    # RAG++ query
    r = client.post("/rag-plus/query", json={"query": "scaffolds porosity", "top_k": 2})
    assert r.status_code == 200
    j = r.json()
    for k in ("answer", "sources", "method", "retrieved_docs"):
        assert k in j


def test_api_security_rag_plus_requires_api_key_when_enabled():
    # Backup current settings
    prev_required = API_SETTINGS.API_KEY_REQUIRED
    prev_keys = API_SETTINGS.API_KEYS

    try:
        # Enforce API key for this test
        API_SETTINGS.API_KEYS = "test-key"
        API_SETTINGS.API_KEY_REQUIRED = True

        app = create_app()
        client = TestClient(app)

        # Without key should fail
        r = client.get("/rag-plus/status")
        assert r.status_code in (401, 403)

        # With Bearer should pass
        r = client.get("/rag-plus/status", headers={"Authorization": "Bearer test-key"})
        assert r.status_code == 200
        j = r.json()
        assert "service" in j

    finally:
        # Restore settings to avoid affecting other tests
        API_SETTINGS.API_KEY_REQUIRED = prev_required
        API_SETTINGS.API_KEYS = prev_keys


@pytest.mark.asyncio
async def test_core_ragplus_concurrency_embeddings():
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    await engine.initialize()

    texts = [f"sample text {i}" for i in range(6)]
    t0 = time.perf_counter()
    embs = await asyncio.gather(*(engine.get_embedding(t) for t in texts))
    dt = time.perf_counter() - t0

    assert len(embs) == len(texts)
    for e in embs:
        assert isinstance(e, list) and len(e) in (64, 768)
    # Concurrent calls should complete quickly in fallback mode
    assert dt < 3.0