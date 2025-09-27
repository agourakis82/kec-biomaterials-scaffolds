import time
import pytest

from darwin_core.rag.rag_plus import RAGPlusConfig, RAGPlusEngine


@pytest.mark.asyncio
async def test_ragplus_initialize_and_embedding_fallback():
    # Arrange
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)

    # Act
    ok = await engine.initialize()

    # Assert (engine should initialize even if cloud deps are unavailable)
    assert ok is True

    # Embedding should work fast and be deterministic in fallback mode
    t0 = time.perf_counter()
    emb = await engine.get_embedding("biomaterials scaffolds porous structure" * 10)
    dt = time.perf_counter() - t0

    assert isinstance(emb, list)
    assert len(emb) in (64, 768)  # fallback (64) or cloud (768)
    # Keep a relaxed bound to avoid flakiness in CI
    assert dt < 2.0


@pytest.mark.asyncio
async def test_ragplus_add_and_query_without_bigquery():
    # Arrange
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    await engine.initialize()

    # Act
    added = await engine.add_document(
        doc_id="doc-001",
        content="Biomaterials scaffolds are porous structures that enable tissue ingrowth.",
        source="unit-test",
        metadata={"domain": "biomaterials", "tags": ["scaffold", "porosity"]},
        discovery_type="manual",
    )

    # If BigQuery isn't configured, add_document returns True (no-op) and query returns []
    results = await engine.query_knowledge_base("scaffold porosity", top_k=3)

    # Assert
    assert added is True
    assert isinstance(results, list)
    assert len(results) == 0  # fallback path without BQ should be empty


@pytest.mark.asyncio
async def test_ragplus_answer_question_structure():
    # Arrange
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    await engine.initialize()

    # Act
    out = await engine.answer_question("What are key properties of scaffolds in tissue engineering?")

    # Assert response schema
    assert isinstance(out, dict)
    for key in ("answer", "sources", "method", "retrieved_docs", "query"):
        assert key in out

    assert isinstance(out["answer"], str)
    assert isinstance(out["sources"], list)
    assert isinstance(out["retrieved_docs"], int)
    assert out["retrieved_docs"] == len(out["sources"])


@pytest.mark.asyncio
async def test_ragplus_status_basic():
    # Arrange
    cfg = RAGPlusConfig.from_env()
    engine = RAGPlusEngine(cfg)
    await engine.initialize()

    # Act
    status = await engine.get_status()

    # Assert
    assert status["service"] == "rag_plus_engine"
    assert "components" in status
    assert "configuration" in status
    assert "vertex_embeddings" in status["components"]
    assert "bigquery" in status["components"]