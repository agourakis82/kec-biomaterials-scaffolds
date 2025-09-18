import asyncio

from services.rag_vertex import get_rag


def test_rag_retrieve_shape():
    rag = get_rag()

    async def run():
        res = await rag.retrieve("test query", k=3)
        assert res.sources and len(res.sources) == 3
        for s in res.sources:
            assert hasattr(s, "title") and s.title
            assert hasattr(s, "snippet") and s.snippet
            assert hasattr(s, "score")
            # url_or_doi may be optional, but present in stubs
            assert s.url_or_doi is None or isinstance(s.url_or_doi, str)

    asyncio.run(run())


def test_rag_answer_includes_text():
    rag = get_rag()

    async def run():
        res = await rag.answer("what is KEC?", k=2)
        assert isinstance(res.text, str) and res.text
        assert res.sources and len(res.sources) == 2

    asyncio.run(run())

