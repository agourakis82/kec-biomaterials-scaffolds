Gemini Function Calling / Extensions

- Register endpoints as Gemini Extensions; auth via header `X-API-Key`.
- Base URL: set via `BASE_URL` in settings; OpenAPI at `/openapi.json`.
- Functions: RAG (/rag), Memory (/memory), Tree-Search (/tree-search), Contracts (/contracts).

Notes

- Keep function schemas stable for caching and discoverability.
- Return citations (url_or_doi) for RAG results to preserve grounding.

