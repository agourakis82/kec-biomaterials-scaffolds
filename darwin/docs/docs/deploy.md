Cloud Run Deploy (FastAPI)

- Use Cloud Run with a Python FastAPI container.
- Map secrets via Secret Manager to env vars for keys (e.g., GEMINI/OPENAI).
- Expose `/.well-known/ai-plugin.json` and `/openapi.json` for Actions/Extensions.

Quickstart

- Build and deploy from source: gcloud run deploy --source . --region $LOCATION
- Set envs: PROJECT_ID, LOCATION, RAG_CORPUS_ID or VECTOR_INDEX_ID/ENDPOINT_ID.
- Configure `BASE_URL` for correct OpenAPI server URL.

Pricing Pointers

- Vertex Vector Search indexing ≈ (examples * dim * 4 bytes) billed at $/GB.
- Serving billed by queries/storage; new users often receive credits (~$1,000).
- Context Cache yields ~75% discount for cached input tokens.

