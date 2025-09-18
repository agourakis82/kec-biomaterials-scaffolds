# KEC Metrics for Porous Biomaterial Scaffolds

Exact implementation of the MSc project pipeline:
micro-CT → segmentation → pore-graph extraction → KEC (Entropy H, Curvature κ_OR/Forman,
Coherence σ/ϕ) + percolation diameter (d_perc) → predictive models.

**Scope lock:** strictly adhere to the PDF (Methods pp. 7–10; Fig. 1 p. 6; Expected Results pp. 11–13).
Data: public μCT sources only; no raw data stored here—metadata and links only.

Folders: /data, /src/kec_biomat, /notebooks, /results, /infra (Darwin bridge).

## Darwin API — Biomaterials Bridge

The Darwin API provides a FastAPI-based RAG and memory service for biomaterials research, deployable locally or on Google Cloud Run.

### Quickstart (Local)

```bash
# Install dependencies (in infra/api)
cd infra/api
pip install -r requirements.txt

# Copy .env.example and edit secrets as needed
cp .env.example .env

# Run locally
make run
# Or directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Quickstart (Cloud Run)

```bash
export GCP_PROJECT_ID=<your-gcp-project>
export GCP_REGION=us-central1
./infra/api/deploy/cloudrun_deploy.sh
```

### Python Client Usage

```python
from darwin_client import DarwinClient
client = DarwinClient(base_url="http://localhost:8080", api_key="replace-with-secret")

# Health check
print(client.health())

# Query RAG
result = client.query("What is percolation diameter?", top_k=3)
print(result)

# Index a document
doc = client.index(text="Sample document text", title="Test Doc")
print(doc)

# Iterative query
iter_result = client.query_iterative("Explain entropy in biomaterials", max_iters=2)
print(iter_result)

# Tree search
tree = client.tree_search("Find optimal scaffold", max_depth=2)
print(tree)
```

API endpoints: `/healthz`, `/rag`, `/rag/index`, `/tree-search/search`, `/rag-plus/query` and more.

