"""Helpers for integrating Darwin Mode B with Google Cloud services."""
import datetime as _dt
import json
import os
from typing import Any, Dict, List

from google.cloud import aiplatform, bigquery


def get_vertex_models() -> Dict[str, str]:
    """Return the Vertex model configuration from environment or defaults."""
    raw = os.getenv("VERTEX_MODELS_JSON")
    if not raw:
        return {"chat": "text-bison", "embed": "textembedding-gecko@003"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"chat": "text-bison", "embed": "textembedding-gecko@003"}


def init_vertex(project_id: str, location: str) -> None:
    """Initialise the Vertex AI context."""
    aiplatform.init(project=project_id, location=location)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts using Vertex AI."""
    models = get_vertex_models()
    embedding = aiplatform.TextEmbeddingModel.from_pretrained(models["embed"])
    vectors: List[List[float]] = []
    for text in texts:
        snippet = text[:8000]
        response = embedding.get_embeddings([snippet])
        vectors.append(response[0].values)
    return vectors


def chat_complete(prompt: str) -> str:
    """Generate a chat completion using Vertex AI."""
    models = get_vertex_models()
    text_model = aiplatform.TextGenerationModel.from_pretrained(models["chat"])
    result = text_model.predict(prompt, temperature=0.2, max_output_tokens=1024)
    return result.text


def bq_client() -> bigquery.Client:
    """Create a BigQuery client using ambient credentials."""
    return bigquery.Client()


def bq_upsert_document(dataset: str, table: str, row: Dict[str, Any]) -> bool:
    """Insert document metadata into BigQuery (append-only)."""
    client = bq_client()
    table_id = f"{client.project}.{dataset}.{table}"
    payload = dict(row)
    payload.setdefault("created_at", _dt.datetime.utcnow().isoformat())
    errors = client.insert_rows_json(table_id, [payload])
    if errors:
        raise RuntimeError(str(errors))
    return True
