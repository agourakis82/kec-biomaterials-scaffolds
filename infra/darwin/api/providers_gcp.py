"""Helpers for integrating Darwin Mode B with Google Cloud services."""
import datetime as _dt
import json
import os
from typing import Any, Dict, List

import vertexai
from google.cloud import bigquery


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
    vertexai.init(project=project_id, location=location)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts using Vertex AI."""
    try:
        from vertexai.language_models import TextEmbeddingModel
        
        models = get_vertex_models()
        model_name = models["embed"]
        
        # Handle different embedding model types
        if model_name.startswith("text-embedding"):
            embedding = TextEmbeddingModel.from_pretrained(model_name)
            vectors: List[List[float]] = []
            for text in texts:
                snippet = text[:8000]
                response = embedding.get_embeddings([snippet])
                vectors.append(response[0].values)
            return vectors
        else:
            # Fallback for older models
            embedding = TextEmbeddingModel.from_pretrained(model_name)
            vectors: List[List[float]] = []
            for text in texts:
                snippet = text[:8000]
                response = embedding.get_embeddings([snippet])
                vectors.append(response[0].values)
            return vectors
    except Exception as e:
        # Fallback: return dummy embeddings when Vertex AI is not available
        import numpy as np
        dummy_vector = [0.1] * 768  # Standard embedding dimension
        return [dummy_vector] * len(texts)


def chat_complete(prompt: str) -> str:
    """Generate a chat completion using Vertex AI."""
    try:
        models = get_vertex_models()
        model_name = models["chat"]
        
        # Handle different chat model types
        if model_name.startswith("gemini"):
            from vertexai.generative_models import GenerativeModel
            model = GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        else:
            # Fallback for older models
            from vertexai.language_models import TextGenerationModel
            text_model = TextGenerationModel.from_pretrained(model_name)
            result = text_model.predict(prompt, temperature=0.2, max_output_tokens=1024)
            return result.text
    except Exception as e:
        # Fallback: return a simple response when Vertex AI is not available
        return f"I apologize, but I'm currently unable to generate a response using AI models. The query was: {prompt[:100]}..."


def bq_client() -> bigquery.Client:
    """Create a BigQuery client using ambient credentials."""
    return bigquery.Client()


def bq_upsert_document(dataset: str, table: str, row: Dict[str, Any]) -> bool:
    """Insert document metadata into BigQuery (append-only)."""
    import logging
    logger = logging.getLogger()  # Use root logger
    
    try:
        client = bq_client()
        table_id = f"{client.project}.{dataset}.{table}"
        logger.info(f"Inserting document into BigQuery table: {table_id}")
        
        payload = dict(row)
        payload.setdefault("created_at", _dt.datetime.utcnow().isoformat())
        logger.info(f"Payload to insert: {payload}")
        
        errors = client.insert_rows_json(table_id, [payload])
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise RuntimeError(str(errors))
        
        logger.info("Successfully inserted document into BigQuery")
        return True
    except Exception as e:
        logger.error(f"Failed to insert document into BigQuery: {str(e)}")
        raise
