#!/usr/bin/env python3
"""
Script para exportar conversas do ChatGPT 5-pro via API e sincronizar com DARWIN.
- Redis: Armazena diálogos em tempo real (memória de conversas).
- ChromaDB: Armazena embeddings para busca semântica no RAG++.

Uso:
python3 sync_chatgpt_to_darwin.py --api-key sk-... --limit 10 --domain "biomaterials"

Pré-requisitos:
- OPENAI_API_KEY no .env ou via --api-key
- Redis rodando (localhost:6379)
- ChromaDB rodando (localhost:8000)
- pip install openai redis chromadb sentence-transformers requests
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Any

import openai
import redis
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions
import requests  # Para fallback se necessário

# Configuração
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ou via argumento
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8000")
COLLECTION_NAME = "chatgpt_conversations"  # Coleção no ChromaDB para RAG++

# Cliente Redis
r = redis.from_url(REDIS_URL)

# Cliente ChromaDB
chroma_client = ChromaClient(ChromaClient.Settings(anonymize_new_users=False, chroma_server_host="localhost", chroma_server_http_port=8000))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # Para busca semântica
)

# Embedding function (leve para local)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def export_chatgpt_conversations(api_key: str, limit: int = 10, domain: str = None):
    """
    Exporta conversas do ChatGPT via API.
    Nota: A API oficial para conversas é limitada/beta. Para uso real, use export manual do UI (conversations.json) ou endpoint específico.
    Aqui, simulo com dados fictícios para demo; substitua pela API real se disponível.
    """
    openai.api_key = api_key
    conversations = []
    
    # Exemplo com API OpenAI (para conversas, use endpoint beta se disponível; senão, export manual)
    # response = openai.beta.conversations.list(limit=limit)  # Se disponível
    # conversations = response.data
    
    # Para demo, use dados fictícios ou carregue de JSON exportado
    # Carregue de arquivo JSON exportado do ChatGPT UI (baixe manualmente)
    json_file = os.getenv("CHATGPT_EXPORT_JSON", "chatgpt_export.json")
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            conversations = data.get("conversations", [])[:limit]
    else:
        # Dados fictícios para demo
        conversations = [
            {
                "id": "conv_1",
                "title": "Scaffold Analysis",
                "create_time": "2025-09-24T10:00:00Z",
                "update_time": "2025-09-24T10:30:00Z",
                "messages": [
                    {"role": "user", "content": "Analise scaffold com porosidade 0.6", "timestamp": "2025-09-24T10:01:00Z"},
                    {"role": "assistant", "content": "Scaffold de grafeno com porosidade 0.6 tem biocompatibilidade alta.", "timestamp": "2025-09-24T10:01:30Z"}
                ]
            },
            {
                "id": "conv_2",
                "title": "Quantum Effects in Biomaterials",
                "create_time": "2025-09-24T11:00:00Z",
                "update_time": "2025-09-24T11:15:00Z",
                "messages": [
                    {"role": "user", "content": "Impacto quântico em scaffolds", "timestamp": "2025-09-24T11:01:00Z"},
                    {"role": "assistant", "content": "Efeitos quânticos melhoram regeneração tecidual em 25%.", "timestamp": "2025-09-24T11:01:45Z"}
                ]
            }
        ]  # Limite a 'limit' conversas; filtre por domain se necessário
    
    if domain:
        conversations = [conv for conv in conversations if domain.lower() in conv.get("title", "").lower()]
    
    print(f"Exportadas {len(conversations)} conversas do ChatGPT.")
    return conversations

def save_to_redis(conversations: List[Dict]):
    """Salva diálogos em Redis para memória em tempo real."""
    for conv in conversations:
        key = f"conversation:{conv['id']}"
        r.set(key, json.dumps(conv), ex=86400)  # Expira em 24h
        print(f"Salvo em Redis: {key}")

def save_to_chromadb(conversations: List[Dict]):
    """Gera embeddings e salva em ChromaDB para RAG++."""
    documents = []
    metadatas = []
    ids = []
    
    for conv in conversations:
        full_text = " ".join([msg["content"] for msg in conv["messages"]])
        documents.append(full_text)
        metadatas.append({
            "conversation_id": conv["id"],
            "title": conv["title"],
            "domain": "biomaterials",  # Inferir ou passar como param
            "create_time": conv["create_time"],
            "update_time": conv["update_time"]
        })
        ids.append(conv["id"])
    
    # Gera embeddings e adiciona à coleção
    embeddings = embedding_fn(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Salvo {len(conversations)} conversas em ChromaDB: {COLLECTION_NAME}")

def main():
    parser = argparse.ArgumentParser(description="Sync ChatGPT to DARWIN")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--limit", type=int, default=10, help="Número de conversas a exportar")
    parser.add_argument("--domain", default=None, help="Filtro por domínio (ex.: biomaterials)")
    
    args = parser.parse_args()
    
    print("Iniciando sync ChatGPT -> DARWIN...")
    
    # Exporta conversas
    conversations = export_chatgpt_conversations(args.api_key, args.limit, args.domain)
    
    # Salva em Redis
    save_to_redis(conversations)
    
    # Salva em ChromaDB
    save_to_chromadb(conversations)
    
    print("Sync concluído! Use no DARWIN para RAG++ e memória de diálogos.")

if __name__ == "__main__":
    main()