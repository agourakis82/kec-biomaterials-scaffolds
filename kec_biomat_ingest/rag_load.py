#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_load.py — constrói um índice ChromaDB a partir de references.jsonl
- Embeddings: sentence-transformers (all-MiniLM-L6-v2, default)
- Persistência: ./build/chroma
- Entrada: ./build/references.jsonl (um JSON por linha, formato do cross_ref.py)
Saídas auxiliares:
  - ./build/corpus_preview.parquet (amostra/tabular compacta)
"""
import argparse, os, json, sys, pathlib
from pathlib import Path
import pandas as pd

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_docs(rows):
    ids, docs, metas = [], [], []
    for r in rows:
        rid = r.get("id")
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        journal = r.get("journal") or ""
        year = r.get("year")
        year_str = str(year) if year is not None else ""
        kws = ", ".join(r.get("keywords") or [])
        cats = ", ".join(r.get("categories") or [])
        doi = r.get("doi") or ""
        impact = r.get("impact") or ""
        # documento textual para embedding
        text = "\n".join([
            f"TITLE: {title}",
            f"ABSTRACT: {abstract}",
            f"JOURNAL: {journal} ({year_str})",
            f"KEYWORDS: {kws}",
            f"CATEGORIES: {cats}",
            f"DOI: {doi}",
            f"IMPACT: {impact}",
        ])
        ids.append(rid)
        docs.append(text)
        metas.append({
            "id": rid,
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "categories": ", ".join(r.get("categories") or []),
            "impact": impact,
            "relevance": r.get("relevance"),
            "cited_by": r.get("cited_by")
        })
    return ids, docs, metas

def main():
    ap = argparse.ArgumentParser(description="KEC_BIOMAT — constrói índice Chroma a partir de references.jsonl")
    ap.add_argument("--jsonl", default="build/references.jsonl", help="Caminho do references.jsonl")
    ap.add_argument("--db", default="build/chroma", help="Diretório de persistência do Chroma")
    ap.add_argument("--collection", default="kec_biomat_refs", help="Nome da coleção no Chroma")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="Modelo sentence-transformers")
    args = ap.parse_args()

    base = Path(".").resolve()
    jsonl_path = base / args.jsonl
    db_dir = base / args.db
    db_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        print(f"[erro] Arquivo não encontrado: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(jsonl_path)
    if not rows:
        print("[erro] JSONL vazio.", file=sys.stderr)
        sys.exit(1)

    ids, docs, metas = build_docs(rows)

    # Preview tabular
    try:
        import pyarrow  # noqa: F401
        df = pd.DataFrame([{
            "id": m["id"],
            "title": m["title"],
            "journal": m["journal"],
            "year": m["year"],
            "doi": m["doi"],
            "categories": ", ".join(m["categories"]) if isinstance(m["categories"], list) else m["categories"],
            "impact": m["impact"],
            "relevance": m["relevance"],
            "cited_by": m["cited_by"],
        } for m in metas])
        (base / "build" / "corpus_preview.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(base / "build" / "corpus_preview.parquet", index=False)
    except Exception:
        pass

    # Chroma client + embedding function
    ef = SentenceTransformerEmbeddingFunction(model_name=args.model)
    client = chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))
    try:
        coll = client.get_collection(name=args.collection, embedding_function=ef)
    except Exception:
        coll = client.create_collection(name=args.collection, embedding_function=ef)

    # Upsert
    coll.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"[ok] Coleção '{args.collection}' atualizada com {len(ids)} itens em {db_dir}")

if __name__ == "__main__":
    main()
