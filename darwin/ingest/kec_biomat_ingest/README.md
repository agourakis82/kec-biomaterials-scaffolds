## Indexação RAG (Chroma)

Após gerar o `build/references.jsonl` com o `cross_ref.py`, crie o índice Chroma:

```bash
make install
make ingest        # ou make ingest_csv se tiver um CSV
make index
```

* O índice persistente ficará em `build/chroma`.
* Uma prévia tabular é salva em `build/corpus_preview.parquet`.
* O loader usa `sentence-transformers` (modelo default: `all-MiniLM-L6-v2`).

```
```
