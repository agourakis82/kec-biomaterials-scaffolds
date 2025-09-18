.ONESHELL:
SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

default: help

help:
	@echo "Targets:"
	@echo "  venv        - cria ambiente virtual"
	@echo "  install     - instala dependências"
	@echo "  ingest      - roda cross_ref.py usando seed/dois.txt"
	@echo "  ingest_csv  - idem, mas com seed/references_seed.csv (se existir)"
	@echo "  index       - constrói índice Chroma a partir de build/references.jsonl"
	@echo "  clean       - remove build/ (menos seed/)"

venv:
	python -m venv $(VENV)
	@echo "[ok] venv created"

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	@echo "[ok] requirements installed"

ingest:
	mkdir -p build
	$(PY) cross_ref.py --input seed/dois.txt --outdir build --emit-bibtex --emit-csv --qmap seed/journal_quartiles.csv

ingest_csv:
	mkdir -p build
	$(PY) cross_ref.py --input seed/references_seed.csv --outdir build --emit-bibtex --emit-csv --qmap seed/journal_quartiles.csv

index:
	mkdir -p build/chroma
	$(PY) rag_load.py --jsonl build/references.jsonl --db build/chroma --collection kec_biomat_refs --model all-MiniLM-L6-v2

clean:
	rm -rf build/*
	mkdir -p build
	@echo "[ok] build cleaned"