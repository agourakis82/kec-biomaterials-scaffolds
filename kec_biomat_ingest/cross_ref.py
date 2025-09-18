#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_ref.py — Seed corpus ingestion & enrichment for KEC_BIOMAT (MSc)
Author: Demetrios C. Agourakis (agourakis82) + Copilot
License: MIT
DOI/Project: (add your Zenodo DOI here)
Description:
    - Reads a list of references (DOIs or metadata) from CSV/JSON/JSONL/TXT
    - Enriches using Crossref API (polite headers), optionally OpenAlex (fallback)
    - Normalizes into a RAG-ready JSONL: one JSON record per line
    - Optionally emits BibTeX and a compact CSV
    - Designed to be deterministic/reproducible (cache hits persist)
Usage:
    python cross_ref.py --input seed/references_seed.csv --outdir build --emit-bibtex --emit-csv
    python cross_ref.py --input seed/dois.txt --outdir build --tag A --relevance 5
    python cross_ref.py --input seed/references_seed.jsonl --outdir build --qmap seed/journal_quartiles.csv
Inputs accepted:
    - .txt: one DOI per line (or URL containing DOI)
    - .csv: columns can include: doi,title,year,authors,category,impact,relevance,notes
    - .json/.jsonl: arbitrary fields; at least a 'doi' or 'title' is recommended
Outputs:
    - build/references.jsonl (RAG-ready)
    - build/references.bib (optional)
    - build/references.csv (optional)
    - build/logs/ingest_*.log
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except Exception:
    print("Please `pip install requests tqdm bibtexparser python-dateutil`", file=sys.stderr)
    raise

from urllib.parse import quote
from dateutil import parser as dtparser

try:
    from tqdm import tqdm
except Exception:
    # Fallback no-op if tqdm not installed
    def tqdm(x, **kwargs):
        return x

# ---------------------------
# Config & Constants
# ---------------------------

DEFAULT_USER_AGENT = "KEC_BIOMAT/1.0 (mailto:demetrios@agourakis.med.br)"
CROSSREF_API = "https://api.crossref.org/works/"
OPENALEX_API = "https://api.openalex.org/works/"
CACHE_DIR = ".cache"

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_\.]+", "_", s.strip())
    return s[:120]

def clean_doi(x: str) -> Optional[str]:
    if not x:
        return None
    x = x.strip()
    # Extract DOI from common URL forms
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x, flags=re.I)
    x = re.sub(r"^doi:\s*", "", x, flags=re.I)
    x = x.strip(" .")
    return x or None

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ---------------------------
# Data Structures
# ---------------------------

@dataclass
class SeedRef:
    doi: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    journal: Optional[str] = None
    category: Optional[str] = None  # A/B/C/D
    impact: Optional[str] = None    # e.g., Q1/Q2/Review/Experimental
    relevance: Optional[int] = None # 1–5
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RagRecord:
    id: str
    doi: Optional[str]
    url: Optional[str]
    title: Optional[str]
    abstract: Optional[str]
    year: Optional[int]
    journal: Optional[str]
    authors: List[Dict[str, str]]
    categories: List[str]
    impact: Optional[str]
    relevance: Optional[int]
    publisher: Optional[str]
    issn: List[str]
    keywords: List[str]
    cited_by: Optional[int]
    references_count: Optional[int]
    source: Dict[str, Any]        # raw api payloads (crossref/opena
    created_at: str
    updated_at: str

# ---------------------------
# Input Parsers
# ---------------------------

def read_input(path: Path) -> List[SeedRef]:
    ext = path.suffix.lower()
    items: List[SeedRef] = []
    if ext == ".txt":
        for line in path.read_text(encoding="utf-8").splitlines():
            doi = clean_doi(line)
            if doi:
                items.append(SeedRef(doi=doi))
    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doi = clean_doi(row.get("doi") or row.get("DOI", ""))
                title = (row.get("title") or row.get("Title") or "").strip() or None
                year = row.get("year") or row.get("Year")
                year = int(year) if year and str(year).isdigit() else None
                authors = [a.strip() for a in (row.get("authors") or "").split(";") if a.strip()]
                journal = (row.get("journal") or "").strip() or None
                category = (row.get("category") or "").strip() or None
                impact = (row.get("impact") or "").strip() or None
                relevance = row.get("relevance")
                relevance = int(relevance) if relevance and str(relevance).isdigit() else None
                notes = (row.get("notes") or "").strip() or None
                extra = {k:v for k,v in row.items() if k.lower() not in {"doi","title","year","authors","journal","category","impact","relevance","notes"}}
                items.append(SeedRef(doi, title, year, authors, journal, category, impact, relevance, notes, extra))
    elif ext in [".json", ".jsonl"]:
        lines = path.read_text(encoding="utf-8").splitlines() if ext==".jsonl" else [path.read_text(encoding="utf-8")]
        for ln in lines:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            doi = clean_doi(obj.get("doi") or obj.get("DOI") or "")
            title = obj.get("title")
            year = obj.get("year")
            try:
                year = int(year) if year is not None else None
            except Exception:
                year = None
            authors = obj.get("authors") or []
            authors = authors if isinstance(authors, list) else []
            journal = obj.get("journal")
            category = obj.get("category")
            impact = obj.get("impact")
            relevance = obj.get("relevance")
            try:
                relevance = int(relevance) if relevance is not None else None
            except Exception:
                relevance = None
            notes = obj.get("notes")
            extra = {k:v for k,v in obj.items() if k not in {"doi","DOI","title","year","authors","journal","category","impact","relevance","notes"}}
            items.append(SeedRef(doi, title, year, authors, journal, category, impact, relevance, notes, extra))
    else:
        raise ValueError(f"Unsupported input extension: {ext}")
    return items

# ---------------------------
# HTTP fetch with polite headers, caching & backoff
# ---------------------------

def http_get_json(url: str, headers: Dict[str,str], cache_dir: Path, max_retries=5, backoff=1.5) -> Dict[str, Any]:
    ensure_dir(cache_dir)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    cache_path = cache_dir / f"{h}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return data
            elif r.status_code == 404:
                return {}
            elif r.status_code == 429:
                # rate limited — exponential backoff
                time.sleep((i+1) * backoff * 2)
            else:
                time.sleep((i+1) * backoff)
        except requests.RequestException:
            time.sleep((i+1) * backoff)
    return {}

def fetch_crossref(doi: str, ua: str, cache_dir: Path) -> Dict[str, Any]:
    url = CROSSREF_API + quote(doi)
    return http_get_json(url, headers={"User-Agent": ua}, cache_dir=cache_dir / "crossref")

def fetch_openalex(doi: str, ua: str, cache_dir: Path) -> Dict[str, Any]:
    url = OPENALEX_API + f"doi:{quote(doi)}"
    return http_get_json(url, headers={"User-Agent": ua}, cache_dir=cache_dir / "openalex")

# ---------------------------
# Normalizers
# ---------------------------

def norm_crossref(work: Dict[str, Any]) -> Dict[str, Any]:
    if not work:
        return {}
    item = work.get("message") or {}
    title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title")
    abstract = item.get("abstract")
    if abstract and abstract.startswith("<"):
        # strip simple tags
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
    authors = []
    for a in item.get("author", []) or []:
        given = a.get("given") or ""
        family = a.get("family") or ""
        name = (given + " " + family).strip() or a.get("name")
        authors.append({"given": given, "family": family, "name": name})
    issued = item.get("issued", {}).get("date-parts", [[None]])[0]
    year = issued[0] if issued and isinstance(issued, list) else None
    container = item.get("container-title") or []
    journal = container[0] if container else None
    issn = item.get("ISSN") or []
    url = item.get("URL")
    doi = item.get("DOI")
    subjects = item.get("subject") or []
    kws = list(dict.fromkeys(subjects))  # unique & order-stable
    cited_by = item.get("is-referenced-by-count")
    refs_count = len(item.get("reference", []) or [])
    publisher = item.get("publisher")

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "journal": journal,
        "issn": issn,
        "url": url,
        "doi": doi,
        "keywords": kws,
        "cited_by": cited_by,
        "references_count": refs_count,
        "publisher": publisher,
        "raw": item,
    }

def norm_openalex(work: Dict[str, Any]) -> Dict[str, Any]:
    if not work or "results" not in work:
        return {}
    # OpenAlex returns 'results' when querying by DOI search; when using specific ID, structure differs.
    # We handle both.
    item = None
    if "results" in work and work["results"]:
        item = work["results"][0]
    elif "id" in work:
        item = work
    if not item:
        return {}
    title = item.get("title")
    year = item.get("publication_year")
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    url = item.get("primary_location", {}).get("source", {}).get("host_organization_lineage", [])
    auths = []
    for a in item.get("authorships", []) or []:
        name = a.get("author", {}).get("display_name")
        parts = (name or "").split()
        given = " ".join(parts[:-1]) if len(parts) > 1 else name
        family = parts[-1] if len(parts) > 1 else ""
        auths.append({"given": given, "family": family, "name": name})
    journal = item.get("primary_location", {}).get("source", {}).get("display_name")
    issn = item.get("primary_location", {}).get("source", {}).get("issn_l")
    issn = [issn] if issn else []
    kws = [c.get("display_name") for c in item.get("concepts", []) or []]
    cited_by = item.get("cited_by_count")
    refs_count = None
    publisher = item.get("primary_location", {}).get("source", {}).get("host_organization")

    return {
        "title": title,
        "abstract": None,
        "authors": auths,
        "year": year,
        "journal": journal,
        "issn": issn,
        "url": None,
        "doi": doi or None,
        "keywords": kws,
        "cited_by": cited_by,
        "references_count": refs_count,
        "publisher": publisher,
        "raw": item,
    }

def merge_priority(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dicts preferring non-empty values from `a`, with `b` as fallback."""
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va in [None, "", [], {}]:
            out[k] = vb
        else:
            out[k] = va
    return out

# ---------------------------
# Quartile & Impact Map
# ---------------------------

def load_quartiles(path: Optional[Path]) -> Dict[str, str]:
    qmap = {}
    if not path:
        return qmap
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            j = (row.get("journal") or "").strip().lower()
            q = (row.get("quartile") or "").strip().upper()
            if j:
                qmap[j] = q
    return qmap

# ---------------------------
# Emitters
# ---------------------------

def to_bibtex(rec: RagRecord) -> str:
    key = slugify((rec.authors[0]["family"] if rec.authors else "anon") + "_" + (str(rec.year) if rec.year else "n.d."))
    lines = ["@article{" + key + ","]
    if rec.authors:
        auths = " and ".join([f'{a.get("family","")}, {a.get("given","")}' for a in rec.authors if a.get("family") or a.get("given")])
        lines.append(f"  author = {{{auths}}},")
    if rec.title:
        lines.append(f"  title = {{{rec.title}}},")
    if rec.journal:
        lines.append(f"  journal = {{{rec.journal}}},")
    if rec.year:
        lines.append(f"  year = {{{rec.year}}},")
    if rec.doi:
        lines.append(f"  doi = {{{rec.doi}}},")
    if rec.url:
        lines.append(f"  url = {{{rec.url}}},")
    lines.append("}\n")
    return "\n".join(lines)

def write_outputs(outdir: Path, records: List[RagRecord], emit_bib=False, emit_csv=False):
    ensure_dir(outdir)
    # JSONL
    jl = outdir / "references.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    # BibTeX
    if emit_bib:
        bibpath = outdir / "references.bib"
        with bibpath.open("w", encoding="utf-8") as bf:
            for r in records:
                bf.write(to_bibtex(r))
    # CSV (compact)
    if emit_csv:
        csvpath = outdir / "references.csv"
        with csvpath.open("w", encoding="utf-8", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["doi","title","year","journal","first_author","categories","impact","relevance","cited_by"])
            for r in records:
                first_author = (r.authors[0]["name"] if r.authors else "")
                w.writerow([r.doi, r.title, r.year, r.journal, first_author, ";".join(r.categories), r.impact, r.relevance, r.cited_by])

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="KEC_BIOMAT — Crossref/OpenAlex enrichment to RAG JSONL")
    ap.add_argument("--input", "-i", required=True, help="Path to seed file (.txt/.csv/.json/.jsonl)")
    ap.add_argument("--outdir", "-o", default="build", help="Output dir (default: build)")
    ap.add_argument("--user-agent", "-u", default=DEFAULT_USER_AGENT, help="Polite User-Agent for APIs")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests to be polite")
    ap.add_argument("--emit-bibtex", action="store_true", help="Emit BibTeX file")
    ap.add_argument("--emit-csv", action="store_true", help="Emit compact CSV")
    ap.add_argument("--qmap", help="Optional CSV with journal→quartile mapping (columns: journal,quartile)")
    ap.add_argument("--tag", help="Default category tag (A/B/C/D) for entries missing one")
    ap.add_argument("--relevance", type=int, help="Default relevance (1–5) for entries missing one")
    args = ap.parse_args()

    seed_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir / "logs")
    ensure_dir(outdir / CACHE_DIR)

    qmap = load_quartiles(Path(args.qmap)) if args.qmap else {}

    # Load seeds
    seeds = read_input(seed_path)
    if not seeds:
        print("No seed items found.", file=sys.stderr)
        sys.exit(1)

    records: List[RagRecord] = []
    for s in tqdm(seeds, desc="Enriching", unit="ref"):
        doi = clean_doi(s.doi) if s.doi else None

        cross = fetch_crossref(doi, args.user_agent, outdir / CACHE_DIR) if doi else {}
        oa = fetch_openalex(doi, args.user_agent, outdir / CACHE_DIR) if doi else {}

        c_norm = norm_crossref(cross) if cross else {}
        o_norm = norm_openalex(oa) if oa else {}
        merged = merge_priority(c_norm, o_norm) if (c_norm or o_norm) else {}

        # Fall back to seed info
        title = merged.get("title") or s.title
        abstract = merged.get("abstract")
        year = merged.get("year") or s.year
        journal = merged.get("journal") or s.journal
        url = merged.get("url")
        doi_final = merged.get("doi") or doi
        authors = merged.get("authors") or [{"name": a} for a in s.authors] or []
        issn = merged.get("issn") or []
        keywords = merged.get("keywords") or []
        cited_by = merged.get("cited_by")
        references_count = merged.get("references_count")
        publisher = merged.get("publisher")

        # Categories / impact / relevance
        categories = []
        if s.category:
            categories = [s.category]
        elif args.tag:
            categories = [args.tag]
        impact = s.impact
        if not impact and journal and qmap:
            impact = qmap.get(journal.strip().lower())
        relevance = s.relevance if s.relevance is not None else args.relevance

        # Build ID
        basis = doi_final or (title or "untitled") + (str(year) if year else "")
        rec_id = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

        # Source payloads
        source = {}
        if cross:
            source["crossref"] = cross.get("message", cross)
        if oa:
            source["openalex"] = oa.get("results", oa)

        record = RagRecord(
            id=rec_id,
            doi=doi_final,
            url=url,
            title=title,
            abstract=abstract,
            year=year,
            journal=journal,
            authors=authors,
            categories=categories,
            impact=impact,
            relevance=relevance,
            publisher=publisher,
            issn=issn,
            keywords=keywords,
            cited_by=cited_by,
            references_count=references_count,
            source=source,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        records.append(record)

        # polite sleep to respect APIs
        time.sleep(max(0.0, args.sleep))

    write_outputs(outdir, records, emit_bib=args.emit_bibtex, emit_csv=args.emit_csv)

    print(f"[ok] Wrote {len(records)} records → {outdir/'references.jsonl'}")
    if args.emit_bibtex:
        print(f"[ok] BibTeX → {outdir/'references.bib'}")
    if args.emit_csv:
        print(f"[ok] CSV    → {outdir/'references.csv'}")

if __name__ == "__main__":
    main()
