"""Discovery pipeline for DARWIN Mode A/B."""
from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, Iterable, List

import requests

import rag

try:  # Optional dependency for richer RSS parsing
    import feedparser
except ImportError:  # pragma: no cover - optional
    feedparser = None

try:  # Optional YAML parser
    import yaml
except ImportError:  # pragma: no cover - optional
    yaml = None

DEFAULT_FEEDS_YML = textwrap.dedent(
    """
    feeds:
      - name: arXiv AI
        url: https://export.arxiv.org/rss/cs.AI
        max: 15
      - name: arXiv LG
        url: https://export.arxiv.org/rss/cs.LG
        max: 15
      - name: Nature ML
        url: https://www.nature.com/subjects/machine-learning/rss
        max: 10
    """
).strip()

DISCOVERY_FROM_SECRET = os.getenv("DISCOVERY_FROM_SECRET", "false").lower() == "true"
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")


def _load_secret(name: str) -> str:
    from google.cloud import secretmanager  # Imported lazily to avoid dependency when unused

    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is required when DISCOVERY_FROM_SECRET=true")
    resource = f"projects/{GCP_PROJECT_ID}/secrets/{name}/versions/latest"
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=resource)
    return response.payload.data.decode("utf-8")


def _parse_yaml(text: str) -> Dict[str, Any]:
    if yaml is not None:  # pragma: no cover - requires PyYAML
        return yaml.safe_load(text) or {}
    return _fallback_parse_yaml(text)


def _fallback_parse_yaml(text: str) -> Dict[str, Any]:
    feeds: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("feeds"):
            continue
        if line.startswith("- "):
            if current:
                feeds.append(current)
            current = {}
            line = line[2:]
            if ":" in line:
                key, value = [part.strip() for part in line.split(":", 1)]
                current[key] = _coerce_value(value)
            continue
        if ":" in line and current is not None:
            key, value = [part.strip() for part in line.split(":", 1)]
            current[key] = _coerce_value(value)
    if current:
        feeds.append(current)
    return {"feeds": feeds}


def _coerce_value(value: str) -> Any:
    if value.isdigit():
        return int(value)
    return value


def load_feeds_configuration() -> List[Dict[str, Any]]:
    """Load RSS feed definitions from secret manager or environment/local defaults."""
    if DISCOVERY_FROM_SECRET:
        yaml_text = _load_secret("DISCOVERY_FEEDS_YML")
    else:
        yaml_text = os.getenv("DISCOVERY_FEEDS_YML", DEFAULT_FEEDS_YML)
    config = _parse_yaml(yaml_text)
    return [feed for feed in config.get("feeds", []) if feed.get("url")]


def _fetch_entries(url: str) -> List[Dict[str, Any]]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = response.text
    if feedparser is not None:  # pragma: no cover - requires feedparser
        parsed = feedparser.parse(text)
        entries = []
        for entry in parsed.entries:
            entries.append(
                {
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "summary": getattr(entry, "summary", ""),
                    "content": getattr(entry, "summary", ""),
                }
            )
        return entries
    return _fallback_rss_parse(text)


def _fallback_rss_parse(text: str) -> List[Dict[str, Any]]:
    from xml.etree import ElementTree as ET

    entries: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return entries
    for item in root.findall(".//item"):
        entries.append(
            {
                "title": (item.findtext("title") or ""),
                "link": (item.findtext("link") or ""),
                "summary": (item.findtext("description") or ""),
                "content": (item.findtext("description") or ""),
            }
        )
    return entries


def run_discovery(run_once: bool = False) -> Dict[str, Any]:
    """Ingest configured feeds into the vector backend and (optionally) BigQuery."""
    feeds = load_feeds_configuration()
    engine = rag.RAGEngine()
    ingested: List[Dict[str, Any]] = []
    for feed in feeds:
        url = feed.get("url")
        name = feed.get("name", url)
        limit = int(feed.get("max", 10))
        entries = _fetch_entries(url)[:limit]
        for entry in entries:
            content = entry.get("content") or entry.get("summary") or ""
            metadata = {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "abstract": entry.get("summary", ""),
                "source": name,
            }
            doc_id = engine.index(content, metadata)
            rag.persist_metadata(
                {
                    "doc_id": doc_id,
                    "title": metadata["title"],
                    "url": metadata["url"],
                    "abstract": metadata["abstract"],
                    "content": content,
                    "source": metadata["source"],
                }
            )
            ingested.append({"doc_id": doc_id, "feed": name, "title": metadata["title"]})
        if run_once:
            break
    return {
        "feeds": feeds, 
        "ingested": ingested,
        "added": len(ingested),
        "message": f"Successfully ingested {len(ingested)} documents from {len(feeds)} feeds"
    }
