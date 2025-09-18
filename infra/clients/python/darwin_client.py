import os
import httpx
from typing import Optional, Dict, Any, List

class DarwinClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, namespace: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY")
        self.namespace = namespace or os.getenv("NAMESPACE", "KEC_BIOMAT_V1")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "X-Namespace": self.namespace,
        }

    def health(self) -> Dict[str, Any]:
        resp = httpx.get(f"{self.base_url}/healthz", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def query(self, question: str, top_k: int = 6) -> Dict[str, Any]:
        payload = {"query": question, "top_k": top_k}
        resp = httpx.post(f"{self.base_url}/rag", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def index(self, text: str, title: Optional[str] = None, url: Optional[str] = None, doi: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"content": text}
        if title:
            payload["title"] = title
        if url:
            payload["url"] = url
        if doi:
            payload["doi"] = doi
        if metadata:
            payload["metadata"] = metadata
        resp = httpx.post(f"{self.base_url}/rag/index", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def query_iterative(self, question: str, max_iters: int = 3, top_k: int = 6) -> Dict[str, Any]:
        payload = {"query": question, "max_iters": max_iters, "top_k": top_k}
        resp = httpx.post(f"{self.base_url}/rag-plus/query", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def tree_search(self, query: str, max_depth: int = 3, c_puct: float = 1.4, budget: int = 32) -> Dict[str, Any]:
        payload = {"query": query, "max_depth": max_depth, "c_puct": c_puct, "budget": budget}
        resp = httpx.post(f"{self.base_url}/tree-search/search", json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

# Usage example (for notebooks):
# from darwin_client import DarwinClient
# client = DarwinClient(base_url="http://localhost:8080", api_key="replace-with-secret")
# print(client.health())
