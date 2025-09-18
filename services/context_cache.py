"""Context cache analysis helpers (docs-aware stubs).

Implements basic prompt prefix extraction to mimic Vertex context cache usage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


def _split_prefix_suffix(prompt: str) -> Tuple[str, str]:
    # naive heuristic: first paragraph is prefix
    parts = prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return prompt, ""


@dataclass
class _ContextCache:
    max_cache_size: int = 128
    cache: Dict[str, str] = field(default_factory=dict)
    prefix_stats: Dict[str, int] = field(default_factory=dict)

    def get_cache_stats(self) -> Dict[str, Any]:
        total = sum(self.prefix_stats.values()) or 1
        hits = int(total * 0.5)
        return {
            "total_entries": len(self.cache),
            "hit_ratio": 0.5,
            "cache_utilization": len(self.cache) / max(1, self.max_cache_size),
            "total_hits": hits,
            "total_requests": total,
            "top_prefixes": [
                {"prefix": k[:40], "uses": v} for k, v in list(self.prefix_stats.items())[:5]
            ],
        }

    def export_cache_config(self) -> Dict[str, Any]:
        return {"entries": len(self.cache)}


_CACHE = _ContextCache()


def get_context_cache() -> _ContextCache:
    return _CACHE


def analyze_prompt_caching(prompt: str, model: str) -> Dict[str, Any]:
    prefix, suffix = _split_prefix_suffix(prompt)
    cacheable = len(prefix) > 0
    return {
        "cacheable": cacheable,
        "cache_available": prefix in _CACHE.cache,
        "prefix_length": len(prefix),
        "suffix_length": len(suffix),
        "recommendations": [
            {"type": "prefixing", "detail": "Keep static policy/methods in prefix"}
        ],
        "estimated_savings": {"input_tokens_discount": 0.75 if cacheable else 0.0},
    }


def prepare_cached_prompt(prompt: str, model: str) -> Dict[str, Any]:
    prefix, suffix = _split_prefix_suffix(prompt)
    if prefix and prefix not in _CACHE.cache:
        if len(_CACHE.cache) >= _CACHE.max_cache_size:
            _CACHE.cache.pop(next(iter(_CACHE.cache)))
        _CACHE.cache[prefix] = "ok"
    _CACHE.prefix_stats[prefix] = _CACHE.prefix_stats.get(prefix, 0) + 1
    return {
        "original_prompt": prompt,
        "cached_prefix": prefix,
        "dynamic_suffix": suffix,
        "cache_key": str(hash(prefix)) if prefix else None,
        "cache_available": prefix in _CACHE.cache,
        "model": model,
        "optimization": {"savings": 0.75 if prefix else 0.0},
    }

