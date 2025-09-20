"""
Sistema H2 - Cache e Performance

Módulo de inicialização do sistema de cache multi-camadas.
"""

from .manager import CacheManager, get_cache_manager
from .memory_cache import MemoryCache
from .metrics import CacheMetrics, get_cache_metrics
from .redis_cache import RedisCache
from .strategies import (
    InvalidationStrategy,
    LFUStrategy,
    LRUStrategy,
    TagStrategy,
    TTLStrategy,
    get_invalidation_manager,
)

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "RedisCache",
    "MemoryCache",
    "CacheMetrics",
    "get_cache_metrics",
    "InvalidationStrategy",
    "TTLStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "TagStrategy",
    "get_invalidation_manager",
]

# Compat exports from cache_core (legacy API used by routers)
# Temporarily exposing imports directly to debug chained import errors
from .cache_core import (
    cache_manager,
    clear_all_cache,
    get_cache_stats,
    invalidate_cache_tags,
)
__all__ += [
    "cache_manager",
    "get_cache_stats",
    "invalidate_cache_tags",
    "clear_all_cache",
]
