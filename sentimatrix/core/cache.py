"""
Sentimatrix Cache Module

Provides caching functionality with multiple backends:
- Memory cache (default)
- Redis cache (distributed)
- SQLite cache (persistent)

Example:
    >>> cache = CacheManager(CacheConfig(backend="memory"))
    >>> await cache.set("key", "value", ttl=3600)
    >>> value = await cache.get("key")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from sentimatrix.core.config import CacheBackend, CacheConfig
from sentimatrix.core.exceptions import (
    CacheConnectionError,
    CacheReadError,
    CacheSerializationError,
    CacheWriteError,
)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Represents a cached entry with metadata."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def ttl_remaining(self) -> Optional[float]:
        """Get remaining TTL in seconds."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
        }


class BaseCacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key. Returns True if key existed."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache. Returns number of entries cleared."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close cache connection/cleanup resources."""
        pass


class MemoryCache(BaseCacheBackend):
    """
    In-memory cache with LRU eviction and TTL support.

    Thread-safe using asyncio locks.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600,
        compression: bool = False,
    ) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None for no expiry)
            compression: Enable value compression
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._compression = compression
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key, returns None if not found or expired."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1

            value = entry.value
            if self._compression and isinstance(value, bytes):
                try:
                    value = pickle.loads(zlib.decompress(value))
                except Exception as e:
                    raise CacheSerializationError(
                        "memory", "decompress", str(e)
                    ) from e

            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        async with self._lock:
            # Compress if enabled
            stored_value = value
            if self._compression:
                try:
                    stored_value = zlib.compress(pickle.dumps(value))
                except Exception as e:
                    raise CacheSerializationError(
                        "memory", "compress", str(e)
                    ) from e

            # Calculate expiry time
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = time.time() + effective_ttl if effective_ttl else None

            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=stored_value,
                expires_at=expires_at,
            )

            # Evict if at capacity and adding new key
            if key not in self._cache and len(self._cache) >= self._max_size:
                await self._evict_lru()

            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._stats.sets += 1
            self._stats.total_entries = len(self._cache)

    async def delete(self, key: str) -> bool:
        """Delete key. Returns True if key existed."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                self._stats.total_entries = len(self._cache)
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                self._stats.evictions += 1
                return False
            return True

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache, optionally filtered by namespace prefix."""
        async with self._lock:
            if namespace is None:
                count = len(self._cache)
                self._cache.clear()
            else:
                prefix = f"{namespace}:"
                keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
                count = len(keys_to_delete)
                for key in keys_to_delete:
                    del self._cache[key]

            self._stats.total_entries = len(self._cache)
            return count

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self._lock:
            self._stats.total_entries = len(self._cache)

            # Calculate approximate size
            total_size = 0
            for entry in self._cache.values():
                if isinstance(entry.value, bytes):
                    total_size += len(entry.value)
                else:
                    try:
                        total_size += len(pickle.dumps(entry.value))
                    except Exception:
                        pass

            self._stats.total_size_bytes = total_size
            return self._stats

    async def close(self) -> None:
        """Close cache (no-op for memory cache)."""
        pass

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # First try to evict expired entries
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys[:max(1, len(expired_keys) // 2)]:
                del self._cache[key]
                self._stats.evictions += 1

            # If still at capacity, evict LRU
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values by keys."""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(
        self, items: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Set multiple key-value pairs."""
        for key, value in items.items():
            await self.set(key, value, ttl)

    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys. Returns count of deleted keys."""
        deleted = 0
        for key in keys:
            if await self.delete(key):
                deleted += 1
        return deleted


class RedisCache(BaseCacheBackend):
    """
    Redis-based distributed cache with connection pooling.

    Features:
    - Distributed caching across multiple processes/machines
    - Connection pooling for efficient resource usage
    - Automatic serialization with pickle
    - Compression support for large values
    - TTL support via Redis EXPIRE
    - Atomic operations

    Example:
        >>> cache = RedisCache(
        ...     redis_url="redis://localhost:6379/0",
        ...     default_ttl=3600,
        ...     compression=True,
        ... )
        >>> await cache.initialize()
        >>> await cache.set("key", {"data": "value"})
        >>> value = await cache.get("key")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: Optional[int] = 3600,
        compression: bool = False,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (redis://host:port/db)
            default_ttl: Default TTL in seconds (None for no expiry)
            compression: Enable value compression
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
        """
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._compression = compression
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout
        self._retry_on_timeout = retry_on_timeout
        self._pool: Any = None
        self._client: Any = None
        self._stats = CacheStats()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self._initialized:
            return

        try:
            import redis.asyncio as redis_async
        except ImportError:
            raise ImportError(
                "redis package is required for Redis cache backend. "
                "Install it with: pip install redis"
            )

        try:
            # Create connection pool
            self._pool = redis_async.ConnectionPool.from_url(
                self._redis_url,
                max_connections=self._max_connections,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                retry_on_timeout=self._retry_on_timeout,
                decode_responses=False,  # We handle binary data
            )
            self._client = redis_async.Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            self._initialized = True
        except Exception as e:
            raise CacheConnectionError("redis", self._redis_url, str(e)) from e

    async def _ensure_initialized(self) -> None:
        """Ensure Redis is initialized."""
        if not self._initialized:
            await self.initialize()

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        try:
            data = pickle.dumps(value)
            if self._compression:
                data = zlib.compress(data)
            return data
        except Exception as e:
            raise CacheSerializationError("redis", "serialize", str(e)) from e

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        try:
            if self._compression:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            raise CacheSerializationError("redis", "deserialize", str(e)) from e

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        await self._ensure_initialized()

        try:
            data = await self._client.get(key)

            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            return self._deserialize(data)
        except CacheSerializationError:
            raise
        except Exception as e:
            raise CacheReadError("redis", key, str(e)) from e

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set key-value pair with optional TTL."""
        await self._ensure_initialized()

        try:
            data = self._serialize(value)
            effective_ttl = ttl if ttl is not None else self._default_ttl

            if effective_ttl:
                await self._client.setex(key, effective_ttl, data)
            else:
                await self._client.set(key, data)

            self._stats.sets += 1
        except CacheSerializationError:
            raise
        except Exception as e:
            raise CacheWriteError("redis", key, str(e)) from e

    async def delete(self, key: str) -> bool:
        """Delete key. Returns True if key existed."""
        await self._ensure_initialized()

        try:
            deleted = await self._client.delete(key)
            if deleted:
                self._stats.deletes += 1
            return deleted > 0
        except Exception as e:
            raise CacheWriteError("redis", key, str(e)) from e

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        await self._ensure_initialized()

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            raise CacheReadError("redis", key, str(e)) from e

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache, optionally filtered by namespace prefix."""
        await self._ensure_initialized()

        try:
            if namespace is None:
                # Clear all keys (dangerous in production!)
                await self._client.flushdb()
                return -1  # Unknown count after flush
            else:
                # Delete keys matching pattern
                pattern = f"{namespace}:*"
                cursor = 0
                count = 0

                while True:
                    cursor, keys = await self._client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        deleted = await self._client.delete(*keys)
                        count += deleted
                    if cursor == 0:
                        break

                return count
        except Exception as e:
            raise CacheWriteError("redis", namespace or "*", str(e)) from e

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        await self._ensure_initialized()

        try:
            info = await self._client.info("memory")
            db_size = await self._client.dbsize()

            self._stats.total_entries = db_size
            self._stats.total_size_bytes = info.get("used_memory", 0)

            return self._stats
        except Exception:
            return self._stats

    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        self._initialized = False

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values by keys."""
        await self._ensure_initialized()

        if not keys:
            return {}

        try:
            values = await self._client.mget(keys)
            results = {}

            for key, data in zip(keys, values):
                if data is not None:
                    results[key] = self._deserialize(data)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1

            return results
        except CacheSerializationError:
            raise
        except Exception as e:
            raise CacheReadError("redis", ",".join(keys[:5]), str(e)) from e

    async def set_many(
        self, items: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Set multiple key-value pairs."""
        await self._ensure_initialized()

        if not items:
            return

        try:
            effective_ttl = ttl if ttl is not None else self._default_ttl

            # Use pipeline for efficiency
            pipe = self._client.pipeline()
            for key, value in items.items():
                data = self._serialize(value)
                if effective_ttl:
                    pipe.setex(key, effective_ttl, data)
                else:
                    pipe.set(key, data)

            await pipe.execute()
            self._stats.sets += len(items)
        except CacheSerializationError:
            raise
        except Exception as e:
            raise CacheWriteError("redis", ",".join(list(items.keys())[:5]), str(e)) from e

    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple keys. Returns count of deleted keys."""
        await self._ensure_initialized()

        if not keys:
            return 0

        try:
            deleted = await self._client.delete(*keys)
            self._stats.deletes += deleted
            return deleted
        except Exception as e:
            raise CacheWriteError("redis", ",".join(keys[:5]), str(e)) from e

    async def ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a key."""
        await self._ensure_initialized()

        try:
            ttl_value = await self._client.ttl(key)
            if ttl_value < 0:
                return None  # Key doesn't exist or no TTL
            return ttl_value
        except Exception as e:
            raise CacheReadError("redis", key, str(e)) from e

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        await self._ensure_initialized()

        try:
            return await self._client.expire(key, ttl)
        except Exception as e:
            raise CacheWriteError("redis", key, str(e)) from e

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment key value atomically."""
        await self._ensure_initialized()

        try:
            return await self._client.incrby(key, amount)
        except Exception as e:
            raise CacheWriteError("redis", key, str(e)) from e

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            await self._ensure_initialized()
            await self._client.ping()
            return True
        except Exception:
            return False


class CacheManager:
    """
    High-level cache manager with namespace support and key hashing.

    Provides a unified interface for all cache backends.
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._backend: Optional[BaseCacheBackend] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize cache backend based on configuration."""
        if self._initialized:
            return

        if not self._config.enabled:
            self._backend = None
            self._initialized = True
            return

        if self._config.backend == CacheBackend.MEMORY:
            self._backend = MemoryCache(
                max_size=self._config.max_size,
                default_ttl=self._config.ttl if self._config.ttl > 0 else None,
                compression=self._config.compression,
            )
        elif self._config.backend == CacheBackend.REDIS:
            self._backend = RedisCache(
                redis_url=self._config.redis_url or "redis://localhost:6379/0",
                default_ttl=self._config.ttl if self._config.ttl > 0 else None,
                compression=self._config.compression,
            )
            await self._backend.initialize()
        elif self._config.backend == CacheBackend.SQLITE:
            # SQLite backend will be implemented later
            raise NotImplementedError("SQLite backend not yet implemented")
        else:
            raise ValueError(f"Unknown cache backend: {self._config.backend}")

        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure cache is initialized."""
        if not self._initialized:
            await self.initialize()

    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self._config.namespace}:{key}"

    @staticmethod
    def hash_key(key: str) -> str:
        """
        Create a hash of the key for use as cache key.

        Useful for long or complex keys.
        """
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value by key.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        await self._ensure_initialized()

        if self._backend is None:
            return default

        try:
            full_key = self._make_key(key)
            value = await self._backend.get(full_key)
            return value if value is not None else default
        except Exception as e:
            raise CacheReadError("memory", key, str(e)) from e

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:
        """
        Set key-value pair.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default)
        """
        await self._ensure_initialized()

        if self._backend is None:
            return

        try:
            full_key = self._make_key(key)
            await self._backend.set(full_key, value, ttl)
        except Exception as e:
            raise CacheWriteError("memory", key, str(e)) from e

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        await self._ensure_initialized()

        if self._backend is None:
            return False

        full_key = self._make_key(key)
        return await self._backend.delete(full_key)

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        await self._ensure_initialized()

        if self._backend is None:
            return False

        full_key = self._make_key(key)
        return await self._backend.exists(full_key)

    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            namespace: Optional namespace prefix to filter (uses configured namespace if None)

        Returns:
            Number of entries cleared
        """
        await self._ensure_initialized()

        if self._backend is None:
            return 0

        ns = namespace or self._config.namespace
        return await self._backend.clear(ns)

    async def get_or_set(
        self,
        key: str,
        default_factory: Any,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value or set if not exists.

        Args:
            key: Cache key
            default_factory: Callable that returns value to cache if not found
            ttl: Time-to-live in seconds

        Returns:
            Cached or newly set value
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Generate new value
        if callable(default_factory):
            if asyncio.iscoroutinefunction(default_factory):
                value = await default_factory()
            else:
                value = default_factory()
        else:
            value = default_factory

        await self.set(key, value, ttl)
        return value

    async def get_stats(self) -> Optional[CacheStats]:
        """
        Get cache statistics.

        Returns:
            CacheStats object or None if cache is disabled
        """
        await self._ensure_initialized()

        if self._backend is None:
            return None

        return await self._backend.get_stats()

    async def close(self) -> None:
        """Close cache and cleanup resources."""
        if self._backend is not None:
            await self._backend.close()
        self._initialized = False

    @property
    def enabled(self) -> bool:
        """Check if cache is enabled."""
        return self._config.enabled

    @property
    def backend_type(self) -> CacheBackend:
        """Get cache backend type."""
        return self._config.backend

    async def __aenter__(self) -> "CacheManager":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


def cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a cache key from arguments.

    Useful for function memoization.

    Example:
        >>> key = cache_key("analyze", url="https://example.com", limit=100)
        >>> # Returns deterministic hash-based key
    """
    # Serialize args and kwargs to string
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)

    # Hash for consistent length
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def cached(
    ttl: Optional[int] = 3600,
    key_prefix: str = "",
    cache_none: bool = False,
):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        cache_none: Whether to cache None results

    Example:
        >>> @cached(ttl=3600, key_prefix="sentiment")
        ... async def analyze_sentiment(text: str) -> dict:
        ...     # Expensive operation
        ...     return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get or create cache manager
            cache_manager = getattr(wrapper, "_cache_manager", None)
            if cache_manager is None:
                wrapper._cache_manager = CacheManager()
                await wrapper._cache_manager.initialize()
                cache_manager = wrapper._cache_manager

            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_value = await cache_manager.get(key)
            if cached_value is not None:
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None or cache_none:
                await cache_manager.set(key, result, ttl)

            return result

        return wrapper

    return decorator
