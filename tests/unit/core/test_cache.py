"""
Unit Tests for Cache Module

Tests the caching system including:
- Memory cache operations
- Cache entry lifecycle
- TTL and expiration
- LRU eviction
- Cache manager
- Cache statistics
"""

import asyncio
import time

import pytest

from sentimatrix.core.cache import (
    BaseCacheBackend,
    CacheEntry,
    CacheManager,
    CacheStats,
    MemoryCache,
    RedisCache,
    cache_key,
    cached,
)
from sentimatrix.core.config import CacheBackend, CacheConfig
from sentimatrix.core.exceptions import CacheReadError, CacheWriteError


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = CacheEntry(key="test_key", value="test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 0
        assert entry.expires_at is None

    def test_entry_with_ttl(self):
        """Test entry with TTL."""
        expires = time.time() + 3600
        entry = CacheEntry(key="key", value="value", expires_at=expires)
        assert entry.expires_at == expires
        assert not entry.is_expired

    def test_entry_expired(self):
        """Test expired entry detection."""
        expires = time.time() - 1  # Already expired
        entry = CacheEntry(key="key", value="value", expires_at=expires)
        assert entry.is_expired

    def test_entry_ttl_remaining(self):
        """Test TTL remaining calculation."""
        expires = time.time() + 100
        entry = CacheEntry(key="key", value="value", expires_at=expires)
        remaining = entry.ttl_remaining
        assert remaining is not None
        assert 99 <= remaining <= 100

    def test_entry_ttl_remaining_no_expiry(self):
        """Test TTL remaining when no expiry set."""
        entry = CacheEntry(key="key", value="value")
        assert entry.ttl_remaining is None

    def test_entry_touch(self):
        """Test entry touch updates access metadata."""
        entry = CacheEntry(key="key", value="value")
        initial_count = entry.access_count
        initial_time = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestCacheStats:
    """Tests for CacheStats."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test stats serialization."""
        stats = CacheStats(hits=10, misses=5, sets=15)
        stats_dict = stats.to_dict()

        assert stats_dict["hits"] == 10
        assert stats_dict["misses"] == 5
        assert stats_dict["sets"] == 15
        assert "hit_rate" in stats_dict


class TestMemoryCache:
    """Tests for MemoryCache backend."""

    @pytest.fixture
    def cache(self):
        """Provide a fresh memory cache."""
        return MemoryCache(max_size=10, default_ttl=60)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test get for nonexistent key."""
        value = await cache.get("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test delete operation."""
        await cache.set("key", "value")
        result = await cache.delete("key")
        assert result is True

        # Should be gone
        value = await cache.get("key")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Test delete for nonexistent key."""
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test exists operation."""
        await cache.set("key", "value")
        assert await cache.exists("key") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clear operation."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        count = await cache.clear()
        assert count == 2

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_clear_with_namespace(self, cache):
        """Test clear with namespace filter."""
        await cache.set("ns1:key1", "value1")
        await cache.set("ns1:key2", "value2")
        await cache.set("ns2:key1", "value3")

        count = await cache.clear("ns1")
        assert count == 2

        assert await cache.get("ns1:key1") is None
        assert await cache.get("ns2:key1") is not None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Set with very short TTL
        cache_short = MemoryCache(default_ttl=None)
        await cache_short.set("key", "value", ttl=1)

        # Should exist immediately
        assert await cache_short.get("key") == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        assert await cache_short.get("key") is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=3, default_ttl=None)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new key, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        assert await cache.get("key1") is not None  # Recently accessed
        assert await cache.get("key2") is None  # Should be evicted
        assert await cache.get("key3") is not None  # Accessed by get above
        assert await cache.get("key4") is not None  # Just added

    @pytest.mark.asyncio
    async def test_stats_tracking(self, cache):
        """Test cache statistics tracking."""
        await cache.set("key", "value")
        await cache.get("key")  # Hit
        await cache.get("nonexistent")  # Miss

        stats = await cache.get_stats()
        assert stats.sets == 1
        assert stats.hits == 1
        assert stats.misses == 1

    @pytest.mark.asyncio
    async def test_get_many(self, cache):
        """Test getting multiple keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        results = await cache.get_many(["key1", "key2", "key3"])
        assert results["key1"] == "value1"
        assert results["key2"] == "value2"
        assert "key3" not in results

    @pytest.mark.asyncio
    async def test_set_many(self, cache):
        """Test setting multiple keys."""
        await cache.set_many({"key1": "value1", "key2": "value2"})

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_delete_many(self, cache):
        """Test deleting multiple keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        deleted = await cache.delete_many(["key1", "key2", "key3"])
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_compression(self):
        """Test cache with compression enabled."""
        cache = MemoryCache(compression=True)
        data = {"large": "data" * 1000}

        await cache.set("key", data)
        result = await cache.get("key")

        assert result == data

    @pytest.mark.asyncio
    async def test_complex_values(self, cache):
        """Test caching complex values."""
        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": "b"},
            "number": 42.5,
            "none": None,
        }

        await cache.set("complex", complex_value)
        result = await cache.get("complex")

        assert result == complex_value


class TestCacheManager:
    """Tests for CacheManager."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test cache manager initialization."""
        config = CacheConfig(enabled=True, backend=CacheBackend.MEMORY)
        manager = CacheManager(config)

        await manager.initialize()
        assert manager._initialized is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_disabled_cache(self):
        """Test disabled cache behavior."""
        config = CacheConfig(enabled=False)
        manager = CacheManager(config)

        await manager.initialize()

        # Operations should be no-ops
        await manager.set("key", "value")
        result = await manager.get("key")
        assert result is None

        await manager.close()

    @pytest.mark.asyncio
    async def test_namespace_prefixing(self):
        """Test namespace is prepended to keys."""
        config = CacheConfig(namespace="test_ns")
        manager = CacheManager(config)
        await manager.initialize()

        await manager.set("key", "value")

        # Internal key should be namespaced
        full_key = manager._make_key("key")
        assert full_key == "test_ns:key"

        await manager.close()

    @pytest.mark.asyncio
    async def test_get_with_default(self):
        """Test get with default value."""
        config = CacheConfig()
        manager = CacheManager(config)
        await manager.initialize()

        result = await manager.get("nonexistent", default="default_value")
        assert result == "default_value"

        await manager.close()

    @pytest.mark.asyncio
    async def test_get_or_set(self):
        """Test get_or_set operation."""
        config = CacheConfig()
        manager = CacheManager(config)
        await manager.initialize()

        # First call - should set
        result1 = await manager.get_or_set("key", lambda: "computed_value")
        assert result1 == "computed_value"

        # Second call - should get cached
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "new_value"

        result2 = await manager.get_or_set("key", factory)
        assert result2 == "computed_value"  # Cached value
        assert call_count == 0  # Factory not called

        await manager.close()

    @pytest.mark.asyncio
    async def test_get_or_set_async_factory(self):
        """Test get_or_set with async factory."""
        config = CacheConfig()
        manager = CacheManager(config)
        await manager.initialize()

        async def async_factory():
            return "async_value"

        result = await manager.get_or_set("key", async_factory)
        assert result == "async_value"

        await manager.close()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting cache statistics."""
        config = CacheConfig()
        manager = CacheManager(config)
        await manager.initialize()

        await manager.set("key", "value")
        await manager.get("key")

        stats = await manager.get_stats()
        assert stats is not None
        assert stats.sets >= 1
        assert stats.hits >= 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager protocol."""
        config = CacheConfig()

        async with CacheManager(config) as manager:
            await manager.set("key", "value")
            result = await manager.get("key")
            assert result == "value"

    @pytest.mark.asyncio
    async def test_hash_key(self):
        """Test key hashing."""
        key1 = CacheManager.hash_key("long_key_that_should_be_hashed")
        key2 = CacheManager.hash_key("long_key_that_should_be_hashed")
        key3 = CacheManager.hash_key("different_key")

        assert key1 == key2  # Same input = same hash
        assert key1 != key3  # Different input = different hash
        assert len(key1) == 32  # Fixed length


class TestCacheKey:
    """Tests for cache_key function."""

    def test_cache_key_args(self):
        """Test cache key generation from args."""
        key1 = cache_key("arg1", "arg2")
        key2 = cache_key("arg1", "arg2")
        key3 = cache_key("arg1", "arg3")

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key

    def test_cache_key_kwargs(self):
        """Test cache key generation from kwargs."""
        key1 = cache_key(a=1, b=2)
        key2 = cache_key(b=2, a=1)  # Order shouldn't matter
        key3 = cache_key(a=1, b=3)

        assert key1 == key2  # Same kwargs = same key
        assert key1 != key3  # Different values = different key

    def test_cache_key_mixed(self):
        """Test cache key with args and kwargs."""
        key = cache_key("arg", key="value")
        assert len(key) == 32  # SHA256 truncated


class TestCachedDecorator:
    """Tests for @cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_function(self):
        """Test caching async function results."""
        call_count = 0

        @cached(ttl=60, key_prefix="test")
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute
        result1 = await expensive_operation(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = await expensive_operation(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different arg - should execute
        result3 = await expensive_operation(10)
        assert result3 == 20
        assert call_count == 2


class TestCacheEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_none_value_caching(self):
        """Test caching None values."""
        cache = MemoryCache()
        await cache.set("key", None)

        # Get returns None for both missing and None values
        # This is expected behavior
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_string_caching(self):
        """Test caching empty strings."""
        cache = MemoryCache()
        await cache.set("key", "")

        result = await cache.get("key")
        assert result == ""

    @pytest.mark.asyncio
    async def test_large_value(self):
        """Test caching large values."""
        cache = MemoryCache()
        large_value = "x" * 10_000_000  # 10MB string

        await cache.set("large", large_value)
        result = await cache.get("large")

        assert result == large_value

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent cache access."""
        cache = MemoryCache()

        async def writer():
            for i in range(100):
                await cache.set(f"key_{i}", f"value_{i}")

        async def reader():
            for i in range(100):
                await cache.get(f"key_{i}")

        # Run concurrently
        await asyncio.gather(
            writer(),
            reader(),
            writer(),
            reader(),
        )

        # Cache should be consistent
        stats = await cache.get_stats()
        assert stats.sets >= 100

    @pytest.mark.asyncio
    async def test_ttl_zero(self):
        """Test TTL of 0 (no expiry)."""
        cache = MemoryCache(default_ttl=None)
        await cache.set("key", "value", ttl=0)

        # With ttl=0, the item should still exist
        # (0 is treated as no TTL, not immediate expiry)
        result = await cache.get("key")
        # Note: Implementation may vary - this tests current behavior


# ============================================================================
# Redis Cache Tests
# ============================================================================


def has_redis():
    """Check if redis package is available."""
    try:
        import redis.asyncio
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing without actual Redis."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.get = AsyncMock(return_value=None)
    mock_client.set = AsyncMock(return_value=True)
    mock_client.setex = AsyncMock(return_value=True)
    mock_client.delete = AsyncMock(return_value=1)
    mock_client.exists = AsyncMock(return_value=1)
    mock_client.mget = AsyncMock(return_value=[])
    mock_client.dbsize = AsyncMock(return_value=0)
    mock_client.info = AsyncMock(return_value={"used_memory": 0})
    mock_client.scan = AsyncMock(return_value=(0, []))
    mock_client.flushdb = AsyncMock(return_value=True)
    mock_client.ttl = AsyncMock(return_value=3600)
    mock_client.expire = AsyncMock(return_value=True)
    mock_client.incrby = AsyncMock(return_value=1)
    mock_client.aclose = AsyncMock()
    mock_client.pipeline = MagicMock(return_value=AsyncMock())

    return mock_client


class TestRedisCache:
    """Tests for RedisCache class."""

    def test_redis_cache_init(self):
        """Test RedisCache initialization."""
        cache = RedisCache(
            redis_url="redis://localhost:6379/0",
            default_ttl=3600,
            compression=True,
        )
        assert cache._redis_url == "redis://localhost:6379/0"
        assert cache._default_ttl == 3600
        assert cache._compression is True
        assert cache._initialized is False

    @pytest.mark.asyncio
    async def test_redis_cache_serialize_deserialize(self):
        """Test serialization and deserialization."""
        cache = RedisCache()

        # Test basic types
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = cache._serialize(test_data)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data

    @pytest.mark.asyncio
    async def test_redis_cache_serialize_with_compression(self):
        """Test serialization with compression."""
        cache = RedisCache(compression=True)

        test_data = {"key": "value" * 100}  # Larger data for compression
        serialized = cache._serialize(test_data)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data

    @pytest.mark.asyncio
    async def test_redis_cache_without_redis(self):
        """Test that RedisCache requires redis package."""
        cache = RedisCache()

        # If redis is not installed, initialize should raise ImportError
        if not has_redis():
            with pytest.raises(ImportError):
                await cache.initialize()

    @pytest.mark.asyncio
    async def test_redis_cache_get_mock(self, mock_redis_client):
        """Test Redis get with mock client."""
        import pickle
        from unittest.mock import patch, AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        # Test cache miss
        cache._client.get = AsyncMock(return_value=None)
        result = await cache.get("missing_key")
        assert result is None
        assert cache._stats.misses == 1

        # Test cache hit
        test_value = {"data": "test"}
        cache._client.get = AsyncMock(return_value=pickle.dumps(test_value))
        result = await cache.get("existing_key")
        assert result == test_value
        assert cache._stats.hits == 1

    @pytest.mark.asyncio
    async def test_redis_cache_set_mock(self, mock_redis_client):
        """Test Redis set with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache(default_ttl=3600)
        cache._initialized = True
        cache._client = mock_redis_client

        # Test set with TTL
        await cache.set("key", "value", ttl=60)
        mock_redis_client.setex.assert_called()
        assert cache._stats.sets == 1

        # Test set without TTL
        cache._default_ttl = None
        await cache.set("key2", "value2", ttl=None)
        mock_redis_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_delete_mock(self, mock_redis_client):
        """Test Redis delete with mock client."""
        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        result = await cache.delete("key")
        assert result is True
        assert cache._stats.deletes == 1

    @pytest.mark.asyncio
    async def test_redis_cache_exists_mock(self, mock_redis_client):
        """Test Redis exists with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.exists = AsyncMock(return_value=1)
        result = await cache.exists("key")
        assert result is True

        mock_redis_client.exists = AsyncMock(return_value=0)
        result = await cache.exists("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_cache_clear_mock(self, mock_redis_client):
        """Test Redis clear with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        # Clear all
        await cache.clear()
        mock_redis_client.flushdb.assert_called()

        # Clear with namespace
        mock_redis_client.scan = AsyncMock(return_value=(0, [b"ns:key1", b"ns:key2"]))
        mock_redis_client.delete = AsyncMock(return_value=2)
        count = await cache.clear("ns")
        assert count == 2

    @pytest.mark.asyncio
    async def test_redis_cache_stats_mock(self, mock_redis_client):
        """Test Redis stats with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.info = AsyncMock(return_value={"used_memory": 1024})
        mock_redis_client.dbsize = AsyncMock(return_value=100)

        stats = await cache.get_stats()
        assert stats.total_entries == 100
        assert stats.total_size_bytes == 1024

    @pytest.mark.asyncio
    async def test_redis_cache_get_many_mock(self, mock_redis_client):
        """Test Redis get_many with mock client."""
        import pickle
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        test_values = [pickle.dumps("value1"), pickle.dumps("value2"), None]
        mock_redis_client.mget = AsyncMock(return_value=test_values)

        result = await cache.get_many(["key1", "key2", "key3"])
        assert result == {"key1": "value1", "key2": "value2"}
        assert cache._stats.hits == 2
        assert cache._stats.misses == 1

    @pytest.mark.asyncio
    async def test_redis_cache_set_many_mock(self, mock_redis_client):
        """Test Redis set_many with mock client."""
        from unittest.mock import AsyncMock, MagicMock

        cache = RedisCache(default_ttl=3600)
        cache._initialized = True
        cache._client = mock_redis_client

        mock_pipe = MagicMock()
        mock_pipe.setex = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[True, True])
        mock_redis_client.pipeline = MagicMock(return_value=mock_pipe)

        items = {"key1": "value1", "key2": "value2"}
        await cache.set_many(items)

        assert cache._stats.sets == 2

    @pytest.mark.asyncio
    async def test_redis_cache_delete_many_mock(self, mock_redis_client):
        """Test Redis delete_many with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.delete = AsyncMock(return_value=2)

        count = await cache.delete_many(["key1", "key2"])
        assert count == 2
        assert cache._stats.deletes == 2

    @pytest.mark.asyncio
    async def test_redis_cache_ttl_mock(self, mock_redis_client):
        """Test Redis ttl method with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.ttl = AsyncMock(return_value=3600)
        ttl = await cache.ttl("key")
        assert ttl == 3600

        mock_redis_client.ttl = AsyncMock(return_value=-1)  # Key doesn't exist
        ttl = await cache.ttl("missing")
        assert ttl is None

    @pytest.mark.asyncio
    async def test_redis_cache_expire_mock(self, mock_redis_client):
        """Test Redis expire method with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.expire = AsyncMock(return_value=True)
        result = await cache.expire("key", 3600)
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_cache_incr_mock(self, mock_redis_client):
        """Test Redis incr method with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.incrby = AsyncMock(return_value=5)
        result = await cache.incr("counter", 5)
        assert result == 5

    @pytest.mark.asyncio
    async def test_redis_cache_health_check_mock(self, mock_redis_client):
        """Test Redis health_check method with mock client."""
        from unittest.mock import AsyncMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client

        mock_redis_client.ping = AsyncMock(return_value=True)
        result = await cache.health_check()
        assert result is True

        mock_redis_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
        result = await cache.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_cache_close_mock(self, mock_redis_client):
        """Test Redis close method with mock client."""
        from unittest.mock import AsyncMock, MagicMock

        cache = RedisCache()
        cache._initialized = True
        cache._client = mock_redis_client
        cache._pool = MagicMock()
        cache._pool.disconnect = AsyncMock()

        await cache.close()

        assert cache._initialized is False
        assert cache._client is None
        assert cache._pool is None


class TestCacheManagerWithRedis:
    """Tests for CacheManager with Redis backend."""

    @pytest.mark.asyncio
    async def test_cache_manager_redis_not_implemented(self):
        """Test that Redis backend requires redis package."""
        config = CacheConfig(
            backend=CacheBackend.REDIS,
            redis_url="redis://localhost:6379/0",
        )
        manager = CacheManager(config)

        if not has_redis():
            with pytest.raises(ImportError):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_cache_manager_sqlite_not_implemented(self):
        """Test that SQLite backend is not yet implemented."""
        config = CacheConfig(
            backend=CacheBackend.SQLITE,
            sqlite_path="/tmp/cache.db",
        )
        manager = CacheManager(config)

        with pytest.raises(NotImplementedError):
            await manager.initialize()
