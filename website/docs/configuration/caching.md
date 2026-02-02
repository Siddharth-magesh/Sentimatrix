---
title: Caching Configuration
description: Configure cache backends for performance
---

# Caching Configuration

Caching improves performance by storing analysis results.

## Memory Cache (Default)

```python
CacheConfig(
    enabled=True,
    backend="memory",
    max_size=1000,
    ttl=3600  # 1 hour
)
```

```yaml
cache:
  enabled: true
  backend: memory
  max_size: 1000
  ttl: 3600
```

## Redis Cache

For distributed deployments:

```python
CacheConfig(
    enabled=True,
    backend="redis",
    redis_url="redis://localhost:6379",
    ttl=3600,
    namespace="sentimatrix"
)
```

```yaml
cache:
  enabled: true
  backend: redis
  redis_url: redis://localhost:6379
  ttl: 3600
```

## SQLite Cache

For persistent local caching:

```python
CacheConfig(
    enabled=True,
    backend="sqlite",
    sqlite_path=".cache/sentimatrix.db",
    ttl=86400  # 24 hours
)
```

## Cache Options

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable caching | `True` |
| `backend` | Backend type | `"memory"` |
| `ttl` | Time-to-live (seconds) | `3600` |
| `max_size` | Max entries (memory) | `1000` |
| `compression` | Compress values | `False` |
| `namespace` | Key prefix | `"sentimatrix"` |

## When to Use Each

| Backend | Use Case |
|---------|----------|
| Memory | Single process, development |
| Redis | Distributed, production |
| SQLite | Local persistence |

## Related

- [Configuration Overview](index.md)
- [Performance Tips](../features/sentiment/index.md#performance-tips)
