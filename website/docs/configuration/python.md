---
title: Python Configuration
description: Configure Sentimatrix programmatically
---

# Python Configuration

Programmatic configuration provides full control and type safety.

## Basic Usage

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze("Hello world!")
```

## Full Configuration

```python
from sentimatrix.config import (
    SentimatrixConfig,
    LLMConfig,
    ScraperConfig,
    ModelConfig,
    CacheConfig,
    LogConfig,
    RetryConfig,
    RateLimitConfig,
)

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        retry=RetryConfig(max_retries=3),
        rate_limit=RateLimitConfig(requests_per_second=2.0)
    ),
    scrapers=ScraperConfig(
        provider="playwright",
        headless=True
    ),
    models=ModelConfig(
        device="cuda"
    ),
    cache=CacheConfig(
        enabled=True,
        backend="memory"
    ),
    logging=LogConfig(
        level="INFO"
    ),
    debug=False
)
```

## Modifying Configuration

```python
# Create modified copy
new_config = config.with_overrides(
    llm={"provider": "openai", "model": "gpt-4o"}
)

# Export
config.save("my-config.yaml")
yaml_str = config.to_yaml()
config_dict = config.to_dict()
```

## Related

- [Configuration Overview](index.md)
- [YAML Configuration](yaml.md)
- [Environment Variables](environment.md)
