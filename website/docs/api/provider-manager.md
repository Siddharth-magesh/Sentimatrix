---
title: Provider Manager
description: LLM provider manager reference
---

# Provider Manager

Manages LLM providers with health monitoring, fallback, and load balancing.

## Features

- **Health Monitoring**: Track provider availability
- **Rate Limit Handling**: Automatic backoff
- **Load Balancing**: Distribute requests
- **Fallback Chain**: Switch on failures
- **Lazy Loading**: Initialize on demand

## Usage

```python
# Automatic management
async with Sentimatrix(config) as sm:
    # Manager handles provider selection
    result = await sm.summarize_reviews(reviews)
```

## Fallback Configuration

```python
from sentimatrix.config import FallbackConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="groq"),
    fallback=FallbackConfig(
        enabled=True,
        providers=["groq", "openai", "anthropic"],
        max_attempts=3
    )
)
```

## Behavior

1. **Primary Attempt**: Use configured provider
2. **On Failure**: Check health, try fallback
3. **Rate Limited**: Wait and retry
4. **All Failed**: Raise exception

## Health Checks

Provider health is monitored automatically:

- Connection status
- Response latency
- Error rates
- Rate limit status

## Related

- [API Overview](index.md)
- [Base Provider](base-provider.md)
- [Providers Guide](../providers/index.md)
