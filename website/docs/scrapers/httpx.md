---
title: HTTPX Scraper
description: Fast async HTTP client for API-based scraping
---

# HTTPX Scraper

HTTPX provides a modern, fast async HTTP client for scraping platforms with APIs or simple HTML pages.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="httpx",
        timeout=30
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="steam")
```

## Configuration

```python
ScraperConfig(
    provider="httpx",
    timeout=30,                    # Request timeout
    user_agent="custom-agent",     # Custom User-Agent
    rate_limit=RateLimitConfig(
        requests_per_second=2.0,
        concurrent_requests=10
    ),
    retry=RetryConfig(
        max_retries=3,
        initial_delay=1.0
    )
)
```

## Features

- **Async Native**: Built for asyncio
- **HTTP/2 Support**: Modern protocol support
- **Connection Pooling**: Efficient connection reuse
- **Automatic Retries**: Built-in retry logic
- **Lightweight**: No browser overhead

## When to Use HTTPX

| Scenario | HTTPX | Playwright |
|----------|-------|------------|
| API endpoints | Best | Overkill |
| Static HTML | Good | Good |
| JavaScript pages | No | Required |
| High volume | Best | Slower |
| Resource usage | Low | High |

## Best For

- **Steam**: Uses API endpoints
- **Reddit**: API-based with OAuth
- **IMDB**: Static HTML pages
- **APIs**: Any REST API

## Example: High-Volume Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="httpx",
        rate_limit=RateLimitConfig(
            requests_per_second=5.0,
            concurrent_requests=20
        )
    )
)

async with Sentimatrix(config) as sm:
    # Fast parallel scraping
    tasks = [
        sm.scrape_reviews(url, platform="steam")
        for url in game_urls
    ]
    results = await asyncio.gather(*tasks)
```

## Headers Configuration

```python
# Custom headers for specific platforms
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="httpx",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
    )
)
```

## Related

- [Scraper Overview](index.md)
- [Playwright](playwright.md)
- [Steam Scraper](steam.md)
