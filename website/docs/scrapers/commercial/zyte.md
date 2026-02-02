---
title: Zyte
description: Integration with Zyte (formerly Scrapinghub) platform
---

# Zyte

Zyte (formerly Scrapinghub) provides a complete web scraping platform with smart proxy management.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="zyte",
        # api_key from ZYTE_API_KEY env var
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export ZYTE_API_KEY="your-api-key"
```

## Configuration

```python
ScraperConfig(
    provider="zyte",
    api_key="your-key",          # Or env var
    timeout=60
)
```

## Products

### Zyte API

All-in-one web scraping API:

- Automatic proxy rotation
- Browser rendering
- CAPTCHA solving
- Anti-bot bypass

### Smart Proxy Manager

Intelligent proxy management:

- Automatic retries
- Ban detection
- Session management
- Geographic targeting

## Pricing

| Product | Price |
|---------|-------|
| Zyte API | $5/1K requests |
| Smart Proxy | Custom |
| Scrapy Cloud | $9+/month |

## Best For

- Scrapy users
- Complex scraping
- Anti-bot bypass
- Managed infrastructure

## Example: With Zyte API

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="zyte")
)

async with Sentimatrix(config) as sm:
    # Zyte handles complexity
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon",
        max_reviews=500
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [Apify](apify.md)
- [ScraperAPI](scraperapi.md)
