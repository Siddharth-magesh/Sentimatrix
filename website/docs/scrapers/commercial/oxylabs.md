---
title: Oxylabs
description: Integration with Oxylabs proxy and scraping services
---

# Oxylabs

Oxylabs provides enterprise-grade proxy infrastructure and web scraping APIs.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig, ProxyConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="oxylabs",
        proxy=ProxyConfig(
            enabled=True,
            provider="oxylabs"
        )
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export OXYLABS_USERNAME="your-username"
export OXYLABS_PASSWORD="your-password"
```

## Configuration

```python
ScraperConfig(
    provider="oxylabs",
    proxy=ProxyConfig(
        enabled=True,
        provider="oxylabs",
        country="us",
        rotation=True
    )
)
```

## Products

### Web Scraper API

Pre-built scrapers for popular sites:

```python
# Amazon, Google, other e-commerce
reviews = await sm.scrape_reviews(
    url,
    platform="amazon",
    use_scraper_api=True  # Use Oxylabs Scraper API
)
```

### Proxy Products

| Product | IPs | Best For |
|---------|-----|----------|
| Residential | 100M+ | Anti-bot sites |
| Datacenter | 2M+ | High volume |
| Mobile | 20M+ | Mobile-specific |
| ISP | 500K+ | Sticky sessions |

## Pricing

| Product | Starting Price |
|---------|---------------|
| Residential | $10/GB |
| Datacenter | $1.20/GB |
| Web Scraper API | $2.70/1K requests |

## Best For

- Enterprise deployments
- E-commerce scraping
- High reliability needs
- SERP scraping

## Example: E-Commerce Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="oxylabs")
)

async with Sentimatrix(config) as sm:
    # Oxylabs handles anti-bot
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon",
        max_reviews=1000
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [Bright Data](brightdata.md)
- [Zyte](zyte.md)
