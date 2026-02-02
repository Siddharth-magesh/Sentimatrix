---
title: ScraperAPI
description: Integration with ScraperAPI proxy service
---

# ScraperAPI

ScraperAPI provides rotating proxies, CAPTCHA solving, and browser rendering for reliable web scraping.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="scraperapi",
        # api_key from SCRAPERAPI_API_KEY env var
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export SCRAPERAPI_API_KEY="your-api-key"
```

## Configuration

```python
ScraperConfig(
    provider="scraperapi",
    api_key="your-key",          # Or env var
    timeout=60,
    # ScraperAPI-specific options passed via kwargs
)
```

## Features

- **Rotating Proxies**: Automatic IP rotation
- **CAPTCHA Handling**: Automatic solving
- **JavaScript Rendering**: Browser rendering
- **Geotargeting**: Country-specific IPs
- **Retries**: Automatic retry logic

## API Options

```python
reviews = await sm.scrape_reviews(
    url,
    platform="amazon",
    # ScraperAPI options
    render=True,           # JavaScript rendering
    country_code="us",     # Target country
    premium=True,          # Premium proxies
    session_number=123,    # Sticky session
)
```

## Pricing

| Plan | Credits/Month | Price |
|------|---------------|-------|
| Hobby | 5,000 | $29 |
| Startup | 100,000 | $99 |
| Business | 500,000 | $299 |
| Enterprise | Custom | Custom |

## Best For

- Amazon (anti-bot protection)
- E-commerce sites
- High-volume scraping
- Sites with CAPTCHAs

## Example: Amazon at Scale

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="scraperapi")
)

async with Sentimatrix(config) as sm:
    # ScraperAPI handles proxies, CAPTCHAs, etc.
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon",
        max_reviews=500
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [Bright Data](brightdata.md)
- [Oxylabs](oxylabs.md)
