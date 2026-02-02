---
title: ScrapingBee
description: Integration with ScrapingBee web scraping API
---

# ScrapingBee

ScrapingBee provides a simple API for web scraping with automatic proxy rotation and JavaScript rendering.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="scrapingbee",
        # api_key from SCRAPINGBEE_API_KEY env var
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export SCRAPINGBEE_API_KEY="your-api-key"
```

## Configuration

```python
ScraperConfig(
    provider="scrapingbee",
    api_key="your-key",          # Or env var
    timeout=60
)
```

## Features

- **Proxy Rotation**: Automatic IP management
- **JavaScript Rendering**: Full browser
- **Screenshot API**: Page captures
- **Google Search API**: SERP scraping
- **Simple Pricing**: Pay per request

## API Options

```python
reviews = await sm.scrape_reviews(
    url,
    platform="amazon",
    # ScrapingBee options
    render_js=True,        # Enable JS rendering
    premium_proxy=True,    # Premium proxies
    country_code="us",     # Target country
    stealth_proxy=True,    # Extra stealth
)
```

## Pricing

| Plan | Credits | Price |
|------|---------|-------|
| Freelance | 1,000 | $49/mo |
| Startup | 10,000 | $99/mo |
| Business | 100,000 | $299/mo |

1 credit = 1 simple request
5 credits = 1 JS rendering request
10-25 credits = premium proxy request

## Best For

- Simple integration
- Occasional scraping
- JavaScript-heavy sites
- Screenshot needs

## Example: Simple Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="scrapingbee")
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon"
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [ScrapingAnt](scrapingant.md)
- [ScraperAPI](scraperapi.md)
