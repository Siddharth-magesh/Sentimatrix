---
title: ScrapingAnt
description: Integration with ScrapingAnt web scraping API
---

# ScrapingAnt

ScrapingAnt provides affordable web scraping with rotating proxies and headless browser support.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="scrapingant",
        # api_key from SCRAPINGANT_API_KEY env var
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export SCRAPINGANT_API_KEY="your-api-key"
```

## Configuration

```python
ScraperConfig(
    provider="scrapingant",
    api_key="your-key",          # Or env var
    timeout=60
)
```

## Features

- **Affordable**: Competitive pricing
- **Browser Rendering**: Headless Chrome
- **Proxy Rotation**: Automatic
- **CAPTCHA Handling**: Auto-solving
- **Simple API**: Easy integration

## API Options

```python
reviews = await sm.scrape_reviews(
    url,
    platform="amazon",
    # ScrapingAnt options
    browser=True,          # Enable browser
    proxy_country="us",    # Proxy location
    wait_for_selector=".reviews",  # Wait for element
)
```

## Pricing

| Plan | Credits | Price |
|------|---------|-------|
| Hobby | 10,000 | $19/mo |
| Startup | 100,000 | $49/mo |
| Business | 500,000 | $149/mo |
| Enterprise | 3M | $399/mo |

Very competitive pricing for the features offered.

## Best For

- Budget-conscious scraping
- Simple use cases
- Startups
- Testing and prototyping

## Example: Budget Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="scrapingant")
)

async with Sentimatrix(config) as sm:
    # Affordable scraping solution
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon",
        max_reviews=200
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [ScrapingBee](scrapingbee.md)
- [ScraperAPI](scraperapi.md)
