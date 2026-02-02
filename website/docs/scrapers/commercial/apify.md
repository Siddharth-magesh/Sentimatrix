---
title: Apify
description: Integration with Apify web scraping platform
---

# Apify

Apify provides pre-built scrapers (Actors) and infrastructure for web scraping at scale.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="apify",
        # api_key from APIFY_TOKEN env var
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export APIFY_TOKEN="your-api-token"
```

## Configuration

```python
ScraperConfig(
    provider="apify",
    api_key="your-token",        # Or env var
    timeout=120,                 # Actor run timeout
)
```

## Features

- **Pre-built Actors**: Ready-to-use scrapers
- **Actor Store**: 1000+ community scrapers
- **Proxy Pool**: Residential & datacenter
- **Storage**: Dataset & key-value storage
- **Scheduling**: Cron-based runs

## Available Actors

| Platform | Actor |
|----------|-------|
| Amazon | amazon-product-reviews |
| Google Reviews | google-maps-reviews |
| Yelp | yelp-scraper |
| Trustpilot | trustpilot-reviews |
| YouTube | youtube-comments |

## Pricing

| Plan | Compute Units | Price |
|------|---------------|-------|
| Free | $5/month | $0 |
| Personal | $49 | $49 |
| Team | $499 | $499 |
| Enterprise | Custom | Custom |

## Best For

- Complex scraping workflows
- Pre-built platform scrapers
- Scheduled scraping jobs
- Large-scale operations

## Example: Using Amazon Actor

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(provider="apify")
)

async with Sentimatrix(config) as sm:
    # Apify's Amazon actor handles everything
    reviews = await sm.scrape_reviews(
        "B08N5WRWNW",  # ASIN
        platform="amazon",
        max_reviews=1000
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [ScraperAPI](scraperapi.md)
- [Zyte](zyte.md)
