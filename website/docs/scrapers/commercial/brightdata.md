---
title: Bright Data
description: Integration with Bright Data proxy network
---

# Bright Data

Bright Data (formerly Luminati) provides the largest proxy network with 72M+ residential IPs.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig, ProxyConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="brightdata",
        proxy=ProxyConfig(
            enabled=True,
            provider="brightdata",
            username="user",
            password="pass"
        )
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
export BRIGHTDATA_USERNAME="your-username"
export BRIGHTDATA_PASSWORD="your-password"
export BRIGHTDATA_ZONE="your-zone"
```

## Configuration

```python
ScraperConfig(
    provider="brightdata",
    proxy=ProxyConfig(
        enabled=True,
        provider="brightdata",
        country="us",           # Target country
        rotation=True,          # IP rotation
    )
)
```

## Features

- **72M+ IPs**: Largest proxy network
- **Residential IPs**: Real user IPs
- **Datacenter IPs**: Fast & cheap
- **Mobile IPs**: Mobile carriers
- **ISP Proxies**: Static residential

## Proxy Types

| Type | Use Case | Speed | Price |
|------|----------|-------|-------|
| Datacenter | APIs, basic sites | Fastest | Cheapest |
| Residential | Anti-bot sites | Fast | Medium |
| ISP | Sticky sessions | Fast | Higher |
| Mobile | Mobile-specific | Medium | Highest |

## Pricing

Pay per GB of data transferred:

| Type | Price/GB |
|------|----------|
| Datacenter | $0.50 |
| Residential | $8-15 |
| ISP | $12-20 |
| Mobile | $20-40 |

## Best For

- Geo-restricted content
- Heavy anti-bot sites
- Large-scale operations
- Sticky sessions needed

## Example: Global Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="brightdata",
        proxy=ProxyConfig(
            enabled=True,
            provider="brightdata",
            country="uk"  # Target UK
        )
    )
)

async with Sentimatrix(config) as sm:
    # Scrape UK Amazon
    reviews = await sm.scrape_reviews(
        "https://amazon.co.uk/product/...",
        platform="amazon"
    )
```

## Related

- [Commercial APIs Overview](index.md)
- [Oxylabs](oxylabs.md)
- [ScraperAPI](scraperapi.md)
