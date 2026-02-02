---
title: Scraper Manager
description: Scraper manager reference
---

# Scraper Manager

Manages platform scrapers with rate limiting, retry, and proxy support.

## Features

- **Platform Detection**: Auto-detect platform from URL
- **Rate Limiting**: Per-platform rate control
- **Retry Logic**: Automatic retry on failures
- **Proxy Support**: Rotating proxy integration
- **Anti-Detection**: Stealth measures

## Usage

```python
async with Sentimatrix(config) as sm:
    # Manager auto-selects appropriate scraper
    reviews = await sm.scrape_reviews(
        url="https://amazon.com/product/...",
        platform="amazon",  # Optional, auto-detected
        max_reviews=100
    )
```

## Registered Scrapers

| Platform | Scraper |
|----------|---------|
| amazon | AmazonScraper |
| steam | SteamScraper |
| youtube | YouTubeScraper |
| reddit | RedditScraper |
| imdb | IMDBScraper |
| yelp | YelpScraper |
| trustpilot | TrustpilotScraper |
| google_reviews | GoogleReviewsScraper |

## Rate Limiting

```python
ScraperConfig(
    rate_limit=RateLimitConfig(
        requests_per_second=2.0,
        concurrent_requests=5
    )
)
```

## Proxy Integration

```python
ScraperConfig(
    proxy=ProxyConfig(
        enabled=True,
        provider="brightdata",
        rotation=True
    )
)
```

## Related

- [API Overview](index.md)
- [Base Scraper](base-scraper.md)
- [Scrapers Guide](../scrapers/index.md)
