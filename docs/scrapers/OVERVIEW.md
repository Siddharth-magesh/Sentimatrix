# Sentimatrix V2 - Scraping Overview

## Architecture

Sentimatrix V2 provides a comprehensive scraping layer with multiple provider options:

```
┌─────────────────────────────────────────────────────────────────┐
│                      SCRAPER MANAGER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    LOCAL     │  │   BROWSER    │  │  COMMERCIAL  │          │
│  │   SCRAPERS   │  │   SCRAPERS   │  │     APIs     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│    - HTTPX           - Playwright       - ScraperAPI            │
│    - BeautifulSoup                      - Apify                 │
│                                         - Bright Data           │
│                                         - Oxylabs               │
│                                         - Zyte                  │
│                                         - ScrapingBee           │
│                                         - ScrapingAnt           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  PLATFORM SCRAPERS                       │    │
│  │  Amazon | Steam | YouTube | Reddit | IMDB | Yelp | ...  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scraper Categories

### 1. Core Scrapers

| Scraper | Type | Async | JS Support | Use Case |
|---------|------|-------|------------|----------|
| `HTTPXScraper` | HTTP Client | Yes | No | Static HTML pages |
| `PlaywrightScraper` | Browser | Yes | Yes | JS-rendered content |

### 2. Platform Scrapers (Implemented)

| Platform | API Type | Auth Required | Status |
|----------|----------|---------------|--------|
| Amazon | HTML + Playwright | No | Working |
| Steam | JSON API | No | Working |
| YouTube | Data API v3 | API Key | Working |
| Reddit | JSON API | Optional OAuth | Working |
| IMDB | HTML | No | Working |
| Yelp | HTML | No | Working |
| Trustpilot | HTML | No | Working |
| Google Reviews | HTML/API | API Key | Working |

### 3. Commercial API Scrapers (Implemented)

| Service | Proxy Pool | JS Render | CAPTCHA | Pricing |
|---------|------------|-----------|---------|---------|
| ScraperAPI | 40M+ | Yes | Yes | From $49/mo |
| Apify | N/A | Actors | Per-actor | Pay-per-use |
| Bright Data | 72M+ | Yes | Yes | From $500/mo |
| Oxylabs | 100M+ | Yes | Yes | Custom |
| Zyte | 50M+ | AI | Yes | From $450/mo |
| ScrapingBee | 1M+ | Yes | Yes | From $49/mo |
| ScrapingAnt | - | Yes | Basic | From $19/mo |

---

## Quick Start

### Steam Reviews (No Browser Required)

```python
import asyncio
from sentimatrix.providers.scrapers.platforms import SteamScraper, SteamConfig

async def main():
    config = SteamConfig(language="english", review_type="all")

    async with SteamScraper(config) as scraper:
        reviews = await scraper.scrape_reviews("730", limit=20)  # CS2

        for review in reviews[:5]:
            print(f"[{'Positive' if review.rating > 0 else 'Negative'}] {review.text[:80]}...")

asyncio.run(main())
```

### Amazon Reviews (Requires Playwright)

```python
import asyncio
from sentimatrix.providers.scrapers.platforms import AmazonScraper, AmazonConfig

async def main():
    config = AmazonConfig(country="us")

    async with AmazonScraper(config) as scraper:
        reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=20)

        for review in reviews[:5]:
            print(f"[{review.rating}/5] {review.text[:80]}...")

asyncio.run(main())
```

### Using Commercial APIs

```python
import asyncio
from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

async def main():
    async with ScraperAPIClient(api_key="your_key") as client:
        content = await client.scrape(
            "https://www.amazon.com/dp/B08N5WRWNW",
            render_js=True,
            country_code="us",
        )
        print(f"Status: {content.status_code}")
        print(f"Content: {len(content.content)} chars")

asyncio.run(main())
```

---

## Platform Support Matrix

| Platform | Local HTTP | Playwright | Commercial API | Priority |
|----------|------------|------------|----------------|----------|
| Amazon | Partial | Required | Recommended | P0 |
| Steam | Full (JSON) | Not needed | Optional | P0 |
| YouTube | API Only | Not needed | Optional | P0 |
| Reddit | Full (JSON) | Not needed | Optional | P0 |
| IMDB | Full | Optional | Optional | P1 |
| Yelp | Partial | Recommended | Recommended | P1 |
| Trustpilot | Full | Optional | Optional | P1 |
| Google Reviews | Partial | Required | Recommended | P1 |

---

## Selection Guide

### By Budget

| Budget | Recommendation |
|--------|----------------|
| Free | Steam/Reddit (JSON APIs) + Playwright for JS sites |
| < $50/mo | ScraperAPI or ScrapingBee |
| $50-500/mo | Apify or Zyte |
| > $500/mo | Bright Data or Oxylabs |

### By Volume

| Volume | Recommendation |
|--------|----------------|
| < 1K pages/day | Local scrapers (HTTPX + Playwright) |
| 1K-10K pages/day | ScraperAPI + local fallback |
| 10K-100K pages/day | Commercial API (Apify/Zyte) |
| > 100K pages/day | Enterprise (Bright Data/Oxylabs) |

### By Target Site

| Target | Recommendation |
|--------|----------------|
| Static HTML | HTTPXScraper |
| JS-rendered | PlaywrightScraper |
| Anti-bot protected | Commercial API |
| Social media | Official APIs (YouTube, Reddit) |

---

## Configuration

### YAML Configuration

```yaml
scrapers:
  default_provider: playwright

  playwright:
    headless: true
    timeout: 30000
    stealth: true

  httpx:
    timeout: 30
    max_retries: 3

  commercial:
    scraperapi:
      api_key: ${SCRAPERAPI_KEY}
      render_js: true
    brightdata:
      api_key: ${BRIGHTDATA_KEY}
      zone: residential

  rate_limiting:
    requests_per_second: 1
    burst_size: 5
    per_domain: true

  retry:
    max_retries: 3
    backoff_factor: 2.0
    retry_on: [429, 500, 502, 503, 504]
```

### Python Configuration

```python
from sentimatrix.providers.scrapers.platforms import AmazonConfig
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy

# Platform config
config = AmazonConfig(
    country="us",
    filter_verified=False,
    requests_per_second=0.5,
    timeout=30,
)

# Rate limiter
limiter = RateLimiter(
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    requests_per_second=2.0,
    burst_size=5,
    per_domain=True,
)
```

---

## Rate Limiting

### Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Token Bucket | Allows bursts up to bucket size | Default, most flexible |
| Fixed Window | Fixed count per time window | Simple rate limiting |
| Sliding Window | Smooth rate distribution | Strict rate control |

### Example

```python
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy

limiter = RateLimiter(
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    requests_per_second=2.0,
    burst_size=10,
    per_domain=True,
    cooldown_on_429=60,  # Wait 60s on rate limit
)

# Use with scraper
scraper = HTTPXScraper(rate_limiter=limiter)
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| `ScraperBlockedError` | Anti-bot detection | Rotate proxy/user-agent |
| `RateLimitError` | 429 response | Exponential backoff |
| `ScraperTimeoutError` | Slow response | Increase timeout |
| `ScraperConnectionError` | Network issue | Retry with backoff |
| `ScraperParseError` | HTML changed | Update selectors |

---

## Best Practices

1. **Use JSON APIs when available** - Steam, Reddit, YouTube have public APIs
2. **Respect rate limits** - Use RateLimiter with appropriate settings
3. **Rotate identifiers** - User-agent, proxy rotation for scale
4. **Cache results** - Avoid redundant requests
5. **Handle failures gracefully** - Implement retry with backoff
6. **Monitor success rates** - Track blocked requests
7. **Use commercial APIs for anti-bot sites** - Amazon, Yelp benefit from proxy services

---

## Related Documentation

- [Platform Scrapers](./PLATFORM_SCRAPERS.md) - Detailed platform scraper docs
- [Commercial APIs](./API_SCRAPERS.md) - Commercial API integration
- [Rate Limiting](../architecture/OVERVIEW.md) - Rate limiting architecture
- [Troubleshooting](../guides/troubleshooting.md) - Common scraper issues
