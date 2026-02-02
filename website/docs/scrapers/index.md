---
title: Scrapers
description: Collect reviews and feedback from popular platforms
---

# Web Scrapers

Sentimatrix provides a comprehensive scraping infrastructure to collect reviews and feedback from popular platforms.

## Scraper Categories

<div class="grid">

<div class="card">
<h3>:material-web: Core Scrapers</h3>
<p>Foundational scrapers for any website.</p>
<p>HTTPX (static), Playwright (dynamic)</p>
</div>

<div class="card">
<h3>:material-store: Platform Scrapers</h3>
<p>Pre-built scrapers for popular platforms.</p>
<p>Amazon, Steam, YouTube, Reddit, IMDB, Yelp, Trustpilot, Google Reviews</p>
</div>

<div class="card">
<h3>:material-api: Commercial APIs</h3>
<p>Enterprise-grade scraping services.</p>
<p>ScraperAPI, Apify, Bright Data, Oxylabs, Zyte, ScrapingBee, ScrapingAnt</p>
</div>

</div>

## Platform Support Matrix

| Platform | Browser Required | Auth Required | Rate Limit | Status |
|----------|:----------------:|:-------------:|:----------:|:------:|
| **Amazon** | :material-check: | :material-close: | 10/min | <span class="status stable">Stable</span> |
| **Steam** | :material-close: | :material-close: | 20/min | <span class="status stable">Stable</span> |
| **YouTube** | :material-close: | API Key | 100/min | <span class="status stable">Stable</span> |
| **Reddit** | :material-close: | OAuth | 60/min | <span class="status stable">Stable</span> |
| **IMDB** | :material-check: | :material-close: | 15/min | <span class="status stable">Stable</span> |
| **Yelp** | :material-check: | API Key | 50/min | <span class="status stable">Stable</span> |
| **Trustpilot** | :material-check: | :material-close: | 10/min | <span class="status stable">Stable</span> |
| **Google Reviews** | :material-check: | :material-close: | 5/min | <span class="status beta">Beta</span> |

## Quick Start

### Basic Scraping (Steam - No Browser)

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        reviews = await sm.scrape_reviews(
            url="https://store.steampowered.com/app/1245620/ELDEN_RING/",
            platform="steam",
            max_reviews=50
        )

        for review in reviews[:5]:
            print(f"Rating: {review.rating}")
            print(f"Text: {review.text[:100]}...")
            print()

asyncio.run(main())
```

### Browser-Based Scraping (Amazon)

```python
async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        url="https://www.amazon.com/dp/B0BSHF7WHW",
        platform="amazon",
        max_reviews=30,
        use_browser=True  # Enables Playwright
    )
```

!!! info "Playwright Required"
    For browser-based scraping:
    ```bash
    pip install sentimatrix[scraping]
    playwright install chromium
    ```

### Using Commercial APIs

```python
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scraper=ScraperConfig(
        api_provider="scraperapi",
        api_key="your-scraperapi-key"
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(
        url="https://www.amazon.com/dp/B0BSHF7WHW",
        platform="amazon",
        max_reviews=100
    )
```

## Commercial API Comparison

| Service | Proxy Pool | JS Rendering | Starting Price | Best For |
|---------|------------|--------------|----------------|----------|
| **ScraperAPI** | 40M+ | :material-check: | $49/mo | General use |
| **Apify** | Varies | :material-check: | Pay-per-use | Pre-built actors |
| **Bright Data** | 72M+ | :material-check: | $500/mo | Enterprise |
| **Oxylabs** | 100M+ | :material-check: | Custom | E-commerce |
| **Zyte** | 50M+ | :material-check: | $450/mo | AI extraction |
| **ScrapingBee** | 1M+ | :material-check: | $49/mo | Screenshots |
| **ScrapingAnt** | 1M+ | :material-check: | $19/mo | Budget |

## Configuration

### YAML Configuration

```yaml title="sentimatrix.yaml"
scraper:
  # Rate limiting
  rate_limit:
    requests_per_second: 2
    burst_size: 5

  # Retry settings
  retry:
    max_retries: 3
    backoff_factor: 2.0

  # Browser settings
  browser:
    headless: true
    timeout: 30000

  # Commercial API (optional)
  api:
    provider: scraperapi
    # api_key loaded from SCRAPERAPI_KEY env var
```

### Python Configuration

```python
from sentimatrix.config import (
    SentimatrixConfig,
    ScraperConfig,
    RateLimitConfig,
    RetryConfig,
    BrowserConfig,
)

config = SentimatrixConfig(
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=2,
            burst_size=5,
        ),
        retry=RetryConfig(
            max_retries=3,
            backoff_factor=2.0,
        ),
        browser=BrowserConfig(
            headless=True,
            timeout=30000,
        ),
    )
)
```

## Error Handling

```python
from sentimatrix.exceptions import (
    ScraperError,
    RateLimitError,
    BlockedError,
    ParseError,
)

async with Sentimatrix() as sm:
    try:
        reviews = await sm.scrape_reviews(url, platform="amazon")
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after}s")
    except BlockedError:
        print("IP blocked, try using a commercial API")
    except ParseError as e:
        print(f"Failed to parse response: {e}")
    except ScraperError as e:
        print(f"Scraping failed: {e}")
```

## Best Practices

1. **Respect Rate Limits**
    - Don't exceed platform limits
    - Implement exponential backoff
    - Use delays between requests

2. **Use Browser Only When Needed**
    - Steam, Reddit: No browser required
    - Amazon, IMDB: Browser recommended

3. **Handle Blocks Gracefully**
    - Rotate user agents
    - Use commercial APIs for scale
    - Implement retry logic

4. **Cache Responses**
    - Avoid repeated requests
    - Store scraped data locally

5. **Stay Compliant**
    - Check robots.txt
    - Follow terms of service
    - Don't overload servers

## Scraper Documentation

### Core Scrapers
- [HTTPX Scraper](httpx.md) - Static pages
- [Playwright Scraper](playwright.md) - Dynamic pages

### Platform Scrapers
- [Amazon](amazon.md) - Product reviews
- [Steam](steam.md) - Game reviews
- [YouTube](youtube.md) - Video comments
- [Reddit](reddit.md) - Posts and comments
- [IMDB](imdb.md) - Movie reviews
- [Yelp](yelp.md) - Business reviews
- [Trustpilot](trustpilot.md) - Company reviews
- [Google Reviews](google-reviews.md) - Local business reviews

### Commercial APIs
- [Overview](commercial/index.md)
- [ScraperAPI](commercial/scraperapi.md)
- [Apify](commercial/apify.md)
- [Bright Data](commercial/brightdata.md)
- [Oxylabs](commercial/oxylabs.md)
- [Zyte](commercial/zyte.md)
- [ScrapingBee](commercial/scrapingbee.md)
- [ScrapingAnt](commercial/scrapingant.md)
