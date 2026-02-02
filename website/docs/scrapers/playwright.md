---
title: Playwright Scraper
description: Browser automation for JavaScript-rendered pages
---

# Playwright Scraper

Playwright provides full browser automation for scraping JavaScript-heavy pages that require rendering.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ScraperConfig

config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="playwright",
        headless=True,
        timeout=30
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="amazon")
```

## Setup

```bash
pip install playwright
playwright install chromium
```

## Configuration

```python
ScraperConfig(
    provider="playwright",
    headless=True,                 # Run without visible browser
    timeout=30,                    # Page load timeout
    viewport_width=1920,           # Browser width
    viewport_height=1080,          # Browser height
    wait_for_selector=".reviews",  # Wait for element
    screenshots=False,             # Capture screenshots
    screenshot_dir="./screenshots"
)
```

## Features

- **Full Rendering**: JavaScript execution
- **Anti-Detection**: Stealth mode
- **Screenshots**: Debug captures
- **Multiple Browsers**: Chromium, Firefox, WebKit
- **Network Interception**: Request/response control

## When to Use Playwright

| Scenario | Playwright | HTTPX |
|----------|------------|-------|
| JavaScript pages | Required | No |
| Login required | Best | Limited |
| Dynamic content | Required | No |
| High volume | Slower | Best |
| Resource usage | High | Low |

## Best For

- **Amazon**: Heavy JavaScript
- **YouTube**: Dynamic loading
- **Yelp**: Interactive pages
- **Google Reviews**: JavaScript rendering

## Example: Amazon Scraping

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="playwright",
        headless=True,
        timeout=60,
        wait_for_selector="[data-hook='review']"
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(
        "https://amazon.com/product/...",
        platform="amazon",
        max_reviews=100
    )
```

## Stealth Mode

Playwright scraper includes anti-detection measures:

- Random user agents
- Human-like delays
- Mouse movement simulation
- Fingerprint randomization

## Proxy Support

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="playwright",
        proxy=ProxyConfig(
            enabled=True,
            url="http://proxy:8080",
            username="user",
            password="pass"
        )
    )
)
```

## Debug with Screenshots

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="playwright",
        screenshots=True,
        screenshot_dir="./debug"
    )
)
```

## Related

- [Scraper Overview](index.md)
- [HTTPX](httpx.md)
- [Amazon Scraper](amazon.md)
