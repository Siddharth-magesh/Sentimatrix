---
title: Commercial Scraping APIs
description: Enterprise-grade scraping with commercial API providers
---

# Commercial Scraping APIs

Commercial scraping APIs provide reliable, scalable web scraping without the hassle of managing proxies, handling CAPTCHAs, or dealing with blocks.

## Why Use Commercial APIs?

| Challenge | Direct Scraping | Commercial API |
|-----------|-----------------|----------------|
| IP Blocks | Manage proxies yourself | Automatic rotation |
| CAPTCHAs | Manual solving | Automatic solving |
| Rate Limits | Careful throttling | Higher limits |
| JavaScript | Run browsers | Cloud rendering |
| Maintenance | Constant updates | Provider handles |
| Scale | Limited | Millions of requests |

## Supported Providers

| Provider | Proxy Pool | JS Rendering | Starting Price |
|----------|------------|--------------|----------------|
| [ScraperAPI](scraperapi.md) | 40M+ | :material-check: | $49/mo |
| [Apify](apify.md) | Varies | :material-check: | Pay-per-use |
| [Bright Data](brightdata.md) | 72M+ | :material-check: | $500/mo |
| [Oxylabs](oxylabs.md) | 100M+ | :material-check: | Custom |
| [Zyte](zyte.md) | 50M+ | :material-check: | $450/mo |
| [ScrapingBee](scrapingbee.md) | 1M+ | :material-check: | $49/mo |
| [ScrapingAnt](scrapingant.md) | 1M+ | :material-check: | $19/mo |

## Quick Start

### Using ScraperAPI

```python
from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

async with ScraperAPIClient(api_key="your-api-key") as client:
    result = await client.scrape("https://example.com")
    print(f"Status: {result.status_code}")
    print(f"Content: {result.content[:200]}")
```

### Using Apify

```python
from sentimatrix.providers.scrapers.commercial import ApifyClient

async with ApifyClient(api_token="your-token") as client:
    # Basic scraping (uses cheerio-scraper)
    result = await client.scrape("https://example.com")

    # Or run specific actors
    run = await client.run_actor(
        "apify/web-scraper",
        input={"startUrls": [{"url": "https://example.com"}]}
    )
    items = await client.get_dataset_items(run["defaultDatasetId"])
```

### Using ScrapingBee

```python
from sentimatrix.providers.scrapers.commercial import ScrapingBeeClient

async with ScrapingBeeClient(api_key="your-api-key") as client:
    result = await client.scrape(
        "https://example.com",
        render_js=True,  # Enable JavaScript rendering
        premium_proxy=True  # Use premium proxies
    )
```

### Environment Variables

```bash
# ScraperAPI
export SCRAPERAPI_KEY="your-key"

# Apify
export APIFY_TOKEN="your-token"

# Bright Data
export BRIGHTDATA_USERNAME="your-username"
export BRIGHTDATA_PASSWORD="your-password"

# Oxylabs
export OXYLABS_USERNAME="your-username"
export OXYLABS_PASSWORD="your-password"

# Zyte
export ZYTE_API_KEY="your-key"

# ScrapingBee
export SCRAPINGBEE_API_KEY="your-key"

# ScrapingAnt
export SCRAPINGANT_API_KEY="your-key"
```

## Provider Comparison

### By Use Case

| Use Case | Recommended |
|----------|-------------|
| **Budget** | ScrapingAnt ($19/mo) |
| **General** | ScraperAPI, ScrapingBee |
| **E-commerce** | Oxylabs, Bright Data |
| **Enterprise** | Bright Data, Zyte |
| **AI Extraction** | Zyte, Apify |

### By Feature

| Feature | Best Provider |
|---------|---------------|
| **Largest Proxy Pool** | Oxylabs (100M+) |
| **Best for Amazon** | Oxylabs |
| **AI-Powered Extraction** | Zyte, Apify |
| **Lowest Price** | ScrapingAnt |
| **Screenshots** | ScrapingBee |
| **Pre-built Scrapers** | Apify |

## Configuration Options

```python
ScraperConfig(
    # Provider selection
    api_provider="scraperapi",  # or brightdata, oxylabs, etc.

    # Authentication
    api_key="your-key",         # For most providers
    username="user",            # For Bright Data, Oxylabs
    password="pass",

    # Request options
    render_js=True,             # Enable JS rendering
    country="us",               # Geo-targeting
    premium_proxy=False,        # Use premium proxies

    # Retry settings
    retry=RetryConfig(
        max_retries=3,
        backoff_factor=2.0,
    ),
)
```

## Cost Estimation

| Volume | ScrapingAnt | ScraperAPI | Bright Data |
|--------|-------------|------------|-------------|
| 10K requests | $19 | $49 | ~$50 |
| 50K requests | ~$50 | ~$100 | ~$150 |
| 100K requests | ~$80 | ~$200 | ~$300 |
| 500K requests | ~$300 | ~$600 | ~$1000 |

## Best Practices

1. **Start with Budget Providers**
    - Test with ScrapingAnt or ScraperAPI
    - Scale up as needed

2. **Use Geo-Targeting**
    - Match target site location
    - Reduces blocks

3. **Enable JS Rendering Selectively**
    - Only when needed
    - Costs more credits

4. **Implement Caching**
    - Avoid repeated requests
    - Save costs

5. **Monitor Usage**
    - Track API calls
    - Set budget alerts
