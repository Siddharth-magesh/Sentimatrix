---
title: Scraper Selection Guide
description: Choose the right scraping approach for your use case
---

# Scraper Selection Guide

This guide helps you choose the best scraping approach based on your requirements.

## Decision Matrix

| Factor | Direct Scraping | Commercial API |
|--------|-----------------|----------------|
| **Volume** | < 1000/day | Unlimited |
| **Reliability** | Variable | High |
| **Cost** | Free | $19-500+/mo |
| **Setup** | Simple | Simple |
| **Anti-bot handling** | Manual | Automatic |
| **Best for** | Development, testing | Production |

## By Platform

### Easy to Scrape (Direct)

| Platform | Difficulty | Browser | Notes |
|----------|------------|---------|-------|
| **Steam** | Easy | No | Official API-like endpoint |
| **Reddit** | Easy | No | Use OAuth for higher limits |
| **YouTube** | Easy | No | Requires API key |

### Moderate (Browser Required)

| Platform | Difficulty | Notes |
|----------|------------|-------|
| **IMDB** | Moderate | Browser + delays |
| **Trustpilot** | Moderate | Rate limiting |
| **Yelp** | Moderate | API available |

### Difficult (Commercial API Recommended)

| Platform | Difficulty | Notes |
|----------|------------|-------|
| **Amazon** | Hard | Aggressive anti-bot |
| **Google Reviews** | Hard | Dynamic loading |
| **TripAdvisor** | Very Hard | Strong protection |

## By Volume

### Low Volume (< 100 reviews/day)

**Use Direct Scraping**

```python
async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        url="https://store.steampowered.com/app/1245620",
        platform="steam",
        max_reviews=50
    )
```

### Medium Volume (100-1000 reviews/day)

**Use Direct Scraping + Rate Limiting**

```python
from sentimatrix.config import SentimatrixConfig, ScraperConfig, RateLimitConfig

config = SentimatrixConfig(
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=0.5,
            burst_size=5,
        ),
        retry=RetryConfig(
            max_retries=3,
            backoff_factor=2.0,
        )
    )
)
```

### High Volume (1000+ reviews/day)

**Use Commercial APIs**

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        api_provider="scraperapi",  # or brightdata, oxylabs
        api_key="your-key"
    )
)
```

## Commercial API Comparison

### By Price

| Service | Starting Price | Best For |
|---------|----------------|----------|
| **ScrapingAnt** | $19/mo | Budget projects |
| **ScraperAPI** | $49/mo | General use |
| **ScrapingBee** | $49/mo | Screenshots |
| **Zyte** | $450/mo | AI extraction |
| **Bright Data** | $500/mo | Enterprise |
| **Oxylabs** | Custom | E-commerce |

### By Feature

| Service | Proxy Pool | JS Rendering | Geo-targeting | AI Extraction |
|---------|:----------:|:------------:|:-------------:|:-------------:|
| **Bright Data** | 72M+ | :material-check: | :material-check: | :material-check: |
| **Oxylabs** | 100M+ | :material-check: | :material-check: | :material-check: |
| **Zyte** | 50M+ | :material-check: | :material-check: | :material-check: |
| **ScraperAPI** | 40M+ | :material-check: | :material-check: | :material-close: |
| **ScrapingBee** | 1M+ | :material-check: | :material-check: | :material-close: |
| **ScrapingAnt** | 1M+ | :material-check: | :material-check: | :material-close: |
| **Apify** | Varies | :material-check: | :material-check: | :material-check: |

### By Use Case

=== "E-commerce (Amazon, eBay)"

    **Recommended**: Oxylabs or Bright Data

    - Specialized e-commerce solutions
    - High success rates on protected sites
    - Product data extraction

    ```python
    config = SentimatrixConfig(
        scraper=ScraperConfig(
            api_provider="oxylabs",
            api_key="your-key"
        )
    )
    ```

=== "General Purpose"

    **Recommended**: ScraperAPI or ScrapingBee

    - Good balance of features and price
    - Easy integration
    - Reliable for most sites

    ```python
    config = SentimatrixConfig(
        scraper=ScraperConfig(
            api_provider="scraperapi",
            api_key="your-key"
        )
    )
    ```

=== "Budget Projects"

    **Recommended**: ScrapingAnt

    - Lowest starting price
    - Basic features
    - Good for low volume

    ```python
    config = SentimatrixConfig(
        scraper=ScraperConfig(
            api_provider="scrapingant",
            api_key="your-key"
        )
    )
    ```

=== "Enterprise"

    **Recommended**: Bright Data or Zyte

    - Largest proxy pools
    - SLAs available
    - Advanced features

    ```python
    config = SentimatrixConfig(
        scraper=ScraperConfig(
            api_provider="brightdata",
            username="your-username",
            password="your-password"
        )
    )
    ```

## Cost Estimation

### Direct Scraping

| Cost Type | Amount |
|-----------|--------|
| Infrastructure | $0-20/mo (compute) |
| Proxies | $0-50/mo (optional) |
| Total | $0-70/mo |

### Commercial APIs

| Volume | ScrapingAnt | ScraperAPI | Bright Data |
|--------|-------------|------------|-------------|
| 10K requests | $19 | $49 | ~$50 |
| 100K requests | ~$100 | ~$200 | ~$200 |
| 1M requests | ~$500 | ~$800 | ~$1000 |

## Reliability Comparison

### Success Rates (Approximate)

| Platform | Direct | Commercial |
|----------|--------|------------|
| Steam | 99% | 99% |
| Reddit | 95% | 99% |
| YouTube | 90% | 99% |
| IMDB | 80% | 95% |
| Amazon | 50-70% | 90%+ |
| Google Reviews | 40-60% | 85%+ |

## Configuration Examples

### Development (Free)

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=0.5,
            burst_size=3,
        ),
        browser=BrowserConfig(
            headless=True,
            stealth_mode=True,
        )
    )
)
```

### Production (Low Budget)

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        api_provider="scrapingant",
        api_key=os.getenv("SCRAPINGANT_KEY"),
        # Fallback to direct scraping
        fallback_to_direct=True,
    )
)
```

### Production (High Volume)

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        api_provider="brightdata",
        username=os.getenv("BRIGHTDATA_USER"),
        password=os.getenv("BRIGHTDATA_PASS"),
        rate_limit=RateLimitConfig(
            requests_per_second=10,  # Higher with commercial
            burst_size=50,
        )
    )
)
```

### Multi-Platform

```python
# Different settings per platform
config = SentimatrixConfig(
    scraper=ScraperConfig(
        platform_overrides={
            "steam": {
                "use_api": False,  # Direct is fine
            },
            "amazon": {
                "use_api": True,
                "api_provider": "scraperapi",
            },
            "google": {
                "use_api": True,
                "api_provider": "brightdata",
            },
        }
    )
)
```

## Summary Recommendations

### For Development/Testing

Use **direct scraping** with rate limiting:

- Free
- Good enough for testing
- Easy to set up

### For Production (Budget)

Use **ScrapingAnt** or **ScraperAPI**:

- $19-49/month
- Good reliability
- Easy integration

### For Production (Scale)

Use **Bright Data** or **Oxylabs**:

- Enterprise features
- Best reliability
- Highest success rates

### For E-commerce Focus

Use **Oxylabs**:

- Specialized for e-commerce
- High success on Amazon, eBay
- Product data extraction
