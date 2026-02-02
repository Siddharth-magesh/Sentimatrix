---
title: Amazon Scraper
description: Scrape product reviews from Amazon
---

# Amazon Scraper

Scrape product reviews from Amazon product pages. Browser (Playwright) required for JavaScript rendering.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Browser Required** | Yes (Playwright) |
| **Authentication** | None |
| **Rate Limit** | 10 requests/min |
| **Data Available** | Reviews, ratings, titles, verified purchase, helpful votes |

## Setup

Install Playwright:

```bash
pip install sentimatrix[scraping]
playwright install chromium
```

## Quick Start

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        reviews = await sm.scrape_reviews(
            url="https://www.amazon.com/dp/B0BSHF7WHW",
            platform="amazon",
            max_reviews=50,
            use_browser=True
        )

        print(f"Scraped {len(reviews)} reviews")

        for review in reviews[:3]:
            print(f"\n[{review.rating}/5] {review.title}")
            print(f"Verified: {review.verified_purchase}")
            print(f"Helpful: {review.helpful_count}")
            print(f"Text: {review.text[:150]}...")

asyncio.run(main())
```

## URL Formats

```python
# Full product URL
url = "https://www.amazon.com/dp/B0BSHF7WHW"

# With product name
url = "https://www.amazon.com/Apple-iPhone-15-Pro-256GB/dp/B0BSHF7WHW"

# Review page URL
url = "https://www.amazon.com/product-reviews/B0BSHF7WHW"

# Just ASIN
reviews = await sm.scrape_reviews(
    url="B0BSHF7WHW",
    platform="amazon"
)
```

## Options

### Filter by Rating

```python
reviews = await sm.scrape_reviews(
    url="https://www.amazon.com/dp/B0BSHF7WHW",
    platform="amazon",
    max_reviews=100,
    rating_filter=5  # 1-5 or None for all
)
```

### Filter by Verified Purchase

```python
reviews = await sm.scrape_reviews(
    url="https://www.amazon.com/dp/B0BSHF7WHW",
    platform="amazon",
    max_reviews=100,
    verified_only=True
)
```

### Sort Order

```python
reviews = await sm.scrape_reviews(
    url="https://www.amazon.com/dp/B0BSHF7WHW",
    platform="amazon",
    max_reviews=100,
    sort_by="recent"  # "recent", "helpful", or "top"
)
```

### Different Marketplaces

```python
# Amazon UK
reviews = await sm.scrape_reviews(
    url="https://www.amazon.co.uk/dp/B0BSHF7WHW",
    platform="amazon"
)

# Amazon Germany
reviews = await sm.scrape_reviews(
    url="https://www.amazon.de/dp/B0BSHF7WHW",
    platform="amazon"
)

# Amazon Japan
reviews = await sm.scrape_reviews(
    url="https://www.amazon.co.jp/dp/B0BSHF7WHW",
    platform="amazon"
)
```

## Response Schema

```python
class AmazonReview:
    text: str               # Review text content
    title: str              # Review title
    rating: int             # 1-5 star rating
    helpful_count: int      # Helpful votes
    verified_purchase: bool # Is verified purchase
    author_name: str        # Reviewer name
    posted_date: datetime   # Review date
    images: list[str]       # Image URLs (if any)
    variant: str            # Product variant purchased
    platform: str           # "amazon"
    marketplace: str        # "amazon.com", "amazon.co.uk", etc.
```

## Using Commercial APIs

For high-volume scraping, use commercial APIs to avoid blocks:

```python
from sentimatrix.config import SentimatrixConfig, ScraperConfig

# Using ScraperAPI
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
        max_reviews=500  # Can scrape more with commercial API
    )
```

## Example: Product Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def analyze_product(asin: str):
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
    )

    async with Sentimatrix(config) as sm:
        # Scrape reviews
        reviews = await sm.scrape_reviews(
            url=asin,
            platform="amazon",
            max_reviews=100,
            use_browser=True
        )

        # Calculate rating distribution
        ratings = {}
        for review in reviews:
            ratings[review.rating] = ratings.get(review.rating, 0) + 1

        print("Rating Distribution:")
        for rating in sorted(ratings.keys(), reverse=True):
            count = ratings[rating]
            pct = count / len(reviews) * 100
            bar = "=" * int(pct / 2)
            print(f"{rating}: {bar} {pct:.1f}%")

        # Analyze sentiments
        results = await sm.analyze_batch([r.text for r in reviews])

        # Generate aspect-based analysis
        aspects = await sm.analyze_aspects(
            [r.text for r in reviews],
            aspects=["quality", "price", "shipping", "durability", "design"]
        )

        print("\nAspect Sentiments:")
        for aspect, sentiment in aspects.items():
            print(f"  {aspect}: {sentiment}")

        # Generate summary
        summary = await sm.summarize_reviews(
            [{"text": r.text, "rating": r.rating} for r in reviews[:50]]
        )
        print(f"\nSummary:\n{summary}")

asyncio.run(analyze_product("B0BSHF7WHW"))
```

## Handling Anti-Bot Measures

Amazon has aggressive anti-bot measures. Recommended approaches:

### 1. Use Stealth Mode

```python
from sentimatrix.config import ScraperConfig, BrowserConfig

config = SentimatrixConfig(
    scraper=ScraperConfig(
        browser=BrowserConfig(
            headless=True,
            stealth_mode=True,  # Use stealth settings
            random_delays=True,  # Add random delays
        )
    )
)
```

### 2. Rotate User Agents

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        rotate_user_agents=True
    )
)
```

### 3. Use Commercial APIs (Recommended)

For reliable scraping at scale:

```python
config = SentimatrixConfig(
    scraper=ScraperConfig(
        api_provider="scraperapi",  # or "brightdata", "oxylabs"
        api_key="your-key"
    )
)
```

## Rate Limiting

```python
from sentimatrix.config import ScraperConfig, RateLimitConfig

config = SentimatrixConfig(
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=0.15,  # ~9 per minute
            burst_size=2,
            cooldown_on_429=60,  # Wait 60s on rate limit
        )
    )
)
```

## Error Handling

```python
from sentimatrix.exceptions import (
    ScraperError,
    BlockedError,
    CaptchaError,
    RateLimitError,
)

try:
    reviews = await sm.scrape_reviews(
        url="https://www.amazon.com/dp/B0BSHF7WHW",
        platform="amazon",
        use_browser=True
    )
except CaptchaError:
    print("CAPTCHA detected, try using commercial API")
except BlockedError:
    print("IP blocked, rotate IP or use proxy")
except RateLimitError as e:
    print(f"Rate limited, wait {e.retry_after}s")
except ScraperError as e:
    print(f"Failed: {e}")
```

## Best Practices

1. **Use Commercial APIs for Scale**
    - Direct scraping works for small volumes
    - Commercial APIs handle anti-bot measures

2. **Implement Delays**
    - Wait between requests
    - Randomize timing

3. **Monitor for Blocks**
    - Check for CAPTCHA responses
    - Rotate IPs if blocked

4. **Cache Results**
    - Store scraped reviews
    - Avoid repeated requests

5. **Filter Verified Purchases**
    - Higher quality reviews
    - More trustworthy data

## Related

- [Scraper Overview](index.md)
- [ScraperAPI](commercial/scraperapi.md)
- [Steam Scraper](steam.md)
