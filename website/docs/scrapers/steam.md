---
title: Steam Scraper
description: Scrape game reviews from Steam
---

# Steam Scraper

Scrape game reviews from Steam's store pages. No browser required.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Browser Required** | No |
| **Authentication** | None |
| **Rate Limit** | 20 requests/min |
| **Data Available** | Reviews, ratings, playtime, helpful votes |

## Quick Start

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        reviews = await sm.scrape_reviews(
            url="https://store.steampowered.com/app/1245620/ELDEN_RING/",
            platform="steam",
            max_reviews=100
        )

        print(f"Scraped {len(reviews)} reviews")

        for review in reviews[:3]:
            print(f"\nRating: {'Positive' if review.rating else 'Negative'}")
            print(f"Playtime: {review.playtime_hours}h")
            print(f"Helpful: {review.helpful_count}")
            print(f"Text: {review.text[:150]}...")

asyncio.run(main())
```

## URL Formats

The Steam scraper accepts various URL formats:

```python
# Full store URL
url = "https://store.steampowered.com/app/1245620/ELDEN_RING/"

# Minimal URL
url = "https://store.steampowered.com/app/1245620"

# Just the app ID (auto-formatted)
reviews = await sm.scrape_reviews(
    url="1245620",
    platform="steam"
)
```

## Options

### Filter by Review Type

```python
reviews = await sm.scrape_reviews(
    url="https://store.steampowered.com/app/1245620",
    platform="steam",
    max_reviews=100,
    review_type="positive"  # "positive", "negative", or "all"
)
```

### Filter by Language

```python
reviews = await sm.scrape_reviews(
    url="https://store.steampowered.com/app/1245620",
    platform="steam",
    max_reviews=100,
    language="english"  # or "all" for all languages
)
```

### Filter by Purchase Type

```python
reviews = await sm.scrape_reviews(
    url="https://store.steampowered.com/app/1245620",
    platform="steam",
    max_reviews=100,
    purchase_type="steam"  # "steam", "non_steam", or "all"
)
```

### Sort Order

```python
reviews = await sm.scrape_reviews(
    url="https://store.steampowered.com/app/1245620",
    platform="steam",
    max_reviews=100,
    sort_by="recent"  # "recent", "helpful", or "funny"
)
```

## Response Schema

```python
class SteamReview:
    text: str              # Review text content
    rating: bool           # True = positive, False = negative
    helpful_count: int     # Number of helpful votes
    funny_count: int       # Number of funny votes
    playtime_hours: float  # Total playtime at review time
    playtime_recent: float # Playtime in last 2 weeks
    author_id: str         # Steam user ID
    posted_date: datetime  # When the review was posted
    language: str          # Review language
    platform: str          # "steam"
```

## Example: Complete Game Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def analyze_game(app_id: str):
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
    )

    async with Sentimatrix(config) as sm:
        # Scrape reviews
        reviews = await sm.scrape_reviews(
            url=app_id,
            platform="steam",
            max_reviews=200
        )

        # Analyze sentiments
        results = await sm.analyze_batch([r.text for r in reviews])

        # Calculate stats
        positive = sum(1 for r in results if r.sentiment == "positive")
        negative = sum(1 for r in results if r.sentiment == "negative")
        total = len(results)

        print(f"Positive: {positive/total*100:.1f}%")
        print(f"Negative: {negative/total*100:.1f}%")

        # Generate summary
        summary = await sm.summarize_reviews(
            [{"text": r.text} for r in reviews[:50]]
        )
        print(f"\nSummary:\n{summary}")

        # Generate insights
        insights = await sm.generate_insights(
            [{"text": r.text} for r in reviews[:50]]
        )

        print("\nPros:")
        for pro in insights.pros[:5]:
            print(f"  + {pro}")

        print("\nCons:")
        for con in insights.cons[:5]:
            print(f"  - {con}")

asyncio.run(analyze_game("1245620"))  # Elden Ring
```

## Rate Limiting

Steam's API has rate limits. Sentimatrix handles this automatically:

```python
from sentimatrix.config import SentimatrixConfig, ScraperConfig, RateLimitConfig

config = SentimatrixConfig(
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=0.3,  # ~18 per minute
            burst_size=3,
        )
    )
)

async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(...)
```

## Pagination

For large numbers of reviews:

```python
# Automatic pagination (default)
reviews = await sm.scrape_reviews(
    url="https://store.steampowered.com/app/1245620",
    platform="steam",
    max_reviews=1000  # Will paginate automatically
)
```

## Popular Game IDs

| Game | App ID |
|------|--------|
| Elden Ring | 1245620 |
| Cyberpunk 2077 | 1091500 |
| Baldur's Gate 3 | 1086940 |
| Counter-Strike 2 | 730 |
| Dota 2 | 570 |
| GTA V | 271590 |
| The Witcher 3 | 292030 |
| Red Dead Redemption 2 | 1174180 |

## Error Handling

```python
from sentimatrix.exceptions import ScraperError, RateLimitError

try:
    reviews = await sm.scrape_reviews(
        url="https://store.steampowered.com/app/invalid",
        platform="steam"
    )
except RateLimitError:
    print("Rate limited, waiting...")
except ScraperError as e:
    print(f"Failed to scrape: {e}")
```

## Best Practices

1. **Use Appropriate Rate Limits**
    - Don't exceed 20 requests/minute
    - Implement delays for bulk scraping

2. **Filter by Language**
    - Reduces noise from non-English reviews
    - Improves sentiment analysis accuracy

3. **Consider Playtime**
    - Reviews from players with more playtime may be more valuable
    - Filter by minimum playtime if needed

4. **Handle Pagination**
    - Steam returns reviews in batches
    - Sentimatrix handles this automatically

## Related

- [Scraper Overview](index.md)
- [Amazon Scraper](amazon.md)
- [IMDB Scraper](imdb.md)
