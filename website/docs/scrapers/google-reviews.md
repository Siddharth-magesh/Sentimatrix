---
title: Google Reviews Scraper
description: Scrape business reviews from Google Maps
---

# Google Reviews Scraper

Scrape reviews and ratings from Google Maps business listings.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        "https://maps.google.com/maps/place/...",
        platform="google_reviews",
        max_reviews=100
    )

    for review in reviews:
        print(f"Rating: {'*' * review.rating}")
        print(f"Review: {review.text[:200]}...")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://maps.google.com/...",
    platform="google_reviews",
    max_reviews=200,           # Max reviews
    sort_by="newest",          # "newest", "highest", "lowest", "relevant"
    use_browser=True,          # Required for Google
)
```

## Supported URL Formats

```python
# Google Maps URL
"https://www.google.com/maps/place/Business+Name/@lat,lng,..."

# Place ID
"ChIJ..."

# Search URL
"https://google.com/maps/search/restaurant+near+me"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Review content
    author: str                  # Reviewer name
    rating: int                  # 1-5 stars
    posted_date: datetime        # Review date
    helpful_count: int           # Likes
    platform: str = "google_reviews"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
review.metadata = {
    "place_id": "ChIJ...",
    "business_name": "Restaurant Name",
    "business_address": "123 Main St",
    "reviewer_reviews_count": 42,
    "reviewer_photos_count": 15,
    "is_local_guide": True,
    "response_from_owner": "Thank you..."
}
```

## Example: Local Business Analysis

```python
async with Sentimatrix(config) as sm:
    # Scrape Google reviews
    reviews = await sm.scrape_reviews(
        "https://maps.google.com/maps/place/...",
        platform="google_reviews",
        max_reviews=200,
        use_browser=True
    )

    # Aspect-based analysis for restaurants
    aspects = ["food", "service", "price", "atmosphere", "cleanliness"]

    for review in reviews[:10]:
        sentiment = await sm.analyze_aspects(review.text, aspects=aspects)
        print(f"Review aspects: {sentiment}")

    # Compare with competitors
    insights = await sm.generate_insights(reviews)
    print("Key themes:", insights.themes)
```

## Browser Requirement

Google Reviews requires browser-based scraping due to JavaScript rendering:

```python
config = SentimatrixConfig(
    scrapers=ScraperConfig(
        provider="playwright",  # Required
        headless=True
    )
)
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Default | 5 req/min |
| With Proxy | Higher |

## Related

- [Scraper Overview](index.md)
- [Yelp](yelp.md)
- [Trustpilot](trustpilot.md)
