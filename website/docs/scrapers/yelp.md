---
title: Yelp Scraper
description: Scrape business reviews from Yelp
---

# Yelp Scraper

Scrape reviews and ratings from Yelp business listings.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        "https://www.yelp.com/biz/restaurant-name-city",
        platform="yelp",
        max_reviews=100
    )

    for review in reviews:
        print(f"Rating: {'*' * review.rating}")
        print(f"Review: {review.text[:200]}...")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://yelp.com/biz/...",
    platform="yelp",
    max_reviews=200,           # Max reviews
    sort_by="date",            # "date", "rating_asc", "rating_desc", "elites"
    filter_rating=None,        # Filter by stars (1-5)
)
```

## Supported URL Formats

```python
# Business page
"https://www.yelp.com/biz/business-name-city"

# With query params
"https://yelp.com/biz/business-name?sort_by=date_desc"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Review content
    author: str                  # Reviewer name
    rating: int                  # 1-5 stars
    posted_date: datetime        # Review date
    helpful_count: int           # Useful votes
    platform: str = "yelp"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
review.metadata = {
    "business_name": "Restaurant Name",
    "business_id": "restaurant-name-city",
    "location": "San Francisco, CA",
    "is_elite": True,
    "photos_count": 3,
    "funny_count": 2,
    "cool_count": 5
}
```

## Example: Restaurant Analysis

```python
async with Sentimatrix(config) as sm:
    # Scrape restaurant reviews
    reviews = await sm.scrape_reviews(
        "https://yelp.com/biz/my-restaurant-sf",
        platform="yelp",
        max_reviews=200
    )

    # Aspect-based analysis
    aspects = ["food", "service", "ambiance", "price"]
    for review in reviews:
        aspect_sentiment = await sm.analyze_aspects(
            review.text,
            aspects=aspects
        )
        print(aspect_sentiment)

    # Generate insights
    insights = await sm.generate_insights(reviews)
    print("PROS:", insights.pros)
    print("CONS:", insights.cons)
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Default | 10 req/min |
| With API Key | 50 req/min |

## Yelp Fusion API

For official API access:

```bash
export YELP_API_KEY="your-api-key"
```

## Related

- [Scraper Overview](index.md)
- [Google Reviews](google-reviews.md)
- [Trustpilot](trustpilot.md)
