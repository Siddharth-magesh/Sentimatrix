---
title: Trustpilot Scraper
description: Scrape company reviews from Trustpilot
---

# Trustpilot Scraper

Scrape reviews and ratings from Trustpilot business profiles.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        "https://www.trustpilot.com/review/example.com",
        platform="trustpilot",
        max_reviews=100
    )

    for review in reviews:
        print(f"Rating: {'*' * review.rating}")
        print(f"Title: {review.title}")
        print(f"Review: {review.text[:200]}...")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://trustpilot.com/review/...",
    platform="trustpilot",
    max_reviews=200,           # Max reviews
    sort_by="recency",         # "recency", "relevance"
    filter_rating=None,        # Filter by stars (1-5)
    language="en",             # Filter by language
)
```

## Supported URL Formats

```python
# Company review page
"https://www.trustpilot.com/review/example.com"

# With filters
"https://trustpilot.com/review/example.com?stars=5"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Review content
    title: str                   # Review title
    author: str                  # Reviewer name
    rating: int                  # 1-5 stars
    posted_date: datetime        # Review date
    helpful_count: int           # Useful votes
    platform: str = "trustpilot"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
review.metadata = {
    "company_name": "Example Company",
    "company_domain": "example.com",
    "review_id": "abc123",
    "verified": True,
    "reply_from_company": "Thank you for...",
    "experience_date": "2024-01-15",
    "country": "United States"
}
```

## Example: Company Reputation Analysis

```python
async with Sentimatrix(config) as sm:
    # Scrape all reviews
    reviews = await sm.scrape_reviews(
        "https://trustpilot.com/review/mycompany.com",
        platform="trustpilot",
        max_reviews=500
    )

    # Analyze trends over time
    timeline = await sm.analyze_temporal([
        {"text": r.text, "date": r.posted_date.isoformat()}
        for r in reviews
    ])

    # Generate competitor comparison
    competitor_reviews = await sm.scrape_reviews(
        "https://trustpilot.com/review/competitor.com",
        platform="trustpilot"
    )

    comparison = await sm.compare_products(
        reviews, competitor_reviews,
        product_a_name="My Company",
        product_b_name="Competitor"
    )
    print(comparison.comparison_summary)
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Default | 10 req/min |
| With Proxy | Higher |

## Related

- [Scraper Overview](index.md)
- [Yelp](yelp.md)
- [Google Reviews](google-reviews.md)
