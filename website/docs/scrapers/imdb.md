---
title: IMDB Scraper
description: Scrape movie and TV show reviews from IMDB
---

# IMDB Scraper

Scrape user reviews and ratings from IMDB for movies, TV shows, and episodes.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        "https://www.imdb.com/title/tt0111161/",  # Shawshank Redemption
        platform="imdb",
        max_reviews=100
    )

    for review in reviews:
        print(f"Rating: {review.rating}/10")
        print(f"Title: {review.title}")
        print(f"Review: {review.text[:200]}...")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://imdb.com/title/tt.../",
    platform="imdb",
    max_reviews=200,           # Max reviews
    sort_by="helpfulness",     # "helpfulness", "date", "rating"
    filter_rating=None,        # Filter by star rating (1-10)
)
```

## Supported URL Formats

```python
# Movie/TV show page
"https://www.imdb.com/title/tt0111161/"

# Reviews page
"https://www.imdb.com/title/tt0111161/reviews"

# Just IMDB ID
"tt0111161"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Review content
    title: str                   # Review title
    author: str                  # Reviewer username
    rating: int                  # 1-10 stars
    posted_date: datetime        # Review date
    helpful_count: int           # Helpful votes
    platform: str = "imdb"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
review.metadata = {
    "imdb_id": "tt0111161",
    "movie_title": "The Shawshank Redemption",
    "movie_year": 1994,
    "review_id": "rw1234567",
    "spoiler_warning": False,
    "total_votes": 150
}
```

## Example: Movie Sentiment Analysis

```python
async with Sentimatrix(config) as sm:
    # Scrape reviews
    reviews = await sm.scrape_reviews(
        "tt0111161",  # Shawshank Redemption
        platform="imdb",
        max_reviews=500
    )

    # Analyze sentiment
    results = await sm.analyze_batch([r.text for r in reviews])

    # Correlate with ratings
    for review, result in zip(reviews, results):
        print(f"Rating: {review.rating}/10, Sentiment: {result.sentiment}")

    # Generate summary
    summary = await sm.summarize_reviews(reviews)
    print(summary)
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Default | 15 req/min |
| With Proxy | Higher |

## Related

- [Scraper Overview](index.md)
- [Steam](steam.md)
- [Selection Guide](selection-guide.md)
