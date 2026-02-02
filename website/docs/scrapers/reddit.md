---
title: Reddit Scraper
description: Scrape posts and comments from Reddit
---

# Reddit Scraper

Scrape posts, comments, and discussions from Reddit subreddits and threads.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    comments = await sm.scrape_reviews(
        "https://reddit.com/r/gaming/comments/...",
        platform="reddit",
        max_reviews=100
    )

    for comment in comments:
        print(f"u/{comment.author}: {comment.text[:100]}...")
        print(f"  Score: {comment.helpful_count}")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://reddit.com/r/subreddit/comments/...",
    platform="reddit",
    max_reviews=200,           # Max comments
    include_replies=True,      # Include nested replies
    min_score=1,               # Minimum upvote score
    sort_by="best",            # "best", "top", "new", "controversial"
)
```

## Supported URL Formats

```python
# Post URL
"https://www.reddit.com/r/gaming/comments/abc123/post_title/"

# Old Reddit
"https://old.reddit.com/r/gaming/comments/abc123/"

# Subreddit (scrapes top posts)
"https://reddit.com/r/gaming"

# Search results
"https://reddit.com/r/gaming/search?q=review"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Comment/post body
    author: str                  # u/username
    posted_date: datetime        # Post date
    helpful_count: int           # Upvote score
    rating: None                 # Not applicable
    platform: str = "reddit"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
comment.metadata = {
    "subreddit": "gaming",
    "post_id": "abc123",
    "comment_id": "xyz789",
    "is_post": False,
    "parent_id": "t1_abc",
    "depth": 2,
    "permalink": "/r/gaming/comments/..."
}
```

## Example: Subreddit Sentiment

```python
async with Sentimatrix(config) as sm:
    # Scrape product discussion subreddit
    posts = await sm.scrape_reviews(
        "https://reddit.com/r/ProductName",
        platform="reddit",
        max_reviews=500
    )

    # Analyze sentiment
    results = await sm.analyze_batch([p.text for p in posts])

    # Generate insights
    insights = await sm.generate_insights(posts)
    print(insights.summary)
```

## Authentication (Optional)

For higher rate limits, use Reddit API:

```bash
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
export REDDIT_USER_AGENT="sentimatrix/1.0"
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Without Auth | 10 req/min |
| With OAuth | 60 req/min |

## Related

- [Scraper Overview](index.md)
- [YouTube](youtube.md)
- [Selection Guide](selection-guide.md)
