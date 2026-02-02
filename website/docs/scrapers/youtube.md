---
title: YouTube Scraper
description: Scrape comments and video metadata from YouTube
---

# YouTube Scraper

Scrape comments, video information, and engagement data from YouTube videos.

## Quick Start

```python
from sentimatrix import Sentimatrix

async with Sentimatrix() as sm:
    comments = await sm.scrape_reviews(
        "https://www.youtube.com/watch?v=VIDEO_ID",
        platform="youtube",
        max_reviews=100
    )

    for comment in comments:
        print(f"{comment.author}: {comment.text}")
        print(f"  Likes: {comment.helpful_count}")
```

## Configuration

```python
reviews = await sm.scrape_reviews(
    url="https://youtube.com/watch?v=...",
    platform="youtube",
    max_reviews=200,           # Max comments to fetch
    include_replies=True,      # Include reply threads
    sort_by="top",             # "top" or "newest"
)
```

## Supported URL Formats

```python
# Video URL
"https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Short URL
"https://youtu.be/dQw4w9WgXcQ"

# Just video ID
"dQw4w9WgXcQ"
```

## Review Object

```python
@dataclass
class Review:
    text: str                    # Comment text
    author: str                  # Channel name
    posted_date: datetime        # Comment date
    helpful_count: int           # Like count
    rating: None                 # Not applicable
    platform: str = "youtube"
    metadata: dict               # Extra data
```

## Metadata Fields

```python
comment.metadata = {
    "author_channel_id": "UC...",
    "reply_count": 42,
    "is_reply": False,
    "parent_id": None,
    "video_id": "dQw4w9WgXcQ",
    "video_title": "Never Gonna Give You Up"
}
```

## Example: Video Sentiment Analysis

```python
async with Sentimatrix(config) as sm:
    # Scrape comments
    comments = await sm.scrape_reviews(
        "https://youtube.com/watch?v=...",
        platform="youtube",
        max_reviews=500
    )

    # Analyze sentiment
    results = await sm.analyze_batch([c.text for c in comments])

    # Calculate stats
    positive = sum(1 for r in results if r.sentiment == "positive")
    negative = sum(1 for r in results if r.sentiment == "negative")

    print(f"Positive: {positive/len(results):.1%}")
    print(f"Negative: {negative/len(results):.1%}")
```

## Rate Limits

| Method | Rate Limit |
|--------|------------|
| Without API | 20 req/min |
| With API Key | 100 req/min |

## Using YouTube Data API

For higher limits, use the official API:

```bash
export YOUTUBE_API_KEY="your-api-key"
```

## Related

- [Scraper Overview](index.md)
- [Reddit](reddit.md)
- [Selection Guide](selection-guide.md)
