---
title: Social Media Examples
description: Sentiment analysis for social media content
---

# Social Media Examples

Examples for analyzing social media content from YouTube, Reddit, and more.

## YouTube Comments Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def analyze_youtube():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        # Scrape comments
        comments = await sm.scrape_reviews(
            "https://youtube.com/watch?v=VIDEO_ID",
            platform="youtube",
            max_reviews=200
        )

        # Analyze sentiment
        results = await sm.analyze_batch([c.text for c in comments])

        # Emotion analysis
        emotions = await sm.detect_emotions_batch([c.text for c in comments])

        # Stats
        positive = sum(1 for r in results if r.sentiment == "positive")
        angry = sum(1 for e in emotions if e.primary == "anger")

        print(f"Positive: {positive/len(results):.1%}")
        print(f"Angry comments: {angry}")

        # Summary
        summary = await sm.summarize_reviews(comments)
        print(f"\nSummary: {summary}")

asyncio.run(analyze_youtube())
```

## Reddit Discussion Analysis

```python
async def analyze_reddit():
    async with Sentimatrix(config) as sm:
        # Scrape discussion
        comments = await sm.scrape_reviews(
            "https://reddit.com/r/gaming/comments/...",
            platform="reddit",
            max_reviews=100
        )

        # Analyze emotions
        for comment in comments[:10]:
            emotion = await sm.detect_emotions(comment.text)
            print(f"u/{comment.author}: {emotion.primary}")
            print(f"  Score: {comment.helpful_count}")

asyncio.run(analyze_reddit())
```

## Brand Monitoring

```python
async def monitor_brand():
    async with Sentimatrix(config) as sm:
        sources = [
            ("youtube", "https://youtube.com/..."),
            ("reddit", "https://reddit.com/r/brand/..."),
            ("trustpilot", "https://trustpilot.com/review/brand.com"),
        ]

        all_reviews = []
        for platform, url in sources:
            reviews = await sm.scrape_reviews(url, platform=platform)
            all_reviews.extend(reviews)

        # Aggregate analysis
        results = await sm.analyze_batch([r.text for r in all_reviews])

        # Timeline
        timeline = await sm.analyze_temporal([
            {"text": r.text, "date": r.posted_date.isoformat()}
            for r in all_reviews if r.posted_date
        ])

        print("Sentiment over time:")
        for period in timeline:
            print(f"  {period.date}: {period.sentiment}")

asyncio.run(monitor_brand())
```

## Related

- [Basic Examples](basic.md)
- [Review Aggregation](review-aggregation.md)
