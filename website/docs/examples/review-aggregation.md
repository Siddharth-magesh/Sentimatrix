---
title: Review Aggregation
description: Aggregate reviews from multiple platforms
---

# Review Aggregation

Examples for aggregating and analyzing reviews from multiple platforms.

## Multi-Platform Aggregation

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def aggregate_reviews():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        # Scrape from multiple platforms
        platforms = {
            "amazon": "https://amazon.com/dp/...",
            "steam": "https://store.steampowered.com/app/...",
            "youtube": "https://youtube.com/watch?v=...",
        }

        all_reviews = []
        for platform, url in platforms.items():
            reviews = await sm.scrape_reviews(
                url,
                platform=platform,
                max_reviews=50
            )
            all_reviews.extend(reviews)

        print(f"Total reviews: {len(all_reviews)}")

        # Unified analysis
        results = await sm.analyze_batch([r.text for r in all_reviews])

        # Per-platform stats
        from collections import defaultdict
        stats = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

        for review, result in zip(all_reviews, results):
            stats[review.platform][result.sentiment] += 1

        for platform, data in stats.items():
            print(f"\n{platform}:")
            total = sum(data.values())
            for sentiment, count in data.items():
                print(f"  {sentiment}: {count/total:.1%}")

        # Combined insights
        insights = await sm.generate_insights(all_reviews)
        print("\nOverall Insights:")
        print("PROS:", insights.pros)
        print("CONS:", insights.cons)

asyncio.run(aggregate_reviews())
```

## Restaurant Review Aggregation

```python
async def aggregate_restaurant():
    async with Sentimatrix(config) as sm:
        # Multiple review platforms
        sources = {
            "yelp": "https://yelp.com/biz/...",
            "google_reviews": "https://maps.google.com/...",
            "trustpilot": "https://trustpilot.com/review/...",
        }

        all_reviews = []
        for platform, url in sources.items():
            reviews = await sm.scrape_reviews(url, platform=platform)
            all_reviews.extend(reviews)

        # Aspect analysis
        aspects = ["food", "service", "ambiance", "price", "cleanliness"]

        aspect_scores = {a: [] for a in aspects}
        for review in all_reviews:
            result = await sm.analyze_aspects(review.text, aspects=aspects)
            for aspect, sentiment in result.items():
                score = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
                aspect_scores[aspect].append(score)

        print("Aspect Scores:")
        for aspect, scores in aspect_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {aspect}: {avg:+.2f}")

asyncio.run(aggregate_restaurant())
```

## Related

- [E-Commerce](ecommerce.md)
- [Social Media](social-media.md)
