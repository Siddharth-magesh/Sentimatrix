---
title: First Analysis
description: Build your first complete sentiment analysis pipeline
---

# Your First Analysis Pipeline

This guide walks you through building a complete sentiment analysis pipeline that scrapes reviews, analyzes them, and generates insights.

## What We'll Build

A pipeline that:

1. Scrapes reviews from Steam
2. Analyzes sentiment for each review
3. Detects emotions
4. Generates an LLM-powered summary
5. Exports results to JSON

## Prerequisites

```bash
# Install with required extras
pip install sentimatrix[llm,scraping]

# Get a free Groq API key from https://console.groq.com
export GROQ_API_KEY="gsk_..."
```

## Step 1: Basic Setup

Create a new file `analyze_game.py`:

```python
import asyncio
import json
from datetime import datetime
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

# Configuration
CONFIG = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile"
    )
)

# Game to analyze
GAME_URL = "https://store.steampowered.com/app/1245620/ELDEN_RING/"
MAX_REVIEWS = 100
```

## Step 2: Scrape Reviews

Add the scraping function:

```python
async def scrape_game_reviews(sm: Sentimatrix) -> list:
    """Scrape reviews from Steam."""
    print(f"Scraping reviews from Steam...")

    reviews = await sm.scrape_reviews(
        url=GAME_URL,
        platform="steam",
        max_reviews=MAX_REVIEWS
    )

    print(f"Scraped {len(reviews)} reviews")
    return reviews
```

## Step 3: Analyze Sentiment

Add sentiment analysis:

```python
async def analyze_sentiments(sm: Sentimatrix, reviews: list) -> list:
    """Analyze sentiment for all reviews."""
    print("Analyzing sentiments...")

    # Extract review texts
    texts = [r.text for r in reviews]

    # Batch analysis for efficiency
    results = await sm.analyze_batch(texts)

    # Combine reviews with results
    analyzed = []
    for review, result in zip(reviews, results):
        analyzed.append({
            "text": review.text,
            "rating": review.rating,
            "helpful_count": review.helpful_count,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "scores": result.scores,
        })

    return analyzed
```

## Step 4: Detect Emotions

Add emotion detection:

```python
async def detect_emotions(sm: Sentimatrix, reviews: list) -> list:
    """Detect emotions for reviews with strong sentiments."""
    print("Detecting emotions...")

    # Only analyze reviews with high confidence
    strong_reviews = [r for r in reviews if r["confidence"] > 0.8]

    for review in strong_reviews:
        emotions = await sm.detect_emotions(review["text"])
        review["emotions"] = {
            "primary": emotions.primary,
            "scores": emotions.scores,
        }

    return reviews
```

## Step 5: Calculate Statistics

Add statistics calculation:

```python
def calculate_stats(reviews: list) -> dict:
    """Calculate aggregate statistics."""
    print("Calculating statistics...")

    total = len(reviews)
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}

    for r in reviews:
        sentiments[r["sentiment"]] += 1

    avg_confidence = sum(r["confidence"] for r in reviews) / total

    # Find most helpful positive and negative reviews
    positive_reviews = [r for r in reviews if r["sentiment"] == "positive"]
    negative_reviews = [r for r in reviews if r["sentiment"] == "negative"]

    most_helpful_positive = max(
        positive_reviews,
        key=lambda x: x.get("helpful_count", 0),
        default=None
    )
    most_helpful_negative = max(
        negative_reviews,
        key=lambda x: x.get("helpful_count", 0),
        default=None
    )

    return {
        "total_reviews": total,
        "sentiment_distribution": sentiments,
        "sentiment_percentages": {
            k: f"{v/total*100:.1f}%" for k, v in sentiments.items()
        },
        "average_confidence": f"{avg_confidence:.2%}",
        "most_helpful_positive": most_helpful_positive["text"][:200] if most_helpful_positive else None,
        "most_helpful_negative": most_helpful_negative["text"][:200] if most_helpful_negative else None,
    }
```

## Step 6: Generate LLM Summary

Add LLM-powered summarization:

```python
async def generate_summary(sm: Sentimatrix, reviews: list) -> dict:
    """Generate an LLM-powered summary of reviews."""
    print("Generating AI summary...")

    # Get positive and negative samples
    positive = [r["text"] for r in reviews if r["sentiment"] == "positive"][:10]
    negative = [r["text"] for r in reviews if r["sentiment"] == "negative"][:10]

    # Generate summary
    summary = await sm.summarize_reviews(
        reviews=[{"text": r["text"]} for r in reviews[:50]],
        style="professional"
    )

    # Generate pros and cons
    insights = await sm.generate_insights(
        reviews=[{"text": r["text"]} for r in reviews[:50]]
    )

    return {
        "summary": summary,
        "pros": insights.pros[:5],
        "cons": insights.cons[:5],
        "recommendation": insights.recommendation,
    }
```

## Step 7: Export Results

Add export functionality:

```python
def export_results(reviews: list, stats: dict, summary: dict) -> str:
    """Export analysis results to JSON."""
    output = {
        "metadata": {
            "url": GAME_URL,
            "analyzed_at": datetime.now().isoformat(),
            "total_reviews": len(reviews),
        },
        "statistics": stats,
        "ai_summary": summary,
        "reviews": reviews[:20],  # Include top 20 reviews
    }

    filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results exported to {filename}")
    return filename
```

## Step 8: Main Pipeline

Tie it all together:

```python
async def main():
    """Run the complete analysis pipeline."""
    print("=" * 60)
    print("Sentimatrix Game Review Analysis Pipeline")
    print("=" * 60)

    async with Sentimatrix(CONFIG) as sm:
        # Step 1: Scrape reviews
        reviews = await scrape_game_reviews(sm)

        # Step 2: Analyze sentiments
        analyzed = await analyze_sentiments(sm, reviews)

        # Step 3: Detect emotions
        analyzed = await detect_emotions(sm, analyzed)

        # Step 4: Calculate statistics
        stats = calculate_stats(analyzed)

        # Step 5: Generate AI summary
        summary = await generate_summary(sm, analyzed)

        # Step 6: Export results
        filename = export_results(analyzed, stats, summary)

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\nTotal Reviews: {stats['total_reviews']}")
        print(f"Sentiment Distribution:")
        for sentiment, pct in stats['sentiment_percentages'].items():
            print(f"  {sentiment:>10}: {pct}")
        print(f"\nAI Summary:\n{summary['summary'][:500]}...")
        print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Complete Script

Here's the full script:

??? example "analyze_game.py (Complete)"

    ```python
    #!/usr/bin/env python3
    """
    Sentimatrix Game Review Analysis Pipeline

    Scrapes Steam reviews, analyzes sentiment, detects emotions,
    and generates AI-powered insights.

    Usage:
        python analyze_game.py
    """

    import asyncio
    import json
    from datetime import datetime
    from sentimatrix import Sentimatrix
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    # Configuration
    CONFIG = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
    )

    GAME_URL = "https://store.steampowered.com/app/1245620/ELDEN_RING/"
    MAX_REVIEWS = 100


    async def scrape_game_reviews(sm: Sentimatrix) -> list:
        print(f"Scraping reviews from Steam...")
        reviews = await sm.scrape_reviews(
            url=GAME_URL,
            platform="steam",
            max_reviews=MAX_REVIEWS
        )
        print(f"Scraped {len(reviews)} reviews")
        return reviews


    async def analyze_sentiments(sm: Sentimatrix, reviews: list) -> list:
        print("Analyzing sentiments...")
        texts = [r.text for r in reviews]
        results = await sm.analyze_batch(texts)

        analyzed = []
        for review, result in zip(reviews, results):
            analyzed.append({
                "text": review.text,
                "rating": review.rating,
                "helpful_count": review.helpful_count,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "scores": result.scores,
            })
        return analyzed


    async def detect_emotions(sm: Sentimatrix, reviews: list) -> list:
        print("Detecting emotions...")
        strong_reviews = [r for r in reviews if r["confidence"] > 0.8]

        for review in strong_reviews:
            emotions = await sm.detect_emotions(review["text"])
            review["emotions"] = {
                "primary": emotions.primary,
                "scores": emotions.scores,
            }
        return reviews


    def calculate_stats(reviews: list) -> dict:
        print("Calculating statistics...")
        total = len(reviews)
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}

        for r in reviews:
            sentiments[r["sentiment"]] += 1

        avg_confidence = sum(r["confidence"] for r in reviews) / total

        positive_reviews = [r for r in reviews if r["sentiment"] == "positive"]
        negative_reviews = [r for r in reviews if r["sentiment"] == "negative"]

        most_helpful_positive = max(
            positive_reviews, key=lambda x: x.get("helpful_count", 0), default=None
        )
        most_helpful_negative = max(
            negative_reviews, key=lambda x: x.get("helpful_count", 0), default=None
        )

        return {
            "total_reviews": total,
            "sentiment_distribution": sentiments,
            "sentiment_percentages": {
                k: f"{v/total*100:.1f}%" for k, v in sentiments.items()
            },
            "average_confidence": f"{avg_confidence:.2%}",
            "most_helpful_positive": most_helpful_positive["text"][:200] if most_helpful_positive else None,
            "most_helpful_negative": most_helpful_negative["text"][:200] if most_helpful_negative else None,
        }


    async def generate_summary(sm: Sentimatrix, reviews: list) -> dict:
        print("Generating AI summary...")

        summary = await sm.summarize_reviews(
            reviews=[{"text": r["text"]} for r in reviews[:50]],
            style="professional"
        )

        insights = await sm.generate_insights(
            reviews=[{"text": r["text"]} for r in reviews[:50]]
        )

        return {
            "summary": summary,
            "pros": insights.pros[:5],
            "cons": insights.cons[:5],
            "recommendation": insights.recommendation,
        }


    def export_results(reviews: list, stats: dict, summary: dict) -> str:
        output = {
            "metadata": {
                "url": GAME_URL,
                "analyzed_at": datetime.now().isoformat(),
                "total_reviews": len(reviews),
            },
            "statistics": stats,
            "ai_summary": summary,
            "reviews": reviews[:20],
        }

        filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results exported to {filename}")
        return filename


    async def main():
        print("=" * 60)
        print("Sentimatrix Game Review Analysis Pipeline")
        print("=" * 60)

        async with Sentimatrix(CONFIG) as sm:
            reviews = await scrape_game_reviews(sm)
            analyzed = await analyze_sentiments(sm, reviews)
            analyzed = await detect_emotions(sm, analyzed)
            stats = calculate_stats(analyzed)
            summary = await generate_summary(sm, analyzed)
            filename = export_results(analyzed, stats, summary)

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"\nTotal Reviews: {stats['total_reviews']}")
            print(f"Sentiment Distribution:")
            for sentiment, pct in stats['sentiment_percentages'].items():
                print(f"  {sentiment:>10}: {pct}")
            print(f"\nResults saved to: {filename}")


    if __name__ == "__main__":
        asyncio.run(main())
    ```

## Running the Pipeline

```bash
python analyze_game.py
```

Expected output:

```
============================================================
Sentimatrix Game Review Analysis Pipeline
============================================================
Scraping reviews from Steam...
Scraped 100 reviews
Analyzing sentiments...
Detecting emotions...
Calculating statistics...
Generating AI summary...
Results exported to analysis_20250129_143022.json

============================================================
ANALYSIS COMPLETE
============================================================

Total Reviews: 100
Sentiment Distribution:
  positive: 72.0%
  negative: 18.0%
   neutral: 10.0%

Results saved to: analysis_20250129_143022.json
```

## Next Steps

- **[LLM Providers](../providers/)** - Try different LLM providers
- **[Platform Scrapers](../scrapers/)** - Scrape from other platforms
- **[Configuration](../configuration/)** - Customize your setup
- **[Examples](../examples/)** - More example pipelines
