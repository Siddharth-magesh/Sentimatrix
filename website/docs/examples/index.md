---
title: Examples
description: Real-world examples and use cases for Sentimatrix
---

# Examples

Practical examples showing how to use Sentimatrix for real-world scenarios.

## Example Categories

<div class="grid">

<div class="card">
<h3>:material-rocket-launch: Basic Usage</h3>
<p>Getting started with sentiment analysis and emotion detection.</p>
<p><a href="basic/">View examples →</a></p>
</div>

<div class="card">
<h3>:material-cart: E-commerce Analysis</h3>
<p>Analyze product reviews from Amazon and other platforms.</p>
<p><a href="ecommerce/">View examples →</a></p>
</div>

<div class="card">
<h3>:material-forum: Social Media</h3>
<p>Monitor sentiment on Reddit, YouTube, and social platforms.</p>
<p><a href="social-media/">View examples →</a></p>
</div>

<div class="card">
<h3>:material-chart-line: Review Aggregation</h3>
<p>Aggregate and analyze reviews across multiple sources.</p>
<p><a href="review-aggregation/">View examples →</a></p>
</div>

</div>

## Quick Examples

### Basic Sentiment Analysis

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Single analysis
        result = await sm.analyze("This product is amazing!")
        print(f"Sentiment: {result.sentiment} ({result.confidence:.0%})")

        # Batch analysis
        texts = [
            "Great quality!",
            "Terrible experience",
            "It's okay",
        ]
        results = await sm.analyze_batch(texts)

        for text, result in zip(texts, results):
            print(f"{result.sentiment:>10}: {text}")

asyncio.run(main())
```

### Steam Game Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def analyze_game(app_id: str):
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        # Scrape reviews
        reviews = await sm.scrape_reviews(
            url=f"https://store.steampowered.com/app/{app_id}",
            platform="steam",
            max_reviews=100
        )

        # Analyze sentiments
        results = await sm.analyze_batch([r.text for r in reviews])

        # Calculate distribution
        positive = sum(1 for r in results if r.sentiment == "positive")
        total = len(results)

        print(f"Positive: {positive/total*100:.1f}%")

        # Generate summary
        summary = await sm.summarize_reviews(
            [{"text": r.text} for r in reviews[:50]]
        )
        print(f"\nSummary:\n{summary}")

asyncio.run(analyze_game("1245620"))  # Elden Ring
```

### Amazon Product Monitoring

```python
import asyncio
from sentimatrix import Sentimatrix

async def monitor_product(asin: str):
    async with Sentimatrix() as sm:
        reviews = await sm.scrape_reviews(
            url=asin,
            platform="amazon",
            max_reviews=50,
            use_browser=True
        )

        # Rating distribution
        ratings = {}
        for review in reviews:
            ratings[review.rating] = ratings.get(review.rating, 0) + 1

        print("Rating Distribution:")
        for rating in sorted(ratings.keys(), reverse=True):
            count = ratings[rating]
            bar = "=" * (count * 2)
            print(f"{rating}: {bar} ({count})")

        # Analyze verified purchases only
        verified = [r for r in reviews if r.verified_purchase]
        if verified:
            results = await sm.analyze_batch([r.text for r in verified])
            positive = sum(1 for r in results if r.sentiment == "positive")
            print(f"\nVerified purchase sentiment: {positive/len(verified)*100:.0f}% positive")

asyncio.run(monitor_product("B0BSHF7WHW"))
```

### Multi-Platform Comparison

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def compare_platforms():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        platforms = {
            "steam": "https://store.steampowered.com/app/1245620",
            "reddit": "https://reddit.com/r/Eldenring",
        }

        results = {}
        for platform, url in platforms.items():
            reviews = await sm.scrape_reviews(
                url=url,
                platform=platform,
                max_reviews=50
            )
            analysis = await sm.analyze_batch([r.text for r in reviews])

            positive = sum(1 for r in analysis if r.sentiment == "positive")
            results[platform] = positive / len(analysis) * 100

        print("Sentiment by Platform:")
        for platform, pct in sorted(results.items(), key=lambda x: -x[1]):
            print(f"  {platform}: {pct:.1f}% positive")

asyncio.run(compare_platforms())
```

## Featured Examples

| Example | Difficulty | Features Used |
|---------|------------|---------------|
| [Basic Analysis](basic.md) | Beginner | Sentiment, Batch |
| [E-commerce Analysis](ecommerce.md) | Intermediate | Scraping, Aspects |
| [Social Media](social-media.md) | Intermediate | Reddit, YouTube |
| [Review Aggregation](review-aggregation.md) | Advanced | Multi-platform, LLM |
| [Real-time Analysis](realtime.md) | Advanced | Streaming, Webhooks |
| [Batch Processing](batch.md) | Advanced | Large datasets |

## Running Examples

Clone the examples repository:

```bash
git clone https://github.com/sentimatrix/sentimatrix-examples.git
cd sentimatrix-examples
pip install -e .
```

Run an example:

```bash
python examples/basic_analysis.py
```

## Contributing Examples

We welcome example contributions! See our [contribution guide](../contributing/index.md).
