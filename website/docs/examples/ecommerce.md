---
title: E-Commerce Examples
description: Sentiment analysis for e-commerce platforms
---

# E-Commerce Examples

Examples for analyzing product reviews from e-commerce platforms.

## Amazon Product Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def analyze_amazon_product():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        # Scrape reviews
        reviews = await sm.scrape_reviews(
            "https://amazon.com/dp/B08N5WRWNW",
            platform="amazon",
            max_reviews=100
        )

        print(f"Scraped {len(reviews)} reviews")

        # Analyze sentiment
        results = await sm.analyze_batch([r.text for r in reviews])

        # Stats
        positive = sum(1 for r in results if r.sentiment == "positive")
        negative = sum(1 for r in results if r.sentiment == "negative")

        print(f"Positive: {positive}, Negative: {negative}")

        # Generate insights
        insights = await sm.generate_insights(reviews)
        print("\nPROS:", insights.pros)
        print("CONS:", insights.cons)

asyncio.run(analyze_amazon_product())
```

## Aspect-Based Product Analysis

```python
async def aspect_analysis():
    async with Sentimatrix(config) as sm:
        reviews = await sm.scrape_reviews(url, platform="amazon")

        aspects = ["quality", "price", "shipping", "packaging"]

        for review in reviews[:10]:
            result = await sm.analyze_aspects(review.text, aspects=aspects)
            print(f"Review: {review.text[:50]}...")
            for aspect, sentiment in result.items():
                print(f"  {aspect}: {sentiment}")

asyncio.run(aspect_analysis())
```

## Product Comparison

```python
async def compare_products():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        # Scrape both products
        product_a = await sm.scrape_reviews(url_a, platform="amazon")
        product_b = await sm.scrape_reviews(url_b, platform="amazon")

        # Compare
        comparison = await sm.compare_products(
            product_a,
            product_b,
            product_a_name="Product A",
            product_b_name="Product B"
        )

        print(comparison.comparison_summary)
        print(f"Recommended: {comparison.winner}")

asyncio.run(compare_products())
```

## Related

- [Basic Examples](basic.md)
- [Review Aggregation](review-aggregation.md)
