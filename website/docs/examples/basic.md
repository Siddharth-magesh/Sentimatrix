---
title: Basic Examples
description: Simple getting started examples
---

# Basic Examples

Simple examples to get started with Sentimatrix.

## Sentiment Analysis

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Single text
        result = await sm.analyze("This product is amazing!")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.2%}")

asyncio.run(main())
```

## Batch Analysis

```python
async def batch_example():
    async with Sentimatrix() as sm:
        texts = [
            "Great quality!",
            "Terrible experience",
            "It's okay, nothing special",
        ]

        results = await sm.analyze_batch(texts)

        for text, result in zip(texts, results):
            print(f"{result.sentiment:>10}: {text}")

asyncio.run(batch_example())
```

## Emotion Detection

```python
async def emotion_example():
    async with Sentimatrix() as sm:
        result = await sm.detect_emotions(
            "I'm so excited about this purchase!"
        )

        print(f"Primary: {result.primary}")
        print(f"Top 3: {result.top_k(3)}")

asyncio.run(emotion_example())
```

## With LLM Summary

```python
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def llm_example():
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            model="llama-3.3-70b-versatile"
        )
    )

    async with Sentimatrix(config) as sm:
        reviews = [
            {"text": "Great product!", "rating": 5},
            {"text": "Good but expensive", "rating": 4},
            {"text": "Not worth it", "rating": 2},
        ]

        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(llm_example())
```

## Related

- [E-Commerce](ecommerce.md)
- [Social Media](social-media.md)
