---
title: Real-Time Analysis
description: Real-time sentiment analysis examples
---

# Real-Time Analysis

Examples for real-time sentiment analysis with streaming.

## Streaming Summaries

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def streaming_summary():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
    )

    async with Sentimatrix(config) as sm:
        reviews = await sm.scrape_reviews(url, platform="amazon")

        print("Generating summary...")
        async for chunk in sm.stream_summary(reviews):
            print(chunk, end="", flush=True)
        print()

asyncio.run(streaming_summary())
```

## Live Analysis Pipeline

```python
async def live_analysis():
    config = SentimatrixConfig(
        llm=LLMConfig(provider="cerebras")  # Fastest inference
    )

    async with Sentimatrix(config) as sm:
        # Simulated live feed
        async def get_live_reviews():
            while True:
                # In practice, this would be a real feed
                yield "Customer message: Great product!"
                await asyncio.sleep(1)

        async for text in get_live_reviews():
            result = await sm.analyze(text)
            emotion = await sm.detect_emotions(text)

            print(f"Sentiment: {result.sentiment}")
            print(f"Emotion: {emotion.primary}")
            print(f"Confidence: {result.confidence:.2%}")
            print("---")

asyncio.run(live_analysis())
```

## Concurrent Processing

```python
async def concurrent_analysis():
    async with Sentimatrix(config) as sm:
        urls = [
            "https://amazon.com/dp/ASIN1",
            "https://amazon.com/dp/ASIN2",
            "https://amazon.com/dp/ASIN3",
        ]

        # Scrape concurrently
        tasks = [
            sm.scrape_reviews(url, platform="amazon")
            for url in urls
        ]
        all_reviews = await asyncio.gather(*tasks)

        # Analyze all
        for reviews in all_reviews:
            results = await sm.analyze_batch([r.text for r in reviews])
            positive = sum(1 for r in results if r.sentiment == "positive")
            print(f"Product: {positive/len(results):.1%} positive")

asyncio.run(concurrent_analysis())
```

## Related

- [Batch Processing](batch.md)
- [Basic Examples](basic.md)
