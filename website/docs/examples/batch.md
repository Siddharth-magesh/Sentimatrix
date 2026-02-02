---
title: Batch Processing
description: High-volume batch processing examples
---

# Batch Processing

Examples for processing large volumes of data efficiently.

## Large Batch Analysis

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, ModelConfig

async def large_batch():
    config = SentimatrixConfig(
        models=ModelConfig(
            device="cuda",    # GPU for speed
            batch_size=64     # Larger batches
        )
    )

    async with Sentimatrix(config) as sm:
        # Load large dataset
        texts = load_texts()  # 10,000+ texts
        print(f"Processing {len(texts)} texts...")

        # Process in batches
        results = await sm.analyze_batch(
            texts,
            batch_size=64
        )

        # Stats
        positive = sum(1 for r in results if r.sentiment == "positive")
        print(f"Positive: {positive/len(results):.1%}")

asyncio.run(large_batch())
```

## Parallel Scraping

```python
async def parallel_scraping():
    async with Sentimatrix(config) as sm:
        # Multiple products
        products = [
            ("Product A", "https://amazon.com/dp/ASIN1"),
            ("Product B", "https://amazon.com/dp/ASIN2"),
            ("Product C", "https://amazon.com/dp/ASIN3"),
        ]

        # Scrape all in parallel
        async def scrape_product(name, url):
            reviews = await sm.scrape_reviews(url, platform="amazon")
            results = await sm.analyze_batch([r.text for r in reviews])
            positive = sum(1 for r in results if r.sentiment == "positive")
            return name, len(reviews), positive / len(results)

        tasks = [scrape_product(name, url) for name, url in products]
        results = await asyncio.gather(*tasks)

        for name, count, positive_ratio in results:
            print(f"{name}: {count} reviews, {positive_ratio:.1%} positive")

asyncio.run(parallel_scraping())
```

## Incremental Processing

```python
async def incremental_processing():
    async with Sentimatrix(config) as sm:
        # Process in chunks to manage memory
        chunk_size = 1000
        all_texts = load_large_dataset()  # 100,000 texts

        total_positive = 0
        total_count = 0

        for i in range(0, len(all_texts), chunk_size):
            chunk = all_texts[i:i + chunk_size]
            results = await sm.analyze_batch(chunk)

            positive = sum(1 for r in results if r.sentiment == "positive")
            total_positive += positive
            total_count += len(results)

            print(f"Processed {total_count}/{len(all_texts)}")

        print(f"Final: {total_positive/total_count:.1%} positive")

asyncio.run(incremental_processing())
```

## CSV Export

```python
import csv

async def export_to_csv():
    async with Sentimatrix(config) as sm:
        reviews = await sm.scrape_reviews(url, platform="amazon")
        results = await sm.analyze_batch([r.text for r in reviews])
        emotions = await sm.detect_emotions_batch([r.text for r in reviews])

        with open("analysis.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "rating", "sentiment", "confidence", "emotion"])

            for review, result, emotion in zip(reviews, results, emotions):
                writer.writerow([
                    review.text[:200],
                    review.rating,
                    result.sentiment,
                    f"{result.confidence:.2f}",
                    emotion.primary
                ])

        print("Exported to analysis.csv")

asyncio.run(export_to_csv())
```

## Related

- [Real-Time](realtime.md)
- [Basic Examples](basic.md)
