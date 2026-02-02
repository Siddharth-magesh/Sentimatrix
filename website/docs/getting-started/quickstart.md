---
title: Quick Start
description: Get started with Sentimatrix in under 5 minutes
---

# Quick Start

This guide will have you running sentiment analysis in under 5 minutes.

## Basic Usage

Sentimatrix uses an async context manager pattern for resource management:

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        result = await sm.analyze("I absolutely love this product!")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.2%}")

asyncio.run(main())
```

Output:

```
Sentiment: positive
Confidence: 96.78%
```

## Sentiment Analysis

### Single Text Analysis

```python
async with Sentimatrix() as sm:
    result = await sm.analyze("The customer service was terrible.")

    print(f"Sentiment: {result.sentiment}")      # negative
    print(f"Confidence: {result.confidence}")    # 0.9234
    print(f"Scores: {result.scores}")            # {'positive': 0.05, 'negative': 0.92, 'neutral': 0.03}
```

### Batch Analysis

Analyze multiple texts efficiently:

```python
texts = [
    "Great product, highly recommend!",
    "Okay, nothing special.",
    "Complete waste of money.",
    "Exceeded all my expectations!",
]

async with Sentimatrix() as sm:
    results = await sm.analyze_batch(texts)

    for text, result in zip(texts, results):
        print(f"{result.sentiment:>10} ({result.confidence:.0%}): {text[:40]}")
```

Output:

```
  positive (97%): Great product, highly recommend!
   neutral (84%): Okay, nothing special.
  negative (95%): Complete waste of money.
  positive (98%): Exceeded all my expectations!
```

## Emotion Detection

Detect emotions using multiple taxonomies:

```python
async with Sentimatrix() as sm:
    # Detect emotions
    emotions = await sm.detect_emotions("I'm so excited about this announcement!")

    print(f"Primary emotion: {emotions.primary}")     # joy
    print(f"Confidence: {emotions.confidence:.2%}")   # 89.45%
    print(f"All emotions: {emotions.scores}")
```

Output:

```
Primary emotion: joy
Confidence: 89.45%
All emotions: {'joy': 0.89, 'surprise': 0.15, 'anticipation': 0.12, ...}
```

### Multi-Label Emotion Detection

Detect multiple emotions above a threshold:

```python
async with Sentimatrix() as sm:
    emotions = await sm.detect_emotions(
        "I'm happy but also a bit nervous about the interview.",
        mode="multi_label",
        threshold=0.3
    )

    print(f"Detected: {emotions.labels}")  # ['joy', 'fear', 'anticipation']
```

## Combined Analysis

Run sentiment and emotion analysis together:

```python
async with Sentimatrix() as sm:
    result = await sm.analyze_full("This movie was absolutely breathtaking!")

    print(f"Sentiment: {result.sentiment.label}")
    print(f"Emotions: {result.emotions.top_k(3)}")
```

## Web Scraping

### Scrape Reviews from Steam

Steam reviews don't require browser automation:

```python
async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        url="https://store.steampowered.com/app/1245620/ELDEN_RING/",
        platform="steam",
        max_reviews=50
    )

    print(f"Scraped {len(reviews)} reviews")

    for review in reviews[:3]:
        print(f"- {review.text[:80]}...")
        print(f"  Rating: {review.rating}, Helpful: {review.helpful_count}")
```

### Scrape Reviews from Amazon

Amazon requires Playwright for JavaScript rendering:

```python
async with Sentimatrix() as sm:
    reviews = await sm.scrape_reviews(
        url="https://www.amazon.com/dp/B0BSHF7WHW",
        platform="amazon",
        max_reviews=30,
        use_browser=True  # Enables Playwright
    )

    for review in reviews:
        print(f"[{review.rating}] {review.title}: {review.text[:60]}...")
```

!!! info "Playwright Required"
    For Amazon and some other platforms, install Playwright:
    ```bash
    pip install sentimatrix[scraping]
    playwright install chromium
    ```

## LLM-Powered Analysis

Enhance analysis with LLM providers:

### Setup with Groq (Free Tier)

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        api_key="your-groq-api-key",  # or use GROQ_API_KEY env var
        model="llama-3.3-70b-versatile"
    )
)

async with Sentimatrix(config) as sm:
    # Standard analysis still works
    result = await sm.analyze("Great product!")

    # LLM-powered summarization
    summary = await sm.summarize_reviews(reviews)
    print(summary)
```

### Generate Insights

```python
async with Sentimatrix(config) as sm:
    reviews = await sm.scrape_reviews(url, platform="steam", max_reviews=100)

    # Generate insights using LLM
    insights = await sm.generate_insights(reviews)

    print("Pros:")
    for pro in insights.pros:
        print(f"  + {pro}")

    print("\nCons:")
    for con in insights.cons:
        print(f"  - {con}")

    print(f"\nRecommendation: {insights.recommendation}")
```

## Configuration Options

### Using YAML Configuration

Create a `sentimatrix.yaml` file:

```yaml title="sentimatrix.yaml"
llm:
  provider: groq
  model: llama-3.3-70b-versatile

scraper:
  rate_limit:
    requests_per_second: 2
  retry:
    max_retries: 3

logging:
  level: INFO
```

Load it automatically:

```python
async with Sentimatrix() as sm:  # Auto-loads sentimatrix.yaml
    result = await sm.analyze("Hello!")
```

### Using Environment Variables

```bash
export SENTIMATRIX_LLM_PROVIDER=groq
export SENTIMATRIX_LLM_MODEL=llama-3.3-70b-versatile
export GROQ_API_KEY=gsk_...
```

### Using Python Objects

```python
from sentimatrix.config import (
    SentimatrixConfig,
    LLMConfig,
    ScraperConfig,
    RateLimitConfig,
)

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
    ),
    scraper=ScraperConfig(
        rate_limit=RateLimitConfig(
            requests_per_second=5,
            burst_size=10,
        ),
        use_browser=True,
    ),
)

async with Sentimatrix(config) as sm:
    # Your code here
    pass
```

## CLI Usage

Sentimatrix includes a command-line interface:

```bash
# Analyze text directly
sentimatrix analyze "This is a great product!"

# Analyze from file
sentimatrix analyze --file reviews.txt

# Scrape and analyze
sentimatrix scrape https://store.steampowered.com/app/1245620 --platform steam

# Get help
sentimatrix --help
```

## Error Handling

Handle common errors gracefully:

```python
from sentimatrix.exceptions import (
    ScraperError,
    ProviderError,
    RateLimitError,
    ConfigurationError,
)

async with Sentimatrix(config) as sm:
    try:
        reviews = await sm.scrape_reviews(url, platform="amazon")
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after}s")
    except ScraperError as e:
        print(f"Scraping failed: {e}")

    try:
        summary = await sm.summarize_reviews(reviews)
    except ProviderError as e:
        print(f"LLM error: {e}")
```

## Next Steps

- **[First Analysis](first-analysis.md)** - Build a complete analysis pipeline
- **[LLM Providers](../providers/)** - Configure LLM providers
- **[Scrapers](../scrapers/)** - Learn about platform scrapers
- **[Configuration](../configuration/)** - Deep dive into configuration
