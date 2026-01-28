# Sentimatrix V2 Quickstart Guide

Get started with Sentimatrix V2 in minutes.

## Installation

```bash
pip install sentimatrix
```

Or install from source:

```bash
git clone https://github.com/your-org/sentimatrix.git
cd sentimatrix
pip install -e .
```

## Basic Usage

### Simple Sentiment Analysis

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    analyzer = SentimentAnalyzer()

    # Analyze a single text
    result = await analyzer.analyze("This product is amazing!")

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2%}")

asyncio.run(main())
```

Output:
```
Sentiment: POSITIVE
Confidence: 95.00%
```

### Batch Analysis

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    analyzer = SentimentAnalyzer()

    texts = [
        "Great product! Highly recommend.",
        "Terrible experience, never buying again.",
        "It's okay, nothing special.",
    ]

    results = await analyzer.analyze_batch(texts)

    for text, result in zip(texts, results):
        print(f"{text[:30]}... -> {result.sentiment}")

asyncio.run(main())
```

### Emotion Detection

```python
import asyncio
from sentimatrix.analysis.emotion import EmotionDetector

async def main():
    detector = EmotionDetector()

    result = await detector.detect("I'm so happy about this purchase!")

    print(f"Primary emotion: {result.primary_emotion}")
    for emotion in result.emotions:
        print(f"  {emotion.label}: {emotion.score:.2%}")

asyncio.run(main())
```

## Using Pipelines

Pipelines allow you to chain multiple analysis steps together.

```python
import asyncio
from sentimatrix.core.pipeline import Pipeline, FunctionStep, PipelineContext

async def main():
    pipeline = Pipeline(name="review_analysis")

    # Step 1: Fetch reviews
    async def fetch_reviews(ctx: PipelineContext, _prev):
        return [
            {"text": "Great product!", "rating": 5},
            {"text": "Not worth it.", "rating": 2},
        ]

    # Step 2: Analyze sentiment
    async def analyze(ctx: PipelineContext, reviews):
        results = []
        for review in reviews:
            sentiment = "positive" if review["rating"] >= 4 else "negative"
            results.append({**review, "sentiment": sentiment})
        return results

    # Step 3: Aggregate
    async def aggregate(ctx: PipelineContext, results):
        positive = sum(1 for r in results if r["sentiment"] == "positive")
        return {"total": len(results), "positive": positive}

    pipeline.add_step(FunctionStep("fetch", fetch_reviews))
    pipeline.add_step(FunctionStep("analyze", analyze))
    pipeline.add_step(FunctionStep("aggregate", aggregate))

    result = await pipeline.run()

    if result.success:
        print(f"Analyzed {result.output['total']} reviews")
        print(f"Positive: {result.output['positive']}")

asyncio.run(main())
```

## Web Scraping Reviews

### Amazon Reviews

```python
import asyncio
from sentimatrix.providers.scrapers import AmazonScraper
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    scraper = AmazonScraper()
    analyzer = SentimentAnalyzer()

    # Scrape reviews
    reviews = await scraper.scrape("B08N5WRWNW", max_reviews=50)

    # Analyze each review
    for review in reviews:
        result = await analyzer.analyze(review.text)
        print(f"Rating: {review.rating} | Sentiment: {result.sentiment}")

asyncio.run(main())
```

### With Rate Limiting

```python
import asyncio
from sentimatrix.providers.scrapers import AmazonScraper
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy

async def main():
    limiter = RateLimiter(
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        requests_per_second=2.0,
        burst_size=5
    )

    scraper = AmazonScraper(rate_limiter=limiter)
    reviews = await scraper.scrape("B08N5WRWNW", max_reviews=100)

    print(f"Scraped {len(reviews)} reviews")

asyncio.run(main())
```

## Exporting Results

### JSON Export

```python
from sentimatrix.output import JSONFormatter

formatter = JSONFormatter()
formatter.write("results.json", results)
```

### CSV Export

```python
from sentimatrix.output import CSVFormatter

formatter = CSVFormatter()
formatter.write("results.csv", results)
```

### HTML Report

```python
from sentimatrix.output import ReportGenerator

generator = ReportGenerator()
report = generator.generate(results, format="html")
generator.write("report.html", report)
```

## Using Different LLM Providers

### OpenAI

```python
from sentimatrix.providers.llm import OpenAIProvider
from sentimatrix.analysis.sentiment import SentimentAnalyzer

provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4"
)

analyzer = SentimentAnalyzer(provider=provider)
```

### Anthropic Claude

```python
from sentimatrix.providers.llm import AnthropicProvider
from sentimatrix.analysis.sentiment import SentimentAnalyzer

provider = AnthropicProvider(
    api_key="...",
    model="claude-3-sonnet-20240229"
)

analyzer = SentimentAnalyzer(provider=provider)
```

### Local Models with Ollama

```python
from sentimatrix.providers.llm import OllamaProvider
from sentimatrix.analysis.sentiment import SentimentAnalyzer

provider = OllamaProvider(
    model="llama2",
    base_url="http://localhost:11434"
)

analyzer = SentimentAnalyzer(provider=provider)
```

## Caching Results

Enable caching to avoid redundant API calls:

```python
from sentimatrix.utils.cache import CacheManager
from sentimatrix.analysis.sentiment import SentimentAnalyzer

cache = CacheManager(backend="memory", ttl=3600)
analyzer = SentimentAnalyzer(cache=cache)

# First call - makes API request
result1 = await analyzer.analyze("Great product!")

# Second call - returns cached result
result2 = await analyzer.analyze("Great product!")
```

### Redis Cache

```python
from sentimatrix.utils.cache import RedisCache

cache = RedisCache(
    host="localhost",
    port=6379,
    ttl=3600
)

analyzer = SentimentAnalyzer(cache=cache)
```

## Environment Variables

Configure Sentimatrix via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export SENTIMATRIX_CACHE_ENABLED=true
export SENTIMATRIX_CACHE_TTL=3600
export SENTIMATRIX_LOG_LEVEL=INFO
```

## Next Steps

- Read the [API Reference](../api/README.md) for detailed documentation
- See [Examples](./examples.md) for more use cases
- Check [Troubleshooting](./troubleshooting.md) if you encounter issues
