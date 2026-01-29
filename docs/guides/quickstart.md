# Sentimatrix V2 Quickstart Guide

Get started with Sentimatrix V2 in minutes.

## Installation

```bash
# Basic installation
pip install sentimatrix

# Full installation with all features
pip install sentimatrix[all]

# Or install from source
git clone https://github.com/your-org/sentimatrix.git
cd sentimatrix
pip install -e ".[dev]"
```

### Optional: Browser Dependencies

For scraping JavaScript-heavy sites (Amazon, etc.), install Playwright browsers:

```bash
# Install Chromium browser
playwright install chromium

# Install system dependencies (Linux only)
sudo playwright install-deps
```

## Basic Usage

### Using the Main Sentimatrix Class (Recommended)

The `Sentimatrix` class provides a unified interface for all functionality:

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Sentiment analysis
        result = await sm.analyze_sentiment("This product is amazing!")
        print(f"Sentiment: {result.sentiment}")  # "positive"
        print(f"Confidence: {result.confidence:.2%}")  # 95.00%

        # Emotion detection
        emotions = await sm.detect_emotions("I'm so excited about this!")
        print(f"Primary emotion: {emotions.primary_emotion.label}")  # "joy"

        # Combined analysis
        analysis = await sm.analyze("I love this product!")
        print(f"Sentiment: {analysis.sentiment.sentiment}")
        print(f"Emotion: {analysis.emotions.primary_emotion.label}")

asyncio.run(main())
```

### Direct Analyzer Usage

For more control, use the analyzers directly:

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer
from sentimatrix.analysis.emotion import EmotionDetector

async def main():
    # Initialize analyzers
    sentiment_analyzer = SentimentAnalyzer()
    emotion_detector = EmotionDetector()

    await sentiment_analyzer.initialize()
    await emotion_detector.initialize()

    try:
        # Sentiment analysis
        result = await sentiment_analyzer.analyze("This product is amazing!")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Polarity: {result.polarity:.2f}")  # -1 to 1 scale

        # Emotion detection
        emotions = await emotion_detector.detect("I'm so happy about this purchase!")
        print(f"Primary: {emotions.primary_emotion.label}")
        for emotion in emotions.emotions[:3]:
            print(f"  {emotion.label}: {emotion.score:.2%}")

    finally:
        await sentiment_analyzer.close()
        await emotion_detector.close()

asyncio.run(main())
```

### Batch Analysis

```python
import asyncio
from sentimatrix.analysis.sentiment import SentimentAnalyzer

async def main():
    analyzer = SentimentAnalyzer()
    await analyzer.initialize()

    texts = [
        "Great product! Highly recommend.",
        "Terrible experience, never buying again.",
        "It's okay, nothing special.",
        "Absolutely love it!",
        "Complete waste of money.",
    ]

    try:
        results = await analyzer.analyze_batch(texts)

        print(f"Total analyzed: {len(results.results)}")
        print(f"Positive: {results.positive_count} ({results.positive_ratio:.1%})")
        print(f"Negative: {results.negative_count} ({results.negative_ratio:.1%})")
        print(f"Neutral: {results.neutral_count} ({results.neutral_ratio:.1%})")
        print(f"Average polarity: {results.average_polarity:.2f}")

        for text, result in zip(texts, results.results):
            print(f"  {text[:30]}... -> {result.sentiment}")

    finally:
        await analyzer.close()

asyncio.run(main())
```

## Web Scraping Reviews

### Steam Reviews (No Browser Required)

Steam uses a JSON API, so it works without Playwright:

```python
import asyncio
from sentimatrix.providers.scrapers.platforms import SteamScraper, SteamConfig

async def main():
    config = SteamConfig(
        language="english",
        review_type="all",  # "positive", "negative", or "all"
    )

    async with SteamScraper(config) as scraper:
        # Scrape by app ID (730 = Counter-Strike 2)
        reviews = await scraper.scrape_reviews("730", limit=20)

        print(f"Scraped {len(reviews)} reviews")
        for review in reviews[:3]:
            rating = "Positive" if review.rating > 0 else "Negative"
            print(f"[{rating}] {review.text[:80]}...")

asyncio.run(main())
```

### Amazon Reviews (Requires Playwright)

```python
import asyncio
from sentimatrix.providers.scrapers.platforms import AmazonScraper, AmazonConfig

async def main():
    config = AmazonConfig(
        country="us",  # us, uk, de, in, jp, etc.
        filter_verified=False,
    )

    async with AmazonScraper(config) as scraper:
        # Scrape by ASIN
        reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=20)

        print(f"Scraped {len(reviews)} reviews")
        for review in reviews[:3]:
            print(f"[{review.rating}/5] {review.text[:80]}...")

asyncio.run(main())
```

### Commercial API Scrapers

For sites with strong anti-bot protection:

```python
import asyncio
from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

async def main():
    async with ScraperAPIClient(api_key="your_scraperapi_key") as client:
        # Scrape with JavaScript rendering
        content = await client.scrape(
            "https://www.amazon.com/dp/B08N5WRWNW",
            render_js=True,
            country_code="us",
        )

        print(f"Status: {content.status_code}")
        print(f"Content length: {len(content.content)}")

asyncio.run(main())
```

## Full Pipeline: Scrape + Analyze + Insights

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    # Configure with Groq for LLM insights
    llm_config = LLMConfig(
        provider="groq",
        api_key="gsk_your_api_key",  # Get from console.groq.com
        model="llama-3.3-70b-versatile",
    )

    async with Sentimatrix(llm_config=llm_config) as sm:
        # 1. Scrape reviews
        print("Scraping Steam reviews...")
        reviews = await sm.scrape_steam("730", limit=30)
        print(f"  Got {len(reviews)} reviews")

        # 2. Analyze sentiment and emotions
        print("\nAnalyzing reviews...")
        analysis = await sm.analyze_reviews(reviews)
        print(f"  Positive: {analysis.positive_ratio:.1%}")
        print(f"  Negative: {analysis.negative_ratio:.1%}")
        print(f"  Average polarity: {analysis.average_polarity:.2f}")

        # 3. Generate LLM-powered insights
        print("\nGenerating insights...")
        insights = await sm.generate_insights(reviews, analysis=analysis)

        print(f"\nSummary: {insights.summary}")

        print("\nPros:")
        for pro in insights.pros[:3]:
            print(f"  + {pro}")

        print("\nCons:")
        for con in insights.cons[:3]:
            print(f"  - {con}")

        print("\nThemes:")
        for theme in insights.themes[:3]:
            print(f"  * {theme}")

asyncio.run(main())
```

## Using LLM Providers

### Groq (Fast, Free Tier)

```python
import asyncio
from sentimatrix.providers.llm import GroqProvider
from sentimatrix.core.config import LLMConfig

async def main():
    config = LLMConfig(
        provider="groq",
        api_key="gsk_...",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    async with GroqProvider(config) as provider:
        response = await provider.generate(
            prompt="Summarize the key points from these reviews...",
            system_prompt="You are a helpful review analyst.",
        )
        print(response.content)
        print(f"Tokens used: {response.usage.total_tokens}")

asyncio.run(main())
```

### OpenAI

```python
from sentimatrix.providers.llm import OpenAIProvider
from sentimatrix.core.config import LLMConfig

config = LLMConfig(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
)

async with OpenAIProvider(config) as provider:
    response = await provider.generate("Analyze this text...")
```

### Ollama (Local)

```python
from sentimatrix.providers.llm import OllamaProvider
from sentimatrix.core.config import LLMConfig

config = LLMConfig(
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:11434",
)

async with OllamaProvider(config) as provider:
    response = await provider.generate("Analyze this text...")
```

## Exporting Results

### JSON Export

```python
from sentimatrix.output.exporters import export_to_json

# Export analysis results
await export_to_json(analysis, "results.json", pretty_print=True)
```

### CSV Export

```python
from sentimatrix.output.exporters import export_to_csv

# Export review data
await export_to_csv(reviews, "reviews.csv")
```

### HTML Report

```python
async with Sentimatrix() as sm:
    analysis = await sm.analyze_reviews(reviews)

    # Generate HTML report
    html = await sm.generate_html_report(
        analysis,
        "report.html",
        title="Product Review Analysis"
    )
```

### Charts

```python
async with Sentimatrix() as sm:
    analysis = await sm.analyze_reviews(reviews)

    # Create sentiment chart
    await sm.create_sentiment_chart(analysis, "sentiment.png", chart_type="pie")

    # Create emotion chart
    await sm.create_emotion_chart(analysis, "emotions.png")
```

## Caching Results

### Memory Cache

```python
from sentimatrix.core.cache import CacheManager

cache = CacheManager(backend="memory", ttl=3600)

# Use with analyzer
analyzer = SentimentAnalyzer(cache=cache)
```

### Redis Cache

```python
from sentimatrix.core.cache import RedisCache

cache = RedisCache(
    host="localhost",
    port=6379,
    ttl=3600,
)

analyzer = SentimentAnalyzer(cache=cache)
```

## Environment Variables

Configure Sentimatrix via environment variables:

```bash
# API Keys
export GROQ_API_KEY="gsk_..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export SCRAPERAPI_KEY="..."

# Logging
export SENTIMATRIX_LOG_LEVEL=INFO
export SENTIMATRIX_LOG_FORMAT=json

# Cache
export SENTIMATRIX_CACHE_ENABLED=true
export SENTIMATRIX_CACHE_TTL=3600
```

## CLI Usage

```bash
# Analyze text
sentimatrix analyze "This product is amazing!"

# Analyze file
sentimatrix analyze-file reviews.txt --output results.json

# Scrape platform
sentimatrix scrape steam 730 --limit 50 --analyze

# Batch process CSV
sentimatrix batch input.csv --text-column review --output results.csv

# Show system info
sentimatrix info
```

## Next Steps

- Read the [API Reference](../api/REFERENCE.md) for detailed documentation
- See [Examples](./examples.md) for more use cases
- Check [Troubleshooting](./troubleshooting.md) if you encounter issues
- Review [Provider Guide](../providers/OVERVIEW.md) for all LLM providers
- Review [Scraper Guide](../scrapers/OVERVIEW.md) for all scraper options
