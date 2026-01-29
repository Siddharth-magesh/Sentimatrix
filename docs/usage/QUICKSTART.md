# Sentimatrix V2 - Quick Start Guide

## Installation

```bash
# Basic installation
pip install sentimatrix

# With all optional dependencies
pip install sentimatrix[all]

# Specific extras
pip install sentimatrix[scrapers]  # Scraping support
pip install sentimatrix[llm]       # All LLM providers
pip install sentimatrix[local]     # Local inference
```

---

## Basic Usage

### Using the Async Context Manager (Recommended)

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Analyze single text
        result = await sm.analyze_sentiment("This product is amazing!")
        print(f"Sentiment: {result.sentiment.value}")  # "positive"
        print(f"Confidence: {result.confidence:.2f}")  # 0.95

asyncio.run(main())
```

### Simple Sentiment Analysis

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Analyze single text
        result = await sm.analyze_sentiment("This product is amazing!")
        print(f"Sentiment: {result.sentiment}")  # positive
        print(f"Confidence: {result.confidence:.2f}")  # 0.95

        # Quick sentiment (just label and score)
        label, score = await sm.get_quick_sentiment("Great quality!")
        print(f"{label}: {score:.2f}")  # positive: 0.93

        # Analyze multiple texts
        texts = [
            "Great quality!",
            "Terrible experience",
            "It's okay"
        ]
        batch_result = await sm.analyze_sentiment_batch(texts)
        print(f"Positive: {batch_result.positive_count}")
        print(f"Negative: {batch_result.negative_count}")
        print(f"Neutral: {batch_result.neutral_count}")

        for result in batch_result.results:
            print(f"{result.text}: {result.sentiment.value} ({result.confidence:.2f})")

asyncio.run(main())
```

### Emotion Detection

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Detect emotions (multi-label)
        result = await sm.detect_emotions("I'm so frustrated with this product!")
        print(f"Primary emotion: {result.primary_emotion.label}")  # anger
        print(f"Confidence: {result.primary_emotion.score:.2f}")

        # See all detected emotions
        for emotion in result.emotions[:3]:
            print(f"  - {emotion.label}: {emotion.score:.2f}")

        # Get Ekman's 6 basic emotions
        ekman = await sm.detect_ekman_emotions("I'm so happy today!")
        print(f"Joy: {ekman['joy']:.2f}")
        print(f"Anger: {ekman['anger']:.2f}")

asyncio.run(main())
```

### Combined Analysis (Sentiment + Emotions)

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Full analysis on single text
        result = await sm.analyze("I absolutely love this amazing product!")

        print(f"Text: {result.text}")
        print(f"Sentiment: {result.sentiment.sentiment.value}")
        print(f"Primary Emotion: {result.emotions.primary_emotion.label}")

asyncio.run(main())
```

---

## Web Scraping + Analysis

### Scrape Amazon Reviews

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Scrape Amazon product reviews
        reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
        print(f"Got {len(reviews)} reviews")

        # Analyze scraped reviews
        analysis = await sm.analyze_reviews(reviews)
        print(f"Total reviews: {analysis.total_count}")
        print(f"Positive: {analysis.positive_ratio:.1%}")
        print(f"Negative: {analysis.negative_ratio:.1%}")
        print(f"Average polarity: {analysis.average_polarity:.2f}")

asyncio.run(main())
```

### Scrape Steam Game Reviews

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Scrape Steam game reviews (CS:GO = 730)
        reviews = await sm.scrape_steam("730", limit=100)

        # Analyze reviews
        analysis = await sm.analyze_reviews(reviews)
        print(f"Positive: {analysis.positive_ratio:.1%}")

asyncio.run(main())
```

### Scrape YouTube Comments

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Requires YouTube Data API key
        comments = await sm.scrape_youtube(
            "dQw4w9WgXcQ",  # Video ID
            limit=100,
            api_key="YOUR_YOUTUBE_API_KEY"
        )

        analysis = await sm.analyze_reviews(comments)
        print(f"Comment sentiment: {analysis.positive_ratio:.1%} positive")

asyncio.run(main())
```

### Scrape Reddit Comments

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Scrape Reddit post comments
        comments = await sm.scrape_reddit("abc123", limit=50)

        analysis = await sm.analyze_reviews(comments)
        print(f"Discussion sentiment: {analysis.positive_ratio:.1%} positive")

asyncio.run(main())
```

---

## LLM-Powered Insights

### Summarize Reviews

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    # Configure with LLM provider
    async with Sentimatrix(llm_config=LLMConfig(
        provider="openai",
        api_key="sk-..."
    )) as sm:
        # Scrape reviews
        reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)

        # Generate summary
        summary = await sm.summarize_reviews(reviews, style="concise")
        print(f"Summary: {summary}")

        # Different summary styles
        detailed = await sm.summarize_reviews(reviews, style="detailed")
        bullet_points = await sm.summarize_reviews(reviews, style="bullet_points")

asyncio.run(main())
```

### Generate Insights

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    async with Sentimatrix(llm_config=LLMConfig(
        provider="openai",
        api_key="sk-..."
    )) as sm:
        reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)

        # Generate comprehensive insights
        insights = await sm.generate_insights(reviews)

        print(f"Summary: {insights.summary}")
        print(f"\nPros:")
        for pro in insights.pros:
            print(f"  + {pro}")
        print(f"\nCons:")
        for con in insights.cons:
            print(f"  - {con}")
        print(f"\nRecommendations:")
        for rec in insights.recommendations:
            print(f"  * {rec}")
        print(f"\nCommon Themes:")
        for theme in insights.themes:
            print(f"  # {theme}")

asyncio.run(main())
```

### Compare Products

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    async with Sentimatrix(llm_config=LLMConfig(
        provider="openai",
        api_key="sk-..."
    )) as sm:
        # Scrape reviews for both products
        reviews_a = await sm.scrape_amazon("ASIN_PRODUCT_A", limit=30)
        reviews_b = await sm.scrape_amazon("ASIN_PRODUCT_B", limit=30)

        # Compare products
        comparison = await sm.compare_products(
            reviews_a, reviews_b,
            item_a_name="iPhone 15",
            item_b_name="Samsung S24"
        )

        print(f"Winner: {comparison.winner}")
        print(f"\niPhone 15:")
        print(f"  Positive: {comparison.item_a_analysis.positive_ratio:.1%}")
        print(f"\nSamsung S24:")
        print(f"  Positive: {comparison.item_b_analysis.positive_ratio:.1%}")
        print(f"\nComparison: {comparison.comparison_summary}")

asyncio.run(main())
```

---

## Full Analysis Pipeline

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    async with Sentimatrix(llm_config=LLMConfig(
        provider="openai",
        api_key="sk-..."
    )) as sm:
        # Run complete analysis pipeline
        # Automatically detects platform from URL
        result = await sm.run_analysis_pipeline(
            "https://amazon.com/dp/B08N5WRWNW",
            limit=50,
            include_insights=True
        )

        print(f"Platform: {result['platform']}")
        print(f"Reviews analyzed: {result['reviews_count']}")
        print(f"Duration: {result['pipeline_duration_ms']:.2f}ms")

        if result['insights']:
            print(f"\nInsights:")
            print(f"Summary: {result['insights']['summary']}")

asyncio.run(main())
```

---

## Configuration

### Using Config File

```python
from sentimatrix import Sentimatrix

# Load from YAML
sm = Sentimatrix(config_path="config.yaml")
```

**config.yaml:**
```yaml
llm:
  provider: groq
  api_key: ${GROQ_API_KEY}
  model: llama-3.3-70b-versatile

scraper:
  timeout: 30
  max_retries: 3

models:
  sentiment_model: cardiffnlp/twitter-roberta-base-sentiment-latest
  emotion_model: SamLowe/roberta-base-go_emotions

cache:
  enabled: true
  max_size: 1000
  ttl: 3600
```

### Using Environment Variables

```bash
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
export SENTIMATRIX_LLM_PROVIDER=openai
```

```python
sm = Sentimatrix()  # Reads from environment
```

### Programmatic Configuration

```python
from sentimatrix import Sentimatrix, SentimatrixConfig, LLMConfig, ScraperConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="groq", api_key="..."),
    scraper=ScraperConfig(timeout=30, max_retries=3),
)
sm = Sentimatrix(config=config)

# Or use dict
sm = Sentimatrix(config={
    "llm": {"provider": "groq", "api_key": "..."},
    "scraper": {"timeout": 30}
})

# Or use individual config overrides
sm = Sentimatrix(
    llm_config=LLMConfig(provider="openai", api_key="..."),
    scraper_config=ScraperConfig(timeout=60)
)
```

---

## Error Handling

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.core.exceptions import (
    SentimatrixError,
    ScraperError,
    ConfigurationError,
    RateLimitError,
    ValidationError,
)

async def main():
    async with Sentimatrix() as sm:
        try:
            result = await sm.scrape_amazon("INVALID_ASIN")
        except ValidationError as e:
            print(f"Invalid input: {e}")
        except ScraperError as e:
            print(f"Failed to scrape: {e}")
        except RateLimitError as e:
            print(f"Rate limited: {e}")
        except ConfigurationError as e:
            print(f"Configuration error: {e}")
        except SentimatrixError as e:
            print(f"General error: {e}")

asyncio.run(main())
```

---

## Manual Initialization (Alternative)

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    sm = Sentimatrix()

    try:
        await sm.initialize()

        result = await sm.analyze_sentiment("Great product!")
        print(result.sentiment)
    finally:
        await sm.close()

asyncio.run(main())
```

---

## Command Line Interface (CLI)

Sentimatrix includes a powerful CLI for quick analysis without writing code.

### Installation

After installing Sentimatrix, the `sentimatrix` command is available:

```bash
pip install sentimatrix
sentimatrix --help
```

### Quick Sentiment Analysis

```bash
# Analyze a single text
sentimatrix analyze "I love this product!"

# Analyze with emotion detection
sentimatrix analyze "I'm so frustrated!" --emotions

# Output as JSON
sentimatrix analyze "Great experience!" --json

# Save to file
sentimatrix analyze "Amazing!" -o result.json
```

### Batch Analysis from File

```bash
# Analyze texts from a file (one per line)
sentimatrix analyze-file reviews.txt

# Analyze from CSV/JSON
sentimatrix analyze-file data.csv -o results.json

# Include emotions
sentimatrix analyze-file reviews.txt --emotions -o results.csv
```

### Web Scraping

```bash
# Scrape Amazon product reviews
sentimatrix scrape amazon B08N5WRWNW --limit 100

# Scrape and analyze Steam game reviews
sentimatrix scrape steam 730 --limit 50 --analyze

# Scrape YouTube comments
sentimatrix scrape youtube dQw4w9WgXcQ --limit 100 -o comments.json

# Scrape Reddit post comments
sentimatrix scrape reddit abc123 --limit 50 --analyze
```

### Batch CSV Processing

```bash
# Process CSV with text column and add sentiment
sentimatrix batch input.csv -o output.csv

# Include emotion detection
sentimatrix batch reviews.csv -o analyzed.csv --emotions
```

### System Information

```bash
# Show version and system info
sentimatrix info

# Output as JSON
sentimatrix info --json
```

### CLI Examples

```bash
# Full workflow: scrape Amazon and analyze
sentimatrix scrape amazon B08N5WRWNW --limit 100 --analyze -o amazon_analysis.json

# Analyze customer feedback file
sentimatrix analyze-file customer_feedback.txt --emotions -o sentiment_report.json

# Quick sentiment check
sentimatrix analyze "This is the best purchase I've ever made!" --json
```

---

## Next Steps

- [Configuration Reference](./CONFIGURATION.md)
- [API Reference](../api/REFERENCE.md)
- [Architecture Overview](../architecture/OVERVIEW.md)
- [Platform Scrapers](../scrapers/PLATFORM_SCRAPERS.md)
