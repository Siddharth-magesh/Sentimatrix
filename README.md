# Sentimatrix V2

Advanced sentiment analysis toolkit with multi-provider LLM support, comprehensive web scraping, and emotion detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-282%20passing-brightgreen.svg)](#testing)

## Features

### LLM Providers (19 Providers)
- **Cloud Providers**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere
- **Inference Providers**: Together AI, Fireworks, OpenRouter, Cerebras, DeepSeek
- **Local Providers**: Ollama, LM Studio, vLLM, llama.cpp, text-generation-webui, ExLlamaV2
- **Enterprise**: Azure OpenAI, AWS Bedrock

### Web Scraping
- **Core Scrapers**: HTTPX (async HTTP), Playwright (browser automation)
- **Platform Scrapers**: Amazon, Steam, YouTube, Reddit, IMDB, Yelp, Trustpilot, Google Reviews
- **Commercial APIs**: ScraperAPI, Apify, Bright Data, Oxylabs, Zyte, ScrapingBee, ScrapingAnt

### Analysis
- **Sentiment Analysis**: 3-class and 5-class classification with transformer models
- **Emotion Detection**: GoEmotions (28 emotions), Ekman mapping (6 basic emotions)
- **Multi-Modal**: Audio transcription (Whisper), image captioning (GPT-4V, Claude Vision)
- **Batch Processing**: Efficient batch analysis with aggregate statistics

### Output & Export
- **Formats**: JSON, CSV, Excel, HTML reports, Markdown
- **Visualizations**: Bar charts, pie charts, histograms, time series, comparison charts

## Installation

```bash
# Basic installation
pip install sentimatrix

# With all LLM providers
pip install sentimatrix[llm]

# With scraping support (includes Playwright)
pip install sentimatrix[scraping]

# With ML models (transformers, torch)
pip install sentimatrix[models]

# Full installation
pip install sentimatrix[all]

# Development installation
pip install -e ".[dev]"
```

### Browser Dependencies (for Amazon/JS-heavy sites)

```bash
# Install Playwright browsers
playwright install chromium

# Install system dependencies (Linux)
sudo playwright install-deps
```

## Quick Start

### Basic Sentiment Analysis

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Quick sentiment analysis
        result = await sm.analyze_sentiment("This product is amazing!")
        print(f"Sentiment: {result.sentiment}")  # "positive"
        print(f"Confidence: {result.confidence:.2%}")  # 95.00%

        # Emotion detection
        emotions = await sm.detect_emotions("I'm so excited about this!")
        print(f"Primary emotion: {emotions.primary_emotion.label}")  # "joy"

asyncio.run(main())
```

### Scraping and Analyzing Reviews

```python
import asyncio
from sentimatrix import Sentimatrix, LLMConfig

async def main():
    # Configure with Groq for fast LLM inference
    llm_config = LLMConfig(
        provider="groq",
        api_key="gsk_your_api_key",
        model="llama-3.3-70b-versatile"
    )

    async with Sentimatrix(llm_config=llm_config) as sm:
        # Scrape Steam reviews (works without browser deps)
        reviews = await sm.scrape_steam("730", limit=50)  # Counter-Strike 2
        print(f"Scraped {len(reviews)} reviews")

        # Analyze sentiment and emotions
        analysis = await sm.analyze_reviews(reviews)
        print(f"Positive: {analysis.positive_ratio:.1%}")
        print(f"Negative: {analysis.negative_ratio:.1%}")

        # Generate LLM-powered insights
        insights = await sm.generate_insights(reviews)
        print(f"Summary: {insights.summary}")
        print(f"Pros: {insights.pros}")
        print(f"Cons: {insights.cons}")

asyncio.run(main())
```

### Using Different Scrapers

```python
import asyncio

# Steam - Uses JSON API (no browser needed)
from sentimatrix.providers.scrapers.platforms import SteamScraper

async def scrape_steam():
    async with SteamScraper() as scraper:
        reviews = await scraper.scrape_reviews("730", limit=20)
        return reviews

# Amazon - Requires Playwright browser
from sentimatrix.providers.scrapers.platforms import AmazonScraper, AmazonConfig

async def scrape_amazon():
    config = AmazonConfig(country="us")
    async with AmazonScraper(config) as scraper:
        reviews = await scraper.scrape_reviews("B08N5WRWNW", limit=20)
        return reviews

# Commercial API - ScraperAPI (for anti-bot bypass)
from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

async def scrape_with_api():
    async with ScraperAPIClient(api_key="your_key") as client:
        content = await client.scrape(
            "https://example.com",
            render_js=True,
            country_code="us"
        )
        return content
```

### Using LLM Providers

```python
import asyncio
from sentimatrix.providers.llm import GroqProvider, OpenAIProvider
from sentimatrix.core.config import LLMConfig

# Groq (fast, free tier available)
async def use_groq():
    config = LLMConfig(
        provider="groq",
        api_key="gsk_...",
        model="llama-3.3-70b-versatile"
    )
    async with GroqProvider(config) as provider:
        response = await provider.generate("Analyze this text...")
        print(response.content)

# OpenAI
async def use_openai():
    config = LLMConfig(
        provider="openai",
        api_key="sk-...",
        model="gpt-4o-mini"
    )
    async with OpenAIProvider(config) as provider:
        response = await provider.generate("Summarize these reviews...")
        print(response.content)
```

## Configuration

### YAML Configuration

```yaml
# config.yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: ${GROQ_API_KEY}
  temperature: 0.7

scrapers:
  default_provider: playwright
  headless: true
  timeout: 30

cache:
  enabled: true
  backend: memory
  ttl: 3600

logging:
  level: INFO
  format: json
```

```python
from sentimatrix import SentimatrixConfig, Sentimatrix

config = SentimatrixConfig.from_file("config.yaml")
sm = Sentimatrix(config)
```

### Environment Variables

```bash
export GROQ_API_KEY="gsk_..."
export OPENAI_API_KEY="sk-..."
export SENTIMATRIX_LOG_LEVEL=INFO
export SENTIMATRIX_CACHE_ENABLED=true
```

## Project Structure

```
sentimatrix/
├── core/
│   ├── config.py          # Configuration management
│   ├── logger.py          # Structured logging
│   ├── exceptions.py      # Exception hierarchy
│   ├── cache.py           # Memory & Redis caching
│   └── pipeline.py        # Pipeline orchestration
├── providers/
│   ├── llm/               # 19 LLM providers
│   ├── scrapers/
│   │   ├── platforms/     # Platform-specific scrapers
│   │   └── commercial/    # Commercial API clients
│   └── models/            # HuggingFace model providers
├── analysis/
│   ├── sentiment.py       # Sentiment analysis
│   ├── emotion.py         # Emotion detection
│   └── multimodal.py      # Audio/image/video analysis
├── input/                 # Input handlers (audio, image, video)
├── output/                # Exporters, formatters, visualizers
├── cli.py                 # Command-line interface
└── main.py                # Main Sentimatrix class
```

## CLI Usage

```bash
# Analyze single text
sentimatrix analyze "This product is amazing!"

# Analyze from file
sentimatrix analyze-file reviews.txt --output results.json

# Scrape and analyze
sentimatrix scrape steam 730 --limit 50 --analyze

# Batch process CSV
sentimatrix batch input.csv --text-column review --output results.csv

# System info
sentimatrix info
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sentimatrix

# Run specific test suite
pytest tests/unit/providers/llm/
pytest tests/unit/providers/scrapers/

# Run live tests (requires API keys)
python test_full_pipeline.py
```

**Test Summary**: 282+ tests passing across all modules.

## Documentation

- [Quickstart Guide](docs/guides/quickstart.md)
- [API Reference](docs/api/REFERENCE.md)
- [Architecture Overview](docs/architecture/OVERVIEW.md)
- [Provider Guide](docs/providers/OVERVIEW.md)
- [Scraper Guide](docs/scrapers/OVERVIEW.md)
- [Configuration Guide](docs/usage/CONFIGURATION.md)
- [Examples](docs/guides/examples.md)
- [Troubleshooting](docs/guides/troubleshooting.md)

## Roadmap

See [ROADMAP.md](docs/tasks/ROADMAP.md) for the development roadmap.

## Contributing

See [CONTRIBUTING.md](docs/contributing/CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Version

Current version: **0.2.0** (Stage 14 - Commercial Scrapers Complete)
