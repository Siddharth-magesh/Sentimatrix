---
title: Changelog
description: Version history and release notes
---

# Changelog

All notable changes to Sentimatrix are documented here.

## [0.2.0] - 2025-01-29

### Added

#### LLM Providers (19 total)
- **Cloud Providers**: OpenAI, Anthropic, Google Gemini, Mistral, Cohere, Groq
- **Inference Providers**: Together AI, Fireworks, OpenRouter, Cerebras, DeepSeek
- **Local Providers**: Ollama, LM Studio, vLLM, llama.cpp, ExLlamaV2
- **Enterprise**: Azure OpenAI, AWS Bedrock

#### Platform Scrapers (8 platforms)
- Amazon product reviews
- Steam game reviews
- YouTube comments
- Reddit posts and comments
- IMDB movie reviews
- Yelp business reviews
- Trustpilot company reviews
- Google Reviews

#### Commercial Scraping APIs (7 services)
- ScraperAPI integration
- Apify integration
- Bright Data integration
- Oxylabs integration
- Zyte integration
- ScrapingBee integration
- ScrapingAnt integration

#### Core Features
- Async-first architecture with `asyncio`
- Context manager pattern (`async with`)
- Batch processing for efficiency
- Comprehensive error handling
- Rate limiting with multiple strategies
- Retry logic with exponential backoff
- Caching (memory, Redis, SQLite)
- Structured logging with `structlog`

#### Sentiment Analysis
- Quick sentiment (3-class)
- Fine-grained sentiment (5-class)
- Aspect-based sentiment analysis
- Comparative sentiment analysis
- Temporal sentiment tracking
- Domain-specific analysis

#### Emotion Detection
- Ekman's 6 basic emotions
- GoEmotions 28-class taxonomy
- Plutchik's wheel of emotions
- Multi-label detection
- Emotion intensity analysis

#### LLM Features
- Review summarization
- Insight generation (pros/cons)
- Streaming responses
- Function calling
- JSON mode output

### Changed
- Complete rewrite from V1
- Pydantic v2 for configuration
- Modern Python 3.10+ features

### Fixed
- ScraperConnectionError missing `provider` argument
- Abstract method implementation in BaseCommercialClient

## [0.1.0] - Initial Release

### Added
- Basic sentiment analysis
- OpenAI integration
- Simple web scraping

---

## Migration Guide

### From V1 to V2

#### Configuration

```python
# V1 (deprecated)
from sentimatrix import SentimentAnalyzer
analyzer = SentimentAnalyzer(api_key="...")

# V2
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(provider="openai", api_key="...")
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze("Hello")
```

#### Async Pattern

```python
# V1 (sync)
result = analyzer.analyze("Hello")

# V2 (async)
async with Sentimatrix() as sm:
    result = await sm.analyze("Hello")
```

#### Web Scraping

```python
# V1
reviews = analyzer.scrape_amazon("url")

# V2
reviews = await sm.scrape_reviews(
    url="url",
    platform="amazon",
    use_browser=True
)
```

---

## Versioning

Sentimatrix follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Releases

| Version | Date | Python | Status |
|---------|------|--------|--------|
| 0.2.0 | 2025-01-29 | 3.10+ | Current |
| 0.1.0 | 2024-01-15 | 3.9+ | Deprecated |
